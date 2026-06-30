"""synra_sort — synthetic *sort-by-stated-key* ordering dataset.

Each example lists N items, each with a ``name`` and exactly one explicitly
stated **sortable key** (an integer, a date, or an ordinal word), rendered in
shuffled natural-language sentences labelled ``Item 1..N``. The task is to
output the items in the order obtained by sorting on that key:

    {"indices": [3, 1, 4, 2]}

The wire shape is identical to ``sentence_ordering`` (a permutation of
``{1..N}`` under the neutral key ``"indices"``), so the same fixed/hungarian
schemas and ``sentence_ordering_eval`` (PMR / Kendall τ) apply unchanged. The
output is always a permutation, so OA's order-blind Hungarian alignment scores
≈1.0 for any permutation while the fixed-order DP alignment reflects ordering
quality — all of the fixed-vs-Hungarian signal lives in the order. This is the
fixed-order analogue of sentence ordering, but fully synthetic and
perturbation-controllable (see ``scripts/prepare_synra_sort_intrinsic.py``).

**Uniqueness of the gold order is guaranteed by construction:** the N sortable
keys are drawn *distinct* (sampled without replacement), so the argsort is
unique — there are never ties. The ``closeness`` knob shrinks the gaps between
the (still distinct) values, never makes them equal.

**Sort direction is a dataset-level convention, not a per-example coin flip.**
All examples in one build share ``direction`` (default ``"asc"``) so that a
prompt optimizer can *discover* the convention from OA feedback across the
training set (mixing asc/desc per example would be unlearnable). It is never
revealed in the seed prompt.

``meta.difficulty`` is N (the stratification key used by the carver and the
PMR/τ scorer).
"""

from __future__ import annotations

import datetime
import random
from collections.abc import Iterator
from typing import Any, TypedDict

# Per-example seed = base_seed + _SEED_MULT * index (deterministic, decorrelated).
_SEED_MULT = 1_000_003

N_VALUES: tuple[int, ...] = (4, 5, 7, 9)
KEY_TYPES: tuple[str, ...] = ("int", "date", "ordinal")
CLOSENESS: tuple[str, ...] = ("wide", "tight")
N_DISTRACTORS: tuple[int, ...] = (0, 2)
# key_mode controls whether the sortable key is the *only* numeric clause
# ("stated") or is buried among plausible numeric decoys ("hidden"): in hidden
# mode the model must also discover *which* numeric field is the key — a
# dataset-level convention never revealed in the prompt.
KEY_MODES: tuple[str, ...] = ("stated", "hidden")

# ~40 distinct item names (call-signs; locale-light).
NAME_POOL: tuple[str, ...] = (
    "Vega", "Lyra", "Orion", "Mira", "Rigel", "Altair", "Sirius", "Vela",
    "Antares", "Capella", "Polaris", "Castor", "Pollux", "Spica", "Deneb",
    "Arcturus", "Procyon", "Aldebaran", "Bellatrix", "Hadar", "Atlas",
    "Maia", "Electra", "Merope", "Tarazed", "Nunki", "Saiph", "Mizar",
    "Alcor", "Dubhe", "Merak", "Phecda", "Megrez", "Alioth", "Thuban",
    "Rasalhague", "Algol", "Sheliak", "Sulafat", "Nashira",
)

# Distractor clauses (carry no sortable signal).
_DISTRACTOR_TEMPLATES: tuple[str, ...] = (
    "It ships from depot {tok}.",
    "Its handler is team {tok}.",
    "It is stored in bay {tok}.",
    "Its colour code is {tok}.",
    "It is routed via hub {tok}.",
)
_DISTRACTOR_TOKENS: tuple[str, ...] = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ")

# Numeric decoy clauses for key_mode="hidden": each is a *plausible alternative
# integer key* on a different attribute than the designated key (which stays the
# "weighs {key} kilograms" clause). Their values may collide freely — only the
# designated key is drawn distinct. The model must discover that weight, not
# price/length/count, is the sort field.
_NUMERIC_DISTRACTOR_TEMPLATES: tuple[str, ...] = (
    "It is priced at {v} dollars.",
    "It measures {v} centimetres.",
    "It comes in a pack of {v} units.",
)

_ORDINAL_WORDS: tuple[str, ...] = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth",
)

_MONTHS: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


class SynraSortExample(TypedDict):
    id: str
    title: str
    context: str
    gold: dict[str, Any]
    meta: dict[str, Any]


# --- key sampling (distinct → unique gold order) ---------------------------


def _draw_int_keys(n: int, closeness: str, rng: random.Random) -> list[int]:
    if closeness == "wide":
        return rng.sample(range(10, 1000), n)
    start = rng.randint(10, 80)
    return rng.sample(range(start, start + n + 3), n)


def _draw_date_keys(n: int, closeness: str, rng: random.Random) -> list[int]:
    """Return distinct day-offsets from an epoch (the numeric sort key)."""
    if closeness == "wide":
        return rng.sample(range(0, 9000), n)  # ~24-year span
    start = rng.randint(0, 8000)
    return rng.sample(range(start, start + n + 5), n)


def _draw_ordinal_keys(n: int, closeness: str, rng: random.Random) -> list[int]:
    hi = len(_ORDINAL_WORDS)
    if closeness == "wide":
        return rng.sample(range(1, hi + 1), n)
    start = rng.randint(1, max(1, hi - n - 1))
    return rng.sample(range(start, start + n + 2), n)


def _draw_keys(n: int, key_type: str, closeness: str, rng: random.Random) -> list[int]:
    if key_type == "int":
        return _draw_int_keys(n, closeness, rng)
    if key_type == "date":
        return _draw_date_keys(n, closeness, rng)
    if key_type == "ordinal":
        return _draw_ordinal_keys(n, closeness, rng)
    raise ValueError(f"unknown key_type {key_type!r}")


_EPOCH = datetime.date(2000, 1, 1)


def _render_key_clause(name: str, key_type: str, key: int) -> str:
    if key_type == "int":
        return f"{name} weighs {key} kilograms."
    if key_type == "date":
        d = _EPOCH + datetime.timedelta(days=key)
        return f"{name} was founded on {d.day} {_MONTHS[d.month - 1]} {d.year}."
    if key_type == "ordinal":
        return f"{name} finished in {_ORDINAL_WORDS[key - 1]} place."
    raise ValueError(f"unknown key_type {key_type!r}")


def _render_distractors(n_distractors: int, rng: random.Random) -> list[str]:
    tmpls = rng.sample(_DISTRACTOR_TEMPLATES, min(n_distractors, len(_DISTRACTOR_TEMPLATES)))
    return [t.format(tok=rng.choice(_DISTRACTOR_TOKENS)) for t in tmpls]


def _pick_numeric_distractors(rng: random.Random) -> tuple[str, ...]:
    """Pick the set of 2–3 numeric decoy *fields* for one example. The same
    fields are reused across all items in the example: if the decoy set varied
    per item, the key field (weight) would be the only numeric attribute present
    in every item, trivially revealing it. Constant fields force genuine
    which-field discovery.
    """
    n_decoys = rng.randint(2, len(_NUMERIC_DISTRACTOR_TEMPLATES))
    return tuple(rng.sample(_NUMERIC_DISTRACTOR_TEMPLATES, n_decoys))


def _render_numeric_distractors(
    templates: tuple[str, ...], rng: random.Random
) -> list[str]:
    """Render one item's decoy clauses with fresh per-item integer values (may
    collide freely — only the designated key is drawn distinct)."""
    return [t.format(v=rng.randint(10, 999)) for t in templates]


# --- example assembly ------------------------------------------------------


def build_example(
    ex_id: str,
    *,
    n: int,
    key_type: str,
    closeness: str,
    n_distractors: int,
    direction: str,
    seed_index: int,
    base_seed: int,
    key_mode: str = "stated",
) -> SynraSortExample:
    """Build one sort-by-key example. Items are presented in a shuffled order
    (labels 1..N); the gold is those labels reordered by sorting on the key.

    ``key_mode``:
      - ``"stated"``: the single numeric key clause is the only numeric signal.
      - ``"hidden"``: 2–3 numeric decoy clauses are emitted alongside the key
        clause and the numeric clauses are shuffled per item, so *which* field
        is the key is an unrevealed convention the optimizer must discover.
    """
    rng = random.Random(base_seed + _SEED_MULT * seed_index)
    keys = _draw_keys(n, key_type, closeness, rng)  # distinct → unique order
    names = rng.sample(NAME_POOL, n)

    decoy_templates: tuple[str, ...] = ()
    if key_mode == "hidden":
        decoy_templates = _pick_numeric_distractors(rng)

    context_lines: list[str] = []
    for j in range(n):
        clause = _render_key_clause(names[j], key_type, keys[j])
        if key_mode == "hidden":
            decoys = _render_numeric_distractors(decoy_templates, rng)
            numeric_block = [clause, *decoys]
            rng.shuffle(numeric_block)  # key is not always first → which-field discovery
            parts = numeric_block
        else:
            parts = [clause]
        extras = _render_distractors(n_distractors, rng)
        context_lines.append(f"Item {j + 1}: " + " ".join([*parts, *extras]).strip())
    context = "Here are the items:\n\n" + "\n".join(context_lines)

    reverse = direction == "desc"
    order = sorted(range(n), key=lambda j: keys[j], reverse=reverse)
    gold_indices = [j + 1 for j in order]

    gold = {"indices": gold_indices}
    meta = {
        "n_items": n,
        "difficulty": n,
        "key_type": key_type,
        "closeness": closeness,
        "n_distractors": n_distractors,
        "key_mode": key_mode,
        "n_numeric_decoys": len(decoy_templates),
        "direction": direction,
        "names": names,
        "keys": keys,
        "gold_sort_order": gold_indices,
    }
    return SynraSortExample(id=ex_id, title=ex_id, context=context, gold=gold, meta=meta)


def default_cells() -> list[tuple[int, str, str, int, str]]:
    """Deterministic product of the difficulty knobs (excludes direction, which
    is a dataset-level convention)."""
    cells: list[tuple[int, str, str, int, str]] = []
    for n in N_VALUES:
        for key_type in KEY_TYPES:
            for closeness in CLOSENESS:
                for nd in N_DISTRACTORS:
                    for key_mode in KEY_MODES:
                        cells.append((n, key_type, closeness, nd, key_mode))
    return cells


def iter_synra_sort_examples(
    *,
    limit: int | None = None,
    base_seed: int = 20260609,
    direction: str = "asc",
    cells: list[tuple[int, str, str, int, str]] | None = None,
) -> Iterator[SynraSortExample]:
    """Yield synthetic sort-by-key examples, round-robin over the difficulty
    cells (so any prefix is difficulty-balanced)."""
    cells = cells or default_cells()
    i = 0
    while limit is None or i < limit:
        n, key_type, closeness, nd, key_mode = cells[i % len(cells)]
        yield build_example(
            f"synra_sort_{i}",
            n=n,
            key_type=key_type,
            closeness=closeness,
            n_distractors=nd,
            direction=direction,
            seed_index=i,
            base_seed=base_seed,
            key_mode=key_mode,
        )
        i += 1


ITERATORS = {"synra_sort": iter_synra_sort_examples}
