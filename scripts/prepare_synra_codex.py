"""Generate the synthetic synra_codex benchmark (synra_nl "v2").

A codebook-discovery variant of ``synra_nl``: same two-scope graph of
``people`` and ``companies`` joined by three relation kinds
(``employment`` / ``acquaintance`` / ``partnership``), but the four
categorical properties — ``title``, ``industry``, ``role`` and
``relation`` — are drawn from **large vocabularies** and, crucially,
**obfuscated**. The rendered NL paragraph contains a *readable alias*
(e.g. "engineer"), while the gold value is an *arbitrary opaque code*
(e.g. "T07"). The alias→code legend lives in ``label_map.json`` and is
never shown to the extractor.

This generalizes graphimg v2 (where a node is drawn *red* but the gold
token is the arbitrary code "A"; see
``data/graphimg/v2/color_label_map.json``) into the text domain. The
seed prompt is deliberately minimal — field names only, no vocabulary —
so under GEPA's ``oa_feedback`` arm the optimizer must reconstruct each
legend from OA's deterministic feedback (``expected 'T07', got
'engineer'``), aggregated across many samples. Because each small graph
realises only a handful of the ~20 codes per property, broad coverage
requires *many* samples — exactly the "needs more and more samples"
pressure we want to study.

The codebook is a property of this generator (fixed under ``CODE_SEED``),
not of any one split: ``label_map.json`` is written to the dataset root
``data/synra_codex/`` as well as into each split dir.

Outputs (``--out-dir``):

  <out_dir>/
    train.jsonl    # GEPA D_feedback (system-prompt selection)
    val.jsonl      # GEPA D_pareto   (reward / reflection)
    test.jsonl     # holdout
    manifest.json  # full generator config + sizes + master seed + legends
    label_map.json # alias→code legends (copy; canonical at dataset root)

Each ``*.jsonl`` row::

    {
      "id": "synra_codex_00042",
      "context": "Alice is an engineer by training. Alice works at Acme ...",
      "n_people": 5, "n_companies": 3, "twin_density": 0.5, "seed": 12345,
      "gold": {
        "people":    [{"id": "<hex6>", "name": "Alice", "title": "T07"}, ...],
        "companies": [{"id": "<hex6>", "name": "Acme",  "industry": "IND14"}, ...],
        "employment":   [{"person": "<p>", "company": "<c>", "role": "R12"}, ...],
        "acquaintance": [{"source": "<p>", "target": "<p>", "relation": "AQ03"}, ...],
        "partnership":  [{"source": "<c>", "target": "<c>", "relation": "PT05"}, ...]
      }
    }

Template tuning::

    uv run python scripts/prepare_synra_codex.py --print-samples 10

prints rendered (context, gold) pairs to stdout and exits without
writing files. The gold carries *codes*; read the context for the
*aliases* and check the two stay consistent via ``label_map.json``.

Materialization::

    uv run python scripts/prepare_synra_codex.py \\
        --out-dir data/synra_codex/splits/gepa_synra_codex/pilot \\
        --seed 20260520

synra_nl as a special case
--------------------------

This generator subsumes synra_nl. ``--preset synra_nl`` turns off the three
codex-specific behaviours so the output is synra_nl-*style* (semantically
equivalent, not byte-identical to the canonical synra_nl splits)::

    uv run python scripts/prepare_synra_codex.py --preset synra_nl \\
        --out-dir data/synra_nl/splits/gepa_synra_nl_v2/pilot --seed 20260520

The preset is shorthand for three orthogonal knobs (each overridable):

  * ``--no-obfuscate``   — identity legend: gold values *are* the readable
                            words, so there is no codebook to discover.
  * ``--role-from-title``— ``employment.role == person.title`` and the separate
                            per-person title sentence is suppressed (title
                            surfaces via the employment sentence).
  * ``--vocab synra_nl`` — the small 6/6/3/3 closed vocabularies synra_nl uses
                            (subsets of the codex pools).

In ``--preset synra_nl`` mode the canonical ``data/synra_codex/label_map.json``
is left untouched — only a genuine codex run (obfuscated, codex vocab) rewrites
it. The standalone ``scripts/prepare_synra_nl.py`` remains the canonical
synra_nl generator; this preset is the "one generator, synra_nl is its
degenerate corner" path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- codebooks ------------------------------------------------------------
#
# Each property has a pool of readable ALIASES rendered into the paragraph
# and an arbitrary opaque CODE stored in the gold. Alias pools are disjoint
# across properties (titles = professions, roles = org-functions) so the
# four legends are independently learnable. `_assign_codes` shuffles the
# alias order under a fixed string seed and assigns sequential codes to the
# shuffled order, so the alias→code map is arbitrary (non-alphabetical) yet
# fully reproducible. Sequential prefixes leak nothing about *unseen*
# mappings — the optimizer must still observe each alias's code, which is
# the whole point.

CODE_SEED = 20260527

# ~24 professions. Vowel-initial entries (engineer, analyst, architect,
# accountant, economist) correctly take "an" under the first-letter rule.
TITLE_ALIASES: tuple[str, ...] = (
    "engineer", "manager", "founder", "scientist", "analyst", "designer",
    "architect", "technician", "researcher", "consultant", "director",
    "accountant", "lawyer", "surgeon", "pilot", "chef", "teacher",
    "journalist", "economist", "biologist", "machinist", "welder",
    "pharmacist", "geologist",
)

# ~20 industries.
INDUSTRY_ALIASES: tuple[str, ...] = (
    "robotics", "biotech", "finance", "media", "retail", "energy",
    "mining", "shipping", "aerospace", "agriculture", "pharma", "telecom",
    "gaming", "fashion", "automotive", "insurance", "logistics",
    "hospitality", "construction", "publishing",
)

# ~20 org-functions, disjoint from TITLE_ALIASES (and from the relation
# verbs — no "mentor"/"partner" here, which would collide with the
# acquaintance/partnership vocabularies).
ROLE_ALIASES: tuple[str, ...] = (
    "lead", "intern", "contractor", "advisor", "deputy", "associate",
    "principal", "fellow", "trainee", "coordinator", "supervisor",
    "specialist", "generalist", "liaison", "steward", "operator",
    "apprentice", "scout", "chair", "aide",
)

# ~12 person↔person relation verbs; each needs a template family below.
ACQUAINTANCE_ALIASES: tuple[str, ...] = (
    "knows", "reports_to", "mentors", "advises", "befriends", "supervises",
    "collaborates_with", "trusts", "recommends", "succeeds", "shadows",
    "assists",
)

# ~12 company↔company relation verbs.
PARTNERSHIP_ALIASES: tuple[str, ...] = (
    "partners_with", "competes_with", "acquired", "supplies", "invests_in",
    "licenses_to", "merged_with", "sponsors", "sued", "outsources_to",
    "distributes_for", "spun_off",
)


# --- synra_nl-vocab subsets ----------------------------------------------
#
# To let this generator emit synra_nl-*style* data (the special case — see
# `build_codebooks` and `--preset synra_nl`), we keep the exact closed
# vocabularies synra_nl uses. They are subsets of the codex pools above, so
# the same template families render them. `role` has no synra_nl pool of its
# own: under synra_nl-mode `role == title` (see `--role-from-title`), so the
# role book just reuses the title aliases.

SYNRA_NL_TITLE_ALIASES: tuple[str, ...] = (
    "engineer", "manager", "founder", "scientist", "analyst", "designer",
)
SYNRA_NL_INDUSTRY_ALIASES: tuple[str, ...] = (
    "robotics", "biotech", "finance", "media", "retail", "energy",
)
SYNRA_NL_ACQUAINTANCE_ALIASES: tuple[str, ...] = ("knows", "reports_to", "mentors")
SYNRA_NL_PARTNERSHIP_ALIASES: tuple[str, ...] = (
    "partners_with", "competes_with", "acquired",
)

# The `small` vocab (6/6/6/3/3) keeps synra_nl's title/industry/relation pools
# but — unlike the `synra_nl` vocab — gives `role` its *own* 6-word pool so it
# stays decoupled from `title` (no role/title conflation). Drawn from the codex
# org-functions, disjoint from TITLE_ALIASES.
SMALL_ROLE_ALIASES: tuple[str, ...] = ROLE_ALIASES[:6]  # lead..associate

# Code-stem prefix per property (used only in obfuscated mode).
_PREFIXES: dict[str, str] = {
    "title": "T",
    "industry": "IND",
    "role": "R",
    "acquaintance": "AQ",
    "partnership": "PT",
}

# Alias pools per `--vocab`: name -> {property: alias-pool}. `codex` and
# `synra_nl` are unchanged (so default / preset output is stable); `small` is
# the new 6/6/6/3/3 vocab with an independent role pool. In the `synra_nl`
# vocab `role` reuses the title pool because that vocab is paired with
# `--role-from-title` (role == title), where the role pool is unused anyway.
VOCAB_POOLS: dict[str, dict[str, tuple[str, ...]]] = {
    "codex": {
        "title": TITLE_ALIASES,
        "industry": INDUSTRY_ALIASES,
        "role": ROLE_ALIASES,
        "acquaintance": ACQUAINTANCE_ALIASES,
        "partnership": PARTNERSHIP_ALIASES,
    },
    "synra_nl": {
        "title": SYNRA_NL_TITLE_ALIASES,
        "industry": SYNRA_NL_INDUSTRY_ALIASES,
        "role": SYNRA_NL_TITLE_ALIASES,
        "acquaintance": SYNRA_NL_ACQUAINTANCE_ALIASES,
        "partnership": SYNRA_NL_PARTNERSHIP_ALIASES,
    },
    "small": {
        "title": SYNRA_NL_TITLE_ALIASES,
        "industry": SYNRA_NL_INDUSTRY_ALIASES,
        "role": SMALL_ROLE_ALIASES,
        "acquaintance": SYNRA_NL_ACQUAINTANCE_ALIASES,
        "partnership": SYNRA_NL_PARTNERSHIP_ALIASES,
    },
}

VOCABS: tuple[str, ...] = tuple(VOCAB_POOLS)


def _assign_codes(
    aliases: tuple[str, ...], prefix: str, obfuscate: bool = True, width: int = 2
) -> dict[str, str]:
    """Return an ``alias -> code`` map for one property.

    Obfuscated (``obfuscate=True``, the default / codex behaviour): an
    arbitrary-but-reproducible ``alias -> "<prefix><nn>"`` map. The alias
    order is shuffled under ``f"{CODE_SEED}:{prefix}"`` (a *string* seed, so
    the result is stable across interpreters regardless of ``PYTHONHASHSEED``)
    before sequential codes are assigned.

    Identity (``obfuscate=False``, the synra_nl special case): the map is
    ``{alias: alias}`` — gold values *are* the readable words, so the rendered
    paragraph's values equal the gold and there is no legend to discover.
    """
    if not obfuscate:
        return {alias: alias for alias in aliases}
    rng = random.Random(f"{CODE_SEED}:{prefix}")
    order = list(aliases)
    rng.shuffle(order)
    return {alias: f"{prefix}{i + 1:0{width}d}" for i, alias in enumerate(order)}


@dataclass(frozen=True)
class Codebook:
    """One property's alias↔code legend plus the sampling/render lookups."""

    code_map: dict[str, str]   # alias -> code
    by_code: dict[str, str]    # code -> alias
    codes: tuple[str, ...]     # code tuple, for sampling


@dataclass(frozen=True)
class Codebooks:
    """The five per-property codebooks for one generator invocation."""

    title: Codebook
    industry: Codebook
    role: Codebook
    acquaintance: Codebook
    partnership: Codebook

    def label_map(self) -> dict[str, dict[str, str]]:
        """alias→code legends keyed as in ``manifest.json`` / ``label_map.json``."""
        return {
            "title": self.title.code_map,
            "industry": self.industry.code_map,
            "role": self.role.code_map,
            "acquaintance_relation": self.acquaintance.code_map,
            "partnership_relation": self.partnership.code_map,
        }


def build_codebooks(vocab: str = "codex", obfuscate: bool = True) -> Codebooks:
    """Build the per-property codebooks for a ``(vocab, obfuscate)`` choice.

    ``vocab="codex"`` + ``obfuscate=True`` reproduces the canonical codex
    legend exactly (fixed under ``CODE_SEED``); ``vocab="synra_nl"`` +
    ``obfuscate=False`` (the ``--preset synra_nl`` corner) yields the small
    readable-word vocabularies that make this generator emit synra_nl-style
    data; ``vocab="small"`` is the 6/6/6/3/3 vocab with an independent role
    pool (the ``obf6`` / ``noobf6`` run variants).
    """
    if vocab not in VOCAB_POOLS:
        raise ValueError(f"unknown vocab {vocab!r}; choose from {VOCABS}")
    pools = VOCAB_POOLS[vocab]
    books: dict[str, Codebook] = {}
    for key, prefix in _PREFIXES.items():
        code_map = _assign_codes(pools[key], prefix, obfuscate=obfuscate)
        books[key] = Codebook(
            code_map=code_map,
            by_code={c: a for a, c in code_map.items()},
            codes=tuple(code_map.values()),
        )
    return Codebooks(**books)


# --- vocabularies (names) -------------------------------------------------

# ~100 locale-light first names (deduplicated — `rng.sample` over a sequence
# with repeated values can otherwise return the same name twice).
NAMES: tuple[str, ...] = tuple(dict.fromkeys((
    "Alice", "Bob", "Cara", "Dan", "Eve", "Finn", "Gina", "Hugo",
    "Ivy", "Jack", "Kira", "Liam", "Mira", "Noah", "Olive", "Paul",
    "Quinn", "Ruth", "Sam", "Tara", "Uma", "Viktor", "Wendy", "Xander",
    "Yara", "Zane", "Anya", "Brent", "Cleo", "Dirk", "Elena", "Felix",
    "Greta", "Holt", "Inga", "Jules", "Kim", "Lana", "Milo", "Nina",
    "Otis", "Pia", "Quill", "Reed", "Saskia", "Theo", "Una", "Vera",
    "Wyatt", "Xenia", "Yusuf", "Zoe", "Arno", "Briar", "Cyril", "Dana",
    "Emil", "Faye", "Gus", "Hanna", "Iris", "Jonas", "Lev",
    "Maya", "Niko", "Oren", "Petra", "Rafe", "Stella", "Tomas", "Ulla",
    "Vince", "Willa", "Yael", "Zara", "Asa", "Bea", "Cy", "Drew",
    "Esme", "Flynn", "Gail", "Hank", "Indra", "Joss", "Knox", "Lior",
    "Mara", "Niall", "Oona", "Piper", "Reza", "Sky", "Talia", "Ugo",
    "Vela", "Wren", "Yannis", "Zev",
)))

COMPANY_NAMES: tuple[str, ...] = tuple(dict.fromkeys((
    "Acme", "Globex", "Initech", "Umbrella", "Soylent", "Wayne",
    "Stark", "Tyrell", "Cyberdyne", "Hooli", "Pied Piper", "Wonka",
    "Massive Dynamic", "Oscorp", "Roxxon", "Vandelay", "Aperture",
    "Black Mesa", "Weyland", "Yutani", "Veridian", "Spacely", "Nakatomi",
    "Strickland", "Genco", "Yoyodyne", "Octan", "Krusty Krab", "Bluth",
    "Pendant", "Vexxor", "Tessier-Ashpool", "Mom Corp", "Planet Express",
    "Spade", "Nimbus", "Drey", "Linton", "Quartz", "Helix",
)))


# --- NL templates ---------------------------------------------------------

# Per-person title sentence. Because `role` is now decoupled from `title`,
# the employment sentence no longer surfaces the title — this family does.
TITLE_TEMPLATES: tuple[str, ...] = (
    "{name} is {title_article} {title} by training.",
    "{name}'s profession is {title}.",
    "By profession, {name} is {title_article} {title}.",
    "{name} trained as {title_article} {title}.",
)

INDUSTRY_TEMPLATES: tuple[str, ...] = (
    "{company} operates in the {industry} sector.",
    "{company} is {industry_article} {industry} company.",
    "In {industry}, {company} is a known name.",
)

# Employment sentence now carries the (independent) `role` alias.
EMPLOYMENT_TEMPLATES: tuple[str, ...] = (
    "{name} works at {company} as {role_article} {role}.",
    "{name} serves as {role_article} {role} at {company}.",
    "At {company}, {name} works as {role_article} {role}.",
    "{company} engaged {name} as {role_article} {role}.",
)

# One template family per acquaintance relation, keyed on the readable
# alias. Phrasings are kept lexically distinct *across* relations so the
# alias→code mapping stays unambiguous (no two different codes share a
# surface verb).
ACQUAINTANCE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "knows": (
        "{a} knows {b}.",
        "{a} is acquainted with {b}.",
        "{a} and {b} know each other.",
    ),
    "reports_to": (
        "{a} reports to {b}.",
        "{a} answers to {b}.",
        "{b} is {a}'s boss.",
    ),
    "mentors": (
        "{a} mentors {b}.",
        "{b} is mentored by {a}.",
        "{a} acts as a mentor to {b}.",
    ),
    "advises": (
        "{a} advises {b}.",
        "{b} seeks advice from {a}.",
        "{a} is an adviser to {b}.",
    ),
    "befriends": (
        "{a} befriends {b}.",
        "{a} and {b} are friends.",
        "{a} became friends with {b}.",
    ),
    "supervises": (
        "{a} supervises {b}.",
        "{b} is supervised by {a}.",
        "{a} heads {b}'s unit.",
    ),
    "collaborates_with": (
        "{a} collaborates with {b}.",
        "{a} and {b} collaborate.",
        "{a} works jointly with {b}.",
    ),
    "trusts": (
        "{a} trusts {b}.",
        "{b} is trusted by {a}.",
        "{a} places trust in {b}.",
    ),
    "recommends": (
        "{a} recommends {b}.",
        "{b} was recommended by {a}.",
        "{a} put in a good word for {b}.",
    ),
    "succeeds": (
        "{a} succeeds {b}.",
        "{a} took over from {b}.",
        "{a} is the successor to {b}.",
    ),
    "shadows": (
        "{a} shadows {b}.",
        "{a} job-shadows {b}.",
        "{b} is shadowed by {a}.",
    ),
    "assists": (
        "{a} assists {b}.",
        "{b} is assisted by {a}.",
        "{a} lends a hand to {b}.",
    ),
}

PARTNERSHIP_TEMPLATES: dict[str, tuple[str, ...]] = {
    "partners_with": (
        "{a} partners with {b}.",
        "{a} and {b} have a strategic partnership.",
        "There is a partnership between {a} and {b}.",
    ),
    "competes_with": (
        "{a} competes with {b}.",
        "{a} is a competitor of {b}.",
        "In the market, {a} and {b} are rivals.",
    ),
    "acquired": (
        "{a} acquired {b}.",
        "{b} was acquired by {a}.",
        "{a} bought {b}.",
    ),
    "supplies": (
        "{a} supplies {b}.",
        "{b} is supplied by {a}.",
        "{a} is a supplier to {b}.",
    ),
    "invests_in": (
        "{a} invests in {b}.",
        "{b} received investment from {a}.",
        "{a} is an investor in {b}.",
    ),
    "licenses_to": (
        "{a} licenses technology to {b}.",
        "{b} licenses technology from {a}.",
        "{a} granted a license to {b}.",
    ),
    "merged_with": (
        "{a} merged with {b}.",
        "{a} and {b} merged.",
        "There was a merger between {a} and {b}.",
    ),
    "sponsors": (
        "{a} sponsors {b}.",
        "{b} is sponsored by {a}.",
        "{a} is a sponsor of {b}.",
    ),
    "sued": (
        "{a} sued {b}.",
        "{b} was sued by {a}.",
        "{a} filed a lawsuit against {b}.",
    ),
    "outsources_to": (
        "{a} outsources to {b}.",
        "{b} handles outsourced work for {a}.",
        "{a} outsources operations to {b}.",
    ),
    "distributes_for": (
        "{a} distributes for {b}.",
        "{a} is a distributor for {b}.",
        "{b}'s products are distributed by {a}.",
    ),
    "spun_off": (
        "{a} spun off {b}.",
        "{b} was spun off from {a}.",
        "{a} created {b} as a spin-off.",
    ),
}

INTRO_TEMPLATES: tuple[str, ...] = (
    "Here is a short report.",
    "The following describes a few people and companies.",
    "Recent notes from our analyst:",
    "Field report:",
)

# Fail fast if a relation vocabulary and its template family drift apart.
assert set(ACQUAINTANCE_TEMPLATES) == set(ACQUAINTANCE_ALIASES), (
    "ACQUAINTANCE_TEMPLATES keys must match ACQUAINTANCE_ALIASES"
)
assert set(PARTNERSHIP_TEMPLATES) == set(PARTNERSHIP_ALIASES), (
    "PARTNERSHIP_TEMPLATES keys must match PARTNERSHIP_ALIASES"
)


def _a_or_an(noun: str) -> str:
    """Return 'a' or 'an' to match the first letter of ``noun``.

    Cheap first-letter heuristic, sufficient for the closed alias pools
    here (all entries obey the spelling rule — no "an hour" cases). Called
    on the *alias*, which is what's rendered into the paragraph.
    """
    return "an" if noun[:1].lower() in "aeiou" else "a"


# --- gold-graph sampler ---------------------------------------------------


def _fresh_hex(rng: random.Random, used: set[str]) -> str:
    while True:
        candidate = f"{rng.randrange(1 << 24):06x}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def _sample_codes(
    codes: tuple[str, ...], n: int, twin_density: float, rng: random.Random
) -> list[str]:
    """Return n codes with a ``twin_density`` fraction sharing one value."""
    n_twins = round(twin_density * n)
    twin = rng.choice(codes)
    labels = [twin] * n_twins + [rng.choice(codes) for _ in range(n - n_twins)]
    rng.shuffle(labels)
    return labels


def sample_gold(
    n_people: int,
    n_companies: int,
    twin_density: float,
    acquaintance_density: float,
    partnership_density: float,
    rng: random.Random,
    codebooks: Codebooks,
    role_from_title: bool = False,
) -> dict[str, Any]:
    """Sample a single gold graph (categorical fields stored as codes).

    Invariants:

    - Each person has exactly one ``employment`` edge (so the person's
      ``role`` always surfaces in the NL).
    - ``role`` is drawn independently from ``title`` (own codebook), *unless*
      ``role_from_title`` is set — then ``role == title`` and no independent
      draw happens (the synra_nl special case; the separate title sentence is
      also suppressed at render time).
    - Acquaintance / partnership edges are sparse, directed, self-loop-free;
      symmetric pairs are deduplicated via ``itertools.combinations``.
    """
    used_ids: set[str] = set()

    title_codes = _sample_codes(codebooks.title.codes, n_people, twin_density, rng)
    industry_codes = _sample_codes(
        codebooks.industry.codes, n_companies, twin_density, rng
    )
    person_names = rng.sample(NAMES, n_people)
    company_names = rng.sample(COMPANY_NAMES, n_companies)

    people = [
        {"id": _fresh_hex(rng, used_ids), "name": person_names[i], "title": title_codes[i]}
        for i in range(n_people)
    ]
    companies = [
        {
            "id": _fresh_hex(rng, used_ids),
            "name": company_names[i],
            "industry": industry_codes[i],
        }
        for i in range(n_companies)
    ]

    employment = [
        {
            "person": p["id"],
            "company": rng.choice(companies)["id"],
            # role == title (no extra draw) in synra_nl-mode; own codebook else.
            "role": p["title"] if role_from_title else rng.choice(codebooks.role.codes),
        }
        for p in people
    ]

    acquaintance: list[dict[str, str]] = []
    for a, b in itertools.combinations(people, 2):
        if rng.random() < acquaintance_density:
            if rng.random() < 0.5:
                a, b = b, a
            acquaintance.append(
                {
                    "source": a["id"],
                    "target": b["id"],
                    "relation": rng.choice(codebooks.acquaintance.codes),
                }
            )

    partnership: list[dict[str, str]] = []
    for a, b in itertools.combinations(companies, 2):
        if rng.random() < partnership_density:
            if rng.random() < 0.5:
                a, b = b, a
            partnership.append(
                {
                    "source": a["id"],
                    "target": b["id"],
                    "relation": rng.choice(codebooks.partnership.codes),
                }
            )

    return {
        "people": people,
        "companies": companies,
        "employment": employment,
        "acquaintance": acquaintance,
        "partnership": partnership,
    }


# --- NL renderer ----------------------------------------------------------


def render_context(
    gold: dict[str, Any],
    rng: random.Random,
    codebooks: Codebooks,
    role_from_title: bool = False,
) -> str:
    """Render a gold graph into a single-paragraph NL ``context``.

    Categorical fields in ``gold`` are *codes*; this maps each back to its
    readable alias (via the reverse legends) for rendering, so the paragraph
    never contains a code and the gold never contains an alias. (Under an
    identity legend — synra_nl-mode — code and alias coincide.) Every gold
    edge and every company's industry map to exactly one sentence; sentence
    order is randomised.

    When ``role_from_title`` is set the per-person *title sentence* is
    suppressed (title surfaces via the employment sentence, since
    ``role == title``) and the employment ``role`` is resolved through the
    *title* codebook rather than the role codebook — matching synra_nl.
    """
    by_person = {p["id"]: p for p in gold["people"]}
    by_company = {c["id"]: c for c in gold["companies"]}
    role_book = codebooks.title if role_from_title else codebooks.role

    sentences: list[str] = []

    for c in gold["companies"]:
        industry = codebooks.industry.by_code[c["industry"]]
        tmpl = rng.choice(INDUSTRY_TEMPLATES)
        sentences.append(
            tmpl.format(
                company=c["name"],
                industry=industry,
                industry_article=_a_or_an(industry),
            )
        )
    if not role_from_title:
        for p in gold["people"]:
            title = codebooks.title.by_code[p["title"]]
            tmpl = rng.choice(TITLE_TEMPLATES)
            sentences.append(
                tmpl.format(name=p["name"], title=title, title_article=_a_or_an(title))
            )
    for e in gold["employment"]:
        role = role_book.by_code[e["role"]]
        tmpl = rng.choice(EMPLOYMENT_TEMPLATES)
        sentences.append(
            tmpl.format(
                name=by_person[e["person"]]["name"],
                company=by_company[e["company"]]["name"],
                role=role,
                role_article=_a_or_an(role),
            )
        )
    for e in gold["acquaintance"]:
        relation = codebooks.acquaintance.by_code[e["relation"]]
        tmpl = rng.choice(ACQUAINTANCE_TEMPLATES[relation])
        sentences.append(
            tmpl.format(
                a=by_person[e["source"]]["name"],
                b=by_person[e["target"]]["name"],
            )
        )
    for e in gold["partnership"]:
        relation = codebooks.partnership.by_code[e["relation"]]
        tmpl = rng.choice(PARTNERSHIP_TEMPLATES[relation])
        sentences.append(
            tmpl.format(
                a=by_company[e["source"]]["name"],
                b=by_company[e["target"]]["name"],
            )
        )

    rng.shuffle(sentences)
    intro = rng.choice(INTRO_TEMPLATES)
    return intro + " " + " ".join(sentences)


# --- example construction -------------------------------------------------


# Difficulty cells: (n_people, n_companies, twin_density). Same grids as
# synra_nl so the two datasets are difficulty-comparable.
CELLS_PILOT: tuple[tuple[int, int, float], ...] = (
    (3, 2, 0.0),
    (3, 2, 1.0),
    (5, 3, 0.0),
    (5, 3, 0.5),
    (5, 3, 1.0),
    (8, 4, 0.0),
    (8, 4, 0.5),
    (8, 4, 1.0),
)

CELLS_HARD: tuple[tuple[int, int, float], ...] = (
    (10, 5, 0.5),
    (10, 5, 1.0),
    (15, 8, 0.5),
    (15, 8, 1.0),
    (20, 10, 1.0),
)

CELLS: tuple[tuple[int, int, float], ...] = CELLS_PILOT


def parse_cells(spec: str) -> tuple[tuple[int, int, float], ...]:
    """Parse ``--cells`` ("10,5,0.5;15,8,1.0") or a preset name ("pilot"/"hard")."""
    if spec == "pilot":
        return CELLS_PILOT
    if spec == "hard":
        return CELLS_HARD
    cells: list[tuple[int, int, float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"--cells: expected '<n_people>,<n_companies>,<twin_density>' "
                f"per cell, got {chunk!r}"
            )
        cells.append((int(parts[0]), int(parts[1]), float(parts[2])))
    if not cells:
        raise ValueError("--cells: no cells parsed")
    return tuple(cells)


def make_row(
    idx: int,
    n_people: int,
    n_companies: int,
    twin_density: float,
    acquaintance_density: float,
    partnership_density: float,
    seed: int,
    codebooks: Codebooks,
    role_from_title: bool = False,
    id_prefix: str = "synra_codex",
) -> dict[str, Any]:
    rng = random.Random(seed)
    gold = sample_gold(
        n_people=n_people,
        n_companies=n_companies,
        twin_density=twin_density,
        acquaintance_density=acquaintance_density,
        partnership_density=partnership_density,
        rng=rng,
        codebooks=codebooks,
        role_from_title=role_from_title,
    )
    context = render_context(
        gold, rng, codebooks=codebooks, role_from_title=role_from_title
    )
    return {
        "id": f"{id_prefix}_{idx:05d}",
        "context": context,
        "n_people": n_people,
        "n_companies": n_companies,
        "twin_density": twin_density,
        "seed": seed,
        "gold": gold,
    }


def build_rows(
    n: int,
    start_idx: int,
    master_rng: random.Random,
    acquaintance_density: float,
    partnership_density: float,
    codebooks: Codebooks,
    role_from_title: bool = False,
    id_prefix: str = "synra_codex",
    cells: tuple[tuple[int, int, float], ...] = CELLS_PILOT,
) -> list[dict[str, Any]]:
    """Round-robin over ``cells`` so every split is stratified by cell."""
    rows: list[dict[str, Any]] = []
    for k in range(n):
        np_, nc_, t_ = cells[k % len(cells)]
        seed = master_rng.randint(0, 1 << 30)
        rows.append(
            make_row(
                idx=start_idx + k,
                n_people=np_,
                n_companies=nc_,
                twin_density=t_,
                acquaintance_density=acquaintance_density,
                partnership_density=partnership_density,
                seed=seed,
                codebooks=codebooks,
                role_from_title=role_from_title,
                id_prefix=id_prefix,
            )
        )
    return rows


# --- coverage check -------------------------------------------------------


def _coverage_specs(
    codebooks: Codebooks, role_from_title: bool
) -> tuple[tuple[str, str, tuple[str, ...], str], ...]:
    """(gold-key, edge-field, full code set, human label) per active codebook.

    Under ``role_from_title`` the standalone ``role`` codebook is dropped:
    role is not sampled independently (``role == title``), so it has no
    coverage target of its own.
    """
    specs = [
        ("people", "title", codebooks.title.codes, "title"),
        ("companies", "industry", codebooks.industry.codes, "industry"),
        ("employment", "role", codebooks.role.codes, "role"),
        ("acquaintance", "relation", codebooks.acquaintance.codes, "acquaintance"),
        ("partnership", "relation", codebooks.partnership.codes, "partnership"),
    ]
    if role_from_title:
        specs = [s for s in specs if s[3] != "role"]
    return tuple(specs)


def coverage(
    rows: list[dict[str, Any]], codebooks: Codebooks, role_from_title: bool = False
) -> dict[str, tuple[set[str], set[str]]]:
    """Return ``label -> (realized_codes, full_codes)`` across ``rows``."""
    out: dict[str, tuple[set[str], set[str]]] = {}
    for gold_key, field, full, label in _coverage_specs(codebooks, role_from_title):
        realized = {
            item[field] for row in rows for item in row["gold"][gold_key]
        }
        out[label] = (realized, set(full))
    return out


def _print_coverage(
    name: str,
    rows: list[dict[str, Any]],
    codebooks: Codebooks,
    role_from_title: bool = False,
) -> dict[str, set[str]]:
    """Print a per-codebook coverage report; return missing codes per label."""
    missing: dict[str, set[str]] = {}
    print(f"[coverage] {name}:")
    for label, (realized, full) in coverage(rows, codebooks, role_from_title).items():
        miss = full - realized
        flag = "" if not miss else f"  MISSING {sorted(miss)}"
        print(f"    {label:>13}: {len(realized & full)}/{len(full)}{flag}")
        if miss:
            missing[label] = miss
    return missing


# --- CLI ------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _print_sample(row: dict[str, Any]) -> None:
    twin = row["twin_density"]
    print(
        f"\n--- {row['id']} "
        f"(n_people={row['n_people']}, n_companies={row['n_companies']}, "
        f"twin_density={twin}, seed={row['seed']}) ---"
    )
    print(f"CONTEXT:\n  {row['context']}")
    g = row["gold"]
    print(
        f"GOLD (counts): people={len(g['people'])} companies={len(g['companies'])} "
        f"employment={len(g['employment'])} acquaintance={len(g['acquaintance'])} "
        f"partnership={len(g['partnership'])}"
    )
    print(f"GOLD JSON:\n{json.dumps(g, ensure_ascii=False, indent=2)}")


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "synra_codex"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write {train,val,test}.jsonl + manifest.json + "
                        "label_map.json. Default: "
                        "data/synra_codex/splits/gepa_synra_codex/pilot")
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-val", type=int, default=60)
    p.add_argument("--n-test", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260520,
                   help="Master RNG seed (default mirrors synra_nl).")
    p.add_argument("--acquaintance-density", type=float, default=0.25,
                   help="Probability an unordered person-pair carries an edge.")
    p.add_argument("--partnership-density", type=float, default=0.35,
                   help="Probability an unordered company-pair carries an edge.")
    # --- generalization knobs: make synra_nl a special case ---------------
    # `--preset` sets {vocab, obfuscate, role_from_title, id_prefix} together;
    # the individual flags below override the preset when given explicitly.
    p.add_argument("--preset", choices=("codex", "synra_nl"), default="codex",
                   help="Umbrella mode. 'codex' (default) is the obfuscated "
                        "codebook-discovery dataset; 'synra_nl' emits "
                        "synra_nl-style data (readable values, role==title, no "
                        "title sentence, 6/6/3/3 vocab) — i.e. synra_nl as a "
                        "special case of this generator.")
    p.add_argument("--vocab", choices=VOCABS, default=None,
                   help="Alias pools to draw from. Default follows --preset "
                        "('codex' large pools / 'synra_nl' small pools).")
    p.add_argument("--obfuscate", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Whether gold values are opaque codes (--obfuscate) or "
                        "the readable words themselves (--no-obfuscate, the "
                        "synra_nl identity legend). Default follows --preset.")
    p.add_argument("--role-from-title", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="If set, employment.role == person.title and the "
                        "separate title sentence is suppressed (synra_nl). "
                        "Default follows --preset.")
    p.add_argument("--id-prefix", type=str, default=None,
                   help="Row-id prefix and manifest 'dataset' name. Default "
                        "follows --preset ('synra_codex' / 'synra_nl').")
    p.add_argument(
        "--cells",
        type=str,
        default="pilot",
        help=(
            "Difficulty grid as '<n_p>,<n_c>,<t>;...', or a preset name "
            "{'pilot', 'hard'}. 'pilot' (default) mirrors synra_nl's small "
            "graphs; 'hard' uses larger graphs biased to high twin density."
        ),
    )
    p.add_argument("--print-samples", type=int, default=0,
                   help="If >0, render this many sample rows to stdout and exit "
                        "without writing files. Use for template tuning.")
    p.add_argument("--allow-partial-coverage", action="store_true",
                   help="Do not abort if the train split fails to realise every "
                        "code in every codebook (legends would be partly "
                        "undiscoverable from train feedback).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing out-dir contents.")
    args = p.parse_args()

    # Resolve preset defaults; explicit flags (non-None) override.
    _PRESETS = {
        "codex": {"vocab": "codex", "obfuscate": True, "role_from_title": False,
                  "id_prefix": "synra_codex"},
        "synra_nl": {"vocab": "synra_nl", "obfuscate": False,
                     "role_from_title": True, "id_prefix": "synra_nl"},
    }
    preset = _PRESETS[args.preset]
    vocab = args.vocab if args.vocab is not None else preset["vocab"]
    obfuscate = args.obfuscate if args.obfuscate is not None else preset["obfuscate"]
    role_from_title = (
        args.role_from_title if args.role_from_title is not None
        else preset["role_from_title"]
    )
    id_prefix = args.id_prefix if args.id_prefix is not None else preset["id_prefix"]

    codebooks = build_codebooks(vocab=vocab, obfuscate=obfuscate)

    cells = parse_cells(args.cells)
    master_rng = random.Random(args.seed)

    if args.print_samples > 0:
        rows = build_rows(
            n=args.print_samples,
            start_idx=0,
            master_rng=master_rng,
            acquaintance_density=args.acquaintance_density,
            partnership_density=args.partnership_density,
            codebooks=codebooks,
            role_from_title=role_from_title,
            id_prefix=id_prefix,
            cells=cells,
        )
        for r in rows:
            _print_sample(r)
        return

    out_dir = args.out_dir or (
        _dataset_root() / "splits" / "gepa_synra_codex" / "pilot"
    )
    out_dir = out_dir.resolve()
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"
    manifest_path = out_dir / "manifest.json"
    label_map_path = out_dir / "label_map.json"

    existing = [
        pth for pth in (train_path, val_path, test_path, manifest_path)
        if pth.exists()
    ]
    if existing and not args.force:
        print(
            f"[abort] {len(existing)} files already exist in {out_dir}; "
            f"pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(2)

    _row_kwargs = dict(
        acquaintance_density=args.acquaintance_density,
        partnership_density=args.partnership_density,
        codebooks=codebooks,
        role_from_title=role_from_title,
        id_prefix=id_prefix,
        cells=cells,
    )
    train = build_rows(
        n=args.n_train, start_idx=0, master_rng=master_rng, **_row_kwargs
    )
    val = build_rows(
        n=args.n_val, start_idx=args.n_train, master_rng=master_rng, **_row_kwargs
    )
    test = build_rows(
        n=args.n_test, start_idx=args.n_train + args.n_val, master_rng=master_rng,
        **_row_kwargs,
    )

    # Coverage: train must realise every code (GEPA learns the legend from
    # train feedback); val/test reported for information only.
    train_missing = _print_coverage("train", train, codebooks, role_from_title)
    _print_coverage("val", val, codebooks, role_from_title)
    _print_coverage("test", test, codebooks, role_from_title)
    if train_missing and not args.allow_partial_coverage:
        print(
            f"[abort] train split does not realise every code "
            f"({ {k: sorted(v) for k, v in train_missing.items()} }). "
            f"Raise --n-train / the densities, or pass --allow-partial-coverage.",
            file=sys.stderr,
        )
        sys.exit(3)

    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    _write_jsonl(test_path, test)

    label_map = codebooks.label_map()
    manifest = {
        "dataset": id_prefix,
        "version": "v1",
        "seed": args.seed,
        "code_seed": CODE_SEED,
        "preset": args.preset,
        "vocab": vocab,
        "obfuscate": obfuscate,
        "role_from_title": role_from_title,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "cells": [list(c) for c in cells],
        "cells_spec": args.cells,
        "label_map": label_map,
        "codebook_sizes": {label: len(m) for label, m in label_map.items()},
        "acquaintance_density": args.acquaintance_density,
        "partnership_density": args.partnership_density,
        "generator": "scripts/prepare_synra_codex.py",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # label_map.json: always a copy in the split dir for self-containment.
    label_blob = json.dumps(label_map, indent=2, ensure_ascii=False)
    label_map_path.write_text(label_blob)
    # The canonical root copy (data/synra_codex/label_map.json) is the *fixed*
    # codex legend under CODE_SEED — only overwrite it from a genuine codex
    # invocation, never from a synra_nl-style / identity-legend run.
    is_canonical_codex = vocab == "codex" and obfuscate
    if is_canonical_codex:
        root_label_map = _dataset_root() / "label_map.json"
        root_label_map.parent.mkdir(parents=True, exist_ok=True)
        root_label_map.write_text(label_blob)
        root_note = f"label_map.json -> {root_label_map}"
    else:
        root_note = "root label_map.json left untouched (non-canonical legend)"

    print(
        f"[done] wrote {len(train)} train + {len(val)} val + {len(test)} test "
        f"to {out_dir}  ({root_note})"
    )


if __name__ == "__main__":
    main()
