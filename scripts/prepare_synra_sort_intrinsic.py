"""Generate the synra_sort INTRINSIC fixed-order benchmark.

A no-LLM order-perturbation study built on the ``synra_sort`` generator. It
exists to probe Object Aligner's **fixed-order (DP) alignment** — graded,
gap-aware partial credit under reordering and length change — a capability STED
lacks (STED matches arrays order-invariantly via Hungarian).

Every gold draws its parameters at random via ``sample_gold_params``
(``N ~ U{3..12}``, key type, closeness, distractor count) — there is no
difficulty grid. For each gold order (a permutation of ``1..N``) we apply a
controlled set of order perturbations and store the perturbed prediction. The
scorer (``scripts/score_synra_sort_intrinsic.py``) scores each pair under the
fixed (DP) and hungarian (STED stand-in) schemas, plus exact-match PMR and
Kendall τ.

Perturbation families (``meta.family`` + magnitude ``meta.k``):

  * ``none``          — identity (len N).
  * ``adj_transpose`` — k adjacent swaps (len N); ``kendall_distance`` recomputed.
  * ``block_reverse`` — reverse a contiguous block (len N).
  * ``block_move``    — move a contiguous block elsewhere (len N).
  * ``deletion``      — drop k labels (len N−k) → exercises DP gap handling.
  * ``insertion``     — inject k duplicate labels (len N+k) → DP gap handling.

Perturbed families are guaranteed to differ from the gold order (identity
outcomes are resampled), so ``family != "none"`` ⇔ actually perturbed.

Length-change families have ``kendall_distance = null`` (undefined across
lengths). Nothing is fitted in this study, so there are no train/validation
splits: all rows land in a single ``test/`` set (the name keeps the
scorer/report plumbing unchanged). Output layout mirrors
``prepare_synra_codex_intrinsic.py``.

Usage::

    uv run python scripts/prepare_synra_sort_intrinsic.py \\
        [--n-golds 300] [--direction asc] \\
        [--seed 20260609] [--force] [--root data/synra_sort/intrinsic/v2]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Project imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from object_aligner_exp.config import ExpConfig  # noqa: E402
from object_aligner_exp.datasets.synra_sort import build_example  # noqa: E402


# --- order-perturbation primitives ----------------------------------------


def kendall_distance(pred: list[int], gold: list[int]) -> int:
    """Number of discordant label pairs between two equal-length permutations
    (the bubble-sort distance / inversion count relative to gold)."""
    n = len(gold)
    gold_pos = {label: i for i, label in enumerate(gold)}
    pred_pos = {label: i for i, label in enumerate(pred)}
    disc = 0
    labels = list(gold)
    for a in range(n):
        for b in range(a + 1, n):
            la, lb = labels[a], labels[b]
            if (gold_pos[la] - gold_pos[lb]) * (pred_pos[la] - pred_pos[lb]) < 0:
                disc += 1
    return disc


def _adj_transpose(gold: list[int], k: int, rng: random.Random) -> list[int]:
    # Resample until the result differs from gold: k random swaps can compose
    # to the identity (e.g. swapping the same pair twice), and a "perturbed"
    # row identical to the gold would poison the unperturbed-vs-perturbed
    # discrimination labels. Non-identity is reachable for every (n ≥ 3, k)
    # this script generates; n = 2 only ever gets k = 1, which never composes
    # to the identity.
    while True:
        seq = list(gold)
        for _ in range(k):
            i = rng.randrange(len(seq) - 1)
            seq[i], seq[i + 1] = seq[i + 1], seq[i]
        if seq != gold:
            return seq


def _block_reverse(gold: list[int], rng: random.Random) -> list[int]:
    n = len(gold)
    a = rng.randrange(n - 1)
    b = rng.randint(a + 2, n)  # block length ≥ 2
    seq = list(gold)
    seq[a:b] = list(reversed(seq[a:b]))
    return seq


def _block_move(gold: list[int], rng: random.Random) -> list[int]:
    # Resample until the result differs from gold: reinserting the block at
    # its original offset reproduces the identity (see _adj_transpose).
    n = len(gold)
    while True:
        a = rng.randrange(n - 1)
        b = rng.randint(a + 1, n)
        block = gold[a:b]
        rest = gold[:a] + gold[b:]
        c = rng.randrange(len(rest) + 1)
        seq = rest[:c] + block + rest[c:]
        if seq != gold:
            return seq


def _deletion(gold: list[int], k: int, rng: random.Random) -> list[int]:
    seq = list(gold)
    for _ in range(k):
        seq.pop(rng.randrange(len(seq)))
    return seq


def _insertion(gold: list[int], k: int, rng: random.Random) -> list[int]:
    seq = list(gold)
    for _ in range(k):
        # A plausible-looking but spurious label (a duplicate of an existing one).
        dup = rng.choice(gold)
        seq.insert(rng.randrange(len(seq) + 1), dup)
    return seq


def perturbations(
    gold: list[int], rng: random.Random
) -> list[tuple[str, int, list[int]]]:
    """Return ``(family, k, pred_indices)`` rows for one gold order."""
    n = len(gold)
    out: list[tuple[str, int, list[int]]] = [("none", 0, list(gold))]
    for k in range(1, n):
        out.append(("adj_transpose", k, _adj_transpose(gold, k, rng)))
    if n >= 2:
        out.append(("block_reverse", 0, _block_reverse(gold, rng)))
        out.append(("block_move", 0, _block_move(gold, rng)))
    for k in range(1, min(3, n - 1) + 1):
        out.append(("deletion", k, _deletion(gold, k, rng)))
    for k in range(1, 4):
        out.append(("insertion", k, _insertion(gold, k, rng)))
    return out


# --- assembly --------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    id: str
    n: int
    key_type: str
    closeness: str
    n_distractors: int
    direction: str
    family: str
    k: int
    kendall_distance: int | None
    len_pred: int
    gold_seed: int
    gold: dict[str, Any]
    pred: dict[str, Any]
    gold_bucket: int  # gold example index


# Per-gold parameter distributions (no difficulty grid — every gold draws its
# own parameters). N range widens the former {4,5,7,9} grid; distractor count
# widens {0,2}. key_type / closeness are categorical by nature and drawn
# uniformly; direction stays a dataset-level convention (asc, as in the GEPA
# task).
N_RANGE: tuple[int, int] = (3, 12)
N_DISTRACTORS_RANGE: tuple[int, int] = (0, 3)
KEY_TYPE_CHOICES: tuple[str, ...] = ("int", "date", "ordinal")
CLOSENESS_CHOICES: tuple[str, ...] = ("wide", "tight")


def sample_gold_params(rng: random.Random) -> tuple[int, str, str, int]:
    """Draw one gold's parameters: ``(n, key_type, closeness, n_distractors)``."""
    return (
        rng.randint(*N_RANGE),
        rng.choice(KEY_TYPE_CHOICES),
        rng.choice(CLOSENESS_CHOICES),
        rng.randint(*N_DISTRACTORS_RANGE),
    )


def materialize_rows(
    n_golds: int, direction: str, master_seed: int
) -> list[Row]:
    rows: list[Row] = []
    counter = 0
    param_rng = random.Random(master_seed ^ 0x9E3779B9)
    for gold_idx in range(n_golds):
        n, key_type, closeness, nd = sample_gold_params(param_rng)
        ex = build_example(
            f"synra_sort_intrinsic_gold_{gold_idx}",
            n=n,
            key_type=key_type,
            closeness=closeness,
            n_distractors=nd,
            direction=direction,
            seed_index=gold_idx,
            base_seed=master_seed,
        )
        gold_indices = ex["gold"]["indices"]
        pert_rng = random.Random(master_seed ^ (gold_idx * 2654435761))
        for family, k, pred_indices in perturbations(gold_indices, pert_rng):
            same_len = len(pred_indices) == n
            kd = kendall_distance(pred_indices, gold_indices) if same_len else None
            rows.append(
                Row(
                    id=f"synra_sort_intrinsic_{counter:05d}",
                    n=n,
                    key_type=key_type,
                    closeness=closeness,
                    n_distractors=nd,
                    direction=direction,
                    family=family,
                    k=k,
                    kendall_distance=kd,
                    len_pred=len(pred_indices),
                    gold_seed=gold_idx,
                    gold={"indices": list(gold_indices)},
                    pred={"indices": pred_indices},
                    gold_bucket=gold_idx,
                )
            )
            counter += 1
    return rows


def _row_to_label_json(r: Row) -> dict[str, Any]:
    return {
        "id": r.id,
        "n": r.n,
        "key_type": r.key_type,
        "closeness": r.closeness,
        "n_distractors": r.n_distractors,
        "direction": r.direction,
        "family": r.family,
        "k": r.k,
        "kendall_distance": r.kendall_distance,
        "len_pred": r.len_pred,
        "gold_seed": r.gold_seed,
        "gold": r.gold,
        "pred": r.pred,
    }


def _row_to_meta_json(r: Row) -> dict[str, Any]:
    d = _row_to_label_json(r)
    d.pop("gold")
    d.pop("pred")
    return d


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary(root: Path, splits: dict[str, list[Row]], config: dict[str, Any]) -> None:
    lines = ["# synra_sort intrinsic v2 — sampling summary", ""]
    lines.append(
        f"_Generator seed_: `{config['seed']}` &nbsp;&nbsp; "
        f"_n_golds_: `{config['n_golds']}` &nbsp;&nbsp; "
        f"_direction_: `{config['direction']}`"
    )
    lines.append("")
    for name, rs in splits.items():
        lines.append(f"## {name} ({len(rs)} rows, {len({r.gold_bucket for r in rs})} golds)")
        lines.append("")
        for axis, key in [
            ("n", lambda r: r.n),
            ("family", lambda r: r.family),
            ("key_type", lambda r: r.key_type),
            ("n_distractors", lambda r: r.n_distractors),
        ]:
            ctr = Counter(key(r) for r in rs)
            cells = ", ".join(f"{k}={ctr[k]}" for k in sorted(ctr, key=str))
            lines.append(f"- **{axis}**: {cells}")
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines))


def _write_config(root: Path, config: dict[str, Any]) -> None:
    (root / "config.jsonc").write_text(
        "// synra_sort intrinsic v2 — generator configuration\n"
        "// Generated by scripts/prepare_synra_sort_intrinsic.py — do not hand-edit.\n"
        + json.dumps(config, indent=2)
        + "\n"
    )


def _parse_args() -> argparse.Namespace:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=cfg.synra_sort_intrinsic_v2)
    p.add_argument("--n-golds", type=int, default=300,
                   help="Number of gold orders; each draws its parameters via "
                        "sample_gold_params.")
    p.add_argument("--direction", choices=["asc", "desc"], default="asc")
    p.add_argument("--seed", type=int, default=20260609)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root: Path = args.root
    if root.exists() and any(root.iterdir()) and not args.force:
        print(f"refusing to overwrite non-empty {root}; pass --force.", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)

    print(f"generating rows: n_golds={args.n_golds} direction={args.direction}", flush=True)
    rows = materialize_rows(args.n_golds, args.direction, args.seed)
    print(f"  → {len(rows)} rows", flush=True)

    # Nothing is fitted in this study — a single set, written as `test/` so
    # the scorer/report plumbing stays unchanged.
    splits = {"test": rows}
    sd = root / "test"
    _write_jsonl(sd / "labels.jsonl", [_row_to_label_json(r) for r in rows])
    _write_jsonl(sd / "meta.jsonl", [_row_to_meta_json(r) for r in rows])

    config_doc = {
        "seed": args.seed,
        "n_golds": args.n_golds,
        "direction": args.direction,
        "n_range": list(N_RANGE),
        "n_distractors_range": list(N_DISTRACTORS_RANGE),
        "key_type_choices": list(KEY_TYPE_CHOICES),
        "closeness_choices": list(CLOSENESS_CHOICES),
        "families": ["none", "adj_transpose", "block_reverse", "block_move", "deletion", "insertion"],
        "n_rows": {name: len(rs) for name, rs in splits.items()},
        "generator": "scripts/prepare_synra_sort_intrinsic.py",
    }
    _write_config(root, config_doc)
    _write_summary(root, splits, config_doc)
    print(f"wrote {root}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
