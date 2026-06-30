"""Build the synra_sort dataset: generate synthetic sort-by-key examples, carve splits.

Mirrors ``scripts/prepare_sentence_ordering.py`` but the corpus is fully
synthetic (no HuggingFace): each example lists N items with an explicitly
stated sortable key, and the gold is the index permutation that sorts them.
Same ``{"indices":[...]}`` wire shape as sentence_ordering, so the fixed/
hungarian schemas and ``sentence_ordering_eval`` apply unchanged.

Pipeline:

  1. Generate a pool of examples (round-robin over difficulty cells).
     Write data/synra_sort/preprocessed/synra_sort.jsonl
  2. SELF-TEST: assert every gold order is a permutation of 1..N AND that the
     keys, read in gold order, are strictly monotone in the sort direction
     (guards both the permutation invariant and the unique-gold guarantee).
  3. Carve a fixed-seed, difficulty-stratified (difficulty = N) split under
       data/synra_sort/splits/gepa_synra_sort/main/
     with train/val/test disjoint plus a larger ``test_full``.

Usage::

    uv run python scripts/prepare_synra_sort.py --print-samples 3
    uv run python scripts/prepare_synra_sort.py [--direction asc] [--force]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets.rebel import load_jsonl, write_jsonl
from object_aligner_exp.datasets.synra_sort import (
    N_DISTRACTORS,
    N_VALUES,
    SynraSortExample,
    iter_synra_sort_examples,
)

DIFFICULTY_KEY = "n_items"
DEFAULT_POOL_CAP = 4000


def _build_cells(
    closeness: str, key_mode: str
) -> list[tuple[int, str, str, int, str]]:
    """Filtered difficulty grid for one build: key_type=int, a single closeness
    and key_mode, full N range, both text-distractor levels (8 cells)."""
    return [
        (n, "int", closeness, nd, key_mode)
        for n in N_VALUES
        for nd in N_DISTRACTORS
    ]


def _preprocess(
    *,
    out_path: Path,
    base_seed: int,
    direction: str,
    pool_cap: int,
    force: bool,
    cells: list[tuple[int, str, str, int, str]] | None = None,
) -> Path:
    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open("r"))
        print(f"[skip] {out_path} already has {n} examples (use --force to rebuild).")
        return out_path
    print(f"[generate] synra_sort (direction={direction}, pool={pool_cap})")
    examples = list(
        iter_synra_sort_examples(
            limit=pool_cap, base_seed=base_seed, direction=direction, cells=cells
        )
    )
    _self_test(examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_path, examples)
    print(f"[write] {out_path}  ({n} examples)")
    return out_path


def _self_test(examples: list[SynraSortExample]) -> None:
    """Abort unless every gold order is a valid permutation of 1..N whose keys,
    read in gold order, are strictly monotone in the sort direction."""
    bad: list[str] = []
    for ex in examples:
        m = ex["meta"]
        order = ex["gold"]["indices"]
        n = m["n_items"]
        keys = m["keys"]
        if sorted(order) != list(range(1, n + 1)):
            bad.append(ex["id"])
            continue
        seq = [keys[lbl - 1] for lbl in order]
        asc = all(a < b for a, b in zip(seq, seq[1:]))
        desc = all(a > b for a, b in zip(seq, seq[1:]))
        ok = asc if m["direction"] == "asc" else desc
        if not ok:
            bad.append(ex["id"])
    if bad:
        sample = ", ".join(bad[:10])
        raise RuntimeError(
            f"SELF-TEST FAILED: {len(bad)}/{len(examples)} examples have a bad "
            f"gold order (construction bug?). First few: {sample}"
        )
    print(f"[self-test] OK — all {len(examples)} gold orders valid & strictly monotone")


def _stratified_order(
    pool: list[SynraSortExample], *, seed: int
) -> list[SynraSortExample]:
    by_bucket: dict[int, list[SynraSortExample]] = defaultdict(list)
    for ex in pool:
        by_bucket[int(ex["meta"]["difficulty"])].append(ex)
    rng = random.Random(seed)
    for b in by_bucket.values():
        rng.shuffle(b)
    buckets = [by_bucket[k] for k in sorted(by_bucket)]
    order: list[SynraSortExample] = []
    i = 0
    while True:
        added = False
        for b in buckets:
            if i < len(b):
                order.append(b[i])
                added = True
        if not added:
            break
        i += 1
    return order


def _bucket_counts(rows: list[SynraSortExample]) -> dict[int, int]:
    c: dict[int, int] = defaultdict(int)
    for ex in rows:
        c[int(ex["meta"]["difficulty"])] += 1
    return dict(sorted(c.items()))


def _carve(
    cfg: ExpConfig,
    *,
    split_dir: Path,
    preprocessed: Path,
    direction: str,
    n_train: int,
    n_val: int,
    n_test: int,
    n_test_full: int,
    seed: int,
    force: bool,
) -> None:
    outputs = [
        split_dir / f"{n}.jsonl" for n in ("train", "val", "test", "test_full")
    ] + [split_dir / "manifest.json"]
    if all(p.exists() for p in outputs) and not force:
        print(f"[skip] {split_dir} already populated (use --force to rebuild).")
        return

    pool = load_jsonl(preprocessed)
    need = n_train + n_val + n_test
    if len(pool) < need:
        raise RuntimeError(f"synra_sort: pool has {len(pool)} < {need} needed")

    order = _stratified_order(pool, seed=seed)
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    rest = order[n_train + n_val :]
    test_full = rest[:n_test_full]
    test = test_full[:n_test]

    split_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(split_dir / "train.jsonl", train)
    write_jsonl(split_dir / "val.jsonl", val)
    write_jsonl(split_dir / "test.jsonl", test)
    write_jsonl(split_dir / "test_full.jsonl", test_full)

    rel = lambda p: str(p.relative_to(cfg.data_root.parent))  # noqa: E731
    manifest = {
        "dataset": "synra_sort — synthetic sort-by-stated-key",
        "source": "synthetic (see datasets/synra_sort.py)",
        "difficulty_key": DIFFICULTY_KEY,
        "direction": direction,
        "seed": seed,
        "note": (
            "train/val/test are mutually disjoint and difficulty-stratified by "
            "item count; test is a stratified subset of test_full. Output is an "
            "index permutation, so Hungarian alignment is ~1.0 by construction "
            "and the fixed-vs-Hungarian gap is pure order. Keys are distinct → "
            "unique gold sort order; direction is a fixed dataset convention "
            "(not revealed in the seed prompt — GEPA discovers it)."
        ),
        "splits": {
            name: {
                "n": len(rows),
                "bucket_counts": _bucket_counts(rows),
                "source_ids": [ex["id"] for ex in rows],
                "path": rel(split_dir / f"{name}.jsonl"),
            }
            for name, rows in (
                ("train", train),
                ("val", val),
                ("test", test),
                ("test_full", test_full),
            )
        },
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(
        f"[write] {split_dir}/  ({len(train)} train + {len(val)} val + "
        f"{len(test)} test; test_full={len(test_full)})"
    )


def _print_samples(
    n: int,
    base_seed: int,
    direction: str,
    cells: list[tuple[int, str, str, int, str]] | None = None,
) -> None:
    for ex in iter_synra_sort_examples(
        limit=n, base_seed=base_seed, direction=direction, cells=cells
    ):
        m = ex["meta"]
        print(f"--- {ex['id']}  (N={m['n_items']}, key={m['key_type']}, "
              f"closeness={m['closeness']}, key_mode={m['key_mode']}, "
              f"distractors={m['n_distractors']}) ---")
        print(ex["context"])
        print("GOLD indices:", ex["gold"]["indices"])
        print("keys (presentation order):", m["keys"])
        print("keys (gold order):", [m["keys"][lbl - 1] for lbl in ex["gold"]["indices"]])
        print()


def main() -> None:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=100, help="train size (D_feedback)")
    p.add_argument("--n-val", type=int, default=100, help="val size (D_pareto)")
    p.add_argument("--n-test", type=int, default=200, help="default held-out test size")
    p.add_argument("--n-test-full", type=int, default=1000, help="cap on test_full")
    p.add_argument("--pool-cap", type=int, default=DEFAULT_POOL_CAP,
                   help="examples to generate before carving")
    p.add_argument("--direction", choices=["asc", "desc"], default="asc",
                   help="dataset-wide sort convention (GEPA discovers it).")
    p.add_argument("--closeness", choices=["wide", "tight"], default=None,
                   help="build a single-closeness GEPA build (key_type=int). "
                        "Omit (with --key-mode) for the legacy full-grid 'main' split.")
    p.add_argument("--key-mode", choices=["stated", "hidden"], default=None,
                   help="stated = single numeric key clause; hidden = key buried "
                        "among numeric decoys (which-field discovery).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="split dir to write (default: derived build dir, or "
                        "gepa_synra_sort/main in legacy mode).")
    p.add_argument("--seed", type=int, default=20260609, help="RNG seed (gen + carve)")
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    p.add_argument("--print-samples", type=int, default=0, metavar="N",
                   help="render N generated examples and exit (no writes)")
    args = p.parse_args()

    # Build mode: one (key_mode × closeness) build per invocation. Either knob
    # alone implies the other's default so --print-samples --key-mode hidden works.
    build_mode = args.closeness is not None or args.key_mode is not None
    if build_mode:
        closeness = args.closeness or "wide"
        key_mode = args.key_mode or "stated"
        build = f"{key_mode}_{closeness}"
        cells = _build_cells(closeness, key_mode)
        preprocessed = cfg.synra_sort_preprocessed / f"synra_sort_{build}.jsonl"
        split_dir = args.out_dir or (
            cfg.synra_sort_root / "splits" / "gepa_synra_sort" / build
        )
    else:
        cells = None  # legacy full-grid pool
        preprocessed = cfg.synra_sort_preprocessed / "synra_sort.jsonl"
        split_dir = args.out_dir or cfg.gepa_synra_sort_main

    if args.print_samples:
        _print_samples(args.print_samples, args.seed, args.direction, cells=cells)
        return

    cfg.synra_sort_preprocessed.mkdir(parents=True, exist_ok=True)
    _preprocess(
        out_path=preprocessed,
        base_seed=args.seed,
        direction=args.direction,
        pool_cap=args.pool_cap,
        force=args.force,
        cells=cells,
    )
    _carve(
        cfg,
        split_dir=split_dir,
        preprocessed=preprocessed,
        direction=args.direction,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_test_full=args.n_test_full,
        seed=args.seed,
        force=args.force,
    )
    print("done.")


if __name__ == "__main__":
    main()
