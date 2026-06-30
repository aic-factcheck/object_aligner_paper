"""Build the sentence-ordering dataset: stream a corpus, shuffle, carve splits.

Pipeline per corpus (mirrors ``scripts/prepare_planbench.py``):

  1. Stream a corpus from HuggingFace, segment into sentences, and build the
     ordered-permutation task (deterministic per-example shuffle; gold = the
     scrambled labels in correct reading order). See ``datasets/sentence_ordering``.
       - ``rocstories``: mintujupally/ROCStories, regex split, keep exactly 5.
       - ``arxiv``:      arXiv abstracts, nltk split, keep N∈[6,12] (built later).
     Write data/sentence_ordering/preprocessed/<corpus>.jsonl
  2. SELF-TEST: assert every example's gold order is a permutation of 1..N AND
     that reordering the shuffled sentences by the gold order reproduces the
     original sentence sequence. Aborts loudly on any failure (guards the
     shuffle/gold construction).
  3. Carve a fixed-seed, difficulty-stratified split (difficulty = N) under
       data/sentence_ordering/splits/gepa_<corpus>/main/
     with train/val/test mutually disjoint (default 100/100/200) plus
     ``test_full`` (a large held-out set, default cap 1000; ``test`` is a
     stratified subset of it).

Idempotent: re-running with the same args is a no-op unless ``--force``.

Usage::

    uv run python scripts/prepare_sentence_ordering.py --corpus rocstories
    uv run python scripts/prepare_sentence_ordering.py --corpus rocstories --print-samples 3
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets.rebel import load_jsonl, write_jsonl
from object_aligner_exp.datasets.sentence_ordering import (
    ITERATORS,
    SentenceOrderingExample,
)

DIFFICULTY_KEY = "n_sentences"

# How many examples to pull from the stream before carving (cap so we don't
# materialise the whole corpus). Must comfortably exceed n_train+n_val+test_full.
DEFAULT_POOL_CAP = 4000


def _preprocess(
    corpus: str, *, out_path: Path, base_seed: int, pool_cap: int, force: bool
) -> Path:
    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open("r"))
        print(f"[skip] {out_path} already has {n} examples (use --force to rebuild).")
        return out_path
    print(f"[parse] sentence_ordering {corpus!r} (HuggingFace stream)")
    iter_fn = ITERATORS[corpus]
    examples = list(
        tqdm(iter_fn(limit=pool_cap, base_seed=base_seed), unit="ex", desc=corpus)
    )
    _self_test(examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_path, examples)
    print(f"[write] {out_path}  ({n} examples)")
    return out_path


def _self_test(examples: list[SentenceOrderingExample]) -> None:
    """Abort unless every gold order is a valid permutation that reconstructs the
    original sentence sequence from the shuffled one."""
    bad: list[str] = []
    for ex in examples:
        m = ex["meta"]
        order = ex["gold"]["indices"]
        n = m["n_sentences"]
        shuf = m["shuffled_sentences"]
        if sorted(order) != list(range(1, n + 1)):
            bad.append(ex["id"])
            continue
        recon = [shuf[lbl - 1] for lbl in order]
        if recon != m["orig_sentences"]:
            bad.append(ex["id"])
    if bad:
        sample = ", ".join(bad[:10])
        raise RuntimeError(
            f"SELF-TEST FAILED: {len(bad)}/{len(examples)} examples have a bad "
            f"gold order (construction bug?). First few: {sample}"
        )
    print(f"[self-test] OK — all {len(examples)} gold orders valid & reconstruct")


def _stratified_order(
    pool: list[SentenceOrderingExample], *, seed: int
) -> list[SentenceOrderingExample]:
    """Round-robin interleave examples across difficulty (N) buckets, so any
    prefix is approximately difficulty-balanced (same logic as prepare_planbench)."""
    by_bucket: dict[int, list[SentenceOrderingExample]] = defaultdict(list)
    for ex in pool:
        by_bucket[int(ex["meta"]["difficulty"])].append(ex)
    rng = random.Random(seed)
    for b in by_bucket.values():
        rng.shuffle(b)
    buckets = [by_bucket[k] for k in sorted(by_bucket)]
    order: list[SentenceOrderingExample] = []
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


def _bucket_counts(rows: list[SentenceOrderingExample]) -> dict[int, int]:
    c: dict[int, int] = defaultdict(int)
    for ex in rows:
        c[int(ex["meta"]["difficulty"])] += 1
    return dict(sorted(c.items()))


def _carve(
    cfg: ExpConfig,
    corpus: str,
    *,
    split_dir: Path,
    preprocessed: Path,
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
        raise RuntimeError(f"{corpus}: pool has {len(pool)} < {need} needed")

    order = _stratified_order(pool, seed=seed)
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    rest = order[n_train + n_val :]          # complement of train+val
    test_full = rest[:n_test_full]           # large held-out set (capped)
    test = test_full[:n_test]                # cheap default test (subset)

    split_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(split_dir / "train.jsonl", train)
    write_jsonl(split_dir / "val.jsonl", val)
    write_jsonl(split_dir / "test.jsonl", test)
    write_jsonl(split_dir / "test_full.jsonl", test_full)

    rel = lambda p: str(p.relative_to(cfg.data_root.parent))  # noqa: E731
    manifest = {
        "dataset": f"Sentence ordering — {corpus}",
        "source": "HuggingFace stream (see datasets/sentence_ordering.py)",
        "corpus": corpus,
        "difficulty_key": DIFFICULTY_KEY,
        "seed": seed,
        "note": (
            "train/val/test are mutually disjoint and difficulty-stratified by "
            "sentence count; test is a stratified subset of test_full (capped), "
            "the large held-out set kept for tight PMR/tau estimates. Output is "
            "an index permutation, so Hungarian alignment is ~1.0 by "
            "construction and the fixed-vs-Hungarian gap is pure order."
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


def _print_samples(corpus: str, n: int, base_seed: int) -> None:
    for ex in ITERATORS[corpus](limit=n, base_seed=base_seed):
        m = ex["meta"]
        print(f"--- {ex['id']}  (N={m['n_sentences']}) ---")
        print(ex["context"])
        print("GOLD indices:", ex["gold"]["indices"])
        print("reads as:", " | ".join(
            m["shuffled_sentences"][lbl - 1] for lbl in ex["gold"]["indices"]
        ))
        print()


def main() -> None:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["rocstories", "arxiv", "both"], default="rocstories")
    p.add_argument("--n-train", type=int, default=100, help="train size (D_feedback)")
    p.add_argument("--n-val", type=int, default=100, help="val size (D_pareto)")
    p.add_argument("--n-test", type=int, default=200, help="default held-out test size")
    p.add_argument("--n-test-full", type=int, default=1000, help="cap on test_full")
    p.add_argument("--pool-cap", type=int, default=DEFAULT_POOL_CAP,
                   help="examples to pull from the stream before carving")
    p.add_argument("--seed", type=int, default=20260530, help="RNG seed (shuffle + carve)")
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    p.add_argument("--print-samples", type=int, default=0, metavar="N",
                   help="render N parsed examples per corpus and exit (no writes)")
    args = p.parse_args()

    corpora = ["rocstories", "arxiv"] if args.corpus == "both" else [args.corpus]
    split_dirs = {
        "rocstories": cfg.gepa_rocstories_main,
        "arxiv": cfg.gepa_arxiv_main,
    }

    if args.print_samples:
        for corpus in corpora:
            print(f"\n===== {corpus} samples =====")
            _print_samples(corpus, args.print_samples, args.seed)
        return

    cfg.sentence_ordering_preprocessed.mkdir(parents=True, exist_ok=True)
    for corpus in corpora:
        preprocessed = cfg.sentence_ordering_preprocessed / f"{corpus}.jsonl"
        _preprocess(
            corpus,
            out_path=preprocessed,
            base_seed=args.seed,
            pool_cap=args.pool_cap,
            force=args.force,
        )
        _carve(
            cfg,
            corpus,
            split_dir=split_dirs[corpus],
            preprocessed=preprocessed,
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
