"""Download NATURAL PLAN, preprocess to OA-aligned JSONL, carve GEPA splits.

Covers two NATURAL PLAN tasks (Trip Planning, Meeting Planning). Pipeline per
task:

  1. Ensure ``data/natural_plan/raw/{trip,meeting}_planning.json`` is present
     (downloaded from github.com/google-deepmind/natural-plan if missing).
  2. Parse each record into the OA-aligned wire shape (see
     ``datasets/natural_plan.py``) and write
       data/natural_plan/preprocessed/{trip,meeting}.jsonl
  3. Carve a fixed-seed, difficulty-stratified split under
       data/natural_plan/splits/gepa_{trip,meeting}_planning/main/
     with ``train``/``val``/``test`` mutually disjoint (default 100/100/200)
     plus ``test_full`` (everything except train+val — the large held-out set
     to compare against SOTA later; ``test`` is a stratified subset of it).

NATURAL PLAN is evaluation-only with no native train split and SOTA is reported
over the full set; we keep train+val tiny so the bulk stays held out. Splits are
stratified by difficulty (``num_cities`` for trip, ``num_people`` for meeting)
so every split mirrors the benchmark's difficulty mix.

Idempotent: re-running with the same args is a no-op unless ``--force``.

Usage::

    uv run python scripts/prepare_natural_plan.py --task both
    uv run python scripts/prepare_natural_plan.py --task trip --print-samples 5
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

from tqdm import tqdm

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets.natural_plan import (
    DATASET_URL,
    MEETING_RAW_NAME,
    TRIP_RAW_NAME,
    NaturalPlanExample,
    iter_meeting_examples,
    iter_trip_examples,
)
from object_aligner_exp.datasets.rebel import load_jsonl, write_jsonl

RAW_BASE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/natural-plan/main/data"
)

TASKS: dict[str, dict] = {
    "trip": {
        "raw_name": TRIP_RAW_NAME,
        "iter": iter_trip_examples,
        "preprocessed_name": "trip.jsonl",
        "difficulty_key": "num_cities",
    },
    "meeting": {
        "raw_name": MEETING_RAW_NAME,
        "iter": iter_meeting_examples,
        "preprocessed_name": "meeting.jsonl",
        "difficulty_key": "num_people",
    },
}


def _download(url: str, dest: Path, *, force: bool) -> Path:
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists (use --force to redownload).")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} → {dest}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    return dest


def _preprocess(
    task: str,
    *,
    iter_fn: Callable[..., Iterator[NaturalPlanExample]],
    raw_dir: Path,
    out_path: Path,
    force: bool,
) -> Path:
    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open("r"))
        print(f"[skip] {out_path} already has {n} examples (use --force to rebuild).")
        return out_path
    print(f"[parse] NATURAL PLAN {task!r}")
    examples = list(tqdm(iter_fn(raw_dir), unit="ex", desc=task))
    n = write_jsonl(out_path, examples)
    print(f"[write] {out_path}  ({n} examples)")
    return out_path


def _stratified_order(
    pool: list[NaturalPlanExample], *, seed: int
) -> list[NaturalPlanExample]:
    """Round-robin interleave examples across difficulty buckets.

    Shuffling within each bucket and then interleaving buckets means any prefix
    of the result is (approximately) difficulty-balanced — so slicing the
    interleaved list sequentially yields stratified train/val/test splits.
    """
    by_bucket: dict[int, list[NaturalPlanExample]] = defaultdict(list)
    for ex in pool:
        by_bucket[int(ex["meta"]["difficulty"])].append(ex)
    rng = random.Random(seed)
    for b in by_bucket.values():
        rng.shuffle(b)
    buckets = [by_bucket[k] for k in sorted(by_bucket)]
    order: list[NaturalPlanExample] = []
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


def _bucket_counts(rows: list[NaturalPlanExample]) -> dict[int, int]:
    c: dict[int, int] = defaultdict(int)
    for ex in rows:
        c[int(ex["meta"]["difficulty"])] += 1
    return dict(sorted(c.items()))


def _carve(
    task: str,
    cfg: ExpConfig,
    *,
    split_dir: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    force: bool,
) -> None:
    outputs = [
        split_dir / f"{n}.jsonl" for n in ("train", "val", "test", "test_full")
    ] + [split_dir / "manifest.json"]
    if all(p.exists() for p in outputs) and not force:
        print(f"[skip] {split_dir} already populated (use --force to rebuild).")
        return

    preprocessed = cfg.natural_plan_preprocessed / TASKS[task]["preprocessed_name"]
    pool = load_jsonl(preprocessed)
    need = n_train + n_val + n_test
    if len(pool) < need:
        raise RuntimeError(f"{task}: pool has {len(pool)} < {need} needed")

    order = _stratified_order(pool, seed=seed)
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    rest = order[n_train + n_val :]          # everything held out (~complement of train+val)
    test = rest[:n_test]                      # cheap default test (subset of test_full)
    test_full = rest                          # large held-out set for SOTA comparison later

    split_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(split_dir / "train.jsonl", train)
    write_jsonl(split_dir / "val.jsonl", val)
    write_jsonl(split_dir / "test.jsonl", test)
    write_jsonl(split_dir / "test_full.jsonl", test_full)

    rel = lambda p: str(p.relative_to(cfg.data_root.parent))  # noqa: E731
    manifest = {
        "dataset": f"NATURAL PLAN — {task} (Zheng et al., arXiv:2406.04520)",
        "source_url": DATASET_URL,
        "task": task,
        "difficulty_key": TASKS[task]["difficulty_key"],
        "seed": seed,
        "note": (
            "train/val/test are mutually disjoint and difficulty-stratified; "
            "test is a stratified subset of test_full (= everything except "
            "train+val), the large held-out set kept for SOTA comparison."
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


def _print_samples(task: str, raw_dir: Path, n: int) -> None:
    iter_fn = TASKS[task]["iter"]
    for ex in iter_fn(raw_dir, limit=n):
        print(f"--- {ex['id']}  (difficulty={ex['meta']['difficulty']}) ---")
        print("CONTEXT:", ex["context"][:200].replace("\n", " "), "...")
        print("GOLD:", json.dumps(ex["gold"], ensure_ascii=False))
        print()


def main() -> None:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["trip", "meeting", "both"], default="both")
    p.add_argument("--n-train", type=int, default=100, help="train size (D_feedback)")
    p.add_argument("--n-val", type=int, default=100, help="val size (D_pareto)")
    p.add_argument("--n-test", type=int, default=200, help="default held-out test size")
    p.add_argument("--seed", type=int, default=20260530, help="RNG seed for the carve")
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    p.add_argument(
        "--print-samples", type=int, default=0, metavar="N",
        help="render N parsed examples per task to stdout and exit (no writes)",
    )
    args = p.parse_args()

    tasks = ["trip", "meeting"] if args.task == "both" else [args.task]
    split_dirs = {
        "trip": cfg.gepa_trip_planning_main,
        "meeting": cfg.gepa_meeting_planning_main,
    }

    if args.print_samples:
        for task in tasks:
            _download(
                f"{RAW_BASE_URL}/{TASKS[task]['raw_name']}",
                cfg.natural_plan_raw / TASKS[task]["raw_name"],
                force=False,
            )
            print(f"\n===== {task} samples =====")
            _print_samples(task, cfg.natural_plan_raw, args.print_samples)
        return

    cfg.natural_plan_preprocessed.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        _download(
            f"{RAW_BASE_URL}/{TASKS[task]['raw_name']}",
            cfg.natural_plan_raw / TASKS[task]["raw_name"],
            force=args.force,
        )
        _preprocess(
            task,
            iter_fn=TASKS[task]["iter"],
            raw_dir=cfg.natural_plan_raw,
            out_path=cfg.natural_plan_preprocessed / TASKS[task]["preprocessed_name"],
            force=args.force,
        )
        _carve(
            task,
            cfg,
            split_dir=split_dirs[task],
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
            force=args.force,
        )

    print("done.")


if __name__ == "__main__":
    main()
