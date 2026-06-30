"""Download SciERC, preprocess into OA-aligned JSONL, carve the GEPA_SCIERC pilot.

Pipeline:

  1. Ensure ``data/scierc/raw/sciERC_processed.tar.gz`` is present (download
     from http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz
     if not).
  2. Extract only ``processed_data/json/{train,dev,test}.json`` (~1 MB)
     from the tarball — the bundled ELMo embeddings (~600 MB) are skipped.
  3. Parse each JSON document into the OA-aligned shape and write to
       data/scierc/preprocessed/{train,dev,test}.jsonl
  4. Carve a fixed-seed three-way pilot under
       data/scierc/splits/gepa_scierc/pilot/{train,val,test}.jsonl
     plus a ``manifest.json`` recording sizes, seed, source ids, and the
     schema path the run was carved against.

The native SciERC split is 350 train / 50 dev / 100 test (500 total). The
pilot mirrors the GEPA paper's protocol:

  * **train** (D_feedback) — sampled from native train (default 50).
  * **val**   (D_pareto)   — sampled from native dev   (default 50; the
    full dev split, no subsampling at the default size).
  * **test**  (held out)   — sampled from native test  (default 100; the
    full test split, no subsampling at the default size).

Idempotent: re-running with the same args is a no-op unless ``--force``.

Usage::

    uv run python scripts/prepare_scierc.py
    uv run python scripts/prepare_scierc.py --pilot-train 50 --pilot-val 50 --pilot-test 100
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

from tqdm import tqdm

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets.scierc import (
    ensure_scierc_json,
    iter_scierc_examples,
    load_jsonl,
    write_jsonl,
)


SCIERC_URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
SCIERC_TARBALL_NAME = "sciERC_processed.tar.gz"


def _download_tarball(raw_dir: Path, *, force: bool) -> Path:
    tarball = raw_dir / SCIERC_TARBALL_NAME
    if tarball.exists() and not force:
        print(f"[skip] {tarball} already exists (use --force to redownload).")
        return tarball

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {SCIERC_URL} → {tarball}")

    # Stream-download with a tqdm bar; the file is ~700 MB because it carries
    # ELMo embeddings we'll throw away in extraction.
    with urllib.request.urlopen(SCIERC_URL) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        tmp = tarball.with_suffix(tarball.suffix + ".part")
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=SCIERC_TARBALL_NAME,
        ) as bar:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(tarball)

    return tarball


def _preprocess_split(
    split: str,
    *,
    raw_dir: Path,
    out_path: Path,
    force: bool,
) -> Path:
    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open("r"))
        print(f"[skip] {out_path} already has {n} examples (use --force to rebuild).")
        return out_path

    print(f"[parse] SciERC {split!r}")
    examples = list(tqdm(iter_scierc_examples(split, raw_dir=raw_dir),
                         unit="doc", desc=split))
    n = write_jsonl(out_path, examples)
    print(f"[write] {out_path}  ({n} examples)")
    return out_path


def _carve_pilot(
    cfg: ExpConfig,
    *,
    pilot_train_n: int,
    pilot_val_n: int,
    pilot_test_n: int,
    seed: int,
    force: bool,
) -> None:
    split_dir = cfg.gepa_scierc_pilot
    train_out = split_dir / "train.jsonl"
    val_out = split_dir / "val.jsonl"
    test_out = split_dir / "test.jsonl"
    manifest_out = split_dir / "manifest.json"

    outputs = (train_out, val_out, test_out, manifest_out)
    if all(p.exists() for p in outputs) and not force:
        print(f"[skip] {split_dir} already populated (use --force to rebuild).")
        return

    pool_train = load_jsonl(cfg.scierc_preprocessed / "train.jsonl")
    pool_dev = load_jsonl(cfg.scierc_preprocessed / "dev.jsonl")
    pool_test = load_jsonl(cfg.scierc_preprocessed / "test.jsonl")

    for label, pool, n in (
        ("train", pool_train, pilot_train_n),
        ("dev", pool_dev, pilot_val_n),
        ("test", pool_test, pilot_test_n),
    ):
        if len(pool) < n:
            raise RuntimeError(
                f"{label} pool only has {len(pool)} examples; need {n}"
            )

    rng = random.Random(seed)
    train_pick = rng.sample(pool_train, pilot_train_n)
    val_pick = rng.sample(pool_dev, pilot_val_n)
    test_pick = rng.sample(pool_test, pilot_test_n)

    write_jsonl(train_out, train_pick)
    write_jsonl(val_out, val_pick)
    write_jsonl(test_out, test_pick)

    manifest = {
        "dataset": "SciERC (Luan et al., EMNLP 2018)",
        "source_url": SCIERC_URL,
        "schema_id": "gepa_scierc",
        "schema_path": str(cfg.scierc_schema_path.relative_to(cfg.data_root.parent)),
        "seed": seed,
        "train": {
            "n": pilot_train_n,
            "source_split": "train",
            "source_ids": [ex["id"] for ex in train_pick],
            "path": str(train_out.relative_to(cfg.data_root.parent)),
        },
        "val": {
            "n": pilot_val_n,
            "source_split": "dev",
            "source_ids": [ex["id"] for ex in val_pick],
            "path": str(val_out.relative_to(cfg.data_root.parent)),
        },
        "test": {
            "n": pilot_test_n,
            "source_split": "test",
            "source_ids": [ex["id"] for ex in test_pick],
            "path": str(test_out.relative_to(cfg.data_root.parent)),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        f"[write] {split_dir}/  "
        f"({pilot_train_n} train + {pilot_val_n} val + {pilot_test_n} test)"
    )


def _carve_native(cfg: ExpConfig, *, force: bool) -> None:
    """Mirror the full native split (train→train, dev→val, test→test).

    No subsampling: each split is the whole preprocessed pool, so this is
    the upper-bound counterpart to the pilot carve (350/50/100 nominal;
    349 train in practice since one source doc drops during parsing).
    """
    split_dir = cfg.gepa_scierc_native
    train_out = split_dir / "train.jsonl"
    val_out = split_dir / "val.jsonl"
    test_out = split_dir / "test.jsonl"
    manifest_out = split_dir / "manifest.json"

    outputs = (train_out, val_out, test_out, manifest_out)
    if all(p.exists() for p in outputs) and not force:
        print(f"[skip] {split_dir} already populated (use --force to rebuild).")
        return

    train_pick = load_jsonl(cfg.scierc_preprocessed / "train.jsonl")
    val_pick = load_jsonl(cfg.scierc_preprocessed / "dev.jsonl")
    test_pick = load_jsonl(cfg.scierc_preprocessed / "test.jsonl")

    split_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_out, train_pick)
    write_jsonl(val_out, val_pick)
    write_jsonl(test_out, test_pick)

    manifest = {
        "dataset": "SciERC (Luan et al., EMNLP 2018)",
        "source_url": SCIERC_URL,
        "schema_id": "gepa_scierc",
        "schema_path": str(cfg.scierc_schema_path.relative_to(cfg.data_root.parent)),
        "seed": None,
        "train": {
            "n": len(train_pick),
            "source_split": "train",
            "source_ids": [ex["id"] for ex in train_pick],
            "path": str(train_out.relative_to(cfg.data_root.parent)),
        },
        "val": {
            "n": len(val_pick),
            "source_split": "dev",
            "source_ids": [ex["id"] for ex in val_pick],
            "path": str(val_out.relative_to(cfg.data_root.parent)),
        },
        "test": {
            "n": len(test_pick),
            "source_split": "test",
            "source_ids": [ex["id"] for ex in test_pick],
            "path": str(test_out.relative_to(cfg.data_root.parent)),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        f"[write] {split_dir}/  "
        f"({len(train_pick)} train + {len(val_pick)} val + {len(test_pick)} test)"
    )


def main() -> None:
    cfg = ExpConfig()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-train", type=int, default=50,
                   help="gepa_scierc pilot train size — D_feedback (default 50)")
    p.add_argument("--pilot-val", type=int, default=50,
                   help="gepa_scierc pilot val size — D_pareto (default 50; full dev)")
    p.add_argument("--pilot-test", type=int, default=100,
                   help="gepa_scierc pilot held-out test size (default 100; full test)")
    p.add_argument("--seed", type=int, default=20260515,
                   help="RNG seed for pilot subsampling (default 20260515)")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if outputs already exist (also re-downloads the tarball)")
    args = p.parse_args()

    cfg.scierc_preprocessed.mkdir(parents=True, exist_ok=True)
    cfg.scierc_splits.mkdir(parents=True, exist_ok=True)

    _download_tarball(cfg.scierc_raw, force=args.force)
    # Tickle extraction now so a missing-file error surfaces before we burn
    # CPU on per-split parsing loops.
    ensure_scierc_json(cfg.scierc_raw)

    for split in ("train", "dev", "test"):
        _preprocess_split(
            split,
            raw_dir=cfg.scierc_raw,
            out_path=cfg.scierc_preprocessed / f"{split}.jsonl",
            force=args.force,
        )

    _carve_pilot(
        cfg,
        pilot_train_n=args.pilot_train,
        pilot_val_n=args.pilot_val,
        pilot_test_n=args.pilot_test,
        seed=args.seed,
        force=args.force,
    )

    _carve_native(cfg, force=args.force)

    print("done.")


if __name__ == "__main__":
    main()
