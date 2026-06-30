"""Download BioRED, preprocess into OA-aligned JSONL, carve a GEPA split.

BioRED (Luo et al., Briefings in Bioinformatics 2022) is 600 PubMed abstracts
with normalized biomedical entities + document-level relations. The corpus is a
single public ``BIORED.zip`` (no gate), so the download is a plain fetch. See
research/opus48_RA_real_world_datasets_2.md for why BioRED is the top *real*
RA-as-fitness demonstrator (opaque accession ids → strict routing starved).

Pipeline:

  1. Ensure data/biored/raw/BioRED/{Train,Dev,Test}.BioC.JSON exist (download +
     unzip BIORED.zip if not).
  2. Parse each split into the OA-aligned shape (DocRED-style entities +
     relations) -> data/biored/preprocessed/{train,dev,test}.jsonl
  3. Carve a fixed-seed split under
       data/biored/splits/gepa_biored/main/{train,val,test,test_full}.jsonl
     plus a manifest.json. `test` is a subset of `test_full` (the full BioRED
     test split) for a tight native-metric estimate, mirroring prepare_docred.py.

Single variant: BioRED's relation types are readable words, and the
non-derivability comes from the gold entity `id` being an opaque concept
accession (never shown to the model), so there is no obfuscation step.
Idempotent unless --force.

Usage::

    uv run python scripts/prepare_biored.py
    uv run python scripts/prepare_biored.py --n-train 50 --n-val 50 --n-test 100
"""

from __future__ import annotations

import argparse
import json
import random

from tqdm import tqdm

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets.biored import (
    BIORED_ZIP_URL,
    ensure_biored_data,
    iter_biored_examples,
    load_jsonl,
    write_jsonl,
)


def _preprocess(*, cfg: ExpConfig, force: bool) -> None:
    out_dir = cfg.biored_preprocessed
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        out_path = out_dir / f"{split}.jsonl"
        if out_path.exists() and not force:
            n = sum(1 for _ in out_path.open("r"))
            print(f"[skip] {out_path} already has {n} examples (use --force to rebuild).")
            continue
        print(f"[parse] BioRED {split!r}")
        examples = list(tqdm(
            iter_biored_examples(split, raw_dir=cfg.biored_raw),
            unit="doc", desc=split,
        ))
        n = write_jsonl(out_path, examples)
        print(f"[write] {out_path}  ({n} examples)")


def _self_test(*, cfg: ExpConfig) -> None:
    """Validate the preprocessed JSONL: ids unique, relations resolve."""
    for split in ("train", "dev", "test"):
        recs = load_jsonl(cfg.biored_preprocessed / f"{split}.jsonl")
        if not recs:
            raise RuntimeError(f"{split}: produced 0 examples")
        for ex in recs:
            gold = ex["gold"]
            ids = {e["id"] for e in gold["entities"]}
            if len(ids) != len(gold["entities"]):
                raise RuntimeError(f"{split} {ex['id']}: duplicate entity ids")
            for rel in gold["relations"]:
                if rel["subject"] not in ids or rel["object"] not in ids:
                    raise RuntimeError(
                        f"{split} {ex['id']}: relation endpoint not in entities"
                    )
    print("[ok] self-test passed")


def _carve(*, cfg: ExpConfig, n_train: int, n_val: int, n_test: int,
           n_test_full: int, seed: int, force: bool) -> None:
    split_dir = cfg.gepa_biored_main
    outs = {
        "train": split_dir / "train.jsonl",
        "val": split_dir / "val.jsonl",
        "test": split_dir / "test.jsonl",
        "test_full": split_dir / "test_full.jsonl",
    }
    manifest_out = split_dir / "manifest.json"
    if all(p.exists() for p in (*outs.values(), manifest_out)) and not force:
        print(f"[skip] {split_dir} already populated (use --force to rebuild).")
        return

    pre = cfg.biored_preprocessed
    pool_train = load_jsonl(pre / "train.jsonl")
    pool_dev = load_jsonl(pre / "dev.jsonl")
    pool_test = load_jsonl(pre / "test.jsonl")

    rng = random.Random(seed)
    train_pick = rng.sample(pool_train, min(n_train, len(pool_train)))
    val_pick = rng.sample(pool_dev, min(n_val, len(pool_dev)))
    # test_full = the whole BioRED test split (deterministic shuffle for order);
    # test = a fixed n_test subset of it.
    test_full = list(pool_test)
    rng.shuffle(test_full)
    test_full = test_full[:min(n_test_full, len(test_full))]
    test_pick = test_full[:min(n_test, len(test_full))]

    split_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outs["train"], train_pick)
    write_jsonl(outs["val"], val_pick)
    write_jsonl(outs["test"], test_pick)
    write_jsonl(outs["test_full"], test_full)

    def block(picks, source_split, out_path):
        return {
            "n": len(picks),
            "source_split": source_split,
            "source_ids": [ex["id"] for ex in picks],
            "path": str(out_path.relative_to(cfg.data_root.parent)),
        }

    manifest = {
        "dataset": "BioRED (Luo et al., Briefings in Bioinformatics 2022)",
        "source": BIORED_ZIP_URL,
        "schema_id": "gepa_biored",
        "schema_path": str(cfg.biored_schema_path.relative_to(cfg.data_root.parent)),
        "seed": seed,
        "note": (
            "DocRED-shaped end-to-end biomedical relation extraction (text -> "
            "typed coref-entities + relations). The gold entity id is the opaque "
            "normalized concept accession (Entrez/MeSH/dbSNP/...), never shown to "
            "the model, so a literal-id relation compare is starved and "
            "referential alignment carries the routing. train/val/test are "
            "disjoint; test is a subset of test_full."
        ),
        "splits": {
            "train": block(train_pick, "train", outs["train"]),
            "val": block(val_pick, "dev", outs["val"]),
            "test": block(test_pick, "test", outs["test"]),
            "test_full": block(test_full, "test", outs["test_full"]),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        f"[write] {split_dir}/  ({len(train_pick)} train + {len(val_pick)} val + "
        f"{len(test_pick)} test, {len(test_full)} test_full)"
    )


def main() -> None:
    cfg = ExpConfig()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train", type=int, default=50, help="carve train size (default 50)")
    p.add_argument("--n-val", type=int, default=50, help="carve val size (default 50)")
    p.add_argument("--n-test", type=int, default=100, help="carve test size (default 100)")
    p.add_argument("--n-test-full", type=int, default=100,
                   help="carve test_full cap (default 100 = full BioRED test)")
    p.add_argument("--seed", type=int, default=20260608,
                   help="RNG seed for the split carve (default 20260608)")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if outputs exist (also re-downloads raw)")
    args = p.parse_args()

    ensure_biored_data(cfg.biored_raw, force=args.force)
    _preprocess(cfg=cfg, force=args.force)
    _self_test(cfg=cfg)
    _carve(cfg=cfg, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
           n_test_full=args.n_test_full, seed=args.seed, force=args.force)

    print("done.")


if __name__ == "__main__":
    main()
