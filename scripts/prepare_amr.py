"""Build the AMR dataset: parse a PENMAN corpus, carve splits.

Pipeline (mirrors ``scripts/prepare_sentence_ordering.py``):

  1. Parse a PENMAN corpus (``# ::id`` / ``# ::snt`` blocks) into the fixed AMR
     wire shape (see ``datasets/amr``). Write
       data/amr/preprocessed/<variant>.jsonl
  2. SELF-TEST: assert every gold graph round-trips
     (penman -> gold JSON -> triples reproduces the graph's triple set).
     Aborts loudly on any failure (guards the conversion).
  3. Carve a fixed-seed split (default 100/100/200) under
       data/amr/splits/gepa_amr_<variant>/main/
     with train/val/test mutually disjoint plus ``test_full`` (the large
     held-out remainder, for tight Smatch estimates; ``test`` is a subset).
     The Little Prince carve is UNSTRATIFIED — a plain deterministic shuffle.

Two corpus variants are wired (select with ``--variant``):

* ``littleprince`` (default) — freely-available Little Prince v3.0 (1,562
  sentences), expected at ``data/amr/raw/amr-bank-struct-v3.0.txt``.

* ``bio`` — Bio AMR Corpus v3.0 (USC/ISI; ~6,952 biomedical sentences),
  expected at ``data/amr/raw/amr-release-bio-v3.0.txt``.

Both corpora are downloaded automatically into ``data/amr/raw/`` on first run
(the original ISI host for the Bio corpus is gone, so we fetch the archived
copy). Override the local path / provide your own corpus with ``--source PATH``;
when ``--source`` is given the auto-download is skipped. Provenance:
https://amr.isi.edu/download.html

Idempotent: re-running with the same args is a no-op unless ``--force``.

Usage::

    uv run python scripts/prepare_amr.py
    uv run python scripts/prepare_amr.py --variant bio
    uv run python scripts/prepare_amr.py --print-samples 3
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

from tqdm import tqdm

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets.amr import AmrExample, iter_amr_examples, roundtrip_ok
from object_aligner_exp.datasets.rebel import load_jsonl, write_jsonl

# Per-variant corpus config. ``raw`` is the default filename under
# ``data/amr/raw/``; ``url`` is fetched automatically when that file is missing
# (and no ``--source`` override is given); ``source`` is the human-readable
# provenance recorded in the split manifest; ``split_attr`` names the ExpConfig
# property for the split dir.
_VARIANTS: dict[str, dict[str, str]] = {
    "littleprince": {
        "raw": "amr-bank-struct-v3.0.txt",
        "url": (
            "https://raw.githubusercontent.com/flipz357/AMR-World/main/"
            "data/reference_amrs/amr-bank-struct-v3.0.txt"
        ),
        "source": (
            "Little Prince v3.0 (amr-bank-struct-v3.0.txt; amr.isi.edu / "
            "github.com/flipz357/AMR-World)"
        ),
        "split_attr": "gepa_amr_littleprince_main",
    },
    "bio": {
        "raw": "amr-release-bio-v3.0.txt",
        # Original ISI host (amr.isi.edu/download/2018-01-25/...) is gone; fetch
        # the archived copy. The `id_` Wayback modifier returns the raw file.
        # NB: use the 2022-12-23 capture — it is the full 8,981,404-byte file;
        # later snapshots (e.g. 2023-12-07) were truncated by the crawler to 1
        # MiB and fail to parse.
        "url": (
            "https://web.archive.org/web/20221223153749id_/"
            "https://amr.isi.edu/download/2018-01-25/amr-release-bio-v3.0.txt"
        ),
        "source": (
            "Bio AMR Corpus v3.0 (amr-release-bio-v3.0.txt; amr.isi.edu / "
            "USC ISI; ~6,952 biomedical sentences)"
        ),
        "split_attr": "gepa_amr_bio_main",
    },
}


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def _preprocess(source: Path, *, out_path: Path, force: bool, url: str | None = None) -> Path:
    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open("r"))
        print(f"[skip] {out_path} already has {n} examples (use --force to rebuild).")
        return out_path
    if not source.exists():
        if url is not None:
            _download_file(url, source)
        else:
            raise FileNotFoundError(
                f"AMR corpus not found at {source}. Download it (see this "
                f"script's docstring) or pass --source PATH."
            )
    print(f"[parse] AMR PENMAN corpus {source}")
    examples = list(tqdm(iter_amr_examples(str(source)), unit="ex", desc="amr"))
    _self_test(source, examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_path, examples)
    print(f"[write] {out_path}  ({n} examples)")
    return out_path


def _self_test(source: Path, examples: list[AmrExample]) -> None:
    """Abort unless every gold AMR round-trips through the JSON wire shape."""
    import penman

    graphs = {
        (g.metadata.get("id") or "").strip(): g
        for g in penman.loads(source.read_text(encoding="utf-8"))
        if g.instances()
    }
    bad: list[str] = []
    for ex in examples:
        g = graphs.get(ex["id"])
        if g is None or not roundtrip_ok(g):
            bad.append(ex["id"])
    if bad:
        sample = ", ".join(bad[:10])
        raise RuntimeError(
            f"SELF-TEST FAILED: {len(bad)}/{len(examples)} examples do not "
            f"round-trip (conversion bug?). First few: {sample}"
        )
    print(f"[self-test] OK — all {len(examples)} gold AMRs round-trip")


def _carve(
    cfg: ExpConfig,
    variant: str,
    *,
    source_desc: str,
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
        raise RuntimeError(f"{variant}: pool has {len(pool)} < {need} needed")

    order = list(pool)
    random.Random(seed).shuffle(order)  # unstratified deterministic shuffle
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    rest = order[n_train + n_val :]      # complement of train+val
    test_full = rest[:n_test_full]       # large held-out remainder (capped)
    test = test_full[:n_test]            # cheap default test (subset)

    split_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(split_dir / "train.jsonl", train)
    write_jsonl(split_dir / "val.jsonl", val)
    write_jsonl(split_dir / "test.jsonl", test)
    write_jsonl(split_dir / "test_full.jsonl", test_full)

    rel = lambda p: str(p.relative_to(cfg.data_root.parent))  # noqa: E731
    manifest = {
        "dataset": f"AMR — {variant}",
        "source": source_desc,
        "variant": variant,
        "difficulty_key": None,
        "seed": seed,
        "note": (
            "train/val/test are mutually disjoint; the carve is a plain "
            "deterministic shuffle (UNSTRATIFIED). test is a subset of "
            "test_full, the large held-out remainder kept for tight Smatch "
            "estimates. AMR variable names are arbitrary, so referential "
            "alignment (amr_ra) = Smatch and the ra-vs-strict gap is pure "
            "id-routing."
        ),
        "splits": {
            name: {
                "n": len(rows),
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


def _print_samples(source: Path, n: int) -> None:
    for ex in iter_amr_examples(str(source), limit=n):
        g = ex["gold"]
        print(f"--- {ex['id']}  (n_nodes={ex['meta']['n_nodes']}) ---")
        print(ex["context"])
        print("root:", g["root"])
        print("nodes:", [(x["id"], x["concept"]) for x in g["nodes"]])
        print("relations:", [(r["source"], r["role"], r["target"]) for r in g["relations"]])
        print()


def main() -> None:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="littleprince", choices=list(_VARIANTS),
                   help="corpus variant name")
    p.add_argument("--source", type=Path, default=None,
                   help="PENMAN corpus path (default: data/amr/raw/<variant raw file>)")
    p.add_argument("--n-train", type=int, default=100, help="train size (D_feedback)")
    p.add_argument("--n-val", type=int, default=100, help="val size (D_pareto)")
    p.add_argument("--n-test", type=int, default=200, help="default held-out test size")
    p.add_argument("--n-test-full", type=int, default=2000, help="cap on test_full")
    p.add_argument("--seed", type=int, default=20260530, help="RNG seed (shuffle + carve)")
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    p.add_argument("--print-samples", type=int, default=0, metavar="N",
                   help="render N parsed examples and exit (no writes)")
    args = p.parse_args()

    entry = _VARIANTS[args.variant]
    source = args.source or (cfg.amr_raw / entry["raw"])

    if args.print_samples:
        _print_samples(source, args.print_samples)
        return

    cfg.amr_preprocessed.mkdir(parents=True, exist_ok=True)
    preprocessed = cfg.amr_preprocessed / f"{args.variant}.jsonl"
    # Auto-download only the canonical corpus; a user-supplied --source is used
    # verbatim (no download).
    url = None if args.source else entry.get("url")
    _preprocess(source, out_path=preprocessed, force=args.force, url=url)
    split_dir = getattr(cfg, entry["split_attr"])
    _carve(
        cfg,
        args.variant,
        source_desc=entry["source"],
        split_dir=split_dir,
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
