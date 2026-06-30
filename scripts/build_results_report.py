"""Render one large, exact, LLM-free Markdown report of every run in data/runs.

The output is the single source document a later stage feeds to an LLM that
writes the paper's "Experimental Results" section: it must be *exact* and
*guaranteed* free of fabricated numbers, so every figure is computed
deterministically by :mod:`object_aligner_exp.report.results` (no LLM) and
rendered through Jinja2 templates under ``report/templates/results/``. A
machine-readable ``results.json`` sidecar carrying the same numbers is written
next to it.

Usage::

    uv run python scripts/build_results_report.py
    uv run python scripts/build_results_report.py --all-datasets --all-models
    uv run python scripts/build_results_report.py \
        --datasets scierc_native_exact,planbench_blocksworld \
        --models gemma4-26b,gemma4-e4b --out data/results/results.md

By default only the publication subset of datasets and the two gemma4 task LMs
are included; everything else that appears under data/runs is recorded and
listed (not silently dropped).
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from object_aligner_exp.report.results import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    Discovery,
    aggregate,
    discover_runs,
)
from object_aligner_exp.report.results_render import TABLES_FILENAME, build_tables, render

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_json(disc: Discovery, agg: dict[str, Any], generated: str) -> dict[str, Any]:
    return {
        "generated_at": generated,
        "n_summaries_found": disc.n_summaries_found,
        "n_runs_included": len(disc.runs),
        "models_kept": list(disc.models_kept),
        "models_seen": sorted(disc.models_seen),
        "datasets_kept": list(disc.datasets_kept) if disc.datasets_kept is not None else None,
        "datasets_seen": sorted(disc.datasets_seen),
        "excluded_datasets": disc.excluded_datasets,
        "duplicate_resolutions": disc.duplicate_resolutions,
        "excluded_model_cells": disc.excluded_model_cells,
        "native_errors": disc.native_errors,
        "skipped_paths": disc.skipped_paths,
        "datasets": agg,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs-root", type=Path, default=REPO_ROOT / "data" / "runs")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "results" / "results.md")
    p.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated task_lm models to include (default: the two gemma4).",
    )
    p.add_argument("--all-models", action="store_true", help="Include every model (no filter).")
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets to include (default: the publication subset).",
    )
    p.add_argument("--all-datasets", action="store_true", help="Include every dataset (no filter).")
    p.add_argument(
        "--appendix",
        action="store_true",
        help="Include the full per-cell OA appendix (hidden by default).",
    )
    p.add_argument(
        "--failed-samples",
        action="store_true",
        help="Include per-cell failed-sample accounting (hidden by default).",
    )
    p.add_argument("--no-json", action="store_true", help="Skip the results.json sidecar.")
    args = p.parse_args()

    models = None if args.all_models else tuple(m for m in args.models.split(",") if m)
    datasets = None if args.all_datasets else tuple(d for d in args.datasets.split(",") if d)

    disc = discover_runs(args.runs_root, models=models, datasets=datasets, repo_root=REPO_ROOT)
    agg = aggregate(disc)
    generated = datetime.datetime.now().isoformat(timespec="seconds")

    tables_path = args.out.with_name(TABLES_FILENAME)
    md = render(
        disc,
        agg,
        runs_root=args.runs_root,
        generated=generated,
        repo_root=REPO_ROOT,
        show_appendix=args.appendix,
        show_failed=args.failed_samples,
        tables_file=tables_path.name,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(
        f"[done] {len(disc.runs)} runs from {disc.n_summaries_found} summaries "
        f"-> {args.out} ({len(md):,} chars)"
    )
    if disc.excluded_datasets:
        print(f"  excluded {len(disc.excluded_datasets)} datasets: {', '.join(disc.excluded_datasets)}")
    if disc.duplicate_resolutions:
        print(f"  resolved {len(disc.duplicate_resolutions)} duplicate seed-batches")
    if disc.excluded_model_cells:
        print(f"  excluded {len(disc.excluded_model_cells)} non-selected-model runs")
    if disc.native_errors:
        print(f"  {len(disc.native_errors)} native recomputation issues (see report)")

    if not args.no_json:
        json_path = args.out.with_suffix(".json")
        json_path.write_text(json.dumps(build_json(disc, agg, generated), indent=2))
        print(f"[done] sidecar -> {json_path}")
        tables = build_tables(disc, agg, generated=generated, repo_root=REPO_ROOT)
        tables_path.write_text(json.dumps(tables, indent=2))
        print(
            f"[done] tables -> {tables_path} "
            f"({len(tables['tables'])} tables, "
            f"{len(tables['seeds']['oa'])} oa-seed rows, "
            f"{len(tables['seeds']['native'])} native-seed rows)"
        )


if __name__ == "__main__":
    main()
