"""Refresh ``summary.json`` aggregates for finished runs (dataset-agnostic).

Re-aggregates each run's ``holdout_scores.jsonl`` *offline* — no LM calls.
Two complementary uses:

1. **Backfill new aggregate fields.** When ``summarize()`` gains new keys
   (e.g. ``mean_score_nonzero``), running this script over older runs
   rewrites their ``summary.json`` with the new fields.

2. **Add cross-schema scores.** Pass one or more ``--cross-schema PATH``;
   each row's cached ``response`` is re-parsed and re-scored against the
   additional schemas, and the per-row ``cross_scores`` + the
   ``summary['holdout']['scores']`` aggregate are rewritten.

Runs that lack ``summary.json`` or ``holdout_scores.jsonl`` are skipped
(e.g. runs aborted mid-optimization).

Usage::

    # Refresh-only (no cross schemas): backfill new summarize() fields.
    uv run python scripts/recompute_cross_scores.py data/runs/gepa_*

    # Add cross schemas (re-scored from cached responses; no LM).
    uv run python scripts/recompute_cross_scores.py \\
        --cross-schema data/scierc/schemas/scierc_strict.jsonc \\
        data/runs/gepa_scierc_2k5_feedback_s* \\
        data/runs/gepa_scierc_no_score_s*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from object_aligner_exp.holdout_eval import rebuild_summary_from_disk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more run directories (expanded by the shell glob).",
    )
    ap.add_argument(
        "--cross-schema",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        dest="cross_schema",
        help=(
            "Additional schema (.jsonc) to also score each cached response "
            "against. Repeat to add several. A cross-schema path that "
            "matches the run's primary schema is dropped with a warning. "
            "Omit to run a refresh-only pass that only re-aggregates "
            "summary.json from existing rows."
        ),
    )
    args = ap.parse_args()
    n_ok = 0
    n_skip = 0
    for run_dir in args.run_dirs:
        try:
            summary = rebuild_summary_from_disk(
                run_dir, cross_schema_paths=args.cross_schema
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {run_dir.name}: {exc}")
            n_skip += 1
            continue
        if summary is None:
            print(f"[skip] {run_dir.name}: missing summary/holdout/schema_path")
            n_skip += 1
            continue
        holdout = summary.get("holdout") or {}
        scores = holdout.get("scores") or {}
        primary = holdout.get("primary_kind")
        if primary and primary in scores:
            head = scores[primary]
        elif scores:
            primary, head = next(iter(scores.items()))
        else:
            head = {}
        cross = scores
        line = (
            f"[ok] {run_dir.name}: "
            f"n={head.get('n', '?')} "
            f"mean={head.get('mean_score', 0.0):.4f} "
            f"mean_nz={head.get('mean_score_nonzero', 0.0):.4f}"
        )
        if len(cross) > 1:
            line += "  cross={" + ", ".join(
                f"{k}={v.get('mean_score', 0.0):.4f}"
                for k, v in cross.items() if k != primary
            ) + "}"
        print(line)
        n_ok += 1
    print(f"\n[summary] {n_ok} updated, {n_skip} skipped")
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
