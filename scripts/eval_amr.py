"""Canonical Smatch F1 evaluation of an OA / GEPA AMR run.

Grades a run's held-out predictions against the AMR test split with the standard
**Smatch** metric (micro-averaged P/R/F1). OA's ``amr_ra`` score is the GEPA
reward; Smatch is the leaderboard metric. Library: ``amr_eval.score_corpus``.

Usage::

    # point at a run directory (uses holdout_scores.jsonl + the test split)
    uv run python scripts/eval_amr.py \\
        --run-dir data/runs/gepa_amr_littleprince/oa_feedback/ra/gemma4-26b/s0_...

    # or be explicit
    uv run python scripts/eval_amr.py \\
        --predictions data/runs/<run>/holdout_scores.jsonl \\
        --split       data/amr/splits/gepa_amr_littleprince/main/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from object_aligner_exp.amr_eval import score_corpus
from object_aligner_exp.config import ExpConfig


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    cfg = ExpConfig()
    default_split = cfg.gepa_amr_littleprince_main / "test.jsonl"
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Run directory; defaults --predictions to "
                        "<run-dir>/holdout_scores.jsonl.")
    p.add_argument("--predictions", type=Path, default=None,
                   help="JSONL of OA predictions: rows {id, response, ...}.")
    p.add_argument("--split", type=Path, default=None,
                   help=f"AMR test split JSONL (default {default_split}).")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write amr_eval.json (default: alongside predictions).")
    args = p.parse_args()

    predictions_path = args.predictions
    if args.run_dir is not None and predictions_path is None:
        predictions_path = args.run_dir / "holdout_scores.jsonl"
    if predictions_path is None:
        p.error("provide --run-dir or --predictions")

    # Prefer the split this run was actually scored on (config.json's test_path),
    # so a non-littleprince run isn't silently graded against the wrong gold.
    split_path = args.split
    if split_path is None and args.run_dir is not None:
        cfg_path = args.run_dir / "config.json"
        if cfg_path.exists():
            test_path = json.loads(cfg_path.read_text()).get("test_path")
            if test_path:
                split_path = Path(test_path)
    if split_path is None:
        split_path = default_split

    predictions = _load_jsonl(predictions_path)
    examples_by_id = {str(ex["id"]): ex for ex in _load_jsonl(split_path)}
    print(f"[load] {len(predictions)} predictions from {predictions_path}; "
          f"{len(examples_by_id)} gold from {split_path}")

    res = score_corpus(predictions, examples_by_id)

    out_path = args.output or predictions_path.parent / "amr_eval.json"
    payload = {
        "predictions_file": str(predictions_path),
        "split_file": str(split_path),
        "n": res.n,
        "n_parse_failures": res.n_parse_failures,
        "match_total": res.match_total,
        "test_total": res.test_total,
        "gold_total": res.gold_total,
        "smatch": {
            "precision": res.smatch_precision,
            "recall": res.smatch_recall,
            "f1": res.smatch_f1,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print()
    print("AMR Smatch evaluation")
    print(f"  Predictions: {predictions_path}  ({res.n} examples)")
    print(f"  Gold:        {split_path}")
    if res.n_parse_failures:
        print(f"  Parse failures: {res.n_parse_failures} (→ empty preds / full FN)")
    print(f"  Smatch  P={res.smatch_precision:.4f}  R={res.smatch_recall:.4f}  "
          f"F1={res.smatch_f1:.4f}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
