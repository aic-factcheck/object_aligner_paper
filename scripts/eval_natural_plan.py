"""Official NATURAL PLAN exact-match evaluation of an OA / GEPA run.

Grades a run's held-out predictions under the *leaderboard* metric (binary
exact-match solve-rate, overall + by difficulty), so the OA / GEPA numbers can
be placed against published NATURAL PLAN SOTA. This is distinct from the OA
graded score used as the GEPA reward — see ``natural_plan_eval`` and the
schemas under ``data/natural_plan/``.

Reads ``holdout_scores.jsonl`` (rows ``{id, response, ...}``) and the split
file the run was evaluated on (for the gold ``meta``: cities/durations for
trip; constraints/dist_matrix/golden_plan for meeting).

Usage::

    # point at a run directory; --task selects trip or meeting
    uv run python scripts/eval_natural_plan.py --run-dir <run> --task trip

    # or be explicit about the predictions + split
    uv run python scripts/eval_natural_plan.py --task meeting \\
        --predictions <run>/holdout_scores.jsonl \\
        --split data/natural_plan/splits/gepa_meeting_planning/main/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets.rebel import load_jsonl
from object_aligner_exp.natural_plan_eval import TaskResult, score_corpus

CAVEATS = [
    "This is the official NATURAL PLAN binary exact-match solve-rate, NOT OA's "
    "graded alignment score (the GEPA reward). The two measure different "
    "things by design.",
    "Trip: a prediction scores 1 only if its (city, days) sequence matches the "
    "gold cities/durations for every gold stay, in order (extra trailing stays "
    "ignored; a short plan fails).",
    "Meeting: a prediction scores 1 only if its count of VALID meetings "
    "(right location, inside the availability window, travel times respected, "
    "no duplicates) equals the golden plan's count. The validator uses each "
    "person's REQUIRED meeting length from constraints, not the 'duration' the "
    "model emits — so 'duration' affects the OA reward but not this metric.",
    "Responses that don't parse to a JSON object count as incorrect "
    "(tallied under n_parse_failures).",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _print_summary(res: TaskResult, *, predictions: Path, split: Path) -> None:
    print()
    print(f"NATURAL PLAN evaluation — {res.task} (official exact-match)")
    print(f"  Predictions: {predictions}  ({res.n} examples)")
    print(f"  Split:       {split}")
    if res.n_parse_failures:
        print(f"  Parse failures: {res.n_parse_failures} (→ counted incorrect)")
    print()
    print(f"  SOLVE RATE (overall): {res.solve_rate:.4f}  "
          f"({res.n_correct}/{res.n})")
    print()
    diff_label = "num_cities" if res.task == "trip" else "num_people"
    print(f"  by {diff_label}:")
    for diff in sorted(res.by_difficulty):
        correct, total = res.by_difficulty[diff]
        rate = correct / total if total else 0.0
        print(f"    {diff:>3}: {rate:.4f}  ({correct}/{total})")


def main() -> None:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task", choices=["trip", "meeting"], required=True)
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Run dir; defaults --predictions to "
                        "<run-dir>/holdout_scores.jsonl.")
    p.add_argument("--predictions", type=Path, default=None,
                   help="JSONL of holdout predictions: rows {id, response, ...}.")
    p.add_argument("--split", type=Path, default=None,
                   help="Split JSONL with gold meta (default: the task's "
                        "gepa_*_planning/main/test.jsonl).")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write natural_plan_eval.json (default: "
                        "alongside the predictions file).")
    args = p.parse_args()

    predictions_path = args.predictions
    if args.run_dir is not None and predictions_path is None:
        predictions_path = args.run_dir / "holdout_scores.jsonl"
    if predictions_path is None:
        p.error("provide --run-dir or --predictions")

    split_path = args.split
    if split_path is None:
        split_dir = (
            cfg.gepa_trip_planning_main if args.task == "trip"
            else cfg.gepa_meeting_planning_main
        )
        split_path = split_dir / "test.jsonl"

    predictions = _load_jsonl(predictions_path)
    examples = load_jsonl(split_path)
    examples_by_id = {str(ex["id"]): ex for ex in examples}
    print(f"[load] {len(predictions)} predictions from {predictions_path}; "
          f"{len(examples_by_id)} examples from {split_path}")

    res = score_corpus(args.task, predictions, examples_by_id)

    out_path = args.output or predictions_path.parent / "natural_plan_eval.json"
    payload = {
        "task": res.task,
        "predictions_file": str(predictions_path),
        "split_file": str(split_path),
        "n": res.n,
        "n_correct": res.n_correct,
        "n_parse_failures": res.n_parse_failures,
        "solve_rate": res.solve_rate,
        "by_difficulty": {
            str(d): {"correct": c, "total": t, "rate": (c / t if t else 0.0)}
            for d, (c, t) in sorted(res.by_difficulty.items())
        },
        "per_example": res.per_example,
        "caveats": CAVEATS,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    _print_summary(res, predictions=predictions_path, split=split_path)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
