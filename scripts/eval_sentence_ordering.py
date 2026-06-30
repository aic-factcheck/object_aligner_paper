"""Official sentence-ordering evaluation (PMR + Kendall's τ) of an OA / GEPA run.

Grades a run's held-out predictions under the standard sentence-ordering
metrics — Perfect-Match Ratio (PMR, exact order) and Kendall's τ (rank
correlation) — so OA / GEPA numbers can be placed against published results.
This is distinct from the OA graded list-alignment score used as the GEPA reward
(see ``sentence_ordering_eval`` and the schemas under ``data/sentence_ordering/``).

Reads ``holdout_scores.jsonl`` (rows ``{id, response, ...}``) and the split file
the run was evaluated on (for the gold ``order`` + ``meta``). Default split is
``test.jsonl`` (200); use ``--split test_full.jsonl`` (1000) for the tight number.

Usage::

    uv run python scripts/eval_sentence_ordering.py --run-dir <run>
    uv run python scripts/eval_sentence_ordering.py \\
        --predictions <run>/holdout_scores.jsonl \\
        --split data/sentence_ordering/splits/gepa_rocstories/main/test_full.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets.rebel import load_jsonl
from object_aligner_exp.sentence_ordering_eval import TaskResult, score_corpus

CAVEATS = [
    "These are the standard sentence-ordering metrics (PMR = exact full-order "
    "match; Kendall's tau = rank correlation), NOT OA's graded list-alignment "
    "score (the GEPA reward). They measure different things by design.",
    "Output is an index permutation of 1..N, so OA's Hungarian schema scores "
    "~1.0 for ANY permutation; the fixed-vs-Hungarian OA gap is therefore pure "
    "order. PMR/tau here grade that order against the single reference.",
    "Scored against a single reference order; multiple coherent orders may exist "
    "(esp. abstracts), so PMR/tau slightly understate the model.",
    "Responses that don't parse to a valid permutation of 1..N count as parse "
    "failures (PMR 0, tau 0).",
    "For the tight number use --split test_full.jsonl; default test.jsonl (200) "
    "is the cheap subset.",
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
    print("Sentence-ordering evaluation (PMR + Kendall's tau)")
    print(f"  Predictions: {predictions}  ({res.n} examples)")
    print(f"  Split:       {split}")
    if res.n_parse_failures:
        print(f"  Parse failures: {res.n_parse_failures} (→ PMR 0, tau 0)")
    print()
    print(f"  PMR  (exact order):     {res.pmr_rate:.4f}  ({res.n_correct_pmr}/{res.n})")
    print(f"  Kendall's tau (mean):   {res.mean_tau:.4f}")
    print()
    print("  by N (sentence count):")
    for diff in sorted(res.by_difficulty):
        pmr_c, tau_s, total = res.by_difficulty[diff]
        pmr_r = pmr_c / total if total else 0.0
        tau_m = tau_s / total if total else 0.0
        print(f"    {diff:>3}: PMR {pmr_r:.4f}  tau {tau_m:+.4f}  (n={int(total)})")


def main() -> None:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Run dir; defaults --predictions to "
                        "<run-dir>/holdout_scores.jsonl.")
    p.add_argument("--predictions", type=Path, default=None,
                   help="JSONL of holdout predictions: rows {id, response, ...}.")
    p.add_argument("--split", type=Path, default=None,
                   help="Split JSONL with gold order (default: "
                        "gepa_rocstories/main/test.jsonl).")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write sentence_ordering_eval.json (default: "
                        "alongside the predictions file).")
    args = p.parse_args()

    predictions_path = args.predictions
    if args.run_dir is not None and predictions_path is None:
        predictions_path = args.run_dir / "holdout_scores.jsonl"
    if predictions_path is None:
        p.error("provide --run-dir or --predictions")

    split_path = args.split
    if split_path is None:
        split_path = cfg.gepa_rocstories_main / "test.jsonl"

    predictions = _load_jsonl(predictions_path)
    examples = load_jsonl(split_path)
    examples_by_id = {str(ex["id"]): ex for ex in examples}
    print(f"[load] {len(predictions)} predictions from {predictions_path}; "
          f"{len(examples_by_id)} examples from {split_path}")

    res = score_corpus(predictions, examples_by_id)

    out_path = args.output or predictions_path.parent / "sentence_ordering_eval.json"
    payload = {
        "task": res.task,
        "predictions_file": str(predictions_path),
        "split_file": str(split_path),
        "n": res.n,
        "n_correct_pmr": res.n_correct_pmr,
        "pmr_rate": res.pmr_rate,
        "mean_tau": res.mean_tau,
        "n_parse_failures": res.n_parse_failures,
        "by_difficulty": {
            str(d): {
                "pmr": (c / t if t else 0.0),
                "tau": (s / t if t else 0.0),
                "n": int(t),
            }
            for d, (c, s, t) in sorted(res.by_difficulty.items())
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
