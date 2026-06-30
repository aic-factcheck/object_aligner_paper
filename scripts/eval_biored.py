"""Coref-aware relation-F1 evaluation of an OA / GEPA BioRED run.

Grades a run's held-out predictions against the BioRED test split with an
end-to-end **relation F1** (entities produced by the model, aligned to gold
clusters by mention-name overlap). OA's ``biored`` score is the GEPA reward;
this F1 is the native task metric.

BioRED's gold is already in the same cluster-level ``{entities, relations}``
shape as DocRED, so the scorer is reused verbatim: ``docred_eval.score_corpus``
(predicted clusters aligned to gold by shared mention surface forms, then
relation tuples compared). Not directly comparable to the official BioRED
leaderboard (which assumes gold entities are given and keys relations by
normalized accession) — this is an honest end-to-end stand-in.

Usage::

    uv run python scripts/eval_biored.py --run-dir data/runs/gepa_biored/ra/.../s0_...

    uv run python scripts/eval_biored.py \\
        --predictions data/runs/<run>/holdout_scores.jsonl \\
        --split       data/biored/splits/gepa_biored/main/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.docred_eval import build_train_facts, score_corpus


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
    default_split = cfg.gepa_biored_main / "test.jsonl"
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Run directory; defaults --predictions to "
                        "<run-dir>/holdout_scores.jsonl.")
    p.add_argument("--predictions", type=Path, default=None,
                   help="JSONL of OA predictions: rows {id, response, ...}.")
    p.add_argument("--split", type=Path, default=None,
                   help=f"BioRED test split JSONL (default {default_split}).")
    p.add_argument("--train-split", type=Path, default=None,
                   help="Train split JSONL; if given, also report Ign-F1.")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write biored_eval.json (default: alongside predictions).")
    args = p.parse_args()

    predictions_path = args.predictions
    if args.run_dir is not None and predictions_path is None:
        predictions_path = args.run_dir / "holdout_scores.jsonl"
    if predictions_path is None:
        p.error("provide --run-dir or --predictions")

    # Prefer the split this run was actually scored on (config.json's test_path).
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

    train_facts = None
    if args.train_split is not None:
        train_facts = build_train_facts(_load_jsonl(args.train_split))
        print(f"[load] {len(train_facts)} train facts from {args.train_split} (for Ign-F1)")

    res = score_corpus(predictions, examples_by_id, train_facts=train_facts)

    out_path = args.output or predictions_path.parent / "biored_eval.json"
    payload = {
        "predictions_file": str(predictions_path),
        "split_file": str(split_path),
        "n": res.n,
        "n_parse_failures": res.n_parse_failures,
        "n_pred_entities": res.n_pred_entities,
        "n_pred_entities_aligned": res.n_pred_entities_aligned,
        "rel": {
            "tp": res.rel.tp, "fp": res.rel.fp, "fn": res.rel.fn,
            "precision": res.rel.precision, "recall": res.rel.recall, "f1": res.rel.f1,
        },
    }
    if res.rel_ign is not None:
        payload["rel_ign"] = {
            "tp": res.rel_ign.tp, "fp": res.rel_ign.fp, "fn": res.rel_ign.fn,
            "precision": res.rel_ign.precision, "recall": res.rel_ign.recall,
            "f1": res.rel_ign.f1,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print()
    print("BioRED coref-aware relation-F1 evaluation")
    print(f"  Predictions: {predictions_path}  ({res.n} examples)")
    print(f"  Gold:        {split_path}")
    if res.n_parse_failures:
        print(f"  Parse failures: {res.n_parse_failures} (→ empty preds / full FN)")
    print(f"  Rel    P={res.rel.precision:.4f}  R={res.rel.recall:.4f}  F1={res.rel.f1:.4f}")
    if res.rel_ign is not None:
        print(f"  RelIgn P={res.rel_ign.precision:.4f}  R={res.rel_ign.recall:.4f}  "
              f"F1={res.rel_ign.f1:.4f}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
