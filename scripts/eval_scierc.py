"""Canonical-style SciERC evaluation of an OA / GEPA run (opus48).

Grades a run's held-out predictions against the SciERC test split under the
metric protocols the community reports, reconciled to our coref-merged output
shape. Reports, side-by-side:

  * NER F1          — mention-level span+type micro-F1 (PURE / PL-Marker /
                      DyGIE++ "Ent", computed 1:1).
  * Rel  / Rel+     — relation boundary / strict F1, at TWO granularities:
                        - mention-pair (canonical PURE/PL-Marker definition,
                          via Cartesian instantiation; biased low — see doc),
                        - entity-level (coref-aware cluster pairs; the fair
                          headline metric for a coref-merged extractor).
  * diagnostics     — type-agnostic boundary NER and boundary-relaxed NER.

Methodology, pseudocode, SOTA table, and caveats:
``research/opus48_scierc_eval.md``. Library: ``scierc_eval.score_corpus``.

Usage::

    # point at a run directory (uses holdout_scores.jsonl + the test split)
    uv run python scripts/eval_scierc.py \\
        --run-dir data/runs/gepa_scierc_native/oa_feedback/ra/qwen35-35b/s0_20260529_155735

    # or be explicit
    uv run python scripts/eval_scierc.py \\
        --predictions data/runs/<run>/holdout_scores.jsonl \\
        --raw-split   data/scierc/raw/processed_data/json/test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from object_aligner_exp.scierc_eval import CorpusEval, DocEval, Metrics, score_corpus


DEFAULT_RAW_TEST = Path("data/scierc/raw/processed_data/json/test.json")


CAVEATS = [
    "Predicted mention spans are recovered by first-unused-occurrence "
    "case-insensitive token match against the raw doc's flat token list "
    "(datasets/scierc.py tokenisation). Mentions that fail to anchor are "
    "dropped from every pred set and counted under n_pred_mentions_unresolved "
    "(~6-7% on the qwen35-35b run).",
    "NER / ner_boundary / ner_relaxed are mention-level and directly defined; "
    "`ner` (span+type exact) IS the PURE/PL-Marker/DyGIE++ 'Ent' metric. The "
    "gap vs SOTA is dominated by genuine boundary-convention disagreement "
    "(the model emits 'Multi-lingual Evaluation Task'; gold spans the whole "
    "'Multi-lingual Evaluation Task -LRB- MET -RRB-'), not by span recovery.",
    "rel_mentionpair / rel_plus_mentionpair use PURE/PL-Marker's exact "
    "definition but feed it Cartesian-expanded cluster relations; they are "
    "biased LOW relative to a true per-mention-pair RE model and are reported "
    "for definitional comparability only.",
    "rel_entity / rel_plus_entity are the headline coref-aware metric: gold "
    "mention-pair relations are projected to gold-cluster pairs, predicted "
    "entities are greedily aligned to gold clusters by shared resolved spans, "
    "and relations are matched at the cluster level. This removes the "
    "Cartesian penalty but is NOT identical to the leaderboard's mention-pair "
    "protocol — it is the entity-level RE convention used by generative-IE "
    "evaluations.",
    "Symmetric relations (COMPARE, CONJUNCTION) match in either argument "
    "order in all relation protocols.",
    "Coreference is not scored (mirrors PURE/PL-Marker).",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_raw_split(path: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            key = str(doc.get("doc_key", ""))
            if key:
                docs[key] = doc
    return docs


def _m(m: Metrics) -> dict[str, Any]:
    return m.as_dict()


def _doc_json(d: DocEval) -> dict[str, Any]:
    return {
        "id": d.doc_id,
        "ner": _m(d.ner),
        "ner_boundary": _m(d.ner_boundary),
        "ner_relaxed": _m(d.ner_relaxed),
        "rel_mentionpair": _m(d.rel_mentionpair),
        "rel_plus_mentionpair": _m(d.rel_plus_mentionpair),
        "rel_entity": _m(d.rel_entity),
        "rel_plus_entity": _m(d.rel_plus_entity),
        "n_pred_mentions_total": d.n_pred_mentions_total,
        "n_pred_mentions_resolved": d.n_pred_mentions_resolved,
        "n_pred_entities": d.n_pred_entities,
        "n_pred_entities_aligned": d.n_pred_entities_aligned,
    }


def _print_summary(
    res: CorpusEval, *, predictions: Path, raw_split: Path
) -> None:
    n_tot = res.n_pred_mentions_total
    n_res = res.n_pred_mentions_resolved
    pct = 100.0 * n_res / n_tot if n_tot else 0.0
    print()
    print("SciERC evaluation (opus48)")
    print(f"  Predictions: {predictions}  ({res.n_docs} docs)")
    print(f"  Gold:        {raw_split}")
    if res.n_parse_failures:
        print(f"  Parse failures: {res.n_parse_failures} (→ empty preds / full FN)")
    print()
    print("  ENTITIES (mention-level, micro-F1)")
    print("    metric          F1      P       R        TP    FP    FN")
    for name, m, tag in (
        ("ner (span+type)", res.ner, "← canonical 'Ent', 1:1"),
        ("ner_boundary", res.ner_boundary, "type-agnostic (diag)"),
        ("ner_relaxed", res.ner_relaxed, "boundary-relaxed (diag)"),
    ):
        print(
            f"    {name:<15} {m.f1:6.4f}  {m.p:6.4f}  {m.r:6.4f}   "
            f"{m.tp:5d} {m.fp:5d} {m.fn:5d}  {tag}"
        )
    print()
    print("  RELATIONS (micro-F1)")
    print("    metric                    F1      P       R        TP    FP    FN")
    for name, m, tag in (
        ("rel_entity", res.rel_entity, "← headline (coref-aware)"),
        ("rel_plus_entity", res.rel_plus_entity, "← headline strict"),
        ("rel_mentionpair", res.rel_mentionpair, "PURE/PLM def, biased low"),
        ("rel_plus_mentionpair", res.rel_plus_mentionpair, "PURE/PLM def, biased low"),
    ):
        print(
            f"    {name:<23} {m.f1:6.4f}  {m.p:6.4f}  {m.r:6.4f}   "
            f"{m.tp:5d} {m.fp:5d} {m.fn:5d}  {tag}"
        )
    print()
    print(
        f"  Pred mentions resolved: {n_res}/{n_tot} ({pct:.1f}%);  "
        f"entities aligned to gold clusters: "
        f"{res.n_pred_entities_aligned}/{res.n_pred_entities}"
    )
    print()
    print("  NOTE: `ner` is 1:1 with the leaderboard 'Ent' metric. Relation "
          "numbers are NOT directly comparable to mention-pair SOTA; see "
          "research/opus48_scierc_eval.md and the JSON 'caveats'.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Run directory; defaults --predictions to "
                        "<run-dir>/holdout_scores.jsonl and --raw-split to the "
                        "SciERC test split.")
    p.add_argument("--predictions", type=Path, default=None,
                   help="JSONL of OA predictions: rows {id, response, ...}.")
    p.add_argument("--raw-split", type=Path, default=None,
                   help=f"Raw SciERC split JSON (default {DEFAULT_RAW_TEST}).")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write scierc_eval.json (default: alongside "
                        "the predictions file).")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-doc F1s as well.")
    args = p.parse_args()

    predictions_path = args.predictions
    raw_split_path = args.raw_split
    if args.run_dir is not None:
        if predictions_path is None:
            predictions_path = args.run_dir / "holdout_scores.jsonl"
        if raw_split_path is None:
            raw_split_path = DEFAULT_RAW_TEST
    if raw_split_path is None:
        raw_split_path = DEFAULT_RAW_TEST
    if predictions_path is None:
        p.error("provide --run-dir or --predictions")

    predictions = _load_jsonl(predictions_path)
    raw_docs = _load_raw_split(raw_split_path)
    print(f"[load] {len(predictions)} predictions from {predictions_path}; "
          f"{len(raw_docs)} raw docs from {raw_split_path}")

    res = score_corpus(predictions, raw_docs)

    if args.verbose:
        print()
        for d in res.per_doc:
            print(f"  {d.doc_id}: ner={d.ner.f1:.3f} rel_e={d.rel_entity.f1:.3f} "
                  f"rel+_e={d.rel_plus_entity.f1:.3f} "
                  f"aligned={d.n_pred_entities_aligned}/{d.n_pred_entities}")

    out_path = args.output or predictions_path.parent / "scierc_eval.json"
    payload = {
        "predictions_file": str(predictions_path),
        "raw_split_file": str(raw_split_path),
        "n_docs": res.n_docs,
        "n_parse_failures": res.n_parse_failures,
        "n_pred_mentions_total": res.n_pred_mentions_total,
        "n_pred_mentions_resolved": res.n_pred_mentions_resolved,
        "n_pred_mentions_unresolved": res.n_pred_mentions_unresolved,
        "n_pred_entities": res.n_pred_entities,
        "n_pred_entities_aligned": res.n_pred_entities_aligned,
        "summary": {
            "ner": _m(res.ner),
            "ner_boundary": _m(res.ner_boundary),
            "ner_relaxed": _m(res.ner_relaxed),
            "rel_mentionpair": _m(res.rel_mentionpair),
            "rel_plus_mentionpair": _m(res.rel_plus_mentionpair),
            "rel_entity": _m(res.rel_entity),
            "rel_plus_entity": _m(res.rel_plus_entity),
        },
        "per_doc": [_doc_json(d) for d in res.per_doc],
        "caveats": CAVEATS,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    _print_summary(res, predictions=predictions_path, raw_split=raw_split_path)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
