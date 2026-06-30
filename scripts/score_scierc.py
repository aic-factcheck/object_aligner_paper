"""PL-Marker-style scoring of an OA / GEPA SciERC run.

Reads a JSONL prediction file (``holdout_scores.jsonl`` or ``scores.jsonl`` —
each row ``{id, score, response, feedback}`` where ``response`` is a
JSON-stringified ``{entities, relations}`` graph) and a raw SciERC split
JSON (one document per line, with the original token offsets in ``ner`` and
``relations``). Computes NER / Rel / Rel+ micro-F1 by recovering token spans
for predicted mentions and applying PL-Marker's set-comparison logic 1:1.

See ``research/opsu47_dataset_scierc_review.md`` for the methodology and
``src/object_aligner_exp/scierc_pl_marker.py`` for the implementation.

Usage::

    uv run python scripts/score_scierc.py \\
        --predictions data/runs/<run>/holdout_scores.jsonl \\
        --raw-split   data/scierc/raw/processed_data/json/test.json

Optional ``--manifest data/scierc/splits/.../manifest.json`` cross-checks
that the prediction ids match the source-split's recorded ``source_ids``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from object_aligner_exp.scierc_pl_marker import (
    CorpusResult,
    DocResult,
    score_corpus,
)


CAVEATS = [
    "Pred mention spans recovered by first-unused-occurrence "
    "case-insensitive token match against the raw doc's flat token list. "
    "Unresolved mentions are dropped from the pred NER/Rel/Rel+ sets and "
    "counted under n_pred_mentions_unresolved.",
    "OA emits cluster-level relations; we expand each relation to the "
    "Cartesian product of resolved (head_mention, tail_mention) span pairs "
    "before scoring. PL-Marker scores per-mention-pair end-to-end, so our "
    "Rel/Rel+ numbers are systematically lower than what a true PL-Marker "
    "RE model would obtain on the same predictions.",
    "Symmetric relations (COMPARE, CONJUNCTION) match in either argument "
    "order, mirroring PL-Marker's sym_labels handling. All other SciERC "
    "predicates are scored as directed.",
    "Numbers are not directly comparable to the PL-Marker SOTA in "
    "research/opsu47_dataset_scierc_review.md (69.9 / 53.2 / 41.6). They "
    "are PL-Marker-shaped but reflect OA's cluster-level output granularity "
    "and the LM's natural-text mention surface.",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
            doc_id = str(doc.get("doc_key", ""))
            if not doc_id:
                continue
            docs[doc_id] = doc
    return docs


def _cross_check_manifest(
    manifest_path: Path,
    raw_split_path: Path,
    prediction_ids: list[str],
) -> None:
    manifest = json.loads(manifest_path.read_text())
    split_filename = raw_split_path.name  # e.g. "test.json"
    split_stem = split_filename.removesuffix(".json")
    if split_stem not in {"train", "dev", "test"}:
        return
    block = manifest.get(
        "val" if split_stem == "dev" else split_stem
    ) or manifest.get("test")
    if not block:
        return
    expected_source_split = block.get("source_split")
    if expected_source_split and expected_source_split != split_stem:
        print(
            f"[score_scierc] warning: manifest's source_split for the chosen "
            f"block is {expected_source_split!r} but raw-split looks like "
            f"{split_stem!r}; ids may not line up."
        )
    src_ids = set(block.get("source_ids") or [])
    if not src_ids:
        return
    missing = [pid for pid in prediction_ids if pid not in src_ids]
    extra = [sid for sid in src_ids if sid not in set(prediction_ids)]
    if missing:
        print(
            f"[score_scierc] warning: {len(missing)} prediction ids not in "
            f"manifest source_ids (first 3: {missing[:3]})"
        )
    if extra:
        print(
            f"[score_scierc] info: {len(extra)} manifest source_ids missing "
            f"from predictions (first 3: {extra[:3]})"
        )


def _doc_to_jsonable(d: DocResult, *, include_unresolved: bool) -> dict[str, Any]:
    out = {
        "id": d.doc_id,
        "ner": d.ner.as_dict(),
        "rel": d.rel.as_dict(),
        "rel_plus": d.rel_plus.as_dict(),
        "n_pred_mentions_total": d.n_pred_mentions_total,
        "n_pred_mentions_resolved": d.n_pred_mentions_resolved,
        "n_pred_mentions_unresolved": (
            d.n_pred_mentions_total - d.n_pred_mentions_resolved
        ),
    }
    if include_unresolved:
        out["unresolved_mentions"] = d.unresolved_mentions
    return out


def _print_summary(
    result: CorpusResult,
    *,
    predictions_path: Path,
    raw_split_path: Path,
) -> None:
    n_unresolved = result.n_pred_mentions_unresolved
    n_total = result.n_pred_mentions_total
    pct = (100.0 * (n_total - n_unresolved) / n_total) if n_total else 0.0
    print()
    print("SciERC PL-Marker-style scoring")
    print(f"  Predictions: {predictions_path}  ({result.n_docs} docs)")
    print(f"  Gold:        {raw_split_path}")
    if result.n_parse_failures:
        print(
            f"  Parse failures: {result.n_parse_failures} "
            "(treated as empty preds → full FN)"
        )
    print()
    for name, m in (
        ("NER  F1", result.ner),
        ("Rel  F1", result.rel),
        ("Rel+ F1", result.rel_plus),
    ):
        print(
            f"  {name}: {m.f1:6.4f}   "
            f"P {m.p:.4f}  R {m.r:.4f}   "
            f"TP {m.tp}  FP {m.fp}  FN {m.fn}"
        )
    print()
    print(
        f"  Pred mentions resolved: {n_total - n_unresolved} / {n_total} "
        f"({pct:.1f}%)"
    )
    print()
    print("  NOTE: not directly comparable to PL-Marker SOTA on SciERC. "
          "See caveats in the JSON output / module docstring.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL file of OA predictions: rows {id, response, ...}.",
    )
    p.add_argument(
        "--raw-split",
        type=Path,
        required=True,
        help=(
            "Raw SciERC split, e.g. "
            "data/scierc/raw/processed_data/json/test.json — one doc per "
            "line with sentences/ner/relations/clusters."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write pl_marker_scores.json (default: alongside the "
            "predictions file)."
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional manifest.json (e.g. from "
            "data/scierc/splits/gepa_scierc/pilot/) to cross-check ids."
        ),
    )
    p.add_argument(
        "--include-unresolved",
        action="store_true",
        help="Include per-mention 'unresolved_mentions' lists in the JSON output.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-doc F1s in addition to the corpus summary.",
    )
    args = p.parse_args()

    predictions = _load_jsonl(args.predictions)
    raw_docs = _load_raw_split(args.raw_split)
    print(
        f"[load] {len(predictions)} predictions from {args.predictions}; "
        f"{len(raw_docs)} raw docs from {args.raw_split}"
    )

    if args.manifest is not None:
        _cross_check_manifest(
            args.manifest,
            args.raw_split,
            [str(row.get("id", "")) for row in predictions],
        )

    result = score_corpus(predictions, raw_docs)

    if args.verbose:
        print()
        for d in result.per_doc:
            print(
                f"  {d.doc_id}: ner_f1={d.ner.f1:.3f} "
                f"rel_f1={d.rel.f1:.3f} rel+_f1={d.rel_plus.f1:.3f} "
                f"unresolved={d.n_pred_mentions_total - d.n_pred_mentions_resolved}"
                f"/{d.n_pred_mentions_total}"
            )

    out_path = args.output or args.predictions.parent / "pl_marker_scores.json"
    payload = {
        "predictions_file": str(args.predictions),
        "raw_split_file": str(args.raw_split),
        "n_docs": result.n_docs,
        "n_parse_failures": result.n_parse_failures,
        "n_pred_mentions_total": result.n_pred_mentions_total,
        "n_pred_mentions_resolved": result.n_pred_mentions_resolved,
        "n_pred_mentions_unresolved": result.n_pred_mentions_unresolved,
        "summary": {
            "ner": result.ner.as_dict(),
            "rel": result.rel.as_dict(),
            "rel_plus": result.rel_plus.as_dict(),
        },
        "per_doc": [
            _doc_to_jsonable(d, include_unresolved=args.include_unresolved)
            for d in result.per_doc
        ],
        "caveats": CAVEATS,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    _print_summary(result, predictions_path=args.predictions, raw_split_path=args.raw_split)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
