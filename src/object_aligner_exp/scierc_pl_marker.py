"""PL-Marker-style scorer for OA SciERC predictions.

Implements the NER F1, relation boundary F1 (``Rel``), and relation strict F1
(``Rel+``) defined by Zhong & Chen 2021 ("A Frustratingly Easy Approach for
Entity and Relation Extraction", PURE) and reused by Ye et al. 2022
("Packed Levitated Marker for Entity and Relation Extraction", PL-Marker);
see ``research/opsu47_dataset_scierc_review.md`` §2.

The OA-aligned predictions and pilot gold drop token offsets (see
``datasets/scierc.py:38-39``), so we

1. take the **raw** SciERC JSON for the gold (it ships flat doc-wide
   inclusive token offsets in ``ner`` / ``relations``);
2. recover spans on the **pred** side by first-unused-occurrence,
   case-insensitive token-match of each predicted ``mention.text`` against
   the document's flat token sequence (the same sequence as
   ``datasets.scierc._flat_tokens``);
3. then port PL-Marker's set-comparison logic 1:1
   (``run_acener.py:599-645``, ``run_re.py:700-800`` at
   https://github.com/thunlp/PL-Marker).

Known biases vs a true PL-Marker run on a per-mention-pair RE model:

* Span recovery is best-effort — LM-paraphrased mentions that don't match a
  token sub-range in the source are dropped from the pred sets and reported
  under ``n_pred_mentions_unresolved``.
* OA emits cluster-level relations; we expand them cartesianly to all
  resolved (head_mention, tail_mention) pairs. A cluster with 3 head spans
  × 2 tail spans becomes 6 pred relation tuples. This systematically biases
  Rel / Rel+ vs PL-Marker.

So: numbers from this scorer are PL-Marker-shaped but **not** directly
comparable to the SOTA table in the review doc.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


# SciERC relations whose argument order is undirected. Mirrors
# PL-Marker's ``sym_labels`` for SciERC (see run_re.py near the
# ``sym_labels`` definition).
SYMMETRIC_RELATIONS: frozenset[str] = frozenset({"COMPARE", "CONJUNCTION"})


# --- data shapes -----------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """Micro P/R/F1 with the underlying TP/FP/FN counts."""

    tp: int
    fp: int
    fn: int

    @property
    def p(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def r(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.p + self.r
        return 2 * self.p * self.r / denom if denom else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "p": self.p,
            "r": self.r,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class DocResult:
    """Per-document scorer output."""

    doc_id: str
    ner: Metrics
    rel: Metrics
    rel_plus: Metrics
    n_pred_mentions_total: int
    n_pred_mentions_resolved: int
    unresolved_mentions: list[dict[str, str]] = field(default_factory=list)


@dataclass
class CorpusResult:
    """Aggregate scorer output across documents."""

    ner: Metrics
    rel: Metrics
    rel_plus: Metrics
    n_docs: int
    n_parse_failures: int
    n_pred_mentions_total: int
    n_pred_mentions_resolved: int
    per_doc: list[DocResult]

    @property
    def n_pred_mentions_unresolved(self) -> int:
        return self.n_pred_mentions_total - self.n_pred_mentions_resolved


# --- gold side -------------------------------------------------------------


def flat_tokens(sentences: list[list[str]]) -> list[str]:
    """Flatten per-sentence tokens — must match ``datasets.scierc._flat_tokens``."""
    return [t for s in sentences for t in s]


def gold_ner_tuples(
    doc_id: str, doc: dict[str, Any]
) -> set[tuple[str, int, int, str]]:
    """All NER tuples from a raw SciERC doc, keyed by flat doc-wide offsets."""
    out: set[tuple[str, int, int, str]] = set()
    for sent_ner in doc.get("ner", []):
        for entry in sent_ner:
            s, e, t = int(entry[0]), int(entry[1]), str(entry[2])
            out.add((doc_id, s, e, t))
    return out


def gold_relation_tuples(
    doc_id: str, doc: dict[str, Any]
) -> tuple[
    set[tuple[str, int, int, int, int, str]],
    set[tuple[str, int, int, str, int, int, str, str]],
]:
    """Return (boundary, strict) gold relation tuple sets for one raw doc."""
    span_to_type: dict[tuple[int, int], str] = {}
    for sent_ner in doc.get("ner", []):
        for entry in sent_ner:
            span_to_type[(int(entry[0]), int(entry[1]))] = str(entry[2])

    rel_set: set[tuple[str, int, int, int, int, str]] = set()
    rel_plus_set: set[tuple[str, int, int, str, int, int, str, str]] = set()
    for sent_rel in doc.get("relations", []):
        for entry in sent_rel:
            hs, he, ts, te, label = (
                int(entry[0]),
                int(entry[1]),
                int(entry[2]),
                int(entry[3]),
                str(entry[4]),
            )
            rel_set.add((doc_id, hs, he, ts, te, label))
            h_type = span_to_type.get((hs, he))
            t_type = span_to_type.get((ts, te))
            if h_type is None or t_type is None:
                # SciERC's invariant guarantees both endpoints are NER-tagged;
                # be defensive in case of dataset noise.
                continue
            rel_plus_set.add((doc_id, hs, he, h_type, ts, te, t_type, label))
    return rel_set, rel_plus_set


# --- pred side: span recovery ---------------------------------------------


def _norm_token(t: str) -> str:
    return t.lower()


def _find_span(
    pred_tokens: list[str],
    flat: list[str],
    flat_lower: list[str],
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """First unused inclusive (start, end) where ``flat[start:end+1]`` matches ``pred_tokens`` (case-insensitive)."""
    n = len(pred_tokens)
    if n == 0:
        return None
    needle = [_norm_token(t) for t in pred_tokens]
    last_start = len(flat) - n
    head = needle[0]
    for start in range(0, last_start + 1):
        if flat_lower[start] != head:
            continue
        end = start + n - 1
        if (start, end) in used:
            continue
        if flat_lower[start : end + 1] == needle:
            return (start, end)
    return None


def resolve_pred_entities(
    pred_graph: dict[str, Any],
    sentences: list[list[str]],
) -> tuple[
    dict[str, list[tuple[int, int]]],
    dict[str, str],
    int,
    list[dict[str, str]],
]:
    """Resolve OA pred entities to token spans against the doc's flat tokens.

    Returns:
        spans_by_eid: maps entity id → list of (start, end) inclusive spans,
            one per resolvable mention.
        type_by_eid: maps entity id → predicted entity type (string, unmodified).
        n_total_mentions: total mention texts attempted across all entities.
        unresolved: list of {entity_id, text} entries that could not be anchored.
    """
    flat = flat_tokens(sentences)
    flat_lower = [_norm_token(t) for t in flat]
    used: set[tuple[int, int]] = set()

    # Process mentions longest-first within each entity, so a short mention
    # doesn't steal a token sub-range needed by a longer one in the same
    # cluster. Across entities we keep predicted-entity order so the result is
    # deterministic.
    spans_by_eid: dict[str, list[tuple[int, int]]] = {}
    type_by_eid: dict[str, str] = {}
    unresolved: list[dict[str, str]] = []
    n_total = 0

    entities = pred_graph.get("entities") or []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        eid = str(ent.get("id", ""))
        if not eid:
            continue
        etype = ent.get("type")
        type_by_eid[eid] = str(etype) if etype is not None else ""

        mentions_raw = ent.get("mentions") or []
        # Pair (original_index, text_token_list) so we can order longest-first
        # but emit in mention-list order? Actually it doesn't matter for the
        # tuple sets which are sets; longest-first is purely a heuristic to
        # raise hit rate. Resolve in (-len, original_index) order.
        items: list[tuple[int, int, list[str]]] = []
        for idx, m in enumerate(mentions_raw):
            if not isinstance(m, dict):
                continue
            text = m.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            toks = text.split()
            if not toks:
                continue
            items.append((-len(toks), idx, toks))
        items.sort()
        n_total += len(items)

        spans: list[tuple[int, int]] = []
        for _, _, toks in items:
            span = _find_span(toks, flat, flat_lower, used)
            if span is None:
                unresolved.append({"entity_id": eid, "text": " ".join(toks)})
                continue
            used.add(span)
            spans.append(span)
        spans_by_eid[eid] = spans

    return spans_by_eid, type_by_eid, n_total, unresolved


# --- pred side: tuple sets ------------------------------------------------


def pred_ner_tuples(
    doc_id: str,
    spans_by_eid: dict[str, list[tuple[int, int]]],
    type_by_eid: dict[str, str],
) -> set[tuple[str, int, int, str]]:
    """Build the pred NER tuple set for one doc from resolved spans."""
    out: set[tuple[str, int, int, str]] = set()
    for eid, spans in spans_by_eid.items():
        ent_type = type_by_eid.get(eid, "")
        for s, e in spans:
            out.add((doc_id, s, e, ent_type))
    return out


def pred_relation_tuples(
    doc_id: str,
    pred_graph: dict[str, Any],
    spans_by_eid: dict[str, list[tuple[int, int]]],
    type_by_eid: dict[str, str],
) -> tuple[
    set[tuple[str, int, int, int, int, str]],
    set[tuple[str, int, int, str, int, int, str, str]],
]:
    """Cartesian-expand each OA relation to per-mention-pair tuples.

    Returns (boundary, strict) sets. Both sets canonicalise symmetric
    relations by sorting (head, tail) span pairs so the swap-check in
    :func:`_match_set` only ever has to look for one direction.
    """
    rel_set: set[tuple[str, int, int, int, int, str]] = set()
    rel_plus_set: set[tuple[str, int, int, str, int, int, str, str]] = set()

    relations = pred_graph.get("relations") or []
    for r in relations:
        if not isinstance(r, dict):
            continue
        subj = str(r.get("subject", ""))
        obj = str(r.get("object", ""))
        label = str(r.get("predicate", ""))
        if not subj or not obj or not label:
            continue
        h_type = type_by_eid.get(subj, "")
        t_type = type_by_eid.get(obj, "")
        for hs, he in spans_by_eid.get(subj, []):
            for ts, te in spans_by_eid.get(obj, []):
                if (hs, he) == (ts, te):
                    # Self-relation between identical spans is degenerate;
                    # PL-Marker filters these naturally (its decoder never
                    # emits them). Skip on the pred side as well.
                    continue
                rel_set.add((doc_id, hs, he, ts, te, label))
                rel_plus_set.add(
                    (doc_id, hs, he, h_type, ts, te, t_type, label)
                )
    return rel_set, rel_plus_set


# --- matching with symmetric-relation swap --------------------------------


def _swap_rel(
    t: tuple[str, int, int, int, int, str],
) -> tuple[str, int, int, int, int, str]:
    doc_id, hs, he, ts, te, label = t
    return (doc_id, ts, te, hs, he, label)


def _swap_rel_plus(
    t: tuple[str, int, int, str, int, int, str, str],
) -> tuple[str, int, int, str, int, int, str, str]:
    doc_id, hs, he, h_type, ts, te, t_type, label = t
    return (doc_id, ts, te, t_type, hs, he, h_type, label)


def _count_matches_rel(
    pred: Iterable[tuple[str, int, int, int, int, str]],
    gold: set[tuple[str, int, int, int, int, str]],
) -> int:
    cor = 0
    for t in pred:
        if t in gold:
            cor += 1
            continue
        if t[5] in SYMMETRIC_RELATIONS and _swap_rel(t) in gold:
            cor += 1
    return cor


def _count_matches_rel_plus(
    pred: Iterable[tuple[str, int, int, str, int, int, str, str]],
    gold: set[tuple[str, int, int, str, int, int, str, str]],
) -> int:
    cor = 0
    for t in pred:
        if t in gold:
            cor += 1
            continue
        if t[7] in SYMMETRIC_RELATIONS and _swap_rel_plus(t) in gold:
            cor += 1
    return cor


def _exact_metrics(
    pred: set, gold: set
) -> Metrics:
    tp = len(pred & gold)
    return Metrics(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


def _rel_metrics_sym(
    pred: set[tuple[str, int, int, int, int, str]],
    gold: set[tuple[str, int, int, int, int, str]],
) -> Metrics:
    tp = _count_matches_rel(pred, gold)
    # Also recompute recall numerator: a gold tuple is matched if either it
    # or its swap appears in pred (for symmetric labels). We invert to count
    # matched-gold-tuples directly so TP, FN, FP stay consistent.
    matched_gold = 0
    for t in gold:
        if t in pred:
            matched_gold += 1
            continue
        if t[5] in SYMMETRIC_RELATIONS and _swap_rel(t) in pred:
            matched_gold += 1
    # In a well-formed prediction (no duplicate (m1, m2, label) entries on
    # either side) tp == matched_gold. Take the conservative min to keep P/R
    # symmetric and bounded.
    tp = min(tp, matched_gold)
    return Metrics(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


def _rel_plus_metrics_sym(
    pred: set[tuple[str, int, int, str, int, int, str, str]],
    gold: set[tuple[str, int, int, str, int, int, str, str]],
) -> Metrics:
    tp = _count_matches_rel_plus(pred, gold)
    matched_gold = 0
    for t in gold:
        if t in pred:
            matched_gold += 1
            continue
        if t[7] in SYMMETRIC_RELATIONS and _swap_rel_plus(t) in pred:
            matched_gold += 1
    tp = min(tp, matched_gold)
    return Metrics(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


# --- top-level scoring -----------------------------------------------------


def score_one_doc(
    doc_id: str,
    raw_doc: dict[str, Any],
    pred_graph: dict[str, Any],
) -> DocResult:
    """Compute per-doc NER / Rel / Rel+ metrics against a raw SciERC doc."""
    sentences = raw_doc.get("sentences") or []
    spans_by_eid, type_by_eid, n_total, unresolved = resolve_pred_entities(
        pred_graph, sentences
    )
    pred_ner = pred_ner_tuples(doc_id, spans_by_eid, type_by_eid)
    pred_rel, pred_rel_plus = pred_relation_tuples(
        doc_id, pred_graph, spans_by_eid, type_by_eid
    )
    gold_ner = gold_ner_tuples(doc_id, raw_doc)
    gold_rel, gold_rel_plus = gold_relation_tuples(doc_id, raw_doc)

    ner_m = _exact_metrics(pred_ner, gold_ner)
    rel_m = _rel_metrics_sym(pred_rel, gold_rel)
    rel_plus_m = _rel_plus_metrics_sym(pred_rel_plus, gold_rel_plus)

    return DocResult(
        doc_id=doc_id,
        ner=ner_m,
        rel=rel_m,
        rel_plus=rel_plus_m,
        n_pred_mentions_total=n_total,
        n_pred_mentions_resolved=n_total - len(unresolved),
        unresolved_mentions=unresolved,
    )


def score_corpus(
    predictions: list[dict[str, Any]],
    raw_docs_by_id: dict[str, dict[str, Any]],
) -> CorpusResult:
    """Score every prediction row against the matching raw SciERC doc.

    ``predictions`` is an iterable of OA-style rows
    ``{"id": ..., "response": "<json-string>" or {...}}``. The ``response``
    field is parsed (json.loads) if it's a string. Rows whose ``id`` is
    missing from ``raw_docs_by_id`` are skipped with a warning; rows whose
    ``response`` doesn't parse become full-FN docs (counted under
    ``n_parse_failures``).
    """
    per_doc: list[DocResult] = []
    n_parse_failures = 0

    # Aggregate TP/FP/FN by summing the per-doc counts. The PL-Marker scorer
    # is micro-averaged the same way (set union over all docs).
    ner_tp = ner_fp = ner_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    rp_tp = rp_fp = rp_fn = 0
    n_total = n_resolved = 0

    for row in predictions:
        doc_id = str(row.get("id", ""))
        raw_doc = raw_docs_by_id.get(doc_id)
        if raw_doc is None:
            print(f"[score_scierc] skipping prediction {doc_id!r}: not in raw split")
            continue

        resp = row.get("response")
        if isinstance(resp, str):
            try:
                pred_graph = json.loads(resp)
            except json.JSONDecodeError:
                pred_graph = None
        elif isinstance(resp, dict):
            pred_graph = resp
        else:
            pred_graph = None

        if not isinstance(pred_graph, dict):
            # Full-FN doc: empty pred sets vs full gold sets.
            n_parse_failures += 1
            pred_graph = {"entities": [], "relations": []}

        doc_result = score_one_doc(doc_id, raw_doc, pred_graph)
        per_doc.append(doc_result)

        ner_tp += doc_result.ner.tp
        ner_fp += doc_result.ner.fp
        ner_fn += doc_result.ner.fn
        rel_tp += doc_result.rel.tp
        rel_fp += doc_result.rel.fp
        rel_fn += doc_result.rel.fn
        rp_tp += doc_result.rel_plus.tp
        rp_fp += doc_result.rel_plus.fp
        rp_fn += doc_result.rel_plus.fn
        n_total += doc_result.n_pred_mentions_total
        n_resolved += doc_result.n_pred_mentions_resolved

    return CorpusResult(
        ner=Metrics(tp=ner_tp, fp=ner_fp, fn=ner_fn),
        rel=Metrics(tp=rel_tp, fp=rel_fp, fn=rel_fn),
        rel_plus=Metrics(tp=rp_tp, fp=rp_fp, fn=rp_fn),
        n_docs=len(per_doc),
        n_parse_failures=n_parse_failures,
        n_pred_mentions_total=n_total,
        n_pred_mentions_resolved=n_resolved,
        per_doc=per_doc,
    )


__all__ = [
    "SYMMETRIC_RELATIONS",
    "CorpusResult",
    "DocResult",
    "Metrics",
    "flat_tokens",
    "gold_ner_tuples",
    "gold_relation_tuples",
    "pred_ner_tuples",
    "pred_relation_tuples",
    "resolve_pred_entities",
    "score_corpus",
    "score_one_doc",
]
