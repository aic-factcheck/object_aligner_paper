"""SciERC evaluation for coreference-cluster-level (generative) extractors.

This is the *opus48* successor to :mod:`object_aligner_exp.scierc_pl_marker`.
Same span-recovery front-end, but it computes **several scoring protocols
side-by-side** so an OA / GEPA run can be placed against the published
SciERC literature as fairly as the cluster-level output shape allows.

Methodology, pseudocode, and the SOTA reference table live in
``research/opus48_scierc_eval.md``. The short version:

Our task LM emits a *coreference-merged* graph — one entity object per
cluster (its surface mentions in ``mentions[]``) and one relation per
*cluster pair*. SciERC gold, however, annotates NER and relations at the
**mention-span** level (inclusive flat token offsets) and does *not*
propagate relations across coreference. Two consequences shape this scorer:

1. **Spans must be recovered.** The OA-aligned wire format drops offsets
   (``datasets/scierc.py``), so we re-anchor each predicted ``mention.text``
   to a token span via :func:`scierc_pl_marker.resolve_pred_entities`
   (first-unused-occurrence, case-insensitive token match).

2. **Granularity must be reconciled.** Scoring cluster-level relations
   against mention-pair gold by Cartesian expansion (what
   ``scierc_pl_marker`` does) inflates false positives and is *not* a fair
   reflection of a coref-merged model. So the headline metric here is an
   **entity-level (coref-aware)** relation F1: gold mention-pair relations
   are projected up to *gold cluster* pairs (exactly mirroring the
   preprocessing in ``datasets/scierc.py``), predicted clusters are aligned
   to gold clusters by shared resolved spans, and relations are compared at
   the cluster level.

Protocols reported (all micro-averaged over the corpus, TP/FP/FN summed):

* ``ner``              — canonical mention-level NER: tuple
                         ``(doc, start, end, type)``, exact span + exact type.
                         **This is the PURE / PL-Marker / DyGIE++ "Ent" metric, 1:1.**
* ``ner_boundary``     — same but type-agnostic ``(doc, start, end)`` (diagnostic).
* ``ner_relaxed``      — type-sensitive, boundary-*relaxed*: a predicted span
                         matches a gold span of the same type if they overlap
                         in tokens (greedy 1:1). Diagnostic for the
                         boundary-disagreement penalty LLM extractors pay.
* ``rel_mentionpair``  / ``rel_plus_mentionpair`` — canonical PURE/PL-Marker
                         ``Rel`` / ``Rel+`` via Cartesian instantiation of
                         cluster relations to mention pairs. Directly
                         comparable in *definition* to the leaderboard but
                         biased low by the granularity mismatch — see the doc.
* ``rel_entity``       / ``rel_plus_entity``      — **headline** coref-aware
                         entity-level ``Rel`` / ``Rel+`` (cluster pairs).

Symmetric relations (``COMPARE``, ``CONJUNCTION``) match in either argument
order in every relation protocol, mirroring PL-Marker's ``sym_labels``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from object_aligner_exp.scierc_pl_marker import (
    SYMMETRIC_RELATIONS,
    Metrics,
    flat_tokens,
    gold_ner_tuples,
    resolve_pred_entities,
    score_one_doc,
)

__all__ = [
    "SYMMETRIC_RELATIONS",
    "CorpusEval",
    "DocEval",
    "Metrics",
    "build_gold_clusters",
    "score_corpus",
    "score_one_doc_eval",
]


# --- gold coreference clusters --------------------------------------------
#
# Mirror datasets/scierc.py:parse_scierc_doc exactly: every `clusters` entry
# is one cluster; NER spans not in any cluster become singletons; cluster
# type is the majority vote over its mentions' NER types.


def build_gold_clusters(
    doc: dict[str, Any],
) -> tuple[list[tuple[frozenset[tuple[int, int]], str]], dict[tuple[int, int], int]]:
    """Return ``(clusters, span_to_cluster_idx)`` for one raw SciERC doc.

    ``clusters[i] = (frozenset_of_spans, majority_type)``. ``span_to_cluster``
    maps every NER-tagged span to its cluster index.
    """
    span_to_type: dict[tuple[int, int], str] = {}
    for sent_ner in doc.get("ner", []):
        for entry in sent_ner:
            span_to_type[(int(entry[0]), int(entry[1]))] = str(entry[2])

    clusters: list[tuple[frozenset[tuple[int, int]], str]] = []
    span_to_cluster: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int]] = set()

    for cluster in doc.get("clusters", []) or []:
        spans = [(int(s), int(e)) for s, e in cluster]
        seen.update(spans)
        types = [span_to_type[s] for s in spans if s in span_to_type]
        if not types:
            # Cluster spans not NER-tagged — SciERC's invariant precludes this.
            continue
        etype = Counter(types).most_common(1)[0][0]
        idx = len(clusters)
        for s in spans:
            if s in span_to_type:
                span_to_cluster[s] = idx
        clusters.append((frozenset(s for s in spans if s in span_to_type), etype))

    for span, etype in span_to_type.items():
        if span not in seen:
            idx = len(clusters)
            span_to_cluster[span] = idx
            clusters.append((frozenset({span}), etype))

    return clusters, span_to_cluster


# --- gold relation tuple sets ---------------------------------------------


def _gold_entity_relations(
    doc: dict[str, Any],
    clusters: list[tuple[frozenset[tuple[int, int]], str]],
    span_to_cluster: dict[tuple[int, int], int],
) -> tuple[set[tuple[int, str, int]], set[tuple[int, str, str, str, int]]]:
    """Project gold mention-pair relations to gold-cluster pairs.

    Returns ``(boundary_set, strict_set)``:

    * boundary tuple = ``(head_cluster, label, tail_cluster)``
    * strict tuple   = ``(head_cluster, head_type, label, tail_type, tail_cluster)``

    Both deduped — SciERC sometimes annotates the same cluster-level relation
    through several mention pairs.
    """
    boundary: set[tuple[int, str, int]] = set()
    strict: set[tuple[int, str, str, str, int]] = set()
    for sent_rel in doc.get("relations", []):
        for entry in sent_rel:
            h = (int(entry[0]), int(entry[1]))
            t = (int(entry[2]), int(entry[3]))
            label = str(entry[4])
            if h not in span_to_cluster or t not in span_to_cluster:
                continue
            hi, ti = span_to_cluster[h], span_to_cluster[t]
            boundary.add((hi, label, ti))
            strict.add((hi, clusters[hi][1], label, clusters[ti][1], ti))
    return boundary, strict


# --- pred → gold cluster alignment ----------------------------------------


def align_pred_to_gold(
    spans_by_eid: dict[str, list[tuple[int, int]]],
    span_to_cluster: dict[tuple[int, int], int],
) -> dict[str, int]:
    """Greedy one-to-one alignment of predicted entities to gold clusters.

    A predicted entity is aligned to the gold cluster with which it shares
    the most *resolved* spans. Ties are broken deterministically (higher
    overlap, then lower gold-cluster index, then entity id). Each gold
    cluster is claimed at most once; unaligned predicted entities are simply
    absent from the returned map (their relations become false positives).
    """
    candidates: list[tuple[int, int, str]] = []  # (-overlap, gold_idx, eid)
    for eid, spans in spans_by_eid.items():
        counts: dict[int, int] = {}
        for sp in spans:
            gi = span_to_cluster.get(sp)
            if gi is not None:
                counts[gi] = counts.get(gi, 0) + 1
        for gi, c in counts.items():
            candidates.append((-c, gi, eid))
    candidates.sort()

    alignment: dict[str, int] = {}
    used_gold: set[int] = set()
    for _, gi, eid in candidates:
        if eid in alignment or gi in used_gold:
            continue
        alignment[eid] = gi
        used_gold.add(gi)
    return alignment


def _pred_entity_relations(
    pred_graph: dict[str, Any],
    alignment: dict[str, int],
    type_by_eid: dict[str, str],
) -> tuple[set[tuple[Any, str, Any]], set[tuple[Any, str, str, str, Any]]]:
    """Build predicted entity-level relation tuple sets via the alignment.

    Unaligned predicted entities get a sentinel cluster key ``("U", eid)`` so
    their relations can never match a gold tuple (they count as FP) while
    still being deduped per predicted cluster pair.
    """
    boundary: set[tuple[Any, str, Any]] = set()
    strict: set[tuple[Any, str, str, str, Any]] = set()
    for r in pred_graph.get("relations") or []:
        if not isinstance(r, dict):
            continue
        subj = str(r.get("subject", ""))
        obj = str(r.get("object", ""))
        label = str(r.get("predicate", ""))
        if not subj or not obj or not label:
            continue
        hi: Any = alignment.get(subj, ("U", subj))
        ti: Any = alignment.get(obj, ("U", obj))
        if hi == ti:
            # Degenerate self relation on a single cluster; SciERC never has
            # these. Skip on the pred side too.
            continue
        boundary.add((hi, label, ti))
        strict.add((hi, type_by_eid.get(subj, ""), label, type_by_eid.get(obj, ""), ti))
    return boundary, strict


# --- symmetric-relation aware set matching --------------------------------


def _match_count_boundary(pred: set, gold: set) -> int:
    """TP count for boundary tuples ``(h, label, t)`` with symmetric labels."""
    tp = 0
    for h, label, t in pred:
        if (h, label, t) in gold or (
            label in SYMMETRIC_RELATIONS and (t, label, h) in gold
        ):
            tp += 1
    return tp


def _match_count_strict(pred: set, gold: set) -> int:
    """TP count for strict tuples ``(h, ht, label, tt, t)`` with symmetric labels."""
    tp = 0
    for h, ht, label, tt, t in pred:
        if (h, ht, label, tt, t) in gold or (
            label in SYMMETRIC_RELATIONS and (t, tt, label, ht, h) in gold
        ):
            tp += 1
    return tp


def _metrics_boundary(pred: set, gold: set) -> Metrics:
    tp = _match_count_boundary(pred, gold)
    # Recount from the gold side to keep TP consistent under symmetry.
    matched_gold = 0
    for h, label, t in gold:
        if (h, label, t) in pred or (
            label in SYMMETRIC_RELATIONS and (t, label, h) in pred
        ):
            matched_gold += 1
    tp = min(tp, matched_gold)
    return Metrics(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


def _metrics_strict(pred: set, gold: set) -> Metrics:
    tp = _match_count_strict(pred, gold)
    matched_gold = 0
    for h, ht, label, tt, t in gold:
        if (h, ht, label, tt, t) in pred or (
            label in SYMMETRIC_RELATIONS and (t, tt, label, ht, h) in pred
        ):
            matched_gold += 1
    tp = min(tp, matched_gold)
    return Metrics(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


# --- relaxed (boundary-overlap) NER ---------------------------------------


def _ner_relaxed_metrics(
    pred_spans_types: list[tuple[int, int, str]],
    gold_spans_types: list[tuple[int, int, str]],
) -> Metrics:
    """Type-sensitive, boundary-relaxed NER via greedy 1:1 token-overlap match.

    A predicted ``(s, e, type)`` matches an unused gold ``(s', e', type)`` of
    the *same type* when their inclusive token ranges overlap. Greedy by
    descending overlap keeps it deterministic and one-to-one. Non-canonical —
    a diagnostic for the boundary-disagreement penalty.
    """
    pairs: list[tuple[int, int, int]] = []  # (-overlap, pred_idx, gold_idx)
    for pi, (ps, pe, pt) in enumerate(pred_spans_types):
        for gi, (gs, ge, gt) in enumerate(gold_spans_types):
            if pt != gt:
                continue
            ov = min(pe, ge) - max(ps, gs) + 1
            if ov > 0:
                pairs.append((-ov, pi, gi))
    pairs.sort()
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    tp = 0
    for _, pi, gi in pairs:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        tp += 1
    return Metrics(
        tp=tp,
        fp=len(pred_spans_types) - tp,
        fn=len(gold_spans_types) - tp,
    )


# --- per-doc / per-corpus drivers -----------------------------------------


@dataclass
class DocEval:
    """Per-document metrics across all protocols."""

    doc_id: str
    ner: Metrics
    ner_boundary: Metrics
    ner_relaxed: Metrics
    rel_mentionpair: Metrics
    rel_plus_mentionpair: Metrics
    rel_entity: Metrics
    rel_plus_entity: Metrics
    n_pred_mentions_total: int
    n_pred_mentions_resolved: int
    n_pred_entities: int
    n_pred_entities_aligned: int


@dataclass
class CorpusEval:
    """Aggregate metrics (micro) across documents."""

    ner: Metrics
    ner_boundary: Metrics
    ner_relaxed: Metrics
    rel_mentionpair: Metrics
    rel_plus_mentionpair: Metrics
    rel_entity: Metrics
    rel_plus_entity: Metrics
    n_docs: int
    n_parse_failures: int
    n_pred_mentions_total: int
    n_pred_mentions_resolved: int
    n_pred_entities: int
    n_pred_entities_aligned: int
    per_doc: list[DocEval] = field(default_factory=list)

    @property
    def n_pred_mentions_unresolved(self) -> int:
        return self.n_pred_mentions_total - self.n_pred_mentions_resolved


def score_one_doc_eval(
    doc_id: str,
    raw_doc: dict[str, Any],
    pred_graph: dict[str, Any],
) -> DocEval:
    """Compute every protocol's metrics for one (raw_doc, prediction) pair."""
    sentences = raw_doc.get("sentences") or []

    # Mention-level NER + canonical mention-pair Rel/Rel+ come straight from
    # the opus47 scorer (its Cartesian expansion is what defines the
    # mention-pair protocol). We re-resolve spans here for the entity-level
    # protocol and the relaxed NER diagnostic.
    pm = score_one_doc(doc_id, raw_doc, pred_graph)

    spans_by_eid, type_by_eid, _, _ = resolve_pred_entities(pred_graph, sentences)
    clusters, span_to_cluster = build_gold_clusters(raw_doc)

    # Relaxed NER inputs.
    pred_spans_types = [
        (s, e, type_by_eid.get(eid, ""))
        for eid, spans in spans_by_eid.items()
        for (s, e) in spans
    ]
    gold_spans_types = [
        (s, e, t) for (_, s, e, t) in gold_ner_tuples(doc_id, raw_doc)
    ]
    ner_relaxed = _ner_relaxed_metrics(pred_spans_types, gold_spans_types)

    # Type-agnostic boundary NER.
    pred_b = {(s, e) for (s, e, _t) in pred_spans_types}
    gold_b = {(s, e) for (s, e, _t) in gold_spans_types}
    ner_boundary = Metrics(
        tp=len(pred_b & gold_b),
        fp=len(pred_b - gold_b),
        fn=len(gold_b - pred_b),
    )

    # Entity-level (coref-aware) relations.
    gold_e_b, gold_e_s = _gold_entity_relations(raw_doc, clusters, span_to_cluster)
    alignment = align_pred_to_gold(spans_by_eid, span_to_cluster)
    pred_e_b, pred_e_s = _pred_entity_relations(pred_graph, alignment, type_by_eid)
    rel_entity = _metrics_boundary(pred_e_b, gold_e_b)
    rel_plus_entity = _metrics_strict(pred_e_s, gold_e_s)

    return DocEval(
        doc_id=doc_id,
        ner=pm.ner,
        ner_boundary=ner_boundary,
        ner_relaxed=ner_relaxed,
        rel_mentionpair=pm.rel,
        rel_plus_mentionpair=pm.rel_plus,
        rel_entity=rel_entity,
        rel_plus_entity=rel_plus_entity,
        n_pred_mentions_total=pm.n_pred_mentions_total,
        n_pred_mentions_resolved=pm.n_pred_mentions_resolved,
        n_pred_entities=len(spans_by_eid),
        n_pred_entities_aligned=len(alignment),
    )


def _parse_response(resp: Any) -> dict[str, Any] | None:
    if isinstance(resp, str):
        try:
            obj = json.loads(resp)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    if isinstance(resp, dict):
        return resp
    return None


def _sum(metrics: list[Metrics]) -> Metrics:
    return Metrics(
        tp=sum(m.tp for m in metrics),
        fp=sum(m.fp for m in metrics),
        fn=sum(m.fn for m in metrics),
    )


def score_corpus(
    predictions: list[dict[str, Any]],
    raw_docs_by_id: dict[str, dict[str, Any]],
) -> CorpusEval:
    """Score every ``{id, response}`` prediction row against its raw SciERC doc.

    Rows whose ``id`` is missing from ``raw_docs_by_id`` are skipped with a
    warning. Rows whose ``response`` does not parse to a JSON object become
    full false-negative documents (counted under ``n_parse_failures``).
    """
    per_doc: list[DocEval] = []
    n_parse_failures = 0

    for row in predictions:
        doc_id = str(row.get("id", ""))
        raw_doc = raw_docs_by_id.get(doc_id)
        if raw_doc is None:
            print(f"[scierc_eval] skipping prediction {doc_id!r}: not in raw split")
            continue
        pred_graph = _parse_response(row.get("response"))
        if pred_graph is None:
            n_parse_failures += 1
            pred_graph = {"entities": [], "relations": []}
        per_doc.append(score_one_doc_eval(doc_id, raw_doc, pred_graph))

    def agg(attr: str) -> Metrics:
        return _sum([getattr(d, attr) for d in per_doc])

    return CorpusEval(
        ner=agg("ner"),
        ner_boundary=agg("ner_boundary"),
        ner_relaxed=agg("ner_relaxed"),
        rel_mentionpair=agg("rel_mentionpair"),
        rel_plus_mentionpair=agg("rel_plus_mentionpair"),
        rel_entity=agg("rel_entity"),
        rel_plus_entity=agg("rel_plus_entity"),
        n_docs=len(per_doc),
        n_parse_failures=n_parse_failures,
        n_pred_mentions_total=sum(d.n_pred_mentions_total for d in per_doc),
        n_pred_mentions_resolved=sum(d.n_pred_mentions_resolved for d in per_doc),
        n_pred_entities=sum(d.n_pred_entities for d in per_doc),
        n_pred_entities_aligned=sum(d.n_pred_entities_aligned for d in per_doc),
        per_doc=per_doc,
    )
