"""DocRED / Re-DocRED evaluation for end-to-end (generative) extractors.

Our task LM emits the OA-aligned graph directly — one entity object per
coreference cluster (its surface forms in ``mentions[]``) and one relation per
*entity pair*. Re-DocRED gold is already in that same cluster-level shape
(``datasets/docred.py``), so — unlike SciERC — there is no token-span recovery
to do; we score cluster pairs against cluster pairs.

The standard DocRED leaderboard metric assumes the gold entity set (vertexSet)
is *given* and matches relations by entity index. We run the harder end-to-end
setting (the model must produce the entities too), so we substitute a
**coref-aware relation F1**: predicted entity clusters are aligned to gold
clusters by shared (normalised) mention surface forms (greedy 1:1), then a
predicted relation ``(subject_cluster, predicate, object_cluster)`` is a true
positive iff a gold relation maps to the same gold-cluster pair with the same
predicate. This mirrors ``scierc_eval``'s headline ``rel_entity`` protocol.

NOTE: because entities are produced by the model rather than given, these
numbers are NOT directly comparable to the official index-based DocRED
leaderboard — they are an honest end-to-end stand-in. ``rel_ign_f1`` (the
DocRED "Ign F1": drop gold facts that also occur in the training set, keyed by
the head/tail *names* + predicate) is computed only when ``train_facts`` is
supplied; the report path leaves it out.

Predicates are compared as written, so this scorer is variant-agnostic: a
``pcode`` run is graded against pcode gold, an ``obf`` run against obf gold.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CorpusEval", "DocEval", "Metrics", "build_train_facts", "score_corpus"]


@dataclass
class Metrics:
    """Micro P/R/F1 with the underlying TP/FP/FN counts."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _norm(text: Any) -> str:
    return str(text).strip().lower()


def _cluster_names(entity: dict[str, Any]) -> frozenset[str]:
    return frozenset(_norm(m.get("text", "")) for m in entity.get("mentions", []) if m.get("text"))


def _gold_clusters(gold: dict[str, Any]) -> tuple[dict[str, int], list[frozenset[str]], list[str]]:
    """Return (id->idx, names_by_idx, type_by_idx) for a gold graph."""
    id_to_idx: dict[str, int] = {}
    names: list[frozenset[str]] = []
    types: list[str] = []
    for i, e in enumerate(gold.get("entities", []) or []):
        id_to_idx[str(e.get("id"))] = i
        names.append(_cluster_names(e))
        types.append(str(e.get("type", "")))
    return id_to_idx, names, types


def _align_pred_to_gold(
    pred: dict[str, Any],
    gold_names: list[frozenset[str]],
) -> dict[str, int]:
    """Greedy 1:1 alignment of predicted entities to gold clusters by name overlap."""
    candidates: list[tuple[int, int, str]] = []  # (-overlap, gold_idx, pred_id)
    for e in pred.get("entities", []) or []:
        pid = str(e.get("id"))
        pnames = _cluster_names(e)
        if not pnames:
            continue
        for gi, gnames in enumerate(gold_names):
            ov = len(pnames & gnames)
            if ov > 0:
                candidates.append((-ov, gi, pid))
    candidates.sort()
    alignment: dict[str, int] = {}
    used_gold: set[int] = set()
    for _, gi, pid in candidates:
        if pid in alignment or gi in used_gold:
            continue
        alignment[pid] = gi
        used_gold.add(gi)
    return alignment


def _gold_rel_tuples(gold: dict[str, Any], id_to_idx: dict[str, int]) -> set[tuple[int, str, int]]:
    out: set[tuple[int, str, int]] = set()
    for r in gold.get("relations", []) or []:
        h = id_to_idx.get(str(r.get("subject")))
        t = id_to_idx.get(str(r.get("object")))
        if h is None or t is None:
            continue
        out.add((h, str(r.get("predicate", "")), t))
    return out


def _pred_rel_tuples(pred: dict[str, Any], alignment: dict[str, int]) -> set[tuple[Any, str, Any]]:
    out: set[tuple[Any, str, Any]] = set()
    for r in pred.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        subj = str(r.get("subject", ""))
        obj = str(r.get("object", ""))
        pred_label = str(r.get("predicate", ""))
        if not subj or not obj or not pred_label:
            continue
        # Unaligned predicted entities get a sentinel so they can never match a
        # gold tuple (they count as FP) while still deduping per pred pair.
        hi: Any = alignment.get(subj, ("U", subj))
        ti: Any = alignment.get(obj, ("U", obj))
        if hi == ti:
            continue
        out.add((hi, pred_label, ti))
    return out


def build_train_facts(train_examples: Iterable[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Name-keyed gold relation facts from the training split, for Ign-F1.

    A fact is ``(head_name, predicate, tail_name)`` over the *first* mention of
    each entity, lower-cased. Used to drop train-seen facts from the dev/test
    gold + predictions (the official DocRED "Ign" adjustment).
    """
    facts: set[tuple[str, str, str]] = set()
    for ex in train_examples:
        gold = ex["gold"]
        names = {str(e.get("id")): _cluster_names(e) for e in gold.get("entities", []) or []}
        for r in gold.get("relations", []) or []:
            hs = names.get(str(r.get("subject")), frozenset())
            ts = names.get(str(r.get("object")), frozenset())
            pred_label = str(r.get("predicate", ""))
            for hn in hs:
                for tn in ts:
                    facts.add((hn, pred_label, tn))
    return facts


@dataclass
class DocEval:
    doc_id: str
    rel: Metrics
    n_pred_entities: int
    n_pred_entities_aligned: int


@dataclass
class CorpusEval:
    rel: Metrics
    rel_ign: Metrics | None
    n: int
    n_parse_failures: int
    n_pred_entities: int
    n_pred_entities_aligned: int
    per_doc: list[DocEval] = field(default_factory=list)

    @property
    def rel_f1(self) -> float:
        return self.rel.f1

    @property
    def rel_precision(self) -> float:
        return self.rel.precision

    @property
    def rel_recall(self) -> float:
        return self.rel.recall

    @property
    def rel_ign_f1(self) -> float | None:
        return self.rel_ign.f1 if self.rel_ign is not None else None


def _parse_response(resp: Any) -> dict[str, Any] | None:
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, str):
        try:
            obj = json.loads(resp)
        except (json.JSONDecodeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None
    return None


def score_one_doc(doc_id: str, gold: dict[str, Any], pred: dict[str, Any]) -> tuple[DocEval, set, set]:
    id_to_idx, gold_names, _types = _gold_clusters(gold)
    alignment = _align_pred_to_gold(pred, gold_names)
    gold_rels = _gold_rel_tuples(gold, id_to_idx)
    pred_rels = _pred_rel_tuples(pred, alignment)
    tp = len(pred_rels & gold_rels)
    m = Metrics(tp=tp, fp=len(pred_rels) - tp, fn=len(gold_rels) - tp)
    doc = DocEval(
        doc_id=doc_id,
        rel=m,
        n_pred_entities=len(pred.get("entities", []) or []),
        n_pred_entities_aligned=len(alignment),
    )
    # Return the resolved tuple sets (with names) for optional Ign-F1.
    idx_to_names = {i: gold_names[i] for i in range(len(gold_names))}

    def name_tuples(rels: set, use_pred: bool) -> set[tuple[frozenset, str, frozenset]]:
        out = set()
        for h, lab, t in rels:
            hn = idx_to_names.get(h, frozenset()) if isinstance(h, int) else frozenset()
            tn = idx_to_names.get(t, frozenset()) if isinstance(t, int) else frozenset()
            out.add((hn, lab, tn))
        return out

    return doc, name_tuples(gold_rels, False), name_tuples(pred_rels & gold_rels, True)


def score_corpus(
    predictions: list[dict[str, Any]],
    examples_by_id: dict[str, dict[str, Any]],
    train_facts: set[tuple[str, str, str]] | None = None,
) -> CorpusEval:
    """Score ``{id, response}`` rows against gold DocRED graphs (coref-aware rel F1).

    ``examples_by_id`` maps id → preprocessed example dict (carrying ``gold``).
    Rows whose id is absent are skipped with a warning. A response that fails to
    parse becomes an empty prediction (all gold relations become FN).
    ``train_facts`` (from :func:`build_train_facts`) enables Ign-F1.
    """
    per_doc: list[DocEval] = []
    n_parse_failures = 0
    rel = Metrics()
    rel_ign: Metrics | None = Metrics() if train_facts is not None else None

    for row in predictions:
        doc_id = str(row.get("id", ""))
        ex = examples_by_id.get(doc_id)
        if ex is None:
            print(f"[docred_eval] skipping {doc_id!r}: not in split")
            continue
        gold = ex["gold"]
        pred = _parse_response(row.get("response"))
        empty = {"entities": [], "relations": []}
        if not isinstance(pred, dict) or not pred.get("entities"):
            n_parse_failures += 1
            pred = empty
        try:
            doc, _gold_names_t, _ = score_one_doc(doc_id, gold, pred)
        except Exception:  # noqa: BLE001 — structurally invalid predicted graph
            # e.g. an entity/mention/relation that is a bare string instead of a
            # dict; score as a parse failure (all gold rels become FN) rather
            # than aborting the whole corpus.
            n_parse_failures += 1
            pred = empty
            doc, _gold_names_t, _ = score_one_doc(doc_id, gold, pred)
        per_doc.append(doc)
        rel.tp += doc.rel.tp
        rel.fp += doc.rel.fp
        rel.fn += doc.rel.fn

        if rel_ign is not None:
            # Recompute on the train-fact-filtered tuple sets (name-keyed).
            id_to_idx, gold_names, _ = _gold_clusters(gold)
            alignment = _align_pred_to_gold(pred, gold_names)
            gold_rels = _gold_rel_tuples(gold, id_to_idx)
            pred_rels = _pred_rel_tuples(pred, alignment)

            def seen_in_train(h: Any, lab: str, t: Any) -> bool:
                hn = gold_names[h] if isinstance(h, int) and h < len(gold_names) else frozenset()
                tn = gold_names[t] if isinstance(t, int) and t < len(gold_names) else frozenset()
                return any((a, lab, b) in train_facts for a in hn for b in tn)

            gold_f = {x for x in gold_rels if not seen_in_train(*x)}
            pred_f = {x for x in pred_rels if not seen_in_train(*x)}
            tp = len(pred_f & gold_f)
            rel_ign.tp += tp
            rel_ign.fp += len(pred_f) - tp
            rel_ign.fn += len(gold_f) - tp

    return CorpusEval(
        rel=rel,
        rel_ign=rel_ign,
        n=len(per_doc),
        n_parse_failures=n_parse_failures,
        n_pred_entities=sum(d.n_pred_entities for d in per_doc),
        n_pred_entities_aligned=sum(d.n_pred_entities_aligned for d in per_doc),
        per_doc=per_doc,
    )
