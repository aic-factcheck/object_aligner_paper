"""WEC-Eng evaluation: standard coreference metrics for a clustering run.

Our task LM emits the OA-aligned membership multigraph directly — a ``mentions``
list (``{id, marker}``) and an ``events`` list (each ``{mentions: [id, ...]}``).
OA's ``wec_eng`` score is the GEPA reward; this module computes the *native*
task metric: the official cross-document coreference family — **MUC**, **B³**,
**CEAFe**, and their mean **CoNLL F1** (the standard headline) — plus **LEA**.

Both the gold and the predicted clustering are reduced to a partition over the
instance's **markers** (the unique ``[mNN]`` tags shown in the prompt): we map
each side's mention ``id`` -> ``marker`` via its own ``mentions`` list, then
group markers by the ``events`` membership. Scoring over the marker universe
makes the comparison invariant to the opaque, self-assigned ids — exactly what
the permutation-invariant coreference metrics are designed for. The gold marker
set is the key mention universe; gold markers missing from the prediction become
singletons (penalised), predicted markers outside the gold universe are dropped,
and a marker emitted in several predicted events is kept in the first.

These numbers are the honest end-to-end metric for the synthesised clustering
instances; they are not directly comparable to leaderboard WEC-Eng numbers
(different mention universe / instance construction).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from scipy.optimize import linear_sum_assignment

__all__ = ["PRF", "DocEval", "CorpusEval", "score_corpus"]


@dataclass
class PRF:
    """Precision/recall accumulator as separate numerator/denominator sums."""

    p_num: float = 0.0
    p_den: float = 0.0
    r_num: float = 0.0
    r_den: float = 0.0

    @property
    def precision(self) -> float:
        return self.p_num / self.p_den if self.p_den else 0.0

    @property
    def recall(self) -> float:
        return self.r_num / self.r_den if self.r_den else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def add(self, other: "PRF") -> None:
        self.p_num += other.p_num
        self.p_den += other.p_den
        self.r_num += other.r_num
        self.r_den += other.r_den


Cluster = frozenset[str]


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


def _id_to_marker(obj: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in obj.get("mentions", []) or []:
        if not isinstance(m, dict):
            continue
        mid, marker = m.get("id"), m.get("marker")
        if mid is None or marker is None:
            continue
        out.setdefault(str(mid), str(marker))
    return out


def _gold_partition(gold: dict[str, Any]) -> tuple[list[Cluster], set[str]]:
    """Gold clusters over markers + the full gold marker universe."""
    id2m = _id_to_marker(gold)
    universe = set(id2m.values())
    clusters: list[Cluster] = []
    seen: set[str] = set()
    for ev in gold.get("events", []) or []:
        markers = {id2m[str(x)] for x in ev.get("mentions", []) if str(x) in id2m}
        if markers:
            clusters.append(frozenset(markers))
            seen |= markers
    # Any gold mention not covered by an event is its own singleton.
    for marker in universe - seen:
        clusters.append(frozenset({marker}))
    return clusters, universe


def _pred_partition(pred: dict[str, Any], universe: set[str]) -> list[Cluster]:
    """Predicted clusters over the gold marker universe (with singleton backfill)."""
    id2m = _id_to_marker(pred)
    clusters: list[Cluster] = []
    assigned: set[str] = set()
    for ev in pred.get("events", []) or []:
        markers: set[str] = set()
        for x in ev.get("mentions", []) or []:
            marker = id2m.get(str(x))
            if marker is not None and marker in universe and marker not in assigned:
                markers.add(marker)
                assigned.add(marker)
        if markers:
            clusters.append(frozenset(markers))
    for marker in universe - assigned:  # uncovered gold markers -> singletons
        clusters.append(frozenset({marker}))
    return clusters


# --- the three CoNLL metrics + LEA ----------------------------------------


def _muc(key: list[Cluster], resp: list[Cluster]) -> PRF:
    def half(a: list[Cluster], b: list[Cluster]) -> tuple[float, float]:
        num = den = 0.0
        for c in a:
            if len(c) <= 1:
                continue
            parts = sum(1 for d in b if c & d)
            num += len(c) - parts
            den += len(c) - 1
        return num, den

    r_num, r_den = half(key, resp)
    p_num, p_den = half(resp, key)
    return PRF(p_num=p_num, p_den=p_den, r_num=r_num, r_den=r_den)


def _b3(key: list[Cluster], resp: list[Cluster]) -> PRF:
    resp_of = {m: c for c in resp for m in c}
    key_of = {m: c for c in key for m in c}
    r_num = p_num = 0.0
    for c in key:
        for m in c:
            inter = len(c & resp_of[m])
            r_num += inter / len(c)
    for c in resp:
        for m in c:
            inter = len(c & key_of[m])
            p_num += inter / len(c)
    n = sum(len(c) for c in key)
    return PRF(p_num=p_num, p_den=float(n), r_num=r_num, r_den=float(n))


def _ceafe(key: list[Cluster], resp: list[Cluster]) -> PRF:
    if not key or not resp:
        return PRF(p_den=float(len(resp)), r_den=float(len(key)))
    sim = [[2 * len(k & r) / (len(k) + len(r)) for r in resp] for k in key]
    rows, cols = linear_sum_assignment([[-s for s in row] for row in sim])
    total = sum(sim[i][j] for i, j in zip(rows, cols))
    return PRF(p_num=total, p_den=float(len(resp)), r_num=total, r_den=float(len(key)))


def _lea(key: list[Cluster], resp: list[Cluster]) -> PRF:
    def link(c: Cluster) -> float:
        n = len(c)
        return n * (n - 1) / 2 if n > 1 else 0.0  # singleton importance via size below

    def half(a: list[Cluster], b: list[Cluster]) -> tuple[float, float]:
        num = den = 0.0
        for c in a:
            n = len(c)
            imp = n  # LEA importance = cluster size
            den += imp
            links_c = link(c)
            if n == 1:
                # singleton resolved iff it is a singleton on the other side too
                resolved = any(d == c for d in b)
                num += imp * (1.0 if resolved else 0.0)
                continue
            shared = 0.0
            for d in b:
                inter = len(c & d)
                if inter > 1:
                    shared += inter * (inter - 1) / 2
            num += imp * (shared / links_c if links_c else 0.0)
        return num, den

    r_num, r_den = half(key, resp)
    p_num, p_den = half(resp, key)
    return PRF(p_num=p_num, p_den=p_den, r_num=r_num, r_den=r_den)


@dataclass
class DocEval:
    doc_id: str
    conll_f1: float
    n_mentions: int
    n_gold_clusters: int
    n_pred_clusters: int


@dataclass
class CorpusEval:
    muc: PRF
    b3: PRF
    ceafe: PRF
    lea: PRF
    n: int
    n_parse_failures: int
    per_doc: list[DocEval] = field(default_factory=list)

    @property
    def muc_f1(self) -> float:
        return self.muc.f1

    @property
    def b3_f1(self) -> float:
        return self.b3.f1

    @property
    def ceafe_f1(self) -> float:
        return self.ceafe.f1

    @property
    def lea_f1(self) -> float:
        return self.lea.f1

    @property
    def conll_f1(self) -> float:
        return (self.muc.f1 + self.b3.f1 + self.ceafe.f1) / 3.0


def score_corpus(
    predictions: list[dict[str, Any]],
    examples_by_id: dict[str, dict[str, Any]],
) -> CorpusEval:
    """Score ``{id, response}`` rows against gold WEC-Eng partitions.

    ``examples_by_id`` maps id -> preprocessed example dict (carrying ``gold``).
    Rows whose id is absent are skipped with a warning. A response that fails to
    parse (or carries no events) is scored as the all-singletons partition.
    """
    muc, b3, ceafe, lea = PRF(), PRF(), PRF(), PRF()
    per_doc: list[DocEval] = []
    n_parse_failures = 0

    for row in predictions:
        doc_id = str(row.get("id", ""))
        ex = examples_by_id.get(doc_id)
        if ex is None:
            print(f"[wec_eng_eval] skipping {doc_id!r}: not in split")
            continue
        gold = ex["gold"]
        key, universe = _gold_partition(gold)
        pred = _parse_response(row.get("response"))
        empty = {"mentions": [], "events": []}
        if not isinstance(pred, dict) or not pred.get("events"):
            n_parse_failures += 1
            pred = empty
        try:
            resp = _pred_partition(pred, universe)
        except Exception:  # noqa: BLE001 — structurally invalid predicted graph
            # e.g. a mention/event that is a bare string instead of a dict; score
            # as the all-singletons partition rather than aborting the corpus.
            n_parse_failures += 1
            resp = _pred_partition(empty, universe)

        d_muc, d_b3, d_ceafe, d_lea = _muc(key, resp), _b3(key, resp), _ceafe(key, resp), _lea(key, resp)
        muc.add(d_muc)
        b3.add(d_b3)
        ceafe.add(d_ceafe)
        lea.add(d_lea)
        per_doc.append(DocEval(
            doc_id=doc_id,
            conll_f1=(d_muc.f1 + d_b3.f1 + d_ceafe.f1) / 3.0,
            n_mentions=len(universe),
            n_gold_clusters=len(key),
            n_pred_clusters=len(resp),
        ))

    return CorpusEval(
        muc=muc, b3=b3, ceafe=ceafe, lea=lea,
        n=len(per_doc), n_parse_failures=n_parse_failures, per_doc=per_doc,
    )
