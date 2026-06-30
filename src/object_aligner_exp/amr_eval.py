"""Official AMR Smatch F1 evaluation of an OA / GEPA run.

OA's referential-alignment score (``amr_ra``) is the GEPA *reward*; **Smatch** is
the standard AMR *leaderboard* metric, so this module recomputes it to place a
run against published numbers. Both are the same idea — a best variable
alignment followed by triple matching — so the OA-RA reward tracks Smatch closely
(see ``research/opus48_RA_real_world_datasets.md``); this module uses the
canonical ``smatch`` package as the authoritative scorer.

Smatch triples are built **directly** from the JSON wire shape via
``datasets.amr.gold_to_smatch_triples`` (no PENMAN re-encoding), so a malformed
prediction — disconnected graph, cycle, dangling endpoint — never raises: it just
scores its mismatched triples as misses. The corpus score is the standard
**micro-average** (sum match / test / gold over all examples, then F1).

Mirrors ``sentence_ordering_eval`` / ``planbench_eval``: ``score_corpus`` takes
``{id, response}`` rows + a dict id → preprocessed example (carrying ``gold``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import smatch

from object_aligner_exp.datasets.amr import gold_to_smatch_triples

__all__ = ["TaskResult", "score_corpus"]


def _prf(match: int, test: int, gold: int) -> tuple[float, float, float]:
    """Precision / recall / F1 with safe zero handling (no smatch div-by-zero)."""
    p = match / test if test else 0.0
    r = match / gold if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


@dataclass
class TaskResult:
    task: str = "amr"
    n: int = 0
    n_parse_failures: int = 0
    match_total: int = 0
    test_total: int = 0
    gold_total: int = 0
    per_example: list[dict[str, Any]] = field(default_factory=list)

    @property
    def smatch_precision(self) -> float:
        return _prf(self.match_total, self.test_total, self.gold_total)[0]

    @property
    def smatch_recall(self) -> float:
        return _prf(self.match_total, self.test_total, self.gold_total)[1]

    @property
    def smatch_f1(self) -> float:
        return _prf(self.match_total, self.test_total, self.gold_total)[2]


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


def _smatch_match(pred: dict[str, Any], gold: dict[str, Any]) -> tuple[int, int, int]:
    """Return (best_match, test_total, gold_total) for one (pred, gold) pair."""
    smatch.match_triple_dict = {}  # reset smatch's per-pair cache
    i1, a1, r1 = gold_to_smatch_triples(pred, "a")
    i2, a2, r2 = gold_to_smatch_triples(gold, "b")
    test_total = len(i1) + len(a1) + len(r1)
    gold_total = len(i2) + len(a2) + len(r2)
    if test_total == 0 or gold_total == 0:
        return 0, test_total, gold_total
    _, match = smatch.get_best_match(i1, a1, r1, i2, a2, r2, "a", "b")
    return match, test_total, gold_total


def score_corpus(
    predictions: list[dict[str, Any]],
    examples_by_id: dict[str, dict[str, Any]],
) -> TaskResult:
    """Score ``{id, response}`` rows against gold AMRs (micro-averaged Smatch).

    ``examples_by_id`` maps id → the preprocessed example dict (carrying ``gold``).
    Rows whose id is absent are skipped with a warning. A response that fails to
    parse to a non-empty graph counts as a parse failure (0 matched triples, all
    gold triples become false negatives).
    """
    res = TaskResult()
    for row in predictions:
        ex_id = str(row.get("id", ""))
        ex = examples_by_id.get(ex_id)
        if ex is None:
            print(f"[amr_eval] skipping {ex_id!r}: not in split")
            continue
        gold = ex["gold"]
        pred = _parse_response(row.get("response"))
        empty = {"root": None, "nodes": [], "relations": []}
        if not isinstance(pred, dict) or not pred.get("nodes"):
            res.n_parse_failures += 1
            pred = empty
        try:
            match, test_total, gold_total = _smatch_match(pred, gold)
        except Exception:  # noqa: BLE001 — structurally invalid predicted graph
            # e.g. a node/attribute that is a bare string instead of a dict;
            # score as a parse failure (0 matched triples) rather than aborting
            # the whole corpus.
            res.n_parse_failures += 1
            match, test_total, gold_total = _smatch_match(empty, gold)
        res.n += 1
        res.match_total += match
        res.test_total += test_total
        res.gold_total += gold_total
        res.per_example.append(
            {"id": ex_id, "match": match, "test": test_total, "gold": gold_total}
        )
    return res
