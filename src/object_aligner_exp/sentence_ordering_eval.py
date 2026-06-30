"""Official sentence-ordering metrics: Perfect-Match Ratio (PMR) + Kendall's τ.

OA's graded list-alignment score is the GEPA *reward*; this module computes the
standard *leaderboard* metrics so an OA / GEPA run can be placed against
published sentence-ordering numbers. Both are computed in pure Python (τ by hand,
no scipy/numpy), mirroring ``planbench_eval.py`` / ``natural_plan_eval.py``.

The prediction and gold are each a permutation of the input labels ``1..N`` (the
scrambled-sentence labels in correct reading order; see
``datasets/sentence_ordering.py``):

* **PMR** — fraction of examples whose predicted order matches the gold order
  exactly (binary per example, the strict metric).
* **Kendall's τ** — rank correlation between the predicted and gold orderings,
  in [-1, 1] (1 identical, −1 reversed); averaged over examples (the graded
  metric).

A response that does not parse to a valid permutation of ``1..N`` counts as a
parse failure → PMR 0 and τ 0 for that example.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TaskResult",
    "parse_order",
    "pmr",
    "kendall_tau",
    "score_corpus",
]


def parse_order(obj: dict[str, Any], n: int) -> list[int] | None:
    """Extract ``obj["indices"]`` as a permutation of ``1..n``; else None.

    Rejects wrong length, non-integers, out-of-range labels, and duplicates.
    """
    order = obj.get("indices")
    if not isinstance(order, list) or len(order) != n:
        return None
    out: list[int] = []
    for v in order:
        if isinstance(v, bool):  # bool is an int subclass — disallow
            return None
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        out.append(iv)
    if sorted(out) != list(range(1, n + 1)):
        return None
    return out


def pmr(pred: list[int], gold: list[int]) -> bool:
    """Perfect-Match Ratio contribution: exact order match."""
    return pred == gold


def kendall_tau(pred: list[int], gold: list[int]) -> float:
    """Kendall's τ between two permutations of the same labels.

    Counts concordant minus discordant label pairs, normalised by the number of
    pairs. τ=1 for identical orders, −1 for reversed. Returns 0.0 for n<2.
    """
    n = len(gold)
    if n < 2 or len(pred) != n:
        return 0.0
    gold_pos = {label: i for i, label in enumerate(gold)}
    pred_pos = {label: i for i, label in enumerate(pred)}
    labels = list(gold)
    concordant = discordant = 0
    for a in range(n):
        for b in range(a + 1, n):
            la, lb = labels[a], labels[b]
            dg = gold_pos[la] - gold_pos[lb]
            dp = pred_pos[la] - pred_pos[lb]
            if dg * dp > 0:
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) // 2
    return (concordant - discordant) / total if total else 0.0


# --- corpus driver (mirrors planbench_eval.TaskResult / score_corpus) -------


@dataclass
class TaskResult:
    task: str = "sentence_ordering"
    n: int = 0
    n_correct_pmr: int = 0
    sum_tau: float = 0.0
    n_parse_failures: int = 0
    # difficulty (=N) -> [pmr_correct, tau_sum, total]
    by_difficulty: dict[int, list[float]] = field(default_factory=dict)
    per_example: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pmr_rate(self) -> float:
        return self.n_correct_pmr / self.n if self.n else 0.0

    @property
    def mean_tau(self) -> float:
        return self.sum_tau / self.n if self.n else 0.0


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


def score_corpus(
    predictions: list[dict[str, Any]],
    examples_by_id: dict[str, dict[str, Any]],
) -> TaskResult:
    """Score ``{id, response}`` rows against their gold orderings.

    ``examples_by_id`` maps id → the preprocessed example dict (carrying ``gold``
    and ``meta``). Rows whose id is absent are skipped with a warning. A response
    that fails to parse to a valid permutation counts as a parse failure
    (PMR 0, τ 0). ``by_difficulty`` buckets by N (number of sentences).
    """
    res = TaskResult()
    for row in predictions:
        ex_id = str(row.get("id", ""))
        ex = examples_by_id.get(ex_id)
        if ex is None:
            print(f"[sentence_ordering_eval] skipping {ex_id!r}: not in split")
            continue
        gold = list(ex["gold"]["indices"])
        n = len(gold)
        diff = int(ex["meta"]["difficulty"])
        obj = _parse_response(row.get("response"))
        pred = parse_order(obj, n) if obj is not None else None
        if pred is None:
            res.n_parse_failures += 1
            is_pmr, tau = False, 0.0
        else:
            is_pmr, tau = pmr(pred, gold), kendall_tau(pred, gold)
        res.n += 1
        res.n_correct_pmr += int(is_pmr)
        res.sum_tau += tau
        bucket = res.by_difficulty.setdefault(diff, [0.0, 0.0, 0.0])
        bucket[0] += int(is_pmr)
        bucket[1] += tau
        bucket[2] += 1
        res.per_example.append(
            {"id": ex_id, "pmr": is_pmr, "tau": tau, "difficulty": diff}
        )
    return res
