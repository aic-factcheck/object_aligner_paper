"""Official PlanBench (Blocksworld) plan-*validity* scoring, in pure Python.

OA's graded list-alignment score is the GEPA *reward*; this module computes the
*leaderboard* metric so an OA / GEPA run can be placed against published
PlanBench Blocksworld plan-generation solve-rates (e.g. o1-preview ≈ 97.8%,
GPT-4 ≈ 34.6%, Llama-3.1-405B ≈ 62.6%).

PlanBench's official metric is plan **validity**, not exact-match against the
single gold plan: an action sequence scores 1 iff, executed from the initial
state, every action's preconditions hold and the resulting state satisfies the
goal. Alternative valid orderings therefore also score 1. The benchmark uses the
VAL plan validator over PDDL; we reimplement the (small, deterministic)
Blocksworld semantics directly in Python so no external binary or PDDL files are
needed — mirroring how ``natural_plan_eval.py`` reimplements that benchmark's
scorer.

State is a set of ground predicates; ``init``/``goal`` arrive as lists of
predicate tuples (see ``datasets/planbench.py``):
``["clear", x]``, ``["ontable", x]``, ``["on", x, y]``, ``["holding", x]``,
``["handempty"]``.

Blocksworld actions (standard IPC semantics):
    pick-up(x)   pre: clear x, ontable x, handempty
                 eff: holding x; ¬ontable x, ¬clear x, ¬handempty
    put-down(x)  pre: holding x
                 eff: ontable x, clear x, handempty; ¬holding x
    stack(x,y)   pre: holding x, clear y
                 eff: on x y, clear x, handempty; ¬holding x, ¬clear y
    unstack(x,y) pre: on x y, clear x, handempty
                 eff: holding x, clear y; ¬on x y, ¬clear x, ¬handempty
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TaskResult",
    "InvalidPlan",
    "plan_reaches_goal",
    "score_blocksworld_example",
    "score_corpus",
]


Fact = tuple[str, ...]


class InvalidPlan(Exception):
    """Raised when an action's preconditions are not met in the current state."""


def _facts_to_set(facts: list[Any]) -> set[Fact]:
    return {tuple(str(t) for t in f) for f in facts}


def _apply_action(state: set[Fact], action: str, args: list[str]) -> None:
    """Mutate ``state`` by applying one action; raise ``InvalidPlan`` if illegal."""
    if action == "pick-up":
        (x,) = args
        if not ({("clear", x), ("ontable", x), ("handempty",)} <= state):
            raise InvalidPlan(f"pick-up {x}: preconditions unmet")
        state.difference_update({("ontable", x), ("clear", x), ("handempty",)})
        state.add(("holding", x))
    elif action == "put-down":
        (x,) = args
        if ("holding", x) not in state:
            raise InvalidPlan(f"put-down {x}: not holding")
        state.discard(("holding", x))
        state.update({("ontable", x), ("clear", x), ("handempty",)})
    elif action == "stack":
        x, y = args
        if not ({("holding", x), ("clear", y)} <= state):
            raise InvalidPlan(f"stack {x} {y}: preconditions unmet")
        state.difference_update({("holding", x), ("clear", y)})
        state.update({("on", x, y), ("clear", x), ("handempty",)})
    elif action == "unstack":
        x, y = args
        if not ({("on", x, y), ("clear", x), ("handempty",)} <= state):
            raise InvalidPlan(f"unstack {x} {y}: preconditions unmet")
        state.difference_update({("on", x, y), ("clear", x), ("handempty",)})
        state.update({("holding", x), ("clear", y)})
    else:
        raise InvalidPlan(f"unknown action {action!r}")


def plan_reaches_goal(
    init: list[Any], goal: list[Any], plan: list[dict[str, Any]]
) -> bool:
    """True iff ``plan`` is executable from ``init`` and reaches ``goal``.

    Each step is ``{"action": str, "args": [str, ...]}``. Any precondition
    violation, malformed step, or wrong action arity makes the plan invalid
    (returns False). The goal is satisfied iff every goal fact holds in the
    final state.
    """
    state = _facts_to_set(init)
    for step in plan:
        try:
            action = str(step["action"])
            args = [str(a) for a in step["args"]]
        except (KeyError, TypeError):
            return False
        # arity check: 1 arg for pick-up/put-down, 2 for stack/unstack.
        expected = 1 if action in ("pick-up", "put-down") else 2
        if len(args) != expected:
            return False
        try:
            _apply_action(state, action, args)
        except InvalidPlan:
            return False
    return _facts_to_set(goal) <= state


def _parse_plan(obj: dict[str, Any]) -> list[dict[str, Any]] | None:
    plan = obj.get("plan")
    if not isinstance(plan, list):
        return None
    return plan


def score_blocksworld_example(pred_obj: dict[str, Any], meta: dict[str, Any]) -> bool:
    """Binary validity for one example: does the predicted plan reach the goal."""
    plan = _parse_plan(pred_obj)
    if plan is None:
        return False
    return plan_reaches_goal(meta["init"], meta["goal"], plan)


# --- corpus driver (mirrors natural_plan_eval.TaskResult / score_corpus) ----


@dataclass
class TaskResult:
    task: str
    n: int = 0
    n_correct: int = 0
    n_parse_failures: int = 0
    by_difficulty: dict[int, list[int]] = field(default_factory=dict)  # diff -> [correct, total]
    per_example: list[dict[str, Any]] = field(default_factory=list)

    @property
    def solve_rate(self) -> float:
        return self.n_correct / self.n if self.n else 0.0


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
    """Score ``{id, response}`` rows against their Blocksworld examples.

    ``examples_by_id`` maps id → the preprocessed example dict (carrying
    ``meta``). Rows whose id is absent are skipped with a warning. A response
    that fails to parse to a JSON object counts as incorrect (tallied under
    ``n_parse_failures``). ``by_difficulty`` buckets by gold plan length.
    """
    res = TaskResult(task="blocksworld")
    for row in predictions:
        ex_id = str(row.get("id", ""))
        ex = examples_by_id.get(ex_id)
        if ex is None:
            print(f"[planbench_eval] skipping {ex_id!r}: not in split")
            continue
        meta = ex["meta"]
        diff = int(meta["difficulty"])
        pred = _parse_response(row.get("response"))
        if pred is None:
            res.n_parse_failures += 1
            correct = False
        else:
            try:
                correct = bool(score_blocksworld_example(pred, meta))
            except Exception:  # noqa: BLE001 — any scorer failure → incorrect
                correct = False
        res.n += 1
        res.n_correct += int(correct)
        bucket = res.by_difficulty.setdefault(diff, [0, 0])
        bucket[0] += int(correct)
        bucket[1] += 1
        res.per_example.append({"id": ex_id, "correct": correct, "difficulty": diff})
    return res
