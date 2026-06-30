"""Official NATURAL PLAN exact-match scoring, reconciled to our wire shape.

OA's graded alignment score is the GEPA *reward*; this module computes the
*leaderboard* metric so an OA / GEPA run can be placed against published
NATURAL PLAN solve-rates. The logic is a faithful reimplementation of the
official ``evaluate_trip_planning.py`` / ``evaluate_meeting_planning.py``
(google-deepmind/natural-plan, Apache-2.0), adapted to read our JSON wire
output instead of the benchmark's free-text predictions.

Both tasks score binary 0/1 per example, averaged into a solve-rate, and we
also break the rate down by difficulty (``num_cities`` for trip,
``num_people`` for meeting).

Trip (``{"itinerary": [{"city", "days"}]}``)
    Sequential exact match: the predicted ``(city, days)`` sequence must match
    the gold ``cities``/``durations`` for every gold stay, in order (matching
    ``compute_example_score`` — count matches left-to-right, break on the first
    mismatch, require all gold stays matched). Extra trailing predicted stays
    are ignored; a short prediction fails.

Meeting (``{"schedule": [{"person", "location", "start_time", "duration"}]}``)
    Validator-based: we walk the schedule (mirroring ``validator_from_dict``),
    travelling between consecutive locations via ``dist_matrix``, and count
    *valid* meetings (right location, inside the person's availability window,
    no duplicates). The example scores 1 iff that count equals the golden
    plan's valid-meeting count (the optimum). NOTE the validator uses the
    *constraint's* required meeting length, not the ``duration`` the model
    emits — so ``duration`` matters for the OA reward but not for official EM.
"""

from __future__ import annotations

import datetime
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TaskResult",
    "score_corpus",
    "score_meeting_example",
    "score_trip_example",
]


# --- Trip ------------------------------------------------------------------


def _parse_trip_pred(obj: dict[str, Any]) -> list[tuple[str, int]] | None:
    """Extract an ordered ``(city, days)`` list from a wire-shape dict."""
    itinerary = obj.get("itinerary")
    if not isinstance(itinerary, list):
        return None
    out: list[tuple[str, int]] = []
    for item in itinerary:
        if not isinstance(item, dict):
            return None
        city = item.get("city")
        days = item.get("days")
        if city is None or days is None:
            return None
        try:
            days_int = int(days)
        except (TypeError, ValueError):
            return None
        out.append((str(city).strip(), days_int))
    return out


def score_trip_example(pred_obj: dict[str, Any], meta: dict[str, Any]) -> bool:
    """Binary exact-match for one trip example (mirrors compute_example_score)."""
    parsed = _parse_trip_pred(pred_obj)
    if not parsed:
        return False
    stays = [str(c).strip() for c in meta["cities"]]
    days = [int(d) for d in meta["durations"]]
    if not stays:
        return False
    num_stays = min(len(stays), len(parsed))
    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed[i][0] and days[i] == parsed[i][1]:
            num_match += 1
        else:
            break
    return num_match / len(stays) >= 1.0


# --- Meeting ---------------------------------------------------------------


def _to_time(time_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(time_str.strip(), "%I:%M%p")


def _process_constraints(rows: list[Any]) -> dict[str, dict[str, Any]]:
    """Build ``{person: {location, start_time, end_time, meeting_time}}``."""
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, location, times, meeting_time in rows:
        out[name]["location"] = location
        start_s, end_s = times.split("to")
        out[name]["start_time"] = _to_time(start_s)
        out[name]["end_time"] = _to_time(end_s)
        out[name]["meeting_time"] = meeting_time
    return out


def _validator_from_dict(
    plan: list[dict[str, Any]],
    constraints: dict[str, Any],
    start_location: str,
    initial_time: str,
    dist_matrix: dict[str, Any],
) -> int:
    """Count valid meetings in a JSON-format plan (mirrors validator_from_dict).

    ``plan[0]`` is treated as the start placeholder and skipped, matching the
    official script's ``for step in plan[1:]``.
    """
    met_with: dict[str, int] = {}
    score = 0
    cur_location = start_location
    cur_time = _to_time(initial_time)
    for step in plan[1:]:
        try:
            location = step["location"]
            if location and location != cur_location:
                cur_time = cur_time + datetime.timedelta(
                    minutes=dist_matrix[cur_location][location]
                )
                cur_location = location
            start_time = _to_time(step["start_time"])
            if start_time < cur_time:
                raise ValueError("Start time too early")
            cur_time = start_time
            person = step["person_name"]
            if person not in constraints:
                continue
            if person in met_with:
                raise ValueError("already met")
            met_with[person] = 1
            new_time = cur_time + datetime.timedelta(
                minutes=constraints[person]["meeting_time"]
            )
            if (
                cur_location == constraints[person]["location"]
                and cur_time >= constraints[person]["start_time"]
                and new_time <= constraints[person]["end_time"]
            ):
                score += 1
                cur_time = new_time
            else:
                raise ValueError("Invalid meeting time or location")
        except (ValueError, KeyError, TypeError):
            break
    return score


def _validator_from_text(
    plan: list[str],
    constraints: dict[str, Any],
    start_location: str,
    initial_time: str,
    dist_matrix: dict[str, Any],
) -> int:
    """Count valid meetings in a text-format plan (mirrors validator_from_text).

    Used to score the gold ``golden_plan`` (the optimum) for comparison.
    """
    met_with: dict[str, int] = {}
    score = 0
    cur_location = start_location
    cur_time = _to_time(initial_time)
    for step in plan:
        try:
            step = step.strip()
            if step.startswith("You start"):
                continue
            elif step.startswith("You travel"):
                destination = step.split("travel to ")[1].split(" in")[0].strip()
                cur_time = cur_time + datetime.timedelta(
                    minutes=dist_matrix[cur_location][destination]
                )
                cur_location = destination
            elif step.startswith("You wait"):
                raw_end = step.split("wait until ")[1].split(".")[0].strip()
                end_time = _to_time(raw_end)
                if end_time <= cur_time:
                    raise ValueError("backwards in time")
                cur_time = end_time
            elif step.startswith("You meet"):
                person = step.split("meet ")[1].split(" for")[0].strip()
                if person in met_with:
                    raise ValueError("already met")
                met_with[person] = 1
                new_time = cur_time + datetime.timedelta(
                    minutes=constraints[person]["meeting_time"]
                )
                if (
                    cur_location == constraints[person]["location"]
                    and cur_time >= constraints[person]["start_time"]
                    and new_time <= constraints[person]["end_time"]
                ):
                    score += 1
                    cur_time = new_time
                else:
                    raise ValueError("Invalid meeting time or location")
            else:
                raise ValueError("Unknown plan format")
        except (ValueError, KeyError, TypeError, IndexError):
            break
    return score


def _parse_meeting_pred(obj: dict[str, Any]) -> list[dict[str, Any]] | None:
    schedule = obj.get("schedule")
    if not isinstance(schedule, list):
        return None
    return schedule


def golden_meeting_score(meta: dict[str, Any]) -> int:
    """Number of valid meetings in the gold plan (the per-example optimum)."""
    constraints = _process_constraints(meta["constraints"][1:])
    start_location, initial_time = meta["constraints"][0]
    return _validator_from_text(
        [str(s) for s in meta["golden_plan"]],
        constraints,
        start_location,
        initial_time,
        meta["dist_matrix"],
    )


def score_meeting_example(pred_obj: dict[str, Any], meta: dict[str, Any]) -> bool:
    """Binary: predicted valid-meeting count equals the golden optimum."""
    schedule = _parse_meeting_pred(pred_obj)
    if schedule is None:
        return False
    constraints = _process_constraints(meta["constraints"][1:])
    start_location, initial_time = meta["constraints"][0]
    plan = [{"location": start_location, "person_name": "N/A", "start_time": initial_time}]
    for it in schedule:
        if not isinstance(it, dict):
            return False
        plan.append(
            {
                "location": str(it.get("location", "")),
                "person_name": str(it.get("person", "")),
                "start_time": str(it.get("start_time", "")),
            }
        )
    score = _validator_from_dict(
        plan, constraints, start_location, initial_time, meta["dist_matrix"]
    )
    return score == golden_meeting_score(meta)


# --- corpus driver ---------------------------------------------------------


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
    task: str,
    predictions: list[dict[str, Any]],
    examples_by_id: dict[str, dict[str, Any]],
) -> TaskResult:
    """Score ``{id, response}`` rows against their NATURAL PLAN examples.

    ``examples_by_id`` maps id → the preprocessed example dict (carrying
    ``meta``). Rows whose id is absent are skipped with a warning. A response
    that fails to parse to a JSON object counts as an incorrect example
    (tallied under ``n_parse_failures``).
    """
    scorer = {"trip": score_trip_example, "meeting": score_meeting_example}[task]
    res = TaskResult(task=task)
    for row in predictions:
        ex_id = str(row.get("id", ""))
        ex = examples_by_id.get(ex_id)
        if ex is None:
            print(f"[natural_plan_eval] skipping {ex_id!r}: not in split")
            continue
        meta = ex["meta"]
        diff = int(meta["difficulty"])
        pred = _parse_response(row.get("response"))
        if pred is None:
            res.n_parse_failures += 1
            correct = False
        else:
            try:
                correct = bool(scorer(pred, meta))
            except Exception:  # noqa: BLE001 — any scorer failure → incorrect
                correct = False
        res.n += 1
        res.n_correct += int(correct)
        bucket = res.by_difficulty.setdefault(diff, [0, 0])
        bucket[0] += int(correct)
        bucket[1] += 1
        res.per_example.append({"id": ex_id, "correct": correct, "difficulty": diff})
    return res
