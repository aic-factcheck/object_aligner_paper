"""NATURAL PLAN dataset utilities (Trip Planning + Meeting Planning).

NATURAL PLAN (Zheng et al., Google DeepMind, arXiv:2406.04520) is an
evaluation-only planning benchmark whose tasks put the *entire* problem in the
prompt (flight tables, travel-time matrices, availability windows) and ask for
a strictly **ordered** plan by pure reasoning — no tools, no retrieval. We use
two of its three tasks, because both produce an ordered list and so let us
contrast OA's order-aware (`"order": "fixed"`) vs order-blind
(`"order": "align"`) list alignment.

Raw files (downloaded by ``scripts/prepare_natural_plan.py`` from
``github.com/google-deepmind/natural-plan``) are JSON objects keyed by example
id.

Trip Planning (``trip_planning.json``, 1,600 examples) — per example::

    {
      "num_cities":   "3",
      "cities":       "Helsinki**Barcelona**Florence",     # gold, in order
      "durations":    "5**5**6",                            # gold stay lengths
      "prompt_0shot": "You plan to visit 3 European cities ...",
      "prompt_5shot": "...",
      "golden_plan":  "**Day 1-5:** ... Helsinki ...",      # free text
      "pred_5shot_pro": "..."                               # their model output
    }

We convert to the OA wire shape used by ``data/natural_plan/trip/schemas/``::

    {"itinerary": [{"city": "Helsinki", "days": 5},
                   {"city": "Barcelona", "days": 5},
                   {"city": "Florence", "days": 6}]}

Note on durations: the fly day counts toward both the departure and arrival
city, so consecutive stays overlap by one day and ``sum(durations) ==
total_days + n_flights`` (here 16 = 14 + 2). The gold ``cities``/``durations``
fields encode this directly, so we don't parse the free-text ``golden_plan``.

Meeting Planning (``meeting_planning.json``, 1,000 examples) — per example::

    {
      "num_people":  1,
      "constraints": [["Marina District", "9:00AM"],                # start
                      ["Stephanie", "Mission District",
                       "10:30AM to 1:30PM", 120]],                  # person...
      "dist_matrix": {"Marina District": {"Mission District": 20}, ...},
      "prompt_0shot": "You are visiting San Francisco ...",
      "golden_plan":  ["You start at Marina District at 9:00AM.",
                       "You travel to Mission District in 20 minutes ...",
                       "You wait until 10:30AM.",
                       "You meet Stephanie for 120 minutes from 10:30AM to 12:30PM."],
      ...
    }

We convert to the OA wire shape used by ``data/natural_plan/meeting/schemas/``::

    {"schedule": [{"person": "Stephanie", "location": "Mission District",
                   "start_time": "10:30AM", "duration": 120}]}

built by walking ``golden_plan`` (tracking the current location across
``start``/``travel`` steps and emitting one object per ``meet`` step).

Each example also carries a ``meta`` dict with the raw gold fields the official
NATURAL PLAN scorer needs (``cities``/``durations``/``num_cities`` for trip;
``constraints``/``dist_matrix``/``golden_plan``/``num_people`` for meeting) and
the difficulty key used for stratified split carving.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

TRIP_RAW_NAME = "trip_planning.json"
MEETING_RAW_NAME = "meeting_planning.json"

DATASET_URL = "https://github.com/google-deepmind/natural-plan"


class NaturalPlanExample(TypedDict):
    """One preprocessed NATURAL PLAN example, ready for task LM + OA scoring."""

    id: str
    title: str
    context: str
    gold: dict[str, Any]
    meta: dict[str, Any]


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object keyed by example id")
    return obj


# --- Trip Planning ---------------------------------------------------------


def parse_trip_example(key: str, ex: dict[str, Any]) -> NaturalPlanExample:
    """Convert one raw Trip Planning record into the OA-aligned shape.

    Raises ``ValueError`` on malformed input (missing fields, length mismatch).
    """
    try:
        cities_raw = ex["cities"]
        durations_raw = ex["durations"]
        context = ex["prompt_0shot"]
    except KeyError as exc:
        raise ValueError(f"missing required field: {exc.args[0]!r}") from exc

    cities = [c.strip() for c in str(cities_raw).split("**") if c.strip()]
    try:
        durations = [int(d) for d in str(durations_raw).split("**") if d.strip()]
    except ValueError as exc:
        raise ValueError(f"non-integer duration in {durations_raw!r}") from exc
    if len(cities) != len(durations):
        raise ValueError(
            f"cities/durations length mismatch: {len(cities)} vs {len(durations)}"
        )
    if not cities:
        raise ValueError("empty itinerary")

    gold = {
        "itinerary": [
            {"city": c, "days": d} for c, d in zip(cities, durations)
        ]
    }
    meta = {
        "task": "trip",
        "num_cities": int(ex.get("num_cities", len(cities))),
        "cities": cities,
        "durations": durations,
        "difficulty": int(ex.get("num_cities", len(cities))),
    }
    return NaturalPlanExample(
        id=key, title=key, context=str(context), gold=gold, meta=meta
    )


def iter_trip_examples(
    raw_dir: Path, *, limit: int | None = None
) -> Iterator[NaturalPlanExample]:
    """Stream Trip Planning examples, skipping malformed records."""
    raw = _load_raw(Path(raw_dir) / TRIP_RAW_NAME)
    yielded = 0
    for key, ex in raw.items():
        try:
            out = parse_trip_example(key, ex)
        except ValueError:
            continue
        yield out
        yielded += 1
        if limit is not None and yielded >= limit:
            return


# --- Meeting Planning ------------------------------------------------------

_RE_START = re.compile(r"You start at (?P<loc>.+?) at (?P<time>[0-9:APM]+)\.?$")
_RE_TRAVEL = re.compile(r"You travel to (?P<loc>.+?) in \d+ minutes")
_RE_MEET = re.compile(
    r"You meet (?P<person>.+?) for (?P<dur>\d+) minutes "
    r"from (?P<start>[0-9:APM]+) to (?P<end>[0-9:APM]+)\.?$"
)


def parse_meeting_golden_plan(steps: list[str]) -> list[dict[str, Any]]:
    """Walk a Meeting Planning ``golden_plan`` into an ordered schedule.

    Tracks the current location across ``start``/``travel`` steps and emits one
    ``{person, location, start_time, duration}`` object per ``meet`` step, in
    order. ``wait`` steps and anything unrecognised are ignored.
    """
    schedule: list[dict[str, Any]] = []
    current_loc: str | None = None
    for step in steps:
        s = step.strip()
        m = _RE_START.search(s)
        if m:
            current_loc = m.group("loc").strip()
            continue
        m = _RE_TRAVEL.search(s)
        if m:
            current_loc = m.group("loc").strip()
            continue
        m = _RE_MEET.search(s)
        if m:
            schedule.append(
                {
                    "person": m.group("person").strip(),
                    "location": current_loc or "",
                    "start_time": m.group("start").strip(),
                    "duration": int(m.group("dur")),
                }
            )
    return schedule


def parse_meeting_example(key: str, ex: dict[str, Any]) -> NaturalPlanExample:
    """Convert one raw Meeting Planning record into the OA-aligned shape.

    Raises ``ValueError`` on malformed input.
    """
    try:
        constraints = ex["constraints"]
        dist_matrix = ex["dist_matrix"]
        context = ex["prompt_0shot"]
        golden_plan = ex["golden_plan"]
    except KeyError as exc:
        raise ValueError(f"missing required field: {exc.args[0]!r}") from exc

    if not isinstance(golden_plan, list):
        raise ValueError("golden_plan is not a list of step strings")
    schedule = parse_meeting_golden_plan([str(s) for s in golden_plan])
    if not schedule:
        raise ValueError("no meet steps parsed from golden_plan")

    gold = {"schedule": schedule}
    meta = {
        "task": "meeting",
        "num_people": int(ex.get("num_people", len(schedule))),
        "constraints": constraints,
        "dist_matrix": dist_matrix,
        "golden_plan": golden_plan,
        "difficulty": int(ex.get("num_people", len(schedule))),
    }
    return NaturalPlanExample(
        id=key, title=key, context=str(context), gold=gold, meta=meta
    )


def iter_meeting_examples(
    raw_dir: Path, *, limit: int | None = None
) -> Iterator[NaturalPlanExample]:
    """Stream Meeting Planning examples, skipping malformed records."""
    raw = _load_raw(Path(raw_dir) / MEETING_RAW_NAME)
    yielded = 0
    for key, ex in raw.items():
        try:
            out = parse_meeting_example(key, ex)
        except ValueError:
            continue
        yield out
        yielded += 1
        if limit is not None and yielded >= limit:
            return


ITERATORS = {"trip": iter_trip_examples, "meeting": iter_meeting_examples}
