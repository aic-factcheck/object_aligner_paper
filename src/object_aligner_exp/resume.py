"""Resume support for the ``run_gepa_*`` scripts.

When a runner is invoked with ``--resume <run_dir>`` instead of starting a
fresh optimization it:

* loads the prior ``config.json`` from that directory,
* inherits CLI args the user didn't re-specify (``--arm``, ``--config``,
  ``--schema``, ``--reflection-sees-image``) from the recorded values, so
  ``--resume <path>`` alone is enough to keep an apples-to-apples
  continuation,
* hands the directory back to ``gepa.optimize`` so GEPA picks up its own
  ``gepa_state.bin`` checkpoint and continues iterating, and
* skips the post-opt holdout phase if it's already complete.

GEPA's own resume mechanism is automatic: passing ``run_dir`` to
``gepa.optimize`` causes it to load ``gepa_state.bin`` from that directory
when present (see ``gepa.core.state.initialize_gepa_state``). The helpers
here just glue the runner-side bookkeeping (config snapshot, arg inheritance,
holdout idempotency) on top.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any


_CONFIG_NAME = "config.json"
_STATE_NAME = "gepa_state.bin"
_HOLDOUT_NAME = "holdout_scores.jsonl"
_SUMMARY_NAME = "summary.json"


def write_fresh_run_dir(
    runs_root: Path,
    run_name: str,
    fresh_config: dict[str, Any],
) -> Path:
    """Create a new timestamped run directory and write ``config.json``.

    Returns the absolute path to the new directory.
    """
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _CONFIG_NAME).write_text(
        json.dumps(fresh_config, indent=2, ensure_ascii=False)
    )
    return run_dir


def peek_resume_config(run_dir: Path) -> dict[str, Any]:
    """Load and structurally validate ``run_dir/config.json``.

    Verifies that ``run_dir`` looks like a runner-managed directory (it has
    ``config.json`` and ``gepa_state.bin``), and returns the recorded config
    as a dict. Performs no value-level validation — the caller decides which
    keys to enforce, inherit, or warn on.

    Raises ``FileNotFoundError`` if any required file is missing.
    """
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(
            f"--resume target {run_dir} does not exist or is not a directory."
        )
    cfg_path = run_dir / _CONFIG_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"--resume target {run_dir} has no {_CONFIG_NAME}; "
            f"not a runner-managed directory."
        )
    if not (run_dir / _STATE_NAME).exists():
        raise FileNotFoundError(
            f"--resume target {run_dir} has no {_STATE_NAME}; "
            f"GEPA never wrote a checkpoint here. Refusing to resume into a "
            f"half-populated directory (would clobber existing JSONL logs)."
        )
    return json.loads(cfg_path.read_text())


def warn_config_drift(
    prior_config: dict[str, Any],
    *,
    keys: dict[str, Any],
) -> None:
    """Print a one-line warning for any key whose current value differs from
    the recorded value in ``prior_config``.

    Intended for non-fatal drift such as LM endpoint URLs that may have
    legitimately changed since the original run.
    """
    for key, current in keys.items():
        recorded = prior_config.get(key)
        if recorded != current:
            print(
                f"[resume] warning: {key!r} changed since original run "
                f"(recorded={recorded!r}, current={current!r}); proceeding "
                f"with current value."
            )


def update_metric_call_cap(
    run_dir: Path,
    prior_config: dict[str, Any],
    new_cap: int | None,
) -> dict[str, Any]:
    """If ``new_cap`` is given and differs from ``prior_config["max_metric_calls"]``,
    rewrite ``config.json`` to record the new cap and append a ``resumes`` entry.

    Passing ``new_cap=None`` means "the user didn't re-specify --max-metric-calls
    on resume" and leaves the saved cap untouched — so a bare ``--resume <dir>``
    keeps the original budget instead of silently snapping it to the runner's
    CLI default.

    Returns the (possibly updated) config dict. Mutates ``prior_config`` in
    place so callers can read the new ``max_metric_calls`` directly.
    """
    if new_cap is None:
        return prior_config
    old_cap = prior_config.get("max_metric_calls")
    if old_cap == new_cap:
        return prior_config
    print(
        f"[resume] extending max_metric_calls: {old_cap} -> {new_cap} "
        f"(per CLI; recording in config.json)"
    )
    prior_config["max_metric_calls"] = new_cap
    resumes = prior_config.setdefault("resumes", [])
    resumes.append(
        {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "old_cap": old_cap,
            "new_cap": new_cap,
        }
    )
    (run_dir / _CONFIG_NAME).write_text(
        json.dumps(prior_config, indent=2, ensure_ascii=False)
    )
    return prior_config


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file. Returns 0 if the file is missing."""
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def holdout_already_done(
    run_dir: Path,
    *,
    best_idx: int | None,
    expected_n_test: int,
) -> bool:
    """Return True iff a complete holdout for the given ``best_idx`` already exists.

    Used to make the post-opt phase idempotent on resume: if the run already
    finished and produced a matching ``summary.json`` + ``holdout_scores.jsonl``,
    skip re-running holdout. Re-runs holdout whenever ``best_idx`` differs
    (budget extension may have promoted a new candidate).
    """
    if best_idx is None:
        return False
    summary_path = run_dir / _SUMMARY_NAME
    holdout_path = run_dir / _HOLDOUT_NAME
    if not summary_path.exists() or not holdout_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    if summary.get("best_candidate_idx") != best_idx:
        return False
    # Current shape: a "holdout" block carrying path/dir, primary_kind,
    # and a "scores" map of per-kind aggregates. Legacy shapes used a
    # top-level "cross_scores" + "holdout_scores" pair, or an even older
    # bare "holdout" aggregate dict — accept any of them as "done".
    has_holdout = bool(
        summary.get("holdout")
        or summary.get("cross_scores")
        or summary.get("holdout_scores")
    )
    if not has_holdout:
        return False
    return _count_jsonl_lines(holdout_path) == expected_n_test


def inherit_cross_schema(
    cli_value: list[Path] | None,
    prior_config: dict[str, Any],
) -> list[Path]:
    """Resolve ``--cross-schema`` on resume.

    If the user passed ``--cross-schema`` on the CLI, that wins. Otherwise
    inherit ``prior_config["cross_schema_paths"]`` so a bare ``--resume``
    keeps the original cross-schema list (and the post-opt holdout refresh
    rescues those scores from the cached responses).
    """
    if cli_value is not None:
        return cli_value
    prior = prior_config.get("cross_schema_paths") or []
    if not prior:
        return []
    paths = [Path(p) for p in prior]
    print(f"[resume] inheriting --cross-schema={[str(p) for p in paths]} from config.json")
    return paths


def resume_banner(run_dir: Path, prior_config: dict[str, Any]) -> None:
    """One-line user-facing print announcing the resume."""
    started = prior_config.get("_started_at") or "?"
    n_resumes = len(prior_config.get("resumes", []))
    print(
        f"[resume] {run_dir} (started={started}, "
        f"prior_resumes={n_resumes}, current_ts={time.strftime('%Y-%m-%dT%H:%M:%S')})"
    )
