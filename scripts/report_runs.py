"""Markdown report over ``data/runs/``.

Walks a runs root (default ``data/runs/``), groups runs by experiment
family (``gepa_<exp>_*``), and prints a Markdown report:

* one per-experiment table of *completed* runs (those with ``summary.json``),
  with NA-filled cells for fields a given run doesn't carry;
* per-experiment list of *incomplete* runs (have ``config.json`` +
  ``gepa_state.bin`` but no ``summary.json``) with a short state line
  and a ready-to-paste ``--resume`` command;
* a trailing "Ignored" section for dirs missing ``schema_path`` or with
  an unreadable / unrecognized config.

Usage:
    uv run python scripts/report_runs.py [--runs-dir DIR] [--filter GLOB]
                                         [--output PATH] [--full]
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob as globlib
import json
import re
import sys
from pathlib import Path
from typing import Any

CONFIG_NAME = "config.json"
STATE_NAME = "gepa_state.bin"
SUMMARY_NAME = "summary.json"
HOLDOUT_NAME = "holdout_scores.jsonl"
RUN_LOG_JSON = "run_log.json"
CANDIDATES_JSON = "candidates.json"
TASK_LM_JSONL = "task_lm.jsonl"

CURATED_COLUMNS: list[tuple[str, tuple[str, ...]]] = [
    ("run", ()),
    ("arm", ("arm",)),
    ("schema_kind", ("schema_kind",)),
    ("best_val", ("best_val_score",)),
    ("best_idx", ("best_candidate_idx",)),
    ("num_cands", ("num_candidates",)),
    ("metric_calls", ("total_metric_calls",)),
    ("full_val_evals", ("num_full_val_evals",)),
    ("holdout_mean", ("__headline__", "mean_score")),
    ("holdout_mean_nz", ("__headline__", "mean_score_nonzero")),
    ("holdout_median", ("__headline__", "median_score")),
    ("holdout_stdev", ("__headline__", "stdev_score")),
    ("holdout_n0", ("__headline__", "n_zero")),
    ("holdout_n1", ("__headline__", "n_perfect")),
    ("holdout_n", ("__headline__", "n")),
    ("sees_image", ("reflection_sees_image",)),
]

FLOAT_COLUMNS = {
    "best_val",
    "holdout_mean",
    "holdout_mean_nz",
    "holdout_median",
    "holdout_stdev",
}

EXP_RE = re.compile(r"^gepa_([A-Za-z0-9]+)_")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("data/runs"),
        help="Runs root directory (default: data/runs).",
    )
    p.add_argument(
        "--filter",
        type=str,
        default=None,
        help=(
            "Optional glob pattern, expanded via glob.glob. "
            "Example: '/abs/path/data/runs/gepa_scierc_*'. "
            "When omitted, all 'gepa_*' subdirectories of --runs-dir are "
            "enumerated (excluding the 'OLD' archive)."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output Markdown file (default: stdout).",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help=(
            "Emit a table per experiment with the union of every leaf key "
            "seen across summaries (NA-filled) instead of the curated set."
        ),
    )
    return p.parse_args(argv)


def discover_run_dirs(runs_dir: Path, pattern: str | None) -> list[Path]:
    """Find run directories under ``runs_dir``.

    With ``--filter``, returns sorted glob matches verbatim (each is then
    classified). Without it, walks ``runs_dir`` and treats any directory
    containing ``config.json`` as a leaf run dir (stops descending there).
    This handles both the flat layout (``runs_dir/gepa_<exp>_*/``) and
    nested layouts like ``runs_dir/<family>/<size>/<run>/``. ``OLD`` dirs
    are skipped anywhere in the tree.
    """
    if pattern is not None:
        return sorted(p for p in (Path(s) for s in globlib.glob(pattern)) if p.is_dir())
    if not runs_dir.is_dir():
        return []
    found: list[Path] = []

    def walk(d: Path) -> None:
        if d.name == "OLD":
            return
        if (d / CONFIG_NAME).exists():
            found.append(d)
            return
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for child in children:
            walk(child)

    walk(runs_dir)
    return found


def safe_load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def get_path(d: dict, keys: tuple[str, ...]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _md_escape(s: str, max_len: int = 80) -> str:
    """Make a string safe for a single Markdown table cell."""
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    s = s.replace("|", "\\|")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def fmt_cell(value: Any, column: str) -> str:
    if value is None:
        return "NA"
    if column in FLOAT_COLUMNS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        return _md_escape(value)
    return _md_escape(str(value))


def experiment_of(run: Path, runs_dir: Path) -> str:
    """Pick the experiment-group key for ``run``.

    Prefers the legacy ``gepa_<exp>_`` name prefix when it matches; otherwise
    falls back to the path segments between ``runs_dir`` and the run dir,
    joined with ``/``. Returns ``"_unknown"`` if the run sits directly under
    ``runs_dir`` with a non-matching name.
    """
    m = EXP_RE.match(run.name)
    if m:
        return m.group(1)
    try:
        rel = run.resolve().relative_to(runs_dir.resolve())
    except ValueError:
        return run.parent.name or "_unknown"
    parts = rel.parts[:-1]
    return "/".join(parts) if parts else "_unknown"


def flatten_leaves(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dotted keys, listing only leaf values."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten_leaves(v, key))
            else:
                out[key] = v
    return out


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out: list[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return out


def _headline_stats(summary: dict) -> dict:
    """Pick the primary-kind aggregate from the new summary shape.

    Current shape: ``summary['holdout']`` carries ``primary_kind`` and a
    ``scores`` map keyed by kind. Legacy fallbacks: top-level
    ``cross_scores`` + ``holdout_scores`` pair (pre-rename), and an even
    older bare ``holdout`` aggregate dict (pre-cross-schema).
    """
    holdout = summary.get("holdout") or {}
    scores = holdout.get("scores")
    if isinstance(scores, dict) and scores:
        primary = holdout.get("primary_kind")
        if primary and primary in scores:
            return scores[primary]
        return next(iter(scores.values()))
    legacy_cross = summary.get("cross_scores") or {}
    legacy_meta = summary.get("holdout_scores") or {}
    primary = legacy_meta.get("primary_kind")
    if primary and primary in legacy_cross:
        return legacy_cross[primary]
    if legacy_cross:
        return next(iter(legacy_cross.values()))
    return holdout


def build_curated_row(run: Path, summary: dict) -> list[str]:
    view = dict(summary)
    view["__headline__"] = _headline_stats(summary)
    cells: list[str] = []
    for column, keys in CURATED_COLUMNS:
        if column == "run":
            cells.append(run.name)
            continue
        cells.append(fmt_cell(get_path(view, keys), column))
    return cells


def build_full_table(
    completed: list[tuple[Path, dict]],
) -> tuple[list[str], list[list[str]]]:
    all_keys: set[str] = set()
    flat_per_run: list[tuple[Path, dict[str, Any]]] = []
    for run, summary in completed:
        flat = flatten_leaves(summary)
        flat_per_run.append((run, flat))
        all_keys.update(flat.keys())
    headers = ["run"] + sorted(all_keys)
    rows: list[list[str]] = []
    for run, flat in flat_per_run:
        row = [run.name]
        for k in headers[1:]:
            row.append(fmt_cell(flat.get(k), k))
        rows.append(row)
    return headers, rows


def schema_kind_of(config: dict) -> str | None:
    """Derive a run's schema kind from ``config["schema_path"]``'s stem.

    ``schema_kind`` is no longer stored as a separate field — it is always
    the schema-file basename without extension (e.g. ``scierc_strict.jsonc``
    → ``scierc_strict``).
    """
    schema_path = config.get("schema_path")
    if not schema_path:
        return None
    return Path(schema_path).stem


def gather_incomplete_state(run: Path, config: dict) -> dict[str, Any]:
    state: dict[str, Any] = {
        "arm": config.get("arm"),
        "schema_kind": schema_kind_of(config),
    }

    log = safe_load_json(run / RUN_LOG_JSON)
    state["iters"] = len(log) if isinstance(log, list) else 0

    cands = safe_load_json(run / CANDIDATES_JSON)
    state["cands"] = len(cands) if isinstance(cands, list) else 0

    task_lm = run / TASK_LM_JSONL
    state["task_lm_lines"] = count_lines(task_lm) if task_lm.exists() else 0
    state["budget"] = config.get("max_metric_calls")

    holdout = run / HOLDOUT_NAME
    state["holdout_done"] = count_lines(holdout) if holdout.exists() else 0
    test_path = config.get("test_path")
    if test_path:
        tp = Path(test_path)
        state["holdout_total"] = count_lines(tp) if tp.exists() else None
    else:
        state["holdout_total"] = None

    summary_corrupt = (
        (run / SUMMARY_NAME).exists()
        and safe_load_json(run / SUMMARY_NAME) is None
    )
    state["summary_corrupt"] = summary_corrupt

    mtimes: list[float] = []
    for fname in (RUN_LOG_JSON, TASK_LM_JSONL, HOLDOUT_NAME):
        fp = run / fname
        if fp.exists():
            mtimes.append(fp.stat().st_mtime)
    if mtimes:
        state["last_activity"] = dt.datetime.fromtimestamp(
            max(mtimes)
        ).isoformat(timespec="seconds")
    else:
        state["last_activity"] = None

    return state


def resume_command(run: Path, exp: str, repo_root: Path) -> str:
    # When grouping is path-based (e.g. "synra_nl/small"), the runner script
    # is keyed by the first segment only.
    exp_head = exp.split("/", 1)[0]
    runner = repo_root / "scripts" / f"run_gepa_{exp_head}.py"
    if not runner.exists():
        return f"(unknown runner for experiment {exp!r})"
    return f"uv run python scripts/run_gepa_{exp_head}.py --resume {run.resolve()}"


def render_incomplete(run: Path, state: dict, exp: str, repo_root: Path) -> list[str]:
    arm = state["arm"] if state["arm"] is not None else "NA"
    sk = state["schema_kind"] if state["schema_kind"] is not None else "NA"
    budget = state["budget"] if state["budget"] is not None else "?"
    ho_total = state["holdout_total"] if state["holdout_total"] is not None else "?"
    last = state["last_activity"] or "?"
    notes = " summary=corrupt" if state["summary_corrupt"] else ""

    lines = [f"- `{run.name}`"]
    lines.append(f"  - arm={arm}  schema_kind={sk}")
    lines.append(
        f"  - iters={state['iters']}  cands={state['cands']}  "
        f"task_lm_lines={state['task_lm_lines']}/{budget}  "
        f"holdout={state['holdout_done']}/{ho_total}  "
        f"last={last}{notes}"
    )
    lines.append(f"  - resume: `{resume_command(run, exp, repo_root)}`")
    return lines


def classify_runs(
    run_dirs: list[Path],
    runs_dir: Path,
) -> tuple[dict[str, dict[str, list]], list[tuple[Path, str]]]:
    """Return (by_experiment, ignored).

    by_experiment[exp] = {"completed": [(run, summary)], "incomplete": [(run, config)]}
    ignored = [(run, reason)]
    """
    by_exp: dict[str, dict[str, list]] = {}
    ignored: list[tuple[Path, str]] = []

    for run in run_dirs:
        if not run.is_dir():
            continue
        if run.name == "OLD":
            continue

        cfg_path = run / CONFIG_NAME
        if not cfg_path.exists():
            ignored.append((run, "no config.json"))
            continue
        config = safe_load_json(cfg_path)
        if config is None:
            ignored.append((run, "corrupt config.json"))
            continue
        if "schema_path" not in config:
            ignored.append((run, "missing `schema_path` in config"))
            continue

        exp = experiment_of(run, runs_dir)

        bucket = by_exp.setdefault(exp, {"completed": [], "incomplete": []})

        summary_path = run / SUMMARY_NAME
        summary = safe_load_json(summary_path) if summary_path.exists() else None
        if summary is not None:
            summary["schema_kind"] = schema_kind_of(config)
            bucket["completed"].append((run, summary))
        else:
            bucket["incomplete"].append((run, config))

    for bucket in by_exp.values():
        bucket["completed"].sort(key=lambda t: t[0].name)
        bucket["incomplete"].sort(key=lambda t: t[0].name)

    return by_exp, ignored


def render_report(
    by_exp: dict[str, dict[str, list]],
    ignored: list[tuple[Path, str]],
    runs_dir: Path,
    pattern: str | None,
    repo_root: Path,
    full: bool,
) -> str:
    out: list[str] = []
    now = dt.datetime.now().isoformat(timespec="seconds")
    out.append("# Runs report")
    out.append("")
    out.append(f"_Generated {now}_")
    out.append(f"_Runs root: `{runs_dir}`_")
    if pattern is not None:
        out.append(f"_Filter: `{pattern}`_")
    out.append("")

    if not by_exp and not ignored:
        out.append("_(no run dirs matched)_")
        out.append("")
        return "\n".join(out)

    for exp in sorted(by_exp.keys()):
        bucket = by_exp[exp]
        completed = bucket["completed"]
        incomplete = bucket["incomplete"]

        out.append(f"## {exp}")
        out.append("")

        out.append(f"### Completed ({len(completed)})")
        out.append("")
        if not completed:
            out.append("_None._")
            out.append("")
        else:
            if full:
                headers, rows = build_full_table(completed)
            else:
                headers = [c for c, _ in CURATED_COLUMNS]
                rows = [build_curated_row(run, summary) for run, summary in completed]
            out.extend(md_table(headers, rows))
            out.append("")

        out.append(f"### Incomplete ({len(incomplete)})")
        out.append("")
        if not incomplete:
            out.append("_None._")
            out.append("")
        else:
            for run, config in incomplete:
                state = gather_incomplete_state(run, config)
                out.extend(render_incomplete(run, state, exp, repo_root))
            out.append("")

    if ignored:
        out.append("## Ignored")
        out.append("")
        for run, reason in sorted(ignored, key=lambda t: t[0].name):
            try:
                rel = run.relative_to(repo_root)
                shown = str(rel)
            except ValueError:
                shown = str(run)
            out.append(f"- `{shown}` — {reason}")
        out.append("")

    out.append(
        "_Note: `task_lm_lines` is a proxy for spent metric calls "
        "(one line per task-LM request)._"
    )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = args.runs_dir.resolve()
    repo_root = Path(__file__).resolve().parent.parent

    run_dirs = discover_run_dirs(runs_dir, args.filter)
    by_exp, ignored = classify_runs(run_dirs, runs_dir)
    text = render_report(
        by_exp=by_exp,
        ignored=ignored,
        runs_dir=runs_dir,
        pattern=args.filter,
        repo_root=repo_root,
        full=args.full,
    )

    if args.output is not None:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
