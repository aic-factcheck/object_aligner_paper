"""Recompute a run leaf's OA holdout on a larger split (e.g. ``test_full.jsonl``).

Every GEPA run leaf under ``data/runs/gepa_*/<arm>/<leg>/<model>/s<seed>_<ts>/``
carries a ``summary.json`` whose ``holdout`` block was scored on the small
held-out split (``test.jsonl``, ~200 examples). This script re-scores the
**already-optimized best prompt** on the larger ``test_full.jsonl`` split
*without* re-running GEPA — only ``task_lm`` is queried (no reflection LM).

It is dataset-agnostic: everything needed is read from the leaf's own
``config.json`` (``arm``, ``schema_path``, ``cross_schema_paths``, ``test_path``)
and ``summary.json`` (``best_prompt``). The holdout machinery
(:func:`evaluate_text_holdout`) is the same one ``run_gepa_*.py`` uses for the
small split, so the two numbers are directly comparable.

Nothing is overwritten in place — the full-split results are written **additively**
under ``*_full`` filenames; the original ``test`` artifacts are left untouched:

  * ``summary_full.json``         <- the run's ``summary.json`` with its holdout
    block swapped for the test_full scores (``summary.json`` itself is untouched)
  * ``holdout_scores_full.jsonl`` <- per-example test_full scores
  * test_full task_lm calls are logged to ``task_lm_full.jsonl``
    (the original ``task_lm.jsonl`` optimization log is left untouched)

Re-running is a no-op once ``summary_full.json`` exists (pass ``--force`` to
recompute anyway). The model served at ``$OAEXP_TASK_LM_BASE_URL`` must match
the model recorded in the leaf's ``config.json`` — otherwise the leaf would be
scored by the wrong model, so a mismatch aborts.

Usage::

    uv run python scripts/evaluate_full.py \\
        --run-dir data/runs/gepa_wec_eng/oa_score/ra/gemma4-e4b/s0_2026... \\
        --split   data/wec_eng/splits/gepa_wec_eng/main/test_full.jsonl \\
        --task-lm-concurrency 16 --task-temperature 0.3

The ``evaluate_*.sh`` wrappers drive this over every existing leaf of a dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets import load_jsonl
from object_aligner_exp.holdout_eval import (
    build_cross_specs_from_paths,
    evaluate_text_holdout,
)
from object_aligner_exp.llm import (
    build_task_lm,
    check_lm_alive,
    with_conversation_log,
)
from object_aligner_exp.oa import wl_integration_for_arm
from object_aligner_exp.report import render_run
from object_aligner_exp.schemas import load_schema_from_path


def _resolve_split(run_dir: Path, cfg_test_path: str | None,
                   split: Path | None, split_name: str) -> Path:
    """Pick the split file to score on.

    ``--split`` wins outright. Otherwise the new split is the sibling of the
    run's recorded ``test_path`` named ``--split-name`` (default
    ``test_full.jsonl``).
    """
    if split is not None:
        return split
    if not cfg_test_path:
        raise SystemExit(
            "[error] config.json has no 'test_path' and --split was not given; "
            "cannot locate the larger split."
        )
    return Path(cfg_test_path).with_name(split_name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run leaf directory to recompute (holds config.json + summary.json).")
    p.add_argument("--split", type=Path, default=None,
                   help="Explicit split JSONL to score on (overrides --split-name).")
    p.add_argument("--split-name", type=str, default="test_full.jsonl",
                   help="Filename of the larger split, looked up next to the run's "
                        "recorded test_path (default: %(default)s).")
    p.add_argument("--config", type=Path, default=None,
                   help="LM config YAML (default: config/experiment.yaml or $OAEXP_CONFIG). "
                        "model/base_url still come from $OAEXP_TASK_LM_* env.")
    p.add_argument("--task-lm-concurrency", type=int, default=16,
                   help="Max concurrent task_lm calls (default: %(default)s; -1 = full fan-out).")
    p.add_argument("--task-temperature", type=float, default=0.3,
                   help="Sampling temperature for the task LM (default: %(default)s).")
    p.add_argument("--task-max-tokens", type=int, default=None,
                   help="Override task_lm max_tokens (default: the run's recorded cap, "
                        "else YAML, else 0 = no cap).")
    p.add_argument("--no-task-call-progress", dest="task_call_progress",
                   action="store_false",
                   help="Suppress the per-task_lm-call one-line status print.")
    p.set_defaults(task_call_progress=True)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if summary_full.json already exists.")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    summary_full_path = run_dir / "summary_full.json"
    holdout_full_path = run_dir / "holdout_scores_full.jsonl"

    if not cfg_path.exists():
        raise SystemExit(f"[error] no config.json in {run_dir}")
    if summary_full_path.exists() and not args.force:
        print(f"[skip] already recomputed (summary_full.json present): {run_dir}")
        return
    if not summary_path.exists():
        raise SystemExit(f"[error] no summary.json in {run_dir}")

    run_cfg = json.loads(cfg_path.read_text())
    summary = json.loads(summary_path.read_text())

    arm = run_cfg.get("arm") or summary.get("arm")
    schema_path = run_cfg.get("schema_path")
    cross_schema_paths = run_cfg.get("cross_schema_paths") or []
    if not arm or not schema_path:
        raise SystemExit(f"[error] config.json missing 'arm'/'schema_path' in {run_dir}")

    best_prompt = summary.get("best_prompt")
    if not best_prompt:
        raise SystemExit(
            f"[skip] no best_prompt in summary.json ({run_dir}) — "
            "the original run produced no best candidate; nothing to score."
        )

    split = _resolve_split(run_dir, run_cfg.get("test_path"), args.split, args.split_name)
    if not split.exists():
        raise SystemExit(f"[error] split not found: {split}")

    # LM config: live endpoint/model from env; faithful max_tokens from the run.
    cfg = ExpConfig.from_yaml(args.config) if args.config else ExpConfig()
    recorded = run_cfg.get("task_lm") or {}
    recorded_model = recorded.get("model")
    if recorded_model and recorded_model != cfg.task_lm.model:
        raise SystemExit(
            f"[error] model mismatch: leaf was scored with {recorded_model!r} but "
            f"$OAEXP_TASK_LM_MODEL resolves to {cfg.task_lm.model!r}. Launch the "
            f"matching model server before re-scoring {run_dir}."
        )
    if args.task_max_tokens is not None:
        task_max_tokens = args.task_max_tokens
    elif recorded.get("max_tokens") is not None:
        task_max_tokens = recorded["max_tokens"]
    else:
        task_max_tokens = cfg.task_lm.max_tokens if cfg.task_lm.max_tokens is not None else 0

    check_lm_alive(cfg.task_lm, label="task_lm")

    schema = load_schema_from_path(schema_path)
    examples = load_jsonl(split)
    gold_by_id = {str(ex["id"]): ex["gold"] for ex in examples}
    primary_kind, cross_specs = build_cross_specs_from_paths(
        Path(schema_path), [Path(c) for c in cross_schema_paths], gold_by_id=gold_by_id
    )
    cross_kinds_str = ", ".join(s.kind for s in cross_specs) or "(none)"

    task_lm_callable = with_conversation_log(
        build_task_lm(
            cfg.task_lm,
            temperature=args.task_temperature,
            max_tokens=task_max_tokens,
            json_mode=True,
        ),
        run_dir / "task_lm_full.jsonl",
        label="task_lm",
        print_progress=args.task_call_progress,
    )

    # Additive: write the test_full results under *_full names. The original
    # ``test`` artifacts (summary.json, holdout_scores.jsonl) are never touched.
    print(
        f"[holdout-full] scoring best prompt on {len(examples)} examples from "
        f"{split} (arm={arm!r}, primary={primary_kind!r}, cross-schema={cross_kinds_str}, "
        f"concurrency={args.task_lm_concurrency}, temp={args.task_temperature})"
    )
    scores = evaluate_text_holdout(
        examples,
        system_prompt=best_prompt,
        task_lm=task_lm_callable,
        schema=schema,
        primary_kind=primary_kind,
        cross_specs=cross_specs,
        wl_integration=wl_integration_for_arm(arm),
        out_path=holdout_full_path,
        desc=f"{run_dir.name}/holdout_full",
        max_workers=args.task_lm_concurrency,
    )

    # Keep every other summary key (best_*, task, etc.); swap the holdout block,
    # and write the result to summary_full.json (leaving summary.json = test).
    summary["holdout"] = {
        "path": str(split),
        "primary_kind": primary_kind,
        "scores": scores,
    }
    summary_full_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    try:
        report_path = render_run(run_dir, verbose=False)
        print(f"[render] {report_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[render] skipped (render failed: {exc!r})", file=sys.stderr)

    print(json.dumps(summary["holdout"], indent=2, ensure_ascii=False))
    print(f"[done] wrote summary_full.json on {split.name} for {run_dir}")


if __name__ == "__main__":
    main()
