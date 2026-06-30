"""GEPA_SCIERC baseline (Arm 4, no GEPA): task LM zero-shot, scored with OA.

Default split is the pilot's held-out ``test.jsonl`` — the same file
``scripts/run_gepa_scierc.py`` evaluates ``best_candidate`` against
post-optimization, so this baseline number is the apples-to-apples partner
for the GEPA-run ``holdout`` block.

Saves results to::

    data/runs/gepa_scierc_baseline_<ts>/
        config.json   — seed prompt + LM config used
        scores.jsonl  — one row per example: {id, score, response, feedback}
        summary.json  — aggregate stats

Usage::

    uv run python scripts/baseline_eval_scierc.py
    uv run python scripts/baseline_eval_scierc.py --split data/scierc/splits/gepa_scierc/pilot/test.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets import load_jsonl
from object_aligner_exp.evaluator import OaFeedbackEvaluator, make_data_inst
from object_aligner_exp.holdout_eval import (
    aggregate_by_kind,
    build_cross_specs_from_paths,
    parse_response,
    score_cross_specs,
)
from object_aligner_exp.llm import build_task_lm, check_lm_alive
from object_aligner_exp.prompts import load_seed_prompt_from_path
from object_aligner_exp.schemas import load_schema_from_path


def _run_name_suffix(schema_kind: str, dataset: str) -> str:
    if schema_kind == dataset:
        return ""
    stripped = schema_kind.removeprefix(dataset).lstrip("_")
    return f"_{stripped or schema_kind}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "LM config YAML to load (default: config/experiment.yaml, or "
            "$OAEXP_CONFIG if set). Example: config/experiment_max.yaml."
        ),
    )
    p.add_argument(
        "--seed-prompt",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "System prompt text file used verbatim as the baseline "
            "(default: data/scierc/seed_prompt.txt)."
        ),
    )
    p.add_argument(
        "--schema",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "Path to the OA schema (.jsonc) to score against. "
            "The file stem is used as the run-name suffix and as the dict "
            "key under summary['holdout']['scores']. "
            "In-repo variants for SciERC: data/scierc/schemas/scierc.jsonc "
            "(RA on) and scierc_strict.jsonc (RA off)."
        ),
    )
    p.add_argument(
        "--cross-schema",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        dest="cross_schema",
        help=(
            "Additional schema (.jsonc) to also score each response "
            "against. Repeat to add several; per-schema aggregates land "
            "in summary['holdout']['scores'] and per-row 'cross_scores'."
        ),
    )
    p.add_argument("--split", type=Path, default=None,
                   help="JSONL file produced by prepare_scierc.py "
                        "(default: <gepa_scierc/pilot/test.jsonl>)")
    p.add_argument("--run-name", type=str, default=None,
                   help="prefix for the run dir under data/runs/ "
                        "(default: gepa_scierc_baseline[_<schema>])")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Override task_lm max_tokens (default: YAML, else 0; 0 = no cap).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None,
                   help="only evaluate the first N examples (smoke testing)")
    p.add_argument(
        "--stream-task-lm",
        action="store_true",
        help=(
            "Stream task_lm response chunks to stderr in real time so a "
            "stalled or slow sample is visible as it happens."
        ),
    )
    args = p.parse_args()

    cfg = ExpConfig.from_yaml(args.config) if args.config else ExpConfig()
    split = args.split or cfg.gepa_scierc_pilot / "test.jsonl"
    schema_path = args.schema
    schema_kind = schema_path.stem
    cross_paths = args.cross_schema or []
    run_name = args.run_name or (
        f"gepa_scierc_baseline{_run_name_suffix(schema_kind, 'scierc')}"
    )

    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    elif cfg.task_lm.max_tokens is not None:
        max_tokens = cfg.task_lm.max_tokens
    else:
        max_tokens = 0

    # Fail fast if the local server is down rather than 30s into the loop.
    check_lm_alive(cfg.task_lm, label="task_lm")

    schema = load_schema_from_path(schema_path)
    seed_prompt_path = args.seed_prompt or cfg.scierc_seed_prompt_path
    seed_system_prompt = load_seed_prompt_from_path(seed_prompt_path)

    examples = load_jsonl(split)
    if args.limit is not None:
        examples = examples[: args.limit]
    print(f"[load] {len(examples)} examples from {split}")

    gold_by_id = {str(ex["id"]): ex["gold"] for ex in examples}
    primary_kind, cross_specs = build_cross_specs_from_paths(
        schema_path, cross_paths, gold_by_id=gold_by_id
    )
    kinds_list: list[str] = [primary_kind]
    for spec in cross_specs:
        if spec.kind not in kinds_list:
            kinds_list.append(spec.kind)

    task_lm = build_task_lm(
        cfg.task_lm,
        temperature=args.temperature,
        max_tokens=max_tokens,
        json_mode=True,
        stream_to_stderr=args.stream_task_lm,
    )
    evaluator = OaFeedbackEvaluator(schema)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_root / f"{run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config_path": str(args.config) if args.config else None,
                "split": str(split),
                "n": len(examples),
                "seed_system_prompt": seed_system_prompt,
                "seed_prompt_path": str(seed_prompt_path),
                "task_lm": asdict(cfg.task_lm),
                "schema_path": str(schema_path),
                "cross_schema_paths": [str(p) for p in cross_paths],
                "temperature": args.temperature,
                "max_tokens": max_tokens,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    scores_path = run_dir / "scores.jsonl"
    rows: list[dict] = []
    with scores_path.open("w") as f:
        for ex in tqdm(examples, unit="ex", desc=run_name):
            messages = [
                {"role": "system", "content": seed_system_prompt},
                {"role": "user", "content": ex["context"]},
            ]
            response = task_lm(messages)

            data_inst = make_data_inst(
                context=ex["context"], gold=ex["gold"], sample_id=ex.get("id")
            )
            result = evaluator(data_inst, response)

            row: dict = {
                "id": ex["id"],
                "score": result.score,
                "response": response,
                "feedback": result.feedback,
            }
            if cross_specs:
                row["cross_scores"] = score_cross_specs(
                    parse_response(response),
                    primary_kind=primary_kind,
                    primary_score=float(result.score),
                    cross_specs=cross_specs,
                    example_id=str(ex.get("id")),
                )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    scores = aggregate_by_kind(
        rows, kinds=kinds_list, primary_kind=primary_kind
    )
    summary = {
        "holdout": {
            "path": str(split),
            "primary_kind": primary_kind,
            "scores": scores,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] run dir: {run_dir}")


if __name__ == "__main__":
    main()
