"""Sentence-ordering optimization: GEPA with an OA-based reward.

A thin sibling of ``scripts/run_gepa_planbench.py`` (same arms, adapter,
holdout, resume — all reused unchanged). The experiment contrasts OA's
order-aware list alignment (``--schema .../sentence_ordering_fixed.jsonc``)
against the order-blind Hungarian ablation
(``.../sentence_ordering_hungarian.jsonc``). Because the output is an index
permutation, Hungarian scores ≈1.0 for any permutation — it is the built-in
no-order-signal control, so the fixed-vs-Hungarian gap is pure order. Run both
arms back-to-back and cross-score (see ``run_sentence_ordering.sh``).

One corpus-agnostic schema pair is shared by both corpora (ROCStories, arXiv);
the corpus is selected purely by ``--split-dir`` / ``--seed-prompt`` (default:
ROCStories). Three arms by ``--arm``: ``oa_score`` / ``oa_feedback`` /
``oa_none``. PMR + Kendall's τ (for SOTA comparison) are computed separately by
``scripts/eval_sentence_ordering.py``.

Usage::

    uv run python scripts/run_gepa_sentence_ordering.py --arm oa_feedback \\
        --schema data/sentence_ordering/schemas/sentence_ordering_fixed.jsonc \\
        --cross-schema data/sentence_ordering/schemas/sentence_ordering_hungarian.jsonc
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

import gepa

from object_aligner_exp.config import ExpConfig
from object_aligner_exp.datasets import load_jsonl
from object_aligner_exp.oa import wl_integration_for_arm
from object_aligner_exp.evaluator import (
    OaFeedbackEvaluator,
    OaFeedbackTopK1Evaluator,
    OaFeedbackTopK10Evaluator,
    OaFeedbackTopKAllEvaluator,
    OaGoldEvaluator,
    OaNoneEvaluator,
    OaScoreOnlyEvaluator,
    OaSemanticFeedbackEvaluator,
    make_data_inst,
)
from object_aligner_exp.gepa_adapter import ParallelDefaultAdapter
from object_aligner_exp.holdout_eval import (
    build_cross_specs_from_paths,
    evaluate_text_holdout,
    rebuild_summary_from_disk,
)
from object_aligner_exp.llm import (
    TaskLMUnreachable,
    build_reflection_lm,
    build_task_lm,
    check_lm_alive,
    make_filtered_run_logger,
    with_conversation_log,
)
from object_aligner_exp.prompts import load_seed_prompt_from_path
from object_aligner_exp.report import RenderOnStateSaved, render_run
from object_aligner_exp.resume import (
    holdout_already_done,
    inherit_cross_schema,
    peek_resume_config,
    resume_banner,
    update_metric_call_cap,
    warn_config_drift,
    write_fresh_run_dir,
)
from object_aligner_exp.schemas import load_schema_from_path


SYSTEM_COMPONENT = "system_prompt"

ARMS = {
    "oa_score": OaScoreOnlyEvaluator,
    "oa_feedback": OaFeedbackEvaluator,
    "oa_feedback_ra": OaSemanticFeedbackEvaluator,
    "oa_none": OaNoneEvaluator,
    # baseline: OA score + full gold output (naive upper-information control).
    "oa_gold": OaGoldEvaluator,
    # oa_feedback at varying top_k (max corrections shown); same score as
    # oa_feedback, which uses OA's default top_k=5.
    "oa_feedback1": OaFeedbackTopK1Evaluator,
    "oa_feedback10": OaFeedbackTopK10Evaluator,
    "oa_feedback_all": OaFeedbackTopKAllEvaluator,
    # WL blend variants (RA-leg only; score with wl_integration="blend").
    "oa_score_blend": OaScoreOnlyEvaluator,
    "oa_feedback_blend": OaFeedbackEvaluator,
    "oa_feedback_ra_blend": OaSemanticFeedbackEvaluator,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=list(ARMS), default=None,
                   help="GEPA arm (required for fresh runs; inherited on --resume).")
    p.add_argument("--schema", type=Path, default=None, metavar="PATH",
                   help="OA schema (.jsonc). The stem (sentence_ordering_fixed/"
                        "sentence_ordering_hungarian) is the run-name suffix and "
                        "the key under holdout['scores'].")
    p.add_argument("--cross-schema", type=Path, action="append", default=None,
                   metavar="PATH", dest="cross_schema",
                   help="Additional schema (.jsonc) to also score the holdout "
                        "against; repeat to add several.")
    p.add_argument("--seed-prompt", type=Path, default=None, metavar="PATH",
                   help="Seed system prompt (default: the ROCStories seed_prompt.txt).")
    p.add_argument("--config", type=Path, default=None,
                   help="LM config YAML (default: config/experiment.yaml or "
                        "$OAEXP_CONFIG).")
    p.add_argument("--split-dir", type=Path, default=None,
                   help="Dir with train/val/test.jsonl (default: gepa_rocstories/main).")
    p.add_argument("--run-name", type=str, default=None,
                   help="prefix for the run dir (default: gepa_sentence_ordering_<arm>_<schema>)")
    p.add_argument("--max-metric-calls", type=int, default=None,
                   help="GEPA metric-call budget (fresh default 300).")
    p.add_argument("--reflection-minibatch-size", type=int, default=3)
    p.add_argument("--task-temperature", type=float, default=0.0)
    p.add_argument("--task-max-tokens", type=int, default=None)
    p.add_argument("--reflection-temperature", type=float, default=1.0)
    p.add_argument("--reflection-max-tokens", type=int, default=None)
    p.add_argument("--task-lm-concurrency", type=int, default=1,
                   help="Max concurrent task_lm calls per batch (1=sequential).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-prompts", action="store_true")
    p.add_argument("--no-task-call-progress", dest="task_call_progress",
                   action="store_false")
    p.set_defaults(task_call_progress=True)
    p.add_argument("--stream-task-lm", action="store_true")
    p.add_argument("--resume", type=Path, default=None,
                   help="Existing run_dir to resume.")
    args = p.parse_args()

    prior_cfg: dict | None = None
    if args.resume is not None:
        run_dir = args.resume.resolve()
        prior_cfg = peek_resume_config(run_dir)
        if args.arm is None:
            args.arm = prior_cfg["arm"]
            print(f"[resume] inheriting --arm={args.arm!r} from config.json")
        elif args.arm != prior_cfg["arm"]:
            p.error(f"--arm mismatch: prior={prior_cfg['arm']!r}, given={args.arm!r}.")
        prior_schema_path = prior_cfg.get("schema_path")
        if prior_schema_path is None:
            p.error("resume config.json has no 'schema_path'; re-launch fresh.")
        prior_schema_path = Path(prior_schema_path)
        if args.schema is None:
            args.schema = prior_schema_path
            print(f"[resume] inheriting --schema={args.schema} from config.json")
        elif args.schema.resolve() != prior_schema_path.resolve():
            p.error(f"--schema mismatch: prior={prior_schema_path}, given={args.schema}.")
        if args.config is None and prior_cfg.get("config_path"):
            args.config = Path(prior_cfg["config_path"])
            print(f"[resume] inheriting --config={args.config} from config.json")
        if args.run_name:
            print(f"[resume] ignoring --run-name={args.run_name!r}")
    else:
        if args.arm is None:
            p.error("--arm is required (omit only when using --resume)")
        if args.schema is None:
            p.error("--schema PATH is required for fresh runs; pass e.g. "
                    "data/sentence_ordering/schemas/sentence_ordering_fixed.jsonc")

    args.cross_schema = inherit_cross_schema(args.cross_schema, prior_cfg or {})

    cfg = ExpConfig.from_yaml(args.config) if args.config else ExpConfig()

    def _resolve_tokens(cli_val: int | None, yaml_val: int | None, default: int) -> int:
        if cli_val is not None:
            return cli_val
        if yaml_val is not None:
            return yaml_val
        return default

    task_max_tokens = _resolve_tokens(args.task_max_tokens, cfg.task_lm.max_tokens, 0)
    reflection_max_tokens = _resolve_tokens(
        args.reflection_max_tokens, cfg.reflection_lm.max_tokens, 0
    )

    schema_path = args.schema
    schema_kind = schema_path.stem
    run_name = args.run_name or f"gepa_sentence_ordering_{args.arm}_{schema_kind}"

    check_lm_alive(cfg.task_lm, label="task_lm")
    check_lm_alive(cfg.reflection_lm, label="reflection_lm")

    schema = load_schema_from_path(schema_path)

    if prior_cfg is not None:
        warn_config_drift(
            prior_cfg,
            keys={
                "task_lm": asdict(cfg.task_lm),
                "reflection_lm": asdict(cfg.reflection_lm),
            },
        )
        resume_banner(run_dir, prior_cfg)
        prior_cfg = update_metric_call_cap(run_dir, prior_cfg, args.max_metric_calls)
        split_dir = Path(prior_cfg["split_dir"])
        train_path = Path(prior_cfg["train_path"])
        val_path = Path(prior_cfg["val_path"])
        test_path = Path(prior_cfg["test_path"])
        seed = int(prior_cfg.get("seed", 0))
        reflection_minibatch_size = int(prior_cfg["reflection_minibatch_size"])
        task_lm_concurrency = int(prior_cfg["task_lm_concurrency"])
        max_metric_calls = int(prior_cfg["max_metric_calls"])
        seed_system_prompt = prior_cfg["seed_system_prompt"]
        if (
            args.seed_prompt is not None
            and load_seed_prompt_from_path(args.seed_prompt) != seed_system_prompt
        ):
            p.error("--seed-prompt mismatch with config.json; cannot resume.")
        resuming = True
    else:
        split_dir = args.split_dir or cfg.gepa_rocstories_main
        train_path = split_dir / "train.jsonl"
        val_path = split_dir / "val.jsonl"
        test_path = split_dir / "test.jsonl"
        seed = args.seed
        reflection_minibatch_size = args.reflection_minibatch_size
        task_lm_concurrency = args.task_lm_concurrency
        max_metric_calls = args.max_metric_calls if args.max_metric_calls is not None else 300
        seed_prompt_path = args.seed_prompt or cfg.rocstories_seed_prompt_path
        seed_system_prompt = load_seed_prompt_from_path(seed_prompt_path)
        run_dir = None
        resuming = False

    if args.stream_task_lm and task_lm_concurrency != 1:
        print(f"[stream] forcing task_lm_concurrency 1 (was {task_lm_concurrency})")
        task_lm_concurrency = 1

    train = [
        make_data_inst(context=ex["context"], gold=ex["gold"], sample_id=ex.get("id"))
        for ex in load_jsonl(train_path)
    ]
    val = [
        make_data_inst(context=ex["context"], gold=ex["gold"], sample_id=ex.get("id"))
        for ex in load_jsonl(val_path)
    ]
    print(f"[load] {len(train)} train, {len(val)} val   (sentence_ordering, arm={args.arm})")

    if not resuming:
        fresh_config = {
            "arm": args.arm,
            "config_path": str(args.config) if args.config else None,
            "task": "sentence_ordering",
            "split_dir": str(split_dir),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "n_train": len(train),
            "n_val": len(val),
            "seed": seed,
            "seed_system_prompt": seed_system_prompt,
            "seed_prompt_path": str(seed_prompt_path),
            "task_lm": asdict(cfg.task_lm),
            "reflection_lm": asdict(cfg.reflection_lm),
            "schema_path": str(schema_path),
            "cross_schema_paths": [str(pp) for pp in args.cross_schema],
            "max_metric_calls": max_metric_calls,
            "reflection_minibatch_size": reflection_minibatch_size,
            "task_lm_concurrency": task_lm_concurrency,
            "_started_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        run_dir = write_fresh_run_dir(cfg.runs_root, run_name, fresh_config)

    task_lm_callable = with_conversation_log(
        build_task_lm(
            cfg.task_lm,
            temperature=args.task_temperature,
            max_tokens=task_max_tokens,
            json_mode=True,
            stream_to_stderr=args.stream_task_lm,
        ),
        run_dir / "task_lm.jsonl",
        label="task_lm",
        print_progress=args.task_call_progress,
    )
    reflection_lm = with_conversation_log(
        build_reflection_lm(
            cfg.reflection_lm,
            temperature=args.reflection_temperature,
            max_tokens=reflection_max_tokens,
        ),
        run_dir / "reflection_lm.jsonl",
        label="reflection_lm",
    )

    drop_patterns: tuple[str, ...] = () if args.show_prompts else ("Proposed new text",)
    run_logger = make_filtered_run_logger(run_dir, drop_patterns=drop_patterns)

    evaluator_cls = ARMS[args.arm]
    adapter = ParallelDefaultAdapter(
        model=task_lm_callable,
        evaluator=evaluator_cls(
            schema,
            lm_logger=task_lm_callable,
            wl_integration=wl_integration_for_arm(args.arm),
        ),
        max_workers=task_lm_concurrency,
    )

    try:
        result = gepa.optimize(
            seed_candidate={SYSTEM_COMPONENT: seed_system_prompt},
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=reflection_minibatch_size,
            max_metric_calls=max_metric_calls,
            run_dir=str(run_dir),
            logger=run_logger,
            seed=seed,
            display_progress_bar=True,
            cache_evaluation=True,
            callbacks=[RenderOnStateSaved(run_dir)],
        )
    except TaskLMUnreachable as exc:
        print(
            f"\n[abort] task_lm became unreachable: {exc}\n"
            f"[abort] last checkpoint preserved in {run_dir}\n"
            f"[abort] resume with: uv run python {sys.argv[0]} --resume {run_dir}\n",
            file=sys.stderr,
        )
        sys.exit(2)

    best_idx = getattr(result, "best_idx", None)
    best_candidate = getattr(result, "best_candidate", None)
    val_scores = getattr(result, "val_aggregate_scores", []) or []
    summary = {
        "arm": args.arm,
        "task": "sentence_ordering",
        "best_candidate_idx": best_idx,
        "best_val_score": val_scores[best_idx] if best_idx is not None and best_idx < len(val_scores) else None,
        "num_candidates": len(getattr(result, "candidates", []) or []),
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "num_full_val_evals": getattr(result, "num_full_val_evals", None),
        "best_prompt": (best_candidate or {}).get(SYSTEM_COMPONENT)
        if isinstance(best_candidate, dict)
        else None,
    }
    best_prompt = summary["best_prompt"]
    if best_prompt:
        test_examples = load_jsonl(test_path)
        if holdout_already_done(
            run_dir, best_idx=best_idx, expected_n_test=len(test_examples)
        ):
            refreshed = rebuild_summary_from_disk(
                run_dir, cross_schema_paths=args.cross_schema
            )
            src = refreshed or json.loads((run_dir / "summary.json").read_text())
            summary["holdout"] = src.get("holdout") or {}
            print(f"[holdout] up to date for best_idx={best_idx} — refreshed from cache")
        else:
            gold_by_id = {str(ex["id"]): ex["gold"] for ex in test_examples}
            primary_kind, cross_specs = build_cross_specs_from_paths(
                schema_path, args.cross_schema, gold_by_id=gold_by_id
            )
            cross_kinds_str = ", ".join(s.kind for s in cross_specs) or "(none)"
            print(
                f"[holdout] scoring best prompt on {len(test_examples)} test examples "
                f"(primary={primary_kind!r}, cross-schema={cross_kinds_str})"
            )
            scores = evaluate_text_holdout(
                test_examples,
                system_prompt=best_prompt,
                task_lm=task_lm_callable,
                schema=schema,
                primary_kind=primary_kind,
                cross_specs=cross_specs,
                wl_integration=wl_integration_for_arm(args.arm),
                out_path=run_dir / "holdout_scores.jsonl",
                desc=f"{run_name}/holdout",
                max_workers=task_lm_concurrency,
            )
            summary["holdout"] = {
                "path": str(test_path),
                "primary_kind": primary_kind,
                "scores": scores,
            }
    else:
        print("[holdout] skipped — gepa.optimize returned no best_candidate")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    try:
        report_path = render_run(run_dir, verbose=False)
        print(f"[render] final: {report_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[render] final render failed: {exc!r}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] run dir: {run_dir}")
    print(
        f"[next] official PMR + Kendall tau: uv run python "
        f"scripts/eval_sentence_ordering.py --run-dir {run_dir}"
    )


if __name__ == "__main__":
    main()
