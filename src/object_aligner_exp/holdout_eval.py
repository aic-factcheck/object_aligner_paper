"""Post-optimization holdout evaluation helpers.

After ``gepa.optimize`` returns a ``best_candidate``, each ``run_gepa_*``
script scores that candidate's system prompt zero-shot against the
truly held-out ``test`` split (the file GEPA never saw during
optimization), with the same ``OaFeedbackEvaluator`` the baselines use.

Two thin variants share a summary helper:

* :func:`evaluate_text_holdout`  — SciERC / REBEL (text input).
* :func:`evaluate_image_holdout` — graphimg (image + instruction).

Both write per-example rows to a JSONL and return a summary dict whose
shape mirrors the ``baseline_eval_*`` scripts so the two numbers can be
compared directly.

Cross-schema scoring
--------------------

If a list of :class:`CrossSchemaSpec` is passed via ``cross_specs``, each
held-out example is *also* scored against the schemas in that list — the
LM is queried once, the response is projected into each target schema's
wire shape (via ``spec.project_pred``) and scored against the matching
gold (``spec.gold_by_id[ex_id]``). Per-example rows then carry an extra
``cross_scores`` dict keyed by ``spec.kind``, and the returned summary's
``holdout`` block gains a ``scores`` sub-mapping (one stats block per
kind). The primary-schema score is replicated under its own kind for
uniformity.
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from tqdm import tqdm

from object_aligner_exp.cross_schema import identity_project
from object_aligner_exp.datasets.graphimg import GraphImgRow
from object_aligner_exp.evaluator import (
    OaFeedbackEvaluator,
    make_data_inst,
    make_image_data_inst,
)
from object_aligner_exp.gepa_adapter_image import _encode_image_data_url
from object_aligner_exp.llm import _CURRENT_SAMPLE_ID, _ConversationLogger
from object_aligner_exp.oa import score_and_feedback, wl_integration_for_arm
from object_aligner_exp.schemas import load_schema_from_path


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [r["score"] for r in rows]
    nonzero = [s for s in scores if s > 0.0]
    return {
        "n": len(rows),
        "mean_score": statistics.fmean(scores) if scores else 0.0,
        "median_score": statistics.median(scores) if scores else 0.0,
        "stdev_score": statistics.pstdev(scores) if scores else 0.0,
        "n_zero": sum(1 for s in scores if s == 0.0),
        "n_perfect": sum(1 for s in scores if s == 1.0),
        "n_valid_nonzero": len(nonzero),
        "mean_score_nonzero": statistics.fmean(nonzero) if nonzero else 0.0,
        "median_score_nonzero": statistics.median(nonzero) if nonzero else 0.0,
        "stdev_score_nonzero": statistics.pstdev(nonzero) if nonzero else 0.0,
    }


@dataclass(frozen=True)
class CrossSchemaSpec:
    """One additional schema to score the same held-out response under.

    The primary schema (passed as ``schema=`` to the holdout helpers) is
    scored through the normal :class:`OaFeedbackEvaluator` path so its
    feedback string and error handling stay byte-identical to the
    pre-cross-schema behaviour. Each ``CrossSchemaSpec`` adds one more
    scoring pass against ``schema``, after projecting the parsed
    prediction via ``project_pred(parsed_pred, source_kind=primary_kind)``
    and looking up the gold via ``gold_by_id[example_id]``.

    ``gold_by_id`` must cover every example id passed to the holdout
    helper; a missing id raises ``KeyError`` (loud, on purpose — silent
    drops would skew the aggregate).
    """

    kind: str
    schema: dict[str, Any]
    gold_by_id: Mapping[str, dict[str, Any]]
    project_pred: Callable[..., dict[str, Any]]


def build_cross_specs_from_paths(
    primary_path: Path | str,
    cross_paths: list[Path | str],
    *,
    gold_by_id: Mapping[str, dict[str, Any]],
) -> tuple[str, list[CrossSchemaSpec]]:
    """Build a ``(primary_kind, cross_specs)`` pair from filesystem paths.

    ``primary_kind`` is the stem of ``primary_path``. Each entry in
    ``cross_paths`` is loaded and turned into a
    :class:`CrossSchemaSpec` with :func:`identity_project` as the
    projection — every in-repo dataset shares its wire shape across its
    variants, so projection is a no-op.

    Dedup is best-effort and noisy on conflict:

    * A cross path that resolves (by ``Path.resolve``) to the same file
      as ``primary_path`` is dropped with a ``[warn]`` line — the
      primary is always scored, so cross-scoring against it would be
      redundant.
    * Cross paths sharing a stem (e.g. two files named ``foo.jsonc`` in
      different dirs) collapse to a single spec; the first occurrence
      wins and subsequent ones print ``[warn]``.
    """
    primary_path = Path(primary_path)
    primary_resolved = primary_path.resolve()
    primary_kind = primary_path.stem

    specs: list[CrossSchemaSpec] = []
    seen_kinds: set[str] = {primary_kind}
    for cp in cross_paths:
        p = Path(cp)
        if p.resolve() == primary_resolved:
            print(
                f"[warn] cross-schema {p} matches --schema; dropping "
                "(primary is always scored)"
            )
            continue
        kind = p.stem
        if kind in seen_kinds:
            print(
                f"[warn] cross-schema {p} shares stem {kind!r} with an "
                "earlier --schema/--cross-schema; dropping"
            )
            continue
        seen_kinds.add(kind)
        specs.append(
            CrossSchemaSpec(
                kind=kind,
                schema=load_schema_from_path(p),
                gold_by_id=gold_by_id,
                project_pred=identity_project,
            )
        )
    return primary_kind, specs


def parse_response(response: str) -> dict[str, Any] | None:
    """Parse the LM response into a dict, or return None on any failure.

    The primary schema path uses :class:`OaFeedbackEvaluator` which raises
    a specific :class:`_ParseError` on failure; here we don't need the
    error message — the primary path already records it — so a flat
    None / dict result is enough to short-circuit cross-scoring.
    """
    try:
        text = (response or "").strip()
        if not text:
            return None
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def score_cross_specs(
    parsed: dict[str, Any] | None,
    *,
    primary_kind: str,
    primary_score: float,
    cross_specs: list[CrossSchemaSpec],
    example_id: str,
    wl_integration: str = "tie_break",
) -> dict[str, float]:
    """Score ``parsed`` against every spec; primary kind echoes ``primary_score``.

    Returns ``{kind: score}``. A parse failure (``parsed is None``) makes
    every cross-score 0.0. An OA exception on a single spec also returns
    0.0 for that spec, mirroring how the primary path handles OA failures.

    ``wl_integration`` is forwarded to OA so cross-schema scoring uses the same
    WL mode as the run's arm. (Inert for strict cross schemas, which carry no
    ``idScope`` / ``ref`` for WL to refine.)
    """
    out: dict[str, float] = {primary_kind: primary_score}
    if parsed is None:
        for spec in cross_specs:
            out[spec.kind] = 0.0
        return out
    for spec in cross_specs:
        if spec.kind == primary_kind:
            # Caller already supplied primary_score; don't recompute.
            continue
        try:
            projected = spec.project_pred(
                parsed,
                source_kind=primary_kind,
                target_kind=spec.kind,
            )
            gold = spec.gold_by_id[example_id]
            score, _ = score_and_feedback(
                gold, projected, spec.schema, wl_integration=wl_integration
            )
            out[spec.kind] = float(score)
        except Exception:  # noqa: BLE001 — OA / projection failures all collapse to 0
            out[spec.kind] = 0.0
    return out


def aggregate_by_kind(
    rows: list[dict[str, Any]],
    *,
    kinds: list[str] | tuple[str, ...],
    primary_kind: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{kind: stats_dict}`` aggregating each ``kind`` over ``rows``.

    For each kind, score data comes from ``row['cross_scores'][kind]`` if
    at least one row carries it. Otherwise — and only when ``kind`` equals
    ``primary_kind`` — the row's top-level ``score`` field is used (so
    single-schema runs without ``cross_scores`` still produce a primary
    aggregate). Kinds with no available data anywhere are skipped.

    Insertion order of ``kinds`` is preserved in the output.
    """
    out: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        rows_with_kind = [
            r for r in rows if kind in (r.get("cross_scores") or {})
        ]
        if rows_with_kind:
            per_kind = [{"score": r["cross_scores"][kind]} for r in rows_with_kind]
        elif kind == primary_kind:
            per_kind = [{"score": r["score"]} for r in rows if "score" in r]
            if not per_kind:
                continue
        else:
            continue
        out[kind] = summarize(per_kind)
    return out


def _resolve_gold_by_id(
    summary: dict[str, Any], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Build ``{id: gold}`` for a run's holdout from its split metadata.

    Supports both text splits (``test.jsonl`` style) and graphimg image
    splits (a directory containing ``labels.jsonl``).
    """
    holdout = (
        summary.get("holdout")
        or summary.get("holdout_scores")
        or {}
    )
    text_path = holdout.get("path") or config.get("test_path")
    if text_path:
        from object_aligner_exp.datasets import load_jsonl

        return {str(ex["id"]): ex["gold"] for ex in load_jsonl(Path(text_path))}
    image_dir = holdout.get("dir") or config.get("test_dir")
    if image_dir:
        from object_aligner_exp.datasets import load_graphimg_split

        return {str(r["id"]): r["gold"] for r in load_graphimg_split(Path(image_dir))}
    raise ValueError(
        "neither summary['holdout']['path'/'dir'] nor "
        "config['test_path'/'test_dir'] is set; cannot locate gold"
    )


def rebuild_summary_from_disk(
    run_dir: Path,
    *,
    cross_schema_paths: list[Path] | None = None,
) -> dict[str, Any] | None:
    """Refresh ``summary.json`` aggregates from cached ``holdout_scores.jsonl``.

    No LM is invoked — only the rows already on disk are re-parsed and
    re-aggregated. Used in two places:

    * Each runner's ``--resume`` path, so a resume picks up the latest
      ``summarize()`` fields (e.g. ``mean_score_nonzero``) and any newly
      requested ``--cross-schema`` without re-running holdout.
    * The standalone ``scripts/recompute_cross_scores.py`` backfill.

    When ``cross_schema_paths`` is non-empty, each row's ``response`` is
    re-parsed and re-scored against the additional schemas; rows are
    rewritten with the merged ``cross_scores`` map. Existing cross-kinds
    on rows are preserved.

    Returns the new summary dict, or ``None`` if the run lacks the files
    needed to rebuild (``summary.json`` / ``holdout_scores.jsonl`` /
    ``config.json`` with ``schema_path``).
    """
    summary_path = run_dir / "summary.json"
    holdout_path = run_dir / "holdout_scores.jsonl"
    config_path = run_dir / "config.json"
    if not summary_path.exists() or not holdout_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    primary_schema_path_str = config.get("schema_path") or summary.get("schema_path")
    if cross_schema_paths is None:
        # Fall back to the cross-schema list recorded in config.json so a
        # bare refresh fills in cross_scores for runs whose rows were
        # written before cross-scoring was wired up (or before this
        # backfill).
        cross_paths = [Path(p) for p in (config.get("cross_schema_paths") or [])]
    else:
        cross_paths = list(cross_schema_paths)
    if cross_paths and not primary_schema_path_str:
        # Re-scoring needs to know what the primary kind is. Pure refresh
        # (no cross_paths) keeps working without it.
        return None
    primary_schema_path = (
        Path(primary_schema_path_str) if primary_schema_path_str else None
    )
    primary_kind = primary_schema_path.stem if primary_schema_path else ""

    rows: list[dict[str, Any]] = []
    with holdout_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    cross_specs: list[CrossSchemaSpec] = []
    if cross_paths:
        gold_by_id = _resolve_gold_by_id(summary, config)
        primary_kind, cross_specs = build_cross_specs_from_paths(
            primary_schema_path, cross_paths, gold_by_id=gold_by_id
        )
        wl_integration = wl_integration_for_arm(config.get("arm"))
        for row in rows:
            ex_id = str(row.get("id"))
            primary_score = float(row.get("score", 0.0))
            parsed = parse_response(row.get("response", ""))
            new_scores = score_cross_specs(
                parsed,
                primary_kind=primary_kind,
                primary_score=primary_score,
                cross_specs=cross_specs,
                example_id=ex_id,
                wl_integration=wl_integration,
            )
            existing = row.get("cross_scores") or {}
            existing.update(new_scores)
            row["cross_scores"] = existing
        with holdout_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Collect kinds in stable order: primary first (when present), then
    # anything found in row data, then any new specs. The primary check is
    # delegated to aggregate_by_kind, which falls back to the row's top-level
    # ``score`` when the primary kind isn't echoed into cross_scores.
    kinds_ordered: list[str] = []
    seen: set[str] = set()
    has_primary_data = primary_kind and (
        any(primary_kind in (r.get("cross_scores") or {}) for r in rows)
        or any("score" in r for r in rows)
    )
    if primary_kind and has_primary_data:
        kinds_ordered.append(primary_kind)
        seen.add(primary_kind)
    for row in rows:
        for k in (row.get("cross_scores") or {}).keys():
            if k not in seen:
                kinds_ordered.append(k)
                seen.add(k)
    for spec in cross_specs:
        if spec.kind not in seen:
            kinds_ordered.append(spec.kind)
            seen.add(spec.kind)

    scores = aggregate_by_kind(
        rows, kinds=kinds_ordered, primary_kind=primary_kind or None
    )

    # Build the new holdout block (path/dir + primary_kind + scores).
    # Preserve path/dir from any prior holdout container; drop the
    # duplicated aggregate fields that used to live there.
    prior_meta = summary.get("holdout") or summary.get("holdout_scores") or {}
    holdout_block: dict[str, Any] = {}
    for k in ("path", "dir"):
        if k in prior_meta:
            holdout_block[k] = prior_meta[k]
    if primary_kind:
        holdout_block["primary_kind"] = primary_kind
    holdout_block["scores"] = scores

    summary["holdout"] = holdout_block
    summary.pop("holdout_scores", None)
    summary.pop("cross_scores", None)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def evaluate_text_holdout(
    examples: list[Mapping[str, Any]],
    *,
    system_prompt: str,
    task_lm: Callable[[list[dict[str, Any]]], str],
    schema: dict[str, Any],
    out_path: Path,
    desc: str = "holdout",
    primary_kind: str | None = None,
    cross_specs: list[CrossSchemaSpec] | None = None,
    wl_integration: str = "tie_break",
    max_workers: int = 1,
) -> dict[str, dict[str, Any]]:
    """Score ``system_prompt`` zero-shot on each text example with OA.

    Writes one row per example (``{id, score, response, feedback}``,
    plus ``cross_scores`` when ``cross_specs`` is provided) to ``out_path``
    and returns the per-kind ``cross_scores`` aggregates dict (primary
    first, then declared cross-schemas in order).

    ``wl_integration`` (``"tie_break"`` | ``"blend"``) is forwarded to OA so the
    holdout score matches the run's arm.

    ``max_workers`` fans out the task_lm calls (``1`` = sequential, identical to
    the original streaming behavior; ``N>1`` = up to N concurrent calls;
    ``-1`` = one worker per example). Rows are always written to ``out_path`` in
    input order regardless of completion order, so the file is deterministic.
    """
    lm_logger = task_lm if isinstance(task_lm, _ConversationLogger) else None
    evaluator = OaFeedbackEvaluator(
        schema, lm_logger=lm_logger, wl_integration=wl_integration
    )
    cross_specs = cross_specs or []
    if cross_specs and primary_kind is None:
        raise ValueError("primary_kind is required when cross_specs is non-empty")
    kinds_list: list[str] = []
    seen: set[str] = set()
    if primary_kind:
        kinds_list.append(primary_kind)
        seen.add(primary_kind)
    for spec in cross_specs:
        if spec.kind not in seen:
            kinds_list.append(spec.kind)
            seen.add(spec.kind)

    rows: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _score_one_example(ex: Mapping[str, Any]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["context"]},
        ]
        token = _CURRENT_SAMPLE_ID.set(ex.get("id"))
        try:
            response = task_lm(messages)
        finally:
            _CURRENT_SAMPLE_ID.reset(token)
        data_inst = make_data_inst(
            context=ex["context"], gold=ex["gold"], sample_id=ex.get("id")
        )
        result = evaluator(data_inst, response)
        row: dict[str, Any] = {
            "id": ex.get("id"),
            "score": result.score,
            "response": response,
            "feedback": result.feedback,
        }
        if cross_specs:
            parsed = parse_response(response)
            row["cross_scores"] = score_cross_specs(
                parsed,
                primary_kind=primary_kind or "",
                primary_score=float(result.score),
                cross_specs=cross_specs,
                example_id=str(ex.get("id")),
                wl_integration=wl_integration,
            )
        return row

    workers = len(examples) if max_workers < 0 else max_workers
    if workers <= 1 or len(examples) <= 1:
        # Sequential path — streaming write, byte-identical to the original.
        with out_path.open("w") as f:
            for ex in tqdm(examples, unit="ex", desc=desc):
                row = _score_one_example(ex)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
    else:
        # Concurrent path — fan out task_lm calls, reassemble in input order.
        workers = min(workers, len(examples))
        ordered: list[dict[str, Any] | None] = [None] * len(examples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_score_one_example, ex): i
                for i, ex in enumerate(examples)
            }
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                unit="ex",
                desc=desc,
            ):
                ordered[futures[fut]] = fut.result()
        rows = [r for r in ordered if r is not None]
        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return aggregate_by_kind(rows, kinds=kinds_list, primary_kind=primary_kind)


def evaluate_image_holdout(
    rows: list[GraphImgRow],
    *,
    system_prompt: str,
    task_instruction: str,
    task_lm: Callable[[list[dict[str, Any]]], str],
    schema: dict[str, Any],
    out_path: Path,
    desc: str = "holdout",
    primary_kind: str | None = None,
    cross_specs: list[CrossSchemaSpec] | None = None,
    wl_integration: str = "tie_break",
) -> dict[str, dict[str, Any]]:
    """Image counterpart of :func:`evaluate_text_holdout`.

    Builds the same ``text + image_url`` user message that
    :class:`ImageGepaAdapter` uses during optimization, so the holdout
    evaluation is apples-to-apples with what GEPA was scoring.

    ``wl_integration`` (``"tie_break"`` | ``"blend"``) is forwarded to OA so the
    holdout score matches the run's arm.
    """
    lm_logger = task_lm if isinstance(task_lm, _ConversationLogger) else None
    evaluator = OaFeedbackEvaluator(
        schema, lm_logger=lm_logger, wl_integration=wl_integration
    )
    cross_specs = cross_specs or []
    if cross_specs and primary_kind is None:
        raise ValueError("primary_kind is required when cross_specs is non-empty")
    kinds_list: list[str] = []
    seen: set[str] = set()
    if primary_kind:
        kinds_list.append(primary_kind)
        seen.add(primary_kind)
    for spec in cross_specs:
        if spec.kind not in seen:
            kinds_list.append(spec.kind)
            seen.add(spec.kind)

    out_rows: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in tqdm(rows, unit="img", desc=desc):
            image_path = str(r["image_abspath"])
            data_url = _encode_image_data_url(image_path)
            user_content = [
                {"type": "text", "text": task_instruction},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            token = _CURRENT_SAMPLE_ID.set(r["id"])
            try:
                response = task_lm(messages)
            finally:
                _CURRENT_SAMPLE_ID.reset(token)
            data_inst = make_image_data_inst(
                image_path=image_path,
                instruction=task_instruction,
                gold=r["gold"],
                sample_id=r["id"],
            )
            result = evaluator(data_inst, response)
            row: dict[str, Any] = {
                "id": r["id"],
                "score": result.score,
                "response": response,
                "feedback": result.feedback,
            }
            if cross_specs:
                parsed = parse_response(response)
                row["cross_scores"] = score_cross_specs(
                    parsed,
                    primary_kind=primary_kind or "",
                    primary_score=float(result.score),
                    cross_specs=cross_specs,
                    example_id=str(r["id"]),
                    wl_integration=wl_integration,
                )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_rows.append(row)
    return aggregate_by_kind(out_rows, kinds=kinds_list, primary_kind=primary_kind)
