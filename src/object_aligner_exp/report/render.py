"""Compose the static `report.html` for a run."""

from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from object_aligner_exp.oa import (
    referential_feedback_for_arm,
    structural_filter_for_arm,
    top_k_for_arm,
    wl_integration_for_arm,
)
from object_aligner_exp.report.align import AlignCache, _hash, make_aligner
from object_aligner_exp.report.loader import LoadedPredictions, Run, load_run_skeleton


_TEMPLATES = Path(__file__).parent / "templates"


class _CrossScorer:
    """Score-only OA evaluation under each cross schema, with a hash cache.

    The main schema produces the full aligned views (via ``AlignCache``); for
    cross schemas the report only needs the scalar score per cell, so this
    keeps one ``ObjectAligner`` per schema and caches ``score`` by
    ``(name, hash(gold), hash(pred))``. Any OA failure yields ``None`` for
    that (schema, cell), mirroring how the holdout path collapses errors to 0.
    """

    def __init__(
        self,
        cross_schemas: list[tuple[str, dict[str, Any]]],
        *,
        wl_integration: str = "tie_break",
    ) -> None:
        self._aligners = [
            (name, make_aligner(schema, wl_integration=wl_integration))
            for name, schema in cross_schemas
        ]
        self._cache: dict[tuple[str, str, str], float | None] = {}

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self._aligners]

    def score(self, gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, float | None]:
        if not self._aligners:
            return {}
        gh, ph = _hash(gold), _hash(pred)
        out: dict[str, float | None] = {}
        for name, oa in self._aligners:
            key = (name, gh, ph)
            if key not in self._cache:
                try:
                    self._cache[key] = float(oa.align(gold, pred).score)
                except Exception:  # noqa: BLE001 — OA raises various types
                    self._cache[key] = None
            out[name] = self._cache[key]
        return out


def _read(name: str) -> str:
    return (_TEMPLATES / name).read_text()


@dataclass(frozen=True)
class _CellPayload:
    """What we ship to the JS for one (candidate, sample) cell."""

    raw: dict[str, Any] | None
    aligned_gold: dict[str, Any] | None
    aligned_pred: dict[str, Any] | None
    aligned_pred_goldids: dict[str, Any] | None
    aligned_pred_goldids_meta: dict[str, list[dict[str, str]]] | None
    score: float | None
    feedback: str | None
    parse_error: str | None
    oa_error: str | None
    available: bool
    pareto_iter: int | None
    cross_scores: dict[str, float | None] = field(default_factory=dict)


def _build_cells(
    run: Run,
    predictions: LoadedPredictions,
    cache: AlignCache,
    cross: _CrossScorer,
) -> dict[tuple[int, int], _CellPayload]:
    out: dict[tuple[int, int], _CellPayload] = {}
    for (cand_idx, sample_idx), (pred, parse_err) in predictions.preds.items():
        if sample_idx >= len(run.samples):
            continue
        pareto_iter = predictions.pareto.get((cand_idx, sample_idx))
        if pred is None:
            out[(cand_idx, sample_idx)] = _CellPayload(
                raw=None,
                aligned_gold=None,
                aligned_pred=None,
                aligned_pred_goldids=None,
                aligned_pred_goldids_meta=None,
                score=None,
                feedback=None,
                parse_error=parse_err,
                oa_error=None,
                available=True,
                pareto_iter=pareto_iter,
                cross_scores={name: None for name in cross.names},
            )
            continue
        gold = run.samples[sample_idx].gold
        gv, pv, pvg, gm, score, feedback, oa_err = cache.get(gold, pred)
        out[(cand_idx, sample_idx)] = _CellPayload(
            raw=pred,
            aligned_gold=gv,
            aligned_pred=pv,
            aligned_pred_goldids=pvg,
            aligned_pred_goldids_meta=gm,
            score=score,
            feedback=feedback,
            parse_error=None,
            oa_error=oa_err,
            available=True,
            pareto_iter=pareto_iter,
            cross_scores=cross.score(gold, pred),
        )
    return out


def _candidate_stats(
    run: Run,
    cells: dict[tuple[int, int], _CellPayload],
    schema_names: list[str],
) -> list[dict[str, Any]]:
    """Per-candidate mean score under each schema (main first, then cross).

    Means are taken over that candidate's available, non-error valset cells.
    ``n`` (cells contributing to the main mean) is reported so partial
    coverage is visible. The main score lives on ``cell.score``; cross scores
    on ``cell.cross_scores[name]``.
    """
    main_name = schema_names[0]
    by_cand: dict[int, list[_CellPayload]] = {c.idx: [] for c in run.candidates}
    for (cand_idx, _sample_idx), cell in cells.items():
        if cand_idx in by_cand:
            by_cand[cand_idx].append(cell)

    out: list[dict[str, Any]] = []
    for c in run.candidates:
        cand_cells = by_cand.get(c.idx, [])
        scores: dict[str, float | None] = {}
        # Main schema mean.
        main_vals = [
            cell.score for cell in cand_cells
            if cell.score is not None and cell.parse_error is None and cell.oa_error is None
        ]
        scores[main_name] = statistics.fmean(main_vals) if main_vals else None
        # Cross schema means.
        for name in schema_names[1:]:
            vals = [
                cell.cross_scores.get(name) for cell in cand_cells
                if cell.cross_scores.get(name) is not None
            ]
            scores[name] = statistics.fmean(vals) if vals else None
        out.append({
            "idx": c.idx,
            "is_best": c.is_best,
            "n": len(main_vals),
            "n_cells": len(cand_cells),
            "scores": scores,
        })
    return out


def _data_payload(
    run: Run,
    cells: dict[tuple[int, int], _CellPayload],
    schema_names: list[str],
) -> dict[str, Any]:
    avail_by_cand: dict[int, list[int]] = {c.idx: [] for c in run.candidates}
    for (cand_idx, sample_idx) in cells:
        if cand_idx in avail_by_cand:
            avail_by_cand[cand_idx].append(sample_idx)
    for v in avail_by_cand.values():
        v.sort()

    candidates_payload = [
        {
            "idx": c.idx,
            "system_prompt": c.system_prompt,
            "is_best": c.is_best,
            "available_samples": avail_by_cand.get(c.idx, []),
        }
        for c in run.candidates
    ]
    samples_payload = [
        {
            "idx": s.idx,
            "sample_id": s.sample_id,
            "title": s.title,
            "context": s.context,
            "gold": s.gold,
            "image_data_url": s.image_data_url,
        }
        for s in run.samples
    ]
    pairs_payload: dict[str, Any] = {}
    for (cand_idx, sample_idx), c in cells.items():
        pairs_payload[f"{cand_idx},{sample_idx}"] = {
            "raw": c.raw,
            "aligned_gold": c.aligned_gold,
            "aligned_pred": c.aligned_pred,
            "aligned_pred_goldids": c.aligned_pred_goldids,
            "aligned_pred_goldids_meta": c.aligned_pred_goldids_meta,
            "score": c.score,
            "cross_scores": c.cross_scores,
            "feedback": c.feedback,
            "parse_error": c.parse_error,
            "oa_error": c.oa_error,
            "available": c.available,
            "pareto_iter": c.pareto_iter,
        }

    return {
        "run_label": run.run_label,
        "arm": run.arm,
        "best_idx": run.best_candidate_idx,
        "task_lm_model": run.task_lm_model,
        "reflection_lm_model": run.reflection_lm_model,
        "schema_names": schema_names,
        "config": run.config,
        "holdout": run.holdout,
        "candidate_stats": _candidate_stats(run, cells, schema_names),
        "candidates": candidates_payload,
        "samples": samples_payload,
        "pairs": pairs_payload,
    }


def _json_for_html(obj: Any) -> str:
    """Serialize and escape so any literal ``</`` inside the JSON cannot
    terminate the surrounding `<script>` block.
    """
    raw = json.dumps(obj, ensure_ascii=False, default=str)
    return raw.replace("</", "<\\/")


def _substitute(template: str, mapping: dict[str, str]) -> str:
    for k, v in mapping.items():
        template = template.replace(k, v)
    return template


def render_run(run_dir: Path, *, output: Path | None = None, verbose: bool = False) -> Path:
    """Render a run dir to a self-contained HTML file.

    Args:
        run_dir: path to ``data/runs/<run>_<ts>/``.
        output:  where to write the HTML. Defaults to ``<run_dir>/report.html``.
        verbose: print one-line progress to stderr.

    Returns:
        The absolute path of the written file.
    """
    run_dir = Path(run_dir).resolve()
    run, predictions = load_run_skeleton(run_dir)
    if verbose:
        s = predictions.task_lm_stats
        print(
            f"[render] run={run.run_label} arm={run.arm} "
            f"candidates={len(run.candidates)} samples={len(run.samples)}",
            file=sys.stderr,
        )
        print(
            f"[render] task_lm.jsonl: records={s.get('records', 0)} "
            f"matched={s.get('matched', 0)} train_skipped={s.get('train_skipped', 0)} "
            f"unmatched_sys={s.get('unmatched_sys', 0)} unmatched_user={s.get('unmatched_user', 0)}",
            file=sys.stderr,
        )
        print(
            f"[render] cells: total={len(predictions.preds)} pareto_marked={predictions.pareto_count} "
            f"response_mismatches={len(predictions.response_mismatches)}",
            file=sys.stderr,
        )

    wl_integration = wl_integration_for_arm(run.arm)
    oa = make_aligner(
        run.schema,
        referential_feedback=referential_feedback_for_arm(run.arm),
        wl_integration=wl_integration,
    )
    cache = AlignCache(
        oa, schema=run.schema,
        structural_filter=structural_filter_for_arm(run.arm),
        top_k=top_k_for_arm(run.arm),
    )
    cross = _CrossScorer(run.cross_schemas, wl_integration=wl_integration)
    schema_names = [run.main_schema_name, *cross.names]
    cells = _build_cells(run, predictions, cache, cross)

    if verbose:
        ok = sum(1 for c in cells.values() if c.parse_error is None and c.oa_error is None and c.aligned_pred is not None)
        pe = sum(1 for c in cells.values() if c.parse_error is not None)
        oe = sum(1 for c in cells.values() if c.oa_error is not None)
        print(f"[render] aligned_ok={ok} parse_errors={pe} oa_errors={oe}", file=sys.stderr)

    payload = _data_payload(run, cells, schema_names)
    template = _read("report.html.tmpl")
    # WL-mode badge: makes the OA ``wl_integration`` scoring mode explicit in the
    # header, distinct from the amr_ra / amr_strict *schema* columns. blend is
    # the non-default knob (the ``_blend`` arms), so highlight it; tie_break is
    # muted. The score columns are computed under this mode.
    wl_label = "blend" if wl_integration == "blend" else "tie-break"
    wl_cls = "wl-blend" if wl_integration == "blend" else "wl-tiebreak"
    wl_badge = (
        f'<span class="wl-badge {wl_cls}" '
        f'title="OA wl_integration — how WL structural color resolves idScope '
        f'ties; scores below are computed under this mode">WL: {wl_label}</span>'
    )
    html = _substitute(
        template,
        {
            "__TITLE__": f"OA report — {run.run_label}",
            "__RUN_LABEL__": run.run_label,
            "__ARM__": run.arm or "—",
            "__WL_BADGE__": wl_badge,
            "__STYLE_CSS__": _read("style.css"),
            "__APP_JS__": _read("app.js"),
            "__DATA_JSON__": _json_for_html(payload),
        },
    )

    out_path = (output or (run_dir / "report.html")).resolve()
    # Atomic write so a watching browser never catches a half-written file.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(html)
    os.replace(tmp_path, out_path)
    return out_path
