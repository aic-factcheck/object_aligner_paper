"""Jinja2 rendering of the aggregated results into Markdown.

Separates *presentation* (the ``.md.j2`` templates under ``templates/results/``
and the prose in ``results_meta.yaml``) from *computation* (``results.py``).
The renderer turns ``aggregate(disc)`` into a flat, template-friendly context
of pre-grouped rows and registers a handful of numeric-formatting filters; the
templates themselves contain no metric logic, so every number still traces back
to ``results.py``.

The headline OA table shows **every schema kind side-by-side** (no single OA
headline) with the OA-feedback-vs-OA-score Δ inside each kind — matching
``notebooks/evaluation.ipynb``. ``★`` marks the kind a row was optimised on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from object_aligner_exp.report.results import (
    DEFAULT_MODELS,
    Discovery,
    load_manifest,
    load_meta,
    native_spec,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates" / "results"
ARMS = ("oa_feedback", "oa_score")
ARM_LABEL = {"oa_feedback": "OA-feedback", "oa_score": "OA-score"}
_DECIMAL_METRICS = {"tau"}  # rendered as signed decimals, not percent

# Default filename of the machine-readable, table-keyed sidecar.
TABLES_FILENAME = "results_tables.json"


# Stable keys linking each Markdown table to its entry in results_tables.json.
def oa_table_key(dataset: str, model: str) -> str:
    return f"oa::{dataset}::{model}"


def native_table_key(dataset: str, model: str) -> str:
    return f"native::{dataset}::{model}"


def splits_table_key(dataset: str) -> str:
    return f"splits::{dataset}"


def failed_table_key(dataset: str) -> str:
    return f"failed::{dataset}"


# --------------------------------------------------------------------------- #
# Jinja filters
# --------------------------------------------------------------------------- #


def _fmt(stat: dict[str, Any] | None, signed: bool = False) -> str:
    """``mean ± std`` on the metric's original scale, 3 decimals; ``—`` if None.

    ``signed=True`` prefixes a sign (for metrics that can be negative, e.g.
    Kendall's τ in [-1, 1]); all other scores are in [0, 1].
    """
    if stat is None:
        return "—"
    mean, std = stat["mean"], stat["std"]
    fmt = "{:+.3f}" if signed else "{:.3f}"
    head = fmt.format(mean)
    return f"{head} ± {std:.3f}" if std is not None else head


def _fmt_delta(fb: dict[str, Any] | None, sc: dict[str, Any] | None) -> str:
    """Δ = feedback − score on the original scale; signed, 3 decimals."""
    if fb is None or sc is None:
        return "—"
    return f"{fb['mean'] - sc['mean']:+.3f}"


def _frac(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else f"{n}/0 (—)"


# --------------------------------------------------------------------------- #
# Context building
# --------------------------------------------------------------------------- #


def _model_sort_key(model: str) -> tuple[int, str]:
    return (DEFAULT_MODELS.index(model) if model in DEFAULT_MODELS else len(DEFAULT_MODELS), model)


def _cell(ds: dict[str, Any], dataset: str, arm: str, ablation: str, model: str) -> dict[str, Any] | None:
    return ds["cells"].get("|".join((dataset, arm, ablation, model)))


def _oa_kind(cell: dict[str, Any] | None, kind: str) -> dict[str, Any] | None:
    if cell is None:
        return None
    block = cell["oa"].get(kind)
    return block["mean_nonzero"] if block else None


def _native_stat(cell: dict[str, Any] | None, metric: str) -> dict[str, Any] | None:
    if cell is None or cell.get("native") is None:
        return None
    return cell["native"]["metrics"].get(metric)


def _seed_str(*cells: dict[str, Any] | None) -> str:
    return "/".join(str(c["n_seeds"]) for c in cells if c is not None) or "—"


def _build_dataset_ctx(dataset: str, ds: dict[str, Any], meta: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    models = sorted(ds["models"], key=_model_sort_key)
    ablations = ds["ablations"]
    kinds = ds["kinds"]
    spec = ds.get("native_spec")
    mlabel = meta["model_display"]

    def label(m: str) -> str:
        return mlabel.get(m, m)

    native_ctx = (
        {"metrics": [{"name": m, "signed": m in _DECIMAL_METRICS} for m in spec["metric_order"]]}
        if spec is not None
        else None
    )

    # Which ablation each schema kind was the optimisation criterion for —
    # used to place the arm (feedback − score) delta on the right diagonal.
    abl_for_kind: dict[str, str] = {}
    for abl, pk in ds["primary_by_ablation"].items():
        abl_for_kind.setdefault(pk, abl)

    # One table PER MODEL. Within it: rows = ablation; per arm the OA score
    # under every kind plus the within-arm cross-kind Δ (kinds[0] − kinds[1]);
    # a final row gives the OA-feedback − OA-score arm Δ at each kind's own
    # optimisation ablation (★).
    model_blocks = []
    for model in models:
        oa_rows = []
        for ablation in ablations:
            fb = _cell(ds, dataset, "oa_feedback", ablation, model)
            sc = _cell(ds, dataset, "oa_score", ablation, model)
            if fb is None and sc is None:
                continue
            primary = (fb or sc).get("primary_kind", "")
            f_cells = [{"kind": k, "stat": _oa_kind(fb, k), "primary": k == primary} for k in kinds]
            s_cells = [{"kind": k, "stat": _oa_kind(sc, k), "primary": k == primary} for k in kinds]
            oa_rows.append(
                {"ablation": ablation, "seeds": _seed_str(fb, sc), "f": f_cells, "s": s_cells}
            )

        # arm Δ (feedback − score) per kind, evaluated where that kind is primary
        arm_delta = []
        for k in kinds:
            a = abl_for_kind.get(k)
            fb = _cell(ds, dataset, "oa_feedback", a, model) if a else None
            sc = _cell(ds, dataset, "oa_score", a, model) if a else None
            arm_delta.append({"kind": k, "f": _oa_kind(fb, k), "s": _oa_kind(sc, k)})

        native_rows = []
        if spec is not None:
            for ablation in ablations:
                fb = _cell(ds, dataset, "oa_feedback", ablation, model)
                sc = _cell(ds, dataset, "oa_score", ablation, model)
                if fb is None and sc is None:
                    continue
                cols = [
                    {
                        "name": m,
                        "signed": m in _DECIMAL_METRICS,
                        "fb": _native_stat(fb, m),
                        "sc": _native_stat(sc, m),
                    }
                    for m in spec["metric_order"]
                ]
                native_rows.append({"ablation": ablation, "seeds": _seed_str(fb, sc), "cols": cols})

        if oa_rows or native_rows:
            model_blocks.append(
                {
                    "model": label(model),
                    "model_id": model,
                    "oa_key": oa_table_key(dataset, model),
                    "native_key": native_table_key(dataset, model),
                    "oa_rows": oa_rows,
                    "arm_delta": arm_delta,
                    "native_rows": native_rows,
                }
            )

    # Failed-sample accounting: per (model, ablation, arm).
    failed_rows = []
    for model in models:
        for ablation in ablations:
            for arm in ARMS:
                cell = _cell(ds, dataset, arm, ablation, model)
                if cell is None:
                    continue
                nz = []
                for kind in kinds:
                    block = cell["oa"].get(kind)
                    nz.append({"kind": kind, "pair": (block["n_zero_sum"], block["n_sum"]) if block else None})
                pf = None
                nb = cell.get("native")
                if nb is not None and not nb.get("errors"):
                    pf = (nb["n_parse_failures_sum"], nb["n_sum"])
                failed_rows.append(
                    {"model": label(model), "ablation": ablation, "arm": ARM_LABEL[arm], "nz": nz, "pf": pf}
                )

    # Appendix: full per-cell, all kinds.
    appendix_rows = []
    for model in models:
        for ablation in ablations:
            for arm in ARMS:
                cell = _cell(ds, dataset, arm, ablation, model)
                if cell is None:
                    continue
                for kind in kinds:
                    block = cell["oa"].get(kind)
                    if not block:
                        continue
                    appendix_rows.append(
                        {
                            "arm": ARM_LABEL[arm],
                            "ablation": ablation,
                            "model": label(model),
                            "kind": kind,
                            "nonzero": block["mean_nonzero"],
                            "all": block["mean_all"],
                            "nzero": (block["n_zero_sum"], block["n_sum"]),
                            "nperfect": block["n_perfect_sum"],
                            "nseeds": cell["n_seeds"],
                        }
                    )

    disp = meta["dataset_display"].get(dataset, {})
    return {
        "id": dataset,
        "title": disp.get("title", dataset),
        "blurb": disp.get("blurb"),
        "kinds": kinds,
        "models_label": ", ".join(label(m) for m in models),
        "arms": ds["arms"],
        "ablations": ablations,
        "native": native_ctx,
        "manifest": load_manifest(dataset, repo_root),
        "models": model_blocks,
        "failed_rows": failed_rows,
        "appendix_rows": appendix_rows,
        "splits_key": splits_table_key(dataset),
        "failed_key": failed_table_key(dataset),
    }


def build_context(
    disc: Discovery,
    agg: dict[str, Any],
    *,
    runs_root: Path,
    generated: str,
    repo_root: Path,
    show_appendix: bool = False,
    show_failed: bool = False,
    tables_file: str = TABLES_FILENAME,
) -> dict[str, Any]:
    meta = load_meta()
    datasets = [_build_dataset_ctx(name, ds, meta, repo_root) for name, ds in agg.items()]

    # Collapse excluded model cells to unique (dataset, arm, ablation, model).
    excl: dict[tuple, int] = {}
    for c in disc.excluded_model_cells:
        k = (c["dataset"], c["arm"], c["ablation"], c["model"])
        excl[k] = excl.get(k, 0) + 1
    excluded_model_cells = [
        {"dataset": d, "arm": a, "ablation": ab, "model": m, "n": n}
        for (d, a, ab, m), n in sorted(excl.items())
    ]

    # Native-metric definition sections (only datasets that have one).
    native_defs = []
    for name, ds in agg.items():
        if native_spec(name) is None:
            continue
        defs = meta["native_metric_defs"].get(name, {})
        native_defs.append(
            {"title": meta["dataset_display"].get(name, {}).get("title", name), "defs": list(defs.values())}
        )

    return {
        "generated": generated,
        "runs_root": str(runs_root),
        "n_summaries": disc.n_summaries_found,
        "n_runs": len(disc.runs),
        "models_kept": disc.models_kept,
        "models_seen": sorted(disc.models_seen),
        "datasets_kept": list(disc.datasets_kept) if disc.datasets_kept is not None else None,
        "datasets_seen": sorted(disc.datasets_seen),
        "excluded_datasets": disc.excluded_datasets,
        "duplicate_resolutions": disc.duplicate_resolutions,
        "excluded_model_cells": excluded_model_cells,
        "native_errors": disc.native_errors,
        "skipped_paths": disc.skipped_paths,
        "oa_score_def": meta["oa_score_def"],
        "native_defs": native_defs,
        "datasets": datasets,
        "show_appendix": show_appendix,
        "show_failed": show_failed,
        "tables_file": tables_file,
    }


def _make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters["fmt"] = _fmt
    env.filters["delta"] = _fmt_delta
    env.filters["frac"] = lambda pair: _frac(*pair) if pair else "—"
    return env


def render(
    disc: Discovery,
    agg: dict[str, Any],
    *,
    runs_root: Path,
    generated: str,
    repo_root: Path,
    show_appendix: bool = False,
    show_failed: bool = False,
    tables_file: str = TABLES_FILENAME,
) -> str:
    ctx = build_context(
        disc,
        agg,
        runs_root=runs_root,
        generated=generated,
        repo_root=repo_root,
        show_appendix=show_appendix,
        show_failed=show_failed,
        tables_file=tables_file,
    )
    return _make_env().get_template("report.md.j2").render(**ctx)


# --------------------------------------------------------------------------- #
# Machine-readable, table-keyed sidecar (Pandas-friendly)
# --------------------------------------------------------------------------- #


_TABLES_SCHEMA = {
    "tables": (
        "Aggregate tables, one per Markdown table, keyed exactly as the MD "
        "points to (e.g. 'oa::<dataset>::<model>'). Each entry has a 'records' "
        "list — load with pandas.DataFrame(entry['records'])."
    ),
    "tables.oa::*.records": (
        "columns: ablation, arm (oa_feedback|oa_score), kind, primary (bool, "
        "kind == optimisation criterion), mean (zero-excluded OA mean), std "
        "(sample std over seeds, null if n_seeds<2), mean_all (zeros included), "
        "std_all, n_zero_total, n_perfect_total, n_seeds."
    ),
    "tables.native::*.records": (
        "columns: ablation, arm, metric, mean, std, n_seeds, "
        "n_parse_failures_total."
    ),
    "tables.splits::*.records": "columns: split, n, source_split (or null).",
    "tables.failed::*.records": (
        "columns: ablation, arm, kind, n_zero, n_total (OA failures); and "
        "metric='__native_parse__' rows with n_parse_failures, n_total."
    ),
    "seeds.oa": (
        "Long/tidy per-seed OA scores — one row per (dataset, model, ablation, "
        "arm, kind, seed). columns: dataset, model, ablation, arm, kind, seed, "
        "primary, score_nonzero, score_all, n, n_zero, n_perfect. Filter by "
        "dataset/model to reproduce a table's seeds; group for boxplots."
    ),
    "seeds.native": (
        "Long/tidy per-seed native metrics — one row per (dataset, model, "
        "ablation, arm, metric, seed). columns: dataset, model, ablation, arm, "
        "metric, seed, value, n, n_parse_failures."
    ),
}


def build_tables(
    disc: Discovery, agg: dict[str, Any], *, generated: str, repo_root: Path
) -> dict[str, Any]:
    """Build the table-keyed, Pandas-friendly sidecar.

    Mirrors every Markdown results table as a tidy ``records`` list under a
    stable key, plus global long-format per-seed frames (``seeds.oa`` /
    ``seeds.native``) suitable for boxplots and other per-seed plots.
    """
    # Per-seed long frames (one row per run × kind / metric).
    oa_seeds: list[dict[str, Any]] = []
    native_seeds: list[dict[str, Any]] = []
    for r in disc.runs:
        k = r.key
        base = {"dataset": k.dataset, "model": k.model, "ablation": k.ablation, "arm": k.arm, "seed": k.seed}
        for kind, s in r.oa.items():
            oa_seeds.append(
                {
                    **base,
                    "kind": kind,
                    "primary": kind == r.primary_kind,
                    "score_nonzero": s.mean_score_nonzero,
                    "score_all": s.mean_score,
                    "n": s.n,
                    "n_zero": s.n_zero,
                    "n_perfect": s.n_perfect,
                }
            )
        if r.native is not None and not r.native.error:
            for m, v in r.native.metrics.items():
                native_seeds.append(
                    {**base, "metric": m, "value": v, "n": r.native.n, "n_parse_failures": r.native.n_parse_failures}
                )

    meta = load_meta()
    mlabel = meta["model_display"]
    tables: dict[str, Any] = {}
    for dataset, ds in agg.items():
        disp = meta["dataset_display"].get(dataset, {})
        title = disp.get("title", dataset)
        models = sorted(ds["models"], key=_model_sort_key)

        # splits
        man = load_manifest(dataset, repo_root)
        split_records = (
            [{"split": s["name"], "n": s.get("n"), "source_split": s.get("source_split")} for s in man["splits"]]
            if man and man.get("splits")
            else []
        )
        tables[splits_table_key(dataset)] = {
            "title": f"{title} — splits",
            "kind": "splits",
            "dataset": dataset,
            "records": split_records,
        }

        # per-model OA + native aggregate tables
        for model in models:
            oa_records: list[dict[str, Any]] = []
            native_records: list[dict[str, Any]] = []
            for arm in ARMS:
                for ablation in ds["ablations"]:
                    cell = _cell(ds, dataset, arm, ablation, model)
                    if cell is None:
                        continue
                    primary = cell.get("primary_kind", "")
                    for kind, blk in cell["oa"].items():
                        nz = blk["mean_nonzero"] or {}
                        al = blk["mean_all"] or {}
                        oa_records.append(
                            {
                                "ablation": ablation,
                                "arm": arm,
                                "kind": kind,
                                "primary": kind == primary,
                                "mean": nz.get("mean"),
                                "std": nz.get("std"),
                                "mean_all": al.get("mean"),
                                "std_all": al.get("std"),
                                "n_zero_total": blk["n_zero_sum"],
                                "n_perfect_total": blk["n_perfect_sum"],
                                "n_seeds": cell["n_seeds"],
                            }
                        )
                    nb = cell.get("native")
                    if nb is not None and not nb.get("errors"):
                        for m, st in nb["metrics"].items():
                            native_records.append(
                                {
                                    "ablation": ablation,
                                    "arm": arm,
                                    "metric": m,
                                    "mean": st["mean"],
                                    "std": st["std"],
                                    "n_seeds": cell["n_seeds"],
                                    "n_parse_failures_total": nb["n_parse_failures_sum"],
                                }
                            )
            tables[oa_table_key(dataset, model)] = {
                "title": f"{title} — OA graded score ({mlabel.get(model, model)})",
                "kind": "oa_scores",
                "dataset": dataset,
                "model": model,
                "records": oa_records,
            }
            if native_records:
                tables[native_table_key(dataset, model)] = {
                    "title": f"{title} — native metrics ({mlabel.get(model, model)})",
                    "kind": "native",
                    "dataset": dataset,
                    "model": model,
                    "records": native_records,
                }

        # failed-sample table (one entry per dataset, all models/arms)
        failed_records: list[dict[str, Any]] = []
        for model in models:
            for ablation in ds["ablations"]:
                for arm in ARMS:
                    cell = _cell(ds, dataset, arm, ablation, model)
                    if cell is None:
                        continue
                    for kind, blk in cell["oa"].items():
                        failed_records.append(
                            {
                                "model": model,
                                "ablation": ablation,
                                "arm": arm,
                                "kind": kind,
                                "n_zero": blk["n_zero_sum"],
                                "n_total": blk["n_sum"],
                            }
                        )
                    nb = cell.get("native")
                    if nb is not None and not nb.get("errors"):
                        failed_records.append(
                            {
                                "model": model,
                                "ablation": ablation,
                                "arm": arm,
                                "kind": "__native_parse__",
                                "n_zero": nb["n_parse_failures_sum"],
                                "n_total": nb["n_sum"],
                            }
                        )
        tables[failed_table_key(dataset)] = {
            "title": f"{title} — failed samples",
            "kind": "failed_samples",
            "dataset": dataset,
            "records": failed_records,
        }

    return {
        "generated_at": generated,
        "schema": _TABLES_SCHEMA,
        "tables": tables,
        "seeds": {"oa": oa_seeds, "native": native_seeds},
    }
