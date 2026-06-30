"""Deterministic aggregation of GEPA × Object-Aligner runs under ``data/runs``.

This module is the LLM-free engine behind ``scripts/build_results_report.py``.
It walks every run directory, reads the already-computed OA graded scores from
``summary.json``, *recomputes* the task-native leaderboard metrics from each
run's ``holdout_scores.jsonl`` (reusing the canonical ``*_eval.score_corpus``
functions), and aggregates everything across seeds. Nothing here calls an LLM;
every number is a pure function of files on disk.

Run-directory layout (one optimisation run per leaf)::

    data/runs/gepa_<dataset>/<arm>/<ablation…>/<model>/s<seed>_<YYYYMMDD_HHMMSS>/

* ``arm`` ∈ {``oa_feedback``, ``oa_score``} — the GEPA reflection-signal arm.
* ``ablation`` is one *or two* path components (e.g. ``fixed``, ``ra``, or
  ``v3_medium/ra`` for ``gepa_graphimg_v3``).
* The leaf name encodes seed and timestamp.

Key conventions (kept identical to how the experiments record results):

* **OA graded score** lives in ``summary.json: holdout.scores[kind]`` per schema
  *kind*, already carrying ``n_zero`` (JSON / schema failures, score 0),
  ``mean_score`` (all samples) and ``mean_score_nonzero`` (zeros excluded). The
  *headline* OA number is the zero-excluded mean; the zero count/proportion is
  always reported alongside.
* **Native metrics** are recomputed here; a parse failure (un-decodable JSON)
  counts as a wrong example in the standard leaderboard fashion and is reported
  as ``n_parse_failures``.
* **Seed aggregation** is mean ± sample standard deviation (``statistics.stdev``,
  defined only for ``n_seeds > 1``).
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from object_aligner_exp import (
    amr_eval,
    docred_eval,
    molecules_eval,
    natural_plan_eval,
    planbench_eval,
    scierc_eval,
    sentence_ordering_eval,
    wec_eng_eval,
)
from object_aligner_exp.datasets.molecules import gold_by_id_from_labels

__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_DATASETS",
    "RunKey",
    "OAKindScore",
    "NativeScore",
    "RunMetrics",
    "Stat",
    "Discovery",
    "discover_runs",
    "aggregate",
    "load_manifest",
    "load_meta",
    "stat_of",
]

# The two gemma4 task LMs the report defaults to. Everything else is recorded as
# an excluded cell so the provenance section can list what was dropped.
DEFAULT_MODELS = ("gemma4-26b", "gemma4-e4b")

# Datasets included by default (the publication-relevant subset). Override on the
# CLI with --datasets / --all-datasets. Excluded by default: graphimg_v3 and the
# two synra_nl splits.
DEFAULT_DATASETS = (
    "graphimg_v2",
    "natural_plan_meeting",
    "natural_plan_trip",
    "planbench_blocksworld",
    "scierc_native_exact",
    "amr_littleprince",
    "amr_bio",
    "docred_pcode",
    "docred_obf",
    "biored",
    "wec_eng",
    "molecules_rendered",
    "molecules_decimer",
    "synra_codex_noobf",
    "synra_codex_noobf6",
    "synra_codex_obf",
    "synra_codex_obf6",
    "synra_sort_stated_wide",
    "synra_sort_stated_tight",
    "synra_sort_hidden_wide",
    "synra_sort_hidden_tight",
    "sentence_ordering_arxiv",
    "sentence_ordering_rocstories",
)

# Leaf directory: s<seed>_<YYYYMMDD>_<HHMMSS>.
_LEAF_RE = re.compile(r"^s(\d+)_(\d{8}_\d{6})$")

# Raw SciERC test docs (keyed by doc_key) — the native scorer needs the original
# token-level annotations, not the OA-shaped split. Mirrors eval_scierc.py.
_SCIERC_RAW_TEST = "data/scierc/raw/processed_data/json/test.json"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunKey:
    """Identity of one optimisation run, parsed from its directory path."""

    dataset: str  # e.g. "natural_plan_meeting" (leading "gepa_" stripped)
    arm: str  # "oa_feedback" | "oa_score"
    ablation: str  # e.g. "fixed", "ra", "v3_medium/ra"
    model: str  # e.g. "gemma4-26b"
    seed: int
    timestamp: str  # "YYYYMMDD_HHMMSS"

    @property
    def cell(self) -> tuple[str, str, str, str]:
        """The (dataset, arm, ablation, model) cell this run aggregates into."""
        return (self.dataset, self.arm, self.ablation, self.model)


@dataclass
class OAKindScore:
    """OA graded score for one schema *kind* of one run (from summary.json)."""

    kind: str
    n: int
    n_zero: int
    n_perfect: int
    mean_score: float  # over ALL samples (zeros included)
    mean_score_nonzero: float  # zeros (JSON/schema failures) excluded — headline


@dataclass
class NativeScore:
    """Recomputed task-native leaderboard metric(s) for one run."""

    n: int
    n_parse_failures: int
    metrics: dict[str, float]  # e.g. {"solve_rate": ..} / {"pmr": .., "tau": ..}
    by_difficulty: dict[int, dict[str, float]] | None = None
    error: str | None = None  # set if recomputation failed; metrics then empty


@dataclass
class RunMetrics:
    """Everything extracted from a single run leaf."""

    key: RunKey
    run_dir: Path
    primary_kind: str
    oa: dict[str, OAKindScore]  # kind -> score
    best_val_score: float | None = None
    total_metric_calls: int | None = None
    native: NativeScore | None = None


@dataclass
class Discovery:
    """Result of walking ``data/runs``: kept runs plus provenance notes."""

    runs: list[RunMetrics] = field(default_factory=list)
    n_summaries_found: int = 0
    skipped_paths: list[str] = field(default_factory=list)  # unparseable layout
    excluded_model_cells: list[dict[str, Any]] = field(default_factory=list)
    duplicate_resolutions: list[dict[str, Any]] = field(default_factory=list)
    native_errors: list[dict[str, Any]] = field(default_factory=list)
    models_seen: set[str] = field(default_factory=set)
    models_kept: tuple[str, ...] = ()
    datasets_seen: set[str] = field(default_factory=set)
    datasets_kept: tuple[str, ...] | None = None  # None = all datasets
    excluded_datasets: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Native-metric registry
# --------------------------------------------------------------------------- #
#
# Each entry knows how to (a) load gold keyed by id and (b) score a run's
# predictions, returning a NativeScore. Datasets absent from this map (synra_*,
# graphimg_*) have no native leaderboard metric — the OA graded score is it.


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _examples_by_id(split_path: Path) -> dict[str, dict[str, Any]]:
    return {str(ex["id"]): ex for ex in _load_jsonl(split_path)}


def _scierc_raw_by_id(raw_path: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for doc in _load_jsonl(raw_path):
        key = str(doc.get("doc_key", ""))
        if key:
            docs[key] = doc
    return docs


def _native_planbench(preds, gold, _task) -> NativeScore:
    res = planbench_eval.score_corpus(preds, gold)
    return NativeScore(
        n=res.n,
        n_parse_failures=res.n_parse_failures,
        metrics={"solve_rate": res.solve_rate},
        by_difficulty={
            d: {"solve_rate": (c / t if t else 0.0), "n": t}
            for d, (c, t) in sorted(res.by_difficulty.items())
        },
    )


def _native_natural_plan(preds, gold, task) -> NativeScore:
    res = natural_plan_eval.score_corpus(task, preds, gold)
    return NativeScore(
        n=res.n,
        n_parse_failures=res.n_parse_failures,
        metrics={"solve_rate": res.solve_rate},
        by_difficulty={
            d: {"solve_rate": (c / t if t else 0.0), "n": t}
            for d, (c, t) in sorted(res.by_difficulty.items())
        },
    )


def _native_sentence_ordering(preds, gold, _task) -> NativeScore:
    res = sentence_ordering_eval.score_corpus(preds, gold)
    return NativeScore(
        n=res.n,
        n_parse_failures=res.n_parse_failures,
        metrics={"pmr": res.pmr_rate, "tau": res.mean_tau},
        by_difficulty={
            int(d): {
                "pmr": (c / t if t else 0.0),
                "tau": (s / t if t else 0.0),
                "n": int(t),
            }
            for d, (c, s, t) in sorted(res.by_difficulty.items())
        },
    )


def _native_scierc(preds, gold, _task) -> NativeScore:
    res = scierc_eval.score_corpus(preds, gold)
    return NativeScore(
        n=res.n_docs,
        n_parse_failures=res.n_parse_failures,
        metrics={
            "rel_entity_f1": res.rel_entity.f1,
            "rel_plus_entity_f1": res.rel_plus_entity.f1,
            "ner_f1": res.ner.f1,
        },
    )


def _native_amr(preds, gold, _task) -> NativeScore:
    res = amr_eval.score_corpus(preds, gold)
    return NativeScore(
        n=res.n,
        n_parse_failures=res.n_parse_failures,
        metrics={"smatch_f1": res.smatch_f1},
    )


def _native_docred(preds, gold, _task) -> NativeScore:
    res = docred_eval.score_corpus(preds, gold)
    return NativeScore(
        n=res.n,
        n_parse_failures=res.n_parse_failures,
        metrics={
            "rel_f1": res.rel_f1,
            "rel_precision": res.rel_precision,
            "rel_recall": res.rel_recall,
        },
    )


def _native_wec_eng(preds, gold, _task) -> NativeScore:
    res = wec_eng_eval.score_corpus(preds, gold)
    return NativeScore(
        n=res.n,
        n_parse_failures=res.n_parse_failures,
        metrics={
            "conll_f1": res.conll_f1,
            "muc_f1": res.muc_f1,
            "b3_f1": res.b3_f1,
            "ceafe_f1": res.ceafe_f1,
            "lea_f1": res.lea_f1,
        },
    )


def _native_molecules(preds, gold, _task) -> NativeScore:
    res = molecules_eval.score_corpus(preds, gold)
    return NativeScore(
        n=res.n,
        n_parse_failures=res.n_parse_failures,
        metrics={"canon_smiles_match": res.canon_smiles_match, "graph_f1": res.graph_f1},
    )


@dataclass(frozen=True)
class _NativeSpec:
    scorer: Callable[[list, dict, str], NativeScore]
    task: str  # passed through to scorer (natural_plan needs trip/meeting)
    gold_kind: str  # "split" (use holdout.path) | "scierc_raw" | "image_dir" (holdout.dir)
    # the metric used in headline arm-vs-arm tables:
    headline_metric: str
    # all metrics this scorer produces, in display order:
    metric_order: tuple[str, ...]


_NATIVE_REGISTRY: dict[str, _NativeSpec] = {
    "planbench_blocksworld": _NativeSpec(
        _native_planbench, "blocksworld", "split", "solve_rate", ("solve_rate",)
    ),
    "natural_plan_trip": _NativeSpec(
        _native_natural_plan, "trip", "split", "solve_rate", ("solve_rate",)
    ),
    "natural_plan_meeting": _NativeSpec(
        _native_natural_plan, "meeting", "split", "solve_rate", ("solve_rate",)
    ),
    "sentence_ordering_arxiv": _NativeSpec(
        _native_sentence_ordering, "", "split", "pmr", ("pmr", "tau")
    ),
    "sentence_ordering_rocstories": _NativeSpec(
        _native_sentence_ordering, "", "split", "pmr", ("pmr", "tau")
    ),
    # synra_sort (Facts2Order) reuses the sentence-ordering native scorer (PMR +
    # Kendall tau, by_difficulty per N) — one entry per 2x2 build.
    "synra_sort_stated_wide": _NativeSpec(
        _native_sentence_ordering, "", "split", "pmr", ("pmr", "tau")
    ),
    "synra_sort_stated_tight": _NativeSpec(
        _native_sentence_ordering, "", "split", "pmr", ("pmr", "tau")
    ),
    "synra_sort_hidden_wide": _NativeSpec(
        _native_sentence_ordering, "", "split", "pmr", ("pmr", "tau")
    ),
    "synra_sort_hidden_tight": _NativeSpec(
        _native_sentence_ordering, "", "split", "pmr", ("pmr", "tau")
    ),
    "scierc_native_exact": _NativeSpec(
        _native_scierc,
        "",
        "scierc_raw",
        "rel_entity_f1",
        ("rel_entity_f1", "rel_plus_entity_f1", "ner_f1"),
    ),
    "amr_littleprince": _NativeSpec(
        _native_amr, "", "split", "smatch_f1", ("smatch_f1",)
    ),
    "amr_bio": _NativeSpec(
        _native_amr, "", "split", "smatch_f1", ("smatch_f1",)
    ),
    "docred_pcode": _NativeSpec(
        _native_docred, "", "split", "rel_f1",
        ("rel_f1", "rel_precision", "rel_recall"),
    ),
    "docred_obf": _NativeSpec(
        _native_docred, "", "split", "rel_f1",
        ("rel_f1", "rel_precision", "rel_recall"),
    ),
    # BioRED shares DocRED's cluster-level {entities, relations} shape, so the
    # coref-aware relation-F1 scorer is reused verbatim.
    "biored": _NativeSpec(
        _native_docred, "", "split", "rel_f1",
        ("rel_f1", "rel_precision", "rel_recall"),
    ),
    # WEC-Eng is cross-document event coreference: the native metric is the
    # official CoNLL coreference family (mean of MUC/B3/CEAFe), with LEA reported.
    "wec_eng": _NativeSpec(
        _native_wec_eng, "", "split", "conll_f1",
        ("conll_f1", "muc_f1", "b3_f1", "ceafe_f1", "lea_f1"),
    ),
    "molecules_rendered": _NativeSpec(
        _native_molecules, "", "image_dir", "canon_smiles_match",
        ("canon_smiles_match", "graph_f1"),
    ),
    "molecules_decimer": _NativeSpec(
        _native_molecules, "", "image_dir", "canon_smiles_match",
        ("canon_smiles_match", "graph_f1"),
    ),
}


def native_spec(dataset: str) -> _NativeSpec | None:
    return _NATIVE_REGISTRY.get(dataset)


# --------------------------------------------------------------------------- #
# Discovery + extraction
# --------------------------------------------------------------------------- #


def _parse_run_key(leaf: Path, runs_root: Path) -> RunKey | None:
    m = _LEAF_RE.match(leaf.name)
    if not m:
        return None
    rel = leaf.relative_to(runs_root).parts
    # (gepa_<dataset>, arm, <ablation…≥1>, model, leaf) → at least 5 parts.
    if len(rel) < 5:
        return None
    dataset_dir, arm = rel[0], rel[1]
    model = rel[-2]
    ablation = "/".join(rel[2:-2])
    if not ablation:
        return None
    dataset = dataset_dir[len("gepa_") :] if dataset_dir.startswith("gepa_") else dataset_dir
    return RunKey(
        dataset=dataset,
        arm=arm,
        ablation=ablation,
        model=model,
        seed=int(m.group(1)),
        timestamp=m.group(2),
    )


def _extract_oa(holdout: dict[str, Any]) -> dict[str, OAKindScore]:
    out: dict[str, OAKindScore] = {}
    for kind, s in (holdout.get("scores") or {}).items():
        out[kind] = OAKindScore(
            kind=kind,
            n=int(s.get("n", 0)),
            n_zero=int(s.get("n_zero", 0)),
            n_perfect=int(s.get("n_perfect", 0)),
            mean_score=float(s.get("mean_score", 0.0)),
            mean_score_nonzero=float(
                s.get("mean_score_nonzero", s.get("mean_score", 0.0))
            ),
        )
    return out


def _recompute_native(
    key: RunKey,
    run_dir: Path,
    holdout: dict[str, Any],
    repo_root: Path,
    gold_cache: dict[str, dict[str, dict[str, Any]]],
) -> tuple[NativeScore | None, str | None]:
    spec = native_spec(key.dataset)
    if spec is None:
        return None, None
    preds_path = run_dir / "holdout_scores.jsonl"
    if not preds_path.exists():
        return (
            NativeScore(n=0, n_parse_failures=0, metrics={}, error="no holdout_scores.jsonl"),
            "no holdout_scores.jsonl",
        )
    if spec.gold_kind == "scierc_raw":
        gold_path = repo_root / _SCIERC_RAW_TEST
        loader = _scierc_raw_by_id
    elif spec.gold_kind == "image_dir":
        # Image runs record the holdout test *dir* (graphimg-style: labels.jsonl
        # + images/), not a single split JSONL — see run_gepa_graphimg/molecules.
        rel = holdout.get("dir")
        if not rel:
            return (
                NativeScore(n=0, n_parse_failures=0, metrics={}, error="no holdout.dir"),
                "no holdout.dir",
            )
        gold_path = repo_root / rel / "labels.jsonl"
        loader = gold_by_id_from_labels
    else:
        rel = holdout.get("path")
        if not rel:
            return (
                NativeScore(n=0, n_parse_failures=0, metrics={}, error="no holdout.path"),
                "no holdout.path",
            )
        gold_path = repo_root / rel
        loader = _examples_by_id
    cache_key = str(gold_path)
    if cache_key not in gold_cache:
        if not gold_path.exists():
            return (
                NativeScore(
                    n=0, n_parse_failures=0, metrics={}, error=f"missing gold {gold_path}"
                ),
                f"missing gold {gold_path}",
            )
        gold_cache[cache_key] = loader(gold_path)
    gold = gold_cache[cache_key]
    preds = _load_jsonl(preds_path)
    try:
        ns = spec.scorer(preds, gold, spec.task)
    except Exception as exc:  # noqa: BLE001 — never let one bad run kill the report
        return (
            NativeScore(n=0, n_parse_failures=0, metrics={}, error=repr(exc)),
            repr(exc),
        )
    return ns, None


def discover_runs(
    runs_root: Path,
    *,
    models: tuple[str, ...] | None = DEFAULT_MODELS,
    datasets: tuple[str, ...] | None = DEFAULT_DATASETS,
    repo_root: Path | None = None,
) -> Discovery:
    """Walk ``runs_root``, extract OA + native metrics, filter models, dedup.

    ``models=None`` keeps every model; ``datasets=None`` keeps every dataset.
    Non-selected datasets are skipped up front (no native recomputation).
    Duplicate timestamp batches for the same (dataset, arm, ablation, model,
    seed) are resolved by keeping the latest timestamp; drops are recorded in
    the returned :class:`Discovery`.
    """
    runs_root = Path(runs_root)
    repo_root = repo_root or runs_root.parent.parent  # data/runs -> repo
    disc = Discovery(
        models_kept=tuple(models) if models else (),
        datasets_kept=tuple(datasets) if datasets is not None else None,
    )
    gold_cache: dict[str, dict[str, dict[str, Any]]] = {}

    # Pass 1: parse + model/dataset-filter, grouping by seed-cell to dedup.
    by_seed_cell: dict[tuple[str, str, str, str, int], list[tuple[RunKey, Path]]] = (
        defaultdict(list)
    )
    for summary in sorted(runs_root.rglob("summary.json")):
        disc.n_summaries_found += 1
        leaf = summary.parent
        key = _parse_run_key(leaf, runs_root)
        if key is None:
            disc.skipped_paths.append(str(leaf.relative_to(runs_root)))
            continue
        disc.datasets_seen.add(key.dataset)
        disc.models_seen.add(key.model)
        if datasets is not None and key.dataset not in datasets:
            continue
        if models is not None and key.model not in models:
            disc.excluded_model_cells.append(
                {
                    "dataset": key.dataset,
                    "arm": key.arm,
                    "ablation": key.ablation,
                    "model": key.model,
                    "seed": key.seed,
                }
            )
            continue
        seed_cell = (key.dataset, key.arm, key.ablation, key.model, key.seed)
        by_seed_cell[seed_cell].append((key, leaf))

    # Pass 2: dedup (latest timestamp wins) + extract metrics.
    for seed_cell, entries in by_seed_cell.items():
        entries.sort(key=lambda kp: kp[0].timestamp)
        chosen_key, chosen_dir = entries[-1]
        if len(entries) > 1:
            disc.duplicate_resolutions.append(
                {
                    "dataset": chosen_key.dataset,
                    "arm": chosen_key.arm,
                    "ablation": chosen_key.ablation,
                    "model": chosen_key.model,
                    "seed": chosen_key.seed,
                    "kept": chosen_key.timestamp,
                    "dropped": [k.timestamp for k, _ in entries[:-1]],
                }
            )
        summary = json.loads((chosen_dir / "summary.json").read_text())
        holdout = summary.get("holdout", {}) or {}
        native, err = _recompute_native(
            chosen_key, chosen_dir, holdout, repo_root, gold_cache
        )
        if err:
            disc.native_errors.append(
                {
                    "dataset": chosen_key.dataset,
                    "arm": chosen_key.arm,
                    "ablation": chosen_key.ablation,
                    "model": chosen_key.model,
                    "seed": chosen_key.seed,
                    "error": err,
                }
            )
        disc.runs.append(
            RunMetrics(
                key=chosen_key,
                run_dir=chosen_dir,
                primary_kind=str(holdout.get("primary_kind", "")),
                oa=_extract_oa(holdout),
                best_val_score=summary.get("best_val_score"),
                total_metric_calls=summary.get("total_metric_calls"),
                native=native,
            )
        )
    if datasets is not None:
        disc.excluded_datasets = sorted(disc.datasets_seen - set(datasets))
    disc.runs.sort(key=lambda r: (r.key.dataset, r.key.ablation, r.key.model, r.key.arm, r.key.seed))
    return disc


# --------------------------------------------------------------------------- #
# Aggregation across seeds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stat:
    """mean ± sample std over ``n`` seeds (std is None when n < 2)."""

    mean: float
    std: float | None
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "n": self.n}


def stat_of(values: list[float]) -> Stat | None:
    """Aggregate seed values into a :class:`Stat` (None for an empty list)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Stat(
        mean=statistics.fmean(vals),
        std=statistics.stdev(vals) if len(vals) > 1 else None,
        n=len(vals),
    )


def aggregate(disc: Discovery) -> dict[str, Any]:
    """Group kept runs into per-(dataset, arm, ablation, model) cells.

    Returns a JSON-serialisable structure that *is* the single source of truth
    behind every table the renderer emits.
    """
    # dataset -> ordered kinds, primary_kind per ablation, cells.
    out: dict[str, Any] = {}
    by_dataset: dict[str, list[RunMetrics]] = defaultdict(list)
    for r in disc.runs:
        by_dataset[r.key.dataset].append(r)

    for dataset, runs in sorted(by_dataset.items()):
        # Ordered union of OA schema kinds (first-seen order).
        kinds: list[str] = []
        primary_by_ablation: dict[str, str] = {}
        for r in runs:
            for k in r.oa:
                if k not in kinds:
                    kinds.append(k)
            if r.primary_kind:
                primary_by_ablation.setdefault(r.key.ablation, r.primary_kind)

        # Group by cell.
        cell_runs: dict[tuple[str, str, str, str], list[RunMetrics]] = defaultdict(list)
        for r in runs:
            cell_runs[r.key.cell].append(r)

        cells: dict[str, Any] = {}
        for cell, crs in cell_runs.items():
            ds, arm, ablation, model = cell
            seeds = sorted(r.key.seed for r in crs)
            # OA per kind.
            oa_block: dict[str, Any] = {}
            for kind in kinds:
                present = [r.oa[kind] for r in crs if kind in r.oa]
                if not present:
                    continue
                oa_block[kind] = {
                    "mean_nonzero": _stat_dict(stat_of([s.mean_score_nonzero for s in present])),
                    "mean_all": _stat_dict(stat_of([s.mean_score for s in present])),
                    "n_zero_sum": sum(s.n_zero for s in present),
                    "n_sum": sum(s.n for s in present),
                    "n_perfect_sum": sum(s.n_perfect for s in present),
                }
            # Native (shared across kinds — recomputed from responses).
            native_block: dict[str, Any] | None = None
            spec = native_spec(dataset)
            if spec is not None:
                metric_stats: dict[str, Any] = {}
                for metric in spec.metric_order:
                    vals = [
                        r.native.metrics[metric]
                        for r in crs
                        if r.native and metric in r.native.metrics
                    ]
                    s = stat_of(vals)
                    if s is not None:
                        metric_stats[metric] = _stat_dict(s)
                pf_sum = sum(
                    r.native.n_parse_failures for r in crs if r.native and not r.native.error
                )
                n_sum = sum(r.native.n for r in crs if r.native and not r.native.error)
                errors = sorted({r.native.error for r in crs if r.native and r.native.error})
                native_block = {
                    "metrics": metric_stats,
                    "n_parse_failures_sum": pf_sum,
                    "n_sum": n_sum,
                    "errors": errors,
                }
            cells[_cell_str(cell)] = {
                "dataset": ds,
                "arm": arm,
                "ablation": ablation,
                "model": model,
                "n_seeds": len(seeds),
                "seeds": seeds,
                "primary_kind": next((r.primary_kind for r in crs if r.primary_kind), ""),
                "oa": oa_block,
                "native": native_block,
            }

        out[dataset] = {
            "kinds": kinds,
            "primary_by_ablation": primary_by_ablation,
            "ablations": sorted({r.key.ablation for r in runs}),
            "models": sorted({r.key.model for r in runs}),
            "arms": sorted({r.key.arm for r in runs}),
            "native_spec": (
                {
                    "headline_metric": spec.headline_metric,
                    "metric_order": list(spec.metric_order),
                }
                if (spec := native_spec(dataset)) is not None
                else None
            ),
            "cells": cells,
        }
    return out


def _stat_dict(s: Stat | None) -> dict[str, Any] | None:
    return s.as_dict() if s is not None else None


def _cell_str(cell: tuple[str, str, str, str]) -> str:
    return "|".join(cell)


# --------------------------------------------------------------------------- #
# Manifest loading (handles the two on-disk shapes)
# --------------------------------------------------------------------------- #


# Map a run dataset id -> the manifest path under data/<root>/splits/<schema>/<variant>.
_MANIFEST_PATHS: dict[str, str] = {
    "natural_plan_meeting": "natural_plan/splits/gepa_meeting_planning/main/manifest.json",
    "natural_plan_trip": "natural_plan/splits/gepa_trip_planning/main/manifest.json",
    "planbench_blocksworld": "planbench/splits/gepa_blocksworld/main/manifest.json",
    "scierc_native_exact": "scierc/splits/gepa_scierc/native/manifest.json",
    "amr_littleprince": "amr/splits/gepa_amr_littleprince/main/manifest.json",
    "amr_bio": "amr/splits/gepa_amr_bio/main/manifest.json",
    "docred_pcode": "docred/splits/gepa_docred_pcode/main/manifest.json",
    "docred_obf": "docred/splits/gepa_docred_obf/main/manifest.json",
    "wec_eng": "wec_eng/splits/gepa_wec_eng/main/manifest.json",
    "sentence_ordering_arxiv": "sentence_ordering/splits/gepa_arxiv/main/manifest.json",
    "sentence_ordering_rocstories": "sentence_ordering/splits/gepa_rocstories/main/manifest.json",
    "synra_nl_hard": "synra_nl/splits/gepa_synra_nl/hard/manifest.json",
    "synra_nl_pilot": "synra_nl/splits/gepa_synra_nl/pilot/manifest.json",
    "synra_codex_noobf": "synra_codex/splits/gepa_synra_codex/noobf/manifest.json",
    "synra_codex_noobf6": "synra_codex/splits/gepa_synra_codex/noobf6/manifest.json",
    "synra_codex_obf": "synra_codex/splits/gepa_synra_codex/obf/manifest.json",
    "synra_codex_obf6": "synra_codex/splits/gepa_synra_codex/obf6/manifest.json",
    "synra_sort_stated_wide": "synra_sort/splits/gepa_synra_sort/stated_wide/manifest.json",
    "synra_sort_stated_tight": "synra_sort/splits/gepa_synra_sort/stated_tight/manifest.json",
    "synra_sort_hidden_wide": "synra_sort/splits/gepa_synra_sort/hidden_wide/manifest.json",
    "synra_sort_hidden_tight": "synra_sort/splits/gepa_synra_sort/hidden_tight/manifest.json",
    # graphimg has no splits/manifest (image dirs); handled as "no manifest".
}


def load_manifest(dataset: str, repo_root: Path) -> dict[str, Any] | None:
    """Return a *normalised* manifest dict, or None when none exists.

    Normalised shape::

        {
          "dataset": str, "source": str, "task": str|None,
          "difficulty_key": str|None, "seed": <int|null>, "note": str|None,
          "splits": [{"name", "n", "bucket_counts"|None, "source_split"|None}],
          "path": str,   # the manifest file
        }
    """
    rel = _MANIFEST_PATHS.get(dataset)
    if rel is None:
        return None
    path = repo_root / "data" / rel
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return _normalise_manifest(raw, str(path.relative_to(repo_root)))


def _normalise_manifest(raw: dict[str, Any], rel_path: str) -> dict[str, Any]:
    splits: list[dict[str, Any]] = []
    if isinstance(raw.get("splits"), dict):
        # Shape A: planbench / natural_plan / sentence_ordering / synra.
        for name, val in raw["splits"].items():
            splits.append(
                {
                    "name": name,
                    "n": val.get("n"),
                    "bucket_counts": val.get("bucket_counts"),
                    "source_split": val.get("source_split"),
                }
            )
    elif any(isinstance(raw.get(n), dict) for n in ("train", "val", "test", "test_full")):
        # Shape B: scierc — split dicts live at the top level.
        for name in ("train", "val", "test", "test_full"):
            val = raw.get(name)
            if isinstance(val, dict):
                splits.append(
                    {
                        "name": name,
                        "n": val.get("n"),
                        "bucket_counts": val.get("bucket_counts"),
                        "source_split": val.get("source_split"),
                    }
                )
    else:
        # Shape C: synra_* — flat n_train / n_val / n_test counts.
        for name, key in (("train", "n_train"), ("val", "n_val"), ("test", "n_test"),
                          ("test_full", "n_test_full")):
            if raw.get(key) is not None:
                splits.append(
                    {"name": name, "n": raw.get(key), "bucket_counts": None, "source_split": None}
                )
    return {
        "dataset": raw.get("dataset"),
        "source": raw.get("source") or raw.get("source_url"),
        "task": raw.get("task"),
        "difficulty_key": raw.get("difficulty_key"),
        "seed": raw.get("seed", "—"),
        "note": raw.get("note"),
        "splits": splits,
        "path": rel_path,
    }


# --------------------------------------------------------------------------- #
# Display metadata (prose) — loaded from results_meta.yaml, never from an LLM.
# --------------------------------------------------------------------------- #

_META_PATH = Path(__file__).parent / "results_meta.yaml"
_META_CACHE: dict[str, Any] | None = None


def load_meta() -> dict[str, Any]:
    """Load (and cache) the descriptive prose from ``results_meta.yaml``.

    Returns a dict with keys ``oa_score_def``, ``dataset_display``,
    ``model_display`` and ``native_metric_defs``. Kept out of code so the only
    hand-written text in the report is reviewable in one place.
    """
    global _META_CACHE
    if _META_CACHE is None:
        _META_CACHE = yaml.safe_load(_META_PATH.read_text())
    return _META_CACHE
