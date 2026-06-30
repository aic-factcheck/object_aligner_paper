"""Walk a run directory into a `Run` dataclass.

The shape of a run directory is documented in
`scripts/run_gepa_rebel.py` (it writes the artifacts). Only the inputs the
report needs are read here:

- ``config.json``        — references the split dir and schema path.
- ``summary.json``       — gives ``arm`` and ``best_candidate_idx``.
- ``candidates.json``    — array of ``{"system_prompt": str, ...}``.
- ``generated_best_outputs_valset/task_<i>/iter_<j>_prog_<k>.json``
  — one file per (candidate ``k``, sample ``i``, iteration ``j``); the
  payload is ``{"full_assistant_response": "<JSON string>"}``.

Two split-dir layouts are supported, auto-detected:

* **Text layout** (REBEL, SciERC): ``<split_dir>/val.jsonl`` with one
  gold per line, shaped ``{"id", "title", "context", "gold"}``. Each row
  becomes a ``Sample`` whose ``context`` is the text input. Older two-way
  split runs (no ``val.jsonl``) fall back to ``test.jsonl``, which served
  as the GEPA valset under that layout.
* **Image layout** (graphimg): ``<split_dir>/labels.jsonl`` with one
  gold per line, shaped ``{"id", "image_path", "graph"}``. The runner
  points ``split_dir`` at the validation directory, so this file is the
  valset. The image is base64-encoded into ``Sample.image_data_url`` so
  the report is self-contained.

The samples list MUST line up index-for-index with GEPA's valset, because
predictions under ``generated_best_outputs_valset/task_<i>/`` are keyed by
valset index.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_IMG_MEDIA_BY_EXT: dict[str, str] = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def _encode_image_data_url(path: Path) -> str:
    mt = _IMG_MEDIA_BY_EXT.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mt};base64,{b64}"


_PRED_RE = re.compile(r"^iter_(\d+)_prog_(\d+)\.json$")
_TASK_RE = re.compile(r"^task_(\d+)$")


@dataclass(frozen=True)
class Candidate:
    idx: int
    system_prompt: str
    is_best: bool


@dataclass(frozen=True)
class Sample:
    idx: int
    sample_id: str
    title: str
    context: str
    gold: dict[str, Any]
    # Image-layout samples carry an image instead of textual context. Both
    # fields are None for text-layout runs.
    image_path: Path | None = None
    image_data_url: str | None = None


@dataclass(frozen=True)
class Pair:
    pred_raw: dict[str, Any] | None
    pred_aligned: dict[str, Any] | None
    score: float | None
    feedback: str | None
    parse_error: str | None
    available: bool
    iter_idx: int | None = None


@dataclass(frozen=True)
class Run:
    run_dir: Path
    run_label: str
    arm: str
    schema: dict[str, Any]
    candidates: list[Candidate]
    samples: list[Sample]
    pairs: dict[tuple[int, int], Pair] = field(default_factory=dict)
    best_candidate_idx: int | None = None
    # Run-level context surfaced in the report header / panels.
    config: dict[str, Any] = field(default_factory=dict)
    task_lm_model: str | None = None
    reflection_lm_model: str | None = None
    main_schema_name: str = "schema"
    # (stem, loaded schema dict) for every config['cross_schema_paths'] entry
    # we could load. Scored per cell alongside the main schema.
    cross_schemas: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    # summary['holdout'] verbatim (best candidate on the test split), or None.
    holdout: dict[str, Any] | None = None


class _ParseError(ValueError):
    pass


def _parse_pred_text(text: str) -> dict[str, Any]:
    """Same tolerant-ish parser as evaluator._parse_json — kept local to avoid
    pulling the evaluator's heavier import surface."""
    text = text.strip()
    if not text:
        raise _ParseError("empty response")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ParseError(
            f"JSONDecodeError: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        ) from exc
    if not isinstance(obj, dict):
        raise _ParseError(
            f"top-level value must be an object, got {type(obj).__name__}"
        )
    return obj


def _load_schema(path: Path) -> dict[str, Any]:
    import json5

    with path.open("r") as f:
        obj = json5.loads(f.read())
    if not isinstance(obj, dict):
        raise ValueError(f"schema at {path} must be a JSON object")
    return obj


def _load_text_samples(test_jsonl: Path) -> list[Sample]:
    samples: list[Sample] = []
    with test_jsonl.open("r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                Sample(
                    idx=i,
                    sample_id=str(row.get("id", f"sample_{i}")),
                    title=str(row.get("title", "")),
                    context=str(row.get("context", "")),
                    gold=row["gold"],
                )
            )
    return samples


def _load_image_samples(labels_jsonl: Path) -> list[Sample]:
    """Load graphimg-shaped samples: each row has id/image_path/graph.

    The image_path is resolved relative to ``labels_jsonl.parent`` (matching
    ``object_aligner_exp.datasets.graphimg.load_graphimg_split``) and the
    file's bytes are inlined as a data URL so the report stays
    self-contained.
    """
    samples: list[Sample] = []
    base = labels_jsonl.parent
    with labels_jsonl.open("r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rel = row.get("image_path")
            graph = row.get("graph") or {}
            if not isinstance(rel, str) or not isinstance(graph, dict):
                raise ValueError(
                    f"{labels_jsonl}:{i + 1}: image-layout row needs "
                    f"'image_path' and 'graph' keys; got {list(row.keys())}"
                )
            abspath = (base / rel).resolve()
            if not abspath.is_file():
                raise FileNotFoundError(
                    f"{labels_jsonl}:{i + 1}: image not found at {abspath}"
                )
            # The dataset stores an internal sample id inside `graph` that
            # the model isn't asked to reproduce; drop it.
            gold = {k: v for k, v in graph.items() if k != "id"}
            samples.append(
                Sample(
                    idx=i,
                    sample_id=str(row.get("id", f"sample_{i}")),
                    title="",
                    context="",
                    gold=gold,
                    image_path=abspath,
                    image_data_url=_encode_image_data_url(abspath),
                )
            )
    return samples


def _resolve_split_dir(config: dict[str, Any]) -> Path:
    """Find the directory containing the valset gold file.

    Honours the canonical ``split_dir`` key. Falls back to ``val_dir`` for
    graphimg-shaped configs written before the loader supported them, then
    to ``split_root / "validation"`` if only the root is known.
    """
    for key in ("split_dir", "val_dir"):
        v = config.get(key)
        if isinstance(v, str) and v:
            return Path(v)
    root = config.get("split_root")
    if isinstance(root, str) and root:
        return Path(root) / "validation"
    raise ValueError(
        "config.json must define 'split_dir' (or 'val_dir' / 'split_root' "
        "for graphimg-shaped runs)"
    )


def _detect_samples_file(split_dir: Path) -> tuple[Path, str]:
    """Return ``(samples_path, layout)`` where layout is ``"text"`` or ``"image"``.

    The samples file MUST be the GEPA valset (matches the indexing of
    ``generated_best_outputs_valset/task_<i>/``). Three-way-split runs put
    that under ``val.jsonl``; older two-way-split runs only have
    ``test.jsonl`` and used it as the valset, so we fall back to it.
    """
    val_jsonl = split_dir / "val.jsonl"
    test_jsonl = split_dir / "test.jsonl"
    labels_jsonl = split_dir / "labels.jsonl"
    if val_jsonl.is_file():
        return val_jsonl, "text"
    if test_jsonl.is_file():
        return test_jsonl, "text"
    if labels_jsonl.is_file():
        return labels_jsonl, "image"
    raise FileNotFoundError(
        f"{split_dir}: no val.jsonl / test.jsonl (text layout) or "
        f"labels.jsonl (image layout) found"
    )


def _load_candidates(run_dir: Path, best_idx: int | None) -> list[Candidate]:
    with (run_dir / "candidates.json").open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"candidates.json at {run_dir} must be a list")
    return [
        Candidate(
            idx=i,
            system_prompt=str(c.get("system_prompt", "")),
            is_best=(best_idx is not None and i == best_idx),
        )
        for i, c in enumerate(data)
    ]


def _discover_predictions(run_dir: Path) -> dict[tuple[int, int], tuple[Path, int]]:
    """Walk ``generated_best_outputs_valset/`` and key by ``(cand_idx, sample_idx)``.

    When several iterations exist for the same (cand, sample), keep the file
    with the highest ``iter`` number (later iterations supersede earlier ones).
    Returns ``{(k, i): (path, iter_idx)}``.
    """
    root = run_dir / "generated_best_outputs_valset"
    out: dict[tuple[int, int], tuple[Path, int]] = {}
    if not root.is_dir():
        return out
    for task_dir in root.iterdir():
        m = _TASK_RE.match(task_dir.name)
        if not m or not task_dir.is_dir():
            continue
        sample_idx = int(m.group(1))
        for pred_file in task_dir.iterdir():
            mp = _PRED_RE.match(pred_file.name)
            if not mp:
                continue
            iter_idx = int(mp.group(1))
            cand_idx = int(mp.group(2))
            key = (cand_idx, sample_idx)
            prev = out.get(key)
            if prev is None or iter_idx > prev[1]:
                out[key] = (pred_file, iter_idx)
    return out


def load_pair_predictions(
    discovered: dict[tuple[int, int], tuple[Path, int]],
) -> dict[tuple[int, int], tuple[dict[str, Any] | None, str | None, int, str | None]]:
    """Load every discovered prediction file.

    Returns ``{(k, i): (pred_dict_or_None, parse_error_or_None, iter_idx, raw_text_or_None)}``.
    Raw text is kept so a caller can sanity-check it against the task_lm.jsonl
    transcript.
    """
    out: dict[tuple[int, int], tuple[dict[str, Any] | None, str | None, int, str | None]] = {}
    for key, (path, iter_idx) in discovered.items():
        with path.open("r") as f:
            wrapper = json.load(f)
        raw_text = wrapper.get("full_assistant_response")
        if raw_text is None:
            out[key] = (None, "missing 'full_assistant_response' field", iter_idx, None)
            continue
        try:
            obj = _parse_pred_text(str(raw_text))
        except _ParseError as exc:
            out[key] = (None, str(exc), iter_idx, str(raw_text))
            continue
        out[key] = (obj, None, iter_idx, str(raw_text))
    return out


def _load_task_lm_records(
    jsonl_path: Path,
    candidates: list[Candidate],
    samples: list[Sample],
) -> tuple[dict[tuple[int, int], tuple[dict[str, Any] | None, str | None, str]], dict[str, int]]:
    """Walk ``task_lm.jsonl`` and reconstruct every (cand, sample) prediction.

    Returns ``(preds, stats)`` where
    ``preds[(cand_idx, sample_idx)] = (parsed_obj_or_None, parse_error_or_None, raw_text)``
    and ``stats`` carries diagnostic counts:
    ``{"records", "matched", "unmatched_sys", "unmatched_user", "train_skipped"}``.

    Mapping policy:
    - Candidate identity comes from an exact match between the record's
      first system message and a `candidates.json` system prompt
      (GEPA only varies that field, so it is unique).
    - Sample identity comes from the last user message. For text-layout
      runs that's an exact match between the user-message string and a
      gold ``context``. For image-layout runs the user content is a list
      of OpenAI content parts; we extract the ``image_url`` and match it
      against each sample's ``image_data_url`` (also a data URL of the
      same bytes — equal data → equal string).
    - Records whose user content doesn't match any val sample are
      training-set evaluations (used during reflection minibatches);
      they're silently skipped.
    - On collision (multiple records for the same (cand, sample)), the
      latest record wins. `task_lm` runs at temperature 0 so duplicates
      are usually byte-identical anyway.
    """
    sys_to_cand: dict[str, int] = {c.system_prompt: c.idx for c in candidates}
    ctx_to_sample: dict[str, int] = {s.context: s.idx for s in samples if not s.image_data_url}
    image_to_sample: dict[str, int] = {
        s.image_data_url: s.idx for s in samples if s.image_data_url
    }

    preds: dict[tuple[int, int], tuple[dict[str, Any] | None, str | None, str]] = {}
    stats = {"records": 0, "matched": 0, "unmatched_sys": 0, "unmatched_user": 0, "train_skipped": 0}

    if not jsonl_path.is_file():
        return preds, stats

    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["records"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("label") != "task_lm":
                continue
            messages = rec.get("messages") or []
            if not messages:
                continue
            sys_msg = next((m for m in messages if m.get("role") == "system"), None)
            user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if sys_msg is None or user_msg is None:
                continue
            cand_idx = sys_to_cand.get(str(sys_msg.get("content", "")))
            if cand_idx is None:
                stats["unmatched_sys"] += 1
                continue

            user_content = user_msg.get("content")
            sample_idx: int | None = None
            if isinstance(user_content, list):
                # Image-layout: scan content parts for an image_url.
                for part in user_content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url")
                        if isinstance(url, str):
                            sample_idx = image_to_sample.get(url)
                            break
            else:
                # Text-layout: full user string is the context.
                sample_idx = ctx_to_sample.get(str(user_content or ""))

            if sample_idx is None:
                # Probably a training-set rollout used during reflection.
                stats["train_skipped"] += 1
                continue

            response = rec.get("response")
            if response is None:
                continue
            response = str(response)
            try:
                obj = _parse_pred_text(response)
                preds[(cand_idx, sample_idx)] = (obj, None, response)
            except _ParseError as exc:
                preds[(cand_idx, sample_idx)] = (None, str(exc), response)
            stats["matched"] += 1
    return preds, stats


@dataclass(frozen=True)
class LoadedPredictions:
    """All per-(candidate, sample) prediction data, merged across sources.

    ``preds`` is keyed by ``(cand_idx, sample_idx)``. Every cell has been
    parsed (or recorded as a parse failure). ``pareto`` flags the cells
    that GEPA stored under ``generated_best_outputs_valset/`` as strict
    per-sample improvements; its value is the iteration index at which
    that improvement was recorded.
    """

    preds: dict[tuple[int, int], tuple[dict[str, Any] | None, str | None]]
    pareto: dict[tuple[int, int], int]
    task_lm_stats: dict[str, int]
    pareto_count: int
    response_mismatches: list[tuple[int, int]]


def _collect_predictions(
    run_dir: Path,
    candidates: list[Candidate],
    samples: list[Sample],
) -> LoadedPredictions:
    pareto_files = load_pair_predictions(_discover_predictions(run_dir))
    task_lm_preds, stats = _load_task_lm_records(
        run_dir / "task_lm.jsonl", candidates, samples
    )

    # Sanity check: if we found valset prediction files but couldn't
    # content-match a single task_lm.jsonl record to a sample, then the
    # samples list isn't the valset GEPA actually evaluated against — every
    # pair in the report would line up gold[i] with the wrong prediction.
    # Refuse to render rather than silently produce a misleading report.
    if (
        pareto_files
        and stats["records"] > 0
        and stats["matched"] == 0
        and samples
    ):
        raise RuntimeError(
            f"{run_dir}: {stats['records']} task_lm.jsonl records but none "
            f"matched any of {len(samples)} samples by context. The samples "
            f"file loaded by the report doesn't line up with GEPA's valset "
            f"— predictions in generated_best_outputs_valset/ would be "
            f"paired with the wrong gold. Check _detect_samples_file()."
        )

    pareto_meta: dict[tuple[int, int], int] = {
        key: iter_idx for key, (_, _, iter_idx, _) in pareto_files.items()
    }

    merged: dict[tuple[int, int], tuple[dict[str, Any] | None, str | None]] = {}
    for key, (obj, err, _raw) in task_lm_preds.items():
        merged[key] = (obj, err)
    for key, (obj, err, _iter_idx, _raw) in pareto_files.items():
        # task_lm.jsonl is the authoritative source for the latest response;
        # only fall back to the pareto file when this cell didn't appear there.
        merged.setdefault(key, (obj, err))

    mismatches: list[tuple[int, int]] = []
    for key in pareto_files.keys() & task_lm_preds.keys():
        p_raw = pareto_files[key][3]
        t_raw = task_lm_preds[key][2]
        if p_raw is not None and t_raw is not None:
            if (p_raw or "").strip() != (t_raw or "").strip():
                mismatches.append(key)

    return LoadedPredictions(
        preds=merged,
        pareto=pareto_meta,
        task_lm_stats=stats,
        pareto_count=len(pareto_meta),
        response_mismatches=mismatches,
    )


def load_run_skeleton(run_dir: Path) -> tuple[Run, LoadedPredictions]:
    """Load everything except the OA-driven aligned predictions.

    Returns ``(run, predictions)`` where ``run.pairs`` is empty. Callers
    fill ``pairs`` after running OA. This split keeps OA logic out of the
    loader.
    """
    run_dir = run_dir.resolve()
    with (run_dir / "config.json").open("r") as f:
        config = json.load(f)

    # summary.json is only written when gepa.optimize() returns; tolerate its
    # absence so the report is renderable mid-run (and for crashed/interrupted
    # runs after the fact). When missing, fall back to `arm` from config and
    # leave `best_candidate_idx` unset.
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r") as f:
            summary = json.load(f)
    else:
        summary = {}

    if "schema_path" not in config:
        raise ValueError(
            f"{run_dir / 'config.json'} is missing 'schema_path'. "
            "This run uses an older layout (e.g. b1_baseline) that the report "
            "generator does not yet support. Only GEPA-arm runs are handled in v1."
        )
    split_dir = _resolve_split_dir(config)
    samples_path, layout = _detect_samples_file(split_dir)
    schema_path = Path(config["schema_path"])

    if layout == "image":
        samples = _load_image_samples(samples_path)
    else:
        samples = _load_text_samples(samples_path)
    best_idx = summary.get("best_candidate_idx")
    candidates = _load_candidates(run_dir, best_idx)
    schema = _load_schema(schema_path)

    # Cross schemas (scored per cell alongside the main schema). Tolerate
    # unreadable / missing files so the report still renders.
    cross_schemas: list[tuple[str, dict[str, Any]]] = []
    for cp in config.get("cross_schema_paths") or []:
        try:
            cross_schemas.append((Path(cp).stem, _load_schema(Path(cp))))
        except (OSError, ValueError) as exc:
            print(f"[render] skipping cross schema {cp}: {exc}", file=__import__("sys").stderr)

    predictions = _collect_predictions(run_dir, candidates, samples)

    arm = str(summary.get("arm") or config.get("arm") or "")
    task_lm_model = (config.get("task_lm") or {}).get("model")
    reflection_lm_model = (config.get("reflection_lm") or {}).get("model")
    run = Run(
        run_dir=run_dir,
        run_label=run_dir.name,
        arm=arm,
        schema=schema,
        candidates=candidates,
        samples=samples,
        pairs={},
        best_candidate_idx=best_idx,
        config=config,
        task_lm_model=task_lm_model,
        reflection_lm_model=reflection_lm_model,
        main_schema_name=schema_path.stem,
        cross_schemas=cross_schemas,
        holdout=summary.get("holdout"),
    )
    return run, predictions
