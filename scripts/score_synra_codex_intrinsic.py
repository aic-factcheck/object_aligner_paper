"""Score the synra_codex intrinsic benchmark under three RA configs.

Reads each row's pre-materialised ``(gold, pred)`` pair from
``data/synra_codex/intrinsic/v1/<split>/labels.jsonl`` and scores it under
two Object Aligner configurations:

  * **ra**     — RA-on schema: full referential alignment (OA's default,
                 including its built-in structural tie-break).
  * **strict** — RA-off schema (``plain``): identifiers compared by value.

Emits per-row scores plus a per-cell summary under
``<root>/<split>_scores.jsonl`` and ``<split>_summary.json``.

Usage::

    uv run python scripts/score_synra_codex_intrinsic.py \\
        [--split {train,validation,test,all}] \\
        [--root data/synra_codex/intrinsic/v1]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

# Project imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from object_aligner_exp.config import ExpConfig  # noqa: E402
from object_aligner_exp.schemas import load_synra_codex_schema  # noqa: E402

from object_aligner import ObjectAligner  # noqa: E402


SPLITS: tuple[str, ...] = ("train", "validation", "test")
CONFIGS: tuple[str, ...] = ("ra", "strict")


# --- io --------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- aligner wrappers ------------------------------------------------------


def _build_aligners(cfg: ExpConfig) -> dict[str, ObjectAligner]:
    """Two aligners keyed by config name: ``ra`` (referential alignment, OA's
    defaults) and ``strict`` (the plain ablation that drops idScope/ref). Both
    flagged to compute confidence so per-row introspection needs no second pass.
    """
    ra = load_synra_codex_schema(cfg, kind="ra")
    strict = load_synra_codex_schema(cfg, kind="strict")
    common = dict(warn_on_ambiguous_mapping=True, compute_confidence=True)
    return {
        "ra": ObjectAligner(ra, **common),
        "strict": ObjectAligner(strict, **common),
    }


def _confidence_from_debug(debug: Any) -> float | None:
    """Top-level confidence from a ``metric(..., debug=True)`` payload.

    OA omits the ``confidence`` key when it equals 1.0, so a perfect match
    returns ``None`` here — callers treat ``None`` as fully confident.
    """
    if isinstance(debug, dict):
        conf = debug.get("confidence")
        if isinstance(conf, (int, float)):
            return float(conf)
    return None


def _score_one(
    aligner: ObjectAligner, gold: dict[str, Any], pred: dict[str, Any]
) -> tuple[float, float | None, bool]:
    """Return ``(score, confidence_or_None, ambiguous_flag)``.

    ``ambiguous_flag`` is True iff OA raised an "Ambiguous mapping" UserWarning
    during the call (the real WL signal — ``ra_nowl`` trips it on twins).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        out = aligner.metric(gold, pred, debug=True)
    ambiguous = any("Ambiguous mapping" in str(w.message) for w in caught)
    score = float(out["score"])
    conf = _confidence_from_debug(out.get("debug"))
    return score, conf, ambiguous


# --- aggregation -----------------------------------------------------------


def _stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _by_axis(rows: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    buckets: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(r[axis], []).append(r)
    out: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: (isinstance(k, str), k)):
        rs = buckets[key]
        entry: dict[str, Any] = {axis: key, "n": len(rs)}
        for cfg_name in CONFIGS:
            entry[f"score_{cfg_name}"] = _stat([r[f"score_{cfg_name}"] for r in rs])
        out.append(entry)
    return out


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {
        f"score_{cfg_name}": _stat([r[f"score_{cfg_name}"] for r in rows])
        for cfg_name in CONFIGS
    }
    return {
        "n_rows": len(rows),
        "configs": list(CONFIGS),
        "overall": overall,
        "by_twin_density": _by_axis(rows, "twin_density"),
        "by_n_people": _by_axis(rows, "n_people"),
    }


# --- driver ----------------------------------------------------------------


def _score_split(
    aligners: dict[str, ObjectAligner],
    split_dir: Path,
    out_root: Path,
    split_name: str,
) -> None:
    labels_path = split_dir / "labels.jsonl"
    if not labels_path.exists():
        print(f"  {split_name}: no labels.jsonl, skipping", flush=True)
        return
    rows_in = _read_jsonl(labels_path)
    print(f"  {split_name}: scoring {len(rows_in)} rows × {len(CONFIGS)} configs …", flush=True)
    out_rows: list[dict[str, Any]] = []
    for r in rows_in:
        gold, pred = r["gold"], r["pred"]
        entry: dict[str, Any] = {
            "id": r["id"],
            "n_people": r["n_people"],
            "n_companies": r["n_companies"],
            "twin_density": r["twin_density"],
            "pred_idx": r["pred_idx"],
            "gold_seed": r["gold_seed"],
        }
        for cfg_name, aligner in aligners.items():
            score, conf, ambiguous = _score_one(aligner, gold, pred)
            entry[f"score_{cfg_name}"] = score
            entry[f"confidence_{cfg_name}"] = conf
            entry[f"ambiguous_{cfg_name}"] = ambiguous
        out_rows.append(entry)

    _write_jsonl(out_root / f"{split_name}_scores.jsonl", out_rows)
    summary = _summarise(out_rows)
    (out_root / f"{split_name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    means = "  ".join(
        f"{c}={summary['overall'][f'score_{c}']['mean']:.3f}" for c in CONFIGS
    )
    print(f"    mean score — {means}", flush=True)


def _parse_args() -> argparse.Namespace:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=cfg.synra_codex_intrinsic_v2)
    p.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = ExpConfig()
    root: Path = args.root
    if not root.exists():
        print(f"missing root {root}", file=sys.stderr)
        return 2

    aligners = _build_aligners(cfg)
    splits = SPLITS if args.split == "all" else (args.split,)
    print(f"scoring under {CONFIGS} at {root}", flush=True)
    for name in splits:
        _score_split(aligners, root / name, root, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
