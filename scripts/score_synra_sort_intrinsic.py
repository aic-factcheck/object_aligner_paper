"""Score the synra_sort intrinsic benchmark: fixed vs hungarian + PMR/τ.

Reads ``<root>/<split>/labels.jsonl`` and scores each ``(gold, pred)`` order
pair under:

  * **fixed**     — OA fixed-order DP alignment (the feature under test).
  * **hungarian** — OA order-blind Hungarian alignment (the STED stand-in).
  * **pmr**       — exact-match (1.0 iff pred == gold), via the official
                    ``sentence_ordering_eval`` parser (length ≠ N → parse
                    failure → 0; the "exact-match can't grade length change"
                    behaviour we want to contrast).
  * **tau**       — Kendall τ between pred and gold (0 on parse failure).

Emits ``<root>/<split>_scores.jsonl`` + ``<split>_summary.json``.

Usage::

    uv run python scripts/score_synra_sort_intrinsic.py \\
        [--split {train,validation,test,all}] \\
        [--root data/synra_sort/intrinsic/v1]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

# Project imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from object_aligner_exp.config import ExpConfig  # noqa: E402
from object_aligner_exp.schemas import load_schema_from_path  # noqa: E402
from object_aligner_exp.sentence_ordering_eval import (  # noqa: E402
    kendall_tau,
    parse_order,
    pmr,
)

from object_aligner import ObjectAligner  # noqa: E402


SPLITS: tuple[str, ...] = ("train", "validation", "test")
SCORERS: tuple[str, ...] = ("fixed", "hungarian", "pmr", "tau")


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


def _build_aligners(cfg: ExpConfig) -> tuple[ObjectAligner, ObjectAligner]:
    fixed = ObjectAligner(load_schema_from_path(cfg.synra_sort_fixed_schema_path))
    hung = ObjectAligner(load_schema_from_path(cfg.synra_sort_hungarian_schema_path))
    return fixed, hung


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
        for s in SCORERS:
            entry[s] = _stat([float(r[s]) for r in rs])
        out.append(entry)
    return out


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "scorers": list(SCORERS),
        "n_parse_failures": sum(1 for r in rows if not r["parse_ok"]),
        "overall": {s: _stat([float(r[s]) for r in rows]) for s in SCORERS},
        "by_family": _by_axis(rows, "family"),
        "by_n": _by_axis(rows, "n"),
    }


def _score_split(
    fixed: ObjectAligner,
    hung: ObjectAligner,
    split_dir: Path,
    out_root: Path,
    split_name: str,
) -> None:
    labels_path = split_dir / "labels.jsonl"
    if not labels_path.exists():
        print(f"  {split_name}: no labels.jsonl, skipping", flush=True)
        return
    rows_in = _read_jsonl(labels_path)
    print(f"  {split_name}: scoring {len(rows_in)} rows …", flush=True)
    out_rows: list[dict[str, Any]] = []
    for r in rows_in:
        gold, pred = r["gold"], r["pred"]
        gold_indices = gold["indices"]
        n = r["n"]
        score_fixed = float(fixed.metric(gold, pred)["score"])
        score_hung = float(hung.metric(gold, pred)["score"])
        parsed = parse_order(pred, n)
        if parsed is None:
            parse_ok, is_pmr, tau = False, 0.0, 0.0
        else:
            parse_ok = True
            is_pmr = 1.0 if pmr(parsed, gold_indices) else 0.0
            tau = kendall_tau(parsed, gold_indices)
        out_rows.append(
            {
                "id": r["id"],
                "n": n,
                "family": r["family"],
                "k": r["k"],
                "kendall_distance": r["kendall_distance"],
                "len_pred": r["len_pred"],
                "fixed": score_fixed,
                "hungarian": score_hung,
                "pmr": is_pmr,
                "tau": tau,
                "parse_ok": parse_ok,
            }
        )

    _write_jsonl(out_root / f"{split_name}_scores.jsonl", out_rows)
    summary = _summarise(out_rows)
    (out_root / f"{split_name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    o = summary["overall"]
    print(
        f"    mean — fixed={o['fixed']['mean']:.3f} hungarian={o['hungarian']['mean']:.3f} "
        f"pmr={o['pmr']['mean']:.3f}  (parse failures: {summary['n_parse_failures']})",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=cfg.synra_sort_intrinsic_v2)
    p.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = ExpConfig()
    root: Path = args.root
    if not root.exists():
        print(f"missing root {root}", file=sys.stderr)
        return 2
    fixed, hung = _build_aligners(cfg)
    splits = SPLITS if args.split == "all" else (args.split,)
    print(f"scoring fixed/hungarian/pmr/tau at {root}", flush=True)
    for name in splits:
        _score_split(fixed, hung, root / name, root, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
