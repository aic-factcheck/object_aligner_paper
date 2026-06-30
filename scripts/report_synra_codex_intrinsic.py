"""Compose a markdown report from a synra_codex intrinsic scoring run.

Reads two inputs under ``--root`` and writes ``<root>/report.md``. Every
metric is reported for the two configs ``ra`` (referential alignment) and
``strict`` (the ``plain`` ablation):

  * ``<split>_scores.jsonl`` — the **id-equivariance probe** (K independent
    id relabelings per gold): per-gold score variance (a relabel-invariant
    score has variance 0), means, twin-density / size marginals.
  * ``per_op_scores.jsonl`` — the **per-operation sweep**
    (``scripts/per_op_synra_codex_intrinsic.py``; run it first), with-relabel
    arm: **discrimination** AUROC of the score as a detector of
    "k = 0 (relabel only) vs. k ≥ 1", and **sensitivity** Spearman & Kendall
    of score vs. the number of applications k, pooled and per operation.

All statistics are computed in-script (no scipy / matplotlib).

Usage::

    uv run python scripts/report_synra_codex_intrinsic.py \\
        [--split test] [--root data/synra_codex/intrinsic/v1]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from object_aligner_exp.config import ExpConfig  # noqa: E402


CONFIGS: tuple[str, ...] = ("ra", "strict")

# Readable names for the per-op sweep (matches per_op_synra_codex_intrinsic.py).
OP_LABELS: dict[str, str] = {
    "cat_relabel": "categorical relabel",
    "ref_misroute": "reference rerouting",
    "node_delete": "record deletion",
    "edge_delete": "edge deletion",
    "node_add": "record insertion",
    "edge_add": "edge insertion",
}


# --- statistics (no scipy) ------------------------------------------------


def _ranks(values: list[float]) -> list[float]:
    """Fractional (average-tie) ranks. Lowest value gets rank 1."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    """Kendall's tau-b — pairs counted with tie adjustment."""
    n = len(xs)
    if n < 2:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann–Whitney U formulation; ``labels`` are 0/1."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    ranks = _ranks(scores)
    rank_by_label_1 = sum(ranks[i] for i, l in enumerate(labels) if l == 1)
    n1, n0 = len(pos), len(neg)
    u1 = rank_by_label_1 - n1 * (n1 + 1) / 2
    return u1 / (n1 * n0)


# --- io / helpers ----------------------------------------------------------


def _read_scores(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return (
        statistics.fmean(values),
        statistics.pstdev(values) if len(values) > 1 else 0.0,
    )


def _fmt(x: float | None, width: int = 5) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  —  "
    return f"{x:.{width - 2}f}"


def _gold_key(r: dict[str, Any]) -> tuple:
    return (r["gold_seed"], r["n_people"], r["n_companies"], r["twin_density"])


# --- metric sections -------------------------------------------------------


def _id_equivariance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_gold_key(r)].append(r)
    per_cfg: dict[str, list[float]] = {c: [] for c in CONFIGS}
    n_groups = 0
    for grp in groups.values():
        if len(grp) < 2:
            continue
        n_groups += 1
        for c in CONFIGS:
            per_cfg[c].append(statistics.pvariance([g[f"score_{c}"] for g in grp]))
    out: dict[str, Any] = {"n_groups": n_groups}
    for c in CONFIGS:
        vals = per_cfg[c]
        out[c] = {
            "mean_var": statistics.fmean(vals) if vals else float("nan"),
            "max_var": max(vals) if vals else float("nan"),
        }
    return out


def _discrimination(op_rows: list[dict[str, Any]]) -> dict[str, float]:
    """AUROC of "k = 0 (relabel only) vs. k ≥ 1" on the with-relabel per-op arm."""
    labels = [1 if r["k"] == 0 else 0 for r in op_rows]
    return {c: auroc([r[f"score_{c}"] for r in op_rows], labels) for c in CONFIGS}


def _sensitivity(op_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Pooled correlation of score vs. k on the with-relabel per-op arm."""
    ks = [float(r["k"]) for r in op_rows]
    out: dict[str, dict[str, float]] = {}
    for c in CONFIGS:
        scores = [r[f"score_{c}"] for r in op_rows]
        out[c] = {
            "spearman": spearman(scores, ks),
            "kendall_tau_b": kendall_tau_b(scores, ks),
        }
    return out


def _sensitivity_per_op(
    op_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Per-operation Spearman of score vs. k on the with-relabel per-op arm."""
    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in op_rows:
        by_op[r["op"]].append(r)
    out: dict[str, dict[str, float]] = {}
    for op, rs in by_op.items():
        ks = [float(r["k"]) for r in rs]
        out[op] = {c: spearman([r[f"score_{c}"] for r in rs], ks) for c in CONFIGS}
    return out


# --- rendering -------------------------------------------------------------


def _section_headline(rows: list[dict[str, Any]]) -> str:
    parts = [
        "## Headline numbers\n",
        "| config | mean | stdev | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in CONFIGS:
        scores = [r[f"score_{c}"] for r in rows]
        m, sd = _mean_std(scores)
        parts.append(
            f"| {c} | {m:.3f} | {sd:.3f} | {min(scores):.3f} | {max(scores):.3f} |"
        )
    return "\n".join(parts) + "\n"


def _section_id_equivariance(stats: dict[str, Any]) -> str:
    parts = [
        "## Id-equivariance — permutation variance ($K$ relabeled pred copies / gold)\n",
        "_RA must be ≈ 0; the plain ablation measures the noise arbitrary id "
        "relabelings inject._\n",
        f"_Qualifying gold groups (≥ 2 preds): {stats['n_groups']}._\n",
        "| config | mean per-gold variance | max per-gold variance |",
        "| --- | --- | --- |",
    ]
    for c in CONFIGS:
        parts.append(
            f"| {c} | {stats[c]['mean_var']:.4e} | {stats[c]['max_var']:.4e} |"
        )
    return "\n".join(parts) + "\n"


def _section_discrimination(stats: dict[str, float]) -> str:
    parts = [
        "## Discrimination — AUROC (per-op sweep, with relabel: k = 0 vs k ≥ 1)\n",
        "_Computed on `per_op_scores.jsonl` (with-relabel arm). RA < 1.0 comes "
        "from perturbed candidates that are graph-isomorphic to the gold "
        "(e.g. a reference rerouted between property-twins) — RA correctly "
        "scores those 1.0._\n",
        "| config | AUROC |",
        "| --- | --- |",
    ]
    for c in CONFIGS:
        parts.append(f"| {c} | {_fmt(stats[c], 5)} |")
    return "\n".join(parts) + "\n"


def _section_sensitivity(
    pooled: dict[str, dict[str, float]],
    per_op: dict[str, dict[str, float]],
) -> str:
    parts = [
        "## Sensitivity — correlation with the number of applications k "
        "(per-op sweep, with relabel)\n",
        "_More applications of an op should lower the score, so the desirable "
        "sign is **negative**. `ref_misroute` is the diagnostic row: only RA "
        "responds; plain has already mismatched every relabeled id._\n",
        "| operation | ra Spearman | strict Spearman |",
        "| --- | --- | --- |",
    ]
    for op in OP_LABELS:
        if op in per_op:
            parts.append(
                f"| {OP_LABELS[op]} | {per_op[op]['ra']:+.3f} "
                f"| {per_op[op]['strict']:+.3f} |"
            )
    parts.append(
        f"| **pooled (all ops)** | **{pooled['ra']['spearman']:+.3f}** "
        f"| **{pooled['strict']['spearman']:+.3f}** |"
    )
    parts += [
        "",
        "_Pooled Kendall τ-b_: "
        + ", ".join(f"{c} {pooled[c]['kendall_tau_b']:+.3f}" for c in CONFIGS)
        + ".",
    ]
    return "\n".join(parts) + "\n"


def _twin_bin(t: float) -> str:
    """Bin the sampled (continuous) twin density; t = 1.0 (the pinned hardest
    case) gets its own bucket."""
    if t == 1.0:
        return "=1.0"
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        if lo <= t < hi:
            return f"[{lo},{hi})"
    return "?"


def _section_by_axis(rows: list[dict[str, Any]], axis: str) -> str:
    key = _twin_bin if axis == "twin_density" else lambda v: v
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[key(r[axis])].append(r)
    lines = [
        f"### by {axis}" + (" (binned)" if axis == "twin_density" else ""),
        "",
        f"| {axis} | n | " + " | ".join(CONFIGS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in CONFIGS) + " |",
    ]
    for k in sorted(buckets, key=lambda x: (isinstance(x, str), x)):
        rs = buckets[k]
        label = f"{k:.2f}" if isinstance(k, float) else str(k)
        means = [f"{statistics.fmean(r[f'score_{c}'] for r in rs):.3f}" for c in CONFIGS]
        lines.append(f"| {label} | {len(rs)} | " + " | ".join(means) + " |")
    lines.append("")
    return "\n".join(lines)


def render_report(
    rows: list[dict[str, Any]],
    op_rows: list[dict[str, Any]],
    split: str,
    source: Path,
    op_source: Path,
) -> str:
    parts: list[str] = []
    parts.append(f"# synra_codex intrinsic — `{split}` split report")
    parts.append("")
    parts.append(
        f"_Equivariance probe_: `{source}` ({len(rows)} rows) &nbsp;&nbsp; "
        f"_Per-op sweep_: `{op_source}` ({len(op_rows)} with-relabel rows) "
        f"&nbsp;&nbsp; _Configs_: {', '.join(CONFIGS)}"
    )
    parts.append("")
    parts.append(_section_headline(rows))
    parts.append("")
    parts.append(_section_id_equivariance(_id_equivariance(rows)))
    parts.append("")
    parts.append(_section_discrimination(_discrimination(op_rows)))
    parts.append("")
    parts.append(_section_sensitivity(_sensitivity(op_rows), _sensitivity_per_op(op_rows)))
    parts.append("")
    parts.append("## Marginal breakdowns (equivariance probe)")
    parts.append("")
    for axis in ("twin_density", "n_people"):
        parts.append(_section_by_axis(rows, axis))
    return "\n".join(parts)


def _parse_args() -> argparse.Namespace:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=cfg.synra_codex_intrinsic_v2)
    p.add_argument("--split", choices=("train", "validation", "test"), default="test")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    scores_path = args.root / f"{args.split}_scores.jsonl"
    if not scores_path.exists():
        print(
            f"missing scores file {scores_path}; run score_synra_codex_intrinsic.py first",
            file=sys.stderr,
        )
        return 2
    op_path = args.root / "per_op_scores.jsonl"
    if not op_path.exists():
        print(
            f"missing per-op scores {op_path}; run per_op_synra_codex_intrinsic.py first",
            file=sys.stderr,
        )
        return 2
    rows = _read_scores(scores_path)
    if not rows:
        print(f"no rows in {scores_path}", file=sys.stderr)
        return 2
    op_rows = [r for r in _read_scores(op_path) if r["relabel"]]
    if not op_rows:
        print(f"no with-relabel rows in {op_path}", file=sys.stderr)
        return 2
    out_path = args.out or (args.root / "report.md")
    out_path.write_text(
        render_report(rows, op_rows, args.split, scores_path, op_path)
    )
    print(f"wrote {out_path} ({len(rows)} probe rows + {len(op_rows)} per-op rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
