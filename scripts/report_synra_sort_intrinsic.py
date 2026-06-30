"""Compose a markdown report from a synra_sort intrinsic scoring run.

Reads ``<root>/<split>_scores.jsonl`` (default ``--split test``) and writes
``<root>/report.md`` with:

  * Headline mean ± stdev for fixed / hungarian / pmr + the fixed−hungarian gap.
  * **M2** — discrimination AUROC (unperturbed vs perturbed) per scorer.
  * **M3** — Spearman & Kendall of score vs Kendall distance (equal-length rows).
  * **Partial-credit curve** — score vs Kendall distance for ``adj_transpose``:
    fixed decays smoothly, hungarian is flat (no order signal), PMR cliffs to 0.
  * **Gap-handling** — score vs #deletions/insertions: fixed degrades gracefully,
    PMR → 0 (length ≠ N → parse failure), hungarian drops only by the bag-mismatch
    fraction (NOT flat — flatness holds only for pure reorderings).

All statistics are computed in-script (no scipy / matplotlib).

Usage::

    uv run python scripts/report_synra_sort_intrinsic.py \\
        [--split test] [--root data/synra_sort/intrinsic/v1]
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


SCORERS: tuple[str, ...] = ("fixed", "hungarian", "pmr", "tau")

_SCORER_HEADER = " | ".join(SCORERS)
_SCORER_RULE = " | ".join("---" for _ in SCORERS)


# --- statistics (no scipy) ------------------------------------------------


def _ranks(values: list[float]) -> list[float]:
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
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  —  "
    return f"{x:.3f}"


# --- sections --------------------------------------------------------------


def _section_headline(rows: list[dict[str, Any]]) -> str:
    parts = [
        "## Headline numbers\n",
        "| scorer | mean | stdev | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in SCORERS:
        vals = [float(r[s]) for r in rows]
        m, sd = _mean_std(vals)
        parts.append(f"| {s} | {m:.3f} | {sd:.3f} | {min(vals):.3f} | {max(vals):.3f} |")
    mf, _ = _mean_std([float(r["fixed"]) for r in rows])
    mh, _ = _mean_std([float(r["hungarian"]) for r in rows])
    parts.append(f"| **fixed − hungarian** | **{mf - mh:+.3f}** | — | — | — |")
    return "\n".join(parts) + "\n"


def _section_m2(rows: list[dict[str, Any]]) -> str:
    labels = [1 if r["family"] == "none" else 0 for r in rows]
    parts = [
        "## M2 — discrimination AUROC (unperturbed vs perturbed)\n",
        "| scorer | AUROC |",
        "| --- | --- |",
    ]
    for s in SCORERS:
        parts.append(f"| {s} | {_fmt(auroc([float(r[s]) for r in rows], labels))} |")
    return "\n".join(parts) + "\n"


def _safe_corr(fn, vals: list[float], kd: list[float]) -> float:
    """Correlation, treating a constant score series as 0 (no relationship)
    rather than undefined — hungarian is exactly constant on equal-length rows."""
    if len(set(vals)) <= 1:
        return 0.0
    return fn(vals, kd)


def _section_m3(rows: list[dict[str, Any]]) -> str:
    eq = [r for r in rows if r["kendall_distance"] is not None]
    kd = [float(r["kendall_distance"]) for r in eq]
    parts = [
        "## M3 — correlation of score vs Kendall distance (equal-length rows)\n",
        "_Desirable sign is **negative**. Hungarian is exactly constant on "
        "equal-length permutations (no order signal) → correlation 0; fixed "
        "tracks Kendall distance tightly._\n",
        f"_Equal-length rows: {len(eq)}._\n",
        "| scorer | Spearman | Kendall τ-b |",
        "| --- | --- | --- |",
    ]
    for s in ("fixed", "hungarian", "tau"):
        vals = [float(r[s]) for r in eq]
        sp = _safe_corr(spearman, vals, kd)
        kt = _safe_corr(kendall_tau_b, vals, kd)
        parts.append(f"| {s} | {sp:+.3f} | {kt:+.3f} |")
    return "\n".join(parts) + "\n"


def _section_partial_credit(rows: list[dict[str, Any]]) -> str:
    src = [r for r in rows if r["family"] in ("none", "adj_transpose")]
    by_d: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in src:
        if r["kendall_distance"] is not None:
            by_d[int(r["kendall_distance"])].append(r)
    parts = [
        "## Partial-credit curve (adj_transpose, by Kendall distance)\n",
        "_fixed decays smoothly; hungarian is flat (no order signal); PMR is "
        "all-or-nothing (1.0 only at distance 0); tau is graded here but "
        "collapses to 0 on any length change._\n",
        f"| Kendall distance | n | {_SCORER_HEADER} |",
        f"| --- | --- | {_SCORER_RULE} |",
    ]
    for d in sorted(by_d):
        rs = by_d[d]
        means = [statistics.fmean(float(r[s]) for r in rs) for s in SCORERS]
        parts.append(f"| {d} | {len(rs)} | " + " | ".join(f"{m:.3f}" for m in means) + " |")
    return "\n".join(parts) + "\n"


def _section_gap_handling(rows: list[dict[str, Any]]) -> str:
    parts = [
        "## Gap-handling (deletion / insertion, by magnitude k)\n",
        "_fixed degrades gracefully via DP gaps; PMR → 0 (length ≠ N → parse "
        "failure); tau → 0 too (undefined across lengths); hungarian drops "
        "only by the bag-mismatch fraction (NOT flat here — flatness holds "
        "only for pure reorderings)._\n",
    ]
    for fam in ("deletion", "insertion"):
        rows_f = [r for r in rows if r["family"] == fam]
        none_rows = [r for r in rows if r["family"] == "none"]
        by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in none_rows:
            by_k[0].append(r)
        for r in rows_f:
            by_k[int(r["k"])].append(r)
        parts.append(f"### {fam}")
        parts.append("")
        parts.append(f"| k | n | {_SCORER_HEADER} |")
        parts.append(f"| --- | --- | {_SCORER_RULE} |")
        for k in sorted(by_k):
            rs = by_k[k]
            means = [statistics.fmean(float(r[s]) for r in rs) for s in SCORERS]
            parts.append(f"| {k} | {len(rs)} | " + " | ".join(f"{m:.3f}" for m in means) + " |")
        parts.append("")
    return "\n".join(parts)


def _section_by_family(rows: list[dict[str, Any]]) -> str:
    order = ["none", "adj_transpose", "block_reverse", "block_move", "deletion", "insertion"]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r["family"]].append(r)
    lines = [
        "## By perturbation family",
        "",
        f"| family | n | {_SCORER_HEADER} |",
        f"| --- | --- | {_SCORER_RULE} |",
    ]
    for fam in [f for f in order if f in buckets]:
        rs = buckets[fam]
        means = [statistics.fmean(float(r[s]) for r in rs) for s in SCORERS]
        lines.append(f"| {fam} | {len(rs)} | " + " | ".join(f"{m:.3f}" for m in means) + " |")
    lines.append("")
    return "\n".join(lines)


def render_report(rows: list[dict[str, Any]], split: str, source: Path) -> str:
    parts = [
        f"# synra_sort intrinsic — `{split}` split report",
        "",
        f"_Source_: `{source}` &nbsp;&nbsp; _Rows_: {len(rows)} &nbsp;&nbsp; "
        f"_Scorers_: fixed, hungarian, pmr, tau",
        "",
        _section_headline(rows),
        "",
        _section_m2(rows),
        "",
        _section_m3(rows),
        "",
        _section_partial_credit(rows),
        "",
        _section_gap_handling(rows),
        "",
        _section_by_family(rows),
    ]
    return "\n".join(parts)


def _parse_args() -> argparse.Namespace:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=cfg.synra_sort_intrinsic_v2)
    p.add_argument("--split", choices=("train", "validation", "test"), default="test")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    scores_path = args.root / f"{args.split}_scores.jsonl"
    if not scores_path.exists():
        print(f"missing scores file {scores_path}; run score_synra_sort_intrinsic.py first",
              file=sys.stderr)
        return 2
    rows = _read_scores(scores_path)
    if not rows:
        print(f"no rows in {scores_path}", file=sys.stderr)
        return 2
    out_path = args.out or (args.root / "report.md")
    out_path.write_text(render_report(rows, args.split, scores_path))
    print(f"wrote {out_path} ({len(rows)} rows summarised)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
