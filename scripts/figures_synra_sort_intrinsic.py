"""Render figures for the synra_sort intrinsic fixed-order study.

Reads ``data/synra_sort/intrinsic/v1/test_scores.jsonl`` (default) and writes:

    imgs/fig1_partial_credit_curve.png — HERO: score vs Kendall distance
    imgs/fig2_gap_handling.png         — score vs #deletions / #insertions
    imgs/fig3_m2_roc.png               — unperturbed vs perturbed ROC
    imgs/fig4_m3_correlation.png       — score vs Kendall distance scatter
    imgs/fig5_family_breakdown.png     — mean score per perturbation family

Pure stdlib + matplotlib. Colours: fixed=tab:blue, hungarian=tab:orange,
pmr=tab:green, tau=tab:red.

Also writes paper-styled variants of fig1/fig2 (paper-facing labels
sequence/set/exact/Kendall τ, no in-figure titles) to ``--paper-dir``
(default ``research/paper_intrinsic/figures``).

Usage::

    uv run python scripts/figures_synra_sort_intrinsic.py [--root data/synra_sort/intrinsic/v1]
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from object_aligner_exp.config import ExpConfig  # noqa: E402


SCORERS: tuple[str, ...] = ("fixed", "hungarian", "pmr", "tau")
COLORS = {"fixed": "tab:blue", "hungarian": "tab:orange", "pmr": "tab:green",
          "tau": "tab:red"}
MARKERS = {"fixed": "o", "hungarian": "s", "pmr": "^", "tau": "D"}
# Paper-facing names (match the draft prose in research/paper_intrinsic).
PAPER_LABELS = {"fixed": "sequence", "hungarian": "set", "pmr": "exact",
                "tau": "Kendall τ"}
# Line styles for the paper variants — coincident curves (set == sequence on
# the gap families; τ == exact there) stay visible when dashed.
PAPER_STYLES = {"fixed": "-", "hungarian": "--", "pmr": "-", "tau": ":"}
DPI = 150


def _read_scores(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _ranks(values: list[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    vx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    vy = math.sqrt(sum((y - my) ** 2 for y in ry))
    return cov / (vx * vy) if vx and vy else float("nan")


def _roc(scores: list[float], labels: list[int]):
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    n_pos = sum(labels)
    n_neg = n - n_pos
    tps = fps = 0
    tpr = [0.0]
    fpr = [0.0]
    prev = None
    for idx in order:
        s = scores[idx]
        if prev is not None and s != prev:
            tpr.append(tps / n_pos if n_pos else 0.0)
            fpr.append(fps / n_neg if n_neg else 0.0)
        if labels[idx] == 1:
            tps += 1
        else:
            fps += 1
        prev = s
    tpr.append(tps / n_pos if n_pos else 0.0)
    fpr.append(fps / n_neg if n_neg else 0.0)
    ranks = _ranks(scores)
    rank_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    if n_pos == 0 or n_neg == 0:
        return fpr, tpr, float("nan")
    u1 = rank_pos - n_pos * (n_pos + 1) / 2
    return fpr, tpr, u1 / (n_pos * n_neg)


# --- figure 1 — partial-credit curve (HERO) -------------------------------


def fig1_partial_credit(rows: list[dict[str, Any]], out: Path, paper: bool = False) -> None:
    src = [
        r for r in rows
        if r["family"] in ("none", "adj_transpose") and r["kendall_distance"] is not None
    ]
    by_d: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in src:
        by_d[int(r["kendall_distance"])].append(r)
    ds = sorted(by_d)
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=DPI)
    for s in SCORERS:
        ys = [statistics.fmean(float(r[s]) for r in by_d[d]) for d in ds]
        ax.plot(ds, ys, marker=MARKERS[s], linewidth=2, color=COLORS[s],
                label=PAPER_LABELS[s] if paper else s,
                linestyle=PAPER_STYLES[s] if paper else "-")
    ax.set_xlabel("Kendall distance (inversions from gold order)")
    ax.set_ylabel("mean score")
    ax.set_ylim(-0.02, 1.05)
    if not paper:
        ax.set_title("Partial-credit curve — fixed decays smoothly, hungarian is flat, PMR cliffs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --- figure 2 — gap handling ----------------------------------------------


def fig2_gap_handling(rows: list[dict[str, Any]], out: Path, paper: bool = False) -> None:
    none_mean = {s: statistics.fmean(float(r[s]) for r in rows if r["family"] == "none") for s in SCORERS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=DPI)
    for ax, fam in zip(axes, ("deletion", "insertion")):
        by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r["family"] == fam:
                by_k[int(r["k"])].append(r)
        ks = [0] + sorted(by_k)
        for s in SCORERS:
            ys = [none_mean[s]] + [statistics.fmean(float(r[s]) for r in by_k[k]) for k in sorted(by_k)]
            ax.plot(ks, ys, marker=MARKERS[s], linewidth=2, color=COLORS[s],
                    label=PAPER_LABELS[s] if paper else s,
                    linestyle=PAPER_STYLES[s] if paper else "-")
        ax.set_xlabel(f"# {fam}s (k); k=0 is unperturbed")
        ax.set_ylabel("mean score")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(fam if paper else f"Gap handling — {fam}")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=False)
    if not paper:
        fig.suptitle("Length-change robustness: fixed degrades gracefully; PMR → 0; hungarian partially responsive",
                     y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --- figure 3 — M2 ROC ----------------------------------------------------


def fig3_m2_roc(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [1 if r["family"] == "none" else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
    ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", label="chance (AUC = 0.500)")
    for s in SCORERS:
        fpr, tpr, auc = _roc([float(r[s]) for r in rows], labels)
        ax.plot(fpr, tpr, color=COLORS[s], linewidth=2, label=f"{s} (AUC = {auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("M2 — ROC for \"is this row unperturbed?\"")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --- figure 4 — M3 correlation --------------------------------------------


def fig4_m3_correlation(rows: list[dict[str, Any]], out: Path) -> None:
    eq = [r for r in rows if r["kendall_distance"] is not None]
    kd = [float(r["kendall_distance"]) for r in eq]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=DPI)
    for s in ("fixed", "hungarian"):
        vals = [float(r[s]) for r in eq]
        rho = 0.0 if len(set(vals)) <= 1 else _spearman(vals, kd)
        ax.scatter([k + (0.08 if s == "hungarian" else -0.08) for k in kd], vals,
                   alpha=0.4, s=22, color=COLORS[s], label=f"{s}  ρ={rho:+.3f}")
    ax.set_xlabel("Kendall distance (equal-length perturbations)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("M3 — score vs Kendall distance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --- figure 5 — family breakdown ------------------------------------------


def fig5_family_breakdown(rows: list[dict[str, Any]], out: Path) -> None:
    import numpy as np

    order = ["none", "adj_transpose", "block_reverse", "block_move", "deletion", "insertion"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["family"]].append(r)
    order = [f for f in order if f in groups]
    x = np.arange(len(order))
    width = 0.8 / len(SCORERS)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    for i, s in enumerate(SCORERS):
        means = [statistics.fmean(float(r[s]) for r in groups[f]) for f in order]
        ax.bar(x + (i - (len(SCORERS) - 1) / 2) * width, means, width,
               color=COLORS[s], label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15)
    ax.set_xlabel("perturbation family")
    ax.set_ylabel("mean score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Score by perturbation family")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=cfg.synra_sort_intrinsic_v2)
    p.add_argument("--scores", type=Path, default=None)
    p.add_argument(
        "--paper-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "research/paper_intrinsic/figures",
        help="where the paper-styled variants (paper labels, no titles) go; "
             "pass an empty string to skip them",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    scores_path = args.scores or (args.root / "test_scores.jsonl")
    if not scores_path.exists():
        print(f"missing scores file {scores_path}", file=sys.stderr)
        return 2
    rows = _read_scores(scores_path)
    print(f"loaded {len(rows)} rows from {scores_path}")

    imgs = args.root / "imgs"
    imgs.mkdir(parents=True, exist_ok=True)

    fig1_partial_credit(rows, imgs / "fig1_partial_credit_curve.png")
    fig2_gap_handling(rows, imgs / "fig2_gap_handling.png")
    fig3_m2_roc(rows, imgs / "fig3_m2_roc.png")
    fig4_m3_correlation(rows, imgs / "fig4_m3_correlation.png")
    fig5_family_breakdown(rows, imgs / "fig5_family_breakdown.png")

    written = [
        imgs / "fig1_partial_credit_curve.png",
        imgs / "fig2_gap_handling.png",
        imgs / "fig3_m2_roc.png",
        imgs / "fig4_m3_correlation.png",
        imgs / "fig5_family_breakdown.png",
    ]

    if args.paper_dir and str(args.paper_dir):
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        fig1_partial_credit(rows, args.paper_dir / "intrinsic_order_partialcredit.png",
                            paper=True)
        fig2_gap_handling(rows, args.paper_dir / "intrinsic_order_gaphandling.png",
                          paper=True)
        written += [
            args.paper_dir / "intrinsic_order_partialcredit.png",
            args.paper_dir / "intrinsic_order_gaphandling.png",
        ]

    for path in written:
        print(f"  wrote {path}  ({path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
