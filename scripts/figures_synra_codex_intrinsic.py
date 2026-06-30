"""Render figures for the synra_codex intrinsic RA study.

Reads ``data/synra_codex/intrinsic/v1/test_scores.jsonl`` (default — the
id-equivariance probe: K independent id relabelings per gold) and writes:

    imgs/fig_id_equivariance.png — per-gold relabel variance (ra vs plain)

Structural-sensitivity figures (per-operation degradation) come from
``scripts/per_op_synra_codex_intrinsic.py``, which also writes the
paper-styled variants. The former severity-ladder figures were removed
2026-06-11 along with the ladder itself.

Pure stdlib + matplotlib. Colours: ra=tab:blue, strict (plain)=tab:orange.

Usage::

    uv run python scripts/figures_synra_codex_intrinsic.py [--root data/synra_codex/intrinsic/v1]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from object_aligner_exp.config import ExpConfig  # noqa: E402


CONFIGS: tuple[str, ...] = ("ra", "strict")
LABELS: dict[str, str] = {"ra": "RA", "strict": "plain"}
COLORS: dict[str, str] = {"ra": "tab:blue", "strict": "tab:orange"}
DPI = 150


def _read_scores(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --- id-equivariance variance ----------------------------------------------


def fig_id_equivariance(rows: list[dict[str, Any]], out: Path) -> None:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["gold_seed"], r["n_people"], r["n_companies"], r["twin_density"])].append(r)
    var_by_cfg: dict[str, list[float]] = {c: [] for c in CONFIGS}
    for grp in groups.values():
        if len(grp) < 2:
            continue
        for c in CONFIGS:
            var_by_cfg[c].append(statistics.pvariance([g[f"score_{c}"] for g in grp]))

    floor = 1e-9
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=DPI)
    ax.axhline(floor, color="0.7", linewidth=0.8, linestyle="--",
               label="$10^{-9}$ display floor")
    for i, c in enumerate(CONFIGS):
        y = [max(v, floor) for v in var_by_cfg[c]]
        x = rng.normal(float(i), 0.05, size=len(y))
        med = statistics.median(var_by_cfg[c]) if var_by_cfg[c] else floor
        ax.scatter(x, y, color=COLORS[c], alpha=0.7, s=42,
                   label=f"{LABELS[c]} — median {med:.2e}")
        ax.plot([i - 0.25, i + 0.25], [max(med, floor)] * 2, color=COLORS[c], linewidth=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([LABELS[c] for c in CONFIGS])
    ax.set_xlim(-0.6, len(CONFIGS) - 0.4)
    ax.set_ylabel("per-gold score variance across id relabelings")
    ax.set_title("Id-equivariance: score variance under id relabeling")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --- CLI ------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=cfg.synra_codex_intrinsic_v2)
    p.add_argument("--scores", type=Path, default=None)
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

    out = imgs / "fig_id_equivariance.png"
    fig_id_equivariance(rows, out)
    print(f"  wrote {out}  ({out.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
