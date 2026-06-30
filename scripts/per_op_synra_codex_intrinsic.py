"""Per-operation degradation sweep for the synra_codex intrinsic study.

The structural-sensitivity probe of the study (the id-equivariance probe lives
in ``prepare_synra_codex_intrinsic.py``): for every gold graph we form
candidates = id-relabel + the SAME op applied k times (k = 0..K), and score them
under RA and plain. The result is one degradation curve per operation, for each
config, showing how each op separately reduces the score. The report script
(``report_synra_codex_intrinsic.py``) computes its discrimination/sensitivity
diagnostics from ``per_op_scores.jsonl``, so run this before it.

A single op is applied with **no cross-op fallback**: when an application
becomes infeasible (e.g. record deletion once a scope is down to its last two
records) the op is exhausted and the curve plateaus, so each curve reflects
only its own operation.

Ops (PERTURBATION_CYCLE): categorical relabel, reference rerouting, record
deletion, edge deletion, record insertion, edge insertion.

Outputs (under ``--root``, default ``data/synra_codex/intrinsic/v1``):
    per_op_scores.jsonl                    # row per (gold, op, k, relabel)
    imgs/fig6_per_operation.png            # WITH id-relabel (k=0 = relabel only)
    imgs/fig7_per_operation_no_relabel.png # NO relabel    (k=0 = unperturbed gold)

Golds draw their parameters via ``sample_probe_params`` (imported from
``prepare_synra_codex_intrinsic``), i.e. the same randomized distribution as
the id-equivariance probe — no difficulty grid.

Usage::

    uv run python scripts/per_op_synra_codex_intrinsic.py \\
        [--n-golds 100] [--max-apply 8] \\
        [--seed 20260609] [--root data/synra_codex/intrinsic/v2]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from object_aligner_exp.config import ExpConfig  # noqa: E402
from object_aligner_exp.schemas import load_synra_codex_schema  # noqa: E402
from prepare_synra_codex import build_codebooks, sample_gold  # noqa: E402
from prepare_synra_codex_intrinsic import (  # noqa: E402
    OBFUSCATE,
    PERTURBATION_CYCLE,
    VOCAB,
    _apply_one_perturbation,
    make_property_twins,
    relabel_ids,
    sample_probe_params,
)

from object_aligner import ObjectAligner  # noqa: E402


# Readable op names (match the six-op list in the prose).
OP_LABELS: dict[str, str] = {
    "cat_relabel": "categorical relabel",
    "ref_misroute": "reference rerouting",
    "node_delete": "record deletion",
    "edge_delete": "edge deletion",
    "node_add": "record insertion",
    "edge_add": "edge insertion",
}
DPI = 150


def apply_op_k(
    gold: dict[str, Any], op: str, k: int, rng, cb, relabel: bool = True
) -> dict[str, Any]:
    """Apply ``op`` exactly ``k`` times (no cross-op fallback).

    With ``relabel=True`` the ids are first rewritten (severity-0 transform), so
    ``k=0`` is the relabel-only baseline; with ``relabel=False`` the gold ids are
    kept, so ``k=0`` is the unperturbed gold and the curves isolate each op's
    effect free of the relabeling confound.

    If an application is infeasible the op falls back internally; we detect that
    (returned op name differs), revert the step, and stop — the op is exhausted,
    so the candidate reflects the maximum effect of that op alone.
    """
    g = relabel_ids(gold, rng) if relabel else copy.deepcopy(gold)
    for _ in range(k):
        before = copy.deepcopy(g)
        applied = _apply_one_perturbation(g, op, rng, cb)
        if applied != op:
            g.clear()
            g.update(before)
            break
    return g


def _score(aligner: ObjectAligner, gold: dict[str, Any], pred: dict[str, Any]) -> float:
    return float(aligner.metric(gold, pred)["score"])


def main() -> int:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=cfg.synra_codex_intrinsic_v2)
    p.add_argument("--n-golds", type=int, default=100,
                   help="Number of gold graphs; each draws its parameters via "
                        "sample_probe_params (same distribution as the "
                        "id-equivariance probe).")
    p.add_argument("--max-apply", type=int, default=8,
                   help="Max number of times a single op is applied (x-axis top).")
    p.add_argument("--no-property-twins", action="store_true",
                   help="Disable property-twin construction (default: on, as in the study).")
    p.add_argument("--seed", type=int, default=20260609)
    p.add_argument(
        "--paper-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "research/paper_intrinsic/figures",
        help="where the paper-styled variants (no suptitle) go; "
             "pass an empty string to skip them",
    )
    args = p.parse_args()

    cb = build_codebooks(vocab=VOCAB, obfuscate=OBFUSCATE)
    ra = ObjectAligner(load_synra_codex_schema(cfg, kind="ra"))
    strict = ObjectAligner(load_synra_codex_schema(cfg, kind="strict"))
    master = random.Random(args.seed)
    ks = list(range(args.max_apply + 1))

    rows: list[dict[str, Any]] = []
    for gold_idx in range(args.n_golds):
        n_people, n_companies, twin, acq, part = sample_probe_params(master)
        gseed = master.randrange(1 << 31)
        gold = sample_gold(n_people, n_companies, twin, acq, part,
                           random.Random(gseed), cb, False)
        if not args.no_property_twins:
            make_property_twins(gold, twin, random.Random(gseed ^ 0x7))
        for relabel in (True, False):
            for op in PERTURBATION_CYCLE:
                for k in ks:
                    prng = random.Random(
                        (gseed * 6 + PERTURBATION_CYCLE.index(op))
                        ^ (k * 2654435761) ^ (0x5A if relabel else 0xA5)
                    )
                    pred = apply_op_k(gold, op, k, prng, cb, relabel=relabel)
                    rows.append({
                        "gold_idx": gold_idx, "gold_seed": gseed,
                        "n_people": n_people, "n_companies": n_companies,
                        "twin_density": twin, "op": op, "k": k,
                        "relabel": relabel,
                        "score_ra": _score(ra, gold, pred),
                        "score_strict": _score(strict, gold, pred),
                    })

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    out_jsonl = root / "per_op_scores.jsonl"
    with out_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_jsonl} ({len(rows)} rows; {args.n_golds} golds, "
          f"{len(PERTURBATION_CYCLE)} ops, k=0..{args.max_apply})")

    # --- aggregate mean by (relabel, config, op, k), stratified into graph-
    #     size bins (sizes are sampled, so exact-size strata no longer exist).
    SIZE_BINS: tuple[tuple[str, int, int], ...] = (
        ("3–4 people", 3, 4),
        ("5–7 people", 5, 7),
        ("8–10 people", 8, 10),
    )

    def size_bin(r: dict[str, Any]) -> str:
        for label, lo, hi in SIZE_BINS:
            if lo <= r["n_people"] <= hi:
                return label
        return "other"

    sizes = [label for label, _, _ in SIZE_BINS]
    size_n = {
        label: len({r["gold_seed"] for r in rows if size_bin(r) == label})
        for label in sizes
    }

    def curve(relabel: bool, key: str, op: str, size, stat: str) -> list[float]:
        out = []
        for k in ks:
            v = [r[key] for r in rows
                 if r["relabel"] == relabel and r["op"] == op and r["k"] == k
                 and size_bin(r) == size]
            if not v:
                out.append(float("nan"))
            elif stat == "mean":
                out.append(sum(v) / len(v))
            else:
                out.append(statistics.pstdev(v) if len(v) > 1 else 0.0)
        return out

    cmap = plt.get_cmap("tab10")
    op_color = {op: cmap(i) for i, op in enumerate(PERTURBATION_CYCLE)}
    imgs = root / "imgs"
    imgs.mkdir(parents=True, exist_ok=True)

    def make_fig(relabel: bool, out_png: Path, suptitle: str | None, k0_note: str) -> None:
        nr = len(sizes)
        fig, axes = plt.subplots(nr, 2, figsize=(13, 4.0 * nr), dpi=DPI,
                                 sharex=True, sharey=True, squeeze=False)
        for i, size in enumerate(sizes):
            for j, (cfg_name, key) in enumerate([("RA", "score_ra"), ("plain", "score_strict")]):
                ax = axes[i][j]
                for op in PERTURBATION_CYCLE:
                    m = curve(relabel, key, op, size, "mean")
                    ax.plot(ks, m, marker="o", markersize=4, linewidth=1.8,
                            color=op_color[op], label=OP_LABELS[op])
                ax.set_ylim(0.0, 1.05)
                ax.set_xticks(ks)
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.set_title(cfg_name)
                if i == nr - 1:
                    ax.set_xlabel(f"number of applications $k$ ({k0_note})")
            axes[i][0].set_ylabel(f"{size}\n(n={size_n[size]} golds)\nmean OA score")
        axes[0][1].legend(loc="upper right", frameon=False, fontsize=8)
        if suptitle:
            fig.suptitle(suptitle, y=1.005, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_png}")

    def pooled_curve(relabel: bool, key: str, op: str) -> list[float]:
        out = []
        for k in ks:
            v = [r[key] for r in rows
                 if r["relabel"] == relabel and r["op"] == op and r["k"] == k]
            out.append(sum(v) / len(v) if v else float("nan"))
        return out

    def make_pooled_fig(relabel: bool, out_png: Path, k0_note: str) -> None:
        """Compact 1×2 paper variant: means pooled over all sizes/seeds."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=DPI,
                                 sharex=True, sharey=True)
        for ax, (cfg_name, key) in zip(
            axes, [("RA", "score_ra"), ("plain", "score_strict")]
        ):
            for op in PERTURBATION_CYCLE:
                ax.plot(ks, pooled_curve(relabel, key, op), marker="o",
                        markersize=4, linewidth=1.8, color=op_color[op],
                        label=OP_LABELS[op])
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(ks)
            ax.grid(True, alpha=0.3)
            ax.set_title(cfg_name)
            ax.set_xlabel(f"number of applications $k$ ({k0_note})")
        axes[0].set_ylabel("mean OA score")
        axes[1].legend(loc="lower left", frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_png}")

    make_fig(True, imgs / "fig6_per_operation.png",
             "Per-operation degradation by graph size — each op alone, on top of an id-relabel",
             "k=0 is relabel only")
    make_fig(False, imgs / "fig7_per_operation_no_relabel.png",
             "Per-operation degradation by graph size — each op alone, no id-relabel",
             "k=0 is the unperturbed gold")

    if args.paper_dir and str(args.paper_dir):
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        make_pooled_fig(True, args.paper_dir / "intrinsic_ra_per_operation_pooled.png",
                        "k=0 is relabel only")
        make_fig(True, args.paper_dir / "intrinsic_ra_per_operation.png",
                 None, "k=0 is relabel only")
        make_fig(False, args.paper_dir / "intrinsic_ra_per_operation_norelabel.png",
                 None, "k=0 is the unperturbed gold")

    # stdout summary: per size bin, score at k=1 and k=max per op (with-relabel)
    for size in sizes:
        print(f"\n=== {size} (n={size_n[size]} golds), WITH relabel ===")
        print(f"{'operation':22} {'RA k=1':>8} {'RA kmax':>8} {'plain k=1':>10} {'plain kmax':>11}")
        for op in PERTURBATION_CYCLE:
            ra = curve(True, "score_ra", op, size, "mean")
            pl = curve(True, "score_strict", op, size, "mean")
            print(f"{OP_LABELS[op]:22} {ra[1]:8.3f} {ra[-1]:8.3f} {pl[1]:10.3f} {pl[-1]:11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
