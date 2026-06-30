"""Generate the synra_codex INTRINSIC referential-alignment benchmark.

A no-LLM perturbation study built on top of the ``synra_codex`` gold-graph
generator (``scripts/prepare_synra_codex.py``). Unlike the GEPA dataset, this
benchmark carries pre-materialised ``(gold, pred)`` pairs and exists purely to
probe Object Aligner's **referential alignment (RA)** — id-equivariance and
Weisfeiler–Leman (WL) twin-disambiguation — features STED has no analogue for.

Gold graphs are generated *simple* (``vocab="small"``, ``obfuscate=False`` →
readable words) so the study isolates RA from codebook discovery. Every gold
draws its parameters at random via ``sample_probe_params`` (sizes, twin
density, edge densities) — there is no difficulty grid; predictions are the
**id-equivariance probe**:
``--severity-zero-preds`` independent id relabelings per gold (no structural
edits — a relabel-invariant score must give every relabeling the same value,
so the per-gold variance is the diagnostic). The scorer
(``scripts/score_synra_codex_intrinsic.py``) scores each pair under two
configs: ``ra`` (referential alignment, OA defaults) and ``strict``/``plain``
(identifiers compared by value).

Structural sensitivity is probed separately by the per-operation sweep
(``scripts/per_op_synra_codex_intrinsic.py``), which imports
``sample_probe_params`` and the perturbation primitives defined here
(``PERTURBATION_CYCLE``, ``_apply_one_perturbation``, ``relabel_ids``,
``make_property_twins``), so both probes draw golds from the same
distribution.

Nothing is fitted in this study, so there are no train/validation splits: all
rows land in a single ``test/`` set (the name keeps the scorer/report
plumbing unchanged).

Outputs (``--root`` defaults to ``data/synra_codex/intrinsic/v2``):

  <root>/
    config.jsonc                # full generator config + master seed
    summary.md                  # marginals over the sampled parameters
    test/
      labels.jsonl              # one row = one (gold, pred) pair
      meta.jsonl                # same rows minus the gold/pred payloads

Usage::

    uv run python scripts/prepare_synra_codex_intrinsic.py \\
        [--n-golds 100] [--severity-zero-preds 5] \\
        [--seed 20260609] [--force] [--root data/synra_codex/intrinsic/v2]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Project imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Sibling-script import (synra_codex gold-graph generator).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from object_aligner_exp.config import ExpConfig  # noqa: E402
from object_aligner_exp.schemas import load_synra_codex_schema  # noqa: E402
from prepare_synra_codex import (  # noqa: E402
    Codebooks,
    _fresh_hex,
    build_codebooks,
    sample_gold,
)

from object_aligner import ObjectAligner  # noqa: E402


# --- generator configuration ----------------------------------------------

# Per-gold parameter distributions (no difficulty grid — every gold draws its
# own parameters; see sample_probe_params). Edge-density ranges are centred on
# the former fixed values (0.35 / 0.5), generous enough that perturbations
# (especially ref_misroute / edge_delete) have edges to act on.
N_PEOPLE_RANGE: tuple[int, int] = (3, 10)
ACQUAINTANCE_DENSITY_RANGE: tuple[float, float] = (0.2, 0.5)
PARTNERSHIP_DENSITY_RANGE: tuple[float, float] = (0.3, 0.7)

# Simple, readable gold (RA isolated from codebook discovery).
VOCAB: str = "small"
OBFUSCATE: bool = False


def sample_probe_params(rng: random.Random) -> tuple[int, int, float, float, float]:
    """Draw one gold's generator parameters.

    Returns ``(n_people, n_companies, twin_density, acquaintance_density,
    partnership_density)``: ``n_people ~ U{3..10}``, ``n_companies ~
    U{2..ceil(n_people/2)}`` (people stay the larger scope), ``twin_density ~
    U[0,1]``, and the edge densities uniform over their ranges. Shared by the
    id-equivariance probe and the per-op sweep so both draw golds from the
    same distribution.
    """
    n_people = rng.randint(*N_PEOPLE_RANGE)
    n_companies = rng.randint(2, max(2, (n_people + 1) // 2))
    twin = rng.random()
    acq = rng.uniform(*ACQUAINTANCE_DENSITY_RANGE)
    part = rng.uniform(*PARTNERSHIP_DENSITY_RANGE)
    return n_people, n_companies, twin, acq, part

# The six structural edit operations, consumed by the per-op sweep
# (scripts/per_op_synra_codex_intrinsic.py). ``ref_misroute`` is the
# RA-critical op: it rewires one edge endpoint to a different in-scope node,
# breaking routing without touching any node — undetectable to a literal-id
# (strict) metric.
PERTURBATION_CYCLE: tuple[str, ...] = (
    "cat_relabel",
    "ref_misroute",
    "node_delete",
    "edge_delete",
    "node_add",
    "edge_add",
)

# Edge arrays and the id-scope each endpoint field references.
_PERSON_FIELDS = {
    "employment": ("person",),
    "acquaintance": ("source", "target"),
}
_COMPANY_FIELDS = {
    "employment": ("company",),
    "partnership": ("source", "target"),
}


# --- property-twin construction --------------------------------------------


def make_property_twins(
    gold: dict[str, Any], twin_density: float, rng: random.Random
) -> dict[str, Any]:
    """Turn a ``twin_density`` fraction of each scope into genuine property-twins.

    synra_codex draws a unique ``name`` per node, and the RA schema scores
    ``name`` (jaro_winkler). So even at twin_density=1.0 the categorical code is
    shared but the name still distinguishes every node — the records never become
    property-identical, the hardest case for referential alignment.

    To stress RA at that hardest case, we pick exactly ``round(twin_density * n)``
    nodes per scope and collapse their ``name`` **and** categorical code to a
    single shared value, making them identical in every scored non-ref property.
    The remaining nodes keep their unique names, so they are *not* twins even if
    a categorical code happens to collide (the small vocab makes that common) —
    this keeps the twin-density gradient clean (0 twins at density 0, all twins
    at density 1). The twin group's members still differ in their ref-graph
    structure (roles / edges), which referential alignment uses to keep the
    score relabel-invariant even here.
    """
    for scope, code_field in (("people", "title"), ("companies", "industry")):
        nodes = gold[scope]
        k = round(twin_density * len(nodes))
        if k >= 2:
            group = rng.sample(nodes, k)
            rep = group[0]
            for nd in group:
                nd["name"] = rep["name"]
                nd[code_field] = rep[code_field]
    return gold


# --- perturbation primitives ----------------------------------------------


def relabel_ids(graph: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Severity-0 perturbation: rewrite every person/company id with a fresh
    hex string in a private namespace (no overlap with gold ids possible).
    Categorical fields and edge topology are untouched.
    """
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for p in graph["people"]:
        mapping[p["id"]] = _fresh_hex(rng, used)
    for c in graph["companies"]:
        mapping[c["id"]] = _fresh_hex(rng, used)
    return {
        "people": [{**p, "id": mapping[p["id"]]} for p in graph["people"]],
        "companies": [{**c, "id": mapping[c["id"]]} for c in graph["companies"]],
        "employment": [
            {**e, "person": mapping[e["person"]], "company": mapping[e["company"]]}
            for e in graph["employment"]
        ],
        "acquaintance": [
            {**e, "source": mapping[e["source"]], "target": mapping[e["target"]]}
            for e in graph["acquaintance"]
        ],
        "partnership": [
            {**e, "source": mapping[e["source"]], "target": mapping[e["target"]]}
            for e in graph["partnership"]
        ],
    }


def _cat_candidates(
    g: dict[str, Any], cb: Codebooks
) -> list[tuple[dict[str, Any], str, tuple[str, ...]]]:
    """(item, field, code-pool) for every categorical leaf present."""
    out: list[tuple[dict[str, Any], str, tuple[str, ...]]] = []
    out += [(p, "title", cb.title.codes) for p in g["people"]]
    out += [(c, "industry", cb.industry.codes) for c in g["companies"]]
    out += [(e, "role", cb.role.codes) for e in g["employment"]]
    out += [(e, "relation", cb.acquaintance.codes) for e in g["acquaintance"]]
    out += [(e, "relation", cb.partnership.codes) for e in g["partnership"]]
    return out


def _ref_candidates(
    g: dict[str, Any],
) -> list[tuple[dict[str, Any], str, list[str]]]:
    """(edge, endpoint-field, candidate-id-pool) for every routable endpoint
    that has ≥ 2 in-scope nodes to choose from."""
    person_ids = [p["id"] for p in g["people"]]
    company_ids = [c["id"] for c in g["companies"]]
    out: list[tuple[dict[str, Any], str, list[str]]] = []
    for arr, fields in _PERSON_FIELDS.items():
        for e in g[arr]:
            for f in fields:
                out.append((e, f, person_ids))
    for arr, fields in _COMPANY_FIELDS.items():
        for e in g[arr]:
            for f in fields:
                out.append((e, f, company_ids))
    return [(e, f, ids) for (e, f, ids) in out if len(ids) >= 2]


def _apply_one_perturbation(
    g: dict[str, Any], op: str, rng: random.Random, cb: Codebooks
) -> str:
    """Apply one perturbation in place; return the op actually applied (may
    fall back to a feasible op — ultimately ``node_add``, always feasible).
    """
    if op == "cat_relabel":
        cands = _cat_candidates(g, cb)
        if cands:
            item, field, codes = rng.choice(cands)
            old = item[field]
            choices = [c for c in codes if c != old] or list(codes)
            item[field] = rng.choice(choices)
            return "cat_relabel"
        return _apply_one_perturbation(g, "node_add", rng, cb)

    if op == "ref_misroute":
        cands = _ref_candidates(g)
        if cands:
            e, field, ids = rng.choice(cands)
            cur = e[field]
            # Avoid creating a self-loop on symmetric (source/target) edges.
            other = None
            if field == "source":
                other = e.get("target")
            elif field == "target":
                other = e.get("source")
            pool = [i for i in ids if i != cur and i != other]
            if pool:
                e[field] = rng.choice(pool)
                return "ref_misroute"
        return _apply_one_perturbation(g, "cat_relabel", rng, cb)

    if op == "node_delete":
        scopes = []
        if len(g["people"]) >= 2:
            scopes.append("people")
        if len(g["companies"]) >= 2:
            scopes.append("companies")
        if scopes:
            scope = rng.choice(scopes)
            if scope == "people":
                victim = rng.choice(g["people"])["id"]
                g["people"] = [p for p in g["people"] if p["id"] != victim]
                g["employment"] = [
                    e for e in g["employment"] if e["person"] != victim
                ]
                g["acquaintance"] = [
                    e
                    for e in g["acquaintance"]
                    if victim not in (e["source"], e["target"])
                ]
            else:
                victim = rng.choice(g["companies"])["id"]
                g["companies"] = [c for c in g["companies"] if c["id"] != victim]
                g["employment"] = [
                    e for e in g["employment"] if e["company"] != victim
                ]
                g["partnership"] = [
                    e
                    for e in g["partnership"]
                    if victim not in (e["source"], e["target"])
                ]
            return "node_delete"
        return _apply_one_perturbation(g, "node_add", rng, cb)

    if op == "edge_delete":
        arrays = [
            name
            for name in ("employment", "acquaintance", "partnership")
            if g[name]
        ]
        if arrays:
            name = rng.choice(arrays)
            g[name].pop(rng.randrange(len(g[name])))
            return "edge_delete"
        return _apply_one_perturbation(g, "node_add", rng, cb)

    if op == "node_add":
        used = {p["id"] for p in g["people"]} | {c["id"] for c in g["companies"]}
        new_id = _fresh_hex(rng, used)
        name = f"Node{new_id[:4]}"
        if rng.random() < 0.5 or not g["companies"]:
            titles = [p["title"] for p in g["people"]] or list(cb.title.codes)
            g["people"].append(
                {"id": new_id, "name": name, "title": rng.choice(titles)}
            )
            if g["companies"]:
                roles = [e["role"] for e in g["employment"]] or list(
                    cb.role.codes
                )
                g["employment"].append(
                    {
                        "person": new_id,
                        "company": rng.choice(g["companies"])["id"],
                        "role": rng.choice(roles),
                    }
                )
        else:
            inds = [c["industry"] for c in g["companies"]] or list(
                cb.industry.codes
            )
            g["companies"].append(
                {"id": new_id, "name": name, "industry": rng.choice(inds)}
            )
        return "node_add"

    if op == "edge_add":
        options = []
        if len(g["people"]) >= 2:
            options.append("acquaintance")
        if len(g["companies"]) >= 2:
            options.append("partnership")
        if options:
            name = rng.choice(options)
            if name == "acquaintance":
                a, b = rng.sample(g["people"], 2)
                g["acquaintance"].append(
                    {
                        "source": a["id"],
                        "target": b["id"],
                        "relation": rng.choice(cb.acquaintance.codes),
                    }
                )
            else:
                a, b = rng.sample(g["companies"], 2)
                g["partnership"].append(
                    {
                        "source": a["id"],
                        "target": b["id"],
                        "relation": rng.choice(cb.partnership.codes),
                    }
                )
            return "edge_add"
        return _apply_one_perturbation(g, "node_add", rng, cb)

    # Unknown op — should not happen.
    return _apply_one_perturbation(g, "node_add", rng, cb)


# --- assembly -------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    id: str
    n_people: int
    n_companies: int
    twin_density: float
    acquaintance_density: float
    partnership_density: float
    pred_idx: int
    gold_seed: int
    pred_seed: int
    gold: dict[str, Any]
    pred: dict[str, Any]


def materialize_rows(
    n_golds: int,
    severity_zero_preds: int,
    codebooks: Codebooks,
    master_rng: random.Random,
    property_twins: bool = True,
) -> list[Row]:
    """Draw ``n_golds`` golds with randomized parameters; each gold gets
    ``severity_zero_preds`` independent id relabelings (the id-equivariance
    probe)."""
    rows: list[Row] = []
    counter = 0
    for _ in range(n_golds):
        n_people, n_companies, twin, acq, part = sample_probe_params(master_rng)
        gold_seed = master_rng.randrange(1 << 31)
        gold_rng = random.Random(gold_seed)
        gold = sample_gold(
            n_people=n_people,
            n_companies=n_companies,
            twin_density=twin,
            acquaintance_density=acq,
            partnership_density=part,
            rng=gold_rng,
            codebooks=codebooks,
            role_from_title=False,
        )
        if property_twins:
            make_property_twins(gold, twin, gold_rng)
        for pred_idx in range(severity_zero_preds):
            pred_seed = master_rng.randrange(1 << 31)
            pred_rng = random.Random(pred_seed)
            pred = relabel_ids(gold, pred_rng)
            row_id = f"synra_codex_intrinsic_{counter:05d}"
            counter += 1
            rows.append(
                Row(
                    id=row_id,
                    n_people=n_people,
                    n_companies=n_companies,
                    twin_density=twin,
                    acquaintance_density=acq,
                    partnership_density=part,
                    pred_idx=pred_idx,
                    gold_seed=gold_seed,
                    pred_seed=pred_seed,
                    gold=gold,
                    pred=pred,
                )
            )
    return rows


def validate_rows(rows: list[Row], cfg: ExpConfig) -> None:
    """Round-trip every (gold, pred) through OA's RA schema to confirm both
    sides validate (no dangling refs, well-formed graph)."""
    schema = load_synra_codex_schema(cfg, kind="ra")
    aligner = ObjectAligner(schema)
    for r in rows:
        out = aligner.metric(r.gold, r.pred)
        if not isinstance(out, dict) or "score" not in out:
            raise RuntimeError(f"row {r.id}: unexpected metric output {out!r}")


# --- materialise to disk --------------------------------------------------


def _row_to_meta_json(r: Row) -> dict[str, Any]:
    return {
        "id": r.id,
        "n_people": r.n_people,
        "n_companies": r.n_companies,
        "twin_density": r.twin_density,
        "acquaintance_density": r.acquaintance_density,
        "partnership_density": r.partnership_density,
        "pred_idx": r.pred_idx,
        "gold_seed": r.gold_seed,
        "pred_seed": r.pred_seed,
    }


def _row_to_label_json(r: Row) -> dict[str, Any]:
    return {**_row_to_meta_json(r), "gold": r.gold, "pred": r.pred}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _twin_bin(t: float) -> str:
    if t == 1.0:
        return "=1.0"
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        if lo <= t < hi:
            return f"[{lo},{hi})"
    return "?"


def _write_summary(
    root: Path, splits: dict[str, list[Row]], config: dict[str, Any]
) -> None:
    lines: list[str] = ["# synra_codex intrinsic v2 — sampling summary", ""]
    lines.append(
        f"_Generator seed_: `{config['seed']}` &nbsp;&nbsp; "
        f"_n_golds_: `{config['n_golds']}` &nbsp;&nbsp; "
        f"_severity_zero_preds_: `{config['severity_zero_preds']}`"
    )
    lines.append("")
    for name, rs in splits.items():
        golds = {r.gold_seed for r in rs}
        lines.append(f"## {name} ({len(rs)} rows, {len(golds)} golds)")
        lines.append("")
        for axis, key in [
            ("n_people", lambda r: r.n_people),
            ("n_companies", lambda r: r.n_companies),
            ("twin_density (binned)", lambda r: _twin_bin(r.twin_density)),
        ]:
            ctr = Counter(key(r) for r in rs)
            cells = ", ".join(f"{k}={ctr[k]}" for k in sorted(ctr, key=str))
            lines.append(f"- **{axis}**: {cells}")
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines))


def _write_config(root: Path, config: dict[str, Any]) -> None:
    body = json.dumps(config, indent=2)
    (root / "config.jsonc").write_text(
        "// synra_codex intrinsic v2 — generator configuration\n"
        "// Generated by scripts/prepare_synra_codex_intrinsic.py — do not hand-edit.\n"
        + body
        + "\n"
    )


# --- CLI ------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    cfg = ExpConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--root",
        type=Path,
        default=cfg.synra_codex_intrinsic_v2,
        help="Output root (defaults to data/synra_codex/intrinsic/v2).",
    )
    p.add_argument(
        "--n-golds",
        type=int,
        default=100,
        help="Number of gold graphs; each draws its parameters via "
        "sample_probe_params.",
    )
    p.add_argument(
        "--severity-zero-preds",
        type=int,
        default=5,
        help="Independent id-relabel preds per gold "
        "(id-equivariance sample size).",
    )
    p.add_argument(
        "--property-twins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse names within shared-code groups so twin_density yields "
        "genuine property-twins (distinguishable only by reference structure), "
        "stressing RA at its hardest case. --no-property-twins keeps unique names "
        "(categorical twins only).",
    )
    p.add_argument("--seed", type=int, default=20260609, help="Master RNG seed.")
    p.add_argument("--force", action="store_true", help="Overwrite existing release.")
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the per-row OA validation pass.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = ExpConfig()
    root: Path = args.root
    if root.exists() and any(root.iterdir()) and not args.force:
        print(
            f"refusing to overwrite non-empty {root}; pass --force to clobber.",
            file=sys.stderr,
        )
        return 2
    root.mkdir(parents=True, exist_ok=True)

    codebooks = build_codebooks(vocab=VOCAB, obfuscate=OBFUSCATE)
    master_rng = random.Random(args.seed)

    print(
        f"generating rows: n_golds={args.n_golds} "
        f"severity_zero_preds={args.severity_zero_preds}",
        flush=True,
    )
    rows = materialize_rows(
        n_golds=args.n_golds,
        severity_zero_preds=args.severity_zero_preds,
        codebooks=codebooks,
        master_rng=master_rng,
        property_twins=args.property_twins,
    )
    print(f"  → {len(rows)} rows", flush=True)

    if not args.skip_validation:
        print("validating rows through ObjectAligner …", flush=True)
        validate_rows(rows, cfg)
        print("  → ok", flush=True)

    # Nothing is fitted in this study — a single set, written as `test/` so
    # the scorer/report plumbing stays unchanged.
    splits = {"test": rows}
    split_root = root / "test"
    _write_jsonl(split_root / "labels.jsonl", [_row_to_label_json(r) for r in rows])
    _write_jsonl(split_root / "meta.jsonl", [_row_to_meta_json(r) for r in rows])

    config_doc = {
        "seed": args.seed,
        "n_golds": args.n_golds,
        "severity_zero_preds": args.severity_zero_preds,
        "n_people_range": list(N_PEOPLE_RANGE),
        "acquaintance_density_range": list(ACQUAINTANCE_DENSITY_RANGE),
        "partnership_density_range": list(PARTNERSHIP_DENSITY_RANGE),
        "vocab": VOCAB,
        "obfuscate": OBFUSCATE,
        "property_twins": args.property_twins,
        "n_rows": {name: len(rs) for name, rs in splits.items()},
        "generator": "scripts/prepare_synra_codex_intrinsic.py",
    }
    _write_config(root, config_doc)
    _write_summary(root, splits, config_doc)
    print(f"wrote {root}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
