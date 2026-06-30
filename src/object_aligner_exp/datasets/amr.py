"""AMR (Abstract Meaning Representation) dataset utilities.

The model is given an English sentence and must output its AMR **meaning
graph**: a rooted, directed graph of *variable* nodes each labelled with a
PropBank/concept symbol (``see-01``, ``boy``), connected by *role* edges
(``:ARG0``, ``:mod``) — with **reentrancy** (one variable referenced by several
edges) making it a true graph, not a tree.

AMR variable names (``s``, ``i``, ``b2``) are **arbitrary**: the same graph is
correct under any consistent relabeling. That is exactly why the standard AMR
metric, **Smatch**, searches for the best variable alignment before counting
matched triples — i.e. Smatch *is* Object Aligner's referential alignment
(``idScope``/``ref``). This dataset is the cleanest real-world RA fit; see
``research/opus48_RA_real_world_datasets.md``.

Wire shape (fixed JSON schema, mirrors Smatch's instance/relation/attribute
triple taxonomy, with constant-valued attributes nested under their node so they
sharpen node individuation):

    {
      "root": "<node-id>",
      "nodes": [
        {"id": "<var>", "concept": "<symbol>",
         "attributes": [{"role": ":opN"|":polarity"|..., "value": "<const>"}]}
      ],
      "relations": [{"source": "<var>", "role": ":ARG0"|..., "target": "<var>"}]
    }

``meta`` keeps the AMR id and ``n_nodes`` (recorded for optional analysis; the
Little Prince carve is unstratified). The native leaderboard metric (Smatch F1)
lives in ``amr_eval.py`` and consumes the same JSON via the smatch-triple
helpers below.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, TypedDict

import penman


class AmrExample(TypedDict):
    """One preprocessed AMR example."""

    id: str
    title: str
    context: str
    gold: dict[str, Any]
    meta: dict[str, Any]


# --- constant (de)quoting --------------------------------------------------

_QUOTED = re.compile(r'^"(.*)"$', re.DOTALL)
# A bare PENMAN constant that needs no quoting (number, +/-, simple symbol).
_BARE_CONST = re.compile(r'^[A-Za-z0-9_.+\-:]+$')


def _dequote(const: str) -> str:
    """Strip PENMAN surface quotes so the gold value matches what an LM emits.

    ``'"True"'`` -> ``'True'``; ``'-'`` -> ``'-'``; ``'6'`` -> ``'6'``.
    """
    m = _QUOTED.match(const)
    return m.group(1) if m else const


def _as_penman_const(value: str) -> str:
    """Render a (dequoted) value as a valid PENMAN constant (re-quote if needed)."""
    s = str(value)
    return s if _BARE_CONST.match(s) else '"' + s.replace('"', '\\"') + '"'


# --- PENMAN graph -> gold JSON ---------------------------------------------


def penman_to_gold(g: penman.Graph) -> dict[str, Any]:
    """Convert a decoded PENMAN graph into the fixed AMR wire shape."""
    concepts = {src: tgt for src, _role, tgt in g.instances()}
    attrs_by_src: dict[str, list[dict[str, str]]] = {v: [] for v in concepts}
    for src, role, tgt in g.attributes():
        attrs_by_src.setdefault(src, []).append(
            {"role": role, "value": _dequote(tgt)}
        )
    nodes = [
        {"id": v, "concept": concepts[v], "attributes": attrs_by_src.get(v, [])}
        for v in concepts  # dict preserves penman's graph order
    ]
    relations = [
        {"source": src, "role": role, "target": tgt} for src, role, tgt in g.edges()
    ]
    return {"root": g.top, "nodes": nodes, "relations": relations}


# --- gold JSON -> triples (for self-test / debug / Smatch) ------------------


def gold_to_triples(gold: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Flatten the wire shape to penman-style triples (instances + attrs + edges).

    Attribute values are kept dequoted (so the set is comparable to a graph whose
    attributes were also dequoted). Variable ids are preserved.
    """
    triples: list[tuple[str, str, str]] = []
    for n in gold.get("nodes", []) or []:
        triples.append((n["id"], ":instance", str(n.get("concept"))))
    for n in gold.get("nodes", []) or []:
        for a in n.get("attributes", []) or []:
            triples.append((n["id"], str(a["role"]), str(a["value"])))
    for r in gold.get("relations", []) or []:
        triples.append((str(r["source"]), str(r["role"]), str(r["target"])))
    return triples


def gold_to_smatch_triples(
    gold: dict[str, Any], prefix: str
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Build smatch's (instance, attribute, relation) triple lists.

    Node ids are renamed to ``<prefix>0``, ``<prefix>1``, … (smatch requires the
    two graphs' variables to share a per-graph prefix and be disjoint). Roles are
    stripped of their leading ``:`` and the root contributes the ``TOP`` pseudo-
    attribute, exactly as ``smatch.amr.AMR.parse_AMR_line`` would produce.
    Robust to malformed predictions: dangling relation endpoints are dropped.
    """
    idmap: dict[str, str] = {}
    instances: list[tuple[str, str, str]] = []
    for i, n in enumerate(gold.get("nodes", []) or []):
        nid = n.get("id")
        if nid is None or nid in idmap:
            continue  # skip missing/duplicate ids
        v = f"{prefix}{i}"
        idmap[nid] = v
        instances.append(("instance", v, str(n.get("concept"))))

    attributes: list[tuple[str, str, str]] = []
    root = gold.get("root")
    if root in idmap:
        attributes.append(("TOP", idmap[root], "top"))
    for n in gold.get("nodes", []) or []:
        v = idmap.get(n.get("id"))
        if v is None:
            continue
        for a in n.get("attributes", []) or []:
            attributes.append((str(a.get("role", "")).lstrip(":"), v, str(a.get("value"))))

    relations: list[tuple[str, str, str]] = []
    for r in gold.get("relations", []) or []:
        s = idmap.get(r.get("source"))
        t = idmap.get(r.get("target"))
        if s is None or t is None:
            continue  # dangling ref -> dropped (counts against P/R, never raises)
        relations.append((str(r.get("role", "")).lstrip(":"), s, t))
    return instances, attributes, relations


def gold_to_penman(gold: dict[str, Any]) -> str:
    """Encode the wire shape back to a PENMAN string (debug / round-trip checks)."""
    var_ids = {n["id"] for n in gold.get("nodes", []) or []}
    triples: list[tuple[str, str, str]] = []
    for n in gold.get("nodes", []) or []:
        triples.append((n["id"], ":instance", str(n.get("concept"))))
        for a in n.get("attributes", []) or []:
            triples.append((n["id"], str(a["role"]), _as_penman_const(a["value"])))
    for r in gold.get("relations", []) or []:
        if r.get("target") in var_ids:
            triples.append((str(r["source"]), str(r["role"]), str(r["target"])))
    top = gold.get("root")
    return penman.encode(penman.Graph(triples, top=top))


# --- corpus reading --------------------------------------------------------


def iter_amr_examples(
    source_path: str, *, limit: int | None = None
) -> Iterator[AmrExample]:
    """Stream AMR examples from a PENMAN file (``# ::id`` / ``# ::snt`` blocks)."""
    text = open(source_path, encoding="utf-8").read()
    graphs = penman.loads(text)
    yielded = 0
    for g in graphs:
        snt = (g.metadata.get("snt") or "").strip()
        amr_id = (g.metadata.get("id") or f"amr_{yielded}").strip()
        if not snt or not g.instances():
            continue  # skip header-only / empty blocks
        gold = penman_to_gold(g)
        ex = AmrExample(
            id=amr_id,
            title=amr_id,
            context=snt,
            gold=gold,
            meta={"n_nodes": len(gold["nodes"]), "difficulty": len(gold["nodes"])},
        )
        yield ex
        yielded += 1
        if limit is not None and yielded >= limit:
            return


# --- self-test -------------------------------------------------------------


def _norm_triples(triples: list[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    """Dequote attribute-style targets so two triple sets are comparable."""
    return {(s, r, _dequote(t)) for s, r, t in triples}


def roundtrip_ok(g: penman.Graph) -> bool:
    """True iff penman -> gold JSON -> triples reproduces the graph's triple set."""
    gold = penman_to_gold(g)
    return _norm_triples(gold_to_triples(gold)) == _norm_triples(list(g.triples))
