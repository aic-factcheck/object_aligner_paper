"""Native (non-OA) evaluation of a molecule OCSR run.

OA's referential-alignment score (``molecules_ra``) is the GEPA *reward*; this
module recomputes the field-standard correctness signals so a run can be placed
against published OCSR numbers:

* ``canon_smiles_match`` — fraction of predictions whose reconstructed molecule
  has the **same RDKit canonical SMILES** as the gold (exact graph match up to
  atom relabeling + aromatic perception; the binary OCSR success criterion,
  analogous to AMR's exact-graph notion). Headline metric.
* ``graph_f1`` — micro-averaged **Smatch-style triple F1** over atom-instance
  triples (element), atom attributes (charge / H-count) and bond relations
  (order). Like AMR's Smatch, the best atom alignment is found before counting
  matched triples — so this is graded partial credit invariant to atom
  relabeling (it tracks OA-RA, while ``canon_smiles_match`` is the strict 0/1).

Mirrors ``amr_eval``: ``score_corpus`` takes ``{id, response}`` rows + a dict
id → gold (carrying the gold ``graph`` and canonical ``smiles``). A malformed or
chemically invalid prediction never raises — it scores 0 matched triples / no
SMILES match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import smatch

from object_aligner_exp.datasets.molecules import (
    canonical_smiles,
    gold_canonical_smiles,
    gold_to_mol,
)

__all__ = ["TaskResult", "score_corpus"]


def _prf(match: int, test: int, gold: int) -> tuple[float, float, float]:
    p = match / test if test else 0.0
    r = match / gold if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


@dataclass
class TaskResult:
    task: str = "molecules"
    n: int = 0
    n_parse_failures: int = 0
    n_smiles_match: int = 0
    match_total: int = 0
    test_total: int = 0
    gold_total: int = 0
    per_example: list[dict[str, Any]] = field(default_factory=list)

    @property
    def canon_smiles_match(self) -> float:
        return self.n_smiles_match / self.n if self.n else 0.0

    @property
    def graph_precision(self) -> float:
        return _prf(self.match_total, self.test_total, self.gold_total)[0]

    @property
    def graph_recall(self) -> float:
        return _prf(self.match_total, self.test_total, self.gold_total)[1]

    @property
    def graph_f1(self) -> float:
        return _prf(self.match_total, self.test_total, self.gold_total)[2]


def _parse_response(resp: Any) -> dict[str, Any] | None:
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, str):
        try:
            obj = json.loads(resp)
        except (json.JSONDecodeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _to_smatch_triples(
    graph: dict[str, Any], prefix: str
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Build smatch (instance, attribute, relation) triple lists from a graph.

    Atoms → instance (element) + attributes (charge, numH); bonds → relation
    (order). Atom ids are renamed to ``<prefix>0…`` (smatch needs disjoint,
    per-graph-prefixed variables). Robust to malformed graphs: missing/duplicate
    ids and dangling bond endpoints are dropped (counting against P/R).
    """
    idmap: dict[Any, str] = {}
    instances: list[tuple[str, str, str]] = []
    attributes: list[tuple[str, str, str]] = []
    for i, a in enumerate(graph.get("atoms", []) or []):
        aid = a.get("id")
        if aid is None or aid in idmap:
            continue
        v = f"{prefix}{i}"
        idmap[aid] = v
        instances.append(("instance", v, str(a.get("element"))))
        attributes.append(("charge", v, str(a.get("charge", 0))))
        if a.get("num_h") is not None:
            attributes.append(("numH", v, str(a.get("num_h"))))

    relations: list[tuple[str, str, str]] = []
    for b in graph.get("bonds", []) or []:
        s = idmap.get(b.get("source"))
        t = idmap.get(b.get("target"))
        if s is None or t is None:
            continue
        relations.append((str(b.get("order", "")), s, t))
    return instances, attributes, relations


def _graph_match(pred: dict[str, Any], gold: dict[str, Any]) -> tuple[int, int, int]:
    smatch.match_triple_dict = {}  # reset smatch's per-pair cache
    i1, a1, r1 = _to_smatch_triples(pred, "a")
    i2, a2, r2 = _to_smatch_triples(gold, "b")
    test_total = len(i1) + len(a1) + len(r1)
    gold_total = len(i2) + len(a2) + len(r2)
    if test_total == 0 or gold_total == 0:
        return 0, test_total, gold_total
    _, match = smatch.get_best_match(i1, a1, r1, i2, a2, r2, "a", "b")
    return match, test_total, gold_total


def _gold_smiles(gold_entry: dict[str, Any]) -> str | None:
    smi = gold_entry.get("smiles")
    if isinstance(smi, str) and smi:
        # Re-canonicalise so a match is exact under one convention.
        return canonical_smiles(smi)
    return gold_canonical_smiles(gold_entry.get("graph") or {})


def score_corpus(
    predictions: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
) -> TaskResult:
    """Score ``{id, response}`` rows against gold molecules.

    ``gold_by_id`` maps id → ``{"graph": {atoms,bonds}, "smiles": <canonical>}``
    (see ``datasets.molecules.gold_by_id_from_labels``). Rows whose id is absent
    are skipped with a warning; an unparseable / empty prediction is a parse
    failure (0 SMILES match, all gold triples become false negatives).
    """
    res = TaskResult()
    for row in predictions:
        ex_id = str(row.get("id", ""))
        entry = gold_by_id.get(ex_id)
        if entry is None:
            print(f"[molecules_eval] skipping {ex_id!r}: not in gold")
            continue
        gold_graph = entry.get("graph") or {}
        gold_smi = _gold_smiles(entry)

        pred = _parse_response(row.get("response"))
        parsed_ok = isinstance(pred, dict) and bool(pred.get("atoms"))
        if not parsed_ok:
            res.n_parse_failures += 1
            pred = {"atoms": [], "bonds": []}

        try:
            match, test_total, gold_total = _graph_match(pred, gold_graph)
        except Exception:  # noqa: BLE001 — structurally invalid predicted graph
            # e.g. an atom/bond that is a bare string instead of a dict; score as
            # a parse failure rather than aborting the whole corpus.
            if parsed_ok:
                res.n_parse_failures += 1
            parsed_ok = False
            pred = {"atoms": [], "bonds": []}
            match, test_total, gold_total = _graph_match(pred, gold_graph)

        pred_smi = None
        if parsed_ok:
            try:
                pred_smi = canonical_smiles(gold_to_mol(pred))
            except Exception:  # noqa: BLE001 — invalid chemistry => no match
                pred_smi = None
        is_match = bool(gold_smi is not None and pred_smi == gold_smi)

        res.n += 1
        res.n_smiles_match += int(is_match)
        res.match_total += match
        res.test_total += test_total
        res.gold_total += gold_total
        res.per_example.append(
            {
                "id": ex_id,
                "smiles_match": is_match,
                "match": match,
                "test": test_total,
                "gold": gold_total,
            }
        )
    return res
