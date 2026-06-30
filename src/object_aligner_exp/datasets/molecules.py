"""Molecule OCSR (optical chemical structure recognition) dataset utilities.

The model is given an **image** of a 2D chemical structure and must output its
**molecular graph**: a set of *atom* nodes (element + formal charge + hydrogen
count) connected by *bond* edges (single / double / triple / aromatic). Atom
ids follow RDKit's arbitrary atom-index order — the same molecule is correct
under any consistent atom relabeling, which is exactly why correctness is graph
isomorphism *up to atom relabeling* and why this is a natural Referential
Alignment fit (atom indices are the arbitrary, model-invisible ids; 1-WL was
literally invented for molecular graph isomorphism). See
``research/opus48_RA_real_world_datasets.md`` (Tier 2, "fair GraphImg rematch").

Wire shape (fixed JSON schema, scored by ``data/molecules/schemas/molecules_*``):

    {
      "atoms": [ {"id": <int>, "element": "C", "charge": 0, "num_h": 1}, ... ],
      "bonds": [ {"source": <int>, "target": <int>,
                  "order": "single"|"double"|"triple"|"aromatic"}, ... ]
    }

``num_h`` (total attached hydrogens) and aromatic bond orders are carried so the
graph reconstructs to a unique molecule: aromatic bonds avoid the benzene-Kekulé
ambiguity (both Kekulé structures would otherwise differ in which bonds are
single/double, unfairly penalising RA), and ``num_h`` disambiguates hetero-
aromatics (pyridine ``n`` vs pyrrole ``[nH]``). The canonical-SMILES + graph-F1
native metric lives in ``molecules_eval.py``; the labels.jsonl row also stores a
gold canonical SMILES for that scorer (outside ``graph`` so OA never scores it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

# Predicted molecule graphs are routinely invalid (bad valences, duplicate
# bonds, unkekulisable rings); we catch those Python-side and score them as
# failures. RDKit's C++ layer still spams stderr regardless of the try/except,
# so silence its logger at import time.
RDLogger.DisableLog("rdApp.*")

# Bond-type <-> wire-order mapping. Aromatic is kept as its own order (not
# Kekulised) so a single canonical wire shape exists per molecule.
_BT = Chem.BondType
_ORDER_TO_BT: dict[str, Any] = {
    "single": _BT.SINGLE,
    "double": _BT.DOUBLE,
    "triple": _BT.TRIPLE,
    "aromatic": _BT.AROMATIC,
}
_BT_TO_ORDER: dict[Any, str] = {
    _BT.SINGLE: "single",
    _BT.DOUBLE: "double",
    _BT.TRIPLE: "triple",
    _BT.AROMATIC: "aromatic",
}


# --- SMILES / mol -> gold JSON ---------------------------------------------


def mol_to_gold(mol: Chem.Mol) -> dict[str, Any]:
    """Convert a sanitised RDKit mol into the fixed atom/bond wire shape.

    ``mol`` must already be sanitised (aromaticity perceived, implicit Hs
    computed) — e.g. the output of :func:`Chem.MolFromSmiles`.
    """
    atoms = [
        {
            "id": a.GetIdx(),
            "element": a.GetSymbol(),
            "charge": a.GetFormalCharge(),
            "num_h": a.GetTotalNumHs(),
        }
        for a in mol.GetAtoms()
    ]
    bonds = []
    for b in mol.GetBonds():
        order = "aromatic" if b.GetIsAromatic() else _BT_TO_ORDER.get(b.GetBondType())
        if order is None:  # exotic bond type (dative, etc.) — not supported
            raise ValueError(f"unsupported bond type {b.GetBondType()!r}")
        bonds.append(
            {"source": b.GetBeginAtomIdx(), "target": b.GetEndAtomIdx(), "order": order}
        )
    return {"atoms": atoms, "bonds": bonds}


def gold_to_mol(gold: dict[str, Any]) -> Chem.Mol:
    """Rebuild a sanitised RDKit mol from the atom/bond wire shape.

    Robust to id gaps / arbitrary id ordering: atoms are added in list order and
    a local ``id -> rdkit index`` map routes the bonds. Raises on a structurally
    invalid graph (RDKit ``SanitizeMol`` failure, dangling bond endpoint) so the
    self-test and native scorer can treat it as a miss rather than crash.
    """
    rw = Chem.RWMol()
    idmap: dict[Any, int] = {}
    for a in gold.get("atoms", []) or []:
        atom = Chem.Atom(str(a["element"]))
        atom.SetFormalCharge(int(a.get("charge", 0) or 0))
        if a.get("num_h") is not None:
            atom.SetNumExplicitHs(int(a["num_h"]))
            atom.SetNoImplicit(True)
        idx = rw.AddAtom(atom)
        idmap[a["id"]] = idx
    for b in gold.get("bonds", []) or []:
        s = idmap.get(b.get("source"))
        t = idmap.get(b.get("target"))
        if s is None or t is None or s == t:
            raise ValueError(f"dangling/self bond {b!r}")
        order = str(b.get("order", "single"))
        bt = _ORDER_TO_BT.get(order)
        if bt is None:
            raise ValueError(f"unknown bond order {order!r}")
        bidx = rw.AddBond(s, t, bt) - 1
        if order == "aromatic":
            bond = rw.GetBondWithIdx(bidx)
            bond.SetIsAromatic(True)
            rw.GetAtomWithIdx(s).SetIsAromatic(True)
            rw.GetAtomWithIdx(t).SetIsAromatic(True)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def canonical_smiles(src: str | Chem.Mol) -> str | None:
    """Canonical SMILES of a SMILES string or mol; ``None`` if unparseable."""
    mol = Chem.MolFromSmiles(src) if isinstance(src, str) else src
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def gold_canonical_smiles(gold: dict[str, Any]) -> str | None:
    """Canonical SMILES reconstructed from the atom/bond wire shape (or None)."""
    try:
        return Chem.MolToSmiles(gold_to_mol(gold))
    except Exception:  # noqa: BLE001 — any RDKit failure => not recoverable
        return None


def roundtrip_ok(smiles: str) -> bool:
    """True iff SMILES -> gold JSON -> mol reproduces the canonical SMILES.

    The molecule analog of ``datasets.amr.roundtrip_ok``: guards the conversion
    so ``prepare_molecules`` aborts loudly if any gold graph is lossy.
    """
    canon = canonical_smiles(smiles)
    if canon is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    try:
        return gold_canonical_smiles(mol_to_gold(mol)) == canon
    except Exception:  # noqa: BLE001
        return False


# --- rendering (the `rendered` variant) ------------------------------------


def render_png(mol: Chem.Mol, *, size: tuple[int, int] = (384, 384)) -> bytes:
    """Render a 2D depiction of ``mol`` to PNG bytes (RDKit Cairo/AGG backend)."""
    img = Draw.MolToImage(mol, size=size)
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- gold loading for the report's native scorer ---------------------------


def gold_by_id_from_labels(labels_path: Path) -> dict[str, dict[str, Any]]:
    """Load a graphimg-style ``labels.jsonl`` into ``{id: {graph, smiles}}``.

    Used by the report's native scorer (``report/results.py`` ``image_dir`` gold
    kind): it needs both the gold ``graph`` (atoms/bonds) and the gold canonical
    ``smiles`` per id, which ``load_graphimg_split`` (OA path) does not surface.
    """
    out: dict[str, dict[str, Any]] = {}
    with Path(labels_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if not isinstance(rid, str):
                continue
            out[rid] = {"graph": rec.get("graph") or {}, "smiles": rec.get("smiles")}
    return out
