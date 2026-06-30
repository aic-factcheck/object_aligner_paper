"""BioRED dataset utilities (document-level biomedical relation extraction).

BioRED (Luo et al., Briefings in Bioinformatics 2022) is 600 PubMed abstracts
annotated with normalized biomedical entities and document-level relations. We
download the official ``BIORED.zip`` and read its **BioC JSON** files
(``Train/Dev/Test.BioC.JSON``). Each document has this shape::

    {
      "id": "14510914",                       # PMID
      "passages": [                            # exactly two: title, then abstract
        {"offset": 0, "text": "Congenital ...",
         "annotations": [
            {"id": "0",
             "infons": {"identifier": "D003409", "type": "DiseaseOrPhenotypicFeature"},
             "text": "Congenital hypothyroidism",
             "locations": [{"offset": 0, "length": 25}]},
            ...
         ]},
        {"offset": ..., "text": "<abstract>", "annotations": [...]}
      ],
      "relations": [
        {"id": "R1",
         "infons": {"entity1": "D050033", "entity2": "D007454",
                    "type": "Association", "novel": "No"}},
        ...
      ]
    }

We convert each document to the OA-aligned shape shared with the SciERC / DocRED
schema family (same task: text -> typed coref-entities + relations)::

    {
      "entities": [
        {"id": "D003409", "type": "DiseaseOrPhenotypicFeature",
         "mentions": [{"text": "Congenital hypothyroidism"}, ...]},
        ...
      ],
      "relations": [
        {"subject": "D050033", "predicate": "Association", "object": "D007454"},
        ...
      ]
    }

Design choices (see research/opus48_RA_real_world_datasets_2.md — BioRED is the
top *real* RA-as-fitness demonstrator under the M1 ∧ M2 rule):

* **Accession-as-id.** Each entity `id` is the gold *normalized concept
  accession* (Entrez gene integer, MeSH ``D``-code, OMIM, dbSNP ``rs...``, NCBI
  Taxonomy id, Cellosaurus ``CVCL_...``, or a tmVar ``|``-delimited variant
  string). These are **non-derivable** from the mention text. The model is
  asked to invent its *own* ids, so under the strict schema relation routing is
  starved (the model can never reproduce the accession), while referential
  alignment recovers the gold↔pred bijection from the entity **name + type**.
  This is exactly what makes BioRED behave like ``synra_codex_obf`` and unlike
  DocRED (whose positional ``e0``/``e1`` ids are a *derivable* convention).
* **Mention = annotation surface form**; an entity is the coref cluster of all
  mentions sharing a concept accession. Entity **type** is a majority vote over
  the cluster's mentions (ties broken by first occurrence), mirroring
  ``docred._entity_type``.
* **Composite mention ids.** A mention tagged with several concepts has a
  ``,``/``;``-joined identifier (e.g. ``"D000438,D018943"``); we split it and
  attach the mention to **each** component cluster, so single-concept relation
  endpoints resolve (this brings the unresolved-endpoint rate to 0 across all
  three splits). ``|`` is NOT a separator — it is the internal format of a
  tmVar SequenceVariant id and is kept verbatim.
* **Unnormalized mentions** (identifier ``"-"`` / empty) are dropped — they can
  be neither a cluster nor a relation endpoint.
* **No offsets / novelty**: only mention text + entity type are kept; locations
  and the relation ``novel`` flag are not needed by the OA schema.

Single data variant: BioRED's eight relation types are readable English words
(``Association``, ``Positive_Correlation``, ...), so — unlike DocRED — there is
no obfuscated/native predicate split; the OA schemas, seed prompt, and runner
are shared.
"""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from object_aligner_exp.datasets.rebel import Example, load_jsonl, write_jsonl

# --- constants -------------------------------------------------------------

# BioRED entity types (closed set; meaningful words, scored as-is).
ENTITY_TYPES: tuple[str, ...] = (
    "GeneOrGeneProduct",
    "DiseaseOrPhenotypicFeature",
    "ChemicalEntity",
    "SequenceVariant",
    "OrganismTaxon",
    "CellLine",
)

# BioRED relation types (closed set; readable, scored as-is — no obfuscation).
RELATION_TYPES: tuple[str, ...] = (
    "Association",
    "Positive_Correlation",
    "Negative_Correlation",
    "Bind",
    "Cotreatment",
    "Comparison",
    "Conversion",
    "Drug_Interaction",
)

# Official corpus archive (NCBI; 600 abstracts, public).
BIORED_ZIP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"
# BioC JSON files inside the archive (extracts to a BioRED/ subdir).
_SPLIT_FILES: dict[str, str] = {
    "train": "BioRED/Train.BioC.JSON",
    "dev": "BioRED/Dev.BioC.JSON",
    "test": "BioRED/Test.BioC.JSON",
}

# Multi-concept mention id separators (NOT '|', which is tmVar-internal).
_ID_SEP_RE = re.compile(r"[,;]")


def _id_components(identifier: str) -> list[str]:
    """Split a (possibly multi-concept) mention identifier into component ids.

    Splits on ``,`` and ``;`` only; ``|`` is preserved (tmVar variant format).
    Empty / ``"-"`` components are dropped.
    """
    out: list[str] = []
    for c in _ID_SEP_RE.split(identifier):
        c = c.strip()
        if c and c != "-":
            out.append(c)
    return out


# --- parsing ---------------------------------------------------------------


def _majority_type(types: list[str]) -> str:
    """Majority vote of a cluster's mention types; ties broken by first occurrence."""
    if not types:
        return "MISC"
    return Counter(types).most_common(1)[0][0]


def parse_biored_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert one BioRED BioC-JSON document into the OA-aligned shape.

    Raises ``ValueError`` on malformed input (callers skip such docs).
    """
    passages = doc.get("passages")
    if not isinstance(passages, list) or not passages:
        raise ValueError("missing or empty 'passages'")

    # Accumulate per-concept clusters in first-seen order.
    order: list[str] = []
    types_by_id: dict[str, list[str]] = {}
    mentions_by_id: dict[str, list[str]] = {}
    seen_mention: dict[str, set[str]] = {}

    for passage in passages:
        for ann in passage.get("annotations", []) or []:
            infons = ann.get("infons", {}) or {}
            identifier = str(infons.get("identifier", "")).strip()
            text = str(ann.get("text", "")).strip()
            etype = str(infons.get("type", "")).strip()
            if not text:
                continue
            for cid in _id_components(identifier):
                if cid not in types_by_id:
                    order.append(cid)
                    types_by_id[cid] = []
                    mentions_by_id[cid] = []
                    seen_mention[cid] = set()
                if etype:
                    types_by_id[cid].append(etype)
                if text not in seen_mention[cid]:
                    seen_mention[cid].add(text)
                    mentions_by_id[cid].append(text)

    if not order:
        raise ValueError("no normalized entities in document")

    entities = [
        {
            "id": cid,
            "type": _majority_type(types_by_id[cid]),
            "mentions": [{"text": t} for t in mentions_by_id[cid]],
        }
        for cid in order
    ]

    valid_ids = set(order)
    seen_rel: set[tuple[str, str, str]] = set()
    relations: list[dict[str, str]] = []
    for rel in doc.get("relations", []) or []:
        infons = rel.get("infons", {}) or {}
        subj = str(infons.get("entity1", "")).strip()
        obj = str(infons.get("entity2", "")).strip()
        pred = str(infons.get("type", "")).strip()
        # Defensive: drop relations whose endpoints aren't normalized entities.
        if subj not in valid_ids or obj not in valid_ids or not pred:
            continue
        key = (subj, pred, obj)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        relations.append({"subject": subj, "predicate": pred, "object": obj})

    return {"entities": entities, "relations": relations}


def doc_context(doc: dict[str, Any]) -> str:
    """Reconstruct the document text (title passage + abstract passage)."""
    return " ".join(str(p.get("text", "")) for p in doc.get("passages", []) if p.get("text"))


def doc_title(doc: dict[str, Any]) -> str:
    """The article title — the text of the first passage."""
    passages = doc.get("passages") or []
    return str(passages[0].get("text", "")) if passages else ""


# --- dataset I/O -----------------------------------------------------------


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def ensure_biored_data(raw_dir: Path, *, force: bool = False) -> Path:
    """Make sure the three BioC-JSON splits exist under ``raw_dir``.

    Downloads + extracts ``BIORED.zip`` if any split file is missing. Returns
    ``raw_dir``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    need = [raw_dir / f for f in _SPLIT_FILES.values()]
    if force or not all(p.exists() for p in need):
        zip_path = raw_dir / "BIORED.zip"
        if force or not zip_path.exists():
            _download_file(BIORED_ZIP_URL, zip_path)
        print(f"[extract] {zip_path} -> {raw_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    return raw_dir


def _load_split_docs(split: str, *, raw_dir: Path) -> list[dict[str, Any]]:
    if split not in _SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}")
    path = raw_dir / _SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run ensure_biored_data first")
    doc = json.loads(path.read_text())
    docs = doc.get("documents")
    if not isinstance(docs, list):
        raise ValueError(f"{path}: expected a 'documents' array")
    return docs


def iter_biored_examples(
    split: str,
    *,
    raw_dir: Path,
    limit: int | None = None,
) -> Iterator[Example]:
    """Stream BioRED examples (parsed into OA-aligned form), skipping malformed.

    ``split`` ∈ {"train", "dev", "test"}.
    """
    ensure_biored_data(raw_dir)
    docs = _load_split_docs(split, raw_dir=raw_dir)
    yielded = 0
    for doc in docs:
        try:
            gold = parse_biored_doc(doc)
        except (ValueError, KeyError):
            continue
        pmid = str(doc.get("id", ""))
        yield Example(
            id=pmid,
            title=doc_title(doc),
            context=doc_context(doc),
            gold=gold,
        )
        yielded += 1
        if limit is not None and yielded >= limit:
            return


__all__ = [
    "BIORED_ZIP_URL",
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "doc_context",
    "doc_title",
    "ensure_biored_data",
    "iter_biored_examples",
    "load_jsonl",
    "parse_biored_doc",
    "write_jsonl",
]
