"""REBEL dataset utilities.

REBEL ships each example as a `triplets` *string* of the form

    <triplet> Subj <subj> Obj <obj> rel <subj> Obj2 <obj> rel2 <triplet> Subj2 ...

i.e. `<triplet>` opens a *subject* block; within that block one or more
`<subj> Object <obj> Relation` pairs share the same subject; a new
`<triplet>` starts a new subject. (This matches `Babelscape/rebel-large`'s
own `extract_triplets`: tokens after `<triplet>` are subject, after `<subj>`
are object, after `<obj>` are relation.)

We convert it to the OA-aligned shape

    {"entities":  [{"id": "e0", "name": "Subj"}, ...],
     "relations": [{"subject": "e0", "predicate": "rel", "object": "e1"}, ...]}

Entities are deduplicated by surface form (case-sensitive, stripped). The id
order matches first-occurrence order so different runs of the same corpus
produce identical ids.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


# --- parsing ---------------------------------------------------------------

# Tokens REBEL uses to delimit (subject, object, relation).
TRIPLET = "<triplet>"
SUBJ = "<subj>"
OBJ = "<obj>"

_MARKER_RE = re.compile(rf"({re.escape(TRIPLET)}|{re.escape(SUBJ)}|{re.escape(OBJ)})")


def parse_triplets(triplets_str: str) -> dict[str, Any]:
    """Parse REBEL's linearized-triplets string into an OA-aligned dict.

    Raises `ValueError` on malformed input — callers should treat malformed
    examples as junk and skip them during preprocessing.
    """
    entities: list[dict[str, str]] = []
    entity_id_by_name: dict[str, str] = {}
    relations: list[dict[str, str]] = []

    def intern(name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("empty entity name")
        if name not in entity_id_by_name:
            eid = f"e{len(entities)}"
            entity_id_by_name[name] = eid
            entities.append({"id": eid, "name": name})
        return entity_id_by_name[name]

    subject = ""
    object_ = ""
    relation = ""
    current: str | None = None  # 't' | 's' | 'o'

    def finalize() -> None:
        s, o, r = subject.strip(), object_.strip(), relation.strip()
        if not (s and o and r):
            raise ValueError(
                f"incomplete triple: subject={s!r}, object={o!r}, relation={r!r}"
            )
        relations.append(
            {"subject": intern(s), "predicate": r, "object": intern(o)}
        )

    for part in _MARKER_RE.split(triplets_str):
        if part == TRIPLET:
            if relation.strip():
                finalize()
                relation = ""
            subject = ""
            current = "t"
        elif part == SUBJ:
            if relation.strip():
                finalize()
                relation = ""
            object_ = ""
            current = "s"
        elif part == OBJ:
            relation = ""
            current = "o"
        else:
            if current == "t":
                subject += part
            elif current == "s":
                object_ += part
            elif current == "o":
                relation += part
            # text before any marker is ignored (matches HF data)

    if relation.strip():
        finalize()

    if not relations:
        raise ValueError("no triplets found")

    return {"entities": entities, "relations": relations}


# --- dataset I/O -----------------------------------------------------------


class Example(TypedDict):
    """One preprocessed REBEL example, ready for both task LM and OA scoring."""

    id: str
    title: str
    context: str
    gold: dict[str, Any]


def iter_rebel_examples(split: str, *, limit: int | None = None) -> Iterator[Example]:
    """Stream REBEL examples (parsed into OA-aligned form), skipping malformed.

    `split` ∈ {"train", "validation", "test"}.

    The HF page for `Babelscape/rebel-dataset` ships a legacy loading *script*,
    which modern `datasets` versions reject. We sidestep that by reading the
    Hub's auto-converted parquet shards at `refs/convert/parquet` directly.
    """
    from datasets import load_dataset

    parquet_glob = f"hf://datasets/Babelscape/rebel-dataset@refs/convert/parquet/REBEL/{split}/*.parquet"
    # `streaming=True` avoids materializing the 3.5M-row train split on disk.
    ds = load_dataset("parquet", data_files=parquet_glob, split="train", streaming=True)

    yielded = 0
    for row in ds:
        try:
            gold = parse_triplets(row["triplets"])
        except ValueError:
            continue

        yield Example(
            id=row["id"],
            title=row["title"],
            context=row["context"],
            gold=gold,
        )
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def write_jsonl(path: Path, examples: Iterable[Example]) -> int:
    """Write examples as JSON lines. Returns the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: Path) -> list[Example]:
    """Load a JSONL file produced by `write_jsonl`."""
    out: list[Example] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
