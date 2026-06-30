"""SciERC dataset utilities.

SciERC (Luan et al., EMNLP 2018) ships per-document JSON (one document per
line in train/dev/test.json) with this shape:

    {
      "doc_key":   "X96-1059",
      "sentences": [ [tok, tok, ...], [tok, ...], ... ],          # per-sentence tokens
      "ner":       [ [[s, e, "Task"], ...], [...], ... ],         # per-sentence; flat doc-wide token offsets
      "relations": [ [[ss, se, os, oe, "USED-FOR"], ...], ... ],  # per-sentence; same offset scheme
      "clusters":  [ [[s, e], [s, e], ...], ... ]                 # doc-level coref; multi-mention clusters only
    }

Token offsets `[start, end]` are **inclusive** and indexed into the *flat*
document-wide token sequence (sentence 1 tokens come after sentence 0
tokens).

We convert each document to the OA-aligned shape used by the schema in
``data/scierc/schemas/scierc.jsonc``:

    {
      "entities": [
        {"id": "e0", "type": "Method",
         "mentions": [{"text": "neural attention model"}, {"text": "Our model"}, ...]},
        ...
      ],
      "relations": [
        {"subject": "e0", "predicate": "USED-FOR", "object": "e1"},
        ...
      ]
    }

Design choices (documented at ``research/opus47_datasets_analysis.md``):

* **Cluster-as-entity**: every coreference cluster becomes one entity. NER
  spans that don't appear in any cluster become singleton entities. This
  exercises OA's nested-mentions Hungarian alignment and `idScope`/`ref`.
* **No offsets**: only mention text is kept. Char/token offsets are noisy
  for LLMs and not needed by the OA schema we chose.
* **Cluster type by majority vote**: each cluster typically has consistent
  type across its mentions; on the rare disagreement we pick the most
  frequent (ties broken by first occurrence).
* **Relation projection**: SciERC annotates relations between mention spans;
  we project subject/object spans to their cluster ids and dedupe at the
  cluster level. A relation whose endpoints we can't resolve to a cluster
  is dropped (shouldn't happen given the dataset's invariants, but we are
  defensive).
"""

from __future__ import annotations

import json
import tarfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypedDict

from object_aligner_exp.datasets.rebel import Example, load_jsonl, write_jsonl


# --- parsing ---------------------------------------------------------------

ENTITY_TYPES: tuple[str, ...] = (
    "Task",
    "Method",
    "Metric",
    "Material",
    "OtherScientificTerm",
    "Generic",
)

RELATION_TYPES: tuple[str, ...] = (
    "USED-FOR",
    "FEATURE-OF",
    "HYPONYM-OF",
    "PART-OF",
    "COMPARE",
    "CONJUNCTION",
    "EVALUATE-FOR",
)


def _flat_tokens(sentences: list[list[str]]) -> list[str]:
    return [t for s in sentences for t in s]


def _span_text(tokens: list[str], start: int, end: int) -> str:
    """Inclusive [start, end] → joined token string."""
    return " ".join(tokens[start : end + 1])


def parse_scierc_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert one SciERC document dict into the OA-aligned shape.

    Raises ``ValueError`` on malformed input.
    """
    try:
        sentences = doc["sentences"]
        ner = doc["ner"]
        relations = doc["relations"]
    except KeyError as exc:
        raise ValueError(f"missing required field: {exc.args[0]!r}") from exc

    if not isinstance(sentences, list) or not sentences:
        raise ValueError("empty or non-list 'sentences'")

    tokens = _flat_tokens(sentences)

    # Collect every NER mention span → (text, type).
    # SciERC allows overlapping/nested spans; we key on the (start, end) tuple
    # but a single span can carry only one type (it's a single NER tag).
    span_to_text: dict[tuple[int, int], str] = {}
    span_to_type: dict[tuple[int, int], str] = {}
    for sent_ner in ner:
        for entry in sent_ner:
            start, end, etype = entry[0], entry[1], entry[2]
            span = (int(start), int(end))
            if not (0 <= span[0] <= span[1] < len(tokens)):
                raise ValueError(f"NER span {span} out of range (n_tokens={len(tokens)})")
            span_to_text[span] = _span_text(tokens, *span)
            span_to_type[span] = str(etype)

    # Coref clusters from `clusters` (multi-mention only); add singletons for
    # NER spans that don't show up in any cluster.
    raw_clusters: list[list[tuple[int, int]]] = []
    seen_in_cluster: set[tuple[int, int]] = set()
    for cluster in doc.get("clusters", []) or []:
        spans = [(int(s), int(e)) for s, e in cluster]
        raw_clusters.append(spans)
        seen_in_cluster.update(spans)
    for span in span_to_text:
        if span not in seen_in_cluster:
            raw_clusters.append([span])

    # Build entities. Order = first-occurrence of each cluster's first span.
    entities: list[dict[str, Any]] = []
    cluster_id_by_span: dict[tuple[int, int], str] = {}
    for cluster_spans in raw_clusters:
        types_in_cluster = [span_to_type[s] for s in cluster_spans if s in span_to_type]
        if not types_in_cluster:
            # Cluster references spans that aren't NER-tagged. SciERC's
            # invariant should preclude this; skip defensively.
            continue
        etype = Counter(types_in_cluster).most_common(1)[0][0]

        # Deduplicate mention surface forms while preserving order.
        seen_texts: set[str] = set()
        mentions: list[dict[str, str]] = []
        for span in cluster_spans:
            text = span_to_text.get(span)
            if text is None:
                continue
            if text not in seen_texts:
                mentions.append({"text": text})
                seen_texts.add(text)

        eid = f"e{len(entities)}"
        for span in cluster_spans:
            if span in span_to_text:
                cluster_id_by_span[span] = eid
        entities.append({"id": eid, "type": etype, "mentions": mentions})

    if not entities:
        raise ValueError("no entities produced from this document")

    # Project mention-level relations to cluster-level. Dedupe on the
    # (subject_cluster, predicate, object_cluster) triple — SciERC sometimes
    # annotates the same cluster-level relation via multiple mention pairs.
    seen_rel: set[tuple[str, str, str]] = set()
    rel_out: list[dict[str, str]] = []
    for sent_rel in relations:
        for entry in sent_rel:
            ss, se, os_, oe, label = entry[0], entry[1], entry[2], entry[3], entry[4]
            subj = cluster_id_by_span.get((int(ss), int(se)))
            obj = cluster_id_by_span.get((int(os_), int(oe)))
            if subj is None or obj is None:
                continue
            key = (subj, str(label), obj)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            rel_out.append({"subject": subj, "predicate": str(label), "object": obj})

    return {"entities": entities, "relations": rel_out}


def doc_context(doc: dict[str, Any]) -> str:
    """Reconstruct the abstract text from per-sentence token lists.

    We join tokens within a sentence with single spaces and sentences with
    a single space too — the result is detokenization-noisy ("[ Tom ] '
    s") but adequate for an LLM input. SciERC's released format does not
    ship raw text, so this is the best we can do without re-running a
    detokenizer.
    """
    return " ".join(" ".join(s) for s in doc["sentences"])


# --- dataset I/O -----------------------------------------------------------


_DEFAULT_JSON_SUBPATH = Path("processed_data/json")


def _extract_json_files(tarball: Path, out_dir: Path) -> None:
    """Extract only the ``processed_data/json/*.json`` files from the tarball.

    The official tarball also bundles 600+MB of ELMo embeddings which we
    don't need; skipping them saves disk and a long extract.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        members = [
            m for m in tf.getmembers()
            if m.name.startswith("processed_data/json/") and m.name.endswith(".json")
        ]
        if not members:
            raise RuntimeError(
                f"no processed_data/json/*.json found inside {tarball}"
            )
        tf.extractall(path=out_dir, members=members)


def ensure_scierc_json(raw_dir: Path) -> Path:
    """Make sure `<raw_dir>/processed_data/json/{train,dev,test}.json` exist.

    Extracts from ``<raw_dir>/sciERC_processed.tar.gz`` if needed. Does NOT
    download — the prepare script handles that and points us here.
    """
    json_dir = raw_dir / _DEFAULT_JSON_SUBPATH
    needed = ("train.json", "dev.json", "test.json")
    if all((json_dir / n).exists() for n in needed):
        return json_dir

    tarball = raw_dir / "sciERC_processed.tar.gz"
    if not tarball.exists():
        raise FileNotFoundError(
            f"Expected SciERC JSON files under {json_dir}, and no tarball at "
            f"{tarball} to extract from. Re-run prepare_scierc with a working "
            "download URL or place the tarball there manually."
        )
    _extract_json_files(tarball, raw_dir)

    missing = [n for n in needed if not (json_dir / n).exists()]
    if missing:
        raise RuntimeError(
            f"After extracting {tarball.name}, still missing: {missing}"
        )
    return json_dir


def iter_scierc_examples(
    split: str,
    *,
    raw_dir: Path,
    limit: int | None = None,
) -> Iterator[Example]:
    """Stream SciERC examples (parsed into OA-aligned form), skipping malformed.

    ``split`` ∈ {"train", "dev", "test"}.
    """
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"unknown split {split!r}")
    json_dir = ensure_scierc_json(raw_dir)
    path = json_dir / f"{split}.json"
    yielded = 0
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                gold = parse_scierc_doc(doc)
            except (ValueError, json.JSONDecodeError):
                continue
            yield Example(
                id=str(doc.get("doc_key", "")),
                title=str(doc.get("doc_key", "")),
                context=doc_context(doc),
                gold=gold,
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return


__all__ = [
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "doc_context",
    "ensure_scierc_json",
    "iter_scierc_examples",
    "load_jsonl",
    "parse_scierc_doc",
    "write_jsonl",
]
