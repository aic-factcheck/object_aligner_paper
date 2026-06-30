"""Loader for the synthetic graphimg dataset (PNG + gold graph JSON).

graphimg's split directories look like::

    <split>/
        labels.jsonl           # one record per line
        images/g_*.png         # one PNG per record

Each ``labels.jsonl`` row is::

    {
      "id":         "g_<hex>",
      "image_path": "images/g_<hex>.png",
      "graph":      { "id", "family", "directed", "nodes", "edges" }
    }

This loader normalises rows to ``{id, image_abspath, gold}`` so callers can
pass ``image_abspath`` straight into the image-aware GEPA adapter and
``gold`` into the OA evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class GraphImgRow(TypedDict):
    id: str
    image_abspath: Path
    gold: dict[str, Any]


def load_graphimg_split(split_dir: Path) -> list[GraphImgRow]:
    """Load ``<split_dir>/labels.jsonl`` and resolve each ``image_path``.

    Raises ``FileNotFoundError`` if ``labels.jsonl`` is missing or any image
    referenced by it does not exist on disk — we want a fast failure rather
    than a 50-row GEPA run that mysteriously scores 0.
    """
    split_dir = Path(split_dir)
    labels_path = split_dir / "labels.jsonl"
    if not labels_path.is_file():
        raise FileNotFoundError(f"graphimg split missing labels.jsonl: {labels_path}")

    rows: list[GraphImgRow] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{labels_path}:{line_no}: invalid JSON ({exc.msg})"
                ) from exc

            rel = rec.get("image_path")
            graph = rec.get("graph")
            rid = rec.get("id") or (graph.get("id") if isinstance(graph, dict) else None)
            if not isinstance(rel, str) or not isinstance(graph, dict) or not isinstance(rid, str):
                raise ValueError(
                    f"{labels_path}:{line_no}: row missing required fields "
                    f"(need 'id', 'image_path', 'graph'); got keys={list(rec.keys())}"
                )

            abspath = (split_dir / rel).resolve()
            if not abspath.is_file():
                raise FileNotFoundError(
                    f"{labels_path}:{line_no}: image not found at {abspath} "
                    f"(referenced as {rel!r})"
                )

            # The graph block carries an internal sample id (e.g. "g_<hex>")
            # that the model can't possibly read from the image and that the
            # OA schema doesn't score. Drop it from the gold so we don't
            # confuse the schema-driven comparison.
            gold = {k: v for k, v in graph.items() if k != "id"}
            rows.append(GraphImgRow(id=rid, image_abspath=abspath, gold=gold))

    return rows
