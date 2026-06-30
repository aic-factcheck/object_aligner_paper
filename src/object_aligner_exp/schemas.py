"""Schema loaders.

Schemas live under `data/<dataset>/schemas/<name>.jsonc` so they can carry
inline comments documenting each `idScope` / `ref` / leaf metric, and so a
manifest can pin the exact schema a run was carved against.

The runners take ``--schema PATH`` (and a repeatable ``--cross-schema PATH``)
and load each file directly through :func:`load_schema_from_path`. The
dataset-specific helpers below are kept as convenience wrappers around the
canonical paths exposed on :class:`ExpConfig`, useful from tests and
notebooks where the path indirection is overkill.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json5

from object_aligner_exp.config import ExpConfig


def load_schema_from_path(path: Path | str) -> dict[str, Any]:
    """Load a JSONC schema from ``path`` into a dict.

    Used directly by every entry point that takes ``--schema PATH`` /
    ``--cross-schema PATH``. Raises :class:`ValueError` on parse failures
    or non-object top-level values.
    """
    p = Path(path)
    text = p.read_text()
    try:
        obj = json5.loads(text)
    except Exception as exc:  # noqa: BLE001 — json5 raises a variety
        raise ValueError(f"failed to parse JSONC schema at {p}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"top-level value in {p} must be an object, got {type(obj).__name__}")
    return obj


def load_rebel_schema(cfg: ExpConfig | None = None) -> dict[str, Any]:
    """Load the REBEL OA schema from `data/rebel/schemas/rebel.jsonc`."""
    cfg = cfg or ExpConfig()
    return load_schema_from_path(cfg.rebel_schema_path)


def load_scierc_schema(
    cfg: ExpConfig | None = None,
    *,
    kind: str = "default",
) -> dict[str, Any]:
    """Load a SciERC OA schema.

    ``kind``:

      * ``"default"`` — full referential alignment (idScope/ref).
      * ``"strict"``  — RA off; ``id``/``subject``/``object`` are scored
                        as literal strings. Same wire shape as ``default``.
    """
    cfg = cfg or ExpConfig()
    if kind == "default":
        path = cfg.scierc_schema_path
    elif kind == "strict":
        path = cfg.scierc_strict_schema_path
    else:
        raise ValueError(
            f"unknown scierc schema kind {kind!r}; "
            "expected 'default' or 'strict'"
        )
    return load_schema_from_path(path)


def load_recipeflow_schema(
    cfg: ExpConfig | None = None,
    *,
    kind: str = "default",
) -> dict[str, Any]:
    """Load a Recipe Flow OA schema.

    ``kind`` parallels :func:`load_scierc_schema`:

      * ``"default"`` — full referential alignment.
      * ``"strict"``  — RA off; literal id compare.
    """
    cfg = cfg or ExpConfig()
    if kind == "default":
        path = cfg.recipeflow_schema_path
    elif kind == "strict":
        path = cfg.recipeflow_strict_schema_path
    else:
        raise ValueError(
            f"unknown recipeflow schema kind {kind!r}; "
            "expected 'default' or 'strict'"
        )
    return load_schema_from_path(path)


def load_cro_schema(
    cfg: ExpConfig | None = None,
    *,
    kind: str = "default",
) -> dict[str, Any]:
    """Load a CRO OA schema.

    ``kind`` parallels :func:`load_scierc_schema`:

      * ``"default"`` — full referential alignment.
      * ``"strict"``  — RA off; literal id compare.
    """
    cfg = cfg or ExpConfig()
    if kind == "default":
        path = cfg.cro_schema_path
    elif kind == "strict":
        path = cfg.cro_strict_schema_path
    else:
        raise ValueError(
            f"unknown cro schema kind {kind!r}; "
            "expected 'default' or 'strict'"
        )
    return load_schema_from_path(path)


def load_graphimg_schema(
    cfg: ExpConfig | None = None,
    *,
    kind: str = "ra",
) -> dict[str, Any]:
    """Load a graphimg OA schema. ``kind`` is ``"ra"`` (default) or ``"strict"``."""
    cfg = cfg or ExpConfig()
    if kind == "ra":
        path = cfg.graphimg_ra_schema_path
    elif kind == "strict":
        path = cfg.graphimg_strict_schema_path
    else:
        raise ValueError(f"unknown graphimg schema kind {kind!r}; expected 'ra' or 'strict'")
    return load_schema_from_path(path)


def load_amr_schema(
    cfg: ExpConfig | None = None,
    *,
    kind: str = "ra",
) -> dict[str, Any]:
    """Load an AMR OA schema. ``kind`` is ``"ra"`` (default) or ``"strict"``."""
    cfg = cfg or ExpConfig()
    if kind == "ra":
        path = cfg.amr_ra_schema_path
    elif kind == "strict":
        path = cfg.amr_strict_schema_path
    else:
        raise ValueError(f"unknown amr schema kind {kind!r}; expected 'ra' or 'strict'")
    return load_schema_from_path(path)


def load_synra_basic_schema(
    cfg: ExpConfig | None = None,
    *,
    kind: str = "default",
) -> dict[str, Any]:
    """Load a synra_basic OA schema.

    ``kind``:

      * ``"default"`` — full referential alignment (idScope/ref).
      * ``"strict"``  — RA off; ``id``/``source``/``target`` are
                        scored as literal strings. Same wire shape.
    """
    cfg = cfg or ExpConfig()
    if kind == "default":
        path = cfg.synra_basic_schema_path
    elif kind == "strict":
        path = cfg.synra_basic_strict_schema_path
    else:
        raise ValueError(
            f"unknown synra_basic schema kind {kind!r}; expected 'default' or 'strict'"
        )
    return load_schema_from_path(path)


def load_synra_codex_schema(
    cfg: ExpConfig | None = None,
    *,
    kind: str = "ra",
) -> dict[str, Any]:
    """Load a synra_codex OA schema.

    ``kind``:

      * ``"ra"``     — full referential alignment (``idScope``/``ref``); the
                       RA-on schema. Pair with ``id_disambiguation`` on the
                       :class:`ObjectAligner` constructor to toggle WL.
      * ``"strict"`` — RA off; ids/refs scored as literal strings. Same wire
                       shape as ``ra``.
    """
    cfg = cfg or ExpConfig()
    if kind == "ra":
        path = cfg.synra_codex_ra_schema_path
    elif kind == "strict":
        path = cfg.synra_codex_strict_schema_path
    else:
        raise ValueError(
            f"unknown synra_codex schema kind {kind!r}; expected 'ra' or 'strict'"
        )
    return load_schema_from_path(path)
