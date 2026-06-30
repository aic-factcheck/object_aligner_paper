"""Seed prompt loading.

Seed prompts (the system prompt GEPA starts evolving from, and the verbatim
system prompt a baseline arm uses) live as plain text under
``data/<dataset>/seed_prompt.txt`` — graphimg keeps a per-version file at
``data/graphimg/v{1,2}/seed_prompt.txt``. Keeping them as files lets a prompt
be edited without touching Python and selected per run via ``--seed-prompt``.

Every entry point resolves a default path from :class:`ExpConfig`
(``cfg.<dataset>_seed_prompt_path``) or an explicit ``--seed-prompt PATH`` and
loads it through :func:`load_seed_prompt_from_path`. The canonical paths are
exposed on :class:`object_aligner_exp.config.ExpConfig`, mirroring how schemas
are handled in :mod:`object_aligner_exp.schemas`.
"""

from __future__ import annotations

from pathlib import Path


def load_seed_prompt_from_path(path: Path | str) -> str:
    """Load a seed system prompt from a UTF-8 text file.

    Surrounding whitespace is stripped. Raises :class:`ValueError` if the file
    is missing or empty.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"seed prompt not found at {p}") from exc
    text = text.strip()
    if not text:
        raise ValueError(f"seed prompt at {p} is empty")
    return text
