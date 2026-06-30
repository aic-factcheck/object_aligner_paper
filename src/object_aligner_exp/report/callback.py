"""GEPA callback that re-renders the static run report after each state save.

GEPA's engine atomically rewrites ``candidates.json``, ``run_log.json``, and
the ``generated_best_outputs_valset/`` tree inside ``state.save(...)``, then
fires ``on_state_saved``. Hooking there guarantees the run dir is fresh on
disk when we rebuild the HTML — analogous to how GEPA itself regenerates
``candidate_tree.html`` after every accepted candidate.

Render failures are caught so a bug in the report code never aborts the
optimization. The first save happens before any candidate has been added
(seed only); the loader tolerates the missing ``summary.json`` so that
first render still succeeds.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

from object_aligner_exp.report.render import render_run


class RenderOnStateSaved:
    """GEPA ``GEPACallback`` (duck-typed) that re-renders the static report.

    Args:
        run_dir: directory passed to ``gepa.optimize(run_dir=...)``.
        verbose: if True, prints one-line status to stderr on each render.
    """

    def __init__(self, run_dir: Path | str, *, verbose: bool = False) -> None:
        self._run_dir = Path(run_dir)
        self._verbose = verbose
        self._calls = 0
        self._failures = 0

    def on_state_saved(self, event: dict[str, Any]) -> None:
        self._calls += 1
        try:
            out = render_run(self._run_dir, verbose=False)
            if self._verbose:
                print(
                    f"[render] iter={event.get('iteration')} -> {out}",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001 — never crash the optimization
            self._failures += 1
            # Print the first few failures with a traceback; after that, just a
            # one-liner so a persistent bug doesn't flood stderr.
            if self._failures <= 3:
                print(
                    f"[render] failed at iter={event.get('iteration')}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
            else:
                print(
                    f"[render] failed at iter={event.get('iteration')}: {exc!r} "
                    f"(failure #{self._failures})",
                    file=sys.stderr,
                )
