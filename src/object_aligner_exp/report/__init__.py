"""Static-HTML report generator for OA experiment runs.

Walks a run directory under ``data/runs/<run>_<ts>/``, pairs every stored
prediction with its gold, runs OA's (LM-free) alignment to derive a
re-ordered "aligned" view, and emits a single self-contained
``report.html`` next to the inputs.
"""

from object_aligner_exp.report.callback import RenderOnStateSaved
from object_aligner_exp.report.render import render_run

__all__ = ["RenderOnStateSaved", "render_run"]
