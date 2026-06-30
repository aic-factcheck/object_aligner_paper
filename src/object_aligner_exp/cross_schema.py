"""Identity projection for the cross-schema holdout panel.

Each dataset's schema variants ship in the same wire shape — they only
differ in how OA scores the shared leaf fields:

* graphimg: ``ra`` (RA on; ``idScope``/``ref``) vs ``strict``
  (literal id compare). Same ``{nodes, edges}`` JSON.
* SciERC / recipeflow: ``default`` (RA on) vs ``strict`` (literal id
  compare). Same ``{entities, relations}`` JSON.

Cross-scoring therefore boils down to "call OA again against the same
parsed prediction with another schema". No projection between wire
shapes is required, but
:class:`object_aligner_exp.holdout_eval.CrossSchemaSpec` expects a
``project_pred(pred, *, source_kind, target_kind)`` callable, so we
expose a single dataset-agnostic identity below. Since runners now take
schemas as arbitrary file paths, there is nothing fixed to validate
against; passing a wildly different schema is the caller's problem and
will surface as an OA scoring error.
"""

from __future__ import annotations

from typing import Any


def identity_project(
    pred: dict[str, Any],
    *,
    source_kind: str,
    target_kind: str,
) -> dict[str, Any]:
    """Return ``pred`` unchanged; kind arguments are accepted but ignored.

    Used as ``CrossSchemaSpec.project_pred`` for the in-repo schemas
    where the wire shape is shared across variants of the same dataset.
    """
    del source_kind, target_kind
    return pred


__all__ = [
    "identity_project",
]
