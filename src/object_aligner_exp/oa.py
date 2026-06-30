"""Thin wrapper over Object Aligner that returns (score, feedback) in one call."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from object_aligner import FeedbackResult, ObjectAligner, render_feedback


def _is_structural_amr(path: str) -> bool:
    """Is this OA feedback fix-location *structural* under the AMR wire shape?

    "Structure" = OA's referential machinery, as instantiated for AMR's
    ``{root, nodes, relations}`` shape:

      * ``"/root"``                       — the TOP pointer (a ``ref``).
      * ``"/relations"`` / ``"/relations/<i>"``        — a whole edge (add/remove), and
        ``"/relations/<i>/source|role|target"``        — its endpoints / label.
      * ``"/nodes/<i>/id"``               — a node's id marker (an ``idScope``).
        Under the strict ablation the id is a plain string, so a wrong id
        surfaces as an unfollowable ``primitive_replace`` ("expected 's'") — pure
        noise the model can never act on, so we drop it too.

    Node *content* is kept: a whole-node add/remove (``"/nodes"`` / ``"/nodes/<i>"``),
    its ``"/nodes/<i>/concept"``, and ``"/nodes/<i>/attributes..."``. These are the
    schema-invariant lessons (concepts, names, attribute values) we want to keep.

    Created for the ``oa_feedback_nostruct`` ablation — see
    ``research/opus48_amr_bio_analysis.md`` ("Follow-up ablation" section): it
    tests whether ``oa_feedback``'s structural *text* (not the score) is what lets
    a strict-trained GEPA run transfer to the RA holdout.
    """
    if path == "/root":
        return True
    if path == "/relations" or path.startswith("/relations/"):
        return True
    parts = path.split("/")  # "/nodes/<i>/id" -> ["", "nodes", "<i>", "id"]
    if len(parts) == 4 and parts[1] == "nodes" and parts[3] == "id":
        return True
    return False


def nostruct_feedback(
    aligner: ObjectAligner,
    gold: dict[str, Any],
    pred: dict[str, Any],
    *,
    referential_feedback: str = "literal",
    top_k: int | None = 5,
    is_structural: Callable[[str], bool] = _is_structural_amr,
) -> FeedbackResult:
    """Like ``aligner.feedback(gold, pred)`` but with *structural* fixes removed.

    Drops every repair op whose JSON-pointer ``path`` satisfies ``is_structural``
    (root / relations / node-id for AMR), then re-renders the remaining
    node-content ops with OA's own renderer — so the "Top K of N" header and the
    synthesis line stay faithful and the ``top_k`` budget is re-applied over the
    node-only ops (no starvation). The *score* is OA's full score, unchanged.

    Uses only public OA API: ``aligner.repair`` + ``render_feedback`` +
    ``dataclasses.replace`` on the (frozen) ``RepairResult``. Assumes the aligner
    uses OA's default feedback templates/style (true for every aligner built in
    this repo), so the re-rendered text matches what ``aligner.feedback`` emits.
    """
    rr = aligner.repair(gold, pred)
    kept = tuple(op for op in rr.ops if not is_structural(op.path))
    if len(kept) == len(rr.ops):
        # Nothing structural to drop (e.g. a validation-failure empty result, or a
        # pred with no structural fixes). Defer to the normal renderer so the text
        # and any validation message stay byte-identical to oa_feedback.
        return aligner.feedback(
            gold, pred, top_k=top_k, referential_feedback=referential_feedback
        )
    return render_feedback(
        dataclasses.replace(rr, ops=kept),
        top_k=top_k,
        style="gepa",
        referential_feedback=referential_feedback,
    )


def score_and_feedback(
    gold: dict[str, Any],
    pred: dict[str, Any],
    schema: dict[str, Any],
    *,
    referential_feedback: str = "literal",
    wl_integration: str = "tie_break",
    structural_filter: Callable[[str], bool] | None = None,
    top_k: int | None = 5,
) -> tuple[float, str]:
    """Score `pred` against `gold` under `schema`, returning (score, feedback).

    ``referential_feedback`` (``"literal"`` | ``"semantic"``) selects how OA
    renders ``ref`` / ``idScope`` mismatches. ``"literal"`` (default) uses
    opaque ids and is byte-identical to earlier releases; ``"semantic"``
    describes the gold endpoint by its discriminative properties and the
    relation label. Only the feedback text changes — the score is identical.

    ``wl_integration`` (``"tie_break"`` | ``"blend"``) selects how OA's
    Weisfeiler–Leman structural color enters the ``idScope`` cost matrix.
    ``"tie_break"`` (default) breaks only *exact* property-cost ties and is
    byte-identical to ``ObjectAligner(schema)``; ``"blend"`` mixes property
    cost and structural agreement (λ = OA's default 0.5), so it can change the
    score itself. Inert on schemas without ``idScope`` / ``ref`` (e.g. the
    strict ablation), where WL has no ref graph to refine.

    ``structural_filter`` (the ``oa_feedback_nostruct`` arm) drops structural
    fix items from the feedback *text* via :func:`nostruct_feedback`; the score
    is unaffected. ``None`` (default) leaves the feedback unfiltered.

    ``top_k`` (the ``oa_feedback{1,10,all}`` arms) caps how many ranked
    corrections the feedback text shows: a positive int shows that many,
    ``None`` shows all. Default ``5`` matches OA's own ``feedback()`` default,
    so it is byte-identical to earlier releases. Affects only the feedback
    text, never the score.
    """
    aligner = ObjectAligner(schema, wl_integration=wl_integration)
    if structural_filter is not None:
        fb = nostruct_feedback(
            aligner, gold, pred,
            referential_feedback=referential_feedback,
            is_structural=structural_filter,
            top_k=top_k,
        )
    else:
        fb = aligner.feedback(
            gold, pred, referential_feedback=referential_feedback, top_k=top_k
        )
    return float(fb.score), fb.text


def referential_feedback_for_arm(arm: str | None) -> str:
    """OA ``referential_feedback`` mode implied by a GEPA ``--arm``.

    Single source of truth shared by the run-time evaluators
    (:class:`object_aligner_exp.evaluator.OaSemanticFeedbackEvaluator`) and the
    report renderer, so regenerated feedback in ``report.html`` matches what the
    reflection LM actually saw. Only the ``oa_feedback_ra`` family (including
    its ``_blend`` variant) uses ``"semantic"``; every other arm uses the
    default ``"literal"``.
    """
    base = arm[: -len("_blend")] if arm and arm.endswith("_blend") else arm
    return "semantic" if base == "oa_feedback_ra" else "literal"


def wl_integration_for_arm(arm: str | None) -> str:
    """OA ``wl_integration`` mode implied by a GEPA ``--arm``.

    Companion to :func:`referential_feedback_for_arm`: the ``_blend`` arm
    variants (``oa_score_blend`` / ``oa_feedback_blend`` /
    ``oa_feedback_ra_blend``) score with WL ``"blend"``; every other arm uses
    the default ``"tie_break"``. The arm name is the single source of truth, so
    ``config.json`` (which records ``arm``) already pins the WL mode for resume
    and report regeneration — no extra config field is needed.
    """
    return "blend" if arm and arm.endswith("_blend") else "tie_break"


def top_k_for_arm(arm: str | None) -> int | None:
    """OA feedback ``top_k`` (max corrections shown) implied by a GEPA ``--arm``.

    Companion to :func:`referential_feedback_for_arm` / :func:`wl_integration_for_arm`
    / :func:`structural_filter_for_arm`: the single source of truth shared by the
    run-time evaluator (:class:`object_aligner_exp.evaluator.OaFeedbackEvaluator`
    subclasses) and the report renderer, so regenerated feedback in ``report.html``
    matches what the reflection LM actually saw.

    The ``oa_feedback{1,10,all}`` family caps the ranked correction list at
    ``1`` / ``10`` / unlimited (``None``); every other arm uses OA's default of
    ``5``. ``top_k`` changes only the feedback *text*, never the score, so it
    composes with WL ``blend`` — the ``_blend`` suffix is stripped before lookup.
    """
    base = arm[: -len("_blend")] if arm and arm.endswith("_blend") else arm
    return {"oa_feedback1": 1, "oa_feedback10": 10, "oa_feedback_all": None}.get(
        base, 5
    )


def structural_filter_for_arm(arm: str | None) -> Callable[[str], bool] | None:
    """Path predicate marking 'structural' feedback to drop for a GEPA ``--arm``.

    Companion to :func:`referential_feedback_for_arm` / :func:`wl_integration_for_arm`:
    the single source of truth shared by the run-time evaluator
    (:class:`object_aligner_exp.evaluator.OaFeedbackNoStructEvaluator`) and the
    report renderer, so regenerated feedback in ``report.html`` matches what the
    reflection LM actually saw.

    Only the ``oa_feedback_nostruct`` family filters (returns
    :func:`_is_structural_amr`); every other arm returns ``None`` (no filtering).
    The predicate is **AMR-specific** — a general cross-dataset version would need
    OA support and is intentionally not implemented here (see
    ``research/opus48_amr_bio_analysis.md``).
    """
    base = arm[: -len("_blend")] if arm and arm.endswith("_blend") else arm
    return _is_structural_amr if base == "oa_feedback_nostruct" else None
