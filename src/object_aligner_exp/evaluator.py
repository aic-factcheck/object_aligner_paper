"""GEPA-compatible `Evaluator`s that use Object Aligner as the reward signal.

Four flavours, one per GEPA "arm" family we run:

- :class:`OaScoreOnlyEvaluator` — Arm 1 (``oa_score``). Score is the OA
  score; feedback is just the bare number, so the reflection LM sees no
  prescriptive content.

- :class:`OaFeedbackEvaluator` — Arm 2 (``oa_feedback``). Score and
  feedback both come from ``ObjectAligner.feedback()``; the feedback string
  is OA's prescriptive list of fix locations (``referential_feedback="literal"``).

- :class:`OaSemanticFeedbackEvaluator` — Arm 4 (``oa_feedback_ra``). Identical
  to :class:`OaFeedbackEvaluator` but renders ``ref`` / ``idScope`` mismatches
  with ``referential_feedback="semantic"`` — the gold endpoint is described by
  its discriminative properties and relation label rather than an opaque id.

- :class:`OaFeedbackTopK1Evaluator` / :class:`OaFeedbackTopK10Evaluator` /
  :class:`OaFeedbackTopKAllEvaluator` — Arms (``oa_feedback1`` / ``oa_feedback10``
  / ``oa_feedback_all``). Identical to :class:`OaFeedbackEvaluator` (literal ref
  feedback, same score) but the feedback text is capped at ``top_k`` = 1 / 10 /
  unlimited ranked corrections instead of OA's default 5. Isolates the marginal
  value of correction breadth.

- :class:`OaNoneEvaluator` — Arm 3 (``oa_none``). Score is the OA score
  (still drives GEPA selection); feedback is a fixed placeholder so the
  reflection LM sees no per-sample signal at all.

- :class:`OaGoldEvaluator` — Arm (``oa_gold``). Score is the OA score (same as
  ``oa_score``); feedback is the bare score line plus the full serialized gold
  output. A deliberately naive "show the answer" baseline: the GEPA paper's
  ``μ_f`` is always diagnostic and never dumps the ground truth, so this is the
  untested upper-information control for structured outputs.

- :class:`OaFeedbackNoStructEvaluator` — Arm (``oa_feedback_nostruct``).
  Identical to :class:`OaFeedbackEvaluator` (literal ref feedback) but the
  feedback *text* has all *structural* (root / relation / node-id) fix items
  removed, leaving only node-content fixes; the score is unchanged. This is the
  ablation that removes ``oa_feedback``'s structural-text "leak" — see
  ``research/opus48_amr_bio_analysis.md`` for why. **AMR-specific** (the
  structural predicate assumes the ``{root, nodes, relations}`` wire shape).

Each evaluator takes a ``wl_integration`` (``"tie_break"`` | ``"blend"``) knob
forwarded to OA. ``"tie_break"`` (default) is byte-identical to today; the
``_blend`` arm variants (``oa_score_blend`` / ``oa_feedback_blend`` /
``oa_feedback_ra_blend``) pass ``"blend"``, which changes the score by mixing
property cost with WL structural agreement. The run script selects the mode
from the arm name via :func:`object_aligner_exp.oa.wl_integration_for_arm`.

Each `DefaultDataInst` we feed GEPA must carry the gold graph; we stash it on
the `additional_context` field as a JSON string (`make_data_inst` builds it).
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from gepa.adapters.default_adapter.default_adapter import (
    DefaultDataInst,
    EvaluationResult,
)

from object_aligner_exp.llm import _ConversationLogger, _tqdm_safe_print
from object_aligner_exp.oa import _is_structural_amr, score_and_feedback


def _emit_score_endline(
    lm_logger: _ConversationLogger | None,
    response: str,
    score: float,
) -> None:
    """Emit the paired ``[task_lm #NNNN] ... score=...`` end line.

    Pops the per-call entry stashed by :class:`_ConversationLogger`. If no
    matching entry is present (e.g. the LM wrapper had ``print_progress``
    off, or scoring is being driven outside the adapter), emits a short
    score-only line so the evaluation still leaves a trace.
    """
    if lm_logger is None:
        return
    info = lm_logger.pop_pending(response)
    if info is None:
        _tqdm_safe_print(f"[{lm_logger.label}] score={score:.3f}")
        return
    idx, t0, sample_id, latency, in_chars, out_chars = info
    elapsed = time.time() - t0
    _tqdm_safe_print(
        f"[{lm_logger.label} #{idx:04d}] sample={sample_id if sample_id is not None else '?'}  "
        f"lm={latency:6.2f}s  total={elapsed:6.2f}s  "
        f"in={in_chars:>6d} chars  out={out_chars:>6d} chars  "
        f"score={score:.3f}"
    )


# --- shared core -----------------------------------------------------------


class _ParseError(ValueError):
    pass


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise _ParseError("empty response")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ParseError(
            f"JSONDecodeError: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        ) from exc
    except ValueError as exc:
        # json.loads can raise non-JSONDecodeError ValueErrors — notably
        # CPython's integer-string-conversion guard ("Exceeds the limit (4300
        # digits) for integer string conversion") when a model emits a
        # pathologically long integer literal. Treat as an unparseable
        # prediction (scored 0 by _score_one) rather than letting it escape and
        # crash gepa.optimize(). Mirrors holdout_eval.parse_response, which
        # already catches (JSONDecodeError, ValueError).
        raise _ParseError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(obj, dict):
        raise _ParseError(f"top-level value must be an object, got {type(obj).__name__}")
    return obj


def _extract_gold(data: DefaultDataInst) -> dict[str, Any]:
    """Recover the gold graph from a DataInst built by `make_data_inst`."""
    ctx = data.get("additional_context") or {}
    gold_json = ctx.get("gold_json")
    if gold_json is None:
        raise RuntimeError("DataInst is missing additional_context['gold_json']")
    return json.loads(gold_json)


def _score_one(
    data: DefaultDataInst,
    response: str,
    schema: dict[str, Any],
    *,
    referential_feedback: str = "literal",
    wl_integration: str = "tie_break",
    structural_filter: Callable[[str], bool] | None = None,
    top_k: int | None = 5,
) -> tuple[float, str | None, str | None]:
    """Try to parse `response` and score it against the gold for `data`.

    Returns ``(score, oa_feedback, error_msg)``:

    - On success: ``(score, oa_feedback, None)``.
    - On JSON parse failure: ``(0.0, None, "<diagnostic>")``.
    - On schema/OA failure:  ``(0.0, None, "<diagnostic>")``.
    """
    gold = _extract_gold(data)

    try:
        pred = _parse_json(response)
    except _ParseError as exc:
        return 0.0, None, (
            "Your output is not valid JSON conforming to the required schema. "
            f"Parse error: {exc}. "
            "The output MUST be a single JSON object with two top-level keys: "
            '"entities" (a list of {"id", "name"}) and "relations" (a list of '
            '{"subject", "predicate", "object"} where subject/object are entity ids).'
        )

    try:
        score, fb_text = score_and_feedback(
            gold, pred, schema,
            referential_feedback=referential_feedback,
            wl_integration=wl_integration,
            structural_filter=structural_filter,
            top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001 — OA raises various types
        return 0.0, None, (
            "Your JSON parsed but did not fit the schema. Object Aligner error: "
            f"{type(exc).__name__}: {exc}. "
            'Required shape: {"entities": [{"id": str, "name": str}, ...], '
            '"relations": [{"subject": <entity-id>, "predicate": str, '
            '"object": <entity-id>}, ...]}.'
        )

    return score, fb_text, None


# --- arm 1: scalar-only ----------------------------------------------------


class OaScoreOnlyEvaluator:
    """Arm 1 (``oa_score``): GEPA sees the OA score and nothing else.

    Feedback string is just ``"score: 0.42/1.0"`` — no prescriptive text.
    The intent is to test how much GEPA's reflection LM can do with the
    scalar reward alone.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        lm_logger: _ConversationLogger | None = None,
        wl_integration: str = "tie_break",
    ):
        self.schema = schema
        self._lm_logger = lm_logger
        self._wl_integration = wl_integration

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        score, _oa_fb, _err = _score_one(
            data, response, self.schema, wl_integration=self._wl_integration
        )
        _emit_score_endline(self._lm_logger, response, score)
        return EvaluationResult(
            score=score,
            feedback=f"score: {score:.3f}/1.0",
            objective_scores=None,
        )


# --- arm: score + gold (naive upper-information baseline) ------------------


class OaGoldEvaluator:
    """Arm (``oa_gold``): score plus the full gold output, nothing else.

    Score is the OA score (identical to :class:`OaScoreOnlyEvaluator`); the
    feedback string is the bare score line followed by the serialized gold
    graph. This is the cleanest comparator to ``oa_score`` — the *only* added
    variable is the raw gold text — so it isolates how much simply *showing the
    answer* helps, against OA's diagnostic feedback arms (``oa_feedback`` /
    ``oa_feedback_ra``).

    It is a deliberately naive baseline. The GEPA paper's feedback function
    ``μ_f`` is always *diagnostic* and never serializes the ground truth; only
    GEPA's trivial exact-match example adapters inline the gold answer. This arm
    is the untested "dump the answer" control for structured outputs — expected
    to help on small answers but be non-actionable / overfit on large graphs.

    Mirrors ``oa_score``'s deliberate withholding: on parse/schema failure the
    feedback is still just ``score: 0.000/1.0`` + gold, *not* the JSON-validity
    diagnostic, so the sole delta vs. ``oa_score`` remains the gold text.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        lm_logger: _ConversationLogger | None = None,
        wl_integration: str = "tie_break",
    ):
        self.schema = schema
        self._lm_logger = lm_logger
        self._wl_integration = wl_integration

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        score, _oa_fb, _err = _score_one(
            data, response, self.schema, wl_integration=self._wl_integration
        )
        gold = _extract_gold(data)
        gold_pretty = json.dumps(gold, indent=2, ensure_ascii=False)
        _emit_score_endline(self._lm_logger, response, score)
        return EvaluationResult(
            score=score,
            feedback=f"score: {score:.3f}/1.0\n\nReference (gold) output:\n{gold_pretty}",
            objective_scores=None,
        )


# --- arm 2: OA prescriptive feedback ---------------------------------------


class OaFeedbackEvaluator:
    """Arm 2 (``oa_feedback``): score AND feedback both come from OA.

    Feedback is OA's prescriptive list of fix locations. On parse/schema
    failure, feedback is a diagnostic that tells the model to emit valid JSON.

    ``referential_feedback`` (``"literal"`` | ``"semantic"``) is forwarded to
    OA; it controls only how ``ref`` / ``idScope`` mismatches are phrased.

    ``top_k`` caps how many ranked corrections the feedback text shows (a
    positive int, or ``None`` for all). Default ``5`` matches OA's own
    ``feedback()`` default; the ``oa_feedback{1,10,all}`` subclasses override
    it. Affects only the feedback text, never the score.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        lm_logger: _ConversationLogger | None = None,
        referential_feedback: str = "literal",
        wl_integration: str = "tie_break",
        structural_filter: Callable[[str], bool] | None = None,
        top_k: int | None = 5,
    ):
        self.schema = schema
        self._lm_logger = lm_logger
        self._referential_feedback = referential_feedback
        self._wl_integration = wl_integration
        self._structural_filter = structural_filter
        self._top_k = top_k

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        score, oa_fb, err = _score_one(
            data, response, self.schema,
            referential_feedback=self._referential_feedback,
            wl_integration=self._wl_integration,
            structural_filter=self._structural_filter,
            top_k=self._top_k,
        )
        feedback = oa_fb if err is None else err
        assert feedback is not None  # exactly one of (oa_fb, err) is set
        _emit_score_endline(self._lm_logger, response, score)
        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)


class OaSemanticFeedbackEvaluator(OaFeedbackEvaluator):
    """Arm 4 (``oa_feedback_ra``): like ``oa_feedback`` but semantic ref feedback.

    Identical to :class:`OaFeedbackEvaluator` except ``ref`` / ``idScope``
    mismatches are rendered with ``referential_feedback="semantic"`` — the gold
    endpoint is described by its discriminative properties and relation label
    rather than an opaque id, a more transferable lesson for prompt optimizers.
    Scores are unchanged; only the feedback text differs.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        lm_logger: _ConversationLogger | None = None,
        wl_integration: str = "tie_break",
    ):
        super().__init__(
            schema,
            lm_logger=lm_logger,
            referential_feedback="semantic",
            wl_integration=wl_integration,
        )


class _OaFeedbackTopKEvaluator(OaFeedbackEvaluator):
    """Base for the ``oa_feedback{1,10,all}`` arms: ``oa_feedback`` at a fixed ``top_k``.

    Identical to :class:`OaFeedbackEvaluator` (literal ref feedback, same score)
    except the feedback text is capped at a different number of ranked
    corrections. Subclasses set ``_TOP_K`` (a positive int, or ``None`` for all);
    the score is OA's full score, unchanged, so this manipulates *only* how much
    of OA's prescriptive critique the reflection LM reads.

    Isolates the marginal value of correction *breadth*: ``oa_feedback`` shows
    OA's default top-5, so these arms bracket it with 1 (single most-impactful
    fix), 10, and unlimited.
    """

    _TOP_K: int | None = 5

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        lm_logger: _ConversationLogger | None = None,
        wl_integration: str = "tie_break",
    ):
        super().__init__(
            schema,
            lm_logger=lm_logger,
            referential_feedback="literal",
            wl_integration=wl_integration,
            top_k=self._TOP_K,
        )


class OaFeedbackTopK1Evaluator(_OaFeedbackTopKEvaluator):
    """Arm ``oa_feedback1``: ``oa_feedback`` showing only the single top correction."""

    _TOP_K = 1


class OaFeedbackTopK10Evaluator(_OaFeedbackTopKEvaluator):
    """Arm ``oa_feedback10``: ``oa_feedback`` showing up to 10 corrections."""

    _TOP_K = 10


class OaFeedbackTopKAllEvaluator(_OaFeedbackTopKEvaluator):
    """Arm ``oa_feedback_all``: ``oa_feedback`` showing every correction (no cap)."""

    _TOP_K = None


class OaFeedbackNoStructEvaluator(OaFeedbackEvaluator):
    """Arm ``oa_feedback_nostruct``: ``oa_feedback`` minus structural feedback.

    Identical to :class:`OaFeedbackEvaluator` (literal ref feedback, same score)
    except the feedback *text* drops every structural fix item — root pointer,
    relation edges, and node-id markers — via
    :func:`object_aligner_exp.oa._is_structural_amr`, leaving only node-content
    fixes (concepts, names, attribute values). The score is OA's full score,
    unchanged, so this manipulates *only* what the reflection LM reads.

    The ablation isolates ``oa_feedback``'s structural-text "leak": under the
    strict schema the score can't reward relations, yet the feedback text still
    describes them, which is what lets a strict-trained run transfer to the RA
    holdout (``research/opus48_amr_bio_analysis.md``). Stripping that text should
    collapse the transfer back toward ``oa_score`` if the leak hypothesis holds.

    **AMR-specific** — the structural predicate assumes the ``{root, nodes,
    relations}`` wire shape; do not reuse for other datasets unchanged.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        lm_logger: _ConversationLogger | None = None,
        wl_integration: str = "tie_break",
    ):
        super().__init__(
            schema,
            lm_logger=lm_logger,
            referential_feedback="literal",
            wl_integration=wl_integration,
            structural_filter=_is_structural_amr,
        )


# --- arm 3: no feedback ----------------------------------------------------


class OaNoneEvaluator:
    """Arm 3 (``oa_none``): OA still supplies the score, but the textual
    channel to the reflection LM carries no per-sample signal.

    Feedback is a fixed placeholder string. The intent is to isolate the
    value of having *any* textual feedback at all (cf. ``oa_score``, which
    at least surfaces the bare score in the feedback string).
    """

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        lm_logger: _ConversationLogger | None = None,
        wl_integration: str = "tie_break",
    ):
        self.schema = schema
        self._lm_logger = lm_logger
        self._wl_integration = wl_integration

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        score, _oa_fb, _err = _score_one(
            data, response, self.schema, wl_integration=self._wl_integration
        )
        _emit_score_endline(self._lm_logger, response, score)
        return EvaluationResult(
            score=score,
            feedback="No feedback for this sample",
            objective_scores=None,
        )


# --- DataInst builder ------------------------------------------------------


def make_data_inst(
    *,
    context: str,
    gold: dict[str, Any],
    sample_id: Any | None = None,
) -> DefaultDataInst:
    """Build a `DefaultDataInst` that carries the gold graph alongside the input.

    ``sample_id`` (any value coerceable to ``str``) is stashed on
    ``additional_context`` so the adapter can surface it on per-call
    progress lines.
    """
    ctx: dict[str, Any] = {"gold_json": json.dumps(gold, ensure_ascii=False)}
    if sample_id is not None:
        ctx["sample_id"] = sample_id
    return DefaultDataInst(
        input=context,
        additional_context=ctx,
        answer="",  # unused — our evaluators do their own scoring
    )


def make_image_data_inst(
    *,
    image_path: Any,
    instruction: str,
    gold: dict[str, Any],
    sample_id: Any | None = None,
) -> DefaultDataInst:
    """Build a `DefaultDataInst` for an image-input task.

    ``image_path`` (anything ``str()``-coerceable, typically a ``pathlib.Path``)
    is stashed on ``additional_context`` so the image-aware adapter can find
    it when constructing the user message and (optionally) when assembling
    the reflective dataset. ``instruction`` is the textual prompt that
    accompanies the image in the user message. ``sample_id`` is stashed
    for the adapter's per-call progress line.
    """
    ctx: dict[str, Any] = {
        "gold_json": json.dumps(gold, ensure_ascii=False),
        "image_path": str(image_path),
    }
    if sample_id is not None:
        ctx["sample_id"] = sample_id
    return DefaultDataInst(
        input=instruction,
        additional_context=ctx,
        answer="",
    )
