"""GEPA ``DefaultAdapter`` subclass that fans out task_lm calls in parallel.

GEPA's :class:`gepa.adapters.default_adapter.default_adapter.DefaultAdapter`
calls a Python-callable task_lm sequentially — once per item in each
evaluation batch — because the ``self._lm.batch_complete`` fast path is only
taken when ``model`` is a LiteLLM model-name string.

For local OpenAI-compatible servers it's often fine (and much faster) to
issue several task_lm requests at once. :class:`ParallelDefaultAdapter`
exposes a ``max_workers`` knob that controls that fan-out:

  * ``1`` (default) — sequential; behavior is identical to ``DefaultAdapter``.
  * ``N > 1``      — up to ``N`` concurrent task_lm calls per batch.
  * ``-1``         — one worker per batch item (full fan-out).

The implementation swaps the adapter's internal ``_lm`` slot for a tiny
wrapper that exposes ``batch_complete(messages_list, max_workers, **kwargs)``
backed by a :class:`concurrent.futures.ThreadPoolExecutor`. That way we reuse
``DefaultAdapter.evaluate`` verbatim and pick up any future changes to it.
"""

from __future__ import annotations

import concurrent.futures
import time
from statistics import mean
from typing import Any, cast

from gepa.adapters.default_adapter.default_adapter import (
    ChatCompletionCallable,
    ChatMessage,
    DefaultAdapter,
    DefaultDataInst,
    DefaultRolloutOutput,
    DefaultTrajectory,
    Evaluator,
)
from gepa.core.adapter import EvaluationBatch

from object_aligner_exp.llm import (
    _CURRENT_SAMPLE_ID,
    TaskLMUnreachable,
    _tqdm_safe_print,
)


def _sample_id_of(data: DefaultDataInst) -> Any | None:
    ctx = data.get("additional_context") or {}
    return ctx.get("sample_id") if isinstance(ctx, dict) else None


def _print_batch_summary(scores: list[float], wall: float) -> None:
    n = len(scores)
    if n == 0:
        _tqdm_safe_print(f"[batch] n=0 wall={wall:.2f}s")
        return
    per = wall / n
    _tqdm_safe_print(
        f"[batch] n={n} wall={wall:.2f}s per_sample={per:.2f}s "
        f"score_mean={mean(scores):.3f}"
    )


class _CallableLMWrapper:
    """Adapts a Python callable to the subset of ``gepa.lm.LM`` that
    ``DefaultAdapter.evaluate`` actually uses (``batch_complete``).

    Also exposes :meth:`call_one`, which the adapter's interleaved
    ``evaluate`` uses to run a single LM call with a sample-id contextvar
    in scope.
    """

    def __init__(self, fn: ChatCompletionCallable) -> None:
        self._fn = fn

    def call_one(self, messages: list[dict[str, Any]], sample_id: Any | None) -> str:
        token = _CURRENT_SAMPLE_ID.set(sample_id)
        try:
            return self._fn(messages)
        finally:
            _CURRENT_SAMPLE_ID.reset(token)

    def batch_complete(
        self,
        messages_list: list[list[dict[str, Any]]],
        max_workers: int = 1,
        sample_ids: list[Any] | None = None,
        **_kwargs: Any,
    ) -> list[str]:
        n = len(messages_list)
        if n == 0:
            return []
        if sample_ids is None:
            sample_ids = [None] * n
        if max_workers < 0:
            workers = n
        elif max_workers == 0:
            workers = 1
        else:
            workers = min(max_workers, n)
        if workers <= 1:
            return [self.call_one(m, s) for m, s in zip(messages_list, sample_ids)]

        results: list[str] = [""] * n
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(self.call_one, m, s): i
                for i, (m, s) in enumerate(zip(messages_list, sample_ids))
            }
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()
        return results


def _resolve_workers(max_workers: int, n: int) -> int:
    if max_workers < 0:
        return max(n, 1)
    if max_workers == 0:
        return 1
    return min(max_workers, n) if n > 0 else 1


def _run_per_sample(
    batch: list[DefaultDataInst],
    messages_list: list[list[ChatMessage]],
    sample_ids: list[Any],
    lm: Any | None,
    model: ChatCompletionCallable | Any,
    evaluator: Evaluator,
    workers: int,
    capture_traces: bool,
) -> tuple[
    list[DefaultRolloutOutput],
    list[float],
    list[dict[str, float] | None],
    list[DefaultTrajectory] | None,
]:
    """Run ``LM-call → evaluator(data, response)`` per sample, interleaved.

    With ``workers=1`` this prints lm-start / score / lm-start / score / ...
    With higher concurrency, each worker thread runs LM + eval back-to-back
    so the matching score line follows its task_lm progress lines instead
    of waiting until the whole batch finishes its LM calls.
    """
    n = len(batch)
    outputs: list[DefaultRolloutOutput | None] = [None] * n
    scores: list[float] = [0.0] * n
    objective_scores: list[dict[str, float] | None] = [None] * n
    trajectories: list[DefaultTrajectory | None] | None = (
        [None] * n if capture_traces else None
    )

    def process(i: int) -> None:
        msgs = messages_list[i]
        sid = sample_ids[i]
        try:
            if isinstance(lm, _CallableLMWrapper):
                response = lm.call_one(msgs, sid)
            else:
                # No wrapper available (string-model path or unexpected LM
                # type) — fall back to direct invocation and set the contextvar
                # locally so any inner `_ConversationLogger` still sees the
                # sample id.
                token = _CURRENT_SAMPLE_ID.set(sid)
                try:
                    response = cast(ChatCompletionCallable, model)(msgs)
                finally:
                    _CURRENT_SAMPLE_ID.reset(token)
        except TaskLMUnreachable:
            # Infra is down after the retry schedule — let it escape so
            # adapter.evaluate() raises, gepa.optimize() unwinds, and the
            # runner can exit cleanly for --resume.
            raise
        except Exception as exc:  # noqa: BLE001 — LM transports raise various types
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            _tqdm_safe_print(f"[task_lm] sample={sid if sid is not None else '?'}  LM_ERROR  {err}")
            outputs[i] = {"full_assistant_response": ""}
            scores[i] = 0.0
            objective_scores[i] = None
            if trajectories is not None:
                trajectories[i] = {
                    "data": batch[i],
                    "full_assistant_response": "",
                    "feedback": (
                        f"task LM call failed: {err}. The model produced no "
                        "output for this sample; the system prompt should "
                        "encourage outputs short enough to fit, and the runtime "
                        "may be saturated."
                    ),
                }
            return
        result = evaluator(batch[i], response)
        outputs[i] = {"full_assistant_response": response}
        scores[i] = result.score
        objective_scores[i] = result.objective_scores
        if trajectories is not None:
            trajectories[i] = {
                "data": batch[i],
                "full_assistant_response": response,
                "feedback": result.feedback,
            }

    if workers <= 1 or n <= 1:
        for i in range(n):
            process(i)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process, i) for i in range(n)]
            for fut in concurrent.futures.as_completed(futs):
                # process() catches LM exceptions itself; this is only here so
                # a stray non-LM bug (e.g. evaluator typo) still surfaces.
                fut.result()

    return (
        cast(list[DefaultRolloutOutput], outputs),
        scores,
        objective_scores,
        cast("list[DefaultTrajectory] | None", trajectories),
    )


class ParallelDefaultAdapter(DefaultAdapter):
    """``DefaultAdapter`` that parallelizes Python-callable task_lm calls.

    Parameters
    ----------
    model:
        Either a LiteLLM model-name string (in which case behavior is exactly
        the upstream ``DefaultAdapter`` with ``max_litellm_workers=max_workers``)
        or a Python callable conforming to ``ChatCompletionCallable``.
    evaluator:
        Same as ``DefaultAdapter``.
    max_workers:
        ``1`` = sequential, ``N>1`` = up to N concurrent calls per batch,
        ``-1`` = one worker per batch item.
    """

    def __init__(
        self,
        model: str | ChatCompletionCallable,
        evaluator: Evaluator | None = None,
        max_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        # max_litellm_workers governs the string-model batch_completion path;
        # mirror max_workers there so both paths honor the same flag.
        litellm_workers = max_workers if max_workers > 0 else 10
        super().__init__(
            model,
            evaluator=evaluator,
            max_litellm_workers=litellm_workers,
            **kwargs,
        )
        if not isinstance(model, str):
            # Force DefaultAdapter.evaluate down the batch_complete branch so
            # our wrapper gets to manage concurrency.
            self._lm = _CallableLMWrapper(model)  # type: ignore[assignment]
            self.max_litellm_workers = max_workers

    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput]:
        """Interleaved LM-call + scoring per sample, with a batch summary.

        Mirrors ``DefaultAdapter.evaluate`` but calls the evaluator
        immediately after each LM response instead of batching all LM
        calls first — so each ``[task_lm #NNNN] ... score=...`` end line
        prints right after its start line.
        """
        system_content = next(iter(candidate.values()))
        messages_list: list[list[ChatMessage]] = [
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"{d['input']}"},
            ]
            for d in batch
        ]
        sample_ids = [_sample_id_of(d) for d in batch]

        workers = _resolve_workers(self.max_litellm_workers, len(batch))
        t_batch = time.perf_counter()
        outputs, scores, objective_scores, trajectories = _run_per_sample(
            batch, messages_list, sample_ids,
            self._lm, self.model, self.evaluator, workers, capture_traces,
        )
        _print_batch_summary(scores, time.perf_counter() - t_batch)

        objective_scores_arg: list[dict[str, float]] | None = None
        if objective_scores:
            all_none = all(x is None for x in objective_scores)
            all_not_none = all(x is not None for x in objective_scores)
            if not (all_none or all_not_none):
                raise ValueError("Objective scores must either be all None or all not None.")
            if all_not_none:
                objective_scores_arg = cast(list[dict[str, float]], objective_scores)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores_arg,
        )
