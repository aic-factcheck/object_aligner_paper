"""LM factories.

- `build_task_lm` returns a `ChatCompletionCallable` (the type GEPA's
  `DefaultAdapter` accepts) backed by the OpenAI Python SDK. We bypass LiteLLM
  for the task LM so we can: (a) point at a no-API-key local server cleanly
  and (b) always request `response_format={"type": "json_object"}`.

- `build_reflection_lm` returns a `gepa.lm.LM` (LiteLLM under the hood),
  which conforms to GEPA's `LanguageModel` protocol.
"""

from __future__ import annotations

import contextvars
import json
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tenacity
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from object_aligner_exp.config import LMConfig


_TASK_LM_TRANSIENT_EXC: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class TaskLMUnreachable(RuntimeError):
    """Raised when the task LM remains unreachable after the retry schedule.

    The original transport exception is attached as ``__cause__`` so callers
    can inspect it without importing openai exception types.
    """


# Per-call sample-id channel. The adapter sets this before invoking the
# task_lm callable so `_ConversationLogger` can include the sample id on
# its per-call start line. The variable is read inside whichever worker
# thread is currently servicing the LM call, so each worker must set it
# explicitly (concurrent.futures threads don't inherit context).
_CURRENT_SAMPLE_ID: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "task_lm_current_sample_id", default=None
)

# How many retries the most recent task_lm call consumed before producing the
# response that was returned. Set inside `_call` and read by
# `_ConversationLogger` so each JSONL record can carry it.
_RETRY_ATTEMPTS_USED: contextvars.ContextVar[int] = contextvars.ContextVar(
    "task_lm_retry_attempts_used", default=0
)


def _is_valid_json_object(text: str) -> bool:
    """Return True iff ``text`` (after stripping) parses as a JSON object."""
    text = text.strip()
    if not text:
        return False
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict)


def _tqdm_safe_print(line: str) -> None:
    """Print without breaking an active tqdm progress bar, if one exists."""
    try:
        from tqdm import tqdm as _tqdm

        _tqdm.write(line)
    except Exception:
        print(line)


class _ConversationLogger:
    """Callable wrapper that appends each (messages, response) pair as a JSONL record.

    Forwards unknown attributes to the wrapped callable so GEPA's
    ``hasattr(reflection_lm, "total_cost")`` check at ``api.py:237`` still
    sees the underlying ``gepa.lm.LM`` counters and skips the ``TrackingLM``
    fallback.

    If ``print_progress`` is True, also emits a one-line summary per call via
    ``tqdm.write`` so each evaluation is visible alongside the progress bar.
    """

    def __init__(
        self,
        fn: Any,
        log_path: Path,
        *,
        label: str,
        print_progress: bool = False,
    ):
        self._fn = fn
        self._log_path = log_path
        self._label = label
        self._print_progress = print_progress
        self._lock = threading.Lock()
        self._call_idx = 0
        # `id(response_str) -> (idx, t0, sample_id, latency, in_chars, out_chars)`.
        # Populated when ``print_progress`` is set so the evaluator can pop
        # the entry and emit a paired ``score=`` end line after scoring.
        self._pending: dict[int, tuple[int, float, Any, float, int, int]] = {}
        log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def label(self) -> str:
        return self._label

    def __call__(self, prompt: Any) -> str:
        sample_id = _CURRENT_SAMPLE_ID.get()
        with self._lock:
            self._call_idx += 1
            idx = self._call_idx
        t0 = time.time()
        if self._print_progress:
            start_str = time.strftime("%H:%M:%S", time.localtime(t0))
            _tqdm_safe_print(
                f"[{self._label} #{idx:04d}] start={start_str} "
                f"sample={sample_id if sample_id is not None else '?'}"
            )
        out = self._fn(prompt)
        latency = time.time() - t0
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [dict(m) for m in prompt]
        in_chars = sum(len(m.get("content", "") or "") for m in messages)
        out_chars = len(out) if isinstance(out, str) else 0
        rec = {
            "ts": time.time(),
            "label": self._label,
            "latency_s": latency,
            "retry_attempts": _RETRY_ATTEMPTS_USED.get(),
            "messages": messages,
            "response": out,
        }
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if self._print_progress and isinstance(out, str):
                self._pending[id(out)] = (
                    idx, t0, sample_id, latency, in_chars, out_chars,
                )
        return out

    def pop_pending(
        self, response: str
    ) -> tuple[int, float, Any, float, int, int] | None:
        """Consume the per-call info recorded for ``response``, if any.

        Returns ``(idx, t0, sample_id, latency, in_chars, out_chars)`` so
        the evaluator can emit a paired ``score=`` end line that includes
        both the original LM latency and the total elapsed wall time.
        """
        with self._lock:
            return self._pending.pop(id(response), None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fn, name)


def with_conversation_log(
    fn: Any,
    log_path: Path,
    *,
    label: str,
    print_progress: bool = False,
) -> _ConversationLogger:
    """Wrap an LM callable so every call is appended to ``log_path`` as JSONL.

    If ``print_progress`` is True, also emit one short status line per call to
    stdout (via ``tqdm.write`` when a progress bar is active).
    """
    return _ConversationLogger(fn, log_path, label=label, print_progress=print_progress)


class FilteringLogger:
    """Wrap a GEPA ``LoggerProtocol`` and drop messages containing any pattern.

    GEPA dumps the full proposed prompt via
    ``reflective_mutation.py:390`` ("Iteration N: Proposed new text for ...").
    Passing ``drop_patterns=("Proposed new text",)`` suppresses that line on
    stdout *and* in ``run_log.txt`` while keeping every other status line.
    """

    def __init__(self, base: Any, drop_patterns: Sequence[str]):
        self._base = base
        self._drops = tuple(drop_patterns)

    def log(self, *args: Any, **kwargs: Any) -> None:
        msg = " ".join(str(a) for a in args) if args else ""
        if any(p in msg for p in self._drops):
            return
        self._base.log(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def make_filtered_run_logger(run_dir: Path, *, drop_patterns: Sequence[str]) -> FilteringLogger:
    """Build the same file-backed logger GEPA would create, wrapped in a filter.

    Mirrors ``gepa.api.py:278`` (``Logger(os.path.join(run_dir, "run_log.txt"))``)
    so that ``run_log.txt`` / ``run_log_stderr.txt`` are still produced.
    """
    from gepa.logging.logger import Logger

    run_dir.mkdir(parents=True, exist_ok=True)
    base = Logger(str(run_dir / "run_log.txt"))
    return FilteringLogger(base, drop_patterns)


def check_lm_alive(cfg: LMConfig, *, label: str, timeout: float = 10.0) -> None:
    """Verify an OpenAI-compatible endpoint is reachable and the named model is loaded.

    Raises `RuntimeError` with a clear message on failure — meant to be called
    at the top of any script that's about to spend real time on this LM.
    """
    print(f"[check] {label}: pinging {cfg.base_url} for model {cfg.model!r} ...")
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key(), timeout=timeout)

    # Preferred: cheap GET /models. Some minimal servers don't implement it,
    # so we fall back to a 1-token chat completion below.
    listed_models: list[str] | None = None
    list_exc: Exception | None = None
    try:
        listing = client.models.list()
        listed_models = [m.id for m in getattr(listing, "data", [])]
    except Exception as exc:  # noqa: BLE001
        list_exc = exc

    if listed_models is not None:
        if cfg.model in listed_models:
            print(f"[check] {label}: ok (model present in /models)")
            return
        # `/models` worked but didn't list our model. Still try a real call
        # — some servers report only base models in /models but still serve
        # aliases.
        print(
            f"[check] {label}: /models did not list {cfg.model!r} "
            f"(saw {len(listed_models)} other models); falling through to a 1-token probe"
        )

    try:
        client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        details = (
            f"  /models probe error: {type(list_exc).__name__}: {list_exc}\n"
            if list_exc is not None
            else ""
        )
        raise RuntimeError(
            f"LM {label!r} unreachable at {cfg.base_url} for model {cfg.model!r}.\n"
            f"{details}"
            f"  chat-completion probe error: {type(exc).__name__}: {exc}"
        ) from exc

    print(f"[check] {label}: ok (1-token probe succeeded)")


_TASK_LM_BUDGET_CAP = 32768


def build_task_lm(
    cfg: LMConfig,
    *,
    temperature: float = 0.0,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    max_tokens: int | None = 0,
    json_mode: bool = True,
    stream_to_stderr: bool = False,
    retry_on_bad_json: int = 0,
    retry_temperature_step: float = 0.2,
):
    """Return a callable `(messages: Sequence[ChatMessage]) -> str`.

    ``top_p`` and ``repetition_penalty`` are passed through to the server when
    set (``None`` = omit, so the server keeps whatever default it was launched
    with — typically the model's HF ``generation_config.json``).
    ``repetition_penalty`` is not part of the OpenAI spec, so it is sent inside
    ``extra_body`` where vLLM picks it up.  Note that at ``temperature=0.0``
    decoding is argmax and ``top_p`` has no effect — only ``repetition_penalty``
    (and frequency/presence penalties) still bite.

    ``retry_on_bad_json`` enables a bounded outer retry loop that fires when the
    returned content does not parse as a JSON object (the common runaway-
    repetition failure mode for local task LMs).  Each retry re-issues the
    same completion with the sampling temperature bumped by
    ``retry_temperature_step`` (capped at 2.0); ``top_p`` and
    ``repetition_penalty`` stay at the base values.  ``0`` (default) disables
    retries so behavior matches the pre-retry implementation byte-for-byte.
    The final response — valid or not after exhausting retries — is what gets
    returned to the caller; the evaluator scores any still-invalid output
    exactly as before.  The number of retries used is published via the
    ``_RETRY_ATTEMPTS_USED`` contextvar so ``_ConversationLogger`` can record
    it on each JSONL line.

    ``max_tokens`` semantics:

    * ``None`` or ``0`` — do not send a token cap.  The model uses its own
      default (e.g. its full context window).  The reasoning-budget auto-grow
      below is also disabled, since there's no cap to grow.
    * positive int — sent as ``max_tokens`` initially; auto-swapped to
      ``max_completion_tokens`` if the server demands it, and doubled (up to
      32768) if a reasoning model truncates the response.

    When ``stream_to_stderr`` is True, the underlying chat-completion call uses
    ``stream=True`` and each ``delta.content`` chunk is written to ``stderr``
    in real time so an operator can watch a stalled or slow sample.  The
    final assembled string is still returned to the caller unchanged, so
    GEPA's interface is preserved.  Intended for single-worker debugging —
    parallel streaming would interleave chunks unreadably.

    Adapts to model-specific quirks on the fly:

    * OpenAI reasoning models (gpt-5, o1, o3, ...) reject ``max_tokens``
      (want ``max_completion_tokens``) and reject custom ``temperature`` values.
      On the first ``BadRequestError`` naming one of these parameters, we
      rewrite/drop the offending key.
    * Reasoning models also charge against the same token budget for their
      *reasoning* tokens.  When that budget is exhausted (``finish_reason ==
      'length'`` with nonzero ``reasoning_tokens``), visible content is empty
      or truncated.  We detect this and double the budget (capped at 32768),
      then retry.
    * Servers that don't support ``stream_options`` are detected by a 400
      and the parameter is dropped (usage data is then unavailable so the
      reasoning-budget auto-grow above is skipped for streamed calls).

    Every adaptation is cached in a lock-protected dict so subsequent and
    concurrent calls reuse the corrected kwargs.  A single ``[adapt]`` line is
    printed per unique adaptation so the change is visible in the run log.
    """

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key())

    base_kwargs: dict[str, Any] = {"temperature": temperature}
    if top_p is not None:
        base_kwargs["top_p"] = top_p
    if max_tokens is not None and max_tokens > 0:
        base_kwargs["max_tokens"] = max_tokens
    if json_mode:
        base_kwargs["response_format"] = {"type": "json_object"}
    if stream_to_stderr:
        base_kwargs["stream"] = True
        base_kwargs["stream_options"] = {"include_usage": True}
    if repetition_penalty is not None:
        # Not OpenAI-spec; vLLM picks it up via extra_body.
        base_kwargs["extra_body"] = {"repetition_penalty": repetition_penalty}

    lock = threading.Lock()
    notified: set[str] = set()

    def _emit_adapt(note: str) -> None:
        with lock:
            fresh = note not in notified
            notified.add(note)
        if fresh:
            _tqdm_safe_print(f"[adapt] {note}")

    def _adapt_error_locked(exc_msg: str) -> str | None:
        """Mutate ``base_kwargs`` to work around a 400; return a short note,
        or None if the error isn't one of the known auto-fixable cases."""
        msg = exc_msg.lower()
        if (
            "max_tokens" in msg
            and "max_completion_tokens" in msg
            and "max_tokens" in base_kwargs
        ):
            base_kwargs["max_completion_tokens"] = base_kwargs.pop("max_tokens")
            return f"task_lm({cfg.model}): switched max_tokens -> max_completion_tokens"
        # "Unsupported value: 'temperature' does not support 0 with this model.
        #  Only the default (1) value is supported."
        if "'temperature'" in msg and "temperature" in base_kwargs:
            base_kwargs.pop("temperature")
            return f"task_lm({cfg.model}): dropped temperature (unsupported)"
        if "stream_options" in msg and "stream_options" in base_kwargs:
            base_kwargs.pop("stream_options")
            return f"task_lm({cfg.model}): dropped stream_options (unsupported)"
        return None

    _RETRY_SENTINEL = "<retry>"

    def _grow_budget_locked(resp: Any, kwargs_used: dict[str, Any]) -> str | None:
        """If a reasoning model truncated due to budget, double the cap.

        Returns:
          - a note string (caller should print it + retry)
          - ``_RETRY_SENTINEL`` (another thread already bumped; just retry)
          - ``None`` (response is fine; caller should return content)
        """
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) != "length":
            return None
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning = getattr(details, "reasoning_tokens", 0) if details else 0
        if not reasoning:
            # Non-reasoning model truncation — respect the user's cap rather
            # than silently inflating it.
            return None
        key = "max_completion_tokens" if "max_completion_tokens" in base_kwargs else "max_tokens"
        current = int(base_kwargs.get(key, 0) or 0)
        my_value = int(kwargs_used.get(key, 0) or 0)
        if current > my_value:
            return _RETRY_SENTINEL  # someone else already bumped; just retry
        if current <= 0:
            return None
        new = min(current * 2, _TASK_LM_BUDGET_CAP)
        if new <= current:
            return None
        base_kwargs[key] = new
        return (
            f"task_lm({cfg.model}): bumped {key} {current} -> {new} "
            f"(reasoning model exhausted budget)"
        )

    def _before_sleep(retry_state: tenacity.RetryCallState) -> None:
        sid = _CURRENT_SAMPLE_ID.get()
        sleep_s = (
            retry_state.next_action.sleep if retry_state.next_action else 0.0
        )
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        err = f"{type(exc).__name__}: {str(exc)[:200]}" if exc else "?"
        _tqdm_safe_print(
            f"[task_lm] sample={sid if sid is not None else '?'}  "
            f"RETRY {retry_state.attempt_number} in {sleep_s:.1f}s after {err}"
        )

    # No ``stop`` — tenacity defaults to ``stop_never``, so transient task_lm
    # errors are retried indefinitely with the same exponential backoff
    # (capped at 60 s per wait). Recovery requires either the server coming
    # back, or the operator sending SIGINT (KeyboardInterrupt propagates out
    # of the retry loop because it is not in ``_TASK_LM_TRANSIENT_EXC``).
    retryer = tenacity.Retrying(
        retry=tenacity.retry_if_exception_type(_TASK_LM_TRANSIENT_EXC),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=_before_sleep,
        reraise=True,
    )

    def _stream_completion(**kw: Any) -> Any:
        """Open a streaming completion, mirror chunks to stderr in real time,
        and return a response-shaped facade for the rest of the pipeline.

        The facade exposes ``choices[0].message.content``,
        ``choices[0].finish_reason``, and (when the server includes a usage
        chunk) ``usage.completion_tokens_details.reasoning_tokens`` so that
        :func:`_grow_budget_locked` keeps working without changes.
        """
        sid = _CURRENT_SAMPLE_ID.get()
        t0 = time.time()
        stream = client.chat.completions.create(**kw)
        sys.stderr.write(
            f"\n[task_lm stream sample={sid if sid is not None else '?'}]\n"
        )
        sys.stderr.flush()
        parts: list[str] = []
        finish_reason: str | None = None
        usage: Any = None
        try:
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                ch = choices[0]
                delta = getattr(ch, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    sys.stderr.write(content)
                    sys.stderr.flush()
                    parts.append(content)
                fr = getattr(ch, "finish_reason", None)
                if fr:
                    finish_reason = fr
        finally:
            sys.stderr.write("\n")
            sys.stderr.flush()
        text = "".join(parts)
        _tqdm_safe_print(
            f"[task_lm stream] done sample={sid if sid is not None else '?'} "
            f"chars={len(text)} dt={time.time() - t0:.1f}s "
            f"finish={finish_reason or '?'}"
        )
        fake_usage: Any = None
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = (
                getattr(details, "reasoning_tokens", 0) if details else 0
            )
            fake_usage = SimpleNamespace(
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=reasoning_tokens
                )
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=fake_usage,
        )

    def _one_attempt(
        messages: Sequence[dict[str, Any]], *, temperature_override: float
    ) -> str:
        """Execute one chat-completion attempt with optional temperature override.

        Wraps the existing tenacity-retry / BadRequest-adaptation / reasoning-
        budget-growth loop; the only call-site customization is the sampling
        temperature, which the outer bad-JSON retry loop walks upward across
        attempts.
        """
        last_exc: BadRequestError | None = None
        for _ in range(8):
            with lock:
                kwargs_snapshot = dict(base_kwargs)
            kwargs = dict(kwargs_snapshot)
            # Apply per-attempt temperature override unless the adaptation
            # path has already dropped 'temperature' (e.g. reasoning model).
            if "temperature" in kwargs:
                kwargs["temperature"] = temperature_override
            try:
                resp = retryer(
                    _stream_completion if stream_to_stderr else client.chat.completions.create,
                    model=cfg.model,
                    messages=list(messages),
                    **kwargs,
                )
            except BadRequestError as exc:
                last_exc = exc
                exc_msg = getattr(exc, "message", "") or str(exc)
                err_note: str | None = None
                with lock:
                    # Race-detection compares the *unmodified* snapshot of
                    # base_kwargs at read time, not the override-mutated copy.
                    if kwargs_snapshot == base_kwargs:
                        err_note = _adapt_error_locked(exc_msg)
                        if err_note is None:
                            raise  # not auto-fixable
                    # else: another thread already adapted; just retry.
                if err_note is not None:
                    _emit_adapt(err_note)
                continue

            with lock:
                grow_note = _grow_budget_locked(resp, kwargs)
            if grow_note is None:
                return resp.choices[0].message.content or ""
            if grow_note != _RETRY_SENTINEL:
                _emit_adapt(grow_note)
            # loop continues -> retry with the larger budget
        raise RuntimeError(
            f"task_lm({cfg.model}): exhausted retries after BadRequestError/"
            f"reasoning-budget growth attempts"
        ) from last_exc

    def _call(messages: Sequence[dict[str, Any]]) -> str:
        max_extra = max(0, retry_on_bad_json)
        sid = _CURRENT_SAMPLE_ID.get()
        _RETRY_ATTEMPTS_USED.set(0)
        response = ""
        for attempt in range(max_extra + 1):
            t_eff = min(temperature + attempt * retry_temperature_step, 2.0)
            response = _one_attempt(messages, temperature_override=t_eff)
            if retry_on_bad_json == 0 or _is_valid_json_object(response):
                _RETRY_ATTEMPTS_USED.set(attempt)
                return response
            if attempt < max_extra:
                next_t = min(
                    temperature + (attempt + 1) * retry_temperature_step, 2.0
                )
                _tqdm_safe_print(
                    f"[task_lm retry] sample={sid if sid is not None else '?'}  "
                    f"attempt={attempt + 1}/{max_extra}  next_T={next_t:.2f}  "
                    f"reason=bad_json  resp_chars={len(response)}"
                )
        _RETRY_ATTEMPTS_USED.set(max_extra)
        return response

    return _call


def build_reflection_lm(
    cfg: LMConfig,
    *,
    temperature: float = 1.0,
    max_tokens: int | None = 0,
):
    """Return a `gepa.lm.LM` that satisfies GEPA's `LanguageModel` protocol.

    ``max_tokens=None`` or ``0`` omits the parameter so the model uses its
    own default (and so litellm's reasoning-model handling kicks in).
    """

    from gepa.lm import LM

    effective_max_tokens = max_tokens if (max_tokens is not None and max_tokens > 0) else None

    return LM(
        f"openai/{cfg.model}",
        temperature=temperature,
        max_tokens=effective_max_tokens,
        api_base=cfg.base_url,
        api_key=cfg.api_key(),
    )
