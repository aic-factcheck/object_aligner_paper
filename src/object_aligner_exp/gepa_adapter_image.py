"""GEPA adapter that feeds an image to the task LM (and optionally to the reflection LM).

The text-side ``ParallelDefaultAdapter`` builds ``{role: user, content: <str>}``
messages from ``DataInst.input``. For an image-input task we need
``content`` to be a list of OpenAI vision content parts: a text block plus
an ``image_url`` block. This module's :class:`ImageGepaAdapter`:

* overrides :meth:`evaluate` to build those multimodal messages (the image
  path is read from ``data["additional_context"]["image_path"]`` set by
  :func:`object_aligner_exp.evaluator.make_image_data_inst`), and
* optionally overrides :meth:`make_reflective_dataset` to wrap each
  evaluated image as a :class:`gepa.image.Image` in the reflective record
  so that GEPA's ``InstructionProposalSignature`` will pass it on as
  multimodal content to a vision-capable reflection LM.

The parallel fan-out and per-call logging path provided by
``ParallelDefaultAdapter`` are inherited unchanged.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from gepa.adapters.default_adapter.default_adapter import (
    ChatCompletionCallable,
    DefaultDataInst,
    DefaultRolloutOutput,
    DefaultTrajectory,
    Evaluator,
)
from gepa.core.adapter import EvaluationBatch

import time

from object_aligner_exp.gepa_adapter import (
    ParallelDefaultAdapter,
    _print_batch_summary,
    _resolve_workers,
    _run_per_sample,
    _sample_id_of,
)


_MEDIA_TYPE_BY_EXT: dict[str, str] = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
}


def _guess_media_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return _MEDIA_TYPE_BY_EXT.get(suffix, "image/png")


@lru_cache(maxsize=512)
def _encode_image_data_url(path: str) -> str:
    """Read ``path`` and return a ``data:<mt>;base64,<b64>`` URL.

    Cached so a 50-item valset isn't re-encoded on every full eval.
    """
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{_guess_media_type(path)};base64,{b64}"


def _image_path_for(data: DefaultDataInst) -> str:
    ctx = data.get("additional_context") or {}
    img = ctx.get("image_path")
    if not isinstance(img, str) or not img:
        raise RuntimeError(
            "ImageGepaAdapter requires DataInst.additional_context['image_path'] "
            "(use object_aligner_exp.evaluator.make_image_data_inst)."
        )
    return img


class ImageGepaAdapter(ParallelDefaultAdapter):
    """``ParallelDefaultAdapter`` that attaches an image to every user message.

    Parameters
    ----------
    model, evaluator, max_workers:
        See :class:`ParallelDefaultAdapter`.
    reflection_sees_image:
        If ``True``, the reflective record returned to GEPA includes the
        same image as a :class:`gepa.image.Image` value so the reflection
        LM receives it as multimodal content. Default ``False`` (cheaper;
        also works with text-only reflection LMs).
    """

    def __init__(
        self,
        model: str | ChatCompletionCallable,
        evaluator: Evaluator | None = None,
        *,
        max_workers: int = 1,
        reflection_sees_image: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            evaluator=evaluator,
            max_workers=max_workers,
            **kwargs,
        )
        self.reflection_sees_image = reflection_sees_image
        # Resolved lazily so an outdated GEPA wheel doesn't break import.
        self._image_cls: Any | None = None

    def _resolve_image_cls(self) -> Any | None:
        if self._image_cls is not None:
            return self._image_cls
        try:
            from gepa.image import Image  # type: ignore[import-not-found]
        except Exception:
            return None
        self._image_cls = Image
        return Image

    # ----- task-side: build multimodal messages --------------------------

    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput]:
        """Interleaved LM-call + scoring per sample, with multimodal messages."""
        system_content = next(iter(candidate.values()))
        messages_list: list[list[dict[str, Any]]] = []
        for data in batch:
            data_url = _encode_image_data_url(_image_path_for(data))
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": data["input"]},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            messages_list.append(
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
            )

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

    # ----- reflection-side: optionally surface the image -----------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        base = super().make_reflective_dataset(candidate, eval_batch, components_to_update)
        if not self.reflection_sees_image:
            return base

        image_cls = self._resolve_image_cls()
        if image_cls is None:
            # Wheel installed here predates gepa.image; degrade gracefully.
            return base

        assert eval_batch.trajectories is not None, (
            "reflection_sees_image=True requires capture_traces=True upstream "
            "in GEPA's evaluation loop"
        )
        trajectories = eval_batch.trajectories
        enriched: dict[str, list[dict[str, Any]]] = {}
        for comp, records in base.items():
            new_records: list[dict[str, Any]] = []
            for traj, rec in zip(trajectories, records, strict=True):
                ctx = traj["data"].get("additional_context") or {}
                img_path = ctx.get("image_path")
                merged = dict(rec)
                if isinstance(img_path, str) and img_path:
                    # Insert Image *before* "Inputs" so format_samples renders
                    # the picture above the textual instruction in markdown.
                    new_rec: dict[str, Any] = {"Image": image_cls(path=img_path)}
                    new_rec.update(merged)
                    new_records.append(new_rec)
                else:
                    new_records.append(merged)
            enriched[comp] = new_records
        return enriched
