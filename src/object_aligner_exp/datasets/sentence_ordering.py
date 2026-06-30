"""Sentence-ordering dataset utilities (ROCStories now, arXiv abstracts later).

The model is given N sentences of a text in **scrambled** order, labelled 1..N,
and must output a permutation of those labels giving the correct reading order
(under the deliberately neutral key ``"indices"``, so neither the schema nor the
seed prompt hints that the task is a reordering):

    {"indices": [3, 1, 4, 2, 5]}

Because the output is a permutation of ``{1..N}``, the *bag* of labels is always
exactly correct, so OA's order-blind Hungarian alignment (``"order":"align"``)
scores ≈1.0 for **any** permutation while the order-aware fixed alignment
(``"order":"fixed"``) reflects the true ordering quality. All of the
fixed-vs-Hungarian signal therefore lives in the order — which is exactly the
regime PlanBench / NATURAL PLAN lacked (there the order was causally entailed by
the items, so the task LM emitted it correctly and the two rewards agreed).

Two corpora share the same wire shape and schema pair:

* **ROCStories** — HuggingFace ``mintujupally/ROCStories`` (~78k 5-sentence
  commonsense stories, stored as a single ``text`` field). We split into
  sentences with a simple regex and keep stories that yield exactly 5.
* **arXiv abstracts** — a HuggingFace abstracts corpus (confirm id at build
  time), split with nltk and length-filtered to N∈[6,12]. Heavier discourse
  ordering → a larger gap; scaffolded here, built later.

Each example carries the deterministically-shuffled sentences + the gold reading
order. ``meta`` keeps the originals and ``difficulty`` (= N) for stratified
split carving and the PMR/τ scorer (``sentence_ordering_eval.py``).
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from typing import Any, TypedDict

ROCSTORIES_HF = "mintujupally/ROCStories"
# Candidate arXiv-abstracts corpus; confirm at arXiv-build time.
ARXIV_HF = "common-pile/arxiv_abstracts"

ROCSTORIES_N = 5
ARXIV_MIN_SENTS = 6
ARXIV_MAX_SENTS = 12

# Per-example shuffle seed = BASE multiplier * index + base_seed (deterministic,
# decorrelated across examples).
_SEED_MULT = 1_000_003


class SentenceOrderingExample(TypedDict):
    """One preprocessed sentence-ordering example."""

    id: str
    title: str
    context: str
    gold: dict[str, Any]
    meta: dict[str, Any]


# --- sentence segmentation -------------------------------------------------

_RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences_simple(text: str) -> list[str]:
    """Regex sentence split for clean, simple text (ROCStories)."""
    parts = [s.strip() for s in _RE_SENT_SPLIT.split(text.strip())]
    return [s for s in parts if s]


def split_sentences_nltk(text: str) -> list[str]:
    """Robust sentence split via nltk (arXiv abstracts). Lazy dependency."""
    try:
        import nltk
    except ImportError as exc:  # pragma: no cover - exercised only on arXiv path
        raise RuntimeError(
            "the arXiv corpus needs nltk; install it (it is a project dep) — "
            "uv sync — and it will fetch the 'punkt' model on first use."
        ) from exc
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:  # pragma: no cover
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:  # pragma: no cover
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:  # noqa: BLE001 - older nltk has no punkt_tab
            pass
    # Collapse internal whitespace (arXiv abstracts are hard-wrapped with
    # newlines) so each labelled sentence is a clean single line.
    return [
        re.sub(r"\s+", " ", s).strip()
        for s in nltk.sent_tokenize(text.strip())
        if s.strip()
    ]


# --- example assembly ------------------------------------------------------


def _shuffle_perm(n: int, rng: random.Random) -> list[int]:
    """A non-identity permutation of range(n) (re-rolled until it moves)."""
    if n < 2:
        return list(range(n))
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if perm != list(range(n)):
            return perm


def build_example(
    corpus: str,
    ex_id: str,
    sentences: list[str],
    *,
    seed_index: int,
    base_seed: int,
) -> SentenceOrderingExample:
    """Shuffle ``sentences`` (correct order) and build the ordered-permutation task.

    ``shuffled[j]`` (label ``j+1``) holds original sentence ``perm[j]``; the gold
    reading order is the labels that reproduce the original sequence:
    ``gold_order[k] = perm.index(k) + 1``.
    """
    n = len(sentences)
    rng = random.Random(base_seed + _SEED_MULT * seed_index)
    perm = _shuffle_perm(n, rng)
    shuffled = [sentences[perm[j]] for j in range(n)]
    gold_order = [perm.index(k) + 1 for k in range(n)]

    context_lines = [f"Sentence {j + 1}: {shuffled[j]}" for j in range(n)]
    context = (
        "Here are the sentences in scrambled order:\n\n" + "\n".join(context_lines)
    )
    # Data key is the neutral "indices" (not "order") so neither the schema nor
    # the seed prompt hints that the task is a reordering. See the schemas.
    gold = {"indices": gold_order}
    meta = {
        "corpus": corpus,
        "n_sentences": n,
        "shuffled_sentences": shuffled,
        "orig_sentences": sentences,
        "difficulty": n,
    }
    return SentenceOrderingExample(
        id=ex_id, title=ex_id, context=context, gold=gold, meta=meta
    )


def iter_rocstories_examples(
    *, limit: int | None = None, base_seed: int = 20260530
) -> Iterator[SentenceOrderingExample]:
    """Stream ROCStories examples (keep stories that split into exactly 5)."""
    from datasets import load_dataset  # heavy optional dep

    ds = load_dataset(ROCSTORIES_HF, split="train")
    yielded = 0
    for i, row in enumerate(ds):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        sents = split_sentences_simple(text)
        if len(sents) != ROCSTORIES_N:
            continue
        yield build_example(
            "rocstories", f"rocstories_{i}", sents, seed_index=i, base_seed=base_seed
        )
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def iter_arxiv_examples(
    *, limit: int | None = None, base_seed: int = 20260530
) -> Iterator[SentenceOrderingExample]:
    """Stream arXiv-abstract examples (nltk-split, keep N∈[6,12]). Built later."""
    from datasets import load_dataset

    ds = load_dataset(ARXIV_HF, split="train", streaming=True)
    yielded = 0
    for i, row in enumerate(ds):
        text = str(row.get("abstract") or row.get("text") or "").strip()
        if not text:
            continue
        sents = split_sentences_nltk(text)
        if not (ARXIV_MIN_SENTS <= len(sents) <= ARXIV_MAX_SENTS):
            continue
        yield build_example(
            "arxiv", f"arxiv_{i}", sents, seed_index=i, base_seed=base_seed
        )
        yielded += 1
        if limit is not None and yielded >= limit:
            return


ITERATORS = {
    "rocstories": iter_rocstories_examples,
    "arxiv": iter_arxiv_examples,
}
