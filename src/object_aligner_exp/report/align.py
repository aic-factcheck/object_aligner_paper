"""Derive per-(gold, pred) display data using Object Aligner.

OA is instantiated with **no extra kwargs**, matching ``oa.py:16``. The
REBEL schema in ``data/rebel/schemas/rebel.jsonc`` uses only
``jaro_winkler`` / ``jaro`` / ``idScope`` / ``ref`` — no embedding metric,
no LM. So merely calling ``ObjectAligner(schema)`` is LM-free; the report
generator never makes a network call.

For each entity / relation list in pred, we derive a permutation that
puts the best-matched gold counterpart at the same index. Unmatched
gold slots are kept (left side shows the gold entity, right side shows a
``__missing__`` placeholder); unmatched pred items are appended at the
end of the pred view (left side shows ``__missing__``).
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
from typing import Any, Callable

from object_aligner import ObjectAligner
from object_aligner.object_aligner import MatchDict, MatchItem, MatchList

from object_aligner_exp.oa import nostruct_feedback


MISSING = {"__missing__": True}


def _is_missing(row: Any) -> bool:
    return isinstance(row, dict) and row.get("__missing__") is True


def _collect_scope_paths(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Walk a JSON-schema dict and locate every ``idScope`` and ``ref``.

    Returns a per-scope dict with keys:

    - ``array_path``: tuple of dict keys (no ``*``) leading to the array
      that holds the idScope's items. ``None`` if the idScope is not
      inside an array (degenerate schema).
    - ``id_field_path``: tuple of dict keys (no ``*``) from a single row
      of that array down to the primitive id field.
    - ``ref_paths``: list of data-paths, each a tuple of strings where
      ``"*"`` marks an array-item descent. Refs may live anywhere in the
      tree, not necessarily in the same array.

    Only the public schema dict is inspected — OA internals are not used.
    """
    scopes: dict[str, dict[str, Any]] = {}

    def ensure(scope: str) -> dict[str, Any]:
        if scope not in scopes:
            scopes[scope] = {"array_path": None, "id_field_path": (), "ref_paths": []}
        return scopes[scope]

    def visit(node: Any, data_path: tuple, array_path: tuple | None) -> None:
        if not isinstance(node, dict):
            return
        idscope = node.get("idScope")
        ref = node.get("ref")
        if isinstance(idscope, str) and array_path is not None:
            entry = ensure(idscope)
            # id_field_path is the part of the data_path *inside* the array's row.
            # data_path looks like (..array_path.., "*", ..inner fields..).
            entry["array_path"] = array_path
            entry["id_field_path"] = data_path[len(array_path) + 1 :]
        if isinstance(ref, str):
            entry = ensure(ref)
            entry["ref_paths"].append(data_path)
        if "items" in node:
            visit(node["items"], data_path + ("*",), data_path)
        if "properties" in node and isinstance(node["properties"], dict):
            for key, child in node["properties"].items():
                visit(child, data_path + (str(key),), array_path)
        if "prefixItems" in node and isinstance(node["prefixItems"], list):
            for i, child in enumerate(node["prefixItems"]):
                visit(child, data_path + (i,), array_path)

    visit(schema, (), None)
    return scopes


def _get_at(data: Any, path: tuple) -> Any:
    cur = data
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _read_id(row: dict[str, Any], id_field_path: tuple) -> Any:
    cur: Any = row
    for k in id_field_path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _write_id(row: dict[str, Any], id_field_path: tuple, val: Any) -> None:
    if not id_field_path:
        return
    cur: Any = row
    for k in id_field_path[:-1]:
        if not isinstance(cur, dict):
            return
        cur = cur.get(k)
    if not isinstance(cur, dict):
        return
    cur[id_field_path[-1]] = val


def _walk_ref_leaves(data: Any, path: tuple):
    """Yield ``(parent, key, top_key, row_index)`` for each leaf at ``path``.

    ``top_key`` and ``row_index`` are populated when the path's first
    element is a dict key and the second is ``"*"`` (the common case in
    current schemas). Otherwise they are ``None``.
    """
    if not path:
        return
    top_key = path[0] if len(path) >= 2 and path[1] == "*" else None

    def rec(cur: Any, remaining: tuple, row_idx: int | None):
        if not remaining:
            return
        step = remaining[0]
        rest = remaining[1:]
        if step == "*":
            if not isinstance(cur, list):
                return
            for i, item in enumerate(cur):
                yield from rec(item, rest, i if row_idx is None else row_idx)
        else:
            if not isinstance(cur, dict) or step not in cur:
                return
            if not rest:
                yield (cur, step, top_key, row_idx)
            else:
                yield from rec(cur[step], rest, row_idx)

    yield from rec(data, path, None)


def _remap_pred_to_gold_ids(
    aligned_gold: dict[str, Any],
    aligned_pred: dict[str, Any],
    schema_meta: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Build a `pred_id → gold_id` map per scope and rewrite the predicted
    dict's id and ref fields in-place on a deep copy.

    Returns ``(remapped_pred, meta)``. ``meta`` is a dict shaped like
    ``{top_key: [{field_name: "id-unrewritten", ...}, ...]}`` and is only
    populated for top-level array keys (which covers every schema in the
    repo). Unmatched-pred id fields and refs that can't be remapped get
    a marker; everything else is left absent from ``meta``.
    """
    remapped = copy.deepcopy(aligned_pred)
    meta: dict[str, list[dict[str, str]]] = {}

    def mark(top_key: str | None, row_idx: int | None, field: str) -> None:
        if top_key is None or row_idx is None:
            return
        rows = meta.setdefault(top_key, [])
        while len(rows) <= row_idx:
            rows.append({})
        rows[row_idx][field] = "id-unrewritten"

    id_maps: dict[str, dict[Any, Any]] = {}

    for scope, info in schema_meta.items():
        array_path = info.get("array_path")
        id_field_path = info.get("id_field_path") or ()
        if not array_path or not id_field_path:
            continue
        gold_list = _get_at(aligned_gold, array_path)
        pred_list = _get_at(remapped, array_path)
        if not isinstance(gold_list, list) or not isinstance(pred_list, list):
            continue
        top_key = array_path[0] if len(array_path) == 1 else None
        id_field_name = id_field_path[-1] if len(id_field_path) == 1 else None
        mapping: dict[Any, Any] = {}
        for i, (g_row, p_row) in enumerate(zip(gold_list, pred_list)):
            if _is_missing(p_row):
                continue
            if _is_missing(g_row):
                # Solo pred row: keep original pred id, mark it.
                if id_field_name is not None and isinstance(p_row, dict) and id_field_name in p_row:
                    mark(top_key, i, id_field_name)
                continue
            if not (isinstance(g_row, dict) and isinstance(p_row, dict)):
                continue
            g_id = _read_id(g_row, id_field_path)
            p_id = _read_id(p_row, id_field_path)
            if g_id is None or p_id is None:
                continue
            mapping[p_id] = g_id
            # Rewrite the id field in the pred row.
            _write_id(p_row, id_field_path, g_id)
        id_maps[scope] = mapping

    for scope, info in schema_meta.items():
        mapping = id_maps.get(scope, {})
        for ref_path in info.get("ref_paths") or []:
            for parent, key, top_key, row_idx in _walk_ref_leaves(remapped, ref_path):
                val = parent[key]
                if val in mapping:
                    parent[key] = mapping[val]
                else:
                    mark(top_key, row_idx, key)

    return remapped, meta


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _reconstruct_record(md: MatchDict, side: str) -> dict[str, Any]:
    """Pull the matched side's primitive values back into a dict.

    `side` is ``"gold"`` or ``"pred"``. For each child key (a MatchItem
    over the key string), take ``getattr(key_match, side)`` as the key,
    then ``getattr(value_match, side)`` as the value if the value match
    is a leaf MatchItem. Only used for relation alignment, where there
    is no idScope to fall back on.
    """
    rec: dict[str, Any] = {}
    for key_match, value_match in md.children.items():
        k = getattr(key_match, side, None)
        if k is None:
            continue
        if isinstance(value_match, MatchItem):
            rec[k] = getattr(value_match, side, None)
    return rec


def _find_index_by_record(lst: list[dict[str, Any]], rec: dict[str, Any]) -> int | None:
    """Find the first element in `lst` whose key/value pairs match `rec`.

    Used because OA reconstructs entity/relation dicts from primitives during
    alignment, so the original Python references are not preserved at the
    MatchDict level.
    """
    for i, item in enumerate(lst):
        if all(item.get(k) == v for k, v in rec.items()):
            return i
    return None


def _derive_list_alignment(
    gold_list: list[dict[str, Any]],
    pred_list: list[dict[str, Any]],
    ml: MatchList,
) -> list[tuple[int | None, int | None]]:
    """Walk a `MatchList` and produce per-child ``(gold_idx, pred_idx)``.

    For a `MatchItem` leaf, exactly one of ``.gold`` / ``.pred`` is a full
    dict; the other is ``None``. For a `MatchDict` child both sides are
    matched and we reconstruct each side's record to look it up by content.
    """
    rows: list[tuple[int | None, int | None]] = []
    for child in ml.children:
        g_idx: int | None = None
        p_idx: int | None = None
        if isinstance(child, MatchItem):
            if isinstance(child.gold, dict):
                g_idx = _find_index_by_record(gold_list, child.gold)
            if isinstance(child.pred, dict):
                p_idx = _find_index_by_record(pred_list, child.pred)
        elif isinstance(child, MatchDict):
            g_rec = _reconstruct_record(child, "gold")
            p_rec = _reconstruct_record(child, "pred")
            if g_rec:
                g_idx = _find_index_by_record(gold_list, g_rec)
            if p_rec:
                p_idx = _find_index_by_record(pred_list, p_rec)
        rows.append((g_idx, p_idx))
    return rows


def _build_aligned_list(
    orig: list[dict[str, Any]],
    indices: list[int | None],
    extras_at_end: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx in indices:
        if idx is None:
            out.append(MISSING.copy())
        else:
            out.append(orig[idx])
    out.extend(extras_at_end)
    return out


def _is_scalar_matchlist(ml: MatchList) -> bool:
    """True if the list's items are scalars (no dict on either side of a child).

    Object-array fields produce ``MatchItem`` children whose ``.gold`` / ``.pred``
    are dicts (or ``MatchDict`` children); scalar arrays (e.g. an array of
    integers) produce ``MatchItem`` children whose values are the scalars
    themselves. An empty list is treated as scalar (both branches yield ``[]``).
    """
    for child in ml.children:
        if isinstance(child, MatchDict):
            return False
        if isinstance(child, MatchItem) and (
            isinstance(child.gold, dict) or isinstance(child.pred, dict)
        ):
            return False
    return True


def _align_scalar_field(
    ml: MatchList,
) -> tuple[list[Any], list[Any]]:
    """Parallel (gold_view, pred_view) for a scalar-item array.

    Built straight from the per-child ``MatchItem``s (each ``.gold`` / ``.pred``
    is the scalar value, or ``None`` when unmatched), which OA already emits in
    display order — matched positions hold the value on both sides; an unmatched
    gold/pred slot gets ``__missing__`` on the other side. The frontend renders
    scalars directly and styles each position via gold-vs-pred comparison.
    """
    gold_view: list[Any] = []
    pred_view: list[Any] = []
    for child in ml.children:
        g = getattr(child, "gold", None)
        p = getattr(child, "pred", None)
        gold_view.append(MISSING.copy() if g is None else g)
        pred_view.append(MISSING.copy() if p is None else p)
    return gold_view, pred_view


def _align_one_field(
    gold_list: list[dict[str, Any]],
    pred_list: list[dict[str, Any]],
    ml: MatchList,
) -> tuple[list[Any], list[Any]]:
    """Return the (gold_view, pred_view) parallel lists for one field.

    Both views have the same length: matched rows first (in alignment
    order), then unmatched-gold rows (with ``__missing__`` on the pred
    side), then unmatched-pred rows (with ``__missing__`` on the gold
    side). The implementation re-uses the per-child ``(g, p)`` pairing
    produced by OA's Hungarian assignment.

    Scalar-item arrays (e.g. an array of integers) are handled separately:
    their ``MatchItem`` children carry the scalar values directly, so we build
    the views from those rather than looking rows up by record.
    """
    if _is_scalar_matchlist(ml):
        return _align_scalar_field(ml)

    rows = _derive_list_alignment(gold_list, pred_list, ml)

    matched: list[tuple[int, int]] = []
    only_gold: list[int] = []
    only_pred: list[int] = []
    for g, p in rows:
        if g is not None and p is not None:
            matched.append((g, p))
        elif g is not None:
            only_gold.append(g)
        elif p is not None:
            only_pred.append(p)

    gold_view: list[dict[str, Any]] = []
    pred_view: list[dict[str, Any]] = []
    for g, p in matched:
        gold_view.append(gold_list[g])
        pred_view.append(pred_list[p])
    for g in only_gold:
        gold_view.append(gold_list[g])
        pred_view.append(MISSING.copy())
    for p in only_pred:
        gold_view.append(MISSING.copy())
        pred_view.append(pred_list[p])

    return gold_view, pred_view


def _empty_view(orig: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    if side == "pred":
        return [MISSING.copy() for _ in orig]
    return [MISSING.copy() for _ in orig]


def align_pair(
    gold: dict[str, Any],
    pred: dict[str, Any],
    oa: ObjectAligner,
    schema: dict[str, Any] | None = None,
    *,
    structural_filter: Callable[[str], bool] | None = None,
    top_k: int | None = 5,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, list[dict[str, str]]] | None,
    float,
    str | None,
]:
    """Compute aligned-view versions of gold and pred plus the OA score.

    Returns ``(gold_view, pred_view, pred_view_goldids, goldids_meta, score, feedback)``.
    The first two share the same top-level key set and (for ``order: "align"``
    arrays) the same row order so the consumer can render them side-by-side.

    ``pred_view_goldids`` is ``pred_view`` with all matched-row idScope
    values rewritten into the gold ID namespace, and all refs into those
    scopes rewritten through the same per-scope map. ``goldids_meta``
    records which fields could not be rewritten (so the JS can style
    them). Both are ``None`` if ``schema`` is omitted or has no idScope.

    Generic in the schema: every array-of-objects field that OA produced a
    ``MatchList`` for is aligned via that match; scalar fields and any
    fields OA didn't match (because they weren't in the schema or the
    input) are copied through unchanged.

    If OA raises (bad refs, schema mismatch, etc.), the caller should
    catch and degrade to ``available=True, score=None, aligned=None``.
    """
    match = oa.align(gold, pred)
    score = float(match.score)

    gold_view: dict[str, Any] = {}
    pred_view: dict[str, Any] = {}

    if isinstance(match, MatchDict):
        for key_match, value_match in match.children.items():
            key = getattr(key_match, "gold", None) or getattr(key_match, "pred", None)
            if not isinstance(key, str):
                continue
            if isinstance(value_match, MatchList):
                g_list = list(gold.get(key, []) or [])
                p_list = list(pred.get(key, []) or [])
                gv, pv = _align_one_field(g_list, p_list, value_match)
                gold_view[key] = gv
                pred_view[key] = pv
            else:
                # Scalar / nested object — fall back to the raw values so
                # the renderer can still display them.
                gold_view[key] = gold.get(key)
                pred_view[key] = pred.get(key)

    # Preserve any top-level keys OA didn't produce a child for so the
    # renderer doesn't silently drop them (e.g. extra debug fields).
    for k, v in gold.items():
        gold_view.setdefault(k, v)
    for k, v in pred.items():
        pred_view.setdefault(k, v)

    pred_view_goldids: dict[str, Any] | None = None
    goldids_meta: dict[str, list[dict[str, str]]] | None = None
    if schema is not None:
        schema_meta = _collect_scope_paths(schema)
        if any(s.get("array_path") for s in schema_meta.values()):
            pred_view_goldids, goldids_meta = _remap_pred_to_gold_ids(
                gold_view, pred_view, schema_meta
            )

    feedback: str | None = None
    try:
        # nostruct arm (structural_filter set): drop structural fix items from the
        # feedback text so the report reproduces exactly what the reflection LM saw.
        if structural_filter is not None:
            fb = nostruct_feedback(
                oa, gold, pred, is_structural=structural_filter, top_k=top_k
            )
        else:
            fb = oa.feedback(gold, pred, top_k=top_k)
        feedback = fb.text
    except Exception:
        feedback = None

    return gold_view, pred_view, pred_view_goldids, goldids_meta, score, feedback


def make_aligner(
    schema: dict[str, Any],
    *,
    referential_feedback: str = "literal",
    wl_integration: str = "tie_break",
) -> ObjectAligner:
    """Construct a fully-offline, LM-free `ObjectAligner` for the given schema.

    ``referential_feedback`` (``"literal"`` | ``"semantic"``) sets the
    constructor default that ``align_pair``'s ``oa.feedback(...)`` inherits, so
    the report can reproduce the same referential feedback the run's arm used.

    ``wl_integration`` (``"tie_break"`` | ``"blend"``) likewise reproduces the
    run's WL scoring mode, so regenerated scores in ``report.html`` match the
    numbers the run recorded.
    """
    return ObjectAligner(
        schema,
        referential_feedback=referential_feedback,
        wl_integration=wl_integration,
    )


_CacheValue = tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, list[dict[str, str]]] | None,
    float,
    str | None,
]


class AlignCache:
    """Per-render cache for (gold, pred) -> aligned views + remap + score.

    GEPA writes the same prediction text under multiple ``iter_*`` files for
    the same ``(candidate, sample)`` when later iterations didn't change it,
    so caching across the run is worth the few-line cost.
    """

    def __init__(
        self,
        oa: ObjectAligner,
        schema: dict[str, Any] | None = None,
        *,
        structural_filter: Callable[[str], bool] | None = None,
        top_k: int | None = 5,
    ) -> None:
        self._oa = oa
        self._schema = schema
        self._structural_filter = structural_filter
        self._top_k = top_k
        self._cache: dict[tuple[str, str], _CacheValue | None] = {}

    def get(
        self,
        gold: dict[str, Any],
        pred: dict[str, Any],
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, list[dict[str, str]]] | None,
        float | None,
        str | None,
        str | None,
    ]:
        """Return ``(gold_view, pred_view, pred_goldids, goldids_meta, score, feedback, error)``.

        On OA failure (raised exception): all data fields are ``None`` and
        ``error`` is the formatted exception message. On success: all
        five views are populated (``pred_goldids`` / ``goldids_meta`` are
        ``None`` if the schema has no idScope) and ``error`` is ``None``.
        ``gold_view`` is included because aligned mode re-orders gold
        rows too (gold unmatched-by-pred entities are pushed to the back
        of both lists; that wouldn't appear in the bare gold).
        """
        key = (_hash(gold), _hash(pred))
        if key in self._cache:
            cached = self._cache[key]
            if cached is None:
                return None, None, None, None, None, None, "cached error"
            gv, pv, pvg, gm, s, fb = cached
            return gv, pv, pvg, gm, s, fb, None
        try:
            gv, pv, pvg, gm, s, fb = align_pair(
                gold, pred, self._oa, self._schema,
                structural_filter=self._structural_filter,
                top_k=self._top_k,
            )
        except Exception as exc:  # noqa: BLE001 — OA raises various types
            self._cache[key] = None
            return None, None, None, None, None, None, f"{type(exc).__name__}: {exc}"
        self._cache[key] = (gv, pv, pvg, gm, s, fb)
        return gv, pv, pvg, gm, s, fb, None
