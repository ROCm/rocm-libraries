# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Checks of the per-arch tuned tables of the attention backward.

The per-arch modules (:mod:`.gfx942_tuning`, :mod:`.gfx950_tuning`) hold the
tuned tables; this module is the gate an entry passes before it is shipped.
An entry is checked by planning it, the way the routing consumes it, on
representative requests of its key:

* a general entry ``{tuning key: knobs}``
  (:func:`kernels.common.attention_bwd_plan.attn_bwd_tuning_key`) on
  :func:`general_key_requests`: requests of the key's performance class and
  buckets that vary what the key leaves free (head counts and group sizes of
  the head mode, sequence lengths at both edges of the kv-tile bucket,
  batches and ``num_cus`` at the edges of the grid-fill bucket, with and
  without ``num_xcds``, both dense layouts, the band forms of the mask kind).
  The entry is the policy's ``tuned_table`` (the plan's most-specific
  lookup);
* a dense entry ``{(arch, dense key): knobs}``
  (:func:`kernels.common.attention_bwd_plan._dense_key`) on
  :func:`dense_key_requests`: the key's shape over several batches, with and
  without ``num_cus`` / ``num_xcds``. The entry is the dense tuning of a
  policy whose dense pack holds the key.

Every representative request that plans without the entry (the gate policy:
every feature group declared, unverified classes waived) must plan with it;
the first one that does not raises ``ValueError`` with the plan's reason
(an unknown knob, an illegal value, a coupling, a value that needs a device
hint the key does not carry). A key without a plannable representative
request is refused too, so an entry is never shipped unchecked.

:func:`check_arch_tables` is the import-time pass over one arch module's
tables (run by :mod:`.routing` when the package is imported; it raises, so a
bad entry fails the import): every knob is a registered axis of its family
with a value that is neither ``not_built``, ``gated`` nor known wrong
(:data:`~.tuning_common.BWD_KNOWN_WRONG_KNOBS`), every entry plans on its
first plannable representative request (the check above with ``limit=1``),
and the performance-status tables name measured classes of the arch. The
check over every representative runs in the tests (it plans hundreds of
requests).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable, Mapping, Optional

from .tuning_common import backward_axis, known_wrong_findings

__all__ = [
    "GENERAL_KEY_LEN",
    "DENSE_KEY_LEN",
    "check_arch_tables",
    "check_dense_entry",
    "check_entry_knobs",
    "check_general_entry",
    "dense_entries",
    "dense_key_requests",
    "general_entries",
    "general_key_requests",
]

# (arch, route, d, dtype, mask kind, head mode, length relation, length mode,
#  stage_vec, kv-tile bucket, fill bucket): attn_bwd_tuning_key
GENERAL_KEY_LEN = 11
# (s_q, s_kv, h_q, h_k, d, dtype, layout, mask kind, band): _dense_key
DENSE_KEY_LEN = 9

# Head counts per head mode: (h_q, h_k, h_v). The GQA groups include odd and
# prime sizes, so a runtime value that only divides some groups (g_split) is
# caught; the h_k != h_v rows have atomic head groups 2 and 1.
_HEADS = {
    "mha": ((1, 1, 1), (12, 12, 12), (32, 32, 32)),
    "gqa": (
        (4, 2, 2),
        (6, 2, 2),
        (8, 2, 2),
        (10, 2, 2),
        (14, 2, 2),
        (16, 2, 2),
        (32, 8, 8),
        (16, 1, 1),
    ),
    "hk_ne_hv": ((8, 4, 2), (8, 2, 4), (15, 5, 3)),
}
_BATCHES = (1, 4)
_XCDS = (None, 8)


def _tuple(x):
    """JSON lists back to the tuples of a key (nested)."""
    if isinstance(x, list):
        return tuple(_tuple(v) for v in x)
    return x


def _knobs(raw: Mapping[str, Any]) -> dict:
    return {k: _tuple(v) for k, v in dict(raw).items()}


def _gate_policy(**kw):
    from kernels.common.attention_bwd_plan import FEATURE_GROUPS, AttnBwdPolicy

    return AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True, **kw
    )


def _strides(layout: str, h_q: int, h_k: int, h_v: int, s_q: int, s_kv: int, d: int):
    """BHSD (``None``: the packed default) or BSHD element strides."""
    if layout == "bhsd":
        return None
    heads = {"q": h_q, "o": h_q, "do": h_q, "dq": h_q}
    heads.update({"k": h_k, "dk": h_k, "v": h_v, "dv": h_v})
    out = {}
    for n, h in heads.items():
        s = s_q if n in ("q", "o", "do", "dq") else s_kv
        out[n] = (s * h * d, d, h * d, 1)
    return out


def _band_forms(mask: str, s_q: int, s_kv: int) -> list[dict]:
    if mask == "none":
        return [{}]
    if mask == "causal_tl":
        return [{"causal": True}]
    if mask == "causal_br":
        return [{"causal_bottom_right": True}]
    if mask == "window":
        w = max(2, min(s_kv, s_q) // 2)
        return [
            {"left_bound": w - 1, "right_bound": 0, "bottom_right": True},
            {"left_bound": w, "right_bound": w, "bottom_right": False},
        ]
    raise ValueError(f"mask kind {mask!r} is not a mask kind")


def _kv_lengths(bucket, kn: int) -> tuple[int, ...]:
    """``s_kv`` at both edges of a kv-tile bucket (at the class's kN0)."""
    from kernels.common.attention_bwd_plan import ANY_BUCKET, KV_TILE_BUCKETS

    edges = []
    lo = 0
    for bound in KV_TILE_BUCKETS:
        edges.append((f"t{bound}", lo * kn + 1, bound * kn))
        lo = bound
    edges.append((f"t>{KV_TILE_BUCKETS[-1]}", lo * kn + 1, 2 * lo * kn))
    if bucket == ANY_BUCKET:
        return tuple(sorted({v for _b, a, z in edges for v in (a, z)}))
    for name, a, z in edges:
        if name == bucket:
            return tuple(sorted({a, z}))
    raise ValueError(f"kv-tile bucket {bucket!r} is not a bucket")


def _q_lengths(rel: str, s_kv: int) -> tuple[int, ...]:
    if rel == "square":
        return (s_kv,)
    if rel == "s_q_1":
        return (1,)
    if rel == "unequal":
        out = {2, 2 * s_kv, max(2, s_kv // 4)}
        return tuple(sorted(v for v in out if v not in (1, s_kv)))
    raise ValueError(f"length relation {rel!r} is not a relation")


def _num_cus(fill, grid: int) -> tuple:
    """``num_cus`` values at the edges of a grid-fill bucket for ``grid``
    CTAs (``under``: grid < CUs; ``full``: CUs <= grid < 2 CUs; ``over``:
    grid >= 2 CUs; ``unknown``: no value)."""
    from kernels.common.attention_bwd_plan import ANY_BUCKET

    vals = {
        "under": (grid + 1, 4 * grid),
        "full": (grid, grid // 2 + 1),
        "over": (grid // 2, max(1, grid // 8)),
        "unknown": (None,),
    }
    if fill == ANY_BUCKET:
        return tuple(dict.fromkeys(v for vs in vals.values() for v in vs))
    if fill not in vals:
        raise ValueError(f"fill bucket {fill!r} is not a bucket")
    return tuple(v for v in vals[fill] if v is None or v >= 1)


def _check_general_key(arch: str, key) -> None:
    from kernels.common.attention_bwd_plan import (
        HEAD_MODES,
        LENGTH_MODES,
        LENGTH_RELATIONS,
        MASK_KINDS,
    )

    if not (isinstance(key, tuple) and len(key) == GENERAL_KEY_LEN):
        raise ValueError(f"tuning key {key!r} is not a tuning key")
    if key[0] != arch or key[1] != "general":
        raise ValueError(f"tuning key {key!r} is not a {arch} general key")
    _a, _r, d, dtype, mask, head, rel, mode, sv = key[:9]
    if not (isinstance(d, int) and isinstance(dtype, str) and sv in (1, 8)):
        raise ValueError(f"tuning key {key!r} has no head size, dtype or stage_vec")
    for val, allowed in (
        (mask, MASK_KINDS),
        (head, HEAD_MODES),
        (rel, LENGTH_RELATIONS),
        (mode, LENGTH_MODES),
    ):
        if val not in allowed:
            raise ValueError(f"tuning key {key!r}: {val!r} not in {allowed}")


def general_key_requests(arch: str, key: tuple) -> list:
    """Representative requests of general tuning key ``key`` on ``arch``:
    every one has ``attn_bwd_tuning_key(req, arch) == key`` (``ANY_BUCKET``
    matching any bucket); see the module docstring for what they vary."""
    from kernels.common.attention_bwd import attn_bwd_default_geometry
    from kernels.common.attention_bwd_plan import (
        ANY_BUCKET,
        AttnBwdRequest,
        _dkv_atomic,
        _units,
        attn_bwd_tuning_key,
    )

    _check_general_key(arch, key)
    _a, _r, d, dtype, mask, head, rel, mode, sv, kvb, fill = key
    seq_mode = "thd" if mode == "thd" else "batched"
    kn = attn_bwd_default_geometry(arch, d, dtype, sv, seq_mode)["block_n"]
    layouts = ("bhsd",) if mode == "thd" else ("bhsd", "bshd")
    out, seen = [], set()

    def add(req):
        got = attn_bwd_tuning_key(req, arch)
        if got[:-2] != key[:-2]:
            return
        if kvb not in (ANY_BUCKET, got[-2]) or fill not in (ANY_BUCKET, got[-1]):
            return
        if req not in seen:
            seen.add(req)
            out.append(req)

    for h_q, h_k, h_v in _HEADS[head]:
        for s_kv in _kv_lengths(kvb, kn):
            for s_q in _q_lengths(rel, s_kv):
                for b in _BATCHES:
                    for li, layout in enumerate(layouts):
                        for bi, band in enumerate(_band_forms(mask, s_q, s_kv)):
                            kw: dict[str, Any] = dict(
                                b=b,
                                h_q=h_q,
                                h_k=h_k,
                                h_v=h_v,
                                s_q=s_q,
                                s_kv=s_kv,
                                d_qk=d,
                                d_v=d,
                                dtype=dtype,
                                tensor_alignment=16 if sv == 8 else 8,
                                strides=_strides(layout, h_q, h_k, h_v, s_q, s_kv, d),
                                **band,
                            )
                            if mode == "thd":
                                kw.update(layout="thd", strides=None)
                            elif mode == "padded":
                                kw.update(
                                    padding=True,
                                    has_seq_len_q=True,
                                    has_seq_len_kv=True,
                                )
                            base = AttnBwdRequest(**kw)
                            grid = b * _units(base, _dkv_atomic(base)) * -(-s_kv // kn)
                            for nc in _num_cus(fill, grid):
                                for xi, xcds in enumerate(_XCDS):
                                    # the layout / band / XCD variants on the
                                    # first batch only (they do not move the
                                    # grid)
                                    if (li or bi or xi) and b != _BATCHES[0]:
                                        continue
                                    add(
                                        dataclasses.replace(
                                            base, num_cus=nc, num_xcds=xcds
                                        )
                                    )
    return out


def _request_text(req) -> str:
    keep = ("b", "h_q", "h_k", "h_v", "s_q", "s_kv", "layout", "num_cus", "num_xcds")
    return ", ".join(f"{k}={getattr(req, k)!r}" for k in keep)


def check_entry_knobs(
    arch: str, key, knobs: Mapping[str, Any], axis_of: Callable[[str], Any]
) -> None:
    """Raise ``ValueError`` unless every knob of an entry is a registered axis
    (``axis_of(name)``; it raises ``KeyError`` / ``ValueError`` for an unknown
    name) with a value that is not ``not_built``, not ``gated`` and not known
    wrong on ``arch``."""
    if not isinstance(knobs, Mapping) or not all(isinstance(k, str) for k in knobs):
        raise ValueError(f"knobs {knobs!r} are not a name -> value mapping")
    for name, value in knobs.items():
        try:
            ax = axis_of(name)
        except (KeyError, ValueError):
            raise ValueError(
                f"entry {key!r}: {name!r} is not a registered knob"
            ) from None
        if value in ax.not_built:
            raise ValueError(f"entry {key!r}: {name}={value!r} is not built")
        if value in ax.gated:
            raise ValueError(f"entry {key!r}: {name}={value!r} is gated")
    wrong = known_wrong_findings(arch, knobs)
    if wrong:
        raise ValueError(f"entry {key!r}: known wrong on {arch}: {', '.join(wrong)}")


def check_general_entry(
    arch: str, key: tuple, knobs: Mapping[str, Any], *, limit: Optional[int] = None
) -> None:
    """Raise ``ValueError`` unless ``(key, knobs)`` is a general entry of
    ``arch`` whose knobs are registered values (:func:`check_entry_knobs`)
    and plan on every representative request of the key that plans without
    them (:func:`general_key_requests`; ``limit``: stop after that many
    planned representatives)."""
    from kernels.common.attention_bwd_plan import attn_bwd_plan

    _check_general_key(arch, key)
    check_entry_knobs(arch, key, knobs, backward_axis)
    base = _gate_policy()
    tuned = _gate_policy(tuned_table={key: dict(knobs)})
    planned = 0
    for req in general_key_requests(arch, key):
        try:
            attn_bwd_plan(req, arch, policy=base)
        except ValueError:
            continue  # not plannable without the entry: nothing to keep
        try:
            plan = attn_bwd_plan(req, arch, policy=tuned)
        except ValueError as ex:
            raise ValueError(
                f"tuning key {key!r}: knobs do not plan on ({_request_text(req)}): "
                f"{ex}"
            ) from ex
        if plan.route != "general":
            raise AssertionError("general entry check planned another route")
        planned += 1
        if limit is not None and planned >= limit:
            break
    if not planned:
        raise ValueError(f"tuning key {key!r} has no plannable representative request")


def _check_dense_key(arch: str, key) -> None:
    if not (isinstance(key, tuple) and len(key) == 2 and key[0] == arch):
        raise ValueError(f"dense key {key!r} is not a {arch} key")
    k = key[1]
    if not (isinstance(k, tuple) and len(k) == DENSE_KEY_LEN):
        raise ValueError(f"dense key {k!r} is not a dense key")
    s_q, s_kv, h_q, h_k, d = k[:5]
    if not all(isinstance(v, int) and v > 0 for v in (s_q, s_kv, h_q, h_k, d)):
        raise ValueError(f"dense key {k!r} has no shape")
    if k[6] not in ("bhsd", "bshd"):
        raise ValueError(f"dense key {k!r} has no dense layout")
    band = k[8]
    if not (isinstance(band, tuple) and len(band) == 3):
        raise ValueError(f"dense key {k!r} has no band")


def dense_key_requests(arch: str, dense_key: tuple, cus=(None,)) -> list:
    """Representative dense-eligible requests of ``dense_key`` on ``arch``:
    its shape over several batches, with and without ``num_xcds`` and with
    each ``num_cus`` of ``cus`` (``None``: no value; ``"grid"``: the edges
    around the default grid of the batch)."""
    from kernels.common.attention_bwd_plan import (
        AttnBwdRequest,
        _dense_key,
        dense_bwd_eligibility,
    )

    _check_dense_key(arch, (arch, dense_key))
    s_q, s_kv, h_q, h_k, d, dtype, layout, _mask, band = dense_key
    left, right, br = band
    out = []
    for b in (1, 2, 8):
        base = AttnBwdRequest(
            b=b,
            h_q=h_q,
            h_k=h_k,
            h_v=h_k,
            s_q=s_q,
            s_kv=s_kv,
            d_qk=d,
            d_v=d,
            dtype=dtype,
            strides=_strides(layout, h_q, h_k, h_k, s_q, s_kv, d),
            left_bound=left,
            right_bound=right,
            bottom_right=br,
        )
        if _dense_key(base) != dense_key or not dense_bwd_eligibility(base, arch)[0]:
            continue
        for nc in cus:
            for xcds in _XCDS:
                out.append(dataclasses.replace(base, num_cus=nc, num_xcds=xcds))
    return out


# num_cus values of the dense representatives: none, one CU (every grid
# overfills it) and a large device
_DENSE_CUS = (None, 1, 1024)


def check_dense_entry(
    arch: str,
    key: tuple,
    knobs: Mapping[str, Any],
    req=None,
    *,
    limit: Optional[int] = None,
    axis_of: Optional[Callable[[str], Any]] = None,
) -> None:
    """Raise ``ValueError`` unless ``(key, knobs)`` is a dense entry of
    ``arch`` (``key = (arch, dense key)``) whose knobs plan with the dense
    family's own rules on every representative request of the key that plans
    without the entry (:func:`dense_key_requests`) and, when given, on
    ``req`` (which must have the key). ``axis_of``: the dense registry's
    axis lookup (:func:`~.dense_tuning.dense_axis`); when given, the knobs
    must be registered values (:func:`check_entry_knobs`). ``limit``: stop
    after that many planned requests."""
    from kernels.common.attention_bwd_plan import _dense_key, attn_bwd_plan
    from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

    _check_dense_key(arch, key)
    if not isinstance(knobs, Mapping) or not all(isinstance(k, str) for k in knobs):
        raise ValueError(f"knobs {knobs!r} are not a name -> value mapping")
    if axis_of is not None:
        check_entry_knobs(arch, key, knobs, axis_of)
    reqs = dense_key_requests(arch, key[1], _DENSE_CUS)
    if req is not None:
        if _dense_key(req) != key[1]:
            raise ValueError(f"request key {_dense_key(req)!r} is not {key[1]!r}")
        reqs = [req] + reqs
    values = dict(knobs)
    # planned as the routing plans it: the entry is the dense tuning and the
    # key is in the dense pack
    tuned = _gate_policy(tuning=lambda *_a: dict(values), dense_pack=frozenset({key}))
    base = _gate_policy()
    planned = 0
    for i, r in enumerate(reqs):
        if i or req is None:
            try:
                attn_bwd_plan(r, arch, policy=base)
            except ValueError:
                continue  # not plannable without the entry
        try:
            attn_dense_bwd_plan(r, arch, policy=tuned)
        except ValueError as ex:
            raise ValueError(
                f"dense key {key!r}: knobs do not plan on ({_request_text(r)}): {ex}"
            ) from ex
        planned += 1
        if limit is not None and planned >= limit:
            break
    if not planned:
        raise ValueError(f"dense key {key!r} has no plannable representative request")


def dense_entries(
    arch: str, doc: Mapping | Iterable, requests: Optional[Mapping] = None
) -> dict:
    """``{(arch, dense key): knobs}`` of the sweep's dense table document
    (``{"entries": [{"arch", "key", "knobs", ...}]}`` or the entry list),
    each checked by :func:`check_dense_entry` (``requests``: ``{dense key:
    request}`` planned besides the representatives)."""
    entries = doc["entries"] if isinstance(doc, Mapping) else list(doc)
    out = {}
    for e in entries:
        key = (e["arch"], _tuple(e["key"]))
        knobs = _knobs(e["knobs"])
        req = None if requests is None else requests.get(key[1])
        check_dense_entry(arch, key, knobs, req)
        if key in out and out[key] != knobs:
            raise ValueError(f"dense key {key!r} has two entries")
        out[key] = knobs
    return out


def general_entries(arch: str, doc: Mapping | Iterable) -> dict:
    """``{tuning key: knobs}`` of a general table document (``{"entries":
    [{"key", "knobs"}]}`` or the entry list), each checked by
    :func:`check_general_entry`."""
    entries = doc["entries"] if isinstance(doc, Mapping) else list(doc)
    out = {}
    for e in entries:
        key = _tuple(e["key"])
        knobs = _knobs(e["knobs"])
        check_general_entry(arch, key, knobs)
        if key in out and out[key] != knobs:
            raise ValueError(f"tuning key {key!r} has two entries")
        out[key] = knobs
    return out


def _check_status_table(arch: str, route: str, table: Mapping) -> None:
    from kernels.common.attention_bwd_plan import PERF_STATUSES

    for pclass, status in table.items():
        if not (isinstance(pclass, tuple) and tuple(pclass[:2]) == (arch, route)):
            raise ValueError(f"status class {pclass!r} is not a {arch} {route} class")
        if status not in PERF_STATUSES or status == "unverified":
            # unverified is the default: a table lists measured outcomes only
            raise ValueError(
                f"status of {pclass!r}: {status!r} is not a measured status"
            )


def check_arch_tables(
    mod, *, dense_axis: Callable[[str], Any], limit: Optional[int] = 1
) -> None:
    """The import-time pass over one per-arch tuning module ``mod``
    (``ARCH``, ``GENERAL_TUNED_TABLE``, ``DENSE_TUNED``,
    ``GENERAL_PERF_STATUS``, ``DENSE_PERF_STATUS``): every entry's knobs are
    registered values and plan on the first ``limit`` plannable
    representatives of its key (``None``: every representative); the status
    tables name measured classes of the arch. Raises ``ValueError`` naming
    the module and the entry."""
    arch = mod.ARCH
    try:
        for key, knobs in mod.GENERAL_TUNED_TABLE.items():
            check_general_entry(arch, key, knobs, limit=limit)
        for key, knobs in mod.DENSE_TUNED.items():
            check_dense_entry(arch, key, knobs, limit=limit, axis_of=dense_axis)
        _check_status_table(arch, "general", mod.GENERAL_PERF_STATUS)
        _check_status_table(arch, "dense", mod.DENSE_PERF_STATUS)
    except ValueError as ex:
        raise ValueError(f"{mod.__name__}: {ex}") from ex
