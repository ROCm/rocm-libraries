# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmarks/common/det_classify.py: which cached conv launches reduce
# deterministically, under "any" and "f32". Host-only.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.common.det_classify import (
    LOWP_PARTIAL_FIELDS,
    classify,
    is_deterministic,
    may_be_deterministic,
    pbd_slot_bytes,
)


def _ident(**kw):
    base = {
        "split_k": 1,
        "two_stage": False,
        "streamk": "off",
        "pp_streamk": 0,
        "tapk_pp_persist": 0,
        "pp_box_det": 0,
        "ws_store": 0,
        "split_det": 0,
        "split_streamk": False,
        "split_hybrid": 0,
    }
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "pbd, any_ok, f32_ok",
    [
        (0, False, False),
        (1, True, True),
        (2, True, False),
        (3, True, True),
        (4, True, False),
    ],
)
def test_dgrad_pp_box_det_modes(pbd, any_ok, f32_ok):
    # pp_box_det 2 / 4 keep bf16 slots: deterministic, but not an f32 fold.
    ident = _ident(split_k=8, pp_boxes=7, pp_box_det=pbd)
    assert is_deterministic("dgrad", ident, None, "any") is any_ok
    assert is_deterministic("dgrad", ident, 8, "any") is any_ok
    assert is_deterministic("dgrad", ident, None, "f32") is f32_ok
    assert is_deterministic("dgrad", ident, 8, "f32") is f32_ok


def test_lowp_fields_match_slot_bytes():
    assert [pbd_slot_bytes(m) for m in range(5)] == [4, 4, 2, 4, 2]
    lowp_pbd = {v for f, v in LOWP_PARTIAL_FIELDS if f == "pp_box_det"}
    assert lowp_pbd == {m for m in range(5) if pbd_slot_bytes(m) == 2}


@pytest.mark.parametrize(
    "ws_store, any_ok, f32_ok", [(0, False, False), (1, True, True), (2, True, False)]
)
def test_wgrad_ws_store(ws_store, any_ok, f32_ok):
    ident = _ident(split_k=2, two_stage=True, ws_store=ws_store, ws_replicas=8)
    assert is_deterministic("wgrad", ident, 4, "any") is any_ok
    assert is_deterministic("wgrad", ident, 4, "f32") is f32_ok
    if ws_store:
        # more slices than slabs: two slices share a slab
        assert not is_deterministic("wgrad", ident, 16, "any")


@pytest.mark.parametrize("split_det", [1, 2, 3])
def test_fwd_split_det(split_det):
    ident = _ident(split_k=2, split_tail=True, split_det=split_det)
    for sk in (None, 2, 3, 8):
        assert is_deterministic("fwd", ident, sk, "any")
        assert is_deterministic("fwd", ident, sk, "f32")


def test_fwd_two_way_rule():
    ident = _ident(split_k=2, split_tail=True)
    assert is_deterministic("fwd", ident, 2, "any")
    assert not is_deterministic("fwd", ident, 2, "f32")
    assert not is_deterministic("fwd", ident, 3, "any")
    assert not is_deterministic("fwd", ident, None, "any")
    assert classify("fwd", ident, 4)[0] is False


def test_binary_prefilter_keeps_two_way_capable_fwd_splits():
    # The binary-level prefilter must not drop a fwd split binary whose
    # degree-2 launches the per-degree rule accepts under "any".
    tail = _ident(split_k=2, split_tail=True)
    assert may_be_deterministic("fwd", tail, "any")
    assert not may_be_deterministic("fwd", tail, "f32")
    for kw in ({"split_streamk": True}, {"split_hybrid": 1}):
        assert not may_be_deterministic("fwd", _ident(split_k=2, **kw), "any")
    # Everything else follows the binary-level answer.
    assert may_be_deterministic("fwd", _ident(), "f32")
    assert not may_be_deterministic("dgrad", _ident(split_k=8), "any")
    assert not may_be_deterministic("dgrad", _ident(split_k=8, pp_box_det=4), "f32")
    assert may_be_deterministic("dgrad", _ident(split_k=8, pp_box_det=3), "f32")
