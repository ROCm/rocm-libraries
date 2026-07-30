# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the dispatch-layer num_sms resolver (AICK-1722).

``num_sms`` drives 2D<->3D routing and the 3D segment count. It historically
defaulted to a stale 120, under-subscribing gfx942 (MI300X = 304 CUs). The
resolver turns the sentinel default into the live gfx942 CU count behind an
override seam. Auto-resolution is scoped to gfx942 for now; other archs keep the
legacy 120 (Future Scope). These tests mock the device query/arch (no GPU) and
assert the resolution order plus the downstream routing/segment effects.
"""
from __future__ import annotations

import os

import dispatch.attention as A
from dispatch.attention import AttentionRequest, _resolve_num_sms
from kernels.common.attention_unified import UnifiedAttentionProblem
import kernels.common.attention_unified as au
import rocke.runtime.hip_module as hipm


def _req(**kw):
    d = dict(
        batch=64,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1,
        seqlen_k=8192,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
    )
    d.update(kw)
    return AttentionRequest(**d)


class _Patch:
    """Minimal save/restore so the file runs under pytest AND as a script."""

    def __init__(self):
        self._env = {}
        self._attrs = []

    def env(self, k, v):
        self._env.setdefault(k, os.environ.get(k))
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    def attr(self, obj, name, val):
        self._attrs.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def restore(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for obj, name, old in reversed(self._attrs):
            setattr(obj, name, old)


def test_gfx942_device_query():
    """gfx942 on a gfx942 box -> the live CU count (228 MI300A / 304 MI300X)."""
    p = _Patch()
    try:
        p.env("ROCKE_NUM_SMS", None)
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx942")
        p.attr(A, "_device_num_cus", lambda: 304)
        assert _resolve_num_sms(_req(num_sms=0)) == 304
        p.attr(A, "_device_num_cus", lambda: 228)  # MI300A
        assert _resolve_num_sms(_req(num_sms=0)) == 228
    finally:
        p.restore()


def test_gfx942_fallback_off_box():
    """gfx942 request on a non-gfx942 / CPU box -> the 304 constant, not the box's count."""
    p = _Patch()
    try:
        p.env("ROCKE_NUM_SMS", None)
        p.attr(hipm, "get_device_arch", lambda *a, **k: None)  # no gfx942 device
        p.attr(A, "_device_num_cus", lambda: 256)  # e.g. a gfx950 box
        assert _resolve_num_sms(_req(num_sms=0)) == 304  # constant, not 256
    finally:
        p.restore()


def test_other_archs_keep_legacy_120():
    """Non-gfx942 archs are NOT auto-resolved yet (Future Scope) -> legacy 120."""
    p = _Patch()
    try:
        p.env("ROCKE_NUM_SMS", None)
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx950")
        p.attr(A, "_device_num_cus", lambda: 256)
        assert _resolve_num_sms(_req(num_sms=0, arch="gfx950")) == 120
        assert _resolve_num_sms(_req(num_sms=0, arch="gfx90a")) == 120
        assert _resolve_num_sms(_req(num_sms=0, arch="gfxZZZ")) == 120
    finally:
        p.restore()


def test_explicit_caller_wins_any_arch():
    """An explicit caller value beats the device query, on any arch."""
    p = _Patch()
    try:
        p.env("ROCKE_NUM_SMS", None)
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx942")
        p.attr(A, "_device_num_cus", lambda: 999)  # must NOT be consulted
        assert _resolve_num_sms(_req(num_sms=200)) == 200
        assert _resolve_num_sms(_req(num_sms=200, arch="gfx950")) == 200
    finally:
        p.restore()


def test_env_pin_wins():
    """ROCKE_NUM_SMS pins deterministically over explicit value + device, any arch."""
    p = _Patch()
    try:
        p.env("ROCKE_NUM_SMS", "77")
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx942")
        p.attr(A, "_device_num_cus", lambda: 304)
        assert _resolve_num_sms(_req(num_sms=200)) == 77
        assert _resolve_num_sms(_req(num_sms=0)) == 77
        assert _resolve_num_sms(_req(num_sms=0, arch="gfx950")) == 77
    finally:
        p.restore()


def test_ignores_bad_env():
    """A non-integer / non-positive env pin is ignored, not fatal."""
    p = _Patch()
    try:
        p.attr(hipm, "get_device_arch", lambda *a, **k: None)
        p.attr(A, "_device_num_cus", lambda: None)
        p.env("ROCKE_NUM_SMS", "garbage")
        assert _resolve_num_sms(_req(num_sms=0)) == 304  # gfx942 fallback
        p.env("ROCKE_NUM_SMS", "0")
        assert _resolve_num_sms(_req(num_sms=0)) == 304
    finally:
        p.restore()


def _prob(nsms, *, nq=64, nk=8, D=64, kv=8192, batch=64):
    return UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=nq,
        num_kv_heads=nk,
        head_size=D,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=kv,
        dtype="bf16",
        sliding_window=0,
        use_sinks=False,
        num_sms=nsms,
    )


def test_routing_scales_with_num_sms():
    """The resolved count changes routing: an under-filled grid flips 2D->3D."""
    # b64 GQA-64/8 D64 kv8192: num_2d=768 -> 2D at 120 (target 480), 3D at 304 (target 1216)
    assert _prob(120).select_path() == "2d"
    assert _prob(304).select_path() == "3d"


def test_segments_bounded_after_bump():
    """The num_sms bump must not over-split D128 decode: clamp == pre-bump."""
    p = _Patch()
    try:
        au._RESOLVED_ATTENTION_ARCH = None
        p.attr(au, "_resolve_attention_arch", lambda: "gfx942")
        # decode D128 kv8192: the bump is clamped -> no over-split (s120 == s304)
        s120 = au._num_segments(_prob(120, nq=32, nk=8, D=128, kv=8192, batch=1))
        s304 = au._num_segments(_prob(304, nq=32, nk=8, D=128, kv=8192, batch=1))
        assert s120 == s304, f"bump over-split D128 decode: {s120} -> {s304}"
        # kv boundary: 16385 and 32767 must STILL clamp (only kv>=32768 uncapped)
        for kv in (16385, 32767):
            b120 = au._num_segments(_prob(120, nq=32, nk=8, D=128, kv=kv, batch=1))
            b304 = au._num_segments(_prob(304, nq=32, nk=8, D=128, kv=kv, batch=1))
            assert b120 == b304, f"D128 kv={kv} must clamp: {b120} -> {b304}"
        # kv>=32768: uncapped -> the bump IS allowed to raise the split
        u120 = au._num_segments(_prob(120, nq=32, nk=8, D=128, kv=32768, batch=1))
        u304 = au._num_segments(_prob(304, nq=32, nk=8, D=128, kv=32768, batch=1))
        assert u304 > u120, f"kv32768 should scale: {u120} -> {u304}"

        # q>1 (prefill / spec-decode) D128: the else-branch clamp also holds
        def qprob(nsms):
            return UnifiedAttentionProblem(
                total_q=4,
                num_seqs=1,
                num_query_heads=32,
                num_kv_heads=8,
                head_size=128,
                block_size=16,
                max_seqlen_q=4,
                max_seqlen_k=8192,
                dtype="bf16",
                sliding_window=0,
                use_sinks=False,
                num_sms=nsms,
            )

        q120 = au._num_segments(qprob(120))
        q304 = au._num_segments(qprob(304))
        assert q120 == q304, f"q>1 clamp must hold: {q120} -> {q304}"
    finally:
        au._RESOLVED_ATTENTION_ARCH = None
        p.restore()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
