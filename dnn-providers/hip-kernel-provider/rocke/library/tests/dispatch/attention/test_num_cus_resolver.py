# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the dispatch-layer num_cus resolver + target_ctas knob.

``num_cus`` drives 2D<->3D routing and the 3D segment count. It historically
defaulted to a stale 120, under-subscribing the device (e.g. 304-CU gfx942,
~256-CU gfx950). The resolver turns the sentinel default into the live device CU
count behind an explicit-caller override seam. Auto-resolution covers the arches
in ``_AUTO_RESOLVE_ARCHS`` (gfx942 and gfx950); every other arch keeps the legacy
120. gfx950's segment clamp is conservative -- the num_cus bump never raises an
already-3D split. ``target_ctas`` is the direct device-subscription target
override: when > 0 it replaces ``num_cus * 4`` for routing/segmentation without a
device CU count. These tests mock the device query/arch (no GPU) and assert the
resolution order plus the downstream routing/segment effects.
"""
from __future__ import annotations

import dispatch.attention as A
import dispatch.attention.common as AC
from dispatch.attention import AttentionRequest, _resolve_num_cus
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
        self._attrs = []

    def attr(self, obj, name, val):
        self._attrs.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def restore(self):
        for obj, name, old in reversed(self._attrs):
            setattr(obj, name, old)


def test_gfx942_device_query():
    """gfx942 on a gfx942 box -> the live CU count (device-dependent within gfx942)."""
    p = _Patch()
    try:
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx942")
        p.attr(AC, "_device_num_cus", lambda: 304)
        assert _resolve_num_cus(_req(num_cus=0)) == 304
        p.attr(AC, "_device_num_cus", lambda: 228)  # smaller-CU gfx942 variant
        assert _resolve_num_cus(_req(num_cus=0)) == 228
    finally:
        p.restore()


def test_gfx942_fallback_off_box():
    """gfx942 request off-box (no visible gfx942 device) -> legacy 120 (matches develop), NOT the box's count, NOT a 304 guess."""
    p = _Patch()
    try:
        p.attr(hipm, "get_device_arch", lambda *a, **k: None)  # no gfx942 device
        p.attr(
            AC, "_device_num_cus", lambda: 256
        )  # e.g. a gfx950 box; must NOT be consulted
        assert (
            _resolve_num_cus(_req(num_cus=0)) == 120
        )  # legacy fallback, not 256, not 304
    finally:
        p.restore()


def test_gfx950_device_query():
    """gfx950 on a gfx950 box -> the live CU count (256 on a 256-CU part)."""
    p = _Patch()
    try:
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx950")
        p.attr(AC, "_device_num_cus", lambda: 256)
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx950")) == 256
        p.attr(AC, "_device_num_cus", lambda: 208)  # smaller-CU gfx950 variant
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx950")) == 208
    finally:
        p.restore()


def test_gfx950_fallback_off_box():
    """gfx950 request off-box (running box != gfx950) -> legacy 120, never the wrong device's count."""
    p = _Patch()
    try:
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx942")  # a gfx942 box
        p.attr(AC, "_device_num_cus", lambda: 304)  # must NOT be consulted
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx950")) == 120
        p.attr(hipm, "get_device_arch", lambda *a, **k: None)  # no visible device
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx950")) == 120
    finally:
        p.restore()


def test_non_autoresolve_archs_keep_legacy_120():
    """Archs outside the auto-resolve set keep the legacy 120 AND never touch the
    device CU query (the arch gate short-circuits before _device_num_cus)."""
    p = _Patch()
    calls = {"n": 0}

    def _spy():
        calls["n"] += 1
        return 256  # a value that must never be used for a non-auto arch

    try:
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx90a")
        p.attr(AC, "_device_num_cus", _spy)
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx90a")) == 120
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx1250")) == 120
        assert _resolve_num_cus(_req(num_cus=0, arch="gfxZZZ")) == 120
        assert calls["n"] == 0, "device CU query consulted for a non-auto-resolve arch"
    finally:
        p.restore()


def test_explicit_caller_wins_any_arch():
    """An explicit caller value beats the device query, on any arch."""
    p = _Patch()
    try:
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx942")
        p.attr(AC, "_device_num_cus", lambda: 999)  # must NOT be consulted
        assert _resolve_num_cus(_req(num_cus=200)) == 200
        assert _resolve_num_cus(_req(num_cus=200, arch="gfx950")) == 200
    finally:
        p.restore()


def _prob(num_cus, *, nq=64, nk=8, D=64, kv=8192, batch=64, tctas=0, arch=None):
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
        num_cus=num_cus,
        target_ctas=tctas,
        arch=arch,
    )


def test_routing_scales_with_num_cus():
    """The resolved count changes routing: an under-filled grid flips 2D->3D."""
    # b64 GQA-64/8 D64 kv8192: num_2d=768 -> 2D at 120 (target 480), 3D at 304 (target 1216)
    assert _prob(120).select_path() == "2d"
    assert _prob(304).select_path() == "3d"


def test_target_ctas_overrides_effective_target():
    """target_ctas (>0) is the effective routing/segment target, bypassing num_cus*4."""
    assert _prob(120)._effective_target_ctas == 480  # auto: 120*4
    assert _prob(120, tctas=1216)._effective_target_ctas == 1216  # override beats 480
    assert _prob(304, tctas=99)._effective_target_ctas == 99  # override beats 1216
    assert _prob(120, tctas=0)._effective_target_ctas == 480  # 0 => auto


def test_target_ctas_flips_routing_without_num_cus():
    """Setting target_ctas alone flips 2D->3D at fixed num_cus (the knob's purpose)."""
    assert _prob(120).select_path() == "2d"  # auto target 480, num_2d=768 -> 2D
    assert _prob(120, tctas=1216).select_path() == "3d"  # same num_cus, pinned target


def test_target_ctas_threaded_through_problem():
    """AttentionRequest.target_ctas reaches the built problem; the resolver ignores it."""
    prob = A._problem(_req(num_cus=200, target_ctas=1216))
    assert prob.target_ctas == 1216
    assert prob.num_cus == 200  # num_cus still resolved independently
    assert prob._effective_target_ctas == 1216
    prob2 = A._problem(_req(num_cus=200, target_ctas=0))
    assert prob2.target_ctas == 0  # unset => auto
    assert prob2._effective_target_ctas == 800  # 200*4


def test_problem_lowercases_arch_for_the_clamp():
    """_problem normalizes req.arch to lowercase. The segment clamp compares
    against lowercase arch literals ("gfx942"/"gfx950"), so a mixed-case request
    arch must not silently skip the clamp on the dispatch path."""
    for raw in ("GFX950", "Gfx950", "gfx950"):
        assert A._problem(_req(num_cus=200, arch=raw)).arch == "gfx950", raw
    # Driven through _problem + _num_segments, request casing must not change the
    # segment count: both normalize to gfx950 and take the same clamp.
    p_mixed = A._problem(_req(num_cus=256, arch="GFX950"))
    p_lower = A._problem(_req(num_cus=256, arch="gfx950"))
    assert au._num_segments(p_mixed) == au._num_segments(p_lower)


def test_segments_bounded_after_bump():
    """The num_cus bump must not over-split D128 decode: clamp == pre-bump."""
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
        def qprob(num_cus):
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
                num_cus=num_cus,
            )

        q120 = au._num_segments(qprob(120))
        q304 = au._num_segments(qprob(304))
        assert q120 == q304, f"q>1 clamp must hold: {q120} -> {q304}"
    finally:
        au._RESOLVED_ATTENTION_ARCH = None
        p.restore()


def test_gfx950_segments_conservatively_clamped():
    """Conservative gfx950 clamp: the num_cus bump never raises the 3D split for
    ANY already-3D shape (including D64 / long-kv that gfx942 leaves uncapped)."""
    p = _Patch()
    try:
        au._RESOLVED_ATTENTION_ARCH = None
        p.attr(au, "_resolve_attention_arch", lambda: "gfx950")
        # Live gfx950 count ~256; every case must equal its num_cus=120 split.
        cases = [
            dict(nq=32, nk=8, D=128, kv=2048, batch=1),
            dict(nq=32, nk=8, D=128, kv=8192, batch=1),
            dict(
                nq=32, nk=8, D=128, kv=32768, batch=1
            ),  # gfx942 uncaps this; gfx950 clamps it
            dict(
                nq=32, nk=4, D=64, kv=8192, batch=1
            ),  # gfx942 uncaps D64; gfx950 clamps it
            dict(nq=32, nk=4, D=64, kv=32768, batch=1),
        ]
        for c in cases:
            s120 = au._num_segments(_prob(120, **c))
            s256 = au._num_segments(_prob(256, **c))
            assert s120 == s256, f"gfx950 over-split {c}: {s120} -> {s256}"
    finally:
        au._RESOLVED_ATTENTION_ARCH = None
        p.restore()


def test_partition_floor_is_gfx950_only():
    """A partitioned device (CPX / NPS4) reports the PARTITION CU count (~32).

    gfx950 floors that to the legacy 120 baseline so the routing target never
    drops below 480 and flips already-3D shapes back to 2D -- the gfx950 segment
    clamp's byte-identical guarantee is defined against that baseline.

    gfx942 is NOT floored: it shipped unfloored on develop, and flooring would
    silently change its 2D/3D routing and segment counts on partitioned parts.
    This PR targets gfx950 only, so gfx942 must resolve exactly as develop does.
    """
    p = _Patch()
    try:
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx950")
        p.attr(AC, "_device_num_cus", lambda: 32)  # a CPX/NPS4 partition
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx950")) == 120
        # A full-device count is above the floor and passes through unchanged.
        p.attr(AC, "_device_num_cus", lambda: 256)
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx950")) == 256

        # gfx942 on a partitioned gfx942 box: raw live count, no floor.
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx942")
        for partition_cus in (32, 38, 64):
            p.attr(AC, "_device_num_cus", lambda n=partition_cus: n)
            assert _resolve_num_cus(_req(num_cus=0, arch="gfx942")) == partition_cus
        # ...and a full gfx942 part is unaffected either way.
        p.attr(AC, "_device_num_cus", lambda: 304)
        assert _resolve_num_cus(_req(num_cus=0, arch="gfx942")) == 304
    finally:
        p.restore()


def test_gfx942_partition_routing_matches_develop():
    """The floor scoping is observable downstream, not just in the resolver.

    On a partitioned gfx942 the resolved count feeds select_path/_num_segments.
    Pin the develop behaviour for the shapes a 120 floor would have moved: at 38
    CUs they must stay on the path/split that 38 CUs produces, NOT the 120 one.
    """
    p = _Patch()
    try:
        au._RESOLVED_ATTENTION_ARCH = None
        p.attr(au, "_resolve_attention_arch", lambda: "gfx942")
        p.attr(hipm, "get_device_arch", lambda *a, **k: "gfx942")
        p.attr(AC, "_device_num_cus", lambda: 38)  # CPX partition
        for shape in (
            dict(nhead_q=32, nhead_k=8, hdim_q=128, hdim_v=128, seqlen_k=8192, batch=1),
            dict(nhead_q=32, nhead_k=8, hdim_q=128, hdim_v=128, seqlen_k=8192, batch=16),
            dict(nhead_q=32, nhead_k=4, hdim_q=64, hdim_v=64, seqlen_k=32768, batch=4),
            dict(nhead_q=32, nhead_k=8, hdim_q=128, hdim_v=128, seqlen_k=4096, batch=32),
        ):
            resolved = A._problem(_req(num_cus=0, arch="gfx942", **shape))
            unfloored = A._problem(_req(num_cus=38, arch="gfx942", **shape))
            assert resolved.num_cus == 38, shape
            assert resolved.select_path() == unfloored.select_path(), shape
            assert au._num_segments(resolved) == au._num_segments(unfloored), shape
    finally:
        au._RESOLVED_ATTENTION_ARCH = None
        p.restore()


def test_segment_clamp_keys_on_request_arch():
    """The split-KV clamp keys on the arch the problem TARGETS, not the running
    box. When ``problem.arch`` is set the running-box resolver is never consulted,
    so an off-box build targeting one arch can't pick up another arch's clamp."""
    p = _Patch()
    calls = {"n": 0}

    def _spy():
        calls["n"] += 1
        return "gfx950"  # the running box

    shape = dict(nq=32, nk=4, D=64, kv=32768, batch=1)
    try:
        au._RESOLVED_ATTENTION_ARCH = None
        p.attr(au, "_resolve_attention_arch", _spy)
        # problem.arch set -> resolver is never consulted.
        au._num_segments(_prob(256, arch="gfx942", **shape))
        assert calls["n"] == 0, "running-box arch consulted despite problem.arch set"
        # problem.arch unset -> falls back to the running-box resolver.
        au._num_segments(_prob(256, **shape))
        assert calls["n"] >= 1, "running-box resolver not used as fallback"
    finally:
        au._RESOLVED_ATTENTION_ARCH = None
        p.restore()


def test_every_auto_resolve_arch_clamps_the_split():
    """Membership in _AUTO_RESOLVE_ARCHS grants an arch the num_cus bump; the
    matching per-arch branch in ``_num_segments`` is what keeps that bump from
    over-splitting already-3D shapes. The two live in different modules and are
    kept in sync by hand, so pin the other half of the bargain here: an arch
    added to the set without a clamp fails this test instead of silently
    shipping an uncapped split.

    The shape is chosen so the unclamped formula blows past the safe ceiling
    (raw 128 vs pre-bump 64) AND so gfx942's head-size-specific branch also
    fires -- D128 decode at kv <= 2048. A shape gfx942 deliberately leaves
    uncapped (D256, or long-kv D128) would fail for a legitimate reason."""
    shape = dict(nq=8, nk=8, D=128, kv=2048, batch=1)
    for arch in sorted(AC._AUTO_RESOLVE_ARCHS):
        prob = _prob(256, arch=arch, **shape)
        raw = prob.select_3d()[0].NUM_SEGMENTS_PER_SEQ
        ceiling = au._pre_bump_segments(prob)
        assert raw > ceiling, (
            f"test shape no longer exercises the clamp on {arch}: "
            f"raw {raw} <= ceiling {ceiling}"
        )
        assert au._num_segments(prob) <= ceiling, (
            f"{arch} is in _AUTO_RESOLVE_ARCHS but has no segment clamp: "
            f"split {au._num_segments(prob)} exceeds the pre-bump ceiling {ceiling}"
        )


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
