# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The split-KV 3D segment workspace is declined before it overruns i32.

``segm_output`` is ``[total_q, num_query_heads, num_segments, head_size]`` fp32
and both the segment kernel and the reduce address it with i32 element offsets,
so an index at or past ``2**31`` wraps negative and stores outside the
allocation. That is an illegal access rather than a wrong number, so the launch
has to refuse it on the host.

The bound is on the INDEX, not the element count, and the difference is one
element: a workspace of exactly ``2**31`` elements reaches index ``2**31 - 1``,
which is ``INT32_MAX``, and is the largest size that still works. Measured on
gfx942: the exactly-``2**31`` cases below complete, and ``total_q + 1`` faults
with ``hipErrorIllegalAddress`` in the segment kernel.

No GPU needed: the arithmetic and decision tests call the host-side guard
directly, and the launch-path tests reach it before any device work.

    python3 -m pytest rocke/library/tests/test_attention_3d_workspace_i32_guard.py
"""

from __future__ import annotations

import types
import unittest

import kernels.common.attention_unified as au

_LIMIT = 2**31


def _problem(total_q: int, hq: int, hd: int, **kw) -> "au.UnifiedAttentionProblem":
    base = dict(
        total_q=total_q,
        num_seqs=1,
        num_query_heads=hq,
        num_kv_heads=8,
        head_size=hd,
        block_size=16,
        max_seqlen_q=total_q,
        max_seqlen_k=2048,
        dtype="bf16",
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


class TestWorkspaceIndexArithmetic(unittest.TestCase):
    def test_max_index_is_numel_minus_one(self):
        p = _problem(65536, 32, 128)
        self.assertEqual(au._3d_workspace_max_index(p, 8), 65536 * 32 * 8 * 128 - 1)

    def test_exactly_2_31_elements_reaches_int32_max(self):
        # The boundary the ticket named as failing: it is in fact the largest
        # workspace that still fits, with nothing to spare.
        p = _problem(65536, 32, 128)
        self.assertEqual(au._3d_workspace_max_index(p, 8), _LIMIT - 1)


class TestWorkspaceGuard(unittest.TestCase):
    def test_accepts_exactly_int32_max(self):
        # Measured: this config completes on gfx942. Declining it would reject
        # a launch that works.
        au._check_3d_workspace_addr_width(_problem(65536, 32, 128), 8)

    def test_rejects_one_token_past_the_boundary(self):
        # Measured: this config faults with hipErrorIllegalAddress on gfx942.
        with self.assertRaises(NotImplementedError) as cm:
            au._check_3d_workspace_addr_width(_problem(65537, 32, 128), 8)
        msg = str(cm.exception)
        self.assertIn("i32", msg)
        self.assertIn("num_segments=8", msg)

    def test_rejection_names_the_segment_count_as_the_lever(self):
        # num_segments is the one factor a caller can change, so the message
        # has to point at it.
        with self.assertRaises(NotImplementedError) as cm:
            au._check_3d_workspace_addr_width(_problem(131072, 32, 128), 8)
        self.assertIn("Lower num_segments", str(cm.exception))

    def test_halving_the_segment_count_recovers_a_rejected_launch(self):
        p = _problem(131072, 32, 128)
        with self.assertRaises(NotImplementedError):
            au._check_3d_workspace_addr_width(p, 8)
        au._check_3d_workspace_addr_width(p, 4)

    def test_scales_with_every_axis(self):
        # Each axis multiplies the workspace, so each one alone can cross it.
        au._check_3d_workspace_addr_width(_problem(16384, 64, 256), 8)
        for bump in ("total_q", "hq", "hd"):
            with self.subTest(axis=bump):
                kw = dict(total_q=16384, hq=64, hd=256)
                kw[bump] *= 2
                with self.assertRaises(NotImplementedError):
                    au._check_3d_workspace_addr_width(
                        _problem(kw["total_q"], kw["hq"], kw["hd"]), 8
                    )


class TestReportedSuiteShapes(unittest.TestCase):
    """The two shapes the ticket reported as hanging.

    Both come from ``attention_shapes_llm.txt`` and both land on exactly
    ``2**31`` elements, so both reach ``INT32_MAX`` and neither overflows by the
    i32 mechanism this guard bounds. They are pinned so a later tightening to
    ``>= 2**31`` elements has to be a deliberate decision rather than a drift.

    OPEN: that leaves their reported hang unexplained. These assertions record
    where the shapes sit against the bound, which is arithmetic, and claim
    nothing about whether they run correctly: the boundary measurement behind
    this guard is a completion, not an fp32 parity run. If the hang is later
    traced to something else at this size, tightening here is the likely
    response and these tests are the thing to revisit.

    Note the suite is versioned and copies disagree on row count, so the
    indices below are from the 251-row snapshot.
    """

    # (name, total_q, num_query_heads, head_size, num_segments)
    _SHAPES = (
        ("idx 193 GLM-5.3 prefill", 16384, 64, 256, 8),
        ("idx 49 Llama-3.1-405B prefill", 8192, 128, 128, 16),
    )

    def test_both_land_exactly_on_int32_max(self):
        for name, tq, hq, hd, seg in self._SHAPES:
            with self.subTest(shape=name):
                p = _problem(tq, hq, hd)
                self.assertEqual(au._3d_workspace_max_index(p, seg), _LIMIT - 1)

    def test_neither_is_declined(self):
        for name, tq, hq, hd, seg in self._SHAPES:
            with self.subTest(shape=name):
                au._check_3d_workspace_addr_width(_problem(tq, hq, hd), seg)


class _PinArch:
    """Pin the launch arch so the path runs without a device."""

    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._resolve_attention_arch
        au._resolve_attention_arch = lambda _a=self.arch: _a
        return self

    def __exit__(self, *_):
        au._resolve_attention_arch = self._old


class _SegSpec:
    """Minimal stand-in for a tuning spec's kernel_spec."""

    def __init__(self, num_segments: int):
        self.num_segments = num_segments


class _TuningSpec:
    def __init__(self, num_segments: int):
        self.kernel_spec = _SegSpec(num_segments)

    def cache_key(self):
        return ("stub", self.kernel_spec.num_segments)


class TestLaunchPathReachesTheGuard(unittest.TestCase):
    """The guard is wired into the launch, not just importable.

    Checking the function alone cannot catch a deleted or misplaced call site:
    every other test here passes with the call removed, and the fault comes
    back with CI green. These drive ``_run_3d_tiled_on_stream`` itself.

    The guard is the third statement in that function and runs before any
    tensor is touched, so every tensor argument below is ``None``: reaching the
    raise is the assertion, and anything that got past it would fail on ``None``
    instead, which is still a failure.
    """

    def _launch(self, problem, tuning_spec=None):
        return au._run_3d_tiled_on_stream(
            problem=problem,
            q=None,
            k=None,
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=None,
            softcap=0.0,
            sinks=None,
            bt_stride=0,
            warmup=0,
            attempts=1,
            tuning_spec=tuning_spec,
        )

    def test_heuristic_segment_count_is_capped_not_declined(self):
        # Nobody chose the heuristic count, so a launch that would overrun runs
        # at a smaller split instead of raising. It fails later on the None
        # tensors, which is the point: the guard is not what stopped it.
        with _PinArch("gfx942"):
            with self.assertRaises(Exception) as cm:
                self._launch(_problem(2**20, 128, 256))
        self.assertNotIsInstance(cm.exception, NotImplementedError)

    def test_tuned_segment_count_is_checked_at_launch(self):
        # num_segments from the tuning spec: the branch a tuner drives, and the
        # one the reported hangs came through.
        with _PinArch("gfx942"):
            with self.assertRaises(NotImplementedError) as cm:
                self._launch(_problem(65537, 32, 128), tuning_spec=_TuningSpec(8))
        self.assertIn("num_segments=8", str(cm.exception))

    def test_launch_does_not_decline_the_accepted_boundary(self):
        # Exactly INT32_MAX must get past the guard. It fails later on the None
        # tensors, which is the point: the guard is not what stopped it.
        with _PinArch("gfx942"):
            with self.assertRaises(Exception) as cm:
                self._launch(_problem(65536, 32, 128), tuning_spec=_TuningSpec(8))
        self.assertNotIsInstance(cm.exception, NotImplementedError)


class TestSegmentCountIsCapped(unittest.TestCase):
    """The split rule never hands back a count the kernels cannot address.

    The count is a performance choice made without reference to the workspace it
    implies, so on a long prefill it can prefer a split whose last index is past
    i32. Refusing there would refuse a shape that runs at a smaller split, so the
    rule caps instead and the launch-time refusal is left to pinned counts.
    """

    # The reported case: gfx942 backend="auto" LSE prefill routes 3D, the rule
    # prefers 8 segments (index twice the bound) and 4 is exactly INT32_MAX.
    def _reported(self):
        return au.UnifiedAttentionProblem(
            total_q=65536,
            num_seqs=1,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=256,
            block_size=32,
            max_seqlen_q=65536,
            max_seqlen_k=65536,
            dtype="bf16",
            num_cus=304,
            emit_lse=True,
        )

    def test_reported_auto_route_caps_to_a_count_that_fits(self):
        p = self._reported()
        self.assertEqual(au.select_auto_path(p, "gfx942"), "3d")
        self.assertEqual(au._num_segments_unbounded(p, "gfx942"), 8)
        self.assertEqual(au._num_segments(p, "gfx942"), 4)
        self.assertEqual(au._3d_workspace_max_index(p, 4), _LIMIT - 1)

    def test_capped_count_passes_the_guard(self):
        p = self._reported()
        au._check_3d_workspace_addr_width(p, au._num_segments(p, "gfx942"))

    def test_cap_leaves_a_fitting_count_untouched(self):
        # Small enough that no split overruns: the rule's own answer stands,
        # so the cap cannot lower a working choice (including a non-power of
        # two, which the power-of-two fallback would otherwise round down).
        p = _problem(1024, 32, 128)
        self.assertEqual(
            au._num_segments(p, "gfx942"), au._num_segments_unbounded(p, "gfx942")
        )

    def test_cap_uses_the_shared_search(self):
        # One search for both workspaces, with the allowed extent passed in.
        from kernels.common.attention_dense_kv_split import max_segments_for_extent

        L = _LIMIT
        self.assertEqual(max_segments_for_extent(65536, 32, 256, 8, L), 4)
        self.assertEqual(max_segments_for_extent(65536, 32, 256, 4, L), 4)
        # At least one is always returned, even when a single segment overruns.
        self.assertEqual(max_segments_for_extent(2**40, 32, 256, 8, L), 1)

    def test_dense_route_keeps_its_stricter_extent(self):
        # The dense kernel pair is unmeasured at the boundary, so it stays one
        # element short of the 3D route rather than inheriting its bound.
        from kernels.common.attention_dense_kv_split import max_kv_segments_for_extent

        self.assertEqual(max_kv_segments_for_extent(1, 32768, 64, 128), 4)


class TestDispatcherSkipsOverBoundVariants(unittest.TestCase):
    """A fixed-segment 3D variant names its own count, so the split rule's cap
    never sees it. ``validate`` has to apply the bound, or the dispatcher
    selects a variant that can only fail at launch."""

    def _space(self, num_segments):
        from dispatch.attention.unified_rules import (
            AttentionGeometryVariant,
            UnifiedKernels,
            UnifiedSpace,
        )

        variant = AttentionGeometryVariant(
            arch="gfx942",
            path="3d",
            codepath="tiled_3d",
            builder_kind="tiled_3d",
            tile_policy="16x16",
            num_segments=num_segments,
        )
        space = UnifiedSpace(
            arch="gfx942",
            abi="unified",
            path="3d",
            variant_id=variant.variant_id,
            candidate_name="test_3d",
            variant=variant,
        )
        return space, UnifiedKernels(
            kernel_spec=types.SimpleNamespace(num_segments=num_segments)
        )

    def test_rejects_a_variant_whose_workspace_overruns(self):
        space, kernel = self._space(8)
        ok, why = space.validate(_problem(65536, 32, 256), kernel)
        self.assertFalse(ok)
        self.assertIn("num_segments=8", why)

    def test_accepts_the_same_variant_at_a_count_that_fits(self):
        space, kernel = self._space(4)
        ok, _ = space.validate(_problem(65536, 32, 256), kernel)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
