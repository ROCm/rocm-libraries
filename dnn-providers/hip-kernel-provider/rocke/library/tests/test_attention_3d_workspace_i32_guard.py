# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The split-KV 3D segment workspace stays inside i32 element addressing.

``segm_output`` is ``[total_q, num_query_heads, num_segments, head_size]`` fp32
and both the segment kernel and the reduce address it with i32 element offsets,
so an index at or past ``2**31`` wraps negative and stores outside the
allocation. That is an illegal access rather than a wrong number.

The bound is on the INDEX, not the element count, and the difference is one
element: a workspace of exactly ``2**31`` elements reaches index ``2**31 - 1``,
which is ``INT32_MAX``, and is the largest that still addresses. Measured on
gfx942: that size completes and one query token more faults with
``hipErrorIllegalAddress`` in the segment kernel. The measurement is a
completion, not an fp32 parity run.

Three mechanisms keep a launch inside the bound, and each is tested once here:
the split rule caps the count it picks, the dispatcher's ``validate`` skips a
fixed-segment variant that overruns, and the launch refuses what neither can
rescue. Every test below maps to a defect that occurred during review.

Two shapes the ticket reported as hanging (16384x64x256 at S=8, 8192x128x128 at
S=16) both compute to exactly ``2**31`` elements and so do not overflow by this
mechanism. Their hang is unexplained and tracked separately; nothing here
asserts anything about them.

No GPU needed: the launch-path tests reach the guard before any device work.

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


def _lse_problem(total_q, hq, hd, num_seqs=1, seqlen=None):
    """A shape ``backend="auto"`` routes to split-KV on gfx942.

    ``select_auto_path`` sends an LSE request the tiled 2D gate declines to
    split-KV, which is how the default route reaches these workspace sizes.
    Block size 32 and D256 are part of that gate, so they are not incidental.
    """
    seqlen = total_q // num_seqs if seqlen is None else seqlen
    return _problem(
        total_q,
        hq,
        hd,
        num_seqs=num_seqs,
        block_size=32,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        num_cus=304,
        emit_lse=True,
    )


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


class _TuningSpec:
    """Minimal stand-in for a dispatcher tuning spec."""

    def __init__(self, num_segments: int):
        self.kernel_spec = types.SimpleNamespace(num_segments=num_segments)

    def cache_key(self):
        return ("stub", self.kernel_spec.num_segments)


class TestTheBound(unittest.TestCase):
    def test_max_index_matches_the_workspace_specs(self):
        """The index is a hand-written product, for speed: both the split rule
        and ``validate`` call it per candidate, and reading it off the specs
        cost 400x more. This pins the fast form against the allocation it
        describes, so a dropped factor or a layout change fails here."""
        for tq, hq, hd, seg in ((4096, 32, 128, 8), (1, 1, 64, 1), (65536, 32, 256, 4)):
            with self.subTest(shape=(tq, hq, hd, seg)):
                p = _problem(tq, hq, hd)
                spec = au._attention_3d_workspace_specs(p, seg, None)[0]
                self.assertEqual(au._3d_workspace_max_index(p, seg), spec.numel() - 1)

    def test_accepts_int32_max_and_refuses_one_element_more(self):
        """Which side of INT32_MAX the threshold sits on is the whole point:
        one query token separates a size that runs on gfx942 from one that
        faults."""
        fits = _problem(65536, 32, 128)
        self.assertEqual(au._3d_workspace_max_index(fits, 8), _LIMIT - 1)
        au._check_3d_workspace_addr_width(fits, 8)

        with self.assertRaises(NotImplementedError) as cm:
            au._check_3d_workspace_addr_width(_problem(65537, 32, 128), 8)
        self.assertIn("num_segments=8", str(cm.exception))


class TestTheCap(unittest.TestCase):
    def test_caps_only_when_the_preferred_count_overruns(self):
        """The reported auto-route shape prefers 8 segments, twice the bound,
        and 4 is exactly INT32_MAX. A count that already fits must come back
        untouched, including a non-power-of-two one, which an unconditional
        power-of-two search would round down."""
        over = _lse_problem(65536, 32, 256)
        self.assertEqual(au._num_segments_unbounded(over, "gfx942"), 8)
        self.assertEqual(au._num_segments(over, "gfx942"), 4)

        fits = _problem(1024, 32, 128)
        self.assertEqual(
            au._num_segments(fits, "gfx942"),
            au._num_segments_unbounded(fits, "gfx942"),
        )

    def test_dense_route_keeps_its_stricter_extent(self):
        """The cap shares the dense bodies' search but not their bound: the
        dense kernel pair is unmeasured at the boundary, so its routing must
        not move with this change."""
        from kernels.common.attention_dense_kv_split import max_kv_segments_for_extent

        self.assertEqual(max_kv_segments_for_extent(1, 32768, 64, 128), 4)


class TestTheDispatcher(unittest.TestCase):
    def test_validate_skips_a_variant_whose_workspace_overruns(self):
        """A fixed-segment variant names its own count, so the split rule's cap
        never sees it. Without this the dispatcher selects a variant that can
        only fail at launch."""
        from dispatch.attention.unified_rules import (
            AttentionGeometryVariant,
            UnifiedKernels,
            UnifiedSpace,
        )

        def space_for(num_segments):
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
            kernel = UnifiedKernels(
                kernel_spec=types.SimpleNamespace(num_segments=num_segments)
            )
            return space, kernel

        problem = _problem(65536, 32, 256)
        space, kernel = space_for(8)
        ok, why = space.validate(problem, kernel)
        self.assertFalse(ok)
        self.assertIn("num_segments=8", why)

        space, kernel = space_for(4)
        self.assertTrue(space.validate(problem, kernel)[0])


class TestTheLaunch(unittest.TestCase):
    """The guard is wired into the launch, not just importable.

    Testing the function alone cannot catch a deleted or misplaced call site:
    every other test here passes with the call removed. These drive
    ``_run_3d_tiled_on_stream`` itself. The guard runs before any tensor is
    touched, so every tensor argument is ``None``: reaching (or not reaching)
    the raise is the assertion, and anything that got past it fails on ``None``
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

    def test_a_capped_count_launches_at_the_capped_value(self):
        """Nobody chose the heuristic count, so a preferred count that would
        overrun runs at a smaller one rather than raising."""
        used = []
        real = au._num_segments
        au._num_segments = lambda prob, arch: used.append(real(prob, arch)) or used[-1]
        try:
            with _PinArch("gfx942"):
                with self.assertRaises(Exception) as cm:
                    self._launch(_lse_problem(65536, 32, 256))
        finally:
            au._num_segments = real
        self.assertNotIsInstance(cm.exception, NotImplementedError)
        self.assertTrue(used)
        self.assertEqual(set(used), {4})

    def test_refuses_when_even_one_segment_overruns(self):
        """The cap floors at one segment, so a long enough prefill overruns
        with nothing to fall back to. Capping cannot save this one."""
        p = _lse_problem(32 * 8193, 32, 256, num_seqs=32, seqlen=8193)
        self.assertEqual(au._num_segments(p, "gfx942"), 1)
        with _PinArch("gfx942"):
            with self.assertRaises(NotImplementedError) as cm:
                self._launch(p)
        self.assertIn("i32", str(cm.exception))

    def test_refuses_a_pinned_count_outright(self):
        """A tuner chose this one, and silently running a different count
        would make the measurement meaningless."""
        with _PinArch("gfx942"):
            with self.assertRaises(NotImplementedError) as cm:
                self._launch(_problem(65537, 32, 128), tuning_spec=_TuningSpec(8))
        self.assertIn("num_segments=8", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
