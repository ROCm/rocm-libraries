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
gfx942 (MI300A, develop): the exactly-``2**31`` cases below complete, and
``total_q + 1`` faults with ``hipErrorIllegalAddress`` in the segment kernel.

No GPU needed: these call the host-side guard directly.

    python3 -m pytest rocke/library/tests/test_attention_3d_workspace_i32_guard.py
"""

from __future__ import annotations

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
    ``2**31`` elements, so both reach ``INT32_MAX`` and neither overflows. They
    are pinned here so that if the guard is ever tightened to ``>= 2**31``
    elements, these stop the change: it would decline two shapes that run.

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


if __name__ == "__main__":
    unittest.main()
