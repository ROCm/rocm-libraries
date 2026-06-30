# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Path-selection + coverage tests for the attention dispatcher family."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.attention import (
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)


def _attn(arch="gfx950", **kw):
    base = dict(
        batch=2,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
    )
    base.update(kw)
    return AttentionRequest(**base)


class TestAttentionDispatch(unittest.TestCase):
    def test_rejects_unsupported_head_size(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(hdim_q=96, hdim_v=96))

    def test_rejects_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(dtype="fp8"))

    def test_rejects_non_divisible_gqa(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(nhead_q=30, nhead_k=8))

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(arch="gfx000"))

    def test_short_kv_routes_2d(self):
        r = dispatch_attention(_attn(seqlen_q=512, seqlen_k=512))
        self.assertEqual(r.spec.path, "2d")
        self.assertEqual(r.candidate.spec_id, "unified_2d")

    def test_sliding_window_routes_2d(self):
        r = dispatch_attention(_attn(seqlen_q=128, seqlen_k=4096, sliding_window=256))
        self.assertEqual(r.spec.path, "2d")

    def test_long_kv_small_grid_routes_3d(self):
        # decode (q=1) long kv, small grid -> 3d split-KV.
        r = dispatch_attention(
            _attn(batch=1, nhead_q=16, nhead_k=16, seqlen_q=1, seqlen_k=8192)
        )
        self.assertEqual(r.spec.path, "3d")
        self.assertEqual(r.candidate.spec_id, "unified_3d")

    def test_large_grid_routes_2d(self):
        # many seqs/heads -> num_2d > target -> 2d even with long kv.
        r = dispatch_attention(
            _attn(batch=8, nhead_q=32, nhead_k=8, seqlen_q=1024, seqlen_k=1024)
        )
        self.assertEqual(r.spec.path, "2d")

    def test_path_candidates_are_mutually_exclusive(self):
        # Exactly one of (2d, 3d) supports any given problem.
        req = _attn(batch=1, nhead_q=16, nhead_k=16, seqlen_q=1, seqlen_k=8192)
        supported = [c for c in attention_candidates() if c.supports(req)[0]]
        self.assertEqual(len(supported), 1)

    def test_spec_records_dims(self):
        r = dispatch_attention(_attn(hdim_q=64, hdim_v=64, kv_block_size=32))
        self.assertEqual(r.spec.head_size, 64)
        self.assertEqual(r.spec.block_size, 32)

    def test_block_size_coverage(self):
        with self.assertRaises(ValueError):
            dispatch_attention(_attn(kv_block_size=128))  # not in {16,32,64}

    def test_unique_candidate_names(self):
        names = [c.name for c in attention_candidates()]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
