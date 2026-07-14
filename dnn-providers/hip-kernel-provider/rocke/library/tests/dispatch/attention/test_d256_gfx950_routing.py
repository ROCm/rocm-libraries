# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Routing tests for the gfx950 bf16 D256 prefill dispatch candidate.

CPU-only: exercises the ``attention_gfx950_d256`` candidate's selection through
``dispatch_attention`` (filter -> priority -> pick). No GPU required.
"""

from __future__ import annotations

import unittest

from dispatch.attention import AttentionRequest, dispatch_attention


def _d256(arch="gfx950", **kw):
    base = dict(
        batch=2,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=4096,
        seqlen_k=4096,
        hdim_q=256,
        hdim_v=256,
        arch=arch,
        dtype="bf16",
        mask_type=1,  # causal prefill
    )
    base.update(kw)
    return AttentionRequest(**base)


def _routed_spec_id(req):
    """spec_id of the winning candidate, or None if nothing supports the req."""
    try:
        return dispatch_attention(req).candidate.spec_id
    except ValueError:
        return None


class TestD256Gfx950Routing(unittest.TestCase):
    def test_selects_gfx950_d256_for_cohort(self):
        r = dispatch_attention(_d256())
        self.assertEqual(r.candidate.spec_id, "gfx950_d256")
        self.assertEqual(r.spec.path, "2d")

    def test_outranks_generic_2d(self):
        # priority 5 must beat the generic unified_2d (priority 10).
        self.assertEqual(dispatch_attention(_d256()).candidate.priority, 5)

    def test_force_by_algorithm(self):
        r = dispatch_attention(_d256(algorithm="d256_gfx950"))
        self.assertEqual(r.candidate.spec_id, "gfx950_d256")

    def test_rejects_non_gfx950(self):
        self.assertNotEqual(_routed_spec_id(_d256(arch="gfx942")), "gfx950_d256")

    def test_rejects_non_bf16(self):
        self.assertNotEqual(_routed_spec_id(_d256(dtype="fp16")), "gfx950_d256")

    def test_rejects_non_d256(self):
        self.assertNotEqual(
            _routed_spec_id(_d256(hdim_q=128, hdim_v=128)), "gfx950_d256"
        )

    def test_rejects_decode(self):
        # q=1 decode over long kv routes to the 3D split-KV path, not our cohort.
        self.assertNotEqual(
            _routed_spec_id(
                _d256(batch=1, nhead_q=16, nhead_k=16, seqlen_q=1, seqlen_k=8192)
            ),
            "gfx950_d256",
        )

    def test_rejects_sliding_window(self):
        self.assertNotEqual(
            _routed_spec_id(_d256(sliding_window=256)), "gfx950_d256"
        )


if __name__ == "__main__":
    unittest.main()
