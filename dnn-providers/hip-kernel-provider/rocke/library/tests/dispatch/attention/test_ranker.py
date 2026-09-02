# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the attention dispatcher's engine-level ranker seam.

Mirrors the GEMM ranker coverage
(``platform/tests/dispatch/dispatch_tests/gemm/test_fp16_rcr.py``
``test_ranker_can_override_auto_priority`` and ``gemm/test_registry.py``
``test_select_rejects_ranker_returning_unsupported_candidate``).

Covers:
  - the default ``priority_ranker`` is behavior-preserving (identity over the
    already-priority-sorted candidates): a gfx942 fp16 MHA shape both
    ``gfx942_dense_pipe`` (priority 5) and ``unified_2d`` (priority 10) support
    routes to dense_pipe with no ranker supplied.
  - a caller-supplied ranker floats a lower-priority candidate to the front and
    ``dispatch_attention`` honors it (engine-level selection seam).
  - the registry safety invariant: a ranker that returns a candidate the request
    does not support raises.
"""

from __future__ import annotations

import os
import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionRequest,
    ATTENTION_REGISTRY,
    attention_candidates,
    dispatch_attention,
    priority_ranker,
)


def _gfx942_fp16_mha(**kw) -> AttentionRequest:
    # MHA (nhead_q == nhead_k), long context: supported by BOTH
    # gfx942_dense_pipe (priority 5) and unified_2d (priority 10), so the
    # winner is decided by ranking, not by a single eligible candidate.
    base = dict(
        batch=2,
        nhead_q=16,
        nhead_k=16,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="fp16",
    )
    base.update(kw)
    return AttentionRequest(**base)


class _Gfx942Arch:
    """Pin _RESOLVED_ATTENTION_ARCH to gfx942 and clear routing env overrides."""

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx942"
        self._env_keys = (
            "HIPDNN_GFX942_K_SLICED_RING",
            "HIPDNN_GFX942_FLASH_MLIM",
            "HIPDNN_GFX942_FLASH_WIDE",
            "HIPDNN_GFX942_BF16_CFVST",
            "HIPDNN_GFX942_BF16_WIDE",
        )
        self._old_env = {k: os.environ.get(k) for k in self._env_keys}
        for k in self._env_keys:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestPriorityRankerDefault(unittest.TestCase):
    def test_two_candidates_support_the_shape(self):
        # Guard the premise: the ranker test is only meaningful if >1 candidate
        # supports the request.
        with _Gfx942Arch():
            supported = {
                c.name for c in ATTENTION_REGISTRY.supported(_gfx942_fp16_mha())
            }
        self.assertIn("attention_gfx942_dense_pipe", supported)
        self.assertIn("attention_unified_2d", supported)

    def test_default_routes_to_priority_winner(self):
        # No ranker -> priority_ranker default -> lowest priority number wins.
        with _Gfx942Arch():
            r = dispatch_attention(_gfx942_fp16_mha())
        self.assertEqual(r.candidate.spec_id, "gfx942_dense_pipe")

    def test_priority_ranker_is_identity(self):
        with _Gfx942Arch():
            supported = ATTENTION_REGISTRY.supported(_gfx942_fp16_mha())
            ranked = priority_ranker(_gfx942_fp16_mha(), supported)
        self.assertEqual([c.name for c in ranked], [c.name for c in supported])

    def test_explicit_default_matches_implicit(self):
        with _Gfx942Arch():
            implicit = dispatch_attention(_gfx942_fp16_mha())
            explicit = dispatch_attention(_gfx942_fp16_mha(), ranker=priority_ranker)
        self.assertEqual(implicit.candidate.name, explicit.candidate.name)


class TestRankerOverride(unittest.TestCase):
    def test_ranker_can_override_priority(self):
        # Float unified_2d (priority 10) ahead of dense_pipe (priority 5).
        def prefer_unified_2d(_request, candidates):
            return sorted(candidates, key=lambda c: c.spec_id != "unified_2d")

        with _Gfx942Arch():
            r = dispatch_attention(_gfx942_fp16_mha(), ranker=prefer_unified_2d)
        self.assertEqual(r.candidate.spec_id, "unified_2d")
        self.assertEqual(r.kernel_id.spec_id, "unified_2d")

    def test_ranker_returning_unsupported_candidate_raises(self):
        # A ranker cannot smuggle in a candidate the request does not support.
        rogue = next(
            c for c in attention_candidates() if c.name == "attention_unified_3d"
        )

        def inject_rogue(_request, _candidates):
            return (rogue,)

        with _Gfx942Arch():
            with self.assertRaises(ValueError):
                dispatch_attention(_gfx942_fp16_mha(), ranker=inject_rogue)


if __name__ == "__main__":
    unittest.main()
