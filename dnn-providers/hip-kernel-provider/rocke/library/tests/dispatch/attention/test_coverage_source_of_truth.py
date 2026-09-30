# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The dispatcher must not restate what a kernel covers -- it must import it.

A `Capability` exists to make coverage filterable and queryable without probing
the kernel. That only works if the capability and the kernel agree, and the
tempting way to write one -- copying the head sizes out of the backend's
predicate -- fails asymmetrically:

* Kernel *loses* coverage: harmless. The capability admits, then the residual
  `supports_native_unified_attention` check inside `support()` rejects. A
  redundant gate, not a wrong answer.
* Kernel *gains* coverage: silent. The prefilter rejects a shape the kernel
  had since learned to run, and nothing anywhere reports that the new coverage
  is unreachable. Nobody sees a failure; the feature just does not exist.

So these tests assert the capability's sets are literally the kernel's objects,
and separately sweep shapes to confirm the prefilter never turns down something
the backend accepts.
"""

from __future__ import annotations

import unittest

from dispatch.attention import ATTENTION_REGISTRY, AttentionRequest
from dispatch.attention.common import _problem
from kernels.common import attention_unified as au
from kernels.gfx1250 import wmma_attention_fwd as wmma


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=8,
        nhead_k=8,
        seqlen_q=256,
        seqlen_k=256,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        dtype="fp16",
        kv_block_size=16,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _shape_range(candidate, dim):
    for rng in candidate.capability.shapes:
        if dim in rng.names():
            return rng
    raise AssertionError(f"{candidate.name} declares no bound on {dim!r}")


class TestUnifiedCoverageIsImportedNotCopied(unittest.TestCase):
    def setUp(self):
        self.candidate = ATTENTION_REGISTRY.get("attention_unified_2d")

    def test_head_sizes_are_the_backends_own_object(self):
        # `assertIs`, not `assertEqual`: a copy compares equal today and drifts
        # tomorrow, which is precisely the failure being excluded.
        self.assertIs(
            _shape_range(self.candidate, "hdim_q").allowed, au.UNIFIED_HEAD_SIZES
        )

    def test_block_sizes_are_the_backends_own_object(self):
        self.assertIs(
            _shape_range(self.candidate, "kv_block_size").allowed,
            au.UNIFIED_BLOCK_SIZES,
        )

    def test_dtypes_are_the_backends_own_object(self):
        self.assertIs(self.candidate.capability.dtypes, au.UNIFIED_DTYPES)

    def test_the_prefilter_never_rejects_what_the_backend_accepts(self):
        """The direction that fails silently, checked behaviourally.

        Swept well past the declared sets so the test still means something if
        the backend's coverage grows: every combination the backend says yes to
        must survive the capability.
        """
        for head_size in (32, 48, 64, 96, 128, 192, 256, 512):
            for block_size in (8, 16, 32, 64, 128):
                for dtype in ("fp16", "bf16", "fp8"):
                    req = _req(
                        hdim_q=head_size,
                        hdim_v=head_size,
                        kv_block_size=block_size,
                        dtype=dtype,
                    )
                    backend_ok, _ = au.supports_native_unified_attention(_problem(req))
                    if not backend_ok:
                        continue
                    with self.subTest(hd=head_size, bs=block_size, dtype=dtype):
                        ok, why = self.candidate.capability.check(req)
                        self.assertTrue(
                            ok, f"capability rejected a supported shape: {why}"
                        )

    def test_the_two_agree_exactly_on_the_declared_sets(self):
        # The other direction is merely redundant rather than dangerous, but
        # pinning it keeps the capability from over-claiming, which would push
        # a rejection later than it needs to be.
        for head_size in (32, 48, 64, 96, 128, 192, 256, 512):
            req = _req(hdim_q=head_size, hdim_v=head_size)
            backend_ok, _ = au.supports_native_unified_attention(_problem(req))
            with self.subTest(hdim_q=head_size):
                self.assertEqual(self.candidate.capability.check(req)[0], backend_ok)


class TestGfx1250CoverageIsImportedNotCopied(unittest.TestCase):
    """Same rule for the standalone kernel, whose gates raise rather than return."""

    def setUp(self):
        self.candidate = ATTENTION_REGISTRY.get("attention_gfx1250_wmma")

    def test_head_size_bound_tracks_the_wmma_contraction(self):
        rng = _shape_range(self.candidate, "hdim_q")
        self.assertEqual(rng.multiple_of, wmma.WMMA_K)
        self.assertEqual(rng.min, wmma.WMMA_K)

    def test_seqlen_bound_tracks_the_m_tile(self):
        rng = _shape_range(self.candidate, "seqlen_q")
        self.assertEqual(rng.multiple_of, wmma.BLOCK_M)

    def test_dtypes_are_the_kernels_own_object(self):
        self.assertIs(self.candidate.capability.dtypes, wmma.DTYPES)

    def test_the_prefilter_catches_everything_the_spec_would_raise_on(self):
        # WmmaAttentionFwdSpec raises from __post_init__, so a shape the
        # capability lets through would surface as a traceback out of
        # select_spec rather than a routing decision.
        for head_size in (32, 48, 64, 96, 128, 160, 256):
            for dtype in ("fp16", "bf16"):
                req = _req(
                    arch="gfx1250",
                    algorithm="wmma_attention_fwd",
                    hdim_q=head_size,
                    hdim_v=head_size,
                    dtype=dtype,
                )
                with self.subTest(hdim_q=head_size, dtype=dtype):
                    if not self.candidate.admits(req)[0]:
                        continue
                    self.candidate.select_spec(req)  # must not raise


if __name__ == "__main__":
    unittest.main()
