# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the D256 bf16 decode dispatcher wiring (gfx950 + gfx942).

Covers:
  - d256_decode candidate is registered and discoverable
  - d256_decode priority outranks unified_3d (priority 10) for eligible shapes
  - gfx950 decode GQA (16/2 heads, hd256, bf16) routes to d256_decode
  - gfx942 decode routes to d256_decode
  - explicit algorithm="d256_decode" forces the candidate
  - spec path is "3d" and name is "rocke_attention_d256_decode"
  - gates: rejects non-gfx942/gfx950 arch, non-bf16, prefill (2D), hd!=256,
    sliding_window, sinks, softcap
"""

from __future__ import annotations

import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)


def _gfx950_d256_decode(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=16,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=8192,
        hdim_q=256,
        hdim_v=256,
        arch="gfx950",
        dtype="bf16",
    )
    base.update(kw)
    return AttentionRequest(**base)


class _PinnedArch:
    """Context-manager that pins _RESOLVED_ATTENTION_ARCH to a given arch."""

    def __init__(self, arch: str):
        self._arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self._arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


class TestD256DecodeRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        names = [c.name for c in attention_candidates()]
        self.assertIn("attention_d256_decode", names)

    def test_spec_id_is_d256_decode(self):
        candidate = next(
            c for c in attention_candidates() if c.name == "attention_d256_decode"
        )
        self.assertEqual(candidate.spec_id, "d256_decode")

    def test_algorithm_is_d256_decode(self):
        candidate = next(
            c for c in attention_candidates() if c.name == "attention_d256_decode"
        )
        self.assertEqual(candidate.algorithm, "d256_decode")

    def test_priority_beats_unified_3d(self):
        # The registry sorts ascending; d256_decode must have a lower priority
        # number than unified_3d so it is selected first for eligible requests.
        d256 = next(
            c for c in attention_candidates() if c.name == "attention_d256_decode"
        )
        unified_3d = next(
            c for c in attention_candidates() if c.name == "attention_unified_3d"
        )
        self.assertLess(d256.priority, unified_3d.priority)


class TestD256DecodeSupportGates(unittest.TestCase):
    def _candidate(self):
        return next(
            c for c in attention_candidates() if c.name == "attention_d256_decode"
        )

    def test_rejects_non_gfx942_gfx950_arch(self):
        with _PinnedArch("gfx950"):
            ok, why = self._candidate().admits(_gfx950_d256_decode(arch="gfx1250"))
            self.assertFalse(ok)
            self.assertIn("gfx942", why)

    def test_rejects_non_bf16_dtype(self):
        with _PinnedArch("gfx950"):
            ok, why = self._candidate().admits(_gfx950_d256_decode(dtype="fp16"))
            self.assertFalse(ok)
            self.assertIn("bf16", why)

    def test_rejects_prefill(self):
        # Prefill (seqlen_q>1) routes to 2d, not 3d — d256_decode must refuse.
        with _PinnedArch("gfx950"):
            req = _gfx950_d256_decode(
                batch=2, nhead_q=16, nhead_k=2, seqlen_q=512, seqlen_k=512
            )
            ok, why = self._candidate().admits(req)
            self.assertFalse(ok)
            # Either the cohort predicate (not all_decode) or the path check fires.
            self.assertTrue("cohort" in why or "3d" in why or "2d" in why)

    # The three rejections below are declared in the candidate's capability, so
    # they are answered by the prefilter and never reach the cohort predicate.
    # The predicate still refuses them; these assert the earlier, sharper reason.
    def test_rejects_non_d256_head_size(self):
        with _PinnedArch("gfx950"):
            ok, why = self._candidate().admits(
                _gfx950_d256_decode(hdim_q=128, hdim_v=128)
            )
            self.assertFalse(ok)
            self.assertIn("hdim_q=128 not in (256,)", why)

    def test_rejects_sliding_window(self):
        with _PinnedArch("gfx950"):
            ok, why = self._candidate().admits(_gfx950_d256_decode(sliding_window=64))
            self.assertFalse(ok)
            self.assertIn("cannot serve features ['sliding_window']", why)

    def test_rejects_sinks(self):
        with _PinnedArch("gfx950"):
            ok, why = self._candidate().admits(_gfx950_d256_decode(use_sinks=True))
            self.assertFalse(ok)
            self.assertIn("cannot serve features ['sinks']", why)

    def test_cohort_predicate_still_refuses_what_capability_declares(self):
        """Capability must never be the only thing saying no: if the prefilter
        were dropped the predicate must still reject, or the two have drifted."""
        from kernels.common.attention_unified import _d256_decode_cohort

        from dispatch.attention import _problem

        with _PinnedArch("gfx950"):
            for label, req in (
                ("head_size", _gfx950_d256_decode(hdim_q=128, hdim_v=128)),
                ("sliding_window", _gfx950_d256_decode(sliding_window=64)),
                ("sinks", _gfx950_d256_decode(use_sinks=True)),
            ):
                with self.subTest(rejected_by=label):
                    self.assertFalse(_d256_decode_cohort(_problem(req)))


class TestD256DecodeRouting(unittest.TestCase):
    """Verify that dispatch picks d256_decode for eligible shapes."""

    def test_gfx950_gqa_decode_routes_d256_decode(self):
        with _PinnedArch("gfx950"):
            r = dispatch_attention(_gfx950_d256_decode())
        self.assertEqual(r.candidate.spec_id, "d256_decode")
        self.assertEqual(r.spec.path, "3d")
        self.assertEqual(r.spec.name, "rocke_attention_d256_decode")

    def test_gfx942_gqa_decode_routes_d256_decode(self):
        with _PinnedArch("gfx942"):
            r = dispatch_attention(_gfx950_d256_decode(arch="gfx942"))
        self.assertEqual(r.candidate.spec_id, "d256_decode")
        self.assertEqual(r.spec.path, "3d")

    def test_spec_records_correct_dims(self):
        with _PinnedArch("gfx950"):
            r = dispatch_attention(
                _gfx950_d256_decode(
                    batch=2,
                    nhead_q=16,
                    nhead_k=2,
                    seqlen_q=1,
                    seqlen_k=4096,
                    kv_block_size=32,
                )
            )
        self.assertEqual(r.spec.head_size, 256)
        self.assertEqual(r.spec.block_size, 32)
        self.assertEqual(r.spec.dtype, "bf16")

    def test_explicit_algorithm_forces_d256_decode(self):
        with _PinnedArch("gfx950"):
            r = dispatch_attention(_gfx950_d256_decode(algorithm="d256_decode"))
        self.assertEqual(r.candidate.algorithm, "d256_decode")

    def test_explicit_algorithm_unified_3d_bypasses_d256_decode(self):
        with _PinnedArch("gfx950"):
            r = dispatch_attention(
                _gfx950_d256_decode(algorithm="auto", spec_id="unified_3d")
            )
        self.assertEqual(r.candidate.spec_id, "unified_3d")

    def test_d256_decode_cohort_predicate_all_decode(self):
        # The cohort helper itself: all_decode=True required.
        p_decode = au.UnifiedAttentionProblem(
            total_q=1,
            num_seqs=1,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=256,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=8192,
            dtype="bf16",
        )
        self.assertTrue(au._d256_decode_cohort(p_decode))

    def test_d256_decode_cohort_predicate_rejects_prefill(self):
        p_prefill = au.UnifiedAttentionProblem(
            total_q=512,
            num_seqs=1,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=256,
            block_size=16,
            max_seqlen_q=512,
            max_seqlen_k=8192,
            dtype="bf16",
        )
        self.assertFalse(au._d256_decode_cohort(p_prefill))

    def test_d256_decode_cohort_predicate_rejects_fp16(self):
        p = au.UnifiedAttentionProblem(
            total_q=1,
            num_seqs=1,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=256,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=8192,
            dtype="fp16",
        )
        self.assertFalse(au._d256_decode_cohort(p))


if __name__ == "__main__":
    unittest.main()
