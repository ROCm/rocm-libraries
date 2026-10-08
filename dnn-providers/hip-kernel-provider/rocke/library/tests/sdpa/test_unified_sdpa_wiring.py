# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wiring tests for the U-03 SDPA unified binding.

Pins: fp16 d=256 is declined on all arches (ATT-34); bf16 d=256 and all fp16
d=64/128 shapes are admitted *by the candidate that actually serves them*; an
invalid request reports its own cause rather than a policy reason; admits() and
candidates() reject exactly the same requests; the "f16" dtype alias normalizes
correctly and does not block non-d256 fp16.

These are CPU-only dispatch tests — no GPU, no torch required.
"""
from __future__ import annotations

import unittest

from rocke.core.arch import known_arches

from dispatch.sdpa.common import SdpaRequest
from dispatch.sdpa.bind_unified import sdpa_unified_admits, sdpa_unified_candidates


def _req(arch: str, dtype: str, head_dim: int) -> SdpaRequest:
    return SdpaRequest(arch=arch, dtype=dtype, head_dim=head_dim)


class TestFp16D256Declined(unittest.TestCase):
    """fp16 d=256 must be declined on every arch (ATT-34)."""

    def test_fp16_d256_gfx942_declined(self):
        ok, why = sdpa_unified_admits(_req("gfx942", "fp16", 256))
        self.assertFalse(ok)
        self.assertIn("ATT-34", why)

    def test_fp16_d256_gfx950_declined(self):
        ok, why = sdpa_unified_admits(_req("gfx950", "fp16", 256))
        self.assertFalse(ok)
        self.assertIn("ATT-34", why)

    def test_fp16_d256_f16_alias_declined(self):
        # "f16" normalizes to "fp16" in __post_init__; the decline must still fire.
        ok, why = sdpa_unified_admits(_req("gfx942", "f16", 256))
        self.assertFalse(ok)
        self.assertIn("ATT-34", why)

    def test_declined_request_offers_no_candidates(self):
        self.assertEqual(sdpa_unified_candidates(_req("gfx942", "fp16", 256)), ())


class TestScalarPathArchesAreDtypeSymmetric(unittest.TestCase):
    """An arch with no dedicated candidate serves fp16 and bf16 alike.

    gfx1151 has no candidate of its own, so it rides the two generic path
    candidates, which route it to the arch-neutral scalar kernel. That kernel
    admits ``dtype in {fp16, bf16}`` with no arch axis, so there is no fp16-only
    gap to express here. (The WMMA FMHA kernels under ``builders/gfx1151/`` are
    a standalone case study, not registered as dispatch candidates — if RD-01
    wires one in, the gap becomes real and gets a rule then.)

    This is a regression guard against re-adding a per-arch dtype decline that
    the dispatch path does not back.
    """

    def test_gfx1151_fp16_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx1151", "fp16", 128))
        self.assertTrue(ok)

    def test_gfx1151_bf16_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx1151", "bf16", 128))
        self.assertTrue(ok)

    def test_gfx1151_serves_both_dtypes_from_the_generic_path(self):
        # Identical verdicts are the point: no specialized candidate claims
        # either dtype, so declining one and admitting the other has no basis.
        fp16 = sdpa_unified_candidates(_req("gfx1151", "fp16", 128))
        bf16 = sdpa_unified_candidates(_req("gfx1151", "bf16", 128))
        self.assertEqual(fp16, bf16)
        self.assertIn("attention_unified_2d", bf16)


class TestBf16D256Admitted(unittest.TestCase):
    """bf16 d=256 is admitted, and by the candidate that really serves it."""

    def test_bf16_d256_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "bf16", 256))
        self.assertTrue(ok)

    def test_bf16_d256_gfx950_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx950", "bf16", 256))
        self.assertTrue(ok)

    def test_bf16_d256_gfx950_uses_dedicated_prefill_candidate(self):
        # Regression guard. A degenerate probe shape (seqlen 1x1, or nhead_q=1)
        # routes to the generic fallback and silently drops this candidate, so
        # the boolean alone would stay green while the fast path went unseen.
        names = sdpa_unified_candidates(_req("gfx950", "bf16", 256))
        self.assertIn("attention_gfx950_d256", names)

    def test_bf16_d256_reaches_the_decode_candidate(self):
        # The decode probe is what surfaces this one; pins that probe's presence.
        names = sdpa_unified_candidates(_req("gfx950", "bf16", 256))
        self.assertIn("attention_d256_decode", names)

    def test_bf16_d256_gfx942_has_no_dedicated_prefill_candidate(self):
        # gfx942's d256 fast path is a cohort *inside* the unified kernel, not a
        # separate candidate -- so a generic verdict is correct here, not a gap.
        names = sdpa_unified_candidates(_req("gfx942", "bf16", 256))
        self.assertIn("attention_unified_2d", names)
        self.assertNotIn("attention_gfx950_d256", names)


class TestSmallerHeadDimsAdmitted(unittest.TestCase):
    """fp16 and bf16 at d=64 and d=128 must be admitted on gfx942 and gfx950."""

    def test_fp16_d128_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "fp16", 128))
        self.assertTrue(ok)

    def test_fp16_d64_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "fp16", 64))
        self.assertTrue(ok)

    def test_fp16_d128_gfx950_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx950", "fp16", 128))
        self.assertTrue(ok)

    def test_fp16_d64_gfx950_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx950", "fp16", 64))
        self.assertTrue(ok)

    def test_bf16_d128_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "bf16", 128))
        self.assertTrue(ok)


class TestUnsupportedInputsDeclined(unittest.TestCase):
    """Head dims off the unified vocabulary, and unknown arches, are declined."""

    def test_non_vocabulary_head_dim_declined(self):
        ok, why = sdpa_unified_admits(_req("gfx942", "bf16", 48))
        self.assertFalse(ok)
        self.assertIn("not a unified head size", why)

    def test_zero_head_dim_declined(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "bf16", 0))
        self.assertFalse(ok)

    def test_unknown_arch_declined(self):
        ok, why = sdpa_unified_admits(_req("not-a-gpu", "bf16", 128))
        self.assertFalse(ok)
        self.assertIn("unknown arch", why)

    def test_untargeted_rocm_arch_declined(self):
        # gfx906 is a real ROCm arch, but not one this build targets — a
        # realistic caller mistake that must not reach the registry.
        ok, why = sdpa_unified_admits(_req("gfx906", "bf16", 128))
        self.assertFalse(ok)
        self.assertIn("unknown arch", why)

    def test_every_known_arch_is_admitted_at_bf16_d128(self):
        # The converse guard: the reject gate must not over-reject. The two
        # generic path candidates declare known_arches(), so every target this
        # build knows about is served at a vanilla shape.
        for arch in sorted(known_arches()):
            with self.subTest(arch=arch):
                ok, why = sdpa_unified_admits(_req(arch, "bf16", 128))
                self.assertTrue(ok, f"{arch} unexpectedly declined: {why}")


class TestInvalidRequestsReportTheirOwnCause(unittest.TestCase):
    """Validation runs before the SDPA policy declines.

    A request that is both invalid *and* matches a policy rule must report the
    invalidity. Otherwise the ATT-34 reason masks the real cause and sends a
    caller off to change dtype when the actual problem is the head_dim or arch.
    """

    def test_unknown_arch_beats_the_fp16_d256_rule(self):
        # The case that actually pins the order: fp16 d=256 matches the ATT-34
        # rule exactly, but the arch is the real problem and must be reported.
        ok, why = sdpa_unified_admits(_req("not-a-gpu", "fp16", 256))
        self.assertFalse(ok)
        self.assertIn("unknown arch", why)
        self.assertNotIn("ATT-34", why)

    def test_bad_head_dim_reports_the_head_dim(self):
        # ATT-34 needs d == 256 exactly, so no policy rule competes here; this
        # pins that the head_dim check still owns the reason for d=255.
        ok, why = sdpa_unified_admits(_req("gfx942", "fp16", 255))
        self.assertFalse(ok)
        self.assertIn("not a unified head size", why)
        self.assertNotIn("ATT-34", why)


class TestCandidatesAgreesWithAdmits(unittest.TestCase):
    """``candidates()`` returns names only for requests ``admits()`` accepts.

    Both route through the same reject gate, so an invalid request is never
    handed to the registry as a probe shape.
    """

    def _assert_declined_everywhere(self, req):
        ok, _ = sdpa_unified_admits(req)
        self.assertFalse(ok)
        self.assertEqual(sdpa_unified_candidates(req), ())

    def test_bad_head_dim_yields_no_candidates(self):
        self._assert_declined_everywhere(_req("gfx942", "bf16", 48))

    def test_zero_head_dim_yields_no_candidates(self):
        self._assert_declined_everywhere(_req("gfx942", "bf16", 0))

    def test_unknown_arch_yields_no_candidates(self):
        self._assert_declined_everywhere(_req("not-a-gpu", "bf16", 128))

    def test_untargeted_rocm_arch_yields_no_candidates(self):
        self._assert_declined_everywhere(_req("gfx906", "bf16", 128))

    def test_policy_declined_request_yields_no_candidates(self):
        # The ATT-34 rule is a policy decline, not a validation failure; both
        # kinds must suppress candidates identically.
        self._assert_declined_everywhere(_req("gfx950", "fp16", 256))

    def test_admitted_request_yields_candidates(self):
        # The converse, so the agreement is not trivially satisfied by
        # candidates() always returning empty.
        req = _req("gfx950", "bf16", 256)
        ok, _ = sdpa_unified_admits(req)
        self.assertTrue(ok)
        self.assertNotEqual(sdpa_unified_candidates(req), ())


class TestDtypeAliasNormalization(unittest.TestCase):
    """The "f16" alias must not block non-d256 fp16 requests."""

    def test_f16_alias_d128_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "f16", 128))
        self.assertTrue(ok)

    def test_bad_dtype_raises(self):
        with self.assertRaises(ValueError):
            SdpaRequest(arch="gfx942", dtype="badtype", head_dim=128)


if __name__ == "__main__":
    unittest.main()
