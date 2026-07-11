# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Registry behavior tests for GEMM dispatcher candidates."""

from __future__ import annotations

import unittest

from rocke.dispatch.core import CandidateRegistry, KernelCandidate, OperatorRequest
from rocke.dispatch.gemm import GemmRequest, gemm_fp16_candidates


def _dummy_candidate(name: str, family: str = "dummy") -> KernelCandidate:
    return KernelCandidate(
        name=name,
        family=family,
        algorithm="dummy_algorithm",
        spec_id="dummy_spec",
        abi_version="dummy/v1",
        priority=0,
        supports=lambda _req: (True, "ok"),
        select_spec=lambda _req: object(),
        signature=lambda _spec: (),
        grid=lambda _spec, _req: (1, 1, 1),
        block=lambda _spec: (1, 1, 1),
        sweep_space=lambda _req: (),
    )


class TestCandidateRegistry(unittest.TestCase):
    def test_duplicate_registration_is_rejected(self):
        registry = CandidateRegistry("dummy")
        registry.register(_dummy_candidate("same"))
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            registry.register(_dummy_candidate("same"))

    def test_wrong_family_registration_is_rejected(self):
        registry = CandidateRegistry("dummy")
        with self.assertRaisesRegex(ValueError, "candidate family"):
            registry.register(_dummy_candidate("wrong", family="other"))

    def test_select_rejects_ranker_returning_unsupported_candidate(self):
        registry = CandidateRegistry("dummy")
        registry.register(_dummy_candidate("supported"))
        req = OperatorRequest()
        rogue = _dummy_candidate("rogue")

        with self.assertRaisesRegex(ValueError, "unsupported candidate"):
            registry.select(req, ranker=lambda _req, _cands: (rogue,))

    def test_gemm_registry_has_unique_names(self):
        candidates = gemm_fp16_candidates()
        names = [c.name for c in candidates]
        self.assertEqual(len(names), len(set(names)))

    def test_gemm_registered_candidates_support_expected_arch_classes(self):
        cdna_req = GemmRequest(M=128, N=128, K=32, arch="gfx950")
        rdna_req = GemmRequest(M=64, N=32, K=16, arch="gfx1151")
        supported_cdna = {
            c.spec_id for c in gemm_fp16_candidates() if c.supports(cdna_req)[0]
        }
        supported_rdna = {
            c.spec_id for c in gemm_fp16_candidates() if c.supports(rdna_req)[0]
        }
        self.assertIn("cdna_cshuffle_default", supported_cdna)
        self.assertIn("cdna_mem_64x128", supported_cdna)
        self.assertIn("rdna_wmma_default", supported_rdna)
        self.assertIn("rdna_wmma_32x32", supported_rdna)


if __name__ == "__main__":
    unittest.main()
