# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declared-coverage invariants applied to every platform registry.

The same checks the GEMM family gets in ``dispatch_tests/gemm/test_capability.py``,
run here across conv, moe, norm and the contraction families so a new family inherits them by being
listed once. Each family supplies a request generator wide enough that every
registered candidate is reachable somewhere in it; without that, the
reachability check below would pass vacuously.
"""

from __future__ import annotations

import unittest

from rocke.core.arch import ArchTarget, known_arches
from rocke.dispatch.families.gett_f64 import GETT_F64_REGISTRY, GettF64Request
from rocke.dispatch.families.moe import MOE_REGISTRY, MoeRequest
from rocke.dispatch.families.norm import NORM_REGISTRY, NormRequest
from rocke.dispatch.families.tensor_gett import (
    TENSOR_GETT_REGISTRY,
    TensorGettRequest,
)
from rocke.instances.common.gett_f64 import gemm_problem
from rocke.instances.common.tensor import GettKnobs, TensorGettSpec


def _moe_requests(arch: str):
    for dtype in ("fp16", "bf16", "fp8"):
        for tokens in (1, 128, 4096):
            for hidden in (512, 4096):
                for inter in (1408, 14336):
                    yield MoeRequest(
                        num_tokens=tokens,
                        hidden=hidden,
                        intermediate=inter,
                        num_experts=32,
                        top_k=4,
                        arch=arch,
                        dtype=dtype,
                    )


def _norm_requests(arch: str):
    for kind in ("rmsnorm", "layernorm"):
        for dtype in ("fp16", "bf16"):
            for cols in (64, 256, 512, 768, 1024, 2048, 4096, 8192):
                yield NormRequest(
                    rows=128, cols=cols, arch=arch, kind=kind, dtype=dtype
                )


def _gett_f64_requests(arch: str):
    # The shape classes need the part's CU count (no per-arch default).
    num_cus = {"gfx90a": 104, "gfx942": 304, "gfx950": 256}.get(arch, 256)
    for cx in (False, True):
        for n in (256, 2048, 8192):
            for runtime_sizes in (False, True):
                yield GettF64Request(
                    problem=gemm_problem(
                        n,
                        n,
                        512,
                        a_strides=(512, 1),
                        b_strides=(512, 1),
                        d_strides=(n, 1),
                        complex_=cx,
                    ),
                    arch=arch,
                    num_cus=num_cus,
                    runtime_sizes=runtime_sizes,
                )


def _tensor_gett_requests(arch: str):
    # a permuted contraction (D modes interleaved from both operands)
    ext = {"K0": 8, "K1": 8, "M0": 16, "M1": 32, "N0": 16, "N1": 24}
    for dtype in ("fp16", "bf16"):
        for bilinear in (False, True):
            yield TensorGettRequest(
                problem=TensorGettSpec.make(
                    ["K1", "M1", "K0", "M0"],
                    ["K1", "N1", "K0", "N0"],
                    ["N1", "M1", "N0", "M0"],
                    ext,
                    dtype=dtype,
                    knobs=GettKnobs(bilinear=bilinear),
                ),
                arch=arch,
            )


_FAMILIES = (
    (MOE_REGISTRY, _moe_requests),
    (NORM_REGISTRY, _norm_requests),
    (GETT_F64_REGISTRY, _gett_f64_requests),
    (TENSOR_GETT_REGISTRY, _tensor_gett_requests),
)

# Candidates that legitimately span both wave sizes. Norm kernels derive
# ``wave_size`` from the target and reduce through LDS rather than an MMA atom,
# so one candidate genuinely serves wave32 and wave64. The invariant exists to
# catch MMA kernels that bake a wave size into their geometry and would emit
# wrong ISA on the other one, which does not apply here.
_WAVE_AGNOSTIC_FAMILIES = frozenset({NORM_REGISTRY.family})


class TestDeclaredCoverage(unittest.TestCase):
    def test_every_candidate_declares_a_capability(self):
        for registry, _ in _FAMILIES:
            for candidate in registry.candidates():
                with self.subTest(family=registry.family, candidate=candidate.name):
                    self.assertIsNotNone(candidate.capability)

    def test_registration_rejects_a_candidate_without_one(self):
        """The mandate is enforced at import time, not merely observed above."""
        from dataclasses import replace

        from rocke.dispatch.core import CandidateRegistry

        registry, _ = _FAMILIES[0]
        naked = replace(registry.candidates()[0], capability=None)
        test_registry = CandidateRegistry(registry.family)
        with self.assertRaisesRegex(ValueError, "declares no capability"):
            test_registry.register(naked)

    def test_no_candidate_admits_an_architecture_it_did_not_declare(self):
        for registry, requests in _FAMILIES:
            for candidate in registry.candidates():
                undeclared = set(known_arches()) - set(candidate.capability.arches)
                for arch in sorted(undeclared):
                    with self.subTest(candidate=candidate.name, arch=arch):
                        for req in requests(arch):
                            self.assertFalse(
                                candidate.admits(req)[0],
                                f"{candidate.name} wrongly admits {arch}",
                            )

    def test_every_declared_architecture_is_reachable(self):
        """Coverage claimed but never served makes the manifest overstate what
        is dispatchable, which is the failure a manifest exists to prevent."""
        for registry, requests in _FAMILIES:
            for candidate in registry.candidates():
                for arch in candidate.capability.arches:
                    with self.subTest(candidate=candidate.name, arch=arch):
                        self.assertTrue(
                            any(candidate.admits(r)[0] for r in requests(arch)),
                            f"{candidate.name} declares {arch} but admits no "
                            "request there",
                        )

    def test_every_mma_candidate_spans_a_single_wave_size(self):
        for registry, _ in _FAMILIES:
            if registry.family in _WAVE_AGNOSTIC_FAMILIES:
                continue
            for candidate in registry.candidates():
                with self.subTest(family=registry.family, candidate=candidate.name):
                    waves = {
                        ArchTarget.from_gfx(a).wave_size
                        for a in candidate.capability.arches
                    }
                    self.assertEqual(
                        len(waves),
                        1,
                        f"{candidate.name} spans wave sizes {sorted(waves)}; "
                        "split it into one candidate per wave size",
                    )

    def test_for_arch_agrees_with_what_each_candidate_declares(self):
        for registry, _ in _FAMILIES:
            for arch in known_arches():
                with self.subTest(family=registry.family, arch=arch):
                    self.assertEqual(
                        {c.name for c in registry.for_arch(arch)},
                        {
                            c.name
                            for c in registry.candidates()
                            if arch in c.capability.arches
                        },
                    )

    def test_coverage_manifest_is_complete(self):
        for registry, _ in _FAMILIES:
            manifest = registry.coverage()
            with self.subTest(family=registry.family):
                self.assertEqual(
                    len(manifest["candidates"]), len(registry.candidates())
                )
                for entry in manifest["candidates"]:
                    self.assertTrue(entry["capability"]["arches"])


if __name__ == "__main__":
    unittest.main()
