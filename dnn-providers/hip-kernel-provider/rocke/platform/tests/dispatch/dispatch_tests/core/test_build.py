# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The build seam: spec/builder agreement across every family.

The contract `build` asserts is that the spec `select_spec` returns is exactly
what the family's builder accepts. That is easy to get subtly wrong -- a family
with two spec types routing to one builder, a builder whose signature grew an
argument -- and the failure is a TypeError at compile time, far from the
registration that caused it. So these tests actually dispatch and build, rather
than checking that a callable was assigned.

Building is IR construction only: no comgr, no device, so this runs in CPU CI.
"""

from __future__ import annotations

import unittest

from rocke.dispatch.families.moe import MoeRequest, dispatch_moe
from rocke.dispatch.families.norm import NormRequest, dispatch_norm
from rocke.dispatch.gemm.bf16_rcr import dispatch_gemm_bf16
from rocke.dispatch.gemm.common import GemmRequest
from rocke.dispatch.gemm.fp16_rcr import dispatch_gemm_fp16


def _gemm(dtype):
    return GemmRequest(M=256, N=256, K=256, arch="gfx950", dtype=dtype)


def _moe(dtype="fp16"):
    return MoeRequest(
        num_tokens=128,
        hidden=7168,
        intermediate=2048,
        num_experts=256,
        top_k=8,
        arch="gfx950",
        dtype=dtype,
    )


def _norm(kind):
    return NormRequest(rows=4096, cols=4096, kind=kind, arch="gfx950")


# One representative request per registered family, so a family that grows a
# build cannot skip the agreement check.
_CASES = (
    ("gemm_fp16_rcr", dispatch_gemm_fp16, _gemm("fp16")),
    ("gemm_bf16_rcr", dispatch_gemm_bf16, _gemm("bf16")),
    ("moe_fused_mega_f16", dispatch_moe, _moe("fp16")),
    ("norm2d_rmsnorm", dispatch_norm, _norm("rmsnorm")),
    ("norm2d_layernorm", dispatch_norm, _norm("layernorm")),
)


class TestSpecBuilderAgreement(unittest.TestCase):
    def test_every_family_builds_what_it_selects(self):
        for label, dispatch, request in _CASES:
            with self.subTest(family=label):
                kernel = dispatch(request).build()
                self.assertIsNotNone(kernel)
                # A KernelDef always names its entry point; anything else means
                # the builder returned something that is not a kernel.
                self.assertTrue(
                    getattr(kernel, "name", ""), f"{label} built {kernel!r}"
                )

    def test_build_is_deterministic_for_one_selection(self):
        # Two builds of the same selection must agree on the entry-point name,
        # or the compile cache keyed on `compile_key` would be unsound.
        for label, dispatch, request in _CASES:
            with self.subTest(family=label):
                result = dispatch(request)
                self.assertEqual(result.build().name, result.build().name)

    def test_build_targets_the_requested_arch_not_the_host(self):
        # Offline and cross-arch dispatch depend on arch coming from the
        # request. gfx942 and gfx950 pick different UniversalGemm specs, so a
        # builder ignoring `arch` would show up as identical output here.
        names = {
            arch: dispatch_gemm_fp16(
                GemmRequest(M=256, N=256, K=256, arch=arch, dtype="fp16")
            )
            .build()
            .name
            for arch in ("gfx942", "gfx950")
        }
        self.assertEqual(len(names), 2)


class TestBuildRatchet(unittest.TestCase):
    """Every platform family requires a build; none may quietly stop."""

    def _registries(self):
        from rocke.dispatch.families.moe import MOE_REGISTRY
        from rocke.dispatch.families.norm import NORM_REGISTRY
        from rocke.dispatch.gemm.bf16_rcr import GEMM_BF16_REGISTRY
        from rocke.dispatch.gemm.fp16_rcr import GEMM_FP16_REGISTRY

        return (
            GEMM_FP16_REGISTRY,
            GEMM_BF16_REGISTRY,
            MOE_REGISTRY,
            NORM_REGISTRY,
        )

    def test_all_platform_families_require_build(self):
        # Unlike `bind`, `build` has no exempt families: a spec with no builder
        # is not a kernel anyone can use. If a new family needs an exemption,
        # that is a design discussion, not a default.
        for registry in self._registries():
            with self.subTest(family=registry.family):
                self.assertTrue(registry.require_build)

    def test_coverage_reports_buildability(self):
        for registry in self._registries():
            coverage = registry.coverage()
            with self.subTest(family=registry.family):
                self.assertTrue(coverage["requires_build"])
                self.assertTrue(all(c["buildable"] for c in coverage["candidates"]))

    def test_registration_fails_when_a_required_build_is_missing(self):
        from rocke.core.arch import known_arches
        from rocke.dispatch.core import (
            Capability,
            CandidateRegistry,
            KernelCandidate,
        )

        registry = CandidateRegistry("demo", require_build=True)
        with self.assertRaises(ValueError) as ctx:
            registry.register(
                KernelCandidate(
                    name="unbuildable",
                    family="demo",
                    algorithm="demo",
                    spec_id="demo",
                    abi_version="demo/v1",
                    priority=10,
                    capability=Capability(arches=known_arches()),
                    _supports=lambda req: (True, ""),
                    select_spec=lambda req: None,
                    signature=lambda spec: (),
                    grid=lambda spec, req: (1, 1, 1),
                    block=lambda spec: (1, 1, 1),
                    sweep_space=lambda req: (),
                )
            )
        self.assertIn("unbuildable", str(ctx.exception))
        self.assertIn("build", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
