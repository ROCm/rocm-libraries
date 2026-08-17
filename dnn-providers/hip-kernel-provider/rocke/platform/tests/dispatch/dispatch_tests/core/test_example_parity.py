# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build every registered kernel, and check it against the examples' own IR.

Two questions, which are easy to conflate:

1. *Does every registered candidate actually build?* `test_build.py` answers
   that for one request per family. Here it is asked exhaustively -- every
   candidate, on every arch it declares -- because a candidate is only as
   registered as its least-exercised arch, and a spec that no request reaches
   is a spec nobody has compiled.

2. *Does going through the dispatcher produce the same kernel as building it by
   hand?* The examples are the hand-built path: they construct a spec directly
   and call the builder. If dispatch is to replace those call sites, the IR has
   to be identical, not merely similar -- so the comparison is
   `canonical_equal` on the serialized IR, which is exact modulo SSA-id gaps.

Both are CPU-only. Building is IR construction; no comgr, no device.
"""

from __future__ import annotations

import unittest

from rocke.core.arch import ArchTarget
from rocke.core.ir_serialize import canonical_equal
from rocke.dispatch.families.moe import MOE_REGISTRY, MoeRequest, dispatch_moe
from rocke.dispatch.families.norm import NORM_REGISTRY, NormRequest, dispatch_norm
from rocke.dispatch.gemm.bf16_rcr import GEMM_BF16_REGISTRY
from rocke.dispatch.gemm.common import GemmRequest
from rocke.dispatch.gemm.fp16_rcr import GEMM_FP16_REGISTRY, dispatch_gemm_fp16

# ---------------------------------------------------------------------------
# Problem shapes lifted from the examples, so "does it build?" is asked about
# sizes someone actually runs rather than sizes chosen to make the test pass.
# ---------------------------------------------------------------------------

# examples/common/universal_gemm_verify.py
GEMM_MNK = (512, 512, 512)
# examples/gfx1250/fused_mega_moe/fused_mega_moe_bench.py
MOE_SHAPE = dict(
    num_tokens=128, hidden=2048, intermediate=768, num_experts=256, top_k=8
)
# examples/common/ck_tile_parity.py
NORM_SHAPE = dict(rows=4096, cols=4096)

# A second shape where one does not reach every candidate. Both are shapes
# those candidates exist *for*, not padding to satisfy a reachability check:
# the bf16 decode candidate caps M at 32, and b1024_v8 spans 8192 columns.
GEMM_DECODE_MNK = (16, 4096, 4096)
NORM_WIDE_SHAPE = dict(rows=4096, cols=8192)


def _gemm_reqs(arch, spec_id, dtype):
    for m, n, k in (GEMM_MNK, GEMM_DECODE_MNK):
        yield GemmRequest(M=m, N=n, K=k, arch=arch, dtype=dtype, spec_id=spec_id)


def _moe_reqs(arch, spec_id, dtype):
    yield MoeRequest(arch=arch, dtype=dtype, spec_id=spec_id, **MOE_SHAPE)


def _norm_reqs(arch, spec_id, dtype):
    kind = "rmsnorm" if "rmsnorm" in spec_id else "layernorm"
    for shape in (NORM_SHAPE, NORM_WIDE_SHAPE):
        yield NormRequest(kind=kind, arch=arch, dtype=dtype, spec_id=spec_id, **shape)


_FAMILIES = (
    (GEMM_FP16_REGISTRY, _gemm_reqs, ("fp16",)),
    (GEMM_BF16_REGISTRY, _gemm_reqs, ("bf16",)),
    (MOE_REGISTRY, _moe_reqs, ("fp16", "fp8")),
    (NORM_REGISTRY, _norm_reqs, ("fp16",)),
)

# A floor, not a fixture: it should rise when candidates or arches are added,
# and a fall means coverage was silently lost. Set with enough slack that
# removing one candidate does not trip it -- the reachability check below is
# what catches that precisely.
_MIN_BUILDS = 300


class TestEveryRegisteredKernelBuilds(unittest.TestCase):
    def _combinations(self):
        """Every (candidate, arch, dtype, shape) the registries can be asked for."""
        for registry, make_reqs, dtypes in _FAMILIES:
            for candidate in registry.candidates():
                for arch in sorted(candidate.capability.arches):
                    for dtype in dtypes:
                        for request in make_reqs(arch, candidate.spec_id, dtype):
                            yield candidate, arch, dtype, request

    def test_every_candidate_builds_on_every_arch_it_declares(self):
        built = 0
        reached = set()
        for candidate, arch, dtype, request in self._combinations():
            admitted, _ = candidate.admits(request)
            if not admitted:
                # A declared arch can still turn a given shape or dtype down;
                # that is the residual predicate doing its job, not a gap.
                continue
            with self.subTest(candidate=candidate.name, arch=arch, dtype=dtype):
                kernel = candidate.built(candidate.select_spec(request), arch)
                self.assertTrue(getattr(kernel, "name", ""))
                built += 1
                reached.add(candidate.name)
        self.assertGreaterEqual(built, _MIN_BUILDS)
        # Every registered candidate must be reachable by *something*. A
        # candidate no request can reach is dead weight the registry still
        # advertises in its coverage manifest.
        for registry, _, _ in _FAMILIES:
            for candidate in registry.candidates():
                with self.subTest(unreachable=candidate.name):
                    self.assertIn(candidate.name, reached)

    def test_building_the_same_selection_twice_gives_identical_ir(self):
        # Determinism is what makes the compile cache keyed on `compile_key`
        # sound, and what makes the parity comparisons below meaningful.
        for candidate, arch, dtype, request in self._combinations():
            if not candidate.admits(request)[0]:
                continue
            with self.subTest(candidate=candidate.name, arch=arch, dtype=dtype):
                spec = candidate.select_spec(request)
                self.assertTrue(
                    canonical_equal(
                        candidate.built(spec, arch), candidate.built(spec, arch)
                    )
                )


class TestExampleParity(unittest.TestCase):
    """Dispatch vs. the examples' hand-built specs, on the examples' shapes.

    Where the registry carries the configuration an example builds, the IR must
    be identical -- that is what lets the example become a dispatch call site
    instead of a parallel construction of the same kernel.
    """

    ARCH = "gfx950"

    def test_norm_matches_the_example_exactly(self):
        # examples/common/ck_tile_parity.py builds (block_size=256, vec=8) for
        # both kinds. The norm registry enumerates that config space, so the
        # example's spec is reachable by spec_id.
        from rocke.instances.common.layernorm2d import (
            LayerNorm2DSpec,
            build_layernorm2d,
        )
        from rocke.instances.common.rmsnorm2d import RMSNorm2DSpec, build_rmsnorm2d

        cols = NORM_SHAPE["cols"]
        cases = (
            ("layernorm", LayerNorm2DSpec, build_layernorm2d),
            ("rmsnorm", RMSNorm2DSpec, build_rmsnorm2d),
        )
        for kind, spec_cls, builder in cases:
            with self.subTest(kind=kind):
                by_hand = builder(spec_cls(n_per_block=cols, block_size=256, vec=8))
                dispatched = dispatch_norm(
                    NormRequest(
                        kind=kind,
                        arch=self.ARCH,
                        spec_id=f"{kind}_b256_v8",
                        **NORM_SHAPE,
                    )
                ).build()
                self.assertEqual(by_hand.name, dispatched.name)
                self.assertTrue(canonical_equal(by_hand, dispatched))

    def test_moe_matches_the_example_exactly(self):
        # examples/gfx1250/fused_mega_moe/fused_mega_moe_bench.py defaults:
        # tile_m=16, tile_n_inter=256, tile_n_down=256.
        from rocke.instances.common.moe_fused_mega import (
            FusedMegaKernelSpec,
            build_moe_fused_mega_gemm,
        )

        dispatched = dispatch_moe(MoeRequest(arch=self.ARCH, dtype="bf16", **MOE_SHAPE))
        by_hand = build_moe_fused_mega_gemm(
            FusedMegaKernelSpec(
                name=dispatched.spec.name,
                dtype="bf16",
                tile_m=16,
                tile_n_inter=256,
                tile_n_down=256,
            ),
            arch=self.ARCH,
        )
        self.assertTrue(canonical_equal(by_hand, dispatched.build()))

    def test_gemm_deliberately_does_not_match_the_example(self):
        """The one family where dispatch and the example disagree, pinned.

        `universal_gemm_verify.py` is a portability harness: it asks the MMA
        catalog for the largest-K 16x16 atom and wraps a 2x2 warp grid around
        it with the plain `mem` pipeline. That is a config chosen to build
        anywhere, not one chosen to be fast, and the registry does not carry
        it. Dispatch answers with the tuned candidate instead.

        Asserted rather than left implicit, because "the example and the
        dispatcher build different kernels" is exactly the kind of thing a
        reader would otherwise assume was an oversight.
        """
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            build_universal_gemm,
        )

        target = ArchTarget.from_gfx(self.ARCH)
        atom = target.mma.select_largest_k(
            family="mma", a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=16, n=16
        )
        by_hand = build_universal_gemm(
            UniversalGemmSpec(
                name=f"ugemm_{self.ARCH}",
                tile=TileSpec(
                    tile_m=4 * atom.m,
                    tile_n=4 * atom.n,
                    tile_k=max(32, atom.k),
                    warp_m=2,
                    warp_n=2,
                    warp_k=1,
                    warp_tile_m=atom.m,
                    warp_tile_n=atom.n,
                    warp_tile_k=atom.k,
                ),
                trait=TraitSpec(
                    pipeline="mem",
                    scheduler="intrawave",
                    epilogue="default",
                    pad_m=True,
                    pad_n=True,
                    pad_k=True,
                ),
                data=DataSpec(
                    dtype_a="fp16",
                    dtype_b="fp16",
                    dtype_c="fp16",
                    dtype_acc="fp32",
                    layout="RCR",
                ),
                wave_size=target.wave_size,
            ),
            arch=self.ARCH,
        )
        m, n, k = GEMM_MNK
        dispatched = dispatch_gemm_fp16(
            GemmRequest(M=m, N=n, K=k, arch=self.ARCH, dtype="fp16")
        )
        self.assertFalse(canonical_equal(by_hand, dispatched.build()))
        # Name the divergence, so a change in either direction is visible.
        spec = dispatched.spec
        self.assertEqual(spec.trait.pipeline, "compv4")
        self.assertEqual(spec.trait.epilogue, "cshuffle")
        self.assertEqual(
            (spec.tile.tile_m, spec.tile.tile_n), (4 * atom.m * 2, 4 * atom.n * 2)
        )


if __name__ == "__main__":
    unittest.main()
