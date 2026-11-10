# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convolution regressions that used to live in the platform unit tests.

These pin platform-engine behaviour (LDS-layout guards, async-DMA ping-pong
scaffolding, chiplet swizzle, SGPR-pinned LDS base, HIP lowering, CShuffle
reuse barrier, the ``pipeline="basic"`` loader split, IR serialization) on the
library's convolution kernels. They moved here so the platform tests no longer
import ``kernels.*``: the dependency rule is ``library -> platform`` only.

Test and class names are kept from their platform origin for traceability:

  - ``platform/tests/test_rocke.py``: ``TestHelpers``, ``TestInstances``,
    ``TestCdnaPrimitives``, ``TestConvDirectGroupedTransforms`` and
    ``TestHipLoweringCoverage`` conv cases;
  - ``platform/tests/core/test_cshuffle_reuse_barrier.py``: the conv cases;
  - ``platform/tests/core/test_pipeline_basic.py``:
    ``TestIsValidSpecBasicPipeline``, ``TestBuildConvBasicPipeline``;
  - ``platform/tests/core/test_ir_serialize.py``: the conv round-trip specimen
    (``TestConvIrSerialize``).

Host-only: they build and lower IR in-process; no GPU, no subprocess.
"""

from __future__ import annotations

import hashlib
import unittest

import pytest
from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    build_direct_conv_4c,
    build_direct_conv_16c,
)
from kernels.common.conv_implicit_gemm import (
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)
from rocke import lower_kernel_to_llvm
from rocke.helpers import LdsLayout

# ---------------------------------------------------------------------
# From platform/tests/test_rocke.py
# ---------------------------------------------------------------------


class TestHelpers(unittest.TestCase):
    def test_wgrad_async_rejects_non_packed_lds_layout(self):
        """An explicit lds_layout must reach validate_for_async() for wgrad too.

        The scalar ``lds_k_pad`` guard cannot stand in for this: an explicit
        ``lds_layout`` object beats the scalar in ``effective_lds_layout()``, and
        the xor swizzle has no scalar analogue at all. Without the call these
        specs build and only fail deep inside the emitter, which the sweep
        drivers turn into a silent skip. Mirrors the conv/dgrad behaviour.
        """
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_wgrad import (
            WgradConvSpec,
            is_valid_wgrad_spec,
        )

        def _spec(**over):
            return WgradConvSpec(
                problem=ConvProblem(N=1, Hi=8, Wi=8, C=32, K=32, Y=3, X=3, pH=1, pW=1),
                name="wgrad_async_guard",
                data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                wave_size=64,
                pipeline="mem",
                epilogue="cshuffle",
                split_k=1,
                lds_k_outer=True,
                async_dma=True,
                **over,
            )

        # The packed layout the async intrinsic actually writes is accepted.
        _spec(lds_layout=LdsLayout.packed_async(64)).validate()

        for bad in (
            LdsLayout.padded_k(64, 8),
            LdsLayout(logical_cols=64, swizzle="xor"),
        ):
            spec = _spec(lds_layout=bad)
            with self.assertRaises(ValueError):
                spec.validate()
            # And the soft validator reports it rather than skipping silently.
            ok, why = is_valid_wgrad_spec(spec, "gfx950")
            self.assertFalse(ok)
            self.assertTrue(why)


class TestInstances(unittest.TestCase):
    def test_implicit_gemm_conv_builds(self):
        prob = ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
        )
        kernel = build_implicit_gemm_conv(spec)
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", ll)
        # The buffer rsrc DW3 flag-word must be 0x00027000, not 0 — the
        # critical correctness fix from the bake-off debugging session.
        self.assertIn("159744", ll)  # 0x27000 = 159744

    def test_implicit_gemm_async_rejects_padded_lds(self):
        prob = ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            async_dma=True,
            lds_k_pad=8,
        )
        with self.assertRaises(ValueError):
            build_implicit_gemm_conv(spec)

    def test_implicit_gemm_async_emits_pingpong_artifacts(self):
        """Async-DMA conv must emit the ping-pong scaffolding.

        We sanity-check that the async-DMA path produces:
          * ``raw_ptr_buffer_load_lds`` intrinsics (the actual async DMA).
          * ``s_setprio`` bookends from the interwave scheduler (the
            high-prio / low-prio pair around each MFMA group).
          * ``s_waitcnt`` with a partial ``vmcnt`` (overlap of the
            next iter's load with this iter's compute), encoded as
            anything other than the all-zero drain.
          * The ``s_barrier`` count is at least 2 per K-iter (one
            ``sync_lds_only`` before each issue + one after the wait),
            confirming the pong-hazard guard is in place.
        """
        prob = ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            async_dma=True,
        )
        kernel = build_implicit_gemm_conv(spec)
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        # 1) The async DMA intrinsic itself is emitted.
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", ll)
        # 2) Interwave ping-pong: setprio bookends. We expect *both*
        # the high (1) and low (0) prio settings from the interwave
        # SchedulePolicy.
        self.assertIn("@llvm.amdgcn.s.setprio(i16 1)", ll)
        self.assertIn("@llvm.amdgcn.s.setprio(i16 0)", ll)
        # 3) At least one barrier per K-iter for the ABA-hazard guard
        # (K_iters = 576 / 64 = 9 in this shape, but counting exactly
        # is brittle; require >= 8 barriers as a lower bound).
        barrier_count = ll.count("@llvm.amdgcn.s.barrier")
        self.assertGreaterEqual(
            barrier_count,
            8,
            msg=f"expected >= 8 s_barriers for ping-pong, found {barrier_count}",
        )

    def test_direct_conv_16c_builds(self):
        prob = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        spec = DirectConv16cSpec(problem=prob, block_groups=4, fold_k32=True)
        kernel = build_direct_conv_16c(spec)
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        # K32-folded hot loop emits ONLY the wide 16x16x32 MFMA: S=0/1 fold
        # into one wide atom and the S=2 residual is promoted to a SECOND
        # wide atom (zero-padded upper K) so both atoms on the accumulator
        # are the same width. Mixing a 16x16x16 residual into the same
        # accumulator triggered a cross-width MFMA accumulator hazard that
        # both comgr and hipcc miscompiled at the H-edges; the all-wide fold
        # is bit-correct.
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll)
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll)
        # The unfolded (gfx942-capable) path still uses only 16x16x16.
        spec_nf = DirectConv16cSpec(problem=prob, block_groups=4, fold_k32=False)
        ll_nf = lower_kernel_to_llvm(build_direct_conv_16c(spec_nf), arch="gfx950")
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll_nf)
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll_nf)

    def test_direct_conv_4c_builds(self):
        prob = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        spec = DirectConv4cSpec(problem=prob, block_q=8, block_groups=16)
        kernel = build_direct_conv_4c(spec)
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        # 4x4x4 atom emits one MFMA per (r, s) tile (9 per output row).
        self.assertIn("@llvm.amdgcn.mfma.f32.4x4x4f16", ll)


class TestCdnaPrimitives(unittest.TestCase):
    def test_implicit_gemm_conv_chiplet_swizzle_compiles(self):
        from kernels.common.conv_implicit_gemm import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        prob = ConvProblem(
            N=4,
            Hi=14,
            Wi=14,
            C=128,
            K=128,
            Y=3,
            X=3,
            sH=1,
            sW=1,
            pH=1,
            pW=1,
            dH=1,
            dW=1,
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            chiplet_swizzle=True,
            chiplet_wgm=4,
            waves_per_eu=2,
        )
        ll = lower_kernel_to_llvm(build_implicit_gemm_conv(spec), arch="gfx950")
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", ll)
        self.assertIn('"amdgpu-waves-per-eu"="2,2"', ll)
        # The chiplet swizzle takes both blockIdx.x and blockIdx.y.
        self.assertIn("@llvm.amdgcn.workgroup.id.x", ll)
        self.assertIn("@llvm.amdgcn.workgroup.id.y", ll)

    def test_implicit_gemm_conv_async_uses_sgpr_pinned_lds_base(self):
        """The async-DMA conv must hoist the per-wave LDS base into
        an SGPR via ``to_sgpr_u32`` (``readfirstlane`` + SGPR-pin asm).
        """
        from kernels.common.conv_implicit_gemm import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        prob = ConvProblem(
            N=8,
            Hi=56,
            Wi=56,
            C=64,
            K=64,
            Y=3,
            X=3,
            sH=1,
            sW=1,
            pH=1,
            pW=1,
            dH=1,
            dW=1,
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            async_dma=True,
        )
        ll = lower_kernel_to_llvm(build_implicit_gemm_conv(spec), arch="gfx950")
        # readfirstlane for wave-uniform LDS offset.
        self.assertIn("@llvm.amdgcn.readfirstlane.i32", ll)
        # SGPR-pin asm constraint.
        self.assertIn('asm "", "=s,0"(i32', ll)
        # CACHE_STREAM coherency on the async loads (aux i32 = 2).
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", ll)


class TestConvDirectGroupedTransforms(unittest.TestCase):
    """The transform-descriptor migration must keep the kernels building."""

    def test_16c_kernel_lowers_to_llvm(self):
        from kernels.common.conv_direct_grouped import (
            DirectConv16cSpec,
            DirectConvProblem,
            build_direct_conv_16c,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = DirectConv16cSpec(
            problem=DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16)
        )
        kernel = build_direct_conv_16c(spec)
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        # Smoke check: the generated LLVM IR mentions amdgpu and the
        # kernel name (proves the body got emitted).
        self.assertIn("amdgpu", ll)
        self.assertIn(kernel.name, ll)

    def test_4c_kernel_lowers_to_llvm(self):
        from kernels.common.conv_direct_grouped import (
            DirectConv4cSpec,
            DirectConvProblem,
            build_direct_conv_4c,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = DirectConv4cSpec(
            problem=DirectConvProblem(N=1, H=8, W=8, groups=16, cpg=4, kpg=4)
        )
        kernel = build_direct_conv_4c(spec)
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("amdgpu", ll)
        self.assertIn(kernel.name, ll)


class TestHipLoweringCoverage(unittest.TestCase):
    """:func:`rocke.core.lower_kernel_to_hip` must lower the conv kernels
    without ``NotImplementedError`` (see the platform class of the same name
    for the full contract)."""

    def _lower_and_check(self, kernel, must_contain):
        from rocke.core.lower_hip import lower_kernel_to_hip

        hip = lower_kernel_to_hip(kernel)
        self.assertIn("#include <hip/hip_runtime.h>", hip)
        self.assertIn("__global__", hip)
        self.assertIn(f"void {kernel.name}(", hip)
        for token in must_contain:
            self.assertIn(
                token,
                hip,
                f"lowered HIP source for {kernel.name!r} missing required token "
                f"{token!r}",
            )

    def test_implicit_gemm_conv_lowers(self):
        from kernels.common.conv_implicit_gemm import (
            ConvProblem,
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )

        p = ConvProblem(
            N=8,
            Hi=56,
            Wi=56,
            C=64,
            K=64,
            Y=3,
            X=3,
            sH=1,
            sW=1,
            pH=1,
            pW=1,
            dH=1,
            dW=1,
        )
        spec = ImplicitGemmConvSpec(
            problem=p,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
        )
        self._lower_and_check(
            build_implicit_gemm_conv(spec),
            must_contain=[
                "__builtin_amdgcn_make_buffer_rsrc",
                "__builtin_amdgcn_raw_buffer_load",
                "__builtin_amdgcn_raw_buffer_store",
                "__builtin_amdgcn_mfma_f32_32x32x16_f16",
            ],
        )


# ---------------------------------------------------------------------
# From platform/tests/core/test_cshuffle_reuse_barrier.py (PR #8844 guard)
# ---------------------------------------------------------------------

_BARRIERS = ("tile.sync", "tile.sync_lds_only", "tile.s_barrier_bare")


def _flatten(region, out):
    for op in region.ops:
        out.append(op)
        for r in op.regions:
            _flatten(r, out)
    return out


def _assert_barrier_before_first_c_store(kernel) -> None:
    """The first smem store after the C_smem allocation must be preceded by a
    workgroup barrier that itself follows the allocation (i.e. no C store may
    race the just-completed A/B reads on the aliased pool)."""
    ops = _flatten(kernel.body, [])

    c_allocs = [
        i
        for i, o in enumerate(ops)
        if o.name == "tile.smem_alloc" and "C_smem" in o.results[0].name
    ]
    assert c_allocs, "expected a C_smem allocation in a cshuffle kernel"
    c_idx = c_allocs[0]

    later_stores = [
        i
        for i, o in enumerate(ops)
        if o.name.startswith("tile.smem_store") and i > c_idx
    ]
    assert later_stores, "expected at least one store into the C staging tile"
    first_store = later_stores[0]

    barriers = [i for i in range(c_idx + 1, first_store) if ops[i].name in _BARRIERS]
    assert barriers, (
        "no workgroup barrier between the C_smem allocation and the first C "
        "store: a fast wave's C write could clobber A/B bytes a slow wave is "
        "still reading for its tail MFMA (cross-wave WAR on the aliased pool)"
    )
    # And no C store may sneak in before that barrier.
    assert not any(
        ops[i].name.startswith("tile.smem_store") for i in range(c_idx + 1, barriers[0])
    ), "a C store precedes the reuse barrier"


def test_conv_cshuffle_has_reuse_barrier():
    """The conv implicit-GEMM cshuffle path (double-buffered compv4, via the
    shared ``CShuffleEpilogue`` helper) emits the reuse barrier before the first
    C store."""
    from kernels.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
    )

    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        name="c",
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        wave_size=64,
        pipeline="compv4",
        epilogue="cshuffle",
    )
    _assert_barrier_before_first_c_store(build_implicit_gemm_conv(spec, arch="gfx950"))


def test_conv_cshuffle_no_alias_lds_budget_is_per_case():
    """is_valid_spec must size the LDS budget per ``cshuffle_no_alias``: aliased
    uses max(ab, c); no-alias uses ab + c. A spec whose max(ab, c) fits gfx950's
    LDS but whose ab + c overflows it must be accepted aliased and rejected
    no-alias (regression for yraparti's PR #8844 review comment)."""
    from kernels.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        is_valid_spec,
    )

    def _mk(no_alias):
        return ImplicitGemmConvSpec(
            problem=ConvProblem(N=4, Hi=28, Wi=28, C=64, K=64, Y=3, X=3),
            tile_m=256,
            tile_n=256,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            wave_size=64,
            cshuffle_no_alias=no_alias,
        )

    ok_alias, _ = is_valid_spec(_mk(False), "gfx950")
    ok_noalias, why = is_valid_spec(_mk(True), "gfx950")
    assert ok_alias, "aliased cshuffle (pool = max(ab, c)) should fit gfx950 LDS"
    assert not ok_noalias, (
        "no-alias cshuffle (pool = ab + c) should exceed gfx950 LDS and be "
        "rejected; sizing it as max(ab, c) would silently accept an over-budget "
        "kernel"
    )
    assert "LDS budget" in why


@pytest.mark.parametrize("cshuffle_no_alias", [False, True])
def test_wavelet_cshuffle_both_branches_have_equal_barrier_count(cshuffle_no_alias):
    """Both branches of the wavelet scf.if_else must emit the same number of
    barriers, or the workgroup hangs at a barrier one role issues but the other
    does not.

    The math branch ends with CShuffleEpilogue (N_epi barriers inside it).
    The load branch ends with a bare stub that emits N_epi raw s_barrier calls.
    This test builds the conv IR and counts barriers in each region to confirm
    both sides agree — a regression guard so future edits to either the epilogue
    or the stub cannot silently diverge.
    """
    from kernels.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
    )

    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, pH=1, pW=1),
        name="wv_barrier_check",
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=1,
        warp_n=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        wave_size=32,
        pipeline="wavelet",
        epilogue="cshuffle",
        num_load_waves=1,
        cshuffle_no_alias=cshuffle_no_alias,
    )
    kernel = build_implicit_gemm_conv(spec, arch="gfx1250")

    ops = _flatten(kernel.body, [])

    # Locate the scf.if_else that splits math vs load waves.
    if_else_ops = [o for o in ops if o.name == "scf.if_else"]
    assert if_else_ops, "expected an scf.if_else in the wavelet kernel"
    ife = if_else_ops[0]

    def _count_barriers_in_region(region):
        buf = []
        _flatten(region, buf)
        return sum(1 for o in buf if o.name in _BARRIERS)

    then_count = _count_barriers_in_region(ife.regions[0])
    else_count = _count_barriers_in_region(ife.regions[1])

    assert then_count == else_count, (
        f"barrier count mismatch (cshuffle_no_alias={cshuffle_no_alias}): "
        f"math branch has {then_count}, load branch has {else_count}. "
        "A workgroup hang will occur at the barrier one role skips."
    )
    assert then_count > 0, "expected at least one barrier in each wavelet branch"


# ---------------------------------------------------------------------
# From platform/tests/core/test_pipeline_basic.py
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# is_valid_spec: pipeline="basic" + async_dma=True must be rejected
# ---------------------------------------------------------------------------


class TestIsValidSpecBasicPipeline:
    def _mk_basic_spec(self, async_dma=False):
        from kernels.common.conv_implicit_gemm import (
            ConvProblem,
            ImplicitGemmConvSpec,
        )

        return ImplicitGemmConvSpec(
            problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1),
            name="t",
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            wave_size=64,
            pipeline="basic",
            async_dma=async_dma,
            vector_size_c=1,
        )

    def test_basic_sync_is_valid(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        ok, why = is_valid_spec(self._mk_basic_spec(async_dma=False), "gfx950")
        assert ok, f"pipeline='basic' with async_dma=False should be valid: {why}"

    def test_basic_async_dma_is_rejected(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        ok, why = is_valid_spec(self._mk_basic_spec(async_dma=True), "gfx950")
        assert not ok, "pipeline='basic' with async_dma=True must be rejected"
        assert (
            "async_dma" in why.lower() or "basic" in why.lower()
        ), f"rejection message should mention the conflict, got: {why!r}"


# ---------------------------------------------------------------------------
# build_implicit_gemm_conv: pipeline="basic" produces correct IR structure
# ---------------------------------------------------------------------------


class TestBuildConvBasicPipeline:
    def _build(self, epilogue="default"):
        from kernels.common.conv_implicit_gemm import (
            ConvProblem,
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )

        spec = ImplicitGemmConvSpec(
            problem=ConvProblem(N=1, Hi=8, Wi=8, C=16, K=32, Y=3, X=3, pH=1, pW=1),
            name="basic_test",
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            wave_size=64,
            pipeline="basic",
            epilogue=epilogue,
            vector_size_c=1,
        )
        return build_implicit_gemm_conv(spec, arch="gfx950")

    def test_build_produces_kernel(self):
        kernel = self._build()
        assert kernel is not None

    def test_build_contains_buffer_load_ops(self):
        """The basic pipeline uses CoalescedTileLoader.load_global which emits
        tile.buffer_load_vN instructions."""
        kernel = self._build()
        ops = _flatten(kernel.body, [])
        load_ops = [op for op in ops if "buffer_load" in op.name]
        assert load_ops, "pipeline='basic' kernel must contain buffer_load ops"

    def test_build_contains_smem_store_ops(self):
        """After the global read, store_lds emits tile.smem_store_vN."""
        kernel = self._build()
        ops = _flatten(kernel.body, [])
        store_ops = [op for op in ops if "smem_store" in op.name]
        assert store_ops, "pipeline='basic' kernel must contain smem_store ops"

    def test_build_contains_sync_ops(self):
        """The basic pipeline emits workgroup barriers (sync) to order reads
        and writes between tiles."""
        kernel = self._build()
        ops = _flatten(kernel.body, [])
        sync_ops = [op for op in ops if "sync" in op.name or "barrier" in op.name]
        assert sync_ops, "pipeline='basic' kernel must contain barrier/sync ops"

    def test_build_cshuffle_epilogue(self):
        """Ensure pipeline='basic' composes correctly with the cshuffle epilogue."""
        kernel = self._build(epilogue="cshuffle")
        assert kernel is not None
        ops = _flatten(kernel.body, [])
        # cshuffle allocates a C staging tile
        c_allocs = [
            op
            for op in ops
            if op.name == "tile.smem_alloc" and "C_smem" in op.results[0].name
        ]
        assert c_allocs, "cshuffle epilogue must allocate a C_smem tile"

    def test_build_uses_scf_for_iter(self):
        """pipeline='basic' now emits a bounded runtime scf.for K-loop
        (same structure as 'mem') so IR size is independent of K_gemm."""
        kernel = self._build()
        ops = _flatten(kernel.body, [])
        for_ops = [op for op in ops if op.name == "scf.for"]
        assert for_ops, "pipeline='basic' K-loop must use scf.for (runtime loop)"


# ---------------------------------------------------------------------
# From platform/tests/core/test_ir_serialize.py: the conv specimen of its
# real-kernel corpus. The platform corpus keeps its platform kernels.
# ---------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TestConvIrSerialize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = build_implicit_gemm_conv(
            ImplicitGemmConvSpec(
                problem=ConvProblem(
                    N=1, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1
                ),
                epilogue="cshuffle",
            )
        )

    def test_roundtrip_byte_identical(self):
        """serialize -> parse -> serialize is byte-identical for the conv kernel."""
        from rocke.core.ir_serialize import parse, serialize

        s1 = serialize(self.kernel)
        s2 = serialize(parse(s1))
        self.assertEqual(_sha(s1), _sha(s2), "conv: round-trip sha differs")
        self.assertEqual(s1, s2, "conv: round-trip text differs")

    def test_roundtrip_lowers_identically(self):
        """The parsed KernelDef lowers to the SAME LLVM IR as the original."""
        from rocke.core.ir_serialize import parse, serialize

        k2 = parse(serialize(self.kernel))
        ll1 = lower_kernel_to_llvm(self.kernel, arch="gfx950")
        ll2 = lower_kernel_to_llvm(k2, arch="gfx950")
        self.assertEqual(_sha(ll1), _sha(ll2), "conv: lowered LLVM sha differs")

    def test_serialization_is_versioned(self):
        from rocke.core.ir_serialize import serialize

        self.assertTrue(serialize(self.kernel).startswith("ckdsl.ir v1\n"))

    def test_canonical_equal_self(self):
        from rocke.core.ir_serialize import canonical_equal, parse, serialize

        self.assertTrue(canonical_equal(self.kernel, parse(serialize(self.kernel))))

    def test_valid_kernels_have_no_errors(self):
        from rocke.core.verify import ERROR, verify, verify_or_raise

        errors = [d for d in verify(self.kernel) if d.severity == ERROR]
        self.assertEqual(errors, [], f"conv: {[str(e) for e in errors]}")
        verify_or_raise(self.kernel)  # must not raise
