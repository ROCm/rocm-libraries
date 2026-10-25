# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fused operand prologue (``gemm_prologue``): knobs, validation and emission shape.

Host-side contract only (the device numerics need a GPU): no prologue = unchanged emission, the name suffix
encodes every knob, the f32 pair view halves the K atoms, each recipe issues its product count of MFMAs per
atom, and the bodies the prologue was not built for are rejected.
"""

import pytest
from rocke.instances.common.gemm_prologue import RECIPES, Prologue, attach_prologue
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


def _spec(name="pro_t"):
    tile = TileSpec(
        tile_m=256,
        tile_n=256,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        direct_to_lds=True,
        dtl_prefetch=True,
        lds_swizzle=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        chiplet_swizzle=True,
        dtl_stages=2,
        interwave_setprio=True,
        acc_agpr=True,
        dtl_hoist=True,
        lds_swizzle_r=1,
    )
    return UniversalGemmSpec(
        name=name,
        tile=tile,
        trait=trait,
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        block_size=256,
    )


def _mma_count(k):
    from rocke.helpers.compile import compile_kernel

    art = compile_kernel(k, arch="gfx950", capture_ir_text=True, backend="python")
    return art.llvm_text.count("call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(")


def test_no_prologue_is_unchanged():
    from rocke.helpers.compile import compile_kernel

    a = compile_kernel(
        build_universal_gemm(_spec(), "gfx950"),
        arch="gfx950",
        capture_ir_text=True,
        backend="python",
    ).llvm_text
    b = compile_kernel(
        build_universal_gemm(_spec(), "gfx950"),
        arch="gfx950",
        capture_ir_text=True,
        backend="python",
    ).llvm_text
    assert a == b
    assert "_pro" not in build_universal_gemm(_spec(), "gfx950").name


@pytest.mark.parametrize(
    "pro,suffix",
    [
        (Prologue(src="f32", recipe="cvt", out_f32=True), "_pro_f32cvt_of32"),
        (
            Prologue(src="f32", recipe="x3", psub=True, split_at="use"),
            "_pro_f32x3_su_pk",
        ),
        (
            Prologue(src="f32", recipe="x6", order="big_first", cap=True),
            "_pro_f32x6_cap_bf",
        ),
        (Prologue(unary_a="relu", unary_impl="bits"), "_pro_uarelu_uibits"),
        (Prologue(src="f32", recipe="cvt", stage="vgpr"), "_pro_f32cvt_vg"),
        (Prologue(unary_b="exp", fast_exp=False), "_pro_ubexp_gexp"),
    ],
)
def test_suffix_encodes_knobs(pro, suffix):
    assert pro.suffix() == suffix


@pytest.mark.parametrize(
    "kw",
    [
        dict(recipe="x3"),  # split recipes need f32 operands
        dict(src="f32", recipe="x5"),
        dict(unary_a="tanh"),
        dict(src="f32", recipe="x3", stage="vgpr"),  # vgpr staging is built for cvt
        dict(order="sideways"),
    ],
)
def test_invalid_prologues_rejected(kw):
    with pytest.raises(ValueError):
        Prologue(**kw).check()


def test_pair_view_kmul():
    assert Prologue(src="f32").kmul == 2
    assert Prologue(src="f32", stage="vgpr").kmul == 1
    assert Prologue(unary_a="abs").kmul == 1


@pytest.mark.parametrize("recipe", ["cvt", "x3", "x6"])
def test_recipe_mfma_count(recipe):
    base = _mma_count(build_universal_gemm(_spec("pro_base"), "gfx950"))
    sp = attach_prologue(
        _spec("pro_" + recipe), Prologue(src="f32", recipe=recipe, out_f32=True)
    )
    k = build_universal_gemm(sp, "gfx950")
    n = _mma_count(k)
    # pair view: tile_k 64 halves = 32 f32 = half the K atoms of the 16-bit body, times the product count
    assert n * 2 == base * len(RECIPES[recipe][1])
    assert any(p.name == "C" and "f32" in p.type.name for p in k.params)


_RILV = dict(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=False,
    chiplet_swizzle=True,
    chiplet_wgm=16,
    chiplet_chunk_size=8,
    acc_agpr=True,
    dtl_stages=2,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    dtl_hoist=True,
    lds_row_interleave=True,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
)


@pytest.mark.parametrize(
    "lever",
    [
        dict(sia_sched=True, sia_plr=True, sia_dtl_spread=True, sia_acq_gap=12),
        dict(hb_sched=True, hb_read_every=1, asm_loop="tensile"),
    ],
)
def test_unsupported_bodies_rejected(lever):
    from rocke.instances.common.gemm_universal import is_valid_spec

    tile = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
    sp = UniversalGemmSpec(
        name="pro_bad",
        tile=tile,
        trait=TraitSpec(**_RILV, **lever),
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
    )
    assert is_valid_spec(sp, arch="gfx950")[0]
    build_universal_gemm(sp, "gfx950")  # the body itself builds
    sp = attach_prologue(sp, Prologue(src="f32", recipe="x6", out_f32=True))
    with pytest.raises(ValueError, match="sia_sched / asm_loop"):
        build_universal_gemm(sp, "gfx950")
