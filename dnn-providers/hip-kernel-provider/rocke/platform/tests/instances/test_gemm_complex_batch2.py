# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM side channels for complex and batched contractions: complex B real embedding (``Prologue.cplx_b``, 4M
complex-f32 in one kernel) and the two-mode batch (``spec._batch2``).

Host-side contract only (the device numerics need a GPU): knob validation, name suffixes, the extra kernel
arguments, and that the complex B rows are addressed as ``r >> 1`` (one more shift / select per fragment, never
a different MFMA count).
"""

from dataclasses import replace

import pytest
from rocke.instances.common.gemm_prologue import Prologue, attach_prologue
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


def _spec(name="cx_t", **kw):
    tile = TileSpec(
        tile_m=256,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    t = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        direct_to_lds=True,
        dtl_prefetch=True,
        lds_swizzle=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        chiplet_swizzle=True,
        dtl_stages=3,
        interwave_setprio=True,
        acc_agpr=True,
        dtl_hoist=True,
        lds_swizzle_r=1,
    )
    t.update(kw)
    return UniversalGemmSpec(
        name=name,
        tile=tile,
        trait=TraitSpec(**t),
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        block_size=256,
    )


def _cx(**kw):
    d = dict(src="f32", recipe="x6", out_f32=True, split_at="use", cplx_b=True)
    d.update(kw)
    return Prologue(**d)


@pytest.mark.parametrize(
    "kw",
    [
        dict(split_at="load"),
        dict(src="same", recipe="cvt"),
        dict(cplx_b=False, conj_b=True),
    ],
)
def test_cplx_rejects(kw):
    with pytest.raises(ValueError):
        _cx(**kw).check()


def test_cplx_suffix_and_conj():
    assert _cx().suffix().endswith("_cx")
    assert _cx(conj_a=True, conj_b=True).suffix().endswith("_cxab")
    assert _cx(conj_b=True).suffix().endswith("_cxb")


def _mfma_and_ops(k):
    from rocke.core.passes import optimize_kernel
    from rocke.helpers.compile import _lower_llvm_via_backend

    optimize_kernel(k)
    t = _lower_llvm_via_backend(k, arch="gfx950", backend="python", spec=None)
    return t.count("@llvm.amdgcn.mfma.f32.16x16x32.bf16("), t


def test_cplx_same_mfma_count_lane_parity():
    plain = attach_prologue(_spec("cx_p"), _cx(cplx_b=False))
    cx = attach_prologue(_spec("cx_c"), _cx())
    n0, t0 = _mfma_and_ops(build_universal_gemm(plain, "gfx950"))
    n1, t1 = _mfma_and_ops(build_universal_gemm(cx, "gfx950"))
    assert n0 == n1 > 0
    # lane-parity mode (non row-interleaved body): per-lane selects appear
    assert t1.count(" select ") > t0.count(" select ")


def test_cplx_rejects_row_interleave_without_vec_epilogue():
    sp = attach_prologue(
        _spec(
            "cx_ri",
            lds_row_interleave=True,
            lds_swizzle=False,
            interwave_setprio=False,
            dtl_stages=2,
            dtl_frag_prefetch=True,
        ),
        _cx(),
    )
    with pytest.raises(ValueError):
        build_universal_gemm(sp, "gfx950")


def test_batch2_params():
    sp = replace(_spec("cx_b2"), batched=True)
    object.__setattr__(sp, "_batch2", True)
    names = [p.name for p in build_universal_gemm(sp, "gfx950").params]
    assert names[-7:] == [
        "stride_a",
        "stride_b",
        "stride_c",
        "stride_a2",
        "stride_b2",
        "stride_c2",
        "L2",
    ]
    sp1 = replace(_spec("cx_b1"), batched=True)
    assert "L2" not in [p.name for p in build_universal_gemm(sp1, "gfx950").params]


# ------------------------------------------------------------------ launch contract (>= 2 GiB complex rows)


def _cx_launch_error(M, Nc, Kc, **kw):
    from rocke.instances.common.gemm_universal import launch_shape_error

    sp = attach_prologue(_spec("cx_l", **kw), _cx())
    # kernel extents: N = 2 Nc real rows of B', K = 4 Kc 16-bit halves
    return launch_shape_error(sp, M, 2 * Nc, 4 * Kc)


@pytest.mark.parametrize(
    "M,Nc,Kc,why",
    [
        # A (2.4 GB) and the complex B (2.26 GB) past the DMA buffer range
        (24576, 23040, 12288, "DTL buffer range"),
        # D = M x 2Nc f32 has more elements than the i32 offsets address
        (46080, 46080, 1536, "elements"),
        (46080, 46080, 2304, "elements"),
    ],
)
def test_cplx_launch_contract_refuses_oversized_problems(M, Nc, Kc, why):
    err = _cx_launch_error(M, Nc, Kc)
    assert err is not None and why in err, err


def test_cplx_launch_contract_counts_complex_b_rows():
    # B' has 2 Nc kernel rows but Nc complex rows in memory: 16384 x 16000
    # interleaved complex f32 is 2.1 GB (inside the range); counting the 2 Nc
    # kernel rows would double it and refuse
    assert _cx_launch_error(8192, 16384, 16000) is None


def test_cplx_launch_contract_needs_whole_complex_rows():
    from rocke.instances.common.gemm_universal import launch_shape_error

    sp = attach_prologue(
        _spec("cx_s", pad_m=True, pad_n=True, dtl_tight_rsrc=True), _cx()
    )
    assert launch_shape_error(sp, 1000, 2046, 4096) is None
    assert "even" in launch_shape_error(sp, 1000, 2047, 4096)


def test_prologue_needs_k_contiguous_operands():
    sp = _spec("cx_nn")
    sp = replace(sp, data=replace(sp.data, layout="RRR"))
    with pytest.raises(ValueError, match="RCR"):
        build_universal_gemm(attach_prologue(sp, _cx()), "gfx950")
