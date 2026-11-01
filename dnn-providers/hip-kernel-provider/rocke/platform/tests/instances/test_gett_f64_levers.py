# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the gett_f64 Python-engine levers: AGPR accumulators,
register-pipelined fragments, the instruction-level interleaved K loop,
K-major LDS tiles, back-to-back MFMA chains and the in-launch split-K
reduction, plus the pipelined dispatch candidate."""

from __future__ import annotations

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.dispatch.families.gett_f64 import GettF64Request, dispatch_gett_f64
from rocke.instances.common.gett_f64 import (
    PIPE_CONFIGS,
    arch_spec,
    build_gett_f64,
    gemm_problem,
    gett_f64_counter_count,
    gett_f64_grid,
    gett_f64_pipe_class,
    gett_f64_signature,
    gett_f64_workspace_bytes,
    is_valid_spec,
    pipe_spec,
    tuned_spec,
)

ARCHS = ("gfx90a", "gfx942", "gfx950")


def _kk(M=512, N=384, K=256):
    """Both operands K-contiguous (A row-major, B column-major as N x K)."""
    return gemm_problem(M, N, K, a_strides=(K, 1), b_strides=(K, 1), d_strides=(1, M))


def _mn(M=512, N=384, K=256):
    """Both operands M / N-contiguous."""
    return gemm_problem(M, N, K, a_strides=(1, M), b_strides=(1, N), d_strides=(1, M))


def _ops(kernel, name):
    out = []

    def walk(ops):
        for op in ops:
            if op.name == name:
                out.append(op)
            for r in getattr(op, "regions", ()) or ():
                walk(r.ops if hasattr(r, "ops") else r)

    walk(kernel.body.ops if hasattr(kernel.body, "ops") else kernel.body)
    return out


@pytest.mark.parametrize(
    "kw,needle",
    [
        (dict(ilv=8), "ilv needs"),  # prefetch depth 1
        (dict(ilv=8, prefetch_depth=2, frag_pipe=True), "ilv needs"),
        (dict(ilv=8, prefetch_depth=2, k_block=4), "ilv needs"),
        (dict(ilv=-1), "ilv and ilv_wgap"),
        (dict(frag_pipe=True, lds_stages=1), "frag_pipe needs"),
        (dict(frag_pipe=True, prefetch_depth=2), "frag_pipe needs"),
        (dict(k_inner=3), "k_inner must divide"),
        (dict(split_serial=1), "split_serial needs"),
        (dict(split_serial=3, split_k=2), "split_serial must be"),
        (dict(kmajor_lds=True, kmajor_pad=1), "kmajor_lds needs"),
        (dict(unroll2=True, prefetch_depth=2), "unroll2 needs"),
        (dict(acc_agpr=True, tile_m=256, tile_n=256, warps_m=4, warps_n=2), "AGPRs"),
    ],
)
def test_lever_combinations_are_checked(kw, needle):
    ok, why = is_valid_spec(arch_spec(_kk(), "gfx950", **kw), arch="gfx950")
    assert not ok and needle in why, why


def test_complex_refuses_the_real_only_levers():
    p = gemm_problem(
        128,
        128,
        64,
        a_strides=(64, 1),
        b_strides=(64, 1),
        d_strides=(1, 128),
        complex_=True,
    )
    for kw in (
        dict(kmajor_lds=True),
        dict(k_inner=2),
        dict(ilv=4, prefetch_depth=2, k_inner=2),
    ):
        assert not is_valid_spec(arch_spec(p, "gfx950", **kw), arch="gfx950")[0], kw
    for kw in (
        dict(acc_agpr=True),
        dict(frag_pipe=True),
        dict(ilv=4, prefetch_depth=2),
    ):
        assert is_valid_spec(arch_spec(p, "gfx950", **kw), arch="gfx950")[0], kw


def test_write_through_split_serial_is_gfx942_gfx950_only():
    s = arch_spec(_kk(K=4096), "gfx90a", split_k=4, split_serial=2)
    assert not is_valid_spec(s, arch="gfx90a")[0]
    assert is_valid_spec(
        arch_spec(_kk(K=4096), "gfx90a", split_k=4, split_serial=1), arch="gfx90a"
    )[0]
    for arch in ("gfx942", "gfx950"):
        s = arch_spec(_kk(K=4096), arch, split_k=4, split_serial=2)
        assert is_valid_spec(s, arch=arch)[0]


def test_levers_name_their_kernels_and_defaults_do_not():
    base = arch_spec(_kk(), "gfx950")
    names = {base.kernel_name()}
    for kw in (
        dict(acc_agpr=True),
        dict(frag_pipe=True),
        dict(kmajor_lds=True),
        dict(kmajor_lds=True, kmajor_pad=4),
        dict(ilv=8, prefetch_depth=2),
        dict(ilv=8, ilv_wgap=2, prefetch_depth=2),
        dict(k_inner=2),
        dict(mfma_fence=True),
        dict(split_k=2, split_serial=1),
        dict(split_k=2, split_serial=2),
    ):
        n = arch_spec(_kk(), "gfx950", **kw).kernel_name()
        assert n not in names, kw
        names.add(n)
    assert "agpr" not in base.kernel_name() and "_il" not in base.kernel_name()


def test_split_serial_signature_workspace_and_counters():
    p = _kk(M=520, N=384, K=4096)
    s = arch_spec(p, "gfx950", tile_m=64, tile_n=64, split_k=4, split_serial=2)
    names = [a["name"] for a in gett_f64_signature(s)]
    assert names[:4] == ["A", "B", "C", "D"] and names[-2:] == ["W", "Cnt"]
    tiles = 9 * 6
    assert gett_f64_counter_count(s) == tiles * 4  # one per (batch, tile, wave)
    # one slot per (split, tile, wave): the wave's 4 atoms x 4 values x 64 lanes
    assert gett_f64_workspace_bytes(s) == 4 * tiles * 4 * (4 * 4 * 64 * 8)
    assert gett_f64_grid(s) == (tiles, 1, 4)
    two = arch_spec(p, "gfx950", tile_m=64, tile_n=64, split_k=4)
    assert gett_f64_counter_count(two) == 0
    assert [a["name"] for a in gett_f64_signature(two)] == ["A", "B", "W"]


@pytest.mark.parametrize("mode", [1, 2])
def test_split_serial_kernel_structure(mode):
    s = arch_spec(
        _kk(K=4096), "gfx950", tile_m=64, tile_n=64, split_k=4, split_serial=mode
    )
    ll = _lower_kernel_to_llvm_python(build_gett_f64(s, arch="gfx950"), arch="gfx950")
    assert ll.count("atomicrmw add") == 1
    if mode == 1:
        assert 'fence syncscope("agent") release' in ll
        assert 'fence syncscope("agent") acquire' in ll
        assert "sc1" not in ll
    else:
        assert "fence syncscope" not in ll
        assert (
            "global_store_dwordx4" in ll and "global_load_dwordx4" in ll and "sc1" in ll
        )


def test_acc_agpr_reserves_the_accumulators():
    s = arch_spec(_kk(), "gfx950", tile_m=128, tile_n=128, acc_agpr=True)
    k = build_gett_f64(s, arch="gfx950")
    assert k.attrs["agpr_alloc"] == (128, 128)  # 16 atoms x 4 f64 x 2 registers
    assert k.attrs["mfma_agpr_form"] is True
    ll = _lower_kernel_to_llvm_python(k, arch="gfx950")
    assert '"amdgpu-agpr-alloc"="128,128"' in ll and '"=a,0"' in ll
    plain = build_gett_f64(
        arch_spec(_kk(), "gfx950", tile_m=128, tile_n=128), arch="gfx950"
    )
    assert "agpr_alloc" not in plain.attrs


@pytest.mark.parametrize("arch", ARCHS)
def test_kmajor_lds_tiles(arch):
    # A K-contiguous (K-major tile), B N-contiguous (the [k][n] tile)
    p = gemm_problem(
        256, 192, 128, a_strides=(128, 1), b_strides=(1, 192), d_strides=(1, 256)
    )
    s = arch_spec(p, arch, tile_m=128, tile_n=64, tile_k=8, kmajor_lds=True)
    assert is_valid_spec(s, arch=arch)[0]
    ll = _lower_kernel_to_llvm_python(build_gett_f64(s, arch=arch), arch=arch)
    # A: 2 stages x 128 rows x (8 + 2) doubles; B: 2 stages x 8 K rows x 64
    # (swizzled, no padding)
    assert f"[{(2 * 128 * 10 + 2 * 8 * 64) * 8} x i8]" in ll
    plain = arch_spec(p, arch, tile_m=128, tile_n=64, tile_k=8)
    ll0 = _lower_kernel_to_llvm_python(build_gett_f64(plain, arch=arch), arch=arch)
    assert f"[{(2 * 8 * 128 + 2 * 8 * 64) * 8} x i8]" in ll0


@pytest.mark.parametrize("unroll2", [False, True])
def test_ilv_barriers_keep_the_global_loads_in_flight(unroll2):
    s = arch_spec(
        _kk(K=512), "gfx950", tile_k=16, prefetch_depth=2, ilv=8, unroll2=unroll2
    )
    k = build_gett_f64(s, arch="gfx950")
    ll = _lower_kernel_to_llvm_python(k, arch="gfx950")
    # the loop's barriers drain only LDS (the next tile's loads stay in flight);
    # every MFMA is followed by a scheduling barrier that pins the order
    assert ll.count("llvm.amdgcn.sched.barrier") >= ll.count(
        "llvm.amdgcn.mfma.f64.16x16x4f64"
    )
    assert "llvm.amdgcn.s.barrier" in ll


def test_frag_pipe_and_k_inner_build_on_every_arch():
    for arch in ARCHS:
        for kw in (
            dict(frag_pipe=True),
            dict(frag_pipe=True, unroll2=True),
            dict(k_inner=2, mfma_fence=True),
            dict(k_inner=4, frag_pipe=True),
            dict(ilv=4, prefetch_depth=2, kmajor_lds=True, acc_agpr=True, unroll2=True),
        ):
            s = arch_spec(_mn(), arch, tile_k=16, **kw)
            assert is_valid_spec(s, arch=arch)[0], (arch, kw)
            _lower_kernel_to_llvm_python(build_gett_f64(s, arch=arch), arch=arch)


# ------------------------------------------------------------- dispatch


def test_pipe_configs_are_valid():
    for (arch, kind, cls), cfg in PIPE_CONFIGS.items():
        p = gemm_problem(
            8192,
            8192,
            2048,
            a_strides=(1, 8192),
            b_strides=(2048, 1),
            d_strides=(1, 8192),
            complex_=kind == "complex",
        )
        assert is_valid_spec(arch_spec(p, arch, **cfg), arch=arch)[0], (arch, cls)


def test_pipe_classes():
    big = _kk(M=8192, N=8192, K=2048)
    assert gett_f64_pipe_class(big, "gfx950", 256) == "long"
    # 4096^2 at a short K: the medium class -> short configuration
    assert gett_f64_pipe_class(_kk(M=4096, N=4096, K=512), "gfx950", 256) == "short"
    # small grids: one 64x64 tile per CU or fewer -> tiny, more -> small
    assert gett_f64_pipe_class(_kk(M=1024, N=1024, K=512), "gfx950", 256) == "tiny"
    assert gett_f64_pipe_class(_kk(M=1536, N=1536, K=512), "gfx950", 256) == "small"
    # a large grid with a short K -> wide
    assert gett_f64_pipe_class(_kk(M=15360, N=15360, K=768), "gfx950", 256) == "wide"
    # a large grid whose last wave of 256x128 CTAs is ragged keeps the tuned path
    assert gett_f64_pipe_class(_kk(M=6720, N=7680, K=1536), "gfx950", 256) is None
    cx = gemm_problem(
        8192,
        8192,
        2048,
        a_strides=(2048, 1),
        b_strides=(2048, 1),
        d_strides=(1, 8192),
        complex_=True,
    )
    assert gett_f64_pipe_class(cx, "gfx950", 256) == "long"
    assert gett_f64_pipe_class(cx, "gfx90a", 104) == "long"  # 4096 tiles >= 8 per CU
    cx_mid = gemm_problem(
        2048,
        2048,
        2048,
        a_strides=(2048, 1),
        b_strides=(2048, 1),
        d_strides=(1, 2048),
        complex_=True,
    )
    assert gett_f64_pipe_class(cx_mid, "gfx90a", 104) is None
    assert gett_f64_pipe_class(cx_mid, "gfx950", 256) == "long"
    short_k = gemm_problem(
        8192,
        8192,
        512,
        a_strides=(512, 1),
        b_strides=(512, 1),
        d_strides=(1, 8192),
        complex_=True,
    )
    assert gett_f64_pipe_class(short_k, "gfx950", 256) is None
    # an interleaved M index with 24-element runs: the short class
    from rocke.instances.common.gett_f64 import GettF64Problem

    inter = GettF64Problem(
        M=24 * 320,
        N=8192,
        K=2048,
        a_m=((24, 1), (320, 24 * 2048)),
        a_k=((2048, 24),),
        b_n=((8192, 1),),
        b_k=((2048, 8192),),
        d_m=((24 * 320, 1),),
        d_n=((8192, 24 * 320),),
    )
    assert gett_f64_pipe_class(inter, "gfx950", 256) == "short"


def test_pipe_candidate_selection_and_refusals():
    req = GettF64Request(
        problem=_kk(M=8192, N=8192, K=2048), arch="gfx950", num_cus=256
    )
    r = dispatch_gett_f64(req)
    assert r.candidate.name == "gett_f64_pipe"
    assert r.spec == pipe_spec(req.problem, "gfx950", num_cus=256)
    assert r.spec.ilv and r.spec.acc_agpr and r.spec.kmajor_lds
    # runtime sizes: the pipelined loop is compile-time only
    rt = GettF64Request(
        problem=_kk(M=8192, N=8192, K=2048),
        arch="gfx950",
        num_cus=256,
        runtime_sizes=True,
    )
    assert dispatch_gett_f64(rt).candidate.name == "gett_f64_tuned"
    # few CTAs and a long K: the tuned path splits K (shorter rounding chains)
    long_k = _kk(M=256, N=256, K=65536)
    assert tuned_spec(long_k, "gfx950", num_cus=256).split_k > 1
    assert pipe_spec(long_k, "gfx950", num_cus=256) is None
    ragged = GettF64Request(
        problem=_kk(M=6720, N=7680, K=1536), arch="gfx950", num_cus=256
    )
    assert dispatch_gett_f64(ragged).candidate.name == "gett_f64_tuned"
