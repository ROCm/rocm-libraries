# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the backward multi-wave block GEMM.

Geometry validation, the per-wave MMA count model (the stage-table formulas of
the tile step), and lowered IR per warp grid, atom and arch at every LLVM
flavor: the MMA call count equals the model, no scratch (static fragment
indexing), no 128-bit transpose read. The relabel chain probe is lowered too.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir import F16, IRBuilder
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common._attention_bwd_caps import BWD_ARCHES, bwd_arch_caps
from kernels.common._attention_bwd_gemm import (
    BlockGemm,
    LdsOperand,
    check_relabel_grids,
    lane_and_wave,
)

from ._attention_bwd_gemm_probe import make_chain_probe, make_gemm_probe

FLAVORS = ("llvm20", "llvm22", "llvm23")
GRIDS = ((1, 4), (4, 1), (2, 2), (1, 1))
_MMA = re.compile(r"call [^\n]*@llvm\.amdgcn\.(mfma|wmma)[.\w]*\(")
_TR_B128 = re.compile(r"tr16[._]b128|tr_b128|load\.tr\.b128", re.IGNORECASE)


def _atom_ks(arch):
    return (16, 32) if arch == "gfx950" else (16,)


def _cases():
    for arch in BWD_ARCHES:
        for grid in GRIDS:
            for ak in _atom_ks(arch):
                yield pytest.param(
                    arch, grid, ak, id=f"{arch}-{grid[0]}x{grid[1]}-k{ak}"
                )


# --- geometry -------------------------------------------------------------------


def test_geometry_and_wave_ranges():
    g = BlockGemm("gfx942", "fp16", 32, 128, 64, (1, 4))
    assert (g.waves, g.wave_size, g.threads) == (4, 64, 256)
    assert (g.wave_m, g.wave_n, g.frags_m, g.frags_n, g.k_steps) == (32, 32, 2, 2, 4)
    assert g.mma_per_wave == 16
    assert [g.wave_cols(w) for w in range(4)] == [
        (0, 32),
        (32, 64),
        (64, 96),
        (96, 128),
    ]
    assert all(g.wave_rows(w) == (0, 32) for w in range(4))
    g2 = BlockGemm("gfx950", "bf16", 64, 64, 64, (2, 2), 32)
    assert [g2.wave_rows(w) for w in range(4)] == [(0, 32), (0, 32), (32, 64), (32, 64)]
    assert [g2.wave_cols(w) for w in range(4)] == [(0, 32), (32, 64)] * 2
    assert g2.k_steps == 2


@pytest.mark.parametrize(
    "k_m0, k_n0, d, w, k0a, k1a, k4a, g4",
    [
        (32, 128, 32, 4, 16, 16, 16, (2, 2)),
        (32, 128, 64, 4, 16, 16, 16, (1, 4)),
        (16, 128, 128, 4, 16, 16, 16, (1, 4)),
        (16, 192, 128, 4, 32, 16, 32, (1, 4)),
        (32, 256, 64, 4, 32, 32, 32, (1, 4)),
        (16, 16, 64, 1, 16, 16, 16, (1, 1)),
    ],
)
def test_mma_counts_match_the_stage_table_model(k_m0, k_n0, d, w, k0a, k1a, k4a, g4):
    arch = "gfx950" if 32 in (k0a, k1a, k4a) else "gfx942"
    g0 = BlockGemm(arch, "fp16", k_m0, k_n0, d, (1, w), k0a)  # S = Q K^T
    g1 = BlockGemm(arch, "fp16", k_n0, d, k_m0, (w, 1), k1a)  # dV = P^T dO
    g4_ = BlockGemm(arch, "fp16", k_m0, d, k_n0, g4, k4a)  # dQ = dS K
    assert g0.mma_per_wave == (k_m0 // 16) * (k_n0 // (16 * w)) * (d // k0a)
    assert g1.mma_per_wave == (k_n0 // (16 * w)) * (d // 16) * (k_m0 // k1a)
    r, c = g4
    assert g4_.mma_per_wave == (k_m0 // (16 * r)) * (d // (16 * c)) * (k_n0 // k4a)
    check_relabel_grids(g0, g1)


@pytest.mark.parametrize(
    "arch, args, match",
    [
        ("gfx942", (32, 128, 64, (4, 2)), "warp grid"),
        ("gfx942", (32, 128, 64, (1, 3)), "not legal"),
        ("gfx1151", (32, 128, 64, (1, 8)), "not legal"),
        ("gfx942", (24, 128, 64, (1, 4)), "does not split"),
        ("gfx942", (32, 96, 64, (1, 4)), "does not split"),
        ("gfx942", (32, 128, 40, (1, 4)), "not a multiple"),
    ],
)
def test_illegal_geometry_raises(arch, args, match):
    m, n, k, grid = args
    with pytest.raises(ValueError, match=match):
        BlockGemm(arch, "fp16", m, n, k, grid)


@pytest.mark.parametrize("arch", ["gfx942", "gfx1151", "gfx1201"])
def test_wide_k_atom_only_on_gfx950(arch):
    with pytest.raises(ValueError, match="16x16x32"):
        BlockGemm(arch, "fp16", 32, 64, 64, (1, 4), 32)
    BlockGemm("gfx950", "fp16", 32, 64, 64, (1, 4), 32)


def test_relabel_grid_conditions():
    g0 = BlockGemm("gfx942", "fp16", 32, 128, 64, (1, 4))
    with pytest.raises(ValueError, match="producer columns"):
        check_relabel_grids(g0, BlockGemm("gfx942", "fp16", 128, 64, 32, (2, 2)))
    with pytest.raises(ValueError, match="transpose product"):
        check_relabel_grids(g0, BlockGemm("gfx942", "fp16", 128, 64, 64, (4, 1)))
    p22 = BlockGemm("gfx942", "fp16", 32, 128, 64, (2, 2))
    with pytest.raises(ValueError):
        check_relabel_grids(p22, BlockGemm("gfx942", "fp16", 128, 64, 32, (2, 2)))
    with pytest.raises(ValueError, match="arch or dtype"):
        check_relabel_grids(g0, BlockGemm("gfx942", "bf16", 128, 64, 32, (4, 1)))
    with pytest.raises(ValueError, match="different wave counts"):
        check_relabel_grids(
            BlockGemm("gfx942", "fp16", 32, 128, 64, (1, 2)),
            BlockGemm("gfx942", "fp16", 128, 64, 32, (4, 1)),
        )
    plan = check_relabel_grids(
        BlockGemm("gfx950", "fp16", 32, 128, 64, (1, 4), 32),
        BlockGemm("gfx950", "fp16", 128, 64, 32, (4, 1), 32),
    )
    assert plan.needs_k_perm


def test_operand_shape_is_checked():
    g = BlockGemm("gfx942", "fp16", 32, 32, 32, (1, 1))
    b = IRBuilder("bwd_gemm_shape")
    acc = g.zero_acc(b)
    with pytest.raises(ValueError, match="register operand A"):
        g.mma(b, acc, [[None]], [[None, None]] * 2)
    s = b.smem_alloc(F16, (32, 32), "s")
    with pytest.raises(ValueError, match="needs lane and origin"):
        g.mma(b, acc, LdsOperand(s), LdsOperand(s))
    with pytest.raises(ValueError, match="accumulator shape"):
        g.mma(b, acc[:1], LdsOperand(s), LdsOperand(s))


# --- lowered IR per warp grid / atom / arch / flavor --------------------------------


def _lower_count(kernel, arch, flavor):
    ir = lower(kernel, arch=arch, llvm_flavor=flavor)
    assert "alloca" not in ir, "dynamic fragment indexing spilled to scratch"
    assert not _TR_B128.search(ir)
    return ir, len(_MMA.findall(ir))


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch, grid, atom_k", list(_cases()))
def test_gemm_ir(arch, grid, atom_k, flavor):
    m, n = 16 * grid[0] * 2, 16 * grid[1] * 2
    kernel, g = make_gemm_probe(
        arch, "fp16", m=m, n=n, k=64, grid=grid, atom_k=atom_k, a_src="reg"
    )
    ir, n_mma = _lower_count(kernel, arch, flavor)
    assert n_mma == g.mma_per_wave == 2 * 2 * (64 // atom_k)
    path = bwd_arch_caps(arch).matrix_path
    tag = {
        ("mfma", 16): "mfma.f32.16x16x16f16",
        ("mfma", 32): "mfma.f32.16x16x32.f16",
        ("wmma", 16): "wmma.f32.16x16x16.f16",
    }[(path, atom_k)]
    assert ir.count(f"@llvm.amdgcn.{tag}") >= n_mma
    assert f'"amdgpu-flat-work-group-size"="64,{max(64, g.threads)}"' in ir


@pytest.mark.parametrize("b_storage", ["k_inner", "k_outer"])
@pytest.mark.parametrize("srcs", [("lds", "lds"), ("reg", "reg"), ("lds", "reg")])
@pytest.mark.parametrize("arch", BWD_ARCHES)
def test_gemm_ir_operand_sources(arch, srcs, b_storage):
    kernel, g = make_gemm_probe(
        arch, "bf16", m=32, n=128, k=32, grid=(1, 4), a_src=srcs[0], b_src=srcs[1],
        b_storage=b_storage,
    )  # fmt: skip
    _, n_mma = _lower_count(kernel, arch, "llvm22")
    assert n_mma == g.mma_per_wave


def _chain_cases():
    for arch in BWD_ARCHES:
        ks = (16, 32) if arch == "gfx950" else (16,)
        for k1 in ks:
            for st in ("k_outer", "k_inner"):
                yield pytest.param(arch, k1, st, id=f"{arch}-k{k1}-{st}")


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch, k1, do_storage", list(_chain_cases()))
def test_relabel_chain_ir(arch, k1, do_storage, flavor):
    kernel, g0, g1, plan = make_chain_probe(
        arch, "fp16", k_m0=32, k_n0=64, d=64, waves=4, atom_k0=k1, atom_k1=k1,
        do_storage=do_storage,
    )  # fmt: skip
    ir, n_mma = _lower_count(kernel, arch, flavor)
    assert n_mma == g0.mma_per_wave + g1.mma_per_wave
    n_x = len(re.findall(r"call i32 @llvm\.amdgcn\.permlanex16\.i32\(", ir))
    frags = g1.frags_m * g1.k_steps
    assert n_x == (4 * frags if arch == "gfx1151" else 0)
    assert plan.needs_k_perm is (k1 == 32)


def test_lane_and_wave_reads_first_lane():
    b = IRBuilder("bwd_gemm_lane")
    lane_and_wave(b, 64)
    names = [op.name for op in b.kernel.body.ops]
    assert any("readfirstlane" in n for n in names)


# --- single-fragment accumulation -------------------------------------------------


@pytest.mark.parametrize("arch", ["gfx942", "gfx1151"])
def test_mma_frag_issues_the_fragment_chain_of_mma(arch):
    """``mma_frag`` emits, for one fragment, the same MMA chain (operands and
    K order) as ``mma`` does for that fragment, and rejects bad indices."""
    dt = "fp16"
    g = BlockGemm(arch, dt, 32, 64, 64, (2, 1))
    b = IRBuilder("mma_frag_probe")
    n = g.op.c_frag_len
    a = [[b.zero_vec_f32(4) for _ in range(g.k_steps)] for _ in range(g.frags_m)]
    bb = [[b.zero_vec_f32(4) for _ in range(g.k_steps)] for _ in range(g.frags_n)]
    acc = g.zero_acc(b)
    full = g.mma(b, acc, a, bb)
    ops = [op for op in b._region_stack[-1].ops if op.name == "tile.mma"]
    for fm in range(g.frags_m):
        for fn in range(g.frags_n):
            start = len(b._region_stack[-1].ops)
            out = g.mma_frag(b, acc[fm][fn], a, bb, fm, fn)
            mine = [o for o in b._region_stack[-1].ops[start:] if o.name == "tile.mma"]
            assert len(mine) == g.k_steps
            # mma issues K steps outermost, then [fm][fn]
            per_ks = g.frags_m * g.frags_n
            chain = [ops[ks * per_ks + fm * g.frags_n + fn] for ks in range(g.k_steps)]
            assert chain[-1].results[0] is full[fm][fn]
            for ks, (o, r) in enumerate(zip(mine, chain)):
                assert o.operands[0] is a[fm][ks] is r.operands[0]
                assert o.operands[1] is bb[fn][ks] is r.operands[1]
            assert out is mine[-1].results[0]
    with pytest.raises(ValueError):
        g.mma_frag(b, b.zero_vec_f32(n), a, bb, g.frags_m, 0)
    with pytest.raises(ValueError):
        g.mma_frag(b, b.zero_vec_f32(n), LdsOperand(None), bb, 0, 0)


# --- AGPR-form inline-asm clusters ------------------------------------------------


def _asm_ops(kernel):
    return [op for op in kernel.body.ops if op.name == "tile.inline_asm"]


def _asm_lines(op):
    return [ln.strip() for ln in op.attrs["template"].split("\n")]


_ASM_SHAPES = [
    pytest.param("gfx942", 64, 16, 16, id="gfx942-1x1-k16"),
    pytest.param("gfx942", 64, 32, 16, id="gfx942-1x2-k16"),
    pytest.param("gfx942", 128, 128, 16, id="gfx942-2x8-k16"),
    pytest.param("gfx950", 64, 16, 32, id="gfx950-1x1-k32"),
    pytest.param("gfx950", 64, 64, 32, id="gfx950-1x4-k32"),
]


@pytest.mark.parametrize("nop", [0, 3, 15])
@pytest.mark.parametrize("arch, m, n, atom_k", _ASM_SHAPES)
def test_asm_cluster_granularity_and_order(arch, m, n, atom_k, nop):
    """``k_step``: one cluster per K step, each ending in the s_nop. ``gemm``:
    one cluster over every K step, the MFMAs K outer / fragment inner (the
    intrinsic order), srcC the tied input on the first step and the running
    output after it, one s_nop at the end and, with a single fragment, one
    between the K steps. ``nop = 0`` emits no s_nop."""
    want = nop
    kw = dict(m=m, n=n, k=4 * atom_k, grid=(4, 1), atom_k=atom_k, asm_nop=nop)
    k_kernel, g = make_gemm_probe(arch, "bf16", asm_cluster="k_step", **kw)
    g_kernel, _ = make_gemm_probe(arch, "bf16", asm_cluster="gemm", **kw)
    frags = g.frags_m * g.frags_n
    nop_line = [f"s_nop {want}"] if want else []
    mnem = f"v_mfma_f32_16x16x{atom_k}_bf16"

    k_ops = _asm_ops(k_kernel)
    assert len(k_ops) == g.k_steps
    for op in k_ops:
        lines = _asm_lines(op)
        assert lines[frags:] == nop_line
        assert all(ln.startswith(mnem + " ") for ln in lines[:frags])
        assert op.attrs["constraints"].count("v") == g.frags_m + g.frags_n

    (op,) = _asm_ops(g_kernel)
    lines = _asm_lines(op)
    between = nop_line if frags == 1 else []
    expect = []
    per_step = g.frags_m + g.frags_n
    for ks in range(g.k_steps):
        if ks:
            expect += between
        a0 = 2 * frags + ks * per_step
        for i in range(frags):
            fm, fn = divmod(i, g.frags_n)
            src_c = frags + i if ks == 0 else i
            expect.append(f"{mnem} ${i}, ${a0 + fm}, ${a0 + g.frags_m + fn}, ${src_c}")
    assert lines == expect + nop_line
    cons = op.attrs["constraints"].split(",")
    assert cons == ["=a"] * frags + [str(i) for i in range(frags)] + ["v"] * (
        g.k_steps * per_step
    )
    # the gemm cluster reads exactly the operands of the per-step clusters
    assert len(op.operands) == frags + sum(len(o.operands) - frags for o in k_ops)


@pytest.mark.parametrize("cluster", ["k_step", "gemm"])
@pytest.mark.parametrize("arch, m, n, atom_k", _ASM_SHAPES)
def test_asm_cluster_ir_lowers(arch, m, n, atom_k, cluster):
    kernel, g = make_gemm_probe(
        arch, "fp16", m=m, n=n, k=4 * atom_k, grid=(4, 1), atom_k=atom_k,
        asm_nop=2, asm_cluster=cluster,
    )  # fmt: skip
    ir = lower(kernel, arch=arch, llvm_flavor="llvm22")
    assert "alloca" not in ir
    assert not _MMA.findall(ir)
    assert len(re.findall(r"v_mfma_f32_16x16x\d+_f16 \$", ir)) == g.mma_per_wave


def test_asm_cluster_knobs_need_the_asm_route():
    g = BlockGemm("gfx942", "fp16", 32, 32, 32, (1, 1))
    b = IRBuilder("bwd_gemm_asm_knobs")
    acc = g.zero_acc(b)
    s = b.smem_alloc(F16, (32, 32), "s")
    with pytest.raises(ValueError, match="route='agpr_asm'"):
        g.mma(b, acc, LdsOperand(s), LdsOperand(s), asm_tail_nop=3)
    with pytest.raises(ValueError, match="route='agpr_asm'"):
        g.mma(b, acc, LdsOperand(s), LdsOperand(s), asm_cluster="gemm")
    with pytest.raises(ValueError, match="granularity"):
        g.mma(b, acc, LdsOperand(s), LdsOperand(s), route="agpr_asm", asm_cluster="x")
    with pytest.raises(ValueError, match="outside"):
        make_gemm_probe("gfx942", "fp16", m=64, n=16, k=32, grid=(4, 1), asm_nop=16)
    from kernels.common._attention_bwd_gemm import mma_agpr_asm_steps

    with pytest.raises(ValueError, match="at least one K step"):
        mma_agpr_asm_steps(g, b, acc, [])
