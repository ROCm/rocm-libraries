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

from kernels.common._attention_bwd_caps import (
    ASM_MFMA_VALU_LEAD_WAIT_STATES,
    BWD_ARCHES,
    bwd_arch_caps,
)
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
    hi = max(64, g.threads)
    # The lowering's minimum is the wave size, capped at the block.
    assert f'"amdgpu-flat-work-group-size"="{min(g.wave_size, hi)},{hi}"' in ir


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


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("k1", [16, 32])
def test_relabel_chain_ir_through_the_transpose_read(k1, flavor):
    """gfx950 chain with dO read by ``ds_read_tr16_b64``: the catalog plan
    (16x16x16 consumer) or the stacked K-permuted plan (16x16x32): the same
    transpose-read count either way, MFMA count as the model, no b128 read."""
    kernel, g0, g1, plan = make_chain_probe(
        "gfx950", "fp16", k_m0=32, k_n0=64, d=64, waves=4, atom_k0=16, atom_k1=k1,
        do_storage="k_outer", do_reader="tr16",
    )  # fmt: skip
    ir, n_mma = _lower_count(kernel, "gfx950", flavor)
    assert n_mma == g0.mma_per_wave + g1.mma_per_wave
    tr = ir.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(")
    assert tr == g1.frags_n * g1.k_steps * (k1 // 16) > 0  # = frags_n * kM0 / 16
    assert not _TR_B128.search(ir) and "alloca" not in ir
    assert plan.needs_k_perm is (k1 == 32)
    with pytest.raises(ValueError, match="K-outer"):
        make_chain_probe(
            "gfx950", "fp16", k_m0=32, k_n0=64, d=64, waves=4, do_storage="k_inner",
            do_reader="tr16",
        )  # fmt: skip


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


# The leading s_nop of every asm cluster: the ISA's VALU-write-to-MFMA-read
# wait states (a VALU write of a source right before the block is invisible
# to the backend's hazard recognizer).
_LEAD = [f"s_nop {ASM_MFMA_VALU_LEAD_WAIT_STATES - 1}"]


@pytest.mark.parametrize("zero", [False, True], ids=["tied", "zero"])
@pytest.mark.parametrize("nop", [0, 3, 15])
@pytest.mark.parametrize("arch, m, n, atom_k", _ASM_SHAPES)
def test_asm_cluster_granularity_and_order(arch, m, n, atom_k, nop, zero):
    """``k_step``: one cluster per K step, each starting with the lead s_nop
    and ending in the tail s_nop. ``gemm``: one cluster over every K step, the
    MFMAs K outer / fragment inner (the intrinsic order), srcC the tied input
    on the first step and the running output after it, one tail s_nop at the
    end and, with a single fragment, one between the K steps. ``nop = 0``
    emits no tail s_nop. A zero accumulator (``acc=None``) reads srcC ``0`` on
    the first K step and has no tied inputs."""
    want = nop
    kw = dict(m=m, n=n, k=4 * atom_k, grid=(4, 1), atom_k=atom_k, asm_nop=nop)
    kw.update(asm_zero=zero)
    k_kernel, g = make_gemm_probe(arch, "bf16", asm_cluster="k_step", **kw)
    g_kernel, _ = make_gemm_probe(arch, "bf16", asm_cluster="gemm", **kw)
    frags = g.frags_m * g.frags_n
    nop_line = [f"s_nop {want}"] if want else []
    mnem = f"v_mfma_f32_16x16x{atom_k}_bf16"

    k_ops = _asm_ops(k_kernel)
    assert len(k_ops) == g.k_steps
    for ks, op in enumerate(k_ops):
        lines = _asm_lines(op)
        assert lines[:1] == _LEAD
        assert lines[1 + frags :] == nop_line
        assert all(ln.startswith(mnem + " ") for ln in lines[1 : 1 + frags])
        assert op.attrs["constraints"].count("v") == g.frags_m + g.frags_n
        first = zero and ks == 0
        assert all(ln.endswith(", 0") is first for ln in lines[1 : 1 + frags])
        cons = op.attrs["constraints"].split(",")
        tied = [] if first else [str(i) for i in range(frags)]
        assert cons == ["=a"] * frags + tied + ["v"] * (g.frags_m + g.frags_n)

    (op,) = _asm_ops(g_kernel)
    lines = _asm_lines(op)
    between = nop_line if frags == 1 else []
    expect = list(_LEAD)
    per_step = g.frags_m + g.frags_n
    n_tied = 0 if zero else frags
    for ks in range(g.k_steps):
        if ks:
            expect += between
        a0 = frags + n_tied + ks * per_step
        for i in range(frags):
            fm, fn = divmod(i, g.frags_n)
            src_c = f"${i}" if ks else ("0" if zero else f"${frags + i}")
            expect.append(f"{mnem} ${i}, ${a0 + fm}, ${a0 + g.frags_m + fn}, {src_c}")
    assert lines == expect + nop_line
    cons = op.attrs["constraints"].split(",")
    assert cons == ["=a"] * frags + [str(i) for i in range(n_tied)] + ["v"] * (
        g.k_steps * per_step
    )
    # the gemm cluster reads exactly the operands of the per-step clusters
    tied_ops = [n_tied] + [frags] * (g.k_steps - 1)
    assert len(op.operands) == n_tied + sum(
        len(o.operands) - t for o, t in zip(k_ops, tied_ops)
    )


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
    # a zero accumulator (acc=None) is a form of the asm route only
    with pytest.raises(ValueError, match="acc=None"):
        g.mma(b, None, LdsOperand(s), LdsOperand(s))
    with pytest.raises(ValueError, match="route='agpr_asm'"):
        g.mma(b, acc, LdsOperand(s), LdsOperand(s), asm_agpr_base=0)
    with pytest.raises(ValueError, match="route='agpr_asm'"):
        g.mma(b, acc, LdsOperand(s), LdsOperand(s), asm_lead_wait=2)


def _lds_gemm(arch, *, m=64, n=64, k=32, grid=(4, 1), atom_k=16, name="g"):
    g = BlockGemm(arch, "bf16", m, n, k, grid, atom_k)
    b = IRBuilder(f"bwd_gemm_{name}_{arch}")
    _, lane, wave = lane_and_wave(b, g.wave_size)
    sa = b.smem_alloc(F16, (m, k), "sa")
    sb = b.smem_alloc(F16, (n, k), "sb")
    return g, b, lane, g.wave_origin(b, wave), LdsOperand(sa), LdsOperand(sb)


@pytest.mark.parametrize("zero", [False, True], ids=["tied", "zero"])
@pytest.mark.parametrize("cluster", ["k_step", "gemm"])
@pytest.mark.parametrize("arch, m, n, atom_k", _ASM_SHAPES)
def test_asm_lead_wait_opens_every_cluster(arch, m, n, atom_k, cluster, zero):
    """``asm_lead_wait = w`` puts ``s_nop w - 1`` in front of every cluster
    and changes nothing else; ``None`` (the default) is the ISA's lead wait
    (2: ``s_nop 1``, the text of the default route); ``0`` emits no lead."""
    texts = {}
    for lead in (0, 1, 2, None):
        g, b, lane, origin, sa, sb = _lds_gemm(
            arch, m=m, n=n, k=2 * atom_k, atom_k=atom_k, name=f"lead{lead}"
        )
        acc = None if zero else g.zero_acc(b)
        g.mma(
            b, acc, sa, sb, lane=lane, origin=origin, route="agpr_asm",
            asm_cluster=cluster, asm_lead_wait=lead,
        )  # fmt: skip
        texts[lead] = [_asm_lines(op) for op in _asm_ops(b.kernel)]
    assert texts[0] and len(texts[0]) == len(texts[1]) == len(texts[2])
    assert texts[None] == texts[2]
    for base, one, two in zip(texts[0], texts[1], texts[2]):
        assert base[0].startswith("v_mfma")
        assert one == ["s_nop 0"] + base and two == _LEAD + base
    g, b, lane, origin, sa, sb = _lds_gemm(arch, m=m, n=n, atom_k=atom_k)
    from kernels.common._attention_bwd_gemm import mma_agpr_asm_steps

    for bad in (-1, 17, True, 1.0):
        with pytest.raises(ValueError, match="lead wait"):
            mma_agpr_asm_steps(g, b, g.zero_acc(b), [([], [])], lead_wait=bad)


@pytest.mark.parametrize("route", ["intrinsic", "agpr_asm"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_preloaded_operands_issue_the_same_mmas(arch, route):
    """``load_operands`` reads A and B per K step in the order ``mma`` reads
    them; ``mma`` on the returned registers issues the same MMAs on the same
    values with no LDS read; ``pre_issue`` runs once, after the first step's
    reads and before the first MMA / cluster."""

    def loads(b):
        return [o for o in b.kernel.body.ops if o.name.startswith(_LOADS)]

    def mmas(b):
        name = "tile.mma" if route == "intrinsic" else "tile.inline_asm"
        return [o for o in b.kernel.body.ops if o.name == name]

    g, b0, lane, origin, sa, sb = _lds_gemm(arch, name="direct")
    g.mma(b0, g.zero_acc(b0), sa, sb, lane=lane, origin=origin, route=route)
    g, b1, lane, origin, sa, sb = _lds_gemm(arch, name="pre")
    a_regs, b_regs = g.load_operands(b1, sa, sb, lane=lane, origin=origin)
    n_loads = len(loads(b1))
    assert n_loads == len(loads(b0)) > 0
    assert [len(x) for x in a_regs] == [g.k_steps] * g.frags_m
    assert [len(x) for x in b_regs] == [g.k_steps] * g.frags_n
    g.mma(b1, g.zero_acc(b1), a_regs, b_regs, route=route)
    assert len(loads(b1)) == n_loads  # nothing read again
    assert len(mmas(b1)) == len(mmas(b0)) > 0
    # the same load ops, in the same order, as the direct call
    assert [o.name for o in loads(b1)] == [o.name for o in loads(b0)]
    # the hook: once, between the first step's reads and the first issue
    # (a k_step cluster or the first intrinsic MMA)
    g, b2, lane, origin, sa, sb = _lds_gemm(arch, name="hook")
    seen = []

    def hook():
        seen.append((len(loads(b2)), len(mmas(b2))))

    g.mma(b2, g.zero_acc(b2), sa, sb, lane=lane, origin=origin, route=route,
          pre_issue=hook)  # fmt: skip
    per_step = len(loads(b0)) // g.k_steps
    assert g.k_steps > 1
    assert len(seen) == 1 and seen[0] == (per_step, 0)
    # gemm granularity: the hook runs after every step's reads, before the
    # one cluster
    if route == "agpr_asm":
        g, b3, lane, origin, sa, sb = _lds_gemm(arch, name="hookg")
        seen.clear()

        def hook3():
            seen.append((len(loads(b3)), len(mmas(b3))))

        g.mma(b3, g.zero_acc(b3), sa, sb, lane=lane, origin=origin, route=route,
              asm_cluster="gemm", pre_issue=hook3)  # fmt: skip
        assert seen == [(len(loads(b0)), 0)]


@pytest.mark.parametrize("cluster", ["k_step", "gemm"])
@pytest.mark.parametrize("zero", [False, True], ids=["tied", "zero"])
def test_asm_cluster_pinned_agprs(cluster, zero):
    """``asm_agpr_base`` pins output ``i`` of every cluster of the call to the
    AGPR tuple ``a[base + 4i : base + 4i + 3]`` (the tied inputs follow their
    outputs); bases outside a0..a255 are refused before any IR."""
    g = BlockGemm("gfx942", "bf16", 32, 64, 64, (1, 4))
    b = IRBuilder("bwd_gemm_pinned")
    lane, origin = b.thread_id_x(), (b.const_i32(0), b.const_i32(0))
    s = b.smem_alloc(F16, (64, 64), "s")
    acc = None if zero else g.zero_acc(b)
    n0 = len(b._region_stack[-1].ops)
    g.mma(
        b, acc, LdsOperand(s), LdsOperand(s), lane=lane, origin=origin,
        route="agpr_asm", asm_cluster=cluster, asm_agpr_base=96,
    )  # fmt: skip
    ops = [o for o in b._region_stack[-1].ops[n0:] if o.name == "tile.inline_asm"]
    assert len(ops) == (1 if cluster == "gemm" else g.k_steps)
    n = g.frags_m * g.frags_n
    want = ["={a[%d:%d]}" % (96 + 4 * i, 99 + 4 * i) for i in range(n)]
    for k, op in enumerate(ops):
        cons = op.attrs["constraints"].split(",")
        assert cons[:n] == want
        tied = [] if zero and k == 0 else [str(i) for i in range(n)]
        assert cons[n : n + len(tied)] == tied
    n1 = len(b._region_stack[-1].ops)
    for bad in (-4, 256 - 4 * n + 1, True, 1.5):
        with pytest.raises(ValueError, match="agpr_base|pinned AGPRs"):
            g.mma(
                b, acc, LdsOperand(s), LdsOperand(s), lane=lane, origin=origin,
                route="agpr_asm", asm_cluster="gemm", asm_agpr_base=bad,
            )  # fmt: skip
    assert len(b._region_stack[-1].ops) == n1


# --- G4 key slicing (mma_k_sliced) ---------------------------------------------------

_LOADS = ("tile.smem_load", "tile.ds_read")


def _sliced_ops(
    arch, *, k_slice=None, buffers=1, first=False, b_src="lds", reader="plain"
):
    """Ops of one G4-shaped block GEMM (kM0 32 x D 64 over kN0 128, grid
    (1, 4)): dS from LDS, K^T from LDS (K-outer, ``reader``) or registers.
    ``k_slice=None``: :meth:`BlockGemm.mma`; else :meth:`mma_k_sliced`."""
    from rocke.core.ir import BF16

    atom = 32 if arch == "gfx950" else 16
    g = BlockGemm(arch, "bf16", 32, 64, 128, (1, 4), atom)
    b = IRBuilder("bwd_gemm_sliced")
    lane = b.thread_id_x()
    origin = (b.const_i32(0), b.const_i32(0))
    sa = b.smem_alloc(BF16, (32, 128), "ds")
    sb = b.smem_alloc(BF16, (128, 64), "k")
    a = LdsOperand(sa)
    bop = LdsOperand(sb, "k_outer", reader=reader)
    start = len(b._region_stack[-1].ops)
    if b_src == "reg":  # inside the traced ops, so sources compare by index
        bop = [
            [b.zero_vec(BF16, g.op.b_frag_len) for _ in range(g.k_steps)]
            for _ in range(g.frags_n)
        ]
    acc = g.zero_acc(b)
    if k_slice is None:
        out = g.mma(b, acc, a, bop, lane=lane, origin=origin)
    else:
        pre = None
        if first:
            pre = g.load_k_slice(b, a, bop, 0, k_slice, lane=lane, origin=origin)
        out = g.mma_k_sliced(
            b, acc, a, bop, k_slice=k_slice, buffers=buffers, lane=lane,
            origin=origin, first=pre,
        )  # fmt: skip
    return g, b._region_stack[-1].ops[start:], out


def _signature(ops, out, g):
    """``(non-MMA op list, {(fm, fn): [(A source, B source) per K step]},
    MMA K step by op id)``: each MMA chain followed back from its result to
    the zero accumulator; a source is the index of its producing op among the
    non-MMA ops (or the value itself for an external operand)."""
    other = [op for op in ops if op.name != "tile.mma"]
    idx = {id(op): i for i, op in enumerate(other)}
    prod = {}
    for op in ops:
        for r in op.results:
            prod[id(r)] = op

    def src(v):
        p = prod.get(id(v))
        return ("op", idx[id(p)]) if p is not None and id(p) in idx else ("ext", id(v))

    chains, step_of = {}, {}
    for fm in range(g.frags_m):
        for fn in range(g.frags_n):
            chain, v = [], out[fm][fn]
            while id(v) in prod and prod[id(v)].name == "tile.mma":
                op = prod[id(v)]
                chain.append((op, src(op.operands[0]), src(op.operands[1])))
                v = op.operands[2]
            chain.reverse()
            for ks, (op, _, _) in enumerate(chain):
                step_of[id(op)] = ks
            chains[(fm, fn)] = [(sa, sb) for _, sa, sb in chain]
    sig = [(op.name, repr(sorted(op.attrs.items()))) for op in other]
    return sig, chains, step_of


def _slice_cases():
    for arch in ("gfx942", "gfx950"):
        atom = 32 if arch == "gfx950" else 16
        # the tr16 reader needs a planner LdsView; the kernel IR tests cover
        # it (gfx950 K^T from LDS under every g4_slicing value)
        readers = ("plain",)
        for k_slice in (16, 32, 64, 128):
            if k_slice % atom:
                continue
            for buffers, first in ((1, False), (2, False), (1, True), (2, True)):
                for b_src in ("lds", "reg"):
                    for reader in readers if b_src == "lds" else ("plain",):
                        yield pytest.param(
                            arch, k_slice, buffers, first, b_src, reader,
                            id=f"{arch}-ks{k_slice}-b{buffers}{'-first' if first else ''}"
                            f"-{b_src}-{reader}",
                        )  # fmt: skip


@pytest.mark.parametrize(
    "arch, k_slice, buffers, first, b_src, reader", list(_slice_cases())
)
def test_mma_k_sliced_issues_the_products_of_mma(
    arch, k_slice, buffers, first, b_src, reader
):
    """Same operand reads (op for op, in the same order: every load reads the
    LDS offset of its global K step) and, per accumulator, the same MMA chain
    (same A / B sources in ascending K order) as :meth:`BlockGemm.mma`."""
    g, ref_ops, ref_out = _sliced_ops(arch, b_src=b_src, reader=reader)
    g, ops, out = _sliced_ops(
        arch, k_slice=k_slice, buffers=buffers, first=first, b_src=b_src, reader=reader
    )
    ref_sig, ref_chains, _ = _signature(ref_ops, ref_out, g)
    sig, chains, _ = _signature(ops, out, g)
    assert sig == ref_sig
    assert chains == ref_chains
    n = sum(op.name == "tile.mma" for op in ops)
    assert n == g.mma_per_wave == sum(op.name == "tile.mma" for op in ref_ops)


@pytest.mark.parametrize(
    "arch, k_slice, buffers, first, b_src, reader", list(_slice_cases())
)
def test_mma_k_sliced_order(arch, k_slice, buffers, first, b_src, reader):
    """``buffers = 2``: every read of slice ``s + 1`` precedes the first MMA
    of slice ``s``; ``buffers = 1``: every read of slice ``s`` follows the
    last MMA of slice ``s - 1``. Reads of one slice precede its MMAs."""
    g, ops, out = _sliced_ops(
        arch, k_slice=k_slice, buffers=buffers, first=first, b_src=b_src, reader=reader
    )
    _, _, step_of = _signature(ops, out, g)
    per = k_slice // g.atom_k
    n_slices = g.k // k_slice
    loads = [i for i, op in enumerate(ops) if op.name.startswith(_LOADS)]
    assert len(loads) % g.k_steps == 0
    per_step = len(loads) // g.k_steps
    load_slice = {i: (j // per_step) // per for j, i in enumerate(loads)}
    mma_slice = {
        i: step_of[id(op)] // per for i, op in enumerate(ops) if op.name == "tile.mma"
    }
    for s in range(n_slices):
        reads = [i for i, sl in load_slice.items() if sl == s]
        mmas = [i for i, sl in mma_slice.items() if sl == s]
        assert max(reads) < min(mmas)  # a slice is read before its MMAs
        if s == 0:
            continue
        prev = [i for i, sl in mma_slice.items() if sl == s - 1]
        if buffers == 2:
            assert max(reads) < min(prev)
        else:
            assert min(reads) > max(prev)


def test_mma_k_sliced_one_step_slices_are_the_mma_order():
    """One atom-K step per slice with one buffer is :meth:`mma` op for op (the
    canonical spelling of ``g4_slicing = "slice"`` at kK4 = atom K)."""
    g, ref_ops, _ = _sliced_ops("gfx942")
    _, ops, _ = _sliced_ops("gfx942", k_slice=16, buffers=1)
    assert [(op.name, repr(sorted(op.attrs.items()))) for op in ops] == [
        (op.name, repr(sorted(op.attrs.items()))) for op in ref_ops
    ]


def test_mma_k_sliced_rejects_bad_slices():
    g = BlockGemm("gfx942", "fp16", 32, 64, 128, (1, 4))
    b = IRBuilder("bwd_gemm_slice_bad")
    s = b.smem_alloc(F16, (32, 128), "s")
    lane, origin = b.thread_id_x(), (b.const_i32(0), b.const_i32(0))
    acc = g.zero_acc(b)
    n0 = len(b._region_stack[-1].ops)
    for bad in (0, 8, 24, 48, 256, True, 16.0):
        with pytest.raises(ValueError, match="K slice"):
            g.mma_k_sliced(
                b, acc, LdsOperand(s), LdsOperand(s), k_slice=bad, buffers=1,
                lane=lane, origin=origin,
            )  # fmt: skip
    with pytest.raises(ValueError, match="buffers"):
        g.mma_k_sliced(
            b, acc, LdsOperand(s), LdsOperand(s), k_slice=32, buffers=3, lane=lane,
            origin=origin,
        )  # fmt: skip
    with pytest.raises(ValueError, match="slice"):
        g.load_k_slice(b, LdsOperand(s), LdsOperand(s), 4, 32, lane=lane, origin=origin)
    with pytest.raises(ValueError, match="first slice"):
        g.mma_k_sliced(
            b, acc, LdsOperand(s), LdsOperand(s), k_slice=32, buffers=1, lane=lane,
            origin=origin, first=[([], [])],
        )  # fmt: skip
    # every refusal comes before any IR
    assert len(b._region_stack[-1].ops) == n0


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_sliced_and_zero_init_probes_lower(arch):
    atom = 32 if arch == "gfx950" else 16
    for kw in (
        dict(k_slice=2 * atom, slice_buffers=2),
        dict(k_slice=2 * atom, slice_buffers=1, slice_first=True),
        dict(asm_nop=7, asm_zero=True),
        dict(asm_nop=7, asm_zero=True, asm_cluster="gemm"),
    ):
        kernel, g = make_gemm_probe(
            arch, "bf16", m=64, n=64, k=4 * atom, grid=(4, 1), atom_k=atom, **kw
        )
        ir = lower(kernel, arch=arch, llvm_flavor="llvm22")
        assert "alloca" not in ir
        n_int = len(_MMA.findall(ir))
        n_asm = len(re.findall(r"v_mfma_f32_16x16x\d+_bf16 \$", ir))
        assert n_int + n_asm == g.mma_per_wave
