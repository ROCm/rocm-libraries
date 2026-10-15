# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hazard evidence for the AGPR-form inline-asm MFMA cluster (the ``asm_g13`` route).

An inline-asm MFMA is opaque to the backend's hazard recognizer, so each cluster
ends in ``s_nop AGPR_ASM_TAIL_NOP[atom K]`` (or the spec's ``asm_tail_nop``) to
cover its result latency (the next cluster reads the same accumulators as srcC;
the epilogue reads them with ``v_accvgpr_read``). The legal floor is the ISA's
wait-state table
(:func:`asm_tail_nop_min`), not a probe: a full-overlap srcC chain needs no wait
state at all, so a block GEMM is bitwise at every count and cannot show a hazard.

The evidence, in three layers:

* the table itself, against the waits the backend inserts after the same MFMA
  as an intrinsic (compile-only);
* a negative control that provably reads an MFMA result too early: one MFMA and
  ``v_accvgpr_read`` of its result in one asm block, nothing in between (the
  disassembly is checked), must give wrong values on the device, while every
  count from the floor up is bitwise the intrinsic; a full-overlap srcC chain
  with nothing in between must be bitwise (the chain needs no tail nop);
* an ISA audit of whole ``asm_g13`` kernels built at the floor, with and
  without ``s_in_agpr`` (G0 / G2 as asm clusters, S / dP read by the softmax
  VALU with ``v_accvgpr_read``): every asm MFMA result is read no earlier than
  the table allows, and no VALU write of an asm MFMA source lands inside the
  ISA's lead window, which each cluster's leading ``s_nop`` covers
  (compile-only); with srcC included, this holds in every stage order and
  under the operand-load hoist (``asm_load_hoist``);
* the zero-accumulator cluster form of G0 / G2 (srcC ``0`` on the first K
  step) is bitwise the intrinsic GEMM at every legal count (device).

The device tests run on gfx942 / gfx950 only.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from kernels.common._attention_bwd_caps import (
    AGPR_ASM_TAIL_NOP,
    ASM_MFMA_VALU_LEAD_WAIT_STATES,
    asm_tail_nop_min,
    check_asm_tail_nop,
    mfma_result_wait_states,
)
from ._attention_bwd_test_utils import device_arch

ARCH = os.environ.get("ROCKE_BWD_ARCH") or os.environ.get("ROCKE_TEST_ARCH")


ARCH = ARCH or device_arch()
_MFMA_ARCHES = ("gfx942", "gfx950")
device = pytest.mark.skipif(
    ARCH not in _MFMA_ARCHES, reason="AGPR-form MFMA probe: gfx942 / gfx950"
)

# Whether a VALU read issued right after the MFMA observes a stale result on
# the device (the negative control fires). Recorded from the device runs; an
# arch that interlocks the read would be False here, and the floor would then
# rest on the ISA table alone.
_NEGATIVE_CONTROL_FIRES = {"gfx942": True, "gfx950": True}


def _atoms(arch):
    return [16, 32] if arch == "gfx950" else [16]


_ARCH_ATOMS = [(a, k) for a in _MFMA_ARCHES for k in _atoms(a)]


# --------------------------------------------------------------------------- table
def test_asm_tail_nop_floor_follows_the_isa_table():
    # 16x16 f16 / bf16 XDL atoms are 4-pass; result read: passes + 3 (gfx942),
    # passes + 4 (gfx950); s_nop n gives n + 1 wait states.
    for dtype in ("fp16", "bf16"):
        assert mfma_result_wait_states("gfx942", 16) == 7
        assert mfma_result_wait_states("gfx950", 16) == 8
        assert mfma_result_wait_states("gfx950", 32) == 8
        assert asm_tail_nop_min("gfx942", dtype, 16) == 6
        assert asm_tail_nop_min("gfx950", dtype, 16) == 7
        assert asm_tail_nop_min("gfx950", dtype, 32) == 7
    for arch, k in _ARCH_ATOMS:
        lo = asm_tail_nop_min(arch, "bf16", k)
        assert AGPR_ASM_TAIL_NOP[k] >= lo  # the shipped count is legal
        for nop in range(lo, 16):
            check_asm_tail_nop(arch, "bf16", k, nop)
        for bad in (lo - 1, 16, -1, True, 7.0):
            with pytest.raises(ValueError, match="not legal"):
                check_asm_tail_nop(arch, "bf16", k, bad)
    with pytest.raises(ValueError, match="gfx950-only"):
        asm_tail_nop_min("gfx942", "fp16", 32)
    for arch in ("gfx1151", "gfx1201"):
        with pytest.raises(ValueError):
            asm_tail_nop_min(arch, "fp16", 16)
    assert ASM_MFMA_VALU_LEAD_WAIT_STATES == 2


# --------------------------------------------------------------------------- ISA helpers
def _compile_isa(kernel, arch):
    from tests._attention_bwd_resources import disassemble, find_objdump

    if find_objdump() is None:
        pytest.skip("llvm-objdump not found")
    try:
        from rocke.helpers.compile import compile_kernel

        art = compile_kernel(kernel, arch=arch, capture_ir_text=False, backend="python")
    except (ImportError, OSError) as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"comgr unavailable: {exc}")
    return disassemble(art.hsaco, arch=arch)


_BRANCH = re.compile(r"^s_(c?branch|setpc|swappc|endpgm)")


def _instrs(text):
    """``(mnemonic, operands, is_label)`` per line of a disassembly."""
    out = []
    for raw in text.splitlines():
        ln = raw.split("//")[0].strip()
        if not ln or ln.startswith("."):
            continue
        if ln.endswith(":"):
            out.append(("", [], True))
            continue
        parts = ln.replace(",", " ").split()
        out.append((parts[0], parts[1:], False))
    return out


def _regs(tok):
    m = re.fullmatch(r"([vas])\[(\d+):(\d+)\]", tok)
    if m:
        return {(m.group(1), i) for i in range(int(m.group(2)), int(m.group(3)) + 1)}
    m = re.fullmatch(r"([vas])(\d+)", tok)
    return {(m.group(1), int(m.group(2)))} if m else set()


def _wait_states(ins):
    mn, ops, _ = ins
    return int(ops[0], 0) + 1 if mn == "s_nop" else 1


def _uses(ops):
    return set().union(set(), *(_regs(o) for o in ops))


def audit_mfma_hazards(text, arch, atom_k, *, srcc=False):
    """ISA audit of the AGPR-destination MFMAs of a disassembly.

    Returns ``(violations, checked)``. For each MFMA writing AGPRs: walking
    forward, the first instruction other than a full-overlap srcC MFMA of the
    same accumulator that touches the result must come at least
    :func:`mfma_result_wait_states` later (a label or branch before that bound
    counts as a violation: the audit does not follow control flow); walking
    back, no VALU write of a source VGPR may lie within
    ``ASM_MFMA_VALU_LEAD_WAIT_STATES`` (``srcc=True``: of the srcC registers
    too, for example an allocator copy into a tied accumulator).
    """
    need = mfma_result_wait_states(arch, atom_k)
    ins = _instrs(text)
    bad, checked = [], 0
    for i, (mn, ops, _) in enumerate(ins):
        if not mn.startswith("v_mfma") or not ops[0].startswith("a"):
            continue
        checked += 1
        dst, srcs = _regs(ops[0]), _uses(ops[1:4] if srcc else ops[1:3])
        w, j = 0, i - 1
        while j >= 0 and w < ASM_MFMA_VALU_LEAD_WAIT_STATES:
            pm, pops, label = ins[j]
            if label:
                bad.append((i, "lead window crosses a label"))
                break
            if pm.startswith("v_") and pops and _regs(pops[0]) & srcs:
                bad.append((i, f"VALU write {pm} {pops[0]} {w} wait states before"))
                break
            w += _wait_states(ins[j])
            j -= 1
        w, k = 0, i + 1
        while k < len(ins) and w < need:
            nm, nops, label = ins[k]
            if label or _BRANCH.match(nm):
                bad.append((i, f"result window crosses control flow at {w}"))
                break
            if nm.startswith("v_mfma") and nops[0] == ops[0] and nops[3] == ops[0]:
                break  # full-overlap srcC chain: the ISA needs no wait state
            if _uses(nops) & dst:
                bad.append((i, f"{nm} touches the result {w} wait states after"))
                break
            w += _wait_states(ins[k])
            k += 1
    return bad, checked


# --------------------------------------------------------------------------- compile-only
@pytest.mark.parametrize("arch,atom_k", _ARCH_ATOMS)
def test_backend_inserts_the_table_waits_after_an_intrinsic_mfma(arch, atom_k):
    """The reference kernel (intrinsic MFMA, then its result stored) gets
    exactly the table's wait states from the backend's hazard recognizer."""
    from tests._attention_bwd_gemm_probe import make_hazard_probe

    kernel, _ = make_hazard_probe(arch, "bf16", atom_k=atom_k, nop=None)
    ins = _instrs(_compile_isa(kernel, arch))
    i = next(n for n, x in enumerate(ins) if x[0].startswith("v_mfma"))
    dst = _regs(ins[i][1][0])
    w, k = 0, i + 1
    while not _uses(ins[k][1]) & dst:
        w += _wait_states(ins[k])
        k += 1
    assert w == mfma_result_wait_states(arch, atom_k)


@pytest.mark.parametrize("consumer", ["valu", "srcc"])
@pytest.mark.parametrize("arch,atom_k", _ARCH_ATOMS)
def test_negative_control_has_nothing_between_the_mfma_and_its_consumer(
    arch, atom_k, consumer
):
    """With ``nop=-1`` the consumer is the very next instruction after the
    producing MFMA (no s_nop, no s_waitcnt inserted by the compiler); with
    ``nop=n`` exactly one ``s_nop n`` sits between them."""
    from tests._attention_bwd_gemm_probe import make_hazard_probe

    want = "v_accvgpr_read_b32" if consumer == "valu" else "v_mfma"
    for nop, between in ((-1, []), (3, [("s_nop", "3")])):
        kernel, _ = make_hazard_probe(
            arch, "fp16", atom_k=atom_k, consumer=consumer, nop=nop
        )
        ins = [x for x in _instrs(_compile_isa(kernel, arch)) if not x[2]]
        i = next(n for n, x in enumerate(ins) if x[0].startswith("v_mfma"))
        assert ins[i][1][0] == "a[0:3]"
        got = [(m, o[0]) for m, o, _ in ins[i + 1 : i + 1 + len(between)]]
        assert got == between
        nxt = ins[i + 1 + len(between)]
        assert nxt[0].startswith(want)
        if consumer == "valu":
            assert nxt[1][1] == "a0"
        else:
            assert nxt[1][0] == "a[0:3]" and nxt[1][3] == "a[0:3]"


# Whole-kernel audit tiles: (head size, tile, reservation) per target. The
# s_in_agpr rows are the S / dP-in-AGPR start points (K / V / K^T in
# registers), where G0 / G2 are asm clusters too.
_AUDIT_TILES = {
    "gfx942": [
        (
            128,
            dict(waves=4, block_m=16, block_n=128, kt_source="lds"),
            (128, 128),
            False,
        ),
        (
            128,
            dict(waves=4, block_m=16, block_n=128, kt_source="reg"),
            (192, 192),
            True,
        ),
        (64, dict(waves=4, block_m=32, block_n=128, kt_source="reg"), (128, 128), True),
    ],
    "gfx950": [
        (
            128,
            dict(waves=4, block_m=32, block_n=128, kt_source="lds"),
            (128, 128),
            False,
        ),
        (
            128,
            dict(waves=4, block_m=32, block_n=128, kt_source="reg"),
            (192, 192),
            True,
        ),
        (
            128,
            dict(waves=4, block_m=16, block_n=192, kt_source="reg"),
            (256, 256),
            True,
        ),
    ],
}
_AUDIT_ROWS = [
    pytest.param(
        arch,
        d,
        tile,
        alloc,
        sia,
        id=f"{arch}-d{d}-{tile['block_m']}x{tile['block_n']}-kt{tile['kt_source']}"
        f"{'-s_in_agpr' if sia else ''}",
    )  # fmt: skip
    for arch, rows in _AUDIT_TILES.items()
    for d, tile, alloc, sia in rows
]


@pytest.mark.parametrize("asm_cluster", ["k_step", "gemm"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("arch,d,tile,alloc,s_in_agpr", _AUDIT_ROWS)
def test_asm_g13_kernel_meets_the_isa_wait_states_at_the_floor(
    arch, d, tile, alloc, s_in_agpr, dtype, asm_cluster
):
    """Whole ``asm_g13`` kernel (with ``s_in_agpr``: G0 / G2 as asm too, S / dP
    read by the softmax VALU with ``v_accvgpr_read``) with every cluster
    ending at the table floor: the audit finds no early result read and no
    VALU write in an asm MFMA's lead window (the backend does not see inside
    the asm for either). Every AGPR-destination MFMA of the kernel is an asm
    one (G0 / G2 / G1 / G3 under s_in_agpr, G1 / G3 otherwise)."""
    from kernels.common._attention_bwd_body import BwdTileGeometry
    from kernels.common.attention_bwd import (
        AttnBwdSpec,
        build_attn_bwd_main,
        validate_attn_bwd_spec,
    )

    kw = dict(head_size=d, dtype=dtype, block_k4=32, warp_grid_g4=(1, 4))
    kw.update(agpr_alloc=alloc, agpr_route="asm_g13", kv_residency="reg")
    kw.update(asm_cluster=asm_cluster, s_in_agpr=s_in_agpr, **tile)
    geom = BwdTileGeometry.from_spec(
        validate_attn_bwd_spec(AttnBwdSpec(**kw), arch), arch
    )
    atom_k = geom.g1.atom_k
    floor = max(
        asm_tail_nop_min(arch, dtype, k)
        for k in {atom_k, geom.g0.atom_k if s_in_agpr else atom_k}
    )
    spec = AttnBwdSpec(asm_tail_nop=floor, **kw)
    text = _compile_isa(build_attn_bwd_main(spec, arch=arch), arch)
    assert f"s_nop {floor}" in text
    bad, checked = audit_mfma_hazards(text, arch, atom_k)
    assert checked > 0
    assert not bad, bad[:8]
    if s_in_agpr:  # S / dP are read from AGPRs by the VALU (v_accvgpr_read)
        assert "v_accvgpr_read_b32" in text


def test_audit_flags_an_early_read_and_a_lead_write():
    """The audit itself fires (gfx942 table: 7 wait states)."""
    mf = "v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[0:3]"
    ok = "\n".join([mf, mf, "s_nop 6", "v_accvgpr_read_b32 v4, a0"])
    assert audit_mfma_hazards(ok, "gfx942", 16) == ([], 2)
    early = "\n".join([mf, "s_nop 5", "v_accvgpr_read_b32 v4, a0"])
    assert len(audit_mfma_hazards(early, "gfx942", 16)[0]) == 1
    lead = "\n".join(["v_mov_b32 v1, 0", mf, "s_nop 15"])
    assert len(audit_mfma_hazards(lead, "gfx942", 16)[0]) == 1
    lead_short = "\n".join(["v_mov_b32 v1, 0", "s_nop 0", mf, "s_nop 15"])
    assert len(audit_mfma_hazards(lead_short, "gfx942", 16)[0]) == 1
    lead_ok = "\n".join(["v_mov_b32 v1, 0", "s_nop 1", mf, "s_nop 15"])
    assert audit_mfma_hazards(lead_ok, "gfx942", 16)[0] == []
    branch = "\n".join([mf, "s_cbranch_scc1 0", "s_nop 15"])
    assert len(audit_mfma_hazards(branch, "gfx942", 16)[0]) == 1
    # srcC: an AGPR write right before the MFMA counts only with srcc=True
    acc = "\n".join(["v_accvgpr_write_b32 a1, v9", mf, "s_nop 15"])
    assert audit_mfma_hazards(acc, "gfx942", 16)[0] == []
    assert len(audit_mfma_hazards(acc, "gfx942", 16, srcc=True)[0]) == 1
    acc_ok = "\n".join(["v_accvgpr_write_b32 a1, v9", "s_nop 1", mf, "s_nop 15"])
    assert audit_mfma_hazards(acc_ok, "gfx942", 16, srcc=True)[0] == []


_HOIST_TILES = {
    # the one-wave 16 x 16 correctness tile and the representative W4 tiles
    "w1": dict(
        waves=1, block_m=16, block_n=16, block_k4=16, warp_grid_g4=(1, 1),
        atom_g02="16x16x16", atom_g13="16x16x16", atom_g4="16x16x16",
        kv_residency="lds", kt_source="lds", transpose_source="lds_plain",
        pt_route="relabel",
    ),
    "w4": dict(waves=4, block_m=32, block_n=128, kt_source="lds", block_k4=32,
               warp_grid_g4=(1, 4), kv_residency="reg", atom_g13="16x16x16"),
}  # fmt: skip


@pytest.mark.parametrize("hoist", [False, True])
@pytest.mark.parametrize("order", ["g1g3g4", "g1g4g3", "g3g4g1"])
@pytest.mark.parametrize("asm_cluster", ["k_step", "gemm"])
@pytest.mark.parametrize("tile", list(_HOIST_TILES))
@pytest.mark.parametrize("arch", _MFMA_ARCHES)
def test_asm_g13_kernel_opens_every_cluster_with_the_lead_wait(
    arch, tile, asm_cluster, order, hoist
):
    """A VALU write of an asm MFMA source (a fragment packed, or a register
    the allocator copies, right before a cluster; ``asm_load_hoist`` keeps
    operands live across clusters and G4) needs the ISA lead wait states, which
    the backend cannot insert for an MFMA inside the asm: every cluster opens
    with them, and the audit (srcC included) finds no VALU write in any asm
    MFMA's lead window and no early result read, in every stage order."""
    from kernels.common.attention_bwd import (
        AttnBwdSpec,
        build_attn_bwd_main,
        validate_attn_bwd_spec,
    )

    kw = dict(head_size=64, dtype="fp16", agpr_alloc=(128, 128), agpr_route="asm_g13")
    kw.update(asm_cluster=asm_cluster, stage_order=order, asm_load_hoist=hoist)
    spec = validate_attn_bwd_spec(AttnBwdSpec(**kw, **_HOIST_TILES[tile]), arch)
    text = _compile_isa(build_attn_bwd_main(spec, arch=arch), arch)
    lead = f"s_nop {ASM_MFMA_VALU_LEAD_WAIT_STATES - 1}"
    ins = [x for x in _instrs(text) if not x[2]]
    firsts = [
        i
        for i, (mn, ops, _) in enumerate(ins)
        if mn.startswith("v_mfma") and ops[0].startswith("a")
        and not (ins[i - 1][0].startswith("v_mfma") and ins[i - 1][1][0].startswith("a"))
    ]  # fmt: skip
    assert firsts
    for i in firsts:  # the cluster's own opening s_nop sits right before it
        assert (ins[i - 1][0], ins[i - 1][1][0]) == tuple(lead.split()), ins[i - 1]
    bad, checked = audit_mfma_hazards(text, arch, 16, srcc=True)
    assert checked > 0
    assert not bad, bad[:8]


# --------------------------------------------------------------------------- device
_SEEDS = (0, 1, 2)


def _hazard(dtype, atom_k, consumer, nop, seed):
    from tests._attention_bwd_gemm_probe import run_hazard_probe

    return run_hazard_probe(
        ARCH, dtype, atom_k=atom_k, consumer=consumer, nop=nop, seed=seed
    )


@device
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("atom_k", _atoms(ARCH))
def test_negative_control_fires_and_the_isa_floor_is_bitwise(dtype, atom_k):
    floor = asm_tail_nop_min(ARCH, dtype, atom_k)
    wrong = {}
    for seed in _SEEDS:
        ref = _hazard(dtype, atom_k, "valu", None, seed)
        safe = _hazard(dtype, atom_k, "valu", "safe", seed)
        assert np.array_equal(safe, ref), "the asm probe itself differs"
        for nop in range(-1, 16):
            got = _hazard(dtype, atom_k, "valu", nop, seed)
            if not np.array_equal(got, ref):
                wrong.setdefault(nop, 0)
                wrong[nop] += int(np.sum(got != ref))
            if nop >= floor:
                assert np.array_equal(got, ref), f"s_nop {nop} (>= floor) differs"
    print(
        f"[agpr-hazard-probe] {ARCH} {dtype} atom_k={atom_k} floor={floor} "
        f"wrong_elements_by_nop={dict(sorted(wrong.items()))} "
        f"first_bitwise={max(wrong) + 1 if wrong else -1}"
    )
    assert (-1 in wrong) == _NEGATIVE_CONTROL_FIRES[ARCH], wrong


@device
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("atom_k", _atoms(ARCH))
def test_full_overlap_srcc_chain_needs_no_nop(dtype, atom_k):
    for seed in _SEEDS:
        ref = _hazard(dtype, atom_k, "srcc", None, seed)
        for nop in (-1, 0, "safe"):
            got = _hazard(dtype, atom_k, "srcc", nop, seed)
            assert np.array_equal(got, ref), f"srcC chain at s_nop {nop}"


# G1-shaped blocks (kN0 x D over K = 4 atom steps), one per wave-grid case. The
# 1 x 1 and 1 x 2 blocks are the gemm cluster's tight cases: with one fragment
# the dependent MFMAs of consecutive K steps are separated only by the s_nop,
# with two by a single independent MFMA.
_SHAPES = [
    dict(m=64, n=16, grid=(4, 1)),  # four waves, 1 x 1 fragment each
    dict(m=64, n=32, grid=(4, 1)),  # 1 x 2 fragments each
    dict(m=64, n=64, grid=(4, 1)),  # 1 x 4 fragments each
    dict(m=128, n=128, grid=(4, 1)),  # 2 x 8 fragments each (d128 tiles)
]
_CLUSTERS = ("k_step", "gemm")


def _run(dtype, shape, atom_k, asm_nop, asm_cluster="k_step", asm_zero=False):
    from tests._attention_bwd_gemm_probe import run_gemm_probe

    return run_gemm_probe(
        ARCH, dtype, k=4 * atom_k, atom_k=atom_k, a_storage="k_inner",
        b_storage="k_inner", asm_nop=asm_nop, asm_cluster=asm_cluster,
        asm_zero=asm_zero, **shape,
    )  # fmt: skip


@device
@pytest.mark.parametrize("asm_cluster", _CLUSTERS)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("atom_k", _atoms(ARCH))
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s['m']}x{s['n']}")
def test_asm_cluster_is_bitwise_the_intrinsic_gemm(dtype, atom_k, shape, asm_cluster):
    """Every legal count (the floor to 15) gives the intrinsic GEMM bitwise.
    Counts below the floor are illegal by the ISA table; this block GEMM
    cannot show it (the srcC chain needs no wait and the store is far from the
    last cluster), so they are neither run nor recorded as evidence."""
    ref, exact = _run(dtype, shape, atom_k, None)
    assert np.allclose(ref, exact, rtol=1e-2, atol=1e-2)  # the intrinsic GEMM itself
    for nop in range(asm_tail_nop_min(ARCH, dtype, atom_k), 16):
        got, _ = _run(dtype, shape, atom_k, nop, asm_cluster)
        assert np.array_equal(
            got, ref
        ), f"{asm_cluster} s_nop {nop}: differs from the intrinsics"


@device
@pytest.mark.parametrize("asm_cluster", _CLUSTERS)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("atom_k", _atoms(ARCH))
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s['m']}x{s['n']}")
def test_zero_init_asm_cluster_is_bitwise_the_intrinsic_gemm(
    dtype, atom_k, shape, asm_cluster
):
    """The G0 / G2 form of the asm route (``s_in_agpr``): no accumulator
    input, the first K step reads srcC ``0`` (no AGPR is zeroed first). Every
    legal count gives the intrinsic GEMM from a zero accumulator bitwise, at
    both granularities."""
    ref, exact = _run(dtype, shape, atom_k, None)
    assert np.allclose(ref, exact, rtol=1e-2, atol=1e-2)
    for nop in range(asm_tail_nop_min(ARCH, dtype, atom_k), 16):
        got, _ = _run(dtype, shape, atom_k, nop, asm_cluster, asm_zero=True)
        assert np.array_equal(
            got, ref
        ), f"zero-init {asm_cluster} s_nop {nop}: differs from the intrinsics"
