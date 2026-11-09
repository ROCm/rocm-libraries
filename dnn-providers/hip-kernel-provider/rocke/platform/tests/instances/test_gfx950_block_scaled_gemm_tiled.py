# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU checks for the tiled gfx950 FP6 GEMM builder and its knob tuner."""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.examples.gfx950.gemm.mxfp6_tiled import tune
from rocke.instances.gfx950.block_scaled_gemm_tiled import (
    _PERM_IDENTITY,
    _PERM_ROT16_XOR,
    _STEP_SCALE_BYTE,
    ABLATIONS,
    CC_EXTRAS,
    CC_PASSES,
    DMA_PLACES,
    DMA_PROBES,
    EPILOGUES,
    FRAG_PIPELINES,
    STEP_VMEM_ORDERS,
    SYNC_PLACES,
    TiledBlockScaledGemmSpec,
    _l2r_source,
    build_tiled_block_scaled_gemm,
    is_valid_spec,
    tiled_block_scaled_gemm_grid,
    tiled_block_scaled_gemm_signature,
)

TARGET = TiledBlockScaledGemmSpec(M=4096, N=4096, K=65536)
SINGLE_MOVES = [(k, v) for k, values in tune.KNOBS.items() for v in values[1:]]


def _lower(spec):
    kernel = build_tiled_block_scaled_gemm(spec, arch="gfx950")
    assert not verify(kernel)
    return lower_kernel_to_llvm(kernel, arch="gfx950", llvm_flavor="llvm23")


@pytest.mark.parametrize("knob,value", [("defaults", None), *SINGLE_MOVES])
def test_every_knob_value_builds_and_lowers(knob, value):
    spec = TARGET if knob == "defaults" else replace(TARGET, **{knob: value})
    assert is_valid_spec(spec) == (True, "ok")
    llvm = _lower(spec)
    mfmas = llvm.count("call <16 x float> @llvm.amdgcn.mfma.scale") + llvm.count(
        "v_mfma_scale_f32_32x32x64_f8f6f4 $"
    )
    stages_emitted = mfmas // 32
    dma = llvm.count("call void @llvm.amdgcn.raw.ptr.buffer.load.lds(")
    S, KT = spec.stages, spec.K // 128
    if spec.frag_pipeline == "step":
        # Non-final stages each issue one stage of DMA; the final one none.
        loops, tail = divmod(KT - 1, S)
        nonfinal = (S if loops else 0) + tail
        assert stages_emitted == nonfinal + 1
        assert dma == 12 * (S + nonfinal)
    elif spec.a_ring in ("line", "line2r"):
        # line2r outside the step pipeline is the three-line ring.
        loops, tail = divmod(KT, spec.unroll)
        phases = (list(range(spec.unroll)) if loops else []) + list(range(tail))
        assert stages_emitted == len(phases)
        # B's 6 DMA loads per stage, a line's 8 in every phase but 1, and the
        # prologue's two lines and B stage.
        assert dma == 2 * 8 + 6 + sum(6 + (8 if u != 1 else 0) for u in phases)
    else:
        loops, tail = divmod(KT, S)
        # k_peel emits the last loop iteration once more, after the loop.
        bodies = (2 if loops > 1 else 1) if spec.k_peel else 1
        assert stages_emitted == (S * bodies if loops else 0) + tail
        # 12 DMA loads per stage (6 per operand) plus the prologue stages.
        assert dma == 12 * (stages_emitted + S - 1)
    assert ("amdgpu-agpr-alloc" in llvm) == (spec.agpr_alloc != "auto")


def test_contract():
    assert tiled_block_scaled_gemm_grid(TARGET) == (256, 1, 1)
    sig = tiled_block_scaled_gemm_signature(TARGET)
    assert [p["name"] for p in sig] == ["A", "B", "A_scale", "B_scale", "C", "M", "N", "K"]
    assert TARGET.lds_bytes == 2 * 512 * 96
    assert TARGET.kernel_name() != replace(TARGET, stages=3).kernel_name()


@pytest.mark.parametrize(
    "changes,needle",
    [
        ({"dtype_a": "fp8e4m3"}, "FP6"),
        ({"K": 65536 + 64}, "multiple of 128"),
        ({"M": 4096 + 128}, "multiples of block_m"),
        ({"stages": 4, "block_m": 256}, "LDS"),
        ({"xcd_remap": "xcd_block", "M": 2048}, "xcd_block"),
        ({"agpr_alloc": "512,512"}, "agpr_alloc"),
        ({"sched": "fast"}, "sched"),
        ({"ablate": "no_mfma+fast"}, "ablate"),
        ({"dma_probe": "far"}, "dma_probe"),
        ({"b_layout": "col"}, "layout"),
        ({"b_scale_layout": "col"}, "scale_layout"),
        ({"mfma_group": 3}, "mfma_group"),
        ({"mfma_group": 3, "mfma_form": "intrinsic"}, "mfma_group"),
        ({"backend": "direct", "frag_pipeline": "step"}, "inline asm"),
        ({"a_scale_fetch": "row"}, "a_scale_fetch"),
        ({"a_scale_fetch": "iter", "a_scale_layout": "kblock"}, "a_scale_fetch"),
        ({"a_scale_fetch": "iter", "frag_pipeline": "step"}, "a_scale_fetch"),
        ({"a_scale_fetch": "iter", "stages": 3, "K": 256}, "a_scale_fetch"),
        ({"a_ring": "lines"}, "a_ring"),
        ({"a_ring": "line", "stages": 3}, "a_ring"),
        ({"a_ring": "line", "a_layout": "kblock"}, "a_ring"),
        ({"a_ring": "line", "frag_pipeline": "step", "stages": 3}, "a_ring"),
        ({"a_ring": "line", "dma_probe": "l2"}, "a_ring"),
        ({"a_ring": "line2r", "stages": 3}, "a_ring"),
        ({"a_ring": "line2r", "frag_pipeline": "step", "backend": "direct", "mfma_form": "intrinsic",
          "agpr_alloc": "auto", "a_layout": "kblock"}, "line2r"),
        ({"a_ring": "line2r", "frag_pipeline": "step", "backend": "direct", "mfma_form": "intrinsic",
          "agpr_alloc": "auto", "waves_n": 4, "waves_per_eu": 2}, "line2r"),
        ({"fill_split": "line2"}, "fill_split"),
        ({"step_vmem_order": "b_first"}, "step_vmem_order"),
        ({"scale_opsel": True, "mfma_form": "intrinsic", "agpr_alloc": "auto"}, "scale_opsel"),
        ({"scale_opsel": True, "frag_pipeline": "step"}, "scale_opsel"),
    ],
)
def test_rejections(changes, needle):
    ok, why = is_valid_spec(replace(TARGET, **changes))
    assert not ok and needle in why
    with pytest.raises(ValueError):
        build_tiled_block_scaled_gemm(replace(TARGET, **changes))


@pytest.mark.parametrize(
    "changes",
    [{"ablate": a} for a in ABLATIONS[1:]]
    + [{"ablate": "no_mfma+no_scales"}, {"ablate": "no_mfma+no_lds_read"}]
    + [{"dma_probe": p} for p in DMA_PROBES[1:]]
    + [{"ablate": "no_mfma", "dma_probe": p, "frag_pipeline": f} for p in DMA_PROBES for f in FRAG_PIPELINES],
)
def test_diagnostics_build(changes):
    spec = replace(TARGET, **changes)
    assert is_valid_spec(spec)[0]
    _lower(spec)


@pytest.mark.parametrize("group", [8, 4, 1])
@pytest.mark.parametrize("pipeline", ["stage", "step"])
def test_mfma_groups_split_steps_and_keep_counts(group, pipeline):
    spec = replace(
        TARGET, stages=3, mfma_group=group, frag_pipeline=pipeline, lds_alias_scopes=True,
        a_layout="kblock", b_layout="kblock", a_scale_layout="kblock", b_scale_layout="kblock",
    )
    llvm = _lower(spec)
    mfmas = llvm.count('"s_nop 1\\0A\\09v_mfma_scale_f32_32x32x64_f8f6f4 $')
    lines = llvm.count("v_mfma_scale_f32_32x32x64_f8f6f4 $")
    # Every block starts with the head nop; the blocks still hold every MFMA.
    assert mfmas * group == lines
    assert lines % 32 == 0


@pytest.mark.parametrize("group", [16, 8, 4, 1])
@pytest.mark.parametrize("pipeline", ["stage", "step"])
def test_intrinsic_mfma_blocks_are_fenced_in_program_order(group, pipeline):
    spec = replace(
        TARGET, stages=3, mfma_group=group, frag_pipeline=pipeline, mfma_form="intrinsic", agpr_alloc="auto",
        b_layout="kblock", b_scale_layout="kblock",
    )
    llvm = _lower(spec)
    mfmas = llvm.count("call <16 x float> @llvm.amdgcn.mfma.scale")
    assert mfmas % 32 == 0 and "v_mfma" not in llvm
    if group < 16:
        # Every split block of a step that has fillers is fenced from the next.
        assert llvm.count("call void @llvm.amdgcn.sched.barrier(i32 0)") >= mfmas // group // 2
    direct = serialize(build_tiled_block_scaled_gemm(replace(spec, backend="direct"), arch="gfx950"))
    assert "inline_asm" not in direct
    assert direct.count("tile.mma") == mfmas


def test_mfma_group_ablations_build():
    for ablate in ("no_mfma", "no_dma", "no_mfma+no_scales"):
        for pipeline in ("stage", "step"):
            _lower(replace(TARGET, stages=3, mfma_group=4, frag_pipeline=pipeline, ablate=ablate))


def test_lds_alias_scopes_cover_dma_and_reads():
    plain = _lower(TARGET)
    assert "!alias.scope" not in plain
    llvm = _lower(replace(TARGET, lds_alias_scopes=True))
    dma = [l for l in llvm.splitlines() if "call void @llvm.amdgcn.raw.ptr.buffer.load.lds(" in l]
    reads = [l for l in llvm.splitlines() if re.search(r"= load <\d+ x i32>, ptr addrspace\(3\)", l)]
    assert dma and all("!alias.scope" in l for l in dma)
    assert reads and all("!noalias" in l for l in reads)


@pytest.mark.parametrize("stages", [2, 3])
def test_a_scale_fetch_iter_loads_each_row_once_per_iteration(stages):
    spec = replace(
        TARGET, stages=stages, a_scale_fetch="iter", b_layout="kblock", b_scale_layout="kblock",
        sched="interleave",
    )
    llvm = _lower(spec)
    loads = re.findall(rf"= load <{stages} x i32>, ptr addrspace\(1\)", llvm)
    # Four A rows per lane, loaded in the prologue and once in the loop body.
    assert len(loads) == 2 * 4
    per_stage = replace(spec, a_scale_fetch="stage")
    scalar = r"= load i32, ptr addrspace\(1\)"
    assert len(re.findall(scalar, _lower(per_stage))) > len(re.findall(scalar, llvm))


@pytest.mark.parametrize("group", [16, 1])
def test_a_ring_line_refills_whole_lines(group):
    spec = replace(
        TARGET, a_ring="line", b_layout="kblock", b_scale_layout="kblock", a_scale_fetch="iter",
        mfma_form="intrinsic", agpr_alloc="auto", backend="direct", mfma_group=group, xcd_remap="xcd_block",
    )
    assert is_valid_spec(spec) == (True, "ok")
    assert spec.lds_bytes == 3 * 256 * 128 + 2 * 256 * 96
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    assert "lds_dma_explicit" in text
    # A loop iteration is four stages: 128 MFMAs, B's DMA every stage and
    # three line refills of eight DMAs per wave; the prologue loads two lines
    # and B's first stage.
    assert text.count("tile.mma") == 4 * 32
    assert text.count("async_buffer_load_lds") == (2 * 8 + 6) + (4 * 6 + 3 * 8)
    # The A scales of an iteration's four stages come in one load per row.
    assert len(re.findall(r"global_load_vN .*vec = i:4", text)) == 2 * 4
    llvm = _lower(replace(spec, backend="llvm"))
    assert "lds_dma_explicit" not in llvm and llvm.count("call <16 x float> @llvm.amdgcn.mfma.scale") == 4 * 32


_CANONICAL_A = {
    "a_ring": "line", "b_layout": "kblock", "b_scale_layout": "kblock", "a_scale_fetch": "iter",
    "mfma_form": "intrinsic", "agpr_alloc": "auto", "backend": "direct", "mfma_group": 1, "xcd_remap": "xcd_block",
}


def _kernel_ops(text: str) -> tuple[list[str], list[str], list[str]]:
    """The serialized kernel's VMEM ops, waits, barriers and MFMAs before, in
    and after the K loop: M (MFMA), A / B (their DMA), Z (A scale DMA), G (A
    scale group load), s (another scale load), V (B load into registers),
    W<vmcnt>, | (barrier)."""
    a_rsrc = re.search(r"(%rsrc\d+) : vec<i32x4> = tile\.buffer_rsrc \( %A,", text).group(1)
    as_rsrc = re.search(r"(%rsrc\d+) : vec<i32x4> = tile\.buffer_rsrc \( %A_scale,", text)
    pre, rest = text.split("region @body {", 1) if "region @body {" in text else (text, "scf.yield")
    body, post = rest.split("scf.yield", 1)

    def ops(part: str) -> list[str]:
        out = []
        for line in part.splitlines():
            if "tile.s_waitcnt" in line and "vmcnt" in line:
                out.append("W" + re.search(r"vmcnt = i:(-?\d+)", line).group(1))
            elif "s_barrier" in line:
                out.append("|")
            elif "tile.mma" in line:
                out.append("M")
            elif "async_buffer_load_lds" in line:
                if as_rsrc and f"( {as_rsrc.group(1)}," in line:
                    out.append("Z")
                else:
                    out.append("A" if f"( {a_rsrc}," in line else "B")
            elif "tile.buffer_load_vN" in line:
                out.append("V")
            elif "global_load_vN" in line and "%A_scale" in line:
                out.append("G")
            elif "memref.global_load" in line:
                out.append("s")
        return out

    return ops(pre), ops(body), ops(post)


def _loop_trips(text: str) -> int:
    """The serialized kernel's K-loop trip count (its constant upper bound; 0 without a loop)."""
    loop = re.search(r"= scf\.for \( %\w+, (%\w+),", text)
    if loop is None:
        return 0
    return int(re.search(rf"{loop.group(1)} : i32 = arith\.constant .* value = i:(\d+)", text).group(1))


def _step1_read_gaps(text: str) -> list[int]:
    """LDS reads after each step-1 MFMA of a K-loop stage (the same in every stage of the body)."""
    body = text.split("region @body {", 1)[1].split("scf.yield", 1)[0]
    gaps: list[int] = []
    for line in body.splitlines():
        if "tile.mma" in line:
            gaps.append(0)
        elif "smem_load_vN" in line and gaps:
            gaps[-1] += 1
    per_stage = [gaps[s * 32 + 16 : s * 32 + 32] for s in range(len(gaps) // 32)]
    assert per_stage and all(p == per_stage[0] for p in per_stage), per_stage
    return per_stage[0]


def _check_barrier_waits(spec) -> list[int]:
    """Replay the kernel's VMEM ops in program order and check every stage
    barrier: the ops its vmcnt leaves in flight may not include data that
    stage or the next reads. Returns the loop body's waits."""
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    pre, body, post = _kernel_ops(text)
    loops = _loop_trips(text)
    in_flight, stage, n_z = [], 0, 0
    for op in pre + body * loops + post:
        if op.startswith("W"):
            n = int(op[1:])
            if n >= 0:
                in_flight = in_flight[max(0, len(in_flight) - n) :]
        elif op == "|":
            # The barrier ends epoch `stage - 1`. With the line ring no B DMA
            # may be in flight (one stage of lookahead) and only A refills of
            # that epoch (two); with the stage ring both have S - 1 stages.
            for kind, at in in_flight:
                if kind == "Z":
                    _check_scale_dma(stage, at)
                elif spec.a_ring == "line":
                    assert kind != "B" and (kind != "A" or at == stage - 1), (stage, in_flight)
                else:
                    assert kind not in "AB" or at > stage - spec.stages + 1, (stage, in_flight)
            stage += 1
        elif op in "ABGsV":
            in_flight.append((op, stage - 1))
        elif op == "Z":
            in_flight.append((op, n_z // 2))
            n_z += 1
    # The step pipeline adds the prologue's barrier.
    assert stage == spec.K // 128 + (spec.frag_pipeline == "step")
    return [int(op[1:]) for op in body if op.startswith("W") and int(op[1:]) >= 0]


@pytest.mark.parametrize("fill_split", ["none", "line1"])
@pytest.mark.parametrize("ring", ["line", "stage"])
def test_barrier_waits_keep_the_next_stage_resident(ring, fill_split):
    for shape in ({"K": 65536}, {"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 1152}):
        changes = dict(_CANONICAL_A, fill_split=fill_split, **shape)
        if ring == "stage":
            changes.update(a_ring="stage", stages=3)
        if "M" in shape:
            changes["xcd_remap"] = "none"
        spec = replace(TARGET, **changes)
        waits = _check_barrier_waits(spec)
        if ring == "line" and shape["K"] == 65536:
            # The refill (and in the second stage the A scale group) stays in flight.
            assert waits == ([8, 8, 0, 8] if fill_split == "none" else [8, 8, 4, 8])


def test_fill_split_moves_a_dma_and_scale_group_to_step_one():
    base = replace(TARGET, **_CANONICAL_A)
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    spec = replace(base, fill_split="line1")
    assert is_valid_spec(spec) == (True, "ok") and spec.kernel_name().endswith("_fsline1_direct")
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    for op in ("tile.mma", "async_buffer_load_lds", "global_load_vN", "memref.global_load_typed"):
        assert text.count(op) == plain.count(op), op
    for ring, n_a in (("line", 8), ("stage", 6)):
        spec = replace(base, fill_split="line1", **({} if ring == "line" else {"a_ring": "stage", "stages": 3}))
        _, body, _ = _kernel_ops(serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950")))
        stages = "".join(body).split("|")[1:]
        assert len(stages) == spec.unroll
        for phase, ops in enumerate(stages):
            # Step 0 keeps B's DMA; step 1 takes A's (with the line ring the
            # refill, which phase 1 does not have) and in phase 1 the A scales.
            cut = [i for i, c in enumerate(ops) if c == "M"][16]
            step0, step1 = ops[:cut], ops[cut:]
            assert step0.count("B") == 6 and "B" not in step1
            assert "A" not in step0 and step1.count("A") == (0 if ring == "line" and phase == 1 else n_a)
            assert "G" not in step0 and step1.count("G") == (4 if phase == 1 else 0)
    # Whole-step MFMA blocks have no step-1 fillers: the knob changes the name only.
    whole = replace(base, mfma_group=16)
    split = serialize(build_tiled_block_scaled_gemm(replace(whole, fill_split="line1"), arch="gfx950"))
    assert split.replace("_fsline1", "") == serialize(build_tiled_block_scaled_gemm(whole, arch="gfx950"))


def test_scale_opsel_rotation_selects_each_steps_byte():
    """Half h needs K32 block 2h in step 0 and 3-2h in step 1; after the
    v_perm rotation both halves read one byte per step."""

    def perm(src0: int, src1: int, sel: int) -> int:  # v_perm_b32, selector bytes 0..7
        pool = src1.to_bytes(4, "little") + src0.to_bytes(4, "little")
        return int.from_bytes(bytes(pool[(sel >> 8 * k) & 0xFF] for k in range(4)), "little")

    for x in (0x04030201, 0x80FF7F01, 0xDEADBEEF, 0x00000000, 0xFFFFFFFF, 0x12345678):
        for h in (0, 1):
            rot = perm(x, x, _PERM_IDENTITY ^ h * _PERM_ROT16_XOR)
            for step, blk in ((0, 2 * h), (1, 3 - 2 * h)):
                assert (rot >> 8 * _STEP_SCALE_BYTE[step]) & 0xFF == (x >> 8 * blk) & 0xFF


@pytest.mark.parametrize("fill_split", ["none", "line1"])
def test_scale_opsel_selects_bytes_in_the_mfma(fill_split):
    base = replace(TARGET, **_CANONICAL_A, setprio=True, fill_split=fill_split)
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    spec = replace(base, scale_opsel=True)
    assert is_valid_spec(spec) == (True, "ok") and "_so" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))

    def body(t: str) -> str:
        return t.split("region @body {", 1)[1].split("scf.yield", 1)[0]

    # Per loop iteration: 16 scale shifts per stage gone; one rotation per
    # dword (4 stages of B's 4, and the next A group's 16) instead.
    assert body(plain).count("arith.lshr") - body(text).count("arith.lshr") == 4 * 16
    assert body(text).count("tile.perm_b32") == 4 * 4 + 16
    sels = re.findall(r"tile\.mma .*scale_sel = l:\[ i:(\d), i:(\d) \]", body(text))
    assert sels == ([("0", "0")] * 16 + [("3", "3")] * 16) * 4
    for op in ("tile.mma", "async_buffer_load_lds", "global_load_vN", "memref.global_load_typed"):
        assert text.count(op) == plain.count(op), op
    # The A group lands in loop-carried dwords: no vector extract per stage.
    assert "vector.extract" not in body(text).split("tile.perm_b32", 1)[0]
    _check_barrier_waits(spec)
    # Small shapes with a tail and the stage ring build too.
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"a_ring": "stage", "stages": 3, "K": 1408, "M": 512, "N": 512}):
        small = replace(spec, xcd_remap="none", **changes)
        assert is_valid_spec(small)[0]
        _lower(small)


def test_scale_opsel_asm_modifiers():
    for b_layout, b_sel in (("row", 0), ("kblock", 1)):
        spec = replace(TARGET, scale_opsel=True, b_scale_layout=b_layout, a_scale_fetch="iter", mfma_group=4)
        llvm = _lower(spec)
        step0 = llvm.count("op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2")
        step1 = llvm.count(f"op_sel:[1,{b_sel},0] op_sel_hi:[1,{b_sel},0] cbsz:2 blgp:2")
        assert step0 == step1 > 0 and "llvm.amdgcn.perm" in llvm
        assert llvm.count("v_mfma_scale_f32_32x32x64_f8f6f4 $") == step0 + step1
    assert "op_sel:" not in _lower(TARGET)


@pytest.mark.parametrize("scale_opsel", [False, True])
def test_line_ring_step_pipeline_puts_the_barrier_between_the_steps(scale_opsel):
    spec = replace(TARGET, **_CANONICAL_A, frag_pipeline="step", scale_opsel=scale_opsel)
    assert is_valid_spec(spec) == (True, "ok") and "_fpstep" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    pre, body, _ = _kernel_ops(text)
    # The prologue: lines 0 and 1 and B's stage 0, vmcnt(0), Bar(0), B's stage 1.
    assert "".join(pre).endswith("W0|" + "B" * 6) and pre.count("A") == 16
    # Between barriers: one stage's step 1, then the next stage's step 0.
    parts = "".join(body).split("|")
    assert len(parts) == 5
    step0, step1 = {0: parts[0]}, {3: parts[4]}
    for p in (1, 2, 3):
        cut = [i for i, c in enumerate(parts[p]) if c == "M"][16]
        step1[p - 1], step0[p] = parts[p][:cut], parts[p][cut:]
    # Step 0's blocks take the next stage's B scale loads, then the refill
    # (not in phase 1) and in phase 1 the next A scale group; step 1's take B
    # two stages ahead.
    for phase in range(4):
        s0, s1 = step0[phase], step1[phase]
        assert s0.count("M") == 16 and s1.count("M") == 16, phase
        assert s0.count("s") == 4 and s0.count("A") == (0 if phase == 1 else 8) and "B" not in s0
        assert s0.count("G") == (4 if phase == 1 else 0)
        assert s1.count("B") == 6 and not set("AsG") & set(s1)
        # B scales issue before the refill, so waiting for them never waits for it.
        if "A" in s0:
            assert s0.rindex("s") < s0.index("A")
    assert _check_barrier_waits(spec) == [12, 8, 12, 12]
    # No fragment read between a barrier and the MFMA after it.
    loop = text.split("region @body {", 1)[1].split("scf.yield", 1)[0]
    for part in loop.split("tile.s_barrier_bare")[1:]:
        assert "smem_load_vN" not in part.split("tile.mma", 1)[0]
    # Same work as the stage pipeline, plus one stage's discarded reads.
    plain = serialize(build_tiled_block_scaled_gemm(replace(spec, frag_pipeline="stage"), arch="gfx950"))
    assert text.count("tile.mma") == plain.count("tile.mma") == 4 * 32
    assert text.count("async_buffer_load_lds") == plain.count("async_buffer_load_lds") + 6
    assert text.count("smem_load_vN") == plain.count("smem_load_vN") + 16
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 512}, {"mfma_group": 16},
                    {"M": 512, "N": 512, "K": 1152, "a_scale_fetch": "stage", "b_scale_layout": "row"}):
        small = replace(spec, **changes, **({"xcd_remap": "none"} if "M" in changes else {}))
        assert is_valid_spec(small)[0]
        _check_barrier_waits(small)
        _lower(small)
    # The LLVM asm form lowers too (barrier waits drain every DMA there).
    asm = replace(TARGET, a_ring="line", frag_pipeline="step", a_scale_fetch="iter", mfma_group=4,
                  scale_opsel=scale_opsel)
    assert is_valid_spec(asm)[0]
    assert _lower(asm).count("v_mfma_scale_f32_32x32x64_f8f6f4 $") == 4 * 32


@pytest.mark.parametrize("order", STEP_VMEM_ORDERS)
def test_step_vmem_order_places_each_steps_vmem_and_counts_its_waits(order):
    base = replace(TARGET, **_CANONICAL_A, frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True)
    spec = replace(base, step_vmem_order=order)
    assert is_valid_spec(spec) == (True, "ok")
    assert (order in spec.kernel_name()) == (order != "reads_first")
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    for op in ("tile.mma", "async_buffer_load_lds", "global_load_vN", "memref.global_load_typed", "smem_load_vN"):
        assert text.count(op) == plain.count(op), op
    _, body, _ = _kernel_ops(text)
    parts = "".join(body).split("|")
    step1 = {p - 1: parts[p][: [i for i, c in enumerate(parts[p]) if c == "M"][16]] for p in (1, 2, 3)}
    step1[3] = parts[4]
    step0 = {0: parts[0], **{p: parts[p][len(step1[p - 1]) :] for p in (1, 2, 3)}}
    for phase in range(4):
        s0, s1 = step0[phase], step1[phase]
        late = order == "group_late" and phase == 1
        assert s0.count("G") == (0 if late or phase != 1 else 4) and s1.count("G") == (4 if late else 0)
        if late:
            assert s1.index("G") > s1.rindex("B")
        if order == "b_early":
            # B's k-th DMA right after the step's MFMA 2k.
            assert [s1[: i].count("M") for i, c in enumerate(s1) if c == "B"] == [1, 3, 5, 7, 9, 11]
        if order == "refill_early":
            n = len([c for c in s0 if c in "AsG"])
            assert [s0[: i].count("M") for i, c in enumerate(s0) if c in "AsG"] == list(range(1, n + 1))
        if order == "frag_first":
            # The eight fragment reads first, then B's k-th DMA right after the step's MFMA 8 + k.
            assert [s1[: i].count("M") for i, c in enumerate(s1) if c == "B"] == [9, 10, 11, 12, 13, 14]
        if order == "frag_mid":
            # B's k-th DMA right after the step's MFMA k + 1.
            assert [s1[: i].count("M") for i, c in enumerate(s1) if c == "B"] == [1, 2, 3, 4, 5, 6]
    if order == "frag_first":
        # Per MFMA gap of step 1: one fragment (two reads) in the first eight, none after.
        assert _step1_read_gaps(text) == [2] * 8 + [0] * 8
    if order == "frag_mid":
        # One fragment per gap from the fifth to the twelfth.
        assert _step1_read_gaps(text) == [0] * 4 + [2] * 8 + [0] * 4
    # Each barrier waits for B and leaves what was issued after it in flight.
    expected = [12, 4, 16, 12] if order == "group_late" else [12, 8, 12, 12]
    assert _check_barrier_waits(spec) == expected
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 1152}, {"M": 512, "N": 512, "K": 512},
                    {"mfma_group": 4}, {"k_peel": True}):
        small = replace(spec, **changes, **({"xcd_remap": "none"} if "M" in changes else {}))
        assert is_valid_spec(small)[0]
        _check_barrier_waits(small)
        _lower(small)


@pytest.mark.parametrize("order", ["reads_first", "frag_first", "frag_mid"])
@pytest.mark.parametrize("place", [p for p in DMA_PLACES[1:] if p.startswith("even")])
def test_dma_place_puts_every_dma_two_blocks_after_the_last(place, order):
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half",
                   a_scale_path="lds2", step_vmem_order=order)
    spec = replace(base, dma_place=place)
    assert is_valid_spec(spec) == (True, "ok") and f"_dp{place}" in spec.kernel_name() and spec.dma_placed
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    for op in ("tile.mma", "global_load_vN", "memref.global_load_typed", "smem_load_vN"):
        assert text.count(op) == plain.count(op), op
    assert text.count("async_buffer_load_lds") == plain.count("async_buffer_load_lds") + 4
    pre, body, _ = _kernel_ops(text)
    # The prologue's step 1 takes the first half of stage 0's refill ahead of B's half.
    assert "".join(pre).endswith("W0|" + "BBB" + "AAAA" + "BBB" + "W-1|")
    halves = "".join(body).split("|")[:-1]
    assert len(halves) == 2 * spec.unroll * 2
    for k, part in enumerate(halves):
        phase, step = k // 2 % 4, k % 2
        dmas = [(part[:i].count("M") - 1, c) for i, c in enumerate(part) if c in "ABZ"]
        gaps = [g for g, _ in dmas]
        # Odd gaps only, one DMA each: two MFMA blocks between a wave's DMAs.
        assert all(g % 2 for g in gaps) and len(set(gaps)) == len(gaps), (k, dmas)
        assert "".join(c for _, c in dmas).count("B") == 3, k
        # A refill in two halves: four DMAs after Bar(s) in step 1 of stage
        # s - 1 and four in step 0 of stage s (phase 1 has none).
        n_a = 0 if (phase if step == 0 else (phase + 1) % 4) == 1 else 4
        a_b = "".join(c for _, c in dmas if c in "AB")
        assert a_b == ("BBB" + "A" * n_a if place == "even_bfirst" else "A" * n_a + "BBB"), (k, dmas)
    # Step 1's reads stay where step_vmem_order puts them; step 0's eight
    # fragments (two reads each) go one per second block from the first.
    assert _step1_read_gaps(text) == _step1_read_gaps(plain)
    loop = text.split("region @body {", 1)[1].split("scf.yield", 1)[0]
    reads = []
    for line in loop.splitlines():
        if "tile.mma" in line:
            reads.append(0)
        elif "smem_load_vN" in line and reads:
            reads[-1] += 1
    assert all(reads[s * 32 : s * 32 + 16] == [2, 0] * 8 for s in range(len(reads) // 32)), reads
    _check_split_barrier_waits(spec)
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 1152}, {"M": 512, "N": 512, "K": 512},
                    {"mfma_group": 4}, {"k_peel": True}):
        small = replace(spec, **changes, **({"xcd_remap": "none"} if "M" in changes else {}))
        assert is_valid_spec(small)[0]
        _check_split_barrier_waits(small)
        _lower(small)


@pytest.mark.parametrize("order", ["reads_first", "frag_first", "frag_mid"])
@pytest.mark.parametrize("place, per_gap", [("dense_bfirst", 1), ("pair_bfirst", 2)])
def test_dma_place_packed_puts_b_then_the_refill_in_consecutive_gaps(place, per_gap, order):
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half",
                   a_scale_path="lds2", step_vmem_order=order)
    spec = replace(base, dma_place=place)
    assert is_valid_spec(spec) == (True, "ok") and f"_dp{place}" in spec.kernel_name() and spec.dma_placed
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    spaced = serialize(build_tiled_block_scaled_gemm(replace(base, dma_place="even_bfirst"), arch="gfx950"))
    for op in ("tile.mma", "global_load_vN", "memref.global_load_typed", "smem_load_vN", "async_buffer_load_lds"):
        assert text.count(op) == spaced.count(op), op
    pre, body, _ = _kernel_ops(text)
    assert "".join(pre).endswith("W0|" + "BBB" + "AAAA" + "BBB" + "W-1|")
    halves = "".join(body).split("|")[:-1]
    assert len(halves) == 2 * spec.unroll * 2
    for k, part in enumerate(halves):
        phase, step = k // 2 % 4, k % 2
        dmas = [(part[:i].count("M") - 1, c) for i, c in enumerate(part) if c in "AB"]
        n_a = 0 if (phase if step == 0 else (phase + 1) % 4) == 1 else 4
        # B's half, then the refill's part, `per_gap` per gap from gap 1.
        assert "".join(c for _, c in dmas) == "BBB" + "A" * n_a, (k, dmas)
        assert [g for g, _ in dmas] == [1 + i // per_gap for i in range(len(dmas))], (k, dmas)
    assert _step1_read_gaps(text) == _step1_read_gaps(spaced)
    _check_split_barrier_waits(spec)
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 1152}, {"mfma_group": 4},
                    {"k_peel": True}):
        small = replace(spec, **changes, **({"xcd_remap": "none"} if "M" in changes else {}))
        assert is_valid_spec(small)[0]
        _check_split_barrier_waits(small)
        _lower(small)
    assert not is_valid_spec(replace(spec, step_vmem_order="b_early"))[0]
    # Elsewhere the value changes the name only.
    for other in (replace(base, b_release="full"), replace(base, frag_pipeline="stage"), TARGET):
        moved = serialize(build_tiled_block_scaled_gemm(replace(other, dma_place=place), arch="gfx950"))
        assert moved.replace(f"_dp{place}", "") == serialize(build_tiled_block_scaled_gemm(other, arch="gfx950"))


def _barrier_groups(text: str) -> list[tuple[int, int, int, int]]:
    """Per K-loop barrier: (its MFMA block in the step, MFMAs back to the last
    lgkmcnt(0), to the last vmcnt wait, to the last LDS read), over the back
    edge where the body has none before it. Asserts that no LDS read issues
    between a barrier and the lgkmcnt(0) wait before it."""
    body = text.split("region @body {", 1)[1].split("scf.yield", 1)[0]
    seq, m = [], 0
    for line in body.splitlines():
        if "tile.mma" in line:
            m += 1
        elif "smem_load_vN" in line:
            seq.append((m, "R"))
        elif "tile.s_waitcnt" in line:
            if re.search(r"lgkmcnt = i:0\b", line):
                seq.append((m, "L"))
            if not re.search(r"vmcnt = i:-1\b", line):
                seq.append((m, "V"))
        elif "s_barrier" in line:
            seq.append((m, "|"))
    out = []
    for k, (mb, kind) in enumerate(seq):
        if kind != "|":
            continue
        ring = seq[:k][::-1] + [(mm - m, c) for mm, c in seq[k:][::-1]]
        back = {c: mb - next(mm for mm, cc in ring if cc == c) for c in "LVR"}
        lgkm = next(i for i, (_, c) in enumerate(ring) if c == "L")
        assert all(c != "R" for _, c in ring[:lgkm]), (mb, ring[:lgkm])
        out.append(((mb - 1) % 16, back["L"], back["V"], back["R"]))
    return out


@pytest.mark.parametrize("sync", SYNC_PLACES[1:])
def test_sync_place_moves_the_barrier_groups_and_keeps_their_waits(sync):
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half",
                   a_scale_path="lds2", mfma_order="serp", dma_place="even_bfirst")
    spec = replace(base, sync_place=sync)
    assert is_valid_spec(spec) == (True, "ok") and f"_sp{sync}" in spec.kernel_name() and spec.sync_placed
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    for op in ("tile.mma", "async_buffer_load_lds", "global_load_vN", "memref.global_load_typed", "smem_load_vN",
               "s_barrier_bare"):
        assert text.count(op) == plain.count(op), op
    # The DMAs keep their gaps.
    _, body, _ = _kernel_ops(text)
    _, plain_body, _ = _kernel_ops(plain)
    assert [c for c in body if c in "MABZGs"] == [c for c in plain_body if c in "MABZGs"]
    groups = _barrier_groups(text)
    assert all(g == (15, 0, 0, g[3]) for g in _barrier_groups(plain))
    early = sync.startswith("split")
    for block, lgkm, vm, read in groups:
        # The barrier after the step's last block, or after the next step's first.
        assert block == (0 if "sink" in sync else 15), groups
        assert lgkm == (2 if sync == "split_sink" else 1 if early else 0), groups
        assert vm == (1 if sync == "split" else lgkm if early and sync != "split_lgkm" else 0), groups
        # No read in a step's last two gaps.
        assert not early or read >= 2 + ("sink" in sync), groups
    waits = _check_split_barrier_waits(spec)
    assert len(waits) == len(_check_split_barrier_waits(base))
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 1152}, {"M": 512, "N": 512, "K": 512},
                    {"mfma_group": 4}, {"k_peel": True}):
        small = replace(spec, **changes, **({"xcd_remap": "none"} if "M" in changes else {}))
        assert is_valid_spec(small)[0]
        _check_split_barrier_waits(small)
        _lower(small)
    # Elsewhere the value changes the name only.
    for other in (replace(base, dma_place="default"), replace(base, b_release="full"), TARGET):
        moved = serialize(build_tiled_block_scaled_gemm(replace(other, sync_place=sync), arch="gfx950"))
        assert moved.replace(f"_sp{sync}", "") == serialize(build_tiled_block_scaled_gemm(other, arch="gfx950"))


_PLACED = {**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2", "frag_pipeline": "step",
           "fill_split": "line1", "scale_opsel": True, "setprio": True, "b_release": "half", "a_scale_path": "lds2",
           "mfma_order": "serp", "dma_place": "even_bfirst"}


def test_prologue_wait_stage0_leaves_line_1_in_flight_until_bar_1():
    base = replace(TARGET, **_PLACED)
    spec = replace(base, prologue_wait="stage0")
    assert is_valid_spec(spec) == (True, "ok") and "_pwstage0_" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    for op in ("tile.mma", "async_buffer_load_lds", "smem_load_vN", "s_barrier_bare"):
        assert text.count(op) == plain.count(op), op
    pre, _, _ = _kernel_ops(text)
    # Line 0, B's stage 0, the A scale DMA, then line 1, which the wait leaves in flight.
    assert "".join(pre).startswith("s" + "A" * 8 + "B" * 6 + "ZZ" + "A" * 8 + "W8|")
    assert "".join(_kernel_ops(plain)[0]).startswith("s" + "A" * 16 + "B" * 6 + "ZZ" + "W0|")
    # The replay checks that Bar(1) finds line 1 landed.
    assert _check_split_barrier_waits(spec) == _check_split_barrier_waits(base)
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 512}, {"sync_place": "sink"}):
        small = replace(spec, **changes, **({"xcd_remap": "none"} if "M" in changes else {}))
        _check_split_barrier_waits(small)
        _lower(small)
    for other in (replace(base, b_release="full"), TARGET):
        moved = serialize(build_tiled_block_scaled_gemm(replace(other, prologue_wait="stage0"), arch="gfx950"))
        assert moved.replace("_pwstage0", "") == serialize(build_tiled_block_scaled_gemm(other, arch="gfx950"))


@pytest.mark.parametrize("store_nt", [False, True])
@pytest.mark.parametrize("epilogue", EPILOGUES)
def test_epilogue_stores_every_element_once(epilogue, store_nt):
    spec = replace(TARGET, **_PLACED, epilogue=epilogue, store_nt=store_nt)
    assert is_valid_spec(spec) == (True, "ok")
    assert (f"_ep{epilogue}" in spec.kernel_name()) == (epilogue != "default")
    assert ("_snt" in spec.kernel_name()) == store_nt
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    post = text.split("scf.yield", 1)[1]
    stores = re.findall(r"memref\.global_store_(?:typed|vN) \( %C, .*\}", post)
    widths = [int(re.search(r"vec = i:(\d+)", s).group(1)) if "vec = " in s else 1 for s in stores]
    # 16 accumulators of 16 elements per lane.
    assert sum(widths) == 256 and set(widths) == {{"ct": 4, "ct4": 8, "lds": 8, "ct_lds": 8}.get(epilogue, 1)}
    assert all(("nontemporal = b:true" in s) == store_nt for s in stores)
    swapped = epilogue in ("ct", "ct4", "ct_lds")
    assert post.count("vector.trunc_f32_to") == {"pk2": 128, "ct": 64, "ct4": 64, "lds": 128, "ct_lds": 64}.get(
        epilogue, 0)
    assert post.count("tile.permlane32_swap") == (64 if epilogue == "ct4" else 0)
    plain = serialize(build_tiled_block_scaled_gemm(replace(spec, epilogue="default", store_nt=False), arch="gfx950"))
    loop, loop_plain = (t.split("region @body {", 1)[1].split("scf.yield", 1)[0] for t in (text, plain))
    if not swapped:
        assert loop == loop_plain
    else:
        # ct: the same MFMAs with A and B (operands, scales and their byte selectors) swapped.
        pat = r"tile\.mma \( (%\w+), (%\w+), (%\w+), (%\w+), (%\w+) \) .*scale_sel = l:\[ i:(\d), i:(\d) \]"
        mma, mma_plain = re.findall(pat, loop), re.findall(pat, loop_plain)
        assert len(mma) == len(mma_plain) == 8 * 32
        assert [(m[1], m[0], m[2], m[4], m[3], m[6], m[5]) for m in mma] == [m[:5] + m[5:] for m in mma_plain]
    _lower(spec)
    _lower(replace(TARGET, epilogue=epilogue, store_nt=store_nt))


@pytest.mark.parametrize("epilogue, unstaged", [("lds", "sa"), ("ct_lds", "ct")])
def test_epilogue_lds_stages_c_through_the_line_ring_after_one_barrier(epilogue, unstaged):
    spec = replace(TARGET, **_PLACED, epilogue=epilogue)
    assert is_valid_spec(spec) == (True, "ok") and spec.c_staged
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(replace(spec, epilogue=unstaged), arch="gfx950"))
    post, post_plain = (t.split("scf.yield", 1)[1] for t in (text, plain))
    assert post.count("s_barrier") == post_plain.count("s_barrier") + 1
    # The barrier comes after the drained DMA and the last LDS reads, ahead of every C write.
    head = post.split("s_barrier", post_plain.count("s_barrier") + 1)
    assert "tile.smem_store" not in "".join(head[:-1]) and "s_waitcnt" in head[-2]
    # Two 64-row passes of a 128 x 128 bf16 tile: dwords ("lds") or 8-byte pieces ("ct_lds").
    assert post.count("tile.smem_store") == (128 if epilogue == "lds" else 64)
    assert post.count("tile.dpp_xor") == (128 if epilogue == "lds" else 0)
    assert len(re.findall(r"tile\.smem_load_vN", post)) == 32
    loop, loop_plain = (t.split("region @body {", 1)[1].split("scf.yield", 1)[0] for t in (text, plain))
    assert loop == loop_plain
    # Off the line ring the name changes only.
    for other in (TARGET,):
        moved = serialize(build_tiled_block_scaled_gemm(replace(other, epilogue=epilogue), arch="gfx950"))
        same = serialize(build_tiled_block_scaled_gemm(replace(other, epilogue=unstaged), arch="gfx950"))
        assert moved.replace(f"_ep{epilogue}", f"_ep{unstaged}") == same
    for changes in ({"M": 512, "N": 512, "K": 1152, "xcd_remap": "none"}, {"store_nt": True}, {"k_peel": True}):
        _lower(replace(spec, **changes))


def _check_scale_dma(t: int, epoch: int) -> None:
    """`a_scale_path="lds2"`: a wave's n-th A scale DMA copies epoch n // 2
    (stages 8e..8e+7), whose first dword stage 8e - 1 reads after Bar(8e); at
    Bar(t) or later it may no longer be in flight."""
    assert t < 8 * epoch, (t, epoch)


def _check_split_barrier_waits(spec) -> list[int]:
    """`_check_barrier_waits` for `b_release="half"`. The n-th B DMA carries
    half (n // 3) % 2 of stage n // 6. Barriers alternate Bar(t), before step
    1 of stage t - 1 (which reads B(t)'s first half), and E(t), before step 0
    of stage t (which reads B(t)'s second half). Returns the loop body's waits."""
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    pre, body, post = _kernel_ops(text)
    in_flight, n_b, n_bar, n_z = [], 0, 0, 0
    for op in pre + body * _loop_trips(text) + post:
        if op.startswith("W"):
            n = int(op[1:])
            if n >= 0:
                in_flight = in_flight[max(0, len(in_flight) - n) :]
        elif op == "|":
            t, end = divmod(n_bar, 2)
            for kind, at in in_flight:
                if kind == "Z":
                    _check_scale_dma(t, at)
                elif kind == "B":
                    assert at[0] > t or (not end and at == (t, 1)), (n_bar, in_flight)
                elif kind == "A" and not end:
                    # Only refills issued after the previous Bar may be in flight.
                    assert at == t - 1, (n_bar, in_flight)
            n_bar += 1
        elif op == "B":
            in_flight.append(("B", (n_b // 6, n_b // 3 % 2)))
            n_b += 1
        elif op in "AGs":
            in_flight.append((op, (n_bar - 1) // 2))
        elif op == "Z":
            in_flight.append((op, n_z // 2))
            n_z += 1
    # A last stage emitted inside the loop keeps its (unneeded) E.
    assert n_bar - 2 * (spec.K // 128) in (1, 2)
    return [int(op[1:]) for op in body if op.startswith("W") and int(op[1:]) >= 0]


def test_b_release_half_rejects_thread_counts_that_split_its_chunks():
    # 256 B columns x 3 chunks per half: 8 waves (512 threads) would copy 512
    # of the 768 chunks and leave a third of every B half unwritten.
    spec = replace(TARGET, **_CANONICAL_A, frag_pipeline="step", fill_split="line1", scale_opsel=True,
                   b_release="half", waves_n=4, waves_per_eu=2)
    assert spec.b_split and spec.threads == 512
    ok, why = is_valid_spec(spec)
    assert not ok and "b_release=half" in why
    with pytest.raises(ValueError):
        build_tiled_block_scaled_gemm(spec, arch="gfx950")
    assert is_valid_spec(replace(spec, b_release="full")) == (True, "ok")


def test_b_release_half_gives_each_b_half_its_own_barrier():
    base = replace(TARGET, **_CANONICAL_A, frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True)
    spec = replace(base, b_release="half")
    assert is_valid_spec(spec) == (True, "ok") and "_brhalf" in spec.kernel_name() and spec.b_split
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    for op in ("tile.mma", "async_buffer_load_lds", "global_load_vN", "memref.global_load_typed", "smem_load_vN"):
        assert text.count(op) == plain.count(op), op
    _, body, _ = _kernel_ops(text)
    # Per stage: step 0 (B's first half three DMAs, then the scales and the
    # refill or group), Bar, step 1 (B's second half), E.
    stages = "".join(body).split("|")[:-1]
    assert len(stages) == 2 * spec.unroll
    for k, part in enumerate(stages):
        assert part.count("B") == 3 and part.count("M") == 16, k
        if k % 2 == 0:
            assert part.index("B") < min(part.index(c) for c in "As" if c in part)
        else:
            assert not set("AsG") & set(part)
    # Bar(s+1) leaves stage s's step-0 VMEM and B(s+1)'s second half in
    # flight; E(s+1) waits for that half only.
    assert _check_split_barrier_waits(spec) == [18, 18, 14, 14, 26, 18, 18, 18]
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 1152}, {"M": 512, "N": 512, "K": 512},
                    {"mfma_group": 4}, {"k_peel": True}, {"step_vmem_order": "b_early"},
                    {"step_vmem_order": "group_late"}):
        small = replace(spec, **changes, **({"xcd_remap": "none"} if "M" in changes else {}))
        assert is_valid_spec(small)[0]
        _check_split_barrier_waits(small)
        _lower(small)
    for ablate in ("no_dma", "no_mfma", "no_scales"):
        _lower(replace(spec, ablate=ablate))
    # Without the step pipeline the knob changes the name only.
    stage = replace(base, frag_pipeline="stage")
    half = serialize(build_tiled_block_scaled_gemm(replace(stage, b_release="half"), arch="gfx950"))
    assert half.replace("_brhalf", "") == serialize(build_tiled_block_scaled_gemm(stage, arch="gfx950"))


@pytest.mark.parametrize("b_release", ["full", "half"])
def test_a_scale_fetch_iter2_loads_the_group_every_other_iteration(b_release):
    base = replace(TARGET, **_CANONICAL_A, frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True,
                   b_release=b_release)
    spec = replace(base, a_scale_fetch="iter2")
    assert is_valid_spec(spec) == (True, "ok") and "_asiter2" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))

    def body(t: str) -> str:
        return t.split("region @body {", 1)[1].split("scf.yield", 1)[0]

    # A two-iteration body: twice the MFMAs, DMA and reads, the group loads
    # once, and one swap per group dword.
    for op in ("tile.mma", "async_buffer_load_lds", "smem_load_vN", "memref.global_load_typed"):
        assert body(text).count(op) == 2 * body(plain).count(op), op
    assert body(text).count("global_load_vN") == body(plain).count("global_load_vN") == 4
    assert body(text).count("tile.permlane32_swap") == 4 * 4 and "permlane32_swap" not in body(plain)
    assert _loop_trips(text) * 2 == _loop_trips(plain)
    check = _check_split_barrier_waits if spec.b_split else _check_barrier_waits
    check(spec)
    # Odd iteration counts, tails, k_peel and single iterations.
    for changes in ({"K": 1408}, {"K": 1152}, {"K": 512}, {"K": 2560}, {"K": 2560, "k_peel": True},
                    {"K": 3072, "k_peel": True}, {"K": 1024}):
        small = replace(spec, M=512, N=512, xcd_remap="none", **changes)
        assert is_valid_spec(small)[0]
        check(small)
        _lower(small)
    _lower(spec)
    # Elsewhere the value fetches as "iter": the name changes only.
    for changes in ({"frag_pipeline": "stage"}, {"scale_opsel": False}):
        other = replace(base, **changes)
        two = serialize(build_tiled_block_scaled_gemm(replace(other, a_scale_fetch="iter2"), arch="gfx950"))
        assert two.replace("_asiter2", "_asiter") == serialize(build_tiled_block_scaled_gemm(other, arch="gfx950"))


@pytest.mark.parametrize("b_release", ["full", "half"])
def test_a_scale_path_lds2_copies_each_epoch_once_and_reads_scales_from_lds(b_release):
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release=b_release)
    spec = replace(base, a_scale_path="lds2")
    assert is_valid_spec(spec) == (True, "ok") and "_aplds2" in spec.kernel_name() and spec.a_lds_scales
    assert spec.lds_bytes == base.lds_bytes + 2 * 256 * 32 == 160 * 1024
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))

    def body(t: str) -> str:
        return t.split("region @body {", 1)[1].split("scf.yield", 1)[0]

    # One epoch of eight stages per loop body: two scale DMAs per wave instead
    # of the four group loads and their swaps, and per stage one LDS read and
    # one rotation per A row block.
    assert body(text).count("tile.mma") == body(plain).count("tile.mma") == 8 * 32
    assert body(text).count("async_buffer_load_lds") == body(plain).count("async_buffer_load_lds") + 2
    assert "%A_scale" not in body(text).replace("tile.buffer_rsrc ( %A_scale", "")
    assert "permlane32_swap" not in body(text)
    assert body(text).count("smem_load_vN") == body(plain).count("smem_load_vN") + 8 * 4
    assert body(text).count("tile.perm_b32") == 8 * 4
    assert _loop_trips(text) == _loop_trips(plain)
    # The DMAs issue in the epoch's second stage, step 0.
    _, ops, _ = _kernel_ops(text)
    parts = "".join(ops).split("|")
    assert "".join(ops).count("Z") == 2 and parts[1 if b_release == "full" else 2].count("Z") == 2
    check = _check_split_barrier_waits if spec.b_split else _check_barrier_waits
    # Two scale DMAs stay in flight where the four group loads did.
    if spec.b_split:
        assert check(base) == [15, 15, 11, 11, 20, 15, 15, 15, 15, 15, 7, 7, 16, 15, 15, 15]
        assert check(spec) == [15, 15, 9, 9, 18, 15, 15, 15, 15, 15, 7, 7, 16, 15, 15, 15]
    else:
        assert check(base) == [9, 5, 9, 9, 9, 1, 9, 9] and check(spec) == [9, 3, 9, 9, 9, 1, 9, 9]
    for changes in ({"K": 1408}, {"K": 1152}, {"K": 512}, {"K": 2560}, {"K": 2560, "k_peel": True},
                    {"K": 3072, "k_peel": True}, {"K": 1024}, {"K": 384, "a_scale_fetch": "stage"},
                    {"K": 1408, "scale_opsel": False}):
        small = replace(spec, M=512, N=512, xcd_remap="none", **changes)
        assert is_valid_spec(small)[0]
        check(small)
        _lower(small)
    _lower(spec)
    for ablate in ("no_dma", "no_mfma", "no_scales"):
        _lower(replace(spec, ablate=ablate))
    # Elsewhere the value changes the name only.
    for changes in ({"frag_pipeline": "stage"}, {"backend": "llvm", "mfma_form": "asm_agpr", "agpr_alloc": "256,256"}):
        other = replace(base, **changes)
        lds2 = serialize(build_tiled_block_scaled_gemm(replace(other, a_scale_path="lds2"), arch="gfx950"))
        assert lds2.replace("_aplds2", "") == serialize(build_tiled_block_scaled_gemm(other, arch="gfx950"))
    assert not is_valid_spec(replace(spec, a_scale_path="lds3"))[0]


@pytest.mark.parametrize("b_release", ["full", "half"])
def test_as_read_b128x4_reads_four_stages_of_a_scales_at_once(b_release):
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release=b_release,
                   a_scale_path="lds2")
    spec = replace(base, as_read="b128x4")
    assert is_valid_spec(spec) == (True, "ok") and "_arb128x4" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))

    def body(t: str) -> str:
        return t.split("region @body {", 1)[1].split("scf.yield", 1)[0]

    def as_reads(t: str) -> list[str]:
        return re.findall(r"smem_load_vN \( %a_scales\w*, .*vec = i:(\d)", t)

    # Per eight-stage body and A row block one dword read per stage before,
    # one 16-byte read per four stages now; the rotations stay per stage.
    assert as_reads(body(plain)) == ["1"] * 8 * 4 and as_reads(body(text)) == ["4"] * 2 * 4
    for op in ("tile.mma", "async_buffer_load_lds", "tile.perm_b32"):
        assert body(text).count(op) == body(plain).count(op), op
    check = _check_split_barrier_waits if spec.b_split else _check_barrier_waits
    assert check(spec) == check(base)
    for changes in ({"K": 1408}, {"K": 1152}, {"K": 512}, {"K": 2560}, {"K": 2560, "k_peel": True},
                    {"K": 3072, "k_peel": True}, {"K": 1024}, {"K": 384, "a_scale_fetch": "stage"},
                    {"K": 1408, "scale_opsel": False}, {"K": 1152, "step_vmem_order": "frag_mid"}):
        small = replace(spec, M=512, N=512, xcd_remap="none", **changes)
        assert is_valid_spec(small)[0]
        check(small)
        _lower(small)
    _lower(spec)
    for ablate in ("no_dma", "no_mfma", "no_scales"):
        _lower(replace(spec, ablate=ablate))
    # Elsewhere the value changes the name only.
    other = replace(base, a_scale_path="direct")
    b128 = serialize(build_tiled_block_scaled_gemm(replace(other, as_read="b128x4"), arch="gfx950"))
    assert b128.replace("_arb128x4", "") == serialize(build_tiled_block_scaled_gemm(other, arch="gfx950"))
    assert not is_valid_spec(replace(spec, as_read="b64x2"))[0]


def test_b_layout_lanes_reads_b_halves_lane_linear():
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half",
                   a_scale_path="lds2")
    spec = replace(base, b_layout="lanes")
    assert is_valid_spec(spec) == (True, "ok") and "_Lrl_" in spec.kernel_name() and spec.b_split and spec.b_lanes
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    # The same memory ops and DMA (a stage half of the tile is still one
    # contiguous block); only the B read addresses change.
    for op in ("tile.mma", "async_buffer_load_lds", "smem_load_vN", "global_load", "tile.perm_b32"):
        assert text.count(op) == plain.count(op), op
    assert text != plain.replace("_Lrk_", "_Lrl_")
    assert _check_split_barrier_waits(spec) == _check_split_barrier_waits(base)
    for changes in ({"K": 1408}, {"K": 1152}, {"K": 512}, {"k_peel": True, "K": 2560}):
        small = replace(spec, M=512, N=512, xcd_remap="none", **changes)
        assert is_valid_spec(small)[0]
        _check_split_barrier_waits(small)
        _lower(small)
    _lower(spec)
    # Without the half release it reads as "kblock": the name changes only.
    for changes in ({"b_release": "full"}, {"frag_pipeline": "stage"}):
        other = replace(base, **changes)
        lanes = replace(other, b_layout="lanes")
        assert is_valid_spec(lanes)[0] and not lanes.b_lanes
        assert serialize(build_tiled_block_scaled_gemm(lanes, arch="gfx950")).replace("_Lrl_", "_Lrk_") == serialize(
            build_tiled_block_scaled_gemm(other, arch="gfx950"))
    assert not is_valid_spec(replace(spec, a_layout="lanes"))[0]


def test_half_lanes_packs_each_lanes_pieces_where_it_reads_them():
    import numpy as np
    from rocke.examples.gfx950.gemm.mxfp6_tiled.bench import half_lanes, split_halves

    n, kt = 64, 2
    b = np.random.default_rng(0).integers(0, 256, (n, kt * 96), dtype=np.uint8)
    kb = np.ascontiguousarray(b.reshape(n, kt, 96).transpose(1, 0, 2))
    dev = half_lanes(split_halves(kb)).reshape(-1)
    assert dev.size == b.size
    for t in range(kt):
        for col in range(n):
            for h in range(2):
                for s, blk in enumerate((2 * h, 3 - 2 * h)):
                    lane, base = 32 * h + col % 32, t * n * 96 + s * n * 48 + col // 32 * 1536
                    got = np.concatenate([dev[base + 16 * lane : base + 16 * lane + 16],
                                          dev[base + 1024 + 8 * lane : base + 1024 + 8 * lane + 8]])
                    assert (got == b[col, 96 * t + 24 * blk : 96 * t + 24 * blk + 24]).all()


def test_k_order_t10_reads_each_lanes_a_bytes_in_order():
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half",
                   a_scale_path="lds2")
    spec = replace(base, k_order="t10")
    assert is_valid_spec(spec) == (True, "ok") and "_kot10_" in spec.kernel_name() and spec.k_t10
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))

    def body(t: str) -> str:
        return t.split("region @body {", 1)[1].split("scf.yield", 1)[0]

    def reads(t: str, buf: str) -> list[str]:
        return re.findall(rf"smem_load_vN \( %{buf}\w*, .*vec = i:(\d)", body(t))

    # Per eight-stage body and A row block: step 0 two 16-byte A reads
    # instead of 16 + 8 bytes; the A scales a halfword per stage, with no
    # rotation.
    assert sorted(reads(plain, "a_lines")) == ["2"] * 64 + ["4"] * 64
    assert sorted(reads(text, "a_lines")) == ["2"] * 32 + ["4"] * 96
    assert reads(text, "a_scales") == reads(plain, "a_scales") == ["1"] * 32
    assert reads(text, "b_lds") == reads(plain, "b_lds")
    assert re.findall(r"smem_load_vN \( %a_scales\w*, .*", body(text))[0].count("f16")
    assert body(plain).count("tile.perm_b32") == 32 and body(text).count("tile.perm_b32") == 0
    for op in ("tile.mma", "async_buffer_load_lds", "global_load"):
        assert body(text).count(op) == body(plain).count(op), op
    assert _check_split_barrier_waits(spec) == _check_split_barrier_waits(base)
    for changes in ({"K": 1408}, {"K": 1152}, {"K": 512}, {"K": 2560, "k_peel": True}, {"K": 1152, "mfma_group": 4},
                    {"K": 1408, "as_read": "b128x4"}, {"K": 1152, "b_layout": "lanes"},
                    {"K": 1408, "dma_place": "even"}):
        small = replace(spec, M=512, N=512, xcd_remap="none", **changes)
        assert is_valid_spec(small)[0] and small.k_t10
        _check_split_barrier_waits(small)
        _lower(small)
    _lower(spec)
    # Elsewhere the value changes the name only.
    for other in (replace(base, b_release="full"), replace(base, frag_pipeline="stage"),
                  replace(base, scale_opsel=False), replace(base, a_scale_path="direct"), TARGET):
        t10 = replace(other, k_order="t10")
        assert is_valid_spec(t10)[0] and not t10.k_t10
        assert serialize(build_tiled_block_scaled_gemm(t10, arch="gfx950")).replace("_kot10", "") == serialize(
            build_tiled_block_scaled_gemm(other, arch="gfx950"))
    assert not is_valid_spec(replace(spec, k_order="t01"))[0]


def test_t10_packs_lane_half_h_step_s_block_2h_plus_s():
    import numpy as np
    from rocke.examples.gfx950.gemm.mxfp6_tiled.bench import (
        T10_BLOCKS,
        scales_kblock4,
        split_halves,
    )

    rng = np.random.default_rng(0)
    n, kt, wave_n = 128, 2, 64
    b = rng.integers(0, 256, (kt, n, 96), dtype=np.uint8)
    halves = split_halves(b, T10_BLOCKS)
    for s in range(2):
        x, y = b[:, :, 24 * s : 24 * s + 24], b[:, :, 24 * (2 + s) : 24 * (2 + s) + 24]
        want = np.concatenate([x[..., :16], x[..., 16:], y[..., 16:], y[..., :16]], axis=2)
        assert (halves[:, s] == want).all()
    sb = rng.integers(0, 256, (4 * kt, n), dtype=np.uint8)
    dev = scales_kblock4(sb, wave_n, T10_BLOCKS)  # [kt][g][h][r][s][j]
    for t, g, h, r, s, j in itertools.product(range(kt), range(n // wave_n), range(2), range(32), range(2), range(4)):
        want = sb[4 * t + 2 * h + s, wave_n * g + 32 * j + r] if j < wave_n // 32 else 0
        assert dev[t, g, h, r, s, j] == want


def _check_l2r_barrier_waits(spec) -> list[int]:
    """`_check_barrier_waits` for `a_ring="line2r"`: B's n-th DMA carries
    B(n // passes) (B(0) before Bar(0), B(1) after it, then stage t's step 0
    B(t+2)); every A DMA is a refill. At Bar(t), after which stage t reads,
    no refill may be in flight and of B only B(t+1). One barrier per stage.
    Returns the loop body's waits."""
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    pre, body, post = _kernel_ops(text)
    passes = spec.block_n * 6 // spec.threads
    in_flight, n_b, n_bar, n_z = [], 0, 0, 0
    for op in pre + body * _loop_trips(text) + post:
        if op.startswith("W"):
            n = int(op[1:])
            if n >= 0:
                in_flight = in_flight[max(0, len(in_flight) - n) :]
        elif op == "|":
            for kind, at in in_flight:
                if kind == "Z":
                    _check_scale_dma(n_bar, at)
                else:
                    assert kind != "A" and (kind != "B" or at == n_bar + 1), (n_bar, in_flight)
            n_bar += 1
        elif op == "B":
            in_flight.append(("B", n_b // passes))
            n_b += 1
        elif op in "AGsV":
            in_flight.append((op, n_bar))
        elif op == "Z":
            in_flight.append((op, n_z // 2))
            n_z += 1
    assert n_bar == spec.K // 128 + 1
    return [int(op[1:]) for op in body if op.startswith("W") and int(op[1:]) >= 0]


def test_a_ring_line2r_keeps_two_line_slots_a_remnant_and_three_b_slots():
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half",
                   a_scale_path="lds2")
    spec = replace(base, a_ring="line2r")
    assert is_valid_spec(spec) == (True, "ok") and spec.a_line2r and not spec.b_split
    assert (spec.stages, spec.b_release, spec.unroll) == (3, "full", 4)
    assert "_s3_" in spec.kernel_name() and "_Aline2r" in spec.kernel_name() and "_br" not in spec.kernel_name()
    # Two line slots and R, three B slots and lds2's A scales fill LDS exactly.
    assert spec.lds_bytes == 2 * 256 * 128 + 256 * 32 + 3 * 256 * 96 + 2 * 256 * 32 == 160 * 1024
    # Every fragment piece (row bytes 96u + offset, by lane half) lies in one
    # chunk of one source: line 3n + j in slot (n + j) % 2, except line 3n's
    # last 32 bytes, which come from R and which phase 1 alone reads.
    offsets = {(0, "b128"): (0, 48), (0, "b64"): (16, 64), (1, "b64"): (72, 24), (1, "b128"): (80, 32)}
    from_r = set()
    for q in range(8):
        for (step, piece), by_half in offsets.items():
            for h in (0, 1):
                line, at = divmod(96 * (q % 4) + by_half[h], 128)
                src, chunk, byte = _l2r_source(q, step, piece, h)
                assert byte + (16 if piece == "b128" else 8) <= 16
                if src == "R":
                    from_r.add((q, step, piece, h))
                    assert (line, chunk, byte) == (0, (at - 96) // 16, at % 16)
                else:
                    assert (src, chunk, byte) == ((q // 4 + line) % 2, at // 16, at % 16)
    assert from_r == {(q, *k) for q in (1, 5) for k in ((0, "b128", 0), (0, "b64", 0), (1, "b64", 1))}
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    assert "lds_dma_explicit" in text and "b_lds" not in text and text.count("tile.smem_alloc") == 2
    body = text.split("region @body {", 1)[1].split("scf.yield", 1)[0]
    # A 24-stage body (B's three slots, the slots' parity flips every window):
    # per stage 32 MFMAs, B's six DMAs and one barrier; per window three line
    # refills of eight DMAs and one remnant copy (two LDS reads and two writes
    # per thread); lds2's two scale DMAs per eight stages.
    assert body.count("tile.mma") == 24 * 32 and body.count("smem_store_vN") == 6 * 2
    _, ops, _ = _kernel_ops(text)
    assert (ops.count("B"), ops.count("A"), ops.count("Z"), ops.count("|")) == (24 * 6, 6 * 3 * 8, 3 * 2, 24)
    parts = "".join(ops).split("|")

    def cut(part: str) -> int:
        """Where a stage's step 1 ends in `part` (after its 16 MFMAs)."""
        mfmas = [i for i, c in enumerate(part) if c == "M"]
        return mfmas[16] if len(mfmas) > 16 else len(part)

    for s in range(24):
        step0 = parts[s][cut(parts[s]) :] if s else parts[0]
        step1 = parts[s + 1][: cut(parts[s + 1])]
        # Step 0: B two stages ahead after MFMAs 0, 2, ..., 10; step 1: the refill after MFMAs 0..7.
        assert [step0[:i].count("M") for i, c in enumerate(step0) if c == "B"] == [1, 3, 5, 7, 9, 11], s
        refill = [step1[:i].count("M") for i, c in enumerate(step1) if c == "A"]
        assert refill == ([] if s % 4 == 1 else list(range(1, 9))), s
        assert "B" not in step1 and "A" not in step0
    waits = _check_l2r_barrier_waits(spec)
    assert len(waits) == 24 and max(waits) <= 9
    for changes in ({"K": 1152}, {"K": 1408}, {"K": 640}, {"K": 3200}, {"K": 6528}, {"K": 6528, "k_peel": True},
                    {"K": 3200, "a_scale_path": "direct"}, {"K": 1408, "a_scale_path": "direct", "a_scale_fetch": "iter"},
                    {"K": 1152, "a_scale_path": "direct", "a_scale_fetch": "stage", "b_scale_layout": "row"},
                    {"K": 3200, "scale_opsel": False}, {"K": 1408, "mfma_group": 4}, {"K": 1408, "lds_swizzle": "none"},
                    {"K": 256, "a_scale_path": "direct", "a_scale_fetch": "stage"}):
        small = replace(spec, M=512, N=512, xcd_remap="none", **changes)
        assert is_valid_spec(small)[0], changes
        _check_l2r_barrier_waits(small)
        _lower(small)
    _lower(spec)
    for ablate in ("no_dma", "no_mfma", "no_scales", "no_lds_read"):
        _lower(replace(spec, ablate=ablate))
    # Elsewhere the value is the three-line ring: the name changes only.
    for changes in ({"frag_pipeline": "stage"}, {"backend": "llvm", "mfma_form": "asm_agpr", "agpr_alloc": "256,256"}):
        other = replace(base, **changes)
        two = replace(other, a_ring="line2r")
        assert not two.a_line2r and two.stages == 2
        assert serialize(build_tiled_block_scaled_gemm(two, arch="gfx950")).replace("_Aline2r", "_Aline") == serialize(
            build_tiled_block_scaled_gemm(other, arch="gfx950"))


@pytest.mark.parametrize("a_scale_path", ["direct", "lds2"])
def test_b_path_vgpr_streams_b_into_registers(a_scale_path):
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half",
                   a_scale_path=a_scale_path)
    spec = replace(base, b_path="vgpr")
    lds2 = a_scale_path == "lds2"
    assert is_valid_spec(spec) == (True, "ok") and "_w1x4_" in spec.kernel_name() and "_bpvgpr" in spec.kernel_name()
    assert spec.b_vgpr and not spec.b_split and (spec.wave_tile_m, spec.wave_tile_n) == (256, 64)
    assert spec.lds_bytes == 3 * 256 * 128 + (2 * 256 * 32 if lds2 else 0)
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))

    def body(t: str) -> str:
        return t.split("region @body {", 1)[1].split("scf.yield", 1)[0]

    # An eight-stage body either way. No B DMA (six per stage before) and no
    # B in LDS: per stage eight loads into registers (two column blocks, two
    # steps, 16 and 8 bytes), the wave's eight A fragments of both steps from
    # LDS, and one B scale dword.
    assert body(text).count("tile.mma") == body(plain).count("tile.mma") == 8 * 32
    assert body(text).count("async_buffer_load_lds") == body(plain).count("async_buffer_load_lds") - 8 * 6
    assert body(text).count("tile.buffer_load_vN") == 8 * 8 and "tile.buffer_load_vN" not in body(plain)
    assert "b_lds" not in text
    assert body(text).count("smem_load_vN") == 8 * (8 * 2 * 2 + (8 if lds2 else 0))
    b_loads = [ln for ln in body(text).splitlines() if "global_load" in ln and "%B_scale" in ln]
    assert len(b_loads) == 8 and all("global_load_typed" in ln for ln in b_loads)
    # Column j's MFMA of step s selects byte 2s + j of the stage's B scale dword.
    sels = re.findall(r"tile\.mma .*scale_sel = l:\[ i:(\d), i:(\d) \]", body(text))
    assert sels == ([("0", str(j)) for _ in range(8) for j in range(2)]
                    + [("3", str(2 + j)) for _ in range(8) for j in range(2)]) * 8
    # The barrier waits for A's refills only; the B loads stay in flight.
    waits = _check_barrier_waits(spec)
    assert waits == [21, 15 if lds2 else 21, 32 if lds2 else 38, 21, 21, 13, 30, 21]
    for changes in ({"K": 1408}, {"K": 1152}, {"K": 512}, {"K": 2560}, {"K": 2560, "k_peel": True},
                    {"K": 3072, "k_peel": True}, {"K": 1024}, {"K": 640}, {"K": 1408, "scale_opsel": False},
                    {"K": 1152, "a_scale_fetch": "stage"}):
        small = replace(spec, M=512, N=512, xcd_remap="none", **changes)
        assert is_valid_spec(small)[0]
        _check_barrier_waits(small)
        _lower(small)
    _lower(spec)
    for ablate in ("no_dma", "no_mfma", "no_scales"):
        _lower(replace(spec, ablate=ablate))
    # Elsewhere the value changes the name only.
    for changes in ({"frag_pipeline": "stage"}, {"backend": "llvm", "mfma_form": "asm_agpr", "agpr_alloc": "256,256"}):
        other = replace(base, **changes)
        vgpr = replace(other, b_path="vgpr")
        assert not vgpr.b_vgpr and (vgpr.waves_m, vgpr.waves_n) == (2, 2)
        assert serialize(build_tiled_block_scaled_gemm(vgpr, arch="gfx950")).replace("_bpvgpr", "") == serialize(
            build_tiled_block_scaled_gemm(other, arch="gfx950"))
    assert not is_valid_spec(replace(spec, b_path="global"))[0]


def test_dma_m0_early_sets_the_kernel_attribute_on_the_direct_backend():
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2"},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half")
    spec = replace(base, dma_m0="early")
    assert is_valid_spec(spec) == (True, "ok") and "_m0early" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    assert "m0_early = b:true" in text and "m0_early" not in plain
    assert text.replace("_m0early", "").replace(", m0_early = b:true", "") == plain
    _lower(spec)
    llvm = replace(base, backend="llvm", mfma_form="asm_agpr", agpr_alloc="256,256")
    assert serialize(build_tiled_block_scaled_gemm(replace(llvm, dma_m0="early"), arch="gfx950")).replace(
        "_m0early", "") == serialize(build_tiled_block_scaled_gemm(llvm, arch="gfx950"))
    assert not is_valid_spec(replace(spec, dma_m0="late"))[0]


@pytest.mark.parametrize("cc_passes", ["dead_lanes", "m0_fill+peep_waits", "all"])
def test_cc_passes_set_only_their_kernel_attributes_on_the_direct_backend(cc_passes):
    base = replace(TARGET, **_PLACED)
    spec = replace(base, cc_passes=cc_passes)
    assert is_valid_spec(spec) == (True, "ok") and f"_cc{cc_passes.replace('+', '_')}" in spec.kernel_name()
    names = CC_PASSES if cc_passes == "all" else cc_passes.split("+")
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    stripped = text.replace(f"_cc{cc_passes.replace('+', '_')}", "")
    for p in names:
        assert f"{p} = b:true" in text and p not in plain
        stripped = stripped.replace(f"{p} = b:true, ", "").replace(f", {p} = b:true", "")
    assert stripped == plain
    _lower(spec)
    llvm = replace(base, backend="llvm", mfma_form="asm_agpr", agpr_alloc="256,256")
    assert serialize(build_tiled_block_scaled_gemm(replace(llvm, cc_passes=cc_passes), arch="gfx950")).replace(
        f"_cc{cc_passes.replace('+', '_')}", "") == serialize(build_tiled_block_scaled_gemm(llvm, arch="gfx950"))
    for bad in ("m0_early", "dead_lanes+dead_lanes", "all+peep_waits", ""):
        assert not is_valid_spec(replace(spec, cc_passes=bad))[0]


@pytest.mark.parametrize(("cc_extra", "token"), [("lds_waw+u16_zext", "_cxwz"), ("pin_vmem", "_cxp"), ("all", "_cxall")])
def test_cc_extra_sets_only_its_kernel_attributes_on_the_direct_backend(cc_extra, token):
    base = replace(TARGET, **_PLACED, cc_passes="all")
    spec = replace(base, cc_extra=cc_extra)
    assert is_valid_spec(spec) == (True, "ok") and token in spec.kernel_name()
    names = CC_EXTRAS if cc_extra == "all" else cc_extra.split("+")
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    stripped = text.replace(token, "")
    for p in names:
        assert f"{p} = b:true" in text and p not in plain
        stripped = stripped.replace(f"{p} = b:true, ", "").replace(f", {p} = b:true", "")
    assert stripped == plain
    _lower(spec)
    llvm = replace(base, backend="llvm", mfma_form="asm_agpr", agpr_alloc="256,256")
    assert serialize(build_tiled_block_scaled_gemm(replace(llvm, cc_extra=cc_extra), arch="gfx950")).replace(
        token, "") == serialize(build_tiled_block_scaled_gemm(llvm, arch="gfx950"))
    for bad in ("dead_lanes", "pin_vmem+pin_vmem", "all+m0_cse", ""):
        assert not is_valid_spec(replace(spec, cc_extra=bad))[0]


_DMA = re.compile(r"tile\.async_buffer_load_lds_addr \( %(\w+), %(\w+), %(\w+), %(\w+) \) \{ ([^}]*) \}")


@pytest.mark.parametrize(("dma_m0", "token"), [("bshare", "_m0bs"), ("early_bshare", "_m0ebs")])
def test_dma_m0_bshare_gives_a_b_halfs_dmas_one_lds_base_and_voffset(dma_m0, token):
    base = replace(TARGET, **_PLACED, dma_m0=dma_m0.replace("bshare", "").rstrip("_") or "default")
    spec = replace(base, dma_m0=dma_m0)
    assert spec.b_share and is_valid_spec(spec) == (True, "ok") and token in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    assert "m0_cse = b:true" in text and "m0_cse" not in plain
    assert ("m0_early = b:true" in text) == (dma_m0 == "early_bshare")
    dmas, dmas0 = _DMA.findall(text), _DMA.findall(plain)
    assert len(dmas) == len(dmas0)
    # The same op sequence; only B's DMAs change (three per half).
    assert _kernel_ops(text) == _kernel_ops(plain)
    b = [d for d in dmas if "offset" in d[4]]
    assert sorted({d[4] for d in b}) == ["aux = i:0, dwords = i:4, offset = i:1024",
                                         "aux = i:0, dwords = i:4, offset = i:2048"]
    # Each B half's three DMAs: one LDS base, one voffset, offsets 0, 1024, 2048.
    for i, d in enumerate(dmas):
        if d[4].endswith("offset = i:1024"):
            first, second = dmas[i - 1], dmas[i + 1]
            assert "offset" not in first[4] and second[4].endswith("offset = i:2048")
            assert first[:4] == d[:4] == second[:4]
    voffs = {d[2] for d in dmas if "offset" in d[4]}
    assert len(voffs) == 1
    assert len({d[1] for d in b}) == 4  # two slots x two halves
    _check_split_barrier_waits(spec)
    _lower(spec)
    for changes in ({"K": 1152, "M": 512, "N": 512, "xcd_remap": "none"}, {"K": 128 * 13}):
        small = replace(spec, **changes)
        assert small.b_share and is_valid_spec(small)[0]
        _check_split_barrier_waits(small)
        _lower(small)
    # Elsewhere the name only.
    for changes in ({"b_release": "full"}, {"backend": "llvm", "mfma_form": "asm_agpr", "agpr_alloc": "256,256"}):
        other = replace(base, **changes)
        assert not replace(other, dma_m0=dma_m0).b_share
        assert serialize(build_tiled_block_scaled_gemm(replace(other, dma_m0=dma_m0), arch="gfx950")).replace(
            token, f"_m0{base.dma_m0}" if base.dma_m0 != "default" else "") == serialize(
            build_tiled_block_scaled_gemm(other, arch="gfx950"))


def test_direct_kernel_names_fit_the_compilers_file_name():
    """The direct compiler writes `<kernel name>.hsaco`: a longer name is refused, not truncated by the OS."""
    long = replace(TARGET, **{**_PLACED, "b_layout": "lanes"}, cc_passes="all", k_order="t10", as_read="b128x4",
                   sync_place="split_sink", dma_m0="early_bshare", cc_extra="lds_waw+u16_zext+pin_vmem")
    assert len(long.kernel_name()) + len(".hsaco") <= 255 and is_valid_spec(long)[0]
    ok, why = is_valid_spec(replace(long, cc_passes="dead_lanes+m0_fill+peep_waits+peep_shifts+ds_rebase"))
    assert not ok and "kernel name" in why
    llvm = replace(long, backend="llvm", mfma_form="asm_agpr", agpr_alloc="256,256",
                   cc_passes="dead_lanes+m0_fill+peep_waits+peep_shifts+ds_rebase")
    assert is_valid_spec(llvm)[0]


def test_acc_init_early_zeroes_the_accumulators_before_the_prologue_wait():
    base = replace(TARGET, **_PLACED, cc_passes="all")
    spec = replace(base, acc_init="early")
    assert is_valid_spec(spec) == (True, "ok") and "_aiearly" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    assert "acc_init_keep = b:true" in text and "acc_init_keep" not in plain

    def prologue(t: str) -> list[str]:
        head = t.split("scf.for", 1)[0].splitlines()
        return [("Z" if "zero_vec" in ln or "constant_vec" in ln else "W" if "s_waitcnt" in ln else "D")
                for ln in head if "zero_vec" in ln or "constant_vec" in ln or "s_waitcnt" in ln
                or "async_buffer_load_lds" in ln]

    pre, pre0 = prologue(text), prologue(plain)
    assert sorted(pre) == sorted(pre0) and "Z" in pre
    first_wait = pre.index("W")
    assert max(i for i, p in enumerate(pre) if p == "Z") < first_wait
    assert min(i for i, p in enumerate(pre0) if p == "Z") > pre0.index("W")
    assert text.count("tile.mma") == plain.count("tile.mma")
    _check_split_barrier_waits(spec)
    _lower(spec)
    # Elsewhere the name only.
    for changes in ({"a_ring": "stage", "frag_pipeline": "stage", "b_release": "full", "b_scale_fetch": "stage",
                     "dma_place": "default", "a_scale_path": "direct", "b_layout": "row", "b_scale_layout": "row",
                     "a_scale_fetch": "stage", "fill_split": "none"},
                    {"backend": "llvm", "mfma_form": "asm_agpr", "agpr_alloc": "256,256"}):
        other = replace(base, **changes)
        assert is_valid_spec(replace(other, acc_init="early"))[0]
        assert serialize(build_tiled_block_scaled_gemm(replace(other, acc_init="early"), arch="gfx950")).replace(
            "_aiearly", "") == serialize(build_tiled_block_scaled_gemm(other, arch="gfx950"))
    assert not is_valid_spec(replace(spec, acc_init="late"))[0]


_B_SCALE_LOAD = re.compile(r"vec<i32x(\d)> = memref\.global_load_vN \( %B_scale")


@pytest.mark.parametrize("K", [65536, 128 * 13, 128 * 8])
def test_b_scale_fetch_pair_loads_two_stages_per_four_dword_load(K):
    base = replace(TARGET, **_PLACED, K=K)
    spec = replace(base, b_scale_fetch="pair")
    assert spec.b_scale_pairs and is_valid_spec(spec) == (True, "ok") and "_bspair" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    pre, body, post = _kernel_ops(text)
    pre0, body0, post0 = _kernel_ops(plain)
    # Only the odd stages load (the last one none).
    for ops, ops0 in ((pre, pre0), (body, body0), (post, post0)):
        assert ops.count("s") == ops0.count("s") // 2 + (ops is pre)
    assert _loop_trips(text) == _loop_trips(plain)

    def shape(ops):
        return [op[0] for op in ops if op != "s"]

    assert (shape(pre), shape(body), shape(post)) == (shape(pre0), shape(body0), shape(post0))
    assert set(_B_SCALE_LOAD.findall(text)) == {"4"} and set(_B_SCALE_LOAD.findall(plain)) == {"2"}
    # Each stage's MFMAs take dwords 0, 1 (even stage) or 2, 3 (odd) of the pair.

    def b_dwords(text: str) -> list[int]:
        vecs = set(re.findall(r"%(\w+) : vec<i32x\d> = memref\.global_load_vN \( %B_scale", text)) | {"sc0"}
        return [int(i) for v, i in re.findall(r"vector\.extract \( %(\w+) \) \{ index = i:(\d+) \}", text)
                if v in vecs]

    got, want = b_dwords(text), b_dwords(plain)
    assert set(want) == {0, 1} and set(got) == {0, 1, 2, 3}
    assert text.count("tile.mma") == plain.count("tile.mma")
    _check_split_barrier_waits(spec)
    _lower(spec)
    other = replace(TARGET, **_CANONICAL_A)
    assert not other.b_scale_pairs
    assert serialize(build_tiled_block_scaled_gemm(replace(other, b_scale_fetch="pair"), arch="gfx950")).replace(
        "_bspair", "") == serialize(build_tiled_block_scaled_gemm(other, arch="gfx950"))
    assert not is_valid_spec(replace(spec, b_scale_fetch="iter2"))[0]


def test_b_scale_pairs_pack_stages_2v_and_2v_plus_1_per_lane():
    import numpy as np
    from rocke.examples.gfx950.gemm.mxfp6_tiled.bench import (
        scales_kblock4,
        scales_pairs,
    )

    rng = np.random.default_rng(1)
    for kt in (4, 5):
        dev = scales_kblock4(rng.integers(0, 256, (4 * kt, 128), dtype=np.uint8), 64)  # [kt][g][h][r][s][j]
        pairs = scales_pairs(dev)  # [kt/2][g][h][r][2][s][j]
        assert pairs.shape == ((kt + 1) // 2, *dev.shape[1:4], 2, *dev.shape[4:])
        for t in range(2 * pairs.shape[0]):
            want = dev[t] if t < kt else 0
            assert (pairs[t // 2, :, :, :, t % 2] == want).all()


@pytest.mark.parametrize("mfma_group", [1, 4, 16])
def test_mfma_order_serp_walks_odd_rows_backward(mfma_group):
    base = replace(TARGET, **{**_CANONICAL_A, "b_scale_layout": "kblock4", "a_scale_fetch": "iter2",
                              "mfma_group": mfma_group},
                   frag_pipeline="step", fill_split="line1", scale_opsel=True, setprio=True, b_release="half",
                   a_scale_path="lds2")
    spec = replace(base, mfma_order="serp")
    assert is_valid_spec(spec) == (True, "ok") and "_moserp" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    mmas = re.findall(r"tile\.mma \( (%\w+), (%\w+), (%\w+),.*scale_sel = l:\[ i:\d, i:(\d) \]", text)
    assert len(mmas) == len(re.findall(r"tile\.mma ", plain)) and len(mmas) % 16 == 0
    serp = [0, 1, 2, 3, 3, 2, 1, 0] * 2
    for x in range(0, len(mmas), 16):
        step = mmas[x : x + 16]
        assert [int(j) for *_, j in step] == serp
        assert len({(a, b) for a, b, *_ in step}) == 16
        # Inside a step every MFMA after the first changes one operand.
        assert all((p[0] != q[0]) + (p[1] != q[1]) == 1 for p, q in itertools.pairwise(step))
    rows = re.findall(r"tile\.mma .*scale_sel = l:\[ i:\d, i:(\d) \]", plain)
    assert [int(j) for j in rows] == [0, 1, 2, 3] * (len(rows) // 4)
    # Same instructions; only the MFMA order (and so the names) changes.
    assert sorted(re.findall(r"= ([\w.]+) ", text)) == sorted(re.findall(r"= ([\w.]+) ", plain))
    _lower(spec)
    small = replace(spec, M=512, N=512, K=1152, xcd_remap="none")
    assert is_valid_spec(small)[0]
    _check_split_barrier_waits(small)
    _lower(replace(TARGET, mfma_order="serp", mfma_group=4))
    assert not is_valid_spec(replace(spec, mfma_order="cols"))[0]


@pytest.mark.parametrize("scale_opsel", [False, True])
def test_b_scale_kblock4_loads_once_per_stage_and_selects_byte_j(scale_opsel):
    base = replace(TARGET, **_CANONICAL_A, frag_pipeline="step", fill_split="line1", scale_opsel=scale_opsel,
                   setprio=True)
    spec = replace(base, b_scale_layout="kblock4")
    assert is_valid_spec(spec) == (True, "ok") and "_Srk4_" in spec.kernel_name()
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))

    def body(t: str) -> str:
        return t.split("region @body {", 1)[1].split("scf.yield", 1)[0]

    # One two-dword load per stage instead of a dword per column.
    b_loads = [ln for ln in body(text).splitlines() if "global_load" in ln and "%B_scale" in ln]
    assert len(b_loads) == 4 and all("global_load_vN" in ln and "vec = i:2" in ln for ln in b_loads)
    assert len([ln for ln in body(plain).splitlines() if "global_load" in ln and "%B_scale" in ln]) == 4 * 4
    for op in ("tile.mma", "async_buffer_load_lds", "smem_load_vN"):
        assert body(text).count(op) == body(plain).count(op), op
    if scale_opsel:
        # B's dwords need no rotation; column j's MFMA selects byte j.
        assert body(plain).count("tile.perm_b32") - body(text).count("tile.perm_b32") == 4 * 4
        sels = re.findall(r"tile\.mma .*scale_sel = l:\[ i:(\d), i:(\d) \]", body(text))
        assert sels == ([(a, str(j)) for a in "0" for _ in range(4) for j in range(4)]
                        + [(a, str(j)) for a in "3" for _ in range(4) for j in range(4)]) * 4
    # Three fewer scale loads per stage stay in flight at every barrier.
    assert _check_barrier_waits(base) == [12, 8, 12, 12] and _check_barrier_waits(spec) == [9, 5, 9, 9]
    for changes in ({"M": 512, "N": 512, "K": 1408}, {"M": 512, "N": 512, "K": 1152, "a_scale_fetch": "iter2"},
                    {"frag_pipeline": "stage"}, {"a_ring": "stage", "stages": 3, "frag_pipeline": "stage"},
                    {"b_release": "half"}, {"ablate": "no_scales"}):
        small = replace(spec, **changes, **({"xcd_remap": "none"} if "M" in changes else {}))
        assert is_valid_spec(small)[0]
        if small.a_ring == "line" and small.ablate == "none":
            (_check_split_barrier_waits if small.b_split else _check_barrier_waits)(small)
        _lower(small)
    # The asm form writes byte j into op_sel / op_sel_hi.
    asm = replace(TARGET, scale_opsel=True, b_scale_layout="kblock4", a_scale_fetch="iter", mfma_group=4)
    llvm = _lower(asm)
    for j in range(4):
        assert f"op_sel:[1,{j & 1},0] op_sel_hi:[1,{j >> 1},0]" in llvm
    assert not is_valid_spec(replace(spec, b_scale_layout="kblock5"))[0]


@pytest.mark.parametrize("pipeline,ring", [("stage", "line"), ("step", "line"), ("stage", "stage")])
def test_k_peel_takes_the_clamps_out_of_the_loop(pipeline, ring):
    changes = {**_CANONICAL_A, "fill_split": "line1", "scale_opsel": True, "frag_pipeline": pipeline}
    if ring == "stage":
        changes.update(a_ring="stage", stages=3)
    base = replace(TARGET, **changes)
    spec = replace(base, k_peel=True)
    assert is_valid_spec(spec) == (True, "ok") and spec.kernel_name().endswith("_kp_direct")
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))

    def body(t: str) -> str:
        return t.split("region @body {", 1)[1].split("scf.yield", 1)[0]

    # Every end-of-K clamp of the loop body is a select; none stays.
    assert body(plain).count("arith.select") > 0 and body(text).count("arith.select") == 0
    # The same work, one iteration of it peeled after the loop. With no tail
    # stage after it, the peeled iteration loads no next A scale group (and the
    # step pipeline's last stage no next-stage scales): the compiler would drop
    # those dead loads, and the barrier waits count them.
    tail = spec.K // 128 % spec.unroll
    dropped = 0 if tail else 4 + (4 if pipeline == "step" else 0)
    for op in ("tile.mma", "async_buffer_load_lds", "smem_load_vN", "global_load"):
        assert text.count(op) == plain.count(op) + body(text).count(op) - (dropped if op == "global_load" else 0), op
    assert ("global_load_vN" in text.split("scf.yield", 1)[1]) == bool(tail)
    assert _loop_trips(text) == _loop_trips(plain) - 1
    _check_barrier_waits(spec)
    for k in (1408, 1152, 512):
        small = replace(spec, M=512, N=512, K=k, xcd_remap="none")
        if is_valid_spec(small)[0]:
            _check_barrier_waits(small)
            _lower(small)


def test_small_shapes_and_tails():
    for changes in (
        {"M": 512, "N": 1024, "K": 1152, "a_ring": "line", "b_layout": "kblock", "b_scale_layout": "kblock"},
        {"M": 512, "N": 512, "K": 1408, "a_ring": "line", "a_scale_fetch": "iter", "mfma_form": "intrinsic",
         "agpr_alloc": "auto", "backend": "direct", "mfma_group": 1},
        {"M": 512, "N": 512, "K": 512, "a_ring": "line", "a_scale_fetch": "iter"},
        {"M": 512, "N": 512, "K": 384, "a_ring": "line"},
        {"M": 512, "N": 512, "K": 1152, "a_ring": "line", "ablate": "no_dma+no_scales"},
        {"M": 512, "N": 512, "K": 1152, "stages": 3, "a_scale_fetch": "iter"},
        {"M": 512, "N": 1024, "K": 1408, "stages": 3, "a_scale_fetch": "iter", "b_layout": "kblock",
         "b_scale_layout": "kblock"},
        {"M": 512, "N": 512, "K": 384, "stages": 3, "a_scale_fetch": "iter", "ablate": "no_scales"},
        {"M": 512, "N": 1024, "K": 1408, "stages": 3, "a_scale_fetch": "iter", "mfma_group": 4,
         "mfma_form": "intrinsic", "agpr_alloc": "auto", "backend": "direct"},
        {"M": 512, "N": 1024, "K": 1408, "stages": 3, "frag_pipeline": "step", "mfma_group": 4,
         "mfma_form": "intrinsic", "agpr_alloc": "auto", "backend": "direct"},
        {"M": 512, "N": 512, "K": 1408, "a_ring": "line", "a_scale_fetch": "iter", "mfma_form": "intrinsic",
         "agpr_alloc": "auto", "backend": "direct", "mfma_group": 2, "fill_split": "line1"},
        {"M": 512, "N": 512, "K": 1152, "a_ring": "line", "a_scale_fetch": "iter", "mfma_group": 4,
         "fill_split": "line1", "ablate": "no_dma+no_scales"},
        {"M": 512, "N": 1024, "K": 1408, "stages": 3, "a_scale_fetch": "iter", "mfma_group": 4,
         "fill_split": "line1"},
        {"M": 512, "N": 512, "K": 1152},
        {"M": 512, "N": 512, "K": 1152, "stages": 3},
        {"M": 512, "N": 512, "K": 1152, "stages": 3, "frag_pipeline": "step"},
        {"M": 512, "N": 512, "K": 128, "frag_pipeline": "step"},
        {"M": 512, "N": 1024, "K": 1152, "a_layout": "kblock"},
        {"M": 512, "N": 1024, "K": 1152, "a_layout": "kblock", "b_layout": "kblock", "stages": 3},
        {"M": 512, "N": 1024, "K": 1152, "a_scale_layout": "kblock", "b_scale_layout": "kblock"},
        {"M": 512, "N": 1024, "K": 1152, "b_scale_layout": "kblock", "b_layout": "kblock"},
        {"M": 512, "N": 1024, "K": 1152, "a_scale_layout": "kblock", "b_scale_layout": "kblock",
         "frag_pipeline": "step", "stages": 3},
    ):
        spec = replace(TARGET, **changes)
        assert is_valid_spec(spec)[0]
        _lower(spec)


def test_tuner_never_accepts_identical_code(tmp_path, monkeypatch):
    """A move that times faster but compiles to the incumbent's code is noise."""
    args = SimpleNamespace(
        ledger=str(tmp_path / "ledger"), cache=str(tmp_path / "cache"), shape="4096,4096,65536",
        data="random", start=None, max_evals=200, max_pairs=200, threshold=0.01, repeats=1,
        timeout=10, pmc="none", sustain_ms=0, confirm=1,
    )
    tuner = tune.Tuner(args)

    def fake_score(cfg):
        others = any(cfg[k] != v[0] for k, v in tune.KNOBS.items() if k != "agpr_alloc")
        return 100.0 + (5.0 if cfg["agpr_alloc"] == "auto" else 0.0) - (1.0 if others else 0.0)

    def fake_evaluate(cfg, *, move, parent, round_, pmc):
        key = tune.config_key(cfg)
        if key not in tuner.records:
            code = tune.config_key({k: v for k, v in cfg.items() if k != "agpr_alloc"})
            rec = {"config": cfg, "status": "ok", "tflops": fake_score(cfg), "move": move,
                   "round": round_, "code_object": {"isa_sha": code}}
            tuner.records[key] = rec
            tuner.history.append(rec)
        return tuner.records[key]

    monkeypatch.setattr(tuner, "evaluate", fake_evaluate)
    monkeypatch.setattr(tuner, "_retime", fake_score)
    assert tuner.run()["config"]["agpr_alloc"] == tune.KNOBS["agpr_alloc"][0]


def test_lookback_retries_failed_knobs(tmp_path, monkeypatch):
    """A knob that fails alone is re-tried on top of each accepted move, and
    pairs of failed moves are tried before the climb stops."""

    # Synthetic score: setprio helps only once stages=3 is in place, and the
    # pair (sched=interleave, dma_cache=2) helps only together.
    def fake_score(cfg):
        s = 100.0
        s += 10 if cfg["stages"] == 3 else 0
        s += 5 if cfg["stages"] == 3 and cfg["setprio"] else -1 if cfg["setprio"] else 0
        if cfg["sched"] == "interleave" and cfg["dma_cache"] == 2:
            s += 4
        elif cfg["sched"] == "interleave" or cfg["dma_cache"] == 2:
            s -= 1
        return s

    # The climb runs over the synthetic knobs plus two that the score ignores,
    # so whether it stops by itself does not depend on how many knobs the
    # tuner has (the look-back grows with the square of the knob count).
    synthetic = ("stages", "setprio", "sched", "dma_cache", "mfma_group", "agpr_alloc")
    monkeypatch.setattr(tune, "KNOBS", {k: tune.KNOBS[k] for k in synthetic})
    args = SimpleNamespace(
        ledger=str(tmp_path / "ledger"),
        cache=str(tmp_path / "cache"),
        shape="4096,4096,65536",
        data="random",
        start=None,
        max_evals=400,
        max_pairs=400,
        threshold=0.01,
        repeats=1,
        timeout=10,
        pmc="none",
        sustain_ms=0,
        confirm=1,
    )
    tuner = tune.Tuner(args)

    def fake_evaluate(cfg, *, move, parent, round_, pmc):
        key = tune.config_key(cfg)
        if key not in tuner.records:
            rec = {"config": cfg, "status": "ok", "tflops": fake_score(cfg), "move": move, "round": round_}
            tuner.records[key] = rec
            tuner.history.append(rec)
        return tuner.records[key]

    monkeypatch.setattr(tuner, "evaluate", fake_evaluate)
    monkeypatch.setattr(tuner, "_retime", fake_score)
    best = tuner.run()["config"]
    # The climb stopped by itself, not on the evaluation budget.
    assert len(tuner.records) < args.max_evals
    assert best["stages"] == 3 and best["setprio"] is True
    assert best["sched"] == "interleave" and best["dma_cache"] == 2
    # setprio was evaluated (and failed) before stages=3 was accepted.
    tried = [json.loads(k) for k in tuner.records]
    assert any(c["setprio"] and c["stages"] == 2 for c in tried)
