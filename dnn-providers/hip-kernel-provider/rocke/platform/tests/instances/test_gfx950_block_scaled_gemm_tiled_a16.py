# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU checks for the tiled gfx950 FP6 GEMM builder's mfma_atom="16x16x128" path."""

from __future__ import annotations

import itertools
import re
import struct
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify
from rocke.examples.common.scaled_gemm_reference import decode_fp6
from rocke.examples.gfx950.gemm.mxfp6_tiled import bench
from rocke.instances.gfx950.block_scaled_gemm_tiled import (
    EPI16S,
    MFMA_ATOMS,
    STAGE_K,
    TiledBlockScaledGemmSpec,
    _barrier_vmcnt16,
    a_piece16,
    build_tiled_block_scaled_gemm,
    is_valid_spec,
    line16_at,
    line16_first_need,
    mx16_acc_lane,
    mx16_operand_lane,
    slots16,
)

# The 32x32x64 configuration the 16x16x128 path keeps the memory side of.
LINE_RING = {
    "stages": 2, "a_ring": "line", "xcd_remap": "xcd_block", "mfma_form": "intrinsic", "agpr_alloc": "auto",
    "backend": "direct", "b_layout": "kblock", "b_scale_layout": "kblock4", "a_scale_fetch": "iter2",
    "mfma_group": 1, "setprio": True, "fill_split": "line1", "scale_opsel": True, "frag_pipeline": "step",
    "b_release": "half", "a_scale_path": "lds2", "mfma_order": "serp", "dma_place": "even_bfirst",
    "cc_passes": "all", "b_scale_fetch": "pair", "prologue_wait": "stage0", "k_order": "t10",
    "cc_extra": "lds_waw+u16_zext", "as_dma_place": "split_light",
}
PORT = TiledBlockScaledGemmSpec(M=4096, N=4096, K=65536, **LINE_RING, mfma_atom="16x16x128")
GATES = [replace(PORT, M=512, N=512, K=k, xcd_remap="none") for k in (1024, 2048)]
SCHED_ALL = "fence+pairlight+midlgkm"


def _ops(spec) -> tuple[str, Counter]:
    kernel = build_tiled_block_scaled_gemm(spec, arch="gfx950")
    assert not verify(kernel)
    text = serialize(kernel)
    return text, Counter(re.findall(r"\b(tile\.[A-Za-z_0-9]+|memref\.[a-z_0-9]+|scf\.for)\b \(", text))


def _loop_ops(spec) -> list:
    kernel = build_tiled_block_scaled_gemm(spec, arch="gfx950")
    assert not verify(kernel)
    return next(op for op in kernel.body.ops if op.name == "scf.for").regions[0].ops


def _stages_of(ops, name: str) -> list[int]:
    """The stage (64 MFMA gaps) of the loop body each `name` op issues in."""
    n, out = 0, []
    for op in ops:
        if op.name == "tile.mma":
            n += 1
        elif op.name == name:
            out.append(max(n - 1, 0) // 64)
    return out


def test_the_target_and_gate_shapes_are_valid_and_the_name_is_short():
    for spec in (PORT, *GATES):
        assert is_valid_spec(spec) == (True, "ok")
        assert "_a16_" in spec.kernel_name()
    assert len(PORT.kernel_name()) < 120
    # The 32x32x64 kernel's name is the one it had before the field.
    base = replace(PORT, mfma_atom="32x32x64")
    assert "_a16" not in base.kernel_name() and base.kernel_name() == replace(base, epi16=EPI16S[0]).kernel_name()


@pytest.mark.parametrize(
    "changes,needle",
    [
        ({"mfma_atom": "16x16x64"}, "mfma_atom must be one of"),
        ({"K": 1152}, "K = 1024"),
        ({"K": 3072}, "power of two"),
        ({"a_ring": "line2r"}, "a_ring"),
        ({"frag_pipeline": "stage"}, "frag_pipeline"),
        ({"a_scale_path": "direct"}, "a_scale_path"),
        ({"b_scale_fetch": "stage"}, "b_scale_fetch"),
        ({"b_release": "full"}, "b_release"),
        ({"backend": "llvm"}, "backend"),
        ({"mfma_group": 4}, "mfma_group"),
        ({"epilogue": "pk2"}, "epilogue"),
        ({"store_nt": True}, "store_nt"),
        ({"block_m": 128}, "block_m"),
        ({"waves_m": 1, "waves_n": 4}, "waves_m"),
        ({"ablate": "no_dma"}, "ablate"),
        ({"epi16": "dword"}, "epi16"),
        ({"sched16": "fast"}, "sched16"),
        ({"sched16": "fence+fence"}, "sched16"),
        ({"peel16": "trim"}, "peel16"),
    ],
)
def test_rejections(changes, needle):
    spec = replace(PORT, **changes)
    ok, why = is_valid_spec(spec)
    assert not ok and needle in why, why
    with pytest.raises(ValueError):
        build_tiled_block_scaled_gemm(spec)


def test_the_name_follows_every_field_the_kernel_reads():
    names = {PORT.kernel_name()}
    for over in ({"setprio": False}, {"dtype_a": "fp6e3m2", "dtype_b": "fp6e3m2"}, {"cc_passes": "none"},
                 {"cc_extra": "none"}, {"dma_cache": 2}, {"waves_per_eu": 0}, {"dtype_c": "fp16"},
                 {"sched16": "fence"}, {"sched16": SCHED_ALL}, {"peel16": "skip"}):
        names.add(replace(PORT, **over).kernel_name())
    assert len(names) == 11
    assert "_sfpm_" in replace(PORT, sched16=SCHED_ALL).kernel_name()
    assert "_kskip_" in replace(PORT, peel16="skip").kernel_name()
    base = replace(PORT, mfma_atom="32x32x64")
    assert replace(base, sched16="fence").kernel_name() == base.kernel_name()
    assert replace(base, peel16="skip").kernel_name() == base.kernel_name()


def test_the_32x32x64_schedule_fields_change_neither_name_nor_kernel():
    text, _ = _ops(GATES[0])
    for over in ({"dma_place": "even"}, {"sync_place": "split"}, {"k_order": "rot"}, {"as_dma_place": "default"},
                 {"prologue_wait": "all"}, {"mfma_order": "rows"}, {"k_peel": True}, {"clamp_hoist": True},
                 {"peel_dma": "skip"}, {"z_addr": "split"}, {"pre_e_frag": "g9"}, {"trip_salu": "hoist"},
                 {"epi_store": "soff"}, {"prologue_order": "dma_first"}, {"aux_place": "pre_z"}):
        spec = replace(GATES[0], **over)
        assert spec.kernel_name() == GATES[0].kernel_name()
        assert _ops(spec)[0] == text


def test_the_launch_fields_mean_what_they_mean_on_the_line_ring():
    base = GATES[0]
    plain = serialize(build_tiled_block_scaled_gemm(base, arch="gfx950"))
    names = {base.kernel_name()}

    def built(**over):
        spec = replace(base, **over)
        assert is_valid_spec(spec) == (True, "ok")
        names.add(spec.kernel_name())
        text = serialize(build_tiled_block_scaled_gemm(spec, arch="gfx950"))
        return spec, text, plain.replace(base.kernel_name(), spec.kernel_name())

    # shape_args=False drops the unread shape parameters and nothing else.
    spec, text, ref = built(shape_args=False)
    shape_lines = ["    %M : i32", "    %N : i32", "    %K : i32"]
    assert [ln for ln in plain.splitlines() if re.search(r"%[MNK]\b", ln)] == shape_lines
    assert text.splitlines() == [ln for ln in ref.splitlines() if ln not in shape_lines]
    names5 = ["A", "B", "A_scale", "B_scale", "C"]
    assert [p.name for p in build_tiled_block_scaled_gemm(spec, arch="gfx950").params] == names5
    assert [p["name"] for p in bench.tiled_block_scaled_gemm_signature(spec)] == names5
    assert bench.kernel_args(spec, [1, 2, 3, 4, 5]) == struct.pack("<QQQQQ", 1, 2, 3, 4, 5)
    assert bench.kernel_args(base, [1, 2, 3, 4, 5]) == struct.pack("<QQQQQiii", 1, 2, 3, 4, 5, 512, 512, 1024)
    # kernarg_preload sets its kernel attribute only.
    spec, text, ref = built(kernarg_preload=True)
    assert "kernarg_preload = b:true" in text and "kernarg_preload" not in plain
    assert re.sub(r"kernarg_preload = b:true(, )?", "", text).replace(", }", " }") == ref
    # loop_pad: priority-0 pads right before the K loop, which leave the
    # priority at 0 (the prologue raises none).
    pad = "    tile.s_setprio (  ) { level = i:0 }"
    for n in (1, 5):
        spec, text, ref = built(loop_pad=n)
        pre, body = text.split("region @body {", 1)
        pre0, body0 = ref.split("region @body {", 1)
        assert body == body0
        lines = pre.splitlines()
        at = [k for k, ln in enumerate(lines) if ln == pad]
        assert at == list(range(at[0], at[0] + n)) and lines[: at[0]] + lines[at[-1] + 1 :] == pre0.splitlines()
        assert all("arith.constant" in ln for ln in lines[at[-1] + 1 : -2]), lines[at[-1] + 1 :]
        assert "tile.s_setprio" not in "\n".join(lines[: at[0]])
    assert len(names) == 5


def test_the_lane_maps_compute_a_16x16x128_tile():
    """Operands packed by `mx16_operand_lane` and the accumulator read by
    `mx16_acc_lane` give A B^T of a 16 x 16 x 128 tile (32-value K blocks with
    their own scales)."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((16, 128))
    b = rng.standard_normal((16, 128))
    sa, sb = rng.integers(-3, 4, size=(16, 4)), rng.integers(-3, 4, size=(16, 4))
    a_s, b_s = a * np.repeat(np.ldexp(1.0, sa), 32, axis=1), b * np.repeat(np.ldexp(1.0, sb), 32, axis=1)
    lane = np.arange(64)
    kg, r = mx16_operand_lane(lane)
    a_op = np.stack([a[r[l], 32 * kg[l] : 32 * kg[l] + 32] for l in lane]) * np.ldexp(1.0, sa[r, kg])[:, None]
    b_op = np.stack([b[r[l], 32 * kg[l] : 32 * kg[l] + 32] for l in lane]) * np.ldexp(1.0, sb[r, kg])[:, None]
    acc = np.zeros((64, 4))
    for l in lane:
        for e in range(4):
            row, col = mx16_acc_lane(l, e)
            acc[l, e] = sum(a_op[16 * g + row] @ b_op[16 * g + col] for g in range(4))
    ref = a_s @ b_s.T
    for l in lane:
        for e in range(4):
            assert acc[l, e] == pytest.approx(ref[mx16_acc_lane(l, e)], rel=1e-12, abs=1e-12)


def test_a_pieces_cover_each_fragment_in_three_aligned_8_byte_reads():
    for u in range(4):
        for kg in range(4):
            got = [a_piece16(u, kg, p) for p in range(3)]
            starts = [line * 128 + byte for line, byte in got]
            assert starts == [96 * u + 24 * kg + 8 * p for p in range(3)]
            for line, byte in got:
                assert 0 <= line < 3 and byte % 8 == 0 and byte % 16 + 8 <= 16


def test_line_refills_go_into_the_slot_stage_s_reads_last_three_stages_ahead():
    for s in range(8):
        line = line16_at(s)
        if s % 4 == 0:
            assert line is None
            continue
        old = line - 3
        # Stage s's own fragments read the old line last; stage s + 3 reads the new one first.
        assert (128 * old + 127) // 96 == s and line16_first_need(line) == s + 3


@pytest.mark.parametrize("n", range(9))
def test_a_column_halfs_vmem_ops_are_four_gaps_apart_from_gap_2(n):
    at = slots16(n)
    assert len(at) == n and all(2 <= g <= 31 for g in at)
    assert all(b - a >= 4 for a, b in itertools.pairwise(at))


def test_too_many_vmem_ops_for_a_column_half_are_refused():
    with pytest.raises(ValueError):
        slots16(9)


def test_the_packers_round_trip_and_put_every_byte_where_a_lane_reads_it():
    rng = np.random.default_rng(3)
    n, k = 512, 2048
    b = rng.integers(0, 256, size=(n, k * 3 // 4), dtype=np.uint8)
    x = bench.b_c0c1p(b)
    assert x.size == b.size and (bench.b_c0c1p_unpack(x, n, k) == b).all()
    sb = rng.integers(0, 256, size=(k // 32, n), dtype=np.uint8)
    y = bench.scales16_pairs(sb)
    assert y.size == sb.size
    for t, h, g, p, plane, lane in ((0, 0, 0, 0, 0, 0), (5, 1, 2, 1, 1, 37), (15, 1, 3, 1, 2, 63)):
        kg, c = mx16_operand_lane(lane)
        stream = [b[128 * g + 16 * (4 * h + 2 * p + q) + c, 96 * t + 24 * kg + e] for q in range(2) for e in range(24)]
        at = (((((t * 2 + h) * (n // 128) + g) * 2 + p) * 3 + plane) * 64 + lane) * 16
        assert list(x[at : at + 16]) == stream[16 * plane : 16 * plane + 16]
        v, q = t // 2, t % 2
        for j in range(4):
            at = (((v * (n // 128) + g) * 64 + lane) * 4 + 2 * q + h) * 4 + j
            assert y[at] == sb[4 * t + kg, 128 * g + 16 * (4 * h + j) + c]


def test_the_packed_b_operands_hold_the_row_major_bytes():
    """bench.device_inputs packs B so that a lane's two operands of a column-block pair are the 48 bytes of three
    planes; decoding them gives the row-major B's values."""
    spec = GATES[0]
    a, b, sa, sb = bench.dyadic_inputs(spec)
    _, bp, _, _ = bench.device_inputs(spec, (a, b, sa, sb))
    x = bp.reshape(spec.K // STAGE_K, 2, spec.N // 128, 2, 3, 64, 16)
    t, h, g, p = 3, 1, 1, 0
    two = np.concatenate([x[t, h, g, p, plane] for plane in range(3)], axis=1)  # [lane][48]
    for q in range(2):
        cols = 128 * g + 16 * (4 * h + 2 * p + q) + np.arange(64) % 16
        want = np.stack([b[cols[l], 96 * t + 24 * (l // 16) : 96 * t + 24 * (l // 16) + 24] for l in range(64)])
        assert (two[:, 24 * q : 24 * q + 24] == want).all()
        assert np.abs(decode_fp6(want, spec.dtype_b)).max() > 0


@pytest.mark.parametrize("spec", GATES + [PORT], ids=["k1024", "k2048", "target"])
def test_the_kernel_is_the_planned_stream(spec):
    text, ops = _ops(spec)
    trips = spec.K // 1024
    # One trip of eight stages in the loop body: 64 MFMAs per stage; per stage 44 LDS reads (24 A pieces, 8 A scale
    # bytes, 12 B planes), 12.25 DMAs and half a B scale pair load; the prologue's 38 DMAs and reads, one pair load.
    assert ops["scf.for"] == 1
    assert ops["tile.mma"] == 512 and ops["tile.sched_barrier"] == 512
    assert ops["tile.async_buffer_load_lds_addr"] == 38 + 98
    assert ops["tile.smem_load_vN"] == 38 + 8 * 44
    assert ops["tile.buffer_load_vN"] == 1 + (3 if trips == 1 else 4)
    assert ops["tile.s_barrier_bare"] == 2 + 16 and ops["tile.buffer_store_bf16"] == 256
    assert text.count('op_id = s:"mfma_scale_f32_16x16x128_f8f6f4"') == 512
    sels = Counter(re.findall(r"scale_sel = l:\[ i:(\d+), i:(\d+) \]", text))
    assert set(sels) == {("0", str(j)) for j in range(4)} and set(sels.values()) == {128}
    assert "lds_dma_explicit = b:true" in text


def test_barrier_waits_count_what_may_stay_in_flight():
    # Prologue ops gated at barriers -2, -1 and 0, then a load the compiler waits for; a two-op body gated at its
    # trip's mid(0) and E(0). Barrier -2 (after four ops) needs op 0, barrier -1 op 1; each trip's mid(0) and E(0)
    # need the body op right before them.
    log = [-2, -1, 0, None, 0, 1]
    bars = [(3, -2, False), (4, -1, False), (5, 0, True), (6, 1, True)]
    assert _barrier_vmcnt16(log, bars, (4, 6), 2) == [2, 2, 0, 0]
    # A body op issued for the next trip's mid(0): the first trip's mid(0) needs the prologue's op only (one op
    # after it), every later trip's the op the trip before issued (none after it).
    assert _barrier_vmcnt16([0, None, 16], [(2, 0, True)], (2, 3), 1) == [1]
    assert _barrier_vmcnt16([0, None, 16], [(2, 0, True)], (2, 3), 4) == [0]
    assert _barrier_vmcnt16([None, None], [(2, 0, True)], (0, 2), 1) == [-1]
    # peel16: a prologue op for mid(0), a body op for the next trip's mid(0) and a peeled last trip whose mid(0)
    # comes after one op the compiler waits for. Two trips: one loop trip, so the peel's mid(0) needs the body's op;
    # one trip: no loop, the prologue's.
    assert _barrier_vmcnt16([0, 16, None], [(3, 0, "peel")], (1, 2), 2, (2, 3)) == [1]
    assert _barrier_vmcnt16([0, None], [(2, 0, "peel")], (1, 1), 1, (1, 2)) == [1]


@pytest.mark.parametrize("spec", GATES + [PORT], ids=["k1024", "k2048", "target"])
def test_the_peeled_last_trip_issues_nothing_for_stages_past_k(spec):
    trips = spec.K // 1024
    peeled = replace(spec, peel16="skip", sched16=SCHED_ALL, epi16="swap")
    assert is_valid_spec(peeled) == (True, "ok")
    _, ops = _ops(peeled)
    # The loop runs trips - 1 trips (none for one trip) and the last trip follows it: per stage the same MFMAs,
    # reads and barriers, but not the next trip's B DMAs (stages 6, 7), line refills (stages 5-7), A scale DMAs
    # (stages 0, 4) and B scale pair, nor stage 7's reads of the next stage.
    assert ops["scf.for"] == (trips > 1)
    assert ops["tile.mma"] == 512 * min(trips, 2)
    assert ops["tile.async_buffer_load_lds_addr"] == 38 + 98 * (trips > 1) + 60
    assert ops["tile.buffer_load_vN"] == 1 + 4 * (trips > 1) + 3
    assert ops["tile.smem_load_vN"] == 38 + 8 * 44 * (trips > 1) + 8 * 44 - (8 * 4 + 6)
    assert ops["tile.s_barrier_bare"] == 2 + 16 * min(trips, 2)
    if trips > 1:
        kernel = build_tiled_block_scaled_gemm(peeled, arch="gfx950")
        top = kernel.body.ops
        at = next(i for i, op in enumerate(top) if op.name == "scf.for")
        assert Counter(_stages_of(top[at + 1 :], "tile.async_buffer_load_lds_addr")) == {
            0: 6, 1: 14, 2: 14, 3: 14, 4: 6, 5: 6}
        assert Counter(_stages_of(top[at].regions[0].ops, "tile.async_buffer_load_lds_addr")) == {
            0: 7, 1: 14, 2: 14, 3: 14, 4: 7, 5: 14, 6: 14, 7: 14}


def test_the_waits_of_the_port():
    kernel = build_tiled_block_scaled_gemm(GATES[1], arch="gfx950")
    waits = [op.attrs["vmcnt"] for op in kernel.body.ops if op.name == "tile.s_waitcnt"]
    loop = next(op for op in kernel.body.ops if op.name == "scf.for")
    body = [op.attrs["vmcnt"] for op in loop.regions[0].ops if op.name == "tile.s_waitcnt"]
    # The prologue's two barriers, then a trip's 16, then the drain before the epilogue.
    assert waits == [26, 19, 0] and len(body) == 16
    assert all(0 <= n <= 63 for n in body)


@pytest.mark.parametrize("spec", GATES, ids=["k1024", "k2048"])
def test_the_schedule_options(spec):
    trips = spec.K // 1024
    _, ops = _ops(spec)
    _, fenced = _ops(replace(spec, sched16="fence"))
    # fence: a second scheduling fence right after every MFMA, nothing else.
    assert fenced["tile.sched_barrier"] == 2 * ops["tile.sched_barrier"]
    assert {k: v for k, v in fenced.items() if k != "tile.sched_barrier"} == {
        k: v for k, v in ops.items() if k != "tile.sched_barrier"}
    # pairlight: the B scale pairs load two at a time in stages 0 and 4 instead of one in each even stage; one trip
    # loads no pair for the stages past K.
    assert _stages_of(_loop_ops(spec), "tile.buffer_load_vN") == [0, 2, 4] + ([6] if trips > 1 else [])
    light = _loop_ops(replace(spec, sched16="pairlight"))
    assert _stages_of(light, "tile.buffer_load_vN") == [0, 0, 4] + ([4] if trips > 1 else [])
    # midlgkm: Bar(mid) leaves the 15 newest LDS reads (A's) in flight, except where they are dead.
    waits = [op.attrs["lgkmcnt"] for op in _loop_ops(replace(spec, sched16="midlgkm")) if op.name == "tile.s_waitcnt"]
    want = [15, 0] * 8
    if trips == 1:
        want[14] = 0
    assert waits == want
    assert [op.attrs["lgkmcnt"] for op in _loop_ops(spec) if op.name == "tile.s_waitcnt"] == [0] * 16


@pytest.mark.parametrize("spec", GATES, ids=["k1024", "k2048"])
def test_the_swap_epilogue_pairs_column_blocks(spec):
    _, ops = _ops(spec)
    _, swapped = _ops(replace(spec, epi16="swap"))
    # One swap per column-block pair, row block and packed dword; the stores and the K loop stay.
    assert ops["tile.permlane32_swap"] == 0 and swapped["tile.permlane32_swap"] == 64
    assert swapped["tile.buffer_store_bf16"] == ops["tile.buffer_store_bf16"] == 256
    assert {k: v for k, v in swapped.items() if k != "tile.permlane32_swap"} == ops
    assert "_eswap_" in replace(spec, epi16="swap").kernel_name()


def test_the_atoms():
    assert MFMA_ATOMS == ("32x32x64", "16x16x128")
