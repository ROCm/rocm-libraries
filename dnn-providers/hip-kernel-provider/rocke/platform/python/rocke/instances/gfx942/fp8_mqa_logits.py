# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 FP8 MQA-logits kernel.

For query row ``m`` and KV position ``n`` inside the row window, compute::

    logits[m, n] = sum_h relu(dot(Q[m, h, :], KV[n, :]) * scale[n])
                          * weights[m, h]

The implementation uses the native 16x16x32 FP8 MFMA atom. Multiple query rows
share each KV load, while independent waves own disjoint groups of KV columns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from ...core.ir import F32, FP8E4M3, I32, I64, IRBuilder, KernelDef, PtrType
from ...helpers.atoms import MfmaAtom
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    validate_arch_and_block_size,
    validate_mfma_atom_in_catalog,
)
from ...helpers.spec import SignatureBuilder, kernel_name_join


_MIN_TILES_PER_SPLIT = 8


@dataclass(frozen=True)
class Fp8MqaLogitsSpec:
    """Compile-time geometry for FP8 MQA logits."""

    num_heads: int = 64
    head_dim: int = 128
    block_kv: int = 128
    rows_per_block: int = 2
    waves_per_block: int = 4
    waves_per_eu: int | None = 2
    name: str = "rocke_fp8_mqa_logits"

    @property
    def atom(self) -> MfmaAtom:
        return MfmaAtom.fp8_16x16x32()

    @property
    def block_size(self) -> int:
        return 64 * self.waves_per_block

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"H{self.num_heads}",
            f"D{self.head_dim}",
            f"BKV{self.block_kv}",
            f"R{self.rows_per_block}",
            f"W{self.waves_per_block}",
        )


def is_valid_spec(spec: Fp8MqaLogitsSpec, arch: str = "gfx942") -> Tuple[bool, str]:
    """Return whether ``spec`` is supported by the gfx942 implementation."""

    if arch != "gfx942":
        return False, f"fp8_mqa_logits currently supports gfx942 only, got {arch!r}"
    ok, reason, _target = validate_arch_and_block_size(arch, spec.block_size)
    if not ok:
        return False, reason
    atom = spec.atom
    if spec.num_heads <= 0 or spec.num_heads % atom.m:
        return False, f"num_heads must be a positive multiple of {atom.m}"
    if spec.head_dim <= 0 or spec.head_dim % atom.k:
        return False, f"head_dim must be a positive multiple of {atom.k}"
    if spec.block_kv <= 0 or spec.block_kv % atom.n:
        return False, f"block_kv must be a positive multiple of {atom.n}"
    if spec.rows_per_block <= 0:
        return False, "rows_per_block must be positive"
    if spec.waves_per_block <= 0:
        return False, "waves_per_block must be positive"
    n_tiles = spec.block_kv // atom.n
    if n_tiles % spec.waves_per_block:
        return False, (
            f"block_kv / {atom.n} ({n_tiles}) must be divisible by "
            f"waves_per_block ({spec.waves_per_block})"
        )
    if spec.waves_per_eu is not None and spec.waves_per_eu <= 0:
        return False, "waves_per_eu must be positive or None"
    return True, "ok"


def _ceildiv(b: IRBuilder, value, divisor):
    one_less = (
        b.const_i32(divisor - 1)
        if isinstance(divisor, int)
        else b.sub(divisor, b.const_i32(1))
    )
    divisor_value = b.const_i32(divisor) if isinstance(divisor, int) else divisor
    return b.div(b.add(value, one_less), divisor_value)


def build_fp8_mqa_logits(spec: Fp8MqaLogitsSpec, arch: str = "gfx942") -> KernelDef:
    """Build the native-FP8 MQA-logits kernel.

    ``seq_len`` must be host-padded to a multiple of ``rows_per_block``.
    Inputs use native gfx942 E4M3 FNUZ byte encoding. Positions outside each
    row's ``[cu_starts, cu_ends)`` window are left untouched.
    """

    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fp8_mqa_logits spec: {why}")
    validate_mfma_atom_in_catalog(spec.atom, arch, where="fp8_mqa_logits")

    atom = spec.atom
    h = spec.num_heads
    d = spec.head_dim
    bkv = spec.block_kv
    rpb = spec.rows_per_block
    wpb = spec.waves_per_block
    n_tiles_per_wave = (bkv // atom.n) // wpb
    m_tiles = h // atom.m
    k_steps = d // atom.k

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    q = b.param("Q", PtrType(FP8E4M3, "global"), readonly=True, align=8)
    kv = b.param("KV", PtrType(FP8E4M3, "global"), readonly=True, align=8)
    kv_scales = b.param("kv_scales", PtrType(F32, "global"), readonly=True, align=4)
    weights = b.param("weights", PtrType(F32, "global"), readonly=True, align=4)
    cu_starts = b.param("cu_starts", PtrType(I32, "global"), readonly=True, align=4)
    cu_ends = b.param("cu_ends", PtrType(I32, "global"), readonly=True, align=4)
    logits = b.param("logits", PtrType(F32, "global"), writeonly=True, align=4)
    seq_len = b.param("seq_len", I32)
    seq_len_kv = b.param("seq_len_kv", I32)
    stride_logits_s = b.param("stride_logits_s", I32)
    num_splits = b.param("num_splits", I32)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    split_id = b.block_id_y()
    wave = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    lane_decode = decode_mfma_lanes(b, atom, lane)

    n_blocks = _ceildiv(b, seq_len, rpb)
    reverse_bid = b.sub(b.sub(n_blocks, bid), b.const_i32(1))
    row0 = b.mul(reverse_bid, b.const_i32(rpb))
    zero_i32 = b.const_i32(0)
    zero_f32 = b.const_f32(0.0)

    starts = []
    ends = []
    q_fragments = []
    weight_fragments = []
    for row_offset in range(rpb):
        row = b.add(row0, b.const_i32(row_offset))
        start = b.smax(b.global_load_i32(cu_starts, row), zero_i32)
        end = b.smin(b.global_load_i32(cu_ends, row), seq_len_kv)
        starts.append(start)
        ends.append(end)

        row_q_fragments = []
        row_weight_fragments = []
        for mi in range(m_tiles):
            head = b.add(b.const_i32(mi * atom.m), lane_decode.m_in_atom)
            row_head = b.add(b.mul(row, b.const_i32(h)), head)
            q_base = b.mul(row_head, b.const_i32(d))
            mi_q_fragments = []
            for kk in range(k_steps):
                k_lane = b.add(
                    b.const_i32(kk * atom.k),
                    b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane)),
                )
                q_addr = b.add(q_base, k_lane)
                mi_q_fragments.append(
                    b.global_load_vN(
                        q,
                        q_addr,
                        FP8E4M3,
                        atom.a_per_lane,
                        align=atom.a_per_lane,
                    )
                )
            row_q_fragments.append(mi_q_fragments)

            mi_weights = []
            for elem in range(atom.c_per_lane):
                head_offset = b.add(
                    b.mul(lane_decode.k_blk, b.const_i32(atom.c_per_lane)),
                    b.const_i32(elem),
                )
                weight_head = b.add(b.const_i32(mi * atom.m), head_offset)
                weight_addr = b.add(b.mul(row, b.const_i32(h)), weight_head)
                mi_weights.append(b.global_load_f32(weights, weight_addr))
            row_weight_fragments.append(mi_weights)
        q_fragments.append(row_q_fragments)
        weight_fragments.append(row_weight_fragments)

    tile_start = starts[0]
    tile_end = ends[0]
    for row_offset in range(1, rpb):
        tile_start = b.smin(tile_start, starts[row_offset])
        tile_end = b.smax(tile_end, ends[row_offset])
    tile_start = b.mul(b.div(tile_start, b.const_i32(bkv)), b.const_i32(bkv))

    window_tiles = _ceildiv(b, b.sub(tile_end, tile_start), bkv)
    split_columns = b.mul(_ceildiv(b, window_tiles, num_splits), b.const_i32(bkv))
    tile_start = b.add(tile_start, b.mul(split_id, split_columns))
    tile_end = b.smin(b.add(tile_start, split_columns), tile_end)

    tile_loop = b.scf_for(
        tile_start,
        tile_end,
        b.const_i32(bkv),
        iv_name="col0",
    )
    with tile_loop as col0:
        wave_tile_base = b.mul(wave, b.const_i32(n_tiles_per_wave))
        columns = []
        scales = []
        kv_fragments = []
        for ni in range(n_tiles_per_wave):
            absolute_ni = b.add(wave_tile_base, b.const_i32(ni))
            column = b.add(
                b.add(col0, b.mul(absolute_ni, b.const_i32(atom.n))),
                lane_decode.n_in_atom,
            )
            columns.append(column)
            clamped_column = b.smin(column, b.sub(seq_len_kv, b.const_i32(1)))
            scales.append(b.global_load_f32(kv_scales, clamped_column))
            kv_base = b.mul(clamped_column, b.const_i32(d))
            ni_kv_fragments = []
            for kk in range(k_steps):
                k_lane = b.add(
                    b.const_i32(kk * atom.k),
                    b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane)),
                )
                kv_addr = b.add(kv_base, k_lane)
                ni_kv_fragments.append(
                    b.global_load_vN(
                        kv,
                        kv_addr,
                        FP8E4M3,
                        atom.b_per_lane,
                        align=atom.b_per_lane,
                    )
                )
            kv_fragments.append(ni_kv_fragments)

        for row_offset in range(rpb):
            row = b.add(row0, b.const_i32(row_offset))
            row_byte_offset = b.mul(
                b.mul(b.sext(row, I64), b.sext(stride_logits_s, I64)),
                b.const_i64(4),
            )
            logits_row = b.global_ptr_add(logits, row_byte_offset)
            for ni in range(n_tiles_per_wave):
                column_sum = zero_f32
                for mi in range(m_tiles):
                    accumulator = atom.zero_acc(b)
                    for kk in range(k_steps):
                        accumulator = atom.emit(
                            b,
                            q_fragments[row_offset][mi][kk],
                            kv_fragments[ni][kk],
                            accumulator,
                        )
                    for elem in range(atom.c_per_lane):
                        score = b.vec_extract(accumulator, elem)
                        relu = b.fmax(score, zero_f32)
                        column_sum = b.fma(
                            relu,
                            weight_fragments[row_offset][mi][elem],
                            column_sum,
                        )
                column_sum = b.fmul(column_sum, scales[ni])
                column_sum = b.fadd(column_sum, b.warp_shuffle_xor(column_sum, 16))
                column_sum = b.fadd(column_sum, b.warp_shuffle_xor(column_sum, 32))

                in_window = b.land(
                    b.cmp_ge(columns[ni], starts[row_offset]),
                    b.cmp_lt(columns[ni], ends[row_offset]),
                )
                is_writer = b.land(
                    b.cmp_eq(lane_decode.k_blk, zero_i32),
                    in_window,
                )
                with b.scf_if(is_writer):
                    b.global_store(logits_row, columns[ni], column_sum, align=4)

    b.ret()
    return b.kernel


def fp8_mqa_logits_num_splits(
    seq_len_padded: int,
    seq_len_kv: int,
    *,
    rows_per_block: int,
    block_kv: int,
    num_cus: int,
    target_blocks_per_cu: int = 4,
) -> int:
    """Choose independent KV-column splits to fill the target."""

    grid_x = seq_len_padded // rows_per_block
    if grid_x == 0 or seq_len_kv < 4096:
        return 1
    if target_blocks_per_cu <= 0:
        raise ValueError("target_blocks_per_cu must be positive")
    target_blocks = target_blocks_per_cu * num_cus
    if grid_x >= target_blocks:
        return 1
    max_splits = max(1, (seq_len_kv // block_kv) // _MIN_TILES_PER_SPLIT)
    return max(1, min(math.ceil(target_blocks / grid_x), max_splits))


def fp8_mqa_logits_grid(
    seq_len_padded: int,
    num_splits: int,
    spec: Fp8MqaLogitsSpec,
) -> Tuple[int, int, int]:
    """Return the launch grid for already-padded query rows."""

    if seq_len_padded % spec.rows_per_block:
        raise ValueError("seq_len_padded must be divisible by rows_per_block")
    return (seq_len_padded // spec.rows_per_block, num_splits, 1)


def fp8_mqa_logits_signature(_spec: Fp8MqaLogitsSpec):
    """Return the packed kernel ABI."""

    return (
        SignatureBuilder()
        .ptr("Q", "fp8e4m3")
        .ptr("KV", "fp8e4m3")
        .ptr("kv_scales", "f32")
        .ptr("weights", "f32")
        .ptr("cu_starts", "i32")
        .ptr("cu_ends", "i32")
        .ptr("logits", "f32")
        .scalar("seq_len", "i32")
        .scalar("seq_len_kv", "i32")
        .scalar("stride_logits_s", "i32")
        .scalar("num_splits", "i32")
        .build()
    )


__all__ = [
    "Fp8MqaLogitsSpec",
    "build_fp8_mqa_logits",
    "fp8_mqa_logits_grid",
    "fp8_mqa_logits_num_splits",
    "fp8_mqa_logits_signature",
    "is_valid_spec",
]
