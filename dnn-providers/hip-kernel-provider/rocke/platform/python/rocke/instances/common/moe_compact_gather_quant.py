# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compact routed-row gather fused with blockwise FP8 quantization.

The routing active-pack kernel produces padded ``tile_m``-row expert blocks.
This builder gathers the original activation rows through those token ids and
requantizes each ``[tile_m, 128]`` slab with one shared scale. Broadcasting the
scale to every row in the block matches the fused MegaMoE accumulator contract:
one MFMA fragment spans several output rows, so a per-token scale cannot be
applied as one scalar after the instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...core.arch import ArchTarget
from ...core.ir import F32, FP8E4M3, I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import io_ir_type
from ...helpers.reduction import (
    block_lds_reduce,
    block_lds_reduce_with_wave_prologue,
    tree_reduce,
)
from ...helpers.spec import SignatureBuilder, kernel_name_join

InputDType = Literal["f16", "bf16"]
_GROUP_K = 128
_FP8_MAX = 448.0
_AMAX_FLOOR = 1e-6


@dataclass(frozen=True)
class MoeCompactGatherQuantSpec:
    """One compact gather + shared-block FP8 quantization configuration."""

    tokens: int
    hidden: int
    max_blocks: int
    tile_m: int = 16
    input_dtype: InputDType = "bf16"
    block_size: int = 256
    vec: int = 4
    wave_size: int = 64
    name: str = "rocke_moe_compact_gather_quant"

    @property
    def hidden_blocks(self) -> int:
        return self.hidden // _GROUP_K

    @property
    def qvecs_per_block(self) -> int:
        return self.tile_m * _GROUP_K // self.vec

    @property
    def passes_per_thread(self) -> int:
        return self.qvecs_per_block // self.block_size

    @property
    def output_rows(self) -> int:
        return self.max_blocks * self.tile_m

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.input_dtype,
            f"T{self.tokens}",
            f"H{self.hidden}",
            f"MB{self.max_blocks}",
            f"tm{self.tile_m}",
            f"b{self.block_size}",
            f"v{self.vec}",
        )


def is_valid_spec(
    spec: MoeCompactGatherQuantSpec, arch: str = "gfx950"
) -> tuple[bool, str]:
    """Return whether the compact gather/quant geometry is supported."""

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as error:
        return False, str(error)
    if spec.tokens <= 0 or spec.hidden <= 0 or spec.max_blocks <= 0:
        return False, "tokens, hidden, and max_blocks must be positive"
    if spec.hidden % _GROUP_K:
        return False, f"hidden ({spec.hidden}) must be divisible by {_GROUP_K}"
    if spec.tile_m <= 0:
        return False, f"tile_m must be positive (got {spec.tile_m})"
    if spec.input_dtype not in ("f16", "bf16"):
        return False, f"unsupported input_dtype {spec.input_dtype!r}"
    if spec.vec != 4:
        return False, "the packed FP8 conversion path requires vec=4"
    if spec.block_size not in (64, 128, 256, 512, 1024):
        return False, f"unsupported block_size {spec.block_size}"
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    if spec.qvecs_per_block % spec.block_size:
        return False, (
            f"tile_m*{_GROUP_K}/vec ({spec.qvecs_per_block}) must be divisible "
            f"by block_size ({spec.block_size})"
        )
    bytes_lds = spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def build_moe_compact_gather_quant(
    spec: MoeCompactGatherQuantSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build compact gather plus ``[tile_m,128]`` shared-scale quantization.

    ABI::

        X              : ptr<input_dtype>  # [tokens, hidden]
        SortedTokenIds : ptr<i32>          # [max_blocks * tile_m]
        BlockExpertIds : ptr<i32>          # [max_blocks], -1 is inactive
        A              : ptr<fp8e4m3>      # [max_blocks * tile_m, hidden]
        AScale         : ptr<f32>          # [max_blocks * tile_m, hidden/128]
        tokens, hidden : i32

    Grid is ``(hidden/128, max_blocks, 1)``.
    """

    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid moe compact-gather-quant spec for {arch}: {why}")

    input_ty = io_ir_type(spec.input_dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    source = b.param(
        "X",
        PtrType(input_ty, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sorted_token_ids = b.param(
        "SortedTokenIds",
        PtrType(I32, "global"),
        noalias=True,
        readonly=True,
        align=4,
    )
    block_expert_ids = b.param(
        "BlockExpertIds",
        PtrType(I32, "global"),
        noalias=True,
        readonly=True,
        align=4,
    )
    activation = b.param(
        "A",
        PtrType(FP8E4M3, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    activation_scale = b.param(
        "AScale",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=4,
    )
    tokens = b.param("tokens", I32)
    _hidden = b.param("hidden", I32)

    tid = b.thread_id_x()
    hidden_block = b.block_id_x()
    routed_block = b.block_id_y()
    block_expert = b.global_load_i32(block_expert_ids, routed_block)
    active_block = b.cmp_ge(block_expert, b.const_i32(0))
    hidden_base = b.mul(hidden_block, b.const_i32(_GROUP_K))
    routed_row_base = b.mul(routed_block, b.const_i32(spec.tile_m))
    lds = b.smem_alloc(F32, [spec.block_size], name_hint="gather_amax")

    amax = b.const_f32(_AMAX_FLOOR)
    cached: list[list] = []
    coordinates: list[tuple] = []
    for pass_index in range(spec.passes_per_thread):
        qvec = b.add(tid, b.const_i32(pass_index * spec.block_size))
        element = b.mul(qvec, b.const_i32(spec.vec))
        local_row = b.div(element, b.const_i32(_GROUP_K))
        local_column = b.mod(element, b.const_i32(_GROUP_K))
        output_row = b.add(routed_row_base, local_row)
        token = b.global_load_i32(sorted_token_ids, output_row)
        valid_token = b.land(
            active_block,
            b.land(
                b.cmp_ge(token, b.const_i32(0)),
                b.cmp_lt(token, tokens),
            ),
        )
        safe_token = b.select(valid_token, token, b.const_i32(0))
        source_offset = b.add(
            b.mul(safe_token, b.const_i32(spec.hidden)),
            b.add(hidden_base, local_column),
        )
        packed = b.global_load_vN(source, source_offset, input_ty, spec.vec)
        values = [
            b.select(
                valid_token,
                b.cast_to_f32(b.vec_extract(packed, index)),
                b.const_f32(0.0),
            )
            for index in range(spec.vec)
        ]
        amax = b.fmax(
            amax,
            tree_reduce(b, b.fmax, [b.fabs(value) for value in values]),
        )
        cached.append(values)
        coordinates.append((output_row, local_column))

    target = ArchTarget.from_gfx(arch)
    if target.wave_size == spec.wave_size and spec.block_size % spec.wave_size == 0:
        block_amax = block_lds_reduce_with_wave_prologue(
            b,
            amax,
            lds,
            tid,
            block_size=spec.block_size,
            combine="max",
            wave_size=spec.wave_size,
        )
    else:
        block_amax = block_lds_reduce(
            b,
            amax,
            lds,
            tid,
            block_size=spec.block_size,
            combine="max",
        )
    scale = b.fmul(
        b.fmax(block_amax, b.const_f32(_AMAX_FLOOR)),
        b.const_f32(1.0 / _FP8_MAX),
    )
    inv_scale = b.rcp_fast(scale)

    for values, (output_row, local_column) in zip(cached, coordinates):
        scaled = b.vec_pack([b.fmul(value, inv_scale) for value in values], F32)
        quantized = b.cvt_pk_fp8_f32x4(scaled)
        output_offset = b.add(
            b.mul(output_row, b.const_i32(spec.hidden)),
            b.add(hidden_base, local_column),
        )
        b.global_store_vN(activation, output_offset, quantized, spec.vec)

    with b.scf_if(b.cmp_lt(tid, b.const_i32(spec.tile_m))):
        scale_row = b.add(routed_row_base, tid)
        scale_offset = b.add(
            b.mul(scale_row, b.const_i32(spec.hidden_blocks)),
            hidden_block,
        )
        b.global_store(activation_scale, scale_offset, scale, align=4)

    b.ret()
    return b.kernel


def moe_compact_gather_quant_grid(
    spec: MoeCompactGatherQuantSpec,
) -> tuple[int, int, int]:
    return (spec.hidden_blocks, spec.max_blocks, 1)


def moe_compact_gather_quant_signature(
    spec: MoeCompactGatherQuantSpec,
) -> list[dict[str, str]]:
    return (
        SignatureBuilder()
        .ptr("X", spec.input_dtype)
        .ptr("SortedTokenIds", "i32")
        .ptr("BlockExpertIds", "i32")
        .ptr("A", "fp8e4m3")
        .ptr("AScale", "f32")
        .scalar("tokens", "i32")
        .scalar("hidden", "i32")
        .build()
    )


__all__ = [
    "MoeCompactGatherQuantSpec",
    "build_moe_compact_gather_quant",
    "is_valid_spec",
    "moe_compact_gather_quant_grid",
    "moe_compact_gather_quant_signature",
]
