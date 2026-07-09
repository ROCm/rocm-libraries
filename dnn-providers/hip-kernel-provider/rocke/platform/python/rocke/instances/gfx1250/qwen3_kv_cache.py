# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 Qwen3-30B-A3B KV-cache kernels (dequant + append/RoPE).

Split out of the former ``attention_qwen3_day0`` scaffold: the attention
prefill/decode entry points now go through the unified attention instance
(``instances/common/attention_unified.py`` + ``attention_tiled_2d`` /
``attention_tiled_3d``). What remains here are the KV-cache-side kernels, which
are not part of the attention dispatcher:

* ``build_qwen3_kv_dequant_smoke`` — fp8e4m3 / bf8e5m2 KV read + dequant smoke.
* ``build_qwen3_kv_append_rope`` — KV append/update with optional RoPE and a
  bf16 / fp8e4m3 / bf8e5m2 quantized store into the paged cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.arch import ArchTarget
from ...core.ir import (
    BF16,
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
)
from ...helpers.attention import (
    PagedKvDescriptor,
    dequant_bf8x8_to_dtype,
    dequant_fp8x8_to_dtype,
)

_SUPPORTED_ARCH = "gfx1250"
_QWEN_KV_HEADS = 4
_QWEN_HEAD_DIM = 64
_QWEN_BLOCK_SIZE = 16

__all__ = [
    "Qwen3KvDequantSpec",
    "Qwen3KvAppendRopeSpec",
    "build_qwen3_kv_dequant_smoke",
    "build_qwen3_kv_append_rope",
]


def _dtype_ir(dtype: str) -> Type:
    if dtype == "fp16":
        return F16
    if dtype == "bf16":
        return BF16
    if dtype == "fp8e4m3":
        return FP8E4M3
    if dtype == "bf8e5m2":
        return BF8E5M2
    raise ValueError(f"unsupported dtype {dtype!r}")


def _require_gfx1250(arch: str) -> Tuple[bool, str]:
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if arch != _SUPPORTED_ARCH:
        return False, f"Qwen3 gfx1250 KV kernels require arch='gfx1250', got {arch!r}"
    if target.wave_size != 32 or target.family != "cdna":
        return False, f"gfx1250 contract expected CDNA wave32 target, got {arch}"
    return True, "supported"


@dataclass(frozen=True)
class Qwen3KvDequantSpec:
    """FP8/BF8 KV-cache read/dequant lowering smoke contract."""

    kv_storage_dtype: str
    output_dtype: str = "bf16"
    head_dim: int = _QWEN_HEAD_DIM
    name: str = "rocke_gfx1250_qwen3_kv_dequant"

    def __post_init__(self) -> None:
        if self.kv_storage_dtype not in ("fp8e4m3", "bf8e5m2"):
            raise ValueError("kv_storage_dtype must be 'fp8e4m3' or 'bf8e5m2'")
        if self.output_dtype not in ("fp16", "bf16"):
            raise ValueError("output_dtype must be 'fp16' or 'bf16'")
        if self.head_dim != _QWEN_HEAD_DIM:
            raise ValueError("Qwen3-30B-A3B head_dim must be 64")

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name, f"d{self.head_dim}", self.kv_storage_dtype, self.output_dtype
        )


def build_qwen3_kv_dequant_smoke(
    spec: Qwen3KvDequantSpec, *, arch: str = _SUPPORTED_ARCH
) -> KernelDef:
    ok, reason = _require_gfx1250(arch)
    if not ok:
        raise NotImplementedError(reason)
    storage_dtype = _dtype_ir(spec.kv_storage_dtype)
    out_dtype = _dtype_ir(spec.output_dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 32
    src = b.param("src", PtrType(storage_dtype, "global"), readonly=True, align=16)
    dst = b.param("dst", PtrType(out_dtype, "global"), writeonly=True, align=16)
    scale = b.param("scale", F32)
    lane = b.thread_id_x()
    base = b.mul(lane, b.const_i32(8))
    raw = b.global_load_vN(src, base, storage_dtype, 8, align=8)
    if spec.kv_storage_dtype == "fp8e4m3":
        deq = dequant_fp8x8_to_dtype(b, raw, scale, out_dtype)
    else:
        deq = dequant_bf8x8_to_dtype(b, raw, scale, out_dtype)
    for i in range(8):
        b.global_store(dst, b.add(base, b.const_i32(i)), b.vec_extract(deq, i), align=2)
    return b.kernel


@dataclass(frozen=True)
class Qwen3KvAppendRopeSpec:
    """KV append/update + RoPE integration scaffold."""

    input_dtype: str = "bf16"
    kv_storage_dtype: str = "bf16"
    head_dim: int = _QWEN_HEAD_DIM
    block_size: int = _QWEN_BLOCK_SIZE
    num_kv_heads: int = _QWEN_KV_HEADS
    use_rope: bool = True
    name: str = "rocke_gfx1250_qwen3_kv_append_rope"

    def __post_init__(self) -> None:
        if self.input_dtype not in ("fp16", "bf16"):
            raise ValueError("input_dtype must be 'fp16' or 'bf16'")
        if self.kv_storage_dtype not in ("bf16", "fp8e4m3", "bf8e5m2"):
            raise ValueError("kv_storage_dtype must be bf16/fp8e4m3/bf8e5m2")
        if self.head_dim != _QWEN_HEAD_DIM or self.block_size != _QWEN_BLOCK_SIZE:
            raise ValueError("Qwen3-30B-A3B KV scaffold is fixed at d64 block16")

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"d{self.head_dim}",
            f"b{self.block_size}",
            f"kvh{self.num_kv_heads}",
            self.input_dtype,
            f"kv{self.kv_storage_dtype}",
            "rope" if self.use_rope else "",
        )


def _quantize_for_kv_store(b: IRBuilder, v: object, storage_dtype: str, scale: object):
    if storage_dtype == "bf16":
        return b.cast_f32_to(v, BF16)
    scaled = b.fdiv(v, scale)
    if storage_dtype == "fp8e4m3":
        return b.cvt_f32_to_fp8(scaled)
    if storage_dtype == "bf8e5m2":
        return b.cvt_f32_to_bf8(scaled)
    raise ValueError(f"unsupported KV storage dtype {storage_dtype!r}")


def build_qwen3_kv_append_rope(
    spec: Qwen3KvAppendRopeSpec, *, arch: str = _SUPPORTED_ARCH
) -> KernelDef:
    ok, reason = _require_gfx1250(arch)
    if not ok:
        raise NotImplementedError(reason)
    in_dtype = _dtype_ir(spec.input_dtype)
    storage_dtype = _dtype_ir(spec.kv_storage_dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 1

    key_in = b.param("key_in", PtrType(in_dtype, "global"), readonly=True, align=16)
    value_in = b.param("value_in", PtrType(in_dtype, "global"), readonly=True, align=16)
    k_cache = b.param(
        "k_cache", PtrType(storage_dtype, "global"), writeonly=True, align=16
    )
    v_cache = b.param(
        "v_cache", PtrType(storage_dtype, "global"), writeonly=True, align=16
    )
    block_tables = b.param(
        "block_tables", PtrType(I32, "global"), readonly=True, align=16
    )
    slot_ids = b.param("slot_ids", PtrType(I32, "global"), readonly=True, align=16)
    cos = b.param("cos", PtrType(F32, "global"), readonly=True, align=16)
    sin = b.param("sin", PtrType(F32, "global"), readonly=True, align=16)
    k_scale = b.param("k_scale", F32)
    v_scale = b.param("v_scale", F32)

    token = b.block_id_x()
    kv_head = b.block_id_y()
    dim = b.block_id_z()
    slot = b.global_load_i32(slot_ids, token)
    logical_block = b.div(slot, b.const_i32(spec.block_size))
    token_in_block = b.mod(slot, b.const_i32(spec.block_size))
    physical_block = b.global_load_i32(block_tables, logical_block)

    in_base = b.add(
        b.mul(
            b.add(b.mul(token, b.const_i32(spec.num_kv_heads)), kv_head),
            b.const_i32(spec.head_dim),
        ),
        dim,
    )
    value_h = b.global_load(value_in, in_base, in_dtype, align=2)
    value_f = b.cast_to_f32(value_h)

    if spec.use_rope:
        pair_base = b.mul(b.div(dim, b.const_i32(2)), b.const_i32(2))
        pair_lane = b.div(dim, b.const_i32(2))
        even_idx = b.add(
            b.mul(
                b.add(b.mul(token, b.const_i32(spec.num_kv_heads)), kv_head),
                b.const_i32(spec.head_dim),
            ),
            pair_base,
        )
        odd_idx = b.add(even_idx, b.const_i32(1))
        even_f = b.cast_to_f32(b.global_load(key_in, even_idx, in_dtype, align=2))
        odd_f = b.cast_to_f32(b.global_load(key_in, odd_idx, in_dtype, align=2))
        trig_idx = b.add(b.mul(slot, b.const_i32(spec.head_dim // 2)), pair_lane)
        c = b.global_load(cos, trig_idx, F32, align=4)
        s = b.global_load(sin, trig_idx, F32, align=4)
        rot_even = b.fsub(b.fmul(even_f, c), b.fmul(odd_f, s))
        rot_odd = b.fadd(b.fmul(even_f, s), b.fmul(odd_f, c))
        is_odd = b.cmp_eq(b.mod(dim, b.const_i32(2)), b.const_i32(1))
        key_f = b.select(is_odd, rot_odd, rot_even)
    else:
        key_f = b.cast_to_f32(b.global_load(key_in, in_base, in_dtype, align=2))

    kv_desc = PagedKvDescriptor(
        block_size=spec.block_size,
        stride_0=spec.block_size * spec.num_kv_heads * spec.head_dim,
        stride_1=spec.num_kv_heads * spec.head_dim,
        stride_2=spec.head_dim,
        stride_3=1,
    )
    out_idx = kv_desc.offset(
        b,
        physical_block=physical_block,
        token_in_block=token_in_block,
        kv_head=kv_head,
        dim=dim,
    )
    k_store = _quantize_for_kv_store(b, key_f, spec.kv_storage_dtype, k_scale)
    v_store = _quantize_for_kv_store(b, value_f, spec.kv_storage_dtype, v_scale)
    store_align = 1 if spec.kv_storage_dtype != "bf16" else 2
    b.global_store(k_cache, out_idx, k_store, align=store_align)
    b.global_store(v_cache, out_idx, v_store, align=store_align)
    return b.kernel
