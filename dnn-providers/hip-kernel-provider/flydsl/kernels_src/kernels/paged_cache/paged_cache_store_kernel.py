# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Ported from: https://github.com/ROCm/FlyDSL
#   commit:   226cde8d2a84bc2687c8d69587607fddf0c16f2f  (v0.5.0)
#   path:     kernels/attention/fused_rope_cache_kernel.py
#             (the KV-cache write half of fused_qk_rope_reshape_and_cache,
#             upstream lines ~250-423)
#   compiler: flydsl 0.3.4 (pip wheel)
#
# Modifications:
#   - Only the cache write is kept: no RoPE, no Q/K outputs, one operand (x)
#     instead of K and V. The rope half is a separate op (rope/).
#   - Written against the FlyDSL 0.3.4 API: raw pointer kernel arguments and
#     kernels.common.buffer_ops resources, not v0.5.0 layout-API tensors.
#   - The container is hipDNN's SDPA paged layout [num_blocks, H, block_size, D],
#     not upstream's flash [num_blocks, block_size, KH, D] or ATOM x-packed one.
#     Every operand is addressed through runtime element strides, so x may be
#     token-packed [T, H, D] or SDPA-native [B, H, S, D] with any strides, and the
#     container may be strided too.
#   - One block per token rather than per (head, token): the block's threads
#     stride over the token's H * D elements, so head_dim is a runtime argument
#     instead of a compile-time VEC_WIDTH = ceil(D / WARP_SIZE) split. A
#     16-byte vector path runs when the host says the operands allow it
#     (`VecOk`); otherwise a scalar path serves any strides and any D.
#   - Slots may be int32 or int64 (runtime `SlotIs64`). A slot that is negative,
#     not representable in 32 bits, or at/past `NumSlots` is skipped rather than
#     written out of bounds.
#   - Scale is multiplied in (hipDNN's `cast(x * scale)`), where upstream divides
#     by its k/v scale (`* rcp(scale)`). It is read from a device f32 when
#     `ScaleFromPtr` is set, otherwise taken from the `ScaleValue` argument, so a
#     graph without a scale passes 1.0.
#   - FP8 caches clamp to the format's finite range before the hardware
#     conversion (NaN passes through), so overflow saturates like hipDNN's
#     reference instead of producing NaN. gfx942's v_cvt_pk_{fp8,bf8}_f32 emit
#     the FNUZ formats, gfx950's the OCP ones; the cache dtype names which.
#
# Upstream-first: the strided [nb, H, bs, D] layout would be a natural
# `flash_layout` sibling upstream; this copy exists so the provider is not
# blocked on that.

"""Paged KV-cache store kernel builder using the @flyc.kernel API.

For every token (b, s) with slot = SlotMapping[b, s] in [0, NumSlots):

    Cache[slot / BlockSize, h, slot % BlockSize, d] = cast(X[b, h, s, d] * scale)

for all h < Heads, d < HeadDim. Every other cache element is left untouched.

Grid: (S, B, 1) -- one block per token. Token-packed x [T, H, D] is B = 1,
S = T with a zero batch stride.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from kernels.common import buffer_ops
from kernels.common.kernels_common import dtype_to_elem_type

BLOCK_THREADS = 256

# The vector path moves 16 bytes of x per thread step.
VECTOR_BYTES = 16

X_DTYPES = ("bf16", "f16", "f32")
# Container dtypes beyond x's own: the FP8 formats the target's hardware
# conversion produces. The name says which encoding the bytes are in.
FP8_CACHE_DTYPES = {
    "fp8_e4m3": ("fp8", 448.0),
    "fp8_e5m2": ("bf8", 57344.0),
    "fp8_e4m3_fnuz": ("fp8", 240.0),
    "fp8_e5m2_fnuz": ("bf8", 57344.0),
}

_ELEM_BYTES = {"bf16": 2, "f16": 2, "f32": 4}


def cache_elem_bytes(cache_dtype: str) -> int:
    return 1 if cache_dtype in FP8_CACHE_DTYPES else _ELEM_BYTES[cache_dtype]


def build_paged_cache_store_module(
    dtype_str: str,
    cache_dtype_str: str,
    BLOCK_THREADS: int = BLOCK_THREADS,
):
    """Build the store for x in `dtype_str` and a container in `cache_dtype_str`.

    `cache_dtype_str` is x's own dtype, or one of FP8_CACHE_DTYPES.
    """
    if dtype_str not in X_DTYPES:
        raise ValueError(f"dtype_str must be one of {X_DTYPES}, got {dtype_str!r}")
    is_fp8 = cache_dtype_str in FP8_CACHE_DTYPES
    if not is_fp8 and cache_dtype_str != dtype_str:
        raise ValueError(
            f"cache_dtype_str must be {dtype_str!r} or one of "
            f"{tuple(FP8_CACHE_DTYPES)}, got {cache_dtype_str!r}"
        )

    x_bytes = _ELEM_BYTES[dtype_str]
    c_bytes = cache_elem_bytes(cache_dtype_str)
    if is_fp8:
        fp8_kind, fp8_max = FP8_CACHE_DTYPES[cache_dtype_str]
        # v_cvt_pk_{fp8,bf8}_f32 emits FNUZ on gfx94x and OCP on gfx95x; gfx90a
        # has no FP8 conversion at all. An object whose name promised the other
        # encoding would write wrong bytes without any error.
        arch = str(get_rocm_arch())
        if arch.startswith("gfx94"):
            native_fnuz = True
        elif arch.startswith("gfx95"):
            native_fnuz = False
        else:
            raise ValueError(f"no FP8 cache conversion on {arch}")
        if cache_dtype_str.endswith("_fnuz") != native_fnuz:
            raise ValueError(
                f"{arch} converts to {'FNUZ' if native_fnuz else 'OCP'} FP8, "
                f"not {cache_dtype_str}"
            )

    VEC = VECTOR_BYTES // x_bytes

    def _byte_rsrc(ptr, base_bytes=None, num_records_bytes=None):
        """A byte-addressed buffer resource at `ptr + base_bytes` (64-bit)."""
        addr = fx.Int64(fx.ptrtoint(ptr))
        if base_bytes is not None:
            addr = addr + fx.Int64(base_bytes)
        return buffer_ops.create_buffer_resource_from_addr(
            addr, num_records_bytes=num_records_bytes
        )

    def _load(rsrc, byte_off, dtype, width):
        nbits = dtype.width * width
        src = fx.make_view(rsrc + fx.Int32(byte_off), fx.make_layout(nbits // 8, 1))
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy(nbits), fx.Int8)
        reg = fx.make_rmem_tensor(fx.make_layout(width, 1), dtype)
        fx.copy(atom, src, reg)
        return fx.memref_load_vec(reg)

    def _store(vec_val, rsrc, byte_off, dtype, width):
        nbits = dtype.width * width
        dst = fx.make_view(rsrc + fx.Int32(byte_off), fx.make_layout(nbits // 8, 1))
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy(nbits), fx.Int8)
        reg = fx.make_rmem_tensor(fx.make_layout(width, 1), dtype)
        fx.memref_store_vec(vec_val, reg)
        fx.copy(atom, reg, dst)

    def _clamp_fp8(v):
        # Comparisons with NaN are false, so NaN reaches the conversion unchanged.
        hi = fx.Float32(fp8_max)
        lo = fx.Float32(-fp8_max)
        v = (v > hi).select(hi, v)
        return (v < lo).select(lo, v)

    def _cvt_pk(a, b, old, word_sel):
        op = _rocdl.cvt_pk_fp8_f32 if fp8_kind == "fp8" else _rocdl.cvt_pk_bf8_f32
        return fx.Int32(
            op(
                res=T.i32,
                src_a=fx.Float32(a).ir_value(),
                src_b=fx.Float32(b).ir_value(),
                old=fx.Int32(old).ir_value(),
                word_sel=word_sel,
            )
        )

    @flyc.kernel
    def paged_cache_store_kernel(
        X: fx.Pointer,
        SlotMapping: fx.Pointer,
        Cache: fx.Pointer,
        Scale: fx.Pointer,
        XStrideB: fx.Int64,
        XStrideS: fx.Int64,
        SlotStrideB: fx.Int64,
        SlotStrideS: fx.Int64,
        CStrideBlock: fx.Int64,
        CStrideSlot: fx.Int64,
        Heads: fx.Int32,
        HeadDim: fx.Int32,
        XStrideH: fx.Int32,
        XStrideD: fx.Int32,
        CStrideH: fx.Int32,
        CStrideD: fx.Int32,
        BlockSize: fx.Int32,
        NumSlots: fx.Int32,
        SlotIs64: fx.Int32,
        ScaleFromPtr: fx.Int32,
        VecOk: fx.Int32,
        ScaleValue: fx.Float32,
    ):
        s_idx = fx.block_idx.x
        b_idx = fx.block_idx.y
        tid = fx.thread_idx.x

        x_dtype = dtype_to_elem_type(dtype_str)

        # --- Slot (wave-uniform) ---
        slot_elem = fx.Int64(b_idx) * SlotStrideB + fx.Int64(s_idx) * SlotStrideS
        slot_dwords = fx.Int64(1) + fx.Int64(SlotIs64)
        slot_rsrc = _byte_rsrc(SlotMapping, slot_elem * slot_dwords * fx.Int64(4))
        lo = _load(slot_rsrc, fx.Int32(0), fx.Int32, 1)[0]
        hi = _load(slot_rsrc, SlotIs64 * fx.Int32(4), fx.Int32, 1)[0]
        # int32: hi re-reads lo. int64: hi must be 0, i.e. the value fits 31 bits.
        hi_ok = hi == (SlotIs64 != fx.Int32(0)).select(fx.Int32(0), lo)
        valid = (lo >= fx.Int32(0)) & (lo < NumSlots) & hi_ok

        if valid:
            blk = lo // BlockSize
            off = lo % BlockSize

            # --- Scale: device scalar, or the by-value argument ---
            # A zero-byte resource when unused: the load is out of bounds and
            # returns 0 without touching memory, so Scale may be any pointer.
            scale_rsrc = _byte_rsrc(Scale, None, fx.Int64(ScaleFromPtr) * fx.Int64(4))
            scale_dev = _load(scale_rsrc, fx.Int32(0), fx.Float32, 1)[0]
            scale = (ScaleFromPtr != fx.Int32(0)).select(scale_dev, ScaleValue)

            x_base = (
                fx.Int64(b_idx) * XStrideB + fx.Int64(s_idx) * XStrideS
            ) * fx.Int64(x_bytes)
            c_base = (
                fx.Int64(blk) * CStrideBlock + fx.Int64(off) * CStrideSlot
            ) * fx.Int64(c_bytes)
            x_rsrc = _byte_rsrc(X, x_base)
            c_rsrc = _byte_rsrc(Cache, c_base)

            def convert(vals):
                """f32 values -> the cache's element vector (or packed i32s for fp8)."""
                scaled = [v * scale for v in vals]
                if const_expr(is_fp8):
                    n = len(scaled)
                    clamped = [_clamp_fp8(v) for v in scaled]
                    if const_expr(n == 1):
                        word = _cvt_pk(clamped[0], fx.Float32(0.0), fx.Int32(0), False)
                        return Vec.from_elements([fx.Int8(word)], fx.Int8), fx.Int8, 1
                    words = []
                    for i in range_constexpr(n // 4):
                        w = _cvt_pk(
                            clamped[4 * i], clamped[4 * i + 1], fx.Int32(0), False
                        )
                        w = _cvt_pk(clamped[4 * i + 2], clamped[4 * i + 3], w, True)
                        words.append(w)
                    return Vec.from_elements(words, fx.Int32), fx.Int32, n // 4
                cache_dtype = dtype_to_elem_type(cache_dtype_str)
                if const_expr(cache_dtype_str == "f32"):
                    elems = scaled
                else:
                    elems = [v.to(cache_dtype) for v in scaled]
                return Vec.from_elements(elems, cache_dtype), cache_dtype, len(elems)

            if VecOk != fx.Int32(0):
                chunks_per_head = HeadDim // fx.Int32(VEC)
                total = Heads * chunks_per_head
                for base, _state in range(0, total, BLOCK_THREADS, init=[]):
                    c = tid + fx.Int32(base)
                    if c < total:
                        h = c // chunks_per_head
                        d = (c % chunks_per_head) * fx.Int32(VEC)
                        xv = _load(
                            x_rsrc, (h * XStrideH + d) * fx.Int32(x_bytes), x_dtype, VEC
                        )
                        vals = [
                            (
                                xv[i]
                                if const_expr(dtype_str == "f32")
                                else xv[i].to(fx.Float32)
                            )
                            for i in range_constexpr(VEC)
                        ]
                        out, out_dtype, width = convert(vals)
                        _store(
                            out,
                            c_rsrc,
                            (h * CStrideH + d) * fx.Int32(c_bytes),
                            out_dtype,
                            width,
                        )
            else:
                total = Heads * HeadDim
                for base, _state in range(0, total, BLOCK_THREADS, init=[]):
                    c = tid + fx.Int32(base)
                    if c < total:
                        h = c // HeadDim
                        d = c % HeadDim
                        xe = _load(
                            x_rsrc,
                            (h * XStrideH + d * XStrideD) * fx.Int32(x_bytes),
                            x_dtype,
                            1,
                        )[0]
                        xf = xe if const_expr(dtype_str == "f32") else xe.to(fx.Float32)
                        out, out_dtype, width = convert([xf])
                        _store(
                            out,
                            c_rsrc,
                            (h * CStrideH + d * CStrideD) * fx.Int32(c_bytes),
                            out_dtype,
                            width,
                        )

    @flyc.jit
    def launch_paged_cache_store(
        X: fx.Pointer,
        SlotMapping: fx.Pointer,
        Cache: fx.Pointer,
        Scale: fx.Pointer,
        XStrideB: fx.Int64,
        XStrideS: fx.Int64,
        SlotStrideB: fx.Int64,
        SlotStrideS: fx.Int64,
        CStrideBlock: fx.Int64,
        CStrideSlot: fx.Int64,
        Heads: fx.Int32,
        HeadDim: fx.Int32,
        XStrideH: fx.Int32,
        XStrideD: fx.Int32,
        CStrideH: fx.Int32,
        CStrideD: fx.Int32,
        BlockSize: fx.Int32,
        NumSlots: fx.Int32,
        SlotIs64: fx.Int32,
        ScaleFromPtr: fx.Int32,
        VecOk: fx.Int32,
        ScaleValue: fx.Float32,
        SeqLen: fx.Int32,
        Batch: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = paged_cache_store_kernel(
            X,
            SlotMapping,
            Cache,
            Scale,
            XStrideB,
            XStrideS,
            SlotStrideB,
            SlotStrideS,
            CStrideBlock,
            CStrideSlot,
            Heads,
            HeadDim,
            XStrideH,
            XStrideD,
            CStrideH,
            CStrideD,
            BlockSize,
            NumSlots,
            SlotIs64,
            ScaleFromPtr,
            VecOk,
            ScaleValue,
        )
        launcher.launch(
            grid=(SeqLen, Batch, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_paged_cache_store
