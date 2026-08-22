/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.io.h -- C99 port of rocke.helpers.io.
 *
 * Scope of THIS file: `io_ir_type` and `store_scalar_from_f32` are ported here
 * (the other I/O helpers in helpers/io.py are out of scope for this phase).
 * io_ir_type maps a dtype
 * spelling string to the canonical IR scalar type, byte-identically to the
 * Python:
 *
 *     def io_ir_type(dtype: str) -> Type:
 *         if dtype in ("f16", "fp16"): return F16
 *         if dtype == "bf16":          return BF16
 *         raise ValueError(...)
 *
 * The Python helper takes no IRBuilder and raises ValueError on an unsupported
 * dtype. C99 has no exceptions, so we expose two faithful spellings:
 *
 *   rocke_io_ir_type(dtype)
 *       Pure map. Returns the interned scalar singleton (rocke_f16()/rocke_bf16())
 *       for "f16"/"fp16"/"bf16"; returns NULL for anything else (the analog of
 *       "raise ValueError"). No error state is set because there is no builder.
 *
 *   rocke_b_io_ir_type(b, dtype)
 *       Builder-aware spelling matching the rest of the C port's sticky-error
 *       model: on an unsupported dtype it records ROCKE_ERR_VALUE + a message on
 *       the builder (mirroring the Python ValueError text) and returns NULL.
 *       This is the form GEMM/IO call sites should use so the ValueError
 *       propagates exactly like the other ported builder calls.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_IO_H
#define ROCKE_HELPER_ROCKE_HELPERS_IO_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pure dtype-string -> canonical scalar Type.
 *
 * Accepts "f16", "fp16" (alias) -> rocke_f16(); "bf16" -> rocke_bf16().
 * Returns NULL for any other value (the Python ValueError path). f8/i8 paths
 * deliberately do NOT resolve here -- their compute dtype isn't f32 and they go
 * through their own helpers, exactly as in Python. */
const rocke_type_t* rocke_io_ir_type(const char* dtype);

/* Builder-aware variant. Same mapping as rocke_io_ir_type, but on an unsupported
 * dtype it sets the builder's sticky error (ROCKE_ERR_VALUE) with the
 * Python-matching message and returns NULL. If the builder is already in an
 * error state it is a no-op returning NULL, like every other rocke_b_* call. */
const rocke_type_t* rocke_b_io_ir_type(rocke_ir_builder_t* b, const char* dtype);

/* C99 port of rocke.helpers.io.store_scalar_from_f32:
 *
 *     def store_scalar_from_f32(b, ptr, idx, value_f32, *, dtype) -> None:
 *         target = io_ir_type(dtype)
 *         b.global_store(ptr, idx, b.cast_f32_to(value_f32, target))
 *
 * Truncates an f32 value to `dtype` ("f16"/"fp16"/"bf16") and stores it as one
 * scalar global store. Mirrors the Python builder-call sequence exactly:
 *   1. resolve `target` via io_ir_type (builder-aware, so an unsupported dtype
 *      records the same ValueError text on the sticky-error model and returns),
 *   2. cast_f32_to(value_f32, target),
 *   3. global_store(ptr, idx, <cast>) with the default align (Python passes no
 *      align kwarg -> align=0 here).
 * No-op when the builder is already in an error state, like every rocke_b_* call. */
void rocke_b_store_scalar_from_f32(rocke_ir_builder_t* b,
                                   rocke_value_t* ptr,
                                   rocke_value_t* idx,
                                   rocke_value_t* value_f32,
                                   const char* dtype);

/* C99 port of rocke.helpers.io.load_scalar:
 *
 *     def load_scalar(b, ptr, idx, *, dtype) -> Value:
 *         if dtype in ("f16", "fp16"): return b.global_load_f16(ptr, idx)
 *         if dtype == "bf16":          return b.global_load_bf16(ptr, idx)
 *         raise ValueError(f"unsupported I/O dtype {dtype!r}")
 *
 * One scalar global load returning a value in the native dtype. The Python
 * helper passes no align kwarg -> align defaults; this port passes 0 to match
 * the Python global_load_f16 / global_load_bf16 default. On an unsupported
 * dtype it sets the same ValueError text on the sticky-error model and returns
 * NULL. No-op (NULL) when the builder is already in an error state. */
rocke_value_t* rocke_b_load_scalar(rocke_ir_builder_t* b,
                                   rocke_value_t* ptr,
                                   rocke_value_t* idx,
                                   const char* dtype);

/* C99 port of rocke.helpers.io.load_scalar_as_f32:
 *
 *     def load_scalar_as_f32(b, ptr, idx, *, dtype) -> Value:
 *         return b.cast_to_f32(load_scalar(b, ptr, idx, dtype=dtype))
 *
 * One scalar global load promoted to f32. Returns NULL on an unsupported dtype
 * or an already-errored builder. */
rocke_value_t* rocke_b_load_scalar_as_f32(rocke_ir_builder_t* b,
                                          rocke_value_t* ptr,
                                          rocke_value_t* idx,
                                          const char* dtype);

/* C99 port of rocke.helpers.io.load_vec:
 *
 *     def load_vec(b, ptr, idx, *, dtype, n) -> Value:
 *         if n not in (2, 4, 8): raise ValueError(...)
 *         ty = io_ir_type(dtype)
 *         if dtype in ("f16", "fp16"): return b.global_load_vN_f16(ptr, idx, n)
 *         return b.global_load_vN(ptr, idx, ty, n)
 *
 * Vectorised global load of `n` consecutive elements (n in {2,4,8}). The Python
 * helper always resolves io_ir_type(dtype) first (so an unsupported dtype
 * raises before the n check is reached on the f16/bf16 paths) -- but the n check
 * comes first in the Python, so this port checks n first as well to keep the
 * raised-error identity. Returns NULL on bad n / bad dtype / errored builder. */
rocke_value_t* rocke_b_load_vec(
    rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* idx, const char* dtype, int n);

/* C99 port of rocke.helpers.io.load_vec_as_f32:
 *
 *     def load_vec_as_f32(b, ptr, idx, *, dtype, n) -> list[Value]:
 *         v = load_vec(b, ptr, idx, dtype=dtype, n=n)
 *         return [b.cast_to_f32(b.vec_extract(v, i)) for i in range(n)]
 *
 * Vectorised load + per-lane f32 promotion. Writes `n` scalar f32 Values into
 * caller-supplied `out` (which must have room for `n` entries). Returns 1 on
 * success, 0 on failure (bad n / bad dtype / errored builder); `out` is left
 * untouched on failure. */
int rocke_b_load_vec_as_f32(rocke_ir_builder_t* b,
                            rocke_value_t* ptr,
                            rocke_value_t* idx,
                            const char* dtype,
                            int n,
                            rocke_value_t** out);

/* C99 port of rocke.helpers.io.load_lane_slice_f32:
 *
 *     def load_lane_slice_f32(b, ptr, row_base, lane_d_base, *, dtype, ept):
 *         if ept in _VEC_WIDTHS:
 *             return load_vec_as_f32(b, ptr, b.add(row_base, lane_d_base),
 *                                    dtype=dtype, n=ept)
 *         return [load_scalar_as_f32(
 *                     b, ptr,
 *                     b.add(row_base, b.add(lane_d_base, b.const_i32(k))),
 *                     dtype=dtype)
 *                 for k in range(ept)]
 *
 * Loads this lane's `ept` consecutive elements as `ept` f32 Values into
 * caller-supplied `out` (room for `ept`). One vectorised path for
 * ept in {2,4,8}; scalar fallback otherwise (ept==1 / ept==3). _VEC_WIDTHS is
 * (2, 4, 8). Returns 1 on success, 0 on failure (errored builder / bad dtype). */
int rocke_b_load_lane_slice_f32(rocke_ir_builder_t* b,
                                rocke_value_t* ptr,
                                rocke_value_t* row_base,
                                rocke_value_t* lane_d_base,
                                const char* dtype,
                                int ept,
                                rocke_value_t** out);

/* C99 port of rocke.helpers.io.store_vec:
 *
 *     def store_vec(b, ptr, idx, value, *, n) -> None:
 *         if n not in (2, 4, 8): raise ValueError(...)
 *         b.global_store_vN(ptr, idx, value, n)
 *
 * Vectorised global store. `value` must already be a <n x T> vector in the
 * target dtype. The Python helper passes no align kwarg -> align defaults;
 * this port passes 0 to match global_store_vN's default. No-op on bad n /
 * errored builder. */
void rocke_b_store_vec(
    rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* idx, rocke_value_t* value, int n);

/* C99 port of rocke.helpers.io.pack_f32_to:
 *
 *     def pack_f32_to(b, scalars_f32, *, dtype) -> Value:
 *         target = io_ir_type(dtype)
 *         casts = [b.cast_f32_to(v, target) for v in scalars_f32]
 *         return b.vec_pack(casts, target)
 *
 * Truncs a list of f32 scalars (count `n`) to `dtype` and packs into a vector.
 * Returns the packed <n x dtype> vector, or NULL on an unsupported dtype /
 * errored builder. */
rocke_value_t* rocke_b_pack_f32_to(rocke_ir_builder_t* b,
                                   rocke_value_t* const* scalars_f32,
                                   int n,
                                   const char* dtype);

/* C99 port of rocke.helpers.io.vector_row_copy:
 *
 *     def vector_row_copy(b, *, src, dst, src_base, dst_base, H, dtype,
 *                         vec_bytes=16) -> None:
 *         ty = io_ir_type(dtype)
 *         elem_bytes = 2
 *         vec = vec_bytes // elem_bytes
 *         if vec not in (2, 4, 8): raise ValueError(...)
 *         n_chunks = H // vec
 *         for c in range(n_chunks):
 *             d = c * vec
 *             src_addr = b.add(src_base, b.const_i32(d))
 *             dst_addr = b.add(dst_base, b.const_i32(d))
 *             v = b.global_load_vN(src, src_addr, ty, vec, align=vec*elem_bytes)
 *             b.global_store_vN(dst, dst_addr, v, vec, align=vec*elem_bytes)
 *         for d in range(n_chunks * vec, H):
 *             s = load_scalar_as_f32(b, src, b.add(src_base, b.const_i32(d)),
 *                                    dtype=dtype)
 *             store_scalar_from_f32(b, dst, b.add(dst_base, b.const_i32(d)), s,
 *                                   dtype=dtype)
 *
 * Vectorised row copy along a head / hidden dim. elem_bytes is fixed at 2
 * (f16/bf16 storage). `vec_bytes` defaults to 16 in Python; this port takes it
 * explicitly (pass 16 for the default). No-op on bad vec mapping / unsupported
 * dtype / errored builder. */
void rocke_b_vector_row_copy(rocke_ir_builder_t* b,
                             rocke_value_t* src,
                             rocke_value_t* dst,
                             rocke_value_t* src_base,
                             rocke_value_t* dst_base,
                             int H,
                             const char* dtype,
                             int vec_bytes);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_IO_H */
