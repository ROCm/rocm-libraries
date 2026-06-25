// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_ck_dsl.helpers.io.c -- C99 port of ck_dsl.helpers.io.io_ir_type,
 * store_scalar_from_f32, load_scalar, load_scalar_as_f32, load_vec,
 * load_vec_as_f32, load_lane_slice_f32, store_vec, pack_f32_to and
 * vector_row_copy.
 *
 * Faithful translation of:
 *
 *     def io_ir_type(dtype: str) -> Type:
 *         if dtype in ("f16", "fp16"): return F16
 *         if dtype == "bf16":          return BF16
 *         raise ValueError(
 *             f"unsupported I/O dtype {dtype!r}; expected f16/fp16/bf16")
 *
 * Mapping invariants (must stay byte-identical to the Python so downstream IR is
 * identical):
 *   "f16"  -> ckc_f16()    (Python F16)
 *   "fp16" -> ckc_f16()    (alias; Python F16)
 *   "bf16" -> ckc_bf16()   (Python BF16)
 *   else   -> NULL         (Python ValueError)
 */

#include "ckc/helper_ck_dsl.helpers.io.h"

#include <string.h>

#include "ckc/arena.h" /* ckc_arena_alloc */
#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */

/* Power-of-two vector widths the DSL's global_load_vN covers. Mirrors the
 * module-level _VEC_WIDTHS = (2, 4, 8) in ck_dsl.helpers.io. */
static int ckc_io_is_vec_width(int n)
{
    return n == 2 || n == 4 || n == 8;
}

const ckc_type_t* ckc_io_ir_type(const char* dtype)
{
    if(dtype == NULL)
    {
        return NULL;
    }
    /* `dtype in ("f16", "fp16")` -> F16 */
    if(strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0)
    {
        return ckc_f16();
    }
    /* `dtype == "bf16"` -> BF16 */
    if(strcmp(dtype, "bf16") == 0)
    {
        return ckc_bf16();
    }
    /* Python: raise ValueError. No builder here, so signal via NULL. */
    return NULL;
}

const ckc_type_t* ckc_b_io_ir_type(ckc_ir_builder_t* b, const char* dtype)
{
    const ckc_type_t* ty;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!ckc_i_live(b))
    {
        return NULL;
    }

    ty = ckc_io_ir_type(dtype);
    if(ty == NULL)
    {
        /* Mirror the Python ValueError, including the {dtype!r} single-quote
         * repr for the (non-NULL) string case. NULL is reported as "None" to
         * match Python's repr(None). */
        return (const ckc_type_t*)ckc_i_set_err(
            b,
            CKC_ERR_VALUE,
            "unsupported I/O dtype %s%s%s; expected f16/fp16/bf16",
            dtype ? "'" : "",
            dtype ? dtype : "None",
            dtype ? "'" : "");
    }
    return ty;
}

void ckc_b_store_scalar_from_f32(ckc_ir_builder_t* b,
                                 ckc_value_t* ptr,
                                 ckc_value_t* idx,
                                 ckc_value_t* value_f32,
                                 const char* dtype)
{
    const ckc_type_t* target;
    ckc_value_t* cast;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!ckc_i_live(b))
    {
        return;
    }

    /* Python: target = io_ir_type(dtype). Use the builder-aware variant so an
     * unsupported dtype records the same ValueError on the sticky-error model;
     * on failure it returns NULL and we stop (Python would have raised). */
    target = ckc_b_io_ir_type(b, dtype);
    if(target == NULL)
    {
        return;
    }

    /* Python: b.global_store(ptr, idx, b.cast_f32_to(value_f32, target)).
     * align defaults to 0 (Python global_store takes no align kwarg here). */
    cast = ckc_b_cast_f32_to(b, value_f32, target);
    ckc_b_global_store(b, ptr, idx, cast, 0);
}

ckc_value_t*
    ckc_b_load_scalar(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, const char* dtype)
{
    if(!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python: if dtype in ("f16", "fp16"): return b.global_load_f16(ptr, idx) */
    if(dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
    {
        /* Python passes no align kwarg -> default. 0 selects the builder default. */
        return ckc_b_global_load_f16(b, ptr, idx, 0);
    }
    /* Python: if dtype == "bf16": return b.global_load_bf16(ptr, idx) */
    if(dtype != NULL && strcmp(dtype, "bf16") == 0)
    {
        return ckc_b_global_load_bf16(b, ptr, idx, 0);
    }
    /* Python: raise ValueError(f"unsupported I/O dtype {dtype!r}"). Note this
     * message has NO ", expected ..." suffix -- it is load_scalar's own text,
     * distinct from io_ir_type's. Mirror it exactly. */
    return (ckc_value_t*)ckc_i_set_err(b,
                                       CKC_ERR_VALUE,
                                       "unsupported I/O dtype %s%s%s",
                                       dtype ? "'" : "",
                                       dtype ? dtype : "None",
                                       dtype ? "'" : "");
}

ckc_value_t* ckc_b_load_scalar_as_f32(ckc_ir_builder_t* b,
                                      ckc_value_t* ptr,
                                      ckc_value_t* idx,
                                      const char* dtype)
{
    ckc_value_t* v;

    if(!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python: return b.cast_to_f32(load_scalar(b, ptr, idx, dtype=dtype)).
     * load_scalar records the ValueError on an unsupported dtype and returns
     * NULL; cast_to_f32 of NULL on an errored builder is itself a NULL no-op. */
    v = ckc_b_load_scalar(b, ptr, idx, dtype);
    if(v == NULL)
    {
        return NULL;
    }
    return ckc_b_cast_to_f32(b, v);
}

ckc_value_t* ckc_b_load_vec(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, const char* dtype, int n)
{
    const ckc_type_t* ty;

    if(!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python checks n FIRST: if n not in (2,4,8): raise ValueError(...). */
    if(!ckc_io_is_vec_width(n))
    {
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "load_vec n must be 2/4/8 (got %d); use load_scalar for n=1", n);
    }
    /* Python: ty = io_ir_type(dtype). Builder-aware: an unsupported dtype
     * records io_ir_type's ValueError and returns NULL here. */
    ty = ckc_b_io_ir_type(b, dtype);
    if(ty == NULL)
    {
        return NULL;
    }
    /* Python: if dtype in ("f16","fp16"): return b.global_load_vN_f16(ptr,idx,n)
     *         return b.global_load_vN(ptr, idx, ty, n)
     * Python passes no align -> default (0). */
    if(strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0)
    {
        return ckc_b_global_load_vN_f16(b, ptr, idx, n, 0);
    }
    return ckc_b_global_load_vN(b, ptr, idx, ty, n, 0);
}

int ckc_b_load_vec_as_f32(ckc_ir_builder_t* b,
                          ckc_value_t* ptr,
                          ckc_value_t* idx,
                          const char* dtype,
                          int n,
                          ckc_value_t** out)
{
    ckc_value_t* v;
    int i;

    if(!ckc_i_live(b))
    {
        return 0;
    }
    if(out == NULL)
    {
        return 0;
    }

    /* Python: v = load_vec(b, ptr, idx, dtype=dtype, n=n) */
    v = ckc_b_load_vec(b, ptr, idx, dtype, n);
    if(v == NULL)
    {
        return 0;
    }
    /* Python: return [b.cast_to_f32(b.vec_extract(v, i)) for i in range(n)] */
    for(i = 0; i < n; ++i)
    {
        out[i] = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, v, i));
    }
    return 1;
}

int ckc_b_load_lane_slice_f32(ckc_ir_builder_t* b,
                              ckc_value_t* ptr,
                              ckc_value_t* row_base,
                              ckc_value_t* lane_d_base,
                              const char* dtype,
                              int ept,
                              ckc_value_t** out)
{
    int k;

    if(!ckc_i_live(b))
    {
        return 0;
    }
    if(out == NULL)
    {
        return 0;
    }

    /* Python: if ept in _VEC_WIDTHS:
     *             return load_vec_as_f32(
     *                 b, ptr, b.add(row_base, lane_d_base), dtype=dtype, n=ept) */
    if(ckc_io_is_vec_width(ept))
    {
        return ckc_b_load_vec_as_f32(b, ptr, ckc_b_add(b, row_base, lane_d_base), dtype, ept, out);
    }
    /* Python scalar fallback:
     *   return [load_scalar_as_f32(
     *               b, ptr,
     *               b.add(row_base, b.add(lane_d_base, b.const_i32(k))),
     *               dtype=dtype)
     *           for k in range(ept)] */
    for(k = 0; k < ept; ++k)
    {
        ckc_value_t* addr
            = ckc_b_add(b, row_base, ckc_b_add(b, lane_d_base, ckc_b_const_i32(b, k)));
        out[k] = ckc_b_load_scalar_as_f32(b, ptr, addr, dtype);
        if(out[k] == NULL)
        {
            /* load_scalar_as_f32 recorded the ValueError already. */
            return 0;
        }
    }
    return 1;
}

void ckc_b_store_vec(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, ckc_value_t* value, int n)
{
    if(!ckc_i_live(b))
    {
        return;
    }

    /* Python: if n not in (2,4,8): raise ValueError(...) */
    if(!ckc_io_is_vec_width(n))
    {
        (void)ckc_i_set_err(
            b, CKC_ERR_VALUE, "store_vec n must be 2/4/8 (got %d); use store_scalar for n=1", n);
        return;
    }
    /* Python: b.global_store_vN(ptr, idx, value, n). No align -> default (0). */
    ckc_b_global_store_vN(b, ptr, idx, value, n, 0);
}

ckc_value_t* ckc_b_pack_f32_to(ckc_ir_builder_t* b,
                               ckc_value_t* const* scalars_f32,
                               int n,
                               const char* dtype)
{
    const ckc_type_t* target;
    ckc_value_t** casts;
    int i;

    if(!ckc_i_live(b))
    {
        return NULL;
    }

    /* Python: target = io_ir_type(dtype). Builder-aware -> records ValueError
     * on an unsupported dtype and returns NULL. */
    target = ckc_b_io_ir_type(b, dtype);
    if(target == NULL)
    {
        return NULL;
    }
    /* Python: casts = [b.cast_f32_to(v, target) for v in scalars_f32].
     * Build the cast list in the arena so vec_pack gets a contiguous array. */
    casts = (ckc_value_t**)ckc_arena_alloc(&b->arena, (size_t)n * sizeof(*casts));
    if(casts == NULL)
    {
        return NULL;
    }
    for(i = 0; i < n; ++i)
    {
        casts[i] = ckc_b_cast_f32_to(b, scalars_f32[i], target);
    }
    /* Python: return b.vec_pack(casts, target). */
    return ckc_b_vec_pack(b, casts, n, target);
}

void ckc_b_vector_row_copy(ckc_ir_builder_t* b,
                           ckc_value_t* src,
                           ckc_value_t* dst,
                           ckc_value_t* src_base,
                           ckc_value_t* dst_base,
                           int H,
                           const char* dtype,
                           int vec_bytes)
{
    const ckc_type_t* ty;
    const int elem_bytes = 2; /* f16 / bf16 storage width */
    int vec;
    int n_chunks;
    int c;
    int d;

    if(!ckc_i_live(b))
    {
        return;
    }

    /* Python: ty = io_ir_type(dtype). Builder-aware -> records ValueError on an
     * unsupported dtype and returns NULL. */
    ty = ckc_b_io_ir_type(b, dtype);
    if(ty == NULL)
    {
        return;
    }
    /* Python: vec = vec_bytes // elem_bytes; if vec not in (2,4,8): raise. */
    vec = vec_bytes / elem_bytes;
    if(!ckc_io_is_vec_width(vec))
    {
        (void)ckc_i_set_err(b,
                            CKC_ERR_VALUE,
                            "vector_row_copy: vec_bytes %d maps to vec=%d; expected 4/8/16-byte "
                            "aligned",
                            vec_bytes,
                            vec);
        return;
    }
    /* Python: n_chunks = H // vec
     *         for c in range(n_chunks):
     *             d = c * vec
     *             src_addr = b.add(src_base, b.const_i32(d))
     *             dst_addr = b.add(dst_base, b.const_i32(d))
     *             v = b.global_load_vN(src, src_addr, ty, vec, align=vec*elem_bytes)
     *             b.global_store_vN(dst, dst_addr, v, vec, align=vec*elem_bytes) */
    n_chunks = H / vec;
    for(c = 0; c < n_chunks; ++c)
    {
        int dd = c * vec;
        ckc_value_t* src_addr = ckc_b_add(b, src_base, ckc_b_const_i32(b, dd));
        ckc_value_t* dst_addr = ckc_b_add(b, dst_base, ckc_b_const_i32(b, dd));
        ckc_value_t* v = ckc_b_global_load_vN(b, src, src_addr, ty, vec, vec * elem_bytes);
        ckc_b_global_store_vN(b, dst, dst_addr, v, vec, vec * elem_bytes);
    }
    /* Python: for d in range(n_chunks * vec, H):
     *             s = load_scalar_as_f32(b, src, b.add(src_base, b.const_i32(d)),
     *                                    dtype=dtype)
     *             store_scalar_from_f32(b, dst, b.add(dst_base, b.const_i32(d)),
     *                                   s, dtype=dtype) */
    for(d = n_chunks * vec; d < H; ++d)
    {
        ckc_value_t* s = ckc_b_load_scalar_as_f32(
            b, src, ckc_b_add(b, src_base, ckc_b_const_i32(b, d)), dtype);
        ckc_b_store_scalar_from_f32(
            b, dst, ckc_b_add(b, dst_base, ckc_b_const_i32(b, d)), s, dtype);
    }
}
