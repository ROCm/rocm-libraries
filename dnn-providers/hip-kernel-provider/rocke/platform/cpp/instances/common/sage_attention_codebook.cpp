// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_sage_attention_sage_attention_codebook.c -- chunked port of the
 * SHARED CODEBOOK / LANE-LOAD PRIMITIVES of
 * rocke/instances/common/sage_attention.py (lines 104-456).
 *
 * Scope (module-private helpers that capture no enclosing locals; raw-builder
 * free functions used by the warp driver + its closures):
 *
 *   rocke_sage_magic_div              <- _magic_div                 (104-117)
 *   rocke_sage_is_lds_handle          <- _is_lds_handle             (364-367)
 *   rocke_sage_stage_codebook_to_lds  <- _stage_codebook_to_lds     (298-326)
 *   rocke_sage_codebook_lds_lookup_f32<- _codebook_lds_lookup_f32   (329-331)
 *   rocke_sage_codebook_i8_to_f32     <- _codebook_i8_to_f32        (334-346)
 *   rocke_sage_codebook_i4_pair_to_f32<- _codebook_i4_pair_to_f32   (349-361)
 *   rocke_sage_vectorised_byte_slice  <- _vectorised_byte_slice     (370-406)
 *   rocke_sage_load_kv_lane_f32       <- _load_kv_lane_f32          (409-441)
 *   rocke_sage_load_q_lane_f32        <- _load_q_lane_f32           (444-456)
 *
 * The dequant primitives live in one TU so the i8/i4 codebook arithmetic is
 * independently testable. Builder-call sequence is byte-identical to the Python.
 */

#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.i4_dequant.h" /* unpack_i4_byte_to_pair_i32 */
#include "rocke/helper_rocke.helpers.io.h" /* load_scalar_as_f32         */
#include "rocke/helper_rocke.helpers.transforms.h" /* magic numbers + division   */
#include "rocke/instance_sage_attention_internal.h"

/* WARP_SIZE = 64 from rocke.instances.common._fmha_warp_body. */
#ifndef ROCKE_FMHA_WARP_SIZE
#define ROCKE_FMHA_WARP_SIZE 64
#endif

/* ---------------------------------------------------------------------
 * _magic_div (lines 104-117)
 * ---------------------------------------------------------------------
 *
 *     def _magic_div(b, dividend, divisor) -> Value:
 *         mult, shift = calculate_magic_numbers(divisor)
 *         return do_magic_division(b, dividend, mult, shift)
 */
rocke_value_t* rocke_sage_magic_div(rocke_ir_builder_t* b, rocke_value_t* dividend, int divisor)
{
    uint64_t mult = 0;
    int shift = 0;
    if(!rocke_calculate_magic_numbers(b, divisor, &mult, &shift))
    {
        return NULL;
    }
    return rocke_do_magic_division(b, dividend, mult, shift);
}

/* ---------------------------------------------------------------------
 * _is_lds_handle (lines 364-367)
 * ---------------------------------------------------------------------
 *
 *     def _is_lds_handle(v) -> bool:
 *         return type(v.type).__name__ == "SmemType"
 *
 * In C the type kind discriminant carries this; SmemType <-> ROCKE_TYPE_SMEM.
 */
bool rocke_sage_is_lds_handle(const rocke_value_t* v)
{
    return v != NULL && v->type != NULL && v->type->kind == ROCKE_TYPE_SMEM;
}

/* ---------------------------------------------------------------------
 * _codebook_lds_lookup_f32 (lines 329-331)
 * ---------------------------------------------------------------------
 *
 *     def _codebook_lds_lookup_f32(b, cb_lds, idx) -> Value:
 *         return b.vec_extract(b.smem_load_vN_f32(cb_lds, idx, n=1), 0)
 */
rocke_value_t* rocke_sage_codebook_lds_lookup_f32(rocke_ir_builder_t* b,
                                                  rocke_value_t* cb_lds,
                                                  rocke_value_t* idx)
{
    rocke_value_t* indices[1];
    rocke_value_t* vec;
    indices[0] = idx;
    vec = rocke_b_smem_load_vN_f32(b, cb_lds, indices, 1, 1);
    return rocke_b_vec_extract(b, vec, 0);
}

/* ---------------------------------------------------------------------
 * _stage_codebook_to_lds (lines 298-326)
 * ---------------------------------------------------------------------
 *
 *     def _stage_codebook_to_lds(b, cb_global, *, n_entries, tid, name_hint):
 *         cb_lds = b.smem_alloc(F32, [n_entries], name_hint=name_hint)
 *         for base in range(0, n_entries, WARP_SIZE):
 *             slot = tid if base == 0 else b.add(tid, b.const_i32(base))
 *             if base + WARP_SIZE <= n_entries:
 *                 val = b.global_load_f32(cb_global, slot)
 *                 b.smem_store_vN_f32(cb_lds, [slot], val, 1)
 *             else:
 *                 with b.scf_if(b.cmp_lt(slot, b.const_i32(n_entries))):
 *                     val = b.global_load_f32(cb_global, slot)
 *                     b.smem_store_vN_f32(cb_lds, [slot], val, 1)
 *         return cb_lds
 */
rocke_value_t* rocke_sage_stage_codebook_to_lds(rocke_ir_builder_t* b,
                                                rocke_value_t* cb_global,
                                                int n_entries,
                                                rocke_value_t* tid,
                                                const char* name_hint)
{
    int shape[1];
    rocke_value_t* cb_lds;
    int base;

    shape[0] = n_entries;
    cb_lds = rocke_b_smem_alloc(b, rocke_f32(), shape, 1, name_hint);

    /* One wave64 stages the table; loop when n_entries > WARP_SIZE. */
    for(base = 0; base < n_entries; base += ROCKE_FMHA_WARP_SIZE)
    {
        rocke_value_t* slot;
        if(base == 0)
        {
            slot = tid;
        }
        else
        {
            slot = rocke_b_add(b, tid, rocke_b_const_i32(b, (int64_t)base));
        }

        if(base + ROCKE_FMHA_WARP_SIZE <= n_entries)
        {
            rocke_value_t* val = rocke_b_global_load_f32(b, cb_global, slot, 0);
            rocke_value_t* idx[1];
            idx[0] = slot;
            rocke_b_smem_store_vN_f32(b, cb_lds, idx, 1, val, 1);
        }
        else
        {
            rocke_value_t* in_range
                = rocke_b_cmp_lt(b, slot, rocke_b_const_i32(b, (int64_t)n_entries));
            rocke_if_t iff = rocke_b_scf_if(b, in_range);
            rocke_value_t* val;
            rocke_value_t* idx[1];
            rocke_b_region_enter(b, iff.then_region);
            val = rocke_b_global_load_f32(b, cb_global, slot, 0);
            idx[0] = slot;
            rocke_b_smem_store_vN_f32(b, cb_lds, idx, 1, val, 1);
            rocke_b_region_leave(b);
        }
    }
    return cb_lds;
}

/* ---------------------------------------------------------------------
 * _codebook_i8_to_f32 (lines 334-346)
 * ---------------------------------------------------------------------
 *
 *     def _codebook_i8_to_f32(b, cb_ptr, i8_value) -> Value:
 *         i32 = b.sext(i8_value, I32)
 *         idx = b.add(i32, b.const_i32(128))
 *         if _is_lds_handle(cb_ptr):
 *             return _codebook_lds_lookup_f32(b, cb_ptr, idx)
 *         return b.global_load_f32(cb_ptr, idx)
 */
rocke_value_t* rocke_sage_codebook_i8_to_f32(rocke_ir_builder_t* b,
                                             rocke_value_t* cb_ptr,
                                             rocke_value_t* i8_value)
{
    rocke_value_t* i32 = rocke_b_sext(b, i8_value, rocke_i32());
    rocke_value_t* idx = rocke_b_add(b, i32, rocke_b_const_i32(b, 128));
    if(rocke_sage_is_lds_handle(cb_ptr))
    {
        return rocke_sage_codebook_lds_lookup_f32(b, cb_ptr, idx);
    }
    return rocke_b_global_load_f32(b, cb_ptr, idx, 0);
}

/* ---------------------------------------------------------------------
 * _codebook_i4_pair_to_f32 (lines 349-361)
 * ---------------------------------------------------------------------
 *
 *     def _codebook_i4_pair_to_f32(b, cb_ptr, packed_byte_i8) -> (Value, Value):
 *         lo_i32, hi_i32 = unpack_i4_byte_to_pair_i32(b, packed_byte_i8)
 *         c8 = b.const_i32(8)
 *         if _is_lds_handle(cb_ptr):
 *             lo = _codebook_lds_lookup_f32(b, cb_ptr, b.add(lo_i32, c8))
 *             hi = _codebook_lds_lookup_f32(b, cb_ptr, b.add(hi_i32, c8))
 *             return lo, hi
 *         lo = b.global_load_f32(cb_ptr, b.add(lo_i32, c8))
 *         hi = b.global_load_f32(cb_ptr, b.add(hi_i32, c8))
 *         return lo, hi
 */
void rocke_sage_codebook_i4_pair_to_f32(rocke_ir_builder_t* b,
                                        rocke_value_t* cb_ptr,
                                        rocke_value_t* packed_byte_i8,
                                        rocke_value_t** out_lo,
                                        rocke_value_t** out_hi)
{
    rocke_value_t* lo_i32 = NULL;
    rocke_value_t* hi_i32 = NULL;
    rocke_value_t* c8;
    rocke_unpack_i4_byte_to_pair_i32(b, packed_byte_i8, &lo_i32, &hi_i32);
    c8 = rocke_b_const_i32(b, 8);
    if(rocke_sage_is_lds_handle(cb_ptr))
    {
        *out_lo = rocke_sage_codebook_lds_lookup_f32(b, cb_ptr, rocke_b_add(b, lo_i32, c8));
        *out_hi = rocke_sage_codebook_lds_lookup_f32(b, cb_ptr, rocke_b_add(b, hi_i32, c8));
        return;
    }
    *out_lo = rocke_b_global_load_f32(b, cb_ptr, rocke_b_add(b, lo_i32, c8), 0);
    *out_hi = rocke_b_global_load_f32(b, cb_ptr, rocke_b_add(b, hi_i32, c8), 0);
}

/* ---------------------------------------------------------------------
 * _vectorised_byte_slice (lines 370-406)
 * ---------------------------------------------------------------------
 *
 *     def _vectorised_byte_slice(b, KV, base, lane_d_base, ept, elem_ty):
 *         addr_base = b.add(base, lane_d_base)
 *         if ept == 1:
 *             return [b.global_load(KV, addr_base, elem_ty)]
 *         is_f16_like = elem_ty.name in ("f16", "bf16")
 *         is_fp8_like = elem_ty.name in ("fp8e4m3", "bf8e5m2")
 *         if (is_f16_like and ept in (2,4,8)) or (is_fp8_like and ept in (2,4,8,16)):
 *             elem_bytes = 2 if is_f16_like else 1
 *             vec = b.global_load_vN(KV, addr_base, elem_ty, ept, align=ept*elem_bytes)
 *             return [b.vec_extract(vec, k) for k in range(ept)]
 *         return [
 *             b.global_load(KV, b.add(addr_base, b.const_i32(k)), elem_ty)
 *             for k in range(ept)
 *         ]
 */
void rocke_sage_vectorised_byte_slice(rocke_ir_builder_t* b,
                                      rocke_value_t* KV,
                                      rocke_value_t* base,
                                      rocke_value_t* lane_d_base,
                                      int ept,
                                      const rocke_type_t* elem_ty,
                                      rocke_value_t** out)
{
    rocke_value_t* addr_base = rocke_b_add(b, base, lane_d_base);
    bool is_f16_like;
    bool is_fp8_like;
    int k;

    if(ept == 1)
    {
        out[0] = rocke_b_global_load(b, KV, addr_base, elem_ty, 0);
        return;
    }

    is_f16_like = elem_ty != NULL && elem_ty->name != NULL
                  && (strcmp(elem_ty->name, "f16") == 0 || strcmp(elem_ty->name, "bf16") == 0);
    is_fp8_like
        = elem_ty != NULL && elem_ty->name != NULL
          && (strcmp(elem_ty->name, "fp8e4m3") == 0 || strcmp(elem_ty->name, "bf8e5m2") == 0);

    if((is_f16_like && (ept == 2 || ept == 4 || ept == 8))
       || (is_fp8_like && (ept == 2 || ept == 4 || ept == 8 || ept == 16)))
    {
        int elem_bytes = is_f16_like ? 2 : 1;
        rocke_value_t* vec
            = rocke_b_global_load_vN(b, KV, addr_base, elem_ty, ept, ept * elem_bytes);
        for(k = 0; k < ept; ++k)
        {
            out[k] = rocke_b_vec_extract(b, vec, k);
        }
        return;
    }

    /* Scalar fallback for i8 / non-power-of-two ept. */
    for(k = 0; k < ept; ++k)
    {
        rocke_value_t* addr = rocke_b_add(b, addr_base, rocke_b_const_i32(b, (int64_t)k));
        out[k] = rocke_b_global_load(b, KV, addr, elem_ty, 0);
    }
}

/* ---------------------------------------------------------------------
 * _load_kv_lane_f32 (lines 409-441)
 * ---------------------------------------------------------------------
 *
 *     def _load_kv_lane_f32(b, *, KV, base, lane_d_base, ept, quant_mode,
 *                           cb_ptr, kv_ty, dtype) -> list[Value]:
 *         if quant_mode in ("fp16_bf16", "fp8_bf16"):
 *             elems = _vectorised_byte_slice(b, KV, base, lane_d_base, ept, kv_ty)
 *             if quant_mode == "fp16_bf16":
 *                 return [b.cast_to_f32(e) for e in elems]
 *             return [b.cvt_fp8_to_f32(e) for e in elems]
 *         if quant_mode == "i8_fp8_bf16":
 *             bytes_ = _vectorised_byte_slice(b, KV, base, lane_d_base, ept, I8)
 *             return [_codebook_i8_to_f32(b, cb_ptr, byt) for byt in bytes_]
 *         raise ValueError(f"unsupported quant_mode for lane load: {quant_mode!r}")
 */
void rocke_sage_load_kv_lane_f32(rocke_ir_builder_t* b,
                                 rocke_value_t* KV,
                                 rocke_value_t* base,
                                 rocke_value_t* lane_d_base,
                                 int ept,
                                 rocke_sage_quant_mode_t quant_mode,
                                 rocke_value_t* cb_ptr,
                                 const rocke_type_t* kv_ty,
                                 const char* dtype,
                                 rocke_value_t** out)
{
    rocke_value_t* elems[ROCKE_SAGE_MAX_EPT];
    int k;
    (void)dtype;

    if(quant_mode == ROCKE_SAGE_QUANT_FP16_BF16 || quant_mode == ROCKE_SAGE_QUANT_FP8_BF16)
    {
        rocke_sage_vectorised_byte_slice(b, KV, base, lane_d_base, ept, kv_ty, elems);
        if(quant_mode == ROCKE_SAGE_QUANT_FP16_BF16)
        {
            for(k = 0; k < ept; ++k)
            {
                out[k] = rocke_b_cast_to_f32(b, elems[k]);
            }
        }
        else
        {
            for(k = 0; k < ept; ++k)
            {
                out[k] = rocke_b_cvt_fp8_to_f32(b, elems[k]);
            }
        }
        return;
    }
    if(quant_mode == ROCKE_SAGE_QUANT_I8_FP8_BF16)
    {
        rocke_sage_vectorised_byte_slice(b, KV, base, lane_d_base, ept, rocke_i8(), elems);
        for(k = 0; k < ept; ++k)
        {
            out[k] = rocke_sage_codebook_i8_to_f32(b, cb_ptr, elems[k]);
        }
        return;
    }

    /* raise ValueError(f"unsupported quant_mode for lane load: ...") */
    b->status = ROCKE_ERR_VALUE;
    snprintf(b->err,
             ROCKE_ERR_MSG_CAP,
             "unsupported quant_mode for lane load: %s",
             rocke_sage_quant_mode_name(quant_mode));
}

/* ---------------------------------------------------------------------
 * _load_q_lane_f32 (lines 444-456)
 * ---------------------------------------------------------------------
 *
 *     def _load_q_lane_f32(b, Q, q_row_base, lane_d_base, ept, dtype, kv_ty):
 *         addr_base = b.add(q_row_base, lane_d_base)
 *         elem_bytes = 2
 *         if ept in (2, 4, 8):
 *             vec = b.global_load_vN(Q, addr_base, kv_ty, ept, align=ept*elem_bytes)
 *             return [b.cast_to_f32(b.vec_extract(vec, k)) for k in range(ept)]
 *         return [
 *             load_scalar_as_f32(b, Q, b.add(addr_base, b.const_i32(k)), dtype=dtype)
 *             for k in range(ept)
 *         ]
 */
void rocke_sage_load_q_lane_f32(rocke_ir_builder_t* b,
                                rocke_value_t* Q,
                                rocke_value_t* q_row_base,
                                rocke_value_t* lane_d_base,
                                int ept,
                                const char* dtype,
                                const rocke_type_t* kv_ty,
                                rocke_value_t** out)
{
    rocke_value_t* addr_base = rocke_b_add(b, q_row_base, lane_d_base);
    const int elem_bytes = 2;
    int k;

    if(ept == 2 || ept == 4 || ept == 8)
    {
        rocke_value_t* vec = rocke_b_global_load_vN(b, Q, addr_base, kv_ty, ept, ept * elem_bytes);
        for(k = 0; k < ept; ++k)
        {
            out[k] = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, vec, k));
        }
        return;
    }
    for(k = 0; k < ept; ++k)
    {
        rocke_value_t* addr = rocke_b_add(b, addr_base, rocke_b_const_i32(b, (int64_t)k));
        out[k] = rocke_b_load_scalar_as_f32(b, Q, addr, dtype);
    }
}
