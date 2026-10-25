// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_addr.py: sequence decode,
 * strided user tensors, the fp32 workspace and the paged caches of the
 * attention backward kernels.
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_addr.h for the mapping
 * and the byte-identity rules. Every op is created in the Python evaluation
 * order; each comment `py:` quotes the Python expression a block reproduces.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"

#include <stdio.h>
#include <string.h>

#include <vector>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ====================================================================== *
 * Integer helpers
 * ====================================================================== */

static bool is_type(const rocke_value_t* v, const rocke_type_t* t)
{
    return v != NULL && v->type != NULL && rocke_type_eq(v->type, t);
}

rocke_value_t* rocke_attn_bwd_as_i32(rocke_ir_builder_t* b, rocke_attn_bwd_iv_t x)
{
    if(x.v == NULL)
        return rocke_b_const_i32(b, x.i);
    return x.v;
}

rocke_value_t* rocke_attn_bwd_as_i64(rocke_ir_builder_t* b, rocke_attn_bwd_iv_t x, bool is_signed)
{
    if(x.v == NULL)
        return rocke_b_const_i64(b, x.i);
    if(is_type(x.v, rocke_i64()))
        return x.v;
    return is_signed ? rocke_b_sext(b, x.v, rocke_i64()) : rocke_b_zext(b, x.v, rocke_i64());
}

/* py: b.select(b.cmp_lt(x, y), x, y) */
static rocke_value_t* min64(rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y)
{
    rocke_value_t* lt = rocke_b_cmp_lt(b, x, y);
    return rocke_b_select(b, lt, x, y);
}

/* py: b.select(b.cmp_lt(x, y), y, x) */
static rocke_value_t* max64(rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y)
{
    rocke_value_t* lt = rocke_b_cmp_lt(b, x, y);
    return rocke_b_select(b, lt, y, x);
}

/* clamp(x, 0, hi) of an i64 value, returned as i32 (hi is i32). */
static rocke_value_t* clamp64_to_i32(rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* hi)
{
    rocke_value_t* c0 = rocke_b_const_i64(b, 0);
    rocke_value_t* lo = max64(b, x, c0);
    rocke_value_t* hi64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(hi), false);
    rocke_value_t* m = min64(b, lo, hi64);
    return rocke_b_trunc(b, m, rocke_i32());
}

rocke_value_t* rocke_attn_bwd_last_row(rocke_ir_builder_t* b, rocke_value_t* length)
{
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* l1 = rocke_b_sub(b, length, c1);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    return rocke_b_smax(b, l1, c0);
}

static rocke_value_t*
    scaled32(rocke_ir_builder_t* b, rocke_value_t* idx, rocke_attn_bwd_iv_t stride)
{
    if(stride.v == NULL)
    {
        if(stride.i == 1)
            return idx;
        if(stride.i == 0)
            return rocke_b_const_i32(b, 0);
    }
    rocke_value_t* s = rocke_attn_bwd_as_i32(b, stride);
    return rocke_b_mul(b, idx, s);
}

static rocke_value_t*
    scaled64(rocke_ir_builder_t* b, rocke_value_t* idx64, rocke_attn_bwd_iv_t stride)
{
    if(stride.v == NULL)
    {
        if(stride.i == 1)
            return idx64;
        rocke_value_t* c = rocke_b_const_i64(b, stride.i);
        return rocke_b_mul(b, idx64, c);
    }
    rocke_value_t* s = rocke_attn_bwd_as_i64(b, stride, false);
    return rocke_b_mul(b, idx64, s);
}

static int elem_bytes(rocke_ir_builder_t* b, const rocke_type_t* dtype)
{
    const char* n = dtype ? dtype->name : "";
    if(strcmp(n, "f16") == 0 || strcmp(n, "bf16") == 0)
        return 2;
    if(strcmp(n, "f32") == 0 || strcmp(n, "i32") == 0)
        return 4;
    rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported element type %s", n);
    return 0;
}

/* ====================================================================== *
 * Sequence decode
 * ====================================================================== */

rocke_value_t* rocke_attn_bwd_load_offset(rocke_ir_builder_t* b,
                                          rocke_value_t* off_ptr,
                                          rocke_value_t* idx,
                                          rocke_attn_bwd_iv_t off64)
{
    if(off64.v == NULL)
    {
        if(off64.i)
        {
            rocke_value_t* c2 = rocke_b_const_i32(b, 2);
            rocke_value_t* two_i = rocke_b_mul(b, idx, c2);
            rocke_value_t* lo = rocke_b_global_load_i32(b, off_ptr, two_i, 4);
            rocke_value_t* c1 = rocke_b_const_i32(b, 1);
            rocke_value_t* i_hi = rocke_b_add(b, two_i, c1);
            rocke_value_t* hi = rocke_b_global_load_i32(b, off_ptr, i_hi, 4);
            rocke_value_t* lo64 = rocke_b_zext(b, lo, rocke_i64());
            rocke_value_t* hi64 = rocke_b_sext(b, hi, rocke_i64());
            rocke_value_t* c32 = rocke_b_const_i64(b, 32);
            rocke_value_t* sh = rocke_b_shl(b, hi64, c32);
            return rocke_b_lor(b, lo64, sh);
        }
        rocke_value_t* w = rocke_b_global_load_i32(b, off_ptr, idx, 4);
        return rocke_b_sext(b, w, rocke_i64());
    }
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* is64 = rocke_b_cmp_ne(b, off64.v, c0);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* two_i = rocke_b_mul(b, idx, c2);
    rocke_value_t* i_lo = rocke_b_select(b, is64, two_i, idx);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* two_i1 = rocke_b_add(b, two_i, c1);
    rocke_value_t* i_hi = rocke_b_select(b, is64, two_i1, idx);
    rocke_value_t* lo = rocke_b_global_load_i32(b, off_ptr, i_lo, 4);
    rocke_value_t* hi = rocke_b_global_load_i32(b, off_ptr, i_hi, 4);
    rocke_value_t* lo64 = rocke_b_zext(b, lo, rocke_i64());
    rocke_value_t* hi64 = rocke_b_sext(b, hi, rocke_i64());
    rocke_value_t* c32 = rocke_b_const_i64(b, 32);
    rocke_value_t* sh = rocke_b_shl(b, hi64, c32);
    rocke_value_t* wide = rocke_b_lor(b, lo64, sh);
    rocke_value_t* narrow = rocke_b_sext(b, lo, rocke_i64());
    return rocke_b_select(b, is64, wide, narrow);
}

rocke_attn_bwd_decode_args_t rocke_attn_bwd_decode_args_default(void)
{
    rocke_attn_bwd_decode_args_t a;
    memset(&a, 0, sizeof(a));
    a.s_max = rocke_attn_bwd_iv_int(0);
    a.off_ptr = NULL;
    a.off64 = rocke_attn_bwd_iv_int(0);
    a.mult = rocke_attn_bwd_iv_int(1);
    a.div = rocke_attn_bwd_iv_int(1);
    a.seq_ptr = NULL;
    a.has_len = rocke_attn_bwd_iv_int(0);
    a.len_stride = rocke_attn_bwd_iv_int(1);
    a.ws_seg = rocke_attn_bwd_iv_int(0);
    return a;
}

/* py: smin(smax(SEQ[batch * len_stride], 0), s_max) */
static rocke_value_t* seq_len_read(rocke_ir_builder_t* b,
                                   rocke_value_t* seq_ptr,
                                   rocke_value_t* batch,
                                   rocke_attn_bwd_iv_t len_stride,
                                   rocke_value_t* s_max,
                                   rocke_value_t* zero)
{
    rocke_value_t* idx = scaled32(b, batch, len_stride);
    rocke_value_t* raw = rocke_b_global_load_i32(b, seq_ptr, idx, 4);
    rocke_value_t* lo = rocke_b_smax(b, raw, zero);
    return rocke_b_smin(b, lo, s_max);
}

/* _seq_len_value: NULL when has_len is statically absent. */
static rocke_value_t* seq_len_value(rocke_ir_builder_t* b,
                                    int* seq_guards,
                                    rocke_value_t* seq_ptr,
                                    rocke_value_t* batch,
                                    rocke_attn_bwd_iv_t has_len,
                                    rocke_attn_bwd_iv_t len_stride,
                                    rocke_value_t* s_max)
{
    if(has_len.v == NULL && !has_len.i)
        return NULL;
    if(seq_ptr == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "has_len needs a SEQ pointer");
        return NULL;
    }
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    if(has_len.v == NULL)
        return seq_len_read(b, seq_ptr, batch, len_stride, s_max, zero);
    rocke_value_t* ne = rocke_b_cmp_ne(b, has_len.v, zero);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* trip = rocke_b_select(b, ne, one, zero);
    const int n = (*seq_guards)++;
    char arg_name[32];
    char iv_name[32];
    snprintf(arg_name, sizeof(arg_name), "seqlen%d", n);
    snprintf(iv_name, sizeof(iv_name), "seq_i%d", n);
    rocke_value_t* step = rocke_b_const_i32(b, 1);
    rocke_iter_arg_t it;
    it.name = arg_name;
    it.init = s_max;
    rocke_for_t loop = rocke_b_scf_for_iter(b, zero, trip, step, &it, 1, iv_name, false, true);
    if(loop.op == NULL)
        return NULL;
    rocke_b_region_enter(b, loop.body);
    rocke_value_t* v = seq_len_read(b, seq_ptr, batch, len_stride, s_max, zero);
    rocke_b_scf_yield(b, &v, 1);
    rocke_b_region_leave(b);
    return loop.op->results[0];
}

rocke_attn_bwd_seq_info_t rocke_attn_bwd_decode_batched(rocke_ir_builder_t* b,
                                                        int* seq_guards,
                                                        rocke_value_t* batch,
                                                        const rocke_attn_bwd_decode_args_t* args)
{
    rocke_attn_bwd_seq_info_t out;
    memset(&out, 0, sizeof(out));
    rocke_value_t* smax = rocke_attn_bwd_as_i32(b, args->s_max);
    rocke_value_t* length
        = seq_len_value(b, seq_guards, args->seq_ptr, batch, args->has_len, args->len_stride, smax);
    if(length == NULL)
        length = smax;
    rocke_value_t* b64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(batch), false);
    rocke_value_t* s64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(smax), false);
    rocke_value_t* ws_row0 = rocke_b_mul(b, b64, s64);
    out.mode = ROCKE_ATTN_BWD_SEQ_BATCHED;
    out.batch = batch;
    out.length = length;
    out.last = rocke_attn_bwd_last_row(b, length);
    out.s_max = smax;
    out.ws_row0 = ws_row0;
    out.tok0 = NULL;
    return out;
}

rocke_attn_bwd_seq_info_t rocke_attn_bwd_decode_thd(rocke_ir_builder_t* b,
                                                    int* seq_guards,
                                                    rocke_value_t* batch,
                                                    const rocke_attn_bwd_decode_args_t* args)
{
    rocke_attn_bwd_seq_info_t out;
    memset(&out, 0, sizeof(out));
    rocke_value_t* smax = rocke_attn_bwd_as_i32(b, args->s_max);
    rocke_value_t* o0 = rocke_attn_bwd_load_offset(b, args->off_ptr, batch, args->off64);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* batch1 = rocke_b_add(b, batch, c1);
    rocke_value_t* o1 = rocke_attn_bwd_load_offset(b, args->off_ptr, batch1, args->off64);
    rocke_value_t* m64 = rocke_attn_bwd_as_i64(b, args->mult, true);
    rocke_value_t* d64 = rocke_attn_bwd_as_i64(b, args->div, true);
    const bool one_mult = rocke_attn_bwd_iv_is_int(args->mult, 1);
    const bool one_div = rocke_attn_bwd_iv_is_int(args->div, 1);

    rocke_value_t* tok0 = o0;
    if(!one_mult)
        tok0 = rocke_b_mul(b, tok0, m64);
    if(!one_div)
        tok0 = rocke_b_div(b, tok0, d64);
    rocke_value_t* n = rocke_b_sub(b, o1, o0);
    if(!one_mult)
        n = rocke_b_mul(b, n, m64);
    if(!one_div)
        n = rocke_b_div(b, n, d64);
    rocke_value_t* length = clamp64_to_i32(b, n, smax);
    rocke_value_t* seq_len
        = seq_len_value(b, seq_guards, args->seq_ptr, batch, args->has_len, args->len_stride, smax);
    if(seq_len != NULL)
        length = rocke_b_smin(b, length, seq_len);
    rocke_value_t* ws_row0 = tok0;
    if(!rocke_attn_bwd_iv_is_int(args->ws_seg, 0))
    {
        rocke_value_t* b64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(batch), false);
        rocke_value_t* s64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(smax), false);
        rocke_value_t* slot0 = rocke_b_mul(b, b64, s64);
        if(args->ws_seg.v == NULL)
        {
            ws_row0 = slot0;
        }
        else
        {
            rocke_value_t* c0 = rocke_b_const_i32(b, 0);
            rocke_value_t* seg = rocke_b_cmp_ne(b, args->ws_seg.v, c0);
            rocke_value_t* sel = rocke_b_select(b, seg, slot0, tok0);
            ws_row0 = rocke_b_readfirstlane(b, sel);
        }
    }
    out.mode = ROCKE_ATTN_BWD_SEQ_THD;
    out.batch = batch;
    out.length = length;
    out.last = rocke_attn_bwd_last_row(b, length);
    out.s_max = smax;
    out.ws_row0 = ws_row0;
    out.tok0 = tok0;
    return out;
}

/* ====================================================================== *
 * RowSlab
 * ====================================================================== */

rocke_value_t* rocke_attn_bwd_row_slab_local_row(const rocke_attn_bwd_row_slab_t* s,
                                                 rocke_value_t* row)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* r = s->last == NULL ? row : rocke_b_smin(b, row, s->last);
    return rocke_b_sub(b, r, s->base_row);
}

rocke_value_t* rocke_attn_bwd_row_slab_index(const rocke_attn_bwd_row_slab_t* s,
                                             rocke_value_t* row,
                                             rocke_attn_bwd_iv_t col)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* lr = rocke_attn_bwd_row_slab_local_row(s, row);
    rocke_value_t* idx = scaled32(b, lr, s->t_stride);
    if(rocke_attn_bwd_iv_is_int(col, 0))
        return idx;
    rocke_value_t* c = rocke_attn_bwd_as_i32(b, col);
    return rocke_b_add(b, idx, c);
}

rocke_value_t* rocke_attn_bwd_row_slab_load(const rocke_attn_bwd_row_slab_t* s,
                                            rocke_value_t* row,
                                            rocke_attn_bwd_iv_t col)
{
    const int esize = elem_bytes(s->b, s->dtype);
    rocke_value_t* idx = rocke_attn_bwd_row_slab_index(s, row, col);
    return rocke_b_global_load(s->b, s->ptr, idx, s->dtype, esize);
}

void rocke_attn_bwd_row_slab_store(const rocke_attn_bwd_row_slab_t* s,
                                   rocke_value_t* row,
                                   rocke_attn_bwd_iv_t col,
                                   rocke_value_t* value)
{
    const int esize = elem_bytes(s->b, s->dtype);
    rocke_value_t* idx = rocke_attn_bwd_row_slab_index(s, row, col);
    rocke_b_global_store(s->b, s->ptr, idx, value, esize);
}

/* idx, or idx + k for k != 0 (py: `idx if k == 0 else b.add(idx, b.const_i32(k))`). */
static rocke_value_t* idx_plus(rocke_ir_builder_t* b, rocke_value_t* idx, int k)
{
    if(k == 0)
        return idx;
    rocke_value_t* c = rocke_b_const_i32(b, k);
    return rocke_b_add(b, idx, c);
}

rocke_value_t* rocke_attn_bwd_row_slab_load_vec(const rocke_attn_bwd_row_slab_t* s,
                                                rocke_value_t* row,
                                                rocke_attn_bwd_iv_t col0,
                                                int n)
{
    rocke_ir_builder_t* b = s->b;
    if(n < 1)
    {
        /* py fails too (parts[0] of an empty list / an empty vec_pack) */
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "load_vec: n=%d must be positive", n);
        return NULL;
    }
    rocke_value_t* idx = rocke_attn_bwd_row_slab_index(s, row, col0);
    const int esize = elem_bytes(b, s->dtype);
    if(s->stage_vec == 8 && n % 8 == 0 && esize == 2)
    {
        if(n == 8)
            return rocke_b_global_load_vN(b, s->ptr, idx, s->dtype, 8, 16);
        std::vector<rocke_value_t*> parts((size_t)(n / 8));
        /* py builds every part first (b.add(idx, b.const_i32(8 * i)) for
         * every i, 0 included), then concatenates */
        for(int i = 0; i < n / 8; ++i)
        {
            rocke_value_t* c = rocke_b_const_i32(b, 8 * i);
            rocke_value_t* at = rocke_b_add(b, idx, c);
            parts[i] = rocke_b_global_load_vN(b, s->ptr, at, s->dtype, 8, 16);
        }
        rocke_value_t* out = parts[0];
        for(int i = 1; i < n / 8; ++i)
            out = rocke_b_vec_concat(b, out, parts[i]);
        return out;
    }
    const int w = s->narrow;
    if(s->stage_vec != 8 && w > 1 && n % w == 0 && esize == 2)
    {
        std::vector<rocke_value_t*> parts;
        parts.reserve((size_t)(n / w));
        for(int k = 0; k < n; k += w)
        {
            rocke_value_t* at = idx_plus(b, idx, k);
            parts.push_back(rocke_b_global_load_vN(b, s->ptr, at, s->dtype, w, 2 * w));
        }
        rocke_value_t* out = parts[0];
        for(size_t i = 1; i < parts.size(); ++i)
            out = rocke_b_vec_concat(b, out, parts[i]);
        return out;
    }
    std::vector<rocke_value_t*> elems((size_t)n);
    for(int i = 0; i < n; ++i)
    {
        rocke_value_t* at = idx_plus(b, idx, i);
        elems[i] = rocke_b_global_load(b, s->ptr, at, s->dtype, esize);
    }
    return rocke_b_vec_pack(b, elems.data(), n, s->dtype);
}

void rocke_attn_bwd_row_slab_store_vec(const rocke_attn_bwd_row_slab_t* s,
                                       rocke_value_t* row,
                                       rocke_attn_bwd_iv_t col0,
                                       rocke_value_t* value)
{
    rocke_ir_builder_t* b = s->b;
    if(!rocke_ir_builder_ok(b) || value == NULL)
        return;
    const int n = value->type->count;
    rocke_value_t* idx = rocke_attn_bwd_row_slab_index(s, row, col0);
    const int esize = elem_bytes(b, s->dtype);
    if(s->stage_vec == 8 && (n == 8 || n == 4 || n == 2) && esize == 2)
    {
        rocke_b_global_store_vN(b, s->ptr, idx, value, n, 2 * n);
        return;
    }
    const int w = s->narrow;
    if(s->stage_vec != 8 && w > 1 && n % w == 0 && esize == 2)
    {
        for(int k = 0; k < n; k += w)
        {
            rocke_value_t* part = value;
            if(n != w)
            {
                rocke_value_t* comps[4];
                for(int i = 0; i < w; ++i)
                    comps[i] = rocke_b_vec_extract(b, value, k + i);
                part = rocke_b_vec_pack(b, comps, w, s->dtype);
            }
            rocke_value_t* at = idx_plus(b, idx, k);
            rocke_b_global_store_vN(b, s->ptr, at, part, w, 2 * w);
        }
        return;
    }
    for(int i = 0; i < n; ++i)
    {
        rocke_value_t* at = idx_plus(b, idx, i);
        rocke_value_t* e = rocke_b_vec_extract(b, value, i);
        rocke_b_global_store(b, s->ptr, at, e, esize);
    }
}

/* ====================================================================== *
 * StridedTensorAddr
 * ====================================================================== */

bool rocke_attn_bwd_strided_tensor_addr_init(rocke_attn_bwd_strided_tensor_addr_t* t,
                                             rocke_ir_builder_t* b,
                                             rocke_value_t* ptr,
                                             const rocke_type_t* dtype,
                                             const rocke_attn_bwd_iv_t strides[3],
                                             const rocke_attn_bwd_seq_info_t* seq,
                                             int stage_vec,
                                             int narrow_vec)
{
    memset(t, 0, sizeof(*t));
    if(stage_vec != 8 && stage_vec != 1)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "stage_vec must be 8 or 1 (got %d)", stage_vec);
        return false;
    }
    if((narrow_vec != 1 && narrow_vec != 2 && narrow_vec != 4)
       || (narrow_vec != 1 && stage_vec != 1))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "narrow_vec=%d needs stage_vec = 1", narrow_vec);
        return false;
    }
    if(!rocke_type_eq(dtype, rocke_f16()) && !rocke_type_eq(dtype, rocke_bf16())
       && !rocke_type_eq(dtype, rocke_f32()))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported tensor dtype %s", dtype->name);
        return false;
    }
    t->b = b;
    t->ptr = ptr;
    t->dtype = dtype;
    t->strides[0] = strides[0];
    t->strides[1] = strides[1];
    t->strides[2] = strides[2];
    t->seq = *seq;
    t->stage_vec = stage_vec;
    t->narrow_vec = narrow_vec;
    t->esize = elem_bytes(b, dtype);
    t->base = ptr;
    if(seq->mode == ROCKE_ATTN_BWD_SEQ_THD)
    {
        if(seq->tok0 == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "THD sequence info without tok0");
            return false;
        }
        rocke_value_t* seq_base = scaled64(b, seq->tok0, strides[2]);
        rocke_value_t* es = rocke_b_const_i64(b, t->esize);
        rocke_value_t* off = rocke_b_mul(b, seq_base, es);
        t->base = rocke_b_global_ptr_add(b, ptr, off);
        t->seq_base = NULL;
    }
    else
    {
        rocke_value_t* b64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(seq->batch), false);
        t->seq_base = scaled64(b, b64, strides[0]);
    }
    return rocke_ir_builder_ok(b);
}

rocke_attn_bwd_row_slab_t
    rocke_attn_bwd_strided_tensor_addr_slab(const rocke_attn_bwd_strided_tensor_addr_t* t,
                                            rocke_attn_bwd_iv_t h,
                                            rocke_attn_bwd_iv_t row0,
                                            bool clamp)
{
    rocke_ir_builder_t* b = t->b;
    rocke_attn_bwd_row_slab_t s;
    memset(&s, 0, sizeof(s));
    rocke_value_t* r0 = rocke_attn_bwd_as_i32(b, row0);
    const bool static_zero = rocke_attn_bwd_iv_is_int(row0, 0);
    rocke_value_t* base_row = (clamp && !static_zero) ? rocke_b_smin(b, r0, t->seq.last) : r0;
    rocke_value_t* off = t->seq_base;
    if(!rocke_attn_bwd_iv_is_int(h, 0))
    {
        rocke_value_t* h64 = rocke_attn_bwd_as_i64(b, h, false);
        rocke_value_t* term = scaled64(b, h64, t->strides[1]);
        off = off == NULL ? term : rocke_b_add(b, off, term);
    }
    if(!static_zero)
    {
        rocke_value_t* r64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(base_row), false);
        rocke_value_t* term = scaled64(b, r64, t->strides[2]);
        off = off == NULL ? term : rocke_b_add(b, off, term);
    }
    rocke_value_t* ptr;
    if(off == NULL)
    {
        ptr = t->base;
    }
    else
    {
        rocke_value_t* es = rocke_b_const_i64(b, t->esize);
        off = rocke_b_mul(b, off, es);
        ptr = rocke_b_global_ptr_add(b, t->base, off);
    }
    s.b = b;
    s.ptr = ptr;
    s.dtype = t->dtype;
    s.t_stride = t->strides[2];
    s.base_row = base_row;
    s.last = clamp ? t->seq.last : NULL;
    s.stage_vec = t->stage_vec;
    s.narrow = t->narrow_vec;
    s.has_parts = true;
    s.parts_base = t->base;
    s.parts_off = off;
    return s;
}

/* ====================================================================== *
 * WsSlab
 * ====================================================================== */

rocke_value_t* rocke_attn_bwd_ws_slab_row_ok(const rocke_attn_bwd_ws_slab_t* s, rocke_value_t* r)
{
    return rocke_b_cmp_lt(s->b, r, s->rows_left);
}

rocke_value_t* rocke_attn_bwd_ws_slab_local_row(const rocke_attn_bwd_ws_slab_t* s, rocke_value_t* r)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* hi = rocke_b_sub(b, s->rows_left, c1);
    rocke_value_t* m = rocke_b_smin(b, r, hi);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    return rocke_b_smax(b, m, c0);
}

rocke_value_t* rocke_attn_bwd_ws_slab_index(const rocke_attn_bwd_ws_slab_t* s,
                                            rocke_value_t* r,
                                            rocke_attn_bwd_iv_t x)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* lr = rocke_attn_bwd_ws_slab_local_row(s, r);
    rocke_value_t* idx = scaled32(b, lr, s->row_stride);
    if(rocke_attn_bwd_iv_is_int(x, 0))
        return idx;
    rocke_value_t* xv = rocke_attn_bwd_as_i32(b, x);
    return rocke_b_add(b, idx, xv);
}

void rocke_attn_bwd_ws_slab_store(const rocke_attn_bwd_ws_slab_t* s,
                                  rocke_value_t* r,
                                  rocke_attn_bwd_iv_t x,
                                  rocke_value_t* v)
{
    rocke_value_t* idx = rocke_attn_bwd_ws_slab_index(s, r, x);
    rocke_b_global_store(s->b, s->ptr, idx, v, 4);
}

void rocke_attn_bwd_ws_slab_store_vec(const rocke_attn_bwd_ws_slab_t* s,
                                      rocke_value_t* r,
                                      rocke_attn_bwd_iv_t x,
                                      rocke_value_t* v)
{
    rocke_ir_builder_t* b = s->b;
    if(!rocke_ir_builder_ok(b) || v == NULL)
        return;
    const int n = v->type->count;
    if(n == 4)
    {
        rocke_value_t* idx = rocke_attn_bwd_ws_slab_index(s, r, x);
        rocke_b_global_store_vN(b, s->ptr, idx, v, 4, 16);
        return;
    }
    if(n % 4)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "workspace vector stores take multiples of 4 floats");
        return;
    }
    for(int i = 0; i < n / 4; ++i)
    {
        rocke_value_t* comps[4];
        for(int j = 0; j < 4; ++j)
            comps[j] = rocke_b_vec_extract(b, v, 4 * i + j);
        rocke_value_t* part = rocke_b_vec_pack(b, comps, 4, rocke_f32());
        rocke_attn_bwd_iv_t xi;
        if(x.v == NULL)
        {
            xi = rocke_attn_bwd_iv_int(x.i + 4 * i);
        }
        else
        {
            rocke_value_t* c = rocke_b_const_i32(b, 4 * i);
            xi = rocke_attn_bwd_iv_val(rocke_b_add(b, x.v, c));
        }
        rocke_value_t* idx = rocke_attn_bwd_ws_slab_index(s, r, xi);
        rocke_b_global_store_vN(b, s->ptr, idx, part, 4, 16);
    }
}

rocke_value_t* rocke_attn_bwd_ws_slab_load_vec4(const rocke_attn_bwd_ws_slab_t* s,
                                                rocke_value_t* r,
                                                rocke_attn_bwd_iv_t x)
{
    rocke_value_t* idx = rocke_attn_bwd_ws_slab_index(s, r, x);
    return rocke_b_global_load_vN(s->b, s->ptr, idx, rocke_f32(), 4, 16);
}

/* ====================================================================== *
 * WorkspaceAddr
 * ====================================================================== */

bool rocke_attn_bwd_workspace_addr_init(rocke_attn_bwd_workspace_addr_t* w,
                                        rocke_ir_builder_t* b,
                                        rocke_value_t* ptr,
                                        rocke_attn_bwd_iv_t ws_rows,
                                        int width,
                                        rocke_value_t* row0,
                                        rocke_attn_bwd_ws_layout_t layout,
                                        const rocke_attn_bwd_iv_t* n_heads)
{
    memset(w, 0, sizeof(*w));
    if(layout != ROCKE_ATTN_BWD_WS_HEAD_MAJOR && layout != ROCKE_ATTN_BWD_WS_TOKEN_MAJOR)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "ws_layout must be head_major or token_major");
        return false;
    }
    if(layout == ROCKE_ATTN_BWD_WS_TOKEN_MAJOR && n_heads == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "token_major workspace needs n_heads");
        return false;
    }
    if(width < 1)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "workspace width must be >= 1");
        return false;
    }
    w->b = b;
    w->ptr = ptr;
    w->ws_rows = rocke_attn_bwd_as_i32(b, ws_rows);
    w->width = width;
    w->row0 = is_type(row0, rocke_i64())
                  ? row0
                  : rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(row0), false);
    w->layout = layout;
    w->has_n_heads = n_heads != NULL;
    if(n_heads != NULL)
        w->n_heads = *n_heads;
    return rocke_ir_builder_ok(b);
}

rocke_attn_bwd_ws_slab_t rocke_attn_bwd_workspace_addr_slab(
    const rocke_attn_bwd_workspace_addr_t* w, rocke_attn_bwd_iv_t h, rocke_attn_bwd_iv_t tile_row0)
{
    rocke_ir_builder_t* b = w->b;
    rocke_attn_bwd_ws_slab_t s;
    memset(&s, 0, sizeof(s));
    rocke_value_t* base = w->row0;
    if(!rocke_attn_bwd_iv_is_int(tile_row0, 0))
    {
        rocke_value_t* t64 = rocke_attn_bwd_as_i64(b, tile_row0, true);
        base = rocke_b_add(b, base, t64);
    }
    rocke_value_t* rows64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(w->ws_rows), false);
    rocke_value_t* one64 = rocke_b_const_i64(b, 1);
    /* py: _max64(b, _min64(b, base, b.sub(rows64, one64)), b.const_i64(0)) */
    rocke_value_t* last64 = rocke_b_sub(b, rows64, one64);
    rocke_value_t* lo = min64(b, base, last64);
    rocke_value_t* c0 = rocke_b_const_i64(b, 0);
    rocke_value_t* base_ws = max64(b, lo, c0);
    /* py: _clamp64_to_i32(b, b.sub(rows64, base), b.const_i32(I32_FAR)) */
    rocke_value_t* left64 = rocke_b_sub(b, rows64, base);
    rocke_value_t* far = rocke_b_const_i32(b, ROCKE_ATTN_BWD_I32_FAR);
    rocke_value_t* rows_left = clamp64_to_i32(b, left64, far);
    rocke_value_t* h64 = rocke_attn_bwd_as_i64(b, h, false);
    rocke_value_t* elem;
    rocke_attn_bwd_iv_t row_stride;
    if(w->layout == ROCKE_ATTN_BWD_WS_HEAD_MAJOR)
    {
        rocke_value_t* hr = rocke_b_mul(b, h64, rows64);
        elem = rocke_b_add(b, hr, base_ws);
        elem = scaled64(b, elem, rocke_attn_bwd_iv_int(w->width));
        row_stride = rocke_attn_bwd_iv_int(w->width);
    }
    else
    {
        rocke_value_t* bh = scaled64(b, base_ws, w->n_heads);
        elem = rocke_b_add(b, bh, h64);
        elem = scaled64(b, elem, rocke_attn_bwd_iv_int(w->width));
        if(w->n_heads.v == NULL)
            row_stride = rocke_attn_bwd_iv_int(w->n_heads.i * w->width);
        else
            row_stride
                = rocke_attn_bwd_iv_val(scaled32(b, w->n_heads.v, rocke_attn_bwd_iv_int(w->width)));
    }
    rocke_value_t* c4 = rocke_b_const_i64(b, 4);
    rocke_value_t* off = rocke_b_mul(b, elem, c4);
    rocke_value_t* ptr = rocke_b_global_ptr_add(b, w->ptr, off);
    s.b = b;
    s.ptr = ptr;
    s.rows_left = rows_left;
    s.row_stride = row_stride;
    s.has_parts = true;
    s.parts_base = w->ptr;
    s.parts_off = off;
    return s;
}

/* ====================================================================== *
 * Paged caches
 * ====================================================================== */

rocke_value_t* rocke_attn_bwd_block_table_entry(rocke_ir_builder_t* b,
                                                rocke_value_t* bt_ptr,
                                                rocke_value_t* seq_idx,
                                                rocke_value_t* j,
                                                rocke_attn_bwd_iv_t bt_stride)
{
    rocke_value_t* s64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(seq_idx), false);
    rocke_value_t* off = scaled64(b, s64, bt_stride);
    rocke_value_t* c4 = rocke_b_const_i64(b, 4);
    rocke_value_t* bytes = rocke_b_mul(b, off, c4);
    rocke_value_t* row = rocke_b_global_ptr_add(b, bt_ptr, bytes);
    return rocke_b_global_load_i32(b, row, j, 4);
}

bool rocke_attn_bwd_paged_cache_addr_init(rocke_attn_bwd_paged_cache_addr_t* p,
                                          rocke_ir_builder_t* b,
                                          rocke_value_t* ptr,
                                          const rocke_type_t* dtype,
                                          const rocke_attn_bwd_iv_t strides[3],
                                          int stage_vec)
{
    memset(p, 0, sizeof(*p));
    if(stage_vec != 8 && stage_vec != 1)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "stage_vec must be 8 or 1 (got %d)", stage_vec);
        return false;
    }
    p->b = b;
    p->ptr = ptr;
    p->dtype = dtype;
    p->strides[0] = strides[0];
    p->strides[1] = strides[1];
    p->strides[2] = strides[2];
    p->stage_vec = stage_vec;
    p->esize = elem_bytes(b, dtype);
    return rocke_ir_builder_ok(b);
}

rocke_attn_bwd_row_slab_t
    rocke_attn_bwd_paged_cache_addr_slab(const rocke_attn_bwd_paged_cache_addr_t* p,
                                         rocke_value_t* page_id,
                                         rocke_attn_bwd_iv_t h,
                                         rocke_attn_bwd_iv_t tok0)
{
    rocke_ir_builder_t* b = p->b;
    rocke_attn_bwd_row_slab_t s;
    memset(&s, 0, sizeof(s));
    rocke_value_t* p64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(page_id), false);
    rocke_value_t* off = scaled64(b, p64, p->strides[0]);
    if(!rocke_attn_bwd_iv_is_int(h, 0))
    {
        rocke_value_t* h64 = rocke_attn_bwd_as_i64(b, h, false);
        rocke_value_t* term = scaled64(b, h64, p->strides[2]);
        off = rocke_b_add(b, off, term);
    }
    rocke_value_t* r0 = rocke_attn_bwd_as_i32(b, tok0);
    if(!rocke_attn_bwd_iv_is_int(tok0, 0))
    {
        rocke_value_t* r64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(r0), false);
        rocke_value_t* term = scaled64(b, r64, p->strides[1]);
        off = rocke_b_add(b, off, term);
    }
    rocke_value_t* es = rocke_b_const_i64(b, p->esize);
    rocke_value_t* bytes = rocke_b_mul(b, off, es);
    rocke_value_t* ptr = rocke_b_global_ptr_add(b, p->ptr, bytes);
    s.b = b;
    s.ptr = ptr;
    s.dtype = p->dtype;
    s.t_stride = p->strides[1];
    s.base_row = r0;
    s.last = NULL;
    s.stage_vec = p->stage_vec;
    s.narrow = 1;
    s.has_parts = false;
    return s;
}
