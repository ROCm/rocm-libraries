// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_ATTENTION_PAGED_BOUNDS_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_PAGED_BOUNDS_INTERNAL_H
#include "rocke/ir.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Reject unsupported bounded geometry before computed spec properties. */
static inline bool rocke_bounds_geometry(int hd,
                                         int qh,
                                         int kh,
                                         int segments,
                                         const char* dtype,
                                         const char* arch,
                                         const char* expected_arch)
{
    if(arch == NULL || strcmp(arch, expected_arch) != 0 || dtype == NULL
       || (strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0)
       || (hd != 64 && hd != 128 && hd != 256) || kh <= 0 || qh <= 0 || segments < 1
       || segments > 128 || qh % kh != 0 || (int64_t)qh * hd * segments * 4 > 0x7FFF0000)
        return false;
    const int ratio = qh / kh;
    return ratio == 1 || ratio == 2 || ratio == 4 || ratio == 8 || ratio == 16;
}

/* Mirror of library/kernels/common/_attention_paged_bounds.py.
 * Every CTA owns one status slot; every lane owns one sticky LDS slot. */
typedef struct rocke_paged_bounds
{
    rocke_value_t *status, *pages, *capacity, *cols, *k_bytes, *v_bytes;
    rocke_value_t *tid, *slot, *failures;
    int sliding_window;
} rocke_paged_bounds_t;

static inline rocke_value_t*
    rocke_bounds_ptr_param(rocke_ir_builder_t* b, const char* name, bool readonly)
{
    rocke_param_opts_t opts = {};
    if(readonly)
    {
        opts.readonly = true;
        opts.readonly_set = true;
    }
    opts.align = 4;
    opts.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, rocke_i32(), "global"), &opts);
}
static inline rocke_value_t*
    rocke_bounds_smem_alloc(rocke_ir_builder_t* b, int size, const char* name)
{
    return rocke_b_smem_alloc(b, rocke_i32(), &size, 1, name);
}
static inline rocke_value_t*
    rocke_bounds_smem_load(rocke_ir_builder_t* b, rocke_value_t* smem, rocke_value_t* index)
{
    return rocke_b_smem_load_vN(b, smem, &index, 1, rocke_i32(), 1);
}
static inline void rocke_bounds_smem_store(rocke_ir_builder_t* b,
                                           rocke_value_t* smem,
                                           rocke_value_t* index,
                                           rocke_value_t* value)
{
    rocke_b_smem_store_vN(b, smem, &index, 1, value, 1);
}
static inline rocke_value_t* rocke_bounds_masked_load(rocke_ir_builder_t* b,
                                                      rocke_value_t* ptr,
                                                      rocke_value_t* index,
                                                      rocke_value_t* mask,
                                                      rocke_value_t* fallback)
{
    return rocke_b_masked_global_load(b, ptr, index, mask, fallback, rocke_i32(), 4);
}
static inline void
    rocke_bounds_declare(rocke_ir_builder_t* B, rocke_paged_bounds_t* state, int sliding_window)
{
    state->sliding_window = sliding_window;
    rocke_value_t* t1 = rocke_bounds_ptr_param(B, "metadata_status_ptr", false);
    state->status = t1;
    rocke_value_t* t2 = rocke_b_param(B, "physical_pages", rocke_i32(), NULL);
    state->pages = t2;
    rocke_value_t* t3 = rocke_b_param(B, "max_seqlen_k", rocke_i32(), NULL);
    state->capacity = t3;
    rocke_value_t* t4 = rocke_b_param(B, "table_cols", rocke_i32(), NULL);
    state->cols = t4;
    rocke_value_t* t5 = rocke_b_param(B, "k_span_bytes", rocke_i32(), NULL);
    state->k_bytes = t5;
    rocke_value_t* t6 = rocke_b_param(B, "v_span_bytes", rocke_i32(), NULL);
    state->v_bytes = t6;
}

static inline void rocke_bounds_start(rocke_ir_builder_t* B,
                                      rocke_paged_bounds_t* state,
                                      rocke_value_t* seq,
                                      rocke_value_t* head,
                                      rocke_value_t* seg,
                                      rocke_value_t* tid,
                                      int heads,
                                      int segments)
{
    state->tid = tid;
    rocke_value_t* t1 = rocke_b_const_i32(B, heads);
    rocke_value_t* t2 = rocke_b_mul(B, seq, t1);
    rocke_value_t* t3 = rocke_b_add(B, t2, head);
    rocke_value_t* t4 = rocke_b_const_i32(B, segments);
    rocke_value_t* t5 = rocke_b_mul(B, t3, t4);
    rocke_value_t* t6 = rocke_b_add(B, t5, seg);
    state->slot = t6;
    rocke_value_t* t7 = rocke_bounds_smem_alloc(B, 64, "metadata_failures");
    state->failures = t7;
    rocke_value_t* t8 = rocke_b_const_i32(B, 0);
    rocke_bounds_smem_store(B, state->failures, tid, t8);
}

static inline void
    rocke_bounds_mark(rocke_ir_builder_t* B, rocke_paged_bounds_t* state, rocke_value_t* invalid)
{
    rocke_value_t* t1 = rocke_bounds_smem_load(B, state->failures, state->tid);
    rocke_value_t* t2 = rocke_b_vec_extract(B, t1, 0);
    rocke_value_t* old = t2;
    rocke_value_t* t3 = rocke_b_const_i32(B, 1);
    rocke_value_t* t4 = rocke_b_select(B, invalid, t3, old);
    rocke_bounds_smem_store(B, state->failures, state->tid, t4);
}

static inline void rocke_bounds_finish(rocke_ir_builder_t* B, rocke_paged_bounds_t* state)
{
    rocke_value_t* t1 = rocke_bounds_smem_load(B, state->failures, state->tid);
    rocke_value_t* t2 = rocke_b_vec_extract(B, t1, 0);
    rocke_value_t* failed = t2;
    rocke_value_t* t3 = rocke_b_const_i32(B, 0);
    rocke_value_t* t4 = rocke_b_cmp_ne(B, failed, t3);
    rocke_value_t* vote_one = rocke_b_const_i32(B, 1);
    rocke_value_t* vote_zero = rocke_b_const_i32(B, 0);
    rocke_value_t* vote_input = rocke_b_select(B, t4, vote_one, vote_zero);
    rocke_value_t* vote = rocke_b_wave_any(B, vote_input);
    rocke_value_t* result_zero = rocke_b_const_i32(B, 0);
    rocke_value_t* t5 = rocke_b_cmp_ne(B, vote, result_zero);
    rocke_value_t* any_failed = t5;
    rocke_value_t* t6 = rocke_b_const_i32(B, 0);
    rocke_value_t* t7 = rocke_b_cmp_eq(B, state->tid, t6);
    {
        rocke_if_t g8 = rocke_b_scf_if(B, t7);
        rocke_b_region_enter(B, g8.then_region);
        rocke_value_t* t9 = rocke_b_const_i32(B, 1);
        rocke_value_t* t10 = rocke_b_const_i32(B, 0);
        rocke_value_t* t11 = rocke_b_select(B, any_failed, t9, t10);
        rocke_b_global_store(B, state->status, state->slot, t11, 4);
        rocke_b_region_leave(B);
    }
}

static inline rocke_value_t* rocke_bounds_physical(rocke_ir_builder_t* B,
                                                   rocke_paged_bounds_t* state,
                                                   rocke_value_t* table,
                                                   rocke_value_t* base,
                                                   rocke_value_t* page,
                                                   rocke_value_t* live)
{
    rocke_value_t* t1 = rocke_b_const_i32(B, 0);
    rocke_value_t* t2 = rocke_b_cmp_ge(B, page, t1);
    rocke_value_t* t3 = rocke_b_cmp_lt(B, page, state->cols);
    rocke_value_t* t4 = rocke_b_land(B, t2, t3);
    rocke_value_t* t5 = rocke_b_land(B, live, t4);
    rocke_value_t* lookup_ok = t5;
    rocke_value_t* t6 = rocke_b_const_i32(B, 0);
    rocke_value_t* t7 = rocke_b_select(B, lookup_ok, page, t6);
    rocke_value_t* safe_page = t7;
    rocke_value_t* t8 = rocke_b_add(B, base, safe_page);
    rocke_value_t* t9 = rocke_b_const_i32(B, 0);
    rocke_value_t* t10 = rocke_bounds_masked_load(B, table, t8, lookup_ok, t9);
    rocke_value_t* physical = t10;
    rocke_value_t* t11 = rocke_b_const_i32(B, 0);
    rocke_value_t* t12 = rocke_b_cmp_ge(B, physical, t11);
    rocke_value_t* t13 = rocke_b_cmp_lt(B, physical, state->pages);
    rocke_value_t* t14 = rocke_b_land(B, t12, t13);
    rocke_value_t* physical_ok = t14;
    rocke_value_t* t15 = rocke_b_land(B, lookup_ok, physical_ok);
    rocke_value_t* good = t15;
    rocke_value_t* t16 = rocke_b_const_i32(B, 0);
    rocke_value_t* t17 = rocke_b_const_i32(B, 0);
    rocke_value_t* false_ = rocke_b_cmp_ne(B, t16, t17);
    rocke_value_t* t19 = rocke_b_cmp_eq(B, good, false_);
    rocke_value_t* t20 = rocke_b_land(B, live, t19);
    rocke_bounds_mark(B, state, t20);
    rocke_value_t* t21 = rocke_b_const_i32(B, 0);
    rocke_value_t* t22 = rocke_b_select(B, good, physical, t21);
    return t22;
}

static inline rocke_value_t* rocke_bounds_offset(rocke_ir_builder_t* B,
                                                 rocke_paged_bounds_t* state,
                                                 rocke_value_t* table,
                                                 rocke_value_t* base,
                                                 rocke_value_t* head,
                                                 rocke_value_t* tile,
                                                 rocke_value_t* linear,
                                                 rocke_value_t* length,
                                                 int hd,
                                                 int page,
                                                 int tile_size,
                                                 int heads)
{
    rocke_value_t* t1 = rocke_b_const_i32(B, tile_size);
    rocke_value_t* t2 = rocke_b_mul(B, tile, t1);
    rocke_value_t* t3 = rocke_b_const_i32(B, hd);
    rocke_value_t* t4 = rocke_b_div(B, linear, t3);
    rocke_value_t* t5 = rocke_b_add(B, t2, t4);
    rocke_value_t* token = t5;
    rocke_value_t* t6 = rocke_b_const_i32(B, 0);
    rocke_value_t* t7 = rocke_b_cmp_ge(B, token, t6);
    rocke_value_t* t8 = rocke_b_cmp_lt(B, token, length);
    rocke_value_t* t9 = rocke_b_land(B, t7, t8);
    rocke_value_t* live = t9;
    if(state->sliding_window > 0)
    {
        rocke_value_t* window_zero = rocke_b_const_i32(B, 0);
        rocke_value_t* window = rocke_b_const_i32(B, state->sliding_window);
        rocke_value_t* window_start_raw = rocke_b_sub(B, length, window);
        rocke_value_t* window_start = rocke_b_smax(B, window_zero, window_start_raw);
        rocke_value_t* window_ok = rocke_b_cmp_ge(B, token, window_start);
        live = rocke_b_land(B, live, window_ok);
    }
    rocke_value_t* t10 = rocke_b_const_i32(B, 0);
    rocke_value_t* t11 = rocke_b_select(B, live, token, t10);
    rocke_value_t* safe_token = t11;
    rocke_value_t* t12 = rocke_b_const_i32(B, page);
    rocke_value_t* t13 = rocke_b_div(B, safe_token, t12);
    rocke_value_t* t14 = rocke_bounds_physical(B, state, table, base, t13, live);
    rocke_value_t* physical = t14;
    rocke_value_t* t15 = rocke_b_const_i32(B, (((page * heads) * hd) * 2));
    rocke_value_t* t16 = rocke_b_mul(B, physical, t15);
    rocke_value_t* page_off = t16;
    rocke_value_t* t17 = rocke_b_const_i32(B, page);
    rocke_value_t* t18 = rocke_b_mod(B, safe_token, t17);
    rocke_value_t* t19 = rocke_b_const_i32(B, ((heads * hd) * 2));
    rocke_value_t* t20 = rocke_b_mul(B, t18, t19);
    rocke_value_t* row_off = t20;
    rocke_value_t* t21 = rocke_b_const_i32(B, (hd * 2));
    rocke_value_t* t22 = rocke_b_mul(B, head, t21);
    rocke_value_t* head_off = t22;
    rocke_value_t* t23 = rocke_b_const_i32(B, hd);
    rocke_value_t* t24 = rocke_b_mod(B, linear, t23);
    rocke_value_t* t25 = rocke_b_const_i32(B, 2);
    rocke_value_t* t26 = rocke_b_mul(B, t24, t25);
    rocke_value_t* dim_off = t26;
    rocke_value_t* t27 = rocke_b_add(B, page_off, row_off);
    rocke_value_t* t28 = rocke_b_add(B, head_off, dim_off);
    rocke_value_t* t29 = rocke_b_add(B, t27, t28);
    rocke_value_t* offset = t29;
    rocke_value_t* t30 = rocke_b_const_i32(B, 2147483647);
    rocke_value_t* t31 = rocke_b_select(B, live, offset, t30);
    return t31;
}

static inline void rocke_bounds_neutral(rocke_ir_builder_t* B,
                                        rocke_value_t* seq,
                                        rocke_value_t* head,
                                        rocke_value_t* seg,
                                        rocke_value_t* tid,
                                        rocke_value_t* out,
                                        rocke_value_t* maximum,
                                        rocke_value_t* expsum,
                                        int hd,
                                        int nqk,
                                        int nqh,
                                        int segments)
{
    for(int qh = 0; qh < nqk; ++qh)
    {
        rocke_value_t* t1 = rocke_b_const_i32(B, nqk);
        rocke_value_t* t2 = rocke_b_mul(B, head, t1);
        rocke_value_t* t3 = rocke_b_const_i32(B, qh);
        rocke_value_t* t4 = rocke_b_add(B, t2, t3);
        rocke_value_t* h = t4;
        rocke_value_t* t5 = rocke_b_const_i32(B, nqh);
        rocke_value_t* t6 = rocke_b_mul(B, seq, t5);
        rocke_value_t* t7 = rocke_b_add(B, t6, h);
        rocke_value_t* t8 = rocke_b_const_i32(B, segments);
        rocke_value_t* t9 = rocke_b_mul(B, t7, t8);
        rocke_value_t* t10 = rocke_b_add(B, t9, seg);
        rocke_value_t* ml = t10;
        rocke_value_t* t11 = rocke_b_const_i32(B, 0);
        rocke_value_t* t12 = rocke_b_cmp_eq(B, tid, t11);
        {
            rocke_if_t g13 = rocke_b_scf_if(B, t12);
            rocke_b_region_enter(B, g13.then_region);
            rocke_value_t* t14 = rocke_b_const_f32(B, -INFINITY);
            rocke_b_global_store(B, maximum, ml, t14, 4);
            rocke_value_t* t15 = rocke_b_const_f32(B, 0.0);
            rocke_b_global_store(B, expsum, ml, t15, 4);
            rocke_b_region_leave(B);
        }
        for(int chunk = 0; chunk < (hd / 64); ++chunk)
        {
            rocke_value_t* t16 = rocke_b_const_i32(B, (chunk * 64));
            rocke_value_t* t17 = rocke_b_add(B, t16, tid);
            rocke_value_t* dim = t17;
            rocke_value_t* t18 = rocke_b_const_i32(B, hd);
            rocke_value_t* t19 = rocke_b_mul(B, ml, t18);
            rocke_value_t* t20 = rocke_b_add(B, t19, dim);
            rocke_value_t* idx = t20;
            rocke_value_t* t21 = rocke_b_const_f32(B, 0.0);
            rocke_b_global_store(B, out, idx, t21, 4);
        }
    }
}

static inline void rocke_bounds_reject(rocke_ir_builder_t* B,
                                       rocke_value_t* status,
                                       rocke_value_t* seq,
                                       rocke_value_t* head,
                                       rocke_value_t* tid,
                                       rocke_value_t* out,
                                       const rocke_type_t* dtype,
                                       int hd,
                                       int nqh,
                                       int nkh,
                                       int segments)
{
    rocke_value_t* t1 = rocke_b_const_i32(B, 0);
    rocke_value_t* t2 = rocke_b_const_i32(B, 0);
    rocke_value_t* false_ = rocke_b_cmp_ne(B, t1, t2);
    rocke_value_t* failed = false_;
    rocke_value_t* t4 = rocke_b_const_i32(B, (nkh * segments));
    rocke_value_t* t5 = rocke_b_mul(B, seq, t4);
    rocke_value_t* base = t5;
    for(int chunk = 0; chunk < (((nkh * segments) + 63) / 64); ++chunk)
    {
        rocke_value_t* t6 = rocke_b_const_i32(B, (chunk * 64));
        rocke_value_t* t7 = rocke_b_add(B, t6, tid);
        rocke_value_t* slot = t7;
        rocke_value_t* t8 = rocke_b_const_i32(B, (nkh * segments));
        rocke_value_t* t9 = rocke_b_cmp_lt(B, slot, t8);
        rocke_value_t* live = t9;
        rocke_value_t* t10 = rocke_b_add(B, base, slot);
        rocke_value_t* t11 = rocke_b_const_i32(B, 0);
        rocke_value_t* t12 = rocke_bounds_masked_load(B, status, t10, live, t11);
        rocke_value_t* value = t12;
        rocke_value_t* t13 = rocke_b_const_i32(B, 0);
        rocke_value_t* t14 = rocke_b_cmp_ne(B, value, t13);
        rocke_value_t* t15 = rocke_b_lor(B, failed, t14);
        failed = t15;
    }
    rocke_value_t* vote_one = rocke_b_const_i32(B, 1);
    rocke_value_t* vote_zero = rocke_b_const_i32(B, 0);
    rocke_value_t* vote_input = rocke_b_select(B, failed, vote_one, vote_zero);
    rocke_value_t* vote = rocke_b_wave_any(B, vote_input);
    rocke_value_t* result_zero = rocke_b_const_i32(B, 0);
    rocke_value_t* t16 = rocke_b_cmp_ne(B, vote, result_zero);
    failed = t16;
    {
        rocke_if_t g17 = rocke_b_scf_if(B, failed);
        rocke_b_region_enter(B, g17.then_region);
        rocke_value_t* t18 = rocke_b_const_i32(B, nqh);
        rocke_value_t* t19 = rocke_b_mul(B, seq, t18);
        rocke_value_t* t20 = rocke_b_add(B, t19, head);
        rocke_value_t* t21 = rocke_b_const_i32(B, hd);
        rocke_value_t* t22 = rocke_b_mul(B, t20, t21);
        rocke_value_t* base_out = t22;
        rocke_value_t* t23 = rocke_b_const_f32(B, NAN);
        rocke_value_t* t24 = rocke_b_cast_f32_to(B, t23, dtype);
        rocke_value_t* nan = t24;
        for(int chunk = 0; chunk < (hd / 64); ++chunk)
        {
            rocke_value_t* t25 = rocke_b_const_i32(B, (chunk * 64));
            rocke_value_t* t26 = rocke_b_add(B, t25, tid);
            rocke_value_t* dim = t26;
            rocke_value_t* t27 = rocke_b_add(B, base_out, dim);
            rocke_b_global_store(B, out, t27, nan, 2);
        }
        rocke_b_ret(B);
        rocke_b_region_leave(B);
    }
}

#endif
