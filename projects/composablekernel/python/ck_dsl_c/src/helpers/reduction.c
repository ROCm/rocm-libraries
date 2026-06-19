/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.reduction.c -- C99 port of selected symbols from
 * ck_dsl/helpers/reduction.py.
 *
 * Each helper reproduces its Python counterpart's ckc_b_* builder-call sequence
 * byte-faithfully (same ops, same order, same operands). The host-side control
 * structure (the halving while-loops, the tree fold, the scf_if scoping) is
 * reproduced exactly so the emitted op stream is identical to the Python.
 *
 * scf_if scoping: Python's ``with b.scf_if(cond):`` maps to
 *   ckc_if_t iff = ckc_b_scf_if(b, cond);
 *   ckc_b_region_enter(b, iff.then_region);
 *   ... body ...
 *   ckc_b_region_leave(b);
 *
 * Lifetime: every node is arena-owned (ckc_ir_builder_t.arena). Nothing is freed
 * individually; the arena bulk-frees the whole graph.
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/error.hpp"
#include "ckc/helper_ck_dsl.helpers.reduction.h"
#include "ckc/ir.h"

/* ----------------------------------------------------------------- helpers */

/* Raise the failure as a ckc::Error (mirroring the Python `raise`). The thrown
 * exception unwinds to the public entry boundary (ckc_build_*_new), which
 * catches it and records the status + message on the builder, so the C ABI is
 * unchanged. The `[[noreturn]]` lets the existing `return (T*)ckc_red_set_err()`
 * call sites stay as-is -- the cast/return is simply never reached. */
[[noreturn]] static void* ckc_red_set_err(ckc_ir_builder_t* b, ckc_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[CKC_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg[sizeof(msg) - 1] = '\0';
    ckc::raise_status(st, msg);
}

static bool ckc_red_is_f32(const ckc_value_t* v)
{
    return v != NULL && v->type != NULL && v->type->name != NULL
           && strcmp(v->type->name, "f32") == 0;
}

static bool ckc_red_is_i32(const ckc_value_t* v)
{
    return v != NULL && v->type != NULL && v->type->name != NULL
           && strcmp(v->type->name, "i32") == 0;
}

/* bit_length()-1 of a positive int (== floor(log2(n)) for n a power of two). */
static int ckc_red_bit_length(int n)
{
    int bits = 0;
    while(n > 0)
    {
        bits += 1;
        n >>= 1;
    }
    return bits;
}

/* _emit_combine(b, combine, a, c): apply the reduction combiner to two f32
 * partials. Mirrors reduction.py::_emit_combine. */
static ckc_value_t*
ckc_red_emit_combine(ckc_ir_builder_t* b, ckc_reduce_combine_t combine, ckc_value_t* a, ckc_value_t* c)
{
    switch(combine)
    {
    case CKC_REDUCE_SUM:
        return ckc_b_fadd(b, a, c);
    case CKC_REDUCE_MAX:
        return ckc_b_fmax(b, a, c);
    case CKC_REDUCE_MIN:
        return ckc_b_fmin(b, a, c);
    case CKC_REDUCE_PROD:
        return ckc_b_fmul(b, a, c);
    default:
        return (ckc_value_t*)ckc_red_set_err(b, CKC_ERR_VALUE, "unknown combine %d", (int)combine);
    }
}

/* ----------------------------------------------------- host-side selectors */

bool ckc_row_norm_needs_two_pass(int elems_per_thread, int max_cached)
{
    return elems_per_thread > max_cached;
}

/* ------------------------------------------------------------ tree_reduce */

ckc_value_t* ckc_tree_reduce(ckc_ir_builder_t* b,
                             ckc_combine_fn combine,
                             void* user,
                             ckc_value_t* const* xs,
                             int n)
{
    ckc_value_t** cur;
    ckc_value_t** nxt;
    int cur_len;
    int i;

    if(n < 1)
    {
        return (ckc_value_t*)ckc_red_set_err(b, CKC_ERR_VALUE, "tree_reduce requires at least one value");
    }

    /* cur = list(xs) -- arena-owned scratch we can overwrite each round. The two
     * buffers ping-pong; each round's output length <= input length, so n slots
     * each is sufficient. */
    cur = (ckc_value_t**)ckc_arena_alloc(&b->arena, (size_t)n * sizeof(*cur));
    nxt = (ckc_value_t**)ckc_arena_alloc(&b->arena, (size_t)n * sizeof(*nxt));
    if(cur == NULL || nxt == NULL)
    {
        return (ckc_value_t*)ckc_red_set_err(b, CKC_ERR_OOM, "tree_reduce arena alloc failed");
    }
    for(i = 0; i < n; ++i)
    {
        cur[i] = xs[i];
    }
    cur_len = n;

    while(cur_len > 1)
    {
        int nxt_len = 0;
        for(i = 0; i + 1 < cur_len; i += 2)
        {
            nxt[nxt_len++] = combine(b, cur[i], cur[i + 1], user);
        }
        if(cur_len % 2 == 1)
        {
            nxt[nxt_len++] = cur[cur_len - 1];
        }
        /* swap cur <-> nxt */
        {
            ckc_value_t** tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
        cur_len = nxt_len;
    }
    return cur[0];
}

/* _emit_combine wrapped as a ckc_combine_fn so _tree_reduce_scalars can use the
 * generic tree_reduce machinery. ``user`` carries the ckc_reduce_combine_t. */
static ckc_value_t*
ckc_red_combine_cb(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, void* user)
{
    ckc_reduce_combine_t combine = (ckc_reduce_combine_t)(intptr_t)user;
    return ckc_red_emit_combine(b, combine, a, c);
}

/* _tree_reduce_scalars(b, combine, parts): balanced tree fold of N f32 scalars
 * using the reduction combiner. Mirrors reduction.py::_tree_reduce_scalars. */
static ckc_value_t* ckc_red_tree_reduce_scalars(
    ckc_ir_builder_t* b, ckc_reduce_combine_t combine, ckc_value_t* const* parts, int n)
{
    return ckc_tree_reduce(b, ckc_red_combine_cb, (void*)(intptr_t)combine, parts, n);
}

/* _warp_xor_reduce(b, val, combine, wave_size): wave-internal XOR butterfly
 * reduce -- no LDS round-trip. Mirrors reduction.py::_warp_xor_reduce. */
static ckc_value_t* ckc_red_warp_xor_reduce(
    ckc_ir_builder_t* b, ckc_value_t* val, ckc_reduce_combine_t combine, int wave_size)
{
    int stages;
    int k;
    ckc_value_t* cur;

    if(wave_size & (wave_size - 1))
    {
        return (ckc_value_t*)ckc_red_set_err(
            b, CKC_ERR_VALUE, "wave_size %d is not a power of two", wave_size);
    }
    stages = ckc_red_bit_length(wave_size) - 1;
    cur = val;
    for(k = 0; k < stages; ++k)
    {
        ckc_value_t* remote = ckc_b_warp_shuffle_xor(b, cur, 1 << k);
        cur = ckc_red_emit_combine(b, combine, cur, remote);
    }
    return cur;
}

/* --------------------------------------------------------- block_lds_reduce */

ckc_value_t* ckc_block_lds_reduce(ckc_ir_builder_t* b,
                                  ckc_value_t* val,
                                  ckc_value_t* lds_buf,
                                  ckc_value_t* tid,
                                  int block_size,
                                  ckc_reduce_combine_t combine)
{
    int n;
    ckc_value_t* zero_idx;
    ckc_value_t* out;

    if(combine != CKC_REDUCE_SUM && combine != CKC_REDUCE_MAX && combine != CKC_REDUCE_MIN
       && combine != CKC_REDUCE_PROD)
    {
        return (ckc_value_t*)ckc_red_set_err(
            b, CKC_ERR_VALUE, "unknown combine %d; expected sum/max/min/prod", (int)combine);
    }
    if(!ckc_red_is_f32(val))
    {
        return (ckc_value_t*)ckc_red_set_err(b,
                               CKC_ERR_VALUE,
                               "block_lds_reduce expects f32 input, got %s",
                               (val && val->type) ? val->type->name : "<null>");
    }

    /* b.smem_store_vN_f32(lds_buf, [tid], val, 1) */
    {
        ckc_value_t* idx[1];
        idx[0] = tid;
        ckc_b_smem_store_vN_f32(b, lds_buf, idx, 1, val, 1);
    }
    ckc_b_sync(b);

    n = block_size;
    while(n > 1)
    {
        int half = n / 2;
        ckc_value_t* c_half = ckc_b_const_i32(b, half);
        ckc_value_t* in_first = ckc_b_cmp_lt(b, tid, c_half);
        ckc_if_t iff = ckc_b_scf_if(b, in_first);
        ckc_b_region_enter(b, iff.then_region);
        {
            ckc_value_t* j = ckc_b_add(b, tid, c_half);
            ckc_value_t* tid_idx[1];
            ckc_value_t* j_idx[1];
            ckc_value_t* a_vec;
            ckc_value_t* c_vec;
            ckc_value_t* a;
            ckc_value_t* c;
            ckc_value_t* combined;
            ckc_value_t* store_idx[1];

            tid_idx[0] = tid;
            j_idx[0] = j;
            a_vec = ckc_b_smem_load_vN_f32(b, lds_buf, tid_idx, 1, 1);
            c_vec = ckc_b_smem_load_vN_f32(b, lds_buf, j_idx, 1, 1);
            a = ckc_b_vec_extract(b, a_vec, 0);
            c = ckc_b_vec_extract(b, c_vec, 0);
            combined = ckc_red_emit_combine(b, combine, a, c);
            store_idx[0] = tid;
            ckc_b_smem_store_vN_f32(b, lds_buf, store_idx, 1, combined, 1);
        }
        ckc_b_region_leave(b);
        ckc_b_sync(b);
        n = half;
    }

    zero_idx = ckc_b_const_i32(b, 0);
    {
        ckc_value_t* idx[1];
        idx[0] = zero_idx;
        out = ckc_b_smem_load_vN_f32(b, lds_buf, idx, 1, 1);
    }
    return ckc_b_vec_extract(b, out, 0);
}

/* ---------------------------------------------------- block_lds_reduce_pair */

int ckc_block_lds_reduce_pair(ckc_ir_builder_t* b,
                              ckc_value_t* val_a,
                              ckc_value_t* val_c,
                              ckc_value_t* lds_a,
                              ckc_value_t* lds_c,
                              ckc_value_t* tid,
                              int block_size,
                              ckc_reduce_combine_t combine_a,
                              ckc_reduce_combine_t combine_c,
                              ckc_value_t** out_a,
                              ckc_value_t** out_c)
{
    int n;

    if(!ckc_red_is_f32(val_a) || !ckc_red_is_f32(val_c))
    {
        ckc_red_set_err(b, CKC_ERR_VALUE, "block_lds_reduce_pair expects f32 inputs");
        return 0;
    }

    /* b.smem_store_vN_f32(lds_a, [tid], val_a, 1)
     * b.smem_store_vN_f32(lds_c, [tid], val_c, 1) */
    {
        ckc_value_t* idx[1];
        idx[0] = tid;
        ckc_b_smem_store_vN_f32(b, lds_a, idx, 1, val_a, 1);
        ckc_b_smem_store_vN_f32(b, lds_c, idx, 1, val_c, 1);
    }
    ckc_b_sync(b);

    n = block_size;
    while(n > 1)
    {
        int half = n / 2;
        ckc_value_t* c_half = ckc_b_const_i32(b, half);
        ckc_value_t* in_first = ckc_b_cmp_lt(b, tid, c_half);
        ckc_if_t iff = ckc_b_scf_if(b, in_first);
        ckc_b_region_enter(b, iff.then_region);
        {
            ckc_value_t* j = ckc_b_add(b, tid, c_half);
            ckc_value_t* tid_idx[1];
            ckc_value_t* j_idx[1];
            ckc_value_t* a_a;
            ckc_value_t* c_a;
            ckc_value_t* a_c;
            ckc_value_t* c_c;
            ckc_value_t* store_idx[1];

            tid_idx[0] = tid;
            j_idx[0] = j;
            a_a = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_a, tid_idx, 1, 1), 0);
            c_a = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_a, j_idx, 1, 1), 0);
            a_c = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_c, tid_idx, 1, 1), 0);
            c_c = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_c, j_idx, 1, 1), 0);
            store_idx[0] = tid;
            ckc_b_smem_store_vN_f32(
                b, lds_a, store_idx, 1, ckc_red_emit_combine(b, combine_a, a_a, c_a), 1);
            ckc_b_smem_store_vN_f32(
                b, lds_c, store_idx, 1, ckc_red_emit_combine(b, combine_c, a_c, c_c), 1);
        }
        ckc_b_region_leave(b);
        ckc_b_sync(b);
        n = half;
    }

    /* Python:
     *   out_a = b.vec_extract(b.smem_load_vN_f32(lds_a, b.const_i32(0), n=1), 0)
     *   out_c = b.vec_extract(b.smem_load_vN_f32(lds_c, b.const_i32(0), n=1), 0)
     * Each call spells a FRESH b.const_i32(0); the builder has no const cache,
     * so reusing one const here would drop an SSA id and shift every later id.
     * Emit a separate const per load to keep the id stream byte-identical. */
    {
        ckc_value_t* idx_a[1];
        ckc_value_t* idx_c[1];
        idx_a[0] = ckc_b_const_i32(b, 0);
        *out_a = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_a, idx_a, 1, 1), 0);
        idx_c[0] = ckc_b_const_i32(b, 0);
        *out_c = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_c, idx_c, 1, 1), 0);
    }
    return 1;
}

/* -------------------------------- block_lds_reduce_with_wave_prologue */

ckc_value_t* ckc_block_lds_reduce_with_wave_prologue(ckc_ir_builder_t* b,
                                                     ckc_value_t* val,
                                                     ckc_value_t* lds_buf,
                                                     ckc_value_t* tid,
                                                     int block_size,
                                                     ckc_reduce_combine_t combine,
                                                     int wave_size)
{
    ckc_value_t* warp_partial;
    int num_warps;
    ckc_value_t* c_wave;
    ckc_value_t* lane;
    ckc_value_t* warp;
    ckc_value_t** parts;
    int w;

    if(!ckc_red_is_f32(val))
    {
        return (ckc_value_t*)ckc_red_set_err(
            b,
            CKC_ERR_VALUE,
            "block_lds_reduce_with_wave_prologue expects f32 input, got %s",
            (val && val->type) ? val->type->name : "<null>");
    }

    warp_partial = ckc_red_warp_xor_reduce(b, val, combine, wave_size);

    num_warps = block_size / wave_size;
    if(num_warps == 1)
    {
        return warp_partial;
    }

    c_wave = ckc_b_const_i32(b, wave_size);
    lane = ckc_b_mod(b, tid, c_wave);
    warp = ckc_b_div(b, tid, c_wave);
    {
        ckc_if_t iff = ckc_b_scf_if(b, ckc_b_cmp_eq(b, lane, ckc_b_const_i32(b, 0)));
        ckc_b_region_enter(b, iff.then_region);
        {
            ckc_value_t* idx[1];
            idx[0] = warp;
            ckc_b_smem_store_vN_f32(b, lds_buf, idx, 1, warp_partial, 1);
        }
        ckc_b_region_leave(b);
    }
    ckc_b_sync(b);

    parts = (ckc_value_t**)ckc_arena_alloc(&b->arena, (size_t)num_warps * sizeof(*parts));
    if(parts == NULL)
    {
        return (ckc_value_t*)ckc_red_set_err(b, CKC_ERR_OOM, "wave prologue parts alloc failed");
    }
    for(w = 0; w < num_warps; ++w)
    {
        ckc_value_t* idx[1];
        ckc_value_t* v_vec;
        idx[0] = ckc_b_const_i32(b, w);
        v_vec = ckc_b_smem_load_vN_f32(b, lds_buf, idx, 1, 1);
        parts[w] = ckc_b_vec_extract(b, v_vec, 0);
    }
    return ckc_red_tree_reduce_scalars(b, combine, parts, num_warps);
}

/* ----------------------------------------------------- welford_block_reduce */

int ckc_welford_block_reduce(ckc_ir_builder_t* b,
                             ckc_value_t* sum_val,
                             ckc_value_t* sum_sq_val,
                             int count_val,
                             ckc_value_t* lds_sum,
                             ckc_value_t* lds_sumsq,
                             ckc_value_t* tid,
                             int block_size,
                             ckc_value_t** out_mean,
                             ckc_value_t** out_var)
{
    ckc_value_t* total_sum = NULL;
    ckc_value_t* total_sumsq = NULL;
    double n_total;
    ckc_value_t* inv_n;
    ckc_value_t* mean;
    ckc_value_t* sq_mean;
    ckc_value_t* var;

    if(!ckc_block_lds_reduce_pair(b,
                                  sum_val,
                                  sum_sq_val,
                                  lds_sum,
                                  lds_sumsq,
                                  tid,
                                  block_size,
                                  CKC_REDUCE_SUM,
                                  CKC_REDUCE_SUM,
                                  &total_sum,
                                  &total_sumsq))
    {
        return 0;
    }

    n_total = (double)((double)count_val * (double)block_size);
    inv_n = ckc_b_const_f32(b, 1.0 / n_total);
    mean = ckc_b_fmul(b, total_sum, inv_n);
    sq_mean = ckc_b_fmul(b, total_sumsq, inv_n);
    var = ckc_b_fsub(b, sq_mean, ckc_b_fmul(b, mean, mean));
    *out_mean = mean;
    *out_var = var;
    return 1;
}

/* -------------------------------------------- welford_block_reduce_stable */

int ckc_welford_block_reduce_stable(ckc_ir_builder_t* b,
                                    ckc_value_t* mean_val,
                                    ckc_value_t* m2_val,
                                    ckc_value_t* count_val,
                                    ckc_value_t* lds_mean,
                                    ckc_value_t* lds_m2,
                                    ckc_value_t* lds_count,
                                    ckc_value_t* tid,
                                    int block_size,
                                    ckc_value_t** out_mean,
                                    ckc_value_t** out_var)
{
    ckc_value_t* zero;
    int n;
    ckc_value_t* mean_out;
    ckc_value_t* m2_out;
    ckc_value_t* count_out;
    ckc_value_t* var_out;

    if(!ckc_red_is_f32(mean_val))
    {
        ckc_red_set_err(b, CKC_ERR_VALUE, "welford_block_reduce_stable expects f32 mean_val");
        return 0;
    }
    if(!ckc_red_is_f32(m2_val))
    {
        ckc_red_set_err(b, CKC_ERR_VALUE, "welford_block_reduce_stable expects f32 m2_val");
        return 0;
    }
    if(!ckc_red_is_f32(count_val))
    {
        ckc_red_set_err(b, CKC_ERR_VALUE, "welford_block_reduce_stable expects f32 count_val");
        return 0;
    }

    {
        ckc_value_t* idx[1];
        idx[0] = tid;
        ckc_b_smem_store_vN_f32(b, lds_mean, idx, 1, mean_val, 1);
        ckc_b_smem_store_vN_f32(b, lds_m2, idx, 1, m2_val, 1);
        ckc_b_smem_store_vN_f32(b, lds_count, idx, 1, count_val, 1);
    }
    ckc_b_sync(b);

    zero = ckc_b_const_f32(b, 0.0);

    n = block_size;
    while(n > 1)
    {
        int half = n / 2;
        ckc_value_t* c_half = ckc_b_const_i32(b, half);
        ckc_value_t* in_first = ckc_b_cmp_lt(b, tid, c_half);
        ckc_if_t iff = ckc_b_scf_if(b, in_first);
        ckc_b_region_enter(b, iff.then_region);
        {
            ckc_value_t* j = ckc_b_add(b, tid, c_half);
            ckc_value_t* tid_idx[1];
            ckc_value_t* j_idx[1];
            ckc_value_t* mean_a;
            ckc_value_t* m2_a;
            ckc_value_t* cnt_a;
            ckc_value_t* mean_b;
            ckc_value_t* m2_b;
            ckc_value_t* cnt_b;
            ckc_value_t* count;
            ckc_value_t* is_empty;
            ckc_value_t* ratio;
            ckc_value_t* count_b_over_count;
            ckc_value_t* delta;
            ckc_value_t* new_mean;
            ckc_value_t* dd;
            ckc_value_t* cross;
            ckc_value_t* new_m2;
            ckc_value_t* store_idx[1];

            tid_idx[0] = tid;
            j_idx[0] = j;
            mean_a = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_mean, tid_idx, 1, 1), 0);
            m2_a = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_m2, tid_idx, 1, 1), 0);
            cnt_a = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_count, tid_idx, 1, 1), 0);
            mean_b = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_mean, j_idx, 1, 1), 0);
            m2_b = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_m2, j_idx, 1, 1), 0);
            cnt_b = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_count, j_idx, 1, 1), 0);

            /* count = count_a + count_b */
            count = ckc_b_fadd(b, cnt_a, cnt_b);
            /* count_b_over_count = count == 0 ? 0 : count_b / count */
            is_empty = ckc_b_fcmp(b, "oeq", count, zero);
            ratio = ckc_b_fmul(b, cnt_b, ckc_b_rcp(b, count));
            count_b_over_count = ckc_b_select(b, is_empty, zero, ratio);
            /* delta = mean_b - mean_a */
            delta = ckc_b_fsub(b, mean_b, mean_a);
            /* mean_a += delta * count_b_over_count */
            new_mean = ckc_b_fadd(b, mean_a, ckc_b_fmul(b, delta, count_b_over_count));
            /* M2_a += M2_b + delta*delta * count_a * count_b_over_count */
            dd = ckc_b_fmul(b, delta, delta);
            cross = ckc_b_fmul(b, ckc_b_fmul(b, dd, cnt_a), count_b_over_count);
            new_m2 = ckc_b_fadd(b, m2_a, ckc_b_fadd(b, m2_b, cross));

            store_idx[0] = tid;
            ckc_b_smem_store_vN_f32(b, lds_mean, store_idx, 1, new_mean, 1);
            ckc_b_smem_store_vN_f32(b, lds_m2, store_idx, 1, new_m2, 1);
            ckc_b_smem_store_vN_f32(b, lds_count, store_idx, 1, count, 1);
        }
        ckc_b_region_leave(b);
        ckc_b_sync(b);
        n = half;
    }

    /* Python:
     *   mean_out  = vec_extract(smem_load_vN_f32(lds_mean,  b.const_i32(0), 1), 0)
     *   m2_out    = vec_extract(smem_load_vN_f32(lds_m2,    b.const_i32(0), 1), 0)
     *   count_out = vec_extract(smem_load_vN_f32(lds_count, b.const_i32(0), 1), 0)
     * Each read allocates a FRESH b.const_i32(0); the literal-0 index is
     * folded inline by lowering but still advances the SSA-id counter, so a
     * single hoisted zero would shift every later id by 2. Allocate one per
     * read to keep SSA numbering byte-identical with Python. */
    {
        ckc_value_t* idx[1];
        idx[0] = ckc_b_const_i32(b, 0);
        mean_out = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_mean, idx, 1, 1), 0);
        idx[0] = ckc_b_const_i32(b, 0);
        m2_out = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_m2, idx, 1, 1), 0);
        idx[0] = ckc_b_const_i32(b, 0);
        count_out = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_count, idx, 1, 1), 0);
    }
    var_out = ckc_b_fmul(b, m2_out, ckc_b_rcp(b, count_out));
    *out_mean = mean_out;
    *out_var = var_out;
    return 1;
}

/* ---------------------------------------------- block_lds_reduce_with_index */

int ckc_block_lds_reduce_with_index(ckc_ir_builder_t* b,
                                    ckc_value_t* val,
                                    ckc_value_t* idx,
                                    ckc_value_t* lds_val,
                                    ckc_value_t* lds_idx,
                                    ckc_value_t* tid,
                                    int block_size,
                                    ckc_index_combine_t combine,
                                    ckc_value_t** out_val,
                                    ckc_value_t** out_idx)
{
    int cluster_len_shift;
    int shift;
    ckc_value_t* zero_idx;
    ckc_value_t* out_val_v;
    ckc_value_t* out_idx_v;

    if(combine != CKC_INDEX_ARGMAX && combine != CKC_INDEX_ARGMIN)
    {
        ckc_red_set_err(b,
                        CKC_ERR_VALUE,
                        "unknown index combine %d; expected argmax or argmin",
                        (int)combine);
        return 0;
    }
    if(!ckc_red_is_f32(val))
    {
        ckc_red_set_err(b, CKC_ERR_VALUE, "block_lds_reduce_with_index expects f32 val");
        return 0;
    }
    if(!ckc_red_is_i32(idx))
    {
        ckc_red_set_err(b, CKC_ERR_VALUE, "block_lds_reduce_with_index expects i32 idx");
        return 0;
    }
    if(block_size & (block_size - 1))
    {
        ckc_red_set_err(b,
                        CKC_ERR_VALUE,
                        "block_lds_reduce_with_index needs power-of-two block_size, got %d",
                        block_size);
        return 0;
    }

    /* b.smem_store_vN_f32(lds_val, [tid], val, 1)
     * b.smem_store_vN_f32(lds_idx, [tid], b.bitcast(idx, F32), 1) */
    {
        ckc_value_t* sidx[1];
        sidx[0] = tid;
        ckc_b_smem_store_vN_f32(b, lds_val, sidx, 1, val, 1);
        ckc_b_smem_store_vN_f32(b, lds_idx, sidx, 1, ckc_b_bitcast(b, idx, ckc_f32()), 1);
    }
    ckc_b_sync(b);

    /* CK doubling tree: indOffset = 1 << I; only lanes with
     * tid % (indOffset*2) == 0 merge slot tid with slot tid + indOffset. */
    cluster_len_shift = ckc_red_bit_length(block_size) - 1;
    for(shift = 0; shift < cluster_len_shift; ++shift)
    {
        int ind_offset = 1 << shift;
        ckc_value_t* c_off = ckc_b_const_i32(b, ind_offset);
        ckc_value_t* c_mod = ckc_b_const_i32(b, ind_offset * 2);
        ckc_value_t* participates =
            ckc_b_cmp_eq(b, ckc_b_mod(b, tid, c_mod), ckc_b_const_i32(b, 0));
        ckc_if_t iff = ckc_b_scf_if(b, participates);
        ckc_b_region_enter(b, iff.then_region);
        {
            ckc_value_t* j = ckc_b_add(b, tid, c_off);
            ckc_value_t* tid_idx[1];
            ckc_value_t* j_idx[1];
            ckc_value_t* v_a;
            ckc_value_t* v_b;
            ckc_value_t* i_a;
            ckc_value_t* i_b;
            ckc_value_t* changed;
            ckc_value_t* new_val;
            ckc_value_t* new_idx;
            ckc_value_t* store_idx[1];

            tid_idx[0] = tid;
            j_idx[0] = j;
            v_a = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_val, tid_idx, 1, 1), 0);
            v_b = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_val, j_idx, 1, 1), 0);
            i_a = ckc_b_bitcast(
                b, ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_idx, tid_idx, 1, 1), 0),
                ckc_i32());
            i_b = ckc_b_bitcast(
                b, ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_idx, j_idx, 1, 1), 0),
                ckc_i32());

            /* strict-improvement test (the `changed` flag): argmax changed when
             * a < b; argmin changed when a > b. Ties never change. */
            if(combine == CKC_INDEX_ARGMAX)
            {
                changed = ckc_b_fcmp(b, "olt", v_a, v_b);
            }
            else
            {
                changed = ckc_b_fcmp(b, "ogt", v_a, v_b);
            }

            new_val = ckc_b_select(b, changed, v_b, v_a);
            new_idx = ckc_b_select(b, changed, i_b, i_a);

            store_idx[0] = tid;
            ckc_b_smem_store_vN_f32(b, lds_val, store_idx, 1, new_val, 1);
            ckc_b_smem_store_vN_f32(
                b, lds_idx, store_idx, 1, ckc_b_bitcast(b, new_idx, ckc_f32()), 1);
        }
        ckc_b_region_leave(b);
        ckc_b_sync(b);
    }

    zero_idx = ckc_b_const_i32(b, 0);
    {
        ckc_value_t* sidx[1];
        sidx[0] = zero_idx;
        out_val_v = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_val, sidx, 1, 1), 0);
        out_idx_v = ckc_b_bitcast(
            b, ckc_b_vec_extract(b, ckc_b_smem_load_vN_f32(b, lds_idx, sidx, 1, 1), 0),
            ckc_i32());
    }
    *out_val = out_val_v;
    *out_idx = out_idx_v;
    return 1;
}
