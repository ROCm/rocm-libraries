// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_counted.py: the counted
 * in-loop waits of the attention backward q step.
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_counted.h for the
 * mapping. Every op is created in the Python evaluation order.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_counted.h"

#include <stdlib.h>
#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

const char* const ROCKE_ATTN_BWD_COUNTED_ISSUES[ROCKE_ATTN_BWD_COUNTED_N_ISSUES]
    = {"early", "late"};

/* py: a // b (floor division); b != 0 */
static int64_t floor_div(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        q -= 1;
    return q;
}

int rocke_attn_bwd_tile_chunks_count(
    rocke_ir_builder_t* b, int rows, int cols, int threads, int vec)
{
    if(vec == 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tile_chunks: vec must not be 0");
        return -1;
    }
    const int64_t per_row = floor_div(cols, vec);
    const int64_t total = (int64_t)rows * per_row;
    return rocke_attn_bwd_deal_chunks_count(b, total, threads);
}

bool rocke_attn_bwd_tile_chunks(rocke_ir_builder_t* b,
                                int rows,
                                int cols,
                                rocke_value_t* tid,
                                int threads,
                                int vec,
                                rocke_attn_bwd_tile_chunk_t* out,
                                int out_cap)
{
    const int n = rocke_attn_bwd_tile_chunks_count(b, rows, cols, threads, vec);
    if(n < 0)
        return false;
    if(n > out_cap)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "tile_chunks: %d chunks exceed the buffer (%d)", n, out_cap);
        return false;
    }
    const int per_row = (int)floor_div(cols, vec);
    const int64_t total64 = (int64_t)rows * per_row;
    if(total64 > INT32_MAX || total64 < INT32_MIN)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tile_chunks: rows*cols/vec does not fit int32");
        return false;
    }
    const int total = (int)total64;
    for(int it = 0; it < n; ++it)
    {
        bool tail = false;
        rocke_value_t* chunk = rocke_attn_bwd_deal_chunks_round(b, tid, total, threads, it, &tail);
        rocke_value_t* src = chunk;
        if(tail)
        {
            rocke_value_t* cm = rocke_b_const_i32(b, total - 1);
            src = rocke_b_smin(b, chunk, cm);
        }
        rocke_value_t* r = NULL;
        rocke_value_t* c = NULL;
        rocke_attn_bwd_chunk_coords(b, src, per_row, vec, &r, &c);
        out[it].row = r;
        out[it].col = c;
        out[it].guard = tail ? chunk : NULL;
        out[it].total = total;
        if(!rocke_ir_builder_ok(b))
            return false;
    }
    return rocke_ir_builder_ok(b);
}

bool rocke_attn_bwd_issue_tile_loads(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_tile_loader_t* loads,
                                     int n_loads,
                                     const rocke_attn_bwd_tile_chunk_t* chunks,
                                     int n_chunks,
                                     rocke_value_t** out,
                                     int out_cap)
{
    if((int64_t)n_loads * n_chunks > out_cap)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "issue_tile_loads: result buffer too small");
        return false;
    }
    int k = 0;
    for(int l = 0; l < n_loads; ++l)
        for(int c = 0; c < n_chunks; ++c)
            out[k++] = loads[l].fn(loads[l].ctx, chunks[c].row, chunks[c].col);
    return rocke_ir_builder_ok(b);
}

bool rocke_attn_bwd_store_tile(rocke_ir_builder_t* b,
                               const rocke_attn_bwd_tile_view_t* views,
                               int n_views,
                               const rocke_attn_bwd_tile_chunk_t* chunks,
                               int n_chunks,
                               rocke_value_t* const* values,
                               int n_values,
                               int vec)
{
    if((int64_t)n_values != (int64_t)n_views * n_chunks)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "staged values do not match the tile chunks");
        return false;
    }
    int k = 0;
    for(int v = 0; v < n_views; ++v)
    {
        for(int c = 0; c < n_chunks; ++c)
        {
            rocke_value_t* val = values[k++];
            const rocke_attn_bwd_tile_chunk_t* ch = &chunks[c];
            if(ch->guard == NULL)
            {
                views[v].fn(views[v].ctx, b, ch->row, ch->col, val, vec);
            }
            else
            {
                rocke_value_t* ct = rocke_b_const_i32(b, ch->total);
                rocke_value_t* cond = rocke_b_cmp_lt(b, ch->guard, ct);
                rocke_if_t st = rocke_b_scf_if(b, cond);
                if(st.op == NULL)
                    return false;
                rocke_b_region_enter(b, st.then_region);
                views[v].fn(views[v].ctx, b, ch->row, ch->col, val, vec);
                rocke_b_region_leave(b);
            }
            if(!rocke_ir_builder_ok(b))
                return false;
        }
    }
    return rocke_ir_builder_ok(b);
}

bool rocke_attn_bwd_zero_like(rocke_ir_builder_t* b,
                              const rocke_type_t* const* types,
                              int n,
                              rocke_value_t** out)
{
    for(int i = 0; i < n; ++i)
    {
        const rocke_type_t* t = types[i];
        if(t != NULL && rocke_type_eq(t, rocke_f32()))
        {
            out[i] = rocke_b_const_f32(b, 0.0);
        }
        else if(t != NULL && t->kind == ROCKE_TYPE_VECTOR)
        {
            out[i] = rocke_b_zero_vec(b, t->elem, t->count);
        }
        else
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "no zero for staged type %s", t ? t->name : "(null)");
            return false;
        }
    }
    return rocke_ir_builder_ok(b);
}

int rocke_attn_bwd_staged_step_flat_len(const rocke_attn_bwd_staged_step_t* s)
{
    return s->n_stats + s->n_qdo;
}

void rocke_attn_bwd_staged_step_flat(const rocke_attn_bwd_staged_step_t* s, rocke_value_t** out)
{
    for(int i = 0; i < s->n_stats; ++i)
        out[i] = s->stats[i];
    for(int i = 0; i < s->n_qdo; ++i)
        out[s->n_stats + i] = s->qdo[i];
}

rocke_attn_bwd_staged_step_t
    rocke_attn_bwd_staged_step_from_flat(rocke_value_t* const* values, int n_values, int n_stats)
{
    rocke_attn_bwd_staged_step_t s;
    int ns = n_stats < 0 ? 0 : n_stats;
    if(ns > n_values)
        ns = n_values;
    s.stats = values;
    s.n_stats = ns;
    s.qdo = values + ns;
    s.n_qdo = n_values - ns;
    return s;
}

bool rocke_attn_bwd_counted_waits_init(rocke_attn_bwd_counted_waits_t* w,
                                       rocke_ir_builder_t* b,
                                       int n_atomics)
{
    memset(w, 0, sizeof(*w));
    if(n_atomics < 0 || n_atomics > ROCKE_ATTN_BWD_COUNTED_VMCNT_MAX)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "%d dQ atomics per wave and step exceed the vmcnt field (%d)",
                        n_atomics,
                        ROCKE_ATTN_BWD_COUNTED_VMCNT_MAX);
        return false;
    }
    w->n_atomics = n_atomics;
    w->since_load = n_atomics;
    w->issued = 0;
    return true;
}

void rocke_attn_bwd_counted_waits_step_entry(rocke_attn_bwd_counted_waits_t* w)
{
    w->since_load = w->n_atomics;
    w->issued = 0;
    w->n_waits = 0; /* py: self.waits = [] */
}

void rocke_attn_bwd_counted_waits_loads(rocke_attn_bwd_counted_waits_t* w)
{
    w->since_load = 0;
}

void rocke_attn_bwd_counted_waits_atomics(rocke_attn_bwd_counted_waits_t* w, int n)
{
    w->since_load += n;
    w->issued += n;
}

bool rocke_attn_bwd_counted_waits_end_step(rocke_attn_bwd_counted_waits_t* w, rocke_ir_builder_t* b)
{
    if(w->issued != w->n_atomics)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "the q step issued %d dQ atomics, the count model assumes %d",
                        w->issued,
                        w->n_atomics);
        return false;
    }
    return true;
}

void rocke_attn_bwd_counted_waits_barrier(rocke_attn_bwd_counted_waits_t* w, rocke_ir_builder_t* b)
{
    rocke_attn_bwd_counted_waits_barrier_fence(w, b, NULL);
}

void rocke_attn_bwd_counted_waits_barrier_fence(rocke_attn_bwd_counted_waits_t* w,
                                                rocke_ir_builder_t* b,
                                                const char* fence)
{
    const int n = w->since_load;
    if(w->n_waits == w->cap_waits)
    {
        const int cap = w->cap_waits ? 2 * w->cap_waits : 16;
        int* p = (int*)realloc(w->waits, (size_t)cap * sizeof(int));
        if(p == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "counted waits: out of memory");
            return;
        }
        w->waits = p;
        w->cap_waits = cap;
    }
    w->waits[w->n_waits++] = n;
    rocke_b_s_waitcnt(b, n, 0, -1);
    rocke_b_s_barrier_bare_fence(b, fence);
}

void rocke_attn_bwd_counted_waits_free(rocke_attn_bwd_counted_waits_t* w)
{
    free(w->waits);
    w->waits = NULL;
    w->n_waits = 0;
    w->cap_waits = 0;
}
