// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_sinks.py: the dQ sinks of the
 * attention backward tile step.
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_sinks.h for the
 * mapping and the byte-identity rules. Every op is created in the Python
 * evaluation order; each comment `py:` quotes the Python expression a block
 * reproduces.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_sinks.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* py: _key(x): the int itself, or ("v", id(x)) for an SSA value. */
struct sink_key
{
    bool is_val;
    int64_t i;
    const rocke_value_t* v;
};

static sink_key key_of(const rocke_attn_bwd_iv_t& x)
{
    sink_key k;
    k.is_val = x.v != NULL;
    k.i = x.v != NULL ? 0 : x.i;
    k.v = x.v;
    return k;
}

static bool key_eq(const sink_key& a, const sink_key& b)
{
    if(a.is_val != b.is_val)
        return false;
    return a.is_val ? a.v == b.v : a.i == b.i;
}

/* ====================================================================== *
 * PendingDq
 * ====================================================================== */

int rocke_attn_bwd_pending_dq_flat_len(const rocke_attn_bwd_pending_dq_t* p)
{
    return p->n_values + 2 + (p->n_heads != NULL ? 1 : 0);
}

void rocke_attn_bwd_pending_dq_flat(const rocke_attn_bwd_pending_dq_t* p, rocke_value_t** out)
{
    int k = 0;
    for(int i = 0; i < p->n_values; ++i)
        out[k++] = p->values[i];
    out[k++] = p->q0;
    out[k++] = p->h;
    if(p->n_heads != NULL)
        out[k++] = p->n_heads;
}

bool rocke_attn_bwd_pending_dq_name(
    const rocke_attn_bwd_pending_dq_t* p, const char* prefix, int i, char* out, size_t out_cap)
{
    int n;
    if(i < 0 || i >= rocke_attn_bwd_pending_dq_flat_len(p))
        return false;
    if(i < p->n_values)
        n = snprintf(out, out_cap, "%sdq%d", prefix, i);
    else if(i == p->n_values)
        n = snprintf(out, out_cap, "%sdqrow", prefix);
    else if(i == p->n_values + 1)
        n = snprintf(out, out_cap, "%sdqhead", prefix);
    else
        n = snprintf(out, out_cap, "%sdqnh", prefix);
    return n >= 0 && (size_t)n < out_cap;
}

bool rocke_attn_bwd_pending_dq_from_flat(rocke_ir_builder_t* b,
                                         const rocke_attn_bwd_pending_dq_t* p,
                                         rocke_value_t* const* vals,
                                         int n_vals,
                                         rocke_attn_bwd_pending_dq_t* out)
{
    const int n = p->n_values;
    if(n_vals != rocke_attn_bwd_pending_dq_flat_len(p))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "carried dQ values do not match the partial");
        return false;
    }
    out->values = vals;
    out->n_values = n;
    out->q0 = vals[n];
    out->h = vals[n + 1];
    out->n_heads = p->n_heads == NULL ? NULL : vals[n + 2];
    out->rows_per_head = p->rows_per_head;
    return true;
}

/* ====================================================================== *
 * Sink construction
 * ====================================================================== */

static void sink_clear(rocke_attn_bwd_dq_sink_t* s)
{
    memset(s, 0, sizeof(*s));
    s->emits_g4 = true;
}

bool rocke_attn_bwd_dq_sink_init_atomic(rocke_attn_bwd_dq_sink_t* s,
                                        rocke_ir_builder_t* b,
                                        const rocke_attn_bwd_workspace_addr_t* ws,
                                        const rocke_attn_bwd_seq_info_t* seq,
                                        rocke_value_t* scale)
{
    (void)b;
    sink_clear(s);
    s->kind = ROCKE_ATTN_BWD_SINK_ATOMIC;
    s->ws = *ws;
    s->seq = *seq;
    s->scale = scale;
    return true;
}

bool rocke_attn_bwd_dq_sink_init_packed_atomic(rocke_attn_bwd_dq_sink_t* s,
                                               rocke_ir_builder_t* b,
                                               const rocke_attn_bwd_workspace_addr_t* ws,
                                               const rocke_attn_bwd_seq_info_t* seq,
                                               rocke_value_t* scale)
{
    bool ok = rocke_attn_bwd_dq_sink_init_atomic(s, b, ws, seq, scale);
    s->kind = ROCKE_ATTN_BWD_SINK_PACKED_ATOMIC;
    return ok;
}

void rocke_attn_bwd_dq_sink_init_null(rocke_attn_bwd_dq_sink_t* s)
{
    sink_clear(s);
    s->kind = ROCKE_ATTN_BWD_SINK_NULL;
    s->emits_g4 = false;
}

bool rocke_attn_bwd_dq_sink_init_register(rocke_attn_bwd_dq_sink_t* s,
                                          rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_strided_tensor_addr_t* out,
                                          rocke_value_t* scale)
{
    (void)b;
    sink_clear(s);
    s->kind = ROCKE_ATTN_BWD_SINK_REGISTER;
    s->out = *out;
    s->scale = scale;
    return true;
}

bool rocke_attn_bwd_dq_sink_init_split_slice(rocke_ir_builder_t* b)
{
    rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "SplitSliceSink is a recorded design and is not built");
    return false;
}

void rocke_attn_bwd_dq_sink_free(rocke_attn_bwd_dq_sink_t* s)
{
    free(s->slots);
    free(s->acc);
    s->slots = NULL;
    s->acc = NULL;
    s->n_slots = 0;
    s->n_acc = 0;
}

/* ====================================================================== *
 * begin_head / finish
 * ====================================================================== */

bool rocke_attn_bwd_dq_sink_begin_head(rocke_attn_bwd_dq_sink_t* s,
                                       rocke_ir_builder_t* b,
                                       rocke_value_t* h)
{
    switch(s->kind)
    {
    case ROCKE_ATTN_BWD_SINK_ATOMIC:
        /* py: self._head = h; self._slab = self.ws.slab(h) if head_major else None */
        s->head = h;
        s->has_slab = false;
        if(s->ws.layout == ROCKE_ATTN_BWD_WS_HEAD_MAJOR)
        {
            s->slab = rocke_attn_bwd_workspace_addr_slab(
                &s->ws, rocke_attn_bwd_iv_val(h), rocke_attn_bwd_iv_int(0));
            s->has_slab = true;
        }
        return rocke_ir_builder_ok(b);
    case ROCKE_ATTN_BWD_SINK_PACKED_ATOMIC:
        s->head = h; /* rebases are per packed row (see consume) */
        return true;
    case ROCKE_ATTN_BWD_SINK_NULL:
        return true;
    case ROCKE_ATTN_BWD_SINK_REGISTER:
        if(s->head != NULL && s->head != h)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "RegisterAccumulateSink holds one head per CTA");
            return false;
        }
        s->head = h;
        return true;
    }
    return false;
}

bool rocke_attn_bwd_dq_sink_finish(rocke_attn_bwd_dq_sink_t* s, rocke_ir_builder_t* b)
{
    if(s->kind != ROCKE_ATTN_BWD_SINK_REGISTER)
        return true;
    if(!s->has_layout)
        return true;
    rocke_attn_bwd_row_slab_t slab = rocke_attn_bwd_strided_tensor_addr_slab(
        &s->out, rocke_attn_bwd_iv_val(s->head), rocke_attn_bwd_iv_int(0), false);
    rocke_value_t* length = s->out.seq.length;
    /* py: by_row.setdefault(_key(row), (row, []))[1].append((col, acc)) */
    struct group
    {
        rocke_attn_bwd_iv_t row;
        std::vector<int> idx;
    };
    std::vector<group> groups;
    for(int i = 0; i < s->n_slots; ++i)
    {
        const sink_key k = key_of(s->slots[i].row);
        size_t g = 0;
        for(; g < groups.size(); ++g)
            if(key_eq(key_of(groups[g].row), k))
                break;
        if(g == groups.size())
        {
            group ng;
            ng.row = s->slots[i].row;
            groups.push_back(ng);
        }
        groups[g].idx.push_back(i);
    }
    for(size_t g = 0; g < groups.size(); ++g)
    {
        rocke_value_t* rc = rocke_attn_bwd_as_i32(b, groups[g].row);
        rocke_value_t* q = rocke_b_add(b, s->q0, rc);
        rocke_value_t* cond = rocke_b_cmp_lt(b, q, length);
        rocke_if_t st = rocke_b_scf_if(b, cond);
        if(st.op == NULL)
            return false;
        rocke_b_region_enter(b, st.then_region);
        for(size_t j = 0; j < groups[g].idx.size(); ++j)
        {
            const int i = groups[g].idx[j];
            rocke_value_t* v = s->acc[i];
            if(s->scale != NULL)
                v = rocke_b_fmul(b, v, s->scale);
            rocke_value_t* cv = rocke_b_cast_f32_to(b, v, s->out.dtype);
            rocke_attn_bwd_row_slab_store(&slab, q, s->slots[i].col, cv);
        }
        rocke_b_region_leave(b);
    }
    return rocke_ir_builder_ok(b);
}

/* ====================================================================== *
 * consume
 * ====================================================================== */

/* py: _RowCache: per-step cache of per-row values keyed by the row expression */
struct row_cache_entry
{
    sink_key key;
    rocke_value_t* value;
};

static bool consume_atomic(rocke_attn_bwd_dq_sink_t* s,
                           rocke_ir_builder_t* b,
                           const rocke_attn_bwd_dq_slot_t* dq_frag,
                           int n_frag,
                           rocke_value_t* rows,
                           rocke_value_t* h,
                           bool clamp)
{
    rocke_value_t* last = clamp ? s->seq.last : NULL;
    rocke_value_t* base = NULL;
    rocke_attn_bwd_ws_slab_t slab;
    if(s->ws.layout == ROCKE_ATTN_BWD_WS_TOKEN_MAJOR)
    {
        /* Rows of one head are H * D apart: rebase in i64 to the step's first
         * target row so the in-slab index stays below k_m0 * H * D. */
        base = rows;
        if(last != NULL)
            base = rocke_b_smin(b, rows, last);
        slab = rocke_attn_bwd_workspace_addr_slab(
            &s->ws, rocke_attn_bwd_iv_val(h), rocke_attn_bwd_iv_val(base));
    }
    else
    {
        if(!s->has_slab || s->head != h)
            rocke_attn_bwd_dq_sink_begin_head(s, b, h);
        base = NULL;
        slab = s->slab;
    }
    std::vector<row_cache_entry> cache;
    for(int i = 0; i < n_frag; ++i)
    {
        const rocke_attn_bwd_dq_slot_t& slot = dq_frag[i];
        const sink_key k = key_of(slot.row);
        rocke_value_t* q = NULL;
        bool hit = false;
        for(size_t c = 0; c < cache.size(); ++c)
        {
            if(key_eq(cache[c].key, k))
            {
                q = cache[c].value;
                hit = true;
                break;
            }
        }
        if(!hit)
        {
            /* py: target(): q = b.add(rows, as_i32(b, slot.row)); smin; sub */
            rocke_value_t* rc = rocke_attn_bwd_as_i32(b, slot.row);
            q = rocke_b_add(b, rows, rc);
            if(last != NULL)
                q = rocke_b_smin(b, q, last);
            if(base != NULL)
                q = rocke_b_sub(b, q, base);
            row_cache_entry e;
            e.key = k;
            e.value = q;
            cache.push_back(e);
        }
        rocke_value_t* v = slot.value;
        if(s->scale != NULL)
            v = rocke_b_fmul(b, v, s->scale);
        rocke_attn_bwd_ws_slab_atomic_add(&slab, q, slot.col, v);
    }
    return rocke_ir_builder_ok(b);
}

static bool consume_packed(rocke_attn_bwd_dq_sink_t* s,
                           rocke_ir_builder_t* b,
                           const rocke_attn_bwd_dq_slot_t* dq_frag,
                           int n_frag,
                           const rocke_attn_bwd_packed_rows_t* rows,
                           bool clamp)
{
    rocke_value_t* last = clamp ? s->seq.last : NULL;
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    /* Group the slots by row so each row's 64-bit rebase is short-lived. */
    struct group
    {
        sink_key key;
        std::vector<int> idx;
    };
    std::vector<group> groups;
    for(int i = 0; i < n_frag; ++i)
    {
        const sink_key k = key_of(dq_frag[i].row);
        size_t g = 0;
        for(; g < groups.size(); ++g)
            if(key_eq(groups[g].key, k))
                break;
        if(g == groups.size())
        {
            group ng;
            ng.key = k;
            groups.push_back(ng);
        }
        groups[g].idx.push_back(i);
    }
    for(size_t g = 0; g < groups.size(); ++g)
    {
        const rocke_attn_bwd_dq_slot_t& first = dq_frag[groups[g].idx[0]];
        rocke_value_t* head = NULL;
        rocke_value_t* q = NULL;
        rocke_value_t* ok = NULL;
        rocke_attn_bwd_packed_rows_locate(rows, b, first.row, &head, &q, &ok);
        if(last != NULL)
            q = rocke_b_smin(b, q, last);
        rocke_attn_bwd_ws_slab_t slab;
        if(s->ws.layout == ROCKE_ATTN_BWD_WS_TOKEN_MAJOR)
        {
            slab = rocke_attn_bwd_workspace_addr_slab(
                &s->ws, rocke_attn_bwd_iv_val(head), rocke_attn_bwd_iv_val(q));
            q = rocke_b_const_i32(b, 0);
        }
        else
        {
            slab = rocke_attn_bwd_workspace_addr_slab(
                &s->ws, rocke_attn_bwd_iv_val(head), rocke_attn_bwd_iv_int(0));
        }
        for(size_t j = 0; j < groups[g].idx.size(); ++j)
        {
            const rocke_attn_bwd_dq_slot_t& slot = dq_frag[groups[g].idx[j]];
            rocke_value_t* v = slot.value;
            if(s->scale != NULL)
                v = rocke_b_fmul(b, v, s->scale);
            rocke_value_t* sel = rocke_b_select(b, ok, v, zero);
            rocke_attn_bwd_ws_slab_atomic_add(&slab, q, slot.col, sel);
        }
    }
    return rocke_ir_builder_ok(b);
}

static bool consume_register(rocke_attn_bwd_dq_sink_t* s,
                             rocke_ir_builder_t* b,
                             const rocke_attn_bwd_dq_slot_t* dq_frag,
                             int n_frag,
                             rocke_value_t* rows,
                             rocke_value_t* h)
{
    if(!rocke_attn_bwd_dq_sink_begin_head(s, b, h))
        return false;
    if(!s->has_layout)
    {
        s->has_layout = true;
        s->slots
            = (rocke_attn_bwd_dq_slot_t*)malloc(sizeof(*s->slots) * (size_t)(n_frag ? n_frag : 1));
        s->acc = (rocke_value_t**)malloc(sizeof(*s->acc) * (size_t)(n_frag ? n_frag : 1));
        if(s->slots == NULL || s->acc == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "register sink: out of memory");
            return false;
        }
        for(int i = 0; i < n_frag; ++i)
        {
            s->slots[i] = dq_frag[i];
            s->acc[i] = dq_frag[i].value;
        }
        s->n_slots = n_frag;
        s->n_acc = n_frag;
        s->q0 = rows;
        return true;
    }
    bool same = n_frag == s->n_slots;
    for(int i = 0; same && i < n_frag; ++i)
        same = key_eq(key_of(dq_frag[i].row), key_of(s->slots[i].row))
               && key_eq(key_of(dq_frag[i].col), key_of(s->slots[i].col));
    if(!same)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "fragment shape changed between consume calls");
        return false;
    }
    if(s->q0 != rows)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "RegisterAccumulateSink holds one q block per CTA");
        return false;
    }
    for(int i = 0; i < n_frag; ++i)
        s->acc[i] = rocke_b_fadd(b, s->acc[i], dq_frag[i].value);
    return rocke_ir_builder_ok(b);
}

bool rocke_attn_bwd_dq_sink_consume(rocke_attn_bwd_dq_sink_t* s,
                                    rocke_ir_builder_t* b,
                                    const rocke_attn_bwd_dq_slot_t* dq_frag,
                                    int n_frag,
                                    rocke_value_t* rows,
                                    const rocke_attn_bwd_packed_rows_t* packed,
                                    rocke_value_t* h,
                                    bool clamp)
{
    switch(s->kind)
    {
    case ROCKE_ATTN_BWD_SINK_ATOMIC:
        return consume_atomic(s, b, dq_frag, n_frag, rows, h, clamp);
    case ROCKE_ATTN_BWD_SINK_PACKED_ATOMIC:
        if(packed == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "the packed sink consumes a PackedRows");
            return false;
        }
        return consume_packed(s, b, dq_frag, n_frag, packed, clamp);
    case ROCKE_ATTN_BWD_SINK_NULL:
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "NullSink receives no dQ fragments (G4 is not emitted)");
        return false;
    case ROCKE_ATTN_BWD_SINK_REGISTER:
        return consume_register(s, b, dq_frag, n_frag, rows, h);
    }
    return false;
}

/* ====================================================================== *
 * RegisterAccumulateSink.state / set_state
 * ====================================================================== */

int rocke_attn_bwd_dq_sink_register_state_len(const rocke_attn_bwd_dq_sink_t* s)
{
    return s->n_acc;
}

void rocke_attn_bwd_dq_sink_register_state(const rocke_attn_bwd_dq_sink_t* s, rocke_value_t** out)
{
    for(int i = 0; i < s->n_acc; ++i)
        out[i] = s->acc[i];
}

bool rocke_attn_bwd_dq_sink_register_set_state(rocke_attn_bwd_dq_sink_t* s,
                                               rocke_ir_builder_t* b,
                                               rocke_value_t* const* values,
                                               int n)
{
    if(n != s->n_acc)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "state length mismatch");
        return false;
    }
    for(int i = 0; i < n; ++i)
        s->acc[i] = values[i];
    return true;
}

/* ====================================================================== *
 * split_slice_workspace_bytes
 * ====================================================================== */

bool rocke_attn_bwd_split_slice_workspace_bytes(rocke_ir_builder_t* b,
                                                int64_t n_split,
                                                int64_t batch,
                                                int64_t s_q,
                                                int64_t h_q,
                                                int64_t head_size,
                                                int64_t* out)
{
    const char* names[5] = {"n_split", "batch", "s_q", "h_q", "head_size"};
    const int64_t vals[5] = {n_split, batch, s_q, h_q, head_size};
    for(int i = 0; i < 5; ++i)
    {
        if(vals[i] < 0)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s must be >= 0", names[i]);
            return false;
        }
    }
    /* py ints are unbounded: a zero factor makes the result 0 whatever the rest */
    for(int i = 0; i < 5; ++i)
    {
        if(vals[i] == 0)
        {
            *out = 0;
            return true;
        }
    }
    __int128 raw = 4;
    const __int128 limit = (__int128)INT64_MAX - 255;
    for(int i = 0; i < 5; ++i)
    {
        raw *= vals[i];
        if(raw > limit)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "split slice workspace does not fit int64");
            return false;
        }
    }
    *out = (int64_t)((raw + 255) / 256 * 256);
    return true;
}
