// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_paged.py: the paged address
 * generator of the unified attention backward main kernel.
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_paged.h for the
 * mapping and the byte-identity rules. Every op is created in the Python
 * evaluation order; each comment `py:` quotes the Python expression a block
 * reproduces.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_paged.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* py: idx if k == 0 else b.add(idx, b.const_i32(k)) */
static rocke_value_t* idx_plus(rocke_ir_builder_t* b, rocke_value_t* idx, int k)
{
    if(k == 0)
        return idx;
    rocke_value_t* c = rocke_b_const_i32(b, k);
    return rocke_b_add(b, idx, c);
}

rocke_value_t* rocke_attn_bwd_live_blocks(rocke_ir_builder_t* b,
                                          rocke_value_t* length,
                                          rocke_attn_bwd_iv_t page)
{
    rocke_value_t* p = rocke_attn_bwd_as_i32(b, page);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* pm1 = rocke_b_sub(b, p, c1);
    rocke_value_t* sum = rocke_b_add(b, length, pm1);
    return rocke_b_div(b, sum, p);
}

/* py: _guarded_entry: row_ptr[j] when j < n_blocks, else 0 without a read. */
static rocke_value_t* guarded_entry(
    rocke_ir_builder_t* b, rocke_value_t* row_ptr, rocke_value_t* j, rocke_value_t* n_blocks, int n)
{
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    /* trip = b.select(b.cmp_lt(j, n_blocks), b.const_i32(1), zero) */
    rocke_value_t* lt = rocke_b_cmp_lt(b, j, n_blocks);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* trip = rocke_b_select(b, lt, c1, zero);
    char name[48];
    char iv[48];
    snprintf(name, sizeof(name), "page_id%d", n);
    snprintf(iv, sizeof(iv), "bt_i%d", n);
    rocke_value_t* step = rocke_b_const_i32(b, 1);
    rocke_iter_arg_t args[1] = {{name, zero}};
    rocke_for_t loop = rocke_b_scf_for_iter(b, zero, trip, step, args, 1, iv, false, true);
    if(loop.op == NULL)
        return NULL;
    rocke_b_region_enter(b, loop.body);
    rocke_value_t* ld = rocke_b_global_load_i32(b, row_ptr, j, 4);
    rocke_value_t* y[1] = {ld};
    rocke_b_scf_yield(b, y, 1);
    rocke_b_region_leave(b);
    return loop.op->results[0];
}

bool rocke_attn_bwd_paged_kv_tile_init(rocke_attn_bwd_paged_kv_tile_t* t,
                                       rocke_ir_builder_t* b,
                                       rocke_value_t* bt_ptr,
                                       rocke_attn_bwd_iv_t bt_stride,
                                       rocke_value_t* seq_idx,
                                       const rocke_attn_bwd_seq_info_t* seq,
                                       rocke_value_t* k0,
                                       int k_n0,
                                       rocke_attn_bwd_iv_t page,
                                       const int* sub_page,
                                       rocke_value_t* num_blocks,
                                       int* reads)
{
    memset(t, 0, sizeof(*t));
    if(sub_page != NULL)
    {
        const int sp = *sub_page;
        if(sp <= 0 || sp % 16 || k_n0 % sp || sp >= k_n0)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "sub_page=%d must be a multiple of 16 dividing block_n=%d and below it",
                            sp,
                            k_n0);
            return false;
        }
        if(page.v != NULL || page.i != sp)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "the sub page class uses the compile-time page");
            return false;
        }
    }
    int local_reads = 0;
    if(reads == NULL)
        reads = &local_reads;
    t->b = b;
    t->k0 = k0;
    t->k_n0 = k_n0;
    t->has_sub_page = sub_page != NULL;
    t->sub_page = sub_page != NULL ? *sub_page : 0;
    t->page = rocke_attn_bwd_as_i32(b, page);
    t->seq = *seq;
    t->n_blocks = rocke_attn_bwd_live_blocks(b, seq->length, page);
    t->cover = rocke_b_mul(b, t->n_blocks, t->page);
    /* py: off = _scaled64(b, as_i64(b, seq_idx), bt_stride) */
    rocke_value_t* s64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(seq_idx), false);
    rocke_value_t* off = rocke_attn_bwd_scaled64(b, s64, bt_stride);
    /* py: row = b.global_ptr_add(bt_ptr, b.mul(off, b.const_i64(4))) */
    rocke_value_t* c4 = rocke_b_const_i64(b, 4);
    rocke_value_t* bytes = rocke_b_mul(b, off, c4);
    rocke_value_t* row = rocke_b_global_ptr_add(b, bt_ptr, bytes);
    const int n_pieces = sub_page == NULL ? 1 : k_n0 / *sub_page;
    t->page_ids = (rocke_value_t**)malloc(sizeof(rocke_value_t*) * (size_t)n_pieces);
    if(t->page_ids == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "paged tile: out of memory");
        return false;
    }
    t->n_pieces = n_pieces;
    if(sub_page == NULL)
    {
        /* py: j = b.div(k0, self.page); self.in0 = b.sub(k0, b.mul(j, self.page)) */
        rocke_value_t* j = rocke_b_div(b, k0, t->page);
        rocke_value_t* jp = rocke_b_mul(b, j, t->page);
        t->in0 = rocke_b_sub(b, k0, jp);
        t->page_ids[0] = guarded_entry(b, row, j, t->n_blocks, (*reads)++);
    }
    else
    {
        rocke_value_t* csp = rocke_b_const_i32(b, *sub_page);
        rocke_value_t* j0 = rocke_b_div(b, k0, csp);
        t->in0 = NULL;
        for(int i = 0; i < n_pieces; ++i)
        {
            rocke_value_t* j = j0;
            if(i != 0)
            {
                rocke_value_t* ci = rocke_b_const_i32(b, i);
                j = rocke_b_add(b, j0, ci);
            }
            t->page_ids[i] = guarded_entry(b, row, j, t->n_blocks, (*reads)++);
        }
    }
    t->guarded = num_blocks != NULL;
    if(t->guarded)
    {
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_value_t* bad = rocke_b_const_i32(b, -1);
        rocke_value_t* nb = num_blocks;
        for(int i = 0; i < n_pieces; ++i)
        {
            rocke_value_t* pid = t->page_ids[i];
            rocke_value_t* ge = rocke_b_cmp_ge(b, pid, zero);
            rocke_value_t* lt = rocke_b_cmp_lt(b, pid, nb);
            rocke_value_t* both = rocke_b_land(b, ge, lt);
            t->page_ids[i] = rocke_b_select(b, both, pid, bad);
        }
    }
    return rocke_ir_builder_ok(b);
}

void rocke_attn_bwd_paged_kv_tile_free(rocke_attn_bwd_paged_kv_tile_t* t)
{
    free(t->page_ids);
    t->page_ids = NULL;
    t->n_pieces = 0;
}

void rocke_attn_bwd_paged_kv_tile_locate(const rocke_attn_bwd_paged_kv_tile_t* t,
                                         rocke_value_t* lr,
                                         rocke_value_t** out_pid,
                                         rocke_value_t** out_slot)
{
    rocke_ir_builder_t* b = t->b;
    if(!t->has_sub_page)
    {
        *out_pid = t->page_ids[0];
        *out_slot = rocke_b_add(b, t->in0, lr);
        return;
    }
    const int p = t->sub_page;
    rocke_value_t* cp = rocke_b_const_i32(b, p);
    rocke_value_t* piece = rocke_b_div(b, lr, cp);
    rocke_value_t* cp2 = rocke_b_const_i32(b, p);
    rocke_value_t* pp = rocke_b_mul(b, piece, cp2);
    rocke_value_t* slot = rocke_b_sub(b, lr, pp);
    rocke_value_t* pid = t->page_ids[t->n_pieces - 1];
    for(int i = t->n_pieces - 2; i >= 0; --i)
    {
        rocke_value_t* ci = rocke_b_const_i32(b, i);
        rocke_value_t* eq = rocke_b_cmp_eq(b, piece, ci);
        pid = rocke_b_select(b, eq, t->page_ids[i], pid);
    }
    *out_pid = pid;
    *out_slot = slot;
}

void rocke_attn_bwd_paged_kv_tile_checked(const rocke_attn_bwd_paged_kv_tile_t* t,
                                          rocke_value_t* pid,
                                          rocke_value_t** out_id,
                                          rocke_value_t** out_ok)
{
    if(!t->guarded)
    {
        *out_id = pid;
        *out_ok = NULL;
        return;
    }
    rocke_ir_builder_t* b = t->b;
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* id = rocke_b_smax(b, pid, zero);
    rocke_value_t* ok = rocke_b_cmp_ge(b, pid, zero);
    *out_id = id;
    *out_ok = ok;
}

void rocke_attn_bwd_paged_kv_tile_store_rule(const rocke_attn_bwd_paged_kv_tile_t* t,
                                             rocke_value_t* k,
                                             rocke_value_t** out_store,
                                             rocke_value_t** out_row_live)
{
    rocke_ir_builder_t* b = t->b;
    rocke_value_t* store = rocke_b_cmp_lt(b, k, t->cover);
    if(t->guarded)
    {
        rocke_value_t* lr = rocke_b_sub(b, k, t->k0);
        rocke_value_t* pid = NULL;
        rocke_value_t* slot = NULL;
        rocke_attn_bwd_paged_kv_tile_locate(t, lr, &pid, &slot);
        rocke_value_t* z = rocke_b_const_i32(b, 0);
        rocke_value_t* ge = rocke_b_cmp_ge(b, pid, z);
        store = rocke_b_land(b, store, ge);
    }
    *out_store = store;
    *out_row_live = rocke_b_cmp_lt(b, k, t->seq.length);
}

/* py: slab_at(pid, slot) of workspace_row */
static bool ws_slab_at(const rocke_attn_bwd_paged_ws_row_t* w,
                       rocke_value_t* pid,
                       rocke_value_t* slot,
                       rocke_attn_bwd_ws_slab_t* out)
{
    rocke_ir_builder_t* b = w->tile->b;
    rocke_value_t* id = NULL;
    rocke_value_t* ok = NULL;
    rocke_attn_bwd_paged_kv_tile_checked(w->tile, pid, &id, &ok);
    /* phys = b.add(b.mul(b.zext(pid, I64), page64), b.zext(slot, I64)) */
    rocke_value_t* pz = rocke_b_zext(b, id, rocke_i64());
    rocke_value_t* pm = rocke_b_mul(b, pz, w->page64);
    rocke_value_t* sz = rocke_b_zext(b, slot, rocke_i64());
    rocke_value_t* phys = rocke_b_add(b, pm, sz);
    if(ok != NULL)
    {
        rocke_value_t* pr = rocke_attn_bwd_as_i64(b, w->page_rows, false);
        phys = rocke_b_select(b, ok, phys, pr);
    }
    rocke_attn_bwd_workspace_addr_t wa;
    if(!rocke_attn_bwd_workspace_addr_init(
           &wa, b, w->ws_ptr, w->page_rows, w->width, phys, ROCKE_ATTN_BWD_WS_HEAD_MAJOR, NULL))
        return false;
    *out = rocke_attn_bwd_workspace_addr_slab(&wa, w->head, rocke_attn_bwd_iv_int(0));
    return rocke_ir_builder_ok(b);
}

bool rocke_attn_bwd_paged_kv_tile_workspace_row(const rocke_attn_bwd_paged_kv_tile_t* t,
                                                rocke_value_t* ws_ptr,
                                                rocke_attn_bwd_iv_t page_rows,
                                                int width,
                                                rocke_attn_bwd_iv_t head,
                                                rocke_attn_bwd_paged_ws_row_t* out)
{
    rocke_ir_builder_t* b = t->b;
    memset(out, 0, sizeof(*out));
    out->tile = t;
    out->ws_ptr = ws_ptr;
    out->page_rows = page_rows;
    out->width = width;
    out->head = head;
    out->page64 = rocke_attn_bwd_as_i64(b, rocke_attn_bwd_iv_val(t->page), false);
    if(!t->has_sub_page)
        return ws_slab_at(out, t->page_ids[0], t->in0, &out->tile_slab);
    out->zero = rocke_b_const_i32(b, 0);
    return rocke_ir_builder_ok(b);
}

bool rocke_attn_bwd_paged_ws_row_slab(const rocke_attn_bwd_paged_ws_row_t* w,
                                      rocke_value_t* lr,
                                      rocke_attn_bwd_ws_slab_t* out_slab,
                                      rocke_value_t** out_row)
{
    if(!w->tile->has_sub_page)
    {
        *out_slab = w->tile_slab;
        *out_row = lr;
        return true;
    }
    rocke_value_t* pid = NULL;
    rocke_value_t* slot = NULL;
    rocke_attn_bwd_paged_kv_tile_locate(w->tile, lr, &pid, &slot);
    if(!ws_slab_at(w, pid, slot, out_slab))
        return false;
    *out_row = w->zero;
    return true;
}

/* ====================================================================== *
 * PagedSlab
 * ====================================================================== */

/* py: _page_ptr */
static rocke_value_t* page_ptr(const rocke_attn_bwd_paged_slab_t* s, rocke_value_t* pid)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* z = rocke_b_zext(b, pid, rocke_i64());
    rocke_value_t* off = rocke_attn_bwd_scaled64(b, z, s->s_blk);
    rocke_value_t* es = rocke_b_const_i64(b, s->esize);
    rocke_value_t* bytes = rocke_b_mul(b, off, es);
    return rocke_b_global_ptr_add(b, s->base, bytes);
}

bool rocke_attn_bwd_paged_slab_init(rocke_attn_bwd_paged_slab_t* s,
                                    rocke_ir_builder_t* b,
                                    const rocke_attn_bwd_paged_kv_tile_t* tile,
                                    rocke_value_t* ptr,
                                    const rocke_type_t* dtype,
                                    const rocke_attn_bwd_iv_t strides[3],
                                    rocke_attn_bwd_iv_t head,
                                    rocke_value_t* last,
                                    int stage_vec)
{
    memset(s, 0, sizeof(*s));
    if(stage_vec != 8 && stage_vec != 1)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "stage_vec must be 8 or 1 (got %d)", stage_vec);
        return false;
    }
    s->b = b;
    s->tile = tile;
    s->dtype = dtype;
    s->last = last;
    s->stage_vec = stage_vec;
    s->esize = rocke_attn_bwd_elem_bytes(b, dtype);
    if(!rocke_ir_builder_ok(b))
        return false;
    s->s_blk = strides[0];
    s->s_tok = strides[1];
    const rocke_attn_bwd_iv_t s_h = strides[2];
    rocke_value_t* head_off = NULL;
    if(!rocke_attn_bwd_iv_is_int(head, 0))
    {
        rocke_value_t* h64 = rocke_attn_bwd_as_i64(b, head, false);
        head_off = rocke_attn_bwd_scaled64(b, h64, s_h);
    }
    /* Head rebase folded into the base once; the page rebase per piece. */
    s->base = ptr;
    if(head_off != NULL)
    {
        rocke_value_t* es = rocke_b_const_i64(b, s->esize);
        rocke_value_t* bytes = rocke_b_mul(b, head_off, es);
        s->base = rocke_b_global_ptr_add(b, ptr, bytes);
    }
    s->tile_ok = NULL;
    if(!tile->has_sub_page)
    {
        rocke_value_t* pid = NULL;
        rocke_attn_bwd_paged_kv_tile_checked(tile, tile->page_ids[0], &pid, &s->tile_ok);
        s->tile_ptr = page_ptr(s, pid);
    }
    else
    {
        s->tile_ptr = NULL;
    }
    return rocke_ir_builder_ok(b);
}

/* py: _address(row, col) -> (page pointer, in-page index, page check or None) */
static void address(const rocke_attn_bwd_paged_slab_t* s,
                    rocke_value_t* row,
                    rocke_attn_bwd_iv_t col,
                    rocke_value_t** out_ptr,
                    rocke_value_t** out_idx,
                    rocke_value_t** out_ok)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* r = s->last == NULL ? row : rocke_b_smin(b, row, s->last);
    rocke_value_t* lr = rocke_b_sub(b, r, s->tile->k0);
    rocke_value_t* pid = NULL;
    rocke_value_t* slot = NULL;
    rocke_attn_bwd_paged_kv_tile_locate(s->tile, lr, &pid, &slot);
    rocke_value_t* ptr;
    rocke_value_t* ok;
    if(s->tile_ptr != NULL)
    {
        ptr = s->tile_ptr;
        ok = s->tile_ok;
    }
    else
    {
        rocke_value_t* id = NULL;
        rocke_attn_bwd_paged_kv_tile_checked(s->tile, pid, &id, &ok);
        ptr = page_ptr(s, id);
    }
    rocke_value_t* idx = rocke_attn_bwd_scaled32(b, slot, s->s_tok);
    if(!rocke_attn_bwd_iv_is_int(col, 0))
    {
        rocke_value_t* c = rocke_attn_bwd_as_i32(b, col);
        idx = rocke_b_add(b, idx, c);
    }
    *out_ptr = ptr;
    *out_idx = idx;
    *out_ok = ok;
}

rocke_value_t* rocke_attn_bwd_paged_slab_load_vec(const rocke_attn_bwd_paged_slab_t* s,
                                                  rocke_value_t* row,
                                                  rocke_attn_bwd_iv_t col0,
                                                  int n)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* ptr = NULL;
    rocke_value_t* idx = NULL;
    rocke_value_t* ok = NULL;
    address(s, row, col0, &ptr, &idx, &ok);
    if(n < 1)
    {
        /* py fails too (parts[0] of an empty list / an empty vec_pack) */
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "load_vec: n=%d must be positive", n);
        return NULL;
    }
    rocke_value_t* out;
    if(s->stage_vec == 8 && n % 8 == 0 && s->esize == 2)
    {
        std::vector<rocke_value_t*> parts((size_t)(n / 8));
        for(int i = 0; i < n / 8; ++i)
        {
            rocke_value_t* at = idx_plus(b, idx, 8 * i);
            parts[(size_t)i] = rocke_b_global_load_vN(b, ptr, at, s->dtype, 8, 16);
        }
        out = parts[0];
        for(size_t i = 1; i < parts.size(); ++i)
            out = rocke_b_vec_concat(b, out, parts[i]);
    }
    else
    {
        std::vector<rocke_value_t*> elems((size_t)n);
        for(int i = 0; i < n; ++i)
        {
            rocke_value_t* at = idx_plus(b, idx, i);
            elems[(size_t)i] = rocke_b_global_load(b, ptr, at, s->dtype, s->esize);
        }
        out = rocke_b_vec_pack(b, elems.data(), n, s->dtype);
    }
    if(ok != NULL)
    {
        rocke_value_t* z = rocke_b_zero_vec(b, s->dtype, n);
        out = rocke_b_select(b, ok, out, z);
    }
    return out;
}

void rocke_attn_bwd_paged_slab_store(const rocke_attn_bwd_paged_slab_t* s,
                                     rocke_value_t* row,
                                     rocke_attn_bwd_iv_t col,
                                     rocke_value_t* value)
{
    rocke_value_t* ptr = NULL;
    rocke_value_t* idx = NULL;
    rocke_value_t* ok = NULL;
    address(s, row, col, &ptr, &idx, &ok);
    rocke_b_global_store(s->b, ptr, idx, value, s->esize);
}

void rocke_attn_bwd_paged_slab_store_vec(const rocke_attn_bwd_paged_slab_t* s,
                                         rocke_value_t* row,
                                         rocke_attn_bwd_iv_t col0,
                                         rocke_value_t* value)
{
    rocke_ir_builder_t* b = s->b;
    if(!rocke_ir_builder_ok(b) || value == NULL)
        return;
    const int n = value->type->count;
    rocke_value_t* ptr = NULL;
    rocke_value_t* idx = NULL;
    rocke_value_t* ok = NULL;
    address(s, row, col0, &ptr, &idx, &ok);
    if(s->stage_vec == 8 && (n == 8 || n == 4 || n == 2) && s->esize == 2)
    {
        rocke_b_global_store_vN(b, ptr, idx, value, n, 2 * n);
        return;
    }
    for(int i = 0; i < n; ++i)
    {
        rocke_value_t* at = idx_plus(b, idx, i);
        rocke_value_t* e = rocke_b_vec_extract(b, value, i);
        rocke_b_global_store(b, ptr, at, e, s->esize);
    }
}

/* ====================================================================== *
 * PagedKv / PagedCacheRows
 * ====================================================================== */

bool rocke_attn_bwd_paged_kv_init(rocke_attn_bwd_paged_kv_t* p,
                                  rocke_ir_builder_t* b,
                                  rocke_value_t* bt_ptr,
                                  rocke_attn_bwd_iv_t bt_stride,
                                  rocke_value_t* seq_idx,
                                  const rocke_attn_bwd_seq_info_t* seq,
                                  int k_n0,
                                  rocke_attn_bwd_iv_t page,
                                  const int* sub_page,
                                  const rocke_attn_bwd_iv_t* page_rows)
{
    memset(p, 0, sizeof(*p));
    p->b = b;
    p->bt_ptr = bt_ptr;
    p->bt_stride = bt_stride;
    p->seq_idx = seq_idx;
    p->seq = *seq;
    p->k_n0 = k_n0;
    p->page = page;
    p->has_sub_page = sub_page != NULL;
    p->sub_page = sub_page != NULL ? *sub_page : 0;
    p->num_blocks = NULL;
    if(page_rows != NULL)
    {
        rocke_value_t* pr = rocke_attn_bwd_as_i32(b, *page_rows);
        rocke_value_t* pg = rocke_attn_bwd_as_i32(b, page);
        p->num_blocks = rocke_b_div(b, pr, pg);
    }
    p->reads = 0;
    return rocke_ir_builder_ok(b);
}

void rocke_attn_bwd_paged_kv_free(rocke_attn_bwd_paged_kv_t* p)
{
    for(int i = 0; i < p->n_tiles; ++i)
    {
        rocke_attn_bwd_paged_kv_tile_free(p->tiles[i].tile);
        free(p->tiles[i].tile);
    }
    free(p->tiles);
    p->tiles = NULL;
    p->n_tiles = 0;
    p->cap_tiles = 0;
}

bool rocke_attn_bwd_paged_kv_guarded(const rocke_attn_bwd_paged_kv_t* p)
{
    return p->num_blocks != NULL;
}

const rocke_attn_bwd_paged_kv_tile_t* rocke_attn_bwd_paged_kv_tile(rocke_attn_bwd_paged_kv_t* p,
                                                                   rocke_value_t* k0)
{
    for(int i = 0; i < p->n_tiles; ++i)
        if(p->tiles[i].k0 == k0)
            return p->tiles[i].tile;
    rocke_attn_bwd_paged_kv_tile_t* t
        = (rocke_attn_bwd_paged_kv_tile_t*)malloc(sizeof(rocke_attn_bwd_paged_kv_tile_t));
    if(t == NULL)
    {
        rocke_i_set_err(p->b, ROCKE_ERR_OOM, "paged kv: out of memory");
        return NULL;
    }
    if(p->n_tiles == p->cap_tiles)
    {
        const int cap = p->cap_tiles ? 2 * p->cap_tiles : 4;
        rocke_attn_bwd_paged_kv_cache_entry_t* e
            = (rocke_attn_bwd_paged_kv_cache_entry_t*)realloc(p->tiles, sizeof(*e) * (size_t)cap);
        if(e == NULL)
        {
            free(t);
            rocke_i_set_err(p->b, ROCKE_ERR_OOM, "paged kv: out of memory");
            return NULL;
        }
        p->tiles = e;
        p->cap_tiles = cap;
    }
    const int sp = p->sub_page;
    if(!rocke_attn_bwd_paged_kv_tile_init(t,
                                          p->b,
                                          p->bt_ptr,
                                          p->bt_stride,
                                          p->seq_idx,
                                          &p->seq,
                                          k0,
                                          p->k_n0,
                                          p->page,
                                          p->has_sub_page ? &sp : NULL,
                                          p->num_blocks,
                                          &p->reads))
    {
        rocke_attn_bwd_paged_kv_tile_free(t);
        free(t);
        return NULL;
    }
    p->tiles[p->n_tiles].k0 = k0;
    p->tiles[p->n_tiles].tile = t;
    ++p->n_tiles;
    return t;
}

rocke_attn_bwd_paged_cache_rows_t
    rocke_attn_bwd_paged_kv_cache(rocke_attn_bwd_paged_kv_t* p,
                                  rocke_value_t* ptr,
                                  const rocke_type_t* dtype,
                                  const rocke_attn_bwd_iv_t strides[3],
                                  int stage_vec)
{
    rocke_attn_bwd_paged_cache_rows_t c;
    memset(&c, 0, sizeof(c));
    c.paged = p;
    c.ptr = ptr;
    c.dtype = dtype;
    c.strides[0] = strides[0];
    c.strides[1] = strides[1];
    c.strides[2] = strides[2];
    c.stage_vec = stage_vec;
    return c;
}

bool rocke_attn_bwd_paged_cache_rows_slab(const rocke_attn_bwd_paged_cache_rows_t* c,
                                          rocke_attn_bwd_iv_t h,
                                          rocke_value_t* k0,
                                          bool clamp,
                                          rocke_attn_bwd_paged_slab_t* out)
{
    const rocke_attn_bwd_paged_kv_tile_t* tile = rocke_attn_bwd_paged_kv_tile(c->paged, k0);
    if(tile == NULL)
        return false;
    return rocke_attn_bwd_paged_slab_init(out,
                                          c->paged->b,
                                          tile,
                                          c->ptr,
                                          c->dtype,
                                          c->strides,
                                          h,
                                          clamp ? c->paged->seq.last : NULL,
                                          c->stage_vec);
}
