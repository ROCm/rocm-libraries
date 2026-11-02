// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of the work-list emitters of
 * library/kernels/common/_attention_bwd_worklist.py (the prep kernel's list
 * writers and the main kernel's CTA decode). See
 * rocke/helper_rocke.instances.common._attention_bwd_worklist.h.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_worklist.h"

#include <stddef.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

rocke_value_t* rocke_attn_bwd_worklist_tiles(rocke_ir_builder_t* b,
                                             rocke_value_t* len_kv,
                                             rocke_value_t* block_n)
{
    /* py: b.div(b.add(len_kv, b.sub(block_n, b.const_i32(1))), block_n) */
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* bn1 = rocke_b_sub(b, block_n, c1);
    rocke_value_t* sum = rocke_b_add(b, len_kv, bn1);
    return rocke_b_div(b, sum, block_n);
}

rocke_value_t* rocke_attn_bwd_worklist_sorts_before(rocke_ir_builder_t* b,
                                                    rocke_value_t* key_o,
                                                    rocke_value_t* o,
                                                    rocke_value_t* key_s,
                                                    rocke_value_t* s)
{
    rocke_value_t* longer = rocke_b_cmp_gt(b, key_o, key_s);
    rocke_value_t* eq = rocke_b_cmp_eq(b, key_o, key_s);
    rocke_value_t* lt = rocke_b_cmp_lt(b, o, s);
    rocke_value_t* tie = rocke_b_land(b, eq, lt);
    return rocke_b_lor(b, longer, tie);
}

void rocke_attn_bwd_worklist_store_item(rocke_ir_builder_t* b,
                                        rocke_value_t* wl,
                                        rocke_value_t* pos,
                                        rocke_value_t* seq,
                                        rocke_value_t* tile)
{
    /* py: w = b.add(b.mul(pos, b.const_i32(2)), b.const_i32(WL_HEADER_WORDS)) */
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* p2 = rocke_b_mul(b, pos, c2);
    rocke_value_t* hdr = rocke_b_const_i32(b, ROCKE_ATTN_BWD_WL_HEADER_WORDS);
    rocke_value_t* w = rocke_b_add(b, p2, hdr);
    rocke_b_global_store(b, wl, w, seq, 4);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* w1 = rocke_b_add(b, w, c1);
    rocke_b_global_store(b, wl, w1, tile, 4);
}

void rocke_attn_bwd_worklist_store_count(rocke_ir_builder_t* b,
                                         rocke_value_t* wl,
                                         rocke_value_t* count)
{
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_b_global_store(b, wl, c0, count, 4);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* z = rocke_b_const_i32(b, 0);
    rocke_b_global_store(b, wl, c1, z, 4);
}

/* ====================================================================== *
 * Main-kernel unit: the CTA decode and the head range
 * ====================================================================== */

void rocke_attn_bwd_worklist_item(rocke_ir_builder_t* b,
                                  const rocke_attn_bwd_work_list_t* wl,
                                  rocke_value_t* idx,
                                  rocke_value_t* n_kv_tiles,
                                  rocke_value_t* grid_batch,
                                  rocke_value_t* grid_kt,
                                  rocke_value_t** out_seq,
                                  rocke_value_t** out_kt)
{
    *out_seq = NULL;
    *out_kt = NULL;
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* on = rocke_b_cmp_ne(b, wl->use, zero);
    rocke_value_t* trip = rocke_b_select(b, on, one, zero);
    rocke_iter_arg_t args[2] = {{"wl_seq", grid_batch}, {"wl_kt", grid_kt}};
    rocke_for_t loop = rocke_b_scf_for_iter(b, zero, trip, one, args, 2, "wl_i", false, true);
    if(loop.op == NULL)
        return;
    rocke_b_region_enter(b, loop.body);
    rocke_value_t* count = rocke_b_global_load_i32(b, wl->ptr, zero, 4);
    rocke_value_t* valid = rocke_b_cmp_lt(b, idx, count);
    rocke_value_t* sel = rocke_b_select(b, valid, idx, zero);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* m = rocke_b_mul(b, sel, c2);
    rocke_value_t* hdr = rocke_b_const_i32(b, ROCKE_ATTN_BWD_WL_HEADER_WORDS);
    rocke_value_t* w = rocke_b_add(b, m, hdr);
    rocke_value_t* seq = rocke_b_global_load_i32(b, wl->ptr, w, 4);
    rocke_value_t* w1 = rocke_b_add(b, w, one);
    rocke_value_t* tile = rocke_b_global_load_i32(b, wl->ptr, w1, 4);
    rocke_value_t* ys[2];
    ys[0] = rocke_b_select(b, valid, seq, zero);
    ys[1] = rocke_b_select(b, valid, tile, n_kv_tiles);
    rocke_b_scf_yield(b, ys, 2);
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b) || loop.op->num_results < 2)
        return;
    *out_seq = loop.op->results[0];
    *out_kt = loop.op->results[1];
}

rocke_value_t* rocke_attn_bwd_cta_work_kv_tile(const rocke_attn_bwd_cta_work_t* w,
                                               rocke_ir_builder_t* b,
                                               rocke_value_t* p)
{
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* eq = rocke_b_cmp_eq(b, p, c0);
    return rocke_b_select(b, eq, w->first_kt, w->second_kt);
}

rocke_value_t* rocke_attn_bwd_xcd_remap_id(rocke_ir_builder_t* b,
                                           rocke_value_t* wgid,
                                           rocke_value_t* num_wgs,
                                           rocke_value_t* n,
                                           rocke_value_t* chunk)
{
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* n_pos = rocke_b_cmp_gt(b, n, zero);
    rocke_value_t* c_pos = rocke_b_cmp_gt(b, chunk, zero);
    rocke_value_t* on = rocke_b_land(b, n_pos, c_pos);
    rocke_value_t* n1 = rocke_b_select(b, on, n, one);
    rocke_value_t* c1 = rocke_b_select(b, on, chunk, one);
    rocke_value_t* block = rocke_b_mul(b, n1, c1);
    rocke_value_t* nb = rocke_b_div(b, num_wgs, block);
    rocke_value_t* limit = rocke_b_mul(b, nb, block);
    rocke_value_t* xcd = rocke_b_mod(b, wgid, n1);
    rocke_value_t* local = rocke_b_div(b, wgid, n1);
    rocke_value_t* lc = rocke_b_div(b, local, c1);
    rocke_value_t* lcb = rocke_b_mul(b, lc, block);
    rocke_value_t* xc = rocke_b_mul(b, xcd, c1);
    rocke_value_t* s = rocke_b_add(b, lcb, xc);
    rocke_value_t* lm = rocke_b_mod(b, local, c1);
    rocke_value_t* nw = rocke_b_add(b, s, lm);
    rocke_value_t* below = rocke_b_cmp_lt(b, wgid, limit);
    rocke_value_t* use = rocke_b_land(b, on, below);
    return rocke_b_select(b, use, nw, wgid);
}

bool rocke_attn_bwd_decode_cta(rocke_ir_builder_t* b,
                               rocke_value_t* x,
                               rocke_value_t* y,
                               rocke_value_t* z,
                               rocke_value_t* n_kv_tiles,
                               rocke_value_t* lb_order,
                               rocke_value_t* pair,
                               rocke_value_t* g_split,
                               bool interleave,
                               const rocke_attn_bwd_xcd_remap_t* xcd,
                               const rocke_attn_bwd_work_list_t* worklist,
                               rocke_attn_bwd_cta_work_t* out)
{
    out->t = out->unit = out->split = out->batch = NULL;
    out->first_kt = out->second_kt = out->n_pass = NULL;
    if(worklist != NULL && pair != NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "the work list decodes one kv tile per CTA (pair = None)");
        return false;
    }
    if(xcd != NULL)
    {
        rocke_value_t* gx = n_kv_tiles;
        if(pair != NULL)
        {
            rocke_value_t* two_ = rocke_b_const_i32(b, 2);
            rocke_value_t* c1 = rocke_b_const_i32(b, 1);
            rocke_value_t* nk1 = rocke_b_add(b, n_kv_tiles, c1);
            rocke_value_t* half = rocke_b_div(b, nk1, two_);
            rocke_value_t* c0 = rocke_b_const_i32(b, 0);
            rocke_value_t* pon = rocke_b_cmp_ne(b, pair, c0);
            gx = rocke_b_select(b, pon, half, n_kv_tiles);
        }
        rocke_value_t* c0 = rocke_b_const_i32(b, 0);
        rocke_value_t* gpos = rocke_b_cmp_gt(b, gx, c0);
        rocke_value_t* c1 = rocke_b_const_i32(b, 1);
        rocke_value_t* gx1 = rocke_b_select(b, gpos, gx, c1);
        rocke_value_t* gy_ = rocke_b_mul(b, gx1, y);
        rocke_value_t* wgid = rocke_b_add(b, x, gy_);
        rocke_value_t* num_wgs = rocke_b_mul(b, gx1, xcd->gy);
        rocke_value_t* plane = rocke_attn_bwd_xcd_remap_id(b, wgid, num_wgs, xcd->n, xcd->chunk);
        rocke_value_t* nx = rocke_b_mod(b, plane, gx1);
        rocke_value_t* ny = rocke_b_div(b, plane, gx1);
        x = nx;
        y = ny;
    }
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t *unit, *split;
    if(g_split == NULL)
    {
        unit = y;
        split = zero;
    }
    else
    {
        unit = rocke_b_div(b, y, g_split);
        split = rocke_b_mod(b, y, g_split);
    }
    rocke_value_t* nk_1 = rocke_b_sub(b, n_kv_tiles, one);
    rocke_value_t* mirror_t = rocke_b_sub(b, nk_1, x);
    rocke_value_t* crev = rocke_b_const_i32(b, ROCKE_ATTN_BWD_LB_REVERSE);
    rocke_value_t* reverse = rocke_b_cmp_eq(b, lb_order, crev);
    rocke_value_t* x_or_woven = x;
    rocke_value_t* woven = NULL;
    rocke_value_t* inter = NULL;
    if(interleave)
    {
        rocke_value_t* c2 = rocke_b_const_i32(b, 2);
        rocke_value_t* half = rocke_b_div(b, x, c2);
        rocke_value_t* xm = rocke_b_mod(b, x, c2);
        rocke_value_t* odd = rocke_b_cmp_ne(b, xm, zero);
        rocke_value_t* n1 = rocke_b_sub(b, n_kv_tiles, one);
        rocke_value_t* hi = rocke_b_sub(b, n1, half);
        woven = rocke_b_select(b, odd, hi, half);
        rocke_value_t* cint = rocke_b_const_i32(b, ROCKE_ATTN_BWD_LB_INTERLEAVE);
        inter = rocke_b_cmp_eq(b, lb_order, cint);
        x_or_woven = rocke_b_select(b, inter, woven, x);
    }
    if(pair == NULL)
    {
        rocke_value_t* first = rocke_b_select(b, reverse, mirror_t, x_or_woven);
        rocke_value_t* batch = z;
        if(worklist != NULL)
        {
            rocke_value_t* nz = rocke_b_mul(b, n_kv_tiles, z);
            rocke_value_t* item = rocke_b_add(b, x, nz);
            rocke_value_t *seq, *kt;
            rocke_attn_bwd_worklist_item(b, worklist, item, n_kv_tiles, z, first, &seq, &kt);
            batch = seq;
            first = kt;
        }
        out->t = x;
        out->unit = unit;
        out->split = split;
        out->batch = batch;
        out->first_kt = first;
        out->second_kt = first;
        out->n_pass = one;
        return rocke_ir_builder_ok(b);
    }
    rocke_value_t* paired = rocke_b_cmp_ne(b, pair, zero);
    rocke_value_t* np = rocke_b_lnot(b, paired);
    rocke_value_t* rev_np = rocke_b_land(b, reverse, np);
    rocke_value_t* first = rocke_b_select(b, rev_np, mirror_t, x);
    if(interleave)
    {
        rocke_value_t* np2 = rocke_b_lnot(b, paired);
        rocke_value_t* int_np = rocke_b_land(b, inter, np2);
        first = rocke_b_select(b, int_np, woven, first);
    }
    rocke_value_t* differ = rocke_b_cmp_ne(b, mirror_t, first);
    rocke_value_t* two = rocke_b_land(b, paired, differ);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* n_pass = rocke_b_select(b, two, c2, one);
    out->t = x;
    out->unit = unit;
    out->split = split;
    out->batch = z;
    out->first_kt = first;
    out->second_kt = mirror_t;
    out->n_pass = n_pass;
    return rocke_ir_builder_ok(b);
}

rocke_attn_bwd_head_range_t rocke_attn_bwd_head_range(rocke_ir_builder_t* b,
                                                      rocke_value_t* unit,
                                                      rocke_value_t* split,
                                                      rocke_value_t* group,
                                                      rocke_value_t* gk,
                                                      rocke_value_t* gv,
                                                      rocke_value_t* g_split)
{
    rocke_attn_bwd_head_range_t r;
    rocke_value_t* base = rocke_b_mul(b, unit, group);
    if(g_split == NULL)
    {
        r.h0 = base;
        r.n_heads = group;
        r.hk = unit;
        r.hv = unit;
        return r;
    }
    rocke_value_t* per = rocke_b_div(b, group, g_split);
    rocke_value_t* sp = rocke_b_mul(b, split, per);
    r.h0 = rocke_b_add(b, base, sp);
    r.n_heads = per;
    r.hk = rocke_b_div(b, base, gk);
    r.hv = rocke_b_div(b, base, gv);
    return r;
}
