// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/* gemm_build_staged.cpp -- C++ mirror of gemm_universal.py `_emit_kloop_staged`:
 * the alias-scoped DTL LDS ring (trait.dtl_stages >= 2), its
 * in-stream barrier / fragment prefetch / DTL issue-spread / iglp_opt levers
 * and the persistent overlapped-epilogue tile loop (trait.persistent_ctas).
 *
 * Every IR op is emitted in the Python order (the byte-identity contract).
 * Python closures that defer emission (the per-instruction DTL issue closures
 * `_dtl_issues` returns) are mirrored by an issue-set record plus
 * staged_issue_one(), called at the same points the Python calls the closures.
 * Nested builder calls with more than one emitting argument are split into
 * explicit temporaries so the C++ evaluation order matches Python's
 * left-to-right order. */

#include <stdio.h> /* snprintf */
#include <string.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.grid.h"
#include "rocke/helper_rocke.helpers.streamk.h"
#include "rocke/instance_gemm_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#define STG_MAX_STAGES 4
#define STG_MAX_FRAGS 64
#define STG_MAX_ISSUES 64

typedef struct staged_state
{
    rocke_gemm_build_ctx_t* ctx;
    rocke_ir_builder_t* b;
    int S; /* stages                                   */
    int L; /* DTL loads per tile per wave              */
    int keep; /* (S-2)*L                                  */
    bool fprf; /* dtl_frag_prefetch                        */
    bool instream; /* fprf || dtl_stream_barrier               */
    bool spread; /* dtl_issue_spread                         */
    bool ilv; /* mfma_interleave                          */
    bool iglp_set;
    int iglp;
    int pers; /* persistent_ctas                          */
    rocke_value_t* k_up;
    rocke_value_t* c2s;
    rocke_value_t* wpo; /* per-wave LDS byte offset (i64) or NULL   */
    rocke_value_t* a_wbase[STG_MAX_STAGES];
    rocke_value_t* b_wbase[STG_MAX_STAGES];
    rocke_value_t* a_voff[ROCKE_GEMM_MAX_DTL_PASSES];
    rocke_value_t* b_voff[ROCKE_GEMM_MAX_DTL_PASSES];
    rocke_value_t* a_tile0;
    rocke_value_t* b_tile0;
    rocke_value_t* c_big;
    rocke_value_t* c_zero_rec;
    /* fragment addressing */
    rocke_value_t* m_in_atom;
    rocke_value_t* k_blk;
    rocke_value_t* n_in_atom;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    rocke_value_t* k_blk_kbase;
    rocke_value_t* a_row[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* b_row[ROCKE_GEMM_MAX_ACCS];
    bool lane_key;
    int offs_a[STG_MAX_STAGES];
    int offs_b[STG_MAX_STAGES];
    int row_bytes;
    /* opaque rows keyed by (operand, shift): at most S distinct shifts each */
    int opq_shift_a[STG_MAX_STAGES];
    rocke_value_t* opq_a[STG_MAX_STAGES];
    int n_opq_a;
    int opq_shift_b[STG_MAX_STAGES];
    rocke_value_t* opq_b[STG_MAX_STAGES];
    int n_opq_b;
    int n_mfma_tile;
    /* driver */
    rocke_value_t* T;
    rocke_value_t* n_main;
    int n_pro;
    /* dtl_tight_rsrc operand bytes; the K bound of the stream-K item
     * being emitted (NULL = k_up), Python's `kup` argument */
    bool tight;
    rocke_value_t* a_bytes;
    rocke_value_t* b_bytes;
    rocke_value_t* cur_kup;
    /* DTL bytes of one pass (rilv: + rilv_pad per wave) */
    int pass_bytes;
    /* wide_addr: 64-bit per-operand base at the output tile's origin (_fold) */
    bool wide;
    rocke_value_t* fold[2];
    /* lds_row_interleave fragment rows (_rbase): per operand, keyed by shift */
    int rb; /* LDS alloc row bytes */
    int roffs_a[STG_MAX_STAGES];
    int roffs_b[STG_MAX_STAGES];
    int rbase_shift[2][STG_MAX_STAGES];
    rocke_value_t* rbase[2][STG_MAX_STAGES];
    int n_rbase[2];
} staged_state_t;

/* The deferred issue closures of one `_dtl_issues(stage, k_ld, tile0s, valid)`
 * call: index 0.._dtl_a_passes-1 are the A loads, then the B loads. */
typedef struct staged_issue_set
{
    int stage;
    rocke_value_t* k_ld;
    rocke_value_t* ok;
    rocke_value_t* nrec;
    rocke_value_t* ta0;
    rocke_value_t* tb0;
    int count;
    /* dtl_hoist: per-operand resource / soffset built with the set */
    rocke_value_t* h_rsrc[2];
    rocke_value_t* h_soff[2];
} staged_issue_set_t;

/* _tight_rs(ptr, tile0, k_ld, ok, nbytes): a resource based at the tile origin
 * whose range ends at the operand's last byte, soffset 0 (dtl_tight_rsrc). */
static void staged_tight_rs(staged_state_t* st,
                            int o,
                            rocke_value_t* ptr,
                            rocke_value_t* tile0,
                            rocke_value_t* k_ld,
                            rocke_value_t* ok,
                            rocke_value_t* nbytes,
                            rocke_value_t** rs,
                            rocke_value_t** soff)
{
    rocke_ir_builder_t* b = st->b;
    if(st->wide)
    {
        /* wide_addr: the tile origin is in the 64-bit base, `nbytes` the bytes
         * left from it */
        rocke_value_t* by = rocke_b_mul(b, k_ld, st->c2s);
        rocke_value_t* org = rocke_b_to_sgpr_u32(b, by);
        rocke_value_t* df = rocke_b_sub(b, nbytes, org);
        rocke_value_t* left = rocke_b_smax(b, df, st->c_zero_rec);
        rocke_value_t* gp = rocke_b_global_ptr_add(b, st->fold[o], org);
        rocke_value_t* sel = rocke_b_select(b, ok, left, st->c_zero_rec);
        *rs = rocke_b_buffer_rsrc(b, gp, sel);
        *soff = st->c_zero_rec;
        return;
    }
    rocke_value_t* sum = rocke_b_add(b, tile0, k_ld);
    rocke_value_t* by = rocke_b_mul(b, sum, st->c2s);
    rocke_value_t* org = rocke_b_to_sgpr_u32(b, by);
    rocke_value_t* df = rocke_b_sub(b, nbytes, org);
    rocke_value_t* left = rocke_b_smax(b, df, st->c_zero_rec);
    rocke_value_t* gp = rocke_b_global_ptr_add(b, ptr, org);
    rocke_value_t* sel = rocke_b_select(b, ok, left, st->c_zero_rec);
    *rs = rocke_b_buffer_rsrc(b, gp, sel);
    *soff = st->c_zero_rec;
}

static staged_issue_set_t staged_dtl_issues(staged_state_t* st,
                                            int stage,
                                            rocke_value_t* k_ld,
                                            rocke_value_t* const* tile0s,
                                            rocke_value_t* valid)
{
    rocke_ir_builder_t* b = st->b;
    staged_issue_set_t s;
    s.stage = stage;
    s.k_ld = k_ld;
    s.ta0 = tile0s != NULL ? tile0s[0] : st->a_tile0;
    s.tb0 = tile0s != NULL ? tile0s[1] : st->b_tile0;
    s.ok = rocke_b_cmp_lt(b, k_ld, st->cur_kup != NULL ? st->cur_kup : st->k_up);
    if(valid != NULL)
        s.ok = rocke_b_land(b, s.ok, valid);
    s.nrec = rocke_b_select(b, s.ok, st->c_big, st->c_zero_rec);
    s.count = st->ctx->dtl_a_passes + st->ctx->dtl_b_passes;
    s.h_rsrc[0] = s.h_rsrc[1] = NULL;
    s.h_soff[0] = s.h_soff[1] = NULL;
    if(st->ctx->spec->trait.dtl_hoist)
    {
        int o;
        for(o = 0; o < 2; ++o)
        {
            rocke_value_t* ptr = o == 0 ? st->ctx->A : st->ctx->Bp;
            rocke_value_t* tile0 = o == 0 ? s.ta0 : s.tb0;
            rocke_value_t* sum;
            rocke_value_t* by;
            rocke_value_t* sel;
            if(st->tight)
            {
                staged_tight_rs(st,
                                o,
                                ptr,
                                tile0,
                                k_ld,
                                s.ok,
                                o == 0 ? st->a_bytes : st->b_bytes,
                                &s.h_rsrc[o],
                                &s.h_soff[o]);
                continue;
            }
            if(st->ctx->spec->trait.dtl_soff0)
            {
                /* origin in the resource base, inline soffset 0 */
                rocke_value_t* org;
                rocke_value_t* gp;
                if(st->fold[o] != NULL && tile0s == NULL)
                {
                    by = rocke_b_mul(b, k_ld, st->c2s);
                    org = rocke_b_to_sgpr_u32(b, by);
                    gp = rocke_b_global_ptr_add(b, st->fold[o], org);
                }
                else
                {
                    sum = rocke_b_add(b, tile0, k_ld);
                    by = rocke_b_mul(b, sum, st->c2s);
                    org = rocke_b_to_sgpr_u32(b, by);
                    gp = rocke_b_global_ptr_add(b, ptr, org);
                }
                s.h_rsrc[o] = rocke_b_buffer_rsrc(b, gp, s.nrec);
                s.h_soff[o] = st->c_zero_rec;
                continue;
            }
            s.h_rsrc[o] = rocke_b_buffer_rsrc(b, ptr, s.nrec);
            sum = rocke_b_add(b, tile0, k_ld);
            by = rocke_b_mul(b, sum, st->c2s);
            sel = rocke_b_select(b, s.ok, by, st->c_zero_rec);
            s.h_soff[o] = rocke_b_to_sgpr_u32(b, sel);
        }
    }
    return s;
}

static void staged_issue_one(staged_state_t* st, const staged_issue_set_t* s, int idx)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    bool is_a = idx < ctx->dtl_a_passes;
    int p = is_a ? idx : idx - ctx->dtl_a_passes;
    rocke_value_t* ptr = is_a ? ctx->A : ctx->Bp;
    rocke_value_t* tile0 = is_a ? s->ta0 : s->tb0;
    rocke_value_t* const* voffs = is_a ? st->a_voff : st->b_voff;
    rocke_value_t* const* wbase = is_a ? st->a_wbase : st->b_wbase;
    int coh = is_a ? ctx->spec->trait.dtl_cache_a : ctx->spec->trait.dtl_cache_b;
    rocke_value_t* rsrc;
    rocke_value_t* soff;
    rocke_value_t* lds;

    if(s->h_rsrc[is_a ? 0 : 1] != NULL)
    {
        rsrc = s->h_rsrc[is_a ? 0 : 1];
        soff = s->h_soff[is_a ? 0 : 1];
    }
    else if(st->tight)
    {
        staged_tight_rs(st,
                        is_a ? 0 : 1,
                        ptr,
                        tile0,
                        s->k_ld,
                        s->ok,
                        is_a ? st->a_bytes : st->b_bytes,
                        &rsrc,
                        &soff);
    }
    else
    {
        rsrc = rocke_b_buffer_rsrc(b, ptr, s->nrec);
        {
            rocke_value_t* sum = rocke_b_add(b, tile0, s->k_ld);
            rocke_value_t* by = rocke_b_mul(b, sum, st->c2s);
            rocke_value_t* sel = rocke_b_select(b, s->ok, by, st->c_zero_rec);
            soff = rocke_b_to_sgpr_u32(b, sel);
        }
    }
    lds = wbase[s->stage];
    if(p)
    {
        rocke_value_t* cst = rocke_b_const_i32(b, p * st->pass_bytes);
        rocke_value_t* z = rocke_b_zext(b, cst, rocke_i64());
        lds = rocke_b_smem_ptr_add(b, lds, z);
    }
    rocke_b_async_buffer_load_lds_addr(b, rsrc, lds, voffs[p], soff, ctx->dtl_dwords, coh);
}

static void staged_issue_all(staged_state_t* st, const staged_issue_set_t* s)
{
    int i;
    for(i = 0; i < s->count; ++i)
        staged_issue_one(st, s, i);
}

/* _spread_pos(issues, first, span): issue j at MFMA index first + (j*span)//n.
 * pos_issue[idx] lists issue indices (in j order) for MFMA index idx. */
typedef struct staged_pos
{
    int n_at[ROCKE_GEMM_MAX_ACCS];
    int at[ROCKE_GEMM_MAX_ACCS][STG_MAX_ISSUES];
    int first;
    int span;
    bool on;
} staged_pos_t;

static void staged_spread_pos(staged_pos_t* pos, int n, int first, int span)
{
    int j;
    memset(pos->n_at, 0, sizeof(pos->n_at));
    pos->first = first;
    pos->span = span;
    pos->on = true;
    for(j = 0; j < n; ++j)
    {
        int idx = first + (j * span) / n - first; /* stored relative to first */
        pos->at[idx][pos->n_at[idx]++] = j;
    }
}

static void staged_stage_bases(staged_state_t* st)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    int j;
    for(j = 0; j < st->S; ++j)
    {
        rocke_value_t* base = rocke_b_smem_addr_of(b, ctx->A_stg[j]);
        st->a_wbase[j] = st->wpo != NULL ? rocke_b_smem_ptr_add(b, base, st->wpo) : base;
    }
    for(j = 0; j < st->S; ++j)
    {
        rocke_value_t* base = rocke_b_smem_addr_of(b, ctx->B_stg[j]);
        st->b_wbase[j] = st->wpo != NULL ? rocke_b_smem_ptr_add(b, base, st->wpo) : base;
    }
}

static void staged_lane_offs(staged_state_t* st, int n_pass, int mf, bool isb, rocke_value_t** offs)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    int p;
    for(p = 0; p < n_pass; ++p)
    {
        rocke_value_t* chunk_idx
            = rocke_b_add(b, ctx->tid, rocke_b_const_i32(b, p * ctx->spec->block_size));
        rocke_value_t* row = rocke_b_div(b, chunk_idx, ctx->dtl_c_chunks_per_row);
        rocke_value_t* cmod = rocke_b_mod(b, chunk_idx, ctx->dtl_c_chunks_per_row);
        rocke_value_t* col = rocke_b_mul(b, cmod, ctx->dtl_c_halves_per_chunk);
        if(ctx->rilv && ctx->spec->trait.rilv_vec_epilogue)
        {
            /* linear rows: the C mapping carries the interleave */
            rocke_value_t* rr = isb ? rocke_gemm_brow(ctx, row) : row;
            rocke_value_t* rk = rocke_b_mul(b, rr, ctx->K);
            rocke_value_t* sum = rocke_b_add(b, rk, col);
            offs[p] = rocke_b_mul(b, sum, st->c2s);
            continue;
        }
        if(ctx->rilv)
        {
            /* LDS row R of a warp block of 16*mf rows holds global row
             * (R % mf) * 16 + R // mf of that block (no column swizzle) */
            rocke_value_t* wblk = rocke_b_const_i32(b, 16 * mf);
            rocke_value_t* r_in = rocke_b_mod(b, row, wblk);
            rocke_value_t* q = rocke_b_div(b, row, wblk);
            rocke_value_t* base = rocke_b_mul(b, q, wblk);
            rocke_value_t* c_mf = rocke_b_const_i32(b, mf);
            rocke_value_t* rm = rocke_b_mod(b, r_in, c_mf);
            rocke_value_t* c16 = rocke_b_const_i32(b, 16);
            rocke_value_t* r16 = rocke_b_mul(b, rm, c16);
            rocke_value_t* c_mf2 = rocke_b_const_i32(b, mf);
            rocke_value_t* rd = rocke_b_div(b, r_in, c_mf2);
            rocke_value_t* in = rocke_b_add(b, r16, rd);
            rocke_value_t* row2 = rocke_b_add(b, base, in);
            rocke_value_t* rk = rocke_b_mul(b, row2, ctx->K);
            rocke_value_t* sum = rocke_b_add(b, rk, col);
            offs[p] = rocke_b_mul(b, sum, st->c2s);
            continue;
        }
        rocke_value_t* rr = isb ? rocke_gemm_brow(ctx, row) : row;
        rocke_value_t* rk = rocke_b_mul(b, rr, ctx->K);
        rocke_value_t* sc = rocke_gemm_swz_col(ctx, col, row);
        rocke_value_t* sum = rocke_b_add(b, rk, sc);
        offs[p] = rocke_b_mul(b, sum, st->c2s);
    }
}

static int staged_shift_rows(const staged_state_t* st, bool is_a, int j)
{
    int off = is_a ? st->offs_a[j] : st->offs_b[j];
    return ((off / 65536) * 65536) / st->row_bytes;
}

static rocke_value_t* staged_opq_get(staged_state_t* st, bool is_a, int sh)
{
    int i;
    int n = is_a ? st->n_opq_a : st->n_opq_b;
    for(i = 0; i < n; ++i)
    {
        if((is_a ? st->opq_shift_a[i] : st->opq_shift_b[i]) == sh)
            return is_a ? st->opq_a[i] : st->opq_b[i];
    }
    return NULL;
}

static rocke_value_t* staged_row_for(staged_state_t* st, bool is_a, int j, int idx)
{
    rocke_ir_builder_t* b = st->b;
    const rocke_gemm_tile_spec_t* t = &st->ctx->spec->tile;
    int sh = staged_shift_rows(st, is_a, j);
    int wt;
    if(sh == 0 || !st->lane_key)
        return is_a ? st->a_row[idx] : st->b_row[idx];
    wt = is_a ? t->warp_tile_m : t->warp_tile_n;
    {
        rocke_value_t* opq = staged_opq_get(st, is_a, sh);
        rocke_value_t* cst = rocke_b_const_i32(b, idx * wt - sh);
        return rocke_b_add(b, opq, cst);
    }
}

/* _rsh(nm, j): the row shift of an interleaved operand's stage j */
static int staged_rsh(const staged_state_t* st, bool is_a, int j)
{
    int off = is_a ? st->roffs_a[j] : st->roffs_b[j];
    return ((off / 65536) * 65536) / st->rb;
}

static rocke_value_t* staged_rbase_get(staged_state_t* st, bool is_a, int sh)
{
    int o = is_a ? 0 : 1;
    for(int i = 0; i < st->n_rbase[o]; ++i)
        if(st->rbase_shift[o][i] == sh)
            return st->rbase[o][i];
    return NULL;
}

/* _rilv_ld(nm, stage, idx, kk): one fragment of the row-interleaved LDS layout */
static rocke_value_t* staged_rilv_ld(staged_state_t* st, bool is_a, int stage, int idx, int kk)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    int sh = staged_rsh(st, is_a, stage);
    rocke_value_t* row = staged_rbase_get(st, is_a, sh);
    if(sh || idx / 8)
    {
        rocke_value_t* c = rocke_b_const_i32(b, idx / 8 - sh);
        row = rocke_b_add(b, row, c);
    }
    rocke_value_t* cc
        = rocke_b_const_i32(b, (idx % 8) * ctx->block_k + kk * t->warp_tile_k * ctx->kmul);
    rocke_value_t* col = rocke_b_add(b, st->k_blk_kbase, cc);
    if(is_a)
        return rocke_gemm_frag_ld(
            ctx, ctx->A_stg[stage], row, col, NULL, ctx->a_per_lane, 'A', false);
    return rocke_gemm_frag_ld(ctx, ctx->B_stg[stage], row, col, NULL, ctx->b_per_lane, 'B', false);
}

/* _frags(stage, kk) -> fa (mfmas_m) + fb (mfmas_n) */
static int staged_frags(staged_state_t* st, int stage, int kk, rocke_value_t** out)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    int mi, ni, n = 0;
    if(ctx->rilv)
    {
        for(mi = 0; mi < ctx->mfmas_m; ++mi)
            out[n++] = staged_rilv_ld(st, true, stage, mi, kk);
        for(ni = 0; ni < ctx->mfmas_n; ++ni)
            out[n++] = staged_rilv_ld(st, false, stage, ni, kk);
        return n;
    }
    rocke_value_t* col_base
        = rocke_b_add(b, st->k_blk_kbase, rocke_b_const_i32(b, kk * t->warp_tile_k * ctx->kmul));
    for(mi = 0; mi < ctx->mfmas_m; ++mi)
    {
        rocke_value_t* row = staged_row_for(st, true, stage, mi);
        rocke_value_t* key = st->lane_key ? st->m_in_atom : st->a_row[mi];
        out[n++] = rocke_gemm_frag_ld(
            ctx, ctx->A_stg[stage], row, col_base, key, ctx->a_per_lane, 'A', true);
    }
    for(ni = 0; ni < ctx->mfmas_n; ++ni)
    {
        rocke_value_t* row = staged_row_for(st, false, stage, ni);
        rocke_value_t* key = st->lane_key ? st->n_in_atom : st->b_row[ni];
        out[n++] = rocke_gemm_frag_ld(
            ctx, ctx->B_stg[stage], row, col_base, key, ctx->b_per_lane, 'B', true);
    }
    return n;
}

static rocke_value_t* staged_mma_ab(
    staged_state_t* st, rocke_value_t* a_frag, rocke_value_t* b_frag, rocke_value_t* acc, int ni)
{
    return rocke_gemm_mma_ab(st->ctx, a_frag, b_frag, acc, ni);
}

/* _cluster(acc, fr, kk, pos) */
static void staged_cluster(staged_state_t* st,
                           rocke_value_t** acc,
                           rocke_value_t* const* fr,
                           int kk,
                           const staged_pos_t* pos,
                           const staged_issue_set_t* iss)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    int mfmas_m = ctx->mfmas_m, mfmas_n = ctx->mfmas_n;
    rocke_value_t* const* fa = fr;
    rocke_value_t* const* fb = fr + mfmas_m;
    int mi, ni;
    if(ctx->interwave)
    {
        rocke_b_sched_barrier(b, 0);
        rocke_b_s_setprio(b, 1);
    }
    for(mi = 0; mi < mfmas_m; ++mi)
    {
        if(ctx->swz && ctx->sched_hints && !ctx->interwave)
            rocke_b_s_setprio(b, 1);
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            int flat = mi * mfmas_n + ni;
            int idx = kk * mfmas_m * mfmas_n + flat;
            if(pos != NULL && pos->on)
            {
                int rel = idx - pos->first;
                if(rel >= 0 && rel < pos->span)
                {
                    int q;
                    for(q = 0; q < pos->n_at[rel]; ++q)
                        staged_issue_one(st, iss, pos->at[rel][q]);
                }
            }
            acc[flat] = staged_mma_ab(st, fa[mi], fb[ni], acc[flat], ni);
        }
        if(ctx->swz && ctx->sched_hints && !ctx->interwave)
        {
            rocke_b_s_setprio(b, 0);
            rocke_b_sched_barrier(b, 0);
        }
    }
    if(ctx->interwave)
    {
        rocke_b_s_setprio(b, 0);
        rocke_b_sched_barrier(b, 0);
    }
}

/* _frag_loaders(stage, kk): the fragments of _frags(stage, kk) as one deferred
 * load each, in MFMA use order (fa0, fb0..fb[n-1], fa1..); each fills its slot
 * of `out` (fa then fb). col_base is emitted lazily by the first load, after
 * that load's row (Python evaluates _row_for before _col_base()). */
typedef struct staged_frag_loader
{
    int stage;
    int kk;
    rocke_value_t* col_base;
    rocke_value_t** out;
} staged_frag_loader_t;

/* order index q -> (is_a, i): q 0 = fa0, 1..n = fb, then fa1.. */
static void staged_frag_order(const rocke_gemm_build_ctx_t* ctx, int q, bool* is_a, int* i)
{
    if(q == 0)
    {
        *is_a = true;
        *i = 0;
    }
    else if(q <= ctx->mfmas_n)
    {
        *is_a = false;
        *i = q - 1;
    }
    else
    {
        *is_a = true;
        *i = q - ctx->mfmas_n;
    }
}

static void staged_frag_one(staged_state_t* st, staged_frag_loader_t* fl, int q)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    bool is_a;
    int i;
    rocke_value_t* row;
    rocke_value_t* key;
    rocke_value_t* col;
    staged_frag_order(ctx, q, &is_a, &i);
    if(ctx->rilv)
    {
        fl->out[is_a ? i : ctx->mfmas_m + i] = staged_rilv_ld(st, is_a, fl->stage, i, fl->kk);
        return;
    }
    row = staged_row_for(st, is_a, fl->stage, i);
    if(fl->col_base == NULL)
        fl->col_base = rocke_b_add(
            b, st->k_blk_kbase, rocke_b_const_i32(b, fl->kk * t->warp_tile_k * ctx->kmul));
    (void)col;
    if(is_a)
    {
        key = st->lane_key ? st->m_in_atom : st->a_row[i];
        fl->out[i] = rocke_gemm_frag_ld(
            ctx, ctx->A_stg[fl->stage], row, fl->col_base, key, ctx->a_per_lane, 'A', true);
    }
    else
    {
        key = st->lane_key ? st->n_in_atom : st->b_row[i];
        fl->out[ctx->mfmas_m + i] = rocke_gemm_frag_ld(
            ctx, ctx->B_stg[fl->stage], row, fl->col_base, key, ctx->b_per_lane, 'B', true);
    }
}

/* ilv_barrier_after: what _at_barrier() emits once MFMA `pre` is reached. */
typedef struct staged_ilv_barrier
{
    int pre; /* 0: no in-cluster barrier                         */
    int s;
    int vm_wait;
    bool with_load;
    rocke_value_t* k_i;
    rocke_value_t* const* tile0s;
} staged_ilv_barrier_t;

/* _ilv_slots + _ilv_cluster(acc, fr, loads, issues, pre, at_barrier): every MFMA
 * followed by its share of the loads, then of the issues (each spread evenly over
 * MFMAs [lo, n): slot lo + (j * (n - lo)) // count, or lo + min(n - lo - 1,
 * j * every) for the issues with ilv_dtl_every), and a sched_barrier(0). With
 * bar->pre > 0 the wait + barrier + DTL issue set + next-stage loaders are
 * emitted right before MFMA `pre` and lo = pre. */
static void staged_ilv_cluster(staged_state_t* st,
                               rocke_value_t** acc,
                               rocke_value_t* const* fr,
                               staged_frag_loader_t* fl,
                               staged_issue_set_t* iss,
                               int n_issues,
                               const staged_ilv_barrier_t* bar)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    int mfmas_m = ctx->mfmas_m, mfmas_n = ctx->mfmas_n;
    int n = mfmas_m * mfmas_n;
    int n_loads = mfmas_m + mfmas_n;
    int every = ctx->spec->trait.ilv_dtl_every;
    int lo = 0;
    bool deferred = bar != NULL && bar->pre > 0;
    rocke_value_t* const* fa = fr;
    rocke_value_t* const* fb = fr + mfmas_m;
    int mi, ni, j;
    if(deferred)
    {
        n_loads = 0; /* nothing is placed before the barrier */
        n_issues = 0;
    }
    if(ctx->interwave)
    {
        rocke_b_sched_barrier(b, 0);
        rocke_b_s_setprio(b, 1);
    }
    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            int flat = mi * mfmas_n + ni;
            if(deferred && flat == bar->pre)
            {
                rocke_b_s_waitcnt(b, bar->vm_wait, 0, -1);
                rocke_b_s_barrier_bare(b);
                if(bar->with_load)
                {
                    rocke_value_t* cst = rocke_b_const_i32(b, st->S * ctx->block_k);
                    rocke_value_t* kld = rocke_b_add(b, bar->k_i, cst);
                    *iss = staged_dtl_issues(st, bar->s, kld, bar->tile0s, NULL);
                    n_issues = iss->count;
                }
                fl->stage = (bar->s + 1) % st->S;
                fl->kk = 0;
                n_loads = mfmas_m + mfmas_n;
                lo = bar->pre;
            }
            acc[flat] = staged_mma_ab(st, fa[mi], fb[ni], acc[flat], ni);
            if(ctx->spec->trait.ilv_read_span > 0 || ctx->spec->trait.ilv_dtl_span > 0)
            {
                /* Python _ilv_slots, ilv_read_span / ilv_dtl_span: the reads over
                 * the first rp % of [lo, n), the DTL issues over the first ip % */
                int rp = ctx->spec->trait.ilv_read_span > 0 ? ctx->spec->trait.ilv_read_span : 100;
                int ip = ctx->spec->trait.ilv_dtl_span > 0 ? ctx->spec->trait.ilv_dtl_span : 100;
                int sp_l = ((n - lo) * rp) / 100;
                int sp_i = ((n - lo) * ip) / 100;
                if(sp_l < 1)
                    sp_l = 1;
                if(sp_i < 1)
                    sp_i = 1;
                for(j = 0; j < n_loads; ++j)
                    if(lo + (j * sp_l) / n_loads == flat)
                        staged_frag_one(st, fl, j);
                for(j = 0; j < n_issues; ++j)
                    if(lo + (j * sp_i) / n_issues == flat)
                        staged_issue_one(st, iss, j);
                rocke_b_sched_barrier(b, 0);
                continue;
            }
            if(ctx->spec->trait.ilv_merge)
            {
                /* one schedule (Python _ilv_slots, ilv_merge): rank by
                 * ((2j+1) * other_count, kind, j), spread the ranks over [lo, n) */
                int ev_n = n_loads + n_issues;
                int r = 0;
                int jl = 0, jd = 0;
                int wl = n_issues > 0 ? n_issues : 1;
                int wd = n_loads > 0 ? n_loads : 1;
                while(jl < n_loads || jd < n_issues)
                {
                    bool take_load;
                    if(jl >= n_loads)
                        take_load = false;
                    else if(jd >= n_issues)
                        take_load = true;
                    else
                        take_load = (2 * jl + 1) * wl <= (2 * jd + 1) * wd;
                    if(lo + (r * (n - lo)) / ev_n == flat)
                    {
                        if(take_load)
                            staged_frag_one(st, fl, jl);
                        else
                            staged_issue_one(st, iss, jd);
                    }
                    if(take_load)
                        ++jl;
                    else
                        ++jd;
                    ++r;
                }
                rocke_b_sched_barrier(b, 0);
                continue;
            }
            for(j = 0; j < n_loads; ++j)
                if(lo + (j * (n - lo)) / n_loads == flat)
                    staged_frag_one(st, fl, j);
            for(j = 0; j < n_issues; ++j)
            {
                int span = n - lo;
                int at = every > 0 ? lo + (j * every < span - 1 ? j * every : span - 1)
                                   : lo + (j * span) / n_issues;
                if(at == flat)
                    staged_issue_one(st, iss, j);
            }
            rocke_b_sched_barrier(b, 0);
        }
    }
    if(ctx->interwave)
    {
        rocke_b_s_setprio(b, 0);
        rocke_b_sched_barrier(b, 0);
    }
}

/* ilv_early_release (early-release interleave): Python _ilv_cluster_ev. Each
 * event runs in front of MFMA `pos`, then its items (one kind per event) are
 * spread over [pos, next event's pos) as Python _ilv_slots does for one list. */
enum
{
    STG_EV_LOADS = 0, /* fragment loads of `fl` (no ops at the event)        */
    STG_EV_RELEASE = 1, /* lgkmcnt(0) + barrier, DTL set of tile i+S into s  */
    STG_EV_ACQUIRE = 2, /* vmcnt(vm) + barrier, fl -> stage (s+1) % S, kk 0   */
    STG_EV_CARRY = 3 /* the DTL issues [n1, count) of the released set        */
};

typedef struct staged_ev
{
    int pos;
    int kind;
} staged_ev_t;

typedef struct staged_ev_ctx
{
    int s;
    int vm;
    bool with_load;
    rocke_value_t* k_i;
    rocke_value_t* const* tile0s;
    int split_rel; /* mode 3: n1 from (n_at - rel, aq); 0: all issues */
    int split_aq;
    staged_issue_set_t* iss; /* the released set (kept for the carry) */
    int* n1; /* issues placed in the release atom                     */
} staged_ev_ctx_t;

static int
    staged_ev_slot(const rocke_gemm_build_ctx_t* ctx, bool is_issue, int lo, int hi, int j, int c)
{
    int span = hi - lo;
    int rp = ctx->spec->trait.ilv_read_span;
    int ip = ctx->spec->trait.ilv_dtl_span;
    if(rp > 0 || ip > 0)
    {
        int pct = is_issue ? (ip > 0 ? ip : 100) : (rp > 0 ? rp : 100);
        int sp = (span * pct) / 100;
        if(sp < 1)
            sp = 1;
        return lo + (j * sp) / c;
    }
    if(!ctx->spec->trait.ilv_merge && is_issue && ctx->spec->trait.ilv_dtl_every > 0)
    {
        int at = j * ctx->spec->trait.ilv_dtl_every;
        return lo + (at < span - 1 ? at : span - 1);
    }
    return lo + (j * span) / c;
}

static void staged_ilv_cluster_ev(staged_state_t* st,
                                  rocke_value_t** acc,
                                  rocke_value_t* const* fr,
                                  staged_frag_loader_t* fl,
                                  const staged_ev_t* evs,
                                  int n_ev,
                                  staged_ev_ctx_t* ec)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    int mfmas_m = ctx->mfmas_m, mfmas_n = ctx->mfmas_n;
    int n = mfmas_m * mfmas_n;
    rocke_value_t* const* fa = fr;
    rocke_value_t* const* fb = fr + mfmas_m;
    int cur = -1; /* active event */
    int lo = 0, hi = 0, cnt = 0, off = 0;
    bool is_issue = false;
    int mi, ni, j, e;
    if(ctx->interwave)
    {
        rocke_b_sched_barrier(b, 0);
        rocke_b_s_setprio(b, 1);
    }
    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            int flat = mi * mfmas_n + ni;
            for(e = 0; e < n_ev; ++e)
            {
                if(evs[e].pos != flat)
                    continue;
                cur = e;
                lo = flat;
                hi = e + 1 < n_ev ? evs[e + 1].pos : n;
                off = 0;
                switch(evs[e].kind)
                {
                case STG_EV_LOADS:
                    cnt = mfmas_m + mfmas_n;
                    is_issue = false;
                    break;
                case STG_EV_RELEASE:
                    rocke_b_s_waitcnt(b, -1, 0, -1);
                    rocke_b_s_barrier_bare(b);
                    ec->iss->count = 0;
                    if(ec->with_load)
                    {
                        rocke_value_t* cst = rocke_b_const_i32(b, st->S * ctx->block_k);
                        rocke_value_t* kld = rocke_b_add(b, ec->k_i, cst);
                        *ec->iss = staged_dtl_issues(st, ec->s, kld, ec->tile0s, NULL);
                    }
                    if(ec->split_rel > 0)
                    {
                        int a = n - ec->split_rel, w = a + ec->split_aq;
                        *ec->n1 = (ec->iss->count * a + w / 2) / w;
                    }
                    else
                        *ec->n1 = ec->iss->count;
                    cnt = *ec->n1;
                    is_issue = true;
                    break;
                case STG_EV_ACQUIRE:
                    rocke_b_s_waitcnt(b, ec->vm, -1, -1);
                    rocke_b_s_barrier_bare(b);
                    fl->stage = (ec->s + 1) % st->S;
                    fl->kk = 0;
                    cnt = mfmas_m + mfmas_n;
                    is_issue = false;
                    break;
                default: /* STG_EV_CARRY */
                    off = *ec->n1;
                    cnt = ec->iss->count - *ec->n1;
                    is_issue = true;
                    break;
                }
                rocke_b_sched_barrier(b, 0);
            }
            acc[flat] = staged_mma_ab(st, fa[mi], fb[ni], acc[flat], ni);
            if(cur >= 0)
                for(j = 0; j < cnt; ++j)
                    if(staged_ev_slot(ctx, is_issue, lo, hi, j, cnt) == flat)
                    {
                        if(is_issue)
                            staged_issue_one(st, ec->iss, off + j);
                        else
                            staged_frag_one(st, fl, j);
                    }
            rocke_b_sched_barrier(b, 0);
        }
    }
    if(ctx->interwave)
    {
        rocke_b_s_setprio(b, 0);
        rocke_b_sched_barrier(b, 0);
    }
}

/* _tile(s, k_i, acc, fr, vm_wait, with_load, tile0s): acc / fr updated in place
 * (fr only with frag prefetch). */
static void staged_tile(staged_state_t* st,
                        int s,
                        rocke_value_t* k_i,
                        rocke_value_t** acc,
                        rocke_value_t** fr,
                        int vm_wait,
                        bool with_load,
                        rocke_value_t* const* tile0s)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    int S = st->S;
    int k_atoms = ctx->k_atoms;
    int per = ctx->mfmas_m * ctx->mfmas_n;
    int kk;
    staged_pos_t pos;
    staged_issue_set_t iss;
    staged_issue_set_t er_iss; /* ilv_early_release: the released DTL set */
    int er_n1 = 0;
    memset(&iss, 0, sizeof(iss));
    memset(&er_iss, 0, sizeof(er_iss));

    if(st->iglp_set)
        rocke_b_iglp_opt(b, st->iglp);
    if(st->instream && !st->fprf)
    {
        for(kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* fr_k[STG_MAX_FRAGS];
            staged_frags(st, s, kk, fr_k);
            pos.on = false;
            if(kk + 1 == k_atoms)
            {
                rocke_b_s_waitcnt(b, vm_wait, 0, -1);
                rocke_b_s_barrier_bare(b);
                if(with_load)
                {
                    rocke_value_t* cst = rocke_b_const_i32(b, S * ctx->block_k);
                    rocke_value_t* kld = rocke_b_add(b, k_i, cst);
                    iss = staged_dtl_issues(st, s, kld, tile0s, NULL);
                    if(st->spread)
                        staged_spread_pos(&pos, iss.count, kk * per, per);
                    else
                        staged_issue_all(st, &iss);
                }
            }
            staged_cluster(st, acc, fr_k, kk, &pos, &iss);
        }
        return;
    }
    if(!st->fprf)
    {
        rocke_b_s_waitcnt(b, vm_wait, 0, -1);
        rocke_b_s_barrier_bare(b);
        pos.on = false;
        if(with_load)
        {
            rocke_value_t* cst = rocke_b_const_i32(b, (S - 1) * ctx->block_k);
            rocke_value_t* kld = rocke_b_add(b, k_i, cst);
            iss = staged_dtl_issues(st, (s + S - 1) % S, kld, tile0s, NULL);
            if(st->spread)
                staged_spread_pos(&pos, iss.count, 0, st->n_mfma_tile);
            else
                staged_issue_all(st, &iss);
        }
        for(kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* fr_k[STG_MAX_FRAGS];
            staged_frags(st, s, kk, fr_k);
            staged_cluster(st, acc, fr_k, kk, &pos, &iss);
        }
        return;
    }
    if(st->ilv)
    {
        /* frag-prefetch order with every ds_read / DTL issue pinned between two
         * MFMAs (Tensile-style interleave) */
        for(kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* nxt[STG_MAX_FRAGS];
            int n_fr = ctx->mfmas_m + ctx->mfmas_n;
            int n_issues = 0;
            int q;
            staged_frag_loader_t fl;
            fl.col_base = NULL;
            fl.out = nxt;
            staged_ilv_barrier_t bar;
            int er = ctx->spec->trait.ilv_early_release;
            int n_ld = ctx->mfmas_m + ctx->mfmas_n;
            memset(&bar, 0, sizeof(bar));
            if(er >= 2 && kk + 2 == k_atoms)
            {
                /* modes 2 / 3: the last atom's fragments are read first, then the
                 * release point inside THIS atom */
                staged_ev_t evs[2];
                staged_ev_ctx_t ec;
                int rel = ctx->spec->trait.ilv_release_at > 0
                              ? ctx->spec->trait.ilv_release_at
                              : (per - 1 < n_ld + n_ld / 2 ? per - 1 : n_ld + n_ld / 2);
                int aq3 = ctx->spec->trait.ilv_acquire_at > 0
                              ? ctx->spec->trait.ilv_acquire_at
                              : (1 > per - n_ld - n_ld / 2 ? 1 : per - n_ld - n_ld / 2);
                fl.stage = s;
                fl.kk = kk + 1;
                evs[0].pos = 0;
                evs[0].kind = STG_EV_LOADS;
                evs[1].pos = rel;
                evs[1].kind = STG_EV_RELEASE;
                memset(&ec, 0, sizeof(ec));
                ec.s = s;
                ec.with_load = with_load;
                ec.k_i = k_i;
                ec.tile0s = tile0s;
                ec.split_rel = er == 3 ? rel : 0;
                ec.split_aq = aq3;
                ec.iss = &er_iss;
                ec.n1 = &er_n1;
                staged_ilv_cluster_ev(st, acc, fr, &fl, evs, 2, &ec);
                for(q = 0; q < n_fr; ++q)
                    fr[q] = nxt[q];
                continue;
            }
            if(er >= 2 && kk + 1 == k_atoms)
            {
                staged_ev_t evs[2];
                staged_ev_ctx_t ec;
                int nev = 0;
                int aq;
                if(er == 2)
                    aq = ctx->spec->trait.ilv_acquire_at > 0
                             ? ctx->spec->trait.ilv_acquire_at
                             : (per - 1 < n_ld + n_ld / 2 ? per - 1 : n_ld + n_ld / 2);
                else
                    aq = ctx->spec->trait.ilv_acquire_at > 0
                             ? ctx->spec->trait.ilv_acquire_at
                             : (1 > per - n_ld - n_ld / 2 ? 1 : per - n_ld - n_ld / 2);
                if(er == 3)
                {
                    evs[nev].pos = 0;
                    evs[nev].kind = STG_EV_CARRY;
                    ++nev;
                }
                evs[nev].pos = aq;
                evs[nev].kind = STG_EV_ACQUIRE;
                ++nev;
                memset(&ec, 0, sizeof(ec));
                ec.s = s;
                ec.vm = vm_wait + (with_load ? st->L : 0);
                ec.with_load = with_load;
                ec.iss = &er_iss;
                ec.n1 = &er_n1;
                staged_ilv_cluster_ev(st, acc, fr, &fl, evs, nev, &ec);
                for(q = 0; q < n_fr; ++q)
                    fr[q] = nxt[q];
                continue;
            }
            if(kk + 1 < k_atoms)
            {
                fl.stage = s;
                fl.kk = kk + 1;
            }
            else if(er == 1)
            {
                /* mode 1: release at MFMA ilv_barrier_after, acquire later */
                staged_ev_t evs[2];
                staged_ev_ctx_t ec;
                int pre = ctx->spec->trait.ilv_barrier_after;
                int aq = ctx->spec->trait.ilv_acquire_at > 0
                             ? ctx->spec->trait.ilv_acquire_at
                             : (pre + 1 > per - 2 * n_ld ? pre + 1 : per - 2 * n_ld);
                evs[0].pos = pre;
                evs[0].kind = STG_EV_RELEASE;
                evs[1].pos = aq;
                evs[1].kind = STG_EV_ACQUIRE;
                memset(&ec, 0, sizeof(ec));
                ec.s = s;
                ec.vm = vm_wait + (with_load ? st->L : 0);
                ec.with_load = with_load;
                ec.k_i = k_i;
                ec.tile0s = tile0s;
                ec.iss = &er_iss;
                ec.n1 = &er_n1;
                staged_ilv_cluster_ev(st, acc, fr, &fl, evs, 2, &ec);
                for(q = 0; q < n_fr; ++q)
                    fr[q] = nxt[q];
                continue;
            }
            else if(ctx->spec->trait.ilv_barrier_after > 0)
            {
                /* the wait + barrier sit in front of MFMA `pre` of this cluster */
                bar.pre = ctx->spec->trait.ilv_barrier_after;
                bar.s = s;
                bar.vm_wait = vm_wait;
                bar.with_load = with_load;
                bar.k_i = k_i;
                bar.tile0s = tile0s;
            }
            else
            {
                rocke_b_s_waitcnt(b, vm_wait, 0, -1);
                rocke_b_s_barrier_bare(b);
                if(with_load)
                {
                    rocke_value_t* cst = rocke_b_const_i32(b, S * ctx->block_k);
                    rocke_value_t* kld = rocke_b_add(b, k_i, cst);
                    iss = staged_dtl_issues(st, s, kld, tile0s, NULL);
                    n_issues = iss.count;
                }
                fl.stage = (s + 1) % S;
                fl.kk = 0;
            }
            staged_ilv_cluster(st, acc, fr, &fl, &iss, n_issues, &bar);
            for(q = 0; q < n_fr; ++q)
                fr[q] = nxt[q];
        }
        return;
    }
    /* frag-prefetch order */
    for(kk = 0; kk < k_atoms; ++kk)
    {
        rocke_value_t* nxt[STG_MAX_FRAGS];
        int n_fr = ctx->mfmas_m + ctx->mfmas_n;
        int q;
        pos.on = false;
        if(kk + 1 < k_atoms)
        {
            staged_frags(st, s, kk + 1, nxt);
        }
        else
        {
            rocke_b_s_waitcnt(b, vm_wait, 0, -1);
            rocke_b_s_barrier_bare(b);
            if(with_load)
            {
                rocke_value_t* cst = rocke_b_const_i32(b, S * ctx->block_k);
                rocke_value_t* kld = rocke_b_add(b, k_i, cst);
                iss = staged_dtl_issues(st, s, kld, tile0s, NULL);
                if(st->spread)
                    staged_spread_pos(&pos, iss.count, kk * per, per);
                else
                    staged_issue_all(st, &iss);
            }
            staged_frags(st, (s + 1) % S, 0, nxt);
        }
        staged_cluster(st, acc, fr, kk, &pos, &iss);
        for(q = 0; q < n_fr; ++q)
            fr[q] = nxt[q];
    }
    (void)t;
}

static void staged_prologue(staged_state_t* st,
                            rocke_value_t* const* tile0s,
                            rocke_value_t* valid,
                            rocke_value_t* klo,
                            rocke_value_t* kup)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    rocke_value_t* kl = klo != NULL ? klo : ctx->k_lo;
    rocke_value_t* save = st->cur_kup;
    int j;
    st->cur_kup = kup;
    for(j = 0; j < st->n_pro; ++j)
    {
        rocke_value_t* cst = rocke_b_const_i32(b, j * ctx->block_k);
        rocke_value_t* kld = rocke_b_add(b, kl, cst);
        staged_issue_set_t iss = staged_dtl_issues(st, j, kld, tile0s, valid);
        staged_issue_all(st, &iss);
    }
    st->cur_kup = save;
}

static const char*
    staged_name(rocke_ir_builder_t* b, const char* fmt, const char* tag, int i, const char* nm)
{
    char buf[96];
    char* out;
    size_t n;
    if(nm != NULL)
        snprintf(buf, sizeof(buf), fmt, tag, i, nm);
    else
        snprintf(buf, sizeof(buf), fmt, tag, i);
    n = strlen(buf) + 1;
    out = (char*)rocke_arena_alloc(&b->arena, n);
    if(out != NULL)
        memcpy(out, buf, n);
    return out;
}

/* _kloop(acc_init, tile0s, tag): main loop + tail; writes out_acc. */
static void staged_kloop(staged_state_t* st,
                         rocke_value_t* const* acc_init,
                         rocke_value_t* const* tile0s,
                         const char* tag,
                         rocke_value_t** out_acc,
                         rocke_value_t* klo,
                         rocke_value_t* Tn,
                         rocke_value_t* kup)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    int S = st->S;
    int na = ctx->num_accs;
    int nf = ctx->mfmas_m + ctx->mfmas_n;
    rocke_value_t* fr0[STG_MAX_FRAGS];
    rocke_iter_arg_t args[ROCKE_GEMM_MAX_ACCS + STG_MAX_FRAGS];
    int nargs = 0;
    rocke_value_t* acc[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* fr[STG_MAX_FRAGS];
    rocke_value_t* k_main_hi;
    rocke_for_t loop;
    int i, s_;
    char ivn[32];
    /* Python kl_ / T_ / nm_ (a stream-K item passes its own K range) */
    rocke_value_t* kl = klo != NULL ? klo : ctx->k_lo;
    rocke_value_t* T_ = Tn != NULL ? Tn : st->T;
    rocke_value_t* nm = st->n_main;
    rocke_value_t* save_kup = st->cur_kup;
    if(Tn != NULL)
    {
        rocke_value_t* cs1 = rocke_b_const_i32(b, S);
        rocke_value_t* dv = rocke_b_div(b, Tn, cs1);
        rocke_value_t* cs2 = rocke_b_const_i32(b, S);
        nm = rocke_b_mul(b, dv, cs2);
    }
    st->cur_kup = kup;

    if(st->instream)
    {
        rocke_b_s_waitcnt(b, (S - 1) * st->L, 0, -1);
        rocke_b_s_barrier_bare(b);
    }
    if(st->fprf)
        staged_frags(st, 0, 0, fr0);

    for(i = 0; i < na; ++i)
    {
        args[nargs].name = staged_name(b, "%ssa%d_%s", tag, i, ctx->acc_names[i]);
        args[nargs].init = acc_init[i];
        ++nargs;
    }
    if(st->fprf)
    {
        for(i = 0; i < nf; ++i)
        {
            args[nargs].name = staged_name(b, "%ssf%d", tag, i, NULL);
            args[nargs].init = fr0[i];
            ++nargs;
        }
    }
    {
        rocke_value_t* m = rocke_b_mul(b, nm, ctx->c_block_k);
        k_main_hi = rocke_b_add(b, kl, m);
    }
    snprintf(ivn, sizeof(ivn), "%skg", tag);
    {
        rocke_value_t* step = rocke_b_const_i32(b, S * ctx->block_k);
        loop = rocke_b_scf_for_iter(
            b, kl, k_main_hi, step, args, nargs, rocke_arena_strdup(&b->arena, ivn), false, true);
    }
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* yv[ROCKE_GEMM_MAX_ACCS + STG_MAX_FRAGS];
        int ny = 0;
        for(i = 0; i < na; ++i)
            acc[i] = loop.iter_vars[i];
        if(st->fprf)
            for(i = 0; i < nf; ++i)
                fr[i] = loop.iter_vars[na + i];
        for(s_ = 0; s_ < S; ++s_)
        {
            rocke_value_t* k_i = loop.iv;
            if(s_)
                k_i = rocke_b_add(b, loop.iv, rocke_b_const_i32(b, s_ * ctx->block_k));
            staged_tile(st, s_, k_i, acc, fr, st->keep, true, tile0s);
        }
        for(i = 0; i < na; ++i)
            yv[ny++] = acc[i];
        if(st->fprf)
            for(i = 0; i < nf; ++i)
                yv[ny++] = fr[i];
        rocke_b_scf_yield(b, yv, ny);
    }
    rocke_b_region_leave(b);
    for(i = 0; i < na; ++i)
        acc[i] = loop.op->results[i];
    if(st->fprf)
        for(i = 0; i < nf; ++i)
            fr[i] = loop.op->results[na + i];

    /* tail: T % S tiles (< S), each a 0/1-trip loop, static stage s */
    {
        rocke_value_t* rem = rocke_b_sub(b, T_, nm);
        for(s_ = 0; s_ < S - 1; ++s_)
        {
            rocke_iter_arg_t targs[ROCKE_GEMM_MAX_ACCS + STG_MAX_FRAGS];
            int nt = 0;
            rocke_for_t tl;
            for(i = 0; i < na; ++i)
            {
                char fmt[32];
                snprintf(fmt, sizeof(fmt), "%%sst%da%%d", s_);
                targs[nt].name = staged_name(b, fmt, tag, i, NULL);
                targs[nt].init = acc[i];
                ++nt;
            }
            if(st->fprf)
            {
                for(i = 0; i < nf; ++i)
                {
                    char fmt[32];
                    snprintf(fmt, sizeof(fmt), "%%sst%df%%d", s_);
                    targs[nt].name = staged_name(b, fmt, tag, i, NULL);
                    targs[nt].init = fr[i];
                    ++nt;
                }
            }
            snprintf(ivn, sizeof(ivn), "%skt%d", tag, s_);
            {
                rocke_value_t* lo = rocke_b_const_i32(b, s_);
                rocke_value_t* step = rocke_b_const_i32(b, S);
                tl = rocke_b_scf_for_iter(
                    b, lo, rem, step, targs, nt, rocke_arena_strdup(&b->arena, ivn), false, true);
            }
            rocke_b_region_enter(b, tl.body);
            {
                rocke_value_t* a_[ROCKE_GEMM_MAX_ACCS];
                rocke_value_t* f_[STG_MAX_FRAGS];
                rocke_value_t* yv[ROCKE_GEMM_MAX_ACCS + STG_MAX_FRAGS];
                rocke_value_t* k_i;
                int ny = 0;
                for(i = 0; i < na; ++i)
                    a_[i] = tl.iter_vars[i];
                if(st->fprf)
                    for(i = 0; i < nf; ++i)
                        f_[i] = tl.iter_vars[na + i];
                {
                    rocke_value_t* cs = rocke_b_const_i32(b, s_);
                    rocke_value_t* ad = rocke_b_add(b, nm, cs);
                    rocke_value_t* mu = rocke_b_mul(b, ad, ctx->c_block_k);
                    k_i = rocke_b_add(b, kl, mu);
                }
                staged_tile(st, s_, k_i, a_, f_, (S - 2 - s_) * st->L, false, tile0s);
                for(i = 0; i < na; ++i)
                    yv[ny++] = a_[i];
                if(st->fprf)
                    for(i = 0; i < nf; ++i)
                        yv[ny++] = f_[i];
                rocke_b_scf_yield(b, yv, ny);
            }
            rocke_b_region_leave(b);
            for(i = 0; i < na; ++i)
                acc[i] = tl.op->results[i];
            if(st->fprf)
                for(i = 0; i < nf; ++i)
                    fr[i] = tl.op->results[na + i];
        }
    }
    for(i = 0; i < na; ++i)
        out_acc[i] = acc[i];
    st->cur_kup = save_kup;
}

/* persistent: _tile_offs(tid_flat) -> (bm, bn, (t0a, t0b)) */
static void staged_tile_offs(staged_state_t* st,
                             rocke_value_t* tid_flat,
                             rocke_value_t* n_pid_m,
                             rocke_value_t* n_pid_n,
                             rocke_value_t** bm,
                             rocke_value_t** bn,
                             rocke_value_t** t0)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    const rocke_gemm_trait_spec_t* tr = &ctx->spec->trait;
    rocke_value_t* row;
    rocke_value_t* col;
    if(tr->chiplet_swizzle)
    {
        rocke_super_tile_swizzle_result_t sw
            = rocke_chiplet_aware_super_tile_dynamic(b,
                                                     tid_flat,
                                                     n_pid_m,
                                                     n_pid_n,
                                                     tr->chiplet_wgm,
                                                     tr->chiplet_num_xcds,
                                                     tr->chiplet_chunk_size);
        row = sw.row;
        col = sw.col;
    }
    else
    {
        row = rocke_b_div(b, tid_flat, n_pid_n);
        col = rocke_b_mod(b, tid_flat, n_pid_n);
    }
    *bm = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, row, ctx->c_block_m));
    *bn = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, col, ctx->c_block_n));
    t0[0] = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, *bm, ctx->K));
    t0[1] = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_gemm_brow(ctx, *bn), ctx->K));
}

/* ===================================================================== *
 * stream-K persistent loop (Python _streamk_loop /
 * _sk_fixup in _emit_kloop_staged). Every op in Python order.
 * ===================================================================== */
typedef struct staged_sk
{
    staged_state_t* st;
    rocke_value_t* c0;
    rocke_value_t* c1;
    rocke_value_t* ipt;
    rocke_value_t* total;
    rocke_value_t* wg;
    rocke_value_t* c_p;
    rocke_value_t* n_pid_m;
    rocke_value_t* n_pid_n;
    rocke_value_t* n_dp;
    rocke_value_t* first_tile;
    rocke_value_t* dp_iters;
    rocke_value_t* per;
    rocke_value_t* extra;
    rocke_streamk_iter_range_t rng;
    rocke_value_t* c_waves;
    rocke_value_t* voff_p;
    int nq;
    int slot;
} staged_sk_t;

static void sk_item(
    staged_sk_t* k, rocke_value_t* it, rocke_value_t** tl, rocke_value_t** lo, rocke_value_t** hi)
{
    rocke_ir_builder_t* b = k->st->b;
    rocke_value_t* is_dp = rocke_b_cmp_lt(b, it, k->n_dp);
    rocke_value_t* mu = rocke_b_mul(b, it, k->c_p);
    rocke_value_t* dp_tile = rocke_b_add(b, k->wg, mu);
    rocke_value_t* su = rocke_b_sub(b, it, k->n_dp);
    rocke_value_t* sk_tile = rocke_b_add(b, k->first_tile, su);
    rocke_value_t* t1 = rocke_b_select(b, is_dp, dp_tile, sk_tile);
    rocke_value_t* l = rocke_b_cmp_lt(b, t1, k->total);
    rocke_value_t* se = rocke_b_select(b, l, t1, k->c0);
    rocke_value_t* ts;
    rocke_value_t* st_;
    rocke_value_t* a1;
    rocke_value_t* en_;
    rocke_value_t* mx;
    rocke_value_t* s1;
    rocke_value_t* a2;
    rocke_value_t* mn;
    rocke_value_t* s2;
    *tl = rocke_b_to_sgpr_u32(b, se);
    ts = rocke_b_mul(b, *tl, k->ipt);
    st_ = rocke_b_select(b, is_dp, ts, k->rng.start);
    a1 = rocke_b_add(b, ts, k->ipt);
    en_ = rocke_b_select(b, is_dp, a1, k->rng.end);
    mx = rocke_b_smax(b, st_, ts);
    s1 = rocke_b_sub(b, mx, ts);
    *lo = rocke_b_to_sgpr_u32(b, s1);
    a2 = rocke_b_add(b, ts, k->ipt);
    mn = rocke_b_smin(b, en_, a2);
    s2 = rocke_b_sub(b, mn, ts);
    *hi = rocke_b_to_sgpr_u32(b, s2);
}

static rocke_value_t* sk_kb(staged_sk_t* k, rocke_value_t* v)
{
    rocke_ir_builder_t* b = k->st->b;
    return rocke_b_to_sgpr_u32(b, rocke_b_mul(b, v, k->st->ctx->c_block_k));
}

static rocke_value_t* sk_flag_idx(staged_sk_t* k, rocke_value_t* w)
{
    rocke_ir_builder_t* b = k->st->b;
    rocke_value_t* mu = rocke_b_mul(b, w, k->c_waves);
    return rocke_b_add(b, mu, k->st->ctx->warp_id);
}

static rocke_value_t* sk_part_rsrc(staged_sk_t* k, rocke_value_t* w)
{
    rocke_ir_builder_t* b = k->st->b;
    rocke_value_t* cs = rocke_b_const_i32(b, k->slot * 4);
    rocke_value_t* mu = rocke_b_mul(b, w, cs);
    rocke_value_t* base = rocke_b_to_sgpr_u32(b, mu);
    rocke_value_t* gp = rocke_b_global_ptr_add(b, k->st->ctx->sk_part, base);
    rocke_value_t* nb = rocke_b_const_i32(b, k->slot * 4);
    return rocke_b_buffer_rsrc(b, gp, nb);
}

static rocke_value_t* sk_soff(staged_sk_t* k, int ai, int q)
{
    return rocke_b_const_i32(k->st->b, (ai * k->nq + q) * k->st->ctx->threads * 16);
}

static rocke_value_t* sk_load_atom(staged_sk_t* k, rocke_value_t* rs, int ai)
{
    rocke_ir_builder_t* b = k->st->b;
    int cpl = k->st->ctx->c_per_lane;
    rocke_value_t* qs[16];
    rocke_value_t* comps[64];
    int q, i;
    for(q = 0; q < k->nq; ++q)
    {
        rocke_value_t* so = sk_soff(k, ai, q);
        qs[q] = rocke_b_buffer_load_vN(b, rs, k->voff_p, so, rocke_f32(), 4);
    }
    if(cpl == 4)
        return qs[0];
    for(i = 0; i < cpl; ++i)
        comps[i] = rocke_b_vec_extract(b, qs[i / 4], i % 4);
    return rocke_b_vec_pack(b, comps, cpl, rocke_f32());
}

static rocke_value_t* sk_cta_of(staged_sk_t* k, rocke_value_t* it)
{
    rocke_ir_builder_t* b = k->st->b;
    rocke_value_t* rel = rocke_b_sub(b, it, k->dp_iters);
    rocke_value_t* wide = rocke_b_add(b, k->per, k->c1);
    rocke_value_t* boundary = rocke_b_mul(b, k->extra, wide);
    rocke_value_t* lead = rocke_b_div(b, rel, wide);
    rocke_value_t* sb = rocke_b_sub(b, rel, boundary);
    rocke_value_t* dv = rocke_b_div(b, sb, k->per);
    rocke_value_t* tail = rocke_b_add(b, k->extra, dv);
    rocke_value_t* l = rocke_b_cmp_lt(b, rel, boundary);
    return rocke_b_select(b, l, lead, tail);
}

static rocke_value_t* sk_agpr_pin_out(rocke_ir_builder_t* b, rocke_value_t* r)
{
    rocke_inline_asm_opts_t opts;
    const rocke_type_t* ty = r->type;
    rocke_op_t* op;
    memset(&opts, 0, sizeof opts);
    opts.sideeffect = true;
    opts.sideeffect_set = true;
    op = rocke_b_inline_asm(b, "", "=a,0", &r, 1, &ty, 1, &opts);
    return op != NULL ? op->results[0] : NULL;
}

static void sk_store_partial(staged_sk_t* k, rocke_value_t* w, rocke_value_t* const* vals)
{
    rocke_gemm_build_ctx_t* ctx = k->st->ctx;
    rocke_ir_builder_t* b = k->st->b;
    int cpl = ctx->c_per_lane;
    rocke_value_t* rs = sk_part_rsrc(k, w);
    int ai, q, i;
    for(ai = 0; ai < ctx->num_accs; ++ai)
    {
        rocke_value_t* v = rocke_gemm_acc_f32_out(b, ctx->spec, vals[ai]);
        rocke_value_t* qv[16];
        if(cpl == 4)
        {
            qv[0] = v;
        }
        else
        {
            for(q = 0; q < k->nq; ++q)
            {
                rocke_value_t* comps[4];
                for(i = 0; i < 4; ++i)
                    comps[i] = rocke_b_vec_extract(b, v, 4 * q + i);
                qv[q] = rocke_b_vec_pack(b, comps, 4, rocke_f32());
            }
        }
        for(q = 0; q < (cpl == 4 ? 1 : k->nq); ++q)
        {
            rocke_value_t* so = sk_soff(k, ai, q);
            rocke_b_buffer_store_vN_f32(b, rs, k->voff_p, so, qv[q], 4);
        }
    }
    rocke_b_fence(b, "agent", "release");
    rocke_b_global_flag_store(b, ctx->sk_flags, sk_flag_idx(k, w), k->c1, "agent", "monotonic");
}

static void sk_fixup(staged_sk_t* k,
                     rocke_value_t* const* acc,
                     rocke_value_t* hi_p,
                     rocke_value_t* bm,
                     rocke_value_t* bn)
{
    rocke_gemm_build_ctx_t* ctx = k->st->ctx;
    rocke_ir_builder_t* b = k->st->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    int na = ctx->num_accs;
    int cpl = ctx->c_per_lane;
    int grp = (spec->trait.acc_agpr ? 64 : 16) / cpl;
    rocke_value_t* vals[ROCKE_GEMM_MAX_ACCS];
    rocke_for_t pl;
    int g0, i;
    if(grp < 1)
        grp = 1;
    pl = rocke_b_scf_for(b, k->c1, hi_p, k->c1, "skw");
    rocke_b_region_enter(b, pl.body);
    {
        rocke_value_t* a = rocke_b_add(b, k->wg, pl.iv);
        rocke_value_t* fi = sk_flag_idx(k, a);
        rocke_b_global_flag_wait_eq_ex(b, ctx->sk_flags, fi, k->c1, "agent", "monotonic");
        rocke_b_fence(b, "agent", "acquire");
    }
    rocke_b_region_leave(b);
    for(i = 0; i < na; ++i)
        vals[i] = acc[i];
    for(g0 = 0; g0 < na; g0 += grp)
    {
        int g1 = g0 + grp < na ? g0 + grp : na;
        int n = g1 - g0;
        rocke_iter_arg_t args[ROCKE_GEMM_MAX_ACCS];
        rocke_for_t fold;
        for(i = 0; i < n; ++i)
        {
            args[i].name = rocke_arena_printf(&b->arena, "skf_%s", ctx->acc_names[g0 + i]);
            args[i].init = rocke_gemm_acc_f32_out(b, spec, vals[g0 + i]);
        }
        fold = rocke_b_scf_for_iter(b,
                                    k->c1,
                                    hi_p,
                                    k->c1,
                                    args,
                                    n,
                                    rocke_arena_printf(&b->arena, "skp%d", g0),
                                    false,
                                    true);
        rocke_b_region_enter(b, fold.body);
        {
            rocke_value_t* a = rocke_b_add(b, k->wg, fold.iv);
            rocke_value_t* rs = sk_part_rsrc(k, a);
            rocke_value_t* parts[ROCKE_GEMM_MAX_ACCS];
            rocke_value_t* sums[ROCKE_GEMM_MAX_ACCS];
            for(i = 0; i < n; ++i)
                parts[i] = sk_load_atom(k, rs, g0 + i);
            for(i = 0; i < n; ++i)
                sums[i] = rocke_b_vector_add(b, fold.iter_vars[i], parts[i]);
            rocke_b_scf_yield(b, sums, n);
        }
        rocke_b_region_leave(b);
        for(i = 0; i < n; ++i)
        {
            rocke_value_t* r = fold.op->results[i];
            vals[g0 + i] = spec->trait.acc_agpr ? sk_agpr_pin_out(b, r) : r;
        }
    }
    pl = rocke_b_scf_for(b, k->c1, hi_p, k->c1, "skz");
    rocke_b_region_enter(b, pl.body);
    {
        rocke_value_t* a = rocke_b_add(b, k->wg, pl.iv);
        rocke_value_t* fi = sk_flag_idx(k, a);
        rocke_b_global_flag_store(b, ctx->sk_flags, fi, k->c0, "agent", "monotonic");
    }
    rocke_b_region_leave(b);
    rocke_gemm_ep_default(ctx, vals, na, bm, bn, NULL, NULL, false);
}

static void staged_streamk(staged_state_t* st,
                           rocke_value_t* total,
                           rocke_value_t* wg,
                           rocke_value_t* c_p,
                           rocke_value_t* n_pid_m,
                           rocke_value_t* n_pid_n)
{
    rocke_gemm_build_ctx_t* ctx = st->ctx;
    rocke_ir_builder_t* b = st->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    staged_sk_t k;
    rocke_streamk_iter_args_t sk_args;
    rocke_value_t* in_pool;
    rocke_value_t* n_sk;
    rocke_value_t* n_items;
    rocke_value_t* tl0;
    rocke_value_t* lo0;
    rocke_value_t* hi0;
    rocke_value_t* bm0;
    rocke_value_t* bn0;
    rocke_value_t* t00[2];
    rocke_for_t pl;
    memset(&k, 0, sizeof(k));
    k.st = st;
    k.c0 = ctx->c0;
    k.ipt = st->T;
    k.total = total;
    k.wg = wg;
    k.c_p = c_p;
    k.n_pid_m = n_pid_m;
    k.n_pid_n = n_pid_n;
    k.c1 = rocke_b_const_i32(b, 1);
    {
        rocke_value_t* rem_t = rocke_b_mod(b, total, c_p);
        rocke_value_t* lt = rocke_b_cmp_lt(b, c_p, total);
        rocke_value_t* ad = rocke_b_add(b, c_p, rem_t);
        rocke_value_t* two_t = rocke_b_select(b, lt, ad, total);
        rocke_value_t* eq = rocke_b_cmp_eq(b, rem_t, k.c0);
        rocke_value_t* sk_t = rocke_b_select(b, eq, k.c0, two_t);
        rocke_value_t* mu = rocke_b_mul(b, sk_t, k.ipt);
        rocke_value_t* l2 = rocke_b_cmp_lt(b, mu, c_p);
        rocke_value_t* se = rocke_b_select(b, l2, k.c0, sk_t);
        rocke_value_t* dp_t;
        rocke_value_t* sk_iters;
        rocke_value_t* n_dp;
        sk_t = rocke_b_to_sgpr_u32(b, se);
        dp_t = rocke_b_to_sgpr_u32(b, rocke_b_sub(b, total, sk_t));
        sk_iters = rocke_b_mul(b, sk_t, k.ipt);
        {
            rocke_value_t* dv = rocke_b_div(b, sk_iters, c_p);
            rocke_value_t* mx = rocke_b_smax(b, dv, k.c1);
            k.per = rocke_b_to_sgpr_u32(b, mx);
        }
        k.extra = rocke_b_to_sgpr_u32(b, rocke_b_mod(b, sk_iters, c_p));
        k.dp_iters = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, dp_t, k.ipt));
        sk_args.iters_per_tile = k.ipt;
        sk_args.dp_tiles = dp_t;
        sk_args.total_dp_iters = k.dp_iters;
        sk_args.iters_per_sk_cta = k.per;
        sk_args.extra_iters = k.extra;
        in_pool = rocke_b_cmp_lt(b, wg, c_p);
        {
            rocke_value_t* s1 = rocke_b_sub(b, dp_t, wg);
            rocke_value_t* s2 = rocke_b_sub(b, c_p, k.c1);
            rocke_value_t* a = rocke_b_add(b, s1, s2);
            rocke_value_t* d = rocke_b_div(b, a, c_p);
            n_dp = rocke_b_select(b, in_pool, d, k.c0);
        }
        k.rng = rocke_emit_streamk_iter_range(b, wg, &sk_args, false);
        k.first_tile = rocke_b_to_sgpr_u32(b, rocke_b_div(b, k.rng.start, k.ipt));
        {
            rocke_value_t* s = rocke_b_sub(b, k.rng.end, k.c1);
            rocke_value_t* last_tile = rocke_b_div(b, s, k.ipt);
            rocke_value_t* s2 = rocke_b_sub(b, last_tile, k.first_tile);
            n_sk = rocke_b_add(b, s2, k.c1);
        }
        {
            rocke_value_t* l = rocke_b_cmp_lt(b, k.c0, sk_t);
            rocke_value_t* la = rocke_b_land(b, in_pool, l);
            n_sk = rocke_b_select(b, la, n_sk, k.c0);
        }
        k.n_dp = rocke_b_to_sgpr_u32(b, n_dp);
        n_items = rocke_b_to_sgpr_u32(b, rocke_b_add(b, k.n_dp, n_sk));
    }
    k.c_waves = rocke_b_const_i32(b, t->warp_m * t->warp_n * t->warp_k);
    k.nq = ctx->c_per_lane % 4 == 0 ? ctx->c_per_lane / 4 : 0;
    k.slot = ctx->num_accs * ctx->c_per_lane * ctx->threads;
    {
        rocke_value_t* c16 = rocke_b_const_i32(b, 16);
        k.voff_p = rocke_b_mul(b, ctx->tid, c16);
    }

    sk_item(&k, k.c0, &tl0, &lo0, &hi0);
    staged_tile_offs(st, tl0, n_pid_m, n_pid_n, &bm0, &bn0, t00);
    {
        rocke_value_t* v = rocke_b_cmp_lt(b, k.c0, n_items);
        rocke_value_t* kl = sk_kb(&k, lo0);
        rocke_value_t* ku = sk_kb(&k, hi0);
        staged_prologue(st, t00, v, kl, ku);
    }
    pl = rocke_b_scf_for(b, k.c0, n_items, k.c1, "skitem");
    rocke_b_region_enter(b, pl.body);
    {
        rocke_value_t* tl;
        rocke_value_t* lo;
        rocke_value_t* hi;
        rocke_value_t* bm;
        rocke_value_t* bn;
        rocke_value_t* t0[2];
        rocke_value_t* acc[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* nxt;
        rocke_value_t* has_next;
        rocke_value_t* tl1;
        rocke_value_t* lo1;
        rocke_value_t* hi1;
        rocke_value_t* bm1;
        rocke_value_t* bn1;
        rocke_value_t* t01[2];
        rocke_value_t* started;
        rocke_value_t* ended;
        rocke_if_else_t ie;
        sk_item(&k, pl.iv, &tl, &lo, &hi);
        staged_tile_offs(st, tl, n_pid_m, n_pid_n, &bm, &bn, t0);
        {
            rocke_value_t* kl = sk_kb(&k, lo);
            rocke_value_t* sb = rocke_b_sub(b, hi, lo);
            rocke_value_t* tn = rocke_b_to_sgpr_u32(b, sb);
            rocke_value_t* ku = sk_kb(&k, hi);
            staged_kloop(st, ctx->acc_inits, t0, "p", acc, kl, tn, ku);
        }
        nxt = rocke_b_add(b, pl.iv, k.c1);
        has_next = rocke_b_cmp_lt(b, nxt, n_items);
        sk_item(&k, rocke_b_select(b, has_next, nxt, pl.iv), &tl1, &lo1, &hi1);
        staged_tile_offs(st, tl1, n_pid_m, n_pid_n, &bm1, &bn1, t01);
        rocke_b_s_waitcnt(b, -1, 0, -1);
        rocke_b_s_barrier_bare(b);
        {
            rocke_value_t* kl1 = sk_kb(&k, lo1);
            rocke_value_t* ku1 = sk_kb(&k, hi1);
            staged_prologue(st, t01, has_next, kl1, ku1);
        }
        started = rocke_b_cmp_eq(b, lo, k.c0);
        ended = rocke_b_cmp_eq(b, hi, k.ipt);
        ie = rocke_b_scf_if_else(b, started);
        rocke_b_region_enter(b, ie.then_region);
        {
            rocke_value_t* mu = rocke_b_mul(b, tl, k.ipt);
            rocke_value_t* tile_end = rocke_b_add(b, mu, k.ipt);
            rocke_value_t* s = rocke_b_sub(b, tile_end, k.c1);
            rocke_value_t* last = sk_cta_of(&k, s);
            rocke_value_t* d = rocke_b_sub(b, last, wg);
            rocke_value_t* n_part = rocke_b_select(b, ended, k.c0, d);
            rocke_value_t* l = rocke_b_cmp_lt(b, k.c0, n_part);
            rocke_if_else_t ie2 = rocke_b_scf_if_else(b, l);
            rocke_b_region_enter(b, ie2.then_region);
            {
                rocke_value_t* a = rocke_b_add(b, n_part, k.c1);
                rocke_value_t* hp = rocke_b_to_sgpr_u32(b, a);
                sk_fixup(&k, acc, hp, bm, bn);
            }
            rocke_b_region_leave(b);
            rocke_b_region_enter(b, ie2.else_region);
            rocke_gemm_ep_default(ctx, acc, ctx->num_accs, bm, bn, NULL, NULL, false);
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        sk_store_partial(&k, wg, acc);
        rocke_b_region_leave(b);
    }
    rocke_b_region_leave(b);
}

void rocke_gemm_emit_kloop_staged(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    staged_state_t* st;
    int i, j;

    st = (staged_state_t*)rocke_arena_alloc(&b->arena, sizeof(staged_state_t));
    if(st == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "gemm staged: state OOM");
        return;
    }
    memset(st, 0, sizeof(*st));
    st->ctx = ctx;
    st->b = b;
    st->S = ctx->stages;
    st->L = ctx->dtl_a_passes + ctx->dtl_b_passes;
    st->keep = (st->S - 2) * st->L;
    st->fprf = spec->trait.dtl_frag_prefetch;
    st->instream = st->fprf || spec->trait.dtl_stream_barrier;
    st->spread = spec->trait.dtl_issue_spread;
    st->ilv = spec->trait.mfma_interleave;
    st->iglp_set = spec->trait.iglp_opt_set;
    st->iglp = spec->trait.iglp_opt;
    st->k_up = ctx->k_upper;
    st->c2s = rocke_b_const_i32(b, 2);

    /* loop-invariant DTL addressing */
    {
        int wave_bytes = spec->wave_size * ctx->dtl_bytes_per_lane;
        st->pass_bytes = ctx->dtl_pass_bytes;
        if(ctx->rilv)
        {
            /* every wave's 1 KiB DTL block is followed by the rilv_pad-byte row
             * pad (one padded alloc row per wave and pass) */
            wave_bytes += spec->trait.rilv_pad ? spec->trait.rilv_pad : 16;
            st->pass_bytes = (spec->block_size / spec->wave_size) * wave_bytes;
        }
        if(t->warp_m * t->warp_n * t->warp_k > 1)
        {
            rocke_value_t* wv = rocke_b_to_sgpr_u32(b, ctx->warp_id);
            rocke_value_t* cst = rocke_b_const_i32(b, wave_bytes);
            rocke_value_t* mu = rocke_b_mul(b, wv, cst);
            rocke_value_t* sg = rocke_b_to_sgpr_u32(b, mu);
            st->wpo = rocke_b_zext(b, sg, rocke_i64());
        }
        else
        {
            st->wpo = NULL;
        }
    }
    staged_stage_bases(st);
    staged_lane_offs(st, ctx->dtl_a_passes, ctx->mfmas_m, false, st->a_voff);
    staged_lane_offs(st, ctx->dtl_b_passes, ctx->mfmas_n, true, st->b_voff);
    st->pers = spec->trait.persistent_ctas;
    if(!st->pers)
    {
        rocke_value_t* ma = rocke_b_mul(b, ctx->block_m_off, ctx->K);
        rocke_value_t* aa = rocke_b_add(b, ctx->batch_off_a, ma);
        rocke_value_t* mb;
        rocke_value_t* ab;
        st->a_tile0 = rocke_b_to_sgpr_u32(b, aa);
        mb = rocke_b_mul(b, rocke_gemm_brow(ctx, ctx->block_n_off), ctx->K);
        ab = rocke_b_add(b, ctx->batch_off_b, mb);
        st->b_tile0 = rocke_b_to_sgpr_u32(b, ab);
    }
    st->c_big = rocke_b_const_i32(b, ROCKE_GEMM_DTL_BUFFER_NUM_RECORDS);
    st->c_zero_rec = rocke_b_const_i32(b, 0);
    st->tight = spec->trait.dtl_tight_rsrc;
    st->cur_kup = NULL;
    st->wide = spec->trait.wide_addr;
    st->fold[0] = st->fold[1] = NULL;
    if(st->wide)
    {
        /* wide_addr: 64-bit per-operand base at the output tile's origin (row
         * origin * K * 2 bytes in i64); a K tile's resource is base + k * 2.
         * dtl_tight_rsrc: the range is the operand bytes left from the tile
         * origin, clamped to the buffer range. */
        rocke_value_t* kb = rocke_b_mul(b, ctx->K, st->c2s);
        rocke_value_t* kbs = rocke_b_to_sgpr_u32(b, kb);
        rocke_value_t* row_b = rocke_b_zext(b, kbs, rocke_i64());
        rocke_value_t* za = rocke_b_zext(b, ctx->block_m_off, rocke_i64());
        rocke_value_t* org_a = rocke_b_mul(b, za, row_b);
        rocke_value_t* bnr = rocke_gemm_brow(ctx, ctx->block_n_off);
        rocke_value_t* zb = rocke_b_zext(b, bnr, rocke_i64());
        rocke_value_t* org_b = rocke_b_mul(b, zb, row_b);
        st->fold[0] = rocke_b_global_ptr_add(b, ctx->A, org_a);
        st->fold[1] = rocke_b_global_ptr_add(b, ctx->Bp, org_b);
        if(st->tight)
        {
            rocke_value_t* c_big64 = rocke_b_const_i64(b, ROCKE_GEMM_DTL_BUFFER_NUM_RECORDS);
            for(int o = 0; o < 2; ++o)
            {
                /* _w_left(rows, org64) */
                rocke_value_t* rows = o == 0 ? ctx->M : rocke_gemm_brow(ctx, ctx->N);
                rocke_value_t* zr = rocke_b_zext(b, rows, rocke_i64());
                rocke_value_t* mr = rocke_b_mul(b, zr, row_b);
                rocke_value_t* left = rocke_b_sub(b, mr, o == 0 ? org_a : org_b);
                rocke_value_t* lt = rocke_b_cmp_lt(b, left, c_big64);
                rocke_value_t* sel = rocke_b_select(b, lt, left, c_big64);
                rocke_value_t* tr32 = rocke_b_trunc(b, sel, rocke_i32());
                rocke_value_t* nb = rocke_b_to_sgpr_u32(b, tr32);
                if(o == 0)
                    st->a_bytes = nb;
                else
                    st->b_bytes = nb;
            }
        }
    }
    else if(st->tight)
    {
        rocke_value_t* mk = rocke_b_mul(b, ctx->M, ctx->K);
        rocke_value_t* by = rocke_b_mul(b, mk, st->c2s);
        st->a_bytes = rocke_b_to_sgpr_u32(b, by);
        mk = rocke_b_mul(b, rocke_gemm_brow(ctx, ctx->N), ctx->K);
        by = rocke_b_mul(b, mk, st->c2s);
        st->b_bytes = rocke_b_to_sgpr_u32(b, by);
    }

    /* loop-invariant fragment addressing */
    st->m_in_atom = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_m));
    st->k_blk = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_m));
    st->n_in_atom = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_n));
    st->warp_m_off = rocke_b_to_sgpr_u32(
        b, rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, ctx->mfmas_m * t->warp_tile_m)));
    st->warp_n_off = rocke_b_to_sgpr_u32(
        b, rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, ctx->mfmas_n * t->warp_tile_n)));
    st->k_blk_kbase = rocke_b_mul(b, st->k_blk, rocke_b_const_i32(b, ctx->a_per_lane * ctx->kmul));
    for(i = 0; i < ctx->mfmas_m; ++i)
        st->a_row[i]
            = rocke_b_add(b,
                          st->warp_m_off,
                          rocke_b_add(b, rocke_b_const_i32(b, i * t->warp_tile_m), st->m_in_atom));
    for(i = 0; i < ctx->mfmas_n; ++i)
        st->b_row[i]
            = rocke_b_add(b,
                          st->warp_n_off,
                          rocke_b_add(b, rocke_b_const_i32(b, i * t->warp_tile_n), st->n_in_atom));
    {
        int span = 1 << (ctx->swz_r + ctx->swz_w);
        st->lane_key = (t->warp_tile_m % span == 0) && (t->warp_tile_n % span == 0);
    }
    /* element width from the storage dtype (Python: type_bits(storage_dtype) // 8) */
    st->row_bytes = ctx->block_k * (rocke_i_type_bits(ctx->storage_dtype) / 8);
    {
        int o = 0;
        int elem = rocke_i_type_bits(ctx->storage_dtype) / 8;
        for(j = 0; j < st->S; ++j)
        {
            st->offs_a[j] = o;
            o += ctx->rilv ? (ctx->block_m / 8) * ctx->rilv_cols * elem
                           : ctx->block_m * st->row_bytes;
        }
        for(j = 0; j < st->S; ++j)
        {
            st->offs_b[j] = o;
            o += ctx->rilv ? (ctx->block_n / 8) * ctx->rilv_cols * elem
                           : ctx->block_n * st->row_bytes;
        }
    }
    if(st->lane_key && !ctx->rilv)
    {
        int op_i;
        for(op_i = 0; op_i < 2; ++op_i)
        {
            bool is_a = op_i == 0;
            for(j = 0; j < st->S; ++j)
            {
                int sh = staged_shift_rows(st, is_a, j);
                if(sh && staged_opq_get(st, is_a, sh) == NULL)
                {
                    rocke_value_t* w_off = is_a ? st->warp_m_off : st->warp_n_off;
                    rocke_value_t* in_atom = is_a ? st->m_in_atom : st->n_in_atom;
                    rocke_value_t* s1 = rocke_b_add(b, w_off, in_atom);
                    rocke_value_t* cst = rocke_b_const_i32(b, sh);
                    rocke_value_t* s2 = rocke_b_add(b, s1, cst);
                    rocke_value_t* ob = rocke_b_optimization_barrier(b, s2);
                    if(is_a)
                    {
                        st->opq_shift_a[st->n_opq_a] = sh;
                        st->opq_a[st->n_opq_a++] = ob;
                    }
                    else
                    {
                        st->opq_shift_b[st->n_opq_b] = sh;
                        st->opq_b[st->n_opq_b++] = ob;
                    }
                }
            }
        }
    }
    if(ctx->rilv)
    {
        /* lds_row_interleave fragment addressing: LDS alloc row (8 tile rows) of
         * the wave's first row of each 8-row group, per operand and row shift */
        int ro = 0;
        st->rb = ctx->rilv_cols * (rocke_i_type_bits(ctx->storage_dtype) / 8);
        for(j = 0; j < st->S; ++j)
        {
            st->roffs_a[j] = ro;
            ro += (ctx->block_m / 8) * st->rb;
        }
        for(j = 0; j < st->S; ++j)
        {
            st->roffs_b[j] = ro;
            ro += (ctx->block_n / 8) * st->rb;
        }
        for(int o = 0; o < 2; ++o)
        {
            bool is_a = o == 0;
            int mf = is_a ? ctx->mfmas_m : ctx->mfmas_n;
            rocke_value_t* w_off = is_a ? st->warp_m_off : st->warp_n_off;
            rocke_value_t* in_atom = is_a ? st->m_in_atom : st->n_in_atom;
            rocke_value_t* c8 = rocke_b_const_i32(b, 8);
            rocke_value_t* dv = rocke_b_div(b, w_off, c8);
            rocke_value_t* cm = rocke_b_const_i32(b, mf / 8);
            rocke_value_t* mu = rocke_b_mul(b, in_atom, cm);
            rocke_value_t* lr8 = rocke_b_add(b, dv, mu);
            st->n_rbase[o] = 0;
            for(j = 0; j < st->S; ++j)
            {
                int sh = staged_rsh(st, is_a, j);
                if(staged_rbase_get(st, is_a, sh) != NULL)
                    continue;
                rocke_value_t* v = lr8;
                if(sh)
                {
                    rocke_value_t* cs = rocke_b_const_i32(b, sh);
                    rocke_value_t* ad = rocke_b_add(b, lr8, cs);
                    v = rocke_b_optimization_barrier(b, ad);
                }
                st->rbase_shift[o][st->n_rbase[o]] = sh;
                st->rbase[o][st->n_rbase[o]++] = v;
            }
        }
    }
    st->n_mfma_tile = ctx->k_atoms * ctx->mfmas_m * ctx->mfmas_n;

    /* driver */
    {
        rocke_value_t* sb = rocke_b_sub(b, st->k_up, ctx->k_lo);
        rocke_value_t* dv = rocke_b_div(b, sb, ctx->c_block_k);
        st->T = rocke_b_to_sgpr_u32(b, dv);
    }
    {
        rocke_value_t* cs1 = rocke_b_const_i32(b, st->S);
        rocke_value_t* dv = rocke_b_div(b, st->T, cs1);
        rocke_value_t* cs2 = rocke_b_const_i32(b, st->S);
        st->n_main = rocke_b_mul(b, dv, cs2);
    }
    st->n_pro = st->instream ? st->S : st->S - 1;

    if(!st->pers)
    {
        staged_prologue(st, NULL, NULL, NULL, NULL);
        staged_kloop(st, ctx->acc_inits, NULL, "", ctx->for_results, NULL, NULL, NULL);
        ctx->num_for_results = ctx->num_accs;
        ctx->staged_persistent_done = false;
        return;
    }

    /* persistent body */
    {
        rocke_value_t* n_pid_m;
        rocke_value_t* n_pid_n;
        rocke_value_t* total;
        rocke_value_t* wg;
        rocke_value_t* c_p;
        rocke_value_t* mine;
        rocke_value_t* first;
        rocke_value_t* bm0;
        rocke_value_t* bn0;
        rocke_value_t* t00[2];
        rocke_value_t* hi;
        rocke_for_t pl;

        if(spec->_fused_epilogue != NULL)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "persistent_ctas does not support a fused epilogue");
            return;
        }
        {
            rocke_value_t* c1 = rocke_b_const_i32(b, ctx->block_m - 1);
            rocke_value_t* ad = rocke_b_add(b, ctx->M, c1);
            n_pid_m = rocke_b_div(b, ad, ctx->c_block_m);
        }
        {
            rocke_value_t* c1 = rocke_b_const_i32(b, ctx->block_n - 1);
            rocke_value_t* ad = rocke_b_add(b, ctx->N, c1);
            n_pid_n = rocke_b_div(b, ad, ctx->c_block_n);
        }
        total = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, n_pid_m, n_pid_n));
        {
            rocke_value_t* by = rocke_b_block_id_y(b);
            rocke_value_t* mu = rocke_b_mul(b, by, n_pid_n);
            rocke_value_t* bx = rocke_b_block_id_x(b);
            rocke_value_t* ad = rocke_b_add(b, mu, bx);
            wg = rocke_b_to_sgpr_u32(b, ad);
        }
        c_p = rocke_b_const_i32(b, st->pers);
        if(spec->trait.streamk)
        {
            staged_streamk(st, total, wg, c_p, n_pid_m, n_pid_n);
            ctx->num_for_results = 0;
            ctx->staged_persistent_done = true;
            return;
        }
        {
            rocke_value_t* l1 = rocke_b_cmp_lt(b, wg, c_p);
            rocke_value_t* l2 = rocke_b_cmp_lt(b, wg, total);
            mine = rocke_b_land(b, l1, l2);
        }
        {
            rocke_value_t* lt = rocke_b_cmp_lt(b, wg, total);
            rocke_value_t* sel = rocke_b_select(b, lt, wg, ctx->c0);
            first = rocke_b_to_sgpr_u32(b, sel);
        }
        staged_tile_offs(st, first, n_pid_m, n_pid_n, &bm0, &bn0, t00);
        staged_prologue(st, t00, mine, NULL, NULL);
        hi = rocke_b_select(b, mine, total, wg);
        pl = rocke_b_scf_for(b, wg, hi, c_p, "ptile");
        rocke_b_region_enter(b, pl.body);
        {
            rocke_value_t* bm;
            rocke_value_t* bn;
            rocke_value_t* t0[2];
            rocke_value_t* acc[ROCKE_GEMM_MAX_ACCS];
            rocke_value_t* tnext;
            rocke_value_t* has_next;
            rocke_value_t* bm1;
            rocke_value_t* bn1;
            rocke_value_t* t01[2];
            staged_tile_offs(st, pl.iv, n_pid_m, n_pid_n, &bm, &bn, t0);
            staged_kloop(st, ctx->acc_inits, t0, "p", acc, NULL, NULL, NULL);
            tnext = rocke_b_add(b, pl.iv, c_p);
            has_next = rocke_b_cmp_lt(b, tnext, total);
            {
                rocke_value_t* sel = rocke_b_select(b, has_next, tnext, pl.iv);
                staged_tile_offs(st, sel, n_pid_m, n_pid_n, &bm1, &bn1, t01);
            }
            rocke_b_s_waitcnt(b, -1, 0, -1);
            rocke_b_s_barrier_bare(b);
            staged_prologue(st, t01, has_next, NULL, NULL);
            rocke_gemm_ep_default(ctx, acc, ctx->num_accs, bm, bn, NULL, NULL, false);
        }
        rocke_b_region_leave(b);
        ctx->num_for_results = 0;
        ctx->staged_persistent_done = true;
    }
}
