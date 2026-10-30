// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/* gemm_build_vp.cpp -- C++ mirror of the register-staged global prefetch body
 * of gemm_universal.py (TraitSpec.vgpr_prefetch, the K-contiguous / RCR form):
 *
 *   rocke_gemm_vp_setup          the `_vp_*` loader state (buffer resources at
 *                                the tile origin, per-lane voffsets) emitted
 *                                after the DTL plumbing, as the Python does
 *   rocke_gemm_emit_kloop_vpf    `_emit_kloop_vpf`      (vgpr_prefetch 1 / 2)
 *   rocke_gemm_emit_kloop_vpf_ilv `_emit_kloop_vpf_ilv` (vgpr_prefetch 3 / 4 and
 *                                the levers vgpr_prefetch_depth, vp_unroll,
 *                                vp_late_reads, vp_release_atoms, vp_tail_pack,
 *                                vp_pair_reads, vp_mfma_first)
 *
 * Every IR op is emitted in the Python order (the byte-identity contract). The
 * Python closures that defer emission (the per-instruction read / write /
 * fragment thunks the interleave places on MFMA slots) are std::function
 * thunks here, created and run at the same points the Python creates and runs
 * them (a thunk list whose creation emits ops -- the soffset of the global
 * reads, the parity rows of the LDS writes -- emits them at creation, as the
 * Python list comprehensions do). Nested builder calls with more than one
 * emitting argument are split into temporaries so the C++ evaluation order
 * matches Python's left-to-right order.
 *
 * Operand hooks. The global reads (vp_global_read), the LDS writes
 * (vp_store_one) and the interleave's fragment reads (vp_frag_read) are the
 * single points every path goes through, so a fused operand prologue (raw
 * f32 operands, complex B) can be added there; the MN-contiguous loader
 * (vp_mn_*) is not mirrored (the converters refuse it). */

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/instance_gemm_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

typedef std::function<void()> vp_thunk_t;

/* a Python `int | Value` LDS parity: v != NULL is the runtime Value form */
struct vp_par
{
    int imm;
    rocke_value_t* v;
};

static vp_par vp_par_imm(int i)
{
    vp_par p;
    p.imm = i;
    p.v = NULL;
    return p;
}

static vp_par vp_par_val(rocke_value_t* v)
{
    vp_par p;
    p.imm = 0;
    p.v = v;
    return p;
}

struct rocke_gemm_vp_state
{
    rocke_value_t* c2; /* _vp_c2                               */
    rocke_value_t* a_rsrc; /* _vp_a_rsrc                           */
    rocke_value_t* b_rsrc; /* _vp_b_rsrc                           */
    std::vector<rocke_value_t*> a_row, a_col; /* _vp_a_rc                */
    std::vector<rocke_value_t*> b_row, b_col; /* _vp_b_rc                */
    std::vector<rocke_value_t*> a_voff, b_voff; /* _vp_a_voff / _vp_b_voff */
    rocke_tile_window_t a_lds; /* _vp_a_lds                            */
    rocke_tile_window_t b_lds; /* _vp_b_lds                            */
    rocke_value_t* oob; /* _vp_oob                              */
    int g; /* _vp_g: elements per LDS write        */
};

extern "C" void rocke_gemm_vp_state_free(struct rocke_gemm_vp_state* st)
{
    delete st;
}

/* ---- the loader state (Python: the `if _vpf:` block after emit_load_phase) ---- */

/* _vp_rsrc(ptr, row0, rows_total, padded) */
/* _vp_batch_off64(nm): the batch element offset of operand nm as an i64
 * (wide_addr); NULL when not batched (no op) */
extern "C" rocke_value_t* rocke_gemm_vp_batch_off64(rocke_gemm_build_ctx_t* ctx, char nm)
{
    rocke_ir_builder_t* b = ctx->b;
    if(!ctx->spec->batched)
        return NULL;
    const rocke_type_t* I64 = rocke_i64();
    rocke_value_t* s1 = nm == 'A' ? ctx->stride_a : (nm == 'B' ? ctx->stride_b : ctx->stride_c);
    rocke_value_t* zi = rocke_b_zext(b, ctx->batch_idx, I64);
    rocke_value_t* zs = rocke_b_zext(b, s1, I64);
    return rocke_b_mul(b, zi, zs);
}

/* _vp_rsrc_wide(ptr, row0, rows_total, padded, nm): the tile origin (batch
 * offset + row0 * K elements) as a 64-bit byte offset of the base; only the
 * tile's own span must fit the i32 voffsets and the buffer range */
static rocke_value_t* vp_rsrc_wide(rocke_gemm_build_ctx_t* ctx,
                                   rocke_value_t* ptr,
                                   rocke_value_t* row0,
                                   rocke_value_t* rows_total,
                                   bool padded,
                                   char nm)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* I64 = rocke_i64();
    rocke_value_t* k64 = rocke_b_zext(b, ctx->K, I64);
    rocke_value_t* b64 = rocke_gemm_vp_batch_off64(ctx, nm);
    rocke_value_t* e = rocke_b_mul(b, rocke_b_zext(b, row0, I64), k64);
    if(b64 != NULL)
        e = rocke_b_add(b, b64, e);
    rocke_value_t* org = rocke_b_mul(b, e, rocke_b_const_i64(b, 2));
    rocke_value_t* rec;
    if(padded)
    {
        rocke_value_t* tt = rocke_b_mul(b, rocke_b_zext(b, rows_total, I64), k64);
        if(b64 != NULL)
            tt = rocke_b_add(b, b64, tt);
        rocke_value_t* tb = rocke_b_mul(b, tt, rocke_b_const_i64(b, 2));
        rocke_value_t* left = rocke_b_sub(b, tb, org);
        rocke_value_t* big = rocke_b_const_i64(b, ROCKE_GEMM_DTL_BUFFER_NUM_RECORDS);
        rocke_value_t* lt = rocke_b_cmp_lt(b, left, big);
        left = rocke_b_select(b, lt, left, big);
        rocke_value_t* neg = rocke_b_cmp_lt(b, left, rocke_b_const_i64(b, 0));
        rocke_value_t* z = rocke_b_const_i64(b, 0);
        rocke_value_t* cl = rocke_b_select(b, neg, z, left);
        rec = rocke_b_to_sgpr_u32(b, rocke_b_trunc(b, cl, rocke_i32()));
    }
    else
        rec = rocke_b_const_i32(b, ROCKE_GEMM_DTL_BUFFER_NUM_RECORDS);
    return rocke_b_buffer_rsrc(b, rocke_b_global_ptr_add(b, ptr, org), rec);
}

static rocke_value_t* vp_rsrc(rocke_gemm_build_ctx_t* ctx,
                              rocke_gemm_vp_state* st,
                              rocke_value_t* ptr,
                              rocke_value_t* row0,
                              rocke_value_t* rows_total,
                              bool padded,
                              char nm)
{
    rocke_ir_builder_t* b = ctx->b;
    if(ctx->spec->trait.wide_addr)
        return vp_rsrc_wide(ctx, ptr, row0, rows_total, padded, nm);
    if(ctx->spec->batched)
    {
        /* batched: the batch element offset rides in the base too (launch rule:
         * batch offset + rows * K < 2^31 elements) */
        rocke_value_t* bo = nm == 'A' ? ctx->batch_off_a : ctx->batch_off_b;
        rocke_value_t* e = rocke_b_add(b, bo, rocke_b_mul(b, row0, ctx->K));
        rocke_value_t* org = rocke_b_to_sgpr_u32(b, rocke_b_add_nuw(b, e, e));
        rocke_value_t* rec;
        if(padded)
        {
            rocke_value_t* tt = rocke_b_add(b, bo, rocke_b_mul(b, rows_total, ctx->K));
            rocke_value_t* nbytes = rocke_b_to_sgpr_u32(b, rocke_b_add_nuw(b, tt, tt));
            rec = rocke_b_smax(b, rocke_b_sub(b, nbytes, org), ctx->c0);
        }
        else
            rec = rocke_b_const_i32(b, ROCKE_GEMM_DTL_BUFFER_NUM_RECORDS);
        return rocke_b_buffer_rsrc(b, rocke_b_global_ptr_add(b, ptr, org), rec);
    }
    /* byte offsets reach 2^32 (launch_shape_error): x + x as an unsigned add */
    rocke_value_t* e = rocke_b_mul(b, row0, ctx->K);
    rocke_value_t* org = rocke_b_to_sgpr_u32(b, rocke_b_add_nuw(b, e, e));
    rocke_value_t* rec;
    if(padded)
    {
        /* range ends at the operand's last byte: rows past M / N read 0 */
        rocke_value_t* nbytes
            = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_mul(b, rows_total, ctx->K), st->c2));
        rec = rocke_b_smax(b, rocke_b_sub(b, nbytes, org), ctx->c0);
    }
    else
        rec = rocke_b_const_i32(b, ROCKE_GEMM_DTL_BUFFER_NUM_RECORDS);
    return rocke_b_buffer_rsrc(b, rocke_b_global_ptr_add(b, ptr, org), rec);
}

/* _vp_rc(e) */
static void vp_rc(rocke_gemm_build_ctx_t* ctx, int e, rocke_value_t** row, rocke_value_t** col)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* vec_idx
        = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), ctx->c_threads), ctx->tid);
    *row = rocke_b_div(b, vec_idx, ctx->c_block_k_div_vec);
    *col = rocke_b_mul(b, rocke_b_mod(b, vec_idx, ctx->c_block_k_div_vec), ctx->c_load_vec);
}

extern "C" void rocke_gemm_vp_setup(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    rocke_gemm_vp_state* st = new rocke_gemm_vp_state();
    ctx->vp = st;
    st->c2 = rocke_b_const_i32(b, 2);
    /* complex B: tile row r of B' is complex row r >> 1, so the base is at
     * complex row block_n_off / 2 and the operand has N / 2 rows */
    rocke_value_t* b_row0 = ctx->block_n_off;
    rocke_value_t* b_rows = ctx->N;
    if(ctx->cplx)
    {
        b_row0 = rocke_gemm_brow(ctx, ctx->block_n_off);
        b_rows = rocke_gemm_brow(ctx, ctx->N);
    }
    st->a_rsrc = vp_rsrc(ctx, st, ctx->A, ctx->block_m_off, ctx->M, spec->trait.pad_m, 'A');
    st->b_rsrc = vp_rsrc(ctx, st, ctx->Bp, b_row0, b_rows, spec->trait.pad_n, 'B');
    for(int e = 0; e < ctx->a_vecs_per_thread; ++e)
    {
        rocke_value_t *r, *c;
        vp_rc(ctx, e, &r, &c);
        st->a_row.push_back(r);
        st->a_col.push_back(c);
    }
    for(int e = 0; e < ctx->b_vecs_per_thread; ++e)
    {
        rocke_value_t *r, *c;
        vp_rc(ctx, e, &r, &c);
        st->b_row.push_back(r);
        st->b_col.push_back(c);
    }
    for(size_t i = 0; i < st->a_row.size(); ++i)
        st->a_voff.push_back(rocke_b_mul(
            b, rocke_b_add(b, rocke_b_mul(b, st->a_row[i], ctx->K), st->a_col[i]), st->c2));
    for(size_t i = 0; i < st->b_row.size(); ++i)
        st->b_voff.push_back(rocke_b_mul(
            b,
            rocke_b_add(
                b, rocke_b_mul(b, rocke_gemm_brow(ctx, st->b_row[i]), ctx->K), st->b_col[i]),
            st->c2));
    {
        int a_len[2] = {ctx->block_m, ctx->block_k};
        int b_len[2] = {ctx->block_n, ctx->block_k};
        rocke_value_t* org[2] = {ctx->c0, ctx->c0};
        rocke_make_tile_window(&st->a_lds, &ctx->a_lds_view, a_len, org, 2);
        rocke_make_tile_window(&st->b_lds, &ctx->b_lds_view, b_len, org, 2);
    }
    /* pad_k: a vector whose K column is past K reads from an offset beyond the
     * resource range (the buffer load returns zeros) */
    st->oob = rocke_b_const_i32(b, 0x7FFFFFF0);
    /* with lds_swizzle the XOR moves 2^L-half groups: a loaded vector wider than
     * a group is written as one ds_write per group */
    st->g = ctx->swz ? std::min(ctx->load_vec, 1 << ctx->swz_l) : ctx->load_vec;
}

/* ---- loads / stores (the operand hook points) ---- */

/* _vp_kmask(voff, col, k_off) */
static rocke_value_t* vp_kmask(rocke_gemm_build_ctx_t* ctx,
                               rocke_value_t* voff,
                               rocke_value_t* col,
                               rocke_value_t* k_off)
{
    rocke_ir_builder_t* b = ctx->b;
    if(!ctx->spec->trait.pad_k)
        return voff;
    rocke_value_t* lt = rocke_b_cmp_lt(b, rocke_b_add(b, col, k_off), ctx->K);
    return rocke_b_select(b, lt, voff, ctx->vp->oob);
}

/* one global read of a K-contiguous operand vector (A: is_a) */
static rocke_value_t* vp_global_read(
    rocke_gemm_build_ctx_t* ctx, bool is_a, size_t i, rocke_value_t* k_off, rocke_value_t* soff)
{
    rocke_gemm_vp_state* st = ctx->vp;
    rocke_value_t* rs = is_a ? st->a_rsrc : st->b_rsrc;
    rocke_value_t* v = is_a ? st->a_voff[i] : st->b_voff[i];
    rocke_value_t* c = is_a ? st->a_col[i] : st->b_col[i];
    rocke_value_t* vo = vp_kmask(ctx, v, c, k_off);
    return rocke_b_buffer_load_vN(ctx->b, rs, vo, soff, ctx->storage_dtype, ctx->load_vec);
}

/* _vp_global_reads(k_off): every read of one K tile (A then B) */
static std::vector<rocke_value_t*> vp_global_reads(rocke_gemm_build_ctx_t* ctx,
                                                   rocke_value_t* k_off)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_gemm_vp_state* st = ctx->vp;
    std::vector<rocke_value_t*> out;
    rocke_value_t* soff = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, k_off, st->c2));
    for(size_t i = 0; i < st->a_voff.size(); ++i)
        out.push_back(vp_global_read(ctx, true, i, k_off, soff));
    for(size_t i = 0; i < st->b_voff.size(); ++i)
        out.push_back(vp_global_read(ctx, false, i, k_off, soff));
    return out;
}

/* _vp_par_row(r, parity, rows) */
static rocke_value_t*
    vp_par_row(rocke_gemm_build_ctx_t* ctx, rocke_value_t* r, vp_par par, int rows)
{
    rocke_ir_builder_t* b = ctx->b;
    if(par.v != NULL)
        return rocke_b_add(b, r, rocke_b_mul(b, par.v, rocke_b_const_i32(b, rows)));
    return par.imm ? rocke_b_add(b, r, rocke_b_const_i32(b, par.imm * rows)) : r;
}

/* one LDS write of elements j .. j + n of the vector v (n == load_vec: the
 * whole vector) at LDS row `row` (parity-full) of the tile row r, column c */
static void vp_store_one(rocke_gemm_build_ctx_t* ctx,
                         const rocke_tile_window_t* win,
                         rocke_value_t* row,
                         rocke_value_t* r,
                         rocke_value_t* c,
                         rocke_value_t* v,
                         int j)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_gemm_vp_state* st = ctx->vp;
    if(st->g == ctx->load_vec)
    {
        rocke_value_t* idx[2] = {row, rocke_gemm_swz_col(ctx, c, r)};
        rocke_tile_window_store_vec(b, win, idx, 2, v, ctx->load_vec);
        return;
    }
    std::vector<rocke_value_t*> comps;
    for(int i = 0; i < st->g; ++i)
        comps.push_back(rocke_b_vec_extract(b, v, j + i));
    rocke_value_t* piece = rocke_b_vec_pack(b, comps.data(), st->g, ctx->storage_dtype);
    rocke_value_t* cj = j ? rocke_b_add(b, c, rocke_b_const_i32(b, j)) : c;
    rocke_value_t* idx[2] = {row, rocke_gemm_swz_col(ctx, cj, r)};
    rocke_tile_window_store_vec(b, win, idx, 2, piece, st->g);
}

/* _vp_store(win, row, r, c, v): every group of one vector, now */
static void vp_store(rocke_gemm_build_ctx_t* ctx,
                     const rocke_tile_window_t* win,
                     rocke_value_t* row,
                     rocke_value_t* r,
                     rocke_value_t* c,
                     rocke_value_t* v)
{
    rocke_gemm_vp_state* st = ctx->vp;
    if(st->g == ctx->load_vec)
    {
        vp_store_one(ctx, win, row, r, c, v, 0);
        return;
    }
    for(int j = 0; j < ctx->load_vec; j += st->g)
        vp_store_one(ctx, win, row, r, c, v, j);
}

/* _vp_lds_writes(vals, parity) */
static void
    vp_lds_writes(rocke_gemm_build_ctx_t* ctx, const std::vector<rocke_value_t*>& vals, vp_par par)
{
    rocke_gemm_vp_state* st = ctx->vp;
    size_t na = st->a_row.size();
    for(size_t i = 0; i < na; ++i)
    {
        rocke_value_t* row = vp_par_row(ctx, st->a_row[i], par, ctx->block_m);
        vp_store(ctx, &st->a_lds, row, st->a_row[i], st->a_col[i], vals[i]);
    }
    for(size_t i = 0; i < st->b_row.size(); ++i)
    {
        rocke_value_t* row = vp_par_row(ctx, st->b_row[i], par, ctx->block_n);
        vp_store(ctx, &st->b_lds, row, st->b_row[i], st->b_col[i], vals[na + i]);
    }
}

/* _vp_read_thunks(k_off, out): the soffset now, one thunk per read (appending
 * to `out` when it runs) */
static std::vector<vp_thunk_t> vp_read_thunks(rocke_gemm_build_ctx_t* ctx,
                                              rocke_value_t* k_off,
                                              std::vector<rocke_value_t*>* out)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_gemm_vp_state* st = ctx->vp;
    std::vector<vp_thunk_t> th;
    rocke_value_t* soff = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, k_off, st->c2));
    for(int op = 0; op < 2; ++op)
    {
        bool is_a = op == 0;
        size_t n = is_a ? st->a_voff.size() : st->b_voff.size();
        for(size_t i = 0; i < n; ++i)
            th.push_back([ctx, is_a, i, k_off, soff, out]() {
                out->push_back(vp_global_read(ctx, is_a, i, k_off, soff));
            });
    }
    return th;
}

/* _vp_store_thunks(win, row, r, c, v) */
static void vp_store_thunks(rocke_gemm_build_ctx_t* ctx,
                            const rocke_tile_window_t* win,
                            rocke_value_t* row,
                            rocke_value_t* r,
                            rocke_value_t* c,
                            rocke_value_t* v,
                            std::vector<vp_thunk_t>* out)
{
    rocke_gemm_vp_state* st = ctx->vp;
    if(st->g == ctx->load_vec)
    {
        out->push_back([ctx, win, row, r, c, v]() { vp_store_one(ctx, win, row, r, c, v, 0); });
        return;
    }
    for(int j = 0; j < ctx->load_vec; j += st->g)
        out->push_back([ctx, win, row, r, c, v, j]() { vp_store_one(ctx, win, row, r, c, v, j); });
}

/* _vp_write_thunks(vals, parity): the parity rows now, one thunk per write */
static std::vector<vp_thunk_t> vp_write_thunks(rocke_gemm_build_ctx_t* ctx,
                                               const std::vector<rocke_value_t*>& vals,
                                               vp_par par)
{
    rocke_gemm_vp_state* st = ctx->vp;
    std::vector<vp_thunk_t> th;
    size_t na = st->a_row.size();
    for(size_t i = 0; i < na; ++i)
    {
        rocke_value_t* row = vp_par_row(ctx, st->a_row[i], par, ctx->block_m);
        vp_store_thunks(ctx, &st->a_lds, row, st->a_row[i], st->a_col[i], vals[i], &th);
    }
    for(size_t i = 0; i < st->b_row.size(); ++i)
    {
        rocke_value_t* row = vp_par_row(ctx, st->b_row[i], par, ctx->block_n);
        vp_store_thunks(ctx, &st->b_lds, row, st->b_row[i], st->b_col[i], vals[na + i], &th);
    }
    return th;
}

/* _vp_k_end(): K (or the split-K slice end), rounded up to whole tiles with
 * pad_k */
static rocke_value_t* vp_k_end(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    if(!ctx->spec->trait.pad_k)
        return ctx->k_upper;
    rocke_value_t* up = rocke_b_add(b, ctx->k_upper, rocke_b_const_i32(b, ctx->block_k - 1));
    return rocke_b_mul(b, rocke_b_div(b, up, ctx->c_block_k), ctx->c_block_k);
}

static const char* vp_name(rocke_ir_builder_t* b, const std::string& s)
{
    return rocke_arena_strdup(&b->arena, s.c_str());
}

/* the prologue reads of tiles 0 .. D: tile 0 -> LDS (parity 0), tiles 1 .. D
 * into D register sets; lgkmcnt(0) + barrier */
static std::vector<rocke_value_t*> vp_prologue(rocke_gemm_build_ctx_t* ctx, rocke_value_t* k_last)
{
    rocke_ir_builder_t* b = ctx->b;
    std::vector<rocke_value_t*> regs0 = vp_global_reads(ctx, ctx->k_lo);
    vp_lds_writes(ctx, regs0, vp_par_imm(0));
    std::vector<rocke_value_t*> regs;
    for(int j = 1; j <= ctx->vpd; ++j)
    {
        rocke_value_t* kj = rocke_b_add(b, ctx->k_lo, rocke_b_const_i32(b, j * ctx->block_k));
        std::vector<rocke_value_t*> r = vp_global_reads(ctx, rocke_b_smin(b, kj, k_last));
        regs.insert(regs.end(), r.begin(), r.end());
    }
    rocke_b_s_waitcnt(b, -1, 0, -1);
    rocke_b_s_barrier_bare(b);
    return regs;
}

/* ===================================================================== *
 *  _emit_kloop_vpf: vgpr_prefetch 1 / 2 (the AB LDS double buffer)
 * ===================================================================== */
extern "C" void rocke_gemm_emit_kloop_vpf(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int vpf = ctx->vpf;
    rocke_value_t* c1_i32 = rocke_b_const_i32(b, 1);
    rocke_value_t* k_last = rocke_b_sub(b, vp_k_end(ctx), ctx->c_block_k);
    rocke_value_t* c_pf_tiles = rocke_b_const_i32(b, (ctx->vpd + 1) * ctx->block_k);

    std::vector<rocke_value_t*> regs = vp_prologue(ctx, k_last);
    int nr = (int)regs.size();
    int ns = nr / ctx->vpd;
    std::vector<rocke_iter_arg_t> loop_args;
    {
        rocke_iter_arg_t a;
        a.name = "par";
        a.init = ctx->c0;
        loop_args.push_back(a);
        for(int i = 0; i < nr; ++i)
        {
            a.name = vp_name(b, "vpr" + std::to_string(i));
            a.init = regs[i];
            loop_args.push_back(a);
        }
        for(int i = 0; i < ctx->num_accs; ++i)
        {
            a.name = ctx->acc_names[i];
            a.init = ctx->acc_inits[i];
            loop_args.push_back(a);
        }
    }
    rocke_for_t for_op = rocke_b_scf_for_iter(b,
                                              ctx->k_lo,
                                              k_last,
                                              ctx->c_block_k,
                                              loop_args.data(),
                                              (int)loop_args.size(),
                                              "k0",
                                              false,
                                              true);
    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* parity = for_op.iter_vars[0];
        std::vector<rocke_value_t*> cur(for_op.iter_vars + 1, for_op.iter_vars + 1 + ns);
        std::vector<rocke_value_t*> later(for_op.iter_vars + 1 + ns, for_op.iter_vars + 1 + nr);
        rocke_value_t* const* acc_iter = for_op.iter_vars + 1 + nr;
        int num_acc = for_op.num_iter_vars - 1 - nr;
        rocke_value_t* next_parity = rocke_b_sub(b, c1_i32, parity);
        rocke_value_t* k_pf = rocke_b_smin(b, rocke_b_add(b, for_op.iv, c_pf_tiles), k_last);
        std::vector<rocke_value_t*> new_regs;
        std::vector<rocke_value_t*> new_accs(num_acc > 0 ? num_acc : 1);
        /* sched_barrier(0) keeps the global reads where they are issued */
        if(vpf == 1)
        {
            new_regs = vp_global_reads(ctx, k_pf);
            vp_lds_writes(ctx, cur, vp_par_val(next_parity));
            rocke_b_sched_barrier(b, 0);
            rocke_gemm_emit_mfma_phase(
                ctx, ctx->A_smem, ctx->B_smem, acc_iter, num_acc, 0, parity, new_accs.data());
        }
        else
        {
            rocke_gemm_emit_mfma_phase(
                ctx, ctx->A_smem, ctx->B_smem, acc_iter, num_acc, 0, parity, new_accs.data());
            rocke_b_sched_barrier(b, 0);
            vp_lds_writes(ctx, cur, vp_par_val(next_parity));
            new_regs = vp_global_reads(ctx, k_pf);
        }
        rocke_b_s_waitcnt(b, -1, 0, -1);
        rocke_b_s_barrier_bare(b);
        std::vector<rocke_value_t*> y;
        y.push_back(next_parity);
        y.insert(y.end(), later.begin(), later.end());
        y.insert(y.end(), new_regs.begin(), new_regs.end());
        y.insert(y.end(), new_accs.begin(), new_accs.begin() + num_acc);
        rocke_b_scf_yield(b, y.data(), (int)y.size());
    }
    rocke_b_region_leave(b);

    rocke_value_t* final_parity = for_op.op->results[0];
    int num_tail = for_op.op->num_results - 1 - nr;
    rocke_gemm_emit_mfma_phase(ctx,
                               ctx->A_smem,
                               ctx->B_smem,
                               for_op.op->results + 1 + nr,
                               num_tail,
                               0,
                               final_parity,
                               ctx->for_results);
    ctx->num_for_results = num_tail;
}

/* ===================================================================== *
 *  _emit_kloop_vpf_ilv: vgpr_prefetch 3 / 4 (the instruction-level
 *  interleave; 4 = one LDS buffer)
 * ===================================================================== */
namespace
{

struct vp_ilv
{
    rocke_gemm_build_ctx_t* ctx;
    rocke_ir_builder_t* b;
    int KA;
    int MPA;
    bool single;
    int last_slot;
    rocke_value_t* m_in_atom;
    rocke_value_t* n_in_atom;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    rocke_value_t* k_blk_kbase;
};

typedef std::map<int, std::vector<vp_thunk_t>> vp_plan_t;

/* rd(kk, idx, is_a, par): one fragment read of K atom kk */
rocke_value_t* vp_frag_read(vp_ilv* s, int kk, int idx, bool is_a, vp_par par)
{
    rocke_gemm_build_ctx_t* ctx = s->ctx;
    rocke_ir_builder_t* b = s->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    rocke_value_t* col
        = rocke_b_add(b, s->k_blk_kbase, rocke_b_const_i32(b, kk * t->warp_tile_k * ctx->kmul));
    if(ctx->pe != NULL)
    {
        /* operand prologue: the raw fragment (f32 pair view: twice the halves)
         * through the prologue hook; each read chunk at its own swizzled column */
        if(is_a)
        {
            rocke_value_t* r0 = rocke_b_add(
                b,
                s->warp_m_off,
                rocke_b_add(b, rocke_b_const_i32(b, idx * t->warp_tile_m), s->m_in_atom));
            rocke_value_t* row = vp_par_row(ctx, r0, par, ctx->block_m);
            return rocke_gemm_frag_ld(ctx, ctx->A_smem, row, col, r0, ctx->a_per_lane, 'A', true);
        }
        rocke_value_t* r0
            = rocke_b_add(b,
                          s->warp_n_off,
                          rocke_b_add(b, rocke_b_const_i32(b, idx * t->warp_tile_n), s->n_in_atom));
        rocke_value_t* row = vp_par_row(ctx, r0, par, ctx->block_n);
        return rocke_gemm_frag_ld(ctx, ctx->B_smem, row, col, r0, ctx->b_per_lane, 'B', true);
    }
    if(is_a)
    {
        rocke_value_t* r0
            = rocke_b_add(b,
                          s->warp_m_off,
                          rocke_b_add(b, rocke_b_const_i32(b, idx * t->warp_tile_m), s->m_in_atom));
        rocke_value_t* row = vp_par_row(ctx, r0, par, ctx->block_m);
        rocke_value_t* sc = rocke_gemm_swz_col(ctx, col, r0);
        return rocke_gemm_emit_smem_load(
            b, ctx->A_smem, row, sc, ctx->a_per_lane, ctx->storage_dtype);
    }
    rocke_value_t* r0 = rocke_b_add(
        b, s->warp_n_off, rocke_b_add(b, rocke_b_const_i32(b, idx * t->warp_tile_n), s->n_in_atom));
    rocke_value_t* row = vp_par_row(ctx, r0, par, ctx->block_n);
    rocke_value_t* sc = rocke_gemm_swz_col(ctx, col, r0);
    return rocke_gemm_emit_smem_load(b, ctx->B_smem, row, sc, ctx->b_per_lane, ctx->storage_dtype);
}

/* frag_thunks(kk, par, fa, fb) */
std::vector<vp_thunk_t> vp_frag_thunks(
    vp_ilv* s, int kk, vp_par par, std::vector<rocke_value_t*>* fa, std::vector<rocke_value_t*>* fb)
{
    std::vector<vp_thunk_t> th;
    for(int i = 0; i < s->ctx->mfmas_m; ++i)
        th.push_back([s, kk, par, fa, i]() { (*fa)[i] = vp_frag_read(s, kk, i, true, par); });
    for(int i = 0; i < s->ctx->mfmas_n; ++i)
        th.push_back([s, kk, par, fb, i]() { (*fb)[i] = vp_frag_read(s, kk, i, false, par); });
    return th;
}

void vp_put(vp_ilv* s, int slot, const vp_thunk_t& it, vp_plan_t* plan)
{
    /* every item runs inside the tile: a slot past the last MFMA runs after the
     * last MFMA, still in order */
    (*plan)[std::min(slot, s->last_slot)].push_back(it);
}

void vp_spread(vp_ilv* s,
               const std::vector<vp_thunk_t>& items,
               std::vector<int> slots,
               vp_plan_t* plan)
{
    if(slots.empty())
        slots.push_back(0);
    int F = (int)slots.size();
    int n = (int)items.size();
    for(int j = 0; j < n; ++j)
        vp_put(s, slots[(j * F) / std::max(1, n)], items[j], plan);
}

std::vector<int> vp_range(int lo, int hi)
{
    std::vector<int> r;
    for(int i = lo; i < hi; ++i)
        r.push_back(i);
    return r;
}

struct vp_next
{
    vp_par npar;
    std::vector<rocke_value_t*> regs;
    rocke_value_t* k_pf;
};

/* tile(par, frags0, accs_in, nxt): one tile's MFMAs. Returns the accumulators;
 * with nxt, *frags_out gets the next tile's atom-0 fragments and *regs_out the
 * global reads of tile i + D + 1. */
bool vp_tile(vp_ilv* s,
             vp_par par,
             const std::vector<rocke_value_t*>& frags0,
             const std::vector<rocke_value_t*>& accs_in,
             const vp_next* nxt,
             std::vector<rocke_value_t*>* accs_out,
             std::vector<rocke_value_t*>* frags_out,
             std::vector<rocke_value_t*>* regs_out)
{
    rocke_gemm_build_ctx_t* ctx = s->ctx;
    rocke_ir_builder_t* b = s->b;
    const rocke_gemm_trait_spec_t* tr = &ctx->spec->trait;
    int mm = ctx->mfmas_m, mn = ctx->mfmas_n;
    int KA = s->KA, MPA = s->MPA;
    /* atoms[kk] = (fa, fb): stable storage (the thunks write into it) */
    std::map<int, std::pair<std::vector<rocke_value_t*>, std::vector<rocke_value_t*>>> atoms;
    atoms[0].first.assign(frags0.begin(), frags0.begin() + mm);
    atoms[0].second.assign(frags0.begin() + mm, frags0.begin() + mm + mn);
    std::vector<rocke_value_t*> new_accs(accs_in);
    vp_plan_t plan;
    std::vector<rocke_value_t*> na_(mm, (rocke_value_t*)NULL), nb_(mn, (rocke_value_t*)NULL);
    int reads_per_atom = mm + mn;
    std::vector<int> free_;
    /* vp_release_atoms R (single LDS buffer, not the last tile) */
    int R = s->single ? (tr->vp_release_atoms ? tr->vp_release_atoms : 1) : 1;
    if(nxt == NULL)
        R = 1;
    if(R > 1)
    {
        int RA = KA - R - 1;
        std::vector<vp_thunk_t> early;
        for(int kk = 0; kk < KA - 1; ++kk)
        {
            auto& at = atoms[kk + 1];
            at.first.assign(mm, nullptr);
            at.second.assign(mn, nullptr);
            std::vector<vp_thunk_t> rth = vp_frag_thunks(s, kk + 1, par, &at.first, &at.second);
            if(kk < RA)
            {
                std::vector<int> rs;
                for(int j = 0; j < (int)rth.size(); ++j)
                    rs.push_back(kk * MPA + (j * (MPA / 2)) / reads_per_atom);
                for(size_t j = 0; j < rth.size(); ++j)
                    vp_put(s, rs[j], rth[j], &plan);
                std::set<int> rset(rs.begin(), rs.end());
                for(int j = 0; j < MPA; ++j)
                    if(!rset.count(kk * MPA + j))
                        free_.push_back(kk * MPA + j);
            }
            else
                early.insert(early.end(), rth.begin(), rth.end());
        }
        /* every hosted read in the first 3/4 of atom RA (in order) */
        std::vector<int> rs;
        for(int j = 0; j < (int)early.size(); ++j)
            rs.push_back(RA * MPA + (j * std::max(1, (3 * MPA) / 4)) / (int)early.size());
        for(size_t j = 0; j < early.size(); ++j)
            vp_put(s, rs[j], early[j], &plan);
        std::set<int> rset(rs.begin(), rs.end());
        for(int j = 0; j < MPA; ++j)
            if(!rset.count(RA * MPA + j))
                free_.push_back(RA * MPA + j);
    }
    else
    {
        for(int kk = 0; kk < KA - 1; ++kk)
        {
            auto& at = atoms[kk + 1];
            at.first.assign(mm, nullptr);
            at.second.assign(mn, nullptr);
            std::vector<vp_thunk_t> rth = vp_frag_thunks(s, kk + 1, par, &at.first, &at.second);
            std::vector<int> rs;
            for(int j = 0; j < (int)rth.size(); ++j)
                rs.push_back(kk * MPA + (j * (MPA / 2)) / reads_per_atom);
            for(size_t j = 0; j < rth.size(); ++j)
                vp_put(s, rs[j], rth[j], &plan);
            std::set<int> rset(rs.begin(), rs.end());
            for(int j = 0; j < MPA; ++j)
                if(!rset.count(kk * MPA + j))
                    free_.push_back(kk * MPA + j);
        }
    }
    regs_out->clear();
    int last0 = (KA - 1) * MPA;
    /* first slot of the release window (the last atom when R = 1) */
    int lastR = (KA - R) * MPA;
    if(nxt != NULL && !s->single)
    {
        std::vector<vp_thunk_t> items = vp_write_thunks(ctx, nxt->regs, nxt->npar);
        std::vector<vp_thunk_t> rth = vp_read_thunks(ctx, nxt->k_pf, regs_out);
        items.insert(items.end(), rth.begin(), rth.end());
        vp_spread(s, items, free_, &plan);
        std::vector<vp_thunk_t> lth = vp_frag_thunks(s, 0, nxt->npar, &na_, &nb_);
        vp_spread(s, lth, vp_range(last0, last0 + std::max(1, MPA / 2)), &plan);
    }
    else if(nxt != NULL)
    {
        /* single LDS buffer: the global reads in the free slots; the LDS write of
         * tile i + 1 in the last atom(s) after the first barrier, then the second
         * barrier, then the next tile's first-atom fragment reads */
        bool late = tr->vp_late_reads != 0;
        bool pair = tr->vp_pair_reads != 0;
        if(!late && !pair)
            vp_spread(s, vp_read_thunks(ctx, nxt->k_pf, regs_out), free_, &plan);
        std::vector<vp_thunk_t> wth = vp_write_thunks(ctx, nxt->regs, vp_par_imm(0));
        int nw = (int)wth.size();
        int wslots = std::max(1, std::min(nw, R == 1 ? MPA / 2 : R * MPA - MPA / 2));
        int tp = tr->vp_tail_pack;
        if(pair)
        {
            std::vector<vp_thunk_t> rth_ = vp_read_thunks(ctx, nxt->k_pf, regs_out);
            int nrd = (int)rth_.size();
            int per = nw / std::max(1, nrd);
            if(per * nrd != nw)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "vp_pair_reads: uneven write groups");
                return false;
            }
            wslots = std::max(1, R * MPA - MPA / 2);
            std::vector<vp_thunk_t> groups;
            if(tr->vp_pair_reads == 2)
            {
                /* one slot per vector: its writes and its load together */
                for(int j = 0; j < nrd; ++j)
                {
                    std::vector<vp_thunk_t> g(wth.begin() + j * per, wth.begin() + (j + 1) * per);
                    g.push_back(rth_[j]);
                    groups.push_back([g]() {
                        for(const auto& f : g)
                            f();
                    });
                }
            }
            else
            {
                /* one instruction per MFMA slot: writes of vector j, its load */
                for(int j = 0; j < nrd; ++j)
                {
                    groups.insert(groups.end(), wth.begin() + j * per, wth.begin() + (j + 1) * per);
                    groups.push_back(rth_[j]);
                }
            }
            vp_spread(s, groups, vp_range(lastR, lastR + wslots), &plan);
        }
        else if(tp == 2)
        {
            /* writes packed, then a gap before the second barrier */
            int d = std::max(2, (nw + wslots - 1) / wslots);
            for(int j = 0; j < nw; ++j)
                vp_put(s, lastR + j / d, wth[j], &plan);
            wslots = std::max(1, std::min(wslots, (nw + d - 1) / d + std::max(1, MPA / 8)));
        }
        else
            vp_spread(s, wth, vp_range(lastR, lastR + wslots), &plan);

        int b2 = std::min(lastR + wslots, s->last_slot);
        vp_put(
            s,
            b2,
            [b]() {
                rocke_b_s_waitcnt(b, -1, 0, -1);
                rocke_b_s_barrier_bare(b);
            },
            &plan);
        std::vector<vp_thunk_t> lth = vp_frag_thunks(s, 0, vp_par_imm(0), &na_, &nb_);
        if(late)
        {
            /* vp_late_reads: the global reads of tile i + 2 behind the LDS writes */
            std::vector<vp_thunk_t> rth = vp_read_thunks(ctx, nxt->k_pf, regs_out);
            lth.insert(lth.end(), rth.begin(), rth.end());
        }
        if(tp)
        {
            int td = tp == 3 ? 1 : 2;
            for(int j = 0; j < (int)lth.size(); ++j)
                vp_put(s, b2 + j / td, lth[j], &plan);
        }
        else
            vp_spread(s, lth, vp_range(b2, std::max(b2 + 1, last0 + MPA - 1)), &plan);
    }
    int vmf = tr->vp_mfma_first;
    for(int kk = 0; kk < KA; ++kk)
    {
        if(kk == KA - R && nxt != NULL)
        {
            rocke_b_s_waitcnt(b, -1, 0, -1);
            rocke_b_s_barrier_bare(b);
            rocke_b_sched_barrier(b, 0);
        }
        auto& at = atoms[kk];
        for(int mi = 0; mi < mm; ++mi)
        {
            for(int ni = 0; ni < mn; ++ni)
            {
                int s_ = kk * MPA + mi * mn + ni;
                int flat = mi * mn + ni;
                new_accs[flat]
                    = rocke_gemm_mma_ab(ctx, at.first[mi], at.second[ni], new_accs[flat], -1);
                if(vmf == 2 || (vmf == 1 && s_ == 0))
                    rocke_b_sched_barrier(b, 0);
                auto it = plan.find(s_);
                if(it != plan.end())
                {
                    std::vector<vp_thunk_t> ths = std::move(it->second);
                    plan.erase(it);
                    for(const auto& th : ths)
                        th();
                }
                rocke_b_sched_barrier(b, 0);
            }
        }
    }
    if(!plan.empty())
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "vgpr_prefetch: unplaced slots");
        return false;
    }
    frags_out->clear();
    if(nxt != NULL)
    {
        for(rocke_value_t* v : na_)
            if(v == NULL)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "vgpr_prefetch: next-tile fragments not read");
                return false;
            }
        for(rocke_value_t* v : nb_)
            if(v == NULL)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "vgpr_prefetch: next-tile fragments not read");
                return false;
            }
        frags_out->insert(frags_out->end(), na_.begin(), na_.end());
        frags_out->insert(frags_out->end(), nb_.begin(), nb_.end());
    }
    *accs_out = new_accs;
    return true;
}

struct vp_loop
{
    vp_ilv* s;
    rocke_value_t* c1;
    rocke_value_t* c_pf_tiles;
    rocke_value_t* k_last;
    int nr;
    int ns;
    int nfr;
};

/* _vp_step(st, k0) */
bool vp_step(vp_loop* L,
             const std::vector<rocke_value_t*>& st,
             rocke_value_t* k0,
             std::vector<rocke_value_t*>* out)
{
    vp_ilv* s = L->s;
    rocke_gemm_build_ctx_t* ctx = s->ctx;
    rocke_ir_builder_t* b = s->b;
    int nr = L->nr, ns = L->ns, nfr = L->nfr;
    rocke_value_t* parity = st[0];
    vp_next nxt;
    nxt.regs.assign(st.begin() + 1, st.begin() + 1 + ns);
    std::vector<rocke_value_t*> later(st.begin() + 1 + ns, st.begin() + 1 + nr);
    std::vector<rocke_value_t*> fr(st.begin() + 1 + nr, st.begin() + 1 + nr + nfr);
    std::vector<rocke_value_t*> acc_iter(st.begin() + 1 + nr + nfr, st.end());
    rocke_value_t* next_parity = s->single ? ctx->c0 : rocke_b_sub(b, L->c1, parity);
    nxt.npar = vp_par_val(next_parity);
    nxt.k_pf = rocke_b_smin(b, rocke_b_add(b, k0, L->c_pf_tiles), L->k_last);
    std::vector<rocke_value_t*> new_accs, new_fr, new_regs;
    if(!vp_tile(s,
                s->single ? vp_par_imm(0) : vp_par_val(parity),
                fr,
                acc_iter,
                &nxt,
                &new_accs,
                &new_fr,
                &new_regs))
        return false;
    out->clear();
    out->push_back(next_parity);
    out->insert(out->end(), later.begin(), later.end());
    out->insert(out->end(), new_regs.begin(), new_regs.end());
    out->insert(out->end(), new_fr.begin(), new_fr.end());
    out->insert(out->end(), new_accs.begin(), new_accs.end());
    return true;
}

} // namespace

extern "C" void rocke_gemm_emit_kloop_vpf_ilv(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    vp_ilv s;
    s.ctx = ctx;
    s.b = b;
    s.KA = ctx->k_atoms;
    s.MPA = ctx->mfmas_m * ctx->mfmas_n;
    /* vgpr_prefetch 4: one LDS buffer (parity always 0, two barriers / tile) */
    s.single = ctx->vpf == 4;
    rocke_value_t* c1_i32 = rocke_b_const_i32(b, 1);
    rocke_value_t* k_last = rocke_b_sub(b, vp_k_end(ctx), ctx->c_block_k);
    rocke_value_t* c_pf_tiles = rocke_b_const_i32(b, (ctx->vpd + 1) * ctx->block_k);
    s.m_in_atom = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_m));
    rocke_value_t* k_blk = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_m));
    s.n_in_atom = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_n));
    s.warp_m_off
        = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, ctx->mfmas_m * t->warp_tile_m));
    s.warp_n_off
        = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, ctx->mfmas_n * t->warp_tile_n));
    s.k_blk_kbase = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, ctx->a_per_lane * ctx->kmul));
    s.last_slot = s.KA * s.MPA - 1;

    std::vector<rocke_value_t*> regs = vp_prologue(ctx, k_last);
    std::vector<rocke_value_t*> fa0(ctx->mfmas_m, (rocke_value_t*)NULL);
    std::vector<rocke_value_t*> fb0(ctx->mfmas_n, (rocke_value_t*)NULL);
    for(const auto& th : vp_frag_thunks(&s, 0, vp_par_imm(0), &fa0, &fb0))
        th();
    std::vector<rocke_value_t*> frags(fa0);
    frags.insert(frags.end(), fb0.begin(), fb0.end());
    vp_loop L;
    L.s = &s;
    L.c1 = c1_i32;
    L.c_pf_tiles = c_pf_tiles;
    L.k_last = k_last;
    L.nr = (int)regs.size();
    L.nfr = (int)frags.size();
    L.ns = L.nr / ctx->vpd;
    std::vector<std::string> names;
    std::vector<rocke_iter_arg_t> loop_args;
    {
        rocke_iter_arg_t a;
        a.name = "par";
        a.init = ctx->c0;
        loop_args.push_back(a);
        names.push_back("par");
        for(int i = 0; i < L.nr; ++i)
        {
            names.push_back("vpr" + std::to_string(i));
            a.name = vp_name(b, names.back());
            a.init = regs[i];
            loop_args.push_back(a);
        }
        for(int i = 0; i < L.nfr; ++i)
        {
            names.push_back("vpf" + std::to_string(i));
            a.name = vp_name(b, names.back());
            a.init = frags[i];
            loop_args.push_back(a);
        }
        for(int i = 0; i < ctx->num_accs; ++i)
        {
            names.push_back(ctx->acc_names[i]);
            a.name = ctx->acc_names[i];
            a.init = ctx->acc_inits[i];
            loop_args.push_back(a);
        }
    }

    /* vp_unroll: D + 1 tiles per loop iteration (no loop-carried register
     * copies); the T % (D + 1) remaining tiles run in 0/1-trip loops */
    int U = ctx->spec->trait.vp_unroll ? ctx->vpd + 1 : 1;
    std::vector<rocke_value_t*> state;
    if(U == 1)
    {
        rocke_for_t for_op = rocke_b_scf_for_iter(b,
                                                  ctx->k_lo,
                                                  k_last,
                                                  ctx->c_block_k,
                                                  loop_args.data(),
                                                  (int)loop_args.size(),
                                                  "k0",
                                                  false,
                                                  true);
        rocke_b_region_enter(b, for_op.body);
        {
            std::vector<rocke_value_t*> st(for_op.iter_vars,
                                           for_op.iter_vars + for_op.num_iter_vars);
            std::vector<rocke_value_t*> y;
            if(!vp_step(&L, st, for_op.iv, &y))
            {
                rocke_b_region_leave(b);
                return;
            }
            rocke_b_scf_yield(b, y.data(), (int)y.size());
        }
        rocke_b_region_leave(b);
        state.assign(for_op.op->results, for_op.op->results + for_op.op->num_results);
    }
    else
    {
        rocke_value_t* c_ub = rocke_b_const_i32(b, U * ctx->block_k);
        rocke_value_t* n_t = rocke_b_div(b, rocke_b_sub(b, k_last, ctx->k_lo), ctx->c_block_k);
        rocke_value_t* k_main = rocke_b_add(
            b, ctx->k_lo, rocke_b_mul(b, rocke_b_div(b, n_t, rocke_b_const_i32(b, U)), c_ub));
        rocke_for_t for_op = rocke_b_scf_for_iter(
            b, ctx->k_lo, k_main, c_ub, loop_args.data(), (int)loop_args.size(), "k0", false, true);
        rocke_b_region_enter(b, for_op.body);
        {
            std::vector<rocke_value_t*> st(for_op.iter_vars,
                                           for_op.iter_vars + for_op.num_iter_vars);
            for(int u = 0; u < U; ++u)
            {
                rocke_value_t* ku
                    = u ? rocke_b_add(b, for_op.iv, rocke_b_const_i32(b, u * ctx->block_k))
                        : for_op.iv;
                std::vector<rocke_value_t*> nst;
                if(!vp_step(&L, st, ku, &nst))
                {
                    rocke_b_region_leave(b);
                    return;
                }
                st = nst;
            }
            rocke_b_scf_yield(b, st.data(), (int)st.size());
        }
        rocke_b_region_leave(b);
        state.assign(for_op.op->results, for_op.op->results + for_op.op->num_results);
        for(int r = 0; r < U - 1; ++r)
        {
            rocke_value_t* lo_r
                = r ? rocke_b_add(b, k_main, rocke_b_const_i32(b, r * ctx->block_k)) : k_main;
            rocke_value_t* hi_r = rocke_b_smin(b, rocke_b_add(b, lo_r, ctx->c_block_k), k_last);
            std::vector<rocke_iter_arg_t> rargs;
            for(size_t i = 0; i < names.size(); ++i)
            {
                rocke_iter_arg_t a;
                a.name = vp_name(b, "vu" + std::to_string(r) + "_" + names[i]);
                a.init = state[i];
                rargs.push_back(a);
            }
            std::string ivn = "kvu" + std::to_string(r);
            rocke_for_t rem = rocke_b_scf_for_iter(b,
                                                   lo_r,
                                                   hi_r,
                                                   ctx->c_block_k,
                                                   rargs.data(),
                                                   (int)rargs.size(),
                                                   vp_name(b, ivn),
                                                   false,
                                                   true);
            rocke_b_region_enter(b, rem.body);
            {
                std::vector<rocke_value_t*> st(rem.iter_vars, rem.iter_vars + rem.num_iter_vars);
                std::vector<rocke_value_t*> y;
                if(!vp_step(&L, st, rem.iv, &y))
                {
                    rocke_b_region_leave(b);
                    return;
                }
                rocke_b_scf_yield(b, y.data(), (int)y.size());
            }
            rocke_b_region_leave(b);
            state.assign(rem.op->results, rem.op->results + rem.op->num_results);
        }
    }

    vp_par final_parity = s.single ? vp_par_imm(0) : vp_par_val(state[0]);
    std::vector<rocke_value_t*> fr(state.begin() + 1 + L.nr, state.begin() + 1 + L.nr + L.nfr);
    std::vector<rocke_value_t*> accs(state.begin() + 1 + L.nr + L.nfr, state.end());
    std::vector<rocke_value_t*> out_accs, out_fr, out_regs;
    if(!vp_tile(&s, final_parity, fr, accs, NULL, &out_accs, &out_fr, &out_regs))
        return;
    for(size_t i = 0; i < out_accs.size(); ++i)
        ctx->for_results[i] = out_accs[i];
    ctx->num_for_results = (int)out_accs.size();
}
