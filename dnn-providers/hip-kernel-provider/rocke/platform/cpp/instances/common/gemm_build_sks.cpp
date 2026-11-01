// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/* gemm_build_sks.cpp -- C++ mirror of gemm_universal.py `_ep_split_k_serial` /
 * `_ep_split_k_ticket`: the in-launch split-K reduction (TraitSpec.split_k_serial)
 * of the vgpr_prefetch body and the direct_to_lds ring.
 *
 *   2: every wave stores its f32 partial with write-through (sc1) global stores
 *      into its slot of W, bumps its (tile, wave) counter; the wave that draws
 *      ticket split_k - 1 sums the split_k partials in z order (write-through
 *      loads), resets the counter and runs the default epilogue;
 *   3: the atomic split-K scatter into W [M, N], the same ticket; the last wave
 *      reads its cells back (sc1), re-zeroes them and runs the default epilogue.
 * Mode 1 (fenced partials) is refused by the validity gate (as in Python, whose
 * SPLIT_K_SERIAL_FENCED_VALID is off), so it has no C++ emission.
 *
 * Every IR op is emitted in the Python order (the byte-identity contract);
 * nested builder calls with more than one emitting argument are split into
 * temporaries. */

#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/instance_gemm_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct sks_state
{
    rocke_gemm_build_ctx_t* ctx;
    rocke_ir_builder_t* b;
    rocke_value_t* c_nw;
    rocke_value_t* tiles_;
    rocke_value_t* tile_;
    rocke_value_t* wave_;
    rocke_value_t* w64;
    rocke_value_t* lane_b;
    int nq;
    long long slot_b;
};

rocke_value_t* sks_asm0(rocke_ir_builder_t* b, rocke_value_t* r, const char* cons)
{
    /* inline_asm("", cons, [r], r.type, sideeffect=True) */
    rocke_inline_asm_opts_t opts;
    memset(&opts, 0, sizeof opts);
    opts.sideeffect = true;
    opts.sideeffect_set = true;
    const rocke_type_t* ty = r->type;
    rocke_op_t* op = rocke_b_inline_asm(b, "", cons, &r, 1, &ty, 1, &opts);
    return op != NULL ? op->results[0] : NULL;
}

/* _addr(z, ai, q): the 64-bit lane address of partial (z, ai, q) */
rocke_value_t* sks_addr(sks_state* s, rocke_value_t* z, int ai, int q)
{
    rocke_ir_builder_t* b = s->b;
    const rocke_type_t* I64 = rocke_i64();
    int ws = s->ctx->spec->wave_size;
    rocke_value_t* sl = rocke_b_add(
        b,
        rocke_b_mul(b, rocke_b_add(b, rocke_b_mul(b, z, s->tiles_), s->tile_), s->c_nw),
        s->wave_);
    rocke_value_t* zs = rocke_b_zext(b, sl, I64);
    rocke_value_t* cs = rocke_b_const_i64(b, s->slot_b);
    rocke_value_t* m = rocke_b_mul(b, zs, cs);
    rocke_value_t* off
        = rocke_b_add(b, m, rocke_b_const_i64(b, (long long)(ai * s->nq + q) * ws * 16));
    return rocke_b_add(b, s->w64, rocke_b_add(b, off, s->lane_b));
}

/* _quads(v) */
std::vector<rocke_value_t*> sks_quads(sks_state* s, rocke_value_t* v)
{
    rocke_ir_builder_t* b = s->b;
    int cpl = s->ctx->c_per_lane;
    std::vector<rocke_value_t*> out;
    if(cpl == 4)
    {
        out.push_back(v);
        return out;
    }
    for(int q = 0; q < s->nq; ++q)
    {
        rocke_value_t* comps[4];
        for(int i = 0; i < 4; ++i)
            comps[i] = rocke_b_vec_extract(b, v, 4 * q + i);
        out.push_back(rocke_b_vec_pack(b, comps, 4, rocke_f32()));
    }
    return out;
}

/* _ep_split_k_ticket(vals): split_k_serial 3 */
void sks_ticket(rocke_gemm_build_ctx_t* ctx, rocke_value_t* const* vals, int num_vals)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    const rocke_gemm_tile_spec_t* tl_ = &spec->tile;
    int n_waves = tl_->warp_m * tl_->warp_n * tl_->warp_k;
    int cpl = ctx->c_per_lane;
    rocke_value_t* c_nw = rocke_b_const_i32(b, n_waves);
    rocke_value_t* npn = rocke_b_div(
        b, rocke_b_add(b, ctx->N, rocke_b_const_i32(b, ctx->block_n - 1)), ctx->c_block_n);
    rocke_value_t* tm = rocke_b_mul(b, rocke_b_div(b, ctx->block_m_off, ctx->c_block_m), npn);
    rocke_value_t* tn = rocke_b_div(b, ctx->block_n_off, ctx->c_block_n);
    rocke_value_t* tile_ = rocke_b_to_sgpr_u32(b, rocke_b_add(b, tm, tn));
    rocke_value_t* cm = rocke_b_mul(b, tile_, c_nw);
    rocke_value_t* cnt_idx = rocke_b_add(b, cm, rocke_b_to_sgpr_u32(b, ctx->warp_id));
    rocke_gemm_emit_epilogue_split_k(b,
                                     spec,
                                     vals,
                                     num_vals,
                                     ctx->warp_m_idx,
                                     ctx->warp_n_idx,
                                     ctx->lane,
                                     ctx->block_m_off,
                                     ctx->block_n_off,
                                     ctx->M,
                                     ctx->N,
                                     ctx->sks_w,
                                     cpl);
    /* the wave's adds are performed before its ticket */
    rocke_b_inline_asm(b, "s_waitcnt vmcnt(0)", "", NULL, 0, NULL, 0, NULL);
    rocke_value_t* is0 = rocke_b_cmp_eq(b, ctx->lane, ctx->c0);
    rocke_value_t* one = rocke_b_select(b, is0, rocke_b_const_i32(b, 1), ctx->c0);
    rocke_value_t* old = rocke_b_global_atomic_add(b, ctx->sks_cnt, cnt_idx, one, "monotonic");
    rocke_value_t* ticket = rocke_b_readfirstlane(b, old);
    rocke_if_t iff
        = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ticket, rocke_b_const_i32(b, ctx->split_k - 1)));
    rocke_b_region_enter(b, iff.then_region);
    {
        /* the cells of this wave (the atomic scatter's coordinates) */
        int mfmas_m = ctx->mfmas_m, mfmas_n = ctx->mfmas_n;
        rocke_value_t* wm_off
            = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, mfmas_m * tl_->warp_tile_m));
        rocke_value_t* wn_off
            = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, mfmas_n * tl_->warp_tile_n));
        rocke_value_t* bwm = rocke_b_add(b, ctx->block_m_off, wm_off);
        rocke_value_t* bwn = rocke_b_add(b, ctx->block_n_off, wn_off);
        const rocke_mfma_atom_t* atom = rocke_b_mfma_atom(
            b, spec->data.dtype_a, tl_->warp_tile_m, tl_->warp_tile_n, tl_->warp_tile_k);
        if(!rocke_ir_builder_ok(b))
        {
            rocke_b_region_leave(b);
            return;
        }
        int kc_m1 = 0, kc_nlane = 0;
        if(rocke_b_c_warp_params(b, atom, NULL, NULL, &kc_m1, &kc_nlane) != ROCKE_OK)
        {
            rocke_b_region_leave(b);
            return;
        }
        rocke_tile_distribution_encoding_t* c_enc = rocke_make_c_warp_dstr_encoding(b, atom);
        rocke_tile_distribution_t* c_dist = rocke_make_static_tile_distribution(b, c_enc);
        if(!rocke_ir_builder_ok(b))
        {
            rocke_b_region_leave(b);
            return;
        }
        rocke_value_t* c_nlane = rocke_b_const_i32(b, kc_nlane);
        rocke_value_t* pl0 = rocke_b_div(b, ctx->lane, c_nlane);
        rocke_value_t* pl1 = rocke_b_mod(b, ctx->lane, c_nlane);
        rocke_value_t* p_lane[2] = {pl0, pl1};
        std::vector<rocke_value_t*> rows_;
        rocke_value_t* col_ = NULL;
        {
            rocke_value_t* const* ps[1] = {p_lane};
            int ps_counts[1] = {2};
            for(int i = 0; i < cpl; ++i)
            {
                rocke_value_t* ys[2]
                    = {rocke_b_const_i32(b, i / kc_m1), rocke_b_const_i32(b, i % kc_m1)};
                rocke_value_t* x_out[2] = {NULL, NULL};
                if(!rocke_tile_distribution_calculate_x(
                       b, c_dist, ys, 2, ps, ps_counts, 1, x_out, 2))
                {
                    rocke_b_region_leave(b);
                    return;
                }
                rows_.push_back(x_out[0]);
                col_ = x_out[1];
            }
        }
        rocke_value_t* m_last = rocke_b_sub(b, ctx->M, rocke_b_const_i32(b, 1));
        rocke_value_t* n_last = rocke_b_sub(b, ctx->N, rocke_b_const_i32(b, 1));
        rocke_value_t* w64 = rocke_b_global_ptr_to_i64(b, ctx->sks_w);
        struct cell
        {
            int f;
            int i;
            rocke_value_t* c_m;
            rocke_value_t* c_n;
            rocke_value_t* off;
        };
        std::vector<cell> cells;
        int flat = 0;
        for(int mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* base_m
                = rocke_b_add(b, bwm, rocke_b_const_i32(b, mi * tl_->warp_tile_m));
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* c_n = rocke_b_add(
                    b, bwn, rocke_b_add(b, rocke_b_const_i32(b, ni * tl_->warp_tile_n), col_));
                rocke_value_t* cn_c = rocke_b_smin(b, c_n, n_last);
                for(int i = 0; i < cpl; ++i)
                {
                    rocke_value_t* c_m = rocke_b_add(b, base_m, rows_[i]);
                    rocke_value_t* mm = rocke_b_mul(b, rocke_b_smin(b, c_m, m_last), ctx->N);
                    rocke_value_t* off = rocke_b_add(b, mm, cn_c);
                    cells.push_back({flat, i, c_m, c_n, off});
                }
                flat += 1;
            }
        }
        std::map<std::pair<int, int>, rocke_value_t*> got;
        for(size_t g0 = 0; g0 < cells.size(); g0 += 16)
        {
            size_t n_ = std::min<size_t>(16, cells.size() - g0);
            std::string tmpl;
            for(size_t j = 0; j < n_; ++j)
            {
                tmpl += "global_load_dword $" + std::to_string(j) + ", $" + std::to_string(n_ + j)
                        + ", off sc1\n";
            }
            tmpl += "s_waitcnt vmcnt(0)";
            std::string cons;
            for(size_t j = 0; j < 2 * n_; ++j)
                cons += std::string(j ? "," : "") + (j < n_ ? "=&v" : "v");
            std::vector<rocke_value_t*> addrs;
            for(size_t j = 0; j < n_; ++j)
            {
                rocke_value_t* z = rocke_b_zext(b, cells[g0 + j].off, rocke_i64());
                rocke_value_t* m4 = rocke_b_mul(b, z, rocke_b_const_i64(b, 4));
                addrs.push_back(rocke_b_add(b, w64, m4));
            }
            std::vector<const rocke_type_t*> rt(n_, rocke_f32());
            rocke_op_t* op = rocke_b_inline_asm_multi(
                b, tmpl.c_str(), cons.c_str(), addrs.data(), (int)n_, rt.data(), (int)n_, NULL);
            if(op == NULL)
            {
                rocke_b_region_leave(b);
                return;
            }
            for(size_t j = 0; j < n_; ++j)
                got[{cells[g0 + j].f, cells[g0 + j].i}] = op->results[j];
        }
        std::vector<rocke_value_t*> out;
        for(int f_ = 0; f_ < num_vals; ++f_)
        {
            std::vector<rocke_value_t*> comps;
            for(int i = 0; i < cpl; ++i)
                comps.push_back(got[{f_, i}]);
            rocke_value_t* v_ = rocke_b_vec_pack(b, comps.data(), cpl, rocke_f32());
            out.push_back(spec->trait.acc_agpr ? sks_asm0(b, v_, "=a,0") : v_);
        }
        /* zero this wave's in-bounds cells for the next launch */
        rocke_value_t* z_ = rocke_b_const_f32(b, 0.0);
        bool pad_m = spec->trait.pad_m, pad_n = spec->trait.pad_n;
        for(const cell& c : cells)
        {
            rocke_value_t* chk[2];
            int nchk = 0;
            if(pad_m)
                chk[nchk++] = rocke_b_cmp_lt(b, c.c_m, ctx->M);
            if(pad_n)
                chk[nchk++] = rocke_b_cmp_lt(b, c.c_n, ctx->N);
            if(nchk)
            {
                rocke_if_t zi
                    = rocke_b_scf_if(b, nchk == 1 ? chk[0] : rocke_b_land(b, chk[0], chk[1]));
                rocke_b_region_enter(b, zi.then_region);
                rocke_b_global_store(b, ctx->sks_w, c.off, z_, 4);
                rocke_b_region_leave(b);
            }
            else
                rocke_b_global_store(b, ctx->sks_w, c.off, z_, 4);
        }
        rocke_b_global_store(b, ctx->sks_cnt, cnt_idx, ctx->c0, 4);
        rocke_gemm_ep_default(ctx,
                              out.data(),
                              (int)out.size(),
                              ctx->block_m_off,
                              ctx->block_n_off,
                              NULL,
                              NULL,
                              false);
    }
    rocke_b_region_leave(b);
}

} // namespace

/* _ep_split_k_serial(vals) */
extern "C" void rocke_gemm_ep_split_k_serial(rocke_gemm_build_ctx_t* ctx,
                                             rocke_value_t* const* vals,
                                             int num_vals)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    int mode = spec->trait.split_k_serial;
    if(mode == 3)
    {
        sks_ticket(ctx, vals, num_vals);
        return;
    }
    if(mode != 2)
    {
        /* the fenced form (1) is refused by is_valid_spec */
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "split_k_serial %d has no C++ emission", mode);
        return;
    }
    const rocke_gemm_tile_spec_t* tl_ = &spec->tile;
    int n_waves = tl_->warp_m * tl_->warp_n * tl_->warp_k;
    int cpl = ctx->c_per_lane;
    sks_state s;
    s.ctx = ctx;
    s.b = b;
    s.c_nw = rocke_b_const_i32(b, n_waves);
    rocke_value_t* c1_ = rocke_b_const_i32(b, 1);
    rocke_value_t* npm = rocke_b_div(
        b, rocke_b_add(b, ctx->M, rocke_b_const_i32(b, ctx->block_m - 1)), ctx->c_block_m);
    rocke_value_t* npn = rocke_b_div(
        b, rocke_b_add(b, ctx->N, rocke_b_const_i32(b, ctx->block_n - 1)), ctx->c_block_n);
    s.tiles_ = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, npm, npn));
    rocke_value_t* tm = rocke_b_mul(b, rocke_b_div(b, ctx->block_m_off, ctx->c_block_m), npn);
    rocke_value_t* tn = rocke_b_div(b, ctx->block_n_off, ctx->c_block_n);
    s.tile_ = rocke_b_to_sgpr_u32(b, rocke_b_add(b, tm, tn));
    s.wave_ = rocke_b_to_sgpr_u32(b, ctx->warp_id);
    rocke_value_t* cnt_idx = rocke_b_add(b, rocke_b_mul(b, s.tile_, s.c_nw), s.wave_);
    s.nq = cpl % 4 == 0 ? cpl / 4 : 0;
    if(!s.nq)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "split_k_serial needs c_per_lane %% 4 == 0");
        return;
    }
    s.slot_b = (long long)num_vals * cpl * spec->wave_size * 4;
    /* voff = lane * 16 (the buffer form of the fenced mode; emitted as Python does) */
    rocke_b_mul(b, ctx->lane, rocke_b_const_i32(b, 16));
    /* write-through mode: 64-bit lane addresses, sc1 global stores / loads in
     * inline-asm groups ending with vmcnt(0) */
    s.w64 = rocke_b_global_ptr_to_i64(b, ctx->sks_w);
    s.lane_b = rocke_b_zext(b, rocke_b_mul(b, ctx->lane, rocke_b_const_i32(b, 16)), rocke_i64());
    rocke_value_t* z_me = rocke_b_to_sgpr_u32(b, rocke_b_block_id_z(b));
    struct stq_t
    {
        int ai;
        int q;
        rocke_value_t* v4;
    };
    std::vector<stq_t> stq;
    for(int ai = 0; ai < num_vals; ++ai)
    {
        std::vector<rocke_value_t*> qs = sks_quads(&s, rocke_gemm_acc_agpr_out(b, spec, vals[ai]));
        for(int q = 0; q < (int)qs.size(); ++q)
            stq.push_back({ai, q, qs[q]});
    }
    for(size_t g0 = 0; g0 < stq.size(); g0 += 8)
    {
        size_t n = std::min<size_t>(8, stq.size() - g0);
        std::string tmpl;
        std::vector<rocke_value_t*> ops;
        for(size_t j = 0; j < n; ++j)
        {
            tmpl += "global_store_dwordx4 $" + std::to_string(2 * j) + ", $"
                    + std::to_string(2 * j + 1) + ", off sc1\n";
            ops.push_back(sks_addr(&s, z_me, stq[g0 + j].ai, stq[g0 + j].q));
            ops.push_back(stq[g0 + j].v4);
        }
        tmpl += "s_waitcnt vmcnt(0)";
        std::string cons;
        for(size_t j = 0; j < ops.size(); ++j)
            cons += j ? ",v" : "v";
        rocke_b_inline_asm(
            b, tmpl.c_str(), cons.c_str(), ops.data(), (int)ops.size(), NULL, 0, NULL);
    }
    rocke_value_t* is0 = rocke_b_cmp_eq(b, ctx->lane, ctx->c0);
    rocke_value_t* one = rocke_b_select(b, is0, c1_, ctx->c0);
    rocke_value_t* old = rocke_b_global_atomic_add(b, ctx->sks_cnt, cnt_idx, one, "monotonic");
    rocke_value_t* ticket = rocke_b_readfirstlane(b, old);
    rocke_if_t iff
        = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ticket, rocke_b_const_i32(b, ctx->split_k - 1)));
    rocke_b_region_enter(b, iff.then_region);
    {
        std::vector<rocke_value_t*> out(vals, vals + num_vals);
        int grp = std::max(1, 16 / cpl);
        const rocke_type_t* v4t = rocke_vector_type(b, rocke_f32(), 4);
        for(int g0 = 0; g0 < num_vals; g0 += grp)
        {
            std::vector<int> idx;
            for(int i = g0; i < std::min(g0 + grp, num_vals); ++i)
                idx.push_back(i);
            /* static unroll: up to 16 write-through loads in flight per asm
             * group, summed in z order (z = 0 .. split_k - 1) */
            std::map<int, rocke_value_t*> sums;
            struct lq_t
            {
                int z, i, q;
            };
            std::vector<lq_t> lq;
            for(int z = 0; z < ctx->split_k; ++z)
                for(int i : idx)
                    for(int q = 0; q < s.nq; ++q)
                        lq.push_back({z, i, q});
            for(size_t c0_ = 0; c0_ < lq.size(); c0_ += 16)
            {
                size_t n_ = std::min<size_t>(16, lq.size() - c0_);
                std::string tmpl;
                for(size_t j = 0; j < n_; ++j)
                    tmpl += "global_load_dwordx4 $" + std::to_string(j) + ", $"
                            + std::to_string(n_ + j) + ", off sc1\n";
                tmpl += "s_waitcnt vmcnt(0)";
                std::string cons;
                for(size_t j = 0; j < 2 * n_; ++j)
                    cons += std::string(j ? "," : "") + (j < n_ ? "=&v" : "v");
                std::vector<rocke_value_t*> addrs;
                for(size_t j = 0; j < n_; ++j)
                {
                    const lq_t& e = lq[c0_ + j];
                    rocke_value_t* cz = rocke_b_const_i32(b, e.z);
                    addrs.push_back(sks_addr(&s, cz, e.i, e.q));
                }
                std::vector<const rocke_type_t*> rt(n_, v4t);
                rocke_op_t* op = rocke_b_inline_asm_multi(
                    b, tmpl.c_str(), cons.c_str(), addrs.data(), (int)n_, rt.data(), (int)n_, NULL);
                if(op == NULL)
                {
                    rocke_b_region_leave(b);
                    return;
                }
                /* got: (z, i) -> {q: r} in first-seen order */
                std::vector<std::pair<std::pair<int, int>, std::map<int, rocke_value_t*>>> got;
                for(size_t j = 0; j < n_; ++j)
                {
                    const lq_t& e = lq[c0_ + j];
                    std::pair<int, int> key(e.z, e.i);
                    size_t k = 0;
                    while(k < got.size() && got[k].first != key)
                        ++k;
                    if(k == got.size())
                        got.push_back({key, {}});
                    got[k].second[e.q] = op->results[j];
                }
                for(auto& g : got)
                {
                    int i = g.first.second;
                    rocke_value_t* v_;
                    if(cpl == 4)
                        v_ = g.second[0];
                    else
                    {
                        std::vector<rocke_value_t*> comps;
                        for(int e = 0; e < cpl; ++e)
                            comps.push_back(rocke_b_vec_extract(b, g.second[e / 4], e % 4));
                        v_ = rocke_b_vec_pack(b, comps.data(), cpl, rocke_f32());
                    }
                    auto it = sums.find(i);
                    sums[i] = it == sums.end() ? v_ : rocke_b_vector_add(b, it->second, v_);
                }
            }
            for(int i : idx)
            {
                rocke_value_t* r = sums[i];
                out[i] = spec->trait.acc_agpr ? sks_asm0(b, r, "=a,0") : r;
            }
        }
        rocke_b_global_store(b, ctx->sks_cnt, cnt_idx, ctx->c0, 4);
        rocke_gemm_ep_default(ctx,
                              out.data(),
                              (int)out.size(),
                              ctx->block_m_off,
                              ctx->block_n_off,
                              NULL,
                              NULL,
                              false);
    }
    rocke_b_region_leave(b);
}
