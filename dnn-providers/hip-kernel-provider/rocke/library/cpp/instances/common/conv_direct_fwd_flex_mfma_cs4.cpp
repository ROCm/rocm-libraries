// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_flex_mfma_cs4.cpp -- C++ engine port of _build_cs4 in
 * library/kernels/common/conv_direct_fwd_flex_mfma.py: the cs 4 flex forward
 * (X128 x K128 tiles, the reduction split four ways inside the block).
 *
 * Group g (waves 4g..4g+3) runs chunks ch0 + g, ch0 + g + 2, ... of the item's
 * piece through its own two LDS stages, the taps of a chunk unrolled with a
 * DMA look-ahead of two steps; wave (xi, sl) of a group reads pixels 64 xi..
 * of the stage and the 32-channel half sl. A tile is drained at its end: the
 * four (group, half) partial sums of each pixel half meet in the stage memory
 * and one pass sums, converts and stores them (or writes the fp32 slab of a
 * split-C piece).
 *
 * Every IR op is emitted in the order the Python builder emits it; see the
 * contract in rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include <stdint.h>
#include <string.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "rocke/instance_conv_direct_fwd_flex_mfma_internal.hpp"
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */

using namespace rocke_dwg;

namespace rocke_ffx
{

namespace
{

/* A DMA: rsrc, (LDS SGPR base, immediate), voffset, soffset. */
struct Cs4Op
{
    V rsrc, base;
    int imm;
    V voff, soff;
};

/* A look-ahead slot: (live, chunk bytes, tap). */
struct Slot
{
    V ok, chb;
    int tap;
};

/* The weight DMA lane parts: non-wt v0 only; wt (v0, v1, the K tile's first
 * output in bytes). */
struct Wo
{
    V v0, v1, k0b;
};

/* A look-ahead slot's resources and chunk bytes. */
struct SlotRs
{
    V xr, wr, chb;
};

} // namespace

rocke_kernel_def_t*
    build_cs4(rocke_ir_builder_t* b, const rocke_fwd_flex_mfma_spec_t* spec, const char* arch)
{
    (void)arch;
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const rocke_type_t* i32_t = rocke_i32();
    const int BX = 128, BK = 128, SP = spec->split_c, KT = 8;
    const int NXI = 4, NWI = 4;
    const int R = p.KH, S = p.KW;
    const int SH = p.stride, PH = p.PAD, PW = rocke_direct_conv_problem_pad_w(&p);
    const int DH = p.dil_h, DW = p.dil_w;
    const int IN_E = BX * kCK; /* input elements of a group's stage */
    const int GE = (BX + BK) * kCK; /* elements of a group's stage */
    const int PART = 32 * BK * 2; /* one partial region (32 pixels x 128 fp32) */

    rocke_attr_set_int(
        b, &b->kernel->attrs, "max_workgroup_size", rocke_fwd_flex_mfma_threads_per_block(spec));
    rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", 2);
    rocke_dconv_params_t P;
    rocke_dconv_emit_params(b, &P, "fwd", io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    auto ci = [&](int64_t v) { return e.ci(v); };
    auto dv = [&](V a, int64_t n) { return e.divi(a, n); };
    auto md = [&](V a, int64_t n) { return e.modi(a, n); };

    V c0 = ci(0);
    V c1 = ci(1);
    V c2 = ci(2);
    V oob = ci(kOob);
    V p_N = P.p_N, p_Hi = P.p_Hi, p_Wi = P.p_Wi;
    V p_Ho = P.p_Ho, p_Wo = P.p_Wo;
    V C = P.p_total_c, K = P.p_total_k;

    V tid = rocke_b_thread_id_x(b);
    V lane = md(tid, kWave);
    V wave = rocke_b_readfirstlane(b, dv(tid, kWave));
    V wq = md(wave, 4);
    V xi = md(wq, 2);
    V sl = dv(wq, 2);

    auto opaque_lane = [&]() { return opaque(b, lane, "=v,0"); };

    V hw = e.mul(p_Ho, p_Wo);
    V NX = e.mul(p_N, hw);
    V nxt = e.cdivi(NX, BX);
    V nkt = e.cdivi(K, BK);
    V TI = e.muli(e.mul(nxt, nkt), SP);
    V G = e.smini(TI, kPersist);
    V nch = dv(C, kCK);
    V RSC = e.mul(ci((int64_t)R * S), C);
    V blk = rocke_b_block_id_x(b);
    if(spec->xcd)
    {
        /* any G: the first 8 * (G // 8) blocks are renumbered XCD-major, the
         * last G % 8 keep their number (a bijection of [0, G)) */
        V g8 = dv(G, 8);
        V xm = e.mul(md(blk, 8), g8);
        V xr = e.add(xm, dv(blk, 8));
        V lim = e.muli(g8, 8);
        blk = e.select(e.lt(blk, lim), xr, blk);
    }
    V full = e.div(TI, G);
    V part = e.mod(TI, G);
    V lr = e.add(blk, e.muli(full, kRot));
    V last_ok = e.lt(e.mod(lr, G), part);
    V rounds = e.add(full, e.select(last_ok, c1, c0));

    auto tile_of = [&](V i) {
        V m = e.mul(i, G);
        V r = e.add(blk, e.muli(i, kRot));
        return e.add(m, e.mod(r, G));
    };

    auto item_info = [&](V t) {
        std::map<std::string, V> out;
        V tl = dv(t, SP);
        V pc = md(t, SP);
        V tx, tk;
        if(spec->kmaj)
        {
            tk = e.div(tl, nxt);
            tx = e.mod(tl, nxt);
        }
        else
        {
            tx = e.div(tl, nkt);
            tk = e.mod(tl, nkt);
        }
        V x0 = e.muli(tx, BX);
        V x1 = e.sub(e.smin(e.addi(x0, BX), NX), c1);
        V n0 = e.div(x0, hw);
        V n1 = e.div(x1, hw);
        V same = e.eq(n0, n1);
        V q0 = e.sub(x0, e.mul(n0, hw));
        V q1 = e.sub(x1, e.mul(n0, hw));
        V ho0 = e.select(same, e.div(q0, p_Wo), c0);
        V ho1d = e.div(q1, p_Wo);
        V ho1 = e.select(same, ho1d, e.sub(p_Ho, c1));
        V row = e.land(same, e.eq(ho0, ho1));
        V wo0 = e.select(row, e.sub(q0, e.mul(ho0, p_Wo)), c0);
        V wo1s = e.sub(q1, e.mul(ho1, p_Wo));
        V wo1 = e.select(row, wo1s, e.sub(p_Wo, c1));
        V lo_h = e.subi(e.muli(ho1, SH), PH);
        V hi_h = e.subi(e.muli(ho0, SH), PH);
        V lo_w = e.subi(e.muli(wo1, SH), PW);
        V hi_w = e.subi(e.muli(wo0, SH), PW);
        /* the valid tap counts in closed form (see the main builder) */
        auto cnt = [&](V v, int d, int n) {
            v = e.smax(v, c0);
            if(d > 1)
            {
                v = e.cdivi(v, d);
            }
            return e.smini(v, n);
        };
        V r0 = cnt(e.sub(c0, lo_h), DH, R);
        V r1 = cnt(e.sub(p_Hi, hi_h), DH, R);
        V s0 = cnt(e.sub(c0, lo_w), DW, S);
        V s1 = cnt(e.sub(p_Wi, hi_w), DW, S);
        V er = e.ge(r0, r1);
        V empty = e.lor(er, e.ge(s0, s1));
        out["pc"] = pc;
        out["tx"] = tx;
        out["tk"] = tk;
        out["r0"] = e.select(empty, c0, r0);
        out["r1"] = e.select(empty, c1, r1);
        out["s0"] = e.select(empty, c0, s0);
        out["s1"] = e.select(empty, c1, s1);
        out["ch0"] = dv(e.mul(pc, nch), SP);
        out["ch1"] = dv(e.mul(e.add(pc, c1), nch), SP);
        return out;
    };

    /* DMA offset table, per wave: [tap][wave][pixel of 8][instr] */
    const int RSP = tab_taps(spec);
    V tab_s = e.smem_alloc_1d(i32_t, RSP * 64 * NXI, "flex_tab");

    auto build_tab = [&](V tx) {
        V ln = opaque_lane();
        V l8 = dv(ln, 8);
        V c = md(ln, 8);
        struct TPix
        {
            V xok, hb, wb, base;
        };
        std::vector<TPix> pix;
        for(int i = 0; i < NXI; ++i)
        {
            V xa = e.muli(tx, BX);
            V wn = e.muli(wq, NXI);
            V wi = e.addi(wn, i);
            V x = e.add(xa, e.add(e.muli(wi, 8), l8));
            V n = e.div(x, hw);
            V rem = e.sub(x, e.mul(n, hw));
            V ho = e.div(rem, p_Wo);
            V wo = e.sub(rem, e.mul(ho, p_Wo));
            V hb = e.subi(e.muli(ho, SH), PH);
            V wb = e.subi(e.muli(wo, SH), PW);
            V nh = e.mul(n, p_Hi);
            V base = e.add(e.mul(e.add(nh, hb), p_Wi), wb);
            pix.push_back(TPix{e.lt(x, NX), hb, wb, base});
        }
        for(int kk = 0; kk < RSP / 8; ++kk)
        {
            V t = e.addi(c, 8 * kk);
            V r = dv(t, S);
            V s_ = e.sub(t, e.muli(r, S));
            V dh = e.muli(r, DH);
            V dw = e.muli(s_, DW);
            V tok = e.lti(t, (int64_t)R * S);
            V toff = e.add(e.mul(dh, p_Wi), dw);
            std::vector<V> ents;
            for(const TPix& q : pix)
            {
                V a = e.land(tok, q.xok);
                V hok = rocke_b_cmp_ult(b, e.add(q.hb, dh), p_Hi);
                V wok = rocke_b_cmp_ult(b, e.add(q.wb, dw), p_Wi);
                V ok = e.land(a, e.land(hok, wok));
                V fo = e.mul(e.mul(e.add(q.base, toff), C), c2);
                ents.push_back(e.select(ok, fo, oob));
            }
            V t8 = e.muli(t, 8);
            V tw = e.add(t8, wave);
            V idx = e.muli(e.add(e.muli(tw, 8), l8), NXI);
            V ix[2] = {c0, idx};
            rocke_b_smem_store_vN(
                b, tab_s, ix, 2, rocke_b_vec_pack(b, ents.data(), (int)ents.size(), i32_t), NXI);
        }
    };

    /* one allocation per stage (both groups' halves): the DMA into one stage
     * and the fragment reads of the other stay apart for the waitcnt pass */
    V st_s[2], st_a[2];
    for(int i = 0; i < 2; ++i)
    {
        std::string h = "flexcs" + std::to_string(i);
        st_s[i] = e.smem_alloc_1d(io_type, 2 * GE, h.c_str());
    }
    for(int i = 0; i < 2; ++i)
    {
        st_a[i] = rocke_b_smem_addr_of(b, st_s[i]);
    }

    V h_r16 = md(lane, 16);
    V h_q = dv(lane, 16);
    V h_l8 = dv(lane, 8);
    V h_gran = e.muli(e.bxor(md(lane, 8), h_l8), 16);
    V s4 = e.muli(sl, 4);
    V sq = e.add(s4, h_q);
    V h_sl = e.muli(e.bxor(sq, md(h_r16, 8)), 8);
    /* (the per-wave table row base; run forms its own in the group's branch,
     * this one is unused but kept: it is part of the kernel's IR) */
    e.muli(e.add(e.muli(wave, 8), h_l8), NXI);

    const int RS = R * S;
    V C2 = e.mul(C, c2);
    V RSK = e.mul(ci(RS), K); /* wt: a reduction row of the KRSC filter */

    auto run = [&](int g) {
        const int gb = g * GE; /* this group's half of a stage, elements */
        /* the fragment-read bases (re-formed at each item's head, from an
         * opaque lane: no kernel-long copy to spill) */
        V hx_g = ci(gb);
        V hx_x = e.muli(xi, 64);
        V hx_r = e.muli(e.add(hx_x, h_r16), kCK);
        V hh_xin = e.add(e.add(hx_g, hx_r), h_sl);
        V hk_g = ci(gb + IN_E);
        V hk_r = e.muli(h_r16, kCK);
        V hh_kw = e.add(hk_g, e.add(hk_r, h_sl));

        auto item_head = [&]() {
            V ln2 = opaque_lane();
            V r2 = md(ln2, 16);
            V q2 = dv(ln2, 16);
            V s4b = e.muli(sl, 4);
            V sqb = e.add(s4b, q2);
            V sl2 = e.muli(e.bxor(sqb, md(r2, 8)), 8);
            V xg = ci(gb);
            V xx = e.muli(xi, 64);
            V xr = e.muli(e.add(xx, r2), kCK);
            hh_xin = e.add(e.add(xg, xr), sl2);
            V kg = ci(gb + IN_E);
            V kr = e.muli(r2, kCK);
            hh_kw = e.add(kg, e.add(kr, sl2));
        };

        auto wlane = [&](V tk) {
            V ln = opaque_lane();
            V l8 = dv(ln, 8);
            V c8 = e.bxor(md(ln, 8), l8);
            V c48 = e.muli(dv(l8, 4), 8);
            V chan = e.add(c48, md(l8, 4));
            V kt0 = e.muli(tk, BK);
            V k = e.add(e.add(kt0, e.muli(wq, NWI * 8)), chan);
            V kr = e.mul(k, RSC);
            return e.mul(e.add(kr, e.muli(c8, 8)), c2);
        };

        auto frags_in = [&](int st, bool a_frags, std::vector<V>& a, std::vector<V>& bb) {
            a.clear();
            bb.clear();
            if(a_frags)
            {
                for(int kt = 0; kt < KT; ++kt)
                {
                    V idx = e.addi(hh_kw, (int64_t)kt * 16 * kCK);
                    a.push_back(e.smem_load(st_s[st], c0, idx, io_type, 8));
                }
            }
            for(int xt = 0; xt < kXT; ++xt)
            {
                V idx = e.addi(hh_xin, (int64_t)xt * 16 * kCK);
                bb.push_back(e.smem_load(st_s[st], c0, idx, io_type, 8));
            }
        };

        auto bar = [&](int prio) {
            rocke_b_s_setprio(b, prio);
            rocke_b_s_barrier_bare(b);
            rocke_b_sched_barrier(b, 0);
        };

        const int nacc = kXT * KT;
        V zero = rocke_b_zero_vec_f32(b, 4);
        const rocke_type_t* vec4 = zero->type;
        const rocke_type_t* i32x2 = rocke_vector_type(b, i32_t, 2);
        const rocke_type_t* half8 = rocke_vector_type(b, io_type, 8);
        V d_rsrc = rocke_b_buffer_rsrc(b, P.D, P.D_bytes);

        auto reduce_store = [&](const std::vector<V>& accs, std::map<std::string, V>& info) {
            V ln = opaque_lane();
            V q = dv(ln, 16);
            V x16 = md(ln, 16);
            V q2 = e.mul(md(q, 2), c2);
            V ubase = e.add(q2, dv(q, 2));
            /* this wave's partial region: stage g, slot (xi, sl) */
            V x2 = e.mul(xi, c2);
            V rbase = e.muli(e.add(x2, sl), PART);
            /* reader: thread -> (pixel half, two (row, 8-channel) items) */
            V rxi = dv(tid, 256);
            V t2 = md(tid, 256);
            V tx = info["tx"], tk = info["tk"];
            V s_rsrc = NULL;
            if(SP > 1)
            {
                V slab = e.muli(e.mul(NX, K), 4);
                V pc64 = e.zext64(info["pc"]);
                V sbase = rocke_b_global_ptr_offset(b, P.D, e.mul(pc64, e.zext64(slab)));
                s_rsrc = rocke_b_buffer_rsrc(b, sbase, slab);
            }
            for(int rr = 0; rr < 2; ++rr)
            {
                for(int xl = 0; xl < 2; ++xl)
                {
                    const int xt = 2 * rr + xl;
                    V xk = ci(xl * 16);
                    V rowv = e.add(xk, x16);
                    for(int kt = 0; kt < KT; ++kt)
                    {
                        V uk = ci(kt * 4);
                        V u = e.add(uk, ubase);
                        V rb = e.muli(rowv, BK * 2);
                        V ux = e.muli(e.bxor(u, rowv), 8);
                        V el = e.add(rbase, e.add(rb, ux));
                        V val = rocke_b_vec_bitcast(b, accs[xt * KT + kt], half8);
                        e.smem_store(st_s[g], c0, el, val, 8);
                    }
                }
                rocke_b_s_waitcnt(b, -1, 0, -1);
                rocke_b_s_barrier_bare(b);
                for(int j = 0; j < 2; ++j)
                {
                    V it = e.addi(t2, 256 * j);
                    V row = dv(it, 16);
                    V k8 = md(it, 16);
                    std::vector<V> sums;
                    for(int h = 0; h < 2; ++h)
                    {
                        V k2 = e.mul(k8, c2);
                        V u = e.addi(k2, h);
                        V rb = e.muli(row, BK * 2);
                        V off = e.add(rb, e.muli(e.bxor(u, row), 8));
                        V acc = NULL;
                        for(int gg = 0; gg < 2; ++gg)
                        {
                            for(int ss = 0; ss < 2; ++ss)
                            {
                                V r2 = e.mul(rxi, c2);
                                V el = e.add(e.muli(e.addi(r2, ss), PART), off);
                                V v = rocke_b_vec_bitcast(
                                    b, e.smem_load(st_s[gg], c0, el, io_type, 8), vec4);
                                acc = acc == NULL ? v : rocke_b_fadd(b, acc, v);
                            }
                        }
                        sums.push_back(acc);
                    }
                    V xa = e.muli(tx, BX);
                    V xr = e.muli(rxi, 64);
                    V xq = ci(32 * rr);
                    V x = e.add(xa, e.add(xr, e.add(xq, row)));
                    V ka = e.muli(tk, BK);
                    V k = e.add(ka, e.muli(k8, 8));
                    V xok = e.lt(x, NX);
                    V ok = e.land(xok, e.lt(k, K));
                    if(SP == 1)
                    {
                        V lo = rocke_b_vec_bitcast(
                            b, rocke_dconv_trunc_f32(b, bf16, sums[0]), i32x2);
                        V hi = rocke_b_vec_bitcast(
                            b, rocke_dconv_trunc_f32(b, bf16, sums[1]), i32x2);
                        V w[4];
                        w[0] = rocke_b_vec_extract(b, lo, 0);
                        w[1] = rocke_b_vec_extract(b, lo, 1);
                        w[2] = rocke_b_vec_extract(b, hi, 0);
                        w[3] = rocke_b_vec_extract(b, hi, 1);
                        V val = rocke_b_vec_bitcast(b, rocke_b_vec_pack(b, w, 4, i32_t), half8);
                        V xk = e.mul(x, K);
                        V off = e.select(ok, e.mul(e.add(xk, k), c2), oob);
                        rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, val, 4);
                    }
                    else
                    {
                        V xk = e.mul(x, K);
                        V off = e.select(ok, e.muli(e.add(xk, k), 4), oob);
                        rocke_b_buffer_store_vN_f32(b, s_rsrc, off, c0, sums[0], 4);
                        V off2 = e.select(ok, e.addi(off, 16), oob);
                        rocke_b_buffer_store_vN_f32(b, s_rsrc, off2, c0, sums[1], 4);
                    }
                }
                rocke_b_s_waitcnt(b, -1, 0, -1);
                rocke_b_s_barrier_bare(b);
            }
        };

        /* ---- the group's items: the filter taps of a chunk unrolled, a DMA
         * look-ahead of two steps (a group's DMAs of step j + 2 go into the
         * stage step j just read, spread over step j's MFMA phase). Slot m of
         * a chunk pair is chunk cA + 2 (m // RS), tap m % RS; it is live while
         * that chunk is one of the group's (index < NQ) and < ch1, else its
         * DMAs are dead (zero fill, kept so every step waits the same
         * vmcnt). ---- */
        const int NDMA = NXI + NWI;
        /* formed in the group's branch (held across it, it went to scratch) */
        V gt8 = e.muli(wave, 8);
        V g_tab = e.muli(e.add(gt8, dv(opaque_lane(), 8)), NXI);

        auto ent_of = [&](int tap) {
            V idx = e.addi(g_tab, (int64_t)tap * 64 * NXI);
            return e.smem_load(tab_s, c0, idx, i32_t, NXI);
        };
        auto rsrcs = [&](V ok, V* xr, V* wr) {
            *xr = rocke_b_buffer_rsrc(b, P.A, e.select(ok, P.A_bytes, c0));
            *wr = rocke_b_buffer_rsrc(b, P.Bp, e.select(ok, P.B_bytes, c0));
        };
        V live_x = rocke_b_buffer_rsrc(b, P.A, P.A_bytes);
        V live_w = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes);

        /* a per-step copy: products of it are formed where used, not hoisted
         * and held across the unrolled sweep */
        auto opaque_s = [&](V v) { return opaque(b, v, "=s,0"); };

        /* this wave's first DMA slot of each stage as one SGPR (16 held
         * addresses cost SGPR spills and VGPR scratch) */
        V sbase[2];
        for(int st = 0; st < 2; ++st)
        {
            V k = ci(gb * 2);
            V off = e.add(k, e.muli(wq, NXI * 1024));
            V pa = rocke_b_smem_ptr_add(b, st_a[st], e.zext64(off));
            sbase[st] = rocke_b_readfirstlane(b, rocke_b_trunc(b, pa, i32_t));
        }

        auto dma_ops
            = [&](V xr, V wr, V ent, V chb, int tap, const Wo& wo, int st, const char* parts) {
                  std::vector<Cs4Op> ops;
                  if(strchr(parts, 'x') != NULL)
                  {
                      for(int i = 0; i < NXI; ++i)
                      {
                          V v = e.add(rocke_b_vec_extract(b, ent, i), h_gran);
                          ops.push_back(Cs4Op{xr, sbase[st], i * 1024, v, chb});
                      }
                  }
                  if(strchr(parts, 'w') == NULL)
                  {
                      return ops;
                  }
                  if(spec->wt)
                  {
                      /* piece q = wq * 4 + i: reduction rows 4 q + lane // 16
                       * of the flipped tap */
                      V rsk = opaque_s(RSK);
                      V kk = opaque_s(K);
                      V c2a = e.mul(e.mul(dv(chb, 2), rsk), c2);
                      V tk = ci(RS - 1 - tap);
                      V c2b = e.mul(e.mul(tk, kk), c2);
                      V s1 = e.add(c2a, c2b);
                      V wqr = e.mul(e.muli(wq, 4 * NWI * 2), RSK);
                      V sb = e.add(s1, e.add(wo.k0b, wqr));
                      for(int i = 0; i < NWI; ++i)
                      {
                          V soff = i ? e.add(sb, e.muli(rsk, 8 * i)) : sb;
                          V vo = ((i >> 1) & 1) ? wo.v1 : wo.v0;
                          ops.push_back(Cs4Op{wr, sbase[st], IN_E * 2 + i * 1024, vo, soff});
                      }
                      return ops;
                  }
                  V tk = ci(tap);
                  V woff = e.add(e.mul(tk, opaque_s(C2)), chb);
                  V rsc = opaque_s(RSC);
                  for(int i = 0; i < NWI; ++i)
                  {
                      const int wsh = ((i & 1) * 4 + (i >> 1) * 16) * 2;
                      V soff = wsh ? e.add(woff, e.muli(rsc, wsh)) : woff;
                      ops.push_back(Cs4Op{wr, sbase[st], IN_E * 2 + i * 1024, wo.v0, soff});
                  }
                  return ops;
              };
        auto issue
            = [&](const Cs4Op& op) { dma_lds_at(b, op.rsrc, op.base, op.imm, op.voff, op.soff); };

        V t_l0 = NULL;
        if(spec->wt)
        {
            /* transposing A reads of the [64 reduction rows][128 outputs]
             * stage: the lane fields are disjoint bit ranges (BK 128), in
             * bytes, so a fragment's XOR stays in the byte domain */
            V ln0 = opaque_lane();
            auto bits = [&](int sh, int mask, int to) {
                V v = sh ? e.lshri(ln0, sh) : ln0;
                v = e.land(v, ci(mask));
                return to ? e.shli(v, to) : v;
            };
            V b1 = bits(4, 3, 11);
            V b2 = bits(2, 3, 8);
            V o1 = e.lor(b1, b2);
            V b3 = bits(2, 3, 5);
            V b4 = bits(4, 1, 7);
            V o2 = e.lor(o1, e.lor(b3, b4));
            V b5 = bits(0, 1, 4);
            V b6 = bits(1, 1, 3);
            V lp = e.lor(o2, e.lor(b5, b6));
            V gk = ci(gb + IN_E);
            V sk = e.add(gk, e.muli(sl, 32 * BK));
            t_l0 = e.lor(lp, rocke_b_shl(b, sk, c1));
        }

        auto frags = [&](int st, std::vector<V>& a, std::vector<V>& bb) {
            if(!spec->wt)
            {
                frags_in(st, true, a, bb);
                return;
            }
            std::vector<V> none;
            frags_in(st, false, none, bb);
            a.clear();
            /* the kernel's lane part (one VGPR), laundered per step */
            V t_l = opaque(b, t_l0, "=v,0");
            for(int kt = 0; kt < KT; ++kt)
            {
                V x = kt ? e.bxor(t_l, ci(kt * 32)) : t_l;
                V base = rocke_b_lshr(b, x, c1);
                V lo = e.ds_read_tr(st_s[st], c0, base, io_type);
                V hi = e.ds_read_tr(st_s[st], c0, e.addi(base, 4 * BK), io_type);
                a.push_back(rocke_b_vec_concat(b, lo, hi));
            }
        };

        /* Step on stage st; the DMAs of step j + 2 (slot rs, tap) go into
         * stage st during the MFMA phase, their offsets formed before the
         * phase barrier. */
        auto step2 = [&](std::vector<V> accs, int st, const SlotRs& rs, int tap, const Wo& wo) {
            V ent = ent_of(tap);
            std::vector<V> a, bb;
            frags(st, a, bb);
            rocke_b_s_waitcnt(b, -1, 0, -1);
            std::vector<Cs4Op> ops = dma_ops(rs.xr, rs.wr, ent, rs.chb, tap, wo, st, "xw");
            rocke_b_sched_barrier(b, 0);
            bar(1);
            const int every = (kXT * KT) / (int)ops.size();
            int j = 0;
            for(int xt = 0; xt < kXT; ++xt)
            {
                for(int kt = 0; kt < KT; ++kt)
                {
                    if(j % every == 0 && j / every < (int)ops.size())
                    {
                        rocke_b_sched_barrier(b, 0);
                        issue(ops[j / every]);
                        rocke_b_sched_barrier(b, 0);
                    }
                    const int i = xt * KT + kt;
                    accs[i] = e.mfma16x16x32(bf16, a[kt], bb[xt], accs[i]);
                    ++j;
                }
            }
            rocke_b_sched_barrier(b, 0);
            rocke_b_s_waitcnt(b, NDMA, -1, -1);
            bar(0);
            return accs;
        };

        const std::vector<std::string> o_names(1, "o_d");
        const std::vector<V> o_inits(1, c0);
        e.for_iter(c0, rounds, c1, o_names, o_inits, "rnd", [&](V rnd, std::vector<V>) {
            V t = tile_of(rnd);
            std::map<std::string, V> info = item_info(t);
            item_head();
            Wo wo = {NULL, NULL, NULL};
            if(spec->wt)
            {
                /* slot l % 16 of row 4 q + l // 16 takes group (l % 16) // 2 ^
                 * (l // 16) ^ 4 ((q >> 1) & 1); the columns past K read other
                 * rows' weights, of outputs never stored */
                V wl = opaque_lane();
                V w16 = dv(wl, 16);
                V s16 = md(wl, 16);
                V cbx = e.muli(e.bxor(dv(s16, 2), w16), 16);
                V cb = e.add(cbx, e.muli(md(s16, 2), 8));
                V wrow = e.mul(w16, RSK);
                wo.v0 = e.mul(e.add(wrow, cb), c2);
                V cx = e.bxor(cb, ci(64));
                wo.v1 = e.mul(e.add(wrow, cx), c2);
                wo.k0b = e.muli(info["tk"], BK * 2);
            }
            else
            {
                wo.v0 = wlane(info["tk"]);
            }
            V ch0 = info["ch0"], ch1 = info["ch1"];
            V NQ = dv(e.add(e.sub(ch1, ch0), c1), 2); /* chunks per group */
            V chA = e.addi(ch0, g);

            auto slot = [&](V base_q, V cbase, int m) {
                const int q = m / RS;
                V ch = e.addi(cbase, 2 * q);
                V qa = e.lt(e.addi(base_q, q), NQ);
                V ok = e.land(qa, e.lt(ch, ch1));
                return Slot{ok, e.muli(ch, kCK * 2), m % RS};
            };

            /* the previous item's reduce ended on a barrier: the stages are
             * free. The weight DMAs of steps 0 and 1 need no table: they go
             * first and land while this wave builds its table rows. */
            std::vector<Slot> pro;
            for(int m = 0; m < 2; ++m)
            {
                pro.push_back(slot(c0, chA, m));
            }
            for(int m = 0; m < 2; ++m)
            {
                V xr, wr;
                rsrcs(pro[m].ok, &xr, &wr);
                for(const Cs4Op& op : dma_ops(xr, wr, NULL, pro[m].chb, pro[m].tap, wo, m % 2, "w"))
                {
                    issue(op);
                }
            }
            build_tab(info["tx"]);
            for(int m = 0; m < 2; ++m)
            {
                V ent = ent_of(pro[m].tap);
                rocke_b_s_waitcnt(b, -1, 0, -1);
                V xr, wr;
                rsrcs(pro[m].ok, &xr, &wr);
                for(const Cs4Op& op : dma_ops(xr, wr, ent, pro[m].chb, pro[m].tap, wo, m % 2, "x"))
                {
                    issue(op);
                }
            }
            rocke_b_s_waitcnt(b, NXI, -1, -1);
            rocke_b_s_barrier_bare(b);
            if(g == 1)
            {
                rocke_b_s_barrier_bare(b); /* group 1 runs one barrier behind */
            }
            rocke_b_sched_barrier(b, 0);
            std::vector<V> accs(nacc, zero);
            V NP = dv(NQ, 2);
            std::vector<V> res = e.for_iter(
                c0, NP, c1, iter_names("a", nacc), accs, "cpair", [&](V qp, std::vector<V> lc) {
                    std::vector<V> la(lc.begin(), lc.begin() + nacc);
                    V q0 = e.mul(qp, c2);
                    V cA = e.add(chA, e.muli(qp, 4));
                    std::map<int, SlotRs> slots;
                    for(int m = 2; m < 2 * RS + 2; ++m)
                    {
                        const int q = m / RS;
                        if(slots.count(q) == 0)
                        {
                            Slot s = slot(q0, cA, m);
                            /* the pair's first chunk is always live */
                            SlotRs r = {live_x, live_w, s.chb};
                            if(q)
                            {
                                rsrcs(s.ok, &r.xr, &r.wr);
                            }
                            slots[q] = r;
                        }
                    }
                    for(int k = 0; k < 2 * RS; ++k)
                    {
                        const int m = k + 2;
                        la = step2(la, k % 2, slots[m / RS], m % RS, wo);
                    }
                    return la;
                });
            if((int)res.size() != nacc)
            {
                return std::vector<V>();
            }
            const std::vector<const rocke_type_t*> types(nacc, vec4);
            V odd = e.eq(md(NQ, 2), c1);
            std::vector<V> fin = if_else(
                b,
                odd,
                types,
                [&]() {
                    std::vector<V> ta = res;
                    V qT = e.mul(NP, c2);
                    V cT = e.add(chA, e.muli(NP, 4));
                    std::map<int, SlotRs> slots;
                    for(int m = 2; m < RS + 2; ++m)
                    {
                        const int q = m / RS;
                        if(slots.count(q) == 0)
                        {
                            Slot s = slot(qT, cT, m);
                            SlotRs r = {NULL, NULL, s.chb};
                            rsrcs(s.ok, &r.xr, &r.wr);
                            slots[q] = r;
                        }
                    }
                    for(int k = 0; k < RS; ++k)
                    {
                        const int m = k + 2;
                        ta = step2(ta, k % 2, slots[m / RS], m % RS, wo);
                    }
                    return ta;
                },
                [&]() { return res; });
            /* the last steps' (dead) DMAs land before any wave reuses the
             * stages */
            rocke_b_s_waitcnt(b, 0, -1, -1);
            rocke_b_s_barrier_bare(b);
            rocke_b_s_setprio(b, 0);
            if(g == 0)
            {
                rocke_b_s_barrier_bare(b); /* wait out group 1's last MFMA phase */
            }
            reduce_store(fin, info);
            rocke_b_sched_barrier(b, 0);
            return std::vector<V>(1, c0);
        });
    };

    if_else_void(b, e.eq(dv(wave, 4), c0), [&]() { run(0); }, [&]() { run(1); });
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace rocke_ffx
