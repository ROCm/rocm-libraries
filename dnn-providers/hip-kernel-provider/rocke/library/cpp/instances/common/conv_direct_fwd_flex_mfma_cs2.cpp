// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_flex_mfma_cs2.cpp -- C++ engine port of _build_cs2 in
 * library/kernels/common/conv_direct_fwd_flex_mfma.py: the cs 2 flex forward
 * (X128 x K256 tiles, the reduction split two ways inside the block).
 *
 * Both wave groups read every step (one tap and one 64-channel chunk) from one
 * of three shared stages; group g takes the chunk's 32-channel half g, and
 * wave (xi, ki) of a group 64 pixels x 128 channels of the tile. Each group
 * stages half of a step, group g the half of step j + 1 + g while it reads
 * step j, issued between its fragment reads. Every filter tap runs; the taps
 * of a chunk are unrolled (a tap count that is a multiple of 3 keeps the stage
 * of each tap fixed). A tile is drained at its end through the stage memory.
 *
 * Every IR op is emitted in the order the Python builder emits it; see the
 * contract in rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include <stdint.h>

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

/* A DMA: rsrc, LDS SGPR base, immediate, voffset, soffset. */
struct Cs2Op
{
    V rsrc, base;
    int imm;
    V voff, soff;
};

/* The weight DMA lane parts: non-wt v0 only; wt (v0, v1, the item's first
 * output in bytes with the group's half). */
struct Cs2Wo
{
    V v0, v1, k0b;
};

} // namespace

rocke_kernel_def_t*
    build_cs2(rocke_ir_builder_t* b, const rocke_fwd_flex_mfma_spec_t* spec, const char* arch)
{
    (void)arch;
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const rocke_type_t* i32_t = rocke_i32();
    const int BX = 128, BK = 256, SP = spec->split_c, KT = 8;
    const int NXI = 2, NWI = 4; /* DMA instructions per wave per step (its group's half) */
    const int R = p.KH, S = p.KW;
    const int RS = R * S;
    const int SH = p.stride, PH = p.PAD, PW = rocke_direct_conv_problem_pad_w(&p);
    const int DH = p.dil_h, DW = p.dil_w;
    const int IN_E = BX * kCK; /* input elements of a stage */
    const int GE = (BX + BK) * kCK; /* elements of a stage */

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
    V ki = dv(wq, 2);

    auto opaque_lane = [&]() { return opaque(b, lane, "=v,0"); };
    auto opaque_s = [&](V v) { return opaque(b, v, "=s,0"); };

    V hw = e.mul(p_Ho, p_Wo);
    V NX = e.mul(p_N, hw);
    V nxt = e.cdivi(NX, BX);
    V nkt = e.cdivi(K, BK);
    V TI = e.muli(e.mul(nxt, nkt), SP);
    V G = e.smini(TI, kPersist);
    V nch = dv(C, kCK);
    V RSC = e.mul(ci(RS), C);
    V C2 = e.mul(C, c2);
    V blk = rocke_b_block_id_x(b);
    if(spec->xcd)
    {
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
        out["pc"] = pc;
        out["ch0"] = dv(e.mul(pc, nch), SP);
        out["ch1"] = dv(e.mul(e.add(pc, c1), nch), SP);
        if(spec->kmaj)
        {
            out["tx"] = e.mod(tl, nxt);
            out["tk"] = e.div(tl, nxt);
        }
        else
        {
            out["tx"] = e.div(tl, nkt);
            out["tk"] = e.mod(tl, nkt);
        }
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
            V wn = e.muli(wave, NXI);
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
            V tok = e.lti(t, RS);
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

    /* one allocation per stage: the DMA into one stage and the fragment reads
     * of another stay apart for the waitcnt pass */
    V st_s[3], st_a[3];
    for(int i = 0; i < 3; ++i)
    {
        std::string h = "flexc2s" + std::to_string(i);
        st_s[i] = e.smem_alloc_1d(io_type, GE, h.c_str());
    }
    for(int i = 0; i < 3; ++i)
    {
        st_a[i] = rocke_b_smem_addr_of(b, st_s[i]);
    }
    /* this wave's first input and weight DMA slots of each stage */
    V sb_in[3], sb_w[3];
    for(int which = 0; which < 2; ++which)
    {
        const int o = which == 0 ? 0 : IN_E * 2;
        const int n = which == 0 ? NXI * 1024 : NWI * 1024;
        for(int st = 0; st < 3; ++st)
        {
            V k = ci(o);
            V off = e.add(k, e.muli(wave, n));
            V pa = rocke_b_smem_ptr_add(b, st_a[st], e.zext64(off));
            V sb = rocke_b_readfirstlane(b, rocke_b_trunc(b, pa, i32_t));
            (which == 0 ? sb_in : sb_w)[st] = sb;
        }
    }

    V r16 = md(lane, 16);
    V gm = md(lane, 8);
    V h_gran = e.muli(e.bxor(gm, dv(lane, 8)), 16);
    /* the group's 32-channel half of the chunk (granules 4 g.. of a row) */
    V g4 = e.muli(dv(wave, 4), 4);
    V gq = e.add(g4, dv(lane, 16));
    V h_sl = e.muli(e.bxor(gq, md(r16, 8)), 8);
    V xx = e.muli(xi, 64);
    V h_xin = e.add(e.muli(e.add(xx, r16), kCK), h_sl);
    V kin = ci(IN_E);
    V kk128 = e.muli(ki, 128);
    V kr = e.muli(e.add(kk128, r16), kCK);
    V h_kw = e.add(kin, e.add(kr, h_sl));
    V ht8 = e.muli(wave, 8);
    V h_tab = e.muli(e.add(ht8, dv(lane, 8)), NXI);

    V RSK = e.mul(ci(RS), K); /* wt: a reduction row of the KRSC filter */
    V GQ = dv(wave, 4); /* the group: its 32-row half of a step's 64 */

    /* wt: the weight DMA lane parts (piece q = 4 wave + i, half q // 16 = the
     * group, rows 4 (q % 16) + lane // 16) and the item's first output */
    auto wlane_wt = [&](V tk) {
        V wl = opaque_lane();
        V w4 = e.lshri(wl, 4);
        V w16 = e.land(w4, ci(3));
        V s16 = e.land(wl, ci(15));
        V sx = e.bxor(rocke_b_lshr(b, s16, c1), w16);
        V cbx = e.muli(sx, 16);
        V cb = e.add(cbx, e.muli(e.land(s16, c1), 8));
        V wrow = e.mul(w16, RSK);
        Cs2Wo w;
        w.v0 = e.mul(e.add(wrow, cb), c2);
        V cx = e.bxor(cb, ci(64));
        w.v1 = e.mul(e.add(wrow, cx), c2);
        V tb = e.muli(tk, BK);
        w.k0b = e.mul(e.add(tb, e.muli(GQ, 128)), c2);
        return w;
    };

    /* wt: a transposing read's element offset in the stage (lane part from
     * shifts and masks), in bytes */
    auto t_lane = [&]() {
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
        V kin2 = ci(IN_E);
        V k1 = e.muli(ki, 8192);
        V k2 = e.add(k1, e.muli(GQ, 4096));
        V sk = e.add(kin2, k2);
        return e.lor(lp, rocke_b_shl(b, sk, c1));
    };

    /* wt: the transposing reads' lane part formed once per kernel (one VGPR) */
    V t_l0 = spec->wt ? t_lane() : NULL;

    auto wlane = [&](V tk) {
        if(spec->wt)
        {
            return wlane_wt(tk);
        }
        V ln = opaque_lane();
        V l8 = dv(ln, 8);
        V c8 = e.bxor(md(ln, 8), l8);
        V c48 = e.muli(dv(l8, 4), 8);
        V chan = e.add(c48, md(l8, 4));
        V kt0 = e.muli(tk, BK);
        V k = e.add(e.add(kt0, e.muli(wave, NWI * 8)), chan);
        V kr2 = e.mul(k, RSC);
        Cs2Wo w = {NULL, NULL, NULL};
        w.v0 = e.mul(e.add(kr2, e.muli(c8, 8)), c2);
        return w;
    };

    auto ent_of = [&](int tap) {
        V idx = e.addi(h_tab, (int64_t)tap * 64 * NXI);
        return e.smem_load(tab_s, c0, idx, i32_t, NXI);
    };

    V xr = rocke_b_buffer_rsrc(b, P.A, P.A_bytes);
    V wr = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes);

    /* The wave's 6 DMAs of (tap, chunk at chb bytes) into stage st. */
    auto dma_ops = [&](V ent, V chb, int tap, const Cs2Wo& wo, int st) {
        std::vector<Cs2Op> ops;
        for(int i = 0; i < NXI; ++i)
        {
            V v = e.add(rocke_b_vec_extract(b, ent, i), h_gran);
            ops.push_back(Cs2Op{xr, sb_in[st], i * 1024, v, chb});
        }
        if(spec->wt)
        {
            /* rows chb / 2 + 16 wq + 4 i + lane // 16 of the flipped tap */
            V rsk = opaque_s(RSK);
            V kk = opaque_s(K);
            V c2a = e.mul(e.mul(dv(chb, 2), rsk), c2);
            V tk = ci(RS - 1 - tap);
            V c2b = e.mul(e.mul(tk, kk), c2);
            V s1 = e.add(c2a, c2b);
            V wqr = e.mul(e.muli(wq, 32), RSK);
            V sb = e.add(s1, e.add(wo.k0b, wqr));
            for(int i = 0; i < NWI; ++i)
            {
                V soff = i ? e.add(sb, e.muli(rsk, 8 * i)) : sb;
                V vo = ((i >> 1) & 1) ? wo.v1 : wo.v0;
                ops.push_back(Cs2Op{wr, sb_w[st], i * 1024, vo, soff});
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
            ops.push_back(Cs2Op{wr, sb_w[st], i * 1024, wo.v0, soff});
        }
        return ops;
    };

    auto bar = [&](int prio) {
        rocke_b_s_setprio(b, prio);
        rocke_b_s_barrier_bare(b);
        rocke_b_sched_barrier(b, 0);
    };

    const int nacc = kXT * KT;
    V zero = rocke_b_zero_vec_f32(b, 4);
    const rocke_type_t* vec4 = zero->type;

    /* One step on stage tap % 3: the fragment reads with this wave's copies
     * (ops, or none) between them, the table entries of the next step's
     * copies (tap ent_next, or none), then the MFMA phase. */
    auto step
        = [&](std::vector<V> accs, int tap, const std::vector<Cs2Op>& ops, int ent_next, V* e2) {
              const int st = tap % 3;
              *e2 = ent_next >= 0 ? ent_of(ent_next) : NULL;
              std::vector<V> a(KT, (V)NULL), bb(kXT, (V)NULL);
              const int nreads = KT + kXT;
              const int nops = (int)ops.size();
              const int per = (nreads + (nops > 1 ? nops : 1) - 1) / (nops > 1 ? nops : 1);
              V tl = NULL;
              if(spec->wt)
              {
                  /* the kernel's lane part, laundered per step (else the compiler
             * keeps the 8 XORed addresses live across the sweep) */
                  tl = opaque(b, t_l0, "=v,0");
              }
              for(int j = 0; j < nreads; ++j)
              {
                  if(j % per == 0 && j / per < nops)
                  {
                      const Cs2Op& op = ops[j / per];
                      dma_lds_at(b, op.rsrc, op.base, op.imm, op.voff, op.soff);
                  }
                  if(j < KT && spec->wt)
                  {
                      /* two transposing reads (rows 8 tb + ti and + 4), group i one
                 * XOR */
                      const int i = j;
                      V x = i ? e.bxor(tl, ci(i * 32)) : tl;
                      V base = rocke_b_lshr(b, x, c1);
                      V lo = e.ds_read_tr(st_s[st], c0, base, io_type);
                      V hi = e.ds_read_tr(st_s[st], c0, e.addi(base, 4 * 128), io_type);
                      a[i] = rocke_b_vec_concat(b, lo, hi);
                  }
                  else if(j < KT)
                  {
                      V idx = e.addi(h_kw, (int64_t)j * 16 * kCK);
                      a[j] = e.smem_load(st_s[st], c0, idx, io_type, 8);
                  }
                  else
                  {
                      const int i = j - KT;
                      V idx = e.addi(h_xin, (int64_t)i * 16 * kCK);
                      bb[i] = e.smem_load(st_s[st], c0, idx, io_type, 8);
                  }
              }
              rocke_b_s_waitcnt(b, -1, 0, -1);
              rocke_b_sched_barrier(b, 0);
              bar(1);
              for(int xt = 0; xt < kXT; ++xt)
              {
                  for(int kt = 0; kt < KT; ++kt)
                  {
                      const int i = xt * KT + kt;
                      accs[i] = e.mfma16x16x32(bf16, a[kt], bb[xt], accs[i]);
                  }
              }
              rocke_b_sched_barrier(b, 0);
              /* this step's copies land before any wave's next memory phase */
              rocke_b_s_waitcnt(b, 0, -1, -1);
              bar(0);
              return accs;
          };

    const rocke_type_t* i32x2 = rocke_vector_type(b, i32_t, 2);
    const rocke_type_t* half8 = rocke_vector_type(b, io_type, 8);
    V d_rsrc = rocke_b_buffer_rsrc(b, P.D, P.D_bytes);
    const int WREG = 16 * 128 * 2; /* one wave's partial region (16 pixels x 128 fp32) */

    /* Four rounds (16-pixel tile xt): every wave writes its partial sums of xt
     * to region wq of its group's stage, then each (pixel, 8 channels) item
     * sums the two groups' regions and is stored. */
    auto reduce_store = [&](const std::vector<V>& accs, std::map<std::string, V>& info, int g) {
        V ln = opaque_lane();
        V q = dv(ln, 16);
        V x16 = md(ln, 16);
        V q2 = e.mul(md(q, 2), c2);
        V ubase = e.add(q2, dv(q, 2));
        V wbase = e.muli(wq, WREG);
        V w64 = e.muli(wave, kWave);
        V t0 = e.add(w64, opaque_lane());
        V tx = info["tx"], tk = info["tk"];
        V s_rsrc = NULL;
        if(SP > 1)
        {
            V slab = e.muli(e.mul(NX, K), 4);
            V pc64 = e.zext64(info["pc"]);
            V sbase = rocke_b_global_ptr_offset(b, P.D, e.mul(pc64, e.zext64(slab)));
            s_rsrc = rocke_b_buffer_rsrc(b, sbase, slab);
        }
        for(int xt = 0; xt < kXT; ++xt)
        {
            for(int kt = 0; kt < KT; ++kt)
            {
                V uk = ci(kt * 4);
                V u = e.add(uk, ubase);
                V xr16 = e.muli(x16, 256);
                V ux = e.muli(e.bxor(u, x16), 8);
                V el = e.add(wbase, e.add(xr16, ux));
                V ix[2] = {c0, el};
                rocke_b_smem_store_vN(
                    b, st_s[g], ix, 2, rocke_b_vec_bitcast(b, accs[xt * KT + kt], half8), 8);
            }
            rocke_b_s_waitcnt(b, -1, 0, -1);
            rocke_b_s_barrier_bare(b);
            for(int j = 0; j < 2; ++j)
            {
                V it = e.addi(t0, 512 * j);
                V wp = dv(it, 256); /* the region's wave position (xi, ki) */
                V row = md(dv(it, 16), 16);
                V k8 = md(it, 16);
                std::vector<V> sums;
                for(int h = 0; h < 2; ++h)
                {
                    V k2 = e.mul(k8, c2);
                    V u = e.addi(k2, h);
                    V wr2 = e.muli(wp, WREG);
                    V rr = e.muli(row, 256);
                    V off = e.add(wr2, e.add(rr, e.muli(e.bxor(u, row), 8)));
                    V v0 = rocke_b_vec_bitcast(b, e.smem_load(st_s[0], c0, off, io_type, 8), vec4);
                    V v1 = rocke_b_vec_bitcast(b, e.smem_load(st_s[1], c0, off, io_type, 8), vec4);
                    sums.push_back(rocke_b_fadd(b, v0, v1));
                }
                V xa = e.muli(tx, BX);
                V xw = e.muli(md(wp, 2), 64);
                V xq = ci(16 * xt);
                V x = e.add(xa, e.add(xw, e.add(xq, row)));
                V ka = e.muli(tk, BK);
                V kw2 = e.muli(dv(wp, 2), 128);
                V k = e.add(ka, e.add(kw2, e.muli(k8, 8)));
                V xok = e.lt(x, NX);
                V ok = e.land(xok, e.lt(k, K));
                if(SP == 1)
                {
                    V lo = rocke_b_vec_bitcast(b, rocke_dconv_trunc_f32(b, bf16, sums[0]), i32x2);
                    V hi = rocke_b_vec_bitcast(b, rocke_dconv_trunc_f32(b, bf16, sums[1]), i32x2);
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

    /* Group g's items: group g stages its half of step j + 1 + g during step
     * j (its own entries read a step ahead). */
    auto run = [&](int g) {
        const std::vector<std::string> o_names(1, "o_d");
        const std::vector<V> o_inits(1, c0);
        e.for_iter(c0, rounds, c1, o_names, o_inits, "rnd", [&](V rnd, std::vector<V>) {
            V t = tile_of(rnd);
            std::map<std::string, V> info = item_info(t);
            Cs2Wo wo = wlane(info["tk"]);
            V ch0 = info["ch0"], ch1 = info["ch1"];
            /* the previous item's reduce ended on a barrier: the stages are
             * free; a wave's table rows are its own */
            build_tab(info["tx"]);
            V cb0 = e.muli(ch0, kCK * 2);
            /* group 1 also stages its half of step 1 */
            for(int m = 0; m < (g == 1 ? 2 : 1); ++m)
            {
                V en = ent_of(m);
                rocke_b_s_waitcnt(b, -1, 0, -1);
                for(const Cs2Op& op : dma_ops(en, cb0, m, wo, m))
                {
                    dma_lds_at(b, op.rsrc, op.base, op.imm, op.voff, op.soff);
                }
            }
            V ent = ent_of(1 + g); /* the first step's copies (tap 1 + g of each chunk) */
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_s_barrier_bare(b);
            if(g == 1)
            {
                rocke_b_s_barrier_bare(b); /* group 1 runs one barrier behind */
            }
            rocke_b_sched_barrier(b, 0);
            std::vector<V> accs(nacc, zero);
            V nloop = e.sub(e.sub(ch1, ch0), c1);
            std::vector<V> la = e.for_iter(
                c0, nloop, c1, iter_names("a", nacc), accs, "chunk", [&](V cc, std::vector<V> lc) {
                    std::vector<V> l(lc.begin(), lc.begin() + nacc);
                    V chb = e.muli(e.add(ch0, cc), kCK * 2);
                    V chn = e.addi(chb, kCK * 2);
                    V en = ent;
                    for(int tp = 0; tp < RS; ++tp)
                    {
                        const int tt = tp + 1 + g;
                        std::vector<Cs2Op> ops
                            = dma_ops(en, tt < RS ? chb : chn, tt % RS, wo, tt % 3);
                        V e2 = NULL;
                        l = step(l, tp, ops, tp + 1 < RS ? (tp + 2 + g) % RS : -1, &e2);
                        en = e2;
                    }
                    return l;
                });
            if((int)la.size() != nacc)
            {
                return std::vector<V>();
            }
            /* the last chunk: no copies past the item's last step */
            V chb = e.muli(e.sub(ch1, c1), kCK * 2);
            V en = ent;
            for(int tp = 0; tp < RS; ++tp)
            {
                const int tt = tp + 1 + g;
                std::vector<Cs2Op> ops;
                if(tt < RS)
                {
                    ops = dma_ops(en, chb, tt, wo, tt % 3);
                }
                V e2 = NULL;
                la = step(la, tp, ops, tp + 2 + g < RS ? tp + 2 + g : -1, &e2);
                en = e2;
            }
            rocke_b_s_setprio(b, 0);
            if(g == 0)
            {
                rocke_b_s_barrier_bare(b); /* wait out group 1's last MFMA phase */
            }
            reduce_store(la, info, g);
            rocke_b_sched_barrier(b, 0);
            return std::vector<V>(1, c0);
        });
    };

    if_else_void(b, e.eq(dv(wave, 4), c0), [&]() { run(0); }, [&]() { run(1); });
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace rocke_ffx
