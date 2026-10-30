/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_dgrad_stem_wmma_internal.hpp -- PRIVATE helpers shared by
 * the C++ ports of the wave32 WMMA stem dgrad kernels (conv_dgrad_patch_wmma.cpp,
 * conv_dgrad_smallc_wmma.cpp). Declares no public API.
 *
 * Both Python builders (library/kernels/common/conv_dgrad_{patch,smallc}_wmma.py)
 * stage their operands exactly like conv_direct_dgrad_wmma.py with a 1x1 filter:
 * one 3-D TDM transfer per operand and dY channel chunk into an LDS ring, a
 * transposed weight read (ds_load_tr16_b128) and a 16x16x32 WMMA per tile pair.
 * The pieces they share op for op live here: the rank-3 TDM descriptor, the
 * per-lane LDS read offsets, and the chunk loop. The issue step differs per
 * kernel and is passed in as a callback.
 *
 * BYTE-IDENTITY CONTRACT. The Python builders emit one op per builder call, in
 * Python's left-to-right argument order; C++ leaves argument evaluation order
 * unspecified. Every nested emitting call is therefore hoisted into an ordered
 * local: a call may nest one emitting argument, never two.
 */
#ifndef ROCKE_INSTANCE_CONV_DGRAD_STEM_WMMA_INTERNAL_HPP
#define ROCKE_INSTANCE_CONV_DGRAD_STEM_WMMA_INTERNAL_HPP

#include <stdint.h>
#include <stdio.h>

#include <functional>
#include <string>
#include <vector>

#include "rocke/ir.h"

namespace rocke_dgstem
{

using V = rocke_value_t*;

/* WMMA_K / CH_PAD / *_VGPR_BUDGET of the Python modules. */
constexpr int kWmmaK = 32;
constexpr int kChPad = 8;
constexpr int kVgprBudget = 448;

/* Python floor division `a // b` (b > 0). */
inline long long floordiv(long long a, long long b)
{
    long long q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

/* Python `-(-a // b)`. */
inline long long ceil_div(long long a, long long b)
{
    return -floordiv(-a, b);
}

inline void set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason && reason_cap > 0)
        snprintf(reason, reason_cap, "%s", msg);
}

inline V ci(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

/* _tdm_descriptor_3d(b, global_ptr, lds_addr, dims, strides, tile0, tile1,
 * tile2) of conv_direct_dgrad_wmma.py: the five D# groups of a rank-3 TDM
 * tile transfer of 2-byte elements. The caller emits tile2 (and any emitting
 * dims / strides) first, as Python evaluates the call's arguments first. */
inline void tdm_descriptor_3d(rocke_ir_builder_t* b,
                              V global_ptr,
                              V lds_addr,
                              V d0,
                              V d1,
                              V d2,
                              V s1,
                              V s2,
                              int tile0,
                              int tile1,
                              V tile2,
                              V out[5])
{
    V c16 = ci(b, 16);
    V gaddr = rocke_b_global_ptr_to_i64(b, global_ptr);
    V c32 = rocke_b_const_i64(b, 32);
    V sh = rocke_b_lshr(b, gaddr, c32);
    V g_hi = rocke_b_trunc(b, sh, rocke_i32());
    V g0[4];
    V g1[8];
    V g2[4];
    g0[0] = ci(b, 1);
    g0[1] = lds_addr;
    g0[2] = rocke_b_trunc(b, gaddr, rocke_i32());
    {
        V m = ci(b, 0x1FFFFFF);
        V lo = rocke_b_land(b, g_hi, m);
        V ty = ci(b, -((int64_t)1 << 31));
        g0[3] = rocke_b_lor(b, lo, ty);
    }
    g1[0] = ci(b, 1 << 16);
    g1[1] = rocke_b_shl(b, d0, c16);
    {
        V x = rocke_b_lshr(b, d0, c16);
        V y = rocke_b_shl(b, d1, c16);
        g1[2] = rocke_b_lor(b, x, y);
    }
    {
        V x = rocke_b_lshr(b, d1, c16);
        V y = ci(b, (int64_t)tile0 << 16);
        g1[3] = rocke_b_lor(b, x, y);
    }
    {
        V x = ci(b, tile1);
        V y = rocke_b_shl(b, tile2, c16);
        g1[4] = rocke_b_lor(b, x, y);
    }
    g1[5] = s1;
    g1[6] = rocke_b_shl(b, s2, c16);
    g1[7] = rocke_b_lshr(b, s2, c16);
    g2[0] = d2;
    g2[1] = ci(b, 0);
    g2[2] = ci(b, 0);
    g2[3] = ci(b, 0);
    {
        V r[8];
        int i;
        for(i = 0; i < 4; ++i)
            r[i] = rocke_b_readfirstlane(b, g0[i]);
        out[0] = rocke_b_vec_pack(b, r, 4, rocke_i32());
        for(i = 0; i < 8; ++i)
            r[i] = rocke_b_readfirstlane(b, g1[i]);
        out[1] = rocke_b_vec_pack(b, r, 8, rocke_i32());
        for(i = 0; i < 4; ++i)
            r[i] = rocke_b_readfirstlane(b, g2[i]);
        out[2] = rocke_b_vec_pack(b, r, 4, rocke_i32());
    }
    out[3] = rocke_b_zero_vec(b, rocke_i32(), 4);
    out[4] = rocke_b_zero_vec(b, rocke_i32(), 8);
}

/* The GEMM part both stem kernels share: per-lane LDS read offsets, the
 * operand reads and the chunk loop over the dY channels. */
struct Gemm
{
    rocke_ir_builder_t* b;
    const rocke_type_t* io_type;
    const char* wmma_op_id;
    bool prefetch;
    int TM, TN, CK, NS, XP, WPITCH, XE, WEI, WEI0;
    V c0, c1, c8, c16;
    V lane, l16, lhalf, wave_c, wave_p;
    V lds, n_chunks;
    std::function<void(V)> issue; /* _issue(chunk) */
    std::vector<V> px_off;
    std::vector<V> wei_off;
};

/* px_off / tr_col / tr_row / wei_off. */
inline void operand_offsets(Gemm& g)
{
    rocke_ir_builder_t* b = g.b;
    g.px_off.assign(g.TN, NULL);
    for(int tn = 0; tn < g.TN; ++tn)
    {
        V x = ci(b, g.TN);
        V y = rocke_b_mul(b, g.wave_p, x);
        V z = ci(b, tn);
        V u = rocke_b_add(b, y, z);
        V v = rocke_b_mul(b, u, g.c16);
        V flat = rocke_b_add(b, v, g.l16);
        V xp = ci(b, g.XP);
        V p0 = rocke_b_mul(b, flat, xp);
        V p1 = rocke_b_mul(b, g.lhalf, g.c16);
        g.px_off[tn] = rocke_b_add(b, p0, p1);
    }
    V tr_col, tr_row;
    {
        V x = rocke_b_div(b, g.l16, g.c8);
        tr_col = rocke_b_mul(b, x, g.c8);
    }
    {
        V x = rocke_b_mul(b, g.lhalf, g.c16);
        V y = rocke_b_mod(b, g.lane, g.c8);
        tr_row = rocke_b_add(b, x, y);
    }
    g.wei_off.assign(g.TM, NULL);
    for(int m = 0; m < g.TM; ++m)
    {
        V col;
        {
            V x = ci(b, g.TM);
            V y = rocke_b_mul(b, g.wave_c, x);
            V z = ci(b, m);
            V u = rocke_b_add(b, y, z);
            col = rocke_b_mul(b, u, g.c16);
        }
        V x = ci(b, g.WPITCH);
        V y = rocke_b_mul(b, tr_row, x);
        V z = ci(b, g.WEI0);
        V u = rocke_b_add(b, tr_col, z);
        V v = rocke_b_add(b, col, u);
        g.wei_off[m] = rocke_b_add(b, y, v);
    }
}

/* _read_b(base, tn, off): the dY (B) fragment of pixel tile tn. */
inline V read_b(Gemm& g, V base, int tn, int off)
{
    rocke_ir_builder_t* b = g.b;
    V t = ci(b, off);
    V x = rocke_b_add(b, g.px_off[tn], t);
    V addr = rocke_b_add(b, base, x);
    V idx0[2] = {g.c0, addr};
    V lo = rocke_b_smem_load_vN(b, g.lds, idx0, 2, g.io_type, 8);
    V addr8 = rocke_b_add(b, addr, g.c8);
    V idx1[2] = {g.c0, addr8};
    V hi = rocke_b_smem_load_vN(b, g.lds, idx1, 2, g.io_type, 8);
    return rocke_b_vec_concat(b, lo, hi);
}

/* _read_a(base, m, off): the W^T (A) fragment of column tile m. */
inline V read_a(Gemm& g, V base, int m, int off)
{
    rocke_ir_builder_t* b = g.b;
    V t = ci(b, off);
    V x = rocke_b_add(b, g.wei_off[m], t);
    V addr = rocke_b_add(b, base, x);
    V idx0[2] = {g.c0, addr};
    V lo = rocke_b_ds_read_tr16_b128(b, g.lds, idx0, 2, g.io_type);
    V o = ci(b, 8 * g.WPITCH);
    V addr8 = rocke_b_add(b, addr, o);
    V idx1[2] = {g.c0, addr8};
    V hi = rocke_b_ds_read_tr16_b128(b, g.lds, idx1, 2, g.io_type);
    return rocke_b_vec_concat(b, lo, hi);
}

/* _frags(ks): TN B fragments, then TM A fragments. */
inline void frags(Gemm& g, V xb, V wb, int ks, std::vector<V>& af, std::vector<V>& bf)
{
    bf.assign(g.TN, NULL);
    af.assign(g.TM, NULL);
    for(int tn = 0; tn < g.TN; ++tn)
        bf[tn] = read_b(g, xb, tn, ks * kWmmaK);
    for(int m = 0; m < g.TM; ++m)
        af[m] = read_a(g, wb, m, ks * kWmmaK * g.WPITCH);
}

/* _chunk(kc, accs): publish chunk kc, refill the ring, multiply the chunk. */
inline void chunk(Gemm& g, V kc, std::vector<V>& accs)
{
    rocke_ir_builder_t* b = g.b;
    if(g.NS == 2)
    {
        rocke_b_s_wait_tensorcnt(b, 0);
    }
    else
    {
        V x = ci(b, g.NS - 2);
        V y = rocke_b_add(b, kc, x);
        V more = rocke_b_cmp_lt(b, y, g.n_chunks);
        rocke_if_else_t ie = rocke_b_scf_if_else(b, more);
        rocke_b_region_enter(b, ie.then_region);
        rocke_b_s_wait_tensorcnt(b, g.NS - 2);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        rocke_b_s_wait_tensorcnt(b, 0);
        rocke_b_region_leave(b);
    }
    rocke_b_sync_lds_only(b);
    {
        V x = ci(b, g.NS - 1);
        V nxt = rocke_b_add(b, kc, x);
        V cond = rocke_b_cmp_lt(b, nxt, g.n_chunks);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        g.issue(nxt);
        rocke_b_region_leave(b);
    }
    V stage, xb, wb;
    {
        V x = ci(b, g.NS);
        stage = rocke_b_mod(b, kc, x);
    }
    {
        V x = ci(b, g.XE);
        xb = rocke_b_mul(b, stage, x);
    }
    {
        V x = ci(b, g.WEI);
        wb = rocke_b_mul(b, stage, x);
    }
    int n_steps = g.CK / kWmmaK;
    rocke_b_s_setprio(b, 1);
    std::vector<V> af, bf, naf, nbf;
    if(g.prefetch)
        frags(g, xb, wb, 0, naf, nbf);
    for(int j = 0; j < n_steps; ++j)
    {
        if(g.prefetch)
        {
            af = naf;
            bf = nbf;
            if(j + 1 < n_steps)
                frags(g, xb, wb, j + 1, naf, nbf);
        }
        else
        {
            frags(g, xb, wb, j, af, bf);
        }
        for(int m = 0; m < g.TM; ++m)
            for(int tn = 0; tn < g.TN; ++tn)
            {
                int i = m * g.TN + tn;
                accs[i] = rocke_b_mma(b, g.wmma_op_id, af[m], bf[tn], accs[i], NULL, 0);
            }
    }
    rocke_b_s_setprio(b, 0);
}

/* The issue prologue and the runtime loop over dY channel chunks; returns
 * the TM * TN accumulators (empty on a builder error). */
inline std::vector<V> k_loop(Gemm& g)
{
    rocke_ir_builder_t* b = g.b;
    for(int st = 0; st < g.NS - 1; ++st)
    {
        V x = ci(b, st);
        V cond = rocke_b_cmp_lt(b, x, g.n_chunks);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        g.issue(ci(b, st));
        rocke_b_region_leave(b);
    }
    int n_acc = g.TM * g.TN;
    V zero_acc = rocke_b_zero_vec_f32(b, 8);
    std::vector<std::string> names(n_acc);
    std::vector<rocke_iter_arg_t> iargs(n_acc);
    for(int i = 0; i < n_acc; ++i)
    {
        names[i] = "acc" + std::to_string(i);
        iargs[i].name = names[i].c_str();
        iargs[i].init = zero_acc;
    }
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, g.c0, g.n_chunks, g.c1, iargs.data(), n_acc, "kc", false, true);
    if(!rocke_ir_builder_ok(b))
        return {};
    rocke_b_region_enter(b, loop.body);
    {
        std::vector<V> accs(loop.iter_vars, loop.iter_vars + n_acc);
        chunk(g, loop.iv, accs);
        rocke_b_scf_yield(b, accs.data(), n_acc);
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return {};
    return std::vector<V>(loop.op->results, loop.op->results + n_acc);
}

} // namespace rocke_dgstem

#endif /* ROCKE_INSTANCE_CONV_DGRAD_STEM_WMMA_INTERNAL_HPP */
