// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dgrad_pp_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_dgrad_pp_mfma.py: the gfx950 LDS-DMA
 * ping-pong MFMA direct dgrad (build_dgrad_pp_mfma), the dgrad's weight half
 * (in-kernel flipped and transposed filter) on the shared pipeline of
 * rocke/instance_conv_direct_pp_mfma_internal.hpp.
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_dgrad_pp_mfma.h"

#include <string.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "rocke/instance_conv_direct_pp_mfma_internal.hpp"
#include "rocke/instance_conv_direct_wgrad_internal.hpp"

using rocke_dwg::Em;
using rocke_dwg::V;

namespace
{

const int kBlockKs[2] = {128, 256};
const rocke_pp::PpFamily kFamily
    = {"dgrad_pp_mfma",
       kBlockKs,
       2,
       "block_k must be 128 or 256 (256-byte weight rows for the swizzle)",
       true};

rocke_pp::PpSpec pp_spec(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return {&s->problem, s->name, s->block_k, s->wave_size};
}

/* _tr_swz(b, row): 2 * (row % 4 + 4 * ((row // 8) % 2)). */
V tr_swz(const Em& e, V row)
{
    V lo = e.modi(row, 4);
    V hi = e.divi(row, 8);
    hi = e.modi(hi, 2);
    V t = e.muli(hi, 4);
    t = e.add(lo, t);
    return e.muli(t, 2);
}

/* _DgradWeights: the flipped KRSC filter staged as [64 reduction rows]
 * [block_k outputs] per tap, read back with transposing reads. */
struct DgradWeights : rocke_pp::PpWeights
{
    /* One rank-0 weight DMA piece: (row, offset, output ok, LDS byte offset). */
    struct WPiece
    {
        V row, off, k_ok, lds_off;
    };
    std::vector<WPiece> w_pieces;
    V row_len = NULL;
    std::map<std::tuple<int, int, int>, V> a_idx;

    void dma_geometry(rocke_pp::PpCtx& c) override
    {
        Em& e = c.e;
        const rocke_direct_conv_problem_t& p = *c.spec.problem;
        const int BK = c.spec.block_k;
        const int W_PER_WAVE = BK * rocke_pp::kCK * 2 / 1024 / rocke_pp::kRankWaves;
        const int GPR = BK / rocke_pp::kGran;
        const int RPI = 1024 / (BK * 2);
        row_len = e.muli(c.K, p.KH * p.KW);
        for(int i = 0; i < W_PER_WAVE; ++i)
        {
            V q = e.addi(e.muli(c.wave_g, W_PER_WAVE), i);
            V r = e.muli(q, RPI);
            V lr = e.divi(c.lane, GPR);
            V row = e.add(r, lr);
            V lg = e.modi(c.lane, GPR);
            V gran = e.bxor(lg, tr_swz(e, row));
            V cc = e.add(c.k0, e.muli(gran, rocke_pp::kGran));
            WPiece wp;
            wp.row = row;
            wp.off = e.add(e.mul(row, row_len), cc);
            wp.k_ok = e.lt(cc, c.K);
            wp.lds_off = e.muli(q, 1024);
            w_pieces.push_back(wp);
        }
    }

    void issue(rocke_pp::PpCtx& c, int buf, int tap, V c_base) override
    {
        Em& e = c.e;
        const int KH = c.spec.problem->KH, KW = c.spec.problem->KW;
        int r = tap / KW, s_ = tap % KW;
        int tap_f = (KH - 1 - r) * KW + (KW - 1 - s_);
        V m1 = e.mul(c_base, row_len);
        V kt = e.ci(tap_f);
        V m2 = e.mul(kt, c.K);
        V base = e.add(m1, m2);
        for(const WPiece& wp : w_pieces)
        {
            V a = e.add(c_base, wp.row);
            V ok = e.land(wp.k_ok, e.lt(a, c.C));
            V voff = e.mul(e.add(wp.off, base), c.c2);
            voff = e.select(ok, voff, c.oob);
            V lds = c.lds_at(c.w_addr[buf], wp.lds_off);
            rocke_b_async_buffer_load_lds(c.b, c.w_rsrc, lds, voff, c.c0, 4, 0);
        }
    }

    void a_geometry(rocke_pp::PpCtx& c) override
    {
        Em& e = c.e;
        const int BK = c.spec.block_k;
        const int WK = BK / 4;
        const int KF = WK / rocke_pp::kAtom;
        V tb = e.divi(c.lane, 16);
        V ti = e.modi(e.divi(c.lane, 4), 4);
        V tj = e.modi(c.lane, 4);
        for(int kf = 0; kf < KF; ++kf)
        {
            V m = e.muli(c.wave_k, WK);
            V kk = e.ci(kf * rocke_pp::kAtom);
            V m2 = e.muli(tj, 4);
            V col = e.add(m, e.add(kk, m2));
            for(int part = 0; part < rocke_pp::kPartsC; ++part)
            {
                for(int rd = 0; rd < 2; ++rd)
                {
                    V kr = e.ci(part * 32 + 4 * rd);
                    V mt = e.muli(tb, 8);
                    V row = e.add(e.add(kr, mt), ti);
                    V sl = e.divi(col, rocke_pp::kGran);
                    V slot = e.bxor(sl, tr_swz(e, row));
                    V mr = e.muli(row, BK);
                    V ms = e.muli(slot, rocke_pp::kGran);
                    V mc = e.modi(col, rocke_pp::kGran);
                    a_idx[{kf, part, rd}] = e.add(mr, e.add(ms, mc));
                }
            }
        }
        c.shared["lane_hi"] = tb;
    }

    V read_a(rocke_pp::PpCtx& c, int buf, int kf, int part) override
    {
        V lo = c.e.ds_read_tr(c.w_bufs[buf], c.c0, a_idx[{kf, part, 0}], c.io_type);
        V hi = c.e.ds_read_tr(c.w_bufs[buf], c.c0, a_idx[{kf, part, 1}], c.io_type);
        return rocke_b_vec_concat(c.b, lo, hi);
    }
};

} // namespace

extern "C" rocke_dgrad_pp_mfma_spec_t rocke_dgrad_pp_mfma_spec_default(void)
{
    rocke_dgrad_pp_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "dgrad_pp_mfma";
    s.block_k = 128;
    s.wave_size = 64;
    return s;
}

extern "C" int rocke_dgrad_pp_mfma_threads_per_block(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return rocke_pp::pp_threads_per_block(pp_spec(s));
}

extern "C" long rocke_dgrad_pp_mfma_lds_bytes(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return rocke_pp::pp_lds_bytes(pp_spec(s));
}

extern "C" void rocke_dgrad_pp_mfma_grid(const rocke_dgrad_pp_mfma_spec_t* s, int grid[3])
{
    rocke_pp::pp_grid(pp_spec(s), grid);
}

extern "C" rocke_status_t
    rocke_dgrad_pp_mfma_kernel_name(const rocke_dgrad_pp_mfma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    return rocke_pp::pp_kernel_name(pp_spec(s), out, out_cap);
}

extern "C" bool rocke_dgrad_pp_mfma_is_valid_spec(const rocke_dgrad_pp_mfma_spec_t* s,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    if(s == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    return rocke_pp::pp_is_valid_spec(pp_spec(s), kFamily, arch, reason, reason_cap);
}

extern "C" rocke_kernel_def_t* rocke_build_dgrad_pp_mfma(rocke_ir_builder_t* b,
                                                         const rocke_dgrad_pp_mfma_spec_t* spec,
                                                         const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_dgrad_pp_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid DgradPpMfmaSpec for ") + arch;
        rocke_dwg::fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    DgradWeights weights;
    return rocke_pp::emit_pp_mfma(b, pp_spec(spec), weights);
}

extern "C" rocke_kernel_def_t* rocke_build_dgrad_pp_mfma_new(rocke_ir_builder_t* b,
                                                             const rocke_dgrad_pp_mfma_spec_t* spec,
                                                             const char* arch)
{
    return rocke_dwg::build_new(
        b,
        spec != NULL,
        "build_dgrad_pp_mfma",
        [&](char* out, size_t cap) { return rocke_dgrad_pp_mfma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_dgrad_pp_mfma(b, spec, arch); });
}

extern "C" rocke_status_t rocke_dgrad_pp_mfma_lower_to_llvm(const rocke_dgrad_pp_mfma_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap)
{
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    return rocke_dwg::lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_dgrad_pp_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_dgrad_pp_mfma_new(b, spec, arch); });
}
