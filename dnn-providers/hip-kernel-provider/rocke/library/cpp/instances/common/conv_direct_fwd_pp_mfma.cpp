// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_pp_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_fwd_pp_mfma.py: the gfx950 LDS-DMA
 * ping-pong MFMA direct forward (build_fwd_pp_mfma), the forward's weight half
 * on the shared pipeline of rocke/instance_conv_direct_pp_mfma_internal.hpp.
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_fwd_pp_mfma.h"

#include <string.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "rocke/instance_conv_direct_pp_mfma_internal.hpp"
#include "rocke/instance_conv_direct_wgrad_internal.hpp"

using rocke_dwg::Em;
using rocke_dwg::V;

namespace
{

const int kBlockKs[3] = {64, 128, 256};
const rocke_pp::PpFamily kFamily
    = {"fwd_pp_mfma", kBlockKs, 3, "block_k must be 64, 128 or 256", false};

rocke_pp::PpSpec pp_spec(const rocke_fwd_pp_mfma_spec_t* s)
{
    return {&s->problem, s->name, s->block_k, s->wave_size};
}

/* _FwdWeights: KRSC weights staged as [block_k rows][64 channels] per tap. */
struct FwdWeights : rocke_pp::PpWeights
{
    /* One rank-0 weight DMA piece: (offset, output ok, LDS byte offset). */
    struct WPiece
    {
        V off, k_ok, lds_off;
    };
    std::vector<WPiece> w_pieces;
    V w_c = NULL;
    std::map<std::pair<int, int>, V> a_idx;

    void dma_geometry(rocke_pp::PpCtx& c) override
    {
        /* [BK output rows][64 channels] per tap; lane l of piece q writes row
         * 8q + l / 8, slot l % 8, logical granule (l % 8) ^ (l / 8). */
        Em& e = c.e;
        const rocke_direct_conv_problem_t& p = *c.spec.problem;
        const int W_PER_WAVE = c.spec.block_k * rocke_pp::kCK * 2 / 1024 / rocke_pp::kRankWaves;
        V l_row = e.divi(c.lane, 8);
        V l_gran = e.bxor(e.modi(c.lane, 8), l_row);
        w_c = e.muli(l_gran, rocke_pp::kGran);
        V taps_c = e.muli(c.C, p.KH * p.KW);
        for(int i = 0; i < W_PER_WAVE; ++i)
        {
            V q = e.addi(e.muli(c.wave_g, W_PER_WAVE), i);
            V kl = e.add(e.muli(q, 8), l_row);
            V k_glob = e.add(c.k0, kl);
            WPiece wp;
            wp.off = e.add(e.mul(k_glob, taps_c), w_c);
            wp.k_ok = e.lt(k_glob, c.K);
            wp.lds_off = e.muli(q, 1024);
            w_pieces.push_back(wp);
        }
        c.shared["l_row"] = l_row;
        c.shared["l_gran"] = l_gran;
        c.shared["lane_c"] = w_c;
    }

    void issue(rocke_pp::PpCtx& c, int buf, int tap, V c_base) override
    {
        Em& e = c.e;
        V kt = e.ci(tap);
        V m = e.mul(kt, c.C);
        V tap_off = e.add(m, c_base);
        V c_ok = e.lt(e.add(c_base, w_c), c.C);
        for(const WPiece& wp : w_pieces)
        {
            V ok = e.land(wp.k_ok, c_ok);
            V voff = e.mul(e.add(wp.off, tap_off), c.c2);
            voff = e.select(ok, voff, c.oob);
            V lds = c.lds_at(c.w_addr[buf], wp.lds_off);
            rocke_b_async_buffer_load_lds(c.b, c.w_rsrc, lds, voff, c.c0, 4, 0);
        }
    }

    void a_geometry(rocke_pp::PpCtx& c) override
    {
        /* A (weights): row k = wave_k * WK + kf * 16 + lane % 16, granule
         * part * 4 + lane / 16 at slot granule ^ (k % 8). */
        Em& e = c.e;
        const int WK = c.spec.block_k / 4;
        const int KF = WK / rocke_pp::kAtom;
        V l16 = e.modi(c.lane, 16);
        V lhi = e.divi(c.lane, 16);
        for(int kf = 0; kf < KF; ++kf)
        {
            V m = e.muli(c.wave_k, WK);
            V kk = e.ci(kf * rocke_pp::kAtom);
            V k_loc = e.add(m, e.add(kk, l16));
            for(int part = 0; part < rocke_pp::kPartsC; ++part)
            {
                V kg = e.ci(part * 4);
                V g = e.add(kg, lhi);
                V slot = e.bxor(g, e.modi(k_loc, 8));
                V mk = e.muli(k_loc, rocke_pp::kCK);
                a_idx[{kf, part}] = e.add(mk, e.muli(slot, rocke_pp::kGran));
            }
        }
        c.shared["lane_hi"] = lhi;
        c.shared["l16"] = l16;
    }

    V read_a(rocke_pp::PpCtx& c, int buf, int kf, int part) override
    {
        return c.e.smem_load(c.w_bufs[buf], c.c0, a_idx[{kf, part}], c.io_type, rocke_pp::kGran);
    }
};

} // namespace

extern "C" rocke_fwd_pp_mfma_spec_t rocke_fwd_pp_mfma_spec_default(void)
{
    rocke_fwd_pp_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "fwd_pp_mfma";
    s.block_k = 128;
    s.wave_size = 64;
    return s;
}

extern "C" int rocke_fwd_pp_mfma_threads_per_block(const rocke_fwd_pp_mfma_spec_t* s)
{
    return rocke_pp::pp_threads_per_block(pp_spec(s));
}

extern "C" long rocke_fwd_pp_mfma_lds_bytes(const rocke_fwd_pp_mfma_spec_t* s)
{
    return rocke_pp::pp_lds_bytes(pp_spec(s));
}

extern "C" void rocke_fwd_pp_mfma_grid(const rocke_fwd_pp_mfma_spec_t* s, int grid[3])
{
    rocke_pp::pp_grid(pp_spec(s), grid);
}

extern "C" rocke_status_t
    rocke_fwd_pp_mfma_kernel_name(const rocke_fwd_pp_mfma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    return rocke_pp::pp_kernel_name(pp_spec(s), out, out_cap);
}

extern "C" bool rocke_fwd_pp_mfma_is_valid_spec(const rocke_fwd_pp_mfma_spec_t* s,
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

extern "C" rocke_kernel_def_t* rocke_build_fwd_pp_mfma(rocke_ir_builder_t* b,
                                                       const rocke_fwd_pp_mfma_spec_t* spec,
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
    if(!rocke_fwd_pp_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid FwdPpMfmaSpec for ") + arch;
        rocke_dwg::fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    FwdWeights weights;
    return rocke_pp::emit_pp_mfma(b, pp_spec(spec), weights);
}

extern "C" rocke_kernel_def_t* rocke_build_fwd_pp_mfma_new(rocke_ir_builder_t* b,
                                                           const rocke_fwd_pp_mfma_spec_t* spec,
                                                           const char* arch)
{
    return rocke_dwg::build_new(
        b,
        spec != NULL,
        "build_fwd_pp_mfma",
        [&](char* out, size_t cap) { return rocke_fwd_pp_mfma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_fwd_pp_mfma(b, spec, arch); });
}

extern "C" rocke_status_t rocke_fwd_pp_mfma_lower_to_llvm(const rocke_fwd_pp_mfma_spec_t* spec,
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
        "build_fwd_pp_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_fwd_pp_mfma_new(b, spec, arch); });
}
