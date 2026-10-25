/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_layout_emit.c -- C-side emitter for the attention
 * backward host decision helpers that emit no IR (the caps, tiles and LDS
 * layout / planner helpers, and the ABI lists). Selects one config by argv[1]
 * (the same config list as attention_bwd_layout_emit.py, rebuilt by the same
 * loops) and prints the same text dump through the C ports, so run_diff can
 * byte-compare the two. Compiled as C++20 against librocke_core.a.
 *
 * Only mode "ll" (the dump) is supported.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_tiles.h"
#include "rocke/instance_attention_bwd.h"

typedef std::vector<std::string> Lines;
typedef rocke_attn_bwd_err_t Err;
using rocke_attn_bwd::py_repr;

static const int NONE = ROCKE_ATTN_BWD_NONE;
static const char* const ARCHES[] = {"gfx942", "gfx950", "gfx1151", "gfx1201"};

/* ------------------------------------------------------------- configs */

static const char* const PLAN_REQUESTS[][3] = {
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:lds_plain"},
    {"gfx942", "0", "head_size:128;k_m0:16;k_n0:128;waves:8;kt_source:lds"},
    {"gfx942",
     "1",
     "head_size:128;k_m0:16;k_n0:128;waves:8;kt_source:lds;transpose_source:lds_plain"},
    {"gfx942",
     "1",
     "head_size:64;k_m0:32;k_n0:64;waves:4;kv_residency:lds;kt_source:lds;lds_order:phase_low"},
    {"gfx942",
     "1",
     "head_size:32;k_m0:32;k_n0:128;waves:4;acc_in_lds:True;dkv_epilogue:lds_staged_pair"},
    {"gfx942",
     "1",
     "head_size:64;k_m0:32;k_n0:128;waves:8;ring_depth:2;transpose_source:lds_plain;lds_swizzle:q="
     "pad8,do=pad16"},
    {"gfx942",
     "1",
     "head_size:64;k_m0:32;k_n0:128;waves:8;dq_mode:split;pt_route:lds;transpose_source:lds_plain"},
    {"gfx942",
     "1",
     "head_size:64;k_m0:32;k_n0:128;waves:8;dq_mode:split;transpose_source:xt_lds;kt_source:reg"},
    {"gfx942",
     "1",
     "head_size:64;k_m0:16;k_n0:64;waves:4;pt_route:lds;ds_slots:2;dkv_epilogue:direct;transpose_"
     "source:lds_plain"},
    {"gfx942",
     "1",
     "head_size:64;k_m0:32;k_n0:128;waves:8;lds_swizzle: k_stage = xor , "
     "qt=pad32,dot=none,kt_stage=pad8,dkv_out=xor,dkv_out=pad16"},
    {"gfx942",
     "1",
     "head_size:128;k_m0:64;k_n0:256;waves:8;kv_residency:lds;kt_source:lds;transpose_source:lds_"
     "plain"},
    {"gfx942",
     "0",
     "head_size:128;k_m0:64;k_n0:256;waves:8;kv_residency:lds;kt_source:lds;transpose_source:lds_"
     "plain;lds_order:phase_low"},
    {"gfx950",
     "1",
     "head_size:64;k_m0:32;k_n0:256;waves:8;kt_source:lds;transpose_source:tr_read;atom_k_g4:32"},
    {"gfx950",
     "1",
     "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:tr_read;global_path:dma;ring_depth:3"},
    {"gfx950",
     "1",
     "head_size:128;k_m0:16;k_n0:128;waves:8;transpose_source:tr_read;global_path:dma;ring_depth:2;"
     "lds_swizzle:ds=pad32"},
    {"gfx950",
     "1",
     "head_size:32;k_m0:32;k_n0:128;waves:4;transpose_source:xt_lds;atom_k_g13:32;atom_k_g4:32;kt_"
     "source:lds"},
    {"gfx950",
     "1",
     "head_size:64;k_m0:16;k_n0:64;waves:4;transpose_source:lds_plain;kv_residency:lds;kt_source:"
     "lds;acc_in_lds:True;dq_mode:split"},
    {"gfx1201",
     "1",
     "head_size:64;k_m0:16;k_n0:16;waves:1;wave_size:32;kv_residency:lds;kt_source:lds;transpose_"
     "source:wmma_lds;pt_route:lds;acc_in_lds:True"},
    {"gfx1151",
     "1",
     "head_size:128;k_m0:16;k_n0:16;waves:1;wave_size:32;kv_residency:lds;kt_source:lds;transpose_"
     "source:xt_lds;pt_route:lds;acc_in_lds:True"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;dtype:fp32"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;kv_residency:hbm"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;kt_source:smem"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;kv_residency:lds"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:ldsx"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;ring_depth:4"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;global_path:tdm"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;global_path:dma"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;pt_route:regs"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;dq_mode:none"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;dkv_epilogue:lds"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;lds_order:high"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;ds_slots:3"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;lds_swizzle:q=pad4"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:128;waves:8;atom_k_g13:8"},
    {"gfx942", "1", "head_size:64;k_m0:0;k_n0:128;waves:8"},
    {"gfx942",
     "1",
     "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:lds_plain;lds_swizzle:q=xor"},
    {"gfx942", "1", "head_size:64;k_m0:32;k_n0:48;waves:8;lds_swizzle:kt_stage=xor"},
    {"gfx942",
     "1",
     "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:lds_plain;lds_swizzle:ds=xor"},
    {"gfx950",
     "1",
     "head_size:64;k_m0:16;k_n0:256;waves:8;transpose_source:tr_read;global_path:dma"},
};

static const int XT_SHAPES[][4] = {
    {64, 16, 2, 8},
    {64, 32, 2, 8},
    {128, 16, 2, 8},
    {32, 64, 2, 8},
    {128, 128, 2, 8},
    {64, 192, 2, 8},
    {64, 48, 2, 8},
    {64, 16, 2, 16},
    {16, 8, 2, 4},
    {0, 16, 2, 8},
    {64, 16, 2, 12},
    {64, 24, 2, 16},
    {32, 256, 2, 16},
    {16, 4, 2, 8},
    {16, 4, 2, 16},
    {16, 16, 3, 8},
};

static const int DMA_SHAPES[][3] = {
    {16, 64, 2},
    {32, 64, 2},
    {16, 128, 2},
    {64, 128, 2},
    {16, 32, 2},
    {8, 32, 2},
    {16, 40, 2},
    {16, 72, 2},
    {0, 64, 2},
    {16, 256, 2},
    {32, 16, 2},
    {16, 36, 2},
};

static std::vector<std::string> configs()
{
    std::vector<std::string> out;
    for(const char* a : ARCHES)
        out.push_back(std::string("caps|") + a);
    out.push_back("caps|gfx90a");
    for(const char* a : ARCHES)
        out.push_back(std::string("proofs|") + a);
    for(int w = 0; w < 9; ++w)
        out.push_back("tiles|" + std::to_string(w));
    for(int d : {32, 64, 128})
        out.push_back("small|" + std::to_string(d));
    for(const auto& s : XT_SHAPES)
        out.push_back("xt|" + std::to_string(s[0]) + "," + std::to_string(s[1]) + ","
                      + std::to_string(s[2]) + "," + std::to_string(s[3]));
    for(int n = 0; n < 3; ++n)
        out.push_back("writer|" + std::to_string(n));
    for(int r : {32, 64, 128})
        for(int c : {16, 32, 64, 128, 192, 256})
            out.push_back("choose|" + std::to_string(r) + "," + std::to_string(c));
    for(const auto& s : DMA_SHAPES)
        out.push_back("dma|" + std::to_string(s[0]) + "," + std::to_string(s[1]) + ","
                      + std::to_string(s[2]));
    for(const auto& p : PLAN_REQUESTS)
        out.push_back(std::string("plan|") + p[0] + "|" + p[1] + "|" + p[2]);
    out.push_back("abi|0");
    out.push_back("access|0");
    return out;
}

/* ------------------------------------------------------------- helpers */

static std::string exc(const Err& e)
{
    const char* n = "Error";
    switch(e.code)
    {
    case ROCKE_ERR_VALUE:
        n = "ValueError";
        break;
    case ROCKE_ERR_KEY:
        n = "KeyError";
        break;
    case ROCKE_ERR_TYPE:
        n = "TypeError";
        break;
    case ROCKE_ERR_NOTIMPL:
        n = "NotImplementedError";
        break;
    default:
        break;
    }
    return std::string(n) + ": " + e.msg;
}

static std::string B(bool v)
{
    return v ? "True" : "False";
}

static std::string I(int v)
{
    return std::to_string(v);
}

static std::string tuple_i(const int* v, int n)
{
    std::string out = "(";
    for(int i = 0; i < n; ++i)
        out += (i ? ", " : "") + I(v[i]);
    if(n == 1)
        out += ",";
    return out + ")";
}

static std::string tuple_s(const char* const* v, int n)
{
    std::string out = "(";
    for(int i = 0; i < n; ++i)
        out += (i ? ", " : "") + py_repr(v[i]);
    if(n == 1)
        out += ",";
    return out + ")";
}

static std::string grid(const rocke_attn_bwd_grid_t& g)
{
    return "(" + I(g.rows) + ", " + I(g.cols) + ")";
}

static std::string grids(const rocke_attn_bwd_grid_t* g, int n)
{
    std::string out = "(";
    for(int i = 0; i < n; ++i)
        out += (i ? ", " : "") + grid(g[i]);
    if(n == 1)
        out += ",";
    return out + ")";
}

/* "label: value" or "label: <exc>" (Python's _try). */
static void
    attempt(Lines& lines, const std::string& label, const std::function<std::string(Err*)>& fn)
{
    Err e;
    e.code = ROCKE_OK;
    std::string v = fn(&e);
    lines.push_back(label + ": " + (e.code == ROCKE_OK ? v : exc(e)));
}

static std::vector<int> split_ints(const std::string& s)
{
    std::vector<int> out;
    size_t start = 0;
    while(true)
    {
        size_t c = s.find(',', start);
        out.push_back(
            atoi(s.substr(start, c == std::string::npos ? std::string::npos : c - start).c_str()));
        if(c == std::string::npos)
            break;
        start = c + 1;
    }
    return out;
}

/* ---------------------------------------------------------------- caps */

static Lines dump_caps(const std::string& arch)
{
    Lines lines;
    Err e;
    rocke_attn_bwd_caps_t c;
    if(rocke_attn_bwd_arch_caps(arch.c_str(), &c, &e) != ROCKE_OK)
        lines.push_back("caps: " + exc(e));
    else
        lines.push_back("caps: wave=" + I(c.wave_size) + " path=" + c.matrix_path
                        + " lds=" + I(c.lds_capacity_bytes)
                        + " dma=" + tuple_i(c.lds_dma_dwords, c.n_lds_dma_dwords)
                        + " max_dma=" + I(rocke_attn_bwd_caps_max_lds_dma_dwords(&c))
                        + " tr64=" + B(c.has_tr_read_b64) + " tr128=" + B(c.has_tr_read_b128)
                        + " has_agprs=" + B(c.has_agprs) + " vgprs=" + I(c.arch_vgprs)
                        + " agprs=" + I(c.agprs) + " atomic=" + B(c.native_fp32_atomic_add)
                        + " waves=" + tuple_i(c.legal_waves, c.n_legal_waves));
    const char* a = arch.c_str();
    for(const char* dt : {"fp16", "f16", "bf16", "fp32"})
    {
        const rocke_mma_op_t* ops[8];
        int n = 0;
        if(rocke_attn_bwd_atoms(a, dt, ops, 8, &n, &e) != ROCKE_OK)
            lines.push_back(std::string("atoms ") + dt + ": " + exc(e));
        else
        {
            std::string s = std::string("atoms ") + dt + ":";
            for(int i = 0; i < n; ++i)
                s += std::string(" ") + ops[i]->op_id + "/k" + I(ops[i]->k) + "/c"
                     + I(ops[i]->c_frag_len) + "/w" + I(ops[i]->wave_size);
            lines.push_back(s);
        }
        for(int k : {16, 32, 8})
            attempt(lines, std::string("select ") + dt + " k" + I(k), [&](Err* er) {
                const rocke_mma_op_t* op = rocke_attn_bwd_select_atom(a, dt, k, er);
                return op ? std::string(op->op_id) : std::string();
            });
        attempt(lines, std::string("wide ") + dt, [&](Err* er) {
            bool v = false;
            rocke_attn_bwd_wide_k_atom_available(a, dt, &v, er);
            return B(v);
        });
    }
    for(int w = 0; w < 10; ++w)
        attempt(lines, "check_waves " + I(w), [&](Err* er) {
            rocke_attn_bwd_check_waves(a, w, er);
            return std::string("None");
        });
    for(long long n : {-1LL, 0LL, 65536LL, 65537LL, 163840LL, 163841LL})
        attempt(lines, "check_lds_bytes " + std::to_string(n), [&](Err* er) {
            rocke_attn_bwd_check_lds_bytes(a, n, er);
            return std::string("None");
        });
    for(int d = 0; d < 6; ++d)
        attempt(lines, "check_lds_dma_width " + I(d), [&](Err* er) {
            rocke_attn_bwd_check_lds_dma_width(a, d, er);
            return std::string("None");
        });
    for(int bits : {32, 64, 128})
        attempt(lines, "check_transpose_read " + I(bits), [&](Err* er) {
            rocke_attn_bwd_check_transpose_read(a, bits, er);
            return std::string("None");
        });
    for(int k : {16, 32, 8})
    {
        int t = rocke_attn_bwd_agpr_asm_tail_nop(k);
        lines.push_back("AGPR_ASM_TAIL_NOP " + I(k) + ": " + (t < 0 ? std::string("None") : I(t)));
        attempt(lines, "wait_states k" + I(k), [&](Err* er) {
            return I(rocke_attn_bwd_mfma_result_wait_states(a, k, er));
        });
        for(const char* dt : {"fp16", "bf16", "fp32"})
        {
            attempt(lines, std::string("mnemonic ") + dt + " k" + I(k), [&](Err* er) {
                const char* m = rocke_attn_bwd_mfma_asm_mnemonic(a, dt, k, er);
                return std::string(m ? m : "");
            });
            attempt(lines, std::string("nop_min ") + dt + " k" + I(k), [&](Err* er) {
                return I(rocke_attn_bwd_asm_tail_nop_min(a, dt, k, er));
            });
            for(int nop : {5, 6, 7, 15, 16})
                attempt(lines,
                        std::string("check_nop ") + dt + " k" + I(k) + " " + I(nop),
                        [&](Err* er) {
                            rocke_attn_bwd_check_asm_tail_nop(a, dt, k, nop, er);
                            return std::string("None");
                        });
        }
    }
    lines.push_back("VALU_LEAD: " + I(ROCKE_ATTN_BWD_ASM_MFMA_VALU_LEAD_WAIT_STATES));
    std::vector<rocke_attn_bwd_agpr_alloc_t> allocs;
    {
        rocke_attn_bwd_agpr_alloc_t x;
        memset(&x, 0, sizeof(x));
        x.kind = ROCKE_ATTN_BWD_AGPR_NONE;
        allocs.push_back(x);
        x.kind = ROCKE_ATTN_BWD_AGPR_STR;
        x.str = "backend";
        allocs.push_back(x);
        const int pairs[][2] = {{0, 0}, {128, 128}, {256, 256}, {64, 64}};
        for(const auto& p : pairs)
        {
            rocke_attn_bwd_agpr_alloc_t y;
            memset(&y, 0, sizeof(y));
            y.kind = ROCKE_ATTN_BWD_AGPR_PAIR;
            y.a = p[0];
            y.b = p[1];
            allocs.push_back(y);
        }
        rocke_attn_bwd_agpr_alloc_t z;
        memset(&z, 0, sizeof(z));
        z.kind = ROCKE_ATTN_BWD_AGPR_STR;
        z.str = "auto";
        allocs.push_back(z);
        z.kind = ROCKE_ATTN_BWD_AGPR_PAIR;
        z.a = 128;
        z.b = 512;
        allocs.push_back(z);
    }
    for(const auto& al : allocs)
        attempt(lines, "check_agpr_alloc " + rocke_attn_bwd::agpr_alloc_repr(al), [&](Err* er) {
            rocke_attn_bwd_check_agpr_alloc(a, al, er);
            return std::string("None");
        });
    rocke_attn_bwd_arch_facts_t f;
    if(rocke_attn_bwd_arch_facts(a, &f, &e) != ROCKE_OK)
        lines.push_back("facts: " + exc(e));
    else
    {
        int iw[4], ir[4];
        const char *sa[4], *st[4], *sp[4];
        int nw = rocke_attn_bwd_facts_legal_waves(&f, iw);
        int na = rocke_attn_bwd_facts_legal_atoms(&f, sa);
        int nt = rocke_attn_bwd_facts_legal_transpose_sources(&f, st);
        int np = rocke_attn_bwd_facts_legal_global_paths(&f, sp);
        int nr = rocke_attn_bwd_facts_legal_ring_depths(&f, ir);
        lines.push_back("facts: wave=" + I(f.wave_size) + " path=" + f.matrix_path + " lds="
                        + I(f.lds_capacity_bytes) + " vgprs=" + I(f.arch_vgprs) + " agprs="
                        + I(f.agprs) + " tr=" + B(f.has_tr_read) + " dma=" + B(f.has_lds_dma)
                        + " wide=" + B(f.wide_k_atom) + " waves=" + tuple_i(iw, nw)
                        + " atoms=" + tuple_s(sa, na) + " transposes=" + tuple_s(st, nt)
                        + " paths=" + tuple_s(sp, np) + " rings=" + tuple_i(ir, nr));
    }
    for(const char* dt : {"fp16", "bf16"})
    {
        const rocke_mma_op_t* ops[8];
        int n = 0;
        if(rocke_attn_bwd_atoms(a, dt, ops, 8, &n, &e) != ROCKE_OK)
            continue;
        for(int i = 0; i < n; ++i)
        {
            std::vector<int> ok;
            for(int w : {1, 2, 4})
                if(rocke_attn_bwd::stats_vec_layout_ok(ops[i], w))
                    ok.push_back(w);
            std::string l = "[";
            for(size_t j = 0; j < ok.size(); ++j)
                l += (j ? ", " : "") + I(ok[j]);
            l += "]";
            lines.push_back(std::string("c_facts ") + ops[i]->op_id + ": stats_vecs=" + l
                            + " quad=" + B(rocke_attn_bwd::c_quad_transposable(ops[i])));
            for(int w : {1, 2, 3, 4, 8})
                attempt(lines,
                        std::string("check_stats_vec_layout ") + ops[i]->op_id + " " + I(w),
                        [&](Err* er) {
                            try
                            {
                                rocke_attn_bwd::check_stats_vec_layout(ops[i], w);
                            }
                            catch(const ckc::Error& x)
                            {
                                rocke_attn_bwd::store_err(er, x.code(), x.what());
                            }
                            return std::string("None");
                        });
        }
    }
    return lines;
}

/* -------------------------------------------------------------- proofs */

static Lines dump_proofs(const std::string& arch)
{
    Lines lines;
    Err e;
    rocke_attn_bwd_arch_facts_t f;
    rocke_attn_bwd_arch_facts(arch.c_str(), &f, &e);
    bool mfma = strcmp(f.matrix_path, "mfma") == 0;
    int w = mfma ? 4 : 2;
    const char *atoms[4], *ts[4];
    int na = rocke_attn_bwd_facts_legal_atoms(&f, atoms);
    int nt = rocke_attn_bwd_facts_legal_transpose_sources(&f, ts);
    std::vector<const char*> routes
        = mfma ? std::vector<const char*>{"relabel", "lds"} : std::vector<const char*>{"lds"};
    const rocke_attn_bwd_grid_t gpairs[2][2] = {{{1, w}, {w, 1}}, {{2, w / 2}, {2, w / 2}}};
    for(const char* dt : {"fp16", "bf16"})
        for(int i02 = 0; i02 < na; ++i02)
            for(int i13 = 0; i13 < na; ++i13)
                for(int i4 = 0; i4 < na; ++i4)
                    for(int it = 0; it < nt; ++it)
                        for(const char* route : routes)
                            for(const auto& gp : gpairs)
                            {
                                if(strcmp(route, "relabel") == 0
                                   && !(gp[0].rows == 1 && gp[0].cols == w))
                                    continue;
                                rocke_attn_bwd_spec_t s = rocke_attn_bwd_spec_default();
                                s.head_size = 64;
                                s.dtype = dt;
                                s.waves = w;
                                s.block_m = 32;
                                s.block_n = 64;
                                s.block_k4 = 32;
                                s.warp_grid_g4 = {true, 1, w};
                                s.atom_g02 = atoms[i02];
                                s.atom_g13 = atoms[i13];
                                s.atom_g4 = atoms[i4];
                                s.pt_route = route;
                                s.transpose_source = ts[it];
                                s.kv_residency = mfma ? "reg" : "lds";
                                s.kt_source = mfma ? "reg" : "lds";
                                s.acc_in_lds = mfma ? 0 : 1;
                                s.ring_depth = 1;
                                s.global_path = "vgpr";
                                s.sched = "none";
                                s.warp_grid_g02 = {true, gp[0].rows, gp[0].cols};
                                s.warp_grid_g13 = {true, gp[1].rows, gp[1].cols};
                                std::string label = std::string(dt) + " " + atoms[i02] + "/"
                                                    + atoms[i13] + "/" + atoms[i4] + " " + ts[it]
                                                    + " " + route + " " + grid(gp[0]);
                                rocke_attn_bwd_spec_t r;
                                if(rocke_attn_bwd_spec_init(&s, &e) != ROCKE_OK
                                   || rocke_attn_bwd_spec_resolved(&s, arch.c_str(), &r, &e)
                                          != ROCKE_OK
                                   || rocke_attn_bwd_tile_geometry_check(&r, arch.c_str(), &e)
                                          != ROCKE_OK)
                                    lines.push_back("geometry " + label + ": " + exc(e));
                                else
                                    lines.push_back("geometry " + label + ": ok");
                            }
    return lines;
}

/* --------------------------------------------------------------- tiles */

static std::vector<rocke_attn_bwd_grid_t> g02s(int w)
{
    rocke_attn_bwd_grid_t g[ROCKE_ATTN_BWD_GRIDS_MAX];
    int n = 0;
    Err e;
    rocke_attn_bwd_g02_grids(w, g, &n, &e);
    return std::vector<rocke_attn_bwd_grid_t>(g, g + n);
}

static std::vector<rocke_attn_bwd_grid_t> g13s(int w)
{
    rocke_attn_bwd_grid_t g[ROCKE_ATTN_BWD_GRIDS_MAX];
    int n = 0;
    Err e;
    rocke_attn_bwd_g13_grids(w, g, &n, &e);
    return std::vector<rocke_attn_bwd_grid_t>(g, g + n);
}

static bool same(const rocke_attn_bwd_grid_t& a, const rocke_attn_bwd_grid_t& b)
{
    return a.rows == b.rows && a.cols == b.cols;
}

static Lines dump_tiles(int w)
{
    Lines lines;
    std::vector<rocke_attn_bwd_grid_t> a = g02s(w), b = g13s(w);
    rocke_attn_bwd_grid_t n02, n13;
    rocke_attn_bwd_natural_grids(w, &n02, &n13);
    lines.push_back("g02_grids: " + grids(a.data(), (int)a.size()));
    lines.push_back("g13_grids: " + grids(b.data(), (int)b.size()));
    lines.push_back("natural: (" + grid(n02) + ", " + grid(n13) + ")");
    std::vector<rocke_attn_bwd_grid_t> gs = a;
    gs.push_back({3, 1});
    gs.push_back({1, 16});
    for(const auto& g : gs)
        lines.push_back("grid " + grid(g) + ": waves=" + I(rocke_attn_bwd_grid_waves(g))
                        + " split=" + I(rocke_attn_bwd_kv_operand_split(g, w))
                        + " replicated=" + B(rocke_attn_bwd_replicated_g02(g, w)));
    std::vector<rocke_attn_bwd_grid_t> cands;
    std::vector<rocke_attn_bwd_grid_t> all = a;
    all.insert(all.end(), b.begin(), b.end());
    all.push_back({4, 4});
    all.push_back({1, 3});
    for(const auto& g : all)
    {
        bool dup = false;
        for(const auto& c : cands)
            dup = dup || same(c, g);
        if(!dup)
            cands.push_back(g);
    }
    const std::pair<const char*, rocke_attn_bwd_grid_t> names[]
        = {{"warp_grid_g02", n02}, {"warp_grid_g13", n13}};
    for(const auto& nm : names)
        for(const auto& g : cands)
            for(const auto& rc : {std::pair<int, int>{16, 16}, {32, 64}, {64, 128}})
                for(const char* route : {"lds", "relabel"})
                    attempt(lines,
                            std::string("route ") + nm.first + " " + grid(g) + " " + I(rc.first)
                                + "x" + I(rc.second) + " " + route,
                            [&](Err* er) {
                                rocke_attn_bwd_check_lds_route_grid(
                                    nm.first, g, nm.second, w, rc.first, rc.second, route, er);
                                return std::string("None");
                            });
    return lines;
}

static Lines dump_small(int d)
{
    Lines lines;
    for(int km : {16, 32, 64})
        for(int kn : {16, 32, 64, 128, 192, 256})
            for(int w : {1, 2, 4, 8})
            {
                rocke_attn_bwd_grid_triple_t t[ROCKE_ATTN_BWD_TRIPLES_MAX];
                int n = 0;
                Err e;
                rocke_attn_bwd_small_tile_grids(d, km, kn, w, t, &n, &e);
                std::string s = "(";
                for(int i = 0; i < n; ++i)
                    s += std::string(i ? ", " : "") + "(" + grid(t[i].g02) + ", " + grid(t[i].g13)
                         + ", " + grid(t[i].g4) + ")";
                if(n == 1)
                    s += ",";
                s += ")";
                lines.push_back("small " + I(km) + "x" + I(kn) + " w" + I(w) + ": " + s);
            }
    return lines;
}

/* ------------------------------------------------------------------ xt */

static std::string join_ints(const std::vector<int>& v)
{
    std::string s;
    for(size_t i = 0; i < v.size(); ++i)
        s += (i ? ", " : "") + I(v[i]);
    return s;
}

static Lines dump_xt(int rows, int cols, int eb, int chunk)
{
    Lines lines;
    Err e;
    for(const char* sw : {"xor", "none", "pad8", "pad16", "pad32", "pad12"})
    {
        rocke_attn_bwd_xt_layout_t l;
        if(rocke_attn_bwd_xt_layout_init(&l, rows, cols, eb, sw, chunk, &e) != ROCKE_OK)
        {
            lines.push_back(std::string("xt ") + sw + ": " + exc(e));
            continue;
        }
        lines.push_back(std::string("xt ") + sw
                        + ": pitch=" + I(rocke_attn_bwd_xt_layout_row_pitch(&l))
                        + " size=" + I(rocke_attn_bwd_xt_layout_size(&l)));
        std::vector<int> offs;
        for(int r = 0; r < rows; r += std::max(1, rows / 16))
            for(int c = 0; c < cols; c += std::max(1, cols / 16))
                offs.push_back(rocke_attn_bwd_xt_layout_offset(&l, r, c, &e));
        lines.push_back(std::string("xt ") + sw + " offsets: " + join_ints(offs));
        const int rc[][2] = {{rows, 0}, {0, cols}, {-1, 0}};
        for(const auto& p : rc)
            attempt(
                lines,
                std::string("xt ") + sw + " offset " + I(p[0]) + "," + I(p[1]),
                [&](Err* er) { return I(rocke_attn_bwd_xt_layout_offset(&l, p[0], p[1], er)); });
        for(int c : {0, 1, 2, 4})
            for(int n : {1, 2, 4, 8})
                attempt(
                    lines, std::string("xt ") + sw + " vector " + I(c) + "," + I(n), [&](Err* er) {
                        rocke_attn_bwd_xt_layout_check_vector(&l, c, n, er);
                        return std::string("None");
                    });
    }
    rocke_attn_bwd_row_xor_layout_t rx;
    if(rocke_attn_bwd_row_xor_layout_init(&rx, rows, cols, eb, chunk, &e) != ROCKE_OK)
        lines.push_back("rowxor: " + exc(e));
    else
    {
        lines.push_back("rowxor: pitch=" + I(rocke_attn_bwd_row_xor_layout_row_pitch(&rx))
                        + " size=" + I(rocke_attn_bwd_row_xor_layout_size(&rx)));
        std::vector<int> offs;
        for(int r = 0; r < rows; r += std::max(1, rows / 16))
            for(int c = 0; c < cols; c += std::max(1, cols / 16))
                offs.push_back(rocke_attn_bwd_row_xor_layout_offset(&rx, r, c, &e));
        lines.push_back("rowxor offsets: " + join_ints(offs));
        attempt(lines, "rowxor vector 4,8", [&](Err* er) {
            rocke_attn_bwd_row_xor_layout_check_vector(&rx, 4, 8, er);
            return std::string("None");
        });
    }
    return lines;
}

static Lines dump_writer(int group)
{
    Lines lines;
    Err e;
    if(group == 0)
    {
        const int cases[][7] = {
            {16, 64, 256, 4, 2, 16, 2},
            {32, 128, 512, 8, 4, 16, 2},
            {16, 64, 64, 4, 2, 8, 2},
            {16, 64, 256, 3, 2, 16, 2},
            {16, 64, 256, 4, 3, 16, 2},
            {16, 60, 256, 4, 2, 16, 2},
            {16, 64, 256, 16, 2, 16, 2},
            {16, 64, 256, 4, 16, 16, 2},
            {16, 64, 256, 4, 2, 12, 2},
            {16, 64, 0, 4, 2, 16, 2},
            {128, 32, 64, 8, 2, 4, 2},
            {16, 128, 128, 2, 4, 16, 2},
        };
        for(const auto& c : cases)
        {
            std::string args = "(";
            for(int i = 0; i < 7; ++i)
                args += (i ? ", " : "") + I(c[i]);
            args += ")";
            rocke_attn_bwd_xt_writer_t w;
            if(rocke_attn_bwd_xt_writer_init(&w, c[0], c[1], c[2], c[3], c[4], c[5], c[6], &e)
               != ROCKE_OK)
            {
                lines.push_back("writer " + args + ": " + exc(e));
                continue;
            }
            lines.push_back("writer " + args + ": n_dg=" + I(rocke_attn_bwd_xt_writer_n_dg(&w))
                            + " n_sg=" + I(rocke_attn_bwd_xt_writer_n_sg(&w))
                            + " d_lanes=" + I(rocke_attn_bwd_xt_writer_d_lanes_eff(&w))
                            + " units=" + I(rocke_attn_bwd_xt_writer_units(&w))
                            + " it=" + I(rocke_attn_bwd_xt_writer_iterations(&w))
                            + " wb=" + I(rocke_attn_bwd_xt_writer_write_bytes(&w))
                            + " stores=" + I(rocke_attn_bwd_xt_writer_stores_per_thread(&w))
                            + " loads=" + I(rocke_attn_bwd_xt_writer_loads_per_thread(&w)));
            std::string s = "coords:";
            for(int it = 0; it < rocke_attn_bwd_xt_writer_iterations(&w) + 1; ++it)
                for(int t = 0; t < w.threads; t += 7)
                {
                    int s0, d0;
                    if(rocke_attn_bwd_xt_writer_unit_coords(&w, t, it, &s0, &d0))
                        s += " " + I(s0) + "," + I(d0);
                    else
                        s += " -";
                }
            lines.push_back(s);
        }
        return lines;
    }
    std::vector<int> threads
        = group == 1 ? std::vector<int>{64, 128, 256, 512} : std::vector<int>{32, 96, 1024};
    for(int seq : {16, 32, 64, 128, 192, 256})
        for(int depth : {32, 64, 128})
            for(int t : threads)
                for(int mvb : {16, 8})
                {
                    std::string label
                        = "default " + I(seq) + "," + I(depth) + "," + I(t) + "," + I(mvb);
                    rocke_attn_bwd_xt_writer_t w;
                    if(rocke_attn_bwd_default_xt_writer(&w, seq, depth, t, 2, mvb, &e) != ROCKE_OK)
                        lines.push_back(label + ": " + exc(e));
                    else
                        lines.push_back(label + ": vec_d=" + I(w.vec_d)
                                        + " kpt=" + I(w.k_per_thread) + " d_lanes=" + I(w.d_lanes));
                }
    return lines;
}

static Lines dump_choose(int rows, int cols)
{
    Lines lines;
    Err e;
    for(const char* arch : {"gfx942", "gfx950", "gfx1201", "gfx90a"})
    {
        for(int threads : {64, 96, 128, 256, 512})
            for(int atom_k : {16, 32})
            {
                std::string label = std::string(arch) + " t" + I(threads) + " k" + I(atom_k);
                rocke_attn_bwd_xt_writer_t w;
                const char* best = NULL;
                rocke_attn_bwd_xt_choice_t table[ROCKE_ATTN_BWD_XT_CANDIDATES_MAX];
                int n = 0;
                if(rocke_attn_bwd_default_xt_writer(&w, cols, rows, threads, 2, 16, &e) != ROCKE_OK
                   || rocke_attn_bwd_choose_xt_swizzle(
                          rows, cols, &w, arch, NULL, 0, 2, atom_k, &best, table, &n, &e)
                          != ROCKE_OK)
                {
                    lines.push_back("choose " + label + ": " + exc(e));
                    continue;
                }
                std::string s = "choose " + label + ": " + best;
                for(int i = 0; i < n; ++i)
                {
                    const rocke_attn_bwd_xt_choice_t& t = table[i];
                    s += std::string(" ") + t.swizzle + "=";
                    if(t.illegal)
                        s += std::string("illegal(") + t.illegal_msg + ")";
                    else
                        s += (t.has_prediction ? "r" + I(t.read) + "w" + I(t.write)
                                               : std::string("None"))
                             + "/" + I(t.bytes);
                }
                lines.push_back(s);
                attempt(lines, "planner " + label, [&](Err* er) {
                    const char* p = rocke_attn_bwd_planner_xt_swizzle(
                        rows, cols, threads, arch, 2, atom_k, er);
                    return std::string(p ? p : "");
                });
            }
        rocke_attn_bwd_xt_layout_t lay;
        rocke_attn_bwd_xt_layout_init(&lay, rows, cols, 2, "xor", 8, &e);
        attempt(lines, std::string("xt_conflicts ") + arch + " no writer", [&](Err* er) {
            bool has = false;
            int r = 0, wr = 0;
            rocke_attn_bwd_xt_conflicts(&lay, NULL, arch, 16, &has, &r, &wr, er);
            return has ? "{'read': " + I(r) + ", 'write': " + I(wr) + "}" : std::string("None");
        });
    }
    return lines;
}

/* ----------------------------------------------------------------- dma */

static Lines dump_dma(int rows, int cols, int eb)
{
    Lines lines;
    Err e;
    rocke_attn_bwd_dma_slab_layout_t l;
    if(rocke_attn_bwd_dma_slab_layout_init(&l, rows, cols, eb, &e) != ROCKE_OK)
    {
        lines.push_back("dma: " + exc(e));
        return lines;
    }
    int size = rocke_attn_bwd_dma_slab_layout_size(&l);
    lines.push_back("dma: row_bytes=" + I(rocke_attn_bwd_dma_slab_layout_row_bytes(&l))
                    + " fold=" + I(rocke_attn_bwd_dma_slab_layout_fold(&l))
                    + " slabs=" + I(rocke_attn_bwd_dma_slab_layout_slabs(&l)) + " size=" + I(size));
    std::vector<int> offs;
    for(int r = 0; r < rows; ++r)
        for(int c = 0; c < cols; c += 8)
            offs.push_back(rocke_attn_bwd_dma_slab_layout_offset(&l, r, c, &e));
    lines.push_back("dma offsets: " + join_ints(offs));
    attempt(lines, "dma offset out", [&](Err* er) {
        return I(rocke_attn_bwd_dma_slab_layout_offset(&l, rows, 0, er));
    });
    std::string src = "dma sources:";
    for(int s = 0; s < size / 16; ++s)
    {
        int r = 0, c = 0;
        rocke_attn_bwd_dma_slab_layout_dma_source(&l, s, &r, &c, &e);
        src += " " + I(r) + "," + I(c);
    }
    lines.push_back(src);
    attempt(lines, "dma source out", [&](Err* er) {
        int r = 0, c = 0;
        rocke_attn_bwd_dma_slab_layout_dma_source(&l, size / 16, &r, &c, er);
        return "(" + I(r) + ", " + I(c) + ")";
    });
    for(int atom_k : {16, 32})
    {
        rocke_attn_bwd_lds_access_t acc[64];
        int n = 0;
        if(rocke_attn_bwd_dma_slab_read_accesses(&l, 0, 0, atom_k, 64, acc, &n, &e) != ROCKE_OK)
            lines.push_back("dma reads k" + I(atom_k) + ": " + exc(e));
        else
        {
            std::string s = "dma reads k" + I(atom_k) + ":";
            for(int i = 0; i < n; ++i)
                s += " " + I(acc[i].lane) + ":" + I(acc[i].byte) + ":" + I(acc[i].width);
            lines.push_back(s);
        }
        for(const char* arch : {"gfx942", "gfx950", "gfx1151"})
            attempt(lines, std::string("dma conflicts ") + arch + " k" + I(atom_k), [&](Err* er) {
                bool has = false;
                int worst = 0;
                rocke_attn_bwd_dma_slab_read_conflicts(&l, arch, atom_k, &has, &worst, er);
                return has ? I(worst) : std::string("None");
            });
    }
    return lines;
}

/* -------------------------------------------------------------- access */

static std::string
    accesses(const std::string& label,
             const std::function<rocke_status_t(rocke_attn_bwd_lds_access_t*, int*, Err*)>& fn)
{
    rocke_attn_bwd_lds_access_t acc[64];
    int n = 0;
    Err e;
    if(fn(acc, &n, &e) != ROCKE_OK)
        return label + ": " + exc(e);
    std::string s = label + ":";
    for(int i = 0; i < n; ++i)
        s += " " + I(acc[i].lane) + ":" + I(acc[i].byte) + ":" + I(acc[i].width);
    return s;
}

static Lines dump_access()
{
    Lines lines;
    Err e;
    rocke_attn_bwd_xt_layout_t lay;
    rocke_attn_bwd_xt_layout_init(&lay, 64, 16, 2, "xor", 8, &e);
    for(int ws : {64, 32})
        for(int k : {16, 32})
            lines.push_back(accesses("xt reads w" + I(ws) + " k" + I(k),
                                     [&](rocke_attn_bwd_lds_access_t* a, int* n, Err* er) {
                                         return rocke_attn_bwd_xt_read_accesses(
                                             &lay, 16, 0, k, ws, a, n, er);
                                     }));
    rocke_attn_bwd_xt_writer_t wr;
    rocke_attn_bwd_default_xt_writer(&wr, 16, 64, 256, 2, 16, &e);
    const int waves[][2] = {{0, 64}, {3, 64}, {1, 32}, {7, 32}};
    for(const auto& w : waves)
        lines.push_back(accesses("xt writes wave" + I(w[0]) + " w" + I(w[1]),
                                 [&](rocke_attn_bwd_lds_access_t* a, int* n, Err* er) {
                                     return rocke_attn_bwd_xt_write_accesses(
                                         &lay, &wr, w[0], 0, 1, w[1], a, n, er);
                                 }));
    const int others[][2] = {{32, 64}, {16, 32}};
    for(const auto& sd : others)
    {
        rocke_attn_bwd_xt_writer_t other;
        rocke_attn_bwd_default_xt_writer(&other, sd[0], sd[1], 256, 2, 16, &e);
        lines.push_back(accesses("xt writes " + I(sd[0]) + "x" + I(sd[1]),
                                 [&](rocke_attn_bwd_lds_access_t* a, int* n, Err* er) {
                                     return rocke_attn_bwd_xt_write_accesses(
                                         &lay, &other, 0, 0, 0, 64, a, n, er);
                                 }));
    }
    rocke_attn_bwd_dma_slab_layout_t slab;
    rocke_attn_bwd_dma_slab_layout_init(&slab, 16, 64, 2, &e);
    for(int ws : {64, 32})
        lines.push_back(
            accesses("dma reads w" + I(ws), [&](rocke_attn_bwd_lds_access_t* a, int* n, Err* er) {
                return rocke_attn_bwd_dma_slab_read_accesses(&slab, 0, 8, 32, ws, a, n, er);
            }));
    return lines;
}

/* ---------------------------------------------------------------- plan */

static const char* keep(const std::string& s)
{
    static std::vector<std::string*> store;
    store.push_back(new std::string(s));
    return store.back()->c_str();
}

static rocke_attn_bwd_lds_request_t parse_request(const std::string& text)
{
    rocke_attn_bwd_lds_request_t r = rocke_attn_bwd_lds_request_default();
    size_t start = 0;
    while(true)
    {
        size_t semi = text.find(';', start);
        std::string item
            = text.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        size_t colon = item.find(':');
        std::string name = item.substr(0, colon), v = item.substr(colon + 1);
        int iv = atoi(v.c_str());
        if(name == "head_size")
            r.head_size = iv;
        else if(name == "k_m0")
            r.k_m0 = iv;
        else if(name == "k_n0")
            r.k_n0 = iv;
        else if(name == "waves")
            r.waves = iv;
        else if(name == "wave_size")
            r.wave_size = iv;
        else if(name == "ring_depth")
            r.ring_depth = iv;
        else if(name == "ds_slots")
            r.ds_slots = iv;
        else if(name == "atom_k_g13")
            r.atom_k_g13 = iv;
        else if(name == "atom_k_g4")
            r.atom_k_g4 = iv;
        else if(name == "acc_in_lds")
            r.acc_in_lds = v == "True";
        else if(name == "dtype")
            r.dtype = keep(v);
        else if(name == "kv_residency")
            r.kv_residency = keep(v);
        else if(name == "kt_source")
            r.kt_source = keep(v);
        else if(name == "transpose_source")
            r.transpose_source = keep(v);
        else if(name == "global_path")
            r.global_path = keep(v);
        else if(name == "pt_route")
            r.pt_route = keep(v);
        else if(name == "dq_mode")
            r.dq_mode = keep(v);
        else if(name == "lds_swizzle")
            r.lds_swizzle = keep(v);
        else if(name == "dkv_epilogue")
            r.dkv_epilogue = keep(v);
        else if(name == "lds_order")
            r.lds_order = keep(v);
        if(semi == std::string::npos)
            break;
        start = semi + 1;
    }
    return r;
}

static std::string phases_text(unsigned phases)
{
    std::string out;
    for(int i = 0; i < 4; ++i)
        if(phases & (1u << i))
            out += std::string(out.empty() ? "" : ",")
                   + rocke_attn_bwd_lds_phase_name((rocke_attn_bwd_lds_phase_t)(1u << i));
    return out;
}

static std::string layout_repr(const rocke_attn_bwd_lds_layout_t& l)
{
    char buf[256];
    switch(l.kind)
    {
    case ROCKE_ATTN_BWD_LAYOUT_XT:
        snprintf(buf,
                 sizeof(buf),
                 "XtLayout(rows=%d, cols=%d, elem_bytes=%d, swizzle=%s, chunk_bytes=%d)",
                 l.xt.rows,
                 l.xt.cols,
                 l.xt.elem_bytes,
                 py_repr(l.xt.swizzle).c_str(),
                 l.xt.chunk_bytes);
        return buf;
    case ROCKE_ATTN_BWD_LAYOUT_ROW_XOR:
        snprintf(buf,
                 sizeof(buf),
                 "RowXorLayout(rows=%d, cols=%d, elem_bytes=%d, chunk_bytes=%d)",
                 l.row_xor.rows,
                 l.row_xor.cols,
                 l.row_xor.elem_bytes,
                 l.row_xor.chunk_bytes);
        return buf;
    case ROCKE_ATTN_BWD_LAYOUT_DMA_SLAB:
        snprintf(buf,
                 sizeof(buf),
                 "DmaSlabLayout(rows=%d, cols=%d, elem_bytes=%d)",
                 l.dma.rows,
                 l.dma.cols,
                 l.dma.elem_bytes);
        return buf;
    default:
        return py_repr(l.desc);
    }
}

static Lines dump_plan(const std::string& arch, bool strict, const std::string& text)
{
    Lines lines;
    Err e;
    rocke_attn_bwd_lds_request_t req = parse_request(text);
    if(rocke_attn_bwd_lds_request_init(&req, &e) != ROCKE_OK)
    {
        lines.push_back("request: " + exc(e));
        return lines;
    }
    lines.push_back("request: elem_bytes=" + I(rocke_attn_bwd_lds_request_elem_bytes(&req))
                    + " threads=" + I(rocke_attn_bwd_lds_request_threads(&req)));
    for(const char* name : {"q", "kt_stage", "dkv_out", "zz"})
        lines.push_back(std::string("swizzle_of ") + name + ": "
                        + rocke_attn_bwd_lds_request_swizzle_of(&req, name, "dflt"));
    static rocke_attn_bwd_lds_plan_t p;
    if(rocke_attn_bwd_plan_lds(&req, arch.c_str(), strict, &p, &e) != ROCKE_OK)
    {
        lines.push_back("plan: " + exc(e));
        return lines;
    }
    lines.push_back("plan: total=" + I(p.total_bytes) + " resident=" + I(p.resident_bytes)
                    + " capacity=" + I(p.capacity_bytes)
                    + " fits=" + B(rocke_attn_bwd_lds_plan_fits(&p))
                    + " phase_offset=" + I(rocke_attn_bwd_lds_plan_phase_offset(&p))
                    + " unaliased=" + I(rocke_attn_bwd_lds_plan_unaliased_bytes(&p)));
    std::string ph = "plan phases:";
    for(int i = 0; i < 4; ++i)
        ph += std::string(" ")
              + rocke_attn_bwd_lds_phase_name((rocke_attn_bwd_lds_phase_t)(1u << i)) + "="
              + I(p.phase_bytes[i]);
    lines.push_back(ph);
    for(int i = 0; i < p.n_buffers; ++i)
    {
        const rocke_attn_bwd_lds_buffer_t& b = p.buffers[i];
        lines.push_back(std::string("plan buffer: ") + b.name + "[" + I(b.slot)
                        + "] offset=" + I(b.offset) + " nbytes=" + I(b.nbytes)
                        + " phases=" + phases_text(b.phases) + " layout=" + b.layout
                        + " resident=" + B(b.resident));
    }
    for(int i = 0; i < p.n_layouts; ++i)
        lines.push_back(std::string("plan layout: ") + p.layouts[i].name + " = "
                        + layout_repr(p.layouts[i]));
    for(const char* name : {"q", "ds", "dkv_out", "k_res", "nope"})
    {
        lines.push_back(std::string("has ") + name + ": "
                        + B(rocke_attn_bwd_lds_plan_has(&p, name)));
        attempt(lines, std::string("offset ") + name + "[1]", [&](Err* er) {
            return I(rocke_attn_bwd_lds_plan_offset(&p, name, 1, er));
        });
        int offs[ROCKE_ATTN_BWD_LDS_MAX_BUFFERS];
        int n = rocke_attn_bwd_lds_plan_ring_offsets(&p, name, offs);
        lines.push_back(std::string("ring_offsets ") + name + ": " + tuple_i(offs, n));
        attempt(lines, std::string("ring_stride ") + name, [&](Err* er) {
            return I(rocke_attn_bwd_lds_plan_ring_stride(&p, name, er));
        });
        attempt(lines, std::string("layout ") + name, [&](Err* er) {
            const rocke_attn_bwd_lds_layout_t* l = rocke_attn_bwd_lds_plan_layout(&p, name, er);
            return l ? layout_repr(*l) : std::string();
        });
    }
    for(int i = 0; i < 4; ++i)
    {
        std::string s = std::string("live ")
                        + rocke_attn_bwd_lds_phase_name((rocke_attn_bwd_lds_phase_t)(1u << i))
                        + ": ";
        bool first = true;
        for(int j = 0; j < p.n_buffers; ++j)
            if(p.buffers[j].phases & (1u << i))
            {
                s += std::string(first ? "" : " ") + p.buffers[j].name + "[" + I(p.buffers[j].slot)
                     + "]";
                first = false;
            }
        lines.push_back(s);
    }
    attempt(lines, "check_capacity", [&](Err* er) {
        rocke_attn_bwd_lds_plan_check_capacity(&p, er);
        return std::string("None");
    });
    static rocke_attn_bwd_lds_plan_t bad;
    bad = p;
    bad.total_bytes -= 128;
    attempt(lines, "check shrunk", [&](Err* er) {
        rocke_attn_bwd_lds_plan_check(&bad, er);
        return std::string("None");
    });
    return lines;
}

/* ----------------------------------------------------------------- abi */

static Lines dump_abi()
{
    Lines lines;
    Err e;
    for(const char* stage : {"main", "prep", "convert", "reduce", "dq", "bwd"})
    {
        const char* names[ROCKE_ATTN_BWD_PARAMS_MAX];
        const char* fmts[ROCKE_ATTN_BWD_PARAMS_MAX];
        int n = 0;
        if(rocke_attn_bwd_params(stage, names, fmts, &n, &e) != ROCKE_OK)
            lines.push_back(std::string("params ") + stage + ": " + exc(e));
        else
        {
            std::string s = std::string("params ") + stage + ":";
            for(int i = 0; i < n; ++i)
                s += std::string(" ") + names[i] + ":" + fmts[i];
            lines.push_back(s);
        }
    }
    for(const char* sc : {"fold_ds", "dq_before_atomic", "convert", "runtime", "never"})
        for(const char* mode : {"direct", "atomic"})
            attempt(lines, std::string("unit_mults ") + sc + " " + mode, [&](Err* er) {
                unsigned m = 0;
                rocke_attn_bwd_static_unit_mults(sc, mode, &m, er);
                std::vector<std::string> v;
                if(m & ROCKE_ATTN_BWD_UNIT_DK_MULT)
                    v.push_back("dk_mult");
                if(m & ROCKE_ATTN_BWD_UNIT_DQ_MULT)
                    v.push_back("dq_mult");
                if(m & ROCKE_ATTN_BWD_UNIT_DS_MULT)
                    v.push_back("ds_mult");
                std::string s;
                for(size_t i = 0; i < v.size(); ++i)
                    s += (i ? "," : "") + v[i];
                return s;
            });
    return lines;
}

static Lines dump(const std::string& config)
{
    size_t bar = config.find('|');
    std::string kind = config.substr(0, bar), arg = config.substr(bar + 1);
    if(kind == "caps")
        return dump_caps(arg);
    if(kind == "proofs")
        return dump_proofs(arg);
    if(kind == "tiles")
        return dump_tiles(atoi(arg.c_str()));
    if(kind == "small")
        return dump_small(atoi(arg.c_str()));
    if(kind == "xt")
    {
        std::vector<int> v = split_ints(arg);
        return dump_xt(v[0], v[1], v[2], v[3]);
    }
    if(kind == "writer")
        return dump_writer(atoi(arg.c_str()));
    if(kind == "choose")
    {
        std::vector<int> v = split_ints(arg);
        return dump_choose(v[0], v[1]);
    }
    if(kind == "dma")
    {
        std::vector<int> v = split_ints(arg);
        return dump_dma(v[0], v[1], v[2]);
    }
    if(kind == "plan")
    {
        size_t b1 = arg.find('|');
        size_t b2 = arg.find('|', b1 + 1);
        return dump_plan(
            arg.substr(0, b1), arg.substr(b1 + 1, b2 - b1 - 1) == "1", arg.substr(b2 + 1));
    }
    if(kind == "access")
        return dump_access();
    return dump_abi();
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    std::vector<std::string> cfgs = configs();
    if(idx < 0 || idx >= (int)cfgs.size())
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    printf("config: %s\n", cfgs[idx].c_str());
    for(const std::string& line : dump(cfgs[idx]))
    {
        fputs(line.c_str(), stdout);
        fputc('\n', stdout);
    }
    return 0;
}
