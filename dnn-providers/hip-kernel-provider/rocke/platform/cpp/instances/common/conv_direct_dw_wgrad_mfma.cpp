// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dw_wgrad_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_dw_wgrad_mfma.py: the depthwise wgrad on
 * the batched 4x4x4 MFMA (build_dw_wgrad_mfma).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_dw_wgrad_mfma.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_direct_wgrad_internal.hpp"

using namespace rocke_dwg;

namespace
{

constexpr int kNW = 4; /* waves per block */
constexpr int kBC = 64; /* channels per block: 16 per wave */
constexpr int kTaps = 4; /* M = N of the 4x4x4 atom */
constexpr int kCH = kBC / 8; /* 16-byte chunks per staged pixel */

const char* bf16_flag[1] = {"bf16"};

int imax(int a, int b)
{
    return a > b ? a : b;
}

int imin(int a, int b)
{
    return a < b ? a : b;
}

int block_q(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return 4 * s->q_tiles;
}

int rg(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return cdiv(s->problem.KH, kTaps);
}

int sg(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return cdiv(s->problem.KW, kTaps);
}

/* A lane's tap shift reaches past one v_perm (3 halves) from KW 4 on. */
bool preshift(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return imin(kTaps, s->problem.KW) - 1 > 2;
}

int win_words(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return 2 * (s->q_tiles - 1) + 2 * (sg(s) - 1) + 3;
}

int win_reads(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return cdiv(win_words(s) + (preshift(s) ? 1 : 0), 4);
}

/* main_in elements per channel: the lane's whole aligned window. */
int in_pitch(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return 8 * win_reads(s);
}

/* 4-column groups of X staged and transposed per row. */
int in_groups(const rocke_dw_wgrad_mfma_spec_t* s)
{
    int cols = 2 * (win_words(s) + (preshift(s) ? 1 : 0));
    return cdiv(cols, 4);
}

int d_pitch(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return block_q(s) + 4;
}

int dnew_elems(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return kBC * d_pitch(s);
}

/* DMA instructions per wave per staged X row. */
int ix(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return cdiv(cdiv(4 * in_groups(s) * kCH, kWave), kNW);
}

int iy(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return cdiv(cdiv(block_q(s) * kCH, kWave), kNW);
}

int n_channel_tiles(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return cdiv(s->problem.groups, kBC);
}

int n_q(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return cdiv(problem_wo(&s->problem), block_q(s));
}

int n_units(const rocke_dw_wgrad_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return cdiv(p->N, s->unfold_n) * cdiv(p->H, s->rows_per_chunk);
}

int n_z(const rocke_dw_wgrad_mfma_spec_t* s)
{
    int planes = n_channel_tiles(s) * n_q(s);
    return imin(n_units(s), imax(1, cdiv(s->target_blocks, planes)));
}

} // namespace

extern "C" rocke_dw_wgrad_mfma_spec_t rocke_dw_wgrad_mfma_spec_default(void)
{
    rocke_dw_wgrad_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "dw_wgrad_mfma";
    s.q_tiles = 8;
    s.rows_per_chunk = 16;
    s.prefetch = 2;
    s.target_blocks = 2048;
    s.unfold_n = 1;
    return s;
}

extern "C" long rocke_dw_wgrad_mfma_lds_bytes(const rocke_dw_wgrad_mfma_spec_t* s)
{
    long stage = (long)s->prefetch * kNW * (ix(s) + iy(s)) * kWave * 16;
    return stage + ((long)kBC * in_pitch(s) + dnew_elems(s)) * 2;
}

extern "C" int rocke_dw_wgrad_mfma_image_pitch(const rocke_dw_wgrad_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    int wo = problem_wo(p);
    return imax(imax(wo, wo + p->KW - 1 - p->PAD), p->PAD + p->W);
}

extern "C" int rocke_dw_wgrad_mfma_n_partitions(const rocke_dw_wgrad_mfma_spec_t* s)
{
    return n_q(s) * n_z(s);
}

extern "C" long rocke_dw_wgrad_mfma_workspace_bytes(const rocke_dw_wgrad_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return (long)rocke_dw_wgrad_mfma_n_partitions(s) * p->KH * p->KW * p->groups * 4;
}

extern "C" void rocke_dw_wgrad_mfma_launch_geometry(const rocke_dw_wgrad_mfma_spec_t* s,
                                                    int grid[3],
                                                    int block[3])
{
    grid[0] = n_channel_tiles(s);
    grid[1] = n_q(s);
    grid[2] = n_z(s);
    block[0] = kNW * kWave;
    block[1] = 1;
    block[2] = 1;
}

extern "C" rocke_status_t
    rocke_dw_wgrad_mfma_kernel_name(const rocke_dw_wgrad_mfma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[64], qt_buf[24], rpc_buf[24], pf_buf[24], un_buf[24], tb_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%dp%ds%d", p->KH, p->KW, p->PAD, p->stride);
    snprintf(qt_buf, sizeof qt_buf, "qt%d", s->q_tiles);
    snprintf(rpc_buf, sizeof rpc_buf, "rpc%d", s->rows_per_chunk);
    snprintf(pf_buf, sizeof pf_buf, "pf%d", s->prefetch);
    snprintf(un_buf, sizeof un_buf, "un%d", s->unfold_n);
    snprintf(tb_buf, sizeof tb_buf, "tb%d", s->target_blocks);
    const char* parts[6] = {f_buf, qt_buf, rpc_buf, pf_buf, un_buf, tb_buf};
    int on[1] = {problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 6, bf16_flag, on, 1, out, out_cap, NULL);
}

extern "C" bool rocke_dw_wgrad_mfma_is_valid_spec(const rocke_dw_wgrad_mfma_spec_t* s,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    char msg[ROCKE_ERR_MSG_CAP];
    if(s == NULL)
    {
        set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(dilation_reason(p, msg, sizeof msg))
    {
        set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* why = NULL;
    if(p->cpg != 1 || p->kpg != 1)
    {
        why = "depthwise only";
    }
    else if(!problem_dtype_ok(p))
    {
        why = "fp16/bf16 only";
    }
    else if(p->stride != 1)
    {
        why = "stride 1 only";
    }
    else if(p->PAD > p->KH - 1 || p->PAD > p->KW - 1)
    {
        why = "PAD <= K - 1";
    }
    else if(p->groups % 8)
    {
        why = "C must be a multiple of 8";
    }
    else if(s->q_tiles < 1 || s->q_tiles > 8)
    {
        why = "q_tiles in 1..8";
    }
    else if(s->prefetch < 1 || s->prefetch > 4)
    {
        why = "prefetch in 1..4";
    }
    else if(s->unfold_n < 1 || s->unfold_n > 4)
    {
        why = "unfold_n in 1..4";
    }
    else if(s->unfold_n > 1 && s->unfold_n * rocke_dw_wgrad_mfma_image_pitch(s) > block_q(s))
    {
        ROCKE_ERR_SNPRINTF(msg,
                           sizeof msg,
                           "%d images at pitch %d exceed the %d-column tile",
                           s->unfold_n,
                           rocke_dw_wgrad_mfma_image_pitch(s),
                           block_q(s));
        why = msg;
    }
    else if(s->rows_per_chunk < 1)
    {
        why = "rows_per_chunk >= 1";
    }
    else if(s->target_blocks < 1)
    {
        why = "target_blocks >= 1";
    }
    else if(rocke_dw_wgrad_mfma_lds_bytes(s) > 160 * 1024)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "LDS %ld B > 160 KiB", rocke_dw_wgrad_mfma_lds_bytes(s));
        why = msg;
    }
    else if(strcmp(arch, "gfx950") != 0)
    {
        why = "ds_read_tr16_b64 needs gfx950";
    }
    if(why != NULL)
    {
        set_reason(reason, reason_cap, why);
        return false;
    }
    set_reason(reason, reason_cap, "ok");
    return true;
}

namespace
{

struct Lane
{
    V col, ok, cc;
};

struct Off
{
    V off, ok, j;
};

/* ring[g][qt] = the two i32 words of a <4 x 16-bit> dY fragment. */
using Ring = std::vector<std::vector<std::vector<V>>>;

rocke_kernel_def_t*
    build_mfma(rocke_ir_builder_t* b, const rocke_dw_wgrad_mfma_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_dw_wgrad_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        fail_invalid(b, "invalid DwWgradMfmaSpec", reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const int KH = p.KH, KW = p.KW, PAD = p.PAD;
    const int TAPS = KH * KW;
    const int QT = spec->q_tiles, BQ = block_q(spec);
    const int RG = rg(spec), SG = sg(spec);
    const int D = spec->prefetch;
    const int IX = ix(spec), IY = iy(spec);
    const int GI = in_groups(spec);
    const int XC = 4 * GI; /* staged X columns */
    const int REAL_COLS = BQ + KW - 1;
    const int HALF = in_pitch(spec);
    const int DP = d_pitch(spec);
    const int WW = win_words(spec);
    const int T = kNW * kWave;
    const int UN = spec->unfold_n;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", T);
    rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    rocke_dconv_params_t P;
    rocke_dconv_emit_params(b, &P, "wgrad", io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    V c0 = e.ci(0);
    V c1 = e.ci(1);
    V c2 = e.ci(2);
    V oob = e.ci(((int64_t)1 << 31) - 1);
    V p_N = P.p_N;
    V p_C = P.p_groups;
    V p_H = P.p_Hi, p_W = P.p_Wi;
    V p_Ho = P.p_Ho, p_Wo = P.p_Wo;
    V sdy_n = P.p_A_stride_n, sdy_h = P.p_A_stride_hi, sdy_w = P.p_A_stride_wi;
    V sx_n = P.p_B_stride_n, sx_h = P.p_B_stride_hi, sx_w = P.p_B_stride_wi;

    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    V ctile = rocke_b_block_id_x(b);
    V qtile = rocke_b_block_id_y(b);
    V zslot = rocke_b_block_id_z(b);
    V c_blk0 = e.muli(ctile, kBC);
    V q0 = e.muli(qtile, BQ);
    V n_q_v = e.cdivi(p_Wo, BQ);
    V n_chunks = e.cdivi(p_H, spec->rows_per_chunk);
    V n_ngrp = e.cdivi(p_N, UN);
    V n_units_v = e.mul(n_ngrp, n_chunks);
    /* Same formula as DwWgradMfmaSpec.n_z, which sizes the grid. */
    V n_ct = e.cdivi(p_C, kBC);
    V planes = e.mul(n_ct, n_q_v);
    V nz = e.ci(spec->target_blocks);
    nz = e.add(nz, e.sub(planes, c1));
    nz = e.div(nz, planes);
    nz = e.smax(nz, c1);
    V n_z_v = e.smin(n_units_v, nz);

    V a_rsrc = rocke_b_buffer_rsrc(b, P.A, P.A_bytes); /* dY */
    V b_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes); /* X */

    /* ---- LDS: one allocation per staging buffer (one alias scope each) ---- */
    const int X_ELEMS = kNW * IX * kWave * 8;
    const int Y_ELEMS = kNW * IY * kWave * 8;
    std::vector<V> x_stage, y_stage, x_addr, y_addr;
    for(int i = 0; i < D; ++i)
    {
        std::string h = "xs" + std::to_string(i);
        x_stage.push_back(e.smem_alloc_1d(io_type, X_ELEMS, h.c_str()));
    }
    for(int i = 0; i < D; ++i)
    {
        std::string h = "ys" + std::to_string(i);
        y_stage.push_back(e.smem_alloc_1d(io_type, Y_ELEMS, h.c_str()));
    }
    for(V s : x_stage)
    {
        x_addr.push_back(rocke_b_smem_addr_of(b, s));
    }
    for(V s : y_stage)
    {
        y_addr.push_back(rocke_b_smem_addr_of(b, s));
    }
    const int MIN = kBC * HALF;
    const int DNEW = dnew_elems(spec);
    V main_in = e.smem_alloc_1d(io_type, MIN, "min");
    /* The newest dY row, channel-major; older rows live in registers. */
    V dnew = e.smem_alloc_1d(io_type, DNEW, "dnew");

    /* DMA lane geometry: slot L = (wave * I + i) * 64 + lane -> (col, chunk). */
    auto dma_lanes = [&](int n_instr, int cols) {
        std::vector<Lane> out;
        for(int i = 0; i < n_instr; ++i)
        {
            V slot = e.muli(wave, n_instr);
            slot = e.addi(slot, i);
            slot = e.muli(slot, kWave);
            slot = e.add(slot, lane);
            Lane l;
            l.col = e.divi(slot, kCH);
            V ch = e.modi(slot, kCH);
            ch = e.muli(ch, 8);
            l.cc = e.add(c_blk0, ch);
            V a = e.lti(l.col, cols);
            l.ok = e.land(a, e.lt(l.cc, p_C));
            out.push_back(l);
        }
        return out;
    };

    /* Lane offsets: (offset without the image and row, ok, packed image j). */
    V pitch = NULL;
    if(UN > 1)
    {
        /* Image j owns tile columns [j * P, (j + 1) * P). */
        V t = e.addi(p_Wo, KW - 1 - PAD);
        pitch = e.smax(p_Wo, t);
        t = e.addi(p_W, PAD);
        pitch = e.smax(pitch, t);
    }
    auto split = [&](V col, V* j, V* u) {
        if(UN > 1)
        {
            *j = e.div(col, pitch);
            V m = e.mul(*j, pitch);
            *u = e.sub(col, m);
        }
        else
        {
            *j = c0;
            *u = e.add(q0, col);
        }
    };

    std::vector<Off> x_offs, y_offs;
    {
        std::vector<Lane> xl = dma_lanes(IX, imin(XC, REAL_COLS));
        for(const Lane& l : xl)
        {
            V j, u;
            split(l.col, &j, &u);
            V wi = e.subi(u, PAD);
            V a = e.ge(wi, c0);
            a = e.land(a, e.lt(wi, p_W));
            V okc = e.land(l.ok, a);
            okc = e.land(okc, e.lti(j, UN));
            V off = e.mul(wi, sx_w);
            off = e.add(off, l.cc);
            off = e.add(off, e.mul(j, sx_n));
            x_offs.push_back({off, okc, j});
        }
        std::vector<Lane> yl = dma_lanes(IY, BQ);
        for(const Lane& l : yl)
        {
            V j, wo;
            split(l.col, &j, &wo);
            V ok = e.land(l.ok, e.lt(wo, p_Wo));
            ok = e.land(ok, e.lti(j, UN));
            V off = e.mul(wo, sdy_w);
            off = e.add(off, l.cc);
            off = e.add(off, e.mul(j, sdy_n));
            y_offs.push_back({off, ok, j});
        }
    }

    auto dma = [&](V rsrc, V addr, int n_instr, int i, V voff) {
        V o = e.muli(wave, n_instr);
        o = e.addi(o, i);
        o = e.muli(o, kWave * 16);
        V lds = rocke_b_smem_ptr_add(b, addr, e.zext64(o));
        rocke_b_async_buffer_load_lds(b, rsrc, lds, voff, c0, 4, 0);
    };

    /* ---- transpose geometry (lane L ends up with channel L) ---- */
    V tr_row = e.modi(e.divi(lane, 4), 4);
    V tr_cg = e.muli(e.divi(lane, 16), 4);
    tr_cg = e.add(tr_cg, e.modi(lane, 4));
    V tr_base = e.muli(tr_row, kBC);
    tr_base = e.add(tr_base, e.muli(tr_cg, 4));

    auto transpose_rows = [&](V stage_x, V stage_y, bool with_x) {
        struct Read
        {
            int g;
            V gg;
            bool is_x;
        };
        std::vector<Read> reads;
        if(with_x)
        {
            for(int g = 0; g < cdiv(GI, kNW); ++g)
            {
                reads.push_back({g, e.addi(wave, g * kNW), true});
            }
        }
        for(int g = 0; g < cdiv(QT, kNW); ++g)
        {
            reads.push_back({g, e.addi(wave, g * kNW), false});
        }
        auto one = [&](V gg, bool is_x) {
            V src = e.muli(gg, 4 * kBC);
            src = e.add(tr_base, src);
            V v = e.ds_read_tr(is_x ? stage_x : stage_y, c0, src, io_type);
            V dst = e.muli(lane, is_x ? HALF : DP);
            dst = e.add(dst, e.muli(gg, 4));
            e.smem_store(is_x ? main_in : dnew, c0, dst, v, 4);
        };
        for(const Read& r : reads)
        {
            int n_groups = r.is_x ? GI : QT;
            if((r.g + 1) * kNW > n_groups)
            {
                e.if_(e.lti(r.gg, n_groups), [&]() { one(r.gg, r.is_x); });
            }
            else
            {
                one(r.gg, r.is_x);
            }
        }
    };

    /* ---- per-lane MFMA operand geometry ---- */
    V lane4 = e.modi(lane, 4);
    V ch_w = e.muli(wave, 16);
    ch_w = e.add(ch_w, e.divi(lane, 4)); /* channel in the tile */
    V lane_s = e.smini(lane4, imin(kTaps, KW) - 1);
    V lane_word = preshift(spec) ? e.div(lane_s, c2) : c0;
    V lane_res = e.sub(lane_s, e.mul(lane_word, c2));
    V sel0 = e.ci(0x03020100);
    V sel1 = e.ci(0x01010101);
    V pre_sel = e.mul(sel1, e.muli(lane_word, 4));
    pre_sel = e.add(sel0, pre_sel);
    V res_sel = e.mul(sel1, e.mul(lane_res, c2));
    res_sel = e.add(sel0, res_sel);
    const rocke_type_t* frag_t = rocke_vector_type(b, io_type, 4);
    /* The four lanes of a channel each read a quarter of its main_in window. */
    const int NWR = win_reads(spec);
    std::vector<V> a_reads, d_reads;
    for(int k = 0; k < cdiv(NWR, 4); ++k)
    {
        V a = e.muli(ch_w, HALF);
        V c = e.addi(lane4, 4 * k);
        c = e.smini(c, NWR - 1);
        c = e.muli(c, 8);
        a_reads.push_back(e.add(a, c));
    }
    /* Likewise the newest dY row: lane j of a channel reads q tiles j, j + 4. */
    for(int k = 0; k < cdiv(QT, 4); ++k)
    {
        V a = e.muli(ch_w, DP);
        V c = e.addi(lane4, 4 * k);
        c = e.smini(c, QT - 1);
        c = e.muli(c, 4);
        d_reads.push_back(e.add(a, c));
    }
    V head = e.eq(lane4, c0);

    auto bcast = [&](V v, int src) { return rocke_b_quad_perm(b, v, src, src, src, src); };
    auto words = [&](V v, int n) {
        V w = rocke_b_vec_bitcast(b, v, rocke_vector_type(b, rocke_i32(), n));
        std::vector<V> out;
        for(int el = 0; el < n; ++el)
        {
            out.push_back(rocke_b_vec_extract(b, w, el));
        }
        return out;
    };

    /* Advance the dY ring one row: lane (g, r) takes (g, r - 1)'s row, lane
     * (g, 0) takes (g - 1, 3)'s, and lane (0, 0) the newest row. */
    auto shift_ring = [&](const Ring& ring) {
        std::vector<std::vector<V>> fresh;
        for(V idx : d_reads)
        {
            fresh.push_back(words(e.smem_load(dnew, c0, idx, io_type, 4), 2));
        }
        Ring rot(RG, std::vector<std::vector<V>>(QT));
        for(int g = 0; g < RG; ++g)
        {
            for(int qt = 0; qt < QT; ++qt)
            {
                for(V d : ring[g][qt])
                {
                    rot[g][qt].push_back(rocke_b_quad_perm(b, d, 3, 0, 1, 2));
                }
            }
        }
        Ring out(RG, std::vector<std::vector<V>>(QT));
        for(int g = 0; g < RG; ++g)
        {
            for(int qt = 0; qt < QT; ++qt)
            {
                std::vector<V> src;
                if(g == 0)
                {
                    for(V d : fresh[qt / 4])
                    {
                        src.push_back(bcast(d, qt % 4));
                    }
                }
                else
                {
                    src = rot[g - 1][qt];
                }
                for(int el = 0; el < 2; ++el)
                {
                    out[g][qt].push_back(e.select(head, src[el], rot[g][qt][el]));
                }
            }
        }
        return out;
    };

    /* MFMA the X row in main_in against the dY rows in ``ring``. */
    auto compute = [&](const Ring& ring, std::vector<V>& accs) {
        std::vector<std::vector<V>> part;
        for(V idx : a_reads)
        {
            part.push_back(words(e.smem_load(main_in, c0, idx, io_type, 8), 4));
        }
        std::vector<V> win;
        for(int m = 0; m < NWR; ++m)
        {
            for(int el = 0; el < 4; ++el)
            {
                win.push_back(bcast(part[m / 4][el], m % 4));
            }
        }
        if(preshift(spec))
        {
            std::vector<V> w2;
            for(int j = 0; j < WW; ++j)
            {
                w2.push_back(rocke_b_perm_b32(b, win[j + 1], win[j], pre_sel));
            }
            win = w2;
        }
        std::vector<std::vector<V>> bfr(RG);
        for(int g = 0; g < RG; ++g)
        {
            for(int qt = 0; qt < QT; ++qt)
            {
                V pk = rocke_b_vec_pack(b, ring[g][qt].data(), 2, rocke_i32());
                bfr[g].push_back(rocke_b_vec_bitcast(b, pk, frag_t));
            }
        }
        for(int sgi = 0; sgi < SG; ++sgi)
        {
            for(int qt = 0; qt < QT; ++qt)
            {
                int k = 2 * qt + 2 * sgi;
                V a0 = rocke_b_perm_b32(b, win[k + 1], win[k], res_sel);
                V a1 = rocke_b_perm_b32(b, win[k + 2], win[k + 1], res_sel);
                V pr[2] = {a0, a1};
                V af = rocke_b_vec_bitcast(b, rocke_b_vec_pack(b, pr, 2, rocke_i32()), frag_t);
                for(int rgi = 0; rgi < RG; ++rgi)
                {
                    int i = rgi * SG + sgi;
                    accs[i] = bf16 ? rocke_b_mfma_f32_4x4x4_bf16(b, af, bfr[rgi][qt], accs[i])
                                   : rocke_b_mfma_f32_4x4x4_f16(b, af, bfr[rgi][qt], accs[i]);
                }
            }
        }
    };

    V zero_acc = rocke_b_zero_vec_f32(b, 4);
    const int n_acc = RG * SG;
    const int PER_STEP = IX + IY;

    auto pack = [&](const std::vector<V>& accs, const Ring& ring) {
        std::vector<V> out(accs);
        for(const auto& g : ring)
        {
            for(const auto& qt : g)
            {
                for(V d : qt)
                {
                    out.push_back(d);
                }
            }
        }
        return out;
    };
    auto unpack = [&](const std::vector<V>& vals, std::vector<V>* accs, Ring* ring) {
        accs->assign(vals.begin(), vals.begin() + n_acc);
        ring->assign(RG, std::vector<std::vector<V>>(QT));
        for(int g = 0; g < RG; ++g)
        {
            for(int qt = 0; qt < QT; ++qt)
            {
                int o = n_acc + (g * QT + qt) * 2;
                (*ring)[g][qt] = {vals[o], vals[o + 1]};
            }
        }
    };

    /* Unit loop and epilogue, PRE dY rows staged before a unit's first X row. */
    auto emit_body = [&](int PRE) {
        std::string acc_prefix = "acc" + std::to_string(PRE) + "_";
        std::string unit_name = "unit" + std::to_string(PRE);
        std::vector<V> accs = e.for_iter(
            zslot,
            n_units_v,
            n_z_v,
            iter_names(acc_prefix.c_str(), n_acc),
            std::vector<V>(n_acc, zero_acc),
            unit_name.c_str(),
            [&](V unit, std::vector<V> carried) {
                /* A fresh ring per unit: its zeros stand in for the dY rows
                 * above the image, which the whole-image prologue skips. */
                std::vector<V> la_accs = carried;
                Ring ring(RG, std::vector<std::vector<V>>(QT, std::vector<V>{c0, c0}));
                V n_img = e.muli(e.mod(unit, n_ngrp), UN);
                V chunk = e.div(unit, n_ngrp);
                V y0 = e.muli(chunk, spec->rows_per_chunk);
                V k_rpc = e.ci(spec->rows_per_chunk);
                V n_rows = e.smin(k_rpc, e.sub(p_H, y0));
                V n_steps = e.addi(n_rows, PRE);
                V x_img = e.mul(n_img, sx_n);
                V y_img = e.mul(n_img, sdy_n);

                /* DMA step t: X row y0 - PRE + t (steps >= PRE), dY row y + PAD. */
                auto issue = [&](int buf, V t) {
                    V y = e.add(e.subi(y0, PRE), t);
                    V x_ok = e.gei(t, PRE);
                    x_ok = e.land(x_ok, e.lt(t, n_steps));
                    V x_row = e.add(x_img, e.mul(y, sx_h));
                    for(int i = 0; i < (int)x_offs.size(); ++i)
                    {
                        V ok = e.land(x_offs[i].ok, x_ok);
                        if(UN > 1)
                        {
                            V a = e.add(n_img, x_offs[i].j);
                            ok = e.land(ok, e.lt(a, p_N));
                        }
                        V voff = e.mul(e.add(x_offs[i].off, x_row), c2);
                        voff = e.select(ok, voff, oob);
                        dma(b_rsrc, x_addr[buf], IX, i, voff);
                    }
                    V pr = e.addi(y, PAD);
                    V y_ok = e.ge(pr, c0);
                    y_ok = e.land(y_ok, e.lt(pr, p_Ho));
                    y_ok = e.land(y_ok, e.lt(t, n_steps));
                    V y_row = e.add(y_img, e.mul(pr, sdy_h));
                    for(int i = 0; i < (int)y_offs.size(); ++i)
                    {
                        V ok = e.land(y_offs[i].ok, y_ok);
                        if(UN > 1)
                        {
                            V a = e.add(n_img, y_offs[i].j);
                            ok = e.land(ok, e.lt(a, p_N));
                        }
                        V voff = e.mul(e.add(y_offs[i].off, y_row), c2);
                        voff = e.select(ok, voff, oob);
                        dma(a_rsrc, y_addr[buf], IY, i, voff);
                    }
                };

                auto step
                    = [&](V t, int buf, std::vector<V>& accs_s, Ring& ring_s, bool do_compute) {
                          /* Issue the row D - 1 ahead into the buffer the last step drained. */
                          issue((buf + D - 1) % D, e.addi(t, D - 1));
                          rocke_b_s_waitcnt(b, (D - 1) * PER_STEP, 0, -1);
                          rocke_b_s_barrier_bare(b);
                          transpose_rows(x_stage[buf], y_stage[buf], do_compute);
                          rocke_b_s_waitcnt(b, -1, 0, -1);
                          rocke_b_s_barrier_bare(b);
                          ring_s = shift_ring(ring_s);
                          if(do_compute)
                          {
                              compute(ring_s, accs_s);
                          }
                      };

                for(int t = 0; t < D - 1; ++t)
                {
                    issue(t % D, e.ci(t));
                }
                /* Prologue: the PRE dY rows the first input row pairs with. */
                for(int t = 0; t < PRE; ++t)
                {
                    step(e.ci(t), t % D, la_accs, ring, false);
                }
                /* Main: one input row per step, unrolled by D. */
                V trips = e.cdivi(n_rows, D);
                trips = e.smax(trips, c1);
                std::string a_prefix = "a" + std::to_string(PRE) + "_";
                std::string rb_name = "rb" + std::to_string(PRE);
                std::vector<V> init = pack(la_accs, ring);
                std::vector<V> res = e.for_iter(c0,
                                                trips,
                                                c1,
                                                iter_names(a_prefix.c_str(), init.size()),
                                                init,
                                                rb_name.c_str(),
                                                [&](V rb, std::vector<V> la) {
                                                    std::vector<V> l_acc;
                                                    Ring l_ring;
                                                    unpack(la, &l_acc, &l_ring);
                                                    for(int u = 0; u < D; ++u)
                                                    {
                                                        V t = e.muli(rb, D);
                                                        t = e.addi(t, PRE);
                                                        t = e.addi(t, u);
                                                        step(t, (PRE + u) % D, l_acc, l_ring, true);
                                                    }
                                                    return pack(l_acc, l_ring);
                                                });
                /* Drain the tail DMAs before the next unit reuses the buffers. */
                rocke_b_s_waitcnt(b, 0, 0, -1);
                rocke_b_s_barrier_bare(b);
                if((int)res.size() < n_acc)
                {
                    return std::vector<V>();
                }
                return std::vector<V>(res.begin(), res.begin() + n_acc);
            });
        if((int)accs.size() != n_acc)
        {
            return;
        }

        /* ---- epilogue: ws[part][tap][c], part = zslot * n_q + qtile ---- */
        V part = e.add(e.mul(zslot, n_q_v), qtile);
        V c_glb = e.add(c_blk0, ch_w);
        V c_ok = e.lt(c_glb, p_C);
        for(int rgi = 0; rgi < RG; ++rgi)
        {
            V r = e.addi(lane4, rgi * kTaps);
            V row0 = e.muli(part, TAPS);
            row0 = e.add(row0, e.muli(r, KW));
            row0 = e.mul(row0, p_C);
            row0 = e.add(row0, c_glb);
            e.if_(e.land(c_ok, e.lti(r, KH)), [&]() {
                for(int sgi = 0; sgi < SG; ++sgi)
                {
                    V acc = accs[rgi * SG + sgi];
                    for(int i = 0; i < kTaps; ++i)
                    {
                        int s = sgi * kTaps + i;
                        if(s < KW)
                        {
                            V ks = e.ci(s);
                            V off = e.add(row0, e.mul(ks, p_C));
                            rocke_b_global_store(b, P.D, off, rocke_b_vec_extract(b, acc, i), 4);
                        }
                    }
                }
            });
        }
    };

    if(PAD == KH - 1)
    {
        emit_body(PAD);
    }
    else
    {
        /* Whole images per unit (one row chunk) skip the KH - 1 - PAD zero rows. */
        rocke_if_else_t ife = rocke_b_scf_if_else(b, e.eq(n_chunks, c1));
        if(ife.then_region == NULL || ife.else_region == NULL)
        {
            return NULL;
        }
        rocke_b_region_enter(b, ife.then_region);
        emit_body(PAD);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ife.else_region);
        emit_body(KH - 1);
        rocke_b_region_leave(b);
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_dw_wgrad_mfma(rocke_ir_builder_t* b,
                                                         const rocke_dw_wgrad_mfma_spec_t* spec,
                                                         const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_mfma(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_dw_wgrad_mfma_new(rocke_ir_builder_t* b,
                                                             const rocke_dw_wgrad_mfma_spec_t* spec,
                                                             const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_dw_wgrad_mfma",
        [&](char* out, size_t cap) { return rocke_dw_wgrad_mfma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_dw_wgrad_mfma(b, spec, arch); });
}

extern "C" rocke_status_t rocke_dw_wgrad_mfma_lower_to_llvm(const rocke_dw_wgrad_mfma_spec_t* spec,
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
    return lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_dw_wgrad_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_dw_wgrad_mfma_new(b, spec, arch); });
}
