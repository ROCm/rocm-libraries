// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_dgrad_win.cpp -- C++ port of the
 * windowed depthwise dgrad kernel (library/kernels/common/conv_direct_grouped.py):
 *
 *   Python                                          C++ (this file)
 *   DirectDepthwiseDgradWindowedSpec             -> rocke_direct_depthwise_dgrad_win_spec_t
 *   _depthwise_dgrad_win_check / is_valid_...    -> rocke_direct_depthwise_dgrad_win_is_valid_spec()
 *   kernel_name()                                -> rocke_direct_depthwise_dgrad_win_kernel_name()
 *   build_direct_depthwise_dgrad_windowed(...)   -> rocke_build_direct_depthwise_dgrad_win_new()
 *   _dw_toeplitz_lanes(...)                      -> dw_toeplitz_lanes()
 *   _dw_toeplitz_atom(...)                       -> dw_toeplitz_atom()
 *   _build_dw_dgrad_win_toeplitz(...)            -> dw_win_build_toeplitz()
 *
 * Stride 1, cpg = kpg = 1. The kernel takes the direct dgrad kernarg block
 * (rocke_dconv_emit_params(direction "dgrad")): N, H, W, Ho, Wo, groups and the
 * dY/dX strides are runtime; KH, KW and PAD are baked capabilities, the tile
 * and the forms (dot2, mfma) are knobs.
 *
 * Every builder call with side effects is issued as its own statement in
 * Python's left-to-right evaluation order, so the IR numbering matches the
 * Python engine byte for byte.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx */
#include "rocke/helper_rocke.helpers.grid.h" /* rocke_chiplet_transform_chunked_runtime */
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_ceil_div */
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

typedef rocke_value_t V;

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */
rocke_direct_depthwise_dgrad_win_spec_t rocke_direct_depthwise_dgrad_win_spec_default(void)
{
    rocke_direct_depthwise_dgrad_win_spec_t s;
    memset(&s, 0, sizeof s);
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_depthwise_dgrad_win";
    s.block_w = 8;
    s.block_waves = 1;
    s.ch_per_lane = 1;
    s.block_h = 8;
    s.dot2 = false;
    s.mfma = false;
    s.w_fold = 1;
    s.prefetch_rows = 1;
    s.wave_size = 64;
    s.fwd = false;
    s.out_stride = 1;
    s.dense = false;
    s.lds_swz = false;
    s.xcd_tiles = false;
    s.dma = false;
    s.wdma = false;
    return s;
}

int rocke_direct_depthwise_dgrad_win_threads_per_block(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_dgrad_win_block_ch(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    if(spec->mfma)
        return spec->block_waves * (spec->dense ? ROCKE_DW_WIN_DENSE_CH : ROCKE_DW_DGRAD_MFMA_CH);
    return rocke_direct_depthwise_dgrad_win_threads_per_block(spec) * spec->ch_per_lane;
}

int rocke_direct_depthwise_dgrad_win_tile_w(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    if(spec->mfma && spec->dense)
        return spec->w_fold > 0 ? 16 / spec->w_fold : 0;
    return spec->mfma ? rocke_dw_mfma_tile_w(spec->w_fold) : spec->block_w;
}

int rocke_dw_mfma_passes(int KW)
{
    return (KW + 4) / 4;
}

bool rocke_dw_mfma_w_fold_ok(int w_fold)
{
    return w_fold == 1 || w_fold == 2 || w_fold == 4;
}

int rocke_dw_mfma_tile_w(int w_fold)
{
    return w_fold > 0 ? 32 / w_fold : 0;
}

int rocke_dw_mfma_in_passes(int tile_w, int KW, int w_fold, int block_ch, int threads)
{
    if(threads <= 0)
        return 0;
    const int win = tile_w + 4 * rocke_dw_mfma_passes(KW) - 2;
    const int per_px = block_ch / ROCKE_DW_DGRAD_MFMA_CH;
    return (int)rocke_ceil_div((long long)w_fold * win * per_px, threads);
}

long rocke_dw_mfma_lds_bytes(int in_passes, int threads, int block_ch, int w_fold, int tile_w)
{
    const int per_px = block_ch / ROCKE_DW_DGRAD_MFMA_CH;
    if(per_px <= 0)
        return 0;
    const int in_px = (int)rocke_ceil_div((long long)in_passes * threads, per_px);
    const int out_px = w_fold * tile_w;
    return 2L * (in_px + out_px) * (block_ch + ROCKE_DW_DGRAD_MFMA_CH) * 2;
}

/* 8-channel input chunks a wave contracts (1 for depthwise and for the
 * block-diagonal cpg 2 / 4 chunks). */
static int dw_win_cin_chunks(const rocke_direct_conv_problem_t* p)
{
    const int c = p->cpg / ROCKE_DW_DGRAD_MFMA_CH;
    return c > 1 ? c : 1;
}

long long rocke_direct_depthwise_dgrad_win_unrolled_fmas(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    return (long long)spec->block_h * p->KH * p->KW * spec->block_w * spec->ch_per_lane;
}

long long rocke_direct_depthwise_dgrad_win_unrolled_mfmas(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(spec->dense)
        return (long long)spec->block_h * p->KH * p->KW * (p->cpg / ROCKE_DW_WIN_DENSE_K);
    return (long long)spec->block_h * p->KH * rocke_dw_mfma_passes(p->KW) * dw_win_cin_chunks(p);
}

/* The 16x16x32 atom of the Toeplitz form (mirrors _DwToeplitzAtom). */
typedef struct dw_toeplitz_atom_t
{
    int wave; /* lanes per wave */
    int k_chunks; /* 8-channel K chunks per lane */
    const char* wmma_op_id; /* NULL: the MFMA intrinsic */
} dw_toeplitz_atom_t;

/* The lane geometry of the Toeplitz atom at wave_size, without its op
 * (mirrors _dw_toeplitz_lanes): the one place the form's wave size picks the
 * MFMA (wave64, one K chunk per lane) or the WMMA (wave32, two). */
static dw_toeplitz_atom_t dw_toeplitz_lanes(int wave_size)
{
    dw_toeplitz_atom_t atom = {64, 1, NULL};
    if(wave_size == 32)
    {
        atom.wave = 32;
        atom.k_chunks = 2;
    }
    return atom;
}

/* VGPRs of one A, B or D fragment: 8 k_chunks 16-bit A / B slots or
 * 4 k_chunks f32 accumulators per lane. */
static int dw_toeplitz_frag_vgprs(const dw_toeplitz_atom_t* atom)
{
    return 4 * atom->k_chunks;
}

static int dw_win_mfma_in_passes(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    if(spec->dense)
    {
        const int win = rocke_direct_depthwise_dgrad_win_tile_w(spec) + spec->problem.KW - 1;
        const int per_px = rocke_direct_depthwise_dgrad_win_block_ch(spec) / ROCKE_DW_DGRAD_MFMA_CH;
        const int threads = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
        if(threads <= 0)
            return 0;
        return (int)rocke_ceil_div((long long)spec->w_fold * win * per_px, threads);
    }
    return rocke_dw_mfma_in_passes(rocke_direct_depthwise_dgrad_win_tile_w(spec),
                                   spec->problem.KW,
                                   spec->w_fold,
                                   rocke_direct_depthwise_dgrad_win_block_ch(spec),
                                   rocke_direct_depthwise_dgrad_win_threads_per_block(spec));
}

int rocke_direct_depthwise_dgrad_win_mfma_frag_vgprs(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    int kwp = rocke_dw_mfma_passes(p->KW) * dw_win_cin_chunks(p);
    if(spec->dense)
        kwp = p->KW * (p->cpg / ROCKE_DW_WIN_DENSE_K);
    const int rows = spec->prefetch_rows * dw_win_mfma_in_passes(spec);
    const dw_toeplitz_atom_t lanes = dw_toeplitz_lanes(spec->wave_size);
    const int frag = dw_toeplitz_frag_vgprs(&lanes);
    return frag * (p->KH * kwp + p->KH + kwp) + 4 * rows;
}

/* ---- swizzled LDS staging (lds_swz) ----
 * Both staging buffers hold 16-byte channel chunks without the per-pixel pad,
 * chunk c of local pixel lo of image f at f * IMG + lo * Q + (c + lo / ROT) % Q
 * (Q = block_ch / 8). ROT and the image pitch IMG are picked per kernel by
 * predicting the gfx950 LDS bank cycles of the four access shapes (mirrors
 * _dw_lds_phase / _dw_lds_cycles / _dw_win_swz_layout). */
namespace
{

struct DwSwzLay
{
    int w, img, rot;
};

struct DwLdsShape
{
    bool read;
    int width;
    std::vector<std::tuple<int, int, int>> lanes; /* (lane, px, channel) */
};

int dw_lds_phase(bool read, int width, int lane)
{
    const int b5 = (lane >> 5) & 1, b4 = (lane >> 4) & 1, b3 = (lane >> 3) & 1,
              b2 = (lane >> 2) & 1;
    if(read)
        return width == 8 ? b5 : (b5 << 1) | (b4 ^ b3 ^ b2);
    return width == 8 ? (b5 << 1) | b4 : (b5 << 2) | (b4 << 1) | b3;
}

int dw_lds_cycles(bool read, int width, const std::vector<std::pair<int, long long>>& addrs)
{
    const long long banks_n = read ? 64 : 32;
    std::map<int, std::set<long long>> phases;
    for(const auto& la : addrs)
        phases[dw_lds_phase(read, width, la.first)].insert(la.second);
    int tot = 0;
    for(const auto& ph : phases)
    {
        std::map<long long, std::set<long long>> banks;
        for(long long a : ph.second)
            for(int d = 0; d < width / 4; ++d)
            {
                const long long dw = a / 4 + d;
                banks[dw % banks_n].insert(dw);
            }
        int mx = 0;
        for(const auto& bk : banks)
            mx = (int)bk.second.size() > mx ? (int)bk.second.size() : mx;
        tot += mx;
    }
    return tot;
}

long long dw_swz_chunk(long long px, int c, int w, int img, int q, int rot)
{
    const long long f = px / w, lo = px % w;
    return f * img + lo * q + (rot ? (c + lo / rot) % q : c);
}

void dw_win_swz_layout(
    int KW, int F, int waves, int cpg, bool dense, int S, DwSwzLay* lay_in, DwSwzLay* lay_out)
{
    const int SUB = 16 / F;
    const int TW = dense ? 16 / F : 32 / F;
    const int KWP = (KW + 4) / 4;
    const int U = dense ? TW + KW - 1 : TW + 4 * KWP - 2;
    const int Q
        = waves * (dense ? ROCKE_DW_WIN_DENSE_CH : ROCKE_DW_DGRAD_MFMA_CH) / ROCKE_DW_DGRAD_MFMA_CH;
    const int T = waves * 64;

    auto side = [&](int w, const std::vector<DwLdsShape>& shapes) {
        const int span = w * Q;
        bool have = false;
        int b_cy = 0, b_img = 0, b_rot = 0;
        const int rots[4] = {0, 1, 2, 4};
        const int n_rot = Q > 1 ? 4 : 1;
        for(int ri = 0; ri < n_rot; ++ri)
        {
            const int rot = rots[ri];
            const int img_hi = F > 1 ? span + 17 : span + 1;
            for(int img = span; img < img_hi; ++img)
            {
                int cy = 0;
                for(const DwLdsShape& sh : shapes)
                {
                    std::vector<std::pair<int, long long>> addrs;
                    addrs.reserve(sh.lanes.size());
                    for(const auto& l : sh.lanes)
                    {
                        const int c = std::get<2>(l);
                        const long long a = dw_swz_chunk(std::get<1>(l), c / 8, w, img, Q, rot) * 16
                                            + (c % 8) * 2;
                        addrs.push_back({std::get<0>(l), a});
                    }
                    cy += dw_lds_cycles(sh.read, sh.width, addrs);
                }
                if(!have || std::make_tuple(cy, img, rot) < std::make_tuple(b_cy, b_img, b_rot))
                {
                    have = true;
                    b_cy = cy;
                    b_img = img;
                    b_rot = rot;
                }
            }
        }
        return DwSwzLay{w, b_img, b_rot};
    };

    std::vector<DwLdsShape> ins, outs;
    for(int wv = 0; wv < waves; ++wv)
    {
        const int wch = wv * (dense ? ROCKE_DW_WIN_DENSE_CH : ROCKE_DW_DGRAD_MFMA_CH);
        if(dense)
        {
            const int grp = wch / cpg * cpg;
            for(int s_ = 0; s_ < KW; ++s_)
                for(int h = 0; h < cpg / ROCKE_DW_WIN_DENSE_K; ++h)
                {
                    DwLdsShape sh{true, 16, {}};
                    for(int l = 0; l < 64; ++l)
                        sh.lanes.push_back({l,
                                            (l % 16) / SUB * U + (l % 16) % SUB + s_,
                                            grp + 8 * (l / 16) + ROCKE_DW_WIN_DENSE_K * h});
                    ins.push_back(sh);
                }
            DwLdsShape sh{false, 8, {}};
            for(int l = 0; l < 64; ++l)
                sh.lanes.push_back({l, (l % 16) / SUB * TW + (l % 16) % SUB, wch + 4 * (l / 16)});
            outs.push_back(sh);
        }
        else
        {
            const int g = cpg > ROCKE_DW_DGRAD_MFMA_CH ? cpg : ROCKE_DW_DGRAD_MFMA_CH;
            const int grp = cpg > 1 ? wch / g * g : wch;
            const int nh = cpg / ROCKE_DW_DGRAD_MFMA_CH > 1 ? cpg / ROCKE_DW_DGRAD_MFMA_CH : 1;
            for(int ph = 0; ph < KWP; ++ph)
                for(int h = 0; h < nh; ++h)
                {
                    DwLdsShape sh{true, 16, {}};
                    for(int l = 0; l < 64; ++l)
                        sh.lanes.push_back(
                            {l,
                             (l % 16) / SUB * U + 2 * ((l % 16) % SUB) + l / 16 + 4 * ph,
                             grp + 8 * h});
                    ins.push_back(sh);
                }
            DwLdsShape sh{false, 8, {}};
            for(int l = 0; l < 64; ++l)
                sh.lanes.push_back({l,
                                    (l % 16) / SUB * TW + 2 * ((l % 16) % SUB) + (l / 16) / 2,
                                    wch + 4 * ((l / 16) % 2)});
            outs.push_back(sh);
        }
        for(int pass = 0; pass < 2; ++pass)
        {
            const int n = pass == 0 ? F * U * Q : F * (TW / S) * Q;
            std::vector<DwLdsShape>& shapes = pass == 0 ? ins : outs;
            const bool read = pass == 1;
            for(int i = 0; i < (n + T - 1) / T; ++i)
            {
                DwLdsShape sh{read, 16, {}};
                for(int l = 0; l < 64; ++l)
                {
                    const int idx = wv * 64 + l + i * T;
                    if(idx < n)
                    {
                        int px = idx / Q;
                        if(read && S > 1)
                            px = px / (TW / S) * TW + px % (TW / S) * S;
                        sh.lanes.push_back({l, px, (idx % Q) * 8});
                    }
                }
                if(!sh.lanes.empty())
                    shapes.push_back(sh);
            }
        }
    }
    *lay_in = side(U, ins);
    *lay_out = side(TW, outs);
}

void dw_win_spec_swz_layout(const rocke_direct_depthwise_dgrad_win_spec_t* spec,
                            DwSwzLay* lay_in,
                            DwSwzLay* lay_out)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    dw_win_swz_layout(p->KW,
                      spec->w_fold,
                      spec->block_waves,
                      p->cpg,
                      spec->dense,
                      spec->out_stride,
                      lay_in,
                      lay_out);
}

/* swz_chunks(): 16-byte chunks per buffer of the swizzled dY and dX staging. */
void dw_win_swz_chunks(const rocke_direct_depthwise_dgrad_win_spec_t* spec,
                       long long* n_in_out,
                       long long* n_out_out)
{
    DwSwzLay li, lo;
    dw_win_spec_swz_layout(spec, &li, &lo);
    const int q = rocke_direct_depthwise_dgrad_win_block_ch(spec) / ROCKE_DW_DGRAD_MFMA_CH;
    const int threads = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    const long long in_px = rocke_ceil_div((long long)dw_win_mfma_in_passes(spec) * threads, q);
    long long n_in = 0, n_out = 0;
    for(long long px = 0; px < in_px; ++px)
        for(int c = 0; c < q; ++c)
        {
            const long long v = dw_swz_chunk(px, c, li.w, li.img, q, li.rot);
            n_in = v > n_in ? v : n_in;
        }
    const int out_px = spec->w_fold * rocke_direct_depthwise_dgrad_win_tile_w(spec);
    for(int px = 0; px < out_px; ++px)
        for(int c = 0; c < q; ++c)
        {
            const long long v = dw_swz_chunk(px, c, lo.w, lo.img, q, lo.rot);
            n_out = v > n_out ? v : n_out;
        }
    *n_in_out = n_in + 1;
    *n_out_out = n_out + 1;
}

/* dma_in_chunks(): the swizzled dY span rounded up to whole block-wide copies. */
long long dw_win_dma_in_chunks(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    long long n_in, n_out;
    dw_win_swz_chunks(spec, &n_in, &n_out);
    const int threads = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    return rocke_ceil_div(n_in, threads) * threads;
}

/* wdma_slot_chunks(): one filter row of a wave's 16 channels (KW x cpg). */
int dw_win_wdma_slot_chunks(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    return ROCKE_DW_WIN_DENSE_CH * spec->problem.KW * spec->problem.cpg * 2 / 16;
}

int dw_win_wdma_slots(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    return spec->prefetch_rows < spec->problem.KH ? spec->prefetch_rows : spec->problem.KH;
}

} // namespace

int rocke_direct_depthwise_dgrad_win_mfma_lds_bytes(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    if(spec->lds_swz)
    {
        long long n_in, n_out;
        dw_win_swz_chunks(spec, &n_in, &n_out);
        if(spec->dma)
        {
            long long w = 0;
            if(spec->wdma)
                w = (long long)dw_win_wdma_slots(spec) * spec->block_waves
                    * dw_win_wdma_slot_chunks(spec);
            return (int)(((spec->prefetch_rows + 1) * dw_win_dma_in_chunks(spec) + 2 * n_out + w)
                         * 16);
        }
        return (int)(2 * (n_in + n_out) * 16);
    }
    return (int)rocke_dw_mfma_lds_bytes(dw_win_mfma_in_passes(spec),
                                        rocke_direct_depthwise_dgrad_win_threads_per_block(spec),
                                        rocke_direct_depthwise_dgrad_win_block_ch(spec),
                                        spec->w_fold,
                                        rocke_direct_depthwise_dgrad_win_tile_w(spec));
}

rocke_status_t rocke_direct_depthwise_dgrad_win_kernel_name(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec, char* out, size_t out_cap)
{
    const rocke_direct_conv_problem_t* p;
    const char* name;
    int n;
    if(spec == NULL || out == NULL || out_cap == 0)
        return ROCKE_ERR_VALUE;
    p = &spec->problem;
    name = spec->name != NULL ? spec->name : "direct_depthwise_dgrad_win";
    char gtag[16] = "";
    if(p->cpg > 1)
        snprintf(gtag, sizeof gtag, "_g%d", p->cpg);
    if(spec->mfma)
        n = snprintf(out,
                     out_cap,
                     "%s_r%ds%dp%d_wv%d_bh%d_f%d_pf%d_mfma%s%s%s%s%s%s%s%s%s",
                     name,
                     p->KH,
                     p->KW,
                     p->PAD,
                     spec->block_waves,
                     spec->block_h,
                     spec->w_fold,
                     spec->prefetch_rows,
                     rocke_direct_conv_problem_is_bf16(p) ? "_bf16" : "",
                     spec->fwd ? "_fwd" : "",
                     gtag,
                     (spec->out_stride == 2 || (!spec->fwd && p->stride == 2)) ? "_s2" : "",
                     spec->dense ? "_dn" : "",
                     spec->lds_swz ? "_sw" : "",
                     spec->xcd_tiles ? "_xc" : "",
                     spec->dma ? "_dma" : "",
                     spec->wdma ? "_wd" : "");
    else
        n = snprintf(out,
                     out_cap,
                     "%s_r%ds%dp%d_bw%d_wv%d_cpl%d_bh%d%s%s",
                     name,
                     p->KH,
                     p->KW,
                     p->PAD,
                     spec->block_w,
                     spec->block_waves,
                     spec->ch_per_lane,
                     spec->block_h,
                     spec->dot2 ? "_dot2" : "",
                     rocke_direct_conv_problem_is_bf16(p) ? "_bf16" : "");
    if(n < 0 || (size_t)n >= out_cap)
        return ROCKE_ERR_VALUE;
    for(char* c = out; *c; ++c)
        if(*c == '/')
            *c = '_';
    return ROCKE_OK;
}

/* _depthwise_dgrad_win_check: true when the spec is buildable for its problem. */
static bool
    dw_win_check(const rocke_direct_depthwise_dgrad_win_spec_t* spec, char* reason, size_t cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const char* dtype = rocke_direct_conv_problem_dtype(p);
    if(strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0)
        ROCKE_SPEC_FAIL(reason, cap, "unsupported dtype '%s'; expected fp16 or bf16", dtype);
    /* The dgrad form (not fwd) is grouped too, on the wave64 MFMA; the forward
     * form also packs 2 or 4 small groups into one 8-channel chunk. */
    const bool grouped8
        = spec->mfma && (spec->fwd || spec->wave_size == 64) && p->cpg == p->kpg
          && (p->cpg % ROCKE_DW_DGRAD_MFMA_CH == 0 || (spec->fwd && (p->cpg == 2 || p->cpg == 4)))
          && p->cpg <= ROCKE_DW_WIN_MAX_GROUP_CH;
    if((p->cpg != 1 || p->kpg != 1) && !grouped8)
        ROCKE_SPEC_FAIL(reason, cap, "requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
    if(grouped8 && rocke_direct_depthwise_dgrad_win_block_ch(spec) % p->cpg != 0)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "a block must hold whole groups (block_ch=%d, cpg=%d)",
                        rocke_direct_depthwise_dgrad_win_block_ch(spec),
                        p->cpg);
    /* A stride-2 dgrad walks the zero-upsampled dY (wave64 MFMA, an even row
     * tile so every tile starts on a real dY row). */
    if(p->stride != 1
       && !(p->stride == 2 && spec->mfma && !spec->fwd && spec->wave_size == 64
            && spec->block_h % 2 == 0))
        ROCKE_SPEC_FAIL(
            reason, cap, "requires stride=1, or 2 on the mfma dgrad form (got %d)", p->stride);
    if(rocke_direct_dilation_reason(p, reason, cap))
        return false;
    const long long Ho = rocke_direct_conv_problem_Ho(p);
    const long long Wo = rocke_direct_conv_problem_Wo(p);
    if(p->PAD < 0 || Ho <= 0 || Wo <= 0)
        ROCKE_SPEC_FAIL(
            reason, cap, "degenerate geometry (PAD=%d, Ho=%lld, Wo=%lld)", p->PAD, Ho, Wo);
    if(spec->block_w < 1)
        ROCKE_SPEC_FAIL(reason, cap, "block_w must be >= 1 (got %d)", spec->block_w);
    if(spec->block_h < 1)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "block_h must be >= 1 (got %d); the image height is a kernarg, so a block "
                        "covers a tile of rows",
                        spec->block_h);
    const int threads = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    if(spec->block_waves < 1 || threads > 1024)
        ROCKE_SPEC_FAIL(
            reason, cap, "block_waves must give 1..1024 threads (got %d)", spec->block_waves);
    const int cpl = spec->ch_per_lane;
    if(cpl != 1 && cpl != 2 && cpl != 4 && cpl != 8)
        ROCKE_SPEC_FAIL(reason, cap, "ch_per_lane must be 1, 2, 4 or 8 (got %d)", cpl);
    if(p->groups % cpl != 0)
        ROCKE_SPEC_FAIL(
            reason, cap, "groups=%d is not a multiple of ch_per_lane=%d", p->groups, cpl);
    if(spec->dot2 && cpl != 1)
        ROCKE_SPEC_FAIL(reason, cap, "dot2 requires ch_per_lane=1 (got %d)", cpl);
    if(spec->prefetch_rows < 1)
        ROCKE_SPEC_FAIL(reason, cap, "prefetch_rows must be >= 1 (got %d)", spec->prefetch_rows);
    if(spec->fwd && !spec->mfma)
        ROCKE_SPEC_FAIL(reason, cap, "fwd requires mfma");
    if(spec->fwd && p->KH != p->KW)
        /* One PAD serves both axes; only a square filter keeps it one value. */
        ROCKE_SPEC_FAIL(reason, cap, "fwd requires a square filter (got %dx%d)", p->KH, p->KW);
    if((spec->out_stride != 1 && spec->out_stride != 2) || (spec->out_stride != 1 && !spec->fwd))
        ROCKE_SPEC_FAIL(
            reason, cap, "out_stride must be 1, or 2 with fwd (got %d)", spec->out_stride);
    if(spec->block_h % spec->out_stride != 0)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "block_h=%d must be a multiple of out_stride=%d",
                        spec->block_h,
                        spec->out_stride);
    if(spec->mfma)
    {
        if(spec->dot2 || cpl != 1)
            ROCKE_SPEC_FAIL(reason, cap, "mfma excludes dot2 and ch_per_lane > 1");
        /* The wave32 WMMA runs the depthwise forward (its tap index is
         * computed per K chunk). The grouped form loads its dense A row as
         * one 8-channel K chunk per lane (k_chunks == 1); at k_chunks == 2
         * that A operand would hold the wrong taps. The stride-2 output is
         * validated only on the wave64 MFMA (the gfx950 forward AOT variant). */
        if(spec->fwd && dw_toeplitz_lanes(spec->wave_size).k_chunks != 1)
        {
            if(p->cpg > 1)
                ROCKE_SPEC_FAIL(
                    reason, cap, "the grouped fwd form needs the wave64 MFMA (got wave_size=32)");
            if(spec->out_stride != 1)
                ROCKE_SPEC_FAIL(reason,
                                cap,
                                "fwd at wave_size=32 needs out_stride=1 (got %d)",
                                spec->out_stride);
        }
        if(spec->dense
           && !(p->cpg > 1 && spec->wave_size == 64 && p->cpg % ROCKE_DW_WIN_DENSE_K == 0))
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "dense needs the wave64 grouped form with cpg %% %d == 0"
                            " (got cpg=%d, wave_size=%d)",
                            ROCKE_DW_WIN_DENSE_K,
                            p->cpg,
                            spec->wave_size);
        if(!rocke_dw_mfma_w_fold_ok(spec->w_fold))
            ROCKE_SPEC_FAIL(reason, cap, "w_fold must be 1, 2 or 4 (got %d)", spec->w_fold);
        if(spec->lds_swz && spec->wave_size != 64)
            ROCKE_SPEC_FAIL(
                reason, cap, "lds_swz needs the wave64 MFMA (got wave_size=%d)", spec->wave_size);
        if(spec->xcd_tiles && spec->wave_size != 64)
            ROCKE_SPEC_FAIL(
                reason, cap, "xcd_tiles needs the wave64 MFMA (got wave_size=%d)", spec->wave_size);
        if(spec->dma
           && !(spec->lds_swz && spec->dense && spec->wave_size == 64
                && (spec->fwd || p->stride == 1)))
            ROCKE_SPEC_FAIL(
                reason, cap, "dma needs the dense lds_swz form (wave64, stride-1 window)");
        if(spec->wdma && !spec->dma)
            ROCKE_SPEC_FAIL(reason, cap, "wdma needs the dma form");
        if(spec->prefetch_rows > ROCKE_DW_DGRAD_MFMA_MAX_PREFETCH_ROWS)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "prefetch_rows must be <= %d (got %d)",
                            ROCKE_DW_DGRAD_MFMA_MAX_PREFETCH_ROWS,
                            spec->prefetch_rows);
        if(spec->block_waves > ROCKE_DW_DGRAD_MFMA_MAX_WAVES)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "mfma needs block_waves <= %d (got %d)",
                            ROCKE_DW_DGRAD_MFMA_MAX_WAVES,
                            spec->block_waves);
        if(!grouped8 && p->groups % ROCKE_DW_DGRAD_MFMA_CH != 0)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "mfma needs groups %% %d == 0 (got %d)",
                            ROCKE_DW_DGRAD_MFMA_CH,
                            p->groups);
        const int frag = rocke_direct_depthwise_dgrad_win_mfma_frag_vgprs(spec);
        if(frag > ROCKE_DW_DGRAD_MFMA_MAX_FRAG_VGPRS)
            ROCKE_SPEC_FAIL(
                reason,
                cap,
                "mfma fragments need %d VGPRs (> %d); lower prefetch_rows or the filter",
                frag,
                ROCKE_DW_DGRAD_MFMA_MAX_FRAG_VGPRS);
    }
    else if(spec->w_fold != 1 || spec->prefetch_rows != 1 || spec->dense)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "w_fold, prefetch_rows and dense require mfma (got w_fold=%d, "
                        "prefetch_rows=%d, dense=%s)",
                        spec->w_fold,
                        spec->prefetch_rows,
                        spec->dense ? "True" : "False");
    else if(spec->lds_swz || spec->xcd_tiles || spec->dma || spec->wdma)
        ROCKE_SPEC_FAIL(reason, cap, "lds_swz, xcd_tiles and dma require mfma");
    {
        const long long dy = (long long)p->N * Ho * Wo * p->groups * p->cpg * 2;
        const long long dx = (long long)p->N * p->H * p->W * p->groups * p->cpg * 2;
        const long long mx = dy > dx ? dy : dx;
        if(mx > ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "dY/dX of %lld bytes exceed the %lld-byte sentinel addressing range",
                            mx,
                            (long long)ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES);
    }
    if(spec->mfma)
    {
        const long long u = rocke_direct_depthwise_dgrad_win_unrolled_mfmas(spec);
        if(u > ROCKE_DW_DGRAD_MFMA_MAX_UNROLL)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "unrolled body too large (%lld MFMAs > %d); lower block_h",
                            u,
                            ROCKE_DW_DGRAD_MFMA_MAX_UNROLL);
        const int lds = rocke_direct_depthwise_dgrad_win_mfma_lds_bytes(spec);
        if(lds > ROCKE_DW_DGRAD_MFMA_LDS_BUDGET)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "mfma stages %d LDS bytes per workgroup (budget %d); lower block_waves",
                            lds,
                            ROCKE_DW_DGRAD_MFMA_LDS_BUDGET);
    }
    else
    {
        const long long u = rocke_direct_depthwise_dgrad_win_unrolled_fmas(spec);
        if(u > ROCKE_DW_DGRAD_WIN_MAX_UNROLL)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "unrolled body too large (%lld FMAs > %d); lower block_w or block_h",
                            u,
                            ROCKE_DW_DGRAD_WIN_MAX_UNROLL);
    }
    return true;
}

bool rocke_direct_depthwise_dgrad_win_is_valid_spec(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    if(spec == NULL)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx950";
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(spec->dot2 && strcmp(arch, "gfx950") != 0)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "dot2 needs one of ('gfx950',) (got %s)", arch);
        return false;
    }
    if(spec->mfma)
    {
        if(spec->wave_size == 32)
        {
            if(strcmp(arch, "gfx1250") != 0)
            {
                if(reason != NULL && reason_cap > 0)
                    snprintf(
                        reason, reason_cap, "wave32 mfma needs one of ('gfx1250',) (got %s)", arch);
                return false;
            }
        }
        else if(strcmp(arch, "gfx950") != 0)
        {
            /* A wave32 target runs the form on the WMMA, at wave_size 32. */
            const char* hint = strcmp(arch, "gfx1250") == 0 ? "; set wave_size=32" : "";
            if(reason != NULL && reason_cap > 0)
                snprintf(
                    reason, reason_cap, "mfma needs one of ('gfx950',) (got %s)%s", arch, hint);
            return false;
        }
    }
    if(spec->xcd_tiles && strcmp(arch, "gfx950") != 0)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "xcd_tiles is built for ['gfx950'] only (got %s)", arch);
        return false;
    }
    return dw_win_check(spec, reason, reason_cap);
}

/* ===================================================================== *
 *  Shared emission helpers
 * ===================================================================== */

void rocke_dw_win_byte_strides(rocke_ir_builder_t* b, const rocke_dconv_params_t* prm, V* out[6])
{
    V* c2 = rocke_b_const_i32(b, 2);
    out[0] = rocke_b_mul(b, prm->p_A_stride_n, c2); /* p_dY_stride_n  */
    out[1] = rocke_b_mul(b, prm->p_A_stride_hi, c2); /* p_dY_stride_ho */
    out[2] = rocke_b_mul(b, prm->p_A_stride_wi, c2); /* p_dY_stride_wo */
    out[3] = rocke_b_mul(b, prm->p_D_stride_n, c2); /* p_dX_stride_n  */
    out[4] = rocke_b_mul(b, prm->p_D_stride_ho, c2); /* p_dX_stride_hi */
    out[5] = rocke_b_mul(b, prm->p_D_stride_wo, c2); /* p_dX_stride_wi */
}

void rocke_dw_win_tile(rocke_ir_builder_t* b, V* p_Hi, int rows, V** n_tile, V** h0)
{
    rocke_dw_win_tile_ex(b, p_Hi, rows, NULL, n_tile, h0);
}

void rocke_dw_win_tile_ex(rocke_ir_builder_t* b, V* p_Hi, int rows, V* bz, V** n_tile, V** h0)
{
    if(bz == NULL)
        bz = rocke_b_block_id_z(b);
    V* c_rows = rocke_b_const_i32(b, rows);
    V* c_rm1 = rocke_b_const_i32(b, rows - 1);
    V* sum = rocke_b_add(b, p_Hi, c_rm1);
    V* h_tiles = rocke_b_div(b, sum, c_rows);
    *n_tile = rocke_b_div(b, bz, h_tiles);
    V* m = rocke_b_mul(b, *n_tile, h_tiles);
    V* d = rocke_b_sub(b, bz, m);
    *h0 = rocke_b_mul(b, d, c_rows);
}

V* rocke_dconv_buf_load_half(rocke_ir_builder_t* b, bool bf16, V* rsrc, V* voff, V* soff)
{
    return bf16 ? rocke_b_buffer_load_bf16(b, rsrc, voff, soff)
                : rocke_b_buffer_load_f16(b, rsrc, voff, soff);
}

V* rocke_dconv_buf_load_vN(rocke_ir_builder_t* b, bool bf16, V* rsrc, V* voff, V* soff, int dwords)
{
    return bf16 ? rocke_b_buffer_load_vN_bf16(b, rsrc, voff, soff, dwords)
                : rocke_b_buffer_load_vN_f16(b, rsrc, voff, soff, dwords);
}

void rocke_dconv_buf_store_vN(
    rocke_ir_builder_t* b, bool bf16, V* rsrc, V* voff, V* soff, V* val, int dwords)
{
    if(bf16)
        rocke_b_buffer_store_vN_bf16(b, rsrc, voff, soff, val, dwords);
    else
        rocke_b_buffer_store_vN_f16(b, rsrc, voff, soff, val, dwords);
}

V* rocke_dconv_trunc_f32(rocke_ir_builder_t* b, bool bf16, V* val)
{
    return bf16 ? rocke_b_vec_trunc_f32_to_bf16(b, val) : rocke_b_vec_trunc_f32_to_f16(b, val);
}

V* rocke_dconv_trunc_f32_elem(rocke_ir_builder_t* b, bool bf16, V* val)
{
    return bf16 ? rocke_b_trunc_f32_to_bf16(b, val) : rocke_b_trunc_f32_to_f16(b, val);
}

/* ===================================================================== *
 *  VALU / dot2 form
 * ===================================================================== */
static rocke_kernel_def_t* dw_win_build_valu(rocke_ir_builder_t* b,
                                             const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool bf16 = rocke_direct_conv_problem_is_bf16(p);
    const int BW = spec->block_w;
    const int CPL = spec->ch_per_lane;
    const int KH = p->KH, KW = p->KW, PAD = p->PAD;
    const int ROWS = spec->block_h;
    const int WIN = BW + KW - 1;
    const int N_PAIRS = (KW + 1) / 2;
    const bool DOT2 = spec->dot2;

    rocke_attr_set_int(b,
                       &b->kernel->attrs,
                       "max_workgroup_size",
                       rocke_direct_depthwise_dgrad_win_threads_per_block(spec));
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, rocke_direct_conv_problem_dtype(p));
    rocke_dconv_params_t prm;
    rocke_dconv_emit_params(b, &prm, "dgrad", io_type);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    V* p_Hi = prm.p_Hi;
    V* p_Wi = prm.p_Wi;
    V* p_Ho = prm.p_Ho;
    V* p_Wo = prm.p_Wo;

    V* c0 = rocke_b_const_i32(b, 0);
    V* oob_lane = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_LANE);
    V* oob_uniform = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_UNIFORM);
    V* zero_f32 = rocke_b_const_f32(b, 0.0);
    V* zero_acc = CPL > 1 ? rocke_b_vector_splat(b, zero_f32, CPL) : zero_f32;

    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* by = rocke_b_block_id_y(b);
    V* n = NULL;
    V* h0 = NULL;
    rocke_dw_win_tile(b, p_Hi, ROWS, &n, &h0);
    V* c_bw = rocke_b_const_i32(b, BW);
    V* wi0 = rocke_b_mul(b, bx, c_bw);
    V* lane_ch = tid;
    if(CPL > 1)
    {
        V* c_cpl = rocke_b_const_i32(b, CPL);
        lane_ch = rocke_b_mul(b, tid, c_cpl);
    }
    V* c_bch = rocke_b_const_i32(b, rocke_direct_depthwise_dgrad_win_block_ch(spec));
    V* by_ch = rocke_b_mul(b, by, c_bch);
    V* ch = rocke_b_add(b, by_ch, lane_ch);
    V* ch_ok = rocke_b_cmp_lt(b, ch, prm.p_groups);
    V* c2 = rocke_b_const_i32(b, 2);
    V* ch_bytes = rocke_b_mul(b, ch, c2);
    V* c_taps = rocke_b_const_i32(b, KH * KW * 2);
    V* ch_taps = rocke_b_mul(b, ch, c_taps);
    V* w_lane = rocke_b_select(b, ch_ok, ch_taps, oob_lane);
    V* bs[6];
    rocke_dw_win_byte_strides(b, &prm, bs);
    V *dy_img = bs[0], *dy_row_bytes = bs[1], *dy_px = bs[2];
    V *dx_img = bs[3], *dx_row_bytes = bs[4], *dx_px = bs[5];

    V* a_rsrc = rocke_b_buffer_rsrc(b, prm.A, prm.A_bytes);
    V* b_rsrc = rocke_b_buffer_rsrc(b, prm.Bp, prm.B_bytes);
    V* d_rsrc = rocke_b_buffer_rsrc(b, prm.D, prm.D_bytes);

    /* weights[r][k|s] */
    std::vector<std::vector<V*>> weights(KH);
    if(DOT2)
    {
        std::vector<std::vector<V*>> raw_w(KH, std::vector<V*>(KW));
        for(int r = 0; r < KH; ++r)
            for(int s = 0; s < KW; ++s)
            {
                V* c = rocke_b_const_i32(b, (r * KW + s) * 2);
                V* off = rocke_b_add(b, w_lane, c);
                raw_w[r][s] = rocke_dconv_buf_load_half(b, bf16, b_rsrc, off, c0);
            }
        V* zero_half = NULL;
        if(KW % 2)
            zero_half = rocke_dconv_trunc_f32_elem(b, bf16, zero_f32);
        for(int r = 0; r < KH; ++r)
            for(int k = 0; k < N_PAIRS; ++k)
            {
                V* comps[2] = {raw_w[r][2 * k], 2 * k + 1 < KW ? raw_w[r][2 * k + 1] : zero_half};
                weights[r].push_back(rocke_b_vec_pack(b, comps, 2, io_type));
            }
    }
    else
    {
        for(int r = 0; r < KH; ++r)
            for(int s = 0; s < KW; ++s)
            {
                std::vector<V*> comps(CPL);
                for(int i = 0; i < CPL; ++i)
                {
                    V* c = rocke_b_const_i32(b, ((i * KH + r) * KW + s) * 2);
                    V* off = rocke_b_add(b, w_lane, c);
                    V* h = rocke_dconv_buf_load_half(b, bf16, b_rsrc, off, c0);
                    comps[i] = rocke_b_cast_to_f32(b, h);
                }
                weights[r].push_back(CPL > 1 ? rocke_b_vec_pack(b, comps.data(), CPL, rocke_f32())
                                             : comps[0]);
            }
    }

    std::vector<V*> win_lane(WIN);
    for(int t = 0; t < WIN; ++t)
    {
        const int rel = PAD - (KW - 1) + t;
        V* c = rocke_b_const_i32(b, rel);
        V* wo = rocke_b_add(b, wi0, c);
        V* ok = rocke_b_cmp_lt(b, wo, p_Wo);
        if(rel < 0)
        {
            V* ge = rocke_b_cmp_ge(b, wo, c0);
            ok = rocke_b_land(b, ge, ok);
        }
        ok = rocke_b_land(b, ch_ok, ok);
        V* m = rocke_b_mul(b, wo, dy_px);
        V* a = rocke_b_add(b, ch_bytes, m);
        win_lane[t] = rocke_b_select(b, ok, a, oob_lane);
    }
    std::vector<V*> out_lane(BW);
    for(int j = 0; j < BW; ++j)
    {
        V* c = rocke_b_const_i32(b, j);
        V* wi = rocke_b_add(b, wi0, c);
        V* lt = rocke_b_cmp_lt(b, wi, p_Wi);
        V* ok = rocke_b_land(b, ch_ok, lt);
        V* m = rocke_b_mul(b, wi, dx_px);
        V* a = rocke_b_add(b, ch_bytes, m);
        out_lane[j] = rocke_b_select(b, ok, a, oob_lane);
    }
    V* dy_n = rocke_b_mul(b, n, dy_img);
    V* dx_n = rocke_b_mul(b, n, dx_img);

    const int n_rows = ROWS + KH - 1;
    std::vector<V*> row_ok(n_rows, NULL);

    auto load_window = [&](int y) -> std::vector<V*> {
        const int rel_ho = y + PAD - (KH - 1);
        V* c = rocke_b_const_i32(b, rel_ho);
        V* ho = rocke_b_add(b, h0, c);
        V* ok = rocke_b_cmp_lt(b, ho, p_Ho);
        if(rel_ho < 0)
        {
            V* ge = rocke_b_cmp_ge(b, ho, c0);
            ok = rocke_b_land(b, ge, ok);
        }
        row_ok[y] = ok;
        V* m = rocke_b_mul(b, ho, dy_row_bytes);
        V* part = rocke_b_add(b, dy_n, m);
        V* dy_row = rocke_b_select(b, ok, part, oob_uniform);
        std::vector<V*> raw(WIN);
        for(int t = 0; t < WIN; ++t)
        {
            V* off = rocke_b_add(b, win_lane[t], dy_row);
            raw[t] = CPL == 1 ? rocke_dconv_buf_load_half(b, bf16, a_rsrc, off, c0)
                              : rocke_dconv_buf_load_vN(b, bf16, a_rsrc, off, c0, CPL / 2);
        }
        rocke_b_sched_barrier(b, 0);
        return raw;
    };

    auto widen = [&](V* v) -> V* {
        if(CPL == 1)
            return rocke_b_cast_to_f32(b, v);
        std::vector<V*> comps(CPL);
        for(int i = 0; i < CPL; ++i)
        {
            V* e = rocke_b_vec_extract(b, v, i);
            comps[i] = rocke_b_cast_to_f32(b, e);
        }
        return rocke_b_vec_pack(b, comps.data(), CPL, rocke_f32());
    };

    auto taps = [&](const std::vector<int>& rs,
                    const std::vector<V*>& raw,
                    const std::vector<V*>& accs) -> std::vector<V*> {
        std::vector<V*> out = accs;
        if(DOT2)
        {
            const int t_lo = KW % 2 == 0 ? 0 : (KW > 1 ? 2 : WIN);
            std::vector<V*> pair(WIN, NULL);
            for(int t = t_lo; t < WIN; ++t)
            {
                V* comps[2] = {raw[t], t > 0 ? raw[t - 1] : raw[t]};
                pair[t] = rocke_b_vec_pack(b, comps, 2, io_type);
            }
            std::vector<V*> tail;
            if(KW % 2)
            {
                const rocke_type_t* v2 = rocke_vector_type(b, io_type, 2);
                for(int j = 0; j < BW; ++j)
                {
                    V* bits = rocke_b_bitcast(b, raw[j], rocke_i16());
                    V* z = rocke_b_zext(b, bits, rocke_i32());
                    tail.push_back(rocke_b_vec_bitcast(b, z, v2));
                }
            }
            std::vector<int> k_order;
            if(KW % 2)
                k_order.push_back(N_PAIRS - 1);
            for(int k = 0; k < (KW % 2 ? N_PAIRS - 1 : N_PAIRS); ++k)
                k_order.push_back(k);
            for(size_t i = 0; i < rs.size(); ++i)
            {
                const int r = rs[i];
                for(int k : k_order)
                    for(int j = 0; j < BW; ++j)
                    {
                        V* x = (2 * k + 1 == KW) ? tail[j] : pair[j + KW - 1 - 2 * k];
                        out[i * BW + j] = rocke_b_fdot2(b, weights[r][k], x, out[i * BW + j]);
                    }
            }
        }
        else
        {
            std::vector<V*> win(WIN);
            for(int t = 0; t < WIN; ++t)
                win[t] = widen(raw[t]);
            for(size_t i = 0; i < rs.size(); ++i)
            {
                const int r = rs[i];
                for(int s = 0; s < KW; ++s)
                    for(int j = 0; j < BW; ++j)
                    {
                        V* x = win[j + KW - 1 - s];
                        V* a = out[i * BW + j];
                        out[i * BW + j] = CPL > 1 ? rocke_b_vector_fma(b, weights[r][s], x, a)
                                                  : rocke_b_fma(b, weights[r][s], x, a);
                    }
            }
        }
        return out;
    };

    V* c1 = rocke_b_const_i32(b, 1);
    std::vector<V*> pending = load_window(0);
    std::vector<std::vector<V*>> acc(KH, std::vector<V*>(BW, zero_acc));
    for(int y = 0; y < n_rows; ++y)
    {
        std::vector<int> rs, slots;
        for(int r = 0; r < KH; ++r)
        {
            const int hl = y + r - (KH - 1);
            if(hl >= 0 && hl < ROWS)
            {
                rs.push_back(r);
                slots.push_back(hl % KH);
            }
        }
        std::vector<V*> raw = pending;
        if(y + 1 < n_rows)
            pending = load_window(y + 1);
        std::vector<V*> accs;
        for(int sl : slots)
            for(int j = 0; j < BW; ++j)
                accs.push_back(acc[sl][j]);
        const int rel_ho = y + PAD - (KH - 1);
        if(rel_ho < 0 || rel_ho >= ROWS)
            rocke_dw_win_guarded(
                b, row_ok[y], c0, c1, accs, "dww", y, true, [&](const std::vector<V*>& carried) {
                    return taps(rs, raw, carried);
                });
        else
            accs = taps(rs, raw, accs);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        for(size_t i = 0; i < slots.size(); ++i)
            for(int j = 0; j < BW; ++j)
                acc[slots[i]][j] = accs[i * BW + j];

        const int hi_local = y - (KH - 1);
        if(hi_local < 0 || hi_local >= ROWS)
            continue;
        const int slot = hi_local % KH;
        V* c = rocke_b_const_i32(b, hi_local);
        V* hi = rocke_b_add(b, h0, c);
        V* m = rocke_b_mul(b, hi, dx_row_bytes);
        V* part = rocke_b_add(b, dx_n, m);
        V* lt = rocke_b_cmp_lt(b, hi, p_Hi);
        V* dx_row = rocke_b_select(b, lt, part, oob_uniform);
        for(int j = 0; j < BW; ++j)
        {
            V* off = rocke_b_add(b, out_lane[j], dx_row);
            if(CPL == 1)
            {
                if(bf16)
                {
                    V* h = rocke_b_trunc_f32_to_bf16(b, acc[slot][j]);
                    rocke_b_buffer_store_bf16(b, d_rsrc, off, c0, h);
                }
                else
                {
                    V* h = rocke_b_trunc_f32_to_f16(b, acc[slot][j]);
                    rocke_b_buffer_store_f16(b, d_rsrc, off, c0, h);
                }
            }
            else
            {
                V* h = rocke_dconv_trunc_f32(b, bf16, acc[slot][j]);
                rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, h, CPL / 2);
            }
            acc[slot][j] = zero_acc;
        }
    }
    return b->kernel;
}

/* ===================================================================== *
 *  Toeplitz form: the 16x16x32 MFMA at wave64 (gfx950) or the 16x16x32 WMMA
 *  at wave32 (gfx1250), picked once by dw_toeplitz_atom. A lane carries
 *  k_chunks 8-channel K chunks j (1 MFMA, 2 WMMA) at u = k_chunks t + j: A two
 *  one-hot weight chunks, B one ds_read_b128 per chunk, D 4 k_chunks channels.
 * ===================================================================== */
/* The atom of spec on arch (no IR emitted): dw_toeplitz_lanes plus the WMMA
 * op on the wave32 form; false (error set) when the target has no wave32 WMMA
 * atom. */
static bool dw_toeplitz_atom(rocke_ir_builder_t* b,
                             const rocke_direct_depthwise_dgrad_win_spec_t* spec,
                             const char* arch,
                             dw_toeplitz_atom_t* atom)
{
    *atom = dw_toeplitz_lanes(spec->wave_size);
    if(atom->k_chunks == 1)
        return true;
    const char* ab = rocke_direct_conv_problem_is_bf16(&spec->problem) ? "bf16" : "f16";
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    const rocke_mmaop_t* op
        = rocke_archtarget_op_for_shape(target, "wmma", ab, ab, "fp32", 16, 16, 32, NULL);
    if(op == NULL || op->op_id == NULL)
    {
        rocke_i_set_err_msg(
            b, ROCKE_ERR_VALUE, "build_direct_depthwise_dgrad_windowed: no WMMA atom");
        return false;
    }
    atom->wmma_op_id = op->op_id;
    return true;
}

static V* dw_toeplitz_mma(
    rocke_ir_builder_t* b, const dw_toeplitz_atom_t* atom, bool bf16, V* a, V* x, V* acc)
{
    if(atom->wmma_op_id != NULL)
        return rocke_b_mma(b, atom->wmma_op_id, a, x, acc, NULL, 0);
    return bf16 ? rocke_b_mfma_f32_16x16x32_bf16(b, a, x, acc)
                : rocke_b_mfma_f32_16x16x32_f16(b, a, x, acc);
}

static rocke_kernel_def_t* dw_win_build_toeplitz(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_win_spec_t* spec, const char* arch)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool bf16 = rocke_direct_conv_problem_is_bf16(p);
    const int KH = p->KH, KW = p->KW, PAD = p->PAD;
    const int F = spec->w_fold;
    const int SUB = 16 / F;
    const int TW = rocke_direct_depthwise_dgrad_win_tile_w(spec);
    const int KWP = rocke_dw_mfma_passes(KW);
    const bool DN = spec->dense;
    const int U = DN ? TW + KW - 1 : TW + 4 * KWP - 2; /* window columns per image */
    const int ROWS = spec->block_h;
    const int THREADS = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    const int BLOCK_CH = rocke_direct_depthwise_dgrad_win_block_ch(spec);
    const int PER_PX = BLOCK_CH / ROCKE_DW_DGRAD_MFMA_CH;
    const int CH_PAD = BLOCK_CH + ROCKE_DW_DGRAD_MFMA_CH;
    const int PF = spec->prefetch_rows;
    const bool SW = spec->lds_swz;
    const bool DMA = spec->dma;
    const bool WD = spec->wdma;
    const bool FWD = spec->fwd;
    const int S = spec->out_stride;
    /* A stride-2 dgrad is the stride-1 dgrad over dY upsampled with zeros
     * (dY_up[2 i] = dY[i]): odd window rows are skipped at build time (block_h
     * is even, so h0 is), odd window columns read zero. */
    const int IS = FWD ? 1 : p->stride;
    /* First window column relative to the tile: the forward pads both axes by
     * the same amount, the row's (equal to the dgrad's for a square filter). */
    const int COL0 = FWD ? PAD - (KH - 1) : PAD - (KW - 1);
    auto ci = [&](int64_t v) { return rocke_b_const_i32(b, v); };

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, rocke_direct_conv_problem_dtype(p));
    rocke_dconv_params_t prm;
    rocke_dconv_emit_params(b, &prm, FWD ? "fwd" : "dgrad", io_type);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    V* p_N = prm.p_N;
    V* p_Hi = prm.p_Hi;
    V* p_Wi = prm.p_Wi;
    V* p_Ho = prm.p_Ho;
    V* p_Wo = prm.p_Wo;
    V* c_S = NULL;
    if(FWD)
    {
        /* The forward ABI: the rows produced (dX) are the forward output, the
         * rows streamed (dY) the forward input; a stride-S output is every
         * S-th row and column of the stride-1 grid walked, S times its extent. */
        p_Hi = prm.p_Ho;
        p_Wi = prm.p_Wo;
        if(S > 1)
        {
            c_S = ci(S);
            p_Hi = rocke_b_mul(b, p_Hi, c_S);
            p_Wi = rocke_b_mul(b, p_Wi, c_S);
        }
        p_Ho = prm.p_Hi;
        p_Wo = prm.p_Wi;
    }
    V* p_G = prm.p_groups;
    const bool G8 = p->cpg > 1;
    const int CPG = p->cpg;
    const int H8 = dw_win_cin_chunks(p);
    if(G8)
    {
        /* The channel count bounding every access: cpg channels per group. */
        V* c_ch = ci(CPG);
        p_G = rocke_b_mul(b, p_G, c_ch);
    }

    V* c0 = ci(0);
    V* oob_lane = ci(ROCKE_DW_DGRAD_WIN_OOB_LANE);
    V* oob_uniform = ci(ROCKE_DW_DGRAD_WIN_OOB_UNIFORM);
    dw_toeplitz_atom_t atom;
    if(!dw_toeplitz_atom(b, spec, arch, &atom))
        return NULL;
    const int J = atom.k_chunks;
    const int ACC = 4 * J; /* f32 accumulators (D channels) per lane */
    V* zero_acc = rocke_b_zero_vec_f32(b, ACC);
    V* zero_f32 = rocke_b_const_f32(b, 0.0);
    V* zero_half = rocke_dconv_trunc_f32_elem(b, bf16, zero_f32);

    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* by = rocke_b_block_id_y(b);
    V* bz = NULL;
    if(spec->xcd_tiles)
    {
        /* Launch order is column tile fastest, then channel block, then (image
         * tile, row band); remapped to XCD-contiguous runs so the column tiles
         * sharing a window halo land in one L2. Only the XCD count is baked. */
        V* k1 = ci(TW - 1);
        V* a1 = rocke_b_add(b, p_Wi, k1);
        V* k2 = ci(TW);
        V* xc_gx = rocke_b_div(b, a1, k2);
        V* k3 = ci(BLOCK_CH - 1);
        V* a2 = rocke_b_add(b, p_G, k3);
        V* k4 = ci(BLOCK_CH);
        V* xc_gy = rocke_b_div(b, a2, k4);
        V* k5 = ci(F - 1);
        V* a3 = rocke_b_add(b, p_N, k5);
        V* k6 = ci(F);
        V* d1 = rocke_b_div(b, a3, k6);
        V* k7 = ci(ROWS - 1);
        V* a4 = rocke_b_add(b, p_Hi, k7);
        V* k8 = ci(ROWS);
        V* d2 = rocke_b_div(b, a4, k8);
        V* xc_gz = rocke_b_mul(b, d1, d2);
        V* xc_gxy = rocke_b_mul(b, xc_gx, xc_gy);
        V* m1 = rocke_b_mul(b, by, xc_gx);
        V* bzr = rocke_b_block_id_z(b);
        V* m2 = rocke_b_mul(b, bzr, xc_gxy);
        V* s1 = rocke_b_add(b, m1, m2);
        V* xc_lin = rocke_b_add(b, bx, s1);
        V* num_wgs = rocke_b_mul(b, xc_gxy, xc_gz);
        /* _XCD_TILES_ARCHES[arch]: is_valid_spec admits gfx950 only. */
        xc_lin = rocke_chiplet_transform_chunked_runtime(b, xc_lin, num_wgs, 8, NULL);
        if(xc_lin == NULL)
            return NULL;
        bx = rocke_b_mod(b, xc_lin, xc_gx);
        V* d3 = rocke_b_div(b, xc_lin, xc_gx);
        by = rocke_b_mod(b, d3, xc_gy);
        bz = rocke_b_div(b, xc_lin, xc_gxy);
    }
    V* n_tile = NULL;
    V* h0 = NULL;
    rocke_dw_win_tile_ex(b, p_Hi, ROWS, bz, &n_tile, &h0);
    V* c_wave_size = ci(atom.wave);
    V* lane = rocke_b_mod(b, tid, c_wave_size);
    V* wave = rocke_b_div(b, tid, c_wave_size);
    V* c_tw = ci(TW);
    V* x0 = rocke_b_mul(b, bx, c_tw);
    V* c_f = ci(F);
    V* img0 = rocke_b_mul(b, n_tile, c_f);
    V* c_bch = ci(BLOCK_CH);
    V* ch_blk = rocke_b_mul(b, by, c_bch);
    V* c_mch = ci(DN ? ROCKE_DW_WIN_DENSE_CH : ROCKE_DW_DGRAD_MFMA_CH);
    V* wave_ch = rocke_b_mul(b, wave, c_mch);
    V* c_wave = rocke_b_add(b, ch_blk, wave_ch);
    V* ch_ok = rocke_b_cmp_lt(b, c_wave, p_G);
    V* c16 = ci(16);
    V* c8 = ci(8);
    V* col16 = rocke_b_mod(b, lane, c16);
    V* t = rocke_b_div(b, lane, c16);
    V* k = rocke_b_mod(b, col16, c8);
    V* q = rocke_b_div(b, col16, c8);
    V* bs[6];
    rocke_dw_win_byte_strides(b, &prm, bs);
    V *dy_img = bs[0], *dy_row_bytes = bs[1], *dy_px = bs[2];
    V *dx_img = bs[3], *dx_row_bytes = bs[4], *dx_px = bs[5];

    V* a_rsrc = rocke_b_buffer_rsrc(b, prm.A, prm.A_bytes);
    V* b_rsrc = rocke_b_buffer_rsrc(b, prm.Bp, prm.B_bytes);
    V* d_rsrc = rocke_b_buffer_rsrc(b, prm.D, prm.D_bytes);

    /* One-hot A fragments weights[r][ph]: W[k, r, KW - 1 - s] at slot 8 j + k,
     * s = 4 ph + u - q; or (grouped form, wave64 only) the group's dense row
     * W[k, ., ., 8h .. 8h+7] as one 16-byte load of input chunk h. */
    const int tap_bytes = CPG * 2;
    const int kk_bytes = KH * KW * tap_bytes;
    V* ck = rocke_b_add(b, c_wave, k);
    V* c_taps = ci(KH * KW * tap_bytes);
    V* w_base = rocke_b_mul(b, ck, c_taps);
    if(G8 && !FWD)
    {
        /* The grouped dgrad contracts the group's dY channels: slot j of input
         * chunk h is W[kb + 8 h + j, r, KW - 1 - s, c_loc], gathered with
         * 16-bit loads (the forward layout keeps the dX channel innermost). */
        V* c_cpg_w = ci(CPG);
        V* dv = rocke_b_div(b, c_wave, c_cpg_w);
        V* kb = rocke_b_mul(b, dv, c_cpg_w);
        V* md = rocke_b_mod(b, c_wave, c_cpg_w);
        V* c_loc = rocke_b_add(b, md, k);
        V* kc = ci(kk_bytes);
        V* m1 = rocke_b_mul(b, kb, kc);
        V* k2 = ci(2);
        V* m2 = rocke_b_mul(b, c_loc, k2);
        w_base = rocke_b_add(b, m1, m2);
    }
    V* c_KW = ci(KW);
    V* is_k[ROCKE_DW_DGRAD_MFMA_CH];
    if(!G8)
        for(int j = 0; j < ROCKE_DW_DGRAD_MFMA_CH; ++j)
        {
            V* c = ci(j);
            is_k[j] = rocke_b_cmp_eq(b, k, c);
        }
    const int H32 = DN ? CPG / ROCKE_DW_WIN_DENSE_K : 0;
    /* Dense A fragments dweights[r][s][h]: lane (m, t) holds output channel
     * c_wave + m against the 8 input channels 32 h + 8 t .. + 7 of its group at
     * tap (r, s). With wdma they are filled row by row from the LDS slots. */
    std::vector<std::vector<std::vector<V*>>> dweights(KH);
    if(DN)
    {
        V* c_m = rocke_b_add(b, c_wave, col16);
        V* kb = NULL;
        V* c_loc2 = NULL;
        if(!FWD)
        {
            V* c_cpg_w = ci(CPG);
            V* dv = rocke_b_div(b, c_m, c_cpg_w);
            kb = rocke_b_mul(b, dv, c_cpg_w);
            V* md = rocke_b_mod(b, c_m, c_cpg_w);
            V* k2 = ci(2);
            c_loc2 = rocke_b_mul(b, md, k2);
        }
        for(int r = 0; r < (WD ? 0 : KH); ++r)
        {
            std::vector<std::vector<V*>> row;
            for(int s_ = 0; s_ < KW; ++s_)
            {
                std::vector<V*> hs;
                for(int h = 0; h < H32; ++h)
                {
                    V* m8 = rocke_b_mul(b, t, c8);
                    V* kh32 = ci(ROCKE_DW_WIN_DENSE_K * h);
                    V* kin = rocke_b_add(b, m8, kh32);
                    if(FWD)
                    {
                        const int tap = (KH - 1 - r) * KW + s_;
                        V* kc = ci(kk_bytes);
                        V* m1 = rocke_b_mul(b, c_m, kc);
                        V* kt = ci(tap * tap_bytes);
                        V* k2 = ci(2);
                        V* m2 = rocke_b_mul(b, kin, k2);
                        V* a1 = rocke_b_add(b, kt, m2);
                        V* off = rocke_b_add(b, m1, a1);
                        V* sel = rocke_b_select(b, ch_ok, off, oob_lane);
                        hs.push_back(rocke_dconv_buf_load_vN(b, bf16, b_rsrc, sel, c0, 4));
                        continue;
                    }
                    const int tap = r * KW + KW - 1 - s_;
                    V* a0 = rocke_b_add(b, kb, kin);
                    V* kc = ci(kk_bytes);
                    V* m1 = rocke_b_mul(b, a0, kc);
                    V* kt = ci(tap * tap_bytes);
                    V* a1 = rocke_b_add(b, kt, c_loc2);
                    V* base = rocke_b_add(b, m1, a1);
                    V* halves[ROCKE_DW_DGRAD_MFMA_CH];
                    for(int i = 0; i < ROCKE_DW_DGRAD_MFMA_CH; ++i)
                    {
                        V* ki = ci(i * kk_bytes);
                        V* ai = rocke_b_add(b, base, ki);
                        V* sel = rocke_b_select(b, ch_ok, ai, oob_lane);
                        halves[i] = rocke_dconv_buf_load_half(b, bf16, b_rsrc, sel, c0);
                    }
                    hs.push_back(rocke_b_vec_pack(b, halves, ROCKE_DW_DGRAD_MFMA_CH, io_type));
                }
                row.push_back(hs);
            }
            dweights[r] = row;
        }
    }
    std::vector<std::vector<V*>> weights(KH, std::vector<V*>(KWP));
    std::vector<std::vector<std::vector<V*>>> gweights(
        KH, std::vector<std::vector<V*>>(KWP, std::vector<V*>(H8)));
    for(int r = 0; r < (DN ? 0 : KH); ++r)
        for(int ph = 0; ph < KWP; ++ph)
        {
            V* comps[2 * ROCKE_DW_DGRAD_MFMA_CH];
            for(int j = 0; j < J; ++j)
            {
                V* u = t;
                if(J != 1)
                {
                    V* cJ = ci(J);
                    u = rocke_b_mul(b, t, cJ);
                }
                V* c = ci(4 * ph + j);
                V* tp = rocke_b_add(b, u, c);
                V* s = rocke_b_sub(b, tp, q);
                V* ge = rocke_b_cmp_ge(b, s, c0);
                V* lt = rocke_b_cmp_lt(b, s, c_KW);
                V* in = rocke_b_land(b, ge, lt);
                V* ok = rocke_b_land(b, ch_ok, in);
                V* tap;
                if(FWD)
                {
                    /* The rotated filter: W[k, KH - 1 - r, s]. */
                    V* cr = ci((KH - 1 - r) * KW);
                    tap = rocke_b_add(b, cr, s);
                }
                else
                {
                    V* cr = ci(r * KW + KW - 1);
                    tap = rocke_b_sub(b, cr, s);
                }
                V* c2 = ci(tap_bytes);
                V* tb = rocke_b_mul(b, tap, c2);
                V* a = rocke_b_add(b, w_base, tb);
                V* off = rocke_b_select(b, ok, a, oob_lane);
                if(G8 && !FWD)
                {
                    for(int h = 0; h < H8; ++h)
                    {
                        V* halves[ROCKE_DW_DGRAD_MFMA_CH];
                        for(int i = 0; i < ROCKE_DW_DGRAD_MFMA_CH; ++i)
                        {
                            V* ki = ci((8 * h + i) * kk_bytes);
                            V* ai = rocke_b_add(b, off, ki);
                            V* sel = rocke_b_select(b, ok, ai, oob_lane);
                            halves[i] = rocke_dconv_buf_load_half(b, bf16, b_rsrc, sel, c0);
                        }
                        gweights[r][ph][h]
                            = rocke_b_vec_pack(b, halves, ROCKE_DW_DGRAD_MFMA_CH, io_type);
                    }
                    continue;
                }
                if(G8 && CPG < ROCKE_DW_DGRAD_MFMA_CH)
                {
                    /* Block-diagonal chunk: slot i (channel c_wave + i) is in
                     * channel k's group iff i / cpg == k / cpg; its weight is
                     * W[k, ., ., i % cpg]. */
                    V* ws[ROCKE_DW_DGRAD_MFMA_CH];
                    for(int i = 0; i < CPG; ++i)
                    {
                        V* ki = ci(2 * i);
                        V* ai = rocke_b_add(b, off, ki);
                        ws[i] = rocke_dconv_buf_load_half(b, bf16, b_rsrc, ai, c0);
                    }
                    V* kcpg = ci(CPG);
                    V* k_grp = rocke_b_div(b, k, kcpg);
                    V* sl[ROCKE_DW_DGRAD_MFMA_CH];
                    for(int i = 0; i < ROCKE_DW_DGRAD_MFMA_CH; ++i)
                    {
                        V* kg = ci(i / CPG);
                        V* eq = rocke_b_cmp_eq(b, k_grp, kg);
                        sl[i] = rocke_b_select(b, eq, ws[i % CPG], zero_half);
                    }
                    gweights[r][ph][0] = rocke_b_vec_pack(b, sl, ROCKE_DW_DGRAD_MFMA_CH, io_type);
                    continue;
                }
                if(G8)
                {
                    for(int h = 0; h < H8; ++h)
                    {
                        V* ch = ci(16 * h);
                        V* oh = rocke_b_add(b, off, ch);
                        gweights[r][ph][h] = rocke_dconv_buf_load_vN(b, bf16, b_rsrc, oh, c0, 4);
                    }
                    continue;
                }
                V* w = rocke_dconv_buf_load_half(b, bf16, b_rsrc, off, c0);
                for(int i = 0; i < ROCKE_DW_DGRAD_MFMA_CH; ++i)
                    comps[j * ROCKE_DW_DGRAD_MFMA_CH + i]
                        = rocke_b_select(b, is_k[i], w, zero_half);
            }
            if(!G8)
                weights[r][ph] = rocke_b_vec_pack(b, comps, J * ROCKE_DW_DGRAD_MFMA_CH, io_type);
        }
    V* grp_ch = NULL;
    if(G8)
    {
        /* The first channel of the wave's group inside the block. */
        V* c_cpg = ci(CPG > ROCKE_DW_DGRAD_MFMA_CH ? CPG : ROCKE_DW_DGRAD_MFMA_CH);
        V* gd = rocke_b_div(b, wave_ch, c_cpg);
        grp_ch = rocke_b_mul(b, gd, c_cpg);
    }

    V* c_sub = ci(SUB);
    V* f = rocke_b_div(b, col16, c_sub);
    V* cm = rocke_b_mod(b, col16, c_sub);
    V* c2a = ci(2);
    V* pair2 = rocke_b_mul(b, cm, c2a);
    V* c_per_px = ci(PER_PX);

    /* sw_at(lay, px, ch): (chunk, element) of channel ch of pixel px in a
     * swizzled staging buffer; the chunk is memoised per (buffer, px, ch). */
    DwSwzLay lay[2] = {{0, 0, 0}, {0, 0, 0}}; /* 0: dY staging, 1: dX staging */
    std::map<std::tuple<int, V*, V*>, V*> sw_memo;
    auto sw_at = [&](int li, V* px, V* ch, V** chunk, V** elem) {
        auto key = std::make_tuple(li, px, ch);
        auto it = sw_memo.find(key);
        if(it == sw_memo.end())
        {
            const DwSwzLay& L = lay[li];
            V* kw = ci(L.w);
            V* fi = rocke_b_div(b, px, kw);
            V* kw2 = ci(L.w);
            V* lo = rocke_b_mod(b, px, kw2);
            V* c = rocke_b_div(b, ch, c8);
            if(L.rot)
            {
                V* kr = ci(L.rot);
                V* lr = rocke_b_div(b, lo, kr);
                V* a = rocke_b_add(b, c, lr);
                V* kq = ci(PER_PX);
                c = rocke_b_mod(b, a, kq);
            }
            V* ki = ci(L.img);
            V* m1 = rocke_b_mul(b, fi, ki);
            V* kq = ci(PER_PX);
            V* m2 = rocke_b_mul(b, lo, kq);
            V* a1 = rocke_b_add(b, m1, m2);
            it = sw_memo.emplace(key, rocke_b_add(b, a1, c)).first;
        }
        *chunk = it->second;
        *elem = rocke_b_mod(b, ch, c8);
    };
    long long sw_in_n = 0, sw_out_n = 0;
    if(SW)
    {
        dw_win_spec_swz_layout(spec, &lay[0], &lay[1]);
        dw_win_swz_chunks(spec, &sw_in_n, &sw_out_n);
    }

    /* dX staging [buffer, image column, channel]: lane (n, t) writes its D
     * fragment (MFMA: 4 channels of column 2 (n % SUB) + t / 2; WMMA: the
     * wave's 8 channels of column 2 (n % SUB) + t). */
    V* o_smem;
    if(SW)
    {
        const int o_shape[3] = {2, (int)sw_out_n, ROCKE_DW_DGRAD_MFMA_CH};
        o_smem = rocke_b_smem_alloc(b, io_type, o_shape, 3, "dx_stage");
    }
    else
    {
        const int o_shape[3] = {2, F * TW, CH_PAD};
        o_smem = rocke_b_smem_alloc(b, io_type, o_shape, 3, "dx_stage");
    }
    V* c_tw2 = ci(TW);
    V* fo = rocke_b_mul(b, f, c_tw2);
    V* th = t;
    if(J != 2)
    {
        V* c2b = ci(2);
        th = rocke_b_div(b, t, c2b);
    }
    V* pt = rocke_b_add(b, pair2, th);
    V* o_col = rocke_b_add(b, fo, pt);
    V* o_ch = wave_ch;
    if(J != 2)
    {
        V* c2c = ci(2);
        V* tm = rocke_b_mod(b, t, c2c);
        V* c4 = ci(4);
        V* tm4 = rocke_b_mul(b, tm, c4);
        o_ch = rocke_b_add(b, wave_ch, tm4);
    }
    if(DN)
    {
        /* Lane (n, t): channels wave_ch + 4 t .. + 3 of column n % SUB. */
        V* kt = ci(TW);
        V* fm = rocke_b_mul(b, f, kt);
        V* md = rocke_b_mod(b, col16, c_sub);
        o_col = rocke_b_add(b, fm, md);
        V* k4 = ci(4);
        V* t4 = rocke_b_mul(b, t, k4);
        o_ch = rocke_b_add(b, wave_ch, t4);
    }
    V* o_at[2] = {o_col, o_ch};
    if(SW)
        sw_at(1, o_col, o_ch, &o_at[0], &o_at[1]);
    const int n_out = F * (TW / S) * PER_PX;
    V* c_TW = ci(TW);
    V* c_TWS = NULL;
    V* x0_out = NULL;
    if(S > 1)
    {
        c_TWS = ci(TW / S);
        x0_out = rocke_b_div(b, x0, c_S);
    }
    std::vector<rocke_dw_win_plan_t> drain;
    for(int i = 0; i < rocke_ceil_div(n_out, THREADS); ++i)
    {
        V* cii = ci(i * THREADS);
        V* idx = rocke_b_add(b, tid, cii);
        V* px = rocke_b_div(b, idx, c_per_px);
        V* im = rocke_b_mod(b, idx, c_per_px);
        V* ch8 = rocke_b_mul(b, im, c8);
        V* img_o;
        V* wi_o;
        V* wi_st;
        if(S > 1)
        {
            /* Output pixel px is staged column S * (px % (TW / S)) of image
             * px / (TW / S); its stride-1 column bounds it against p_Wi. */
            V* f_o = rocke_b_div(b, px, c_TWS);
            V* cs = rocke_b_mod(b, px, c_TWS);
            img_o = rocke_b_add(b, img0, f_o);
            V* fm = rocke_b_mul(b, f_o, c_TW);
            V* cm2 = rocke_b_mul(b, cs, c_S);
            px = rocke_b_add(b, fm, cm2);
            V* cm3 = rocke_b_mul(b, cs, c_S);
            wi_o = rocke_b_add(b, x0, cm3);
            wi_st = rocke_b_add(b, x0_out, cs);
        }
        else
        {
            V* pd = rocke_b_div(b, px, c_TW);
            img_o = rocke_b_add(b, img0, pd);
            V* pm = rocke_b_mod(b, px, c_TW);
            wi_o = rocke_b_add(b, x0, pm);
            wi_st = wi_o;
        }
        V* ch_o = rocke_b_add(b, ch_blk, ch8);
        V* cn = ci(n_out);
        V* l1 = rocke_b_cmp_lt(b, idx, cn);
        V* l2 = rocke_b_cmp_lt(b, img_o, p_N);
        V* a1 = rocke_b_land(b, l1, l2);
        V* l3 = rocke_b_cmp_lt(b, wi_o, p_Wi);
        V* l4 = rocke_b_cmp_lt(b, ch_o, p_G);
        V* a2 = rocke_b_land(b, l3, l4);
        V* ok = rocke_b_land(b, a1, a2);
        V* m1 = rocke_b_mul(b, img_o, dx_img);
        V* m2 = rocke_b_mul(b, wi_st, dx_px);
        V* c2 = ci(2);
        V* m3 = rocke_b_mul(b, ch_o, c2);
        V* s1 = rocke_b_add(b, m2, m3);
        V* g_off = rocke_b_add(b, m1, s1);
        rocke_dw_win_plan_t e = {px, ch8, rocke_b_select(b, ok, g_off, oob_lane)};
        drain.push_back(e);
    }

    /* dY staging [buffer, window column, channel]: the block loads the U
     * window columns of each image with 16-byte chunks; padding columns,
     * images past p_N and channels past p_groups read zero through the lane
     * sentinel. */
    const int n_in = F * U * PER_PX;
    const int in_passes = rocke_ceil_div(n_in, THREADS);
    const int in_px = rocke_ceil_div(in_passes * THREADS, PER_PX);
    const int NBUF = PF + 1;
    const int NSD = DMA ? (int)dw_win_dma_in_chunks(spec) : 0;
    V* i_smem;
    if(DMA)
    {
        const int i_shape[3] = {NBUF, NSD, ROCKE_DW_DGRAD_MFMA_CH};
        i_smem = rocke_b_smem_alloc(b, io_type, i_shape, 3, "dy_stage");
    }
    else if(SW)
    {
        const int i_shape[3] = {2, (int)sw_in_n, ROCKE_DW_DGRAD_MFMA_CH};
        i_smem = rocke_b_smem_alloc(b, io_type, i_shape, 3, "dy_stage");
    }
    else
    {
        const int i_shape[3] = {2, in_px, CH_PAD};
        i_smem = rocke_b_smem_alloc(b, io_type, i_shape, 3, "dy_stage");
    }
    V* c_U = ci(U);
    std::vector<rocke_dw_win_plan_t> in_plan;
    const int n_dma = DMA ? NSD / THREADS : 0;
    for(int i = 0; i < (DMA ? n_dma : in_passes); ++i)
    {
        V* cii = ci(i * THREADS);
        V* idx = rocke_b_add(b, tid, cii);
        V* px;
        V* ch8;
        V* img_i;
        V* wo;
        V* slot_ok = NULL;
        if(DMA)
        {
            /* LDS slot idx of the swizzled ring buffer: invert sw_at (image
             * pitch, pixel, rotated chunk) to the source pixel and chunk. */
            const DwSwzLay& L = lay[0];
            V* ki = ci(L.img);
            V* fi = rocke_b_div(b, idx, ki);
            V* ki2 = ci(L.img);
            V* rem = rocke_b_mod(b, idx, ki2);
            V* lo = rocke_b_div(b, rem, c_per_px);
            V* cs = rocke_b_mod(b, rem, c_per_px);
            if(L.rot)
            {
                V* a = rocke_b_add(b, cs, c_per_px);
                V* kr = ci(L.rot);
                V* lr = rocke_b_div(b, lo, kr);
                V* lm = rocke_b_mod(b, lr, c_per_px);
                V* sb = rocke_b_sub(b, a, lm);
                cs = rocke_b_mod(b, sb, c_per_px);
            }
            V* fu = rocke_b_mul(b, fi, c_U);
            px = rocke_b_add(b, fu, lo);
            ch8 = rocke_b_mul(b, cs, c8);
            img_i = rocke_b_add(b, img0, fi);
            V* kc = ci(COL0);
            V* lc = rocke_b_add(b, lo, kc);
            wo = rocke_b_add(b, x0, lc);
            V* kf = ci(F);
            V* l1 = rocke_b_cmp_lt(b, fi, kf);
            V* ku = ci(U * PER_PX);
            V* l2 = rocke_b_cmp_lt(b, rem, ku);
            slot_ok = rocke_b_land(b, l1, l2);
        }
        else
        {
            px = rocke_b_div(b, idx, c_per_px);
            V* im = rocke_b_mod(b, idx, c_per_px);
            ch8 = rocke_b_mul(b, im, c8);
            V* pd = rocke_b_div(b, px, c_U);
            img_i = rocke_b_add(b, img0, pd);
            V* pm = rocke_b_mod(b, px, c_U);
            V* cp = ci(COL0);
            V* pmp = rocke_b_add(b, pm, cp);
            wo = rocke_b_add(b, x0, pmp);
        }
        V* ch_i = rocke_b_add(b, ch_blk, ch8);
        if(slot_ok == NULL)
        {
            V* cn = ci(n_in);
            slot_ok = rocke_b_cmp_lt(b, idx, cn);
        }
        V* l2 = rocke_b_cmp_lt(b, img_i, p_N);
        V* ok = rocke_b_land(b, slot_ok, l2);
        if(IS > 1)
        {
            /* dY_up column wo is dY column wo / 2 when even, zero when odd. */
            V* c_IS = ci(IS);
            V* wm = rocke_b_mod(b, wo, c_IS);
            V* ev = rocke_b_cmp_eq(b, wm, c0);
            ok = rocke_b_land(b, ok, ev);
            wo = rocke_b_div(b, wo, c_IS);
        }
        V* g1 = rocke_b_cmp_ge(b, wo, c0);
        V* l3 = rocke_b_cmp_lt(b, wo, p_Wo);
        V* a2 = rocke_b_land(b, g1, l3);
        V* l4 = rocke_b_cmp_lt(b, ch_i, p_G);
        V* a3 = rocke_b_land(b, a2, l4);
        ok = rocke_b_land(b, ok, a3);
        V* m1 = rocke_b_mul(b, img_i, dy_img);
        V* m2 = rocke_b_mul(b, wo, dy_px);
        V* c2 = ci(2);
        V* m3 = rocke_b_mul(b, ch_i, c2);
        V* s1 = rocke_b_add(b, m2, m3);
        V* g_off = rocke_b_add(b, m1, s1);
        rocke_dw_win_plan_t e = {px, ch8, rocke_b_select(b, ok, g_off, oob_lane)};
        in_plan.push_back(e);
    }
    V* f_u = rocke_b_mul(b, f, c_U);
    /* b_px[ph][j]: window column 2 (n % SUB) + u + 4 ph of chunk j. */
    V* u0 = t;
    if(J != 1)
    {
        V* c2t = ci(J);
        u0 = rocke_b_mul(b, t, c2t);
    }
    std::vector<std::vector<V*>> b_px(KWP, std::vector<V*>(J));
    for(int ph = 0; ph < KWP; ++ph)
        for(int j = 0; j < J; ++j)
        {
            V* c = ci(4 * ph + j);
            V* tc = rocke_b_add(b, u0, c);
            V* pt2 = rocke_b_add(b, pair2, tc);
            b_px[ph][j] = rocke_b_add(b, f_u, pt2);
        }
    /* Dense form: lane (n, t) reads window column n % SUB + s of image n / SUB. */
    std::vector<V*> b_px_dn;
    if(DN)
        for(int s_ = 0; s_ < KW; ++s_)
        {
            V* md = rocke_b_mod(b, col16, c_sub);
            V* ks = ci(s_);
            V* a = rocke_b_add(b, md, ks);
            b_px_dn.push_back(rocke_b_add(b, f_u, a));
        }
    /* Dense form: lane (n, t) reads channels grp_ch + 8 t + 32 h. */
    std::vector<V*> dn_ch;
    if(DN && SW)
        for(int h = 0; h < H32; ++h)
        {
            V* m8 = rocke_b_mul(b, t, c8);
            V* kh = ci(ROCKE_DW_WIN_DENSE_K * h);
            V* a = rocke_b_add(b, m8, kh);
            dn_ch.push_back(rocke_b_add(b, grp_ch, a));
        }

    const int n_rows = ROWS + KH - 1;
    std::vector<V*> row_ok(n_rows, NULL);
    std::vector<bool> row_zero(n_rows, false); /* a zero row of dY_up (IS > 1) */
    /* VMEM ops in issue order: 'd' + row copies, 'w' + row weight copies, 's' stores. */
    std::vector<std::string> vm_log;
    V* dma_base = NULL;
    if(DMA)
    {
        V* addr = rocke_b_smem_addr_of(b, i_smem);
        V* kw16 = ci(atom.wave * 16);
        V* m = rocke_b_mul(b, wave, kw16);
        V* z = rocke_b_zext(b, m, rocke_i64());
        V* pa = rocke_b_smem_ptr_add(b, addr, z);
        V* tr = rocke_b_trunc(b, pa, rocke_i32());
        dma_base = rocke_b_readfirstlane(b, tr);
    }
    /* _dma_lds_at: 16-byte buffer_load ... lds of every lane into LDS at the
     * wave-uniform base plus the constant imm (inline asm; the kernel's explicit
     * s_waitcnt vmcnt before the barrier is the only wait). */
    auto dma_lds_at = [&](V* rsrc, V* base, long long imm, V* voff, V* soff) {
        std::string tmpl = "s_add_u32 m0, $3, " + std::to_string(imm)
                           + "\n\tbuffer_load_dwordx4 $0, $1, $2 offen lds";
        V* ops[4] = {voff, rsrc, soff, base};
        rocke_b_inline_asm(
            b, tmpl.c_str(), "v,s,s,s,~{m0},~{scc},~{memory}", ops, 4, NULL, 0, NULL);
    };

    const int WSC = WD ? dw_win_wdma_slot_chunks(spec) : 0;
    const int NWS = WD ? dw_win_wdma_slots(spec) : 0;
    const int WAVES = spec->block_waves;
    const int tap_b = CPG * 2;
    V* w_smem = NULL;
    V* w_dma_base = NULL;
    std::vector<V*> w_src;
    V *tr_j = NULL, *tr_el = NULL, *tr_half = NULL;
    if(WD)
    {
        /* Per-wave weight slots: slot q of wave w at chunk (q * WAVES + w) * WSC. */
        const int w_shape[2] = {NWS * WAVES * WSC, ROCKE_DW_DGRAD_MFMA_CH};
        w_smem = rocke_b_smem_alloc(b, io_type, w_shape, 2, "w_stage");
        V* addr = rocke_b_smem_addr_of(b, w_smem);
        V* kw16 = ci(WSC * 16);
        V* m = rocke_b_mul(b, wave, kw16);
        V* z = rocke_b_zext(b, m, rocke_i64());
        V* pa = rocke_b_smem_ptr_add(b, addr, z);
        V* tr = rocke_b_trunc(b, pa, rocke_i32());
        w_dma_base = rocke_b_readfirstlane(b, tr);
        for(int i = 0; i < WSC / atom.wave; ++i)
        {
            V* ke = ci(i * atom.wave);
            V* e = rocke_b_add(b, lane, ke);
            V* off;
            if(FWD)
            {
                /* Chunk e = channel m's (KW x cpg) filter row, 16 B piece rem. */
                const int per_m = KW * CPG / 8;
                V* kp = ci(per_m);
                V* dv = rocke_b_div(b, e, kp);
                V* a = rocke_b_add(b, c_wave, dv);
                V* kc = ci(kk_bytes);
                V* m1 = rocke_b_mul(b, a, kc);
                V* kp2 = ci(per_m);
                V* md = rocke_b_mod(b, e, kp2);
                V* m2 = rocke_b_mul(b, md, c16);
                off = rocke_b_add(b, m1, m2);
            }
            else
            {
                /* Chunk e = (K row kr, tap column s, half): W[kb + kr, r,
                 * KW - 1 - s, c_loc0 + 8 half .. + 7]. */
                V* c_cpg_u = ci(CPG);
                V* dv = rocke_b_div(b, c_wave, c_cpg_u);
                V* kb_w = rocke_b_mul(b, dv, c_cpg_u);
                V* c_loc0 = rocke_b_mod(b, c_wave, c_cpg_u);
                V* k2kw = ci(2 * KW);
                V* kr = rocke_b_div(b, e, k2kw);
                V* k2kw2 = ci(2 * KW);
                V* em = rocke_b_mod(b, e, k2kw2);
                V* k2 = ci(2);
                V* s_e = rocke_b_div(b, em, k2);
                V* k2b = ci(2);
                V* half = rocke_b_mod(b, e, k2b);
                V* a0 = rocke_b_add(b, kb_w, kr);
                V* kc = ci(kk_bytes);
                V* m1 = rocke_b_mul(b, a0, kc);
                V* kkw = ci(KW - 1);
                V* sb = rocke_b_sub(b, kkw, s_e);
                V* ktb = ci(tap_b);
                V* m2 = rocke_b_mul(b, sb, ktb);
                V* hm = rocke_b_mul(b, half, c8);
                V* a1 = rocke_b_add(b, c_loc0, hm);
                V* k2c = ci(2);
                V* m3 = rocke_b_mul(b, a1, k2c);
                V* a2 = rocke_b_add(b, m2, m3);
                off = rocke_b_add(b, m1, a2);
            }
            w_src.push_back(rocke_b_select(b, ch_ok, off, oob_lane));
        }
        if(!FWD)
        {
            V* k4 = ci(4);
            tr_j = rocke_b_div(b, col16, k4);
            V* k4b = ci(4);
            V* tr_a = rocke_b_mod(b, col16, k4b);
            V* k2 = ci(2);
            V* am = rocke_b_mod(b, tr_a, k2);
            V* k4c = ci(4);
            tr_el = rocke_b_mul(b, am, k4c);
            V* k2b = ci(2);
            tr_half = rocke_b_div(b, tr_a, k2b);
        }
    }

    /* Copy the filter row needed first by window row y (r = KH-1-y) into the
     * wave's slot y % NWS. */
    auto dma_w = [&](int y) {
        const int r = KH - 1 - y;
        const int tap_row = FWD ? (KH - 1 - r) : r;
        V* soff = ci((int64_t)tap_row * KW * tap_b);
        for(size_t i = 0; i < w_src.size(); ++i)
        {
            dma_lds_at(b_rsrc,
                       w_dma_base,
                       ((long long)(y % NWS) * WAVES * WSC + (long long)i * atom.wave) * 16,
                       w_src[i],
                       soff);
            vm_log.push_back("w" + std::to_string(y));
        }
    };
    /* Weights row r = KH-1-y (dweights[r][s][h]) from slot y % NWS. */
    auto read_w = [&](int y) {
        V* kw = ci(WSC);
        V* m = rocke_b_mul(b, wave, kw);
        V* ko = ci((int64_t)(y % NWS) * WAVES * WSC);
        V* sb = rocke_b_add(b, m, ko);
        std::vector<std::vector<V*>> row;
        for(int s_ = 0; s_ < KW; ++s_)
        {
            std::vector<V*> hs;
            for(int h = 0; h < H32; ++h)
            {
                if(FWD)
                {
                    V* k1 = ci(KW * CPG / 8);
                    V* m1 = rocke_b_mul(b, col16, k1);
                    V* k2 = ci(s_ * CPG / 8 + 4 * h);
                    V* a1 = rocke_b_add(b, t, k2);
                    V* a2 = rocke_b_add(b, m1, a1);
                    V* ch = rocke_b_add(b, sb, a2);
                    V* ix[2] = {ch, c0};
                    hs.push_back(rocke_b_smem_load_vN(b, w_smem, ix, 2, io_type, 8));
                    continue;
                }
                V* parts[2];
                for(int pp = 0; pp < 2; ++pp)
                {
                    V* m8 = rocke_b_mul(b, t, c8);
                    V* kp = ci(ROCKE_DW_WIN_DENSE_K * h + 4 * pp);
                    V* aj = rocke_b_add(b, tr_j, kp);
                    V* kr = rocke_b_add(b, m8, aj);
                    V* kkw = ci(KW);
                    V* mk = rocke_b_mul(b, kr, kkw);
                    V* ks = ci(s_);
                    V* as = rocke_b_add(b, mk, ks);
                    V* k2 = ci(2);
                    V* m2 = rocke_b_mul(b, as, k2);
                    V* ah = rocke_b_add(b, m2, tr_half);
                    V* ch = rocke_b_add(b, sb, ah);
                    V* ix[2] = {ch, tr_el};
                    parts[pp] = rocke_b_ds_read_tr16_b64(b, w_smem, ix, 2, io_type);
                }
                hs.push_back(rocke_b_vec_concat(b, parts[0], parts[1]));
            }
            row.push_back(hs);
        }
        return row;
    };

    /* Copy the dY window row y into ring buffer y % NBUF. */
    auto dma_row = [&](int y) -> std::vector<V*> {
        if(WD && y < KH)
            dma_w(y);
        V* kr = ci(y + PAD - (KH - 1));
        V* ho = rocke_b_add(b, h0, kr);
        V* ge = rocke_b_cmp_ge(b, ho, c0);
        V* lt = rocke_b_cmp_lt(b, ho, p_Ho);
        V* ok = rocke_b_land(b, ge, lt);
        row_ok[y] = ok;
        V* m = rocke_b_mul(b, ho, dy_row_bytes);
        V* dy_row = rocke_b_select(b, ok, m, oob_uniform);
        for(size_t i = 0; i < in_plan.size(); ++i)
        {
            V* voff = rocke_b_add(b, in_plan[i].g_off, dy_row);
            dma_lds_at(a_rsrc,
                       dma_base,
                       ((long long)(y % NBUF) * NSD + (long long)i * THREADS) * 16,
                       voff,
                       c0);
            vm_log.push_back("d" + std::to_string(y));
        }
        return std::vector<V*>();
    };

    auto load_row = [&](int y) -> std::vector<V*> {
        if(DMA)
            return dma_row(y);
        const int rel_ho = y + PAD - (KH - 1);
        std::vector<V*> vals;
        if(IS > 1)
        {
            /* dY_up row h0 + rel_ho (h0 even): an odd one is zero (never
             * loaded), an even one is dY row h0 / IS + rel_ho / IS. */
            const int rem = ((rel_ho % IS) + IS) % IS;
            if(rem)
            {
                row_zero[y] = true;
                return vals;
            }
            V* kis = ci(IS);
            V* hd = rocke_b_div(b, h0, kis);
            const int fl = (rel_ho - rem) / IS;
            V* kr = ci(fl);
            V* ho = rocke_b_add(b, hd, kr);
            V* ge = rocke_b_cmp_ge(b, ho, c0);
            V* lt = rocke_b_cmp_lt(b, ho, p_Ho);
            V* ok = rocke_b_land(b, ge, lt);
            row_ok[y] = ok;
            V* m = rocke_b_mul(b, ho, dy_row_bytes);
            V* dy_row = rocke_b_select(b, ok, m, oob_uniform);
            for(const rocke_dw_win_plan_t& e : in_plan)
            {
                V* off = rocke_b_add(b, e.g_off, dy_row);
                vals.push_back(rocke_dconv_buf_load_vN(b, bf16, a_rsrc, off, c0, 4));
            }
            rocke_b_sched_barrier(b, 0);
            return vals;
        }
        V* c = ci(rel_ho);
        V* ho = rocke_b_add(b, h0, c);
        V* ok = rocke_b_cmp_lt(b, ho, p_Ho);
        if(rel_ho < 0)
        {
            V* ge = rocke_b_cmp_ge(b, ho, c0);
            ok = rocke_b_land(b, ge, ok);
        }
        row_ok[y] = ok;
        V* m = rocke_b_mul(b, ho, dy_row_bytes);
        V* dy_row = rocke_b_select(b, ok, m, oob_uniform);
        for(const rocke_dw_win_plan_t& e : in_plan)
        {
            V* off = rocke_b_add(b, e.g_off, dy_row);
            vals.push_back(rocke_dconv_buf_load_vN(b, bf16, a_rsrc, off, c0, 4));
        }
        rocke_b_sched_barrier(b, 0);
        return vals;
    };

    auto dx_row_part = [&](int hi_local) -> V* {
        V* c = ci(hi_local);
        V* hi = rocke_b_add(b, h0, c);
        /* A stride-S output row is stride-1 row hi / S. */
        V* hr = S > 1 ? rocke_b_div(b, hi, c_S) : hi;
        V* part = rocke_b_mul(b, hr, dx_row_bytes);
        V* lt = rocke_b_cmp_lt(b, hi, p_Hi);
        return rocke_b_select(b, lt, part, oob_uniform);
    };

    auto drain_out = [&](V* buf, V* dx_row) {
        for(const rocke_dw_win_plan_t& e : drain)
        {
            V* v;
            if(SW)
            {
                V *chunk, *elem;
                sw_at(1, e.px, e.ch8, &chunk, &elem);
                V* idx[3] = {buf, chunk, elem};
                v = rocke_b_smem_load_vN(b, o_smem, idx, 3, io_type, 8);
            }
            else
            {
                V* idx[3] = {buf, e.px, e.ch8};
                v = rocke_b_smem_load_vN(b, o_smem, idx, 3, io_type, 8);
            }
            V* off = rocke_b_add(b, e.g_off, dx_row);
            rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, v, 4);
            vm_log.push_back("s");
        }
    };

    std::vector<std::vector<V*>> pending(n_rows);
    for(int y = 0; y < (PF < n_rows ? PF : n_rows); ++y)
        pending[y] = load_row(y);

    auto stage_row = [&](const std::vector<V*>& raw, V* buf) {
        for(size_t i = 0; i < in_plan.size(); ++i)
        {
            if(SW)
            {
                V *chunk, *elem;
                sw_at(0, in_plan[i].px, in_plan[i].ch8, &chunk, &elem);
                V* idx[3] = {buf, chunk, elem};
                rocke_b_smem_store_vN(b, i_smem, idx, 3, raw[i], 8);
            }
            else
            {
                V* idx[3] = {buf, in_plan[i].px, in_plan[i].ch8};
                rocke_b_smem_store_vN(b, i_smem, idx, 3, raw[i], 8);
            }
        }
    };

    /* Swizzled form: every B-read address once, ahead of the guarded rows. */
    std::map<std::tuple<int, int, int>, std::pair<V*, V*>> sw_b;
    if(SW && !DN)
        for(int ph = 0; ph < KWP; ++ph)
            for(int h = 0; h < (G8 ? H8 : 1); ++h)
            {
                V* ch_h = wave_ch;
                if(G8)
                {
                    V* k8h = ci(8 * h);
                    ch_h = rocke_b_add(b, grp_ch, k8h);
                }
                for(int j = 0; j < J; ++j)
                {
                    V *chunk, *elem;
                    sw_at(0, b_px[ph][j], ch_h, &chunk, &elem);
                    sw_b[std::make_tuple(ph, j, h)] = {chunk, elem};
                }
            }

    /* The B fragment of pass ph: one 8-channel read per K chunk. */
    auto b_frag = [&](V* buf, int ph, V* ch, int h) -> V* {
        V* reads[2];
        for(int j = 0; j < J; ++j)
        {
            V* idx[3];
            idx[0] = buf;
            if(SW)
            {
                const std::pair<V*, V*>& a = sw_b[std::make_tuple(ph, j, h)];
                idx[1] = a.first;
                idx[2] = a.second;
            }
            else
            {
                idx[1] = b_px[ph][j];
                idx[2] = ch;
            }
            reads[j] = rocke_b_smem_load_vN(b, i_smem, idx, 3, io_type, 8);
        }
        return J == 1 ? reads[0] : rocke_b_vec_concat(b, reads[0], reads[1]);
    };

    auto multiply = [&](const std::vector<V*>& accs, V* buf, const std::vector<int>& rs) {
        std::vector<V*> out = accs;
        if(DN)
        {
            for(int s_ = 0; s_ < KW; ++s_)
                for(int h = 0; h < H32; ++h)
                {
                    V* ch_dn;
                    if(SW)
                        ch_dn = dn_ch[h];
                    else
                    {
                        V* m8 = rocke_b_mul(b, t, c8);
                        V* kh = ci(ROCKE_DW_WIN_DENSE_K * h);
                        V* a = rocke_b_add(b, m8, kh);
                        ch_dn = rocke_b_add(b, grp_ch, a);
                    }
                    V* idx[3];
                    idx[0] = buf;
                    if(SW)
                        sw_at(0, b_px_dn[s_], ch_dn, &idx[1], &idx[2]);
                    else
                    {
                        idx[1] = b_px_dn[s_];
                        idx[2] = ch_dn;
                    }
                    V* x = rocke_b_smem_load_vN(b, i_smem, idx, 3, io_type, 8);
                    for(size_t i = 0; i < rs.size(); ++i)
                        out[i] = dw_toeplitz_mma(b, &atom, bf16, dweights[rs[i]][s_][h], x, out[i]);
                }
            return out;
        }
        if(G8)
        {
            /* Input chunk h of the wave's group: its 8 channels at grp_ch + 8h. */
            std::vector<std::vector<V*>> gx(KWP, std::vector<V*>(H8));
            for(int ph = 0; ph < KWP; ++ph)
                for(int h = 0; h < H8; ++h)
                {
                    if(SW)
                        gx[ph][h] = b_frag(buf, ph, NULL, h);
                    else
                    {
                        V* c8h = ci(8 * h);
                        V* chh = rocke_b_add(b, grp_ch, c8h);
                        gx[ph][h] = b_frag(buf, ph, chh, h);
                    }
                }
            for(int ph = 0; ph < KWP; ++ph)
                for(int h = 0; h < H8; ++h)
                    for(size_t i = 0; i < rs.size(); ++i)
                        out[i] = dw_toeplitz_mma(
                            b, &atom, bf16, gweights[rs[i]][ph][h], gx[ph][h], out[i]);
            return out;
        }
        std::vector<V*> xs(KWP);
        for(int ph = 0; ph < KWP; ++ph)
            xs[ph] = b_frag(buf, ph, wave_ch, 0);
        for(int ph = 0; ph < KWP; ++ph)
            for(size_t i = 0; i < rs.size(); ++i)
                out[i] = dw_toeplitz_mma(b, &atom, bf16, weights[rs[i]][ph], xs[ph], out[i]);
        return out;
    };

    /* The KH - 1 halo rows of the dY window run under a block-uniform guard:
     * the staging under an scf.if, the B reads and MFMAs in a one-trip loop
     * carrying the accumulator tiles they touch. */
    if(SW)
    {
        /* Every other swizzled address ahead of the guarded rows too (the memo
         * must not hand a value defined inside a guard to a later row). */
        V *chunk, *elem;
        for(const rocke_dw_win_plan_t& e : in_plan)
            sw_at(0, e.px, e.ch8, &chunk, &elem);
        for(const rocke_dw_win_plan_t& e : drain)
            sw_at(1, e.px, e.ch8, &chunk, &elem);
        for(int s_ = 0; s_ < (DN ? KW : 0); ++s_)
            for(int h = 0; h < H32; ++h)
                sw_at(0, b_px_dn[s_], dn_ch[h], &chunk, &elem);
    }
    V* c1 = ci(1);
    bool have_out = false;
    V* pend_val = NULL;
    V* pend_row = NULL;
    std::vector<V*> acc(KH, zero_acc);
    for(int y = 0; y < n_rows; ++y)
    {
        /* Output rows y + r - (KH - 1) of the tile; a stride-S output keeps
         * every S-th one. */
        std::vector<int> rs, slots;
        for(int r = 0; r < KH; ++r)
        {
            const int hl = y + r - (KH - 1);
            if(hl >= 0 && hl < ROWS && hl % S == 0)
            {
                rs.push_back(r);
                slots.push_back(hl % KH);
            }
        }
        std::vector<V*> raw = pending[y];
        V* buf = ci(y % 2);
        V* ibuf = DMA ? ci(y % NBUF) : buf;
        const int rel_ho = y + PAD - (KH - 1);
        const bool halo = rel_ho < 0 || rel_ho >= ROWS;

        if(DMA)
        {
            /* Row y's copies done (the younger copies and stores may fly). */
            const std::string tag = "d" + std::to_string(y);
            int last = -1;
            for(int i = 0; i < (int)vm_log.size(); ++i)
                if(vm_log[i] == tag)
                    last = i;
            rocke_b_s_waitcnt(b, (int)vm_log.size() - 1 - last, -1, -1);
            if(WD && y < KH)
            {
                /* The slot's filter row landed with (before) row y's copies;
                 * the empty asm keeps its LDS reads below the wait. */
                rocke_b_inline_asm(b, ";", "~{memory}", NULL, 0, NULL, 0, NULL);
                dweights[KH - 1 - y] = read_w(y);
                if(y + NWS < KH)
                    /* Slot y % NWS is refilled right after this row's barrier. */
                    rocke_b_s_waitcnt(b, -1, 0, -1);
            }
        }
        else if(row_zero[y])
        {
            /* A zero row of dY_up (IS > 1): no staging, no MFMAs. */
            rs.clear();
            slots.clear();
        }
        else if(halo)
        {
            rocke_if_t g = rocke_b_scf_if(b, row_ok[y]);
            rocke_b_region_enter(b, g.then_region);
            stage_row(raw, buf);
            rocke_b_region_leave(b);
        }
        else
            stage_row(raw, buf);
        if(have_out)
        {
            V* idx[3] = {buf, o_at[0], o_at[1]};
            rocke_b_smem_store_vN(b, o_smem, idx, 3, pend_val, ACC);
        }
        rocke_b_sync_lds_only(b);
        if(have_out)
        {
            drain_out(buf, pend_row);
            have_out = false;
        }
        if(y + PF < n_rows)
            pending[y + PF] = load_row(y + PF);
        std::vector<V*> accs;
        for(int sl : slots)
            accs.push_back(acc[sl]);
        if(rs.empty())
        {
            /* A stride-2 window row whose output rows are all dropped. */
        }
        else if(halo)
            rocke_dw_win_guarded(
                b, row_ok[y], c0, c1, accs, "dwm", y, true, [&](const std::vector<V*>& a) {
                    return multiply(a, ibuf, rs);
                });
        else
            accs = multiply(accs, ibuf, rs);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        for(size_t i = 0; i < slots.size(); ++i)
            acc[slots[i]] = accs[i];
        const int hi_local = y - (KH - 1);
        if(hi_local < 0 || hi_local >= ROWS || hi_local % S != 0)
            continue;
        const int slot = hi_local % KH;
        pend_val = rocke_dconv_trunc_f32(b, bf16, acc[slot]);
        pend_row = dx_row_part(hi_local);
        have_out = true;
        acc[slot] = zero_acc;
    }
    if(have_out)
    {
        V* buf = ci(n_rows % 2);
        V* idx[3] = {buf, o_at[0], o_at[1]};
        rocke_b_smem_store_vN(b, o_smem, idx, 3, pend_val, ACC);
        rocke_b_sync_lds_only(b);
        drain_out(buf, pend_row);
    }
    return b->kernel;
}

/* ===================================================================== *
 *  Entry
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_win(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_win_spec_t* spec, const char* arch)
{
    if(spec->mfma)
        return dw_win_build_toeplitz(b, spec, arch);
    return dw_win_build_valu(b, spec);
}

rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_win_new(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_win_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        char why[512];
        if(b == NULL || spec == NULL)
            return NULL;
        /* Validated before the Python builds its IRBuilder. */
        if(!rocke_direct_depthwise_dgrad_win_is_valid_spec(spec, arch, why, sizeof why))
        {
            rocke_ir_builder_init(b, "direct_depthwise_dgrad_win_invalid");
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "invalid DirectDepthwiseDgradWindowedSpec for %s: %s",
                            arch ? arch : "gfx950",
                            why);
            return NULL;
        }
        if(rocke_direct_depthwise_dgrad_win_kernel_name(spec, name, sizeof name) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_depthwise_dgrad_win(b, spec, arch);
    });
}
