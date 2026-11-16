// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dgrad_pp_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_dgrad_pp_mfma.py: the gfx950 LDS-DMA
 * ping-pong MFMA direct dgrad (build_dgrad_pp_mfma).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_dgrad_pp_mfma.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "rocke/arch_target.h" /* rocke_arch_target_from_gfx, rocke_mma_catalog_has_shape */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_{async_lds_max_dwords,mma} */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_ceil_div */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_direct_wgrad_internal.hpp"
#include "rocke/intexpr.h" /* rocke_intexpr_checked_mul */

using namespace rocke_dwg;

namespace
{

constexpr int kCK = 64; /* channels per chunk */
constexpr int kGran = 8; /* halves per 16-byte granule */
constexpr int kTH = 16; /* output rows per block */
constexpr int kTW = 16; /* output columns per block */
constexpr int kWaves = 8;
constexpr int kRankWaves = 4;
constexpr int kAtom = 16;
constexpr int kPartsC = kCK / 32; /* 32-channel MFMA steps per chunk and tap */
constexpr int64_t kOob = ((int64_t)1 << 31) - 1;

const char* bf16_flag[1] = {"bf16"};

int in_h(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return kTH + s->problem.KH - 1;
}

int in_w(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return kTW + s->problem.KW - 1;
}

/* Staged rows, rounded up to whole DMA groups. */
int in_rows(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return cdiv(in_h(s), kRankWaves) * kRankWaves;
}

long in_elems(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return (long)in_rows(s) * in_w(s) * kCK;
}

long w_elems(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return (long)s->block_k * kCK;
}

/* One DMA piece of an input buffer: (row, first column, dwords per lane). */
struct Piece
{
    int row, col, dw;
};

/* _input_pieces(in_rows, in_w): 8-column 16-byte pieces, then the row's last
 * in_w % 8 columns as 2-column 4-byte pieces, grouped 4 rows per group. */
std::vector<std::vector<Piece>> input_pieces(int rows, int width)
{
    std::vector<std::pair<int, int>> cols;
    for(int c0 = 0; c0 < width - 7; c0 += 8)
    {
        cols.push_back({c0, 4});
    }
    for(int c0 = width - width % 8; c0 < width; c0 += 2)
    {
        cols.push_back({c0, 1});
    }
    std::vector<std::vector<Piece>> groups;
    for(const auto& cd : cols)
    {
        for(int r0 = 0; r0 < rows; r0 += kRankWaves)
        {
            std::vector<Piece> g;
            for(int i = 0; i < kRankWaves; ++i)
            {
                g.push_back({r0 + i, cd.first, cd.second});
            }
            groups.push_back(g);
        }
    }
    return groups;
}

/* a * b, saturated at INT64_MAX (Python's ints do not wrap). */
long long mul_sat(long long a, long long b)
{
    int64_t out;
    return rocke_intexpr_checked_mul(a, b, &out) ? out : INT64_MAX;
}

/* _dgrad_pp_mfma_limits_reason(spec): the buffer descriptors span one input
 * image, one output image and the whole filter, and a masked lane reads at
 * byte offset kOob (INT32_MAX), which is out of range only while each of them
 * is at most INT32_MAX bytes; the flat block id is an i32 too. Sizes
 * saturate in 64 bits; the extents are validated >= 1 before this runs, and
 * once the tensors fit, the block count cannot overflow. */
bool limits_reason(const rocke_dgrad_pp_mfma_spec_t* s, char* msg, size_t cap)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    const long long ho = rocke_direct_conv_problem_Ho(p);
    const long long wo = rocke_direct_conv_problem_Wo(p);
    const long long tc = rocke_direct_conv_problem_total_c(p);
    const long long tk = rocke_direct_conv_problem_total_k(p);
    const struct
    {
        const char* what;
        long long bytes;
    } tensors[] = {
        {"input image", mul_sat(mul_sat(mul_sat(2, p->H), p->W), tc)},
        {"output image", mul_sat(mul_sat(mul_sat(2, ho), wo), tk)},
        {"filter", mul_sat(mul_sat(mul_sat(mul_sat(2, tk), p->KH), p->KW), p->cpg)},
    };
    for(const auto& t : tensors)
    {
        if(t.bytes > kOob)
        {
            ROCKE_ERR_SNPRINTF(
                msg, cap, "%s exceeds %lld B, the i32 buffer range", t.what, (long long)kOob);
            return true;
        }
    }
    const long long blocks = (long long)p->N * rocke_ceil_div(ho, kTH) * rocke_ceil_div(wo, kTW)
                             * rocke_ceil_div(tk, s->block_k);
    if(blocks > kOob)
    {
        ROCKE_ERR_SNPRINTF(msg, cap, "%lld workgroups exceed the i32 block id", blocks);
        return true;
    }
    return false;
}

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
    return kWaves * s->wave_size;
}

extern "C" long rocke_dgrad_pp_mfma_lds_bytes(const rocke_dgrad_pp_mfma_spec_t* s)
{
    return 2 * (2 * in_elems(s) + 2 * w_elems(s));
}

extern "C" void rocke_dgrad_pp_mfma_grid(const rocke_dgrad_pp_mfma_spec_t* s, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    long long n = (long long)p->N * cdiv(problem_ho(p), kTH) * cdiv(problem_wo(p), kTW)
                  * cdiv((int)rocke_direct_conv_problem_total_k(p), s->block_k);
    grid[0] = (int)n;
    grid[1] = 1;
    grid[2] = 1;
}

extern "C" rocke_status_t
    rocke_dgrad_pp_mfma_kernel_name(const rocke_dgrad_pp_mfma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[48], p_buf[24], bk_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%d", p->KH, p->KW);
    snprintf(p_buf, sizeof p_buf, "p%d", p->PAD);
    snprintf(bk_buf, sizeof bk_buf, "bk%d", s->block_k);
    const char* parts[3] = {f_buf, p_buf, bk_buf};
    int on[1] = {problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 3, bf16_flag, on, 1, out, out_cap, NULL);
}

extern "C" bool rocke_dgrad_pp_mfma_is_valid_spec(const rocke_dgrad_pp_mfma_spec_t* s,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    char msg[ROCKE_ERR_MSG_CAP];
    if(s == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "unknown arch %s", arch);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    long lds = rocke_dgrad_pp_mfma_lds_bytes(s);
    const char* ab = problem_is_bf16(p) ? "bf16" : "f16";
    const char* why = NULL;
    if(target->wave_size != 64 || s->wave_size != 64)
    {
        why = "dgrad_pp_mfma is a wave64 MFMA kernel";
    }
    else if(!target->memory.has_ds_read_tr)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "dgrad_pp_mfma needs ds_read_b64_tr_b16 (absent on %s)", arch);
        why = msg;
    }
    else if(rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "dgrad_pp_mfma needs the 16-byte LDS DMA (absent on %s)", arch);
        why = msg;
    }
    else if(!problem_dtype_ok(p))
    {
        why = "fp16/bf16 only";
    }
    else if(!rocke_mma_catalog_has_shape(
                rocke_archtarget_mma(target), "mma", ab, ab, "fp32", 16, 16, 32))
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "missing mfma_f32_16x16x32_%s on %s", ab, arch);
        why = msg;
    }
    else if(rocke_direct_problem_dims_reason(p, msg, sizeof msg))
    {
        why = msg;
    }
    else if(p->groups != 1)
    {
        why = "groups must be 1";
    }
    else if(p->stride != 1)
    {
        why = "stride must be 1";
    }
    else if(p->dil_h != 1 || p->dil_w != 1)
    {
        why = "dilation must be 1";
    }
    else if(p->KH != p->KW || p->KH < 3 || p->KH % 2 == 0)
    {
        why = "square odd filters >= 3 only";
    }
    else if(p->PAD != (p->KH - 1) / 2)
    {
        ROCKE_ERR_SNPRINTF(msg,
                           sizeof msg,
                           "'same' padding only: PAD must be %d (got %d)",
                           (p->KH - 1) / 2,
                           p->PAD);
        why = msg;
    }
    else if(rocke_direct_conv_one_padding_reason(p, "", msg, sizeof msg))
    {
        why = msg;
    }
    else if(p->cpg % kGran)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "C %d must be a multiple of %d (16-byte granules)", p->cpg, kGran);
        why = msg;
    }
    else if(p->kpg % 4)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "K %d must be a multiple of 4 (8-byte output writes)", p->kpg);
        why = msg;
    }
    else if(s->block_k != 128 && s->block_k != 256)
    {
        why = "block_k must be 128 or 256 (256-byte weight rows for the swizzle)";
    }
    else if(!rocke_arch_fits_lds(target, lds))
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "LDS %ld B > %d", lds, (int)target->lds_capacity_bytes);
        why = msg;
    }
    else if(limits_reason(s, msg, sizeof msg))
    {
        why = msg;
    }
    if(why != NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, why);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

namespace
{

/* One rank-0 weight DMA piece: (row, offset, output ok, LDS byte offset). */
struct WPiece
{
    V row, off, k_ok, lds_off;
};

/* One input lane geometry: (offset without the row and chunk, channel, ok). */
struct InGeo
{
    V off, ch, ok;
};

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

rocke_kernel_def_t*
    build_pp(rocke_ir_builder_t* b, const rocke_dgrad_pp_mfma_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_dgrad_pp_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid DgradPpMfmaSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const int KH = p.KH, KW = p.KW, PAD = p.PAD;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const int BK = spec->block_k;
    const int WK = BK / 4; /* spec.wave_k */
    const int KF = WK / kAtom; /* spec.k_frags */
    const int IN_H = in_h(spec), IN_W = in_w(spec);
    const int ROW_E = IN_W * kCK;
    const int TAPS = KH * KW;
    const int STEPS = TAPS * kPartsC;
    const int W_PIECES = BK * kCK * 2 / 1024;
    const int W_PER_WAVE = W_PIECES / kRankWaves;
    const int GPR = BK / kGran;
    const int RPI = 1024 / (BK * 2);
    const std::vector<std::vector<Piece>> in_groups = input_pieces(in_rows(spec), IN_W);
    const int NG = (int)in_groups.size();
    std::vector<std::vector<int>> in_sched(STEPS);
    const int span = std::max(1, STEPS - 4);
    for(int i = 0; i < NG; ++i)
    {
        in_sched[i * span / NG].push_back(i);
    }

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", kWaves * spec->wave_size);
    rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", 2);
    rocke_dconv_params_t P;
    rocke_dconv_emit_params(b, &P, "fwd", io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    V c0 = e.ci(0);
    V c1 = e.ci(1);
    V c2 = e.ci(2);
    V oob = e.ci(kOob);

    V p_Hi = P.p_Hi, p_Wi = P.p_Wi;
    V p_Ho = P.p_Ho, p_Wo = P.p_Wo;
    V C = P.p_total_c, K = P.p_total_k;
    V sa_n = P.p_A_stride_n, sa_h = P.p_A_stride_hi, sa_w = P.p_A_stride_wi;
    V sd_n = P.p_D_stride_n, sd_h = P.p_D_stride_ho, sd_w = P.p_D_stride_wo;

    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    V wave_j = e.mod(wave, c2);
    V wave_k = e.div(wave, c2);
    V wave_g = e.modi(wave, kRankWaves);
    V rank = e.divi(wave, kRankWaves);

    /* ---- grid decode: block -> (k tile fastest, w tile, h tile, image) ---- */
    V n_kt = e.cdivi(K, BK);
    V n_wt = e.cdivi(p_Wo, kTW);
    V n_ht = e.cdivi(p_Ho, kTH);
    V blk = rocke_b_block_id_x(b);
    V k_tile = e.mod(blk, n_kt);
    V rest = e.div(blk, n_kt);
    V w_tile = e.mod(rest, n_wt);
    rest = e.div(rest, n_wt);
    V h_tile = e.mod(rest, n_ht);
    V n_img = e.div(rest, n_ht);
    V oh0 = e.muli(h_tile, kTH);
    V ow0 = e.muli(w_tile, kTW);
    V k0 = e.muli(k_tile, BK);

    /* Per-image buffer bases keep every offset inside 32 bits. */
    auto img_base = [&](V ptr, V stride_n) {
        V n64 = e.zext64(n_img);
        V s64 = e.zext64(e.mul(stride_n, c2));
        return rocke_b_global_ptr_offset(b, ptr, e.mul(n64, s64));
    };
    V a_img = img_base(P.A, sa_n);
    V d_img = img_base(P.D, sd_n);
    V a_rsrc = rocke_b_buffer_rsrc(b, a_img, e.mul(sa_n, c2));
    V w_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes);
    V d_rsrc = rocke_b_buffer_rsrc(b, d_img, e.mul(sd_n, c2));

    /* ---- LDS ---- */
    V in_bufs[2], w_bufs[2], in_addr[2], w_addr[2];
    for(int i = 0; i < 2; ++i)
    {
        std::string h = "in" + std::to_string(i);
        in_bufs[i] = e.smem_alloc_1d(io_type, (int)in_elems(spec), h.c_str());
    }
    for(int i = 0; i < 2; ++i)
    {
        std::string h = "w" + std::to_string(i);
        w_bufs[i] = e.smem_alloc_1d(io_type, (int)w_elems(spec), h.c_str());
    }
    for(int i = 0; i < 2; ++i)
    {
        in_addr[i] = rocke_b_smem_addr_of(b, in_bufs[i]);
    }
    for(int i = 0; i < 2; ++i)
    {
        w_addr[i] = rocke_b_smem_addr_of(b, w_bufs[i]);
    }
    auto lds_at
        = [&](V base, V byte_off) { return rocke_b_smem_ptr_add(b, base, e.zext64(byte_off)); };

    /* ---- rank 0: weight DMA geometry ---- */
    V row_len = e.muli(K, TAPS);
    std::vector<WPiece> w_pieces;
    for(int i = 0; i < W_PER_WAVE; ++i)
    {
        V q = e.addi(e.muli(wave_g, W_PER_WAVE), i);
        V r = e.muli(q, RPI);
        V lr = e.divi(lane, GPR);
        V row = e.add(r, lr);
        V lg = e.modi(lane, GPR);
        V gran = e.bxor(lg, tr_swz(e, row));
        V cc = e.add(k0, e.muli(gran, kGran));
        WPiece wp;
        wp.row = row;
        wp.off = e.add(e.mul(row, row_len), cc);
        wp.k_ok = e.lt(cc, K);
        wp.lds_off = e.muli(q, 1024);
        w_pieces.push_back(wp);
    }

    auto issue_weights = [&](int buf, int tap, V c_base) {
        int r = tap / KW, s_ = tap % KW;
        int tap_f = (KH - 1 - r) * KW + (KW - 1 - s_);
        V m1 = e.mul(c_base, row_len);
        V kt = e.ci(tap_f);
        V m2 = e.mul(kt, K);
        V base = e.add(m1, m2);
        for(const WPiece& wp : w_pieces)
        {
            V a = e.add(c_base, wp.row);
            V ok = e.land(wp.k_ok, e.lt(a, C));
            V voff = e.mul(e.add(wp.off, base), c2);
            voff = e.select(ok, voff, oob);
            V lds = lds_at(w_addr[buf], wp.lds_off);
            rocke_b_async_buffer_load_lds(b, w_rsrc, lds, voff, c0, 4, 0);
        }
    };

    /* ---- rank 1: input DMA geometry ---- */
    V l_row = e.divi(lane, 8);
    V l_gran = e.bxor(e.modi(lane, 8), l_row);
    V ih0 = e.subi(oh0, PAD);
    V iw0 = e.subi(ow0, PAD);
    std::set<std::pair<int, int>> kinds;
    for(const auto& g : in_groups)
    {
        kinds.insert({g[0].col, g[0].dw});
    }
    std::map<std::pair<int, int>, InGeo> lane_geo;
    for(const auto& kd : kinds)
    {
        int c_first = kd.first, dw = kd.second;
        V col, ch;
        if(dw == 4)
        {
            V kc = e.ci(c_first);
            col = e.add(kc, l_row);
            ch = e.muli(l_gran, kGran);
        }
        else
        {
            V kc = e.ci(c_first);
            V d = e.divi(lane, 32);
            col = e.add(kc, d);
            V el = e.mul(e.modi(lane, 32), c2);
            V g1 = e.divi(el, kGran);
            V g2 = e.modi(col, 8);
            V gran = e.bxor(g1, g2);
            V m = e.muli(gran, kGran);
            V md = e.modi(el, kGran);
            ch = e.add(m, md);
        }
        V iw = e.add(iw0, col);
        V ge = e.ge(iw, c0);
        V ok = e.land(ge, e.lt(iw, p_Wi));
        V off = e.add(e.mul(iw, sa_w), ch);
        lane_geo[kd] = {off, ch, ok};
    }

    auto issue_input_group = [&](int buf, int gi, V c_base) {
        const Piece& pc = in_groups[gi][0];
        const InGeo& g = lane_geo[{pc.col, pc.dw}];
        V kr = e.ci(pc.row);
        V row = e.add(kr, wave_g);
        V ih = e.add(ih0, row);
        V r_lt = e.lti(row, IN_H);
        V ih_ge = e.ge(ih, c0);
        V ih_ok = e.land(ih_ge, e.lt(ih, p_Hi));
        V row_ok = e.land(r_lt, ih_ok);
        V c_ok = e.lt(e.add(c_base, g.ch), C);
        V g_off = e.add(g.off, e.mul(ih, sa_h));
        g_off = e.add(g_off, c_base);
        V all = e.land(e.land(g.ok, row_ok), c_ok);
        V voff = e.mul(g_off, c2);
        voff = e.select(all, voff, oob);
        V lds_off = e.muli(row, ROW_E * 2);
        lds_off = e.addi(lds_off, (int64_t)pc.col * kCK * 2);
        V lds = lds_at(in_addr[buf], lds_off);
        rocke_b_async_buffer_load_lds(b, a_rsrc, lds, voff, c0, pc.dw, 0);
    };

    /* ---- fragment reads ---- */
    V tb = e.divi(lane, 16);
    V ti = e.modi(e.divi(lane, 4), 4);
    V tj = e.modi(lane, 4);
    std::map<std::tuple<int, int, int>, V> a_idx;
    for(int kf = 0; kf < KF; ++kf)
    {
        V m = e.muli(wave_k, WK);
        V kk = e.ci(kf * kAtom);
        V m2 = e.muli(tj, 4);
        V col = e.add(m, e.add(kk, m2));
        for(int part = 0; part < kPartsC; ++part)
        {
            for(int rd = 0; rd < 2; ++rd)
            {
                V kr = e.ci(part * 32 + 4 * rd);
                V mt = e.muli(tb, 8);
                V row = e.add(e.add(kr, mt), ti);
                V sl = e.divi(col, kGran);
                V slot = e.bxor(sl, tr_swz(e, row));
                V mr = e.muli(row, BK);
                V ms = e.muli(slot, kGran);
                V mc = e.modi(col, kGran);
                a_idx[{kf, part, rd}] = e.add(mr, e.add(ms, mc));
            }
        }
    }
    V l16 = e.modi(lane, 16);
    std::map<std::pair<int, int>, V> b_lane;
    for(int s = 0; s < KW; ++s)
    {
        V col = e.addi(l16, s);
        for(int part = 0; part < kPartsC; ++part)
        {
            V kg = e.ci(part * 4);
            V g = e.add(kg, tb);
            V slot = e.bxor(g, e.modi(col, 8));
            V mc = e.muli(col, kCK);
            b_lane[{s, part}] = e.add(mc, e.muli(slot, kGran));
        }
    }
    V row0 = e.muli(wave_j, 8 * ROW_E);

    auto read_a = [&](int buf, int kf, int part) {
        V lo = e.ds_read_tr(w_bufs[buf], c0, a_idx[{kf, part, 0}], io_type);
        V hi = e.ds_read_tr(w_bufs[buf], c0, a_idx[{kf, part, 1}], io_type);
        return rocke_b_vec_concat(b, lo, hi);
    };
    auto read_b = [&](int buf, int jb, int r, int s, int part) {
        V a = e.addi(row0, (int64_t)(jb + r) * ROW_E);
        V idx = e.add(b_lane[{s, part}], a);
        return e.smem_load(in_bufs[buf], c0, idx, io_type, kGran);
    };

    /* ---- one chunk ---- */
    const int n_acc = KF * 8;
    auto chunk = [&](std::vector<V> accs, int par, V c_base, int my_rank) {
        V c_next = e.addi(c_base, kCK);
        for(int step = 0; step < STEPS; ++step)
        {
            int tap = step / kPartsC, part = step % kPartsC;
            int r = tap / KW, s = tap % KW;
            int wbuf = (par * TAPS + tap) % 2;
            std::vector<V> a_fr, b_fr;
            for(int kf = 0; kf < KF; ++kf)
            {
                a_fr.push_back(read_a(wbuf, kf, part));
            }
            for(int jb = 0; jb < 8; ++jb)
            {
                b_fr.push_back(read_b(par, jb, r, s, part));
            }
            if(my_rank == 0 && part == 0)
            {
                /* Next tap's weights (tap 0 of the next chunk after the last). */
                if(tap + 1 < TAPS)
                {
                    issue_weights(1 - wbuf, tap + 1, c_base);
                }
                else
                {
                    issue_weights(1 - wbuf, 0, c_next);
                }
            }
            if(my_rank == 1)
            {
                for(int gi : in_sched[step])
                {
                    issue_input_group(1 - par, gi, c_next);
                }
                if(step == STEPS - 1)
                {
                    /* The next chunk's input, before rank 0 reads it. */
                    rocke_b_s_waitcnt(b, 0, -1, -1);
                }
            }
            rocke_b_sched_barrier(b, 0);
            rocke_b_s_barrier_bare(b);
            rocke_b_sched_barrier(b, 0);
            rocke_b_s_setprio(b, 1);
            for(int kf = 0; kf < KF; ++kf)
            {
                for(int o = 0; o < 8; ++o)
                {
                    int jb = (step % 2 == 0) ? o : 7 - o;
                    int i = kf * 8 + jb;
                    accs[i] = e.mfma16x16x32(bf16, a_fr[kf], b_fr[jb], accs[i]);
                }
            }
            rocke_b_s_setprio(b, 0);
            rocke_b_sched_barrier(b, 0);
            if(my_rank == 0 && part == kPartsC - 1)
            {
                /* The next tap's weights, before either rank reads them. */
                rocke_b_s_waitcnt(b, 0, -1, -1);
            }
            rocke_b_s_barrier_bare(b);
            rocke_b_sched_barrier(b, 0);
        }
        return accs;
    };

    V n_chunks = e.cdivi(C, kCK);
    V n_pairs = e.div(n_chunks, c2);
    V zero = rocke_b_zero_vec_f32(b, 4);
    const rocke_type_t* acc_t = rocke_vector_type(b, rocke_f32(), 4);
    std::vector<const rocke_type_t*> acc_types(n_acc, acc_t);

    /* scf_if_else with n_acc vector results; then_fn / else_fn return the
     * values each region yields. */
    auto if_else = [&](V cond, auto&& then_fn, auto&& else_fn) {
        rocke_if_else_t ife = rocke_b_scf_if_else_results(b, cond, acc_types.data(), n_acc);
        if(ife.op == NULL || ife.then_region == NULL || ife.else_region == NULL)
        {
            return std::vector<V>();
        }
        rocke_b_region_enter(b, ife.then_region);
        std::vector<V> t = then_fn();
        rocke_b_scf_yield(b, t.data(), (int)t.size());
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ife.else_region);
        std::vector<V> f = else_fn();
        rocke_b_scf_yield(b, f.data(), (int)f.size());
        rocke_b_region_leave(b);
        return std::vector<V>(ife.op->results, ife.op->results + ife.op->num_results);
    };

    auto rank_body = [&](int my_rank) {
        /* Prologue: tap 0 weights (rank 0) and chunk 0's input (rank 1). */
        if(my_rank == 0)
        {
            issue_weights(0, 0, c0);
        }
        else
        {
            for(int gi = 0; gi < NG; ++gi)
            {
                issue_input_group(0, gi, c0);
            }
        }
        rocke_b_s_waitcnt(b, 0, -1, -1);
        rocke_b_s_barrier_bare(b);
        if(my_rank == 1)
        {
            rocke_b_s_barrier_bare(b); /* rank 1 runs one barrier behind */
        }
        rocke_b_sched_barrier(b, 0);
        std::string prefix = "acc" + std::to_string(my_rank) + "_";
        std::string iv = "pair" + std::to_string(my_rank);
        std::vector<V> accs = e.for_iter(c0,
                                         n_pairs,
                                         c1,
                                         iter_names(prefix.c_str(), n_acc),
                                         std::vector<V>(n_acc, zero),
                                         iv.c_str(),
                                         [&](V pi, std::vector<V> carried) {
                                             V c_even = e.muli(pi, 2 * kCK);
                                             std::vector<V> a = chunk(carried, 0, c_even, my_rank);
                                             return chunk(a, 1, e.addi(c_even, kCK), my_rank);
                                         });
        if((int)accs.size() != n_acc)
        {
            return std::vector<V>();
        }
        V odd = e.eq(e.mod(n_chunks, c2), c1);
        accs = if_else(
            odd,
            [&]() { return chunk(accs, 0, e.muli(n_pairs, 2 * kCK), my_rank); },
            [&]() { return accs; });
        if(my_rank == 0)
        {
            rocke_b_s_barrier_bare(b); /* pair rank 1's last barrier */
        }
        return accs;
    };

    std::vector<V> accs
        = if_else(e.eq(rank, c0), [&]() { return rank_body(0); }, [&]() { return rank_body(1); });
    if((int)accs.size() != n_acc)
    {
        return NULL;
    }

    /* ---- epilogue: lane holds channels tb*4 .. +3 of pixel (h, lane % 16) ---- */
    V k_lane = e.add(k0, e.muli(wave_k, WK));
    k_lane = e.add(k_lane, e.muli(tb, 4));
    V ow = e.add(ow0, l16);
    V w_ok = e.lt(ow, p_Wo);
    for(int jb = 0; jb < 8; ++jb)
    {
        V hh = e.addi(e.muli(wave_j, 8), jb);
        V oh = e.add(oh0, hh);
        V hw_ok = e.land(w_ok, e.lt(oh, p_Ho));
        V m1 = e.mul(oh, sd_h);
        V pix = e.add(m1, e.mul(ow, sd_w));
        for(int kf = 0; kf < KF; ++kf)
        {
            V k = e.addi(k_lane, kf * kAtom);
            V ok = e.land(hw_ok, e.lt(k, K));
            V voff = e.mul(e.add(pix, k), c2);
            voff = e.select(ok, voff, oob);
            V val = rocke_dconv_trunc_f32(b, bf16, accs[kf * 8 + jb]);
            rocke_dconv_buf_store_vN(b, bf16, d_rsrc, voff, c0, val, 2);
        }
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_dgrad_pp_mfma(rocke_ir_builder_t* b,
                                                         const rocke_dgrad_pp_mfma_spec_t* spec,
                                                         const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_pp(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_dgrad_pp_mfma_new(rocke_ir_builder_t* b,
                                                             const rocke_dgrad_pp_mfma_spec_t* spec,
                                                             const char* arch)
{
    return build_new(
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
    return lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_dgrad_pp_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_dgrad_pp_mfma_new(b, spec, arch); });
}
