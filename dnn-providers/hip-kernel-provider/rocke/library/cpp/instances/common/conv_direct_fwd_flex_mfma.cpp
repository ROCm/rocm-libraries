// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_flex_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_fwd_flex_mfma.py: the gfx950
 * flattened-pixel implicit-GEMM MFMA direct forward (build_fwd_flex_mfma).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_fwd_flex_mfma.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
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
constexpr int kBX = 256; /* pixels per block tile */
constexpr int kXT = 4; /* 16-pixel MFMA tiles per wave */
constexpr int kPersist = 256; /* blocks of the persistent grid */
constexpr int kRot = 97; /* per-round rotation of the item walk */
constexpr int kWaves = 8;
constexpr int64_t kOob = ((int64_t)1 << 31) - 1;

const char* bf16_flag[1] = {"bf16"};

/* a * b, saturated at INT64_MAX (Python's ints do not wrap). */
long long mul_sat(long long a, long long b)
{
    int64_t out;
    return rocke_intexpr_checked_mul(a, b, &out) ? out : INT64_MAX;
}

/* The tail of fwd_flex_mfma_reason: every buffer offset is an i32 over the
 * whole tensor (or workspace slab). Sizes saturate in 64 bits; the extents
 * are validated >= 1 before this runs. */
bool limits_reason(const rocke_fwd_flex_mfma_spec_t* s, char* msg, size_t cap)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    const long long tc = rocke_direct_conv_problem_total_c(p);
    const long long tk = rocke_direct_conv_problem_total_k(p);
    const long long nx
        = mul_sat(mul_sat(p->N, rocke_direct_conv_problem_Ho(p)), rocke_direct_conv_problem_Wo(p));
    const struct
    {
        const char* what;
        long long bytes;
    } tensors[] = {
        {"input", mul_sat(mul_sat(mul_sat(mul_sat(2, p->N), p->H), p->W), tc)},
        {"output", mul_sat(mul_sat(2, nx), tk)},
        {"weights", mul_sat(mul_sat(mul_sat(mul_sat(2, tk), p->KH), p->KW), tc)},
        {"workspace slab", s->split_c > 1 ? mul_sat(mul_sat(4, nx), tk) : 0},
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
    /* the whole workspace's size is an i32 kernarg of the conv and the fold */
    if(s->split_c > 1 && mul_sat(mul_sat(mul_sat(4, s->split_c), nx), tk) > kOob)
    {
        ROCKE_ERR_SNPRINTF(
            msg, cap, "workspace exceeds %lld B, the i32 byte-count kernarg", (long long)kOob);
        return true;
    }
    return false;
}

} // namespace

extern "C" rocke_fwd_flex_mfma_spec_t rocke_fwd_flex_mfma_spec_default(void)
{
    rocke_fwd_flex_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.bk = 256;
    s.split_c = 1;
    s.name = "fwd_flex_mfma";
    s.wave_size = 64;
    return s;
}

extern "C" int rocke_fwd_flex_mfma_kt(const rocke_fwd_flex_mfma_spec_t* s)
{
    return s->bk == 256 ? 8 : 4;
}

extern "C" int rocke_fwd_flex_mfma_threads_per_block(const rocke_fwd_flex_mfma_spec_t* s)
{
    return kWaves * s->wave_size;
}

extern "C" long long rocke_fwd_flex_mfma_items(const rocke_fwd_flex_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(s->bk <= 0)
    {
        return 0;
    }
    const long long nx
        = (long long)p->N * rocke_direct_conv_problem_Ho(p) * rocke_direct_conv_problem_Wo(p);
    const long long tiles
        = rocke_ceil_div(nx, kBX) * rocke_ceil_div(rocke_direct_conv_problem_total_k(p), s->bk);
    return tiles * s->split_c;
}

extern "C" void rocke_fwd_flex_mfma_grid(const rocke_fwd_flex_mfma_spec_t* s, int grid[3])
{
    const long long items = rocke_fwd_flex_mfma_items(s);
    grid[0] = (int)(items < kPersist ? items : kPersist);
    grid[1] = 1;
    grid[2] = 1;
}

extern "C" long long rocke_fwd_flex_mfma_workspace_bytes(const rocke_fwd_flex_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(s->split_c == 1)
    {
        return 0;
    }
    return 4LL * s->split_c * p->N * rocke_direct_conv_problem_Ho(p)
           * rocke_direct_conv_problem_Wo(p) * rocke_direct_conv_problem_total_k(p);
}

extern "C" rocke_status_t
    rocke_fwd_flex_mfma_kernel_name(const rocke_fwd_flex_mfma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[48], p_buf[48], s_buf[24], d_buf[48], k_buf[24], sc_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%d", p->KH, p->KW);
    snprintf(p_buf, sizeof p_buf, "p%dx%d", p->PAD, rocke_direct_conv_problem_pad_w(p));
    snprintf(s_buf, sizeof s_buf, "s%d", p->stride);
    snprintf(d_buf, sizeof d_buf, "d%dx%d", p->dil_h, p->dil_w);
    snprintf(k_buf, sizeof k_buf, "k%d", s->bk);
    snprintf(sc_buf, sizeof sc_buf, "sc%d", s->split_c);
    const char* parts[6] = {f_buf, p_buf, s_buf, d_buf, k_buf, sc_buf};
    int on[1] = {problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 6, bf16_flag, on, 1, out, out_cap, NULL);
}

extern "C" bool rocke_fwd_flex_mfma_is_valid_spec(const rocke_fwd_flex_mfma_spec_t* s,
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
    const long long stage = ((long long)kBX + s->bk) * kCK * 2;
    const long long tc = rocke_direct_conv_problem_total_c(p);
    const long long tk = rocke_direct_conv_problem_total_k(p);
    const char* ab = problem_is_bf16(p) ? "bf16" : "f16";
    const char* why = NULL;
    if(target->wave_size != 64 || s->wave_size != 64)
    {
        why = "fwd_flex_mfma is a wave64 MFMA kernel";
    }
    else if(rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "fwd_flex_mfma needs the 16-byte LDS DMA (absent on %s)", arch);
        why = msg;
    }
    else if(!rocke_arch_fits_lds(target, (long)(2 * stage)))
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "LDS %lld B > %d", 2 * stage, (int)target->lds_capacity_bytes);
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
    else if(p->stride < 1 || p->dil_h < 1 || p->dil_w < 1)
    {
        ROCKE_ERR_SNPRINTF(msg,
                           sizeof msg,
                           "stride and dilation must be >= 1 (got s%d d%dx%d)",
                           p->stride,
                           p->dil_h,
                           p->dil_w);
        why = msg;
    }
    else if(rocke_direct_conv_pad_w_reason(p, "", msg, sizeof msg))
    {
        why = msg;
    }
    else if(s->bk != 128 && s->bk != 256)
    {
        why = "bk must be 128 or 256";
    }
    else if(tc % kCK)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "C %lld must be a multiple of %d", tc, kCK);
        why = msg;
    }
    else if(tk % 8)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "K %lld must be a multiple of 8 (16-byte output writes)", tk);
        why = msg;
    }
    else if(s->split_c < 1 || s->split_c > tc / kCK)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "split_c %d must be in [1, C / %d]", s->split_c, kCK);
        why = msg;
    }
    else if((long long)p->KH * p->KW > 32)
    {
        why = "more than 32 taps (one valid-tap bit per tap)";
    }
    else if(rocke_direct_conv_problem_Ho(p) < 1 || rocke_direct_conv_problem_Wo(p) < 1)
    {
        why = "empty output";
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

/* The fields of item_info(t) the builder reads. */
struct Info
{
    V tx, tk, r0, r1, s0, s1, ch0, ch1;
};

/* The DMA cursor, in Python's `names = SC + LN` order: the scalar part
 * (i, t, ch, ch1, r, s, r0, r1, s0, s1), then the lane part (xb0.., vm0..,
 * wo). */
enum
{
    cI,
    cT,
    cCh,
    cCh1,
    cR,
    cS,
    cR0,
    cR1,
    cS0,
    cS1,
    cLanes
};

rocke_kernel_def_t*
    build_flex(rocke_ir_builder_t* b, const rocke_fwd_flex_mfma_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_fwd_flex_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid FwdFlexMfmaSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const int BK = spec->bk, KT = rocke_fwd_flex_mfma_kt(spec), SP = spec->split_c;
    const int WKN = BK / (KT * 16); /* waves of a group along K */
    const int NXI = kBX / 64, NWI = BK / 64; /* DMA instructions per wave per step */
    const int R = p.KH, S = p.KW;
    const int SH = p.stride, PH = p.PAD, PW = rocke_direct_conv_problem_pad_w(&p);
    const int DH = p.dil_h, DW = p.dil_w;
    const int IN_E = kBX * kCK, W_E = BK * kCK; /* elements per stage */
    const int NAMES = cLanes + 2 * NXI + 1;
    const int nacc = kXT * KT;
    auto xb = [&](int i) { return cLanes + i; };
    auto vm = [&](int i) { return cLanes + NXI + i; };
    const int cWo = cLanes + 2 * NXI;
    std::vector<std::string> names = {"i", "t", "ch", "ch1", "r", "s", "r0", "r1", "s0", "s1"};
    for(int i = 0; i < NXI; ++i)
    {
        names.push_back("xb" + std::to_string(i));
    }
    for(int i = 0; i < NXI; ++i)
    {
        names.push_back("vm" + std::to_string(i));
    }
    names.push_back("wo");

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
    V p_N = P.p_N, p_Hi = P.p_Hi, p_Wi = P.p_Wi;
    V p_Ho = P.p_Ho, p_Wo = P.p_Wo;
    V C = P.p_total_c, K = P.p_total_k;

    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    V g = e.divi(wave, 4);
    V wq = e.modi(wave, 4);
    V wk = e.modi(wq, WKN);
    V wx = e.divi(wq, WKN);

    /* Lane-derived addressing is recomputed where it is used instead of being
     * hoisted into long-lived VGPRs. */
    const rocke_type_t* i32_t = rocke_i32();
    auto opaque_lane = [&]() -> V {
        rocke_inline_asm_opts_t opts = {};
        opts.sideeffect = true;
        opts.sideeffect_set = true;
        rocke_op_t* op = rocke_b_inline_asm(b, "", "=v,0", &lane, 1, &i32_t, 1, &opts);
        return op ? op->results[0] : NULL;
    };

    V hw = e.mul(p_Ho, p_Wo);
    V NX = e.mul(p_N, hw);
    V nxt = e.cdivi(NX, kBX);
    V nkt = e.cdivi(K, BK);
    V TI = e.muli(e.mul(nxt, nkt), SP);
    V G = e.smini(TI, kPersist);
    V nch = e.divi(C, kCK);
    V RSC = e.mul(e.ci((int64_t)R * S), C);
    V blk = rocke_b_block_id_x(b);
    V full = e.div(TI, G);
    V part = e.mod(TI, G);
    V last_ok = e.lt(e.mod(e.add(blk, e.muli(full, kRot)), G), part);
    V rounds = e.add(full, e.select(last_ok, c1, c0));

    auto tile_of = [&](V i) {
        V m = e.mul(i, G);
        return e.add(m, e.mod(e.add(blk, e.muli(i, kRot)), G));
    };

    /* ---- item -> (tile, piece, valid tap ranges) ---- */
    auto item_info = [&](V t) {
        Info in;
        V tl = e.divi(t, SP);
        V pc = e.modi(t, SP);
        in.tx = e.div(tl, nkt);
        in.tk = e.mod(tl, nkt);
        V x0 = e.muli(in.tx, kBX);
        V x1 = e.sub(e.smin(e.addi(x0, kBX), NX), c1);
        V n0 = e.div(x0, hw);
        V n1 = e.div(x1, hw);
        V same = e.eq(n0, n1);
        V q0 = e.sub(x0, e.mul(n0, hw));
        V q1 = e.sub(x1, e.mul(n0, hw));
        V ho0 = e.select(same, e.div(q0, p_Wo), c0);
        V ho1_in = e.div(q1, p_Wo);
        V ho1 = e.select(same, ho1_in, e.sub(p_Ho, c1));
        V row = e.land(same, e.eq(ho0, ho1));
        V wo0 = e.select(row, e.sub(q0, e.mul(ho0, p_Wo)), c0);
        V wo1_in = e.sub(q1, e.mul(ho1, p_Wo));
        V wo1 = e.select(row, wo1_in, e.sub(p_Wo, c1));
        /* Valid taps are an interval: a prefix of the rows reads only the top
         * padding (lo < 0 at the tile's last row) and a suffix only the bottom. */
        V lo_h = e.subi(e.muli(ho1, SH), PH);
        V hi_h = e.subi(e.muli(ho0, SH), PH);
        V lo_w = e.subi(e.muli(wo1, SH), PW);
        V hi_w = e.subi(e.muli(wo0, SH), PW);
        V r0 = c0;
        V r1 = e.ci(R);
        V s0 = c0;
        V s1 = e.ci(S);
        for(int r = 0; r < R; ++r)
        {
            r0 = e.add(r0, e.select(e.lt(e.addi(lo_h, (int64_t)r * DH), c0), c1, c0));
            r1 = e.sub(r1, e.select(e.ge(e.addi(hi_h, (int64_t)r * DH), p_Hi), c1, c0));
        }
        for(int s = 0; s < S; ++s)
        {
            s0 = e.add(s0, e.select(e.lt(e.addi(lo_w, (int64_t)s * DW), c0), c1, c0));
            s1 = e.sub(s1, e.select(e.ge(e.addi(hi_w, (int64_t)s * DW), p_Wi), c1, c0));
        }
        /* A tile reading only padding still runs one (zero) step. */
        V er = e.ge(r0, r1);
        V empty = e.lor(er, e.ge(s0, s1));
        in.r0 = e.select(empty, c0, r0);
        in.r1 = e.select(empty, c1, r1);
        in.s0 = e.select(empty, c0, s0);
        in.s1 = e.select(empty, c1, s1);
        in.ch0 = e.divi(e.mul(pc, nch), SP);
        in.ch1 = e.divi(e.mul(e.add(pc, c1), nch), SP);
        return in;
    };

    /* ---- DMA cursor ---- */
    auto lanes_of = [&](const Info& in, std::vector<V>& cur) {
        V ln = opaque_lane();
        V l8 = e.divi(ln, 8);
        V c8 = e.bxor(e.modi(ln, 8), l8);
        for(int i = 0; i < NXI; ++i)
        {
            V xt0 = e.muli(in.tx, kBX);
            V xw = e.muli(e.addi(e.muli(wave, NXI), i), 8);
            V x = e.add(xt0, e.add(xw, l8));
            V n = e.div(x, hw);
            V rem = e.sub(x, e.mul(n, hw));
            V ho = e.div(rem, p_Wo);
            V wo = e.sub(rem, e.mul(ho, p_Wo));
            V hb = e.subi(e.muli(ho, SH), PH);
            V wb = e.subi(e.muli(wo, SH), PW);
            V pix = e.add(e.mul(e.add(e.mul(n, p_Hi), hb), p_Wi), wb);
            V pc_ = e.mul(pix, C);
            cur[xb(i)] = e.mul(e.add(pc_, e.muli(c8, 8)), c2);
            /* bit (r * S + s) set: tap (r, s) reads an in-image pixel */
            V v = c0;
            V xok = e.lt(x, NX);
            for(int r = 0; r < R; ++r)
            {
                V hok = e.land(xok, rocke_b_cmp_ult(b, e.addi(hb, (int64_t)r * DH), p_Hi));
                for(int s = 0; s < S; ++s)
                {
                    V ok = e.land(hok, rocke_b_cmp_ult(b, e.addi(wb, (int64_t)s * DW), p_Wi));
                    const int tap = r * S + s;
                    V bit = e.ci(tap < 31 ? ((int64_t)1 << tap) : INT32_MIN);
                    v = e.add(v, e.select(ok, bit, c0));
                }
            }
            cur[vm(i)] = v;
        }
        /* Weight rows: wave w DMAs LDS rows (w * NWI + i) * 8 + l / 8, row
         * 4g + j of a 16-row tile holding channel (g & 1) * 8 + (g >> 1) * 4 + j;
         * tail rows (k >= K) fall past B_bytes. */
        V m = l8;
        V gq = e.divi(m, 4);
        V g8 = e.muli(gq, 8);
        V chan = e.add(g8, e.modi(m, 4));
        V kt0 = e.muli(in.tk, BK);
        V k = e.add(e.add(kt0, e.muli(wave, NWI * 8)), chan);
        V krsc = e.mul(k, RSC);
        cur[cWo] = e.mul(e.add(krsc, e.muli(c8, 8)), c2);
    };

    auto cursor_at = [&](V i) {
        std::vector<V> cur(NAMES, (V)NULL);
        V t = tile_of(i);
        Info in = item_info(t);
        cur[cI] = i;
        cur[cT] = t;
        cur[cCh] = in.ch0;
        cur[cCh1] = in.ch1;
        cur[cR] = in.r0;
        cur[cS] = in.s0;
        cur[cR0] = in.r0;
        cur[cR1] = in.r1;
        cur[cS0] = in.s0;
        cur[cS1] = in.s1;
        lanes_of(in, cur);
        return cur;
    };

    /* scf_if_else(cond, types): then_fn / else_fn return the yielded values. */
    auto if_else =
        [&](V cond, const std::vector<const rocke_type_t*>& types, auto&& then_fn, auto&& else_fn) {
            rocke_if_else_t ife
                = rocke_b_scf_if_else_results(b, cond, types.data(), (int)types.size());
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

    const std::vector<const rocke_type_t*> cur_types(NAMES, i32_t);
    auto cur_adv = [&](const std::vector<V>& cur) {
        V s = e.add(cur[cS], c1);
        V ws = e.ge(s, cur[cS1]);
        s = e.select(ws, cur[cS0], s);
        V r = e.add(cur[cR], e.select(ws, c1, c0));
        V wr = e.ge(r, cur[cR1]);
        r = e.select(wr, cur[cR0], r);
        V ch = e.add(cur[cCh], e.select(wr, c1, c0));
        V wc = e.ge(ch, cur[cCh1]);
        std::vector<V> keep = cur;
        keep[cS] = s;
        keep[cR] = r;
        keep[cCh] = ch;
        std::vector<V> out = if_else(
            wc, cur_types, [&]() { return cursor_at(e.add(cur[cI], c1)); }, [&]() { return keep; });
        if((int)out.size() != NAMES)
        {
            out.assign(NAMES, (V)NULL);
        }
        return out;
    };

    /* ---- LDS: one allocation per stage and operand ---- */
    V in_s[2], w_s[2], in_a[2], w_a[2];
    for(int i = 0; i < 2; ++i)
    {
        std::string h = "flex_in" + std::to_string(i);
        in_s[i] = e.smem_alloc_1d(io_type, IN_E, h.c_str());
    }
    for(int i = 0; i < 2; ++i)
    {
        std::string h = "flex_w" + std::to_string(i);
        w_s[i] = e.smem_alloc_1d(io_type, W_E, h.c_str());
    }
    for(int i = 0; i < 2; ++i)
    {
        in_a[i] = rocke_b_smem_addr_of(b, in_s[i]);
    }
    for(int i = 0; i < 2; ++i)
    {
        w_a[i] = rocke_b_smem_addr_of(b, w_s[i]);
    }

    /* This wave's NXI input + NWI weight DMAs of the cursor's step into stage st. */
    auto dma = [&](const std::vector<V>& cur, int st) {
        V live = e.lt(cur[cT], TI);
        V dh = e.muli(cur[cR], DH);
        V dw = e.muli(cur[cS], DW);
        V tsp = e.mul(e.add(e.mul(dh, p_Wi), dw), C);
        V toff = e.mul(e.add(tsp, e.muli(cur[cCh], kCK)), c2);
        V xr = rocke_b_buffer_rsrc(b, P.A, e.select(live, P.A_bytes, c0));
        V tap = e.add(e.muli(cur[cR], S), cur[cS]);
        V sh_t = e.sub(e.ci(31), tap);
        for(int i = 0; i < NXI; ++i)
        {
            V ok = e.lt(rocke_b_shl(b, cur[vm(i)], sh_t), c0); /* bit tap set */
            V voff = e.select(ok, e.add(cur[xb(i)], toff), oob);
            V dst = rocke_b_smem_ptr_add(
                b, in_a[st], e.zext64(e.muli(e.addi(e.muli(wave, NXI), i), 1024)));
            rocke_b_async_buffer_load_lds(b, xr, dst, voff, c0, 4, 0);
        }
        V tc = e.mul(tap, C);
        V woff = e.mul(e.add(tc, e.muli(cur[cCh], kCK)), c2);
        V wr = rocke_b_buffer_rsrc(b, P.Bp, e.select(live, P.B_bytes, c0));
        for(int i = 0; i < NWI; ++i)
        {
            V dst = rocke_b_smem_ptr_add(
                b, w_a[st], e.zext64(e.muli(e.addi(e.muli(wave, NWI), i), 1024)));
            const int wsh = ((i & 1) * 4 + (i >> 1) * 16) * 2;
            V soff = e.add(woff, e.muli(RSC, wsh));
            rocke_b_async_buffer_load_lds(b, wr, dst, cur[cWo], soff, 4, 0);
        }
    };

    /* Fragments of stage st: a[h][kt] (weights), bb[h][xt] (input). */
    auto frags = [&](int st, std::vector<std::vector<V>>& a, std::vector<std::vector<V>>& bb) {
        V ln = opaque_lane();
        V r16 = e.modi(ln, 16);
        V q = e.divi(ln, 16);
        a.assign(2, std::vector<V>());
        bb.assign(2, std::vector<V>());
        for(int h = 0; h < 2; ++h)
        {
            V kh = e.ci(h * 4);
            V hq = e.add(kh, q);
            V sl = e.muli(e.bxor(hq, e.modi(r16, 8)), 8);
            V xg = e.muli(g, kBX / 2);
            V xrow = e.add(e.add(xg, e.muli(wx, 64)), r16);
            V krow = e.add(e.muli(wk, KT * 16), r16);
            for(int xt = 0; xt < kXT; ++xt)
            {
                V idx = e.add(e.muli(e.addi(xrow, xt * 16), kCK), sl);
                bb[h].push_back(e.smem_load(in_s[st], c0, idx, io_type, 8));
            }
            for(int kt = 0; kt < KT; ++kt)
            {
                V idx = e.add(e.muli(e.addi(krow, kt * 16), kCK), sl);
                a[h].push_back(e.smem_load(w_s[st], c0, idx, io_type, 8));
            }
        }
    };

    auto mfmas = [&](std::vector<V> accs,
                     const std::vector<std::vector<V>>& a,
                     const std::vector<std::vector<V>>& bb) {
        for(int h = 0; h < 2; ++h)
        {
            for(int xt = 0; xt < kXT; ++xt)
            {
                for(int kt = 0; kt < KT; ++kt)
                {
                    const int i = xt * KT + kt;
                    accs[i] = e.mfma16x16x32(bf16, a[h][kt], bb[h][xt], accs[i]);
                }
            }
        }
        return accs;
    };

    auto bar = [&](int prio) {
        rocke_b_s_setprio(b, prio);
        rocke_b_s_barrier_bare(b);
        rocke_b_sched_barrier(b, 0);
    };

    /* ---- epilogue ---- */
    const rocke_type_t* i32x2 = rocke_vector_type(b, i32_t, 2);
    const rocke_type_t* half8 = rocke_vector_type(b, io_type, 8);
    V d_rsrc = rocke_b_buffer_rsrc(b, P.D, P.D_bytes);

    auto epilogue = [&](const std::vector<V>& accs, V t) {
        V tl = e.divi(t, SP);
        V tx = e.div(tl, nkt);
        V tk = e.mod(tl, nkt);
        V ln = opaque_lane();
        V l16 = e.modi(ln, 16);
        V q = e.divi(ln, 16);
        V xa = e.muli(tx, kBX);
        V xg = e.add(xa, e.muli(g, kBX / 2));
        V xw = e.add(e.muli(wx, 64), l16);
        V xbase = e.add(xg, xw);
        V ka = e.muli(tk, BK);
        V kbase = e.add(ka, e.muli(wk, KT * 16));
        if(SP == 1)
        {
            V qa = e.muli(e.div(q, c2), 16);
            V kq = e.add(qa, e.muli(e.mod(q, c2), 8));
            for(int xt = 0; xt < kXT; ++xt)
            {
                V x = e.addi(xbase, xt * 16);
                for(int cc = 0; cc < KT / 2; ++cc)
                {
                    V da = rocke_b_vec_bitcast(
                        b, rocke_dconv_trunc_f32(b, bf16, accs[xt * KT + 2 * cc]), i32x2);
                    V db = rocke_b_vec_bitcast(
                        b, rocke_dconv_trunc_f32(b, bf16, accs[xt * KT + 2 * cc + 1]), i32x2);
                    V lo0, hi0, lo1, hi1;
                    V a0 = rocke_b_vec_extract(b, da, 0);
                    rocke_b_permlane32_swap(b, a0, rocke_b_vec_extract(b, db, 0), &lo0, &hi0);
                    V a1 = rocke_b_vec_extract(b, da, 1);
                    rocke_b_permlane32_swap(b, a1, rocke_b_vec_extract(b, db, 1), &lo1, &hi1);
                    V comps[4] = {lo0, lo1, hi0, hi1};
                    V val = rocke_b_vec_bitcast(b, rocke_b_vec_pack(b, comps, 4, i32_t), half8);
                    V k = e.add(e.addi(kbase, cc * 32), kq);
                    V xok = e.lt(x, NX);
                    V ok = e.land(xok, e.lt(k, K));
                    V xk = e.mul(x, K);
                    V off = e.select(ok, e.mul(e.add(xk, k), c2), oob);
                    rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, val, 4);
                }
            }
        }
        else
        {
            V pc = e.modi(t, SP);
            V slab = e.muli(e.mul(NX, K), 4);
            V pc64 = e.zext64(pc);
            V base = rocke_b_global_ptr_offset(b, P.D, e.mul(pc64, e.zext64(slab)));
            V s_rsrc = rocke_b_buffer_rsrc(b, base, slab);
            V qa = e.muli(e.mod(q, c2), 8);
            V kq = e.add(qa, e.muli(e.div(q, c2), 4));
            for(int xt = 0; xt < kXT; ++xt)
            {
                V x = e.addi(xbase, xt * 16);
                for(int kt = 0; kt < KT; ++kt)
                {
                    V k = e.add(e.addi(kbase, kt * 16), kq);
                    V xok = e.lt(x, NX);
                    V ok = e.land(xok, e.lt(k, K));
                    V xk = e.mul(x, K);
                    V off = e.select(ok, e.muli(e.add(xk, k), 4), oob);
                    rocke_b_buffer_store_vN_f32(b, s_rsrc, off, c0, accs[xt * KT + kt], 4);
                }
            }
        }
    };

    /* ---- the two groups ---- */
    V zero = rocke_b_zero_vec_f32(b, 4);
    const rocke_type_t* vec4 = zero->type;
    std::vector<const rocke_type_t*> res_types(nacc, vec4);
    res_types.insert(res_types.end(), NAMES + 1, i32_t);

    /* One step reading stage st (static); updates accs and cur. */
    auto step = [&](std::vector<V>& accs, std::vector<V>& cur, int st, bool g1) {
        std::vector<std::vector<V>> a, bb;
        if(!g1)
        {
            dma(cur, 1 - st);
            cur = cur_adv(cur);
            frags(st, a, bb);
            rocke_b_s_waitcnt(b, -1, 0, -1);
            rocke_b_sched_barrier(b, 0);
            bar(1);
            accs = mfmas(accs, a, bb);
            rocke_b_sched_barrier(b, 0);
            rocke_b_s_waitcnt(b, 0, -1, -1);
            bar(0);
        }
        else
        {
            frags(st, a, bb);
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_sched_barrier(b, 0);
            bar(1);
            dma(cur, st); /* step + 2 into this stage (read by both groups) */
            cur = cur_adv(cur);
            accs = mfmas(accs, a, bb);
            rocke_b_sched_barrier(b, 0);
            bar(0);
        }
    };

    /* All steps of an item that starts on stage p: pairs, then an odd tail.
     * Returns accs + cur + the next stage. */
    auto item_steps = [&](const std::vector<V>& accs0,
                          const std::vector<V>& cur0,
                          V steps,
                          int par,
                          bool g1,
                          const std::string& tag) {
        std::vector<std::string> it_names;
        std::vector<V> inits;
        for(int i = 0; i < nacc; ++i)
        {
            it_names.push_back(tag + "a" + std::to_string(i));
            inits.push_back(accs0[i]);
        }
        for(int k = 0; k < NAMES; ++k)
        {
            it_names.push_back(tag + "c_" + names[k]);
            inits.push_back(cur0[k]);
        }
        std::string iv = tag + "pair";
        std::vector<V> res = e.for_iter(
            c0, e.div(steps, c2), c1, it_names, inits, iv.c_str(), [&](V, std::vector<V> lc) {
                std::vector<V> la(lc.begin(), lc.begin() + nacc);
                std::vector<V> lcur(lc.begin() + nacc, lc.end());
                step(la, lcur, par, g1);
                step(la, lcur, 1 - par, g1);
                la.insert(la.end(), lcur.begin(), lcur.end());
                return la;
            });
        if((int)res.size() != nacc + NAMES)
        {
            return std::vector<V>();
        }
        std::vector<V> accs(res.begin(), res.begin() + nacc);
        std::vector<V> cur(res.begin() + nacc, res.end());
        V odd = e.eq(e.mod(steps, c2), c1);
        return if_else(
            odd,
            res_types,
            [&]() {
                std::vector<V> ta = accs, tcur = cur;
                step(ta, tcur, par, g1);
                ta.insert(ta.end(), tcur.begin(), tcur.end());
                ta.push_back(e.ci(1 - par));
                return ta;
            },
            [&]() {
                std::vector<V> out = accs;
                out.insert(out.end(), cur.begin(), cur.end());
                out.push_back(e.ci(par));
                return out;
            });
    };

    auto run = [&](bool g1) {
        std::vector<V> cur = cursor_at(c0);
        dma(cur, 0);
        cur = cur_adv(cur);
        rocke_b_s_waitcnt(b, 0, -1, -1);
        rocke_b_s_barrier_bare(b);
        if(g1)
        {
            rocke_b_s_barrier_bare(b); /* group 1 runs one barrier behind */
            dma(cur, 1);
            cur = cur_adv(cur);
        }
        rocke_b_sched_barrier(b, 0);
        std::vector<std::string> o_names;
        for(int k = 0; k < NAMES; ++k)
        {
            o_names.push_back("o_" + names[k]);
        }
        o_names.push_back("o_st");
        std::vector<V> o_inits = cur;
        o_inits.push_back(c0);
        e.for_iter(c0, rounds, c1, o_names, o_inits, "rnd", [&](V rnd, std::vector<V> oc) {
            std::vector<V> ocur(oc.begin(), oc.begin() + NAMES);
            V st = oc[NAMES];
            V t = tile_of(rnd);
            Info in = item_info(t);
            V dr = e.sub(in.r1, in.r0);
            V nrs = e.mul(dr, e.sub(in.s1, in.s0));
            V steps = e.mul(nrs, e.sub(in.ch1, in.ch0));
            std::vector<V> accs(nacc, zero);
            std::vector<V> res = if_else(
                e.eq(st, c0),
                res_types,
                [&]() { return item_steps(accs, ocur, steps, 0, g1, "p0"); },
                [&]() { return item_steps(accs, ocur, steps, 1, g1, "p1"); });
            if((int)res.size() != nacc + NAMES + 1)
            {
                return std::vector<V>();
            }
            epilogue(std::vector<V>(res.begin(), res.begin() + nacc), t);
            rocke_b_sched_barrier(b, 0);
            return std::vector<V>(res.begin() + nacc, res.end());
        });
        rocke_b_s_setprio(b, 0);
        if(!g1)
        {
            rocke_b_s_barrier_bare(b); /* balance the barrier count */
        }
    };

    rocke_if_else_t top = rocke_b_scf_if_else(b, e.eq(g, c0));
    if(top.then_region == NULL || top.else_region == NULL)
    {
        return NULL;
    }
    rocke_b_region_enter(b, top.then_region);
    run(false);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, top.else_region);
    run(true);
    rocke_b_region_leave(b);
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_fwd_flex_mfma(rocke_ir_builder_t* b,
                                                         const rocke_fwd_flex_mfma_spec_t* spec,
                                                         const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_flex(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_fwd_flex_mfma_new(rocke_ir_builder_t* b,
                                                             const rocke_fwd_flex_mfma_spec_t* spec,
                                                             const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_fwd_flex_mfma",
        [&](char* out, size_t cap) { return rocke_fwd_flex_mfma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_fwd_flex_mfma(b, spec, arch); });
}

extern "C" rocke_status_t rocke_fwd_flex_mfma_lower_to_llvm(const rocke_fwd_flex_mfma_spec_t* spec,
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
        "build_fwd_flex_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_fwd_flex_mfma_new(b, spec, arch); });
}
