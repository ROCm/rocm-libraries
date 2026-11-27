// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_tap1d_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_fwd_tap1d_mfma.py: the gfx950 1-D
 * tap-reuse MFMA direct forward (build_fwd_tap1d_mfma).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_fwd_tap1d_mfma.h"

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

constexpr int kKT = 2; /* 16-channel MFMA tiles per wave along K */
constexpr int64_t kOob = ((int64_t)1 << 31) - 1;

/* a * b, saturated at INT64_MAX (Python's ints do not wrap). */
long long mul_sat(long long a, long long b)
{
    int64_t out;
    return rocke_intexpr_checked_mul(a, b, &out) ? out : INT64_MAX;
}

int rows_per_dma(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    return s->ck > 0 ? 1024 / (s->ck * 2) : 0;
}

} // namespace

extern "C" rocke_fwd_tap1d_mfma_spec_t rocke_fwd_tap1d_mfma_spec_default(void)
{
    rocke_fwd_tap1d_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.bk = 128;
    s.nwv = 8;
    s.seg = 9;
    s.ck = 64;
    s.one = false;
    s.name = "fwd_tap1d_mfma";
    s.wave_size = 64;
    return s;
}

extern "C" int rocke_fwd_tap1d_mfma_nwk(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    return (int)rocke_floor_div(s->bk, kKT * 16);
}

extern "C" int rocke_fwd_tap1d_mfma_nwx(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    const int nwk = rocke_fwd_tap1d_mfma_nwk(s);
    return nwk != 0 ? (int)rocke_floor_div(s->nwv, nwk) : 0;
}

extern "C" int rocke_fwd_tap1d_mfma_ncol(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    const int d = s->problem.dil_w;
    return d != 0 ? d * (int)rocke_floor_div(rocke_fwd_tap1d_mfma_nwx(s) * 16, d) : 0;
}

extern "C" int rocke_fwd_tap1d_mfma_bx(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    return rocke_fwd_tap1d_mfma_ncol(s) * s->seg;
}

extern "C" int rocke_fwd_tap1d_mfma_win(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    return rocke_fwd_tap1d_mfma_bx(s) + s->problem.dil_w * (s->problem.KW - 1);
}

extern "C" int rocke_fwd_tap1d_mfma_win_dmas(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    const int rpd = rows_per_dma(s);
    return rpd > 0 ? (int)rocke_ceil_div(rocke_fwd_tap1d_mfma_win(s), rpd) : 0;
}

extern "C" int rocke_fwd_tap1d_mfma_wgt_dmas(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    const int rpd = rows_per_dma(s);
    return rpd > 0 ? (int)rocke_ceil_div(s->bk, rpd) : 0;
}

extern "C" long long rocke_fwd_tap1d_mfma_lds_bytes(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    const int nwb = s->one ? 1 : 2;
    return ((long long)nwb * rocke_fwd_tap1d_mfma_win_dmas(s)
            + 2LL * rocke_fwd_tap1d_mfma_wgt_dmas(s))
           * 1024;
}

extern "C" int rocke_fwd_tap1d_mfma_threads_per_block(const rocke_fwd_tap1d_mfma_spec_t* s)
{
    return s->nwv * s->wave_size;
}

extern "C" void rocke_fwd_tap1d_mfma_grid(const rocke_fwd_tap1d_mfma_spec_t* s, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    const int bx = rocke_fwd_tap1d_mfma_bx(s);
    long long g = 0;
    if(bx != 0 && s->bk != 0)
    {
        g = mul_sat(mul_sat(mul_sat(p->N, rocke_direct_conv_problem_Ho(p)),
                            rocke_ceil_div(rocke_direct_conv_problem_Wo(p), bx)),
                    rocke_floor_div(rocke_direct_conv_problem_total_k(p), s->bk));
    }
    grid[0] = (int)g;
    grid[1] = 1;
    grid[2] = 1;
}

extern "C" rocke_status_t rocke_fwd_tap1d_mfma_kernel_name(const rocke_fwd_tap1d_mfma_spec_t* s,
                                                           char* out,
                                                           size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[32], p_buf[32], d_buf[32], k_buf[32], w_buf[32], g_buf[32], c_buf[32];
    snprintf(f_buf, sizeof f_buf, "f1x%d", p->KW);
    snprintf(p_buf, sizeof p_buf, "p%d", rocke_direct_conv_problem_pad_w(p));
    snprintf(d_buf, sizeof d_buf, "d%d", p->dil_w);
    snprintf(k_buf, sizeof k_buf, "k%d", s->bk);
    snprintf(w_buf, sizeof w_buf, "w%d", s->nwv);
    snprintf(g_buf, sizeof g_buf, "g%d", s->seg);
    snprintf(c_buf, sizeof c_buf, "c%d", s->ck);
    const char* parts[7] = {f_buf, p_buf, d_buf, k_buf, w_buf, g_buf, c_buf};
    const char* flags[2] = {"one", "bf16"};
    int on[2] = {s->one ? 1 : 0, problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 7, flags, on, 2, out, out_cap, NULL);
}

extern "C" bool rocke_fwd_tap1d_mfma_is_valid_spec(const rocke_fwd_tap1d_mfma_spec_t* s,
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
    const long long tc = rocke_direct_conv_problem_total_c(p);
    const long long tk = rocke_direct_conv_problem_total_k(p);
    const int PW = rocke_direct_conv_problem_pad_w(p);
    const char* ab = problem_is_bf16(p) ? "bf16" : "f16";
    const char* why = NULL;
    if(target->wave_size != 64 || s->wave_size != 64)
    {
        why = "fwd_tap1d_mfma is a wave64 MFMA kernel";
    }
    else if(rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "fwd_tap1d_mfma needs the 16-byte LDS DMA (absent on %s)", arch);
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
    else if(rocke_direct_conv_pad_w_reason(p, "", msg, sizeof msg))
    {
        why = msg;
    }
    else if(p->groups != 1)
    {
        why = "groups must be 1";
    }
    else if(p->KH != 1 || p->PAD != 0 || p->stride != 1)
    {
        why = "1 x S filters, no height padding, stride 1 only";
    }
    else if(PW < 0 || (long long)PW > (long long)p->dil_w * (p->KW - 1))
    {
        why = "width padding must be in [0, dilation * (S - 1)]";
    }
    else if((s->bk != 32 && s->bk != 64 && s->bk != 128) || (s->nwv != 4 && s->nwv != 8)
            || s->nwv % rocke_fwd_tap1d_mfma_nwk(s))
    {
        why = "bk must be 32/64/128 and nwv 4/8 with nwv % (bk / 32) == 0";
    }
    else if(s->seg < 1 || rocke_fwd_tap1d_mfma_ncol(s) < 1)
    {
        why = "seg must be >= 1 and the dilation at most 16 * waves along X";
    }
    else if((s->ck != 32 && s->ck != 64) || tc % s->ck)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "ck %d must be 32 or 64 and divide C %lld", s->ck, tc);
        why = msg;
    }
    else if(s->one != (tc == s->ck))
    {
        ROCKE_ERR_SNPRINTF(msg,
                           sizeof msg,
                           "one (single chunk) must be set exactly when C == ck (%lld vs %d)",
                           tc,
                           s->ck);
        why = msg;
    }
    else if(tk % s->bk)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "K %lld must be a multiple of bk %d", tk, s->bk);
        why = msg;
    }
    else if(rocke_direct_conv_problem_Wo(p) < 1)
    {
        why = "empty output";
    }
    else if(!rocke_arch_fits_lds(target, (long)rocke_fwd_tap1d_mfma_lds_bytes(s)))
    {
        ROCKE_ERR_SNPRINTF(msg,
                           sizeof msg,
                           "LDS %lld B > %d",
                           rocke_fwd_tap1d_mfma_lds_bytes(s),
                           (int)target->lds_capacity_bytes);
        why = msg;
    }
    else
    {
        const long long nx = mul_sat(mul_sat(p->N, rocke_direct_conv_problem_Ho(p)),
                                     rocke_direct_conv_problem_Wo(p));
        const struct
        {
            const char* what;
            long long bytes;
        } tensors[] = {
            {"input", mul_sat(mul_sat(mul_sat(mul_sat(2, p->N), p->H), p->W), tc)},
            {"output", mul_sat(mul_sat(2, nx), tk)},
            {"weights", mul_sat(mul_sat(mul_sat(2, tk), p->KW), tc)},
        };
        for(const auto& t : tensors)
        {
            if(t.bytes > kOob)
            {
                ROCKE_ERR_SNPRINTF(msg,
                                   sizeof msg,
                                   "%s exceeds %lld B, the i32 buffer range",
                                   t.what,
                                   (long long)kOob);
                why = msg;
                break;
            }
        }
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

rocke_kernel_def_t*
    build_tap1d(rocke_ir_builder_t* b, const rocke_fwd_tap1d_mfma_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_fwd_tap1d_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid FwdTap1dMfmaSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const int S = p.KW, DIL = p.dil_w, PW = rocke_direct_conv_problem_pad_w(&p);
    const int BK = spec->bk, NWV = spec->nwv, SEG = spec->seg, CK = spec->ck;
    const int NWK = rocke_fwd_tap1d_mfma_nwk(spec), NCOL = rocke_fwd_tap1d_mfma_ncol(spec);
    const int BX = rocke_fwd_tap1d_mfma_bx(spec), WIN = rocke_fwd_tap1d_mfma_win(spec);
    const int G = CK / 8; /* 16-byte granules per LDS row */
    const int RPI = 1024 / (CK * 2); /* LDS rows per DMA instruction */
    const int NH = CK / 32; /* 32-channel MFMA halves per chunk */
    const int WIN_I = rocke_fwd_tap1d_mfma_win_dmas(spec);
    const int WGT_I = rocke_fwd_tap1d_mfma_wgt_dmas(spec);
    const int WIN_P = cdiv(WIN_I, NWV * S); /* window DMAs per wave and step */
    const int WGT_P = cdiv(WGT_I, NWV); /* weight DMAs per wave and step */
    const bool ONE = spec->one;
    const rocke_type_t* i32_t = rocke_i32();

    rocke_attr_set_int(
        b, &b->kernel->attrs, "max_workgroup_size", rocke_fwd_tap1d_mfma_threads_per_block(spec));
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
    V p_Wi = P.p_Wi, p_Wo = P.p_Wo;
    V C = P.p_total_c, K = P.p_total_k;

    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    V kg = e.modi(wave, NWK);
    V xg = e.divi(wave, NWK);

    /* lane-derived addressing is recomputed where it is used instead of being
     * hoisted into long-lived VGPRs */
    auto opaque_lane = [&]() -> V {
        rocke_inline_asm_opts_t opts = {};
        opts.sideeffect = true;
        opts.sideeffect_set = true;
        rocke_op_t* op = rocke_b_inline_asm(b, "", "=v,0", &lane, 1, &i32_t, 1, &opts);
        return op ? op->results[0] : NULL;
    };

    /* block -> (output row, X tile, K tile); K tile fastest */
    V nkb = e.divi(K, BK);
    V nxt = e.cdivi(p_Wo, BX);
    V t = rocke_b_block_id_x(b);
    V kb = e.mod(t, nkb);
    t = e.div(t, nkb);
    V wo0 = e.muli(e.mod(t, nxt), BX);
    V nrow = e.div(t, nxt); /* n * Ho + ho (= input row: KH 1, no height padding) */
    V nch = e.divi(C, CK);

    V a_bytes = P.A_bytes;
    V w_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes);

    /* ---- LDS: one allocation per buffer ---- */
    std::vector<V> win_s, wgt_s, win_a, wgt_a;
    for(int i = 0; i < (ONE ? 1 : 2); ++i)
    {
        std::string h = "tap1d_x" + std::to_string(i);
        win_s.push_back(e.smem_alloc_1d(io_type, WIN_I * 512, h.c_str()));
    }
    for(int i = 0; i < 2; ++i)
    {
        std::string h = "tap1d_w" + std::to_string(i);
        wgt_s.push_back(e.smem_alloc_1d(io_type, WGT_I * 512, h.c_str()));
    }
    for(V v : win_s)
    {
        win_a.push_back(rocke_b_smem_addr_of(b, v));
    }
    for(V v : wgt_s)
    {
        wgt_a.push_back(rocke_b_smem_addr_of(b, v));
    }

    /* LDS granule slot of granule gr in LDS row `row` (gr NULL: the constant
     * granule gr_c). */
    auto swz_slot = [&](V row, V gr) {
        V sh = e.lshri(row, 3);
        V x = e.bxor(row, sh);
        V m = e.land(x, e.ci(G - 1));
        return e.bxor(gr, m);
    };

    /* Emit emit() for DMA instruction ii = first .. last over the waves:
     * statically, or behind a wave-uniform ii < nmax branch. */
    auto guarded = [&](int first, int last, int nmax, V ii, auto&& emit) {
        if(first >= nmax)
        {
            return;
        }
        if(last < nmax)
        {
            emit();
            return;
        }
        e.if_(e.lti(ii, nmax), emit);
    };

    /* Window piece st of chunk ch into window buffer buf. */
    auto dma_win = [&](V ch, int buf, int st, V live) {
        V rs = rocke_b_buffer_rsrc(b, P.A, live == NULL ? a_bytes : e.select(live, a_bytes, c0));
        for(int j = 0; j < WIN_P; ++j)
        {
            V k0 = e.ci((int64_t)st * NWV);
            V ia = e.add(k0, wave);
            V ii = e.addi(e.muli(ia, WIN_P), j);
            auto emit = [&, ii]() {
                V ln = opaque_lane();
                V ra = e.muli(ii, RPI);
                V row = e.add(ra, e.divi(ln, G));
                V gr = swz_slot(row, e.modi(ln, G)); /* the swizzle is an involution */
                V xx = e.add(e.subi(wo0, PW), row);
                V rlt = e.lti(row, WIN);
                V ok = e.land(rlt, rocke_b_cmp_ult(b, xx, p_Wi));
                V pix = e.add(e.mul(nrow, p_Wi), xx);
                V oa = e.mul(pix, C);
                V ob = e.add(oa, e.muli(ch, CK));
                V oc = e.add(ob, e.muli(gr, 8));
                V off = e.mul(oc, c2);
                V dst = rocke_b_smem_ptr_add(b, win_a[buf], e.zext64(e.muli(ii, 1024)));
                rocke_b_async_buffer_load_lds(b, rs, dst, e.select(ok, off, oob), c0, 4, 0);
            };
            guarded((st * NWV) * WIN_P + j, (st * NWV + NWV - 1) * WIN_P + j, WIN_I, ii, emit);
        }
    };

    /* Weights of (chunk ch, tap) into weight stage `stage`. */
    auto dma_wgt = [&](V ch, int tap, int stage, V live) {
        V rs = live == NULL ? w_rsrc : rocke_b_buffer_rsrc(b, P.Bp, e.select(live, P.B_bytes, c0));
        V kt = e.ci(tap);
        V sa = e.mul(kt, C);
        V soff = e.mul(e.add(sa, e.muli(ch, CK)), c2);
        for(int j = 0; j < WGT_P; ++j)
        {
            V ii = e.addi(e.muli(wave, WGT_P), j);
            auto emit = [&, ii]() {
                V ln = opaque_lane();
                V ra = e.muli(ii, RPI);
                V row = e.add(ra, e.divi(ln, G));
                V gr = swz_slot(row, e.modi(ln, G));
                V k = e.add(e.muli(kb, BK), row);
                V oa = e.mul(e.muli(k, S), C);
                V off = e.mul(e.add(oa, e.muli(gr, 8)), c2);
                V dst = rocke_b_smem_ptr_add(b, wgt_a[stage], e.zext64(e.muli(ii, 1024)));
                rocke_b_async_buffer_load_lds(b, rs, dst, off, soff, 4, 0);
            };
            guarded(j, (NWV - 1) * WGT_P + j, WGT_I, ii, emit);
        }
    };

    auto rd = [&](V buf, V row, V gr) {
        V ia = e.muli(row, CK);
        V idx = e.add(ia, e.muli(swz_slot(row, gr), 8));
        return e.smem_load(buf, c0, idx, io_type, 8);
    };

    const int nacc = SEG * kKT;

    /* fragment-read lane bases: computed once (hoisted out of the tap / chunk
     * loops, unlike the DMA lane math) */
    V l16r = e.modi(lane, 16);
    V qr = e.divi(lane, 16);
    V colr = e.add(e.muli(xg, 16), l16r);
    if(NCOL < NWV / NWK * 16)
    {
        V ge = e.gei(colr, NCOL);
        colr = e.select(ge, e.subi(colr, DIL), colr);
    }
    V xa = e.modi(colr, DIL);
    V xrow0 = e.add(xa, e.muli(e.divi(colr, DIL), DIL * SEG));
    V krow0 = e.add(e.muli(kg, kKT * 16), l16r);

    /* One chunk read from window buffer wb; the weight stage of tap s is
     * (ws0 + s) & 1. nxt_live: there is a next chunk (NULL: statically none). */
    auto chunk = [&](std::vector<V> accs, V ch, int wb, int ws0, V nxt_live) {
        std::vector<std::vector<V>> fr(NH, std::vector<V>(SEG + S - 1, (V)NULL));
        for(int s = 0; s < S; ++s)
        {
            const int ws = (ws0 + s) & 1;
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_s_barrier_bare(b);
            rocke_b_sched_barrier(b, 0);
            if(s + 1 < S)
            {
                dma_wgt(ch, s + 1, ws ^ 1, NULL);
            }
            else if(nxt_live != NULL)
            {
                dma_wgt(e.add(ch, c1), 0, ws ^ 1, nxt_live);
            }
            if(nxt_live != NULL)
            {
                dma_win(e.add(ch, c1), wb ^ 1, s, nxt_live);
            }
            std::vector<std::vector<V>> a;
            for(int h = 0; h < NH; ++h)
            {
                V hk = e.ci(h * 4);
                V gr = e.add(hk, qr);
                std::vector<V> ah;
                for(int kt = 0; kt < kKT; ++kt)
                {
                    ah.push_back(rd(wgt_s[ws], e.addi(krow0, kt * 16), gr));
                }
                a.push_back(ah);
                const int v0 = s == 0 ? 0 : SEG + s - 1;
                const int v1 = s == 0 ? SEG : SEG + s;
                for(int v = v0; v < v1; ++v)
                {
                    fr[h][v] = rd(win_s[wb], e.addi(xrow0, (int64_t)DIL * v), gr);
                }
            }
            rocke_b_sched_barrier(b, 0);
            for(int h = 0; h < NH; ++h)
            {
                for(int u = 0; u < SEG; ++u)
                {
                    for(int kt = 0; kt < kKT; ++kt)
                    {
                        const int i = u * kKT + kt;
                        accs[i] = e.mfma16x16x32(bf16, a[h][kt], fr[h][u + s], accs[i]);
                    }
                }
            }
            rocke_b_sched_barrier(b, 0);
        }
        return accs;
    };

    V zero = rocke_b_zero_vec_f32(b, 4);
    const rocke_type_t* vec4 = zero->type;
    std::vector<V> accs(nacc, zero);

    /* prologue: the whole window of chunk 0 and the weights of its first tap */
    for(int st = 0; st < S; ++st)
    {
        dma_win(c0, 0, st, NULL);
    }
    dma_wgt(c0, 0, 0, NULL);

    if(ONE)
    {
        accs = chunk(accs, c0, 0, 0, NULL);
    }
    else
    {
        accs = e.for_iter(c0,
                          e.div(nch, c2),
                          c1,
                          iter_names("a", accs.size()),
                          accs,
                          "pair",
                          [&](V pair, std::vector<V> la) {
                              V ch = e.mul(pair, c2);
                              V l0 = e.lt(e.add(ch, c1), nch);
                              la = chunk(la, ch, 0, 0, l0);
                              V ch1 = e.add(ch, c1);
                              V l1 = e.lt(e.add(ch, c2), nch);
                              return chunk(la, ch1, 1, S & 1, l1);
                          });
        std::vector<const rocke_type_t*> types(nacc, vec4);
        V cond = e.eq(e.mod(nch, c2), c1);
        rocke_if_else_t ife = rocke_b_scf_if_else_results(b, cond, types.data(), (int)types.size());
        if(ife.op == NULL || ife.then_region == NULL || ife.else_region == NULL)
        {
            return NULL;
        }
        rocke_b_region_enter(b, ife.then_region);
        std::vector<V> tv = chunk(accs, e.sub(nch, c1), 0, 0, NULL);
        rocke_b_scf_yield(b, tv.data(), (int)tv.size());
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ife.else_region);
        rocke_b_scf_yield(b, accs.data(), (int)accs.size());
        rocke_b_region_leave(b);
        accs.assign(ife.op->results, ife.op->results + ife.op->num_results);
    }

    /* ---- epilogue: acc[u][kt] holds channels k .. k + 3 of pixel px ---- */
    V d_rsrc = rocke_b_buffer_rsrc(b, P.D, P.D_bytes);
    V ln = opaque_lane();
    V l16 = e.modi(ln, 16);
    V q = e.divi(ln, 16);
    V col = e.add(e.muli(xg, 16), l16);
    V colok = e.lti(col, NCOL);
    V pa = e.modi(col, DIL);
    V pb = e.add(pa, e.muli(e.divi(col, DIL), DIL * SEG));
    V px0 = e.add(wo0, pb);
    V ka = e.muli(kb, BK);
    V kbb = e.add(ka, e.muli(kg, kKT * 16));
    V kbase = e.add(kbb, e.muli(q, 4));
    V obase = e.mul(nrow, p_Wo);
    for(int u = 0; u < SEG; ++u)
    {
        V px = e.addi(px0, (int64_t)DIL * u);
        V ok = e.land(colok, e.lt(px, p_Wo));
        for(int kt = 0; kt < kKT; ++kt)
        {
            V k = e.addi(kbase, kt * 16);
            V oa = e.mul(e.add(obase, px), K);
            V offv = e.mul(e.add(oa, k), c2);
            V off = e.select(ok, offv, oob);
            V val = rocke_dconv_trunc_f32(b, bf16, accs[u * kKT + kt]);
            rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, val, 2);
        }
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_fwd_tap1d_mfma(rocke_ir_builder_t* b,
                                                          const rocke_fwd_tap1d_mfma_spec_t* spec,
                                                          const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_tap1d(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_fwd_tap1d_mfma_new(
    rocke_ir_builder_t* b, const rocke_fwd_tap1d_mfma_spec_t* spec, const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_fwd_tap1d_mfma",
        [&](char* out, size_t cap) { return rocke_fwd_tap1d_mfma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_fwd_tap1d_mfma(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_fwd_tap1d_mfma_lower_to_llvm(const rocke_fwd_tap1d_mfma_spec_t* spec,
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
        "build_fwd_tap1d_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_fwd_tap1d_mfma_new(b, spec, arch); });
}
