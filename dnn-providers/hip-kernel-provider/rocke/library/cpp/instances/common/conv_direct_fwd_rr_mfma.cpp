// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_rr_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_fwd_rr_mfma.py: the gfx950
 * rows-in-registers MFMA direct forward (build_fwd_rr_mfma) and its
 * weight-format helper (build_fwd_rr_wfmt).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_fwd_rr_mfma.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <functional>
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
constexpr int kTP = 16, kTQ = 32; /* block tile: rows x pixels */
constexpr int kWP = 8; /* output rows per wave */
constexpr int kNIN = kWP + 2; /* input row fragments per step */
constexpr int kPXB = kCK * 2; /* bytes per staged pixel (64 channels) */
constexpr int kBufB = 80 * 1024; /* bytes per input buffer */
constexpr int kTileB = 1024; /* one 16 x 32 A-operand tile */
constexpr int kStNt = 2; /* store aux: nt */
constexpr int kPersist = ROCKE_FWD_RR_MFMA_PERSIST;
constexpr int kXcdGfx950 = 8; /* XCDs of gfx950 (_XCD_ARCHES) */
/* RR_SLIM / RR_WIMM (module switches, both on). */
constexpr bool kRrSlim = true;
constexpr bool kRrWimm = true;
constexpr int64_t kOob = ((int64_t)1 << 31) - 1;

/* a * b, saturated at INT64_MAX (Python's ints do not wrap). */
long long mul_sat(long long a, long long b)
{
    int64_t out;
    return rocke_intexpr_checked_mul(a, b, &out) ? out : INT64_MAX;
}

int step_b(int kw16, int nwk)
{
    return nwk * 3 * kw16 * kTileB;
}

int stores_of(int kw16)
{
    return kWP * (kw16 > 2 ? 2 : 1);
}

} // namespace

/* ===================================================================== *
 *  FwdRrMfmaSpec
 * ===================================================================== */

extern "C" rocke_fwd_rr_mfma_spec_t rocke_fwd_rr_mfma_spec_default(void)
{
    rocke_fwd_rr_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "fwd_rr_mfma";
    s.wave_size = 64;
    s.kw16 = 4;
    return s;
}

extern "C" int rocke_fwd_rr_mfma_nwk(const rocke_fwd_rr_mfma_spec_t* s)
{
    return s->k4w ? 4 : 2;
}

extern "C" int rocke_fwd_rr_mfma_tp(const rocke_fwd_rr_mfma_spec_t* s)
{
    return s->k4w ? 8 : kTP;
}

extern "C" int rocke_fwd_rr_mfma_threads_per_block(const rocke_fwd_rr_mfma_spec_t* s)
{
    return 8 * s->wave_size;
}

extern "C" int rocke_fwd_rr_mfma_tk(const rocke_fwd_rr_mfma_spec_t* s)
{
    return 16 * rocke_fwd_rr_mfma_nwk(s) * s->kw16;
}

extern "C" long long rocke_fwd_rr_mfma_tiles(const rocke_fwd_rr_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    const long long wo = rocke_direct_conv_problem_Wo(p);
    const long long ho = rocke_direct_conv_problem_Ho(p);
    const int tk = rocke_fwd_rr_mfma_tk(s);
    long long cols = s->unfold ? rocke_ceil_div(p->N, kTQ) : mul_sat(p->N, rocke_ceil_div(wo, kTQ));
    if(s->chunk)
    {
        cols = rocke_ceil_div(mul_sat(p->N, rocke_ceil_div(wo, 8)), 4);
    }
    if(tk == 0)
    {
        return 0;
    }
    return mul_sat(mul_sat(cols, rocke_ceil_div(ho, rocke_fwd_rr_mfma_tp(s))),
                   rocke_floor_div(rocke_direct_conv_problem_total_k(p), tk));
}

extern "C" void rocke_fwd_rr_mfma_grid(const rocke_fwd_rr_mfma_spec_t* s, int grid[3])
{
    const long long t = rocke_fwd_rr_mfma_tiles(s);
    grid[0] = (int)(t < kPersist ? t : kPersist);
    grid[1] = 1;
    grid[2] = 1;
}

extern "C" long long rocke_fwd_rr_weight_bytes(const rocke_direct_conv_problem_t* p)
{
    return mul_sat(
        mul_sat(mul_sat(mul_sat(2, rocke_direct_conv_problem_total_k(p)), 3 * (long long)p->KW),
                rocke_ceil_div(p->cpg, kCK)),
        kCK);
}

extern "C" long long rocke_fwd_rr_mfma_workspace_bytes(const rocke_fwd_rr_mfma_spec_t* s)
{
    return rocke_fwd_rr_weight_bytes(&s->problem);
}

extern "C" rocke_status_t
    rocke_fwd_rr_mfma_kernel_name(const rocke_fwd_rr_mfma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    const int pw = rocke_direct_conv_problem_pad_w(p);
    char f_buf[48], p_buf[48], k_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%d", p->KH, p->KW);
    if(pw == p->PAD)
    {
        snprintf(p_buf, sizeof p_buf, "p%d", p->PAD);
    }
    else
    {
        snprintf(p_buf, sizeof p_buf, "p%dx%d", p->PAD, pw);
    }
    k_buf[0] = '\0';
    if(s->kw16 != 4)
    {
        snprintf(k_buf, sizeof k_buf, "k%d", rocke_fwd_rr_mfma_tk(s));
    }
    const char* parts[3] = {f_buf, p_buf, k_buf};
    const char* flags[10]
        = {"oddc", "c64", "unf", "k4w", "ctail", "onec", "grp", "xcd", "chk", "bf16"};
    int on[10] = {s->odd_c,
                  s->c64,
                  s->unfold,
                  s->k4w,
                  s->c_tail,
                  s->one_c,
                  s->grp,
                  s->xcd,
                  s->chunk,
                  problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 3, flags, on, 10, out, out_cap, NULL);
}

extern "C" bool rocke_fwd_rr_mfma_is_valid_spec(const rocke_fwd_rr_mfma_spec_t* s,
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
    const long long Ho = rocke_direct_conv_problem_Ho(p);
    const long long Wo = rocke_direct_conv_problem_Wo(p);
    const int PW = rocke_direct_conv_problem_pad_w(p);
    const int TK = rocke_fwd_rr_mfma_tk(s);
    const char* ab = problem_is_bf16(p) ? "bf16" : "f16";
    const char* why = NULL;
    long long nchk = 0;
    if(target->wave_size != 64 || s->wave_size != 64)
    {
        why = "fwd_rr_mfma is a wave64 MFMA kernel";
    }
    else if(rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "fwd_rr_mfma needs the 16-byte LDS DMA (absent on %s)", arch);
        why = msg;
    }
    else if(!rocke_arch_fits_lds(target, (long)(2 * kBufB)))
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "LDS %d B > %d", 2 * kBufB, (int)target->lds_capacity_bytes);
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
    else if(p->groups != 1 && !s->grp)
    {
        why = "groups must be 1 (grp takes grouped rows)";
    }
    else if(s->grp && (p->groups < 2 || (TK != 0 && p->kpg % TK) || s->unfold))
    {
        why = "grp: groups >= 2, a whole number of K blocks per group, no unfold";
    }
    else if(p->KH != 3 || (p->KW != 3 && p->KW != 1) || p->stride != 1 || p->dil_h != 1
            || p->dil_w != 1)
    {
        why = "3x3 or 3x1, stride 1, dilation 1 only";
    }
    else if(p->pad_w < -1 || !(0 <= p->PAD && p->PAD <= 2 && 0 <= PW && PW <= p->KW - 1))
    {
        ROCKE_ERR_SNPRINTF(msg,
                           sizeof msg,
                           "padding must be 0..2 (rows) and 0..KW-1 (columns) (got %dx%d)",
                           p->PAD,
                           PW);
        why = msg;
    }
    else if(s->c64 && p->KW == 1)
    {
        why = "c64 needs at least 2 filter columns";
    }
    else if(s->unfold && (p->KW != 1 || p->W != 1))
    {
        why = "unfold needs a width-1 layer with a 3x1 filter";
    }
    else if(p->KW == 1 && !s->unfold && Wo <= 8)
    {
        ROCKE_ERR_SNPRINTF(msg,
                           sizeof msg,
                           "output width %lld wastes most of the 32-column tile (unfold runs "
                           "width 1)",
                           Wo);
        why = msg;
    }
    else if(s->unfold && mul_sat(mul_sat(64, p->H), tc) > kOob)
    {
        why = "unfold: 32 images exceed the i32 buffer range";
    }
    if(why == NULL)
    {
        if(s->one_c)
        {
            if(p->cpg > kCK || p->cpg % 8)
            {
                ROCKE_ERR_SNPRINTF(msg, sizeof msg, "one_c needs C <= %d, a multiple of 8", kCK);
                why = msg;
            }
            else if((p->cpg < kCK) != s->c_tail || s->odd_c)
            {
                why = "one_c: c_tail exactly when C < 64, no odd_c";
            }
            nchk = 1;
        }
        else if(s->c_tail)
        {
            if(p->cpg % 8 || p->cpg % kCK == 0)
            {
                ROCKE_ERR_SNPRINTF(
                    msg, sizeof msg, "c_tail needs C a multiple of 8 but not of %d", kCK);
                why = msg;
            }
            nchk = rocke_ceil_div(p->cpg, kCK);
        }
        else if(p->cpg % kCK)
        {
            ROCKE_ERR_SNPRINTF(msg,
                               sizeof msg,
                               "C %d must be a multiple of %d (c_tail takes the rest)",
                               p->cpg,
                               kCK);
            why = msg;
        }
        else
        {
            nchk = p->cpg / kCK;
        }
    }
    if(why == NULL)
    {
        if(s->one_c)
        {
        }
        else if(s->odd_c)
        {
            if(nchk % 2 == 0 || nchk < 3)
            {
                ROCKE_ERR_SNPRINTF(
                    msg, sizeof msg, "odd_c needs an odd number (>= 3) of %d-channel chunks", kCK);
                why = msg;
            }
        }
        else if(nchk % 2)
        {
            ROCKE_ERR_SNPRINTF(
                msg, sizeof msg, "an odd number of %d-channel chunks needs odd_c", kCK);
            why = msg;
        }
    }
    if(why == NULL && s->kw16 != 4 && s->kw16 != 3 && s->kw16 != 2)
    {
        why = "kw16 must be one of (4, 3, 2)";
    }
    if(why == NULL && tk % TK)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "K %lld must be a multiple of %d", tk, TK);
        why = msg;
    }
    if(why == NULL)
    {
        const struct
        {
            const char* what;
            long long bytes;
        } tensors[] = {
            {"input image", mul_sat(mul_sat(mul_sat(2, p->H), p->W), tc)},
            {"output image", mul_sat(mul_sat(mul_sat(2, Ho), Wo), tk)},
            {"weights", rocke_fwd_rr_weight_bytes(p)},
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
    if(why == NULL && rocke_fwd_rr_mfma_tiles(s) > kOob)
    {
        why = "tile count exceeds i32";
    }
    if(why == NULL && s->chunk)
    {
        if(!s->k4w || p->KW != 3 || s->unfold || s->grp)
        {
            why = "chunk: k4w 3x3 tiles only (no unfold / grp)";
        }
        else if(mul_sat(mul_sat(mul_sat(mul_sat(2, p->N), p->H), p->W), tc) > kOob
                || mul_sat(mul_sat(mul_sat(mul_sat(2, p->N), Ho), Wo), tk) > kOob)
        {
            why = "chunk: the whole input / output must fit the i32 buffer range";
        }
    }
    if(why == NULL && s->xcd && strcmp(arch, "gfx950") != 0)
    {
        why = "xcd: the XCD-local walk is built for ['gfx950'] only";
    }
    if(why != NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, why);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  FwdRrWfmtSpec
 * ===================================================================== */

extern "C" rocke_fwd_rr_wfmt_spec_t rocke_fwd_rr_wfmt_spec_default(void)
{
    rocke_fwd_rr_wfmt_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "fwd_rr_wfmt";
    s.threads = 256;
    s.kw16 = 4;
    s.fw = 3;
    s.nwk = 2;
    return s;
}

extern "C" rocke_status_t
    rocke_fwd_rr_wfmt_kernel_name(const rocke_fwd_rr_wfmt_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    char k_buf[24], w_buf[24], wk_buf[24];
    k_buf[0] = w_buf[0] = wk_buf[0] = '\0';
    if(s->kw16 != 4)
    {
        snprintf(k_buf, sizeof k_buf, "k%d", 32 * s->kw16);
    }
    if(s->fw != 3)
    {
        snprintf(w_buf, sizeof w_buf, "w%d", s->fw);
    }
    if(s->nwk != 2)
    {
        snprintf(wk_buf, sizeof wk_buf, "wk%d", s->nwk);
    }
    const char* parts[4] = {k_buf, w_buf, wk_buf, s->dg ? "dg" : ""};
    const char* flags[1] = {"bf16"};
    int on[1] = {problem_is_bf16(&s->problem) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 4, flags, on, 1, out, out_cap, NULL);
}

extern "C" void rocke_fwd_rr_wfmt_grid(const rocke_fwd_rr_wfmt_spec_t* s, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    const long long granules = rocke_floor_div(
        mul_sat(mul_sat(mul_sat(mul_sat(rocke_direct_conv_problem_total_k(p), 3), s->fw),
                        rocke_ceil_div(p->cpg, kCK)),
                kCK),
        8);
    grid[0] = s->threads > 0 ? (int)rocke_ceil_div(granules, s->threads) : 0;
    grid[1] = 1;
    grid[2] = 1;
}

namespace
{

/* A pointer kernarg of the helper: noalias, readonly (A) / writeonly (D),
 * 16-byte aligned. */
V wfmt_ptr(rocke_ir_builder_t* b, const char* name, const rocke_type_t* io_type, bool write)
{
    rocke_param_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    if(write)
    {
        opts.writeonly = true;
        opts.writeonly_set = true;
    }
    else
    {
        opts.readonly = true;
        opts.readonly_set = true;
    }
    opts.align = 16;
    opts.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, io_type, "global"), &opts);
}

rocke_kernel_def_t* build_wfmt(rocke_ir_builder_t* b, const rocke_fwd_rr_wfmt_spec_t* spec)
{
    Em e{b};
    const bool bf16 = problem_is_bf16(&spec->problem);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->threads);
    V src = wfmt_ptr(b, "A", io_type, false);
    V dst = wfmt_ptr(b, "D", io_type, true);
    V src_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), NULL);
    V dst_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), NULL);
    V C = rocke_b_param(b, "p_cpg", rocke_i32(), NULL);
    V kpg = rocke_b_param(b, "p_kpg", rocke_i32(), NULL);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    V bid = rocke_b_block_id_x(b);
    V o0 = e.muli(bid, spec->threads);
    V o = e.add(o0, rocke_b_thread_id_x(b));
    V n_gran = e.divi(dst_bytes, 16);
    V rest = o;
    V lane = e.modi(rest, 64);
    rest = e.divi(rest, 64);
    const int KW = spec->kw16;
    V k4 = e.modi(rest, KW);
    rest = e.divi(rest, KW);
    V r = e.modi(rest, 3);
    rest = e.divi(rest, 3);
    const int NWKF = spec->nwk;
    V wk = e.modi(rest, NWKF);
    rest = e.divi(rest, NWKF);
    const int FW = spec->fw;
    V st = e.modi(rest, 2 * FW);
    rest = e.divi(rest, 2 * FW);
    V nch = e.cdivi(C, kCK);
    V ch = e.mod(rest, nch);
    V kb = e.div(rest, nch);
    V s = e.divi(st, 2);
    V half = e.modi(st, 2);
    V m = e.modi(lane, 16);
    V g = e.divi(m, 4);
    V chan_a = e.muli(e.modi(g, 2), 8);
    V chan_b = e.muli(e.divi(g, 2), 4);
    V chan_ab = e.add(chan_a, chan_b);
    V chan = e.add(chan_ab, e.modi(m, 4));
    V k_a = e.muli(kb, 16 * NWKF * KW);
    V k_b = e.muli(wk, 16 * KW);
    V k_ab = e.add(k_a, k_b);
    V k_c = e.add(e.muli(k4, 16), chan);
    V k = e.add(k_ab, k_c);
    V c_a = e.muli(ch, kCK);
    V c_b = e.muli(half, 32);
    V c_ab = e.add(c_a, c_b);
    V c = e.add(c_ab, e.muli(e.divi(lane, 16), 8));
    V ok = e.lt(o, n_gran);
    V src_rsrc = rocke_b_buffer_rsrc(b, src, src_bytes);
    V dst_rsrc = rocke_b_buffer_rsrc(b, dst, dst_bytes);
    V oob = e.ci(kOob);
    V v = NULL;
    if(spec->dg)
    {
        V K = kpg;
        V k2 = e.ci(2);
        V rf = e.sub(k2, r);
        V kf = e.ci(FW - 1);
        V sf = e.sub(kf, s);
        std::vector<V> vals;
        for(int j = 0; j < 8; ++j)
        {
            V cj = e.addi(c, j);
            V t = e.add(e.muli(cj, 3), rf);
            t = e.add(e.muli(t, FW), sf);
            V el = e.add(e.mul(t, K), k);
            V cok = e.land(ok, e.lt(cj, C));
            V off = e.muli(el, 2);
            V sel = e.select(cok, off, oob);
            vals.push_back(rocke_dconv_buf_load_half(b, bf16, src_rsrc, sel, e.ci(0)));
        }
        v = rocke_b_vec_pack(b, vals.data(), (int)vals.size(), io_type);
    }
    else
    {
        V t = e.add(e.muli(k, 3), r);
        t = e.add(e.muli(t, FW), s);
        V src_el = e.add(e.mul(t, C), c);
        V c_ok = e.lt(c, C); /* zero-pad C to a whole chunk */
        V lok = e.land(ok, c_ok);
        V off = e.muli(src_el, 2);
        V sel = e.select(lok, off, oob);
        v = rocke_dconv_buf_load_vN(b, bf16, src_rsrc, sel, e.ci(0), 4);
    }
    V doff = e.muli(o, 16);
    V dsel = e.select(ok, doff, oob);
    rocke_dconv_buf_store_vN(b, bf16, dst_rsrc, dsel, e.ci(0), v, 4);
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_fwd_rr_wfmt(rocke_ir_builder_t* b,
                                                       const rocke_fwd_rr_wfmt_spec_t* spec,
                                                       const char* arch)
{
    (void)arch;
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_wfmt(b, spec);
}

extern "C" rocke_kernel_def_t* rocke_build_fwd_rr_wfmt_new(rocke_ir_builder_t* b,
                                                           const rocke_fwd_rr_wfmt_spec_t* spec,
                                                           const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_fwd_rr_wfmt",
        [&](char* out, size_t cap) { return rocke_fwd_rr_wfmt_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_fwd_rr_wfmt(b, spec, arch); });
}

extern "C" rocke_status_t rocke_fwd_rr_wfmt_lower_to_llvm(const rocke_fwd_rr_wfmt_spec_t* spec,
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
        "build_fwd_rr_wfmt failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_fwd_rr_wfmt_new(b, spec, arch); });
}

/* ===================================================================== *
 *  build_fwd_rr_mfma
 * ===================================================================== */
namespace
{

/* decode(v): the tile walk's per-tile scalars. Entries a variant does not
 * produce stay NULL. */
struct Ctx
{
    V kb = NULL, w0 = NULL, h0 = NULL, n = NULL, valid = NULL;
    V gc = NULL; /* grp */
    V ncol = NULL; /* unfold */
    V sn[4] = {NULL, NULL, NULL, NULL}; /* chunk */
    V sq[4] = {NULL, NULL, NULL, NULL};
    V sv[4] = {NULL, NULL, NULL, NULL};
};

/* dma_scalars(): opaque per-step copies. */
struct Sq
{
    V h0, col, row, lds;
};

using W3 = std::vector<std::vector<V>>; /* [r 3][k kw16] weight tiles */

rocke_kernel_def_t*
    build_rr(rocke_ir_builder_t* b, const rocke_fwd_rr_mfma_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_fwd_rr_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid FwdRrMfmaSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const int KW = spec->kw16, TK = rocke_fwd_rr_mfma_tk(spec);
    const int PH = p.PAD, PW = rocke_direct_conv_problem_pad_w(&p);
    const bool C64 = spec->c64;
    const int FW = p.KW; /* filter width: 3, or 1 (3x1) */
    const bool UNF = spec->unfold;
    const bool CHK = spec->chunk;
    const bool GRP = spec->grp;
    /* staged columns (chunk: 4 slots x 8 columns, then their 4 x 2 halo columns) */
    const int TC = CHK ? 4 * 10 : kTQ + FW - 1;
    const int SIDE_B = CHK ? 1024 : 256; /* bytes per side DMA */
    const int NWK = rocke_fwd_rr_mfma_nwk(spec), TPV = rocke_fwd_rr_mfma_tp(spec);
    const int TR = TPV + 2; /* staged rows */
    const int DMA_MAIN = cdiv(TR, 2);
    const int DMA_N = DMA_MAIN + (FW == 3 ? cdiv(TR, 8) : 0);
    const int FSTEPS = 2 * FW;
    const int NSTEP = C64 ? FW : FSTEPS;
    const int DPS = cdiv(DMA_N, NSTEP - 1 > 1 ? NSTEP - 1 : 1);
    const int STEP_B = step_b(KW, NWK);
    const int CHUNK_B = FSTEPS * STEP_B;
    const int STORES = stores_of(KW);
    const bool SLIM = kRrSlim && !spec->unfold && !spec->odd_c;
    const bool WIMM = kRrWimm && !spec->unfold && !spec->odd_c;
    const int ST_AUX = KW == 3 ? 0 : kStNt;
    const bool ODD = spec->odd_c;
    const bool ONE = spec->one_c;
    const rocke_type_t* i32_t = rocke_i32();

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", 8 * spec->wave_size);
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
    V sa_n = P.p_A_stride_n;
    V sd_n = P.p_D_stride_n, sd_h = P.p_D_stride_ho, sd_w = P.p_D_stride_wo;

    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    V wp = e.mod(wave, c2);
    V wq = e.mod(e.div(wave, c2), c2);
    V grp = e.divi(wave, 4); /* ping-pong group */
    V wk = NULL;
    if(spec->k4w)
    {
        wp = c0;
        wq = e.mod(wave, c2);
        wk = e.div(wave, c2);
    }
    else
    {
        wk = grp;
    }

    auto opaque = [&](V v, const char* cons) -> V {
        rocke_inline_asm_opts_t opts = {};
        opts.sideeffect = true;
        opts.sideeffect_set = true;
        rocke_op_t* op = rocke_b_inline_asm(b, "", cons, &v, 1, &i32_t, 1, &opts);
        return op ? op->results[0] : NULL;
    };
    auto opaque_lane = [&]() { return opaque(lane, "=v,0"); };
    auto opaque_s = [&](V v) { return opaque(v, "=s,0"); };

    /* ---- tile walk: t -> (k block fastest, pixel tile, row tile, image) ---- */
    V nkb = e.divi(K, TK);
    V n_ht = e.cdivi(p_Ho, TPV);
    V n_wt = NULL, T = NULL, nqc = NULL, nsl = NULL;
    if(UNF)
    {
        n_wt = e.cdivi(p_N, kTQ);
        T = e.mul(e.mul(n_ht, n_wt), nkb);
    }
    else if(CHK)
    {
        nqc = e.cdivi(p_Wo, 8);
        nsl = e.mul(p_N, nqc);
        n_wt = e.cdivi(nsl, 4);
        T = e.mul(e.mul(n_ht, n_wt), nkb);
    }
    else
    {
        n_wt = e.cdivi(p_Wo, kTQ);
        T = e.mul(e.mul(e.mul(p_N, n_ht), n_wt), nkb);
    }
    V CPG = GRP ? e.div(C, P.p_groups) : C;
    V KPG = GRP ? e.div(K, P.p_groups) : K;
    V nch = e.cdivi(CPG, kCK);
    V blk = rocke_b_block_id_x(b);
    V rounds = e.divi(e.addi(e.sub(T, blk), kPersist - 1), kPersist);
    V img_a = e.mul(sa_n, c2);
    V img_d = e.mul(sd_n, c2);
    V a_all = CHK ? e.mul(p_N, img_a) : NULL;

    const int NX = kXcdGfx950;
    V x_per = NULL, x_rem = NULL;
    if(spec->xcd)
    {
        x_per = e.divi(T, NX);
        x_rem = e.modi(T, NX);
    }

    auto xcd_tile = [&](V v) {
        V x = e.land(v, e.ci(NX - 1));
        V j = e.divi(e.land(v, e.ci(kPersist - 1)), NX);
        V r = e.divi(v, kPersist);
        V xp = e.mul(x, x_per);
        V start = e.add(xp, e.smin(x, x_rem));
        return e.add(start, e.add(e.muli(r, kPersist / NX), j));
    };

    auto decode = [&](V v) {
        Ctx c;
        V t = spec->xcd ? xcd_tile(v) : v;
        c.kb = e.mod(t, nkb);
        V rest = e.div(t, nkb);
        c.w0 = e.muli(e.mod(rest, n_wt), kTQ);
        rest = e.div(rest, n_wt);
        c.h0 = e.muli(e.mod(rest, n_ht), TPV);
        c.n = e.div(rest, n_ht);
        c.valid = e.lt(v, T);
        if(GRP)
        {
            V g = e.div(e.muli(c.kb, TK), KPG);
            c.gc = e.mul(e.mul(g, CPG), c2);
            return c;
        }
        if(CHK)
        {
            V g0 = e.muli(e.divi(c.w0, kTQ), 4);
            V sl[4];
            for(int i = 0; i < 4; ++i)
            {
                sl[i] = e.addi(g0, i);
            }
            c.w0 = c0;
            c.n = c0;
            for(int i = 0; i < 4; ++i)
            {
                c.sn[i] = e.div(sl[i], nqc);
            }
            for(int i = 0; i < 4; ++i)
            {
                c.sq[i] = e.muli(e.mod(sl[i], nqc), 8);
            }
            for(int i = 0; i < 4; ++i)
            {
                c.sv[i] = e.lt(sl[i], nsl);
            }
            return c;
        }
        if(UNF)
        {
            V m = e.smini(e.sub(p_N, c.w0), kTQ);
            c.ncol = e.select(c.valid, m, c0);
            c.n = c.w0;
            c.w0 = c0;
            return c;
        }
        return c;
    };

    auto a_base = [&](const Ctx& ctx) {
        V n64 = e.zext64(ctx.n);
        V off = e.mul(n64, e.zext64(img_a));
        return rocke_b_global_ptr_offset(b, P.A, off);
    };

    auto w_tile_off = [&](const Ctx& ctx) {
        V a = e.muli(e.mul(ctx.kb, nch), FSTEPS * NWK * 3 * KW);
        V w = e.add(a, e.muli(wk, 3 * KW));
        return e.muli(w, kTileB);
    };

    /* ---- LDS ---- */
    const int elems = kBufB / 2;
    V bufs[2], addrs[2];
    for(int i = 0; i < 2; ++i)
    {
        std::string h = "in" + std::to_string(i);
        bufs[i] = e.smem_alloc_1d(io_type, elems, h.c_str());
    }
    for(int i = 0; i < 2; ++i)
    {
        addrs[i] = rocke_b_smem_addr_of(b, bufs[i]);
    }
    auto lds_at
        = [&](int i, V byte_off) { return rocke_b_smem_ptr_add(b, addrs[i], e.zext64(byte_off)); };

    /* ---- DMA ---- */
    auto sel4 = [&](const V* vals, V i) {
        V r = vals[0];
        for(int k = 1; k < 4; ++k)
        {
            V eqk = e.eqi(i, k);
            r = e.select(eqk, vals[k], r);
        }
        return r;
    };

    auto dma_lanes_chunk = [&](const Ctx& ctx, V& lt_out, V& ls_out) {
        V ln = opaque_lane();
        V l8 = e.divi(ln, 8);
        V gran = e.bxor(e.modi(ln, 8), l8);
        V sm = e.modi(wave, 4);
        V col = e.add(e.subi(sel4(ctx.sq, sm), PW), l8);
        V okv = sel4(ctx.sv, sm);
        V cge = e.ge(col, c0);
        V ok = e.land(okv, e.land(cge, e.lt(col, p_Wi)));
        V lt_a = e.mul(sel4(ctx.sn, sm), img_a);
        V lt_b = e.mul(col, C);
        V lt_c = e.add(lt_b, e.muli(gran, 8));
        V lt = e.add(lt_a, e.mul(lt_c, c2));
        V hs = e.div(l8, c2);
        V hq = e.subi(sel4(ctx.sq, hs), PW - 8);
        V hcol = e.add(hq, e.mod(l8, c2));
        V hv = sel4(ctx.sv, hs);
        V sok = e.land(hv, e.lt(hcol, p_Wi));
        V ls_a = e.mul(sel4(ctx.sn, hs), img_a);
        V ls_b = e.mul(hcol, C);
        V ls_c = e.add(ls_b, e.muli(gran, 8));
        V ls = e.add(ls_a, e.mul(ls_c, c2));
        lt_out = e.select(ok, lt, oob);
        ls_out = e.select(sok, ls, oob);
    };

    auto dma_lanes = [&](const Ctx& ctx, V& lt_out, V& ls_out) {
        if(CHK)
        {
            dma_lanes_chunk(ctx, lt_out, ls_out);
            return;
        }
        V ln = opaque_lane();
        V l8 = e.divi(ln, 8);
        V l32 = e.divi(ln, 32);
        V gt = rocke_b_cmp_gt(b, ctx.w0, c0);
        V kpw = e.ci(PW);
        V sh = e.select(gt, kpw, c0);
        V col = e.add(e.muli(e.modi(wave, 4), 8), l8);
        V gran = e.bxor(e.modi(ln, 8), l8);
        if(UNF)
        {
            V a = e.mul(col, img_a);
            V lt = e.add(a, e.muli(gran, 16));
            lt_out = lt;
            ls_out = lt;
            return;
        }
        V x1 = e.add(e.subi(ctx.w0, PW), col);
        V ge0 = e.ge(x1, c0);
        V x2 = e.add(e.subi(ctx.w0, PW), col);
        V ok = e.land(ge0, e.lt(x2, p_Wi));
        V t = e.add(e.subi(col, PW), sh);
        V tm = e.mul(t, C);
        V lt = e.mul(e.add(tm, e.muli(gran, 8)), c2);
        V k32 = e.ci(32);
        V scol = e.add(k32, l32);
        V sgran = e.bxor(e.divi(e.modi(ln, 32), 4), l32);
        V sx = e.add(e.subi(ctx.w0, PW), scol);
        V sok = e.lt(sx, p_Wi);
        V st = e.add(e.subi(scol, PW), sh);
        V stm = e.mul(st, C);
        V sb = e.mul(e.add(stm, e.muli(sgran, 8)), c2);
        V ls = e.add(sb, e.muli(e.modi(ln, 4), 4));
        lt_out = e.select(ok, lt, oob);
        ls_out = e.select(sok, ls, oob);
    };

    auto dma_scalars = [&](const Ctx& ctx, V ch) {
        V gt = rocke_b_cmp_gt(b, ctx.w0, c0);
        V kpw = e.ci(PW);
        V sh = e.select(gt, kpw, c0);
        V hs = e.subi(ctx.h0, PH);
        V far = e.ci(-(1 << 20));
        V h0 = e.select(ctx.valid, hs, far);
        V cb = e.mul(e.mul(e.sub(ctx.w0, sh), C), c2);
        V col_b = e.add(cb, e.muli(ch, kPXB));
        if(GRP)
        {
            col_b = e.add(col_b, ctx.gc);
        }
        Sq sq;
        sq.h0 = opaque_s(h0);
        sq.col = opaque_s(col_b);
        sq.row = opaque_s(e.mul(e.mul(p_Wi, C), c2));
        V la = e.muli(grp, TC * kPXB);
        V lb = e.muli(e.modi(wave, 4), 1024);
        sq.lds = opaque_s(e.add(la, lb));
        return sq;
    };

    auto c_tail_voff = [&](int u, V ch, V voff) {
        V ln = opaque_lane();
        V lch = NULL;
        if(u < DMA_MAIN || CHK)
        {
            V a = e.modi(ln, 8);
            lch = e.muli(e.bxor(a, e.divi(ln, 8)), 8);
        }
        else
        {
            V a = e.divi(e.modi(ln, 32), 4);
            V sg = e.bxor(a, e.divi(ln, 32));
            V t = e.muli(sg, 8);
            lch = e.add(t, e.mul(e.modi(ln, 4), c2));
        }
        V cok = e.lt(e.add(lch, e.muli(ch, kCK)), CPG);
        return e.select(cok, voff, oob);
    };

    /* the side DMA's LDS destination (real halo row, or a dummy slot) */
    auto side_dst = [&](V r, V real) {
        V a = e.addi(e.muli(r, TC * kPXB), 32 * kPXB);
        V k = e.ci((int64_t)TR * TC * kPXB);
        V m = e.muli(e.subi(r, TR), SIDE_B);
        V bb = e.add(k, m);
        return e.select(real, a, bb);
    };

    auto dma_u = [&](int u, const Ctx& ctx, V ch, int buf, V lt, V ls, const Sq* sq) {
        if(sq != NULL)
        {
            V h, dst, voff, rok;
            int dw;
            if(u < DMA_MAIN)
            {
                h = e.add(sq->h0, e.addi(grp, 2 * u));
                dst = e.addi(sq->lds, (int64_t)2 * u * TC * kPXB);
                voff = lt;
                dw = 4;
                rok = rocke_b_cmp_ult(b, h, p_Hi);
            }
            else
            {
                V r = e.addi(wave, 8 * (u - DMA_MAIN));
                V real = e.lti(r, TR);
                dst = side_dst(r, real);
                h = e.add(sq->h0, r);
                voff = ls;
                dw = CHK ? 4 : 1;
                rok = e.land(real, rocke_b_cmp_ult(b, h, p_Hi));
            }
            /* a row outside the image reads through a zero-size descriptor */
            V so = e.add(e.mul(h, sq->row), sq->col);
            if(spec->c_tail)
            {
                voff = c_tail_voff(u, ch, voff);
            }
            V rng = UNF ? e.mul(ctx.ncol, img_a) : (CHK ? a_all : img_a);
            V base = a_base(ctx);
            V rsrc = rocke_b_buffer_rsrc(b, base, e.select(rok, rng, c0));
            rocke_b_async_buffer_load_lds(b, rsrc, lds_at(buf, dst), voff, so, dw, 0);
            return;
        }
        V gt = rocke_b_cmp_gt(b, ctx.w0, c0);
        V kpw = e.ci(PW);
        V sh = e.select(gt, kpw, c0);
        V r, dst, voff, real = NULL;
        int dw;
        if(u < DMA_MAIN)
        {
            r = e.addi(grp, 2 * u); /* < TR: TR is even */
            V a = e.muli(r, TC * kPXB);
            dst = e.add(a, e.muli(e.modi(wave, 4), 1024));
            voff = lt;
            dw = 4;
        }
        else
        {
            r = e.addi(wave, 8 * (u - DMA_MAIN));
            real = e.lti(r, TR);
            dst = side_dst(r, real);
            voff = ls;
            dw = CHK ? 4 : 1;
        }
        V h = e.add(e.subi(ctx.h0, PH), r);
        V hge = e.ge(h, c0);
        V rok = e.land(ctx.valid, e.land(hge, e.lt(h, p_Hi)));
        if(u >= DMA_MAIN)
        {
            rok = e.land(rok, real);
        }
        V hw = e.mul(h, p_Wi);
        V row_a = e.add(hw, e.sub(ctx.w0, sh));
        V row = e.mul(row_a, e.mul(C, c2));
        V so_a = e.add(row, e.muli(ch, kPXB));
        V so = e.select(rok, so_a, c0);
        if(GRP)
        {
            so = e.add(so, ctx.gc);
        }
        if(spec->c_tail)
        {
            voff = c_tail_voff(u, ch, voff);
        }
        V rng = UNF ? e.mul(ctx.ncol, img_a) : (CHK ? a_all : img_a);
        V base = a_base(ctx);
        V rsrc = rocke_b_buffer_rsrc(b, base, e.select(rok, rng, c0));
        rocke_b_async_buffer_load_lds(b, rsrc, lds_at(buf, dst), voff, so, dw, 0);
    };

    /* ---- weights ---- */
    V w_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes);
    V lane16 = e.muli(lane, 16);

    auto load_w = [&](V soff) {
        W3 out(3, std::vector<V>());
        if(WIMM)
        {
            /* tile i at soff + (i // 4) * 4 KiB plus (i % 4) KiB in the immediate */
            std::vector<V> sg = {soff};
            for(int g = 1; g < cdiv(3 * KW, 4); ++g)
            {
                sg.push_back(e.addi(soff, (int64_t)g * 4096));
            }
            for(int r = 0; r < 3; ++r)
            {
                for(int k = 0; k < KW; ++k)
                {
                    V vo = e.addi(lane16, (int64_t)((r * KW + k) % 4) * kTileB);
                    out[r].push_back(
                        rocke_dconv_buf_load_vN(b, bf16, w_rsrc, vo, sg[(r * KW + k) / 4], 4));
                }
            }
            return out;
        }
        for(int r = 0; r < 3; ++r)
        {
            for(int k = 0; k < KW; ++k)
            {
                V so = e.addi(soff, (int64_t)(r * KW + k) * kTileB);
                out[r].push_back(rocke_dconv_buf_load_vN(b, bf16, w_rsrc, lane16, so, 4));
            }
        }
        return out;
    };

    /* ---- LDS fragment reads: row wp * 8 + t, column wq * 16 + s + lane % 16 ---- */
    auto read_in = [&](int buf, int s, int half) {
        V ln = opaque_lane();
        V l16 = e.modi(ln, 16);
        V lhi = e.divi(ln, 16);
        V col = e.add(e.addi(e.muli(wq, 16), s), l16);
        if(CHK && s > 0) /* past the slot's 8 columns: its halo columns */
        {
            V j8 = e.modi(l16, 8);
            V ha = e.muli(wq, 4);
            V ha2 = e.add(ha, e.mul(e.divi(l16, 8), c2));
            V hb = e.addi(j8, 32 + s - 8);
            V halo = e.add(ha2, hb);
            V lt8 = e.lti(j8, 8 - s);
            col = e.select(lt8, col, halo);
        }
        V hk = e.ci(half * 4);
        V hq = e.add(hk, lhi);
        V gran = e.bxor(hq, e.modi(col, 8));
        V ba = e.add(e.muli(wp, kWP * TC), col);
        V bc = e.muli(ba, kCK);
        V base = e.add(bc, e.muli(gran, 8));
        std::vector<V> out;
        for(int t = 0; t < kNIN; ++t)
        {
            V idx = e.addi(base, (int64_t)t * TC * kCK);
            out.push_back(e.smem_load(bufs[buf], c0, idx, io_type, 8));
        }
        return out;
    };

    /* ---- epilogue ---- */
    const rocke_type_t* i32x2 = rocke_vector_type(b, i32_t, 2);
    const rocke_type_t* half8 = rocke_vector_type(b, io_type, 8);
    V dcol = UNF ? sd_n : sd_w; /* output column stride */

    auto trunc = [&](V acc) { return rocke_dconv_trunc_f32(b, bf16, acc); };
    auto store = [&](V rsrc, V off, V val, int dwords) {
        rocke_dconv_buf_store_vN_ex(b, bf16, rsrc, off, c0, val, dwords, ST_AUX);
    };
    auto ncol_of = [&](const Ctx& ctx) { return UNF ? ctx.ncol : p_Wo; };
    auto row_ok = [&](const Ctx& ctx, int pr) -> V {
        if(!UNF)
        {
            return NULL;
        }
        V a = e.addi(e.muli(wp, kWP), pr);
        return e.lt(e.add(ctx.h0, a), p_Ho);
    };

    /* one permlane32_swap per dword pair: [lo0, lo1, hi0, hi1] */
    auto swap_pair = [&](V acc_a, V acc_b) {
        V da = rocke_b_vec_bitcast(b, trunc(acc_a), i32x2);
        V db = rocke_b_vec_bitcast(b, trunc(acc_b), i32x2);
        V lo0, hi0, lo1, hi1;
        V a0 = rocke_b_vec_extract(b, da, 0);
        rocke_b_permlane32_swap(b, a0, rocke_b_vec_extract(b, db, 0), &lo0, &hi0);
        V a1 = rocke_b_vec_extract(b, da, 1);
        rocke_b_permlane32_swap(b, a1, rocke_b_vec_extract(b, db, 1), &lo1, &hi1);
        return std::vector<V>{lo0, lo1, hi0, hi1};
    };
    auto pack8 = [&](const std::vector<V>& vals) {
        V pk = rocke_b_vec_pack(b, vals.data(), (int)vals.size(), i32_t);
        return rocke_b_vec_bitcast(b, pk, half8);
    };
    /* the row-shift DPP exchange between pixel halves */
    auto exchange
        = [&](const std::vector<std::vector<V>>& v, V hi, std::vector<V>& n0, std::vector<V>& n1) {
              for(int d = 0; d < 4; ++d)
              {
                  V r0 = rocke_b_mov_dpp(b, v[1][d], 8, -1, false);
                  V r1 = rocke_b_mov_dpp(b, v[0][d], -1, 8, false);
                  n0.push_back(e.select(hi, r0, v[0][d]));
                  n1.push_back(e.select(hi, v[1][d], r1));
              }
          };

    using StoreRow = std::function<void(std::vector<V>&, int)>;

    auto narrow_k = [&](const Ctx& ctx, V lhi, V& k_pair, V& k_one) {
        V kb_ = e.muli(ctx.kb, TK);
        V kbase = e.add(kb_, e.muli(wk, 16 * KW));
        V q1 = e.mod(lhi, c2);
        V q2 = e.div(lhi, c2);
        V pa = e.muli(q2, 16);
        V pb = e.add(pa, e.muli(q1, 8));
        k_pair = e.add(kbase, pb);
        V k32 = e.ci(32);
        V oa = e.muli(q1, 8);
        V ob = e.add(oa, e.muli(q2, 4));
        V oc = e.add(k32, ob);
        k_one = e.add(kbase, oc);
    };

    auto epilogue_narrow = [&](const Ctx& ctx, V l16, V lhi, V d_rsrc) -> StoreRow {
        V ca = e.add(ctx.w0, e.muli(wq, 16));
        V col = e.add(ca, l16);
        V ok_c = e.lt(col, ncol_of(ctx));
        V ra = e.add(ctx.h0, e.muli(wp, kWP));
        V rb = e.mul(ra, sd_h);
        V pix0 = e.add(rb, e.mul(col, dcol));
        V k_pair, k_one;
        narrow_k(ctx, lhi, k_pair, k_one);
        return [=, &e, &ctx](std::vector<V>& accs, int pr) {
            V row = e.add(pix0, e.muli(sd_h, pr));
            V rok = row_ok(ctx, pr);
            V ok = rok == NULL ? ok_c : e.land(ok_c, rok);
            V val = pack8(swap_pair(accs[pr * KW], accs[pr * KW + 1]));
            V off = e.mul(e.add(row, k_pair), c2);
            store(d_rsrc, e.select(ok, off, oob), val, 4);
            if(KW == 3)
            {
                V off1 = e.mul(e.add(row, k_one), c2);
                V sel = e.select(ok, off1, oob);
                store(d_rsrc, sel, trunc(accs[pr * KW + 2]), 2);
            }
        };
    };

    auto gran_of = [&](V l16, V lhi, V& hi, V& gran) {
        hi = e.gei(l16, 8);
        V ga = e.mul(e.div(lhi, c2), c2);
        V gran0 = e.add(ga, e.mod(lhi, c2));
        V k4 = e.ci(4);
        gran = e.add(gran0, e.select(hi, k4, c0));
    };

    auto epilogue_chunk = [&](const Ctx& ctx) -> StoreRow {
        V ln = opaque_lane();
        V l16 = e.modi(ln, 16);
        V lhi = e.divi(ln, 16);
        V j8 = e.modi(l16, 8);
        V d_rsrc = rocke_b_buffer_rsrc(b, P.D, e.mul(p_N, img_d));
        V sa = e.mul(wq, c2);
        V sb = e.add(e.mul(wq, c2), c1);
        auto pix = [&](V si, V sel_i, V sel_c, V& off_out, V& ok_out) {
            V n_ = sel4(ctx.sn, si);
            V q_ = sel4(ctx.sq, si);
            V v_ = sel4(ctx.sv, si);
            if(sel_i != NULL)
            {
                V n2 = sel4(ctx.sn, sel_i);
                V q2_ = sel4(ctx.sq, sel_i);
                V v2 = sel4(ctx.sv, sel_i);
                n_ = e.select(sel_c, n2, n_);
                q_ = e.select(sel_c, q2_, q_);
                v_ = e.select(sel_c, v2, v_);
            }
            V col = e.add(q_, j8);
            ok_out = e.land(v_, e.lt(col, p_Wo));
            V oa = e.mul(n_, sd_n);
            V oa2 = e.add(oa, e.mul(ctx.h0, sd_h));
            off_out = e.add(oa2, e.mul(col, sd_w));
        };
        auto rowok = [&, h0 = ctx.h0](int pr) { return e.lt(e.addi(h0, pr), p_Ho); };
        if(KW < 4)
        {
            V hi8 = e.gei(l16, 8);
            V pix0, ok_c;
            pix(sa, sb, hi8, pix0, ok_c);
            V k_pair, k_one;
            narrow_k(ctx, lhi, k_pair, k_one);
            return [=, &e](std::vector<V>& accs, int pr) {
                V row = e.add(pix0, e.muli(sd_h, pr));
                V ok = e.land(ok_c, rowok(pr));
                V val = pack8(swap_pair(accs[pr * KW], accs[pr * KW + 1]));
                V off = e.mul(e.add(row, k_pair), c2);
                store(d_rsrc, e.select(ok, off, oob), val, 4);
                if(KW == 3)
                {
                    V off1 = e.mul(e.add(row, k_one), c2);
                    V sel = e.select(ok, off1, oob);
                    store(d_rsrc, sel, trunc(accs[pr * KW + 2]), 2);
                }
            };
        }
        V hi, gran;
        gran_of(l16, lhi, hi, gran);
        V kt = e.muli(ctx.kb, TK);
        V ka = e.add(kt, e.muli(wk, 64));
        V kch = e.add(ka, e.muli(gran, 8));
        V pa, oka, pb, okb;
        pix(sa, NULL, NULL, pa, oka);
        pix(sb, NULL, NULL, pb, okb);
        V va0 = e.mul(e.add(pa, kch), c2);
        V vb0 = e.mul(e.add(pb, kch), c2);
        return [=, &e](std::vector<V>& accs, int pr) {
            std::vector<std::vector<V>> v;
            for(int cc = 0; cc < 2; ++cc)
            {
                v.push_back(swap_pair(accs[pr * KW + 2 * cc], accs[pr * KW + 2 * cc + 1]));
            }
            std::vector<V> n0, n1;
            exchange(v, hi, n0, n1);
            V dr = e.muli(e.mul(sd_h, c2), pr);
            V rok = rowok(pr);
            V offa = e.add(va0, dr);
            V offb = e.add(vb0, dr);
            const std::vector<V>* vals[2] = {&n0, &n1};
            V offs[2] = {offa, offb};
            V oks[2] = {oka, okb};
            for(int i = 0; i < 2; ++i)
            {
                V val = pack8(*vals[i]);
                V ok = e.land(oks[i], rok);
                store(d_rsrc, e.select(ok, offs[i], oob), val, 4);
            }
        };
    };

    auto epilogue = [&](const Ctx& ctx) -> StoreRow {
        if(CHK)
        {
            return epilogue_chunk(ctx);
        }
        V ln = opaque_lane();
        V l16 = e.modi(ln, 16);
        V lhi = e.divi(ln, 16);
        /* lane (px, g) stores granule 2 * (g >> 1) + (g & 1) (+ 4 for px >= 8) */
        V hi, gran;
        gran_of(l16, lhi, hi, gran);
        V n64 = e.zext64(ctx.n);
        V d_base = rocke_b_global_ptr_offset(b, P.D, e.mul(n64, e.zext64(img_d)));
        V d_rsrc = rocke_b_buffer_rsrc(b, d_base, UNF ? e.mul(ctx.ncol, img_d) : img_d);
        V ca = e.add(ctx.w0, e.muli(wq, 16));
        V col0 = e.add(ca, e.modi(l16, 8));
        V ra = e.add(ctx.h0, e.muli(wp, kWP));
        V rb = e.mul(ra, sd_h);
        V pix0 = e.add(rb, e.mul(col0, dcol));
        V ok0 = e.lt(col0, ncol_of(ctx));
        V ok1 = e.lt(e.addi(col0, 8), ncol_of(ctx));
        if(KW < 4)
        {
            return epilogue_narrow(ctx, l16, lhi, d_rsrc);
        }
        V kt = e.muli(ctx.kb, TK);
        V ka = e.add(kt, e.muli(wk, 64));
        V kch = e.add(ka, e.muli(gran, 8));
        V voff0 = e.mul(e.add(pix0, kch), c2);
        V step8 = e.mul(e.muli(dcol, 8), c2);
        return [=, &e, &ctx](std::vector<V>& accs, int pr) {
            std::vector<std::vector<V>> v;
            for(int cc = 0; cc < 2; ++cc)
            {
                v.push_back(swap_pair(accs[pr * KW + 2 * cc], accs[pr * KW + 2 * cc + 1]));
            }
            std::vector<V> n0, n1;
            exchange(v, hi, n0, n1);
            V voff = e.add(voff0, e.muli(e.mul(sd_h, c2), pr));
            V rok = row_ok(ctx, pr);
            V offb = e.add(voff, step8);
            const std::vector<V>* vals[2] = {&n0, &n1};
            V offs[2] = {voff, offb};
            V oks[2] = {ok0, ok1};
            for(int i = 0; i < 2; ++i)
            {
                V ok = oks[i];
                if(rok != NULL)
                {
                    ok = e.land(ok, rok);
                }
                V val = pack8(*vals[i]);
                store(d_rsrc, e.select(ok, offs[i], oob), val, 4);
            }
        };
    };

    /* ---- one 64-channel chunk ---- */
    /* wpre: the tile's first weights ([0] and, for c64, [1]); epi: the
     * tile's last chunk (epilogue interleaved into the last step). */
    auto chunk = [&](std::vector<V> accs,
                     const std::vector<W3>* wpre,
                     int cur,
                     const Ctx& dctx,
                     V dch,
                     V lt,
                     V ls,
                     V wsoff,
                     bool first,
                     const Ctx* epi) {
        const int NH = C64 ? 2 : 1;
        for(int j = 0; j < NSTEP; ++j)
        {
            if(j > 0)
            {
                rocke_b_s_setprio(b, 0);
                rocke_b_s_barrier_bare(b);
                rocke_b_sched_barrier(b, 0);
            }
            W3 wf[2];
            std::vector<V> fr[2];
            for(int hv = 0; hv < NH; ++hv)
            {
                const int s = C64 ? j : j / 2;
                const int half = C64 ? hv : j % 2;
                const int ws = 2 * s + half;
                if(first && ws == 0)
                {
                    wf[hv] = (*wpre)[0];
                }
                else if(first && C64 && ws == 1)
                {
                    wf[hv] = (*wpre)[1];
                }
                else
                {
                    wf[hv] = load_w(e.addi(wsoff, (int64_t)ws * STEP_B));
                }
            }
            int nd = DMA_N - j * DPS;
            nd = nd < DPS ? nd : DPS;
            nd = nd > 0 ? nd : 0;
            Sq sq;
            const bool use_sq = SLIM && nd;
            if(use_sq)
            {
                sq = dma_scalars(dctx, dch);
            }
            for(int d = 0; d < nd; ++d)
            {
                dma_u(j * DPS + d, dctx, dch, 1 - cur, lt, ls, use_sq ? &sq : NULL);
            }
            for(int hv = 0; hv < NH; ++hv)
            {
                const int s = C64 ? j : j / 2;
                const int half = C64 ? hv : j % 2;
                fr[hv] = read_in(cur, s, half);
            }
            rocke_b_s_waitcnt(b, nd, 0, -1);
            rocke_b_sched_barrier(b, 0);
            rocke_b_s_setprio(b, 1);
            rocke_b_s_barrier_bare(b);
            rocke_b_sched_barrier(b, 0);
            if(epi != NULL && j == NSTEP - 1)
            {
                /* rows finish one by one; row pr - 1's epilogue rides on row
                 * pr's MFMAs (fenced) */
                StoreRow store_row = epilogue(*epi);
                for(int pr = 0; pr < kWP; ++pr)
                {
                    for(int hv = 0; hv < NH; ++hv)
                    {
                        for(int r = 0; r < 3; ++r)
                        {
                            for(int k = 0; k < KW; ++k)
                            {
                                const int i = pr * KW + k;
                                accs[i]
                                    = e.mfma16x16x32(bf16, wf[hv][r][k], fr[hv][pr + r], accs[i]);
                            }
                        }
                    }
                    if(pr > 0)
                    {
                        store_row(accs, pr - 1);
                        rocke_b_sched_barrier(b, 0);
                    }
                }
                store_row(accs, kWP - 1);
                continue;
            }
            for(int hv = 0; hv < NH; ++hv)
            {
                for(int r = 0; r < 3; ++r)
                {
                    for(int k = 0; k < KW; ++k)
                    {
                        for(int pr = 0; pr < kWP; ++pr)
                        {
                            const int i = pr * KW + k;
                            accs[i] = e.mfma16x16x32(bf16, wf[hv][r][k], fr[hv][pr + r], accs[i]);
                        }
                    }
                }
            }
        }
        rocke_b_sched_barrier(b, 0);
        rocke_b_s_waitcnt(b, epi != NULL ? STORES : 0, -1, -1);
        rocke_b_s_setprio(b, 0);
        rocke_b_s_barrier_bare(b);
        rocke_b_sched_barrier(b, 0);
        return accs;
    };

    auto flat = [&](const W3& w, std::vector<V>& out) {
        for(int r = 0; r < 3; ++r)
        {
            for(int k = 0; k < KW; ++k)
            {
                out.push_back(w[r][k]);
            }
        }
    };

    /* ---- prologue ---- */
    Ctx ctx0 = decode(blk);
    V lt0, ls0;
    dma_lanes(ctx0, lt0, ls0);
    for(int u = 0; u < DMA_N; ++u)
    {
        dma_u(u, ctx0, c0, 0, lt0, ls0, NULL);
    }
    W3 w0 = load_w(w_tile_off(ctx0));
    W3 w0b;
    if(C64)
    {
        w0b = load_w(e.addi(w_tile_off(ctx0), STEP_B));
    }
    rocke_b_s_waitcnt(b, 0, -1, -1);
    rocke_b_s_barrier_bare(b);
    e.if_(e.eq(grp, c1),
          [&]() { rocke_b_s_barrier_bare(b); }); /* group 1 runs one barrier behind */
    rocke_b_sched_barrier(b, 0);

    V zero = rocke_b_zero_vec_f32(b, 4);
    std::vector<V> w_flat;
    flat(w0, w_flat);
    if(C64)
    {
        flat(w0b, w_flat);
    }
    const int NW = (int)w_flat.size();

    /* the next tile's first weights */
    auto next_weights = [&](const Ctx& cn) {
        std::vector<V> nxt;
        W3 wn = load_w(w_tile_off(cn));
        flat(wn, nxt);
        if(C64)
        {
            wn = load_w(e.addi(w_tile_off(cn), STEP_B));
            flat(wn, nxt);
        }
        rocke_b_sched_barrier(b, 0);
        return nxt;
    };

    /* One tile starting on LDS buffer p; returns the next tile's carried
     * state (first weights, DMA lane offsets). */
    auto tile_body = [&](V t, const std::vector<V>& carried, int p) {
        auto unflat = [&](int off) {
            W3 w(3, std::vector<V>());
            for(int r = 0; r < 3; ++r)
            {
                for(int k = 0; k < KW; ++k)
                {
                    w[r].push_back(carried[off + r * KW + k]);
                }
            }
            return w;
        };
        std::vector<W3> wpre = {unflat(0)};
        if(C64)
        {
            wpre.push_back(unflat(3 * KW));
        }
        V lt = carried[NW], ls = carried[NW + 1];
        Ctx ctx = decode(t);
        Ctx cn = decode(e.addi(t, kPersist));
        V wsoff = w_tile_off(ctx);
        std::vector<V> accs(kWP * KW, zero);
        if(ONE) /* the tile's only chunk DMAs the next tile's into buffer 1 - p */
        {
            V lt_n, ls_n;
            dma_lanes(cn, lt_n, ls_n);
            chunk(accs, &wpre, p, cn, c0, lt_n, ls_n, wsoff, true, &ctx);
            std::vector<V> nxt = next_weights(cn);
            nxt.push_back(lt_n);
            nxt.push_back(ls_n);
            return nxt;
        }
        accs = chunk(accs, &wpre, p, ctx, c1, lt, ls, wsoff, true, NULL);
        V mids = e.div(e.sub(nch, ODD ? e.ci(3) : c2), c2);
        accs = e.for_iter(c0,
                          mids,
                          c1,
                          iter_names("acc", accs.size()),
                          accs,
                          "pair",
                          [&](V pi, std::vector<V> macc) {
                              V ch = e.add(c1, e.mul(pi, c2));
                              V dch = e.add(ch, c1);
                              V wso = e.add(wsoff, e.muli(ch, CHUNK_B));
                              std::vector<V> m
                                  = chunk(macc, NULL, 1 - p, ctx, dch, lt, ls, wso, false, NULL);
                              V dch2 = e.add(ch, c2);
                              V wso2 = e.add(wsoff, e.muli(e.add(ch, c1), CHUNK_B));
                              return chunk(m, NULL, p, ctx, dch2, lt, ls, wso2, false, NULL);
                          });
        if(ODD) /* one more chunk: the last one then lands on buffer p */
        {
            V dch = e.sub(nch, c1);
            V wso = e.add(wsoff, e.muli(e.sub(nch, c2), CHUNK_B));
            accs = chunk(accs, NULL, 1 - p, ctx, dch, lt, ls, wso, false, NULL);
        }
        V lt_n, ls_n;
        dma_lanes(cn, lt_n, ls_n);
        V last = e.sub(nch, c1);
        V wso = e.add(wsoff, e.muli(last, CHUNK_B));
        chunk(accs, NULL, ODD ? p : 1 - p, cn, c0, lt_n, ls_n, wso, false, &ctx);
        /* the next tile's first weights (its first wait also drains the stores) */
        std::vector<V> nxt = next_weights(cn);
        nxt.push_back(lt_n);
        nxt.push_back(ls_n);
        return nxt;
    };

    std::vector<std::string> init_names = iter_names("wp", w_flat.size());
    init_names.push_back("lt");
    init_names.push_back("ls");
    std::vector<V> init = w_flat;
    init.push_back(lt0);
    init.push_back(ls0);
    if(!(ODD || ONE))
    {
        e.for_iter(c0, rounds, c1, init_names, init, "round", [&](V rnd, std::vector<V> carried) {
            V t = e.add(blk, e.muli(rnd, kPersist));
            return tile_body(t, carried, 0);
        });
    }
    else
    {
        /* An odd chunk count flips the starting buffer every tile: two tiles
         * per iteration (buffers 0 then 1), the second one guarded. */
        V twos = e.div(e.add(rounds, c1), c2);
        e.for_iter(c0, twos, c1, init_names, init, "round2", [&](V r2, std::vector<V> carried) {
            V rnd = e.mul(r2, c2);
            V t = e.add(blk, e.muli(rnd, kPersist));
            std::vector<V> nxt = tile_body(t, carried, 0);
            std::vector<const rocke_type_t*> types;
            for(V v : nxt)
            {
                types.push_back(v->type);
            }
            V cond = e.lt(e.add(rnd, c1), rounds);
            rocke_if_else_t ife
                = rocke_b_scf_if_else_results(b, cond, types.data(), (int)types.size());
            if(ife.op == NULL || ife.then_region == NULL || ife.else_region == NULL)
            {
                return std::vector<V>();
            }
            rocke_b_region_enter(b, ife.then_region);
            std::vector<V> tv = tile_body(e.addi(t, kPersist), nxt, 1);
            rocke_b_scf_yield(b, tv.data(), (int)tv.size());
            rocke_b_region_leave(b);
            rocke_b_region_enter(b, ife.else_region);
            rocke_b_scf_yield(b, nxt.data(), (int)nxt.size());
            rocke_b_region_leave(b);
            return std::vector<V>(ife.op->results, ife.op->results + ife.op->num_results);
        });
    }
    rocke_b_s_setprio(b, 0);
    e.if_(e.eq(grp, c0), [&]() { rocke_b_s_barrier_bare(b); }); /* balance the barrier count */
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_fwd_rr_mfma(rocke_ir_builder_t* b,
                                                       const rocke_fwd_rr_mfma_spec_t* spec,
                                                       const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_rr(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_fwd_rr_mfma_new(rocke_ir_builder_t* b,
                                                           const rocke_fwd_rr_mfma_spec_t* spec,
                                                           const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_fwd_rr_mfma",
        [&](char* out, size_t cap) { return rocke_fwd_rr_mfma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_fwd_rr_mfma(b, spec, arch); });
}

extern "C" rocke_status_t rocke_fwd_rr_mfma_lower_to_llvm(const rocke_fwd_rr_mfma_spec_t* spec,
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
        "build_fwd_rr_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_fwd_rr_mfma_new(b, spec, arch); });
}
