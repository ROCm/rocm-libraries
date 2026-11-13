// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_winograd_fused_1d_wmma.cpp -- C++ port of build_fused_wino_1d
 * (library/kernels/common/conv_winograd_fused_1d_wmma.py): the fused 1-D
 * Winograd forward conv for gfx1250. See the header and the Python module for
 * the algorithm.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_winograd_fused_1d_wmma.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, op_for_shape */
#include "rocke/instance_conv_winograd_internal.h" /* toom_cook, load_raw, raw_to_f32 */
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

/* conv_winograd_fused_1d_wmma module constants. */
#define FW_WAVE 32
#define FW_VEC 8
#define FW_PAD_H 8
#define FW_PAD_F 4
#define FW_VGPR_BUDGET 440

namespace
{

using rocke_wino::ToomCook;

int64_t fw_cdiv(int64_t a, int64_t b)
{
    /* Python -(-a // b) (floor division). */
    int64_t q = (-a) / b;
    if(((-a) % b != 0) && (((-a) < 0) != (b < 0)))
        --q;
    return -q;
}

int64_t fw_floordiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

void fw_set(char* out, size_t cap, const char* msg)
{
    if(out && cap > 0)
        snprintf(out, cap, "%s", msg);
}

bool fw_is_wmma(const rocke_winograd_fused_1d_spec_t* s)
{
    return s->xf != NULL && strcmp(s->xf, "wmma") == 0;
}

/* Geometry of FusedWino1DSpec (the Python properties). */
struct Geo
{
    int m, n, S, r1;
    int64_t tb, kb, wo, sub_w, tiles_w, rows, bpr, tile_blocks, k_blocks, vt, raw_rows;
    int64_t xf_tiles, xf_win, xf_groups, raw_alloc_rows;
};

Geo fw_geo(const rocke_winograd_fused_1d_spec_t* s)
{
    Geo g;
    const rocke_conv_problem_t* p = &s->problem;
    g.m = s->ax_m;
    g.S = s->ax_seg;
    g.r1 = s->ax_seg > 1 ? s->ax_m : s->ax_r;
    g.n = g.m + g.r1 - 1;
    g.tb = 16LL * s->q;
    g.kb = 16LL * s->fk;
    g.wo = fw_floordiv((int64_t)p->Wi + 2LL * p->pW - (int64_t)p->dW * (p->X - 1) - 1, p->sW) + 1;
    g.sub_w = p->dW > 0 ? fw_cdiv(g.wo, p->dW) : 0;
    g.tiles_w = g.m > 0 ? fw_cdiv(g.sub_w, g.m) : 0;
    g.rows = (int64_t)p->N * p->dW;
    g.bpr = g.tb > 0 ? fw_cdiv(g.tiles_w, g.tb) : 0;
    g.tile_blocks = g.rows * g.bpr;
    g.k_blocks = g.kb > 0 ? fw_floordiv(p->K, g.kb) : 0;
    g.vt = g.tb + g.S - 1;
    g.raw_rows = g.vt * g.m + g.n - g.m;
    int64_t t = g.n > 0 ? 16 / g.n : 0;
    while(t > 1 && (t - 1) * g.m + g.n > 32)
        --t;
    g.xf_tiles = t;
    g.xf_win = ((g.xf_tiles - 1) * g.m + g.n <= 16) ? 16 : 32;
    g.xf_groups = g.xf_tiles > 0 ? fw_cdiv(g.vt, g.xf_tiles) : 0;
    int64_t rr = g.raw_rows;
    if(fw_is_wmma(s))
    {
        int64_t w = (g.xf_groups - 1) * g.xf_tiles * g.m + g.xf_win;
        if(w > rr)
            rr = w;
    }
    g.raw_alloc_rows = rr + 1;
    return g;
}

int64_t fw_grid(const rocke_winograd_fused_1d_spec_t* s, const Geo& g)
{
    int64_t kbn = g.k_blocks;
    if(kbn <= 0)
        return 0;
    int64_t per_k = fw_floordiv(s->grid_blocks, kbn);
    if(per_k < 1)
        per_k = 1;
    if(per_k > g.tile_blocks)
        per_k = g.tile_blocks;
    return per_k * kbn;
}

int64_t fw_lds(const rocke_winograd_fused_1d_spec_t* s, const Geo& g)
{
    int64_t c = s->problem.C;
    int64_t raw = (s->prefetch ? 2 : 1) * g.raw_alloc_rows * (c + FW_PAD_H) * 2;
    int64_t v = g.n * (g.vt + 1) * (c + FW_PAD_H) * 2;
    int64_t mm;
    if(s->m_f16)
        mm = (g.n + 1) * g.tb * (g.kb + FW_PAD_H) * 2;
    else
        mm = g.n * g.tb * (g.kb + FW_PAD_F) * 4;
    return raw + v + mm;
}

/* float(np.float16(v)) == v. */
bool fw_fp16_exact(double v)
{
    if(v == 0.0)
        return true;
    if(fabs(v) > 65504.0)
        return false;
    int e = 0;
    frexp(v, &e);
    int shift = 11 - e;
    if(shift > 24)
        shift = 24;
    double sc = ldexp(v, shift);
    return sc == floor(sc);
}

std::string fw_frac(const rocke_wino::Rational& r)
{
    if(r.den == 1)
        return std::to_string((long long)r.num);
    return std::to_string((long long)r.num) + "/" + std::to_string((long long)r.den);
}

/* WinoAxis.check() + matrices(); "" when valid. */
std::string fw_axis_reason(int m, int r, int seg)
{
    if(m < 1 || r < 1 || seg < 1)
        return "bad axis WinoAxis(m=" + std::to_string(m) + ", r=" + std::to_string(r)
               + ", seg=" + std::to_string(seg) + ", pts=None)";
    if(seg > 1 && seg * m < r)
        return std::to_string(seg) + " segments of " + std::to_string(m)
               + " taps < r=" + std::to_string(r);
    if(seg > 1 && (seg - 1) * m >= r)
        return "an all-zero segment";
    int r1 = seg > 1 ? m : r;
    if(rocke_wino::toom_cook(m, r1) == nullptr)
        return "no default points for F(" + std::to_string(m) + "," + std::to_string(r1) + ")";
    return "";
}

} // namespace

/* ===================================================================== *
 *  Spec defaults + derived geometry
 * ===================================================================== */

rocke_winograd_fused_1d_spec_t rocke_winograd_fused_1d_spec_default(void)
{
    rocke_winograd_fused_1d_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem.sH = 1;
    s.problem.sW = 1;
    s.problem.dH = 1;
    s.problem.dW = 1;
    s.problem.groups = 1;
    s.ax_m = 4;
    s.ax_r = 3;
    s.ax_seg = 1;
    s.q = 4;
    s.fk = 2;
    s.grid_blocks = 256;
    s.prefetch = false;
    s.m_f16 = false;
    s.xf = "valu";
    s.xf_batch = 4;
    s.merge = false;
    s.waves = 8;
    s.waves_per_eu = 0;
    s.name = "conv_wino_fused1d";
    return s;
}

int rocke_winograd_fused_1d_grid(const rocke_winograd_fused_1d_spec_t* spec)
{
    Geo g = fw_geo(spec);
    return (int)fw_grid(spec, g);
}

long rocke_winograd_fused_1d_lds_bytes(const rocke_winograd_fused_1d_spec_t* spec)
{
    Geo g = fw_geo(spec);
    return (long)fw_lds(spec, g);
}

rocke_status_t rocke_winograd_fused_1d_kernel_name(const rocke_winograd_fused_1d_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    const rocke_conv_problem_t* p = &spec->problem;
    Geo g = fw_geo(spec);
    char buf[256];
    std::string tag;
    snprintf(buf, sizeof buf, "N%dH%dW%dC%d_K%dY%dX%d", p->N, p->Hi, p->Wi, p->C, p->K, p->Y, p->X);
    tag = buf;
    if(p->groups > 1)
        tag += "G" + std::to_string(p->groups);
    if(spec->ax_seg > 1)
        snprintf(buf, sizeof buf, "%dxF%d_%d", spec->ax_seg, spec->ax_m, spec->ax_m);
    else
        snprintf(buf, sizeof buf, "F%d_%d", spec->ax_m, spec->ax_r);
    tag += std::string("_") + buf;
    snprintf(buf, sizeof buf, "_q%d_fk%d_g%lld", spec->q, spec->fk, (long long)fw_grid(spec, g));
    tag += buf;
    if(spec->xf == NULL || strcmp(spec->xf, "valu") != 0)
        tag += std::string("_x") + (spec->xf ? spec->xf : "None") + std::to_string(spec->xf_batch);
    if(spec->prefetch)
        tag += "_pf";
    if(spec->merge)
        tag += "_mg";
    if(spec->waves != 8)
        tag += "_w" + std::to_string(spec->waves);
    if(spec->waves_per_eu)
        tag += "_o" + std::to_string(spec->waves_per_eu);
    if(spec->m_f16)
        tag += "_mh";
    std::string name = spec->name ? spec->name : "";
    std::string full = name.empty() ? tag : name + "_" + tag;
    for(char& ch : full)
        if(ch == '/')
            ch = '_';
    if(out == NULL || full.size() + 1 > out_cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, full.c_str(), full.size() + 1);
    return ROCKE_OK;
}

/* ===================================================================== *
 *  FusedWino1DSpec.reason
 * ===================================================================== */

bool rocke_winograd_fused_1d_is_valid_spec(const rocke_winograd_fused_1d_spec_t* spec,
                                           char* reason,
                                           size_t reason_cap)
{
    char msg[512];
    if(spec == NULL)
    {
        fw_set(reason, reason_cap, "null spec");
        return false;
    }
    const rocke_conv_problem_t* p = &spec->problem;
    if(p->is_3d || p->Y != 1 || p->Hi != 1)
    {
        fw_set(reason, reason_cap, "1-D convs only (Y = 1, H = 1)");
        return false;
    }
    if(p->sH != 1 || p->sW != 1 || p->pH != 0 || p->groups != 1)
    {
        fw_set(reason, reason_cap, "stride 1, no H pad, groups 1");
        return false;
    }
    if(spec->ax_r != p->X)
    {
        fw_set(reason, reason_cap, "axis taps must match the filter");
        return false;
    }
    {
        std::string why = fw_axis_reason(spec->ax_m, spec->ax_r, spec->ax_seg);
        if(!why.empty())
        {
            fw_set(reason, reason_cap, why.c_str());
            return false;
        }
    }
    if(p->C % 32 || p->K % 16)
    {
        fw_set(reason, reason_cap, "C must be a multiple of 32 and K of 16");
        return false;
    }
    Geo g = fw_geo(spec);
    if(g.kb <= 0 || p->K % g.kb)
    {
        snprintf(msg, sizeof msg, "K %d not a multiple of Kb %lld", p->K, (long long)g.kb);
        fw_set(reason, reason_cap, msg);
        return false;
    }
    if(g.n > 16)
    {
        fw_set(reason, reason_cap, "n > 16");
        return false;
    }
    if(spec->merge && !spec->prefetch)
    {
        fw_set(reason, reason_cap, "merge needs prefetch");
        return false;
    }
    if(spec->xf == NULL || (strcmp(spec->xf, "valu") != 0 && strcmp(spec->xf, "wmma") != 0))
    {
        fw_set(reason, reason_cap, "xf must be valu / wmma");
        return false;
    }
    if(fw_is_wmma(spec))
    {
        const ToomCook* tc = rocke_wino::toom_cook(spec->ax_m, g.r1);
        for(int i = 0; i < tc->n; ++i)
            for(int j = 0; j < tc->n; ++j)
            {
                const rocke_wino::Rational& v = tc->BT(i, j);
                if(!fw_fp16_exact(v.value()))
                {
                    std::string m = "B^T entry " + fw_frac(v) + " is not exact in fp16";
                    fw_set(reason, reason_cap, m.c_str());
                    return false;
                }
            }
        if(g.n > 16 || g.n > 32)
        {
            fw_set(reason, reason_cap, "n too large for the WMMA transform");
            return false;
        }
    }
    if(spec->q < 1 || spec->fk < 1)
    {
        fw_set(reason, reason_cap, "q, fk >= 1");
        return false;
    }
    const rocke_archtarget_t* t = rocke_archtarget_from_gfx("gfx1250");
    if(t == NULL || t->wave_size != 32)
    {
        fw_set(reason, reason_cap, "wave32 only");
        return false;
    }
    int64_t lds = fw_lds(spec, g);
    if(!rocke_archtarget_fits_lds(t, (long)lds))
    {
        snprintf(msg, sizeof msg, "LDS %lld B too large", (long long)lds);
        fw_set(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves != 4 && spec->waves != 8)
    {
        fw_set(reason, reason_cap, "waves must be 4 or 8");
        return false;
    }
    {
        int64_t c32 = p->C / 32;
        int64_t bpw = fw_cdiv(g.n, spec->waves);
        int64_t u = bpw * g.S * c32 * spec->fk * 8;
        int64_t acc = (int64_t)spec->q * spec->fk * 8;
        int64_t vf = ((int64_t)spec->q + g.S - 1) * 8;
        int64_t est = u + acc + 2 * vf + 40;
        if(est > (spec->waves == 8 ? FW_VGPR_BUDGET : 980))
        {
            snprintf(msg, sizeof msg, "~%lld VGPRs", (long long)est);
            fw_set(reason, reason_cap, msg);
            return false;
        }
    }
    if(fw_grid(spec, g) <= 0)
    {
        fw_set(reason, reason_cap, "empty grid");
        return false;
    }
    {
        int64_t xb = (int64_t)p->N * p->Wi * p->C * 2;
        int64_t ub = (int64_t)g.n * g.S * p->K * p->C * 2;
        int64_t yb = (int64_t)p->N * g.wo * p->K * 2;
        if(xb > rocke_wino::kOob || ub > rocke_wino::kOob || yb > rocke_wino::kOob)
        {
            fw_set(reason, reason_cap, "a buffer exceeds 32-bit offsets");
            return false;
        }
    }
    fw_set(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  Builder
 * ===================================================================== */
namespace
{

typedef rocke_value_t V;

struct Ctx
{
    rocke_ir_builder_t* b;
    const rocke_winograd_fused_1d_spec_t* spec;
    Geo g;
    const char* wmma;
    std::vector<double> at; /* m x n */
    std::vector<double> bt; /* n x n */
    int m, n, S, Q, FK, C, K, d, pw, Wi, Wo;
    int64_t TB, KB, VT, RR, CP, MP, C32, CV, WAVES, NTH, KBN, GRID, GPK, RA, MDUMMY, MPH;
    bool PF, MH;
    V *x, *x_rsrc, *u_rsrc, *y_rsrc, *c0, *oob;
    V *raw_lds, *v_lds, *m_lds;
    V *tid, *wave, *lane, *l16, *lhalf, *wg, *kbi, *first, *k0, *n_iter;
    V *tr_col, *tr_row, *xf_tg, *xf_bin, *xf_colok, *bt_frag, *zero8;
    /* ufr[slot][s][ks][f] flattened */
    std::vector<V*> ufr;
    int64_t n_raw;
};

V* ci(Ctx& c, int64_t v)
{
    return rocke_b_const_i32(c.b, v);
}
V* add(Ctx& c, V* a, V* x)
{
    return rocke_b_add(c.b, a, x);
}
V* sub(Ctx& c, V* a, V* x)
{
    return rocke_b_sub(c.b, a, x);
}
V* mul(Ctx& c, V* a, V* x)
{
    return rocke_b_mul(c.b, a, x);
}
V* divi(Ctx& c, V* a, V* x)
{
    return rocke_b_div(c.b, a, x);
}
V* modi(Ctx& c, V* a, V* x)
{
    return rocke_b_mod(c.b, a, x);
}

V*& UFR(Ctx& c, int slot, int s, int ks, int f)
{
    return c.ufr[(((size_t)slot * c.S + s) * c.C32 + ks) * c.FK + f];
}

void store_vN(Ctx& c, V* smem, V* i0, V* i1, V* val, int n)
{
    V* idx[2] = {i0, i1};
    rocke_b_smem_store_vN(c.b, smem, idx, 2, val, n);
}
V* load_vN(Ctx& c, V* smem, V* i0, V* i1, int n)
{
    V* idx[2] = {i0, i1};
    return rocke_b_smem_load_vN(c.b, smem, idx, 2, rocke_f16(), n);
}

struct Dec
{
    V *img, *a, *t0;
};

Dec decode(Ctx& c, V* blk)
{
    Dec r;
    V* row = divi(c, blk, ci(c, c.g.bpr));
    V* tbi = modi(c, blk, ci(c, c.g.bpr));
    r.img = c.d > 1 ? divi(c, row, ci(c, c.d)) : row;
    r.a = c.d > 1 ? modi(c, row, ci(c, c.d)) : c.c0;
    r.t0 = mul(c, tbi, ci(c, c.TB));
    return r;
}

struct RawOff
{
    V *off, *okit, *pr, *cv, *ok, *elem;
};

RawOff raw_offsets(Ctx& c, const Dec& dc, V* item, int64_t n_raw)
{
    RawOff o;
    V* wi0;
    {
        V* x1 = ci(c, c.m);
        V* x2 = mul(c, dc.t0, x1);
        V* x3 = ci(c, c.d);
        V* x4 = mul(c, x2, x3);
        V* x5 = add(c, dc.a, x4);
        V* x6 = ci(c, c.pw);
        wi0 = sub(c, x5, x6);
    }
    o.pr = divi(c, item, ci(c, c.CV));
    o.cv = modi(c, item, ci(c, c.CV));
    V* wi;
    {
        V* x1 = mul(c, o.pr, ci(c, c.d));
        wi = add(c, wi0, x1);
    }
    o.okit = rocke_b_cmp_lt(c.b, item, ci(c, n_raw));
    {
        V* ge = rocke_b_cmp_ge(c.b, wi, c.c0);
        V* lt = rocke_b_cmp_lt(c.b, wi, ci(c, c.Wi));
        V* l1 = rocke_b_land(c.b, ge, lt);
        o.ok = rocke_b_land(c.b, o.okit, l1);
    }
    {
        V* x1 = mul(c, dc.img, ci(c, c.Wi));
        V* x2 = add(c, x1, wi);
        V* x3 = mul(c, x2, ci(c, c.C));
        V* x4 = mul(c, o.cv, ci(c, FW_VEC));
        o.elem = add(c, x3, x4);
    }
    {
        V* x1 = mul(c, o.elem, ci(c, 2));
        o.off = rocke_b_select(c.b, o.ok, x1, c.oob);
    }
    return o;
}

void issue_raw(Ctx& c, V* blk, V* buf)
{
    rocke_ir_builder_t* b = c.b;
    Dec dc = decode(c, blk);
    V* rb = mul(c, buf, ci(c, c.RA));
    int64_t reps = fw_cdiv(c.n_raw, c.NTH);
    for(int64_t rep = 0; rep < reps; ++rep)
    {
        V* item = add(c, c.tid, ci(c, rep * c.NTH));
        RawOff o = raw_offsets(c, dc, item, c.n_raw);
        V* i0 = add(c, rb, o.pr);
        V* i1 = mul(c, o.cv, ci(c, FW_VEC));
        V* idx[2] = {i0, i1};
        rocke_if_else_t ie = rocke_b_scf_if_else(b, o.ok);
        rocke_b_region_enter(b, ie.then_region);
        rocke_b_global_load_async_to_lds(b, c.x, o.elem, c.raw_lds, idx, 2, 16, 0, 0);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        {
            rocke_if_t g = rocke_b_scf_if(b, o.okit);
            rocke_b_region_enter(b, g.then_region);
            rocke_b_smem_store_vN(b, c.raw_lds, idx, 2, c.zero8, FW_VEC);
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);
    }
}

void start_raw(Ctx& c, V* blk, V* buf, V* guard)
{
    if(guard == NULL)
    {
        issue_raw(c, blk, buf);
        return;
    }
    rocke_if_t g = rocke_b_scf_if(c.b, guard);
    rocke_b_region_enter(c.b, g.then_region);
    issue_raw(c, blk, buf);
    rocke_b_region_leave(c.b);
}

void land_raw(Ctx& c)
{
    rocke_b_s_wait_asynccnt(c.b, 0);
}

/* Python's 1-D linear combination of the transforms (B^T rows: fneg allowed
 * for a leading -1; A^T rows: a leading coefficient != 1 is an fmul). */
V* lin(Ctx& c, const double* coef, const std::vector<V*>& xs, bool neg_first)
{
    rocke_ir_builder_t* b = c.b;
    V* acc = NULL;
    for(size_t u = 0; u < xs.size(); ++u)
    {
        double cf = coef[u];
        if(cf == 0.0)
            continue;
        V* xv = xs[u];
        if(acc == NULL)
        {
            if(cf == 1.0)
                acc = xv;
            else if(neg_first && cf == -1.0)
                acc = rocke_b_fneg(b, xv);
            else
            {
                V* k = rocke_b_const_f32(b, cf);
                acc = rocke_b_fmul(b, k, xv);
            }
        }
        else if(cf == 1.0)
            acc = rocke_b_fadd(b, acc, xv);
        else if(cf == -1.0)
            acc = rocke_b_fsub(b, acc, xv);
        else
        {
            V* k = rocke_b_const_f32(b, cf);
            acc = rocke_b_fma(b, k, xv, acc);
        }
    }
    return acc != NULL ? acc : rocke_b_const_f32(b, 0.0);
}

void do_xf(Ctx& c, V* rrow0)
{
    rocke_ir_builder_t* b = c.b;
    V* vb = c.c0;
    int64_t n_v = c.VT * c.CV;
    int64_t reps = fw_is_wmma(c.spec) ? 0 : fw_cdiv(n_v, c.NTH);
    for(int64_t rep = 0; rep < reps; ++rep)
    {
        V* item = add(c, c.tid, ci(c, rep * c.NTH));
        V* okit = rocke_b_cmp_lt(b, item, ci(c, n_v));
        V* j = divi(c, item, ci(c, c.CV));
        {
            V* x = ci(c, c.VT);
            j = rocke_b_select(b, okit, j, x);
        }
        V* cc;
        {
            V* x = modi(c, item, ci(c, c.CV));
            cc = mul(c, x, ci(c, FW_VEC));
        }
        V* rbase;
        {
            V* x = mul(c, j, ci(c, c.m));
            V* y = rocke_b_select(b, okit, x, c.c0);
            rbase = add(c, rrow0, y);
        }
        std::vector<std::vector<V*>> xs;
        for(int uu = 0; uu < c.n; ++uu)
        {
            V* r = add(c, rbase, ci(c, uu));
            V* raw = load_vN(c, c.raw_lds, r, cc, FW_VEC);
            xs.push_back(rocke_wino::raw_to_f32(b, raw, FW_VEC));
        }
        for(int bb = 0; bb < c.n; ++bb)
        {
            std::vector<V*> casts;
            std::vector<V*> vals;
            for(int ch = 0; ch < FW_VEC; ++ch)
            {
                std::vector<V*> col;
                for(int uu = 0; uu < c.n; ++uu)
                    col.push_back(xs[uu][ch]);
                vals.push_back(lin(c, &c.bt[(size_t)bb * c.n], col, true));
            }
            for(V* v : vals)
                casts.push_back(rocke_b_cast_f32_to(b, v, rocke_f16()));
            V* packed = rocke_b_vec_pack(b, casts.data(), (int)casts.size(), rocke_f16());
            V* x1 = ci(c, (int64_t)bb * (c.VT + 1));
            V* x2 = add(c, j, x1);
            V* x3 = add(c, vb, x2);
            store_vN(c, c.v_lds, x3, cc, packed, FW_VEC);
        }
    }
    if(fw_is_wmma(c.spec))
    {
        int64_t TPW = c.g.xf_tiles;
        int64_t NG = c.g.xf_groups * (c.C / 16);
        int64_t NR = fw_cdiv(NG, c.WAVES);
        int64_t XB = c.spec->xf_batch;
        for(int64_t r0 = 0; r0 < NR; r0 += XB)
        {
            int64_t r1 = r0 + XB < NR ? r0 + XB : NR;
            std::vector<V*> frags, okgs, j0s, cgs;
            for(int64_t r = r0; r < r1; ++r)
            {
                V* wi = add(c, c.wave, ci(c, r * c.WAVES));
                V* okg = rocke_b_cmp_lt(b, wi, ci(c, NG));
                wi = rocke_b_select(b, okg, wi, c.c0);
                V* grp = divi(c, wi, ci(c, c.C / 16));
                V* cg = modi(c, wi, ci(c, c.C / 16));
                V* j0 = mul(c, grp, ci(c, TPW));
                V* prow;
                {
                    V* x1 = mul(c, j0, ci(c, c.m));
                    V* x2 = add(c, x1, c.tr_row);
                    prow = add(c, rrow0, x2);
                }
                V* pcol;
                {
                    V* x1 = mul(c, cg, ci(c, 16));
                    pcol = add(c, x1, c.tr_col);
                }
                V* lo;
                {
                    V* idx[2] = {prow, pcol};
                    lo = rocke_b_ds_read_tr16_b128(b, c.raw_lds, idx, 2, rocke_f16());
                }
                V* hi;
                {
                    V* p8 = add(c, prow, ci(c, 8));
                    V* idx[2] = {p8, pcol};
                    hi = rocke_b_ds_read_tr16_b128(b, c.raw_lds, idx, 2, rocke_f16());
                }
                frags.push_back(rocke_b_vec_concat(b, lo, hi));
                okgs.push_back(okg);
                j0s.push_back(j0);
                cgs.push_back(cg);
            }
            std::vector<V*> outs;
            for(V* f : frags)
            {
                V* z = rocke_b_zero_vec_f32(b, 8);
                outs.push_back(rocke_b_mma(b, c.wmma, f, c.bt_frag, z, NULL, 0));
            }
            for(size_t i = 0; i < outs.size(); ++i)
            {
                V* jj = add(c, j0s[i], c.xf_tg);
                V* okw;
                {
                    V* x1 = rocke_b_cmp_lt(b, jj, ci(c, c.VT));
                    V* x2 = rocke_b_land(b, c.xf_colok, x1);
                    okw = rocke_b_land(b, okgs[i], x2);
                }
                V* vrow;
                {
                    V* x1 = mul(c, c.xf_bin, ci(c, c.VT + 1));
                    V* x2 = add(c, x1, jj);
                    V* x3 = ci(c, c.VT);
                    vrow = rocke_b_select(b, okw, x2, x3);
                }
                vrow = add(c, vrow, vb);
                V* colv;
                {
                    V* x1 = mul(c, cgs[i], ci(c, 16));
                    V* x2 = mul(c, c.lhalf, ci(c, 8));
                    colv = add(c, x1, x2);
                }
                V* cast = rocke_b_vec_cast_f32_to(b, outs[i], rocke_f16());
                store_vN(c, c.v_lds, vrow, colv, cast, 8);
            }
        }
    }
}

void do_gemm(Ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    V* vb = c.c0;
    V* mb = c.c0;
    int bpw = (int)fw_cdiv(c.n, c.WAVES);
    for(int slot = 0; slot < bpw; ++slot)
    {
        V* binv = add(c, c.wave, ci(c, c.WAVES * slot));
        V* okb = rocke_b_cmp_lt(b, binv, ci(c, c.n));
        rocke_if_t gate = rocke_b_scf_if(b, okb);
        rocke_b_region_enter(b, gate.then_region);
        V* vrow0;
        {
            V* x1 = rocke_b_select(b, okb, binv, c.c0);
            V* x2 = mul(c, x1, ci(c, c.VT + 1));
            vrow0 = add(c, vb, x2);
        }
        V* mrow0;
        {
            V* x1 = mul(c, binv, ci(c, c.TB));
            V* x2 = add(c, mb, x1);
            V* x3 = ci(c, c.MDUMMY);
            mrow0 = rocke_b_select(b, okb, x2, x3);
        }
        std::vector<V*> accs;
        for(int i = 0; i < c.Q * c.FK; ++i)
            accs.push_back(rocke_b_zero_vec_f32(b, 8));
        V* col = mul(c, c.lhalf, ci(c, 16));
        for(int ks = 0; ks < c.C32; ++ks)
        {
            std::vector<V*> vf;
            for(int g = 0; g < c.Q + c.S - 1; ++g)
            {
                V* r_;
                {
                    V* x1 = mul(c, c.l16, ci(c, c.Q));
                    V* x2 = add(c, x1, ci(c, g));
                    r_ = add(c, vrow0, x2);
                }
                V* lo;
                {
                    V* x1 = add(c, col, ci(c, (int64_t)ks * 32));
                    lo = load_vN(c, c.v_lds, r_, x1, 8);
                }
                V* hi;
                {
                    V* x1 = add(c, col, ci(c, (int64_t)ks * 32 + 8));
                    hi = load_vN(c, c.v_lds, r_, x1, 8);
                }
                vf.push_back(rocke_b_vec_concat(b, lo, hi));
            }
            for(int s = 0; s < c.S; ++s)
                for(int qq = 0; qq < c.Q; ++qq)
                    for(int f = 0; f < c.FK; ++f)
                    {
                        int i_ = qq * c.FK + f;
                        accs[i_] = rocke_b_mma(
                            b, c.wmma, UFR(c, slot, s, ks, f), vf[qq + s], accs[i_], NULL, 0);
                    }
        }
        for(int qq = 0; qq < c.Q; ++qq)
        {
            V* jr;
            {
                V* x1 = mul(c, c.l16, ci(c, c.Q));
                V* x2 = add(c, x1, ci(c, qq));
                jr = add(c, mrow0, x2);
            }
            for(int f = 0; f < c.FK; ++f)
            {
                V* acc = accs[qq * c.FK + f];
                V* kc;
                {
                    V* x1 = ci(c, (int64_t)f * 16);
                    V* x2 = mul(c, c.lhalf, ci(c, 8));
                    kc = add(c, x1, x2);
                }
                if(c.MH)
                {
                    V* cast = rocke_b_vec_cast_f32_to(b, acc, rocke_f16());
                    store_vN(c, c.m_lds, jr, kc, cast, 8);
                    continue;
                }
                V* e4[4];
                for(int e = 0; e < 4; ++e)
                    e4[e] = rocke_b_vec_extract(b, acc, e);
                V* lo = rocke_b_vec_pack(b, e4, 4, rocke_f32());
                for(int e = 0; e < 4; ++e)
                    e4[e] = rocke_b_vec_extract(b, acc, 4 + e);
                V* hi = rocke_b_vec_pack(b, e4, 4, rocke_f32());
                {
                    V* idx[2] = {jr, kc};
                    rocke_b_smem_store_vN_f32(b, c.m_lds, idx, 2, lo, 4);
                }
                {
                    V* k4 = add(c, kc, ci(c, 4));
                    V* idx[2] = {jr, k4};
                    rocke_b_smem_store_vN_f32(b, c.m_lds, idx, 2, hi, 4);
                }
            }
        }
        rocke_b_region_leave(b);
    }
}

void do_out(Ctx& c, const Dec& dc)
{
    rocke_ir_builder_t* b = c.b;
    V* mb = c.c0;
    int64_t KV = c.KB / FW_VEC;
    int64_t n_o = c.TB * KV;
    V* yimg = mul(c, dc.img, ci(c, c.Wo));
    int64_t reps = fw_cdiv(n_o, c.NTH);
    for(int64_t rep = 0; rep < reps; ++rep)
    {
        V* item = add(c, c.tid, ci(c, rep * c.NTH));
        V* okit = rocke_b_cmp_lt(b, item, ci(c, n_o));
        V* j;
        {
            V* x1 = divi(c, item, ci(c, KV));
            j = rocke_b_select(b, okit, x1, c.c0);
        }
        V* kc;
        {
            V* x1 = modi(c, item, ci(c, KV));
            kc = mul(c, x1, ci(c, FW_VEC));
        }
        std::vector<std::vector<V*>> ms;
        for(int bb = 0; bb < c.n; ++bb)
        {
            V* r_;
            {
                V* x1 = ci(c, (int64_t)bb * c.TB);
                V* x2 = add(c, x1, j);
                r_ = add(c, mb, x2);
            }
            if(c.MH)
            {
                V* raw = load_vN(c, c.m_lds, r_, kc, 8);
                ms.push_back(rocke_wino::raw_to_f32(b, raw, 8));
                continue;
            }
            V* lo;
            {
                V* idx[2] = {r_, kc};
                lo = rocke_b_smem_load_vN_f32(b, c.m_lds, idx, 2, 4);
            }
            V* hi;
            {
                V* k4 = add(c, kc, ci(c, 4));
                V* idx[2] = {r_, k4};
                hi = rocke_b_smem_load_vN_f32(b, c.m_lds, idx, 2, 4);
            }
            std::vector<V*> v8;
            for(int e = 0; e < 4; ++e)
                v8.push_back(rocke_b_vec_extract(b, lo, e));
            for(int e = 0; e < 4; ++e)
                v8.push_back(rocke_b_vec_extract(b, hi, e));
            ms.push_back(v8);
        }
        V* tile = add(c, dc.t0, j);
        for(int i = 0; i < c.m; ++i)
        {
            std::vector<V*> vals;
            for(int ch = 0; ch < FW_VEC; ++ch)
            {
                std::vector<V*> col;
                for(int bb = 0; bb < c.n; ++bb)
                    col.push_back(ms[bb][ch]);
                vals.push_back(lin(c, &c.at[(size_t)i * c.n], col, false));
            }
            V* wo;
            {
                V* x1 = mul(c, tile, ci(c, c.m));
                V* x2 = add(c, x1, ci(c, i));
                V* x3 = mul(c, x2, ci(c, c.d));
                wo = add(c, dc.a, x3);
            }
            V* ok;
            {
                V* x1 = rocke_b_cmp_lt(b, wo, ci(c, c.Wo));
                ok = rocke_b_land(b, okit, x1);
            }
            V* off;
            {
                V* x1 = add(c, yimg, wo);
                V* x2 = mul(c, x1, ci(c, c.K));
                V* x3 = add(c, c.k0, kc);
                V* x4 = add(c, x2, x3);
                off = mul(c, x4, ci(c, 2));
            }
            std::vector<V*> casts;
            for(V* v : vals)
                casts.push_back(rocke_b_cast_f32_to(b, v, rocke_f16()));
            V* packed = rocke_b_vec_pack(b, casts.data(), (int)casts.size(), rocke_f16());
            V* sel = rocke_b_select(b, ok, off, c.oob);
            rocke_b_buffer_store_vN_f16(b, c.y_rsrc, sel, c.c0, packed, FW_VEC / 2);
        }
    }
}

V* smem2(Ctx& c, const rocke_type_t* ty, int64_t d0, int64_t d1, const char* hint)
{
    int shape[2] = {(int)d0, (int)d1};
    return rocke_b_smem_alloc_ex(c.b, ty, shape, 2, hint, 1);
}

rocke_kernel_def_t*
    build(rocke_ir_builder_t* b, const rocke_winograd_fused_1d_spec_t* spec, const char* arch)
{
    char reason[512];
    if(!rocke_winograd_fused_1d_is_valid_spec(spec, reason, sizeof reason))
    {
        std::string m = std::string("FusedWino1DSpec: ") + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, m.c_str());
        return NULL;
    }
    Ctx c;
    c.b = b;
    c.spec = spec;
    c.g = fw_geo(spec);
    const rocke_conv_problem_t* p = &spec->problem;
    c.C = p->C;
    c.K = p->K;
    c.m = c.g.m;
    c.n = c.g.n;
    c.S = c.g.S;
    c.Q = spec->q;
    c.FK = spec->fk;
    c.TB = c.g.tb;
    c.KB = c.g.kb;
    c.VT = c.g.vt;
    c.RR = c.g.raw_rows;
    c.d = p->dW;
    c.pw = p->pW;
    c.Wi = p->Wi;
    c.Wo = (int)c.g.wo;
    {
        const ToomCook* tc = rocke_wino::toom_cook(c.m, c.g.r1);
        c.at.resize((size_t)c.m * c.n);
        c.bt.resize((size_t)c.n * c.n);
        for(int i = 0; i < c.m; ++i)
            for(int j = 0; j < c.n; ++j)
                c.at[(size_t)i * c.n + j] = tc->AT(i, j).value();
        for(int i = 0; i < c.n; ++i)
            for(int j = 0; j < c.n; ++j)
                c.bt[(size_t)i * c.n + j] = tc->BT(i, j).value();
    }
    c.CP = c.C + FW_PAD_H;
    c.MP = c.KB + FW_PAD_F;
    c.C32 = c.C / 32;
    c.CV = c.C / FW_VEC;
    c.WAVES = spec->waves;
    c.NTH = c.WAVES * FW_WAVE;
    c.KBN = c.g.k_blocks;
    c.GRID = fw_grid(spec, c.g);
    c.GPK = c.GRID / c.KBN;
    {
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        const rocke_mmaop_t* op = target
                                      ? rocke_archtarget_op_for_shape(
                                            target, "wmma", "f16", "f16", "fp32", 16, 16, 32, NULL)
                                      : NULL;
        if(op == NULL || op->op_id == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_fused_wino_1d: no WMMA atom");
            return NULL;
        }
        c.wmma = op->op_id;
    }

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.NTH);
    if(spec->waves_per_eu)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
    const rocke_type_t* f16p = rocke_ptr_type(b, rocke_f16(), "global");
    rocke_param_opts_t ro;
    memset(&ro, 0, sizeof ro);
    ro.noalias = true;
    ro.noalias_set = true;
    ro.readonly = true;
    ro.readonly_set = true;
    rocke_param_opts_t wo;
    memset(&wo, 0, sizeof wo);
    wo.noalias = true;
    wo.noalias_set = true;
    wo.writeonly = true;
    wo.writeonly_set = true;
    c.x = rocke_b_param(b, "x", f16p, &ro);
    V* u = rocke_b_param(b, "u", f16p, &ro);
    V* y = rocke_b_param(b, "y", f16p, &wo);
    {
        V* nb = rocke_b_param(b, "x_bytes", rocke_i32(), NULL);
        c.x_rsrc = rocke_b_buffer_rsrc(b, c.x, nb);
    }
    {
        V* nb = rocke_b_param(b, "u_bytes", rocke_i32(), NULL);
        c.u_rsrc = rocke_b_buffer_rsrc(b, u, nb);
    }
    {
        V* nb = rocke_b_param(b, "y_bytes", rocke_i32(), NULL);
        c.y_rsrc = rocke_b_buffer_rsrc(b, y, nb);
    }
    c.c0 = ci(c, 0);
    c.oob = ci(c, rocke_wino::kOob);

    c.PF = spec->prefetch;
    c.RA = c.g.raw_alloc_rows;
    c.raw_lds = smem2(c, rocke_f16(), (c.PF ? 2 : 1) * c.RA, c.CP, "raw");
    c.MDUMMY = (int64_t)c.n * c.TB;
    c.v_lds = smem2(c, rocke_f16(), (int64_t)c.n * (c.VT + 1), c.CP, "vx");
    c.MH = spec->m_f16;
    c.MPH = c.KB + FW_PAD_H;
    if(c.MH)
        c.m_lds = smem2(c, rocke_f16(), c.MDUMMY + c.TB, c.MPH, "mm");
    else
        c.m_lds = smem2(c, rocke_f32(), (int64_t)c.n * c.TB, c.MP, "mm");

    c.tid = rocke_b_thread_id_x(b);
    c.wave = divi(c, c.tid, ci(c, FW_WAVE));
    c.lane = modi(c, c.tid, ci(c, FW_WAVE));
    c.l16 = modi(c, c.lane, ci(c, 16));
    c.lhalf = divi(c, c.lane, ci(c, 16));
    c.wg = rocke_b_block_id_x(b);
    c.kbi = modi(c, c.wg, ci(c, c.KBN));
    c.first = divi(c, c.wg, ci(c, c.KBN));
    c.k0 = mul(c, c.kbi, ci(c, c.KB));

    /* ---- prologue: this wave's transformed weights (A fragments). */
    int bpw = (int)fw_cdiv(c.n, c.WAVES);
    c.ufr.assign((size_t)bpw * c.S * c.C32 * c.FK, NULL);
    for(int slot = 0; slot < bpw; ++slot)
    {
        V* binv = add(c, c.wave, ci(c, c.WAVES * slot));
        V* okb = rocke_b_cmp_lt(b, binv, ci(c, c.n));
        for(int s = 0; s < c.S; ++s)
            for(int ks = 0; ks < c.C32; ++ks)
                for(int f = 0; f < c.FK; ++f)
                {
                    V* row;
                    {
                        V* x1 = mul(c, binv, ci(c, c.S));
                        V* x2 = add(c, x1, ci(c, s));
                        V* x3 = mul(c, x2, ci(c, c.K));
                        V* x4 = add(c, c.k0, ci(c, (int64_t)f * 16));
                        row = add(c, x3, x4);
                    }
                    row = add(c, row, c.l16);
                    V* off;
                    {
                        V* x1 = mul(c, row, ci(c, c.C));
                        V* x2 = ci(c, (int64_t)ks * 32);
                        V* x3 = mul(c, c.lhalf, ci(c, 16));
                        V* x4 = add(c, x2, x3);
                        off = add(c, x1, x4);
                    }
                    {
                        V* x1 = mul(c, off, ci(c, 2));
                        off = rocke_b_select(b, okb, x1, c.oob);
                    }
                    V* lo = rocke_wino::load_raw(b, c.u_rsrc, off, "fp16", 8);
                    V* o16 = add(c, off, ci(c, 16));
                    V* hi = rocke_wino::load_raw(b, c.u_rsrc, o16, "fp16", 8);
                    UFR(c, slot, s, ks, f) = rocke_b_vec_concat(b, lo, hi);
                }
    }

    {
        V* x1 = ci(c, c.g.tile_blocks);
        V* x2 = sub(c, x1, c.first);
        V* x3 = add(c, x2, ci(c, c.GPK - 1));
        c.n_iter = divi(c, x3, ci(c, c.GPK));
    }
    if(fw_is_wmma(spec))
    {
        {
            V* x1 = divi(c, c.l16, ci(c, 8));
            c.tr_col = mul(c, x1, ci(c, 8));
        }
        if(c.g.xf_win == 16)
            c.tr_row = modi(c, c.lane, ci(c, 8));
        else
        {
            V* x1 = mul(c, c.lhalf, ci(c, 16));
            V* x2 = modi(c, c.lane, ci(c, 8));
            c.tr_row = add(c, x1, x2);
        }
        int64_t TPW = c.g.xf_tiles;
        c.xf_tg = divi(c, c.l16, ci(c, c.n));
        c.xf_bin = modi(c, c.l16, ci(c, c.n));
        c.xf_colok = rocke_b_cmp_lt(b, c.l16, ci(c, TPW * c.n));
        std::vector<V*> elems;
        for(int e = 0; e < 16; ++e)
        {
            V* pos;
            {
                V* x1 = mul(c, c.lhalf, ci(c, 16));
                pos = add(c, x1, ci(c, e));
            }
            V* rel;
            {
                V* x1 = mul(c, c.xf_tg, ci(c, c.m));
                rel = sub(c, pos, x1);
            }
            V* val = rocke_b_const_f32(b, 0.0);
            for(int bb = 0; bb < c.n; ++bb)
                for(int uu = 0; uu < c.n; ++uu)
                {
                    double cf = c.bt[(size_t)bb * c.n + uu];
                    if(cf == 0.0)
                        continue;
                    V* x1 = rocke_b_cmp_eq(b, c.xf_bin, ci(c, bb));
                    V* x2 = rocke_b_cmp_eq(b, rel, ci(c, uu));
                    V* x3 = rocke_b_land(b, x1, x2);
                    V* hit = rocke_b_land(b, x3, c.xf_colok);
                    V* k = rocke_b_const_f32(b, cf);
                    val = rocke_b_select(b, hit, k, val);
                }
            elems.push_back(rocke_b_cast_f32_to(b, val, rocke_f16()));
        }
        c.bt_frag = rocke_b_vec_pack(b, elems.data(), (int)elems.size(), rocke_f16());
        int64_t tail0 = c.RR;
        int64_t n_tail = (c.RA - 1 - tail0) * c.CV;
        for(int buf = 0; buf < (c.PF ? 2 : 1); ++buf)
        {
            int64_t reps = fw_cdiv(n_tail, c.NTH);
            for(int64_t rep = 0; rep < reps; ++rep)
            {
                V* item = add(c, c.tid, ci(c, rep * c.NTH));
                V* ok = rocke_b_cmp_lt(b, item, ci(c, n_tail));
                V* rr_;
                {
                    V* x1 = ci(c, buf * c.RA + tail0);
                    V* x2 = divi(c, item, ci(c, c.CV));
                    V* x3 = add(c, x1, x2);
                    V* x4 = ci(c, buf * c.RA + c.RA - 1);
                    rr_ = rocke_b_select(b, ok, x3, x4);
                }
                V* colv;
                {
                    V* x1 = modi(c, item, ci(c, c.CV));
                    colv = mul(c, x1, ci(c, FW_VEC));
                }
                V* z = rocke_b_zero_vec(b, rocke_f16(), FW_VEC);
                store_vN(c, c.raw_lds, rr_, colv, z, FW_VEC);
            }
        }
        rocke_b_sync_lds_only(b);
    }

    c.n_raw = c.RR * c.CV;
    c.zero8 = rocke_b_zero_vec(b, rocke_f16(), FW_VEC);

    if(c.PF)
    {
        start_raw(c, c.first, c.c0, NULL);
        land_raw(c);
        rocke_b_sync_lds_only(b);
    }

    if(spec->merge)
    {
        do_xf(c, c.c0);
        rocke_b_sync_lds_only(b);
        V* one = ci(c, 1);
        rocke_for_t loop = rocke_b_scf_for(b, c.c0, c.n_iter, one, "it");
        if(!rocke_ir_builder_ok(b))
            return NULL;
        rocke_b_region_enter(b, loop.body);
        {
            V* it = loop.iv;
            V* blk;
            {
                V* x1 = mul(c, it, ci(c, c.GPK));
                blk = add(c, c.first, x1);
            }
            Dec dc = decode(c, blk);
            V* cur = modi(c, it, ci(c, 2));
            V* more;
            {
                V* x1 = add(c, it, ci(c, 1));
                more = rocke_b_cmp_lt(b, x1, c.n_iter);
            }
            {
                V* nblk = add(c, blk, ci(c, c.GPK));
                V* nbuf = sub(c, ci(c, 1), cur);
                start_raw(c, nblk, nbuf, more);
            }
            do_gemm(c);
            land_raw(c);
            rocke_b_sync_lds_only(b);
            do_out(c, dc);
            {
                rocke_if_t g = rocke_b_scf_if(b, more);
                rocke_b_region_enter(b, g.then_region);
                V* x1 = sub(c, ci(c, 1), cur);
                V* x2 = mul(c, x1, ci(c, c.RA));
                do_xf(c, x2);
                rocke_b_region_leave(b);
            }
            rocke_b_sync_lds_only(b);
        }
        rocke_b_region_leave(b);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        return b->kernel;
    }

    V* one = ci(c, 1);
    rocke_for_t loop = rocke_b_scf_for(b, c.c0, c.n_iter, one, "it");
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, loop.body);
    {
        V* it = loop.iv;
        V* blk;
        {
            V* x1 = mul(c, it, ci(c, c.GPK));
            blk = add(c, c.first, x1);
        }
        Dec dc = decode(c, blk);
        V* cur = c.PF ? modi(c, it, ci(c, 2)) : c.c0;
        V* rrow0 = c.PF ? mul(c, cur, ci(c, c.RA)) : c.c0;
        if(!c.PF)
        {
            int64_t reps = fw_cdiv(c.n_raw, c.NTH);
            for(int64_t rep = 0; rep < reps; ++rep)
            {
                V* item = add(c, c.tid, ci(c, rep * c.NTH));
                RawOff o = raw_offsets(c, dc, item, c.n_raw);
                V* val = rocke_wino::load_raw(b, c.x_rsrc, o.off, "fp16", FW_VEC);
                V* prs = rocke_b_select(b, o.okit, o.pr, ci(c, c.RR));
                V* colv = mul(c, o.cv, ci(c, FW_VEC));
                store_vN(c, c.raw_lds, prs, colv, val, FW_VEC);
            }
            rocke_b_sync_lds_only(b);
        }
        do_xf(c, rrow0);
        rocke_b_sync_lds_only(b);
        if(c.PF)
        {
            V* nblk = add(c, blk, ci(c, c.GPK));
            V* nbuf = sub(c, ci(c, 1), cur);
            V* x1 = add(c, it, ci(c, 1));
            V* guard = rocke_b_cmp_lt(b, x1, c.n_iter);
            start_raw(c, nblk, nbuf, guard);
        }
        do_gemm(c);
        rocke_b_sync_lds_only(b);
        do_out(c, dc);
        if(c.PF)
        {
            land_raw(c);
            rocke_b_sync_lds_only(b);
        }
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

} // namespace

rocke_kernel_def_t* rocke_build_winograd_fused_1d(rocke_ir_builder_t* b,
                                                  const rocke_winograd_fused_1d_spec_t* spec,
                                                  const char* arch)
{
    if(b == NULL)
        return NULL;
    if(spec == NULL)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_fused_wino_1d: null spec");
        return NULL;
    }
    return build(b, spec, arch ? arch : "gfx1250");
}

rocke_kernel_def_t* rocke_build_winograd_fused_1d_new(rocke_ir_builder_t* b,
                                                      const rocke_winograd_fused_1d_spec_t* spec,
                                                      const char* arch)
{
    if(b != NULL)
    {
        memset(b, 0, sizeof(*b));
    }
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[1024];
        char reason[512];
        if(b == NULL)
            return NULL;
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_fused_wino_1d: null spec");
            return NULL;
        }
        /* Python raises in FusedWino1DSpec.__post_init__ before any kernel. */
        if(!rocke_winograd_fused_1d_is_valid_spec(spec, reason, sizeof reason))
        {
            std::string m = std::string("FusedWino1DSpec: ") + reason;
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, m.c_str());
            return NULL;
        }
        if(rocke_winograd_fused_1d_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_fused_wino_1d: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_winograd_fused_1d(b, spec, arch);
    });
}

rocke_status_t rocke_winograd_fused_1d_lower_to_llvm(const rocke_winograd_fused_1d_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap)
{
    rocke_ir_builder_t b = {};
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        fw_set(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    kernel = rocke_build_winograd_fused_1d_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        fw_set(err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_fused_wino_1d failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
