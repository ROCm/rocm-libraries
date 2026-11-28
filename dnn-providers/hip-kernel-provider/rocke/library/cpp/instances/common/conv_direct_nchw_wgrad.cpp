// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_nchw_wgrad.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_nchw_wgrad.py: the native-NCHW direct
 * wgrad (backward-weight) convolution (build_direct_conv_nchw_wgrad). X is
 * consumed in NCHW, dY in NKHW, and dW produced in KCYX, directly -- no host
 * transpose of either activation.
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the byte-identity contract at the top of
 * rocke/instance_conv_direct_wgrad_internal.hpp, which this port reuses (the
 * `Em` helpers: at most one emitting argument per call). The Python module's
 * test/profiling-only `_MUTATIONS` fault injection is always empty in a real
 * build (and in the byte-identity gate), so this port only emits the
 * unmutated path.
 */
#include "rocke/instance_conv_direct_nchw_wgrad.h"

#include <string.h>

#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/helper_helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_* */
#include "rocke/instance_conv_direct_wgrad_internal.hpp" /* Em, V, cdiv, fail_invalid, build_new */

using rocke_dwg::cdiv;
using rocke_dwg::Em;
using rocke_dwg::fail_invalid;
using rocke_dwg::problem_dtype_ok;
using rocke_dwg::problem_is_bf16;
using rocke_dwg::V;

namespace
{

constexpr int32_t kOobBase = 0x7F000000;
constexpr int kAtom = 16; /* mfma 16x16x32 */
constexpr int kAk = 32; /* reduction depth of one MFMA */
constexpr int kRun = 8; /* 16-bit elements in one dwordx4 */
constexpr int kWave = 64;
constexpr int kXcds = 8; /* gfx950 XCDs per device */
constexpr int kDyPad = 8; /* dY row stride pad: conflict-free ds_read_b128 */

int halo_vec(int h)
{
    int v = 2;
    while(v < h)
    {
        v *= 2;
    }
    return v;
}

/* Every host-computed (compile-time) geometry constant of the spec: the C++
 * twin of DirectNchwWgradSpec's @property methods. */
struct Derived
{
    int S;
    int rows_per_group;
    int wg_taps;
    int taps_per_wave;
    int mt_w, ct_w;
    int threads;
    int halo_l, halo_r;
    int lds_h, lds_w, lds_w_stride;
    int c_stride;
    int dy_stride;
    long long lds_x_elems;
    long long lds_dy_elems;
    long long epi_elems;
    long long x_buf_stride;
    long long dy_buf_stride;
    long long lds_bytes;
    int k_steps;
    long long acc_vgprs;
    int n_tiles_h, n_tiles_w;
    long long n_tiles;
};

Derived derive(const rocke_direct_nchw_wgrad_spec_t* s)
{
    Derived d{};
    const rocke_direct_conv_problem_t* p = &s->problem;
    d.S = p->stride;
    d.rows_per_group = p->KH / s->y_groups;
    d.wg_taps = d.rows_per_group * p->KW;
    d.taps_per_wave = d.wg_taps / s->waves_t;
    d.mt_w = s->tile_m / (s->waves_m * kAtom);
    d.ct_w = s->tile_c / (s->waves_c * kAtom);
    d.threads = s->waves_m * s->waves_c * s->waves_t * kWave;
    d.halo_l = p->PAD;
    int halo_r = p->KW - p->PAD - d.S;
    d.halo_r = halo_r > 0 ? halo_r : 0;
    d.lds_h = (s->tile_h - 1) * d.S + d.rows_per_group;
    d.lds_w = d.halo_l + d.S * s->tile_w + d.halo_r;
    d.lds_w_stride = d.lds_w + s->lds_wpad;
    d.c_stride = s->tile_c + s->lds_pad;
    d.dy_stride = s->tile_h * s->tile_w + kDyPad;
    d.lds_x_elems = (long long)d.lds_h * d.lds_w_stride * d.c_stride;
    d.lds_dy_elems = (long long)s->tile_m * d.dy_stride;
    d.epi_elems = strcmp(s->epilogue, "lds") == 0
                      ? (long long)s->waves_m * kAtom * s->waves_c * kAtom * d.wg_taps
                      : 0;
    d.x_buf_stride = (long long)cdiv((int)(d.lds_x_elems + kRun), kRun) * kRun;
    d.dy_buf_stride = d.lds_dy_elems + kRun;
    int nb = s->lds_buffers;
    long long buf0 = (long long)nb * d.x_buf_stride;
    long long x_part = buf0 > d.epi_elems ? buf0 : d.epi_elems;
    d.lds_bytes = 2 * (x_part + (long long)nb * d.dy_buf_stride);
    d.k_steps = s->tile_h * s->tile_w / kAk;
    d.acc_vgprs = (long long)d.mt_w * d.ct_w * d.taps_per_wave * 4;
    d.n_tiles_h = cdiv((int)rocke_direct_conv_problem_Ho(p), s->tile_h);
    d.n_tiles_w = cdiv((int)rocke_direct_conv_problem_Wo(p), s->tile_w);
    d.n_tiles = (long long)p->N * d.n_tiles_h * d.n_tiles_w;
    return d;
}

/* spec.validate(): host-only checks (raises ValueError in Python); returns
 * false + reason on the first that fails, in Python's order. On success fills
 * *derived. */
bool validate_host(const rocke_direct_nchw_wgrad_spec_t* s,
                   Derived* derived,
                   char* reason,
                   size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(!problem_dtype_ok(p))
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "unsupported dtype %s; expected one of fp16, bf16", p->dtype);
        return false;
    }
    if(p->groups != 1)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "groups must be 1 (got %d)", p->groups);
        return false;
    }
    if(p->stride != 1 && p->stride != 2)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "stride must be 1 or 2 (got %d)", p->stride);
        return false;
    }
    if(p->dil_h != 1 || p->dil_w != 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "dilation is not supported");
        return false;
    }
    int pad_w = rocke_direct_conv_problem_pad_w(p);
    if(pad_w != p->PAD)
    {
        rocke_spec_set_reason(reason, reason_cap, "per-axis padding is not supported");
        return false;
    }
    long long Wo = rocke_direct_conv_problem_Wo(p);
    if(p->W % kRun != 0 || Wo % kRun != 0)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "W %d and Wo %lld must be multiples of %d", p->W, Wo, kRun);
        return false;
    }
    Derived d = derive(s);
    *derived = d;
    if(d.halo_l > kRun || d.halo_r > kRun)
    {
        rocke_spec_set_reason(reason, reason_cap, "halo wider than 8 columns");
        return false;
    }
    int tw = s->tile_w;
    if(tw < kRun || (tw & (tw - 1)) != 0)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "tile_w %d must be a power of two >= 8", tw);
        return false;
    }
    if(s->tile_h < 1 || (s->tile_h * tw) % kAk != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tile_h * tile_w must be a positive multiple of 32");
        return false;
    }
    if(s->y_groups < 1 || p->KH % s->y_groups != 0)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "y_groups %d must divide KH = %d", s->y_groups, p->KH);
        return false;
    }
    if(s->waves_t < 1 || d.wg_taps % s->waves_t != 0)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "waves_t %d must divide the %d taps", s->waves_t, d.wg_taps);
        return false;
    }
    int tpw = d.taps_per_wave, kw = p->KW;
    if((tpw % kw != 0) && (kw % tpw != 0))
    {
        rocke_spec_set_reason(reason, reason_cap, "taps per wave must be a multiple or a divisor of KW");
        return false;
    }
    long long total_k = rocke_direct_conv_problem_total_k(p);
    long long total_c = rocke_direct_conv_problem_total_c(p);
    if(s->tile_m % (s->waves_m * kAtom) != 0 || total_k % s->tile_m != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tile_m must divide K and be a multiple of waves_m*16");
        return false;
    }
    if(s->tile_c % (s->waves_c * kAtom) != 0 || total_c % s->tile_c != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tile_c must divide C and be a multiple of waves_c*16");
        return false;
    }
    int g = s->stage_group;
    if((g != 4 && g != 8) || s->tile_c % g != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "stage_group must be 4 or 8 and divide tile_c");
        return false;
    }
    if(s->lds_pad < 0 || s->lds_pad % g != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "lds_pad must be a non-negative multiple of stage_group");
        return false;
    }
    if(total_c % 2 != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "C must be even (packed 16-bit dW atomics)");
        return false;
    }
    if(d.acc_vgprs > 256)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "%lld accumulator registers per lane (max 256)", d.acc_vgprs);
        return false;
    }
    if(d.threads > 1024)
    {
        rocke_spec_set_reason(reason, reason_cap, "too many threads");
        return false;
    }
    if(s->waves_per_eu < 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "waves_per_eu must be >= 0");
        return false;
    }
    bool epi_lds = strcmp(s->epilogue, "lds") == 0;
    bool epi_direct = strcmp(s->epilogue, "direct") == 0;
    if(!epi_lds && !epi_direct)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "epilogue %s must be 'lds' or 'direct'", s->epilogue);
        return false;
    }
    if(epi_lds && s->y_groups != 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "the lds epilogue needs y_groups == 1");
        return false;
    }
    bool kmap_row = strcmp(s->kmap, "row") == 0;
    bool kmap_2row = strcmp(s->kmap, "2row") == 0;
    if(!kmap_row && !kmap_2row)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "kmap %s must be 'row' or '2row'", s->kmap);
        return false;
    }
    if(kmap_2row && (s->tile_w % 16 != 0 || s->tile_h % 2 != 0))
    {
        rocke_spec_set_reason(reason, reason_cap, "kmap '2row' needs tile_w % 16 == 0 and an even tile_h");
        return false;
    }
    if((s->lds_buffers != 1 && s->lds_buffers != 2)
       || (s->lds_buffers == 2 && !s->prefetch))
    {
        rocke_spec_set_reason(reason, reason_cap, "lds_buffers must be 1, or 2 with prefetch");
        return false;
    }
    if(!s->atomic && !epi_lds)
    {
        rocke_spec_set_reason(reason, reason_cap, "atomic=False needs the lds epilogue");
        return false;
    }
    if(s->lds_wpad < 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "lds_wpad must be >= 0");
        return false;
    }
    bool order_row = strcmp(s->mfma_order, "row") == 0;
    bool order_ks = strcmp(s->mfma_order, "ks") == 0;
    if(!order_row && !order_ks)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "mfma_order %s must be 'row' or 'ks'", s->mfma_order);
        return false;
    }
    const long long eb = 2;
    long long big = eb * p->N * total_c * p->H * p->W;
    long long b2 = eb * p->N * total_k * rocke_direct_conv_problem_Ho(p) * Wo;
    long long b3 = eb * total_k * total_c * p->KH * p->KW;
    if(b2 > big)
    {
        big = b2;
    }
    if(b3 > big)
    {
        big = b3;
    }
    if(big + eb * s->stage_group * p->H * p->W + 64 > kOobBase)
    {
        rocke_spec_set_reason(reason, reason_cap, "tensors must fit below the masked-load offset");
        return false;
    }
    return true;
}

} // namespace

extern "C" rocke_direct_nchw_wgrad_spec_t rocke_direct_nchw_wgrad_spec_default(void)
{
    rocke_direct_nchw_wgrad_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "conv_direct_nchw_wgrad";
    s.tile_m = 64;
    s.tile_c = 64;
    s.tile_h = 4;
    s.tile_w = 32;
    s.waves_m = 2;
    s.waves_c = 2;
    s.waves_t = 1;
    s.y_groups = 1;
    s.stage_group = 4;
    s.lds_pad = 8;
    s.waves_per_eu = 0;
    s.mfma_order = "row";
    s.prefetch = true;
    s.xcd_remap = true;
    s.epilogue = "lds";
    s.kmap = "row";
    s.lds_wpad = 0;
    s.lds_buffers = 1;
    s.atomic = true;
    return s;
}

extern "C" rocke_status_t rocke_direct_nchw_wgrad_kernel_name(
    const rocke_direct_nchw_wgrad_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char pshort[64], t[64], sz[32], w[32], g[32], pad[32], yg[32], wpe[32], wp[32];
    if(rocke_direct_conv_problem_short(p, pshort, sizeof pshort) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(t, sizeof t, "t%dx%d", s->tile_m, s->tile_c);
    snprintf(sz, sizeof sz, "s%dx%d", s->tile_h, s->tile_w);
    snprintf(w, sizeof w, "w%dx%dx%d", s->waves_m, s->waves_c, s->waves_t);
    snprintf(g, sizeof g, "g%d", s->stage_group);
    snprintf(pad, sizeof pad, "p%d", s->lds_pad);
    yg[0] = '\0';
    if(s->y_groups > 1)
    {
        snprintf(yg, sizeof yg, "yg%d", s->y_groups);
    }
    wpe[0] = '\0';
    if(s->waves_per_eu)
    {
        snprintf(wpe, sizeof wpe, "wpe%d", s->waves_per_eu);
    }
    wp[0] = '\0';
    if(s->lds_wpad)
    {
        snprintf(wp, sizeof wp, "wp%d", s->lds_wpad);
    }
    const char* parts[] = {
        pshort,
        t,
        sz,
        w,
        g,
        pad,
        yg,
        wpe,
        strcmp(s->mfma_order, "ks") == 0 ? "oks" : "",
        s->prefetch ? "" : "nopf",
        s->xcd_remap ? "" : "noxcd",
        strcmp(s->epilogue, "lds") == 0 ? "" : "edir",
        strcmp(s->kmap, "2row") == 0 ? "k2r" : "",
        wp,
        s->lds_buffers == 2 ? "db" : "",
        s->atomic ? "" : "st",
        problem_is_bf16(p) ? "bf16" : "",
    };
    return rocke_kernel_name_join(
        s->name, parts, sizeof(parts) / sizeof(parts[0]), NULL, NULL, 0, out, out_cap, NULL);
}

extern "C" bool rocke_direct_nchw_wgrad_is_valid_spec(const rocke_direct_nchw_wgrad_spec_t* s,
                                                      const char* arch,
                                                      char* reason,
                                                      size_t reason_cap)
{
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
        rocke_spec_set_reasonf(reason, reason_cap, "unknown arch %s", arch);
        return false;
    }
    Derived d{};
    if(!validate_host(s, &d, reason, reason_cap))
    {
        return false;
    }
    const char* ab = problem_is_bf16(&s->problem) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(&target->mma, "mma", ab, ab, "fp32", 16, 16, 32))
    {
        rocke_spec_set_reasonf(reason, reason_cap, "missing mfma_f32_16x16x32_%s on %s", ab, arch);
        return false;
    }
    if(target->wave_size != kWave || !target->memory.has_ds_read_tr)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "needs wave64 + ds_read_b64_tr_b16 (absent on %s)", arch);
        return false;
    }
    if(d.threads > rocke_arch_max_threads_per_block(target))
    {
        rocke_spec_set_reason(reason, reason_cap, "too many threads for the arch limit");
        return false;
    }
    if(!rocke_arch_fits_lds(target, d.lds_bytes))
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "LDS %lld B exceeds %d B", d.lds_bytes, target->lds_capacity_bytes);
        return false;
    }
    return true;
}

namespace
{

/* kstep_origin(spec, ks): build-time (row, col) of k step ks (lane group 0's
 * first position). */
std::pair<int, int> kstep_origin(const rocke_direct_nchw_wgrad_spec_t* s, int ks)
{
    int tw = s->tile_w;
    if(strcmp(s->kmap, "2row") == 0)
    {
        int halves = tw / 16;
        return {ks / halves, 16 * (ks % halves)};
    }
    if(tw >= kAk)
    {
        return {(ks * kAk) / tw, (ks * kAk) % tw};
    }
    return {ks * (kAk / tw), 0};
}

/* One dY staging unit: (src, row, v, dst, in_tile). */
struct DyUnit
{
    V src, row, v, dst, in_tile;
};

/* One X staging unit (body or halo): see the Python tuple
 * (src, wr, kind, elems, dsts, width, in_tile); `kind` is `v` (a Value) for a
 * body unit, `(side, wofs)` (host-only) for a halo unit. */
struct XUnit
{
    V src, wr;
    bool is_halo;
    V v; /* body only */
    char side; /* halo only */
    int wofs; /* halo only */
    std::vector<int> elems;
    std::vector<V> dsts;
    int width;
    V in_tile;
};

rocke_kernel_def_t* build_kernel(rocke_ir_builder_t* b,
                                 const rocke_direct_nchw_wgrad_spec_t* spec,
                                 const Derived& d)
{
    Em e{b};
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const char* dtype = p->dtype;
    bool bf16 = problem_is_bf16(p);
    const rocke_type_t* io = rocke_dconv_io_type(b, dtype);
    const int EB = 2;
    long long C = rocke_direct_conv_problem_total_c(p);
    long long K = rocke_direct_conv_problem_total_k(p);
    int H = p->H, W = p->W;
    int Ho = (int)rocke_direct_conv_problem_Ho(p);
    int Wo = (int)rocke_direct_conv_problem_Wo(p);
    int S = p->stride, Pd = p->PAD;
    int KW = p->KW;
    int T = p->KH * KW;
    long long HW = (long long)H * W, HoWo = (long long)Ho * Wo;
    int YG = d.rows_per_group;
    int TM = spec->tile_m, TC = spec->tile_c, TH = spec->tile_h, TW = spec->tile_w;
    int G = spec->stage_group;
    int THREADS = d.threads;
    int MT_W = d.mt_w, CT_W = d.ct_w, TPW = d.taps_per_wave;
    int LDS_H = d.lds_h, LDS_W = d.lds_w_stride, CS = d.c_stride;
    int DYS = d.dy_stride;
    int HALO_L = d.halo_l, HALO_R = d.halo_r;
    int KSTEPS = d.k_steps;
    int NHT = d.n_tiles_h, NWT = d.n_tiles_w;
    long long N_TILES = d.n_tiles;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    if(spec->waves_per_eu)
    {
        const int64_t wpe[2] = {spec->waves_per_eu, 8};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "waves_per_eu", wpe, 2);
    }

    rocke_param_opts_t ro;
    memset(&ro, 0, sizeof(ro));
    ro.noalias = ro.noalias_set = true;
    ro.readonly = ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;
    rocke_param_opts_t wo;
    memset(&wo, 0, sizeof(wo));
    wo.noalias = wo.noalias_set = true;
    wo.align = 4;
    wo.align_set = true;
    V dY = rocke_b_param(b, "dY", rocke_ptr_type(b, io, "global"), &ro);
    V X = rocke_b_param(b, "X", rocke_ptr_type(b, io, "global"), &ro);
    V dW = rocke_b_param(b, "dW", rocke_ptr_type(b, io, "global"), &wo);
    V dY_bytes = rocke_b_param(b, "dY_bytes", rocke_i32(), NULL);
    V X_bytes = rocke_b_param(b, "X_bytes", rocke_i32(), NULL);
    V split = rocke_b_param(b, "split", rocke_i32(), NULL);

    V c0 = e.ci(0), c1 = e.ci(1);
    V oob = e.ci(kOobBase);
    V dy_rsrc = rocke_b_buffer_rsrc(b, dY, dY_bytes);
    V x_rsrc = rocke_b_buffer_rsrc(b, X, X_bytes);

    int NBUF = spec->lds_buffers;
    long long XB = d.x_buf_stride, DB = d.dy_buf_stride;
    long long x_alloc = (long long)NBUF * XB;
    if(d.epi_elems > x_alloc)
    {
        x_alloc = d.epi_elems;
    }
    int x_shape[2] = {1, (int)x_alloc};
    int dy_shape[2] = {1, (int)(NBUF * DB)};
    V x_lds = rocke_b_smem_alloc(b, io, x_shape, 2, "lds_x");
    V dy_lds = rocke_b_smem_alloc(b, io, dy_shape, 2, "lds_dy");
    V x_dump = e.ci(d.lds_x_elems);
    V dy_dump = e.ci(d.lds_dy_elems);

    V tid = rocke_b_thread_id_x(b);
    V lane = e.mod(tid, e.ci(kWave));
    V wave_id = rocke_b_readfirstlane(b, e.div(tid, e.ci(kWave)));
    V wave_m = e.mod(wave_id, e.ci(spec->waves_m));
    V rest_w = e.div(wave_id, e.ci(spec->waves_m));
    V wave_c = e.mod(rest_w, e.ci(spec->waves_c));
    V wave_t = e.div(rest_w, e.ci(spec->waves_c));

    /* grid(1)[0]/[1]: compile-time tile counts (split is host 1 here). */
    int gx1 = (int)(C / TC) * spec->y_groups;
    int gy1 = (int)(K / TM);
    int n_xy = gx1 * gy1;

    V bx, by, z;
    if(spec->xcd_remap)
    {
        V lin = rocke_b_block_id_x(b);
        V n_blocks = e.mul(split, e.ci(n_xy));
        V xcd = e.mod(lin, e.ci(kXcds));
        V pos = e.div(lin, e.ci(kXcds));
        V per_x = e.div(n_blocks, e.ci(kXcds));
        V rem = e.mod(n_blocks, e.ci(kXcds));
        V lin1 = e.mul(xcd, per_x);
        V lin2 = e.smin(xcd, rem);
        V lin3 = e.add(lin1, lin2);
        lin = e.add(lin3, pos);
        bx = e.mod(lin, e.ci(gx1));
        V rest = e.div(lin, e.ci(gx1));
        by = e.mod(rest, e.ci(gy1));
        z = e.div(rest, e.ci(gy1));
    }
    else
    {
        bx = rocke_b_block_id_x(b);
        by = rocke_b_block_id_y(b);
        z = rocke_b_block_id_z(b);
    }
    V c_tile, y0;
    if(spec->y_groups > 1)
    {
        V y_grp = e.mod(bx, e.ci(spec->y_groups));
        c_tile = e.div(bx, e.ci(spec->y_groups));
        y0 = e.muli(y_grp, YG);
    }
    else
    {
        c_tile = bx;
        y0 = c0;
    }
    V m_base = e.muli(by, TM);
    V c_base = e.mul(c_tile, e.ci(TC));

    /* ---- spatial-tile slice of this split ---------------------------- */
    V tN = e.ci(N_TILES);
    V tsub = e.sub(split, c1);
    V tadd = e.add(tN, tsub);
    V per = e.div(tadd, split);
    V t_lo = e.mul(z, per);
    V tadd2 = e.add(t_lo, per);
    V tN2 = e.ci(N_TILES);
    V t_hi = e.smin(tadd2, tN2);

    /* ---- staging units ------------------------------------------------ */
    auto passes = [&](int n_units) {
        std::vector<std::pair<V, V>> out;
        int npass = cdiv(n_units, THREADS);
        for(int j = 0; j < npass; ++j)
        {
            V u = j ? e.add(tid, e.ci(j * THREADS)) : tid;
            V in_tile = NULL;
            if(n_units % THREADS)
            {
                in_tile = e.lt(u, e.ci(n_units));
            }
            out.push_back({u, in_tile});
        }
        return out;
    };

    /* dY units (k, row, v), v fastest. */
    int NVY = TW / kRun;
    std::vector<DyUnit> dy_units;
    for(const auto& up : passes(TM * TH * NVY))
    {
        V u = up.first, in_tile = up.second;
        V v = e.mod(u, e.ci(NVY));
        V r2 = e.div(u, e.ci(NVY));
        V row = e.mod(r2, e.ci(TH));
        V k = e.div(r2, e.ci(TH));
        V dyA = e.add(m_base, k);
        V dyA2 = e.muli(dyA, HoWo);
        V dyB2 = e.muli(v, kRun);
        V src = e.add(dyA2, dyB2);
        V dstX = e.muli(k, DYS);
        V dstY = e.muli(row, TW);
        V dstZ = e.muli(v, kRun);
        V dstW = e.add(dstY, dstZ);
        V dst = e.add(dstX, dstW);
        if(in_tile != NULL)
        {
            dst = e.select(in_tile, dst, dy_dump);
        }
        dy_units.push_back({src, row, v, dst, in_tile});
    }

    /* X body units (cg, wr, v), v fastest. */
    int NVX = S * TW / kRun;
    int NCG = TC / G;
    std::vector<XUnit> x_body;
    for(const auto& up : passes(NCG * LDS_H * NVX))
    {
        V u = up.first, in_tile = up.second;
        V v = e.mod(u, e.ci(NVX));
        V r2 = e.div(u, e.ci(NVX));
        V cg = e.mod(r2, e.ci(NCG));
        V wr = e.div(r2, e.ci(NCG));
        V p1 = e.muli(cg, G);
        V p2 = e.add(c_base, p1);
        V p3 = e.muli(p2, HW);
        V p4 = e.muli(v, kRun);
        V src = e.add(p3, p4);
        V q1 = e.muli(wr, LDS_W);
        V q2 = e.muli(v, kRun);
        V q3 = e.ci(HALO_L);
        V q4 = e.add(q2, q3);
        V q5 = e.add(q1, q4);
        V q7 = e.muli(q5, CS);
        V q8 = e.muli(cg, G);
        V dst0 = e.add(q7, q8);
        V step = NULL;
        if(in_tile != NULL)
        {
            dst0 = e.select(in_tile, dst0, x_dump);
            step = e.select(in_tile, e.ci(CS), c0);
        }
        std::vector<int> elems;
        std::vector<V> dsts;
        for(int ee = 0; ee < kRun; ++ee)
        {
            V dd;
            if(ee == 0)
            {
                dd = dst0;
            }
            else if(step == NULL)
            {
                dd = e.addi(dst0, ee * CS);
            }
            else
            {
                dd = e.add(dst0, e.muli(step, ee));
            }
            elems.push_back(ee);
            dsts.push_back(dd);
        }
        x_body.push_back({src, wr, false, v, 0, 0, elems, dsts, kRun, in_tile});
    }

    /* X halo units (cg, wr) per side: aligned loads, only the |d| columns
     * next to the body are written. */
    std::vector<XUnit> x_halo;
    struct Side
    {
        char ch;
        int h;
    };
    for(const Side& side : {Side{'l', HALO_L}, Side{'r', HALO_R}})
    {
        if(!side.h)
        {
            continue;
        }
        int hv = halo_vec(side.h);
        for(const auto& up : passes(NCG * LDS_H))
        {
            V u = up.first, in_tile = up.second;
            V cg = e.mod(u, e.ci(NCG));
            V wr = e.div(u, e.ci(NCG));
            V r1 = e.muli(cg, G);
            V r2 = e.add(c_base, r1);
            V src = e.muli(r2, HW);
            std::vector<int> elems;
            int col0, wofs;
            if(side.ch == 'l')
            {
                for(int i = hv - side.h; i < hv; ++i)
                {
                    elems.push_back(i);
                }
                col0 = 0;
                wofs = -hv;
            }
            else
            {
                for(int i = 0; i < side.h; ++i)
                {
                    elems.push_back(i);
                }
                col0 = HALO_L + S * TW;
                wofs = S * TW;
            }
            V s1 = e.muli(wr, LDS_W);
            V s2 = e.ci(col0);
            V s3 = e.add(s1, s2);
            V s5 = e.muli(s3, CS);
            V s6 = e.muli(cg, G);
            V base = e.add(s5, s6);
            std::vector<V> dsts;
            for(size_t i = 0; i < elems.size(); ++i)
            {
                V dd = i ? e.addi(base, (int)i * CS) : base;
                if(in_tile != NULL)
                {
                    dd = e.select(in_tile, dd, x_dump);
                }
                dsts.push_back(dd);
            }
            x_halo.push_back({src, wr, true, NULL, side.ch, wofs, elems, dsts, hv, in_tile});
        }
    }

    std::vector<XUnit> x_units;
    x_units.reserve(x_body.size() + x_halo.size());
    x_units.insert(x_units.end(), x_body.begin(), x_body.end());
    x_units.insert(x_units.end(), x_halo.begin(), x_halo.end());

    auto tile_origin = [&](V t, V* n, V* ho0, V* wo0) {
        V tw_i = e.mod(t, e.ci(NWT));
        V r2 = e.div(t, e.ci(NWT));
        V th_i = e.mod(r2, e.ci(NHT));
        *n = e.div(r2, e.ci(NHT));
        *ho0 = e.muli(th_i, TH);
        *wo0 = e.muli(tw_i, TW);
    };

    auto issue_loads = [&](V t, std::vector<V>* dys_out, std::vector<V>* xs_out) {
        V n, ho0, wo0;
        tile_origin(t, &n, &ho0, &wo0);
        V live = e.lt(t, t_hi);
        V dyi1 = e.muli(n, (int64_t)((long long)K * HoWo));
        V dyi2 = e.muli(ho0, Wo);
        V dyi3 = e.add(dyi2, wo0);
        V dy_img = e.add(dyi1, dyi3);
        std::vector<V> dys;
        for(const DyUnit& u : dy_units)
        {
            V ho = e.add(ho0, u.row);
            V wo = e.add(wo0, e.muli(u.v, kRun));
            V okv1 = e.lti(ho, Ho);
            V okv2 = e.lti(wo, Wo);
            V okv3 = e.land(okv1, okv2);
            V ok = e.land(live, okv3);
            if(u.in_tile != NULL)
            {
                ok = e.land(ok, u.in_tile);
            }
            V evw1 = e.muli(u.row, Wo);
            V evw2 = e.add(u.src, evw1);
            V ev = e.add(dy_img, evw2);
            V off = e.select(ok, e.muli(ev, EB), oob);
            dys.push_back(rocke_b_buffer_load_vN(b, dy_rsrc, off, c0, io, kRun));
        }
        V hh1 = e.muli(ho0, S);
        V hh2 = e.subi(hh1, Pd);
        V h0 = e.add(hh2, y0);
        V w0 = e.muli(wo0, S);
        V xi1 = e.muli(n, (int64_t)((long long)C * HW));
        V x_img = e.add(xi1, w0);
        std::vector<V> xs;
        for(const XUnit& u : x_units)
        {
            V h = e.add(h0, u.wr);
            V ok = live;
            V okx1 = e.ge(h, c0);
            V okx2 = e.lti(h, H);
            V okx3 = e.land(okx1, okx2);
            ok = e.land(ok, okx3);
            V ev;
            if(!u.is_halo)
            {
                V col = e.muli(u.v, kRun);
                V w = e.add(w0, col);
                ok = e.land(ok, e.lti(w, W));
                V evb1 = e.add(x_img, u.src);
                V evb2 = e.muli(h, W);
                ev = e.add(evb1, evb2);
            }
            else
            {
                V w = e.addi(w0, u.wofs);
                V wok = u.side == 'l' ? e.ge(w, c0) : e.lti(w, W);
                ok = e.land(ok, wok);
                V evh1 = e.add(x_img, u.src);
                V evh2 = e.muli(h, W);
                V evh3 = e.add(evh1, evh2);
                ev = e.addi(evh3, u.wofs);
            }
            if(u.in_tile != NULL)
            {
                ok = e.land(ok, u.in_tile);
            }
            V off = e.select(ok, e.muli(ev, EB), oob);
            for(int g = 0; g < G; ++g)
            {
                xs.push_back(rocke_b_buffer_load_vN(
                    b, x_rsrc, off, e.ci((long long)g * HW * EB), io, u.width));
            }
        }
        *dys_out = dys;
        *xs_out = xs;
    };

    auto commit = [&](const std::vector<V>& dys, const std::vector<V>& xs, V yo, V xo) {
        for(size_t i = 0; i < dy_units.size(); ++i)
        {
            V dst = dy_units[i].dst;
            V d2 = yo == NULL ? dst : e.add(dst, yo);
            e.smem_store(dy_lds, c0, d2, dys[i], kRun);
        }
        size_t i = 0;
        for(const XUnit& u : x_units)
        {
            std::vector<V> runs(xs.begin() + i, xs.begin() + i + G);
            i += G;
            for(size_t k = 0; k < u.elems.size(); ++k)
            {
                int elem = u.elems[k];
                std::vector<V> vals;
                for(V r : runs)
                {
                    vals.push_back(rocke_b_vec_extract(b, r, elem));
                }
                V vec = rocke_b_vec_pack(b, vals.data(), (int)vals.size(), io);
                V d2 = u.dsts[k];
                if(xo != NULL)
                {
                    d2 = e.add(d2, xo);
                }
                e.smem_store(x_lds, c0, d2, vec, G);
            }
        }
    };

    /* ---- MFMA fragment addresses --------------------------------------- */
    V lane_lo = e.mod(lane, e.ci(kAtom));
    V lane_g = e.div(lane, e.ci(kAtom));
    V lq1 = e.div(lane, e.ci(4));
    V lane_q = e.modi(lq1, 4);
    V lane_c4 = e.mod(lane, e.ci(4));
    V rowg, colg;
    if(strcmp(spec->kmap, "2row") == 0)
    {
        rowg = e.muli(e.mod(lane_g, e.ci(2)), TH / 2);
        colg = e.muli(e.div(lane_g, e.ci(2)), kRun);
    }
    else if(TW >= kAk)
    {
        rowg = c0;
        colg = e.muli(lane_g, kRun);
    }
    else
    {
        V rg1 = e.muli(lane_g, kRun);
        rowg = e.divi(rg1, TW);
        V cg1 = e.muli(lane_g, kRun);
        colg = e.modi(cg1, TW);
    }
    V aA1 = e.muli(wave_m, MT_W * kAtom);
    V aA2 = e.add(aA1, lane_lo);
    V aA3 = e.muli(aA2, DYS);
    V aB1 = e.muli(rowg, TW);
    V aB2 = e.add(aB1, colg);
    V a_base = e.add(aA3, aB2);
    V tap0 = e.muli(wave_t, TPW);
    V y_w = e.div(tap0, e.ci(KW));
    V x_w = e.mod(tap0, e.ci(KW));
    V bC1 = e.muli(rowg, S);
    V bC2 = e.add(bC1, y_w);
    V bC3 = e.muli(bC2, LDS_W);
    V bC4 = e.add(colg, lane_q);
    V bC5 = e.muli(bC4, S);
    V bC6 = e.add(bC5, x_w);
    V bC7 = e.add(bC3, bC6);
    V bC8 = e.muli(bC7, CS);
    V bC9 = e.muli(lane_c4, 4);
    V bC10 = e.add(bC8, bC9);
    V bC11 = e.muli(wave_c, CT_W * kAtom);
    V b_base = e.add(bC10, bC11);

    auto mfma = [&](V a, V bb, V acc) -> V {
        return bf16 ? rocke_b_mfma_f32_16x16x32_bf16(b, a, bb, acc)
                    : rocke_b_mfma_f32_16x16x32_f16(b, a, bb, acc);
    };
    auto acc_i = [&](int mt, int t, int ct) { return (mt * TPW + t) * CT_W + ct; };
    auto trunc = [&](V v) -> V {
        return bf16 ? rocke_b_trunc_f32_to_bf16(b, v) : rocke_b_trunc_f32_to_f16(b, v);
    };

    /* Fragment-read bases of the buffer the MFMAs of this tile read. */
    V cur_a = a_base;
    V cur_b = NULL; /* set to b_base below */

    auto read_a = [&](int ks, int mt) -> V {
        auto rc = kstep_origin(spec, ks);
        int off = mt * kAtom * DYS + rc.first * TW + rc.second;
        return e.smem_load(dy_lds, c0, e.addi(cur_a, off), io, kRun);
    };
    auto read_b = [&](int wrow, int ck, int xx, int ct) -> V {
        V frs[2];
        for(int rd = 0; rd < 2; ++rd)
        {
            int off = (wrow * LDS_W + (ck + rd * 4) * S + xx) * CS + ct * kAtom;
            frs[rd] = e.ds_read_tr(x_lds, c0, e.addi(cur_b, off), io);
        }
        return rocke_b_vec_concat(b, frs[0], frs[1]);
    };

    auto emit_mfmas = [&](std::vector<V>& accs) {
        if(strcmp(spec->mfma_order, "ks") == 0)
        {
            for(int ks = 0; ks < KSTEPS; ++ks)
            {
                std::vector<V> a_fr;
                for(int mt = 0; mt < MT_W; ++mt)
                {
                    a_fr.push_back(read_a(ks, mt));
                }
                auto rc = kstep_origin(spec, ks);
                int rk = rc.first, ck = rc.second;
                for(int t = 0; t < TPW; ++t)
                {
                    for(int ct = 0; ct < CT_W; ++ct)
                    {
                        V b_fr = read_b(rk * S + t / KW, ck, t % KW, ct);
                        for(int mt = 0; mt < MT_W; ++mt)
                        {
                            int i = acc_i(mt, t, ct);
                            accs[i] = mfma(a_fr[mt], b_fr, accs[i]);
                        }
                    }
                }
            }
            return;
        }
        /* "row": one X fragment per (window row, column phase, tap column, c
         * atom), shared by every (k step, filter row) pair that lands on it. */
        std::map<std::pair<int, int>, V> a_fr;
        std::map<std::tuple<int, int, int>, std::vector<std::pair<int, int>>> uses;
        for(int ks = 0; ks < KSTEPS; ++ks)
        {
            auto rc = kstep_origin(spec, ks);
            int rk = rc.first, ck = rc.second;
            for(int t = 0; t < TPW; ++t)
            {
                uses[{rk * S + t / KW, ck, t % KW}].push_back({ks, t});
            }
        }
        for(const auto& kv : uses)
        {
            int wrow = std::get<0>(kv.first), ck = std::get<1>(kv.first), xx = std::get<2>(kv.first);
            for(int ct = 0; ct < CT_W; ++ct)
            {
                V b_fr = read_b(wrow, ck, xx, ct);
                for(const auto& kst : kv.second)
                {
                    int ks = kst.first, t = kst.second;
                    for(int mt = 0; mt < MT_W; ++mt)
                    {
                        auto key = std::make_pair(ks, mt);
                        if(a_fr.find(key) == a_fr.end())
                        {
                            a_fr[key] = read_a(ks, mt);
                        }
                        int i = acc_i(mt, t, ct);
                        accs[i] = mfma(a_fr[key], b_fr, accs[i]);
                    }
                }
            }
        }
    };

    int n_acc = MT_W * TPW * CT_W;
    std::vector<std::string> names;
    std::vector<V> inits;
    for(int i = 0; i < n_acc; ++i)
    {
        names.push_back("acc" + std::to_string(i));
        inits.push_back(rocke_b_zero_vec_f32(b, 4));
    }
    std::vector<V> dy0, xs0, dy1, xs1;
    size_t n_dy = 0;
    if(spec->prefetch)
    {
        issue_loads(t_lo, &dy0, &xs0);
        n_dy = dy0.size();
        for(size_t i = 0; i < dy0.size(); ++i)
        {
            names.push_back("y" + std::to_string(i));
            inits.push_back(dy0[i]);
        }
        for(size_t i = 0; i < xs0.size(); ++i)
        {
            names.push_back("x" + std::to_string(i));
            inits.push_back(xs0[i]);
        }
    }
    cur_b = b_base;
    if(NBUF == 2)
    {
        commit(dy0, xs0, NULL, NULL);
        issue_loads(e.add(t_lo, c1), &dy1, &xs1);
        names.resize((size_t)n_acc);
        inits.resize((size_t)n_acc);
        for(size_t i = 0; i < dy1.size(); ++i)
        {
            names.push_back("y" + std::to_string(i));
            inits.push_back(dy1[i]);
        }
        for(size_t i = 0; i < xs1.size(); ++i)
        {
            names.push_back("x" + std::to_string(i));
            inits.push_back(xs1[i]);
        }
        n_dy = dy1.size();
        rocke_b_sync_fence(b, "workgroup");
    }

    std::vector<V> accs_out;
    size_t n_x = 0;
    auto body_fn = [&](V tv, const std::vector<V>& carried) -> std::vector<V> {
        std::vector<V> accs(carried.begin(), carried.begin() + n_acc);
        if(NBUF == 2)
        {
            std::vector<V> dys(carried.begin() + n_acc, carried.begin() + n_acc + (int)n_dy);
            std::vector<V> xs(carried.begin() + n_acc + (int)n_dy, carried.end());
            V par = e.land(e.sub(tv, t_lo), c1);
            V nxt = e.sub(c1, par);
            V dbv = e.ci(DB);
            V yo_mul = e.mul(nxt, dbv);
            V xbv = e.ci(XB);
            V xo_mul = e.mul(nxt, xbv);
            commit(dys, xs, yo_mul, xo_mul);
            std::vector<V> dyn, xsn;
            issue_loads(e.addi(tv, 2), &dyn, &xsn);
            rocke_b_sched_barrier(b, 0);
            cur_a = e.add(a_base, e.mul(par, e.ci(DB)));
            cur_b = e.add(b_base, e.mul(par, e.ci(XB)));
            emit_mfmas(accs);
            rocke_b_s_waitcnt(b, -1, 0, -1);
            rocke_b_s_barrier_bare(b);
            std::vector<V> out = accs;
            out.insert(out.end(), dyn.begin(), dyn.end());
            out.insert(out.end(), xsn.begin(), xsn.end());
            return out;
        }
        std::vector<V> dys, xs;
        if(spec->prefetch)
        {
            dys.assign(carried.begin() + n_acc, carried.begin() + n_acc + (int)n_dy);
            xs.assign(carried.begin() + n_acc + (int)n_dy, carried.end());
        }
        else
        {
            issue_loads(tv, &dys, &xs);
        }
        rocke_b_sync_fence(b, "workgroup");
        commit(dys, xs, NULL, NULL);
        rocke_b_sync_fence(b, "workgroup");
        if(spec->prefetch)
        {
            std::vector<V> dyn, xsn;
            issue_loads(e.add(tv, c1), &dyn, &xsn);
            rocke_b_sched_barrier(b, 0);
            emit_mfmas(accs);
            std::vector<V> out = accs;
            out.insert(out.end(), dyn.begin(), dyn.end());
            out.insert(out.end(), xsn.begin(), xsn.end());
            return out;
        }
        emit_mfmas(accs);
        return accs;
    };
    n_x = (names.size() - (size_t)n_acc) - n_dy;
    (void)n_x;
    std::vector<V> loop_results
        = e.for_iter(t_lo, t_hi, c1, names, inits, "tile", body_fn);
    accs_out.assign(loop_results.begin(), loop_results.begin() + n_acc);

    /* ---- epilogue: packed 16-bit atomics into KCYX dW --------------------- */
    V t0 = e.add(e.muli(y0, KW), tap0);

    auto pairs_for = [&](int par) {
        std::vector<int> out;
        for(int s = -1; s < TPW; ++s)
        {
            if((par + s) % 2 == 0)
            {
                out.push_back(s);
            }
        }
        return out;
    };
    std::vector<int> p0v = pairs_for(0), p1v = pairs_for(1);
    size_t n_pairs = std::max(p0v.size(), p1v.size());
    std::vector<bool> pad0(n_pairs, false), pad1(n_pairs, false);
    for(size_t i = p0v.size(); i < n_pairs; ++i)
    {
        pad0[i] = true;
    }
    for(size_t i = p1v.size(); i < n_pairs; ++i)
    {
        pad1[i] = true;
    }
    int p0_first = p0v.empty() ? 0 : p0v[0];
    int p1_first = p1v.empty() ? 0 : p1v[0];
    while(p0v.size() < n_pairs)
    {
        p0v.push_back(p0_first);
    }
    while(p1v.size() < n_pairs)
    {
        p1v.push_back(p1_first);
    }

    V zero16 = rocke_b_vec_extract(b, rocke_b_zero_vec(b, io, 2), 0);
    V has_work = e.lt(t_lo, t_hi);
    V dw_rsrc = NULL;
    if(!spec->atomic)
    {
        dw_rsrc = rocke_b_buffer_rsrc(b, dW, e.ci((long long)EB * K * C * T));
    }
    auto atomic_add = [&](V idx, V pk) {
        if(!spec->atomic)
        {
            rocke_b_buffer_store_vN(b, dw_rsrc, e.mul(idx, e.ci(EB)), c0, pk);
            return;
        }
        if(bf16)
        {
            rocke_b_global_atomic_add_pk_bf16(b, dW, idx, pk, "monotonic");
        }
        else
        {
            rocke_b_global_atomic_add_pk_f16(b, dW, idx, pk, "monotonic");
        }
    };

    e.if_(has_work, [&]() {
        V m_wave = e.add(m_base, e.muli(wave_m, MT_W * kAtom));
        if(strcmp(spec->epilogue, "lds") == 0)
        {
            int WG_T = d.wg_taps;
            int WC16 = spec->waves_c * kAtom;
            int RUN_D = kAtom * WG_T / 2;
            int n_dw = spec->waves_m * kAtom * spec->waves_c * RUN_D;
            V row0a = e.muli(wave_m, kAtom);
            V row0b = e.muli(lane_g, 4);
            V row0 = e.add(row0a, row0b);
            V col = e.add(e.muli(wave_c, kAtom), lane_lo);
            for(int mt = 0; mt < MT_W; ++mt)
            {
                for(int ct = 0; ct < CT_W; ++ct)
                {
                    rocke_b_sync_fence(b, "workgroup");
                    for(int i = 0; i < 4; ++i)
                    {
                        V bi1 = e.addi(row0, i);
                        V bi2 = e.muli(bi1, WC16);
                        V bi3 = e.add(bi2, col);
                        V bi4 = e.muli(bi3, WG_T);
                        V base_i = e.add(bi4, tap0);
                        for(int t = 0; t < TPW; ++t)
                        {
                            V v = trunc(rocke_b_vec_extract(b, accs_out[acc_i(mt, t, ct)], i));
                            e.smem_store(x_lds, c0, e.addi(base_i, t), v, 1);
                        }
                    }
                    rocke_b_sync_fence(b, "workgroup");
                    int n_passes = cdiv(n_dw, THREADS);
                    for(int j = 0; j < n_passes; ++j)
                    {
                        V dd = j ? e.add(tid, e.ci(j * THREADS)) : tid;
                        V seg = e.div(dd, e.ci(RUN_D));
                        V off = e.mod(dd, e.ci(RUN_D));
                        V row = e.div(seg, e.ci(spec->waves_c));
                        V wcb = e.mod(seg, e.ci(spec->waves_c));
                        V l1 = e.muli(row, WC16);
                        V l2 = e.muli(wcb, kAtom);
                        V l3 = e.add(l1, l2);
                        V l4 = e.muli(l3, WG_T);
                        V l5 = e.muli(off, 2);
                        V lds_i = e.add(l4, l5);
                        V k1 = e.addi(m_base, mt * kAtom);
                        V k2 = e.divi(row, kAtom);
                        V k3 = e.muli(k2, MT_W * kAtom);
                        V k4 = e.modi(row, kAtom);
                        V k5 = e.add(k3, k4);
                        V k = e.add(k1, k5);
                        V cg1 = e.addi(c_base, ct * kAtom);
                        V cg2 = e.muli(wcb, CT_W * kAtom);
                        V cg = e.add(cg1, cg2);
                        V i1 = e.muli(k, (int64_t)((long long)C * T));
                        V i2 = e.muli(cg, T);
                        V i3 = e.add(i1, i2);
                        V i4 = e.muli(off, 2);
                        V idx = e.add(i3, i4);
                        if(n_dw % THREADS)
                        {
                            e.if_(e.lt(dd, e.ci(n_dw)), [&]() {
                                atomic_add(idx, e.smem_load(x_lds, c0, lds_i, io, 2));
                            });
                        }
                        else
                        {
                            atomic_add(idx, e.smem_load(x_lds, c0, lds_i, io, 2));
                        }
                    }
                }
            }
        }
        else
        {
            for(int ct = 0; ct < CT_W; ++ct)
            {
                V c1 = e.muli(wave_c, CT_W * kAtom);
                V c2 = e.add(c_base, c1);
                V c3 = e.ci(ct * kAtom);
                V c4 = e.add(c3, lane_lo);
                V c = e.add(c2, c4);
                V ctap = e.add(e.mul(c, e.ci(T)), t0);
                V odd = rocke_b_cmp_ne(b, e.mod(ctap, e.ci(2)), c0);
                for(int mt = 0; mt < MT_W; ++mt)
                {
                    for(int i = 0; i < 4; ++i)
                    {
                        V kk1 = e.addi(m_wave, mt * kAtom + i);
                        V kk2 = e.muli(lane_g, 4);
                        V k = e.add(kk1, kk2);
                        V flat0 = e.add(e.mul(k, e.ci((long long)C * T)), ctap);
                        std::vector<V> vals;
                        for(int t = 0; t < TPW; ++t)
                        {
                            vals.push_back(trunc(rocke_b_vec_extract(b, accs_out[acc_i(mt, t, ct)], i)));
                        }
                        auto vv = [&](int t) -> V { return (t >= 0 && t < TPW) ? vals[t] : zero16; };
                        for(size_t q = 0; q < n_pairs; ++q)
                        {
                            int s0 = p0v[q], s1 = p1v[q];
                            V lo = e.select(odd, pad1[q] ? zero16 : vv(s1), pad0[q] ? zero16 : vv(s0));
                            V hi = e.select(
                                odd, pad1[q] ? zero16 : vv(s1 + 1), pad0[q] ? zero16 : vv(s0 + 1));
                            V idxHi = e.addi(flat0, s1);
                            V idxLo = e.addi(flat0, s0);
                            V idx = e.select(odd, idxHi, idxLo);
                            V pk[2] = {lo, hi};
                            atomic_add(idx, rocke_b_vec_pack(b, pk, 2, io));
                        }
                    }
                }
            }
        }
    });

    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_direct_nchw_wgrad(rocke_ir_builder_t* b,
                                                              const rocke_direct_nchw_wgrad_spec_t* spec,
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
    if(!rocke_direct_nchw_wgrad_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid DirectNchwWgradSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Derived d{};
    char dummy[ROCKE_ERR_MSG_CAP];
    if(!validate_host(spec, &d, dummy, sizeof dummy))
    {
        return NULL;
    }
    return build_kernel(b, spec, d);
}

extern "C" rocke_kernel_def_t*
    rocke_build_direct_nchw_wgrad_new(rocke_ir_builder_t* b,
                                      const rocke_direct_nchw_wgrad_spec_t* spec,
                                      const char* arch)
{
    return rocke_dwg::build_new(
        b,
        spec != NULL,
        "build_direct_conv_nchw_wgrad",
        [&](char* out, size_t cap) { return rocke_direct_nchw_wgrad_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_direct_nchw_wgrad(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_direct_nchw_wgrad_lower_to_llvm(const rocke_direct_nchw_wgrad_spec_t* spec,
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
        "build_direct_conv_nchw_wgrad failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_direct_nchw_wgrad_new(b, spec, arch); });
}
