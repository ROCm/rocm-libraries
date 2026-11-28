// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_nchw_dgrad.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_nchw_dgrad.py: the native-NCHW direct
 * dgrad convolution (build_direct_conv_nchw_dgrad). dY is consumed in NKHW and
 * dX produced in NCHW, directly -- no host transpose of the activations.
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the byte-identity contract at the top of
 * rocke/instance_conv_direct_wgrad_internal.hpp, which this port reuses (the
 * `Em` helpers: at most one emitting argument per call).
 */
#include "rocke/instance_conv_direct_nchw_dgrad.h"

#include <string.h>

#include <map>
#include <string>
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

constexpr int kAlign = 8; /* elements in 16 bytes of a 16-bit dtype */
constexpr int32_t kOobBase = 0x7F000000;

struct Atom
{
    int tile, k, frag;
};

bool atom_lookup(const char* atom, Atom* out)
{
    if(atom == NULL)
    {
        return false;
    }
    if(strcmp(atom, "32x32x16") == 0)
    {
        *out = {32, 16, 8};
        return true;
    }
    if(strcmp(atom, "32x32x8") == 0)
    {
        *out = {32, 8, 4};
        return true;
    }
    if(strcmp(atom, "16x16x32") == 0)
    {
        *out = {16, 32, 8};
        return true;
    }
    if(strcmp(atom, "16x16x16") == 0)
    {
        *out = {16, 16, 4};
        return true;
    }
    return false;
}

int halo_vec(int h)
{
    int v = 2;
    while(v < h)
    {
        v *= 2;
    }
    return v;
}

/* phase_taps(n_taps, pad, stride): per stride phase, the (tap, d) pairs that
 * reach it. Since every pair here has an exact ((ph + pad - t) % stride == 0)
 * division, C++'s truncating `/` agrees with Python's floor `//`. */
std::vector<std::vector<std::pair<int, int>>> phase_taps(int n_taps, int pad, int stride)
{
    std::vector<std::vector<std::pair<int, int>>> out;
    for(int ph = 0; ph < stride; ++ph)
    {
        std::vector<std::pair<int, int>> row;
        for(int t = 0; t < n_taps; ++t)
        {
            int num = ph + pad - t;
            if(num % stride == 0)
            {
                row.push_back({t, num / stride});
            }
        }
        out.push_back(row);
    }
    return out;
}

/* _span(taps): (min, max) of every d over every phase; (0, 0) if empty. */
void span(const std::vector<std::vector<std::pair<int, int>>>& taps, int* lo, int* hi)
{
    bool any = false;
    int mn = 0, mx = 0;
    for(const auto& ph : taps)
    {
        for(const auto& td : ph)
        {
            int d = td.second;
            if(!any)
            {
                mn = mx = d;
                any = true;
            }
            else
            {
                if(d < mn)
                {
                    mn = d;
                }
                if(d > mx)
                {
                    mx = d;
                }
            }
        }
    }
    *lo = mn;
    *hi = mx;
}

/* Every host-computed (compile-time) geometry constant of the spec: the C++
 * twin of DirectNchwDgradSpec's @property methods, computed once the atom is
 * known valid. */
struct Derived
{
    int atom_tile, atom_k, frag;
    int acc_per_lane;
    int n_col_blocks;
    int m_tiles_total;
    int m_tiles_per_wave;
    int rows_per_wave;
    int k_atoms;
    int threads_per_block;
    int d_h_lo, d_h_hi;
    int d_w_lo, d_w_hi;
    int halo_l, halo_r;
    int lds_h, lds_w;
    int c_stride;
    long long lds_x_elems;
    long long lds_w_elems;
    long long lds_bytes;
    int n_phases;
    long long acc_vgprs;
    std::vector<std::vector<std::pair<int, int>>> taps_h, taps_w;
};

Derived derive(const rocke_direct_nchw_dgrad_spec_t* s, const Atom& a)
{
    Derived d{};
    const rocke_direct_conv_problem_t* p = &s->problem;
    int S = p->stride;
    d.atom_tile = a.tile;
    d.atom_k = a.k;
    d.frag = a.frag;
    d.acc_per_lane = a.tile * a.tile / s->wave_size;
    d.n_col_blocks = s->tile_j / a.tile;
    d.m_tiles_total = s->tile_c / a.tile;
    d.m_tiles_per_wave = d.m_tiles_total / s->waves_m;
    d.rows_per_wave = s->tile_i / s->waves_n;
    d.k_atoms = s->ck / a.k;
    d.threads_per_block = s->waves_m * s->waves_n * s->wave_size;
    d.taps_h = phase_taps(p->KH, p->PAD, S);
    int pad_w = rocke_direct_conv_problem_pad_w(p);
    d.taps_w = phase_taps(p->KW, pad_w, S);
    span(d.taps_h, &d.d_h_lo, &d.d_h_hi);
    span(d.taps_w, &d.d_w_lo, &d.d_w_hi);
    d.lds_h = s->tile_i + d.d_h_hi - d.d_h_lo;
    d.lds_w = s->tile_j + d.d_w_hi - d.d_w_lo;
    d.halo_l = d.d_w_lo < 0 ? -d.d_w_lo : 0;
    d.halo_r = d.d_w_hi > 0 ? d.d_w_hi : 0;
    d.c_stride = s->ck + s->lds_pad;
    d.lds_x_elems = (long long)d.lds_h * d.lds_w * d.c_stride;
    d.lds_w_elems = (long long)s->tile_c * p->KH * p->KW * s->ck;
    d.lds_bytes = 2 * (d.lds_x_elems + kAlign + d.lds_w_elems + d.frag);
    d.n_phases = S * S;
    d.acc_vgprs = (long long)d.n_phases * d.rows_per_wave * d.n_col_blocks
                  * d.m_tiles_per_wave * d.acc_per_lane;
    return d;
}

/* spec.validate(): host-only checks (raises ValueError in Python); returns
 * false + reason on the first that fails, in Python's order. On success fills
 * *atom and *derived. */
bool validate_host(const rocke_direct_nchw_dgrad_spec_t* s,
                   Atom* atom,
                   Derived* derived,
                   char* reason,
                   size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(!problem_dtype_ok(p))
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "unsupported dtype %s (16-bit storage only)", p->dtype);
        return false;
    }
    if(p->groups != 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "groups==1 only");
        return false;
    }
    if(p->dil_h != 1 || p->dil_w != 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "dilation is not supported");
        return false;
    }
    if(p->stride != 1 && p->stride != 2)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "stride %d not supported (1 or 2)", p->stride);
        return false;
    }
    Atom a;
    if(!atom_lookup(s->atom, &a))
    {
        rocke_spec_set_reasonf(reason, reason_cap, "unknown atom %s", s->atom ? s->atom : "(null)");
        return false;
    }
    *atom = a;
    Derived d = derive(s, a);
    *derived = d;
    int t = d.atom_tile;
    if(s->tile_j % t != 0 || s->tile_j % kAlign != 0)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "tile_j must be a multiple of %d and %d", t, kAlign);
        return false;
    }
    if(s->tile_c % t != 0)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "tile_c must be a multiple of %d", t);
        return false;
    }
    if(d.m_tiles_total % s->waves_m != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tile_c / atom not divisible by waves_m");
        return false;
    }
    if(s->tile_i % s->waves_n != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tile_i not divisible by waves_n");
        return false;
    }
    if(s->ck % d.atom_k != 0 || s->ck % kAlign != 0)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "ck must be a multiple of %d and %d", d.atom_k, kAlign);
        return false;
    }
    if((s->stage_group != 1 && s->stage_group != 2 && s->stage_group != 4
        && s->stage_group != 8)
       || s->ck % s->stage_group != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "stage_group must be 1/2/4/8 and divide ck");
        return false;
    }
    if(s->lds_pad < 0 || s->lds_pad % kAlign != 0)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "lds_pad must be a non-negative multiple of %d", kAlign);
        return false;
    }
    if(p->kpg % s->ck != 0)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "K %d not divisible by ck %d", p->kpg, s->ck);
        return false;
    }
    long long Wo = rocke_direct_conv_problem_Wo(p);
    if(Wo % kAlign != 0)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "Wo %lld must be a multiple of %d (whole dY runs)", Wo, kAlign);
        return false;
    }
    if(!(d.d_w_lo <= 0 && 0 <= d.d_w_hi))
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "column tap offsets (%d, %d) must straddle 0", d.d_w_lo, d.d_w_hi);
        return false;
    }
    int halo_max = d.halo_l > d.halo_r ? d.halo_l : d.halo_r;
    if(halo_max > kAlign)
    {
        rocke_spec_set_reason(reason, reason_cap, "halo wider than 8 columns");
        return false;
    }
    if(d.threads_per_block > 1024)
    {
        rocke_spec_set_reason(reason, reason_cap, "too many threads");
        return false;
    }
    if(d.acc_vgprs > 256)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "accumulators need %lld registers per lane", d.acc_vgprs);
        return false;
    }
    const long long eb = 2;
    long long total_c = rocke_direct_conv_problem_total_c(p);
    long long total_k = rocke_direct_conv_problem_total_k(p);
    long long Ho = rocke_direct_conv_problem_Ho(p);
    long long big = eb * p->N * total_k * Ho * Wo;
    long long b2 = eb * total_c * total_k * p->KH * p->KW;
    long long b3 = eb * p->N * total_c * p->H * p->W;
    if(b2 > big)
    {
        big = b2;
    }
    if(b3 > big)
    {
        big = b3;
    }
    if(big + eb * s->ck * Ho * Wo > kOobBase)
    {
        rocke_spec_set_reason(reason, reason_cap, "tensors must fit below the masked-access offset");
        return false;
    }
    return true;
}

} // namespace

extern "C" rocke_direct_nchw_dgrad_spec_t rocke_direct_nchw_dgrad_spec_default(void)
{
    rocke_direct_nchw_dgrad_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "conv_direct_nchw_dgrad";
    s.tile_i = 8;
    s.tile_j = 32;
    s.tile_c = 64;
    s.ck = 32;
    s.waves_m = 2;
    s.waves_n = 2;
    s.atom = "32x32x16";
    s.wave_size = 64;
    s.lds_pad = 8;
    s.stage_group = 8;
    return s;
}

extern "C" rocke_status_t rocke_direct_nchw_dgrad_kernel_name(
    const rocke_direct_nchw_dgrad_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f[64], t[64], ck[32], w[32], a[48], g[32], lp[32];
    snprintf(f, sizeof f, "f%dx%ds%dp%d", p->KH, p->KW, p->stride, p->PAD);
    snprintf(t, sizeof t, "t%dx%dx%d", s->tile_i, s->tile_j, s->tile_c);
    snprintf(ck, sizeof ck, "ck%d", s->ck);
    snprintf(w, sizeof w, "w%dx%d", s->waves_m, s->waves_n);
    snprintf(a, sizeof a, "a%s", s->atom ? s->atom : "");
    snprintf(g, sizeof g, "g%d", s->stage_group);
    snprintf(lp, sizeof lp, "lp%d", s->lds_pad);
    const char* parts[] = {
        f,
        t,
        ck,
        w,
        a,
        g,
        lp,
        problem_is_bf16(p) ? "bf16" : "",
    };
    return rocke_kernel_name_join(
        s->name, parts, sizeof(parts) / sizeof(parts[0]), NULL, NULL, 0, out, out_cap, NULL);
}

extern "C" bool rocke_direct_nchw_dgrad_is_valid_spec(const rocke_direct_nchw_dgrad_spec_t* s,
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
    Atom atom;
    Derived d{};
    if(!validate_host(s, &atom, &d, reason, reason_cap))
    {
        return false;
    }
    const char* ab = problem_is_bf16(&s->problem) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           &target->mma, "mma", ab, ab, "fp32", atom.tile, atom.tile, atom.k))
    {
        rocke_spec_set_reasonf(reason, reason_cap, "missing mfma %s %s on %s", s->atom, ab, arch);
        return false;
    }
    if(s->wave_size != target->wave_size)
    {
        rocke_spec_set_reason(reason, reason_cap, "wave size mismatch");
        return false;
    }
    if(d.threads_per_block > rocke_arch_max_threads_per_block(target))
    {
        rocke_spec_set_reason(reason, reason_cap, "too many threads for arch");
        return false;
    }
    if(!rocke_arch_fits_lds(target, d.lds_bytes))
    {
        rocke_spec_set_reasonf(reason,
                               reason_cap,
                               "LDS %lld B exceeds %d B",
                               d.lds_bytes,
                               target->lds_capacity_bytes);
        return false;
    }
    return true;
}

namespace
{

/* One dY staging unit: a byte offset select (OOB-masked), the window columns
 * it writes, their LDS destinations (OOB-dumped), and its vector width. */
struct XUnit
{
    V off;
    std::vector<int> elems;
    std::vector<V> dsts;
    int width;
};

/* One W staging slot: (global byte offset, LDS element index). */
struct WSlot
{
    V off;
    V lds_idx;
};

rocke_kernel_def_t* build_kernel(rocke_ir_builder_t* b,
                                 const rocke_direct_nchw_dgrad_spec_t* spec,
                                 const Derived& d)
{
    Em e{b};
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const char* dtype = p->dtype;
    bool bf16 = problem_is_bf16(p);
    const rocke_type_t* io = rocke_dconv_io_type(b, dtype);
    const int EB = 2;
    const int KH = p->KH, KW = p->KW, S = p->stride;
    const int TI = spec->tile_i, TJ = spec->tile_j, TC = spec->tile_c, CK = spec->ck;
    const int THREADS = d.threads_per_block;
    const int WAVE = spec->wave_size;
    const int AT = d.atom_tile, AK = d.atom_k, FRAG = d.frag;
    const int ACC = d.acc_per_lane;
    const int QUADS = ACC / 4;
    const int NCB = d.n_col_blocks;
    const int M_TILES = d.m_tiles_total;
    const int MT_W = d.m_tiles_per_wave;
    const int ROWS_W = d.rows_per_wave;
    const int KATOMS = d.k_atoms;
    const int CSTRIDE = d.c_stride;
    const int LDS_H = d.lds_h, LDS_W = d.lds_w;
    const int DH_LO = d.d_h_lo;
    const int DW_LO = d.d_w_lo;
    const int G = spec->stage_group;
    const int HALO_L = d.halo_l, HALO_R = d.halo_r;
    const int HV_L = HALO_L ? halo_vec(HALO_L) : 0;
    const int HV_R = HALO_R ? halo_vec(HALO_R) : 0;
    const auto& TAPS_H = d.taps_h;
    const auto& TAPS_W = d.taps_w;
    const int N_TAPS = KH * KW;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    rocke_dconv_params_t params;
    rocke_dconv_emit_params(b, &params, "fwd", io);
    V dY = params.A, Wt = params.Bp, dX = params.D;
    V p_Hi = params.p_Hi, p_Wi = params.p_Wi;
    V p_Ho = params.p_Ho, p_Wo = params.p_Wo;
    V C = params.p_total_c;
    V K = params.p_total_k;

    V c0 = e.ci(0);
    V c1 = e.ci(1);
    V c_eb = e.ci(EB);
    V oob = e.ci(kOobBase);
    V dy_rsrc = rocke_b_buffer_rsrc(b, dY, params.A_bytes);
    V w_rsrc = rocke_b_buffer_rsrc(b, Wt, params.B_bytes);
    V dx_rsrc = rocke_b_buffer_rsrc(b, dX, params.D_bytes);

    V dy_k_stride = e.mul(p_Ho, p_Wo);
    V dy_n_stride = e.mul(K, dy_k_stride);
    V dx_c_stride = e.mul(p_Hi, p_Wi);
    V dx_n_stride = e.mul(C, dx_c_stride);

    long long x_stage = d.lds_x_elems + kAlign;
    long long w_stage = d.lds_w_elems + FRAG;
    int x_shape[2] = {1, (int)x_stage};
    int w_shape[2] = {1, (int)w_stage};
    V X_smem = rocke_b_smem_alloc(b, io, x_shape, 2, "lds_dy");
    V W_smem = rocke_b_smem_alloc(b, io, w_shape, 2, "lds_w");
    V x_dump = e.ci((int64_t)d.lds_x_elems);
    long long w_dump = d.lds_w_elems;

    V tid = rocke_b_thread_id_x(b);
    V lane = e.mod(tid, e.ci(WAVE));
    V wave_id = e.div(tid, e.ci(WAVE));
    V wave_m = e.div(wave_id, e.ci(spec->waves_n));
    V wave_n = e.mod(wave_id, e.ci(spec->waves_n));
    V lane_lo = e.mod(lane, e.ci(AT));
    V lane_hi = e.div(lane, e.ci(AT));

    /* grid decode: c tile fastest (the dY window is shared by c tiles). */
    V hp_a = e.add(p_Hi, e.ci(S - 1));
    V hp = e.div(hp_a, e.ci(S));
    V wp_a = e.add(p_Wi, e.ci(S - 1));
    V wp = e.div(wp_a, e.ci(S));
    V n_jt_a = e.add(wp, e.ci(TJ - 1));
    V n_jt = e.div(n_jt_a, e.ci(TJ));
    V n_it_a = e.add(hp, e.ci(TI - 1));
    V n_it = e.div(n_it_a, e.ci(TI));
    V n_ct_a = e.add(C, e.ci(TC - 1));
    V n_ct = e.div(n_ct_a, e.ci(TC));
    V wgid = rocke_b_block_id_x(b);
    V c_tile = e.mod(wgid, n_ct);
    V cell = e.div(wgid, n_ct);
    V n_ij = e.mul(n_jt, n_it);
    V n_img = e.div(cell, n_ij);
    V st = e.mod(cell, n_ij);
    V j_tile = e.mod(st, n_jt);
    V i_tile = e.div(st, n_jt);
    V i0 = e.mul(i_tile, e.ci(TI));
    V j0 = e.mul(j_tile, e.ci(TJ));
    V c_base = e.mul(c_tile, e.ci(TC));

    V dy_img = e.mul(n_img, dy_n_stride);
    V row0 = e.addi(i0, DH_LO); /* first staged dY row */

    /* dY staging metadata */
    const int n_kg = CK / G;
    const int NV = TJ / kAlign;

    auto unit_passes = [&](int n_units) {
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

    auto row_meta = [&](V u, int n_inner, V* v, V* row, V* kg, V* ok_h, V* src) {
        if(n_inner > 1)
        {
            *v = e.mod(u, e.ci(n_inner));
        }
        else
        {
            *v = c0;
        }
        V rest = (n_inner > 1) ? e.div(u, e.ci(n_inner)) : u;
        *row = e.mod(rest, e.ci(LDS_H));
        *kg = e.div(rest, e.ci(LDS_H));
        V ho = e.add(row0, *row);
        V ok_lo = e.ge(ho, c0);
        V ok_hi = e.lt(ho, p_Ho);
        *ok_h = e.land(ok_lo, ok_hi);
        V k0 = e.muli(*kg, G);
        V t1 = e.mul(k0, dy_k_stride);
        V t2 = e.add(dy_img, t1);
        V t3 = e.mul(ho, p_Wo);
        *src = e.add(t2, t3);
    };

    /* Body: columns [j0, j0+TJ) of the phase grid == dY columns (offset 0). */
    std::vector<XUnit> x_units;
    for(const auto& up : unit_passes(n_kg * LDS_H * NV))
    {
        V u = up.first, in_tile = up.second;
        V v, row, kg, ok_h, src;
        row_meta(u, NV, &v, &row, &kg, &ok_h, &src);
        V vk = e.muli(v, kAlign);
        V wo0 = e.add(j0, vk);
        V valid_lt = e.lt(wo0, p_Wo);
        V valid = e.land(ok_h, valid_lt);
        if(in_tile != NULL)
        {
            valid = e.land(valid, in_tile);
        }
        V t1 = e.add(src, wo0);
        V t2 = e.mul(t1, c_eb);
        V off = e.select(valid, t2, oob);
        V col0_a = e.muli(v, kAlign);
        V col0 = e.addi(col0_a, -DW_LO);
        V row_lw = e.muli(row, LDS_W);
        V base_a = e.add(row_lw, col0);
        V base_b = e.mul(base_a, e.ci(CSTRIDE));
        V dst0 = e.add(base_b, e.muli(kg, G));
        V step = NULL;
        if(in_tile != NULL)
        {
            dst0 = e.select(in_tile, dst0, x_dump);
            step = e.select(in_tile, e.ci(CSTRIDE), c0);
        }
        std::vector<int> elems;
        std::vector<V> dsts;
        for(int ee = 0; ee < kAlign; ++ee)
        {
            V dd;
            if(ee == 0)
            {
                dd = dst0;
            }
            else if(step == NULL)
            {
                dd = e.addi(dst0, ee * CSTRIDE);
            }
            else
            {
                V t3 = e.muli(step, ee);
                dd = e.add(dst0, t3);
            }
            elems.push_back(ee);
            dsts.push_back(dd);
        }
        x_units.push_back({off, elems, dsts, kAlign});
    }

    /* Halo columns: one aligned HV-wide load per (kg, row); only the |d|
     * columns adjacent to the body are written. */
    struct HaloSide
    {
        char side;
        int h, hv;
    };
    std::vector<HaloSide> halo_sides;
    if(HALO_L)
    {
        halo_sides.push_back({'l', HALO_L, HV_L});
    }
    if(HALO_R)
    {
        halo_sides.push_back({'r', HALO_R, HV_R});
    }
    for(const HaloSide& hs : halo_sides)
    {
        for(const auto& up : unit_passes(n_kg * LDS_H))
        {
            V u = up.first, in_tile = up.second;
            V ignore_v, row, kg, ok_h, src;
            row_meta(u, 1, &ignore_v, &row, &kg, &ok_h, &src);
            V wo0;
            V valid;
            V col0;
            std::vector<int> elems;
            if(hs.side == 'l')
            {
                wo0 = e.subi(j0, hs.hv);
                valid = e.land(ok_h, e.ge(wo0, c0));
                for(int ee = hs.hv - hs.h; ee < hs.hv; ++ee)
                {
                    elems.push_back(ee);
                }
                col0 = e.ci(-DW_LO - hs.hv);
            }
            else
            {
                wo0 = e.addi(j0, TJ);
                valid = e.land(ok_h, e.lt(wo0, p_Wo));
                for(int ee = 0; ee < hs.h; ++ee)
                {
                    elems.push_back(ee);
                }
                col0 = e.ci(-DW_LO + TJ);
            }
            if(in_tile != NULL)
            {
                valid = e.land(valid, in_tile);
            }
            V t1 = e.add(src, wo0);
            V t2 = e.mul(t1, c_eb);
            V off = e.select(valid, t2, oob);
            V row_lw = e.muli(row, LDS_W);
            V base_a = e.add(row_lw, col0);
            V base_b = e.mul(base_a, e.ci(CSTRIDE));
            V base = e.add(base_b, e.muli(kg, G));
            std::vector<V> dsts;
            for(int ee : elems)
            {
                V dd = e.addi(base, ee * CSTRIDE);
                if(in_tile != NULL)
                {
                    dd = e.select(in_tile, dd, x_dump);
                }
                dsts.push_back(dd);
            }
            x_units.push_back({off, elems, dsts, hs.hv});
        }
    }

    /* W staging metadata: fragment-slot layout, [C][KH][KW][K] source. */
    const int W_SLOTS = N_TAPS * M_TILES * KATOMS * WAVE;
    std::vector<WSlot> w_meta;
    int w_passes = cdiv(W_SLOTS, THREADS);
    for(int j = 0; j < w_passes; ++j)
    {
        V g = e.add(tid, e.ci(j * THREADS));
        V g_lane = e.mod(g, e.ci(WAVE));
        V rest = e.div(g, e.ci(WAVE));
        V katom = e.mod(rest, e.ci(KATOMS));
        V rest2 = e.div(rest, e.ci(KATOMS));
        V m_tile = e.mod(rest2, e.ci(M_TILES));
        V tap = e.div(rest2, e.ci(M_TILES));
        V cb1 = e.mul(m_tile, e.ci(AT));
        V cb2 = e.mod(g_lane, e.ci(AT));
        V cb3 = e.add(cb1, cb2);
        V c_out = e.add(c_base, cb3);
        V ki1 = e.mul(katom, e.ci(AK));
        V ki2 = e.div(g_lane, e.ci(AT));
        V ki3 = e.mul(ki2, e.ci(FRAG));
        V k_in = e.add(ki1, ki3);
        V valid = e.lt(c_out, C);
        V lds_idx = e.mul(g, e.ci(FRAG));
        if(W_SLOTS % THREADS)
        {
            V in_tile = e.lt(g, e.ci(W_SLOTS));
            valid = e.land(valid, in_tile);
            lds_idx = e.select(in_tile, lds_idx, e.ci((int64_t)w_dump));
        }
        V em1 = e.mul(c_out, e.ci(N_TAPS));
        V em2 = e.add(em1, tap);
        V em3 = e.mul(em2, K);
        V elems = e.add(em3, k_in);
        V em4 = e.mul(elems, c_eb);
        V off = e.select(valid, em4, oob);
        w_meta.push_back({off, lds_idx});
    }

    V x_chunk_bytes_a = e.mul(e.ci(CK), dy_k_stride);
    V x_chunk_bytes = e.mul(x_chunk_bytes_a, c_eb);
    V w_chunk_bytes = e.ci(CK * EB);

    auto issue_loads = [&](V cx, V cw, std::vector<V>* xs_out, std::vector<V>* ws_out) {
        for(const XUnit& u : x_units)
        {
            for(int ee = 0; ee < G; ++ee)
            {
                V a;
                if(ee)
                {
                    V t1 = e.mul(dy_k_stride, e.ci(ee * EB));
                    V t2 = e.add(cx, t1);
                    a = e.add(u.off, t2);
                }
                else
                {
                    a = e.add(u.off, cx);
                }
                if(u.width == 1)
                {
                    V v = bf16 ? rocke_b_buffer_load_bf16(b, dy_rsrc, a, c0)
                               : rocke_b_buffer_load_f16(b, dy_rsrc, a, c0);
                    V one[1] = {v};
                    v = rocke_b_vec_pack(b, one, 1, io);
                    xs_out->push_back(v);
                }
                else
                {
                    xs_out->push_back(
                        rocke_dconv_buf_load_vN(b, bf16, dy_rsrc, a, c0, u.width / 2));
                }
            }
        }
        for(const WSlot& ws : w_meta)
        {
            V a = e.add(ws.off, cw);
            ws_out->push_back(rocke_dconv_buf_load_vN(b, bf16, w_rsrc, a, c0, FRAG / 2));
        }
    };

    auto commit = [&](const std::vector<V>& xs, const std::vector<V>& ws) {
        size_t i = 0;
        for(const XUnit& u : x_units)
        {
            std::vector<V> runs(xs.begin() + i, xs.begin() + i + G);
            i += G;
            for(size_t k = 0; k < u.elems.size(); ++k)
            {
                int elem = u.elems[k];
                V d2 = u.dsts[k];
                std::vector<V> vals;
                for(V r : runs)
                {
                    vals.push_back(rocke_b_vec_extract(b, r, elem));
                }
                if(G == 1)
                {
                    e.smem_store(X_smem, c0, d2, vals[0], 1);
                }
                else
                {
                    V packed = rocke_b_vec_pack(b, vals.data(), (int)vals.size(), io);
                    e.smem_store(X_smem, c0, d2, packed, G);
                }
            }
        }
        for(size_t k = 0; k < w_meta.size(); ++k)
        {
            e.smem_store(W_smem, c0, w_meta[k].lds_idx, ws[k], FRAG);
        }
    };

    /* ---- LDS read bases ---------------------------------------------- */
    V x_base_1 = e.muli(wave_n, ROWS_W * LDS_W);
    V x_base_2 = e.add(x_base_1, lane_lo);
    V x_base_3 = e.mul(x_base_2, e.ci(CSTRIDE));
    V x_base_4 = e.mul(lane_hi, e.ci(FRAG));
    V x_base = e.add(x_base_3, x_base_4);
    V w_base_1 = e.muli(wave_m, MT_W * KATOMS * WAVE * FRAG);
    V w_base_2 = e.mul(lane, e.ci(FRAG));
    V w_base = e.add(w_base_1, w_base_2);

    auto read_w = [&](int tap, int mt, int katom) -> V {
        int off = ((tap * M_TILES + mt) * KATOMS + katom) * WAVE * FRAG;
        V idx = e.addi(w_base, off);
        return e.smem_load(W_smem, c0, idx, io, FRAG);
    };

    auto read_x = [&](int lrow, int lcol, int katom) -> V {
        int off = (lrow * LDS_W + lcol) * CSTRIDE + katom * AK;
        V idx = e.addi(x_base, off);
        return e.smem_load(X_smem, c0, idx, io, FRAG);
    };

    auto mfma = [&](V a, V bb, V acc) -> V {
        if(strcmp(spec->atom, "32x32x16") == 0)
        {
            return bf16 ? rocke_b_mfma_f32_32x32x16_bf16(b, a, bb, acc)
                        : rocke_b_mfma_f32_32x32x16_f16(b, a, bb, acc);
        }
        if(strcmp(spec->atom, "32x32x8") == 0)
        {
            return bf16 ? rocke_b_mfma_f32_32x32x8_bf16(b, a, bb, acc)
                        : rocke_b_mfma_f32_32x32x8_f16(b, a, bb, acc);
        }
        if(strcmp(spec->atom, "16x16x32") == 0)
        {
            return bf16 ? rocke_b_mfma_f32_16x16x32_bf16(b, a, bb, acc)
                        : rocke_b_mfma_f32_16x16x32_f16(b, a, bb, acc);
        }
        return bf16 ? rocke_b_mfma_f32_16x16x16_bf16(b, a, bb, acc)
                    : rocke_b_mfma_f32_16x16x16_f16(b, a, bb, acc);
    };

    /* Accumulator index: (ph, pw, row, cb, mt). */
    std::map<std::tuple<int, int, int, int, int>, int> acc_idx;
    for(int ph = 0; ph < S; ++ph)
    {
        for(int pw = 0; pw < S; ++pw)
        {
            for(int r = 0; r < ROWS_W; ++r)
            {
                for(int cb = 0; cb < NCB; ++cb)
                {
                    for(int mt = 0; mt < MT_W; ++mt)
                    {
                        int idx = (int)acc_idx.size();
                        acc_idx[{ph, pw, r, cb, mt}] = idx;
                    }
                }
            }
        }
    }
    int n_acc = (int)acc_idx.size();

    /* Group the (pw, s, d_s) users by column offset d_s so each dY fragment
     * column is read once per k atom. */
    std::map<int, std::vector<std::pair<int, int>>> by_dw;
    for(int pw = 0; pw < S; ++pw)
    {
        for(const auto& td : TAPS_W[pw])
        {
            int s = td.first;
            int dw = td.second;
            by_dw[dw].push_back({pw, s});
        }
    }

    auto emit_mfmas = [&](std::vector<V>& accs) {
        for(int katom = 0; katom < KATOMS; ++katom)
        {
            std::map<std::pair<int, int>, V> wf;
            for(const auto& kv : by_dw)
            {
                int dw = kv.first;
                std::map<std::pair<int, int>, V> xf;
                for(const auto& pws : kv.second)
                {
                    int pw = pws.first, s = pws.second;
                    for(int ph = 0; ph < S; ++ph)
                    {
                        for(const auto& rd : TAPS_H[ph])
                        {
                            int r = rd.first, dh = rd.second;
                            int tap = r * KW + s;
                            for(int mt = 0; mt < MT_W; ++mt)
                            {
                                auto key = std::make_pair(tap, mt);
                                if(wf.find(key) == wf.end())
                                {
                                    wf[key] = read_w(tap, mt, katom);
                                }
                            }
                            for(int row = 0; row < ROWS_W; ++row)
                            {
                                int lrow = row + dh - DH_LO;
                                for(int cb = 0; cb < NCB; ++cb)
                                {
                                    auto key = std::make_pair(lrow, cb);
                                    if(xf.find(key) == xf.end())
                                    {
                                        xf[key] = read_x(lrow, cb * AT + dw - DW_LO, katom);
                                    }
                                    for(int mt = 0; mt < MT_W; ++mt)
                                    {
                                        int ix = acc_idx[{ph, pw, row, cb, mt}];
                                        accs[ix] = mfma(xf[key], wf[{tap, mt}], accs[ix]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    std::vector<V> xs0, ws0;
    issue_loads(c0, c0, &xs0, &ws0);
    size_t n_x = xs0.size();
    V zero = rocke_b_zero_vec_f32(b, ACC);

    std::vector<std::string> names;
    std::vector<V> inits;
    for(int i = 0; i < n_acc; ++i)
    {
        names.push_back("acc" + std::to_string(i));
        inits.push_back(zero);
    }
    for(size_t i = 0; i < xs0.size(); ++i)
    {
        names.push_back("xs" + std::to_string(i));
        inits.push_back(xs0[i]);
    }
    for(size_t i = 0; i < ws0.size(); ++i)
    {
        names.push_back("ws" + std::to_string(i));
        inits.push_back(ws0[i]);
    }
    V n_chunks_a = e.div(K, e.ci(CK));
    V n_chunks = e.smax(n_chunks_a, c1);

    std::vector<V> accs_out;
    auto body_fn = [&](V k_iv, const std::vector<V>& carried) -> std::vector<V> {
        std::vector<V> accs(carried.begin(), carried.begin() + n_acc);
        std::vector<V> xs(carried.begin() + n_acc, carried.begin() + n_acc + n_x);
        std::vector<V> ws(carried.begin() + n_acc + n_x, carried.end());
        rocke_b_sync_fence(b, "workgroup");
        commit(xs, ws);
        rocke_b_sync_fence(b, "workgroup");
        V k_next = e.add(k_iv, c1);
        V cx = e.mul(k_next, x_chunk_bytes);
        V cw = e.mul(k_next, w_chunk_bytes);
        std::vector<V> xs_n, ws_n;
        issue_loads(cx, cw, &xs_n, &ws_n);
        emit_mfmas(accs);
        std::vector<V> out = accs;
        out.insert(out.end(), xs_n.begin(), xs_n.end());
        out.insert(out.end(), ws_n.begin(), ws_n.end());
        return out;
    };
    std::vector<V> loop_results = e.for_iter(c0, n_chunks, c1, names, inits, "k_iter", body_fn);
    accs_out.assign(loop_results.begin(), loop_results.begin() + n_acc);

    /* ---- epilogue: NCHW, w-contiguous vector stores -------------------- */
    V c_wave_1 = e.muli(wave_m, MT_W * AT);
    V c_wave = e.add(c_base, c_wave_1);
    V i_wave_1 = e.muli(wave_n, ROWS_W);
    V i_wave = e.add(i0, i_wave_1);
    V dx_img = e.mul(n_img, dx_n_stride);
    V lane_j = e.muli(lane_hi, 4);
    const int VW = 4 * S; /* elements per lane per quad */
    for(int mt = 0; mt < MT_W; ++mt)
    {
        V cw1 = e.addi(c_wave, mt * AT);
        V c_out = e.add(cw1, lane_lo);
        V c_ok = e.lt(c_out, C);
        V c_off_1 = e.mul(c_out, dx_c_stride);
        V c_off = e.add(dx_img, c_off_1);
        for(int ph = 0; ph < S; ++ph)
        {
            for(int row = 0; row < ROWS_W; ++row)
            {
                V h_1 = e.addi(i_wave, row);
                V h_2 = e.muli(h_1, S);
                V h = e.addi(h_2, ph);
                V hc_ok_1 = e.lt(h, p_Hi);
                V hc_ok = e.land(c_ok, hc_ok_1);
                V row_off_1 = e.mul(h, p_Wi);
                V row_off = e.add(c_off, row_off_1);
                for(int cb = 0; cb < NCB; ++cb)
                {
                    std::vector<V> vals_pw;
                    for(int pw = 0; pw < S; ++pw)
                    {
                        V acc = accs_out[acc_idx[{ph, pw, row, cb, mt}]];
                        vals_pw.push_back(rocke_dconv_trunc_f32(b, bf16, acc));
                    }
                    for(int q = 0; q < QUADS; ++q)
                    {
                        V j_const = e.ci(cb * AT + q * (AT / 4));
                        V j_inner = e.add(j_const, lane_j);
                        V j = e.add(j0, j_inner);
                        V w0 = (S > 1) ? e.muli(j, S) : j;
                        std::vector<V> comps;
                        for(int ee = 0; ee < VW; ++ee)
                        {
                            comps.push_back(
                                rocke_b_vec_extract(b, vals_pw[ee % S], 4 * q + ee / S));
                        }
                        V vec = rocke_b_vec_pack(b, comps.data(), VW, io);
                        V d_el = e.add(row_off, w0);
                        V w0vw = e.addi(w0, VW - 1);
                        V full_lt = e.lt(w0vw, p_Wi);
                        V full = e.land(hc_ok, full_lt);
                        rocke_if_else_t br = rocke_b_scf_if_else(b, full);
                        if(br.then_region == NULL || br.else_region == NULL)
                        {
                            return NULL;
                        }
                        rocke_b_region_enter(b, br.then_region);
                        {
                            V d_el_b = e.mul(d_el, c_eb);
                            rocke_dconv_buf_store_vN(b, bf16, dx_rsrc, d_el_b, c0, vec, VW / 2);
                        }
                        rocke_b_region_leave(b);
                        rocke_b_region_enter(b, br.else_region);
                        {
                            for(int ee = 0; ee < VW; ++ee)
                            {
                                V we = ee ? e.addi(w0, ee) : w0;
                                V ok_e_lt = e.lt(we, p_Wi);
                                V ok_e = e.land(hc_ok, ok_e_lt);
                                V de = ee ? e.addi(d_el, ee) : d_el;
                                V deb = e.mul(de, c_eb);
                                V o = e.select(ok_e, deb, oob);
                                if(bf16)
                                {
                                    rocke_b_buffer_store_bf16(b, dx_rsrc, o, c0, comps[ee]);
                                }
                                else
                                {
                                    rocke_b_buffer_store_f16(b, dx_rsrc, o, c0, comps[ee]);
                                }
                            }
                        }
                        rocke_b_region_leave(b);
                    }
                }
            }
        }
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_direct_nchw_dgrad(rocke_ir_builder_t* b,
                                                             const rocke_direct_nchw_dgrad_spec_t* spec,
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
    if(!rocke_direct_nchw_dgrad_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid DirectNchwDgradSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    Atom atom;
    Derived d{};
    char dummy[ROCKE_ERR_MSG_CAP];
    if(!validate_host(spec, &atom, &d, dummy, sizeof dummy))
    {
        return NULL;
    }
    (void)atom;
    return build_kernel(b, spec, d);
}

extern "C" rocke_kernel_def_t*
    rocke_build_direct_nchw_dgrad_new(rocke_ir_builder_t* b,
                                      const rocke_direct_nchw_dgrad_spec_t* spec,
                                      const char* arch)
{
    return rocke_dwg::build_new(
        b,
        spec != NULL,
        "build_direct_conv_nchw_dgrad",
        [&](char* out, size_t cap) { return rocke_direct_nchw_dgrad_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_direct_nchw_dgrad(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_direct_nchw_dgrad_lower_to_llvm(const rocke_direct_nchw_dgrad_spec_t* spec,
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
        "build_direct_conv_nchw_dgrad failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_direct_nchw_dgrad_new(b, spec, arch); });
}
