// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_nchw_fwd.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_nchw_fwd.py: the native-NCHW direct
 * forward convolution (build_direct_conv_nchw_fwd). X is consumed in NCHW and
 * Y produced in NKHW, directly -- no host transpose of the activations.
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the byte-identity contract at the top of
 * rocke/instance_conv_direct_wgrad_internal.hpp, which this port reuses (the
 * `Em` helpers: at most one emitting argument per call).
 */
#include "rocke/instance_conv_direct_nchw_fwd.h"

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

/* Every host-computed (compile-time) geometry constant of the spec: the C++
 * twin of DirectNchwFwdSpec's @property methods, computed once the atom is
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
    int halo_l, halo_r;
    int body_w;
    int lds_h, lds_w;
    int plane_w;
    int lds_row;
    int c_stride;
    long long lds_x_elems;
    long long lds_w_elems;
    long long lds_bytes;
    long long acc_vgprs;
};

Derived derive(const rocke_direct_nchw_fwd_spec_t* s, const Atom& a)
{
    Derived d{};
    const rocke_direct_conv_problem_t* p = &s->problem;
    int S = p->stride;
    d.atom_tile = a.tile;
    d.atom_k = a.k;
    d.frag = a.frag;
    d.acc_per_lane = a.tile * a.tile / s->wave_size;
    d.n_col_blocks = s->tile_j / a.tile;
    d.m_tiles_total = s->tile_k / a.tile;
    d.m_tiles_per_wave = d.m_tiles_total / s->waves_m;
    d.rows_per_wave = s->tile_i / s->waves_n;
    d.k_atoms = s->ck / a.k;
    d.threads_per_block = s->waves_m * s->waves_n * s->wave_size;
    int pad_w = rocke_direct_conv_problem_pad_w(p);
    d.halo_l = pad_w;
    int halo_r = p->KW - pad_w - S;
    d.halo_r = halo_r > 0 ? halo_r : 0;
    d.body_w = s->tile_j * S;
    d.lds_h = (s->tile_i - 1) * S + p->KH;
    d.lds_w = d.halo_l + d.body_w + d.halo_r;
    d.plane_w = cdiv(d.lds_w, 2);
    int w = s->col_split ? 2 * d.plane_w : d.lds_w;
    d.lds_row = w + s->lds_wpad;
    d.c_stride = s->ck + s->lds_pad;
    d.lds_x_elems = (long long)d.lds_h * d.lds_row * d.c_stride;
    d.lds_w_elems = (long long)s->tile_k * p->KH * p->KW * s->ck;
    d.lds_bytes = 2 * (d.lds_x_elems + kAlign + d.lds_w_elems + d.frag);
    d.acc_vgprs = (long long)d.rows_per_wave * d.n_col_blocks * d.m_tiles_per_wave
                  * d.acc_per_lane;
    return d;
}

/* lds_col(x): window column x -> LDS position within a row. */
int lds_col(const rocke_direct_nchw_fwd_spec_t* s, const Derived& d, int x)
{
    if(s->col_split)
    {
        return (x % 2) * d.plane_w + x / 2;
    }
    return x;
}

/* spec.validate(): host-only checks (raises ValueError in Python); returns
 * false + reason on the first that fails, in Python's order. On success fills
 * *atom and *derived. */
bool validate_host(const rocke_direct_nchw_fwd_spec_t* s,
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
    if(s->tile_k % t != 0)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "tile_k must be a multiple of %d", t);
        return false;
    }
    if(d.m_tiles_total % s->waves_m != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tile_k / atom not divisible by waves_m");
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
    if(s->lds_wpad < 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "lds_wpad must be >= 0");
        return false;
    }
    if(s->col_split && p->stride != 2)
    {
        rocke_spec_set_reason(reason, reason_cap, "col_split needs stride 2");
        return false;
    }
    if(s->waves_per_eu < 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "waves_per_eu must be >= 0");
        return false;
    }
    if(p->cpg % s->ck != 0)
    {
        rocke_spec_set_reasonf(reason, reason_cap, "C %d not divisible by ck %d", p->cpg, s->ck);
        return false;
    }
    if(p->W % kAlign != 0)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "W %d must be a multiple of %d (whole X runs)", p->W, kAlign);
        return false;
    }
    int pad_w = rocke_direct_conv_problem_pad_w(p);
    if(!(pad_w >= 0 && pad_w <= kAlign) || d.halo_r > kAlign)
    {
        rocke_spec_set_reason(reason, reason_cap, "halo wider than 8 columns");
        return false;
    }
    if(p->PAD < 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "negative padding");
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
    long long Wo = rocke_direct_conv_problem_Wo(p);
    long long big = eb * p->N * total_c * p->H * p->W;
    long long b2 = eb * total_c * total_k * p->KH * p->KW;
    long long b3 = eb * p->N * total_k * Ho * Wo;
    if(b2 > big)
    {
        big = b2;
    }
    if(b3 > big)
    {
        big = b3;
    }
    if(big + eb * s->ck * p->H * p->W > kOobBase)
    {
        rocke_spec_set_reason(reason, reason_cap, "tensors must fit below the masked-access offset");
        return false;
    }
    return true;
}

} // namespace

extern "C" rocke_direct_nchw_fwd_spec_t rocke_direct_nchw_fwd_spec_default(void)
{
    rocke_direct_nchw_fwd_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "conv_direct_nchw_fwd";
    s.tile_i = 8;
    s.tile_j = 32;
    s.tile_k = 64;
    s.ck = 32;
    s.waves_m = 2;
    s.waves_n = 2;
    s.atom = "32x32x16";
    s.wave_size = 64;
    s.lds_pad = 8;
    s.lds_wpad = 0;
    s.stage_group = 4;
    s.col_split = false;
    s.waves_per_eu = 0;
    s.pin_prefetch = false;
    return s;
}

extern "C" rocke_status_t
    rocke_direct_nchw_fwd_kernel_name(const rocke_direct_nchw_fwd_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f[64], t[64], ck[32], w[32], a[48], g[32], lp[32], wp[32], wpe[32];
    snprintf(f, sizeof f, "f%dx%ds%dp%d", p->KH, p->KW, p->stride, p->PAD);
    snprintf(t, sizeof t, "t%dx%dx%d", s->tile_i, s->tile_j, s->tile_k);
    snprintf(ck, sizeof ck, "ck%d", s->ck);
    snprintf(w, sizeof w, "w%dx%d", s->waves_m, s->waves_n);
    snprintf(a, sizeof a, "a%s", s->atom ? s->atom : "");
    snprintf(g, sizeof g, "g%d", s->stage_group);
    snprintf(lp, sizeof lp, "lp%d", s->lds_pad);
    wp[0] = '\0';
    if(s->lds_wpad)
    {
        snprintf(wp, sizeof wp, "wp%d", s->lds_wpad);
    }
    wpe[0] = '\0';
    if(s->waves_per_eu)
    {
        snprintf(wpe, sizeof wpe, "wpe%d", s->waves_per_eu);
    }
    const char* parts[] = {
        f,
        t,
        ck,
        w,
        a,
        g,
        lp,
        wp,
        s->col_split ? "cs" : "",
        wpe,
        s->pin_prefetch ? "pin" : "",
        problem_is_bf16(p) ? "bf16" : "",
    };
    return rocke_kernel_name_join(
        s->name, parts, sizeof(parts) / sizeof(parts[0]), NULL, NULL, 0, out, out_cap, NULL);
}

extern "C" bool rocke_direct_nchw_fwd_is_valid_spec(const rocke_direct_nchw_fwd_spec_t* s,
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

/* One X staging unit: a byte offset select (OOB-masked), the window columns
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
                                 const rocke_direct_nchw_fwd_spec_t* spec,
                                 const Derived& d)
{
    Em e{b};
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const char* dtype = p->dtype;
    bool bf16 = problem_is_bf16(p);
    const rocke_type_t* io = rocke_dconv_io_type(b, dtype);
    const int EB = 2;
    const int KH = p->KH, KW = p->KW, S = p->stride;
    const int TI = spec->tile_i, TJ = spec->tile_j, TK = spec->tile_k, CK = spec->ck;
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
    const int LDS_H = d.lds_h, LDS_ROW = d.lds_row;
    const int G = spec->stage_group;
    const int HALO_L = d.halo_l, HALO_R = d.halo_r;
    const int HV_L = HALO_L ? halo_vec(HALO_L) : 0;
    const int HV_R = HALO_R ? halo_vec(HALO_R) : 0;
    const int BODY = d.body_w;
    const int N_TAPS = KH * KW;
    const bool SPLIT = spec->col_split;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    if(spec->waves_per_eu)
    {
        const int64_t wpe[2] = {spec->waves_per_eu, 8};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "waves_per_eu", wpe, 2);
    }
    rocke_dconv_params_t params;
    rocke_dconv_emit_params(b, &params, "fwd", io);
    V X = params.A, Wt = params.Bp, Y = params.D;
    V p_Hi = params.p_Hi, p_Wi = params.p_Wi;
    V p_Ho = params.p_Ho, p_Wo = params.p_Wo;
    V C = params.p_total_c;
    V K = params.p_total_k;

    V c0 = e.ci(0);
    V c1 = e.ci(1);
    V c_eb = e.ci(EB);
    V oob = e.ci(kOobBase);
    V x_rsrc = rocke_b_buffer_rsrc(b, X, params.A_bytes);
    V w_rsrc = rocke_b_buffer_rsrc(b, Wt, params.B_bytes);
    V y_rsrc = rocke_b_buffer_rsrc(b, Y, params.D_bytes);

    V x_c_stride = e.mul(p_Hi, p_Wi);
    V x_n_stride = e.mul(C, x_c_stride);
    V y_k_stride = e.mul(p_Ho, p_Wo);
    V y_n_stride = e.mul(K, y_k_stride);

    long long x_stage = d.lds_x_elems + kAlign;
    long long w_stage = d.lds_w_elems + FRAG;
    int x_shape[2] = {1, (int)x_stage};
    int w_shape[2] = {1, (int)w_stage};
    V X_smem = rocke_b_smem_alloc(b, io, x_shape, 2, "lds_x");
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

    /* grid decode: k tile fastest (the X window is shared by k tiles). */
    V n_jt_a = e.add(p_Wo, e.ci(TJ - 1));
    V n_jt = e.div(n_jt_a, e.ci(TJ));
    V n_it_a = e.add(p_Ho, e.ci(TI - 1));
    V n_it = e.div(n_it_a, e.ci(TI));
    V n_kt_a = e.add(K, e.ci(TK - 1));
    V n_kt = e.div(n_kt_a, e.ci(TK));
    V wgid = rocke_b_block_id_x(b);
    V k_tile = e.mod(wgid, n_kt);
    V cell = e.div(wgid, n_kt);
    V n_ij = e.mul(n_jt, n_it);
    V n_img = e.div(cell, n_ij);
    V st = e.mod(cell, n_ij);
    V j_tile = e.mod(st, n_jt);
    V i_tile = e.div(st, n_jt);
    V i0 = e.mul(i_tile, e.ci(TI));
    V j0 = e.mul(j_tile, e.ci(TJ));
    V k_base = e.mul(k_tile, e.ci(TK));

    V x_img = e.mul(n_img, x_n_stride);
    V row0_a = e.mul(i0, e.ci(S));
    V row0 = e.sub(row0_a, e.ci(p->PAD)); /* first X row */
    V xcol0 = e.mul(j0, e.ci(S)); /* first body column (multiple of 8) */

    /* X staging metadata */
    const int n_kg = CK / G;
    const int NV = BODY / kAlign;

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

    auto row_meta = [&](V u, int n_inner, V* inner, V* row, V* cg, V* ok_h, V* src) {
        if(n_inner > 1)
        {
            *inner = e.mod(u, e.ci(n_inner));
        }
        else
        {
            *inner = c0;
        }
        V rest = (n_inner > 1) ? e.div(u, e.ci(n_inner)) : u;
        *row = e.mod(rest, e.ci(LDS_H));
        *cg = e.div(rest, e.ci(LDS_H));
        V h = e.add(row0, *row);
        V ok_lo = e.ge(h, c0);
        V ok_hi = e.lt(h, p_Hi);
        *ok_h = e.land(ok_lo, ok_hi);
        V cc0 = e.mul(*cg, e.ci(G));
        V t1 = e.mul(cc0, x_c_stride);
        V t2 = e.add(x_img, t1);
        V t3 = e.mul(h, p_Wi);
        *src = e.add(t2, t3);
    };

    auto pos = [&](V row, V x_const_or_val, V cg) -> V {
        V t1 = e.mul(row, e.ci(LDS_ROW));
        V t2 = e.add(t1, x_const_or_val);
        V t3 = e.mul(t2, e.ci(CSTRIDE));
        V t4 = e.mul(cg, e.ci(G));
        return e.add(t3, t4);
    };

    /* Body: window columns [HALO_L, HALO_L + BODY) == input [xcol0, xcol0+BODY). */
    std::vector<XUnit> x_units;
    for(const auto& up : unit_passes(n_kg * LDS_H * NV))
    {
        V u = up.first, in_tile = up.second;
        V v, row, cg, ok_h, src;
        row_meta(u, NV, &v, &row, &cg, &ok_h, &src);
        V vk = e.muli(v, kAlign);
        V w0 = e.add(xcol0, vk);
        V valid_lt = e.lt(w0, p_Wi);
        V valid = e.land(ok_h, valid_lt);
        if(in_tile != NULL)
        {
            valid = e.land(valid, in_tile);
        }
        V t1 = e.add(src, w0);
        V t2 = e.mul(t1, c_eb);
        V off = e.select(valid, t2, oob);
        std::vector<int> elems;
        std::vector<V> dsts;
        for(int ee = 0; ee < kAlign; ++ee)
        {
            int x = HALO_L + ee;
            V colv;
            if(SPLIT)
            {
                V cv1 = e.muli(v, kAlign / 2);
                colv = e.addi(cv1, (x % 2) * d.plane_w + x / 2);
            }
            else
            {
                V cv1 = e.muli(v, kAlign);
                colv = e.addi(cv1, x);
            }
            V dd = pos(row, colv, cg);
            if(in_tile != NULL)
            {
                dd = e.select(in_tile, dd, x_dump);
            }
            elems.push_back(ee);
            dsts.push_back(dd);
        }
        x_units.push_back({off, elems, dsts, kAlign});
    }

    /* Halo columns: one aligned HV-wide load per (cg, row); only the columns
     * adjacent to the body are written. */
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
            V ignore_v, row, cg, ok_h, src;
            row_meta(u, 1, &ignore_v, &row, &cg, &ok_h, &src);
            V w0;
            V valid;
            std::vector<int> elems, xs_;
            if(hs.side == 'l')
            {
                w0 = e.subi(xcol0, hs.hv);
                valid = e.land(ok_h, e.ge(w0, c0));
                for(int ee = hs.hv - hs.h; ee < hs.hv; ++ee)
                {
                    elems.push_back(ee);
                    xs_.push_back(ee - (hs.hv - hs.h));
                }
            }
            else
            {
                w0 = e.addi(xcol0, BODY);
                valid = e.land(ok_h, e.lt(w0, p_Wi));
                for(int ee = 0; ee < hs.h; ++ee)
                {
                    elems.push_back(ee);
                    xs_.push_back(HALO_L + BODY + ee);
                }
            }
            if(in_tile != NULL)
            {
                valid = e.land(valid, in_tile);
            }
            V t1 = e.add(src, w0);
            V t2 = e.mul(t1, c_eb);
            V off = e.select(valid, t2, oob);
            std::vector<V> dsts;
            for(int x : xs_)
            {
                V dd = pos(row, e.ci(lds_col(spec, d, x)), cg);
                if(in_tile != NULL)
                {
                    dd = e.select(in_tile, dd, x_dump);
                }
                dsts.push_back(dd);
            }
            x_units.push_back({off, elems, dsts, hs.hv});
        }
    }

    /* W staging metadata: fragment-slot layout, [K][KH][KW][C] source. */
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
        V kb1 = e.mul(m_tile, e.ci(AT));
        V kb2 = e.mod(g_lane, e.ci(AT));
        V kb3 = e.add(kb1, kb2);
        V k_out = e.add(k_base, kb3);
        V ci1 = e.mul(katom, e.ci(AK));
        V ci2 = e.div(g_lane, e.ci(AT));
        V ci3 = e.mul(ci2, e.ci(FRAG));
        V c_in = e.add(ci1, ci3);
        V valid = e.lt(k_out, K);
        V lds_idx = e.mul(g, e.ci(FRAG));
        if(W_SLOTS % THREADS)
        {
            V in_tile = e.lt(g, e.ci(W_SLOTS));
            valid = e.land(valid, in_tile);
            lds_idx = e.select(in_tile, lds_idx, e.ci((int64_t)w_dump));
        }
        V em1 = e.mul(k_out, e.ci(N_TAPS));
        V em2 = e.add(em1, tap);
        V em3 = e.mul(em2, C);
        V elems = e.add(em3, c_in);
        V em4 = e.mul(elems, c_eb);
        V off = e.select(valid, em4, oob);
        w_meta.push_back({off, lds_idx});
    }

    V x_chunk_bytes_a = e.mul(e.ci(CK), x_c_stride);
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
                    V t1 = e.mul(x_c_stride, e.ci(ee * EB));
                    V t2 = e.add(cx, t1);
                    a = e.add(u.off, t2);
                }
                else
                {
                    a = e.add(u.off, cx);
                }
                xs_out->push_back(rocke_dconv_buf_load_vN(b, bf16, x_rsrc, a, c0, u.width / 2));
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
    V lane_col = SPLIT ? lane_lo : (S > 1 ? e.muli(lane_lo, S) : lane_lo);
    V x_base_1 = e.muli(wave_n, ROWS_W * S * LDS_ROW);
    V x_base_2 = e.add(x_base_1, lane_col);
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

    auto col_off = [&](int cb, int s) -> int {
        if(SPLIT)
        {
            return (s % 2) * d.plane_w + cb * AT + s / 2;
        }
        return cb * AT * S + s;
    };

    auto read_x = [&](int lrow, int cb, int s, int katom) -> V {
        int off = (lrow * LDS_ROW + col_off(cb, s)) * CSTRIDE + katom * AK;
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

    std::map<std::tuple<int, int, int>, int> acc_idx;
    for(int row = 0; row < ROWS_W; ++row)
    {
        for(int cb = 0; cb < NCB; ++cb)
        {
            for(int mt = 0; mt < MT_W; ++mt)
            {
                int idx = (int)acc_idx.size();
                acc_idx[{row, cb, mt}] = idx;
            }
        }
    }
    int n_acc = (int)acc_idx.size();

    auto emit_mfmas = [&](std::vector<V>& accs) {
        for(int katom = 0; katom < KATOMS; ++katom)
        {
            std::map<std::pair<int, int>, V> wf;
            for(int s = 0; s < KW; ++s)
            {
                std::map<std::pair<int, int>, V> xf;
                for(int r = 0; r < KH; ++r)
                {
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
                        int lrow = row * S + r;
                        for(int cb = 0; cb < NCB; ++cb)
                        {
                            auto key = std::make_pair(lrow, cb);
                            if(xf.find(key) == xf.end())
                            {
                                xf[key] = read_x(lrow, cb, s, katom);
                            }
                            for(int mt = 0; mt < MT_W; ++mt)
                            {
                                int ix = acc_idx[{row, cb, mt}];
                                accs[ix] = mfma(xf[key], wf[{tap, mt}], accs[ix]);
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
    V n_chunks_a = e.div(C, e.ci(CK));
    V n_chunks = e.smax(n_chunks_a, c1);

    std::vector<V> accs_out;
    auto body_fn = [&](V c_iv, const std::vector<V>& carried) -> std::vector<V> {
        std::vector<V> accs(carried.begin(), carried.begin() + n_acc);
        std::vector<V> xs(carried.begin() + n_acc, carried.begin() + n_acc + n_x);
        std::vector<V> ws(carried.begin() + n_acc + n_x, carried.end());
        rocke_b_sync_fence(b, "workgroup");
        commit(xs, ws);
        rocke_b_sync_fence(b, "workgroup");
        V c_next = e.add(c_iv, c1);
        V cx = e.mul(c_next, x_chunk_bytes);
        V cw = e.mul(c_next, w_chunk_bytes);
        std::vector<V> xs_n, ws_n;
        issue_loads(cx, cw, &xs_n, &ws_n);
        if(spec->pin_prefetch)
        {
            rocke_b_sched_barrier(b, 0);
        }
        emit_mfmas(accs);
        std::vector<V> out = accs;
        out.insert(out.end(), xs_n.begin(), xs_n.end());
        out.insert(out.end(), ws_n.begin(), ws_n.end());
        return out;
    };
    std::vector<V> loop_results = e.for_iter(c0, n_chunks, c1, names, inits, "c_iter", body_fn);
    accs_out.assign(loop_results.begin(), loop_results.begin() + n_acc);

    /* ---- epilogue: NKHW, wo-contiguous dwordx2 stores ------------------ */
    V k_wave_1 = e.muli(wave_m, MT_W * AT);
    V k_wave = e.add(k_base, k_wave_1);
    V i_wave_1 = e.muli(wave_n, ROWS_W);
    V i_wave = e.add(i0, i_wave_1);
    V y_img = e.mul(n_img, y_n_stride);
    V lane_j = e.muli(lane_hi, 4);
    for(int mt = 0; mt < MT_W; ++mt)
    {
        V kw1 = e.addi(k_wave, mt * AT);
        V k_out = e.add(kw1, lane_lo);
        V k_ok = e.lt(k_out, K);
        V k_off_1 = e.mul(k_out, y_k_stride);
        V k_off = e.add(y_img, k_off_1);
        for(int row = 0; row < ROWS_W; ++row)
        {
            V ho = e.addi(i_wave, row);
            V hk_ok_1 = e.lt(ho, p_Ho);
            V hk_ok = e.land(k_ok, hk_ok_1);
            V row_off_1 = e.mul(ho, p_Wo);
            V row_off = e.add(k_off, row_off_1);
            for(int cb = 0; cb < NCB; ++cb)
            {
                V acc = accs_out[acc_idx[{row, cb, mt}]];
                V vals = rocke_dconv_trunc_f32(b, bf16, acc);
                for(int q = 0; q < QUADS; ++q)
                {
                    V wo_const = e.ci(cb * AT + q * (AT / 4));
                    V wo_inner = e.add(wo_const, lane_j);
                    V wo = e.add(j0, wo_inner);
                    std::vector<V> comps;
                    for(int ee = 0; ee < 4; ++ee)
                    {
                        comps.push_back(rocke_b_vec_extract(b, vals, 4 * q + ee));
                    }
                    V vec = rocke_b_vec_pack(b, comps.data(), 4, io);
                    V d_el = e.add(row_off, wo);
                    V wo3 = e.addi(wo, 3);
                    V full_lt = e.lt(wo3, p_Wo);
                    V full = e.land(hk_ok, full_lt);
                    rocke_if_else_t br = rocke_b_scf_if_else(b, full);
                    if(br.then_region == NULL || br.else_region == NULL)
                    {
                        return NULL;
                    }
                    rocke_b_region_enter(b, br.then_region);
                    {
                        V d_el_b = e.mul(d_el, c_eb);
                        rocke_dconv_buf_store_vN(b, bf16, y_rsrc, d_el_b, c0, vec, 2);
                    }
                    rocke_b_region_leave(b);
                    rocke_b_region_enter(b, br.else_region);
                    {
                        for(int ee = 0; ee < 4; ++ee)
                        {
                            V we = ee ? e.addi(wo, ee) : wo;
                            V ok_e_lt = e.lt(we, p_Wo);
                            V ok_e = e.land(hk_ok, ok_e_lt);
                            V de = ee ? e.addi(d_el, ee) : d_el;
                            V deb = e.mul(de, c_eb);
                            V o = e.select(ok_e, deb, oob);
                            if(bf16)
                            {
                                rocke_b_buffer_store_bf16(b, y_rsrc, o, c0, comps[ee]);
                            }
                            else
                            {
                                rocke_b_buffer_store_f16(b, y_rsrc, o, c0, comps[ee]);
                            }
                        }
                    }
                    rocke_b_region_leave(b);
                }
            }
        }
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_direct_nchw_fwd(rocke_ir_builder_t* b,
                                                           const rocke_direct_nchw_fwd_spec_t* spec,
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
    if(!rocke_direct_nchw_fwd_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid DirectNchwFwdSpec for ") + arch;
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
    rocke_build_direct_nchw_fwd_new(rocke_ir_builder_t* b,
                                    const rocke_direct_nchw_fwd_spec_t* spec,
                                    const char* arch)
{
    return rocke_dwg::build_new(
        b,
        spec != NULL,
        "build_direct_conv_nchw_fwd",
        [&](char* out, size_t cap) { return rocke_direct_nchw_fwd_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_direct_nchw_fwd(b, spec, arch); });
}

extern "C" rocke_status_t
    rocke_direct_nchw_fwd_lower_to_llvm(const rocke_direct_nchw_fwd_spec_t* spec,
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
        "build_direct_conv_nchw_fwd failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_direct_nchw_fwd_new(b, spec, arch); });
}
