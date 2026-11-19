/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C++ port of library/kernels/common/conv_dgrad_interior.py: the host-side
 * algebra of interior-only dX launches (dx_interior / dx_interior_axes and the
 * xh_dx_wmask W-extended view) and the halo shell zero-fill kernel. Every
 * operand of the kernel is bound to a temporary in Python's left-to-right
 * evaluation order (the emitted IR is byte-compared).
 */

#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_dgrad_interior.h"

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/helper_rocke.helpers.tensor_view.h" /* rocke_tv_make_buffer_resource */
#include "rocke/ir.h"

#define DXI_REASON(...)                                \
    do                                                 \
    {                                                  \
        if(reason && reason_cap)                       \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        return false;                                  \
    } while(0)

void rocke_dgrad_halo_widths(const rocke_conv_problem_t* p, int axes, int out[3])
{
    out[0] = (p->is_3d && (axes & ROCKE_DXA_D)) ? (p->Z - 1) / 2 : 0;
    out[1] = (axes & ROCKE_DXA_H) ? (p->Y - 1) / 2 : 0;
    out[2] = (axes & ROCKE_DXA_W) ? (p->X - 1) / 2 : 0;
}

bool rocke_dgrad_dx_interior_problem_ok(const rocke_conv_problem_t* p,
                                        int mask,
                                        char* reason,
                                        size_t reason_cap)
{
    struct axis
    {
        int f, s, pad, dil, bit;
    };
    struct axis axes[3] = {{p->Y, p->sH, p->pH, p->dH, ROCKE_DXA_H},
                           {p->X, p->sW, p->pW, p->dW, ROCKE_DXA_W},
                           {p->Z, p->sD, p->pD, p->dD, ROCKE_DXA_D}};
    const int n = p->is_3d ? 3 : 2;
    for(int i = 0; i < n; ++i)
    {
        if(axes[i].s != 1 || axes[i].dil != 1)
            DXI_REASON("dx_interior serves stride 1 / dilation 1 only");
        if(!(mask & axes[i].bit))
            continue;
        if(axes[i].pad != 0)
            DXI_REASON("dx_interior serves the halo form (pad 0) only");
        if(axes[i].f % 2 == 0)
            DXI_REASON("dx_interior needs an odd filter");
    }
    if(p->groups != 1)
        DXI_REASON("dx_interior serves groups == 1 only");
    if(rocke_conv_problem_ho(p) < 1 || rocke_conv_problem_wo(p) < 1
       || (p->is_3d && rocke_conv_problem_do(p) < 1))
        DXI_REASON("dx_interior: empty output");
    if(reason && reason_cap)
        snprintf(reason, reason_cap, "ok");
    return true;
}

rocke_conv_problem_t
    rocke_dgrad_interior_view_problem(const rocke_conv_problem_t* p, bool keep_w, int axes)
{
    int h[3];
    rocke_dgrad_halo_widths(p, axes, h);
    rocke_conv_problem_t v = *p;
    if(h[1])
    {
        v.Hi = p->Hi - 2 * h[1];
        v.pH = h[1];
    }
    if(h[2] && !keep_w)
    {
        v.Wi = p->Wi - 2 * h[2];
        v.pW = h[2];
    }
    if(h[0])
    {
        v.Di = p->Di - 2 * h[0];
        v.pD = h[0];
    }
    return v;
}

/* _full_strides(p): the dense NDHWC element strides of p's dX. */
static void dxi_full_strides(
    const rocke_conv_problem_t* p, int64_t* sn, int64_t* sd, int64_t* sh, int64_t* sw)
{
    *sw = p->C;
    *sh = (int64_t)p->Wi * p->C;
    *sd = (int64_t)p->Hi * *sh;
    *sn = p->is_3d ? (int64_t)p->Di * *sd : *sd;
}

void rocke_dgrad_interior_view_dx(const rocke_conv_problem_t* p,
                                  bool keep_w,
                                  int axes,
                                  rocke_dgrad_dx_view_t* out)
{
    int h[3];
    rocke_dgrad_halo_widths(p, axes, h);
    int64_t sn, sd, sh, sw;
    dxi_full_strides(p, &sn, &sd, &sh, &sw);
    out->offset = (int64_t)h[0] * sd * (p->is_3d ? 1 : 0) + (int64_t)h[1] * sh
                  + (int64_t)(keep_w ? 0 : h[2]) * sw;
    out->stride_n = sn;
    out->stride_hi = sh;
    out->stride_wi = sw;
    out->stride_di = 0;
    if(p->is_3d)
    {
        out->stride_di = sd;
        /* One image: its stride is never used; the view's own (Do planes)
         * keeps the n-stride check of the plane-gapped linear store true. */
        if(p->N == 1)
            out->stride_n = sd * (int64_t)(p->Di - 2 * h[0]);
    }
}

void rocke_dgrad_wmask_view(const rocke_conv_problem_t* p,
                            rocke_conv_problem_t* view,
                            rocke_dgrad_dx_view_t* out)
{
    int h[3];
    rocke_dgrad_halo_widths(p, ROCKE_DXA_ALL, h);
    *view = *p;
    view->Wi = p->Wi + 2 * h[2];
    view->pW = 0;
    int64_t sn, sd, sh, sw;
    dxi_full_strides(p, &sn, &sd, &sh, &sw);
    out->offset = -(int64_t)h[2] * sw;
    out->stride_n = sn;
    out->stride_hi = sh;
    out->stride_wi = sw;
    out->stride_di = p->is_3d ? sd : 0;
}

bool rocke_dgrad_dx_interior_launch_ok(const rocke_dgrad_conv_spec_t* s,
                                       int dx_interior,
                                       int dx_interior_axes,
                                       const rocke_conv_problem_t* p,
                                       char* reason,
                                       size_t reason_cap)
{
    if(s->xh_dx_wmask && !dx_interior)
    {
        const int hw = (p->X - 1) / 2;
        if(!s->tapk_xhalo || hw < 1 || p->pW != hw)
            DXI_REASON("xh_dx_wmask serves pW == (X-1)/2 problems (or a dx_interior launch)");
        if(p->sW != 1 || p->dW != 1)
            DXI_REASON("xh_dx_wmask serves stride 1 / dilation 1 W");
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "ok");
        return true;
    }
    if(!dx_interior)
    {
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "ok");
        return true;
    }
    if(dx_interior != ROCKE_DXI_SKIP && dx_interior != ROCKE_DXI_ZERO)
        DXI_REASON("dx_interior=%d is not a mode", dx_interior);
    if(!s->tap_outer_k)
        DXI_REASON("dx_interior serves the stride-1 tap-outer kernels only");
    if(s->split_k > 1 && !(s->pp_boxes && s->pp_box_det))
        DXI_REASON("dx_interior: split-K only as pp_box_det (no dX zeroing)");
    if(s->pp_streamk || s->tapk_pp_persist == 2)
        DXI_REASON("dx_interior: no stream-K (it zeroes dX)");
    const int ax = dx_interior_axes;
    if(ax < 0 || ax > 7)
        DXI_REASON("dx_interior_axes is a bit mask 0..7");
    if(s->xh_dx_wmask && ax && !(ax & ROCKE_DXA_W))
        DXI_REASON("xh_dx_wmask skips the W shell: dx_interior_axes must include W");
    return rocke_dgrad_dx_interior_problem_ok(p, ax ? ax : ROCKE_DXA_ALL, reason, reason_cap);
}

/* ---------------------------------------------------------------------- */
/* zero-fill kernel                                                        */
/* ---------------------------------------------------------------------- */

rocke_dgrad_halo_zero_spec_t rocke_dgrad_halo_zero_spec_default(void)
{
    rocke_dgrad_halo_zero_spec_t s;
    memset(&s, 0, sizeof(s));
    s.block = 256;
    s.name = "conv_dgrad_halo_zero";
    return s;
}

rocke_status_t rocke_dgrad_halo_zero_kernel_name(const rocke_dgrad_halo_zero_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char blk[32];
    snprintf(blk, sizeof(blk), "b%d", spec->block);
    const char* parts[2] = {blk, "v2"};
    return rocke_kernel_name_join(spec->name ? spec->name : "conv_dgrad_halo_zero",
                                  parts,
                                  2,
                                  NULL,
                                  NULL,
                                  0,
                                  out,
                                  out_cap,
                                  NULL);
}

namespace
{

/* The kernarg order of the scalars after dx / dx_bytes. */
const char* const kHzNames[ROCKE_DGRAD_HALO_ZERO_SCALARS] = {
    "Di", "Do", "Hi", "Ho", "ru", "hd", "hh", "e", "L1", "L2", "L3", "n1", "n2", "u1", "u2", "u3"};
enum
{
    HZ_DI,
    HZ_DO,
    HZ_HI,
    HZ_HO,
    HZ_RU,
    HZ_HD,
    HZ_HH,
    HZ_E,
    HZ_L1,
    HZ_L2,
    HZ_L3,
    HZ_N1,
    HZ_N2,
    HZ_U1,
    HZ_U2,
    HZ_U3
};

struct hz_ctx
{
    rocke_value_t* k[ROCKE_DGRAD_HALO_ZERO_SCALARS];
    rocke_value_t* plane;
};

/* sec1: (n, side, offset) over L1 = 2 * hd * plane units per image. */
rocke_value_t* hz_sec1(rocke_ir_builder_t* b, const hz_ctx* c, rocke_value_t* t)
{
    rocke_value_t* const* k = c->k;
    rocke_value_t* n = rocke_b_div(b, t, k[HZ_L1]);
    rocke_value_t* nl = rocke_b_mul(b, n, k[HZ_L1]);
    rocke_value_t* r = rocke_b_sub(b, t, nl);
    rocke_value_t* half = rocke_b_mul(b, k[HZ_HD], c->plane);
    rocke_value_t* side = rocke_b_div(b, r, half);
    rocke_value_t* sh = rocke_b_mul(b, side, half);
    rocke_value_t* o = rocke_b_sub(b, r, sh);
    rocke_value_t* dmh = rocke_b_sub(b, k[HZ_DI], k[HZ_HD]);
    rocke_value_t* last = rocke_b_mul(b, dmh, c->plane);
    rocke_value_t* nd = rocke_b_mul(b, n, k[HZ_DI]);
    rocke_value_t* ndp = rocke_b_mul(b, nd, c->plane);
    rocke_value_t* sl = rocke_b_mul(b, side, last);
    rocke_value_t* a = rocke_b_add(b, ndp, sl);
    return rocke_b_add(b, a, o);
}

rocke_value_t* hz_sec2(rocke_ir_builder_t* b, const hz_ctx* c, rocke_value_t* t)
{
    rocke_value_t* const* k = c->k;
    rocke_value_t* g = rocke_b_div(b, t, k[HZ_L2]);
    rocke_value_t* gl = rocke_b_mul(b, g, k[HZ_L2]);
    rocke_value_t* r = rocke_b_sub(b, t, gl);
    rocke_value_t* half = rocke_b_mul(b, k[HZ_HH], k[HZ_RU]);
    rocke_value_t* side = rocke_b_div(b, r, half);
    rocke_value_t* sh = rocke_b_mul(b, side, half);
    rocke_value_t* o = rocke_b_sub(b, r, sh);
    rocke_value_t* n = rocke_b_div(b, g, k[HZ_DO]);
    rocke_value_t* nd = rocke_b_mul(b, n, k[HZ_DO]);
    rocke_value_t* d = rocke_b_sub(b, g, nd);
    rocke_value_t* ndi = rocke_b_mul(b, n, k[HZ_DI]);
    rocke_value_t* dh = rocke_b_add(b, d, k[HZ_HD]);
    rocke_value_t* pl = rocke_b_add(b, ndi, dh);
    rocke_value_t* base = rocke_b_mul(b, pl, c->plane);
    rocke_value_t* hmh = rocke_b_sub(b, k[HZ_HI], k[HZ_HH]);
    rocke_value_t* last = rocke_b_mul(b, hmh, k[HZ_RU]);
    rocke_value_t* sl = rocke_b_mul(b, side, last);
    rocke_value_t* a = rocke_b_add(b, base, sl);
    return rocke_b_add(b, a, o);
}

rocke_value_t* hz_sec3(rocke_ir_builder_t* b, const hz_ctx* c, rocke_value_t* t)
{
    rocke_value_t* const* k = c->k;
    rocke_value_t* g = rocke_b_div(b, t, k[HZ_L3]);
    rocke_value_t* gl = rocke_b_mul(b, g, k[HZ_L3]);
    rocke_value_t* r = rocke_b_sub(b, t, gl);
    rocke_value_t* side = rocke_b_div(b, r, k[HZ_E]);
    rocke_value_t* se = rocke_b_mul(b, side, k[HZ_E]);
    rocke_value_t* o = rocke_b_sub(b, r, se);
    rocke_value_t* nd = rocke_b_div(b, g, k[HZ_HO]);
    rocke_value_t* ndh = rocke_b_mul(b, nd, k[HZ_HO]);
    rocke_value_t* h = rocke_b_sub(b, g, ndh);
    rocke_value_t* n = rocke_b_div(b, nd, k[HZ_DO]);
    rocke_value_t* nn = rocke_b_mul(b, n, k[HZ_DO]);
    rocke_value_t* d = rocke_b_sub(b, nd, nn);
    rocke_value_t* ndi = rocke_b_mul(b, n, k[HZ_DI]);
    rocke_value_t* dh = rocke_b_add(b, d, k[HZ_HD]);
    rocke_value_t* pl = rocke_b_add(b, ndi, dh);
    rocke_value_t* plh = rocke_b_mul(b, pl, k[HZ_HI]);
    rocke_value_t* hh = rocke_b_add(b, h, k[HZ_HH]);
    rocke_value_t* row = rocke_b_add(b, plh, hh);
    rocke_value_t* rr = rocke_b_mul(b, row, k[HZ_RU]);
    rocke_value_t* rme = rocke_b_sub(b, k[HZ_RU], k[HZ_E]);
    rocke_value_t* sr = rocke_b_mul(b, side, rme);
    rocke_value_t* a = rocke_b_add(b, rr, sr);
    return rocke_b_add(b, a, o);
}

void hz_store(rocke_ir_builder_t* b,
              rocke_value_t* ok,
              rocke_value_t* unit,
              rocke_value_t* c16,
              rocke_value_t* oob,
              rocke_value_t* rsrc,
              rocke_value_t* c0,
              rocke_value_t* zero)
{
    rocke_value_t* ub = rocke_b_mul(b, unit, c16);
    rocke_value_t* off = rocke_b_select(b, ok, ub, oob);
    rocke_b_buffer_store_vN(b, rsrc, off, c0, zero);
}

} // namespace

rocke_kernel_def_t* rocke_build_dgrad_halo_zero(rocke_ir_builder_t* b,
                                                const rocke_dgrad_halo_zero_spec_t* spec)
{
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block);
    rocke_param_opts_t wo;
    memset(&wo, 0, sizeof(wo));
    wo.noalias = wo.noalias_set = true;
    wo.writeonly = wo.writeonly_set = true;
    wo.align = 16;
    wo.align_set = true;
    rocke_value_t* dx_ptr = rocke_b_param(b, "dx", rocke_ptr_type(b, rocke_i32(), "global"), &wo);
    rocke_value_t* dx_bytes = rocke_b_param(b, "dx_bytes", rocke_i32(), NULL);
    hz_ctx c;
    for(int i = 0; i < ROCKE_DGRAD_HALO_ZERO_SCALARS; ++i)
        c.k[i] = rocke_b_param(b, kHzNames[i], rocke_i32(), NULL);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    (void)rocke_b_const_i32(b, 1);
    rocke_value_t* blk = rocke_b_block_id_x(b);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* cb = rocke_b_const_i32(b, spec->block);
    c.plane = rocke_b_mul(b, c.k[HZ_HI], c.k[HZ_RU]);

    rocke_value_t* in2 = rocke_b_cmp_ge(b, blk, c.k[HZ_N1]);
    rocke_value_t* in3 = rocke_b_cmp_ge(b, blk, c.k[HZ_N2]);
    rocke_value_t* b3 = rocke_b_sub(b, blk, c.k[HZ_N2]);
    rocke_value_t* b2 = rocke_b_sub(b, blk, c.k[HZ_N1]);
    rocke_value_t* s2 = rocke_b_select(b, in2, b2, blk);
    rocke_value_t* sec_blk = rocke_b_select(b, in3, b3, s2);
    rocke_value_t* sb = rocke_b_mul(b, sec_blk, cb);
    rocke_value_t* t = rocke_b_add(b, sb, tid);
    rocke_value_t* c2 = rocke_b_select(b, in2, c.k[HZ_U2], c.k[HZ_U1]);
    rocke_value_t* cnt = rocke_b_select(b, in3, c.k[HZ_U3], c2);
    rocke_tv_buffer_resource_t dx;
    rocke_tv_make_buffer_resource(b, &dx, dx_ptr, dx_bytes, NULL);
    rocke_value_t* zero = rocke_b_zero_vec(b, rocke_i32(), 4);
    rocke_value_t* oob = rocke_b_const_i32(b, 0x7FFFFFF0);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* ok = rocke_b_cmp_lt(b, t, cnt);

    rocke_if_else_t ie3 = rocke_b_scf_if_else(b, in3);
    rocke_b_region_enter(b, ie3.then_region);
    hz_store(b, ok, hz_sec3(b, &c, t), c16, oob, dx.rsrc, c0, zero);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, ie3.else_region);
    {
        rocke_if_else_t ie2 = rocke_b_scf_if_else(b, in2);
        rocke_b_region_enter(b, ie2.then_region);
        hz_store(b, ok, hz_sec2(b, &c, t), c16, oob, dx.rsrc, c0, zero);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie2.else_region);
        hz_store(b, ok, hz_sec1(b, &c, t), c16, oob, dx.rsrc, c0, zero);
        rocke_b_region_leave(b);
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_dgrad_halo_zero_new(rocke_ir_builder_t* b,
                                                    const rocke_dgrad_halo_zero_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if(b == NULL || spec == NULL)
            return NULL;
        char name[256];
        if(rocke_dgrad_halo_zero_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_dgrad_halo_zero(b, spec);
    });
}

rocke_status_t rocke_dgrad_halo_zero_units(const rocke_conv_problem_t* p,
                                           int elem_bytes,
                                           int axes,
                                           int64_t out[ROCKE_DGRAD_HALO_ZERO_UNITS])
{
    int h[3];
    rocke_dgrad_halo_widths(p, axes, h);
    const int64_t pix = (int64_t)p->C * elem_bytes;
    if(pix % 16)
        return ROCKE_ERR_VALUE;
    const int64_t pu = pix / 16;
    const int64_t N = p->N;
    const int64_t Di = p->is_3d ? p->Di : 1;
    const int64_t Do = Di - 2 * h[0];
    const int64_t Hi = p->Hi, ru = (int64_t)p->Wi * pu;
    const int64_t Ho = Hi - 2 * h[1];
    const int64_t e = h[2] * pu;
    const int64_t L1 = 2 * h[0] * Hi * ru, L2 = 2 * h[1] * ru, L3 = 2 * e;
    const int64_t v[ROCKE_DGRAD_HALO_ZERO_UNITS]
        = {Di, Do, Hi, Ho, ru, h[0], h[1], e, L1, L2, L3, N * L1, N * Do * L2, N * Do * Ho * L3};
    memcpy(out, v, sizeof(v));
    return ROCKE_OK;
}

rocke_status_t rocke_dgrad_halo_zero_launch(const rocke_conv_problem_t* p,
                                            int elem_bytes,
                                            int axes,
                                            const rocke_dgrad_halo_zero_spec_t* spec,
                                            int64_t values[ROCKE_DGRAD_HALO_ZERO_SCALARS],
                                            int grid[3],
                                            int block[3])
{
    int64_t u[ROCKE_DGRAD_HALO_ZERO_UNITS];
    rocke_status_t st = rocke_dgrad_halo_zero_units(p, elem_bytes, axes, u);
    if(st != ROCKE_OK)
        return st;
    const int64_t B = spec->block;
    const int64_t b1 = (u[11] + B - 1) / B, b2 = (u[12] + B - 1) / B, b3 = (u[13] + B - 1) / B;
    for(int i = 0; i < 7; ++i)
        values[i] = u[i];
    /* Placeholder divisors of an empty section (e and L1..L3; Python
     * halo_zero_launch): a fully empty shell (1-tap filters) still launches
     * block 0 into the section-3 branch, every store out of bounds. */
    for(int i = 7; i < 11; ++i)
        values[i] = u[i] > 1 ? u[i] : 1;
    values[11] = b1;
    values[12] = b1 + b2;
    values[13] = u[11];
    values[14] = u[12];
    values[15] = u[13];
    const int64_t nb = b1 + b2 + b3;
    grid[0] = (int)(nb > 1 ? nb : 1);
    grid[1] = 1;
    grid[2] = 1;
    block[0] = (int)B;
    block[1] = 1;
    block[2] = 1;
    return ROCKE_OK;
}
