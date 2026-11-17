// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_winograd_gemm_wmma.cpp -- C++ port of build_winograd_gemm_wmma and
 * build_winograd_gemm_reduce (library/kernels/common/conv_winograd_gemm_wmma.py):
 * the TDM + WMMA batched GEMM of the Winograd forward (K3) for gfx1250 and its
 * split-K slab reduce. See the header and the Python module for the algorithm.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_winograd_gemm_wmma.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, op_for_shape */
#include "rocke/helper_rocke.helpers.grid.h" /* rocke_chiplet_transform_chunked_runtime */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

/* conv_winograd_gemm_wmma.WMMA_K / CH_PAD / VGPR_BUDGET / _OOB / REDUCE_BLOCK. */
#define WGW_WMMA_K 32
#define WGW_CH_PAD 8
#define WGW_VGPR_BUDGET 480
#define WGW_OOB 0x7FFFFF00
#define WGW_REDUCE_BLOCK 256

namespace
{

int wgw_ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

void wgw_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason && reason_cap > 0)
        snprintf(reason, reason_cap, "%s", msg);
}

const char* wgw_str(const char* s, const char* dflt)
{
    return s ? s : dflt;
}

bool wgw_is_bf16(const char* dtype)
{
    return dtype != NULL && strcmp(dtype, "bf16") == 0;
}

const rocke_type_t* wgw_io_type(const char* dtype)
{
    return wgw_is_bf16(dtype) ? rocke_bf16() : rocke_f16();
}

int wgw_pitch(const rocke_winograd_gemm_wmma_spec_t* s)
{
    return s->tile_k + WGW_CH_PAD;
}

int wgw_stage_elems(const rocke_winograd_gemm_wmma_spec_t* s)
{
    return (rocke_winograd_gemm_wmma_tile_m(s) + rocke_winograd_gemm_wmma_tile_n(s)) * wgw_pitch(s);
}

int wgw_out_elem_bytes(const rocke_winograd_gemm_wmma_spec_t* s)
{
    return s->split_k ? 4 : 2;
}

int wgw_out_pitch(const rocke_winograd_gemm_wmma_spec_t* s)
{
    return rocke_winograd_gemm_wmma_tile_n(s) + 16 / wgw_out_elem_bytes(s);
}

int wgw_waves(const rocke_winograd_gemm_wmma_spec_t* s)
{
    return s->waves_m * s->waves_n;
}

int wgw_vgpr_estimate(const rocke_winograd_gemm_wmma_spec_t* s)
{
    int frags = 8 * (s->frags_m + s->frags_n);
    return 8 * s->frags_m * s->frags_n + frags * (s->prefetch ? 2 : 1);
}

bool wgw_sched_set(const rocke_winograd_gemm_wmma_spec_t* s)
{
    /* Python truthiness of `sched`: None and "" are off. */
    return s->sched != NULL && s->sched[0] != '\0';
}

} // namespace

/* ===================================================================== *
 *  Spec defaults + derived geometry
 * ===================================================================== */

rocke_winograd_gemm_wmma_spec_t rocke_winograd_gemm_wmma_spec_default(void)
{
    rocke_winograd_gemm_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.dtype = "fp16";
    s.name = "wino_gemm_wmma";
    s.waves_m = 2;
    s.waves_n = 4;
    s.frags_m = 8;
    s.frags_n = 4;
    s.tile_k = 128;
    s.stages = 2;
    s.lds_pad = "tdm";
    s.epilogue = "tdm";
    s.split_k = false;
    s.prefetch = true;
    s.xcd_remap = true;
    s.num_xcds = 8;
    s.sched = "iterative-ilp";
    s.wave_size = 32;
    s.taps = 1;
    return s;
}

int rocke_winograd_gemm_wmma_tile_m(const rocke_winograd_gemm_wmma_spec_t* spec)
{
    return spec->waves_m * spec->frags_m * 16;
}

int rocke_winograd_gemm_wmma_tile_n(const rocke_winograd_gemm_wmma_spec_t* spec)
{
    return spec->waves_n * spec->frags_n * 16;
}

int rocke_winograd_gemm_wmma_threads_per_block(const rocke_winograd_gemm_wmma_spec_t* spec)
{
    return wgw_waves(spec) * spec->wave_size;
}

long rocke_winograd_gemm_wmma_lds_bytes(const rocke_winograd_gemm_wmma_spec_t* spec)
{
    long ring = 2L * spec->stages * wgw_stage_elems(spec);
    long epi = (long)rocke_winograd_gemm_wmma_tile_m(spec) * wgw_out_pitch(spec)
               * wgw_out_elem_bytes(spec);
    if(strcmp(wgw_str(spec->epilogue, "tdm"), "tdm") != 0)
        epi = 0;
    return ring > epi ? ring : epi;
}

void rocke_winograd_gemm_wmma_grid(const rocke_winograd_gemm_wmma_spec_t* spec,
                                   int T,
                                   int G,
                                   int Cg,
                                   int Kg,
                                   int splits,
                                   int nb,
                                   int grid[3])
{
    int n = spec->taps * wgw_ceil_div(Cg, spec->tile_k);
    int s = 1;
    if(spec->split_k)
    {
        int sp = splits < 1 ? 1 : (splits > n ? n : splits);
        int cps = wgw_ceil_div(n, sp);
        s = wgw_ceil_div(n, cps);
    }
    if(nb <= 0)
        nb = G;
    grid[0] = wgw_ceil_div(T, rocke_winograd_gemm_wmma_tile_m(spec))
              * wgw_ceil_div(Kg, rocke_winograd_gemm_wmma_tile_n(spec)) * nb * s;
    grid[1] = 1;
    grid[2] = 1;
}

rocke_status_t rocke_winograd_gemm_wmma_kernel_name(const rocke_winograd_gemm_wmma_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap)
{
    char wm[32], wn[32], fm[32], fn[32], tk[32], st[32], xc[32], taps[32];
    std::string pad = std::string("p") + wgw_str(spec->lds_pad, "tdm");
    std::string sched;
    snprintf(wm, sizeof wm, "wm%d", spec->waves_m);
    snprintf(wn, sizeof wn, "wn%d", spec->waves_n);
    snprintf(fm, sizeof fm, "fm%d", spec->frags_m);
    snprintf(fn, sizeof fn, "fn%d", spec->frags_n);
    snprintf(tk, sizeof tk, "tk%d", spec->tile_k);
    snprintf(st, sizeof st, "st%d", spec->stages);
    if(spec->xcd_remap)
        snprintf(xc, sizeof xc, "x%d", spec->num_xcds);
    else
        xc[0] = '\0';
    if(wgw_sched_set(spec))
        sched = std::string("s") + spec->sched;
    if(spec->taps > 1)
        snprintf(taps, sizeof taps, "taps%d", spec->taps);
    else
        taps[0] = '\0';
    const char* parts[] = {wm,
                           wn,
                           fm,
                           fn,
                           tk,
                           st,
                           pad.c_str(),
                           wgw_str(spec->epilogue, "tdm"),
                           spec->split_k ? "sk" : "",
                           spec->prefetch ? "pf" : "",
                           xc,
                           sched.c_str(),
                           taps};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {wgw_is_bf16(spec->dtype) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "wino_gemm_wmma",
                                  parts,
                                  sizeof(parts) / sizeof(parts[0]),
                                  flag_names,
                                  flag_on,
                                  1,
                                  out,
                                  out_cap,
                                  NULL);
}

/* ===================================================================== *
 *  winograd_gemm_reason
 * ===================================================================== */

bool rocke_winograd_gemm_wmma_is_valid_spec(const rocke_winograd_gemm_wmma_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const char* dtype;
    const char* ab;
    const char* epi;
    const char* pad;
    int tk;

    if(spec == NULL)
    {
        wgw_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx1250";
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(target->wave_size != 32 || spec->wave_size != 32)
    {
        wgw_set_reason(reason, reason_cap, "wino_gemm_wmma is a wave32 kernel");
        return false;
    }
    if(!target->memory.has_tdm)
    {
        snprintf(msg, sizeof msg, "no TDM on %s", arch);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    dtype = wgw_str(spec->dtype, "fp16");
    if(strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0)
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", dtype);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    ab = wgw_is_bf16(dtype) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "wmma", ab, ab, "fp32", 16, 16, WGW_WMMA_K))
    {
        snprintf(msg, sizeof msg, "missing wmma_f32_16x16x32_%s on %s", ab, arch);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves_m < 1 || spec->waves_n < 1 || spec->frags_m < 1 || spec->frags_n < 1)
    {
        wgw_set_reason(reason, reason_cap, "waves and fragments must be >= 1");
        return false;
    }
    tk = spec->tile_k;
    if(tk <= 0 || tk % WGW_WMMA_K != 0)
    {
        snprintf(msg, sizeof msg, "tile_k %d must be a positive multiple of %d", tk, WGW_WMMA_K);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->stages < 2)
    {
        wgw_set_reason(reason, reason_cap, "stages must be >= 2");
        return false;
    }
    if(spec->taps < 1)
    {
        wgw_set_reason(reason, reason_cap, "taps must be >= 1");
        return false;
    }
    epi = wgw_str(spec->epilogue, "");
    if(strcmp(epi, "tdm") != 0 && strcmp(epi, "direct") != 0)
    {
        snprintf(msg, sizeof msg, "epilogue '%s' not in ('tdm', 'direct')", epi);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    pad = wgw_str(spec->lds_pad, "");
    if(strcmp(pad, "tdm") != 0 && strcmp(pad, "clip") != 0)
    {
        snprintf(msg, sizeof msg, "lds_pad '%s' not in ('tdm', 'clip')", pad);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(strcmp(pad, "tdm") == 0 && ((tk & (tk - 1)) != 0 || !(32 <= tk && tk <= 512)))
    {
        wgw_set_reason(reason, reason_cap, "lds_pad tdm needs a power-of-two tile_k in 32..512");
        return false;
    }
    if(spec->num_xcds < 1)
    {
        snprintf(msg, sizeof msg, "num_xcds %d < 1", spec->num_xcds);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(rocke_winograd_gemm_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(msg,
                 sizeof msg,
                 "%d threads per block",
                 rocke_winograd_gemm_wmma_threads_per_block(spec));
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(rocke_winograd_gemm_wmma_tile_m(spec) > 0xFFFF
       || rocke_winograd_gemm_wmma_tile_n(spec) > 0xFFFF || wgw_pitch(spec) > 0xFFFF
       || wgw_out_pitch(spec) > 0xFFFF)
    {
        wgw_set_reason(reason, reason_cap, "tile too large for a TDM descriptor");
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_winograd_gemm_wmma_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof msg,
                 "LDS %ld B > %d",
                 rocke_winograd_gemm_wmma_lds_bytes(spec),
                 target->lds_capacity_bytes);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(wgw_vgpr_estimate(spec) > WGW_VGPR_BUDGET * (wgw_waves(spec) <= 4 ? 2 : 1))
    {
        snprintf(
            msg, sizeof msg, "~%d VGPRs of accumulators and fragments", wgw_vgpr_estimate(spec));
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->sched != NULL && !rocke_scheduler_strategy_is_valid(spec->sched))
    {
        snprintf(msg, sizeof msg, "unsupported scheduler_strategy '%s'", spec->sched);
        wgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    wgw_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  Builder
 * ===================================================================== */
namespace
{

struct wgw_tile
{
    rocke_value_t *m0, *n0, *g, *split, *m_ext, *n_ext, *v_base, *u_base, *d_base;
};

struct wgw_ctx
{
    rocke_ir_builder_t* b;
    const rocke_winograd_gemm_wmma_spec_t* spec;
    const rocke_type_t* io_type;
    const char* wmma_op_id;

    int WAVE, WN, FM, FN, BM, BN, CK, NS, P, STAGE, U_OFF, KSTEPS, U_WAVE, OPS_PER_CHUNK;
    int TAPS, EB;
    bool split_k;

    rocke_value_t *V, *U, *D;
    rocke_value_t *c_T, *c_G, *c_Cg, *c_Kg, *c_splits, *c_cps, *c_g0, *c_nb;
    rocke_value_t *c0, *c1, *c2, *c8, *c16;
    rocke_value_t *tid, *wave_id, *lane, *l16, *lhalf, *wave_n, *wave_m;
    rocke_value_t *nx, *ny, *nxy, *n_tiles, *n_cc, *n_chunks, *row_v, *out_row;
    rocke_value_t *chunk0, *n_loc, *lds, *lds_addr, *col;
    wgw_tile cur;
    int tile0, pad_interval, pad_amount;
    std::vector<rocke_value_t*> v_off, u_off;
};

rocke_value_t* ci(wgw_ctx& c, int64_t v)
{
    return rocke_b_const_i32(c.b, v);
}

/* _remap(f, n): the XCD-contiguous remap (identity without xcd_remap). */
rocke_value_t* remap(wgw_ctx& c, rocke_value_t* f, rocke_value_t* n)
{
    if(!c.spec->xcd_remap)
        return f;
    return rocke_chiplet_transform_chunked_runtime(c.b, f, n, c.spec->num_xcds, NULL);
}

/* _tile(f): coordinates of flat tile f. */
wgw_tile tile_of(wgw_ctx& c, rocke_value_t* f)
{
    rocke_ir_builder_t* b = c.b;
    wgw_tile t;
    rocke_value_t* bx = rocke_b_mod(b, f, c.nx);
    rocke_value_t* by;
    {
        rocke_value_t* x = rocke_b_mod(b, f, c.nxy);
        by = rocke_b_div(b, x, c.nx);
    }
    rocke_value_t* z = rocke_b_div(b, f, c.nxy);
    {
        rocke_value_t* x = ci(c, c.BM);
        t.m0 = rocke_b_mul(b, bx, x);
    }
    {
        rocke_value_t* x = ci(c, c.BN);
        t.n0 = rocke_b_mul(b, by, x);
    }
    {
        rocke_value_t* x = rocke_b_mod(b, z, c.c_nb);
        t.g = rocke_b_add(b, c.c_g0, x);
    }
    t.split = rocke_b_div(b, z, c.c_nb);
    {
        rocke_value_t* x = rocke_b_mul(b, t.m0, c.out_row);
        rocke_value_t* y = rocke_b_mul(b, t.g, c.c_Kg);
        rocke_value_t* w = rocke_b_add(b, y, t.n0);
        t.d_base = rocke_b_add(b, x, w);
    }
    if(c.split_k)
    {
        rocke_value_t* x = rocke_b_mul(b, c.c_T, c.out_row);
        rocke_value_t* y = rocke_b_mul(b, t.split, x);
        t.d_base = rocke_b_add(b, t.d_base, y);
    }
    t.m_ext = rocke_b_sub(b, c.c_T, t.m0);
    t.n_ext = rocke_b_sub(b, c.c_Kg, t.n0);
    {
        rocke_value_t* x = rocke_b_mul(b, t.m0, c.row_v);
        rocke_value_t* y = rocke_b_mul(b, t.g, c.c_Cg);
        t.v_base = rocke_b_add(b, x, y);
    }
    {
        rocke_value_t* x = rocke_b_mul(b, t.g, c.c_Kg);
        rocke_value_t* y = rocke_b_add(b, x, t.n0);
        t.u_base = rocke_b_mul(b, y, c.c_Cg);
    }
    return t;
}

/* _issue(q): TDM local chunk q into ring slot q % NS. */
void issue(wgw_ctx& c, rocke_value_t* q)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* slot;
    rocke_value_t* slot_addr;
    {
        rocke_value_t* x = ci(c, c.NS);
        slot = rocke_b_mod(b, q, x);
    }
    {
        rocke_value_t* x = ci(c, (int64_t)c.STAGE * 2);
        rocke_value_t* y = rocke_b_mul(b, slot, x);
        slot_addr = rocke_b_add(b, c.lds_addr, y);
    }
    const int waves[2] = {0, c.U_WAVE};
    for(int opnd = 0; opnd < 2; ++opnd)
    {
        rocke_value_t* w = ci(c, waves[opnd]);
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, w);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        {
            const wgw_tile& t = c.cur;
            rocke_value_t *c_k0, *v_base, *v_ext, *u_base;
            if(c.TAPS > 1)
            {
                rocke_value_t* gq = rocke_b_add(b, c.chunk0, q);
                rocke_value_t* tap = rocke_b_div(b, gq, c.n_cc);
                {
                    rocke_value_t* x = rocke_b_mul(b, tap, c.n_cc);
                    rocke_value_t* y = rocke_b_sub(b, gq, x);
                    rocke_value_t* z = ci(c, c.CK);
                    c_k0 = rocke_b_mul(b, y, z);
                }
                {
                    rocke_value_t* x = rocke_b_mul(b, tap, c.row_v);
                    v_base = rocke_b_add(b, t.v_base, x);
                }
                {
                    rocke_value_t* x = ci(c, c.TAPS - 1);
                    rocke_value_t* y = rocke_b_add(b, t.m_ext, x);
                    v_ext = rocke_b_sub(b, y, tap);
                }
                rocke_value_t* slab;
                {
                    rocke_value_t* x = ci(c, c.TAPS - 1);
                    rocke_value_t* y = rocke_b_mul(b, t.g, x);
                    slab = rocke_b_add(b, y, tap);
                }
                {
                    rocke_value_t* x = rocke_b_mul(b, c.c_Kg, c.c_Cg);
                    rocke_value_t* y = rocke_b_mul(b, slab, x);
                    u_base = rocke_b_add(b, y, t.u_base);
                }
            }
            else
            {
                rocke_value_t* x = rocke_b_add(b, c.chunk0, q);
                rocke_value_t* y = ci(c, c.CK);
                c_k0 = rocke_b_mul(b, x, y);
                v_base = t.v_base;
                v_ext = t.m_ext;
                u_base = t.u_base;
            }
            rocke_value_t* k_ext;
            {
                rocke_value_t* x = rocke_b_sub(b, c.c_Cg, c_k0);
                rocke_value_t* y = ci(c, c.CK);
                k_ext = rocke_b_smin(b, x, y);
            }
            rocke_value_t* ptr = opnd == 0 ? c.V : c.U;
            rocke_value_t* base = opnd == 0 ? v_base : u_base;
            rocke_value_t* ext = opnd == 0 ? v_ext : t.n_ext;
            rocke_value_t* stride = opnd == 0 ? c.row_v : c.c_Cg;
            int rows = opnd == 0 ? c.BM : c.BN;
            int lds_off = opnd == 0 ? 0 : c.U_OFF;
            rocke_value_t* gptr;
            {
                rocke_value_t* x = rocke_b_add(b, base, c_k0);
                rocke_value_t* y = rocke_b_mul(b, x, c.c2);
                gptr = rocke_b_global_ptr_add(b, ptr, y);
            }
            rocke_value_t* laddr;
            {
                rocke_value_t* x = ci(c, (int64_t)lds_off * 2);
                laddr = rocke_b_add(b, slot_addr, x);
            }
            rocke_value_t* d[5];
            if(rocke_b_tdm_descriptor_2d(b,
                                         gptr,
                                         laddr,
                                         2,
                                         k_ext,
                                         ext,
                                         stride,
                                         c.tile0,
                                         rows,
                                         0,
                                         c.pad_interval,
                                         c.pad_amount,
                                         d))
                rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        }
        rocke_b_region_leave(b);
    }
}

/* _wait_for(q). */
void wait_for(wgw_ctx& c, rocke_value_t* q)
{
    rocke_ir_builder_t* b = c.b;
    if(c.NS == 2)
    {
        rocke_b_s_wait_tensorcnt(b, 0);
        return;
    }
    rocke_value_t* x = ci(c, c.NS - 2);
    rocke_value_t* y = rocke_b_add(b, q, x);
    rocke_value_t* more = rocke_b_cmp_lt(b, y, c.n_loc);
    rocke_if_else_t ie = rocke_b_scf_if_else(b, more);
    rocke_b_region_enter(b, ie.then_region);
    rocke_b_s_wait_tensorcnt(b, (c.NS - 2) * c.OPS_PER_CHUNK);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, ie.else_region);
    rocke_b_s_wait_tensorcnt(b, 0);
    rocke_b_region_leave(b);
}

/* _frag_off(wave_pos, nfrag, f, extra). */
rocke_value_t* frag_off(wgw_ctx& c, rocke_value_t* wave_pos, int nfrag, int f, int extra)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* row;
    {
        rocke_value_t* x = ci(c, nfrag);
        rocke_value_t* y = rocke_b_mul(b, wave_pos, x);
        rocke_value_t* z = ci(c, f);
        rocke_value_t* u = rocke_b_add(b, y, z);
        rocke_value_t* v = rocke_b_mul(b, u, c.c16);
        row = rocke_b_add(b, v, c.l16);
    }
    rocke_value_t* x = ci(c, c.P);
    rocke_value_t* y = rocke_b_mul(b, row, x);
    rocke_value_t* z = ci(c, extra);
    rocke_value_t* u = rocke_b_add(b, c.col, z);
    return rocke_b_add(b, y, u);
}

/* _read(base, off, ks). */
rocke_value_t* read_frag(wgw_ctx& c, rocke_value_t* base, rocke_value_t* off, int ks)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* addr;
    {
        rocke_value_t* x = ci(c, (int64_t)ks * WGW_WMMA_K);
        rocke_value_t* y = rocke_b_add(b, off, x);
        addr = rocke_b_add(b, base, y);
    }
    rocke_value_t* idx0[2] = {c.c0, addr};
    rocke_value_t* lo = rocke_b_smem_load_vN(b, c.lds, idx0, 2, c.io_type, 8);
    rocke_value_t* addr8 = rocke_b_add(b, addr, c.c8);
    rocke_value_t* idx1[2] = {c.c0, addr8};
    rocke_value_t* hi = rocke_b_smem_load_vN(b, c.lds, idx1, 2, c.io_type, 8);
    return rocke_b_vec_concat(b, lo, hi);
}

/* _frags(q, ks): FN U fragments, then FM V fragments. */
std::vector<rocke_value_t*> frags(wgw_ctx& c, rocke_value_t* q, int ks)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* base;
    {
        rocke_value_t* x = ci(c, c.NS);
        rocke_value_t* y = rocke_b_mod(b, q, x);
        rocke_value_t* z = ci(c, c.STAGE);
        base = rocke_b_mul(b, y, z);
    }
    std::vector<rocke_value_t*> out;
    for(int i = 0; i < c.FN; ++i)
        out.push_back(read_frag(c, base, c.u_off[i], ks));
    for(int j = 0; j < c.FM; ++j)
        out.push_back(read_frag(c, base, c.v_off[j], ks));
    return out;
}

/* _publish(q). */
void publish(wgw_ctx& c, rocke_value_t* q)
{
    rocke_ir_builder_t* b = c.b;
    wait_for(c, q);
    rocke_b_sync_lds_only(b);
    rocke_value_t* x = ci(c, c.NS - 1);
    rocke_value_t* nxt = rocke_b_add(b, q, x);
    rocke_value_t* cond = rocke_b_cmp_lt(b, nxt, c.n_loc);
    rocke_if_t gate = rocke_b_scf_if(b, cond);
    rocke_b_region_enter(b, gate.then_region);
    issue(c, nxt);
    rocke_b_region_leave(b);
}

/* _rows_cols(i, j). */
void rows_cols(wgw_ctx& c, int i, int j, rocke_value_t** r, rocke_value_t** col)
{
    rocke_ir_builder_t* b = c.b;
    {
        rocke_value_t* x = ci(c, c.FM);
        rocke_value_t* y = rocke_b_mul(b, c.wave_m, x);
        rocke_value_t* z = ci(c, j);
        rocke_value_t* u = rocke_b_add(b, y, z);
        rocke_value_t* v = rocke_b_mul(b, u, c.c16);
        *r = rocke_b_add(b, v, c.l16);
    }
    {
        rocke_value_t* x = ci(c, c.FN);
        rocke_value_t* y = rocke_b_mul(b, c.wave_n, x);
        rocke_value_t* z = ci(c, i);
        rocke_value_t* u = rocke_b_add(b, y, z);
        rocke_value_t* v = rocke_b_mul(b, u, c.c16);
        rocke_value_t* w = rocke_b_mul(b, c.lhalf, c.c8);
        *col = rocke_b_add(b, v, w);
    }
}

/* _halves(acc). */
void halves(wgw_ctx& c, rocke_value_t* acc, rocke_value_t** lo, rocke_value_t** hi)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* e[4];
    for(int k = 0; k < 4; ++k)
        e[k] = rocke_b_vec_extract(b, acc, k);
    *lo = rocke_b_vec_pack(b, e, 4, rocke_f32());
    for(int k = 0; k < 4; ++k)
        e[k] = rocke_b_vec_extract(b, acc, 4 + k);
    *hi = rocke_b_vec_pack(b, e, 4, rocke_f32());
}

void epilogue_tdm(wgw_ctx& c, rocke_value_t* const* accs)
{
    rocke_ir_builder_t* b = c.b;
    const wgw_tile& t = c.cur;
    int OP = wgw_out_pitch(c.spec);
    int HPE = c.EB / 2;
    rocke_b_sync_lds_only(b);
    for(int i = 0; i < c.FN; ++i)
        for(int j = 0; j < c.FM; ++j)
        {
            rocke_value_t *r, *col;
            rows_cols(c, i, j, &r, &col);
            rocke_value_t* acc = accs[i * c.FM + j];
            rocke_value_t* idx;
            {
                rocke_value_t* x = ci(c, OP);
                rocke_value_t* y = rocke_b_mul(b, r, x);
                rocke_value_t* z = rocke_b_add(b, y, col);
                rocke_value_t* u = ci(c, HPE);
                idx = rocke_b_mul(b, z, u);
            }
            if(c.split_k)
            {
                rocke_value_t *lo, *hi;
                halves(c, acc, &lo, &hi);
                rocke_value_t* i0[2] = {c.c0, idx};
                rocke_b_smem_store_vN_f32(b, c.lds, i0, 2, lo, 4);
                rocke_value_t* idx8 = rocke_b_add(b, idx, c.c8);
                rocke_value_t* i1[2] = {c.c0, idx8};
                rocke_b_smem_store_vN_f32(b, c.lds, i1, 2, hi, 4);
            }
            else
            {
                rocke_value_t* val = rocke_b_vec_cast_f32_to(b, acc, c.io_type);
                rocke_value_t* i0[2] = {c.c0, idx};
                rocke_b_smem_store_vN(b, c.lds, i0, 2, val, 8);
            }
        }
    rocke_b_sync_lds_only(b);
    rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, c.c0);
    rocke_if_t gate = rocke_b_scf_if(b, cond);
    rocke_b_region_enter(b, gate.then_region);
    {
        rocke_value_t* dst;
        {
            rocke_value_t* x = ci(c, c.EB);
            rocke_value_t* y = rocke_b_mul(b, t.d_base, x);
            dst = rocke_b_global_ptr_add(b, c.D, y);
        }
        rocke_value_t* dim0;
        {
            rocke_value_t* x = ci(c, c.BN);
            dim0 = rocke_b_smin(b, t.n_ext, x);
        }
        rocke_value_t* d[5];
        if(rocke_b_tdm_descriptor_2d(
               b, dst, c.lds_addr, c.EB, dim0, t.m_ext, c.out_row, OP, c.BM, 0, -1, 0, d))
            rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        rocke_b_s_wait_tensorcnt(b, 0);
    }
    rocke_b_region_leave(b);
}

void epilogue_direct(wgw_ctx& c, rocke_value_t* const* accs)
{
    rocke_ir_builder_t* b = c.b;
    const wgw_tile& t = c.cur;
    rocke_value_t* d_bytes;
    {
        rocke_value_t* x = rocke_b_mul(b, c.c_T, c.out_row);
        rocke_value_t* y = ci(c, c.EB);
        d_bytes = rocke_b_mul(b, x, y);
    }
    if(c.split_k)
        d_bytes = rocke_b_mul(b, d_bytes, c.c_splits);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, c.D, d_bytes);
    rocke_value_t* oob = ci(c, WGW_OOB);
    for(int i = 0; i < c.FN; ++i)
        for(int j = 0; j < c.FM; ++j)
        {
            rocke_value_t *r, *col;
            rows_cols(c, i, j, &r, &col);
            rocke_value_t* ok;
            {
                rocke_value_t* x = rocke_b_add(b, t.m0, r);
                rocke_value_t* y = rocke_b_cmp_lt(b, x, c.c_T);
                rocke_value_t* z = rocke_b_add(b, t.n0, col);
                rocke_value_t* u = rocke_b_cmp_lt(b, z, c.c_Kg);
                ok = rocke_b_land(b, y, u);
            }
            rocke_value_t* off;
            {
                rocke_value_t* x = rocke_b_mul(b, r, c.out_row);
                rocke_value_t* y = rocke_b_add(b, x, col);
                off = rocke_b_add(b, t.d_base, y);
            }
            {
                rocke_value_t* x = ci(c, c.EB);
                rocke_value_t* y = rocke_b_mul(b, off, x);
                off = rocke_b_select(b, ok, y, oob);
            }
            rocke_value_t* acc = accs[i * c.FM + j];
            if(c.split_k)
            {
                rocke_value_t *lo, *hi;
                halves(c, acc, &lo, &hi);
                rocke_b_buffer_store_vN_f32(b, rsrc, off, c.c0, lo, 4);
                rocke_value_t* c16b = ci(c, 16);
                rocke_b_buffer_store_vN_f32(b, rsrc, off, c16b, hi, 4);
            }
            else if(wgw_is_bf16(c.spec->dtype))
            {
                rocke_value_t* val = rocke_b_vec_cast_f32_to(b, acc, c.io_type);
                rocke_b_buffer_store_vN_bf16(b, rsrc, off, c.c0, val, 4);
            }
            else
            {
                rocke_value_t* val = rocke_b_vec_cast_f32_to(b, acc, c.io_type);
                rocke_b_buffer_store_vN_f16(b, rsrc, off, c.c0, val, 4);
            }
        }
}

rocke_value_t* ptr_param(rocke_ir_builder_t* b,
                         const char* name,
                         const rocke_type_t* elem,
                         bool readonly,
                         bool writeonly)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    if(writeonly)
    {
        o.writeonly = true;
        o.writeonly_set = true;
    }
    o.align = 16;
    o.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

} // namespace

rocke_kernel_def_t* rocke_build_winograd_gemm_wmma(rocke_ir_builder_t* b,
                                                   const rocke_winograd_gemm_wmma_spec_t* spec,
                                                   const char* arch)
{
    char reason[512];
    wgw_ctx c;

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx1250";
    if(!rocke_winograd_gemm_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid WinogradGemmWmmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    c.b = b;
    c.spec = spec;
    c.io_type = wgw_io_type(spec->dtype);
    {
        const char* ab = wgw_is_bf16(spec->dtype) ? "bf16" : "f16";
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        const rocke_mmaop_t* op = rocke_archtarget_op_for_shape(
            target, "wmma", ab, ab, "fp32", 16, 16, WGW_WMMA_K, NULL);
        if(op == NULL || op->op_id == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_gemm_wmma: no WMMA atom");
            return NULL;
        }
        c.wmma_op_id = op->op_id;
    }
    c.WAVE = spec->wave_size;
    c.WN = spec->waves_n;
    c.FM = spec->frags_m;
    c.FN = spec->frags_n;
    c.BM = rocke_winograd_gemm_wmma_tile_m(spec);
    c.BN = rocke_winograd_gemm_wmma_tile_n(spec);
    c.CK = spec->tile_k;
    c.NS = spec->stages;
    c.P = wgw_pitch(spec);
    c.STAGE = wgw_stage_elems(spec);
    c.U_OFF = c.BM * c.P;
    c.KSTEPS = c.CK / WGW_WMMA_K;
    c.U_WAVE = wgw_waves(spec) > 1 ? 1 : 0;
    c.OPS_PER_CHUNK = wgw_waves(spec) > 1 ? 1 : 2;
    c.TAPS = spec->taps;
    c.split_k = spec->split_k;
    c.EB = wgw_out_elem_bytes(spec);

    rocke_attr_set_int(b,
                       &b->kernel->attrs,
                       "max_workgroup_size",
                       rocke_winograd_gemm_wmma_threads_per_block(spec));
    if(wgw_sched_set(spec))
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->sched);
    const rocke_type_t* out_ty = c.split_k ? rocke_f32() : c.io_type;
    c.V = ptr_param(b, "v", c.io_type, true, false);
    c.U = ptr_param(b, "u", c.io_type, true, false);
    c.D = ptr_param(b, "d", out_ty, false, true);
    c.c_T = rocke_b_param(b, "T", rocke_i32(), NULL);
    c.c_G = rocke_b_param(b, "G", rocke_i32(), NULL);
    c.c_Cg = rocke_b_param(b, "Cg", rocke_i32(), NULL);
    c.c_Kg = rocke_b_param(b, "Kg", rocke_i32(), NULL);
    c.c_splits = rocke_b_param(b, "splits", rocke_i32(), NULL);
    c.c_cps = rocke_b_param(b, "cps", rocke_i32(), NULL);
    c.c_g0 = rocke_b_param(b, "g0", rocke_i32(), NULL);
    c.c_nb = rocke_b_param(b, "nb", rocke_i32(), NULL);

    c.c0 = ci(c, 0);
    c.c1 = ci(c, 1);
    c.c2 = ci(c, 2);
    c.c8 = ci(c, 8);
    c.c16 = ci(c, 16);
    c.tid = rocke_b_thread_id_x(b);
    {
        rocke_value_t* x = ci(c, c.WAVE);
        c.wave_id = rocke_b_div(b, c.tid, x);
        rocke_value_t* y = ci(c, c.WAVE);
        c.lane = rocke_b_mod(b, c.tid, y);
    }
    c.l16 = rocke_b_mod(b, c.lane, c.c16);
    c.lhalf = rocke_b_div(b, c.lane, c.c16);
    {
        rocke_value_t* x = ci(c, c.WN);
        c.wave_n = rocke_b_mod(b, c.wave_id, x);
        rocke_value_t* y = ci(c, c.WN);
        c.wave_m = rocke_b_div(b, c.wave_id, y);
    }
    {
        rocke_value_t* x = ci(c, c.BM - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_T, x);
        rocke_value_t* z = ci(c, c.BM);
        c.nx = rocke_b_div(b, y, z);
    }
    {
        rocke_value_t* x = ci(c, c.BN - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_Kg, x);
        rocke_value_t* z = ci(c, c.BN);
        c.ny = rocke_b_div(b, y, z);
    }
    c.nxy = rocke_b_mul(b, c.nx, c.ny);
    {
        rocke_value_t* x = rocke_b_mul(b, c.c_nb, c.c_splits);
        c.n_tiles = rocke_b_mul(b, c.nxy, x);
    }
    {
        rocke_value_t* x = ci(c, c.CK - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_Cg, x);
        rocke_value_t* z = ci(c, c.CK);
        c.n_cc = rocke_b_div(b, y, z);
    }
    if(c.TAPS > 1)
    {
        rocke_value_t* x = ci(c, c.TAPS);
        c.n_chunks = rocke_b_mul(b, c.n_cc, x);
    }
    else
    {
        c.n_chunks = c.n_cc;
    }
    c.row_v = rocke_b_mul(b, c.c_G, c.c_Cg);
    c.out_row = rocke_b_mul(b, c.c_G, c.c_Kg);

    {
        rocke_value_t* bid = rocke_b_block_id_x(b);
        rocke_value_t* f = remap(c, bid, c.n_tiles);
        if(f == NULL)
            return NULL;
        c.cur = tile_of(c, f);
    }
    c.chunk0 = rocke_b_mul(b, c.cur.split, c.c_cps);
    {
        rocke_value_t* x = rocke_b_sub(b, c.n_chunks, c.chunk0);
        rocke_value_t* y = rocke_b_smin(b, x, c.c_cps);
        c.n_loc = rocke_b_smax(b, y, c.c0);
    }
    {
        int shape[2] = {1, (int)(rocke_winograd_gemm_wmma_lds_bytes(spec) / 2)};
        c.lds = rocke_b_smem_alloc(b, c.io_type, shape, 2, "wg_lds");
    }
    {
        rocke_value_t* a = rocke_b_smem_addr_of(b, c.lds);
        c.lds_addr = rocke_b_trunc(b, a, rocke_i32());
    }
    c.tile0 = c.P;
    c.pad_interval = -1;
    c.pad_amount = 0;
    if(strcmp(wgw_str(spec->lds_pad, "tdm"), "tdm") == 0)
    {
        /* pad_interval = (CK // 2).bit_length() - 2 */
        int v = c.CK / 2;
        int bits = 0;
        while(v)
        {
            ++bits;
            v >>= 1;
        }
        c.pad_interval = bits - 2;
        c.pad_amount = 3;
        c.tile0 = c.CK;
    }

    /* ---- per-lane fragment offsets. */
    c.col = rocke_b_mul(b, c.lhalf, c.c16);
    for(int j = 0; j < c.FM; ++j)
        c.v_off.push_back(frag_off(c, c.wave_m, c.FM, j, 0));
    for(int i = 0; i < c.FN; ++i)
        c.u_off.push_back(frag_off(c, c.wave_n, c.FN, i, c.U_OFF));

    /* ---- main loop. */
    for(int st = 0; st < c.NS - 1; ++st)
    {
        rocke_value_t* x = ci(c, st);
        rocke_value_t* cond = rocke_b_cmp_lt(b, x, c.n_loc);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        issue(c, ci(c, st));
        rocke_b_region_leave(b);
    }
    int n_acc = c.FN * c.FM;
    rocke_value_t* zero_acc = rocke_b_zero_vec_f32(b, 8);
    std::vector<std::string> names(n_acc);
    std::vector<rocke_iter_arg_t> iargs(n_acc);
    for(int k = 0; k < n_acc; ++k)
    {
        names[k] = "acc" + std::to_string(k);
        iargs[k].name = names[k].c_str();
        iargs[k].init = zero_acc;
    }
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, c.c0, c.n_loc, c.c1, iargs.data(), n_acc, "q", false, true);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, loop.body);
    {
        std::vector<rocke_value_t*> accs(loop.iter_vars, loop.iter_vars + n_acc);
        rocke_value_t* q = loop.iv;
        publish(c, q);
        rocke_b_s_setprio(b, 1);
        std::vector<rocke_value_t*> fr = frags(c, q, 0);
        for(int ks = 0; ks < c.KSTEPS; ++ks)
        {
            std::vector<rocke_value_t*> nxt;
            if(spec->prefetch)
            {
                if(ks + 1 < c.KSTEPS)
                    nxt = frags(c, q, ks + 1);
            }
            else if(ks)
            {
                fr = frags(c, q, ks);
            }
            for(int i = 0; i < c.FN; ++i)
                for(int j = 0; j < c.FM; ++j)
                {
                    int k = i * c.FM + j;
                    accs[k] = rocke_b_mma(b, c.wmma_op_id, fr[i], fr[c.FN + j], accs[k], NULL, 0);
                }
            fr = nxt;
        }
        rocke_b_s_setprio(b, 0);
        rocke_b_scf_yield(b, accs.data(), n_acc);
    }
    rocke_b_region_leave(b);
    if(strcmp(wgw_str(spec->epilogue, "tdm"), "tdm") == 0)
        epilogue_tdm(c, loop.op->results);
    else
        epilogue_direct(c, loop.op->results);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_winograd_gemm_wmma_new(rocke_ir_builder_t* b,
                                                       const rocke_winograd_gemm_wmma_spec_t* spec,
                                                       const char* arch)
{
    if(b != NULL)
        memset(b, 0, sizeof(*b));
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        if(b == NULL)
            return NULL;
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_gemm_wmma: null spec");
            return NULL;
        }
        if(rocke_winograd_gemm_wmma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_winograd_gemm_wmma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_winograd_gemm_wmma(b, spec, arch);
    });
}

namespace
{

template <typename BuildFn>
rocke_status_t wgw_lower(BuildFn build,
                         const char* arch,
                         rocke_llvm_flavor_t flavor,
                         char** out_ll,
                         char* err,
                         size_t err_cap,
                         const char* what)
{
    rocke_ir_builder_t b = {};
    rocke_kernel_def_t* kernel;
    rocke_status_t st;
    if(out_ll != NULL)
        *out_ll = NULL;
    if(out_ll == NULL)
    {
        wgw_set_reason(err, err_cap, "lower_to_llvm: null out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    kernel = build(&b, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        wgw_set_reason(err, err_cap, (m != NULL && m[0] != '\0') ? m : what);
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

} // namespace

rocke_status_t rocke_winograd_gemm_wmma_lower_to_llvm(const rocke_winograd_gemm_wmma_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap)
{
    if(spec == NULL)
    {
        if(out_ll != NULL)
            *out_ll = NULL;
        wgw_set_reason(err, err_cap, "lower_to_llvm: null spec");
        return ROCKE_ERR_VALUE;
    }
    return wgw_lower([&](rocke_ir_builder_t* b,
                         const char* a) { return rocke_build_winograd_gemm_wmma_new(b, spec, a); },
                     arch,
                     flavor,
                     out_ll,
                     err,
                     err_cap,
                     "build_winograd_gemm_wmma failed");
}

/* ===================================================================== *
 *  Split-K reduce
 * ===================================================================== */

rocke_winograd_gemm_reduce_spec_t rocke_winograd_gemm_reduce_spec_default(void)
{
    rocke_winograd_gemm_reduce_spec_t s;
    s.dtype = "fp16";
    return s;
}

rocke_status_t rocke_winograd_gemm_reduce_kernel_name(const rocke_winograd_gemm_reduce_spec_t* spec,
                                                      char* out,
                                                      size_t out_cap)
{
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {wgw_is_bf16(spec->dtype) ? 1 : 0};
    return rocke_kernel_name_join(
        "wino_gemm_reduce", NULL, 0, flag_names, flag_on, 1, out, out_cap, NULL);
}

rocke_kernel_def_t* rocke_build_winograd_gemm_reduce(rocke_ir_builder_t* b,
                                                     const rocke_winograd_gemm_reduce_spec_t* spec,
                                                     const char* arch)
{
    (void)arch;
    if(b == NULL || spec == NULL)
        return NULL;
    const char* dtype = wgw_str(spec->dtype, "fp16");
    const rocke_type_t* io_type = wgw_io_type(dtype);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", WGW_REDUCE_BLOCK);
    rocke_value_t* W = ptr_param(b, "w", rocke_f32(), true, false);
    rocke_value_t* D = ptr_param(b, "d", io_type, false, true);
    rocke_value_t* n_vec = rocke_b_param(b, "n_vec", rocke_i32(), NULL);
    rocke_value_t* splits = rocke_b_param(b, "splits", rocke_i32(), NULL);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* gid;
    {
        rocke_value_t* x = rocke_b_block_id_x(b);
        rocke_value_t* y = rocke_b_const_i32(b, WGW_REDUCE_BLOCK);
        rocke_value_t* z = rocke_b_mul(b, x, y);
        rocke_value_t* u = rocke_b_thread_id_x(b);
        gid = rocke_b_add(b, z, u);
    }
    rocke_value_t* ok = rocke_b_cmp_lt(b, gid, n_vec);
    rocke_value_t* oob = rocke_b_const_i32(b, WGW_OOB);
    rocke_value_t* slab;
    {
        rocke_value_t* x = rocke_b_const_i32(b, 32);
        slab = rocke_b_mul(b, n_vec, x);
    }
    rocke_value_t* w_rsrc;
    {
        rocke_value_t* x = rocke_b_mul(b, slab, splits);
        w_rsrc = rocke_b_buffer_rsrc(b, W, x);
    }
    rocke_value_t* d_rsrc;
    {
        rocke_value_t* x = rocke_b_const_i32(b, 16);
        rocke_value_t* y = rocke_b_mul(b, n_vec, x);
        d_rsrc = rocke_b_buffer_rsrc(b, D, y);
    }
    rocke_value_t* w_off;
    {
        rocke_value_t* x = rocke_b_const_i32(b, 32);
        rocke_value_t* y = rocke_b_mul(b, gid, x);
        w_off = rocke_b_select(b, ok, y, oob);
    }
    rocke_value_t* zero4 = rocke_b_zero_vec_f32(b, 4);
    rocke_iter_arg_t iargs[2] = {{"lo", zero4}, {"hi", zero4}};
    rocke_for_t loop = rocke_b_scf_for_iter(b, c0, splits, c1, iargs, 2, "s", false, true);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* s = loop.iv;
        rocke_value_t* lo = loop.iter_vars[0];
        rocke_value_t* hi = loop.iter_vars[1];
        rocke_value_t* off;
        {
            rocke_value_t* x = rocke_b_mul(b, s, slab);
            rocke_value_t* y = rocke_b_add(b, w_off, x);
            off = rocke_b_select(b, ok, y, oob);
        }
        {
            rocke_value_t* x = rocke_b_buffer_load_vN(b, w_rsrc, off, c0, rocke_f32(), 4);
            lo = rocke_b_fadd(b, lo, x);
        }
        {
            rocke_value_t* x = rocke_b_const_i32(b, 16);
            rocke_value_t* y = rocke_b_buffer_load_vN(b, w_rsrc, off, x, rocke_f32(), 4);
            hi = rocke_b_fadd(b, hi, y);
        }
        rocke_value_t* ys[2] = {lo, hi};
        rocke_b_scf_yield(b, ys, 2);
    }
    rocke_b_region_leave(b);
    rocke_value_t* val;
    {
        rocke_value_t* x = rocke_b_vec_concat(b, loop.op->results[0], loop.op->results[1]);
        val = rocke_b_vec_cast_f32_to(b, x, io_type);
    }
    rocke_value_t* d_off;
    {
        rocke_value_t* x = rocke_b_const_i32(b, 16);
        rocke_value_t* y = rocke_b_mul(b, gid, x);
        d_off = rocke_b_select(b, ok, y, oob);
    }
    if(wgw_is_bf16(dtype))
        rocke_b_buffer_store_vN_bf16(b, d_rsrc, d_off, c0, val, 4);
    else
        rocke_b_buffer_store_vN_f16(b, d_rsrc, d_off, c0, val, 4);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_winograd_gemm_reduce_new(
    rocke_ir_builder_t* b, const rocke_winograd_gemm_reduce_spec_t* spec, const char* arch)
{
    if(b != NULL)
        memset(b, 0, sizeof(*b));
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL)
            return NULL;
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_gemm_reduce: null spec");
            return NULL;
        }
        if(rocke_winograd_gemm_reduce_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_winograd_gemm_reduce: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_winograd_gemm_reduce(b, spec, arch);
    });
}

rocke_status_t
    rocke_winograd_gemm_reduce_lower_to_llvm(const rocke_winograd_gemm_reduce_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap)
{
    if(spec == NULL)
    {
        if(out_ll != NULL)
            *out_ll = NULL;
        wgw_set_reason(err, err_cap, "lower_to_llvm: null spec");
        return ROCKE_ERR_VALUE;
    }
    return wgw_lower(
        [&](rocke_ir_builder_t* b, const char* a) {
            return rocke_build_winograd_gemm_reduce_new(b, spec, a);
        },
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_winograd_gemm_reduce failed");
}
