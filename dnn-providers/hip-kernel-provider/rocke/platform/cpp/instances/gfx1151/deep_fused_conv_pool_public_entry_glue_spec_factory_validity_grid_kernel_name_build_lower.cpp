// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1151_deep_fused_conv_pool_Public entry + glue
 *   (spec/factory/validity/grid/kernel_name/build/lower).c
 *
 * The PUBLIC entry + glue bucket of the chunked C99 port of the gfx1151
 * (RDNA3.5 / Strix Halo, wave32, WMMA 16x16x16) GENUINE-int8/int4 deep-fused
 * conv0 -> conv1 -> 2x2/s2 maxpool builder
 *   (rocke/instances/gfx1151/deep_fused_conv_pool.py, 3238 LOC).
 *
 * SCOPE OF THIS PART-FILE.
 *   Unlike the gfx950 / gfx1201 shims (thin wrappers over the common builder),
 *   the gfx1151 module is its OWN full spec + builder. This bucket implements the
 *   value-type spec surface + the public build driver glue:
 *
 *     - rocke_gfx1151_deep_fused_conv_pool_make_spec / _spec_default
 *         build the ConvProblem (sH=sW=pH=pW=dH=dW=1) + FusedConvPoolProblem
 *         (conv1_k=k1), auto-derive tile_m = conv_tile_h*conv_tile_w, stamp every
 *         lever + m0/m0b/m1/mf default (Python lines 387-485, 74-296).
 *     - the derived-quantity accessors (warp_tile_*, block_size, kpad,
 *         conv_tile_h/w, foot_h/w) -- Python @property mirrors (lines 298-337).
 *     - rocke_gfx1151_deep_fused_conv_pool_kernel_name -- composes name +
 *         problem.short() + tile/pool/warp tags + "wmma16x16x16" +
 *         (directa|im2col) + non-default lever tags + "i8i4_realquant" via
 *         kernel_name_join (Python lines 339-384).
 *     - rocke_gfx1151_deep_fused_conv_pool_is_valid_spec -- the full arch/geometry/
 *         lever precondition gate (Python lines 488-629).
 *     - rocke_gfx1151_deep_fused_conv_pool_grid (Python lines 632-645).
 *     - rocke_build_gfx1151_deep_fused_conv_pool -- the public build driver: calls
 *         the peer rocke_gfx1151_dfcp_build_ctx_init, then dispatches to the
 *         persistent or single-tile body sub-phase (peer fns).
 *     - rocke_build_gfx1151_deep_fused_conv_pool_new (init builder w/ kernel_name)
 *         and rocke_gfx1151_deep_fused_conv_pool_lower_to_llvm (build -> .ll).
 *
 * The numeric body (staging / WMMA-GEMM / scatter / handoff / maxpool) and the
 * ctx init live in peer translation units, reached via the internal header.
 */
#include "rocke/instance_gfx1151_deep_fused_conv_pool.h"
#include "rocke/instance_gfx1151_deep_fused_conv_pool_internal.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* ArchTarget gate (is_valid_spec) */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* ConvProblem props */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

/* ------------------------------------------------------------------ *
 * Pinned gfx1151 module constants (Python `_WMMA` / `_WAVE` / `_OP_ID_*` /
 * `_K_PER_I32` / `_I4_PER_I32`). Local aliases of the public-header macros.
 * ------------------------------------------------------------------ */
#define GFX1151_WMMA ROCKE_GFX1151_DFCP_WMMA
#define GFX1151_WAVE ROCKE_GFX1151_DFCP_WAVE
#define GFX1151_OP_ID_IU8 ROCKE_GFX1151_DFCP_OP_ID_IU8
#define GFX1151_OP_ID_IU4 ROCKE_GFX1151_DFCP_OP_ID_IU4
#define GFX1151_K_PER_I32 ROCKE_GFX1151_DFCP_K_PER_I32
#define GFX1151_ARCH ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_ARCH
#define GFX1151_ARCH_GENERIC ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_ARCH_GENERIC

/* ===================================================================== *
 * Spec defaults + factory (Python @dataclass defaults + make_spec).
 * ===================================================================== */

/* Default-constructed spec carrying every Python dataclass default (lines
 * 74-296). The caller fills `problem` (or uses make_spec, which derives it). */
rocke_gfx1151_deep_fused_conv_pool_spec_t rocke_gfx1151_deep_fused_conv_pool_spec_default(void)
{
    rocke_gfx1151_deep_fused_conv_pool_spec_t s;

    memset(&s, 0, sizeof(s));

    /* problem left zeroed (caller fills); the FusedConvPoolProblem pool defaults
     * (pool_y=pool_x=2, pool_stride_h=pool_stride_w=2) are stamped here so the
     * derived accessors are correct on a default spec. */
    s.problem.pool_y = 2;
    s.problem.pool_x = 2;
    s.problem.pool_stride_h = 2;
    s.problem.pool_stride_w = 2;

    s.name = ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_NAME;

    /* Tile geometry (Python defaults). tile_m is auto-derived by make_spec; the
     * default value here mirrors the dataclass default 256. */
    s.tile_m = 256;
    s.tile_n = 32;
    s.pool_tile_h = 8;
    s.pool_tile_w = 8;
    s.warp_m = 4;
    s.warp_n = 2;

    /* Optimization toggles. */
    s.vectorize_conv0_a = true;
    s.vectorize_maxpool = true;
    s.early_w1 = true;
    s.direct_conv0 = true;
    s.w_fast = false;

    /* Multi-lever latency-hiding campaign toggles. */
    s.waves_per_eu = 0;
    s.sched_policy = "mem";
    s.mask_maxpool = false;

    /* Native integer cleanup levers. */
    s.specialized_rne = false;
    s.interior_fastpath = false;
    s.static_direct_kmap = false;
    s.packed_c0_handoff = false;
    s.repack_c0 = false;
    s.fused_c0a1 = false;
    s.butterfly_conv01 = false;
    s.native_int = false;
    s.batch_loads = true;
    s.pk_maxpool = false;
    s.conv1_prefetch_k = false;
    s.conv1_sched_fuse = false;
    s.conv1_int8 = false;
    s.persistent = false;
    s.persistent_ctas = 16;

    /* Per-node inverse requant multipliers. */
    s.m0 = 0.0625f;
    s.m0b = 0.5f;
    s.m1 = 0.25f;
    s.mf = 1.0f;

    return s;
}

/* make_deep_fused_conv_pool_spec(**kwargs) -> Gfx1151DeepFusedConvPoolSpec
 * (Python lines 387-485). Builds the ConvProblem (sH=sW=1, pH=pW=1, dH=dW=1) +
 * FusedConvPoolProblem(conv, conv1_k=k1), auto-derives
 *   tile_m = conv_tile_h*conv_tile_w
 *          = (pool_tile_h*pool_stride_h) * (pool_tile_w*pool_stride_w),
 * and stamps every lever / multiplier. */
rocke_gfx1151_deep_fused_conv_pool_spec_t
    rocke_gfx1151_deep_fused_conv_pool_make_spec(int n,
                                                 int h,
                                                 int w,
                                                 int c,
                                                 int k0,
                                                 int k1,
                                                 int r,
                                                 int s,
                                                 int pool_tile_h,
                                                 int pool_tile_w,
                                                 int tile_n,
                                                 int warp_m,
                                                 int warp_n,
                                                 bool vectorize_conv0_a,
                                                 bool vectorize_maxpool,
                                                 bool early_w1,
                                                 bool direct_conv0,
                                                 bool w_fast,
                                                 int waves_per_eu,
                                                 const char* sched_policy,
                                                 bool mask_maxpool,
                                                 bool specialized_rne,
                                                 bool interior_fastpath,
                                                 bool static_direct_kmap,
                                                 bool packed_c0_handoff,
                                                 bool repack_c0,
                                                 bool fused_c0a1,
                                                 bool butterfly_conv01,
                                                 bool native_int,
                                                 bool batch_loads,
                                                 bool pk_maxpool,
                                                 bool conv1_prefetch_k,
                                                 bool conv1_sched_fuse,
                                                 bool conv1_int8,
                                                 bool persistent,
                                                 int persistent_ctas,
                                                 float m0,
                                                 float m0b,
                                                 float m1,
                                                 float mf)
{
    rocke_gfx1151_deep_fused_conv_pool_spec_t spec;
    rocke_conv_problem_t conv;
    int conv_tile_h;
    int conv_tile_w;

    memset(&spec, 0, sizeof(spec));

    /* conv = ConvProblem(N=n, Hi=h, Wi=w, C=c, K=k0, R=r, S=s,
     *                    sH=1, sW=1, pH=1, pW=1, dH=1, dW=1) */
    conv = rocke_conv_problem_make(n,
                                   h,
                                   w,
                                   c,
                                   k0,
                                   r,
                                   s,
                                   /*sH*/ 1,
                                   /*sW*/ 1,
                                   /*pH*/ 1,
                                   /*pW*/ 1,
                                   /*dH*/ 1,
                                   /*dW*/ 1);

    /* problem = FusedConvPoolProblem(conv=conv, conv1_k=k1)  (pool defaults). */
    spec.problem = rocke_fused_conv_pool_problem_make(conv,
                                                      /*conv1_k*/ k1,
                                                      /*pool_y*/ 2,
                                                      /*pool_x*/ 2,
                                                      /*pool_stride_h*/ 2,
                                                      /*pool_stride_w*/ 2);

    /* conv_tile_h = pool_tile_h * problem.pool_stride_h
     * conv_tile_w = pool_tile_w * problem.pool_stride_w
     * tile_m = conv_tile_h * conv_tile_w */
    conv_tile_h = pool_tile_h * spec.problem.pool_stride_h;
    conv_tile_w = pool_tile_w * spec.problem.pool_stride_w;

    spec.name = ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_NAME;
    spec.tile_m = conv_tile_h * conv_tile_w;
    spec.tile_n = tile_n;
    spec.pool_tile_h = pool_tile_h;
    spec.pool_tile_w = pool_tile_w;
    spec.warp_m = warp_m;
    spec.warp_n = warp_n;

    spec.vectorize_conv0_a = vectorize_conv0_a;
    spec.vectorize_maxpool = vectorize_maxpool;
    spec.early_w1 = early_w1;
    spec.direct_conv0 = direct_conv0;
    spec.w_fast = w_fast;

    spec.waves_per_eu = waves_per_eu;
    spec.sched_policy = (sched_policy != NULL) ? sched_policy : "mem";
    spec.mask_maxpool = mask_maxpool;

    spec.specialized_rne = specialized_rne;
    spec.interior_fastpath = interior_fastpath;
    spec.static_direct_kmap = static_direct_kmap;
    spec.packed_c0_handoff = packed_c0_handoff;
    spec.repack_c0 = repack_c0;
    spec.fused_c0a1 = fused_c0a1;
    spec.butterfly_conv01 = butterfly_conv01;
    spec.native_int = native_int;
    spec.batch_loads = batch_loads;
    spec.pk_maxpool = pk_maxpool;
    spec.conv1_prefetch_k = conv1_prefetch_k;
    spec.conv1_sched_fuse = conv1_sched_fuse;
    spec.conv1_int8 = conv1_int8;
    spec.persistent = persistent;
    spec.persistent_ctas = persistent_ctas;

    spec.m0 = m0;
    spec.m0b = m0b;
    spec.m1 = m1;
    spec.mf = mf;

    return spec;
}

/* ===================================================================== *
 * Derived-quantity accessors (Python @property mirrors, lines 298-337).
 * ===================================================================== */

int rocke_gfx1151_dfcp_warp_tile_m(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    (void)spec;
    return GFX1151_WMMA;
}

int rocke_gfx1151_dfcp_warp_tile_n(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    (void)spec;
    return GFX1151_WMMA;
}

int rocke_gfx1151_dfcp_warp_tile_k(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    (void)spec;
    return GFX1151_WMMA;
}

/* block_size = warp_m * warp_n * _WAVE. */
int rocke_gfx1151_dfcp_block_size(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->warp_m * spec->warp_n * GFX1151_WAVE;
}

/* kpad = ceil(conv.K_gemm / _WMMA) * _WMMA. */
int rocke_gfx1151_dfcp_kpad(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    int kg;
    if(spec == NULL)
    {
        return 0;
    }
    kg = rocke_conv_problem_k_gemm(&spec->problem.conv);
    return ((kg + GFX1151_WMMA - 1) / GFX1151_WMMA) * GFX1151_WMMA;
}

/* conv_tile_h = pool_tile_h * pool_stride_h. */
int rocke_gfx1151_dfcp_conv_tile_h(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->pool_tile_h * spec->problem.pool_stride_h;
}

/* conv_tile_w = pool_tile_w * pool_stride_w. */
int rocke_gfx1151_dfcp_conv_tile_w(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->pool_tile_w * spec->problem.pool_stride_w;
}

/* foot_h = (conv_tile_h-1)*sH + (R-1)*dH + 1. */
int rocke_gfx1151_dfcp_foot_h(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    const rocke_conv_problem_t* c;
    int conv_tile_h;
    if(spec == NULL)
    {
        return 0;
    }
    c = &spec->problem.conv;
    conv_tile_h = rocke_gfx1151_dfcp_conv_tile_h(spec);
    return (conv_tile_h - 1) * c->sH + (c->Y - 1) * c->dH + 1;
}

/* foot_w = (conv_tile_w-1)*sW + (S-1)*dW + 1. */
int rocke_gfx1151_dfcp_foot_w(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec)
{
    const rocke_conv_problem_t* c;
    int conv_tile_w;
    if(spec == NULL)
    {
        return 0;
    }
    c = &spec->problem.conv;
    conv_tile_w = rocke_gfx1151_dfcp_conv_tile_w(spec);
    return (conv_tile_w - 1) * c->sW + (c->X - 1) * c->dW + 1;
}

/* ===================================================================== *
 * kernel_name() (Python lines 339-384).
 *
 * parts = [name, problem.short(), f"t{tile_m}x{tile_n}",
 *          f"pt{pool_tile_h}x{pool_tile_w}", f"w{warp_m}x{warp_n}",
 *          "wmma16x16x16", "directa" if direct_conv0 else "im2col"]
 * + non-default lever tags (in Python declaration order)
 * + "i8i4_realquant", joined via kernel_name_join.
 * ===================================================================== */
rocke_status_t rocke_gfx1151_deep_fused_conv_pool_kernel_name(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec, char* out, size_t out_cap)
{
    /* The Python builds the part list then calls kernel_name_join(*parts) -- the
     * prefix is parts[0] (name) and the rest are tail parts. We build the tail
     * parts array; the largest fixed tag count plus the persistent/lever tags is
     * bounded, so a small static array suffices. */
    char short_buf[128];
    char tile_buf[48];
    char pt_buf[48];
    char w_buf[48];
    char wpe_buf[32];
    char sch_buf[32];
    char persist_buf[40];
    const char* parts[40];
    size_t np = 0;
    rocke_status_t st;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* problem.short() */
    st = rocke_fused_conv_pool_problem_short(&spec->problem, short_buf, sizeof(short_buf), NULL);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* Fixed tile / pool / warp / wmma / a-mode tags. */
    snprintf(tile_buf, sizeof(tile_buf), "t%dx%d", spec->tile_m, spec->tile_n);
    snprintf(pt_buf, sizeof(pt_buf), "pt%dx%d", spec->pool_tile_h, spec->pool_tile_w);
    snprintf(w_buf, sizeof(w_buf), "w%dx%d", spec->warp_m, spec->warp_n);

    parts[np++] = short_buf;
    parts[np++] = tile_buf;
    parts[np++] = pt_buf;
    parts[np++] = w_buf;
    parts[np++] = "wmma16x16x16";
    parts[np++] = spec->direct_conv0 ? "directa" : "im2col";

    /* Lever tags (only when non-default), in Python declaration order. */
    if(spec->waves_per_eu)
    {
        snprintf(wpe_buf, sizeof(wpe_buf), "wpe%d", spec->waves_per_eu);
        parts[np++] = wpe_buf;
    }
    if(spec->sched_policy != NULL && strcmp(spec->sched_policy, "mem") != 0)
    {
        snprintf(sch_buf, sizeof(sch_buf), "sch%s", spec->sched_policy);
        parts[np++] = sch_buf;
    }
    if(spec->mask_maxpool)
    {
        parts[np++] = "maskpool";
    }
    if(spec->specialized_rne)
    {
        parts[np++] = "vecrne";
    }
    if(spec->interior_fastpath)
    {
        parts[np++] = "interior";
    }
    if(spec->static_direct_kmap)
    {
        parts[np++] = "statick";
    }
    if(spec->packed_c0_handoff)
    {
        parts[np++] = "packedc0";
    }
    if(spec->repack_c0)
    {
        parts[np++] = "repackc0";
    }
    if(spec->butterfly_conv01)
    {
        parts[np++] = "butterfly";
    }
    if(spec->native_int)
    {
        parts[np++] = "nativeiu8";
    }
    if(spec->pk_maxpool)
    {
        parts[np++] = "pkpool";
    }
    if(spec->conv1_prefetch_k)
    {
        parts[np++] = "pfk";
    }
    if(spec->conv1_sched_fuse)
    {
        parts[np++] = "schfuse";
    }
    if(spec->conv1_int8)
    {
        parts[np++] = "conv1iu8";
    }
    if(spec->fused_c0a1)
    {
        parts[np++] = "fusedc0a1";
    }
    if(spec->persistent)
    {
        snprintf(persist_buf, sizeof(persist_buf), "persist%d", spec->persistent_ctas);
        parts[np++] = persist_buf;
    }
    parts[np++] = "i8i4_realquant";

    /* kernel_name_join(name, *parts): prefix = spec->name, no flags. */
    return rocke_kernel_name_join(spec->name,
                                  parts,
                                  np,
                                  /*flag_names*/ NULL,
                                  /*flag_on*/ NULL,
                                  /*num_flags*/ 0,
                                  out,
                                  out_cap,
                                  NULL);
}

/* ===================================================================== *
 * is_valid_spec(spec, arch) -> (bool, reason). Python lines 488-629.
 * ===================================================================== */

/* Small helper: write a NUL-terminated reason (truncating safely). */
static void g1151_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", msg);
    }
}

static void g1151_set_reasonf(char* reason, size_t reason_cap, const char* fmt, int a)
{
    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, fmt, a);
    }
}

bool rocke_gfx1151_deep_fused_conv_pool_is_valid_spec(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    const rocke_fused_conv_pool_problem_t* p;
    const rocke_conv_problem_t* c;
    const rocke_arch_target_t* target;
    int conv_tile_h;
    int conv_tile_w;
    int k0;
    int conv1_channels;
    int pool_ho;
    int pool_wo;

    if(spec == NULL)
    {
        g1151_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = GFX1151_ARCH;
    }

    /* arch in ("gfx1151", "gfx11-generic"). */
    if(strcmp(arch, GFX1151_ARCH) != 0 && strcmp(arch, GFX1151_ARCH_GENERIC) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "gfx1151 deep fused conv/pool needs the gfx1151 wave32/WMMA "
                     "ABI (gfx1151 or gfx11-generic); got '%s'",
                     arch);
        }
        return false;
    }

    p = &spec->problem;
    c = &p->conv;

    /* target = ArchTarget.from_gfx(arch); except KeyError -> reason. */
    target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown arch '%s'", arch);
        }
        return false;
    }

    /* WMMA 16x16x16 f16 atom present. */
    if(!rocke_mma_catalog_has_shape(
           &target->mma, "wmma", "f16", "f16", "fp32", GFX1151_WMMA, GFX1151_WMMA, GFX1151_WMMA))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "WMMA 16x16x16 f16 atom absent on %s", arch);
        }
        return false;
    }

    /* wave32. */
    if(target->wave_size != GFX1151_WAVE)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(
                reason, reason_cap, "this kernel is wave32; %s is wave%d", arch, target->wave_size);
        }
        return false;
    }

    /* only 2x2 stride-2 maxpool. */
    if(p->pool_y != 2 || p->pool_x != 2 || p->pool_stride_h != 2 || p->pool_stride_w != 2)
    {
        g1151_set_reason(reason, reason_cap, "only 2x2 stride-2 maxpool is supported");
        return false;
    }

    /* N == 1. */
    if(c->N != 1)
    {
        g1151_set_reasonf(reason, reason_cap, "tiled schedule supports only N=1 (got N=%d)", c->N);
        return false;
    }

    /* positive pool tiles. */
    if(spec->pool_tile_h <= 0 || spec->pool_tile_w <= 0)
    {
        g1151_set_reason(reason, reason_cap, "pool_tile_h and pool_tile_w must be positive");
        return false;
    }

    conv_tile_h = spec->pool_tile_h * p->pool_stride_h;
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;

    /* tile_m == conv tile area. */
    if(spec->tile_m != conv_tile_h * conv_tile_w)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "tile_m=%d must equal conv tile %dx%d=%d",
                     spec->tile_m,
                     conv_tile_h,
                     conv_tile_w,
                     conv_tile_h * conv_tile_w);
        }
        return false;
    }

    pool_ho = rocke_fused_conv_pool_problem_pool_ho(p);
    pool_wo = rocke_fused_conv_pool_problem_pool_wo(p);

    /* pool dims divisible by pool tiles. */
    if((pool_ho % spec->pool_tile_h) || (pool_wo % spec->pool_tile_w))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "pool dims (%d,%d) must be divisible by pool tile (%d,%d)",
                     pool_ho,
                     pool_wo,
                     spec->pool_tile_h,
                     spec->pool_tile_w);
        }
        return false;
    }

    k0 = c->K;
    conv1_channels = rocke_fused_conv_pool_problem_conv1_channels(p);

    /* K0 <= tile_n. */
    if(k0 > spec->tile_n)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "one CTA owns all conv0 channels: K0=%d > tile_n=%d",
                     k0,
                     spec->tile_n);
        }
        return false;
    }

    /* K1 <= tile_n. */
    if(conv1_channels > spec->tile_n)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "one CTA owns all conv1 channels: K1=%d > tile_n=%d",
                     conv1_channels,
                     spec->tile_n);
        }
        return false;
    }

    /* tile_m % (warp_m*16). */
    if(spec->tile_m % (spec->warp_m * GFX1151_WMMA))
    {
        g1151_set_reason(reason, reason_cap, "tile_m must divide warp_m * 16");
        return false;
    }

    /* tile_n % (warp_n*16). */
    if(spec->tile_n % (spec->warp_n * GFX1151_WMMA))
    {
        g1151_set_reason(reason, reason_cap, "tile_n must divide warp_n * 16");
        return false;
    }

    /* K0 % 16. */
    if(c->K % GFX1151_WMMA)
    {
        g1151_set_reasonf(
            reason, reason_cap, "conv0 channels K0=%d must be a multiple of 16 (conv1 K)", c->K);
        return false;
    }

    /* (im2col only) tile_m*kpad % block_size. */
    if(!spec->direct_conv0)
    {
        int kpad = rocke_gfx1151_dfcp_kpad(spec);
        int bs = rocke_gfx1151_dfcp_block_size(spec);
        if(bs != 0 && (spec->tile_m * kpad) % bs)
        {
            g1151_set_reason(
                reason, reason_cap, "tile_m*kpad must divide block_size (A0 staging is untailed)");
            return false;
        }
    }

    /* known sched_policy. */
    if(spec->sched_policy == NULL
       || (strcmp(spec->sched_policy, "mem") != 0 && strcmp(spec->sched_policy, "compv3") != 0
           && strcmp(spec->sched_policy, "compv4") != 0
           && strcmp(spec->sched_policy, "intrawave") != 0))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "unknown sched_policy '%s'",
                     spec->sched_policy ? spec->sched_policy : "(null)");
        }
        return false;
    }

    /* butterfly_conv01 rejected (analyzed non-lever; not implemented). */
    if(spec->butterfly_conv01)
    {
        g1151_set_reason(reason,
                         reason_cap,
                         "butterfly_conv01 is an analyzed non-lever on gfx1151 WMMA "
                         "(cross-lane C->A transpose costs more LDS-unit ops than the "
                         "c0_smem round-trip it would replace); not implemented");
        return false;
    }

    /* native_int lever preconditions. */
    if(spec->native_int)
    {
        if(rocke_mma_catalog_by_op_id(&target->mma, GFX1151_OP_ID_IU8) == NULL)
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "%s atom absent on %s", GFX1151_OP_ID_IU8, arch);
            }
            return false;
        }
        if(rocke_mma_catalog_by_op_id(&target->mma, GFX1151_OP_ID_IU4) == NULL)
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "%s atom absent on %s", GFX1151_OP_ID_IU4, arch);
            }
            return false;
        }
        if(spec->direct_conv0 && (c->C % GFX1151_K_PER_I32))
        {
            g1151_set_reason(reason,
                             reason_cap,
                             "native_int direct conv0 requires C to be divisible by 4 "
                             "so each iu8 packed fragment slot stays inside one "
                             "footprint pixel");
            return false;
        }
        if(spec->butterfly_conv01)
        {
            g1151_set_reason(
                reason, reason_cap, "native_int is incompatible with butterfly_conv01");
            return false;
        }
    }

    /* interior_fastpath -> native_int + direct_conv0. */
    if(spec->interior_fastpath && !(spec->native_int && spec->direct_conv0))
    {
        g1151_set_reason(
            reason, reason_cap, "interior_fastpath is only implemented for native direct conv0");
        return false;
    }

    /* static_direct_kmap. */
    if(spec->static_direct_kmap)
    {
        if(!(spec->native_int && spec->direct_conv0))
        {
            g1151_set_reason(reason,
                             reason_cap,
                             "static_direct_kmap is only implemented for native direct conv0");
            return false;
        }
        if(c->C != 8 || c->Y != 3 || c->X != 3)
        {
            g1151_set_reason(reason, reason_cap, "static_direct_kmap assumes C=8 and R=S=3");
            return false;
        }
    }

    /* packed_c0_handoff. */
    if(spec->packed_c0_handoff)
    {
        if(!spec->native_int)
        {
            g1151_set_reason(
                reason, reason_cap, "packed_c0_handoff is only implemented for native_int");
            return false;
        }
        if(c->K % 2)
        {
            g1151_set_reason(reason, reason_cap, "packed_c0_handoff requires even conv0 channels");
            return false;
        }
    }

    /* repack_c0. */
    if(spec->repack_c0)
    {
        if(!spec->native_int)
        {
            g1151_set_reason(reason, reason_cap, "repack_c0 is only implemented for native_int");
            return false;
        }
        if(spec->packed_c0_handoff)
        {
            g1151_set_reason(
                reason, reason_cap, "repack_c0 and packed_c0_handoff are mutually exclusive");
            return false;
        }
        if(c->K % 2)
        {
            g1151_set_reason(reason, reason_cap, "repack_c0 requires even conv0 channels");
            return false;
        }
    }

    /* fused_c0a1. */
    if(spec->fused_c0a1)
    {
        if(!(spec->native_int && spec->direct_conv0))
        {
            g1151_set_reason(
                reason, reason_cap, "fused_c0a1 is only implemented for native direct conv0");
            return false;
        }
        if(spec->packed_c0_handoff || spec->repack_c0)
        {
            g1151_set_reason(reason,
                             reason_cap,
                             "fused_c0a1 is mutually exclusive with packed_c0_handoff/repack_c0");
            return false;
        }
        if(c->K % GFX1151_WMMA)
        {
            g1151_set_reason(reason, reason_cap, "fused_c0a1 requires conv0 K0 divisible by 16");
            return false;
        }
    }

    /* conv1_int8. */
    if(spec->conv1_int8)
    {
        if(!(spec->native_int && spec->fused_c0a1))
        {
            g1151_set_reason(reason,
                             reason_cap,
                             "conv1_int8 only implemented for the native_int fused_c0a1 "
                             "register handoff path");
            return false;
        }
    }

    /* persistent. */
    if(spec->persistent)
    {
        if(!(spec->native_int && spec->direct_conv0 && spec->fused_c0a1))
        {
            g1151_set_reason(reason,
                             reason_cap,
                             "persistent requires native_int + direct_conv0 + fused_c0a1 "
                             "(the only path with register weight/handoff and no per-tile "
                             "c0_smem, so W0/W1 can be hoisted above the grid-stride loop)");
            return false;
        }
        if(spec->persistent_ctas <= 0)
        {
            g1151_set_reason(reason, reason_cap, "persistent_ctas must be positive");
            return false;
        }
    }

    g1151_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 * deep_fused_conv_pool_grid(spec) -> (gx, gy, gz). Python lines 632-645.
 * ===================================================================== */
rocke_status_t
    rocke_gfx1151_deep_fused_conv_pool_grid(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec,
                                            int out[3])
{
    const rocke_fused_conv_pool_problem_t* p;
    int h_tiles;
    int w_tiles;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    p = &spec->problem;
    h_tiles = rocke_fused_conv_pool_problem_pool_ho(p) / spec->pool_tile_h;
    w_tiles = rocke_fused_conv_pool_problem_pool_wo(p) / spec->pool_tile_w;

    if(spec->persistent)
    {
        /* (persistent_ctas, 1, 1). */
        out[0] = spec->persistent_ctas;
        out[1] = 1;
        out[2] = 1;
        return ROCKE_OK;
    }
    if(spec->w_fast)
    {
        /* (1, w_tiles, h_tiles). */
        out[0] = 1;
        out[1] = w_tiles;
        out[2] = h_tiles;
        return ROCKE_OK;
    }
    /* (1, h_tiles, w_tiles). */
    out[0] = 1;
    out[1] = h_tiles;
    out[2] = w_tiles;
    return ROCKE_OK;
}

/* ===================================================================== *
 * rocke_build_gfx1151_deep_fused_conv_pool -- the public build driver.
 *
 * Mirrors Python build_deep_fused_conv_pool (lines 2684-3085): the validity
 * gate, op/op0/op1 resolution, param declaration, WarpGrid + policy + LDS
 * allocation are all performed by the peer ctx-init
 * (rocke_gfx1151_dfcp_build_ctx_init); this driver then dispatches to the
 * persistent grid-stride body or the single-tile body (peer sub-phases) per the
 * native_int + direct_conv0 + fused_c0a1 + persistent lever combination.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_gfx1151_deep_fused_conv_pool(
    rocke_ir_builder_t* b, const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec, const char* arch)
{
    rocke_gfx1151_dfcp_build_ctx_t ctx;
    rocke_status_t st;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = GFX1151_ARCH; /* arch default "gfx1151" */
    }

    /* Populate the shared ctx: runs is_valid_spec, resolves op/op0/op1, declares
     * X/W0/Y/W1, builds + binds the WarpGrid, stamps waves_per_eu + policy,
     * resolves the LDS dtypes + kpad, and allocates the LDS tiles. Errors are
     * routed through b's sticky error (mirroring the Python ValueError paths). */
    st = rocke_gfx1151_dfcp_build_ctx_init(&ctx, b, spec, arch);
    if(st != ROCKE_OK)
    {
        /* ctx_init already recorded the message on b for the ValueError paths;
         * set a generic one if it somehow did not. */
        if(rocke_ir_builder_status(b) == ROCKE_OK)
        {
            rocke_i_set_err(b, st, "gfx1151 deep fused conv/pool: ctx init failed");
        }
        return NULL;
    }

    /* Dispatch to the matching body sub-phase. The persistent grid-stride variant
     * is gated (by is_valid_spec) to native_int + direct_conv0 + fused_c0a1; the
     * single-tile pipeline covers every other resolved configuration. */
    if(spec->persistent)
    {
        rocke_gfx1151_dfcp_emit_persistent_body(&ctx);
    }
    else
    {
        rocke_gfx1151_dfcp_emit_single_tile_body(&ctx);
    }

    /* Surface any sticky error the body emission raised. */
    if(rocke_ir_builder_status(b) != ROCKE_OK)
    {
        return NULL;
    }

    /* The KernelDef is the builder's kernel (== b->kernel). */
    return rocke_ir_builder_kernel(b);
}

/* Convenience: init `b` with the gfx1151 kernel name, then build. */
rocke_kernel_def_t* rocke_build_gfx1151_deep_fused_conv_pool_new(
    rocke_ir_builder_t* b, const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec, const char* arch)
{
    /* Catch validity/build raises at this boundary so an invalid spec is
     * reported as a clean NULL+error (matching the sibling instances) rather
     * than escaping to std::terminate. */
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_gfx1151_deep_fused_conv_pool_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_gfx1151_deep_fused_conv_pool(b, spec, arch);
    });
}

/* ===================================================================== *
 * rocke_gfx1151_deep_fused_conv_pool_lower_to_llvm -- build + lower to .ll.
 * Owns and frees its own IRBuilder (mirrors the sibling instance ports).
 * `arch` NULL => "gfx1151".
 * ===================================================================== */
rocke_status_t rocke_gfx1151_deep_fused_conv_pool_lower_to_llvm(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = GFX1151_ARCH;
    }

    kernel = rocke_build_gfx1151_deep_fused_conv_pool_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_gfx1151_deep_fused_conv_pool failed";
            }
            snprintf(err, err_cap, "%s", m);
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
