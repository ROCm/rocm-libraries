// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dw_fwd_wmma.cpp -- C++ port of the Toeplitz WMMA depthwise
 * forward (library/kernels/common/conv_direct_dw_fwd_wmma.py):
 *
 *   Python                                    C++ (this file)
 *   DwFwdWmmaSpec                          -> rocke_direct_dw_fwd_wmma_spec_t
 *   dw_fwd_wmma_reason / is_valid_...      -> rocke_direct_dw_fwd_wmma_is_valid_spec()
 *   kernel_name()                          -> rocke_direct_dw_fwd_wmma_kernel_name()
 *   dw_fwd_wmma_win_spec(spec)             -> rocke_direct_dw_fwd_wmma_win_spec()
 *   build_dw_fwd_wmma(spec, arch)          -> rocke_build_direct_dw_fwd_wmma_new()
 *
 * The kernel is the wave32 fwd form of the windowed depthwise Toeplitz
 * builder (conv_direct_grouped_build_depthwise_dgrad_win.cpp) under this
 * family's name; this file owns the spec, the name and the limits.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_ceil_div, ... */
#include "rocke/instance_conv_direct_dw_fwd_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */
rocke_direct_dw_fwd_wmma_spec_t rocke_direct_dw_fwd_wmma_spec_default(void)
{
    rocke_direct_dw_fwd_wmma_spec_t s;
    memset(&s, 0, sizeof s);
    s.problem = rocke_direct_conv_problem_default();
    s.name = "rocke_direct_dw_fwd_wmma";
    s.block_waves = 4;
    s.block_h = 8;
    s.w_fold = 1;
    s.prefetch_rows = 2;
    return s;
}

int rocke_direct_dw_fwd_wmma_threads_per_block(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return spec->block_waves * ROCKE_DW_FWD_WMMA_WAVE;
}

int rocke_direct_dw_fwd_wmma_block_ch(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return spec->block_waves * ROCKE_DW_DGRAD_MFMA_CH;
}

int rocke_direct_dw_fwd_wmma_tile_w(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return rocke_dw_mfma_tile_w(spec->w_fold);
}

int rocke_direct_dw_fwd_wmma_in_passes(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return rocke_dw_mfma_in_passes(rocke_direct_dw_fwd_wmma_tile_w(spec),
                                   spec->problem.KW,
                                   spec->w_fold,
                                   rocke_direct_dw_fwd_wmma_block_ch(spec),
                                   rocke_direct_dw_fwd_wmma_threads_per_block(spec));
}

int rocke_direct_dw_fwd_wmma_frag_vgprs(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int kwp = rocke_dw_mfma_passes(p->KW);
    return 8 * (p->KH * kwp + p->KH + kwp)
           + 4 * spec->prefetch_rows * rocke_direct_dw_fwd_wmma_in_passes(spec);
}

long rocke_direct_dw_fwd_wmma_lds_bytes(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return rocke_dw_mfma_lds_bytes(rocke_direct_dw_fwd_wmma_in_passes(spec),
                                   rocke_direct_dw_fwd_wmma_threads_per_block(spec),
                                   rocke_direct_dw_fwd_wmma_block_ch(spec),
                                   spec->w_fold,
                                   rocke_direct_dw_fwd_wmma_tile_w(spec));
}

int rocke_direct_dw_fwd_wmma_unrolled_wmmas(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return spec->block_h * spec->problem.KH * rocke_dw_mfma_passes(spec->problem.KW);
}

void rocke_direct_dw_fwd_wmma_grid(const rocke_direct_dw_fwd_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    grid[0] = (int)rocke_ceil_div(rocke_direct_conv_problem_Wo(p),
                                  rocke_direct_dw_fwd_wmma_tile_w(spec));
    grid[1] = (int)rocke_ceil_div(p->groups, rocke_direct_dw_fwd_wmma_block_ch(spec));
    grid[2] = (int)rocke_ceil_div(p->N, spec->w_fold)
              * (int)rocke_ceil_div(rocke_direct_conv_problem_Ho(p), spec->block_h);
}

rocke_status_t rocke_direct_dw_fwd_wmma_kernel_name(const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap)
{
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    char rsp[64], wv[32], bh[32], f[32], pf[32];
    snprintf(rsp, sizeof rsp, "r%ds%dp%d", p->KH, p->KW, p->PAD);
    snprintf(wv, sizeof wv, "wv%d", spec->block_waves);
    snprintf(bh, sizeof bh, "bh%d", spec->block_h);
    snprintf(f, sizeof f, "f%d", spec->w_fold);
    snprintf(pf, sizeof pf, "pf%d", spec->prefetch_rows);
    const char* parts[] = {rsp, wv, bh, f, pf};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {rocke_direct_conv_problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "rocke_direct_dw_fwd_wmma",
                                  parts,
                                  sizeof(parts) / sizeof(parts[0]),
                                  flag_names,
                                  flag_on,
                                  1,
                                  out,
                                  out_cap,
                                  NULL);
}

/* dw_fwd_wmma_reason(spec): the checks in the Python order and wording. */
static bool dfw_check(const rocke_direct_dw_fwd_wmma_spec_t* spec, char* reason, size_t cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(rocke_direct_problem_dims_reason(p, reason, cap))
        return false;
    const char* dtype = rocke_direct_conv_problem_dtype(p);
    if(strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0)
        ROCKE_SPEC_FAIL(reason, cap, "unsupported dtype '%s'; expected fp16 or bf16", dtype);
    if(p->cpg != 1 || p->kpg != 1)
        ROCKE_SPEC_FAIL(reason, cap, "requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
    if(p->stride != 1)
        ROCKE_SPEC_FAIL(reason, cap, "requires stride=1 (got %d)", p->stride);
    if(rocke_direct_dilation_reason(p, reason, cap)
       || rocke_direct_conv_one_padding_reason(p, "", reason, cap))
        return false;
    {
        const int kmax = p->KH > p->KW ? p->KH : p->KW;
        const long long Ho = rocke_direct_conv_problem_Ho(p), Wo = rocke_direct_conv_problem_Wo(p);
        if(!(0 <= p->PAD && p->PAD < kmax) || Ho <= 0 || Wo <= 0)
            ROCKE_SPEC_FAIL(
                reason, cap, "degenerate geometry (PAD=%d, Ho=%lld, Wo=%lld)", p->PAD, Ho, Wo);
    }
    if(spec->block_h < 1)
        ROCKE_SPEC_FAIL(reason, cap, "block_h must be >= 1 (got %d)", spec->block_h);
    if(spec->block_waves < 1 || spec->block_waves > ROCKE_DW_FWD_WMMA_MAX_WAVES)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "block_waves must be 1..%d (got %d)",
                        ROCKE_DW_FWD_WMMA_MAX_WAVES,
                        spec->block_waves);
    if(!rocke_dw_mfma_w_fold_ok(spec->w_fold))
        ROCKE_SPEC_FAIL(reason, cap, "w_fold must be one of (1, 2, 4) (got %d)", spec->w_fold);
    if(spec->prefetch_rows < 1 || spec->prefetch_rows > ROCKE_DW_DGRAD_MFMA_MAX_PREFETCH_ROWS)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "prefetch_rows must be 1..%d (got %d)",
                        ROCKE_DW_DGRAD_MFMA_MAX_PREFETCH_ROWS,
                        spec->prefetch_rows);
    if(p->groups % ROCKE_DW_DGRAD_MFMA_CH != 0)
        ROCKE_SPEC_FAIL(
            reason, cap, "groups %% %d != 0 (got %d)", ROCKE_DW_DGRAD_MFMA_CH, p->groups);
    {
        const int frag = rocke_direct_dw_fwd_wmma_frag_vgprs(spec);
        if(spec->block_waves > ROCKE_DW_FWD_WMMA_WIDE_WAVES
           && frag > ROCKE_DW_FWD_WMMA_WIDE_FRAG_VGPRS)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "%d waves need fragments <= %d VGPRs (got %d)",
                            spec->block_waves,
                            ROCKE_DW_FWD_WMMA_WIDE_FRAG_VGPRS,
                            frag);
        if(frag > ROCKE_DW_FWD_WMMA_MAX_FRAG_VGPRS)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "fragments need %d VGPRs (> %d); lower prefetch_rows or the filter",
                            frag,
                            ROCKE_DW_FWD_WMMA_MAX_FRAG_VGPRS);
    }
    {
        const int u = rocke_direct_dw_fwd_wmma_unrolled_wmmas(spec);
        if(u > ROCKE_DW_FWD_WMMA_MAX_UNROLL)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "unrolled body too large (%d WMMAs > %d); lower block_h",
                            u,
                            ROCKE_DW_FWD_WMMA_MAX_UNROLL);
    }
    {
        const long lds = rocke_direct_dw_fwd_wmma_lds_bytes(spec);
        if(lds > ROCKE_DW_DGRAD_MFMA_LDS_BUDGET)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "stages %ld LDS bytes per workgroup (budget %d); lower block_waves",
                            lds,
                            ROCKE_DW_DGRAD_MFMA_LDS_BUDGET);
    }
    {
        const long long x = (long long)p->N * p->H * p->W * p->groups * 2;
        const long long y = (long long)p->N * rocke_direct_conv_problem_Ho(p)
                            * rocke_direct_conv_problem_Wo(p) * p->groups * 2;
        const long long mx = x > y ? x : y;
        if(mx > ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "X/Y of %lld bytes exceed the %lld-byte sentinel addressing range",
                            mx,
                            (long long)ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES);
    }
    return true;
}

bool rocke_direct_dw_fwd_wmma_is_valid_spec(const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap)
{
    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx1250";
    if(strcmp(arch, "gfx1250") != 0)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "needs one of ('gfx1250',) (got %s)", arch);
        return false;
    }
    if(!dfw_check(spec, reason, reason_cap))
        return false;
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_dw_fwd_wmma: the windowed Toeplitz builder's wave32 fwd form
 * ===================================================================== */
rocke_direct_depthwise_dgrad_win_spec_t
    rocke_direct_dw_fwd_wmma_win_spec(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    rocke_direct_depthwise_dgrad_win_spec_t w = rocke_direct_depthwise_dgrad_win_spec_default();
    w.problem = *p;
    w.problem.H = (int)rocke_direct_conv_problem_Ho(p);
    w.problem.W = (int)rocke_direct_conv_problem_Wo(p);
    w.problem.PAD = p->KH - 1 - p->PAD;
    w.name = spec->name;
    w.block_waves = spec->block_waves;
    w.block_h = spec->block_h;
    w.mfma = true;
    w.w_fold = spec->w_fold;
    w.prefetch_rows = spec->prefetch_rows;
    w.wave_size = ROCKE_DW_FWD_WMMA_WAVE;
    w.fwd = true;
    return w;
}

rocke_kernel_def_t* rocke_build_direct_dw_fwd_wmma(rocke_ir_builder_t* b,
                                                   const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                   const char* arch)
{
    const rocke_direct_depthwise_dgrad_win_spec_t w = rocke_direct_dw_fwd_wmma_win_spec(spec);
    /* NULL arch => gfx1250, as Python's build_dw_fwd_wmma default. */
    return rocke_build_direct_depthwise_dgrad_win(b, &w, arch != NULL ? arch : "gfx1250");
}

rocke_kernel_def_t* rocke_build_direct_dw_fwd_wmma_new(rocke_ir_builder_t* b,
                                                       const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                       const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        char why[512];
        if(b == NULL || spec == NULL)
            return NULL;
        if(arch == NULL)
            arch = "gfx1250";
        /* Validated before the Python builds its IRBuilder. */
        if(!rocke_direct_dw_fwd_wmma_is_valid_spec(spec, arch, why, sizeof why))
        {
            rocke_ir_builder_init(b, "direct_dw_fwd_wmma_invalid");
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid DwFwdWmmaSpec for %s: %s", arch, why);
            return NULL;
        }
        if(rocke_direct_dw_fwd_wmma_kernel_name(spec, name, sizeof name) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_dw_fwd_wmma(b, spec, arch);
    });
}

rocke_status_t rocke_direct_dw_fwd_wmma_lower_to_llvm(const rocke_direct_dw_fwd_wmma_spec_t* spec,
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
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    kernel = rocke_build_direct_dw_fwd_wmma_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_spec_set_reason(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_dw_fwd_wmma failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
