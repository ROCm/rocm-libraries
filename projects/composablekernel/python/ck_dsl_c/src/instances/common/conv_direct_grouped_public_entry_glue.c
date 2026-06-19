/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_conv_direct_grouped_public_entry_glue.c -- public build entry +
 * lower glue for the C99 chunked port of build_direct_conv_16c and
 * build_direct_conv_4c (ck_dsl/instances/common/conv_direct_grouped.py).
 *
 * SCOPE (this TU only):
 *   - ckc_build_direct_conv_16c / _new
 *   - ckc_build_direct_conv_4c  / _new
 *   - ckc_direct_conv_16c_lower_to_llvm
 *   - ckc_direct_conv_4c_lower_to_llvm
 *
 * These are the convenience entries: they construct + populate the shared
 * context struct (ckc_dconv_16c_ctx_t / ckc_dconv_4c_ctx_t) and drive the phase
 * functions in the exact order the Python builder runs them. Every phase is a
 * peer (implemented in a sibling TU) declared in
 * ckc/instance_conv_direct_grouped_internal.h; this TU calls them but does not
 * implement them.
 *
 * Byte-identical builder-call sequence:
 *   16c (Python build_direct_conv_16c, lines 256-740):
 *     prologue            -> validate() + is_valid_spec_16c gate + geometry
 *                            + params + SSA consts + thread/grid decode
 *                            + LDS alloc + buffer rsrcs (lines 256-355)
 *     load_weights        -> b_desc + k_out_val + weights[*]      (357-415)
 *     build_chunk_meta    -> chunk_desc + chunk_meta[*]           (444-473)
 *     build_descriptors   -> a_desc + d_desc                      (475-519,637-641)
 *     prologue_prefetch   -> store_to_lds(issue_dram_load(c0)) + sync (609-616)
 *     stream_h_loop       -> the unrolled H-row loop, returns b.kernel (618-740)
 *   4c (Python build_direct_conv_4c, lines 833-1033):
 *     prologue            -> validate() + is_valid_spec_4c gate + params
 *                            + SSA consts + thread/grid decode + rsrcs (833-876)
 *     load_weights        -> b_desc + k_out_val + weights[*]      (878-901)
 *     build_descriptors   -> a_desc + d_desc + acc seed + invariants (903-965)
 *     stream_h_loop       -> the unrolled H-row loop, returns b.kernel (967-1033)
 *
 * The phase functions own the IR emission; this TU owns only ctx lifetime,
 * field seeding for the inputs the prologue reads, and the call ordering.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_conv_direct_grouped.h"
#include "ckc/instance_conv_direct_grouped_internal.h"
#include "ckc/lower_llvm.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ===================================================================== *
 *  16c BUILD ENTRY
 *
 *  build_direct_conv_16c(spec, arch) -> KernelDef
 *
 *  The Python prologue's validate() + is_valid_spec_16c gate + geometry
 *  derivation all live in ckc_dconv16c_prologue (per the internal-header
 *  contract); this driver seeds the ctx inputs the prologue reads, then runs
 *  the phases in Python order and returns the kernel the H-loop phase built.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_direct_conv_16c(ckc_ir_builder_t* b,
                                            const ckc_direct_conv_16c_spec_t* spec,
                                            const char* arch)
{
    ckc_dconv_16c_ctx_t ctx;

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950"; /* Python default: arch="gfx950" */
    }

    /* Zero the whole context so every unfilled handle/table slot starts NULL
     * (mirrors the Python locals being undefined until first assignment). The
     * prologue then fills the input-derived fields in Python source order. */
    memset(&ctx, 0, sizeof(ctx));
    ctx.b = b;
    ctx.spec = spec;
    ctx.arch = arch;
    ctx.p = spec->problem; /* p = spec.problem (by value) */

    /* spec.validate(); is_valid_spec_16c gate; geometry; params; consts;
     * thread/grid decode; LDS alloc; buffer rsrcs.  (lines 256-355)
     * Returns false with the builder error set on a rejected spec /
     * geometry violation (e.g. NUM_VEC4 == 0). */
    if (!ckc_dconv16c_prologue(&ctx))
    {
        return NULL;
    }

    /* ---- weight loads (constant across H-loop) ---- (lines 357-415) */
    ckc_dconv16c_load_weights(&ctx);

    /* ---- per-thread chunk decode table ---- (lines 444-473) */
    ckc_dconv16c_build_chunk_meta(&ctx);

    /* ---- A / D descriptors ---- (lines 475-519, 637-641) */
    ckc_dconv16c_build_descriptors(&ctx);

    /* ---- prologue: prefetch row 0 into A_smem + sync ---- (lines 609-616) */
    ckc_dconv16c_prologue_prefetch(&ctx);

    /* ---- the H-row streaming loop ----  (lines 618-740)
     * Returns b.kernel on success, NULL on builder error. */
    return ckc_dconv16c_stream_h_loop(&ctx);
}

/* Convenience: init `b` with spec.kernel_name(), then build_direct_conv_16c. */
ckc_kernel_def_t* ckc_build_direct_conv_16c_new(ckc_ir_builder_t* b,
                                                const ckc_direct_conv_16c_spec_t* spec,
                                                const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if (b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()) */
        if (ckc_direct_conv_16c_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if (ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_direct_conv_16c(b, spec, arch);
    });
}

/* ===================================================================== *
 *  4c BUILD ENTRY
 *
 *  build_direct_conv_4c(spec, arch) -> KernelDef
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_direct_conv_4c(ckc_ir_builder_t* b,
                                           const ckc_direct_conv_4c_spec_t* spec,
                                           const char* arch)
{
    ckc_dconv_4c_ctx_t ctx;

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950"; /* Python default: arch="gfx950" */
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b = b;
    ctx.spec = spec;
    ctx.arch = arch;
    ctx.p = spec->problem; /* p = spec.problem (by value) */

    /* spec.validate(); is_valid_spec_4c gate; params; consts; thread/grid
     * decode; buffer rsrcs.  (lines 833-876) Returns false on a rejected
     * spec / geometry violation. */
    if (!ckc_dconv4c_prologue(&ctx))
    {
        return NULL;
    }

    /* ---- weight loads ---- (lines 878-901) */
    ckc_dconv4c_load_weights(&ctx);

    /* ---- A / D descriptors + acc seed + loop-invariant locals ----
     * (lines 903-965) */
    ckc_dconv4c_build_descriptors(&ctx);

    /* ---- the H-row loop ----  (lines 967-1033)
     * Returns b.kernel on success, NULL on builder error. */
    return ckc_dconv4c_stream_h_loop(&ctx);
}

/* Convenience: init `b` with spec.kernel_name(), then build_direct_conv_4c. */
ckc_kernel_def_t* ckc_build_direct_conv_4c_new(ckc_ir_builder_t* b,
                                               const ckc_direct_conv_4c_spec_t* spec,
                                               const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if (b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()) */
        if (ckc_direct_conv_4c_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if (ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_direct_conv_4c(b, spec, arch);
    });
}

/* ===================================================================== *
 *  LOWER-TO-LLVM GLUE
 *
 *  Convenience: build -> lower to LLVM .ll text. Each owns and frees its own
 *  IRBuilder. On CKC_OK *out_ll receives a malloc'd NUL-terminated string the
 *  caller frees with free(); on failure it is left NULL and (if err != NULL,
 *  cap err_cap) a diagnostic is written.
 * ===================================================================== */

/* Copy `msg` into the (err, err_cap) buffer, NUL-terminated and truncated to
 * fit. No-op if err is NULL or err_cap is 0. */
static void ckc_dconv_set_err(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if (err == NULL || err_cap == 0)
    {
        return;
    }
    if (msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if (n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, msg, n);
    err[n] = '\0';
}

ckc_status_t ckc_direct_conv_16c_lower_to_llvm(const ckc_direct_conv_16c_spec_t* spec,
                                               const char* arch,
                                               ckc_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        ckc_dconv_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* build -> the convenience entry owns the builder init via spec.kernel_name(). */
    kernel = ckc_build_direct_conv_16c_new(&b, spec, arch);
    if (kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_dconv_set_err(err, err_cap,
                          (m != NULL && m[0] != '\0') ? m : "build_direct_conv_16c failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

ckc_status_t ckc_direct_conv_4c_lower_to_llvm(const ckc_direct_conv_4c_spec_t* spec,
                                              const char* arch,
                                              ckc_llvm_flavor_t flavor,
                                              char** out_ll,
                                              char* err,
                                              size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        ckc_dconv_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_direct_conv_4c_new(&b, spec, arch);
    if (kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_dconv_set_err(err, err_cap,
                          (m != NULL && m[0] != '\0') ? m : "build_direct_conv_4c failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
