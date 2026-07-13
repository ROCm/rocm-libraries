// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_public_entry_glue.c -- public build entry +
 * lower glue for the C99 chunked port of build_direct_conv_16c and
 * build_direct_conv_4c (rocke/instances/common/conv_direct_grouped.py).
 *
 * SCOPE (this TU only):
 *   - rocke_build_direct_conv_16c / _new
 *   - rocke_build_direct_conv_4c  / _new
 *   - rocke_direct_conv_16c_lower_to_llvm
 *   - rocke_direct_conv_4c_lower_to_llvm
 *
 * These are the convenience entries: they construct + populate the shared
 * context struct (rocke_dconv_16c_ctx_t / rocke_dconv_4c_ctx_t) and drive the phase
 * functions in the exact order the Python builder runs them. Every phase is a
 * peer (implemented in a sibling TU) declared in
 * rocke/instance_conv_direct_grouped_internal.h; this TU calls them but does not
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

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 *  16c BUILD ENTRY
 *
 *  build_direct_conv_16c(spec, arch) -> KernelDef
 *
 *  The Python prologue's validate() + is_valid_spec_16c gate + geometry
 *  derivation all live in rocke_dconv16c_prologue (per the internal-header
 *  contract); this driver seeds the ctx inputs the prologue reads, then runs
 *  the phases in Python order and returns the kernel the H-loop phase built.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_conv_16c(rocke_ir_builder_t* b,
                                                const rocke_direct_conv_16c_spec_t* spec,
                                                const char* arch)
{
    rocke_dconv_16c_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
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
    if(!rocke_dconv16c_prologue(&ctx))
    {
        return NULL;
    }

    /* ---- weight loads (constant across H-loop) ---- (lines 357-415) */
    rocke_dconv16c_load_weights(&ctx);

    /* ---- per-thread chunk decode table ---- (lines 444-473) */
    rocke_dconv16c_build_chunk_meta(&ctx);

    /* ---- A / D descriptors ---- (lines 475-519, 637-641) */
    rocke_dconv16c_build_descriptors(&ctx);

    /* ---- prologue: prefetch row 0 into A_smem + sync ---- (lines 609-616) */
    rocke_dconv16c_prologue_prefetch(&ctx);

    /* ---- the H-row streaming loop ----  (lines 618-740)
     * Returns b.kernel on success, NULL on builder error. */
    return rocke_dconv16c_stream_h_loop(&ctx);
}

/* Convenience: init `b` with spec.kernel_name(), then build_direct_conv_16c. */
rocke_kernel_def_t* rocke_build_direct_conv_16c_new(rocke_ir_builder_t* b,
                                                    const rocke_direct_conv_16c_spec_t* spec,
                                                    const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()) */
        if(rocke_direct_conv_16c_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_direct_conv_16c(b, spec, arch);
    });
}

/* ===================================================================== *
 *  4c BUILD ENTRY
 *
 *  build_direct_conv_4c(spec, arch) -> KernelDef
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_conv_4c(rocke_ir_builder_t* b,
                                               const rocke_direct_conv_4c_spec_t* spec,
                                               const char* arch)
{
    rocke_dconv_4c_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
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
    if(!rocke_dconv4c_prologue(&ctx))
    {
        return NULL;
    }

    /* ---- weight loads ---- (lines 878-901) */
    rocke_dconv4c_load_weights(&ctx);

    /* ---- A / D descriptors + acc seed + loop-invariant locals ----
     * (lines 903-965) */
    rocke_dconv4c_build_descriptors(&ctx);

    /* ---- the H-row loop ----  (lines 967-1033)
     * Returns b.kernel on success, NULL on builder error. */
    return rocke_dconv4c_stream_h_loop(&ctx);
}

/* Convenience: init `b` with spec.kernel_name(), then build_direct_conv_4c. */
rocke_kernel_def_t* rocke_build_direct_conv_4c_new(rocke_ir_builder_t* b,
                                                   const rocke_direct_conv_4c_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()) */
        if(rocke_direct_conv_4c_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_direct_conv_4c(b, spec, arch);
    });
}

/* ===================================================================== *
 *  LOWER-TO-LLVM GLUE
 *
 *  Convenience: build -> lower to LLVM .ll text. Each owns and frees its own
 *  IRBuilder. On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 *  caller frees with free(); on failure it is left NULL and (if err != NULL,
 *  cap err_cap) a diagnostic is written.
 * ===================================================================== */

/* Copy `msg` into the (err, err_cap) buffer, NUL-terminated and truncated to
 * fit. No-op if err is NULL or err_cap is 0. */
static void rocke_dconv_set_err(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    if(msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, msg, n);
    err[n] = '\0';
}

rocke_status_t rocke_direct_conv_16c_lower_to_llvm(const rocke_direct_conv_16c_spec_t* spec,
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
        rocke_dconv_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* build -> the convenience entry owns the builder init via spec.kernel_name(). */
    kernel = rocke_build_direct_conv_16c_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_dconv_set_err(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_direct_conv_16c failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

rocke_status_t rocke_direct_conv_4c_lower_to_llvm(const rocke_direct_conv_4c_spec_t* spec,
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
        rocke_dconv_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_direct_conv_4c_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_dconv_set_err(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_direct_conv_4c failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
