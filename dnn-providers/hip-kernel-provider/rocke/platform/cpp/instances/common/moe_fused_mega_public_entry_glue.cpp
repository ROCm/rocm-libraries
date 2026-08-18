// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_public_entry_glue.c -- PUBLIC entry / glue for the
 * C99 chunked port of build_moe_fused_mega_gemm
 * (rocke/instances/common/moe_fused_mega.py, lines 434-740).
 *
 * SCOPE (this TU only):
 *   - rocke_build_moe_fused_mega_gemm        (the driver: gates -> attrs ->
 *                                           ctx_init -> guarded body -> kernel)
 *   - rocke_build_moe_fused_mega_gemm_new    (init builder from spec.kernel_name
 *                                           then build)
 *   - rocke_moe_fused_mega_lower_to_llvm     (own+free IRBuilder, build, lower)
 *
 * This is the "bucket that calls phases": it reproduces the Python
 * build_moe_fused_mega_gemm control flow byte-for-byte. The whole prologue
 * (params, geometry, thread decode, per-expert B byte bases, LDS, views,
 * plans/operands, acc inits, down setup) is owned by rocke_moe_mega_build_ctx_init
 * (a peer TU declared in the internal header); the STAGE 1..5 body is owned by
 * rocke_moe_mega_emit_body (peer TU). This TU owns only:
 *   1. deriving u_gu / u_down scratch + running the two validity gates,
 *   2. setting the builder attrs,
 *   3. driving ctx_init then emitting the scf_if(expert_idx >= 0) guard around
 *      rocke_moe_mega_emit_body,
 *   4. returning b->kernel.
 *
 * Byte-identical builder-call sequence (Python build_moe_fused_mega_gemm):
 *   u_gu  = spec.gate_up_universal_spec(); is_valid_gemm_spec(u_gu)   (448-451)
 *   u_down= spec.down_universal_spec();    is_valid_gemm_spec(u_down) (452-455)
 *   b.kernel.attrs["max_workgroup_size"] = spec.block_size           (458)
 *   if waves_per_eu is not None: attrs["waves_per_eu"] = ...          (459-460)
 *   <whole prologue>                  -> rocke_moe_mega_build_ctx_init  (462-634)
 *   with b.scf_if(b.cmp_ge(expert_idx, c0)): _emit_body()            (737-738)
 *   return b.kernel                                                   (740)
 */
#include <stdlib.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/instance_gemm_universal.h"
#include "rocke/instance_moe_fused_mega.h"
#include "rocke/instance_moe_fused_mega_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err (sticky-error setter)            */
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 *  PRIMARY build entry -- build_moe_fused_mega_gemm(spec, arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_moe_fused_mega_gemm(rocke_ir_builder_t* b,
                                                    const rocke_moe_fused_mega_kernel_spec_t* spec,
                                                    const char* arch)
{
    /* u_gu / u_down scratch: caller-owned, lives for the whole build (the ctx
     * holds pointers the body forwards). Locals here outlive emit_body. */
    rocke_gemm_universal_spec_t u_gu;
    rocke_gemm_universal_spec_t u_down;
    rocke_moe_mega_build_ctx_t ctx;
    char reason[ROCKE_ERR_MSG_CAP];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950"; /* Python default: arch="gfx950" */
    }

    /* ---- u_gu = spec.gate_up_universal_spec(); is_valid_gemm_spec gate ---- *
     * Python (448-451):
     *   u_gu = spec.gate_up_universal_spec()
     *   ok, why = is_valid_gemm_spec(u_gu, arch=arch)
     *   if not ok: raise ValueError(f"invalid fused-mega gate+up GEMM spec: {why}")
     */
    rocke_moe_fused_mega_gate_up_universal_spec(spec, &u_gu);
    reason[0] = '\0';
    if(!rocke_gemm_universal_is_valid_spec(&u_gu, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid fused-mega gate+up GEMM spec: %s", reason);
        return NULL;
    }

    /* ---- u_down = spec.down_universal_spec(); is_valid_gemm_spec gate ---- *
     * Python (452-455):
     *   u_down = spec.down_universal_spec()
     *   ok, why = is_valid_gemm_spec(u_down, arch=arch)
     *   if not ok: raise ValueError(f"invalid fused-mega down GEMM spec: {why}")
     */
    rocke_moe_fused_mega_down_universal_spec(spec, &u_down);
    reason[0] = '\0';
    if(!rocke_gemm_universal_is_valid_spec(&u_down, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid fused-mega down GEMM spec: %s", reason);
        return NULL;
    }

    /* ---- builder attrs ---- *
     * Python (458-460):
     *   b.kernel.attrs["max_workgroup_size"] = spec.block_size
     *   if spec.trait.waves_per_eu is not None:
     *       b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu
     */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    if(spec->trait.waves_per_eu_set)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->trait.waves_per_eu);
    }

    /* ---- whole prologue into the ctx ---- (Python 462-634)
     * params -> geometry -> thread decode -> per-expert B byte bases -> LDS
     * allocs -> views -> plans/operands -> acc inits -> down setup. On a
     * builder error the sticky status is set; bail with NULL. */
    if(rocke_moe_mega_build_ctx_init(&ctx, b, spec, arch, &u_gu, &u_down) != ROCKE_OK)
    {
        return NULL;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* ---- guarded body ---- *
     * Python (737-738):
     *   with b.scf_if(b.cmp_ge(expert_idx, c0)):
     *       _emit_body()
     * Empty tail block (BlockExpertIds == -1) skips all work. */
    {
        rocke_if_t guard = rocke_b_scf_if(b, rocke_b_cmp_ge(b, ctx.expert_idx, ctx.c0));
        rocke_b_region_enter(b, guard.then_region);
        rocke_moe_mega_emit_body(&ctx);
        rocke_b_region_leave(b);
    }

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    /* Python (740): return b.kernel */
    return b->kernel;
}

/* ===================================================================== *
 *  Convenience: init `b` with spec.kernel_name(), then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_moe_fused_mega_gemm_new(
    rocke_ir_builder_t* b, const rocke_moe_fused_mega_kernel_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[1024];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()) */
        if(rocke_moe_fused_mega_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_fused_mega_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  LOWER-TO-LLVM GLUE
 *
 *  Convenience: build -> lower to LLVM .ll text. Owns and frees its own
 *  IRBuilder. On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 *  caller frees with free(); on failure it is left NULL and (if err != NULL,
 *  cap err_cap) a diagnostic is written.
 * ===================================================================== */

/* Copy `msg` into the (err, err_cap) buffer, NUL-terminated and truncated to
 * fit. No-op if err is NULL or err_cap is 0. */
static void rocke_moe_mega_set_err(char* err, size_t err_cap, const char* msg)
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

rocke_status_t rocke_moe_fused_mega_lower_to_llvm(const rocke_moe_fused_mega_kernel_spec_t* spec,
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
        rocke_moe_mega_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* build -> the convenience entry owns the builder init via spec.kernel_name(). */
    kernel = rocke_build_moe_fused_mega_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_moe_mega_set_err(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_moe_fused_mega_gemm failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
