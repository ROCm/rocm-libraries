// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_public_entry_glue.c -- PUBLIC entry / glue for the
 * C99 chunked port of build_moe_fused_mega_gemm
 * (ck_dsl/instances/common/moe_fused_mega.py, lines 434-740).
 *
 * SCOPE (this TU only):
 *   - ckc_build_moe_fused_mega_gemm        (the driver: gates -> attrs ->
 *                                           ctx_init -> guarded body -> kernel)
 *   - ckc_build_moe_fused_mega_gemm_new    (init builder from spec.kernel_name
 *                                           then build)
 *   - ckc_moe_fused_mega_lower_to_llvm     (own+free IRBuilder, build, lower)
 *
 * This is the "bucket that calls phases": it reproduces the Python
 * build_moe_fused_mega_gemm control flow byte-for-byte. The whole prologue
 * (params, geometry, thread decode, per-expert B byte bases, LDS, views,
 * plans/operands, acc inits, down setup) is owned by ckc_moe_mega_build_ctx_init
 * (a peer TU declared in the internal header); the STAGE 1..5 body is owned by
 * ckc_moe_mega_emit_body (peer TU). This TU owns only:
 *   1. deriving u_gu / u_down scratch + running the two validity gates,
 *   2. setting the builder attrs,
 *   3. driving ctx_init then emitting the scf_if(expert_idx >= 0) guard around
 *      ckc_moe_mega_emit_body,
 *   4. returning b->kernel.
 *
 * Byte-identical builder-call sequence (Python build_moe_fused_mega_gemm):
 *   u_gu  = spec.gate_up_universal_spec(); is_valid_gemm_spec(u_gu)   (448-451)
 *   u_down= spec.down_universal_spec();    is_valid_gemm_spec(u_down) (452-455)
 *   b.kernel.attrs["max_workgroup_size"] = spec.block_size           (458)
 *   if waves_per_eu is not None: attrs["waves_per_eu"] = ...          (459-460)
 *   <whole prologue>                  -> ckc_moe_mega_build_ctx_init  (462-634)
 *   with b.scf_if(b.cmp_ge(expert_idx, c0)): _emit_body()            (737-738)
 *   return b.kernel                                                   (740)
 */
#include <stdlib.h>
#include <string.h>

#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "ckc/instance_gemm_universal.h"
#include "ckc/instance_moe_fused_mega.h"
#include "ckc/instance_moe_fused_mega_internal.h"
#include "ckc/ir.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err (sticky-error setter)            */
#include "ckc/lower_llvm.h"

/* ===================================================================== *
 *  PRIMARY build entry -- build_moe_fused_mega_gemm(spec, arch)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_moe_fused_mega_gemm(ckc_ir_builder_t* b,
                                                const ckc_moe_fused_mega_kernel_spec_t* spec,
                                                const char* arch)
{
    /* u_gu / u_down scratch: caller-owned, lives for the whole build (the ctx
     * holds pointers the body forwards). Locals here outlive emit_body. */
    ckc_gemm_universal_spec_t u_gu;
    ckc_gemm_universal_spec_t u_down;
    ckc_moe_mega_build_ctx_t ctx;
    char reason[CKC_ERR_MSG_CAP];

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
    ckc_moe_fused_mega_gate_up_universal_spec(spec, &u_gu);
    reason[0] = '\0';
    if(!ckc_gemm_universal_is_valid_spec(&u_gu, arch, reason, sizeof(reason)))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "invalid fused-mega gate+up GEMM spec: %s", reason);
        return NULL;
    }

    /* ---- u_down = spec.down_universal_spec(); is_valid_gemm_spec gate ---- *
     * Python (452-455):
     *   u_down = spec.down_universal_spec()
     *   ok, why = is_valid_gemm_spec(u_down, arch=arch)
     *   if not ok: raise ValueError(f"invalid fused-mega down GEMM spec: {why}")
     */
    ckc_moe_fused_mega_down_universal_spec(spec, &u_down);
    reason[0] = '\0';
    if(!ckc_gemm_universal_is_valid_spec(&u_down, arch, reason, sizeof(reason)))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "invalid fused-mega down GEMM spec: %s", reason);
        return NULL;
    }

    /* ---- builder attrs ---- *
     * Python (458-460):
     *   b.kernel.attrs["max_workgroup_size"] = spec.block_size
     *   if spec.trait.waves_per_eu is not None:
     *       b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu
     */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    if(spec->trait.waves_per_eu_set)
    {
        ckc_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->trait.waves_per_eu);
    }

    /* ---- whole prologue into the ctx ---- (Python 462-634)
     * params -> geometry -> thread decode -> per-expert B byte bases -> LDS
     * allocs -> views -> plans/operands -> acc inits -> down setup. On a
     * builder error the sticky status is set; bail with NULL. */
    if(ckc_moe_mega_build_ctx_init(&ctx, b, spec, arch, &u_gu, &u_down) != CKC_OK)
    {
        return NULL;
    }
    if(!ckc_ir_builder_ok(b))
    {
        return NULL;
    }

    /* ---- guarded body ---- *
     * Python (737-738):
     *   with b.scf_if(b.cmp_ge(expert_idx, c0)):
     *       _emit_body()
     * Empty tail block (BlockExpertIds == -1) skips all work. */
    {
        ckc_if_t guard = ckc_b_scf_if(b, ckc_b_cmp_ge(b, ctx.expert_idx, ctx.c0));
        ckc_b_region_enter(b, guard.then_region);
        ckc_moe_mega_emit_body(&ctx);
        ckc_b_region_leave(b);
    }

    if(!ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    /* Python (740): return b.kernel */
    return b->kernel;
}

/* ===================================================================== *
 *  Convenience: init `b` with spec.kernel_name(), then build.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_moe_fused_mega_gemm_new(ckc_ir_builder_t* b,
                                                    const ckc_moe_fused_mega_kernel_spec_t* spec,
                                                    const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[1024];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()) */
        if(ckc_moe_fused_mega_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_fused_mega_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  LOWER-TO-LLVM GLUE
 *
 *  Convenience: build -> lower to LLVM .ll text. Owns and frees its own
 *  IRBuilder. On CKC_OK *out_ll receives a malloc'd NUL-terminated string the
 *  caller frees with free(); on failure it is left NULL and (if err != NULL,
 *  cap err_cap) a diagnostic is written.
 * ===================================================================== */

/* Copy `msg` into the (err, err_cap) buffer, NUL-terminated and truncated to
 * fit. No-op if err is NULL or err_cap is 0. */
static void ckc_moe_mega_set_err(char* err, size_t err_cap, const char* msg)
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

ckc_status_t ckc_moe_fused_mega_lower_to_llvm(const ckc_moe_fused_mega_kernel_spec_t* spec,
                                              const char* arch,
                                              ckc_llvm_flavor_t flavor,
                                              char** out_ll,
                                              char* err,
                                              size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        ckc_moe_mega_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* build -> the convenience entry owns the builder init via spec.kernel_name(). */
    kernel = ckc_build_moe_fused_mega_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_moe_mega_set_err(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_moe_fused_mega_gemm failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
