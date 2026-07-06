/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fused_moe_e2e_emit.c -- C-side emitter for the fused_moe_e2e
 * (end-to-end fused-MoE forward orchestrator) parity harness.
 *
 * Selects one of the sampled FusedMoeForwardSpec configs by argv[1] (the config
 * index), materialises the spec via rocke_fmoe_forward_spec_default() + the per-
 * config shape fields, then lowers each lowerable pipeline stage to AMDGPU LLVM
 * IR text via rocke_fused_moe_forward_lower_to_llvm (arch gfx950, flavor AUTO) and
 * prints them concatenated to stdout, each prefixed with a stage banner, so the
 * output can be byte-compared with the Python emitter fused_moe_e2e_emit.py.
 *
 * The orchestrator emits NO single monolithic kernel; rocke_fused_moe_forward_
 * lower_to_llvm runs __init__'s arch resolve + tile-swap policy internally (via
 * the build ctx) and delegates to the spec-selected sub-kernel builder. The
 * three lowerable stages are ROUTER (topk_softmax) and GATE_UP_GEMM / DOWN_GEMM
 * (both the batched-GEMM builder shape). The remaining stages (sort hist/scan/
 * scatter, gather, silu_mul, topk_reduce) are TODO(port) NOTIMPL and are
 * therefore excluded here, matching the Python emitter.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_batched_gemm.h"
#include "rocke/instance_fused_moe_e2e.h"
#include "rocke/instance_fused_moe_e2e_internal.h"
#include "rocke/instance_topk_softmax.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Populate the FusedMoeForwardSpec for config `idx`. Returns 0, or -1 on an
 * unknown index. Every non-shape field stays at the dataclass default
 * (rocke_fmoe_forward_spec_default), so the tile-swap policy and the static gate
 * see exactly the Python defaults; only the enumerated shape + dtype differ. */
static int make_spec(int idx, rocke_fmoe_forward_spec_t* spec)
{
    *spec = rocke_fmoe_forward_spec_default();
    spec->arch = "gfx950";
    switch(idx)
    {
    case 0: /* tokens=1   E8  K2 H4096 I7168 f16  */
        spec->tokens = 1;
        spec->experts = 8;
        spec->topk = 2;
        spec->hidden = 4096;
        spec->intermediate = 7168;
        spec->dtype = "f16";
        break;
    case 1: /* tokens=8   E8  K2 H4096 I7168 f16  */
        spec->tokens = 8;
        spec->experts = 8;
        spec->topk = 2;
        spec->hidden = 4096;
        spec->intermediate = 7168;
        spec->dtype = "f16";
        break;
    case 2: /* tokens=32  E8  K2 H4096 I7168 f16  */
        spec->tokens = 32;
        spec->experts = 8;
        spec->topk = 2;
        spec->hidden = 4096;
        spec->intermediate = 7168;
        spec->dtype = "f16";
        break;
    case 3: /* tokens=128 E8  K2 H4096 I7168 f16  */
        spec->tokens = 128;
        spec->experts = 8;
        spec->topk = 2;
        spec->hidden = 4096;
        spec->intermediate = 7168;
        spec->dtype = "f16";
        break;
    case 4: /* tokens=1   E8  K2 H4096 I7168 bf16 */
        spec->tokens = 1;
        spec->experts = 8;
        spec->topk = 2;
        spec->hidden = 4096;
        spec->intermediate = 7168;
        spec->dtype = "bf16";
        break;
    case 5: /* tokens=128 E32 K5 H8192 I8192 f16  */
        spec->tokens = 128;
        spec->experts = 32;
        spec->topk = 5;
        spec->hidden = 8192;
        spec->intermediate = 8192;
        spec->dtype = "f16";
        break;
    default:
        return -1;
    }
    return 0;
}

/* Lowerable stages, in the Python emitter's order. GATE_UP_GEMM and DOWN_GEMM
 * both resolve to the batched-GEMM builder shape inside
 * rocke_fused_moe_forward_lower_to_llvm. */
static const struct
{
    const char* banner;
    rocke_fmoe_stage_t stage;
} STAGES[] = {
    {"ROUTER", ROCKE_FMOE_STAGE_ROUTER},
    {"GATE_UP_GEMM", ROCKE_FMOE_STAGE_GATE_UP_GEMM},
    {"DOWN_GEMM", ROCKE_FMOE_STAGE_DOWN_GEMM},
};

/* Build a kernel for one stage using the tile-policy-adjusted spec.
 * The caller must rocke_ir_builder_free(&b) when done. Returns NULL on failure. */
static rocke_kernel_def_t* build_stage_kernel(rocke_ir_builder_t* b,
                                              const rocke_fmoe_forward_spec_t* spec,
                                              rocke_fmoe_stage_t stage,
                                              const char* banner)
{
    /* Apply tile-swap policy via the build ctx (same as lower_to_llvm). */
    rocke_fmoe_build_ctx_t* ctx = (rocke_fmoe_build_ctx_t*)calloc(1, sizeof(*ctx));
    if(!ctx)
        return NULL;
    rocke_status_t st = rocke_fmoe_build_ctx_init(ctx, spec, "gfx950");
    if(st != ROCKE_OK)
    {
        rocke_fmoe_build_ctx_destroy(ctx);
        free(ctx);
        return NULL;
    }
    rocke_fmoe_forward_spec_t adj = ctx->spec;
    const char* arch = ctx->arch;
    rocke_kernel_def_t* kernel = NULL;

    if(stage == ROCKE_FMOE_STAGE_ROUTER)
    {
        rocke_topk_softmax_spec_t s = rocke_fmoe_forward_spec_to_topk_softmax_spec(&adj);
        kernel = rocke_build_topk_softmax_new(b, &s, arch);
    }
    else if(stage == ROCKE_FMOE_STAGE_GATE_UP_GEMM || stage == ROCKE_FMOE_STAGE_DOWN_GEMM)
    {
        char name_buf[256];
        rocke_batched_gemm_spec_t s;
        st = rocke_fmoe_forward_spec_to_batched_gemm_spec(&adj, name_buf, sizeof(name_buf), &s);
        if(st == ROCKE_OK)
        {
            kernel = rocke_build_batched_gemm_new(b, &s, arch);
        }
    }
    else
    {
        fprintf(stderr, "build_stage_kernel: stage %s not wired for ir/verify\n", banner);
    }
    rocke_fmoe_build_ctx_destroy(ctx);
    free(ctx);
    return kernel;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_fmoe_forward_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const size_t nstages = sizeof(STAGES) / sizeof(STAGES[0]);

    if(strcmp(mode, "ll") == 0)
    {
        for(size_t i = 0; i < nstages; i++)
        {
            char* llvm_text = NULL;
            char err[ROCKE_ERR_MSG_CAP];
            err[0] = 0;
            rocke_status_t st = rocke_fused_moe_forward_lower_to_llvm(&spec,
                                                                      "gfx950",
                                                                      STAGES[i].stage,
                                                                      ROCKE_LLVM_FLAVOR_AUTO,
                                                                      &llvm_text,
                                                                      err,
                                                                      sizeof err);
            if(st != ROCKE_OK || !llvm_text)
            {
                fprintf(stderr,
                        "lower failed (config %d stage %s): status=%d err=%s\n",
                        idx,
                        STAGES[i].banner,
                        (int)st,
                        err);
                free(llvm_text);
                return 1;
            }
            printf("; === fused_moe_e2e stage: %s ===\n", STAGES[i].banner);
            fputs(llvm_text, stdout);
            size_t n = strlen(llvm_text);
            if(n == 0 || llvm_text[n - 1] != '\n')
            {
                fputc('\n', stdout);
            }
            free(llvm_text);
        }
        return 0;
    }

    /* ir / verify: build each stage kernel and serialize/verify */
    for(size_t i = 0; i < nstages; i++)
    {
        rocke_ir_builder_t b;
        rocke_kernel_def_t* kernel
            = build_stage_kernel(&b, &spec, STAGES[i].stage, STAGES[i].banner);
        if(!kernel)
        {
            fprintf(stderr, "build failed (config %d stage %s)\n", idx, STAGES[i].banner);
            rocke_ir_builder_free(&b);
            return 1;
        }
        printf("; === fused_moe_e2e stage: %s ===\n", STAGES[i].banner);
        if(strcmp(mode, "ir") == 0)
        {
            char* t = NULL;
            rocke_status_t st = rocke_ir_serialize(kernel, &t);
            if(st != ROCKE_OK || !t)
            {
                fprintf(stderr, "serialize failed: status=%d\n", (int)st);
                rocke_ir_builder_free(&b);
                return 1;
            }
            fputs(t, stdout);
            size_t n = strlen(t);
            if(n == 0 || t[n - 1] != '\n')
                fputc('\n', stdout);
            free(t);
        }
        else if(strcmp(mode, "verify") == 0)
        {
            rocke_diag_t* d = NULL;
            size_t n = 0;
            rocke_verify(kernel, &d, &n);
            for(size_t j = 0; j < n; j++)
            {
                char* s = rocke_diag_to_string(&d[j]);
                if(s)
                {
                    puts(s);
                    free(s);
                }
            }
            rocke_diags_free(d, n);
        }
        else
        {
            fprintf(stderr, "unknown mode %s\n", mode);
            rocke_ir_builder_free(&b);
            return 2;
        }
        rocke_ir_builder_free(&b);
    }
    return 0;
}
