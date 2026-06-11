/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fused_moe_e2e_emit.c -- C-side emitter for the fused_moe_e2e
 * (end-to-end fused-MoE forward orchestrator) parity harness.
 *
 * Selects one of the sampled FusedMoeForwardSpec configs by argv[1] (the config
 * index), materialises the spec via ckc_fmoe_forward_spec_default() + the per-
 * config shape fields, then lowers each lowerable pipeline stage to AMDGPU LLVM
 * IR text via ckc_fused_moe_forward_lower_to_llvm (arch gfx950, flavor AUTO) and
 * prints them concatenated to stdout, each prefixed with a stage banner, so the
 * output can be byte-compared with the Python emitter fused_moe_e2e_emit.py.
 *
 * The orchestrator emits NO single monolithic kernel; ckc_fused_moe_forward_
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

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_fused_moe_e2e.h"

/* Populate the FusedMoeForwardSpec for config `idx`. Returns 0, or -1 on an
 * unknown index. Every non-shape field stays at the dataclass default
 * (ckc_fmoe_forward_spec_default), so the tile-swap policy and the static gate
 * see exactly the Python defaults; only the enumerated shape + dtype differ. */
static int make_spec(int idx, ckc_fmoe_forward_spec_t *spec) {
    *spec = ckc_fmoe_forward_spec_default();
    spec->arch = "gfx950";
    switch (idx) {
    case 0: /* tokens=1   E8  K2 H4096 I7168 f16  */
        spec->tokens = 1;   spec->experts = 8;  spec->topk = 2;
        spec->hidden = 4096; spec->intermediate = 7168; spec->dtype = "f16";
        break;
    case 1: /* tokens=8   E8  K2 H4096 I7168 f16  */
        spec->tokens = 8;   spec->experts = 8;  spec->topk = 2;
        spec->hidden = 4096; spec->intermediate = 7168; spec->dtype = "f16";
        break;
    case 2: /* tokens=32  E8  K2 H4096 I7168 f16  */
        spec->tokens = 32;  spec->experts = 8;  spec->topk = 2;
        spec->hidden = 4096; spec->intermediate = 7168; spec->dtype = "f16";
        break;
    case 3: /* tokens=128 E8  K2 H4096 I7168 f16  */
        spec->tokens = 128; spec->experts = 8;  spec->topk = 2;
        spec->hidden = 4096; spec->intermediate = 7168; spec->dtype = "f16";
        break;
    case 4: /* tokens=1   E8  K2 H4096 I7168 bf16 */
        spec->tokens = 1;   spec->experts = 8;  spec->topk = 2;
        spec->hidden = 4096; spec->intermediate = 7168; spec->dtype = "bf16";
        break;
    case 5: /* tokens=128 E32 K5 H8192 I8192 f16  */
        spec->tokens = 128; spec->experts = 32; spec->topk = 5;
        spec->hidden = 8192; spec->intermediate = 8192; spec->dtype = "f16";
        break;
    default:
        return -1;
    }
    return 0;
}

/* Lowerable stages, in the Python emitter's order. GATE_UP_GEMM and DOWN_GEMM
 * both resolve to the batched-GEMM builder shape inside
 * ckc_fused_moe_forward_lower_to_llvm. */
static const struct { const char *banner; ckc_fmoe_stage_t stage; } STAGES[] = {
    {"ROUTER",       CKC_FMOE_STAGE_ROUTER},
    {"GATE_UP_GEMM", CKC_FMOE_STAGE_GATE_UP_GEMM},
    {"DOWN_GEMM",    CKC_FMOE_STAGE_DOWN_GEMM},
};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_fmoe_forward_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const size_t nstages = sizeof(STAGES) / sizeof(STAGES[0]);
    for (size_t i = 0; i < nstages; i++) {
        char *llvm_text = NULL;
        char err[CKC_ERR_MSG_CAP];
        err[0] = 0;
        ckc_status_t st = ckc_fused_moe_forward_lower_to_llvm(
            &spec, "gfx950", STAGES[i].stage, CKC_LLVM_FLAVOR_AUTO,
            &llvm_text, err, sizeof err);
        if (st != CKC_OK || !llvm_text) {
            fprintf(stderr, "lower failed (config %d stage %s): status=%d err=%s\n",
                    idx, STAGES[i].banner, (int)st, err);
            free(llvm_text);
            return 1;
        }
        printf("; === fused_moe_e2e stage: %s ===\n", STAGES[i].banner);
        fputs(llvm_text, stdout);
        size_t n = strlen(llvm_text);
        if (n == 0 || llvm_text[n - 1] != '\n') {
            fputc('\n', stdout);
        }
        free(llvm_text);
    }
    return 0;
}
