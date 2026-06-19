/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/moe_fused_mega_fp8_emit.c -- C-side emitter for the FP8 fused-MoE
 * MEGA-kernel parity harness. Selects one of N sampled spec configs by argv[1]
 * (the config index), builds ckc_fused_mega_kernel_spec_fp8_t identically to the
 * Python emitter, builds via ckc_build_moe_fused_mega_gemm_fp8_new and lowers via
 * ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO), printing the .ll to stdout
 * so the two outputs can be byte-compared.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_moe_fused_mega_fp8.h"

/* Fill `spec` for config index `idx`; sets *persistent. Returns 0, or -1 if
 * unknown. Mirrors the Python FusedMegaKernelSpecFp8(...) constructions. */
static int make_spec(int idx, ckc_fused_mega_kernel_spec_fp8_t *spec,
                     bool *persistent) {
    *spec = ckc_fused_mega_kernel_spec_fp8_default();
    *persistent = false;

    switch (idx) {
    case 0: /* baseline: gate_up_k=32, down_k=32, use_dtla=False, no cadence */
        spec->name              = "moe_fused_mega_fp8_baseline";
        spec->tile_m            = 16;
        spec->tile_n_inter      = 256;
        spec->gate_up_k         = 32;
        spec->down_k            = 32;
        spec->use_dtla          = false;
        spec->has_sched_cadence = false; /* Python None */
        break;
    case 1: /* l7 hero: gate_up_k=128, down_k=128, use_dtla=False, no cadence */
        spec->name              = "moe_fused_mega_fp8_l7_hero";
        spec->tile_m            = 16;
        spec->tile_n_inter      = 256;
        spec->gate_up_k         = 128;
        spec->down_k            = 128;
        spec->use_dtla          = false;
        spec->has_sched_cadence = false; /* Python None */
        break;
    case 2: /* l8 dtla: use_dtla=True, sched_cadence="none" */
        spec->name              = "moe_fused_mega_fp8_l8_dtla";
        spec->tile_m            = 16;
        spec->tile_n_inter      = 256;
        spec->gate_up_k         = 128;
        spec->down_k            = 128;
        spec->use_dtla          = true;
        spec->has_sched_cadence = true;
        spec->sched_cadence     = "none";
        break;
    case 3: /* l9 iglp: use_dtla=True, sched_cadence="iglp1" */
        spec->name              = "moe_fused_mega_fp8_l9_iglp";
        spec->tile_m            = 16;
        spec->tile_n_inter      = 256;
        spec->gate_up_k         = 128;
        spec->down_k            = 128;
        spec->use_dtla          = true;
        spec->has_sched_cadence = true;
        spec->sched_cadence     = "iglp1";
        break;
    case 4: /* prod: l9 config, persistent=False */
        spec->name              = "moe_fused_mega_fp8_prod";
        spec->tile_m            = 16;
        spec->tile_n_inter      = 256;
        spec->gate_up_k         = 128;
        spec->down_k            = 128;
        spec->use_dtla          = true;
        spec->has_sched_cadence = true;
        spec->sched_cadence     = "iglp1";
        *persistent             = false;
        break;
    case 5: /* persistent: l9 config, persistent=True */
        spec->name              = "moe_fused_mega_fp8_persistent";
        spec->tile_m            = 16;
        spec->tile_n_inter      = 256;
        spec->gate_up_k         = 128;
        spec->down_k            = 128;
        spec->use_dtla          = true;
        spec->has_sched_cadence = true;
        spec->sched_cadence     = "iglp1";
        *persistent             = true;
        break;
    default:
        return -1;
    }
    /* __post_init__: resolve block_size from warp_m*warp_n*wave_size. */
    ckc_fused_mega_kernel_spec_fp8_post_init(spec);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char *mode = (argc > 2) ? argv[2] : "ll";

    if (strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 &&
        strcmp(mode, "verify") != 0) {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    ckc_fused_mega_kernel_spec_fp8_t spec;
    bool persistent = false;
    if (make_spec(idx, &spec, &persistent) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_ir_builder_t b;
    /* levers NULL => Python import-time defaults (golden-safe). */
    ckc_kernel_def_t *kernel = ckc_build_moe_fused_mega_gemm_fp8_new(
        &b, &spec, "gfx950", persistent, NULL);
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    if (strcmp(mode, "ll") == 0) {
        char *llvm_text = NULL;
        ckc_status_t st = ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO,
                                                   "gfx950", &llvm_text);
        if (st != CKC_OK || !llvm_text) {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    } else if (strcmp(mode, "ir") == 0) {
        char *text = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &text);
        if (st != CKC_OK || !text) {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    } else { /* verify */
        ckc_diag_t *d = NULL;
        size_t n = 0;
        ckc_verify(kernel, &d, &n);
        for (size_t i = 0; i < n; i++) {
            char *s = ckc_diag_to_string(&d[i]);
            if (s) { puts(s); free(s); }
        }
        ckc_diags_free(d, n);
    }

    ckc_ir_builder_free(&b);
    return 0;
}
