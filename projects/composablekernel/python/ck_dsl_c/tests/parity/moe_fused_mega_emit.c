/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/moe_fused_mega_emit.c -- C-side emitter for the
 * moe_fused_mega parity harness. Selects one of N sampled spec configs by
 * argv[1] (the config index), builds ckc_moe_fused_mega_kernel_spec_t
 * identically to the Python emitter moe_fused_mega_emit.py, builds the kernel
 * via ckc_build_moe_fused_mega_gemm_new + lowers via ckc_lower_kernel_to_llvm
 * (arch gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs
 * can be byte-compared.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_moe_fused_mega.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_moe_fused_mega_kernel_spec_t *spec) {
    *spec = ckc_moe_fused_mega_kernel_spec_default();

    switch (idx) {
    case 0: /* moe_mega_baseline */
        spec->name        = "moe_mega_baseline";
        spec->tile_m      = 32;
        spec->tile_n_inter= 256;
        spec->tile_k_gu   = 32;
        spec->tile_n_down = 256;
        spec->tile_k_down = 64;
        spec->dtype       = "fp16";
        break;
    case 1: /* moe_mega_tuned_m16 */
        spec->name        = "moe_mega_tuned_m16";
        spec->tile_m      = 16;
        spec->tile_n_inter= 256;
        spec->tile_k_gu   = 32;
        spec->tile_n_down = 256;
        spec->tile_k_down = 64;
        spec->dtype       = "fp16";
        break;
    case 2: /* moe_mega_large_k */
        spec->name        = "moe_mega_large_k";
        spec->tile_m      = 32;
        spec->tile_n_inter= 256;
        spec->tile_k_gu   = 64;
        spec->tile_n_down = 256;
        spec->tile_k_down = 128;
        spec->dtype       = "fp16";
        break;
    case 3: /* moe_mega_wide_n */
        spec->name        = "moe_mega_wide_n";
        spec->tile_m      = 32;
        spec->tile_n_inter= 512;
        spec->tile_k_gu   = 32;
        spec->tile_n_down = 512;
        spec->tile_k_down = 64;
        spec->dtype       = "fp16";
        break;
    case 4: /* moe_mega_fp8 */
        spec->name        = "moe_mega_fp8";
        spec->tile_m      = 32;
        spec->tile_n_inter= 256;
        spec->tile_k_gu   = 32;
        spec->tile_n_down = 256;
        spec->tile_k_down = 64;
        spec->dtype       = "fp8e4m3";
        break;
    case 5: /* moe_mega_bf16 */
        spec->name        = "moe_mega_bf16";
        spec->tile_m      = 32;
        spec->tile_n_inter= 256;
        spec->tile_k_gu   = 32;
        spec->tile_n_down = 256;
        spec->tile_k_down = 64;
        spec->dtype       = "bf16";
        break;
    default:
        return -1;
    }
    ckc_moe_fused_mega_kernel_spec_finalize(spec);
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

    ckc_moe_fused_mega_kernel_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_ir_builder_t b;
    ckc_kernel_def_t *kernel =
        ckc_build_moe_fused_mega_gemm_new(&b, &spec, "gfx950");
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
