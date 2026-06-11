/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/block_scale_gemm_emit.c -- C-side emitter for the
 * block_scale_gemm parity harness. Selects one of N sampled spec configs by
 * argv[1] (the config index), builds ckc_block_scale_gemm_spec_t identically to
 * the Python emitter block_scale_gemm_emit.py, builds the kernel via
 * ckc_build_block_scale_gemm + lowers via ckc_lower_kernel_to_llvm (arch gfx950,
 * flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_block_scale_gemm.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_block_scale_gemm_spec_t *spec) {
    *spec = ckc_block_scale_gemm_spec_default();
    spec->quant_mode    = "abquant";
    spec->block_tile_m  = 16;
    spec->block_tile_n  = 16;

    switch (idx) {
    case 0:
        spec->M = 32; spec->N = 32; spec->K = 64;
        spec->mantissa_dtype = "fp8e4m3";
        spec->group_m = 1; spec->group_n = 1; spec->group_k = 64;
        break;
    case 1:
        spec->M = 64; spec->N = 64; spec->K = 128;
        spec->mantissa_dtype = "fp8e4m3";
        spec->group_m = 1; spec->group_n = 1; spec->group_k = 128;
        break;
    case 2:
        spec->M = 16; spec->N = 16; spec->K = 128;
        spec->mantissa_dtype = "bf8e5m2";
        spec->group_m = 1; spec->group_n = 1; spec->group_k = 64;
        break;
    case 3:
        spec->M = 128; spec->N = 128; spec->K = 256;
        spec->mantissa_dtype = "fp8e4m3";
        spec->group_m = 1; spec->group_n = 1; spec->group_k = 256;
        break;
    case 4:
        spec->M = 48; spec->N = 48; spec->K = 96;
        spec->mantissa_dtype = "bf8e5m2";
        spec->group_m = 1; spec->group_n = 1; spec->group_k = 96;
        break;
    case 5:
        spec->M = 80; spec->N = 80; spec->K = 160;
        spec->mantissa_dtype = "fp8e4m3";
        spec->group_m = 1; spec->group_n = 1; spec->group_k = 160;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_block_scale_gemm_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_ir_builder_t b;
    ckc_kernel_def_t *kernel = ckc_build_block_scale_gemm_new(&b, &spec, "gfx950");
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

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
    ckc_ir_builder_free(&b);
    return 0;
}
