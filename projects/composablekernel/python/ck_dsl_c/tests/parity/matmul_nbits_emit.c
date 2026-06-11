/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/matmul_nbits_emit.c -- C-side emitter for the MatMulNBits parity
 * harness. Selects one of 6 configs by argv[1] (config index 0..5), builds the
 * ckc_matmul_nbits_spec_t identically to the Python emitter matmul_nbits_emit.py,
 * dispatches+builds via ckc_build_matmul_nbits (after initialising the builder
 * with spec.kernel_name(), exactly as Python's IRBuilder(spec.kernel_name())),
 * lowers via ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and prints the
 * .ll to stdout so the two outputs can be byte-compared.
 *
 * On a validation reject (or any other build/lower failure) nothing is written
 * to stdout and the program exits non-zero; the harness treats a both-sides
 * reject (empty stdout + nonzero exit) as parity.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_matmul_nbits.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_matmul_nbits_spec_t *spec) {
    *spec = ckc_matmul_nbits_spec_default();

    switch (idx) {
    case 0:
        spec->name = "matmul_nbits_gfx950";
        spec->N = 4096; spec->K = 4096;
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 64, .tile_n = 128, .tile_k = 16,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "large_n";
        spec->optimized = false;
        break;
    case 1:
        spec->name = "matmul_nbits_gfx950_skinny";
        spec->N = 32; spec->K = 4096;
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 64, .tile_n = 32, .tile_k = 16,
            .warp_m = 2, .warp_n = 1, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "skinny_n";
        spec->optimized = false;
        break;
    case 2:
        spec->name = "matmul_nbits_gfx950_gemv";
        spec->N = 248320; spec->K = 4096;
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 1, .tile_n = 256, .tile_k = 16,
            .warp_m = 1, .warp_n = 8, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "decode_gemv";
        spec->optimized = false;
        break;
    case 3:
        spec->name = "matmul_nbits_gfx950_large_8k";
        spec->N = 8192; spec->K = 4096;
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 64, .tile_n = 128, .tile_k = 16,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp32";
        spec->family = "large_n";
        spec->optimized = false;
        break;
    case 4:
        spec->name = "matmul_nbits_gfx950_opt";
        spec->N = 4096; spec->K = 4096;
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 64, .tile_n = 128, .tile_k = 32,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "large_n";
        spec->optimized = true;
        break;
    case 5:
        spec->name = "matmul_nbits_gfx950_12k";
        spec->N = 12288; spec->K = 4096;
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 64, .tile_n = 128, .tile_k = 16,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "large_n";
        spec->optimized = false;
        break;
    default:
        return -1;
    }
    ckc_matmul_nbits_spec_finalize(spec);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_matmul_nbits_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    /* Mirror Python build_matmul_nbits: IRBuilder(spec.kernel_name()) then
     * dispatch via build_matmul_nbits(spec, arch). */
    char kname[256];
    if (ckc_matmul_nbits_kernel_name(&spec, kname, sizeof kname) != CKC_OK) {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, kname) != CKC_OK) {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    ckc_kernel_def_t *kernel = ckc_build_matmul_nbits(&b, &spec, "gfx950");
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    char *llvm_text = NULL;
    ckc_status_t st =
        ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
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
