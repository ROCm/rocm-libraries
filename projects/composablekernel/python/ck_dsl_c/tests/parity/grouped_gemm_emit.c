/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/grouped_gemm_emit.c -- C-side emitter for the grouped-GEMM
 * parity harness. Selects one of the sampled GroupedGemmSpec configs by
 * argv[1] (the config index), builds ckc_grouped_gemm_spec_t identically to
 * the Python emitter grouped_gemm_emit.py, builds via the C build entry
 * (ckc_build_grouped_gemm) and lowers via ckc_lower_kernel_to_llvm (arch
 * gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared. Uses the convenience one-call lower path
 * ckc_grouped_gemm_lower_to_llvm which owns its own IRBuilder.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_grouped_gemm.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_grouped_gemm_spec_t *spec) {
    *spec = ckc_grouped_gemm_spec_default();

    switch (idx) {
    case 0:
        spec->name = "ggemm_fp16_m128n128k32";
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 128, .tile_n = 128, .tile_k = 32,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    case 1:
        spec->name = "ggemm_bf16_m32n32k32";
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 32, .tile_n = 32, .tile_k = 32,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 32};
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "cshuffle";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->dtype = "bf16";
        spec->wave_size = 64;
        break;
    case 2:
        spec->name = "ggemm_fp16_m64n64k64";
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 64, .tile_n = 64, .tile_k = 64,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    case 3:
        spec->name = "ggemm_fp16_m256n256k128";
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 256, .tile_n = 256, .tile_k = 128,
            .warp_m = 4, .warp_n = 4, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.chiplet_swizzle = true;
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    default:
        return -1;
    }
    ckc_grouped_gemm_spec_finalize(spec);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..3>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_grouped_gemm_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char *llvm_text = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = 0;
    ckc_status_t st = ckc_grouped_gemm_lower_to_llvm(
        &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    return 0;
}
