/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/flatmm_emit.c -- C-side emitter for the FlatMM parity harness.
 * Selects one of 6 sampled FlatMMSpec configs by argv[1] (0..5), builds
 * ckc_flatmm_spec_t identically to the Python emitter flatmm_emit.py, builds
 * the kernel via ckc_build_flatmm and lowers via ckc_lower_kernel_to_llvm
 * (arch gfx950, flavor AUTO), and prints the .ll to stdout so the two outputs
 * can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_flatmm.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_flatmm_spec_t *spec) {
    *spec = ckc_flatmm_spec_default();
    spec->name = "ck_dsl_flatmm";
    spec->wave_size = 64;
    spec->batch_size = 0;
    spec->preshuffle_b = false;

    switch (idx) {
    case 0:
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 128, .tile_n = 128, .tile_k = 64,
            .warp_m = 1, .warp_n = 4, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.scheduler = "intrawave";
        spec->trait.epilogue = "cshuffle";
        spec->block_size = 256;
        break;
    case 1:
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 128, .tile_n = 128, .tile_k = 64,
            .warp_m = 1, .warp_n = 4, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        spec->trait.pipeline = "mem";
        spec->trait.scheduler = "intrawave";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 2:
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 128, .tile_n = 128, .tile_k = 64,
            .warp_m = 1, .warp_n = 4, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.scheduler = "intrawave";
        spec->trait.epilogue = "cshuffle";
        spec->block_size = 256;
        break;
    case 3:
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 64, .tile_n = 64, .tile_k = 32,
            .warp_m = 1, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.scheduler = "intrawave";
        spec->trait.epilogue = "cshuffle";
        spec->block_size = 128;
        break;
    case 4:
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 128, .tile_n = 128, .tile_k = 64,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.scheduler = "intrawave";
        spec->trait.epilogue = "cshuffle";
        spec->block_size = 512;
        break;
    case 5:
        spec->tile = (ckc_gemm_tile_spec_t){
            .tile_m = 256, .tile_n = 256, .tile_k = 64,
            .warp_m = 2, .warp_n = 4, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.scheduler = "intrawave";
        spec->trait.epilogue = "cshuffle";
        spec->block_size = 512;
        break;
    default:
        return -1;
    }
    ckc_flatmm_spec_finalize(spec);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_flatmm_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_ir_builder_t b;
    char namebuf[256];
    if (ckc_flatmm_kernel_name(&spec, namebuf, sizeof namebuf) != CKC_OK) {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }
    ckc_ir_builder_init(&b, namebuf);

    ckc_kernel_def_t *kernel = ckc_build_flatmm(&b, &spec, "gfx950");
    if (!kernel) {
        fprintf(stderr, "build_flatmm failed: %s\n", ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return 1;
    }

    char *llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(
        kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
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
