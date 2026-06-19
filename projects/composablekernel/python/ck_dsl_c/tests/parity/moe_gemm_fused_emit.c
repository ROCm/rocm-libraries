/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/moe_gemm_fused_emit.c -- C-side emitter for the moe_gemm_fused
 * parity harness. Selects one of N sampled spec configs by argv[1] (the config
 * index), builds the matching Fused* spec identically to the Python emitter
 * moe_gemm_fused_emit.py, builds the kernel via the matching build entry +
 * lowers via ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and prints the
 * .ll to stdout so the two outputs can be byte-compared.
 *
 * The six configs map onto the three primary builders:
 *   [0] gate_up_silu  f16            (pad_m/n)
 *   [1] gate_up_silu  f16  grouped   (pad_m/n)
 *   [2] interleaved   f16            (pad_m/n)
 *   [3] down_reduce   f16            (pad_m/n)
 *   [4] interleaved   bf16 grouped   (default trait)
 *   [5] down_reduce   f16  grouped   (default trait)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_moe_gemm_fused.h"

/* TraitSpec(pad_m=True, pad_n=True): a fresh TraitSpec dataclass default
 * (epilogue defaults to "cshuffle", NOT the spec field-default "default"),
 * with pad_m/pad_n set. Mutate the spec-default trait to match. */
static void set_trait_pad_mn(ckc_gemm_trait_spec_t* tr) {
    tr->epilogue = "cshuffle"; /* TraitSpec dataclass default */
    tr->pad_m = true;
    tr->pad_n = true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char *mode = (argc > 2) ? argv[2] : "ll";

    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = NULL;

    switch (idx) {
    case 0: {
        ckc_moe_gate_up_silu_gemm_spec_t spec =
            ckc_moe_gate_up_silu_gemm_spec_default();
        spec.name = "moe_gate_up_silu_f16";
        spec.tile = (ckc_gemm_tile_spec_t){
            .tile_m = 32, .tile_n = 128, .tile_k = 64,
            .warp_m = 1, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        set_trait_pad_mn(&spec.trait);
        spec.dtype = "f16";
        spec.grouped = false;
        ckc_moe_gate_up_silu_gemm_spec_finalize(&spec);
        kernel = ckc_build_moe_gate_up_silu_gemm_new(&b, &spec, "gfx950");
        break;
    }
    case 1: {
        ckc_moe_gate_up_silu_gemm_spec_t spec =
            ckc_moe_gate_up_silu_gemm_spec_default();
        spec.name = "moe_gate_up_silu_grouped_f16";
        spec.tile = (ckc_gemm_tile_spec_t){
            .tile_m = 32, .tile_n = 128, .tile_k = 64,
            .warp_m = 1, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        set_trait_pad_mn(&spec.trait);
        spec.dtype = "f16";
        spec.grouped = true;
        ckc_moe_gate_up_silu_gemm_spec_finalize(&spec);
        kernel = ckc_build_moe_gate_up_silu_gemm_new(&b, &spec, "gfx950");
        break;
    }
    case 2: {
        ckc_moe_interleaved_gate_up_silu_gemm_spec_t spec =
            ckc_moe_interleaved_gate_up_silu_gemm_spec_default();
        spec.name = "moe_interleaved_gate_up_silu_f16";
        spec.tile = (ckc_gemm_tile_spec_t){
            .tile_m = 32, .tile_n = 32, .tile_k = 32,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 16};
        set_trait_pad_mn(&spec.trait);
        spec.dtype = "f16";
        spec.grouped = false;
        ckc_moe_interleaved_gate_up_silu_gemm_spec_finalize(&spec);
        kernel = ckc_build_moe_interleaved_gate_up_silu_gemm_new(&b, &spec, "gfx950");
        break;
    }
    case 3: {
        ckc_moe_down_reduce_gemm_spec_t spec =
            ckc_moe_down_reduce_gemm_spec_default();
        spec.name = "moe_down_reduce_f16";
        spec.tile = (ckc_gemm_tile_spec_t){
            .tile_m = 32, .tile_n = 128, .tile_k = 64,
            .warp_m = 1, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        set_trait_pad_mn(&spec.trait);
        spec.dtype = "f16";
        spec.grouped = false;
        ckc_moe_down_reduce_gemm_spec_finalize(&spec);
        kernel = ckc_build_moe_down_reduce_gemm_new(&b, &spec, "gfx950");
        break;
    }
    case 4: {
        ckc_moe_interleaved_gate_up_silu_gemm_spec_t spec =
            ckc_moe_interleaved_gate_up_silu_gemm_spec_default();
        spec.name = "moe_interleaved_bf16_grouped";
        spec.tile = (ckc_gemm_tile_spec_t){
            .tile_m = 32, .tile_n = 32, .tile_k = 32,
            .warp_m = 2, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 16, .warp_tile_n = 16, .warp_tile_k = 16};
        /* No trait override in Python -> spec field default trait. */
        spec.dtype = "bf16";
        spec.grouped = true;
        ckc_moe_interleaved_gate_up_silu_gemm_spec_finalize(&spec);
        kernel = ckc_build_moe_interleaved_gate_up_silu_gemm_new(&b, &spec, "gfx950");
        break;
    }
    case 5: {
        ckc_moe_down_reduce_gemm_spec_t spec =
            ckc_moe_down_reduce_gemm_spec_default();
        spec.name = "moe_down_grouped";
        spec.tile = (ckc_gemm_tile_spec_t){
            .tile_m = 32, .tile_n = 128, .tile_k = 64,
            .warp_m = 1, .warp_n = 2, .warp_k = 1,
            .warp_tile_m = 32, .warp_tile_n = 32, .warp_tile_k = 16};
        /* No trait override in Python -> spec field default trait. */
        spec.dtype = "f16";
        spec.grouped = true;
        ckc_moe_down_reduce_gemm_spec_finalize(&spec);
        kernel = ckc_build_moe_down_reduce_gemm_new(&b, &spec, "gfx950");
        break;
    }
    default:
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    if (kernel == NULL) {
        const char* m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    if (strcmp(mode, "ll") == 0) {
        char* llvm_text = NULL;
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
        char *t = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if (st != CKC_OK || !t) {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    } else if (strcmp(mode, "verify") == 0) {
        ckc_diag_t *d = NULL;
        size_t n = 0;
        ckc_verify(kernel, &d, &n);
        for (size_t i = 0; i < n; i++) {
            char *s = ckc_diag_to_string(&d[i]);
            if (s) { puts(s); free(s); }
        }
        ckc_diags_free(d, n);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        ckc_ir_builder_free(&b);
        return 2;
    }
    ckc_ir_builder_free(&b);
    return 0;
}
