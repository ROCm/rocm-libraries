/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/transpose_emit.c -- C-side emitter for the 2D transpose instance
 * parity harness. Selects one of the sampled (tile_m,tile_n,vec,dtype,lds_pad,
 * grid_order) configs by argv[1] (the config index), builds
 * ckc_transpose2d_spec_t identically to the Python emitter transpose_emit.py,
 * builds into a fresh IRBuilder via ckc_build_transpose2d_new (the C build
 * entry, which auto-inits the builder with spec.kernel_name()), lowers via
 * ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_transpose.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_transpose2d_spec_t *spec) {
    *spec = ckc_transpose2d_spec_default();

    switch (idx) {
    case 0:
        spec->tile_m = 16; spec->tile_n = 16; spec->vec = 2;
        spec->dtype = "f16"; spec->lds_pad = 8; spec->grid_order = "row";
        break;
    case 1:
        spec->tile_m = 32; spec->tile_n = 32; spec->vec = 4;
        spec->dtype = "f16"; spec->lds_pad = 8; spec->grid_order = "row";
        break;
    case 2:
        spec->tile_m = 64; spec->tile_n = 64; spec->vec = 8;
        spec->dtype = "f16"; spec->lds_pad = 8; spec->grid_order = "row";
        break;
    case 3:
        spec->tile_m = 64; spec->tile_n = 64; spec->vec = 8;
        spec->dtype = "bf16"; spec->lds_pad = 8; spec->grid_order = "row";
        break;
    case 4:
        spec->tile_m = 32; spec->tile_n = 32; spec->vec = 4;
        spec->dtype = "bf16"; spec->lds_pad = 8; spec->grid_order = "morton";
        break;
    case 5:
        spec->tile_m = 16; spec->tile_n = 16; spec->vec = 4;
        spec->dtype = "f16"; spec->lds_pad = 8; spec->grid_order = "row";
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_transpose2d_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char *arch = "gfx950";

    /* Init IRBuilder with spec.kernel_name() and build into it. */
    ckc_ir_builder_t b;
    ckc_kernel_def_t *kernel = ckc_build_transpose2d_new(&b, &spec, arch);
    if (!kernel) {
        fprintf(stderr, "build failed: %s\n", b.err);
        ckc_ir_builder_free(&b);
        return 1;
    }

    /* lower_kernel_to_llvm(kernel, arch='gfx950'). */
    char *llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(
        kernel, CKC_LLVM_FLAVOR_AUTO, arch, &llvm_text);
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
