/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/moe_sorting_emit.c -- C-side emitter for the MoE-sorting
 * instance parity harness. Selects one of the sampled configs by argv[2] (the
 * config index) and the phase by argv[1] ("hist"/"scan"/"scatter"), builds
 * ckc_moe_sorting_spec_t identically to the Python emitter moe_sorting_emit.py,
 * validates via ckc_moe_sorting_is_valid_spec, builds into a fresh IRBuilder via
 * the matching ckc_build_moe_sort_*_new C build entry, lowers via
 * ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_moe_sorting.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_moe_sorting_spec_t *spec) {
    *spec = ckc_moe_sorting_spec_default();

    switch (idx) {
    case 0:
        spec->tokens = 2;   spec->topk = 8;  spec->experts = 8;
        spec->block_size = 64;
        break;
    case 1:
        spec->tokens = 16;  spec->topk = 4;  spec->experts = 32;
        spec->block_size = 256;
        break;
    case 2:
        spec->tokens = 32;  spec->topk = 8;  spec->experts = 64;
        spec->block_size = 256;
        break;
    case 3:
        spec->tokens = 128; spec->topk = 2;  spec->experts = 32;
        spec->block_size = 512;
        break;
    case 4:
        spec->tokens = 8;   spec->topk = 16; spec->experts = 16;
        spec->block_size = 128;
        break;
    case 5:
        spec->tokens = 2;   spec->topk = 8;  spec->experts = 64;
        spec->block_size = 256;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <phase hist|scan|scatter> <config_index 0..5>\n",
                argv[0]);
        return 2;
    }
    const char *phase = argv[1];
    int idx = atoi(argv[2]);

    ckc_moe_sorting_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char *arch = "gfx950";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[CKC_ERR_MSG_CAP];
    reason[0] = 0;
    if (!ckc_moe_sorting_is_valid_spec(&spec, arch, reason, sizeof reason)) {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* Select the phase build entry. */
    ckc_kernel_def_t *(*build_new)(ckc_ir_builder_t *,
                                   const ckc_moe_sorting_spec_t *,
                                   const char *) = NULL;
    if (strcmp(phase, "hist") == 0) {
        build_new = ckc_build_moe_sort_histogram_new;
    } else if (strcmp(phase, "scan") == 0) {
        build_new = ckc_build_moe_sort_scan_new;
    } else if (strcmp(phase, "scatter") == 0) {
        build_new = ckc_build_moe_sort_scatter_new;
    } else {
        fprintf(stderr, "unknown phase '%s'\n", phase);
        return 2;
    }

    /* Init IRBuilder with spec.kernel_name(<phase>) and build into it. */
    ckc_ir_builder_t b;
    ckc_kernel_def_t *kernel = build_new(&b, &spec, arch);
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
