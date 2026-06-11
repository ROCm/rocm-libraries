/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/layernorm2d_emit.c -- C-side emitter for the LayerNorm2D parity
 * harness. Selects one of 6 sampled configs by argv[1] (the config index 0..5),
 * builds ckc_layernorm2d_spec_t identically to the Python emitter
 * layernorm2d_emit.py, lowers via ckc_layernorm2d_lower_to_llvm (arch gfx950,
 * flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_layernorm2d.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_layernorm2d_spec_t *spec) {
    *spec = ckc_layernorm2d_spec_default();

    switch (idx) {
    case 0:
        spec->n_per_block = 4096;
        spec->block_size = 256;
        spec->vec = 4;
        spec->dtype = "f16";
        spec->save_mean_invstd = false;
        break;
    case 1:
        spec->n_per_block = 4096;
        spec->block_size = 256;
        spec->vec = 8;
        spec->dtype = "f16";
        spec->save_mean_invstd = false;
        break;
    case 2:
        spec->n_per_block = 4096;
        spec->block_size = 256;
        spec->vec = 4;
        spec->dtype = "bf16";
        spec->save_mean_invstd = false;
        break;
    case 3:
        spec->n_per_block = 2048;
        spec->block_size = 128;
        spec->vec = 4;
        spec->dtype = "f16";
        spec->save_mean_invstd = true;
        break;
    case 4:
        spec->n_per_block = 8192;
        spec->block_size = 256;
        spec->vec = 8;
        spec->dtype = "f16";
        spec->save_mean_invstd = false;
        break;
    case 5:
        spec->n_per_block = 1024;
        spec->block_size = 256;
        spec->vec = 2;
        spec->dtype = "bf16";
        spec->save_mean_invstd = true;
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

    ckc_layernorm2d_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char *llvm_text = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = 0;
    ckc_status_t st = ckc_layernorm2d_lower_to_llvm(
        &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    return 0;
}
