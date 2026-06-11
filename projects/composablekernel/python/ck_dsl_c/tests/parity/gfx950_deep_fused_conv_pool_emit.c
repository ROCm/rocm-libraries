/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx950_deep_fused_conv_pool_emit.c -- C reference emitter for the
 * gfx950 deep fused conv0 -> conv1 -> maxpool parity harness. Selects one of the
 * sampled spec configs by argv[1], builds the gfx950-pinned spec via
 * ckc_gfx950_deep_fused_conv_pool_make_spec(...), builds+lowers via
 * ckc_gfx950_deep_fused_conv_pool_lower_to_llvm(spec, "gfx950", AUTO, ...) and
 * prints the .ll to stdout so it can be byte/sha256-compared with the Python
 * reference deep_fused_conv_pool_emit.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_gfx950_deep_fused_conv_pool.h"

static int make_cfg(int idx, ckc_gfx950_deep_fused_conv_pool_spec_t *spec) {
    switch (idx) {
    case 0:
        *spec = ckc_gfx950_deep_fused_conv_pool_make_spec(
            1, 112, 112, 64, 64, 64, 3, 3, 4, 8, 32, 16, 0,
            2, 1, NULL, false, false, false, false);
        return 0;
    case 1:
        *spec = ckc_gfx950_deep_fused_conv_pool_make_spec(
            1, 56, 56, 128, 128, 128, 3, 3, 4, 8, 32, 16, 0,
            2, 1, NULL, false, false, false, false);
        return 0;
    case 2:
        *spec = ckc_gfx950_deep_fused_conv_pool_make_spec(
            1, 28, 28, 256, 256, 256, 3, 3, 4, 8, 32, 16, 0,
            2, 1, NULL, false, false, false, false);
        return 0;
    case 3:
        /* cache_input_footprint=true */
        *spec = ckc_gfx950_deep_fused_conv_pool_make_spec(
            1, 56, 56, 32, 32, 32, 3, 3, 4, 8, 32, 16, 0,
            2, 1, NULL, false, false, true, false);
        return 0;
    case 4:
        /* direct_conv0_from_input_cache=true */
        *spec = ckc_gfx950_deep_fused_conv_pool_make_spec(
            1, 28, 28, 64, 64, 64, 3, 3, 4, 8, 32, 16, 0,
            2, 1, NULL, false, false, false, true);
        return 0;
    default:
        return -1;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    ckc_gfx950_deep_fused_conv_pool_spec_t spec;
    if (make_cfg(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    char *llvm = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = '\0';
    ckc_status_t st = ckc_gfx950_deep_fused_conv_pool_lower_to_llvm(
        &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm, err, sizeof(err));
    if (st != CKC_OK || !llvm) {
        fprintf(stderr, "lower failed cfg%d status=%d (%s)\n",
                idx, (int)st, err[0] ? err : "(no message)");
        return 1;
    }
    fputs(llvm, stdout);
    free(llvm);
    return 0;
}
