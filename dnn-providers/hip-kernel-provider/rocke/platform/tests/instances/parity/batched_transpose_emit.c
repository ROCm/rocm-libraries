/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/batched_transpose_emit.c -- C-side emitter for the batched 2D
 * transpose instance parity harness. Selects one of the sampled (tile_m,
 * tile_n,vec,dtype,lds_pad,name,arch) configs by argv[1] (the config index),
 * builds rocke_batched_transpose2d_spec_t identically to the Python emitter
 * batched_transpose_emit.py, validates it with
 * rocke_batched_transpose2d_is_valid_spec, builds into a fresh IRBuilder via
 * rocke_build_batched_transpose2d_new (the C build entry, which inits the
 * builder with spec.kernel_name()), lowers via rocke_lower_kernel_to_llvm
 * (flavor AUTO, the config's arch) and prints the .ll to stdout so the two
 * outputs can be byte-compared.
 *
 * A rejected config prints "invalid spec: <reason>" to stdout and exits 1 on
 * both sides, so the gate compares the reject reasons byte for byte.
 *
 * Optional argv[2] = mode: "ll" (default), "ir", "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_batched_transpose.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` and `arch` for config index `idx`. Returns 0 on success, -1 if
 * unknown. */
static int make_spec(int idx, rocke_batched_transpose2d_spec_t* spec, const char** arch)
{
    *spec = rocke_batched_transpose2d_spec_default();
    *arch = "gfx950";

    switch(idx)
    {
    case 0:
        spec->tile_m = 16;
        spec->tile_n = 16;
        spec->vec = 2;
        spec->dtype = "f16";
        spec->lds_pad = 8;
        break;
    case 1:
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->vec = 4;
        spec->dtype = "f16";
        spec->lds_pad = 8;
        break;
    case 2:
        /* Every field at the dataclass default. */
        break;
    case 3:
        spec->tile_m = 64;
        spec->tile_n = 64;
        spec->vec = 8;
        spec->dtype = "bf16";
        spec->lds_pad = 8;
        break;
    case 4:
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->vec = 8;
        spec->dtype = "bf16";
        spec->lds_pad = 0;
        break;
    case 5:
        spec->tile_m = 16;
        spec->tile_n = 16;
        spec->vec = 4;
        spec->dtype = "f16";
        spec->lds_pad = 4;
        break;
    case 6:
        spec->tile_m = 64;
        spec->tile_n = 64;
        spec->vec = 4;
        spec->dtype = "bf16";
        spec->lds_pad = 16;
        break;
    case 7:
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->vec = 2;
        spec->dtype = "fp16";
        spec->lds_pad = 8;
        spec->name = "rocke_bxpose_custom";
        break;
    case 8:
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->vec = 4;
        spec->dtype = "f16";
        spec->lds_pad = 8;
        *arch = "gfx1151";
        break;
    /* Rejected by both engines; the reason text is compared. */
    case 9:
        *arch = "gfx000";
        break;
    case 10:
        spec->dtype = "f32";
        break;
    case 11:
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->vec = 3;
        break;
    case 12:
        spec->tile_m = 32;
        spec->tile_n = 16;
        spec->vec = 4;
        break;
    case 13:
        /* (tile_m * tile_n) // vec floors to -2. */
        spec->tile_m = -3;
        spec->tile_n = 1;
        spec->vec = 2;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..13> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_batched_transpose2d_spec_t spec;
    const char* arch = NULL;
    if(make_spec(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = 0;
    if(!rocke_batched_transpose2d_is_valid_spec(&spec, arch, reason, sizeof reason))
    {
        printf("invalid spec: %s\n", reason);
        return 1;
    }

    /* Init IRBuilder with spec.kernel_name() and build into it. */
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof(b));
    rocke_kernel_def_t* kernel = rocke_build_batched_transpose2d_new(&b, &spec, arch);
    if(!kernel)
    {
        fprintf(stderr, "build failed: %s\n", b.err);
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    else
    {
        /* mode == "ll" */
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    rocke_ir_builder_free(&b);
    return 0;
}
