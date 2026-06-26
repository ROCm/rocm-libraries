/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1250_qwen3_qk_norm_rope_emit.c -- C-side emitter for the
 * gfx1250 fused QK-norm + RoPE parity harness. Selects one of the sampled
 * Qwen3QkNormRopeSpec configs by argv[1], builds it exactly as the Python
 * emitter does, and lowers to LLVM .ll text at arch=gfx1250 (flavor AUTO).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_qwen3_qk_norm_rope.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec)
{
    *spec = rocke_qwen3_qk_norm_rope_gfx1250_spec_default();

    switch(idx)
    {
    case 0: /* Qwen3QkNormRopeSpec(num_heads=32) -- Q on Qwen3-30B-A3B */
        spec->num_heads = 32;
        break;
    case 1: /* num_heads=4 -- K on Qwen3-30B-A3B */
        spec->num_heads = 4;
        break;
    case 2: /* num_heads=32, dtype="fp16" */
        spec->num_heads = 32;
        spec->dtype = "fp16";
        break;
    case 3: /* num_heads=8, rope_layout="interleaved" */
        spec->num_heads = 8;
        spec->rope_layout = "interleaved";
        break;
    case 4: /* num_heads=16, head_dim=128, block_size=128 */
        spec->num_heads = 16;
        spec->head_dim = 128;
        spec->block_size = 128;
        break;
    case 5: /* num_heads=4, head_dim=128, dtype="fp16", interleaved */
        spec->num_heads = 4;
        spec->head_dim = 128;
        spec->dtype = "fp16";
        spec->rope_layout = "interleaved";
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
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    rocke_qwen3_qk_norm_rope_gfx1250_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char name[256];
    if(rocke_qwen3_qk_norm_rope_gfx1250_kernel_name(&spec, name, sizeof name) != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_build_qwen3_qk_norm_rope_gfx1250(&b, &spec, "gfx1250");
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx1250", &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else
    {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        rocke_ir_builder_free(&b);
        return 2;
    }

    rocke_ir_builder_free(&b);
    return 0;
}
