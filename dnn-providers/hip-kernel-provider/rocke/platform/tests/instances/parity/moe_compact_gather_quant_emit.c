/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C-side parity emitter for the compact routed gather plus blockwise FP8
 * quantization instance. Configs match moe_compact_gather_quant_emit.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_moe_compact_gather_quant.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

static int make_spec(int index, rocke_moe_compact_gather_quant_spec_t* spec)
{
    *spec = rocke_moe_compact_gather_quant_spec_default();
    switch(index)
    {
    case 0:
        spec->tokens = 8;
        spec->hidden = 3584;
        spec->max_blocks = 128;
        spec->input_dtype = "bf16";
        break;
    case 1:
        spec->tokens = 32;
        spec->hidden = 4096;
        spec->max_blocks = 64;
        spec->input_dtype = "f16";
        spec->block_size = 128;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    int index;
    const char* mode;
    rocke_moe_compact_gather_quant_spec_t spec;
    rocke_ir_builder_t builder;
    rocke_kernel_def_t* kernel;
    char reason[ROCKE_ERR_MSG_CAP] = {0};

    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..1> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    index = atoi(argv[1]);
    mode = argc > 2 ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(make_spec(index, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", index);
        return 2;
    }
    if(!rocke_moe_compact_gather_quant_is_valid_spec(&spec, "gfx950", reason, sizeof(reason)))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    kernel = rocke_build_moe_compact_gather_quant_new(&builder, &spec, "gfx950");
    if(kernel == NULL)
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&builder));
        rocke_ir_builder_free(&builder);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* text = NULL;
        char error[ROCKE_ERR_MSG_CAP] = {0};
        rocke_status_t status = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &text, error, sizeof(error));
        if(status != ROCKE_OK || text == NULL)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)status, error);
            rocke_ir_builder_free(&builder);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t status = rocke_ir_serialize(kernel, &text);
        if(status != ROCKE_OK || text == NULL)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)status);
            rocke_ir_builder_free(&builder);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    {
        rocke_diag_t* diagnostics = NULL;
        size_t count = 0;
        size_t diagnostic_index;
        rocke_verify(kernel, &diagnostics, &count);
        for(diagnostic_index = 0; diagnostic_index < count; ++diagnostic_index)
        {
            char* text = rocke_diag_to_string(&diagnostics[diagnostic_index]);
            if(text != NULL)
            {
                puts(text);
                free(text);
            }
        }
        rocke_diags_free(diagnostics, count);
    }

    rocke_ir_builder_free(&builder);
    return 0;
}
