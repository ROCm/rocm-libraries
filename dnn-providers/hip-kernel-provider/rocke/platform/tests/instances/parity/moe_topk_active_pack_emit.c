/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C-side parity emitter for the fused top-k active-expert pack builder.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_moe_topk_active_pack.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

static int make_spec(int index, rocke_moe_topk_active_pack_spec_t* spec)
{
    *spec = rocke_moe_topk_active_pack_spec_default();
    if(index == 0)
    {
        spec->tokens = 1;
        spec->experts = 896;
        spec->topk = 16;
        return 0;
    }
    if(index == 1)
    {
        spec->tokens = 8;
        spec->experts = 896;
        spec->topk = 16;
        return 0;
    }
    return -1;
}

int main(int argc, char** argv)
{
    int index;
    const char* mode;
    rocke_moe_topk_active_pack_spec_t spec;
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
    if(!rocke_moe_topk_active_pack_is_valid_spec(&spec, "gfx950", reason, sizeof(reason)))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    kernel = rocke_build_moe_topk_active_pack_new(&builder, &spec, "gfx950");
    if(kernel == NULL)
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&builder));
        rocke_ir_builder_free(&builder);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* text = NULL;
        rocke_status_t status
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &text);
        if(status != ROCKE_OK || text == NULL)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)status);
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
        size_t i;
        rocke_verify(kernel, &diagnostics, &count);
        for(i = 0; i < count; ++i)
        {
            char* text = rocke_diag_to_string(&diagnostics[i]);
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
