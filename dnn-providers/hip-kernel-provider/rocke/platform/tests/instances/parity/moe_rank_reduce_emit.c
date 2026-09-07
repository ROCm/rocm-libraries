/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_moe_rank_reduce.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

static int emit_rmsnorm(int index, const char* mode)
{
    rocke_moe_rank_reduce_rmsnorm_spec_t spec = rocke_moe_rank_reduce_rmsnorm_spec_default();
    rocke_ir_builder_t builder;
    rocke_kernel_def_t* kernel;

    if(index == 0)
    {
        spec.width = 3584;
        spec.world_size = 8;
    }
    else
    {
        spec.width = 2048;
        spec.world_size = 4;
        spec.dtype = "f16";
        spec.block_size = 128;
        spec.vec = 4;
        spec.fp32_internal = true;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* text = NULL;
        char error[ROCKE_ERR_MSG_CAP] = {0};
        rocke_status_t status = rocke_moe_rank_reduce_rmsnorm_lower_to_llvm(
            &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &text, error, sizeof(error));
        if(status != ROCKE_OK || text == NULL)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)status, error);
            return 1;
        }
        fputs(text, stdout);
        free(text);
        return 0;
    }

    kernel = rocke_build_moe_rank_reduce_rmsnorm_new(&builder, &spec, "gfx950");
    if(kernel == NULL)
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&builder));
        rocke_ir_builder_free(&builder);
        return 1;
    }
    if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t status = rocke_ir_serialize(kernel, &text);
        if(status != ROCKE_OK || text == NULL)
        {
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

static int emit_scatter(const char* mode)
{
    rocke_moe_rank_reduce_scatter_spec_t spec = rocke_moe_rank_reduce_scatter_spec_default();
    rocke_ir_builder_t builder;
    rocke_kernel_def_t* kernel;
    spec.width = 7168;
    spec.world_size = 8;

    if(strcmp(mode, "ll") == 0)
    {
        char* text = NULL;
        char error[ROCKE_ERR_MSG_CAP] = {0};
        rocke_status_t status = rocke_moe_rank_reduce_scatter_lower_to_llvm(
            &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &text, error, sizeof(error));
        if(status != ROCKE_OK || text == NULL)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)status, error);
            return 1;
        }
        fputs(text, stdout);
        free(text);
        return 0;
    }

    kernel = rocke_build_moe_rank_reduce_scatter_new(&builder, &spec, "gfx950");
    if(kernel == NULL)
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&builder));
        rocke_ir_builder_free(&builder);
        return 1;
    }
    if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t status = rocke_ir_serialize(kernel, &text);
        if(status != ROCKE_OK || text == NULL)
        {
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

int main(int argc, char** argv)
{
    int index;
    const char* mode;
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    index = atoi(argv[1]);
    mode = argc > 2 ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(index == 0 || index == 1)
        return emit_rmsnorm(index, mode);
    if(index == 2)
        return emit_scatter(mode);
    fprintf(stderr, "unknown config index %d\n", index);
    return 2;
}
