/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Standalone C++ builder oracle for peer_tile_copy_emit.py.
 * Configs intentionally include partial thread chunks and both supported
 * targets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_peer_tile_copy.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: <config_index> [ll|ir|verify]\n");
        return 2;
    }
    const int rows[] = { 4, 4, 3, 3, 7, 7, 9, 9, 1, 1 };
    const int cols[] = { 64, 64, 17, 17, 33, 33, 65, 65, 1, 1 };
    const int blocks[] = { 64, 64, 64, 64, 128, 128, 256, 256, 256, 256 };
    const char* dtypes[]
        = { "bf16", "bf16", "f16", "f16", "bf16", "bf16", "f16", "f16", "bf16", "bf16" };
    int index = atoi(argv[1]);
    if(index < 0 || index >= 10)
    {
        fprintf(stderr, "unknown config index %d\n", index);
        return 2;
    }
    const char* mode = argc > 2 ? argv[2] : "ll";
    if(strcmp(mode, "ll") && strcmp(mode, "ir") && strcmp(mode, "verify"))
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    rocke_peer_tile_copy_spec_t spec = rocke_peer_tile_copy_spec_default();
    spec.tile_rows = rows[index];
    spec.tile_cols = cols[index];
    spec.block_size = blocks[index];
    spec.dtype = dtypes[index];
    const char* arch = index % 2 ? "gfx942" : "gfx950";
    rocke_ir_builder_t builder;
    memset(&builder, 0, sizeof(builder));
    rocke_kernel_def_t* kernel = rocke_build_peer_tile_copy_new(&builder, &spec, arch);
    if(kernel == NULL)
    {
        fprintf(stderr, "build failed: %s\n", builder.err);
        rocke_ir_builder_free(&builder);
        return 1;
    }
    int status = 0;
    if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diagnostics = NULL;
        size_t count = 0;
        rocke_verify(kernel, &diagnostics, &count);
        for(size_t i = 0; i < count; ++i)
        {
            char* text = rocke_diag_to_string(&diagnostics[i]);
            if(text)
            {
                puts(text);
                free(text);
            }
        }
        rocke_diags_free(diagnostics, count);
    }
    else
    {
        char* text = NULL;
        rocke_status_t result = strcmp(mode, "ir") == 0
            ? rocke_ir_serialize(kernel, &text)
            : rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &text);
        if(result != ROCKE_OK || text == NULL)
        {
            fprintf(stderr, "emit failed: status=%d\n", (int)result);
            status = 1;
        }
        else
        {
            fputs(text, stdout);
        }
        free(text);
    }
    rocke_ir_builder_free(&builder);
    return status;
}
