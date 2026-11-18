/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 * C++ builder oracle matching peer_gemm_emit.py, including both supported targets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_peer_gemm.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int index = atoi(argv[1]);
    const char* mode = argc > 2 ? argv[2] : "ll";
    if(index < 0 || index >= 12)
    {
        fprintf(stderr, "unknown config index %d\n", index);
        return 2;
    }
    if(strcmp(mode, "ll") && strcmp(mode, "ir") && strcmp(mode, "verify"))
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    const int rows[] = { 16, 16, 32, 16, 32, 32 };
    const int columns[] = { 16, 16, 16, 32, 32, 32 };
    const int k[] = { 16, 16, 32, 64, 64, 32 };
    const int sources[] = { 1, 2, 3, 8, 4, 3 };
    const char* dtype[] = { "f16", "bf16", "f16", "bf16", "f16", "bf16" };
    rocke_peer_gemm_spec_t spec = rocke_peer_gemm_spec_default();
    spec.rows_per_source = rows[index / 2];
    spec.cols_per_rank = columns[index / 2];
    spec.k = k[index / 2];
    spec.sources = sources[index / 2];
    spec.dtype = dtype[index / 2];
    const char* arch = index % 2 ? "gfx942" : "gfx950";
    rocke_ir_builder_t builder;
    memset(&builder, 0, sizeof(builder));
    rocke_kernel_def_t* kernel = rocke_build_peer_gemm_new(&builder, &spec, arch);
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
