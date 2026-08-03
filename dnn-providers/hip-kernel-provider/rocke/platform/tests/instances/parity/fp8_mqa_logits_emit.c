/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C-side emitter for gfx942 FP8-logits byte identity.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_fp8_mqa_logits.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static int make_spec(int idx, rocke_fp8_mqa_logits_spec_t* spec)
{
    *spec = rocke_fp8_mqa_logits_spec_default();
    switch(idx)
    {
    case 0:
        break;
    case 1:
        spec->waves_per_block = 2;
        break;
    case 2:
        spec->head_dim = 64;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    rocke_fp8_mqa_logits_spec_t spec;
    int idx;
    const char* mode;

    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..2> [mode]\n", argv[0]);
        return 2;
    }
    idx = atoi(argv[1]);
    mode = argc > 2 ? argv[2] : "ll";
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        rocke_status_t status;
        err[0] = '\0';
        status = rocke_fp8_mqa_logits_lower_to_llvm(
            &spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof(err));
        if(status != ROCKE_OK || llvm_text == NULL)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)status, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0 || strcmp(mode, "verify") == 0)
    {
        rocke_ir_builder_t builder;
        rocke_kernel_def_t* kernel = rocke_build_fp8_mqa_logits_new(&builder, &spec, "gfx942");
        if(kernel == NULL || !rocke_ir_builder_ok(&builder))
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
            rocke_verify(kernel, &diagnostics, &count);
            for(size_t i = 0; i < count; ++i)
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
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    return 0;
}
