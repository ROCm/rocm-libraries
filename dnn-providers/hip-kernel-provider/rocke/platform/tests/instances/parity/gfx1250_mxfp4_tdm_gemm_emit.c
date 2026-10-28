/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C-side representative MX FP4 TDM/LDS GEMM parity emitter. Its config order
 * matches gfx1250_mxfp4_tdm_gemm_emit.py exactly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_mxfp4_tdm_gemm.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

int main(int argc, char** argv)
{
    rocke_mxfp4_tdm_gemm_spec_t spec = rocke_mxfp4_tdm_gemm_spec_default();
    const char* mode;
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..0> [mode]\n", argv[0]);
        return 2;
    }
    if(atoi(argv[1]) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", atoi(argv[1]));
        return 2;
    }
    mode = argc > 2 ? argv[2] : "ll";
    spec.M = 64;
    spec.N = 32;
    spec.K = 256;
    spec.tile_m = 64;
    spec.tile_n = 32;
    if(rocke_ir_builder_init(&b, "mxfp4_tdm_gemm_mxfp4_tdm_M64N32K256_t64x32x128_w2x2x1_r1x1_db")
       != ROCKE_OK)
    {
        return 1;
    }
    kernel = rocke_build_mxfp4_tdm_gemm(&b, &spec, "gfx1250");
    if(kernel == NULL || !rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    if(strcmp(mode, "ll") == 0)
    {
        char* text = NULL;
        char err[ROCKE_ERR_MSG_CAP] = {0};
        if(rocke_lower_kernel_to_llvm_ex(
               kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx1250", &text, err, sizeof(err))
               != ROCKE_OK
           || text == NULL)
        {
            fprintf(stderr, "lower failed: %s\n", err);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        if(rocke_ir_serialize(kernel, &text) != ROCKE_OK || text == NULL)
        {
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diags = NULL;
        size_t count = 0;
        rocke_verify(kernel, &diags, &count);
        rocke_diags_free(diags, count);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        rocke_ir_builder_free(&b);
        return 2;
    }
    rocke_ir_builder_free(&b);
    return 0;
}
