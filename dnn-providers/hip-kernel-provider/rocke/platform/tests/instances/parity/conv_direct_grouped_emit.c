/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_grouped_emit.c -- C-side emitter for the direct
 * grouped convolution parity harness. Selects one of N sampled spec configs by
 * argv[1] (the config index), builds the rocke_direct_conv_16c_spec_t /
 * rocke_direct_conv_4c_spec_t / rocke_direct_conv_8c_spec_t /
 * rocke_direct_conv_32c_spec_t / rocke_direct_depthwise_spec_t identically to
 * the Python emitter conv_direct_grouped_emit.py, builds the kernel via the
 * matching rocke_build_direct_conv_*_new function and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

enum
{
    KIND_16C = 0,
    KIND_4C = 1,
    KIND_8C = 2,
    KIND_32C = 3,
    KIND_DW = 4
};

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *kind, the matching spec struct, and *arch. */
static int make_cfg(int idx,
                    int* kind,
                    rocke_direct_conv_16c_spec_t* s16,
                    rocke_direct_conv_4c_spec_t* s4,
                    rocke_direct_conv_8c_spec_t* s8,
                    rocke_direct_conv_32c_spec_t* s32,
                    rocke_direct_depthwise_spec_t* sdw,
                    const char** arch)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.KH = 3;
    p.KW = 3;
    p.PAD = 1;
    p.stride = 1;

    switch(idx)
    {
    case 0:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 4;
        s16->fold_k32 = true;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 1:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 8;
        s16->fold_k32 = true;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 2:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 64;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 4;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 3:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 64;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 8;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 4:
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 1;
        s16->fold_k32 = false;
        *kind = KIND_16C;
        *arch = "gfx942";
        return 0;
    case 5:
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 16;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 4;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 6:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 8;
        p.kpg = 8;
        *s8 = rocke_direct_conv_8c_spec_default();
        s8->problem = p;
        s8->block_q = 16;
        s8->block_groups = 8;
        s8->double_buffer = true;
        *kind = KIND_8C;
        *arch = "gfx950";
        return 0;
    case 7:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 8;
        p.cpg = 32;
        p.kpg = 32;
        *s32 = rocke_direct_conv_32c_spec_default();
        s32->problem = p;
        s32->block_q = 32;
        s32->block_groups = 4;
        s32->double_buffer = true;
        *kind = KIND_32C;
        *arch = "gfx950";
        return 0;
    case 8:
        /* groups must be divisible by block_ch = block_waves * wave_size (2 * 64 = 128) */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 128;
        p.cpg = 1;
        p.kpg = 1;
        *sdw = rocke_direct_depthwise_spec_default();
        sdw->problem = p;
        sdw->block_w = 16;
        sdw->block_waves = 2;
        *kind = KIND_DW;
        *arch = "gfx950";
        return 0;
    default:
        return -1;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    int kind = KIND_16C;
    rocke_direct_conv_16c_spec_t s16;
    rocke_direct_conv_4c_spec_t s4;
    rocke_direct_conv_8c_spec_t s8;
    rocke_direct_conv_32c_spec_t s32;
    rocke_direct_depthwise_spec_t sdw;
    const char* arch = "gfx950";
    if(make_cfg(idx, &kind, &s16, &s4, &s8, &s32, &sdw, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = NULL;
    if(kind == KIND_16C)
        kernel = rocke_build_direct_conv_16c_new(&b, &s16, arch);
    else if(kind == KIND_4C)
        kernel = rocke_build_direct_conv_4c_new(&b, &s4, arch);
    else if(kind == KIND_8C)
        kernel = rocke_build_direct_conv_8c_new(&b, &s8, arch);
    else if(kind == KIND_32C)
        kernel = rocke_build_direct_conv_32c_new(&b, &s32, arch);
    else
        kernel = rocke_build_direct_depthwise_new(&b, &sdw, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
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
    else if(strcmp(mode, "ir") == 0)
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
        fprintf(stderr, "unknown mode %s\n", mode);
        rocke_ir_builder_free(&b);
        return 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
