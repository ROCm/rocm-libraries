/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_fwd_split_k_reduce_emit.c -- C-side emitter for the
 * split-K forward Stage 2 (conv_fwd_split_k_reduce) parity harness. Selects one
 * of N sampled spec configs by argv[1] (the config index), builds the
 * rocke_fwd_split_k_reduce_spec_t identically to the Python emitter
 * conv_fwd_split_k_reduce_emit.py, builds the kernel via
 * rocke_build_conv_fwd_split_k_reduce_new and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * Config index table (must stay in sync with the Python emitter):
 *   0  bf16, vec 4, runtime slab loop, gfx1250
 *   1  fp16, vec 4, runtime slab loop, gfx1250
 *   2  bf16, vec 2, runtime slab loop, gfx1250
 *   3  fp16, vec 1 (scalar store), runtime slab loop, gfx1250
 *   4  bf16, vec 4, unrolled fold, degree 4, gfx1250
 *   5  fp16, vec 2, unrolled fold, degree 3, gfx1250
 *   6  bf16, vec 1, unrolled fold, degree 16, gfx1250
 *   7  fp16, vec 4, block 128, unrolled fold, degree 8, gfx1250
 *   8  bf16, vec 4, runtime slab loop, gfx950
 *   9  vec 3 (rejected by both engines)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_fwd_split_k_reduce.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill the spec for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_cfg(int idx, rocke_fwd_split_k_reduce_spec_t* spec, const char** arch)
{
    static const struct
    {
        const char* dtype_d;
        int vec;
        int block_size;
        int degree;
        const char* arch;
    } kCfgs[] = {
        {"bf16", 4, 256, 0, "gfx1250"},
        {"fp16", 4, 256, 0, "gfx1250"},
        {"bf16", 2, 256, 0, "gfx1250"},
        {"fp16", 1, 256, 0, "gfx1250"},
        {"bf16", 4, 256, 4, "gfx1250"},
        {"fp16", 2, 256, 3, "gfx1250"},
        {"bf16", 1, 256, 16, "gfx1250"},
        {"fp16", 4, 128, 8, "gfx1250"},
        {"bf16", 4, 256, 0, "gfx950"},
        {"bf16", 3, 256, 0, "gfx1250"},
    };
    if(idx < 0 || idx >= (int)(sizeof(kCfgs) / sizeof(kCfgs[0])))
        return -1;
    *spec = rocke_fwd_split_k_reduce_spec_default();
    spec->dtype_d = kCfgs[idx].dtype_d;
    spec->vec = kCfgs[idx].vec;
    spec->block_size = kCfgs[idx].block_size;
    spec->degree = kCfgs[idx].degree;
    *arch = kCfgs[idx].arch;
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_fwd_split_k_reduce_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_conv_fwd_split_k_reduce_new(&b, &spec, arch);
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
        char* ir_text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &ir_text);
        if(st != ROCKE_OK || !ir_text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(ir_text, stdout);
        free(ir_text);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diags = NULL;
        size_t ndiag = 0;
        rocke_verify(kernel, &diags, &ndiag);
        for(size_t i = 0; i < ndiag; i++)
        {
            char* s = rocke_diag_to_string(&diags[i]);
            if(s)
            {
                fputs(s, stdout);
                fputc('\n', stdout);
                free(s);
            }
        }
        rocke_diags_free(diags, ndiag);
    }
    else
    {
        fprintf(stderr, "unknown mode '%s' (expected ll|ir|verify)\n", mode);
        ret = 2;
    }

    rocke_ir_builder_free(&b);
    return ret;
}
