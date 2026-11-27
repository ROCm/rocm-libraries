// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* C counterpart of batchnorm2d_ro_emit.py: builds the read-once batchnorm2d
 * kernel of one config and prints its .ll / IR / verifier output. The config
 * table MUST match CONFIGS in batchnorm2d_ro_emit.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_batchnorm2d_ro.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct
{
    const char* kind;
    int block_size, vec, regs;
    const char* dtype;
    bool fast_reduce;
    bool save_mean_invstd;
} ro_cfg_t;

/* MUST match CONFIGS in batchnorm2d_ro_emit.py (index for index). */
static const ro_cfg_t CONFIGS[] = {
    {"train", 256, 8, 16, "f16", false, true},
    {"bwd", 256, 8, 16, "f16", false, true},
    {"train", 512, 4, 2, "bf16", false, true},
    {"bwd", 512, 4, 32, "bf16", false, true},
    {"train", 256, 8, 4, "f16", true, true},
    {"bwd", 256, 4, 8, "f16", true, true},
    {"train", 256, 8, 2, "f16", false, false},
    {"train", 512, 8, 32, "f16", false, true},
    {"bwd", 256, 8, 2, "bf16", false, true},
};

#define NCFG ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NCFG - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= NCFG)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_batchnorm2d_ro_spec_t spec = rocke_batchnorm2d_ro_spec_default();
    spec.block_size = CONFIGS[idx].block_size;
    spec.vec = CONFIGS[idx].vec;
    spec.regs = CONFIGS[idx].regs;
    spec.dtype = CONFIGS[idx].dtype;
    spec.fast_reduce = CONFIGS[idx].fast_reduce;
    spec.save_mean_invstd = CONFIGS[idx].save_mean_invstd;

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel_built = rocke_build_batchnorm2d_ro_new(&b, &spec, CONFIGS[idx].kind);
    if(kernel_built == NULL || !rocke_ir_builder_ok(&b))
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    { /* verify */
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

    rocke_ir_builder_free(&b);
    return 0;
}
