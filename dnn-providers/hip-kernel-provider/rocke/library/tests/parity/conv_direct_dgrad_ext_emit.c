/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_dgrad_ext_emit.c -- C-side emitter for the direct
 * dgrad parity family the AOT dgrad kernels added after conv_direct_grouped
 * extend. Selects one config by argv[1], builds the spec identically to the
 * Python emitter conv_direct_dgrad_ext_emit.py, builds the kernel and lowers
 * it via rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing
 * the .ll to stdout so the two outputs can be byte-compared.
 *
 * Indices 0..10: rocke_direct_depthwise_dgrad_win_spec_t, VALU and dot2 forms;
 * 11..18: its Toeplitz MFMA form; 19: block_h = 0 (both reject).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct win_cfg
{
    int N, H, W, G, KH, KW, PAD;
    const char* dtype;
    const char* arch;
    int block_w, block_waves, ch_per_lane, block_h, dot2, mfma, w_fold, prefetch_rows;
} win_cfg_t;

/* Knob columns: block_w, block_waves, ch_per_lane, block_h, dot2, mfma,
 * w_fold, prefetch_rows (the spec defaults where the Python leaves them). */
static const win_cfg_t WIN_CFGS[] = {
    {2, 14, 14, 64, 7, 7, 3, "bf16", "gfx950", 7, 1, 1, 14, 0, 0, 1, 1},
    {2, 14, 14, 64, 7, 7, 3, "bf16", "gfx950", 7, 1, 1, 14, 1, 0, 1, 1},
    {2, 12, 12, 128, 3, 3, 1, "fp16", "gfx950", 6, 2, 2, 12, 0, 0, 1, 1},
    {2, 13, 13, 64, 5, 5, 2, "fp16", "gfx950", 5, 1, 1, 4, 0, 0, 1, 1},
    {1, 8, 13, 40, 3, 3, 1, "bf16", "gfx950", 4, 1, 1, 4, 1, 0, 1, 1},
    {2, 9, 9, 72, 4, 4, 1, "fp16", "gfx950", 8, 1, 1, 9, 1, 0, 1, 1},
    {2, 8, 8, 64, 3, 3, 1, "bf16", "gfx942", 4, 1, 1, 8, 1, 0, 1, 1},
    {2, 7, 7, 128, 3, 3, 1, "bf16", "gfx950", 4, 1, 4, 8, 0, 0, 1, 1},
    {2, 9, 11, 96, 3, 1, 1, "fp16", "gfx950", 5, 1, 1, 9, 1, 0, 1, 1},
    {1, 13, 10, 72, 3, 5, 2, "fp16", "gfx950", 10, 2, 1, 5, 1, 0, 1, 1},
    {2, 9, 9, 64, 5, 5, 2, "fp16", "gfx942", 8, 4, 1, 7, 0, 0, 1, 1},
    {3, 14, 14, 64, 7, 7, 3, "bf16", "gfx950", 8, 8, 1, 14, 0, 1, 2, 2},
    {2, 13, 37, 40, 5, 5, 2, "fp16", "gfx950", 8, 4, 1, 4, 0, 1, 1, 1},
    {2, 10, 11, 16, 9, 9, 4, "bf16", "gfx950", 8, 2, 1, 5, 0, 1, 4, 3},
    {1, 13, 13, 64, 11, 11, 5, "fp16", "gfx950", 8, 8, 1, 4, 0, 1, 2, 2},
    {2, 14, 14, 64, 7, 7, 3, "bf16", "gfx942", 8, 8, 1, 14, 0, 1, 2, 2},
    {1, 9, 12, 8, 3, 3, 0, "fp16", "gfx950", 8, 1, 1, 9, 0, 1, 2, 1},
    {2, 14, 14, 12, 7, 7, 3, "bf16", "gfx950", 8, 1, 1, 14, 0, 1, 2, 2},
    {2, 11, 15, 32, 7, 5, 2, "bf16", "gfx950", 8, 4, 1, 12, 0, 1, 2, 2},
    {2, 8, 8, 64, 3, 3, 1, "fp16", "gfx950", 8, 1, 1, 0, 0, 0, 1, 1},
};

static int make_cfg(int idx, rocke_direct_depthwise_dgrad_win_spec_t* spec, const char** arch)
{
    const int n = (int)(sizeof(WIN_CFGS) / sizeof(WIN_CFGS[0]));
    if(idx < 0 || idx >= n)
        return -1;
    const win_cfg_t* c = &WIN_CFGS[idx];
    *spec = rocke_direct_depthwise_dgrad_win_spec_default();
    spec->problem = rocke_direct_conv_problem_default();
    spec->problem.N = c->N;
    spec->problem.H = c->H;
    spec->problem.W = c->W;
    spec->problem.groups = c->G;
    spec->problem.cpg = 1;
    spec->problem.kpg = 1;
    spec->problem.KH = c->KH;
    spec->problem.KW = c->KW;
    spec->problem.PAD = c->PAD;
    spec->problem.stride = 1;
    spec->problem.dtype = c->dtype;
    spec->block_w = c->block_w;
    spec->block_waves = c->block_waves;
    spec->ch_per_lane = c->ch_per_lane;
    spec->block_h = c->block_h;
    spec->dot2 = c->dot2 != 0;
    spec->mfma = c->mfma != 0;
    spec->w_fold = c->w_fold;
    spec->prefetch_rows = c->prefetch_rows;
    *arch = c->arch;
    return 0;
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

    rocke_direct_depthwise_dgrad_win_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_direct_depthwise_dgrad_win_new(&b, &spec, arch);
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
