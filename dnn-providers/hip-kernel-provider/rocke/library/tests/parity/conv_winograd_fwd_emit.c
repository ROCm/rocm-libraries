/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_winograd_fwd_emit.c -- C-side emitter for the Winograd
 * forward transform parity harness. Selects one of N sampled configs by
 * argv[1], builds the rocke_winograd_fwd_spec_t identically to the Python
 * emitter conv_winograd_fwd_emit.py (same table, same order), builds the stage
 * (rocke_build_winograd_fwd_stage_new) and lowers it for the config's arch,
 * printing the .ll (or the serialized IR / verifier output) to stdout.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_winograd_fwd.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct cfg
{
    int N, Hi, Wi, C, K, Y, X, pad, dil, groups;
    int m;
    const char* io;
    const char* op;
    int vec, block, ps, tpt;
    const char* pmode;
    int bh, bw, wgr, nt, lanes;
    const char* stage;
    const char* arch;
} cfg_t;

static const cfg_t CONFIGS[] = {
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K1", "gfx1250"}, /* 0: F4 fp16 K1 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 1: F4 fp16 K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K4", "gfx1250"}, /* 2: F4 fp16 K4 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K1", "gfx1250"}, /* 3: F2 K1 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "fp16",   "fp16",
     8, 256, 0,  2,  "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 4: F2 K2, 2 tiles per thread */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K4", "gfx1250"}, /* 5: F2 K4 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "bf16",   "fp16",
     8, 256, 1,  1,  "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 6: F4 bf16 fused prescale K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "bf16",   "fp16",
     8, 256, 1,  1,  "fused", 0, 0, 1, 1, 32, "K4", "gfx1250"}, /* 7: F4 bf16 fused prescale K4 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,     "bf16",   "fp16",
     8, 256, 1,  1,  "split", 0, 0, 1, 1, 32, "K2a", "gfx1250"}, /* 8: F4 bf16 split prescale K2a */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "bf16",   "fp16",
     8, 256, 1,  1,  "split", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 9: F4 bf16 split prescale K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "bf16",   "bf16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K1", "gfx1250"}, /* 10: F2 bf16 operands K1 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "bf16",   "bf16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 11: F2 bf16 operands K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "bf16",   "bf16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K4", "gfx1250"}, /* 12: F2 bf16 operands K4 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "fp16",   "fp16",
     8, 256, 0,  4,  "fused", 4, 8, 1, 1, 16, "K2", "gfx1250"}, /* 13: F2 LDS region 4x8, 4 tiles per thread, 16 lanes K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "bf16",   "fp16",
     8, 256, 1,  1,  "fused", 2, 4, 1, 1, 32, "K2", "gfx1250"}, /* 14: F4 bf16 LDS region 2x4 + split prescale K2 */
    {1, 40,  40, 128, 64,      3, 3, 2, 2, 1,  4,    "fp16",   "fp16",
     4, 256, 0,  1,   "fused", 0, 0, 1, 1, 32, "K1", "gfx1250"}, /* 15: F4 dilation 2 K1 (vec 4) */
    {1, 40,  40, 128, 64,      3, 3, 2, 2, 1,  4,    "fp16",   "fp16",
     4, 256, 0,  1,   "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 16: F4 dilation 2 K2 */
    {1, 40,  40, 128, 64,      3, 3, 2, 2, 1,  4,    "fp16",   "fp16",
     4, 256, 0,  1,   "fused", 0, 0, 1, 1, 32, "K4", "gfx1250"}, /* 17: F4 dilation 2 K4 */
    {2, 16,  16, 128, 128,     3, 3, 1, 1, 4,  2,    "fp16",   "fp16",
     8, 256, 0,  2,   "fused", 0, 0, 2, 1, 32, "K2", "gfx1250"}, /* 18: F2 groups 4 K2, 2 tiles per thread, 2 tile rows */
    {2, 16,  16, 128, 128,     3, 3, 1, 1, 4,  2,    "fp16",   "fp16",
     8, 256, 0,  1,   "fused", 0, 0, 1, 1, 32, "K1", "gfx1250"}, /* 19: F2 groups 4 K1 */
    {2, 16,  16, 128, 128,     3, 3, 1, 1, 4,  2,    "fp16",   "fp16",
     8, 256, 0,  1,   "fused", 0, 0, 1, 1, 32, "K4", "gfx1250"}, /* 20: F2 groups 4 K4 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 0, 32, "K1", "gfx1250"}, /* 21: F4 regular stores K1 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 0, 32, "K2", "gfx1250"}, /* 22: F4 regular stores K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 0, 32, "K4", "gfx1250"}, /* 23: F4 regular stores K4 */
    {1, 13,  11, 33, 35,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K1", "gfx1250"}, /* 24: F4 odd channels (vec 1) K1 */
    {1, 13,  11, 33, 35,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 25: F4 odd channels (vec 1) K2 */
    {1, 13,  11, 33, 35,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K4", "gfx1250"}, /* 26: F4 odd channels (vec 1) K4 */
    {1, 13,  11, 33, 35,      3, 3, 1, 1, 1,  4,     "bf16",   "fp16",
     8, 256, 1,  1,  "split", 0, 0, 1, 1, 32, "K2a", "gfx1250"}, /* 27: F4 bf16 odd channels split prescale K2a */
    {1, 8,   8, 512, 64,      3, 3, 1, 1, 1,  4,    "bf16",   "fp16",
     4, 256, 1, 1,   "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 28: F4 bf16 fused prescale over 128 lanes (LDS group max) K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  3,  "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 29: F4 K2 3 tiles per thread (partial runs) */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  4,    "fp16",  "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K2", "gfx950"}, /* 30: gfx950: F4 K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "bf16",  "fp16",
     8, 256, 1,  1,  "fused", 0, 0, 1, 1, 32, "K2", "gfx950"}, /* 31: gfx950: F2 bf16 fused prescale K2 */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  3,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K2", "gfx1250"}, /* 32: rejected: m = 3 */
    {1, 12,  12, 64, 64,      5, 5, 2, 1, 1,  4,    "fp16",   "fp16",
     8, 256, 0,  1,  "fused", 0, 0, 1, 1, 32, "K1", "gfx1250"}, /* 33: rejected: 5x5 filter */
    {2, 20,  24, 64, 96,      3, 3, 1, 1, 1,  2,    "fp16",   "fp16",
     8, 256, 0,  4,  "fused", 2, 3, 1, 1, 32, "K2", "gfx1250"}, /* 34: rejected: tiles_per_thread does not divide the LDS block */
};
static const int NUM_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(idx < 0 || idx >= NUM_CONFIGS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const cfg_t* c = &CONFIGS[idx];
    rocke_winograd_fwd_spec_t s = rocke_winograd_fwd_spec_default();
    s.problem = rocke_conv_problem_make(
        c->N, c->Hi, c->Wi, c->C, c->K, c->Y, c->X, 1, 1, c->pad, c->pad, c->dil, c->dil);
    s.problem.groups = c->groups;
    s.m = c->m;
    s.io_dtype = c->io;
    s.operand_dtype = c->op;
    s.vec = c->vec;
    s.block_size = c->block;
    s.prescale = c->ps != 0;
    s.tiles_per_thread = c->tpt;
    s.prescale_mode = c->pmode;
    s.input_block_tiles[0] = c->bh;
    s.input_block_tiles[1] = c->bw;
    s.wg_tile_rows = c->wgr;
    s.nt_stores = c->nt != 0;
    s.lds_lanes = c->lanes;

    rocke_winograd_fwd_stage_t stage;
    if(!rocke_winograd_fwd_stage_from_name(c->stage, &stage))
    {
        fprintf(stderr, "bad stage %s\n", c->stage);
        return 2;
    }
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_winograd_fwd_stage_new(&b, &s, stage, c->arch);
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
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, c->arch, &llvm_text);
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
            char* t = rocke_diag_to_string(&diags[i]);
            if(t)
            {
                fputs(t, stdout);
                fputc('\n', stdout);
                free(t);
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
