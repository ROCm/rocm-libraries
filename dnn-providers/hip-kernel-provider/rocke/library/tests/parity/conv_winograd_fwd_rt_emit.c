/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_winograd_fwd_rt_emit.c -- C-side emitter for the
 * shape-generic Winograd forward transform parity harness. Selects one of N
 * sampled configs by argv[1], builds the rocke_winograd_rt_spec_t identically
 * to the Python emitter conv_winograd_fwd_rt_emit.py (same table, same
 * order), builds the stage (rocke_build_winograd_rt_new) and lowers it for the
 * config's arch, printing the .ll (or the serialized IR / verifier output) to
 * stdout.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_winograd_fwd_rt.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct cfg
{
    const char* stage;
    const char* arch;
    int mh, mw;
    const char* io;
    const char* op;
    int vec, fr, block, ps, tpt, bh, bw, lanes, nt, fused;
} cfg_t;

/* clang-format off */
static const cfg_t CONFIGS[] = {
    {"K1", "gfx1250", 4, 4, "fp16", "fp16", 8, 2, 256, 0, 1, 0, 0, 16, 1, 0}, /* 0: F4 fp16 K1 (2 filter rows) */
    {"K1", "gfx1250", 2, 2, "bf16", "bf16", 8, 4, 256, 0, 1, 0, 0, 16, 1, 0}, /* 1: F2 bf16 operands K1 (all rows) */
    {"K1", "gfx1250", 6, 4, "fp16", "fp16", 4, 2, 256, 0, 1, 0, 0, 16, 1, 0}, /* 2: F(6x4) K1 vec 4 */
    {"K1", "gfx1250", 2, 6, "fp16", "fp16", 1, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 3: F(2x6) K1 vec 1, 1 filter row */
    {"K1", "gfx1250", 4, 2, "bf16", "fp16", 2, 3, 256, 0, 1, 0, 0, 16, 1, 0}, /* 4: F(4x2) bf16 I/O K1 vec 2 */
    {"K2a", "gfx1250", 4, 4, "fp16", "fp16", 8, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 5: K2a fp16 vec 8 */
    {"K2a", "gfx1250", 4, 4, "bf16", "fp16", 1, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 6: K2a bf16 vec 1 */
    {"K2a", "gfx1250", 4, 4, "bf16", "fp16", 4, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 7: K2a bf16 vec 4 */
    {"K2", "gfx1250", 4, 4, "fp16", "fp16", 4, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 8: F4 K2 register, 1 tile */
    {"K2", "gfx1250", 2, 2, "fp16", "fp16", 8, 1, 256, 0, 2, 0, 0, 16, 1, 0}, /* 9: F2 K2 register, 2 tiles per thread */
    {"K2", "gfx1250", 6, 4, "fp16", "fp16", 4, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 10: F(6x4) K2 register vec 4 */
    {"K2", "gfx1250", 2, 4, "fp16", "fp16", 2, 1, 256, 0, 3, 0, 0, 16, 1, 0}, /* 11: F(2x4) K2 register 3 tiles per thread (partial runs) */
    {"K2", "gfx1250", 4, 4, "fp16", "fp16", 8, 1, 64, 0, 1, 0, 0, 16, 0, 0}, /* 12: F4 K2 regular stores, block 64 */
    {"K2", "gfx1250", 4, 4, "bf16", "fp16", 4, 1, 256, 1, 1, 0, 0, 16, 1, 0}, /* 13: F4 bf16 K2 split prescale */
    {"K2", "gfx1250", 4, 6, "bf16", "fp16", 2, 1, 256, 1, 2, 0, 0, 16, 1, 0}, /* 14: F(4x6) bf16 K2 split prescale 2 tiles */
    {"K2", "gfx1250", 4, 4, "bf16", "fp16", 4, 1, 256, 1, 1, 0, 0, 16, 1, 1}, /* 15: F4 bf16 K2 fused prescale block 256 (LDS group max) */
    {"K2", "gfx1250", 6, 4, "bf16", "fp16", 2, 1, 1024, 1, 1, 0, 0, 16, 1, 1}, /* 16: F(6x4) bf16 K2 fused prescale block 1024 */
    {"K2", "gfx1250", 4, 4, "bf16", "fp16", 1, 1, 32, 1, 1, 0, 0, 16, 1, 1}, /* 17: F4 bf16 K2 fused prescale block 32 (shuffles only) */
    {"K2", "gfx1250", 2, 2, "bf16", "bf16", 8, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 18: F2 bf16 operands K2 register */
    {"K2", "gfx1250", 2, 2, "fp16", "fp16", 8, 1, 256, 0, 4, 4, 8, 16, 1, 0}, /* 19: F2 K2 LDS 4x8, 4 tiles per thread, 16 lanes */
    {"K2", "gfx1250", 2, 2, "bf16", "bf16", 1, 1, 256, 0, 4, 4, 8, 16, 1, 0}, /* 20: F2 bf16 K2 LDS 4x8 vec 1 */
    {"K2", "gfx1250", 4, 4, "bf16", "fp16", 4, 1, 256, 1, 1, 2, 4, 8, 1, 0}, /* 21: F4 bf16 K2 LDS 2x4 + split prescale, 8 lanes */
    {"K2", "gfx1250", 4, 2, "fp16", "fp16", 2, 1, 256, 0, 1, 4, 1, 1, 1, 0}, /* 22: F(4x2) K2 LDS 4x1 (one tile column), 1 lane */
    {"K4", "gfx1250", 4, 4, "fp16", "fp16", 8, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 23: F4 K4 */
    {"K4", "gfx1250", 2, 2, "fp16", "fp16", 1, 1, 64, 0, 1, 0, 0, 16, 1, 0}, /* 24: F2 K4 vec 1 block 64 */
    {"K4", "gfx1250", 6, 4, "bf16", "fp16", 4, 1, 256, 1, 1, 0, 0, 16, 1, 0}, /* 25: F(6x4) bf16 K4 prescale */
    {"K4", "gfx1250", 2, 6, "fp16", "fp16", 8, 1, 256, 0, 1, 0, 0, 16, 0, 0}, /* 26: F(2x6) K4 regular stores */
    {"K4", "gfx1250", 6, 6, "fp16", "fp16", 2, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 27: F6 K4 */
    {"K2", "gfx950", 4, 4, "fp16", "fp16", 8, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 28: gfx950: F4 K2 register */
    {"K2", "gfx950", 4, 4, "bf16", "fp16", 8, 1, 256, 1, 1, 0, 0, 16, 1, 1}, /* 29: gfx950: F4 bf16 K2 fused prescale */
    {"K2", "gfx950", 2, 2, "fp16", "fp16", 8, 1, 256, 0, 4, 4, 8, 16, 1, 0}, /* 30: gfx950: F2 K2 LDS */
    {"K1", "gfx950", 4, 4, "fp16", "fp16", 8, 2, 256, 0, 1, 0, 0, 16, 1, 0}, /* 31: gfx950: F4 K1 */
    {"K4", "gfx950", 4, 4, "bf16", "fp16", 8, 1, 256, 1, 1, 0, 0, 16, 1, 0}, /* 32: gfx950: F4 bf16 K4 prescale */
    {"K2a", "gfx950", 4, 4, "bf16", "fp16", 8, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 33: gfx950: K2a */
    {"K2", "gfx1250", 3, 4, "fp16", "fp16", 8, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 34: rejected: tile 3 */
    {"K1", "gfx1250", 4, 4, "bf16", "bf16", 8, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 35: rejected: bf16 operands with F4 */
    {"K4", "gfx1250", 4, 4, "fp16", "fp16", 3, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 36: rejected: vec 3 */
    {"K2", "gfx1250", 4, 4, "fp16", "fp16", 8, 1, 256, 0, 1, 0, 0, 16, 1, 1}, /* 37: rejected: fused without prescale */
    {"K2", "gfx1250", 2, 2, "bf16", "fp16", 8, 1, 256, 1, 4, 4, 8, 16, 1, 1}, /* 38: rejected: fused on the LDS path */
    {"K2", "gfx1250", 2, 2, "fp16", "fp16", 8, 1, 256, 0, 4, 2, 3, 16, 1, 0}, /* 39: rejected: tiles_per_thread does not divide the LDS block */
    {"K2", "gfx1250", 2, 2, "fp16", "fp16", 8, 1, 256, 0, 1, 8, 8, 32, 1, 0}, /* 40: rejected: LDS region over 1024 threads */
    {"K1", "gfx1250", 4, 4, "fp16", "fp16", 8, 4, 256, 0, 1, 0, 0, 16, 1, 0}, /* 41: rejected: filter_rows does not divide n */
    {"K4", "gfx1250", 4, 4, "fp16", "fp16", 8, 1, 96, 0, 1, 0, 0, 16, 1, 0}, /* 42: rejected: block not a power of two */
    {"K2a", "gfx1250", 4, 4, "fp16", "fp16", 16, 1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 43: rejected: K2a vec 16 */
    {"K1", "gfx1250", 4, 4, "fp16", "fp16", 8, -1, 256, 0, 1, 0, 0, 16, 1, 0}, /* 44: rejected: filter_rows -1 */
    {"K2", "gfx1250", 4, 4, "fp16", "fp16", 8, 1, 2048, 0, 1, 0, 0, 16, 1, 0}, /* 45: rejected: K2 block over 1024 */
    {"K4", "gfx1250", 4, 4, "fp16", "fp16", 8, 1, 2048, 0, 1, 0, 0, 16, 1, 0}, /* 46: rejected: K4 block over 1024 */
    {"K2", "gfx1250", 2, 2, "fp16", "fp16", 8, 1, 256, 0, 4, 4, 8, 3, 1, 0}, /* 47: rejected: LDS lanes not a power of two */
    {"K2", "gfx1250", 2, 2, "fp16", "fp16", 8, 1, 256, 0, 4, 4, 8, 0, 1, 0}, /* 48: rejected: LDS lanes 0 */
    {"K2", "gfx1250", 2, 2, "fp16", "fp16", 8, 1, 256, 0, 1, 4, 0, 16, 1, 0}, /* 49: rejected: LDS region of zero width */
};
/* clang-format on */

#define NUM_CONFIGS ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

static int stage_of(const char* name, rocke_winograd_rt_stage_t* out)
{
    if(strcmp(name, "K1") == 0)
        *out = ROCKE_WINOGRAD_RT_K1;
    else if(strcmp(name, "K2a") == 0)
        *out = ROCKE_WINOGRAD_RT_K2A;
    else if(strcmp(name, "K2") == 0)
        *out = ROCKE_WINOGRAD_RT_K2;
    else if(strcmp(name, "K4") == 0)
        *out = ROCKE_WINOGRAD_RT_K4;
    else
        return 0;
    return 1;
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
    if(idx < 0 || idx >= NUM_CONFIGS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const cfg_t* c = &CONFIGS[idx];
    rocke_winograd_rt_stage_t stage;
    if(!stage_of(c->stage, &stage))
    {
        fprintf(stderr, "bad stage %s\n", c->stage);
        return 2;
    }
    rocke_winograd_rt_spec_t s = rocke_winograd_rt_spec_default(stage);
    s.io_dtype = c->io;
    s.vec = c->vec;
    if(stage != ROCKE_WINOGRAD_RT_K2A)
    {
        s.mh = c->mh;
        s.mw = c->mw;
        s.operand_dtype = c->op;
    }
    if(stage == ROCKE_WINOGRAD_RT_K1)
        s.filter_rows = c->fr;
    if(stage == ROCKE_WINOGRAD_RT_K2 || stage == ROCKE_WINOGRAD_RT_K4)
    {
        s.block_size = c->block;
        s.prescale = c->ps != 0;
        s.nt_stores = c->nt != 0;
    }
    if(stage == ROCKE_WINOGRAD_RT_K2)
    {
        s.tiles_per_thread = c->tpt;
        s.lds[0] = c->bh;
        s.lds[1] = c->bw;
        s.lds_lanes = c->lanes;
        s.fused = c->fused != 0;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_winograd_rt_new(&b, &s, c->arch);
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
