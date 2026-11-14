/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_lds_emit.c -- C-side emitter for the IR-emitting
 * LDS helper of the attention backward (rocke_attn_bwd_emit_lds_dma of
 * helper_rocke.instances.common._attention_bwd_caps.h). Builds, identically to
 * attention_bwd_lds_emit.py, a probe kernel that issues one LDS DMA per
 * (dwords, coherency) of the config, lowers it via rocke_lower_kernel_to_llvm
 * (flavor AUTO) and prints the .ll so the two outputs can be byte-compared.
 *
 * Config index map (keep in sync with attention_bwd_lds_emit.py):
 *   0  gfx942   1 dword
 *   1  gfx942   1 dword, coherency 1, 2, 3
 *   2  gfx950   1, 3 and 4 dwords
 *   3  gfx950   4 dwords coherency 3, 3 dwords coherency 2, 1 dword coherency 1
 *   4  gfx942   rejected: 1 dword, then 4 dwords
 *   5  gfx950   rejected: 2 dwords
 *   6  gfx1151  rejected: no LDS DMA path
 *   7  gfx1201  rejected: no LDS DMA path
 *   8  gfx950   rejected: 0 dwords
 *   9  gfx950   rejected: coherency 4 (builder)
 *
 * argv[2] selects the output mode: "ll" (default), "ir" or "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct
{
    int dwords;
    int coherency;
} load_t;

typedef struct
{
    const char* arch;
    load_t loads[3];
    int n;
} config_t;

static const config_t CONFIGS[] = {
    {"gfx942", {{1, 0}}, 1},
    {"gfx942", {{1, 1}, {1, 2}, {1, 3}}, 3},
    {"gfx950", {{1, 0}, {3, 0}, {4, 0}}, 3},
    {"gfx950", {{4, 3}, {3, 2}, {1, 1}}, 3},
    {"gfx942", {{1, 0}, {4, 0}}, 2},
    {"gfx950", {{2, 0}}, 1},
    {"gfx1151", {{1, 0}}, 1},
    {"gfx1201", {{4, 0}}, 1},
    {"gfx950", {{0, 0}}, 1},
    {"gfx950", {{4, 4}}, 1},
};

#define N_CONFIGS ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

static rocke_kernel_def_t* build(rocke_ir_builder_t* b, const config_t* c)
{
    if(rocke_ir_builder_init(b, "attn_bwd_lds_probe") != ROCKE_OK)
        return NULL;
    const rocke_type_t* src_ty = rocke_ptr_type(b, rocke_f16(), "global");
    rocke_value_t* src = rocke_b_param(b, "src", src_ty, NULL);
    rocke_value_t* nbytes = rocke_b_param(b, "nbytes", rocke_i32(), NULL);
    rocke_value_t* soff = rocke_b_param(b, "soff", rocke_i32(), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, src, nbytes);
    int shape[2] = {c->n, 512};
    rocke_value_t* slab = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "slab");
    rocke_value_t* addr = rocke_b_smem_addr_of(b, slab);
    for(int i = 0; i < c->n; ++i)
    {
        rocke_value_t* piece = rocke_b_const_i32(b, i * 1024);
        rocke_value_t* piece64 = rocke_b_zext(b, piece, rocke_i64());
        rocke_value_t* lds = rocke_b_smem_ptr_add(b, addr, piece64);
        rocke_value_t* width = rocke_b_const_i32(b, c->loads[i].dwords * 4);
        rocke_value_t* voff = rocke_b_mul(b, tid, width);
        if(rocke_attn_bwd_emit_lds_dma(
               b, c->arch, rsrc, lds, voff, soff, c->loads[i].dwords, c->loads[i].coherency)
           != ROCKE_OK)
            return NULL;
    }
    rocke_b_ret(b);
    return rocke_ir_builder_status(b) == ROCKE_OK ? rocke_ir_builder_kernel(b) : NULL;
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
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= N_CONFIGS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const config_t* c = &CONFIGS[idx];
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = build(&b, c);
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
        char* text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, c->arch, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(text, stdout);
            free(text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(text, stdout);
            free(text);
        }
    }
    else
    {
        rocke_diag_t* diags = NULL;
        size_t ndiag = 0;
        rocke_verify(kernel, &diags, &ndiag);
        for(size_t i = 0; i < ndiag; i++)
        {
            char* d = rocke_diag_to_string(&diags[i]);
            if(d)
            {
                fputs(d, stdout);
                fputc('\n', stdout);
                free(d);
            }
        }
        rocke_diags_free(diags, ndiag);
    }
    rocke_ir_builder_free(&b);
    return ret;
}
