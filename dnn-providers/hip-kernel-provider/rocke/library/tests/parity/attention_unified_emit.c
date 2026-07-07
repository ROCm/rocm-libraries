/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_unified_emit.c -- C-side emitter for the unified
 * (scalar 2D) attention parity harness. Selects one of the sampled configs by
 * argv[1] (the config index), fills a rocke_unified_attention_problem_t
 * identically to the Python emitter attention_unified_emit.py, builds the
 * scalar 2D kernel via rocke_build_unified_attention_2d_scalar (kernel name from
 * rocke_unified_attention_2d_scalar_kernel_name), lowers the returned KernelDef
 * via rocke_lower_kernel_to_llvm_ex (arch gfx950, flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_attention_unified.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `p` for config index `idx`. Returns 0 on success, -1 on unknown idx. */
static int make_problem(int idx, rocke_unified_attention_problem_t* p)
{
    *p = rocke_unified_attention_problem_default();
    switch(idx)
    {
    case 0:
        p->head_size = 64;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 64;
        p->num_kv_heads = 8;
        p->total_q = 128;
        p->max_seqlen_q = 512;
        p->max_seqlen_k = 512;
        p->use_sinks = true;
        p->softcap = 0.0;
        break;
    case 1:
        p->head_size = 128;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 64;
        p->num_kv_heads = 8;
        p->total_q = 256;
        p->max_seqlen_q = 1024;
        p->max_seqlen_k = 1024;
        p->use_sinks = true;
        p->softcap = 0.0;
        break;
    case 2:
        p->head_size = 256;
        p->block_size = 16;
        p->dtype = "fp16";
        p->num_query_heads = 32;
        p->num_kv_heads = 4;
        p->total_q = 64;
        p->max_seqlen_q = 256;
        p->max_seqlen_k = 256;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;
    case 3:
        p->head_size = 64;
        p->block_size = 16;
        p->dtype = "bf16";
        p->num_query_heads = 16;
        p->num_kv_heads = 2;
        p->total_q = 512;
        p->max_seqlen_q = 2048;
        p->max_seqlen_k = 2048;
        p->use_sinks = true;
        p->sliding_window = 128;
        p->softcap = 0.0;
        break;
    case 4:
        p->head_size = 128;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 64;
        p->num_kv_heads = 8;
        p->total_q = 1;
        p->max_seqlen_q = 1;
        p->max_seqlen_k = 4096;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;
    case 5:
        p->head_size = 64;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 128;
        p->num_kv_heads = 16;
        p->total_q = 1024;
        p->max_seqlen_q = 4096;
        p->max_seqlen_k = 4096;
        p->use_sinks = true;
        p->softcap = 50.0;
        break;
    default:
        return -1;
    }
    /* num_seqs is a runtime kernel param (not a build-time constant) in the
     * scalar 2D builder, so its value does not affect the emitted IR. Mirror
     * the Python emitter's choice for clarity. */
    p->num_seqs = 1;
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

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_unified_attention_problem_t p;
    if(make_problem(idx, &p) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }

    rocke_ir_builder_t b;
    char kname[256];
    kname[0] = 0;
    rocke_status_t nst
        = rocke_unified_attention_2d_scalar_kernel_name(&p, NULL, kname, sizeof kname);
    if(nst != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed: status=%d\n", (int)nst);
        return 1;
    }
    rocke_ir_builder_init(&b, kname);

    rocke_kernel_def_t* kernel = rocke_build_unified_attention_2d_scalar(&b, &p, NULL);
    if(!kernel)
    {
        fprintf(stderr, "build failed for config %d (sticky err)\n", idx);
        return 1;
    }

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
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
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
