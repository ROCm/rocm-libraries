/* Stress parity emitter for unified attention scalar 2D (C side). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_attention_unified.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static int make_problem(int idx, rocke_unified_attention_problem_t* p)
{
    *p = rocke_unified_attention_problem_default();
    p->num_seqs = 1;
    switch(idx)
    {
    /* ---- baseline sampled (kept) ---- */
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

    /* ---- block_size=64 coverage (untested by baseline) ---- */
    case 6:
        p->head_size = 64;
        p->block_size = 64;
        p->dtype = "bf16";
        p->num_query_heads = 8;
        p->num_kv_heads = 1;
        p->total_q = 64;
        p->max_seqlen_q = 128;
        p->max_seqlen_k = 128;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;
    case 7:
        p->head_size = 256;
        p->block_size = 64;
        p->dtype = "fp16";
        p->num_query_heads = 8;
        p->num_kv_heads = 8;
        p->total_q = 32;
        p->max_seqlen_q = 64;
        p->max_seqlen_k = 64;
        p->use_sinks = true;
        p->softcap = 0.0;
        break;

    /* ---- fp16 + sinks + softcap combos ---- */
    case 8:
        p->head_size = 128;
        p->block_size = 16;
        p->dtype = "fp16";
        p->num_query_heads = 32;
        p->num_kv_heads = 8;
        p->total_q = 200;
        p->max_seqlen_q = 300;
        p->max_seqlen_k = 300;
        p->use_sinks = true;
        p->softcap = 30.0;
        break;
    case 9:
        p->head_size = 64;
        p->block_size = 32;
        p->dtype = "fp16";
        p->num_query_heads = 40;
        p->num_kv_heads = 5;
        p->total_q = 77;
        p->max_seqlen_q = 128;
        p->max_seqlen_k = 128;
        p->use_sinks = false;
        p->softcap = 10.0;
        break;

    /* ---- edge shapes: M/N/K = 1 ---- */
    case 10:
        p->head_size = 64;
        p->block_size = 16;
        p->dtype = "bf16";
        p->num_query_heads = 1;
        p->num_kv_heads = 1;
        p->total_q = 1;
        p->max_seqlen_q = 1;
        p->max_seqlen_k = 1;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;
    case 11:
        p->head_size = 128;
        p->block_size = 32;
        p->dtype = "fp16";
        p->num_query_heads = 1;
        p->num_kv_heads = 1;
        p->total_q = 1;
        p->max_seqlen_q = 1;
        p->max_seqlen_k = 16;
        p->use_sinks = true;
        p->softcap = 0.0;
        break;
    case 12:
        p->head_size = 256;
        p->block_size = 64;
        p->dtype = "bf16";
        p->num_query_heads = 2;
        p->num_kv_heads = 1;
        p->total_q = 1;
        p->max_seqlen_q = 1;
        p->max_seqlen_k = 1;
        p->use_sinks = false;
        p->softcap = 5.0;
        break;

    /* ---- prime dimensions ---- */
    case 13:
        p->head_size = 64;
        p->block_size = 16;
        p->dtype = "bf16";
        p->num_query_heads = 7;
        p->num_kv_heads = 7;
        p->total_q = 13;
        p->max_seqlen_q = 17;
        p->max_seqlen_k = 19;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;
    case 14:
        p->head_size = 128;
        p->block_size = 32;
        p->dtype = "fp16";
        p->num_query_heads = 11;
        p->num_kv_heads = 11;
        p->total_q = 101;
        p->max_seqlen_q = 103;
        p->max_seqlen_k = 107;
        p->use_sinks = true;
        p->softcap = 0.0;
        break;
    case 15:
        p->head_size = 64;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 13;
        p->num_kv_heads = 1;
        p->total_q = 251;
        p->max_seqlen_q = 257;
        p->max_seqlen_k = 263;
        p->use_sinks = false;
        p->sliding_window = 31;
        p->softcap = 0.0;
        break;

    /* ---- very large dimensions ---- */
    case 16:
        p->head_size = 256;
        p->block_size = 64;
        p->dtype = "bf16";
        p->num_query_heads = 128;
        p->num_kv_heads = 8;
        p->total_q = 65536;
        p->max_seqlen_q = 32768;
        p->max_seqlen_k = 131072;
        p->use_sinks = true;
        p->softcap = 0.0;
        break;
    case 17:
        p->head_size = 128;
        p->block_size = 16;
        p->dtype = "fp16";
        p->num_query_heads = 96;
        p->num_kv_heads = 12;
        p->total_q = 100000;
        p->max_seqlen_q = 8192;
        p->max_seqlen_k = 262144;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;

    /* ---- sliding_window variety ---- */
    case 18:
        p->head_size = 64;
        p->block_size = 16;
        p->dtype = "bf16";
        p->num_query_heads = 32;
        p->num_kv_heads = 4;
        p->total_q = 300;
        p->max_seqlen_q = 512;
        p->max_seqlen_k = 512;
        p->use_sinks = false;
        p->sliding_window = 1;
        p->softcap = 0.0;
        break;
    case 19:
        p->head_size = 128;
        p->block_size = 32;
        p->dtype = "fp16";
        p->num_query_heads = 64;
        p->num_kv_heads = 8;
        p->total_q = 300;
        p->max_seqlen_q = 512;
        p->max_seqlen_k = 512;
        p->use_sinks = true;
        p->sliding_window = 4096;
        p->softcap = 0.0;
        break;
    case 20:
        p->head_size = 256;
        p->block_size = 16;
        p->dtype = "bf16";
        p->num_query_heads = 16;
        p->num_kv_heads = 16;
        p->total_q = 128;
        p->max_seqlen_q = 256;
        p->max_seqlen_k = 256;
        p->use_sinks = false;
        p->sliding_window = 63;
        p->softcap = 0.0;
        break;

    /* ---- softcap boundary: exactly 0 vs tiny positive ---- */
    case 21:
        p->head_size = 64;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 8;
        p->num_kv_heads = 8;
        p->total_q = 64;
        p->max_seqlen_q = 128;
        p->max_seqlen_k = 128;
        p->use_sinks = false;
        p->softcap = 0.0001;
        break;
    case 22:
        p->head_size = 64;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 8;
        p->num_kv_heads = 8;
        p->total_q = 64;
        p->max_seqlen_q = 128;
        p->max_seqlen_k = 128;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;

    /* ---- GQA ratios / head divisibility extremes ---- */
    case 23:
        p->head_size = 128;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 128;
        p->num_kv_heads = 1;
        p->total_q = 512;
        p->max_seqlen_q = 1024;
        p->max_seqlen_k = 1024;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;
    case 24:
        p->head_size = 64;
        p->block_size = 16;
        p->dtype = "fp16";
        p->num_query_heads = 256;
        p->num_kv_heads = 64;
        p->total_q = 256;
        p->max_seqlen_q = 512;
        p->max_seqlen_k = 512;
        p->use_sinks = true;
        p->softcap = 0.0;
        break;

    /* ---- decode (q=1) with various head_size/dtype ---- */
    case 25:
        p->head_size = 64;
        p->block_size = 16;
        p->dtype = "fp16";
        p->num_query_heads = 32;
        p->num_kv_heads = 4;
        p->total_q = 1;
        p->max_seqlen_q = 1;
        p->max_seqlen_k = 8192;
        p->use_sinks = true;
        p->softcap = 0.0;
        break;
    case 26:
        p->head_size = 256;
        p->block_size = 32;
        p->dtype = "bf16";
        p->num_query_heads = 16;
        p->num_kv_heads = 2;
        p->total_q = 1;
        p->max_seqlen_q = 1;
        p->max_seqlen_k = 2048;
        p->use_sinks = false;
        p->softcap = 20.0;
        break;

    /* ---- num_seqs variety (runtime param, must NOT change IR) ---- */
    case 27:
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
        p->num_seqs = 16;
        break;

    /* ---- max_seqlen_q vs max_seqlen_k asymmetry ---- */
    case 28:
        p->head_size = 128;
        p->block_size = 64;
        p->dtype = "fp16";
        p->num_query_heads = 32;
        p->num_kv_heads = 8;
        p->total_q = 64;
        p->max_seqlen_q = 1;
        p->max_seqlen_k = 65535;
        p->use_sinks = false;
        p->softcap = 0.0;
        break;
    case 29:
        p->head_size = 64;
        p->block_size = 16;
        p->dtype = "bf16";
        p->num_query_heads = 8;
        p->num_kv_heads = 8;
        p->total_q = 2;
        p->max_seqlen_q = 2;
        p->max_seqlen_k = 3;
        p->use_sinks = true;
        p->sliding_window = 2;
        p->softcap = 7.0;
        break;

    /* ---- all features on at once ---- */
    case 30:
        p->head_size = 128;
        p->block_size = 16;
        p->dtype = "bf16";
        p->num_query_heads = 40;
        p->num_kv_heads = 8;
        p->total_q = 333;
        p->max_seqlen_q = 1000;
        p->max_seqlen_k = 2000;
        p->use_sinks = true;
        p->sliding_window = 256;
        p->softcap = 42.0;
        break;
    case 31:
        p->head_size = 256;
        p->block_size = 64;
        p->dtype = "fp16";
        p->num_query_heads = 64;
        p->num_kv_heads = 64;
        p->total_q = 997;
        p->max_seqlen_q = 1009;
        p->max_seqlen_k = 1013;
        p->use_sinks = true;
        p->sliding_window = 511;
        p->softcap = 99.0;
        break;

    default:
        return -1;
    }
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
        return 3;
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
            return 4;
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
