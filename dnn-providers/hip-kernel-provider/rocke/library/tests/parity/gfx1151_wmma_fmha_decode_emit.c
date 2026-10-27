/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1151_wmma_fmha_decode_emit.c -- C-side emitter for the RDNA
 * (gfx11 / gfx12) split-KV WMMA decode + merge parity harness. The config
 * table is built by the same loops, in the same order, as
 * gfx1151_wmma_fmha_decode_emit.py (see its header for the index map).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1151_wmma_fmha_decode.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define MAX_CONFIGS 256

typedef struct
{
    const char* arch;
    int merge;
    rocke_wmma_fmha_decode_spec_t spec;
} config_t;

static config_t g_cfgs[MAX_CONFIGS];
static int g_n;

static rocke_wmma_fmha_decode_spec_t base_spec(int hs, const char* dtype, const char* bias)
{
    rocke_wmma_fmha_decode_spec_t s = rocke_wmma_fmha_decode_spec_default();
    s.head_size = hs;
    s.dtype = dtype;
    s.bias = bias;
    return s;
}

static void push(const char* arch, int merge, rocke_wmma_fmha_decode_spec_t s)
{
    g_cfgs[g_n].arch = arch;
    g_cfgs[g_n].merge = merge;
    g_cfgs[g_n].spec = s;
    ++g_n;
}

/* FEATURES rows of the Python emitter; flags: c softcap, s sinks, a ALiBi,
 * q QQ-bias, 8 FP8 (OCP E4M3FN) K/V. */
typedef struct
{
    const char* dtype;
    int hs;
    const char* bias;
    const char* flags;
    const char* layout;
    int ps;
} feature_row_t;

static const feature_row_t k_features[] = {
    {"fp16", 128, "", "c", "dense", 0},      {"bf16", 256, "", "c", "dense", 0},
    {"bf16", 64, "", "s", "dense", 0},       {"fp16", 128, "", "a", "dense", 0},
    {"bf16", 64, "", "q", "dense", 0},       {"bf16", 64, "", "8", "dense", 0},
    {"fp16", 128, "", "8", "dense", 0},      {"bf16", 256, "", "8", "dense", 0},
    {"bf16", 128, "", "", "ragged", 0},      {"fp16", 64, "", "", "paged", 16},
    {"bf16", 128, "", "", "paged", 8},       {"bf16", 256, "", "", "paged", 64},
    {"bf16", 64, "", "s8", "paged", 16},     {"fp16", 128, "", "ca", "dense", 0},
    {"bf16", 128, "", "aq", "ragged", 0},    {"bf16", 256, "", "cs8", "ragged", 0},
    {"bf16", 64, "q", "csaq8", "paged", 16}, {"fp16", 128, "f32", "s", "ragged", 0},
    {"bf16", 64, "f32", "ac8", "paged", 32}, {"fp16", 256, "q", "q", "paged", 4},
};

static rocke_wmma_fmha_decode_spec_t feature_spec(const feature_row_t* r)
{
    rocke_wmma_fmha_decode_spec_t s = base_spec(r->hs, r->dtype, r->bias);
    s.softcap = strchr(r->flags, 'c') != NULL;
    s.sinks = strchr(r->flags, 's') != NULL;
    s.alibi = strchr(r->flags, 'a') != NULL;
    s.qq_bias = strchr(r->flags, 'q') != NULL;
    s.kv_dtype = strchr(r->flags, '8') != NULL ? "fp8e4m3" : "";
    s.layout = r->layout;
    s.page_block_size = r->ps;
    return s;
}

static rocke_wmma_fmha_decode_spec_t knob_spec(int k)
{
    rocke_wmma_fmha_decode_spec_t s = base_spec(128, "bf16", "");
    switch(k)
    {
    case 0:
        s.num_waves = 2;
        break;
    case 1:
        s.num_waves = 8;
        s.head_size = 64;
        break;
    case 2:
        s.num_waves = 1;
        break;
    case 3:
        s.waves_per_eu = 1;
        break;
    case 4:
        s.scheduler_strategy = "default";
        break;
    case 5:
        s.scheduler_strategy = "max-ilp";
        s.head_size = 256;
        break;
    case 6:
        s.dv_split = 2;
        break;
    case 7:
        s.q_mode = "lds";
        break;
    case 8:
        s.q_mode = "reg";
        s.head_size = 256;
        break;
    case 9:
        s.v_store_early = 1;
        break;
    case 10:
        s.v_store_early = 0;
        break;
    case 11:
        s.prefetch = 1;
        break;
    case 12:
        s.prefetch = 0;
        s.head_size = 64;
        break;
    case 13:
        s.lockstep = 1;
        break;
    case 14:
        s.lockstep = 0;
        break;
    case 15:
        s.fp8_cvt = 1;
        s.kv_dtype = "fp8e4m3";
        break;
    case 16:
        s.fp8_cvt = 0;
        s.kv_dtype = "fp8e4m3";
        s.head_size = 64;
        break;
    case 17:
        s.tanh_fast = 1;
        s.softcap = true;
        s.head_size = 256;
        break;
    default:
        s.tanh_fast = 0;
        s.softcap = true;
        break;
    }
    return s;
}

#define N_KNOBS 19

static void build_table(void)
{
    static const char* archs[] = {"gfx11-generic", "gfx12-generic"};
    static const char* dtypes[] = {"fp16", "bf16"};
    static const int buckets[] = {64, 128, 256};
    static const char* biases[] = {"", "q", "f32"};
    static const char* layouts[] = {"dense", "ragged"};
    int a, d, k, i;
    rocke_wmma_fmha_decode_spec_t s;

    g_n = 0;
    for(a = 0; a < 2; ++a)
        for(d = 0; d < 2; ++d)
            for(k = 0; k < 3; ++k)
                for(i = 0; i < 3; ++i)
                    push(archs[a], 0, base_spec(buckets[k], dtypes[d], biases[i]));
    for(a = 0; a < 2; ++a)
        for(d = 0; d < 2; ++d)
            for(i = 0; i < 2; ++i)
            {
                s = base_spec(64, dtypes[d], "");
                s.layout = layouts[i];
                push(archs[a], 1, s);
            }
    for(a = 0; a < 2; ++a)
        for(i = 0; i < (int)(sizeof(k_features) / sizeof(k_features[0])); ++i)
            push(archs[a], 0, feature_spec(&k_features[i]));
    for(a = 0; a < 2; ++a)
        for(k = 0; k < N_KNOBS; ++k)
            push(archs[a], 0, knob_spec(k));
    push("gfx1151", 0, base_spec(128, "bf16", ""));
    s = base_spec(64, "fp16", "");
    s.layout = "ragged";
    push("gfx1151", 1, s);
    push("gfx1151", 0, feature_spec(&k_features[12]));
    push("gfx1201", 0, base_spec(64, "fp16", "q"));
    push("gfx1201", 0, feature_spec(&k_features[15]));
    push("gfx1201", 1, base_spec(64, "bf16", ""));
}

int main(int argc, char** argv)
{
    build_table();
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..%d>\n", argv[0], g_n - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(idx < 0 || idx >= g_n)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const char* arch = g_cfgs[idx].arch;
    const int merge = g_cfgs[idx].merge;
    rocke_wmma_fmha_decode_spec_t spec = g_cfgs[idx].spec;

    char reason[256];
    reason[0] = 0;
    char name[512];
    if(merge)
    {
        if(rocke_wmma_fmha_decode_merge_kernel_name(&spec, name, sizeof name) != ROCKE_OK)
        {
            fprintf(stderr, "merge kernel_name failed\n");
            return 1;
        }
    }
    else
    {
        if(!rocke_wmma_fmha_decode_is_valid_spec(&spec, arch, reason, sizeof reason))
        {
            fprintf(stderr, "invalid spec: %s\n", reason);
            return 1;
        }
        if(rocke_wmma_fmha_decode_resolve(&spec, arch, &spec, reason, sizeof reason) != ROCKE_OK
           || rocke_wmma_fmha_decode_kernel_name(&spec, name, sizeof name) != ROCKE_OK)
        {
            fprintf(stderr, "kernel_name failed: %s\n", reason);
            return 1;
        }
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }
    rocke_kernel_def_t* kernel = merge ? rocke_build_wmma_fmha_decode_merge(&b, &spec, arch)
                                       : rocke_build_wmma_fmha_decode(&b, &spec, arch);
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
            ret = 1;
        }
        else
        {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(t, stdout);
            free(t);
        }
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
        ret = 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
