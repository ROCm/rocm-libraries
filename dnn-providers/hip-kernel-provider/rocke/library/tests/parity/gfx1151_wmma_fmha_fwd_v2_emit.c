/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1151_wmma_fmha_fwd_v2_emit.c -- C-side emitter for the RDNA
 * (gfx11 / gfx12) multi-wave WMMA attention prefill parity harness. The
 * config table is built by the same loops, in the same order, as
 * gfx1151_wmma_fmha_fwd_v2_emit.py (see its header for the index map).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1151_wmma_fmha_fwd_v2.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define MAX_CONFIGS 256

typedef struct
{
    const char* arch;
    rocke_wmma_fmha_fwd_v2_spec_t spec;
} config_t;

static config_t g_cfgs[MAX_CONFIGS];
static int g_n;

static rocke_wmma_fmha_fwd_v2_spec_t
    base_spec(const char* dtype, int hd, int hdv, const char* mask, const char* bias)
{
    rocke_wmma_fmha_fwd_v2_spec_t s = rocke_wmma_fmha_fwd_v2_spec_default();
    s.dtype = dtype;
    s.head_dim = hd;
    s.head_dim_v = hdv;
    s.mask = mask;
    s.bias = bias;
    return s;
}

static void push(const char* arch, rocke_wmma_fmha_fwd_v2_spec_t s)
{
    g_cfgs[g_n].arch = arch;
    g_cfgs[g_n].spec = s;
    ++g_n;
}

/* FEATURES rows of the Python emitter; flags: c softcap, s sinks, a ALiBi,
 * q QQ-bias, 8 FP8 (OCP E4M3FN) K/V; pair -1 = None. */
typedef struct
{
    const char* dtype;
    int hd, hdv;
    const char* mask;
    const char* bias;
    const char* flags;
    const char* layout;
    int ps, bn, pair;
} feature_row_t;

static const feature_row_t k_features[] = {
    {"fp16", 128, 128, "none", NULL, "c", "dense", 0, 32, -1},
    {"bf16", 256, 256, "causal", NULL, "c", "dense", 0, 32, -1},
    {"bf16", 64, 64, "causal", NULL, "s", "dense", 0, 32, -1},
    {"fp16", 128, 128, "none", NULL, "s", "dense", 0, 32, -1},
    {"bf16", 128, 128, "causal", NULL, "a", "dense", 0, 32, -1},
    {"fp16", 128, 128, "none", NULL, "a", "dense", 0, 32, -1},
    {"fp16", 128, 128, "causal", NULL, "q", "dense", 0, 32, -1},
    {"bf16", 64, 64, "none", NULL, "q", "dense", 0, 32, -1},
    {"bf16", 128, 128, "none", NULL, "8", "dense", 0, 32, -1},
    {"fp16", 64, 64, "causal", NULL, "8", "dense", 0, 32, -1},
    {"bf16", 256, 256, "causal", NULL, "8", "dense", 0, 32, -1},
    {"bf16", 128, 128, "causal", NULL, "", "ragged", 0, 32, -1},
    {"fp16", 64, 64, "none", NULL, "", "ragged", 0, 32, -1},
    {"bf16", 128, 128, "causal", NULL, "", "paged", 16, 32, -1},
    {"bf16", 128, 128, "causal", NULL, "", "paged", 32, 32, -1},
    {"fp16", 64, 64, "none", NULL, "", "paged", 64, 32, -1},
    {"bf16", 64, 64, "causal", NULL, "", "paged", 8, 32, -1},
    {"bf16", 256, 256, "causal", NULL, "", "paged", 16, 32, -1},
    {"bf16", 64, 64, "causal", NULL, "s", "paged", 16, 32, -1},
    {"bf16", 128, 128, "causal", NULL, "8", "ragged", 0, 32, -1},
    {"fp16", 128, 128, "causal", NULL, "8", "paged", 32, 32, -1},
    {"bf16", 128, 128, "causal", NULL, "sa", "paged", 16, 32, -1},
    {"fp16", 128, 128, "causal", NULL, "q", "ragged", 0, 32, -1},
    {"bf16", 128, 128, "none", NULL, "ca", "dense", 0, 32, -1},
    {"fp16", 64, 64, "causal", "f32", "s", "dense", 0, 32, -1},
    {"bf16", 128, 128, "causal", "q", "csaq8", "paged", 16, 32, -1},
    {"bf16", 256, 256, "causal", NULL, "cs", "dense", 0, 32, -1},
    {"bf16", 64, 64, "causal", NULL, "", "paged", 32, 64, -1},
    {"bf16", 64, 64, "causal", NULL, "s", "paged", 64, 64, -1},
    {"bf16", 128, 128, "causal", NULL, "8", "ragged", 0, 16, -1},
};

static const feature_row_t k_features_gfx12[] = {
    {"bf16", 192, 128, "causal", NULL, "c8", "dense", 0, 32, 0},
    {"fp16", 256, 128, "none", NULL, "s", "paged", 32, 32, 0},
};

static rocke_wmma_fmha_fwd_v2_spec_t feature_spec(const feature_row_t* r)
{
    rocke_wmma_fmha_fwd_v2_spec_t s = base_spec(r->dtype, r->hd, r->hdv, r->mask, r->bias);
    s.use_softcap = strchr(r->flags, 'c') != NULL;
    s.use_sinks = strchr(r->flags, 's') != NULL;
    s.use_alibi = strchr(r->flags, 'a') != NULL;
    s.use_qq_bias = strchr(r->flags, 'q') != NULL;
    s.layout = r->layout;
    s.page_block_size = r->ps;
    s.kv_dtype = strchr(r->flags, '8') != NULL ? "fp8e4m3" : "";
    s.block_n = r->bn;
    s.wave_pair = r->pair;
    return s;
}

static void build_table(void)
{
    static const char* archs[] = {"gfx11-generic", "gfx12-generic"};
    static const char* dtypes[] = {"fp16", "bf16"};
    static const int buckets[][2]
        = {{64, 64}, {96, 96}, {128, 128}, {192, 128}, {256, 128}, {256, 256}};
    static const char* masks[] = {"none", "causal"};
    static const struct
    {
        int hd, hdv;
        const char* mask;
    } bias_shapes[] = {{64, 64, "causal"}, {128, 128, "none"}, {256, 256, "causal"}};
    static const char* biases[] = {"f32", "q"};
    int a, d, k, m, i;
    rocke_wmma_fmha_fwd_v2_spec_t s;

    g_n = 0;
    for(a = 0; a < 2; ++a)
        for(d = 0; d < 2; ++d)
            for(k = 0; k < 6; ++k)
                for(m = 0; m < 2; ++m)
                    push(archs[a],
                         base_spec(dtypes[d], buckets[k][0], buckets[k][1], masks[m], NULL));
    for(a = 0; a < 2; ++a)
        for(d = 0; d < 2; ++d)
            for(k = 0; k < 3; ++k)
                for(i = 0; i < 2; ++i)
                    push(archs[a],
                         base_spec(dtypes[d],
                                   bias_shapes[k].hd,
                                   bias_shapes[k].hdv,
                                   bias_shapes[k].mask,
                                   biases[i]));
    for(d = 0; d < 2; ++d)
    {
        static const int hds[] = {192, 256};
        for(k = 0; k < 2; ++k)
            for(m = 0; m < 2; ++m)
            {
                s = base_spec(dtypes[d], hds[k], 128, masks[m], NULL);
                s.wave_pair = 0;
                push("gfx12-generic", s);
            }
    }
    s = base_spec("bf16", 256, 128, "causal", "q");
    s.wave_pair = 0;
    push("gfx12-generic", s);
    s = base_spec("fp16", 192, 128, "none", "f32");
    s.wave_pair = 0;
    push("gfx12-generic", s);
    for(a = 0; a < 2; ++a)
    {
        for(k = 0; k < 10; ++k)
        {
            s = base_spec("bf16", 128, 128, "causal", NULL);
            switch(k)
            {
            case 0:
                s.head_dim = 64;
                s.head_dim_v = 64;
                s.block_n = 64;
                break;
            case 1:
                s.block_n = 16;
                break;
            case 2:
                s.block_m = 64;
                break;
            case 3:
                s.waves_per_eu = 0;
                break;
            case 4:
                s.scheduler_strategy = "max-ilp";
                break;
            case 5:
                s.lazy_rescale = false;
                break;
            case 6:
                s.k_pad = 0;
                s.v_pad = 16;
                break;
            case 7:
                s.cta_order = "hbq";
                break;
            case 8:
                s.wave_pair = 1;
                break;
            default:
                s.head_dim = 256;
                s.head_dim_v = 256;
                s.block_m = 32;
                break;
            }
            push(archs[a], s);
        }
    }
    push("gfx1151", base_spec("fp16", 64, 64, "none", NULL));
    push("gfx1151", base_spec("bf16", 128, 128, "causal", "q"));
    push("gfx1201", base_spec("fp16", 64, 64, "none", NULL));
    s = base_spec("bf16", 256, 128, "causal", NULL);
    s.wave_pair = 0;
    push("gfx1201", s);
    for(a = 0; a < 2; ++a)
        for(i = 0; i < (int)(sizeof(k_features) / sizeof(k_features[0])); ++i)
            push(archs[a], feature_spec(&k_features[i]));
    for(i = 0; i < (int)(sizeof(k_features_gfx12) / sizeof(k_features_gfx12[0])); ++i)
        push("gfx12-generic", feature_spec(&k_features_gfx12[i]));
    push("gfx1151", feature_spec(&k_features[2]));
    push("gfx1201", feature_spec(&k_features[8]));
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
    const rocke_wmma_fmha_fwd_v2_spec_t* spec = &g_cfgs[idx].spec;
    const char* arch = g_cfgs[idx].arch;

    char reason[256];
    reason[0] = 0;
    if(!rocke_wmma_fmha_fwd_v2_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }
    char name[256];
    if(rocke_wmma_fmha_fwd_v2_kernel_name(spec, name, sizeof name) != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }
    rocke_kernel_def_t* kernel = rocke_build_wmma_fmha_fwd_v2(&b, spec, arch);
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
