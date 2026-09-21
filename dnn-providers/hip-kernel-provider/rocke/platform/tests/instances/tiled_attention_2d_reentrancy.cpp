// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only re-entrancy regression test for the CK DSL C tiled-attention
 * builders. Each lower_to_llvm call owns and frees an IRBuilder; subsequent
 * calls in this process must not reuse static pointers allocated from a
 * previous builder arena.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/lower_llvm.h"

typedef rocke_status_t (*lower_to_llvm_fn)(const rocke_attention_tiled_2d_spec_t* spec,
                                           const char* arch,
                                           rocke_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap);

typedef void (*fill_spec_fn)(rocke_attention_tiled_2d_spec_t* spec);

typedef struct tiled_reentrancy_case
{
    const char* name;
    const char* arch;
    lower_to_llvm_fn lower;
    fill_spec_fn fill;
} tiled_reentrancy_case_t;

static void fill_required(rocke_attention_tiled_2d_spec_t* s,
                          int head_size,
                          int block_size,
                          int num_query_heads,
                          int num_kv_heads,
                          const char* dtype)
{
    *s = rocke_attention_tiled_2d_spec_default();
    s->head_size = head_size;
    s->block_size = block_size;
    s->num_query_heads = num_query_heads;
    s->num_kv_heads = num_kv_heads;
    s->dtype = dtype;
    s->use_sinks = false;
    s->sliding_window = 0;
    s->has_softcap = false;
}

static void fill_gfx950_c32_dist(rocke_attention_tiled_2d_spec_t* s)
{
    fill_required(s, 64, 32, 64, 8, "bf16");
    s->num_warps = 4;
    s->block_m_per_warp = 32;
    s->has_tile_size = true;
    s->tile_size = 64;
    s->use_mfma_32x32 = true;
    s->use_transposed_qk_32x32 = true;
}

static void fill_gfx950_register_pv(rocke_attention_tiled_2d_spec_t* s)
{
    fill_required(s, 64, 32, 32, 32, "bf16");
    s->use_register_pv = true;
}

static void fill_gfx950_register_pv_wide(rocke_attention_tiled_2d_spec_t* s)
{
    fill_required(s, 64, 64, 32, 32, "bf16");
    s->use_register_pv = true;
}

static void fill_gfx942_c32_dist(rocke_attention_tiled_2d_spec_t* s)
{
    fill_required(s, 64, 32, 32, 32, "fp16");
    s->block_m_per_warp = 32;
    s->has_tile_size = true;
    s->tile_size = 64;
    s->use_mfma_32x32x8 = true;
    s->use_transposed_qk_32x32 = true;
}

static void fill_gfx942_register_pv(rocke_attention_tiled_2d_spec_t* s)
{
    fill_required(s, 64, 32, 32, 32, "bf16");
    s->use_register_pv = true;
}

static void fill_gfx942_register_pv_wide(rocke_attention_tiled_2d_spec_t* s)
{
    fill_required(s, 64, 64, 32, 32, "bf16");
    s->use_register_pv = true;
}

static int run_case(const tiled_reentrancy_case_t* c, size_t index, size_t count)
{
    rocke_attention_tiled_2d_spec_t spec;
    char* llvm_text = NULL;
    char err[1024];
    rocke_status_t st;

    memset(err, 0, sizeof(err));
    c->fill(&spec);
    st = c->lower(&spec, c->arch, ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof(err));
    if(st != ROCKE_OK || llvm_text == NULL || llvm_text[0] == '\0')
    {
        fprintf(stderr,
                "call %zu/%zu %-24s arch=%s failed: status=%d err=%s\n",
                index + 1,
                count,
                c->name,
                c->arch,
                (int)st,
                err[0] != '\0' ? err : "<none>");
        free(llvm_text);
        return 1;
    }

    printf("call %zu/%zu %-24s arch=%s -> OK ll=%zuB\n",
           index + 1,
           count,
           c->name,
           c->arch,
           strlen(llvm_text));
    free(llvm_text);
    return 0;
}

int main(void)
{
    static const tiled_reentrancy_case_t cases[] = {
        {"gfx950_c32_dist",
         "gfx950",
         rocke_gfx950_attention_tiled_2d_lower_to_llvm,
         fill_gfx950_c32_dist},
        {"gfx950_c32_dist_repeat",
         "gfx950",
         rocke_gfx950_attention_tiled_2d_lower_to_llvm,
         fill_gfx950_c32_dist},
        {"gfx950_register_pv_wide",
         "gfx950",
         rocke_gfx950_attention_tiled_2d_lower_to_llvm,
         fill_gfx950_register_pv_wide},
        {"gfx950_register_pv",
         "gfx950",
         rocke_gfx950_attention_tiled_2d_lower_to_llvm,
         fill_gfx950_register_pv},
        {"gfx950_register_pv_repeat",
         "gfx950",
         rocke_gfx950_attention_tiled_2d_lower_to_llvm,
         fill_gfx950_register_pv},
        {"gfx942_c32_dist",
         "gfx942",
         rocke_gfx942_attention_tiled_2d_lower_to_llvm,
         fill_gfx942_c32_dist},
        {"gfx942_c32_dist_repeat",
         "gfx942",
         rocke_gfx942_attention_tiled_2d_lower_to_llvm,
         fill_gfx942_c32_dist},
        {"gfx942_register_pv_wide",
         "gfx942",
         rocke_gfx942_attention_tiled_2d_lower_to_llvm,
         fill_gfx942_register_pv_wide},
        {"gfx942_register_pv",
         "gfx942",
         rocke_gfx942_attention_tiled_2d_lower_to_llvm,
         fill_gfx942_register_pv},
        {"gfx942_register_pv_repeat",
         "gfx942",
         rocke_gfx942_attention_tiled_2d_lower_to_llvm,
         fill_gfx942_register_pv},
    };
    const size_t count = sizeof(cases) / sizeof(cases[0]);
    size_t i;

    for(i = 0; i < count; ++i)
    {
        if(run_case(&cases[i], i, count) != 0)
        {
            return 1;
        }
    }

    printf("ALL %zu TILED ATTENTION RE-ENTRANCY CALLS SURVIVED\n", count);
    return 0;
}
