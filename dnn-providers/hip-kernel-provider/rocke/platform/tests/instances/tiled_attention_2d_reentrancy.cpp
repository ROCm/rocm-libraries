// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only re-entrancy regression test for the CK DSL C tiled-attention
 * builders. Each lower_to_llvm call owns and frees an IRBuilder; subsequent
 * calls in this process must not reuse static pointers allocated from a
 * previous builder arena.
 *
 * A stale static can corrupt the IR without crashing, so each repeat call's
 * LLVM text must equal the text the first call with the same builder and spec
 * returned, byte for byte.
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

/* The index of the first case before @p index that calls the same builder on
 * the same spec, or @p index when there is none. */
static size_t first_call_of(const tiled_reentrancy_case_t* cases, size_t index)
{
    size_t earlier;

    for(earlier = 0; earlier < index; ++earlier)
    {
        if(cases[earlier].lower == cases[index].lower && cases[earlier].fill == cases[index].fill)
        {
            return earlier;
        }
    }
    return index;
}

/* Lowers case @p index into texts[index]. A repeat of an earlier case must
 * return exactly the text that case returned. */
static int run_case(const tiled_reentrancy_case_t* cases, char** texts, size_t index, size_t count)
{
    const tiled_reentrancy_case_t* c = &cases[index];
    rocke_attention_tiled_2d_spec_t spec;
    char err[1024];
    rocke_status_t st;
    size_t first;

    memset(err, 0, sizeof(err));
    c->fill(&spec);
    st = c->lower(&spec, c->arch, ROCKE_LLVM_FLAVOR_AUTO, &texts[index], err, sizeof(err));
    if(st != ROCKE_OK || texts[index] == NULL || texts[index][0] == '\0')
    {
        fprintf(stderr,
                "call %zu/%zu %-24s arch=%s failed: status=%d err=%s\n",
                index + 1,
                count,
                c->name,
                c->arch,
                (int)st,
                err[0] != '\0' ? err : "<none>");
        return 1;
    }

    first = first_call_of(cases, index);
    if(first != index && strcmp(texts[index], texts[first]) != 0)
    {
        fprintf(stderr,
                "call %zu/%zu %-24s arch=%s returned %zuB of LLVM text that differs from "
                "the %zuB call %zu (%s) returned for the same builder and spec\n",
                index + 1,
                count,
                c->name,
                c->arch,
                strlen(texts[index]),
                strlen(texts[first]),
                first + 1,
                cases[first].name);
        return 1;
    }

    printf("call %zu/%zu %-24s arch=%s -> OK ll=%zuB%s\n",
           index + 1,
           count,
           c->name,
           c->arch,
           strlen(texts[index]),
           first != index ? " (same text as its first call)" : "");
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
    char* texts[sizeof(cases) / sizeof(cases[0])] = {NULL};
    size_t i;
    int failed = 0;

    for(i = 0; i < count && !failed; ++i)
    {
        failed = run_case(cases, texts, i, count);
    }
    for(i = 0; i < count; ++i)
    {
        free(texts[i]);
    }
    if(failed)
    {
        return 1;
    }

    printf("ALL %zu TILED ATTENTION RE-ENTRANCY CALLS SURVIVED\n", count);
    return 0;
}
