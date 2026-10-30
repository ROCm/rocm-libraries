// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only regression test for the three scalar-attention `_lower_to_llvm`
 * wrappers (2D, 3D, reduce).
 *
 * They are the only lowering entries whose signature carries no target, so each
 * lowers for a fixed one on the caller's behalf: the codegen default, gfx950, as
 * instance_attention_unified.h documents. The lowerer itself has no default and
 * rejects a NULL arch, so a wrapper that forwarded NULL would return
 * ROCKE_ERR_VALUE for every valid problem. Nothing else in the tree calls these
 * wrappers, so that regression was invisible until a reviewer read the diff.
 *
 * For each wrapper the lowering must succeed, and its text must be byte-identical
 * to lowering the same kernel through the target-taking entry for gfx950. These
 * kernels are arch-neutral (no MMA atom, no LDS, no cross-lane op), so the text
 * is the same on every gfx9 target: the comparison pins the wrapper's behavior,
 * not which gfx9 constant it names.
 *
 * It also pins the scalar kernel names (fp8 K/V and mask tokens) and the 2D
 * signature's fp8 K/V pointer type to the values the Python specs produce, and
 * mirrors test_attention_scalar_cache_key.py: the dynamic shape never reaches the
 * kernel name or IR, and the block-table row stride is a kernarg.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/instance_attention_unified.h"
#include "rocke/lower_llvm.h"

#define ERR_CAP 1024
#define NUM_SEGMENTS 8 /* the 3D kernel's Python default; the reduce kernel's input */

/* The target the wrappers' header documents. */
static const char WRAPPER_TARGET[] = "gfx950";

typedef enum
{
    KIND_2D,
    KIND_3D,
    KIND_REDUCE
} kind_t;

static const char* kind_name(kind_t kind)
{
    switch(kind)
    {
    case KIND_2D:
        return "2d_scalar";
    case KIND_3D:
        return "3d_scalar";
    case KIND_REDUCE:
        return "reduce_scalar";
    }
    return "?";
}

static rocke_unified_attention_problem_t make_problem(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    p.total_q = 4;
    p.num_seqs = 2;
    p.num_query_heads = 8;
    p.num_kv_heads = 2;
    p.head_size = 64;
    p.block_size = 16;
    p.max_seqlen_q = 2;
    p.max_seqlen_k = 64;
    p.dtype = "fp16";
    return p;
}

/* The entry under test: no target in the signature. */
static rocke_status_t lower_through_wrapper(
    kind_t kind, const rocke_unified_attention_problem_t* p, char** ll, char* err, size_t err_cap)
{
    switch(kind)
    {
    case KIND_2D:
        return rocke_unified_attention_2d_scalar_lower_to_llvm(
            p, NULL, ROCKE_LLVM_FLAVOR_AUTO, ll, err, err_cap);
    case KIND_3D:
        return rocke_unified_attention_3d_scalar_lower_to_llvm(
            p, NULL, NUM_SEGMENTS, ROCKE_LLVM_FLAVOR_AUTO, ll, err, err_cap);
    case KIND_REDUCE:
        return rocke_unified_attention_reduce_scalar_lower_to_llvm(
            p, NUM_SEGMENTS, NULL, ROCKE_LLVM_FLAVOR_AUTO, ll, err, err_cap);
    }
    return ROCKE_ERR_VALUE;
}

/* The reference: the same kernel, lowered for a target named by the caller. */
static rocke_status_t lower_for_target(kind_t kind,
                                       const rocke_unified_attention_problem_t* p,
                                       const char* arch,
                                       char** ll,
                                       char* err,
                                       size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = NULL;
    rocke_status_t st;

    switch(kind)
    {
    case KIND_2D:
        k = rocke_build_unified_attention_2d_scalar_new(&b, p, NULL);
        break;
    case KIND_3D:
        k = rocke_build_unified_attention_3d_scalar_new(&b, p, NULL, NUM_SEGMENTS);
        break;
    case KIND_REDUCE:
        k = rocke_build_unified_attention_reduce_scalar_new(&b, p, NUM_SEGMENTS, NULL);
        break;
    }
    if(k == NULL)
    {
        snprintf(err, err_cap, "build failed: %s", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return ROCKE_ERR_VALUE;
    }
    st = rocke_lower_kernel_to_llvm_ex(k, ROCKE_LLVM_FLAVOR_AUTO, arch, ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

static int run_case(kind_t kind)
{
    const rocke_unified_attention_problem_t p = make_problem();
    char* via_wrapper = NULL;
    char* via_target = NULL;
    char err[ERR_CAP];
    rocke_status_t st;
    int rc = 1;

    err[0] = '\0';
    st = lower_through_wrapper(kind, &p, &via_wrapper, err, sizeof err);
    if(st != ROCKE_OK || via_wrapper == NULL || via_wrapper[0] == '\0')
    {
        fprintf(stderr,
                "%-14s wrapper failed: status=%d err=%s\n",
                kind_name(kind),
                (int)st,
                err[0] != '\0' ? err : "<none>");
        goto done;
    }

    err[0] = '\0';
    st = lower_for_target(kind, &p, WRAPPER_TARGET, &via_target, err, sizeof err);
    if(st != ROCKE_OK || via_target == NULL)
    {
        fprintf(stderr,
                "%-14s reference lowering for %s failed: status=%d err=%s\n",
                kind_name(kind),
                WRAPPER_TARGET,
                (int)st,
                err[0] != '\0' ? err : "<none>");
        goto done;
    }

    if(strcmp(via_wrapper, via_target) != 0)
    {
        fprintf(stderr,
                "%-14s wrapper text (%zuB) differs from lowering for %s (%zuB)\n",
                kind_name(kind),
                strlen(via_wrapper),
                WRAPPER_TARGET,
                strlen(via_target));
        goto done;
    }

    printf("%-14s -> OK ll=%zuB, identical to lowering for %s\n",
           kind_name(kind),
           strlen(via_wrapper),
           WRAPPER_TARGET);
    rc = 0;
done:
    free(via_wrapper);
    free(via_target);
    return rc;
}

/* Kernel names and the fp8 K/V signature, from the Python specs for the same
 * problem (make_problem() plus the listed fields). */
static int check_names_and_signature(void)
{
    static const struct
    {
        bool use_fp8, top_left;
        int sliding_window, right_bound;
        const char* name_2d;
        const char* name_3d;
    } rows[] = {
        {true,
         false,
         0,
         0,
         "rocke_unified_attention_2d_scalar_h8_kv2_d64_b16_fp16_fp8kv",
         "rocke_unified_attention_3d_scalar_h8_kv2_d64_b16_seg8_fp16"},
        {false,
         true,
         64,
         5,
         "rocke_unified_attention_2d_scalar_h8_kv2_d64_b16_fp16_sw_tl_rb5",
         "rocke_unified_attention_3d_scalar_h8_kv2_d64_b16_seg8_fp16_tl_rb5_sw"},
        {false,
         false,
         0,
         -1,
         "rocke_unified_attention_2d_scalar_h8_kv2_d64_b16_fp16_rbu",
         "rocke_unified_attention_3d_scalar_h8_kv2_d64_b16_seg8_fp16_rbu"},
        /* The default-mask 3D kernel never applies the window: no "sw". */
        {false,
         false,
         64,
         0,
         "rocke_unified_attention_2d_scalar_h8_kv2_d64_b16_fp16_sw",
         "rocke_unified_attention_3d_scalar_h8_kv2_d64_b16_seg8_fp16"},
    };
    int failures = 0;
    size_t i;

    for(i = 0; i < sizeof rows / sizeof rows[0]; ++i)
    {
        rocke_unified_attention_problem_t p = make_problem();
        char name[256];
        p.use_fp8 = rows[i].use_fp8;
        p.causal_top_left = rows[i].top_left;
        p.sliding_window = rows[i].sliding_window;
        p.right_bound = rows[i].right_bound;
        if(rocke_unified_attention_2d_scalar_kernel_name(&p, NULL, name, sizeof name) != ROCKE_OK
           || strcmp(name, rows[i].name_2d) != 0)
        {
            fprintf(stderr, "2d name: got %s, want %s\n", name, rows[i].name_2d);
            ++failures;
        }
        if(rocke_unified_attention_3d_scalar_kernel_name(&p, NULL, NUM_SEGMENTS, name, sizeof name)
               != ROCKE_OK
           || strcmp(name, rows[i].name_3d) != 0)
        {
            fprintf(stderr, "3d name: got %s, want %s\n", name, rows[i].name_3d);
            ++failures;
        }
    }

    for(i = 0; i < 2; ++i)
    {
        rocke_unified_attention_problem_t p = make_problem();
        const char* want = i == 0 ? "ptr<f16, global>" : "ptr<fp8e4m3, global>";
        rocke_arena_t arena;
        const rocke_sig_entry_t* items = NULL;
        size_t n = 0;
        p.use_fp8 = i == 1;
        rocke_arena_init(&arena, 4096);
        if(rocke_unified_attention_2d_scalar_signature(&p, &arena, &items, &n) != ROCKE_OK || n < 4
           || strcmp(items[2].name, "key_cache_ptr") != 0 || strcmp(items[2].type, want) != 0
           || strcmp(items[3].name, "value_cache_ptr") != 0 || strcmp(items[3].type, want) != 0)
        {
            fprintf(
                stderr, "2d signature (use_fp8=%d): K/V pointer type is not %s\n", (int)i, want);
            ++failures;
        }
        rocke_arena_destroy(&arena);
    }
    return failures;
}

/* The shape test_attention_scalar_cache_key.py uses for the Python side. */
static rocke_unified_attention_problem_t make_cache_key_base(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    p.total_q = 64;
    p.num_seqs = 4;
    p.num_query_heads = 8;
    p.num_kv_heads = 2;
    p.head_size = 128;
    p.block_size = 16;
    p.max_seqlen_q = 16;
    p.max_seqlen_k = 256;
    p.dtype = "bf16";
    return p;
}

static rocke_status_t
    kernel_name_for(kind_t kind, const rocke_unified_attention_problem_t* p, char* name, size_t cap)
{
    switch(kind)
    {
    case KIND_2D:
        return rocke_unified_attention_2d_scalar_kernel_name(p, NULL, name, cap);
    case KIND_3D:
        return rocke_unified_attention_3d_scalar_kernel_name(p, NULL, NUM_SEGMENTS, name, cap);
    case KIND_REDUCE:
        return rocke_unified_attention_reduce_scalar_kernel_name(p, NULL, NUM_SEGMENTS, name, cap);
    }
    return ROCKE_ERR_VALUE;
}

/* Name and .ll of one scalar kernel; the caller frees *ll. */
static int name_and_ll(
    kind_t kind, const rocke_unified_attention_problem_t* p, char* name, size_t cap, char** ll)
{
    char err[ERR_CAP];
    err[0] = '\0';
    *ll = NULL;
    if(kernel_name_for(kind, p, name, cap) != ROCKE_OK)
    {
        fprintf(stderr, "%s: kernel name failed\n", kind_name(kind));
        return 1;
    }
    if(lower_for_target(kind, p, WRAPPER_TARGET, ll, err, sizeof err) != ROCKE_OK || *ll == NULL)
    {
        fprintf(stderr, "%s: lowering failed: %s\n", kind_name(kind), err);
        free(*ll);
        *ll = NULL;
        return 1;
    }
    return 0;
}

/* The scalar kernels read total_q / num_seqs / max_seqlen_q / max_seqlen_k only
 * through kernargs and the grid, and the block-table row stride is the
 * block_table_stride kernarg. So shapes that differ only there must share one
 * kernel name (the binary's cache identity) and lower to the same bytes, while
 * every static field the build reads must split the name. */
static int check_dynamic_shape_shares_binary(void)
{
    static const kind_t kinds[] = {KIND_2D, KIND_3D, KIND_REDUCE};
    static const struct
    {
        int total_q, num_seqs, max_seqlen_q, max_seqlen_k;
    } dynamic[] = {
        {200, 4, 16, 256},
        {16, 1, 16, 256},
        {37, 37, 1, 256},
        {64, 4, 50, 256},
        {64, 4, 16, 250},
        {64, 4, 16, 1000},
        {1, 1, 1, 65536},
    };
    int failures = 0;
    size_t k, i;

    for(k = 0; k < sizeof kinds / sizeof kinds[0]; ++k)
    {
        const rocke_unified_attention_problem_t base = make_cache_key_base();
        char base_name[256];
        char* base_ll = NULL;
        if(name_and_ll(kinds[k], &base, base_name, sizeof base_name, &base_ll) != 0)
        {
            ++failures;
            continue;
        }
        if(kinds[k] != KIND_REDUCE
           && (strstr(base_ll, "i32 %block_table_stride)") == NULL
               || strstr(base_ll, ", %block_table_stride\n") == NULL))
        {
            fprintf(stderr, "%s: block_table_stride is not a kernarg\n", kind_name(kinds[k]));
            ++failures;
        }
        for(i = 0; i < sizeof dynamic / sizeof dynamic[0]; ++i)
        {
            rocke_unified_attention_problem_t p = base;
            char name[256];
            char* ll = NULL;
            p.total_q = dynamic[i].total_q;
            p.num_seqs = dynamic[i].num_seqs;
            p.max_seqlen_q = dynamic[i].max_seqlen_q;
            p.max_seqlen_k = dynamic[i].max_seqlen_k;
            if(name_and_ll(kinds[k], &p, name, sizeof name, &ll) != 0)
            {
                ++failures;
                continue;
            }
            if(strcmp(name, base_name) != 0 || strcmp(ll, base_ll) != 0)
            {
                fprintf(stderr,
                        "%s: dynamic variant %zu changes the %s\n",
                        kind_name(kinds[k]),
                        i,
                        strcmp(name, base_name) != 0 ? "kernel name" : "IR");
                ++failures;
            }
            free(ll);
        }
        free(base_ll);
    }

    for(i = 0; i < 11; ++i)
    {
        rocke_unified_attention_problem_t base = make_cache_key_base();
        rocke_unified_attention_problem_t p = base;
        char base_name[256], name[256];
        char *base_ll = NULL, *ll = NULL;
        switch(i)
        {
        case 0:
            p.num_query_heads = 16;
            break;
        case 1:
            p.num_kv_heads = 8;
            break;
        case 2:
            p.head_size = 64;
            break;
        case 3:
            p.block_size = 32;
            break;
        case 4:
            p.dtype = "fp16";
            break;
        case 5:
            p.sliding_window = 128;
            break;
        case 6:
            p.use_sinks = true;
            break;
        case 7:
            p.softcap = 30.0;
            break;
        case 8:
            p.use_fp8 = true;
            break;
        case 9:
            p.causal_top_left = true;
            break;
        default:
            p.right_bound = 8;
            break;
        }
        if(name_and_ll(KIND_2D, &base, base_name, sizeof base_name, &base_ll) != 0
           || name_and_ll(KIND_2D, &p, name, sizeof name, &ll) != 0)
        {
            ++failures;
        }
        else if(strcmp(name, base_name) == 0 || strcmp(ll, base_ll) == 0)
        {
            fprintf(stderr, "2d_scalar: static variant %zu does not split name and IR\n", i);
            ++failures;
        }
        free(base_ll);
        free(ll);
    }
    return failures;
}

int main(void)
{
    static const kind_t kinds[] = {KIND_2D, KIND_3D, KIND_REDUCE};
    int failures = 0;
    size_t i;

    for(i = 0; i < sizeof kinds / sizeof kinds[0]; ++i)
        failures += run_case(kinds[i]);
    failures += check_names_and_signature();
    failures += check_dynamic_shape_shares_binary();

    if(failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
