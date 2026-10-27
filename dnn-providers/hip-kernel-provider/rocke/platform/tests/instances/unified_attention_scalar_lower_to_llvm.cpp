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
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(void)
{
    static const kind_t kinds[] = {KIND_2D, KIND_3D, KIND_REDUCE};
    int failures = 0;
    size_t i;

    for(i = 0; i < sizeof kinds / sizeof kinds[0]; ++i)
        failures += run_case(kinds[i]);

    if(failures != 0)
    {
        fprintf(stderr, "%d of %zu wrappers failed\n", failures, sizeof kinds / sizeof kinds[0]);
        return 1;
    }
    return 0;
}
