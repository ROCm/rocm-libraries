// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test for rocke_gfx942_attention_4warp_gqa_lower_to_llvm, the
 * convenience entry that owns its builder.
 *
 * Pinned here:
 *   1. a valid spec lowers to the same text as building with
 *      rocke_build_gfx942_4warp_gqa and lowering through the target-taking
 *      entry, and the caller owns the returned string;
 *   2. a refused spec returns the builder's status (VALUE / NOTIMPL) with its
 *      diagnostic and leaves *out_ll NULL;
 *   3. a NULL spec or out pointer is ROCKE_ERR_VALUE;
 *   4. a nonpositive block_size is rejected before any division by it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_4warp_gqa.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#define ERR_CAP 1024

static int g_failures = 0;

static void fail(const char* what, const char* detail)
{
    fprintf(stderr, "FAIL %s: %s\n", what, detail);
    ++g_failures;
}

static rocke_attention_tiled_2d_spec_t d256_spec(void)
{
    rocke_attention_tiled_2d_spec_t s = rocke_attention_tiled_2d_spec_default();
    s.use_sinks = false;
    s.has_softcap = false;
    s.num_warps = 1;
    s.block_m_per_warp = 32;
    s.has_waves_per_eu = true;
    s.waves_per_eu = 2;
    s.head_size = 256;
    s.block_size = 16;
    s.num_query_heads = 16;
    s.num_kv_heads = 2;
    s.dtype = "bf16";
    s.sliding_window = 0;
    s.num_seqs = 1;
    s.has_tile_size = true;
    s.tile_size = 32;
    return s;
}

static void test_lowers_like_the_builder(void)
{
    rocke_attention_tiled_2d_spec_t spec = d256_spec();
    char err[ERR_CAP] = "";
    char* got = NULL;
    char* want = NULL;
    rocke_ir_builder_t b;

    rocke_status_t st = rocke_gfx942_attention_4warp_gqa_lower_to_llvm(
        &spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &got, err, sizeof(err));
    if(st != ROCKE_OK || got == NULL || strstr(got, "_4wgqa") == NULL)
    {
        fail("lower/valid_spec", err);
        free(got);
        return;
    }

    rocke_kernel_def_t* kernel = rocke_build_gfx942_4warp_gqa(&b, &spec, "gfx942");
    if(kernel == NULL)
    {
        fail("lower/reference_build", rocke_ir_builder_error(&b));
    }
    else if(rocke_lower_kernel_to_llvm_ex(
                kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx942", &want, err, sizeof(err))
            != ROCKE_OK)
    {
        fail("lower/reference_lower", err);
    }
    else if(strcmp(got, want) != 0)
    {
        fail("lower/matches_builder", "entry text differs from builder + lower");
    }
    else
    {
        printf("ok   lower/matches_builder\n");
    }
    rocke_ir_builder_free(&b);
    free(want);
    free(got);

    /* NULL arch means gfx942. */
    got = NULL;
    st = rocke_gfx942_attention_4warp_gqa_lower_to_llvm(
        &spec, NULL, ROCKE_LLVM_FLAVOR_AUTO, &got, err, sizeof(err));
    if(st != ROCKE_OK || got == NULL)
        fail("lower/null_arch", err);
    else
        printf("ok   lower/null_arch\n");
    free(got);
}

static void expect_refused(const char* what,
                           const rocke_attention_tiled_2d_spec_t* spec,
                           rocke_status_t want,
                           const char* needle)
{
    char err[ERR_CAP] = "";
    char* ll = (char*)"sentinel";
    rocke_status_t st = rocke_gfx942_attention_4warp_gqa_lower_to_llvm(
        spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    if(st != want)
    {
        char detail[ERR_CAP + 64];
        snprintf(detail, sizeof(detail), "status %d, expected %d (%s)", (int)st, (int)want, err);
        fail(what, detail);
    }
    else if(ll != NULL)
    {
        fail(what, "*out_ll not reset to NULL on failure");
    }
    else if(strstr(err, needle) == NULL)
    {
        char detail[ERR_CAP + 64];
        snprintf(detail, sizeof(detail), "diagnostic \"%s\" lacks \"%s\"", err, needle);
        fail(what, detail);
    }
    else
    {
        printf("ok   %s\n", what);
    }
}

static void test_refusals(void)
{
    rocke_attention_tiled_2d_spec_t spec;
    char err[ERR_CAP] = "";
    char* ll = NULL;

    spec = d256_spec();
    spec.emit_lse = true;
    expect_refused("refuse/emit_lse", &spec, ROCKE_ERR_NOTIMPL, "LSE output");

    spec = d256_spec();
    spec.block_size = 0;
    expect_refused("refuse/zero_block_size", &spec, ROCKE_ERR_VALUE, "block_size must be positive");

    spec = d256_spec();
    spec.block_size = -16;
    expect_refused(
        "refuse/negative_block_size", &spec, ROCKE_ERR_VALUE, "block_size must be positive");

    if(rocke_gfx942_attention_4warp_gqa_lower_to_llvm(
           NULL, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err))
           != ROCKE_ERR_VALUE
       || ll != NULL)
        fail("refuse/null_spec", err);
    else
        printf("ok   refuse/null_spec\n");

    spec = d256_spec();
    if(rocke_gfx942_attention_4warp_gqa_lower_to_llvm(
           &spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, NULL, err, sizeof(err))
       != ROCKE_ERR_VALUE)
        fail("refuse/null_out", err);
    else
        printf("ok   refuse/null_out\n");
}

int main(void)
{
    test_lowers_like_the_builder();
    test_refusals();

    if(g_failures != 0)
    {
        fprintf(stderr, "%d GFX942 4-WARP GQA LOWER CHECK(S) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL GFX942 4-WARP GQA LOWER CHECKS PASSED\n");
    return 0;
}
