// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only spec-validation test for the gfx942 tiled attention builders. C
 * mirror of the gfx942 attention_tiled_2d / attention_tiled_3d __post_init__
 * guards and supports_tiled_2d, so the two engines agree on which specs they
 * ACCEPT (the byte-identity gate only compares the IR of specs both engines
 * already accept, so an acceptance divergence is invisible to it).
 *
 * Pinned here:
 *   1. the 2D spec validator runs on the build path (a pre-existing guard
 *      rejects through lower_to_llvm), while known-good specs still build;
 *   2. the register_pv, tile_size, fast_paged_kv_desc and mask_limit guards
 *      reject with ROCKE_ERR_VALUE;
 *   3. supports_tiled_2d rejects num_warps / block_m_per_warp outside the
 *      MFMA distribution's range;
 *   4. the 3D segment builder rejects num_segments < 1 (ROCKE_ERR_VALUE) and
 *      paged use_i64_kv_addr (ROCKE_ERR_NOTIMPL);
 *   5. the 2D validator's gfx950-only-knob and fp8 K/V rejects carry Python's
 *      exact text, the knob list untruncated and in Python's order.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/error.hpp"
#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/lower_llvm.h"

static int g_failures = 0;

static void fail(const char* what, const char* detail)
{
    fprintf(stderr, "FAIL %s: %s\n", what, (detail != NULL) ? detail : "<none>");
    ++g_failures;
}

static void expect_contains(const char* what, const char* haystack, const char* needle)
{
    if(haystack == NULL || strstr(haystack, needle) == NULL)
    {
        char detail[1024];
        snprintf(detail,
                 sizeof(detail),
                 "expected substring \"%s\" in \"%s\"",
                 needle,
                 (haystack != NULL) ? haystack : "");
        fail(what, detail);
        return;
    }
    printf("ok   %s (contains \"%s\")\n", what, needle);
}

/* ----------------------------------------------------------------------- 2D */

static rocke_attention_tiled_2d_spec_t
    spec_2d(int head_size, int block_size, int num_query_heads, int num_kv_heads, const char* dtype)
{
    rocke_attention_tiled_2d_spec_t s = rocke_attention_tiled_2d_spec_default();
    s.head_size = head_size;
    s.block_size = block_size;
    s.num_query_heads = num_query_heads;
    s.num_kv_heads = num_kv_heads;
    s.dtype = dtype;
    s.use_sinks = false;
    s.sliding_window = 0;
    s.has_softcap = false;
    return s;
}

/* The x8 transposed path (S^T = K @ Q^T on the 32x32x8 atom). */
static rocke_attention_tiled_2d_spec_t spec_2d_x8_transposed(void)
{
    rocke_attention_tiled_2d_spec_t s = spec_2d(64, 32, 32, 32, "fp16");
    s.block_m_per_warp = 32;
    s.has_tile_size = true;
    s.tile_size = 64;
    s.use_mfma_32x32x8 = true;
    s.use_transposed_qk_32x32 = true;
    return s;
}

static void expect_2d_builds(const char* what, const rocke_attention_tiled_2d_spec_t* spec)
{
    char err[ROCKE_ERR_MSG_CAP];
    char* ll = NULL;
    rocke_status_t st;

    err[0] = '\0';
    st = rocke_gfx942_attention_tiled_2d_lower_to_llvm(
        spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    if(st != ROCKE_OK || ll == NULL || ll[0] == '\0')
    {
        fail(what, err);
    }
    else
    {
        printf("ok   %s (builds)\n", what);
    }
    free(ll);
}

static void expect_2d_rejects(const char* what,
                              const rocke_attention_tiled_2d_spec_t* spec,
                              rocke_status_t want,
                              const char* needle)
{
    char err[ROCKE_ERR_MSG_CAP];
    char* ll = NULL;
    rocke_status_t st;

    err[0] = '\0';
    st = rocke_gfx942_attention_tiled_2d_lower_to_llvm(
        spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    free(ll);
    if(st == ROCKE_OK)
    {
        fail(what, "expected rejection, spec was accepted");
        return;
    }
    if(st != want)
    {
        char detail[ROCKE_ERR_MSG_CAP + 64];
        snprintf(detail, sizeof(detail), "status %d, expected %d (%s)", (int)st, (int)want, err);
        fail(what, detail);
        return;
    }
    expect_contains(what, err, needle);
}

static void test_2d_spec_guards(void)
{
    rocke_attention_tiled_2d_spec_t spec;

    /* 1. Known-good specs still build once the validator is on the path. */
    spec = spec_2d_x8_transposed();
    expect_2d_builds("2d/x8_transposed_builds", &spec);

    spec = spec_2d(64, 32, 32, 32, "bf16");
    spec.use_register_pv = true;
    expect_2d_builds("2d/register_pv_builds", &spec);

    /* 2. A guard that predates the build-path wiring: it only rejects if the
     * validator actually runs. */
    spec = spec_2d(64, 32, 32, 32, "fp16");
    spec.use_conflict_free_v = true;
    expect_2d_rejects("2d/validator_on_build_path",
                      &spec,
                      ROCKE_ERR_VALUE,
                      "use_conflict_free_v requires the transposed-x8 path");

    /* 3. register_pv v1 is bf16-only. */
    spec = spec_2d(64, 32, 32, 32, "fp16");
    spec.use_register_pv = true;
    expect_2d_rejects("2d/register_pv_dtype", &spec, ROCKE_ERR_VALUE, "dtype='bf16'");

    /* 4. A tile that is not a whole number of KV blocks. */
    spec = spec_2d(64, 32, 32, 32, "fp16");
    spec.has_tile_size = true;
    spec.tile_size = 48;
    expect_2d_rejects(
        "2d/tile_size_multiple", &spec, ROCKE_ERR_VALUE, "positive multiple of block_size");

    /* 5. The fast paged-KV descriptor is shape-restricted. */
    spec = spec_2d(64, 32, 32, 32, "bf16");
    spec.use_fast_paged_kv_desc = true;
    expect_2d_rejects(
        "2d/fast_paged_kv_desc_shape", &spec, ROCKE_ERR_VALUE, "use_fast_paged_kv_desc");

    /* 6. mask_limit on the full R4_s1mask path but with a sliding window: a
     * precondition failure (VALUE), not the unported-codegen reject. */
    spec = spec_2d_x8_transposed();
    spec.use_transposed_scalar_state = true;
    spec.use_transposed_mask_once = true;
    spec.use_transposed_mask_limit = true;
    spec.sliding_window = 64;
    expect_2d_rejects("2d/mask_limit_sliding_window", &spec, ROCKE_ERR_VALUE, "no sliding window");
}

/* Runs the validator directly, outside the builder's ROCKE_ERR_MSG_CAP error
 * latch, so a long message is seen whole. */
static void expect_2d_validate_message(const char* what,
                                       const rocke_attention_tiled_2d_spec_t* spec,
                                       const char* want)
{
    rocke_ir_builder_t b;

    if(rocke_ir_builder_init(&b, "spec_guards") != ROCKE_OK)
    {
        fail(what, "builder init failed");
        return;
    }
    try
    {
        rocke_attention_tiled_2d_spec_validate(&b, spec);
        fail(what, "expected rejection, spec was accepted");
    }
    catch(const ckc::Error& e)
    {
        if(e.code() != ROCKE_ERR_VALUE || strcmp(e.what(), want) != 0)
        {
            char detail[1024];
            snprintf(detail,
                     sizeof(detail),
                     "status %d, message \"%s\", expected \"%s\"",
                     (int)e.code(),
                     e.what(),
                     want);
            fail(what, detail);
        }
        else
        {
            printf("ok   %s (exact message)\n", what);
        }
    }
    rocke_ir_builder_free(&b);
}

static void test_2d_reject_messages(void)
{
    rocke_attention_tiled_2d_spec_t spec;

    /* Every gfx950-only knob: past ROCKE_ERR_MSG_CAP, listed in Python's
     * order whatever order the flags are set in. */
    spec = spec_2d(64, 32, 32, 32, "bf16");
    spec.use_fp8_mfma_pv = true;
    spec.use_fp8_mfma_qk = true;
    spec.use_grouped_kv2_softmax = true;
    spec.use_mfma32_skip_legacy_qreg = true;
    spec.use_transposed_half_local_pv = true;
    spec.use_mfma_32x32 = true;
    expect_2d_validate_message(
        "2d/gfx950_only_knobs_message",
        &spec,
        "gfx942 tiled-2D attention supports only the narrow 16x16x16 default path and the K=8 "
        "32x32x8 wide path; these are not available on gfx942: use_mfma_32x32, "
        "use_transposed_half_local_pv, use_mfma32_skip_legacy_qreg, use_grouped_kv2_softmax, "
        "use_fp8_mfma_qk, use_fp8_mfma_pv");

    spec = spec_2d(64, 32, 32, 32, "bf16");
    spec.kv_storage_dtype = "fp8e4m3";
    expect_2d_validate_message("2d/fp8_kv_message",
                               &spec,
                               "gfx942 tiled-2D attention does not support the fp8 K/V cache "
                               "(its PV path uses ds_read_tr_b8, a gfx950-only transpose read)");
}

/* ----------------------------------------------------------------- supports */

static rocke_gfx942_attention_tiled_2d_supports_args_t supports_args(void)
{
    rocke_gfx942_attention_tiled_2d_supports_args_t a
        = rocke_gfx942_attention_tiled_2d_supports_args_default();
    a.head_size = 64;
    a.block_size = 32;
    a.dtype = "bf16";
    a.num_queries_per_kv = 1;
    a.arch = "gfx942";
    return a;
}

static void expect_supports(const char* what,
                            const rocke_gfx942_attention_tiled_2d_supports_args_t* a,
                            bool want,
                            const char* needle)
{
    char reason[512];
    bool ok;

    reason[0] = '\0';
    ok = rocke_gfx942_attention_tiled_2d_supports(a, reason, sizeof(reason));
    if(ok != want)
    {
        char detail[600];
        snprintf(
            detail, sizeof(detail), "supports=%d, expected %d (%s)", (int)ok, (int)want, reason);
        fail(what, detail);
        return;
    }
    expect_contains(what, reason, needle);
}

static void test_supports_guards(void)
{
    rocke_gfx942_attention_tiled_2d_supports_args_t a;

    a = supports_args();
    expect_supports("supports/default", &a, true, "supported");

    a = supports_args();
    a.num_warps = 3;
    expect_supports("supports/num_warps", &a, false, "num_warps in {1,2,4,8}");

    a = supports_args();
    a.block_m_per_warp = 8;
    expect_supports("supports/block_m_per_warp", &a, false, "block_m_per_warp in {16,32}");
}

/* ----------------------------------------------------------------------- 3D */

static rocke_unified_attention_3d_tiled_spec_t spec_3d(void)
{
    rocke_unified_attention_3d_tiled_spec_t s = rocke_unified_attention_3d_tiled_spec_default();
    s.head_size = 64;
    s.block_size = 16;
    s.num_query_heads = 8;
    s.num_kv_heads = 8;
    s.dtype = "fp16";
    s.num_segments = 8;
    s.use_sinks = false;
    s.sliding_window = 0;
    s.has_softcap = false;
    return s;
}

static rocke_status_t
    lower_3d(const rocke_unified_attention_3d_tiled_spec_t* spec, char* err, size_t err_cap)
{
    char* ll = NULL;
    rocke_status_t st;

    err[0] = '\0';
    st = rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
        spec, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, err_cap);
    if(st == ROCKE_OK && (ll == NULL || ll[0] == '\0'))
    {
        st = ROCKE_ERR_VALUE;
        snprintf(err, err_cap, "empty .ll");
    }
    free(ll);
    return st;
}

static void expect_3d_rejects(const char* what,
                              const rocke_unified_attention_3d_tiled_spec_t* spec,
                              rocke_status_t want,
                              const char* needle)
{
    char err[ROCKE_ERR_MSG_CAP];
    rocke_status_t st = lower_3d(spec, err, sizeof(err));

    if(st == ROCKE_OK)
    {
        fail(what, "expected rejection, spec was accepted");
        return;
    }
    if(st != want)
    {
        char detail[ROCKE_ERR_MSG_CAP + 64];
        snprintf(detail, sizeof(detail), "status %d, expected %d (%s)", (int)st, (int)want, err);
        fail(what, detail);
        return;
    }
    expect_contains(what, err, needle);
}

static void test_3d_spec_guards(void)
{
    rocke_unified_attention_3d_tiled_spec_t spec;
    char err[ROCKE_ERR_MSG_CAP];

    spec = spec_3d();
    if(lower_3d(&spec, err, sizeof(err)) != ROCKE_OK)
    {
        fail("3d/default_builds", err);
    }
    else
    {
        printf("ok   3d/default_builds (builds)\n");
    }

    spec = spec_3d();
    spec.num_segments = 0;
    expect_3d_rejects("3d/num_segments", &spec, ROCKE_ERR_VALUE, "num_segments must be >= 1");

    /* The segment loaders never read the flag, so the paged >2 GiB case would
     * otherwise build with 32-bit KV addressing. */
    spec = spec_3d();
    spec.use_i64_kv_addr = true;
    expect_3d_rejects("3d/i64_kv_addr", &spec, ROCKE_ERR_NOTIMPL, "use_i64_kv_addr");
}

int main(void)
{
    test_2d_spec_guards();
    test_2d_reject_messages();
    test_supports_guards();
    test_3d_spec_guards();

    if(g_failures != 0)
    {
        fprintf(stderr, "%d GFX942 ATTENTION SPEC-GUARD CHECK(S) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL GFX942 ATTENTION SPEC GUARDS HELD\n");
    return 0;
}
