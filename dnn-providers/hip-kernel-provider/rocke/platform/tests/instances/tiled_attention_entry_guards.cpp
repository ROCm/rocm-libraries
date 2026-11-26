// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test that the C tiled-attention entries refuse what the Python
 * builders refuse, with the Python reason text, instead of crashing:
 *
 *   - a zero or non-dividing head pair, on the gfx942/gfx950 2D and 3D lowers
 *     and the gfx950 fastKV register-P lower (the kernel names carry the GQA
 *     ratio, and the shared 2D scaffold divides by num_kv_heads);
 *   - fastKV register-P on gfx942, on its lower and its support predicate (the
 *     body is gfx950-only);
 *   - the gfx950 2D spec pairing use_staggered_iter_wait with
 *     use_early_v_schedule (the two schedules overlap);
 *   - the gfx950 2D spec laying out the KQ LDS pad while Q is staged in the
 *     single K slot (Q is written and read unpadded there);
 *   - the gfx950 2D spec setting use_fp8_mfma_qk over a bf16 KV cache, with the
 *     KQ pad otherwise laid out (the fp8 refusal comes first, as in Python);
 *   - the gfx942 3D spec with use_i64_kv_addr (its loaders have no i64 path, so a
 *     spec built that way would address a cache past 2 GiB with 32 bits);
 *   - the gfx942 2D spec with use_i64_kv_addr where a wave's part of a K/V copy
 *     crosses a page (the pre-loop refusal), or where a conflict-free V store
 *     load call spans pages (the cfvst loader's refusal, inside the KV loop).
 *
 * The expected texts are the Python ones: kernels.common._fmha_common
 * check_head_counts and the fastKV module's _NOT_GFX950. The pytest twin,
 * test_attention_tiled_cpp_entry_guards.py, compares the two engines directly
 * where the rocke_engine binding is built.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/instance_gfx950_attention_tiled_2d_fastkv_regp.h"
#include "rocke/instance_gfx950_attention_tiled_3d.h"
#include "rocke/lower_llvm.h"

static const char* const NOT_POSITIVE = "num_query_heads / num_kv_heads must be > 0 (got 32, 0)";
static const char* const NOT_DIVIDING
    = "num_query_heads (32) must be divisible by num_kv_heads (6) for GQA / MQA";
static const char* const NOT_GFX950
    = "fastKV register-P is a gfx950 kernel; gfx942 is not supported";
static const char* const STAGGERED_EARLY_V
    = "use_staggered_iter_wait does not support use_early_v_schedule";
static const char* const GFX942_3D_I64
    = "gfx942 tiled 3D kernel does not support use_i64_kv_addr (paged KV cache > 2 GiB) yet";
static const char* const GFX942_2D_I64_WAVE_PAGE
    = "i64 K/V copies need a wave's part of a copy in one page";
static const char* const GFX942_2D_I64_CFVST_PAGE
    = "i64 K/V cfvst needs each V load call within one page";
static const char* const FP8_QK_NEEDS_FP8_KV
    = "use_fp8_mfma_qk requires kv_storage_dtype='fp8e4m3'";
static const char* const KQ_PAD_STAGED_Q
    = "KQ LDS pad failed - use_q_reread=False, use_q_direct_reg=False, expected one of them "
      "True (transposed 32x32 path) or use_kq_lds_pad=False (Q staged in the padded single K "
      "slot is read unpadded)";

typedef struct head_pair
{
    int num_query_heads;
    int num_kv_heads;
    const char* reason;
} head_pair_t;

static const head_pair_t BAD_PAIRS[] = {{32, 0, NOT_POSITIVE}, {32, 6, NOT_DIVIDING}};

/* Fails unless the call was refused (non-OK, no output) with `reason` in its
 * diagnostic. Frees any output it was handed. */
static int expect_refused(
    const char* what, rocke_status_t st, char* ll, const char* err, const char* reason)
{
    const bool refused = st != ROCKE_OK && ll == NULL;
    const bool says_why = strstr(err, reason) != NULL;
    free(ll);
    if(refused && says_why)
    {
        printf("%-40s refused: %s\n", what, err);
        return 0;
    }
    fprintf(stderr,
            "%-40s expected a refusal with \"%s\", got status=%d ll=%s err=\"%s\"\n",
            what,
            reason,
            (int)st,
            ll != NULL ? "<set>" : "NULL",
            err);
    return 1;
}

static rocke_attention_tiled_2d_spec_t spec_2d(int num_query_heads, int num_kv_heads)
{
    rocke_attention_tiled_2d_spec_t s = rocke_attention_tiled_2d_spec_default();
    s.head_size = 128;
    s.block_size = 16;
    s.num_query_heads = num_query_heads;
    s.num_kv_heads = num_kv_heads;
    s.dtype = "bf16";
    s.use_sinks = false;
    s.sliding_window = 0;
    s.has_softcap = false;
    return s;
}

/* make_fastkv_register_p_spec over the experiment's bf16 d64_b32 shape. */
static rocke_attention_tiled_2d_spec_t spec_fastkv(int num_query_heads, int num_kv_heads)
{
    rocke_attention_tiled_2d_spec_t s = spec_2d(num_query_heads, num_kv_heads);
    s.head_size = 64;
    s.block_size = 32;
    s.use_fast_paged_kv_desc = true;
    s.use_mfma_32x32 = true;
    s.use_transposed_qk_32x32 = true;
    return s;
}

static rocke_unified_attention_3d_tiled_spec_t spec_3d(int num_query_heads, int num_kv_heads)
{
    rocke_unified_attention_3d_tiled_spec_t s = rocke_unified_attention_3d_tiled_spec_default();
    s.head_size = 128;
    s.block_size = 16;
    s.num_query_heads = num_query_heads;
    s.num_kv_heads = num_kv_heads;
    s.dtype = "bf16";
    s.use_sinks = false;
    s.sliding_window = 0;
    s.has_softcap = false;
    s.num_segments = 16;
    return s;
}

static int check_head_counts(void)
{
    int failures = 0;
    for(size_t i = 0; i < sizeof(BAD_PAIRS) / sizeof(BAD_PAIRS[0]); ++i)
    {
        const head_pair_t* p = &BAD_PAIRS[i];
        char what[64];
        char err[512];
        char* ll;
        rocke_status_t st;

        const rocke_attention_tiled_2d_spec_t s2 = spec_2d(p->num_query_heads, p->num_kv_heads);
        const rocke_unified_attention_3d_tiled_spec_t s3
            = spec_3d(p->num_query_heads, p->num_kv_heads);
        const rocke_attention_tiled_2d_spec_t sf = spec_fastkv(p->num_query_heads, p->num_kv_heads);

        ll = NULL;
        err[0] = '\0';
        st = rocke_gfx942_attention_tiled_2d_lower_to_llvm(
            &s2, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
        snprintf(what, sizeof(what), "gfx942 2D %d/%d", p->num_query_heads, p->num_kv_heads);
        failures += expect_refused(what, st, ll, err, p->reason);

        ll = NULL;
        err[0] = '\0';
        st = rocke_gfx950_attention_tiled_2d_lower_to_llvm(
            &s2, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
        snprintf(what, sizeof(what), "gfx950 2D %d/%d", p->num_query_heads, p->num_kv_heads);
        failures += expect_refused(what, st, ll, err, p->reason);

        ll = NULL;
        err[0] = '\0';
        st = rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
            &s3, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
        snprintf(what, sizeof(what), "gfx942 3D %d/%d", p->num_query_heads, p->num_kv_heads);
        failures += expect_refused(what, st, ll, err, p->reason);

        ll = NULL;
        err[0] = '\0';
        st = rocke_build_unified_attention_3d_tiled_gfx950_lower_to_llvm(
            &s3, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
        snprintf(what, sizeof(what), "gfx950 3D %d/%d", p->num_query_heads, p->num_kv_heads);
        failures += expect_refused(what, st, ll, err, p->reason);

        ll = NULL;
        err[0] = '\0';
        st = rocke_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm(
            &sf, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
        snprintf(
            what, sizeof(what), "gfx950 fastKV regp %d/%d", p->num_query_heads, p->num_kv_heads);
        failures += expect_refused(what, st, ll, err, p->reason);
    }
    return failures;
}

static int check_fastkv_arch(void)
{
    int failures = 0;
    char err[512];
    char* ll = NULL;
    const rocke_attention_tiled_2d_spec_t s = spec_fastkv(64, 8);

    err[0] = '\0';
    const rocke_status_t st = rocke_gfx950_attention_tiled_2d_fastkv_regp_lower_to_llvm(
        &s, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    failures += expect_refused("gfx950 fastKV regp lower on gfx942", st, ll, err, NOT_GFX950);

    /* supports_fastkv_register_p_2d(arch="gfx942") over a shape it otherwise admits. */
    rocke_gfx950_supports_fastkv_register_p_2d_args_t a;
    memset(&a, 0, sizeof(a));
    a.head_size = 64;
    a.block_size = 32;
    a.dtype = "bf16";
    a.num_queries_per_kv = 8;
    a.num_query_heads = 64;
    a.num_kv_heads = 8;
    a.arch = "gfx942";
    err[0] = '\0';
    const bool ok = rocke_gfx950_supports_fastkv_register_p_2d(&a, err, sizeof(err));
    failures += expect_refused("gfx950 fastKV regp predicate on gfx942",
                               ok ? ROCKE_OK : ROCKE_ERR_VALUE,
                               NULL,
                               err,
                               NOT_GFX950);
    return failures;
}

/* The pytest twin's spec: bf16 transposed-32x32 with the V double buffer. */
static int check_gfx950_staggered_early_v(void)
{
    char err[512];
    char* ll = NULL;
    rocke_attention_tiled_2d_spec_t s = spec_2d(32, 32);
    s.head_size = 64;
    s.block_size = 32;
    s.use_mfma_32x32 = true;
    s.use_transposed_qk_32x32 = true;
    s.block_m_per_warp = 32;
    s.has_tile_size = true;
    s.tile_size = 64;
    s.use_v_double_buffer = true;
    s.use_staggered_iter_wait = true;
    s.use_early_v_schedule = true;

    err[0] = '\0';
    const rocke_status_t st = rocke_gfx950_attention_tiled_2d_lower_to_llvm(
        &s, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    return expect_refused("gfx950 2D staggered wait + early-V", st, ll, err, STAGGERED_EARLY_V);
}

static int check_gfx950_kq_pad_staged_q(void)
{
    char err[512];
    char* ll = NULL;
    rocke_attention_tiled_2d_spec_t s = spec_2d(32, 32);
    s.head_size = 64;
    s.block_size = 32;
    s.use_mfma_32x32 = true;
    s.use_transposed_qk_32x32 = true;
    s.num_warps = 2;
    s.block_m_per_warp = 32;
    s.has_tile_size = true;
    s.tile_size = 64;
    s.use_k_single_buffer = true;
    s.use_kq_lds_pad = true;
    s.kq_lds_pad_halves = 8;

    err[0] = '\0';
    const rocke_status_t st = rocke_gfx950_attention_tiled_2d_lower_to_llvm(
        &s, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    return expect_refused("gfx950 2D KQ pad + staged Q", st, ll, err, KQ_PAD_STAGED_Q);
}

static int check_gfx950_fp8_qk_without_fp8_kv(void)
{
    char err[512];
    char* ll = NULL;
    rocke_attention_tiled_2d_spec_t s = spec_2d(32, 32);
    s.head_size = 64;
    s.block_size = 32;
    s.use_mfma_32x32 = true;
    s.use_transposed_qk_32x32 = true;
    s.num_warps = 2;
    s.block_m_per_warp = 32;
    s.has_tile_size = true;
    s.tile_size = 64;
    s.use_k_single_buffer = true;
    s.use_kq_lds_pad = true;
    s.kq_lds_pad_halves = 8;
    s.use_fp8_mfma_qk = true;

    err[0] = '\0';
    const rocke_status_t st = rocke_gfx950_attention_tiled_2d_lower_to_llvm(
        &s, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    return expect_refused("gfx950 2D fp8 QK over bf16 KV", st, ll, err, FP8_QK_NEEDS_FP8_KV);
}

static int check_gfx942_3d_i64_kv_addr(void)
{
    char err[512];
    char* ll = NULL;
    rocke_unified_attention_3d_tiled_spec_t s = spec_3d(32, 8);
    s.use_i64_kv_addr = true;

    err[0] = '\0';
    const rocke_status_t st = rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
        &s, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    return expect_refused("gfx942 3D use_i64_kv_addr", st, ll, err, GFX942_3D_I64);
}

/* The two gfx942 2D i64 page refusals, with the specs the pytest twin uses. */
static int check_gfx942_2d_i64_pages(void)
{
    int failures = 0;
    char err[512];
    char* ll = NULL;

    /* D96 rows in 2-token pages: a wave's 128-half copy part crosses a page. */
    rocke_attention_tiled_2d_spec_t s = spec_2d(32, 8);
    s.head_size = 96;
    s.block_size = 2;
    s.num_warps = 1;
    s.has_tile_size = true;
    s.tile_size = 16;
    s.use_i64_kv_addr = true;
    err[0] = '\0';
    rocke_status_t st = rocke_gfx942_attention_tiled_2d_lower_to_llvm(
        &s, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    failures += expect_refused(
        "gfx942 2D i64 wave part across pages", st, ll, err, GFX942_2D_I64_WAVE_PAGE);

    /* 8-token pages: the pre-loop accepts the spec and the cfvst loader
     * refuses its 16-token load calls. */
    s = spec_2d(32, 8);
    s.head_size = 64;
    s.block_size = 8;
    s.num_warps = 4;
    s.block_m_per_warp = 32;
    s.has_tile_size = true;
    s.tile_size = 64;
    s.use_mfma_32x32x8 = true;
    s.use_transposed_qk_32x32 = true;
    s.use_conflict_free_v_store = true;
    s.use_i64_kv_addr = true;
    ll = NULL;
    err[0] = '\0';
    st = rocke_gfx942_attention_tiled_2d_lower_to_llvm(
        &s, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    failures += expect_refused(
        "gfx942 2D i64 cfvst call across pages", st, ll, err, GFX942_2D_I64_CFVST_PAGE);
    return failures;
}

int main(void)
{
    const int failures = check_head_counts() + check_fastkv_arch()
                         + check_gfx950_staggered_early_v() + check_gfx950_kq_pad_staged_q()
                         + check_gfx950_fp8_qk_without_fp8_kv() + check_gfx942_3d_i64_kv_addr()
                         + check_gfx942_2d_i64_pages();
    if(failures != 0)
    {
        fprintf(stderr, "%d TILED ATTENTION ENTRY GUARD CHECK(S) FAILED\n", failures);
        return 1;
    }
    printf("ALL TILED ATTENTION ENTRY GUARDS REFUSED WITH THE PYTHON REASON\n");
    return 0;
}
