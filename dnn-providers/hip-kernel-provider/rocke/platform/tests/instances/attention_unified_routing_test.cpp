// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test for the C port of unified-attention routing and coverage
 * gates: select_path / select_2d / select_3d, select_*_config,
 * _canonical_arch_for_selection, supports_native_unified_attention and
 * supports_native_unified_attention_3d_tiled. Expected values were produced by
 * the Python implementations (library/kernels/common/attention_unified.py,
 * platform/python/rocke/helpers/attention.py) on the same inputs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors.h"
#include "rocke/instance_attention_unified.h"

static int g_failures = 0;

#define CHECK(cond)                                                                  \
    do                                                                               \
    {                                                                                \
        if(!(cond))                                                                  \
        {                                                                            \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                            \
        }                                                                            \
    } while(0)

#define CHECK_EQ_INT(got, want)                                                                   \
    do                                                                                            \
    {                                                                                             \
        int got_ = (got);                                                                         \
        int want_ = (want);                                                                       \
        if(got_ != want_)                                                                         \
        {                                                                                         \
            fprintf(stderr, "%s:%d: %s == %d, want %d\n", __FILE__, __LINE__, #got, got_, want_); \
            ++g_failures;                                                                         \
        }                                                                                         \
    } while(0)

#define CHECK_EQ_STR(got, want)                           \
    do                                                    \
    {                                                     \
        const char* got_ = (got);                         \
        const char* want_ = (want);                       \
        if(got_ == NULL || strcmp(got_, want_) != 0)      \
        {                                                 \
            fprintf(stderr,                               \
                    "%s:%d: %s == \"%s\", want \"%s\"\n", \
                    __FILE__,                             \
                    __LINE__,                             \
                    #got,                                 \
                    got_ != NULL ? got_ : "(null)",       \
                    want_);                               \
            ++g_failures;                                 \
        }                                                 \
    } while(0)

/* bf16 D128 GQA 32/8 decode, 4 seqs, bs16, max_seqlen_k 8192 (num_2d = 40). */
static rocke_unified_attention_problem_t decode_problem(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    p.total_q = 4;
    p.num_seqs = 4;
    p.num_query_heads = 32;
    p.num_kv_heads = 8;
    p.head_size = 128;
    p.block_size = 16;
    p.max_seqlen_q = 1;
    p.max_seqlen_k = 8192;
    p.dtype = "bf16";
    return p;
}

static void check_3d(const rocke_attention_3d_config_t* c, int tile, int segs, int warps)
{
    CHECK_EQ_INT(c->tile_size, tile);
    CHECK_EQ_INT(c->num_segments_per_seq, segs);
    CHECK_EQ_INT(c->num_warps, warps);
    CHECK_EQ_INT(c->num_stages, 1);
    CHECK_EQ_INT(c->waves_per_eu, 2);
}

static void test_select_path(void)
{
    rocke_unified_attention_problem_t p = decode_problem();
    /* num_2d (40) <= target (480), long KV, no window -> 3d */
    CHECK_EQ_STR(rocke_unified_attention_problem_select_path(&p), "3d");

    p.sliding_window = 128;
    CHECK_EQ_STR(rocke_unified_attention_problem_select_path(&p), "2d");
    p.sliding_window = 0;

    p.max_seqlen_k = 512;
    CHECK_EQ_STR(rocke_unified_attention_problem_select_path(&p), "2d");
    p.max_seqlen_k = 8192;

    p.target_ctas = 8; /* overrides num_cus * 4 */
    CHECK_EQ_STR(rocke_unified_attention_problem_select_path(&p), "2d");

    p.num_query_heads = 30; /* not divisible by num_kv_heads */
    CHECK(rocke_unified_attention_problem_select_path(&p) == NULL);
}

static void test_select_2d(void)
{
    rocke_unified_attention_problem_t p = decode_problem();
    rocke_attention_2d_config_t c;
    CHECK(rocke_unified_attention_problem_select_2d(&p, &c));
    CHECK_EQ_INT(c.block_m, 16);
    CHECK_EQ_INT(c.block_q, 4);
    CHECK_EQ_INT(c.tile_size, 16);
    CHECK_EQ_INT(c.num_warps, 2);
    CHECK_EQ_INT(c.num_stages, 3);
    CHECK_EQ_INT(c.waves_per_eu, 2);

    /* fp16 D256 prefill, GQA 40/2 (nqpk 20 > 16), bs64, seqlen 4096. */
    p.total_q = 4096;
    p.num_seqs = 1;
    p.num_query_heads = 40;
    p.num_kv_heads = 2;
    p.head_size = 256;
    p.block_size = 64;
    p.max_seqlen_q = 4096;
    p.max_seqlen_k = 4096;
    p.dtype = "fp16";
    CHECK(rocke_unified_attention_problem_select_2d(&p, &c));
    CHECK_EQ_INT(c.block_m, 128);
    CHECK_EQ_INT(c.block_q, 6);
    CHECK_EQ_INT(c.tile_size, 64);
    CHECK_EQ_INT(c.num_warps, 4);
    CHECK_EQ_INT(c.num_stages, 1);

    CHECK(!rocke_attention_select_2d_config(16, 128, 0, true, 1, 1, 0, 1, &c));
}

static void test_select_3d(void)
{
    rocke_unified_attention_problem_t p = decode_problem();
    rocke_attention_3d_config_t attn, reduce;

    /* ceil(480/40)=12 -> 16 == min_segments (tile 16) -> reduce warps 1 */
    CHECK(rocke_unified_attention_problem_select_3d(&p, &attn, &reduce));
    check_3d(&attn, 16, 16, 2);
    check_3d(&reduce, 16, 16, 1);

    /* ceil(1216/40)=31 -> 32 */
    p.num_cus = 304;
    CHECK(rocke_unified_attention_problem_select_3d(&p, &attn, &reduce));
    check_3d(&attn, 16, 32, 2);
    check_3d(&reduce, 16, 32, 2);

    /* fp16 D64 MHA 8/8 single decode, bs32: ceil(1216/8)=152 -> 256 -> cap 128 */
    p = decode_problem();
    p.total_q = 1;
    p.num_seqs = 1;
    p.num_query_heads = 8;
    p.num_kv_heads = 8;
    p.head_size = 64;
    p.block_size = 32;
    p.max_seqlen_k = 65536;
    p.dtype = "fp16";
    p.num_cus = 304;
    CHECK(rocke_unified_attention_problem_select_3d(&p, &attn, &reduce));
    check_3d(&attn, 32, 128, 2);
    check_3d(&reduce, 32, 128, 2);

    CHECK(!rocke_attention_select_3d_config(128, 16, 2, 8192, 480, 0, &attn, &reduce));
}

static void test_canonical_arch(void)
{
    char out[64];
    CHECK_EQ_STR(rocke_unified_attention_canonical_arch("gfx942:sramecc+", out, sizeof(out)),
                 "gfx942");
    CHECK_EQ_STR(rocke_unified_attention_canonical_arch(" GFX950 ", out, sizeof(out)), "gfx950");
    CHECK_EQ_STR(rocke_unified_attention_canonical_arch("gfx1250-strict", out, sizeof(out)),
                 "gfx1250");
    CHECK_EQ_STR(rocke_unified_attention_canonical_arch("gfx1100", out, sizeof(out)),
                 "gfx11-generic");

    CHECK_EQ_STR(rocke_unified_attention_canonical_arch(NULL, out, sizeof(out)), "gfx950");
    CHECK_EQ_STR(rocke_unified_attention_canonical_arch("", out, sizeof(out)), "gfx950");
}

static void test_supports_tiled_3d(void)
{
    rocke_unified_attention_problem_t p = decode_problem();
    char reason[512];

    CHECK(rocke_unified_attention_supports_tiled_3d(&p, "gfx950", reason, sizeof(reason)));
    CHECK_EQ_STR(reason, "supported");

    p.causal_top_left = true;
    CHECK(!rocke_unified_attention_supports_tiled_3d(&p, "gfx942", reason, sizeof(reason)));
    CHECK_EQ_STR(reason, "gfx942 tiled attention does not implement top-left/causal masks");
    /* Python admits this on gfx950; the C++ builder does not implement it. */
    CHECK(!rocke_unified_attention_supports_tiled_3d(&p, "gfx950", reason, sizeof(reason)));
    CHECK_EQ_STR(reason, "tiled 3D top-left/causal masks are not ported to C++ yet");
    p.causal_top_left = false;

    p.use_fp8 = true;
    CHECK(
        !rocke_unified_attention_supports_tiled_3d(&p, "gfx942:sramecc+", reason, sizeof(reason)));
    CHECK_EQ_STR(reason,
                 "fp8 K/V on gfx942 (target_family='gfx9_mfma') decodes as e4m3fnuz/e5m2fnuz, "
                 "but the problem declares OCP-quantised K/V (fp8_fnuz=False); the mismatch "
                 "would silently mis-decode K/V. Quantise K/V to the arch-native format and set "
                 "fp8_fnuz to match, or run on an arch whose native fp8 format matches the "
                 "data.");

    /* A short buffer truncates rather than overflowing. */
    char tiny[8];
    CHECK(!rocke_unified_attention_supports_tiled_3d(&p, "gfx942", tiny, sizeof(tiny)));
    CHECK_EQ_STR(tiny, "fp8 K/V");
}

static void check_tiled_2d(const rocke_unified_attention_problem_t* p,
                           const char* arch,
                           bool want_ok,
                           const char* want_reason)
{
    char reason[512];
    CHECK(rocke_unified_attention_supports_tiled_2d(p, arch, reason, sizeof(reason)) == want_ok);
    CHECK_EQ_STR(reason, want_reason);
}

/* The gfx942 cases below depend on these flash knobs; run with the defaults. */
static void clear_gfx942_flash_env(void)
{
    static const char* const knobs[] = {"HIPDNN_GFX942_FLASH_MLIM",
                                        "HIPDNN_GFX942_K_SLICED_RING",
                                        "HIPDNN_GFX942_K_SLICE_HD",
                                        "HIPDNN_GFX942_D128_SMALLTILE_DK",
                                        "HIPDNN_GFX942_BF16_CFVST",
                                        "HIPDNN_GFX942_FLASH_WIDE"};
    for(size_t i = 0; i < sizeof(knobs) / sizeof(knobs[0]); ++i)
    {
#ifdef _WIN32
        _putenv_s(knobs[i], "");
#else
        unsetenv(knobs[i]);
#endif
    }
}

/* One case per outcome; reasons and verdicts are Python's for the same problem. */
static void test_supports_tiled_2d(void)
{
    clear_gfx942_flash_env();
    rocke_unified_attention_problem_t p = decode_problem();
    check_tiled_2d(&p, "gfx950", true, "supported");
    check_tiled_2d(&p, "gfx942", true, "supported");
    check_tiled_2d(&p, "gfx1250", false, "gfx1250 tiled 2D is not ported to C++");

    p.head_size = 96;
    check_tiled_2d(
        &p, "gfx950", false, "tiled 2D kernel only supports head_size in {64,128,256} (got 96)");

    p = decode_problem();
    p.use_additive_bias = true;
    check_tiled_2d(&p, "gfx950", false, "tiled 2D additive bias is not ported to C++ yet");
    check_tiled_2d(&p, "gfx942", false, "additive bias is only implemented on gfx950");

    /* gfx942 prefill: D128 sliding window (4-warp), fp16 D64 MHA (ring), bf16 D128 (mask-limit). */
    p = decode_problem();
    p.sliding_window = 256;
    p.max_seqlen_q = 512;
    p.total_q = 2048;
    check_tiled_2d(&p, "gfx942", false, "gfx942 4-warp GQA tiled 2D is not ported to C++ yet");

    p = decode_problem();
    p.dtype = "fp16";
    p.head_size = 64;
    p.num_query_heads = 8;
    p.max_seqlen_q = 2048;
    p.total_q = 8192;
    check_tiled_2d(&p, "gfx942", false, "gfx942 tiled 2D sliced-K ring is not ported to C++ yet");

    p = decode_problem();
    p.max_seqlen_q = 2048;
    p.total_q = 8192;
    check_tiled_2d(
        &p, "gfx942", false, "gfx942 tiled 2D flash mask-limit is not ported to C++ yet");

    /* gfx950: single-batch fp16 D64 MHA picks the V double-buffer. */
    p = decode_problem();
    p.dtype = "fp16";
    p.head_size = 64;
    p.num_query_heads = 8;
    p.num_seqs = 1;
    p.max_seqlen_q = 512;
    p.total_q = 512;
    check_tiled_2d(&p, "gfx950", false, "tiled 2D V double-buffer is not ported to C++ yet");

    p = decode_problem();
    p.dtype = "fp16";
    p.num_query_heads = 16;
    p.num_kv_heads = 16;
    p.max_seqlen_q = 512;
    p.total_q = 2048;
    p.use_alibi = true;
    /* ALiBi on the transposed 32x32 path (no mask-once) builds. */
    check_tiled_2d(&p, "gfx950", true, "supported");
}

static void test_supports_scalar(void)
{
    rocke_unified_attention_problem_t p = decode_problem();
    char reason[512];

    CHECK(rocke_unified_attention_supports_scalar(&p, "gfx950", reason, sizeof(reason)));
    CHECK_EQ_STR(reason, "supported by scalar CK DSL 2D attention backend");

    p.head_size = 96;
    CHECK(!rocke_unified_attention_supports_scalar(&p, "gfx950", reason, sizeof(reason)));
    CHECK_EQ_STR(reason, "unsupported head_size 96");
}

int main(void)
{
    test_select_path();
    test_select_2d();
    test_select_3d();
    test_canonical_arch();
    test_supports_tiled_3d();
    test_supports_tiled_2d();
    test_supports_scalar();
    if(g_failures)
    {
        fprintf(stderr, "attention_unified_routing_test: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("attention_unified_routing_test: OK\n");
    return 0;
}
