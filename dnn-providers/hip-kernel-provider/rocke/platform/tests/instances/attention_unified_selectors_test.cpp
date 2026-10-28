// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test pinning the C++ tiled-2D selectors to the values the Python
 * selectors (library/kernels/common/attention_unified.py) return for the same
 * problem and arch. Each case targets one selector branch; the expected values
 * were taken from the Python selectors.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors.h"
#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors_3d.h"

#define NA (-1) /* field not checked for this case */

static int g_failures = 0;

typedef struct
{
    const char* name;
    const char* arch;
    rocke_unified_attention_problem_t p;
    int tile_size;
    int num_warps;
    int block_m_per_warp;
    int waves_per_eu; /* NA, or 0 for Python None */
    int single_batch_combo; /* NA / 0 / 1 */
} selector_case_t;

/* total_q=2048, num_seqs=1, 32/8 heads, D128, bs16, q=k=2048, bf16. */
static rocke_unified_attention_problem_t base(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    p.total_q = 2048;
    p.num_seqs = 1;
    p.num_query_heads = 32;
    p.num_kv_heads = 8;
    p.head_size = 128;
    p.block_size = 16;
    p.max_seqlen_q = 2048;
    p.max_seqlen_k = 2048;
    p.dtype = "bf16";
    return p;
}

static selector_case_t make_case(const char* name,
                                 const char* arch,
                                 int tile_size,
                                 int num_warps,
                                 int block_m_per_warp,
                                 int waves_per_eu,
                                 int single_batch_combo)
{
    selector_case_t c;
    c.name = name;
    c.arch = arch;
    c.p = base();
    c.tile_size = tile_size;
    c.num_warps = num_warps;
    c.block_m_per_warp = block_m_per_warp;
    c.waves_per_eu = waves_per_eu;
    c.single_batch_combo = single_batch_combo;
    return c;
}

static void check_int(const char* name, const char* what, int got, int want)
{
    if(want != NA && got != want)
    {
        fprintf(stderr, "%s: %s = %d, want %d\n", name, what, got, want);
        ++g_failures;
    }
}

static void run_case(const selector_case_t* c)
{
    int wpe = -1;
    bool has_wpe;

    check_int(
        c->name, "tile_size", rocke_unified_attn_select_2d_tile_size(&c->p, c->arch), c->tile_size);
    check_int(
        c->name, "num_warps", rocke_unified_attn_select_2d_num_warps(&c->p, c->arch), c->num_warps);
    check_int(c->name,
              "block_m_per_warp",
              rocke_unified_attn_select_2d_block_m_per_warp(&c->p, c->arch),
              c->block_m_per_warp);
    has_wpe = rocke_unified_attn_select_2d_waves_per_eu(&c->p, c->arch, &wpe);
    check_int(c->name, "waves_per_eu", has_wpe ? wpe : 0, c->waves_per_eu);
    check_int(c->name,
              "single_batch_combo",
              rocke_unified_attn_enable_single_batch_combo(&c->p, c->arch) ? 1 : 0,
              c->single_batch_combo);
}

static void check_spec_flag(const char* name, const char* what, bool got, bool want)
{
    if(got != want)
    {
        fprintf(stderr, "%s: spec.%s = %d, want %d\n", name, what, (int)got, (int)want);
        ++g_failures;
    }
}

/* ------------------------------------------------ shared / gfx950 / gfx1250 */

static void test_shared_and_gfx950(void)
{
    selector_case_t c;

    c = make_case("d256 gfx950", "gfx950", 64, 2, 32, NA, NA);
    c.p.head_size = 256;
    c.p.block_size = 32;
    run_case(&c);

    /* Additive bias leaves the D256 cohort: generic geometry. */
    c = make_case("d256 gfx950 additive", "gfx950", 64, 1, 16, NA, NA);
    c.p.head_size = 256;
    c.p.block_size = 32;
    c.p.use_additive_bias = true;
    run_case(&c);

    c = make_case("gfx1250", "gfx1250", 16, 1, 16, 0, NA);
    run_case(&c);

    /* fp8 multi-seq prefill: block_m_per_warp is 32 on gfx950, 16 on gfx1250. */
    c = make_case("gfx1250 fp8 prefill", "gfx1250", 16, 1, 16, 0, NA);
    c.p.use_fp8 = true;
    c.p.num_seqs = 2;
    c.p.max_seqlen_q = 512;
    run_case(&c);

    c = make_case("gfx1250 wpe pin", "gfx1250", NA, NA, NA, 3, NA);
    c.p.waves_per_eu = 3;
    c.p.waves_per_eu_set = true;
    run_case(&c);

    /* The combo cohort selects wpe=4; the pin overrides it. */
    c = make_case("wpe pin on combo", "gfx950", NA, NA, NA, 1, NA);
    c.p.head_size = 64;
    c.p.block_size = 32;
    c.p.num_query_heads = 64;
    c.p.num_seqs = 4;
    c.p.max_seqlen_q = 512;
    c.p.waves_per_eu = 1;
    c.p.waves_per_eu_set = true;
    run_case(&c);

    c = make_case("single-batch fp16 D128 bs16 SW", "gfx950", 64, 4, 32, 2, 1);
    c.p.dtype = "fp16";
    c.p.sliding_window = 256;
    run_case(&c);
    {
        rocke_attention_tiled_2d_spec_t s
            = rocke_unified_attn_tiled_spec_from_problem(&c.p, "gfx950");
        check_spec_flag(c.name, "use_transposed_qk_32x32", s.use_transposed_qk_32x32, true);
        check_spec_flag(
            c.name, "use_transposed_scalar_state", s.use_transposed_scalar_state, false);
        check_spec_flag(c.name, "use_transposed_mask_once", s.use_transposed_mask_once, false);
        check_spec_flag(c.name, "use_transposed_mask_limit", s.use_transposed_mask_limit, false);
    }

    c = make_case("single-batch fp16 D128 bs32 SW", "gfx950", 64, 2, 16, NA, 0);
    c.p.dtype = "fp16";
    c.p.block_size = 32;
    c.p.sliding_window = 256;
    run_case(&c);

    c = make_case("single-batch bf16 D128 bs16 SW", "gfx950", 64, 4, 32, 2, 1);
    c.p.sliding_window = 256;
    run_case(&c);
}

/* ---------------------------------------------------------------- gfx942 */

static void set_env(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value != NULL ? value : "");
#else
    if(value != NULL)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

/* Unset every HIPDNN_GFX942_* knob the selectors read. */
static void clear_gfx942_env(void)
{
    static const char* const knobs[] = {"HIPDNN_GFX942_FLASH_MLIM",
                                        "HIPDNN_GFX942_K_SLICED_RING",
                                        "HIPDNN_GFX942_K_SLICE_HD",
                                        "HIPDNN_GFX942_K_LDSSEQ",
                                        "HIPDNN_GFX942_D128_SMALLTILE_DK",
                                        "HIPDNN_GFX942_BF16_CFVST",
                                        "HIPDNN_GFX942_FLASH_WIDE"};
    size_t i;

    for(i = 0; i < sizeof(knobs) / sizeof(knobs[0]); ++i)
        set_env(knobs[i], NULL);
}

static void test_gfx942(void)
{
    static const struct
    {
        const char* value;
        int num_warps;
    } wide[] = {
        {NULL, 4},
        {"4", 4},
        {"2", 2},
        {"0", 1},
        {"off", 1},
        {"disabled", 1},
        {" DISABLE ", 1},
        {"garbage", 4},
        {"8", 4},
    };
    selector_case_t c;
    size_t i;

    set_env("HIPDNN_GFX942_FLASH_WIDE", NULL);

    /* 4-warp GQA route: D128 sliding window (new cohort) and D256. */
    c = make_case("route D128 SWA bs16", "gfx942", 32, 1, 32, NA, NA);
    c.p.sliding_window = 4096;
    run_case(&c);

    c = make_case("route D128 SWA bs64", "gfx942", 64, 1, 32, NA, NA);
    c.p.sliding_window = 4096;
    c.p.block_size = 64;
    run_case(&c);

    c = make_case("route D256 bs32", "gfx942", 32, 1, 32, NA, NA);
    c.p.head_size = 256;
    c.p.block_size = 32;
    run_case(&c);

    /* An i64-sized cache leaves the route; fp16 flash then takes D128 SW at 2 warps. */
    c = make_case("D128 SWA fp16, i64 cache", "gfx942", 64, 2, 32, NA, NA);
    c.p.dtype = "fp16";
    c.p.sliding_window = 4096;
    c.p.num_kv_blocks = 1 << 20;
    run_case(&c);

    /* fp16 flash excludes small-q only for GQA; MHA small-q keeps it. */
    c = make_case("fp16 MHA D64 small-q", "gfx942", 64, 1, 32, NA, NA);
    c.p.dtype = "fp16";
    c.p.head_size = 64;
    c.p.num_query_heads = 8;
    c.p.max_seqlen_q = 512;
    c.p.total_q = 512;
    run_case(&c);

    c = make_case("fp16 MHA D128 small-q", "gfx942", 64, 1, 32, NA, NA);
    c.p.dtype = "fp16";
    c.p.num_query_heads = 8;
    c.p.max_seqlen_q = 512;
    c.p.total_q = 512;
    run_case(&c);

    c = make_case("fp16 GQA D128 small-q", "gfx942", 64, 2, 16, NA, NA);
    c.p.dtype = "fp16";
    c.p.max_seqlen_q = 512;
    c.p.total_q = 512;
    run_case(&c);

    /* Additive bias keeps a problem out of the route and out of fp16 flash. */
    c = make_case("fp16 MHA D128 SW additive", "gfx942", 64, 2, 16, NA, NA);
    c.p.dtype = "fp16";
    c.p.num_query_heads = 16;
    c.p.num_kv_heads = 16;
    c.p.sliding_window = 4096;
    c.p.max_seqlen_q = 512;
    c.p.total_q = 512;
    c.p.use_additive_bias = true;
    run_case(&c);

    for(i = 0; i < sizeof(wide) / sizeof(wide[0]); ++i)
    {
        char name[64];
        snprintf(
            name, sizeof(name), "FLASH_WIDE=%s", wide[i].value != NULL ? wide[i].value : "<unset>");
        set_env("HIPDNN_GFX942_FLASH_WIDE", wide[i].value);
        c = make_case(name, "gfx942", 64, wide[i].num_warps, 32, NA, NA);
        c.p.dtype = "fp16";
        run_case(&c);
    }
    set_env("HIPDNN_GFX942_FLASH_WIDE", NULL);
}

/* ------------------------------------------------------------------ bias */

static void check_gate(const char* name, const char* what, bool got, bool want)
{
    if(got != want)
    {
        fprintf(stderr, "%s: %s = %d, want %d\n", name, what, (int)got, (int)want);
        ++g_failures;
    }
}

/* --------------------------------------------------- gfx942 flash selectors */

/* The base() problem varied per case id: A bf16 D128, B bf16 D128 bs32,
 * C bf16 D64, D bf16 D128 small-q, E fp16 D128, F fp16 D128 SW, G fp16 D64 MHA. */
static rocke_unified_attention_problem_t flash_problem(char id)
{
    rocke_unified_attention_problem_t p = base();
    switch(id)
    {
    case 'B':
        p.block_size = 32;
        break;
    case 'C':
        p.head_size = 64;
        break;
    case 'D':
        p.max_seqlen_q = 512;
        p.total_q = 512;
        break;
    case 'E':
        p.dtype = "fp16";
        break;
    case 'F':
        p.dtype = "fp16";
        p.sliding_window = 256;
        break;
    case 'G':
        p.dtype = "fp16";
        p.head_size = 64;
        p.num_query_heads = 8;
        break;
    default:
        break;
    }
    return p;
}

static void test_gfx942_flash(void)
{
    /* Values from the Python selectors for the same problem and env. */
    static const struct
    {
        const char* env;
        const char* value;
        char id;
        int bf16_flash, smalltile_dk, wide_tile, wide_nw, wide_single_k, wide_cfvst;
        int flash_cfvst, flash_single, mask_limit, ring, ring_depth, k_slice_hd, flash_nw;
    } rows[] = {
        {NULL, NULL, 'A', 1, 0, 64, 2, 1, 0, 1, 0, 1, 0, 3, 32, 2},
        {NULL, NULL, 'B', 1, 1, 32, 2, 0, 0, 1, 0, 1, 0, 3, 32, 2},
        {NULL, NULL, 'C', 1, 0, 64, 4, 0, 1, 1, 0, 1, 1, 3, 16, 4},
        {NULL, NULL, 'D', 0, 0, 64, 2, 1, 0, 1, 0, 0, 0, 3, 32, 4},
        {NULL, NULL, 'E', 0, 0, 64, 2, 1, 0, 1, 0, 1, 1, 2, 32, 4},
        {NULL, NULL, 'F', 0, 0, 64, 2, 1, 0, 0, 1, 0, 0, 2, 32, 2},
        {NULL, NULL, 'G', 0, 0, 64, 4, 0, 1, 1, 0, 1, 1, 3, 16, 4},
        {"HIPDNN_GFX942_FLASH_MLIM", "d64", 'A', 1, 0, 64, 2, 1, 0, 1, 0, 0, 0, 3, 32, 2},
        {"HIPDNN_GFX942_FLASH_MLIM", "d64", 'C', 1, 0, 64, 4, 0, 1, 1, 0, 1, 1, 3, 16, 4},
        {"HIPDNN_GFX942_FLASH_MLIM", "d64", 'E', 0, 0, 64, 2, 1, 0, 1, 0, 0, 1, 2, 32, 4},
        {"HIPDNN_GFX942_K_SLICED_RING", "0", 'C', 1, 0, 64, 4, 0, 1, 1, 0, 1, 0, 3, 16, 4},
        {"HIPDNN_GFX942_K_SLICED_RING", "0", 'E', 0, 0, 64, 2, 1, 0, 1, 0, 1, 0, 2, 32, 4},
        {"HIPDNN_GFX942_K_SLICE_HD", "008", 'A', 1, 0, 64, 2, 1, 0, 1, 0, 1, 0, 3, 8, 2},
        {"HIPDNN_GFX942_K_SLICE_HD", "008", 'C', 1, 0, 64, 4, 0, 1, 1, 0, 1, 1, 3, 8, 4},
        {"HIPDNN_GFX942_K_SLICE_HD", "64", 'A', 1, 0, 64, 2, 1, 0, 1, 0, 1, 0, 3, 64, 2},
        {"HIPDNN_GFX942_K_SLICE_HD", "64", 'C', 1, 0, 64, 4, 0, 1, 1, 0, 1, 1, 3, 16, 4},
        {"HIPDNN_GFX942_K_SLICE_HD", "12", 'C', 1, 0, 64, 4, 0, 1, 1, 0, 1, 1, 3, 16, 4},
        {"HIPDNN_GFX942_D128_SMALLTILE_DK", " OFF ", 'B', 1, 0, 64, 2, 1, 0, 1, 0, 1, 0, 3, 32, 2},
        {"HIPDNN_GFX942_BF16_CFVST", "0", 'C', 1, 0, 64, 4, 0, 0, 1, 0, 1, 1, 3, 16, 4},
        {"HIPDNN_GFX942_FLASH_WIDE", "0", 'A', 1, 0, 64, 2, 1, 0, 0, 1, 1, 0, 3, 32, 2},
        {"HIPDNN_GFX942_FLASH_WIDE", "0", 'C', 1, 0, 64, 4, 0, 1, 0, 1, 1, 1, 3, 16, 1},
        {"HIPDNN_GFX942_FLASH_WIDE", "0", 'E', 0, 0, 64, 2, 1, 0, 0, 1, 1, 1, 2, 32, 1},
    };
    size_t i;

    clear_gfx942_env();
    for(i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
    {
        rocke_unified_attention_problem_t p = flash_problem(rows[i].id);
        char name[96];
        int nw = -1;
        bool single_k = false;

        if(rows[i].env != NULL)
            set_env(rows[i].env, rows[i].value);
        snprintf(name,
                 sizeof(name),
                 "flash %c %s=%s",
                 rows[i].id,
                 rows[i].env != NULL ? rows[i].env : "<default>",
                 rows[i].value != NULL ? rows[i].value : "");
        rocke_unified_attn_gfx942_bf16_wide_geometry(&p, "gfx942", &nw, &single_k);
        check_int(name,
                  "bf16_flash",
                  rocke_unified_attn_enable_gfx942_bf16_flash(&p, "gfx942"),
                  rows[i].bf16_flash);
        check_int(name,
                  "smalltile_doublek",
                  rocke_unified_attn_enable_gfx942_d128_smalltile_doublek(&p, "gfx942"),
                  rows[i].smalltile_dk);
        check_int(name,
                  "bf16_wide_tile_size",
                  rocke_unified_attn_gfx942_bf16_wide_tile_size(&p, "gfx942"),
                  rows[i].wide_tile);
        check_int(name, "bf16_wide nw", nw, rows[i].wide_nw);
        check_int(name, "bf16_wide single_k", single_k, rows[i].wide_single_k);
        check_int(name,
                  "bf16_wide_use_cfvst",
                  rocke_unified_attn_gfx942_bf16_wide_use_cfvst(&p),
                  rows[i].wide_cfvst);
        check_int(name,
                  "flash_use_cfvst",
                  rocke_unified_attn_gfx942_flash_use_cfvst(&p),
                  rows[i].flash_cfvst);
        check_int(name,
                  "flash_use_single_buffer",
                  rocke_unified_attn_gfx942_flash_use_single_buffer(&p),
                  rows[i].flash_single);
        check_int(name,
                  "flash_mask_limit",
                  rocke_unified_attn_enable_gfx942_flash_mask_limit(&p, "gfx942"),
                  rows[i].mask_limit);
        check_int(name,
                  "k_sliced_ring",
                  rocke_unified_attn_enable_gfx942_flash_k_sliced_ring(&p, "gfx942"),
                  rows[i].ring);
        check_int(name,
                  "ring_depth",
                  rocke_unified_attn_select_gfx942_flash_ring_depth(&p, "gfx942"),
                  rows[i].ring_depth);
        check_int(name,
                  "k_slice_hd",
                  rocke_unified_attn_select_gfx942_flash_k_slice_hd(&p),
                  rows[i].k_slice_hd);
        check_int(name,
                  "flash_num_warps",
                  rocke_unified_attn_select_gfx942_flash_num_warps(&p, "gfx942"),
                  rows[i].flash_nw);
        if(rows[i].env != NULL)
            set_env(rows[i].env, NULL);
    }

    /* The D128 sliding-window cohort routes to the 4-warp builder. */
    {
        rocke_unified_attention_problem_t p = flash_problem('F');
        check_gate(
            "flash F", "4warp_fast", rocke_unified_attn_gfx942_4warp_fast(&p, "gfx942"), true);
        check_gate("flash F",
                   "4warp_fast gfx950",
                   rocke_unified_attn_gfx942_4warp_fast(&p, "gfx950"),
                   false);
    }
    /* Every gfx942 flash predicate is off on gfx950. */
    {
        rocke_unified_attention_problem_t p = flash_problem('A');
        check_gate("flash A",
                   "bf16_flash gfx950",
                   rocke_unified_attn_enable_gfx942_bf16_flash(&p, "gfx950"),
                   false);
        p = flash_problem('E');
        check_gate("flash E",
                   "fp16_flash gfx950",
                   rocke_unified_attn_enable_gfx942_fp16_flash(&p, "gfx950"),
                   false);
        check_int("flash E",
                  "ring_depth gfx950",
                  rocke_unified_attn_select_gfx942_flash_ring_depth(&p, "gfx950"),
                  3);
    }
}

/* ------------------------------------------------------------- fp8 MFMA QK */

/* base() as fp8 decode: 4 seqs, q=1, k=512. */
static rocke_unified_attention_problem_t fp8_decode(void)
{
    rocke_unified_attention_problem_t p = base();
    p.use_fp8 = true;
    p.num_seqs = 4;
    p.total_q = 4;
    p.max_seqlen_q = 1;
    p.max_seqlen_k = 512;
    return p;
}

static void test_fp8_mfma_qk(void)
{
    static const char* const archs[] = {"gfx950", "gfx942"};
    /* Values from the Python selectors; the same on both archs. */
    static const struct
    {
        const char* name;
        int seqlen_q, seqlen_k, sliding_window, head_size, block_size, nq, use_fp8;
        int want;
    } rows[] = {
        {"decode k512", 1, 512, 0, 128, 16, 32, 1, 1},
        {"decode k4096", 1, 4096, 0, 128, 16, 32, 1, 0},
        {"decode k4096 SW", 1, 4096, 256, 128, 16, 32, 1, 1},
        {"q512", 512, 512, 0, 128, 16, 32, 1, 0},
        {"q256 k512", 256, 512, 0, 128, 16, 32, 1, 1},
        {"no fp8", 1, 512, 0, 128, 16, 32, 0, 0},
        {"d64 bs32 nq64 q256", 256, 512, 0, 64, 32, 64, 1, 1},
    };
    /* Explicit tile_size / num_warps, on gfx950 d64 decode k512. */
    static const struct
    {
        int tile_size, num_warps, fits, want;
    } overrides[] = {
        {16, 8, 0, 0}, /* 1 KB of K: no payload fits 512 lanes */
        {16, 4, 1, 0}, /* fits, but k512 > 16*T */
        {48, 4, 1, 1}, /* only the 12-byte payload fits */
        {32, 8, 1, 1},
    };
    size_t i;
    size_t a;
    char what[64];

    for(a = 0; a < sizeof(archs) / sizeof(archs[0]); ++a)
    {
        for(i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
        {
            rocke_unified_attention_problem_t p = fp8_decode();
            p.max_seqlen_q = rows[i].seqlen_q;
            p.total_q = rows[i].seqlen_q * p.num_seqs;
            p.max_seqlen_k = rows[i].seqlen_k;
            p.sliding_window = rows[i].sliding_window;
            p.head_size = rows[i].head_size;
            p.block_size = rows[i].block_size;
            p.num_query_heads = rows[i].nq;
            p.use_fp8 = rows[i].use_fp8 != 0;
            snprintf(what, sizeof(what), "fp8_mfma_qk %s", archs[a]);
            check_gate(rows[i].name,
                       what,
                       rocke_unified_attn_enable_fp8_mfma_qk(&p, archs[a], 0, 0),
                       rows[i].want != 0);
        }
    }
    for(i = 0; i < sizeof(overrides) / sizeof(overrides[0]); ++i)
    {
        rocke_unified_attention_problem_t p = fp8_decode();
        p.head_size = 64;
        snprintf(what, sizeof(what), "T%d nw%d", overrides[i].tile_size, overrides[i].num_warps);
        check_gate(what,
                   "fp8_qk_loader_fits",
                   rocke_unified_attn_fp8_qk_loader_fits(
                       &p, "gfx950", overrides[i].tile_size, overrides[i].num_warps),
                   overrides[i].fits != 0);
        check_gate(what,
                   "fp8_mfma_qk",
                   rocke_unified_attn_enable_fp8_mfma_qk(
                       &p, "gfx950", overrides[i].tile_size, overrides[i].num_warps),
                   overrides[i].want != 0);
    }
    {
        rocke_unified_attention_problem_t p = fp8_decode();
        check_spec_flag("fp8 decode k512",
                        "use_fp8_mfma_qk",
                        rocke_unified_attn_tiled_spec_from_problem(&p, "gfx950").use_fp8_mfma_qk,
                        true);
        p.max_seqlen_k = 4096;
        check_spec_flag("fp8 decode k4096",
                        "use_fp8_mfma_qk",
                        rocke_unified_attn_tiled_spec_from_problem(&p, "gfx950").use_fp8_mfma_qk,
                        false);
    }
}

/* ------------------------------------------------- gfx942 spec from problem */

/* The base() problem varied per case id: A fp16 D128, B fp16 D128 SW (4-warp
 * route), C bf16 D64, D bf16 D128, E bf16 D128 bs32, F bf16 D128 small-q,
 * G fp16 D64 MHA decode. */
static rocke_unified_attention_problem_t spec_problem(char id)
{
    rocke_unified_attention_problem_t p = base();
    switch(id)
    {
    case 'A':
        p.dtype = "fp16";
        break;
    case 'B':
        p.dtype = "fp16";
        p.sliding_window = 256;
        break;
    case 'C':
        p.head_size = 64;
        break;
    case 'E':
        p.block_size = 32;
        break;
    case 'F':
        p.max_seqlen_q = 512;
        p.total_q = 512;
        break;
    case 'G':
        p.dtype = "fp16";
        p.head_size = 64;
        p.num_query_heads = 8;
        p.max_seqlen_q = 1;
        p.total_q = 1;
        break;
    default:
        break;
    }
    return p;
}

static void test_gfx942_spec(void)
{
    /* Values from Python's attention_spec_builder._tiled_spec_from_problem. */
    static const struct
    {
        const char* ldsseq_env;
        char id;
        int nw, tile, mw, x8, tqk, mlim, cfvst, single_k, ring, depth, kslice, ldsseq, qdirect,
            policy_all, regpv;
    } rows[] = {
        {NULL, 'A', 4, 64, 32, 1, 1, 1, 1, 0, 1, 2, 32, 0, 0, 1, 0},
        {NULL, 'B', 1, 32, 32, 0, 0, 0, 0, 0, 0, 3, 32, 0, 0, 0, 0},
        {NULL, 'C', 4, 64, 32, 1, 1, 1, 1, 0, 1, 3, 16, 0, 1, 1, 0},
        {NULL, 'D', 2, 64, 32, 1, 1, 1, 0, 1, 0, 3, 32, 0, 0, 1, 0},
        {NULL, 'E', 2, 32, 32, 1, 1, 1, 0, 0, 0, 3, 32, 0, 0, 1, 0},
        {NULL, 'F', 2, 64, 16, 0, 0, 0, 0, 0, 0, 3, 32, 0, 0, 0, 1},
        {NULL, 'G', 4, 64, 32, 1, 1, 1, 1, 0, 0, 3, 16, 0, 1, 1, 0},
        {"ck", 'C', 4, 64, 32, 1, 1, 1, 1, 0, 1, 3, 16, 1, 1, 1, 0},
        {"ck", 'A', 4, 64, 32, 1, 1, 1, 1, 0, 1, 2, 32, 0, 0, 1, 0},
    };
    static const struct
    {
        const char* value;
        int bf16_d64, fp16_d128;
    } ldsseq[] = {{"ck", 1, 0}, {" On ", 1, 0}, {"0", 0, 0}, {"2", 0, 0}};
    size_t i;

    clear_gfx942_env();
    for(i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
    {
        rocke_unified_attention_problem_t p = spec_problem(rows[i].id);
        rocke_attention_tiled_2d_spec_t s;
        char name[48];

        set_env("HIPDNN_GFX942_K_LDSSEQ", rows[i].ldsseq_env);
        s = rocke_unified_attn_tiled_spec_from_problem(&p, "gfx942");
        snprintf(name,
                 sizeof(name),
                 "gfx942 spec %c%s",
                 rows[i].id,
                 rows[i].ldsseq_env != NULL ? " ldsseq" : "");
        check_int(name, "num_warps", s.num_warps, rows[i].nw);
        check_int(name, "tile_size", s.has_tile_size ? s.tile_size : -2, rows[i].tile);
        check_int(name, "block_m_per_warp", s.block_m_per_warp, rows[i].mw);
        check_int(name, "waves_per_eu", s.has_waves_per_eu ? s.waves_per_eu : 0, 2);
        check_spec_flag(name, "use_mfma_32x32x8", s.use_mfma_32x32x8, rows[i].x8);
        check_spec_flag(name, "use_transposed_qk_32x32", s.use_transposed_qk_32x32, rows[i].tqk);
        check_spec_flag(
            name, "use_transposed_mask_limit", s.use_transposed_mask_limit, rows[i].mlim);
        check_spec_flag(
            name, "use_transposed_invariant_hoist", s.use_transposed_invariant_hoist, rows[i].mlim);
        check_spec_flag(
            name, "use_conflict_free_v_store", s.use_conflict_free_v_store, rows[i].cfvst);
        check_spec_flag(name, "use_k_single_buffer", s.use_k_single_buffer, rows[i].single_k);
        check_spec_flag(name, "use_k_sliced_ring", s.use_k_sliced_ring, rows[i].ring);
        check_int(name, "ring_depth", s.ring_depth, rows[i].depth);
        check_int(name, "k_slice_hd", s.k_slice_hd, rows[i].kslice);
        check_spec_flag(name, "use_k_sliced_ldsseq", s.use_k_sliced_ldsseq, rows[i].ldsseq);
        check_spec_flag(name, "use_q_direct_global", s.use_q_direct_global, rows[i].qdirect);
        check_spec_flag(
            name, "kv_cache_policy all", strcmp(s.kv_cache_policy, "all") == 0, rows[i].policy_all);
        check_spec_flag(name, "use_register_pv", s.use_register_pv, rows[i].regpv);
        check_spec_flag(name, "use_mfma_32x32", s.use_mfma_32x32, false);
    }
    for(i = 0; i < sizeof(ldsseq) / sizeof(ldsseq[0]); ++i)
    {
        rocke_unified_attention_problem_t c = spec_problem('C');
        rocke_unified_attention_problem_t a = spec_problem('A');
        set_env("HIPDNN_GFX942_K_LDSSEQ", ldsseq[i].value);
        check_gate(ldsseq[i].value,
                   "k_sliced_ldsseq bf16 D64",
                   rocke_unified_attn_enable_gfx942_flash_k_sliced_ldsseq(&c, "gfx942"),
                   ldsseq[i].bf16_d64 != 0);
        check_gate(ldsseq[i].value,
                   "k_sliced_ldsseq fp16 D128 (depth 2)",
                   rocke_unified_attn_enable_gfx942_flash_k_sliced_ldsseq(&a, "gfx942"),
                   ldsseq[i].fp16_d128 != 0);
    }
    set_env("HIPDNN_GFX942_K_LDSSEQ", NULL);
    {
        rocke_unified_attention_problem_t c = spec_problem('C');
        rocke_unified_attention_problem_t f = spec_problem('F');
        rocke_attention_tiled_2d_spec_t s;
        check_gate(
            "C", "q_direct", rocke_unified_attn_enable_gfx942_flash_q_direct(&c, "gfx942"), true);
        check_gate("C",
                   "q_direct gfx950",
                   rocke_unified_attn_enable_gfx942_flash_q_direct(&c, "gfx950"),
                   false);
        check_gate(
            "F", "q_direct", rocke_unified_attn_enable_gfx942_flash_q_direct(&f, "gfx942"), false);
        check_spec_flag("C",
                        "kv_cache_policy",
                        strcmp(rocke_unified_attn_gfx942_flash_kv_cache_policy(&c), "all") == 0,
                        true);
        /* Every non-gfx950 arch other than gfx1250 takes the generic builder. */
        s = rocke_unified_attn_tiled_spec_from_problem(&f, "gfx1201");
        check_int("gfx1201 F", "num_warps", s.num_warps, 2);
        check_int("gfx1201 F", "tile_size", s.tile_size, 32);
        check_int("gfx1201 F", "block_m_per_warp", s.block_m_per_warp, 16);
        check_spec_flag("gfx1201 F", "use_register_pv", s.use_register_pv, true);
        s = rocke_unified_attn_tiled_spec_from_problem(&f, "gfx1250");
        check_spec_flag("gfx1250 F", "default spec", s.head_size == 0 && !s.has_tile_size, true);
    }
}

/* ------------------------------------------------------------ 3D selectors */

/* Decode base: bf16 D128 bs16 32/8 heads, 4 seqs, q=1, k=2048, 304 CUs. */
static rocke_unified_attention_problem_t decode_3d(void)
{
    rocke_unified_attention_problem_t p = base();
    p.num_seqs = 4;
    p.total_q = 4;
    p.max_seqlen_q = 1;
    p.max_seqlen_k = 2048;
    p.num_cus = 304;
    return p;
}

static void test_3d_selectors(void)
{
    /* Values from the Python 3D selectors (tile overrides: -1 is None). */
    static const struct
    {
        const char* name;
        const char* arch;
        int head_size, block_size, num_seqs, seqlen_q, seqlen_k, sliding_window, num_cus,
            target_ctas, use_fp8;
        int segments, tile_override, wide_kv, decode_grid, stream, swizzle, page_ids;
    } rows[] = {
        {"942 decode rule", "gfx942", 128, 16, 4, 1, 2048, 0, 304, 0, 0, 38, -1, 1, 1, 1, 1, 1},
        {"942 D64 bs32 short kv",
         "gfx942",
         64,
         32,
         4,
         1,
         2048,
         0,
         304,
         0,
         0,
         32,
         -1,
         1,
         1,
         1,
         1,
         1},
        {"942 D128 bs32 short kv",
         "gfx942",
         128,
         32,
         4,
         1,
         2048,
         0,
         304,
         0,
         0,
         16,
         16,
         1,
         1,
         1,
         1,
         0},
        {"942 D256 bs32 short kv",
         "gfx942",
         256,
         32,
         4,
         1,
         2048,
         0,
         304,
         0,
         0,
         32,
         16,
         0,
         1,
         1,
         0,
         0},
        {"942 D128 bs32 k16384",
         "gfx942",
         128,
         32,
         4,
         1,
         16384,
         0,
         304,
         0,
         0,
         16,
         16,
         1,
         1,
         1,
         1,
         0},
        {"942 prefill clamp", "gfx942", 128, 16, 4, 8, 2048, 0, 1024, 0, 0, 16, -1, 1, 0, 0, 0, 0},
        {"942 sliding window",
         "gfx942",
         128,
         16,
         4,
         1,
         2048,
         256,
         304,
         0,
         0,
         32,
         -1,
         1,
         1,
         1,
         1,
         1},
        {"942 fp8 decode", "gfx942", 128, 16, 4, 1, 2048, 0, 304, 0, 1, 16, -1, 0, 1, 0, 0, 0},
        {"942 D256 bs16", "gfx942", 256, 16, 4, 1, 2048, 0, 304, 0, 0, 32, -1, 0, 1, 1, 1, 1},
        {"950 thin decode", "gfx950", 128, 16, 1, 1, 65536, 0, 304, 0, 0, 64, 64, 0, 1, 1, 1, 1},
        {"950 thin decode 600 cus",
         "gfx950",
         128,
         16,
         1,
         1,
         65536,
         0,
         600,
         0,
         0,
         128,
         64,
         0,
         1,
         1,
         1,
         1},
        {"950 decode rule", "gfx950", 128, 16, 16, 1, 2048, 0, 304, 0, 0, 9, -1, 0, 1, 1, 0, 1},
        {"950 prefill clamp", "gfx950", 128, 16, 4, 8, 2048, 0, 1024, 0, 0, 16, -1, 0, 0, 0, 0, 0},
        {"950 target_ctas", "gfx950", 128, 16, 4, 8, 2048, 0, 304, 2048, 0, 32, -1, 0, 0, 0, 0, 0},
    };
    size_t i;

    set_env("HIPDNN_GFX942_3D_WKV", NULL);
    for(i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
    {
        rocke_unified_attention_problem_t p = decode_3d();
        rocke_unified_attention_3d_tiled_spec_t s;
        int tile = -1;
        bool is950 = strcmp(rows[i].arch, "gfx950") == 0;

        p.head_size = rows[i].head_size;
        p.block_size = rows[i].block_size;
        p.num_seqs = rows[i].num_seqs;
        p.max_seqlen_q = rows[i].seqlen_q;
        p.total_q = rows[i].seqlen_q * rows[i].num_seqs;
        p.max_seqlen_k = rows[i].seqlen_k;
        p.sliding_window = rows[i].sliding_window;
        p.num_cus = rows[i].num_cus;
        p.target_ctas = rows[i].target_ctas;
        p.use_fp8 = rows[i].use_fp8 != 0;

        check_int(rows[i].name,
                  "num_segments",
                  rocke_unified_attn_num_segments(&p, rows[i].arch),
                  rows[i].segments);
        if(is950)
            (void)rocke_unified_attn_gfx950_3d_tile_size_override(&p, rows[i].arch, &tile);
        else
            (void)rocke_unified_attn_gfx942_3d_tile_size_override(&p, rows[i].arch, &tile);
        check_int(rows[i].name, "tile_size_override", tile, rows[i].tile_override);
        check_gate(rows[i].name,
                   "wide_kv_load",
                   rocke_unified_attn_enable_gfx942_3d_wide_kv_load(&p, rows[i].arch),
                   rows[i].wide_kv);
        check_gate(rows[i].name,
                   "decode_grid",
                   rocke_unified_attn_use_decode_grid(&p, rows[i].arch),
                   rows[i].decode_grid);
        check_gate(rows[i].name,
                   "kv_cache_policy stream",
                   strcmp(rocke_unified_attn_kv_cache_policy_3d(&p, rows[i].arch), "stream") == 0,
                   rows[i].stream);
        check_gate(rows[i].name,
                   "kv_lds_swizzle",
                   rocke_unified_attn_kv_lds_swizzle_3d(&p, rows[i].arch),
                   rows[i].swizzle);
        check_gate(rows[i].name,
                   "wave_page_id_load",
                   rocke_unified_attn_wave_page_id_load_3d(&p, rows[i].arch),
                   rows[i].page_ids);

        /* The assembled spec carries the same values. */
        s = rocke_unified_attn_tiled_3d_spec_from_problem(&p, rows[i].arch);
        check_int(rows[i].name, "spec.num_segments", s.num_segments, rows[i].segments);
        check_int(rows[i].name,
                  "spec.tile_size_override",
                  s.has_tile_size_override ? s.tile_size_override : -1,
                  rows[i].tile_override);
        check_spec_flag(rows[i].name, "use_wide_kv_load", s.use_wide_kv_load, rows[i].wide_kv);
        check_spec_flag(rows[i].name, "use_decode_grid", s.use_decode_grid, rows[i].decode_grid);
        check_spec_flag(rows[i].name, "use_kv_lds_swizzle", s.use_kv_lds_swizzle, rows[i].swizzle);
        check_spec_flag(
            rows[i].name, "use_wave_page_id_load", s.use_wave_page_id_load, rows[i].page_ids);
        check_spec_flag(rows[i].name, "waves_per_eu None", !s.has_waves_per_eu, true);
    }
    {
        rocke_unified_attention_problem_t p = decode_3d();
        int wpe = -1;
        rocke_unified_attention_3d_tiled_spec_t s;

        check_int("3d", "pre_bump_segments", rocke_unified_attn_pre_bump_segments(&p), 16);
        set_env("HIPDNN_GFX942_3D_WKV", " Off ");
        check_gate("3d WKV off",
                   "wide_kv_load",
                   rocke_unified_attn_enable_gfx942_3d_wide_kv_load(&p, "gfx942"),
                   false);
        check_gate("3d WKV off",
                   "kv_lds_swizzle",
                   rocke_unified_attn_kv_lds_swizzle_3d(&p, "gfx942"),
                   false);
        set_env("HIPDNN_GFX942_3D_WKV", NULL);
        check_gate("3d gfx1250",
                   "waves_per_eu",
                   rocke_unified_attn_select_3d_waves_per_eu(&p, "gfx1250", &wpe),
                   true);
        check_int("3d gfx1250", "waves_per_eu", wpe, 2);
        p.waves_per_eu_set = true;
        p.waves_per_eu = 3;
        s = rocke_unified_attn_tiled_3d_spec_from_problem(&p, "gfx942");
        check_int("3d wpe pin", "spec.waves_per_eu", s.has_waves_per_eu ? s.waves_per_eu : 0, 3);
        s = rocke_unified_attn_tiled_3d_spec_from_problem(&p, "gfx1250");
        check_spec_flag("3d gfx1250", "default spec", s.head_size == 0, true);
    }
}

/* gfx950 d64 bs32 64/8 heads, 4 seqs, q=512: the combo cohort. */
static selector_case_t combo_case(const char* name)
{
    selector_case_t c = make_case(name, "gfx950", 64, 4, 32, 4, NA);
    c.p.head_size = 64;
    c.p.block_size = 32;
    c.p.num_query_heads = 64;
    c.p.num_seqs = 4;
    c.p.max_seqlen_q = 512;
    return c;
}

static void test_bias(void)
{
    static const char* const kinds[] = {"alibi", "qq", "softcap", "additive"};
    selector_case_t c;
    size_t i;

    /* Bias rides the combo stack without mask-once / mask-limit. */
    for(i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i)
    {
        char name[48];
        rocke_attention_tiled_2d_spec_t s;
        snprintf(name, sizeof(name), "combo + %s", kinds[i]);
        c = combo_case(name);
        c.p.use_alibi = (i == 0);
        c.p.use_qq_bias = (i == 1);
        c.p.softcap = (i == 2) ? 50.0 : 0.0;
        c.p.use_additive_bias = (i == 3);
        run_case(&c);
        check_gate(name, "combo_2d", rocke_unified_attn_enable_combo_2d(&c.p, "gfx950"), true);
        check_gate(name,
                   "transposed_subflags",
                   rocke_unified_attn_enable_transposed_subflags(&c.p, "gfx950"),
                   false);
        s = rocke_unified_attn_tiled_spec_from_problem(&c.p, "gfx950");
        check_spec_flag(name, "use_transposed_qk_32x32", s.use_transposed_qk_32x32, true);
        check_spec_flag(name, "use_transposed_scalar_state", s.use_transposed_scalar_state, true);
        check_spec_flag(name, "use_transposed_mask_once", s.use_transposed_mask_once, false);
        check_spec_flag(name, "use_transposed_mask_limit", s.use_transposed_mask_limit, false);
        check_spec_flag(name, "use_fast_paged_kv_desc", s.use_fast_paged_kv_desc, true);
    }

    c = make_case("transposed-32x32 + alibi", "gfx950", 32, 4, 32, NA, NA);
    c.p.dtype = "fp16";
    c.p.num_query_heads = 16;
    c.p.num_kv_heads = 16;
    c.p.num_seqs = 4;
    c.p.max_seqlen_q = 512;
    c.p.use_alibi = true;
    run_case(&c);
    check_gate(c.name,
               "transposed_qk_32x32",
               rocke_unified_attn_enable_transposed_qk_32x32(&c.p, "gfx950"),
               true);

    c = make_case("single-batch + alibi", "gfx950", 32, 4, 32, NA, 1);
    c.p.use_alibi = true;
    run_case(&c);
    {
        rocke_attention_tiled_2d_spec_t s
            = rocke_unified_attn_tiled_spec_from_problem(&c.p, "gfx950");
        check_spec_flag(c.name, "use_transposed_qk_32x32", s.use_transposed_qk_32x32, true);
        check_spec_flag(
            c.name, "use_transposed_scalar_state", s.use_transposed_scalar_state, false);
    }

    c = make_case("single-batch + qq d64", "gfx950", 128, 4, 32, NA, 1);
    c.p.head_size = 64;
    c.p.use_qq_bias = true;
    run_case(&c);

    /* Additive bias joins the other score terms in every exclusion. */
    c = make_case("register-PV additive", "gfx950", 32, 2, 16, NA, NA);
    c.p.head_size = 64;
    c.p.num_seqs = 2;
    c.p.max_seqlen_q = 128;
    check_gate(c.name, "register_pv", rocke_unified_attn_enable_register_pv(&c.p, "gfx950"), true);
    c.p.use_additive_bias = true;
    check_gate(c.name, "register_pv", rocke_unified_attn_enable_register_pv(&c.p, "gfx950"), false);

    c = make_case("sink wpe3", "gfx950", 64, 4, 16, 3, NA);
    c.p.head_size = 64;
    c.p.use_sinks = true;
    c.p.max_seqlen_q = 1024;
    run_case(&c);
    c.name = "sink wpe3 additive";
    c.waves_per_eu = 2;
    c.p.use_additive_bias = true;
    run_case(&c);

    c = make_case("small-q MHA", "gfx942", 64, 1, 16, NA, NA);
    c.p.num_query_heads = 8;
    c.p.max_seqlen_q = 512;
    c.p.total_q = 512;
    run_case(&c);
    c.name = "small-q MHA additive";
    c.num_warps = 2;
    c.p.use_additive_bias = true;
    run_case(&c);

    /* Arch-independent branch: use an arch no earlier rule claims. */
    c = make_case("Qwen block_m", "gfx90a", 64, 4, 32, NA, NA);
    c.p.head_size = 64;
    c.p.num_query_heads = 32;
    c.p.num_kv_heads = 4;
    c.p.max_seqlen_q = 1024;
    c.p.total_q = 1024;
    run_case(&c);
    c.name = "Qwen block_m additive";
    c.block_m_per_warp = 16;
    c.p.use_additive_bias = true;
    run_case(&c);
}

int main(void)
{
    test_shared_and_gfx950();
    test_gfx942();
    test_bias();
    test_gfx942_flash();
    test_fp8_mfma_qk();
    test_gfx942_spec();
    test_3d_selectors();

    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("attention_unified_selectors: all checks passed\n");
    return 0;
}
