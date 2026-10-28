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
    static const char* const knobs[] = {"HIPDNN_GFX942_FLASH_MLIM",
                                        "HIPDNN_GFX942_K_SLICED_RING",
                                        "HIPDNN_GFX942_K_SLICE_HD",
                                        "HIPDNN_GFX942_D128_SMALLTILE_DK",
                                        "HIPDNN_GFX942_BF16_CFVST",
                                        "HIPDNN_GFX942_FLASH_WIDE"};
    size_t i;

    for(i = 0; i < sizeof(knobs) / sizeof(knobs[0]); ++i)
        set_env(knobs[i], NULL);
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

    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("attention_unified_selectors: all checks passed\n");
    return 0;
}
