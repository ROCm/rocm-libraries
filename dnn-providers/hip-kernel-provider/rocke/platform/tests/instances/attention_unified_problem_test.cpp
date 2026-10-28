// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test for the C port of UnifiedAttentionProblem
 * (library/kernels/common/attention_unified.py): field defaults, the
 * @property helpers, the single problem struct shared by the scalar builders
 * and the selectors, and the selectors' explicit ``arch`` argument.
 */
#include <stdio.h>

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

static void test_defaults(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    CHECK(!p.use_additive_bias);
    CHECK(!p.fp8_fnuz);
    CHECK_EQ_INT(p.target_ctas, 0);
    CHECK(!p.causal_top_left);
    CHECK_EQ_INT(p.right_bound, 0);
    CHECK_EQ_INT(p.num_cus, 120);
}

static void test_causal_and_default_mask(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();

    CHECK(rocke_unified_attention_problem_causal(&p));
    CHECK(rocke_unified_attention_problem_default_mask(&p));

    p.causal_top_left = true;
    CHECK(rocke_unified_attention_problem_causal(&p));
    CHECK(!rocke_unified_attention_problem_default_mask(&p));

    p.causal_top_left = false;
    p.right_bound = -1;
    CHECK(!rocke_unified_attention_problem_causal(&p));
    CHECK(!rocke_unified_attention_problem_default_mask(&p));

    p.right_bound = 4;
    CHECK(!rocke_unified_attention_problem_causal(&p));
    CHECK(!rocke_unified_attention_problem_default_mask(&p));
}

/* Expected values computed from au.py:182-190 with total_q=100, num_seqs=3. */
static void test_total_num_q_blocks_upper_bound(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    p.total_q = 100;
    p.num_seqs = 3;
    p.num_kv_heads = 2;

    p.num_query_heads = 2; /* nqpk=1:  block_m=16, block_q=16 -> 6 + 3 */
    CHECK_EQ_INT(rocke_unified_attention_problem_total_num_q_blocks_upper_bound(&p), 9);

    p.num_query_heads = 8; /* nqpk=4:  block_m=16, block_q=4  -> 25 + 3 */
    CHECK_EQ_INT(rocke_unified_attention_problem_total_num_q_blocks_upper_bound(&p), 28);

    p.num_query_heads = 32; /* nqpk=16: block_m=16, block_q=1 -> 100 + 3 */
    CHECK_EQ_INT(rocke_unified_attention_problem_total_num_q_blocks_upper_bound(&p), 103);

    p.num_query_heads = 40; /* nqpk=20: block_m=next_pow2(20)=32, block_q=1 */
    CHECK_EQ_INT(rocke_unified_attention_problem_total_num_q_blocks_upper_bound(&p), 103);

    p.num_query_heads = 6; /* nqpk=3:  block_m=16, block_q=5  -> 20 + 3 */
    CHECK_EQ_INT(rocke_unified_attention_problem_total_num_q_blocks_upper_bound(&p), 23);

    p.num_query_heads = 3; /* non-divisible head ratio */
    CHECK_EQ_INT(rocke_unified_attention_problem_total_num_q_blocks_upper_bound(&p), -1);
}

static void test_effective_target_ctas(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    p.num_cus = 80;
    CHECK_EQ_INT(rocke_unified_attention_problem_effective_target_ctas(&p), 320);
    p.target_ctas = 500;
    CHECK_EQ_INT(rocke_unified_attention_problem_effective_target_ctas(&p), 500);
}

/* A bf16 d64 bs32 nqpk=8 multi-batch prefill problem: in the gfx950 combo
 * cohort, which every combo gate rejects on any other arch. */
static rocke_unified_attention_problem_t make_gfx950_combo_problem(void)
{
    rocke_unified_attention_problem_t p = rocke_unified_attention_problem_default();
    p.total_q = 2048;
    p.num_seqs = 4;
    p.num_query_heads = 64;
    p.num_kv_heads = 8;
    p.head_size = 64;
    p.block_size = 32;
    p.max_seqlen_q = 512;
    p.max_seqlen_k = 512;
    p.dtype = "bf16";
    return p;
}

/* One instance feeds both a scalar-builder helper and a selector. */
static void test_one_struct_for_builders_and_selectors(void)
{
    rocke_unified_attention_problem_t p = make_gfx950_combo_problem();
    CHECK_EQ_INT(rocke_unified_attention_problem_num_queries_per_kv(&p), 8);
    CHECK_EQ_INT(rocke_unified_attn_num_queries_per_kv(NULL, &p), 8);
    CHECK(rocke_unified_attn_select_2d_tile_size(&p, "gfx950") > 0);
}

static void test_selectors_use_explicit_arch(void)
{
    rocke_unified_attention_problem_t p = make_gfx950_combo_problem();
    rocke_attention_tiled_2d_spec_t s;

    CHECK(rocke_unified_attn_enable_combo_2d(&p, "gfx950"));
    CHECK(!rocke_unified_attn_enable_combo_2d(&p, "gfx942"));
    CHECK(!rocke_unified_attn_enable_combo_2d(&p, NULL));

    s = rocke_unified_attn_tiled_spec_from_problem(&p, "gfx950");
    CHECK(s.use_transposed_qk_32x32);
    CHECK(s.use_transposed_scalar_state);
}

int main(void)
{
    test_defaults();
    test_causal_and_default_mask();
    test_total_num_q_blocks_upper_bound();
    test_effective_target_ctas();
    test_one_struct_for_builders_and_selectors();
    test_selectors_use_explicit_arch();

    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("attention_unified_problem: all checks passed\n");
    return 0;
}
