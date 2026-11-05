// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/mma_result_dtype.cpp -- the C++ engine's MMA result dtype: the
 * dtype an instruction returns where it differs from its accumulator.
 *
 * Pins the same answers as tests/core/test_mma_result_dtype.py:
 * - on every known target, every catalog op's result dtype
 *   (rocke_mma_op_dst_dtype) is its accumulator's, except gfx1250
 *   wmma_gfx1250_bf16f32_16x16x32_bf16 (fp32 accumulator, bf16 result);
 * - the gfx1250 enumerate / has_shape / select_largest_k / op_for_shape
 *   answers with no result dtype, with "fp32" and with "bf16";
 * - the bare-op_id lookup rocke_arch_mma_op_id_dst_dtype, and the tile.mma
 *   result rocke_b_mma builds from the bare op_id (<8 x bf16>).
 *
 * Plain executable: returns non-zero when a check fails.
 */
#include <cstdio>
#include <cstring>

#include "rocke/arch_target.h"
#include "rocke/ir.h"

static int g_failures = 0;

#define CHECK(cond, what)                                                                 \
    do                                                                                    \
    {                                                                                     \
        if(!(cond))                                                                       \
        {                                                                                 \
            fprintf(stderr, "FAIL: %s: %s (%s:%d)\n", (what), #cond, __FILE__, __LINE__); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while(0)

static const char* const k_op_id = "wmma_gfx1250_bf16f32_16x16x32_bf16";
static const char* const k_f32_result_op_id = "wmma_gfx1250_f32_16x16x32_bf16";

static bool is(const rocke_mma_op_t* op, const char* op_id)
{
    return op != NULL && strcmp(op->op_id, op_id) == 0;
}

static void check_every_other_op_returns_its_accumulator_dtype()
{
    int count = 0;
    const char* const* arches = rocke_known_arches(&count);
    int differing = 0;
    for(int i = 0; i < count; ++i)
    {
        const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(arches[i]);
        CHECK(arch != NULL, arches[i]);
        if(!arch)
            continue;
        for(int j = 0; j < arch->mma.num_ops; ++j)
        {
            const rocke_mma_op_t* op = &arch->mma.ops[j];
            if(strcmp(rocke_mma_op_dst_dtype(op), op->c_dtype) == 0)
                continue;
            ++differing;
            CHECK(strcmp(arches[i], "gfx1250") == 0 && is(op, k_op_id), op->op_id);
        }
    }
    CHECK(differing == 1, "exactly one op returns a dtype other than its accumulator's");
}

static void check_the_row()
{
    const rocke_arch_target_t* arch = rocke_arch_target_from_gfx("gfx1250");
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&arch->mma, k_op_id);
    CHECK(op != NULL, k_op_id);
    if(!op)
        return;
    CHECK(strcmp(op->family, "wmma") == 0 && strcmp(op->a_dtype, "bf16") == 0
              && strcmp(op->b_dtype, "bf16") == 0 && strcmp(op->c_dtype, "fp32") == 0
              && strcmp(rocke_mma_op_dst_dtype(op), "bf16") == 0,
          k_op_id);
    CHECK(op->m == 16 && op->n == 16 && op->k == 32, k_op_id);
    CHECK(op->a_frag_len == 16 && op->b_frag_len == 16 && op->c_frag_len == 8
              && op->wave_size == 32,
          k_op_id);
    CHECK(op->a_layout == NULL && op->b_layout == NULL && op->c_layout == NULL, "no lane maps yet");
}

/* One query with result dtype `dst` (NULL: equal to the accumulator) must
 * select `expected` through every catalog entry point. */
static void check_query(const char* dst, const char* expected)
{
    const rocke_mma_catalog_t* cat = &rocke_arch_target_from_gfx("gfx1250")->mma;
    const char* what = dst ? dst : "(no result dtype)";
    const rocke_mma_op_t* found[4] = {NULL, NULL, NULL, NULL};
    const int total = rocke_mma_catalog_enumerate(
        cat, "wmma", "bf16", "bf16", "fp32", 16, 16, found, 4, NULL, dst);
    CHECK(total == 1 && is(found[0], expected), what);
    CHECK(is(rocke_mma_catalog_select_largest_k(
                 cat, "wmma", "bf16", "bf16", "fp32", 16, 16, -1, NULL, dst),
             expected),
          what);
    CHECK(is(rocke_mma_catalog_select_largest_k(
                 cat, "wmma", "bf16", "bf16", "fp32", 16, 16, 32, NULL, dst),
             expected),
          what);
    CHECK(is(rocke_mma_catalog_op_for_shape(
                 cat, "wmma", "bf16", "bf16", "fp32", 16, 16, 32, NULL, dst),
             expected),
          what);
    CHECK(rocke_mma_catalog_has_shape(cat, "wmma", "bf16", "bf16", "fp32", 16, 16, 32, NULL, dst),
          what);
    CHECK(!rocke_mma_catalog_has_shape(cat, "wmma", "bf16", "bf16", "fp32", 16, 16, 16, NULL, dst),
          what);
}

static void check_queries()
{
    check_query(NULL, k_f32_result_op_id);
    check_query("fp32", k_f32_result_op_id);
    check_query("bf16", k_op_id);
    const rocke_mma_catalog_t* cat = &rocke_arch_target_from_gfx("gfx1250")->mma;
    CHECK(rocke_mma_catalog_enumerate(
              cat, "wmma", "bf16", "bf16", "fp32", -1, -1, NULL, 0, NULL, "fp16")
              == 0,
          "a result dtype no instruction returns selects nothing");
}

static void check_bare_op_id()
{
    const char* dst = rocke_arch_mma_op_id_dst_dtype(k_op_id);
    CHECK(dst != NULL && strcmp(dst, "bf16") == 0, k_op_id);
    const char* f32 = rocke_arch_mma_op_id_dst_dtype(k_f32_result_op_id);
    CHECK(f32 != NULL && strcmp(f32, "fp32") == 0, k_f32_result_op_id);
    CHECK(rocke_arch_mma_op_id_dst_dtype("not_a_real_op_id") == NULL, "unknown op_id");

    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "bf16f32_result") == ROCKE_OK, k_op_id);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    rocke_value_t* d = rocke_b_mma(&b, k_op_id, zero, zero, zero, NULL, 0);
    CHECK(d && d->type->kind == ROCKE_TYPE_VECTOR && d->type->count == 8, k_op_id);
    if(d)
        CHECK(d->type->elem == rocke_dtype_to_ir_type("bf16"), k_op_id);
    rocke_ir_builder_free(&b);
}

int main(void)
{
    check_every_other_op_returns_its_accumulator_dtype();
    check_the_row();
    check_queries();
    check_bare_op_id();
    if(g_failures)
    {
        fprintf(stderr, "rocke_mma_result_dtype: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("rocke_mma_result_dtype: all checks passed.\n");
    return 0;
}
