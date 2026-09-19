// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Exercise public catalog queries against stored keys, without normalizing
 * away malformed table entries. Alias queries must find the same atom. */
#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/wmma_scale_internal.h"

#include <initializer_list>
#include <stdio.h>
#include <string.h>

static const char* short_dtype(const char* dtype)
{
    const char* aliases[][2] = {{"fp8e4m3", "fp8"},
                                {"bf8e5m2", "bf8"},
                                {"fp6e2m3", "fp6"},
                                {"fp6e3m2", "bf6"},
                                {"fp4e2m1", "fp4"}};
    for(const auto& pair : aliases)
    {
        if(strcmp(dtype, pair[0]) == 0)
            return pair[1];
    }
    return dtype;
}

#define CHECK(condition)                                             \
    do                                                               \
    {                                                                \
        if(!(condition))                                             \
        {                                                            \
            fprintf(stderr, "line %d: %s\\n", __LINE__, #condition); \
            return 1;                                                \
        }                                                            \
    } while(0)

template <typename F>
static bool rejects_query(F query)
{
    try
    {
        query();
    }
    catch(const ckc::Error& e)
    {
        return e.code() == ROCKE_ERR_VALUE;
    }
    return false;
}

static int test_scale_contracts()
{
    const auto* arch = rocke_arch_target_from_gfx("gfx1250");
    const rocke_mma_scale_filter_t e8 = {"e8m0", "e8m0", ROCKE_MMA_SCALE_K32};
    const auto* base = rocke_mma_catalog_op_for_shape(
        &arch->mma, "wmma_scaled", "fp8", "fp8", "fp32", 16, 16, 128, &e8);
    CHECK(base);
    rocke_mma_op_t rows[8];
    rows[0] = *base;
    rows[0].b_dtype = "bf8e5m2";
    auto packing_atom = rows[0];
    packing_atom.b_frag_len = 8;
    const auto packing = rocke_scaled_wmma_contract(&packing_atom);
    CHECK(packing.matrix_formats[0] == 0 && packing.matrix_formats[1] == 1);
    CHECK(packing.matrix_words[0] == 16 && packing.matrix_words[1] == 8);
    CHECK(strstr(packing.intrinsic, "v16i32.v8i32"));
    for(int input = 0; input < 2; ++input)
    {
        auto unsupported = *base;
        (input == 0 ? unsupported.a_scale_dtype : unsupported.b_scale_dtype) = "e4m3";
        CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&unsupported); }));
    }
    int count = 1;
    for(int input = 0; input < 2; ++input)
    {
        for(const char* dtype : {"e4m3", "e5m3"})
        {
            rows[count] = rows[0];
            (input == 0 ? rows[count].a_scale_dtype : rows[count].b_scale_dtype) = dtype;
            ++count;
        }
    }
    rows[count] = rows[0];
    rows[count++].scale_block_k = ROCKE_MMA_SCALE_K16;
    rows[count] = rows[0];
    rows[count].a_scale_dtype = NULL;
    rows[count].b_scale_dtype = NULL;
    rows[count++].scale_block_k = ROCKE_MMA_SCALE_NONE;
    rocke_mma_catalog_t cat = {rows, count};
    for(int i = 0; i < count; ++i)
    {
        const auto& op = rows[i];
        const rocke_mma_scale_filter_t scales
            = {op.a_scale_dtype, op.b_scale_dtype, op.scale_block_k};
        CHECK(rocke_mma_catalog_enumerate(
                  &cat, "wmma_scaled", "fp8", "bf8", "f32", 16, 16, NULL, 0, &scales)
              == 1);
        CHECK(rocke_mma_catalog_has_shape(
            &cat, "wmma_scaled", "fp8", "bf8", "f32", 16, 16, 128, &scales));
        CHECK(rocke_mma_catalog_op_for_shape(
                  &cat, "wmma_scaled", "fp8", "bf8", "f32", 16, 16, 128, &scales)
              == &op);
        CHECK(rocke_mma_catalog_select_largest_k(
                  &cat, "wmma_scaled", "fp8", "bf8", "f32", 16, 16, -1, &scales)
              == &op);
        CHECK(!rocke_mma_catalog_select_largest_k(
            &cat, "wmma_scaled", "fp8", "bf8", "f32", 16, 16, 64, &scales));
    }
    CHECK(rocke_mma_catalog_enumerate(&cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, NULL, 0)
          == count);
    CHECK(rocke_mma_catalog_has_shape(&cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128));
    CHECK(rejects_query([&] {
        rocke_mma_catalog_op_for_shape(&cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128);
    }));
    CHECK(rejects_query([&] {
        rocke_mma_catalog_select_largest_k(&cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, -1);
    }));
    const rocke_mma_scale_filter_t unscaled = {NULL, NULL, ROCKE_MMA_SCALE_NONE};
    CHECK(rocke_mma_catalog_op_for_shape(
              &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128, &unscaled)
          == &rows[6]);
    const rocke_mma_scale_filter_t alias = {"fp8e4m3", "e8m0", ROCKE_MMA_SCALE_K32};
    CHECK(rocke_mma_catalog_op_for_shape(
              &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128, &alias)
          == &rows[1]);
    const rocke_mma_scale_filter_t alias_b = {"e8m0", "fp8e4m3", ROCKE_MMA_SCALE_K32};
    CHECK(rocke_mma_catalog_op_for_shape(
              &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128, &alias_b)
          == &rows[3]);
    rows[7] = rows[0];
    rows[7].k = 256;
    cat.num_ops = 8;
    CHECK(rocke_mma_catalog_select_largest_k(&cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, -1)
          == &rows[7]);
    CHECK(rejects_query([&] {
        rocke_mma_catalog_select_largest_k(&cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128);
    }));
    for(const auto scales :
        {rocke_mma_scale_filter_t{"i32", "e8m0", ROCKE_MMA_SCALE_K32},
         rocke_mma_scale_filter_t{"e8m0", "i32", ROCKE_MMA_SCALE_K32},
         rocke_mma_scale_filter_t{"e8m0", "e8m0", static_cast<rocke_mma_scale_block_k_t>(8)},
         rocke_mma_scale_filter_t{NULL, "e8m0", ROCKE_MMA_SCALE_K32},
         rocke_mma_scale_filter_t{"e8m0", NULL, ROCKE_MMA_SCALE_K32},
         rocke_mma_scale_filter_t{NULL, NULL, ROCKE_MMA_SCALE_K32},
         rocke_mma_scale_filter_t{"e8m0", "e8m0", ROCKE_MMA_SCALE_NONE}})
    {
        CHECK(rejects_query([&] {
            rocke_mma_catalog_enumerate(
                &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, NULL, 0, &scales);
        }));
    }
    const rocke_mma_scale_filter_t valid_missing = {"e5m3", "e5m3", ROCKE_MMA_SCALE_K32};
    CHECK(!rocke_mma_catalog_op_for_shape(
        &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128, &valid_missing));
    char id[256];
    int packed_rows = 0, scaled_rows = 0;
    for(int i = 0; i < arch->mma.num_ops; ++i)
    {
        const auto& op = arch->mma.ops[i];
        if(strncmp(op.op_id, "wmma_gfx1250_f32_16x16x64_", 25) == 0)
        {
            CHECK(op.a_frag_len == 8 && op.b_frag_len == 8 && op.c_frag_len == 8);
            ++packed_rows;
        }
        if(strcmp(op.family, "wmma_scaled") == 0)
        {
            snprintf(id,
                     sizeof(id),
                     "wmma_gfx1250_f32_16x16x128_%s_%s_scale_%s_%s_k%d",
                     short_dtype(op.a_dtype),
                     short_dtype(op.b_dtype),
                     op.a_scale_dtype,
                     op.b_scale_dtype,
                     op.scale_block_k);
            CHECK(strcmp(id, op.op_id) == 0);
            CHECK(rocke_mma_catalog_by_op_id(&arch->mma, id) == &op);
            ++scaled_rows;
        }
    }
    CHECK(packed_rows == 4 && scaled_rows == 4);
    return 0;
}

int main()
{
    if(test_scale_contracts())
        return 1;
    int checked = 0;
    for(const char* gfx : {"gfx950", "gfx1250"})
    {
        const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(gfx);
        if(!arch)
            return 1;
        for(int i = 0; i < arch->mma.num_ops; ++i)
        {
            const rocke_mma_op_t* op = &arch->mma.ops[i];
            for(const char* dtype : {op->a_dtype, op->b_dtype, op->c_dtype})
            {
                char scratch[64];
                if(strcmp(dtype, rocke_normalize_dtype(dtype, scratch, sizeof(scratch))) != 0)
                {
                    fprintf(stderr, "%s: noncanonical stored dtype %s\n", op->op_id, dtype);
                    return 1;
                }
            }
            const rocke_mma_scale_filter_t scales
                = {op->a_scale_dtype, op->b_scale_dtype, op->scale_block_k};
            for(const char* a : {op->a_dtype, short_dtype(op->a_dtype)})
            {
                for(const char* b : {op->b_dtype, short_dtype(op->b_dtype)})
                {
                    if(!rocke_mma_catalog_has_shape(
                           &arch->mma, op->family, a, b, op->c_dtype, op->m, op->n, op->k, &scales)
                       || rocke_mma_catalog_op_for_shape(&arch->mma,
                                                         op->family,
                                                         a,
                                                         b,
                                                         op->c_dtype,
                                                         op->m,
                                                         op->n,
                                                         op->k,
                                                         &scales)
                              != op
                       || rocke_mma_catalog_select_largest_k(&arch->mma,
                                                             op->family,
                                                             a,
                                                             b,
                                                             op->c_dtype,
                                                             op->m,
                                                             op->n,
                                                             op->k,
                                                             &scales)
                              != op
                       || rocke_mma_catalog_enumerate(&arch->mma,
                                                      op->family,
                                                      a,
                                                      b,
                                                      op->c_dtype,
                                                      op->m,
                                                      op->n,
                                                      NULL,
                                                      0,
                                                      &scales)
                              < 1)
                    {
                        fprintf(stderr, "%s: catalog query missed %s/%s\n", op->op_id, a, b);
                        return 1;
                    }
                }
            }
            ++checked;
        }
    }
    printf("catalog lookup: %d rows passed canonical and alias queries\n", checked);
    return 0;
}
