// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Exercise public catalog queries against stored keys, without normalizing
 * away malformed table entries. Alias queries must find the same atom. */
#include "mma_family_index.h"
#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/wmma_scale_internal.h"

#include <atomic>
#include <initializer_list>
#include <stdio.h>
#include <string.h>
#include <thread>

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

static int test_family_index_first_use()
{
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::thread workers[8];
    bool passed[8] = {};
    for(int i = 0; i < 8; ++i)
        workers[i] = std::thread([&, i] {
            ++ready;
            while(!start.load())
                std::this_thread::yield();
            int count = 0;
            const auto* arches = rocke_known_arches(&count);
            bool ok = true;
            for(int a = 0; a < count; ++a)
            {
                const auto* arch = rocke_arch_target_from_gfx(arches[(a + i) % count]);
                for(int j = 0; j < arch->mma.num_ops; ++j)
                {
                    const auto& op = arch->mma.ops[j];
                    char copied_id[256];
                    snprintf(copied_id, sizeof(copied_id), "%s", op.op_id);
                    const char* family = rocke_arch_mma_op_id_family(copied_id);
                    ok &= family && strcmp(family, op.family) == 0;
                    ok &= family == rocke_arch_mma_op_id_family(op.op_id);
                }
            }
            passed[i] = ok && !rocke_arch_mma_op_id_family(NULL) && !rocke_arch_mma_op_id_family("")
                        && !rocke_arch_mma_op_id_family("unknown");
        });
    while(ready.load() != 8)
        std::this_thread::yield();
    start = true;
    for(auto& worker : workers)
        worker.join();
    for(bool ok : passed)
        CHECK(ok);
    return 0;
}

static int test_family_index_duplicates()
{
    rocke_mma_op_t ops[6] = {};
    const char* ids[] = {"same", "conflict", "z", "same", "conflict", "a"};
    const char* families[] = {"wmma", "mma", "wmma_scaled", "wmma", "wmma", "mma"};
    for(int i = 0; i < 6; ++i)
    {
        ops[i].op_id = ids[i];
        ops[i].family = families[i];
    }
    rocke_arch_target_t targets[2] = {};
    targets[0].mma = {ops, 3};
    targets[1].mma = {ops + 3, 3};
    const rocke_ati_arch_row_t registry[]
        = {{"first", &targets[0]}, {"missing", NULL}, {"second", &targets[1]}};
    const rocke_ati_mma_family_index index(registry, 3);
    CHECK(strcmp(index.lookup("same"), "wmma") == 0);
    CHECK(strcmp(index.lookup("a"), "mma") == 0);
    CHECK(strcmp(index.lookup("z"), "wmma_scaled") == 0);
    CHECK(!index.lookup(NULL) && !index.lookup("") && !index.lookup("between"));
    bool conflict = false;
    try
    {
        index.lookup("conflict");
    }
    catch(const ckc::Error& e)
    {
        conflict
            = e.code() == ROCKE_ERR_VALUE
              && strcmp(e.what(), "arch SSOT drift: op_id has inconsistent family across arches")
                     == 0;
    }
    CHECK(conflict);
    CHECK(strcmp(index.lookup("a"), "mma") == 0);
    const rocke_ati_mma_family_index empty(NULL, 0);
    CHECK(!empty.lookup("same"));
    return 0;
}

static int test_mma_result_names()
{
    const char* ids[] = {"mfma_f32_16x16x16_f16",
                         "wmma_f32_16x16x16_f16",
                         "wmma_gfx1250_f32_16x16x64_fp8_fp8",
                         "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32",
                         "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k16",
                         "wmma_gfx1250_f32_16x16x128_bf8_bf8_scale_e8m0_e8m0_k32",
                         "wmma_gfx1250_f32_16x16x128_bf8_bf8_scale_e8m0_e8m0_k16"};
    for(int i = 0; i < 7; ++i)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "mma_names") == ROCKE_OK);
        auto* value = rocke_b_const_i32(&b, 0);
        auto* result = rocke_b_mma(&b, ids[i], value, value, value, NULL, 0);
        CHECK(result && result->name);
        CHECK(strncmp(result->name, i < 3 ? "%acc" : "%mxacc", i < 3 ? 4 : 6) == 0);
        rocke_ir_builder_free(&b);
    }
    return 0;
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
    for(const char* dtype : {"e5m2", "fp8e5m2", "bf8e5m2", "bf8"})
    {
        for(int input = 0; input < 2; ++input)
        {
            rocke_mma_scale_filter_t scales = {"e8m0", "e8m0", ROCKE_MMA_SCALE_K32};
            (input == 0 ? scales.a_scale_dtype : scales.b_scale_dtype) = dtype;
            CHECK(rejects_query([&] {
                rocke_mma_catalog_enumerate(
                    &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, NULL, 0, &scales);
            }));
            CHECK(rejects_query([&] {
                rocke_mma_catalog_has_shape(
                    &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128, &scales);
            }));
            CHECK(rejects_query([&] {
                rocke_mma_catalog_op_for_shape(
                    &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, 128, &scales);
            }));
            CHECK(rejects_query([&] {
                rocke_mma_catalog_select_largest_k(
                    &cat, "wmma_scaled", "fp8", "bf8", "fp32", 16, 16, -1, &scales);
            }));
        }
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
    CHECK(packed_rows == 4 && scaled_rows == 6);
    return 0;
}

static int test_scale_layouts_and_families()
{
    CHECK(!rocke_arch_mma_op_id_family(NULL));
    CHECK(!rocke_arch_mma_op_id_family("unknown"));
    int num_arches = 0;
    const char* const* arches = rocke_known_arches(&num_arches);
    int scaled = 0;
    for(int i = 0; i < num_arches; ++i)
    {
        const auto* arch = rocke_arch_target_from_gfx(arches[i]);
        for(int j = 0; j < arch->mma.num_ops; ++j)
        {
            const auto* op = &arch->mma.ops[j];
            CHECK(strcmp(rocke_arch_mma_op_id_family(op->op_id), op->family) == 0);
            if(!op->a_scale_dtype)
            {
                CHECK(op->a_scale_frag_len == 0 && op->b_scale_frag_len == 0);
                CHECK(!op->a_scale_layout && !op->b_scale_layout);
                for(auto getter : {rocke_mma_op_a_scale_layout, rocke_mma_op_b_scale_layout})
                {
                    bool rejected = false;
                    try
                    {
                        getter(op, NULL);
                    }
                    catch(const ckc::Error& e)
                    {
                        rejected = e.code() == ROCKE_ERR_NOTIMPL;
                    }
                    CHECK(rejected);
                }
                continue;
            }
            ++scaled;
            CHECK(strcmp(arches[i], "gfx1250") == 0);
            const int count = op->scale_block_k == ROCKE_MMA_SCALE_K16 ? 8 : 4;
            CHECK(op->a_scale_frag_len == count && op->b_scale_frag_len == count);
            for(int source = 0; source < 2; ++source)
            {
                const auto* map = source ? rocke_mma_op_b_scale_layout(op, NULL)
                                         : rocke_mma_op_a_scale_layout(op, NULL);
                CHECK(map && map->frag_len == count && map->wave_size == 32);
                CHECK(map->role == (source ? ROCKE_MMA_ROLE_B_SCALE : ROCKE_MMA_ROLE_A_SCALE));
                for(int slot = -1; slot <= count; ++slot)
                {
                    rocke_ir_builder_t b;
                    CHECK(rocke_ir_builder_init(&b, "scale_layout") == ROCKE_OK);
                    auto* lane = rocke_b_thread_id_x(&b);
                    rocke_value_t *x = NULL, *y = NULL;
                    const bool ok = rocke_layout_map_coord(map, &b, lane, slot, &x, &y);
                    if(slot < 0 || slot == count)
                    {
                        CHECK(!ok && !x && !y && b.status == ROCKE_ERR_VALUE);
                        CHECK(strstr(b.err, source ? "'b_scale'" : "'a_scale'"));
                    }
                    else
                        CHECK(ok && x && y && b.status == ROCKE_OK);
                    rocke_ir_builder_free(&b);
                }
                auto invalid = *op;
                (source ? invalid.b_scale_frag_len : invalid.a_scale_frag_len) = count / 2;
                CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&invalid); }));
            }
        }
    }
    CHECK(scaled == 6);
    return 0;
}

int main()
{
    // Keep this first: no previous query may prime the shared family index.
    if(test_family_index_first_use() || test_family_index_duplicates() || test_mma_result_names()
       || test_scale_contracts() || test_scale_layouts_and_families())
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
                       || rocke_archtarget_op_for_shape(
                              arch, op->family, a, b, op->c_dtype, op->m, op->n, op->k, &scales)
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
