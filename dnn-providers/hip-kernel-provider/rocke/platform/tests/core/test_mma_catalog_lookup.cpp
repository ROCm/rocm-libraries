// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Exercise public catalog queries against stored keys, without normalizing
 * away malformed table entries. Alias queries must find the same atom. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <atomic>
#include <initializer_list>
#include <thread>

#include "mma_family_index.h"
#include "rocke/abi.h"
#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/wmma_scale_internal.h"

/* Layout canary. Growing or reordering rocke_mma_op_t changes the stride of the
 * ops[] array every catalog hands out, so a caller built against the old header
 * indexes it wrongly while still passing its own ABI check -- unless
 * ROCKE_ABI_VERSION moved with the struct. This pins the two together: editing
 * the struct fails the build here, and the fix is to bump ROCKE_ABI_VERSION in
 * rocke/abi.h and update both numbers in the same change. 64-bit only, where
 * the pointer width that is part of the layout is known. */
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(rocke_mma_op_t) == 160 && ROCKE_ABI_VERSION >= 3,
              "rocke_mma_op_t changed size: bump ROCKE_ABI_VERSION in rocke/abi.h, then "
              "update this pin");
#endif

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
        (input == 0 ? unsupported.a_scale_dtype : unsupported.b_scale_dtype) = "e5m2";
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
            const auto contract = rocke_scaled_wmma_contract(&op);
            CHECK(contract.scale_formats[0] >= 0 && contract.scale_formats[0] <= 2);
            CHECK(contract.scale_formats[1] >= 0 && contract.scale_formats[1] <= 2);
            auto invalid = op;
            invalid.k = 64;
            CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&invalid); }));
            ++scaled_rows;
        }
    }
    // Ordered identity fingerprint from the independently filtered 72c28a69 catalog.
    // Also exercises all generated static ID pointers after repeated queries.
    uint64_t fingerprint = UINT64_C(14695981039346656037);
    for(int i = 0; i < arch->mma.num_ops; ++i)
    {
        const char* text = arch->mma.ops[i].op_id;
        do
        {
            fingerprint ^= static_cast<unsigned char>(*text);
            fingerprint *= UINT64_C(1099511628211);
        } while(*text++);
    }
    CHECK(arch->mma.num_ops == 93);
    CHECK(fingerprint == UINT64_C(8996553530891217874));
    CHECK(packed_rows == 4 && scaled_rows == 86);
    return 0;
}

static int test_scale_layouts_and_families()
{
    CHECK(!rocke_arch_mma_op_id_family(NULL));
    CHECK(!rocke_arch_mma_op_id_family("unknown"));
    CHECK(!rocke_arch_mma_op_by_id(NULL));
    CHECK(!rocke_arch_mma_op_by_id("unknown"));
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
            /* The target-free lookup finds the same atom from its op_id alone. */
            const auto* found = rocke_arch_mma_op_by_id(op->op_id);
            CHECK(found && strcmp(found->op_id, op->op_id) == 0
                  && strcmp(found->family, op->family) == 0);
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
            if(strcmp(arches[i], "gfx950") == 0)
            {
                CHECK(strcmp(op->family, "mfma_scaled") == 0);
                CHECK(op->m == 32 && op->n == 32 && op->k == 64);
                CHECK(op->a_frag_len == 32 && op->b_frag_len == 32 && op->c_frag_len == 16);
                CHECK(op->scale_block_k == ROCKE_MMA_SCALE_K32);
                for(int source = 0; source < 2; ++source)
                {
                    const auto* map = source ? rocke_mma_op_b_scale_layout(op, NULL)
                                             : rocke_mma_op_a_scale_layout(op, NULL);
                    CHECK(map && map->frag_len == 1 && map->wave_size == 64);
                    CHECK(map->role == (source ? ROCKE_MMA_ROLE_B_SCALE : ROCKE_MMA_ROLE_A_SCALE));
                    for(int slot : {-1, 0, 1})
                    {
                        rocke_ir_builder_t b;
                        CHECK(rocke_ir_builder_init(&b, "mfma_scale_layout") == ROCKE_OK);
                        auto* lane = rocke_b_thread_id_x(&b);
                        rocke_value_t *x = NULL, *y = NULL;
                        const bool ok = rocke_layout_map_coord(map, &b, lane, slot, &x, &y);
                        CHECK(ok == (slot == 0));
                        CHECK(slot == 0 ? (x && y && b.status == ROCKE_OK)
                                        : (!x && !y && b.status == ROCKE_ERR_VALUE));
                        rocke_ir_builder_free(&b);
                    }
                }
                continue;
            }
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
    CHECK(scaled == 111);
    return 0;
}

static int test_scaled_wmma_supported_combinations()
{
    const auto* arch = rocke_arch_target_from_gfx("gfx1250");
    const char* inputs[] = {"fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1"};
    const char* scales[] = {"e8m0", "e5m3", "e4m3"};
    const rocke_mma_scale_filter_t e8 = {"e8m0", "e8m0", ROCKE_MMA_SCALE_K32};
    const auto* base = rocke_mma_catalog_op_for_shape(
        &arch->mma, "wmma_scaled", "fp8", "fp8", "fp32", 16, 16, 128, &e8);
    CHECK(base);
    int accepted = 0, rejected = 0;
    for(auto block : {ROCKE_MMA_SCALE_K32, ROCKE_MMA_SCALE_K16})
        for(int a = 0; a < 5; ++a)
            for(int b = 0; b < 5; ++b)
                for(int sa = 0; sa < 3; ++sa)
                    for(int sb = 0; sb < 3; ++sb)
                    {
                        // Independent oracle for LLVM commit 7597c0c938b3.
                        const bool legal = (sa == 0 || a == 4) && (sb == 0 || b == 4)
                                           && (!(a == 4 && b == 4) || sa == sb);
                        const rocke_mma_scale_filter_t filter = {scales[sa], scales[sb], block};
                        const auto* atom = rocke_mma_catalog_op_for_shape(&arch->mma,
                                                                          "wmma_scaled",
                                                                          inputs[a],
                                                                          inputs[b],
                                                                          "fp32",
                                                                          16,
                                                                          16,
                                                                          128,
                                                                          &filter);
                        CHECK((atom != NULL) == legal);
                        auto injected = *base;
                        injected.a_dtype = inputs[a];
                        injected.b_dtype = inputs[b];
                        injected.a_scale_dtype = scales[sa];
                        injected.b_scale_dtype = scales[sb];
                        injected.scale_block_k = block;
                        injected.a_scale_frag_len = injected.b_scale_frag_len = 128 / block;
                        if(legal)
                        {
                            rocke_scaled_wmma_contract(&injected);
                            ++accepted;
                        }
                        else
                        {
                            CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&injected); }));
                            ++rejected;
                        }
                    }
    CHECK(accepted == 86 && rejected == 364);
    return 0;
}

/* An emulated row restates the physical atom's fragment metadata -- this static
 * table has no build step to derive it, unlike the Python catalog. That makes
 * drift possible, so pin it: same frag lengths, same wave size, and the *same*
 * layout objects (pointer equality, not a copy). Also refuse an expansion chain:
 * the physical atom must itself be native, so one hop reaches real hardware. */
static int test_emulated_rows_mirror_physical()
{
    int checked = 0;
    for(const char* gfx : {"gfx942", "gfx950", "gfx1151", "gfx1201", "gfx1250"})
    {
        const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(gfx);
        if(!arch)
            continue;
        for(int i = 0; i < arch->mma.num_ops; ++i)
        {
            const rocke_mma_op_t* op = &arch->mma.ops[i];
            if(rocke_arch_mma_op_is_native(op))
                continue;
            const char* physical_id = rocke_arch_mma_op_physical_op_id(op);
            CHECK(physical_id != NULL);
            const rocke_mma_op_t* physical = NULL;
            for(int j = 0; j < arch->mma.num_ops; ++j)
            {
                if(strcmp(arch->mma.ops[j].op_id, physical_id) == 0)
                    physical = &arch->mma.ops[j];
            }
            if(!physical)
            {
                fprintf(
                    stderr, "%s: physical atom %s absent from %s\n", op->op_id, physical_id, gfx);
                return 1;
            }
            CHECK(rocke_arch_mma_op_is_native(physical));
            CHECK(op->m == physical->m && op->n == physical->n && op->k == physical->k);
            CHECK(op->a_frag_len == physical->a_frag_len);
            CHECK(op->b_frag_len == physical->b_frag_len);
            CHECK(op->c_frag_len == physical->c_frag_len);
            CHECK(op->wave_size == physical->wave_size);
            CHECK(op->a_layout == physical->a_layout);
            CHECK(op->b_layout == physical->b_layout);
            CHECK(op->c_layout == physical->c_layout);
            /* The bare-op_id path (rocke_b_mma's) reaches the same physical atom
             * and fragment as the row; the physical atom resolves to nothing. */
            const char* bare_physical = rocke_arch_mma_op_id_physical_op_id(op->op_id);
            CHECK(bare_physical != NULL && strcmp(bare_physical, physical_id) == 0);
            CHECK(rocke_arch_mma_c_frag_len(op->op_id) == physical->c_frag_len);
            CHECK(rocke_arch_mma_op_id_physical_op_id(physical_id) == NULL);
            /* The verifier's query answers with the same fragments, from the row. */
            int a_len = -1, b_len = -1, c_len = -1;
            CHECK(rocke_arch_mma_op_id_emulated_operands(op->op_id, &a_len, &b_len, &c_len));
            CHECK(a_len == physical->a_frag_len && b_len == physical->b_frag_len
                  && c_len == physical->c_frag_len);
            CHECK(!rocke_arch_mma_op_id_emulated_operands(physical_id, NULL, NULL, NULL));
            ++checked;
        }
    }
    CHECK(rocke_arch_mma_op_id_physical_op_id("not_a_real_op_id") == NULL);
    CHECK(rocke_arch_mma_op_id_physical_op_id(NULL) == NULL);
    CHECK(!rocke_arch_mma_op_id_emulated_operands("not_a_real_op_id", NULL, NULL, NULL));
    CHECK(!rocke_arch_mma_op_id_emulated_operands(NULL, NULL, NULL, NULL));
    printf("emulated rows: %d mirrored their physical atom\n", checked);
    return 0;
}

/* The C twin of tests/core/test_xf32_selection_regression.py: no query that
 * predates XF32 may reach an emulated atom, on any arch, through any entry
 * point.
 *
 * Every pre-existing row is native, so "the catalog as it stood before the
 * emulated rows" is just the native subset -- which means the expected answer
 * can be computed here rather than tabulated, and stays correct as native atoms
 * are added. enumerate() is checked on count *and* order, because instances
 * that take the first match would otherwise move silently. */
static int test_pre_existing_selection_is_native_only()
{
    int count = 0;
    const char* const* arches = rocke_known_arches(&count);
    int checked = 0;
    for(int g = 0; g < count; ++g)
    {
        const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(arches[g]);
        if(!arch)
            continue;
        for(int i = 0; i < arch->mma.num_ops; ++i)
        {
            const rocke_mma_op_t* op = &arch->mma.ops[i];
            if(!rocke_arch_mma_op_is_native(op))
                continue; /* an xf32 query is allowed to reach an emulated atom */
            const rocke_mma_scale_filter_t scales
                = {op->a_scale_dtype, op->b_scale_dtype, op->scale_block_k};

            /* enumerate: same atoms, same order, as filtering natives by hand. */
            const rocke_mma_op_t* got[64];
            int n = rocke_mma_catalog_enumerate(&arch->mma,
                                                op->family,
                                                op->a_dtype,
                                                op->b_dtype,
                                                op->c_dtype,
                                                -1,
                                                -1,
                                                got,
                                                (int)(sizeof(got) / sizeof(got[0])),
                                                &scales);
            CHECK(n >= 1 && n <= (int)(sizeof(got) / sizeof(got[0])));
            int seen = 0;
            for(int j = 0; j < arch->mma.num_ops; ++j)
            {
                const rocke_mma_op_t* cand = &arch->mma.ops[j];
                if(strcmp(cand->family, op->family) != 0 || strcmp(cand->a_dtype, op->a_dtype) != 0
                   || strcmp(cand->b_dtype, op->b_dtype) != 0
                   || strcmp(cand->c_dtype, op->c_dtype) != 0)
                    continue;
                if(cand->a_scale_dtype != op->a_scale_dtype
                   || cand->b_scale_dtype != op->b_scale_dtype
                   || cand->scale_block_k != op->scale_block_k)
                    continue;
                CHECK(rocke_arch_mma_op_is_native(cand));
                CHECK(seen < n && got[seen] == cand);
                ++seen;
            }
            CHECK(seen == n);

            /* The narrowing entry points resolve to the atom itself, unchanged. */
            CHECK(rocke_mma_catalog_has_shape(&arch->mma,
                                              op->family,
                                              op->a_dtype,
                                              op->b_dtype,
                                              op->c_dtype,
                                              op->m,
                                              op->n,
                                              op->k,
                                              &scales));
            const rocke_mma_op_t* exact = rocke_mma_catalog_op_for_shape(&arch->mma,
                                                                         op->family,
                                                                         op->a_dtype,
                                                                         op->b_dtype,
                                                                         op->c_dtype,
                                                                         op->m,
                                                                         op->n,
                                                                         op->k,
                                                                         &scales);
            CHECK(exact == op);
            const rocke_mma_op_t* largest = rocke_mma_catalog_select_largest_k(&arch->mma,
                                                                               op->family,
                                                                               op->a_dtype,
                                                                               op->b_dtype,
                                                                               op->c_dtype,
                                                                               op->m,
                                                                               op->n,
                                                                               -1,
                                                                               &scales);
            CHECK(largest != NULL && rocke_arch_mma_op_is_native(largest));

            /* Lookup by handle is order-sensitive; the new rows must not shadow. */
            CHECK(rocke_mma_catalog_by_op_id(&arch->mma, op->op_id) == op);
            ++checked;
        }
    }
    printf("pre-existing selection: %d native atoms unmoved\n", checked);
    return 0;
}

/* fp32 in must never select an emulated xf32 atom: XF32 is opt-in, and a caller
 * asking for fp32 gets the native fp32 MFMA or nothing. */
static int test_fp32_never_reaches_an_xf32_atom()
{
    int count = 0;
    const char* const* arches = rocke_known_arches(&count);
    for(int g = 0; g < count; ++g)
    {
        const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(arches[g]);
        if(!arch)
            continue;
        const rocke_mma_scale_filter_t unscaled = {NULL, NULL, ROCKE_MMA_SCALE_NONE};
        const rocke_mma_op_t* got[64];
        int n = rocke_mma_catalog_enumerate(&arch->mma,
                                            "mma",
                                            "fp32",
                                            "fp32",
                                            "fp32",
                                            -1,
                                            -1,
                                            got,
                                            (int)(sizeof(got) / sizeof(got[0])),
                                            &unscaled);
        CHECK(n <= (int)(sizeof(got) / sizeof(got[0])));
        for(int i = 0; i < n; ++i)
        {
            CHECK(rocke_arch_mma_op_is_native(got[i]));
            CHECK(strcmp(got[i]->a_dtype, "fp32") == 0);
        }
    }
    return 0;
}

/* Mirrors test_rdna_integer_wmma_rows_name_i8_and_i4: the instruction names keep
 * "iu8"/"iu4", the operand dtypes are i8/i4, and only i8/i4 select the op. */
static int test_rdna_integer_wmma_rows_name_i8_and_i4()
{
    const char* rows[][3]
        = {{"wmma_i32_16x16x16_iu8", "i8", "iu8"}, {"wmma_i32_16x16x16_iu4", "i4", "iu4"}};
    for(const char* gfx : {"gfx1151", "gfx11-generic"})
    {
        const auto* arch = rocke_arch_target_from_gfx(gfx);
        CHECK(arch);
        for(const auto& row : rows)
        {
            const auto* op = rocke_mma_catalog_by_op_id(&arch->mma, row[0]);
            CHECK(op && strcmp(op->a_dtype, row[1]) == 0 && strcmp(op->b_dtype, row[1]) == 0
                  && strcmp(op->c_dtype, "i32") == 0);
            CHECK(rocke_mma_catalog_op_for_shape(
                      &arch->mma, "wmma", row[1], row[1], "i32", 16, 16, 16)
                  == op);
            CHECK(rocke_mma_catalog_op_for_shape(
                      &arch->mma, "wmma", row[2], row[2], "i32", 16, 16, 16)
                  == NULL);
        }
    }
    return 0;
}

int main()
{
    // Keep this first: no previous query may prime the shared family index.
    if(test_family_index_first_use() || test_family_index_duplicates() || test_mma_result_names()
       || test_scale_contracts() || test_scale_layouts_and_families()
       || test_scaled_wmma_supported_combinations() || test_emulated_rows_mirror_physical()
       || test_pre_existing_selection_is_native_only() || test_fp32_never_reaches_an_xf32_atom()
       || test_rdna_integer_wmma_rows_name_i8_and_i4())
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
