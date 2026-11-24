// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Exercise public catalog queries against stored keys, without normalizing
 * away malformed table entries. Alias queries must find the same atom. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <initializer_list>
#include <thread>
#include <vector>

#include "mma_family_index.h"
#include "rocke/abi.h"
#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.mfma_attention.h"
#include "rocke/instance_gemm_internal.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/mma_operand_internal.h"
#include "rocke/wmma_scale_internal.h"

/* Layout canary. Growing or reordering rocke_mma_op_t changes the stride of the
 * ops[] array every catalog hands out. The ABI floor excludes libraries
 * predating indexed operands; unrelated later ABI bumps do not invalidate
 * these layout pins. A layout change requires reviewing the binary ABI and
 * updating the size/offset pins. 64-bit only, where the pointer width that is
 * part of the layout is known. */
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(rocke_mma_op_t) == 216 && ROCKE_ABI_VERSION >= 12,
              "rocke_mma_op_t layout or ABI floor mismatch: review rocke/abi.h and "
              "update this pin");
static_assert(offsetof(rocke_mma_op_t, srcs) == 8 && offsetof(rocke_mma_op_t, dst) == 152
                  && offsetof(rocke_mma_op_t, op_id) == 192
                  && offsetof(rocke_mma_op_t, emulation_physical_op_id) == 208,
              "rocke_mma_op_t fields moved: review the binary ABI and update this pin");
/* Same canary for rocke_memory_caps_t, which sits inside rocke_arch_target_t
 * ahead of its limits. */
static_assert(sizeof(rocke_memory_caps_t) == 20 && ROCKE_ABI_VERSION >= 4,
              "rocke_memory_caps_t changed size: bump ROCKE_ABI_VERSION in rocke/abi.h, then "
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

static int check_wmma_source_result_roles(
    const rocke_region_t* region, int source_width, int result_width, int* zeros, int* mmas)
{
    for(int i = 0; i < region->num_ops; ++i)
    {
        const auto* op = region->ops[i];
        if(strcmp(op->name, "arith.constant_vec") == 0
           && op->results[0]->type->elem->scalar == ROCKE_SCALAR_F32)
        {
            CHECK(op->results[0]->type->count == source_width);
            ++*zeros;
        }
        if(strcmp(op->name, "tile.mma") == 0)
        {
            CHECK(op->operands[2]->type->count == source_width);
            CHECK(op->results[0]->type->count == result_width);
            ++*mmas;
        }
        for(int r = 0; r < op->num_regions; ++r)
            CHECK(check_wmma_source_result_roles(
                      op->regions[r], source_width, result_width, zeros, mmas)
                  == 0);
    }
    return 0;
}

static int test_wmma_attention_accumulator_source_width()
{
    for(const char* gfx : {"gfx1151", "gfx11-generic", "gfx1201", "gfx12-generic"})
        for(const char* dtype : {"fp16", "bf16"})
        {
            auto target = *rocke_arch_target_from_gfx(gfx);
            char id[48];
            rocke_wmma_attn_op_id(&target, dtype, id, sizeof(id));
            auto atom = *rocke_mma_catalog_by_op_id(&target.mma, id);
            // Asymmetric metadata tests the builder role, not a new physical ABI.
            atom.srcs[2].frag_len = atom.dst.frag_len + 1;
            target.mma.ops = &atom;
            target.mma.num_ops = 1;
            rocke_ir_builder_t builder;
            CHECK(rocke_ir_builder_init(&builder, "wmma_operand_roles") == ROCKE_OK);
            rocke_mfma_attn_params_t p = {};
            const auto* elem = strcmp(dtype, "bf16") == 0 ? rocke_bf16() : rocke_f16();
            const auto* ptr = rocke_ptr_type(&builder, elem, "global");
            p.Q = rocke_b_param(&builder, "Q", ptr, NULL);
            p.K = rocke_b_param(&builder, "K", ptr, NULL);
            p.V = rocke_b_param(&builder, "V", ptr, NULL);
            p.O = rocke_b_param(&builder, "O", ptr, NULL);
            auto* zero = rocke_b_const_i32(&builder, 0);
            auto* stride = rocke_b_const_i32(&builder, 16);
            p.head_size = 16;
            p.seqlen_k = rocke_b_param(&builder, "n", rocke_i32(), NULL);
            p.q_tile_base = p.head_idx = p.kv_head_idx = zero;
            p.stride_q_token = p.stride_q_head = p.stride_k_token = p.stride_k_head = stride;
            p.stride_v_token = p.stride_v_head = p.stride_o_token = p.stride_o_head = stride;
            p.scale_log2 = rocke_b_param(&builder, "scale", rocke_f32(), NULL);
            p.dtype = dtype;
            p.arch = gfx;
            CHECK(rocke_wmma_attention_fwd_inner_body(&builder, &p, false, &target) == ROCKE_OK);
            int zeros = 0, mmas = 0;
            CHECK(check_wmma_source_result_roles(
                      builder.kernel->body, atom.srcs[2].frag_len, atom.dst.frag_len, &zeros, &mmas)
                  == 0);
            CHECK(zeros == 2 && mmas == 2);
            rocke_ir_builder_free(&builder);
        }
    return 0;
}

static int test_wmma_scatter_uses_destination_layout()
{
    const auto* target = rocke_arch_target_from_gfx("gfx1250");
    for(const char* id : {"wmma_gfx1250_f32_16x16x32_f16", "wmma_gfx1250_f32_16x16x32_bf16"})
    {
        const auto* original = rocke_mma_catalog_by_op_id(&target->mma, id);
        CHECK(original != NULL);
        const auto* storage
            = strcmp(original->srcs[0].dtype, "bf16") == 0 ? rocke_bf16() : rocke_f16();
        // Keep the physical ABI unchanged; only map availability differs.
        for(int missing = 0; missing < 3; ++missing)
        {
            auto atom = *original;
            if(missing == 1)
                atom.srcs[2].layout = NULL;
            if(missing == 2)
                atom.dst.layout = NULL;
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "wmma_result_scatter") == ROCKE_OK);
            rocke_gemm_universal_spec_t spec = {};
            spec.tile.tile_m = spec.tile.warp_tile_m = atom.m;
            spec.tile.tile_n = spec.tile.warp_tile_n = atom.n;
            spec.tile.warp_m = spec.tile.warp_n = 1;
            auto* lane = rocke_b_param(&b, "lane", rocke_i32(), NULL);
            auto* zero = rocke_b_const_i32(&b, 0);
            rocke_value_t* accs[] = {rocke_b_zero_vec_f32(&b, atom.dst.frag_len)};
            int visits = 0;
            bool rejected = false;
            try
            {
                rocke_gemm_emit_wmma_acc_scatter(
                    &b,
                    &spec,
                    &atom,
                    lane,
                    accs,
                    1,
                    zero,
                    zero,
                    atom.dst.frag_len,
                    storage,
                    [](rocke_ir_builder_t*,
                       rocke_value_t*,
                       rocke_value_t*,
                       rocke_value_t*,
                       int,
                       void* user) { ++*static_cast<int*>(user); },
                    &visits);
            }
            catch(const ckc::Error& e)
            {
                CHECK(missing == 2);
                CHECK(e.code() == ROCKE_ERR_NOTIMPL);
                CHECK(strstr(e.what(), "no verified 'dst' layout map") != NULL);
                CHECK(strstr(e.what(), id) != NULL);
                rejected = true;
            }
            if(missing == 2)
            {
                CHECK(rejected && visits == 0);
            }
            else
            {
                CHECK(!rejected && rocke_ir_builder_ok(&b));
                CHECK(visits == atom.dst.frag_len);
            }
            rocke_ir_builder_free(&b);
        }
    }
    return 0;
}

/* Independent native sequences pinned before integration. Native gfx950/
 * gfx1250 ordering differs from Python; preserve each contract. */
static const char* const order_gfx90a[] = {
    "mfma_f32_16x16x16_f16",
    "mfma_f32_32x32x8_f16",
    "mfma_f32_16x16x16_bf16",
    "mfma_f32_32x32x8_bf16",
    "mfma_f32_16x16x8_bf16",
    "mfma_f32_32x32x4_bf16",
    "mfma_f32_16x16x4_f32",
    "mfma_f32_32x32x2_f32",
    "mfma_i32_16x16x16_i8",
    "mfma_i32_32x32x8_i8",
    "mfma_f64_16x16x4_f64",
};
static const char* const order_gfx942[] = {
    "mfma_f32_16x16x8_xf32",     "mfma_f32_32x32x4_xf32",     "mfma_f32_16x16x4_f32",
    "mfma_f32_32x32x2_f32",      "mfma_f64_16x16x4_f64",      "mfma_f64_4x4x4_f64",
    "mfma_f32_16x16x16_f16",     "mfma_f32_32x32x8_f16",      "mfma_f32_16x16x16_bf16",
    "mfma_f32_32x32x8_bf16",     "mfma_f32_16x16x32_fp8",     "mfma_f32_32x32x16_fp8",
    "mfma_f32_16x16x32_bf8",     "mfma_f32_32x32x16_bf8",     "mfma_i32_16x16x32_i8",
    "mfma_i32_32x32x16_i8",      "mfma_f32_16x16x32_fp8_bf8", "mfma_f32_16x16x32_bf8_fp8",
    "mfma_f32_32x32x16_fp8_bf8", "mfma_f32_32x32x16_bf8_fp8",
};
static const char* const order_gfx950[] = {
    "mfma_gfx950_f32_32x32x64_fp8_fp8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp8_bf8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp8_fp6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp8_bf6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp8_fp4_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf8_fp8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf8_bf8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf8_fp6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf8_bf6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf8_fp4_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp6_fp8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp6_bf8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp6_fp6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp6_bf6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp6_fp4_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf6_fp8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf6_bf8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf6_fp6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf6_bf6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_bf6_fp4_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp4_fp8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp4_bf8_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp4_fp6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp4_bf6_scale_e8m0_e8m0_k32",
    "mfma_gfx950_f32_32x32x64_fp4_fp4_scale_e8m0_e8m0_k32",
    "mfma_f32_16x16x4_f32",
    "mfma_f32_32x32x2_f32",
    "mfma_f32_16x16x16_f16",
    "mfma_f32_16x16x32_f16",
    "mfma_f32_32x32x8_f16",
    "mfma_f32_32x32x16_f16",
    "mfma_f32_16x16x16_bf16",
    "mfma_f32_16x16x32_bf16",
    "mfma_f32_32x32x8_bf16",
    "mfma_f32_32x32x16_bf16",
    "mfma_f32_16x16x32_fp8",
    "mfma_f32_32x32x16_fp8",
    "mfma_f32_16x16x32_bf8",
    "mfma_f32_32x32x16_bf8",
    "mfma_i32_16x16x32_i8",
    "mfma_i32_32x32x16_i8",
    "mfma_f64_16x16x4_f64",
    "mfma_f32_16x16x32_fp8_bf8",
    "mfma_f32_16x16x32_bf8_fp8",
    "mfma_f32_32x32x16_fp8_bf8",
    "mfma_f32_32x32x16_bf8_fp8",
    "mfma_i32_16x16x64_i8",
    "mfma_i32_32x32x32_i8",
    "mfma_f32_16x16x32_xf32",
    "mfma_f32_32x32x16_xf32",
    "mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp6e2m3",
    "mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp6e3m2",
    "mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1",
    "mfma_f32_16x16x128_f8f6f4_bf8e5m2_fp6e2m3",
    "mfma_f32_16x16x128_f8f6f4_bf8e5m2_fp6e3m2",
    "mfma_f32_16x16x128_f8f6f4_bf8e5m2_fp4e2m1",
    "mfma_f32_16x16x128_f8f6f4_fp6e2m3_fp8e4m3",
    "mfma_f32_16x16x128_f8f6f4_fp6e2m3_bf8e5m2",
    "mfma_f32_16x16x128_f8f6f4_fp6e2m3_fp6e2m3",
    "mfma_f32_16x16x128_f8f6f4_fp6e2m3_fp6e3m2",
    "mfma_f32_16x16x128_f8f6f4_fp6e2m3_fp4e2m1",
    "mfma_f32_16x16x128_f8f6f4_fp6e3m2_fp8e4m3",
    "mfma_f32_16x16x128_f8f6f4_fp6e3m2_bf8e5m2",
    "mfma_f32_16x16x128_f8f6f4_fp6e3m2_fp6e2m3",
    "mfma_f32_16x16x128_f8f6f4_fp6e3m2_fp6e3m2",
    "mfma_f32_16x16x128_f8f6f4_fp6e3m2_fp4e2m1",
    "mfma_f32_16x16x128_f8f6f4_fp4e2m1_fp8e4m3",
    "mfma_f32_16x16x128_f8f6f4_fp4e2m1_bf8e5m2",
    "mfma_f32_16x16x128_f8f6f4_fp4e2m1_fp6e2m3",
    "mfma_f32_16x16x128_f8f6f4_fp4e2m1_fp6e3m2",
    "mfma_f32_16x16x128_f8f6f4_fp4e2m1_fp4e2m1",
    "mfma_f32_32x32x64_f8f6f4_fp8e4m3_fp6e2m3",
    "mfma_f32_32x32x64_f8f6f4_fp8e4m3_fp6e3m2",
    "mfma_f32_32x32x64_f8f6f4_fp8e4m3_fp4e2m1",
    "mfma_f32_32x32x64_f8f6f4_bf8e5m2_fp6e2m3",
    "mfma_f32_32x32x64_f8f6f4_bf8e5m2_fp6e3m2",
    "mfma_f32_32x32x64_f8f6f4_bf8e5m2_fp4e2m1",
    "mfma_f32_32x32x64_f8f6f4_fp6e2m3_fp8e4m3",
    "mfma_f32_32x32x64_f8f6f4_fp6e2m3_bf8e5m2",
    "mfma_f32_32x32x64_f8f6f4_fp6e2m3_fp6e2m3",
    "mfma_f32_32x32x64_f8f6f4_fp6e2m3_fp6e3m2",
    "mfma_f32_32x32x64_f8f6f4_fp6e2m3_fp4e2m1",
    "mfma_f32_32x32x64_f8f6f4_fp6e3m2_fp8e4m3",
    "mfma_f32_32x32x64_f8f6f4_fp6e3m2_bf8e5m2",
    "mfma_f32_32x32x64_f8f6f4_fp6e3m2_fp6e2m3",
    "mfma_f32_32x32x64_f8f6f4_fp6e3m2_fp6e3m2",
    "mfma_f32_32x32x64_f8f6f4_fp6e3m2_fp4e2m1",
    "mfma_f32_32x32x64_f8f6f4_fp4e2m1_fp8e4m3",
    "mfma_f32_32x32x64_f8f6f4_fp4e2m1_bf8e5m2",
    "mfma_f32_32x32x64_f8f6f4_fp4e2m1_fp6e2m3",
    "mfma_f32_32x32x64_f8f6f4_fp4e2m1_fp6e3m2",
    "mfma_f32_32x32x64_f8f6f4_fp4e2m1_fp4e2m1",
    "mfma_f32_16x16x128_fp8",
    "mfma_f32_16x16x128_f8f6f4_fp8e4m3_bf8e5m2",
    "mfma_f32_16x16x128_f8f6f4_bf8e5m2_fp8e4m3",
    "mfma_f32_16x16x128_f8f6f4_bf8e5m2_bf8e5m2",
    "mfma_f32_32x32x64_f8f6f4_fp8e4m3_fp8e4m3",
    "mfma_f32_32x32x64_f8f6f4_fp8e4m3_bf8e5m2",
    "mfma_f32_32x32x64_f8f6f4_bf8e5m2_fp8e4m3",
    "mfma_f32_32x32x64_f8f6f4_bf8e5m2_bf8e5m2",
};
static const char* const order_gfx1151[] = {
    "wmma_f32_16x16x16_f16",
    "wmma_f32_16x16x16_bf16",
    "wmma_f16_16x16x16_f16",
    "wmma_bf16_16x16x16_bf16",
    "wmma_i32_16x16x16_iu8",
    "wmma_i32_16x16x16_iu8_u8_i8",
    "wmma_i32_16x16x16_iu8_i8_u8",
    "wmma_i32_16x16x16_iu8_u8_u8",
    "wmma_i32_16x16x16_iu4",
    "wmma_i32_16x16x16_iu4_u4_i4",
    "wmma_i32_16x16x16_iu4_i4_u4",
    "wmma_i32_16x16x16_iu4_u4_u4",
};
static const char* const order_gfx1201[] = {
    "wmma_gfx12_f32_16x16x16_f16",       "wmma_gfx12_f32_16x16x16_bf16",
    "wmma_gfx12_f32_16x16x16_fp8_fp8",   "wmma_gfx12_f32_16x16x16_fp8_bf8",
    "wmma_gfx12_f32_16x16x16_bf8_fp8",   "wmma_gfx12_f32_16x16x16_bf8_bf8",
    "wmma_gfx12_i32_16x16x16_iu8",       "wmma_gfx12_i32_16x16x16_iu8_u8_i8",
    "wmma_gfx12_i32_16x16x16_iu8_i8_u8", "wmma_gfx12_i32_16x16x16_iu8_u8_u8",
    "wmma_gfx12_i32_16x16x16_iu4",       "wmma_gfx12_i32_16x16x16_iu4_u4_i4",
    "wmma_gfx12_i32_16x16x16_iu4_i4_u4", "wmma_gfx12_i32_16x16x16_iu4_u4_u4",
    "wmma_gfx12_i32_16x16x32_iu4",       "wmma_gfx12_i32_16x16x32_iu4_u4_i4",
    "wmma_gfx12_i32_16x16x32_iu4_i4_u4", "wmma_gfx12_i32_16x16x32_iu4_u4_u4",
    "wmma_gfx12_f16_16x16x16_f16",       "wmma_gfx12_bf16_16x16x16_bf16",
};
static const char* const order_gfx1250[] = {
    "wmma_gfx1250_f32_16x16x4_f32",
    "wmma_gfx1250_f32_16x16x32_f16",
    "wmma_gfx1250_f32_16x16x32_bf16",
    "wmma_gfx1250_bf16f32_16x16x32_bf16",
    "wmma_gfx1250_f16_16x16x32_f16",
    "wmma_gfx1250_bf16_16x16x32_bf16",
    "wmma_gfx1250_f32_16x16x64_fp8_fp8",
    "wmma_gfx1250_f32_16x16x64_fp8_bf8",
    "wmma_gfx1250_f32_16x16x64_bf8_fp8",
    "wmma_gfx1250_f32_16x16x64_bf8_bf8",
    "wmma_gfx1250_f32_16x16x128_fp8_fp8",
    "wmma_gfx1250_f32_16x16x128_fp8_bf8",
    "wmma_gfx1250_f32_16x16x128_bf8_fp8",
    "wmma_gfx1250_f32_16x16x128_bf8_bf8",
    "wmma_gfx1250_f16_16x16x64_fp8_fp8",
    "wmma_gfx1250_f16_16x16x64_fp8_bf8",
    "wmma_gfx1250_f16_16x16x64_bf8_fp8",
    "wmma_gfx1250_f16_16x16x64_bf8_bf8",
    "wmma_gfx1250_f16_16x16x128_fp8_fp8",
    "wmma_gfx1250_f16_16x16x128_fp8_bf8",
    "wmma_gfx1250_f16_16x16x128_bf8_fp8",
    "wmma_gfx1250_f16_16x16x128_bf8_bf8",
    "wmma_gfx1250_i32_16x16x64_iu8",
    "wmma_gfx1250_i32_16x16x64_iu8_u8_i8",
    "wmma_gfx1250_i32_16x16x64_iu8_i8_u8",
    "wmma_gfx1250_i32_16x16x64_iu8_u8_u8",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp8e4m3_fp6e2m3",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp8e4m3_fp6e3m2",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_bf8e5m2_fp6e2m3",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_bf8e5m2_fp6e3m2",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_bf8e5m2_fp4e2m1",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e2m3_fp8e4m3",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e2m3_bf8e5m2",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e2m3_fp6e2m3",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e2m3_fp6e3m2",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e2m3_fp4e2m1",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e3m2_fp8e4m3",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e3m2_bf8e5m2",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e3m2_fp6e2m3",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e3m2_fp6e3m2",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e3m2_fp4e2m1",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp4e2m1_fp8e4m3",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp4e2m1_bf8e5m2",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp4e2m1_fp6e2m3",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp4e2m1_fp6e3m2",
    "wmma_gfx1250_f32_16x16x128_f8f6f4_fp4e2m1_fp4e2m1",
    "wmma_gfx1250_f32_32x16x128_f4",
    "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp4_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp6_fp6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf6_bf6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf8_bf8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp4_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp6_fp6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf6_bf6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf8_bf8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp8_bf8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp8_fp6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp8_bf6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp8_fp4_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf8_fp8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf8_fp6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf8_bf6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf8_fp4_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp6_fp8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp6_bf8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp6_bf6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp6_fp4_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf6_fp8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf6_bf8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf6_fp6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_bf6_fp4_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_bf8_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_bf6_scale_e8m0_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp8_bf8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp8_fp6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp8_bf6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp8_fp4_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf8_fp8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf8_fp6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf8_bf6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf8_fp4_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp6_fp8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp6_bf8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp6_bf6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp6_fp4_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf6_fp8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf6_bf8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf6_fp6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_bf6_fp4_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_bf8_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_bf6_scale_e8m0_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp8_fp4_scale_e8m0_e5m3_k32",
    "wmma_gfx1250_f32_16x16x128_fp8_fp4_scale_e8m0_e4m3_k32",
    "wmma_gfx1250_f32_16x16x128_bf8_fp4_scale_e8m0_e5m3_k32",
    "wmma_gfx1250_f32_16x16x128_bf8_fp4_scale_e8m0_e4m3_k32",
    "wmma_gfx1250_f32_16x16x128_fp6_fp4_scale_e8m0_e5m3_k32",
    "wmma_gfx1250_f32_16x16x128_fp6_fp4_scale_e8m0_e4m3_k32",
    "wmma_gfx1250_f32_16x16x128_bf6_fp4_scale_e8m0_e5m3_k32",
    "wmma_gfx1250_f32_16x16x128_bf6_fp4_scale_e8m0_e4m3_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp8_scale_e5m3_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp8_scale_e4m3_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_bf8_scale_e5m3_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_bf8_scale_e4m3_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp6_scale_e5m3_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp6_scale_e4m3_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_bf6_scale_e5m3_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_bf6_scale_e4m3_e8m0_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp4_scale_e5m3_e5m3_k32",
    "wmma_gfx1250_f32_16x16x128_fp4_fp4_scale_e4m3_e4m3_k32",
    "wmma_gfx1250_f32_16x16x128_fp8_fp4_scale_e8m0_e5m3_k16",
    "wmma_gfx1250_f32_16x16x128_fp8_fp4_scale_e8m0_e4m3_k16",
    "wmma_gfx1250_f32_16x16x128_bf8_fp4_scale_e8m0_e5m3_k16",
    "wmma_gfx1250_f32_16x16x128_bf8_fp4_scale_e8m0_e4m3_k16",
    "wmma_gfx1250_f32_16x16x128_fp6_fp4_scale_e8m0_e5m3_k16",
    "wmma_gfx1250_f32_16x16x128_fp6_fp4_scale_e8m0_e4m3_k16",
    "wmma_gfx1250_f32_16x16x128_bf6_fp4_scale_e8m0_e5m3_k16",
    "wmma_gfx1250_f32_16x16x128_bf6_fp4_scale_e8m0_e4m3_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp8_scale_e5m3_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp8_scale_e4m3_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_bf8_scale_e5m3_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_bf8_scale_e4m3_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp6_scale_e5m3_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp6_scale_e4m3_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_bf6_scale_e5m3_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_bf6_scale_e4m3_e8m0_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp4_scale_e5m3_e5m3_k16",
    "wmma_gfx1250_f32_16x16x128_fp4_fp4_scale_e4m3_e4m3_k16",
};

static bool catalog_order_matches(const rocke_mma_catalog_t* catalog,
                                  const char* const* expected,
                                  size_t count)
{
    if((size_t)catalog->num_ops != count)
        return false;
    for(size_t i = 0; i < count; ++i)
        if(strcmp(catalog->ops[i].op_id, expected[i]) != 0)
            return false;
    return true;
}

static int test_catalog_keeps_established_order()
{
    struct expected_catalog
    {
        const char* gfx;
        const char* const* ids;
        size_t count;
    };
#define ORDER(gfx, ids)                        \
    {                                          \
        gfx, ids, sizeof(ids) / sizeof(ids[0]) \
    }
    const expected_catalog expected[] = {
        ORDER("gfx90a", order_gfx90a),
        ORDER("gfx942", order_gfx942),
        ORDER("gfx950", order_gfx950),
        ORDER("gfx1151", order_gfx1151),
        ORDER("gfx1201", order_gfx1201),
        ORDER("gfx1250", order_gfx1250),
        ORDER("gfx11-generic", order_gfx1151),
        ORDER("gfx12-generic", order_gfx1201),
    };
#undef ORDER
    for(const auto& row : expected)
    {
        const auto* arch = rocke_arch_target_from_gfx(row.gfx);
        if(arch && !catalog_order_matches(&arch->mma, row.ids, row.count))
        {
            fprintf(stderr,
                    "%s: expected %zu catalog rows, got %d\n",
                    row.gfx,
                    row.count,
                    arch->mma.num_ops);
            for(int i = 0; i < arch->mma.num_ops; ++i)
                fprintf(stderr, "  %d: %s\n", i, arch->mma.ops[i].op_id);
        }
        CHECK(arch && catalog_order_matches(&arch->mma, row.ids, row.count));
        CHECK(row.count >= 2);
        std::vector<rocke_mma_op_t> swapped(arch->mma.ops, arch->mma.ops + arch->mma.num_ops);
        const auto first = swapped[0];
        swapped[0] = swapped[1];
        swapped[1] = first;
        auto catalog = arch->mma;
        catalog.ops = swapped.data();
        CHECK(!catalog_order_matches(&catalog, row.ids, row.count));
    }
    return 0;
}

static int test_required_indexed_dtypes()
{
    const auto* arch = rocke_arch_target_from_gfx("gfx942");
    CHECK(arch);
    for(int missing = 0; missing < 4; ++missing)
    {
        const char* srcs[] = {"fp32", "fp32", "fp32"};
        const char* dst = "fp32";
        if(missing < 3)
            srcs[missing] = nullptr;
        const char* const* sources = missing == 3 ? nullptr : srcs;
        CHECK(rejects_query([&] {
            rocke_mma_catalog_enumerate_indexed(
                &arch->mma, nullptr, sources, dst, 16, 16, nullptr, 0);
        }));
        CHECK(rejects_query([&] {
            rocke_mma_catalog_has_shape_indexed(&arch->mma, nullptr, sources, dst, 16, 16, 4);
        }));
        CHECK(rejects_query([&] {
            rocke_mma_catalog_select_largest_k_indexed(
                &arch->mma, nullptr, sources, dst, 16, 16, -1);
        }));
        CHECK(rejects_query([&] {
            rocke_mma_catalog_op_for_shape_indexed(&arch->mma, nullptr, sources, dst, 16, 16, 4);
        }));
        CHECK(rejects_query(
            [&] { rocke_arch_supports_dtype_combo_indexed(arch, sources, dst, nullptr); }));
    }
    const char* srcs[] = {"fp32", "fp32", "fp32"};
    CHECK(rocke_arch_supports_dtype_combo_indexed(arch, srcs, nullptr, nullptr));
    CHECK(rocke_mma_catalog_op_for_shape_indexed(&arch->mma, nullptr, srcs, "fp32", 16, 16, 4));
    CHECK(!rocke_mma_catalog_op_for_shape_indexed(&arch->mma, nullptr, srcs, "", 16, 16, 4));
    CHECK(rejects_query([&] {
        rocke_mma_catalog_op_for_shape(&arch->mma, nullptr, nullptr, "fp32", "fp32", 16, 16, 4);
    }));
    return 0;
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
    auto compatibility_op = *base;
    rocke_layout_map_t src2_layout = {}, dst_layout = {};
    compatibility_op.srcs[2].layout = &src2_layout;
    compatibility_op.dst.layout = &dst_layout;
    CHECK(rocke_mmaop_c_layout(&compatibility_op, NULL) == &src2_layout);
    CHECK(rocke_mmaop_acc_layout(&compatibility_op, NULL) == &src2_layout);
    CHECK(rocke_mma_op_dst_layout(&compatibility_op, NULL) == &dst_layout);
    // Indexed queries must match both the independent result and both scales.
    rocke_mma_op_t indexed_rows[3] = {*base, *base, *base};
    indexed_rows[1].dst.dtype = indexed_rows[2].dst.dtype = "i32";
    indexed_rows[2].srcs[0].scale_block_size = ROCKE_MMA_SCALE_K16;
    indexed_rows[2].srcs[1].scale_block_size = ROCKE_MMA_SCALE_K16;
    const char* sources[] = {"fp8", "fp8", "fp32"};
    const rocke_mma_catalog_t indexed = {indexed_rows, 3};
    CHECK(rocke_mma_catalog_enumerate_indexed(
              &indexed, "wmma_scaled", sources, nullptr, 16, 16, NULL, 0, &e8)
          == 1);
    CHECK(rocke_mma_catalog_has_shape_indexed(
        &indexed, "wmma_scaled", sources, nullptr, 16, 16, 128, &e8));
    CHECK(rocke_mma_catalog_op_for_shape_indexed(
              &indexed, "wmma_scaled", sources, nullptr, 16, 16, 128, &e8)
          == &indexed_rows[0]);
    CHECK(rocke_mma_catalog_select_largest_k_indexed(
              &indexed, "wmma_scaled", sources, nullptr, 16, 16, -1, &e8)
          == &indexed_rows[0]);
    CHECK(rocke_mma_catalog_enumerate(
              &indexed, "wmma_scaled", sources[0], sources[1], sources[2], 16, 16, NULL, 0, &e8)
          == 1);
    CHECK(rocke_mma_catalog_has_shape(
        &indexed, "wmma_scaled", sources[0], sources[1], sources[2], 16, 16, 128, &e8));
    CHECK(rocke_mma_catalog_op_for_shape(
              &indexed, "wmma_scaled", sources[0], sources[1], sources[2], 16, 16, 128, &e8)
          == &indexed_rows[0]);
    CHECK(rocke_mma_catalog_select_largest_k(
              &indexed, "wmma_scaled", sources[0], sources[1], sources[2], 16, 16, -1, &e8)
          == &indexed_rows[0]);
    rocke_arch_target_t synthetic = *arch;
    synthetic.mma = {&indexed_rows[1], 1};
    CHECK(!rocke_arch_supports_dtype_combo_indexed(&synthetic, sources, nullptr, "wmma_scaled"));
    CHECK(rocke_arch_supports_dtype_combo_indexed(&synthetic, sources, "i32", "wmma_scaled"));
    CHECK(rocke_mma_catalog_enumerate_indexed(
              &indexed, "wmma_scaled", sources, "i32", 16, 16, NULL, 0, &e8)
          == 1);
    CHECK(rocke_mma_catalog_has_shape_indexed(
        &indexed, "wmma_scaled", sources, "i32", 16, 16, 128, &e8));
    CHECK(rocke_mma_catalog_op_for_shape_indexed(
              &indexed, "wmma_scaled", sources, "i32", 16, 16, 128, &e8)
          == &indexed_rows[1]);
    CHECK(rocke_mma_catalog_select_largest_k_indexed(
              &indexed, "wmma_scaled", sources, "i32", 16, 16, -1, &e8)
          == &indexed_rows[1]);
    CHECK(rocke_mma_catalog_enumerate_indexed(
              &indexed, "wmma_scaled", sources, "", 16, 16, NULL, 0, &e8)
          == 0);
    CHECK(!rocke_mma_catalog_has_shape_indexed(
        &indexed, "wmma_scaled", sources, "", 16, 16, 128, &e8));
    CHECK(!rocke_mma_catalog_op_for_shape_indexed(
        &indexed, "wmma_scaled", sources, "", 16, 16, 128, &e8));
    CHECK(!rocke_mma_catalog_select_largest_k_indexed(
        &indexed, "wmma_scaled", sources, "", 16, 16, -1, &e8));
    CHECK(rejects_query([&] {
        rocke_mma_catalog_op_for_shape_indexed(
            &indexed, "wmma_scaled", sources, "i32", 16, 16, 128);
    }));
    auto inconsistent = *base;
    inconsistent.srcs[1].scale_block_size = ROCKE_MMA_SCALE_K16;
    CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&inconsistent); }));
    const rocke_mma_catalog_t mixed_blocks = {&inconsistent, 1};
    CHECK(rocke_mma_catalog_enumerate_indexed(
              &mixed_blocks, "wmma_scaled", sources, "fp32", 16, 16, NULL, 0, &e8)
          == 0);
    inconsistent = *base;
    inconsistent.srcs[2].dtype = "i32";
    CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&inconsistent); }));
    rocke_mma_op_t rows[8];
    rows[0] = *base;
    rows[0].srcs[1].dtype = "bf8e5m2";
    auto packing_atom = rows[0];
    packing_atom.srcs[1].frag_len = 8;
    const auto packing = rocke_scaled_wmma_contract(&packing_atom);
    CHECK(packing.matrix_formats[0] == 0 && packing.matrix_formats[1] == 1);
    CHECK(packing.matrix_words[0] == 16 && packing.matrix_words[1] == 8);
    CHECK(strstr(packing.intrinsic, "v16i32.v8i32"));
    for(int input = 0; input < 2; ++input)
    {
        auto unsupported = *base;
        (input == 0 ? unsupported.srcs[0].scale_dtype : unsupported.srcs[1].scale_dtype) = "e5m2";
        CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&unsupported); }));
    }
    int count = 1;
    for(int input = 0; input < 2; ++input)
    {
        for(const char* dtype : {"e4m3", "e5m3"})
        {
            rows[count] = rows[0];
            (input == 0 ? rows[count].srcs[0].scale_dtype : rows[count].srcs[1].scale_dtype)
                = dtype;
            ++count;
        }
    }
    rows[count] = rows[0];
    rows[count].srcs[1].scale_block_size = rows[count].srcs[0].scale_block_size
        = ROCKE_MMA_SCALE_K16;
    ++count;
    rows[count] = rows[0];
    rows[count].srcs[0].scale_dtype = NULL;
    rows[count].srcs[1].scale_dtype = NULL;
    rows[count].srcs[1].scale_block_size = rows[count].srcs[0].scale_block_size
        = ROCKE_MMA_SCALE_NONE;
    ++count;
    rocke_mma_catalog_t cat = {rows, count};
    for(int i = 0; i < count; ++i)
    {
        const auto& op = rows[i];
        const rocke_mma_scale_filter_t scales
            = {op.srcs[0].scale_dtype, op.srcs[1].scale_dtype, op.srcs[0].scale_block_size};
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
            CHECK(op.srcs[0].frag_len == 8 && op.srcs[1].frag_len == 8 && op.dst.frag_len == 8);
            ++packed_rows;
        }
        if(strcmp(op.family, "wmma_scaled") == 0)
        {
            snprintf(id,
                     sizeof(id),
                     "wmma_gfx1250_f32_16x16x128_%s_%s_scale_%s_%s_k%d",
                     short_dtype(op.srcs[0].dtype),
                     short_dtype(op.srcs[1].dtype),
                     op.srcs[0].scale_dtype,
                     op.srcs[1].scale_dtype,
                     op.srcs[0].scale_block_size);
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
    for(const char* a_dtype : {"fp8", "bf8", "fp6", "bf6", "fp4"})
    {
        for(const char* b_dtype : {"fp8", "bf8", "fp6", "bf6", "fp4"})
        {
            for(auto block_k : {ROCKE_MMA_SCALE_K16, ROCKE_MMA_SCALE_K32})
            {
                const rocke_mma_scale_filter_t scales = {"e8m0", "e8m0", block_k};
                const char* src_dtypes[] = {a_dtype, b_dtype, "fp32"};
                const auto* op = rocke_mma_catalog_op_for_shape_indexed(
                    &arch->mma, "wmma_scaled", src_dtypes, "fp32", 16, 16, 128, &scales);
                CHECK(op);
                for(int source = 0; source < 2; ++source)
                {
                    const auto& src = op->srcs[source];
                    CHECK(strcmp(src.scale_dtype, "e8m0") == 0);
                    CHECK(src.scale_block_size == block_k);
                    CHECK(src.scale_frag_len == (block_k == ROCKE_MMA_SCALE_K16 ? 8 : 4));
                    const auto layout = rocke_scaled_wmma_matrix_layout(op, source == 1);
                    CHECK(layout.fragment.carrier_bits == 32);
                }
                CHECK(!op->srcs[2].scale_dtype && op->srcs[2].scale_block_size == 0);
                CHECK(op->k == 128);
            }
        }
    }
    // Ordered identity fingerprint from the independently filtered 72c28a69 catalog,
    // plus the bf16-result row, the 16-bit-output, fp8-to-f16 and i8 rows, the
    // f8f6f4 and f4 32x16x128 rows, and the fp8 / bf8 -> f32 K 128 rows. Also
    // exercises all generated static ID pointers after repeated queries.
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
    CHECK(arch->mma.num_ops == 134);
    CHECK(fingerprint == UINT64_C(7565811404630989321));
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
            if(!op->srcs[0].scale_dtype)
            {
                CHECK(op->srcs[0].scale_frag_len == 0 && op->srcs[1].scale_frag_len == 0);
                CHECK(!op->srcs[0].scale_layout && !op->srcs[1].scale_layout);
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
                CHECK(op->srcs[0].frag_len == 32 && op->srcs[1].frag_len == 32
                      && op->dst.frag_len == 16);
                CHECK(op->srcs[0].scale_block_size == ROCKE_MMA_SCALE_K32);
                for(int source = 0; source < 2; ++source)
                {
                    const auto* map = source ? rocke_mma_op_b_scale_layout(op, NULL)
                                             : rocke_mma_op_a_scale_layout(op, NULL);
                    CHECK(map && map->frag_len == 1 && map->wave_size == 64);
                    CHECK(map->role
                          == (source ? ROCKE_MMA_ROLE_SCALE_SRC1 : ROCKE_MMA_ROLE_SCALE_SRC0));
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
            const int count = op->srcs[0].scale_block_size == ROCKE_MMA_SCALE_K16 ? 8 : 4;
            CHECK(op->srcs[0].scale_frag_len == count && op->srcs[1].scale_frag_len == count);
            for(int source = 0; source < 2; ++source)
            {
                const auto* map = source ? rocke_mma_op_b_scale_layout(op, NULL)
                                         : rocke_mma_op_a_scale_layout(op, NULL);
                CHECK(map && map->frag_len == count && map->wave_size == 32);
                CHECK(map->role
                      == (source ? ROCKE_MMA_ROLE_SCALE_SRC1 : ROCKE_MMA_ROLE_SCALE_SRC0));
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
                        CHECK(strstr(b.err, source ? "'scale_src1'" : "'scale_src0'"));
                    }
                    else
                        CHECK(ok && x && y && b.status == ROCKE_OK);
                    rocke_ir_builder_free(&b);
                }
                auto invalid = *op;
                (source ? invalid.srcs[1].scale_frag_len : invalid.srcs[0].scale_frag_len)
                    = count / 2;
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
                        injected.srcs[0].dtype = inputs[a];
                        injected.srcs[1].dtype = inputs[b];
                        injected.srcs[0].scale_dtype = scales[sa];
                        injected.srcs[1].scale_dtype = scales[sb];
                        injected.srcs[1].scale_block_size = injected.srcs[0].scale_block_size
                            = block;
                        injected.srcs[0].scale_frag_len = injected.srcs[1].scale_frag_len
                            = 128 / block;
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
            CHECK(op->srcs[0].frag_len == physical->srcs[0].frag_len);
            CHECK(op->srcs[1].frag_len == physical->srcs[1].frag_len);
            CHECK(op->dst.frag_len == physical->dst.frag_len);
            CHECK(op->wave_size == physical->wave_size);
            CHECK(op->srcs[0].layout == physical->srcs[0].layout);
            CHECK(op->srcs[1].layout == physical->srcs[1].layout);
            CHECK(op->dst.layout == physical->dst.layout);
            /* The bare-op_id path (rocke_b_mma's) reaches the same physical atom
             * and fragment as the row; the physical atom resolves to nothing. */
            const char* bare_physical = rocke_arch_mma_op_id_physical_op_id(op->op_id);
            CHECK(bare_physical != NULL && strcmp(bare_physical, physical_id) == 0);
            CHECK(rocke_arch_mma_dst_frag_len(op->op_id) == physical->dst.frag_len);
            CHECK(rocke_arch_mma_op_id_physical_op_id(physical_id) == NULL);
            /* The verifier's query answers with the same fragments, from the row. */
            int a_len = -1, b_len = -1, c_len = -1;
            CHECK(rocke_arch_mma_op_id_emulated_operands(op->op_id, &a_len, &b_len, &c_len));
            CHECK(a_len == physical->srcs[0].frag_len && b_len == physical->srcs[1].frag_len
                  && c_len == physical->dst.frag_len);
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
                = {op->srcs[0].scale_dtype, op->srcs[1].scale_dtype, op->srcs[0].scale_block_size};
            const char* dst = op->dst.dtype;

            /* enumerate: same atoms, same order, as filtering natives by hand. */
            const rocke_mma_op_t* got[64];
            int n = rocke_mma_catalog_enumerate(&arch->mma,
                                                op->family,
                                                op->srcs[0].dtype,
                                                op->srcs[1].dtype,
                                                op->srcs[2].dtype,
                                                -1,
                                                -1,
                                                got,
                                                (int)(sizeof(got) / sizeof(got[0])),
                                                &scales,
                                                dst);
            CHECK(n >= 1 && n <= (int)(sizeof(got) / sizeof(got[0])));
            int seen = 0;
            for(int j = 0; j < arch->mma.num_ops; ++j)
            {
                const rocke_mma_op_t* cand = &arch->mma.ops[j];
                if(strcmp(cand->family, op->family) != 0
                   || strcmp(cand->srcs[0].dtype, op->srcs[0].dtype) != 0
                   || strcmp(cand->srcs[1].dtype, op->srcs[1].dtype) != 0
                   || strcmp(cand->srcs[2].dtype, op->srcs[2].dtype) != 0
                   || strcmp(cand->dst.dtype, op->dst.dtype) != 0)
                    continue;
                if(cand->srcs[0].scale_dtype != op->srcs[0].scale_dtype
                   || cand->srcs[1].scale_dtype != op->srcs[1].scale_dtype
                   || cand->srcs[0].scale_block_size != op->srcs[0].scale_block_size)
                    continue;
                CHECK(rocke_arch_mma_op_is_native(cand));
                CHECK(seen < n && got[seen] == cand);
                ++seen;
            }
            CHECK(seen == n);

            /* The narrowing entry points resolve to the atom itself, unchanged. */
            CHECK(rocke_mma_catalog_has_shape(&arch->mma,
                                              op->family,
                                              op->srcs[0].dtype,
                                              op->srcs[1].dtype,
                                              op->srcs[2].dtype,
                                              op->m,
                                              op->n,
                                              op->k,
                                              &scales,
                                              dst));
            const rocke_mma_op_t* exact = rocke_mma_catalog_op_for_shape(&arch->mma,
                                                                         op->family,
                                                                         op->srcs[0].dtype,
                                                                         op->srcs[1].dtype,
                                                                         op->srcs[2].dtype,
                                                                         op->m,
                                                                         op->n,
                                                                         op->k,
                                                                         &scales,
                                                                         dst);
            CHECK(exact == op);
            const rocke_mma_op_t* largest = rocke_mma_catalog_select_largest_k(&arch->mma,
                                                                               op->family,
                                                                               op->srcs[0].dtype,
                                                                               op->srcs[1].dtype,
                                                                               op->srcs[2].dtype,
                                                                               op->m,
                                                                               op->n,
                                                                               -1,
                                                                               &scales,
                                                                               dst);
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
            CHECK(strcmp(got[i]->srcs[0].dtype, "fp32") == 0);
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
            CHECK(op && strcmp(op->srcs[0].dtype, row[1]) == 0
                  && strcmp(op->srcs[1].dtype, row[1]) == 0 && strcmp(op->dst.dtype, "i32") == 0);
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

static int test_independent_accumulator_width()
{
    rocke_ir_builder_t b = {};
    CHECK(rocke_ir_builder_init(&b, "independent_accumulator") == ROCKE_OK);
    for(const auto* dtype : {"fp32", "fp64"})
    {
        rocke_mmaop_t op = {};
        op.srcs[2].dtype = dtype;
        op.srcs[2].frag_len = 4;
        op.dst.dtype = strcmp(dtype, "fp32") == 0 ? "fp64" : "fp32";
        op.dst.frag_len = 7;
        auto* acc = rocke_gemm_emit_zero_acc_op(&b, &op);
        CHECK(acc && acc->type->count == 4);
        CHECK(acc->type->elem == (strcmp(dtype, "fp32") == 0 ? rocke_f32() : rocke_f64()));
    }
    rocke_ir_builder_free(&b);
    return 0;
}

static int test_scaled_accumulator_contracts()
{
    const auto* arch = rocke_arch_target_from_gfx("gfx1250");
    for(int i = 0; i < arch->mma.num_ops; ++i)
    {
        const auto& base = arch->mma.ops[i];
        if(strcmp(base.family, "wmma_scaled") != 0)
            continue;
        CHECK(base.srcs[2].frag_len == 8 && base.dst.frag_len == 8);
        CHECK(strstr(rocke_scaled_wmma_contract(&base).intrinsic, ".v8f32."));
        for(int role = 0; role < 4; ++role)
            for(const char* dtype : {static_cast<const char*>(nullptr), "", "unknown"})
            {
                auto invalid = base;
                if(role == 3)
                    invalid.dst.dtype = dtype;
                else
                    invalid.srcs[role].dtype = dtype;
                try
                {
                    rocke_scaled_wmma_contract(&invalid);
                    CHECK(false);
                }
                catch(const ckc::Error& error)
                {
                    CHECK(error.code() == ROCKE_ERR_VALUE);
                    char expected[256];
                    snprintf(expected,
                             sizeof(expected),
                             "unsupported scaled WMMA backend contract: %s",
                             base.op_id);
                    CHECK(strcmp(error.what(), expected) == 0);
                }
            }
        for(int source = 0; source < 2; ++source)
        {
            for(int width : {0, -1})
            {
                auto invalid = base;
                invalid.srcs[source].frag_len = width;
                CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&invalid); }));
            }
            auto independent = base;
            independent.srcs[source].frag_len = 8;
            auto signature = rocke_scaled_wmma_contract(&independent);
            CHECK(signature.matrix_words[source] == 8);
            CHECK(signature.matrix_words[1 - source] == base.srcs[1 - source].frag_len);
        }
        for(int role = 0; role < 3; ++role)
            for(int width : {0, 4, 7, 16})
            {
                auto invalid = base;
                if(role != 1)
                    invalid.srcs[2].frag_len = width;
                if(role != 0)
                    invalid.dst.frag_len = width;
                CHECK(rejects_query([&] { rocke_scaled_wmma_contract(&invalid); }));
            }
    }
    return 0;
}

static int test_scaled_hip_accumulator_values()
{
    for(const char* id : {"wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32",
                          "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k16"})
        for(int variant = 0; variant < 15; ++variant)
        {
            rocke_ir_builder_t b = {};
            CHECK(rocke_ir_builder_init(&b, "scaled_values") == ROCKE_OK);
            auto* a = rocke_b_param(&b, "a", rocke_vector_type(&b, rocke_i32(), 16), NULL);
            auto* c = rocke_b_param(&b, "c", rocke_vector_type(&b, rocke_f32(), 8), NULL);
            auto* s = rocke_b_param(&b, "s", strstr(id, "_k16") ? rocke_i64() : rocke_i32(), NULL);
            rocke_value_t* scales[] = {s, s};
            auto* result = rocke_b_mma(&b, id, a, a, c, scales, 2);
            CHECK(result);
            auto* op = result->op;
            if(variant < 10)
            {
                const int widths[] = {4, 7, 16, 8, 8};
                int kind = variant % 5;
                const auto* type
                    = kind == 3 ? rocke_f32()
                                : rocke_vector_type(
                                      &b, kind == 4 ? rocke_i32() : rocke_f32(), widths[kind]);
                (variant < 5 ? c : result)->type = type;
            }
            rocke_value_t* operands[] = {a, a, c, s, s, s};
            rocke_value_t extra = {"%extra", result->type, op};
            rocke_value_t* results[] = {result, &extra};
            op->operands = operands;
            op->results = results;
            if(variant == 10 || variant == 11)
                op->num_operands = variant == 10 ? 4 : 6;
            if(variant == 12 || variant == 13)
                op->num_results = variant == 12 ? 0 : 2;
            rocke_b_ret(&b);
            rocke_strbuf_t hip;
            CHECK(rocke_strbuf_init(&hip, 0) == 0);
            rocke_lower_hip_opts_t opts = {};
            opts.arch = "gfx1250";
            auto status = rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip);
            if(status != (variant == 14 ? ROCKE_OK : ROCKE_ERR_VALUE))
                fprintf(stderr, "scaled HIP: %s variant=%d status=%d\n", id, variant, status);
            CHECK(status == (variant == 14 ? ROCKE_OK : ROCKE_ERR_VALUE));
            rocke_strbuf_free(&hip);
            rocke_ir_builder_free(&b);
        }
    return 0;
}

static int test_scaled_source_values()
{
    const auto* target = rocke_arch_target_from_gfx("gfx1250");
    for(int row = 0; row < target->mma.num_ops; ++row)
    {
        const auto& atom = target->mma.ops[row];
        if(strcmp(atom.family, "wmma_scaled"))
            continue;
        for(int role : {0, 1, 3, 4})
            for(int kind = 0; kind < 4; ++kind)
            {
                rocke_ir_builder_t b = {};
                CHECK(rocke_ir_builder_init(&b, "scaled_source_values") == ROCKE_OK);
                const auto* scale_type
                    = atom.srcs[0].scale_block_size == 16 ? rocke_i64() : rocke_i32();
                auto* a = rocke_b_param(
                    &b, "a", rocke_vector_type(&b, rocke_i32(), atom.srcs[0].frag_len), nullptr);
                auto* bb = rocke_b_param(
                    &b, "b", rocke_vector_type(&b, rocke_i32(), atom.srcs[1].frag_len), nullptr);
                auto* c = rocke_b_param(&b, "c", rocke_vector_type(&b, rocke_f32(), 8), nullptr);
                auto* sa = rocke_b_param(&b, "sa", scale_type, nullptr);
                auto* sb = rocke_b_param(&b, "sb", scale_type, nullptr);
                rocke_value_t* scales[] = {sa, sb};
                auto* result = rocke_b_mma(&b, atom.op_id, a, bb, c, scales, 2);
                CHECK(result);
                auto* value = result->op->operands[role];
                char expected[ROCKE_ERR_MSG_CAP];
                if(role < 2)
                {
                    int width = atom.srcs[role].frag_len;
                    value->type = kind == 0
                                      ? rocke_i32()
                                      : rocke_vector_type(&b,
                                                          kind == 1 ? rocke_f32() : rocke_i32(),
                                                          width
                                                              + (kind == 2   ? -1
                                                                 : kind == 3 ? 1
                                                                             : 0));
                    snprintf(expected,
                             sizeof(expected),
                             "scaled WMMA requires src%d to be vec<i32x%d>",
                             role,
                             width);
                }
                else
                {
                    value->type = kind == 0
                                      ? (scale_type == rocke_i64() ? rocke_i32() : rocke_i64())
                                  : kind == 1 ? rocke_f32()
                                              : rocke_vector_type(&b, rocke_i32(), kind - 1);
                    snprintf(expected,
                             sizeof(expected),
                             "tile.%s expects %s scale operands, got %s/%s",
                             atom.op_id,
                             scale_type->name,
                             sa->type->name,
                             sb->type->name);
                }
                rocke_b_ret(&b);
                char* ll = nullptr;
                char error[ROCKE_ERR_MSG_CAP] = {};
                CHECK(rocke_lower_kernel_to_llvm_ex(
                          b.kernel, ROCKE_LLVM_FLAVOR_LLVM23, "gfx1250", &ll, error, sizeof(error))
                      == ROCKE_ERR_VALUE);
                char llvm_expected[ROCKE_ERR_MSG_CAP];
                snprintf(llvm_expected, sizeof(llvm_expected), "%s", expected);
                if(role < 2)
                {
                    const char* dtype = atom.srcs[role].dtype;
                    const char* name = role == 0 ? "a" : "b";
                    const int width = atom.srcs[role].frag_len;
                    const bool eight_bit
                        = strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "bf8e5m2") == 0;
                    if(eight_bit && kind == 1)
                        snprintf(llvm_expected,
                                 sizeof(llvm_expected),
                                 "MMA operand %s of '%s' holds f32 values; the instruction takes "
                                 "%s or i32 -- convert the values first",
                                 name,
                                 atom.op_id,
                                 dtype);
                    else if(eight_bit)
                        snprintf(llvm_expected,
                                 sizeof(llvm_expected),
                                 "MMA operand %s of '%s' is %s (%d bits); the instruction takes "
                                 "vec<i32x%d> (%d bits) -- pass a %d-bit value",
                                 name,
                                 atom.op_id,
                                 value->type->name,
                                 kind == 0 ? 32 : value->type->count * 32,
                                 width,
                                 width * 32,
                                 width * 32);
                    else
                    {
                        const int words = strcmp(dtype, "fp4e2m1") == 0 ? 8 : 12;
                        snprintf(llvm_expected,
                                 sizeof(llvm_expected),
                                 "MMA operand %s of '%s' is %s; the instruction takes the "
                                 "operand's %d packed 32-bit words as vec<i32x%d>, or the whole "
                                 "<%d x i32> register -- pass one of those",
                                 name,
                                 atom.op_id,
                                 value->type->name,
                                 words,
                                 words,
                                 width);
                    }
                }
                CHECK(ll == nullptr && strcmp(error, llvm_expected) == 0);
                rocke_strbuf_t hip;
                CHECK(rocke_strbuf_init(&hip, 0) == 0);
                rocke_lower_hip_opts_t opts = {};
                opts.arch = "gfx1250";
                CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_ERR_VALUE);
                rocke_strbuf_free(&hip);
                rocke_ir_builder_free(&b);
            }
    }
    return 0;
}

// Exercise the C ABI independently of the Python serializer and binding.
static int test_unscaled_wmma_values()
{
    struct Case
    {
        const char* arch;
        const char* id;
        const rocke_type_t* matrix_elem;
        int matrix_width;
        const rocke_type_t* accum_elem;
    };
    const Case cases[] = {
        {"gfx1151", "wmma_i32_16x16x16_iu8", rocke_i32(), 4, rocke_i32()},
        {"gfx1151", "wmma_i32_16x16x16_iu4", rocke_i32(), 2, rocke_i32()},
        {"gfx1151", "wmma_f32_16x16x16_f16", rocke_f16(), 16, rocke_f32()},
        {"gfx1151", "wmma_f32_16x16x16_bf16", rocke_bf16(), 16, rocke_f32()},
        {"gfx1201", "wmma_gfx12_f32_16x16x16_f16", rocke_f16(), 8, rocke_f32()},
        {"gfx1201", "wmma_gfx12_f32_16x16x16_bf16", rocke_bf16(), 8, rocke_f32()},
        {"gfx1250", "wmma_gfx1250_f32_16x16x32_f16", rocke_f16(), 16, rocke_f32()},
        {"gfx1250", "wmma_gfx1250_f32_16x16x32_bf16", rocke_bf16(), 16, rocke_f32()},
        {"gfx1250", "wmma_gfx1250_f32_16x16x64_fp8_fp8", rocke_i32(), 8, rocke_f32()},
        {"gfx1250", "wmma_gfx1250_f32_16x16x64_fp8_bf8", rocke_i32(), 8, rocke_f32()},
        {"gfx1250", "wmma_gfx1250_f32_16x16x64_bf8_fp8", rocke_i32(), 8, rocke_f32()},
        {"gfx1250", "wmma_gfx1250_f32_16x16x64_bf8_bf8", rocke_i32(), 8, rocke_f32()},
    };
    for(const auto& spec : cases)
        for(int variant = 0; variant < 23; ++variant)
        {
            rocke_ir_builder_t b = {};
            CHECK(rocke_ir_builder_init(&b, "unscaled_values") == ROCKE_OK);
            auto* a = rocke_b_param(
                &b, "a", rocke_vector_type(&b, spec.matrix_elem, spec.matrix_width), NULL);
            auto* bb = rocke_b_param(
                &b, "b", rocke_vector_type(&b, spec.matrix_elem, spec.matrix_width), NULL);
            auto* c = rocke_b_param(&b, "c", rocke_vector_type(&b, spec.accum_elem, 8), NULL);
            auto* result = rocke_b_mma(&b, spec.id, a, bb, c, NULL, 0);
            CHECK(result);
            auto* op = result->op;
            char expected[ROCKE_ERR_MSG_CAP];
            snprintf(expected,
                     sizeof(expected),
                     "unscaled WMMA requires src2 and dst to be vec<%sx8>",
                     spec.accum_elem->name);
            if(variant < 10)
            {
                const int widths[] = {4, 7, 16, 8, 8};
                int kind = variant % 5;
                const auto* wrong_elem = spec.accum_elem == rocke_i32() ? rocke_f32() : rocke_i32();
                const auto* type
                    = kind == 3 ? spec.accum_elem
                                : rocke_vector_type(
                                      &b, kind == 4 ? wrong_elem : spec.accum_elem, widths[kind]);
                (variant < 5 ? c : result)->type = type;
                if(spec.accum_elem == rocke_i32())
                {
                    const bool source = variant < 5;
                    const char* label = source ? "MMA operand c" : "MMA result";
                    const char* verb = source ? "takes" : "returns";
                    if(kind == 4)
                        snprintf(expected,
                                 sizeof(expected),
                                 "%s of '%s' holds f32 values; the instruction %s i32 -- %s",
                                 label,
                                 spec.id,
                                 verb,
                                 source ? "convert the values first"
                                        : "declare the result in an element type it returns");
                    else
                        snprintf(expected,
                                 sizeof(expected),
                                 "%s of '%s' is %s (%d bits); the instruction %s vec<i32x8> "
                                 "(256 bits) -- %s",
                                 label,
                                 spec.id,
                                 type->name,
                                 kind == 3 ? 32 : widths[kind] * 32,
                                 verb,
                                 source ? "pass a 256-bit value" : "declare a 256-bit result");
                }
            }
            rocke_value_t* operands[] = {a, bb, c, c};
            rocke_value_t extra = {"%extra", result->type, op};
            rocke_value_t* results[] = {result, &extra};
            op->operands = operands;
            op->results = results;
            if(variant == 10 || variant == 11)
                op->num_operands = variant == 10 ? 2 : 4;
            if(variant == 12 || variant == 13)
                op->num_results = variant == 12 ? 0 : 2;
            if(variant >= 10)
                snprintf(
                    expected, sizeof(expected), "unscaled WMMA expects 3 operands and 1 result");
            if(variant >= 15)
            {
                int role = (variant - 15) / 4;
                int kind = (variant - 15) % 4;
                const auto* type
                    = kind == 0 ? spec.matrix_elem
                                : rocke_vector_type(&b,
                                                    kind == 1 ? rocke_f32() : spec.matrix_elem,
                                                    spec.matrix_width
                                                        + (kind == 2   ? -1
                                                           : kind == 3 ? 1
                                                                       : 0));
                (role == 0 ? a : bb)->type = type;
                snprintf(expected,
                         sizeof(expected),
                         "unscaled WMMA requires src%d to be vec<%sx%d>",
                         role,
                         spec.matrix_elem->name,
                         spec.matrix_width);
            }
            if(variant >= 15 && spec.matrix_elem == rocke_i32())
            {
                int role = (variant - 15) / 4;
                try
                {
                    rocke_mma_operand_needs_reinterpret(
                        spec.id,
                        role == 0 ? ROCKE_MMA_ROLE_SRC0 : ROCKE_MMA_ROLE_SRC1,
                        (role == 0 ? a : bb)->type,
                        rocke_vector_type(&b, spec.matrix_elem, spec.matrix_width),
                        spec.arch);
                }
                catch(const ckc::Error& error)
                {
                    snprintf(expected, sizeof(expected), "%s", error.what());
                }
            }
            rocke_b_ret(&b);
            for(auto flavor :
                {ROCKE_LLVM_FLAVOR_LLVM20, ROCKE_LLVM_FLAVOR_LLVM22, ROCKE_LLVM_FLAVOR_LLVM23})
            {
                char* text = NULL;
                char error[ROCKE_ERR_MSG_CAP] = {};
                auto status = rocke_lower_kernel_to_llvm_ex(
                    b.kernel, flavor, spec.arch, &text, error, sizeof(error));
                CHECK(status == (variant == 14 ? ROCKE_OK : ROCKE_ERR_VALUE));
                if(variant != 14)
                {
                    CHECK(text == NULL);
                    if(variant >= 10 && variant <= 13 && spec.accum_elem == rocke_i32())
                    {
                        if(variant <= 11)
                            snprintf(expected,
                                     sizeof(expected),
                                     "tile.%s expects 3 operands, got %d",
                                     spec.id,
                                     variant == 10 ? 2 : 4);
                        else
                            snprintf(expected,
                                     sizeof(expected),
                                     "tile.%s: expected exactly one result, got %d",
                                     spec.id,
                                     variant == 12 ? 0 : 2);
                    }
                    CHECK(strcmp(error, expected) == 0);
                }
                else
                    CHECK(text != NULL);
                free(text);
            }
            if(variant == 14)
            {
                // These operations have no native HIP handler yet.
                rocke_strbuf_t hip;
                CHECK(rocke_strbuf_init(&hip, 0) == 0);
                rocke_lower_hip_opts_t opts = {};
                opts.arch = spec.arch;
                CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_ERR_NOTIMPL);
                rocke_strbuf_free(&hip);
            }
            rocke_ir_builder_free(&b);
        }
    return 0;
}

int main()
{
    // Keep this first: no previous query may prime the shared family index.
    if(test_required_indexed_dtypes() || test_family_index_first_use()
       || test_family_index_duplicates() || test_mma_result_names() || test_scale_contracts()
       || test_scale_layouts_and_families() || test_independent_accumulator_width()
       || test_scaled_accumulator_contracts() || test_scaled_hip_accumulator_values()
       || test_scaled_source_values() || test_unscaled_wmma_values()
       || test_scaled_wmma_supported_combinations() || test_emulated_rows_mirror_physical()
       || test_pre_existing_selection_is_native_only() || test_fp32_never_reaches_an_xf32_atom()
       || test_rdna_integer_wmma_rows_name_i8_and_i4() || test_catalog_keeps_established_order()
       || test_wmma_attention_accumulator_source_width()
       || test_wmma_scatter_uses_destination_layout())
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
            for(const char* dtype :
                {op->srcs[0].dtype, op->srcs[1].dtype, op->srcs[2].dtype, op->dst.dtype})
            {
                char scratch[64];
                if(strcmp(dtype, rocke_normalize_dtype(dtype, scratch, sizeof(scratch))) != 0)
                {
                    fprintf(stderr, "%s: noncanonical stored dtype %s\n", op->op_id, dtype);
                    return 1;
                }
            }
            const rocke_mma_scale_filter_t scales
                = {op->srcs[0].scale_dtype, op->srcs[1].scale_dtype, op->srcs[0].scale_block_size};
            const char* dst = op->dst.dtype;
            const bool result_is_accumulator = strcmp(dst, op->srcs[2].dtype) == 0;
            for(const char* a : {op->srcs[0].dtype, short_dtype(op->srcs[0].dtype)})
            {
                for(const char* b : {op->srcs[1].dtype, short_dtype(op->srcs[1].dtype)})
                {
                    if(!rocke_mma_catalog_has_shape(&arch->mma,
                                                    op->family,
                                                    a,
                                                    b,
                                                    op->srcs[2].dtype,
                                                    op->m,
                                                    op->n,
                                                    op->k,
                                                    &scales,
                                                    dst)
                       || rocke_mma_catalog_op_for_shape(&arch->mma,
                                                         op->family,
                                                         a,
                                                         b,
                                                         op->srcs[2].dtype,
                                                         op->m,
                                                         op->n,
                                                         op->k,
                                                         &scales,
                                                         dst)
                              != op
                       || (result_is_accumulator
                           && rocke_archtarget_op_for_shape(arch,
                                                            op->family,
                                                            a,
                                                            b,
                                                            op->srcs[2].dtype,
                                                            op->m,
                                                            op->n,
                                                            op->k,
                                                            &scales)
                                  != op)
                       || rocke_mma_catalog_select_largest_k(&arch->mma,
                                                             op->family,
                                                             a,
                                                             b,
                                                             op->srcs[2].dtype,
                                                             op->m,
                                                             op->n,
                                                             op->k,
                                                             &scales,
                                                             dst)
                              != op
                       || rocke_mma_catalog_enumerate(&arch->mma,
                                                      op->family,
                                                      a,
                                                      b,
                                                      op->srcs[2].dtype,
                                                      op->m,
                                                      op->n,
                                                      NULL,
                                                      0,
                                                      &scales,
                                                      dst)
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
