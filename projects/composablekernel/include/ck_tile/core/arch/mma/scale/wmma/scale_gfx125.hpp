// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/scale/scale_traits.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @defgroup scale_wmma_gfx125 Scale WMMA for GFX125
 * @brief Scale specializations of @ref amdgcn_mma for GFX125 family.
 *
 * Template parameters A/B/C denote input/output types,
 * M/N/K are the fragment (MmaTile) sizes,
 * and `enable_if_target_*` restricts the specialization to specific GPU targets.
 *
 * @tparam CompilerTarget Current compiler target.
 *
 * @sa amdgcn_mma_base for base template parameter documentation.
 * @{
 */

// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires

// clang-format off
#define WMMA_SCALE_IMPL(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B, OP_FAMILY, INSTRUCTION, SCALE_TYPE)                                    \
    template <typename CompilerTarget>                                                                                               \
    /*               |A B C DataTypes       |MNK            |                                                                     */ \
    struct amdgcn_mma<A_TYPE, B_TYPE, fp32_t, 16u, 16u, 128u, CompilerTarget, OP_FAMILY, enable_if_target_gfx1250_t<CompilerTarget>> \
    /*                                                      |WS  |AParams          |BPar         |CPar |                          */ \
    : amdgcn_mma_base<A_TYPE, B_TYPE, fp32_t, 16u, 16u, 128u, 32u, 64, NUM_ACC_A, 1, NUM_ACC_B, 1, 8, 1, WmmaOp, OP_FAMILY>          \
    {                                                                                                                                \
        static constexpr const char* instruction_name = #INSTRUCTION;                                                                \
                                                                                                                                     \
        template <typename... Params>                                                                                                \
        CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,                                                                    \
                                            BVecType const& bVec,                                                                    \
                                            CVecType const& cVec,                                                                    \
                                            SCALE_TYPE scaleA,                                                                       \
                                            SCALE_TYPE scaleB)                                                                       \
        {                                                                                                                            \
            using P = WarpGemmParamsParser<Params...>;                                                                               \
            static_assert(                                                                                                           \
                scale::detail::is_legal_combination<A_TYPE, B_TYPE, P::scale_a, P::scale_b>,                                         \
                "Unsupported ADataType/BDataType/scale_a/scale_b combination");                                                      \
            return {INSTRUCTION(PackedDataTypeToFlag_v<A_TYPE>,                                                                      \
                                to_type<int32x16_t>(aVec),                                                                           \
                                PackedDataTypeToFlag_v<B_TYPE>,                                                                      \
                                to_type<int32x16_t>(bVec),                                                                           \
                                0,                                                                                                   \
                                cVec,                                                                                                \
                                P::op_sel_a,                                                                                         \
                                P::scale_a,                                                                                          \
                                scaleA,                                                                                              \
                                P::op_sel_b,                                                                                         \
                                P::scale_b,                                                                                          \
                                scaleB,                                                                                              \
                                P::reuse_a,                                                                                          \
                                P::reuse_b)};                                                                                        \
        }                                                                                                                            \
    };
#define WMMA_SCALE32_IMPL(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B)       \
    WMMA_SCALE_IMPL(A_TYPE,                                           \
                    B_TYPE,                                           \
                    NUM_ACC_A,                                        \
                    NUM_ACC_B,                                        \
                    MmaOpFamily::SCALE,                               \
                    __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, \
                    int32_t)
#define WMMA_SCALE16_IMPL(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B)         \
    WMMA_SCALE_IMPL(A_TYPE,                                             \
                    B_TYPE,                                             \
                    NUM_ACC_A,                                          \
                    NUM_ACC_B,                                          \
                    MmaOpFamily::SCALE16,                               \
                    __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, \
                    int64_t)

WMMA_SCALE32_IMPL(fp8_t,       fp8_t,       1, 1)
WMMA_SCALE32_IMPL(fp8_t,       bf8_t,       1, 1)
WMMA_SCALE32_IMPL(fp8_t,       pk_fp6x16_t, 4, 2)
WMMA_SCALE32_IMPL(fp8_t,       pk_bf6x16_t, 4, 2)
WMMA_SCALE32_IMPL(fp8_t,       pk_fp4_t,    4, 2)
WMMA_SCALE32_IMPL(bf8_t,       fp8_t,       1, 1)
WMMA_SCALE32_IMPL(bf8_t,       bf8_t,       1, 1)
WMMA_SCALE32_IMPL(bf8_t,       pk_fp6x16_t, 4, 2)
WMMA_SCALE32_IMPL(bf8_t,       pk_bf6x16_t, 4, 2)
WMMA_SCALE32_IMPL(bf8_t,       pk_fp4_t,    4, 2)
WMMA_SCALE32_IMPL(pk_fp6x16_t, fp8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_fp6x16_t, bf8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_fp6x16_t, pk_fp6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_fp6x16_t, pk_bf6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_fp6x16_t, pk_fp4_t,    1, 1)
WMMA_SCALE32_IMPL(pk_bf6x16_t, fp8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_bf6x16_t, bf8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_bf6x16_t, pk_fp6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_bf6x16_t, pk_bf6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_bf6x16_t, pk_fp4_t,    1, 1)
WMMA_SCALE32_IMPL(pk_fp4_t,    fp8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_fp4_t,    bf8_t,       2, 4)
WMMA_SCALE32_IMPL(pk_fp4_t,    pk_fp6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_fp4_t,    pk_bf6x16_t, 1, 1)
WMMA_SCALE32_IMPL(pk_fp4_t,    pk_fp4_t,    1, 1)

#undef WMMA_SCALE32_IMPL

WMMA_SCALE16_IMPL(fp8_t,       fp8_t,       1, 1)
WMMA_SCALE16_IMPL(fp8_t,       bf8_t,       1, 1)
WMMA_SCALE16_IMPL(fp8_t,       pk_fp6x16_t, 4, 2)
WMMA_SCALE16_IMPL(fp8_t,       pk_bf6x16_t, 4, 2)
WMMA_SCALE16_IMPL(fp8_t,       pk_fp4_t,    4, 2)
WMMA_SCALE16_IMPL(bf8_t,       fp8_t,       1, 1)
WMMA_SCALE16_IMPL(bf8_t,       bf8_t,       1, 1)
WMMA_SCALE16_IMPL(bf8_t,       pk_fp6x16_t, 4, 2)
WMMA_SCALE16_IMPL(bf8_t,       pk_bf6x16_t, 4, 2)
WMMA_SCALE16_IMPL(bf8_t,       pk_fp4_t,    4, 2)
WMMA_SCALE16_IMPL(pk_fp6x16_t, fp8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_fp6x16_t, bf8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_fp6x16_t, pk_fp6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_fp6x16_t, pk_bf6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_fp6x16_t, pk_fp4_t,    1, 1)
WMMA_SCALE16_IMPL(pk_bf6x16_t, fp8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_bf6x16_t, bf8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_bf6x16_t, pk_fp6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_bf6x16_t, pk_bf6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_bf6x16_t, pk_fp4_t,    1, 1)
WMMA_SCALE16_IMPL(pk_fp4_t,    fp8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_fp4_t,    bf8_t,       2, 4)
WMMA_SCALE16_IMPL(pk_fp4_t,    pk_fp6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_fp4_t,    pk_bf6x16_t, 1, 1)
WMMA_SCALE16_IMPL(pk_fp4_t,    pk_fp4_t,    1, 1)

#undef WMMA_SCALE16_IMPL
#undef WMMA_SCALE_IMPL

#define WMMA_SCALE_IMPL32(OP_FAMILY, INSTRUCTION, SCALE_TYPE) \
template <typename CompilerTarget>\
    /*               |A B C DataTypes           |MNK            |                                                                     */ \
    struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 16u, 128u, CompilerTarget, OP_FAMILY, enable_if_target_gfx1250_t<CompilerTarget>> \
    /*                                                          |WS  |AParams  |BPar |CPar  |                                         */ \
    : amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 16, 2, WmmaOp, OP_FAMILY>                         \
    {                                                                                                                                    \
        static constexpr const char* instruction_name = #INSTRUCTION;                                                                    \
                                                                                                                                         \
        template <typename... Params>                                                                                                    \
        CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,                                                                        \
                                            BVecType const& bVec,                                                                        \
                                            CVecType const& cVec,                                                                        \
                                            SCALE_TYPE scaleA,                                                                           \
                                            SCALE_TYPE scaleB)                                                                           \
        {                                                                                                                                \
            using P = WarpGemmParamsParser<Params...>;                                                                                   \
            static_assert(                                                                                                               \
                scale::detail::is_legal_combination<pk_fp4_t, pk_fp4_t, P::scale_a, P::scale_b>,                                         \
                "Unsupported ADataType/BDataType/scale_a/scale_b combination");                                                          \
            return {INSTRUCTION(to_type<int32x16_t>(aVec),                                                                               \
                                to_type<int32x8_t>(bVec),                                                                                \
                                0,                                                                                                       \
                                cVec,                                                                                                    \
                                P::op_sel_a,                                                                                             \
                                P::scale_a,                                                                                              \
                                scaleA,                                                                                                  \
                                P::op_sel_b,                                                                                             \
                                P::scale_b,                                                                                              \
                                scaleB,                                                                                                  \
                                P::reuse_a,                                                                                              \
                                P::reuse_b)};                                                                                            \
        }                                                                                                                                \
    };

WMMA_SCALE_IMPL32(MmaOpFamily::SCALE,   __builtin_amdgcn_wmma_scale_f32_32x16x128_f4,   int32_t)
WMMA_SCALE_IMPL32(MmaOpFamily::SCALE16, __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4, int64_t)

#undef WMMA_SCALE_IMPL32

// Some type combinations already have a DENSE specialisation with a dedicated builtin. 
// Here, we provide remaining no-scale specialisations because for gfx1250 WMMA,
// the caller wants to use an actual no-scale instruction.
// Contrast this with MFMA: the LLVM backend selects the plain v_mfma_f32_16x16x128_f8f6f4 instruction 
// instead of v_mfma_scale_f32_16x16x128_f8f6f4 whenever the scale args passed to the intrinsic are literal 0.
// See: https://github.com/ROCm/llvm-project/blob/therock-7.13/llvm/lib/Target/AMDGPU/SIInstrInfo.td#L317-L327
#define WMMA_UNSCALED_IMPL(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B)                                                                              \
    template <typename CompilerTarget>                                                                                                        \
    /*               |A B C DataTypes       |MNK            |                                                                              */ \
    struct amdgcn_mma<A_TYPE, B_TYPE, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>> \
    /*                                                      |WS  |AParams          |BPar         |CPar |                                   */ \
    : amdgcn_mma_base<A_TYPE, B_TYPE, fp32_t, 16u, 16u, 128u, 32u, 64, NUM_ACC_A, 1, NUM_ACC_B, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>          \
    {                                                                                                                                         \
        static constexpr const char* instruction_name =                                                                                       \
            "__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4";                                                                                     \
                                                                                                                                              \
        template <typename... Params>                                                                                                         \
        CK_TILE_DEVICE static CVecType                                                                                                        \
        exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)                                                                \
        {                                                                                                                                     \
            return {__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<A_TYPE>,                                                \
                                                               to_type<int32x16_t>(aVec),                                                     \
                                                               PackedDataTypeToFlag_v<B_TYPE>,                                                \
                                                               to_type<int32x16_t>(bVec),                                                     \
                                                               0,                                                                             \
                                                               cVec)};                                                                        \
        }                                                                                                                                     \
    };

WMMA_UNSCALED_IMPL(fp8_t,       pk_fp6x16_t, 4, 2)
WMMA_UNSCALED_IMPL(fp8_t,       pk_bf6x16_t, 4, 2)
WMMA_UNSCALED_IMPL(fp8_t,       pk_fp4_t,    4, 2)
WMMA_UNSCALED_IMPL(bf8_t,       pk_fp6x16_t, 4, 2)
WMMA_UNSCALED_IMPL(bf8_t,       pk_bf6x16_t, 4, 2)
WMMA_UNSCALED_IMPL(bf8_t,       pk_fp4_t,    4, 2)
WMMA_UNSCALED_IMPL(pk_fp6x16_t, fp8_t,       2, 4)
WMMA_UNSCALED_IMPL(pk_fp6x16_t, bf8_t,       2, 4)
WMMA_UNSCALED_IMPL(pk_fp6x16_t, pk_bf6x16_t, 1, 1)
WMMA_UNSCALED_IMPL(pk_fp6x16_t, pk_fp4_t,    1, 1)
WMMA_UNSCALED_IMPL(pk_bf6x16_t, fp8_t,       2, 4)
WMMA_UNSCALED_IMPL(pk_bf6x16_t, bf8_t,       2, 4)
WMMA_UNSCALED_IMPL(pk_bf6x16_t, pk_fp6x16_t, 1, 1)
WMMA_UNSCALED_IMPL(pk_bf6x16_t, pk_bf6x16_t, 1, 1)
WMMA_UNSCALED_IMPL(pk_bf6x16_t, pk_fp4_t,    1, 1)
WMMA_UNSCALED_IMPL(pk_fp4_t,    fp8_t,       2, 4)
WMMA_UNSCALED_IMPL(pk_fp4_t,    bf8_t,       2, 4)
WMMA_UNSCALED_IMPL(pk_fp4_t,    pk_fp6x16_t, 1, 1)
WMMA_UNSCALED_IMPL(pk_fp4_t,    pk_bf6x16_t, 1, 1)

#undef WMMA_UNSCALED_IMPL
// clang-format on

/** @} */ // scale_wmma_gfx125

} // namespace ck_tile::core::arch::mma
