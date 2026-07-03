// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

namespace scale::detail {

template <typename ValueT, typename T>
inline constexpr int32x8_t to_mfma_scale_arg(const T& vec)
{
    if constexpr(is_any_of<ValueT, fp8_t, bf8_t>::value)
    {
        return bit_cast<int32x8_t>(vec);
    }
    else if constexpr(is_any_of<ValueT, pk_fp6x16_t, pk_bf6x16_t>::value)
    {
        return int32x8_t{
            vec.data[0], vec.data[1], vec.data[2], vec.data[3], vec.data[4], vec.data[5], 0, 0};
    }
    else if constexpr(is_any_of<ValueT, pk_fp4_t>::value)
    {
        int32x4_t tmp = bit_cast<int32x4_t>(vec);
        return int32x8_t{tmp[0], tmp[1], tmp[2], tmp[3], 0, 0, 0, 0};
    }
    else
    {
        static_assert(sizeof(ValueT) == 0, "unsupported ValueT for to_mfma_scale_arg");
        return int32x8_t{};
    }
}

} // namespace scale::detail

/**
 * @defgroup scale_mfma_gfx9 Scale MFMA for GFX9
 * @brief Scale specializations of @ref amdgcn_mma for GFX9 family.
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
#define MFMA_SCALE_IMPL(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B, FRAG_MN, FRAG_K, PER_LANE_C, NUM_ACC_C, INSTRUCTION)                                                           \
    template <typename CompilerTarget>                                                                                                                                       \
    /*               |A B C DataTypes       |MNK                      |                                                                                                   */ \
    struct amdgcn_mma<A_TYPE, B_TYPE, fp32_t, FRAG_MN, FRAG_MN, FRAG_K, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> \
    /*                                                                |WS  |AParams          |BPar         |CPar                  |                                       */ \
    : amdgcn_mma_base<A_TYPE, B_TYPE, fp32_t, FRAG_MN, FRAG_MN, FRAG_K, 64u, 32, NUM_ACC_A, 1, NUM_ACC_B, 1, PER_LANE_C, NUM_ACC_C, MfmaOp, MmaOpFamily::SCALE>              \
    {                                                                                                                                                                        \
        static constexpr const char* instruction_name = #INSTRUCTION;                                                                                                        \
                                                                                                                                                                             \
        template <typename... Params>                                                                                                                                        \
        CK_TILE_DEVICE static CVecType exec(AVecType const& aVec,                                                                                                            \
                                            BVecType const& bVec,                                                                                                            \
                                            CVecType const& cVec,                                                                                                            \
                                            int32_t scale_A,                                                                                                                 \
                                            int32_t scale_B)                                                                                                                 \
        {                                                                                                                                                                    \
            using P = WarpGemmParamsParser<Params...>;                                                                                                                       \
            return {INSTRUCTION(scale::detail::to_mfma_scale_arg<A_TYPE>(aVec),                                                                                              \
                                scale::detail::to_mfma_scale_arg<B_TYPE>(bVec),                                                                                              \
                                cVec,                                                                                                                                        \
                                PackedDataTypeToFlag_v<A_TYPE>,                                                                                                              \
                                PackedDataTypeToFlag_v<B_TYPE>,                                                                                                              \
                                P::op_sel_a,                                                                                                                                 \
                                scale_A,                                                                                                                                     \
                                P::op_sel_b,                                                                                                                                 \
                                scale_B)};                                                                                                                                   \
        }                                                                                                                                                                    \
    };
#define MFMA_SCALE_IMPL_16(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B) \
    MFMA_SCALE_IMPL(A_TYPE,                                      \
                    B_TYPE,                                      \
                    NUM_ACC_A,                                   \
                    NUM_ACC_B,                                   \
                    16u,                                         \
                    128u,                                        \
                    4,                                           \
                    1,                                           \
                    __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
#define MFMA_SCALE_IMPL_32(A_TYPE, B_TYPE, NUM_ACC_A, NUM_ACC_B) \
    MFMA_SCALE_IMPL(A_TYPE,                                      \
                    B_TYPE,                                      \
                    NUM_ACC_A,                                   \
                    NUM_ACC_B,                                   \
                    32u,                                         \
                    64u,                                         \
                    16,                                          \
                    4,                                           \
                    __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4)

// Note on the intrinsic NumAccess values we use here: In principle the "canonical" NumAccess values
// for A and B for gfx950 scale intrinsic is determined by the A and B datatypes. 8-bit datatypes
// require a NumAccess of 2, and 4 and 6-bit types a NumAccess of 1. We follow this *BUT* we do
// allow (1,1) for the cases where A and B are both 8 bit. In these cases, NumAccess (1,1) could
// still be valid when not using scale values.

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
MFMA_SCALE_IMPL_16(fp8_t,       fp8_t,       1, 1)
MFMA_SCALE_IMPL_16(fp8_t,       bf8_t,       1, 1)
MFMA_SCALE_IMPL_16(fp8_t,       pk_fp6x16_t, 2, 1)
MFMA_SCALE_IMPL_16(fp8_t,       pk_bf6x16_t, 2, 1)
MFMA_SCALE_IMPL_16(fp8_t,       pk_fp4_t,    2, 1)
MFMA_SCALE_IMPL_16(bf8_t,       fp8_t,       1, 1)
MFMA_SCALE_IMPL_16(bf8_t,       bf8_t,       1, 1)
MFMA_SCALE_IMPL_16(bf8_t,       pk_fp6x16_t, 2, 1)
MFMA_SCALE_IMPL_16(bf8_t,       pk_bf6x16_t, 2, 1)
MFMA_SCALE_IMPL_16(bf8_t,       pk_fp4_t,    2, 1)
MFMA_SCALE_IMPL_16(pk_fp6x16_t, fp8_t,       1, 2)
MFMA_SCALE_IMPL_16(pk_fp6x16_t, bf8_t,       1, 2)
MFMA_SCALE_IMPL_16(pk_fp6x16_t, pk_fp6x16_t, 1, 1)
MFMA_SCALE_IMPL_16(pk_fp6x16_t, pk_bf6x16_t, 1, 1)
MFMA_SCALE_IMPL_16(pk_fp6x16_t, pk_fp4_t,    1, 1)
MFMA_SCALE_IMPL_16(pk_bf6x16_t, fp8_t,       1, 2)
MFMA_SCALE_IMPL_16(pk_bf6x16_t, bf8_t,       1, 2)
MFMA_SCALE_IMPL_16(pk_bf6x16_t, pk_fp6x16_t, 1, 1)
MFMA_SCALE_IMPL_16(pk_bf6x16_t, pk_bf6x16_t, 1, 1)
MFMA_SCALE_IMPL_16(pk_bf6x16_t, pk_fp4_t,    1, 1)
MFMA_SCALE_IMPL_16(pk_fp4_t,    fp8_t,       1, 2)
MFMA_SCALE_IMPL_16(pk_fp4_t,    bf8_t,       1, 2)
MFMA_SCALE_IMPL_16(pk_fp4_t,    pk_fp6x16_t, 1, 1)
MFMA_SCALE_IMPL_16(pk_fp4_t,    pk_bf6x16_t, 1, 1)
MFMA_SCALE_IMPL_16(pk_fp4_t,    pk_fp4_t,    1, 1)

#undef MFMA_SCALE_IMPL_16

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4
MFMA_SCALE_IMPL_32(fp8_t,       fp8_t,       1, 1)
MFMA_SCALE_IMPL_32(fp8_t,       bf8_t,       1, 1)
MFMA_SCALE_IMPL_32(fp8_t,       pk_fp6x16_t, 2, 1)
MFMA_SCALE_IMPL_32(fp8_t,       pk_bf6x16_t, 2, 1)
MFMA_SCALE_IMPL_32(fp8_t,       pk_fp4_t,    2, 1)
MFMA_SCALE_IMPL_32(bf8_t,       fp8_t,       1, 1)
MFMA_SCALE_IMPL_32(bf8_t,       bf8_t,       1, 1)
MFMA_SCALE_IMPL_32(bf8_t,       pk_fp6x16_t, 2, 1)
MFMA_SCALE_IMPL_32(bf8_t,       pk_bf6x16_t, 2, 1)
MFMA_SCALE_IMPL_32(bf8_t,       pk_fp4_t,    2, 1)
MFMA_SCALE_IMPL_32(pk_fp6x16_t, fp8_t,       1, 2)
MFMA_SCALE_IMPL_32(pk_fp6x16_t, bf8_t,       1, 2)
MFMA_SCALE_IMPL_32(pk_fp6x16_t, pk_fp6x16_t, 1, 1)
MFMA_SCALE_IMPL_32(pk_fp6x16_t, pk_bf6x16_t, 1, 1)
MFMA_SCALE_IMPL_32(pk_fp6x16_t, pk_fp4_t,    1, 1)
MFMA_SCALE_IMPL_32(pk_bf6x16_t, fp8_t,       1, 2)
MFMA_SCALE_IMPL_32(pk_bf6x16_t, bf8_t,       1, 2)
MFMA_SCALE_IMPL_32(pk_bf6x16_t, pk_fp6x16_t, 1, 1)
MFMA_SCALE_IMPL_32(pk_bf6x16_t, pk_bf6x16_t, 1, 1)
MFMA_SCALE_IMPL_32(pk_bf6x16_t, pk_fp4_t,    1, 1)
MFMA_SCALE_IMPL_32(pk_fp4_t,    fp8_t,       1, 2)
MFMA_SCALE_IMPL_32(pk_fp4_t,    bf8_t,       1, 2)
MFMA_SCALE_IMPL_32(pk_fp4_t,    pk_fp6x16_t, 1, 1)
MFMA_SCALE_IMPL_32(pk_fp4_t,    pk_bf6x16_t, 1, 1)
MFMA_SCALE_IMPL_32(pk_fp4_t,    pk_fp4_t,    1, 1)

#undef MFMA_SCALE_IMPL_32
#undef MFMA_SCALE_IMPL
// clang-format on

/** @} */ // scale_mfma_gfx9

} // namespace ck_tile::core::arch::mma
