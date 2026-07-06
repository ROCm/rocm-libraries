// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

#include <type_traits>

namespace ck_tile::core::arch::mma::scale::detail {

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

template <typename ValueT, typename T>
inline constexpr int32x16_t to_wmma_scale_arg(const T& vec)
{
    if constexpr(is_any_of<ValueT, fp8_t, bf8_t>::value)
    {
        return bit_cast<int32x16_t>(vec);
    }
    else if constexpr(is_any_of<ValueT, pk_fp6x16_t, pk_bf6x16_t>::value)
    {
        // clang-format off
        return int32x16_t{vec.data[0], vec.data[1], vec.data[2],  vec.data[3],  vec.data[4], vec.data[5], vec.data[6], vec.data[7],
                          vec.data[8], vec.data[9], vec.data[10], vec.data[11], 0, 0, 0, 0};
        // clang-format on
    }
    else if constexpr(is_any_of<ValueT, pk_fp4_t>::value)
    {
        int32x8_t tmp = bit_cast<int32x8_t>(vec);
        return int32x16_t{
            tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], 0, 0, 0, 0, 0, 0, 0, 0};
    }
    else
    {
        static_assert(sizeof(ValueT) == 0, "unsupported ValueT for to_wmma_scale_arg");
        return int32x16_t{};
    }
}

template <typename DataType, int32_t ScaleFlag>
inline constexpr bool is_valid_ScaleVecType()
{
    [[maybe_unused]] constexpr int32_t data_type_check = PackedDataTypeToFlag_v<DataType>;

    if constexpr(std::is_same_v<DataType, pk_fp4_t>)
    {
        return ScaleFlag == static_cast<int32_t>(ScaleDataType::E8M0) ||
               ScaleFlag == static_cast<int32_t>(ScaleDataType::E5M3) ||
               ScaleFlag == static_cast<int32_t>(ScaleDataType::E4M3);
    }
    else
    {
        return ScaleFlag == static_cast<int32_t>(ScaleDataType::E8M0);
    }
}

template <typename ADataType, typename BDataType, int32_t ScaleAFlag, int32_t ScaleBFlag>
inline constexpr bool is_legal_combination =
    is_valid_ScaleVecType<ADataType, ScaleAFlag>() &&
    is_valid_ScaleVecType<BDataType, ScaleBFlag>() &&
    (!(std::is_same_v<ADataType, pk_fp4_t> && std::is_same_v<BDataType, pk_fp4_t>) ||
     ScaleAFlag == ScaleBFlag);

} // namespace ck_tile::core::arch::mma::scale::detail
