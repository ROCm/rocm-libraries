// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/numeric/ext_vector_base.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

#include <type_traits>

namespace ck_tile::core::arch::mma::scale::detail {

template <typename OutT, typename ValueT, typename T>
inline constexpr OutT to_scale_arg(const T& vec)
{
    static_assert(is_any_of<OutT, int32x8_t, int32x16_t>::value,
                  "Only support int32x8_t or int32x16_t as output type.");

    constexpr index_t N = vector_traits<OutT>::vector_size;
    if constexpr(is_any_of<ValueT, fp8_t, bf8_t>::value)
    {
        return bit_cast<OutT>(vec);
    }
    else if constexpr(is_any_of<ValueT, pk_fp6x16_t, pk_bf6x16_t>::value)
    {
        constexpr index_t init_N = N * 3 / 4;
        OutT out{};
        static_for<0, init_N, 1>{}([&out, &vec](auto i) { out[i.value] = vec.data[i.value]; });
        return out;
    }
    else if constexpr(is_any_of<ValueT, pk_fp4_t>::value)
    {
        constexpr index_t init_N = N / 2;
        using HalfOutT           = ext_vector_t<int32_t, init_N>;
        HalfOutT tmp             = bit_cast<HalfOutT>(vec);
        OutT out{};
        static_for<0, init_N, 1>{}([&out, &tmp](auto i) { out[i.value] = tmp[i.value]; });
        return out;
    }
    else
    {
        static_assert(sizeof(ValueT) == 0, "unsupported ValueT for to_scale_arg");
        return OutT{};
    }
}

template <typename ValueT, typename T>
inline constexpr int32x8_t to_mfma_scale_arg(const T& vec)
{
    return to_scale_arg<int32x8_t, ValueT>(vec);
}

template <typename ValueT, typename T>
inline constexpr int32x16_t to_wmma_scale_arg(const T& vec)
{
    return to_scale_arg<int32x16_t, ValueT>(vec);
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
