// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"

#include <cstdint>
#include <stdio.h>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

namespace ck_tile::core::arch::mma {

namespace scale::detail {

// Backward-compatible aliases — prefer MatrixFmtCode / MatrixFmtCode_v from mma_data_format.hpp
template <typename T>
using ScaleDataTypeToFlag = MatrixFmtCode<T>;

template <typename T>
inline constexpr int32_t ScaleDataTypeToFlag_v = MatrixFmtCode_v<T>;

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaDataTypeToFlag
 * @brief  Expresses the interface of required members for each DataTypeToFlag type on Gfx9
 */
template <typename DataTypeToFlag>
concept ScaleMfmaDataTypeToFlag = requires(DataTypeToFlag dataTypeToFlag) {
    // Flag members for scale MFMA instructions
    { DataTypeToFlag::value } -> std::convertible_to<int32_t>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace scale::detail

struct DefaultScaleMfmaCtrlFlags
{
    static constexpr int32_t OPSEL_A = 0;
    static constexpr int32_t OPSEL_B = 0;
};

CK_TILE_HOST_DEVICE void print_flags(DefaultScaleMfmaCtrlFlags const& ctrlFlags)
{
    printf("CtrlFlags      OPSEL_A / OPSEL_B        : %d / %d\n",
           ctrlFlags.OPSEL_A,
           ctrlFlags.OPSEL_B);
}

/**
 * @struct DefaultScaleWmmaCtrlFlags
 * @brief Default WMMA scale control flags for GFX1250 scale WMMA operations.
 */
struct DefaultScaleWmmaCtrlFlags
{
};

CK_TILE_HOST_DEVICE void print_flags(DefaultScaleWmmaCtrlFlags const&)
{
    printf("CtrlFlags      (ScaleWmma, no flags)\n");
}

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaCtrlFlags
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept ScaleMfmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for scale MFMA instructions
    { CtrlFlags::OPSEL_A } -> std::convertible_to<int32_t>;
    { CtrlFlags::OPSEL_B } -> std::convertible_to<int32_t>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
