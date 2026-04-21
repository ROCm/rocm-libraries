// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

#include <stdio.h>
#include <type_traits>

namespace ck_tile::core::arch::mma {

/**
 * @enum WmmaCtrlFlags
 * @brief Common wmma control flags for gfx11 and gfx12
 */
enum struct WmmaCtrlFlags : bool
{
    // Only has an effect on gfx11 when the accumulator is 16-bit
    // Determines which half of the 32-bit accum register to use
    // Low = bits [15:0]
    // High = bits[31:16]
    LOW  = false,
    HIGH = true,

    // Only has an effect on gfx11 / 12 when the input is 8-bit int
    // Signage indicator of inputs / accum
    UNSIGNED = false,
    SIGNED   = true
};

/**
 * @class DefaultWmmaCtrlFlags
 * @brief Generates default WMMA control flags based on data types.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 */
template <typename ADataType, typename BDataType, typename CDataType>
struct DefaultWmmaCtrlFlags
{
    // Generate default flags for signage
    // Only used currently for integer inputs / accum in gfx11 / gfx12
    constexpr static WmmaCtrlFlags InputSignA =
        std::is_signed_v<ADataType> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;
    constexpr static WmmaCtrlFlags InputSignB =
        std::is_signed_v<BDataType> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;
    constexpr static WmmaCtrlFlags AccumSign =
        std::is_signed_v<CDataType> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;

    // Generate default flags for accumulator destination bits.
    // Only used if accumulation size is 16-bit in gfx11
    constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
};

/**
 * @struct WmmaOp
 * @brief Meta-tag for the WMMA operation. This will be used in the MmaOp struct to
 * identify the operation as an WMMA operation.
 */
struct WmmaOp
{
};

CK_TILE_HOST_DEVICE constexpr const char* to_string(WmmaOp) { return "WmmaOp"; }

/**
 * @class is_mma_op_wmma
 * @brief Trait to check if MmaOp is an WMMA operation
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp, typename = void>
struct is_mma_op_wmma : std::false_type
{
};

/**
 * @struct is_mma_op_wmma
 * @brief MmaOp specialization for WMMA operations, confirming the OpType matches WmmaOp
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
// TODO: c++20 requires
struct is_mma_op_wmma<MmaOp, std::enable_if_t<std::is_same_v<typename MmaOp::OpType, WmmaOp>>>
    : std::true_type
{
};

/**
 * @brief Convenience evaluator for is_mma_op_wmma trait
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
static constexpr bool is_mma_op_wmma_v = is_mma_op_wmma<MmaOp>::value;

/**
 * @struct DefaultWmmaCtrlFlags
 * @brief Default WMMA control flags for dense and sparse WMMA operations.
 */
struct DefaultWmmaCtrlFlags
{
    constexpr static bool Clamp = false;

    // Only has an effect on gfx11 when the accumulator is 16-bit.
    // Determines which half of the 32-bit accum register to use for the 16-bit result.
    // false = low bits [15:0], true = high bits [31:16]
    constexpr static bool UseHighAccumBits = true;
};

CK_TILE_HOST_DEVICE void print_flags(DefaultWmmaCtrlFlags const& ctrlFlags)
{
    printf("CtrlFlags      Clamp / UseHighAccumBits : %d / %d\n",
           ctrlFlags.Clamp,
           ctrlFlags.UseHighAccumBits);
}

} // namespace ck_tile::core::arch::mma
