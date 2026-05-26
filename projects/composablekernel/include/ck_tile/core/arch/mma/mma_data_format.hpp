// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"

#include <cstdint>

namespace ck_tile::core::arch::mma {

/**
 * @struct MatrixFmtCode
 * @brief Maps a data type to its hardware matrix format code used by f8f6f4 builtins.
 *
 * Format codes are used by `__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4`,
 * `__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4`, and
 * `__builtin_amdgcn_mfma_scale_f32_*_f8f6f4` to identify the data format of A/B operands.
 */
template <typename T>
struct MatrixFmtCode;

template <>
struct MatrixFmtCode<fp8_t> // e4m3
{
    static constexpr int32_t value = 0;
};

template <>
struct MatrixFmtCode<bf8_t> // e5m2
{
    static constexpr int32_t value = 1;
};

template <>
struct MatrixFmtCode<pk_fp6x16_t> // e2m3
{
    static constexpr int32_t value = 2;
};

template <>
struct MatrixFmtCode<pk_bf6x16_t> // e3m2
{
    static constexpr int32_t value = 3;
};

template <>
struct MatrixFmtCode<pk_fp4_t> // e2m1
{
    static constexpr int32_t value = 4;
};

template <typename T>
inline constexpr int32_t MatrixFmtCode_v = MatrixFmtCode<T>::value;

} // namespace ck_tile::core::arch::mma
