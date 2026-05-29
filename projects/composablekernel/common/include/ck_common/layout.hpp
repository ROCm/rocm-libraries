// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Common Layout enum and constexpr helpers. No runtime, no CK deps.

#pragma once

#include "ck_common/platform.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ck_common {

// Auto is a resolve-time placeholder -- Signature::resolve() replaces it with
// the concrete layout from the operator slot. It never reaches the kernel.
// PackedExternal is used by sparse/packed tensor formats in the dispatcher.
enum struct Layout : uint8_t
{
    Row            = 0,
    Col            = 1,
    Auto           = 2,
    PackedExternal = 3
};

constexpr const char* layoutName(Layout layout)
{
    switch(layout)
    {
    case Layout::Row: return "Row";
    case Layout::Col: return "Col";
    case Layout::Auto: return "Auto";
    case Layout::PackedExternal: return "PackedExternal";
    }
    CK_COMMON_UNREACHABLE();
}

constexpr bool isValidLayoutForRank(Layout layout, int rank)
{
    switch(layout)
    {
    case Layout::Row: return rank == 2;
    case Layout::Col: return rank == 2;
    case Layout::Auto: return false;
    case Layout::PackedExternal: return false;
    }
    CK_COMMON_UNREACHABLE();
}

template <typename T, std::size_t N>
constexpr T leadingDimStride(Layout layout, const std::array<T, N>& strides)
{
    switch(layout)
    {
    case Layout::Row: return strides[0];
    case Layout::Col: return strides[1];
    case Layout::Auto: throw "leadingDimStride requires Row or Col layout";
    case Layout::PackedExternal: throw "leadingDimStride requires Row or Col layout";
    }
    CK_COMMON_UNREACHABLE();
}

constexpr std::array<int, 2> layoutStrides(Layout layout, int rows, int cols)
{
    switch(layout)
    {
    case Layout::Row: return {cols, 1};
    case Layout::Col: return {1, rows};
    case Layout::Auto: throw "layoutStrides requires Row or Col layout";
    case Layout::PackedExternal: throw "layoutStrides requires Row or Col layout";
    }
    CK_COMMON_UNREACHABLE();
}

} // namespace ck_common
