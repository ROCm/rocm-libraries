// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../include/rocblaslt-types.h"

#include <Tensile/SwizzleLayout.hpp>

// TensileLite SwizzleLayout id of an operand stored in this order. Every order
// without a layout of its own, including HIPBLASLT_ORDER_COL256_R128_EXT, maps to
// Default.
inline int rocblaslt_swizzle_layout(hipblasLtOrder_t order)
{
    using Layout = TensileLite::SwizzleLayout;
    switch(order)
    {
    case HIPBLASLT_ORDER_COL256_R128_P48_EXT:
        return static_cast<int>(Layout::P48);
    case HIPBLASLT_ORDER_COL256_R128_P16_EXT:
        return static_cast<int>(Layout::P16);
    case HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT:
        return static_cast<int>(Layout::P48Page24);
    case HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT:
        return static_cast<int>(Layout::P48Page24Split);
    default:
        return static_cast<int>(Layout::Default);
    }
}
