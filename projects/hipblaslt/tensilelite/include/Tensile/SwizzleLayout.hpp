// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace TensileLite
{
    // The tile layout of a swizzled operand, stored as ProblemType::swizzleLayoutA and
    // swizzleLayoutB. Contractions.py writes these values from the problem type's
    // SwizzleLayoutA and SwizzleLayoutB. Default is the layout SwizzleTensorA or
    // SwizzleTensorB alone describes, which for FP6 is HIPBLASLT_ORDER_COL256_R128_EXT;
    // the others are the HIPBLASLT_ORDER_COL256_R128_* orders of the same name.
    // Append only.
    enum class SwizzleLayout : int
    {
        Default        = 0,
        P48            = 1,
        P16            = 2,
        P48Page24      = 3,
        P48Page24Split = 4,
    };
}
