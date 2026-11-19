// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace TensileLite
{
    // Packed storage for two four-bit weights, low nibble first.
    struct Int4x2
    {
        uint8_t data;
    };

    static_assert(sizeof(Int4x2) == 1, "Int4x2 must be exactly one byte");
} // namespace TensileLite
