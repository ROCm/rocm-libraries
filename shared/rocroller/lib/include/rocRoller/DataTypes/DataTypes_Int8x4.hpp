// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocRoller
{
    struct Int8x4
    {
        Int8x4()
            : a(0)
            , b(0)
            , c(0)
            , d(0)
        {
        }

        Int8x4(int8_t xa, int8_t xb, int8_t xc, int8_t xd)
            : a(xa)
            , b(xb)
            , c(xc)
            , d(xd)
        {
        }

        explicit Int8x4(uint32_t v)
            : a(v & 0xff)
            , b((v >> 8) & 0xff)
            , c((v >> 16) & 0xff)
            , d((v >> 24) & 0xff)
        {
        }

        int8_t a, b, c, d;

        inline bool operator==(Int8x4 const& rhs) const
        {
            return a == rhs.a && b == rhs.b && c == rhs.c && d == rhs.d;
        }
    };

    static_assert(sizeof(Int8x4) == 4, "Int8x4 must be 4 bytes.");
} // namespace rocRoller
