// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocRoller
{
    struct E5M3x4
    {
        E5M3x4()
            : a(0)
            , b(0)
            , c(0)
            , d(0)
        {
        }

        E5M3x4(uint8_t xa, uint8_t xb, uint8_t xc, uint8_t xd)
            : a(xa)
            , b(xb)
            , c(xc)
            , d(xd)
        {
        }

        explicit E5M3x4(uint32_t v)
            : a(v & 0xff)
            , b((v >> 8) & 0xff)
            , c((v >> 16) & 0xff)
            , d((v >> 24) & 0xff)
        {
        }

        uint8_t a, b, c, d;

        inline bool operator==(E5M3x4 const& rhs) const
        {
            return a == rhs.a && b == rhs.b && c == rhs.c && d == rhs.d;
        }
    };

    static_assert(sizeof(E5M3x4) == 4, "E5M3x4 must be 4 bytes.");
} // namespace rocRoller
