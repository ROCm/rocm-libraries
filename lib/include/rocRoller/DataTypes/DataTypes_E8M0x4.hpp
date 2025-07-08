/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <cstdint>

namespace rocRoller
{
    struct E8M0x4
    {
        E8M0x4()
            : a(0)
            , b(0)
            , c(0)
            , d(0)
        {
        }

        E8M0x4(uint8_t xa, uint8_t xb, uint8_t xc, uint8_t xd)
            : a(xa)
            , b(xb)
            , c(xc)
            , d(xd)
        {
        }

        explicit E8M0x4(uint32_t v)
            : a(v & 0xff)
            , b((v >> 8) & 0xff)
            , c((v >> 16) & 0xff)
            , d((v >> 24) & 0xff)
        {
        }

        uint8_t a, b, c, d;

        inline bool operator==(E8M0x4 const& rhs) const
        {
            return a == rhs.a && b == rhs.b && c == rhs.c && d == rhs.d;
        }
    };

    static_assert(sizeof(E8M0x4) == 4, "E8M0x4 must be 4 bytes.");
} // namespace rocRoller
