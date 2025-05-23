/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

namespace TensileLite
{
    /**
 * \ingroup DataTypes
 */
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

        Int8x4(uint32_t v)
            : a(v & 0xff)
            , b((v << 8) & 0xff)
            , c((v << 16) & 0xff)
            , d((v << 24) & 0xff)
        {
        }

        int8_t a, b, c, d;

        int32_t operator*(Int8x4 const& other) const
        {
            return static_cast<int32_t>(a) * static_cast<int32_t>(other.a)
                   + static_cast<int32_t>(b) * static_cast<int32_t>(other.b)
                   + static_cast<int32_t>(c) * static_cast<int32_t>(other.c)
                   + static_cast<int32_t>(d) * static_cast<int32_t>(other.d);
        }

        inline bool operator==(Int8x4 const& rhs) const
        {
            return a == rhs.a && b == rhs.b && c == rhs.c && d == rhs.d;
        }
    };

    static_assert(sizeof(Int8x4) == 4, "Int8x4 must be 4 bytes.");
} // namespace TensileLite

namespace std
{
    inline std::string to_string(const TensileLite::Int8x4& val)
    {
        return "(" + std::to_string(val.a) + ", " + std::to_string(val.b) + ", "
               + std::to_string(val.c) + ", " + std::to_string(val.d) + ")";
    }
} // namespace std
