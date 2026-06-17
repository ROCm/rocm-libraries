/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc.
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

#include <Tensile/Macros.hpp>

TENSILE_HIDDEN_BEGIN

#define TENSILE_USE_MX_SCALE

#ifdef TENSILE_USE_MX_SCALE

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

namespace TensileLite
{
    // data type
    struct E8
    {
        uint8_t data;

        // default constructor
        explicit HIP_HOST_DEVICE E8() = default;

        // constructor from uint8_t
        explicit HIP_HOST_DEVICE E8(uint8_t v0)
        {
            data = v0;
        }

        // constructor from int
        explicit HIP_HOST_DEVICE E8(int v0)
        {
            data = v0;
        }

        // constructor from int
        explicit HIP_HOST_DEVICE E8(size_t v0)
        {
            data = v0;
        }

        explicit HIP_HOST_DEVICE E8(float v0)
        {
            union {
                uint32_t x;
                float f;
            } v;
            v.f = v0;

            data = ((v.x & 0x7fffffff) >> 23);
        }

        explicit HIP_HOST_DEVICE E8(double v0)
            : E8(float(v0))
        {
        }

        // check for zero
        inline HIP_HOST_DEVICE bool is_zero() const
        {
            return false;
        }

        // check for nan
        inline HIP_HOST_DEVICE bool is_nan() const
        {
            return data == 0xff;
        }

        // check for inf
        inline HIP_HOST_DEVICE bool is_inf() const
        {
            return false;
        }

        explicit inline HIP_HOST_DEVICE operator float() const
        {
            union {
                uint32_t x;
                float f;
            } v;

            if (is_nan()) {
                v.x = ((data << 23) | 0x1);
            } else {
                v.x = (data << 23);
            }

            return v.f;
        }

        // convert to double
        explicit inline HIP_HOST_DEVICE operator double() const
        {
            return double(float(*this)); // convert to float, then convert to f16
        }

        // convert to bf16
        explicit inline HIP_HOST_DEVICE operator BFloat16() const
        {
            return BFloat16(float(*this)); // convert to float, then convert to bf16
        }

        // convert to bf16
        explicit inline HIP_HOST_DEVICE operator _Float16() const
        {
            return _Float16(float(*this)); // convert to float, then convert to bf16
        }

        // convert to int
        explicit inline HIP_HOST_DEVICE operator int() const
        {
            return int(float(*this)); // convert to float, then convert to f16
        }

    };

    inline float operator*(TensileLite::E8 a, float b)
    {
        return static_cast<float>(static_cast<float>(a) * b);
    }

    inline float operator*(float a, TensileLite::E8 b)
    {
        return static_cast<float>(a * static_cast<float>(b));
    }
} // end of namespace TensileLite

namespace std
{
    inline std::string to_string(const TensileLite::E8& a)
    {
        return std::to_string(static_cast<float>(a));
    }

    inline ostream& operator<<(ostream& stream, const TensileLite::E8 a)
    {
        return stream << static_cast<float>(a);
    }
} // namespace std

#endif // TENSILE_USE_MX_SCALE

TENSILE_HIDDEN_END
