/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef HIPTENSOR_TYPES_EXT_HPP
#define HIPTENSOR_TYPES_EXT_HPP

#include <cmath>
#include <hip/hip_bfloat16.h>
#include <limits>
#include <ostream>

#include "type_traits.hpp"

namespace hiptensor
{
    ////////////////////////////////////////////////////////////////////////
    ///////////  hiptensor::hfloat16_t host and device conversions  //////////
    ////////////////////////////////////////////////////////////////////////
    template <typename Outgoing,
              typename Incoming,
              typename std::enable_if_t<!std::is_same_v<Incoming, Outgoing>, int> = 0>
    __host__ __device__ inline Outgoing convert(const Incoming& value)
    {
#if !HIPTENSOR_NO_HALF
        if constexpr(std::is_same_v<Outgoing, hfloat16_t>)
        {

#if defined(__HIP_NO_HALF_CONVERSIONS__)
            detail::Fp16Bits fp16(static_cast<float16_t>(value));
            return fp16.h16;
#else
            return static_cast<hfloat16_t>(value);
#endif // defined(__HIP_NO_HALF_CONVERSIONS__)
        }
        else if constexpr(std::is_same_v<Incoming, hfloat16_t>)
        {

#if defined(__HIP_NO_HALF_CONVERSIONS__)
            detail::Fp16Bits fp16(value);
            return static_cast<Outgoing>(fp16.f16);
#else
            return static_cast<Outgoing>(value);
#endif // defined(__HIP_NO_HALF_CONVERSIONS__)
        }
        else
#endif // !HIPTENSOR_NO_HALF
        {
            return static_cast<Outgoing>(value);
        }
    }

    template <typename Outgoing,
              typename Incoming,
              typename std::enable_if_t<std::is_same_v<Incoming, Outgoing>, int> = 0>
    __host__ __device__ inline Outgoing const& convert(const Incoming& value)
    {
        return value;
    }

    ////////////////////////////////////////////////////////////////////
    ///////////  hiptensor::hfloat16_t host & device operators  //////////
    ///////////////////////////////////////////////////////////////////

#if defined(__HIP_NO_HALF_OPERATORS__)
// No operators defined for host or device
#define HIPTENSOR_HALF_OP_ATTR HIPTENSOR_HOST_DEVICE
#else
// No operators defined just for host
#define HIPTENSOR_HALF_OP_ATTR HIPTENSOR_HOST
#endif // defined(__HIP_NO_HALF_OPERATORS__)

#if !HIPTENSOR_NO_HALF

    HIPTENSOR_HALF_OP_ATTR inline bool operator==(const hfloat16_t& x, const hfloat16_t& y)
    {
        auto absDiff = std::fabs(__half2float(x) - __half2float(y));
        auto absAdd  = std::fabs(__half2float(x) + __half2float(y));
        return absDiff <= __half2float(std::numeric_limits<hfloat16_t>::epsilon()) * absAdd * 2.0f
               || absDiff < __half2float(std::numeric_limits<hfloat16_t>::min());
    }

    HIPTENSOR_HALF_OP_ATTR inline bool operator!=(const hfloat16_t& x, const hfloat16_t& y)
    {
        return !(x == y);
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t operator-(const hfloat16_t& x)
    {
        detail::Fp16Bits fp16(x);
        fp16.i16 ^= 0x8000; // Flip sign
        return fp16.h16;
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t operator+(const hfloat16_t& x, const hfloat16_t& y)
    {
        return convert<hfloat16_t>(convert<float16_t>(x) + convert<float16_t>(y));
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t operator-(const hfloat16_t& x, const hfloat16_t& y)
    {
        return convert<hfloat16_t>(convert<float16_t>(x) - convert<float16_t>(y));
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t operator*(const hfloat16_t& x, const hfloat16_t& y)
    {
        return convert<hfloat16_t>(convert<float16_t>(x) * convert<float16_t>(y));
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t operator/(const hfloat16_t& x, const hfloat16_t& y)
    {
        return convert<hfloat16_t>(convert<float16_t>(x) / convert<float16_t>(y));
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t& operator+=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = x + y;
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t& operator-=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = x - y;
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t& operator*=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = x * y;
    }

    HIPTENSOR_HALF_OP_ATTR inline hfloat16_t& operator/=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = x / y;
    }

#endif // !HIPTENSOR_NO_HALF
} // namespace hiptensor

namespace std
{
    ///////////////////////////////////////////////////////////
    //////////  std::ostream::operator<<(float16_t)  //////////
    ///////////////////////////////////////////////////////////

    inline ostream& operator<<(ostream& stream, hiptensor::float16_t const& val)
    {
        return stream << static_cast<float>(val);
    }

    ///////////////////////////////////////////////////////////
    //////////  std::ostream::operator<<(hfloat16_t)  /////////
    ///////////////////////////////////////////////////////////
#if !HIPTENSOR_NO_HALF
    inline ostream& operator<<(ostream& stream, hiptensor::hfloat16_t const& val)
    {
        return stream << __half2float(val);
    }
#endif // !HIPTENSOR_NO_HALF
} // namespace std

#endif // HIPTENSOR_TYPES_EXT_HPP
