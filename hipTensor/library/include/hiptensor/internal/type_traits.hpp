/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_TYPE_TRAITS_HPP
#define HIPTENSOR_TYPE_TRAITS_HPP
#include <cfloat>

#include "config.hpp"
#include "native_types.hpp"

namespace hiptensor
{
    namespace detail
    {
        struct Fp16Bits
        {
            union
            {
                uint16_t  i16;
                float16_t f16;
#if !HIPTENSOR_NO_HALF
                hfloat16_t h16;
#endif // !HIPTENSOR_NO_HALF
                bfloat16_t b16;
            };
            constexpr Fp16Bits(uint16_t initVal)
                : i16(initVal)
            {
            }
#define TEST_TEST 1
            constexpr Fp16Bits(float16_t initVal)
                : f16(initVal)
            {
            }
#if !HIPTENSOR_NO_HALF
            constexpr Fp16Bits(hfloat16_t initVal)
                : h16(initVal)
            {
            }
#endif
            constexpr Fp16Bits(bfloat16_t initVal)
                : b16(initVal)
            {
            }
        };

        struct Fp32Bits
        {
            union
            {
                uint32_t  i32;
                float32_t f32;
            };
            constexpr Fp32Bits(uint32_t initVal)
                : i32(initVal)
            {
            }
            constexpr Fp32Bits(float32_t initVal)
                : f32(initVal)
            {
            }
        };

    } // namespace detail
} // namespace hiptensor

namespace std
{
    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<float16_t>  //////////////
    ///////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::float16_t
        numeric_limits<hiptensor::float16_t>::epsilon() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.f16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::float16_t
        numeric_limits<hiptensor::float16_t>::infinity() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7C00));
        return eps.f16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::float16_t
        numeric_limits<hiptensor::float16_t>::lowest() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0xFBFF));
        return eps.f16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::float16_t
        numeric_limits<hiptensor::float16_t>::max() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7BFF));
        return eps.f16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::float16_t
        numeric_limits<hiptensor::float16_t>::min() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.f16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::float16_t
        numeric_limits<hiptensor::float16_t>::quiet_NaN() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FFF));
        return eps.f16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::float16_t
        numeric_limits<hiptensor::float16_t>::signaling_NaN() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7DFF));
        return eps.f16;
    }

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<hfloat16_t>  /////////////
    ///////////////////////////////////////////////////////////
#if !HIPTENSOR_NO_HALF
    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::hfloat16_t
        numeric_limits<hiptensor::hfloat16_t>::epsilon() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.h16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::hfloat16_t
        numeric_limits<hiptensor::hfloat16_t>::infinity() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7C00));
        return eps.h16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::hfloat16_t
        numeric_limits<hiptensor::hfloat16_t>::lowest() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0xFBFF));
        return eps.h16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::hfloat16_t
        numeric_limits<hiptensor::hfloat16_t>::max() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7BFF));
        return eps.h16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::hfloat16_t
        numeric_limits<hiptensor::hfloat16_t>::min() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.h16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::hfloat16_t
        numeric_limits<hiptensor::hfloat16_t>::quiet_NaN() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FFF));
        return eps.h16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::hfloat16_t
        numeric_limits<hiptensor::hfloat16_t>::signaling_NaN() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7DFF));
        return eps.h16;
    }

#endif // !HIPTENSOR_NO_HALF

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<bfloat16_t>  /////////////
    ///////////////////////////////////////////////////////////

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::bfloat16_t
        numeric_limits<hiptensor::bfloat16_t>::epsilon() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x3C00));
        return eps.b16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::bfloat16_t
        numeric_limits<hiptensor::bfloat16_t>::infinity() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7F80));
        return eps.b16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::bfloat16_t
        numeric_limits<hiptensor::bfloat16_t>::lowest() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0xFF7F));
        return eps.b16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::bfloat16_t
        numeric_limits<hiptensor::bfloat16_t>::max() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7F7F));
        return eps.b16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::bfloat16_t
        numeric_limits<hiptensor::bfloat16_t>::min() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x007F));
        return eps.b16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::bfloat16_t
        numeric_limits<hiptensor::bfloat16_t>::quiet_NaN() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FC0));
        return eps.b16;
    }

    template <>
    HIPTENSOR_HOST_DEVICE constexpr hiptensor::bfloat16_t
        numeric_limits<hiptensor::bfloat16_t>::signaling_NaN() noexcept
    {
        hiptensor::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FC0));
        return eps.b16;
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace std

namespace hiptensor
{
    template <typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
    constexpr auto maxExactInteger() -> decltype(std::numeric_limits<T>::max())
    {
        return std::numeric_limits<T>::max();
    }

    template <typename T,
              typename std::enable_if_t<std::is_floating_point<T>::value
                                            && std::numeric_limits<T>::digits,
                                        int>
              = 0>
    constexpr auto maxExactInteger() ->
        typename std::conditional_t<std::is_same<T, float64_t>::value, int64_t, int32_t>
    {
        using RetT =
            typename std::conditional_t<std::is_same<T, float64_t>::value, int64_t, int32_t>;
        return ((RetT)1 << std::numeric_limits<T>::digits);
    }

    template <typename T,
              typename std::enable_if_t<
#if !HIPTENSOR_NO_HALF
                  std::is_same<T, hfloat16_t>::value ||
#endif // !HIPTENSOR_NO_HALF
                      std::is_same<T, float16_t>::value,
                  int>
              = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // f16 mantissa is 10 bits
        return ((int32_t)1 << 11);
    }

    template <typename T, typename std::enable_if_t<std::is_same<T, bfloat16_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // b16 mantissa is 7 bits
        return ((int32_t)1 << 8);
    }
} // namespace hiptensor

#endif // HIPTENSOR_TYPE_TRAITS_HPP
