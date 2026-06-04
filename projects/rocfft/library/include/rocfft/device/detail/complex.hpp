// Copyright (C) 2021 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*! @file complex.hpp
 *  @brief Device-side complex type and type traits for the rocFFT device library.
 *  @details Extracted from shared/rocfft_complex.h and
 *  library/src/device/kernels/common.h. Definitions are guarded so this header
 *  is safe to include both standalone and inside rocFFT's RTC build, where the
 *  canonical types are already provided.
 */

#pragma once

#ifndef ROCFFT_DEVICE_COMPLEX_HPP
#define ROCFFT_DEVICE_COMPLEX_HPP

// Pull in the canonical rocfft_complex type if not already defined.
// When used standalone (outside rocFFT RTC), we provide a self-contained
// definition.  When used inside rocFFT RTC, the embedded headers will have
// already defined these types.
#ifndef ROCFFT_COMPLEX_H

#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

/*! @brief Self-contained device/host complex number type.
 *  @tparam Treal Underlying real scalar type (float or double).
 */
template <typename Treal>
struct rocfft_complex
{
    Treal x;
    Treal y;

    __device__ __host__                 rocfft_complex()                       = default;
    __device__ __host__                 rocfft_complex(const rocfft_complex&)  = default;
    __device__ __host__                 rocfft_complex(rocfft_complex&&)       = default;
    __device__ __host__ rocfft_complex& operator=(const rocfft_complex& rhs) & = default;
    __device__ __host__ rocfft_complex& operator=(rocfft_complex&& rhs) &      = default;
    __device__                          __host__ ~rocfft_complex()             = default;

    __device__ __host__ constexpr rocfft_complex(Treal real, Treal imag)
        : x{real}
        , y{imag}
    {
    }

    template <typename U>
    __device__ __host__ explicit constexpr rocfft_complex(const rocfft_complex<U>& z)
        : x(z.x)
        , y(z.y)
    {
    }

    __device__ __host__ constexpr Treal real() const
    {
        return x;
    }
    __device__ __host__ constexpr Treal imag() const
    {
        return y;
    }

    __forceinline__ __device__ __host__ rocfft_complex operator-() const
    {
        return {-x, -y};
    }
    __forceinline__ __device__ __host__ rocfft_complex operator+() const
    {
        return *this;
    }

    __device__ __host__ auto& operator+=(const rocfft_complex& rhs)
    {
        return *this = {x + rhs.x, y + rhs.y};
    }
    __device__ __host__ auto operator+(const rocfft_complex& rhs) const
    {
        auto lhs = *this;
        return lhs += rhs;
    }
    __device__ __host__ auto& operator-=(const rocfft_complex& rhs)
    {
        return *this = {x - rhs.x, y - rhs.y};
    }
    __device__ __host__ auto operator-(const rocfft_complex& rhs) const
    {
        auto lhs = *this;
        return lhs -= rhs;
    }
    __device__ __host__ auto& operator*=(const rocfft_complex& rhs)
    {
        return *this = {x * rhs.x - y * rhs.y, y * rhs.x + x * rhs.y};
    }
    __device__ __host__ auto operator*(const rocfft_complex& rhs) const
    {
        auto lhs = *this;
        return lhs *= rhs;
    }

    template <typename U>
    __device__ __host__ auto& operator*=(const U& rhs)
    {
        return (x *= Treal(rhs)), (y *= Treal(rhs)), *this;
    }

    template <typename U>
    __device__ __host__ auto operator*(const U& rhs) const
    {
        auto lhs = *this;
        return lhs *= Treal(rhs);
    }
};

template <typename U, typename Treal>
__device__ __host__ rocfft_complex<Treal> operator*(const U& lhs, const rocfft_complex<Treal>& rhs)
{
    return {Treal(lhs) * rhs.x, Treal(lhs) * rhs.y};
}

#endif // ROCFFT_COMPLEX_H

// Type traits — always define these with our own guard.
#ifndef ROCFFT_DEVICE_TYPE_TRAITS_DEFINED
#define ROCFFT_DEVICE_TYPE_TRAITS_DEFINED

//! @brief Maps a complex element type to its real scalar type.
#ifndef COMMON_H
template <class T>
struct real_type;

template <>
struct real_type<rocfft_complex<float>>
{
    typedef float type;
};

template <>
struct real_type<rocfft_complex<double>>
{
    typedef double type;
};

template <class T>
using real_type_t = typename real_type<T>::type;

//! @brief Maps a real scalar type to its complex element type.
template <class T>
struct complex_type;

template <>
struct complex_type<float>
{
    typedef rocfft_complex<float> type;
};

template <>
struct complex_type<double>
{
    typedef rocfft_complex<double> type;
};

template <class T>
using complex_type_t = typename complex_type<T>::type;
#endif // COMMON_H

//! @brief Stride kind used by the generated Stockham functions.
#ifndef DEVICE_ENUM_H
enum StrideBin
{
    SB_UNIT, //!< Unit stride between consecutive elements.
    SB_NONUNIT, //!< Arbitrary (non-unit) stride between elements.
};
#endif // DEVICE_ENUM_H

#endif // ROCFFT_DEVICE_TYPE_TRAITS_DEFINED

#endif // ROCFFT_DEVICE_COMPLEX_HPP
