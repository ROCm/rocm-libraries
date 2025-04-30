/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */
#ifndef HIPBLAS_COMPLEX_HPP
#define HIPBLAS_COMPLEX_HPP

#include "hipblas.h"
#include <complex>

inline hipblasComplex& operator+=(hipblasComplex& lhs, const hipblasComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        += reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipblasDoubleComplex& operator+=(hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        += reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipblasComplex operator+(hipblasComplex lhs, const hipblasComplex& rhs)
{
    return lhs += rhs;
}

inline hipblasDoubleComplex operator+(hipblasDoubleComplex lhs, const hipblasDoubleComplex& rhs)
{
    return lhs += rhs;
}

inline hipblasComplex& operator-=(hipblasComplex& lhs, const hipblasComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        -= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipblasDoubleComplex& operator-=(hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        -= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipblasComplex operator-(hipblasComplex lhs, const hipblasComplex& rhs)
{
    return lhs -= rhs;
}

inline hipblasDoubleComplex operator-(hipblasDoubleComplex lhs, const hipblasDoubleComplex& rhs)
{
    return lhs -= rhs;
}

inline hipblasComplex& operator*=(hipblasComplex& lhs, const hipblasComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        *= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipblasDoubleComplex& operator*=(hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        *= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipblasComplex operator*(hipblasComplex lhs, const hipblasComplex& rhs)
{
    return lhs *= rhs;
}

inline hipblasDoubleComplex operator*(hipblasDoubleComplex lhs, const hipblasDoubleComplex& rhs)
{
    return lhs *= rhs;
}

inline hipblasComplex& operator/=(hipblasComplex& lhs, const hipblasComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        /= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipblasDoubleComplex& operator/=(hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        /= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipblasComplex operator/(hipblasComplex lhs, const hipblasComplex& rhs)
{
    return lhs /= rhs;
}

inline hipblasDoubleComplex operator/(hipblasDoubleComplex lhs, const hipblasDoubleComplex& rhs)
{
    return lhs /= rhs;
}

inline bool operator==(const hipblasComplex& lhs, const hipblasComplex& rhs)
{
    return reinterpret_cast<const std::complex<float>&>(lhs)
           == reinterpret_cast<const std::complex<float>&>(rhs);
}

inline bool operator!=(const hipblasComplex& lhs, const hipblasComplex& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    return reinterpret_cast<const std::complex<double>&>(lhs)
           == reinterpret_cast<const std::complex<double>&>(rhs);
}

inline bool operator!=(const hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    return !(lhs == rhs);
}

inline hipblasComplex operator-(const hipblasComplex& x)
{
    return {-x.real(), -x.imag()};
}

inline hipblasDoubleComplex operator-(const hipblasDoubleComplex& x)
{
    return {-x.real(), -x.imag()};
}

inline hipblasComplex operator+(const hipblasComplex& x)
{
    return x;
}

inline hipblasDoubleComplex operator+(const hipblasDoubleComplex& x)
{
    return x;
}

namespace std
{
    inline float real(const hipblasComplex& z)
    {
        return z.real();
    }

    inline double real(const hipblasDoubleComplex& z)
    {
        return z.real();
    }

    inline float imag(const hipblasComplex& z)
    {
        return z.imag();
    }

    inline double imag(const hipblasDoubleComplex& z)
    {
        return z.imag();
    }

    inline hipblasComplex conj(const hipblasComplex& z)
    {
        return {z.real(), -z.imag()};
    }

    inline hipblasDoubleComplex conj(const hipblasDoubleComplex& z)
    {
        return {z.real(), -z.imag()};
    }

    inline float conj(const float& r)
    {
        return r;
    }

    inline double conj(const double& r)
    {
        return r;
    }
}

#endif
