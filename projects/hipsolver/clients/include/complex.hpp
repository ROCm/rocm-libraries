/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"
#include <complex>

typedef struct hipsolverComplex
{
private:
    float x, y;

public:
    hipsolverComplex() = default;

    hipsolverComplex(float r, float i = 0)
        : x(r)
        , y(i)
    {
    }

    float real() const
    {
        return x;
    }
    float imag() const
    {
        return y;
    }
    void real(float r)
    {
        x = r;
    }
    void imag(float i)
    {
        y = i;
    }
} hipsolverComplex;

typedef struct hipsolverDoubleComplex
{
private:
    double x, y;

public:
    hipsolverDoubleComplex() = default;

    hipsolverDoubleComplex(double r, double i = 0)
        : x(r)
        , y(i)
    {
    }

    double real() const
    {
        return x;
    }
    double imag() const
    {
        return y;
    }
    void real(double r)
    {
        x = r;
    }
    void imag(double i)
    {
        y = i;
    }
} hipsolverDoubleComplex;

inline hipsolverComplex& operator+=(hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        += reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipsolverDoubleComplex& operator+=(hipsolverDoubleComplex&       lhs,
                                          const hipsolverDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        += reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipsolverComplex operator+(hipsolverComplex lhs, const hipsolverComplex& rhs)
{
    return lhs += rhs;
}

inline hipsolverDoubleComplex operator+(hipsolverDoubleComplex        lhs,
                                        const hipsolverDoubleComplex& rhs)
{
    return lhs += rhs;
}

inline hipsolverComplex& operator-=(hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        -= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipsolverDoubleComplex& operator-=(hipsolverDoubleComplex&       lhs,
                                          const hipsolverDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        -= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipsolverComplex operator-(hipsolverComplex lhs, const hipsolverComplex& rhs)
{
    return lhs -= rhs;
}

inline hipsolverDoubleComplex operator-(hipsolverDoubleComplex        lhs,
                                        const hipsolverDoubleComplex& rhs)
{
    return lhs -= rhs;
}

inline hipsolverComplex& operator*=(hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        *= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipsolverDoubleComplex& operator*=(hipsolverDoubleComplex&       lhs,
                                          const hipsolverDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        *= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipsolverComplex operator*(hipsolverComplex lhs, const hipsolverComplex& rhs)
{
    return lhs *= rhs;
}

inline hipsolverDoubleComplex operator*(hipsolverDoubleComplex        lhs,
                                        const hipsolverDoubleComplex& rhs)
{
    return lhs *= rhs;
}

inline hipsolverComplex& operator/=(hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        /= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipsolverDoubleComplex& operator/=(hipsolverDoubleComplex&       lhs,
                                          const hipsolverDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        /= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipsolverComplex operator/(hipsolverComplex lhs, const hipsolverComplex& rhs)
{
    return lhs /= rhs;
}

inline hipsolverDoubleComplex operator/(hipsolverDoubleComplex        lhs,
                                        const hipsolverDoubleComplex& rhs)
{
    return lhs /= rhs;
}

inline bool operator==(const hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    return reinterpret_cast<const std::complex<float>&>(lhs)
           == reinterpret_cast<const std::complex<float>&>(rhs);
}

inline bool operator!=(const hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const hipsolverDoubleComplex& lhs, const hipsolverDoubleComplex& rhs)
{
    return reinterpret_cast<const std::complex<double>&>(lhs)
           == reinterpret_cast<const std::complex<double>&>(rhs);
}

inline bool operator!=(const hipsolverDoubleComplex& lhs, const hipsolverDoubleComplex& rhs)
{
    return !(lhs == rhs);
}

inline hipsolverComplex operator-(const hipsolverComplex& x)
{
    return {-x.real(), -x.imag()};
}

inline hipsolverDoubleComplex operator-(const hipsolverDoubleComplex& x)
{
    return {-x.real(), -x.imag()};
}

inline hipsolverComplex operator+(const hipsolverComplex& x)
{
    return x;
}

inline hipsolverDoubleComplex operator+(const hipsolverDoubleComplex& x)
{
    return x;
}

namespace std
{
    inline float real(const hipsolverComplex& z)
    {
        return z.real();
    }

    inline double real(const hipsolverDoubleComplex& z)
    {
        return z.real();
    }

    inline float imag(const hipsolverComplex& z)
    {
        return z.imag();
    }

    inline double imag(const hipsolverDoubleComplex& z)
    {
        return z.imag();
    }

    inline hipsolverComplex conj(const hipsolverComplex& z)
    {
        return {z.real(), -z.imag()};
    }

    inline hipsolverDoubleComplex conj(const hipsolverDoubleComplex& z)
    {
        return {z.real(), -z.imag()};
    }

    inline float abs(const hipsolverComplex& z)
    {
        return abs(reinterpret_cast<const complex<float>&>(z));
    }

    inline double abs(const hipsolverDoubleComplex& z)
    {
        return abs(reinterpret_cast<const complex<double>&>(z));
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
