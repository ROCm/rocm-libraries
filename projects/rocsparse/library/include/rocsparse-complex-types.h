/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief rocsparse-complex-types.h defines complex data types used in rocsparse
 */

#ifndef ROCSPARSE_COMPLEX_TYPES_H
#define ROCSPARSE_COMPLEX_TYPES_H

#if __cplusplus < 201402L || (!defined(__HIPCC__))

/* If this is a C compiler, C++ compiler below C++14, or a host-only compiler, only
   include minimal definitions of rocsparse_float_complex and rocsparse_double_complex */

/*! \brief Struct to represent a complex number with float precision real and imaginary parts. */
typedef struct
{
    float x; /**< real part. */
    float y; /**< imaginary part. */
} rocsparse_float_complex;

/*! \brief Struct to represent a complex number with double precision real and imaginary parts. */
typedef struct
{
    double x; /**< real part. */
    double y; /**< imaginary part. */
} rocsparse_double_complex;

#else /* __cplusplus < 201402L || (!defined(__HIPCC__)) */

// If this is a full internal build, add full support of complex arithmetic and classes
// including __host__ and __device__ and such we need to use <hip/hip_runtime.h>.

#include <cmath>
#include <complex>
#include <hip/hip_runtime.h>
#include <ostream>
#include <sstream>

template <typename T>
class rocsparse_complex_num
{
public:
    __device__ __host__ rocsparse_complex_num(void)                         = default;
    __device__ __host__ rocsparse_complex_num(const rocsparse_complex_num&) = default;
    template <typename U>
    __device__ __host__ rocsparse_complex_num(const rocsparse_complex_num<U>& a);
    __device__ __host__ rocsparse_complex_num(rocsparse_complex_num&&) = default;
    __device__ __host__ rocsparse_complex_num& operator=(const rocsparse_complex_num& rhs)
        = default;
    __device__ __host__ rocsparse_complex_num& operator=(rocsparse_complex_num&& rhs) = default;

    __device__ __host__ ~rocsparse_complex_num(void) = default;

    // Constructors
    __device__ __host__ rocsparse_complex_num(T r, T i)
        : x(r)
        , y(i)
    {
    }

    __device__ __host__ rocsparse_complex_num(T r)
        : x(r)
        , y(static_cast<T>(0))
    {
    }

    // Conversion from std::complex<T>
    __device__ __host__ rocsparse_complex_num(const std::complex<T>& z)
        : x(z.real())
        , y(z.imag())
    {
    }

    // Conversion to std::complex<T>
    __device__ __host__ operator std::complex<T>() const
    {
        return {x, y};
    }

    // Accessors
    friend __device__ __host__ T std::real(const rocsparse_complex_num& z);
    friend __device__ __host__ T std::imag(const rocsparse_complex_num& z);

    // Stream output
    friend auto& operator<<(std::ostream& out, const rocsparse_complex_num& z)
    {
        std::stringstream ss;
        ss << '(' << z.x << ',' << z.y << ')';
        return out << ss.str();
    }

    friend __device__ __host__ rocsparse_complex_num std::fma(rocsparse_complex_num p,
                                                              rocsparse_complex_num q,
                                                              rocsparse_complex_num r);
    friend __device__ __host__ rocsparse_complex_num std::conj(const rocsparse_complex_num& z);
    friend __device__ __host__ T                     std::abs(const rocsparse_complex_num<T>& z);

    // Unary operations
    __device__ __host__ rocsparse_complex_num operator-() const
    {
        return {-x, -y};
    }

    // In-place complex-complex operations
    __device__ __host__ auto& operator*=(const rocsparse_complex_num& rhs)
    {
        T real = fma(x, rhs.x, -y * rhs.y);
        T imag = fma(y, rhs.x, x * rhs.y);

        return *this = {real, imag};
    }

    __device__ __host__ auto& operator+=(const rocsparse_complex_num& rhs)
    {
        return *this = {x + rhs.x, y + rhs.y};
    }

    __device__ __host__ auto& operator-=(const rocsparse_complex_num& rhs)
    {
        return *this = {x - rhs.x, y - rhs.y};
    }

    __device__ __host__ auto& operator/=(const rocsparse_complex_num& rhs)
    {
        T sqabs = static_cast<T>(1) / fma(rhs.x, rhs.x, rhs.y * rhs.y);

        T real = fma(x, rhs.x, y * rhs.y) * sqabs;
        T imag = fma(y, rhs.x, -x * rhs.y) * sqabs;

        return *this = {real, imag};
    }

    // Out-of-place complex-complex operations
    __device__ __host__ auto operator+(const rocsparse_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs += rhs;
    }

    __device__ __host__ auto operator-(const rocsparse_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs -= rhs;
    }

    __device__ __host__ auto operator*(const rocsparse_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs *= rhs;
    }

    __device__ __host__ auto operator/(const rocsparse_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs /= rhs;
    }

    __device__ __host__ bool operator==(const rocsparse_complex_num& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }

    __device__ __host__ bool operator!=(const rocsparse_complex_num& rhs) const
    {
        return !(*this == rhs);
    }

private:
    // Internal real absolute function, to be sure we're on both device and host
    static __forceinline__ __device__ __host__ T abs(T x)
    {
        return x < 0 ? -x : x;
    }

    static __forceinline__ __device__ __host__ float sqrt(float x)
    {
        return ::sqrtf(x);
    }

    static __forceinline__ __device__ __host__ double sqrt(double x)
    {
        return ::sqrt(x);
    }

    static __forceinline__ __device__ __host__ float fma(float p, float q, float r)
    {
        return ::fmaf(p, q, r);
    }

    static __forceinline__ __device__ __host__ double fma(double p, double q, double r)
    {
        return ::fma(p, q, r);
    }

    T x;
    T y;
};

// Inject standard functions into namespace std
namespace std
{
    template <typename T>
    __device__ __host__ inline T real(const rocsparse_complex_num<T>& z)
    {
        return z.x;
    }

    template <typename T>
    __device__ __host__ inline T imag(const rocsparse_complex_num<T>& z)
    {
        return z.y;
    }

    template <typename T>
    __device__ __host__ inline rocsparse_complex_num<T>
        fma(rocsparse_complex_num<T> p, rocsparse_complex_num<T> q, rocsparse_complex_num<T> r)
    {
        T real = rocsparse_complex_num<T>::fma(
            -p.y, q.y, rocsparse_complex_num<T>::fma(p.x, q.x, r.x));
        T imag
            = rocsparse_complex_num<T>::fma(p.x, q.y, rocsparse_complex_num<T>::fma(p.y, q.x, r.y));

        return {real, imag};
    }

    template <typename T>
    __device__ __host__ inline rocsparse_complex_num<T> conj(const rocsparse_complex_num<T>& z)
    {
        return {z.x, -z.y};
    }

    template <typename T>
    __device__ __host__ inline T abs(const rocsparse_complex_num<T>& z)
    {
        T real = rocsparse_complex_num<T>::abs(z.x);
        T imag = rocsparse_complex_num<T>::abs(z.y);

        return real > imag ? (imag /= real, real * rocsparse_complex_num<T>::sqrt(imag * imag + 1))
               : imag      ? (real /= imag, imag * rocsparse_complex_num<T>::sqrt(real * real + 1))
                           : 0;
    }
}

// Test for C compatibility
template <typename T>
class rocsparse_complex_num_check
{
    static_assert(
        std::is_standard_layout<rocsparse_complex_num<T>>{},
        "rocsparse_complex_num<T> is not a standard layout type, and thus is incompatible with C.");
    static_assert(
        std::is_trivial<rocsparse_complex_num<T>>{},
        "rocsparse_complex_num<T> is not a trivial type, and thus is incompatible with C.");
    static_assert(
        sizeof(rocsparse_complex_num<T>) == 2 * sizeof(T),
        "rocsparse_complex_num<T> is not the correct size, and thus is incompatible with C.");
};

template class rocsparse_complex_num_check<float>;
template class rocsparse_complex_num_check<double>;

// rocSPARSE complex data types
using rocsparse_float_complex  = rocsparse_complex_num<float>;
using rocsparse_double_complex = rocsparse_complex_num<double>;

template <typename T>
template <typename U>
__device__ __host__
    rocsparse_complex_num<T>::rocsparse_complex_num(const rocsparse_complex_num<U>& a)
    : x(std::real(a))
    , y(std::imag(a))
{
}

#endif /* __cplusplus < 201402L || (!defined(__HIPCC__)) */

#endif /* ROCSPARSE_COMPLEX_TYPES_H */
