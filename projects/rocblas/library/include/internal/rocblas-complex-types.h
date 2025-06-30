/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

/*! \file
 * \brief rocblas-complex-types.h defines complex data types used by rocblas
 */

#ifndef ROCBLAS_COMPLEX_TYPES_H
#define ROCBLAS_COMPLEX_TYPES_H

/* Workaround clang bug:

   https://bugs.llvm.org/show_bug.cgi?id=35863

   This macro expands to static if clang is used; otherwise it expands empty.
   It is intended to be used in variable template specializations, where clang
   requires static in order for the specializations to have internal linkage,
   while technically, storage class specifiers besides thread_local are not
   allowed in template specializations, and static in the primary template
   definition should imply internal linkage for all specializations.

   If clang shows an error for improperly using a storage class specifier in
   a specialization, then ROCBLAS_CLANG_STATIC should be redefined as empty,
   and perhaps removed entirely, if the above bug has been fixed.
*/
#if __clang__
#define ROCBLAS_CLANG_STATIC static
#else
#define ROCBLAS_CLANG_STATIC
#endif

#if defined(ROCM_MATHLIBS_API_USE_HIP_COMPLEX)
// If this is not internal use, or a C compiler, C++ compiler below C++14, or we expose the API
// as using hip types for rocblas_float_complex and rocblas_double_complex
// This allows for consistency of type reuse across hipified applications and other libraries like hipBLAS

#include <hip/hip_complex.h>

/*! \brief hip type to represent a complex number with single precision real and imaginary parts. */
typedef hipFloatComplex rocblas_float_complex;

/*! \brief hip type to represent a complex number with double precision real and imaginary parts. */
typedef hipDoubleComplex rocblas_double_complex;

#if __cplusplus >= 201402L

// Test for compatibility with hipComplex API option when using C++14 or later

static_assert(sizeof(hipFloatComplex) == 2 * sizeof(float),
              "hipFloatComplex is not compatible with rocblas internal float complex.");

static_assert(sizeof(hipDoubleComplex) == 2 * sizeof(double),
              "hipDoubleComplex is not compatible with rocblas internal double complex.");

#endif

#elif __cplusplus < 201402L || (!defined(__HCC__) && !defined(__HIPCC__))

// If this is a C compiler, C++ compiler below C++14, or a host-only compiler, we only
// include minimal definitions of rocblas_float_complex and rocblas_double_complex

/*! \brief Struct to represent a complex number with single precision real and imaginary parts. */
typedef struct
{
    float x, y;
} rocblas_float_complex;

/*! \brief Struct to represent a complex number with double precision real and imaginary parts. */
typedef struct
{
    double x, y;
} rocblas_double_complex;

#else // __cplusplus < 201402L || (!defined(__HCC__) && !defined(__HIPCC__))

// If this a full internal build, we need full support of complex arithmetic
// and classes. We need __host__ and __device__ so we use <hip/hip_runtime.h>.

#include <complex>
#include <math.h>
#include <ostream>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "rocblas-export.h"

/*! \brief rocblas_complex_num is a structure which represents a complex number
 *         with precision T.
 */
template <typename T>
class ROCBLAS_EXPORT rocblas_complex_num
{
    T x; // The real part of the number.
    T y; // The imaginary part of the number.

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

public:
    // We do not initialize the members x or y by default, to ensure that it can
    // be used in __shared__ and that it is a trivial class compatible with C.
    __device__ __host__ rocblas_complex_num()                           = default;
    __device__ __host__ rocblas_complex_num(const rocblas_complex_num&) = default;
    __device__ __host__ rocblas_complex_num(rocblas_complex_num&&)      = default;
    __device__ __host__ rocblas_complex_num& operator=(const rocblas_complex_num& rhs) & = default;
    __device__ __host__ rocblas_complex_num& operator=(rocblas_complex_num&& rhs) & = default;
    __device__                               __host__ ~rocblas_complex_num()        = default;
    using value_type                                                                = T;

    // Constructor
    __device__ __host__ constexpr rocblas_complex_num(T r, T i)
        : x{r}
        , y{i}
    {
    }

    // Conversion from real
    // TODO: Make constexpr after HSA_STATUS_ERROR_INVALID_ISA bug goes away
    __device__ __host__ rocblas_complex_num(T r)
        : x{r}
        , y{0}
    {
    }

    // Conversion from std::complex<T>
    __device__ __host__ constexpr rocblas_complex_num(const std::complex<T>& z)
        : x{z.real()}
        , y{z.imag()}
    {
    }

    // Conversion to std::complex<T>
    __device__ __host__ constexpr operator std::complex<T>() const
    {
        return {x, y};
    }

    /*  possible transition helpers for use of both internal and hipComplex

    // Conversion from hipFloatComplex
    template <typename U, std::enable_if_t<std::is_same_v<T, float> && std::is_same_v<U, hipFloatComplex>, int> = 0>
    __device__ __host__ explicit constexpr rocblas_complex_num(const U& z)
        : x{z.x()}
        , y{z.y()}
    {
    }

    // Conversion to hipFloatComplex
    template <typename U, std::enable_if_t<std::is_same_v<T, float> && std::is_same_v<U, hipFloatComplex>, int> = 0>
    __device__ __host__ explicit constexpr operator U() const
    {
        return {x, y};
    }

    // Conversion from hipDoubleComplex
    template <typename U, std::enable_if_t<std::is_same_v<T, double> && std::is_same_v<U, hipDoubleComplex>, int> = 0>
    __device__ __host__ explicit constexpr rocblas_complex_num(const U& z)
        : x{z.x()}
        , y{z.y()}
    {
    }

    // Conversion to hipDoubleComplex
    template <typename U, std::enable_if_t<std::is_same_v<T, double> && std::is_same_v<U, hipDoubleComplex>, int> = 0>
    __device__ __host__ explicit constexpr operator U() const
    {
        return {x, y};
    }

    */

    // Conversion from different complex (explicit)
    template <typename U, std::enable_if_t<std::is_constructible<T, U>{}, int> = 0>
    __device__ __host__ explicit constexpr rocblas_complex_num(const rocblas_complex_num<U>& z)
        : x(z.real())
        , y(z.imag())
    {
    }

    // Conversion to bool
    __device__ __host__ constexpr explicit operator bool() const
    {
        return x || y;
    }

    // Setters like C++20
    __device__ __host__ constexpr void real(T r)
    {
        x = r;
    }

    __device__ __host__ constexpr void imag(T i)
    {
        y = i;
    }

    // Accessors
    friend __device__ __host__ T std::real(const rocblas_complex_num& z);
    friend __device__ __host__ T std::imag(const rocblas_complex_num& z);

    __device__ __host__ constexpr T real() const
    {
        return x;
    }

    __device__ __host__ constexpr T imag() const
    {
        return y;
    }

    // stream output
    friend auto& operator<<(std::ostream& out, const rocblas_complex_num& z)
    {
        return out << '(' << z.x << ',' << z.y << ')';
    }

    // complex-real operations
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ auto& operator+=(const U& rhs)
    {
        return (x += T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ auto& operator-=(const U& rhs)
    {
        return (x -= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ auto& operator*=(const U& rhs)
    {
        return (x *= rhs), (y *= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ auto& operator/=(const U& rhs)
    {
        return (x /= T(rhs)), (y /= T(rhs)), *this;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ constexpr bool operator==(const U& rhs) const
    {
        return x == T(rhs) && y == 0;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    __device__ __host__ constexpr bool operator!=(const U& rhs) const
    {
        return !(*this == rhs);
    }

    // Increment and decrement
    __device__ __host__ auto& operator++()
    {
        return ++x, *this;
    }

    __device__ __host__ rocblas_complex_num operator++(int)
    {
        return {x++, y};
    }

    __device__ __host__ auto& operator--()
    {
        return --x, *this;
    }

    __device__ __host__ rocblas_complex_num operator--(int)
    {
        return {x--, y};
    }

    // Unary operations
    __forceinline__ __device__ __host__ rocblas_complex_num operator-() const
    {
        return {-x, -y};
    }

    __forceinline__ __device__ __host__ rocblas_complex_num operator+() const
    {
        return *this;
    }

    friend __device__ __host__ T asum(const rocblas_complex_num& z)
    {
        return abs(z.x) + abs(z.y);
    }

    friend __device__ __host__ rocblas_complex_num std::conj(const rocblas_complex_num& z);
    friend __device__ __host__ T                   std::norm(const rocblas_complex_num& z);
    friend __device__ __host__ T                   std::abs(const rocblas_complex_num<T>& z);

    // in-place complex-complex operations
    __device__ __host__ auto& operator*=(const rocblas_complex_num& rhs)
    {
        return *this = {x * rhs.x - y * rhs.y, y * rhs.x + x * rhs.y};
    }

    __device__ __host__ auto& operator+=(const rocblas_complex_num& rhs)
    {
        return *this = {x + rhs.x, y + rhs.y};
    }

    __device__ __host__ auto& operator-=(const rocblas_complex_num& rhs)
    {
        return *this = {x - rhs.x, y - rhs.y};
    }

    __device__ __host__ auto& operator/=(const rocblas_complex_num& rhs)
    {
        // Form of Robert L. Smith's Algorithm 116
        if(abs(rhs.x) > abs(rhs.y))
        {
            T ratio = rhs.y / rhs.x;
            T scale = 1 / (rhs.x + rhs.y * ratio);
            *this   = {(x + y * ratio) * scale, (y - x * ratio) * scale};
        }
        else
        {
            T ratio = rhs.x / rhs.y;
            T scale = 1 / (rhs.x * ratio + rhs.y);
            *this   = {(y + x * ratio) * scale, (y * ratio - x) * scale};
        }
        return *this;
    }

    // out-of-place complex-complex operations
    __device__ __host__ auto operator+(const rocblas_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs += rhs;
    }

    __device__ __host__ auto operator-(const rocblas_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs -= rhs;
    }

    __device__ __host__ auto operator*(const rocblas_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs *= rhs;
    }

    __device__ __host__ auto operator/(const rocblas_complex_num& rhs) const
    {
        auto lhs = *this;
        return lhs /= rhs;
    }

    __device__ __host__ constexpr bool operator==(const rocblas_complex_num& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }

    __device__ __host__ constexpr bool operator!=(const rocblas_complex_num& rhs) const
    {
        return !(*this == rhs);
    }

    // real-complex operations (complex-real is handled above)
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ rocblas_complex_num operator+(const U&                   lhs,
                                                             const rocblas_complex_num& rhs)
    {
        return {T(lhs) + rhs.x, rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ rocblas_complex_num operator-(const U&                   lhs,
                                                             const rocblas_complex_num& rhs)
    {
        return {T(lhs) - rhs.x, -rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ rocblas_complex_num operator*(const U&                   lhs,
                                                             const rocblas_complex_num& rhs)
    {
        return {T(lhs) * rhs.x, T(lhs) * rhs.y};
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ rocblas_complex_num operator/(const U&                   lhs,
                                                             const rocblas_complex_num& rhs)
    {
        // Form of Robert L. Smith's Algorithm 116
        if(abs(rhs.x) > abs(rhs.y))
        {
            T ratio = rhs.y / rhs.x;
            T scale = T(lhs) / (rhs.x + rhs.y * ratio);
            return {scale, -scale * ratio};
        }
        else
        {
            T ratio = rhs.x / rhs.y;
            T scale = T(lhs) / (rhs.x * ratio + rhs.y);
            return {ratio * scale, -scale};
        }
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ constexpr bool operator==(const U&                   lhs,
                                                         const rocblas_complex_num& rhs)
    {
        return T(lhs) == rhs.x && 0 == rhs.y;
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    friend __device__ __host__ constexpr bool operator!=(const U&                   lhs,
                                                         const rocblas_complex_num& rhs)
    {
        return !(lhs == rhs);
    }
};

// Inject standard functions into namespace std
namespace std
{
    template <typename T>
    __device__ __host__ constexpr T real(const rocblas_complex_num<T>& z)
    {
        return z.x;
    }

    template <typename T>
    __device__ __host__ constexpr T imag(const rocblas_complex_num<T>& z)
    {
        return z.y;
    }

    template <typename T>
    __device__ __host__ constexpr rocblas_complex_num<T> conj(const rocblas_complex_num<T>& z)
    {
        return {z.x, -z.y};
    }

    template <typename T>
    __device__ __host__ inline T norm(const rocblas_complex_num<T>& z)
    {
        return (z.x * z.x) + (z.y * z.y);
    }

    template <typename T>
    __device__ __host__ inline T abs(const rocblas_complex_num<T>& z)
    {
        T tr = rocblas_complex_num<T>::abs(z.x), ti = rocblas_complex_num<T>::abs(z.y);
        // clang-format off
        return tr > ti ? (ti /= tr, tr * rocblas_complex_num<T>::sqrt(ti * ti + 1))
                       : ti ? (tr /= ti, ti * rocblas_complex_num<T>::sqrt(tr * tr + 1)) : 0;
        // clang-format on
    }
}

// Test for C compatibility
template <typename T>
class rocblas_complex_num_check
{
    static_assert(
        std::is_standard_layout<rocblas_complex_num<T>>{},
        "rocblas_complex_num<T> is not a standard layout type, and thus is incompatible with C.");

    static_assert(std::is_trivial<rocblas_complex_num<T>>{},
                  "rocblas_complex_num<T> is not a trivial type, and thus is incompatible with C.");

    static_assert(
        sizeof(rocblas_complex_num<T>) == 2 * sizeof(T),
        "rocblas_complex_num<T> is not the correct size, and thus is incompatible with C.");
};

template class rocblas_complex_num_check<float>;
template class rocblas_complex_num_check<double>;

// rocBLAS complex data types
using rocblas_float_complex  = rocblas_complex_num<float>;
using rocblas_double_complex = rocblas_complex_num<double>;

/*! \brief rocblas_is_complex<T> returns true iff T is complex */
template <typename T>
inline constexpr bool rocblas_is_complex = false;

template <>
inline constexpr bool rocblas_is_complex<rocblas_float_complex> = true;

template <>
inline constexpr bool rocblas_is_complex<rocblas_double_complex> = true;

//!
//! @brief Struct to define pair of value and index.
//!
template <typename T>
struct ROCBLAS_INTERNAL_EXPORT rocblas_index_value_t
{
    //! @brief Important: index must come first, so that rocblas_index_value_t* can be cast to rocblas_int*
    rocblas_int index;
    //! @brief The value.
    T value;
};

//!
//! @brief Struct to define pair of value and 64-bit index.
//!
template <typename T>
struct ROCBLAS_INTERNAL_EXPORT rocblas_index_64_value_t
{
    //! @brief Important: index must come first, so that rocblas_index_64_value_t* can be cast to int64_t*
    int64_t index;
    //! @brief The value.
    T value;
};

#endif // using internal complex class for API

#endif /* ROCBLAS_COMPLEX_TYPES_H */
