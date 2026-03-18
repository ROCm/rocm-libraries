/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef ROCSPARSE_RANDOM_HPP
#define ROCSPARSE_RANDOM_HPP

#include "rocsparse_math.hpp"

#include <random>
#include <type_traits>

/* ==================================================================================== */
// Random number generator

using rocsparse_rng_t = std::mt19937;

void rocsparse_rng_set(rocsparse_rng_t a);

void rocsparse_seed_set(rocsparse_rng_t a);

void rocsparse_rng_nan_set(rocsparse_rng_t a);

rocsparse_rng_t& rocsparse_rng_get();

rocsparse_rng_t& rocsparse_seed_get();

rocsparse_rng_t& rocsparse_rng_nan_get();

// extern  rocsparse_rng_t rocsparse_rng, rocsparse_seed, rocsparse_rng_nan;

extern int rocsparse_rand_uniform_idx;
extern int rocsparse_rand_normal_idx;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
void rocsparse_seedrand();

int    rocsparse_uniform_int(int a, int b);
float  rocsparse_uniform_float(float a, float b);
double rocsparse_uniform_double(double a, double b);
double rocsparse_normal_double();

/* ==================================================================================== */
/*! \brief  Random number generator which generates NaN values */
class rocsparse_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union u_t
        {
            u_t() {}
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(rocsparse_rng_nan_get());
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(rocsparse_rng_nan_get());
    }

    // Random int8_t
    explicit operator int8_t()
    {
        return (int8_t)std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(),
                                                          std::numeric_limits<int8_t>::max())(
            rocsparse_rng_nan_get());
    }

    // Random char
    explicit operator char()
    {
        return (char)std::uniform_int_distribution<int>(std::numeric_limits<char>::min(),
                                                        std::numeric_limits<char>::max())(
            rocsparse_rng_nan_get());
    }

    // Random NaN half
    explicit operator _Float16()
    {
        return random_nan_data<_Float16, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator rocsparse_bfloat16()
    {
        return random_nan_data<rocsparse_bfloat16, uint16_t, 7, 8>();
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    explicit operator rocsparse_float_complex()
    {
        return {float(*this), float(*this)};
    }

    explicit operator rocsparse_double_complex()
    {
        return {double(*this), double(*this)};
    }
};

/* ==================================================================================== */
/* generate random number :*/

/*! \brief  generate a random number in range [a,b] using integer numbers*/
template <typename T>
inline T random_generator_exact(int a = 1, int b = 10)
{
    return std::uniform_int_distribution<int>(a, b)(rocsparse_rng_get());
}

template <>
inline rocsparse_float_complex random_generator_exact<rocsparse_float_complex>(int a, int b)
{
    return rocsparse_float_complex(random_generator_exact<float>(a, b),
                                   random_generator_exact<float>(a, b));
}

template <>
inline rocsparse_double_complex random_generator_exact<rocsparse_double_complex>(int a, int b)
{
    return rocsparse_double_complex(random_generator_exact<double>(a, b),
                                    random_generator_exact<double>(a, b));
}

/*! \brief  generate a random number in range [a,b]*/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline T random_generator(T a = static_cast<T>(1), T b = static_cast<T>(10))
{
    return random_generator_exact<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline T random_generator(T a = static_cast<T>(0), T b = static_cast<T>(1))
{
    return std::uniform_real_distribution<T>(a, b)(rocsparse_rng_get());
}

template <>
inline rocsparse_float_complex random_generator<rocsparse_float_complex>(rocsparse_float_complex a,
                                                                         rocsparse_float_complex b)
{
    float theta = random_generator<float>(0.0f, 2.0f * acos(-1.0f));
    float r     = random_generator<float>(std::abs(a), std::abs(b));

    return rocsparse_float_complex(r * cos(theta), r * sin(theta));
}

template <>
inline rocsparse_double_complex
    random_generator<rocsparse_double_complex>(rocsparse_double_complex a,
                                               rocsparse_double_complex b)
{
    double theta = random_generator<double>(0.0, 2.0 * acos(-1.0));
    double r     = random_generator<double>(std::abs(a), std::abs(b));

    return rocsparse_double_complex(r * cos(theta), r * sin(theta));
}

/*! \brief  generate a random number in range [a,b] from a predetermined finite cache using integer numbers*/
template <typename T>
inline T random_cached_generator_exact(int a = 1, int b = 10)
{
    return rocsparse_uniform_int(a, b);
}

template <>
inline float random_cached_generator_exact(int a, int b)
{
    return static_cast<float>(rocsparse_uniform_int(a, b));
}

template <>
inline double random_cached_generator_exact(int a, int b)
{
    return static_cast<double>(rocsparse_uniform_int(a, b));
}

template <>
inline rocsparse_float_complex random_cached_generator_exact<rocsparse_float_complex>(int a, int b)
{
    return rocsparse_float_complex(random_cached_generator_exact<float>(a, b),
                                   random_cached_generator_exact<float>(a, b));
}

template <>
inline rocsparse_double_complex random_cached_generator_exact<rocsparse_double_complex>(int a,
                                                                                        int b)
{
    return rocsparse_double_complex(random_cached_generator_exact<double>(a, b),
                                    random_cached_generator_exact<double>(a, b));
}

/*! \brief  generate a random number in range [a,b] from a predetermined finite cache*/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline T random_cached_generator(T a = static_cast<T>(1), T b = static_cast<T>(10))
{
    return random_cached_generator_exact<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline T random_cached_generator(T a = static_cast<T>(0), T b = static_cast<T>(1))
{
    return static_cast<T>(rocsparse_uniform_float(a, b));
}

template <>
inline double random_cached_generator(double a, double b)
{
    return rocsparse_uniform_double(a, b);
}

template <>
inline rocsparse_float_complex
    random_cached_generator<rocsparse_float_complex>(rocsparse_float_complex a,
                                                     rocsparse_float_complex b)
{
    float theta = random_cached_generator<float>(0.0f, 2.0f * acos(-1.0f));
    float r     = random_cached_generator<float>(std::abs(a), std::abs(b));

    return rocsparse_float_complex(r * cos(theta), r * sin(theta));
}

template <>
inline rocsparse_double_complex
    random_cached_generator<rocsparse_double_complex>(rocsparse_double_complex a,
                                                      rocsparse_double_complex b)
{
    double theta = random_cached_generator<double>(0.0, 2.0 * acos(-1.0));
    double r     = random_cached_generator<double>(std::abs(a), std::abs(b));

    return rocsparse_double_complex(r * cos(theta), r * sin(theta));
}

/*! \brief generate a random normally distributed number around 0 with stddev 1 from a predetermined finite cache */
template <typename T>
inline T random_cached_generator_normal()
{
    return static_cast<T>(rocsparse_normal_double());
}

#endif // ROCSPARSE_RANDOM_HPP
