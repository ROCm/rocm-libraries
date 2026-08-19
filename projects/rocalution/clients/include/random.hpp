/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_RANDOM_HPP
#define TESTING_RANDOM_HPP

#include <algorithm>
#include <complex>
#include <random>
#include <string>

/* ==================================================================================== */
// Random number generator

using rocalution_rng_t = std::mt19937;

void rocalution_rng_set(rocalution_rng_t a);

void rocalution_seed_set(rocalution_rng_t a);

rocalution_rng_t& rocalution_rng_get();

rocalution_rng_t& rocalution_seed_get();

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocalution_seedrand()
{
    rocalution_rng_set(rocalution_seed_get());
}

/* ==================================================================================== */
/* generate random number :*/

/*! \brief  generate a random number in range [a,b] */

template <typename T>
inline T random_generator_exact(int a = 1, int b = 10)
{
    return std::uniform_int_distribution<int>(a, b)(rocalution_rng_get());
}

template <>
inline std::complex<float> random_generator_exact<std::complex<float>>(int a, int b)
{
    return std::complex<float>(random_generator_exact<float>(a, b),
                               random_generator_exact<float>(a, b));
}

template <>
inline std::complex<double> random_generator_exact<std::complex<double>>(int a, int b)
{
    return std::complex<double>(random_generator_exact<double>(a, b),
                                random_generator_exact<double>(a, b));
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline T random_generator(T a = static_cast<T>(1), T b = static_cast<T>(10))
{
    return random_generator_exact<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline T random_generator(T a = static_cast<T>(0), T b = static_cast<T>(1))
{
    return std::uniform_real_distribution<T>(a, b)(rocalution_rng_get());
}

template <>
inline std::complex<float> random_generator<std::complex<float>>(std::complex<float> a,
                                                                 std::complex<float> b)
{
    float theta = random_generator<float>(0.0f, 2.0f * acos(-1.0f));
    float r     = random_generator<float>(std::abs(a), std::abs(b));

    return std::complex<float>(r * cos(theta), r * sin(theta));
}

template <>
inline std::complex<double> random_generator<std::complex<double>>(std::complex<double> a,
                                                                   std::complex<double> b)
{
    double theta = random_generator<double>(0.0, 2.0 * acos(-1.0));
    double r     = random_generator<double>(std::abs(a), std::abs(b));

    return std::complex<double>(r * cos(theta), r * sin(theta));
}

#endif // TESTING_RANDOM_HPP
