/* **************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas/rocblas.h"
#include "rocblas_math.hpp"
#include <cinttypes>
#include <random>
#include <thread>
#include <type_traits>

/* ============================================================================================
 */
// Random number generator
using rocblas_rng_t = std::mt19937;
extern thread_local rocblas_rng_t rocblas_rng;
extern const rocblas_rng_t rocblas_seed;
extern const std::thread::id main_thread_id;

// For the main thread, we use rocblas_seed; for other threads, we start with a
// different seed but deterministically based on the thread id's hash function.
inline rocblas_rng_t get_seed()
{
    auto tid = std::this_thread::get_id();
    return tid == main_thread_id ? rocblas_seed : rocblas_rng_t(std::hash<std::thread::id>{}(tid));
}

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocblas_seedrand()
{
    rocblas_rng = get_seed();
}

/* ============================================================================================
 */
/*! \brief  Random number generator which generates NaN values */
class rocblas_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union
        {
            UINT_T u;
            T fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(rocblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T() const
    {
        return std::uniform_int_distribution<T>{}(rocblas_rng);
    }

    // Random NaN double
    explicit operator double() const
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    explicit operator float() const
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN half
    explicit operator rocblas_half() const
    {
        return random_nan_data<rocblas_half, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator rocblas_bfloat16() const
    {
        return random_nan_data<rocblas_bfloat16, uint16_t, 7, 8>();
    }

    explicit operator rocblas_float_complex() const
    {
        return {float(*this), float(*this)};
    }

    explicit operator rocblas_double_complex() const
    {
        return {double(*this), double(*this)};
    }
};

/* ============================================================================================
 */
/* generate random number :*/

// generate a random number in range [l1, l2] */
template <typename T>
inline T random_generator(const rocblas_int l1 = 1, const rocblas_int l2 = 10)
{
    return std::uniform_int_distribution<int>(l1, l2)(rocblas_rng);
}

// for rocblas_float_complex, generate two random ints (same behaviour as for
// floats)
template <>
inline rocblas_float_complex random_generator<rocblas_float_complex>(const rocblas_int l1,
                                                                     const rocblas_int l2)
{
    return {float(std::uniform_int_distribution<int>(l1, l2)(rocblas_rng)),
            float(std::uniform_int_distribution<int>(l1, l2)(rocblas_rng))};
};

// for rocblas_double_complex, generate two random ints (same behaviour as for
// doubles)
template <>
inline rocblas_double_complex random_generator<rocblas_double_complex>(const rocblas_int l1,
                                                                       const rocblas_int l2)
{
    return {double(std::uniform_int_distribution<int>(l1, l2)(rocblas_rng)),
            double(std::uniform_int_distribution<int>(l1, l2)(rocblas_rng))};
};

/*// for rocblas_half, generate float, and convert to rocblas_half
template <>
inline rocblas_half random_generator<rocblas_half>()
{
    return rocblas_half(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng));
};

// for rocblas_bfloat16, generate float, and convert to rocblas_bfloat16
template <>
inline rocblas_bfloat16 random_generator<rocblas_bfloat16>()
{
    return rocblas_bfloat16(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng));
};

//  generate a random number in range [1,2,3]
template <>
inline int8_t random_generator<int8_t>()
{
    return int8_t(std::uniform_int_distribution<int>(1, 3)(rocblas_rng));
};*/

//  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(rocblas_rng);
}
