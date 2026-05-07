/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_math.hpp"

#include <array>
#include <random>
#include <type_traits>

using rocsparse_rng_t = std::mt19937;

namespace rocsparse
{
    class rng_t
    {
    private:
        rng_t();
        ~rng_t()            = default;
        rng_t(const rng_t&) = delete;
        rng_t& operator=(const rng_t&) = delete;

        static constexpr size_t s_rand_cache_size = 1024;
        static_assert(s_rand_cache_size > 0 && (s_rand_cache_size & (s_rand_cache_size - 1)) == 0,
                      "s_rand_cache_size must be a positive power of two "
                      "(required by the bitmask used in cache index updates)");

        rocsparse_rng_t m_rng;
        rocsparse_rng_t m_rng_nan;
        rocsparse_rng_t m_rng_seed;

        size_t                                m_rand_uniform_idx = s_rand_cache_size - 1;
        size_t                                m_rand_normal_idx  = s_rand_cache_size - 1;
        std::array<double, s_rand_cache_size> m_rand_uniform_cache;
        std::array<double, s_rand_cache_size> m_rand_normal_cache;

        double uniform_double(double a, double b);
        double normal_double();

    public:
        static rng_t& Instance();

        void reset_seed();

        void set_rng(rocsparse_rng_t a);
        void set_rng_nan(rocsparse_rng_t a);
        void set_rng_seed(rocsparse_rng_t a);

        rocsparse_rng_t& get_rng();
        rocsparse_rng_t& get_rng_nan();
        rocsparse_rng_t& get_rng_seed();

        /*! \brief  generate a random number in range [a,b]*/
        template <typename T>
        T generator(T a, T b);

        /*! \brief  generate an exact random number in range [a,b]*/
        template <typename T>
        T generator_exact(int a, int b);

        /*! \brief  generate a random number in range [a,b] from a predetermined finite cache*/
        template <typename T>
        T cached_generator(T a, T b);

        /*! \brief  generate an exact random number in range [a,b] from a predetermined finite cache*/
        template <typename T>
        T cached_generator_exact(int a, int b);

        /*! \brief generate a random normally distributed number around 0 with stddev 1 from a predetermined finite cache */
        template <typename T>
        T cached_generator_normal();
    };
}

/*
 * ===========================================================================
 *    Wrappers
 * ===========================================================================
 */

/*! \brief  generate a random number in range [a,b] using integer numbers*/
template <typename T>
T random_generator_exact(int a = 1, int b = 10);

// /*! \brief  generate a random number in range [a,b]*/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
T random_generator(T a = static_cast<T>(1), T b = static_cast<T>(10));

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool> = true>
T random_generator(T a = static_cast<T>(0), T b = static_cast<T>(1));

/*! \brief  generate a random number in range [a,b] from a predetermined finite cache using integer numbers*/
template <typename T>
T random_cached_generator_exact(int a = 1, int b = 10);

/*! \brief  generate a random number in range [a,b] from a predetermined finite cache*/
template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
T random_cached_generator(T a = static_cast<T>(1), T b = static_cast<T>(10));

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool> = true>
T random_cached_generator(T a = static_cast<T>(0), T b = static_cast<T>(1));

/*! \brief generate a random normally distributed number around 0 with stddev 1 from a predetermined finite cache */
template <typename T>
T random_cached_generator_normal();

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
void rocsparse_seedrand();

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
            x.u = std::uniform_int_distribution<UINT_T>{}(
                rocsparse::rng_t::Instance().get_rng_nan());
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(rocsparse::rng_t::Instance().get_rng_nan());
    }

    // Random int8_t
    explicit operator int8_t()
    {
        return (int8_t)std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(),
                                                          std::numeric_limits<int8_t>::max())(
            rocsparse::rng_t::Instance().get_rng_nan());
    }

    // Random char
    explicit operator char()
    {
        return (char)std::uniform_int_distribution<int>(std::numeric_limits<char>::min(),
                                                        std::numeric_limits<char>::max())(
            rocsparse::rng_t::Instance().get_rng_nan());
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
