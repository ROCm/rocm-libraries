/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "rocblas.h"
#include "rocblas_math.hpp"
#include <cinttypes>
#include <random>
#include <thread>
#include <type_traits>

/* ============================================================================================ */
// Random number generator
using rocblas_rng_t = std::mt19937;

extern rocblas_rng_t   g_rocblas_seed;
extern std::thread::id g_main_thread_id;

extern thread_local rocblas_rng_t t_rocblas_rng;
extern thread_local int           t_rocblas_rand_idx;

// optimized helper
float rocblas_uniform_int_1_10();

void rocblas_uniform_int_1_10_run_float(float* ptr, size_t num);
void rocblas_uniform_int_1_10_run_double(double* ptr, size_t num);
void rocblas_uniform_int_1_10_run_float_complex(rocblas_float_complex* ptr, size_t num);
void rocblas_uniform_int_1_10_run_double_complex(rocblas_double_complex* ptr, size_t num);

// For the main thread, we use g_rocblas_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
inline rocblas_rng_t get_seed()
{
    auto tid = std::this_thread::get_id();
    return tid == g_main_thread_id ? g_rocblas_seed
                                   : rocblas_rng_t(std::hash<std::thread::id>{}(tid));
}

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocblas_seedrand()
{
    t_rocblas_rng      = get_seed();
    t_rocblas_rand_idx = 0;
}

/* ============================================================================================ */
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
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(t_rocblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(t_rocblas_rng);
    }

    // Random int8_t
    explicit operator int8_t()
    {
        return static_cast<int8_t>(std::uniform_int_distribution<int>{}(t_rocblas_rng));
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN half
    explicit operator rocblas_half()
    {
        return random_nan_data<rocblas_half, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator rocblas_bfloat16()
    {
        return random_nan_data<rocblas_bfloat16, uint16_t, 7, 8>();
    }

    explicit operator rocblas_float_complex()
    {
        return {float(*this), float(*this)};
    }

    explicit operator rocblas_double_complex()
    {
        return {double(*this), double(*this)};
    }
};

/* ============================================================================================ */
/*! \brief  Random number generator which generates denorm values */
class rocblas_denorm_rng
{
    // Generate random value
    static unsigned rand2()
    {
        return std::uniform_int_distribution<unsigned>(0, 1)(t_rocblas_rng);
    }

    // Generate random denorm values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_denorm_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union
        {
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(t_rocblas_rng);
        while(!(x.u &= (((UINT_T)1 << SIG) - 1))); // make exponent = 0 and random significand
        return (rand2() ? -(x.fp) : x.fp); // denorm with random sign bits
    }

public:
    // Random denorm double
    explicit operator double()
    {
        return random_denorm_data<double, uint64_t, 52, 11>();
    }

    // Random denorm float
    explicit operator float()
    {
        return random_denorm_data<float, uint32_t, 23, 8>();
    }

    // Random denorm half
    explicit operator rocblas_half()
    {
        return random_denorm_data<rocblas_half, uint16_t, 10, 5>();
    }

    // Random denorm bfloat16
    explicit operator rocblas_bfloat16()
    {
        return random_denorm_data<rocblas_bfloat16, uint16_t, 7, 8>();
    }

    explicit operator rocblas_float_complex()
    {
        return {float(*this), float(*this)};
    }

    explicit operator rocblas_double_complex()
    {
        return {double(*this), double(*this)};
    }
};

/* ============================================================================================ */
/*! \brief  Random number generator which generates Inf values */
class rocblas_inf_rng
{
    // Generate random Inf values
    static unsigned rand2()
    {
        return std::uniform_int_distribution<unsigned>(0, 1)(t_rocblas_rng);
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return rand2() ? std::numeric_limits<T>::min : std::numeric_limits<T>::max;
    }
    // Random float
    template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return T(rand2() ? -std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::infinity());
    }
};

/* ============================================================================================ */
/*! \brief  Random number generator which generates zero values */
class rocblas_zero_rng
{
    // Generate random zero values
    static unsigned rand2()
    {
        return std::uniform_int_distribution<unsigned>(0, 1)(t_rocblas_rng);
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return 0;
    }
    // Random float
    template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return T(rand2() ? -0.0 : 0.0);
    }
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random NaN number */
template <typename T>
inline T random_nan_generator()
{
    return T(rocblas_nan_rng{});
}

/*! \brief  generate a random Inf number */
template <typename T>
inline T random_inf_generator()
{
    return T(rocblas_inf_rng{});
}

/*! \brief  generate a random Inf number */
template <typename T>
inline T random_zero_generator()
{
    return T(rocblas_zero_rng{});
}

/*! \brief  generate a random denorm number */
template <typename T>
inline T random_denorm_generator()
{
    return T(rocblas_denorm_rng{});
}

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline T random_generator()
{
    return T(rocblas_uniform_int_1_10());
}

template <>
inline float random_generator()
{
    return rocblas_uniform_int_1_10();
}

// for rocblas_float_complex, generate two random ints (same behaviour as for floats)
template <>
inline rocblas_float_complex random_generator<rocblas_float_complex>()
{
    return {float(rocblas_uniform_int_1_10()), float(rocblas_uniform_int_1_10())};
};

// for rocblas_double_complex, generate two random ints (same behaviour as for doubles)
template <>
inline rocblas_double_complex random_generator<rocblas_double_complex>()
{
    return {double(rocblas_uniform_int_1_10()), double(rocblas_uniform_int_1_10())};
};

// for rocblas_half, generate float, and convert to rocblas_half
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocblas_half random_generator<rocblas_half>()
{
    return rocblas_half((float)std::uniform_int_distribution<int>(-2, 2)(t_rocblas_rng));
};

// for rocblas_bfloat16, generate float, and convert to rocblas_bfloat16
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocblas_bfloat16 random_generator<rocblas_bfloat16>()
{
    return rocblas_bfloat16((float)std::uniform_int_distribution<int>(-2, 2)(t_rocblas_rng));
};

/*! \brief  generate a random number in range [1,2,3] */
template <>
inline int8_t random_generator<int8_t>()
{
    return static_cast<int8_t>(std::uniform_int_distribution<unsigned short>(1, 3)(t_rocblas_rng));
};

/*! \brief  generate a sequence of random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline void random_run_generator(T* ptr, size_t num)
{
    for(size_t i = 0; i < num; i++)
    {
        ptr[i] = random_generator<T>();
    }
}

template <>
inline void random_run_generator<float>(float* ptr, size_t num)
{
    rocblas_uniform_int_1_10_run_float(ptr, num);
};

template <>
inline void random_run_generator<double>(double* ptr, size_t num)
{
    rocblas_uniform_int_1_10_run_double(ptr, num);
};

template <>
inline void random_run_generator<rocblas_float_complex>(rocblas_float_complex* ptr, size_t num)
{
    rocblas_uniform_int_1_10_run_float_complex(ptr, num);
};

template <>
inline void random_run_generator<rocblas_double_complex>(rocblas_double_complex* ptr, size_t num)
{
    rocblas_uniform_int_1_10_run_double_complex(ptr, num);
};

// HPL

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(t_rocblas_rng);
}

// for rocblas_bfloat16, generate float, and convert to rocblas_bfloat16
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <>
inline rocblas_bfloat16 random_hpl_generator()
{
    return rocblas_bfloat16(std::uniform_real_distribution<float>(-0.5, 0.5)(t_rocblas_rng));
}

/*! \brief  generate a random ASCII string of up to length n */
inline std::string random_string(size_t n)
{
    std::string str;
    if(n)
    {
        size_t len = std::uniform_int_distribution<size_t>(1, n)(t_rocblas_rng);
        str.reserve(len);
        for(size_t i = 0; i < len; ++i)
            str.push_back(static_cast<char>(
                std::uniform_int_distribution<unsigned short>(0x20, 0x7E)(t_rocblas_rng)));
    }
    return str;
}

template <typename T>
inline T random_zero_one_generator()
{
    return (std::uniform_int_distribution<int>(0, 1)(t_rocblas_rng));
};

// NOTE: need to convert it into float first, otherwise the bits may be reinterpreted incorrectly
template <>
inline rocblas_half random_zero_one_generator<rocblas_half>()
{
    return rocblas_half((float)std::uniform_int_distribution<int>(0, 1)(t_rocblas_rng));
}

template <>
inline rocblas_bfloat16 random_zero_one_generator<rocblas_bfloat16>()
{
    return rocblas_bfloat16((float)std::uniform_int_distribution<int>(0, 1)(t_rocblas_rng));
}
