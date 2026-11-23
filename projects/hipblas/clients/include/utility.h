/* ************************************************************************
 * Copyright (C) 2016-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <hipblas/hipblas.h>
#include <stdbool.h>

#include "type_utils.h"
#ifdef __cplusplus
#include "hipblas_datatype2string.hpp"
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>
#endif

#include <stdio.h>
#include <stdlib.h>
#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

/*!\file
 * \brief provide data initialization, timing, hipblas type <-> lapack char conversion utilities.
 */

#ifdef __cplusplus

#define BLAS_1_RESULT_PRINT                                \
    do                                                     \
    {                                                      \
        if(arg.timing)                                     \
        {                                                  \
            std::cout << "N, hipblas (us), ";              \
            if(arg.norm_check)                             \
            {                                              \
                std::cout << "CPU (us), error";            \
            }                                              \
            std::cout << std::endl;                        \
            std::cout << N << ',' << gpu_time_used << ','; \
            if(arg.norm_check)                             \
            {                                              \
                std::cout << cpu_time_used << ',';         \
                std::cout << hipblas_error;                \
            }                                              \
            std::cout << std::endl;                        \
        }                                                  \
    } while(0)

/* =============================================================================================== */
/* Epsilon helpers for near checks.                                                                */
template <typename>
inline constexpr double hipblas_type_epsilon = 0;
template <>
inline constexpr double hipblas_type_epsilon<float> = std::numeric_limits<float>::epsilon();
template <>
inline constexpr double hipblas_type_epsilon<double> = std::numeric_limits<double>::epsilon();
template <>
inline constexpr double
    hipblas_type_epsilon<std::complex<float>> = std::numeric_limits<float>::epsilon();
template <>
inline constexpr double
    hipblas_type_epsilon<std::complex<double>> = std::numeric_limits<double>::epsilon();
template <>
inline constexpr double hipblas_type_epsilon<
    hipblasHalf> = 0.0009765625; // in fp16 diff between 0x3C00 (1.0) and fp16 0x3C01
template <>
inline constexpr double hipblas_type_epsilon<
    hipblasBfloat16> = 0.0078125; // in bf16 diff between 0x3F80 (1.0) and bf16 0x3F81 in double precision

/* =============================================================================================== */
/* 64-bit value which will overflow 32-bit integers                                                */
extern int64_t c_i32_overflow;

/* =============================================================================================== */
/* For GTEST_SKIP() we search for these sub-strings in listener to determine skip category         */
#define LIMITED_RAM_STRING "skip: RAM"
#define LIMITED_VRAM_STRING "skip: VRAM"

/* ============================================================================================ */
/*! \brief  Random number generation for the client test data */

// Random number engine for the client test data (SplitMix64). Its state is one 64-bit word, so it
// is cheap to seed; the parallel fills reseed it for every block of elements they write.
// Satisfies UniformRandomBitGenerator, so it works with the std:: distributions.
class hipblas_rng_t
{
public:
    using result_type = uint64_t;

    explicit hipblas_rng_t(uint64_t seed = 0)
        : m_state(seed)
    {
    }

    static constexpr result_type min()
    {
        return 0;
    }

    static constexpr result_type max()
    {
        return ~result_type(0);
    }

    result_type operator()()
    {
        uint64_t z = (m_state += 0x9e3779b97f4a7c15ull);
        z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

private:
    uint64_t m_state;
};

// The fixed seed every random sequence starts from
extern const hipblas_rng_t g_hipblas_seed;

// Engine used by the random generators below. It is thread_local so that threads never share
// engine state: glibc rand() and a single shared engine serialize (or race) when an OpenMP
// parallel fill calls them from many threads.
extern thread_local hipblas_rng_t t_hipblas_rng;

// Reset the calling thread's random sequence (mainly to ensure repeatability of failures in a given
// suite)
inline void hipblas_seedrand()
{
    t_hipblas_rng = g_hipblas_seed;
}

// Seeding for one parallel fill, done through each thread's own engine (t_hipblas_rng). The fill
// draws one seed from the calling thread's random sequence when it starts, and every thread
// reseeds its t_hipblas_rng from that seed and the position of each block it writes. The values
// therefore do not depend on which thread writes a block, so a fill produces the same data for any
// number of OpenMP threads and any schedule. The calling thread also writes blocks, which reseeds
// its engine, so its random sequence is restored when the fill ends: the next fill continues from
// the same point whatever the thread count.
class hipblas_parallel_fill_seeder
{
public:
    hipblas_parallel_fill_seeder()
        : m_seed(t_hipblas_rng())
        , m_saved(t_hipblas_rng)
    {
    }

    ~hipblas_parallel_fill_seeder()
    {
        t_hipblas_rng = m_saved;
    }

    hipblas_parallel_fill_seeder(const hipblas_parallel_fill_seeder&) = delete;
    hipblas_parallel_fill_seeder& operator=(const hipblas_parallel_fill_seeder&) = delete;

    // Call in the parallel loop, before generating the values of block `block` of batch `batch`:
    // reseeds the calling thread's t_hipblas_rng
    void reseed_thread_rng(uint64_t batch, uint64_t block) const
    {
        t_hipblas_rng = hipblas_rng_t(mix(mix(m_seed ^ batch) ^ block));
    }

private:
    static uint64_t mix(uint64_t x)
    {
        return hipblas_rng_t(x)();
    }

    uint64_t      m_seed;
    hipblas_rng_t m_saved;
};

// Number of vector elements generated from one block seed
constexpr int64_t hipblas_fill_block_size = 4096;

/*! \brief  Random number generator which generates NaN values */
class hipblas_nan_rng
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
            x.u = std::uniform_int_distribution<UINT_T>{}(t_hipblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(t_hipblas_rng);
    }

    explicit operator signed char()
    {
        return static_cast<signed char>(std::uniform_int_distribution<int>{}(t_hipblas_rng));
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

    // Random NaN half (non-template hipblasHalf takes precedence over integer template above)
    explicit operator hipblasHalf()
    {
        return random_nan_data<hipblasHalf, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator hipblasBfloat16()
    {
        return random_nan_data<hipblasBfloat16, uint16_t, 7, 8>();
    }

    // Random NaN Complex
    explicit operator std::complex<float>()
    {
        return {float(*this), float(*this)};
    }

    // Random NaN Double Complex
    explicit operator std::complex<double>()
    {
        return {double(*this), double(*this)};
    }

    explicit operator hipComplex()
    {
        return {float(*this), float(*this)};
    }

    explicit operator hipDoubleComplex()
    {
        return {double(*this), double(*this)};
    }

    // // Currently not needed
    // // Random complex integers
    // explicit operator hipblasInt8Complex()
    // {
    //     return static_cast<int8_t>(
    //         std::uniform_int_distribution<unsigned short>(1, 3)(t_hipblas_rng));
    // }
};

/* ============================================================================================ */
/*! \brief negate a value */

// Can rename to simply "negate" after removing usage of `using namespace std;`
template <class T>
inline T hipblas_negate(T x)
{
    return -x;
}

template <>
inline hipblasHalf hipblas_negate(hipblasHalf arg)
{
    union
    {
        hipblasHalf fp;
        uint16_t    data;
    } x = {arg};

    x.data ^= 0x8000;
    return x.fp;
}

template <>
inline hipblasBfloat16 hipblas_negate(hipblasBfloat16 x)
{
    x.data ^= 0x8000;
    return x;
}

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
T random_generator()
{
    return T(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng));
};

/*! \brief  generate a random NaN number */
template <typename T>
inline T random_nan_generator()
{
    return T(hipblas_nan_rng{});
}

// for hipblasHalf, generate float, and convert to hipblasHalf
/*! \brief  generate a random number in range [1,2,3] */
template <>
inline hipblasHalf random_generator<hipblasHalf>()
{
    return float_to_half(
        float(std::uniform_int_distribution<int>(1, 3)(t_hipblas_rng))); // range [1,2,3]
};

// for hipblasBfloat16, generate float, and convert to hipblasBfloat16
template <>
inline hipblasBfloat16 random_generator<hipblasBfloat16>()
{
    return float_to_bfloat16(
        float(std::uniform_int_distribution<int>(1, 3)(t_hipblas_rng))); // range [1,2,3]
}

// for std::complex<float>, generate 2 floats
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline std::complex<float> random_generator<std::complex<float>>()
{
    return {float(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng)),
            float(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng))};
}

// for std::complex<double>, generate 2 doubles
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline std::complex<double> random_generator<std::complex<double>>()
{
    return {double(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng)),
            double(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng))};
}

/*! \brief  generate a random number in range [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] */
template <typename T>
inline T random_generator_negative()
{
    return -T(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng));
};

// for hipblasHalf, generate float, and convert to hipblasHalf
/*! \brief  generate a random number in range [-1,-2,-3] */
template <>
inline hipblasHalf random_generator_negative<hipblasHalf>()
{
    return float_to_half(-float(std::uniform_int_distribution<int>(1, 3)(t_hipblas_rng)));
};

// for hipblasBfloat16, generate float, and convert to hipblasBfloat16
/*! \brief  generate a random number in range [-1,-2,-3] */
template <>
inline hipblasBfloat16 random_generator_negative<hipblasBfloat16>()
{
    return float_to_bfloat16(-float(std::uniform_int_distribution<int>(1, 3)(t_hipblas_rng)));
};

// for complex, generate two values, convert both to negative
/*! \brief  generate a random real value in range [-1, -10] and random
*           imaginary value in range [-1, -10]
*/
template <>
inline std::complex<float> random_generator_negative<std::complex<float>>()
{
    return {float(-(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng))),
            float(-(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng)))};
}

template <>
inline std::complex<double> random_generator_negative<std::complex<double>>()
{
    return {double(-(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng))),
            double(-(std::uniform_int_distribution<int>(1, 10)(t_hipblas_rng)))};
}

// HPL
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(t_hipblas_rng);
}

// for hipblasBfloat16, generate float, and convert to hipblasBfloat16
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <>
inline hipblasBfloat16 random_hpl_generator()
{
    return hipblasBfloat16(
        float_to_bfloat16(std::uniform_real_distribution<float>(-0.5f, 0.5f)(t_hipblas_rng)));
}

template <>
inline hipblasHalf random_hpl_generator()
{
    return hipblasHalf(
        float_to_half(std::uniform_real_distribution<float>(-0.5f, 0.5f)(t_hipblas_rng)));
}

/* ============================================================================================= */
/*! \brief For testing purposes, prepares matrix hA for a triangular solve.                      *
 *         Makes hA strictly diagonal dominant (SPD), then calculates Cholesky factorization     *
 *         of hA.                                                                                */
/*
template <typename T>
void prepare_triangular_solve(T* hA, int64_t lda, T* AAT, int64_t N, char char_uplo)
{
    //  calculate AAT = hA * hA ^ T
    ref_gemm<T>(HIPBLAS_OP_N, HIPBLAS_OP_C, N, N, N, T(1.0), hA, lda, hA, lda, T(0.0), AAT, lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int64_t i = 0; i < N; i++)
    {
        T t = 0.0;
        for(int64_t j = 0; j < N; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += hipblas_abs(AAT[i + j * lda]);
        }
        hA[i + i * lda] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    ref_potrf<T>(char_uplo, N, hA, lda);
}
*/

/* ============================================================================================ */
/*! \brief  turn float -> 's', double -> 'd', hipblas_float_complex -> 'c', hipblas_double_complex
 * -> 'z' */
template <typename T>
char type2char();

/* ============================================================================================ */
/*! \brief  turn float -> int, double -> int, hipblas_float_complex.real() -> int,
 * hipblas_double_complex.real() -> int */
template <typename T>
int type2int(T val);

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void print_matrix(const std::vector<T>& CPU_result,
                  const std::vector<T>& GPU_result,
                  int64_t               m,
                  int64_t               n,
                  int64_t               lda)
{
    for(int64_t i = 0; i < m; i++)
        for(int64_t j = 0; j < n; j++)
            printf("matrix  col %ld, row %ld, CPU result=%.8g, GPU result=%.8g\n",
                   i,
                   j,
                   double(CPU_result[j + i * lda]),
                   double(GPU_result[j + i * lda]));
}

/*! \brief  Debugging purpose, print out CPU and GPU result matrix, valid for complex number  */
template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
void print_matrix(const std::vector<T>& CPU_result,
                  const std::vector<T>& GPU_result,
                  int64_t               m,
                  int64_t               n,
                  int64_t               lda)
{
    for(int64_t i = 0; i < m; i++)
        for(int64_t j = 0; j < n; j++)
            printf("matrix  col %ld, row %ld, CPU result=(%.8g,%.8g), GPU result=(%.8g,%.8g)\n",
                   i,
                   j,
                   double(CPU_result[j + i * lda].real()),
                   double(CPU_result[j + i * lda].imag()),
                   double(GPU_result[j + i * lda].real()),
                   double(GPU_result[j + i * lda].imag()));
}

/* ============================================================================================= */
/*! \brief For testing purposes, copy one matrix into another with different leading dimensions  */
template <typename T, typename U>
void copy_matrix_with_different_leading_dimensions(T&      hB,
                                                   U&      hC,
                                                   int64_t M,
                                                   int64_t N,
                                                   size_t  ldb,
                                                   size_t  ldc,
                                                   size_t  strideb     = 0,
                                                   size_t  stridec     = 0,
                                                   int64_t batch_count = 1)
{
    for(int64_t b = 0; b < batch_count; b++)
    {
        auto* B = hB + b * strideb;
        auto* C = hC + b * stridec;
        for(int64_t i = 0; i < M; i++)
            for(int64_t j = 0; j < N; j++)
                C[i + j * ldc] = B[i + j * ldb];
    }
}

template <typename T, typename U>
void copy_matrix_with_different_leading_dimensions(T& hB, U& hC)
{
    const int64_t M           = hB.m();
    const int64_t N           = hB.n();
    const size_t  ldb         = hB.lda();
    const size_t  ldc         = hC.lda();
    const int64_t batch_count = hB.batch_count();
    for(int64_t b = 0; b < batch_count; b++)
    {
        auto* B = hB[b];
        auto* C = hC[b];
        for(int64_t i = 0; i < M; i++)
            for(int64_t j = 0; j < N; j++)
                C[i + j * ldc] = B[i + j * ldb];
    }
}

template <typename T, typename U>
void copy_matrix_with_different_leading_dimensions_batched(
    T& hB, U& hC, int64_t M, int64_t N, size_t ldb, size_t ldc)
{
    int64_t batch_count = hB.batch_count();
    for(int64_t b = 0; b < batch_count; b++)
    {
        auto* B = hB[b];
        auto* C = hC[b];
        for(int64_t i = 0; i < M; i++)
            for(int64_t j = 0; j < N; j++)
                C[i + j * ldc] = B[i + j * ldb];
    }
}

/* =============================================================================================== */

/* ============================================================================================ */
// Return path of this executable
std::string hipblas_exepath();

/* ============================================================================================ */
// Temp directory rooted random path
std::string hipblas_tempname();

std::string getArchString();

#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  device query and print out their ID and name */
int query_device_property();

/*  set current device to device_id */
void set_device(int device_id);

typedef enum hipblasClientProcessor
{
    // matching enum used in hipGcnArch
    // only including supported types
    gfx803  = 803,
    gfx900  = 900,
    gfx906  = 906,
    gfx908  = 908,
    gfx90a  = 910,
    gfx90c  = 912,
    gfx942  = 942,
    gfx950  = 950,
    gfx1010 = 1010,
    gfx1011 = 1011,
    gfx1012 = 1012,
    gfx1030 = 1030,
    gfx1031 = 1031,
    gfx1032 = 1032,
    gfx1034 = 1034,
    gfx1035 = 1035,
    gfx1100 = 1100,
    gfx1101 = 1101,
    gfx1102 = 1102,
    gfx1103 = 1103,
    gfx1150 = 1150,
    gfx1151 = 1151,
    gfx1152 = 1152,
    gfx1153 = 1153,
    gfx1170 = 1170,
    gfx1171 = 1171,
    gfx1200 = 1200,
    gfx1201 = 1201,
    gfx1250 = 1250
} hipblasClientProcessor;

/* get architecture number */
hipblasClientProcessor getArch();
int                    getArchMajor();

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipblas sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */

#ifdef __cplusplus

struct Arguments;

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */
class hipblasLocalHandle
{
    hipblasHandle_t m_handle;
    void*           m_memory = nullptr;

public:
    hipblasLocalHandle();

    explicit hipblasLocalHandle(const Arguments& arg);

    ~hipblasLocalHandle();

    hipblasLocalHandle(const hipblasLocalHandle&) = delete;
    hipblasLocalHandle(hipblasLocalHandle&&)      = delete;
    hipblasLocalHandle& operator=(const hipblasLocalHandle&) = delete;
    hipblasLocalHandle& operator=(hipblasLocalHandle&&) = delete;

    // Allow hipblasLocalHandle to be used anywhere hipblas_handle is expected
    operator hipblasHandle_t&()
    {
        return m_handle;
    }
    operator const hipblasHandle_t&() const
    {
        return m_handle;
    }
};

hipblasStatus_t hipblas_internal_convert_hip_to_hipblas_status(hipError_t status);

hipblasStatus_t hipblas_internal_convert_hip_to_hipblas_status_and_log(hipError_t status);

#include "hipblas_arguments.hpp"

#endif // __cplusplus
