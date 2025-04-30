/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef __cplusplus
#include "complex.hpp"
#include "hipsolver_datatype2string.hpp"
#include <cassert>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>
#endif

#include <stdio.h>
#include <stdlib.h>

/*!\file
 * \brief provide data initialization, timing, hipsolver type <-> lapack char conversion utilities.
 */

#ifdef GOOGLE_TEST

#include <gtest/gtest.h>

#define CHECK_HIP_ERROR(error) ASSERT_EQ(error, hipSuccess)

inline void hipsolver_expect_status(hipsolverStatus_t status, hipsolverStatus_t expected)
{
    if(status != HIPSOLVER_STATUS_NOT_SUPPORTED)
        ASSERT_EQ(status, expected);
}

#define EXPECT_ROCBLAS_STATUS(status, expected) hipsolver_expect_status(status, expected)
#define CHECK_ROCBLAS_ERROR(status) hipsolver_expect_status(status, HIPSOLVER_STATUS_SUCCESS)

#else

#define CHECK_HIP_ERROR(error)                    \
    do                                            \
    {                                             \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

inline void hipsolver_expect_status(hipsolverStatus_t status, hipsolverStatus_t expected)
{
    if(status != expected && status != HIPSOLVER_STATUS_NOT_SUPPORTED)
    {
        fprintf(stderr,
                "hipSOLVER status error: Expected: %s, Actual: %s\n",
                hipsolver2string_status(expected),
                hipsolver2string_status(status));
        if(expected == HIPSOLVER_STATUS_SUCCESS)
            exit(EXIT_FAILURE);
    }
}

#define EXPECT_ROCBLAS_STATUS(status, expected) hipsolver_expect_status(status, expected)
#define CHECK_ROCBLAS_ERROR(status) hipsolver_expect_status(status, HIPSOLVER_STATUS_SUCCESS)

// The info provided to EXPECT macros is used in hipsolver-test, but
// in hipsolver-bench, the information is just discarded.
struct hipsolver_info_discarder
{
    template <typename T>
    hipsolver_info_discarder& operator<<(T&&)
    {
        return *this;
    }
};

#define EXPECT_EQ(v1, v2) hipsolver_info_discarder()
#define EXPECT_NE(v1, v2) hipsolver_info_discarder()
#define EXPECT_LT(v1, v2) hipsolver_info_discarder()
#define EXPECT_LE(v1, v2) hipsolver_info_discarder()
#define EXPECT_GT(v1, v2) hipsolver_info_discarder()
#define EXPECT_GE(v1, v2) hipsolver_info_discarder()

#endif

#ifdef __cplusplus

/* ============================================================================================
 */
/*! \brief  local handle which is automatically created and destroyed  */
class hipsolver_local_handle
{
    hipsolverHandle_t m_handle;

public:
    hipsolver_local_handle()
    {
        if(hipsolverCreate(&m_handle) != HIPSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("ERROR: Could not create hipsolverHandle_t");
    }
    ~hipsolver_local_handle()
    {
        hipsolverDestroy(m_handle);
    }

    hipsolver_local_handle(const hipsolver_local_handle&) = delete;

    hipsolver_local_handle(hipsolver_local_handle&&) = delete;

    hipsolver_local_handle& operator=(const hipsolver_local_handle&) = delete;

    hipsolver_local_handle& operator=(hipsolver_local_handle&&) = delete;

    // Allow hipsolver_local_handle to be used anywhere hipsolverHandle_t is expected
    operator hipsolverHandle_t&()
    {
        return m_handle;
    }
    operator const hipsolverHandle_t&() const
    {
        return m_handle;
    }
};

class hipsolverRf_local_handle
{
    hipsolverRfHandle_t m_handle;

public:
    hipsolverRf_local_handle()
    {
        if(hipsolverRfCreate(&m_handle) != HIPSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("ERROR: Could not create hipsolverRfHandle_t");
    }
    ~hipsolverRf_local_handle()
    {
        hipsolverRfDestroy(m_handle);
    }

    hipsolverRf_local_handle(const hipsolverRf_local_handle&) = delete;

    hipsolverRf_local_handle(hipsolverRf_local_handle&&) = delete;

    hipsolverRf_local_handle& operator=(const hipsolverRf_local_handle&) = delete;

    hipsolverRf_local_handle& operator=(hipsolverRf_local_handle&&) = delete;

    // Allow hipsolverRf_local_handle to be used anywhere hipsolverRfHandle_t is expected
    operator hipsolverRfHandle_t&()
    {
        return m_handle;
    }
    operator const hipsolverRfHandle_t&() const
    {
        return m_handle;
    }
};

class hipsolverSp_local_handle
{
    hipsolverSpHandle_t m_handle;

public:
    hipsolverSp_local_handle()
    {
        if(hipsolverSpCreate(&m_handle) != HIPSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("ERROR: Could not create hipsolverSpHandle_t");
    }
    ~hipsolverSp_local_handle()
    {
        hipsolverSpDestroy(m_handle);
    }

    hipsolverSp_local_handle(const hipsolverSp_local_handle&) = delete;

    hipsolverSp_local_handle(hipsolverSp_local_handle&&) = delete;

    hipsolverSp_local_handle& operator=(const hipsolverSp_local_handle&) = delete;

    hipsolverSp_local_handle& operator=(hipsolverSp_local_handle&&) = delete;

    // Allow hipsolverSp_local_handle to be used anywhere hipsolverSpHandle_t is expected
    operator hipsolverSpHandle_t&()
    {
        return m_handle;
    }
    operator const hipsolverSpHandle_t&() const
    {
        return m_handle;
    }
};

/* ============================================================================================
 */
/*! \brief  local gesvdj params which is automatically created and destroyed  */
class hipsolver_local_gesvdj_info
{
    hipsolverGesvdjInfo_t m_info;

public:
    hipsolver_local_gesvdj_info()
    {
        if(hipsolverDnCreateGesvdjInfo(&m_info) != HIPSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("ERROR: Could not create hipsolverGesvdjInfo_t");
    }
    ~hipsolver_local_gesvdj_info()
    {
        hipsolverDnDestroyGesvdjInfo(m_info);
    }

    hipsolver_local_gesvdj_info(const hipsolver_local_gesvdj_info&) = delete;

    hipsolver_local_gesvdj_info(hipsolver_local_gesvdj_info&&) = delete;

    hipsolver_local_gesvdj_info& operator=(const hipsolver_local_gesvdj_info&) = delete;

    hipsolver_local_gesvdj_info& operator=(hipsolver_local_gesvdj_info&&) = delete;

    // Allow hipsolver_local_gesvdj_info to be used anywhere hipsolverGesvdjInfo_t is expected
    operator hipsolverGesvdjInfo_t&()
    {
        return m_info;
    }
    operator const hipsolverGesvdjInfo_t&() const
    {
        return m_info;
    }
};

/* ============================================================================================
 */
/*! \brief  local syevj params which is automatically created and destroyed  */
class hipsolver_local_syevj_info
{
    hipsolverSyevjInfo_t m_info;

public:
    hipsolver_local_syevj_info()
    {
        if(hipsolverDnCreateSyevjInfo(&m_info) != HIPSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("ERROR: Could not create hipsolverSyevjInfo_t");
    }
    ~hipsolver_local_syevj_info()
    {
        hipsolverDnDestroySyevjInfo(m_info);
    }

    hipsolver_local_syevj_info(const hipsolver_local_syevj_info&) = delete;

    hipsolver_local_syevj_info(hipsolver_local_syevj_info&&) = delete;

    hipsolver_local_syevj_info& operator=(const hipsolver_local_syevj_info&) = delete;

    hipsolver_local_syevj_info& operator=(hipsolver_local_syevj_info&&) = delete;

    // Allow hipsolver_local_syevj_info to be used anywhere hipsolverSyevjInfo_t is expected
    operator hipsolverSyevjInfo_t&()
    {
        return m_info;
    }
    operator const hipsolverSyevjInfo_t&() const
    {
        return m_info;
    }
};

/* ============================================================================================
 */
/*! \brief  local params which is automatically created and destroyed  */
class hipsolver_local_params
{
    hipsolverDnParams_t m_info;

public:
    hipsolver_local_params()
    {
        if(hipsolverDnCreateParams(&m_info) != HIPSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("ERROR: Could not create hipsolverDnParams_t");
    }
    ~hipsolver_local_params()
    {
        hipsolverDnDestroyParams(m_info);
    }

    hipsolver_local_params(const hipsolver_local_params&) = delete;

    hipsolver_local_params(hipsolver_local_params&&) = delete;

    hipsolver_local_params& operator=(const hipsolver_local_params&) = delete;

    hipsolver_local_params& operator=(hipsolver_local_params&&) = delete;

    // Allow hipsolver_local_params to be used anywhere hipsolverDnParams_t is expected
    operator hipsolverDnParams_t&()
    {
        return m_info;
    }
    operator const hipsolverDnParams_t&() const
    {
        return m_info;
    }
};

/* ============================================================================================
 */

// Return true if value is NaN
template <typename T>
inline bool hipsolver_isnan(T)
{
    return false;
}
inline bool hipsolver_isnan(double arg)
{
    return std::isnan(arg);
}
inline bool hipsolver_isnan(float arg)
{
    return std::isnan(arg);
}
inline bool hipsolver_isnan(hipsolverComplex arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}
inline bool hipsolver_isnan(hipsolverDoubleComplex arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}

/* =============================================================================================== */
/* Complex / real helpers.                                                                         */

/* Workaround for clang bug:
   https://bugs.llvm.org/show_bug.cgi?id=35863
*/
#if __clang__
#define HIPSOLVER_CLANG_STATIC static
#else
#define HIPSOLVER_CLANG_STATIC
#endif

template <typename T>
static constexpr bool is_complex = false;

// cppcheck-suppress syntaxError
template <>
HIPSOLVER_CLANG_STATIC constexpr bool is_complex<hipsolverComplex> = true;

template <>
HIPSOLVER_CLANG_STATIC constexpr bool is_complex<hipsolverDoubleComplex> = true;

// Get base types from complex types.
template <typename T, typename = void>
struct real_t_impl
{
    using type = T;
};

template <typename T>
struct real_t_impl<T, std::enable_if_t<is_complex<T>>>
{
    using type = decltype(T{}.real());
};

template <typename T>
using real_t = typename real_t_impl<T>::type;

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */

using hipsolver_rng_t = std::mt19937;
extern hipsolver_rng_t hipsolver_rng, hipsolver_seed;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void hipsolver_seedrand()
{
    hipsolver_rng = hipsolver_seed;
}

class hipsolver_nan_rng
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
            x.u = std::uniform_int_distribution<UINT_T>{}(hipsolver_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T() const
    {
        return std::uniform_int_distribution<T>{}(hipsolver_rng);
    }

    // Random NaN float
    explicit operator float() const
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN double
    explicit operator double() const
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN Complex
    explicit operator hipsolverComplex() const
    {
        return {float(*this), float(*this)};
    }

    // Random NaN Double Complex
    explicit operator hipsolverDoubleComplex() const
    {
        return {double(*this), double(*this)};
    }
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
T random_generator()
{
    return std::uniform_int_distribution<int>(1, 10)(hipsolver_rng);
};

// for hipsolverComplex, generate 2 floats
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline hipsolverComplex random_generator<hipsolverComplex>()
{
    return hipsolverComplex(float(std::uniform_int_distribution<int>(1, 10)(hipsolver_rng)),
                            float(std::uniform_int_distribution<int>(1, 10)(hipsolver_rng)));
}

// for hipsolverDoubleComplex, generate 2 doubles
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline hipsolverDoubleComplex random_generator<hipsolverDoubleComplex>()
{
    return hipsolverDoubleComplex(double(std::uniform_int_distribution<int>(1, 10)(hipsolver_rng)),
                                  double(std::uniform_int_distribution<int>(1, 10)(hipsolver_rng)));
}

/*! \brief  generate a random number in range [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] */
template <typename T>
inline T random_generator_negative()
{
    return std::uniform_int_distribution<int>(-10, -1)(hipsolver_rng);
};

// for complex, generate two values, convert both to negative
/*! \brief  generate a random real value in range [-1, -10] and random
*           imaginary value in range [-1, -10]
*/
template <>
inline hipsolverComplex random_generator_negative<hipsolverComplex>()
{
    return hipsolverComplex(float(std::uniform_int_distribution<int>(-10, -1)(hipsolver_rng)),
                            float(std::uniform_int_distribution<int>(-10, -1)(hipsolver_rng)));
}

template <>
inline hipsolverDoubleComplex random_generator_negative<hipsolverDoubleComplex>()
{
    return hipsolverDoubleComplex(
        double(std::uniform_int_distribution<int>(-10, -1)(hipsolver_rng)),
        double(std::uniform_int_distribution<int>(-10, -1)(hipsolver_rng)));
}

/* ============================================================================================ */

/* ============================================================================================ */
/*! \brief Packs strided_batched matricies into groups of 4 in N */
template <typename T>
void hipsolver_packInt8(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t batch_count = 1, size_t stride_a = 0)
{
    std::vector<T> temp(A);
    for(size_t b = 0; b < batch_count; b++)
        for(size_t colBase = 0; colBase < N; colBase += 4)
            for(size_t row = 0; row < lda; row++)
                for(size_t colOffset = 0; colOffset < 4; colOffset++)
                    A[(colBase * lda + 4 * row) + colOffset + (stride_a * b)]
                        = temp[(colBase + colOffset) * lda + row + (stride_a * b)];
}

/* ============================================================================================ */

/* ============================================================================================ */
/*! \brief  turn float -> 's', double -> 'd', hipsolverComplex -> 'c', hipsolverDoubleComplex
 * -> 'z' */
template <typename T>
char type2char();

/* ============================================================================================ */
/*! \brief  turn float -> int, double -> int, hipsolverComplex.real() -> int,
 * hipsolverDoubleComplex.real() -> int */
template <typename T>
int type2int(T val);

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void print_matrix(T* result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            printf("matrix  col %d, row %d, result=%.8g\n", j, i, double(result[i + j * lda]));
}

/*! \brief  Debugging purpose, print out CPU and GPU result matrix, valid for complex number  */
template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
void print_matrix(T* result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            printf("matrix  col %d, row %d, result=(%.8g,%.8g)\n",
                   j,
                   i,
                   double(result[i + j * lda].real()),
                   double(result[i + j * lda].imag()));
}

template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void print_matrix(T* CPU_result, T* GPU_result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            printf("matrix  col %d, row %d, CPU result=%.8g, GPU result=%.8g\n",
                   j,
                   i,
                   double(CPU_result[i + j * lda]),
                   double(GPU_result[i + j * lda]));
}

/*! \brief  Debugging purpose, print out CPU and GPU result matrix, valid for complex number  */
template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
void print_matrix(T* CPU_result, T* GPU_result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            printf("matrix  col %d, row %d, CPU result=(%.8g,%.8g), GPU result=(%.8g,%.8g)\n",
                   j,
                   i,
                   double(CPU_result[i + j * lda].real()),
                   double(CPU_result[i + j * lda].imag()),
                   double(GPU_result[i + j * lda].real()),
                   double(GPU_result[i + j * lda].imag()));
}

/*! \brief  Debugging purpose, print out sparse matrix, not valid in complex number  */
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void print_sparse_matrix(int m, int n, int* csrRowPtr, int* csrColInd, T* csrVal)
{
    printf("%d by %d matrix\n", m, n);

    int idx = 0;
    for(int i = 0; i < m; i++)
    {
        printf("    ");
        for(int j = 0; j < n; j++)
        {
            if(idx < csrRowPtr[i + 1] && j == csrColInd[idx])
            {
                printf("%.8g, ", csrVal[idx]);
                idx++;
            }
            else
                printf("0, ");
        }
        printf("\n");
    }

    printf("\n");
}

/* ============================================================================================ */
/*  read matrix or values from file */

// integers:
inline void
    read_matrix(const std::string filenameS, const int m, const int n, int* A, const int lda)
{
    const char* filename = filenameS.c_str();
    FILE*       mat;
    mat = fopen(filename, "r");
    int v;

    if(mat == NULL)
        throw std::invalid_argument(std::string("Error: Could not open file ") + filename
                                    + " with test data...");

    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            int read = fscanf(mat, "%d", &v);
            if(read != 1)
                throw std::out_of_range(std::string("Error: Could not read element from file ")
                                        + filename);
            A[i + j * lda] = v;
        }
    }

    fclose(mat);
}
inline void read_last(const std::string filenameS, int* A)
{
    const char* filename = filenameS.c_str();
    FILE*       mat;
    mat = fopen(filename, "r");
    int v;

    if(mat == NULL)
        throw std::invalid_argument(std::string("Error: Could not open file ") + filename
                                    + " with test data...");

    while(fscanf(mat, "%d", &v) == 1)
    {
        // do nothing
    }

    *A = v;
}

// singles:
inline void
    read_matrix(const std::string filenameS, const int m, const int n, float* A, const int lda)
{
    const char* filename = filenameS.c_str();
    FILE*       mat;
    mat = fopen(filename, "r");
    float v;

    if(mat == NULL)
        throw std::invalid_argument(std::string("Error: Could not open file ") + filename
                                    + " with test data...");

    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            int read = fscanf(mat, "%g", &v);
            if(read != 1)
                throw std::out_of_range(std::string("Error: Could not read element from file ")
                                        + filename);
            A[i + j * lda] = v;
        }
    }

    fclose(mat);
}

// doubles:
inline void
    read_matrix(const std::string filenameS, const int m, const int n, double* A, const int lda)
{
    const char* filename = filenameS.c_str();
    FILE*       mat;
    mat = fopen(filename, "r");
    double v;

    if(mat == NULL)
        throw std::invalid_argument(std::string("Error: Could not open file ") + filename
                                    + " with test data...");

    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            int read = fscanf(mat, "%lg", &v);
            if(read != 1)
                throw std::out_of_range(std::string("Error: Could not read element from file ")
                                        + filename);
            A[i + j * lda] = v;
        }
    }

    fclose(mat);
}

// complex float:
inline void read_matrix(
    const std::string filenameS, const int m, const int n, hipsolverComplex* A, const int lda)
{
    const char* filename = filenameS.c_str();
    FILE*       mat;
    mat = fopen(filename, "r");
    float v;

    if(mat == NULL)
        throw std::invalid_argument(std::string("Error: Could not open file ") + filename
                                    + " with test data...");

    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            int read = fscanf(mat, "%g", &v);
            if(read != 1)
                throw std::out_of_range(std::string("Error: Could not read element from file ")
                                        + filename);
            A[i + j * lda] = {v, 0};
        }
    }

    fclose(mat);
}

// complex double:
inline void read_matrix(
    const std::string filenameS, const int m, const int n, hipsolverDoubleComplex* A, const int lda)
{
    const char* filename = filenameS.c_str();
    FILE*       mat;
    mat = fopen(filename, "r");
    double v;

    if(mat == NULL)
        throw std::invalid_argument(std::string("Error: Could not open file ") + filename
                                    + " with test data...");

    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            int read = fscanf(mat, "%lg", &v);
            if(read != 1)
                throw std::out_of_range(std::string("Error: Could not read element from file ")
                                        + filename);
            A[i + j * lda] = {v, 0};
        }
    }

    fclose(mat);
}

/* =============================================================================================== */

/* ============================================================================================ */

#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipsolver sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us();

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

/*! \brief  CPU Timer(in microsecond): no GPU synchronization and return wall time */
double get_time_us_no_sync();

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */
