/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include <algorithm>
#include <assert.h>
#include <complex>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <iostream>

#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
/*! \brief Return path of this executable */
std::string hipsparse_exepath();
/*! \brief Return path where the test data file (hipsparse_test.data) is located */
std::string hipsparse_datapath();

/*!\file
 * \brief provide data initialization and timing utilities.
 */

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) \
    ((dir == HIPSPARSE_DIRECTION_ROW) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))
#define BSR_IND_R(j, bi, bj) (bsr_dim * bsr_dim * (j) + (bi)*bsr_dim + (bj))
#define BSR_IND_C(j, bi, bj) (bsr_dim * bsr_dim * (j) + (bi) + (bj)*bsr_dim)

#define CHECK_HIP_ERROR(error)                \
    if(error != hipSuccess)                   \
    {                                         \
        fprintf(stderr,                       \
                "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),     \
                error,                        \
                __FILE__,                     \
                __LINE__);                    \
        exit(EXIT_FAILURE);                   \
    }

#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11003))

#define CHECK_HIPSPARSE_ERROR_CASE__(token_) \
    case token_:                             \
        fprintf(stderr, #token_);            \
        break

#define CHECK_HIPSPARSE_ERROR(error)                                                      \
    {                                                                                     \
        auto local_error = (error);                                                       \
        if(local_error != HIPSPARSE_STATUS_SUCCESS)                                       \
        {                                                                                 \
            fprintf(stderr, "hipSPARSE error: ");                                         \
            switch(local_error)                                                           \
            {                                                                             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_SUCCESS);                   \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_INITIALIZED);           \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ALLOC_FAILED);              \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INVALID_VALUE);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ARCH_MISMATCH);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MAPPING_ERROR);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_EXECUTION_FAILED);          \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INTERNAL_ERROR);            \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED); \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ZERO_PIVOT);                \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_SUPPORTED);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);    \
            }                                                                             \
            fprintf(stderr, "\n");                                                        \
            return local_error;                                                           \
        }                                                                                 \
    }                                                                                     \
    (void)0

#else

#define CHECK_HIPSPARSE_ERROR_CASE__(token_) \
    case token_:                             \
        fprintf(stderr, #token_);            \
        break

#define CHECK_HIPSPARSE_ERROR(error)                                                      \
    {                                                                                     \
        auto local_error = (error);                                                       \
        if(local_error != HIPSPARSE_STATUS_SUCCESS)                                       \
        {                                                                                 \
            fprintf(stderr, "hipSPARSE error: ");                                         \
            switch(local_error)                                                           \
            {                                                                             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_SUCCESS);                   \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_INITIALIZED);           \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ALLOC_FAILED);              \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INVALID_VALUE);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ARCH_MISMATCH);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MAPPING_ERROR);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_EXECUTION_FAILED);          \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INTERNAL_ERROR);            \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED); \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ZERO_PIVOT);                \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_SUPPORTED);             \
            }                                                                             \
            fprintf(stderr, "\n");                                                        \
            return local_error;                                                           \
        }                                                                                 \
    }                                                                                     \
    (void)0

#endif

#ifdef __HIP_PLATFORM_NVIDIA__
static inline hipComplex operator-(const hipComplex& op)
{
    hipComplex ret;
    ret.x = -op.x;
    ret.y = -op.y;
    return ret;
}
static inline hipDoubleComplex operator-(const hipDoubleComplex& op)
{
    hipDoubleComplex ret;
    ret.x = -op.x;
    ret.y = -op.y;
    return ret;
}

static inline bool operator==(const hipComplex& lhs, const hipComplex& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}
static inline bool operator==(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

static inline bool operator!=(const hipComplex& lhs, const hipComplex& rhs)
{
    return !(lhs == rhs);
}
static inline bool operator!=(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    return !(lhs == rhs);
}

static inline hipComplex operator+(const hipComplex& lhs, const hipComplex& rhs)
{
    hipComplex ret;
    ret.x = lhs.x + rhs.x;
    ret.y = lhs.y + rhs.y;
    return ret;
}
static inline hipDoubleComplex operator+(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    hipDoubleComplex ret;
    ret.x = lhs.x + rhs.x;
    ret.y = lhs.y + rhs.y;
    return ret;
}

static inline hipComplex operator-(const hipComplex& lhs, const hipComplex& rhs)
{
    hipComplex ret;
    ret.x = lhs.x - rhs.x;
    ret.y = lhs.y - rhs.y;
    return ret;
}
static inline hipDoubleComplex operator-(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    hipDoubleComplex ret;
    ret.x = lhs.x - rhs.x;
    ret.y = lhs.y - rhs.y;
    return ret;
}

static inline hipComplex operator*(const hipComplex& lhs, const hipComplex& rhs)
{
    hipComplex ret;
    ret.x = lhs.x * rhs.x - lhs.y * rhs.y;
    ret.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return ret;
}
static inline hipDoubleComplex operator*(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    hipDoubleComplex ret;
    ret.x = lhs.x * rhs.x - lhs.y * rhs.y;
    ret.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return ret;
}

static inline hipComplex operator/(const hipComplex& lhs, const hipComplex& rhs)
{
    hipComplex ret;
    ret.x = (lhs.x * rhs.x + lhs.y * rhs.y);
    ret.y = (rhs.x * lhs.y - lhs.x * rhs.y);
    ret.x = ret.x / (rhs.x * rhs.x + rhs.y * rhs.y);
    ret.y = ret.y / (rhs.x * rhs.x + rhs.y * rhs.y);
    return ret;
}
static inline hipDoubleComplex operator/(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    hipDoubleComplex ret;
    ret.x = (lhs.x * rhs.x + lhs.y * rhs.y);
    ret.y = (rhs.x * lhs.y - lhs.x * rhs.y);
    ret.x = ret.x / (rhs.x * rhs.x + rhs.y * rhs.y);
    ret.y = ret.y / (rhs.x * rhs.x + rhs.y * rhs.y);
    return ret;
}

static inline hipComplex operator+=(hipComplex& lhs, const hipComplex& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
static inline hipDoubleComplex operator+=(hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
#endif

/* ============================================================================================ */
/*! \brief Make data type */
template <typename T>
inline T make_DataType2(double real, double imag)
{
    return static_cast<T>(real);
}

template <>
inline hipComplex make_DataType2(double real, double imag)
{
    return make_hipFloatComplex(static_cast<float>(real), static_cast<float>(imag));
}

template <>
inline hipDoubleComplex make_DataType2(double real, double imag)
{
    return make_hipDoubleComplex(real, imag);
}

template <typename T>
inline T make_DataType(double real, double imag = 0.0)
{
    return make_DataType2<T>(real, imag);
}

/* ============================================================================================ */
/*! \brief mult */
template <typename T>
inline T testing_mult(T p, T q)
{
    return p * q;
}

template <>
inline hipComplex testing_mult(hipComplex p, hipComplex q)
{
    return hipCmulf(p, q);
}

template <>
inline hipDoubleComplex testing_mult(hipDoubleComplex p, hipDoubleComplex q)
{
    return hipCmul(p, q);
}

/* ============================================================================================ */
/*! \brief div */
template <typename T>
inline T testing_div(T p, T q)
{
    return p / q;
}

template <>
inline hipComplex testing_div(hipComplex p, hipComplex q)
{
    return hipCdivf(p, q);
}

template <>
inline hipDoubleComplex testing_div(hipDoubleComplex p, hipDoubleComplex q)
{
    return hipCdiv(p, q);
}

/* ============================================================================================ */
/*! \brief fma */
template <typename T>
inline T testing_fma(T p, T q, T r)
{
    return std::fma(p, q, r);
}

template <>
inline hipComplex testing_fma(hipComplex p, hipComplex q, hipComplex r)
{
    return hipCfmaf(p, q, r);
}

template <>
inline hipDoubleComplex testing_fma(hipDoubleComplex p, hipDoubleComplex q, hipDoubleComplex r)
{
    return hipCfma(p, q, r);
}

/* ============================================================================================ */
/*! \brief abs */
static inline float testing_abs(float x)
{
    return std::abs(x);
}

static inline double testing_abs(double x)
{
    return std::abs(x);
}

static inline float testing_abs(hipComplex x)
{
    return hipCabsf(x);
}

static inline double testing_abs(hipDoubleComplex x)
{
    return hipCabs(x);
}

/* ============================================================================================ */
/*! \brief conj */
static inline float testing_conj(float x)
{
    return x;
}

static inline double testing_conj(double x)
{
    return x;
}

static inline hipComplex testing_conj(hipComplex x)
{
    return make_DataType<hipComplex>(x.x, -x.y);
}

static inline hipDoubleComplex testing_conj(hipDoubleComplex x)
{
    return make_DataType<hipDoubleComplex>(x.x, -x.y);
}

template <typename T>
inline T testing_conj(T val, bool conj)
{
    return conj ? testing_conj(val) : val;
}

/* ============================================================================================ */
/*! \brief real */
static inline float testing_real(float x)
{
    return std::real(x);
}

static inline double testing_real(double x)
{
    return std::real(x);
}

static inline float testing_real(hipComplex x)
{
    return hipCrealf(x);
}

static inline double testing_real(hipDoubleComplex x)
{
    return hipCreal(x);
}

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number between [0, 0.999...] . */
template <typename T>
inline T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return make_DataType<T>(rand() % 10 + 1,
                            rand() % 10 + 1); // generate a integer number between [1, 10]
};

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX);
// for complex number, the real/imag part would be initialized with the same value
template <typename T>
void hipsparseInit(std::vector<T>& A, int M, int N)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            A[i + j] = random_generator<T>();
        }
    }
};

/* ============================================================================================ */
/*! \brief  vector initialization: */
// initialize sparse index vector with nnz entries ranging from start to end
template <typename I>
void hipsparseInitIndex(I* x, int nnz, int start, int end)
{
    std::vector<bool> check(end - start, false);
    int               num = 0;
    while(num < nnz)
    {
        int val = start + rand() % (end - start);
        if(!check[val - start])
        {
            x[num]             = val;
            check[val - start] = true;
            ++num;
        }
    }
    std::sort(x, x + nnz);
};

/* ============================================================================================ */
/*! \brief  csr matrix initialization */
template <typename T>
void hipsparseInitCSR(
    std::vector<int>& ptr, std::vector<int>& col, std::vector<T>& val, int nrow, int ncol, int nnz)
{
    // Row offsets
    ptr[0]    = 0;
    ptr[nrow] = nnz;

    for(int i = 1; i < nrow; ++i)
    {
        ptr[i] = rand() % (nnz - 1) + 1;
    }
    std::sort(ptr.begin(), ptr.end());

    // Column indices
    for(int i = 0; i < nrow; ++i)
    {
        hipsparseInitIndex(&col[ptr[i]], ptr[i + 1] - ptr[i], 0, ncol - 1);
        std::sort(&col[ptr[i]], &col[ptr[i + 1]]);
    }

    // Random values
    for(int i = 0; i < nnz; ++i)
    {
        val[i] = random_generator<T>();
    }
}

/* ============================================================================================ */
/*! \brief  Generate 2D laplacian on unit square in CSR format */
template <typename I, typename J, typename T>
J gen_2d_laplacian(int                  ndim,
                   std::vector<I>&      rowptr,
                   std::vector<J>&      col,
                   std::vector<T>&      val,
                   hipsparseIndexBase_t idx_base)
{
    if(ndim == 0)
    {
        return 0;
    }

    J n       = ndim * ndim;
    I nnz_mat = n * 5 - ndim * 4;

    rowptr.resize(n + 1);
    col.resize(nnz_mat);
    val.resize(nnz_mat);

    I nnz = 0;

    // Fill local arrays
    for(int i = 0; i < ndim; ++i)
    {
        for(int j = 0; j < ndim; ++j)
        {
            J idx       = i * ndim + j;
            rowptr[idx] = nnz + idx_base;
            // if no upper boundary element, connect with upper neighbor
            if(i != 0)
            {
                col[nnz] = idx - ndim + idx_base;
                val[nnz] = make_DataType<T>(-1.0);
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if(j != 0)
            {
                col[nnz] = idx - 1 + idx_base;
                val[nnz] = make_DataType<T>(-1.0);
                ++nnz;
            }
            // element itself
            col[nnz] = idx + idx_base;
            val[nnz] = make_DataType<T>(4.0);
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if(j != ndim - 1)
            {
                col[nnz] = idx + 1 + idx_base;
                val[nnz] = make_DataType<T>(-1.0);
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if(i != ndim - 1)
            {
                col[nnz] = idx + ndim + idx_base;
                val[nnz] = make_DataType<T>(-1.0);
                ++nnz;
            }
        }
    }
    rowptr[n] = nnz + idx_base;

    return n;
}

/* ============================================================================================ */
/*! \brief  Generate a random sparsity pattern with a dense format, generated floating point values of type T are positive and normalized. */
template <typename T>
void gen_dense_random_sparsity_pattern(
    int m, int n, T* A, int lda, hipsparseOrder_t order, float sparsity_ratio = 0.3)
{
    if(order == HIPSPARSE_ORDER_COL)
    {
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < m; ++i)
            {
                const float d  = ((float)rand()) / ((float)RAND_MAX);
                A[j * lda + i] = (d < sparsity_ratio) ? testing_div(make_DataType<T>(rand()),
                                                                    make_DataType<T>(RAND_MAX))
                                                      : make_DataType<T>(0);
            }
        }
    }
    else
    {
        for(int j = 0; j < m; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                const float d  = ((float)rand()) / ((float)RAND_MAX);
                A[j * lda + i] = (d < sparsity_ratio) ? testing_div(make_DataType<T>(rand()),
                                                                    make_DataType<T>(RAND_MAX))
                                                      : make_DataType<T>(0);
            }
        }
    }
}

/* ============================================================================================ */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename I, typename T>
void gen_matrix_coo(I                    m,
                    I                    n,
                    I                    nnz,
                    std::vector<I>&      row_ind,
                    std::vector<I>&      col_ind,
                    std::vector<T>&      val,
                    hipsparseIndexBase_t idx_base)
{
    if((I)row_ind.size() != nnz)
    {
        row_ind.resize(nnz);
    }
    if((I)col_ind.size() != nnz)
    {
        col_ind.resize(nnz);
    }
    if((I)val.size() != nnz)
    {
        val.resize(nnz);
    }

    // Uniform distributed row indices
    for(I i = 0; i < nnz; ++i)
    {
        row_ind[i] = rand() % m;
    }

    // Sort row indices
    std::sort(row_ind.begin(), row_ind.end());

    // Sample column indices
    std::vector<bool> check(nnz, false);

    {
        I i = 0;
        while(i < nnz)
        {
            I begin = i;
            while(row_ind[i] == row_ind[begin])
            {
                ++i;
                if(i >= nnz)
                {
                    break;
                }
            }

            // Sample i disjunct column indices
            I idx = begin;
            while(idx < i)
            {
#define MM_PI 3.1415
                // Normal distribution around the diagonal
                I rng = (i - begin) * sqrt(-2.0 * log((double)rand() / RAND_MAX))
                        * cos(2.0 * MM_PI * (double)rand() / RAND_MAX);

                if(m <= n)
                {
                    rng += row_ind[begin];
                }

                // Repeat if running out of bounds
                if(rng < 0 || rng > n - 1)
                {
                    continue;
                }

                // Check for disjunct column index in current row
                if(!check[rng])
                {
                    check[rng]   = true;
                    col_ind[idx] = rng;
                    ++idx;
                }
            }

            // Reset disjunct check array
            for(I j = begin; j < i; ++j)
            {
                check[col_ind[j]] = false;
            }

            // Partially sort column indices
            std::sort(&col_ind[begin], &col_ind[i]);
        }
    }

    // Correct index base accordingly
    if(idx_base == HIPSPARSE_INDEX_BASE_ONE)
    {
        for(I i = 0; i < nnz; ++i)
        {
            ++row_ind[i];
            ++col_ind[i];
        }
    }

    // Sample random values
    for(I i = 0; i < nnz; ++i)
    {
        val[i] = random_generator<T>(); //(double) rand() / RAND_MAX;
    }
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <typename I>
static inline void read_mtx_value(std::istringstream& is, I& row, I& col, int8_t& val)
{
    is >> row >> col >> val;
}

template <typename I>
static void read_mtx_value(std::istringstream& is, I& row, I& col, float& val)
{
    is >> row >> col >> val;
}

template <typename I>
static void read_mtx_value(std::istringstream& is, I& row, I& col, double& val)
{
    is >> row >> col >> val;
}

template <typename I>
static void read_mtx_value(std::istringstream& is, I& row, I& col, hipComplex& val)
{
    float real;
    float imag;

    is >> row >> col >> real >> imag;

    val = make_DataType<hipComplex>(real, imag);
}

template <typename I>
static void read_mtx_value(std::istringstream& is, I& row, I& col, hipDoubleComplex& val)
{
    double real;
    double imag;

    is >> row >> col >> real >> imag;

    val = make_DataType<hipDoubleComplex>(real, imag);
}

template <typename I>
static void sort(std::vector<I>& perm, std::vector<I>& unsorted_row, std::vector<I>& unsorted_col)
{
    std::sort(perm.begin(), perm.end(), [&](const I& a, const I& b) {
        if(unsorted_row[a] < unsorted_row[b])
        {
            return true;
        }
        else if(unsorted_row[a] == unsorted_row[b])
        {
            return (unsorted_col[a] < unsorted_col[b]);
        }
        else
        {
            return false;
        }
    });
}

template <typename I>
inline void scan(const char* line, I* nrow, I* ncol, int64_t* nnz)
{
    sscanf(line, "%d %d %ld", nrow, ncol, nnz);
}

template <>
inline void scan<int64_t>(const char* line, int64_t* nrow, int64_t* ncol, int64_t* nnz)
{
    sscanf(line, "%ld %ld %ld", nrow, ncol, nnz);
}

template <typename I, typename T>
int read_mtx_matrix(const char*          filename,
                    I&                   nrow,
                    I&                   ncol,
                    int64_t&             nnz,
                    std::vector<I>&      row,
                    std::vector<I>&      col,
                    std::vector<T>&      val,
                    hipsparseIndexBase_t idx_base)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("Reading matrix %s...", filename);
        fflush(stdout);
    }

    FILE* f = fopen(filename, "r");
    if(!f)
    {
        fprintf(stderr,
                "Failed to open matrix file %s because it does not exist. Please download the "
                "matrix file using the install script with -c flag.",
                filename);
        return -1;
    }

    char line[1024];

    // Check for banner
    if(!fgets(line, 1024, f))
    {
        return -1;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if(sscanf(line, "%15s %15s %15s %15s %15s", banner, array, coord, data, type) != 5)
    {
        return -1;
    }

    // Convert to lower case
    for(char* p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        return -1;
    }

    // Check array type
    if(strcmp(array, "matrix") != 0)
    {
        return -1;
    }

    // Check coord
    if(strcmp(coord, "coordinate") != 0)
    {
        return -1;
    }

    // Check data
    if(strcmp(data, "real") != 0 && strcmp(data, "integer") != 0 && strcmp(data, "pattern") != 0)
    {
        return -1;
    }

    // Check type
    if(strcmp(type, "general") != 0 && strcmp(type, "symmetric") != 0)
    {
        return -1;
    }

    // Symmetric flag
    int symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if(line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    int64_t snnz;

    scan<I>(line, &nrow, &ncol, &snnz);
    nnz = symm ? (snnz - nrow) * 2 + nrow : snnz;

    std::vector<I> unsorted_row(nnz);
    std::vector<I> unsorted_col(nnz);
    std::vector<T> unsorted_val(nnz);

    // Read entries
    int64_t idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= nnz)
        {
            return 1;
        }

        I irow;
        I icol;
        T ival;

        std::istringstream ss(line);

        if(!strcmp(data, "pattern"))
        {
            ss >> irow >> icol;
            ival = make_DataType<T>(1.0);
        }
        else
        {
            read_mtx_value(ss, irow, icol, ival);
        }

        if(idx_base == HIPSPARSE_INDEX_BASE_ZERO)
        {
            --irow;
            --icol;
        }

        unsorted_row[idx] = irow;
        unsorted_col[idx] = icol;
        unsorted_val[idx] = ival;

        ++idx;

        if(symm && irow != icol)
        {
            if(idx >= nnz)
            {
                return 1;
            }

            unsorted_row[idx] = icol;
            unsorted_col[idx] = irow;
            unsorted_val[idx] = ival;
            ++idx;
        }
    }
    fclose(f);

    row.resize(nnz);
    col.resize(nnz);
    val.resize(nnz);

    // Sort by row and column index
    std::vector<I> perm(nnz);
    for(int64_t i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    sort(perm, unsorted_row, unsorted_col);

    for(int64_t i = 0; i < nnz; ++i)
    {
        row[i] = unsorted_row[perm[i]];
        col[i] = unsorted_col[perm[i]];
        val[i] = unsorted_val[perm[i]];
    }

    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("done.\n");
        fflush(stdout);
    }

    return 0;
}

/* ============================================================================================ */
/*! \brief  Read matrix from binary file in CSR format */
template <typename I, typename J, typename T>
int read_bin_matrix(const char*          filename,
                    J&                   nrow,
                    J&                   ncol,
                    I&                   nnz,
                    std::vector<I>&      ptr,
                    std::vector<J>&      col,
                    std::vector<T>&      val,
                    hipsparseIndexBase_t idx_base)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("Reading matrix %s...", filename);
        fflush(stdout);
    }

    FILE* f = fopen(filename, "rb");
    if(!f)
    {
        return -1;
    }

    int err;

    int nrowf, ncolf, nnzf;

    err = fread(&nrowf, sizeof(int), 1, f);
    err |= fread(&ncolf, sizeof(int), 1, f);
    err |= fread(&nnzf, sizeof(int), 1, f);
    if(!err)
    {
        fclose(f);
        return -1;
    }
    nrow = (J)nrowf;
    ncol = (J)ncolf;
    nnz  = (I)nnzf;

    // Allocate memory
    std::vector<int>    ptrf(nrow + 1);
    std::vector<int>    colf(nnz);
    std::vector<double> valf(nnz);
    ptr.resize(nrow + 1);
    col.resize(nnz);
    val.resize(nnz);

    err |= fread(ptrf.data(), sizeof(int), nrow + 1, f);
    err |= fread(colf.data(), sizeof(int), nnz, f);
    err |= fread(valf.data(), sizeof(double), nnz, f);
    if(!err)
    {
        fclose(f);
        return -1;
    }

    fclose(f);

    for(J i = 0; i < nrow + 1; ++i)
    {
        ptr[i] = (I)ptrf[i];
    }

    for(I i = 0; i < nnz; ++i)
    {
        col[i] = (J)colf[i];
        val[i] = make_DataType<T>(valf[i]);
    }

    if(idx_base == HIPSPARSE_INDEX_BASE_ONE)
    {
        for(J i = 0; i < nrow + 1; ++i)
        {
            ++ptr[i];
        }

        for(I i = 0; i < nnz; ++i)
        {
            ++col[i];
        }
    }

    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("done.\n");
        fflush(stdout);
    }

    return 0;
}

/* ============================================================================================ */
/*! \brief  Generate CSR matrix from file. File can be either mtx or bin. If filename is empty, a random matrix is generated*/
template <typename I, typename J, typename T>
bool generate_csr_matrix(const std::string    filename,
                         J&                   nrow,
                         J&                   ncol,
                         I&                   nnz,
                         std::vector<I>&      csr_row_ptr,
                         std::vector<J>&      csr_col_ind,
                         std::vector<T>&      csr_val,
                         hipsparseIndexBase_t idx_base)
{
    // If no filename passed, generate matrix
    if(filename == "")
    {
        double scale = 0.02;
        if(nrow > 1000 || ncol > 1000)
        {
            scale = 2.0 / std::max(nrow, ncol);
        }
        nnz = nrow * scale * ncol;

        std::vector<J> coo_row_ind;
        gen_matrix_coo(nrow, ncol, (J)nnz, coo_row_ind, csr_col_ind, csr_val, idx_base);

        csr_row_ptr.resize(nrow + 1, 0);
        for(int i = 0; i < nnz; ++i)
        {
            ++csr_row_ptr[coo_row_ind[i] + 1 - idx_base];
        }

        csr_row_ptr[0] = idx_base;
        for(int i = 0; i < nrow; ++i)
        {
            csr_row_ptr[i + 1] += csr_row_ptr[i];
        }

        return true;
    }
    else
    {
        std::string extension = filename.substr(filename.find_last_of(".") + 1);
        if(extension == "bin")
        {
            if(read_bin_matrix(
                   filename.c_str(), nrow, ncol, nnz, csr_row_ptr, csr_col_ind, csr_val, idx_base)
               == 0)
            {
                return true;
            }
        }
        else if(extension == "mtx")
        {
            int64_t        nnz_count;
            std::vector<J> coo_row_ind;
            if(read_mtx_matrix(filename.c_str(),
                               nrow,
                               ncol,
                               nnz_count,
                               coo_row_ind,
                               csr_col_ind,
                               csr_val,
                               idx_base)
               == 0)
            {
                if(nnz_count < std::numeric_limits<I>::max())
                {
                    nnz = (I)nnz_count;

                    csr_row_ptr.resize(nrow + 1, 0);
                    for(int i = 0; i < nnz; ++i)
                    {
                        ++csr_row_ptr[coo_row_ind[i] + 1 - idx_base];
                    }

                    csr_row_ptr[0] = idx_base;
                    for(int i = 0; i < nrow; ++i)
                    {
                        csr_row_ptr[i + 1] += csr_row_ptr[i];
                    }

                    return true;
                }
            }
        }
    }

    return false;
}

/* ============================================================================================ */
/*! \brief  Generate COO matrix from file. File can be either mtx or bin. If filename is empty, a random matrix is generated*/
template <typename I, typename T>
bool generate_coo_matrix(const std::string    filename,
                         I&                   nrow,
                         I&                   ncol,
                         I&                   nnz,
                         std::vector<I>&      coo_row_ind,
                         std::vector<I>&      coo_col_ind,
                         std::vector<T>&      coo_val,
                         hipsparseIndexBase_t idx_base)
{
    // If no filename passed, generate matrix
    if(filename == "")
    {
        double scale = 0.02;
        if(nrow > 1000 || ncol > 1000)
        {
            scale = 2.0 / std::max(nrow, ncol);
        }
        nnz = nrow * scale * ncol;

        gen_matrix_coo(nrow, ncol, nnz, coo_row_ind, coo_col_ind, coo_val, idx_base);

        return true;
    }
    else
    {
        std::string extension = filename.substr(filename.find_last_of(".") + 1);
        if(extension == "bin")
        {
            std::vector<I> csr_row_ptr;
            if(read_bin_matrix(
                   filename.c_str(), nrow, ncol, nnz, csr_row_ptr, coo_col_ind, coo_val, idx_base)
               == 0)
            {
                coo_row_ind.resize(nnz);
                for(I i = 0; i < nrow; ++i)
                {
                    I row_begin = csr_row_ptr[i] - idx_base;
                    I row_end   = csr_row_ptr[i + 1] - idx_base;

                    for(I j = row_begin; j < row_end; ++j)
                    {
                        coo_row_ind[j] = i + idx_base;
                    }
                }

                return true;
            }
        }
        else if(extension == "mtx")
        {
            int64_t nnz_count;
            if(read_mtx_matrix(filename.c_str(),
                               nrow,
                               ncol,
                               nnz_count,
                               coo_row_ind,
                               coo_col_ind,
                               coo_val,
                               idx_base)
               == 0)
            {
                if(nnz_count < std::numeric_limits<I>::max())
                {
                    nnz = (I)nnz_count;
                    return true;
                }
            }
        }
    }

    return false;
}

/* ============================================================================================ */
/*! \brief  Compute incomplete LU factorization without fill-ins and no pivoting using CSR
 *  matrix storage format.
 */
static inline float testing_neg(float val)
{
    return -val;
}

static inline double testing_neg(double val)
{
    return -val;
}

static inline hipComplex testing_neg(hipComplex val)
{
    hipComplex ret;
    ret.x = -val.x;
    ret.y = -val.y;
    return ret;
}

static inline hipDoubleComplex testing_neg(hipDoubleComplex val)
{
    hipDoubleComplex ret;
    ret.x = -val.x;
    ret.y = -val.y;
    return ret;
}

template <typename T>
void host_nnz(hipsparseDirection_t      dirA,
              int                       m,
              int                       n,
              const hipsparseMatDescr_t descrA,
              const T*                  A,
              int                       lda,
              int*                      nnzPerRowColumn,
              int*                      nnzTotalDevHostPtr)
{
    int mn = (dirA == HIPSPARSE_DIRECTION_ROW) ? m : n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int j = 0; j < mn; ++j)
    {
        nnzPerRowColumn[j] = 0;
    }

    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            if(A[j * lda + i] != make_DataType<T>(0))
            {
                if(dirA == HIPSPARSE_DIRECTION_ROW)
                {
                    nnzPerRowColumn[i] += 1;
                }
                else
                {
                    nnzPerRowColumn[j] += 1;
                }
            }
        }
    }

    int sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for(int j = 0; j < mn; ++j)
    {
        sum = sum + nnzPerRowColumn[j];
    }
    nnzTotalDevHostPtr[0] = sum;
}

template <hipsparseDirection_t DIRA, typename T>
void host_dense2csx(int                  m,
                    int                  n,
                    hipsparseIndexBase_t base,
                    const T*             A,
                    int                  ld,
                    const int*           nnz_per_row_columns,
                    T*                   csx_val,
                    int*                 csx_row_col_ptr,
                    int*                 csx_col_row_ind)
{
    static constexpr T s_zero = {};
    int                len    = (HIPSPARSE_DIRECTION_ROW == DIRA) ? m : n;
    *csx_row_col_ptr          = base;
    for(int i = 0; i < len; ++i)
    {
        csx_row_col_ptr[i + 1] = nnz_per_row_columns[i] + csx_row_col_ptr[i];
    }

    switch(DIRA)
    {
    case HIPSPARSE_DIRECTION_COLUMN:
    {
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < m; ++i)
            {
                if(A[j * ld + i] != s_zero)
                {
                    *csx_val++         = A[j * ld + i];
                    *csx_col_row_ind++ = i + base;
                }
            }
        }
        break;
    }

    case HIPSPARSE_DIRECTION_ROW:
    {
        //
        // Does not matter having an orthogonal traversal ... testing only.
        // Otherwise, we would use csxRowPtrA to store the shifts.
        // and once the job is done a simple memory move would reinitialize the csxRowPtrA to its initial state)
        //
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(A[j * ld + i] != s_zero)
                {
                    *csx_val++         = A[j * ld + i];
                    *csx_col_row_ind++ = j + base;
                }
            }
        }
        break;
    }
    }
}

template <typename T>
void host_prune_dense2csr(int                   m,
                          int                   n,
                          const std::vector<T>& A,
                          int                   lda,
                          hipsparseIndexBase_t  base,
                          T                     threshold,
                          int&                  nnz,
                          std::vector<T>&       csr_val,
                          std::vector<int>&     csr_row_ptr,
                          std::vector<int>&     csr_col_ind)
{
    csr_row_ptr.resize(m + 1, 0);
    csr_row_ptr[0] = base;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(testing_abs(A[lda * j + i]) > threshold)
            {
                csr_row_ptr[i + 1]++;
            }
        }
    }

    for(int i = 1; i <= m; i++)
    {
        csr_row_ptr[i] += csr_row_ptr[i - 1];
    }

    nnz = csr_row_ptr[m] - csr_row_ptr[0];

    csr_col_ind.resize(nnz);
    csr_val.resize(nnz);

    int index = 0;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(testing_abs(A[lda * j + i]) > threshold)
            {
                csr_val[index]     = A[lda * j + i];
                csr_col_ind[index] = j + base;

                index++;
            }
        }
    }
}

template <typename T>
void host_prune_dense2csr_by_percentage(int                   m,
                                        int                   n,
                                        const std::vector<T>& A,
                                        int                   lda,
                                        hipsparseIndexBase_t  base,
                                        T                     percentage,
                                        int&                  nnz,
                                        std::vector<T>&       csr_val,
                                        std::vector<int>&     csr_row_ptr,
                                        std::vector<int>&     csr_col_ind)
{
    int nnz_A = m * n;
    int pos   = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos       = std::min(pos, nnz_A - 1);
    pos       = std::max(pos, 0);

    std::vector<T> sorted_A(nnz_A);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            sorted_A[m * i + j] = std::abs(A[lda * i + j]);
        }
    }

    std::sort(sorted_A.begin(), sorted_A.end());

    T threshold = (nnz_A > 0) ? sorted_A[pos] : make_DataType<T>(0);
    host_prune_dense2csr<T>(m, n, A, lda, base, threshold, nnz, csr_val, csr_row_ptr, csr_col_ind);
}

template <hipsparseDirection_t DIRA, typename T>
void host_csx2dense(int                  m,
                    int                  n,
                    hipsparseIndexBase_t base,
                    const T*             csx_val,
                    const int*           csx_row_col_ptr,
                    const int*           csx_col_row_ind,
                    T*                   A,
                    int                  ld)
{
    static constexpr T s_zero = {};
    switch(DIRA)
    {
    case HIPSPARSE_DIRECTION_COLUMN:
    {
        for(int col = 0; col < n; ++col)
        {
            for(int row = 0; row < m; ++row)
            {
                A[row + ld * col] = s_zero;
            }
            const int bound = csx_row_col_ptr[col + 1] - base;
            for(int at = csx_row_col_ptr[col] - base; at < bound; ++at)
            {
                A[(csx_col_row_ind[at] - base) + ld * col] = csx_val[at];
            }
        }
        break;
    }

    case HIPSPARSE_DIRECTION_ROW:
    {
        for(int row = 0; row < m; ++row)
        {
            for(int col = 0; col < n; ++col)
            {
                A[col * ld + row] = s_zero;
            }

            const int bound = csx_row_col_ptr[row + 1] - base;
            for(int at = csx_row_col_ptr[row] - base; at < bound; ++at)
            {
                A[(csx_col_row_ind[at] - base) * ld + row] = csx_val[at];
            }
        }
        break;
    }
    }
}

template <typename T>
inline void host_csr_to_csr_compress(int                     M,
                                     int                     N,
                                     const std::vector<int>& csr_row_ptr_A,
                                     const std::vector<int>& csr_col_ind_A,
                                     const std::vector<T>&   csr_val_A,
                                     std::vector<int>&       csr_row_ptr_C,
                                     std::vector<int>&       csr_col_ind_C,
                                     std::vector<T>&         csr_val_C,
                                     hipsparseIndexBase_t    base,
                                     T                       tol)
{
    if(M <= 0 || N <= 0)
    {
        return;
    }

    // find how many entries will be in each compressed CSR matrix row
    std::vector<int> nnz_per_row(M);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; i++)
    {
        int start = csr_row_ptr_A[i] - base;
        int end   = csr_row_ptr_A[i + 1] - base;
        int count = 0;

        for(int j = start; j < end; j++)
        {
            if(testing_abs(csr_val_A[j]) > testing_real(tol)
               && testing_abs(csr_val_A[j]) > (std::numeric_limits<float>::min)())
            {
                count++;
            }
        }

        nnz_per_row[i] = count;
    }

    // add up total number of entries
    int nnz_C = 0;
    for(int i = 0; i < M; i++)
    {
        nnz_C += nnz_per_row[i];
    }

    //column indices and value arrays for compressed CSR matrix
    csr_col_ind_C.resize(nnz_C);
    csr_val_C.resize(nnz_C);

    // fill in row pointer array for compressed CSR matrix
    csr_row_ptr_C.resize(M + 1);

    csr_row_ptr_C[0] = base;
    for(int i = 0; i < M; i++)
    {
        csr_row_ptr_C[i + 1] = csr_row_ptr_C[i] + nnz_per_row[i];
    }

    // fill in column indices and value arrays for compressed CSR matrix
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; i++)
    {
        int start = csr_row_ptr_A[i] - base;
        int end   = csr_row_ptr_A[i + 1] - base;
        int index = csr_row_ptr_C[i] - base;

        for(int j = start; j < end; j++)
        {
            if(testing_abs(csr_val_A[j]) > testing_real(tol)
               && testing_abs(csr_val_A[j]) > (std::numeric_limits<float>::min)())
            {
                csr_col_ind_C[index] = csr_col_ind_A[j];
                csr_val_C[index]     = csr_val_A[j];
                index++;
            }
        }
    }
}

template <typename T>
inline void host_prune_csr_to_csr(int                     M,
                                  int                     N,
                                  int                     nnz_A,
                                  const std::vector<int>& csr_row_ptr_A,
                                  const std::vector<int>& csr_col_ind_A,
                                  const std::vector<T>&   csr_val_A,
                                  int&                    nnz_C,
                                  std::vector<int>&       csr_row_ptr_C,
                                  std::vector<int>&       csr_col_ind_C,
                                  std::vector<T>&         csr_val_C,
                                  hipsparseIndexBase_t    csr_base_A,
                                  hipsparseIndexBase_t    csr_base_C,
                                  T                       threshold)
{
    csr_row_ptr_C.resize(M + 1, 0);
    csr_row_ptr_C[0] = csr_base_C;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; i++)
    {
        for(int j = csr_row_ptr_A[i] - csr_base_A; j < csr_row_ptr_A[i + 1] - csr_base_A; j++)
        {
            if(testing_abs(csr_val_A[j]) > threshold
               && testing_abs(csr_val_A[j]) > (std::numeric_limits<float>::min)())
            {
                csr_row_ptr_C[i + 1]++;
            }
        }
    }

    for(int i = 1; i <= M; i++)
    {
        csr_row_ptr_C[i] += csr_row_ptr_C[i - 1];
    }

    nnz_C = csr_row_ptr_C[M] - csr_row_ptr_C[0];

    csr_col_ind_C.resize(nnz_C);
    csr_val_C.resize(nnz_C);

    int index = 0;
    for(int i = 0; i < M; i++)
    {
        for(int j = csr_row_ptr_A[i] - csr_base_A; j < csr_row_ptr_A[i + 1] - csr_base_A; j++)
        {
            if(testing_abs(csr_val_A[j]) > threshold
               && testing_abs(csr_val_A[j]) > (std::numeric_limits<float>::min)())
            {
                csr_col_ind_C[index] = (csr_col_ind_A[j] - csr_base_A) + csr_base_C;
                csr_val_C[index]     = csr_val_A[j];

                index++;
            }
        }
    }
}

template <typename T>
void host_prune_csr_to_csr_by_percentage(int                     M,
                                         int                     N,
                                         int                     nnz_A,
                                         const std::vector<int>& csr_row_ptr_A,
                                         const std::vector<int>& csr_col_ind_A,
                                         const std::vector<T>&   csr_val_A,
                                         int&                    nnz_C,
                                         std::vector<int>&       csr_row_ptr_C,
                                         std::vector<int>&       csr_col_ind_C,
                                         std::vector<T>&         csr_val_C,
                                         hipsparseIndexBase_t    csr_base_A,
                                         hipsparseIndexBase_t    csr_base_C,
                                         T                       percentage)
{
    int pos = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos     = std::min(pos, nnz_A - 1);
    pos     = std::max(pos, 0);

    std::vector<T> sorted_A(nnz_A);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < nnz_A; i++)
    {
        sorted_A[i] = testing_abs(csr_val_A[i]);
    }

    std::sort(sorted_A.begin(), sorted_A.end());

    T threshold = sorted_A[pos];

    host_prune_csr_to_csr<T>(M,
                             N,
                             nnz_A,
                             csr_row_ptr_A,
                             csr_col_ind_A,
                             csr_val_A,
                             nnz_C,
                             csr_row_ptr_C,
                             csr_col_ind_C,
                             csr_val_C,
                             csr_base_A,
                             csr_base_C,
                             threshold);
}

template <typename I, typename J, typename T>
inline void host_csr_to_csc(J        M,
                            J        N,
                            I        nnz,
                            const I* csr_row_ptr,
                            const J* csr_col_ind,
                            const T* csr_val,
                            //const std::vector<I>& csr_row_ptr,
                            //const std::vector<J>& csr_col_ind,
                            //const std::vector<T>&   csr_val,
                            std::vector<J>&      csc_row_ind,
                            std::vector<I>&      csc_col_ptr,
                            std::vector<T>&      csc_val,
                            hipsparseAction_t    action,
                            hipsparseIndexBase_t base)
{
    csc_row_ind.resize(nnz);
    csc_col_ptr.resize(N + 1, 0);
    csc_val.resize(nnz);

    // Determine nnz per column
    for(I i = 0; i < nnz; ++i)
    {
        ++csc_col_ptr[csr_col_ind[i] + 1 - base];
    }

    // Scan
    for(J i = 0; i < N; ++i)
    {
        csc_col_ptr[i + 1] += csc_col_ptr[i];
    }

    // Fill row indices and values
    for(J i = 0; i < M; ++i)
    {
        I row_begin = csr_row_ptr[i] - base;
        I row_end   = csr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; ++j)
        {
            J col = csr_col_ind[j] - base;
            I idx = csc_col_ptr[col];

            csc_row_ind[idx] = i + base;
            csc_val[idx]     = csr_val[j];

            ++csc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(J i = N; i > 0; --i)
    {
        csc_col_ptr[i] = csc_col_ptr[i - 1] + base;
    }

    csc_col_ptr[0] = base;
}

template <typename T>
inline void host_csr_to_bsr(hipsparseDirection_t    direction,
                            int                     M,
                            int                     N,
                            int                     block_dim,
                            int&                    nnzb,
                            hipsparseIndexBase_t    csr_base,
                            const std::vector<int>& csr_row_ptr,
                            const std::vector<int>& csr_col_ind,
                            const std::vector<T>&   csr_val,
                            hipsparseIndexBase_t    bsr_base,
                            std::vector<int>&       bsr_row_ptr,
                            std::vector<int>&       bsr_col_ind,
                            std::vector<T>&         bsr_val)
{
    int mb = (M + block_dim - 1) / block_dim;
    int nb = (N + block_dim - 1) / block_dim;

    // quick return if block_dim == 1
    if(block_dim == 1)
    {
        bsr_row_ptr.resize(mb + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_row_ptr.size(); i++)
        {
            bsr_row_ptr[i] = (csr_row_ptr[i] - csr_base) + bsr_base;
        }

        nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

        bsr_col_ind.resize(nnzb, 0);
        bsr_val.resize(nnzb * block_dim * block_dim, make_DataType<T>(0));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_col_ind.size(); i++)
        {
            bsr_col_ind[i] = (csr_col_ind[i] - csr_base) + bsr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_val.size(); i++)
        {
            bsr_val[i] = csr_val[i];
        }

        return;
    }

    // determine number of non-zero block columns for each block row of the bsr matrix
    bsr_row_ptr.resize(mb + 1, 0);

    bsr_row_ptr[0] = bsr_base;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < mb; i++)
    {
        int start = csr_row_ptr[i * block_dim] - csr_base;
        int end   = csr_row_ptr[std::min(M, block_dim * i + block_dim)] - csr_base;

        std::vector<int> temp(nb, 0);
        for(int j = start; j < end; j++)
        {
            int blockCol   = (csr_col_ind[j] - csr_base) / block_dim;
            temp[blockCol] = 1;
        }

        int sum = 0;
        for(size_t j = 0; j < temp.size(); j++)
        {
            sum += temp[j];
        }

        bsr_row_ptr[i + 1] = sum;
    }

    for(int i = 0; i < mb; i++)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

    // find bsr col indices array
    bsr_col_ind.resize(nnzb, 0);
    bsr_val.resize(nnzb * block_dim * block_dim, make_DataType<T>(0));

    int colIndex = 0;

    for(int i = 0; i < mb; i++)
    {
        int start = csr_row_ptr[i * block_dim] - csr_base;
        int end   = csr_row_ptr[std::min(M, block_dim * i + block_dim)] - csr_base;

        std::vector<int> temp(nb, 0);

        for(int j = start; j < end; j++)
        {
            int blockCol   = (csr_col_ind[j] - csr_base) / block_dim;
            temp[blockCol] = 1;
        }

        for(int j = 0; j < nb; j++)
        {
            if(temp[j] == 1)
            {
                bsr_col_ind[colIndex] = j + bsr_base;
                colIndex++;
            }
        }
    }

    // find bsr values array
    for(int i = 0; i < M; i++)
    {
        int blockRow = i / block_dim;

        int start = csr_row_ptr[i] - csr_base;
        int end   = csr_row_ptr[i + 1] - csr_base;

        for(int j = start; j < end; j++)
        {
            int blockCol = (csr_col_ind[j] - csr_base) / block_dim;

            colIndex = -1;
            for(int k = bsr_row_ptr[blockRow] - bsr_base; k < bsr_row_ptr[blockRow + 1] - bsr_base;
                k++)
            {
                if(bsr_col_ind[k] - bsr_base == blockCol)
                {
                    colIndex = k - (bsr_row_ptr[blockRow] - bsr_base);
                    break;
                }
            }

            assert(colIndex != -1);

            int blockIndex = 0;
            if(direction == HIPSPARSE_DIRECTION_ROW)
            {
                blockIndex = (csr_col_ind[j] - csr_base) % block_dim + (i % block_dim) * block_dim;
            }
            else
            {
                blockIndex
                    = ((csr_col_ind[j] - csr_base) % block_dim) * block_dim + (i % block_dim);
            }

            int index = (bsr_row_ptr[blockRow] - bsr_base) * block_dim * block_dim
                        + colIndex * block_dim * block_dim + blockIndex;

            bsr_val[index] = csr_val[j];
        }
    }
}

template <typename T>
void host_bsr_to_bsc(int                  mb,
                     int                  nb,
                     int                  nnzb,
                     int                  bsr_dim,
                     const int*           bsr_row_ptr,
                     const int*           bsr_col_ind,
                     const T*             bsr_val,
                     std::vector<int>&    bsc_row_ind,
                     std::vector<int>&    bsc_col_ptr,
                     std::vector<T>&      bsc_val,
                     hipsparseIndexBase_t bsr_base,
                     hipsparseIndexBase_t bsc_base)
{
    bsc_row_ind.resize(nnzb);
    bsc_col_ptr.resize(nb + 1, 0);
    bsc_val.resize(nnzb * bsr_dim * bsr_dim);

    // Determine nnz per column
    for(int i = 0; i < nnzb; ++i)
    {
        ++bsc_col_ptr[bsr_col_ind[i] + 1 - bsr_base];
    }

    // Scan
    for(int i = 0; i < nb; ++i)
    {
        bsc_col_ptr[i + 1] += bsc_col_ptr[i];
    }

    // Fill row indices and values
    for(int i = 0; i < mb; ++i)
    {
        int row_begin = bsr_row_ptr[i] - bsr_base;
        int row_end   = bsr_row_ptr[i + 1] - bsr_base;

        for(int j = row_begin; j < row_end; ++j)
        {
            int col = bsr_col_ind[j] - bsr_base;
            int idx = bsc_col_ptr[col];

            bsc_row_ind[idx] = i + bsc_base;

            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                for(int bj = 0; bj < bsr_dim; ++bj)
                {
                    bsc_val[bsr_dim * bsr_dim * idx + bi + bj * bsr_dim]
                        = bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj];
                }
            }

            ++bsc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(int i = nb; i > 0; --i)
    {
        bsc_col_ptr[i] = bsc_col_ptr[i - 1] + bsc_base;
    }

    bsc_col_ptr[0] = bsc_base;
}

template <typename T>
inline void host_gebsr_to_csr(hipsparseDirection_t    direction,
                              int                     mb,
                              int                     nb,
                              int                     nnzb,
                              const std::vector<T>&   bsr_val,
                              const std::vector<int>& bsr_row_ptr,
                              const std::vector<int>& bsr_col_ind,
                              int                     row_block_dim,
                              int                     col_block_dim,
                              hipsparseIndexBase_t    bsr_base,
                              std::vector<T>&         csr_val,
                              std::vector<int>&       csr_row_ptr,
                              std::vector<int>&       csr_col_ind,
                              hipsparseIndexBase_t    csr_base)
{

    csr_col_ind.resize(nnzb * row_block_dim * col_block_dim);
    csr_row_ptr.resize(mb * row_block_dim + 1);
    csr_val.resize(nnzb * row_block_dim * col_block_dim);
    int at         = 0;
    csr_row_ptr[0] = csr_base;
    for(int i = 0; i < mb; ++i)
    {
        for(int r = 0; r < row_block_dim; ++r)
        {
            int row = i * row_block_dim + r;
            for(int k = bsr_row_ptr[i] - bsr_base; k < bsr_row_ptr[i + 1] - bsr_base; ++k)
            {
                int j = bsr_col_ind[k] - bsr_base;
                for(int c = 0; c < col_block_dim; ++c)
                {
                    int col         = col_block_dim * j + c;
                    csr_col_ind[at] = col + csr_base;
                    if(direction == HIPSPARSE_DIRECTION_ROW)
                    {
                        csr_val[at]
                            = bsr_val[k * row_block_dim * col_block_dim + col_block_dim * r + c];
                    }
                    else
                    {
                        csr_val[at]
                            = bsr_val[k * row_block_dim * col_block_dim + row_block_dim * c + r];
                    }
                    ++at;
                }
            }

            csr_row_ptr[row + 1]
                = csr_row_ptr[row] + (bsr_row_ptr[i + 1] - bsr_row_ptr[i]) * col_block_dim;
        }
    }
}

template <typename T>
inline void host_csr_to_gebsr(hipsparseDirection_t    direction,
                              int                     m,
                              int                     n,
                              int                     row_block_dim,
                              int                     col_block_dim,
                              int&                    nnzb,
                              hipsparseIndexBase_t    csr_base,
                              const std::vector<int>& csr_row_ptr,
                              const std::vector<int>& csr_col_ind,
                              const std::vector<T>&   csr_val,
                              hipsparseIndexBase_t    bsr_base,
                              std::vector<int>&       bsr_row_ptr,
                              std::vector<int>&       bsr_col_ind,
                              std::vector<T>&         bsr_val)
{
    int mb  = (m + row_block_dim - 1) / row_block_dim;
    int nnz = csr_col_ind.size();

    bsr_row_ptr.resize(mb + 1, 0);

    std::vector<int> temp(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < nnz; i++)
    {
        temp[i] = (csr_col_ind[i] - csr_base) / col_block_dim;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < mb; i++)
    {
        int frow = row_block_dim * i;
        int lrow = row_block_dim * (i + 1);

        if(lrow > m)
        {
            lrow = m;
        }

        int start = csr_row_ptr[frow] - csr_base;
        int end   = csr_row_ptr[lrow] - csr_base;

        std::sort(temp.begin() + start, temp.begin() + end);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < mb; i++)
    {
        int frow = row_block_dim * i;
        int lrow = row_block_dim * (i + 1);

        if(lrow > m)
        {
            lrow = m;
        }

        int start = csr_row_ptr[frow] - csr_base;
        int end   = csr_row_ptr[lrow] - csr_base;

        int col   = -1;
        int count = 0;
        for(int j = start; j < end; j++)
        {
            if(temp[j] > col)
            {
                col                 = temp[j];
                temp[j]             = -1;
                temp[start + count] = col;
                count++;
            }
            else
            {
                temp[j] = -1;
            }
        }

        bsr_row_ptr[i + 1] = count;
    }

    // fill GEBSR row pointer array
    bsr_row_ptr[0] = bsr_base;
    for(int i = 0; i < mb; i++)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];
    bsr_col_ind.resize(nnzb);
    bsr_val.resize(nnzb * row_block_dim * col_block_dim, make_DataType<T>(0));

    // fill GEBSR col indices array
    {
        int index = 0;
        for(int i = 0; i < nnz; i++)
        {
            if(temp[i] != -1)
            {
                bsr_col_ind[index] = temp[i] + bsr_base;
                index++;
            }
        }
    }

    // fill GEBSR values array
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i] - csr_base;
        int end   = csr_row_ptr[i + 1] - csr_base;

        int bstart = bsr_row_ptr[i / row_block_dim] - bsr_base;
        int bend   = bsr_row_ptr[i / row_block_dim + 1] - bsr_base;

        int local_row = i % row_block_dim;

        for(int j = start; j < end; j++)
        {
            int col = csr_col_ind[j] - csr_base;

            int local_col = col % col_block_dim;

            {
                int index = 0;
                for(int k = bstart; k < bend; k++)
                {
                    if(bsr_col_ind[k] - bsr_base == col / col_block_dim)
                    {
                        index  = k;
                        bstart = k;
                        break;
                    }
                }

                if(direction == HIPSPARSE_DIRECTION_ROW)
                {
                    bsr_val[row_block_dim * col_block_dim * index + col_block_dim * local_row
                            + local_col]
                        = csr_val[j];
                }
                else
                {
                    bsr_val[row_block_dim * col_block_dim * index + row_block_dim * local_col
                            + local_row]
                        = csr_val[j];
                }
            }
        }
    }
}

template <typename T>
inline void host_gebsr_to_gebsr(hipsparseDirection_t    direction,
                                int                     mb,
                                int                     nb,
                                int                     nnzb,
                                const std::vector<T>&   bsr_val_A,
                                const std::vector<int>& bsr_row_ptr_A,
                                const std::vector<int>& bsr_col_ind_A,
                                int                     row_block_dim_A,
                                int                     col_block_dim_A,
                                hipsparseIndexBase_t    base_A,
                                std::vector<T>&         bsr_val_C,
                                std::vector<int>&       bsr_row_ptr_C,
                                std::vector<int>&       bsr_col_ind_C,
                                int                     row_block_dim_C,
                                int                     col_block_dim_C,
                                hipsparseIndexBase_t    base_C)
{
    int m = mb * row_block_dim_A;
    int n = nb * col_block_dim_A;

    // convert GEBSR to CSR format
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<T>   csr_val;

    host_gebsr_to_csr(direction,
                      mb,
                      nb,
                      nnzb,
                      bsr_val_A,
                      bsr_row_ptr_A,
                      bsr_col_ind_A,
                      row_block_dim_A,
                      col_block_dim_A,
                      base_A,
                      csr_val,
                      csr_row_ptr,
                      csr_col_ind,
                      HIPSPARSE_INDEX_BASE_ZERO);

    // convert CSR to GEBSR format
    int nnzb_C;
    host_csr_to_gebsr(direction,
                      m,
                      n,
                      row_block_dim_C,
                      col_block_dim_C,
                      nnzb_C,
                      HIPSPARSE_INDEX_BASE_ZERO,
                      csr_row_ptr,
                      csr_col_ind,
                      csr_val,
                      base_C,
                      bsr_row_ptr_C,
                      bsr_col_ind_C,
                      bsr_val_C);
}

template <typename T>
void host_gebsr_to_gebsc(int                     Mb,
                         int                     Nb,
                         int                     nnzb,
                         const std::vector<int>& bsr_row_ptr,
                         const std::vector<int>& bsr_col_ind,
                         const std::vector<T>&   bsr_val,
                         int                     row_block_dim,
                         int                     col_block_dim,
                         std::vector<int>&       bsc_row_ind,
                         std::vector<int>&       bsc_col_ptr,
                         std::vector<T>&         bsc_val,
                         hipsparseAction_t       action,
                         hipsparseIndexBase_t    base)
{
    bsc_row_ind.resize(nnzb);
    bsc_col_ptr.resize(Nb + 1, 0);
    bsc_val.resize(nnzb);

    const int block_shift = row_block_dim * col_block_dim;

    //
    // Determine nnz per column
    //
    for(int i = 0; i < nnzb; ++i)
    {
        ++bsc_col_ptr[bsr_col_ind[i] + 1 - base];
    }

    // Scan
    for(int i = 0; i < Nb; ++i)
    {
        bsc_col_ptr[i + 1] += bsc_col_ptr[i];
    }

    // Fill row indices and values
    for(int i = 0; i < Mb; ++i)
    {
        const int row_begin = bsr_row_ptr[i] - base;
        const int row_end   = bsr_row_ptr[i + 1] - base;

        for(int j = row_begin; j < row_end; ++j)
        {
            const int col = bsr_col_ind[j] - base;
            const int idx = bsc_col_ptr[col];

            bsc_row_ind[idx] = i + base;
            for(int k = 0; k < block_shift; ++k)
            {
                bsc_val[idx * block_shift + k] = bsr_val[j * block_shift + k];
            }

            ++bsc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(int i = Nb; i > 0; --i)
    {
        bsc_col_ptr[i] = bsc_col_ptr[i - 1] + base;
    }

    bsc_col_ptr[0] = base;
}

template <typename T>
inline void host_bsr_to_csr(hipsparseDirection_t    direction,
                            int                     Mb,
                            int                     Nb,
                            int                     block_dim,
                            hipsparseIndexBase_t    bsr_base,
                            const std::vector<int>& bsr_row_ptr,
                            const std::vector<int>& bsr_col_ind,
                            const std::vector<T>&   bsr_val,
                            hipsparseIndexBase_t    csr_base,
                            std::vector<int>&       csr_row_ptr,
                            std::vector<int>&       csr_col_ind,
                            std::vector<T>&         csr_val)
{
    int m    = Mb * block_dim;
    int nnzb = bsr_row_ptr[Mb] - bsr_row_ptr[0];

    csr_row_ptr.resize(m + 1, 0);
    csr_col_ind.resize(nnzb * block_dim * block_dim, 0);
    csr_val.resize(nnzb * block_dim * block_dim, make_DataType<T>(0));

    // quick return if block_dim == 1
    if(block_dim == 1)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_row_ptr.size(); i++)
        {
            csr_row_ptr[i] = (bsr_row_ptr[i] - bsr_base) + csr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_col_ind.size(); i++)
        {
            csr_col_ind[i] = (bsr_col_ind[i] - bsr_base) + csr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_val.size(); i++)
        {
            csr_val[i] = bsr_val[i];
        }

        return;
    }

    csr_row_ptr[0] = csr_base;

    // find csr row ptr array
    for(int i = 0; i < Mb; i++)
    {
        int entries_in_row = block_dim * (bsr_row_ptr[i + 1] - bsr_row_ptr[i]);

        for(int j = 0; j < block_dim; j++)
        {
            csr_row_ptr[i * block_dim + j + 1] = csr_row_ptr[i * block_dim + j] + entries_in_row;
        }
    }

    int entries_in_block = block_dim * block_dim;

    // find csr col indices and values arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < Mb; i++)
    {
        int entries_in_Row     = (bsr_row_ptr[i + 1] - bsr_row_ptr[i]) * block_dim;
        int entries_in_row_sum = (bsr_row_ptr[i] - bsr_base) * entries_in_block;

        for(int j = bsr_row_ptr[i] - bsr_base; j < bsr_row_ptr[i + 1] - bsr_base; j++)
        {
            int col    = bsr_col_ind[j] - bsr_base;
            int offset = entries_in_row_sum + block_dim * (j - (bsr_row_ptr[i] - bsr_base));

            for(int k = 0; k < block_dim; k++)
            {
                for(int l = 0; l < block_dim; l++)
                {
                    csr_col_ind[offset + k * entries_in_Row + l] = block_dim * col + l + csr_base;
                    if(direction == HIPSPARSE_DIRECTION_ROW)
                    {
                        csr_val[offset + k * entries_in_Row + l]
                            = bsr_val[j * entries_in_block + k * block_dim + l];
                    }
                    else
                    {
                        csr_val[offset + k * entries_in_Row + l]
                            = bsr_val[j * entries_in_block + k + block_dim * l];
                    }
                }
            }
        }
    }
}

template <typename T>
inline void host_bsrmv(hipsparseDirection_t dir,
                       hipsparseOperation_t trans,
                       int                  mb,
                       int                  nb,
                       int                  nnzb,
                       T                    alpha,
                       const int*           bsr_row_ptr,
                       const int*           bsr_col_ind,
                       const T*             bsr_val,
                       int                  bsr_dim,
                       const T*             x,
                       T                    beta,
                       T*                   y,
                       hipsparseIndexBase_t base)
{
    // Quick return
    if(alpha == make_DataType<T>(0))
    {
        if(beta != make_DataType<T>(1))
        {
            for(int i = 0; i < mb * bsr_dim; ++i)
            {
                y[i] = testing_mult(beta, y[i]);
            }
        }

        return;
    }

    int WFSIZE;

    if(bsr_dim == 2)
    {
        int blocks_per_row = nnzb / mb;

        if(blocks_per_row < 8)
        {
            WFSIZE = 4;
        }
        else if(blocks_per_row < 16)
        {
            WFSIZE = 8;
        }
        else if(blocks_per_row < 32)
        {
            WFSIZE = 16;
        }
        else if(blocks_per_row < 64)
        {
            WFSIZE = 32;
        }
        else
        {
            WFSIZE = 64;
        }
    }
    else if(bsr_dim <= 8)
    {
        WFSIZE = 8;
    }
    else if(bsr_dim <= 16)
    {
        WFSIZE = 16;
    }
    else
    {
        WFSIZE = 32;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int row = 0; row < mb; ++row)
    {
        int row_begin = bsr_row_ptr[row] - base;
        int row_end   = bsr_row_ptr[row + 1] - base;

        if(bsr_dim == 2)
        {
            std::vector<T> sum0(WFSIZE, make_DataType<T>(0));
            std::vector<T> sum1(WFSIZE, make_DataType<T>(0));

            for(int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        int col = bsr_col_ind[j + k] - base;

                        if(dir == HIPSPARSE_DIRECTION_COLUMN)
                        {
                            sum0[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 0],
                                                  x[col * bsr_dim + 0],
                                                  sum0[k]);
                            sum1[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 1],
                                                  x[col * bsr_dim + 0],
                                                  sum1[k]);
                            sum0[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 2],
                                                  x[col * bsr_dim + 1],
                                                  sum0[k]);
                            sum1[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 3],
                                                  x[col * bsr_dim + 1],
                                                  sum1[k]);
                        }
                        else
                        {
                            sum0[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 0],
                                                  x[col * bsr_dim + 0],
                                                  sum0[k]);
                            sum0[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 1],
                                                  x[col * bsr_dim + 1],
                                                  sum0[k]);
                            sum1[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 2],
                                                  x[col * bsr_dim + 0],
                                                  sum1[k]);
                            sum1[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 3],
                                                  x[col * bsr_dim + 1],
                                                  sum1[k]);
                        }
                    }
                }
            }

            for(int j = 1; j < WFSIZE; j <<= 1)
            {
                for(int k = 0; k < WFSIZE - j; ++k)
                {
                    sum0[k] = sum0[k] + sum0[k + j];
                    sum1[k] = sum1[k] + sum1[k + j];
                }
            }

            if(beta != make_DataType<T>(0))
            {
                y[row * bsr_dim + 0]
                    = testing_fma(beta, y[row * bsr_dim + 0], testing_mult(alpha, sum0[0]));
                y[row * bsr_dim + 1]
                    = testing_fma(beta, y[row * bsr_dim + 1], testing_mult(alpha, sum1[0]));
            }
            else
            {
                y[row * bsr_dim + 0] = testing_mult(alpha, sum0[0]);
                y[row * bsr_dim + 1] = testing_mult(alpha, sum1[0]);
            }
        }
        else
        {
            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                std::vector<T> sum(WFSIZE, make_DataType<T>(0));

                for(int j = row_begin; j < row_end; ++j)
                {
                    int col = bsr_col_ind[j] - base;

                    for(int bj = 0; bj < bsr_dim; bj += WFSIZE)
                    {
                        for(int k = 0; k < WFSIZE; ++k)
                        {
                            if(bj + k < bsr_dim)
                            {
                                if(dir == HIPSPARSE_DIRECTION_COLUMN)
                                {
                                    sum[k] = testing_fma(
                                        bsr_val[bsr_dim * bsr_dim * j + bsr_dim * (bj + k) + bi],
                                        x[bsr_dim * col + (bj + k)],
                                        sum[k]);
                                }
                                else
                                {
                                    sum[k] = testing_fma(
                                        bsr_val[bsr_dim * bsr_dim * j + bsr_dim * bi + (bj + k)],
                                        x[bsr_dim * col + (bj + k)],
                                        sum[k]);
                                }
                            }
                        }
                    }
                }

                for(int j = 1; j < WFSIZE; j <<= 1)
                {
                    for(int k = 0; k < WFSIZE - j; ++k)
                    {
                        sum[k] = sum[k] + sum[k + j];
                    }
                }

                if(beta != make_DataType<T>(0))
                {
                    y[row * bsr_dim + bi]
                        = testing_fma(beta, y[row * bsr_dim + bi], testing_mult(alpha, sum[0]));
                }
                else
                {
                    y[row * bsr_dim + bi] = testing_mult(alpha, sum[0]);
                }
            }
        }
    }
}

template <typename I, typename J, typename T>
inline void host_csrmv(hipsparseOperation_t trans,
                       J                    M,
                       J                    N,
                       I                    nnz,
                       T                    alpha,
                       const I*             csr_row_ptr,
                       const J*             csr_col_ind,
                       const T*             csr_val,
                       const T*             x,
                       T                    beta,
                       T*                   y,
                       hipsparseIndexBase_t base)
{
    if(trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        // Get device properties
        int             dev;
        hipDeviceProp_t prop;

        std::ignore = hipGetDevice(&dev);
        std::ignore = hipGetDeviceProperties(&prop, dev);

        int WF_SIZE;
        J   nnz_per_row = (M == 0) ? 0 : (nnz / M);

        if(nnz_per_row < 4)
            WF_SIZE = 2;
        else if(nnz_per_row < 8)
            WF_SIZE = 4;
        else if(nnz_per_row < 16)
            WF_SIZE = 8;
        else if(nnz_per_row < 32)
            WF_SIZE = 16;
        else if(nnz_per_row < 64 || prop.warpSize == 32)
            WF_SIZE = 32;
        else
            WF_SIZE = 64;

        for(J i = 0; i < M; ++i)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;

            std::vector<T> sum(WF_SIZE, static_cast<T>(0));

            for(I j = row_begin; j < row_end; j += WF_SIZE)
            {
                for(int k = 0; k < WF_SIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        sum[k] = testing_fma(testing_mult(alpha, csr_val[j + k]),
                                             x[csr_col_ind[j + k] - base],
                                             sum[k]);
                    }
                }
            }

            for(int j = 1; j < WF_SIZE; j <<= 1)
            {
                for(int k = 0; k < WF_SIZE - j; ++k)
                {
                    sum[k] = sum[k] + sum[k + j];
                }
            }

            if(beta == make_DataType<T>(0.0))
            {
                y[i] = sum[0];
            }
            else
            {
                y[i] = testing_fma(beta, y[i], sum[0]);
            }
        }
    }
    else
    {
        // Scale y with beta
        for(J i = 0; i < N; ++i)
        {
            y[i] = testing_mult(y[i], beta);
        }

        // Transposed SpMV
        for(J i = 0; i < M; ++i)
        {
            I row_begin = csr_row_ptr[i] - base;
            I row_end   = csr_row_ptr[i + 1] - base;
            T row_val   = testing_mult(alpha, x[i]);

            for(I j = row_begin; j < row_end; ++j)
            {
                J col = csr_col_ind[j] - base;
                T val = (trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                            ? testing_conj(csr_val[j])
                            : csr_val[j];

                y[col] = testing_fma(val, row_val, y[col]);
            }
        }
    }
}

template <typename T>
inline void host_bsrmm(int                     Mb,
                       int                     N,
                       int                     Kb,
                       int                     block_dim,
                       hipsparseDirection_t    dir,
                       hipsparseOperation_t    transA,
                       hipsparseOperation_t    transB,
                       T                       alpha,
                       const std::vector<int>& bsr_row_ptr_A,
                       const std::vector<int>& bsr_col_ind_A,
                       const std::vector<T>&   bsr_val_A,
                       const std::vector<T>&   B,
                       int                     ldb,
                       T                       beta,
                       std::vector<T>&         C,
                       int                     ldc,
                       hipsparseIndexBase_t    base)
{
    if(transA != HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        return;
    }

    if(transB != HIPSPARSE_OPERATION_NON_TRANSPOSE && transB != HIPSPARSE_OPERATION_TRANSPOSE)
    {
        return;
    }

    int M = Mb * block_dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; i++)
    {
        int local_row = i % block_dim;

        int row_begin = bsr_row_ptr_A[i / block_dim] - base;
        int row_end   = bsr_row_ptr_A[i / block_dim + 1] - base;

        for(int j = 0; j < N; j++)
        {
            int idx_C = i + j * ldc;

            T sum = make_DataType<T>(0.0);

            for(int s = row_begin; s < row_end; s++)
            {
                for(int t = 0; t < block_dim; t++)
                {
                    int idx_A = (dir == HIPSPARSE_DIRECTION_ROW)
                                    ? block_dim * block_dim * s + block_dim * local_row + t
                                    : block_dim * block_dim * s + block_dim * t + local_row;
                    int idx_B = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                                    ? j * ldb + block_dim * (bsr_col_ind_A[s] - base) + t
                                    : (block_dim * (bsr_col_ind_A[s] - base) + t) * ldb + j;

                    sum = sum + testing_mult(alpha, testing_mult(bsr_val_A[idx_A], B[idx_B]));
                }
            }

            if(beta == make_DataType<T>(0.0))
            {
                C[idx_C] = sum;
            }
            else
            {
                C[idx_C] = sum + testing_mult(beta, C[idx_C]);
            }
        }
    }
}

template <typename I, typename J, typename T>
void host_csrmm(J                    M,
                J                    N,
                J                    K,
                hipsparseOperation_t transA,
                hipsparseOperation_t transB,
                T                    alpha,
                const I*             csr_row_ptr_A,
                const J*             csr_col_ind_A,
                const T*             csr_val_A,
                const T*             B,
                J                    ldb,
                hipsparseOrder_t     orderB,
                T                    beta,
                T*                   C,
                J                    ldc,
                hipsparseOrder_t     orderC,
                hipsparseIndexBase_t base,
                bool                 force_conj_A)
{
    bool conj_A = (transA == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE || force_conj_A);
    bool conj_B = (transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE);

    if(transA == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(J i = 0; i < M; i++)
        {
            for(J j = 0; j < N; ++j)
            {
                I row_begin = csr_row_ptr_A[i] - base;
                I row_end   = csr_row_ptr_A[i + 1] - base;
                J idx_C     = orderC == HIPSPARSE_ORDER_COL ? i + j * ldc : i * ldc + j;

                T sum = make_DataType<T>(0);

                for(I k = row_begin; k < row_end; ++k)
                {
                    J idx_B = 0;
                    if((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE
                        && orderB == HIPSPARSE_ORDER_COL)
                       || (transB == HIPSPARSE_OPERATION_TRANSPOSE && orderB != HIPSPARSE_ORDER_COL)
                       || (transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE
                           && orderB != HIPSPARSE_ORDER_COL))
                    {
                        idx_B = (csr_col_ind_A[k] - base + j * ldb);
                    }
                    else
                    {
                        idx_B = (j + (csr_col_ind_A[k] - base) * ldb);
                    }

                    sum = testing_fma(
                        testing_conj(csr_val_A[k], conj_A), testing_conj(B[idx_B], conj_B), sum);
                }

                if(beta == make_DataType<T>(0))
                {
                    C[idx_C] = testing_mult(alpha, sum);
                }
                else
                {
                    C[idx_C] = testing_fma(beta, C[idx_C], testing_mult(alpha, sum));
                }
            }
        }
    }
    else
    {
        // scale C by beta
        for(J i = 0; i < K; i++)
        {
            for(J j = 0; j < N; ++j)
            {
                J idx_C  = (orderC == HIPSPARSE_ORDER_COL) ? i + j * ldc : i * ldc + j;
                C[idx_C] = testing_mult(beta, C[idx_C]);
            }
        }

        for(J i = 0; i < M; i++)
        {
            I row_begin = csr_row_ptr_A[i] - base;
            I row_end   = csr_row_ptr_A[i + 1] - base;

            for(J j = 0; j < N; ++j)
            {
                for(I k = row_begin; k < row_end; ++k)
                {
                    J col = csr_col_ind_A[k] - base;
                    T val = testing_conj(csr_val_A[k], conj_A);

                    J idx_B = 0;

                    if((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE
                        && orderB == HIPSPARSE_ORDER_COL)
                       || (transB == HIPSPARSE_OPERATION_TRANSPOSE && orderB != HIPSPARSE_ORDER_COL)
                       || (transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE
                           && orderB != HIPSPARSE_ORDER_COL))
                    {
                        idx_B = (i + j * ldb);
                    }
                    else
                    {
                        idx_B = (j + i * ldb);
                    }

                    J idx_C = (orderC == HIPSPARSE_ORDER_COL) ? col + j * ldc : col * ldc + j;

                    C[idx_C]
                        = C[idx_C]
                          + testing_mult(alpha, testing_mult(val, testing_conj(B[idx_B], conj_B)));
                }
            }
        }
    }
}

template <typename T, typename I, typename J>
void host_csrmm_batched(J                    M,
                        J                    N,
                        J                    K,
                        J                    batch_count_A,
                        J                    offsets_batch_stride_A,
                        I                    columns_values_batch_stride_A,
                        hipsparseOperation_t transA,
                        hipsparseOperation_t transB,
                        T                    alpha,
                        const I*             csr_row_ptr_A,
                        const J*             csr_col_ind_A,
                        const T*             csr_val_A,
                        const T*             B,
                        J                    ldb,
                        J                    batch_count_B,
                        I                    batch_stride_B,
                        hipsparseOrder_t     order_B,
                        T                    beta,
                        T*                   C,
                        J                    ldc,
                        J                    batch_count_C,
                        I                    batch_stride_C,
                        hipsparseOrder_t     order_C,
                        hipsparseIndexBase_t base,
                        bool                 force_conj_A)
{
    bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return;
    }

    if(Ci_A_Bi)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A,
                       csr_col_ind_A,
                       csr_val_A,
                       B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
    else if(Ci_Ai_B)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A + offsets_batch_stride_A * i,
                       csr_col_ind_A + columns_values_batch_stride_A * i,
                       csr_val_A + columns_values_batch_stride_A * i,
                       B,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
    else if(Ci_Ai_Bi)
    {
        for(J i = 0; i < batch_count_C; i++)
        {
            host_csrmm(M,
                       N,
                       K,
                       transA,
                       transB,
                       alpha,
                       csr_row_ptr_A + offsets_batch_stride_A * i,
                       csr_col_ind_A + columns_values_batch_stride_A * i,
                       csr_val_A + columns_values_batch_stride_A * i,
                       B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base,
                       force_conj_A);
        }
    }
}

template <typename T, typename I, typename J>
void host_cscmm(J                    M,
                J                    N,
                J                    K,
                hipsparseOperation_t transA,
                hipsparseOperation_t transB,
                T                    alpha,
                const I*             csc_col_ptr_A,
                const J*             csc_row_ind_A,
                const T*             csc_val_A,
                const T*             B,
                J                    ldb,
                hipsparseOrder_t     order_B,
                T                    beta,
                T*                   C,
                J                    ldc,
                hipsparseOrder_t     order_C,
                hipsparseIndexBase_t base)
{
    switch(transA)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
    {
        return host_csrmm(K,
                          N,
                          M,
                          HIPSPARSE_OPERATION_TRANSPOSE,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          B,
                          ldb,
                          order_B,
                          beta,
                          C,
                          ldc,
                          order_C,
                          base,
                          false);
    }
    case HIPSPARSE_OPERATION_TRANSPOSE:
    {
        return host_csrmm(K,
                          N,
                          M,
                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          B,
                          ldb,
                          order_B,
                          beta,
                          C,
                          ldc,
                          order_C,
                          base,
                          false);
    }
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
    {
        return host_csrmm(K,
                          N,
                          M,
                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          transB,
                          alpha,
                          csc_col_ptr_A,
                          csc_row_ind_A,
                          csc_val_A,
                          B,
                          ldb,
                          order_B,
                          beta,
                          C,
                          ldc,
                          order_C,
                          base,
                          true);
    }
    }
}

template <typename T, typename I, typename J>
void host_cscmm_batched(J                    M,
                        J                    N,
                        J                    K,
                        J                    batch_count_A,
                        I                    offsets_batch_stride_A,
                        I                    rows_values_batch_stride_A,
                        hipsparseOperation_t transA,
                        hipsparseOperation_t transB,
                        T                    alpha,
                        const I*             csc_col_ptr_A,
                        const J*             csc_row_ind_A,
                        const T*             csc_val_A,
                        const T*             B,
                        J                    ldb,
                        J                    batch_count_B,
                        I                    batch_stride_B,
                        hipsparseOrder_t     order_B,
                        T                    beta,
                        T*                   C,
                        J                    ldc,
                        J                    batch_count_C,
                        I                    batch_stride_C,
                        hipsparseOrder_t     order_C,
                        hipsparseIndexBase_t base)
{
    switch(transA)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  HIPSPARSE_OPERATION_TRANSPOSE,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  false);
    }
    case HIPSPARSE_OPERATION_TRANSPOSE:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  false);
    }
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
    {
        return host_csrmm_batched(K,
                                  N,
                                  M,
                                  batch_count_A,
                                  offsets_batch_stride_A,
                                  rows_values_batch_stride_A,
                                  HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                  transB,
                                  alpha,
                                  csc_col_ptr_A,
                                  csc_row_ind_A,
                                  csc_val_A,
                                  B,
                                  ldb,
                                  batch_count_B,
                                  batch_stride_B,
                                  order_B,
                                  beta,
                                  C,
                                  ldc,
                                  batch_count_C,
                                  batch_stride_C,
                                  order_C,
                                  base,
                                  true);
    }
    }
}

template <typename T, typename I>
void host_coomm(I                    M,
                I                    N,
                I                    K,
                I                    nnz,
                hipsparseOperation_t transA,
                hipsparseOperation_t transB,
                T                    alpha,
                const I*             coo_row_ind_A,
                const I*             coo_col_ind_A,
                const T*             coo_val_A,
                const T*             B,
                I                    ldb,
                hipsparseOrder_t     order_B,
                T                    beta,
                T*                   C,
                I                    ldc,
                hipsparseOrder_t     order_C,
                hipsparseIndexBase_t base)
{
    bool conj_A = (transA == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
    bool conj_B = (transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE);

    if(transA == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        for(I j = 0; j < N; j++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(I i = 0; i < M; ++i)
            {
                I idx_C = (order_C == HIPSPARSE_ORDER_COL) ? i + j * ldc : i * ldc + j;

                C[idx_C] = testing_mult(beta, C[idx_C]);
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I j = 0; j < N; j++)
        {
            for(I i = 0; i < nnz; ++i)
            {
                I row = coo_row_ind_A[i] - base;
                I col = coo_col_ind_A[i] - base;
                T val = testing_mult(alpha, coo_val_A[i]);

                I idx_C = (order_C == HIPSPARSE_ORDER_COL) ? row + j * ldc : row * ldc + j;

                I idx_B = 0;
                if((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE && order_B == HIPSPARSE_ORDER_COL)
                   || (transB != HIPSPARSE_OPERATION_NON_TRANSPOSE
                       && order_B != HIPSPARSE_ORDER_COL))
                {
                    idx_B = (col + j * ldb);
                }
                else
                {
                    idx_B = (j + col * ldb);
                }

                C[idx_C] = testing_fma(val, testing_conj(B[idx_B], conj_B), C[idx_C]);
            }
        }
    }
    else
    {
        for(I j = 0; j < N; j++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(I i = 0; i < K; ++i)
            {
                I idx_C = (order_C == HIPSPARSE_ORDER_COL) ? i + j * ldc : i * ldc + j;

                C[idx_C] = testing_mult(beta, C[idx_C]);
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(I j = 0; j < N; j++)
        {
            for(I i = 0; i < nnz; ++i)
            {
                I row = coo_row_ind_A[i] - base;
                I col = coo_col_ind_A[i] - base;
                T val = testing_mult(alpha, testing_conj(coo_val_A[i], conj_A));

                I idx_C = (order_C == HIPSPARSE_ORDER_COL) ? col + j * ldc : col * ldc + j;

                I idx_B = 0;
                if((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE && order_B == HIPSPARSE_ORDER_COL)
                   || (transB != HIPSPARSE_OPERATION_NON_TRANSPOSE
                       && order_B != HIPSPARSE_ORDER_COL))
                {
                    idx_B = (row + j * ldb);
                }
                else
                {
                    idx_B = (j + row * ldb);
                }

                C[idx_C] = testing_fma(val, testing_conj(B[idx_B], conj_B), C[idx_C]);
            }
        }
    }
}

template <typename T, typename I>
void host_coomm_batched(I                    M,
                        I                    N,
                        I                    K,
                        I                    nnz,
                        I                    batch_count_A,
                        I                    batch_stride_A,
                        hipsparseOperation_t transA,
                        hipsparseOperation_t transB,
                        T                    alpha,
                        const I*             coo_row_ind_A,
                        const I*             coo_col_ind_A,
                        const T*             coo_val_A,
                        const T*             B,
                        I                    ldb,
                        I                    batch_count_B,
                        I                    batch_stride_B,
                        hipsparseOrder_t     order_B,
                        T                    beta,
                        T*                   C,
                        I                    ldc,
                        I                    batch_count_C,
                        I                    batch_stride_C,
                        hipsparseOrder_t     order_C,
                        hipsparseIndexBase_t base)
{
    bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return;
    }

    if(Ci_A_Bi)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A,
                       coo_col_ind_A,
                       coo_val_A,
                       B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
    else if(Ci_Ai_B)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A + batch_stride_A * i,
                       coo_col_ind_A + batch_stride_A * i,
                       coo_val_A + batch_stride_A * i,
                       B,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
    else if(Ci_Ai_Bi)
    {
        for(I i = 0; i < batch_count_C; i++)
        {
            host_coomm(M,
                       N,
                       K,
                       nnz,
                       transA,
                       transB,
                       alpha,
                       coo_row_ind_A + batch_stride_A * i,
                       coo_col_ind_A + batch_stride_A * i,
                       coo_val_A + batch_stride_A * i,
                       B + batch_stride_B * i,
                       ldb,
                       order_B,
                       beta,
                       C + batch_stride_C * i,
                       ldc,
                       order_C,
                       base);
        }
    }
}

template <typename T>
int csrilu0(int                  m,
            const int*           ptr,
            const int*           col,
            T*                   val,
            hipsparseIndexBase_t idx_base,
            bool                 boost,
            double               boost_tol,
            T                    boost_val)
{
    // pointer of upper part of each row
    std::vector<int> diag_offset(m);
    std::vector<int> nnz_entries(m, 0);

    // ai = 0 to N loop over all rows
    for(int ai = 0; ai < m; ++ai)
    {
        // ai-th row entries
        int row_start = ptr[ai] - idx_base;
        int row_end   = ptr[ai + 1] - idx_base;
        int j;

        // nnz position of ai-th row in val array
        for(j = row_start; j < row_end; ++j)
        {
            nnz_entries[col[j] - idx_base] = j;
        }

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_start; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if(col[j] - idx_base < ai)
            {
                int col_j  = col[j] - idx_base;
                int diag_j = diag_offset[col_j];

                T diag_val = val[diag_j];

                if(boost)
                {
                    diag_val    = (boost_tol >= testing_abs(diag_val)) ? boost_val : diag_val;
                    val[diag_j] = diag_val;
                }
                else
                {
                    // Check for numeric pivot
                    if(diag_val == make_DataType<T>(0.0))
                    {
                        // Numerical zero diagonal
                        return col_j + idx_base;
                    }
                }

                // multiplication factor
                val[j] = testing_div(val[j], diag_val);

                // loop over upper offset pointer and do linear combination for nnz entry
                for(int k = diag_j + 1; k < ptr[col_j + 1] - idx_base; ++k)
                {
                    // if nnz at this position do linear combination
                    if(nnz_entries[col[k] - idx_base] != 0)
                    {
                        int idx  = nnz_entries[col[k] - idx_base];
                        val[idx] = testing_fma(testing_neg(val[j]), val[k], val[idx]);
                    }
                }
            }
            else if(col[j] - idx_base == ai)
            {
                has_diag = true;
                break;
            }
            else
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural zero digonal
            return ai + idx_base;
        }

        // set diagonal pointer to diagonal element
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_start; j < row_end; ++j)
        {
            nnz_entries[col[j] - idx_base] = 0;
        }
    }

    return -1;
}

template <typename T>
inline void host_bsrilu02(hipsparseDirection_t    dir,
                          int                     mb,
                          int                     bsr_dim,
                          const std::vector<int>& bsr_row_ptr,
                          const std::vector<int>& bsr_col_ind,
                          std::vector<T>&         bsr_val,
                          hipsparseIndexBase_t    base,
                          int*                    struct_pivot,
                          int*                    numeric_pivot,
                          bool                    boost,
                          double                  boost_tol,
                          T                       boost_val)
{
    // Initialize pivots
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    // Temporary vector to hold diagonal offset to access diagonal BSR block
    std::vector<int> diag_offset(mb);
    std::vector<int> nnz_entries(mb, -1);

    // First diagonal block is index 0
    if(mb > 0)
    {
        diag_offset[0] = 0;
    }

    // Loop over all BSR rows
    for(int i = 0; i < mb; ++i)
    {
        // Flag whether we have a diagonal block or not
        bool has_diag = false;

        // BSR column entry and exit point
        int row_begin = bsr_row_ptr[i] - base;
        int row_end   = bsr_row_ptr[i + 1] - base;

        int j;

        // Set up entry points for linear combination
        for(j = row_begin; j < row_end; ++j)
        {
            int col_j          = bsr_col_ind[j] - base;
            nnz_entries[col_j] = j;
        }

        // Process lower diagonal BSR blocks (diagonal BSR block is excluded)
        for(j = row_begin; j < row_end; ++j)
        {
            // Column index of current BSR block
            int bsr_col = bsr_col_ind[j] - base;

            // If this is a diagonal block, set diagonal flag to true and skip
            // all upcoming blocks as we exceed the lower matrix part
            if(bsr_col == i)
            {
                has_diag = true;
                break;
            }

            // Skip all upper matrix blocks
            if(bsr_col > i)
            {
                break;
            }

            // Process all lower matrix BSR blocks

            // Obtain corresponding row entry and exit point that corresponds with the
            // current BSR column. Actually, we skip all lower matrix column indices,
            // therefore starting with the diagonal entry.
            int diag_j    = diag_offset[bsr_col];
            int row_end_j = bsr_row_ptr[bsr_col + 1] - base;

            // Loop through all rows within the BSR block
            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                T diag = bsr_val[BSR_IND(diag_j, bi, bi, dir)];

                // Process all rows within the BSR block
                for(int bk = 0; bk < bsr_dim; ++bk)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // Multiplication factor
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val = testing_div(val, diag);

                    // Loop through columns of bk-th row and do linear combination
                    for(int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = testing_fma(-val,
                                          bsr_val[BSR_IND(diag_j, bi, bj, dir)],
                                          bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }

            // Loop over upper offset pointer and do linear combination for nnz entry
            for(int k = diag_j + 1; k < row_end_j; ++k)
            {
                int bsr_col_k = bsr_col_ind[k] - base;

                if(nnz_entries[bsr_col_k] != -1)
                {
                    int m = nnz_entries[bsr_col_k];

                    // Loop through all rows within the BSR block
                    for(int bi = 0; bi < bsr_dim; ++bi)
                    {
                        // Loop through columns of bi-th row and do linear combination
                        for(int bj = 0; bj < bsr_dim; ++bj)
                        {
                            T sum = make_DataType<T>(0);

                            for(int bk = 0; bk < bsr_dim; ++bk)
                            {
                                sum = testing_fma(bsr_val[BSR_IND(j, bi, bk, dir)],
                                                  bsr_val[BSR_IND(k, bk, bj, dir)],
                                                  sum);
                            }

                            bsr_val[BSR_IND(m, bi, bj, dir)]
                                = bsr_val[BSR_IND(m, bi, bj, dir)] - sum;
                        }
                    }
                }
            }
        }

        // Check for structural pivot
        if(!has_diag)
        {
            *struct_pivot = std::min(*struct_pivot, i + base);
            break;
        }

        // Process diagonal
        if(bsr_col_ind[j] - base == i)
        {
            // Loop through all rows within the BSR block
            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                T diag = bsr_val[BSR_IND(j, bi, bi, dir)];

                if(boost)
                {
                    diag = (boost_tol >= testing_abs(diag)) ? boost_val : diag;
                    bsr_val[BSR_IND(j, bi, bi, dir)] = diag;
                }
                else
                {
                    // Check for numeric pivot
                    if(diag == make_DataType<T>(0))
                    {
                        *numeric_pivot = std::min(*numeric_pivot, bsr_col_ind[j]);
                        continue;
                    }
                }

                // Process all rows within the BSR block after bi-th row
                for(int bk = bi + 1; bk < bsr_dim; ++bk)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // Multiplication factor
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val = testing_div(val, diag);

                    // Loop through remaining columns of bk-th row and do linear combination
                    for(int bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = testing_fma(-val,
                                          bsr_val[BSR_IND(j, bi, bj, dir)],
                                          bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }
        }

        // Store diagonal BSR block entry point
        int row_diag = diag_offset[i] = j;

        // Process upper diagonal BSR blocks
        for(j = row_diag + 1; j < row_end; ++j)
        {
            // Loop through all rows within the BSR block
            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                // Process all rows within the BSR block after bi-th row
                for(int bk = bi + 1; bk < bsr_dim; ++bk)
                {
                    // Loop through columns of bk-th row and do linear combination
                    for(int bj = 0; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = testing_fma(-bsr_val[BSR_IND(row_diag, bk, bi, dir)],
                                          bsr_val[BSR_IND(j, bi, bj, dir)],
                                          bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }
        }

        // Reset entry points
        for(j = row_begin; j < row_end; ++j)
        {
            int col_j          = bsr_col_ind[j] - base;
            nnz_entries[col_j] = -1;
        }
    }

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

template <typename T>
inline void host_bsric02(hipsparseDirection_t    direction,
                         int                     Mb,
                         int                     block_dim,
                         const std::vector<int>& bsr_row_ptr,
                         const std::vector<int>& bsr_col_ind,
                         std::vector<T>&         bsr_val,
                         hipsparseIndexBase_t    base,
                         int*                    struct_pivot,
                         int*                    numeric_pivot)
{
    int M = Mb * block_dim;

    // Initialize pivot
    *struct_pivot  = -1;
    *numeric_pivot = -1;

    if(bsr_col_ind.size() == 0 && bsr_val.size() == 0)
    {
        return;
    }

    // pointer of upper part of each row
    std::vector<int> diag_block_offset(Mb);
    std::vector<int> diag_offset(M, -1);
    std::vector<int> nnz_entries(M, -1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < Mb; i++)
    {
        int row_begin = bsr_row_ptr[i] - base;
        int row_end   = bsr_row_ptr[i + 1] - base;

        for(int j = row_begin; j < row_end; j++)
        {
            if(bsr_col_ind[j] - base == i)
            {
                diag_block_offset[i] = j;
                break;
            }
        }
    }

    for(int i = 0; i < M; i++)
    {
        int local_row = i % block_dim;

        int row_begin = bsr_row_ptr[i / block_dim] - base;
        int row_end   = bsr_row_ptr[i / block_dim + 1] - base;

        for(int j = row_begin; j < row_end; j++)
        {
            int block_col_j = bsr_col_ind[j] - base;

            for(int k = 0; k < block_dim; k++)
            {
                if(direction == HIPSPARSE_DIRECTION_ROW)
                {
                    nnz_entries[block_dim * block_col_j + k]
                        = block_dim * block_dim * j + block_dim * local_row + k;
                }
                else
                {
                    nnz_entries[block_dim * block_col_j + k]
                        = block_dim * block_dim * j + block_dim * k + local_row;
                }
            }
        }

        T   sum            = make_DataType<T>(0);
        int diag_val_index = -1;

        bool has_diag         = false;
        bool break_outer_loop = false;

        for(int j = row_begin; j < row_end; j++)
        {
            int block_col_j = bsr_col_ind[j] - base;

            for(int k = 0; k < block_dim; k++)
            {
                int col_j = block_dim * block_col_j + k;

                // Mark diagonal and skip row
                if(col_j == i)
                {
                    diag_val_index = block_dim * block_dim * j + block_dim * k + k;

                    has_diag         = true;
                    break_outer_loop = true;
                    break;
                }

                // Skip upper triangular
                if(col_j > i)
                {
                    break_outer_loop = true;
                    break;
                }

                T val_j;
                if(direction == HIPSPARSE_DIRECTION_ROW)
                {
                    val_j = bsr_val[block_dim * block_dim * j + block_dim * local_row + k];
                }
                else
                {
                    val_j = bsr_val[block_dim * block_dim * j + block_dim * k + local_row];
                }

                int local_row_j = col_j % block_dim;

                int row_begin_j = bsr_row_ptr[col_j / block_dim] - base;
                int row_end_j   = diag_block_offset[col_j / block_dim];
                int row_diag_j  = diag_offset[col_j];

                T local_sum = make_DataType<T>(0);
                T inv_diag  = row_diag_j != -1 ? bsr_val[row_diag_j] : make_DataType<T>(0);

                // Check for numeric zero
                if(inv_diag == make_DataType<T>(0))
                {
                    // Numerical non-invertible block diagonal
                    if(*numeric_pivot == -1)
                    {
                        *numeric_pivot = block_col_j + base;
                    }

                    *numeric_pivot = std::min(*numeric_pivot, block_col_j + base);

                    inv_diag = make_DataType<T>(1);
                }

                inv_diag = testing_div(make_DataType<T>(1), inv_diag);

                // loop over upper offset pointer and do linear combination for nnz entry
                for(int l = row_begin_j; l < row_end_j + 1; l++)
                {
                    int block_col_l = bsr_col_ind[l] - base;

                    for(int m = 0; m < block_dim; m++)
                    {
                        int idx = nnz_entries[block_dim * block_col_l + m];

                        if(idx != -1 && block_dim * block_col_l + m < col_j)
                        {
                            if(direction == HIPSPARSE_DIRECTION_ROW)
                            {
                                local_sum = testing_fma(bsr_val[block_dim * block_dim * l
                                                                + block_dim * local_row_j + m],
                                                        testing_conj(bsr_val[idx]),
                                                        local_sum);
                            }
                            else
                            {
                                local_sum = testing_fma(bsr_val[block_dim * block_dim * l
                                                                + block_dim * m + local_row_j],
                                                        testing_conj(bsr_val[idx]),
                                                        local_sum);
                            }
                        }
                    }
                }

                val_j = testing_mult((val_j - local_sum), inv_diag);
                sum   = testing_fma(val_j, testing_conj(val_j), sum);

                if(direction == HIPSPARSE_DIRECTION_ROW)
                {
                    bsr_val[block_dim * block_dim * j + block_dim * local_row + k] = val_j;
                }
                else
                {
                    bsr_val[block_dim * block_dim * j + block_dim * k + local_row] = val_j;
                }
            }

            if(break_outer_loop)
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural missing block diagonal
            if(*struct_pivot == -1)
            {
                *struct_pivot = i / block_dim + base;
            }
        }

        // Process diagonal entry
        if(has_diag)
        {
            T diag_entry = make_DataType<T>(std::sqrt(testing_abs(bsr_val[diag_val_index] - sum)));
            bsr_val[diag_val_index] = diag_entry;

            if(diag_entry == make_DataType<T>(0))
            {
                // Numerical non-invertible block diagonal
                if(*numeric_pivot == -1)
                {
                    *numeric_pivot = i / block_dim + base;
                }

                *numeric_pivot = std::min(*numeric_pivot, i / block_dim + base);
            }

            // Store diagonal offset
            diag_offset[i] = diag_val_index;
        }

        for(int j = row_begin; j < row_end; j++)
        {
            int block_col_j = bsr_col_ind[j] - base;

            for(int k = 0; k < block_dim; k++)
            {
                if(direction == HIPSPARSE_DIRECTION_ROW)
                {
                    nnz_entries[block_dim * block_col_j + k] = -1;
                }
                else
                {
                    nnz_entries[block_dim * block_col_j + k] = -1;
                }
            }
        }
    }
}

template <typename T>
void csric0(int                  M,
            const int*           csr_row_ptr,
            const int*           csr_col_ind,
            T*                   csr_val,
            hipsparseIndexBase_t idx_base,
            int&                 struct_pivot,
            int&                 numeric_pivot)
{
    // Initialize pivot
    struct_pivot  = -1;
    numeric_pivot = -1;

    // pointer of upper part of each row
    std::vector<int> diag_offset(M);
    std::vector<int> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(int ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        int row_begin = csr_row_ptr[ai] - idx_base;
        int row_end   = csr_row_ptr[ai + 1] - idx_base;
        int j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - idx_base] = j;
        }

        T sum = make_DataType<T>(0.0);

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            int col_j = csr_col_ind[j] - idx_base;
            T   val_j = csr_val[j];

            // Mark diagonal and skip row
            if(col_j == ai)
            {
                has_diag = true;
                break;
            }

            // Skip upper triangular
            if(col_j > ai)
            {
                break;
            }

            int row_begin_j = csr_row_ptr[col_j] - idx_base;
            int row_diag_j  = diag_offset[col_j];

            T local_sum = make_DataType<T>(0.0);
            T inv_diag  = csr_val[row_diag_j];

            // Check for numeric zero
            if(inv_diag == make_DataType<T>(0.0))
            {
                // Numerical zero diagonal
                numeric_pivot = col_j + idx_base;
                return;
            }

            inv_diag = testing_div(make_DataType<T>(1.0), inv_diag);

            // loop over upper offset pointer and do linear combination for nnz entry
            for(int k = row_begin_j; k < row_diag_j; ++k)
            {
                int col_k = csr_col_ind[k] - idx_base;

                // if nnz at this position do linear combination
                if(nnz_entries[col_k] != 0)
                {
                    int idx   = nnz_entries[col_k];
                    local_sum = testing_fma(csr_val[k], testing_conj(csr_val[idx]), local_sum);
                }
            }

            val_j = testing_mult((val_j - local_sum), inv_diag);
            sum   = testing_fma(val_j, testing_conj(val_j), sum);

            csr_val[j] = val_j;
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            struct_pivot  = ai + idx_base;
            numeric_pivot = ai + idx_base;
            return;
        }

        // Process diagonal entry
        T diag_entry = make_DataType<T>(std::sqrt(testing_abs(csr_val[j] - sum)));
        csr_val[j]   = diag_entry;

        // Store diagonal offset
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - idx_base] = 0;
        }
    }
}

/* ============================================================================================ */
/*! \brief  Sparse triangular system solve using CSR storage format. */
template <typename I, typename J, typename T>
static inline void host_lssolve(J                     M,
                                J                     nrhs,
                                hipsparseOperation_t  transB,
                                T                     alpha,
                                const std::vector<I>& csr_row_ptr,
                                const std::vector<J>& csr_col_ind,
                                const std::vector<T>& csr_val,
                                std::vector<T>&       B,
                                J                     ldb,
                                hipsparseOrder_t      order_B,
                                hipsparseDiagType_t   diag_type,
                                hipsparseIndexBase_t  base,
                                J*                    struct_pivot,
                                J*                    numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    std::ignore = hipGetDevice(&dev);
    std::ignore = hipGetDeviceProperties(&prop, dev);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < nrhs; ++i)
    {
        std::vector<T> temp(prop.warpSize);

        // Process lower triangular part
        for(J row = 0; row < M; ++row)
        {
            temp.assign(prop.warpSize, make_DataType<T>(0.0));

            J idx_B
                = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE && order_B == HIPSPARSE_ORDER_COL)
                      ? i * ldb + row
                      : row * ldb + i;

            if(transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                temp[0] = testing_mult(alpha, testing_conj(B[idx_B]));
            }
            else
            {
                temp[0] = testing_mult(alpha, B[idx_B]);
            }

            I diag      = -1;
            I row_begin = csr_row_ptr[row] - base;
            I row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = make_DataType<T>(0.0);

            for(I l = row_begin; l < row_end; l += prop.warpSize)
            {
                for(int k = 0; k < prop.warpSize; ++k)
                {
                    I j = l + k;

                    // Do not run out of bounds
                    if(j >= row_end)
                    {
                        break;
                    }

                    J local_col = csr_col_ind[j] - base;
                    T local_val = csr_val[j];

                    if(local_val == make_DataType<T>(0.0) && local_col == row
                       && diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                    {
                        // Numerical zero pivot found, avoid division by 0 and store
                        // index for later use
                        *numeric_pivot = std::min(*numeric_pivot, row + base);
                        local_val      = make_DataType<T>(1.0);
                    }

                    // Ignore all entries that are above the diagonal
                    if(local_col > row)
                    {
                        break;
                    }

                    // Diagonal entry
                    if(local_col == row)
                    {
                        // If diagonal type is non unit, do division by diagonal entry
                        // This is not required for unit diagonal for obvious reasons
                        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                        {
                            diag     = j;
                            diag_val = testing_div(make_DataType<T>(1.0), local_val);
                        }

                        break;
                    }

                    // Lower triangular part
                    J idx     = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE
                             && order_B == HIPSPARSE_ORDER_COL)
                                    ? i * ldb + local_col
                                    : local_col * ldb + i;
                    T neg_val = testing_mult(make_DataType<T>(-1.0), local_val);

                    if(transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                    {
                        temp[k] = testing_fma(neg_val, testing_conj(B[idx]), temp[k]);
                    }
                    else
                    {
                        temp[k] = testing_fma(neg_val, B[idx], temp[k]);
                    }
                }
            }

            for(int j = 1; j < prop.warpSize; j <<= 1)
            {
                for(int k = 0; k < prop.warpSize - j; ++k)
                {
                    temp[k] = temp[k] + temp[k + j];
                }
            }

            if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                B[idx_B] = testing_mult(temp[0], diag_val);
            }
            else
            {
                B[idx_B] = temp[0];
            }
        }
    }
}

template <typename I, typename J, typename T>
static inline void host_ussolve(J                     M,
                                J                     nrhs,
                                hipsparseOperation_t  transB,
                                T                     alpha,
                                const std::vector<I>& csr_row_ptr,
                                const std::vector<J>& csr_col_ind,
                                const std::vector<T>& csr_val,
                                std::vector<T>&       B,
                                J                     ldb,
                                hipsparseOrder_t      order_B,
                                hipsparseDiagType_t   diag_type,
                                hipsparseIndexBase_t  base,
                                J*                    struct_pivot,
                                J*                    numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    std::ignore = hipGetDevice(&dev);
    std::ignore = hipGetDeviceProperties(&prop, dev);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < nrhs; ++i)
    {
        std::vector<T> temp(prop.warpSize);

        // Process upper triangular part
        for(J row = M - 1; row >= 0; --row)
        {
            temp.assign(prop.warpSize, make_DataType<T>(0.0));

            J idx_B
                = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE && order_B == HIPSPARSE_ORDER_COL)
                      ? i * ldb + row
                      : row * ldb + i;

            if(transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                temp[0] = testing_mult(alpha, testing_conj(B[idx_B]));
            }
            else
            {
                temp[0] = testing_mult(alpha, B[idx_B]);
            }

            I diag      = -1;
            I row_begin = csr_row_ptr[row] - base;
            I row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = make_DataType<T>(0.0);

            for(I l = row_end - 1; l >= row_begin; l -= prop.warpSize)
            {
                for(int k = 0; k < prop.warpSize; ++k)
                {
                    I j = l - k;

                    // Do not run out of bounds
                    if(j < row_begin)
                    {
                        break;
                    }

                    J local_col = csr_col_ind[j] - base;
                    T local_val = csr_val[j];

                    // Ignore all entries that are below the diagonal
                    if(local_col < row)
                    {
                        continue;
                    }

                    // Diagonal entry
                    if(local_col == row)
                    {
                        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                        {
                            // Check for numerical zero
                            if(local_val == make_DataType<T>(0.0))
                            {
                                *numeric_pivot = std::min(*numeric_pivot, row + base);
                                local_val      = make_DataType<T>(1.0);
                            }

                            diag     = j;
                            diag_val = testing_div(make_DataType<T>(1.0), local_val);
                        }

                        continue;
                    }

                    // Upper triangular part
                    J idx = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE
                             && order_B == HIPSPARSE_ORDER_COL)
                                ? i * ldb + local_col
                                : local_col * ldb + i;

                    T neg_val = testing_mult(make_DataType<T>(-1.0), local_val);

                    if(transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                    {
                        temp[k] = testing_fma(neg_val, testing_conj(B[idx]), temp[k]);
                    }
                    else
                    {
                        temp[k] = testing_fma(neg_val, B[idx], temp[k]);
                    }
                }
            }

            for(int j = 1; j < prop.warpSize; j <<= 1)
            {
                for(int k = 0; k < prop.warpSize - j; ++k)
                {
                    temp[k] = temp[k] + temp[k + j];
                }
            }

            if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
            {
                if(diag == -1)
                {
                    *struct_pivot = std::min(*struct_pivot, row + base);
                }

                B[idx_B] = testing_mult(temp[0], diag_val);
            }
            else
            {
                B[idx_B] = temp[0];
            }
        }
    }
}

template <typename T>
void host_bsrsm(int                  mb,
                int                  nrhs,
                int                  nnzb,
                hipsparseDirection_t dir,
                hipsparseOperation_t transA,
                hipsparseOperation_t transX,
                T                    alpha,
                const int*           bsr_row_ptr,
                const int*           bsr_col_ind,
                const T*             bsr_val,
                int                  bsr_dim,
                const T*             B,
                int                  ldb,
                T*                   X,
                int                  ldx,
                hipsparseDiagType_t  diag_type,
                hipsparseFillMode_t  fill_mode,
                hipsparseIndexBase_t base,
                int*                 struct_pivot,
                int*                 numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    if(transA == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            bsr_lsolve(dir,
                       transX,
                       mb,
                       nrhs,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       bsr_dim,
                       B,
                       ldb,
                       X,
                       ldx,
                       diag_type,
                       base,
                       struct_pivot,
                       numeric_pivot);
        }
        else
        {
            bsr_usolve(dir,
                       transX,
                       mb,
                       nrhs,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       bsr_dim,
                       B,
                       ldb,
                       X,
                       ldx,
                       diag_type,
                       base,
                       struct_pivot,
                       numeric_pivot);
        }
    }
    else if(transA == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        // Transpose matrix
        std::vector<int> bsrt_row_ptr(mb + 1);
        std::vector<int> bsrt_col_ind(nnzb);
        std::vector<T>   bsrt_val(nnzb * bsr_dim * bsr_dim);

        host_bsr_to_bsc(mb,
                        mb,
                        nnzb,
                        bsr_dim,
                        bsr_row_ptr,
                        bsr_col_ind,
                        bsr_val,
                        bsrt_col_ind,
                        bsrt_row_ptr,
                        bsrt_val,
                        base,
                        base);

        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            bsr_usolve(dir,
                       transX,
                       mb,
                       nrhs,
                       alpha,
                       bsrt_row_ptr.data(),
                       bsrt_col_ind.data(),
                       bsrt_val.data(),
                       bsr_dim,
                       B,
                       ldb,
                       X,
                       ldx,
                       diag_type,
                       base,
                       struct_pivot,
                       numeric_pivot);
        }
        else
        {
            bsr_lsolve(dir,
                       transX,
                       mb,
                       nrhs,
                       alpha,
                       bsrt_row_ptr.data(),
                       bsrt_col_ind.data(),
                       bsrt_val.data(),
                       bsr_dim,
                       B,
                       ldb,
                       X,
                       ldx,
                       diag_type,
                       base,
                       struct_pivot,
                       numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename J, typename T>
void host_csr_lsolve(J                    M,
                     T                    alpha,
                     const I*             csr_row_ptr,
                     const J*             csr_col_ind,
                     const T*             csr_val,
                     const T*             x,
                     T*                   y,
                     hipsparseDiagType_t  diag_type,
                     hipsparseIndexBase_t base,
                     J*                   struct_pivot,
                     J*                   numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    std::ignore = hipGetDevice(&dev);
    std::ignore = hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    // Process lower triangular part
    for(J row = 0; row < M; ++row)
    {
        temp.assign(prop.warpSize, make_DataType<T>(0.0));
        temp[0] = testing_mult(alpha, x[row]);

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = make_DataType<T>(0.0);

        for(I l = row_begin; l < row_end; l += prop.warpSize)
        {
            for(int k = 0; k < prop.warpSize; ++k)
            {
                I j = l + k;

                // Do not run out of bounds
                if(j >= row_end)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                if(local_val == make_DataType<T>(0.0) && local_col == row
                   && diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                {
                    // Numerical zero pivot found, avoid division by 0
                    // and store index for later use.
                    *numeric_pivot = std::min(*numeric_pivot, row + base);
                    local_val      = make_DataType<T>(1);
                }

                // Ignore all entries that are above the diagonal
                if(local_col > row)
                {
                    break;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    // This is not required for unit diagonal for obvious reasons
                    if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                    {
                        diag     = j;
                        diag_val = testing_div(make_DataType<T>(1), local_val);
                    }

                    break;
                }

                // Lower triangular part
                temp[k] = testing_fma(-local_val, y[local_col], temp[k]);
            }
        }

        for(int j = 1; j < prop.warpSize; j <<= 1)
        {
            for(int k = 0; k < prop.warpSize - j; ++k)
            {
                temp[k] = temp[k] + temp[k + j];
            }
        }

        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
        {
            if(diag == -1)
            {
                *struct_pivot = std::min(*struct_pivot, row + base);
            }

            y[row] = testing_mult(temp[0], diag_val);
        }
        else
        {
            y[row] = temp[0];
        }
    }
}

template <typename I, typename J, typename T>
void host_csr_usolve(J                    M,
                     T                    alpha,
                     const I*             csr_row_ptr,
                     const J*             csr_col_ind,
                     const T*             csr_val,
                     const T*             x,
                     T*                   y,
                     hipsparseDiagType_t  diag_type,
                     hipsparseIndexBase_t base,
                     J*                   struct_pivot,
                     J*                   numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    std::ignore = hipGetDevice(&dev);
    std::ignore = hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    // Process upper triangular part
    for(J row = M - 1; row >= 0; --row)
    {
        temp.assign(prop.warpSize, make_DataType<T>(0));
        temp[0] = testing_mult(alpha, x[row]);

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = make_DataType<T>(0);

        for(I l = row_end - 1; l >= row_begin; l -= prop.warpSize)
        {
            for(int k = 0; k < prop.warpSize; ++k)
            {
                I j = l - k;

                // Do not run out of bounds
                if(j < row_begin)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                // Ignore all entries that are below the diagonal
                if(local_col < row)
                {
                    continue;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                    {
                        // Check for numerical zero
                        if(local_val == make_DataType<T>(0))
                        {
                            *numeric_pivot = std::min(*numeric_pivot, row + base);
                            local_val      = make_DataType<T>(1);
                        }

                        diag     = j;
                        diag_val = testing_div(make_DataType<T>(1), local_val);
                    }

                    continue;
                }

                // Upper triangular part
                temp[k] = testing_fma(-local_val, y[local_col], temp[k]);
            }
        }

        for(int j = 1; j < prop.warpSize; j <<= 1)
        {
            for(int k = 0; k < prop.warpSize - j; ++k)
            {
                temp[k] = temp[k] + temp[k + j];
            }
        }

        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
        {
            if(diag == -1)
            {
                *struct_pivot = std::min(*struct_pivot, row + base);
            }

            y[row] = testing_mult(temp[0], diag_val);
        }
        else
        {
            y[row] = temp[0];
        }
    }
}

template <typename I, typename J, typename T>
void host_csrsv(hipsparseOperation_t trans,
                J                    M,
                I                    nnz,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const T*             csr_val,
                const T*             x,
                T*                   y,
                hipsparseDiagType_t  diag_type,
                hipsparseFillMode_t  fill_mode,
                hipsparseIndexBase_t base,
                J*                   struct_pivot,
                J*                   numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    if(trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            host_csr_lsolve(M,
                            alpha,
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_usolve(M,
                            alpha,
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }
    else if(trans == HIPSPARSE_OPERATION_TRANSPOSE
            || trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        // Transpose matrix
        std::vector<I> csrt_row_ptr(M + 1);
        std::vector<J> csrt_col_ind(nnz);
        std::vector<T> csrt_val(nnz);

        host_csr_to_csc(M,
                        M,
                        nnz,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        csrt_col_ind,
                        csrt_row_ptr,
                        csrt_val,
                        HIPSPARSE_ACTION_NUMERIC,
                        base);

        if(trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        {
            for(size_t i = 0; i < csrt_val.size(); i++)
            {
                csrt_val[i] = testing_conj(csrt_val[i]);
            }
        }

        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            host_csr_usolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_lsolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename T>
void host_coosv(hipsparseOperation_t  trans,
                I                     M,
                I                     nnz,
                T                     alpha,
                const std::vector<I>& coo_row_ind,
                const std::vector<I>& coo_col_ind,
                const std::vector<T>& coo_val,
                const std::vector<T>& x,
                std::vector<T>&       y,
                hipsparseDiagType_t   diag_type,
                hipsparseFillMode_t   fill_mode,
                hipsparseIndexBase_t  base,
                I*                    struct_pivot,
                I*                    numeric_pivot)
{
    std::vector<I> csr_row_ptr(M + 1);

    //host_coo_to_csr(M, coo_row_ind, csr_row_ptr, base);
    // coo2csr on host
    for(I i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr[coo_row_ind[i] + 1 - base];
    }

    csr_row_ptr[0] = base;
    for(I i = 0; i < M; ++i)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    host_csrsv(trans,
               M,
               nnz,
               alpha,
               csr_row_ptr.data(),
               coo_col_ind.data(),
               coo_val.data(),
               x.data(),
               y.data(),
               diag_type,
               fill_mode,
               base,
               struct_pivot,
               numeric_pivot);
}

template <typename I, typename J, typename T>
void host_csrsm2(J                     M,
                 J                     nrhs,
                 I                     nnz,
                 hipsparseOperation_t  transA,
                 hipsparseOperation_t  transB,
                 T                     alpha,
                 const std::vector<I>& csr_row_ptr,
                 const std::vector<J>& csr_col_ind,
                 const std::vector<T>& csr_val,
                 std::vector<T>&       B,
                 J                     ldb,
                 hipsparseOrder_t      order_B,
                 hipsparseDiagType_t   diag_type,
                 hipsparseFillMode_t   fill_mode,
                 hipsparseIndexBase_t  base,
                 J*                    struct_pivot,
                 J*                    numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    if(transA == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            host_lssolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val,
                         B,
                         ldb,
                         order_B,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
        else
        {
            host_ussolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val,
                         B,
                         ldb,
                         order_B,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
    }
    else if(transA == HIPSPARSE_OPERATION_TRANSPOSE
            || transA == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        // Transpose matrix
        std::vector<I> csrt_row_ptr(M + 1);
        std::vector<J> csrt_col_ind(nnz);
        std::vector<T> csrt_val(nnz);

        host_csr_to_csc(M,
                        M,
                        nnz,
                        csr_row_ptr.data(),
                        csr_col_ind.data(),
                        csr_val.data(),
                        csrt_col_ind,
                        csrt_row_ptr,
                        csrt_val,
                        HIPSPARSE_ACTION_NUMERIC,
                        base);

        if(transA == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        {
            for(size_t i = 0; i < csrt_val.size(); i++)
            {
                csrt_val[i] = testing_conj(csrt_val[i]);
            }
        }

        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            host_ussolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csrt_row_ptr,
                         csrt_col_ind,
                         csrt_val,
                         B,
                         ldb,
                         order_B,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
        else
        {
            host_lssolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csrt_row_ptr,
                         csrt_col_ind,
                         csrt_val,
                         B,
                         ldb,
                         order_B,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename J, typename T>
void host_csrsm(J                     M,
                J                     nrhs,
                I                     nnz,
                hipsparseOperation_t  trans_A,
                hipsparseOperation_t  trans_B,
                T                     alpha,
                const std::vector<I>& csr_row_ptr,
                const std::vector<J>& csr_col_ind,
                const std::vector<T>& csr_val,
                const std::vector<T>& B,
                J                     ldb,
                hipsparseOrder_t      order_B,
                std::vector<T>&       C,
                J                     ldc,
                hipsparseOrder_t      order_C,
                hipsparseDiagType_t   diag_type,
                hipsparseFillMode_t   fill_mode,
                hipsparseIndexBase_t  base,
                J*                    struct_pivot,
                J*                    numeric_pivot)
{
    J B_m = (trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? M : nrhs;
    J B_n = (trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? nrhs : M;
    J C_m = M;
    J C_n = nrhs;

    // Copy B to C
    if(order_B == HIPSPARSE_ORDER_COL)
    {
        if(trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            if(order_C == HIPSPARSE_ORDER_COL)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        C[i + ldc * j] = B[i + ldb * j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        C[i * ldc + j] = B[i + ldb * j];
                    }
                }
            }
        }
        else
        {
            if(order_C == HIPSPARSE_ORDER_COL)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        C[i * ldc + j] = B[i + ldb * j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        C[i + ldc * j] = B[i + ldb * j];
                    }
                }
            }
        }
    }
    else
    {
        if(trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            if(order_C == HIPSPARSE_ORDER_COL)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        C[i + ldc * j] = B[ldb * i + j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        C[i * ldc + j] = B[ldb * i + j];
                    }
                }
            }
        }
        else
        {
            if(order_C == HIPSPARSE_ORDER_COL)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        C[i * ldc + j] = B[ldb * i + j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        C[i + ldc * j] = B[ldb * i + j];
                    }
                }
            }
        }
    }

    if(trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        if(order_C == HIPSPARSE_ORDER_COL)
        {
            for(J j = 0; j < C_n; j++)
            {
                for(J i = 0; i < C_m; i++)
                {
                    C[i + ldc * j] = testing_conj(C[i + ldc * j]);
                }
            }
        }
        else
        {
            for(J i = 0; i < C_m; i++)
            {
                for(J j = 0; j < C_n; j++)
                {
                    C[ldc * i + j] = testing_conj(C[ldc * i + j]);
                }
            }
        }
    }

    hipsparseOperation_t trans_C = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    host_csrsm2(M,
                nrhs,
                nnz,
                trans_A,
                trans_C,
                alpha,
                csr_row_ptr,
                csr_col_ind,
                csr_val,
                C,
                ldc,
                order_C,
                diag_type,
                fill_mode,
                base,
                struct_pivot,
                numeric_pivot);
}

template <typename I, typename T>
void host_coosm(I                     M,
                I                     nrhs,
                I                     nnz,
                hipsparseOperation_t  transA,
                hipsparseOperation_t  transB,
                T                     alpha,
                const std::vector<I>& coo_row_ind,
                const std::vector<I>& coo_col_ind,
                const std::vector<T>& coo_val,
                const std::vector<T>& B,
                I                     ldb,
                hipsparseOrder_t      order_B,
                std::vector<T>&       C,
                I                     ldc,
                hipsparseOrder_t      order_C,
                hipsparseDiagType_t   diag_type,
                hipsparseFillMode_t   fill_mode,
                hipsparseIndexBase_t  base,
                I*                    struct_pivot,
                I*                    numeric_pivot)
{
    I B_m = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? M : nrhs;
    I B_n = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? nrhs : M;
    I C_m = M;
    I C_n = nrhs;

    // Copy B to C
    if(order_B == HIPSPARSE_ORDER_COL)
    {
        if(transB == HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            if(order_C == HIPSPARSE_ORDER_COL)
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        C[i + ldc * j] = B[i + ldb * j];
                    }
                }
            }
            else
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        C[i * ldc + j] = B[i + ldb * j];
                    }
                }
            }
        }
        else
        {
            if(order_C == HIPSPARSE_ORDER_COL)
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        C[i * ldc + j] = B[i + ldb * j];
                    }
                }
            }
            else
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        C[i + ldc * j] = B[i + ldb * j];
                    }
                }
            }
        }
    }
    else
    {
        if(transB == HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            if(order_C == HIPSPARSE_ORDER_COL)
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        C[i + ldc * j] = B[ldb * i + j];
                    }
                }
            }
            else
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        C[i * ldc + j] = B[ldb * i + j];
                    }
                }
            }
        }
        else
        {
            if(order_C == HIPSPARSE_ORDER_COL)
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        C[i * ldc + j] = B[ldb * i + j];
                    }
                }
            }
            else
            {
                for(I j = 0; j < B_n; j++)
                {
                    for(I i = 0; i < B_m; i++)
                    {
                        C[i + ldc * j] = B[ldb * i + j];
                    }
                }
            }
        }
    }

    if(transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        if(order_C == HIPSPARSE_ORDER_COL)
        {
            for(I j = 0; j < C_n; j++)
            {
                for(I i = 0; i < C_m; i++)
                {
                    C[i + ldc * j] = testing_conj(C[i + ldc * j]);
                }
            }
        }
        else
        {
            for(I i = 0; i < C_m; i++)
            {
                for(I j = 0; j < C_n; j++)
                {
                    C[ldc * i + j] = testing_conj(C[ldc * i + j]);
                }
            }
        }
    }

    hipsparseOperation_t transC = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    std::vector<I> csr_row_ptr(M + 1);

    //host_coo_to_csr(M, coo_row_ind, csr_row_ptr, base);
    // coo2csr on host
    for(I i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr[coo_row_ind[i] + 1 - base];
    }

    csr_row_ptr[0] = base;
    for(I i = 0; i < M; ++i)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    host_csrsm2(M,
                nrhs,
                nnz,
                transA,
                transC,
                alpha,
                csr_row_ptr,
                coo_col_ind,
                coo_val,
                C,
                ldc,
                order_C,
                diag_type,
                fill_mode,
                base,
                struct_pivot,
                numeric_pivot);
}

/* ============================================================================================ */
/*! \brief  Sparse triangular lower solve using BSR storage format. */
template <typename T>
void bsr_lsolve(hipsparseDirection_t dir,
                hipsparseOperation_t trans_X,
                int                  mb,
                int                  nrhs,
                T                    alpha,
                const int*           bsr_row_ptr,
                const int*           bsr_col_ind,
                const T*             bsr_val,
                int                  bsr_dim,
                const T*             B,
                int                  ldb,
                T*                   X,
                int                  ldx,
                hipsparseDiagType_t  diag_type,
                hipsparseIndexBase_t base,
                int*                 struct_pivot,
                int*                 numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < nrhs; ++i)
    {
        // Process lower triangular part
        for(int bsr_row = 0; bsr_row < mb; ++bsr_row)
        {
            int bsr_row_begin = bsr_row_ptr[bsr_row] - base;
            int bsr_row_end   = bsr_row_ptr[bsr_row + 1] - base;

            // Loop over blocks rows
            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                int diag      = -1;
                int local_row = bsr_row * bsr_dim + bi;

                int idx_B = (trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? i * ldb + local_row
                                                                           : local_row * ldb + i;
                int idx_X = (trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? i * ldx + local_row
                                                                           : local_row * ldx + i;

                T sum      = testing_mult(alpha, B[idx_B]);
                T diag_val = make_DataType<T>(0);

                // Loop over BSR columns
                for(int j = bsr_row_begin; j < bsr_row_end; ++j)
                {
                    int bsr_col = bsr_col_ind[j] - base;

                    // Loop over blocks columns
                    for(int bj = 0; bj < bsr_dim; ++bj)
                    {
                        int local_col = bsr_col * bsr_dim + bj;
                        T   local_val = (dir == HIPSPARSE_DIRECTION_ROW)
                                            ? bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj]
                                            : bsr_val[bsr_dim * bsr_dim * j + bi + bj * bsr_dim];

                        if(local_val == make_DataType<T>(0) && local_col == local_row
                           && diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                        {
                            // Numerical zero pivot found, avoid division by 0
                            // and store index for later use.
                            *numeric_pivot = std::min(*numeric_pivot, bsr_row + base);
                            local_val      = make_DataType<T>(1);
                        }

                        // Ignore all entries that are above the diagonal
                        if(local_col > local_row)
                        {
                            break;
                        }

                        // Diagonal
                        if(local_col == local_row)
                        {
                            // If diagonal type is non unit, do division by diagonal entry
                            // This is not required for unit diagonal for obvious reasons
                            if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                            {
                                diag     = j;
                                diag_val = testing_div(make_DataType<T>(1), local_val);
                            }

                            break;
                        }

                        // Lower triangular part
                        int idx = (trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                                      ? i * ldx + local_col
                                      : local_col * ldx + i;
                        sum     = testing_fma(-local_val, X[idx], sum);
                    }
                }

                if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                {
                    if(diag == -1)
                    {
                        *struct_pivot = std::min(*struct_pivot, bsr_row + base);
                    }

                    X[idx_X] = testing_mult(sum, diag_val);
                }
                else
                {
                    X[idx_X] = sum;
                }
            }
        }
    }
}

/* ============================================================================================ */
/*! \brief  Sparse triangular upper solve using BSR storage format. */
template <typename T>
void bsr_usolve(hipsparseDirection_t dir,
                hipsparseOperation_t trans_X,
                int                  mb,
                int                  nrhs,
                T                    alpha,
                const int*           bsr_row_ptr,
                const int*           bsr_col_ind,
                const T*             bsr_val,
                int                  bsr_dim,
                const T*             B,
                int                  ldb,
                T*                   X,
                int                  ldx,
                hipsparseDiagType_t  diag_type,
                hipsparseIndexBase_t base,
                int*                 struct_pivot,
                int*                 numeric_pivot)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < nrhs; ++i)
    {
        // Process upper triangular part
        for(int bsr_row = mb - 1; bsr_row >= 0; --bsr_row)
        {
            int bsr_row_begin = bsr_row_ptr[bsr_row] - base;
            int bsr_row_end   = bsr_row_ptr[bsr_row + 1] - base;

            for(int bi = bsr_dim - 1; bi >= 0; --bi)
            {
                int local_row = bsr_row * bsr_dim + bi;

                int idx_B = (trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? i * ldb + local_row
                                                                           : local_row * ldb + i;
                int idx_X = (trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? i * ldx + local_row
                                                                           : local_row * ldx + i;
                T   sum   = testing_mult(alpha, B[idx_B]);

                int diag     = -1;
                T   diag_val = make_DataType<T>(0);

                for(int j = bsr_row_end - 1; j >= bsr_row_begin; --j)
                {
                    int bsr_col = bsr_col_ind[j] - base;

                    for(int bj = bsr_dim - 1; bj >= 0; --bj)
                    {
                        int local_col = bsr_col * bsr_dim + bj;
                        T   local_val = dir == HIPSPARSE_DIRECTION_ROW
                                            ? bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj]
                                            : bsr_val[bsr_dim * bsr_dim * j + bi + bj * bsr_dim];

                        // Ignore all entries that are below the diagonal
                        if(local_col < local_row)
                        {
                            continue;
                        }

                        // Diagonal
                        if(local_col == local_row)
                        {
                            if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                            {
                                // Check for numerical zero
                                if(local_val == make_DataType<T>(0))
                                {
                                    *numeric_pivot = std::min(*numeric_pivot, bsr_row + base);
                                    local_val      = make_DataType<T>(1);
                                }

                                diag     = j;
                                diag_val = testing_div(make_DataType<T>(1), local_val);
                            }

                            continue;
                        }

                        // Upper triangular part
                        int idx = (trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                                      ? i * ldx + local_col
                                      : local_col * ldx + i;
                        sum     = testing_fma(-local_val, X[idx], sum);
                    }
                }

                if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                {
                    if(diag == -1)
                    {
                        *struct_pivot = std::min(*struct_pivot, bsr_row + base);
                    }

                    X[idx_X] = testing_mult(sum, diag_val);
                }
                else
                {
                    X[idx_X] = sum;
                }
            }
        }
    }
}

template <typename T>
void bsrsv(hipsparseOperation_t trans,
           hipsparseDirection_t dir,
           int                  mb,
           int                  nnzb,
           T                    alpha,
           const int*           bsr_row_ptr,
           const int*           bsr_col_ind,
           const T*             bsr_val,
           int                  bsr_dim,
           const T*             x,
           T*                   y,
           hipsparseDiagType_t  diag_type,
           hipsparseFillMode_t  fill_mode,
           hipsparseIndexBase_t base,
           int*                 struct_pivot,
           int*                 numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    if(trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            bsr_lsolve(dir,
                       HIPSPARSE_OPERATION_NON_TRANSPOSE,
                       mb,
                       1,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       bsr_dim,
                       x,
                       mb * bsr_dim,
                       y,
                       mb * bsr_dim,
                       diag_type,
                       base,
                       struct_pivot,
                       numeric_pivot);
        }
        else
        {
            bsr_usolve(dir,
                       HIPSPARSE_OPERATION_NON_TRANSPOSE,
                       mb,
                       1,
                       alpha,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_val,
                       bsr_dim,
                       x,
                       mb * bsr_dim,
                       y,
                       mb * bsr_dim,
                       diag_type,
                       base,
                       struct_pivot,
                       numeric_pivot);
        }
    }
    else if(trans == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        // Transpose matrix
        std::vector<int> bsrt_row_ptr;
        std::vector<int> bsrt_col_ind;
        std::vector<T>   bsrt_val;

        host_bsr_to_bsc(mb,
                        mb,
                        nnzb,
                        bsr_dim,
                        bsr_row_ptr,
                        bsr_col_ind,
                        bsr_val,
                        bsrt_col_ind,
                        bsrt_row_ptr,
                        bsrt_val,
                        base,
                        base);

        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            bsr_usolve(dir,
                       HIPSPARSE_OPERATION_NON_TRANSPOSE,
                       mb,
                       1,
                       alpha,
                       bsrt_row_ptr.data(),
                       bsrt_col_ind.data(),
                       bsrt_val.data(),
                       bsr_dim,
                       x,
                       mb * bsr_dim,
                       y,
                       mb * bsr_dim,
                       diag_type,
                       base,
                       struct_pivot,
                       numeric_pivot);
        }
        else
        {
            bsr_lsolve(dir,
                       HIPSPARSE_OPERATION_NON_TRANSPOSE,
                       mb,
                       1,
                       alpha,
                       bsrt_row_ptr.data(),
                       bsrt_col_ind.data(),
                       bsrt_val.data(),
                       bsr_dim,
                       x,
                       mb * bsr_dim,
                       y,
                       mb * bsr_dim,
                       diag_type,
                       base,
                       struct_pivot,
                       numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;
}

/* ============================================================================================ */
/*! \brief  Sparse triangular lower solve using CSR storage format. */
template <typename T>
int csr_lsolve(hipsparseOperation_t trans,
               int                  m,
               const int*           ptr,
               const int*           col,
               const T*             val,
               T                    alpha,
               const T*             x,
               T*                   y,
               hipsparseIndexBase_t idx_base,
               hipsparseDiagType_t  diag_type,
               unsigned int         wf_size)
{
    const int* csr_row_ptr = ptr;
    const int* csr_col_ind = col;
    const T*   csr_val     = val;

    std::vector<int> vptr;
    std::vector<int> vcol;
    std::vector<T>   vval;

    if(trans == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        int nnz = ptr[m] - idx_base;

        vptr.resize(m + 1);
        vcol.resize(nnz);
        vval.resize(nnz);

        // Transpose
        transpose_csr(
            m, m, nnz, ptr, col, val, vptr.data(), vcol.data(), vval.data(), idx_base, idx_base);

        csr_row_ptr = vptr.data();
        csr_col_ind = vcol.data();
        csr_val     = vval.data();
    }

    int            pivot = (std::numeric_limits<int>::max)();
    std::vector<T> temp(wf_size);

    for(int i = 0; i < m; ++i)
    {
        temp.assign(wf_size, make_DataType<T>(0.0));
        temp[0] = testing_mult(alpha, x[i]);

        int diag      = -1;
        int row_begin = csr_row_ptr[i] - idx_base;
        int row_end   = csr_row_ptr[i + 1] - idx_base;

        T diag_val = make_DataType<T>(0.0);

        for(int l = row_begin; l < row_end; l += wf_size)
        {
            for(unsigned int k = 0; k < wf_size; ++k)
            {
                int j = l + k;

                // Do not run out of bounds
                if(j >= row_end)
                {
                    break;
                }

                int col_j = csr_col_ind[j] - idx_base;
                T   val_j = csr_val[j];

                if(col_j < i)
                {
                    // Lower part
                    temp[k] = testing_fma(-csr_val[j], y[col_j], temp[k]);
                }
                else if(col_j == i)
                {
                    // Diagonal
                    if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                    {
                        // Check for numerical zero
                        if(val_j == make_DataType<T>(0.0))
                        {
                            pivot = std::min(pivot, i + idx_base);
                            val_j = make_DataType<T>(1.0);
                        }

                        diag     = j;
                        diag_val = testing_div(make_DataType<T>(1.0), val_j);
                    }

                    break;
                }
                else
                {
                    // Upper part
                    break;
                }
            }
        }

        for(unsigned int j = 1; j < wf_size; j <<= 1)
        {
            for(unsigned int k = 0; k < wf_size - j; ++k)
            {
                temp[k] = temp[k] + temp[k + j];
            }
        }

        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
        {
            if(diag == -1)
            {
                pivot = std::min(pivot, i + idx_base);
            }

            y[i] = testing_mult(temp[0], diag_val);
        }
        else
        {
            y[i] = temp[0];
        }
    }

    if(pivot != (std::numeric_limits<int>::max)())
    {
        return pivot;
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Sparse triangular upper solve using CSR storage format. */
template <typename T>
int csr_usolve(hipsparseOperation_t trans,
               int                  m,
               const int*           ptr,
               const int*           col,
               const T*             val,
               T                    alpha,
               const T*             x,
               T*                   y,
               hipsparseIndexBase_t idx_base,
               hipsparseDiagType_t  diag_type,
               unsigned int         wf_size)
{
    const int* csr_row_ptr = ptr;
    const int* csr_col_ind = col;
    const T*   csr_val     = val;

    std::vector<int> vptr;
    std::vector<int> vcol;
    std::vector<T>   vval;

    if(trans == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        int nnz = ptr[m] - idx_base;

        vptr.resize(m + 1);
        vcol.resize(nnz);
        vval.resize(nnz);

        // Transpose
        transpose_csr(
            m, m, nnz, ptr, col, val, vptr.data(), vcol.data(), vval.data(), idx_base, idx_base);

        csr_row_ptr = vptr.data();
        csr_col_ind = vcol.data();
        csr_val     = vval.data();
    }

    int            pivot = (std::numeric_limits<int>::max)();
    std::vector<T> temp(wf_size);

    for(int i = m - 1; i >= 0; --i)
    {
        temp.assign(wf_size, make_DataType<T>(0.0));
        temp[0] = testing_mult(alpha, x[i]);

        int diag      = -1;
        int row_begin = csr_row_ptr[i] - idx_base;
        int row_end   = csr_row_ptr[i + 1] - idx_base;

        T diag_val = make_DataType<T>(0.0);

        for(int l = row_end - 1; l >= row_begin; l -= wf_size)
        {
            for(unsigned int k = 0; k < wf_size; ++k)
            {
                int j = l - k;

                // Do not run out of bounds
                if(j < row_begin)
                {
                    break;
                }

                int col_j = csr_col_ind[j] - idx_base;
                T   val_j = csr_val[j];

                if(col_j < i)
                {
                    // Lower part
                    continue;
                }
                else if(col_j == i)
                {
                    // Diagonal
                    if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                    {
                        // Check for numerical zero
                        if(val_j == make_DataType<T>(0.0))
                        {
                            pivot = std::min(pivot, i + idx_base);
                            val_j = make_DataType<T>(1.0);
                        }

                        diag     = j;
                        diag_val = testing_div(make_DataType<T>(1.0), val_j);
                    }

                    continue;
                }
                else
                {
                    // Upper part
                    temp[k] = testing_fma(-csr_val[j], y[col_j], temp[k]);
                }
            }
        }

        for(unsigned int j = 1; j < wf_size; j <<= 1)
        {
            for(unsigned int k = 0; k < wf_size - j; ++k)
            {
                temp[k] = temp[k] + temp[k + j];
            }
        }

        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
        {
            if(diag == -1)
            {
                pivot = std::min(pivot, i + idx_base);
            }

            y[i] = testing_mult(temp[0], diag_val);
        }
        else
        {
            y[i] = temp[0];
        }
    }

    if(pivot != (std::numeric_limits<int>::max)())
    {
        return pivot;
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Transpose sparse matrix using CSR storage format. */
template <typename I, typename J, typename T>
void transpose_csr(J                    m,
                   J                    n,
                   I                    nnz,
                   const I*             csr_row_ptr_A,
                   const J*             csr_col_ind_A,
                   const T*             csr_val_A,
                   I*                   csr_row_ptr_B,
                   J*                   csr_col_ind_B,
                   T*                   csr_val_B,
                   hipsparseIndexBase_t idx_base_A,
                   hipsparseIndexBase_t idx_base_B)
{
    memset(csr_row_ptr_B, 0, sizeof(I) * (n + 1));

    // Determine nnz per column
    for(I i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr_B[csr_col_ind_A[i] + 1 - idx_base_A];
    }

    // Scan
    for(J i = 0; i < n; ++i)
    {
        csr_row_ptr_B[i + 1] += csr_row_ptr_B[i];
    }

    // Fill row indices and values
    for(J i = 0; i < m; ++i)
    {
        I row_begin = csr_row_ptr_A[i] - idx_base_A;
        I row_end   = csr_row_ptr_A[i + 1] - idx_base_A;

        for(I j = row_begin; j < row_end; ++j)
        {
            J col = csr_col_ind_A[j] - idx_base_A;
            I idx = csr_row_ptr_B[col];

            csr_col_ind_B[idx] = i + idx_base_B;
            csr_val_B[idx]     = csr_val_A[j];

            ++csr_row_ptr_B[col];
        }
    }

    // Shift column pointer array
    for(J i = n; i > 0; --i)
    {
        csr_row_ptr_B[i] = csr_row_ptr_B[i - 1] + idx_base_B;
    }

    csr_row_ptr_B[0] = idx_base_B;
}

/* ============================================================================================ */
/*! \brief  Transpose sparse matrix using CSR storage format. */
template <typename T>
void transpose_bsr(int                  mb,
                   int                  nb,
                   int                  nnzb,
                   int                  bsr_dim,
                   const int*           bsr_row_ptr_A,
                   const int*           bsr_col_ind_A,
                   const T*             bsr_val_A,
                   int*                 bsr_row_ptr_B,
                   int*                 bsr_col_ind_B,
                   T*                   bsr_val_B,
                   hipsparseIndexBase_t idx_base_A,
                   hipsparseIndexBase_t idx_base_B)
{
    memset(bsr_row_ptr_B, 0, sizeof(int) * (nb + 1));

    // Determine nnz per column
    for(int i = 0; i < nnzb; ++i)
    {
        ++bsr_row_ptr_B[bsr_col_ind_A[i] + 1 - idx_base_A];
    }

    // Scan
    for(int i = 0; i < nb; ++i)
    {
        bsr_row_ptr_B[i + 1] += bsr_row_ptr_B[i];
    }

    // Fill row indices and values
    for(int i = 0; i < mb; ++i)
    {
        int row_begin = bsr_row_ptr_A[i] - idx_base_A;
        int row_end   = bsr_row_ptr_A[i + 1] - idx_base_A;

        for(int j = row_begin; j < row_end; ++j)
        {
            int col = bsr_col_ind_A[j] - idx_base_A;
            int idx = bsr_row_ptr_B[col];

            bsr_col_ind_B[idx] = i + idx_base_B;

            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                for(int bj = 0; bj < bsr_dim; ++bj)
                {
                    bsr_val_B[bsr_dim * bsr_dim * idx + bi + bj * bsr_dim]
                        = bsr_val_A[bsr_dim * bsr_dim * j + bi * bsr_dim + bj];
                }
            }

            ++bsr_row_ptr_B[col];
        }
    }

    // Shift column pointer array
    for(int i = nb; i > 0; --i)
    {
        bsr_row_ptr_B[i] = bsr_row_ptr_B[i - 1] + idx_base_B;
    }

    bsr_row_ptr_B[0] = idx_base_B;
}

/* ============================================================================================ */
/*! \brief  Compute sparse matrix sparse matrix addition. */
template <typename T>
static int host_csrgeam_nnz(int                  M,
                            int                  N,
                            T                    alpha,
                            const int*           csr_row_ptr_A,
                            const int*           csr_col_ind_A,
                            T                    beta,
                            const int*           csr_row_ptr_B,
                            const int*           csr_col_ind_B,
                            int*                 csr_row_ptr_C,
                            hipsparseIndexBase_t base_A,
                            hipsparseIndexBase_t base_B,
                            hipsparseIndexBase_t base_C)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<int> nnz(N, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int nthreads = 1;
        int tid      = 0;
#endif

        int rows_per_thread = (M + nthreads - 1) / nthreads;
        int chunk_begin     = rows_per_thread * tid;
        int chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Index base
        csr_row_ptr_C[0] = base_C;

        // Loop over rows
        for(int i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            int row_begin_A = csr_row_ptr_A[i] - base_A;
            int row_end_A   = csr_row_ptr_A[i + 1] - base_A;

            // Loop over columns of A
            for(int j = row_begin_A; j < row_end_A; ++j)
            {
                int col_A = csr_col_ind_A[j] - base_A;

                nnz[col_A] = i;
                ++csr_row_ptr_C[i + 1];
            }

            int row_begin_B = csr_row_ptr_B[i] - base_B;
            int row_end_B   = csr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(int j = row_begin_B; j < row_end_B; ++j)
            {
                int col_B = csr_col_ind_B[j] - base_B;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    ++csr_row_ptr_C[i + 1];
                }
            }
        }
    }

    // Scan to obtain row offsets
    for(int i = 0; i < M; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    return csr_row_ptr_C[M] - base_C;
}

template <typename T>
static void host_csrgeam(int                  M,
                         int                  N,
                         T                    alpha,
                         const int*           csr_row_ptr_A,
                         const int*           csr_col_ind_A,
                         const T*             csr_val_A,
                         T                    beta,
                         const int*           csr_row_ptr_B,
                         const int*           csr_col_ind_B,
                         const T*             csr_val_B,
                         const int*           csr_row_ptr_C,
                         int*                 csr_col_ind_C,
                         T*                   csr_val_C,
                         hipsparseIndexBase_t base_A,
                         hipsparseIndexBase_t base_B,
                         hipsparseIndexBase_t base_C)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<int> nnz(N, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int nthreads = 1;
        int tid      = 0;
#endif

        int rows_per_thread = (M + nthreads - 1) / nthreads;
        int chunk_begin     = rows_per_thread * tid;
        int chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Loop over rows
        for(int i = chunk_begin; i < chunk_end; ++i)
        {
            int row_begin_C = csr_row_ptr_C[i] - base_C;
            int row_end_C   = row_begin_C;

            int row_begin_A = csr_row_ptr_A[i] - base_A;
            int row_end_A   = csr_row_ptr_A[i + 1] - base_A;

            // Copy A into C
            for(int j = row_begin_A; j < row_end_A; ++j)
            {
                // Current column of A
                int col_A = csr_col_ind_A[j] - base_A;

                // Current value of A
                T val_A = testing_mult(alpha, csr_val_A[j]);

                nnz[col_A] = row_end_C;

                csr_col_ind_C[row_end_C] = col_A + base_C;
                csr_val_C[row_end_C]     = val_A;
                ++row_end_C;
            }

            int row_begin_B = csr_row_ptr_B[i] - base_B;
            int row_end_B   = csr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(int j = row_begin_B; j < row_end_B; ++j)
            {
                // Current column of B
                int col_B = csr_col_ind_B[j] - base_B;

                // Current value of B
                T val_B = testing_mult(beta, csr_val_B[j]);

                // Check if a new nnz is generated or if the value is added
                if(nnz[col_B] < row_begin_C)
                {
                    nnz[col_B] = row_end_C;

                    csr_col_ind_C[row_end_C] = col_B + base_C;
                    csr_val_C[row_end_C]     = val_B;
                    ++row_end_C;
                }
                else
                {
                    csr_val_C[nnz[col_B]] = csr_val_C[nnz[col_B]] + val_B;
                }
            }
        }
    }

    int nnz = csr_row_ptr_C[M] - base_C;

    std::vector<int> col(nnz);
    std::vector<T>   val(nnz);

    for(int i = 0; i < nnz; ++i)
    {
        col[i] = csr_col_ind_C[i];
        val[i] = csr_val_C[i];
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; ++i)
    {
        int row_begin = csr_row_ptr_C[i] - base_C;
        int row_end   = csr_row_ptr_C[i + 1] - base_C;
        int row_nnz   = row_end - row_begin;

        std::vector<int> perm(row_nnz);
        for(int j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        int* col_entry = &col[row_begin];
        T*   val_entry = &val[row_begin];

        std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
            return col_entry[a] <= col_entry[b];
        });

        for(int j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

/* ============================================================================================ */
/*! \brief  Compute sparse matrix sparse matrix multiplication. */
template <typename I, typename J, typename T>
static I host_csrgemm2_nnz(J                    m,
                           J                    n,
                           J                    k,
                           const T*             alpha,
                           const I*             csr_row_ptr_A,
                           const J*             csr_col_ind_A,
                           const I*             csr_row_ptr_B,
                           const J*             csr_col_ind_B,
                           const T*             beta,
                           const I*             csr_row_ptr_D,
                           const J*             csr_col_ind_D,
                           I*                   csr_row_ptr_C,
                           hipsparseIndexBase_t idx_base_A,
                           hipsparseIndexBase_t idx_base_B,
                           hipsparseIndexBase_t idx_base_C,
                           hipsparseIndexBase_t idx_base_D)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<J> nnz(n, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int nthreads = 1;
        int tid      = 0;
#endif

        J rows_per_thread = (m + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, m);

        // Index base
        csr_row_ptr_C[0] = idx_base_C;

        // Loop over rows of A
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            if(alpha)
            {
                I row_begin_A = csr_row_ptr_A[i] - idx_base_A;
                I row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

                // Loop over columns of A
                for(I j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    J col_A = csr_col_ind_A[j] - idx_base_A;

                    I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                    I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                    // Loop over columns of B in row col_A
                    for(I irow = row_begin_B; irow < row_end_B; ++irow)
                    {
                        // Current column of B
                        J col_B = csr_col_ind_B[irow] - idx_base_B;

                        // Check if a new nnz is generated
                        if(nnz[col_B] != i)
                        {
                            nnz[col_B] = i;
                            ++csr_row_ptr_C[i + 1];
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                I row_begin_D = csr_row_ptr_D[i] - idx_base_D;
                I row_end_D   = csr_row_ptr_D[i + 1] - idx_base_D;

                // Loop over columns of D
                for(I j = row_begin_D; j < row_end_D; ++j)
                {
                    J col_D = csr_col_ind_D[j] - idx_base_D;

                    // Check if a new nnz is generated
                    if(nnz[col_D] != i)
                    {
                        nnz[col_D] = i;
                        ++csr_row_ptr_C[i + 1];
                    }
                }
            }
        }
    }

    // Scan to obtain row offsets
    for(J i = 0; i < m; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    return csr_row_ptr_C[m] - idx_base_C;
}

template <typename I, typename J, typename T>
static void host_csrgemm2(J                    m,
                          J                    n,
                          J                    k,
                          const T*             alpha,
                          const I*             csr_row_ptr_A,
                          const J*             csr_col_ind_A,
                          const T*             csr_val_A,
                          const I*             csr_row_ptr_B,
                          const J*             csr_col_ind_B,
                          const T*             csr_val_B,
                          const T*             beta,
                          const I*             csr_row_ptr_D,
                          const J*             csr_col_ind_D,
                          const T*             csr_val_D,
                          const I*             csr_row_ptr_C,
                          J*                   csr_col_ind_C,
                          T*                   csr_val_C,
                          hipsparseIndexBase_t idx_base_A,
                          hipsparseIndexBase_t idx_base_B,
                          hipsparseIndexBase_t idx_base_C,
                          hipsparseIndexBase_t idx_base_D)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<I> nnz(n, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int nthreads = 1;
        int tid      = 0;
#endif

        J rows_per_thread = (m + nthreads - 1) / nthreads;
        J chunk_begin     = rows_per_thread * tid;
        J chunk_end       = std::min(chunk_begin + rows_per_thread, m);

        // Loop over rows of A
        for(J i = chunk_begin; i < chunk_end; ++i)
        {
            I row_begin_C = csr_row_ptr_C[i] - idx_base_C;
            I row_end_C   = row_begin_C;

            if(alpha)
            {
                I row_begin_A = csr_row_ptr_A[i] - idx_base_A;
                I row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

                // Loop over columns of A
                for(I j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    J col_A = csr_col_ind_A[j] - idx_base_A;
                    // Current value of A
                    T val_A = testing_mult(*alpha, csr_val_A[j]);

                    I row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                    I row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                    // Loop over columns of B in row col_A
                    for(I l = row_begin_B; l < row_end_B; ++l)
                    {
                        // Current column of B
                        J col_B = csr_col_ind_B[l] - idx_base_B;
                        // Current value of B
                        T val_B = csr_val_B[l];

                        // Check if a new nnz is generated or if the product is appended
                        if(nnz[col_B] < row_begin_C)
                        {
                            nnz[col_B]               = row_end_C;
                            csr_col_ind_C[row_end_C] = col_B + idx_base_C;
                            csr_val_C[row_end_C]     = testing_mult(val_A, val_B);
                            ++row_end_C;
                        }
                        else
                        {
                            csr_val_C[nnz[col_B]]
                                = csr_val_C[nnz[col_B]] + testing_mult(val_A, val_B);
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                I row_begin_D = csr_row_ptr_D[i] - idx_base_D;
                I row_end_D   = csr_row_ptr_D[i + 1] - idx_base_D;

                // Loop over columns of D
                for(I j = row_begin_D; j < row_end_D; ++j)
                {
                    // Current column of D
                    J col_D = csr_col_ind_D[j] - idx_base_D;
                    // Current value of D
                    T val_D = testing_mult(*beta, csr_val_D[j]);

                    // Check if a new nnz is generated or if the value is added
                    if(nnz[col_D] < row_begin_C)
                    {
                        nnz[col_D] = row_end_C;

                        csr_col_ind_C[row_end_C] = col_D + idx_base_C;
                        csr_val_C[row_end_C]     = val_D;
                        ++row_end_C;
                    }
                    else
                    {
                        csr_val_C[nnz[col_D]] = csr_val_C[nnz[col_D]] + val_D;
                    }
                }
            }
        }
    }

    I nnz_C = csr_row_ptr_C[m] - idx_base_C;

    std::vector<J> col(nnz_C);
    std::vector<T> val(nnz_C);

    memcpy(col.data(), csr_col_ind_C, sizeof(J) * nnz_C);
    memcpy(val.data(), csr_val_C, sizeof(T) * nnz_C);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(J i = 0; i < m; ++i)
    {
        I row_begin = csr_row_ptr_C[i] - idx_base_C;
        I row_end   = csr_row_ptr_C[i + 1] - idx_base_C;
        J row_nnz   = row_end - row_begin;

        std::vector<J> perm(row_nnz);
        for(J j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        J* col_entry = &col[row_begin];
        T* val_entry = &val[row_begin];

        std::sort(perm.begin(), perm.end(), [&](const I& a, const I& b) {
            return col_entry[a] <= col_entry[b];
        });

        for(J j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  query for hipsparse version and git commit SHA-1. */
void query_version(char* version);

/* ============================================================================================ */
/*  device query and print out their ID and name */
int query_device_property();

/*  set current device to device_id */
void set_device(int device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipsparse sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

#ifdef __cplusplus
}
#endif

inline void missing_file_error_message(const char* filename)
{
    std::cerr << "#" << std::endl;
    std::cerr << "# error:" << std::endl;
    std::cerr << "# cannot open file '" << filename << "'" << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr << "# PLEASE READ CAREFULLY !" << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr << "# What could be the reason of this error: " << std::endl;
    std::cerr << "# You are running the testing application and it expects to find the file "
                 "at the specified location. This means that either you did not download the test "
                 "matrices, or you did not specify the location of the folder containing your "
                 "files. If you want to specify the location of the folder containing your files, "
                 "then you will find the needed information with 'hipsparse-test --help'."
                 "If you need to download matrices, then a cmake script "
                 "'hipsparse_clientmatrices.cmake' is available from the hipsparse client package."
              << std::endl;
    std::cerr << "#" << std::endl;
    std::cerr
        << "# Examples: 'hipsparse_clientmatrices.cmake -DCMAKE_MATRICES_DIR=<path-of-your-folder>'"
        << std::endl;
    std::cerr << "#           'hipsparse-test --matrices-dir <path-of-your-folder>'" << std::endl;
    std::cerr << "# (or        'export "
                 "HIPSPARSE_CLIENTS_MATRICES_DIR=<path-of-your-folder>;hipsparse-test')"
              << std::endl;
    std::cerr << "#" << std::endl;
}

static const char* s_hipsparse_clients_matrices_dir = nullptr;

inline const char* get_hipsparse_clients_matrices_dir()
{
    return s_hipsparse_clients_matrices_dir;
}

inline std::string get_filename(const std::string& bin_file)
{
    const char* matrices_dir = get_hipsparse_clients_matrices_dir();
    if(matrices_dir == nullptr)
    {
        matrices_dir = getenv("HIPSPARSE_CLIENTS_MATRICES_DIR");
    }

    std::string r;
    if(matrices_dir != nullptr)
    {
        r = std::string(matrices_dir) + "/" + bin_file;
    }
    else
    {
        r = hipsparse_exepath() + "../matrices/" + bin_file;
    }

    FILE* tmpf = fopen(r.c_str(), "r");
    if(!tmpf)
    {
        missing_file_error_message(r.c_str());
        std::cerr << "exit(HIPSPARSE_STATUS_INTERNAL_ERROR)" << std::endl;
        exit(HIPSPARSE_STATUS_INTERNAL_ERROR);
    }
    else
    {
        fclose(tmpf);
    }
    return r;
}

struct testhyb
{
    int                     m;
    int                     n;
    hipsparseHybPartition_t partition;
    int                     ell_nnz;
    int                     ell_width;
    int*                    ell_col_ind;
    void*                   ell_val;
    int                     coo_nnz;
    int*                    coo_row_ind;
    int*                    coo_col_ind;
    void*                   coo_val;
};

template <typename I>
hipsparseIndexType_t getIndexType()
{
    return (typeid(I) == typeid(int32_t)) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
}

template <typename T>
hipDataType getDataType()
{
    return (typeid(T) == typeid(int8_t)) ? HIP_R_8I
           : (typeid(T) == typeid(float))
               ? HIP_R_32F
               : ((typeid(T) == typeid(double))
                      ? HIP_R_64F
                      : ((typeid(T) == typeid(hipComplex) ? HIP_C_32F : HIP_C_64F)));
}

#endif // TESTING_UTILITY_HPP
