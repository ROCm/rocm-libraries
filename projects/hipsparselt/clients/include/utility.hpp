/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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
 *******************************************************************************/

#pragma once

#include "hipsparselt_vector.hpp"
#include <cstdio>
#include <hipsparselt/hipsparselt.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <stdlib.h>

/*!\file
 * \brief provide common utilities
 */

// We use hipsparselt_cout and hipsparselt_cerr instead of std::cout, std::cerr, stdout and stderr,
// for thread-safe IO.
//
// All stdio and std::ostream functions related to stdout and stderr are poisoned, as are
// functions which can create buffer overflows, or which are inherently thread-unsafe.
//
// This must come after the header #includes above, to avoid poisoning system headers.
//
// This is only enabled for hipsparselt-test and hipsparselt-bench.
//
// If you are here because of a poisoned identifier error, here is the rationale for each
// included identifier:
//
// cout, stdout: hipsparselt_cout should be used instead, for thread-safe and atomic line buffering
// cerr, stderr: hipsparselt_cerr should be used instead, for thread-safe and atomic line buffering
// clog: C++ stream which should not be used
// gets: Always unsafe; buffer-overflows; removed from later versions of the language; use fgets
// puts, putchar, fputs, printf, fprintf, vprintf, vfprintf: Use hipsparselt_cout or hipsparselt_cerr
// sprintf, vsprintf: Possible buffer overflows; us snprintf or vsnprintf instead
// strerror: Thread-unsafe; use snprintf / dprintf with %m or strerror_* alternatives
// strsignal: Thread-unsafe; use sys_siglist[signal] instead
// strtok: Thread-unsafe; use strtok_r
// gmtime, ctime, asctime, localtime: Thread-unsafe
// tmpnam: Thread-unsafe; use mkstemp or related functions instead
// putenv: Use setenv instead
// clearenv, fcloseall, ecvt, fcvt: Miscellaneous thread-unsafe functions
// sleep: Might interact with signals by using alarm(); use nanosleep() instead
// abort: Does not abort as cleanly as hipsparselt_abort, and can be caught by a signal handler

#if defined(GOOGLE_TEST) || defined(HIPSPARSELT_BENCH)
#undef stdout
#undef stderr
#pragma GCC poison cout cerr clog stdout stderr gets puts putchar fputs fprintf printf sprintf    \
    vfprintf vprintf vsprintf perror strerror strtok gmtime ctime asctime localtime tmpnam putenv \
        clearenv fcloseall ecvt fcvt sleep abort strsignal
#else
// Suppress warnings about hipMalloc(), hipFree() except in hipsparselt-test and hipsparselt-bench
#undef hipMalloc
#undef hipFree
#endif

#define LIMITED_MEMORY_STRING "Error: Attempting to allocate more memory than available."
#define TOO_MANY_DEVICES_STRING "Error: Too many devices requested."
#define HMM_NOT_SUPPORTED "Error: HMM not supported."

// TODO: This is dependent on internal gtest behaviour.
// Compared with result.message() when a test ended. Note that "Succeeded\n" is
// added to the beginning of the message automatically by gtest, so this must be compared.
#define LIMITED_MEMORY_STRING_GTEST "Succeeded\n" LIMITED_MEMORY_STRING
#define TOO_MANY_DEVICES_STRING_GTEST "Succeeded\n" TOO_MANY_DEVICES_STRING
#define HMM_NOT_SUPPORTED_GTEST "Succeeded\n" HMM_NOT_SUPPORTED

enum class hipsparselt_batch_type
{
    none = 0,
    batched,
    strided_batched
};

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */
class hipsparselt_local_handle
{
    hipsparseLtHandle_t m_handle;
    void*               m_memory = nullptr;

public:
    hipsparselt_local_handle();

    explicit hipsparselt_local_handle(const Arguments& arg);

    ~hipsparselt_local_handle();

    hipsparselt_local_handle(const hipsparselt_local_handle&)            = delete;
    hipsparselt_local_handle(hipsparselt_local_handle&&)                 = delete;
    hipsparselt_local_handle& operator=(const hipsparselt_local_handle&) = delete;
    hipsparselt_local_handle& operator=(hipsparselt_local_handle&&)      = delete;

    // Allow hipsparselt_local_handle to be used anywhere hipsparseLtHandle_t is expected
    operator hipsparseLtHandle_t&()
    {
        return m_handle;
    }
    operator const hipsparseLtHandle_t&() const
    {
        return m_handle;
    }
    operator hipsparseLtHandle_t*()
    {
        return &m_handle;
    }
    operator const hipsparseLtHandle_t*() const
    {
        return &m_handle;
    }
};

typedef enum hipsparselt_matrix_type_
{
    hipsparselt_matrix_type_dense      = 1, /**< dense matrix type. */
    hipsparselt_matrix_type_structured = 2, /**< structured matrix type. */
} hipsparselt_matrix_type;

/* ============================================================================================ */
/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class hipsparselt_local_mat_descr
{
    hipsparseLtMatDescriptor_t m_descr;
    hipsparseStatus_t          m_status  = HIPSPARSE_STATUS_NOT_INITIALIZED;
    static constexpr int       alignment = 16;

public:
    hipsparselt_local_mat_descr(hipsparselt_matrix_type    mtype,
                                const hipsparseLtHandle_t* handle,
                                int64_t                    row,
                                int64_t                    col,
                                int64_t                    ld,
                                hipDataType                type,
                                hipsparseOrder_t           order)
    {
        if(mtype == hipsparselt_matrix_type_structured)
            this->m_status = hipsparseLtStructuredDescriptorInit(handle,
                                                                 &this->m_descr,
                                                                 row,
                                                                 col,
                                                                 ld,
                                                                 this->alignment,
                                                                 type,
                                                                 order,
                                                                 HIPSPARSELT_SPARSITY_50_PERCENT);
        else
            this->m_status = hipsparseLtDenseDescriptorInit(
                handle, &this->m_descr, row, col, ld, this->alignment, type, order);
    }

    ~hipsparselt_local_mat_descr()
    {
        if(this->m_status == HIPSPARSE_STATUS_SUCCESS)
            hipsparseLtMatDescriptorDestroy(&this->m_descr);
    }

    hipsparselt_local_mat_descr(const hipsparselt_local_mat_descr&)            = delete;
    hipsparselt_local_mat_descr(hipsparselt_local_mat_descr&&)                 = delete;
    hipsparselt_local_mat_descr& operator=(const hipsparselt_local_mat_descr&) = delete;
    hipsparselt_local_mat_descr& operator=(hipsparselt_local_mat_descr&&)      = delete;

    hipsparseStatus_t status()
    {
        return m_status;
    }

    // Allow hipsparselt_local_mat_descr to be used anywhere hipsparseLtMatDescriptor_t is expected
    operator hipsparseLtMatDescriptor_t&()
    {
        return m_descr;
    }
    operator const hipsparseLtMatDescriptor_t&() const
    {
        return m_descr;
    }
    operator hipsparseLtMatDescriptor_t*()
    {
        return &m_descr;
    }
    operator const hipsparseLtMatDescriptor_t*() const
    {
        return &m_descr;
    }
};

/* ============================================================================================ */
/*! \brief  local matrix multiplication descriptor which is automatically created and destroyed  */
class hipsparselt_local_matmul_descr
{
    hipsparseLtMatmulDescriptor_t m_descr;
    hipsparseStatus_t             m_status = HIPSPARSE_STATUS_NOT_INITIALIZED;

public:
    hipsparselt_local_matmul_descr(const hipsparseLtHandle_t*        handle,
                                   hipsparseOperation_t              opA,
                                   hipsparseOperation_t              opB,
                                   const hipsparseLtMatDescriptor_t* matA,
                                   const hipsparseLtMatDescriptor_t* matB,
                                   const hipsparseLtMatDescriptor_t* matC,
                                   const hipsparseLtMatDescriptor_t* matD,
                                   hipsparseLtComputetype_t          compute_type)
    {
        this->m_status = hipsparseLtMatmulDescriptorInit(
            handle, &this->m_descr, opA, opB, matA, matB, matC, matD, compute_type);
    }

    ~hipsparselt_local_matmul_descr() {}

    hipsparselt_local_matmul_descr(const hipsparselt_local_matmul_descr&)            = delete;
    hipsparselt_local_matmul_descr(hipsparselt_local_matmul_descr&&)                 = delete;
    hipsparselt_local_matmul_descr& operator=(const hipsparselt_local_matmul_descr&) = delete;
    hipsparselt_local_matmul_descr& operator=(hipsparselt_local_matmul_descr&&)      = delete;

    hipsparseStatus_t status()
    {
        return m_status;
    }

    // Allow hipsparselt_local_matmul_descr to be used anywhere hipsparseLtMatmulDescriptor_t is expected
    operator hipsparseLtMatmulDescriptor_t&()
    {
        return m_descr;
    }
    operator const hipsparseLtMatmulDescriptor_t&() const
    {
        return m_descr;
    }
    operator hipsparseLtMatmulDescriptor_t*()
    {
        return &m_descr;
    }
    operator const hipsparseLtMatmulDescriptor_t*() const
    {
        return &m_descr;
    }
};

/* ================================================================================================================= */
/*! \brief  local matrix multiplication algorithm selection descriptor which is automatically created and destroyed  */
class hipsparselt_local_matmul_alg_selection
{
    hipsparseLtMatmulAlgSelection_t m_alg_sel;
    hipsparseStatus_t               m_status = HIPSPARSE_STATUS_NOT_INITIALIZED;

public:
    hipsparselt_local_matmul_alg_selection(const hipsparseLtHandle_t*           handle,
                                           const hipsparseLtMatmulDescriptor_t* matmul,
                                           hipsparseLtMatmulAlg_t               alg)
    {

        this->m_status = hipsparseLtMatmulAlgSelectionInit(handle, &this->m_alg_sel, matmul, alg);
    }

    ~hipsparselt_local_matmul_alg_selection() {}

    hipsparselt_local_matmul_alg_selection(const hipsparselt_local_matmul_alg_selection&) = delete;
    hipsparselt_local_matmul_alg_selection(hipsparselt_local_matmul_alg_selection&&)      = delete;
    hipsparselt_local_matmul_alg_selection& operator=(const hipsparselt_local_matmul_alg_selection&)
        = delete;
    hipsparselt_local_matmul_alg_selection& operator=(hipsparselt_local_matmul_alg_selection&&)
        = delete;

    hipsparseStatus_t status()
    {
        return m_status;
    }

    operator hipsparseLtMatmulAlgSelection_t&()
    {
        return m_alg_sel;
    }
    operator const hipsparseLtMatmulAlgSelection_t&() const
    {
        return m_alg_sel;
    }
    operator hipsparseLtMatmulAlgSelection_t*()
    {
        return &m_alg_sel;
    }
    operator const hipsparseLtMatmulAlgSelection_t*() const
    {
        return &m_alg_sel;
    }
};

/* ================================================================================================================= */
/*! \brief  local matrix multiplication plan descriptor which is automatically created and destroyed  */
class hipsparselt_local_matmul_plan
{
    hipsparseLtMatmulPlan_t m_plan;
    hipsparseStatus_t       m_status = HIPSPARSE_STATUS_NOT_INITIALIZED;

public:
    hipsparselt_local_matmul_plan(const hipsparseLtHandle_t*             handle,
                                  const hipsparseLtMatmulDescriptor_t*   matmul,
                                  const hipsparseLtMatmulAlgSelection_t* alg_sel)
    {

        this->m_status = hipsparseLtMatmulPlanInit(handle, &this->m_plan, matmul, alg_sel);
    }

    ~hipsparselt_local_matmul_plan()
    {
        if(this->m_status == HIPSPARSE_STATUS_SUCCESS)
            hipsparseLtMatmulPlanDestroy(&this->m_plan);
    }

    hipsparselt_local_matmul_plan(const hipsparselt_local_matmul_plan&)            = delete;
    hipsparselt_local_matmul_plan(hipsparselt_local_matmul_plan&&)                 = delete;
    hipsparselt_local_matmul_plan& operator=(const hipsparselt_local_matmul_plan&) = delete;
    hipsparselt_local_matmul_plan& operator=(hipsparselt_local_matmul_plan&&)      = delete;

    hipsparseStatus_t status()
    {
        return this->m_status;
    }

    operator hipsparseLtMatmulPlan_t&()
    {
        return m_plan;
    }
    operator const hipsparseLtMatmulPlan_t&() const
    {
        return m_plan;
    }
    operator hipsparseLtMatmulPlan_t*()
    {
        return &m_plan;
    }
    operator const hipsparseLtMatmulPlan_t*() const
    {
        return &m_plan;
    }
};

/* ============================================================================================ */
/*  device query and print out their ID and name */
int64_t query_device_property();

/*  set current device to device_id */
void set_device(int64_t device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipsparselt sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us_sync_device();

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

/*! \brief  CPU Timer(in microsecond): no GPU synchronization and return wall time */
double get_time_us_no_sync();

/* ============================================================================================ */
// Return path of this executable
std::string hipsparselt_exepath();

/* ============================================================================================ */
// Temp directory rooted random path
std::string hipsparselt_tempname();

/* ============================================================================================ */
/* Read environment variable */
const char* read_env_var(const char* env_var);

/* ============================================================================================ */
/* Compute strided batched matrix allocation size allowing for strides smaller than full matrix */
size_t strided_batched_matrix_size(int rows, int cols, int lda, int64_t stride, int batch_count);

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
inline void hipsparselt_print_matrix(
    std::vector<T> CPU_result, std::vector<T> GPU_result, size_t m, size_t n, size_t lda)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
        {
            hipsparselt_cout << "matrix  col " << i << ", row " << j
                             << ", CPU result=" << CPU_result[j + i * lda]
                             << ", GPU result=" << GPU_result[j + i * lda] << "\n";
        }
}

template <typename T>
void hipsparselt_print_matrix(const char* name, T* A, size_t m, size_t n, size_t lda)
{
    hipsparselt_cout << "---------- " << name << " ----------\n";
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
            hipsparselt_cout << std::setprecision(0) << std::setw(5) << A[i + j * lda] << " ";
        hipsparselt_cout << std::endl;
    }
}

/* ============================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a banded matrix. */
template <typename T>
inline void regular_to_banded(
    bool upper, const T* A, int64_t lda, T* AB, int64_t ldab, int64_t n, int64_t k)
{
    // convert regular hA matrix to banded hAB matrix
    for(int j = 0; j < n; j++)
    {
        int64_t min1 = upper ? std::max(0, static_cast<int>(j - k)) : j;
        int64_t max1 = upper ? j : std::min(static_cast<int>(n - 1), static_cast<int>(j + k));
        int64_t m    = upper ? k - j : -j;

        // Move bands of hA into new banded hAB format.
        for(int i = min1; i <= max1; i++)
            AB[j * ldab + (m + i)] = A[j * lda + i];

        min1 = upper ? k + 1 : std::min(k + 1, n - j);
        max1 = ldab - 1;

        // fill in bottom with random data to ensure we aren't using it.
        // for !upper, fill in bottom right triangle as well.
        for(int i = min1; i <= max1; i++)
            hipsparselt_init<T>(AB + j * ldab + i, 1, 1, 1);

        // for upper, fill in top left triangle with random data to ensure
        // we aren't using it.
        if(upper)
        {
            for(int i = 0; i < m; i++)
                hipsparselt_init<T>(AB + j * ldab + i, 1, 1, 1);
        }
    }
}

/* =============================================================================== */
/*! \brief For testing purposes, zeros out elements not needed in a banded matrix. */
template <typename T>
inline void banded_matrix_setup(bool upper, T* A, int64_t lda, int64_t n, int64_t k)
{
    // Made A a banded matrix with k sub/super-diagonals
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(upper && (j > k + i || i > j))
                A[j * n + i] = T(0);
            else if(!upper && (i > k + j || j > i))
                A[j * n + i] = T(0);
        }
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a packed matrix.                  */
template <typename T>
inline void regular_to_packed(bool upper, const T* A, T* AP, int64_t n)
{
    int index = 0;
    if(upper)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = i; j < n; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
}

template <typename T>
void print_strided_batched(
    const char* name, T* A, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    constexpr bool is_int = std::is_same<T, int8_t>();
    using Tp              = std::conditional_t<is_int, int32_t, float>;
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    hipsparselt_cout << "---------- " << name << " ----------\n";
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                hipsparselt_cout << static_cast<Tp>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]) << "|";
            }
            hipsparselt_cout << "\n";
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            hipsparselt_cout << "\n";
    }
    hipsparselt_cout << std::flush;
}

inline hipsparseStatus_t expected_hipsparse_status_of_matrix_size(hipDataType      type,
                                                                  int64_t          m,
                                                                  int64_t          n,
                                                                  int64_t          ld,
                                                                  hipsparseOrder_t order,
                                                                  bool             isSparse = false)
{
    int row_  = 8;
    int col_  = 8;
    int ld_   = -1;
    int bytes = 1;
#ifdef __HIP_PLATFORM_NVIDIA__
    switch(type)
    {
    case HIP_R_8I:
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E5M2:
        if(isSparse)
            row_ = col_ = ld_ = 32;
        else
            row_ = col_ = ld_ = 16;
        break;
    case HIP_R_16BF:
    case HIP_R_16F:
        bytes = 2;
        if(isSparse)
            row_ = col_ = ld_ = 16;
        else
            ld_ = 8;
        break;
    default:
        break;
    }
#else
    switch(type)
    {
    case HIP_R_8I:
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E5M2:
        row_ = col_ = 16;
        break;
    case HIP_R_16BF:
    case HIP_R_16F:
        bytes = 2;
        break;
    default:
        break;
    }
#endif

    if(m <= 0 || n <= 0)
        return HIPSPARSE_STATUS_INVALID_VALUE;

#ifdef __HIP_PLATFORM_AMD__
    if(m < row_ || n < col_)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;

    if(m % row_ != 0 || n % col_ != 0)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif

    {
        if(order == HIPSPARSE_ORDER_COL)
        {

            if(m > ld)
                return HIPSPARSE_STATUS_INVALID_VALUE;

            if(ld_ != -1 && ld % ld_ != 0)
                return HIPSPARSE_STATUS_NOT_SUPPORTED;
#ifdef __HIP_PLATFORM_NVIDIA__
            if(n * ld * bytes > 4294967295)
                return HIPSPARSE_STATUS_INVALID_VALUE;
#endif
        }
        else
        {
            if(n > ld)
                return HIPSPARSE_STATUS_INVALID_VALUE;

            if(ld_ != -1 && ld % ld_ != 0)
                return HIPSPARSE_STATUS_NOT_SUPPORTED;
#ifdef __HIP_PLATFORM_NVIDIA__
            if(m * ld * bytes > 4294967295)
                return HIPSPARSE_STATUS_INVALID_VALUE;
#endif
        }
    }

#ifdef __HIP_PLATFORM_NVIDIA__
    if(m < row_ || n < col_)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;

    if(m % row_ != 0 || n % col_ != 0)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

inline hipsparseStatus_t expected_hipsparse_status_of_matrix_stride(
    int64_t stride, int64_t m, int64_t n, int64_t ld, hipsparseOrder_t order)
{
    if(stride == 0 || stride >= (order == HIPSPARSE_ORDER_COL ? ld * n : ld * m))
        return HIPSPARSE_STATUS_SUCCESS;
    else
        return HIPSPARSE_STATUS_INVALID_VALUE;
}

class Logger
{

public:
    Logger(int log_level)
    {
        this->log_level = log_level;
        this->pre_log_level = -1;
        if(this->log_level)
        {
            char* str_layer_mode;
            if((str_layer_mode = getenv("HIPSPARSELT_LOG_LEVEL")) != NULL)
            {
                this->pre_log_level = atoi(str_layer_mode);
            }
            setenv("HIPSPARSELT_LOG_LEVEL", std::to_string(this->log_level).c_str(), 1);
        }
    }
    ~Logger()
    {
        if(this->log_level)
        {
            if(this->pre_log_level == -1)
                unsetenv("HIPSPARSELT_LOG_LEVEL");
            else
                setenv("HIPSPARSELT_LOG_LEVEL", std::to_string(this->pre_log_level).c_str(), 1);
        }
    }
    int log_level;
    int pre_log_level;
};
