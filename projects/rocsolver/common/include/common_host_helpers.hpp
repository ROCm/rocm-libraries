/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <hip/hip_runtime_api.h>

#include "fmt_rocblas_types.hpp"
#include "rocblas_utility.hpp"

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_BEGIN_NAMESPACE
#endif

/*
 * ===========================================================================
 *    common location for functions that are used by both the rocSOLVER
 *    library and rocSOLVER client code.
 * ===========================================================================
 */

#define THROW_IF_HIP_ERROR(expr)                                                       \
    do                                                                                 \
    {                                                                                  \
        hipError_t _status = (expr);                                                   \
        if(_status != hipSuccess)                                                      \
            throw std::runtime_error(fmt::format("{}:{}: [{}] {}", __FILE__, __LINE__, \
                                                 hipGetErrorName(_status),             \
                                                 hipGetErrorString(_status)));         \
    } while(0)

/* =============================================================================================== */

/* Number properties functions.                                                                    */

template <typename T>
constexpr double get_epsilon()
{
    using S = decltype(std::real(T{}));
    return std::numeric_limits<S>::epsilon();
}

template <typename T>
constexpr double get_safemin()
{
    using S = decltype(std::real(T{}));
    auto eps = get_epsilon<S>();
    auto s1 = std::numeric_limits<S>::min();
    auto s2 = 1 / std::numeric_limits<S>::max();
    if(s2 > s1)
        return s2 * (1 + eps);
    return s1;
}

/* =============================================================================================== */
/* Timing functions.                                                                               */

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and
 * return wall time */
double get_time_us();

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and
 * return wall time */
double get_time_us_sync(hipStream_t stream);

/*! \brief  CPU Timer(in microsecond): no GPU synchronization and return wall
 * time */
double get_time_us_no_sync();

/* =============================================================================================== */
/* Print functions.                                                                                */
inline void pairs_to_string(std::string& str, const char* sep)
{
    // do nothing
}
template <typename T1, typename T2, typename... Ts>
void pairs_to_string(std::string& str, const char* sep, T1 arg1, T2 arg2, Ts... args)
{
    str += fmt::format("{} {}", arg1, arg2);

    if(sizeof...(Ts) > 0)
    {
        str += sep;
        pairs_to_string(str, sep, args...);
    }
}

/** Set of helpers to print out data hosted in the CPU and/or the GPU **/
/***********************************************************************/

/*! \brief Print provided data into specified stream */
template <typename T>
void print_to_stream(std::ostream& os,
                     const std::string name,
                     const rocblas_int m,
                     const rocblas_int n,
                     T* A,
                     const rocblas_int inca,
                     const rocblas_int lda,
                     const rocblas_fill uplo)
{
    std::string s;
    bool empty = name.empty();
    if(!empty)
        s += fmt::format("{}-by-{} matrix: {}\n", m, n, name);

    if(uplo == rocblas_fill_full)
    {
        // normal case
        for(int i = 0; i < m; i++)
        {
            if(!empty)
                s += "    ";
            for(int j = 0; j < n; j++)
            {
                s += fmt::format("{}", A[j * lda + i * inca]);
                if(j < n - 1)
                    s += ", ";
            }
            s += '\n';
        }
    }
    else
    {
        // symmetric case
        for(int i = 0; i < std::min(m, n); i++)
        {
            if(!empty)
                s += "    ";
            for(int j = 0; j < std::min(m, n); j++)
            {
                if(uplo == rocblas_fill_upper)
                {
                    if(i < j)
                        s += fmt::format("{}", A[j * lda + i * inca]);
                    else
                        s += fmt::format("{}", A[i * lda + j * inca]);
                }
                else
                {
                    if(i > j)
                        s += fmt::format("{}", A[j * lda + i * inca]);
                    else
                        s += fmt::format("{}", A[i * lda + j * inca]);
                }

                if(j < n - 1)
                    s += ", ";
            }
            s += '\n';
        }
    }

    s += '\n';
    os << s;
    os.flush();
}

/*! \brief Print data from a normal or strided_batched array on the GPU to screen*/
template <typename T>
void print_device_matrix(std::ostream& os,
                         const std::string name,
                         const rocblas_int m,
                         const rocblas_int n,
                         T* A,
                         const rocblas_int lda,
                         const rocblas_stride stride = 1,
                         const rocblas_int idx = 0,
                         const rocblas_fill uplo = rocblas_fill_full,
                         const rocblas_int inca = 1)
{
    size_t to_read = std::max(inca * (m - 1) + n, lda * (n - 1) + m);

    std::vector<T> hA(to_read);
    THROW_IF_HIP_ERROR(
        hipMemcpy(hA.data(), A + idx * stride, sizeof(T) * to_read, hipMemcpyDeviceToHost));

    print_to_stream<T>(os, name, m, n, hA.data(), inca, lda, uplo);
}

/*! \brief Print data from a batched array on the GPU to screen*/
template <typename T>
void print_device_matrix(std::ostream& os,
                         const std::string name,
                         const rocblas_int m,
                         const rocblas_int n,
                         T* const A[],
                         const rocblas_int lda,
                         const rocblas_stride stride = 1,
                         const rocblas_int idx = 0,
                         const rocblas_fill uplo = rocblas_fill_full,
                         const rocblas_int inca = 1)
{
    size_t to_read = std::max(inca * (m - 1) + n, lda * (n - 1) + m);

    std::vector<T> hA(to_read);
    T* AA[1];
    THROW_IF_HIP_ERROR(hipMemcpy(AA, A + idx, sizeof(T*), hipMemcpyDeviceToHost));
    THROW_IF_HIP_ERROR(hipMemcpy(hA.data(), AA[0], sizeof(T) * to_read, hipMemcpyDeviceToHost));

    print_to_stream<T>(os, name, m, n, hA.data(), inca, lda, uplo);
}

/*! \brief Print data from a normal or strided_batched array on the GPU to file*/
template <typename T>
void print_device_matrix(const std::string file,
                         const rocblas_int m,
                         const rocblas_int n,
                         T* A,
                         const rocblas_int lda,
                         const rocblas_stride stride = 1,
                         const rocblas_int idx = 0,
                         const rocblas_fill uplo = rocblas_fill_full,
                         const rocblas_int inca = 1)
{
    size_t to_read = std::max(inca * (m - 1) + n, lda * (n - 1) + m);

    std::ofstream os(file);
    std::vector<T> hA(to_read);
    THROW_IF_HIP_ERROR(
        hipMemcpy(hA.data(), A + idx * stride, sizeof(T) * to_read, hipMemcpyDeviceToHost));

    print_to_stream<T>(os, "", m, n, hA.data(), inca, lda, uplo);
}

/*! \brief Print data from a batched array on the GPU to file*/
template <typename T>
void print_device_matrix(const std::string file,
                         const rocblas_int m,
                         const rocblas_int n,
                         T* const A[],
                         const rocblas_int lda,
                         const rocblas_stride stride = 1,
                         const rocblas_int idx = 0,
                         const rocblas_fill uplo = rocblas_fill_full,
                         const rocblas_int inca = 1)
{
    size_t to_read = std::max(inca * (m - 1) + n, lda * (n - 1) + m);

    std::ofstream os(file);
    std::vector<T> hA(to_read);
    T* AA[1];
    THROW_IF_HIP_ERROR(hipMemcpy(AA, A + idx, sizeof(T*), hipMemcpyDeviceToHost));
    THROW_IF_HIP_ERROR(hipMemcpy(hA.data(), AA[0], sizeof(T) * to_read, hipMemcpyDeviceToHost));

    print_to_stream<T>(os, "", m, n, hA.data(), inca, lda, uplo);
}

/*! \brief Print data from a normal or strided_batched array on the CPU to screen*/
template <typename T>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* A,
                       const rocblas_int lda,
                       const rocblas_stride stride = 1,
                       const rocblas_int idx = 0,
                       const rocblas_fill uplo = rocblas_fill_full,
                       const rocblas_int inca = 1)
{
    print_to_stream<T>(os, name, m, n, A + idx * stride, inca, lda, uplo);
}

/*! \brief Print data from a batched array on the CPU to screen*/
template <typename T>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* const A[],
                       const rocblas_int lda,
                       const rocblas_stride stride = 1,
                       const rocblas_int idx = 0,
                       const rocblas_fill uplo = rocblas_fill_full,
                       const rocblas_int inca = 1)
{
    print_to_stream<T>(os, name, m, n, A[idx], inca, lda, uplo);
}

/*! \brief Print data from a normal or strided_batched array on the CPU to file*/
template <typename T>
void print_host_matrix(const std::string file,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* A,
                       const rocblas_int lda,
                       const rocblas_stride stride = 1,
                       const rocblas_int idx = 0,
                       const rocblas_fill uplo = rocblas_fill_full,
                       const rocblas_int inca = 1)
{
    std::ofstream os(file);
    print_to_stream<T>(os, "", m, n, A + idx * stride, inca, lda, uplo);
}

/*! \brief Print data from a batched array on the CPU to file*/
template <typename T>
void print_host_matrix(const std::string file,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* const A[],
                       const rocblas_int lda,
                       const rocblas_stride stride = 1,
                       const rocblas_int idx = 0,
                       const rocblas_fill uplo = rocblas_fill_full,
                       const rocblas_int inca = 1)
{
    std::ofstream os(file);
    print_to_stream<T>(os, "", m, n, A[idx], inca, lda, uplo);
}

/*! \brief  Debugging purpose, print out CPU and GPU result matrix */
/*******************************************************************/
template <typename T>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* CPU_result,
                       T* GPU_result,
                       const rocblas_int lda)
{
    std::string s;
    bool empty = name.empty();
    if(!empty)
        s += fmt::format("{}-by-{} matrix: {}\n", m, n, name);

    for(size_t j = 0; j < n; j++)
    {
        for(size_t i = 0; i < m; i++)
        {
            s += fmt::format("matrix  row {}, col {}, CPU result={}, GPU result={}\n", i, j,
                             CPU_result[j * lda + i], GPU_result[j * lda + i]);
        }
    }
    s += '\n';
    os << s;
    os.flush();
}

template <typename T>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* CPU_result,
                       T* GPU_result,
                       const rocblas_int lda,
                       double error_tolerance)
{
    std::string s;
    bool empty = name.empty();
    if(!empty)
        s += fmt::format("{}-by-{} matrix: {}\n", m, n, name);

    for(size_t j = 0; j < n; j++)
    {
        for(size_t i = 0; i < m; i++)
        {
            T comp = (CPU_result[j * lda + i] - GPU_result[j * lda + i]) / CPU_result[j * lda + i];
            if(std::abs(comp) > error_tolerance)
                s += fmt::format("matrix  row {}, col {}, CPU result={}, GPU result={}\n", i, j,
                                 CPU_result[j * lda + i], GPU_result[j * lda + i]);
        }
    }
    s += '\n';
    os << s;
    os.flush();
}

/********* Helpers to read matrix and/or values from file **********/
/*******************************************************************/

// integers:
inline void read_matrix(const std::string filenameS,
                        const rocblas_int m,
                        const rocblas_int n,
                        rocblas_int* A,
                        const rocblas_int lda)
{
    const char* filename = filenameS.c_str();
    int const idebug = 0;
    if(idebug >= 1)
    {
        printf("filename=%s, m=%d, n=%d, lda=%d\n", filename, m, n, lda);
    }

    FILE* mat = fopen(filename, "r");

    if(mat == NULL)
        throw std::invalid_argument(
            fmt::format("Error: Could not open file {} with test data...", filename));

    rewind(mat);

    for(rocblas_int i = 0; i < m; ++i)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            rocblas_int v;
            int read = fscanf(mat, "%d", &v);
            if(read != 1)
                throw std::out_of_range(
                    fmt::format("Error: Could not read element {},{} from file {}", i, j, filename));
            A[i + j * lda] = v;
        }
    }

    if(fclose(mat) != 0)
    {
        throw std::invalid_argument(
            fmt::format("Error: Could not close file {} with test data...", filename));
    }
}
inline void read_last(const std::string filenameS, rocblas_int* A)
{
    const char* filename = filenameS.c_str();
    FILE* mat = fopen(filename, "r");

    if(mat == NULL)
        throw std::invalid_argument(
            fmt::format("Error: Could not open file {} with test data...", filename));

    rewind(mat);

    rocblas_int v;
    while(fscanf(mat, "%d", &v) == 1)
    {
        // do nothing
    }

    *A = v;

    if(fclose(mat) != 0)
    {
        throw std::invalid_argument(
            fmt::format("Error: Could not close file {} with test data...", filename));
    }
}

// singles:
inline void read_matrix(const std::string filenameS,
                        const rocblas_int m,
                        const rocblas_int n,
                        float* A,
                        const rocblas_int lda)
{
    const char* filename = filenameS.c_str();
    int const idebug = 0;
    if(idebug >= 1)
    {
        printf("filename=%s, m=%d, n=%d, lda=%d\n", filename, m, n, lda);
    }

    FILE* mat = fopen(filename, "r");

    if(mat == NULL)
        throw std::invalid_argument(
            fmt::format("Error: Could not open file {} with test data...", filename));

    rewind(mat);

    for(rocblas_int i = 0; i < m; ++i)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            float v;
            int read = fscanf(mat, "%g", &v);
            if(read != 1)
                throw std::out_of_range(
                    fmt::format("Error: Could not read element {},{} from file {}", i, j, filename));
            A[i + j * lda] = v;
        }
    }

    if(fclose(mat) != 0)
    {
        throw std::invalid_argument(
            fmt::format("Error: Could not close file {} with test data...", filename));
    }
}

// doubles:
inline void read_matrix(const std::string filenameS,
                        const rocblas_int m,
                        const rocblas_int n,
                        double* A,
                        const rocblas_int lda)
{
    const char* filename = filenameS.c_str();
    int const idebug = 0;
    if(idebug >= 1)
    {
        printf("filename=%s, m=%d, n=%d, lda=%d\n", filename, m, n, lda);
    }
    FILE* mat = fopen(filename, "r");

    if(mat == NULL)
        throw std::invalid_argument(
            fmt::format("Error: Could not open file {} with test data...", filename));

    rewind(mat);

    for(rocblas_int i = 0; i < m; ++i)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            double v;
            int read = fscanf(mat, "%lg", &v);
            if(read != 1)
                throw std::out_of_range(
                    fmt::format("Error: Could not read element {},{} from file {}", i, j, filename));
            A[i + j * lda] = v;
        }
    }

    if(fclose(mat) != 0)
    {
        throw std::invalid_argument(
            fmt::format("Error: Could not close file {} with test data...", filename));
    }
}

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_END_NAMESPACE
#endif
