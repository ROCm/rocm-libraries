/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "cblas.h"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_vector.hpp"
#include "utility.hpp"
#include <cmath>
#include <cstdio>
#include <hipblaslt/hipblaslt.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

/* =====================================================================
        allclose check: abs(A-B) <= atol + abs(rtol * B)
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides allclose check
 *
 * The check tries every (atol, rtol) pair from allclose_tolerances(), tightest atol first and,
 * within one atol, tightest rtol first. It reports the first pair under which every element
 * passes, or a pair of 1 when none does. A check over several matrices (batches, or the gemms
 * of a grouped GEMM) reports the loosest pair any one matrix needed, so a pair of 1 means at
 * least one matrix failed. An empty matrix reports 0, the tightest possible result.
 */

inline const std::vector<double>& allclose_tolerances()
{
    static const std::vector<double> tolerances{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    return tolerances;
}

/*! \brief the elements that fail the loosest pair, and the one furthest from its reference.
 *  The real and imaginary parts of a complex matrix count as separate elements. */
struct AllcloseFailure
{
    size_t  bad_count      = 0;
    size_t  compared_count = 0;
    int64_t batch          = 0;
    int64_t row            = 0;
    int64_t col            = 0;
    bool    imag           = false;
    double  expected       = 0;
    double  got            = 0;
    double  error          = 0;
};

inline bool allclose_within(double a, double b, double atol, double rtol)
{
    return a == b || std::abs(a - b) <= atol + std::abs(rtol * b);
}

/*! \brief keeps the looser of (atol, rtol) and (one_atol, one_rtol); start a running pair at 0 */
inline void allclose_merge_pair(double& atol, double& rtol, double one_atol, double one_rtol)
{
    if(one_atol > atol || (one_atol == atol && one_rtol > rtol))
    {
        atol = one_atol;
        rtol = one_rtol;
    }
}

/*! \brief adds one matrix's AllcloseFailure, found in batch `batch`, to a running total */
inline void allclose_merge_failure(AllcloseFailure& total, AllcloseFailure one, int64_t batch)
{
    total.compared_count += one.compared_count;
    if(one.bad_count == 0)
        return;

    // A NaN error ranks worst, and the first NaN found stays.
    const bool worse = total.bad_count == 0 || (std::isnan(one.error) && !std::isnan(total.error))
                       || one.error > total.error;
    total.bad_count += one.bad_count;
    if(worse)
    {
        one.batch          = batch;
        one.bad_count      = total.bad_count;
        one.compared_count = total.compared_count;
        total              = one;
    }
}

inline std::string allclose_failure_message(const AllcloseFailure& f)
{
    const double       loosest = allclose_tolerances().back();
    std::ostringstream s;
    s << std::setprecision(9) << f.bad_count << " of " << f.compared_count
      << " elements are outside atol " << loosest << " + rtol " << loosest
      << " * |result|; the worst is batch " << f.batch << ", row " << f.row << ", column " << f.col
      << (f.imag ? " (imaginary part)" : "") << ": result " << f.got << ", reference "
      << f.expected;
    return s.str();
}

template <typename T>
bool allclose(size_t* N, T* a, T* b, double atol, double rtol, bool equal_nan = false)
{
    for(size_t i = 0; i < *N; i++)
    {
        //Returning ture immediately if 2 elements are identical.
        //This also can handle inf == inf case
        if(a[i] == b[i])
            continue;
        //NaNs compare equal when equal_nan is true
        if(equal_nan && (std::isnan(a[i]) && std::isnan(b[i])))
            continue;
        if(equal_nan && (std::isnan(a[i]) ^ std::isnan(b[i])))
            return false;

        if(!allclose_within(a[i], b[i], atol, rtol))
            return false;
    }
    return true;
}

/*! \brief searches the tolerance pairs for one M x N column-major matrix, already converted to
 *  double. Columns from n_real onward hold the imaginary parts of a complex matrix. */
inline bool allclose_search(int64_t          M,
                            int64_t          N,
                            int64_t          lda,
                            int64_t          n_real,
                            double*          hCPU,
                            double*          hGPU,
                            double&          hipblaslt_atol,
                            double&          hipblaslt_rtol,
                            AllcloseFailure* failure)
{
    if(failure)
        *failure = {};
    if(M * N == 0)
    {
        hipblaslt_atol = 0;
        hipblaslt_rtol = 0;
        return true;
    }

    size_t size       = N * (size_t)lda;
    double atol_found = 1;
    double rtol_found = 1;
    for(double atol : allclose_tolerances())
    {
        for(double rtol : allclose_tolerances())
        {
            if(allclose(&size, hCPU, hGPU, atol, rtol, false))
            {
                atol_found = atol;
                rtol_found = rtol;
                break;
            }
        }
        if(atol_found != 1)
            break;
    }
    hipblaslt_atol = atol_found;
    hipblaslt_rtol = rtol_found;

    if(failure)
    {
        failure->compared_count = size_t(M) * size_t(N);
        if(atol_found == 1)
        {
            const double loosest = allclose_tolerances().back();
            for(int64_t col = 0; col < N; col++)
            {
                for(int64_t row = 0; row < M; row++)
                {
                    const size_t idx = row + col * (size_t)lda;
                    const double a   = hCPU[idx];
                    const double b   = hGPU[idx];
                    if(allclose_within(a, b, loosest, loosest))
                        continue;
                    const double error = std::abs(a - b);
                    const bool   worse = failure->bad_count == 0
                                       || (std::isnan(error) && !std::isnan(failure->error))
                                       || error > failure->error;
                    failure->bad_count++;
                    if(worse)
                    {
                        failure->row      = row;
                        failure->imag     = col >= n_real;
                        failure->col      = failure->imag ? col - n_real : col;
                        failure->expected = a;
                        failure->got      = b;
                        failure->error    = error;
                    }
                }
            }
        }
    }
    return atol_found != 1;
}

/* ============== Allclose Check for General Matrix ============= */
/*! \brief compare the allclose error of two matrices hCPU & hGPU */

template <
    typename T,
    std::enable_if_t<!(std::is_same<T, hipblaslt_f8_fnuz>{} || std::is_same<T, hipblaslt_bf8_fnuz>{}
                       || std::is_same<T, hipblaslt_f8>{} || std::is_same<T, hipblaslt_bf8>{}
                       || std::is_same<T, std::complex<float>>{}
                       || std::is_same<T, std::complex<double>>{}),
                     int>
    = 0>
bool allclose_check_general(char             allclose_type,
                            int64_t          M,
                            int64_t          N,
                            int64_t          lda,
                            T*               hCPU,
                            T*               hGPU,
                            double&          hipblaslt_atol,
                            double&          hipblaslt_rtol,
                            AllcloseFailure* failure = nullptr)
{
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    return allclose_search(M,
                           N,
                           lda,
                           N,
                           hCPU_double.data(),
                           hGPU_double.data(),
                           hipblaslt_atol,
                           hipblaslt_rtol,
                           failure);
}

template <
    typename T,
    std::enable_if_t<(std::is_same<T, hipblaslt_f8_fnuz>{} || std::is_same<T, hipblaslt_bf8_fnuz>{}
                      || std::is_same<T, hipblaslt_f8>{} || std::is_same<T, hipblaslt_bf8>{}),
                     int>
    = 0>
bool allclose_check_general(char             allclose_type,
                            int64_t          M,
                            int64_t          N,
                            int64_t          lda,
                            T*               hCPU,
                            T*               hGPU,
                            double&          hipblaslt_atol,
                            double&          hipblaslt_rtol,
                            AllcloseFailure* failure = nullptr)
{
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(float(hCPU[idx]));
            hGPU_double[idx] = double(float(hGPU[idx]));
        }
    }

    return allclose_search(M,
                           N,
                           lda,
                           N,
                           hCPU_double.data(),
                           hGPU_double.data(),
                           hipblaslt_atol,
                           hipblaslt_rtol,
                           failure);
}

// For BF16 and half, we convert the results to double first
template <typename T,
          typename VEC,
          std::enable_if_t<std::is_same<T, hipblasLtHalf>{} || std::is_same<T, hip_bfloat16>{}, int>
          = 0>
bool allclose_check_general(char             allclose_type,
                            int64_t          M,
                            int64_t          N,
                            int64_t          lda,
                            VEC&&            hCPU,
                            T*               hGPU,
                            double&          hipblaslt_atol,
                            double&          hipblaslt_rtol,
                            AllcloseFailure* failure = nullptr)
{
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = hCPU[idx];
            hGPU_double[idx] = hGPU[idx];
        }
    }

    return allclose_check_general<double>(allclose_type,
                                          M,
                                          N,
                                          lda,
                                          hCPU_double,
                                          hGPU_double,
                                          hipblaslt_atol,
                                          hipblaslt_rtol,
                                          failure);
}

// For int8, we convert the results to int first
template <typename T, typename VEC, std::enable_if_t<std::is_same<T, int8_t>{}, int> = 0>
bool allclose_check_general(char             allclose_type,
                            int64_t          M,
                            int64_t          N,
                            int64_t          lda,
                            VEC&&            hCPU,
                            T*               hGPU,
                            double&          hipblaslt_atol,
                            double&          hipblaslt_rtol,
                            AllcloseFailure* failure = nullptr)
{
    size_t           size = N * (size_t)lda;
    host_vector<int> hCPU_int(size);
    host_vector<int> hGPU_int(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx    = j + i * (size_t)lda;
            hCPU_int[idx] = hCPU[idx];
            hGPU_int[idx] = hGPU[idx];
        }
    }

    return allclose_check_general<int>(
        allclose_type, M, N, lda, hCPU_int, hGPU_int, hipblaslt_atol, hipblaslt_rtol, failure);
}

//For complex datatypes
template <typename T,
          // Enable this function if T is std::complex<float> OR std::complex<double>
          std::enable_if_t<std::is_same_v<T, std::complex<float>>
                               || std::is_same_v<T, std::complex<double>>,
                           int>
          = 0>
bool allclose_check_general(char             allclose_type,
                            int64_t          M,
                            int64_t          N,
                            int64_t          lda,
                            T*               hCPU,
                            T*               hGPU,
                            double&          hipblaslt_atol,
                            double&          hipblaslt_rtol,
                            AllcloseFailure* failure = nullptr)
{
    // The real parts fill columns 0 to N - 1 and the imaginary parts columns N to 2N - 1.
    size_t size = N * (size_t)lda;

    host_vector<double> hCPU_real_and_imag(2 * size);
    host_vector<double> hGPU_real_and_imag(2 * size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t complex_idx = j + i * (size_t)lda;
            size_t real_idx    = complex_idx;

            size_t imag_idx = complex_idx + size;

            hCPU_real_and_imag[real_idx] = static_cast<double>(hCPU[complex_idx].real());
            hGPU_real_and_imag[real_idx] = static_cast<double>(hGPU[complex_idx].real());

            hCPU_real_and_imag[imag_idx] = static_cast<double>(hCPU[complex_idx].imag());
            hGPU_real_and_imag[imag_idx] = static_cast<double>(hGPU[complex_idx].imag());
        }
    }

    return allclose_search(M,
                           2 * N,
                           lda,
                           N,
                           hCPU_real_and_imag.data(),
                           hGPU_real_and_imag.data(),
                           hipblaslt_atol,
                           hipblaslt_rtol,
                           failure);
}

/* ============== allclose check for strided_batched case ============= */
template <typename T, typename T_hpa>
bool allclose_check_general(char             allclose_type,
                            int64_t          M,
                            int64_t          N,
                            int64_t          lda,
                            int64_t          stride_a,
                            T_hpa*           hCPU,
                            T*               hGPU,
                            int64_t          batch_count,
                            double&          hipblaslt_atol,
                            double&          hipblaslt_rtol,
                            AllcloseFailure* failure = nullptr)
{
    double          atol = 0;
    double          rtol = 0;
    AllcloseFailure total;
    for(int64_t i = 0; i < batch_count; i++)
    {
        auto            index = i * stride_a;
        double          one_atol;
        double          one_rtol;
        AllcloseFailure one;
        allclose_check_general(allclose_type,
                               M,
                               N,
                               lda,
                               hCPU + index,
                               hGPU + index,
                               one_atol,
                               one_rtol,
                               failure ? &one : nullptr);
        allclose_merge_pair(atol, rtol, one_atol, one_rtol);
        if(failure)
            allclose_merge_failure(total, one, i);
        else if(atol == 1)
            break;
    }

    hipblaslt_atol = atol;
    hipblaslt_rtol = rtol;
    if(failure)
        *failure = total;
    return atol != 1;
}

/* ============== allclose check for batched case ============= */
template <typename T>
bool allclose_check_general(char             allclose_type,
                            int64_t          M,
                            int64_t          N,
                            int64_t          lda,
                            T*               hCPU[],
                            T*               hGPU[],
                            int64_t          batch_count,
                            double&          hipblaslt_atol,
                            double&          hipblaslt_rtol,
                            AllcloseFailure* failure = nullptr)
{
    double          atol = 0;
    double          rtol = 0;
    AllcloseFailure total;
    for(int64_t i = 0; i < batch_count; i++)
    {
        double          one_atol;
        double          one_rtol;
        AllcloseFailure one;
        allclose_check_general<T>(allclose_type,
                                  M,
                                  N,
                                  lda,
                                  hCPU[i],
                                  hGPU[i],
                                  one_atol,
                                  one_rtol,
                                  failure ? &one : nullptr);
        allclose_merge_pair(atol, rtol, one_atol, one_rtol);
        if(failure)
            allclose_merge_failure(total, one, i);
        else if(atol == 1)
            break;
    }

    hipblaslt_atol = atol;
    hipblaslt_rtol = rtol;
    if(failure)
        *failure = total;
    return atol != 1;
}

inline bool allclose_check_general(char             allclose_type,
                                   int64_t          M,
                                   int64_t          N,
                                   int64_t          lda,
                                   int64_t          stride_a,
                                   void*            hCPU,
                                   void*            hGPU,
                                   int64_t          batch_count,
                                   double&          hipblaslt_atol,
                                   double&          hipblaslt_rtol,
                                   hipDataType      type,
                                   AllcloseFailure* failure = nullptr)
{
    auto check = [&](auto* cpu, auto* gpu) {
        return allclose_check_general(allclose_type,
                                      M,
                                      N,
                                      lda,
                                      stride_a,
                                      cpu,
                                      gpu,
                                      batch_count,
                                      hipblaslt_atol,
                                      hipblaslt_rtol,
                                      failure);
    };

    switch(type)
    {
    case HIP_R_32F:
        return check(static_cast<float*>(hCPU), static_cast<float*>(hGPU));
    case HIP_R_64F:
        return check(static_cast<double*>(hCPU), static_cast<double*>(hGPU));
    case HIP_C_32F:
        return check(static_cast<std::complex<float>*>(hCPU),
                     static_cast<std::complex<float>*>(hGPU));
    case HIP_C_64F:
        return check(static_cast<std::complex<double>*>(hCPU),
                     static_cast<std::complex<double>*>(hGPU));
    case HIP_R_16F:
        return check(static_cast<hipblasLtHalf*>(hCPU), static_cast<hipblasLtHalf*>(hGPU));
    case HIP_R_16BF:
        return check(static_cast<hip_bfloat16*>(hCPU), static_cast<hip_bfloat16*>(hGPU));
    case HIP_R_8F_E4M3_FNUZ:
        return check(static_cast<hipblaslt_f8_fnuz*>(hCPU), static_cast<hipblaslt_f8_fnuz*>(hGPU));
    case HIP_R_8F_E5M2_FNUZ:
        return check(static_cast<hipblaslt_bf8_fnuz*>(hCPU),
                     static_cast<hipblaslt_bf8_fnuz*>(hGPU));
    case HIP_R_8F_E4M3:
        return check(static_cast<hipblaslt_f8*>(hCPU), static_cast<hipblaslt_f8*>(hGPU));
    case HIP_R_8F_E5M2:
        return check(static_cast<hipblaslt_bf8*>(hCPU), static_cast<hipblaslt_bf8*>(hGPU));
    case HIP_R_32I:
        return check(static_cast<int32_t*>(hCPU), static_cast<int32_t*>(hGPU));
    case HIP_R_8I:
        return check(static_cast<hipblasLtInt8*>(hCPU), static_cast<hipblasLtInt8*>(hGPU));
    default:
        hipblaslt_cerr << "Error type in allclose_check_general" << std::endl;
        return false;
    }
}
