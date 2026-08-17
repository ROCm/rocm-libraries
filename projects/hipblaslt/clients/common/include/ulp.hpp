/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
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

#include "hipblaslt_ostream.hpp"
#include "hipblaslt_vector.hpp"
#include "utility.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <hipblaslt/hipblaslt.h>
#include <limits>
#include <cfloat>

/* =====================================================================
        ULP check: per-element error measured in units in the last place
        of the output datatype. Reports the maximum and the average ULP
        error between two results (usually CPU reference and GPU result).
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides ULP check
 */

/*! \brief mantissa (fraction) bit count for the output datatype, used to size
 *         the spacing between representable values when computing ULP error.
 *         A return of 0 marks an integer type, where one ULP == 1. */
inline int ulp_mantissa_bits(hipDataType type)
{
    switch(type)
    {
    case HIP_R_64F:
    case HIP_C_64F:
        return 52;
    case HIP_R_32F:
    case HIP_C_32F:
        return 23;
    case HIP_R_16F:
        return 10;
    case HIP_R_16BF:
        return 7;
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E4M3:
        return 3;
    case HIP_R_8F_E5M2_FNUZ:
    case HIP_R_8F_E5M2:
        return 2;
    case HIP_R_4F_E2M1:
        return 1;
    case HIP_R_32I:
    case HIP_R_8I:
        return 0; // integer types: 1 ULP == 1
    default:
        return 23;
    }
}

/*! \brief widen a stored element to double without losing precision.
 *         Low-precision types are promoted through float first. */
inline double ulp_as_double(float x)
{
    return static_cast<double>(x);
}
inline double ulp_as_double(double x)
{
    return x;
}
inline double ulp_as_double(hipblasLtHalf x)
{
    return static_cast<double>(static_cast<float>(x));
}
inline double ulp_as_double(hip_bfloat16 x)
{
    return static_cast<double>(static_cast<float>(x));
}
inline double ulp_as_double(hipblaslt_f8_fnuz x)
{
    return static_cast<double>(static_cast<float>(x));
}
inline double ulp_as_double(hipblaslt_bf8_fnuz x)
{
    return static_cast<double>(static_cast<float>(x));
}
inline double ulp_as_double(hipblaslt_f8 x)
{
    return static_cast<double>(static_cast<float>(x));
}
inline double ulp_as_double(hipblaslt_bf8 x)
{
    return static_cast<double>(static_cast<float>(x));
}
inline double ulp_as_double(int32_t x)
{
    return static_cast<double>(x);
}
inline double ulp_as_double(hipblasLtInt8 x)
{
    return static_cast<double>(x);
}

/*! \brief calculate the distance in units of ULP between two double values */
inline double ulp_distance(double exact, double approximation, int mant_bits)
{
    // 1. Handle identical values immediately to prevent 0/0 errors
    if (exact == approximation) return 0.0;

    int e = 0;
    double mantissa = std::frexp(exact, &e);

    // 2. Fix the power-of-2 boundary condition
    if (std::abs(mantissa) == 0.5) {
        e--;
    }

    // 3. Compute step size using the output type's mantissa width (mant_bits is
    //    passed in by the caller, e.g. 24 for f32 / 53 for f64); the ULP step
    //    must reflect the result precision, not always double.
    const double ulp_size = std::ldexp(1.0, e - mant_bits);
    
    // 4. Return the distance in units of ULP
    double diff = std::abs(exact - approximation);
    return diff / ulp_size;
}

/* ============== ULP Check for strided_batched case ============= */
/*! \brief accumulate max and sum of ULP error over a strided batched matrix */
template <typename T>
void ulp_accumulate_general(int64_t M,
                            int64_t N,
                            int64_t lda,
                            int64_t stride_a,
                            T*      hCPU,
                            T*      hGPU,
                            int64_t batch_count,
                            int     mant_bits,
                            double& max_ulp,
                            double& sum_ulp,
                            size_t& count)
{
    if(M * N == 0)
        return;

    for(int64_t b = 0; b < batch_count; b++)
    {
        const int64_t base = b * stride_a;
        for(int64_t i = 0; i < N; i++)
        {
            for(int64_t j = 0; j < M; j++)
            {
                const size_t idx = base + j + i * (size_t)lda;
                const double a   = ulp_as_double(hCPU[idx]);
                const double g   = ulp_as_double(hGPU[idx]);
                const double u   = ulp_distance(a, g, mant_bits);
                max_ulp          = std::max(max_ulp, u);
                sum_ulp += u;
                count++;
            }
        }
    }
}

/*! \brief type-dispatching ULP check for a strided batched matrix.
 *         Updates max_ulp (running maximum), sum_ulp (running sum) and count
 *         (number of compared elements) so the caller can derive the average. */
inline void ulp_check_general(int64_t     M,
                              int64_t     N,
                              int64_t     lda,
                              int64_t     stride_a,
                              void*       hCPU,
                              void*       hGPU,
                              int64_t     batch_count,
                              double&     max_ulp,
                              double&     sum_ulp,
                              size_t&     count,
                              hipDataType type)
{
    const int mant_bits = ulp_mantissa_bits(type);
    switch(type)
    {
    case HIP_R_32F:
        ulp_accumulate_general<float>(M,
                                      N,
                                      lda,
                                      stride_a,
                                      static_cast<float*>(hCPU),
                                      static_cast<float*>(hGPU),
                                      batch_count,
                                      mant_bits,
                                      max_ulp,
                                      sum_ulp,
                                      count);
        break;
    case HIP_R_64F:
        ulp_accumulate_general<double>(M,
                                       N,
                                       lda,
                                       stride_a,
                                       static_cast<double*>(hCPU),
                                       static_cast<double*>(hGPU),
                                       batch_count,
                                       mant_bits,
                                       max_ulp,
                                       sum_ulp,
                                       count);
        break;
    case HIP_R_16F:
        ulp_accumulate_general<hipblasLtHalf>(M,
                                              N,
                                              lda,
                                              stride_a,
                                              static_cast<hipblasLtHalf*>(hCPU),
                                              static_cast<hipblasLtHalf*>(hGPU),
                                              batch_count,
                                              mant_bits,
                                              max_ulp,
                                              sum_ulp,
                                              count);
        break;
    case HIP_R_16BF:
        ulp_accumulate_general<hip_bfloat16>(M,
                                             N,
                                             lda,
                                             stride_a,
                                             static_cast<hip_bfloat16*>(hCPU),
                                             static_cast<hip_bfloat16*>(hGPU),
                                             batch_count,
                                             mant_bits,
                                             max_ulp,
                                             sum_ulp,
                                             count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        ulp_accumulate_general<hipblaslt_f8_fnuz>(M,
                                                  N,
                                                  lda,
                                                  stride_a,
                                                  static_cast<hipblaslt_f8_fnuz*>(hCPU),
                                                  static_cast<hipblaslt_f8_fnuz*>(hGPU),
                                                  batch_count,
                                                  mant_bits,
                                                  max_ulp,
                                                  sum_ulp,
                                                  count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        ulp_accumulate_general<hipblaslt_bf8_fnuz>(M,
                                                   N,
                                                   lda,
                                                   stride_a,
                                                   static_cast<hipblaslt_bf8_fnuz*>(hCPU),
                                                   static_cast<hipblaslt_bf8_fnuz*>(hGPU),
                                                   batch_count,
                                                   mant_bits,
                                                   max_ulp,
                                                   sum_ulp,
                                                   count);
        break;
    case HIP_R_8F_E4M3:
        ulp_accumulate_general<hipblaslt_f8>(M,
                                             N,
                                             lda,
                                             stride_a,
                                             static_cast<hipblaslt_f8*>(hCPU),
                                             static_cast<hipblaslt_f8*>(hGPU),
                                             batch_count,
                                             mant_bits,
                                             max_ulp,
                                             sum_ulp,
                                             count);
        break;
    case HIP_R_8F_E5M2:
        ulp_accumulate_general<hipblaslt_bf8>(M,
                                              N,
                                              lda,
                                              stride_a,
                                              static_cast<hipblaslt_bf8*>(hCPU),
                                              static_cast<hipblaslt_bf8*>(hGPU),
                                              batch_count,
                                              mant_bits,
                                              max_ulp,
                                              sum_ulp,
                                              count);
        break;
    case HIP_R_32I:
        ulp_accumulate_general<int32_t>(M,
                                        N,
                                        lda,
                                        stride_a,
                                        static_cast<int32_t*>(hCPU),
                                        static_cast<int32_t*>(hGPU),
                                        batch_count,
                                        mant_bits,
                                        max_ulp,
                                        sum_ulp,
                                        count);
        break;
    case HIP_R_8I:
        ulp_accumulate_general<hipblasLtInt8>(M,
                                              N,
                                              lda,
                                              stride_a,
                                              static_cast<hipblasLtInt8*>(hCPU),
                                              static_cast<hipblasLtInt8*>(hGPU),
                                              batch_count,
                                              mant_bits,
                                              max_ulp,
                                              sum_ulp,
                                              count);
        break;
    default:
        hipblaslt_cerr << "Error type in ulp_check_general" << std::endl;
        break;
    }
}
