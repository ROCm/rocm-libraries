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

// Product-private hipBLASLt adapter.

#include "datatype_interface.hpp"
#include "hipblaslt_ostream.hpp"
#include <hipblaslt/hipblaslt.h>
#include <type_traits>

/*!\file
 * \brief hipBLASLt type-erased bridge to the shared host reference GEMM.
 */

#ifdef HIPBLASLT_ENABLE_BLIS
void setup_blis();
#endif

// gemm
template <typename Tc>
void hipblaslt_reference_gemm(hipblasOperation_t       transA,
                hipblasOperation_t       transB,
                int64_t                  m,
                int64_t                  n,
                int64_t                  k,
                Tc                       alpha,
                const void*              A,
                int64_t                  lda,
                const void*              B,
                int64_t                  ldb,
                Tc                       beta,
                const void*              C,
                int64_t                  ldc,
                std::add_pointer_t<void> D,
                int64_t                  ldd,
                const void*              AlphaVec,
                const void*              scaleA,
                const void*              scaleB,
                Tc                       scaleD,
                bool                     isScaleAVec,
                bool                     isScaleBVec,
                hipDataType              tiA,
                hipDataType              tiB,
                hipDataType              tiC,
                hipDataType              to,
                hipDataType              tc,
                hipDataType              tciA,
                hipDataType              tciB,
                bool                     isScaleAMXFormat = false,
                bool                     isScaleBMXFormat = false);

inline void hipblaslt_reference_gemm(hipblasOperation_t       transA,
                       hipblasOperation_t       transB,
                       int64_t                  m,
                       int64_t                  n,
                       int64_t                  k,
                       computeTypeInterface     alpha,
                       const void*              A,
                       int64_t                  lda,
                       const void*              B,
                       int64_t                  ldb,
                       computeTypeInterface     beta,
                       const void*              C,
                       int64_t                  ldc,
                       std::add_pointer_t<void> D,
                       int64_t                  ldd,
                       const void*              AlphaVec,
                       const void*              scaleA,
                       const void*              scaleB,
                       void*                    scaleD,
                       bool                     isScaleAVec,
                       bool                     isScaleBVec,
                       hipDataType              tiA,
                       hipDataType              tiB,
                       hipDataType              tiC,
                       hipDataType              to,
                       hipDataType              tc,
                       hipDataType              tciA,
                       hipDataType              tciB,
                       bool                     isScaleAMXFormat = false,
                       bool                     isScaleBMXFormat = false)
{
#ifdef HIPBLASLT_ENABLE_BLIS
    // Runs once, lazily, on the first CPU reference call.
    [[maybe_unused]] static const bool blis_configured = [] { setup_blis(); return true; }();
#endif

    if(tiA == HIP_C_32F || tiA == HIP_C_64F)
    {
        if(tiA == HIP_C_32F)
        {
            hipblaslt_reference_gemm<std::complex<float>>(transA,
                                            transB,
                                            m,
                                            n,
                                            k,
                                            alpha.cf,
                                            A,
                                            lda,
                                            B,
                                            ldb,
                                            beta.cf,
                                            C,
                                            ldc,
                                            D,
                                            ldd,
                                            AlphaVec,
                                            scaleA,
                                            scaleB,
                                            *(std::complex<float>*)scaleD,
                                            isScaleAVec,
                                            isScaleBVec,
                                            tiA,
                                            tiB,
                                            tiC,
                                            to,
                                            tc,
                                            tiA,
                                            tiB,
                                            isScaleAMXFormat,
                                            isScaleBMXFormat);
            return;
        }
        else
        {
            hipblaslt_reference_gemm<std::complex<double>>(transA,
                                             transB,
                                             m,
                                             n,
                                             k,
                                             alpha.cd,
                                             A,
                                             lda,
                                             B,
                                             ldb,
                                             beta.cd,
                                             C,
                                             ldc,
                                             D,
                                             ldd,
                                             AlphaVec,
                                             scaleA,
                                             scaleB,
                                             *(std::complex<double>*)scaleD,
                                             isScaleAVec,
                                             isScaleBVec,
                                             tiA,
                                             tiB,
                                             tiC,
                                             to,
                                             tc,
                                             tiA,
                                             tiB);
            return;
        }
    }
    else
    {
        switch(tc)
        {
        case HIP_R_16F: // setting compute_type to f16_r will fallback to f32_r
            hipblaslt_reference_gemm<hipblasLtHalf>(transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha.f16,
                                      A,
                                      lda,
                                      B,
                                      ldb,
                                      beta.f16,
                                      C,
                                      ldc,
                                      D,
                                      ldd,
                                      AlphaVec,
                                      scaleA,
                                      scaleB,
                                      *(hipblasLtHalf*)scaleD,
                                      isScaleAVec,
                                      isScaleBVec,
                                      tiA,
                                      tiB,
                                      tiC,
                                      to,
                                      tc,
                                      tciA,
                                      tciB);
            return;
        case HIP_R_32F:
            hipblaslt_reference_gemm<float>(transA,
                              transB,
                              m,
                              n,
                              k,
                              alpha.f32,
                              A,
                              lda,
                              B,
                              ldb,
                              beta.f32,
                              C,
                              ldc,
                              D,
                              ldd,
                              AlphaVec,
                              scaleA,
                              scaleB,
                              *(float*)scaleD,
                              isScaleAVec,
                              isScaleBVec,
                              tiA,
                              tiB,
                              tiC,
                              to,
                              tc,
                              tciA,
                              tciB,
                              isScaleAMXFormat,
                              isScaleBMXFormat);

            return;
        case HIP_R_64F:
            hipblaslt_reference_gemm<double>(transA,
                               transB,
                               m,
                               n,
                               k,
                               alpha.f64,
                               A,
                               lda,
                               B,
                               ldb,
                               beta.f64,
                               C,
                               ldc,
                               D,
                               ldd,
                               AlphaVec,
                               scaleA,
                               scaleB,
                               *(double*)scaleD,
                               isScaleAVec,
                               isScaleBVec,
                               tiA,
                               tiB,
                               tiC,
                               to,
                               tc,
                               tciA,
                               tciB);
            return;
        case HIP_R_32I:
            hipblaslt_reference_gemm<int32_t>(transA,
                                transB,
                                m,
                                n,
                                k,
                                alpha.i32,
                                A,
                                lda,
                                B,
                                ldb,
                                beta.i32,
                                C,
                                ldc,
                                D,
                                ldd,
                                AlphaVec,
                                scaleA,
                                scaleB,
                                *(int32_t*)scaleD,
                                isScaleAVec,
                                isScaleBVec,
                                tiA,
                                tiB,
                                tiC,
                                to,
                                tc,
                                tciA,
                                tciB);
            return;
        default:
            hipblaslt_cerr << "Error type in hipblaslt_reference_gemm()" << std::endl;
            return;
        }
    }
}
