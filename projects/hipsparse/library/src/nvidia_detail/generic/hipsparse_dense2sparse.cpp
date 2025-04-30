/* ************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "hipsparse.h"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>

#include "../utility.h"

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseConstDnMatDescr_t  matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_bufferSize((cusparseHandle_t)handle,
                                         (cusparseConstDnMatDescr_t)matA,
                                         (cusparseSpMatDescr_t)matB,
                                         hipsparse::hipDnToSpAlgToCudaDnToSpAlg(alg),
                                         pBufferSizeInBytes));
}
#elif(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseDnMatDescr_t       matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_bufferSize((cusparseHandle_t)handle,
                                         (cusparseDnMatDescr_t)matA,
                                         (cusparseSpMatDescr_t)matB,
                                         hipsparse::hipDnToSpAlgToCudaDnToSpAlg(alg),
                                         pBufferSizeInBytes));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseConstDnMatDescr_t  matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_analysis((cusparseHandle_t)handle,
                                       (cusparseConstDnMatDescr_t)matA,
                                       (cusparseSpMatDescr_t)matB,
                                       hipsparse::hipDnToSpAlgToCudaDnToSpAlg(alg),
                                       externalBuffer));
}
#elif(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseDnMatDescr_t       matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_analysis((cusparseHandle_t)handle,
                                       (cusparseDnMatDescr_t)matA,
                                       (cusparseSpMatDescr_t)matB,
                                       hipsparse::hipDnToSpAlgToCudaDnToSpAlg(alg),
                                       externalBuffer));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseConstDnMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_convert((cusparseHandle_t)handle,
                                      (cusparseConstDnMatDescr_t)matA,
                                      (cusparseSpMatDescr_t)matB,
                                      hipsparse::hipDnToSpAlgToCudaDnToSpAlg(alg),
                                      externalBuffer));
}
#elif(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseDnMatDescr_t       matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_convert((cusparseHandle_t)handle,
                                      (cusparseDnMatDescr_t)matA,
                                      (cusparseSpMatDescr_t)matB,
                                      hipsparse::hipDnToSpAlgToCudaDnToSpAlg(alg),
                                      externalBuffer));
}
#endif
