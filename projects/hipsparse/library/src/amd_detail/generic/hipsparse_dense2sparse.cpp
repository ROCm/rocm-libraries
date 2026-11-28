/*! \file */
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

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../utility.h"

hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseConstDnMatDescr_t  matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes)
{
    rocsparse_dense_to_sparse_alg_ backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipDnToSpAlgToHCCDnToSpAlg(alg, &backend_alg));

    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dense_to_sparse((rocsparse_handle)handle,
                                  (rocsparse_const_dnmat_descr)matA,
                                  to_rocsparse_spmat_descr(matB),
                                  backend_alg,
                                  pBufferSizeInBytes,
                                  nullptr));
}

hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseConstDnMatDescr_t  matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer)
{
    rocsparse_dense_to_sparse_alg_ backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipDnToSpAlgToHCCDnToSpAlg(alg, &backend_alg));

    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dense_to_sparse((rocsparse_handle)handle,
                                  (rocsparse_const_dnmat_descr)matA,
                                  to_rocsparse_spmat_descr(matB),
                                  backend_alg,
                                  nullptr,
                                  externalBuffer));
}

hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseConstDnMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer)
{
    rocsparse_dense_to_sparse_alg_ backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipDnToSpAlgToHCCDnToSpAlg(alg, &backend_alg));

    size_t bufferSize = 4;
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dense_to_sparse((rocsparse_handle)handle,
                                  (rocsparse_const_dnmat_descr)matA,
                                  to_rocsparse_spmat_descr(matB),
                                  backend_alg,
                                  externalBuffer != nullptr ? &bufferSize : nullptr,
                                  externalBuffer));
}
