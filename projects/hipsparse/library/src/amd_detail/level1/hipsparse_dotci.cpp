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

hipsparseStatus_t hipsparseCdotci(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const hipComplex*    xVal,
                                  const int*           xInd,
                                  const hipComplex*    y,
                                  hipComplex*          result,
                                  hipsparseIndexBase_t idxBase)
{
    // Obtain stream, to explicitly sync (cusparse dotci is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Dotci
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cdotci((rocsparse_handle)handle,
                                               nnz,
                                               (const rocsparse_float_complex*)xVal,
                                               xInd,
                                               (const rocsparse_float_complex*)y,
                                               (rocsparse_float_complex*)result,
                                               hipsparse::hipBaseToHCCBase(idxBase)));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZdotci(hipsparseHandle_t       handle,
                                  int                     nnz,
                                  const hipDoubleComplex* xVal,
                                  const int*              xInd,
                                  const hipDoubleComplex* y,
                                  hipDoubleComplex*       result,
                                  hipsparseIndexBase_t    idxBase)
{
    // Obtain stream, to explicitly sync (cusparse dotci is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Dotci
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zdotci((rocsparse_handle)handle,
                                               nnz,
                                               (const rocsparse_double_complex*)xVal,
                                               xInd,
                                               (const rocsparse_double_complex*)y,
                                               (rocsparse_double_complex*)result,
                                               hipsparse::hipBaseToHCCBase(idxBase)));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}
