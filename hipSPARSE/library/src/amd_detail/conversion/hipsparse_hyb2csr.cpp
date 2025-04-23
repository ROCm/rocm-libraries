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

hipsparseStatus_t hipsparseShyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    float*                    csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_hyb2csr_buffer_size((rocsparse_handle)handle,
                                                            (const rocsparse_mat_descr)descrA,
                                                            (const rocsparse_hyb_mat)hybA,
                                                            csrSortedRowPtrA,
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_shyb2csr((rocsparse_handle)handle,
                           (const rocsparse_mat_descr)descrA,
                           (const rocsparse_hyb_mat)hybA,
                           csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return status;
}

hipsparseStatus_t hipsparseDhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    double*                   csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_hyb2csr_buffer_size((rocsparse_handle)handle,
                                                            (const rocsparse_mat_descr)descrA,
                                                            (const rocsparse_hyb_mat)hybA,
                                                            csrSortedRowPtrA,
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dhyb2csr((rocsparse_handle)handle,
                           (const rocsparse_mat_descr)descrA,
                           (const rocsparse_hyb_mat)hybA,
                           csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return status;
}

hipsparseStatus_t hipsparseChyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipComplex*               csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_hyb2csr_buffer_size((rocsparse_handle)handle,
                                                            (const rocsparse_mat_descr)descrA,
                                                            (const rocsparse_hyb_mat)hybA,
                                                            csrSortedRowPtrA,
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_chyb2csr((rocsparse_handle)handle,
                           (const rocsparse_mat_descr)descrA,
                           (const rocsparse_hyb_mat)hybA,
                           (rocsparse_float_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return status;
}

hipsparseStatus_t hipsparseZhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipDoubleComplex*         csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_hyb2csr_buffer_size((rocsparse_handle)handle,
                                                            (const rocsparse_mat_descr)descrA,
                                                            (const rocsparse_hyb_mat)hybA,
                                                            csrSortedRowPtrA,
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zhyb2csr((rocsparse_handle)handle,
                           (const rocsparse_mat_descr)descrA,
                           (const rocsparse_hyb_mat)hybA,
                           (rocsparse_double_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return status;
}
