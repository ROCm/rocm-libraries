/* ************************************************************************
* Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "utility.h"

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreate((cusparseHandle_t*)handle));
}

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroy((cusparseHandle_t)handle));
}

#if CUDART_VERSION > 10000
const char* hipsparseGetErrorName(hipsparseStatus_t status)
{
    return cusparseGetErrorName(hipsparse::hipSPARSEStatusToCUSPARSEStatus(status));
}

const char* hipsparseGetErrorString(hipsparseStatus_t status)
{
    return cusparseGetErrorString(hipsparse::hipSPARSEStatusToCUSPARSEStatus(status));
}
#endif

hipsparseStatus_t hipsparseGetVersion(hipsparseHandle_t handle, int* version)
{
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    }

    *version = hipsparseVersionMajor * 100000 + hipsparseVersionMinor * 100 + hipsparseVersionPatch;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseGetGitRevision(hipsparseHandle_t handle, char* rev)
{
    // Get hipSPARSE revision
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    }

    if(rev == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    static constexpr char v[] = TO_STR(hipsparseVersionTweak);

    char hipsparse_rev[64];
    memcpy(hipsparse_rev, v, sizeof(v));

    // Get cuSPARSE version
    int cusparse_ver;
    RETURN_IF_CUSPARSE_ERROR(cusparseGetVersion((cusparseHandle_t)handle, &cusparse_ver));

    // Combine
    snprintf(rev,
             256,
             "%.64s (cuSPARSE %d.%d.%d)",
             hipsparse_rev,
             cusparse_ver / 100000,
             cusparse_ver / 100 % 1000,
             cusparse_ver % 100);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSetStream((cusparseHandle_t)handle, streamId));
}

hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseGetStream((cusparseHandle_t)handle, streamId));
}

hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t mode)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSetPointerMode(
        (cusparseHandle_t)handle, hipsparse::hipPointerModeToCudaPointerMode(mode)));
}

hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t* mode)
{
    cusparsePointerMode_t cusparseMode;
    cusparseStatus_t      status = cusparseGetPointerMode((cusparseHandle_t)handle, &cusparseMode);
    *mode                        = hipsparse::CudaPointerModeToHIPPointerMode(cusparseMode);
    return hipsparse::hipCUSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateMatDescr((cusparseMatDescr_t*)descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyMatDescr((cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA, hipsparseMatrixType_t type)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSetMatType(
        (cusparseMatDescr_t)descrA, hipsparse::hipMatrixTypeToCudaMatrixType(type)));
}

hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA)
{
    return hipsparse::CudaMatrixTypeToHIPMatrixType(
        cusparseGetMatType((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatFillMode(hipsparseMatDescr_t descrA, hipsparseFillMode_t fillMode)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSetMatFillMode((cusparseMatDescr_t)descrA, hipsparse::hipFillToCudaFill(fillMode)));
}

hipsparseFillMode_t hipsparseGetMatFillMode(const hipsparseMatDescr_t descrA)
{
    return hipsparse::CudaFillToHIPFill(cusparseGetMatFillMode((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatDiagType(hipsparseMatDescr_t descrA, hipsparseDiagType_t diagType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSetMatDiagType(
        (cusparseMatDescr_t)descrA, hipsparse::hipDiagonalToCudaDiagonal(diagType)));
}

hipsparseDiagType_t hipsparseGetMatDiagType(const hipsparseMatDescr_t descrA)
{
    return hipsparse::CudaDiagonalToHIPDiagonal(
        cusparseGetMatDiagType((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, hipsparseIndexBase_t base)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSetMatIndexBase(
        (cusparseMatDescr_t)descrA, hipsparse::hipIndexBaseToCudaIndexBase(base)));
}

hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA)
{
    return hipsparse::CudaIndexBaseToHIPIndexBase(
        cusparseGetMatIndexBase((const cusparseMatDescr_t)descrA));
}

#if CUDART_VERSION < 11000
hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreateHybMat((cusparseHybMat_t*)hybA));
}

hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyHybMat((cusparseHybMat_t)hybA));
}

// cuSPARSE exposes no public accessor for its opaque cusparseHybMat_t, so this
// mirrors its internal layout the same way the pre-11.0 cuSPARSE HYB routines
// have always been assumed to lay it out. Kept private to this translation
// unit: callers only ever go through hipsparseHybMatGetInfo/SetInfo.
struct cusparse_hyb_mat_mirror
{
    int                    m;
    int                    n;
    cusparseHybPartition_t partition;
    int                    ell_nnz;
    int                    ell_width;
    int*                   ell_col_ind;
    void*                  ell_val;
    int                    coo_nnz;
    int*                   coo_row_ind;
    int*                   coo_col_ind;
    void*                  coo_val;
};

hipsparseStatus_t hipsparseHybMatGetInfo(const hipsparseHybMat_t  hyb,
                                         int*                     m,
                                         int*                     n,
                                         hipsparseHybPartition_t* partition,
                                         int64_t*                 ell_nnz,
                                         int*                     ell_width,
                                         const int**              ell_col_ind,
                                         const void**             ell_val,
                                         int*                     coo_nnz,
                                         const int**              coo_row_ind,
                                         const int**              coo_col_ind,
                                         const void**             coo_val)
{
    if(hyb == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    const cusparse_hyb_mat_mirror* dhyb = (const cusparse_hyb_mat_mirror*)hyb;

    if(m != nullptr)
    {
        *m = dhyb->m;
    }
    if(n != nullptr)
    {
        *n = dhyb->n;
    }
    if(partition != nullptr)
    {
        *partition = hipsparse::cudaHybPartitionToHipHybPartition(dhyb->partition);
    }
    if(ell_nnz != nullptr)
    {
        *ell_nnz = dhyb->ell_nnz;
    }
    if(ell_width != nullptr)
    {
        *ell_width = dhyb->ell_width;
    }
    if(ell_col_ind != nullptr)
    {
        *ell_col_ind = dhyb->ell_col_ind;
    }
    if(ell_val != nullptr)
    {
        *ell_val = dhyb->ell_val;
    }
    if(coo_nnz != nullptr)
    {
        *coo_nnz = dhyb->coo_nnz;
    }
    if(coo_row_ind != nullptr)
    {
        *coo_row_ind = dhyb->coo_row_ind;
    }
    if(coo_col_ind != nullptr)
    {
        *coo_col_ind = dhyb->coo_col_ind;
    }
    if(coo_val != nullptr)
    {
        *coo_val = dhyb->coo_val;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseHybMatSetInfo(hipsparseHybMat_t              hyb,
                                         const int*                     m,
                                         const int*                     n,
                                         const hipsparseHybPartition_t* partition,
                                         const int64_t*                 ell_nnz,
                                         const int*                     ell_width,
                                         int* const*                    ell_col_ind,
                                         void* const*                   ell_val,
                                         const int*                     coo_nnz,
                                         int* const*                    coo_row_ind,
                                         int* const*                    coo_col_ind,
                                         void* const*                   coo_val)
{
    if(hyb == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    cusparse_hyb_mat_mirror* dhyb = (cusparse_hyb_mat_mirror*)hyb;

    if(m != nullptr)
    {
        dhyb->m = *m;
    }
    if(n != nullptr)
    {
        dhyb->n = *n;
    }
    if(partition != nullptr)
    {
        dhyb->partition = hipsparse::hipHybPartitionToCudaHybPartition(*partition);
    }
    if(ell_nnz != nullptr)
    {
        dhyb->ell_nnz = static_cast<int>(*ell_nnz);
    }
    if(ell_width != nullptr)
    {
        dhyb->ell_width = *ell_width;
    }
    if(ell_col_ind != nullptr)
    {
        dhyb->ell_col_ind = *ell_col_ind;
    }
    if(ell_val != nullptr)
    {
        dhyb->ell_val = *ell_val;
    }
    if(coo_nnz != nullptr)
    {
        dhyb->coo_nnz = *coo_nnz;
    }
    if(coo_row_ind != nullptr)
    {
        dhyb->coo_row_ind = *coo_row_ind;
    }
    if(coo_col_ind != nullptr)
    {
        dhyb->coo_col_ind = *coo_col_ind;
    }
    if(coo_val != nullptr)
    {
        dhyb->coo_val = *coo_val;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if CUDART_VERSION < 14000
hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreateBsrsv2Info((bsrsv2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyBsrsv2Info((bsrsv2Info_t)info));
}
#endif

#if CUDART_VERSION < 14000
hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreateBsrsm2Info((bsrsm2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyBsrsm2Info((bsrsm2Info_t)info));
}
#endif

#if CUDART_VERSION < 14000
hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateBsrilu02Info((bsrilu02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyBsrilu02Info((bsrilu02Info_t)info));
}
#endif

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrsv2Info((csrsv2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrsv2Info((csrsv2Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrsm2Info((csrsm2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrsm2Info((csrsm2Info_t)info));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseCreateColorInfo(hipsparseColorInfo_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateColorInfo((cusparseColorInfo_t*)info));
}

hipsparseStatus_t hipsparseDestroyColorInfo(hipsparseColorInfo_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyColorInfo((cusparseColorInfo_t)info));
}
#endif

#if CUDART_VERSION < 14000
hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCsrilu02Info((csrilu02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyCsrilu02Info((csrilu02Info_t)info));
}
#endif

#if CUDART_VERSION < 14000
hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreateBsric02Info((bsric02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyBsric02Info((bsric02Info_t)info));
}
#endif

#if CUDART_VERSION < 14000
hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreateCsric02Info((csric02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsric02Info((csric02Info_t)info));
}
#endif

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCsrgemm2Info((csrgemm2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyCsrgemm2Info((csrgemm2Info_t)info));
}
#endif

#if CUDART_VERSION < 14000
hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreatePruneInfo((pruneInfo_t*)info));
}

hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyPruneInfo((pruneInfo_t)info));
}
#endif
