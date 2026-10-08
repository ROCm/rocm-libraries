/*! \file */
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

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "utility.h"
#include <cstdio>

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    int        deviceId;
    hipError_t err;

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        return hipsparse::rocSPARSEStatusToHIPStatus(
            rocsparse_create_handle((rocsparse_handle*)handle));
    }

    return hipsparse::hipErrorToHIPSPARSEStatus(err);
}

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_handle((rocsparse_handle)handle));
}

const char* hipsparseGetErrorName(hipsparseStatus_t status)
{
    return rocsparse_get_status_name(hipsparse::hipSPARSEStatusToRocSPARSEStatus(status));
}

const char* hipsparseGetErrorString(hipsparseStatus_t status)
{
    return rocsparse_get_status_description(hipsparse::hipSPARSEStatusToRocSPARSEStatus(status));
}

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

    // Get rocSPARSE revision
    char rocsparse_rev[64];
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_git_rev((rocsparse_handle)handle, rocsparse_rev));

    // Get rocSPARSE version
    int rocsparse_ver;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_version((rocsparse_handle)handle, &rocsparse_ver));

    // Combine
    snprintf(rev,
             256,
             "%.64s (rocSPARSE %d.%d.%d-%.64s)",
             hipsparse_rev,
             rocsparse_ver / 100000,
             rocsparse_ver / 100 % 1000,
             rocsparse_ver % 100,
             rocsparse_rev);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_set_stream((rocsparse_handle)handle, streamId));
}

hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_get_stream((rocsparse_handle)handle, streamId));
}

hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t mode)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_set_pointer_mode(
        (rocsparse_handle)handle, hipsparse::hipPtrModeToHCCPtrMode(mode)));
}

hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t* mode)
{
    rocsparse_pointer_mode_ rocsparse_mode;
    rocsparse_status status = rocsparse_get_pointer_mode((rocsparse_handle)handle, &rocsparse_mode);
    *mode                   = hipsparse::HCCPtrModeToHIPPtrMode(rocsparse_mode);
    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_descr((rocsparse_mat_descr*)descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_descr((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_copy_mat_descr((rocsparse_mat_descr)dest, (const rocsparse_mat_descr)src));
}

hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA, hipsparseMatrixType_t type)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_set_mat_type(
        (rocsparse_mat_descr)descrA, hipsparse::hipMatTypeToHCCMatType(type)));
}

hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA)
{
    return hipsparse::HCCMatTypeToHIPMatType(rocsparse_get_mat_type((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseSetMatFillMode(hipsparseMatDescr_t descrA, hipsparseFillMode_t fillMode)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_set_mat_fill_mode(
        (rocsparse_mat_descr)descrA, hipsparse::hipFillModeToHCCFillMode(fillMode)));
}

hipsparseFillMode_t hipsparseGetMatFillMode(const hipsparseMatDescr_t descrA)
{
    return hipsparse::HCCFillModeToHIPFillMode(
        rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseSetMatDiagType(hipsparseMatDescr_t descrA, hipsparseDiagType_t diagType)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_set_mat_diag_type(
        (rocsparse_mat_descr)descrA, hipsparse::hipDiagTypeToHCCDiagType(diagType)));
}

hipsparseDiagType_t hipsparseGetMatDiagType(const hipsparseMatDescr_t descrA)
{
    return hipsparse::HCCDiagTypeToHIPDiagType(
        rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, hipsparseIndexBase_t base)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_set_mat_index_base(
        (rocsparse_mat_descr)descrA, hipsparse::hipBaseToHCCBase(base)));
}

hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA)
{
    return hipsparse::HCCBaseToHIPBase(rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_hyb_mat((rocsparse_hyb_mat*)hybA));
}

hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_hyb_mat((rocsparse_hyb_mat)hybA));
}

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
                                         const void**             coo_val,
                                         hipDataType*             data_type)
{
    // Read into locals first. Conversions below can fail, and a failure must
    // not leave the caller's outputs partially written.
    rocsparse_hyb_partition rocsparse_partition   = rocsparse_hyb_partition_auto;
    rocsparse_int           rocsparse_m           = 0;
    rocsparse_int           rocsparse_n           = 0;
    int64_t                 rocsparse_ell_nnz     = 0;
    rocsparse_int           rocsparse_ell_width   = 0;
    const rocsparse_int*    rocsparse_ell_col_ind = nullptr;
    const void*             rocsparse_ell_val     = nullptr;
    rocsparse_int           rocsparse_coo_nnz     = 0;
    const rocsparse_int*    rocsparse_coo_row_ind = nullptr;
    const rocsparse_int*    rocsparse_coo_col_ind = nullptr;
    const void*             rocsparse_coo_val     = nullptr;
    rocsparse_datatype      rocsparse_data_type   = rocsparse_datatype_f32_r;

    hipsparseStatus_t status
        = hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_hyb_mat_get_info((rocsparse_hyb_mat)hyb,
                                                                           &rocsparse_m,
                                                                           &rocsparse_n,
                                                                           &rocsparse_partition,
                                                                           &rocsparse_ell_nnz,
                                                                           &rocsparse_ell_width,
                                                                           &rocsparse_ell_col_ind,
                                                                           &rocsparse_ell_val,
                                                                           &rocsparse_coo_nnz,
                                                                           &rocsparse_coo_row_ind,
                                                                           &rocsparse_coo_col_ind,
                                                                           &rocsparse_coo_val,
                                                                           &rocsparse_data_type));
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        return status;
    }

    hipsparseHybPartition_t hip_partition = HIPSPARSE_HYB_PARTITION_AUTO;
    if(partition != nullptr)
    {
        try
        {
            hip_partition = hipsparse::HCCHybPartToHIPHybPart(rocsparse_partition);
        }
        catch(...)
        {
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }

    hipDataType hip_data_type = HIP_R_32F;
    if(data_type != nullptr)
    {
        try
        {
            // The HYB may hold a rocSPARSE data type with no hipDataType equivalent.
            hip_data_type = hipsparse::HCCDataTypeToHIPDataType(rocsparse_data_type);
        }
        catch(...)
        {
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        }
    }

    if(m != nullptr)
    {
        *m = rocsparse_m;
    }
    if(n != nullptr)
    {
        *n = rocsparse_n;
    }
    if(partition != nullptr)
    {
        *partition = hip_partition;
    }
    if(ell_nnz != nullptr)
    {
        *ell_nnz = rocsparse_ell_nnz;
    }
    if(ell_width != nullptr)
    {
        *ell_width = rocsparse_ell_width;
    }
    if(ell_col_ind != nullptr)
    {
        *ell_col_ind = (const int*)rocsparse_ell_col_ind;
    }
    if(ell_val != nullptr)
    {
        *ell_val = rocsparse_ell_val;
    }
    if(coo_nnz != nullptr)
    {
        *coo_nnz = rocsparse_coo_nnz;
    }
    if(coo_row_ind != nullptr)
    {
        *coo_row_ind = (const int*)rocsparse_coo_row_ind;
    }
    if(coo_col_ind != nullptr)
    {
        *coo_col_ind = (const int*)rocsparse_coo_col_ind;
    }
    if(coo_val != nullptr)
    {
        *coo_val = rocsparse_coo_val;
    }
    if(data_type != nullptr)
    {
        *data_type = hip_data_type;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseHybMatSetInfo(hipsparseHybMat_t       hyb,
                                         int                     m,
                                         int                     n,
                                         hipsparseHybPartition_t partition,
                                         int64_t                 ell_nnz,
                                         int                     ell_width,
                                         int*                    ell_col_ind,
                                         void*                   ell_val,
                                         int                     coo_nnz,
                                         int*                    coo_row_ind,
                                         int*                    coo_col_ind,
                                         void*                   coo_val,
                                         hipDataType             data_type)
{
    // rocSPARSE set_info checks negative sizes, that ell_nnz fits the stored
    // ELL non-zero count, and that the enums are valid. It does not check that
    // the arrays are long enough for those sizes; do not add a second policy.
    // The enum conversions throw on values they do not know.
    rocsparse_hyb_partition rocsparse_partition;
    rocsparse_datatype      rocsparse_data_type;
    try
    {
        rocsparse_partition = hipsparse::hipHybPartToHCCHybPart(partition);
        rocsparse_data_type = hipsparse::hipDataTypeToHCCDataType(data_type);
    }
    catch(...)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_hyb_mat_set_info((rocsparse_hyb_mat)hyb,
                                   m,
                                   n,
                                   rocsparse_partition,
                                   ell_nnz,
                                   ell_width,
                                   (rocsparse_int*)ell_col_ind,
                                   ell_val,
                                   coo_nnz,
                                   (rocsparse_int*)coo_row_ind,
                                   (rocsparse_int*)coo_col_ind,
                                   coo_val,
                                   rocsparse_data_type));
}

hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateColorInfo(hipsparseColorInfo_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyColorInfo(hipsparseColorInfo_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}
