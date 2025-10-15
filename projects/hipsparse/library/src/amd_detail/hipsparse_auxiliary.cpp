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
