// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt/hipblaslt-user-kernel.h"
#include "exceptions.hpp"
#include "hipblaslt_internal.hpp"
#include "tensile_host.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

hipblasStatus_t hipblasLtUserKernelRegister(hipblasLtHandle_t handle,
                                            const char*       libraryPath,
                                            const char*       codeObjectPath,
                                            int*              kernelIndices,
                                            int               maxIndices,
                                            int*              numIndices)
try
{
    if(numIndices)
        *numIndices = 0;

    std::vector<int> assigned;
    auto             status = rocblaslt_user_kernel_register(
        (rocblaslt_handle)handle, libraryPath, codeObjectPath, assigned);

    if(status != rocblaslt_status_success)
        return RocBlasLtStatusToHIPStatus(status);

    if(numIndices)
        *numIndices = static_cast<int>(assigned.size());

    if(kernelIndices && maxIndices > 0)
    {
        const size_t n = std::min<size_t>(assigned.size(), static_cast<size_t>(maxIndices));
        std::copy_n(assigned.begin(), n, kernelIndices);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtUserKernelSetExactMatch(
    hipblasLtHandle_t handle, int kernelIndex, size_t m, size_t n, size_t batch, size_t k)
try
{
    return RocBlasLtStatusToHIPStatus(rocblaslt_user_kernel_set_exact_match(
        (rocblaslt_handle)handle, kernelIndex, m, n, batch, k));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtUserKernelRefresh(hipblasLtHandle_t handle, size_t* registered)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_user_kernel_refresh((rocblaslt_handle)handle, registered));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtUserKernelGetCounts(hipblasLtHandle_t handle, size_t* registered, size_t* selectable)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_user_kernel_counts((rocblaslt_handle)handle, registered, selectable));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulAlgoIsUserKernel(const hipblasLtMatmulAlgo_t* algo,
                                                int*                         isUserKernel)
try
{
    if(!algo || !isUserKernel)
        return HIPBLAS_STATUS_INVALID_VALUE;

    // Same layout getIndexFromAlgo reads.
    int index = 0;
    std::memcpy(&index, algo->data, sizeof(index));
    *isUserKernel = rocblaslt_is_user_kernel_index(index) ? 1 : 0;

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
