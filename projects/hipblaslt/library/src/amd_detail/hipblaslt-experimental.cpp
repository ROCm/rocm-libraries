// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The library always implements the current experimental API version, so it
// opts in to whatever version the header declares.
#define HIPBLASLT_ENABLE_EXPERIMENTAL_API HIPBLASLT_EXPERIMENTAL_API_VERSION
#include "hipblaslt/hipblaslt-experimental.h"

#include <limits>

int hipblasLtExperimentalGetApiVersion(void)
{
    return HIPBLASLT_EXPERIMENTAL_API_VERSION;
}

hipblasStatus_t hipblasLtExperimentalDemoAlignSize(size_t                              size,
                                                   size_t                              alignment,
                                                   hipblasLtExperimentalDemoRounding_t rounding,
                                                   size_t*                             result)
{
    if(result == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0)
        return HIPBLAS_STATUS_INVALID_VALUE;

    size_t down = size & ~(alignment - 1);
    switch(rounding)
    {
    case HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_DOWN:
        *result = down;
        return HIPBLAS_STATUS_SUCCESS;
    case HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP:
        if(down == size)
        {
            *result = size;
            return HIPBLAS_STATUS_SUCCESS;
        }
        if(down > std::numeric_limits<size_t>::max() - alignment)
            return HIPBLAS_STATUS_INVALID_VALUE;
        *result = down + alignment;
        return HIPBLAS_STATUS_SUCCESS;
    }
    return HIPBLAS_STATUS_INVALID_VALUE;
}
