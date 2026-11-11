// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Second C translation unit for the header check; see c_headers_a.c.

#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt-local-kernel.h>
#include <hipblaslt/hipblaslt.h>

#include <string.h>

int hipblaslt_c_headers_b(int dataType)
{
    hipblasLtMatmulAlgo_t algo;
    memset(&algo, 0, sizeof(algo));

    hipblasStatus_t (*destroy)(const hipblasLtHandle_t) = hipblasLtDestroy;
    if(!destroy || algo.max_workspace_bytes != 0)
        return 1;

    // Case labels need integer constant expressions, which C requires of
    // these extension type ids.
    switch(dataType)
    {
    case HIP_R_6F_E2M3_EXT:
    case HIP_R_6F_E3M2_EXT:
    case HIP_R_4F_E2M1_EXT:
    case HIP_R_8F_E5M3_EXT:
        return 1;
    default:
        return 0;
    }
}
