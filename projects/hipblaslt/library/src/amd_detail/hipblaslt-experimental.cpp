// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The library always implements the current experimental API version, so it
// opts in to whatever version the header declares.
#define HIPBLASLT_ENABLE_EXPERIMENTAL_API HIPBLASLT_EXPERIMENTAL_API_VERSION
#include "hipblaslt/hipblaslt-experimental.h"

int hipblasLtExperimentalGetApiVersion(void)
{
    return HIPBLASLT_EXPERIMENTAL_API_VERSION;
}
