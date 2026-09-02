// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private adapter from hipBLASLt storage descriptors to the
// product-independent host-numerics reduction API.

#include <cstdint>
#include <hipblaslt/hipblaslt.h>
#include <roc/host_numerics/reduction.hpp>

namespace hipblaslt::host_numerics
{
    struct ReductionArguments
    {
        int64_t     rows            = 0;
        int64_t     columns         = 0;
        int64_t     rowStride       = 0;
        int64_t     columnStride    = 0;
        const void* input           = nullptr;
        hipDataType inputType       = HIP_R_32F;
        void*       output          = nullptr;
        hipDataType outputType      = HIP_R_32F;
        int64_t     outputStride    = 1;
        hipDataType accumulatorType = HIP_R_32F;
    };

    void referenceSum(const ReductionArguments& arguments);
} // namespace hipblaslt::host_numerics
