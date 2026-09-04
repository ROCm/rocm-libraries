// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipblaslt_arguments.hpp"
#include <hipblaslt/host_numerics/Types.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace hipblaslt::client
{
    struct MatmulDataTypes
    {
        hipDataType computeScalar;
        hipDataType computeInputA;
        hipDataType computeInputB;
        hipDataType coefficient;
        hipDataType bias;
        hipDataType biasStorage;
        hipDataType auxiliary;
    };

    struct MatmulMatrix
    {
        hipDataType                      apiType;
        roc::host_numerics::ScalarType hostType;
        roc::host_numerics::Layout     layout;
        size_t                           allocationElements;

        int64_t rows() const
        {
            return static_cast<int64_t>(layout.shape().extent(0));
        }

        int64_t columns() const
        {
            return static_cast<int64_t>(layout.shape().extent(1));
        }

        int64_t leadingDimension() const
        {
            return static_cast<int64_t>(layout.stride(1));
        }

        int64_t batchStride() const
        {
            return static_cast<int64_t>(layout.stride(2));
        }
    };

    struct MatmulProblem
    {
        int64_t m;
        int64_t n;
        int64_t k;

        hipblasOperation_t   operationA;
        hipblasOperation_t   operationB;
        hipblasLtBatchMode_t batchMode;
        int32_t              batchCount;

        MatmulMatrix                a;
        MatmulMatrix                b;
        MatmulMatrix                c;
        MatmulMatrix                d;
        std::optional<MatmulMatrix> auxiliary;

        bool cEqualsD;

        size_t auxiliaryAllocationElements() const
        {
            return auxiliary ? auxiliary->allocationElements : 0;
        }
    };

    MatmulDataTypes resolveMatmulDataTypes(const Arguments& arguments);

    std::vector<MatmulProblem> normalizeMatmulProblems(const Arguments& arguments);
} // namespace hipblaslt::client
