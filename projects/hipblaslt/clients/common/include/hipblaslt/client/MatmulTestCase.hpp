// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipblaslt_arguments.hpp"
#include <hipblaslt/host_validation/Types.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace hipblaslt::client
{
    struct MatmulMatrix
    {
        hipDataType                      apiType;
        roc::host_validation::ScalarType hostType;
        roc::host_validation::Layout     layout;
    };

    struct MatmulTestCase
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
    };

    std::vector<MatmulTestCase> normalizeMatmulCases(const Arguments& arguments);
} // namespace hipblaslt::client
