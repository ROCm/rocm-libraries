// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

#include <cassert>
#include <cstdint>
#include <ostream>
#include <vector>

namespace gpu_rmsnorm_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

struct RMSNormTestCase
{
    std::vector<int64_t> ioDims;
    std::vector<int64_t> scaleDims;
    TensorLayout layout;

    friend std::ostream& operator<<(std::ostream& os, const RMSNormTestCase& tc)
    {
        os << "(io dims: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.ioDims);
        os << " scale dims: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.scaleDims);
        os << " layout:" << tc.layout.name;
        os << ")";

        return os;
    }
};

template <typename T>
void assertAllClose(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                    hipdnn_data_sdk::utilities::TensorBase<T>& actual,
                    float tolerance)
{
    auto validator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<T>(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

inline std::vector<RMSNormTestCase> getRMSnormTestCases()
{
    return {
        {{2, 3, 4, 4}, {1, 3, 4, 4}, TensorLayout::NCHW},
        {{2, 3, 4, 4}, {1, 3, 4, 4}, TensorLayout::NHWC},
        {{5, 256, 14, 14}, {1, 256, 14, 14}, TensorLayout::NCHW},
        {{5, 256, 14, 14}, {1, 256, 14, 14}, TensorLayout::NHWC},
        {{2, 3, 4, 4}, {1, 3, 4, 4}, TensorLayout::NCHW},
        {{2, 3, 4, 4}, {1, 1, 4, 4}, TensorLayout::NCHW},
    };
}

inline std::vector<RMSNormTestCase> getRMSnormFullTestCases()
{
    return {
        {{1, 3, 14, 14}, {1, 3, 14, 14}, TensorLayout::NCHW},
        {{1, 3, 14, 14}, {1, 1, 14, 14}, TensorLayout::NCHW},
        {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NCHW},
        {{1, 256, 1, 1}, {1, 256, 1, 1}, TensorLayout::NCHW},
        {{2, 3, 1, 1}, {1, 3, 1, 1}, TensorLayout::NCHW},
        {{32, 1, 14, 14}, {1, 1, 14, 14}, TensorLayout::NCHW},
        {{32, 3, 1, 14}, {1, 3, 1, 14}, TensorLayout::NCHW},
        {{32, 3, 14, 1}, {1, 3, 14, 1}, TensorLayout::NCHW},
        {{32, 3, 14, 1}, {1, 1, 14, 1}, TensorLayout::NCHW},

        {{1, 3, 14, 14}, {1, 3, 14, 14}, TensorLayout::NHWC},
        {{1, 3, 14, 14}, {1, 1, 14, 14}, TensorLayout::NHWC},
        {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NHWC},
        {{1, 256, 1, 1}, {1, 256, 1, 1}, TensorLayout::NHWC},
        {{2, 3, 1, 1}, {1, 3, 1, 1}, TensorLayout::NHWC},
        {{32, 1, 14, 14}, {1, 1, 14, 14}, TensorLayout::NHWC},
        {{32, 3, 1, 14}, {1, 3, 1, 14}, TensorLayout::NHWC},
        {{32, 3, 14, 1}, {1, 3, 14, 1}, TensorLayout::NHWC},
        {{32, 3, 14, 1}, {1, 1, 14, 1}, TensorLayout::NHWC},
    };
}

inline std::vector<RMSNormTestCase> getRMSnorm3dTestCases()
{
    return {
        {{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}, TensorLayout::NCDHW},
        {{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}, TensorLayout::NCDHW},
        {{2, 3, 4, 2, 2}, {1, 1, 4, 2, 2}, TensorLayout::NCDHW},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}, TensorLayout::NCDHW},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}, TensorLayout::NCDHW},

        {{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}, TensorLayout::NDHWC},
        {{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}, TensorLayout::NDHWC},
        {{2, 3, 4, 2, 2}, {1, 1, 4, 2, 2}, TensorLayout::NDHWC},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}, TensorLayout::NDHWC},
        {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}, TensorLayout::NDHWC},
    };
}

} // namespace gpu_rmsnorm_ref_test
