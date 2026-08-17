// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

#include <cassert>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

// ============================================================================
// ConvShapeCase — unified shape parameters for parameterized convolution tests.
// Shared across forward, dgrad, and wgrad directions.
// ============================================================================

namespace gpu_conv_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

struct ConvShapeCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> wDims;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> padding;
    int64_t groups = 1;
    std::string tag;

    // When non-null, input/output tensors use this channel-last layout (NHWC/NDHWC/NLC).
    // Weights always use default packed (KCRS) strides regardless.
    // When null, all tensors use default packed strides (NCHW/NCDHW/NCL).
    const TensorLayout* layout = nullptr;

    // Computes the forward output dimensions (= dy dims for dgrad/wgrad).
    // Requires rank >= 3 (N, C, and at least one spatial dimension).
    std::vector<int64_t> computeOutputDims() const
    {
        assert(xDims.size() >= 3 && "ConvShapeCase requires at least 3 dims (N, C, spatial)");
        auto numSpatialDims = xDims.size() - 2;
        std::vector<int64_t> yDims = {xDims[0], wDims[0]};
        for(size_t i = 0; i < numSpatialDims; ++i)
        {
            auto outputSize
                = (xDims[2 + i] + 2 * padding[i] - dilations[i] * (wDims[2 + i] - 1) - 1)
                      / strides[i]
                  + 1;
            yDims.push_back(outputSize);
        }
        return yDims;
    }

    friend std::ostream& operator<<(std::ostream& os, const ConvShapeCase& tc)
    {
        return os << tc.tag;
    }
};

// Name generator for parameterized tests — extracts the tag from ConvShapeCase.
// Usage: INSTANTIATE_TEST_SUITE_P(Smoke, Suite, ::testing::ValuesIn(cases), byTag());
inline auto byTag()
{
    return [](const auto& info) { return info.param.tag; };
}

// Validates that two tensors are element-wise close using the standard allClose validator.
// Handles NaN/Inf detection, stride-aware indexing, and parallel comparison.
// Shared across fwd/dgrad/wgrad fixture headers.
template <typename T>
void assertAllClose(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                    hipdnn_data_sdk::utilities::TensorBase<T>& actual,
                    float tolerance)
{
    auto validator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<T>(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

} // namespace gpu_conv_ref_test
