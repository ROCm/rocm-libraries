// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceValidation.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <ostream>
#include <vector>

namespace gpu_rmsnorm_ref_test
{

struct RMSNormTestCase
{
    std::vector<int64_t> ioDims;
    std::vector<int64_t> scaleDims;
    hipdnn_data_sdk::utilities::TensorLayout layout;

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
    auto validator = hipdnn_gpu_ref::GpuFpReferenceValidation<T>(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

} // namespace gpu_rmsnorm_ref_test
