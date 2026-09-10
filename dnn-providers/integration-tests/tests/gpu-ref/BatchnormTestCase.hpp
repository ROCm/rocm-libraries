// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceValidation.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <ostream>
#include <vector>

namespace gpu_batchnorm_ref_test
{

struct BatchnormTestCase
{
    std::vector<int64_t> ioDims;

    friend std::ostream& operator<<(std::ostream& os, const BatchnormTestCase& tc)
    {
        os << "(io dims: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.ioDims);
        os << ")";

        return os;
    }
};

template <typename T>
void assertAllClose(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                    hipdnn_data_sdk::utilities::TensorBase<T>& actual,
                    float tolerance)
{
    auto validator = hipdnn_gpu_ref::GpuFpReferenceValidation<T>(tolerance, tolerance);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

} // namespace gpu_batchnorm_ref_test
