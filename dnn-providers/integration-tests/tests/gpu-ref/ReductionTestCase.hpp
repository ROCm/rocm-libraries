// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceValidation.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>

#include <cstdint>
#include <ostream>
#include <vector>

namespace gpu_reduction_ref_test
{

struct ReductionTestCase
{
    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;
    hipdnn_data_sdk::utilities::TensorLayout layout;
    hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode;

    friend std::ostream& operator<<(std::ostream& os, const ReductionTestCase& tc)
    {
        os << "(input dims: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.inputDims);
        os << " output dims: ";
        hipdnn_data_sdk::utilities::vecToStream(os, tc.outputDims);
        os << " layout:" << tc.layout.name;
        os << " mode:" << hipdnn_flatbuffers_sdk::data_objects::EnumNameReductionMode(tc.mode);
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

} // namespace gpu_reduction_ref_test
