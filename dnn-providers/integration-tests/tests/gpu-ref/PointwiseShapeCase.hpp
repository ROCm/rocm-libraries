// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/ReferencePointwiseBase.hpp>

#include <cstdint>
#include <ostream>
#include <vector>

namespace gpu_pointwise_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

struct PointwiseTestCase
{
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation;
    std::vector<int64_t> ioDims;

    friend std::ostream& operator<<(std::ostream& os, const PointwiseTestCase& tc)
    {
        os << "(operation "
           << hipdnn_flatbuffers_sdk::data_objects::EnumNamePointwiseMode(tc.operation);
        os << ", io dims: ";
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
    auto validator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<T>(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

} // namespace gpu_pointwise_ref_test
