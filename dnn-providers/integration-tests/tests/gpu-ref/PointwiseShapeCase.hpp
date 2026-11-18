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

// assertAllClose treats any NaN or infinity as a failure. This accepts a NaN where both sides have
// one and an infinity where both have the same one, and compares finite values with the tolerance
// scaled by the size of the expected value. Both tensors must have the same layout.
template <typename T>
void assertAllCloseWithSpecialValues(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                                     hipdnn_data_sdk::utilities::TensorBase<T>& actual,
                                     float tolerance)
{
    ASSERT_EQ(expected.elementSpace(), actual.elementSpace());
    const auto* expectedData = expected.memory().hostData();
    const auto* actualData = actual.memory().hostData();
    for(size_t i = 0; i < expected.elementSpace(); ++i)
    {
        const auto expectedValue = static_cast<float>(expectedData[i]);
        const auto actualValue = static_cast<float>(actualData[i]);
        if(std::isnan(expectedValue) || std::isinf(expectedValue))
        {
            ASSERT_TRUE(std::isnan(expectedValue) ? std::isnan(actualValue)
                                                  : actualValue == expectedValue)
                << "Mismatch at index " << i << ": expected " << expectedValue << ", actual "
                << actualValue;
        }
        else
        {
            ASSERT_LE(std::fabs(actualValue - expectedValue),
                      tolerance * std::max(1.0f, std::fabs(expectedValue)))
                << "Mismatch at index " << i << ": expected " << expectedValue << ", actual "
                << actualValue;
        }
    }
}

// Exact comparison for results that are not subject to rounding (bool, selection, MAX/MIN).
// A NaN matches a NaN. Both tensors must have the same layout.
template <typename T>
void assertAllEqual(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                    hipdnn_data_sdk::utilities::TensorBase<T>& actual)
{
    ASSERT_EQ(expected.elementSpace(), actual.elementSpace());
    const auto* expectedData = expected.memory().hostData();
    const auto* actualData = actual.memory().hostData();
    for(size_t i = 0; i < expected.elementSpace(); ++i)
    {
        const auto expectedValue = static_cast<float>(expectedData[i]);
        const auto actualValue = static_cast<float>(actualData[i]);
        ASSERT_TRUE((std::isnan(expectedValue) && std::isnan(actualValue))
                    || expectedValue == actualValue)
            << "Mismatch at index " << i << ": expected " << expectedValue << ", actual "
            << actualValue;
    }
}

} // namespace gpu_pointwise_ref_test
