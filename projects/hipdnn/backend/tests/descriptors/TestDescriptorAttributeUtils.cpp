// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestMacros.hpp"
#include "descriptors/DescriptorAttributeUtils.hpp"
#include <array>
#include <gtest/gtest.h>
#include <vector>

namespace hipdnn_backend
{
namespace testing
{

TEST(TestDescriptorAttributeUtils, SetInt64VectorThrowsOnNegativeElementCount)
{
    std::vector<int64_t> target;
    std::array<int64_t, 3> data = {1, 2, 3};

    ASSERT_THROW_HIPDNN_STATUS(setInt64Vector(target, HIPDNN_TYPE_INT64, -1, data.data(), "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorThrowsOnNegativeRequestedElementCount)
{
    std::vector<int64_t> source = {1, 2, 3};
    std::array<int64_t, 3> output = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_INT64, -1, &count, output.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

} // namespace testing
} // namespace hipdnn_backend
