// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// scanNonFinite() is what turns a reference smoke run red. A reference never writes
// the padding of a strided output, which still holds the NaN sentinel, so a flat walk
// over the allocation would fail every non-packed case; these pin that only logical
// elements count, and that a real NaN or Inf is still found.

#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include "harness/reference-validation/BundleReferenceSmokeHarness.hpp"

using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_integration_tests::bundle::scanNonFinite;

namespace
{

void setHost(Tensor<float>& tensor, int64_t offset, float value)
{
    *static_cast<float*>(tensor.hostDataOffsetFromIndex(offset)) = value;
    tensor.markHostModified();
}

} // namespace

TEST(TestReferenceSmokeScan, PaddingBetweenStridedElementsIsNotScanned)
{
    // [2, 3] with a row stride of 4: offsets 3 and 7 are padding.
    Tensor<float> tensor({2, 3}, {4, 1});
    tensor.fillWithSentinelValue();
    for(const int64_t offset : {0, 1, 2, 4, 5, 6})
    {
        setHost(tensor, offset, 1.0f);
    }

    const auto scan = scanNonFinite(tensor);

    EXPECT_TRUE(scan.scanned);
    EXPECT_EQ(scan.elements, 6u);
    EXPECT_EQ(scan.nans, 0u);
    EXPECT_EQ(scan.infs, 0u);
}

TEST(TestReferenceSmokeScan, NanAndInfInLogicalElementsAreCounted)
{
    Tensor<float> tensor({2, 3}, {4, 1});
    tensor.fillTensorWithValue(1.0f);
    setHost(tensor, 5, std::numeric_limits<float>::quiet_NaN()); // index [1,1]
    setHost(tensor, 6, std::numeric_limits<float>::infinity()); // index [1,2]

    const auto scan = scanNonFinite(tensor);

    EXPECT_EQ(scan.nans, 1u);
    EXPECT_EQ(scan.infs, 1u);
    ASSERT_TRUE(scan.firstBadIndex.has_value());
    EXPECT_EQ(*scan.firstBadIndex, (std::vector<int64_t>{1, 1}));
}

TEST(TestReferenceSmokeScan, UnwrittenOutputIsReportedThroughTheSentinel)
{
    Tensor<float> tensor({4});
    tensor.fillWithSentinelValue();

    EXPECT_EQ(scanNonFinite(tensor).nans, 4u);
}

TEST(TestReferenceSmokeScan, IntegerOutputIsNotScanned)
{
    Tensor<int32_t> tensor({4});
    tensor.fillTensorWithValue(1.0f);

    EXPECT_FALSE(scanNonFinite(tensor).scanned);
}
