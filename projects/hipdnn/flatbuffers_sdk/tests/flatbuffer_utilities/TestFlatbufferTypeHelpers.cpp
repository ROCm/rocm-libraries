// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>

using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;

namespace
{

std::string streamed(AttentionImplementation implementation)
{
    std::ostringstream oss;
    oss << implementation;
    return oss.str();
}

} // namespace

TEST(TestFlatbufferTypeHelpers, OstreamOperatorAttentionImplementation)
{
    EXPECT_EQ(streamed(AttentionImplementation::AUTO), "AUTO");
    EXPECT_EQ(streamed(AttentionImplementation::COMPOSITE), "COMPOSITE");
    EXPECT_EQ(streamed(AttentionImplementation::UNIFIED), "UNIFIED");
}

TEST(TestFlatbufferTypeHelpers, OstreamOperatorAttentionImplementationOutOfRangeIsEmpty)
{
    // A value past the schema's last enumerator, as a newer serializer could write, prints
    // nothing instead of reading past the generated name table.
    EXPECT_TRUE(streamed(static_cast<AttentionImplementation>(42)).empty());
}
