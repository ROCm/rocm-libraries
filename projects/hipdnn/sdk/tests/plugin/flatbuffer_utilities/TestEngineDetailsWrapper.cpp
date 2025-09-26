// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/EngineDetailsWrapper.hpp>

using namespace hipdnn_plugin;

flatbuffers::FlatBufferBuilder buildValidEngineDetailsBuffer(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto config = hipdnn_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(config);
    return builder;
}

TEST(TestEngineDetailsWrapper, InvalidBufferIsNotValid)
{
    EngineDetailsWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), HipdnnPluginException);
    EXPECT_THROW(wrapper.getEngineDetails(), HipdnnPluginException);
}

TEST(TestEngineDetailsWrapper, ValidBufferIsValid)
{
    int64_t testEngineId = 42;
    auto builder = buildValidEngineDetailsBuffer(testEngineId);
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.engineId(), testEngineId);
    EXPECT_NO_THROW(wrapper.getEngineDetails());
}

TEST(TestEngineDetailsWrapper, CorruptedBufferIsNotValid)
{
    std::vector<uint8_t> buffer(16, 0xFF); // Not a valid flatbuffer
    EngineDetailsWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), HipdnnPluginException);
}
