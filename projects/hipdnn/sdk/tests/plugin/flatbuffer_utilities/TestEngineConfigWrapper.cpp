// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/EngineConfigWrapper.hpp>

using namespace hipdnn_plugin;

flatbuffers::FlatBufferBuilder buildValidEngineConfigBuffer(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto config = hipdnn_sdk::data_objects::CreateEngineConfig(builder, engineId);
    builder.Finish(config);
    return builder;
}

TEST(TestEngineConfigWrapper, InvalidBufferIsNotValid)
{
    EngineConfigWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), HipdnnPluginException);
    EXPECT_THROW(wrapper.getEngineConfig(), HipdnnPluginException);
}

TEST(TestEngineConfigWrapper, ValidBufferIsValid)
{
    int64_t testEngineId = 42;
    auto builder = buildValidEngineConfigBuffer(testEngineId);
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.engineId(), testEngineId);
    EXPECT_NO_THROW(wrapper.getEngineConfig());
}

TEST(TestEngineConfigWrapper, CorruptedBufferIsNotValid)
{
    std::vector<uint8_t> buffer(16, 0xFF); // Not a valid flatbuffer
    EngineConfigWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), HipdnnPluginException);
}
