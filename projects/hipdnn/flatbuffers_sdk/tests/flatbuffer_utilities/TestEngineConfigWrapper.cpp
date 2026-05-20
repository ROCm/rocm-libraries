// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>

using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

flatbuffers::FlatBufferBuilder buildValidEngineConfigBuffer(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto config = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(builder, engineId);
    builder.Finish(config);
    return builder;
}

flatbuffers::FlatBufferBuilder buildEngineConfigWithKnobSettings(
    int64_t engineId, const std::vector<std::pair<std::string, int64_t>>& knobIdValues)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>>
        knobSettings;
    knobSettings.reserve(knobIdValues.size());

    for(const auto& [knobId, value] : knobIdValues)
    {
        auto knobIdOffset = builder.CreateString(knobId);
        auto intValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(builder, value);
        hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder settingBuilder(builder);
        settingBuilder.add_knob_id(knobIdOffset);
        settingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
        settingBuilder.add_value(intValue.Union());
        knobSettings.push_back(settingBuilder.Finish());
    }

    auto knobsVector = builder.CreateVector(knobSettings);
    auto config
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(builder, engineId, knobsVector);
    builder.Finish(config);
    return builder;
}

} // namespace

TEST(TestEngineConfigWrapper, InvalidBufferIsNotValid)
{
    const EngineConfigWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), std::invalid_argument);
    EXPECT_THROW(wrapper.getEngineConfig(), std::invalid_argument);
}

TEST(TestEngineConfigWrapper, ValidBufferIsValid)
{
    const int64_t testEngineId = 42;
    auto builder = buildValidEngineConfigBuffer(testEngineId);
    const EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.engineId(), testEngineId);
    EXPECT_NO_THROW(wrapper.getEngineConfig());
}

TEST(TestEngineConfigWrapper, CorruptedBufferIsNotValid)
{
    std::vector<uint8_t> buffer(16, 0xFF); // Not a valid flatbuffer
    const EngineConfigWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), std::invalid_argument);
}

TEST(TestEngineConfigWrapper, KnobSettingCountEmpty)
{
    auto builder = buildValidEngineConfigBuffer(42);
    const EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobSettingCount(), 0u);
}

TEST(TestEngineConfigWrapper, KnobSettingCountNonEmpty)
{
    auto builder = buildEngineConfigWithKnobSettings(
        42, {{"knob_1", 100}, {"knob_2", 200}, {"knob_3", 300}});
    const EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobSettingCount(), 3u);
}

TEST(TestEngineConfigWrapper, KnobSettingWrappersEmpty)
{
    auto builder = buildValidEngineConfigBuffer(42);
    const EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    const auto& wrappers = wrapper.knobSettingWrappers();
    EXPECT_TRUE(wrappers.empty());
}

TEST(TestEngineConfigWrapper, KnobSettingWrappersPopulated)
{
    auto builder = buildEngineConfigWithKnobSettings(42, {{"knob_1", 100}, {"knob_2", 200}});
    const EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    const auto& wrappers = wrapper.knobSettingWrappers();
    EXPECT_EQ(wrappers.size(), 2u);
    EXPECT_EQ(wrappers[0]->knobId(), "knob_1");
    EXPECT_EQ(wrappers[1]->knobId(), "knob_2");
}

TEST(TestEngineConfigWrapper, GetKnobSettingByNameFound)
{
    auto builder
        = buildEngineConfigWithKnobSettings(42, {{"test_knob_100", 1000}, {"test_knob_200", 2000}});
    const EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& knobSetting = wrapper.getKnobSettingByName("test_knob_100");
    EXPECT_EQ(knobSetting.knobId(), "test_knob_100");

    const auto& knobSetting2 = wrapper.getKnobSettingByName("test_knob_200");
    EXPECT_EQ(knobSetting2.knobId(), "test_knob_200");
}

TEST(TestEngineConfigWrapper, GetKnobSettingByNameNotFound)
{
    auto builder = buildEngineConfigWithKnobSettings(42, {{"test_knob_100", 1000}});
    const EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(wrapper.getKnobSettingByName("NONEXISTENT_KNOB"), std::out_of_range);
}

TEST(TestEngineConfigWrapper, KnobSettingMethodsOnInvalidWrapperThrow)
{
    const EngineConfigWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());

    EXPECT_THROW(wrapper.knobSettingCount(), std::invalid_argument);
    EXPECT_THROW(wrapper.knobSettingWrappers(), std::invalid_argument);
    EXPECT_THROW(wrapper.getKnobSettingByName("test"), std::invalid_argument);
    EXPECT_THROW(wrapper.hasKnobSetting("test"), std::invalid_argument);
}

TEST(TestEngineConfigWrapper, HasKnobSettingByName)
{
    auto builder
        = buildEngineConfigWithKnobSettings(42, {{"test_knob_100", 1000}, {"test_knob_200", 2000}});
    const EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(wrapper.hasKnobSetting("test_knob_100"));
    EXPECT_TRUE(wrapper.hasKnobSetting("test_knob_200"));
    EXPECT_FALSE(wrapper.hasKnobSetting("test_knob_300"));
}
