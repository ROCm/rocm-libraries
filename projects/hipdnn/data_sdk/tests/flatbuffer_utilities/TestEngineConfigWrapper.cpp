// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

using namespace hipdnn_plugin_sdk;

flatbuffers::FlatBufferBuilder buildValidEngineConfigBuffer(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto config = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, engineId);
    builder.Finish(config);
    return builder;
}

flatbuffers::FlatBufferBuilder
    buildEngineConfigWithKnobSettings(int64_t engineId,
                                      const std::vector<std::pair<int64_t, int64_t>>& knobIdValues)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::KnobSetting>> knobSettings;
    knobSettings.reserve(knobIdValues.size());

    for(const auto& [knobId, value] : knobIdValues)
    {
        auto intValue = hipdnn_data_sdk::data_objects::CreateIntValue(builder, value);
        hipdnn_data_sdk::data_objects::KnobSettingBuilder settingBuilder(builder);
        settingBuilder.add_knob_id(knobId);
        settingBuilder.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
        settingBuilder.add_value(intValue.Union());
        knobSettings.push_back(settingBuilder.Finish());
    }

    auto knobsVector = builder.CreateVector(knobSettings);
    auto config = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, engineId, knobsVector);
    builder.Finish(config);
    return builder;
}

TEST(TestEngineConfigWrapper, InvalidBufferIsNotValid)
{
    EngineConfigWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), std::invalid_argument);
    EXPECT_THROW(wrapper.getEngineConfig(), std::invalid_argument);
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
    EXPECT_THROW(wrapper.engineId(), std::invalid_argument);
}

TEST(TestEngineConfigWrapper, KnobSettingCountEmpty)
{
    auto builder = buildValidEngineConfigBuffer(42);
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobSettingCount(), 0u);
}

TEST(TestEngineConfigWrapper, KnobSettingCountNonEmpty)
{
    auto builder = buildEngineConfigWithKnobSettings(42, {{1, 100}, {2, 200}, {3, 300}});
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobSettingCount(), 3u);
}

TEST(TestEngineConfigWrapper, KnobSettingWrappersEmpty)
{
    auto builder = buildValidEngineConfigBuffer(42);
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    const auto& wrappers = wrapper.knobSettingWrappers();
    EXPECT_TRUE(wrappers.empty());
}

TEST(TestEngineConfigWrapper, KnobSettingWrappersPopulated)
{
    auto builder = buildEngineConfigWithKnobSettings(42, {{1, 100}, {2, 200}});
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    const auto& wrappers = wrapper.knobSettingWrappers();
    EXPECT_EQ(wrappers.size(), 2u);
    EXPECT_EQ(wrappers[0]->knobId(), 1);
    EXPECT_EQ(wrappers[1]->knobId(), 2);
}

TEST(TestEngineConfigWrapper, GetKnobSettingByIdFound)
{
    auto builder = buildEngineConfigWithKnobSettings(42, {{100, 1000}, {200, 2000}});
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& knobSetting = wrapper.getKnobSettingById(100);
    EXPECT_EQ(knobSetting.knobId(), 100);

    const auto& knobSetting2 = wrapper.getKnobSettingById(200);
    EXPECT_EQ(knobSetting2.knobId(), 200);
}

TEST(TestEngineConfigWrapper, GetKnobSettingByIdNotFound)
{
    auto builder = buildEngineConfigWithKnobSettings(42, {{100, 1000}});
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(wrapper.getKnobSettingById(999), std::out_of_range);
}

TEST(TestEngineConfigWrapper, GetKnobSettingByNameFound)
{
    // Use fnv1aHash to generate a known knob ID from a name
    auto knobName = "TEST_KNOB";
    auto knobId = static_cast<int64_t>(hipdnn_data_sdk::utilities::fnv1aHash(knobName));

    auto builder = buildEngineConfigWithKnobSettings(42, {{knobId, 500}});
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& knobSetting = wrapper.getKnobSettingByName(knobName);
    EXPECT_EQ(knobSetting.knobId(), knobId);
}

TEST(TestEngineConfigWrapper, GetKnobSettingByNameNotFound)
{
    auto builder = buildEngineConfigWithKnobSettings(42, {{100, 1000}});
    EngineConfigWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(wrapper.getKnobSettingByName("NONEXISTENT_KNOB"), std::out_of_range);
}

TEST(TestEngineConfigWrapper, KnobSettingMethodsOnInvalidWrapperThrow)
{
    EngineConfigWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());

    EXPECT_THROW(wrapper.knobSettingCount(), std::invalid_argument);
    EXPECT_THROW(wrapper.knobSettingWrappers(), std::invalid_argument);
    EXPECT_THROW(wrapper.getKnobSettingById(1), std::invalid_argument);
    EXPECT_THROW(wrapper.getKnobSettingByName("test"), std::invalid_argument);
}
