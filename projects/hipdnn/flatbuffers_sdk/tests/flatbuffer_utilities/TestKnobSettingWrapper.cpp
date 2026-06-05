// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <gtest/gtest.h>
#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/KnobSettingWrapper.hpp>

using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

class TestKnobSettingWrapper : public ::testing::Test
{
protected:
    static flatbuffers::DetachedBuffer createKnobSetting(const std::string& knobId, int64_t value)
    {
        flatbuffers::FlatBufferBuilder builder;

        auto knobIdOffset = builder.CreateString(knobId);
        // Create the value
        auto intValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(builder, value);

        // Create the knob setting
        hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder settingBuilder(builder);
        settingBuilder.add_knob_id(knobIdOffset);
        settingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
        settingBuilder.add_value(intValue.Union());

        auto setting = settingBuilder.Finish();
        builder.Finish(setting);

        return builder.Release();
    }

    static flatbuffers::DetachedBuffer createKnobSettingWithString(const std::string& knobId,
                                                                   const std::string& value)
    {
        flatbuffers::FlatBufferBuilder builder;

        auto knobIdOffset = builder.CreateString(knobId);
        auto strOffset = builder.CreateString(value);
        auto stringValue
            = hipdnn_flatbuffers_sdk::data_objects::CreateStringValue(builder, strOffset);

        hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder settingBuilder(builder);
        settingBuilder.add_knob_id(knobIdOffset);
        settingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
        settingBuilder.add_value(stringValue.Union());

        auto setting = settingBuilder.Finish();
        builder.Finish(setting);

        return builder.Release();
    }

    static flatbuffers::DetachedBuffer createKnobSettingWithFloat(const std::string& knobId,
                                                                  double value)
    {
        flatbuffers::FlatBufferBuilder builder;

        auto knobIdOffset = builder.CreateString(knobId);
        auto floatValue = hipdnn_flatbuffers_sdk::data_objects::CreateFloatValue(builder, value);

        hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder settingBuilder(builder);
        settingBuilder.add_knob_id(knobIdOffset);
        settingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
        settingBuilder.add_value(floatValue.Union());

        auto setting = settingBuilder.Finish();
        builder.Finish(setting);

        return builder.Release();
    }
};

TEST_F(TestKnobSettingWrapper, ConstructFromFlatbufferPointer)
{
    auto buffer = createKnobSetting("test_knob_42", 100);
    auto setting
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(buffer.data());

    const KnobSettingWrapper wrapper(setting);
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobId(), "test_knob_42");
    EXPECT_EQ(wrapper.valueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
}

TEST_F(TestKnobSettingWrapper, ConstructFromBuffer)
{
    auto buffer = createKnobSetting("test_knob_42", 100);

    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobId(), "test_knob_42");
    EXPECT_EQ(wrapper.valueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
}

TEST_F(TestKnobSettingWrapper, ConstructFromNullPointer)
{
    const KnobSettingWrapper wrapper(
        static_cast<hipdnn_flatbuffers_sdk::data_objects::KnobSetting*>(nullptr));
    EXPECT_FALSE(wrapper.isValid());
}

TEST_F(TestKnobSettingWrapper, ConstructFromNullBuffer)
{
    const KnobSettingWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
}

TEST_F(TestKnobSettingWrapper, ConstructFromInvalidBuffer)
{
    std::array<uint8_t, 10> invalidData
        = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    const KnobSettingWrapper wrapper(invalidData.data(), invalidData.size());
    EXPECT_FALSE(wrapper.isValid());
}

TEST_F(TestKnobSettingWrapper, GetKnobIdFromValidWrapper)
{
    auto buffer = createKnobSetting("test_knob_999", 123);
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_EQ(wrapper.knobId(), "test_knob_999");
}

TEST_F(TestKnobSettingWrapper, ValueAsIntValue)
{
    auto buffer = createKnobSetting("test_knob_42", 100);
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_EQ(wrapper.valueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);

    const auto& intValue = wrapper.valueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>();
    EXPECT_EQ(intValue.value(), 100);
}

TEST_F(TestKnobSettingWrapper, ValueAsStringValue)
{
    auto buffer = createKnobSettingWithString("test_knob_42", "test_value");
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_EQ(wrapper.valueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);

    const auto& stringValue = wrapper.valueAs<hipdnn_flatbuffers_sdk::data_objects::StringValue>();
    EXPECT_EQ(std::string(stringValue.value()->c_str()), "test_value");
}

TEST_F(TestKnobSettingWrapper, ValueAsFloatValue)
{
    auto buffer = createKnobSettingWithFloat("test_knob_42", 3.14159);
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_EQ(wrapper.valueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);

    const auto& floatValue = wrapper.valueAs<hipdnn_flatbuffers_sdk::data_objects::FloatValue>();
    EXPECT_DOUBLE_EQ(floatValue.value(), 3.14159);
}

TEST_F(TestKnobSettingWrapper, ValueAsTypeMismatchThrows)
{
    auto buffer = createKnobSetting("test_knob_42", 100);
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());

    // Value is IntValue, trying to get as FloatValue should throw
    EXPECT_THROW(wrapper.valueAs<hipdnn_flatbuffers_sdk::data_objects::FloatValue>(),
                 std::invalid_argument);
    EXPECT_THROW(wrapper.valueAs<hipdnn_flatbuffers_sdk::data_objects::StringValue>(),
                 std::invalid_argument);
}

TEST_F(TestKnobSettingWrapper, GetKnobSettingFromValidWrapper)
{
    auto buffer = createKnobSetting("test_knob_42", 100);
    auto setting
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(buffer.data());

    const KnobSettingWrapper wrapper(setting);
    const auto& retrievedSetting = wrapper.getKnobSetting();
    EXPECT_EQ(&retrievedSetting, setting);
}

TEST_F(TestKnobSettingWrapper, AccessMethodsOnInvalidWrapperThrow)
{
    const KnobSettingWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());

    EXPECT_THROW(wrapper.knobId(), std::invalid_argument);
    EXPECT_THROW(wrapper.valueType(), std::invalid_argument);
    EXPECT_THROW(wrapper.getKnobSetting(), std::invalid_argument);
}

TEST_F(TestKnobSettingWrapper, MultipleDifferentKnobSettings)
{
    // Test that wrapper correctly handles different knob settings
    auto buffer1 = createKnobSetting("knob_1", 100);
    auto buffer2 = createKnobSettingWithString("knob_2", "config_value");
    auto buffer3 = createKnobSettingWithFloat("knob_3", 2.718);

    const KnobSettingWrapper wrapper1(buffer1.data(), buffer1.size());
    const KnobSettingWrapper wrapper2(buffer2.data(), buffer2.size());
    const KnobSettingWrapper wrapper3(buffer3.data(), buffer3.size());

    EXPECT_EQ(wrapper1.knobId(), "knob_1");
    EXPECT_EQ(wrapper1.valueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);

    EXPECT_EQ(wrapper2.knobId(), "knob_2");
    EXPECT_EQ(wrapper2.valueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);

    EXPECT_EQ(wrapper3.knobId(), "knob_3");
    EXPECT_EQ(wrapper3.valueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
}

TEST_F(TestKnobSettingWrapper, EmptyKnobId)
{
    auto buffer = createKnobSetting("", 42);
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobId(), "");
}

TEST_F(TestKnobSettingWrapper, LongKnobId)
{
    const std::string longKnobId = "this.is.a.very.long.knob.id.with.many.parts.for.testing";
    auto buffer = createKnobSetting(longKnobId, 42);
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobId(), longKnobId);
}

TEST_F(TestKnobSettingWrapper, ToKnobSettingTWithIntValue)
{
    auto buffer = createKnobSetting("test_knob_42", 100);
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());

    auto knobSettingT = wrapper.toKnobSettingT();

    ASSERT_NE(knobSettingT, nullptr);
    EXPECT_EQ(knobSettingT->knob_id, "test_knob_42");
    EXPECT_EQ(knobSettingT->value.type, hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    ASSERT_NE(knobSettingT->value.AsIntValue(), nullptr);
    EXPECT_EQ(knobSettingT->value.AsIntValue()->value, 100);
}

TEST_F(TestKnobSettingWrapper, ToKnobSettingTWithStringValue)
{
    auto buffer = createKnobSettingWithString("test_knob_99", "test_string_value");
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());

    auto knobSettingT = wrapper.toKnobSettingT();

    ASSERT_NE(knobSettingT, nullptr);
    EXPECT_EQ(knobSettingT->knob_id, "test_knob_99");
    EXPECT_EQ(knobSettingT->value.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
    ASSERT_NE(knobSettingT->value.AsStringValue(), nullptr);
    EXPECT_EQ(knobSettingT->value.AsStringValue()->value, "test_string_value");
}

TEST_F(TestKnobSettingWrapper, ToKnobSettingTWithFloatValue)
{
    auto buffer = createKnobSettingWithFloat("test_knob_123", 3.14159);
    const KnobSettingWrapper wrapper(buffer.data(), buffer.size());

    auto knobSettingT = wrapper.toKnobSettingT();

    ASSERT_NE(knobSettingT, nullptr);
    EXPECT_EQ(knobSettingT->knob_id, "test_knob_123");
    EXPECT_EQ(knobSettingT->value.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
    ASSERT_NE(knobSettingT->value.AsFloatValue(), nullptr);
    EXPECT_DOUBLE_EQ(knobSettingT->value.AsFloatValue()->value, 3.14159);
}

TEST_F(TestKnobSettingWrapper, ToKnobSettingTOnInvalidWrapperThrows)
{
    const KnobSettingWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());

    EXPECT_THROW(wrapper.toKnobSettingT(), std::invalid_argument);
}
