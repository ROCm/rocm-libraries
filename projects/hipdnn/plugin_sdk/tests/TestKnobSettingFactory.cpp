// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/KnobSettingFactory.hpp>

using namespace hipdnn_plugin_sdk;

TEST(TestKnobSettingFactory, CreateIntKnobSetting)
{
    auto buffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 42);

    auto root
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(buffer.data());

    EXPECT_STREQ(root->knob_id()->c_str(), "test.int_knob");
    EXPECT_EQ(root->value_type(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    EXPECT_EQ(root->value_as_IntValue()->value(), 42);
}

TEST(TestKnobSettingFactory, CreateIntKnobSettingNegativeValue)
{
    auto buffer = KnobSettingFactory::createIntKnobSetting("negative_knob", -100);

    auto root
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(buffer.data());

    EXPECT_STREQ(root->knob_id()->c_str(), "negative_knob");
    EXPECT_EQ(root->value_as_IntValue()->value(), -100);
}

TEST(TestKnobSettingFactory, CreateFloatKnobSetting)
{
    auto buffer = KnobSettingFactory::createFloatKnobSetting("test.float_knob", 3.14159);

    auto root
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(buffer.data());

    EXPECT_STREQ(root->knob_id()->c_str(), "test.float_knob");
    EXPECT_EQ(root->value_type(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
    EXPECT_DOUBLE_EQ(root->value_as_FloatValue()->value(), 3.14159);
}

TEST(TestKnobSettingFactory, CreateFloatKnobSettingZeroValue)
{
    auto buffer = KnobSettingFactory::createFloatKnobSetting("zero_knob", 0.0);

    auto root
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(buffer.data());

    EXPECT_STREQ(root->knob_id()->c_str(), "zero_knob");
    EXPECT_DOUBLE_EQ(root->value_as_FloatValue()->value(), 0.0);
}

TEST(TestKnobSettingFactory, CreateStringKnobSetting)
{
    auto buffer = KnobSettingFactory::createStringKnobSetting("test.string_knob", "fast");

    auto root
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(buffer.data());

    EXPECT_STREQ(root->knob_id()->c_str(), "test.string_knob");
    EXPECT_EQ(root->value_type(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
    EXPECT_STREQ(root->value_as_StringValue()->value()->c_str(), "fast");
}

TEST(TestKnobSettingFactory, CreateStringKnobSettingEmptyValue)
{
    auto buffer = KnobSettingFactory::createStringKnobSetting("empty_knob", "");

    auto root
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(buffer.data());

    EXPECT_STREQ(root->knob_id()->c_str(), "empty_knob");
    EXPECT_STREQ(root->value_as_StringValue()->value()->c_str(), "");
}
