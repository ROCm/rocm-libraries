// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

using namespace hipdnn_plugin_sdk;

TEST(TestKnobFactory, CreateIntKnob)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> options = {10, 20, 30};
    auto knob
        = KnobFactory::createIntKnob(builder, "int_knob", "description", 10, 0, 100, 1, options);
    builder.Finish(knob);

    auto root = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(
        builder.GetBufferPointer());

    EXPECT_STREQ(root->knob_id()->c_str(), "int_knob");
    EXPECT_STREQ(root->description()->c_str(), "description");
    EXPECT_EQ(root->default_value_type(),
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    EXPECT_EQ(root->default_value_as_IntValue()->value(), 10);
    EXPECT_EQ(root->constraint_type(),
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::IntConstraint);
    EXPECT_EQ(root->constraint_as_IntConstraint()->min_value(), 0);
    EXPECT_EQ(root->constraint_as_IntConstraint()->max_value(), 100);
    EXPECT_EQ(root->constraint_as_IntConstraint()->step(), 1);
    EXPECT_FALSE(root->deprecated());

    auto validValues = root->constraint_as_IntConstraint()->valid_values();
    ASSERT_EQ(validValues->size(), 3);
    EXPECT_EQ(validValues->Get(0), 10);
    EXPECT_EQ(validValues->Get(1), 20);
    EXPECT_EQ(validValues->Get(2), 30);
}

TEST(TestKnobFactory, CreateIntKnobDeprecated)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> options = {};
    auto knob = KnobFactory::createIntKnob(
        builder, "deprecated_int_knob", "deprecated description", 5, 0, 10, 1, options, true);
    builder.Finish(knob);

    auto root = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(
        builder.GetBufferPointer());

    EXPECT_STREQ(root->knob_id()->c_str(), "deprecated_int_knob");
    EXPECT_TRUE(root->deprecated());
}

TEST(TestKnobFactory, CreateFloatKnob)
{
    flatbuffers::FlatBufferBuilder builder;
    auto knob
        = KnobFactory::createFloatKnob(builder, "float_knob", "description", 1.5f, 0.0f, 10.0f);
    builder.Finish(knob);

    auto root = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(
        builder.GetBufferPointer());

    EXPECT_STREQ(root->knob_id()->c_str(), "float_knob");
    EXPECT_STREQ(root->description()->c_str(), "description");
    EXPECT_EQ(root->default_value_type(),
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
    EXPECT_FLOAT_EQ(root->default_value_as_FloatValue()->value(), 1.5f);
    EXPECT_EQ(root->constraint_type(),
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::FloatConstraint);
    EXPECT_FLOAT_EQ(root->constraint_as_FloatConstraint()->min_value(), 0.0f);
    EXPECT_FLOAT_EQ(root->constraint_as_FloatConstraint()->max_value(), 10.0f);
    EXPECT_FALSE(root->deprecated());
}

TEST(TestKnobFactory, CreateFloatKnobDeprecated)
{
    flatbuffers::FlatBufferBuilder builder;
    auto knob = KnobFactory::createFloatKnob(
        builder, "deprecated_float_knob", "deprecated description", 0.5f, 0.0f, 1.0f, true);
    builder.Finish(knob);

    auto root = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(
        builder.GetBufferPointer());

    EXPECT_STREQ(root->knob_id()->c_str(), "deprecated_float_knob");
    EXPECT_TRUE(root->deprecated());
}

TEST(TestKnobFactory, CreateStringKnob)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<std::string> options = {"option1", "option2"};
    auto knob
        = KnobFactory::createStringKnob(builder, "string_knob", "description", "option1", options);
    builder.Finish(knob);

    auto root = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(
        builder.GetBufferPointer());

    EXPECT_STREQ(root->knob_id()->c_str(), "string_knob");
    EXPECT_STREQ(root->description()->c_str(), "description");
    EXPECT_EQ(root->default_value_type(),
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
    EXPECT_STREQ(root->default_value_as_StringValue()->value()->c_str(), "option1");
    EXPECT_EQ(root->constraint_type(),
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::StringConstraint);
    EXPECT_FALSE(root->deprecated());

    auto validValues = root->constraint_as_StringConstraint()->valid_values();
    ASSERT_EQ(validValues->size(), 2);
    EXPECT_STREQ(validValues->Get(0)->c_str(), "option1");
    EXPECT_STREQ(validValues->Get(1)->c_str(), "option2");
}

TEST(TestKnobFactory, CreateStringKnobDeprecated)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<std::string> options = {"a", "b"};
    auto knob = KnobFactory::createStringKnob(
        builder, "deprecated_string_knob", "deprecated description", "a", options, true);
    builder.Finish(knob);

    auto root = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(
        builder.GetBufferPointer());

    EXPECT_STREQ(root->knob_id()->c_str(), "deprecated_string_knob");
    EXPECT_TRUE(root->deprecated());
}
