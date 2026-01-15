// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Knob.hpp>

using namespace hipdnn_frontend;
namespace fb = hipdnn_data_sdk::data_objects;

// Helper functions to create flatbuffer Knob objects for testing
namespace
{

flatbuffers::DetachedBuffer createIntKnobFlatbuffer(const std::string& knobIdStr,
                                                    const std::string& description,
                                                    int64_t defaultValue,
                                                    bool deprecated = false,
                                                    const fb::IntConstraintT* constraint = nullptr)
{
    flatbuffers::FlatBufferBuilder builder;

    auto knobIdStrOffset = builder.CreateString(knobIdStr);
    auto descriptionOffset = builder.CreateString(description);

    auto defaultValueOffset = fb::CreateIntValue(builder, defaultValue);

    flatbuffers::Offset<void> constraintOffset = 0;
    fb::KnobConstraint constraintType = fb::KnobConstraint::NONE;

    if(constraint != nullptr)
    {
        constraintOffset = fb::CreateIntConstraint(builder, constraint).Union();
        constraintType = fb::KnobConstraint::IntConstraint;
    }

    auto knobOffset = fb::CreateKnob(builder,
                                     hipdnn_frontend::Knob::makeKnobId(knobIdStr),
                                     knobIdStrOffset,
                                     descriptionOffset,
                                     fb::KnobValue::IntValue,
                                     defaultValueOffset.Union(),
                                     fb::KnobValue::IntValue,
                                     defaultValueOffset.Union(),
                                     constraintType,
                                     constraintOffset,
                                     deprecated);

    builder.Finish(knobOffset);
    return builder.Release();
}

flatbuffers::DetachedBuffer createFloatKnobFlatbuffer(const std::string& knobIdStr,
                                                      const std::string& description,
                                                      double defaultValue,
                                                      bool deprecated = false,
                                                      const fb::FloatConstraintT* constraint
                                                      = nullptr)
{
    flatbuffers::FlatBufferBuilder builder;

    auto knobIdStrOffset = builder.CreateString(knobIdStr);
    auto descriptionOffset = builder.CreateString(description);

    auto defaultValueOffset = fb::CreateFloatValue(builder, defaultValue);

    flatbuffers::Offset<void> constraintOffset = 0;
    fb::KnobConstraint constraintType = fb::KnobConstraint::NONE;

    if(constraint != nullptr)
    {
        constraintOffset = fb::CreateFloatConstraint(builder, constraint).Union();
        constraintType = fb::KnobConstraint::FloatConstraint;
    }

    auto knobOffset = fb::CreateKnob(builder,
                                     hipdnn_frontend::Knob::makeKnobId(knobIdStr),
                                     knobIdStrOffset,
                                     descriptionOffset,
                                     fb::KnobValue::FloatValue,
                                     defaultValueOffset.Union(),
                                     fb::KnobValue::FloatValue,
                                     defaultValueOffset.Union(),
                                     constraintType,
                                     constraintOffset,
                                     deprecated);

    builder.Finish(knobOffset);
    return builder.Release();
}

flatbuffers::DetachedBuffer createStringKnobFlatbuffer(const std::string& knobIdStr,
                                                       const std::string& description,
                                                       const std::string& defaultValue,
                                                       bool deprecated = false,
                                                       const fb::StringConstraintT* constraint
                                                       = nullptr)
{
    flatbuffers::FlatBufferBuilder builder;

    auto knobIdStrOffset = builder.CreateString(knobIdStr);
    auto descriptionOffset = builder.CreateString(description);
    auto defaultValueOffset = fb::CreateStringValueDirect(builder, defaultValue.c_str());

    flatbuffers::Offset<void> constraintOffset = 0;
    fb::KnobConstraint constraintType = fb::KnobConstraint::NONE;

    if(constraint != nullptr)
    {
        constraintOffset = fb::CreateStringConstraint(builder, constraint).Union();
        constraintType = fb::KnobConstraint::StringConstraint;
    }

    auto knobOffset = fb::CreateKnob(builder,
                                     hipdnn_frontend::Knob::makeKnobId(knobIdStr),
                                     knobIdStrOffset,
                                     descriptionOffset,
                                     fb::KnobValue::StringValue,
                                     defaultValueOffset.Union(),
                                     fb::KnobValue::StringValue,
                                     defaultValueOffset.Union(),
                                     constraintType,
                                     constraintOffset,
                                     deprecated);

    builder.Finish(knobOffset);
    return builder.Release();
}

} // anonymous namespace

// ============================================================================
// Basic Knob Creation and Accessors Tests
// ============================================================================

TEST(TestKnob, CreateIntKnob)
{
    auto buffer = createIntKnobFlatbuffer("test_int_knob", "Test integer knob", 42);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_EQ(knob.getKnobIdStr(), "test_int_knob");
    EXPECT_EQ(knob.getDescription(), "Test integer knob");
    EXPECT_EQ(knob.getValueType(), KnobValueType::INT64);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = std::get_if<int64_t>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, 42);
}

TEST(TestKnob, CreateFloatKnob)
{
    auto buffer = createFloatKnobFlatbuffer("test_float_knob", "Test float knob", 3.14);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_EQ(knob.getKnobIdStr(), "test_float_knob");
    EXPECT_EQ(knob.getDescription(), "Test float knob");
    EXPECT_EQ(knob.getValueType(), KnobValueType::FLOAT64);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = std::get_if<double>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_DOUBLE_EQ(*defaultValue, 3.14);
}

TEST(TestKnob, CreateStringKnob)
{
    auto buffer = createStringKnobFlatbuffer("test_string_knob", "Test string knob", "default");
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_EQ(knob.getKnobIdStr(), "test_string_knob");
    EXPECT_EQ(knob.getDescription(), "Test string knob");
    EXPECT_EQ(knob.getValueType(), KnobValueType::STRING);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = std::get_if<std::string>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, "default");
}

TEST(TestKnob, CreateDeprecatedKnob)
{
    auto buffer = createIntKnobFlatbuffer("deprecated_knob", "Deprecated knob", 0, true);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_TRUE(knob.isDeprecated());
}

TEST(TestKnob, KnobIdGeneration)
{
    auto buffer1 = createIntKnobFlatbuffer("knob_a", "First knob", 1);
    auto fbKnob1 = flatbuffers::GetRoot<fb::Knob>(buffer1.data());
    auto knob1 = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob1);

    auto buffer2 = createIntKnobFlatbuffer("knob_b", "Second knob", 2);
    auto fbKnob2 = flatbuffers::GetRoot<fb::Knob>(buffer2.data());
    auto knob2 = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob2);

    auto buffer3 = createIntKnobFlatbuffer("knob_a", "Duplicate ID", 3);
    auto fbKnob3 = flatbuffers::GetRoot<fb::Knob>(buffer3.data());
    auto knob3 = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob3);

    // Different string IDs should produce different numeric IDs
    EXPECT_NE(knob1.getKnobId(), knob2.getKnobId());

    // Same string ID should produce same numeric ID
    EXPECT_EQ(knob1.getKnobId(), knob3.getKnobId());
}

// ============================================================================
// setChoice and getChoice Tests
// ============================================================================

TEST(TestKnob, SetAndGetIntChoice)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 10);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Set a choice
    knob.setChoice<int64_t>(42);

    // Get the choice back
    auto choice = std::get_if<int64_t>(&knob.getChoice());
    ASSERT_NE(choice, nullptr);
    EXPECT_EQ(*choice, 42);

    // Default value should remain unchanged
    auto defaultValue = std::get_if<int64_t>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, 10);
}

TEST(TestKnob, SetAndGetFloatChoice)
{
    auto buffer = createFloatKnobFlatbuffer("float_knob", "Float knob", 1.0);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    knob.setChoice<double>(2.718);

    auto choice = std::get_if<double>(&knob.getChoice());
    ASSERT_NE(choice, nullptr);
    EXPECT_DOUBLE_EQ(*choice, 2.718);
}

TEST(TestKnob, SetAndGetStringChoice)
{
    auto buffer = createStringKnobFlatbuffer("string_knob", "String knob", "default");
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    knob.setChoice<std::string>("custom_value");

    auto choice = std::get_if<std::string>(&knob.getChoice());
    ASSERT_NE(choice, nullptr);
    EXPECT_EQ(*choice, "custom_value");
}

TEST(TestKnob, GetChoiceWrongType)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 10);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    knob.setChoice<int64_t>(42);

    // Try to get as wrong type
    auto wrongChoice = std::get_if<double>(&knob.getChoice());
    EXPECT_EQ(wrongChoice, nullptr);

    auto stringChoice = std::get_if<std::string>(&knob.getChoice());
    EXPECT_EQ(stringChoice, nullptr);
}

TEST(TestKnob, UpdateChoice)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 10);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    knob.setChoice<int64_t>(42);
    auto choice1 = std::get_if<int64_t>(&knob.getChoice());
    ASSERT_NE(choice1, nullptr);
    EXPECT_EQ(*choice1, 42);

    // Update the choice
    knob.setChoice<int64_t>(100);
    auto choice2 = std::get_if<int64_t>(&knob.getChoice());
    ASSERT_NE(choice2, nullptr);
    EXPECT_EQ(*choice2, 100);
}

// ============================================================================
// hasChoice Tests
// ============================================================================

TEST(TestKnob, HasChoiceInitialStateInt)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 10);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Newly created knob should not have a choice set
    EXPECT_FALSE(knob.hasChoice());
}

TEST(TestKnob, HasChoiceInitialStateFloat)
{
    auto buffer = createFloatKnobFlatbuffer("float_knob", "Float knob", 1.0);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_FALSE(knob.hasChoice());
}

TEST(TestKnob, HasChoiceInitialStateString)
{
    auto buffer = createStringKnobFlatbuffer("string_knob", "String knob", "default");
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_FALSE(knob.hasChoice());
}

TEST(TestKnob, HasChoiceAfterSetInt)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 10);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_FALSE(knob.hasChoice());

    knob.setChoice<int64_t>(42);

    EXPECT_TRUE(knob.hasChoice());
}

TEST(TestKnob, HasChoiceAfterSetFloat)
{
    auto buffer = createFloatKnobFlatbuffer("float_knob", "Float knob", 1.0);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_FALSE(knob.hasChoice());

    knob.setChoice<double>(2.718);

    EXPECT_TRUE(knob.hasChoice());
}

TEST(TestKnob, HasChoiceAfterSetString)
{
    auto buffer = createStringKnobFlatbuffer("string_knob", "String knob", "default");
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_FALSE(knob.hasChoice());

    knob.setChoice<std::string>("custom");

    EXPECT_TRUE(knob.hasChoice());
}

TEST(TestKnob, HasChoiceAfterMultipleUpdates)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 10);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    EXPECT_FALSE(knob.hasChoice());

    // Set choice first time
    knob.setChoice<int64_t>(42);
    EXPECT_TRUE(knob.hasChoice());

    // Update choice - should still be true
    knob.setChoice<int64_t>(100);
    EXPECT_TRUE(knob.hasChoice());

    // Update again
    knob.setChoice<int64_t>(200);
    EXPECT_TRUE(knob.hasChoice());
}

// ============================================================================
// Constraint Tests
// ============================================================================

TEST(TestKnobIntConstraint, ConstraintToString)
{
    hipdnn_frontend::IntConstraint constraint(0, 100, 1);
    std::string str = constraint.toString();

    EXPECT_NE(str.find("IntConstraint"), std::string::npos);
    EXPECT_NE(str.find("min=0"), std::string::npos);
    EXPECT_NE(str.find("max=100"), std::string::npos);
    EXPECT_NE(str.find("step=1"), std::string::npos);
}

TEST(TestKnobIntConstraint, ConstraintWithValidValues)
{
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;
    constraintT.valid_values = {1, 2, 4, 8, 16};

    auto buffer = createIntKnobFlatbuffer("test_knob", "Test knob", 1, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    auto constraint = knob.getConstraint();
    ASSERT_NE(constraint, nullptr);

    std::string str = constraint->toString();
    EXPECT_NE(str.find("validValues"), std::string::npos);
    // Values should be sorted in output
    EXPECT_NE(str.find("[1, 2, 4, 8, 16]"), std::string::npos);
}

TEST(TestKnobFloatConstraint, ConstraintToString)
{
    hipdnn_frontend::FloatConstraint constraint(0.0, 1.0);
    std::string str = constraint.toString();

    EXPECT_NE(str.find("FloatConstraint"), std::string::npos);
    EXPECT_NE(str.find("min=0"), std::string::npos);
    EXPECT_NE(str.find("max=1"), std::string::npos);
}

TEST(TestKnobFloatConstraint, ConstraintFromFlatbuffer)
{
    fb::FloatConstraintT constraintT;
    constraintT.min_value = 0.0;
    constraintT.max_value = 1.0;

    auto buffer = createFloatKnobFlatbuffer("test_knob", "Test knob", 0.5, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    auto constraint = knob.getConstraint();
    ASSERT_NE(constraint, nullptr);

    std::string str = constraint->toString();
    EXPECT_NE(str.find("FloatConstraint"), std::string::npos);
}

TEST(TestKnobStringConstraint, ConstraintToString)
{
    hipdnn_frontend::StringConstraint constraint(100);
    std::string str = constraint.toString();

    EXPECT_NE(str.find("StringConstraint"), std::string::npos);
    EXPECT_NE(str.find("maxLength=100"), std::string::npos);
}

TEST(TestKnobStringConstraint, ConstraintWithValidValues)
{
    fb::StringConstraintT constraintT;
    constraintT.max_length = 100;
    constraintT.valid_values = {"option1", "option2", "option3"};

    auto buffer
        = createStringKnobFlatbuffer("test_knob", "Test knob", "option1", false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    auto constraint = knob.getConstraint();
    ASSERT_NE(constraint, nullptr);

    std::string str = constraint->toString();
    EXPECT_NE(str.find("validValues"), std::string::npos);
    // Check that values are quoted
    EXPECT_NE(str.find("\"option1\""), std::string::npos);
    EXPECT_NE(str.find("\"option2\""), std::string::npos);
    EXPECT_NE(str.find("\"option3\""), std::string::npos);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(TestKnob, ValidateIntKnobInRange)
{
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;

    auto buffer = createIntKnobFlatbuffer("test_knob", "Test knob", 50, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Valid value
    knob.setChoice<int64_t>(50);
    auto err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::OK);

    // Value at min boundary
    knob.setChoice<int64_t>(0);
    err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::OK);

    // Value at max boundary
    knob.setChoice<int64_t>(100);
    err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestKnob, ValidateIntKnobOutOfRange)
{
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;

    auto buffer = createIntKnobFlatbuffer("test_knob", "Test knob", 50, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Value below min
    knob.setChoice<int64_t>(-1);
    auto err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("out of range"), std::string::npos);

    // Value above max
    knob.setChoice<int64_t>(101);
    err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("out of range"), std::string::npos);
}

TEST(TestKnob, ValidateIntKnobWithStep)
{
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 10;

    auto buffer = createIntKnobFlatbuffer("test_knob", "Test knob", 0, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Valid step values
    knob.setChoice<int64_t>(0);
    EXPECT_EQ(knob.validate().code, ErrorCode::OK);

    knob.setChoice<int64_t>(10);
    EXPECT_EQ(knob.validate().code, ErrorCode::OK);

    knob.setChoice<int64_t>(100);
    EXPECT_EQ(knob.validate().code, ErrorCode::OK);

    // Invalid step value
    knob.setChoice<int64_t>(15);
    auto err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("step constraint"), std::string::npos);
}

TEST(TestKnob, ValidateIntKnobWithValidValues)
{
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;
    constraintT.valid_values = {1, 2, 4, 8, 16};

    auto buffer = createIntKnobFlatbuffer("test_knob", "Test knob", 1, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Valid values
    for(auto validVal : {1, 2, 4, 8, 16})
    {
        knob.setChoice<int64_t>(validVal);
        EXPECT_EQ(knob.validate().code, ErrorCode::OK);
    }

    // Invalid value
    knob.setChoice<int64_t>(3);
    auto err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("not in the list of valid values"), std::string::npos);
}

TEST(TestKnob, ValidateFloatKnobInRange)
{
    fb::FloatConstraintT constraintT;
    constraintT.min_value = 0.0;
    constraintT.max_value = 1.0;

    auto buffer = createFloatKnobFlatbuffer("test_knob", "Test knob", 0.5, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Valid value
    knob.setChoice<double>(0.5);
    EXPECT_EQ(knob.validate().code, ErrorCode::OK);

    // Boundary values
    knob.setChoice<double>(0.0);
    EXPECT_EQ(knob.validate().code, ErrorCode::OK);

    knob.setChoice<double>(1.0);
    EXPECT_EQ(knob.validate().code, ErrorCode::OK);
}

TEST(TestKnob, ValidateFloatKnobOutOfRange)
{
    fb::FloatConstraintT constraintT;
    constraintT.min_value = 0.0;
    constraintT.max_value = 1.0;

    auto buffer = createFloatKnobFlatbuffer("test_knob", "Test knob", 0.5, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Value below min
    knob.setChoice<double>(-0.1);
    auto err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);

    // Value above max
    knob.setChoice<double>(1.1);
    err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
}

TEST(TestKnob, ValidateStringKnobWithValidValues)
{
    fb::StringConstraintT constraintT;
    constraintT.max_length = 100;
    constraintT.valid_values = {"option1", "option2", "option3"};

    auto buffer
        = createStringKnobFlatbuffer("test_knob", "Test knob", "option1", false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Valid values
    for(const auto& validVal : {"option1", "option2", "option3"})
    {
        knob.setChoice<std::string>(validVal);
        EXPECT_EQ(knob.validate().code, ErrorCode::OK);
    }

    // Invalid value
    knob.setChoice<std::string>("invalid");
    auto err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("not in the list of valid values"), std::string::npos);
}

TEST(TestKnob, ValidateStringKnobMaxLength)
{
    fb::StringConstraintT constraintT;
    constraintT.max_length = 10;

    auto buffer
        = createStringKnobFlatbuffer("test_knob", "Test knob", "short", false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Valid length
    knob.setChoice<std::string>("short");
    EXPECT_EQ(knob.validate().code, ErrorCode::OK);

    // Exactly at max length
    knob.setChoice<std::string>("1234567890");
    EXPECT_EQ(knob.validate().code, ErrorCode::OK);

    // Exceeds max length
    knob.setChoice<std::string>("12345678901");
    auto err = knob.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("exceeds maximum length"), std::string::npos);
}

// ============================================================================
// toString Tests
// ============================================================================

TEST(TestKnob, ToStringIntKnob)
{
    auto buffer = createIntKnobFlatbuffer("test_knob", "Test description", 42);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    knob.setChoice<int64_t>(100);

    std::string str = knob.toString();

    EXPECT_NE(str.find("knobIdStr=\"test_knob\""), std::string::npos);
    EXPECT_NE(str.find("description=\"Test description\""), std::string::npos);
    EXPECT_NE(str.find("defaultValue=42"), std::string::npos);
    EXPECT_NE(str.find("choice=100"), std::string::npos);
    EXPECT_NE(str.find("deprecated=false"), std::string::npos);
}

TEST(TestKnob, ToStringFloatKnob)
{
    auto buffer = createFloatKnobFlatbuffer("float_knob", "Float test", 3.14);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    knob.setChoice<double>(2.718);

    std::string str = knob.toString();

    EXPECT_NE(str.find("defaultValue=3.14"), std::string::npos);
    EXPECT_NE(str.find("choice=2.718"), std::string::npos);
}

TEST(TestKnob, ToStringStringKnob)
{
    auto buffer = createStringKnobFlatbuffer("string_knob", "String test", "default");
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    knob.setChoice<std::string>("custom");

    std::string str = knob.toString();

    EXPECT_NE(str.find("defaultValue=\"default\""), std::string::npos);
    EXPECT_NE(str.find("choice=\"custom\""), std::string::npos);
}

TEST(TestKnob, ToStringDeprecatedKnob)
{
    auto buffer = createIntKnobFlatbuffer("deprecated", "Deprecated knob", 0, true);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    std::string str = knob.toString();

    EXPECT_NE(str.find("deprecated=true"), std::string::npos);
}

TEST(TestKnob, ToStringWithConstraint)
{
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;

    auto buffer = createIntKnobFlatbuffer("test_knob", "Test knob", 50, false, &constraintT);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    std::string str = knob.toString();

    EXPECT_NE(str.find("constraint="), std::string::npos);
    EXPECT_NE(str.find("IntConstraint"), std::string::npos);
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST(TestKnob, GetDefaultValueWrongType)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 42);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    // Try to get default value as wrong type
    auto wrongDefault = std::get_if<double>(&knob.getDefaultValue());
    EXPECT_EQ(wrongDefault, nullptr);

    auto stringDefault = std::get_if<std::string>(&knob.getDefaultValue());
    EXPECT_EQ(stringDefault, nullptr);

    // Correct type should work
    auto correctDefault = std::get_if<int64_t>(&knob.getDefaultValue());
    ASSERT_NE(correctDefault, nullptr);
    EXPECT_EQ(*correctDefault, 42);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TestKnob, EmptyStringKnob)
{
    auto buffer = createStringKnobFlatbuffer("empty_string", "Empty string knob", "");
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    auto defaultValue = std::get_if<std::string>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, "");

    knob.setChoice<std::string>("");
    auto choice = std::get_if<std::string>(&knob.getChoice());
    ASSERT_NE(choice, nullptr);
    EXPECT_EQ(*choice, "");
}

TEST(TestKnob, NegativeIntValues)
{
    auto buffer = createIntKnobFlatbuffer("negative_knob", "Negative values", -100);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    auto defaultValue = std::get_if<int64_t>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, -100);

    knob.setChoice<int64_t>(-50);
    auto choice = std::get_if<int64_t>(&knob.getChoice());
    ASSERT_NE(choice, nullptr);
    EXPECT_EQ(*choice, -50);
}

TEST(TestKnob, ZeroValues)
{
    auto intBuffer = createIntKnobFlatbuffer("zero_int", "Zero int", 0);
    auto fbIntKnob = flatbuffers::GetRoot<fb::Knob>(intBuffer.data());
    auto intKnob = hipdnn_frontend::Knob::fromFlatbuffer(fbIntKnob);

    auto floatBuffer = createFloatKnobFlatbuffer("zero_float", "Zero float", 0.0);
    auto fbFloatKnob = flatbuffers::GetRoot<fb::Knob>(floatBuffer.data());
    auto floatKnob = hipdnn_frontend::Knob::fromFlatbuffer(fbFloatKnob);

    auto intDefault = std::get_if<int64_t>(&intKnob.getDefaultValue());
    ASSERT_NE(intDefault, nullptr);
    EXPECT_EQ(*intDefault, 0);

    auto floatDefault = std::get_if<double>(&floatKnob.getDefaultValue());
    ASSERT_NE(floatDefault, nullptr);
    EXPECT_DOUBLE_EQ(*floatDefault, 0.0);
}

TEST(TestKnob, LargeIntValues)
{
    int64_t largeValue = 9223372036854775807LL; // INT64_MAX
    auto buffer = createIntKnobFlatbuffer("large_knob", "Large value", largeValue);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    auto defaultValue = std::get_if<int64_t>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, largeValue);
}

TEST(TestKnob, SpecialFloatValues)
{
    auto buffer = createFloatKnobFlatbuffer("special_float", "Special float", 1.23456789);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    knob.setChoice<double>(0.0);
    auto zero = std::get_if<double>(&knob.getChoice());
    ASSERT_NE(zero, nullptr);
    EXPECT_DOUBLE_EQ(*zero, 0.0);

    knob.setChoice<double>(-1.5);
    auto negative = std::get_if<double>(&knob.getChoice());
    ASSERT_NE(negative, nullptr);
    EXPECT_DOUBLE_EQ(*negative, -1.5);
}

TEST(TestKnob, LongStrings)
{
    std::string longString(1000, 'a');
    auto buffer = createStringKnobFlatbuffer("long_string", "Long string knob", longString);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    auto defaultValue = std::get_if<std::string>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, longString);

    std::string anotherLongString(500, 'b');
    knob.setChoice<std::string>(anotherLongString);
    auto choice = std::get_if<std::string>(&knob.getChoice());
    ASSERT_NE(choice, nullptr);
    EXPECT_EQ(*choice, anotherLongString);
}

TEST(TestKnob, SpecialCharactersInStrings)
{
    std::string specialChars = "Test\nwith\ttabs\rand\"quotes\"";
    auto buffer = createStringKnobFlatbuffer("special_chars", "Special characters", specialChars);
    auto fbKnob = flatbuffers::GetRoot<fb::Knob>(buffer.data());
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(fbKnob);

    auto defaultValue = std::get_if<std::string>(&knob.getDefaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, specialChars);
}
