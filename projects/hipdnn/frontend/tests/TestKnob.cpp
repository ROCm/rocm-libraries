// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/knob/Knob.hpp>

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

    auto defaultValueOffset = fb::CreateIntValue(builder, defaultValue);

    flatbuffers::Offset<void> constraintOffset = 0;
    fb::KnobConstraint constraintType = fb::KnobConstraint::NONE;

    if(constraint != nullptr)
    {
        constraintOffset = fb::CreateIntConstraint(builder, constraint).Union();
        constraintType = fb::KnobConstraint::IntConstraint;
    }

    auto knobOffset = fb::CreateKnobDirect(builder,
                                           knobIdStr.c_str(),
                                           description.c_str(),
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

    auto defaultValueOffset = fb::CreateFloatValue(builder, defaultValue);

    flatbuffers::Offset<void> constraintOffset = 0;
    fb::KnobConstraint constraintType = fb::KnobConstraint::NONE;

    if(constraint != nullptr)
    {
        constraintOffset = fb::CreateFloatConstraint(builder, constraint).Union();
        constraintType = fb::KnobConstraint::FloatConstraint;
    }

    auto knobOffset = fb::CreateKnobDirect(builder,
                                           knobIdStr.c_str(),
                                           description.c_str(),
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

    auto defaultValueOffset = fb::CreateStringValueDirect(builder, defaultValue.c_str());

    flatbuffers::Offset<void> constraintOffset = 0;
    fb::KnobConstraint constraintType = fb::KnobConstraint::NONE;

    if(constraint != nullptr)
    {
        constraintOffset = fb::CreateStringConstraint(builder, constraint).Union();
        constraintType = fb::KnobConstraint::StringConstraint;
    }

    auto knobOffset = fb::CreateKnobDirect(builder,
                                           knobIdStr.c_str(),
                                           description.c_str(),
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
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    EXPECT_EQ(knob.knobId(), "test_int_knob");
    EXPECT_EQ(knob.description(), "Test integer knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::INT64);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = std::get_if<int64_t>(&knob.defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, 42);
}

TEST(TestKnob, CreateFloatKnob)
{
    auto buffer = createFloatKnobFlatbuffer("test_float_knob", "Test float knob", 3.14);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    EXPECT_EQ(knob.knobId(), "test_float_knob");
    EXPECT_EQ(knob.description(), "Test float knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::FLOAT64);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = std::get_if<double>(&knob.defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_DOUBLE_EQ(*defaultValue, 3.14);
}

TEST(TestKnob, CreateStringKnob)
{
    auto buffer = createStringKnobFlatbuffer("test_string_knob", "Test string knob", "default");
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    EXPECT_EQ(knob.knobId(), "test_string_knob");
    EXPECT_EQ(knob.description(), "Test string knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::STRING);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = std::get_if<std::string>(&knob.defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, "default");
}

TEST(TestKnob, CreateDeprecatedKnob)
{
    auto buffer = createIntKnobFlatbuffer("deprecated_knob", "Deprecated knob", 0, true);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    EXPECT_TRUE(knob.isDeprecated());
}

// ============================================================================
// KnobSetting Tests
// ============================================================================

TEST(TestKnobSetting, CreateIntKnobSetting)
{
    KnobSetting setting("test.knob", 42);

    EXPECT_EQ(setting.knobId(), "test.knob");
    auto value = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, 42);
}

TEST(TestKnobSetting, CreateFloatKnobSetting)
{
    KnobSetting setting("test.knob", 3.14);

    EXPECT_EQ(setting.knobId(), "test.knob");
    auto value = std::get_if<double>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_DOUBLE_EQ(*value, 3.14);
}

TEST(TestKnobSetting, CreateStringKnobSetting)
{
    KnobSetting setting("test.knob", std::string("custom_value"));

    EXPECT_EQ(setting.knobId(), "test.knob");
    auto value = std::get_if<std::string>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, "custom_value");
}

TEST(TestKnobSetting, SetAndGetValue)
{
    KnobSetting setting("test.knob", 42);

    auto value1 = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(value1, nullptr);
    EXPECT_EQ(*value1, 42);

    setting.setValue(100);

    auto value2 = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(value2, nullptr);
    EXPECT_EQ(*value2, 100);
}

TEST(TestKnobSetting, GetValueWrongType)
{
    KnobSetting setting("test.knob", 42);

    // Try to get as wrong type
    auto wrongValue = std::get_if<double>(&setting.value());
    EXPECT_EQ(wrongValue, nullptr);

    auto stringValue = std::get_if<std::string>(&setting.value());
    EXPECT_EQ(stringValue, nullptr);
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
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    auto constraint = knob.constraint();
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
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    auto constraint = knob.constraint();
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
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    auto constraint = knob.constraint();
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
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Valid value
    KnobSetting setting1(knob.knobId(), 50);
    auto err = knob.validate(setting1);
    EXPECT_EQ(err.code, ErrorCode::OK);

    // Value at min boundary
    KnobSetting setting2(knob.knobId(), 0);
    err = knob.validate(setting2);
    EXPECT_EQ(err.code, ErrorCode::OK);

    // Value at max boundary
    KnobSetting setting3(knob.knobId(), 100);
    err = knob.validate(setting3);
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestKnob, ValidateIntKnobOutOfRange)
{
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;

    auto buffer = createIntKnobFlatbuffer("test_knob", "Test knob", 50, false, &constraintT);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Value below min
    KnobSetting setting1(knob.knobId(), -1);
    auto err = knob.validate(setting1);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("out of range"), std::string::npos);

    // Value above max
    KnobSetting setting2(knob.knobId(), 101);
    err = knob.validate(setting2);
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
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Valid step values
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 0)).code, ErrorCode::OK);
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 10)).code, ErrorCode::OK);
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 100)).code, ErrorCode::OK);

    // Invalid step value
    KnobSetting invalidSetting(knob.knobId(), 15);
    auto err = knob.validate(invalidSetting);
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
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Valid values
    for(auto validVal : {1, 2, 4, 8, 16})
    {
        KnobSetting setting(knob.knobId(), validVal);
        EXPECT_EQ(knob.validate(setting).code, ErrorCode::OK);
    }

    // Invalid value
    KnobSetting invalidSetting(knob.knobId(), 3);
    auto err = knob.validate(invalidSetting);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("not in the list of valid values"), std::string::npos);
}

TEST(TestKnob, ValidateFloatKnobInRange)
{
    fb::FloatConstraintT constraintT;
    constraintT.min_value = 0.0;
    constraintT.max_value = 1.0;

    auto buffer = createFloatKnobFlatbuffer("test_knob", "Test knob", 0.5, false, &constraintT);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Valid value
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 0.5)).code, ErrorCode::OK);

    // Boundary values
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 0.0)).code, ErrorCode::OK);
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 1.0)).code, ErrorCode::OK);
}

TEST(TestKnob, ValidateFloatKnobOutOfRange)
{
    fb::FloatConstraintT constraintT;
    constraintT.min_value = 0.0;
    constraintT.max_value = 1.0;

    auto buffer = createFloatKnobFlatbuffer("test_knob", "Test knob", 0.5, false, &constraintT);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Value below min
    KnobSetting setting1(knob.knobId(), -0.1);
    auto err = knob.validate(setting1);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);

    // Value above max
    KnobSetting setting2(knob.knobId(), 1.1);
    err = knob.validate(setting2);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
}

TEST(TestKnob, ValidateStringKnobWithValidValues)
{
    fb::StringConstraintT constraintT;
    constraintT.max_length = 100;
    constraintT.valid_values = {"option1", "option2", "option3"};

    auto buffer
        = createStringKnobFlatbuffer("test_knob", "Test knob", "option1", false, &constraintT);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Valid values
    for(const auto& validVal : {"option1", "option2", "option3"})
    {
        KnobSetting setting(knob.knobId(), std::string(validVal));
        EXPECT_EQ(knob.validate(setting).code, ErrorCode::OK);
    }

    // Invalid value
    KnobSetting invalidSetting(knob.knobId(), std::string("invalid"));
    auto err = knob.validate(invalidSetting);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("not in the list of valid values"), std::string::npos);
}

TEST(TestKnob, ValidateStringKnobMaxLength)
{
    fb::StringConstraintT constraintT;
    constraintT.max_length = 10;

    auto buffer
        = createStringKnobFlatbuffer("test_knob", "Test knob", "short", false, &constraintT);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Valid length
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), std::string("short"))).code, ErrorCode::OK);

    // Exactly at max length
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), std::string("1234567890"))).code,
              ErrorCode::OK);

    // Exceeds max length
    KnobSetting invalidSetting(knob.knobId(), std::string("12345678901"));
    auto err = knob.validate(invalidSetting);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("exceeds maximum length"), std::string::npos);
}

// ============================================================================
// toString Tests
// ============================================================================

TEST(TestKnob, ToStringIntKnob)
{
    auto buffer = createIntKnobFlatbuffer("test_knob", "Test description", 42);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    std::string str = knob.toString();

    EXPECT_NE(str.find("knobIdStr=\"test_knob\""), std::string::npos);
    EXPECT_NE(str.find("description=\"Test description\""), std::string::npos);
    EXPECT_NE(str.find("defaultValue=42"), std::string::npos);
    EXPECT_NE(str.find("deprecated=false"), std::string::npos);
}

TEST(TestKnob, ToStringFloatKnob)
{
    auto buffer = createFloatKnobFlatbuffer("float_knob", "Float test", 3.14);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    std::string str = knob.toString();

    EXPECT_NE(str.find("defaultValue=3.14"), std::string::npos);
}

TEST(TestKnob, ToStringStringKnob)
{
    auto buffer = createStringKnobFlatbuffer("string_knob", "String test", "default");
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    std::string str = knob.toString();

    EXPECT_NE(str.find("defaultValue=\"default\""), std::string::npos);
}

TEST(TestKnobSetting, ToStringIntSetting)
{
    KnobSetting setting("test.knob", 42);

    std::string str = setting.toString();

    EXPECT_NE(str.find("KnobSetting"), std::string::npos);
    EXPECT_NE(str.find("value=42"), std::string::npos);
}

TEST(TestKnobSetting, ToStringFloatSetting)
{
    KnobSetting setting("test.knob", 3.14);

    std::string str = setting.toString();

    EXPECT_NE(str.find("value=3.14"), std::string::npos);
}

TEST(TestKnobSetting, ToStringStringSetting)
{
    KnobSetting setting("test.knob", std::string("custom"));

    std::string str = setting.toString();

    EXPECT_NE(str.find("value=\"custom\""), std::string::npos);
}

TEST(TestKnob, ToStringDeprecatedKnob)
{
    auto buffer = createIntKnobFlatbuffer("deprecated", "Deprecated knob", 0, true);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

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
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    std::string str = knob.toString();

    EXPECT_NE(str.find("constraint="), std::string::npos);
    EXPECT_NE(str.find("IntConstraint"), std::string::npos);
}

TEST(TestKnob, GetDefaultValueWrongType)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 42);
    hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto knob = hipdnn_frontend::Knob::fromFlatbuffer(bufferData);

    // Try to get default value as wrong type
    auto wrongDefault = std::get_if<double>(&knob.defaultValue());
    EXPECT_EQ(wrongDefault, nullptr);

    auto stringDefault = std::get_if<std::string>(&knob.defaultValue());
    EXPECT_EQ(stringDefault, nullptr);

    // Correct type should work
    auto correctDefault = std::get_if<int64_t>(&knob.defaultValue());
    ASSERT_NE(correctDefault, nullptr);
    EXPECT_EQ(*correctDefault, 42);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TestKnobSetting, EmptyStringValue)
{
    KnobSetting setting("test.knob", std::string(""));

    auto value = std::get_if<std::string>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, "");
}

TEST(TestKnobSetting, NegativeIntValue)
{
    KnobSetting setting("test.knob", static_cast<int64_t>(-100));

    auto value = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, -100);
}

TEST(TestKnobSetting, ZeroValues)
{
    KnobSetting intSetting("test.int", static_cast<int64_t>(0));
    KnobSetting floatSetting("test.float", 0.0);

    auto intValue = std::get_if<int64_t>(&intSetting.value());
    ASSERT_NE(intValue, nullptr);
    EXPECT_EQ(*intValue, 0);

    auto floatValue = std::get_if<double>(&floatSetting.value());
    ASSERT_NE(floatValue, nullptr);
    EXPECT_DOUBLE_EQ(*floatValue, 0.0);
}

TEST(TestKnobSetting, LargeIntValue)
{
    int64_t largeValue = 9223372036854775807LL; // INT64_MAX
    KnobSetting setting("test.knob", largeValue);

    auto value = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, largeValue);
}

TEST(TestKnobSetting, LongStringValue)
{
    std::string longString(1000, 'a');
    KnobSetting setting("test.knob", longString);

    auto value = std::get_if<std::string>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, longString);
}

TEST(TestKnobSetting, SpecialCharactersInString)
{
    std::string specialChars = "Test\nwith\ttabs\rand\"quotes\"";
    KnobSetting setting("test.knob", specialChars);

    auto value = std::get_if<std::string>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, specialChars);
}

// ============================================================================
// packKnobSetting Tests
// ============================================================================

TEST(TestKnobSetting, PackIntKnobSetting)
{
    KnobSetting setting("test.knob", 42);

    flatbuffers::FlatBufferBuilder builder;
    auto knobSettingOffset = setting.packKnobSetting(builder);
    builder.Finish(knobSettingOffset);

    auto packedKnobSetting = flatbuffers::GetRoot<fb::KnobSetting>(builder.GetBufferPointer());
    EXPECT_EQ(packedKnobSetting->knob_id()->str(), setting.knobId());
    EXPECT_EQ(packedKnobSetting->value_type(), fb::KnobValue::IntValue);

    auto intValue = packedKnobSetting->value_as_IntValue();
    ASSERT_NE(intValue, nullptr);
    EXPECT_EQ(intValue->value(), 42);
}

TEST(TestKnobSetting, PackFloatKnobSetting)
{
    KnobSetting setting("test.knob", 3.14);

    flatbuffers::FlatBufferBuilder builder;
    auto knobSettingOffset = setting.packKnobSetting(builder);
    builder.Finish(knobSettingOffset);

    auto packedKnobSetting = flatbuffers::GetRoot<fb::KnobSetting>(builder.GetBufferPointer());
    EXPECT_EQ(packedKnobSetting->knob_id()->str(), setting.knobId());
    EXPECT_EQ(packedKnobSetting->value_type(), fb::KnobValue::FloatValue);

    auto floatValue = packedKnobSetting->value_as_FloatValue();
    ASSERT_NE(floatValue, nullptr);
    EXPECT_DOUBLE_EQ(floatValue->value(), 3.14);
}

TEST(TestKnobSetting, PackStringKnobSetting)
{
    KnobSetting setting("test.knob", std::string("custom_value"));

    flatbuffers::FlatBufferBuilder builder;
    auto knobSettingOffset = setting.packKnobSetting(builder);
    builder.Finish(knobSettingOffset);

    auto packedKnobSetting = flatbuffers::GetRoot<fb::KnobSetting>(builder.GetBufferPointer());
    EXPECT_EQ(packedKnobSetting->knob_id()->str(), setting.knobId());
    EXPECT_EQ(packedKnobSetting->value_type(), fb::KnobValue::StringValue);

    auto stringValue = packedKnobSetting->value_as_StringValue();
    ASSERT_NE(stringValue, nullptr);
    ASSERT_NE(stringValue->value(), nullptr);
    EXPECT_EQ(stringValue->value()->str(), "custom_value");
}

TEST(TestKnobSetting, PackKnobSettingWithEmptyString)
{
    KnobSetting setting("test.knob", std::string(""));

    flatbuffers::FlatBufferBuilder builder;
    auto knobSettingOffset = setting.packKnobSetting(builder);
    builder.Finish(knobSettingOffset);

    auto packedKnobSetting = flatbuffers::GetRoot<fb::KnobSetting>(builder.GetBufferPointer());
    auto stringValue = packedKnobSetting->value_as_StringValue();
    ASSERT_NE(stringValue, nullptr);
    ASSERT_NE(stringValue->value(), nullptr);
    EXPECT_EQ(stringValue->value()->str(), "");
}

TEST(TestKnobSetting, PackKnobSettingWithLongString)
{
    std::string longString(1000, 'a');
    KnobSetting setting("test.knob", longString);

    flatbuffers::FlatBufferBuilder builder;
    auto knobSettingOffset = setting.packKnobSetting(builder);
    builder.Finish(knobSettingOffset);

    auto packedKnobSetting = flatbuffers::GetRoot<fb::KnobSetting>(builder.GetBufferPointer());
    auto stringValue = packedKnobSetting->value_as_StringValue();
    ASSERT_NE(stringValue, nullptr);
    ASSERT_NE(stringValue->value(), nullptr);
    EXPECT_EQ(stringValue->value()->str(), longString);
}

TEST(TestKnobSetting, PackKnobSettingWithSpecialCharacters)
{
    std::string specialChars = "Test\nwith\ttabs\rand\"quotes\"";
    KnobSetting setting("test.knob", specialChars);

    flatbuffers::FlatBufferBuilder builder;
    auto knobSettingOffset = setting.packKnobSetting(builder);
    builder.Finish(knobSettingOffset);

    auto packedKnobSetting = flatbuffers::GetRoot<fb::KnobSetting>(builder.GetBufferPointer());
    auto stringValue = packedKnobSetting->value_as_StringValue();
    ASSERT_NE(stringValue, nullptr);
    ASSERT_NE(stringValue->value(), nullptr);
    EXPECT_EQ(stringValue->value()->str(), specialChars);
}

TEST(TestKnobSetting, PackKnobSettingPreservesKnobId)
{
    KnobSetting setting1("knob_a", 1);
    KnobSetting setting2("knob_b", 2);

    flatbuffers::FlatBufferBuilder builder1;
    auto knobSetting1Offset = setting1.packKnobSetting(builder1);
    builder1.Finish(knobSetting1Offset);

    flatbuffers::FlatBufferBuilder builder2;
    auto knobSetting2Offset = setting2.packKnobSetting(builder2);
    builder2.Finish(knobSetting2Offset);

    auto packedKnobSetting1 = flatbuffers::GetRoot<fb::KnobSetting>(builder1.GetBufferPointer());
    auto packedKnobSetting2 = flatbuffers::GetRoot<fb::KnobSetting>(builder2.GetBufferPointer());

    EXPECT_EQ(packedKnobSetting1->knob_id()->str(), setting1.knobId());
    EXPECT_EQ(packedKnobSetting2->knob_id()->str(), setting2.knobId());
    EXPECT_NE(packedKnobSetting1->knob_id()->str(), packedKnobSetting2->knob_id()->str());
}
