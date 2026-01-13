// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Knob.hpp>

using namespace hipdnn_frontend;

// ============================================================================
// KnobSetting Tests
// ============================================================================

TEST(TestKnobSetting, ConstructWithInt64)
{
    KnobSetting setting(123, static_cast<int64_t>(42));

    EXPECT_EQ(setting.getKnobId(), 123);
    EXPECT_EQ(setting.getValueType(), KnobValueType::INT64);

    auto value = setting.getValue<int64_t>();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), 42);
}

TEST(TestKnobSetting, ConstructWithDouble)
{
    KnobSetting setting(456, 3.14);

    EXPECT_EQ(setting.getKnobId(), 456);
    EXPECT_EQ(setting.getValueType(), KnobValueType::FLOAT64);

    auto value = setting.getValue<double>();
    ASSERT_TRUE(value.has_value());
    EXPECT_DOUBLE_EQ(value.value(), 3.14);
}

TEST(TestKnobSetting, ConstructWithString)
{
    KnobSetting setting(789, std::string("test_value"));

    EXPECT_EQ(setting.getKnobId(), 789);
    EXPECT_EQ(setting.getValueType(), KnobValueType::STRING);

    auto value = setting.getValue<std::string>();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "test_value");
}

TEST(TestKnobSetting, GetValueWithWrongType)
{
    KnobSetting intSetting(123, static_cast<int64_t>(42));

    // Try to get as wrong types
    EXPECT_FALSE(intSetting.getValue<double>().has_value());
    EXPECT_FALSE(intSetting.getValue<std::string>().has_value());

    KnobSetting floatSetting(456, 3.14);
    EXPECT_FALSE(floatSetting.getValue<int64_t>().has_value());
    EXPECT_FALSE(floatSetting.getValue<std::string>().has_value());

    KnobSetting stringSetting(789, std::string("test"));
    EXPECT_FALSE(stringSetting.getValue<int64_t>().has_value());
    EXPECT_FALSE(stringSetting.getValue<double>().has_value());
}

// ============================================================================
// IntConstraint Tests
// ============================================================================

TEST(TestKnobIntConstraint, ValidateInRange)
{
    IntConstraint constraint(0, 100, 1);
    KnobSetting setting(1, static_cast<int64_t>(50));

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

TEST(TestKnobIntConstraint, ValidateBelowMin)
{
    IntConstraint constraint(10, 100, 1);
    KnobSetting setting(1, static_cast<int64_t>(5));

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("out of range"), std::string::npos);
}

TEST(TestKnobIntConstraint, ValidateAboveMax)
{
    IntConstraint constraint(0, 100, 1);
    KnobSetting setting(1, static_cast<int64_t>(150));

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("out of range"), std::string::npos);
}

TEST(TestKnobIntConstraint, ValidateStride)
{
    IntConstraint constraint(0, 100, 10);

    KnobSetting validSetting(1, static_cast<int64_t>(50));
    EXPECT_EQ(constraint.validateKnobSetting(validSetting).get_code(), ErrorCode::OK);

    KnobSetting invalidSetting(1, static_cast<int64_t>(55));
    Error result = constraint.validateKnobSetting(invalidSetting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("stride constraint"), std::string::npos);
}

TEST(TestKnobIntConstraint, ValidateExplicitValues)
{
    IntConstraint constraint(0, 100, 1, {8, 16, 32, 64});

    KnobSetting validSetting(1, static_cast<int64_t>(32));
    EXPECT_EQ(constraint.validateKnobSetting(validSetting).get_code(), ErrorCode::OK);

    KnobSetting invalidSetting(1, static_cast<int64_t>(24));
    Error result = constraint.validateKnobSetting(invalidSetting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("not in the list of valid values"), std::string::npos);
}

TEST(TestKnobIntConstraint, ValidateWrongType)
{
    IntConstraint constraint(0, 100, 1);
    KnobSetting setting(1, 3.14); // Double instead of int64

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("does not contain an integer value"), std::string::npos);
}

TEST(TestKnobIntConstraint, ToStringWithRange)
{
    IntConstraint constraint(0, 100, 5);
    std::string str = constraint.toString();

    EXPECT_NE(str.find("IntConstraint"), std::string::npos);
    EXPECT_NE(str.find("min=0"), std::string::npos);
    EXPECT_NE(str.find("max=100"), std::string::npos);
    EXPECT_NE(str.find("stride=5"), std::string::npos);
}

TEST(TestKnobIntConstraint, ToStringWithValidValues)
{
    IntConstraint constraint(0, 100, 1, {8, 16, 32});
    std::string str = constraint.toString();

    EXPECT_NE(str.find("validValues"), std::string::npos);
    EXPECT_NE(str.find('8'), std::string::npos);
    EXPECT_NE(str.find("16"), std::string::npos);
    EXPECT_NE(str.find("32"), std::string::npos);
}

// ============================================================================
// FloatConstraint Tests
// ============================================================================

TEST(TestKnobFloatConstraint, ValidateInRange)
{
    FloatConstraint constraint(0.0, 1.0);
    KnobSetting setting(1, 0.5);

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

TEST(TestKnobFloatConstraint, ValidateBelowMin)
{
    FloatConstraint constraint(0.0, 1.0);
    KnobSetting setting(1, -0.5);

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("out of range"), std::string::npos);
}

TEST(TestKnobFloatConstraint, ValidateAboveMax)
{
    FloatConstraint constraint(0.0, 1.0);
    KnobSetting setting(1, 1.5);

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("out of range"), std::string::npos);
}

TEST(TestKnobFloatConstraint, ValidateExplicitValues)
{
    FloatConstraint constraint(0.0, 1.0, {0.1, 0.5, 0.9});

    KnobSetting validSetting(1, 0.5);
    EXPECT_EQ(constraint.validateKnobSetting(validSetting).get_code(), ErrorCode::OK);

    KnobSetting invalidSetting(1, 0.7);
    Error result = constraint.validateKnobSetting(invalidSetting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("not in the list of valid values"), std::string::npos);
}

TEST(TestKnobFloatConstraint, ValidateWrongType)
{
    FloatConstraint constraint(0.0, 1.0);
    KnobSetting setting(1, static_cast<int64_t>(42));

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("does not contain a float value"), std::string::npos);
}

TEST(TestKnobFloatConstraint, ToString)
{
    FloatConstraint constraint(0.0, 1.0, {0.1, 0.5, 0.9});
    std::string str = constraint.toString();

    EXPECT_NE(str.find("FloatConstraint"), std::string::npos);
    EXPECT_NE(str.find("min=0"), std::string::npos);
    EXPECT_NE(str.find("max=1"), std::string::npos);
    EXPECT_NE(str.find("validValues"), std::string::npos);
}

// ============================================================================
// StringConstraint Tests
// ============================================================================

TEST(TestKnobStringConstraint, ValidateWithinMaxLength)
{
    StringConstraint constraint(10);
    KnobSetting setting(1, std::string("short"));

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

TEST(TestKnobStringConstraint, ValidateExceedsMaxLength)
{
    StringConstraint constraint(5);
    KnobSetting setting(1, std::string("toolongstring"));

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("exceeds maximum length"), std::string::npos);
}

TEST(TestKnobStringConstraint, ValidateExplicitValues)
{
    StringConstraint constraint(100, {"option1", "option2", "option3"});

    KnobSetting validSetting(1, std::string("option2"));
    EXPECT_EQ(constraint.validateKnobSetting(validSetting).get_code(), ErrorCode::OK);

    KnobSetting invalidSetting(1, std::string("option4"));
    Error result = constraint.validateKnobSetting(invalidSetting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("not in the list of valid values"), std::string::npos);
}

TEST(TestKnobStringConstraint, ValidateWrongType)
{
    StringConstraint constraint(100);
    KnobSetting setting(1, static_cast<int64_t>(42));

    Error result = constraint.validateKnobSetting(setting);
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.get_message().find("does not contain a string value"), std::string::npos);
}

TEST(TestKnobStringConstraint, ToString)
{
    StringConstraint constraint(50, {"opt1", "opt2"});
    std::string str = constraint.toString();

    EXPECT_NE(str.find("StringConstraint"), std::string::npos);
    EXPECT_NE(str.find("maxLength=50"), std::string::npos);
    EXPECT_NE(str.find("validValues"), std::string::npos);
    EXPECT_NE(str.find("opt1"), std::string::npos);
}

// ============================================================================
// Knob Tests
// ============================================================================

TEST(TestKnob, MakeKnobId)
{
    // Test that the same string produces the same hash
    int64_t id1 = Knob::makeKnobId("test.knob.name");
    int64_t id2 = Knob::makeKnobId("test.knob.name");
    EXPECT_EQ(id1, id2);

    // Test that different strings produce different hashes
    int64_t id3 = Knob::makeKnobId("different.knob.name");
    EXPECT_NE(id1, id3);
}

// Note: Since Knob has a private constructor and requires a factory function
// that depends on flatbuffer schemas (not yet implemented), we cannot directly
// test Knob construction. The following tests would be added once the factory
// function is implemented:
//
// TEST(TestKnob, GetKnobId)
// TEST(TestKnob, GetKnobIdStr)
// TEST(TestKnob, GetDescription)
// TEST(TestKnob, IsDeprecated)
// TEST(TestKnob, GetValueTypeInt64)
// TEST(TestKnob, GetValueTypeFloat64)
// TEST(TestKnob, GetValueTypeString)
// TEST(TestKnob, GetDefaultValueInt64)
// TEST(TestKnob, GetDefaultValueFloat64)
// TEST(TestKnob, GetDefaultValueString)
// TEST(TestKnob, GetDefaultValueWrongType)
// TEST(TestKnob, ToDefaultKnobSetting)
// TEST(TestKnob, ValidateKnobSettingMatchingTypes)
// TEST(TestKnob, ValidateKnobSettingMismatchedTypes)
// TEST(TestKnob, ValidateKnobSettingMismatchedIds)
// TEST(TestKnob, ValidateWithConstraint)

// ============================================================================
// Integration Tests
// ============================================================================

TEST(TestKnobIntegration, IntConstraintWithKnobSetting)
{
    IntConstraint constraint(0, 100, 10, {10, 20, 30, 40, 50});

    // Valid value from list
    KnobSetting validSetting(1, static_cast<int64_t>(30));
    EXPECT_EQ(constraint.validateKnobSetting(validSetting).get_code(), ErrorCode::OK);

    // Invalid value not in list
    KnobSetting invalidSetting(1, static_cast<int64_t>(35));
    EXPECT_EQ(constraint.validateKnobSetting(invalidSetting).get_code(), ErrorCode::INVALID_VALUE);
}

TEST(TestKnobIntegration, FloatConstraintWithKnobSetting)
{
    FloatConstraint constraint(0.0, 1.0);

    KnobSetting validSetting(1, 0.5);
    EXPECT_EQ(constraint.validateKnobSetting(validSetting).get_code(), ErrorCode::OK);

    KnobSetting invalidSetting(1, 1.5);
    EXPECT_EQ(constraint.validateKnobSetting(invalidSetting).get_code(), ErrorCode::INVALID_VALUE);
}

TEST(TestKnobIntegration, StringConstraintWithKnobSetting)
{
    StringConstraint constraint(20, {"small", "medium", "large"});

    KnobSetting validSetting(1, std::string("medium"));
    EXPECT_EQ(constraint.validateKnobSetting(validSetting).get_code(), ErrorCode::OK);

    KnobSetting invalidSetting(1, std::string("extra-large"));
    EXPECT_EQ(constraint.validateKnobSetting(invalidSetting).get_code(), ErrorCode::INVALID_VALUE);
}

TEST(TestKnobIntegration, ConstraintErrorMessages)
{
    IntConstraint intConstraint(0, 10, 1);
    KnobSetting intSetting(1, static_cast<int64_t>(20));
    Error intError = intConstraint.validateKnobSetting(intSetting);
    EXPECT_NE(intError.get_message().find("20"), std::string::npos);
    EXPECT_NE(intError.get_message().find("[0, 10]"), std::string::npos);

    FloatConstraint floatConstraint(0.0, 1.0, {0.1, 0.5, 0.9});
    KnobSetting floatSetting(1, 0.7);
    Error floatError = floatConstraint.validateKnobSetting(floatSetting);
    EXPECT_NE(floatError.get_message().find("0.7"), std::string::npos);
    EXPECT_NE(floatError.get_message().find("0.1"), std::string::npos);

    StringConstraint stringConstraint(10, {"opt1", "opt2"});
    KnobSetting stringSetting(1, std::string("opt3"));
    Error stringError = stringConstraint.validateKnobSetting(stringSetting);
    EXPECT_NE(stringError.get_message().find("opt3"), std::string::npos);
    EXPECT_NE(stringError.get_message().find("opt1"), std::string::npos);
}
