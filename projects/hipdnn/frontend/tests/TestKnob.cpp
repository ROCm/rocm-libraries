// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Knob.hpp>

using namespace hipdnn_frontend;

// Test helper class to create Knob instances since constructor is private
class KnobTestHelper
{
public:
    static Knob createIntKnob(const std::string& knobIdStr,
                              const std::string& description,
                              int64_t defaultValue,
                              bool deprecated = false)
    {
        return Knob(knobIdStr, description, defaultValue, deprecated);
    }

    static Knob createFloatKnob(const std::string& knobIdStr,
                                const std::string& description,
                                double defaultValue,
                                bool deprecated = false)
    {
        return Knob(knobIdStr, description, defaultValue, deprecated);
    }

    static Knob createStringKnob(const std::string& knobIdStr,
                                 const std::string& description,
                                 const std::string& defaultValue,
                                 bool deprecated = false)
    {
        return Knob(knobIdStr, description, defaultValue, deprecated);
    }

    // Helper to set constraint on a knob (requires friend access or reflection)
    template <typename ConstraintType>
    static void setConstraint(Knob& knob, std::unique_ptr<ConstraintType> constraint)
    {
        // This would require friend access to Knob's _constraint member
        // For now, we'll test knobs without constraints in most cases
        // and rely on integration tests for constraint validation
        (void)knob;
        (void)constraint;
    }
};

// ============================================================================
// Basic Knob Creation and Accessors Tests
// ============================================================================

TEST(TestKnob, CreateIntKnob)
{
    auto knob = KnobTestHelper::createIntKnob("test_int_knob", "Test integer knob", 42);

    EXPECT_EQ(knob.getKnobIdStr(), "test_int_knob");
    EXPECT_EQ(knob.getDescription(), "Test integer knob");
    EXPECT_EQ(knob.getValueType(), KnobValueType::INT64);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = knob.getDefaultValue<int64_t>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), 42);
}

TEST(TestKnob, CreateFloatKnob)
{
    auto knob = KnobTestHelper::createFloatKnob("test_float_knob", "Test float knob", 3.14);

    EXPECT_EQ(knob.getKnobIdStr(), "test_float_knob");
    EXPECT_EQ(knob.getDescription(), "Test float knob");
    EXPECT_EQ(knob.getValueType(), KnobValueType::FLOAT64);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = knob.getDefaultValue<double>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_DOUBLE_EQ(defaultValue.value(), 3.14);
}

TEST(TestKnob, CreateStringKnob)
{
    auto knob = KnobTestHelper::createStringKnob("test_string_knob", "Test string knob", "default");

    EXPECT_EQ(knob.getKnobIdStr(), "test_string_knob");
    EXPECT_EQ(knob.getDescription(), "Test string knob");
    EXPECT_EQ(knob.getValueType(), KnobValueType::STRING);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = knob.getDefaultValue<std::string>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), "default");
}

TEST(TestKnob, CreateDeprecatedKnob)
{
    auto knob = KnobTestHelper::createIntKnob("deprecated_knob", "Deprecated knob", 0, true);

    EXPECT_TRUE(knob.isDeprecated());
}

TEST(TestKnob, KnobIdGeneration)
{
    auto knob1 = KnobTestHelper::createIntKnob("knob_a", "First knob", 1);
    auto knob2 = KnobTestHelper::createIntKnob("knob_b", "Second knob", 2);
    auto knob3 = KnobTestHelper::createIntKnob("knob_a", "Duplicate ID", 3);

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
    auto knob = KnobTestHelper::createIntKnob("int_knob", "Integer knob", 10);

    // Initially, choice should not be set
    auto initialChoice = knob.getChoice<int64_t>();
    EXPECT_FALSE(initialChoice.has_value());

    // Set a choice
    knob.setChoice<int64_t>(42);

    // Get the choice back
    auto choice = knob.getChoice<int64_t>();
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice.value(), 42);

    // Default value should remain unchanged
    auto defaultValue = knob.getDefaultValue<int64_t>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), 10);
}

TEST(TestKnob, SetAndGetFloatChoice)
{
    auto knob = KnobTestHelper::createFloatKnob("float_knob", "Float knob", 1.0);

    knob.setChoice<double>(2.718);

    auto choice = knob.getChoice<double>();
    ASSERT_TRUE(choice.has_value());
    EXPECT_DOUBLE_EQ(choice.value(), 2.718);
}

TEST(TestKnob, SetAndGetStringChoice)
{
    auto knob = KnobTestHelper::createStringKnob("string_knob", "String knob", "default");

    knob.setChoice<std::string>("custom_value");

    auto choice = knob.getChoice<std::string>();
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice.value(), "custom_value");
}

TEST(TestKnob, GetChoiceWrongType)
{
    auto knob = KnobTestHelper::createIntKnob("int_knob", "Integer knob", 10);
    knob.setChoice<int64_t>(42);

    // Try to get as wrong type
    auto wrongChoice = knob.getChoice<double>();
    EXPECT_FALSE(wrongChoice.has_value());

    auto stringChoice = knob.getChoice<std::string>();
    EXPECT_FALSE(stringChoice.has_value());
}

TEST(TestKnob, UpdateChoice)
{
    auto knob = KnobTestHelper::createIntKnob("int_knob", "Integer knob", 10);

    knob.setChoice<int64_t>(42);
    auto choice1 = knob.getChoice<int64_t>();
    ASSERT_TRUE(choice1.has_value());
    EXPECT_EQ(choice1.value(), 42);

    // Update the choice
    knob.setChoice<int64_t>(100);
    auto choice2 = knob.getChoice<int64_t>();
    ASSERT_TRUE(choice2.has_value());
    EXPECT_EQ(choice2.value(), 100);
}

// ============================================================================
// Constraint Tests
// ============================================================================

TEST(TestIntConstraint, ConstraintToString)
{
    IntConstraint constraint(0, 100, 1);
    std::string str = constraint.toString();

    EXPECT_NE(str.find("IntConstraint"), std::string::npos);
    EXPECT_NE(str.find("min=0"), std::string::npos);
    EXPECT_NE(str.find("max=100"), std::string::npos);
    EXPECT_NE(str.find("step=1"), std::string::npos);
}

TEST(TestIntConstraint, ConstraintWithValidValues)
{
    std::unordered_set<int64_t> validValues = {1, 2, 4, 8, 16};
    IntConstraint constraint(0, 100, 1, validValues);

    std::string str = constraint.toString();
    EXPECT_NE(str.find("validValues"), std::string::npos);
    // Values should be sorted in output
    EXPECT_NE(str.find("[1, 2, 4, 8, 16]"), std::string::npos);
}

TEST(TestFloatConstraint, ConstraintToString)
{
    FloatConstraint constraint(0.0, 1.0);
    std::string str = constraint.toString();

    EXPECT_NE(str.find("FloatConstraint"), std::string::npos);
    EXPECT_NE(str.find("min=0"), std::string::npos);
    EXPECT_NE(str.find("max=1"), std::string::npos);
}

TEST(TestStringConstraint, ConstraintToString)
{
    StringConstraint constraint(100);
    std::string str = constraint.toString();

    EXPECT_NE(str.find("StringConstraint"), std::string::npos);
    EXPECT_NE(str.find("maxLength=100"), std::string::npos);
}

TEST(TestStringConstraint, ConstraintWithValidValues)
{
    std::unordered_set<std::string> validValues = {"option1", "option2", "option3"};
    StringConstraint constraint(100, validValues);

    std::string str = constraint.toString();
    EXPECT_NE(str.find("validValues"), std::string::npos);
    // Check that values are quoted
    EXPECT_NE(str.find("\"option1\""), std::string::npos);
    EXPECT_NE(str.find("\"option2\""), std::string::npos);
    EXPECT_NE(str.find("\"option3\""), std::string::npos);
}

// ============================================================================
// toString Tests
// ============================================================================

TEST(TestKnob, ToStringIntKnob)
{
    auto knob = KnobTestHelper::createIntKnob("test_knob", "Test description", 42);
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
    auto knob = KnobTestHelper::createFloatKnob("float_knob", "Float test", 3.14);
    knob.setChoice<double>(2.718);

    std::string str = knob.toString();

    EXPECT_NE(str.find("defaultValue=3.14"), std::string::npos);
    EXPECT_NE(str.find("choice=2.718"), std::string::npos);
}

TEST(TestKnob, ToStringStringKnob)
{
    auto knob = KnobTestHelper::createStringKnob("string_knob", "String test", "default");
    knob.setChoice<std::string>("custom");

    std::string str = knob.toString();

    EXPECT_NE(str.find("defaultValue=\"default\""), std::string::npos);
    EXPECT_NE(str.find("choice=\"custom\""), std::string::npos);
}

TEST(TestKnob, ToStringDeprecatedKnob)
{
    auto knob = KnobTestHelper::createIntKnob("deprecated", "Deprecated knob", 0, true);

    std::string str = knob.toString();

    EXPECT_NE(str.find("deprecated=true"), std::string::npos);
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST(TestKnob, GetDefaultValueWrongType)
{
    auto knob = KnobTestHelper::createIntKnob("int_knob", "Integer knob", 42);

    // Try to get default value as wrong type
    auto wrongDefault = knob.getDefaultValue<double>();
    EXPECT_FALSE(wrongDefault.has_value());

    auto stringDefault = knob.getDefaultValue<std::string>();
    EXPECT_FALSE(stringDefault.has_value());

    // Correct type should work
    auto correctDefault = knob.getDefaultValue<int64_t>();
    ASSERT_TRUE(correctDefault.has_value());
    EXPECT_EQ(correctDefault.value(), 42);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TestKnob, EmptyStringKnob)
{
    auto knob = KnobTestHelper::createStringKnob("empty_string", "Empty string knob", "");

    auto defaultValue = knob.getDefaultValue<std::string>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), "");

    knob.setChoice<std::string>("");
    auto choice = knob.getChoice<std::string>();
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice.value(), "");
}

TEST(TestKnob, NegativeIntValues)
{
    auto knob = KnobTestHelper::createIntKnob("negative_knob", "Negative values", -100);

    auto defaultValue = knob.getDefaultValue<int64_t>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), -100);

    knob.setChoice<int64_t>(-50);
    auto choice = knob.getChoice<int64_t>();
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice.value(), -50);
}

TEST(TestKnob, ZeroValues)
{
    auto intKnob = KnobTestHelper::createIntKnob("zero_int", "Zero int", 0);
    auto floatKnob = KnobTestHelper::createFloatKnob("zero_float", "Zero float", 0.0);

    auto intDefault = intKnob.getDefaultValue<int64_t>();
    ASSERT_TRUE(intDefault.has_value());
    EXPECT_EQ(intDefault.value(), 0);

    auto floatDefault = floatKnob.getDefaultValue<double>();
    ASSERT_TRUE(floatDefault.has_value());
    EXPECT_DOUBLE_EQ(floatDefault.value(), 0.0);
}

TEST(TestKnob, LargeIntValues)
{
    int64_t largeValue = 9223372036854775807LL; // INT64_MAX
    auto knob = KnobTestHelper::createIntKnob("large_knob", "Large value", largeValue);

    auto defaultValue = knob.getDefaultValue<int64_t>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), largeValue);
}

TEST(TestKnob, SpecialFloatValues)
{
    auto knob = KnobTestHelper::createFloatKnob("special_float", "Special float", 1.23456789);

    knob.setChoice<double>(0.0);
    auto zero = knob.getChoice<double>();
    ASSERT_TRUE(zero.has_value());
    EXPECT_DOUBLE_EQ(zero.value(), 0.0);

    knob.setChoice<double>(-1.5);
    auto negative = knob.getChoice<double>();
    ASSERT_TRUE(negative.has_value());
    EXPECT_DOUBLE_EQ(negative.value(), -1.5);
}

TEST(TestKnob, LongStrings)
{
    std::string longString(1000, 'a');
    auto knob = KnobTestHelper::createStringKnob("long_string", "Long string knob", longString);

    auto defaultValue = knob.getDefaultValue<std::string>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), longString);

    std::string anotherLongString(500, 'b');
    knob.setChoice<std::string>(anotherLongString);
    auto choice = knob.getChoice<std::string>();
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice.value(), anotherLongString);
}

TEST(TestKnob, SpecialCharactersInStrings)
{
    std::string specialChars = "Test\nwith\ttabs\rand\"quotes\"";
    auto knob
        = KnobTestHelper::createStringKnob("special_chars", "Special characters", specialChars);

    auto defaultValue = knob.getDefaultValue<std::string>();
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(defaultValue.value(), specialChars);
}
