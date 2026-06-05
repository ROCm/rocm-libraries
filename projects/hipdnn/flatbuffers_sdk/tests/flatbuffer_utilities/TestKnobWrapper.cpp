// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <gtest/gtest.h>
#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/KnobWrapper.hpp>

using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

class TestKnobWrapper : public ::testing::Test
{
protected:
    // Note: default_value is now required in the schema, so all knobs must have a default value
    static flatbuffers::DetachedBuffer createKnob(const std::string& knobIdStr,
                                                  const std::string& description,
                                                  bool deprecated = false,
                                                  int64_t defaultValue = 100,
                                                  bool withConstraint = false)
    {
        flatbuffers::FlatBufferBuilder builder;

        auto knobIdStrOffset = builder.CreateString(knobIdStr);
        auto descOffset = builder.CreateString(description);

        // default_value is required - always create one
        auto defaultVal
            = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(builder, defaultValue);

        flatbuffers::Offset<void> constraintOffset;
        hipdnn_flatbuffers_sdk::data_objects::KnobConstraint constraintType
            = hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::NONE;

        if(withConstraint)
        {
            // Create a simple IntConstraint with min/max/step
            auto constraint
                = hipdnn_flatbuffers_sdk::data_objects::CreateIntConstraint(builder, 0, 200, 10);
            constraintOffset = constraint.Union();
            constraintType = hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::IntConstraint;
        }

        hipdnn_flatbuffers_sdk::data_objects::KnobBuilder knobBuilder(builder);
        knobBuilder.add_knob_id(knobIdStrOffset);
        knobBuilder.add_description(descOffset);
        knobBuilder.add_deprecated(deprecated);

        // default_value is required - always set it
        knobBuilder.add_default_value_type(
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
        knobBuilder.add_default_value(defaultVal.Union());

        if(withConstraint)
        {
            knobBuilder.add_constraint_type(constraintType);
            knobBuilder.add_constraint(constraintOffset);
        }

        auto knob = knobBuilder.Finish();
        builder.Finish(knob);

        return builder.Release();
    }

    static flatbuffers::DetachedBuffer createKnobWithFloatValue(const std::string& knobIdStr,
                                                                double defaultValue)
    {
        flatbuffers::FlatBufferBuilder builder;

        auto knobIdStrOffset = builder.CreateString(knobIdStr);
        auto descOffset = builder.CreateString("Float knob");

        auto floatVal
            = hipdnn_flatbuffers_sdk::data_objects::CreateFloatValue(builder, defaultValue);

        hipdnn_flatbuffers_sdk::data_objects::KnobBuilder knobBuilder(builder);
        knobBuilder.add_knob_id(knobIdStrOffset);
        knobBuilder.add_description(descOffset);
        knobBuilder.add_default_value_type(
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
        knobBuilder.add_default_value(floatVal.Union());

        auto knob = knobBuilder.Finish();
        builder.Finish(knob);

        return builder.Release();
    }

    static flatbuffers::DetachedBuffer createKnobWithFloatConstraint(const std::string& knobIdStr,
                                                                     double minValue,
                                                                     double maxValue)
    {
        flatbuffers::FlatBufferBuilder builder;

        auto knobIdStrOffset = builder.CreateString(knobIdStr);
        auto descOffset = builder.CreateString("Knob with float constraint");

        auto floatVal = hipdnn_flatbuffers_sdk::data_objects::CreateFloatValue(builder, 1.0);
        auto floatConstraint = hipdnn_flatbuffers_sdk::data_objects::CreateFloatConstraint(
            builder, minValue, maxValue);

        hipdnn_flatbuffers_sdk::data_objects::KnobBuilder knobBuilder(builder);
        knobBuilder.add_knob_id(knobIdStrOffset);
        knobBuilder.add_description(descOffset);
        knobBuilder.add_default_value_type(
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
        knobBuilder.add_default_value(floatVal.Union());
        knobBuilder.add_constraint_type(
            hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::FloatConstraint);
        knobBuilder.add_constraint(floatConstraint.Union());

        auto knob = knobBuilder.Finish();
        builder.Finish(knob);

        return builder.Release();
    }

    static flatbuffers::DetachedBuffer
        createKnobWithStringConstraint(const std::string& knobIdStr,
                                       int32_t maxLength,
                                       const std::vector<std::string>& validValues)
    {
        flatbuffers::FlatBufferBuilder builder;

        auto knobIdStrOffset = builder.CreateString(knobIdStr);
        auto descOffset = builder.CreateString("Knob with string constraint");

        auto strVal = builder.CreateString("default_val");
        auto stringValue = hipdnn_flatbuffers_sdk::data_objects::CreateStringValue(builder, strVal);

        std::vector<flatbuffers::Offset<flatbuffers::String>> validValOffsets;
        validValOffsets.reserve(validValues.size());
        for(const auto& val : validValues)
        {
            validValOffsets.push_back(builder.CreateString(val));
        }
        auto validValuesVector = builder.CreateVector(validValOffsets);
        auto stringConstraint = hipdnn_flatbuffers_sdk::data_objects::CreateStringConstraint(
            builder, maxLength, validValuesVector);

        hipdnn_flatbuffers_sdk::data_objects::KnobBuilder knobBuilder(builder);
        knobBuilder.add_knob_id(knobIdStrOffset);
        knobBuilder.add_description(descOffset);
        knobBuilder.add_default_value_type(
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
        knobBuilder.add_default_value(stringValue.Union());
        knobBuilder.add_constraint_type(
            hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::StringConstraint);
        knobBuilder.add_constraint(stringConstraint.Union());

        auto knob = knobBuilder.Finish();
        builder.Finish(knob);

        return builder.Release();
    }
};

TEST_F(TestKnobWrapper, ConstructFromFlatbufferPointer)
{
    auto buffer = createKnob("KNOB_42", "Test knob");
    auto knob = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(buffer.data());

    const KnobWrapper wrapper(knob);
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobId(), "KNOB_42");
    EXPECT_EQ(wrapper.description(), "Test knob");
}

TEST_F(TestKnobWrapper, ConstructFromBuffer)
{
    auto buffer = createKnob("KNOB_42", "Test knob");

    const KnobWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobId(), "KNOB_42");
    EXPECT_EQ(wrapper.description(), "Test knob");
}

TEST_F(TestKnobWrapper, ConstructFromNullPointer)
{
    const KnobWrapper wrapper(static_cast<hipdnn_flatbuffers_sdk::data_objects::Knob*>(nullptr));
    EXPECT_FALSE(wrapper.isValid());
}

TEST_F(TestKnobWrapper, ConstructFromNullBuffer)
{
    const KnobWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
}

TEST_F(TestKnobWrapper, ConstructFromInvalidBuffer)
{
    std::array<uint8_t, 10> invalidData
        = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    const KnobWrapper wrapper(invalidData.data(), invalidData.size());
    EXPECT_FALSE(wrapper.isValid());
}

TEST_F(TestKnobWrapper, GetKnobIdFromValidWrapper)
{
    auto buffer = createKnob("CUSTOM_KNOB_NAME", "Test");
    const KnobWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_EQ(wrapper.knobId(), "CUSTOM_KNOB_NAME");
}

TEST_F(TestKnobWrapper, GetDescriptionFromValidWrapper)
{
    auto buffer = createKnob("KNOB_42", "This is a detailed description");
    const KnobWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_EQ(wrapper.description(), "This is a detailed description");
}

TEST_F(TestKnobWrapper, GetDefaultValueType)
{
    auto buffer = createKnob("KNOB_42", "Test");
    const KnobWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_EQ(wrapper.defaultValueType(),
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
}

TEST_F(TestKnobWrapper, IsDeprecated)
{
    auto buffer1 = createKnob("KNOB_42", "Test", false);
    const KnobWrapper wrapper1(buffer1.data(), buffer1.size());
    EXPECT_FALSE(wrapper1.isDeprecated());

    auto buffer2 = createKnob("KNOB_42", "Test", true);
    const KnobWrapper wrapper2(buffer2.data(), buffer2.size());
    EXPECT_TRUE(wrapper2.isDeprecated());
}

TEST_F(TestKnobWrapper, HasDefaultValue)
{
    // default_value is now required, so all valid knobs have a default value
    auto buffer = createKnob("KNOB_42", "Test", false);
    const KnobWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_TRUE(wrapper.hasDefaultValue());
    EXPECT_EQ(wrapper.defaultValueType(),
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
}

TEST_F(TestKnobWrapper, DefaultValueAsIntValue)
{
    auto buffer = createKnob("KNOB_42", "Test", false, 100, false);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.hasDefaultValue());
    const auto& intValue = wrapper.defaultValueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>();
    EXPECT_EQ(intValue.value(), 100);
}

TEST_F(TestKnobWrapper, DefaultValueAsFloatValue)
{
    auto buffer = createKnobWithFloatValue("FLOAT_KNOB", 3.14159);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.hasDefaultValue());
    EXPECT_EQ(wrapper.defaultValueType(),
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
    const auto& floatValue
        = wrapper.defaultValueAs<hipdnn_flatbuffers_sdk::data_objects::FloatValue>();
    EXPECT_DOUBLE_EQ(floatValue.value(), 3.14159);
}

TEST_F(TestKnobWrapper, DefaultValueAsTypeMismatchThrows)
{
    auto buffer = createKnob("KNOB_42", "Test", false, 100, false);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    // Default value is IntValue, trying to get as FloatValue should throw
    EXPECT_THROW(wrapper.defaultValueAs<hipdnn_flatbuffers_sdk::data_objects::FloatValue>(),
                 std::invalid_argument);
    EXPECT_THROW(wrapper.defaultValueAs<hipdnn_flatbuffers_sdk::data_objects::StringValue>(),
                 std::invalid_argument);
}

TEST_F(TestKnobWrapper, HasConstraints)
{
    auto buffer1 = createKnob("KNOB_42", "Test", false, 100, true);
    const KnobWrapper wrapper1(buffer1.data(), buffer1.size());
    EXPECT_TRUE(wrapper1.hasConstraint());
    EXPECT_EQ(wrapper1.constraintType(),
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::IntConstraint);

    auto buffer2 = createKnob("KNOB_42", "Test", false, 100, false);
    const KnobWrapper wrapper2(buffer2.data(), buffer2.size());
    EXPECT_FALSE(wrapper2.hasConstraint());
}

TEST_F(TestKnobWrapper, ConstraintsAsIntConstraint)
{
    auto buffer = createKnob("KNOB_42", "Test", false, 100, true);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.hasConstraint());
    const auto& intConstraint
        = wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::IntConstraint>();
    EXPECT_EQ(intConstraint.min_value(), 0);
    EXPECT_EQ(intConstraint.max_value(), 200);
    EXPECT_EQ(intConstraint.step(), 10);
}

TEST_F(TestKnobWrapper, ConstraintsAsTypeMismatchThrows)
{
    auto buffer = createKnob("KNOB_42", "Test", false, 100, true);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    // Constraint is IntConstraint, trying to get as FloatConstraint should throw
    EXPECT_THROW(wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::FloatConstraint>(),
                 std::invalid_argument);
    EXPECT_THROW(wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::StringConstraint>(),
                 std::invalid_argument);
}

TEST_F(TestKnobWrapper, GetKnobFromValidWrapper)
{
    auto buffer = createKnob("KNOB_42", "Test");
    auto knob = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(buffer.data());

    const KnobWrapper wrapper(knob);
    const auto& retrievedKnob = wrapper.getKnob();
    EXPECT_EQ(&retrievedKnob, knob);
}

TEST_F(TestKnobWrapper, AccessMethodsOnInvalidWrapperThrow)
{
    const KnobWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());

    EXPECT_THROW(wrapper.knobId(), std::invalid_argument);
    EXPECT_THROW(wrapper.description(), std::invalid_argument);
    EXPECT_THROW(wrapper.defaultValueType(), std::invalid_argument);
    EXPECT_THROW(wrapper.isDeprecated(), std::invalid_argument);
    EXPECT_THROW(wrapper.hasDefaultValue(), std::invalid_argument);
    EXPECT_THROW(wrapper.hasConstraint(), std::invalid_argument);
    EXPECT_THROW(wrapper.constraintType(), std::invalid_argument);
    EXPECT_THROW(wrapper.getKnob(), std::invalid_argument);
}

TEST_F(TestKnobWrapper, EmptyStringsHandling)
{
    // Create a knob with empty strings
    auto buffer = createKnob("", "", false);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobId(), "");
    EXPECT_EQ(wrapper.description(), "");
}

TEST_F(TestKnobWrapper, MultipleDifferentKnobs)
{
    // Test that wrapper correctly handles different knob configurations
    // Note: default_value is now required, so all knobs have a default value
    auto buffer1 = createKnob("KNOB_1", "First", false, 50, false);
    auto buffer2 = createKnob("KNOB_2", "Second", true, 75, false);
    auto buffer3 = createKnob("KNOB_3", "Third", false, 100, true);

    const KnobWrapper wrapper1(buffer1.data(), buffer1.size());
    const KnobWrapper wrapper2(buffer2.data(), buffer2.size());
    const KnobWrapper wrapper3(buffer3.data(), buffer3.size());

    EXPECT_EQ(wrapper1.knobId(), "KNOB_1");
    EXPECT_TRUE(wrapper1.hasDefaultValue());
    EXPECT_FALSE(wrapper1.hasConstraint());
    EXPECT_FALSE(wrapper1.isDeprecated());

    EXPECT_EQ(wrapper2.knobId(), "KNOB_2");
    EXPECT_TRUE(wrapper2.hasDefaultValue());
    EXPECT_FALSE(wrapper2.hasConstraint());
    EXPECT_TRUE(wrapper2.isDeprecated());

    EXPECT_EQ(wrapper3.knobId(), "KNOB_3");
    EXPECT_TRUE(wrapper3.hasDefaultValue());
    EXPECT_TRUE(wrapper3.hasConstraint());
    EXPECT_FALSE(wrapper3.isDeprecated());
}

TEST_F(TestKnobWrapper, HasFloatConstraint)
{
    auto buffer = createKnobWithFloatConstraint("FLOAT_CONSTRAINT_KNOB", 0.0, 100.0);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_TRUE(wrapper.hasConstraint());
    EXPECT_EQ(wrapper.constraintType(),
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::FloatConstraint);
}

TEST_F(TestKnobWrapper, ConstraintAsFloatConstraint)
{
    auto buffer = createKnobWithFloatConstraint("FLOAT_CONSTRAINT_KNOB", -10.5, 50.75);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.hasConstraint());
    const auto& floatConstraint
        = wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::FloatConstraint>();
    EXPECT_DOUBLE_EQ(floatConstraint.min_value(), -10.5);
    EXPECT_DOUBLE_EQ(floatConstraint.max_value(), 50.75);
}

TEST_F(TestKnobWrapper, FloatConstraintTypeMismatchThrows)
{
    auto buffer = createKnobWithFloatConstraint("FLOAT_CONSTRAINT_KNOB", 0.0, 100.0);
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    // Constraint is FloatConstraint, trying to get as IntConstraint should throw
    EXPECT_THROW(wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::IntConstraint>(),
                 std::invalid_argument);
    EXPECT_THROW(wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::StringConstraint>(),
                 std::invalid_argument);
}

TEST_F(TestKnobWrapper, HasStringConstraint)
{
    auto buffer = createKnobWithStringConstraint(
        "STRING_CONSTRAINT_KNOB", 256, {"option1", "option2", "option3"});
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_TRUE(wrapper.hasConstraint());
    EXPECT_EQ(wrapper.constraintType(),
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::StringConstraint);
}

TEST_F(TestKnobWrapper, ConstraintAsStringConstraint)
{
    auto buffer
        = createKnobWithStringConstraint("STRING_CONSTRAINT_KNOB", 128, {"alpha", "beta", "gamma"});
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.hasConstraint());
    const auto& stringConstraint
        = wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::StringConstraint>();
    EXPECT_EQ(stringConstraint.max_length(), 128);
    ASSERT_NE(stringConstraint.valid_values(), nullptr);
    ASSERT_EQ(stringConstraint.valid_values()->size(), 3u);
    EXPECT_STREQ(stringConstraint.valid_values()->Get(0)->c_str(), "alpha");
    EXPECT_STREQ(stringConstraint.valid_values()->Get(1)->c_str(), "beta");
    EXPECT_STREQ(stringConstraint.valid_values()->Get(2)->c_str(), "gamma");
}

TEST_F(TestKnobWrapper, StringConstraintTypeMismatchThrows)
{
    auto buffer = createKnobWithStringConstraint("STRING_CONSTRAINT_KNOB", 100, {"opt1", "opt2"});
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    // Constraint is StringConstraint, trying to get as IntConstraint should throw
    EXPECT_THROW(wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::IntConstraint>(),
                 std::invalid_argument);
    EXPECT_THROW(wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::FloatConstraint>(),
                 std::invalid_argument);
}

TEST_F(TestKnobWrapper, StringConstraintEmptyValidValues)
{
    auto buffer = createKnobWithStringConstraint("STRING_CONSTRAINT_KNOB", 64, {});
    const KnobWrapper wrapper(buffer.data(), buffer.size());

    EXPECT_TRUE(wrapper.hasConstraint());
    const auto& stringConstraint
        = wrapper.constraintAs<hipdnn_flatbuffers_sdk::data_objects::StringConstraint>();
    EXPECT_EQ(stringConstraint.max_length(), 64);
    ASSERT_NE(stringConstraint.valid_values(), nullptr);
    EXPECT_EQ(stringConstraint.valid_values()->size(), 0u);
}
