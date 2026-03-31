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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    EXPECT_EQ(knob.description(), "Test float knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::FLOAT64);

    auto defaultValue = std::get_if<double>(&knob.defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_DOUBLE_EQ(*defaultValue, 3.14);
}

TEST(TestKnob, CreateStringKnob)
{
    auto buffer = createStringKnobFlatbuffer("test_string_knob", "Test string knob", "default");
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_TRUE(knob.isDeprecated());
}

TEST(TestKnob, TryCreateIntKnob)
{
    auto [error, knob]
        = hipdnn_frontend::Knob::tryCreate("test_int_knob",
                                           "Test integer knob",
                                           int64_t{42},
                                           false,
                                           std::make_shared<IntConstraint>(0, 100, 2));

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(knob.knobId(), "test_int_knob");
    EXPECT_EQ(knob.description(), "Test integer knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::INT64);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = std::get_if<int64_t>(&knob.defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, 42);

    auto* constraint = dynamic_cast<const IntConstraint*>(knob.constraint());
    ASSERT_NE(constraint, nullptr);
    EXPECT_EQ(constraint->getMinValue(), 0);
    EXPECT_EQ(constraint->getMaxValue(), 100);
    EXPECT_EQ(constraint->getStep(), 2);
}

TEST(TestKnob, TryCreateFloatKnob)
{
    auto [error, knob]
        = hipdnn_frontend::Knob::tryCreate("test_float_knob",
                                           "Test float knob",
                                           0.5,
                                           true,
                                           std::make_shared<FloatConstraint>(0.0, 1.0));

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(knob.knobId(), "test_float_knob");
    EXPECT_EQ(knob.description(), "Test float knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::FLOAT64);
    EXPECT_TRUE(knob.isDeprecated());

    auto defaultValue = std::get_if<double>(&knob.defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_DOUBLE_EQ(*defaultValue, 0.5);

    auto* constraint = dynamic_cast<const FloatConstraint*>(knob.constraint());
    ASSERT_NE(constraint, nullptr);
    EXPECT_DOUBLE_EQ(constraint->getMinValue(), 0.0);
    EXPECT_DOUBLE_EQ(constraint->getMaxValue(), 1.0);
}

TEST(TestKnob, TryCreateStringKnob)
{
    auto [error, knob] = hipdnn_frontend::Knob::tryCreate(
        "test_string_knob",
        "Test string knob",
        std::string("balanced"),
        false,
        std::make_shared<StringConstraint>(
            16, std::unordered_set<std::string>{"fast", "balanced", "accurate"}));

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(knob.knobId(), "test_string_knob");
    EXPECT_EQ(knob.description(), "Test string knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::STRING);
    EXPECT_FALSE(knob.isDeprecated());

    auto defaultValue = std::get_if<std::string>(&knob.defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, "balanced");

    auto* constraint = dynamic_cast<const StringConstraint*>(knob.constraint());
    ASSERT_NE(constraint, nullptr);
    EXPECT_EQ(constraint->getMaxLength(), 16);
    EXPECT_EQ(constraint->getValidValues(),
              (std::unordered_set<std::string>{"fast", "balanced", "accurate"}));
}

TEST(TestKnob, TryCreateNullConstraintUsesEmptyConstraint)
{
    auto [error, knob] = hipdnn_frontend::Knob::tryCreate(
        "test_knob", "No constraint", int64_t{7}, false, nullptr);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    auto* emptyConstraint
        = dynamic_cast<const hipdnn_frontend::EmptyConstraint*>(knob.constraint());
    ASSERT_NE(emptyConstraint, nullptr);
}

TEST(TestKnob, TryCreateRejectsIntDefaultOutsideRange)
{
    auto [error, knob]
        = hipdnn_frontend::Knob::tryCreate("test_knob",
                                           "Out of range int",
                                           int64_t{101},
                                           false,
                                           std::make_shared<IntConstraint>(0, 100, 1));

    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("violates its constraint"), std::string::npos);
    EXPECT_NE(error.err_msg.find("out of range"), std::string::npos);
    (void)knob;
}

TEST(TestKnob, TryCreateRejectsIntDefaultStepViolation)
{
    auto [error, knob]
        = hipdnn_frontend::Knob::tryCreate("test_knob",
                                           "Bad step int",
                                           int64_t{15},
                                           false,
                                           std::make_shared<IntConstraint>(0, 100, 10));

    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("violates its constraint"), std::string::npos);
    EXPECT_NE(error.err_msg.find("step constraint"), std::string::npos);
    (void)knob;
}

TEST(TestKnob, TryCreateRejectsIntDefaultNotInValidValues)
{
    auto [error, knob] = hipdnn_frontend::Knob::tryCreate(
        "test_knob",
        "Bad valid-values int",
        int64_t{3},
        false,
        std::make_shared<IntConstraint>(0, 100, 1, std::unordered_set<int64_t>{1, 2, 4, 8}));

    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("violates its constraint"), std::string::npos);
    EXPECT_NE(error.err_msg.find("not in the list of valid values"), std::string::npos);
    (void)knob;
}

TEST(TestKnob, TryCreateRejectsFloatDefaultOutsideRange)
{
    auto [error, knob] = hipdnn_frontend::Knob::tryCreate(
        "test_knob", "Out of range float", 1.5, false, std::make_shared<FloatConstraint>(0.0, 1.0));

    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("violates its constraint"), std::string::npos);
    EXPECT_NE(error.err_msg.find("out of range"), std::string::npos);
    (void)knob;
}

TEST(TestKnob, TryCreateRejectsStringDefaultTooLong)
{
    auto [error, knob] = hipdnn_frontend::Knob::tryCreate(
        "test_knob",
        "Too long string default",
        std::string("toolong"),
        false,
        std::make_shared<StringConstraint>(4, std::unordered_set<std::string>{}));

    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("violates its constraint"), std::string::npos);
    EXPECT_NE(error.err_msg.find("exceeds maximum length"), std::string::npos);
    (void)knob;
}

TEST(TestKnob, TryCreateRejectsStringDefaultNotInValidValues)
{
    auto [error, knob] = hipdnn_frontend::Knob::tryCreate(
        "test_knob",
        "Bad string default",
        std::string("turbo"),
        false,
        std::make_shared<StringConstraint>(
            16, std::unordered_set<std::string>{"fast", "balanced", "accurate"}));

    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("violates its constraint"), std::string::npos);
    EXPECT_NE(error.err_msg.find("not in the list of valid values"), std::string::npos);
    (void)knob;
}

// ============================================================================
// KnobSetting Tests
// ============================================================================

TEST(TestKnobSetting, CreateIntKnobSetting)
{
    const KnobSetting setting("test.knob", 42);

    EXPECT_EQ(setting.knobId(), "test.knob");
    auto value = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, 42);
}

TEST(TestKnobSetting, CreateFloatKnobSetting)
{
    const KnobSetting setting("test.knob", 3.14);

    EXPECT_EQ(setting.knobId(), "test.knob");
    auto value = std::get_if<double>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_DOUBLE_EQ(*value, 3.14);
}

TEST(TestKnobSetting, CreateStringKnobSetting)
{
    const KnobSetting setting("test.knob", std::string("custom_value"));

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
    const KnobSetting setting("test.knob", 42);

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
    const hipdnn_frontend::IntConstraint constraint(0, 100, 1);
    const std::string str = constraint.toString();

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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    auto constraint = knob.constraint();
    ASSERT_NE(constraint, nullptr);

    const std::string str = constraint->toString();
    EXPECT_NE(str.find("validValues"), std::string::npos);
    // Values should be sorted in output
    EXPECT_NE(str.find("[1, 2, 4, 8, 16]"), std::string::npos);
}

TEST(TestKnobFloatConstraint, ConstraintToString)
{
    const hipdnn_frontend::FloatConstraint constraint(0.0, 1.0);
    const std::string str = constraint.toString();

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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    auto constraint = knob.constraint();
    ASSERT_NE(constraint, nullptr);

    const std::string str = constraint->toString();
    EXPECT_NE(str.find("FloatConstraint"), std::string::npos);
}

TEST(TestKnobStringConstraint, ConstraintToString)
{
    const hipdnn_frontend::StringConstraint constraint(100);
    const std::string str = constraint.toString();

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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    auto constraint = knob.constraint();
    ASSERT_NE(constraint, nullptr);

    const std::string str = constraint->toString();

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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Valid value
    const KnobSetting setting1(knob.knobId(), 50);
    auto err = knob.validate(setting1);
    EXPECT_EQ(err.code, ErrorCode::OK);

    // Value at min boundary
    const KnobSetting setting2(knob.knobId(), 0);
    err = knob.validate(setting2);
    EXPECT_EQ(err.code, ErrorCode::OK);

    // Value at max boundary
    const KnobSetting setting3(knob.knobId(), 100);
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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Value below min
    const KnobSetting setting1(knob.knobId(), -1);
    auto err = knob.validate(setting1);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("out of range"), std::string::npos);

    // Value above max
    const KnobSetting setting2(knob.knobId(), 101);
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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Valid step values
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 0)).code, ErrorCode::OK);
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 10)).code, ErrorCode::OK);
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), 100)).code, ErrorCode::OK);

    // Invalid step value
    const KnobSetting invalidSetting(knob.knobId(), 15);
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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Valid values
    for(auto validVal : {1, 2, 4, 8, 16})
    {
        const KnobSetting setting(knob.knobId(), validVal);
        EXPECT_EQ(knob.validate(setting).code, ErrorCode::OK);
    }

    // Invalid value
    const KnobSetting invalidSetting(knob.knobId(), 3);
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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Value below min
    const KnobSetting setting1(knob.knobId(), -0.1);
    auto err = knob.validate(setting1);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);

    // Value above max
    const KnobSetting setting2(knob.knobId(), 1.1);
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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Valid values
    for(const auto& validVal : {"option1", "option2", "option3"})
    {
        const KnobSetting setting(knob.knobId(), std::string(validVal));
        EXPECT_EQ(knob.validate(setting).code, ErrorCode::OK);
    }

    // Invalid value
    const KnobSetting invalidSetting(knob.knobId(), std::string("invalid"));
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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Valid length
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), std::string("short"))).code, ErrorCode::OK);

    // Exactly at max length
    EXPECT_EQ(knob.validate(KnobSetting(knob.knobId(), std::string("1234567890"))).code,
              ErrorCode::OK);

    // Exceeds max length
    const KnobSetting invalidSetting(knob.knobId(), std::string("12345678901"));
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
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    const std::string str = knob.toString();

    EXPECT_NE(str.find("knobIdStr=\"test_knob\""), std::string::npos);
    EXPECT_NE(str.find("description=\"Test description\""), std::string::npos);
    EXPECT_NE(str.find("defaultValue=42"), std::string::npos);
    EXPECT_NE(str.find("deprecated=false"), std::string::npos);
}

TEST(TestKnob, ToStringFloatKnob)
{
    auto buffer = createFloatKnobFlatbuffer("float_knob", "Float test", 3.14);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    const std::string str = knob.toString();

    EXPECT_NE(str.find("defaultValue=3.14"), std::string::npos);
}

TEST(TestKnob, ToStringStringKnob)
{
    auto buffer = createStringKnobFlatbuffer("string_knob", "String test", "default");
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    const std::string str = knob.toString();

    EXPECT_NE(str.find("defaultValue=\"default\""), std::string::npos);
}

TEST(TestKnobSetting, ToStringIntSetting)
{
    const KnobSetting setting("test.knob", 42);

    const std::string str = setting.toString();

    EXPECT_NE(str.find("KnobSetting"), std::string::npos);
    EXPECT_NE(str.find("value=42"), std::string::npos);
}

TEST(TestKnobSetting, ToStringFloatSetting)
{
    const KnobSetting setting("test.knob", 3.14);

    const std::string str = setting.toString();

    EXPECT_NE(str.find("value=3.14"), std::string::npos);
}

TEST(TestKnobSetting, ToStringStringSetting)
{
    const KnobSetting setting("test.knob", std::string("custom"));

    const std::string str = setting.toString();

    EXPECT_NE(str.find("value=\"custom\""), std::string::npos);
}

TEST(TestKnob, ToStringDeprecatedKnob)
{
    auto buffer = createIntKnobFlatbuffer("deprecated", "Deprecated knob", 0, true);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    const std::string str = knob.toString();

    EXPECT_NE(str.find("deprecated=true"), std::string::npos);
}

TEST(TestKnob, ToStringWithConstraint)
{
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;

    auto buffer = createIntKnobFlatbuffer("test_knob", "Test knob", 50, false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    const std::string str = knob.toString();

    EXPECT_NE(str.find("constraint="), std::string::npos);
    EXPECT_NE(str.find("IntConstraint"), std::string::npos);
}

TEST(TestKnob, GetDefaultValueWrongType)
{
    auto buffer = createIntKnobFlatbuffer("int_knob", "Integer knob", 42);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

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
    const KnobSetting setting("test.knob", std::string(""));

    auto value = std::get_if<std::string>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, "");
}

TEST(TestKnobSetting, NegativeIntValue)
{
    const KnobSetting setting("test.knob", static_cast<int64_t>(-100));

    auto value = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, -100);
}

TEST(TestKnobSetting, ZeroValues)
{
    const KnobSetting intSetting("test.int", static_cast<int64_t>(0));
    const KnobSetting floatSetting("test.float", 0.0);

    auto intValue = std::get_if<int64_t>(&intSetting.value());
    ASSERT_NE(intValue, nullptr);
    EXPECT_EQ(*intValue, 0);

    auto floatValue = std::get_if<double>(&floatSetting.value());
    ASSERT_NE(floatValue, nullptr);
    EXPECT_DOUBLE_EQ(*floatValue, 0.0);
}

TEST(TestKnobSetting, LargeIntValue)
{
    const int64_t largeValue = 9223372036854775807LL; // INT64_MAX
    const KnobSetting setting("test.knob", largeValue);

    auto value = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, largeValue);
}

TEST(TestKnobSetting, LongStringValue)
{
    const std::string longString(1000, 'a');
    const KnobSetting setting("test.knob", longString);

    auto value = std::get_if<std::string>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, longString);
}

TEST(TestKnobSetting, SpecialCharactersInString)
{
    const std::string specialChars = "Test\nwith\ttabs\rand\"quotes\"";
    const KnobSetting setting("test.knob", specialChars);

    auto value = std::get_if<std::string>(&setting.value());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, specialChars);
}

// ============================================================================
// packKnobSetting Tests
// ============================================================================

TEST(TestKnobSetting, PackIntKnobSetting)
{
    const KnobSetting setting("test.knob", 42);

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
    const KnobSetting setting("test.knob", 3.14);

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
    const KnobSetting setting("test.knob", std::string("custom_value"));

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
    const KnobSetting setting("test.knob", std::string(""));

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
    const std::string longString(1000, 'a');
    const KnobSetting setting("test.knob", longString);

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
    const std::string specialChars = "Test\nwith\ttabs\rand\"quotes\"";
    const KnobSetting setting("test.knob", specialChars);

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
    const KnobSetting setting1("knob_a", 1);
    const KnobSetting setting2("knob_b", 2);

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

// ============================================================================
// NONE Default Value Tests (Graceful Error Handling)
// ============================================================================

TEST(TestKnob, InvalidDefaultValueReturnsError)
{
    // Create a knob with constraint min=0, max=100 but default_value=200 (violates constraint)
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;

    // Create knob with default_value=200 which is outside the constraint range
    auto buffer
        = createIntKnobFlatbuffer("bad_default_knob", "Invalid default", 200, false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    // Should return error because default_value violates constraint
    EXPECT_NE(error.code, ErrorCode::OK);

    EXPECT_NE(error.err_msg.find("violates"), std::string::npos);
    EXPECT_NE(error.err_msg.find("bad_default_knob"), std::string::npos);
}

TEST(TestKnob, InvalidDefaultValueBelowMinReturnsError)
{
    // Create a knob with constraint min=10, max=100 but default_value=5 (below min)
    fb::IntConstraintT constraintT;
    constraintT.min_value = 10;
    constraintT.max_value = 100;
    constraintT.step = 1;

    auto buffer
        = createIntKnobFlatbuffer("below_min_knob", "Default below min", 5, false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("violates"), std::string::npos);
}

TEST(TestKnob, InvalidDefaultFloatValueReturnsError)
{
    // Create a float knob with constraint min=0.0, max=1.0 but default_value=2.0
    fb::FloatConstraintT constraintT;
    constraintT.min_value = 0.0;
    constraintT.max_value = 1.0;

    auto buffer = createFloatKnobFlatbuffer(
        "bad_float_knob", "Invalid float default", 2.0, false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("violates"), std::string::npos);
}

TEST(TestKnob, InvalidDefaultStringValueReturnsError)
{
    // Create a string knob with valid_values={"a", "b", "c"} but default_value="invalid"
    fb::StringConstraintT constraintT;
    constraintT.max_length = 100;
    constraintT.valid_values = {"a", "b", "c"};

    auto buffer = createStringKnobFlatbuffer(
        "bad_string_knob", "Invalid string default", "invalid", false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("violates"), std::string::npos);
}

TEST(TestKnob, ValidDefaultValueSucceeds)
{
    // Create a knob with valid default_value within constraint
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;

    auto buffer = createIntKnobFlatbuffer("valid_knob", "Valid default", 50, false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    // Should succeed
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestKnob, NullFlatbufferDataReturnsError)
{
    const hipdnnBackendFlatbufferData_t bufferData{nullptr, 0};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("nullptr"), std::string::npos);
}

TEST(TestKnob, ZeroSizeFlatbufferDataReturnsError)
{
    uint8_t data = 0;
    const hipdnnBackendFlatbufferData_t bufferData{&data, 0};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("zero size"), std::string::npos);
}

TEST(TestKnob, InvalidFlatbufferDataReturnsError)
{
    // Create garbage data that won't pass flatbuffer verification
    std::vector<uint8_t> garbageData = {0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD, 0xFC};
    const hipdnnBackendFlatbufferData_t bufferData{garbageData.data(), garbageData.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("failed verification"), std::string::npos);
}

// Note: KnobValue::NONE and unknown default_value types are defensive code paths
// that cannot be hit with a valid flatbuffer (the builder enforces required fields).
// The InvalidFlatbufferDataReturnsError test covers corrupted data scenarios.

TEST(TestKnob, InvalidDefaultValueNotInValidValuesReturnsError)
{
    // Create an int knob with valid_values={1,2,3} but default_value=5
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 1;
    constraintT.valid_values = {1, 2, 3};

    auto buffer = createIntKnobFlatbuffer(
        "bad_valid_values_knob", "Default not in valid values", 5, false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("violates"), std::string::npos);
    EXPECT_NE(error.err_msg.find("bad_valid_values_knob"), std::string::npos);
}

TEST(TestKnob, InvalidDefaultStringTooLongReturnsError)
{
    // Create a string knob with max_length=5 but default_value="toolong"
    fb::StringConstraintT constraintT;
    constraintT.max_length = 5;

    auto buffer = createStringKnobFlatbuffer(
        "too_long_knob", "Default too long", "toolong", false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("violates"), std::string::npos);
}

TEST(TestKnob, InvalidDefaultIntStepViolationReturnsError)
{
    // Create an int knob with step=10 but default_value=5 (not a multiple of step)
    fb::IntConstraintT constraintT;
    constraintT.min_value = 0;
    constraintT.max_value = 100;
    constraintT.step = 10;

    auto buffer = createIntKnobFlatbuffer(
        "step_violation_knob", "Default violates step", 5, false, &constraintT);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    EXPECT_NE(error.code, ErrorCode::OK);
    EXPECT_NE(error.err_msg.find("violates"), std::string::npos);
    EXPECT_NE(error.err_msg.find("step_violation_knob"), std::string::npos);
}

// ============================================================================
// EmptyConstraint Tests
// ============================================================================

TEST(TestEmptyConstraint, ToString)
{
    const hipdnn_frontend::EmptyConstraint constraint;
    const std::string str = constraint.toString();

    EXPECT_EQ(str, "EmptyConstraint{}");
}

TEST(TestEmptyConstraint, ValidatesAnyIntValue)
{
    const hipdnn_frontend::EmptyConstraint constraint;

    // Any int value should be valid
    const KnobSetting setting1("test", static_cast<int64_t>(0));
    EXPECT_EQ(constraint.validateKnobSetting(setting1).code, ErrorCode::OK);

    const KnobSetting setting2("test", static_cast<int64_t>(-1000000));
    EXPECT_EQ(constraint.validateKnobSetting(setting2).code, ErrorCode::OK);

    const KnobSetting setting3("test", static_cast<int64_t>(999999999));
    EXPECT_EQ(constraint.validateKnobSetting(setting3).code, ErrorCode::OK);
}

TEST(TestEmptyConstraint, ValidatesAnyFloatValue)
{
    const hipdnn_frontend::EmptyConstraint constraint;

    const KnobSetting setting1("test", 0.0);
    EXPECT_EQ(constraint.validateKnobSetting(setting1).code, ErrorCode::OK);

    const KnobSetting setting2("test", -1e10);
    EXPECT_EQ(constraint.validateKnobSetting(setting2).code, ErrorCode::OK);

    const KnobSetting setting3("test", 1e10);
    EXPECT_EQ(constraint.validateKnobSetting(setting3).code, ErrorCode::OK);
}

TEST(TestEmptyConstraint, ValidatesAnyStringValue)
{
    const hipdnn_frontend::EmptyConstraint constraint;

    const KnobSetting setting1("test", std::string(""));
    EXPECT_EQ(constraint.validateKnobSetting(setting1).code, ErrorCode::OK);

    const KnobSetting setting2("test", std::string("any string value"));
    EXPECT_EQ(constraint.validateKnobSetting(setting2).code, ErrorCode::OK);

    const std::string longString(10000, 'x');
    const KnobSetting setting3("test", longString);
    EXPECT_EQ(constraint.validateKnobSetting(setting3).code, ErrorCode::OK);
}

TEST(TestKnob, NoneConstraintCreatesEmptyConstraint)
{
    // Create a knob WITHOUT a constraint (KnobConstraint::NONE)
    // The helper functions pass nullptr for constraint, which results in NONE
    auto buffer = createIntKnobFlatbuffer("unconstrained_knob", "Unconstrained knob", 42);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Constraint should NOT be nullptr - should be EmptyConstraint
    ASSERT_NE(knob.constraint(), nullptr);

    // Should be EmptyConstraint
    auto* emptyConstraint
        = dynamic_cast<const hipdnn_frontend::EmptyConstraint*>(knob.constraint());
    EXPECT_NE(emptyConstraint, nullptr)
        << "Expected EmptyConstraint but got: " << knob.constraint()->toString();
}

TEST(TestKnob, EmptyConstraintInToString)
{
    // Create a knob without constraint
    auto buffer = createIntKnobFlatbuffer("test_knob", "Test", 42);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    const std::string str = knob.toString();
    EXPECT_NE(str.find("EmptyConstraint"), std::string::npos);
}

TEST(TestKnob, UnconstrainedKnobValidatesAnyValue)
{
    // Create an unconstrained int knob
    auto buffer = createIntKnobFlatbuffer("unconstrained_knob", "No constraints", 0);
    const hipdnnBackendFlatbufferData_t bufferData{buffer.data(), buffer.size()};
    auto [error, knob] = hipdnn_frontend::Knob::tryFromFlatbuffer(bufferData);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    // Any int value should be valid for an unconstrained knob
    EXPECT_EQ(knob.validate(KnobSetting("test", static_cast<int64_t>(-999999))).code,
              ErrorCode::OK);
    EXPECT_EQ(knob.validate(KnobSetting("test", static_cast<int64_t>(0))).code, ErrorCode::OK);
    EXPECT_EQ(knob.validate(KnobSetting("test", static_cast<int64_t>(999999))).code, ErrorCode::OK);
}

TEST(TestKnob, TryCreateRejectsEmptyKnobId)
{
    auto [error, knob] = Knob::tryCreate("", "description", static_cast<int64_t>(0), false);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("empty"), std::string::npos);
}
