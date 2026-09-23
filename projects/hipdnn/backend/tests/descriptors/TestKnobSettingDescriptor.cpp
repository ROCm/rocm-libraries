// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/KnobSettingDescriptor.hpp"

#include <gtest/gtest.h>

#include <array>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;

class TestKnobSettingDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<KnobSettingDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<KnobSettingDescriptor>();
    }

    void setKnobId(const std::string& knobId) const
    {
        ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                                      HIPDNN_TYPE_CHAR,
                                                      static_cast<int64_t>(knobId.size()),
                                                      knobId.c_str()));
    }

    void setInt64Value(int64_t value) const
    {
        ASSERT_NO_THROW(getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, &value));
    }

    void setDoubleValue(double value) const
    {
        ASSERT_NO_THROW(getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_DOUBLE, 1, &value));
    }

    void setStringValue(const std::string& value) const
    {
        ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                                      HIPDNN_TYPE_CHAR,
                                                      static_cast<int64_t>(value.size()),
                                                      value.c_str()));
    }

    void makeFinalized(const std::string& knobId = "test_knob", int64_t value = 42) const
    {
        setKnobId(knobId);
        setInt64Value(value);
        ASSERT_NO_THROW(getDescriptor()->finalize());
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper;

    void SetUp() override
    {
        _wrapper = createDescriptor<KnobSettingDescriptor>();
    }
};

TEST_F(TestKnobSettingDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
}

TEST_F(TestKnobSettingDescriptor, FinalizeWithKnobIdAndInt64Value)
{
    setKnobId("test_knob_100");
    setInt64Value(42);
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestKnobSettingDescriptor, FinalizeWithoutKnobIdFails)
{
    setInt64Value(42);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, FinalizeWithoutValueFails)
{
    setKnobId("test_knob");
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, FinalizeAlreadyFinalizedFails)
{
    makeFinalized();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, SetAttributeAfterFinalizeFails)
{
    makeFinalized();
    const std::string knobId = "another_knob";
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                                             HIPDNN_TYPE_CHAR,
                                                             static_cast<int64_t>(knobId.size()),
                                                             knobId.c_str()),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestKnobSettingDescriptor, SetAttributeUnsupportedAttribute)
{
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestKnobSettingDescriptor, SetKnobIdAsChar)
{
    const std::string knobId = "test_knob_id";
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(knobId.size()),
                                                  knobId.c_str()));
}

TEST_F(TestKnobSettingDescriptor, SetKnobIdWrongTypeFails)
{
    int64_t val = 42;
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(
                                   HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_INT64, 1, &val),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, SetKnobIdNullFails)
{
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(
                                   HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, 5, nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestKnobSettingDescriptor, GetKnobIdAfterFinalize)
{
    const std::string expectedId = "test_knob_id";
    setKnobId(expectedId);
    setInt64Value(42);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Query size first
    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(expectedId.size() + 1));

    // Get the value
    std::vector<char> buffer(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, count, nullptr, buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), expectedId);
}

TEST_F(TestKnobSettingDescriptor, SetAndGetInt64Value)
{
    const int64_t expectedValue = 42;
    setKnobId("test_knob");
    setInt64Value(expectedValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t actualValue = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &actualValue));
    ASSERT_EQ(actualValue, expectedValue);
}

TEST_F(TestKnobSettingDescriptor, SetAndGetDoubleValue)
{
    const double expectedValue = 3.14;
    setKnobId("test_knob");
    setDoubleValue(expectedValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    double actualValue = 0.0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_DOUBLE, 1, nullptr, &actualValue));
    ASSERT_DOUBLE_EQ(actualValue, expectedValue);
}

TEST_F(TestKnobSettingDescriptor, SetAndGetStringValue)
{
    const std::string expectedValue = "strval";
    setKnobId("test_knob");
    setStringValue(expectedValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Query size first (two-call pattern)
    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(expectedValue.size() + 1));

    // Get the value
    std::vector<char> buffer(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_CHAR, count, nullptr, buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), expectedValue);
}

TEST_F(TestKnobSettingDescriptor, SetKnobValueWrongElementCountFailsInt64)
{
    int64_t val = 42;
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(
                                   HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 2, &val),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, SetKnobValueWrongElementCountFailsDouble)
{
    double val = 1.0;
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(
                                   HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_DOUBLE, 2, &val),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, SetKnobValueNullPointerFails)
{
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestKnobSettingDescriptor, SetKnobValueUnsupportedTypeFails)
{
    bool val = true;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_BOOLEAN, 1, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, GetAttributeBeforeFinalizeFails)
{
    setKnobId("test_knob");
    setInt64Value(42);

    int64_t val = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &val),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestKnobSettingDescriptor, GetAttributeUnsupportedAttributeFails)
{
    makeFinalized();
    int64_t val = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr, &val),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestKnobSettingDescriptor, GetKnobValueTypeMismatchFails)
{
    makeFinalized(); // sets int64 value
    double val = 0.0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_DOUBLE, 1, nullptr, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, ToKnobSettingTProducesCorrectObject)
{
    const std::string knobId = "test_knob_100";
    const int64_t value = 42;
    setKnobId(knobId);
    setInt64Value(value);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knobSettingT = getDescriptor()->toKnobSettingT();
    ASSERT_NE(knobSettingT, nullptr);
    ASSERT_EQ(knobSettingT->knob_id, knobId);
    ASSERT_EQ(knobSettingT->value.type, hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    ASSERT_EQ(knobSettingT->value.AsIntValue()->value, value);
}

TEST_F(TestKnobSettingDescriptor, ToKnobSettingTWithDoubleValue)
{
    const std::string knobId = "double_knob";
    const double value = 2.718;
    setKnobId(knobId);
    setDoubleValue(value);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knobSettingT = getDescriptor()->toKnobSettingT();
    ASSERT_NE(knobSettingT, nullptr);
    ASSERT_EQ(knobSettingT->knob_id, knobId);
    ASSERT_EQ(knobSettingT->value.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
    ASSERT_DOUBLE_EQ(knobSettingT->value.AsFloatValue()->value, value);
}

TEST_F(TestKnobSettingDescriptor, ToKnobSettingTBeforeFinalizeFails)
{
    setKnobId("test_knob");
    setInt64Value(42);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->toKnobSettingT(), HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestKnobSettingDescriptor, GetKnobIdWithElementCount)
{
    const std::string expectedId = "my_knob";
    makeFinalized(expectedId);

    int64_t count = 0;
    std::vector<char> buffer(64);
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, 64, &count, buffer.data()));
    ASSERT_EQ(count, static_cast<int64_t>(expectedId.size() + 1));
    ASSERT_EQ(std::string(buffer.data()), expectedId);
}

TEST_F(TestKnobSettingDescriptor, GetKnobValueWithElementCount)
{
    const int64_t expectedValue = 99;
    makeFinalized("knob", expectedValue);

    int64_t count = 0;
    int64_t actualValue = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, &count, &actualValue));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(actualValue, expectedValue);
}

TEST_F(TestKnobSettingDescriptor, GetStringValueTruncatesToBufferSize)
{
    const std::string longValue = "a_long_string_value";
    setKnobId("test_knob");
    setStringValue(longValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Provide a buffer smaller than the string
    const int64_t smallBufferSize = 8;
    std::array<char, 8> buffer = {};
    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                                  HIPDNN_TYPE_CHAR,
                                                  smallBufferSize,
                                                  &count,
                                                  buffer.data()));

    // Should be truncated and null-terminated
    ASSERT_EQ(count, smallBufferSize);
    ASSERT_EQ(buffer.back(), '\0');
    ASSERT_EQ(std::string(buffer.data()),
              longValue.substr(0, static_cast<size_t>(smallBufferSize - 1)));
}

TEST_F(TestKnobSettingDescriptor, GetStringValueSizeQueryWithNullBuffer)
{
    const std::string expectedValue = "query_size";
    setKnobId("test_knob");
    setStringValue(expectedValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(expectedValue.size() + 1));
}

TEST_F(TestKnobSettingDescriptor, GetStringValueSizeQueryNullElementCountFails)
{
    const std::string expectedValue = "test";
    setKnobId("test_knob");
    setStringValue(expectedValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_CHAR, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestKnobSettingDescriptor, ToKnobSettingTWithStringValue)
{
    const std::string knobId = "string_knob";
    const std::string value = "hello_world";
    setKnobId(knobId);
    setStringValue(value);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knobSettingT = getDescriptor()->toKnobSettingT();
    ASSERT_NE(knobSettingT, nullptr);
    ASSERT_EQ(knobSettingT->knob_id, knobId);
    ASSERT_EQ(knobSettingT->value.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
    ASSERT_EQ(knobSettingT->value.AsStringValue()->value, value);
}

// ============================================================================
// Validation limit tests
// ============================================================================

TEST_F(TestKnobSettingDescriptor, SetEmptyKnobIdFails)
{
    const std::string emptyId;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, 0, emptyId.c_str()),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, SetKnobIdExceedsMaxLengthFails)
{
    const std::string longId(static_cast<size_t>(KnobSettingDescriptor::MAX_KNOB_ID_LENGTH + 1),
                             'x');
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                                             HIPDNN_TYPE_CHAR,
                                                             static_cast<int64_t>(longId.size()),
                                                             longId.c_str()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, SetKnobIdAtMaxLengthSucceeds)
{
    const std::string maxId(static_cast<size_t>(KnobSettingDescriptor::MAX_KNOB_ID_LENGTH), 'x');
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(maxId.size()),
                                                  maxId.c_str()));
}

TEST_F(TestKnobSettingDescriptor, SetKnobStringValueExceedsMaxLengthFails)
{
    const std::string longValue(
        static_cast<size_t>(KnobSettingDescriptor::MAX_KNOB_STRING_VALUE_LENGTH + 1), 'y');
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                                             HIPDNN_TYPE_CHAR,
                                                             static_cast<int64_t>(longValue.size()),
                                                             longValue.c_str()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, SetKnobStringValueAtMaxLengthSucceeds)
{
    const std::string maxValue(
        static_cast<size_t>(KnobSettingDescriptor::MAX_KNOB_STRING_VALUE_LENGTH), 'y');
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(maxValue.size()),
                                                  maxValue.c_str()));
}

TEST_F(TestKnobSettingDescriptor, SetKnobIdNegativeElementCountFails)
{
    const std::string knobId = "test";
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, -1, knobId.c_str()),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, GetKnobIdNegativeRequestedCountFails)
{
    makeFinalized();
    std::array<char, 16> buffer{};
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, -1, nullptr, buffer.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobSettingDescriptor, GetStringValueNegativeRequestedCountFails)
{
    setKnobId("test_knob");
    setStringValue("some_value");
    ASSERT_NO_THROW(getDescriptor()->finalize());

    std::array<char, 16> buffer{};
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_CHAR, -1, nullptr, buffer.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// ============================================================================
// Overwrite behavior tests
// ============================================================================

TEST_F(TestKnobSettingDescriptor, OverwriteKnobIdBeforeFinalize)
{
    setKnobId("first_id");
    setKnobId("second_id");
    setInt64Value(42);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Verify the second ID took effect
    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(std::string("second_id").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, count, nullptr, buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), "second_id");
}

TEST_F(TestKnobSettingDescriptor, OverwriteKnobValueBeforeFinalize)
{
    setKnobId("test_knob");
    setInt64Value(10);
    setInt64Value(99);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Verify the second value took effect
    int64_t actualValue = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &actualValue));
    ASSERT_EQ(actualValue, 99);
}

TEST_F(TestKnobSettingDescriptor, OverwriteKnobValueTypeBeforeFinalize)
{
    setKnobId("test_knob");
    setInt64Value(42);
    setDoubleValue(3.14);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Verify the value is now a double
    double actualValue = 0.0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_DOUBLE, 1, nullptr, &actualValue));
    ASSERT_DOUBLE_EQ(actualValue, 3.14);
}

TEST_F(TestKnobSettingDescriptor, OverwriteIntValueWithStringBeforeFinalize)
{
    setKnobId("test_knob");
    setInt64Value(42);
    setStringValue("overwritten");
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Verify the value is now a string
    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(std::string("overwritten").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_CHAR, count, nullptr, buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), "overwritten");
}
