// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/DescriptorFactory.hpp"
#include "descriptors/KnobDescriptor.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;

class TestKnobDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<KnobDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<KnobDescriptor>();
    }

    void setKnobId(const std::string& knobId) const
    {
        ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE,
                                                      HIPDNN_TYPE_CHAR,
                                                      static_cast<int64_t>(knobId.size()),
                                                      knobId.c_str()));
    }

    void setInt64Default(int64_t value) const
    {
        ASSERT_NO_THROW(getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 1, &value));
    }

    void setDoubleDefault(double value) const
    {
        ASSERT_NO_THROW(getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_DOUBLE, 1, &value));
    }

    void setStringDefault(const std::string& value) const
    {
        ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE,
                                                      HIPDNN_TYPE_CHAR,
                                                      static_cast<int64_t>(value.size()),
                                                      value.c_str()));
    }

    void makeFinalized(const std::string& knobId = "test_knob", int64_t defaultVal = 0) const
    {
        setKnobId(knobId);
        setInt64Default(defaultVal);
        ASSERT_NO_THROW(getDescriptor()->finalize());
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper;

    void SetUp() override
    {
        _wrapper = createDescriptor<KnobDescriptor>();
    }
};

// ============================================================================
// Creation and lifecycle
// ============================================================================

TEST_F(TestKnobDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR);
}

TEST_F(TestKnobDescriptor, FinalizeWithKnobIdAndInt64Default)
{
    setKnobId("my_knob");
    setInt64Default(42);
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestKnobDescriptor, FinalizeWithoutKnobIdFails)
{
    setInt64Default(42);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeWithoutDefaultValueFails)
{
    setKnobId("my_knob");
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeAlreadyFinalizedFails)
{
    makeFinalized();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetAttributeAfterFinalizeFails)
{
    makeFinalized();
    const std::string knobId = "another_knob";
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE,
                                                             HIPDNN_TYPE_CHAR,
                                                             static_cast<int64_t>(knobId.size()),
                                                             knobId.c_str()),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

// ============================================================================
// Finalize type-consistency validation
// ============================================================================

TEST_F(TestKnobDescriptor, FinalizeIntKnobWithDoubleMinMaxFails)
{
    setKnobId("test_knob");
    setInt64Default(0);
    double val = 1.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &val));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeIntKnobWithValidValuesStringFails)
{
    using namespace std::string_literals;
    setKnobId("test_knob");
    setInt64Default(0);
    const auto buf = "a\0b"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(buf.size()),
                                                  buf.data()));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeIntKnobWithStringMaxLengthFails)
{
    setKnobId("test_knob");
    setInt64Default(0);
    int32_t maxLen = 100;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeDoubleKnobWithIntMinMaxFails)
{
    setKnobId("test_knob");
    setDoubleDefault(0.5);
    int64_t val = 1;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &val));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeDoubleKnobWithStrideFails)
{
    setKnobId("test_knob");
    setDoubleDefault(0.5);
    int64_t stride = 1;
    ASSERT_NO_THROW(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeDoubleKnobWithValidValuesIntFails)
{
    setKnobId("test_knob");
    setDoubleDefault(0.5);
    std::vector<int64_t> vals = {1, 2, 3};
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                                  HIPDNN_TYPE_INT64,
                                                  static_cast<int64_t>(vals.size()),
                                                  vals.data()));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeDoubleKnobWithValidValuesStringFails)
{
    using namespace std::string_literals;
    setKnobId("test_knob");
    setDoubleDefault(0.5);
    const auto buf = "a"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(buf.size()),
                                                  buf.data()));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeDoubleKnobWithStringMaxLengthFails)
{
    setKnobId("test_knob");
    setDoubleDefault(0.5);
    int32_t maxLen = 100;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeStringKnobWithIntMinMaxFails)
{
    setKnobId("test_knob");
    setStringDefault("hello");
    int64_t val = 1;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &val));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeStringKnobWithDoubleMinMaxFails)
{
    setKnobId("test_knob");
    setStringDefault("hello");
    double val = 1.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &val));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeStringKnobWithStrideFails)
{
    setKnobId("test_knob");
    setStringDefault("hello");
    int64_t stride = 1;
    ASSERT_NO_THROW(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeStringKnobWithValidValuesIntFails)
{
    setKnobId("test_knob");
    setStringDefault("hello");
    std::vector<int64_t> vals = {1, 2, 3};
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                                  HIPDNN_TYPE_INT64,
                                                  static_cast<int64_t>(vals.size()),
                                                  vals.data()));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// ============================================================================
// Knob ID
// ============================================================================

TEST_F(TestKnobDescriptor, SetKnobIdAsChar)
{
    const std::string knobId = "test_knob_id";
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(knobId.size()),
                                                  knobId.c_str()));
    setInt64Default(0);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(knobId.size() + 1));

    std::vector<char> buf(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, count, nullptr, buf.data()));
    ASSERT_EQ(std::string(buf.data()), knobId);
}

TEST_F(TestKnobDescriptor, SetKnobIdWrongTypeFails)
{
    int64_t val = 42;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_INT64, 1, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetKnobIdNullFails)
{
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 5, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestKnobDescriptor, SetEmptyKnobIdFails)
{
    const std::string emptyId;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 0, emptyId.c_str()),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetKnobIdExceedsMaxLengthFails)
{
    const std::string longId(static_cast<size_t>(KnobDescriptor::MAX_KNOB_ID_LENGTH + 1), 'x');
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE,
                                                             HIPDNN_TYPE_CHAR,
                                                             static_cast<int64_t>(longId.size()),
                                                             longId.c_str()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetKnobIdAtMaxLengthSucceeds)
{
    const std::string maxId(static_cast<size_t>(KnobDescriptor::MAX_KNOB_ID_LENGTH), 'x');
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(maxId.size()),
                                                  maxId.c_str()));
}

TEST_F(TestKnobDescriptor, GetKnobIdAfterFinalize)
{
    const std::string expectedId = "my_knob_123";
    setKnobId(expectedId);
    setInt64Default(0);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Query size first
    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(expectedId.size() + 1));

    // Get the value
    std::vector<char> buffer(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, count, nullptr, buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), expectedId);
}

TEST_F(TestKnobDescriptor, OverwriteKnobIdBeforeFinalize)
{
    setKnobId("first_id");
    setKnobId("second_id");
    setInt64Default(0);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));

    std::vector<char> buffer(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, count, nullptr, buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), "second_id");
}

TEST_F(TestKnobDescriptor, GetKnobIdTruncatesToBufferSize)
{
    const std::string knobId = "a_long_knob_id_name";
    setKnobId(knobId);
    setInt64Default(0);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    constexpr int64_t SMALL_BUF_SIZE = 6;
    std::vector<char> buf(static_cast<size_t>(SMALL_BUF_SIZE));
    int64_t written = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, SMALL_BUF_SIZE, &written, buf.data()));
    ASSERT_EQ(written, SMALL_BUF_SIZE);
    ASSERT_EQ(buf.back(), '\0');
    ASSERT_EQ(std::string(buf.data()), "a_lon");
}

// ============================================================================
// Description
// ============================================================================

TEST_F(TestKnobDescriptor, SetAndGetDescription)
{
    const std::string desc = "This is a helpful knob description.";
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_DESCRIPTION,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(desc.size()),
                                                  desc.c_str()));
    makeFinalized();

    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(desc.size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_CHAR, count, nullptr, buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), desc);
}

TEST_F(TestKnobDescriptor, DefaultDescriptionIsEmpty)
{
    makeFinalized();

    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    // Empty string → size 1 (null terminator only)
    ASSERT_EQ(count, 1);
}

TEST_F(TestKnobDescriptor, SetDescriptionWrongTypeFails)
{
    int64_t val = 0;
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(
                                   HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_INT64, 1, &val),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetDescriptionNullPtrWithCountFails)
{
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(
                                   HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_CHAR, 5, nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestKnobDescriptor, SetDescriptionZeroCountClears)
{
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_CHAR, 5, "hello"));
    ASSERT_NO_THROW(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_CHAR, 0, ""));
    makeFinalized();

    int64_t size = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_CHAR, 0, &size, nullptr));
    ASSERT_EQ(size, 1); // empty string: just null terminator
}

// ============================================================================
// Default value
// ============================================================================

TEST_F(TestKnobDescriptor, SetAndGetInt64Default)
{
    const int64_t expectedValue = 42;
    setKnobId("test_knob");
    setInt64Default(expectedValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t actualValue = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &actualValue));
    ASSERT_EQ(actualValue, expectedValue);
}

TEST_F(TestKnobDescriptor, SetAndGetDoubleDefault)
{
    const double expectedValue = 3.14;
    setKnobId("test_knob");
    setDoubleDefault(expectedValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    double actualValue = 0.0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_DOUBLE, 1, nullptr, &actualValue));
    ASSERT_DOUBLE_EQ(actualValue, expectedValue);
}

TEST_F(TestKnobDescriptor, SetAndGetStringDefault)
{
    const std::string expectedValue = "default_str";
    setKnobId("test_knob");
    setStringDefault(expectedValue);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(expectedValue.size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_CHAR, count, nullptr, buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), expectedValue);
}

TEST_F(TestKnobDescriptor, DefaultValueTypeMismatchFails)
{
    setKnobId("test_knob");
    setInt64Default(42);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    double val = 0.0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_DOUBLE, 1, nullptr, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, OverwriteDefaultValueBeforeFinalize)
{
    setKnobId("test_knob");
    setInt64Default(10);
    setInt64Default(99);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t actualValue = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &actualValue));
    ASSERT_EQ(actualValue, 99);
}

TEST_F(TestKnobDescriptor, OverwriteDefaultValueTypeBeforeFinalize)
{
    setKnobId("test_knob");
    setInt64Default(42);
    setDoubleDefault(2.718);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    double actualValue = 0.0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_DOUBLE, 1, nullptr, &actualValue));
    ASSERT_DOUBLE_EQ(actualValue, 2.718);
}

TEST_F(TestKnobDescriptor, DefaultValueUnsupportedTypeFails)
{
    bool val = true;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_BOOLEAN, 1, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetDefaultValueNullFails)
{
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestKnobDescriptor, GetDefaultValueTypeInt64)
{
    setKnobId("test_knob");
    setInt64Default(42);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t valueType = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE, HIPDNN_TYPE_INT64, 1, nullptr, &valueType));
    ASSERT_EQ(static_cast<hipdnnBackendAttributeType_t>(valueType), HIPDNN_TYPE_INT64);
}

TEST_F(TestKnobDescriptor, GetDefaultValueTypeDouble)
{
    setKnobId("test_knob");
    setDoubleDefault(3.14);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t valueType = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE, HIPDNN_TYPE_INT64, 1, nullptr, &valueType));
    ASSERT_EQ(static_cast<hipdnnBackendAttributeType_t>(valueType), HIPDNN_TYPE_DOUBLE);
}

TEST_F(TestKnobDescriptor, GetDefaultValueTypeChar)
{
    setKnobId("test_knob");
    setStringDefault("hello");
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t valueType = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE, HIPDNN_TYPE_INT64, 1, nullptr, &valueType));
    ASSERT_EQ(static_cast<hipdnnBackendAttributeType_t>(valueType), HIPDNN_TYPE_CHAR);
}

// ============================================================================
// Deprecated
// ============================================================================

TEST_F(TestKnobDescriptor, DefaultDeprecatedIsFalse)
{
    makeFinalized();

    bool val = true; // initialize to true to confirm it's overwritten
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, 1, nullptr, &val));
    ASSERT_FALSE(val);
}

TEST_F(TestKnobDescriptor, SetAndGetDeprecatedTrue)
{
    bool deprecated = true;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, 1, &deprecated));
    makeFinalized();

    bool val = false;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, 1, nullptr, &val));
    ASSERT_TRUE(val);
}

TEST_F(TestKnobDescriptor, SetDeprecatedWrongTypeFails)
{
    int64_t val = 1;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_INT64, 1, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

// ============================================================================
// Maximum / Minimum value
// ============================================================================

TEST_F(TestKnobDescriptor, SetAndGetMaxValueInt64)
{
    int64_t minVal = 0;
    int64_t maxVal = 100;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    makeFinalized();

    int64_t result = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &result));
    ASSERT_EQ(result, maxVal);
}

TEST_F(TestKnobDescriptor, SetAndGetMinValueInt64)
{
    int64_t minVal = -50;
    int64_t maxVal = 100;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    makeFinalized();

    int64_t result = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &result));
    ASSERT_EQ(result, minVal);
}

TEST_F(TestKnobDescriptor, GetMaxValueNotSetReturnsZeroCount)
{
    makeFinalized();

    int64_t count = 99;
    int64_t result = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &count, &result));
    ASSERT_EQ(count, 0);
}

TEST_F(TestKnobDescriptor, GetMinValueNotSetReturnsZeroCount)
{
    makeFinalized();

    int64_t count = 99;
    int64_t result = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &count, &result));
    ASSERT_EQ(count, 0);
}

TEST_F(TestKnobDescriptor, SetAndGetMaxValueDouble)
{
    double minVal = 0.0;
    double maxVal = 1.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &maxVal));
    setKnobId("test_knob");
    setDoubleDefault(0.5);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    double result = 0.0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, nullptr, &result));
    ASSERT_DOUBLE_EQ(result, maxVal);
}

TEST_F(TestKnobDescriptor, FinalizeIntMinGreaterThanMaxFails)
{
    setKnobId("test_knob");
    setInt64Default(0);
    int64_t minVal = 100;
    int64_t maxVal = 50;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeDoubleMinGreaterThanMaxFails)
{
    setKnobId("test_knob");
    setDoubleDefault(0.5);
    double minVal = 1.0;
    double maxVal = 0.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &maxVal));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeIntMinEqualsMaxSucceeds)
{
    setKnobId("test_knob");
    setInt64Default(5);
    int64_t val = 5;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &val));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &val));
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestKnobDescriptor, FinalizeDoubleMinEqualsMaxSucceeds)
{
    setKnobId("test_knob");
    setDoubleDefault(1.0);
    double val = 1.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &val));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &val));
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestKnobDescriptor, OnlyMinSetFinalizeFails)
{
    setKnobId("test_knob");
    setInt64Default(5);
    int64_t minVal = 1;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, OnlyMaxSetFinalizeFails)
{
    setKnobId("test_knob");
    setInt64Default(5);
    int64_t maxVal = 100;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// ============================================================================
// Stride
// ============================================================================

TEST_F(TestKnobDescriptor, SetAndGetStrideInt64)
{
    int64_t stride = 4;
    ASSERT_NO_THROW(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride));
    makeFinalized();

    int64_t result = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, nullptr, &result));
    ASSERT_EQ(result, stride);
}

TEST_F(TestKnobDescriptor, GetStrideNotSetReturnsZeroCount)
{
    makeFinalized();

    int64_t count = 99;
    int64_t result = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &count, &result));
    ASSERT_EQ(count, 0);
}

TEST_F(TestKnobDescriptor, SetZeroStrideFails)
{
    int64_t stride = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetNegativeStrideFails)
{
    int64_t stride = -1;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride),
        HIPDNN_STATUS_BAD_PARAM);
}

// ============================================================================
// Valid values int (array)
// ============================================================================

TEST_F(TestKnobDescriptor, SetAndGetValidValuesInt)
{
    std::vector<int64_t> values = {1, 2, 4, 8, 16};
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                                  HIPDNN_TYPE_INT64,
                                                  static_cast<int64_t>(values.size()),
                                                  values.data()));
    // Use a default value that is in the valid-values list (required by finalize validation).
    setKnobId("test_knob");
    setInt64Default(4);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Query count
    int64_t count = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT, HIPDNN_TYPE_INT64, 0, &count, nullptr));
    ASSERT_EQ(count, static_cast<int64_t>(values.size()));

    // Get values
    std::vector<int64_t> result(static_cast<size_t>(count));
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT, HIPDNN_TYPE_INT64, count, nullptr, result.data()));
    ASSERT_EQ(result, values);
}

TEST_F(TestKnobDescriptor, ValidValuesIntEmptyArray)
{
    makeFinalized();

    int64_t count = 99;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT, HIPDNN_TYPE_INT64, 0, &count, nullptr));
    ASSERT_EQ(count, 0);
}

TEST_F(TestKnobDescriptor, SetValidValuesIntNullPtrWithCountFails)
{
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT, HIPDNN_TYPE_INT64, 3, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// ============================================================================
// Valid values string (multi-call append)
// ============================================================================

TEST_F(TestKnobDescriptor, SetAndGetValidValuesString)
{
    // Flat null-separated buffer: "option_a\0option_b\0option_c\0"
    using namespace std::string_literals;
    const auto input = "option_a\0option_b\0option_c"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(input.size()),
                                                  input.data()));
    setKnobId("test_knob");
    // Use a default that is in the valid-values list (required by finalize validation).
    setStringDefault("option_a");
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Size query: total bytes needed
    int64_t totalBytes = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, &totalBytes, nullptr));
    // "option_a\0option_b\0option_c\0" = 9+9+9 = 27 bytes
    ASSERT_EQ(totalBytes, 27);

    // Copy into buffer and verify
    std::vector<char> buf(static_cast<size_t>(totalBytes));
    int64_t written = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  totalBytes,
                                                  &written,
                                                  buf.data()));
    ASSERT_EQ(written, totalBytes);

    // Parse the flat buffer back into strings
    const char* pos = buf.data();
    const char* end = pos + written;
    ASSERT_STREQ(pos, "option_a");
    pos += std::strlen(pos) + 1;
    ASSERT_STREQ(pos, "option_b");
    pos += std::strlen(pos) + 1;
    ASSERT_STREQ(pos, "option_c");
    pos += std::strlen(pos) + 1;
    ASSERT_EQ(pos, end);
}

TEST_F(TestKnobDescriptor, ValidValuesStringEmpty)
{
    makeFinalized();

    int64_t count = 99;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, &count, nullptr));
    ASSERT_EQ(count, 0);
}

TEST_F(TestKnobDescriptor, GetValidValuesStringTruncatesToBufferSize)
{
    // "a\0b\0c\0" = 6 bytes total
    using namespace std::string_literals;
    const auto input = "a\0b\0c"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(input.size()),
                                                  input.data()));
    setKnobId("test_knob");
    // Use a default that is in the valid-values list (required by finalize validation).
    setStringDefault("a");
    ASSERT_NO_THROW(getDescriptor()->finalize());

    // Request buffer of only 4 bytes — fits "a\0b\0" but not "c\0"
    std::array<char, 4> buf = {};
    int64_t written = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 4, &written, buf.data()));
    ASSERT_EQ(written, 4);
    ASSERT_STREQ(buf.data(), "a");
    ASSERT_STREQ(buf.data() + 2, "b");
}

TEST_F(TestKnobDescriptor, SetValidValuesStringNullPtrFails)
{
    ASSERT_THROW(getDescriptor()->setAttribute(
                     HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, nullptr),
                 HipdnnException);
}

TEST_F(TestKnobDescriptor, SetValidValuesStringReplaces)
{
    // Second set replaces, not appends
    using namespace std::string_literals;
    const auto first = "a\0b"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(first.size()),
                                                  first.data()));

    const auto second = "x"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(second.size()),
                                                  second.data()));

    setKnobId("test_knob");
    // Use a default that is in the final valid-values list (the second set wins: ["x"]).
    setStringDefault("x");
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int64_t totalBytes = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, &totalBytes, nullptr));
    ASSERT_EQ(totalBytes, 2); // "x\0"

    std::array<char, 2> buf = {};
    int64_t written = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 2, &written, buf.data()));
    ASSERT_EQ(written, 2);
    ASSERT_STREQ(buf.data(), "x");
}

// ============================================================================
// String max length
// ============================================================================

TEST_F(TestKnobDescriptor, SetAndGetStringMaxLength)
{
    int32_t maxLen = 256;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen));
    setKnobId("test_knob");
    setStringDefault("default");
    ASSERT_NO_THROW(getDescriptor()->finalize());

    int32_t result = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, nullptr, &result));
    ASSERT_EQ(result, maxLen);
}

TEST_F(TestKnobDescriptor, GetStringMaxLengthNotSetReturnsZeroCount)
{
    makeFinalized();

    int64_t count = 99;
    int32_t result = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &count, &result));
    ASSERT_EQ(count, 0);
}

TEST_F(TestKnobDescriptor, SetStringMaxLengthZeroFails)
{
    int32_t maxLen = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetStringMaxLengthNegativeFails)
{
    int32_t maxLen = -1;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, SetStringMaxLengthPositiveSucceeds)
{
    int32_t maxLen = 64;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen));
    setKnobId("test_knob");
    setStringDefault("hello");
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

// ============================================================================
// Finalize default value vs constraints validation — Int knob
// ============================================================================

TEST_F(TestKnobDescriptor, FinalizeFailsIntDefaultBelowMinimum)
{
    setKnobId("test_knob");
    setInt64Default(5);
    int64_t minVal = 10;
    int64_t maxVal = 20;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeFailsIntDefaultAboveMaximum)
{
    setKnobId("test_knob");
    setInt64Default(25);
    int64_t minVal = 10;
    int64_t maxVal = 20;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeSucceedsIntDefaultInRange)
{
    setKnobId("test_knob");
    setInt64Default(15);
    int64_t minVal = 10;
    int64_t maxVal = 20;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestKnobDescriptor, FinalizeFailsIntDefaultNotInValidValues)
{
    setKnobId("test_knob");
    setInt64Default(5);
    std::vector<int64_t> validVals = {1, 2, 3};
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                                  HIPDNN_TYPE_INT64,
                                                  static_cast<int64_t>(validVals.size()),
                                                  validVals.data()));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeSucceedsIntDefaultInValidValues)
{
    setKnobId("test_knob");
    setInt64Default(2);
    std::vector<int64_t> validVals = {1, 2, 3};
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                                  HIPDNN_TYPE_INT64,
                                                  static_cast<int64_t>(validVals.size()),
                                                  validVals.data()));
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestKnobDescriptor, FinalizeFailsIntDefaultNotAlignedToStride)
{
    // default=12, min=10, max=20, stride=5: (12-10)=2, 2%5!=0 → fail
    setKnobId("test_knob");
    setInt64Default(12);
    int64_t minVal = 10;
    int64_t maxVal = 20;
    int64_t stride = 5;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_NO_THROW(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeSucceedsIntDefaultAlignedToStride)
{
    // default=15, min=10, max=20, stride=5: (15-10)=5, 5%5==0 → succeed
    setKnobId("test_knob");
    setInt64Default(15);
    int64_t minVal = 10;
    int64_t maxVal = 20;
    int64_t stride = 5;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_NO_THROW(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride));
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

// ============================================================================
// Finalize default value vs constraints validation — Double knob
// ============================================================================

TEST_F(TestKnobDescriptor, FinalizeFailsDoubleDefaultBelowMinimum)
{
    setKnobId("test_knob");
    setDoubleDefault(0.5);
    double minVal = 1.0;
    double maxVal = 10.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &maxVal));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeFailsDoubleDefaultAboveMaximum)
{
    setKnobId("test_knob");
    setDoubleDefault(15.0);
    double minVal = 1.0;
    double maxVal = 10.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &maxVal));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeSucceedsDoubleDefaultInRange)
{
    setKnobId("test_knob");
    setDoubleDefault(5.0);
    double minVal = 1.0;
    double maxVal = 10.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &maxVal));
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

// ============================================================================
// Finalize default value vs constraints validation — String knob
// ============================================================================

TEST_F(TestKnobDescriptor, FinalizeFailsStringDefaultExceedsMaxLength)
{
    // default="abcdefgh" (len 8), string_max_length=4 → fail
    setKnobId("test_knob");
    setStringDefault("abcdefgh");
    int32_t maxLen = 4;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeSucceedsStringDefaultWithinMaxLength)
{
    // default="ab" (len 2), string_max_length=4 → succeed
    setKnobId("test_knob");
    setStringDefault("ab");
    int32_t maxLen = 4;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen));
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestKnobDescriptor, FinalizeFailsStringDefaultNotInValidValues)
{
    setKnobId("test_knob");
    setStringDefault("d");
    using namespace std::string_literals;
    const auto validValues = "a\0b\0c"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(validValues.size()),
                                                  validValues.data()));
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKnobDescriptor, FinalizeSucceedsStringDefaultInValidValues)
{
    setKnobId("test_knob");
    setStringDefault("b");
    using namespace std::string_literals;
    const auto validValues = "a\0b\0c"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(validValues.size()),
                                                  validValues.data()));
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

// ============================================================================
// toKnobT
// ============================================================================

TEST_F(TestKnobDescriptor, ToKnobTBeforeFinalizeFails)
{
    setKnobId("test_knob");
    setInt64Default(0);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->toKnobT(), HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestKnobDescriptor, ToKnobTWithInt64Default)
{
    const std::string knobId = "int_knob";
    const int64_t value = 42;
    setKnobId(knobId);
    setInt64Default(value);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knob = getDescriptor()->toKnobT();
    ASSERT_NE(knob, nullptr);
    ASSERT_EQ(knob->knob_id, knobId);
    ASSERT_EQ(knob->default_value.type, hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    ASSERT_EQ(knob->default_value.AsIntValue()->value, value);
    ASSERT_EQ(knob->constraint.type, hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::NONE);
}

TEST_F(TestKnobDescriptor, ToKnobTWithDoubleDefault)
{
    const std::string knobId = "float_knob";
    const double value = 2.718;
    setKnobId(knobId);
    setDoubleDefault(value);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knob = getDescriptor()->toKnobT();
    ASSERT_NE(knob, nullptr);
    ASSERT_EQ(knob->knob_id, knobId);
    ASSERT_EQ(knob->default_value.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue);
    ASSERT_DOUBLE_EQ(knob->default_value.AsFloatValue()->value, value);
}

TEST_F(TestKnobDescriptor, ToKnobTWithStringDefault)
{
    const std::string knobId = "string_knob";
    const std::string value = "hello";
    setKnobId(knobId);
    setStringDefault(value);
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knob = getDescriptor()->toKnobT();
    ASSERT_NE(knob, nullptr);
    ASSERT_EQ(knob->knob_id, knobId);
    ASSERT_EQ(knob->default_value.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
    ASSERT_EQ(knob->default_value.AsStringValue()->value, value);
}

TEST_F(TestKnobDescriptor, ToKnobTWithIntConstraints)
{
    setKnobId("constrained_int_knob");
    setInt64Default(4);

    // min=2, stride=2: (4-2)=2, 2%2==0 → default is aligned to stride.
    // valid_values contains 4 → default is in the set.
    int64_t minVal = 2;
    int64_t maxVal = 16;
    int64_t stride = 2;
    std::vector<int64_t> validVals = {2, 4, 8, 16};

    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_NO_THROW(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                                  HIPDNN_TYPE_INT64,
                                                  static_cast<int64_t>(validVals.size()),
                                                  validVals.data()));
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knob = getDescriptor()->toKnobT();
    ASSERT_NE(knob, nullptr);
    ASSERT_EQ(knob->constraint.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::IntConstraint);

    const auto* constraint = knob->constraint.AsIntConstraint();
    ASSERT_NE(constraint, nullptr);
    ASSERT_EQ(constraint->min_value, minVal);
    ASSERT_EQ(constraint->max_value, maxVal);
    ASSERT_EQ(constraint->step, stride);
    ASSERT_EQ(constraint->valid_values, validVals);
}

TEST_F(TestKnobDescriptor, ToKnobTWithFloatConstraints)
{
    setKnobId("float_constrained_knob");
    setDoubleDefault(0.5);

    double minVal = 0.0;
    double maxVal = 1.0;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1, &maxVal));
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knob = getDescriptor()->toKnobT();
    ASSERT_NE(knob, nullptr);
    ASSERT_EQ(knob->constraint.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::FloatConstraint);

    const auto* constraint = knob->constraint.AsFloatConstraint();
    ASSERT_NE(constraint, nullptr);
    ASSERT_DOUBLE_EQ(constraint->min_value, minVal);
    ASSERT_DOUBLE_EQ(constraint->max_value, maxVal);
}

TEST_F(TestKnobDescriptor, ToKnobTWithStringConstraints)
{
    setKnobId("string_constrained_knob");
    setStringDefault("option_a");

    using namespace std::string_literals;
    const std::vector<std::string> validStrings = {"option_a", "option_b"};
    const auto validValues = "option_a\0option_b"s;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(validValues.size()),
                                                  validValues.data()));
    int32_t maxLen = 32;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, &maxLen));
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knob = getDescriptor()->toKnobT();
    ASSERT_NE(knob, nullptr);
    ASSERT_EQ(knob->constraint.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::StringConstraint);

    const auto* constraint = knob->constraint.AsStringConstraint();
    ASSERT_NE(constraint, nullptr);
    ASSERT_EQ(constraint->max_length, static_cast<int32_t>(maxLen));
    ASSERT_EQ(constraint->valid_values, validStrings);
}

TEST_F(TestKnobDescriptor, ToKnobTWithDeprecatedAndDescription)
{
    setKnobId("deprecated_knob");
    setInt64Default(0);

    const std::string desc = "This knob is deprecated.";
    ASSERT_NO_THROW(getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_DESCRIPTION,
                                                  HIPDNN_TYPE_CHAR,
                                                  static_cast<int64_t>(desc.size()),
                                                  desc.c_str()));
    bool deprecated = true;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, 1, &deprecated));
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto knob = getDescriptor()->toKnobT();
    ASSERT_NE(knob, nullptr);
    ASSERT_EQ(knob->description, desc);
    ASSERT_TRUE(knob->deprecated);
}

// ============================================================================
// toString
// ============================================================================

TEST_F(TestKnobDescriptor, ToStringWithIntConstraints)
{
    setKnobId("ts_knob");
    // default=3, min=1, stride=2: (3-1)=2, 2%2==0 → aligned to stride.
    setInt64Default(3);
    int64_t minVal = 1;
    int64_t maxVal = 16;
    int64_t stride = 2;
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minVal));
    ASSERT_NO_THROW(getDescriptor()->setAttribute(
        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxVal));
    ASSERT_NO_THROW(
        getDescriptor()->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &stride));
    ASSERT_NO_THROW(getDescriptor()->finalize());

    auto str = getDescriptor()->toString();
    EXPECT_NE(str.find("ts_knob"), std::string::npos);
    EXPECT_NE(str.find("intConstraint"), std::string::npos);
    EXPECT_NE(str.find("min=1"), std::string::npos);
    EXPECT_NE(str.find("max=16"), std::string::npos);
    EXPECT_NE(str.find("step=2"), std::string::npos);
}

// ============================================================================
// getAttribute before finalize
// ============================================================================

TEST_F(TestKnobDescriptor, GetAttributeBeforeFinalizeFails)
{
    setKnobId("test_knob");
    setInt64Default(0);

    int64_t val = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 1, nullptr, &val),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestKnobDescriptor, GetAttributeUnsupportedAttributeFails)
{
    makeFinalized();
    int64_t val = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr, &val),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestKnobDescriptor, SetAttributeUnsupportedAttributeFails)
{
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// ============================================================================
// fromKnobT factory method
// ============================================================================

TEST(TestKnobDescriptorFromKnobT, IntKnobWithConstraints)
{
    hipdnn_flatbuffers_sdk::data_objects::KnobT knobT;
    knobT.knob_id = "test.int_knob";
    knobT.description = "An int knob";
    knobT.deprecated = false;
    hipdnn_flatbuffers_sdk::data_objects::IntValueT intDefault;
    intDefault.value = 50;
    knobT.default_value.Set(intDefault);

    hipdnn_flatbuffers_sdk::data_objects::IntConstraintT intConstraint;
    intConstraint.min_value = 0;
    intConstraint.max_value = 100;
    intConstraint.step = 10;
    knobT.constraint.Set(std::move(intConstraint));

    auto desc = KnobDescriptor::fromKnobT(knobT);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    auto roundTrip = desc->toKnobT();
    ASSERT_NE(roundTrip, nullptr);
    EXPECT_EQ(roundTrip->knob_id, "test.int_knob");
    EXPECT_EQ(roundTrip->description, "An int knob");
    EXPECT_FALSE(roundTrip->deprecated);
    EXPECT_EQ(roundTrip->default_value.AsIntValue()->value, 50);

    const auto* c = roundTrip->constraint.AsIntConstraint();
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->min_value, 0);
    EXPECT_EQ(c->max_value, 100);
    EXPECT_EQ(c->step, 10);
    EXPECT_TRUE(c->valid_values.empty());
}

TEST(TestKnobDescriptorFromKnobT, FloatKnobWithConstraints)
{
    hipdnn_flatbuffers_sdk::data_objects::KnobT knobT;
    knobT.knob_id = "test.float_knob";
    knobT.description = "A float knob";
    hipdnn_flatbuffers_sdk::data_objects::FloatValueT floatDefault;
    floatDefault.value = 0.5;
    knobT.default_value.Set(floatDefault);

    hipdnn_flatbuffers_sdk::data_objects::FloatConstraintT floatConstraint;
    floatConstraint.min_value = 0.0;
    floatConstraint.max_value = 1.0;
    knobT.constraint.Set(floatConstraint);

    auto desc = KnobDescriptor::fromKnobT(knobT);
    ASSERT_NE(desc, nullptr);

    auto roundTrip = desc->toKnobT();
    ASSERT_NE(roundTrip, nullptr);
    EXPECT_EQ(roundTrip->knob_id, "test.float_knob");
    EXPECT_DOUBLE_EQ(roundTrip->default_value.AsFloatValue()->value, 0.5);

    const auto* c = roundTrip->constraint.AsFloatConstraint();
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->min_value, 0.0);
    EXPECT_DOUBLE_EQ(c->max_value, 1.0);
}

TEST(TestKnobDescriptorFromKnobT, StringKnobWithConstraints)
{
    hipdnn_flatbuffers_sdk::data_objects::KnobT knobT;
    knobT.knob_id = "test.string_knob";
    knobT.description = "A string knob";
    hipdnn_flatbuffers_sdk::data_objects::StringValueT stringDefault;
    stringDefault.value = "fast";
    knobT.default_value.Set(std::move(stringDefault));

    hipdnn_flatbuffers_sdk::data_objects::StringConstraintT stringConstraint;
    stringConstraint.max_length = 32;
    stringConstraint.valid_values = {"fast", "slow", "balanced"};
    knobT.constraint.Set(std::move(stringConstraint));

    auto desc = KnobDescriptor::fromKnobT(knobT);
    ASSERT_NE(desc, nullptr);

    auto roundTrip = desc->toKnobT();
    ASSERT_NE(roundTrip, nullptr);
    EXPECT_EQ(roundTrip->knob_id, "test.string_knob");
    EXPECT_EQ(roundTrip->default_value.AsStringValue()->value, "fast");

    const auto* c = roundTrip->constraint.AsStringConstraint();
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->max_length, 32);
    EXPECT_EQ(c->valid_values.size(), 3u);
}

TEST(TestKnobDescriptorFromKnobT, DeprecatedKnob)
{
    hipdnn_flatbuffers_sdk::data_objects::KnobT knobT;
    knobT.knob_id = "test.deprecated";
    knobT.deprecated = true;
    hipdnn_flatbuffers_sdk::data_objects::IntValueT intDefault;
    intDefault.value = 0;
    knobT.default_value.Set(intDefault);

    auto desc = KnobDescriptor::fromKnobT(knobT);
    ASSERT_NE(desc, nullptr);

    auto roundTrip = desc->toKnobT();
    ASSERT_NE(roundTrip, nullptr);
    EXPECT_TRUE(roundTrip->deprecated);
}

TEST(TestKnobDescriptorFromKnobT, IntKnobNoConstraint)
{
    hipdnn_flatbuffers_sdk::data_objects::KnobT knobT;
    knobT.knob_id = "test.unconstrained";
    hipdnn_flatbuffers_sdk::data_objects::IntValueT intDefault;
    intDefault.value = 42;
    knobT.default_value.Set(intDefault);

    auto desc = KnobDescriptor::fromKnobT(knobT);
    ASSERT_NE(desc, nullptr);

    auto roundTrip = desc->toKnobT();
    ASSERT_NE(roundTrip, nullptr);
    EXPECT_EQ(roundTrip->default_value.AsIntValue()->value, 42);
    EXPECT_EQ(roundTrip->constraint.type,
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::NONE);
}

TEST(TestKnobDescriptorFromKnobT, IntKnobWithValidValues)
{
    hipdnn_flatbuffers_sdk::data_objects::KnobT knobT;
    knobT.knob_id = "test.valid_values";
    hipdnn_flatbuffers_sdk::data_objects::IntValueT intDefault;
    intDefault.value = 16;
    knobT.default_value.Set(intDefault);

    hipdnn_flatbuffers_sdk::data_objects::IntConstraintT intConstraint;
    intConstraint.valid_values = {8, 16, 32, 64};
    knobT.constraint.Set(std::move(intConstraint));

    auto desc = KnobDescriptor::fromKnobT(knobT);
    ASSERT_NE(desc, nullptr);

    auto roundTrip = desc->toKnobT();
    ASSERT_NE(roundTrip, nullptr);
    const auto* c = roundTrip->constraint.AsIntConstraint();
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->valid_values, (std::vector<int64_t>{8, 16, 32, 64}));
}

TEST(TestKnobDescriptorFromKnobT, UnknownValueTypeReturnsNull)
{
    hipdnn_flatbuffers_sdk::data_objects::KnobT knobT;
    knobT.knob_id = "test.unknown_type";
    // default_value is NONE by default (no Set called)

    auto desc = KnobDescriptor::fromKnobT(knobT);
    EXPECT_EQ(desc, nullptr);
}

// ============================================================================
// DescriptorFactory
// ============================================================================

TEST(TestKnobDescriptorFactory, CreateKnobInfoDescriptorViaFactory)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(hipdnn_backend::DescriptorFactory::create(HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
                                                              &descriptor));
    EXPECT_NE(descriptor, nullptr);

    auto knobDesc = descriptor->asDescriptor<KnobDescriptor>();
    EXPECT_NE(knobDesc, nullptr);
    EXPECT_FALSE(knobDesc->isFinalized());
    EXPECT_EQ(knobDesc->getType(), HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR);

    ASSERT_NO_THROW(hipdnn_backend::DescriptorFactory::destroy(descriptor));
}
