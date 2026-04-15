// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/KnobUnpacker.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>

#include "fake_backend/BackendTestMatchers.hpp"
#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
using namespace hipdnn_frontend::test;
using namespace ::testing;

namespace
{

class TestKnobUnpacker : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);
    }
    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    // Helper: mock reading a string attribute via the size-query + data-read pattern
    void mockStringAttr(hipdnnBackendDescriptor_t desc,
                        hipdnnBackendAttributeName_t attrName,
                        const std::string& value)
    {
        // Size query (count only, nullptr data)
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(desc, attrName, HIPDNN_TYPE_CHAR, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(value.size() + 1)),
                            Return(HIPDNN_STATUS_SUCCESS)));

        // Data read
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(desc,
                                        attrName,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(value.size() + 1),
                                        _,
                                        NotNull()))
            .WillOnce(DoAll(
                SetArgPointee<4>(static_cast<int64_t>(value.size() + 1)),
                Invoke([value](hipdnnBackendDescriptor_t,
                               hipdnnBackendAttributeName_t,
                               hipdnnBackendAttributeType_t,
                               int64_t,
                               int64_t*,
                               void* out) { std::memcpy(out, value.c_str(), value.size() + 1); }),
                Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Helper: mock reading a scalar attribute
    template <typename T>
    void mockScalarAttr(hipdnnBackendDescriptor_t desc,
                        hipdnnBackendAttributeName_t attrName,
                        hipdnnBackendAttributeType_t attrType,
                        T value)
    {
        EXPECT_CALL(*_mockBackend, backendGetAttribute(desc, attrName, attrType, 1, _, NotNull()))
            .WillOnce(DoAll(Invoke([value](hipdnnBackendDescriptor_t,
                                           hipdnnBackendAttributeName_t,
                                           hipdnnBackendAttributeType_t,
                                           int64_t,
                                           int64_t*,
                                           void* out) { *static_cast<T*>(out) = value; }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Helper: mock an optional scalar attribute that is present
    template <typename T>
    void mockOptionalScalarAttr(hipdnnBackendDescriptor_t desc,
                                hipdnnBackendAttributeName_t attrName,
                                hipdnnBackendAttributeType_t attrType,
                                T value)
    {
        // Count query (returns 1 = present)
        EXPECT_CALL(*_mockBackend, backendGetAttribute(desc, attrName, attrType, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(1), Return(HIPDNN_STATUS_SUCCESS)));

        // Value read
        mockScalarAttr(desc, attrName, attrType, value);
    }

    // Helper: mock an optional scalar attribute that is absent
    void mockOptionalScalarAbsent(hipdnnBackendDescriptor_t desc,
                                  hipdnnBackendAttributeName_t attrName,
                                  hipdnnBackendAttributeType_t attrType)
    {
        EXPECT_CALL(*_mockBackend, backendGetAttribute(desc, attrName, attrType, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Helper: mock int64 vector attribute (empty)
    void mockEmptyVecAttr(hipdnnBackendDescriptor_t desc, hipdnnBackendAttributeName_t attrName)
    {
        // Count query returns 0
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(desc, attrName, HIPDNN_TYPE_INT64, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));
    }
};

// ============================================================================
// unpackKnobDescriptor tests
// ============================================================================

TEST_F(TestKnobUnpacker, UnpackIntKnobWithConstraints)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    // Mock: knob ID
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.int_knob");

    // Mock: description
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "An integer knob");

    // Mock: deprecated flag
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);

    // Mock: default value type = INT64
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_INT64));

    // Mock: default value = 50
    mockScalarAttr<int64_t>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 50);

    // Mock: int constraints (min=0, max=100, stride=10)
    mockOptionalScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 0);
    mockOptionalScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 100);
    mockOptionalScalarAttr<int64_t>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 10);

    // Mock: valid values (empty)
    mockEmptyVecAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_TRUE(knob.has_value());
    const auto& unpackedKnob = knob.value();
    EXPECT_EQ(unpackedKnob.knobId(), "test.int_knob");
    EXPECT_EQ(unpackedKnob.description(), "An integer knob");
    EXPECT_FALSE(unpackedKnob.isDeprecated());
    EXPECT_EQ(unpackedKnob.valueType(), KnobValueType::INT64);

    auto* defaultVal = std::get_if<int64_t>(&unpackedKnob.defaultValue());
    ASSERT_NE(defaultVal, nullptr);
    EXPECT_EQ(*defaultVal, 50);

    // Check constraint
    ASSERT_NE(unpackedKnob.constraint(), nullptr);
    auto* intConstraint = dynamic_cast<const IntConstraint*>(unpackedKnob.constraint());
    ASSERT_NE(intConstraint, nullptr);
    EXPECT_EQ(intConstraint->getMinValue(), 0);
    EXPECT_EQ(intConstraint->getMaxValue(), 100);
    EXPECT_EQ(intConstraint->getStep(), 10);
}

TEST_F(TestKnobUnpacker, UnpackFloatKnobWithConstraints)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.float_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "A float knob");
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);

    // Default value type = DOUBLE
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_DOUBLE));

    // Default value = 0.5
    mockScalarAttr<double>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_DOUBLE, 0.5);

    // Float constraints: min=0.0, max=1.0
    mockOptionalScalarAttr<double>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 0.0);
    mockOptionalScalarAttr<double>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE, 1.0);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_TRUE(knob.has_value());
    const auto& unpackedKnob = knob.value();
    EXPECT_EQ(unpackedKnob.knobId(), "test.float_knob");
    EXPECT_EQ(unpackedKnob.valueType(), KnobValueType::FLOAT64);

    auto* defaultVal = std::get_if<double>(&unpackedKnob.defaultValue());
    ASSERT_NE(defaultVal, nullptr);
    EXPECT_DOUBLE_EQ(*defaultVal, 0.5);

    ASSERT_NE(unpackedKnob.constraint(), nullptr);
    auto* floatConstraint = dynamic_cast<const FloatConstraint*>(unpackedKnob.constraint());
    ASSERT_NE(floatConstraint, nullptr);
    EXPECT_DOUBLE_EQ(floatConstraint->getMinValue(), 0.0);
    EXPECT_DOUBLE_EQ(floatConstraint->getMaxValue(), 1.0);
}

TEST_F(TestKnobUnpacker, UnpackStringKnobWithValidValues)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.string_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "A string knob");
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);

    // Default value type = CHAR
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_CHAR));

    // Default value = "fast"
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, "fast");

    // String constraints: max length = 20
    mockOptionalScalarAttr<int32_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 20);

    // Valid values string: "fast\0accurate\0balanced\0"
    const std::string validValBuf
        = std::string("fast") + '\0' + "accurate" + '\0' + "balanced" + '\0';
    const auto validValBufLen = static_cast<int64_t>(validValBuf.size());

    // Size query
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(validValBufLen), Return(HIPDNN_STATUS_SUCCESS)));

    // Data read
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                    HIPDNN_TYPE_CHAR,
                                    validValBufLen,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(SetArgPointee<4>(validValBufLen),
                        Invoke([&validValBuf](hipdnnBackendDescriptor_t,
                                              hipdnnBackendAttributeName_t,
                                              hipdnnBackendAttributeType_t,
                                              int64_t,
                                              int64_t*,
                                              void* out) {
                            std::memcpy(out, validValBuf.data(), validValBuf.size());
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_TRUE(knob.has_value());
    const auto& unpackedKnob = knob.value();
    EXPECT_EQ(unpackedKnob.knobId(), "test.string_knob");
    EXPECT_EQ(unpackedKnob.valueType(), KnobValueType::STRING);

    auto* defaultVal = std::get_if<std::string>(&unpackedKnob.defaultValue());
    ASSERT_NE(defaultVal, nullptr);
    EXPECT_EQ(*defaultVal, "fast");

    ASSERT_NE(unpackedKnob.constraint(), nullptr);
    auto* strConstraint = dynamic_cast<const StringConstraint*>(unpackedKnob.constraint());
    ASSERT_NE(strConstraint, nullptr);
    EXPECT_EQ(strConstraint->getMaxLength(), 20);

    const auto& validValues = strConstraint->getValidValues();
    EXPECT_EQ(validValues.size(), 3u);
    EXPECT_NE(validValues.find("fast"), validValues.end());
    EXPECT_NE(validValues.find("accurate"), validValues.end());
    EXPECT_NE(validValues.find("balanced"), validValues.end());
}

TEST_F(TestKnobUnpacker, UnpackDeprecatedKnob)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.deprecated_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "A deprecated knob");

    // Deprecated = true
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, true);

    // Default value type = INT64
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_INT64));
    // Default value = 0
    mockScalarAttr<int64_t>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 0);

    // No constraints
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64);
    mockEmptyVecAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_TRUE(knob.has_value());
    EXPECT_EQ(knob->knobId(), "test.deprecated_knob");
    EXPECT_TRUE(knob->isDeprecated());
}

TEST_F(TestKnobUnpacker, UnpackStringKnobIgnoresEmptyValidValueEntries)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.sparse_string_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "String knob with gaps");
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_CHAR));
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, "fast");
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32);

    // Buffer contains an empty entry between valid values.
    const std::string validValBuf = std::string("fast") + '\0' + '\0' + "balanced" + '\0';
    const auto validValBufLen = static_cast<int64_t>(validValBuf.size());

    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(validValBufLen), Return(HIPDNN_STATUS_SUCCESS)));

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                    HIPDNN_TYPE_CHAR,
                                    validValBufLen,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(SetArgPointee<4>(validValBufLen),
                        Invoke([&validValBuf](hipdnnBackendDescriptor_t,
                                              hipdnnBackendAttributeName_t,
                                              hipdnnBackendAttributeType_t,
                                              int64_t,
                                              int64_t*,
                                              void* out) {
                            std::memcpy(out, validValBuf.data(), validValBuf.size());
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_TRUE(knob.has_value());
    auto* strConstraint = dynamic_cast<const StringConstraint*>(knob->constraint());
    ASSERT_NE(strConstraint, nullptr);
    EXPECT_EQ(strConstraint->getValidValues(),
              (std::unordered_set<std::string>{"fast", "balanced"}));
}

TEST_F(TestKnobUnpacker, UnpackStringKnobFailsWhenValidValuesReadFails)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.bad_string_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "String knob read failure");
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_CHAR));
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, "fast");
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32);

    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{8}), Return(HIPDNN_STATUS_SUCCESS)));

    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 8, _, NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    EXPECT_TRUE(err.is_bad());
    EXPECT_FALSE(knob.has_value());
    EXPECT_NE(err.get_message().find("failed to read valid string values"), std::string::npos);
}

TEST_F(TestKnobUnpacker, UnpackStringKnobWithNotSupportedValidValuesGetsEmptyConstraint)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.legacy_string_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "Legacy string knob");
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_CHAR));
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, "fast");
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32);

    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_NOT_SUPPORTED));

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_TRUE(knob.has_value());
    auto* emptyConstraint = dynamic_cast<const EmptyConstraint*>(knob->constraint());
    EXPECT_NE(emptyConstraint, nullptr)
        << "Expected EmptyConstraint but got: " << knob->constraint()->toString();
}

TEST_F(TestKnobUnpacker, UnpackKnobWithNoConstraintsGetsEmptyConstraint)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.unconstrained");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "Unconstrained knob");
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);

    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_INT64));
    mockScalarAttr<int64_t>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 42);

    // All optional constraints absent
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64);
    mockEmptyVecAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_TRUE(knob.has_value());
    ASSERT_NE(knob->constraint(), nullptr);

    auto* emptyConstraint = dynamic_cast<const EmptyConstraint*>(knob->constraint());
    EXPECT_NE(emptyConstraint, nullptr)
        << "Expected EmptyConstraint but got: " << knob->constraint()->toString();
}

TEST_F(TestKnobUnpacker, UnpackFailsWithEmptyKnobId)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    // Mock: knob ID size query returns 0
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    EXPECT_TRUE(err.is_bad());
    EXPECT_FALSE(knob.has_value());
    EXPECT_NE(err.get_message().find("empty knob ID"), std::string::npos);
}

TEST_F(TestKnobUnpacker, UnpackFailsWithUnknownValueType)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.bad_type");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "Bad type knob");
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);

    // Default value type = some unknown type (9999)
    mockScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE, HIPDNN_TYPE_INT64, 9999);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    EXPECT_TRUE(err.is_bad());
    EXPECT_FALSE(knob.has_value());
    EXPECT_NE(err.get_message().find("unknown default value type"), std::string::npos);
}

TEST_F(TestKnobUnpacker, UnpackIntKnobWithValidValues)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, "test.valid_values_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "Knob with valid values");
    mockScalarAttr<bool>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);

    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_INT64));
    mockScalarAttr<int64_t>(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 8);

    // No min/max/stride
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64);

    // Valid values: {8, 16, 32, 64}
    std::vector<int64_t> validValues = {8, 16, 32, 64};
    // Count query
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT, HIPDNN_TYPE_INT64, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(validValues.size())),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Data read
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                    HIPDNN_TYPE_INT64,
                                    static_cast<int64_t>(validValues.size()),
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(validValues.size())),
                        Invoke([&validValues](hipdnnBackendDescriptor_t,
                                              hipdnnBackendAttributeName_t,
                                              hipdnnBackendAttributeType_t,
                                              int64_t,
                                              int64_t*,
                                              void* out) {
                            std::memcpy(
                                out, validValues.data(), validValues.size() * sizeof(int64_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);
    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_TRUE(knob.has_value());

    ASSERT_NE(knob->constraint(), nullptr);
    auto* intConstraint = dynamic_cast<const IntConstraint*>(knob->constraint());
    ASSERT_NE(intConstraint, nullptr);

    const auto& vv = intConstraint->getValidValues();
    EXPECT_EQ(vv.size(), 4u);
    EXPECT_NE(vv.find(8), vv.end());
    EXPECT_NE(vv.find(16), vv.end());
    EXPECT_NE(vv.find(32), vv.end());
    EXPECT_NE(vv.find(64), vv.end());
}

// ============================================================================
// unpackKnobsFromDescriptors tests
// ============================================================================

class TestUnpackKnobsFromDescriptors : public TestKnobUnpacker
{
protected:
    int _desc1Placeholder = 1;
    int _desc2Placeholder = 2;
    hipdnnBackendDescriptor_t _fakeDesc1
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_desc1Placeholder);
    hipdnnBackendDescriptor_t _fakeDesc2
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_desc2Placeholder);

    int _enginePlaceholder = 0;
    hipdnnBackendDescriptor_t _fakeEngine
        = reinterpret_cast<hipdnnBackendDescriptor_t>(&_enginePlaceholder);

    /// Set up HIPDNN_ATTR_ENGINE_KNOB_INFO count+data queries on _fakeEngine to return descs.
    void expectKnobDescArrayQuery(const std::vector<hipdnnBackendDescriptor_t>& descs)
    {
        auto count = static_cast<int64_t>(descs.size());

        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(_fakeEngine,
                                        HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        _,
                                        nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(count), Return(HIPDNN_STATUS_SUCCESS)));

        if(count > 0)
        {
            EXPECT_CALL(*_mockBackend,
                        backendGetAttribute(_fakeEngine,
                                            HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            count,
                                            _,
                                            NotNull()))
                .WillOnce(DoAll(SetArgPointee<4>(count),
                                Invoke([descs](hipdnnBackendDescriptor_t,
                                               hipdnnBackendAttributeName_t,
                                               hipdnnBackendAttributeType_t,
                                               int64_t,
                                               int64_t*,
                                               void* out) {
                                    auto* arr = static_cast<hipdnnBackendDescriptor_t*>(out);
                                    for(size_t i = 0; i < descs.size(); ++i)
                                    {
                                        arr[i] = descs[i];
                                    }
                                }),
                                Return(HIPDNN_STATUS_SUCCESS)));
        }
    }

    /// Fully mock unpackKnobDescriptor for an int knob on fakeDesc.
    void expectIntKnobMocks(hipdnnBackendDescriptor_t fakeDesc, const std::string& knobId)
    {
        mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, knobId);
        mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, "desc");
        mockScalarAttr<bool>(
            fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, false);
        mockScalarAttr<int64_t>(fakeDesc,
                                HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                                HIPDNN_TYPE_INT64,
                                static_cast<int64_t>(HIPDNN_TYPE_INT64));
        mockScalarAttr<int64_t>(
            fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_INT64, 0);
        mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64);
        mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64);
        mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64);
        mockEmptyVecAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT);
    }
};

TEST_F(TestUnpackKnobsFromDescriptors, EmptyArrayReturnsEmptyKnobs)
{
    expectKnobDescArrayQuery({});

    std::vector<Knob> knobs;
    auto err = unpackKnobsFromDescriptors(_fakeEngine, knobs);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestUnpackKnobsFromDescriptors, AggregatesTwoKnobs)
{
    expectKnobDescArrayQuery({_fakeDesc1, _fakeDesc2});

    expectIntKnobMocks(_fakeDesc1, "knob.one");
    expectIntKnobMocks(_fakeDesc2, "knob.two");

    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::vector<Knob> knobs;
    auto err = unpackKnobsFromDescriptors(_fakeEngine, knobs);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    ASSERT_EQ(knobs.size(), 2u);
    EXPECT_EQ(knobs[0].knobId(), "knob.one");
    EXPECT_EQ(knobs[1].knobId(), "knob.two");
}

TEST_F(TestUnpackKnobsFromDescriptors, CountQueryFailurePropagates)
{
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_fakeEngine,
                                    HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::vector<Knob> knobs;
    auto err = unpackKnobsFromDescriptors(_fakeEngine, knobs);

    EXPECT_TRUE(err.is_bad());
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestUnpackKnobsFromDescriptors, MidArrayFailureSkipsInvalidKnob)
{
    expectKnobDescArrayQuery({_fakeDesc1, _fakeDesc2});

    // First knob unpacks successfully
    expectIntKnobMocks(_fakeDesc1, "knob.one");

    // Second knob fails — knob ID query returns empty string → triggers "empty knob ID" error
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _fakeDesc2, HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{0}), Return(HIPDNN_STATUS_SUCCESS)));

    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::vector<Knob> knobs;
    auto err = unpackKnobsFromDescriptors(_fakeEngine, knobs);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_NE(err.get_message().find("Loaded 1 knobs, skipped 1 invalid/duplicate knobs"),
              std::string::npos);
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knobId(), "knob.one");
}

TEST_F(TestUnpackKnobsFromDescriptors, DuplicateKnobIdsAreSkipped)
{
    expectKnobDescArrayQuery({_fakeDesc1, _fakeDesc2});

    expectIntKnobMocks(_fakeDesc1, "knob.duplicate");
    expectIntKnobMocks(_fakeDesc2, "knob.duplicate");

    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_fakeDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::vector<Knob> knobs;
    auto err = unpackKnobsFromDescriptors(_fakeEngine, knobs);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_NE(err.get_message().find("Loaded 1 knobs, skipped 1 invalid/duplicate knobs"),
              std::string::npos);
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knobId(), "knob.duplicate");
}

} // anonymous namespace
