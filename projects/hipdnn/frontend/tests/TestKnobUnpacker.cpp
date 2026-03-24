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
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(desc, attrName, attrType, 1, _, NotNull()))
            .WillOnce(DoAll(
                Invoke([value](hipdnnBackendDescriptor_t,
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
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(desc, attrName, attrType, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(1), Return(HIPDNN_STATUS_SUCCESS)));

        // Value read
        mockScalarAttr(desc, attrName, attrType, value);
    }

    // Helper: mock an optional scalar attribute that is absent
    void mockOptionalScalarAbsent(hipdnnBackendDescriptor_t desc,
                                  hipdnnBackendAttributeName_t attrName,
                                  hipdnnBackendAttributeType_t attrType)
    {
        EXPECT_CALL(*_mockBackend,
                    backendGetAttribute(desc, attrName, attrType, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));
    }

    // Helper: mock int64 vector attribute (empty)
    void mockEmptyVecAttr(hipdnnBackendDescriptor_t desc,
                          hipdnnBackendAttributeName_t attrName)
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
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, "test.int_knob");

    // Mock: description
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, "An integer knob");

    // Mock: deprecated flag
    mockScalarAttr<bool>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT, HIPDNN_TYPE_BOOLEAN, false);

    // Mock: default value type = INT64
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE_EXT,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_INT64));

    // Mock: default value = 50
    mockScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT, HIPDNN_TYPE_INT64, 50);

    // Mock: int constraints (min=0, max=100, stride=10)
    mockOptionalScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT, HIPDNN_TYPE_INT64, 0);
    mockOptionalScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT, HIPDNN_TYPE_INT64, 100);
    mockOptionalScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE_EXT, HIPDNN_TYPE_INT64, 10);

    // Mock: valid values (empty)
    mockEmptyVecAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT_EXT);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);

    ASSERT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(knob.knobId(), "test.int_knob");
    EXPECT_EQ(knob.description(), "An integer knob");
    EXPECT_FALSE(knob.isDeprecated());
    EXPECT_EQ(knob.valueType(), KnobValueType::INT64);

    auto* defaultVal = std::get_if<int64_t>(&knob.defaultValue());
    ASSERT_NE(defaultVal, nullptr);
    EXPECT_EQ(*defaultVal, 50);

    // Check constraint
    ASSERT_NE(knob.constraint(), nullptr);
    auto* intConstraint = dynamic_cast<const IntConstraint*>(knob.constraint());
    ASSERT_NE(intConstraint, nullptr);
    EXPECT_EQ(intConstraint->getMinValue(), 0);
    EXPECT_EQ(intConstraint->getMaxValue(), 100);
    EXPECT_EQ(intConstraint->getStep(), 10);
}

TEST_F(TestKnobUnpacker, UnpackFloatKnobWithConstraints)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, "test.float_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, "A float knob");
    mockScalarAttr<bool>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT, HIPDNN_TYPE_BOOLEAN, false);

    // Default value type = DOUBLE
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE_EXT,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_DOUBLE));

    // Default value = 0.5
    mockScalarAttr<double>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT, HIPDNN_TYPE_DOUBLE, 0.5);

    // Float constraints: min=0.0, max=1.0
    mockOptionalScalarAttr<double>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT, HIPDNN_TYPE_DOUBLE, 0.0);
    mockOptionalScalarAttr<double>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT, HIPDNN_TYPE_DOUBLE, 1.0);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);

    ASSERT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(knob.knobId(), "test.float_knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::FLOAT64);

    auto* defaultVal = std::get_if<double>(&knob.defaultValue());
    ASSERT_NE(defaultVal, nullptr);
    EXPECT_DOUBLE_EQ(*defaultVal, 0.5);

    ASSERT_NE(knob.constraint(), nullptr);
    auto* floatConstraint = dynamic_cast<const FloatConstraint*>(knob.constraint());
    ASSERT_NE(floatConstraint, nullptr);
    EXPECT_DOUBLE_EQ(floatConstraint->getMinValue(), 0.0);
    EXPECT_DOUBLE_EQ(floatConstraint->getMaxValue(), 1.0);
}

TEST_F(TestKnobUnpacker, UnpackStringKnobWithValidValues)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, "test.string_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, "A string knob");
    mockScalarAttr<bool>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT, HIPDNN_TYPE_BOOLEAN, false);

    // Default value type = CHAR
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE_EXT,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_CHAR));

    // Default value = "fast"
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT, "fast");

    // String constraints: max length = 20
    mockOptionalScalarAttr<int32_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH_EXT, HIPDNN_TYPE_INT32, 20);

    // Valid values string: "fast\0accurate\0balanced\0"
    const std::string validValBuf = std::string("fast") + '\0' + "accurate" + '\0' + "balanced" + '\0';
    const auto validValBufLen = static_cast<int64_t>(validValBuf.size());

    // Size query
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(validValBufLen), Return(HIPDNN_STATUS_SUCCESS)));

    // Data read
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    validValBufLen,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            SetArgPointee<4>(validValBufLen),
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
    EXPECT_EQ(knob.knobId(), "test.string_knob");
    EXPECT_EQ(knob.valueType(), KnobValueType::STRING);

    auto* defaultVal = std::get_if<std::string>(&knob.defaultValue());
    ASSERT_NE(defaultVal, nullptr);
    EXPECT_EQ(*defaultVal, "fast");

    ASSERT_NE(knob.constraint(), nullptr);
    auto* strConstraint = dynamic_cast<const StringConstraint*>(knob.constraint());
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

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, "test.deprecated_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, "A deprecated knob");

    // Deprecated = true
    mockScalarAttr<bool>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT, HIPDNN_TYPE_BOOLEAN, true);

    // Default value type = INT64
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE_EXT,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_INT64));
    // Default value = 0
    mockScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT, HIPDNN_TYPE_INT64, 0);

    // No constraints
    mockOptionalScalarAbsent(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE_EXT, HIPDNN_TYPE_INT64);
    mockEmptyVecAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT_EXT);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);

    ASSERT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(knob.knobId(), "test.deprecated_knob");
    EXPECT_TRUE(knob.isDeprecated());
}

TEST_F(TestKnobUnpacker, UnpackKnobWithNoConstraintsGetsEmptyConstraint)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, "test.unconstrained");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, "Unconstrained knob");
    mockScalarAttr<bool>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT, HIPDNN_TYPE_BOOLEAN, false);

    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE_EXT,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_INT64));
    mockScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT, HIPDNN_TYPE_INT64, 42);

    // All optional constraints absent
    mockOptionalScalarAbsent(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE_EXT, HIPDNN_TYPE_INT64);
    mockEmptyVecAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT_EXT);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);

    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_NE(knob.constraint(), nullptr);

    auto* emptyConstraint = dynamic_cast<const EmptyConstraint*>(knob.constraint());
    EXPECT_NE(emptyConstraint, nullptr)
        << "Expected EmptyConstraint but got: " << knob.constraint()->toString();
}

TEST_F(TestKnobUnpacker, UnpackFailsWithEmptyKnobId)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    // Mock: knob ID size query returns 0
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_INFO_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);

    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("empty knob ID"), std::string::npos);
}

TEST_F(TestKnobUnpacker, UnpackFailsWithUnknownValueType)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, "test.bad_type");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, "Bad type knob");
    mockScalarAttr<bool>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT, HIPDNN_TYPE_BOOLEAN, false);

    // Default value type = some unknown type (9999)
    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE_EXT,
                            HIPDNN_TYPE_INT64,
                            9999);

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);

    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("unknown default value type"), std::string::npos);
}

TEST_F(TestKnobUnpacker, UnpackIntKnobWithValidValues)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);

    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, "test.valid_values_knob");
    mockStringAttr(fakeDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, "Knob with valid values");
    mockScalarAttr<bool>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT, HIPDNN_TYPE_BOOLEAN, false);

    mockScalarAttr<int64_t>(fakeDesc,
                            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE_EXT,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(HIPDNN_TYPE_INT64));
    mockScalarAttr<int64_t>(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT, HIPDNN_TYPE_INT64, 8);

    // No min/max/stride
    mockOptionalScalarAbsent(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(
        fakeDesc, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT, HIPDNN_TYPE_INT64);
    mockOptionalScalarAbsent(fakeDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE_EXT, HIPDNN_TYPE_INT64);

    // Valid values: {8, 16, 32, 64}
    std::vector<int64_t> validValues = {8, 16, 32, 64};
    // Count query
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT_EXT,
                                    HIPDNN_TYPE_INT64,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(validValues.size())),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Data read
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT_EXT,
                                    HIPDNN_TYPE_INT64,
                                    static_cast<int64_t>(validValues.size()),
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            SetArgPointee<4>(static_cast<int64_t>(validValues.size())),
            Invoke([&validValues](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* out) {
                std::memcpy(out, validValues.data(), validValues.size() * sizeof(int64_t));
            }),
            Return(HIPDNN_STATUS_SUCCESS)));

    auto [err, knob] = unpackKnobDescriptor(fakeDesc);

    ASSERT_TRUE(err.is_good()) << err.get_message();

    ASSERT_NE(knob.constraint(), nullptr);
    auto* intConstraint = dynamic_cast<const IntConstraint*>(knob.constraint());
    ASSERT_NE(intConstraint, nullptr);

    const auto& vv = intConstraint->getValidValues();
    EXPECT_EQ(vv.size(), 4u);
    EXPECT_NE(vv.find(8), vv.end());
    EXPECT_NE(vv.find(16), vv.end());
    EXPECT_NE(vv.find(32), vv.end());
    EXPECT_NE(vv.find(64), vv.end());
}

} // anonymous namespace
