// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/KnobPacker.hpp>
#include <hipdnn_frontend/detail/KnobSettingUnpacker.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include "fake_backend/BackendTestMatchers.hpp"
#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
using namespace hipdnn_frontend::test;
using namespace ::testing;

namespace
{

class TestKnobSettingUnpacker : public ::testing::Test
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
};

// ============================================================================
// unpackKnobSettingDescriptor tests
// ============================================================================

TEST_F(TestKnobSettingUnpacker, UnpackIntKnobSetting)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);

    // Mock reading knob ID string
    // First call: size query (count only)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(14), Return(HIPDNN_STATUS_SUCCESS)));

    // Second call: actual data
    const std::string knobId = "test.int_knob";
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    14,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            SetArgPointee<4>(14),
            Invoke([&knobId](hipdnnBackendDescriptor_t,
                             hipdnnBackendAttributeName_t,
                             hipdnnBackendAttributeType_t,
                             int64_t,
                             int64_t*,
                             void* out) {
                std::memcpy(out, knobId.c_str(), knobId.size() + 1);
            }),
            Return(HIPDNN_STATUS_SUCCESS)));

    // Mock reading value as int64 (first attempt succeeds)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            Invoke([](hipdnnBackendDescriptor_t,
                      hipdnnBackendAttributeName_t,
                      hipdnnBackendAttributeType_t,
                      int64_t,
                      int64_t*,
                      void* out) { *static_cast<int64_t*>(out) = 42; }),
            Return(HIPDNN_STATUS_SUCCESS)));

    KnobSetting setting("", KnobValueVariant{int64_t{0}});
    auto err = unpackKnobSettingDescriptor(fakeDesc, setting);

    ASSERT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(setting.knobId(), knobId);

    auto* intVal = std::get_if<int64_t>(&setting.value());
    ASSERT_NE(intVal, nullptr);
    EXPECT_EQ(*intVal, 42);
}

TEST_F(TestKnobSettingUnpacker, UnpackDoubleKnobSetting)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);

    // Mock reading knob ID
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(16), Return(HIPDNN_STATUS_SUCCESS)));

    const std::string knobId = "test.float_knob";
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    16,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            SetArgPointee<4>(16),
            Invoke([&knobId](hipdnnBackendDescriptor_t,
                             hipdnnBackendAttributeName_t,
                             hipdnnBackendAttributeType_t,
                             int64_t,
                             int64_t*,
                             void* out) {
                std::memcpy(out, knobId.c_str(), knobId.size() + 1);
            }),
            Return(HIPDNN_STATUS_SUCCESS)));

    // Int64 attempt fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    // Double attempt succeeds
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_DOUBLE,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            Invoke([](hipdnnBackendDescriptor_t,
                      hipdnnBackendAttributeName_t,
                      hipdnnBackendAttributeType_t,
                      int64_t,
                      int64_t*,
                      void* out) { *static_cast<double*>(out) = 3.14; }),
            Return(HIPDNN_STATUS_SUCCESS)));

    KnobSetting setting("", KnobValueVariant{int64_t{0}});
    auto err = unpackKnobSettingDescriptor(fakeDesc, setting);

    ASSERT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(setting.knobId(), knobId);

    auto* doubleVal = std::get_if<double>(&setting.value());
    ASSERT_NE(doubleVal, nullptr);
    EXPECT_DOUBLE_EQ(*doubleVal, 3.14);
}

TEST_F(TestKnobSettingUnpacker, UnpackStringKnobSetting)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);

    // Mock reading knob ID
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(17), Return(HIPDNN_STATUS_SUCCESS)));

    const std::string knobId = "test.string_knob";
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    17,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            SetArgPointee<4>(17),
            Invoke([&knobId](hipdnnBackendDescriptor_t,
                             hipdnnBackendAttributeName_t,
                             hipdnnBackendAttributeType_t,
                             int64_t,
                             int64_t*,
                             void* out) {
                std::memcpy(out, knobId.c_str(), knobId.size() + 1);
            }),
            Return(HIPDNN_STATUS_SUCCESS)));

    // Int64 attempt fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    // Double attempt fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_DOUBLE,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    // String attempt: size query
    const std::string expectedValue = "accurate";
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(expectedValue.size() + 1)),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // String attempt: read data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(expectedValue.size() + 1),
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            SetArgPointee<4>(static_cast<int64_t>(expectedValue.size() + 1)),
            Invoke([&expectedValue](hipdnnBackendDescriptor_t,
                                    hipdnnBackendAttributeName_t,
                                    hipdnnBackendAttributeType_t,
                                    int64_t,
                                    int64_t*,
                                    void* out) {
                std::memcpy(out, expectedValue.c_str(), expectedValue.size() + 1);
            }),
            Return(HIPDNN_STATUS_SUCCESS)));

    KnobSetting setting("", KnobValueVariant{int64_t{0}});
    auto err = unpackKnobSettingDescriptor(fakeDesc, setting);

    ASSERT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(setting.knobId(), knobId);

    auto* strVal = std::get_if<std::string>(&setting.value());
    ASSERT_NE(strVal, nullptr);
    EXPECT_EQ(*strVal, expectedValue);
}

TEST_F(TestKnobSettingUnpacker, UnpackFailsWithEmptyKnobId)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);

    // Mock reading empty knob ID: size query returns 0
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));

    KnobSetting setting("", KnobValueVariant{int64_t{0}});
    auto err = unpackKnobSettingDescriptor(fakeDesc, setting);

    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("empty knob ID"), std::string::npos);
}

TEST_F(TestKnobSettingUnpacker, UnpackFailsWhenAllValueTypesUnsupported)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);

    // Mock reading knob ID
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(10), Return(HIPDNN_STATUS_SUCCESS)));

    const std::string knobId = "bad.knob";
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    10,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(
            SetArgPointee<4>(10),
            Invoke([&knobId](hipdnnBackendDescriptor_t,
                             hipdnnBackendAttributeName_t,
                             hipdnnBackendAttributeType_t,
                             int64_t,
                             int64_t*,
                             void* out) {
                std::memcpy(out, knobId.c_str(), knobId.size() + 1);
            }),
            Return(HIPDNN_STATUS_SUCCESS)));

    // All value type reads fail
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_INT64,
                                    _,
                                    _,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_DOUBLE,
                                    _,
                                    _,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    // String size query fails
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(fakeDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    // getLastErrorString for the error message
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    KnobSetting setting("", KnobValueVariant{int64_t{0}});
    auto err = unpackKnobSettingDescriptor(fakeDesc, setting);

    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("Failed to read knob value"), std::string::npos);
}

} // anonymous namespace
