// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnBackendFlatbufferData.h"
#include "TestMacros.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/KnobSettingDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockDescriptor.hpp"
#include "mocks/MockEnginePluginResourceManager.hpp"
#include "mocks/MockHandle.hpp"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>

#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
using namespace hipdnn_backend::test_utilities;
using namespace ::testing;

using ::testing::Return;

class TestEngineConfigDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<EngineConfigDescriptor> getEngineConfigDescriptor() const
    {
        return _engineConfigWrapper->asDescriptor<EngineConfigDescriptor>();
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngine() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineWrapper.get());
    }

    std::shared_ptr<MockEngineDescriptor> getMockEngineBadType() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockEngineDescriptor>(
            _mockEngineBadTypeWrapper.get());
    }

    std::shared_ptr<MockGraphDescriptor> getMockGraphDescriptor() const
    {
        return MockDescriptorUtility::asDescriptorUnsafe<MockGraphDescriptor>(
            _mockGraphWrapper.get());
    }

    void setEngine() const
    {
        EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(getEngineConfigDescriptor()->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));
    }

    void makeEngineConfigFinalized() const
    {
        // TODO: These expectations being hidden in here are dangerous.
        // It's easy to forget they exist, and if any of these functions are called prior to this
        // call it is undefined behavior. Similarly, any expectations on functions called within
        // "finalize" or "setup" made after calling this function are UB
        EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));
        EXPECT_CALL(*getMockEngine(), getGraph()).WillRepeatedly(Return(getMockGraphDescriptor()));
        EXPECT_CALL(*getMockGraphDescriptor(), getHandle()).WillOnce(Return(_mockHandle.get()));
        EXPECT_CALL(*_mockHandle, getPluginResourceManager())
            .WillOnce(Return(_mockEnginePluginResourceManager));
        EXPECT_CALL(*_mockEnginePluginResourceManager, getWorkspaceSize(_, _, _))
            .WillOnce(Return(1024));

        setEngine();
        ASSERT_NO_THROW(getEngineConfigDescriptor()->finalize());
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _engineConfigWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockEngineBadTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockWrongTypeWrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _mockGraphWrapper = nullptr;
    std::unique_ptr<MockHandle> _mockHandle = nullptr;
    std::shared_ptr<MockEnginePluginResourceManager> _mockEnginePluginResourceManager = nullptr;

    void SetUp() override
    {
        _engineConfigWrapper = createDescriptor<EngineConfigDescriptor>();
        _mockEngineWrapper = createDescriptor<MockEngineDescriptor>();
        _mockEngineBadTypeWrapper = createDescriptor<MockEngineDescriptor>();
        _mockWrongTypeWrapper = createDescriptor<MockDescriptor<EngineConfigDescriptor>>();
        _mockGraphWrapper = createDescriptor<MockGraphDescriptor>();
        _mockHandle = std::make_unique<MockHandle>();
        _mockEnginePluginResourceManager = std::make_shared<MockEnginePluginResourceManager>();
    }
};

TEST_F(TestEngineConfigDescriptor, CreateEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_NE(engineConfig, nullptr);
    ASSERT_FALSE(engineConfig->isFinalized());
    ASSERT_EQ(engineConfig->getType(), HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
}

TEST_F(TestEngineConfigDescriptor, SetEngineConfigDescriptorEngine)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngineBadType(), isFinalized()).Times(1);
    EXPECT_CALL(*getMockEngine(), getEngineId()).Times(AnyNumber());
    EXPECT_CALL(*getMockEngine(), isFinalized()).WillOnce(Return(false)).WillOnce(Return(true));

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_mockEngineWrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engine = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mockEngineBadTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mockWrongTypeWrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetAttrOnFinalizedEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, FinalizeEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->finalize(), HIPDNN_STATUS_BAD_PARAM);

    makeEngineConfigFinalized();
}

TEST_F(TestEngineConfigDescriptor, GetAttrOnUnfinalizedEngineConfigDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnBackendDescriptor_t dummyEngine = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &dummyEngine),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestEngineConfigDescriptor, GetEngineConfigDescriptorUnsupportedAttr)
{
    auto engineConfig = getEngineConfigDescriptor();
    hipdnnBackendDescriptor_t dummy = nullptr;

    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          nullptr,
                                                          &dummy),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestEngineConfigDescriptor, GetEngineConfigDescriptorEngine)
{
    auto engineConfig = getEngineConfigDescriptor();
    ScopedDescriptor engine;
    ScopedDescriptor engine2;
    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr, engine.getPtr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          2,
                                                          nullptr,
                                                          engine.getPtr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               1,
                                               nullptr,
                                               static_cast<void*>(engine.getPtr())));
    ASSERT_EQ(*engine.get(), *(_mockEngineWrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               1,
                                               &count,
                                               static_cast<void*>(engine2.getPtr())));
    ASSERT_EQ(count, 1);
}

TEST_F(TestEngineConfigDescriptor, GetEngineThrowsIfNotFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getEngine(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestEngineConfigDescriptor, GetEngineReturnsPointerIfFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();
    auto enginePtr = engineConfig->getEngine();
    ASSERT_NE(enginePtr, nullptr);
    ASSERT_EQ(static_cast<const IBackendDescriptor*>(enginePtr.get()),
              static_cast<const IBackendDescriptor*>(getMockEngine().get()));
}

TEST_F(TestEngineConfigDescriptor, GetEngineDescriptorMaxWorkspaceSize)
{
    auto engineConfig = getEngineConfigDescriptor();
    int64_t workspaceSize = 0;

    makeEngineConfigFinalized();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->getAttribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          nullptr,
                                                          &workspaceSize),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 2, nullptr, &workspaceSize),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->getAttribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspaceSize));
    ASSERT_EQ(workspaceSize, 1024);

    int64_t count;
    ASSERT_NO_THROW(engineConfig->getAttribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &count, &workspaceSize));
    ASSERT_EQ(count, 1);
}

// Helper function to create a serialized KnobSetting
static flatbuffers::DetachedBuffer createSerializedKnobSetting(const std::string& knobId,
                                                               int64_t value)
{
    flatbuffers::FlatBufferBuilder builder;
    auto knobIdOffset = builder.CreateString(knobId);
    auto intValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(builder, value);
    auto knobSetting = hipdnn_flatbuffers_sdk::data_objects::CreateKnobSetting(
        builder,
        knobIdOffset,
        hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue,
        intValue.Union());
    builder.Finish(knobSetting);
    return builder.Release();
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceInvalidType)
{
    auto engineConfig = getEngineConfigDescriptor();

    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    // Wrong attribute type
    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE, HIPDNN_TYPE_INT64, 1, &knobData),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceInvalidCount)
{
    auto engineConfig = getEngineConfigDescriptor();

    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    // Element count < 1
    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          0,
                                                          &knobData),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceNullPointer)
{
    auto engineConfig = getEngineConfigDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          1,
                                                          nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceNullFlatbufferPointer)
{
    auto engineConfig = getEngineConfigDescriptor();

    hipdnnBackendFlatbufferData_t knobData = {nullptr, 100};

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          1,
                                                          &knobData),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceZeroSize)
{
    auto engineConfig = getEngineConfigDescriptor();

    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), 0};

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          1,
                                                          &knobData),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceSuccess)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    // Set engine first
    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    // Now set a knob choice
    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                               HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                               1,
                                               &knobData));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceMultipleKnobs)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    // Set engine first
    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    // Create multiple knob settings
    auto knobBuffer1 = createSerializedKnobSetting("test_knob_100", 42);
    auto knobBuffer2 = createSerializedKnobSetting("test_knob_101", 84);

    std::vector<hipdnnBackendFlatbufferData_t> knobDataArray
        = {{knobBuffer1.data(), knobBuffer1.size()}, {knobBuffer2.data(), knobBuffer2.size()}};

    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                               HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                               2,
                                               knobDataArray.data()));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceOnFinalizedDescriptor)
{
    auto engineConfig = getEngineConfigDescriptor();
    makeEngineConfigFinalized();

    auto knobBuffer = createSerializedKnobSetting("test_knob_100", 42);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    ASSERT_THROW_HIPDNN_STATUS(engineConfig->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                                          HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                          1,
                                                          &knobData),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

// Helper to create a finalized KnobSettingDescriptor
static std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedKnobSettingDescriptor(const std::string& knobId, int64_t value)
{
    auto wrapper = test_utilities::createDescriptor<KnobSettingDescriptor>();
    auto desc = wrapper->asDescriptor<KnobSettingDescriptor>();
    desc->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(knobId.size()),
                       knobId.c_str());
    desc->setAttribute(HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, &value);
    desc->finalize();
    return wrapper;
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorSuccess)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    auto knobWrapper = createFinalizedKnobSettingDescriptor("test_knob_100", 42);
    auto* knobPtr = knobWrapper.get();

    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               1,
                                               static_cast<const void*>(&knobPtr)));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorMultiple)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    auto knobWrapper1 = createFinalizedKnobSettingDescriptor("knob_1", 10);
    auto knobWrapper2 = createFinalizedKnobSettingDescriptor("knob_2", 20);
    std::array<HipdnnBackendDescriptor*, 2> knobPtrs = {knobWrapper1.get(), knobWrapper2.get()};

    ASSERT_NO_THROW(engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               2,
                                               static_cast<const void*>(knobPtrs.data())));
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorRejectNotFinalized)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    // Create a non-finalized knob descriptor
    auto knobWrapper = test_utilities::createDescriptor<KnobSettingDescriptor>();
    auto* knobPtr = knobWrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &knobPtr),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorRejectWrongType)
{
    auto engineConfig = getEngineConfigDescriptor();

    auto knobWrapper = createFinalizedKnobSettingDescriptor("test_knob", 42);
    auto* knobPtr = knobWrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(
            HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES, HIPDNN_TYPE_INT64, 1, &knobPtr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestEngineConfigDescriptor, SetKnobChoiceViaDescriptorRejectExceedsMaxCount)
{
    auto engineConfig = getEngineConfigDescriptor();

    EXPECT_CALL(*getMockEngine(), isFinalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*getMockEngine(), getEngineId()).WillRepeatedly(Return(1));

    ASSERT_NO_THROW(engineConfig->setAttribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mockEngineWrapper));

    auto knobWrapper = createFinalizedKnobSettingDescriptor("test_knob", 42);
    auto* knobPtr = knobWrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        engineConfig->setAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                   EngineConfigDescriptor::MAX_KNOB_CHOICES + 1,
                                   &knobPtr),
        HIPDNN_STATUS_BAD_PARAM);
}
