// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/IncompatibleBackend.hpp>

#include <array>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

TEST(TestIncompatibleBackendWrapper, AllFunctionsReturnFailed)
{
    const IncompatibleBackendWrapper backendWrapper;
}

TEST(TestIncompatibleBackendWrapper, Create)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnHandle_t handle = nullptr;
    EXPECT_EQ(backendWrapper.create(&handle), hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, Destroy)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnHandle_t handle = nullptr;
    EXPECT_EQ(backendWrapper.destroy(handle), hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, SetStream)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnHandle_t handle = nullptr;
    hipStream_t streamId = nullptr;
    EXPECT_EQ(backendWrapper.setStream(handle, streamId),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, GetStream)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnHandle_t handle = nullptr;
    hipStream_t streamId = nullptr;
    EXPECT_EQ(backendWrapper.getStream(handle, &streamId),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendCreateDescriptor)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(backendWrapper.backendCreateDescriptor(
                  hipdnnBackendDescriptorType_t::HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &descriptor),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendDestroyDescriptor)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(backendWrapper.backendDestroyDescriptor(descriptor),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendExecute)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnHandle_t handle = nullptr;
    hipdnnBackendDescriptor_t executionPlan = nullptr;
    hipdnnBackendDescriptor_t variantPack = nullptr;
    EXPECT_EQ(backendWrapper.backendExecute(handle, executionPlan, variantPack),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendFinalize)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(backendWrapper.backendFinalize(descriptor),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendGetAttribute)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnBackendDescriptor_t descriptor = nullptr;
    int64_t elementCount = 0;
    void* arrayOfElements = nullptr;
    EXPECT_EQ(
        backendWrapper.backendGetAttribute(descriptor,
                                           hipdnnBackendAttributeName_t::HIPDNN_ATTR_KNOB_INFO_TYPE,
                                           hipdnnBackendAttributeType_t::HIPDNN_TYPE_DATA_TYPE,
                                           1,
                                           &elementCount,
                                           arrayOfElements),
        hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendSetAttribute)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnBackendDescriptor_t descriptor = nullptr;
    int64_t value = 0;
    EXPECT_EQ(
        backendWrapper.backendSetAttribute(descriptor,
                                           hipdnnBackendAttributeName_t::HIPDNN_ATTR_KNOB_INFO_TYPE,
                                           hipdnnBackendAttributeType_t::HIPDNN_TYPE_DATA_TYPE,
                                           1,
                                           &value),
        hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendCreateAndDeserializeGraphExt)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnBackendDescriptor_t descriptor;
    const uint8_t* serializedGraph = nullptr;
    EXPECT_EQ(backendWrapper.backendCreateAndDeserializeGraphExt(&descriptor, serializedGraph, 0),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendGetSerializedExecutionPlanExt)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnBackendDescriptor_t descriptor = nullptr;
    size_t planByteSize = 0;
    EXPECT_EQ(
        backendWrapper.backendGetSerializedExecutionPlanExt(descriptor, 0, &planByteSize, nullptr),
        hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, BackendCreateAndDeserializeExecutionPlanExt)
{
    IncompatibleBackendWrapper backendWrapper;
    hipdnnHandle_t handle = nullptr;
    hipdnnBackendDescriptor_t descriptor = nullptr;
    const std::array<uint8_t, 1> serializedPlan{0};
    EXPECT_EQ(backendWrapper.backendCreateAndDeserializeExecutionPlanExt(
                  handle, &descriptor, serializedPlan.data(), serializedPlan.size()),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, VersionString)
{
    IncompatibleBackendWrapper backendWrapper;
    EXPECT_STREQ(backendWrapper.versionString(), "");
}

TEST(TestIncompatibleBackendWrapper, Version)
{
    IncompatibleBackendWrapper backendWrapper;
    EXPECT_EQ(backendWrapper.version(), hipdnn_data_sdk::utilities::Version(-1, 0, 0));
}

TEST(TestIncompatibleBackendWrapper, SetEnginePluginPathsExt)
{
    IncompatibleBackendWrapper backendWrapper;
    std::array<const char*, 2> paths = {"path1", "path2"};
    EXPECT_EQ(backendWrapper.setEnginePluginPathsExt(
                  paths.size(),
                  paths.data(),
                  hipdnnPluginLoadingMode_ext_t::HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST(TestIncompatibleBackendWrapper, GetErrorString)
{
    IncompatibleBackendWrapper backendWrapper;
    const char* errorString = backendWrapper.getErrorString(hipdnnStatus_t::HIPDNN_STATUS_SUCCESS);
    EXPECT_STREQ(errorString, "");
}

TEST(TestIncompatibleBackendWrapper, GetLastErrorString)
{
    IncompatibleBackendWrapper backendWrapper;
    std::string message = "initial";
    // getLastErrorString is a void function that does nothing
    backendWrapper.getLastErrorString(message.data(), message.size());
    // Message should remain unchanged since the function does nothing
    EXPECT_EQ(message, std::string{"initial"});
}

TEST(TestIncompatibleBackendWrapper, LoggingCallbackExt)
{
    IncompatibleBackendWrapper backendWrapper;
    // loggingCallbackExt is a void function that does nothing
    // Just verify it doesn't crash
    backendWrapper.loggingCallbackExt(hipdnnSeverity_t::HIPDNN_SEV_INFO, "test message");
}
