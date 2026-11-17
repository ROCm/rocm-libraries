// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>

#include <array>
#include <cstdint>
#include <string_view>

namespace hipdnn_tests::backend_wrapper
{

template <typename Backend>
void expectVersionMatchesBackend(Backend& backend, const char* expectedVersionString)
{
    EXPECT_STREQ(backend.versionString(), expectedVersionString);
    EXPECT_EQ(backend.version(),
              hipdnn_data_sdk::utilities::Version{std::string_view(expectedVersionString)});
}

template <typename Backend>
void expectHandleLifecycleForwardsToBackend(Backend& backend)
{
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(backend.create(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(backend.destroy(handle), HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(backend.create(nullptr), HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.destroy(nullptr), HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.setStream(nullptr, nullptr), HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipStream_t stream = nullptr;
    EXPECT_EQ(backend.getStream(nullptr, &stream), HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

template <typename Backend>
void expectDescriptorApiForwardsToBackend(Backend& backend, const char* expectedSuccessString)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(backend.backendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    ASSERT_EQ(
        backend.backendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(descriptor, nullptr);

    EXPECT_EQ(backend.backendExecute(nullptr, nullptr, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendFinalize(nullptr), HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendGetAttribute(nullptr,
                                          static_cast<hipdnnBackendAttributeName_t>(0),
                                          static_cast<hipdnnBackendAttributeType_t>(0),
                                          0,
                                          nullptr,
                                          nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendSetAttribute(nullptr,
                                          static_cast<hipdnnBackendAttributeName_t>(0),
                                          static_cast<hipdnnBackendAttributeType_t>(0),
                                          0,
                                          nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);

    EXPECT_STREQ(backend.getErrorString(HIPDNN_STATUS_SUCCESS), expectedSuccessString);
    std::array<char, 128> message{};
    backend.getLastErrorString(message.data(), message.size());
}

template <typename Backend>
void expectSerializationApiForwardsToBackend(Backend& backend)
{
    EXPECT_EQ(backend.backendCreateAndDeserializeGraphExt(nullptr, nullptr, 0),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    size_t byteSize = 0;
    EXPECT_EQ(backend.backendGetSerializedBinaryGraphExt(nullptr, 0, &byteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendGetSerializedJsonGraphExt(nullptr, 0, &byteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendCreateAndDeserializeJsonGraphExt(nullptr, nullptr, 0),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendGetSerializedExecutionPlanExt(nullptr, 0, &byteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    const std::array<uint8_t, 1> serializedPlan{0};
    EXPECT_EQ(backend.backendCreateAndDeserializeExecutionPlanExt(
                  nullptr, &descriptor, serializedPlan.data(), serializedPlan.size()),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(
        backend.backendGetSerializedBinaryGraphAndPlanExt(nullptr, nullptr, 0, &byteSize, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    int contentFlags = 0;
    const uint8_t blob = 0;
    EXPECT_EQ(backend.backendGetSerializedBinaryContentsExt(nullptr, 0, &contentFlags),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendGetSerializedBinaryContentsExt(&blob, 1, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendGetSerializedBinaryContentsExt(&blob, 1, &contentFlags),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(contentFlags, HIPDNN_SERIALIZED_CONTENT_GRAPH);
}

template <typename Backend>
void expectPluginAndHeuristicApiForwardsToBackend(Backend& backend)
{
    EXPECT_EQ(backend.setEnginePluginPathsExt(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.setHeuristicPluginPathsExt(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.setEnginePluginPathsExt(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(backend.setHeuristicPluginPathsExt(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE),
              HIPDNN_STATUS_SUCCESS);

    size_t count = 0;
    size_t byteSize = 0;
    EXPECT_EQ(backend.getLoadedEnginePluginPathsExt(nullptr, &count, nullptr, &byteSize),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.getHeuristicPolicyCount(nullptr, &count),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    int64_t policyId = 0;
    size_t policyNameLen = 0;
    size_t pluginNameLen = 0;
    size_t pluginVersionLen = 0;
    size_t apiVersionLen = 0;
    EXPECT_EQ(backend.getHeuristicPolicyInfo(nullptr,
                                             0,
                                             &policyId,
                                             nullptr,
                                             &policyNameLen,
                                             nullptr,
                                             &pluginNameLen,
                                             nullptr,
                                             &pluginVersionLen,
                                             nullptr,
                                             &apiVersionLen),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

template <typename Backend>
void expectLoggingApiForwardsToBackend(Backend& backend)
{
    EXPECT_EQ(
        backend.setUserLogCallbackExt(nullptr, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC, nullptr),
        HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(backend.backendSetGlobalLogLevelExt(static_cast<hipdnnSeverity_t>(999)),
              HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(backend.backendGetGlobalLogLevelExt(nullptr), HIPDNN_STATUS_BAD_PARAM);
}

} // namespace hipdnn_tests::backend_wrapper
