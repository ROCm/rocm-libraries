// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <atomic>
#include <cstdint>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#include <hipdnn_frontend/detail/HipdnnDynamicBackendWrapper.hpp>
#include <hipdnn_frontend/detail/IncompatibleBackend.hpp>
#include <hipdnn_frontend/version.h>

#ifndef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND
#include <hipdnn_frontend/detail/HipdnnDirectBackendWrapper.hpp>
#endif

namespace
{
using hipdnn_data_sdk::utilities::Version;

Version frontendVersion()
{
    return {HIPDNN_FRONTEND_VERSION_MAJOR,
            HIPDNN_FRONTEND_VERSION_MINOR,
            HIPDNN_FRONTEND_VERSION_PATCH};
}

template <typename Backend>
void expectAdditionalBackendApiForwardsToBackend(Backend& backend)
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

    EXPECT_STREQ(backend.getErrorString(HIPDNN_STATUS_SUCCESS),
                 hipdnnGetErrorString(HIPDNN_STATUS_SUCCESS));
    std::array<char, 128> message{};
    backend.getLastErrorString(message.data(), message.size());

    EXPECT_EQ(backend.backendCreateAndDeserializeGraphExt(nullptr, nullptr, 0),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    size_t byteSize = 0;
    EXPECT_EQ(backend.backendGetSerializedBinaryGraphExt(nullptr, 0, &byteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendGetSerializedJsonGraphExt(nullptr, 0, &byteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.backendCreateAndDeserializeJsonGraphExt(nullptr, nullptr, 0),
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

    EXPECT_EQ(backend.setEnginePluginPathsExt(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    EXPECT_EQ(backend.setHeuristicPluginPathsExt(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    size_t count = 0;
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

    EXPECT_EQ(
        backend.setUserLogCallbackExt(nullptr, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC, nullptr),
        HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(backend.backendSetGlobalLogLevelExt(static_cast<hipdnnSeverity_t>(999)),
              HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(backend.backendGetGlobalLogLevelExt(nullptr), HIPDNN_STATUS_BAD_PARAM);
}

void expectIncompatibleBackendStatus(hipdnnStatus_t status)
{
    EXPECT_EQ(status, HIPDNN_STATUS_NOT_INITIALIZED);
}

} // namespace

#ifndef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND
TEST(IntegrationBackendWrapperDetail, DirectBackendWrapperAdditionalApiForwardsToBackend)
{
    SKIP_IF_NO_DEVICES();

    hipdnn_frontend::detail::HipdnnDirectBackendWrapper backend(frontendVersion());
    expectAdditionalBackendApiForwardsToBackend(backend);
}
#endif

TEST(IntegrationBackendWrapperDetail, DynamicBackendSymbolResolutionCachesLoadedSymbols)
{
    if(hipdnn_frontend::detail::backendLibraryHandle() == nullptr)
    {
        GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
    }

    std::atomic<void*> cache{nullptr};
    auto resolved = hipdnn_frontend::detail::resolveBackendSymbol<decltype(&hipdnnGetErrorString)>(
        cache, "hipdnnGetErrorString");

    ASSERT_NE(resolved, nullptr);
    EXPECT_NE(cache.load(std::memory_order_acquire), nullptr);
    EXPECT_STREQ(resolved(HIPDNN_STATUS_SUCCESS), hipdnnGetErrorString(HIPDNN_STATUS_SUCCESS));

    auto cached = hipdnn_frontend::detail::resolveBackendSymbol<decltype(&hipdnnGetErrorString)>(
        cache, "hipdnnMissingSymbolForCacheHit");
    EXPECT_EQ(cached, resolved);

    std::atomic<void*> missingCache{nullptr};
    auto missing = hipdnn_frontend::detail::resolveBackendSymbol<decltype(&hipdnnGetErrorString)>(
        missingCache, "hipdnnMissingSymbolForCacheMiss");
    EXPECT_EQ(missing, nullptr);
    EXPECT_EQ(missingCache.load(std::memory_order_acquire), nullptr);
}

TEST(IntegrationBackendWrapperDetail, DynamicBackendWrapperAdditionalApiForwardsToBackend)
{
    SKIP_IF_NO_DEVICES();

    if(hipdnn_frontend::detail::backendLibraryHandle() == nullptr)
    {
        GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
    }

    hipdnn_frontend::detail::HipdnnDynamicBackendWrapper backend(frontendVersion());
    expectAdditionalBackendApiForwardsToBackend(backend);
}

TEST(IntegrationBackendWrapperDetail, IncompatibleBackendAdditionalApiReturnsNotInitialized)
{
    hipdnn_frontend::detail::IncompatibleBackendWrapper backend;

    size_t byteSize = 0;
    expectIncompatibleBackendStatus(
        backend.backendGetSerializedBinaryGraphExt(nullptr, 0, &byteSize, nullptr));
    expectIncompatibleBackendStatus(
        backend.backendGetSerializedJsonGraphExt(nullptr, 0, &byteSize, nullptr));
    expectIncompatibleBackendStatus(
        backend.backendCreateAndDeserializeJsonGraphExt(nullptr, nullptr, 0));
    expectIncompatibleBackendStatus(
        backend.backendGetSerializedBinaryGraphAndPlanExt(nullptr, nullptr, 0, &byteSize, nullptr));

    int contentFlags = 0;
    expectIncompatibleBackendStatus(
        backend.backendGetSerializedBinaryContentsExt(nullptr, 0, &contentFlags));

    expectIncompatibleBackendStatus(
        backend.setHeuristicPluginPathsExt(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    size_t count = 0;
    expectIncompatibleBackendStatus(
        backend.getLoadedEnginePluginPathsExt(nullptr, &count, nullptr, &byteSize));
    expectIncompatibleBackendStatus(backend.getHeuristicPolicyCount(nullptr, &count));

    int64_t policyId = 0;
    size_t policyNameLen = 0;
    size_t pluginNameLen = 0;
    size_t pluginVersionLen = 0;
    size_t apiVersionLen = 0;
    expectIncompatibleBackendStatus(backend.getHeuristicPolicyInfo(nullptr,
                                                                   0,
                                                                   &policyId,
                                                                   nullptr,
                                                                   &policyNameLen,
                                                                   nullptr,
                                                                   &pluginNameLen,
                                                                   nullptr,
                                                                   &pluginVersionLen,
                                                                   nullptr,
                                                                   &apiVersionLen));
    expectIncompatibleBackendStatus(backend.setUserLogCallbackExt(
        nullptr, HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_ASYNC, nullptr));
    expectIncompatibleBackendStatus(backend.backendSetGlobalLogLevelExt(HIPDNN_SEV_INFO));
    expectIncompatibleBackendStatus(backend.backendGetGlobalLogLevelExt(nullptr));
}
