// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BackendWrapperForwardingTest.hpp"

#include <array>
#include <atomic>
#include <memory>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_backend.h>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#include <hipdnn_frontend/detail/HipdnnDynamicBackendWrapper.hpp>

namespace
{

class IntegrationHipdnnDynamicBackendWrapper : public testing::Test
{
protected:
    void SetUp() override
    {
        if(hipdnn_frontend::detail::backendLibraryHandle() == nullptr)
        {
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
        }

        _backend = hipdnn_frontend::detail::hipdnnBackend();
        if(_backend->versionString()[0] == '\0')
        {
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
        }
    }

    hipdnn_frontend::detail::HipdnnDynamicBackendWrapper makeWrapper() const
    {
        return hipdnn_frontend::detail::HipdnnDynamicBackendWrapper(_backend->version());
    }

    const char* successString() const
    {
        return _backend->getErrorString(HIPDNN_STATUS_SUCCESS);
    }

    std::shared_ptr<hipdnn_frontend::detail::IHipdnnBackend> _backend;
};

void* missingSymbolResolver(const char* /*unused*/)
{
    return nullptr;
}

} // namespace

TEST(TestHipdnnDynamicBackendWrapper, MissingStatusReturningSymbolsReturnVersionMismatch)
{
    using hipdnn_data_sdk::utilities::Version;
    using hipdnn_frontend::detail::HipdnnDynamicBackendWrapper;

    HipdnnDynamicBackendWrapper backend(Version{1, 0, 0}, missingSymbolResolver);

    EXPECT_EQ(backend.create(nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.destroy(nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.setStream(nullptr, nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.getStream(nullptr, nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(
        backend.backendCreateDescriptor(static_cast<hipdnnBackendDescriptorType_t>(0), nullptr),
        HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendDestroyDescriptor(nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendExecute(nullptr, nullptr, nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendFinalize(nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendGetAttribute(nullptr,
                                          static_cast<hipdnnBackendAttributeName_t>(0),
                                          static_cast<hipdnnBackendAttributeType_t>(0),
                                          0,
                                          nullptr,
                                          nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendSetAttribute(nullptr,
                                          static_cast<hipdnnBackendAttributeName_t>(0),
                                          static_cast<hipdnnBackendAttributeType_t>(0),
                                          0,
                                          nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendCreateAndDeserializeGraphExt(nullptr, nullptr, 0),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendGetSerializedBinaryGraphExt(nullptr, 0, nullptr, nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendGetSerializedJsonGraphExt(nullptr, 0, nullptr, nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendCreateAndDeserializeJsonGraphExt(nullptr, nullptr, 0),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendGetSerializedExecutionPlanExt(nullptr, 0, nullptr, nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendCreateAndDeserializeExecutionPlanExt(nullptr, nullptr, nullptr, 0),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(
        backend.backendGetSerializedBinaryGraphAndPlanExt(nullptr, nullptr, 0, nullptr, nullptr),
        HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendGetSerializedBinaryContentsExt(nullptr, 0, nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.setEnginePluginPathsExt(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.setHeuristicPluginPathsExt(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.getLoadedEnginePluginPathsExt(nullptr, nullptr, nullptr, nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.getHeuristicPolicyCount(nullptr, nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.getHeuristicPolicyInfo(nullptr,
                                             0,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(
        backend.setUserLogCallbackExt(nullptr, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, nullptr),
        HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendSetGlobalLogLevelExt(HIPDNN_SEV_OFF), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.backendGetGlobalLogLevelExt(nullptr), HIPDNN_STATUS_VERSION_MISMATCH);
    EXPECT_EQ(backend.writeEngineRankingResultsExt(nullptr, nullptr, nullptr, 0, nullptr),
              HIPDNN_STATUS_VERSION_MISMATCH);

    EXPECT_STREQ(backend.getErrorString(HIPDNN_STATUS_VERSION_MISMATCH),
                 "HIPDNN_STATUS_VERSION_MISMATCH");
}

TEST(TestHipdnnDynamicBackendWrapper, MissingLastErrorStringClearsBuffer)
{
    using hipdnn_data_sdk::utilities::Version;
    using hipdnn_frontend::detail::HipdnnDynamicBackendWrapper;

    HipdnnDynamicBackendWrapper backend(Version{1, 0, 0}, missingSymbolResolver);
    std::array<char, sizeof("stale error")> message{"stale error"};

    backend.getLastErrorString(message.data(), message.size());

    EXPECT_STREQ(message.data(), "");
}

TEST(TestHipdnnDynamicBackendWrapper, MissingLastErrorStringAllowsNullAndZeroSize)
{
    using hipdnn_data_sdk::utilities::Version;
    using hipdnn_frontend::detail::HipdnnDynamicBackendWrapper;

    HipdnnDynamicBackendWrapper backend(Version{1, 0, 0}, missingSymbolResolver);
    std::array<char, sizeof("unchanged")> message{"unchanged"};

    backend.getLastErrorString(nullptr, 1);
    backend.getLastErrorString(nullptr, 0);
    backend.getLastErrorString(message.data(), 0);

    EXPECT_STREQ(message.data(), "unchanged");
}

TEST_F(IntegrationHipdnnDynamicBackendWrapper, VersionStringMatchesBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectVersionMatchesBackend(backend, _backend->versionString());
}

TEST_F(IntegrationHipdnnDynamicBackendWrapper, HandleLifecycleForwardsToBackend)
{
    SKIP_IF_NO_DEVICES();

    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectHandleLifecycleForwardsToBackend(backend);
}

TEST_F(IntegrationHipdnnDynamicBackendWrapper, DescriptorApiForwardsToBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectDescriptorApiForwardsToBackend(backend, successString());
}

TEST_F(IntegrationHipdnnDynamicBackendWrapper, SerializationApiForwardsToBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectSerializationApiForwardsToBackend(backend);
}

TEST_F(IntegrationHipdnnDynamicBackendWrapper, PluginAndHeuristicApiForwardsToBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectPluginAndHeuristicApiForwardsToBackend(backend);
}

TEST_F(IntegrationHipdnnDynamicBackendWrapper, LoggingApiForwardsToBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectLoggingApiForwardsToBackend(backend);
}

TEST_F(IntegrationHipdnnDynamicBackendWrapper, SymbolResolutionCachesLoadedSymbols)
{
    std::atomic<void*> cache{nullptr};
    auto resolved = hipdnn_frontend::detail::resolveBackendSymbol<decltype(&hipdnnGetErrorString)>(
        cache, "hipdnnGetErrorString");

    ASSERT_NE(resolved, nullptr);
    EXPECT_NE(cache.load(std::memory_order_acquire), nullptr);
    EXPECT_STREQ(resolved(HIPDNN_STATUS_SUCCESS), _backend->getErrorString(HIPDNN_STATUS_SUCCESS));

    auto cached = hipdnn_frontend::detail::resolveBackendSymbol<decltype(&hipdnnGetErrorString)>(
        cache, "hipdnnMissingSymbolForCacheHit");
    EXPECT_EQ(cached, resolved);

    std::atomic<void*> missingCache{nullptr};
    auto missing = hipdnn_frontend::detail::resolveBackendSymbol<decltype(&hipdnnGetErrorString)>(
        missingCache, "hipdnnMissingSymbolForCacheMiss");
    EXPECT_EQ(missing, nullptr);
    EXPECT_EQ(missingCache.load(std::memory_order_acquire), nullptr);
}
