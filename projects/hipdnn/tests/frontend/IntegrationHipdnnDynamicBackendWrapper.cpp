// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BackendWrapperForwardingTest.hpp"

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

} // namespace

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
