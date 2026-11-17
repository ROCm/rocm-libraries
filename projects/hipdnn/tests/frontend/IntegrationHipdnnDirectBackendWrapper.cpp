// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BackendWrapperForwardingTest.hpp"

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_frontend/detail/HipdnnDirectBackendWrapper.hpp>

#include <string_view>

namespace
{

class IntegrationHipdnnDirectBackendWrapper : public testing::Test
{
protected:
    static hipdnn_frontend::detail::HipdnnDirectBackendWrapper makeWrapper()
    {
        return {hipdnn_data_sdk::utilities::Version{std::string_view(hipdnnVersionString_ext())}};
    }

    static const char* successString()
    {
        return hipdnnGetErrorString(HIPDNN_STATUS_SUCCESS);
    }
};

} // namespace

TEST_F(IntegrationHipdnnDirectBackendWrapper, VersionStringMatchesBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectVersionMatchesBackend(backend, hipdnnVersionString_ext());
}

TEST_F(IntegrationHipdnnDirectBackendWrapper, HandleLifecycleForwardsToBackend)
{
    SKIP_IF_NO_DEVICES();

    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectHandleLifecycleForwardsToBackend(backend);
}

TEST_F(IntegrationHipdnnDirectBackendWrapper, DescriptorApiForwardsToBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectDescriptorApiForwardsToBackend(backend, successString());
}

TEST_F(IntegrationHipdnnDirectBackendWrapper, SerializationApiForwardsToBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectSerializationApiForwardsToBackend(backend);
}

TEST_F(IntegrationHipdnnDirectBackendWrapper, PluginAndHeuristicApiForwardsToBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectPluginAndHeuristicApiForwardsToBackend(backend);
}

TEST_F(IntegrationHipdnnDirectBackendWrapper, LoggingApiForwardsToBackend)
{
    auto backend = makeWrapper();
    hipdnn_tests::backend_wrapper::expectLoggingApiForwardsToBackend(backend);
}
