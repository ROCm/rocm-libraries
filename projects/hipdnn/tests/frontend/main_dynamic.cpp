// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <test_plugins/TestPluginConstants.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn_frontend::initializeFrontendLogging();

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    auto backend = hipdnn_frontend::detail::hipdnnBackend();
    if(backend->versionString()[0] != '\0')
    {
        const std::array<std::string, 1> heuristicPaths
            = {hipdnn_tests::plugin_constants::testGoodHeuristicPluginPath()};
        const auto error = hipdnn_frontend::setHeuristicPluginPaths(
            heuristicPaths, hipdnn_frontend::PluginLoadingMode::MODE_ABSOLUTE);
        if(error.is_bad())
        {
            return 1;
        }
    }

    hipdnn_data_sdk::utilities::setEnv(
        "HIPDNN_HEUR_POLICY_ORDER", hipdnn_tests::plugin_constants::testGoodHeuristicPolicyName());

    return RUN_ALL_TESTS();
}
