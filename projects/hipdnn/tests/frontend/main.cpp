/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <array>
#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <test_plugins/TestPluginConstants.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize test log recording infrastructure to always forward logs to hipdnnLoggingCallback_ext().
    // NOTE: The frontend logger must be initialized with recordingCallback to ensure the logs
    // are recorded and forwarded to the hipdnnLoggingCallback_ext() function.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeChainedTestLogRecordingShared(
        hipdnnLoggingCallback_ext);

    // Initialize frontend logging with test recording callback returned above.
    // This must be called to override the default callback that would otherwise
    // be used by HIPDNN_FE_LOG_* macros with lazy-initialization.
    hipdnn_frontend::initializeFrontendLogging(recordingCallback);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    // Frontend integration tests don't exercise heuristic-plugin behavior; they just
    // need a generic working heuristic so engine selection succeeds. Wire the
    // engine-agnostic test_good_heuristic_plugin in once before any test runs (no
    // active handles allowed when changing heuristic plugin paths).
    const std::array<const char*, 1> heuristicPaths
        = {hipdnn_tests::plugin_constants::testGoodHeuristicPluginPath().c_str()};
    if(hipdnnSetHeuristicPluginPaths_ext(
           heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE)
       != HIPDNN_STATUS_SUCCESS)
    {
        return 1;
    }
    hipdnn_data_sdk::utilities::setEnv(
        "HIPDNN_HEUR_POLICY_ORDER", hipdnn_tests::plugin_constants::testGoodHeuristicPolicyName());

    auto result = RUN_ALL_TESTS();
    return result;
}
