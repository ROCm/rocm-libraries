/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <filesystem>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Keep the ingestor's winner cache out of the developer's ~/.cache/hipdnn: the
    // dispatch cases benchmark, and benchmarking writes a shard through to disk.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir("hip-kernel-provider-unit");

#ifdef HIPDNN_TEST_DESCRIPTOR_DIR
    // Point this binary at the descriptors staged beside the build's plugin. The engine
    // implementation is linked in statically here, so its module-relative lookup has no
    // module to measure from and falls through to the install prefix, which a build tree
    // has never written. Done here rather than in the CTest environment so the binary
    // runs standalone -- and so a path from this machine stays out of the install-time
    // CTest file, which is generated from that same environment.
    //
    // Never overrides a value the caller set, and never sets one naming nothing: on an
    // installed run the staged tree is absent and resolution should reach the installed
    // copy instead.
    if(std::error_code notFound;
       hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR").empty()
       && std::filesystem::is_directory(HIPDNN_TEST_DESCRIPTOR_DIR, notFound))
    {
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_DESCRIPTOR_DIR", HIPDNN_TEST_DESCRIPTOR_DIR);
    }
#endif

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    // NOTE: Logs are not routed to the backend by the recordingCallback returned here
    // which is the desired behaviour because this is a plugin unit test harness.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    // Initialize plugin logger with test recording callback so that plugin logs
    // logs are first routed to the log recorder for capture and use by the unit tests.
    hipdnn_plugin_sdk::logging::initializeCallbackLogging("hip_kernel-provider_tests",
                                                          recordingCallback);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
