/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <filesystem>
#include <iostream>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "TestDescriptorRoot.hpp"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Keep the ingestor's winner cache out of the developer's ~/.cache/hipdnn: the
    // dispatch cases benchmark, and benchmarking writes a shard through to disk.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir("hip-kernel-provider-unit");

#ifdef HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR
    // Point this binary at the descriptors staged beside it. The engine implementation is
    // linked in statically here, so its module-relative lookup measures from this
    // executable and would otherwise fall through to the install prefix, which a build
    // tree has never written. Done here rather than in the CTest environment so the binary
    // runs standalone, and so nothing machine-specific reaches the install-time CTest
    // file, which is generated from that same environment.
    //
    // Never overrides a value the caller set. Fail the process when the resolved root
    // holds no descriptor, which is otherwise indistinguishable from a run on a device
    // the descriptors do not cover.
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR").empty())
    {
        const auto descriptors = hip_kernel_provider::testing::descriptorSetRoot(
            HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR);
        const auto unusable
            = hip_kernel_provider::testing::describeUnusableDescriptorRoot(descriptors);
        if(!unusable.empty())
        {
            std::cerr << unusable
                      << ". Build the descriptor staging targets, or set "
                         "HIPDNN_DESCRIPTOR_DIR to a root that holds them.\n";
            return 1;
        }

        hipdnn_data_sdk::utilities::setEnv("HIPDNN_DESCRIPTOR_DIR", descriptors.string().c_str());
    }
#endif

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
    // Add the shipped packs alongside the test root. HIPDNN_DESCRIPTOR_DIR above replaces
    // discovery rather than extending it, so without this the cases that assert what this
    // build ships would see only the test descriptors and skip themselves -- indefinitely,
    // and silently, since an empty census is indistinguishable from a build that packaged
    // nothing.
    //
    // Appended rather than assigned, so a caller-supplied list survives. Done here rather
    // than in a fixture because discoverDescriptorSets() memoizes on first use: a later
    // setter is read only if nothing in the binary touched the ingestor first.
    //
    // Absence is not an error, unlike the test root: the production tree is only populated
    // when a production descriptor source was configured, and every case reading it skips.
    {
        const auto product = hip_kernel_provider::testing::descriptorSetRoot(
            HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR);
        if(hip_kernel_provider::testing::describeUnusableDescriptorRoot(product).empty())
        {
#ifdef _WIN32
            constexpr char PATH_SEPARATOR = ';';
#else
            constexpr char PATH_SEPARATOR = ':';
#endif
            auto list = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_PATH");
            if(!list.empty())
            {
                list += PATH_SEPARATOR;
            }
            list += product.string();
            hipdnn_data_sdk::utilities::setEnv("HIPDNN_DESCRIPTOR_PATH", list.c_str());
        }
    }
#endif

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    // NOTE: Logs are not routed to the backend by the recordingCallback returned here
    // which is the desired behaviour because this is a plugin unit test harness.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    // Initialize plugin logger with test recording callback so that plugin logs
    // are first routed to the log recorder for capture and use by the unit tests.
    hipdnn_plugin_sdk::logging::initializeCallbackLogging("hip_kernel-provider_tests",
                                                          recordingCallback);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
