/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <filesystem>
#include <system_error>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "TestDescriptorRoot.hpp"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Keep hipDNN's on-disk caches out of the developer's ~/.cache/hipdnn. The ingestor
    // suite asserts on benchmarking behaviour, which a persisted winner shard from an
    // earlier run of this same build would satisfy without benchmarking at all.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "hip-kernel-provider-integration");

#ifdef HIPKERNELPROVIDER_TEST_SET_EMBEDDED_ENGINE_RELDIR
    // Point this binary at the one descriptor set its cases resolve (override default
    // if not already set in env).
    // Can be set only once per test process.
    if(std::error_code notFound;
       hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR").empty())
    {
        const auto descriptors = hip_kernel_provider::testing::descriptorSetRoot(
            HIPKERNELPROVIDER_TEST_SET_EMBEDDED_ENGINE_RELDIR);
        if(std::filesystem::is_directory(descriptors, notFound))
        {
            hipdnn_data_sdk::utilities::setEnv("HIPDNN_DESCRIPTOR_DIR",
                                               descriptors.string().c_str());
        }
    }
#endif

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
