/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Keep hipDNN's on-disk caches out of the developer's ~/.cache/hipdnn. The ingestor
    // suite asserts on benchmarking behaviour, which a persisted winner shard from an
    // earlier run of this same build would satisfy without benchmarking at all.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "hip-kernel-provider-integration");

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
