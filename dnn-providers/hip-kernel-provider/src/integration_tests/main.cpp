/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <filesystem>
#include <iostream>
#include <system_error>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#ifdef HIPKERNELPROVIDER_TEST_SET_INTEGRATION_RELDIR
#include "TestDescriptorRoot.hpp"
#endif

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Keep hipDNN's on-disk caches out of the developer's ~/.cache/hipdnn. The ingestor
    // suite asserts on benchmarking behaviour, which a persisted winner shard from an
    // earlier run of this same build would satisfy without benchmarking at all.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "hip-kernel-provider-integration");

#ifdef HIPKERNELPROVIDER_TEST_SET_INTEGRATION_RELDIR
    // Point this binary at the one descriptor set its cases resolve, unless the caller
    // already named one. Can be set only once per test process. Fail the process when the
    // resolved root holds no descriptor, which is otherwise indistinguishable from a run
    // on a device the descriptors do not cover.
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR").empty())
    {
        const auto descriptors = hip_kernel_provider::testing::descriptorSetRoot(
            HIPKERNELPROVIDER_TEST_SET_INTEGRATION_RELDIR);
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

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
    // The production descriptor tree, loaded beside the set above. The offset is fixed at
    // build time whatever GPU targets the build had, and the tree is looked up here at run
    // time: an install may pair this binary with device content another build produced,
    // and a tree that is absent adds nothing. The set above replaces the plugin-relative
    // tree the production engines would otherwise come from; HIPDNN_DESCRIPTOR_RUNTIME_DIR
    // is additive, so both load. The root is the arch-neutral one: the loader walks every
    // arch subtree under it and prunes each pack by its `arch` list against the running
    // device at match time, so the device's own shard is the only one that can serve. A
    // caller-set value is kept as given.
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
    {
        const auto product = hip_kernel_provider::testing::descriptorSetRoot(
            HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR);
        std::error_code absent;
        if(!product.empty() && std::filesystem::is_directory(product, absent))
        {
            hipdnn_data_sdk::utilities::setEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR",
                                               product.string().c_str());
        }
    }
#endif
#endif

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
