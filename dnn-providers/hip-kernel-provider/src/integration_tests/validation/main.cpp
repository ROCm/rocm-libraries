/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <iostream>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "TestDescriptorRoot.hpp"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Keep hipDNN's on-disk caches out of the developer's ~/.cache/hipdnn. The cases record
    // winners, and a persisted shard from an earlier run would be served without
    // benchmarking.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir("hip-kernel-provider-validation");

    // Point this binary at the validation_fixture descriptor set, unless the caller already
    // named one. The set is the only descriptor root this binary loads: the production tree
    // is not loaded, so the fixture's wrong kernel is the only content in play. Fail the
    // process when the resolved root holds no descriptor, which is otherwise
    // indistinguishable from a run on a device the descriptors do not cover.
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR").empty())
    {
        const auto descriptors = hip_kernel_provider::testing::descriptorSetRoot(
            HIPKERNELPROVIDER_TEST_SET_VALIDATION_FIXTURE_RELDIR);
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

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
