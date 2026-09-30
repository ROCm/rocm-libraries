/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <spdlog/spdlog.h>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>

#include "logging/Logging.hpp"

namespace
{
// Point the plugin manager at a non-existent directory so unit tests never load
// production engine plugins (which would trigger GPU initialization). The plugin
// scanner treats a missing directory as zero plugins. Respects an externally
// provided HIPDNN_PLUGIN_DIR override.
class EmptyPluginDirEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_PLUGIN_DIR").empty())
        {
            hipdnn_data_sdk::utilities::setEnv("HIPDNN_PLUGIN_DIR",
                                               "hipdnn_nonexistent_plugin_dir_for_unit_tests");
        }
    }
};
} // namespace

int main(int argc, char** argv)
{
    ::testing::AddGlobalTestEnvironment(new EmptyPluginDirEnvironment);

    ::testing::InitGoogleTest(&argc, argv);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    auto result = RUN_ALL_TESTS();
    return result;
}
