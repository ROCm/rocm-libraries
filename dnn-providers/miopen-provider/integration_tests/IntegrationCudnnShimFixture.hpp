// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace miopen_plugin::test_utilities
{

// Base for the cuDNN-shim integration fixtures: loads the MIOpen plugin and owns the shim
// handle. PLUGIN_PATH is supplied by this directory's CMakeLists.txt.
template <typename TestBase>
class CudnnShimFixture : public TestBase
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
        SKIP_IF_ASAN();

        ASSERT_EQ(hipInit(0), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(cudnnCreate(&_handle), CUDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(cudnnDestroy(_handle), CUDNN_STATUS_SUCCESS);
        }
    }

    cudnnHandle_t _handle = nullptr;
};

} // namespace miopen_plugin::test_utilities
