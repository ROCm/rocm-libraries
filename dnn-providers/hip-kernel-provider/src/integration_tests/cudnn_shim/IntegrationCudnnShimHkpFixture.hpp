// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HKP_CUDNN_SHIM_SDPA_TESTS) && defined(HIPDNN_ENABLE_SDPA)

#include <array>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>

namespace hip_kernel_provider::test_utilities
{

// Base for the cuDNN-shim integration fixtures of this binary: loads the hip kernel provider
// plugin and owns the shim handle and the device's architecture. PLUGIN_PATH is the target-wide
// definition of integration_tests/CMakeLists.txt.
template <typename TestBase>
class CudnnShimHkpFixture : public TestBase
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        const auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(cudnnCreate(&_handle), CUDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        _arch = hip_kernel_provider_common::getDeviceString(_stream);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(cudnnDestroy(_handle), CUDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    cudnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    std::string _arch;
};

} // namespace hip_kernel_provider::test_utilities

#endif // HKP_CUDNN_SHIM_SDPA_TESTS && HIPDNN_ENABLE_SDPA
