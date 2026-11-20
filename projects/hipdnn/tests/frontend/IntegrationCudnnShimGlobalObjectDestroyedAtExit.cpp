// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A handle from the cuDNN-compatibility shim's create_cudnn_handle(), owned by a
// namespace-scope object and destroyed during process exit. See
// GlobalObjectDestroyedAtExitFixture.hpp for how the child process is checked.
//
// Gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in tests/frontend/CMakeLists.txt.

#include <memory>

#include <gtest/gtest.h>

#include <hipdnn_compatibility/cudnn/cudnn.h>

#include "GlobalObjectDestroyedAtExitFixture.hpp"

namespace
{

hipdnn_tests::PluginUnloadProbe gPluginUnloadProbe;
std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter> gCudnnHandle;

class IntegrationCudnnShimGlobalObjectDestroyedAtExit
    : public hipdnn_tests::IntegrationGlobalObjectDestroyedAtExit
{
};

TEST_F(IntegrationCudnnShimGlobalObjectDestroyedAtExit, HandleExitsCleanly)
{
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild(gPluginUnloadProbe));
    gCudnnHandle = create_cudnn_handle();
    ASSERT_NE(gCudnnHandle, nullptr);
    ASSERT_TRUE(isTestGoodPluginLoaded());
}

} // namespace
