// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Integration coverage for the cuDNN-compatibility shim's stub C-API
// (`cudnn.h`) against the *real* hipDNN backend (RFC 0012 §4.7).
// The frontend unit tests (frontend/tests/TestCudnnShimHandle.cpp) exercise the
// same entry points against a mock backend; here the same calls run through the
// live backend wrapper, proving the forwarding actually reaches hipDNN.
//
// Device requirements (intentionally split):
//   * cudnnGetVersion / cudnnGetErrorString and the version macros do NOT touch
//     a GPU — cudnnGetVersion returns a compile-time constant and
//     cudnnGetErrorString is a pure status->string lookup in the backend (its
//     lazy init only queries the backend version, never opens a device). These
//     run without SKIP_IF_NO_DEVICES().
//   * cudnnCreate / cudnnDestroy / cudnnSetStream / cudnnGetStream and
//     create_cudnn_handle create a real hipDNN handle, which requires a GPU.
//     They use IntegrationTestFixture, whose SetUp() calls SKIP_IF_NO_DEVICES()
//     (and hipInit), so they skip cleanly on device-less hosts.
//
// Gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in tests/frontend/CMakeLists.txt.

#include <string>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

namespace
{
using hipdnn_tests::IntegrationTestFixture;

// ---------------------------------------------------------------------------
// Device-independent entry points (no SKIP_IF_NO_DEVICES needed).
// ---------------------------------------------------------------------------

TEST(IntegrationCudnnShimVersion, ReportsClaimedRuntimeAndFrontendVersions)
{
    EXPECT_EQ(cudnnGetVersion(), static_cast<size_t>(CUDNN_VERSION));
    EXPECT_EQ(CUDNN_VERSION, 92200); // claimed cuDNN runtime version (9.22.0)
    EXPECT_EQ(CUDNN_FRONTEND_VERSION, 12400); // pinned cuDNN FE version (1.24.0)
}

TEST(IntegrationCudnnShimError, GetErrorStringForwardsToRealBackend)
{
    // Forwards to the live backend's status->string lookup; no device required.
    const char* message = cudnnGetErrorString(CUDNN_STATUS_SUCCESS);
    ASSERT_NE(message, nullptr);
    EXPECT_FALSE(std::string(message).empty());
}

// ---------------------------------------------------------------------------
// Device-dependent handle/stream lifecycle. IntegrationTestFixture::SetUp()
// runs SKIP_IF_NO_DEVICES() + hipInit(), so these skip without a GPU.
// ---------------------------------------------------------------------------

class IntegrationCudnnShimHandle : public IntegrationTestFixture
{
};

TEST_F(IntegrationCudnnShimHandle, CreateAndDestroyRealHandle)
{
    cudnnHandle_t handle = nullptr;
    ASSERT_EQ(cudnnCreate(&handle), CUDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(cudnnDestroy(handle), CUDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationCudnnShimHandle, SetAndGetStreamRoundTrip)
{
    cudnnHandle_t handle = nullptr;
    ASSERT_EQ(cudnnCreate(&handle), CUDNN_STATUS_SUCCESS);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    EXPECT_EQ(cudnnSetStream(handle, stream), CUDNN_STATUS_SUCCESS);

    hipStream_t boundStream = nullptr;
    EXPECT_EQ(cudnnGetStream(handle, &boundStream), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(boundStream, stream);

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    EXPECT_EQ(cudnnDestroy(handle), CUDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationCudnnShimHandle, CreateCudnnHandleHelperReturnsLiveHandle)
{
    auto handle = create_cudnn_handle();
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(*handle, nullptr);
    // Bind the default stream through the managed handle to prove it is live.
    EXPECT_EQ(cudnnSetStream(*handle, nullptr), CUDNN_STATUS_SUCCESS);
    // Destroyed automatically by CudnnHandleDeleter on scope exit.
}

} // namespace
