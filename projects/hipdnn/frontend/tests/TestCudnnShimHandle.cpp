// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ALMIOPEN-2036 acceptance criterion #3: a handle-lifecycle / stream-binding
// unit test for the cuDNN-compatibility shim's stub C-API (`cudnn.h`). The shim
// entry points forward through hipdnn_frontend::detail::hipdnnBackend(), so this
// test drives them against the in-tree mock backend (same pattern as
// TestHandle.cpp). Gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in the
// frontend tests CMakeLists, so it is only built when the shim is enabled.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

namespace
{

// AC#1: cudnnHandle_t must be the hipDNN handle type, not a parallel typedef.
static_assert(std::is_same_v<cudnnHandle_t, ::hipdnnHandle_t>,
              "cudnnHandle_t must alias the hipDNN handle type");

class TestCudnnShimHandle : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestCudnnShimHandle, CreateForwardsAndMapsSuccess)
{
    auto fakeHandle = reinterpret_cast<cudnnHandle_t>(0x1234);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });

    cudnnHandle_t handle = nullptr;
    EXPECT_EQ(cudnnCreate(&handle), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(handle, fakeHandle);
}

TEST_F(TestCudnnShimHandle, CreateMapsBackendFailure)
{
    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([](hipdnnHandle_t*) {
        return HIPDNN_STATUS_INTERNAL_ERROR;
    });

    cudnnHandle_t handle = nullptr;
    EXPECT_EQ(cudnnCreate(&handle), CUDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCudnnShimHandle, DestroyForwards)
{
    auto fakeHandle = reinterpret_cast<cudnnHandle_t>(0x2345);

    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_EQ(cudnnDestroy(fakeHandle), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimHandle, SetStreamForwards)
{
    auto fakeHandle = reinterpret_cast<cudnnHandle_t>(0x3456);
    auto fakeStream = reinterpret_cast<hipStream_t>(0xABCD);

    EXPECT_CALL(*_mockBackend, setStream(fakeHandle, fakeStream))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_EQ(cudnnSetStream(fakeHandle, fakeStream), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimHandle, GetStreamForwardsAndReturnsStream)
{
    auto fakeHandle = reinterpret_cast<cudnnHandle_t>(0x4567);
    auto fakeStream = reinterpret_cast<hipStream_t>(0xBCDE);

    EXPECT_CALL(*_mockBackend, getStream(fakeHandle, _))
        .WillOnce([&fakeStream](hipdnnHandle_t, hipStream_t* out) {
            *out = fakeStream;
            return HIPDNN_STATUS_SUCCESS;
        });

    hipStream_t stream = nullptr;
    EXPECT_EQ(cudnnGetStream(fakeHandle, &stream), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(stream, fakeStream);
}

TEST_F(TestCudnnShimHandle, SetStreamMapsBadParam)
{
    auto fakeHandle = reinterpret_cast<cudnnHandle_t>(0x5678);
    auto fakeStream = reinterpret_cast<hipStream_t>(0xCDEF);

    EXPECT_CALL(*_mockBackend, setStream(fakeHandle, fakeStream))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH));

    EXPECT_EQ(cudnnSetStream(fakeHandle, fakeStream), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimHandle, GetErrorStringMapsAndForwards)
{
    const char* fakeMessage = "fake backend message";

    EXPECT_CALL(*_mockBackend, getErrorString(HIPDNN_STATUS_NOT_SUPPORTED))
        .WillOnce(Return(fakeMessage));

    EXPECT_STREQ(cudnnGetErrorString(CUDNN_STATUS_NOT_SUPPORTED), fakeMessage);
}

TEST_F(TestCudnnShimHandle, GetVersionReturnsClaimedRuntimeVersion)
{
    // Earmarked for review (ALMIOPEN-2036): claimed cuDNN runtime version 9.14.0.
    EXPECT_EQ(cudnnGetVersion(), static_cast<size_t>(91400));
    EXPECT_EQ(cudnnGetVersion(), static_cast<size_t>(CUDNN_VERSION));
}

TEST_F(TestCudnnShimHandle, FrontendVersionMacroMatchesUpstreamPin)
{
    // RFC 0012 §4.8 / §2: pinned to cuDNN FE v1.24.0.
    EXPECT_EQ(CUDNN_FRONTEND_VERSION, 12400);
}

TEST_F(TestCudnnShimHandle, CreateCudnnHandleReturnsManagedHandleAndDestroysOnScopeExit)
{
    auto fakeHandle = reinterpret_cast<cudnnHandle_t>(0x7890);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    {
        auto handle = create_cudnn_handle();
        ASSERT_NE(handle, nullptr);
        EXPECT_EQ(*handle, fakeHandle);
    }
    // destroy is verified by the EXPECT_CALL expectation on scope exit.
}

namespace shim_detail = hipdnn_frontend::compatibility::cudnn_frontend::detail;

// One test per direction for the status translation (detail/status_translation.h):
// a representative direct mapping, a collapsed/grouped value, and the default
// fallback for an unmapped input.
TEST(TestCudnnShimStatusTranslation, HipdnnToCudnn)
{
    EXPECT_EQ(shim_detail::toCudnnStatus(HIPDNN_STATUS_SUCCESS), CUDNN_STATUS_SUCCESS);
    // The BAD_PARAM family collapses to CUDNN_STATUS_BAD_PARAM.
    EXPECT_EQ(shim_detail::toCudnnStatus(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER),
              CUDNN_STATUS_BAD_PARAM);
    // PLUGIN_ERROR has no cuDNN equivalent and falls through to the default.
    EXPECT_EQ(shim_detail::toCudnnStatus(HIPDNN_STATUS_PLUGIN_ERROR), CUDNN_STATUS_INTERNAL_ERROR);
}

TEST(TestCudnnShimStatusTranslation, CudnnToHipdnn)
{
    EXPECT_EQ(shim_detail::toHipdnnStatus(CUDNN_STATUS_SUCCESS), HIPDNN_STATUS_SUCCESS);
    // ARCH_MISMATCH maps onto NOT_SUPPORTED.
    EXPECT_EQ(shim_detail::toHipdnnStatus(CUDNN_STATUS_ARCH_MISMATCH), HIPDNN_STATUS_NOT_SUPPORTED);
    // A cuDNN-only code with no hipDNN counterpart falls through to the default.
    EXPECT_EQ(shim_detail::toHipdnnStatus(CUDNN_STATUS_VERSION_MISMATCH),
              HIPDNN_STATUS_INTERNAL_ERROR);
}

} // namespace
