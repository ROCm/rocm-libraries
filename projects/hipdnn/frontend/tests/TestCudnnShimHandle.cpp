// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Handle-lifecycle / stream-binding unit test for the cuDNN-compatibility
// shim's stub C-API (`cudnn.h`). The shim
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
    // Claimed cuDNN runtime version 9.22.0 (matches the cuDNN FE v1.24.0
    // recommendation; see cudnn_runtime_version.h).
    EXPECT_EQ(cudnnGetVersion(), static_cast<size_t>(92200));
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

// Status translation (detail/status_translation.h) is covered with one
// parameterized case per enum mapping, in both directions. Adding or updating a
// mapping is then a single-row edit here, with full coverage of every switch
// case (including the grouped/collapsed values and the default fallback).

struct HipdnnToCudnnCase
{
    hipdnnStatus_t input;
    cudnnStatus_t expected;
};

class TestCudnnShimToCudnnStatus : public ::testing::TestWithParam<HipdnnToCudnnCase>
{
};

TEST_P(TestCudnnShimToCudnnStatus, MapsToExpected)
{
    EXPECT_EQ(shim_detail::toCudnnStatus(GetParam().input), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    AllMappings,
    TestCudnnShimToCudnnStatus,
    ::testing::Values(
        HipdnnToCudnnCase{HIPDNN_STATUS_SUCCESS, CUDNN_STATUS_SUCCESS},
        HipdnnToCudnnCase{HIPDNN_STATUS_NOT_INITIALIZED, CUDNN_STATUS_NOT_INITIALIZED},
        // The BAD_PARAM family collapses to CUDNN_STATUS_BAD_PARAM.
        HipdnnToCudnnCase{HIPDNN_STATUS_BAD_PARAM, CUDNN_STATUS_BAD_PARAM},
        HipdnnToCudnnCase{HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, CUDNN_STATUS_BAD_PARAM},
        HipdnnToCudnnCase{HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED, CUDNN_STATUS_BAD_PARAM},
        HipdnnToCudnnCase{HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND, CUDNN_STATUS_BAD_PARAM},
        HipdnnToCudnnCase{HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT, CUDNN_STATUS_BAD_PARAM},
        HipdnnToCudnnCase{HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH, CUDNN_STATUS_BAD_PARAM},
        HipdnnToCudnnCase{HIPDNN_STATUS_NOT_SUPPORTED, CUDNN_STATUS_NOT_SUPPORTED},
        // The allocation-failure family collapses to CUDNN_STATUS_ALLOC_FAILED.
        HipdnnToCudnnCase{HIPDNN_STATUS_ALLOC_FAILED, CUDNN_STATUS_ALLOC_FAILED},
        HipdnnToCudnnCase{HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED,
                          CUDNN_STATUS_ALLOC_FAILED},
        HipdnnToCudnnCase{HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED,
                          CUDNN_STATUS_ALLOC_FAILED},
        HipdnnToCudnnCase{HIPDNN_STATUS_EXECUTION_FAILED, CUDNN_STATUS_EXECUTION_FAILED},
        HipdnnToCudnnCase{HIPDNN_STATUS_INTERNAL_ERROR, CUDNN_STATUS_INTERNAL_ERROR},
        // PLUGIN_ERROR has no cuDNN equivalent and falls through to the default.
        HipdnnToCudnnCase{HIPDNN_STATUS_PLUGIN_ERROR, CUDNN_STATUS_INTERNAL_ERROR}));

struct CudnnToHipdnnCase
{
    cudnnStatus_t input;
    hipdnnStatus_t expected;
};

class TestCudnnShimToHipdnnStatus : public ::testing::TestWithParam<CudnnToHipdnnCase>
{
};

TEST_P(TestCudnnShimToHipdnnStatus, MapsToExpected)
{
    EXPECT_EQ(shim_detail::toHipdnnStatus(GetParam().input), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    AllMappings,
    TestCudnnShimToHipdnnStatus,
    ::testing::Values(
        CudnnToHipdnnCase{CUDNN_STATUS_SUCCESS, HIPDNN_STATUS_SUCCESS},
        CudnnToHipdnnCase{CUDNN_STATUS_NOT_INITIALIZED, HIPDNN_STATUS_NOT_INITIALIZED},
        CudnnToHipdnnCase{CUDNN_STATUS_ALLOC_FAILED, HIPDNN_STATUS_ALLOC_FAILED},
        CudnnToHipdnnCase{CUDNN_STATUS_BAD_PARAM, HIPDNN_STATUS_BAD_PARAM},
        CudnnToHipdnnCase{CUDNN_STATUS_INVALID_VALUE, HIPDNN_STATUS_BAD_PARAM},
        CudnnToHipdnnCase{CUDNN_STATUS_NOT_SUPPORTED, HIPDNN_STATUS_NOT_SUPPORTED},
        // ARCH_MISMATCH maps onto NOT_SUPPORTED.
        CudnnToHipdnnCase{CUDNN_STATUS_ARCH_MISMATCH, HIPDNN_STATUS_NOT_SUPPORTED},
        CudnnToHipdnnCase{CUDNN_STATUS_EXECUTION_FAILED, HIPDNN_STATUS_EXECUTION_FAILED},
        CudnnToHipdnnCase{CUDNN_STATUS_INTERNAL_ERROR, HIPDNN_STATUS_INTERNAL_ERROR},
        // cuDNN-only codes with no hipDNN counterpart fall through to the default.
        CudnnToHipdnnCase{CUDNN_STATUS_MAPPING_ERROR, HIPDNN_STATUS_INTERNAL_ERROR},
        CudnnToHipdnnCase{CUDNN_STATUS_LICENSE_ERROR, HIPDNN_STATUS_INTERNAL_ERROR},
        CudnnToHipdnnCase{CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING, HIPDNN_STATUS_INTERNAL_ERROR},
        CudnnToHipdnnCase{CUDNN_STATUS_RUNTIME_IN_PROGRESS, HIPDNN_STATUS_INTERNAL_ERROR},
        CudnnToHipdnnCase{CUDNN_STATUS_RUNTIME_FP_OVERFLOW, HIPDNN_STATUS_INTERNAL_ERROR},
        CudnnToHipdnnCase{CUDNN_STATUS_VERSION_MISMATCH, HIPDNN_STATUS_INTERNAL_ERROR}));

} // namespace
