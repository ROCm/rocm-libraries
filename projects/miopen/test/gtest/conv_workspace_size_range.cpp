// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for miopenConvolution{Forward,BackwardData,BackwardWeights}GetWorkSpaceSizeRange.
//
// The APIs are query-only (no device allocations, no kernel launches), so the
// tests run against a real handle but never touch device memory.  Each test:
//   1. Verifies the call returns miopenStatusSuccess.
//   2. Verifies min <= max (sanity on the returned range).
//   3. Cross-checks that max >= the value returned by the existing
//      GetWorkSpaceSize API (which returns the workspace of the single best
//      solver and must therefore be <= the true maximum).

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include "gtest_desc_guard.hpp"
#include "gtest_handle_guard.hpp"

#include <hip/hip_runtime_api.h>

// These functions are exported from libMIOpen but intentionally not declared
// in the public miopen.h header. Declare the prototypes locally.
extern "C" {
miopenStatus_t miopenConvolutionForwardGetWorkSpaceSizeRange(miopenHandle_t,
                                                             const miopenTensorDescriptor_t,
                                                             const miopenTensorDescriptor_t,
                                                             const miopenConvolutionDescriptor_t,
                                                             const miopenTensorDescriptor_t,
                                                             size_t*,
                                                             size_t*);
miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSizeRange(miopenHandle_t,
                                                   const miopenTensorDescriptor_t,
                                                   const miopenTensorDescriptor_t,
                                                   const miopenConvolutionDescriptor_t,
                                                   const miopenTensorDescriptor_t,
                                                   size_t*,
                                                   size_t*);
miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange(miopenHandle_t,
                                                      const miopenTensorDescriptor_t,
                                                      const miopenTensorDescriptor_t,
                                                      const miopenConvolutionDescriptor_t,
                                                      const miopenTensorDescriptor_t,
                                                      size_t*,
                                                      size_t*);
}

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct ConvConfig
{
    // input  (N, C, H, W)
    int n, c, h, w;
    // filter (K, C, R, S)
    int k, r, s;
    // conv params
    int pad_h, pad_w, stride_h, stride_w, dil_h, dil_w;
};

// Compute output spatial size for a single spatial dim.
static int outSz(int in, int filt, int pad, int stride, int dil)
{
    return (in + 2 * pad - dil * (filt - 1) - 1) / stride + 1;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class GPU_ConvWorkSpaceSizeRange_FP32 : public ::testing::TestWithParam<ConvConfig>
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipStreamCreate(&stream_), hipSuccess);
        handle_.create(stream_);
        ASSERT_EQ(handle_.getStatus(), miopenStatusSuccess);
    }

    void TearDown() override
    {
        if(stream_ != nullptr)
            EXPECT_EQ(hipStreamDestroy(stream_), hipSuccess);
    }

    // Build tensor + conv descriptors for the given config and direction,
    // then exercise all three invariants.
    void RunFwd(const ConvConfig& cfg)
    {
        TensorDescGuard xDesc, wDesc, yDesc;
        ConvDescGuard convDesc;

        // x: NCHW
        ASSERT_EQ(miopenSet4dTensorDescriptor(xDesc, miopenFloat, cfg.n, cfg.c, cfg.h, cfg.w),
                  miopenStatusSuccess);
        // w: KCRS
        ASSERT_EQ(miopenSet4dTensorDescriptor(wDesc, miopenFloat, cfg.k, cfg.c, cfg.r, cfg.s),
                  miopenStatusSuccess);
        // y: NK out_h out_w
        int oh = outSz(cfg.h, cfg.r, cfg.pad_h, cfg.stride_h, cfg.dil_h);
        int ow = outSz(cfg.w, cfg.s, cfg.pad_w, cfg.stride_w, cfg.dil_w);
        ASSERT_EQ(miopenSet4dTensorDescriptor(yDesc, miopenFloat, cfg.n, cfg.k, oh, ow),
                  miopenStatusSuccess);

        ASSERT_EQ(miopenInitConvolutionDescriptor(convDesc,
                                                  miopenConvolution,
                                                  cfg.pad_h,
                                                  cfg.pad_w,
                                                  cfg.stride_h,
                                                  cfg.stride_w,
                                                  cfg.dil_h,
                                                  cfg.dil_w),
                  miopenStatusSuccess);

        size_t minWs = 42, maxWs = 0;
        ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSizeRange(
                      handle_, wDesc, xDesc, convDesc, yDesc, &minWs, &maxWs),
                  miopenStatusSuccess);

        EXPECT_LE(minWs, maxWs) << "Fwd: min > max";

        size_t singleWs = 0;
        ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(
                      handle_, wDesc, xDesc, convDesc, yDesc, &singleWs),
                  miopenStatusSuccess);

        EXPECT_LE(singleWs, maxWs) << "Fwd: GetWorkSpaceSize result exceeds reported max";
    }

    void RunBwdData(const ConvConfig& cfg)
    {
        TensorDescGuard dyDesc, wDesc, dxDesc;
        ConvDescGuard convDesc;

        int oh = outSz(cfg.h, cfg.r, cfg.pad_h, cfg.stride_h, cfg.dil_h);
        int ow = outSz(cfg.w, cfg.s, cfg.pad_w, cfg.stride_w, cfg.dil_w);

        ASSERT_EQ(miopenSet4dTensorDescriptor(dyDesc, miopenFloat, cfg.n, cfg.k, oh, ow),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(wDesc, miopenFloat, cfg.k, cfg.c, cfg.r, cfg.s),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(dxDesc, miopenFloat, cfg.n, cfg.c, cfg.h, cfg.w),
                  miopenStatusSuccess);

        ASSERT_EQ(miopenInitConvolutionDescriptor(convDesc,
                                                  miopenConvolution,
                                                  cfg.pad_h,
                                                  cfg.pad_w,
                                                  cfg.stride_h,
                                                  cfg.stride_w,
                                                  cfg.dil_h,
                                                  cfg.dil_w),
                  miopenStatusSuccess);

        size_t minWs = 42, maxWs = 0;
        ASSERT_EQ(miopenConvolutionBackwardDataGetWorkSpaceSizeRange(
                      handle_, dyDesc, wDesc, convDesc, dxDesc, &minWs, &maxWs),
                  miopenStatusSuccess);

        EXPECT_LE(minWs, maxWs) << "BwdData: min > max";

        size_t singleWs = 0;
        ASSERT_EQ(miopenConvolutionBackwardDataGetWorkSpaceSize(
                      handle_, dyDesc, wDesc, convDesc, dxDesc, &singleWs),
                  miopenStatusSuccess);

        EXPECT_LE(singleWs, maxWs) << "BwdData: GetWorkSpaceSize result exceeds reported max";
    }

    void RunBwdWeights(const ConvConfig& cfg)
    {
        TensorDescGuard dyDesc, xDesc, dwDesc;
        ConvDescGuard convDesc;

        int oh = outSz(cfg.h, cfg.r, cfg.pad_h, cfg.stride_h, cfg.dil_h);
        int ow = outSz(cfg.w, cfg.s, cfg.pad_w, cfg.stride_w, cfg.dil_w);

        ASSERT_EQ(miopenSet4dTensorDescriptor(dyDesc, miopenFloat, cfg.n, cfg.k, oh, ow),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(xDesc, miopenFloat, cfg.n, cfg.c, cfg.h, cfg.w),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(dwDesc, miopenFloat, cfg.k, cfg.c, cfg.r, cfg.s),
                  miopenStatusSuccess);

        ASSERT_EQ(miopenInitConvolutionDescriptor(convDesc,
                                                  miopenConvolution,
                                                  cfg.pad_h,
                                                  cfg.pad_w,
                                                  cfg.stride_h,
                                                  cfg.stride_w,
                                                  cfg.dil_h,
                                                  cfg.dil_w),
                  miopenStatusSuccess);

        size_t minWs = 42, maxWs = 0;
        ASSERT_EQ(miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange(
                      handle_, dyDesc, xDesc, convDesc, dwDesc, &minWs, &maxWs),
                  miopenStatusSuccess);

        EXPECT_LE(minWs, maxWs) << "BwdWeights: min > max";

        size_t singleWs = 0;
        ASSERT_EQ(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                      handle_, dyDesc, xDesc, convDesc, dwDesc, &singleWs),
                  miopenStatusSuccess);

        EXPECT_LE(singleWs, maxWs) << "BwdWeights: GetWorkSpaceSize result exceeds reported max";
    }

    hipStream_t stream_ = nullptr;
    HandleGuard handle_;
};

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

// cppcheck-suppress syntaxError
TEST_P(GPU_ConvWorkSpaceSizeRange_FP32, Forward) { RunFwd(GetParam()); }
TEST_P(GPU_ConvWorkSpaceSizeRange_FP32, BackwardData) { RunBwdData(GetParam()); }
TEST_P(GPU_ConvWorkSpaceSizeRange_FP32, BackwardWeights) { RunBwdWeights(GetParam()); }

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvWorkSpaceSizeRange_FP32,
    ::testing::Values(
        // n  c    h    w    k  r  s  ph pw sh sw dh dw  (out computed)
        ConvConfig{1, 64, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1},  // 3x3, unit stride
        ConvConfig{1, 64, 56, 56,  64, 1, 1, 0, 0, 1, 1, 1, 1},  // 1x1, unit stride
        ConvConfig{1,  3, 32, 32,  16, 5, 5, 2, 2, 1, 1, 1, 1},  // 5x5
        ConvConfig{1, 32, 14, 14,  64, 3, 3, 1, 1, 2, 2, 1, 1}   // strided
    ));
// clang-format on

} // namespace
