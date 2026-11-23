// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// One forwarded convolution on a fresh handle, then miopenDestroy on it. The
// miopen-provider-plugin-reentry-guard entry runs this through script/check_plugin_reentry.py,
// with hipDNN loading only a plugin built against the public libMIOpen.so (it calls back into
// the wrapper from inside hipDNN). With MIOPEN_TEST_EXPECT_REENTRY_REJECTION=1 the convolution
// must fail as the wrapper promises; without it, it must work, as with hipDNN's default plugin.
// Either way miopenDestroy must return: it releases the handle's hipDNN state, and the plugin's
// handle destructor calls miopenDestroy for the plugin's own handle.
//
// "Shim" is not in the suite name on purpose: that puts a suite in the forwarding parity run.

#include <gtest/gtest.h>

#include "../../src/private/routing.hpp"

#include <miopen/hipdnn_forwarding.h>
#include <miopen/miopen.h>

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>

namespace {

bool ExpectRejection()
{
    const char* value = std::getenv("MIOPEN_TEST_EXPECT_REENTRY_REJECTION");
    return value != nullptr && std::string(value) == "1";
}

void DestroyWithinTimeout(miopenHandle_t handle)
{
    constexpr std::chrono::seconds timeout{30};
    // On its own thread so that a deadlock can be reported instead of hanging the test.
    auto destroyed = std::async(std::launch::async, [handle] { return miopenDestroy(handle); });
    if(destroyed.wait_for(timeout) != std::future_status::ready)
    {
        std::cerr << "miopenDestroy did not return within " << timeout.count() << " s\n";
        // The stuck thread cannot be joined, and the future's destructor would wait for it.
        std::_Exit(1);
    }
    EXPECT_EQ(destroyed.get(), miopenStatusSuccess);
}

} // namespace

TEST(GPU_HipdnnReentry_FP32, RejectedCallThenDestroyReturns)
{
    if(miopen::wrapper::GetForwardingMode() != miopen::wrapper::ForwardingMode::Enabled)
        GTEST_SKIP() << "needs MIOPEN_HIPDNN_FORWARDING=1; run it through "
                        "script/check_plugin_reentry.py";

    miopenHandle_t handle = nullptr;
    ASSERT_EQ(miopenCreate(&handle), miopenStatusSuccess);

    miopenTensorDescriptor_t x         = nullptr;
    miopenTensorDescriptor_t w         = nullptr;
    miopenTensorDescriptor_t y         = nullptr;
    miopenConvolutionDescriptor_t conv = nullptr;
    ASSERT_EQ(miopenCreateTensorDescriptor(&x), miopenStatusSuccess);
    ASSERT_EQ(miopenCreateTensorDescriptor(&w), miopenStatusSuccess);
    ASSERT_EQ(miopenCreateTensorDescriptor(&y), miopenStatusSuccess);
    ASSERT_EQ(miopenCreateConvolutionDescriptor(&conv), miopenStatusSuccess);

    ASSERT_EQ(miopenSet4dTensorDescriptor(x, miopenFloat, 1, 1, 4, 4), miopenStatusSuccess);
    ASSERT_EQ(miopenSet4dTensorDescriptor(w, miopenFloat, 1, 1, 3, 3), miopenStatusSuccess);
    ASSERT_EQ(miopenInitConvolutionDescriptor(conv, miopenConvolution, 1, 1, 1, 1, 1, 1),
              miopenStatusSuccess);
    int n    = 0;
    int c    = 0;
    int h    = 0;
    int wOut = 0;
    ASSERT_EQ(miopenGetConvolutionForwardOutputDim(conv, x, w, &n, &c, &h, &wOut),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSet4dTensorDescriptor(y, miopenFloat, n, c, h, wOut), miopenStatusSuccess);

    void* xData = nullptr;
    void* wData = nullptr;
    void* yData = nullptr;
    ASSERT_EQ(hipMalloc(&xData, sizeof(float) * 16), hipSuccess);
    ASSERT_EQ(hipMalloc(&wData, sizeof(float) * 9), hipSuccess);
    ASSERT_EQ(hipMalloc(&yData, sizeof(float) * 16), hipSuccess);
    ASSERT_EQ(hipMemset(xData, 0, sizeof(float) * 16), hipSuccess);
    ASSERT_EQ(hipMemset(wData, 0, sizeof(float) * 9), hipSuccess);

    const float alpha           = 1.0f;
    const float beta            = 0.0f;
    const miopenStatus_t status = miopenConvolutionForward(handle,
                                                           &alpha,
                                                           x,
                                                           xData,
                                                           w,
                                                           wData,
                                                           conv,
                                                           miopenConvolutionFwdAlgoDirect,
                                                           &beta,
                                                           y,
                                                           yData,
                                                           nullptr,
                                                           0);

    if(ExpectRejection())
    {
        // Without forwarding MIOpen serves the call and it succeeds; say so instead of leaving
        // a bare status mismatch.
        EXPECT_EQ(status, miopenStatusInternalError)
            << "the call must fail; is MIOPEN_HIPDNN_FORWARDING=1 and is the plugin the one "
               "built against the public library?";
        miopenStatus_t forwarded = miopenStatusSuccess;
        EXPECT_EQ(miopenGetLastForwardedError(&forwarded), miopenStatusSuccess);
        EXPECT_EQ(forwarded, miopenStatusInternalError);
        EXPECT_EQ(std::string(miopenGetErrorString(status)).rfind("[hipDNN-forwarded]", 0), 0u);
    }
    else
    {
        EXPECT_EQ(status, miopenStatusSuccess);
    }

    static_cast<void>(hipFree(xData));
    static_cast<void>(hipFree(wData));
    static_cast<void>(hipFree(yData));
    static_cast<void>(miopenDestroyConvolutionDescriptor(conv));
    static_cast<void>(miopenDestroyTensorDescriptor(x));
    static_cast<void>(miopenDestroyTensorDescriptor(w));
    static_cast<void>(miopenDestroyTensorDescriptor(y));

    DestroyWithinTimeout(handle);
}
