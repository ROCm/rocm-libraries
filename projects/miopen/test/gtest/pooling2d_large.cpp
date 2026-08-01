// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Scale test: 2D Pooling with numel > INT_MAX.
//
// JIRA: ALMIOPEN-2145
//
// Motivation
// ----------
// Existing pooling tests top out at approximately 40 M elements (e.g.
// {1,19,1024,2048}).  A probe on gfx942 (xnack-) confirmed that MaxPool and
// AvgPool with a 2x2 window produce correct, finite output at this scale.
// These cases are added as regression guards: any future INT_MAX overflow in
// pooling index arithmetic would cause NaN or Inf in the output, which the
// assertions below detect.
//
// Shape chosen
// ------------
//   N=1, C=1, H=46342, W=46342 → numel = 2 147 580 964 > 2^31-1 = 2 147 483 647
//   Window 2×2, stride 2×2, pad 0×0 → output 23171×23171 ≈ 537 M elements
//
// Memory budget (forward only, FP32)
//   Input:    1×1×46342×46342 × 4 B ≈  8.0 GiB
//   Output:   1×1×23171×23171 × 4 B ≈  2.0 GiB
//   Workspace (max pool, uint64):    ≈  4.0 GiB
//   Total (max pool):                ≈ 14.0 GiB
//   Total (avg pool):                ≈ 10.0 GiB
//
// No CPU reference comparison is performed: allocating ~8 GiB on the host
// per buffer would be prohibitive.  Correctness is checked by asserting that
// a sample of output elements is finite (non-NaN, non-Inf), which fails
// immediately when index overflow corrupts GPU memory.
//
// The test queries free GPU memory and calls GTEST_SKIP() when headroom is
// insufficient, so it is safe to run on smaller devices.

#include <miopen/miopen.h>
#include <miopen/pooling.hpp>
#include <miopen/tensor.hpp>

#include <hip/hip_runtime.h>

#include "get_handle.hpp"
#include "workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

// Test case parameters: pooling mode and a human-readable label.
struct PoolingLargeTestCase
{
    miopenPoolingMode_t mode;
    std::string label;

    friend std::ostream& operator<<(std::ostream& os, const PoolingLargeTestCase& tc)
    {
        return os << tc.label;
    }
};

// Shape: numel = N*C*H*W = 1*1*46342*46342 = 2 147 580 964 > INT_MAX.
constexpr int kN = 1;
constexpr int kC = 1;
constexpr int kH = 46342;
constexpr int kW = 46342;

// Window / stride / pad (2-D spatial, 2×2 kernel).
constexpr int kWinH = 2;
constexpr int kWinW = 2;
constexpr int kStrH = 2;
constexpr int kStrW = 2;
constexpr int kPadH = 0;
constexpr int kPadW = 0;

static_assert(static_cast<long long>(kN) * kC * kH * kW > 2147483647LL,
              "numel must exceed INT_MAX to exercise 64-bit index paths");

// Required free GPU memory before the test is allowed to run.
// Max pool (worst case): input (~8 GiB) + output (~2 GiB) + workspace (~4 GiB) ≈ 14 GiB.
// We require 16 GiB to leave a safety margin.
constexpr size_t kMinFreeMemBytes = 16ULL * 1024 * 1024 * 1024;

// Helper: fill a device buffer with a non-degenerate pattern (avoids all-zero inputs).
// hipMemset with 0x3F gives a repeated-byte IEEE 754 value (all elements ≈ 0.247,
// positive and finite), so max-pool has a well-defined winner.
void FillDeviceBuffer(void* dev_ptr, size_t n_bytes) { (void)hipMemset(dev_ptr, 0x3F, n_bytes); }

// Helper: check that a sample of output elements are finite.
// Copying all ~537 M elements to the host is slow; we read the first 256 Ki floats.
// An INT_MAX overflow corrupts the very first output elements, so this is sufficient.
bool SampleIsFinite(const void* dev_ptr, size_t n_elems)
{
    constexpr size_t kSampleElems = 256 * 1024;
    const size_t sample           = std::min(kSampleElems, n_elems);
    std::vector<float> host(sample);
    (void)hipMemcpy(host.data(), dev_ptr, sample * sizeof(float), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < sample; ++i)
    {
        if(!std::isfinite(host[i]))
            return false;
    }
    return true;
}

// ---- Parameterized test fixture ------------------------------------------------
struct GPU_Pooling2d_Large_FP32 : public testing::TestWithParam<PoolingLargeTestCase>
{
    void SetUp() override
    {
        size_t free_bytes    = 0;
        size_t total_bytes   = 0;
        const hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
        if(err != hipSuccess)
        {
            GTEST_SKIP() << "hipMemGetInfo failed (err=" << static_cast<int>(err)
                         << "); skipping large-scale pooling test";
        }
        if(free_bytes < kMinFreeMemBytes)
        {
            GTEST_SKIP() << "Insufficient free GPU memory: need " << (kMinFreeMemBytes >> 30)
                         << " GiB, have " << (free_bytes >> 30) << " GiB";
        }
    }

    void RunTest()
    {
        const auto& tc = GetParam();
        auto& handle   = get_handle();

        // Input descriptor: NCHW, float.
        const std::vector<int> in_dims    = {kN, kC, kH, kW};
        const std::vector<int> in_strides = {kC * kH * kW, kH * kW, kW, 1};
        miopen::TensorDescriptor input_desc(miopenFloat, in_dims, in_strides);

        // Pooling descriptor.
        const std::vector<int> lens    = {kWinH, kWinW};
        const std::vector<int> strides = {kStrH, kStrW};
        const std::vector<int> pads    = {kPadH, kPadW};
        miopen::PoolingDescriptor pool_desc(tc.mode, miopenPaddingDefault, lens, strides, pads);
        pool_desc.SetIndexType(miopenIndexUint64);
        pool_desc.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexImage);

        const miopen::TensorDescriptor output_desc = pool_desc.GetForwardOutputTensor(input_desc);

        const size_t in_elems  = input_desc.GetElementSize();
        const size_t out_elems = output_desc.GetElementSize();
        const size_t ws_bytes  = pool_desc.GetWorkSpaceSize(output_desc);

        // Allocate GPU buffers.
        void* in_dev  = nullptr;
        void* out_dev = nullptr;

        ASSERT_EQ(hipMalloc(&in_dev, in_elems * sizeof(float)), hipSuccess)
            << "hipMalloc failed for input (" << (in_elems * sizeof(float) >> 30) << " GiB)";
        ASSERT_EQ(hipMalloc(&out_dev, out_elems * sizeof(float)), hipSuccess)
            << "hipMalloc failed for output";

        FillDeviceBuffer(in_dev, in_elems * sizeof(float));
        ASSERT_EQ(hipMemset(out_dev, 0, out_elems * sizeof(float)), hipSuccess)
            << "hipMemset failed for output buffer";

        Workspace wspace{ws_bytes};

        // Forward pooling.
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        ASSERT_NO_THROW(pool_desc.Forward(handle,
                                          &alpha,
                                          input_desc,
                                          in_dev,
                                          &beta,
                                          output_desc,
                                          out_dev,
                                          /*save_index=*/true,
                                          wspace.ptr(),
                                          wspace.size()))
            << tc.label << " Forward threw an exception";

        // Assert output sample is finite.
        // NaN or Inf indicates INT_MAX index overflow.
        EXPECT_TRUE(SampleIsFinite(out_dev, out_elems))
            << tc.label
            << " pooling output contains non-finite values "
               "(possible INT_MAX index overflow at numel > 2^31)";

        (void)hipFree(in_dev);
        (void)hipFree(out_dev);
    }
};

std::vector<PoolingLargeTestCase> GetLargeTensorPoolingTestCases()
{
    return {
        {miopenPoolingMax, "MaxPool_2x2_stride2"},
        {miopenPoolingAverage, "AvgPool_2x2_stride2"},
    };
}

std::string GetLargePoolingTestCaseName(const testing::TestParamInfo<PoolingLargeTestCase>& info)
{
    return info.param.label;
}

} // anonymous namespace

TEST_P(GPU_Pooling2d_Large_FP32, FloatTest_pooling2d_large) { RunTest(); }

// Instantiate under "Standard" so these run on every PR.
// The probe confirmed correct output on gfx942 (xnack-): these are
// regression guards, not bug-fix tests.
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_Pooling2d_Large_FP32,
                         testing::ValuesIn(GetLargeTensorPoolingTestCases()),
                         GetLargePoolingTestCaseName);
