/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Scale test: CTC loss with numel (T * N * C) > INT_MAX.
//
// JIRA: ALMIOPEN-2148
//
// Motivation
// ----------
// MIOpen's CTC kernel passes dimension products through plain `int` locals
// (class_sz * batch_size * max_time_step in ctc.cpp).  When T*N*C exceeds
// 2^31-1 those multiplications wrap, producing negative workspace offsets and
// silent index-out-of-bounds access.  This test exercises the boundary with
// the smallest shape whose logit-tensor element count exceeds INT_MAX:
//
//   T=1000, N=2048, C=1049  →  T*N*C = 2 148 352 000  >  2^31-1 = 2 147 483 647
//
// Memory requirement: T*N*C * sizeof(float) ≈ 8.59 GiB for the logits tensor
// alone, plus workspace.  The test queries free GPU memory and calls
// GTEST_SKIP() when headroom is insufficient so it is safe to run on
// smaller devices.
//
// No CPU reference comparison is performed: host allocation of 8.59 GiB per
// buffer would be prohibitive.  Instead, correctness is checked by asserting
// that every loss value is finite (not NaN / ±Inf), which would fail
// immediately when INT_MAX overflow corrupts workspace offsets.

#include <miopen/ctc.hpp>

#include <hip/hip_runtime.h>

#include "get_handle.hpp"
#include "workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

// Shape whose logit numel = T * N * C just exceeds INT_MAX.
// T=1000, N=2048, C=1049 → 2 148 352 000 > 2 147 483 647 = INT_MAX.
// These dimensions represent a plausible large-vocabulary ASR model:
//   T  = sequence length (time steps)
//   N  = mini-batch size
//   C  = vocabulary size (not counting blank; class_sz = C+1 internally)
constexpr int kT = 1000; // max time steps
constexpr int kN = 2048; // batch size
constexpr int kC = 1048; // num_class parameter (class_sz = kC+1 = 1049)

// Label / input-length parameters kept small so alpha/workspace remains
// manageable.  The overflow risk is in the *logit* dimensions, not the
// label-path bookkeeping.
constexpr int kLabelLen = 2;  // labels per sample (short utterances)
constexpr int kInputLen = 10; // actual sequence length used per sample

static_assert(static_cast<long long>(kT) * kN * (kC + 1) > 2147483647LL,
              "numel must exceed INT_MAX");
static_assert(kInputLen >= kLabelLen * 2 + 1,
              "inputLen must satisfy CTC constraint: inputLen >= 2*labelLen+1");

// Required free GPU memory in bytes before the test is allowed to run.
// Logits: T*N*(C+1)*sizeof(float) = 1000*2048*1049*4 ≈ 8.59 GiB
// Gradients: same  ≈ 8.59 GiB
// Workspace: estimated as a fraction of logits ≈ ~1 GiB
// Total headroom requested: 19 GiB (conservative; skip on smaller GPUs).
constexpr size_t kMinFreeMemBytes = 19ULL * 1024 * 1024 * 1024;

struct GPU_CTCLargeScale_FP32 : public ::testing::Test
{
    void SetUp() override
    {
        size_t free_bytes        = 0;
        size_t total_bytes       = 0;
        const hipError_t mem_err = hipMemGetInfo(&free_bytes, &total_bytes);
        if(mem_err != hipSuccess)
        {
            GTEST_SKIP() << "hipMemGetInfo failed (err=" << mem_err
                         << "); skipping large-scale CTC test";
        }
        if(free_bytes < kMinFreeMemBytes)
        {
            GTEST_SKIP() << "Insufficient free GPU memory for large-scale CTC test: need "
                         << (kMinFreeMemBytes >> 30) << " GiB, have " << (free_bytes >> 30)
                         << " GiB";
        }
    }

    void Run()
    {
        auto& handle = get_handle();

        const int class_sz      = kC + 1; // MIOpen adds 1 for blank internally
        const int batch_sz      = kN;
        const int max_time_step = kT;

        // Build TensorDescriptors for probs and gradients: shape [T, N, C+1].
        const std::vector<int> dims    = {max_time_step, batch_sz, class_sz};
        const std::vector<int> strides = {batch_sz * class_sz, class_sz, 1};

        miopen::TensorDescriptor probsDesc(miopenFloat, dims, strides);
        miopen::TensorDescriptor gradsDesc(miopenFloat, dims, strides);

        // label lengths and input lengths.
        const std::vector<int> labelLengths(batch_sz, kLabelLen);
        std::vector<int> inputLengths(batch_sz, kInputLen);

        // Labels: two non-blank, non-consecutive classes per sample.
        const size_t total_label_sz =
            static_cast<size_t>(batch_sz) * static_cast<size_t>(kLabelLen);
        std::vector<int> labels(total_label_sz);
        for(size_t i = 0; i < total_label_sz; ++i)
            labels[i] = static_cast<int>(1 + (i % static_cast<size_t>(kC - 1)));

        // Descriptor: softmax applied by MIOpen, blank_label_id=0.
        miopen::CTCLossDescriptor ctcDesc;
        ctcDesc.dataType            = miopenFloat;
        ctcDesc.apply_softmax_layer = true;
        ctcDesc.blank_label_id      = 0;

        // Query workspace size.
        size_t wsp_bytes = 0;
        ASSERT_NO_THROW(wsp_bytes = ctcDesc.GetCTCLossWorkspaceSize(handle,
                                                                    probsDesc,
                                                                    gradsDesc,
                                                                    labels.data(),
                                                                    labelLengths.data(),
                                                                    inputLengths.data(),
                                                                    miopenCTCLossAlgo_t(0)));
        ASSERT_GT(wsp_bytes, 0u) << "GetCTCLossWorkspaceSize returned zero";

        // Allocate GPU buffers.
        const size_t logit_elems = static_cast<size_t>(max_time_step) *
                                   static_cast<size_t>(batch_sz) * static_cast<size_t>(class_sz);
        const size_t logit_bytes = logit_elems * sizeof(float);

        void* probs_dev  = nullptr;
        void* grads_dev  = nullptr;
        void* losses_dev = nullptr;

        ASSERT_EQ(hipMalloc(&probs_dev, logit_bytes), hipSuccess)
            << "hipMalloc failed for probs (" << (logit_bytes >> 30) << " GiB)";
        ASSERT_EQ(hipMalloc(&grads_dev, logit_bytes), hipSuccess)
            << "hipMalloc failed for grads (" << (logit_bytes >> 30) << " GiB)";
        ASSERT_EQ(hipMalloc(&losses_dev, static_cast<size_t>(batch_sz) * sizeof(float)), hipSuccess)
            << "hipMalloc failed for losses";

        // Zero-fill all three buffers.  All-zero logits are valid inputs:
        // softmax(0,...,0) = uniform(1/class_sz,...) — finite and non-degenerate.
        ASSERT_EQ(hipMemset(probs_dev, 0, logit_bytes), hipSuccess);
        ASSERT_EQ(hipMemset(grads_dev, 0, logit_bytes), hipSuccess);
        ASSERT_EQ(hipMemset(losses_dev, 0, static_cast<size_t>(batch_sz) * sizeof(float)),
                  hipSuccess);

        Workspace wspace{wsp_bytes};

        // Run forward CTC loss.
        ASSERT_NO_THROW(ctcDesc.CTCLoss(handle,
                                        probsDesc,
                                        probs_dev,
                                        labels.data(),
                                        labelLengths.data(),
                                        inputLengths.data(),
                                        losses_dev,
                                        gradsDesc,
                                        grads_dev,
                                        miopenCTCLossAlgo_t(0),
                                        wspace.ptr(),
                                        wspace.size()));

        // Copy losses back and assert all are finite.
        // NaN or Inf indicates workspace index overflow or uninitialized memory.
        std::vector<float> losses_host(static_cast<size_t>(batch_sz), 0.0f);
        ASSERT_EQ(hipMemcpy(losses_host.data(),
                            losses_dev,
                            static_cast<size_t>(batch_sz) * sizeof(float),
                            hipMemcpyDeviceToHost),
                  hipSuccess);

        for(int i = 0; i < batch_sz; ++i)
        {
            EXPECT_TRUE(std::isfinite(losses_host[i]))
                << "Loss at batch index " << i << " is not finite: " << losses_host[i]
                << " — likely caused by INT_MAX overflow in workspace index computation";
        }

        // Clean up.
        hipFree(probs_dev);
        hipFree(grads_dev);
        hipFree(losses_dev);
    }
};

} // namespace

// Named Scale to indicate this belongs to large-scale / nightly coverage.
// The test is excluded from Smoke and Standard tiers via test_categories.yaml.
// Run manually with:
//   ./miopen_gtest --gtest_filter='GPU_CTCLargeScale_FP32.FullScale'
// or via MIOpenDriver:
//   MIOpenDriver ctc -T 1000 -N 2048 -C 1048 --forw 1 -V 0 -t 1

TEST_F(GPU_CTCLargeScale_FP32, FullScale) { this->Run(); }
