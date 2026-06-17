// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Regression test: workspace tensor element count exceeds INT_MAX for fp16 LSTM.

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include <hip/hip_runtime.h>

#include <bit>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "get_handle.hpp"
#include "gtest_desc_guard.hpp"
#include "../workspace.hpp"

namespace {

// Smallest config that pushes fp16 workspace element count past INT_MAX (~5.5e9 elts).
struct LargeRnnConfig
{
    int batch_per_step              = 1000;
    int seq_len                     = 600;
    int input_size                  = 60;
    int hidden_size                 = 128;
    int num_layers                  = 4;
    miopenRNNDirectionMode_t dir    = miopenRNNbidirection;
    miopenRNNMode_t mode            = miopenLSTM;
    miopenRNNInputMode_t input_mode = miopenRNNlinear;
    miopenRNNBiasMode_t bias_mode   = miopenRNNwithBias;
    miopenRNNAlgo_t algo            = miopenRNNdefault;
    miopenDataType_t dtype          = miopenHalf;
};

void MakeXDescs(const LargeRnnConfig& cfg, TensorDescVecGuard& xDescs)
{
    xDescs.descs.assign(cfg.seq_len, nullptr);
    for(int i = 0; i < cfg.seq_len; ++i)
    {
        ASSERT_EQ(miopenCreateTensorDescriptor(&xDescs[i]), miopenStatusSuccess);
        std::array<int, 2> lens = {cfg.batch_per_step, cfg.input_size};
        ASSERT_EQ(miopenSetTensorDescriptor(xDescs[i], cfg.dtype, 2, lens.data(), nullptr),
                  miopenStatusSuccess);
    }
}

} // namespace

// Parameterized only so that INSTANTIATE_TEST_SUITE_P can put these tests under the
// "Standard/" prefix used by test_categories.yaml; the int param itself is unused.
struct GPU_LSTM_LargeWorkspace_FP16 : public ::testing::TestWithParam<int>
{
};

// Tripwire: if a future MIOpen change shrinks the workspace below INT_MAX bytes for this
// config, this assertion fires and someone needs to choose a larger LargeRnnConfig.
TEST_P(GPU_LSTM_LargeWorkspace_FP16, GetWorkspaceSizeOverflowsInt)
{
    auto&& handle = get_handle();
    LargeRnnConfig cfg;

    RNNDescGuard rnn;
    ASSERT_EQ(rnn.getStatus(), miopenStatusSuccess);
    // Free the internal DropoutDescriptor that miopenCreateRNNDescriptor allocated;
    // miopenSetRNNDescriptor below will overwrite the field and otherwise leak it.
    DestroyInternalRnnDropoutDesc(rnn);
    ASSERT_EQ(miopenSetRNNDescriptor(rnn,
                                     cfg.hidden_size,
                                     cfg.num_layers,
                                     cfg.input_mode,
                                     cfg.dir,
                                     cfg.mode,
                                     cfg.bias_mode,
                                     cfg.algo,
                                     cfg.dtype),
              miopenStatusSuccess);

    TensorDescVecGuard xDescs;
    MakeXDescs(cfg, xDescs);

    size_t ws_size = 0;
    auto status    = miopenGetRNNWorkspaceSize(&handle, rnn, cfg.seq_len, xDescs.data(), &ws_size);

    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_GT(ws_size, static_cast<size_t>(INT_MAX));

    // miopenDestroyRNNDescriptor does not free the internal DropoutDescriptor that
    // miopenSetRNNDescriptor (8-arg form, no user-owned dropout) allocated.
    DestroyInternalRnnDropoutDesc(rnn);
}

TEST_P(GPU_LSTM_LargeWorkspace_FP16, ForwardInferenceSucceedsWhenWorkspaceExceedsInt)
{
    auto&& handle = get_handle();
    LargeRnnConfig cfg;

    RNNDescGuard rnn;
    ASSERT_EQ(rnn.getStatus(), miopenStatusSuccess);
    // Free the internal DropoutDescriptor that miopenCreateRNNDescriptor allocated;
    // miopenSetRNNDescriptor below will overwrite the field and otherwise leak it.
    DestroyInternalRnnDropoutDesc(rnn);
    ASSERT_EQ(miopenSetRNNDescriptor(rnn,
                                     cfg.hidden_size,
                                     cfg.num_layers,
                                     cfg.input_mode,
                                     cfg.dir,
                                     cfg.mode,
                                     cfg.bias_mode,
                                     cfg.algo,
                                     cfg.dtype),
              miopenStatusSuccess);

    TensorDescVecGuard xDescs;
    TensorDescVecGuard yDescs;
    MakeXDescs(cfg, xDescs);

    const int directions = (cfg.dir == miopenRNNbidirection) ? 2 : 1;
    const int y_vec      = cfg.hidden_size * directions;
    yDescs.descs.assign(cfg.seq_len, nullptr);
    for(int i = 0; i < cfg.seq_len; ++i)
    {
        ASSERT_EQ(miopenCreateTensorDescriptor(&yDescs[i]), miopenStatusSuccess);
        std::array<int, 2> lens = {cfg.batch_per_step, y_vec};
        ASSERT_EQ(miopenSetTensorDescriptor(yDescs[i], cfg.dtype, 2, lens.data(), nullptr),
                  miopenStatusSuccess);
    }

    TensorDescGuard hxDesc;
    ASSERT_EQ(hxDesc.getStatus(), miopenStatusSuccess);
    {
        std::array<int, 3> lens = {
            cfg.num_layers * directions, cfg.batch_per_step, cfg.hidden_size};
        ASSERT_EQ(miopenSetTensorDescriptor(hxDesc, cfg.dtype, 3, lens.data(), nullptr),
                  miopenStatusSuccess);
    }

    size_t ws_size = 0;
    ASSERT_EQ(miopenGetRNNWorkspaceSize(&handle, rnn, cfg.seq_len, xDescs.data(), &ws_size),
              miopenStatusSuccess);
    ASSERT_GT(ws_size, static_cast<size_t>(INT_MAX));

    size_t w_size = 0;
    ASSERT_EQ(miopenGetRNNParamsSize(&handle, rnn, xDescs[0], &w_size, cfg.dtype),
              miopenStatusSuccess);
    TensorDescGuard wDesc;
    ASSERT_EQ(wDesc.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNParamsDescriptor(&handle, rnn, xDescs[0], wDesc, cfg.dtype),
              miopenStatusSuccess);

    // Use the official MIOpen sizing APIs so that buffer sizes account for all internal
    // alignment / padding requirements.
    std::size_t x_bytes = 0;
    std::size_t y_bytes = 0;
    std::size_t h_bytes = 0;
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &x_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, yDescs.data(), &y_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNHiddenTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &h_bytes),
              miopenStatusSuccess);

    // Add headroom over the raw tensor sum for runtime/library reservations,
    // allocator fragmentation, and (on consumer cards) the display compositor.
    // Use max(+1 GiB, +10%) to cover both the absolute and the proportional
    // components, then round up to the next power of two.
    const std::size_t raw_mem      = ws_size + x_bytes + y_bytes + w_size + 4 * h_bytes;
    const std::size_t headroom     = std::max<std::size_t>(1ULL << 30, raw_mem / 10);
    const std::size_t required_mem = std::bit_ceil(raw_mem + headroom);
    const std::size_t device_mem   = handle.GetGlobalMemorySize();
    if(device_mem < required_mem)
    {
        GTEST_SKIP() << "Insufficient device memory: need " << required_mem
                     << " bytes (rounded up to next power of 2), device has " << device_mem;
    }

    Workspace x_buf{x_bytes};
    Workspace y_buf{y_bytes};
    Workspace w_buf{w_size};
    Workspace hx_buf{h_bytes};
    Workspace cx_buf{h_bytes};
    Workspace hy_buf{h_bytes};
    Workspace cy_buf{h_bytes};
    Workspace ws_buf{ws_size};

    // Zero weights, hidden state, and inputs so the analytic LSTM output is exactly zero
    // (sigmoid(0)*tanh(0) = 0). Pre-fill y with a sentinel so we can detect any later
    // "silent" CopyTensor failure that leaves output regions unwritten.
    ASSERT_EQ(hipMemset(w_buf.ptr(), 0, w_size), hipSuccess);
    ASSERT_EQ(hipMemset(hx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(cx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(x_buf.ptr(), 0, x_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(y_buf.ptr(), 0xFF, y_bytes), hipSuccess);

    auto status = miopenRNNForwardInference(&handle,
                                            rnn,
                                            cfg.seq_len,
                                            xDescs.data(),
                                            x_buf.ptr(),
                                            hxDesc,
                                            hx_buf.ptr(),
                                            hxDesc,
                                            cx_buf.ptr(),
                                            wDesc,
                                            w_buf.ptr(),
                                            yDescs.data(),
                                            y_buf.ptr(),
                                            hxDesc,
                                            hy_buf.ptr(),
                                            hxDesc,
                                            cy_buf.ptr(),
                                            ws_buf.ptr(),
                                            ws_size);

    ASSERT_EQ(status, miopenStatusSuccess);

    // With zero inputs/weights/biases/hidden state, every fp16 output element should be
    // exactly +0.0. Any nonzero byte indicates either silent corruption (e.g. a truncated
    // CopyTensor offset writing stale workspace bytes into y) or an unwritten region
    // still holding the 0xFF sentinel.
    std::vector<std::uint16_t> y_host(y_bytes / sizeof(std::uint16_t));
    ASSERT_EQ(hipMemcpy(y_host.data(), y_buf.ptr(), y_bytes, hipMemcpyDeviceToHost), hipSuccess);
    for(size_t i = 0; i < y_host.size(); ++i)
    {
        ASSERT_EQ(y_host[i], 0u) << "Nonzero fp16 bits at element " << i << "/" << y_host.size()
                                 << " (offset " << i * sizeof(std::uint16_t) << " bytes)";
    }

    // miopenDestroyRNNDescriptor does not free the internal DropoutDescriptor that
    // miopenSetRNNDescriptor (8-arg form, no user-owned dropout) allocated.
    DestroyInternalRnnDropoutDesc(rnn);
}

INSTANTIATE_TEST_SUITE_P(Standard, GPU_LSTM_LargeWorkspace_FP16, testing::Values(0));
