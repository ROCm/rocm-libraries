// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Scale tests: workspace tensor byte count exceeds INT_MAX for GRU (FP32),
// vanilla RNN (FP32), and LSTM (FP32) — covering the training (fwd+bwd)
// paths and cell modes absent from lstm_large_workspace.cpp (ALMIOPEN-2149).
//
// Workspace formula (legacy API, no padding, non-dynamic algo):
//   ws_bytes = workspaceScale * nLayers * seqLen * batchSize
//              * hiddenSize * sizeof(dtype) * (bidir ? 2 : 1)
//
// workspaceScale per mode: RNN=1, GRU=4, LSTM=6
//
// Config used here (same spatial shape as lstm_large_workspace.cpp):
//   batchSize=1000, seqLen=600, inputSize=60, hiddenSize=128, nLayers=4,
//   bidirectional — dtype varied per suite.
//
//   GRU  FP32:  4*4 * 600000 * 128 * 4 * 2  ~  9.8 GB > INT_MAX
//   RNN  FP32:  1*4 * 600000 * 128 * 4 * 2  ~  2.5 GB > INT_MAX
//   LSTM FP32:  6*4 * 600000 * 128 * 4 * 2  ~ 14.7 GB > INT_MAX

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

struct LargeRnnScaleConfig
{
    int batch_per_step              = 1000;
    int seq_len                     = 600;
    int input_size                  = 60;
    int hidden_size                 = 128;
    int num_layers                  = 4;
    miopenRNNDirectionMode_t dir    = miopenRNNbidirection;
    miopenRNNInputMode_t input_mode = miopenRNNlinear;
    miopenRNNBiasMode_t bias_mode   = miopenRNNwithBias;
    miopenRNNAlgo_t algo            = miopenRNNdefault;

    // Varied per suite:
    miopenRNNMode_t mode   = miopenGRU;
    miopenDataType_t dtype = miopenFloat;
};

// Build the seq-length array of 2-D input tensor descriptors.
void MakeXDescs(const LargeRnnScaleConfig& cfg, TensorDescVecGuard& xDescs)
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

// Build the seq-length array of 2-D output tensor descriptors.
void MakeYDescs(const LargeRnnScaleConfig& cfg, TensorDescVecGuard& yDescs)
{
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
}

// Create and configure an RNN descriptor.  Caller must invoke
// DestroyInternalRnnDropoutDesc() around each miopenSetRNNDescriptor call to
// avoid leaking the internally allocated dropout descriptor.
void SetupRnnDesc(const LargeRnnScaleConfig& cfg, RNNDescGuard& rnn)
{
    ASSERT_EQ(rnn.getStatus(), miopenStatusSuccess);
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
}

// Returns the minimum device-memory requirement for a training run, rounded
// up to the next power of two with a headroom allowance.
std::size_t TrainingMemoryRequired(std::size_t ws_size,
                                   std::size_t rs_size,
                                   std::size_t x_bytes,
                                   std::size_t y_bytes,
                                   std::size_t w_size,
                                   std::size_t h_bytes)
{
    // 8 hidden-sized tensors: hx, cx, hy, cy (fwd) + dhx, dcx, dhy, dcy (bwd).
    const std::size_t raw_mem  = ws_size + rs_size + x_bytes + y_bytes + w_size + 8 * h_bytes;
    const std::size_t headroom = std::max<std::size_t>(1ULL << 30, raw_mem / 10);
    return std::bit_ceil(raw_mem + headroom);
}

} // namespace

// ---------------------------------------------------------------------------
// GRU — FP32 — training (forward + backward data)
// ---------------------------------------------------------------------------

struct GPU_GRU_LargeWorkspace_FP32 : public ::testing::TestWithParam<int>
{
};

TEST_P(GPU_GRU_LargeWorkspace_FP32, GetWorkspaceSizeOverflowsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenGRU;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs;
    MakeXDescs(cfg, xDescs);

    size_t ws_size = 0;
    auto status    = miopenGetRNNWorkspaceSize(&handle, rnn, cfg.seq_len, xDescs.data(), &ws_size);

    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_GT(ws_size, static_cast<size_t>(INT_MAX));

    DestroyInternalRnnDropoutDesc(rnn);
}

TEST_P(GPU_GRU_LargeWorkspace_FP32, ForwardTrainingSucceedsWhenWorkspaceExceedsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenGRU;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs, yDescs;
    MakeXDescs(cfg, xDescs);
    MakeYDescs(cfg, yDescs);

    const int directions = (cfg.dir == miopenRNNbidirection) ? 2 : 1;

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

    size_t rs_size = 0;
    ASSERT_EQ(
        miopenGetRNNTrainingReserveSize(&handle, rnn, cfg.seq_len, xDescs.data(), &rs_size),
        miopenStatusSuccess);

    size_t w_size = 0;
    ASSERT_EQ(miopenGetRNNParamsSize(&handle, rnn, xDescs[0], &w_size, cfg.dtype),
              miopenStatusSuccess);
    TensorDescGuard wDesc;
    ASSERT_EQ(wDesc.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNParamsDescriptor(&handle, rnn, xDescs[0], wDesc, cfg.dtype),
              miopenStatusSuccess);

    std::size_t x_bytes = 0, y_bytes = 0, h_bytes = 0;
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &x_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, yDescs.data(), &y_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNHiddenTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &h_bytes),
              miopenStatusSuccess);

    const std::size_t required_mem =
        TrainingMemoryRequired(ws_size, rs_size, x_bytes, y_bytes, w_size, h_bytes);
    const std::size_t device_mem = handle.GetGlobalMemorySize();
    if(device_mem < required_mem)
    {
        GTEST_SKIP() << "Insufficient device memory: need " << required_mem
                     << " bytes (rounded to next power of 2), device has " << device_mem;
    }

    Workspace x_buf{x_bytes};
    Workspace y_buf{y_bytes};
    Workspace w_buf{w_size};
    Workspace hx_buf{h_bytes};
    Workspace cx_buf{h_bytes}; // zeroed; ignored by GRU but must be a valid pointer
    Workspace hy_buf{h_bytes};
    Workspace cy_buf{h_bytes};
    Workspace ws_buf{ws_size};
    Workspace rs_buf{rs_size};

    ASSERT_EQ(hipMemset(w_buf.ptr(), 0, w_size), hipSuccess);
    ASSERT_EQ(hipMemset(hx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(cx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(x_buf.ptr(), 0, x_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(y_buf.ptr(), 0xFF, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(ws_buf.ptr(), 0, ws_size), hipSuccess);
    ASSERT_EQ(hipMemset(rs_buf.ptr(), 0, rs_size), hipSuccess);

    ASSERT_EQ(miopenRNNForwardTraining(&handle,
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
                                       ws_size,
                                       rs_buf.ptr(),
                                       rs_size),
              miopenStatusSuccess);

    DestroyInternalRnnDropoutDesc(rnn);
}

TEST_P(GPU_GRU_LargeWorkspace_FP32, BackwardDataSucceedsWhenWorkspaceExceedsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenGRU;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs, yDescs;
    MakeXDescs(cfg, xDescs);
    MakeYDescs(cfg, yDescs);

    const int directions = (cfg.dir == miopenRNNbidirection) ? 2 : 1;

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

    size_t rs_size = 0;
    ASSERT_EQ(
        miopenGetRNNTrainingReserveSize(&handle, rnn, cfg.seq_len, xDescs.data(), &rs_size),
        miopenStatusSuccess);

    size_t w_size = 0;
    ASSERT_EQ(miopenGetRNNParamsSize(&handle, rnn, xDescs[0], &w_size, cfg.dtype),
              miopenStatusSuccess);
    TensorDescGuard wDesc;
    ASSERT_EQ(wDesc.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNParamsDescriptor(&handle, rnn, xDescs[0], wDesc, cfg.dtype),
              miopenStatusSuccess);

    std::size_t x_bytes = 0, y_bytes = 0, h_bytes = 0;
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &x_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, yDescs.data(), &y_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNHiddenTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &h_bytes),
              miopenStatusSuccess);

    const std::size_t required_mem =
        TrainingMemoryRequired(ws_size, rs_size, x_bytes, y_bytes, w_size, h_bytes);
    const std::size_t device_mem = handle.GetGlobalMemorySize();
    if(device_mem < required_mem)
    {
        GTEST_SKIP() << "Insufficient device memory: need " << required_mem
                     << " bytes (rounded to next power of 2), device has " << device_mem;
    }

    // Run forward first to populate reserve space.
    Workspace x_buf{x_bytes};
    Workspace y_buf{y_bytes};
    Workspace w_buf{w_size};
    Workspace hx_buf{h_bytes};
    Workspace cx_buf{h_bytes};
    Workspace hy_buf{h_bytes};
    Workspace cy_buf{h_bytes};
    Workspace ws_buf{ws_size};
    Workspace rs_buf{rs_size};
    Workspace dx_buf{x_bytes};
    Workspace dy_buf{y_bytes};
    Workspace dhx_buf{h_bytes};
    Workspace dcx_buf{h_bytes};
    Workspace dhy_buf{h_bytes};
    Workspace dcy_buf{h_bytes};

    ASSERT_EQ(hipMemset(w_buf.ptr(), 0, w_size), hipSuccess);
    ASSERT_EQ(hipMemset(hx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(cx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(x_buf.ptr(), 0, x_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(y_buf.ptr(), 0, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(ws_buf.ptr(), 0, ws_size), hipSuccess);
    ASSERT_EQ(hipMemset(rs_buf.ptr(), 0, rs_size), hipSuccess);
    ASSERT_EQ(hipMemset(dy_buf.ptr(), 0, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(dhy_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(dcy_buf.ptr(), 0, h_bytes), hipSuccess);

    ASSERT_EQ(miopenRNNForwardTraining(&handle,
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
                                       ws_size,
                                       rs_buf.ptr(),
                                       rs_size),
              miopenStatusSuccess);

    ASSERT_EQ(miopenRNNBackwardData(&handle,
                                    rnn,
                                    cfg.seq_len,
                                    yDescs.data(),
                                    y_buf.ptr(),
                                    yDescs.data(),
                                    dy_buf.ptr(),
                                    hxDesc,
                                    dhy_buf.ptr(),
                                    hxDesc,
                                    dcy_buf.ptr(),
                                    wDesc,
                                    w_buf.ptr(),
                                    hxDesc,
                                    hx_buf.ptr(),
                                    hxDesc,
                                    cx_buf.ptr(),
                                    xDescs.data(),
                                    dx_buf.ptr(),
                                    hxDesc,
                                    dhx_buf.ptr(),
                                    hxDesc,
                                    dcx_buf.ptr(),
                                    ws_buf.ptr(),
                                    ws_size,
                                    rs_buf.ptr(),
                                    rs_size),
              miopenStatusSuccess);

    DestroyInternalRnnDropoutDesc(rnn);
}

INSTANTIATE_TEST_SUITE_P(Standard, GPU_GRU_LargeWorkspace_FP32, testing::Values(0));

// ---------------------------------------------------------------------------
// Vanilla RNN (tanh) — FP32 — training (forward + backward data)
// ---------------------------------------------------------------------------

struct GPU_RNNVanilla_LargeWorkspace_FP32 : public ::testing::TestWithParam<int>
{
};

TEST_P(GPU_RNNVanilla_LargeWorkspace_FP32, GetWorkspaceSizeOverflowsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenRNNTANH;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs;
    MakeXDescs(cfg, xDescs);

    size_t ws_size = 0;
    auto status    = miopenGetRNNWorkspaceSize(&handle, rnn, cfg.seq_len, xDescs.data(), &ws_size);

    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_GT(ws_size, static_cast<size_t>(INT_MAX));

    DestroyInternalRnnDropoutDesc(rnn);
}

TEST_P(GPU_RNNVanilla_LargeWorkspace_FP32, ForwardTrainingSucceedsWhenWorkspaceExceedsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenRNNTANH;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs, yDescs;
    MakeXDescs(cfg, xDescs);
    MakeYDescs(cfg, yDescs);

    const int directions = (cfg.dir == miopenRNNbidirection) ? 2 : 1;

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

    size_t rs_size = 0;
    ASSERT_EQ(
        miopenGetRNNTrainingReserveSize(&handle, rnn, cfg.seq_len, xDescs.data(), &rs_size),
        miopenStatusSuccess);

    size_t w_size = 0;
    ASSERT_EQ(miopenGetRNNParamsSize(&handle, rnn, xDescs[0], &w_size, cfg.dtype),
              miopenStatusSuccess);
    TensorDescGuard wDesc;
    ASSERT_EQ(wDesc.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNParamsDescriptor(&handle, rnn, xDescs[0], wDesc, cfg.dtype),
              miopenStatusSuccess);

    std::size_t x_bytes = 0, y_bytes = 0, h_bytes = 0;
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &x_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, yDescs.data(), &y_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNHiddenTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &h_bytes),
              miopenStatusSuccess);

    const std::size_t required_mem =
        TrainingMemoryRequired(ws_size, rs_size, x_bytes, y_bytes, w_size, h_bytes);
    const std::size_t device_mem = handle.GetGlobalMemorySize();
    if(device_mem < required_mem)
    {
        GTEST_SKIP() << "Insufficient device memory: need " << required_mem
                     << " bytes (rounded to next power of 2), device has " << device_mem;
    }

    Workspace x_buf{x_bytes};
    Workspace y_buf{y_bytes};
    Workspace w_buf{w_size};
    Workspace hx_buf{h_bytes};
    Workspace cx_buf{h_bytes}; // zeroed; ignored by vanilla RNN but must be a valid pointer
    Workspace hy_buf{h_bytes};
    Workspace cy_buf{h_bytes};
    Workspace ws_buf{ws_size};
    Workspace rs_buf{rs_size};

    ASSERT_EQ(hipMemset(w_buf.ptr(), 0, w_size), hipSuccess);
    ASSERT_EQ(hipMemset(hx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(cx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(x_buf.ptr(), 0, x_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(y_buf.ptr(), 0xFF, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(ws_buf.ptr(), 0, ws_size), hipSuccess);
    ASSERT_EQ(hipMemset(rs_buf.ptr(), 0, rs_size), hipSuccess);

    ASSERT_EQ(miopenRNNForwardTraining(&handle,
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
                                       ws_size,
                                       rs_buf.ptr(),
                                       rs_size),
              miopenStatusSuccess);

    DestroyInternalRnnDropoutDesc(rnn);
}

TEST_P(GPU_RNNVanilla_LargeWorkspace_FP32, BackwardDataSucceedsWhenWorkspaceExceedsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenRNNTANH;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs, yDescs;
    MakeXDescs(cfg, xDescs);
    MakeYDescs(cfg, yDescs);

    const int directions = (cfg.dir == miopenRNNbidirection) ? 2 : 1;

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

    size_t rs_size = 0;
    ASSERT_EQ(
        miopenGetRNNTrainingReserveSize(&handle, rnn, cfg.seq_len, xDescs.data(), &rs_size),
        miopenStatusSuccess);

    size_t w_size = 0;
    ASSERT_EQ(miopenGetRNNParamsSize(&handle, rnn, xDescs[0], &w_size, cfg.dtype),
              miopenStatusSuccess);
    TensorDescGuard wDesc;
    ASSERT_EQ(wDesc.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNParamsDescriptor(&handle, rnn, xDescs[0], wDesc, cfg.dtype),
              miopenStatusSuccess);

    std::size_t x_bytes = 0, y_bytes = 0, h_bytes = 0;
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &x_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, yDescs.data(), &y_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNHiddenTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &h_bytes),
              miopenStatusSuccess);

    const std::size_t required_mem =
        TrainingMemoryRequired(ws_size, rs_size, x_bytes, y_bytes, w_size, h_bytes);
    const std::size_t device_mem = handle.GetGlobalMemorySize();
    if(device_mem < required_mem)
    {
        GTEST_SKIP() << "Insufficient device memory: need " << required_mem
                     << " bytes (rounded to next power of 2), device has " << device_mem;
    }

    // Run forward first to populate reserve space.
    Workspace x_buf{x_bytes};
    Workspace y_buf{y_bytes};
    Workspace w_buf{w_size};
    Workspace hx_buf{h_bytes};
    Workspace cx_buf{h_bytes};
    Workspace hy_buf{h_bytes};
    Workspace cy_buf{h_bytes};
    Workspace ws_buf{ws_size};
    Workspace rs_buf{rs_size};
    Workspace dx_buf{x_bytes};
    Workspace dy_buf{y_bytes};
    Workspace dhx_buf{h_bytes};
    Workspace dcx_buf{h_bytes};
    Workspace dhy_buf{h_bytes};
    Workspace dcy_buf{h_bytes};

    ASSERT_EQ(hipMemset(w_buf.ptr(), 0, w_size), hipSuccess);
    ASSERT_EQ(hipMemset(hx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(cx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(x_buf.ptr(), 0, x_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(y_buf.ptr(), 0, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(ws_buf.ptr(), 0, ws_size), hipSuccess);
    ASSERT_EQ(hipMemset(rs_buf.ptr(), 0, rs_size), hipSuccess);
    ASSERT_EQ(hipMemset(dy_buf.ptr(), 0, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(dhy_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(dcy_buf.ptr(), 0, h_bytes), hipSuccess);

    ASSERT_EQ(miopenRNNForwardTraining(&handle,
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
                                       ws_size,
                                       rs_buf.ptr(),
                                       rs_size),
              miopenStatusSuccess);

    ASSERT_EQ(miopenRNNBackwardData(&handle,
                                    rnn,
                                    cfg.seq_len,
                                    yDescs.data(),
                                    y_buf.ptr(),
                                    yDescs.data(),
                                    dy_buf.ptr(),
                                    hxDesc,
                                    dhy_buf.ptr(),
                                    hxDesc,
                                    dcy_buf.ptr(),
                                    wDesc,
                                    w_buf.ptr(),
                                    hxDesc,
                                    hx_buf.ptr(),
                                    hxDesc,
                                    cx_buf.ptr(),
                                    xDescs.data(),
                                    dx_buf.ptr(),
                                    hxDesc,
                                    dhx_buf.ptr(),
                                    hxDesc,
                                    dcx_buf.ptr(),
                                    ws_buf.ptr(),
                                    ws_size,
                                    rs_buf.ptr(),
                                    rs_size),
              miopenStatusSuccess);

    DestroyInternalRnnDropoutDesc(rnn);
}

INSTANTIATE_TEST_SUITE_P(Standard, GPU_RNNVanilla_LargeWorkspace_FP32, testing::Values(0));

// ---------------------------------------------------------------------------
// LSTM — FP32 — training (forward + backward data)
// Existing lstm_large_workspace.cpp covers FP16 inference only;
// this suite adds FP32 training.
// ---------------------------------------------------------------------------

struct GPU_LSTM_LargeWorkspace_FP32 : public ::testing::TestWithParam<int>
{
};

TEST_P(GPU_LSTM_LargeWorkspace_FP32, GetWorkspaceSizeOverflowsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenLSTM;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs;
    MakeXDescs(cfg, xDescs);

    size_t ws_size = 0;
    auto status    = miopenGetRNNWorkspaceSize(&handle, rnn, cfg.seq_len, xDescs.data(), &ws_size);

    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_GT(ws_size, static_cast<size_t>(INT_MAX));

    DestroyInternalRnnDropoutDesc(rnn);
}

TEST_P(GPU_LSTM_LargeWorkspace_FP32, ForwardTrainingSucceedsWhenWorkspaceExceedsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenLSTM;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs, yDescs;
    MakeXDescs(cfg, xDescs);
    MakeYDescs(cfg, yDescs);

    const int directions = (cfg.dir == miopenRNNbidirection) ? 2 : 1;

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

    size_t rs_size = 0;
    ASSERT_EQ(
        miopenGetRNNTrainingReserveSize(&handle, rnn, cfg.seq_len, xDescs.data(), &rs_size),
        miopenStatusSuccess);

    size_t w_size = 0;
    ASSERT_EQ(miopenGetRNNParamsSize(&handle, rnn, xDescs[0], &w_size, cfg.dtype),
              miopenStatusSuccess);
    TensorDescGuard wDesc;
    ASSERT_EQ(wDesc.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNParamsDescriptor(&handle, rnn, xDescs[0], wDesc, cfg.dtype),
              miopenStatusSuccess);

    std::size_t x_bytes = 0, y_bytes = 0, h_bytes = 0;
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &x_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, yDescs.data(), &y_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNHiddenTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &h_bytes),
              miopenStatusSuccess);

    const std::size_t required_mem =
        TrainingMemoryRequired(ws_size, rs_size, x_bytes, y_bytes, w_size, h_bytes);
    const std::size_t device_mem = handle.GetGlobalMemorySize();
    if(device_mem < required_mem)
    {
        GTEST_SKIP() << "Insufficient device memory: need " << required_mem
                     << " bytes (rounded to next power of 2), device has " << device_mem;
    }

    Workspace x_buf{x_bytes};
    Workspace y_buf{y_bytes};
    Workspace w_buf{w_size};
    Workspace hx_buf{h_bytes};
    Workspace cx_buf{h_bytes};
    Workspace hy_buf{h_bytes};
    Workspace cy_buf{h_bytes};
    Workspace ws_buf{ws_size};
    Workspace rs_buf{rs_size};

    ASSERT_EQ(hipMemset(w_buf.ptr(), 0, w_size), hipSuccess);
    ASSERT_EQ(hipMemset(hx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(cx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(x_buf.ptr(), 0, x_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(y_buf.ptr(), 0xFF, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(ws_buf.ptr(), 0, ws_size), hipSuccess);
    ASSERT_EQ(hipMemset(rs_buf.ptr(), 0, rs_size), hipSuccess);

    ASSERT_EQ(miopenRNNForwardTraining(&handle,
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
                                       ws_size,
                                       rs_buf.ptr(),
                                       rs_size),
              miopenStatusSuccess);

    DestroyInternalRnnDropoutDesc(rnn);
}

TEST_P(GPU_LSTM_LargeWorkspace_FP32, BackwardDataSucceedsWhenWorkspaceExceedsInt)
{
    auto&& handle = get_handle();
    LargeRnnScaleConfig cfg;
    cfg.mode  = miopenLSTM;
    cfg.dtype = miopenFloat;

    RNNDescGuard rnn;
    SetupRnnDesc(cfg, rnn);

    TensorDescVecGuard xDescs, yDescs;
    MakeXDescs(cfg, xDescs);
    MakeYDescs(cfg, yDescs);

    const int directions = (cfg.dir == miopenRNNbidirection) ? 2 : 1;

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

    size_t rs_size = 0;
    ASSERT_EQ(
        miopenGetRNNTrainingReserveSize(&handle, rnn, cfg.seq_len, xDescs.data(), &rs_size),
        miopenStatusSuccess);

    size_t w_size = 0;
    ASSERT_EQ(miopenGetRNNParamsSize(&handle, rnn, xDescs[0], &w_size, cfg.dtype),
              miopenStatusSuccess);
    TensorDescGuard wDesc;
    ASSERT_EQ(wDesc.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNParamsDescriptor(&handle, rnn, xDescs[0], wDesc, cfg.dtype),
              miopenStatusSuccess);

    std::size_t x_bytes = 0, y_bytes = 0, h_bytes = 0;
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &x_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, yDescs.data(), &y_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNHiddenTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &h_bytes),
              miopenStatusSuccess);

    const std::size_t required_mem =
        TrainingMemoryRequired(ws_size, rs_size, x_bytes, y_bytes, w_size, h_bytes);
    const std::size_t device_mem = handle.GetGlobalMemorySize();
    if(device_mem < required_mem)
    {
        GTEST_SKIP() << "Insufficient device memory: need " << required_mem
                     << " bytes (rounded to next power of 2), device has " << device_mem;
    }

    // Run forward first to populate reserve space.
    Workspace x_buf{x_bytes};
    Workspace y_buf{y_bytes};
    Workspace w_buf{w_size};
    Workspace hx_buf{h_bytes};
    Workspace cx_buf{h_bytes};
    Workspace hy_buf{h_bytes};
    Workspace cy_buf{h_bytes};
    Workspace ws_buf{ws_size};
    Workspace rs_buf{rs_size};
    Workspace dx_buf{x_bytes};
    Workspace dy_buf{y_bytes};
    Workspace dhx_buf{h_bytes};
    Workspace dcx_buf{h_bytes};
    Workspace dhy_buf{h_bytes};
    Workspace dcy_buf{h_bytes};

    ASSERT_EQ(hipMemset(w_buf.ptr(), 0, w_size), hipSuccess);
    ASSERT_EQ(hipMemset(hx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(cx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(x_buf.ptr(), 0, x_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(y_buf.ptr(), 0, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(ws_buf.ptr(), 0, ws_size), hipSuccess);
    ASSERT_EQ(hipMemset(rs_buf.ptr(), 0, rs_size), hipSuccess);
    ASSERT_EQ(hipMemset(dy_buf.ptr(), 0, y_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(dhy_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(dcy_buf.ptr(), 0, h_bytes), hipSuccess);

    ASSERT_EQ(miopenRNNForwardTraining(&handle,
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
                                       ws_size,
                                       rs_buf.ptr(),
                                       rs_size),
              miopenStatusSuccess);

    ASSERT_EQ(miopenRNNBackwardData(&handle,
                                    rnn,
                                    cfg.seq_len,
                                    yDescs.data(),
                                    y_buf.ptr(),
                                    yDescs.data(),
                                    dy_buf.ptr(),
                                    hxDesc,
                                    dhy_buf.ptr(),
                                    hxDesc,
                                    dcy_buf.ptr(),
                                    wDesc,
                                    w_buf.ptr(),
                                    hxDesc,
                                    hx_buf.ptr(),
                                    hxDesc,
                                    cx_buf.ptr(),
                                    xDescs.data(),
                                    dx_buf.ptr(),
                                    hxDesc,
                                    dhx_buf.ptr(),
                                    hxDesc,
                                    dcx_buf.ptr(),
                                    ws_buf.ptr(),
                                    ws_size,
                                    rs_buf.ptr(),
                                    rs_size),
              miopenStatusSuccess);

    DestroyInternalRnnDropoutDesc(rnn);
}

INSTANTIATE_TEST_SUITE_P(Standard, GPU_LSTM_LargeWorkspace_FP32, testing::Values(0));
