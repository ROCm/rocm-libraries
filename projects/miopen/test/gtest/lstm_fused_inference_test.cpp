/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

// Correctness coverage for the experimental fused single-layer LSTM inference
// path, gated by MIOPEN_DEBUG_RNN_FUSED_INFERENCE (see src/ocl/rnnocl.cpp).
//
// The fused path only replaces the per-timestep hidden-state recurrence inside
// RNNForwardInferencePacked; the batched input projection (the GEMM that
// dominates throughput) is identical to the stock path. The natural oracle is
// therefore the stock path itself: run miopenRNNForwardInference twice on the
// same inputs in the same process -- once with the flag off (reference) and once
// with it on (fused) -- and require the two GPU outputs to agree to a tight
// tolerance. A bug in the fused recurrence (gate order, cell update, direction
// offset, batch indexing) shifts the fused output and trips the assertion, so
// the comparison has teeth that the float-epsilon CPU reference in lstm.hpp does
// not. The shared LSTM_test base is reused only for its tensor-setup fields.

#include "lstm.hpp"
#include "gtest_common.hpp"
#include "rnn_util.hpp"
#include "workspace.hpp"
#include "verify.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_RNN_FUSED_INFERENCE)

namespace {

struct FusedInferCase
{
    int dirMode;   // 0 = unidirectional, 1 = bidirectional
    int batchSize; // uniform batch across the sequence
    int seqLength;
    int inVecLen;
    int hiddenSize;

    friend std::ostream& operator<<(std::ostream& os, const FusedInferCase& tc)
    {
        return os << "dir-mode:" << tc.dirMode << " batch:" << tc.batchSize
                  << " seq:" << tc.seqLength << " in-vec:" << tc.inVecLen
                  << " hidden:" << tc.hiddenSize;
    }
};

std::vector<FusedInferCase> GetTestCases()
{
    return std::vector{
        // clang-format off
        //             dir batch seq invec hidden
        // N=1 launch-latency path that the fusion targets (Kokoro decoder shape).
        FusedInferCase{0,  1,    16,  128,  128},
        FusedInferCase{1,  1,    16,  128,  128},
        FusedInferCase{0,  1,    25,  256,  256},
        FusedInferCase{1,  1,    25,  256,  256},
        // Non-square in/hidden, still N=1.
        FusedInferCase{0,  1,    8,   64,   128},
        FusedInferCase{1,  1,    8,   64,   128},
        // batch > 1 (the fused path also handles a uniform batch).
        FusedInferCase{0,  4,    10,  64,   64},
        FusedInferCase{1,  4,    10,  64,   64},
        FusedInferCase{0,  16,   5,   32,   48},
        FusedInferCase{1,  16,   5,   32,   48},
        // single timestep edge case.
        FusedInferCase{0,  2,    1,   32,   32},
        FusedInferCase{1,  2,    1,   32,   32}
        // clang-format on
    };
}

template <typename T>
struct GPU_LSTM_fused_infer_test : LSTM_test<T>, testing::TestWithParam<FusedInferCase>
{
protected:
    void SetUp() override
    {
        const auto tc = GetParam();

        this->dataType   = miopen_type<T>{};
        this->batchSize  = tc.batchSize;
        this->seqLength  = tc.seqLength;
        this->batchSeq   = std::vector<int>(tc.seqLength, tc.batchSize);
        this->inVecLen   = tc.inVecLen;
        this->hiddenSize = tc.hiddenSize;
        this->numLayers  = 1;
        this->inputMode  = 0; // linear input (skip mode is unsupported by the fused path)
        this->biasMode   = 1; // with bias
        this->dirMode    = tc.dirMode;
        this->algoMode   = 0; // miopenRNNdefault
        this->nohx       = false;
        this->nocx       = false;
        this->nohy       = false;
        this->nocy       = false;
    }

    using InferenceResult = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>>;

    void RunInference(miopenRNNDescriptor_t rnnDesc,
                      const std::vector<T>& input,
                      const std::vector<T>& hx,
                      const std::vector<T>& cx,
                      const std::vector<T>& weights,
                      InferenceResult& result)
    {
        auto&& handle = get_handle();

        const auto inVecReal = (this->inputMode != 0) ? this->hiddenSize : this->inVecLen;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(inputCPPDescs, inputDescs, this->batchSeq, inVecReal, this->dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              this->batchSeq,
                              this->hiddenSize * ((this->dirMode != 0) ? 2 : 1),
                              this->dataType);

        size_t workspace_size = 0;
        ASSERT_EQ(miopenStatusSuccess,
                  miopenGetRNNWorkspaceSize(
                      &handle, rnnDesc, this->seqLength, inputDescs.data(), &workspace_size));
        Workspace wspace{workspace_size};

        size_t out_sz = 0;
        ASSERT_EQ(miopenStatusSuccess,
                  miopenGetRNNInputTensorSize(
                      &handle, rnnDesc, this->seqLength, outputDescs.data(), &out_sz));
        std::vector<T> output(out_sz / sizeof(T));

        std::vector<int> hlens{
            this->numLayers * ((this->dirMode != 0) ? 2 : 1), this->batchSeq[0], this->hiddenSize};
        miopen::TensorDescriptor hiddenDesc(this->dataType, hlens);

        std::vector<int> wlen{static_cast<int>(weights.size())};
        miopen::TensorDescriptor weightDesc(this->dataType, wlen);

        std::vector<T> hy(hx.size(), T(0));
        std::vector<T> cy(cx.size(), T(0));

        auto input_dev   = handle.Write(input);
        auto output_dev  = handle.Write(output);
        auto weights_dev = handle.Write(weights);
        auto hx_dev      = handle.Write(hx);
        auto cx_dev      = handle.Write(cx);
        auto hy_dev      = handle.Write(hy);
        auto cy_dev      = handle.Write(cy);

        ASSERT_EQ(miopenStatusSuccess,
                  miopenRNNForwardInference(&handle,
                                            rnnDesc,
                                            this->seqLength,
                                            inputDescs.data(),
                                            input_dev.get(),
                                            &hiddenDesc,
                                            hx_dev.get(),
                                            &hiddenDesc,
                                            cx_dev.get(),
                                            &weightDesc,
                                            weights_dev.get(),
                                            outputDescs.data(),
                                            output_dev.get(),
                                            &hiddenDesc,
                                            hy_dev.get(),
                                            &hiddenDesc,
                                            cy_dev.get(),
                                            wspace.ptr(),
                                            wspace.size()));

        result = {handle.Read<T>(output_dev, output.size()),
                  handle.Read<T>(hy_dev, hy.size()),
                  handle.Read<T>(cy_dev, cy.size())};
    }

    void RunFusedInference()
    {
        auto&& handle = get_handle();
        if(!handle.CooperativeLaunchSupported())
            GTEST_SKIP() << "Cooperative launch is not supported";

        RNNDescGuard rnnDesc;
        DestroyInternalRnnDropoutDesc(rnnDesc);
        ASSERT_EQ(miopenStatusSuccess,
                  miopenSetRNNDescriptor(rnnDesc,
                                         this->hiddenSize,
                                         this->numLayers,
                                         miopenRNNInputMode_t(this->inputMode),
                                         miopenRNNDirectionMode_t(this->dirMode),
                                         miopenLSTM,
                                         miopenRNNBiasMode_t(this->biasMode),
                                         miopenRNNAlgo_t(this->algoMode),
                                         this->dataType));

        const auto inVecReal = (this->inputMode != 0) ? this->hiddenSize : this->inVecLen;

        const std::size_t in_sz = getSuperTensorSize(this->batchSeq,
                                                     this->seqLength,
                                                     inVecReal,
                                                     this->hiddenSize,
                                                     0,
                                                     this->dirMode != 0,
                                                     true,
                                                     0);
        std::vector<T> input(in_sz);
        for(auto& v : input)
            v = prng::gen_descreet_unsigned<T>(this->dataScale, 100);

        const std::size_t hx_sz = ((this->dirMode != 0) ? 2ULL : 1ULL) * this->hiddenSize *
                                  this->batchSize * this->numLayers;
        std::vector<T> hx(hx_sz);
        std::vector<T> cx(hx_sz);
        for(auto& v : hx)
            v = prng::gen_descreet_unsigned<T>(this->dataScale, 100);
        for(auto& v : cx)
            v = prng::gen_descreet_unsigned<T>(this->dataScale, 100);

        std::vector<int> inlens{this->batchSeq.at(0), inVecReal};
        auto firstInputDesc = miopen::TensorDescriptor(this->dataType, inlens);
        size_t wei_bytes    = 0;
        ASSERT_EQ(
            miopenStatusSuccess,
            miopenGetRNNParamsSize(&handle, rnnDesc, &firstInputDesc, &wei_bytes, this->dataType));
        std::vector<T> weights(wei_bytes / sizeof(T));
        for(auto& v : weights)
            v = prng::gen_descreet_uniform_sign<T>(this->dataScale, 100);

        const auto fused_kernel_name = "RNNFusedLSTMInfer";
        const auto network_config = "rnnfusedinf-fp32-h" + std::to_string(this->hiddenSize) + "-b" +
                                    std::to_string(handle.GetMaxComputeUnits()) + "-bi" +
                                    std::to_string(this->dirMode != 0 ? 2 : 1);
        handle.ClearKernels(fused_kernel_name, network_config);

        // Reference: stock per-timestep path (flag explicitly off).
        InferenceResult ref;
        {
            ScopedEnvironment<bool> fused_env(MIOPEN_DEBUG_RNN_FUSED_INFERENCE, false);
            RunInference(rnnDesc, input, hx, cx, weights, ref);
        }
        ASSERT_TRUE(handle.GetKernelsImpl(fused_kernel_name, network_config).empty());

        // Under test: fused cooperative-grid path (flag on).
        InferenceResult fused;
        {
            ScopedEnvironment<bool> fused_env(MIOPEN_DEBUG_RNN_FUSED_INFERENCE, true);
            RunInference(rnnDesc, input, hx, cx, weights, fused);
        }
        ASSERT_FALSE(handle.GetKernelsImpl(fused_kernel_name, network_config).empty());

        DestroyInternalRnnDropoutDesc(rnnDesc);

        // Both paths share the input-projection GEMM and differ only in the
        // recurrence arithmetic, so they should agree to a few ULPs of fp32
        // accumulation noise. rms_range normalizes by magnitude; 1e-5 is far
        // below the ~1e-2 shift any gate-order/offset bug produces.
        constexpr double kTol = 1e-5;

        ASSERT_EQ(std::get<0>(ref).size(), std::get<0>(fused).size());
        EXPECT_LT(miopen::rms_range(std::get<0>(ref), std::get<0>(fused)), kTol) << "output";
        EXPECT_LT(miopen::rms_range(std::get<1>(ref), std::get<1>(fused)), kTol) << "hy";
        EXPECT_LT(miopen::rms_range(std::get<2>(ref), std::get<2>(fused)), kTol) << "cy";
    }
};

} // namespace

using GPU_LSTM_fused_infer_FP32 = GPU_LSTM_fused_infer_test<float>;

TEST_P(GPU_LSTM_fused_infer_FP32, FloatTest) { RunFusedInference(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_LSTM_fused_infer_FP32, testing::ValuesIn(GetTestCases()));
