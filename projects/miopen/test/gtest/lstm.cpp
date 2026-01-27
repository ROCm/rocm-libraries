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

#include "lstm_common.hpp"
#include <hip/hip_runtime.h>
#include <gtest/gtest_common.hpp>

template <typename T>
class GPU_lstm_Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        //if(hipGetDeviceCount(&device_count) != hipSuccess)
        //    device_count = 0;
    }

    void Run()
    {
        //std::vector<std::string> params = GetParam();
        std::cout << "running MIOpenDriver...\n";

        const double Data_scale = 0.001;
#if(MIOPEN_BACKEND_OPENCL == 1)
#if WORKAROUND_ISSUE_692 == 1
        std::cout << "Skip test for Issue #692: " << std::endl;
        exit(EXIT_SUCCESS); // NOLINT (concurrency-mt-unsafe)
#endif
        if(type == miopenHalf)
            exit(EXIT_SUCCESS); // NOLINT (concurrency-mt-unsafe)
#endif

        if(batchSeq.empty() || 0 == batchSeq[0])
        {
            std::cout << "Empty batch sequence. Filling uniformly with batch size: " << batchSize
                      << std::endl;
            if(flatBatchFill)
            {
                batchSeq.clear();
                batchSeq.resize(seqLength, batchSize);
            }
            else
            {
                batchSeq = generate_batchSeq(batchSize, seqLength)[0];
            }
        }

        if(batchSeq.size() != seqLength)
        {
            std::cerr << "FAILED: Batch sequence vector length, does not match sequence length."
                      << std::endl;
            std::abort();
        }

#if(MIO_LSTM_TEST_DEBUG == 2)
        for(int i = 0; i < seqLength; i++)
        {
            std::cout << "batch seq[" << i << "]: " << batchSeq.at(i) << std::endl;
        }
#endif

        auto&& handle = get_handle();

        int batch_n = std::accumulate(batchSeq.begin(), batchSeq.end(), 0);

        miopenRNNDescriptor_t rnnDesc;
        miopenCreateRNNDescriptor(&rnnDesc);
        miopenDropoutDescriptor_t DropoutDesc;
        miopenCreateDropoutDescriptor(&DropoutDesc);
        size_t statesSizeInBytes = 0;

        if(useDropout != 0)
        {
// Workaround for issue #2335.
// OpenCL error creating buffer: 0 Invalid Buffer Size
#if MIOPEN_BACKEND_OPENCL
            std::cout << "Skip test for Issue #2335: " << std::endl;
            return;
#endif
            miopenHandle_t mio_handle;
            miopenCreateWithStream(&mio_handle, handle.GetStream());

            float dropout_rate              = 0.5;
            unsigned long long dropout_seed = 0ULL;
            miopenDropoutGetStatesSize(mio_handle, &statesSizeInBytes);

#if MIOPEN_BACKEND_OPENCL
            cl_context ctx;
            clGetCommandQueueInfo(
                handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
            cl_mem dropout_state_buf =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, statesSizeInBytes, nullptr, nullptr);
#elif MIOPEN_BACKEND_HIP
            void* dropout_state_buf;
            hipMalloc(static_cast<void**>(&dropout_state_buf), statesSizeInBytes);
#endif

            miopenSetDropoutDescriptor(DropoutDesc,
                                       mio_handle,
                                       dropout_rate,
                                       dropout_state_buf,
                                       statesSizeInBytes,
                                       dropout_seed,
                                       false,
                                       false,
                                       MIOPEN_RNG_PSEUDO_XORWOW);

            miopenSetRNNDescriptor_V2(rnnDesc,
                                      hiddenSize,
                                      numLayers,
                                      DropoutDesc,
                                      miopenRNNInputMode_t(inputMode),
                                      miopenRNNDirectionMode_t(dirMode),
                                      miopenLSTM,
                                      miopenRNNBiasMode_t(biasMode),
                                      miopenRNNAlgo_t(algoMode),
                                      type);
        }
        else
        {
            miopenSetRNNDescriptor(rnnDesc,
                                   hiddenSize,
                                   numLayers,
                                   miopenRNNInputMode_t(inputMode),
                                   miopenRNNDirectionMode_t(dirMode),
                                   miopenLSTM,
                                   miopenRNNBiasMode_t(biasMode),
                                   miopenRNNAlgo_t(algoMode),
                                   type); // defined in superclass testdriver
        }

        if(usePadding)
        {
            miopenSetRNNPaddingMode(rnnDesc, miopenRNNPaddingMode_t::miopenRNNIOWithPadding);
        }

        // Create input tensor
        // If we are in skip mode, take the real input size to be the vector length.
        auto inVecReal = (inputMode != 0) ? hiddenSize : inVecLen;

        int batch_padding = usePadding ? batchSeq[0] : 0;

        std::size_t in_sz = getSuperTensorSize(batchSeq,
                                               seqLength,
                                               inVecReal,
                                               hiddenSize,
                                               batch_padding,
                                               dirMode != 0,
                                               true,
                                               usePadding);
        std::vector<T> input(in_sz);
        for(std::size_t i = 0; i < in_sz; i++)
        {
            input[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
        }

        std::size_t hx_sz = ((dirMode != 0) ? 2ULL : 1ULL) * hiddenSize * batchSize * numLayers;
        std::vector<T> hx(hx_sz);
        std::vector<T> cx(hx_sz);
        std::vector<T> dhyin(hx_sz);
        std::vector<T> dcyin(hx_sz);

        size_t wei_bytes = 0;
        std::vector<int> inlens(2, 0);
        inlens.at(0)        = batchSeq.at(0);
        inlens.at(1)        = inVecReal;
        auto firstInputDesc = miopen::TensorDescriptor(miopen::deref(rnnDesc).dataType, inlens);
        miopenGetRNNParamsSize(
            &handle, rnnDesc, &firstInputDesc, &wei_bytes, miopen::deref(rnnDesc).dataType);
        auto wei_sz = int(wei_bytes / sizeof(T));
        std::vector<T> weights(wei_sz);
        for(std::size_t i = 0; i < wei_sz; i++)
        {
            weights[i] = prng::gen_descreet_uniform_sign<T>(Data_scale, 100);
        }

#if(MIO_LSTM_TEST_DEBUG > 0)
        printf("inputMode: %d, biasMode: %d, dirMode: %d\n", inputMode, biasMode, dirMode);
        printf("hz: %d, batch_n: %d, seqLength: %d, inputLen: %d, numLayers: %d\n",
               hiddenSize,
               batch_n,
               seqLength,
               inVecLen,
               numLayers);
        std::cout << "nohx: " << nohx;
        std::cout << ", nocx: " << nocx;
        std::cout << ", nodhy: " << nodhy;
        std::cout << ", nodcy: " << nodcy << std::endl;
        std::cout << "nohy: " << nohy;
        std::cout << ", nocy: " << nocy;
        std::cout << ", nodhx: " << nodhx;
        std::cout << ", nodcx: " << nodcx << std::endl;
#endif

        if(!nohx)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                hx[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        if(!nodhy)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                dhyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        if(!nocx)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                cx[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        if(!nodcy)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                dcyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batchSeq, inVecLen, miopen::deref(rnnDesc).dataType);
        size_t reserveSpaceSize;
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batchSeq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        size_t out_sz = getSuperTensorSize(batchSeq,
                                           seqLength,
                                           inVecLen,
                                           hiddenSize,
                                           batchSeq[0],
                                           dirMode != 0,
                                           false,
                                           usePadding);

        size_t workspace_size;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);

        size_t total_mem = statesSizeInBytes + reserveSpaceSize + workspace_size +
                           (2 * out_sz + in_sz + wei_sz + (nohx ? 0 : hx_sz) + (nohy ? 0 : hx_sz) +
                            (nodhx ? 0 : hx_sz) + (nodhy ? 0 : hx_sz) + (nocx ? 0 : hx_sz) +
                            (nocy ? 0 : hx_sz) + (nodcx ? 0 : hx_sz) + (nodcy ? 0 : hx_sz)) *
                               sizeof(T);
        size_t device_mem = handle.GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
        }

        reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) / sizeof(T);
        std::vector<T> rsvgpu(reserveSpaceSize, T(0));

        size_t inputBatchLenSum =
            std::accumulate(batchSeq.begin(), batchSeq.begin() + seqLength, 0ULL);
        reserveSpaceSize =
            2ULL * 6 * numLayers * inputBatchLenSum * hiddenSize * ((dirMode != 0) ? 2 : 1);
        if(useDropout != 0)
        {
            reserveSpaceSize +=
                (numLayers - 1) * inputBatchLenSum * hiddenSize * ((dirMode != 0) ? 2 : 1);
            reserveSpaceSize *= sizeof(T);
            reserveSpaceSize +=
                (numLayers - 1) * inputBatchLenSum * hiddenSize * ((dirMode != 0) ? 2 : 1);
            reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) / sizeof(T);
        }

        std::vector<T> rsvcpu(reserveSpaceSize, T(0));

        auto fwdTrainOutputPair = verify(verify_forward_train_lstm<T>{
            rnnDesc,          input,      hx,      cx,        weights,   batchSeq, rsvgpu,
            rsvcpu,           hiddenSize, batch_n, seqLength, numLayers, biasMode, dirMode,
            inputMode,        inVecReal,  hx_sz,   nohx,      nocx,      nohy,     nocy,
            bool(useDropout), usePadding});

        /// RETURNS std::make_tuple(output, hiddenState, cellState, reserveSpace);
        auto yin = std::get<0>(fwdTrainOutputPair.second);
        // auto curHiddenState = std::get<1>(fwdTrainOutputPair.second);
        // auto curCellState   = std::get<2>(fwdTrainOutputPair.second);

        if(yin.size() != out_sz)
        {
            std::cout << "FWD FAILED: yin.size() != out_sz." << std::endl
                      << "yin.size()=" << yin.size() << std::endl
                      << "out_sz=" << out_sz << std::endl;

            exit(-1); // NOLINT (concurrency-mt-unsafe)
        }

        std::vector<T> dyin(out_sz);
        for(std::size_t i = 0; i < out_sz; i++)
        {
            dyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
        }

#if(MIO_LSTM_TEST_DEBUG > 0)
        printf("Running backward data LSTM.\n");
#endif
        auto bwdDataOutputPair =
            verify(verify_backward_data_lstm<T>{rnnDesc,   yin,       dyin,
                                                dhyin,     hx,        dcyin,
                                                cx,        weights,   rsvgpu,
                                                rsvcpu,    batchSeq,  hiddenSize,
                                                batch_n,   seqLength, numLayers,
                                                biasMode,  dirMode,   inputMode,
                                                inVecReal, hx_sz,     nohx,
                                                nocx,      nodhy,     nodcy,
                                                nodhx,     nodcx,     bool(useDropout),
                                                usePadding});

        // RETURNS:  std::make_tuple(dx, dhx, dcx, reserveSpace, workSpace);
        auto workSpaceBwdData = std::get<3>(bwdDataOutputPair.second);

#if(MIO_LSTM_TEST_DEBUG > 0)
        printf("Running backward weights LSTM.\n");
        printf("reserve sz: %zu, workSpace sz: %zu, weight sz: %d\n",
               rsvcpu.size(),
               workSpaceBwdData.size(),
               wei_sz);
        fflush(nullptr);
#endif
        // auto dweights_pair =
        verify(verify_backward_weights_lstm<T>{
            rnnDesc,  input,      dyin,      hx,      rsvgpu,    rsvcpu,           workSpaceBwdData,
            batchSeq, hiddenSize, wei_sz,    batch_n, seqLength, numLayers,        biasMode,
            dirMode,  inputMode,  inVecReal, hx_sz,   nohx,      bool(useDropout), usePadding});

    }

    int device_count = 1;//0

    int seqLength;
    int inVecLen;
    int hiddenSize;
    int numLayers;
    int inputMode;
    int biasMode;
    int dirMode;
    int algoMode;
    int batchSize;
    int useDropout;
    std::vector<int> batchSeq;

    bool nohx          = false;
    bool nodhy         = false;
    bool nocx          = false;
    bool nodcy         = false;
    bool nohy          = false;
    bool nodhx         = false;
    bool nocy          = false;
    bool nodcx         = false;
    bool flatBatchFill = false;
    bool usePadding    = false;
};

using GPU_lstm_FP32 = GPU_lstm_Test<float>;

TEST_F(GPU_lstm_FP32, FloatTest_lstm)
{
    if(device_count == 0)
    {
        GTEST_SKIP() << "No HIP devices available for testing";
    }

    testing::internal::CaptureStderr();
    Run();
    auto capture = testing::internal::GetCapturedStderr();
    std::cout << capture;
}
