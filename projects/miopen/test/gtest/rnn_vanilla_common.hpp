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
#include "../rnn_util.hpp"
#include "get_handle.hpp"
#include <common_utils/float_equal.hpp>
#include "../dropout_util.hpp"
#include "../cpu_rnn.hpp"
#include "compare_helper.hpp"
#include "gtest_common.hpp"
#include "miopen/rnn.hpp"
#include "miopen/tensor.hpp"
#include "workspace.hpp"
#include "gtest_desc_guard.hpp"

#define MIO_RNN_TEST_DEBUG 0
#define MIO_RNN_TIME_EVERYTHING 0

namespace {

using RNNParam =
    std::tuple<int, int, int, int, int, bool, bool, bool, bool, bool, int, int, int, int, int>;

auto GenCases(bool full_tests = false, bool use_dropout = false)
{
    int dropout = use_dropout ? 1 : 0;
    std::vector<int> modes(2, 0);
    modes[1] = 1;
    // std::vector<int> defaultBS(1);
    // auto flat_batch_fill = use_dropout ? ({false, true}) : ({true});

    if(full_tests)
    {
        return ::testing::Combine(::testing::ValuesIn(get_rnn_batchSize()),
                                  ::testing::ValuesIn(get_rnn_seq_len()),
                                  ::testing::ValuesIn(get_rnn_vector_len()),
                                  ::testing::ValuesIn(get_rnn_hidden_size()),
                                  ::testing::ValuesIn(get_rnn_num_layers()),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({false, true}),
                                  ::testing::ValuesIn({dropout}),
                                  ::testing::ValuesIn({1}),
                                  ::testing::ValuesIn({1}),
                                  ::testing::ValuesIn(modes),
                                  ::testing::ValuesIn(modes));
    }
    else
    {
        return ::testing::Combine(::testing::ValuesIn({5}),
                                  ::testing::ValuesIn({1}),
                                  ::testing::ValuesIn(get_rnn_vector_len()),
                                  ::testing::ValuesIn(get_rnn_hidden_size()),
                                  ::testing::ValuesIn(get_rnn_num_layers()),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({false, true}),
                                  ::testing::ValuesIn({dropout}),
                                  ::testing::ValuesIn({1}),
                                  ::testing::ValuesIn({1}),
                                  ::testing::ValuesIn({1}),
                                  ::testing::ValuesIn({1}));
    }
}

//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T>
struct verify_forward_infer_rnn
{
    std::vector<T> input;
    std::vector<T> initHidden;
    std::vector<T> weights;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int rnnMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nohy;

    verify_forward_infer_rnn(miopenRNNDescriptor_t pRD,
                             const std::vector<T>& px,
                             const std::vector<T>& phx,
                             const std::vector<T>& pW,
                             const std::vector<int>& pBS,
                             const int pHS,
                             const int pBN,
                             const int pS,
                             const int pNL,
                             const int pBM,
                             const int pDM,
                             const int pIM,
                             const int pRM,
                             const int pVL,
                             const size_t pHXZ,
                             const bool pnohx = false,
                             const bool pnohy = false)
        : input(px),
          initHidden(phx),
          weights(pW),
          batch_seq(pBS),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          rnnMode(pRM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          nohy(pnohy)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);
    }

    std::vector<T> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = (dirMode != 0) ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t out_sz = 0;

        size_t reserveSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        std::vector<T> reserveSpace(reserveSpaceSize / sizeof(T));
        std::vector<T> output(out_sz / sizeof(T));
        std::vector<T> hiddenState(initHidden.size());

        RNNFwdTrainCPUVerify(handle,
                             false,
                             miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                             input,
                             weights,     // [ input_state_weight_trans
                                          // hidden_state_weight0_trans input1_trans
                                          // hidden1_trans ... output_weight;
                                          // bidirectional reversed weights ]
                             hiddenState, // current/final hidden state
                             initHidden,  // initial hidden state
                             output,
                             batch_seq,    // input batch size
                             inputVecLen,  // input data length
                             seqLength,    // Number of iterations to unroll over
                             dirMode,      // whether using bidirectional net
                             biasMode,     // whether using bias
                             bi * nLayers, // 1 by numlayer (number of stacks of hidden layers) for
                                           // unidirection, 2 by numlayer for bidirection
                             batch_seq.at(0), // equal to input batch size in_n[0]
                             hiddenSize,      // hidden state number
                             bi_stride, // 1 by hy_h related function for unidirection, 2 by hy_h
                                        // related function for bidirection
                             rnnMode,
                             inputMode,
                             reserveSpace,
                             nohx);

#if(MIO_RNN_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n", i, output[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward inference RNN pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward inference CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return output;
    }

    std::vector<T> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        size_t out_sz        = 0;
        size_t workSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        Workspace wspace{workSpaceSize};

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T));
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);
        auto hy          = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * ((dirMode != 0) ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

        auto hx_dev = nohx ? miopen::Allocator::ManageDataPtr{} : handle.Write(initHidden);

        miopenRNNForwardInference(&handle,
                                  rnnDesc,
                                  seqLength,
                                  inputDescs.data(),
                                  input_dev.get(),
                                  &hiddenDesc,
                                  hx_dev.get(),
                                  &hiddenDesc,
                                  nullptr,
                                  &weightDesc,
                                  weights_dev.get(),
                                  outputDescs.data(),
                                  output_dev.get(),
                                  &hiddenDesc,
                                  ((nohy) ? nullptr : hy_dev.get()),
                                  &hiddenDesc,
                                  nullptr,
                                  wspace.ptr(),
                                  wspace.size());

#if(MIO_RNN_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n", i, outdata[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward inference GPU" << std::endl;
#endif
        return (handle.Read<T>(output_dev, output.size()));
    }

    void fail() const
    {
        std::stringstream ss{};
        ss << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                ss << batch_seq.at(i) << ",";
            }
            else
            {
                ss << batch_seq.at(i);
            }
        }
        ss << " -m " << ((rnnMode != 0) ? "tanh" : "relu") << " -k " << seqLength << " -H "
           << hiddenSize << " -W " << inputVecLen << " -l " << nLayers << " -F 0 -r " << dirMode
           << " -b " << biasMode << " -p " << inputMode << std::endl;
        ss << "Forward Inference RNN vanilla: " << std::endl;
        ss << "Output tensor output failed verification." << std::endl;
        GTEST_FAIL() << ss.str();
    }
};
//~~~~~~~~~~~~ END FWD INFERENCE ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_rnn
{
    std::vector<T> input;
    std::vector<T> initHidden;
    std::vector<T> weights;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int rnnMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nohy;
    bool use_dropout;

    verify_forward_train_rnn(miopenRNNDescriptor_t pRD,
                             const std::vector<T>& px,
                             const std::vector<T>& phx,
                             const std::vector<T>& pW,
                             const std::vector<int>& pBS,
                             const int pHS,
                             const int pBN,
                             const int pS,
                             const int pNL,
                             const int pBM,
                             const int pDM,
                             const int pIM,
                             const int pRM,
                             const int pVL,
                             const size_t pHXZ,
                             const bool pnohx        = false,
                             const bool pnohy        = false,
                             const bool puse_dropout = false)
        : input(px),
          initHidden(phx),
          weights(pW),
          batch_seq(pBS),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          rnnMode(pRM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          nohy(pnohy),
          use_dropout(puse_dropout)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = (dirMode != 0) ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t out_sz = 0;

        size_t reserveSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        std::vector<T> reserveSpace((reserveSpaceSize + sizeof(T) - 1) / sizeof(T));
        std::vector<T> output(out_sz / sizeof(T));
        std::vector<T> hiddenState(initHidden.size());

        RNNFwdTrainCPUVerify(handle,
                             use_dropout,
                             miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                             input,
                             weights,     // [ input_state_weight_trans
                                          // hidden_state_weight0_trans input1_trans
                                          // hidden1_trans ... output_weight;
                                          // bidirectional reversed weights ]
                             hiddenState, // current/final hidden state
                             initHidden,  // initial hidden state
                             output,
                             batch_seq,    // input batch size
                             inputVecLen,  // input data length
                             seqLength,    // Number of iterations to unroll over
                             dirMode,      // whether using bidirectional net
                             biasMode,     // whether using bias
                             bi * nLayers, // 1 by numlayer (number of stacks of hidden layers) for
                                           // unidirection, 2 by numlayer for bidirection
                             batch_seq.at(0), // equal to input batch size in_n[0]
                             hiddenSize,      // hidden state number
                             bi_stride, // 1 by hy_h related function for unidirection, 2 by hy_h
                                        // related function for bidirection
                             rnnMode,
                             inputMode,
                             reserveSpace,
                             nohx);

#if(MIO_RNN_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n", i, output[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward train RNN pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        auto retSet = std::make_tuple(output, (nohy ? initHidden : hiddenState), reserveSpace);

#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward train CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        size_t out_sz           = 0;
        size_t workSpaceSize    = 0;
        size_t reserveSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        Workspace wspace{workSpaceSize};

        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        reserveSpaceSize = (reserveSpaceSize + (sizeof(T) - 1)) & ~(sizeof(T) - 1);

        Workspace rspace{reserveSpaceSize};

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T));
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);
        // auto hx_dev      = handle.Write(initHidden);
        auto hy = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * ((dirMode != 0) ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

        auto hx_dev = nohx ? miopen::Allocator::ManageDataPtr{} : handle.Write(initHidden);

        miopenRNNForwardTraining(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 hx_dev.get(),
                                 &hiddenDesc,
                                 nullptr,
                                 &weightDesc,
                                 weights_dev.get(),
                                 outputDescs.data(),
                                 output_dev.get(),
                                 &hiddenDesc,
                                 ((nohy) ? nullptr : hy_dev.get()),
                                 &hiddenDesc,
                                 nullptr,
                                 wspace.ptr(),
                                 wspace.size(),
                                 rspace.ptr(),
                                 rspace.size());

#if(MIO_RNN_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n", i, outdata[i]);
        }
#endif

        auto retSet = std::make_tuple(handle.Read<T>(output_dev, output.size()),
                                      (nohy ? initHidden : handle.Read<T>(hy_dev, hy.size())),
                                      rspace.Read<std::vector<T>>());
#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward train GPU" << std::endl;
#endif
        return retSet;
    }

    void fail() const
    {
        std::stringstream ss{};
        ss << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                ss << batch_seq.at(i) << ",";
            }
            else
            {
                ss << batch_seq.at(i);
            }
        }
        ss << " -m " << ((rnnMode != 0) ? "tanh" : "relu") << " -k " << seqLength << " -H "
           << hiddenSize << " -W " << inputVecLen << " -l " << nLayers << " -F 0 -r " << dirMode
           << " -b " << biasMode << " -p " << inputMode << " -U " << int(use_dropout) << std::endl;
        ss << "Forward Train RNN vanilla: " << std::endl;
        GTEST_FAIL() << ss.str();
    }
};
//~~~~~~~~~~~~ END FWD TRAIN ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS DATA
//****************************************************
template <class T>
struct verify_backward_data_rnn
{
    std::vector<T> yin;        // Y
    std::vector<T> dy;         // dY
    std::vector<T> dhy;        // dHY
    std::vector<T> initHidden; // HX
    std::vector<T> weights;
    std::vector<T> reserveSpace;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int rnnMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    bool nohx;
    bool nodhy;
    bool nodhx;
    bool use_dropout;
    size_t realHiddenSize;

    verify_backward_data_rnn(miopenRNNDescriptor_t pRD,
                             const std::vector<T>& py,
                             const std::vector<T>& pdy,
                             const std::vector<T>& pdhy,
                             const std::vector<T>& phx,
                             const std::vector<T>& pW,
                             const std::vector<T>& pRS,
                             const std::vector<int>& pBS,
                             const int pHS,
                             const int pBN,
                             const int pS,
                             const int pNL,
                             const int pBM,
                             const int pDM,
                             const int pIM,
                             const int pRM,
                             const int pVL,
                             const size_t pHXZ,
                             const bool pnohx        = false,
                             const bool pnodhy       = false,
                             const bool pnodhx       = false,
                             const bool puse_dropout = false)
        : yin(py),
          dy(pdy),
          dhy(pdhy),
          initHidden(phx),
          weights(pW),
          reserveSpace(pRS),
          batch_seq(pBS),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          rnnMode(pRM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          nohx(pnohx),
          nodhy(pnodhy),
          nodhx(pnodhx),
          use_dropout(puse_dropout),
          realHiddenSize(pHXZ)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);

        if(!nodhy)
            dhy = pdhy; // this may be intentionally a nullptr
        else
            dhy.resize(realHiddenSize);
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = (dirMode != 0) ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t workSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        std::vector<T> dx(in_sz / sizeof(T));

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        std::vector<T> workSpace(workSpaceSize / sizeof(T));

        std::vector<T> dhx(initHidden.size());

        RNNBwdDataCPUVerify(use_dropout,
                            miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                            dx,              // OUTPUT
                            weights,         // [ input_state_weight_trans
                                             // hidden_state_weight0_trans input1_trans
                                             // hidden1_trans ... output_weight;
                                             // bidirectional reversed weights ]
                            dhy,             // dhy -- input: current/final hidden state
                            dhx,             // dhx OUTPUT
                            initHidden,      // HX initial hidden state
                            yin,             // Y input
                            dy,              // dY -- input
                            batch_seq,       // input batch size
                            inputVecLen,     // input data length
                            seqLength,       // Number of iterations to unroll over
                            dirMode,         // whether using bidirectional net
                            biasMode,        // whether using bias
                            bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers)
                                             // for unidirection, 2 by numlayer for bidirection
                            batch_seq.at(0), // equal to input batch size in_n[0]
                            hiddenSize,      // hidden state number
                            bi_stride,       // 1 by hy_h related function for unidirection, 2 by
                                             // hy_h related function for bidirection
                            rnnMode,
                            inputMode,
                            reserveSpace,
                            workSpace,
                            nodhy);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_data_rnn_vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        auto retSet = std::make_tuple(dx, (nodhx ? initHidden : dhx), reserveSpace, workSpace);

#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN backward data CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        size_t out_sz        = 0;
        size_t workSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        Workspace wspace{workSpaceSize};

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        auto yin_dev  = handle.Write(yin);
        auto dyin_dev = handle.Write(dy);
        // auto dhyin_dev        = handle.Write(dhy);
        Workspace rspace{};
        rspace.Write(reserveSpace);
        auto weights_dev = handle.Write(weights);
        // auto hx_dev           = handle.Write(initHidden);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * ((dirMode != 0) ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        std::vector<T> dx(in_sz / sizeof(T));
        auto dx_dev = handle.Write(dx);

        std::vector<T> dhx(initHidden.size());
        auto dhx_dev = handle.Write(dhx);

        auto dhy_dev = nodhy ? miopen::Allocator::ManageDataPtr{} : handle.Write(dhy);
        auto hx_dev  = nohx ? miopen::Allocator::ManageDataPtr{} : handle.Write(initHidden);

        miopenRNNBackwardData(&handle,
                              rnnDesc,
                              seqLength,
                              outputDescs.data(),
                              yin_dev.get(),
                              outputDescs.data(),
                              dyin_dev.get(),
                              &hiddenDesc,
                              dhy_dev.get(),
                              &hiddenDesc,
                              nullptr,
                              &weightDesc,
                              weights_dev.get(),
                              &hiddenDesc,
                              hx_dev.get(),
                              &hiddenDesc,
                              nullptr,
                              inputDescs.data(),
                              dx_dev.get(),
                              &hiddenDesc,
                              ((nodhx) ? nullptr : dhx_dev.get()),
                              &hiddenDesc,
                              nullptr,
                              wspace.ptr(),
                              wspace.size(),
                              rspace.ptr(),
                              rspace.size());

        auto retSet = std::make_tuple(handle.Read<T>(dx_dev, dx.size()),
                                      (nodhx ? initHidden : handle.Read<T>(dhx_dev, dhx.size())),
                                      rspace.Read<std::vector<T>>(),
                                      wspace.Read<std::vector<T>>());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward data RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN backward data GPU" << std::endl;
#endif
        return retSet;
    }

    void fail() const
    {
        std::stringstream ss{};
        ss << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                ss << batch_seq.at(i) << ",";
            }
            else
            {
                ss << batch_seq.at(i);
            }
        }
        ss << " -m " << ((rnnMode != 0) ? "tanh" : "relu") << " -k " << seqLength << " -H "
           << hiddenSize << " -W " << inputVecLen << " -l " << nLayers << " -F 0 -r " << dirMode
           << " -b " << biasMode << " -p " << inputMode << " -U " << int(use_dropout) << std::endl;
        ss << "Backward Data RNN vanilla: " << std::endl;
        GTEST_FAIL() << ss.str();
    }
};
//~~~~~~~~~~~~ END BACKWARD DATA ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS WEIGHTS
//****************************************************
template <class T>
struct verify_backward_weights_rnn
{
    std::vector<T> input;      // Y
    std::vector<T> dy;         // dY
    std::vector<T> initHidden; // HX
    std::vector<T> reserveSpace;
    std::vector<T> workSpace;
    std::vector<int> batch_seq;
    int weightSize;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int rnnMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    bool nohx;
    bool use_dropout;
    size_t realHiddenSize;

    verify_backward_weights_rnn(miopenRNNDescriptor_t pRD,
                                const std::vector<T>& px,
                                const std::vector<T>& pdy,
                                const std::vector<T>& phx,
                                const std::vector<T>& pRS,
                                const std::vector<T>& pWS,
                                const std::vector<int>& pBS,
                                const int pHS,
                                const int pW,
                                const int pBN,
                                const int pS,
                                const int pNL,
                                const int pBM,
                                const int pDM,
                                const int pIM,
                                const int pRM,
                                const int pVL,
                                const size_t pHXZ,
                                const bool pnohx        = false,
                                const bool puse_dropout = false)
        : input(px),
          dy(pdy),
          initHidden(phx),
          reserveSpace(pRS),
          workSpace(pWS),
          batch_seq(pBS),
          weightSize(pW),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          rnnMode(pRM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          nohx(pnohx),
          use_dropout(puse_dropout),
          realHiddenSize(pHXZ)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);
    }

    std::tuple<std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        int bi        = (dirMode != 0) ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        std::vector<T> dweights(weightSize);

        RNNBwdWeightCPUVerify(use_dropout,
                              input,
                              dweights,   // [ input_state_weight_trans
                                          // hidden_state_weight0_trans
                                          // input1_trans hidden1_trans ...
                                          // output_weight; bidirectional
                                          // reversed weights ]
                              initHidden, // initial hidden state
                              dy,
                              batch_seq,       // input batch size
                              inputVecLen,     // input data length
                              seqLength,       // Number of iterations to unroll over
                              dirMode,         // whether using bidirectional net
                              biasMode,        // whether using bias
                              bi * nLayers,    // 1 by numlayer (number of stacks of hidden
                                               // layers) for unidirection, 2 by numlayer for
                                               // bidirection
                              batch_seq.at(0), // equal to input batch size in_n[0]
                              hiddenSize,      // hidden state number
                              bi_stride,       // 1 by hy_h related function for unidirection, 2
                                               // by hy_h related function for bidirection
                              rnnMode,
                              inputMode,
                              reserveSpace,
                              workSpace,
                              nohx);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_weights_rnn_vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN backward weights CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return std::make_tuple(dweights);
    }

    std::tuple<std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        Workspace wspace{};
        wspace.Write(workSpace);
        Workspace rspace{};
        rspace.Write(reserveSpace);

        std::vector<T> dweights(weightSize);
        auto dweights_dev = handle.Write(dweights);
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, {weightSize});

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * ((dirMode != 0) ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);
        auto hx_dev    = nohx ? miopen::Allocator::ManageDataPtr{} : handle.Write(initHidden);
        auto dy_dev    = handle.Write(dy);
        auto input_dev = handle.Write(input);

        miopenRNNBackwardWeights(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 hx_dev.get(),
                                 outputDescs.data(),
                                 dy_dev.get(),
                                 &weightDesc,
                                 dweights_dev.get(),
                                 wspace.ptr(),
                                 wspace.size(),
                                 rspace.ptr(),
                                 rspace.size());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backwards_weights RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_RNN_TEST_DEBUG > 0)
        std::cout << "Done with RNN backward weights GPU" << std::endl;
#endif
        auto retvec = handle.Read<T>(dweights_dev, dweights.size());
        return std::make_tuple(retvec);
    }

    void fail() const
    {
        std::stringstream ss{};
        ss << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                ss << batch_seq.at(i) << ",";
            }
            else
            {
                ss << batch_seq.at(i);
            }
        }
        ss << " -m " << ((rnnMode != 0) ? "tanh" : "relu") << " -k " << seqLength << " -H "
           << hiddenSize << " -W " << inputVecLen << " -l " << nLayers << " -F 0 -r " << dirMode
           << " -b " << biasMode << " -p " << inputMode << " -U " << int(use_dropout) << std::endl;
        ss << "Backward Weights RNN vanilla: " << std::endl;
        GTEST_FAIL() << ss.str();
    }
};

template <typename T>
struct RNNVanillaCommon : ::testing::TestWithParam<RNNParam>
{
protected:
    void SetUp() override
    {
        std::tie(batchSize,
                 seqLength,
                 inVecLen,
                 hiddenSize,
                 numLayers,
                 nohx,
                 nodhy,
                 nohy,
                 nodhx,
                 flatBatchFill,
                 useDropout,
                 inputMode,
                 biasMode,
                 dirMode,
                 rnnMode) = GetParam();
    }

    void run()
    {

#if(MIOPEN_BACKEND_OPENCL == 1)
        if(miopen_type<T>{} == miopenHalf)
            GTEST_SKIP() << "FP16 not supported for MIOPEN_BACKEND_OPENCL == 1" << std::endl;
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

        ASSERT_EQ(batchSeq.size(), seqLength)
            << "FAILED: Batch sequence vector length, does not match sequence length." << std::endl;

#if(MIO_RNN_TEST_DEBUG == 2)
        printf("seqLen: %d, batch_seq array len: %d\n", seqLength, batchSeq.size());
        for(int i = 0; i < seqLength; i++)
        {
            std::cout << "batch seq[" << i << "]: " << batchSeq.at(i) << std::endl;
        }
#endif

        auto&& handle = get_handle();

        int batch_n = std::accumulate(batchSeq.begin(), batchSeq.end(), 0);

        RNNDescGuard rnnDesc;
        DropoutDescGuard DropoutDesc;
        size_t statesSizeInBytes = 0;

        // See DestroyInternalRnnDropoutDesc — frees the descriptor allocated
        // by miopenCreateRNNDescriptor that the upcoming Set* will leak.
        DestroyInternalRnnDropoutDesc(rnnDesc);

        miopenRNNAlgo_t algoMode  = miopenRNNdefault;
        miopenHandle_t mio_handle = nullptr;
#if MIOPEN_BACKEND_HIP
        void* dropout_state_buf = nullptr;
#elif MIOPEN_BACKEND_OPENCL
        cl_mem dropout_state_buf = nullptr;
#endif
        if(useDropout != 0)
        {
// Workaround for issue #2335.
// OpenCL error creating buffer: 0 Invalid Buffer Size
#if MIOPEN_BACKEND_OPENCL
            GTEST_SKIP() << "Skip test for Issue #2335: " << std::endl;
#endif
            miopenCreateWithStream(&mio_handle, handle.GetStream());

            float dropout_rate              = 0.5;
            unsigned long long dropout_seed = 0ULL;
            miopenDropoutGetStatesSize(mio_handle, &statesSizeInBytes);

#if MIOPEN_BACKEND_OPENCL
            cl_context ctx;
            clGetCommandQueueInfo(
                handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
            dropout_state_buf =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, statesSizeInBytes, nullptr, nullptr);
#elif MIOPEN_BACKEND_HIP
            (void)hipMalloc(static_cast<void**>(&dropout_state_buf), statesSizeInBytes);
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
                                      miopenRNNMode_t(rnnMode),
                                      miopenRNNBiasMode_t(biasMode),
                                      miopenRNNAlgo_t(algoMode),
                                      miopen_type<T>{});
        }
        else
        {
            miopenSetRNNDescriptor(rnnDesc,
                                   hiddenSize,
                                   numLayers,
                                   miopenRNNInputMode_t(inputMode),
                                   miopenRNNDirectionMode_t(dirMode),
                                   miopenRNNMode_t(rnnMode),
                                   miopenRNNBiasMode_t(biasMode),
                                   miopenRNNAlgo_t(algoMode),
                                   miopen_type<T>{}); // defined in superclass testdriver
        }

        // Create input tensor
        // If we are in skip mode, take the real input size to be the vector length.
        auto inVecReal    = (inputMode != 0) ? hiddenSize : inVecLen;
        std::size_t in_sz = static_cast<std::size_t>(inVecReal) * batch_n;
        std::vector<T> input(in_sz);
        for(std::size_t i = 0; i < in_sz; i++)
        {
            input[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
        }

        std::size_t hx_sz = ((dirMode != 0) ? 2ULL : 1ULL) * hiddenSize * batchSize * numLayers;
        std::vector<T> hx;
        if(!nohx)
            hx.resize(hx_sz);

        std::vector<T> dhyin;
        if(!nodhy)
            dhyin.resize(hx_sz);

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
            weights[i] = prng::gen_descreet_uniform_sign<T>(Data_scale / 10, 100);
        }

#if(MIO_RNN_TEST_DEBUG > 0)
        printf("inputMode: %d, biasMode: %d, rnnMode: %d, dirMode: %d\n",
               inputMode,
               biasMode,
               rnnMode,
               dirMode);
        printf("hsize: %d, batch_n: %d, seqLength: %d, inputLen: %d, numLayers: %d\n",
               hiddenSize,
               batch_n,
               seqLength,
               inVecLen,
               numLayers);
#endif

        /* normal hx/cx/dhy/dcy input test */

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

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batchSeq, inVecReal, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batchSeq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        size_t out_sz;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        size_t reserveSpaceSize;
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        size_t workSpaceSize;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);

        size_t total_mem = statesSizeInBytes + reserveSpaceSize + workSpaceSize + 2 * out_sz +
                           (in_sz + wei_sz + (nohx ? 0 : hx_sz) + (nohy ? 0 : hx_sz) +
                            (nodhx ? 0 : hx_sz) + (nodhy ? 0 : hx_sz)) *
                               sizeof(T);
        size_t device_mem = handle.GetGlobalMemorySize();

        ASSERT_LT(total_mem, device_mem) << "Config requires " << total_mem
                                         << " Bytes to write all necessary tensors to GPU. GPU has "
                                         << device_mem << " Bytes of memory." << std::endl;

        auto fwdTrainOutputPair =
            test_helpers::CompareResults(verify_forward_train_rnn<T>{rnnDesc,
                                                                     input,
                                                                     hx,
                                                                     weights,
                                                                     batchSeq,
                                                                     hiddenSize,
                                                                     batch_n,
                                                                     seqLength,
                                                                     numLayers,
                                                                     biasMode,
                                                                     dirMode,
                                                                     inputMode,
                                                                     rnnMode,
                                                                     inVecReal,
                                                                     hx_sz,
                                                                     nohx,
                                                                     nohy,
                                                                     bool(useDropout)});

        /// RETURNS std::make_tuple(output, hiddenState, reserveSpace);
        auto reserveSpaceFwdTrain = std::get<2>(fwdTrainOutputPair.second);
        // auto curHiddenState       = std::get<1>(fwdTrainOutputPair.second);
        auto yin = std::get<0>(fwdTrainOutputPair.second);

        std::vector<T> dyin(yin.size());
        for(std::size_t i = 0; i < yin.size(); i++)
        {
            dyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
        }
#if(MIO_RNN_TEST_DEBUG > 0)
        printf("Running backward data RNN.\n");
#endif
        auto bwdDataOutputPair =
            test_helpers::CompareResults(verify_backward_data_rnn<T>{rnnDesc,
                                                                     yin,
                                                                     dyin,
                                                                     dhyin,
                                                                     hx,
                                                                     weights,
                                                                     reserveSpaceFwdTrain,
                                                                     batchSeq,
                                                                     hiddenSize,
                                                                     batch_n,
                                                                     seqLength,
                                                                     numLayers,
                                                                     biasMode,
                                                                     dirMode,
                                                                     inputMode,
                                                                     rnnMode,
                                                                     inVecReal,
                                                                     hx_sz,
                                                                     nohx,
                                                                     nodhy,
                                                                     nodhx,
                                                                     bool(useDropout)});

        // RETURNS:  std::make_tuple(dx, dhx, reserveSpace, workSpace);
        auto reserveSpaceBwdData = std::get<2>(bwdDataOutputPair.second);
        auto workSpaceBwdData    = std::get<3>(bwdDataOutputPair.second);

#if(MIO_RNN_TEST_DEBUG > 0)
        printf("Running backward weights RNN.\n");
        printf("reserve sz: %d, workSpace sz: %d, weight sz: %d\n",
               reserveSpaceBwdData.size(),
               workSpaceBwdData.size(),
               wei_sz);
        fflush(nullptr);
#endif
        // auto dweights_pair =
        test_helpers::CompareResults(verify_backward_weights_rnn<T>{
            rnnDesc,          input,     dyin,       hx,      reserveSpaceBwdData,
            workSpaceBwdData, batchSeq,  hiddenSize, wei_sz,  batch_n,
            seqLength,        numLayers, biasMode,   dirMode, inputMode,
            rnnMode,          inVecReal, hx_sz,      nohx,    bool(useDropout)});

/// \todo Resolve the issue and remove workaround.
/// ROCm3.3, Radeon VII: Many test cases always fail with:
/// "Forward Inference RNN vanilla:"
/// "Output tensor output failed verification."
#if 0
        if(useDropout == 0)
        {
            test_helpers::CompareResults(verify_forward_infer_rnn<T>{rnnDesc,
                                               input,
                                               hx,
                                               weights,
                                               batchSeq,
                                               hiddenSize,
                                               batch_n,
                                               seqLength,
                                               numLayers,
                                               biasMode,
                                               dirMode,
                                               inputMode,
                                               rnnMode,
                                               inVecReal,
                                               hx_sz,
                                               nohx,
                                               nohy});
        }
#endif
        /* normal hx/cx/dhy/dcy input test end */

        // DLOWELL: This part may produce NAN and infinities. Further investigation is needed.
        //        auto dweights = std::get<1>(dweights_pair);
        //        std::transform(weightData.begin( ), weightData.end( ), dweights.begin( ),
        //        weightData.begin( ),std::minus<T>( ));
        //        test_helpers::CompareResults(verify_forward_infer_rnn<T>{rnnDesc, inputData,
        //                                        curHiddenState, weightData, batchSeq,
        //                                        hiddenSize, batch_n,
        //                                        seqLength, numLayers,
        //                                        biasMode, dirMode,
        //                                        inputMode, rnnMode, inVecReal});

        // Free the DropoutDescriptor that miopenSetRNNDescriptor just allocated.
        // In the dropout path, the internal pointer aliases the user-owned
        // DropoutDescGuard — freeing it would double-free.
        if(useDropout == 0)
            DestroyInternalRnnDropoutDesc(rnnDesc);

        if(useDropout != 0)
        {
#if MIOPEN_BACKEND_HIP
            (void)hipFree(dropout_state_buf);
#elif MIOPEN_BACKEND_OPENCL
            (void)clReleaseMemObject(dropout_state_buf);
#endif
            miopenDestroy(mio_handle);
        }
    }

private:
    std::vector<int> batchSeq;
    int seqLength{};
    int inVecLen{};
    int hiddenSize{};
    int numLayers{};
    int inputMode{};
    int biasMode{};
    int dirMode{};
    int rnnMode{};
    int batchSize{};
    int useDropout{};

    // Null pointer input
    bool nohx  = false;
    bool nodhy = false;
    bool nohy  = false;
    bool nodhx = false;

    // use this to uniformly fill the batch per time step
    bool flatBatchFill = false;

    const double Data_scale = 0.001;
};

} // namespace
