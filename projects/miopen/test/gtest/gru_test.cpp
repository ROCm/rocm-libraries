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

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <miopen/rnn.hpp>
#include <miopen/tensor.hpp>
#include "tensor_holder.hpp"
#include <half/half.hpp>

#include <vector>
#include <sstream>

#include "compare_helper.hpp"
#include "get_handle.hpp"
#include "../driver/gru_verify_gemm.hpp"
#include "../rnn_util.hpp"
#include "../workspace.hpp"


namespace
{
inline std::vector<int> GenBatchSeq(const int batchSize, const int seqLength)
{

    static constexpr int modval = 3;

    int currentval = batchSize;
    std::vector<int> batchSeq;
    batchSeq.reserve(seqLength);
    for(int i = 0; i < seqLength; i++)
    {
        if(i > 0)
        {
            int nvalue = currentval - prng::gen_0_to_B(modval);
            currentval = (nvalue < 1) ? 1 : nvalue;
        }
        batchSeq.push_back(currentval);
    }
    return batchSeq;
}


//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T>
struct verify_forward_infer_gru
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
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nohy;

    verify_forward_infer_gru(miopenRNNDescriptor_t pRD,
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

    std::tuple<std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode != 0 ? 2 : 1;
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

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        RunGRUForwardGEMMCPUVerify(&handle,
                        input,
                        weights,     // [ input_state_weight_trans
                                     // hidden_state_weight0_trans input1_trans
                                     // hidden1_trans ... output_weight;
                                     // bidirectional reversed weights ]
                        hiddenState, // current/final hidden state
                        initHidden,  // initial hidden state
                        output,
                        batch_seq,       // input batch size
                        inputVecLen,     // input data length
                        seqLength,       // Number of iterations to unroll over
                        dirMode,         // whether using bidirectional net
                        biasMode,        // whether using bias
                        bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers) for
                                         // unidirection, 2 by numlayer for bidirection
                        batch_seq.at(0), // equal to input batch size in_n[0]
                        hiddenSize,      // hidden state number
                        bi_stride,       // 1 by hy_h related function for unidirection, 2 by hy_h
                                         // related function for bidirection
                        inputMode,
                        reserveSpace,
                        false,
                        &(miopen::deref(miopen::deref(rnnDesc).dropoutDesc)),
                        nohx);

#if(MIO_GRU_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            std::cout << "CPU outdata[" << i << "]: "output[i] << "\n";
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward inference GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: CPU forward inference GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU forward inference CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return std::make_tuple(output);
    }

    std::tuple<std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        size_t out_sz         = 0;
        size_t workspace_size = 0;

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

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);
        Workspace wspace{workspace_size};

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T));
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);
        auto hy          = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        miopenRNNForwardInference(&handle,
                                  rnnDesc,
                                  seqLength,
                                  inputDescs.data(),
                                  input_dev.get(),
                                  &hiddenDesc,
                                  ((nohx) ? nullptr : handle.Write(initHidden).get()),
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

#if(MIO_GRU_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            std::cout << "GPU outdata[" << i << "]: " << outdata[i] << "\n";
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: GPU forward_infer GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU forward inference GPU" << std::endl;
#endif
        return std::make_tuple((handle.Read<T>(output_dev, output.size())));
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
        ss << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;

        ss << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        ss << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers << std::endl;
        ss << "Forward Inference GRU: " << std::endl;
        ss << "Output tensor output failed verification." << std::endl;
        GTEST_FAIL() << ss.str();
    }
};
//~~~~~~~~~~~~ END FWD INFERENCE ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_gru
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
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nohy;
    bool use_dropout;

    verify_forward_train_gru(miopenRNNDescriptor_t pRD,
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

        int bi        = dirMode != 0 ? 2 : 1;
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

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif
        RunGRUForwardGEMMCPUVerify(&handle,
                        input,
                        weights,     // [ input_state_weight_trans
                                     // hidden_state_weight0_trans input1_trans
                                     // hidden1_trans ... output_weight;
                                     // bidirectional reversed weights ]
                        hiddenState, // current/final hidden state
                        initHidden,  // initial hidden state
                        output,
                        batch_seq,       // input batch size
                        inputVecLen,     // input data length
                        seqLength,       // Number of iterations to unroll over
                        dirMode,         // whether using bidirectional net
                        biasMode,        // whether using bias
                        bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers) for
                                         // unidirection, 2 by numlayer for bidirection
                        batch_seq.at(0), // equal to input batch size in_n[0]
                        hiddenSize,      // hidden state number
                        bi_stride,       // 1 by hy_h related function for unidirection, 2 by hy_h
                                         // related function for bidirection
                        inputMode,
                        reserveSpace,
                        use_dropout,
                        &(miopen::deref(miopen::deref(rnnDesc).dropoutDesc)),
                        nohx);

#if(MIO_GRU_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            std::cout << "CPU outdata[" << i << "]: " << output[i] << "\n";
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward train GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
        std::cout << "Wall clock: CPU forward train GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif

        auto retSet = std::make_tuple(output, (nohy ? initHidden : hiddenState), reserveSpace);

#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU forward train CPU" << std::endl;
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
        size_t workspace_size   = 0;
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

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);
        Workspace wspace{workspace_size};

        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) & ~(sizeof(T) - 1);
        assert(reserveSpaceSize % sizeof(T) == 0);
        Workspace rspace{reserveSpaceSize};

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T));
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);

        auto hy = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        miopenRNNForwardTraining(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 ((nohx) ? nullptr : handle.Write(initHidden).get()),
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

#if(MIO_GRU_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            std::cout << "GPU outdata[" << i << "]: " << outdata[i] << "\n";
        }
#endif

        auto retSet = std::make_tuple(handle.Read<T>(output_dev, output.size()),
                                      (nohy ? initHidden : handle.Read<T>(hy_dev, hy.size())),
                                      rspace.Read<std::vector<T>>());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: GPU forward_train GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
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
        ss << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;

        ss << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        ss << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        ss << "Forward Train GRU: " << std::endl;
        GTEST_FAIL() << ss.str();
    }
};
//~~~~~~~~~~~~ END FWD TRAIN ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS DATA
//****************************************************
template <class T>
struct verify_backward_data_gru
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
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nodhy;
    bool nodhx;
    bool use_dropout;

    verify_backward_data_gru(miopenRNNDescriptor_t pRD,
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
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          nodhy(pnodhy),
          nodhx(pnodhx),
          use_dropout(puse_dropout)
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

        int bi        = dirMode != 0 ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t workspace_size;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        // Outputs ----------
        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);
        std::vector<T> workSpace(workspace_size / sizeof(T));
        std::vector<T> dx(in_sz / sizeof(T));
        std::vector<T> dhx(initHidden.size());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        RunGRUBackwardDataGEMMCPUVerify(dx,              // DX (output)
                                        weights,         // [ input_state_weight_trans
                                                        //   hidden_state_weight0_trans input1_trans
                                                        //   hidden1_trans ... output_weight;
                                                        //   bidirectional reversed weights ]
                                        dhy,             // current/final hidden state
                                        dhx,             // DHX (output)
                                        initHidden,      // HX initial hidden state
                                        // yin,             // Y
                                        dy,              // DY
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
                                        inputMode,
                                        reserveSpace,
                                        workSpace,
                                        use_dropout,
                                        &(miopen::deref(miopen::deref(rnnDesc).dropoutDesc)),
                                        nohx,
                                        nodhy);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward data GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: CPU backward data GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif

        auto retSet = std::make_tuple(dx, (nodhx ? initHidden : dhx), reserveSpace, workSpace);

#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU backward data CPU" << std::endl;
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

        size_t out_sz = 0;

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

        size_t workspace_size = 0;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);
        Workspace wspace{workspace_size};

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        auto yin_dev     = handle.Write(yin);
        auto dyin_dev    = handle.Write(dy);
        auto weights_dev = handle.Write(weights);

        Workspace rspace{};
        rspace.Write(reserveSpace);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
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

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        miopenRNNBackwardData(&handle,
                              rnnDesc,
                              seqLength,
                              outputDescs.data(),
                              yin_dev.get(),
                              outputDescs.data(),
                              dyin_dev.get(),
                              &hiddenDesc,
                              ((nodhy) ? nullptr : handle.Write(dhy).get()),
                              &hiddenDesc,
                              nullptr,
                              &weightDesc,
                              weights_dev.get(),
                              &hiddenDesc,
                              ((nohx) ? nullptr : handle.Write(initHidden).get()),
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

        std::cout << "Wall clock: GPU backward data GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: GPU backward data GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU backward data GPU" << std::endl;
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
        ss << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;
        ss << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        ss << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        ss << "Backward Data GRU: " << std::endl;
        GTEST_FAIL() << ss.str();
    }
};
//~~~~~~~~~~~~ END BACKWARD DATA ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS WEIGHTS
//****************************************************
template <class T>
struct verify_backward_weights_gru
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
    bool biasMode{false};
    bool dirMode{false};
    int inputMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool use_dropout;

    verify_backward_weights_gru(miopenRNNDescriptor_t pRD,
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
          biasMode(pBM == 0 ? false : true),
          dirMode(pDM == 0 ? false : true),
          inputMode(pIM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          use_dropout(puse_dropout)
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
        int bi = dirMode ? 2 : 1;
        std::vector<T> dweights(weightSize);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif
        RunGRUBackwardWeightGEMMCPUVerify(input,
                                          dweights,        // (output) [ input_state_weight_trans
                                                          // hidden_state_weight0_trans
                                                          // input1_trans hidden1_trans ...
                                                          // output_weight; bidirectional
                                                          // reversed weights ]
                                          initHidden,      // initial hidden state
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
                                          bi * hiddenSize,
                                          inputMode,
                                          reserveSpace,
                                          workSpace,
                                          use_dropout,
                                          nohx);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_weights GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
        std::cout << "Wall clock: CPU backward_weights GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU backward weights CPU" << std::endl;
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
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);
        auto dy_dev    = handle.Write(dy);
        auto input_dev = handle.Write(input);

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start1 = std::chrono::high_resolution_clock::now();
#endif

        miopenRNNBackwardWeights(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 ((nohx) ? nullptr : handle.Write(initHidden).get()),
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

        std::cout << "Wall clock: GPU backwards_weights GRU pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;

        std::cout << "Wall clock: GPU backwards_weights GRU pass time (core): "
                  << std::chrono::duration<double>(t_end - t_start1).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_GRU_TEST_DEBUG > 0)
        std::cout << "Done with GRU backward weights GPU" << std::endl;
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
        ss << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;
        ss << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        ss << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        ss << "Backward Weights GRU: " << std::endl;
        GTEST_FAIL() << ss.str();
    }
};
//~~~~~~~~~~~~ END BACKWARD WEIGHTS ~~~~~~~~~~~~~~~~~~~~~~~~
} // anonymous namespace

using GruTestCase = std::tuple<int, int, int, int, int, int, int, int, bool, bool, bool, bool, bool, bool>;

auto GenCases(bool full_tests = false, bool gen_dropout=false)
{
    std::vector<int> modes(2, 0);
    modes[1] = 1;

    if(gen_dropout)
    {
            return ::testing::Combine(::testing::ValuesIn({23}),
                                  ::testing::ValuesIn({13}),
                                  ::testing::ValuesIn({67}),
                                  ::testing::ValuesIn({3}),
#if(MIO_GRU_TEST_DEBUG == 3)
                                  ::testing::ValuesIn({0}),
                                  ::testing::ValuesIn({1}),
                                  ::testing::ValuesIn({0}),
#else
                                  ::testing::ValuesIn({0}),
                                  ::testing::ValuesIn({0}),
                                  ::testing::ValuesIn(modes),
#endif
                                  ::testing::ValuesIn({17}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true, false}));
    }

    if(full_tests)
    {
        return ::testing::Combine(::testing::ValuesIn(get_gru_seq_len()),
                                  ::testing::ValuesIn(get_gru_vector_len()),
                                  ::testing::ValuesIn(get_gru_hidden_size()),
                                  ::testing::ValuesIn(get_gru_num_layers()),
                                  ::testing::ValuesIn(modes),
                                  ::testing::ValuesIn(modes),
                                  ::testing::ValuesIn(modes),
                                  ::testing::ValuesIn(get_gru_batchSize()),
                                  ::testing::ValuesIn({false}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}));
    }
    return ::testing::Combine(::testing::ValuesIn({2}),
                                  ::testing::ValuesIn(get_gru_vector_len()),
                                  ::testing::ValuesIn(get_gru_hidden_size()),
                                  ::testing::ValuesIn(get_gru_num_layers()),
#if(MIO_GRU_TEST_DEBUG == 3)
                                  ::testing::ValuesIn({0}),
                                  ::testing::ValuesIn({1}),
                                  ::testing::ValuesIn({0}),
#else
                                  ::testing::ValuesIn({0}),
                                  ::testing::ValuesIn({0}),
                                  ::testing::ValuesIn({0}),
#endif
                                  ::testing::ValuesIn({17}),
                                  ::testing::ValuesIn({false}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}),
                                  ::testing::ValuesIn({true}));
}

template<typename T>
struct GruTolerance
{
    static double value() { return 500000.f; }
};

template<>
struct GruTolerance<half_float::half>
{
    static double value() { return 100000.f; };
};

template <class T>
class gru_test : public testing::TestWithParam<GruTestCase>
{
    int seqLength{};
    int inVecLen{};
    int hiddenSize{};
    int numLayers{};
    int inputMode{};
    int biasMode{};
    int dirMode{};
    int batchSize{};
    bool useDropout{false};
    const double tolerance = GruTolerance<T>::value(); // Will be multiplied by std::numeric_limits<T>::epsilon()

    // Null pointer input
    bool nohx          = false;
    bool nodhy         = false;
    bool nohy          = false;
    bool nodhx         = false;
    bool flatBatchFill = false;

    std::vector<int> batchSeq;

public:
    void fill_buffers(std::vector<T>& input, std::vector<T>& hx, std::vector<T>& weights)
    {
        auto fill_array_via_gen = [](std::vector<T>& dst, auto gen, int seed_offset = 0) {
            prng::reset_seed(seed_offset);
            size_t dst_sz = dst.size();
            for(size_t it = 0; it < dst_sz; it++)
                dst[it] = gen();
        };

        auto pos_gen = [](double scale, int range) {
            return [=]() -> T { return prng::gen_descreet_unsigned<T>(scale, range); };
        };

        auto sign_gen = [](double scale, int range) {
            return [=]() -> T { return prng::gen_descreet_uniform_sign<T>(scale, range); };
        };

        const double data_max_v = sqrt(1. / hiddenSize);
        int data_range          = 100;
        const double data_scale = data_max_v / data_range;
        fill_array_via_gen(input, pos_gen(data_scale, data_range), 0);

        if(!nohx)
        {
            fill_array_via_gen(hx, pos_gen(data_scale, data_range), 1);
        }

        // filter
        const double weights_max_v = sqrt(1. / hiddenSize);
        int weights_range          = 64;
        const double weights_scale = weights_max_v / weights_range;

        fill_array_via_gen(weights, sign_gen(weights_scale, weights_range), 2);
    }

    void fill_bwd_buffers(std::vector<T>& dy, std::vector<T>& dhy)
    {
        auto fill_array_via_gen = [](std::vector<T>& dst, auto gen, int seed_offset = 0) {
            prng::reset_seed(seed_offset);
            size_t dst_sz = dst.size();

            for(size_t it = 0; it < dst_sz; it++)
                dst[it] = gen();
        };

        auto sign_gen = [](double scale, int range) {
            return [=]() { return prng::gen_descreet_uniform_sign<T>(scale, range); };
        };

        const double bwd_data_max_v = sqrt(1. / hiddenSize) / 8;
        int bwd_data_range          = 100;
        const double bwd_data_scale = bwd_data_max_v / bwd_data_range;

        if(!nodhy)
        {
            fill_array_via_gen(dhy, sign_gen(bwd_data_scale, bwd_data_range), 3);
        }

        fill_array_via_gen(dy, sign_gen(bwd_data_scale, bwd_data_range), 4);
        prng::reset_seed();
    }


    void SetUp() override
    {
        // prng::reset_seed();
        auto param = GetParam();
        seqLength = std::get<0>(param);
        inVecLen = std::get<1>(param);
        hiddenSize = std::get<2>(param);
        numLayers = std::get<3>(param);
        inputMode = std::get<4>(param);
        biasMode = std::get<5>(param);
        dirMode = std::get<6>(param);
        batchSize = std::get<7>(param);
        useDropout = std::get<8>(param);
        nohx = std::get<9>(param);
        nodhy = std::get<10>(param);
        nohy = std::get<11>(param);
        nodhx = std::get<12>(param);
        flatBatchFill = std::get<13>(param);
    }

    void Run()
    {
        if(batchSeq.empty() || 0 == batchSeq[0])
        {
            if(flatBatchFill)
            {
                batchSeq.clear();
                batchSeq.resize(seqLength, batchSize);
            }
            else
            {
                batchSeq = GenBatchSeq(batchSize, seqLength);
            }
        }

        auto&& handle = get_handle();

        int batch_n = std::accumulate(batchSeq.begin(), batchSeq.end(), 0);

        miopenRNNDescriptor_t rnnDesc;
        miopenCreateRNNDescriptor(&rnnDesc);
        miopenRNNAlgo_t algoMode = miopenRNNdefault;

        miopenDropoutDescriptor_t DropoutDesc;
        miopenCreateDropoutDescriptor(&DropoutDesc);
        size_t statesSizeInBytes = 0;

        if(useDropout)
        {
// Workaround for issue #2335.
// OpenCL error creating buffer: 0 Invalid Buffer Size
#if MIOPEN_BACKEND_OPENCL
            GTEST_SUCCESS() << "Skip test for Issue #2335: " << std::endl;
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
                                      miopenGRU,
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
                                   miopenGRU,
                                   miopenRNNBiasMode_t(biasMode),
                                   miopenRNNAlgo_t(algoMode),
                                   miopen_type<T>{}); // defined in superclass testdriver
        }

        // Create input tensor
        // If we are in skip mode, take the real input size to be the vector length.
        auto inVecReal    = (inputMode != 0) ? hiddenSize : inVecLen;
        std::size_t in_sz = static_cast<std::size_t>(inVecReal) * batch_n;
        std::size_t hx_sz = ((dirMode != 0) ? 2ULL : 1ULL) * hiddenSize * batchSize * numLayers;

        std::vector<T> input(in_sz), hx(hx_sz), dhyin(hx_sz);

        size_t wei_bytes = [&]() {
            size_t filter_bytes;
            std::vector<int> inlens(2, 0);
            inlens.at(0)        = batchSeq.at(0);
            inlens.at(1)        = inVecReal;
            auto firstInputDesc = miopen::TensorDescriptor(miopen::deref(rnnDesc).dataType, inlens);
            miopenGetRNNParamsSize(
                &handle, rnnDesc, &firstInputDesc, &filter_bytes, miopen::deref(rnnDesc).dataType);
            return filter_bytes;
        }();

        auto wei_sz = wei_bytes / sizeof(T);
        std::vector<T> weights(wei_sz);

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
        size_t workspace_size;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workspace_size);

        size_t total_mem = statesSizeInBytes + reserveSpaceSize + workspace_size + 2 * out_sz +
                           (in_sz + wei_sz + (nohx ? 0 : hx_sz) + (nohy ? 0 : hx_sz) +
                            (nodhx ? 0 : hx_sz) + (nodhy ? 0 : hx_sz)) *
                               sizeof(T);
        size_t device_mem = handle.GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            ADD_FAILURE() << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
        }

        fill_buffers(input, hx, weights);

        auto fwdTrainOutputPair = test_helpers::CompareResults(verify_forward_train_gru<T>{rnnDesc,
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
                                                                     inVecReal,
                                                                     hx_sz,
                                                                     nohx,
                                                                     nohy,
                                                                     useDropout}, tolerance);

        /// RETURNS std::make_tuple(output, hiddenState, reserveSpace);
        auto yin = std::get<0>(fwdTrainOutputPair.second);
        // auto curHiddenState       = std::get<1>(fwdTrainOutputPair.second);
        auto reserveSpaceFwdTrain = std::get<2>(fwdTrainOutputPair.second);

        std::vector<T> dyin(yin.size());

        fill_bwd_buffers(dyin, dhyin);

        auto bwdDataOutputPair = test_helpers::CompareResults(verify_backward_data_gru<T>{
            rnnDesc,   yin,        dyin,    dhyin,     hx,        weights,  reserveSpaceFwdTrain,
            batchSeq,  hiddenSize, batch_n, seqLength, numLayers, biasMode, dirMode,
            inputMode, inVecReal,  hx_sz,   nohx,      nodhy,     nodhx,    useDropout}, tolerance);

        // RETURNS:  std::make_tuple(dx, dhx, reserveSpace, workSpace);
        auto reserveSpaceBwdData = std::get<2>(bwdDataOutputPair.second);
        auto workSpaceBwdData    = std::get<3>(bwdDataOutputPair.second);
        // auto dweights_pair       =
        test_helpers::CompareResults(verify_backward_weights_gru<T>{rnnDesc,
                                              input,
                                              dyin,
                                              hx,
                                              reserveSpaceBwdData,
                                              workSpaceBwdData,
                                              batchSeq,
                                              hiddenSize,
                                              static_cast<int>(wei_sz),
                                              batch_n,
                                              seqLength,
                                              numLayers,
                                              biasMode,
                                              dirMode,
                                              inputMode,
                                              inVecReal,
                                              hx_sz,
                                              nohx,
                                              useDropout}, tolerance);

        if(!useDropout)
        {
            test_helpers::CompareResults(verify_forward_infer_gru<T>{rnnDesc,
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
                                               inVecReal,
                                               hx_sz,
                                               nohx,
                                               nohy}, tolerance);
        }
        // DLOWELL: Subtracting delta weights may produce NAN and infinities. Further investigation
        // is needed.
        //        auto dweights = std::get<1>(dweights_pair);
        //        std::transform(weightData.begin( ), weightData.end( ), dweights.begin( ),
        //        weightData.begin( ),std::minus<T>( ));
        //        test_helpers::CompareResults(verify_forward_infer_gru<T>{rnnDesc, inputData,
        //                                        curHiddenState, curCellState, weightData,
        //                                        batchSeq,
        //                                        hiddenSize, batch_n,
        //                                        seqLength, numLayers,
        //                                        biasMode, dirMode,
        //                                        inputMode, inVecReal});
    }
};

using GPU_GRU_FP32 = gru_test<float>;

TEST_P(GPU_GRU_FP32, TestFloat32) { Run(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_GRU_FP32, GenCases(true));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_GRU_FP32, GenCases());
INSTANTIATE_TEST_SUITE_P(SmokeDropout, GPU_GRU_FP32, GenCases(false, true));


#if !(MIOPEN_BACKEND_OPENCL == 1)
using GPU_GRU_FP16 = gru_test<half_float::half>;

TEST_P(GPU_GRU_FP16, TestFloat16) { Run(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_GRU_FP16, GenCases(true));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_GRU_FP16, GenCases());
INSTANTIATE_TEST_SUITE_P(SmokeDropout, GPU_GRU_FP16, GenCases(false, true));
#endif // !(MIOPEN_BACKEND_OPENCL == 1)
