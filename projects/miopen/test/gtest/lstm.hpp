/*******************************************************************************
 *
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
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
#include <gtest/gtest_common.hpp>

struct Verifier
{
    int iteration{0};
    int time_iter{1};
    int warmup_iter{0};
    bool time{false};
    bool verbose{false};
    bool rethrow{false};
    bool no_validate{false};
    double tolerance{80.0};

    template <class CpuRange, class GpuRange, class Compare, class Report, class Fail>
    bool compare_and_report(
        const CpuRange& out_cpu, const GpuRange& out_gpu, Compare compare, Report report, Fail fail)
    {
        std::vector<double> error;
        bool pass = compare(error, out_cpu, out_gpu);
        return report(pass, error, out_cpu, out_gpu, fail);
    }

    template <class... CpuRanges, class... GpuRanges, class Compare, class Report, class Fail>
    bool compare_and_report(const std::tuple<CpuRanges...>& out_cpu,
                            const std::tuple<GpuRanges...>& out_gpu,
                            Compare compare,
                            Report report,
                            Fail fail)
    {
        static_assert(sizeof...(CpuRanges) == sizeof...(GpuRanges), "CPU and GPU mismatch");
        return miopen::sequence([&](auto... is) {
            bool continue_ = true;
            miopen::each_args(
                [&](auto i) {
                    // cppcheck-suppress knownConditionTrueFalse
                    if(continue_)
                    {
                        continue_ = this->compare_and_report(
                            std::get<i>(out_cpu), std::get<i>(out_gpu), compare, report, [&](int) {
                                return fail(i);
                            });
                    }
                },
                is...);
            return continue_;
        })(std::integral_constant<std::size_t, sizeof...(CpuRanges)>{});
    }

    auto verify_reporter()
    {
        return [=](bool pass,
                   std::vector<double> error,
                   const auto& out_cpu,
                   const auto& out_gpu,
                   auto fail) {
            if(not pass or verbose)
            {
                if(not error.empty() or not pass)
                {
                    if(not error.empty())
                        std::cout << (pass ? "error: " : "FAILED: ") << error.front() << std::endl;
                    else
                        std::cout << "FAILED: " << std::endl;

                    if(not verbose)
                    {
                        std::cout << "Iteration: " << iteration << std::endl;
                        fail(-1);
                    }
                }

                auto mxdiff = miopen::max_diff(out_cpu, out_gpu);
                std::cout << "Max diff: " << mxdiff << std::endl;

                if(miopen::range_zero(out_cpu))
                    std::cout << "CPU data is all zeros" << std::endl;
                if(miopen::range_zero(out_gpu))
                    std::cout << "GPU data is all zeros" << std::endl;

                auto idx = miopen::mismatch_idx(out_cpu, out_gpu, miopen::float_equal);
                if(idx < miopen::range_distance(out_cpu))
                {
                    std::cout << "Mismatch at " << idx << ": " << out_cpu[idx]
                              << " != " << out_gpu[idx] << std::endl;
                }

                auto cpu_nan_idx = find_idx(out_cpu, miopen::not_finite);
                if(cpu_nan_idx >= 0)
                {
                    std::cout << "Non finite number found in CPU data at " << cpu_nan_idx << ": "
                              << out_cpu[cpu_nan_idx] << std::endl;
                }

                auto gpu_nan_idx = find_idx(out_gpu, miopen::not_finite);
                if(gpu_nan_idx >= 0)
                {
                    std::cout << "Non finite number found in GPU data at " << gpu_nan_idx << ": "
                              << out_gpu[gpu_nan_idx] << std::endl;
                }
            }
            else if(miopen::range_zero(out_cpu) and miopen::range_zero(out_gpu) and
                    (miopen::range_distance(out_cpu) != 0))
            {
                std::cout << "Warning: Both CPU and GPU data is all zero" << std::endl;
                fail(-1);
            }
            return true;
        };
    }

    template <class F, class V, class... Ts>
    auto verify_impl(F&& f, V&& v, Ts&&... xs)
        -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
    {
        decltype(v.cpu(xs...)) cpu;
        decltype(v.gpu(xs...)) gpu;

        try
        {
            auto&& h = get_handle();
            // Compute cpu
            std::future<decltype(v.cpu(xs...))> cpuf;
            {
                cpuf = cpu_async(v, xs...);
            }
            // Compute gpu
            if(time)
            {
                for(size_t i = 0; i < warmup_iter; ++i)
                {
                    v.gpu(xs...);
                }
                h.EnableProfiling();
                h.ResetKernelTime();
            }
            gpu = v.gpu(xs...);
            if(time)
            {
                float total_time = h.GetKernelTime();
                for(size_t i = 1; i < time_iter; ++i)
                {
                    h.ResetKernelTime();
                    v.gpu(xs...);
                    total_time += h.GetKernelTime();
                }
                std::cout << "Kernel time: " << (total_time / time_iter) << " ms" << std::endl;
                h.EnableProfiling(false);
            }

            // Validate
            if(not no_validate)
            {
                cpu         = cpuf.get();
                auto report = verify_reporter();
                compare_and_report(cpu, gpu, f, report, [&](int mode) { v.fail(mode, xs...); });
            }

            if(verbose or time)
                v.fail(std::integral_constant<int, -1>{}, xs...);
        }
        catch(const std::exception& ex)
        {
            std::cout << "FAILED: " << ex.what() << std::endl;
            v.fail(-1, xs...);
            if(rethrow)
                throw;
        }
        catch(...)
        {
            std::cout << "FAILED with unknown exception" << std::endl;
            v.fail(-1, xs...);
            if(rethrow)
                throw;
        }
        if(no_validate)
        {
            return std::make_pair(gpu, gpu);
        }
        else
        {
            return std::make_pair(cpu, gpu);
        }
    }

    template <class V, class... Ts>
    auto verify(V&& v, Ts&&... xs) -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
    {
        return verify_impl(
            [&](std::vector<double>& error, auto&& cpu, auto&& gpu) {
                CHECK(miopen::range_distance(cpu) == miopen::range_distance(gpu));

                using value_type = miopen::range_value<decltype(gpu)>;
                double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
                error            = {miopen::rms_range(cpu, gpu)};
                return error.front() <= threshold;
            },
            v,
            xs...);
    }
};

template <typename T>
struct LSTM_test : Verifier
{
    int batchSize{0};
    int seqLength{0};
    int inVecLen{0};
    int hiddenSize{0};
    int numLayers{1};
    int useDropout{0};
    int usePadding{0};
    int flatBatchFill{0};
    int inputMode{0};
    int biasMode{0};
    int dirMode{0};
    int algoMode{0};
    bool nohx{false};
    bool nodhy{false};
    bool nocx{false};
    bool nodcy{false};
    bool nohy{false};
    bool nodhx{false};
    bool nocy{false};
    bool nodcx{false};
    std::vector<int> batchSeq;
    const double Data_scale = 0.001;
    miopenDataType_t dataType{miopenFloat};

    void RunTest()
    {
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
            GTEST_SKIP() << "FAILED: Batch sequence vector length, does not match sequence length.";
        }

#if(MIO_LSTM_TEST_DEBUG == 2)
        for(int i = 0; i < seqLength; i++)
        {
            std::cout << "batch seq[" << i << "]: " << batchSeq.at(i) << std::endl;
        }
#endif

        auto&& handle = get_handle();
        miopenRNNDescriptor_t rnnDesc;
        miopenCreateRNNDescriptor(&rnnDesc);
        miopenDropoutDescriptor_t DropoutDesc;
        miopenCreateDropoutDescriptor(&DropoutDesc);
        size_t statesSizeInBytes = 0;

        if(useDropout != 0)
        {
            miopenHandle_t mio_handle;
            miopenCreateWithStream(&mio_handle, handle.GetStream());

            float dropout_rate{0.5f};
            unsigned long long dropout_seed{0ULL};
            miopenDropoutGetStatesSize(mio_handle, &statesSizeInBytes);

            void* dropout_state_buf;
            hipMalloc(static_cast<void**>(&dropout_state_buf), statesSizeInBytes);

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
                                      dataType);
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
                                   dataType);
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
        auto firstInputDesc = miopen::TensorDescriptor(dataType, inlens);
        miopenGetRNNParamsSize(&handle, rnnDesc, &firstInputDesc, &wei_bytes, dataType);
        auto wei_sz = int(wei_bytes / sizeof(T));
        std::vector<T> weights(wei_sz);
        for(std::size_t i = 0; i < wei_sz; i++)
        {
            weights[i] = prng::gen_descreet_uniform_sign<T>(Data_scale, 100);
        }

        int batch_n = std::accumulate(batchSeq.begin(), batchSeq.end(), 0);
#if(MIO_LSTM_TEST_DEBUG == 2)
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
        createTensorDescArray(inputCPPDescs, inputDescs, batchSeq, inVecLen, dataType);
        size_t reserveSpaceSize;
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(
            outputCPPDescs, outputDescs, batchSeq, hiddenSize * ((dirMode != 0) ? 2 : 1), dataType);

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
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " bytes of memory." << std::endl;
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

#if(MIO_LSTM_TEST_DEBUG == 2)
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

#if(MIO_LSTM_TEST_DEBUG == 2)
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
};
