// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/rnn.hpp>
#include <miopen/miopen.h>
#include "tensor_holder.hpp"

#include "compare_helper.hpp"
#include "gtest_desc_guard.hpp"
#include "gtest_handle_guard.hpp"
#include "../dropout_util.hpp"
#include "../rnn_util.hpp"
#include "../workspace.hpp"

#include <math.h>
namespace {
/// Specific version for compating CPU and GPU results, because this test is order-dependent
template <class VerifyT>
auto GruTestCompareResults(VerifyT&& verifier, double tolerance = 80.f)
    -> std::pair<decltype(verifier.cpu()), decltype(verifier.gpu())>
{
    const auto gpu_result = verifier.gpu();
    const auto cpu_result = verifier.cpu();
    if(!test_helpers::Compare(cpu_result, gpu_result, tolerance))
    {
        verifier.fail();
    }

    return std::make_pair(cpu_result, gpu_result);
}

/**********************************************
 * CPU verification functions
 *
 **********************************************/
template <typename T>
void GRUFwdCPUVerify(const miopen::Handle& handle,
                     bool use_dropout,
                     miopen::DropoutDescriptor& dropoutDesc,
                     std::vector<T>& in,
                     std::vector<T>& wei, // [ input_state_weight_trans
                                          // hidden_state_weight0_trans input1_trans
                                          // hidden1_trans ... output_weight;
                                          // bidirectional reversed weights ]
                     std::vector<T>& hy,  // current/final hidden state
                     std::vector<T>& hx,  // initial hidden state
                     std::vector<T>& out,
                     const std::vector<int>& in_n, // input batch size
                     int in_h,                     // input data length
                     int seqLength,                // Number of iterations to unroll over
                     bool bidirection,             // whether using bidirectional net
                     bool biased,                  // whether using bias
                     int hy_d,  // 1 by numlayer (number of stacks of hidden layers) for
                                // unidirection, 2 by numlayer for bidirection
                     int hy_n,  // equal to input batch size in_n[0]
                     int hy_h,  // hidden state number
                     int out_h, // 1 by hy_h related function for unidirection, 2 by hy_h
                                // related function for bidirection
                     int inputMode,
                     std::vector<T>& rsvspace,
                     bool hx_is_null = false)
{
    int batch_n = sumvc(in_n);

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 3 * hy_h;
    int hy_stride  = bi * 4 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    int wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;

    // initial dropoput
    std::vector<rocrand_state_xorwow> dropout_states_host;
    std::vector<unsigned char> dropout_reservespace_host;
    std::vector<T> dropout_hid_state;
    TensorDescGuard dropout_inputTensor;
    TensorDescGuard dropout_outputTensor;
    if(use_dropout)
    {
        size_t states_size  = dropoutDesc.stateSizeInBytes / sizeof(rocrand_state_xorwow);
        dropout_states_host = std::vector<rocrand_state_xorwow>(states_size);
        InitKernelStateEmulator(dropout_states_host, dropoutDesc);

        std::array<int, 2> drop_in_len  = {{batch_n, hy_h * bi}};
        std::array<int, 2> drop_in_str  = {{hy_stride, 1}};
        std::array<int, 2> drop_out_str = {{hy_h * bi, 1}};
        miopenSetTensorDescriptor(
            dropout_inputTensor, miopenFloat, 2, drop_in_len.data(), drop_in_str.data());
        miopenSetTensorDescriptor(
            dropout_outputTensor, miopenFloat, 2, drop_in_len.data(), drop_out_str.data());

        size_t reserveSpaceSizeInBytes = 0;
        miopenDropoutGetReserveSpaceSize(dropout_inputTensor, &reserveSpaceSizeInBytes);
        size_t reserve_size       = reserveSpaceSizeInBytes / sizeof(unsigned char);
        dropout_reservespace_host = std::vector<unsigned char>(reserve_size * (numlayer - 1),
                                                               static_cast<unsigned char>(1));

        dropout_hid_state = std::vector<T>((numlayer - 1) * batch_n * hy_h * bi, static_cast<T>(0));
    }

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        int hid_shift           = li * batch_n * hy_stride;
        int hx_shift            = li * in_n.at(0) * h_stride;
        int wei_shift_bias_temp = wei_shift_bias + li * 2 * wei_stride;

        // from input
        if(li == 0)
        {
            if(inputMode == 1)
            {
                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int gi = 0; gi < 3; gi++)
                        {
                            rsvspace[hid_shift + bs * hy_stride + gi * hy_h + h] +=
                                in[bs * in_stride + h];
                            if(bidirection)
                            {
                                rsvspace[hid_shift + bs * hy_stride + (gi + 3) * hy_h + h] +=
                                    in[bs * in_stride + h];
                            }
                        }
                    }
                }

                // from bias
                if(biased)
                {
                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        for(int h = 0; h < wei_stride; h++)
                        {
                            rsvspace[hid_shift + bs * hy_stride + h] += wei[wei_shift_bias + h];
                        }
                    }
                }
            }
            else
            {
                gemm_cpu(in.data(),
                         in_h,
                         batch_n,
                         in_stride,
                         false,
                         wei.data(), // wei_state.data(),
                         in_h,
                         hy_h * bi * 3,
                         in_stride,
                         true,
                         &rsvspace[hid_shift],
                         hy_h * bi * 3,
                         batch_n,
                         hy_stride,
                         1,
                         1);

                // from bias
                if(biased)
                {
                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        for(int h = 0; h < wei_stride; h++)
                        {
                            rsvspace[hid_shift + bs * hy_stride + h] += wei[wei_shift_bias + h];
                        }
                    }
                }
            }
        }
        else
        {
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * 3 * hy_h;
            if(use_dropout)
            {
                auto dropout_states_tmp = dropout_states_host;
                size_t drop_out_offset  = (static_cast<size_t>(li) - 1) * batch_n * hy_h * bi;

                DropoutForwardVerify<T>(handle,
                                        dropoutDesc,
                                        miopen::deref(dropout_inputTensor.get()),
                                        rsvspace,
                                        miopen::deref(dropout_outputTensor.get()),
                                        dropout_hid_state,
                                        dropout_reservespace_host,
                                        dropout_states_tmp,
                                        prelayer_shift,
                                        drop_out_offset,
                                        drop_out_offset);

                prelayer_shift = drop_out_offset;
            }

            gemm_cpu(use_dropout ? &dropout_hid_state[prelayer_shift] : &rsvspace[prelayer_shift],
                     hy_h * bi,
                     batch_n,
                     use_dropout ? hy_h * bi : hy_stride,
                     false,
                     &wei[wei_shift], //&wei_state[wei_shift],
                     hy_h * bi,
                     hy_h * bi * 3,
                     bi_stride,
                     true,
                     &rsvspace[hid_shift],
                     hy_h * bi * 3,
                     batch_n,
                     hy_stride,
                     1,
                     1);

            // from bias
            if(biased)
            {
                for(int bs = 0; bs < batch_n; bs++)
                {
                    for(int h = 0; h < wei_stride; h++)
                    {
                        rsvspace[hid_shift + bs * hy_stride + h] += wei[wei_shift_bias_temp + h];
                    }
                }
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n;
        for(int ti = 0; ti < seqLength; ti++)
        {
            baccbi -= in_n.at(seqLength - 1 - ti);
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift;

            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    gemm_cpu(&hx[hx_shift],
                             hy_h,
                             in_n.at(ti),
                             uni_stride,
                             false,
                             &wei[wei_shift],
                             hy_h,
                             hy_h * 2,
                             uni_stride,
                             true,
                             &rsvspace[hid_shift + bacc * hy_stride],
                             hy_h * 2,
                             in_n.at(ti),
                             hy_stride,
                             1,
                             1);

                    if(biased)
                    {
                        for(int bs = 0; bs < in_n.at(ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                for(int gi = 0; gi < 2; gi++)
                                {
                                    rsvspace[hid_shift + (bacc + bs) * hy_stride + gi * hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + gi * hy_h + h];
                                }
                            }
                        }
                    }

                    gemm_cpu(&hx[hx_shift],
                             hy_h,
                             in_n.at(ti),
                             uni_stride,
                             false,
                             &wei[wei_shift + 2 * hy_h * uni_stride],
                             hy_h,
                             hy_h,
                             uni_stride,
                             true,
                             &rsvspace[hid_shift + bacc * hy_stride + bi * 3 * hy_h],
                             hy_h,
                             in_n.at(ti),
                             hy_stride,
                             1,
                             1);

                    if(biased)
                    {
                        for(int bs = 0; bs < in_n.at(ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                    wei[wei_shift_bias_temp + wei_stride + 2 * hy_h + h];
                            }
                        }
                    }

                    if(bidirection)
                    {
                        gemm_cpu(&hx[hx_shift + hy_n * hy_h],
                                 hy_h,
                                 in_n.at(seqLength - 1 - ti),
                                 uni_stride,
                                 false,
                                 &wei[wei_shift + 3 * hy_h * uni_stride],
                                 hy_h,
                                 hy_h * 2,
                                 uni_stride,
                                 true,
                                 &rsvspace[hid_shift + baccbi * hy_stride + 3 * hy_h],
                                 hy_h * 2,
                                 in_n.at(seqLength - 1 - ti),
                                 hy_stride,
                                 1,
                                 1);

                        if(biased)
                        {
                            for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    for(int gi = 0; gi < 2; gi++)
                                    {
                                        rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                 (3 + gi) * hy_h + h] +=
                                            wei[wei_shift_bias_temp + wei_stride + (3 + gi) * hy_h +
                                                h];
                                    }
                                }
                            }
                        }

                        gemm_cpu(&hx[hx_shift + hy_n * hy_h],
                                 hy_h,
                                 in_n.at(seqLength - 1 - ti),
                                 uni_stride,
                                 false,
                                 &wei[wei_shift + 5 * hy_h * uni_stride],
                                 hy_h,
                                 hy_h,
                                 uni_stride,
                                 true,
                                 &rsvspace[hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h],
                                 hy_h,
                                 in_n.at(seqLength - 1 - ti),
                                 hy_stride,
                                 1,
                                 1);

                        if(biased)
                        {
                            for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + 5 * hy_h + h];
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                gemm_cpu(&hy[hx_shift],
                         hy_h,
                         in_n.at(ti),
                         uni_stride,
                         false,
                         &wei[wei_shift],
                         hy_h,
                         hy_h * 2,
                         uni_stride,
                         true,
                         &rsvspace[hid_shift + bacc * hy_stride],
                         hy_h * 2,
                         in_n.at(ti),
                         hy_stride,
                         1,
                         1);

                if(biased)
                {
                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            for(int gi = 0; gi < 2; gi++)
                            {
                                rsvspace[hid_shift + (bacc + bs) * hy_stride + gi * hy_h + h] +=
                                    wei[wei_shift_bias_temp + wei_stride + gi * hy_h + h];
                            }
                        }
                    }
                }

                gemm_cpu(&hy[hx_shift],
                         hy_h,
                         in_n.at(ti),
                         uni_stride,
                         false,
                         &wei[wei_shift + 2 * hy_h * uni_stride],
                         hy_h,
                         hy_h,
                         uni_stride,
                         true,
                         &rsvspace[hid_shift + bacc * hy_stride + bi * 3 * hy_h],
                         hy_h,
                         in_n.at(ti),
                         hy_stride,
                         1,
                         1);

                if(biased)
                {
                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                wei[wei_shift_bias_temp + wei_stride + 2 * hy_h + h];
                        }
                    }
                }

                if(bidirection)
                {

                    if(!hx_is_null && in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti))
                    {
                        gemm_cpu(
                            &hx[hx_shift + hy_n * hy_h + in_n.at(seqLength - ti) * hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            uni_stride,
                            false,
                            &wei[wei_shift + 3 * hy_h * uni_stride],
                            hy_h,
                            hy_h * 2,
                            uni_stride,
                            true,
                            &rsvspace[hid_shift + (baccbi + in_n.at(seqLength - ti)) * hy_stride +
                                      3 * hy_h],
                            hy_h * 2,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            hy_stride,
                            1,
                            1);

                        if(biased)
                        {
                            for(int bs = in_n.at(seqLength - ti); bs < in_n.at(seqLength - 1 - ti);
                                bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    for(int gi = 0; gi < 2; gi++)
                                    {
                                        rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                 (3 + gi) * hy_h + h] +=
                                            wei[wei_shift_bias_temp + wei_stride + (3 + gi) * hy_h +
                                                h];
                                    }
                                }
                            }
                        }

                        gemm_cpu(
                            &hx[hx_shift + hy_n * hy_h + in_n.at(seqLength - ti) * hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            uni_stride,
                            false,
                            &wei[wei_shift + 5 * hy_h * uni_stride],
                            hy_h,
                            hy_h,
                            uni_stride,
                            true,
                            &rsvspace[hid_shift + (baccbi + in_n.at(seqLength - ti)) * hy_stride +
                                      bi * 3 * hy_h + hy_h],
                            hy_h,
                            (in_n.at(seqLength - 1 - ti) - in_n.at(seqLength - ti)),
                            hy_stride,
                            1,
                            1);

                        if(biased)
                        {
                            for(int bs = in_n.at(seqLength - ti); bs < in_n.at(seqLength - 1 - ti);
                                bs++)
                            {
                                for(int h = 0; h < hy_h; h++)
                                {
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + 5 * hy_h + h];
                                }
                            }
                        }
                    }

                    gemm_cpu(&hy[hx_shift + hy_n * hy_h],
                             hy_h,
                             in_n.at(seqLength - ti),
                             uni_stride,
                             false,
                             &wei[wei_shift + 3 * hy_h * uni_stride],
                             hy_h,
                             hy_h * 2,
                             uni_stride,
                             true,
                             &rsvspace[hid_shift + baccbi * hy_stride + 3 * hy_h],
                             hy_h * 2,
                             in_n.at(seqLength - ti),
                             hy_stride,
                             1,
                             1);

                    if(biased)
                    {
                        for(int bs = 0; bs < in_n.at(seqLength - ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                for(int gi = 0; gi < 2; gi++)
                                {
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                             (3 + gi) * hy_h + h] +=
                                        wei[wei_shift_bias_temp + wei_stride + (3 + gi) * hy_h + h];
                                }
                            }
                        }
                    }

                    gemm_cpu(&hy[hx_shift + hy_n * hy_h],
                             hy_h,
                             in_n.at(seqLength - ti),
                             uni_stride,
                             false,
                             &wei[wei_shift + 5 * hy_h * uni_stride],
                             hy_h,
                             hy_h,
                             uni_stride,
                             true,
                             &rsvspace[hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h],
                             hy_h,
                             in_n.at(seqLength - ti),
                             hy_stride,
                             1,
                             1);

                    if(biased)
                    {
                        for(int bs = 0; bs < in_n.at(seqLength - ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] +=
                                    wei[wei_shift_bias_temp + wei_stride + 5 * hy_h + h];
                            }
                        }
                    }
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h +
                             numlayer * batch_n * hy_stride] =
                        rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h];

                    rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] +=
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h], 2) *
                        rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h];
                    rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] = 0;

                    if(ti == 0)
                    {
                        if(!hx_is_null)
                        {
                            rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                ((1 -
                                  activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2)) *
                                     activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride +
                                                        2 * hy_h + h],
                                               1) +
                                 activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2) *
                                     hx[hx_shift + bs * uni_stride + h]);
                        }
                        else
                        {
                            rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                ((1 -
                                  activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2)) *
                                 activfunc(
                                     rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                     1));
                        }
                    }
                    else
                    {

                        pretime_shift = li * batch_n * hy_stride +
                                        (bacc - in_n.at(ti - 1)) * hy_stride + bi * 3 * hy_h;

                        rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                            ((1 - activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2)) *
                                 activfunc(
                                     rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                     1) +
                             activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2) *
                                 rsvspace[pretime_shift + bs * hy_stride + h]);
                    }

                    rsvspace[hid_shift + (bacc + bs) * hy_stride + h +
                             numlayer * batch_n * hy_stride] =
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2);
                    rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h +
                             numlayer * batch_n * hy_stride] =
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h], 2);
                    rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h +
                             numlayer * batch_n * hy_stride] =
                        activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h], 1);

                    // Update final state
                    hy[hx_shift + bs * uni_stride + h] =
                        rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h];
                }
            }

            if(bidirection)
            {
                pretime_shift = li * batch_n * hy_stride +
                                (baccbi + in_n.at(seqLength - 1 - ti)) * hy_stride + bi * 3 * hy_h +
                                hy_h;

                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h];

                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h], 2) *
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h];
                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h + h] =
                            0;

                        if(ti == 0)
                        {
                            if(!hx_is_null)
                            {
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] +=
                                    ((1 - activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                             3 * hy_h + h],
                                                    2)) *
                                         activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            5 * hy_h + h],
                                                   1) +
                                     activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                        3 * hy_h + h],
                                               2) *
                                         hx[hx_shift + bs * uni_stride + hy_n * hy_h + h]);
                            }
                            else
                            {
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] +=
                                    ((1 - activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                             3 * hy_h + h],
                                                    2)) *
                                     activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                        5 * hy_h + h],
                                               1));
                            }
                        }
                        else
                        {
                            if(!hx_is_null && in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti))
                            {
                                if(bs >= in_n.at(seqLength - ti))
                                {
                                    rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] +=
                                        (activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            3 * hy_h + h],
                                                   2) *
                                         hx[hx_shift + bs * uni_stride + hy_n * hy_h + h]);
                                }
                            }

                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h] +=
                                ((1 - activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                         3 * hy_h + h],
                                                2)) *
                                 activfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h],
                                     1));

                            if(bs < in_n.at(seqLength - ti))
                            {
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] +=
                                    (activfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                        3 * hy_h + h],
                                               2) *
                                     rsvspace[pretime_shift + bs * hy_stride + h]);
                            }
                        }

                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h], 2);
                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h], 2);
                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            activfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h], 1);

                        // Update final hidden state
                        hy[hx_shift + bs * uni_stride + hy_n * hy_h + h] =
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h];
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n * hy_stride + bi * 3 * hy_h;
    for(int bs = 0; bs < batch_n; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            out[bs * out_stride + h] = rsvspace[prelayer_shift + bs * hy_stride + h];
        }
    }

    if(use_dropout)
    {
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            rsvspace.at(numlayer * batch_n * hy_stride * 2 + i) = dropout_hid_state.at(i);
        }
        auto p_drop_rsv = reinterpret_cast<unsigned char*>(&rsvspace.at(
            numlayer * batch_n * hy_stride * 2 + (numlayer - 1) * batch_n * hy_h * bi));
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            *(p_drop_rsv + i) = dropout_reservespace_host.at(i);
        }
    }
}

template <typename T>
void GRUBwdDataCPUVerify(bool use_dropout,
                         miopen::DropoutDescriptor& dropoutDesc,
                         std::vector<T>& din,
                         std::vector<T>& wei, // [ input_state_weight_trans
                                              // hidden_state_weight0_trans input1_trans
                                              // hidden1_trans ... output_weight;
                                              // bidirectional reversed weights ]
                         std::vector<T>& dhy, // current/final hidden state
                         std::vector<T>& dhx,
                         std::vector<T>& hx, // initial hidden state
                         std::vector<T>& out,
                         std::vector<T>& dout,
                         const std::vector<int>& in_n, // input batch size
                         int in_h,                     // input data length
                         int seqLength,                // Number of iterations to unroll over
                         bool bidirection,             // whether using bidirectional net
                         bool,                         // whether using bias
                         int hy_d,  // 1 by numlayer (number of stacks of hidden layers)
                                    // for unidirection, 2 by numlayer for bidirection
                         int hy_n,  // equal to input batch size in_n[0]
                         int hy_h,  // hidden state number
                         int out_h, // 1 by hy_h related function for unidirection, 2 by
                                    // hy_h related function for bidirection
                         int inputMode,
                         std::vector<T>& rsvspace,
                         std::vector<T>& wkspace,
                         bool hx_is_null  = false,
                         bool dhy_is_null = false)
{
    int batch_n = sumvc(in_n);
    (void)out;

    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 3 * hy_h;
    int hy_stride  = bi * 4 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    // initial hidden states
    auto ihs = hy_d * hy_n * hy_h;
    std::vector<T> dcx(ihs);

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    // initial dropoput
    TensorDescGuard dropout_inputTensor;
    std::vector<unsigned char> dropout_reservespace_host;
    if(use_dropout)
    {
        std::array<int, 2> drop_in_len = {{batch_n, hy_h * bi}};
        std::array<int, 2> drop_in_str = {{hy_stride, 1}};
        miopenSetTensorDescriptor(
            dropout_inputTensor, miopenFloat, 2, drop_in_len.data(), drop_in_str.data());

        size_t reserveSpaceSizeInBytes = 0;
        miopenDropoutGetReserveSpaceSize(dropout_inputTensor, &reserveSpaceSizeInBytes);
        size_t reserve_size       = reserveSpaceSizeInBytes / sizeof(unsigned char);
        dropout_reservespace_host = std::vector<unsigned char>(reserve_size * (numlayer - 1),
                                                               static_cast<unsigned char>(0));

        auto p_drop_rsv = reinterpret_cast<unsigned char*>(&rsvspace.at(
            numlayer * batch_n * hy_stride * 2 + (numlayer - 1) * batch_n * hy_h * bi));
        for(int i = 0; i < (numlayer - 1) * batch_n * hy_h * bi; i++)
        {
            dropout_reservespace_host.at(i) = *(p_drop_rsv + i);
        }
    }

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        int wei_shift     = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
        int hid_shift     = li * batch_n * hy_stride;
        int hx_shift      = li * in_n.at(0) * h_stride;
        int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

        if(li == numlayer - 1)
        {
            for(int bs = 0; bs < batch_n; bs++)
            {
                for(int h = 0; h < out_h; h++)
                {
                    wkspace[hid_shift + bi * 3 * hy_h + bs * hy_stride + h] +=
                        dout[bs * out_stride + h];
                }
            }
        }
        else
        {
            int prelayer_shift = (li + 1) * batch_n * hy_stride;

            gemm_cpu(&wkspace[prelayer_shift],
                     hy_h * bi * 3,
                     batch_n,
                     hy_stride,
                     false,
                     &wei[wei_shift],
                     hy_h * bi,
                     hy_h * bi * 3,
                     bi_stride,
                     false,
                     &wkspace[hid_shift + bi * 3 * hy_h],
                     hy_h * bi,
                     batch_n,
                     hy_stride,
                     1,
                     1);

            if(use_dropout)
            {
                DropoutBackwardVerify<T>(dropoutDesc,
                                         miopen::deref(dropout_inputTensor.get()),
                                         wkspace,
                                         miopen::deref(dropout_inputTensor.get()),
                                         wkspace,
                                         dropout_reservespace_host,
                                         hid_shift + bi * 3 * hy_h,
                                         hid_shift + bi * 3 * hy_h,
                                         li * batch_n * hy_h * bi);
            }
        }

        // from hidden state
        int bacc   = batch_n;
        int baccbi = 0;
        for(int ti = seqLength - 1; ti >= 0; ti--)
        {
            bacc -= in_n.at(ti);

            if(ti == seqLength - 1)
            {
                if(!dhy_is_null)
                {
                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                dhy[hx_shift + bs * uni_stride + h];
                        }
                    }

                    if(bidirection)
                    {
                        for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                        {
                            for(int h = 0; h < hy_h; h++)
                            {
                                wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                        hy_h + h] +=
                                    dhy[hx_shift + bs * uni_stride + hy_n * hy_h + h];
                            }
                        }
                    }
                }
            }
            else
            {
                if(!dhy_is_null && in_n.at(ti) > in_n.at(ti + 1))
                {
                    for(int bs = in_n.at(ti + 1); bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                                dhy[hx_shift + bs * uni_stride + h];
                        }
                    }
                }

                int pretime_shift = li * batch_n * hy_stride + (bacc + in_n.at(ti)) * hy_stride;

                gemm_cpu(&wkspace[pretime_shift],
                         hy_h * 2,
                         in_n.at(ti + 1),
                         hy_stride,
                         false,
                         &wei[weitime_shift],
                         hy_h,
                         hy_h * 2,
                         uni_stride,
                         false,
                         &wkspace[hid_shift + bacc * hy_stride + bi * 3 * hy_h],
                         hy_h,
                         in_n.at(ti + 1),
                         hy_stride,
                         1,
                         1);

                for(int bs = 0; bs < in_n.at(ti + 1); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] +=
                            wkspace[pretime_shift + bs * hy_stride + bi * 3 * hy_h + h] *
                            activfunc(rsvspace[pretime_shift + bs * hy_stride + h], 2);

                        wkspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] =
                            wkspace[pretime_shift + bs * hy_stride + 2 * hy_h + h] *
                            activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
                    }
                }

                gemm_cpu(&wkspace[hid_shift + bacc * hy_stride + 2 * hy_h],
                         hy_h,
                         in_n.at(ti + 1),
                         hy_stride,
                         false,
                         &wei[weitime_shift + 2 * hy_h * uni_stride],
                         hy_h,
                         hy_h,
                         uni_stride,
                         false,
                         &wkspace[hid_shift + bacc * hy_stride + bi * 3 * hy_h],
                         hy_h,
                         in_n.at(ti + 1),
                         hy_stride,
                         1,
                         1);

                for(int bs = 0; bs < in_n.at(ti + 1); bs++)
                {
                    auto subidx = hid_shift + (bacc + bs) * hy_stride + 2 * hy_h;
                    std::fill(wkspace.begin() + subidx, wkspace.begin() + subidx + hy_h, 0);
                }

                if(bidirection)
                {
                    pretime_shift = li * batch_n * hy_stride +
                                    (baccbi - in_n.at(seqLength - 2 - ti)) * hy_stride + hy_h * 3;

                    gemm_cpu(&wkspace[pretime_shift],
                             hy_h * 2,
                             in_n.at(seqLength - 1 - ti),
                             hy_stride,
                             false,
                             &wei[weitime_shift + hy_h * 3 * uni_stride],
                             hy_h,
                             hy_h * 2,
                             uni_stride,
                             false,
                             &wkspace[hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h],
                             hy_h,
                             in_n.at(seqLength - 1 - ti),
                             hy_stride,
                             1,
                             1);

                    for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                    h] +=
                                wkspace[pretime_shift + bs * hy_stride + 3 * hy_h + hy_h + h] *
                                activfunc(rsvspace[pretime_shift + bs * hy_stride + h], 2);

                            wkspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] =
                                wkspace[pretime_shift + bs * hy_stride + 2 * hy_h + h] *
                                activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
                        }
                    }

                    gemm_cpu(&wkspace[hid_shift + baccbi * hy_stride + 5 * hy_h],
                             hy_h,
                             in_n.at(seqLength - 1 - ti),
                             hy_stride,
                             false,
                             &wei[weitime_shift + 5 * hy_h * uni_stride],
                             hy_h,
                             hy_h,
                             uni_stride,
                             false,
                             &wkspace[hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h],
                             hy_h,
                             in_n.at(seqLength - 1 - ti),
                             hy_stride,
                             1,
                             1);

                    for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                    {
                        auto subidx = hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h;
                        std::fill(wkspace.begin() + subidx, wkspace.begin() + (subidx + hy_h), 0);
                    }
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    wkspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] +=
                        wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] *
                        (1 - activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2)) *
                        dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                      1);

                    wkspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h] =
                        (rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h +
                                  numlayer * batch_n * hy_stride] *
                         wkspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] *
                         dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h],
                                       2));

                    if(ti == 0)
                    {
                        if(!hx_is_null)
                        {
                            wkspace[hid_shift + (bacc + bs) * hy_stride + h] +=
                                (wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] *
                                 hx[hx_shift + bs * uni_stride + h] *
                                 dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h],
                                               2));
                        }
                        wkspace[hid_shift + (bacc + bs) * hy_stride + h] -=
                            (wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] *
                             activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                       1) *
                             dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2));
                    }
                    else
                    {
                        wkspace[hid_shift + (bacc + bs) * hy_stride + h] +=
                            wkspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h] *
                            (rsvspace[hid_shift + (bacc - in_n.at(ti - 1) + bs) * hy_stride +
                                      bi * 3 * hy_h + h] -
                             activfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h],
                                       1)) *
                            dervactivfunc(rsvspace[hid_shift + (bacc + bs) * hy_stride + h], 2);
                    }

                    rsvspace[hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h + h +
                             numlayer * batch_n * hy_stride] =
                        wkspace[hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h] *
                        rsvspace[hid_shift + (bacc + bs) * hy_stride + hy_h + h +
                                 numlayer * batch_n * hy_stride];
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n.at(seqLength - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wkspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] +=
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                    h] *
                            (1 - activfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h],
                                     2)) *
                            dervactivfunc(
                                rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h], 1);

                        wkspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h] =
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h +
                                     h + numlayer * batch_n * hy_stride];

                        wkspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h] *=
                            (wkspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] *
                             dervactivfunc(
                                 rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h],
                                 2));

                        if(ti == 0)
                        {
                            if(!hx_is_null)
                            {
                                wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] +=
                                    (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] *
                                     hx[hx_shift + bs * uni_stride + hy_n * hy_h + h] *
                                     dervactivfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            3 * hy_h + h],
                                                   2));
                            }
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] -=
                                (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] *
                                 activfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h],
                                     1) *
                                 dervactivfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h],
                                     2));
                        }
                        else
                        {
                            if(!hx_is_null &&
                               in_n.at(seqLength - 1 - ti) > in_n.at(seqLength - ti) &&
                               bs >= in_n.at(seqLength - ti))
                            {
                                wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] +=
                                    (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] *
                                     hx[hx_shift + bs * uni_stride + hy_n * hy_h + h] *
                                     dervactivfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            3 * hy_h + h],
                                                   2));
                            }

                            if(bs < in_n.at(seqLength - ti))
                            {
                                wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] +=
                                    (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                             hy_h + h] *
                                     rsvspace[hid_shift +
                                              (baccbi + in_n.at(seqLength - 1 - ti) + bs) *
                                                  hy_stride +
                                              bi * 3 * hy_h + hy_h + h] *
                                     dervactivfunc(rsvspace[hid_shift + (baccbi + bs) * hy_stride +
                                                            3 * hy_h + h],
                                                   2));
                            }
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h] -=
                                (wkspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                         hy_h + h] *
                                 activfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h],
                                     1) *
                                 dervactivfunc(
                                     rsvspace[hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h + h],
                                     2));
                        }

                        rsvspace[hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h + hy_h + h +
                                 numlayer * batch_n * hy_stride] =
                            wkspace[hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h] *
                            rsvspace[hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h +
                                     numlayer * batch_n * hy_stride];
                    }
                }
            }

            baccbi += in_n.at(seqLength - 1 - ti);
        }

        // dhx
        int pretime_shift = li * batch_n * hy_stride;

        gemm_cpu(&wkspace[pretime_shift],
                 hy_h * 2,
                 in_n.at(0),
                 hy_stride,
                 false,
                 &wei[weitime_shift],
                 hy_h,
                 hy_h * 2,
                 uni_stride,
                 false,
                 &dhx[hx_shift],
                 hy_h,
                 in_n.at(0),
                 uni_stride,
                 1,
                 1);

        for(int bs = 0; bs < in_n.at(0); bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                dhx[hx_shift + bs * uni_stride + h] +=
                    wkspace[pretime_shift + bs * hy_stride + bi * 3 * hy_h + h] *
                    activfunc(rsvspace[pretime_shift + bs * hy_stride + h], 2);

                dcx[hx_shift + bs * uni_stride + h] =
                    wkspace[pretime_shift + bs * hy_stride + 2 * hy_h + h] *
                    activfunc(rsvspace[pretime_shift + bs * hy_stride + hy_h + h], 2);
            }
        }

        gemm_cpu(&dcx[hx_shift],
                 hy_h,
                 in_n.at(0),
                 uni_stride,
                 false,
                 &wei[weitime_shift + 2 * hy_h * uni_stride],
                 hy_h,
                 hy_h,
                 uni_stride,
                 false,
                 &dhx[hx_shift],
                 hy_h,
                 in_n.at(0),
                 uni_stride,
                 1,
                 1);

        if(bidirection)
        {
            int ti = seqLength - 1, cur_bat = 0, pre_bat = batch_n;

            while(ti >= 0)
            {
                pre_bat -= in_n.at(ti);
                if(in_n.at(ti) > cur_bat)
                {
                    pretime_shift = li * batch_n * hy_stride + (pre_bat + cur_bat) * hy_stride;

                    gemm_cpu(&wkspace[pretime_shift + 3 * hy_h],
                             hy_h * 2,
                             (in_n.at(ti) - cur_bat),
                             hy_stride,
                             false,
                             &wei[weitime_shift + 3 * hy_h * uni_stride],
                             hy_h,
                             hy_h * 2,
                             uni_stride,
                             false,
                             &dhx[hx_shift + hy_n * hy_h + cur_bat * hy_h],
                             hy_h,
                             (in_n.at(ti) - cur_bat),
                             uni_stride,
                             1,
                             1);

                    for(int bs = cur_bat; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            dhx[hx_shift + bs * uni_stride + hy_n * hy_h + h] +=
                                wkspace[pretime_shift + (bs - cur_bat) * hy_stride + bi * 3 * hy_h +
                                        hy_h + h] *
                                activfunc(rsvspace[pretime_shift + (bs - cur_bat) * hy_stride +
                                                   3 * hy_h + h],
                                          2);

                            dcx[hx_shift + bs * uni_stride + hy_n * hy_h + h] =
                                wkspace[pretime_shift + (bs - cur_bat) * hy_stride + 5 * hy_h + h] *
                                activfunc(rsvspace[pretime_shift + (bs - cur_bat) * hy_stride +
                                                   4 * hy_h + h],
                                          2);
                        }
                    }

                    gemm_cpu(&dcx[hx_shift + hy_n * hy_h + cur_bat * hy_h],
                             hy_h,
                             (in_n.at(ti) - cur_bat),
                             uni_stride,
                             false,
                             &wei[weitime_shift + 5 * hy_h * uni_stride],
                             hy_h,
                             hy_h,
                             uni_stride,
                             false,
                             &dhx[hx_shift + hy_n * hy_h + cur_bat * hy_h],
                             hy_h,
                             (in_n.at(ti) - cur_bat),
                             uni_stride,
                             1,
                             1);
                }
                cur_bat = in_n.at(ti--);
            }
        }
    }

    // dinput
    if(inputMode == 1)
    {
        for(int bs = 0; bs < batch_n; bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                for(int gi = 0; gi < 3; gi++)
                {
                    din[bs * in_stride + h] += wkspace[bs * hy_stride + gi * hy_h + h];
                    if(bidirection)
                    {
                        din[bs * in_stride + h] += wkspace[bs * hy_stride + (gi + 3) * hy_h + h];
                    }
                }
            }
        }
    }
    else
    {
        gemm_cpu(wkspace.data(),
                 hy_h * bi * 3,
                 batch_n,
                 hy_stride,
                 false,
                 wei.data(),
                 in_h,
                 hy_h * bi * 3,
                 in_stride,
                 false,
                 din.data(),
                 in_h,
                 batch_n,
                 in_stride,
                 1,
                 1);
    }
}

template <typename T>
void GRUBwdWeightCPUVerify(bool use_dropout,
                           std::vector<T>& in,
                           std::vector<T>& dwei,         // [ input_state_weight_trans
                                                         // hidden_state_weight0_trans
                                                         // input1_trans hidden1_trans ...
                                                         // output_weight; bidirectional
                                                         // reversed weights ]
                           std::vector<T>& hx,           // initial hidden state
                           const std::vector<int>& in_n, // input batch size
                           int in_h,                     // input data length
                           int seqLength,                // Number of iterations to unroll over
                           bool bidirection,             // whether using bidirectional net
                           bool biased,                  // whether using bias
                           int hy_d, // 1 by numlayer (number of stacks of hidden
                                     // layers) for unidirection, 2 by numlayer for
                                     // bidirection
                           int hy_n, // equal to input batch size in_n[0]
                           int hy_h, // hidden state number
                                     // by hy_h related function for bidirection
                           int inputMode,
                           std::vector<T>& rsvspace,
                           std::vector<T>& wkspace,
                           bool hx_is_null = false)
{
    int batch_n  = sumvc(in_n);
    int numlayer = bidirection ? hy_d / 2 : hy_d;
    int bi       = bidirection ? 2 : 1;

    int in_stride  = in_h;
    int wei_stride = bi * 3 * hy_h;
    int hy_stride  = bi * 4 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    int wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;

    // bwd weights emulator
    for(int li = 0; li < numlayer; li++)
    {
        // between layers
        if(li == 0)
        {
            if(inputMode == 0)
            {
                gemm_cpu(wkspace.data(),
                         hy_h * bi * 3,
                         batch_n,
                         hy_stride,
                         true,
                         in.data(),
                         in_h,
                         batch_n,
                         in_stride,
                         false,
                         dwei.data(),
                         in_h,
                         hy_h * bi * 3,
                         in_stride,
                         1,
                         1);
            }

            if(biased)
            {
                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei[wei_shift_bias + h] += wkspace[w * hy_stride + h];
                    }
                }
            }
        }
        else
        {
            int prelayer_shift =
                use_dropout ? 2 * numlayer * batch_n * hy_stride + (li - 1) * batch_n * hy_h * bi
                            : (li - 1) * batch_n * hy_stride + bi * hy_h * 3;
            int hid_shift = li * batch_n * hy_stride;
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

            gemm_cpu(&wkspace[hid_shift],
                     hy_h * bi * 3,
                     batch_n,
                     hy_stride,
                     true,
                     &rsvspace[prelayer_shift],
                     hy_h * bi,
                     batch_n,
                     use_dropout ? hy_h * bi : hy_stride,
                     false,
                     &dwei[wei_shift],
                     hy_h * bi,
                     hy_h * bi * 3,
                     bi_stride,
                     1,
                     1);

            if(biased)
            {
                wei_shift = wei_shift_bias + li * 2 * wei_stride;

                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n; w++)
                    {
                        dwei[wei_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                    }
                }
            }
        }

        // between time
        int bacc = 0;
        for(int ti = 0; ti < seqLength; ti++)
        {
            int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
            int hx_shift  = li * in_n.at(0) * h_stride;
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift;

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    wkspace[hid_shift + bs * hy_stride + 2 * hy_h + h] *=
                        activfunc(rsvspace[hid_shift + bs * hy_stride + hy_h + h], 2);
                }
            }

            // between time
            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    gemm_cpu(&wkspace[hid_shift],
                             hy_h * 3,
                             in_n.at(ti),
                             hy_stride,
                             true,
                             &hx[hx_shift],
                             hy_h,
                             in_n.at(ti),
                             uni_stride,
                             false,
                             &dwei[wei_shift],
                             hy_h,
                             hy_h * 3,
                             uni_stride,
                             1,
                             1);

                    if(biased)
                    {
                        int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 3; h++)
                        {
                            for(int w = 0; w < in_n.at(ti); w++)
                            {
                                dwei[bias_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                            }
                        }
                    }
                }
            }
            else
            {
                pretime_shift =
                    li * batch_n * hy_stride + (bacc - in_n.at(ti - 1)) * hy_stride + bi * 3 * hy_h;

                gemm_cpu(&wkspace[hid_shift],
                         hy_h * 3,
                         in_n.at(ti),
                         hy_stride,
                         true,
                         &rsvspace[pretime_shift],
                         hy_h,
                         in_n.at(ti),
                         hy_stride,
                         false,
                         &dwei[wei_shift],
                         hy_h,
                         hy_h * 3,
                         uni_stride,
                         1,
                         1);

                if(biased)
                {
                    int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                    for(int h = 0; h < hy_h * 3; h++)
                    {
                        for(int w = 0; w < in_n.at(ti); w++)
                        {
                            dwei[bias_shift + h] += wkspace[hid_shift + w * hy_stride + h];
                        }
                    }
                }
            }

            if(bidirection)
            {
                for(int bs = 0; bs < in_n.at(ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        wkspace[hid_shift + bs * hy_stride + 5 * hy_h + h] *=
                            activfunc(rsvspace[hid_shift + bs * hy_stride + 4 * hy_h + h], 2);
                    }
                }

                if(ti == seqLength - 1)
                {
                    if(!hx_is_null)
                    {
                        gemm_cpu(&wkspace[hid_shift + 3 * hy_h],
                                 hy_h * 3,
                                 in_n.at(ti),
                                 hy_stride,
                                 true,
                                 &hx[hx_shift + hy_n * hy_h],
                                 hy_h,
                                 in_n.at(ti),
                                 uni_stride,
                                 false,
                                 &dwei[wei_shift + 3 * hy_h * uni_stride],
                                 hy_h,
                                 hy_h * 3,
                                 uni_stride,
                                 1,
                                 1);

                        if(biased)
                        {
                            int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 3; h++)
                            {
                                for(int w = 0; w < in_n.at(ti); w++)
                                {
                                    dwei[bias_shift + 3 * hy_h + h] +=
                                        wkspace[hid_shift + 3 * hy_h + w * hy_stride + h];
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(!hx_is_null && in_n.at(ti) > in_n.at(ti + 1))
                    {
                        gemm_cpu(&wkspace[hid_shift + 3 * hy_h + in_n.at(ti + 1) * hy_stride],
                                 hy_h * 3,
                                 (in_n.at(ti) - in_n.at(ti + 1)),
                                 hy_stride,
                                 true,
                                 &hx[hx_shift + hy_n * hy_h + in_n.at(ti + 1) * hy_h],
                                 hy_h,
                                 (in_n.at(ti) - in_n.at(ti + 1)),
                                 uni_stride,
                                 false,
                                 &dwei[wei_shift + 3 * hy_h * uni_stride],
                                 hy_h,
                                 hy_h * 3,
                                 uni_stride,
                                 1,
                                 1);

                        if(biased)
                        {
                            int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 3; h++)
                            {
                                for(int w = in_n.at(ti + 1); w < in_n.at(ti); w++)
                                {
                                    dwei[bias_shift + 3 * hy_h + h] +=
                                        wkspace[hid_shift + 3 * hy_h + w * hy_stride + h];
                                }
                            }
                        }
                    }

                    pretime_shift =
                        li * batch_n * hy_stride + (bacc + in_n.at(ti)) * hy_stride + bi * 3 * hy_h;

                    gemm_cpu(&wkspace[hid_shift + 3 * hy_h],
                             hy_h * 3,
                             in_n.at(ti + 1),
                             hy_stride,
                             true,
                             &rsvspace[pretime_shift + hy_h],
                             hy_h,
                             in_n.at(ti + 1),
                             hy_stride,
                             false,
                             &dwei[wei_shift + 3 * hy_h * uni_stride],
                             hy_h,
                             hy_h * 3,
                             uni_stride,
                             1,
                             1);

                    if(biased)
                    {
                        int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 3; h++)
                        {
                            for(int w = 0; w < in_n.at(ti + 1); w++)
                            {
                                dwei[bias_shift + 3 * hy_h + h] +=
                                    wkspace[hid_shift + 3 * hy_h + w * hy_stride + h];
                            }
                        }
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }
}

//////=========END CPU VERIFICATION FUNCTIONS=============

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

        GRUFwdCPUVerify(handle,
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
                        nohx);
        return std::make_tuple(output);
    }

    std::tuple<std::vector<T>> gpu()
    {
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
        ss << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen << " -l "
           << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p " << inputMode
           << std::endl;

        ss << "inputMode: " << inputMode << " biasMode: " << biasMode << " dirMode: " << dirMode
           << std::endl;
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

        GRUFwdCPUVerify(handle,
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
                        nohx);

        auto retSet = std::make_tuple(output, (nohy ? initHidden : hiddenState), reserveSpace);
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> gpu()
    {
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

        auto outdata = handle.Read<T>(output_dev, output.size());

        auto retSet = std::make_tuple(outdata,
                                      (nohy ? initHidden : handle.Read<T>(hy_dev, hy.size())),
                                      rspace.Read<std::vector<T>>());
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
        ss << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen << " -l "
           << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p " << inputMode
           << std::endl;

        ss << "inputMode: " << inputMode << " biasMode: " << biasMode << " dirMode: " << dirMode
           << std::endl;
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

        GRUBwdDataCPUVerify(use_dropout,
                            miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                            dx,              // DX (output)
                            weights,         // [ input_state_weight_trans
                                             //   hidden_state_weight0_trans input1_trans
                                             //   hidden1_trans ... output_weight;
                                             //   bidirectional reversed weights ]
                            dhy,             // current/final hidden state
                            dhx,             // DHX (output)
                            initHidden,      // HX initial hidden state
                            yin,             // Y
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
                            nohx,
                            nodhy);
        return std::make_tuple(dx, (nodhx ? initHidden : dhx), reserveSpace, workSpace);
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> gpu()
    {
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
        return std::make_tuple(handle.Read<T>(dx_dev, dx.size()),
                               (nodhx ? initHidden : handle.Read<T>(dhx_dev, dhx.size())),
                               rspace.Read<std::vector<T>>(),
                               wspace.Read<std::vector<T>>());
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
        ss << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen << " -l "
           << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p " << inputMode
           << std::endl;
        ss << "inputMode: " << inputMode << " biasMode: " << biasMode << " dirMode: " << dirMode
           << std::endl;
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
        int bi = dirMode ? 2 : 1;
        std::vector<T> dweights(weightSize);
        GRUBwdWeightCPUVerify(use_dropout,
                              input,
                              dweights,        // (output) [ input_state_weight_trans
                                               // hidden_state_weight0_trans
                                               // input1_trans hidden1_trans ...
                                               // output_weight; bidirectional
                                               // reversed weights ]
                              initHidden,      // initial hidden state
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
                              inputMode,
                              reserveSpace,
                              workSpace,
                              nohx);
        return std::make_tuple(dweights);
    }

    std::tuple<std::vector<T>> gpu()
    {
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
        ss << " -m gru -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen << " -l "
           << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p " << inputMode
           << std::endl;
        ss << "inputMode: " << inputMode << " biasMode: " << biasMode << " dirMode: " << dirMode
           << std::endl;
        ss << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
           << " inputLen: " << inputVecLen << " numLayers: " << nLayers
           << " useDropout: " << int(use_dropout) << std::endl;
        ss << "Backward Weights GRU: " << std::endl;
        GTEST_FAIL() << ss.str();
    }
};
//~~~~~~~~~~~~ END BACKWARD WEIGHTS ~~~~~~~~~~~~~~~~~~~~~~~~

struct TestCase
{
    int batchSize{};
    int seqLength{};
    int inVecLen{};
    int hiddenSize{};
    int numLayers{};
    int inputMode{};
    int biasMode{};
    int dirMode{};
    bool useDropout    = false;
    bool nohx          = false;
    bool nodhy         = false;
    bool nohy          = false;
    bool nodhx         = false;
    bool flatBatchFill = false;
    std::vector<int> batchSeq{};
};

using TestParam = std::tuple<int,
                             int,
                             int,
                             int,
                             int,
                             int,
                             int,
                             int,
                             bool,
                             bool,
                             bool,
                             bool,
                             bool,
                             bool,
                             std::vector<int>>;

TestCase ConvertParam(TestParam const& param)
{
    auto [batchSize,
          seqLength,
          inVecLen,
          hiddenSize,
          numLayers,
          inputMode,
          biasMode,
          dirMode,
          useDropout,
          nohx,
          nodhy,
          nohy,
          nodhx,
          flatBatchFill,
          batchSeq] = param;

    return TestCase{batchSize,
                    seqLength,
                    inVecLen,
                    hiddenSize,
                    numLayers,
                    inputMode,
                    biasMode,
                    dirMode,
                    useDropout,
                    nohx,
                    nodhy,
                    nohy,
                    nodhx,
                    flatBatchFill,
                    batchSeq};
}

std::vector<TestParam> deepbench_cases = {
    {32, 1500, 216, 216, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 750, 286, 286, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 375, 286, 286, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 10, 2816, 2816, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 1500, 248, 248, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 12, 2048, 2048, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 1500, 156, 156, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 500, 156, 156, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 12, 1536, 1536, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 1500, 256, 256, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 500, 256, 256, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 10, 2560, 2560, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 1, 512, 512, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {32, 50, 1024, 1024, 1, 1, 0, 0, false, false, false, false, false, true, {0}},
    {64, 50, 1024, 1024, 1, 1, 0, 0, false, false, false, false, false, true, {0}}};

std::vector<TestParam> extra_cases = {
    {32, 3, 128, 128, 1, 0, 0, 0, false, true, false, false, false, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 0, false, false, true, false, false, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 0, false, true, true, false, false, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 1, false, true, false, false, false, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 1, false, false, true, false, false, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 1, false, true, true, false, false, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 0, false, false, false, false, true, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 0, false, false, false, true, true, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 1, false, false, false, true, false, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 1, false, false, false, false, true, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 1, false, false, false, true, true, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 0, false, true, true, true, true, false, {32, 32, 32}},
    {32, 3, 128, 128, 1, 0, 0, 1, false, true, true, true, true, false, {32, 32, 32}}};

auto GenCases(bool gen_dropout)
{
    std::vector<int> defaultBS(1);

    std::vector<TestParam> cases{};
    cases.push_back({17,
                     (gen_dropout ? 23 : 2),
                     13,
                     67,
                     (gen_dropout ? 3 : 1),
                     0,
                     0,
                     0,
                     gen_dropout,
                     false,
                     false,
                     false,
                     false,
                     false,
                     defaultBS});
    return ::testing::ValuesIn(cases);
}

auto GetFullBaseTests()
{
    std::vector<int> modes(2, 0);
    modes[1] = 1;
    std::vector<int> defaultBS(1);
    return ::testing::Combine(::testing::ValuesIn(get_gru_batchSize()),
                              ::testing::ValuesIn(get_gru_seq_len()),
                              ::testing::ValuesIn(get_gru_vector_len()),
                              ::testing::ValuesIn(get_gru_hidden_size()),
                              ::testing::ValuesIn(get_gru_num_layers()),
                              ::testing::ValuesIn(modes),
                              ::testing::ValuesIn(modes),
                              ::testing::ValuesIn(modes),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({defaultBS}));
}

auto GetDropoutTests()
{
    std::vector<int> modes(2, 0);
    modes[1] = 1;
    std::vector<int> defaultBS(1);
    return ::testing::Combine(::testing::ValuesIn({17}),
                              ::testing::ValuesIn({23}),
                              ::testing::ValuesIn({13}),
                              ::testing::ValuesIn({67}),
                              ::testing::ValuesIn({3}),
                              ::testing::ValuesIn({0}),
                              ::testing::ValuesIn({0}),
                              ::testing::ValuesIn(modes),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({false}),
                              ::testing::ValuesIn({defaultBS}));
}

auto const& GetDropoutSmokeTests()
{
    static auto cases = GenCases(true);
    return cases;
}

auto const& GetDeepbenchTests() { return deepbench_cases; }

auto const& GetExtraTests() { return extra_cases; }

auto const& GetSmokeTests()
{
    static auto cases = GenCases(false);
    return cases;
}

template <class T>
class gru_test : public testing::TestWithParam<TestParam>
{
    const double tolerance = 80; // Will be multiplied by std::numeric_limits<T>::epsilon()
    TestCase param;
    int batch_n{};
    size_t statesSizeInBytes{0U};

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

        const double data_max_v = sqrt(1. / param.hiddenSize);
        int data_range          = 100;
        const double data_scale = data_max_v / data_range;
        fill_array_via_gen(input, pos_gen(data_scale, data_range), 0);

        if(!param.nohx)
        {
            fill_array_via_gen(hx, pos_gen(data_scale, data_range), 1);
        }

        // filter
        const double weights_max_v = sqrt(1. / param.hiddenSize);
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

        const double bwd_data_max_v = sqrt(1. / param.hiddenSize) / 8;
        int bwd_data_range          = 100;
        const double bwd_data_scale = bwd_data_max_v / bwd_data_range;

        if(!param.nodhy)
        {
            fill_array_via_gen(dhy, sign_gen(bwd_data_scale, bwd_data_range), 3);
        }

        fill_array_via_gen(dy, sign_gen(bwd_data_scale, bwd_data_range), 4);
        prng::reset_seed();
    }

    void SetUp() override
    {
        prng::reset_seed();
        param = ConvertParam(GetParam());
        if(!param.useDropout)
        {
            param.batchSeq = generate_batchSeq(param.batchSize, param.seqLength)[0];
        }
    }

    void Run()
    {
        if(param.batchSeq.empty() || 0 == param.batchSeq[0])
        {
            std::cout << "Empty batch sequence. Filling uniformly with batch size: "
                      << param.batchSize << std::endl;
            if(param.flatBatchFill)
            {
                param.batchSeq.clear();
                param.batchSeq.resize(param.seqLength, param.batchSize);
            }
            else
            {
                param.batchSeq = GenBatchSeq(param.batchSize, param.seqLength);
            }
        }

        if(param.batchSeq.size() != param.seqLength)
        {
            std::cerr << "FAILED: Batch sequence vector length, does not match sequence length."
                      << std::endl;
            GTEST_SKIP();
        }

        auto&& handle = get_handle();

        batch_n = std::accumulate(param.batchSeq.begin(), param.batchSeq.end(), 0);

        RNNDescGuard rnnDesc;
        miopenRNNAlgo_t algoMode = miopenRNNdefault;

        DropoutDescGuard DropoutDesc;

        HandleGuard mio_handle;
        void* dropout_state_buf = nullptr;

        // See DestroyInternalRnnDropoutDesc — frees the descriptor allocated
        // by miopenCreateRNNDescriptor that the upcoming Set* will leak.
        DestroyInternalRnnDropoutDesc(rnnDesc);

        if(param.useDropout)
        {
            mio_handle.create(handle.GetStream());

            float dropout_rate              = 0.5;
            unsigned long long dropout_seed = 0ULL;
            miopenDropoutGetStatesSize(mio_handle, &statesSizeInBytes);

            (void)hipMalloc(static_cast<void**>(&dropout_state_buf), statesSizeInBytes);

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
                                      param.hiddenSize,
                                      param.numLayers,
                                      DropoutDesc,
                                      miopenRNNInputMode_t(param.inputMode),
                                      miopenRNNDirectionMode_t(param.dirMode),
                                      miopenGRU,
                                      miopenRNNBiasMode_t(param.biasMode),
                                      miopenRNNAlgo_t(algoMode),
                                      miopen_type<T>{});
        }
        else
        {
            miopenSetRNNDescriptor(rnnDesc,
                                   param.hiddenSize,
                                   param.numLayers,
                                   miopenRNNInputMode_t(param.inputMode),
                                   miopenRNNDirectionMode_t(param.dirMode),
                                   miopenGRU,
                                   miopenRNNBiasMode_t(param.biasMode),
                                   miopenRNNAlgo_t(algoMode),
                                   miopen_type<T>{}); // defined in superclass testdriver
        }

        // Create input tensor
        // If we are in skip mode, take the real input size to be the vector length.
        auto inVecReal    = (param.inputMode != 0) ? param.hiddenSize : param.inVecLen;
        std::size_t in_sz = static_cast<std::size_t>(inVecReal) * batch_n;
        std::size_t hx_sz = ((param.dirMode != 0) ? 2ULL : 1ULL) * param.hiddenSize *
                            param.batchSize * param.numLayers;

        std::vector<T> input(in_sz), hx(hx_sz), dhyin(hx_sz);

        size_t wei_bytes = [&]() {
            size_t filter_bytes;
            std::vector<int> inlens(2, 0);
            inlens.at(0)        = param.batchSeq.at(0);
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
            inputCPPDescs, inputDescs, param.batchSeq, inVecReal, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              param.batchSeq,
                              param.hiddenSize * ((param.dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        size_t out_sz;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, param.seqLength, outputDescs.data(), &out_sz);
        size_t reserveSpaceSize;
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, param.seqLength, inputDescs.data(), &reserveSpaceSize);
        size_t workspace_size;
        miopenGetRNNWorkspaceSize(
            &handle, rnnDesc, param.seqLength, inputDescs.data(), &workspace_size);

        size_t total_mem = statesSizeInBytes + reserveSpaceSize + workspace_size + 2 * out_sz +
                           (in_sz + wei_sz + (param.nohx ? 0 : hx_sz) + (param.nohy ? 0 : hx_sz) +
                            (param.nodhx ? 0 : hx_sz) + (param.nodhy ? 0 : hx_sz)) *
                               sizeof(T);
        size_t device_mem = handle.GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            GTEST_SKIP() << "Config requires " << total_mem
                         << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                         << " Bytes of memory." << std::endl;
        }

        fill_buffers(input, hx, weights);

        auto fwdTrainOutputPair =
            GruTestCompareResults(verify_forward_train_gru<T>{rnnDesc,
                                                              input,
                                                              hx,
                                                              weights,
                                                              param.batchSeq,
                                                              param.hiddenSize,
                                                              batch_n,
                                                              param.seqLength,
                                                              param.numLayers,
                                                              param.biasMode,
                                                              param.dirMode,
                                                              param.inputMode,
                                                              inVecReal,
                                                              hx_sz,
                                                              param.nohx,
                                                              param.nohy,
                                                              param.useDropout},
                                  tolerance);

        /// RETURNS std::make_tuple(output, hiddenState, reserveSpace);
        auto yin = std::get<0>(fwdTrainOutputPair.second);
        // auto curHiddenState       = std::get<1>(fwdTrainOutputPair.second);
        auto reserveSpaceFwdTrain = std::get<2>(fwdTrainOutputPair.second);

        std::vector<T> dyin(yin.size());

        fill_bwd_buffers(dyin, dhyin);

        auto bwdDataOutputPair =
            GruTestCompareResults(verify_backward_data_gru<T>{rnnDesc,
                                                              yin,
                                                              dyin,
                                                              dhyin,
                                                              hx,
                                                              weights,
                                                              reserveSpaceFwdTrain,
                                                              param.batchSeq,
                                                              param.hiddenSize,
                                                              batch_n,
                                                              param.seqLength,
                                                              param.numLayers,
                                                              param.biasMode,
                                                              param.dirMode,
                                                              param.inputMode,
                                                              inVecReal,
                                                              hx_sz,
                                                              param.nohx,
                                                              param.nodhy,
                                                              param.nodhx,
                                                              param.useDropout},
                                  tolerance);

        // RETURNS:  std::make_tuple(dx, dhx, reserveSpace, workSpace);
        auto reserveSpaceBwdData = std::get<2>(bwdDataOutputPair.second);
        auto workSpaceBwdData    = std::get<3>(bwdDataOutputPair.second);
        // auto dweights_pair       =
        GruTestCompareResults(verify_backward_weights_gru<T>{rnnDesc,
                                                             input,
                                                             dyin,
                                                             hx,
                                                             reserveSpaceBwdData,
                                                             workSpaceBwdData,
                                                             param.batchSeq,
                                                             param.hiddenSize,
                                                             static_cast<int>(wei_sz),
                                                             batch_n,
                                                             param.seqLength,
                                                             param.numLayers,
                                                             param.biasMode,
                                                             param.dirMode,
                                                             param.inputMode,
                                                             inVecReal,
                                                             hx_sz,
                                                             param.nohx,
                                                             param.useDropout},
                              tolerance);

        if(!param.useDropout)
        {
            GruTestCompareResults(verify_forward_infer_gru<T>{rnnDesc,
                                                              input,
                                                              hx,
                                                              weights,
                                                              param.batchSeq,
                                                              param.hiddenSize,
                                                              batch_n,
                                                              param.seqLength,
                                                              param.numLayers,
                                                              param.biasMode,
                                                              param.dirMode,
                                                              param.inputMode,
                                                              inVecReal,
                                                              hx_sz,
                                                              param.nohx,
                                                              param.nohy},
                                  tolerance);
        }
        // DLOWELL: Subtracting delta weights may produce NAN and infinities. Further investigation
        // is needed.
        //        auto dweights = std::get<1>(dweights_pair);
        //        std::transform(weightData.begin( ), weightData.end( ), dweights.begin( ),
        //        weightData.begin( ),std::minus<T>( ));
        //        GruTestCompareResults(verify_forward_infer_gru<T>{rnnDesc, inputData,
        //                                        curHiddenState, curCellState, weightData,
        //                                        batchSeq,
        //                                        hiddenSize, batch_n,
        //                                        seqLength, numLayers,
        //                                        biasMode, dirMode,
        //                                        inputMode, inVecReal});

        // Free the DropoutDescriptor that miopenSetRNNDescriptor just allocated.
        // In the dropout path, the internal pointer aliases the user-owned
        // DropoutDescGuard — freeing it would double-free.
        if(!param.useDropout)
            DestroyInternalRnnDropoutDesc(rnnDesc);

        if(param.useDropout)
            (void)hipFree(dropout_state_buf);
    }
};

struct TestNameGenerator
{
    std::string operator()(const ::testing::TestParamInfo<TestParam>& param_info)
    {
        std::stringstream ss{};
        auto print_bool = [](bool value) { return value ? "true" : "false"; };

        auto print_batch_seq = [](std::vector<int> const& vec) {
            std::stringstream vec_ss{};
            for(auto el : vec)
            {
                vec_ss << std::to_string(el) << "_";
            }
            return vec_ss.str();
        };

        auto param = ConvertParam(param_info.param);

        ss << "batchSize_" << param.batchSize << "_seqLength_" << param.seqLength << "_inVecLen_"
           << param.inVecLen << "_hiddenSize_" << param.hiddenSize << "_numLayers_"
           << param.numLayers << "_inputMode_" << param.inputMode << "_biasMode_" << param.biasMode
           << "_dirMode_" << param.dirMode << "_useDropout_" << print_bool(param.useDropout)
           << "_nohx_" << print_bool(param.nohx) << "_nodhy_" << print_bool(param.nodhy) << "_nohy_"
           << print_bool(param.nohy) << "_nodhx_" << print_bool(param.nodhx) << "_flatBatchFill_"
           << print_bool(param.flatBatchFill) << "_batch_seq_" << print_batch_seq(param.batchSeq);
        return ss.str();
    }
};

template <typename T>
struct gru_dropout_test : public gru_test<T>
{
    // intentionally empty
};

template <typename T>
struct gru_deepbench_test : public gru_test<T>
{
    // intentionally empty
};

template <typename T>
struct gru_extra_test : public gru_test<T>
{
    // intentionally empty
};

} // anonymous namespace

using GPU_GRU_Base_FP32 = gru_test<float>;
using GPU_GRU_Base_FP16 = gru_test<half_float::half>;

using GPU_GRU_Dropout_FP32 = gru_dropout_test<float>;
using GPU_GRU_Dropout_FP16 = gru_dropout_test<half_float::half>;

using GPU_GRU_Deepbench_FP32 = gru_deepbench_test<float>;
using GPU_GRU_Deepbench_FP16 = gru_deepbench_test<half_float::half>;

using GPU_GRU_Extra_FP32 = gru_extra_test<float>;
using GPU_GRU_Extra_FP16 = gru_extra_test<half_float::half>;

TEST_P(GPU_GRU_Base_FP32, TestFloat32) { Run(); }
TEST_P(GPU_GRU_Base_FP16, TestFloat16) { Run(); }

// Base tests
INSTANTIATE_TEST_SUITE_P(Full, GPU_GRU_Base_FP32, GetFullBaseTests(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_GRU_Base_FP32, GetSmokeTests(), TestNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Full, GPU_GRU_Base_FP16, GetFullBaseTests(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_GRU_Base_FP16, GetSmokeTests(), TestNameGenerator{});

// Dropout tests
TEST_P(GPU_GRU_Dropout_FP32, TestFloat32) { Run(); }
TEST_P(GPU_GRU_Dropout_FP16, TestFloat16) { Run(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_GRU_Dropout_FP32, GetDropoutTests(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_GRU_Dropout_FP32, GetDropoutSmokeTests(), TestNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Full, GPU_GRU_Dropout_FP16, GetDropoutTests(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_GRU_Dropout_FP16, GetDropoutSmokeTests(), TestNameGenerator{});

// // Deepbench tests
TEST_P(GPU_GRU_Deepbench_FP32, TestFloat32) { Run(); }
TEST_P(GPU_GRU_Deepbench_FP16, TestFloat16) { Run(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GRU_Deepbench_FP32,
                         ::testing::ValuesIn(GetDeepbenchTests()),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GRU_Deepbench_FP16,
                         ::testing::ValuesIn(GetDeepbenchTests()),
                         TestNameGenerator{});

// Extra tests
TEST_P(GPU_GRU_Extra_FP32, TestFloat32) { Run(); }
TEST_P(GPU_GRU_Extra_FP16, TestFloat16) { Run(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GRU_Extra_FP32,
                         ::testing::ValuesIn(GetExtraTests()),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GRU_Extra_FP16,
                         ::testing::ValuesIn(GetExtraTests()),
                         TestNameGenerator{});
