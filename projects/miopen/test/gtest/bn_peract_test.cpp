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

#include <cfloat>
#include <cmath>
#include <ctime>

#include <array>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>

#include "driver.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "test_parameter_name_generator.hpp"
#include "verify.hpp"

// Run CPU emulations in hierarchical reduction mode.
// #define MIO_HEIRARCH_SEL 0
#define MIO_BN_TEST_EXPAVGFACTOR 0.1
#define MIO_BN_TEST_EPSILON 1e-5
#define MIO_BN_USE_MIX_PREC 1
#if MIO_BN_USE_MIX_PREC == 1
#define PREC_TYPE float
#else
#define PREC_TYPE T
#endif

namespace {

using TestCase = NamedContainer<std::vector<int>>;

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T, class U>
struct verify_forward_train_bn_per_activation
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> cpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, height, width};
        std::fill(out.begin(), out.end(), 0);

        std::size_t rs_n_batch, rs_channels, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);

        std::tie(rs_n_batch, rs_channels, rs_height, rs_width) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        tensor<U> runMean;
        tensor<U> runVar;

        if(input.desc.GetType() == miopenFloat)
        {
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
            runVar = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
        }
        else
        {
            prng::reset_seed();
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width};
            runVar  = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width};

            const double Data_scale = 0.001;
            for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
            {
                runMean[i] = prng::gen_descreet_uniform_sign<U>(Data_scale, 100);
                runVar[i]  = prng::gen_descreet_unsigned<U>(Data_scale, 100);
            }
        }

        auto saveMean   = tensor<U>{1, channels, height, width};
        auto saveInvVar = tensor<U>{1, channels, height, width};
        const auto n    = double(n_batch);

        miopen::par_for(channels, 1, [&](int cidx) {
            double mean_accum     = 0.;
            double variance_accum = 0.;
            double elemStd        = 0.;
            double elemInvVar     = 0.;
            double inhat          = 0.;
            double newRunMean     = 0.;
            double adjust         = 0.;

            // process the batch per channel
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns

                    mean_accum = 0.;
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // #1 calculate the mean :: iterating through the stack of images in the
                        // mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    }
                    mean_accum /= n;

                    elemStd = variance_accum = 0.;
                    // #2 calculate the variances :: sigma^2 = (1/batch_mean) * sum( (x_i -
                    // batch_mean)^2 )
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        elemStd = (input(bidx, cidx, row, column) -
                                   mean_accum); // (x_i - mean) //this is reused but needs recalc
                        variance_accum += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                    } // end for(n)
                    variance_accum /= n; // (1/N)*sum{ (x_i - mean)^2 }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    elemInvVar = 1.0 / double(sqrt(variance_accum + epsilon));

                    // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum -
                    // epsilon)
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {                                                            // via mini_batch
                        elemStd = (input(bidx, cidx, row, column) - mean_accum); // (x_i - mean)
                        inhat   = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, row, column) * inhat + shift(0, cidx, row, column);
                    } // end for(n_batch)

                    newRunMean = runMean(0, cidx, row, column) * (1.0 - expAvgFactor);
                    runMean(0, cidx, row, column) =
                        mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp

                    // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
                    adjust = (n_batch == 1) ? variance_accum : (n / (n - 1.0)) * variance_accum;
                    runVar(0, cidx, row, column) =
                        (1 - expAvgFactor) * runVar(0, cidx, row, column) + expAvgFactor * adjust;

                    saveMean(0, cidx, row, column)   = mean_accum;
                    saveInvVar(0, cidx, row, column) = elemInvVar;

                } // for (column)
            } // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train_bn_per_activation pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> gpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        std::size_t rs_n_batch, rs_channels, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);

        std::tie(rs_n_batch, rs_channels, rs_height, rs_width) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        tensor<U> runMean;
        tensor<U> runVar;

        if(input.desc.GetType() == miopenFloat)
        {
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
            runVar = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
        }
        else
        {
            prng::reset_seed();
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width};
            runVar  = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width};

            const double Data_scale = 0.001;
            for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
            {
                runMean[i] = prng::gen_descreet_uniform_sign<U>(Data_scale, 100);
                runVar[i]  = prng::gen_descreet_unsigned<U>(Data_scale, 100);
            }
        }

        auto saveMean   = tensor<U>{1, channels, height, width};
        auto saveInvVar = tensor<U>{1, channels, height, width};

        // in buffers
        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        // out buffers
        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<U>(channels * height * width);
        auto saveInvVar_dev = handle.Create<U>(channels * height * width);
        auto out_dev        = handle.Create<T>(n_batch * channels * height * width);

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        float alpha = 1.;
        float beta  = 0.;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardTraining(handle,
                                         miopenBNPerActivation,
                                         &alpha,
                                         &beta,
                                         input.desc,
                                         in_dev.get(),
                                         out.desc,
                                         out_dev.get(),
                                         scale.desc,
                                         shift.desc,
                                         shift.desc,
                                         shift.desc,
                                         scale_dev.get(),
                                         shift_dev.get(),
                                         expAvgFactor,
                                         runMean_dev.get(),
                                         runVar_dev.get(),
                                         epsilon,
                                         saveMean_dev.get(),
                                         saveInvVar_dev.get(),
                                         actDesc);

        saveMean.data   = handle.Read<U>(saveMean_dev, saveMean.data.size());
        saveInvVar.data = handle.Read<U>(saveInvVar_dev, saveInvVar.data.size());
        runMean.data    = handle.Read<U>(runMean_dev, runMean.data.size());
        runVar.data     = handle.Read<U>(runVar_dev, runVar.data.size());
        out.data        = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train_bn_per_activation pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    void fail(int badtensor) const
    {
        std::cout << "Forward Train Per Activation Batch Normalization: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Running Mean output tensor failed verification." << std::endl; break;
        case(2):
            std::cout << "Running Variance output tensor failed verification." << std::endl;
            break;
        case(3): std::cout << "Saved Mean tensor failed verification." << std::endl; break;
        case(4): std::cout << "Saved Variance tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};

//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T, class U>
struct verify_forward_infer_bn_per_activation_recalc
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;

    tensor<T> cpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, height, width};
        std::fill(out.begin(), out.end(), 0);

        const auto n = double(n_batch);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd        = 0.;
            double elemInvVar     = 0.;
            double mean_accum     = 0.;
            double variance_accum = 0.;
            double inhat          = 0.;

            // process the batch per channel
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    mean_accum = 0.;

                    // #1 calculate the mean
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // iterating through the stack of images in the mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    }
                    mean_accum /= n;

                    elemStd        = 0.;
                    variance_accum = 0.;
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {                                                          // via mini_batch
                        elemStd = input(bidx, cidx, row, column) - mean_accum; // (x_i - mean)
                        variance_accum += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                    } // end for(n)
                    variance_accum /= n; // (1/N)*sum{ (x_i - mean)^2 }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    elemInvVar = 1.0 / double(sqrt(variance_accum + epsilon));

                    // #4 apply the normalization
                    // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        elemStd = input(bidx, cidx, row, column) - mean_accum; // (x_i - mean)
                        inhat   = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust // y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, row, column) * inhat + shift(0, cidx, row, column);
                    } // end for(n_batchs)
                } // for (column)
            } // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    tensor<T> gpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);
        auto out_dev   = handle.Write(out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        float alpha = 1.;
        float beta  = 0.;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardInference(handle,
                                          miopenBNPerActivation,
                                          &alpha,
                                          &beta,
                                          input.desc,
                                          in_dev.get(),
                                          out.desc,
                                          out_dev.get(),
                                          scale.desc,
                                          shift.desc,
                                          shift.desc,
                                          shift.desc,
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          nullptr,
                                          nullptr,
                                          epsilon,
                                          actDesc);
        out.data = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int) const
    {
        std::cout << "Forward Inference Per Activation Batch Normalization Recalc: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T, class U>
struct verify_forward_infer_bn_per_activation_use_est
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;
    const tensor<U> estMean;
    const tensor<U> estVar;

    tensor<T> cpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, height, width};
        std::fill(out.begin(), out.end(), 0);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd    = 0.;
            double mean       = 0.;
            double variance   = 0.;
            double inhat      = 0.;
            double elemInvVar = 0.;

            // process the batch per channel
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    mean       = estMean(0, cidx, row, column);
                    variance   = estVar(0, cidx, row, column);
                    elemInvVar = 1.0 / double(sqrt(variance + epsilon));
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {                                                    // via mini_batch
                        elemStd = input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        inhat   = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, row, column) * inhat + shift(0, cidx, row, column);
                    } // end for(n_batchs)
                } // for (column)
            } // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_per_activation_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    tensor<T> gpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev      = handle.Write(input.data);
        auto scale_dev   = handle.Write(scale.data);
        auto shift_dev   = handle.Write(shift.data);
        auto estMean_dev = handle.Write(estMean.data);
        auto estVar_dev  = handle.Write(estVar.data);
        auto out_dev     = handle.Write(out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        float alpha = 1.;
        float beta  = 0.;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardInference(handle,
                                          miopenBNPerActivation,
                                          &alpha,
                                          &beta,
                                          input.desc,
                                          in_dev.get(),
                                          out.desc,
                                          out_dev.get(),
                                          scale.desc,
                                          shift.desc,
                                          shift.desc,
                                          shift.desc,
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          estMean_dev.get(),
                                          estVar_dev.get(),
                                          epsilon,
                                          actDesc); // TODO: add multi-in
        out.data = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_per_activation_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int) const
    {
        std::cout << "Forward Inference Per Activation Batch Normalization Use Estimated: "
                  << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

//****************************************************
// BACKWARDS PROPAGATION
//****************************************************
template <class T, class U>
struct verify_backward_bn_per_activation_use_saved
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<U> scale;
    const tensor<U> savedMean;
    const tensor<U> savedInvVar;

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{1, channels, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{1, channels, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto n                  = double(n_batch);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd = 0.;
            unsigned int xhat_index;
            double mean       = 0.;
            double elemInvVar = 0.;
            double dyelem     = 0.;
            double dxhat      = 0.;
            double dxhathat   = 0.;
            double tmp1       = 0.;
            std::vector<double> xhat(n_batch * in_cstride);

            // process the batch per channel
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    dxhat    = 0.;
                    dxhathat = 0.;

                    mean       = savedMean(0, cidx, row, column);   // HxW elements
                    elemInvVar = savedInvVar(0, cidx, row, column); // HxW elements

                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * elemInvVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, row, column) += dyelem;
                        dscale(0, cidx, row, column) += xhat[xhat_index] * dyelem;
                        tmp1 = scale(0, cidx, row, column) * dyelem;
                        dxhat += tmp1;
                        dxhathat += tmp1 * xhat[xhat_index];

                    } // end for(n_batchs)

                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index  = in_cstride * bidx + (width * row + column);
                        tmp1        = xhat[xhat_index] * dxhathat + dxhat;
                        double tmp2 = n_batch * (scale(0, cidx, row, column) *
                                                 dy_input(bidx, cidx, row, column)) -
                                      tmp1;
                        double tmp3                     = elemInvVar / (double(n));
                        dx_out(bidx, cidx, row, column) = tmp3 * tmp2;
                    } // end for(n_batchs)
                } // for (column)
            } // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn_per_activation_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{1, channels, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{1, channels, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        auto xin_dev         = handle.Write(x_input.data);
        auto dyin_dev        = handle.Write(dy_input.data);
        auto scale_dev       = handle.Write(scale.data);
        auto dscale_dev      = handle.Write(dscale.data);
        auto dshift_dev      = handle.Write(dshift.data);
        auto dx_out_dev      = handle.Write(dx_out.data);
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);

        float alpha = 1.;
        float beta  = 0.;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormBackward(handle,
                                  miopenBNPerActivation,
                                  &alpha,
                                  &beta,
                                  &alpha,
                                  &beta,
                                  x_input.desc,
                                  xin_dev.get(),
                                  dy_input.desc,
                                  dyin_dev.get(),
                                  dx_out.desc,
                                  dx_out_dev.get(),
                                  scale.desc,
                                  dshift.desc,
                                  dshift.desc,
                                  dshift.desc,
                                  scale_dev.get(),
                                  nullptr,
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  savedMean_dev.get(),
                                  savedInvVar_dev.get(),
                                  actDesc);
        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<U>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<U>(dshift_dev, dshift.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward_bn_per_activation_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor) const
    {
        std::cout << "Backward Batch Per Activation Normalization Using Saved Mean and Variance: "
                  << std::endl;
        std::cout << "X Input tensor: " << x_input.desc.ToString() << std::endl;
        std::cout << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0):
            std::cout << "Delta X output tensor output failed verification." << std::endl;
            break;
        case(1): std::cout << "Delta scale output tensor failed verification." << std::endl; break;
        case(2): std::cout << "Delta shift output tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};

template <class T, class U>
struct verify_backward_bn_per_activation_recalc
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<U> scale;

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{1, channels, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{1, channels, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto n                  = double(n_batch);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd = 0.;
            unsigned int xhat_index;
            double mean       = 0.;
            double elemInvVar = 0.;
            double dyelem     = 0.;
            double variance   = 0.;
            double dxhat      = 0.;
            double dxhathat   = 0.;
            double tmp1       = 0.;
            std::vector<double> xhat(n_batch * in_cstride);

            // process the batch per channel
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    mean = 0.;
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // #1 calculate the mean
                        mean += x_input(bidx, cidx, row, column);
                    }
                    mean /= n;

                    elemStd  = 0.;
                    variance = 0.;
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        elemStd = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        variance += elemStd * elemStd;                     // sum{ (x_i - mean)^2 }
                    } // end for(n)
                    variance /= n; // (1/N)*sum{ (x_i - mean)^2 }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    elemInvVar = 1.0 / double(sqrt(variance + epsilon));

                    dxhat    = 0.;
                    dxhathat = 0.;

                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * elemInvVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, row, column) += dyelem;
                        dscale(0, cidx, row, column) += xhat[xhat_index] * dyelem;
                        tmp1 = scale(0, cidx, row, column) * dyelem;
                        dxhat += tmp1;
                        dxhathat += tmp1 * xhat[xhat_index];

                    } // end for(n_batchs)

                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index  = in_cstride * bidx + (width * row + column);
                        tmp1        = xhat[xhat_index] * dxhathat + dxhat;
                        double tmp2 = n_batch * (scale(0, cidx, row, column) *
                                                 dy_input(bidx, cidx, row, column)) -
                                      tmp1;
                        double tmp3                     = elemInvVar / double(n);
                        dx_out(bidx, cidx, row, column) = tmp3 * tmp2;
                    } // end for(n_batchs)
                } // for (column)
            } // for (row)
        });
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        // std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{1, channels, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{1, channels, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        auto xin_dev    = handle.Write(x_input.data);
        auto dyin_dev   = handle.Write(dy_input.data);
        auto scale_dev  = handle.Write(scale.data);
        auto dscale_dev = handle.Write(dscale.data);
        auto dshift_dev = handle.Write(dshift.data);
        auto dx_out_dev = handle.Write(dx_out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        float alpha = 1.;
        float beta  = 0.;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormBackward(handle,
                                  miopenBNPerActivation,
                                  &alpha,
                                  &beta,
                                  &alpha,
                                  &beta,
                                  x_input.desc,
                                  xin_dev.get(),
                                  dy_input.desc,
                                  dyin_dev.get(),
                                  dx_out.desc,
                                  dx_out_dev.get(),
                                  scale.desc,
                                  dshift.desc,
                                  dshift.desc,
                                  dshift.desc,
                                  scale_dev.get(),
                                  nullptr,
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  nullptr,
                                  nullptr,
                                  actDesc);
        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<U>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<U>(dshift_dev, dshift.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor) const
    {
        std::cout << "Backward Batch Per Activation Normalization Recalc Mean and Variance: "
                  << std::endl;
        std::cout << "X Input tensor: " << x_input.desc.ToString() << std::endl;
        std::cout << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0):
            std::cout << "Delta X output tensor output failed verification." << std::endl;
            break;
        case(1): std::cout << "Delta scale output tensor failed verification." << std::endl; break;
        case(2): std::cout << "Delta shift output tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};

//====== DRIVERS ===========================================

inline auto GenSmokeTestCases()
{
    return testing::Values(
        NamedContainer<std::vector<int>>("dims", std::vector<int>{16, 32, 8, 8}, "x"));
}

inline auto GetSmokeTestCases()
{
    static const auto cases = GenSmokeTestCases();
    return cases;
}

inline auto GenFullTestCases()
{
    return MakeNamedParameterCollectionValues<std::vector<int>>(
        "dims", get_bn_peract_inputs(4), "x");
}

inline auto GetFullTestCases()
{
    static const auto cases = GenFullTestCases();
    return cases;
}

} // namespace

template <typename T>
struct BnPeractTest : public testing::TestWithParam<TestCase>
{
    static const constexpr uint64_t MaxValue{miopen_type<T>{} == miopenHalf ? 5 : 17};

    void SetUp() override
    {
        prng::reset_seed();
        const auto dims = GetParam();
        input           = tensor<T>{dims()}.generate(tensor_elem_gen_integer{MaxValue});
    }

    void Run()
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        const auto t_start = std::chrono::high_resolution_clock::now();
#endif
        tensor<PREC_TYPE> scale;
        tensor<PREC_TYPE> shift;

        std::size_t n, c, h, w;
        std::tie(n, c, h, w) = miopen::tien<4>(input.desc.GetLengths());
        double tolerance     = 80 * input.desc.GetElementSize();

        if(n == 1)
        {
            std::cout << "Invalid batch size for batch norm tests.\nExiting...\n" << std::endl;
            return;
        }

        size_t ssn, ssc, ssh, ssw;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);
        std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());

        if(input.desc.GetType() == miopenFloat)
        {
            scale = tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{17});
            shift = tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{17});
        }
        else
        {
            scale = tensor<PREC_TYPE>{ssn, ssc, ssh, ssw};
            shift = tensor<PREC_TYPE>{ssn, ssc, ssh, ssw};

            const constexpr double data_scale = 0.001;

            for(size_t i = 0; i < scale.desc.GetElementSize(); ++i)
            {
                scale[i] = prng::gen_descreet_uniform_sign<PREC_TYPE>(data_scale, 100);
                shift[i] = prng::gen_descreet_uniform_sign<PREC_TYPE>(data_scale, 100);
            }

            for(size_t i = 0; i < input.desc.GetElementSize(); ++i)
            {
                input[i] = prng::gen_descreet_uniform_sign<T>(1e-4, 100);
            }
        }

        // train
        const auto outpair = Verify(
            verify_forward_train_bn_per_activation<T, PREC_TYPE>{input, scale, shift}, tolerance);
        // returns:  std::make_tuple(out,runMean,runVar,saveMean,saveInvVar);

        // inference recalc
        Verify(verify_forward_infer_bn_per_activation_recalc<T, PREC_TYPE>{input, scale, shift},
               tolerance);

        // inference use estimated running values
        const auto estMean = std::get<1>(outpair.second);
        const auto estVar  = std::get<2>(outpair.second);
        Verify(
            verify_forward_infer_bn_per_activation_use_est<T, PREC_TYPE>{
                input, scale, shift, estMean, estVar},
            tolerance);

        // backprop recalc
        tolerance           = 8000 * input.desc.GetElementSize();
        const auto dy_input = tensor<T>{n, c, h, w}.generate(
            tensor_elem_gen_integer{MaxValue}); //= std::get<0>(outpair.first);//
        Verify(verify_backward_bn_per_activation_recalc<T, PREC_TYPE>{input, dy_input, scale},
               tolerance);

        // backprop use saved values
        const auto savedMean   = std::get<3>(outpair.second);
        const auto savedInvVar = std::get<4>(outpair.second);
        Verify(
            verify_backward_bn_per_activation_use_saved<T, PREC_TYPE>{
                input, dy_input, scale, savedMean, savedInvVar},
            tolerance);

#if(MIO_BN_TIME_EVERYTHING == 1)
        const auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "Wall clock: full PER_ACTIVATION test pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
    }

private:
    tensor<T> input;

private:
    auto Verify(auto&& v, double tolerance) -> decltype(std::make_pair(v.cpu(), v.gpu()))
    {
        const auto cpu = v.cpu();
        const auto gpu = v.gpu();

        Compare(v, cpu, gpu, tolerance);

        return std::make_pair(cpu, gpu);
    }

    template <typename... CpuRanges, typename... GpuRanges>
    void Compare(auto&& v,
                 const std::tuple<CpuRanges...>& cpu,
                 const std::tuple<GpuRanges...>& gpu,
                 double tolerance)
    {
        static_assert(sizeof...(CpuRanges) == sizeof...(GpuRanges), "CPU and GPU mismatch");

        miopen::sequence([&](auto... is) {
            miopen::each_args(
                [&](auto i) {
                    const auto& c = std::get<i>(cpu);
                    const auto& g = std::get<i>(gpu);

                    ASSERT_EQ(miopen::range_distance(c), miopen::range_distance(g));

                    using value_type = miopen::range_value<decltype(g)>;

                    const double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
                    const double error     = miopen::rms_range(c, g);

                    EXPECT_LE(error, threshold);

                    if(error > threshold)
                    {
                        v.fail(i);
                    }
                },
                is...);
        })(std::integral_constant<std::size_t, sizeof...(CpuRanges)>{});
    }

    template <typename CpuRanges, typename GpuRanges>
    void Compare(auto&& v, const CpuRanges& cpu, const GpuRanges& gpu, double tolerance)
    {
        ASSERT_EQ(miopen::range_distance(cpu), miopen::range_distance(gpu));

        using value_type = miopen::range_value<decltype(gpu)>;

        const double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
        const double error     = miopen::rms_range(cpu, gpu);

        EXPECT_LE(error, threshold);

        if(error > threshold)
        {
            v.fail(0);
        }
    }
};

struct TestNameGenerator
{
    std::string operator()(const auto& testCase)
    {
        const auto& dims = testCase.param;
        std::stringstream ss;
        std::string str;

        ss << "dims_" << GetRangeAsString(dims(), "x") << "_test_id_" << testCase.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

// Exercise the packed retention boundary independently of the backward tests.
// Odd spatial and batch sizes also leave partial workgroups and an unpaired half.
template <typename T>
struct BnPeractRetentionTest : testing::TestWithParam<int>
{
    void Run()
    {
        const auto n = static_cast<std::size_t>(GetParam());
        constexpr std::size_t c = 3, h = 7, w = 9;
        constexpr std::size_t features = c * h * w;
        constexpr std::size_t guard    = 16;
        constexpr float canary        = 30000;
        constexpr double factor       = 0.25;
        constexpr double epsilon      = 1e-5;
        constexpr bool is_bfloat      = miopen_type<T>{} == miopenBFloat16;

        tensor<T> input{n, c, h, w};
        tensor<float> parameters{1, c, h, w};
        const auto guarded = [](std::size_t count, auto value) {
            return std::vector<decltype(value)>(count + 2 * guard, value);
        };
        auto x        = guarded(n * features, T{canary});
        auto y        = guarded(n * features, T{canary});
        auto scale    = guarded(features, canary);
        auto shift    = guarded(features, canary);
        auto run_mean = guarded(features, canary);
        auto run_var  = guarded(features, canary);
        auto mean     = guarded(features, canary);
        auto inv_var  = guarded(features, canary);

        for(std::size_t p = 0; p < features; ++p)
        {
            scale[guard + p] =
                (p % 2 == 0 ? 1.0f : -1.0f) * (0.75f + 0.125f * (p % 7));
            shift[guard + p]    = 0.25f * (static_cast<int>(p % 9) - 4);
            run_mean[guard + p] = 0.5f * (static_cast<int>(p % 5) - 2);
            run_var[guard + p]  = 1.0f + 0.25f * (p % 11);
            for(std::size_t b = 0; b < n; ++b)
            {
                // The last batch differs deliberately from the preceding pair.
                const auto sample = b + 1 == n ? 37 : (13 * b + 7 * p) % 29;
                const double offset = p % 2 == 0 ? -3.0 : (is_bfloat ? 1024.0 : 120.0);
                const double step   = p % 2 == 0 ? 0.5 : (is_bfloat ? 8.0 : 0.5);
                x[guard + b * features + p] = T(offset + step * sample);
            }
        }
        const auto initial_run_mean = run_mean;
        const auto initial_run_var  = run_var;
        auto&& handle = get_handle();
        auto x_dev        = handle.Write(x);
        auto y_dev        = handle.Write(y);
        auto scale_dev    = handle.Write(scale);
        auto shift_dev    = handle.Write(shift);
        auto run_mean_dev = handle.Write(run_mean);
        auto run_var_dev  = handle.Write(run_var);
        auto mean_dev     = handle.Write(mean);
        auto inv_var_dev  = handle.Write(inv_var);
        const auto data = [](auto& buffer, auto value) {
            return static_cast<decltype(value)*>(buffer.get()) + guard;
        };
        float alpha = 1.0f, beta = 0.0f;
        ASSERT_EQ(miopenBatchNormalizationForwardTraining(&handle,
                                                         miopenBNPerActivation,
                                                         &alpha,
                                                         &beta,
                                                         &input.desc,
                                                         data(x_dev, T{}),
                                                         &input.desc,
                                                         data(y_dev, T{}),
                                                         &parameters.desc,
                                                         data(scale_dev, float{}),
                                                         data(shift_dev, float{}),
                                                         factor,
                                                         data(run_mean_dev, float{}),
                                                         data(run_var_dev, float{}),
                                                         epsilon,
                                                         data(mean_dev, float{}),
                                                         data(inv_var_dev, float{})),
                  miopenStatusSuccess);
        y        = handle.Read<T>(y_dev, y.size());
        run_mean = handle.Read<float>(run_mean_dev, run_mean.size());
        run_var  = handle.Read<float>(run_var_dev, run_var.size());
        mean     = handle.Read<float>(mean_dev, mean.size());
        inv_var  = handle.Read<float>(inv_var_dev, inv_var.size());

        const auto check_guards = [&](const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            const double expected_canary = double(Value{canary});
            for(std::size_t i = 0; i < guard; ++i)
            {
                EXPECT_EQ(double(values[i]), expected_canary) << "leading guard " << i;
                EXPECT_EQ(double(values[values.size() - guard + i]), expected_canary)
                    << "trailing guard " << i;
            }
        };
        check_guards(y);
        check_guards(run_mean);
        check_guards(run_var);
        check_guards(mean);
        check_guards(inv_var);
        const auto check_input = [&](const auto& buffer, const auto& expected) {
            using Value = typename std::decay_t<decltype(expected)>::value_type;
            const auto actual = handle.Read<Value>(buffer, expected.size());
            for(std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_EQ(double(actual[i]), double(expected[i])) << "input element " << i;
        };
        check_input(x_dev, x);
        check_input(scale_dev, scale);
        check_input(shift_dev, shift);

        for(std::size_t p = 0; p < features; ++p)
        {
            SCOPED_TRACE(testing::Message() << "batch " << n << ", feature " << p);
            double expected_mean = 0;
            for(std::size_t b = 0; b < n; ++b)
                expected_mean += double(x[guard + b * features + p]);
            expected_mean /= n;
            double variance = 0;
            for(std::size_t b = 0; b < n; ++b)
            {
                const double centered = double(x[guard + b * features + p]) - expected_mean;
                variance += centered * centered;
            }
            variance /= n;
            const double expected_inv_var = 1.0 / std::sqrt(variance + epsilon);
            // Moments are FP32, including with low-precision activations. Allow FP32
            // accumulation error for the high-offset data, not activation rounding.
            const double mean_tolerance = 4 * std::numeric_limits<float>::epsilon() *
                                          std::max(1.0, std::abs(expected_mean));
            EXPECT_NEAR(mean[guard + p], expected_mean, mean_tolerance);
            EXPECT_NEAR(inv_var[guard + p], expected_inv_var, 5e-4 * expected_inv_var);
            const double expected_run_mean =
                (1 - factor) * initial_run_mean[guard + p] + factor * expected_mean;
            const double expected_run_var =
                (1 - factor) * initial_run_var[guard + p] + factor * variance * n / (n - 1);
            EXPECT_NEAR(run_mean[guard + p], expected_run_mean, mean_tolerance);
            EXPECT_NEAR(run_var[guard + p], expected_run_var, 5e-4 * expected_run_var);
            for(std::size_t b = 0; b < n; ++b)
            {
                const auto i = guard + b * features + p;
                const double expected =
                    double(T(scale[guard + p] * (double(x[i]) - expected_mean) * expected_inv_var +
                             shift[guard + p]));
                // Compare with the rounded destination value, with room for at most
                // two low-precision rounding units from FP32 normalization.
                const double tolerance = 2 * double(std::numeric_limits<T>::epsilon()) *
                                         std::max(1.0, std::abs(expected));
                EXPECT_NEAR(double(y[i]), expected, tolerance) << "batch element " << b;
            }
        }
    }
};

using GPU_BnPeractRetention_FP16  = BnPeractRetentionTest<half_float::half>;
using GPU_BnPeractRetention_BFP16 = BnPeractRetentionTest<bfloat16>;

TEST_P(GPU_BnPeractRetention_FP16, ForwardTraining) { this->Run(); }
TEST_P(GPU_BnPeractRetention_BFP16, ForwardTraining) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BnPeractRetention_FP16, testing::Values(3, 63, 64, 65));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BnPeractRetention_BFP16, testing::Values(3, 63, 64, 65));

using GPU_BnPeract_FP16  = BnPeractTest<half_float::half>;
using GPU_BnPeract_FP32  = BnPeractTest<float>;
using GPU_BnPeract_BFP16 = BnPeractTest<bfloat16>;

TEST_P(GPU_BnPeract_FP16, TestFloat16) { this->Run(); }
TEST_P(GPU_BnPeract_FP32, TestFloat32) { this->Run(); }
TEST_P(GPU_BnPeract_BFP16, TestBFloat16) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BnPeract_FP16, GetSmokeTestCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BnPeract_FP32, GetSmokeTestCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BnPeract_BFP16, GetSmokeTestCases(), TestNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Full, GPU_BnPeract_FP16, GetFullTestCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_BnPeract_FP32, GetFullTestCases(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_BnPeract_BFP16, GetFullTestCases(), TestNameGenerator{});
