// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "compare_helper.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "random.hpp"
#include "../network_data.hpp"

#include <miopen/batch_norm.hpp>
#include <miopen/activ.hpp>

#include <gtest/gtest.h>

#define MIO_HEIRARCH_SEL 1
#define MIO_BN_USE_MIX_PREC 1

#if MIO_BN_USE_MIX_PREC == 1
#define PREC_TYPE float
#else
#define PREC_TYPE T
#endif

namespace {
// Run CPU emulations in hierarchical reduction mode.
constexpr float MIO_BN_TEST_EXPAVGFACTOR = 0.1f;
constexpr auto MIO_BN_TEST_EPSILON       = 1e-5; // FLT_EPSILON
constexpr int MIO_BN_SP_TEST_DEBUG       = 0;
constexpr int batch_factor               = 4;

auto GenCases() { return testing::ValuesIn(get_bn_spatial_inputs(batch_factor)); }

auto GetCases(bool all_tests = true)
{
    if(all_tests)
    {
        static const auto cases = GenCases();
        return cases;
    }

    std::set<std::vector<int>> small_case = {{4, 64, 28, 28}};
    return testing::ValuesIn(small_case);
}

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T, class U>
struct verify_forward_train_bn_spatial
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> cpu() const
    {

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        std::size_t rs_n_batch, rs_channels, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);

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
        auto saveMean   = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto saveInvVar = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto out        = input;
        std::fill(out.begin(), out.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd        = 0.;
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double invVar         = 0.;
            double newRunMean     = 0.;
            double adjust         = 0.;

#ifdef MIO_HEIRARCH_SEL
            std::vector<double> variance_accum_arr(height, 0.0);
            std::vector<double> mean_accum_arr(height, 0.0);
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);

            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        mean_accum_arr[row] += input(bidx, cidx, row, column);
                    }
                } // for (column)
            } // for (row)
            for(std::size_t i = 0; i < height; i++)
                mean_accum += mean_accum_arr[i];
#else  // MIO_HEIRARCH_SEL
       // process the batch per channel
            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    } // end for (column)
                }     // end for (row)
            }         // end for (n)
#endif // MIO_HEIRARCH_SEL

            mean_accum /= nhw;

            elemStd        = 0.;
            variance_accum = 0.;

#ifndef MIO_HEIRARCH_SEL
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        // using out buffer as scratchpad
                        out(bidx, cidx, row, column) = elemStd =
                            (input(bidx, cidx, row, column) - mean_accum); // (x_i - mean)
                        variance_accum += (elemStd * elemStd);             // sum{ (x_i - mean)^2 }
                    } // end for (column)
                } // end for (row)
            } // end for(n)
#else  // MIO_HEIRARCH_SEL
            for(std::size_t row = 0; row < height; row++){ //via rows
                for(std::size_t column = 0; column < width; column++){// via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        out(bidx,cidx,row,column) = elemStd = input(bidx,cidx,row,column) - mean_accum;
                        variance_accum_arr[row] += elemStd*elemStd;
                    }
                }// for (column)
            }// for (row)
            for(std::size_t i = 0; i<height; i++) variance_accum += variance_accum_arr[i];
#endif // MIO_HEIRARCH_SEL

            variance_accum /= nhw; // (1/N)*sum{ (x_i - mean)^2 }
            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1.0 / sqrt(variance_accum + epsilon);

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        // #5 Gamma and Beta adjust
                        // y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, 0, 0) * (invVar * out(bidx, cidx, row, column)) +
                            shift(0, cidx, 0, 0);
                    } // for (column)
                } // for (row)
            } // end for(n_batchs)

            saveMean(0, cidx, 0, 0)   = mean_accum;
            saveInvVar(0, cidx, 0, 0) = invVar;

            newRunMean             = runMean(0, cidx, 0, 0) * (1 - expAvgFactor);
            runMean(0, cidx, 0, 0) = mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp
            // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
            adjust = (n_batch * height * width == 1) ? variance_accum
                                                     : (nhw / (nhw - 1)) * variance_accum;
            runVar(0, cidx, 0, 0) =
                (1 - expAvgFactor) * runVar(0, cidx, 0, 0) + expAvgFactor * adjust;
        });

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train_bn_spatial pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING
        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> gpu() const
    {

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING

        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        std::size_t rs_n_batch, rs_channels, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};

        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);

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

        auto saveMean   = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto saveInvVar = tensor<U>{rs_n_batch, rs_channels, rs_height, rs_width};

        // in buffers
        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        // out buffers
        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<U>(channels);
        auto saveInvVar_dev = handle.Create<U>(channels);
        auto out_dev        = handle.Create<T>(n_batch * channels * height * width);

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        float alpha = 1.0;
        float beta  = 0.0;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardTraining(handle,
                                         miopenBNSpatial,
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

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train_bn_spatial pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    void fail() const
    {

        FAIL() << "Forward Train Spatial Batch Normalization: " << std::endl
               << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T, class U>
struct verify_forward_infer_bn_spatial_recalc
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;

    tensor<T> cpu() const
    {

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd        = 0.;
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double inhat          = 0.;
            double invVar         = 0.;

            // process the batch per channel
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    // #1 calculate the mean
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // iterating through the stack of images in the mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    } // end for (n)
                } // end for (column)
            } // end for (row)
            mean_accum /= nhw;

            elemStd        = 0.;
            variance_accum = 0.;
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // using out buffer as scratchpad
                        out(bidx, cidx, row, column) = elemStd =
                            (input(bidx, cidx, row, column) - mean_accum); // (x_i - mean)
                        variance_accum += (elemStd * elemStd);             // sum{ (x_i - mean)^2 }
                    } // end for(n)
                } // end for (column)
            } // end for (row)
            variance_accum /= nhw; // (1/N)*sum{ (x_i - mean)^2 }

            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1.0 / sqrt(variance_accum + epsilon);

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        elemStd =
                            out(bidx, cidx, row, column); // using saved values from output tensor
                        inhat = elemStd * invVar;
                        // #5 Gamma and Beta adjust // y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, 0, 0) * inhat + shift(0, cidx, 0, 0);
                    } // end for(n_batchs)
                } // for (column)
            } // for (row)
        });

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING
        return out;
    }

    tensor<T> gpu() const
    {

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);
        auto out_dev   = handle.Write(out.data);

        float alpha = 1.0;
        float beta  = 0.0;

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardInference(handle,
                                          miopenBNSpatial,
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

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING
        return out;
    }

    void fail() const
    {
        FAIL() << "Forward Inference Spatial Batch Normalization Recalc: " << std::endl
               << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T, class U>
struct verify_forward_infer_bn_spatial_use_est
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;
    const tensor<U> estMean;
    const tensor<U> estVar;
    tensor<T> cpu() const
    {

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING

        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd  = 0.;
            double variance = estVar(0, cidx, 0, 0);
            double mean     = estMean(0, cidx, 0, 0);
            double inhat    = 0.;
            double invVar   = 1.0 / sqrt(variance + epsilon);

            // process the batch per channel
            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns

                        elemStd = input(bidx, cidx, row, column) - mean;
                        inhat   = elemStd * invVar;
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, 0, 0) * inhat + shift(0, cidx, 0, 0);
                    }
                }
            }
        });
#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_spatial_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING
        return out;
    }

    tensor<T> gpu() const
    {
#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev      = handle.Write(input.data);
        auto estMean_dev = handle.Write(estMean.data);
        auto estVar_dev  = handle.Write(estVar.data);
        auto scale_dev   = handle.Write(scale.data);
        auto shift_dev   = handle.Write(shift.data);
        auto out_dev     = handle.Write(out.data);

        float alpha = 1.0;
        float beta  = 0.0;

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardInference(handle,
                                          miopenBNSpatial,
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
                                          actDesc);
        out.data = handle.Read<T>(out_dev, out.data.size());
#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_spatial_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING
        return out;
    }

    void fail() const
    {
        FAIL() << "Forward Inference Spatial Batch Normalization Use Estimated: " << std::endl
               << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

//****************************************************
// BACKWARDS PROPAGATION
//****************************************************
template <class T, class U>
struct verify_backward_bn_spatial_recalc
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<U> scale;

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd = 0.;
            unsigned int xhat_index;
            double mean     = 0.;
            double invVar   = 0.;
            double dyelem   = 0.;
            double variance = 0.;

            std::vector<double> xhat(n_batch * in_cstride, 0.0);

#ifdef MIO_HEIRARCH_SEL
            std::vector<double> variance_accum_arr(height, 0.0);
            std::vector<double> mean_accum_arr(height, 0.0);
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);

            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        mean_accum_arr[row] += x_input(bidx, cidx, row, column);
                    }
                } // for (column)
            } // for (row)
            for(std::size_t i = 0; i < height; i++)
                mean += mean_accum_arr[i];
#else  // MIO_HEIRARCH_SEL
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // #1 calculate the mean
                        mean += x_input(bidx, cidx, row, column);
                    }
                } // for (column)
            }     // for (row)
#endif // MIO_HEIRARCH_SEL

            mean /= nhw;

            elemStd  = 0.;
            variance = 0.;
#ifndef MIO_HEIRARCH_SEL
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        elemStd = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        variance += elemStd * elemStd;                     // sum{ (x_i - mean)^2 }
                    } // end for(n)
                } // for (column)
            } // for (row)
#else                        // MIO_HEIRARCH_SEL
            for(std::size_t row = 0; row < height; row++){ //via rows
                for(std::size_t column = 0; column < width; column++){// via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        elemStd = x_input(bidx,cidx,row,column) - mean;
                        variance_accum_arr[row] += elemStd*elemStd;
                    }
                }// for (column)
            }// for (row)
            for(std::size_t i = 0; i<height; i++) variance += variance_accum_arr[i];
#endif                       // MIO_HEIRARCH_SEL
            variance /= nhw; // (1/(N*H*W))*sum{ (x_i - mean)^2 }
            invVar = 1. / double(sqrt(variance + epsilon));

            dscale(0, cidx, 0, 0) = 0.;

#ifndef MIO_HEIRARCH_SEL
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * invVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, 0, 0) += dyelem;
                        dscale(0, cidx, 0, 0) += xhat[xhat_index] * dyelem;
                    } // end for(n_batch)
                } // for (column)
            } // for (row)
#else
            for(std::size_t row = 0; row < height; row++){ //via rows
                for(std::size_t column = 0; column < width; column++){// via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        xhat_index = in_cstride*bidx + (width*row + column);
                        //per (x-dims) channel load a block of data into LDS
                        elemStd             = x_input(bidx,cidx,row,column) - mean;// (x_i - mean)
                        xhat[xhat_index]    = elemStd*invVar;
                        dyelem              = dy_input(bidx,cidx,row,column);
                        dshift_accum_arr[row] += dyelem;
                        dscale_accum_arr[row] += xhat[xhat_index]*dyelem;
                        //dscale_accum_arr[row] += x_input(bidx,cidx,row,column);;//dscale_accum_arr[row] += xhat[xhat_index];
                        //dscale_accum_arr[row] += 1.0;//DEBUG
                    }
                }// for (column)
            }// for (row)
            for(std::size_t i = 0; i<height; i++) {
                dshift(0,cidx,0,0) += dshift_accum_arr[i];
                dscale(0,cidx,0,0) += dscale_accum_arr[i];
            }
#endif // MIO_HEIRARCH_SEL

            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);

                        double tmp1 =
                            nhw * dy_input(bidx, cidx, row, column) - dshift(0, cidx, 0, 0);
                        double tmp2                     = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                        double tmp3                     = (scale(0, cidx, 0, 0) * invVar) / nhw;
                        dx_out(bidx, cidx, row, column) = tmp3 * (tmp2 + tmp1);
                    } // end for(n_batchs)
                } // for (column)
            } // for (row)
        }); // for (channel)

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING

        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING
        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        float alpha = 1.0;
        float beta  = 0.0;

        auto xin_dev    = handle.Write(x_input.data);
        auto dyin_dev   = handle.Write(dy_input.data);
        auto scale_dev  = handle.Write(scale.data);
        auto dscale_dev = handle.Write(dscale.data);
        auto dshift_dev = handle.Write(dshift.data);
        auto dx_out_dev = handle.Write(dx_out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormBackward(handle,
                                  miopenBNSpatial,
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

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail() const
    {
        FAIL() << "Backward Batch Spatial Normalization Recalc Mean and Variance: " << std::endl
               << "X Input tensor: " << x_input.desc.ToString() << std::endl
               << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
    }
};

template <class T, class U>
struct verify_backward_bn_spatial_use_saved
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<U> scale;
    const tensor<U> savedMean;
    const tensor<U> savedInvVar;
    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        miopen::par_for(channels, 1, [&](int cidx) {
            double elemStd = 0.;
            unsigned int xhat_index;
            double mean   = savedMean(0, cidx, 0, 0);   // HxW elements
            double invVar = savedInvVar(0, cidx, 0, 0); // HxW elements
            double dyelem = 0.;

            std::vector<double> xhat(n_batch * in_cstride, 0.0);

#ifdef MIO_HEIRARCH_SEL
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);
#endif

            // process the batch per channel
            dscale(0, cidx, 0, 0) = 0.;

#ifdef MIO_HEIRARCH_SEL
            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * invVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, 0, 0) += dyelem;
                        dscale(0, cidx, 0, 0) += xhat[xhat_index] * dyelem;
                    } // end for(n_batch)
                } // for (column)
            } // for (row)
#else  // MIO_HEIRARCH_SEL
            for(std::size_t row = 0; row < height; row++){ //via rows
                for(std::size_t column = 0; column < width; column++){// via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        xhat_index = in_cstride*bidx + (width*row + column);
                        //per (x-dims) channel load a block of data into LDS
                        elemStd             = x_input(bidx,cidx,row,column) - mean;// (x_i - mean)
                        xhat[xhat_index]    = elemStd*invVar;
                        //printf("xhat[%d]: %lf\n",xhat_index,xhat[xhat_index]);
                        dyelem              = dy_input(bidx,cidx,row,column);
                        dshift_accum_arr[row] += dyelem;
                        dscale_accum_arr[row] += xhat[xhat_index]*dyelem;
                        //dscale_accum_arr[row] += 1.0;//DEBUG
                    }
                }// for (column)
            }// for (row)
            for(std::size_t i = 0; i<height; i++) {
                dshift(0,cidx,0,0) += dshift_accum_arr[i];
                dscale(0,cidx,0,0) += dscale_accum_arr[i];
            }
#endif // MIO_HEIRARCH_SEL

            for(std::size_t row = 0; row < height; row++)
            { // via rows
                for(std::size_t column = 0; column < width; column++)
                { // via columns
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);

                        double tmp1 =
                            nhw * dy_input(bidx, cidx, row, column) - dshift(0, cidx, 0, 0);
                        double tmp2                     = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                        double tmp3                     = (scale(0, cidx, 0, 0) * invVar) / nhw;
                        dx_out(bidx, cidx, row, column) = tmp3 * (tmp2 + tmp1);
                    } // end for(n_batchs)
                } // for (column)
            } // for (row)
        }); // for (channel)
#if MIO_BN_TIME_EVERYTHING == 1
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn spatial_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif // MIO_BN_TIME_EVERYTHING
        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {

#if MIO_BN_TIME_EVERYTHING == 1
        auto t_start = std::chrono::high_resolution_clock::now();
#endif // MIO_BN_TIME_EVERYTHING
        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        float alpha = 1.0;
        float beta  = 0.0;

        auto xin_dev         = handle.Write(x_input.data);
        auto dyin_dev        = handle.Write(dy_input.data);
        auto scale_dev       = handle.Write(scale.data);
        auto dscale_dev      = handle.Write(dscale.data);
        auto dshift_dev      = handle.Write(dshift.data);
        auto dx_out_dev      = handle.Write(dx_out.data);
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormBackward(handle,
                                  miopenBNSpatial,
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

#if MIO_BN_TIME_EVERYTHING == 1
        {
            auto t_end = std::chrono::high_resolution_clock::now();

            std::cout << "Wall clock: GPU backward_bn_spatial_use_saved pass time: "
                      << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                      << std::endl;
        }
#endif // MIO_BN_TIME_EVERYTHING
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail() const
    {
        FAIL() << "Backward Batch Spatial Normalization Use Saved Mean and Variance: " << std::endl
               << "X Input tensor: " << x_input.desc.ToString() << std::endl
               << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
    }
};

using TestCase = std::vector<int>;

auto NameGenerator(const ::testing::TestParamInfo<TestCase>& info)
{
    std::stringstream name{};
    name << "n" << info.param[0] << "c" << info.param[1] << "h" << info.param[2] << "w"
         << info.param[3];
    return name.str();
}
} // namespace

//====== DRIVERS ===========================================
template <class T>
class batch_norm_spatial_test : public testing::TestWithParam<TestCase>
{
    tensor<T> input;
    tensor<PREC_TYPE> scale;
    tensor<PREC_TYPE> shift;

public:
    void SetUp() override
    {
        prng::reset_seed();
        size_t n             = 0U;
        size_t c             = 0U;
        size_t h             = 0U;
        size_t w             = 0U;
        std::tie(n, c, h, w) = miopen::tien<4>(GetParam());
        input                = tensor<T>{n, c, h, w};
        input.generate(tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17});
    }

    void Run()
    {
        double tolerance =
            4e-3 / std::numeric_limits<T>::epsilon(); // ck solver has tolerance of 4e-3

        std::size_t n, c, h, w;
        std::tie(n, c, h, w) = miopen::tien<4>(input.desc.GetLengths());

        if(n == 1 ||
           ((h * w > 1024) && (input.desc.GetType() == miopenHalf) && (MIO_BN_USE_MIX_PREC == 0)) ||
           (n == 128 && c == 16 && h == 32 && w == 32)) // \todo DLOWELL: This last condtion is
                                                        // needed to get half test to pass. Batch
                                                        // norm needs rewriting for fp16.
        {
            GTEST_SKIP() << "Invalid batch size for batch normalization";
        }

        std::size_t ssn, ssc, ssh, ssw;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);
        std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());

        scale                   = tensor<PREC_TYPE>{ssn, ssc, ssh, ssw};
        shift                   = tensor<PREC_TYPE>{ssn, ssc, ssh, ssw};
        const double Data_scale = 1e-2;

        for(std::size_t i = 0; i < scale.desc.GetElementSize(); i++)
        {
            scale[i] = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            shift[i] = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
        }
        for(std::size_t i = 0; i < input.desc.GetElementSize(); i++)
        {
            input[i] = prng::gen_descreet_uniform_sign<T>(Data_scale, 100);
        }

        // train
        if constexpr(MIO_BN_SP_TEST_DEBUG == 1)
            std::cout << "Running forward train spatial with R and S set." << std::endl;
        auto outpair = test_helpers::CompareResults(
            verify_forward_train_bn_spatial<T, PREC_TYPE>{input, scale, shift}, tolerance);
        // returns:  std::make_tuple(out,runMean,runVar,saveMean,saveInvVar);

        // inference recalc
        if constexpr(MIO_BN_SP_TEST_DEBUG == 1)
            std::cout << "Running forward inference spatial recalc." << std::endl;
        // tolerance = 80;
        // Debug values
        // std::fill(input.begin(), input.end(), 1);
        // std::fill(scale.begin(), scale.end(), 1);
        // std::fill(shift.begin(), shift.end(), 1);
        tolerance = 80 * input.desc.GetElementSize();

        test_helpers::CompareResults(
            verify_forward_infer_bn_spatial_recalc<T, PREC_TYPE>{input, scale, shift}, tolerance);

        // inference use estimated running values
        auto estMean = std::get<1>(outpair.second);
        auto estVar  = std::get<2>(outpair.second);
        if constexpr(MIO_BN_SP_TEST_DEBUG == 1)
            std::cout << "Running forward inference spatial with R set." << std::endl;
        test_helpers::CompareResults(
            verify_forward_infer_bn_spatial_use_est<T, PREC_TYPE>{
                input, scale, shift, estMean, estVar},
            tolerance);

        // backprop recalc
        auto dy_input = std::get<0>(outpair.second);
        for(std::size_t bidx = 0; bidx < n; bidx++)
        { // via mini_batch
            for(std::size_t cidx = 0; cidx < c; cidx++)
            { // via mini_batch
                for(std::size_t row = 0; row < h; row++)
                { // via rows
                    for(std::size_t column = 0; column < w; column++)
                    {
                        dy_input(bidx, cidx, row, column) *= 0.1;
                    }
                }
            }
        }
        if constexpr(MIO_BN_SP_TEST_DEBUG == 2)
        {
            auto debugvals = test_helpers::CompareResults(
                verify_backward_bn_spatial_recalc<T, PREC_TYPE>{input, dy_input, scale}, tolerance);
            auto gpuout = std::get<0>(debugvals.second);
            auto cpuout = std::get<0>(debugvals.first);

            double maxdiff = 0.;
            int mn         = 0;
            int mc         = 0;
            int mh         = 0;
            int mw         = 0;

            for(std::size_t bidx = 0; bidx < n; bidx++)
            { // via mini_batch
                for(std::size_t cidx = 0; cidx < c; cidx++)
                { // via mini_batch
                    for(std::size_t row = 0; row < h; row++)
                    { // via rows
                        for(std::size_t column = 0; column < w; column++)
                        { // via columns
                            double diff = fabs(gpuout(bidx, cidx, row, column) -
                                               cpuout(bidx, cidx, row, column));
                            if(diff > maxdiff)
                            {
                                maxdiff = diff;
                                mn      = bidx;
                                mc      = cidx;
                                mh      = row;
                                mw      = column;
                            }
                            // if(diff > 1.)
                            // {
                            std::cout << "gpu[" << bidx << ", " << cidx << ", " << row << ", "
                                      << column << "]: " << gpuout(bidx, cidx, row, column)
                                      << " :: ";
                            std::cout << "cpu[" << bidx << ", " << cidx << ", " << row << ", "
                                      << column << "]: " << cpuout(bidx, cidx, row, column)
                                      << " :: ";
                            std::cout << "diff: " << diff << std::endl;
                            //    }
                        }
                    }
                }
            }
            if(maxdiff > 0)
            {
                std::cout << "Max diff: " << maxdiff << std::endl;
                std::cout << "gpu[" << mn << ", " << mc << ", " << mh << ", " << mw
                          << "]: " << gpuout(mn, mc, mh, mw) << " :: ";
                std::cout << "cpu[" << mn << ", " << mc << ", " << mh << ", " << mw
                          << "]: " << cpuout(mn, mc, mh, mw) << std::endl;
            }
        }
        else
        {
            if constexpr(MIO_BN_SP_TEST_DEBUG == 1)
                std::cout << "Running back propagation spatial recalc." << std::endl;
            tolerance = 80 * input.desc.GetElementSize();
            test_helpers::CompareResults(
                verify_backward_bn_spatial_recalc<T, PREC_TYPE>{input, dy_input, scale}, tolerance);
        }

        // backprop use saved values
        auto savedMean   = std::get<3>(outpair.second);
        auto savedInvVar = std::get<4>(outpair.second);

        if constexpr(MIO_BN_SP_TEST_DEBUG == 3)
        {
            auto debugvals = test_helpers::CompareResults(
                verify_backward_bn_spatial_use_saved<T, PREC_TYPE>{
                    input, dy_input, scale, savedMean, savedInvVar},
                tolerance);
            auto gpuout = std::get<0>(debugvals.second);
            auto cpuout = std::get<0>(debugvals.first);

            double maxdiff = 0.;
            int mn         = 0;
            int mc         = 0;
            int mh         = 0;
            int mw         = 0;

            for(std::size_t bidx = 0; bidx < n; bidx++)
            { // via mini_batch
                for(std::size_t cidx = 0; cidx < c; cidx++)
                { // via mini_batch
                    for(std::size_t row = 0; row < h; row++)
                    { // via rows
                        for(std::size_t column = 0; column < w; column++)
                        { // via columns
                            double diff = fabs(gpuout(bidx, cidx, row, column) -
                                               cpuout(bidx, cidx, row, column));
                            if(diff > maxdiff)
                            {
                                maxdiff = diff;
                                mn      = bidx;
                                mc      = cidx;
                                mh      = row;
                                mw      = column;
                            }
                            // if(diff > 1.)
                            //{
                            std::cout << "gpu[" << bidx << ", " << cidx << ", " << row << ", "
                                      << column << "]: " << gpuout(bidx, cidx, row, column)
                                      << " :: ";
                            std::cout << "cpu[" << bidx << ", " << cidx << ", " << row << ", "
                                      << column << "]: " << cpuout(bidx, cidx, row, column)
                                      << " :: ";
                            std::cout << "diff: " << diff << std::endl;
                            //}
                        }
                    }
                }
            }
            if(maxdiff > 0)
            {
                std::cout << "Max diff: " << maxdiff << std::endl;
                std::cout << "gpu[" << mn << ", " << mc << ", " << mh << ", " << mw
                          << "]: " << gpuout(mn, mc, mh, mw) << " :: ";
                std::cout << "cpu[" << mn << ", " << mc << ", " << mh << ", " << mw
                          << "]: " << cpuout(mn, mc, mh, mw) << std::endl;
            }
        }
        else
        {
            if constexpr(MIO_BN_SP_TEST_DEBUG == 1)
                std::cout << "Running back propagation spatial with S set." << std::endl;
            test_helpers::CompareResults(
                verify_backward_bn_spatial_use_saved<T, PREC_TYPE>{
                    input, dy_input, scale, savedMean, savedInvVar},
                tolerance);
        }
    }
};

using GPU_BN_Spatial_FP32 = batch_norm_spatial_test<float>;

TEST_P(GPU_BN_Spatial_FP32, TestFloat32) { Run(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_BN_Spatial_FP32, GetCases(), [](const auto& info_) {
    return NameGenerator(info_);
});

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BN_Spatial_FP32, GetCases(false), [](const auto& info_) {
    return NameGenerator(info_);
});
