// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "compare_helper.hpp"
#include "get_handle.hpp"
#include "miopen/readonlyramdb.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "random.hpp"
#include "../network_data.hpp"

#include <cmath>
#include <initializer_list>
#include <numbers>
#include <miopen/batch_norm.hpp>
#include <miopen/activ.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batchnorm/solvers.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/stringutils.hpp>

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

auto GetCasesFwdSpatialPerVariant(int variant)
{
    std::set<std::vector<int>> ret = {};
    switch(variant)
    {
    case 0:
        ret = {{8, 64, 14, 14},   {8, 128, 4, 4},    {8, 128, 14, 14},  {8, 160, 7, 7},
               {8, 192, 7, 7},    {8, 192, 14, 14},  {8, 224, 14, 14},  {8, 256, 7, 7},
               {8, 352, 7, 7},    {8, 1024, 1, 1},   {25, 32, 8, 8},    {32, 256, 12, 12},
               {38, 256, 20, 20}, {38, 512, 20, 20}, {64, 256, 14, 14}, {64, 256, 20, 20},
               {64, 512, 7, 7},   {64, 512, 7, 7},   {64, 512, 20, 20}, {64, 2048, 7, 7},
               {192, 1, 8, 8},    {192, 1024, 1, 1}};
        break;
    case 1:
        ret = {
            // Explicit Variant1 cases below leave whole waves empty in the Welford reduction.
            {1, 4, 1, 16},
            {1, 4, 1, 65},
            {8, 3, 224, 224},
            {8, 4, 1024, 2048},
            {8, 64, 56, 56},
            {8, 192, 56, 56},
            {8, 192, 256, 512},
            {8, 480, 128, 256},
            {8, 528, 64, 128},
            {38, 32, 160, 160},
            {38, 64, 80, 80},
            {38, 64, 160, 160},
            {38, 128, 80, 80},
            {64, 3, 227, 227},
            {64, 64, 56, 56},
            {64, 64, 112, 112},
            {64, 256, 56, 56},
        };
        break;
    case 2:
        ret = {
            {8, 64, 28, 28},
            {8, 96, 28, 28},
            {8, 256, 28, 28},
            {64, 128, 28, 28},
            {64, 512, 28, 28},
            {128, 16, 32, 32},
        };
        break;
    case 3:
        ret = {
            {16, 1, 32, 32},
        };
        break;
    }
    return testing::ValuesIn(ret);
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

/*
 * A test that replicates the issue observed in MIOpen#3900
 * 2026-07-20: This test has the tolerance for the dscale value
 * set to a very high value. This is due to
 * instruction order issued for different targets.
 *
 * Channging the fill value from 0.1 to something that is
 * exactly represented in FP32 like 0.125 does not work --
 * then the test does not expose the presence or lack of
 * Welford's algorithm in the targeted kernel.
 * This test is currently disabled until the proper
 * cause of the error in the failing GPU targets is found.
 *
 * There is another test suite that fails on
 * the absence of Welford's algorithm in Batchnorm Fwd Training.
 * so this does not decrease the test coverage significantly.
 */

TEST(GPU_BN_Spatial_FP32, MIOpen3900Regression)
{
    /*
     * Overview of the test:
     * - kernel params 1, 2048, 512, 512, fp32, currently triggers variant 1
     * - all values initialized with 0.1
     * - calculate the output norm
     * - for the bwd pass, initialize the gradient as 1.
     * - run the bwd pass
     * - check that for each
     * - the forward kernel corresponds to:
     *   - MIOpenDriver bnorm -n 1 -c 2048 -H 512 -W 512 --forw 1 -m 1 -s 1 -r 0 -i 1
     */

    // Set kernel shape
    size_t n                = 1U;
    size_t c                = 2048U;
    size_t h                = 512U;
    size_t w                = 512U;
    tensor<float> input     = tensor<float>{n, c, h, w};
    tensor<float> init_grad = tensor<float>{n, c, h, w};
    tensor<float> scale;
    tensor<float> shift;
    /*
     * Note: the Python script that originally showed the issue (MIOpen#3900)
     * uses the shape [8, 256, 512, 512], but PyTorch launches the kernel
     * with one batch (so using the shape [1, 2048, 512, 512]), which is
     * what this test does to replicate the effect of the script in the test suite
     * Later, the dscale/dbias arrays which are shaped [1, 2048, 1, 1] get
     * reshaped to [1, 256, 1, 1] likely somewhere in PyTorch, but even
     * without doing that transformation, the test will fail in the same manner as
     * the Python script.
     */

    // Set up forward pass

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

    scale = tensor<PREC_TYPE>{rs_n_batch, rs_channels, rs_height, rs_width};
    shift = tensor<PREC_TYPE>{rs_n_batch, rs_channels, rs_height, rs_width};

    tensor<float> runMean = tensor<float>{rs_n_batch, rs_channels, rs_height, rs_width};
    tensor<float> runVar  = tensor<float>{rs_n_batch, rs_channels, rs_height, rs_width};

    auto saveMean   = tensor<float>{rs_n_batch, rs_channels, rs_height, rs_width};
    auto saveInvVar = tensor<float>{rs_n_batch, rs_channels, rs_height, rs_width};

    // All values initialized with 0.1
    for(std::size_t i = 0; i < input.GetSize(); i++)
    {
        input[i]     = 0.1f;
        init_grad[i] = 1.f; // the initial values of the dy_input tensor of the backwards pass
    }

    for(std::size_t i = 0; i < runMean.GetSize(); i++)
    {
        // Corresponds to the momentum parameter in the original
        // issue demonstrator -- according to the PyTorch docs,
        // the momentum is the value with which the running mean
        // and running variance are initialized
        runMean[i] = 0.1f;
        runVar[i]  = 0.1f;
        scale[i]   = 0.1f;
        shift[i]   = 0.1f;
    }

    // in buffers
    auto in_dev    = handle.Write(input.data);
    auto scale_dev = handle.Write(scale.data);
    auto shift_dev = handle.Write(shift.data);

    // out buffers
    auto runMean_dev    = handle.Write(runMean.data);
    auto runVar_dev     = handle.Write(runVar.data);
    auto saveMean_dev   = handle.Create<float>(channels);
    auto saveInvVar_dev = handle.Create<float>(channels);
    auto out_dev        = handle.Create<float>(n_batch * channels * height * width);

    double epsilon =
        MIO_BN_TEST_EPSILON; // Same epsilon as in the original issue demonstrator -- 1e-5
    double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

    float alpha = 1.0f;
    float beta  = 0.0f;

    miopenStatus_t res = miopenStatusUnknownError;

    res = miopenBatchNormalizationForwardTraining(&handle,
                                                  miopenBNSpatial,
                                                  &alpha,
                                                  &beta,
                                                  &input.desc,
                                                  in_dev.get(),
                                                  &out.desc,
                                                  out_dev.get(),
                                                  &scale.desc,
                                                  scale_dev.get(),
                                                  shift_dev.get(),
                                                  expAvgFactor,
                                                  nullptr,
                                                  nullptr, // Not calculating running mean/variance
                                                  epsilon,
                                                  saveMean_dev.get(),
                                                  saveInvVar_dev.get());

    if(res != miopenStatusSuccess)
    {
        GTEST_FAIL() << "Forward pass failed with status code " << res
                     << ". Check the enum miopenStatus_t";
    }

    saveMean.data   = handle.Read<float>(saveMean_dev, saveMean.data.size());
    saveInvVar.data = handle.Read<float>(saveInvVar_dev, saveInvVar.data.size());
    out.data        = handle.Read<float>(out_dev, out.data.size());

    // Backwards pass

    std::tie(n_batch, channels, height, width) = miopen::tien<4>(out.desc.GetLengths());

    auto dx_out = tensor<float>{n_batch, channels, height, width};
    std::fill(dx_out.begin(), dx_out.end(), 0);

    std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
    auto derivedBnDescBwd = miopen::TensorDescriptor{};
    miopen::DeriveBNTensorDescriptor(derivedBnDescBwd, out.desc, miopenBNSpatial);
    std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
        miopen::tien<4>(derivedBnDesc.GetLengths());

    auto dscale = tensor<float>{ss_n_batch, ss_channels, ss_height, ss_width};
    std::fill(dscale.begin(), dscale.end(), 0);

    auto dshift = tensor<float>{ss_n_batch, ss_channels, ss_height, ss_width};
    std::fill(dshift.begin(), dshift.end(), 0);

    auto xin_dev         = handle.Write(out.data);
    auto dyin_dev        = handle.Write(init_grad.data);
    scale_dev            = handle.Write(scale.data);
    auto dscale_dev      = handle.Write(dscale.data);
    auto dshift_dev      = handle.Write(dshift.data);
    auto dx_out_dev      = handle.Write(dx_out.data);
    auto savedMean_dev   = handle.Write(saveMean.data);
    auto savedInvVar_dev = handle.Write(saveInvVar.data);

    miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
    miopen::BatchNormBackward(
        handle,
        miopenBNSpatial,
        &alpha,
        &beta,
        &alpha,
        &beta,
        input.desc,
        in_dev.get(), // PyTorch takes the same mem address as the input, not the normalized output
        init_grad.desc,
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

    dx_out.data = handle.Read<float>(dx_out_dev, dx_out.data.size());
    dscale.data = handle.Read<float>(dscale_dev, dscale.data.size());
    dshift.data = handle.Read<float>(dshift_dev, dshift.data.size());

    /*
     * The tolerance is set to 5.0, because this value is caused by the accumulation
     * of a floating-point rounding error on the scale of 0x1p-24 within the mean,
     * which accumulates over the large tensor used in this test.
     * Nonetheless, this test will fail when Welford's algorithm is not used.
     * On some architectures, the value of dscale will be exactly 0. because
     * the mean is bitwise identical to the fill value.
     */
    double tolerance = 5.0f;

    for(std::size_t bidx = 0; bidx < ss_n_batch; bidx++)
    { // via mini_batch
        for(std::size_t cidx = 0; cidx < ss_channels; cidx++)
        { // via mini_batch
            for(std::size_t row = 0; row < ss_height; row++)
            { // via rows
                for(std::size_t column = 0; column < ss_width; column++)
                { // via columns
                    if(abs(dscale(bidx, cidx, row, column)) > tolerance)
                    {
                        GTEST_FAIL()
                            << "dscale should be zero, but found an element with an absolute value "
                            << dscale(bidx, cidx, row, column) << " at location[" << bidx << ", "
                            << cidx << ", " << row << ", " << column
                            << "] with the tolerance set to " << tolerance
                            << ". This could be an indicator of a bug in the variance calculation";
                    }
                }
            }
        }
    }
}

/*
 * A test suite that demonstrates the need for Welford's online algorithm for calculating variance
 * in Batchnorm Fwd
 */

// Idea: make this a template to also test this for FP16, BF16?
template <class T>
class batch_norm_fwd_spatial_welford : public testing::TestWithParam<TestCase>
{
    tensor<T> input;
    tensor<float> scale;
    tensor<float> shift;
    size_t n = 0U;
    size_t c = 0U;
    size_t h = 0U;
    size_t w = 0U;

public:
    void SetUp() override
    {
        prng::reset_seed();

        std::tie(n, c, h, w) = miopen::tien<4>(GetParam());
        input                = tensor<T>{n, c, h, w};
    }

    void Run()
    {
        // Set up forward pass

        // Initialize input
        const int range = 1000000;

        const double mu    = 10000;
        const double sigma = 4.0;

        /*
         * Notes on the choice of value for mu:
         * Running experiments on values to use for the test showed
         * that FP32 can handle an acceptable precision as long as
         * the mantissa of the mean value is within 7 digits.
         * This means that the moment the mean's exponent exceeds 7,
         * the variance quickly starts to diverge.
         * Yet, this test case already fails when Welford's algorithm
         * is not used.
         */

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        // Box-Muller transform for generating data
        for(std::size_t i = 0; i < input.GetSize() / 2; i++)
        {
            auto u1 = prng::gen_descreet_unsigned<float>(1.0 / range, range);
            auto u2 = prng::gen_descreet_unsigned<float>(1.0 / range, range);

            input[2 * i] =
                sigma * sqrt(-2 * log(u1 + 1e-7)) * cos(2 * std::numbers::pi_v<float> * u2) + mu;
            if(2 * i + 1 < input.GetSize())
                input[2 * i + 1] =
                    sigma * sqrt(-2 * log(u1 + 1e-7)) * sin(2 * std::numbers::pi_v<float> * u2) +
                    mu;
        }

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

        scale = tensor<PREC_TYPE>{rs_n_batch, rs_channels, rs_height, rs_width};
        shift = tensor<PREC_TYPE>{rs_n_batch, rs_channels, rs_height, rs_width};

        tensor<float> runMean = tensor<float>{rs_n_batch, rs_channels, rs_height, rs_width};
        tensor<float> runVar  = tensor<float>{rs_n_batch, rs_channels, rs_height, rs_width};

        auto saveMean   = tensor<float>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto saveInvVar = tensor<float>{rs_n_batch, rs_channels, rs_height, rs_width};

        auto cpuMean = tensor<double>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto cpuVar  = tensor<double>{rs_n_batch, rs_channels, rs_height, rs_width};

        computeVariance(cpuMean, cpuVar);

        for(std::size_t i = 0; i < runMean.GetSize(); i++)
        {
            // Corresponds to the momentum parameter in the original
            // issue demonstrator -- according to the PyTorch docs,
            // the momentum is the value with which the running mean
            // and running variance are initialized
            runMean[i] = 0.1f;
            runVar[i]  = 0.1f;
            scale[i]   = 0.1f;
            shift[i]   = 0.1f;
        }

        // in buffers
        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        // out buffers
        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<float>(channels);
        auto saveInvVar_dev = handle.Create<float>(channels);
        auto out_dev        = handle.Create<float>(n_batch * channels * height * width);

        const auto problem = miopen::batchnorm::ProblemDescription{
            miopenBNSpatial,
            input.desc,
            out.desc,
            scale.desc,
            shift.desc,
            saveMean.desc,
            saveInvVar.desc,
            expAvgFactor,
            epsilon,
            true,
            false,
            std::max(size_t{1}, size_t(0.6f * handle.GetMaxComputeUnits())),
            miopen::ActivationDescriptor{miopenActivationPASTHRU, 0.0, 0.0, 0.0}};
        const auto ctx = miopen::ExecutionContext{&handle};
        const auto solver = miopen::solver::batchnorm::BnFwdTrainingSpatial{};
        // Select the Welford implementation explicitly rather than depending on the heuristic or
        // a perf-db entry. Small NHW regression cases would otherwise use a different variant.
        const auto config = miopen::solver::batchnorm::PerformanceConfigBnFwdTraining{0, "Variant1-1"};
        ASSERT_TRUE(solver.IsApplicable(ctx, problem));
        const auto solution = solver.GetSolution(ctx, problem, config);
        ASSERT_TRUE(solution.Succeeded());
        ASSERT_TRUE(solution.invoker_factory);
        const auto invoker =
            handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);

        auto invoke_params = miopen::batchnorm::FwdTrainInvokeParams{};
        invoke_params.type                  = miopen::InvokeType::Run;
        invoke_params.x                     = in_dev.get();
        invoke_params.y                     = out_dev.get();
        invoke_params.bnScale               = scale_dev.get();
        invoke_params.bnBias                = shift_dev.get();
        invoke_params.expAvgFactor          = expAvgFactor;
        invoke_params.epsilon               = epsilon;
        invoke_params.resultSaveMean        = saveMean_dev.get();
        invoke_params.resultSaveInvVariance = saveInvVar_dev.get();
        invoker(handle, invoke_params);
        handle.Finish();

        saveMean.data   = handle.Read<float>(saveMean_dev, saveMean.data.size());
        saveInvVar.data = handle.Read<float>(saveInvVar_dev, saveInvVar.data.size());
        out.data        = handle.Read<float>(out_dev, out.data.size());

        if(n * h * w < 1024)
        {
            for(size_t cidx = 0; cidx < c; ++cidx)
            {
                EXPECT_NEAR(saveMean(0, cidx, 0, 0), cpuMean(0, cidx, 0, 0), 0.02)
                    << "channel " << cidx;
                const double inv_var = 1.0 / std::sqrt(cpuVar(0, cidx, 0, 0) + epsilon);
                for(size_t bidx = 0; bidx < n; ++bidx)
                    for(size_t row = 0; row < h; ++row)
                        for(size_t column = 0; column < w; ++column)
                        {
                            const double expected =
                                (double(input(bidx, cidx, row, column)) -
                                 cpuMean(0, cidx, 0, 0)) *
                                    inv_var * scale(0, cidx, 0, 0) +
                                shift(0, cidx, 0, 0);
                            EXPECT_NEAR(out(bidx, cidx, row, column), expected, 0.002)
                                << "at {" << bidx << ", " << cidx << ", " << row << ", "
                                << column << "}";
                        }
            }
        }

        bool variance_fitting = true;

        for(std::size_t nidx = 0; nidx < rs_n_batch; nidx++)
        {
            for(std::size_t cidx = 0; cidx < rs_channels; cidx++)
            {
                double invVar      = (1.0 / (sqrt(cpuVar(nidx, cidx, 0, 0) + epsilon)));
                bool curVarFitting = (abs(saveInvVar(nidx, cidx, 0, 0) - invVar) < 0.001);
                variance_fitting &= curVarFitting;
                if constexpr(MIO_BN_SP_TEST_DEBUG == 1)
                {
                    std::cout << "At {" << nidx << ", " << cidx
                              << ", 0, 0}, cpu: " << saveInvVar(nidx, cidx, 0, 0)
                              << " gpu: " << invVar << (curVarFitting ? " ok" : " FAIL")
                              << std::endl;
                }
            }
        }
        if(!variance_fitting)
        {
            GTEST_FAIL() << "Variance is not fitting the expected value of " << sigma;
        }
    }

private:
    void computeVariance(tensor<double>& cpuMean, tensor<double>& cpuVar)
    {

        miopen::par_for(c, 1, [&](int cidx) {
            double variance_accum = 0.;
            double mean_accum     = 0.;

            // Two-pass variance calculation

            // process the batch per channel
            for(int bidx = 0; bidx < n; bidx++)
            { // via mini_batch
                for(int row = 0; row < h; row++)
                { // via rows
                    for(int column = 0; column < w; column++)
                    { // via columns
                        auto inval = static_cast<double>(input(bidx, cidx, row, column));
                        mean_accum += inval;
                    } // end for (column)
                } // end for (row)
            } // end for (n)

            mean_accum /= (n * h * w);

            for(int bidx = 0; bidx < n; bidx++)
            { // via mini_batch
                for(int row = 0; row < h; row++)
                { // via rows
                    for(int column = 0; column < w; column++)
                    { // via columns
                        auto inval = static_cast<double>(input(bidx, cidx, row, column));
                        variance_accum += (inval - mean_accum) * (inval - mean_accum);
                    } // end for (column)
                } // end for (row)
            } // end for (n)

            cpuMean(0, cidx, 0, 0) = mean_accum;
            cpuVar(0, cidx, 0, 0)  = variance_accum / (n * h * w);
        });
    }
};

using GPU_BN_Spatial_FP32             = batch_norm_spatial_test<float>;
using GPU_BN_Fwd_Spatial_Welford_FP32 = batch_norm_fwd_spatial_welford<float>;

TEST_P(GPU_BN_Spatial_FP32, TestFloat32) { Run(); }

TEST_P(GPU_BN_Fwd_Spatial_Welford_FP32, TestFloat32) { Run(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_BN_Spatial_FP32, GetCases(), [](const auto& info_) {
    return NameGenerator(info_);
});

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BN_Spatial_FP32, GetCases(false), [](const auto& info_) {
    return NameGenerator(info_);
});

// Currently, the test suite for variance is running only for variant 1 cases.
// All other variants are failing.
// TODOs: Replace GetCasesFwdSpatialPerVariant with GetCases()
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BN_Fwd_Spatial_Welford_FP32,
                         GetCasesFwdSpatialPerVariant(1),
                         [](const auto& info_) { return NameGenerator(info_); });

TEST(GPU_BN_Spatial_BF16, SplitBatchPaddedTile)
{
    constexpr size_t n = 42, c = 3, h = 30, w = 40;
    constexpr size_t spatial = h * w;
    constexpr size_t element_count = n * c * spatial;
    constexpr size_t guard_count = 14 * c * spatial;
    const bfloat16 canary{123.0f};
    tensor<bfloat16> input{n, c, h, w};
    // Distinct channels and batches expose contributions from padded batch lanes.
    for(size_t bidx = 0; bidx < n; ++bidx)
        for(size_t cidx = 0; cidx < c; ++cidx)
            for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
                input[bidx * c * spatial + cidx * spatial + hwidx] =
                    bfloat16{float(int((bidx * 13 + hwidx * 7) % 31) - 15) * 0.125f +
                             float(cidx) * 2.0f};

    tensor<float> scale{1, c, 1, 1};
    tensor<float> shift{1, c, 1, 1};
    std::vector<double> means(c, 0), variances(c, 0);
    for(size_t cidx = 0; cidx < c; ++cidx)
    {
        scale[cidx] = 0.5f + float(cidx) * 0.25f;
        shift[cidx] = -0.25f + float(cidx) * 0.125f;
        for(size_t bidx = 0; bidx < n; ++bidx)
            for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
                means[cidx] += double(input[bidx * c * spatial + cidx * spatial + hwidx]);
        means[cidx] /= n * spatial;
        for(size_t bidx = 0; bidx < n; ++bidx)
            for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
            {
                const double delta =
                    double(input[bidx * c * spatial + cidx * spatial + hwidx]) - means[cidx];
                variances[cidx] += delta * delta;
            }
        variances[cidx] /= n * spatial;
    }

    auto&& handle = get_handle();
    auto padded_input = input.data;
    padded_input.resize(element_count + guard_count, canary);
    const auto in_dev = handle.Write(padded_input);
    const auto scale_dev = handle.Write(scale.data);
    const auto shift_dev = handle.Write(shift.data);
    const auto problem = miopen::batchnorm::ProblemDescription{
        miopenBNSpatial,
        input.desc,
        input.desc,
        scale.desc,
        shift.desc,
        scale.desc,
        scale.desc,
        MIO_BN_TEST_EXPAVGFACTOR,
        MIO_BN_TEST_EPSILON,
        true,
        false,
        std::max(size_t{1}, size_t(0.6f * handle.GetMaxComputeUnits())),
        miopen::ActivationDescriptor{miopenActivationPASTHRU, 0.0, 0.0, 0.0}};
    const auto ctx = miopen::ExecutionContext{&handle};
    const auto solver = miopen::solver::batchnorm::BnFwdTrainingSpatial{};
    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    for(const int vector_size : {1, 2, 4, 8})
    {
        if(vector_size == 8 &&
           (!miopen::StartsWith(handle.GetDeviceName(), "gfx125") ||
            handle.GetWavefrontWidth() != 32))
            continue;
        SCOPED_TRACE(vector_size);
        const auto config = miopen::solver::batchnorm::PerformanceConfigBnFwdTraining{
            0, "Variant2-" + std::to_string(vector_size) + "-1-256-2-14"};
        ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, config));
        const auto solution = solver.GetSolution(ctx, problem, config);
        ASSERT_TRUE(solution.Succeeded());
        ASSERT_TRUE(solution.invoker_factory);
        const auto invoker =
            handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
        // The final z lane starts at n=42. Padding the backing allocation makes any
        // erroneous write observable without corrupting an unrelated device allocation.
        auto out_dev =
            handle.Write(std::vector<bfloat16>(element_count + guard_count, canary));
        auto mean_dev = handle.Create<float>(c);
        auto inv_var_dev = handle.Create<float>(c);
        auto invoke_params = miopen::batchnorm::FwdTrainInvokeParams{};
        invoke_params.type                  = miopen::InvokeType::Run;
        invoke_params.x                     = in_dev.get();
        invoke_params.y                     = out_dev.get();
        invoke_params.bnScale               = scale_dev.get();
        invoke_params.bnBias                = shift_dev.get();
        invoke_params.expAvgFactor          = MIO_BN_TEST_EXPAVGFACTOR;
        invoke_params.epsilon               = MIO_BN_TEST_EPSILON;
        invoke_params.resultSaveMean        = mean_dev.get();
        invoke_params.resultSaveInvVariance = inv_var_dev.get();
        invoker(handle, invoke_params);
        handle.Finish();

        const auto gpu_means = handle.Read<float>(mean_dev, c);
        const auto gpu_inv_vars = handle.Read<float>(inv_var_dev, c);
        const auto output = handle.Read<bfloat16>(out_dev, element_count + guard_count);
        for(size_t cidx = 0; cidx < c; ++cidx)
        {
            const double inv_var =
                1.0 / std::sqrt(variances[cidx] + MIO_BN_TEST_EPSILON);
            ASSERT_NEAR(gpu_means[cidx], means[cidx], 1e-4) << "channel " << cidx;
            ASSERT_NEAR(gpu_inv_vars[cidx], inv_var, 1e-4) << "channel " << cidx;
            for(size_t bidx = 0; bidx < n; ++bidx)
                for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
                {
                    const size_t idx = bidx * c * spatial + cidx * spatial + hwidx;
                    const double expected =
                        (double(input[idx]) - means[cidx]) * inv_var * scale[cidx] + shift[cidx];
                    // Allow BF16 output rounding, not contamination from a padded batch.
                    ASSERT_NEAR(double(output[idx]), expected,
                                0.008 * std::max(1.0, std::abs(expected)))
                        << "at batch " << bidx << ", channel " << cidx << ", HW " << hwidx;
                }
        }
        for(size_t idx = element_count; idx < output.size(); ++idx)
            ASSERT_EQ(float(output[idx]), float(canary)) << "padded output index " << idx;
    }
}

template <class T>
void RunBufferedStableVarianceAndInPlace(std::initializer_list<size_t> batch_sizes,
                                       std::initializer_list<int> variants,
                                       float input_spread,
                                       double output_tolerance,
                                       bool round_reference)
{
    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx1250" || handle.GetWavefrontWidth() != 32)
        GTEST_SKIP() << "Buffered batchnorm requires gfx1250 wave32";

    constexpr size_t c = 3, guard_count = 64;
    const T canary{123.0f};
    constexpr float stats_canary = -321.0f;
    const auto ctx = miopen::ExecutionContext{&handle};
    const auto solver = miopen::solver::batchnorm::BnFwdTrainingSpatial{};
    tensor<float> scale{1, c, 1, 1};
    tensor<float> shift{1, c, 1, 1};
    for(size_t channel = 0; channel < c; ++channel)
    {
        scale[channel] = 0.5f + float(channel) * 0.25f;
        shift[channel] = -0.25f + float(channel) * 0.125f;
    }
    const auto scale_dev = handle.Write(scale.data);
    const auto shift_dev = handle.Write(shift.data);

    for(const size_t n : batch_sizes)
    {
        // The small batch leaves whole waves empty and a partial final vector-block.
        // Odd C exposes channel crossing; channel two has zero variance.
        const size_t spatial = n == 42 ? 1200 : 1208;
        const size_t element_count = n * c * spatial;
        SCOPED_TRACE("N=" + std::to_string(n) + ", HW=" + std::to_string(spatial));
        tensor<T> input{n, c, size_t{1}, spatial};
        std::vector<double> means(c, 0), variances(c, 0);
        for(size_t batch = 0; batch < n; ++batch)
            for(size_t channel = 0; channel < c; ++channel)
                for(size_t hw = 0; hw < spatial; ++hw)
                {
                    const float offset = channel == 1 ? -2048.0f : 2048.0f;
                    const float spread = channel == 2 ? 0.0f
                        : ((batch * 13 + hw * 7) % 3 == 0 ? -input_spread : input_spread);
                    const size_t idx = batch * c * spatial + channel * spatial + hw;
                    input[idx] = T{offset + spread};
                    means[channel] += double(input[idx]);
                }
        for(size_t channel = 0; channel < c; ++channel)
        {
            means[channel] /= n * spatial;
            for(size_t batch = 0; batch < n; ++batch)
                for(size_t hw = 0; hw < spatial; ++hw)
                {
                    const double delta =
                        double(input[batch * c * spatial + channel * spatial + hw]) -
                        means[channel];
                    variances[channel] += delta * delta;
                }
            variances[channel] /= n * spatial;
        }
        auto padded_input = input.data;
        padded_input.resize(element_count + guard_count, canary);
        const auto problem = miopen::batchnorm::ProblemDescription{
            miopenBNSpatial, input.desc, input.desc, scale.desc, shift.desc,
            scale.desc, scale.desc, MIO_BN_TEST_EXPAVGFACTOR, MIO_BN_TEST_EPSILON,
            true, false, std::max(size_t{1}, size_t(0.6f * handle.GetMaxComputeUnits())),
            miopen::ActivationDescriptor{miopenActivationPASTHRU, 0.0, 0.0, 0.0}};

        for(const int variant : variants)
          for(const size_t block : {512, 1024})
            for(const size_t vector_size : {4, 8})
            {
                const auto id = "Variant" + std::to_string(variant) + "-" +
                                std::to_string(vector_size) + "-" +
                                std::to_string(block) + "-1-1-1";
                SCOPED_TRACE(id);
                const auto config =
                    miopen::solver::batchnorm::PerformanceConfigBnFwdTraining{0, id};
                ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, config));
                const auto solution = solver.GetSolution(ctx, problem, config);
                ASSERT_TRUE(solution.Succeeded());
                ASSERT_TRUE(solution.invoker_factory);
                const auto invoker =
                    handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                for(const bool in_place : {false, true})
                {
                    SCOPED_TRACE(in_place ? "in-place" : "disjoint");
                    auto in_dev = handle.Write(padded_input);
                    auto out_dev =
                        handle.Write(std::vector<T>(element_count + guard_count, canary));
                    auto mean_dev = handle.Write(std::vector<float>(c + guard_count, stats_canary));
                    auto inv_var_dev =
                        handle.Write(std::vector<float>(c + guard_count, stats_canary));
                    auto invoke_params = miopen::batchnorm::FwdTrainInvokeParams{};
                    invoke_params.type                  = miopen::InvokeType::Run;
                    invoke_params.x                     = in_dev.get();
                    invoke_params.y                     = in_place ? in_dev.get() : out_dev.get();
                    invoke_params.bnScale               = scale_dev.get();
                    invoke_params.bnBias                = shift_dev.get();
                    invoke_params.expAvgFactor          = MIO_BN_TEST_EXPAVGFACTOR;
                    invoke_params.epsilon               = MIO_BN_TEST_EPSILON;
                    invoke_params.resultSaveMean        = mean_dev.get();
                    invoke_params.resultSaveInvVariance = inv_var_dev.get();
                    invoker(handle, invoke_params);
                    handle.Finish();
                    const auto gpu_means = handle.Read<float>(mean_dev, c + guard_count);
                    const auto gpu_inv_vars = handle.Read<float>(inv_var_dev, c + guard_count);
                    const auto output = handle.Read<T>(
                        in_place ? in_dev : out_dev, element_count + guard_count);
                    for(size_t channel = 0; channel < c; ++channel)
                    {
                        const double inv_var =
                            1.0 / std::sqrt(variances[channel] + MIO_BN_TEST_EPSILON);
                        ASSERT_NEAR(gpu_means[channel], means[channel], 5e-4);
                        ASSERT_NEAR(gpu_inv_vars[channel], inv_var, 1e-5 * inv_var);
                        for(size_t batch = 0; batch < n; ++batch)
                            for(size_t hw = 0; hw < spatial; ++hw)
                            {
                                const size_t idx = batch * c * spatial + channel * spatial + hw;
                                double expected =
                                    (double(input[idx]) - means[channel]) * inv_var *
                                        scale[channel] + shift[channel];
                                if(round_reference)
                                    expected = double(T{float(expected)});
                                ASSERT_NEAR(double(output[idx]), expected,
                                            output_tolerance * std::max(1.0, std::abs(expected)))
                                    << "batch " << batch << ", channel " << channel << ", HW " << hw;
                            }
                    }
                    for(size_t idx = element_count; idx < output.size(); ++idx)
                        ASSERT_EQ(float(output[idx]), float(canary)) << "output guard " << idx;
                    for(size_t idx = c; idx < gpu_means.size(); ++idx)
                    {
                        ASSERT_EQ(gpu_means[idx], stats_canary) << "mean guard " << idx;
                        ASSERT_EQ(gpu_inv_vars[idx], stats_canary) << "inverse variance guard " << idx;
                    }
                }
            }
    }
}

TEST(GPU_BN_Spatial_BF16, BufferedStableVarianceAndInPlace)
{
    RunBufferedStableVarianceAndInPlace<bfloat16>({42, 1}, {4, 6}, 16.0f, 0.008, false);
}

TEST(GPU_BN_Spatial_FP16, BufferedStableVarianceAndInPlace)
{
    // ±4 remains exactly representable around ±2048 in FP16, while naive
    // E[x²] - E[x]² loses precision. N=3 also retains odd pairs of half values.
    RunBufferedStableVarianceAndInPlace<half_float::half>({42, 3}, {4}, 4.0f, 0.002, true);
}

TEST(GPU_BN_Spatial_BF16, BufferedConfigurationBounds)
{
    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx1250" || handle.GetWavefrontWidth() != 32)
        GTEST_SKIP() << "Buffered BF16 batchnorm requires gfx1250 wave32";

    const auto ctx = miopen::ExecutionContext{&handle};
    const auto solver = miopen::solver::batchnorm::BnFwdTrainingSpatial{};
    tensor<float> params{1, 3, 1, 1};
    auto make_problem = [&](size_t n, size_t spatial) {
        tensor<bfloat16> input{n, size_t{3}, size_t{1}, spatial};
        return miopen::batchnorm::ProblemDescription{
            miopenBNSpatial, input.desc, input.desc, params.desc, params.desc,
            params.desc, params.desc, MIO_BN_TEST_EXPAVGFACTOR, MIO_BN_TEST_EPSILON,
            true, false, size_t{1},
            miopen::ActivationDescriptor{miopenActivationPASTHRU, 0.0, 0.0, 0.0}};
    };
    auto expect_rejected = [&](const auto& problem, const std::string& id) {
        SCOPED_TRACE(id);
        const auto config = miopen::solver::batchnorm::PerformanceConfigBnFwdTraining{0, id};
        EXPECT_FALSE(solver.IsValidPerformanceConfig(ctx, problem, config));
        EXPECT_THROW(solver.GetSolution(ctx, problem, config), miopen::Exception);
    };
    const auto supported = make_problem(42, 1200);
    for(const auto* id : {"Variant4-2-512-1-1-1", "Variant4-4-128-1-1-1",
                         "Variant4-4-512-2-1-1", "Variant4-4-512-1-1-2",
                         "Variant4-4"})
        expect_rejected(supported, id);
    expect_rejected(make_problem(65, 1200), "Variant4-4-1024-1-1-1");
    expect_rejected(make_problem(1, 4808), "Variant4-8-1024-1-1-1");
    expect_rejected(make_problem(1, 1204), "Variant4-8-512-1-1-1");
    expect_rejected(make_problem(64, 4800), "Variant4-4-1024-1-1-1");
    expect_rejected(make_problem(42, 4800), "Variant4-8-1024-1-1-1");
    expect_rejected(supported, "Variant4-4-256-1-1-1");
}

TEST(GPU_BN_Spatial_BF16, BackwardSplitBatchPaddedTile)
{
    constexpr size_t n = 42, c = 3, h = 30, w = 40;
    constexpr size_t spatial = h * w;
    constexpr size_t element_count = n * c * spatial;
    constexpr size_t guard_count = 14 * c * spatial;
    constexpr double nhw = n * spatial;
    const bfloat16 canary{123.0f};
    constexpr float gradient_canary = -321.0f;
    tensor<bfloat16> input{n, c, h, w};
    tensor<bfloat16> dy{n, c, h, w};
    tensor<float> scale{1, c, 1, 1};
    tensor<float> shift{1, c, 1, 1};
    tensor<float> saved_mean{1, c, 1, 1};
    tensor<float> saved_inv_var{1, c, 1, 1};
    // Correlated but distinct input and dy make both parameter gradients nonzero,
    // and expose stale contributions from the final, padded batch-split lane.
    for(size_t bidx = 0; bidx < n; ++bidx)
        for(size_t cidx = 0; cidx < c; ++cidx)
            for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
            {
                const size_t idx = bidx * c * spatial + cidx * spatial + hwidx;
                const float value =
                    float(int((bidx * 13 + hwidx * 7 + cidx * 5) % 31) - 15) * 0.125f +
                    float(cidx) * 0.75f + float(bidx % 7) * 0.0625f;
                input[idx] = bfloat16{value};
                dy[idx] = bfloat16{
                    0.25f * float(input[idx]) + 0.5f + float(cidx) * 0.125f +
                    float(int((bidx * 3 + hwidx * 11 + cidx) % 17) - 8) * 0.0625f};
            }

    std::vector<double> means(c, 0), variances(c, 0), inv_vars(c, 0);
    std::vector<double> dbias(c, 0), dscale(c, 0);
    for(size_t cidx = 0; cidx < c; ++cidx)
    {
        scale[cidx] = 0.5f + float(cidx) * 0.25f;
        shift[cidx] = -0.25f + float(cidx) * 0.125f;
        for(size_t bidx = 0; bidx < n; ++bidx)
            for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
                means[cidx] += double(input[bidx * c * spatial + cidx * spatial + hwidx]);
        means[cidx] /= nhw;
        // Independent double-precision, two-pass population variance.
        for(size_t bidx = 0; bidx < n; ++bidx)
            for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
            {
                const size_t idx = bidx * c * spatial + cidx * spatial + hwidx;
                const double delta = double(input[idx]) - means[cidx];
                variances[cidx] += delta * delta;
            }
        variances[cidx] /= nhw;
        inv_vars[cidx] = 1.0 / std::sqrt(variances[cidx] + MIO_BN_TEST_EPSILON);
        saved_mean[cidx] = float(means[cidx]);
        saved_inv_var[cidx] = float(inv_vars[cidx]);
        for(size_t bidx = 0; bidx < n; ++bidx)
            for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
            {
                const size_t idx = bidx * c * spatial + cidx * spatial + hwidx;
                const double xhat = (double(input[idx]) - means[cidx]) * inv_vars[cidx];
                dbias[cidx] += double(dy[idx]);
                dscale[cidx] += double(dy[idx]) * xhat;
            }
    }

    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx1250" || handle.GetWavefrontWidth() != 32)
        GTEST_SKIP() << "Split-batch BF16 backward batchnorm requires gfx1250 wave32";

    auto padded_input = input.data;
    auto padded_dy = dy.data;
    padded_input.resize(element_count + guard_count, canary);
    padded_dy.resize(element_count + guard_count, canary);
    const auto in_dev = handle.Write(padded_input);
    const auto dy_dev = handle.Write(padded_dy);
    const auto scale_dev = handle.Write(scale.data);
    const auto shift_dev = handle.Write(shift.data);
    const auto mean_dev = handle.Write(saved_mean.data);
    const auto inv_var_dev = handle.Write(saved_inv_var.data);
    const auto ctx = miopen::ExecutionContext{&handle};
    const auto solver = miopen::solver::batchnorm::BnBwdTrainingSpatial{};
    for(const bool use_saved : {true, false})
    {
        SCOPED_TRACE(use_saved);
        const auto problem = miopen::batchnorm::ProblemDescription{
            miopenBNSpatial,
            input.desc,
            dy.desc,
            input.desc,
            scale.desc,
            shift.desc,
            saved_mean.desc,
            saved_inv_var.desc,
            MIO_BN_TEST_EPSILON,
            use_saved,
            std::max(size_t{1}, size_t(0.6f * handle.GetMaxComputeUnits())),
            miopen::ActivationDescriptor{miopenActivationPASTHRU, 0.0, 0.0, 0.0}};
        ASSERT_TRUE(solver.IsApplicable(ctx, problem));
        for(const int vector_size : {1, 2, 4, 8})
        {
            SCOPED_TRACE(vector_size);
            const auto config = miopen::solver::batchnorm::PerformanceConfigBnBwdBackward{
                0, "Variant2-" + std::to_string(vector_size) + "-1-256-2-14"};
            ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, config));
            const auto solution = solver.GetSolution(ctx, problem, config);
            ASSERT_TRUE(solution.Succeeded());
            ASSERT_TRUE(solution.invoker_factory);
            const auto invoker =
                handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
            // Three 14-batch tiles occupy two z=2 workgroups. The final lane
            // starts at n=42, so any unguarded read or write touches this padding.
            auto dx_dev =
                handle.Write(std::vector<bfloat16>(element_count + guard_count, canary));
            auto dscale_dev = handle.Write(std::vector<float>(c + 4, gradient_canary));
            auto dbias_dev = handle.Write(std::vector<float>(c + 4, gradient_canary));
            auto invoke_params = miopen::batchnorm::BwdInvokeParams{};
            invoke_params.type = miopen::InvokeType::Run;
            invoke_params.x = in_dev.get();
            invoke_params.dy = dy_dev.get();
            invoke_params.dx = dx_dev.get();
            invoke_params.bnScale = scale_dev.get();
            invoke_params.bnBias = shift_dev.get();
            invoke_params.resultBnScaleDiff = dscale_dev.get();
            invoke_params.resultBnBiasDiff = dbias_dev.get();
            invoke_params.epsilon = MIO_BN_TEST_EPSILON;
            invoke_params.savedMean = use_saved ? mean_dev.get() : nullptr;
            invoke_params.savedInvVariance = use_saved ? inv_var_dev.get() : nullptr;
            invoker(handle, invoke_params);
            handle.Finish();

            const auto output = handle.Read<bfloat16>(dx_dev, element_count + guard_count);
            const auto gpu_dscale = handle.Read<float>(dscale_dev, c + 4);
            const auto gpu_dbias = handle.Read<float>(dbias_dev, c + 4);
            for(size_t cidx = 0; cidx < c; ++cidx)
            {
                ASSERT_NEAR(gpu_dscale[cidx], dscale[cidx],
                            2e-4 * std::max(1.0, std::abs(dscale[cidx])))
                    << "dscale channel " << cidx;
                ASSERT_NEAR(gpu_dbias[cidx], dbias[cidx],
                            2e-4 * std::max(1.0, std::abs(dbias[cidx])))
                    << "dbias channel " << cidx;
                for(size_t bidx = 0; bidx < n; ++bidx)
                    for(size_t hwidx = 0; hwidx < spatial; ++hwidx)
                    {
                        const size_t idx = bidx * c * spatial + cidx * spatial + hwidx;
                        const double xhat =
                            (double(input[idx]) - means[cidx]) * inv_vars[cidx];
                        const double expected = double(scale[cidx]) * inv_vars[cidx] / nhw *
                            (nhw * double(dy[idx]) - dbias[cidx] - xhat * dscale[cidx]);
                        ASSERT_NEAR(double(output[idx]), expected,
                                    0.008 * std::abs(expected) + 0.001)
                            << "dx at batch " << bidx << ", channel " << cidx
                            << ", HW " << hwidx;
                    }
            }
            const auto gpu_input = handle.Read<bfloat16>(in_dev, element_count + guard_count);
            const auto gpu_dy = handle.Read<bfloat16>(dy_dev, element_count + guard_count);
            for(size_t idx = element_count; idx < output.size(); ++idx)
            {
                ASSERT_EQ(float(output[idx]), float(canary)) << "padded dx index " << idx;
                ASSERT_EQ(float(gpu_input[idx]), float(canary)) << "padded input index " << idx;
                ASSERT_EQ(float(gpu_dy[idx]), float(canary)) << "padded dy index " << idx;
            }
            for(size_t idx = c; idx < gpu_dscale.size(); ++idx)
            {
                ASSERT_EQ(gpu_dscale[idx], gradient_canary) << "padded dscale index " << idx;
                ASSERT_EQ(gpu_dbias[idx], gradient_canary) << "padded dbias index " << idx;
            }
        }
    }
}

TEST(GPU_BN_Spatial_BF16, BackwardBufferedSavedActivationAndInPlace)
{
    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx1250" || handle.GetWavefrontWidth() != 32)
        GTEST_SKIP() << "Buffered BF16 backward batchnorm requires gfx1250 wave32";
    constexpr size_t c = 3, guard = 32;
    const bfloat16 canary{123.0f};
    constexpr float gradient_canary = -321.0f;
    const auto ctx = miopen::ExecutionContext{&handle};
    const auto solver = miopen::solver::batchnorm::BnBwdTrainingSpatial{};
    for(const auto shape : {std::pair<size_t, size_t>{42, 1200}, {1, 1208}, {42, 4800}})
    {
        const size_t n = shape.first, hw = shape.second, count = n * c * hw;
        const double nhw = n * hw;
        SCOPED_TRACE(::testing::Message() << "N=" << n << " HW=" << hw);
        tensor<bfloat16> x{n, c, 1, hw}, dy{n, c, 1, hw};
        tensor<float> scale{1, c, 1, 1}, bias{1, c, 1, 1};
        tensor<float> mean{1, c, 1, 1}, inv{1, c, 1, 1};
        std::vector<double> db(c, 0), ds(c, 0), expected(count);
        for(size_t ch = 0; ch < c; ++ch)
        {
            scale[ch] = 0.75f + 0.25f * ch;
            bias[ch] = -0.25f;
            double sum = 0, variance = 0;
            for(size_t b = 0; b < n; ++b)
                for(size_t s = 0; s < hw; ++s)
                {
                    const size_t i = b * c * hw + ch * hw + s;
                    x[i] = bfloat16{256.0f + float(ch * 8) +
                                   float(int((b * 7 + s * 3) % 17) - 8) * 2.0f};
                    dy[i] = bfloat16{0.5f + float(int((b * 3 + s * 5) % 13) - 6) * 0.125f};
                    sum += double(x[i]);
                }
            mean[ch] = float(sum / nhw);
            for(size_t b = 0; b < n; ++b)
                for(size_t s = 0; s < hw; ++s)
                {
                    const double delta = double(x[b * c * hw + ch * hw + s]) - mean[ch];
                    variance += delta * delta;
                }
            inv[ch] = float(1.0 / std::sqrt(variance / nhw + MIO_BN_TEST_EPSILON));
            for(size_t b = 0; b < n; ++b)
                for(size_t s = 0; s < hw; ++s)
                {
                    const size_t i = b * c * hw + ch * hw + s;
                    const double xhat = (double(x[i]) - mean[ch]) * inv[ch];
                    const double value = xhat * scale[ch] + bias[ch] > 0 ? double(dy[i]) : 0;
                    db[ch] += value;
                    ds[ch] += value * xhat;
                }
            for(size_t b = 0; b < n; ++b)
                for(size_t s = 0; s < hw; ++s)
                {
                    const size_t i = b * c * hw + ch * hw + s;
                    const double xhat = (double(x[i]) - mean[ch]) * inv[ch];
                    const double value = xhat * scale[ch] + bias[ch] > 0 ? double(dy[i]) : 0;
                    expected[i] = double(scale[ch]) * inv[ch] / nhw *
                                  (nhw * value - db[ch] - xhat * ds[ch]);
                }
        }
        const auto problem = miopen::batchnorm::ProblemDescription{
            miopenBNSpatial, x.desc, dy.desc, x.desc, scale.desc, bias.desc, mean.desc, inv.desc,
            MIO_BN_TEST_EPSILON, true,
            std::max(size_t{1}, size_t(0.6f * handle.GetMaxComputeUnits())),
            miopen::ActivationDescriptor{miopenActivationRELU, 0.0, 0.0, 0.0}};
        const auto scale_dev = handle.Write(scale.data), bias_dev = handle.Write(bias.data);
        const auto mean_dev = handle.Write(mean.data), inv_dev = handle.Write(inv.data);
        auto padded_x = x.data, padded_dy = dy.data;
        padded_x.resize(count + guard, canary);
        padded_dy.resize(count + guard, canary);
        for(const int variant : {5, 8})
          for(const int vector_size : {4, 8})
            for(const int alias : {0, 1, 2})
            {
                if(variant == 5 && hw > 1208)
                    continue;
                SCOPED_TRACE(::testing::Message() << "variant=" << variant
                                                 << " V=" << vector_size << " alias=" << alias);
                const auto config = miopen::solver::batchnorm::PerformanceConfigBnBwdBackward{
                    0, "Variant" + std::to_string(variant) + "-" +
                       std::to_string(vector_size) + "-1024-1-1-1"};
                ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, config));
                const auto solution = solver.GetSolution(ctx, problem, config);
                ASSERT_TRUE(solution.Succeeded());
                ASSERT_TRUE(solution.invoker_factory);
                const auto invoker =
                    handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                auto x_dev = handle.Write(padded_x), dy_dev = handle.Write(padded_dy);
                auto dx_dev = handle.Write(std::vector<bfloat16>(count + guard, canary));
                auto ds_dev = handle.Write(std::vector<float>(c + guard, gradient_canary));
                auto db_dev = handle.Write(std::vector<float>(c + guard, gradient_canary));
                const auto& output_dev = alias == 1 ? x_dev : alias == 2 ? dy_dev : dx_dev;
                auto params = miopen::batchnorm::BwdInvokeParams{};
                params.type = miopen::InvokeType::Run;
                params.x = x_dev.get();
                params.dy = dy_dev.get();
                params.dx = output_dev.get();
                params.bnScale = scale_dev.get();
                params.bnBias = bias_dev.get();
                params.resultBnScaleDiff = ds_dev.get();
                params.resultBnBiasDiff = db_dev.get();
                params.savedMean = mean_dev.get();
                params.savedInvVariance = inv_dev.get();
                params.epsilon = MIO_BN_TEST_EPSILON;
                invoker(handle, params);
                handle.Finish();
                const auto output = handle.Read<bfloat16>(output_dev, count + guard);
                const auto gpu_ds = handle.Read<float>(ds_dev, c + guard);
                const auto gpu_db = handle.Read<float>(db_dev, c + guard);
                for(size_t i = 0; i < count; ++i)
                    ASSERT_NEAR(double(output[i]), expected[i],
                                0.008 * std::abs(expected[i]) + 0.001) << "DX index " << i;
                for(size_t ch = 0; ch < c; ++ch)
                {
                    ASSERT_NEAR(gpu_ds[ch], ds[ch], 2e-4 * std::max(1.0, std::abs(ds[ch])));
                    ASSERT_NEAR(gpu_db[ch], db[ch], 2e-4 * std::max(1.0, std::abs(db[ch])));
                }
                const auto gpu_x = handle.Read<bfloat16>(x_dev, count + guard);
                const auto gpu_dy = handle.Read<bfloat16>(dy_dev, count + guard);
                for(size_t i = count; i < count + guard; ++i)
                {
                    ASSERT_EQ(float(output[i]), float(canary));
                    ASSERT_EQ(float(gpu_x[i]), float(canary));
                    ASSERT_EQ(float(gpu_dy[i]), float(canary));
                }
                for(size_t i = c; i < c + guard; ++i)
                {
                    ASSERT_EQ(gpu_ds[i], gradient_canary);
                    ASSERT_EQ(gpu_db[i], gradient_canary);
                }
            }
    }
}

TEST(GPU_BN_Spatial_BF16, BackwardVariant1OddSpatial)
{
    auto&& handle = get_handle();
    const auto ctx = miopen::ExecutionContext{&handle};
    const auto solver = miopen::solver::batchnorm::BnBwdTrainingSpatial{};
    constexpr size_t c = 3, guard_count = 64;
    const bfloat16 canary{123.0f};
    constexpr float gradient_canary = -321.0f;
    // Odd HW uses scalar reads/writes. Adjacent aligned cases retain the two- and
    // four-element fast paths, including partial workgroup tails.
    for(const auto shape : {std::pair<size_t, size_t>{12, 5369},
                            {3, 4097}, {3, 4098}, {3, 4100}})
    {
        const size_t n = shape.first, spatial = shape.second;
        const size_t count = n * c * spatial;
        const double nhw = double(n * spatial);
        SCOPED_TRACE(n);
        SCOPED_TRACE(spatial);
        tensor<bfloat16> input{n, c, 1, spatial};
        tensor<bfloat16> dy{n, c, 1, spatial};
        tensor<float> scale{1, c, 1, 1};
        tensor<float> shift{1, c, 1, 1};
        tensor<float> saved_mean{1, c, 1, 1};
        tensor<float> saved_inv_var{1, c, 1, 1};
        for(size_t batch = 0; batch < n; ++batch)
            for(size_t channel = 0; channel < c; ++channel)
                for(size_t hw = 0; hw < spatial; ++hw)
                {
                    const size_t idx = batch * c * spatial + channel * spatial + hw;
                    // The four-position pattern exposes scalar loops that still
                    // start at 4*lid; batch/channel offsets expose wrong partitions.
                    input[idx] = bfloat16{
                        -0.5f + float(channel) + float(hw % 4) * 0.25f +
                        float(batch % 5) * 0.0625f +
                        float(int((hw * 7 + batch * 3) % 9) - 4) * 0.03125f};
                    dy[idx] = bfloat16{
                        0.125f * float(input[idx]) + 0.5f + float(channel) * 0.125f +
                        float(int((hw * 11 + batch * 5 + channel) % 17) - 8) * 0.0625f};
                }

        std::vector<double> means(c, 0), variances(c, 0), inv_vars(c, 0);
        std::vector<double> dbias(c, 0), dscale(c, 0);
        for(size_t channel = 0; channel < c; ++channel)
        {
            scale[channel] = 0.5f + float(channel) * 0.25f;
            shift[channel] = -0.25f + float(channel) * 0.125f;
            for(size_t batch = 0; batch < n; ++batch)
                for(size_t hw = 0; hw < spatial; ++hw)
                    means[channel] += double(input[batch * c * spatial + channel * spatial + hw]);
            means[channel] /= nhw;
            // Independent double-precision, two-pass population variance.
            for(size_t batch = 0; batch < n; ++batch)
                for(size_t hw = 0; hw < spatial; ++hw)
                {
                    const size_t idx = batch * c * spatial + channel * spatial + hw;
                    const double delta = double(input[idx]) - means[channel];
                    variances[channel] += delta * delta;
                }
            variances[channel] /= nhw;
            inv_vars[channel] = 1.0 / std::sqrt(variances[channel] + MIO_BN_TEST_EPSILON);
            saved_mean[channel] = float(means[channel]);
            saved_inv_var[channel] = float(inv_vars[channel]);
            for(size_t batch = 0; batch < n; ++batch)
                for(size_t hw = 0; hw < spatial; ++hw)
                {
                    const size_t idx = batch * c * spatial + channel * spatial + hw;
                    const double xhat = (double(input[idx]) - means[channel]) * inv_vars[channel];
                    dbias[channel] += double(dy[idx]);
                    dscale[channel] += double(dy[idx]) * xhat;
                }
        }
        auto padded_input = input.data, padded_dy = dy.data;
        padded_input.resize(count + guard_count, canary);
        padded_dy.resize(count + guard_count, canary);
        const auto input_dev = handle.Write(padded_input);
        const auto dy_dev = handle.Write(padded_dy);
        const auto scale_dev = handle.Write(scale.data);
        const auto shift_dev = handle.Write(shift.data);
        const auto mean_dev = handle.Write(saved_mean.data);
        const auto inv_var_dev = handle.Write(saved_inv_var.data);
        for(const bool use_saved : {false, true})
        {
            SCOPED_TRACE(use_saved);
            const auto problem = miopen::batchnorm::ProblemDescription{
                miopenBNSpatial, input.desc, dy.desc, input.desc, scale.desc, shift.desc,
                saved_mean.desc, saved_inv_var.desc, MIO_BN_TEST_EPSILON, use_saved,
                std::max(size_t{1}, size_t(0.6f * handle.GetMaxComputeUnits())),
                miopen::ActivationDescriptor{miopenActivationPASTHRU, 0.0, 0.0, 0.0}};
            const auto config =
                miopen::solver::batchnorm::PerformanceConfigBnBwdBackward{0, "Variant1-1"};
            ASSERT_TRUE(solver.IsApplicable(ctx, problem));
            ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, config));
            const auto solution = solver.GetSolution(ctx, problem, config);
            ASSERT_TRUE(solution.Succeeded());
            ASSERT_TRUE(solution.invoker_factory);
            const auto invoker =
                handle.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
            auto dx_dev = handle.Write(std::vector<bfloat16>(count + guard_count, canary));
            auto ds_dev = handle.Write(std::vector<float>(c + guard_count, gradient_canary));
            auto db_dev = handle.Write(std::vector<float>(c + guard_count, gradient_canary));
            auto params = miopen::batchnorm::BwdInvokeParams{};
            params.type = miopen::InvokeType::Run;
            params.x = input_dev.get();
            params.dy = dy_dev.get();
            params.dx = dx_dev.get();
            params.bnScale = scale_dev.get();
            params.bnBias = shift_dev.get();
            params.resultBnScaleDiff = ds_dev.get();
            params.resultBnBiasDiff = db_dev.get();
            params.epsilon = MIO_BN_TEST_EPSILON;
            params.savedMean = use_saved ? mean_dev.get() : nullptr;
            params.savedInvVariance = use_saved ? inv_var_dev.get() : nullptr;
            invoker(handle, params);
            handle.Finish();
            const auto output = handle.Read<bfloat16>(dx_dev, count + guard_count);
            const auto gpu_ds = handle.Read<float>(ds_dev, c + guard_count);
            const auto gpu_db = handle.Read<float>(db_dev, c + guard_count);
            for(size_t channel = 0; channel < c; ++channel)
            {
                ASSERT_NEAR(gpu_ds[channel], dscale[channel],
                            2e-4 * std::max(1.0, std::abs(dscale[channel])))
                    << "dscale channel " << channel;
                ASSERT_NEAR(gpu_db[channel], dbias[channel],
                            2e-4 * std::max(1.0, std::abs(dbias[channel])))
                    << "dbias channel " << channel;
                for(size_t batch = 0; batch < n; ++batch)
                    for(size_t hw = 0; hw < spatial; ++hw)
                    {
                        const size_t idx = batch * c * spatial + channel * spatial + hw;
                        const double xhat =
                            (double(input[idx]) - means[channel]) * inv_vars[channel];
                        const double expected = double(scale[channel]) * inv_vars[channel] / nhw *
                            (nhw * double(dy[idx]) - dbias[channel] - xhat * dscale[channel]);
                        ASSERT_NEAR(double(output[idx]), expected,
                                    0.008 * std::abs(expected) + 0.001)
                            << "dx at batch " << batch << ", channel " << channel
                            << ", HW " << hw;
                    }
            }
            for(size_t idx = count; idx < output.size(); ++idx)
                ASSERT_EQ(float(output[idx]), float(canary)) << "dx guard " << idx;
            for(size_t idx = c; idx < gpu_ds.size(); ++idx)
            {
                ASSERT_EQ(gpu_ds[idx], gradient_canary) << "dscale guard " << idx;
                ASSERT_EQ(gpu_db[idx], gradient_canary) << "dbias guard " << idx;
            }
        }
    }
}

TEST(GPU_BN_Spatial_BF16, NhwcConfiguredOutputRounding)
{
    auto&& handle = get_handle();
    constexpr size_t n = 16, c = 2, h = 24, w = 16;
    constexpr size_t spatial = h * w, count = n * c * spatial, guard_count = 32;
    constexpr double epsilon = 0.25;
    const bfloat16 canary{123.0f};
    constexpr float stats_canary = -321.0f;
    const std::vector<size_t> lengths{n, c, h, w};
    const std::vector<size_t> stats_lengths{1, c, 1, 1};
    tensor<bfloat16> input{miopenTensorNHWC, lengths};
    tensor<bfloat16> output{miopenTensorNHWC, lengths};
    tensor<float> scale{miopenTensorNHWC, stats_lengths};
    tensor<float> bias{miopenTensorNHWC, stats_lengths};
    tensor<float> saved_mean{miopenTensorNHWC, stats_lengths};
    tensor<float> saved_inv_var{miopenTensorNHWC, stats_lengths};
    scale[0] = 1.123f;
    scale[1] = -0.873f;
    bias[0] = 0.023f;
    bias[1] = -0.117f;

    // Each channel has exactly balanced, BF16-exact +/-1 values. Batch/channel
    // offsets distinguish NHWC indexing while keeping mean=0 and variance=1.
    for(size_t batch = 0; batch < n; ++batch)
        for(size_t hw = 0; hw < spatial; ++hw)
            for(size_t channel = 0; channel < c; ++channel)
                input[(batch * spatial + hw) * c + channel] =
                    bfloat16{(batch + hw + channel) % 2 == 0 ? -1.0f : 1.0f};

    std::vector<double> means(c, 0), variances(c, 0), inv_vars(c, 0);
    std::vector<bfloat16> expected(2 * c);
    for(size_t channel = 0; channel < c; ++channel)
    {
        // Independent double-precision, two-pass population statistics.
        for(size_t batch = 0; batch < n; ++batch)
            for(size_t hw = 0; hw < spatial; ++hw)
                means[channel] += double(input[(batch * spatial + hw) * c + channel]);
        means[channel] /= double(n * spatial);
        for(size_t batch = 0; batch < n; ++batch)
            for(size_t hw = 0; hw < spatial; ++hw)
            {
                const double delta =
                    double(input[(batch * spatial + hw) * c + channel]) - means[channel];
                variances[channel] += delta * delta;
            }
        variances[channel] /= double(n * spatial);
        inv_vars[channel] = 1.0 / std::sqrt(variances[channel] + epsilon);
        for(size_t sign = 0; sign < 2; ++sign)
        {
            const float value = sign == 0 ? -1.0f : 1.0f;
            const float normalized =
                (value - float(means[channel])) * float(inv_vars[channel]);
            // Use the configured host BF16 conversion, not a BF16-sized tolerance.
            // The four FP32 values are at least 9.7e-5 from both BF16 values and
            // their midpoints, so small GPU rsqrt/reduction errors cannot change
            // the expected quantization. Three values differ between RNE and RTZ.
            expected[sign * c + channel] =
                bfloat16{std::fma(scale[channel], normalized, bias[channel])};
        }
    }

    auto padded_input = input.data;
    padded_input.resize(count + guard_count, canary);
    const auto input_dev = handle.Write(padded_input);
    const auto scale_dev = handle.Write(scale.data);
    const auto bias_dev = handle.Write(bias.data);
    const auto output_dev = handle.Write(std::vector<bfloat16>(count + guard_count, canary));
    const auto mean_dev = handle.Write(std::vector<float>(c + guard_count, stats_canary));
    const auto inv_var_dev = handle.Write(std::vector<float>(c + guard_count, stats_canary));
    float alpha = 1.0f;
    float beta = 0.0f;
    ASSERT_EQ(miopenBatchNormalizationForwardTraining_V2(&handle,
                                                        miopenBNSpatial,
                                                        &alpha,
                                                        &beta,
                                                        &input.desc,
                                                        input_dev.get(),
                                                        &output.desc,
                                                        output_dev.get(),
                                                        &scale.desc,
                                                        &bias.desc,
                                                        &saved_mean.desc,
                                                        &saved_inv_var.desc,
                                                        scale_dev.get(),
                                                        bias_dev.get(),
                                                        MIO_BN_TEST_EXPAVGFACTOR,
                                                        nullptr,
                                                        nullptr,
                                                        epsilon,
                                                        mean_dev.get(),
                                                        inv_var_dev.get()),
              miopenStatusSuccess);
    handle.Finish();
    const auto gpu_output = handle.Read<bfloat16>(output_dev, count + guard_count);
    const auto gpu_mean = handle.Read<float>(mean_dev, c + guard_count);
    const auto gpu_inv_var = handle.Read<float>(inv_var_dev, c + guard_count);
    for(size_t channel = 0; channel < c; ++channel)
    {
        ASSERT_NEAR(gpu_mean[channel], means[channel], 1e-6) << "mean channel " << channel;
        ASSERT_NEAR(gpu_inv_var[channel], inv_vars[channel], 1e-6)
            << "inverse standard deviation channel " << channel;
    }
    for(size_t i = 0; i < count; ++i)
    {
        const size_t channel = i % c;
        const size_t sign = float(input[i]) < 0 ? 0 : 1;
        ASSERT_EQ(float(gpu_output[i]), float(expected[sign * c + channel]))
            << "output index " << i << ", channel " << channel;
    }
    for(size_t i = count; i < gpu_output.size(); ++i)
        ASSERT_EQ(float(gpu_output[i]), float(canary)) << "output guard " << i;
    for(size_t i = c; i < gpu_mean.size(); ++i)
    {
        ASSERT_EQ(gpu_mean[i], stats_canary) << "mean guard " << i;
        ASSERT_EQ(gpu_inv_var[i], stats_canary) << "inverse standard deviation guard " << i;
    }
}
