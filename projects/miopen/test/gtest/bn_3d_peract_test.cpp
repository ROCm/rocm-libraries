// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <chrono>

#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <miopen/activ.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/ford.hpp>
#include <miopen/miopen.h>
#include <miopen/par_for.hpp>
#include <miopen/tensor.hpp>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "test_parameter_name_generator.hpp"
#include "verify.hpp"

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
struct verify_forward_train_3d_bn_per_activation
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

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(out.begin(), out.end(), 0);

        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);

        std::size_t rs_n_batch, rs_channels, rs_depth, rs_height, rs_width;
        std::tie(rs_n_batch, rs_channels, rs_depth, rs_height, rs_width) =
            miopen::tien<5>(derivedBnDesc.GetLengths());

        tensor<U> runMean;
        tensor<U> runVar;

        if(input.desc.GetType() == miopenFloat)
        {
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
            runVar = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
        }
        else
        {
            prng::reset_seed();
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};
            runVar  = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};

            const double Data_scale = 0.001;
            for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
            {
                runMean[i] = prng::gen_descreet_uniform_sign<U>(Data_scale, 100);
                runVar[i]  = prng::gen_descreet_unsigned<U>(Data_scale, 100);
            }
        }

        auto saveMean   = tensor<U>{1, channels, depth, height, width};
        auto saveInvVar = tensor<U>{1, channels, depth, height, width};
        const auto n    = static_cast<double>(n_batch);

        const auto& strides  = input.desc.GetStrides();
        const auto& dstrides = derivedBnDesc.GetStrides();

        miopen::par_for(channels, 1, [&](int cidx) {
            miopen::ford(depth, height, width)([&](int didx, int row, int col) {
                double mean_accum     = 0.;
                double variance_accum = 0.;
                double elemStd        = 0.;
                double elemInvVar     = 0.;
                double inhat          = 0.;
                double newRunMean     = 0.;
                double adjust         = 0.;

                std::size_t base_idx =
                    cidx * strides[1] + didx * strides[2] + row * strides[3] + col * strides[4];
                std::size_t d_base_idx =
                    cidx * dstrides[1] + didx * dstrides[2] + row * dstrides[3] + col * dstrides[4];

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    mean_accum += input.data[bidx * strides[0] + base_idx];
                }
                mean_accum /= n;

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    elemStd = (input.data[bidx * strides[0] + base_idx] - mean_accum);
                    variance_accum += elemStd * elemStd;
                }
                variance_accum /= n;

                elemInvVar = 1.0 / double(sqrt(variance_accum + epsilon));

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    elemStd = (input.data[bidx * strides[0] + base_idx] - mean_accum);
                    inhat   = elemStd * elemInvVar;
                    out.data[bidx * strides[0] + base_idx] =
                        scale.data[d_base_idx] * inhat + shift.data[d_base_idx];
                }

                newRunMean = runMean.data[d_base_idx] * (1.0 - expAvgFactor);
                runMean.data[d_base_idx] =
                    mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp

                adjust = (n_batch == 1) ? variance_accum : (n / (n - 1.0)) * variance_accum;
                runVar.data[d_base_idx] =
                    (1 - expAvgFactor) * runVar.data[d_base_idx] + expAvgFactor * adjust;

                saveMean.data[d_base_idx]   = mean_accum;
                saveInvVar.data[d_base_idx] = elemInvVar;
            });
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train_3d_bn_per_activation pass time: "
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

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        std::size_t rs_n_batch, rs_channels, rs_depth, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);

        std::tie(rs_n_batch, rs_channels, rs_depth, rs_height, rs_width) =
            miopen::tien<5>(derivedBnDesc.GetLengths());

        tensor<U> runMean;
        tensor<U> runVar;

        if(input.desc.GetType() == miopenFloat)
        {
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
            runVar = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
        }
        else
        {
            prng::reset_seed();
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};
            runVar  = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};

            const double Data_scale = 0.001;
            for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
            {
                runMean[i] = prng::gen_descreet_uniform_sign<U>(Data_scale, 100);
                runVar[i]  = prng::gen_descreet_unsigned<U>(Data_scale, 100);
            }
        }

        auto saveMean   = tensor<U>{1, channels, depth, height, width};
        auto saveInvVar = tensor<U>{1, channels, depth, height, width};

        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<U>(channels * depth * height * width);
        auto saveInvVar_dev = handle.Create<U>(channels * depth * height * width);
        auto out_dev        = handle.Create<T>(n_batch * depth * channels * height * width);

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        float alpha = 1.;
        float beta  = 0.;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardTraining(handle,
                                         miopenBNPerActivation,
                                         &alpha,
                                         &beta,
                                         BuildReshaped4DTensorDescriptor(input.desc),
                                         in_dev.get(),
                                         BuildReshaped4DTensorDescriptor(out.desc),
                                         out_dev.get(),
                                         BuildReshaped4DTensorDescriptor(scale.desc),
                                         BuildReshaped4DTensorDescriptor(shift.desc),
                                         BuildReshaped4DTensorDescriptor(shift.desc),
                                         BuildReshaped4DTensorDescriptor(shift.desc),
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

        std::cout << "Wall clock: GPU forward_train_3d_bn_per_activation pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    void fail(int badtensor) const
    {
        std::cout << "Forward Train Per Activation 3D Batch Normalization: " << std::endl;
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
struct verify_forward_infer_3d_bn_per_activation_recalc
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

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(out.begin(), out.end(), 0);

        const auto n = static_cast<double>(n_batch);

        const auto& strides  = input.desc.GetStrides();
        const auto& dstrides = scale.desc.GetStrides();

        miopen::par_for(channels, 1, [&](int cidx) {
            miopen::ford(depth, height, width)([&](int didx, int row, int col) {
                double mean_accum     = 0.;
                double variance_accum = 0.;
                double elemStd        = 0.;
                double elemInvVar     = 0.;
                double inhat          = 0.;

                std::size_t base_idx =
                    cidx * strides[1] + didx * strides[2] + row * strides[3] + col * strides[4];
                std::size_t d_base_idx =
                    cidx * dstrides[1] + didx * dstrides[2] + row * dstrides[3] + col * dstrides[4];

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    mean_accum += input.data[bidx * strides[0] + base_idx];
                }
                mean_accum /= n;

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    elemStd = input.data[bidx * strides[0] + base_idx] - mean_accum;
                    variance_accum += elemStd * elemStd;
                }
                variance_accum /= n;

                elemInvVar = 1.0 / double(sqrt(variance_accum + epsilon));

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    elemStd = input.data[bidx * strides[0] + base_idx] - mean_accum;
                    inhat   = elemStd * elemInvVar;
                    out.data[bidx * strides[0] + base_idx] =
                        scale.data[d_base_idx] * inhat + shift.data[d_base_idx];
                }
            });
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_3d_bn_per_activation_recalc pass time: "
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
        float alpha    = 1.;
        float beta     = 0.;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardInference(handle,
                                          miopenBNPerActivation,
                                          &alpha,
                                          &beta,
                                          BuildReshaped4DTensorDescriptor(input.desc),
                                          in_dev.get(),
                                          BuildReshaped4DTensorDescriptor(out.desc),
                                          out_dev.get(),
                                          BuildReshaped4DTensorDescriptor(scale.desc),
                                          BuildReshaped4DTensorDescriptor(shift.desc),
                                          BuildReshaped4DTensorDescriptor(shift.desc),
                                          BuildReshaped4DTensorDescriptor(shift.desc),
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          nullptr,
                                          nullptr,
                                          epsilon,
                                          actDesc);
        out.data = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_3d_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int) const
    {
        std::cout << "Forward Inference Per Activation 3D Batch Normalization Recalc: "
                  << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T, class U>
struct verify_forward_infer_3d_bn_per_activation_use_est
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

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(input.desc.GetLengths());

        auto out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(out.begin(), out.end(), 0);

        const auto& strides  = input.desc.GetStrides();
        const auto& dstrides = scale.desc.GetStrides();

        miopen::par_for(channels, 1, [&](int cidx) {
            miopen::ford(depth, height, width)([&](int didx, int row, int col) {
                std::size_t base_idx =
                    cidx * strides[1] + didx * strides[2] + row * strides[3] + col * strides[4];
                std::size_t d_base_idx =
                    cidx * dstrides[1] + didx * dstrides[2] + row * dstrides[3] + col * dstrides[4];

                double mean       = estMean.data[d_base_idx];
                double variance   = estVar.data[d_base_idx];
                double elemInvVar = 1.0 / double(sqrt(variance + epsilon));

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    double elemStd = input.data[bidx * strides[0] + base_idx] - mean;
                    double inhat   = elemStd * elemInvVar;
                    out.data[bidx * strides[0] + base_idx] =
                        scale.data[d_base_idx] * inhat + shift.data[d_base_idx];
                }
            });
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_3d_bn_per_activation_use_est pass time: "
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
        float alpha    = 1.;
        float beta     = 0.;

        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        miopen::BatchNormForwardInference(handle,
                                          miopenBNPerActivation,
                                          &alpha,
                                          &beta,
                                          BuildReshaped4DTensorDescriptor(input.desc),
                                          in_dev.get(),
                                          BuildReshaped4DTensorDescriptor(out.desc),
                                          out_dev.get(),
                                          BuildReshaped4DTensorDescriptor(scale.desc),
                                          BuildReshaped4DTensorDescriptor(shift.desc),
                                          BuildReshaped4DTensorDescriptor(shift.desc),
                                          BuildReshaped4DTensorDescriptor(shift.desc),
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          estMean_dev.get(),
                                          estVar_dev.get(),
                                          epsilon,
                                          actDesc);
        out.data = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_3d_bn_per_activation_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int) const
    {
        std::cout << "Forward Inference Per Activation 3D Batch Normalization Use Estimated: "
                  << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

//****************************************************
// BACKWARDS PROPAGATION
//****************************************************
template <class T, class U>
struct verify_backward_3d_bn_per_activation_use_saved
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
        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{1, channels, depth, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{1, channels, depth, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const auto n = static_cast<double>(n_batch);

        const auto& strides  = x_input.desc.GetStrides();
        const auto& dstrides = scale.desc.GetStrides();

        miopen::par_for(channels, 1, [&](int cidx) {
            miopen::ford(depth, height, width)([&](int didx, int row, int col) {
                std::size_t base_idx =
                    cidx * strides[1] + didx * strides[2] + row * strides[3] + col * strides[4];
                std::size_t d_base_idx =
                    cidx * dstrides[1] + didx * dstrides[2] + row * dstrides[3] + col * dstrides[4];

                double dxhat    = 0.;
                double dxhathat = 0.;

                double mean       = savedMean.data[d_base_idx];
                double elemInvVar = savedInvVar.data[d_base_idx];

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    double elemStd = x_input.data[bidx * strides[0] + base_idx] - mean;
                    double xhat    = elemStd * elemInvVar;
                    double dyelem  = dy_input.data[bidx * strides[0] + base_idx];
                    dshift.data[d_base_idx] += dyelem;
                    dscale.data[d_base_idx] += xhat * dyelem;
                    double tmp1 = scale.data[d_base_idx] * dyelem;
                    dxhat += tmp1;
                    dxhathat += tmp1 * xhat;
                }

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    double elemStd = x_input.data[bidx * strides[0] + base_idx] - mean;
                    double xhat    = elemStd * elemInvVar;
                    double tmp1    = xhat * dxhathat + dxhat;
                    double tmp2    = n_batch * (scale.data[d_base_idx] *
                                             dy_input.data[bidx * strides[0] + base_idx]) -
                                  tmp1;
                    double tmp3                               = elemInvVar / n;
                    dx_out.data[bidx * strides[0] + base_idx] = tmp3 * tmp2;
                }
            });
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_3d_bn_per_activation_use_saved pass time: "
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

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{1, channels, depth, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{1, channels, depth, height, width};
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
                                  BuildReshaped4DTensorDescriptor(x_input.desc),
                                  xin_dev.get(),
                                  BuildReshaped4DTensorDescriptor(dy_input.desc),
                                  dyin_dev.get(),
                                  BuildReshaped4DTensorDescriptor(dx_out.desc),
                                  dx_out_dev.get(),
                                  BuildReshaped4DTensorDescriptor(scale.desc),
                                  BuildReshaped4DTensorDescriptor(dshift.desc),
                                  BuildReshaped4DTensorDescriptor(dshift.desc),
                                  BuildReshaped4DTensorDescriptor(dshift.desc),
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

        std::cout << "Wall clock: GPU backward_3d_bn_per_activation_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor) const
    {
        std::cout << "Backward 3D Batch Per ActivationNormalization Using Saved Mean and Variance: "
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
struct verify_backward_3d_bn_per_activation_recalc
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

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{1, channels, depth, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{1, channels, depth, height, width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const auto n = static_cast<double>(n_batch);

        const auto& strides  = x_input.desc.GetStrides();
        const auto& dstrides = scale.desc.GetStrides();

        miopen::par_for(channels, 1, [&](int cidx) {
            miopen::ford(depth, height, width)([&](int didx, int row, int col) {
                std::size_t base_idx =
                    cidx * strides[1] + didx * strides[2] + row * strides[3] + col * strides[4];
                std::size_t d_base_idx =
                    cidx * dstrides[1] + didx * dstrides[2] + row * dstrides[3] + col * dstrides[4];

                double mean = 0.;
                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    mean += x_input.data[bidx * strides[0] + base_idx];
                }
                mean /= n;

                double variance = 0.;
                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    double elemStd = x_input.data[bidx * strides[0] + base_idx] - mean;
                    variance += elemStd * elemStd;
                }
                variance /= n;

                double elemInvVar = 1.0 / double(sqrt(variance + epsilon));

                double dxhat    = 0.;
                double dxhathat = 0.;

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    double elemStd = x_input.data[bidx * strides[0] + base_idx] - mean;
                    double xhat    = elemStd * elemInvVar;
                    double dyelem  = dy_input.data[bidx * strides[0] + base_idx];
                    dshift.data[d_base_idx] += dyelem;
                    dscale.data[d_base_idx] += xhat * dyelem;
                    double tmp1 = scale.data[d_base_idx] * dyelem;
                    dxhat += tmp1;
                    dxhathat += tmp1 * xhat;
                }

                for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                {
                    double elemStd = x_input.data[bidx * strides[0] + base_idx] - mean;
                    double xhat    = elemStd * elemInvVar;
                    double tmp1    = xhat * dxhathat + dxhat;
                    double tmp2    = n_batch * (scale.data[d_base_idx] *
                                             dy_input.data[bidx * strides[0] + base_idx]) -
                                  tmp1;
                    double tmp3                               = elemInvVar / n;
                    dx_out.data[bidx * strides[0] + base_idx] = tmp3 * tmp2;
                }
            });
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_3d_bn_per_activation_recalc pass time: "
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

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, depth, height, width};
        // std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{1, channels, depth, height, width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{1, channels, depth, height, width};
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
                                  BuildReshaped4DTensorDescriptor(x_input.desc),
                                  xin_dev.get(),
                                  BuildReshaped4DTensorDescriptor(dy_input.desc),
                                  dyin_dev.get(),
                                  BuildReshaped4DTensorDescriptor(dx_out.desc),
                                  dx_out_dev.get(),
                                  BuildReshaped4DTensorDescriptor(scale.desc),
                                  BuildReshaped4DTensorDescriptor(dshift.desc),
                                  BuildReshaped4DTensorDescriptor(dshift.desc),
                                  BuildReshaped4DTensorDescriptor(dshift.desc),
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

        std::cout << "Wall clock: GPU backward_3d_bn_per_activation_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor) const
    {
        std::cout << "Backward 3D Batch Per Activation Normalization Recalc Mean and Variance: "
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

inline auto GenSmokeTestCases()
{
    return testing::Values(
        NamedContainer<std::vector<int>>("dims", std::vector<int>{2, 2, 3, 4, 4}, "x"));
}

inline auto GetSmokeTestCases()
{
    static const auto cases = GenSmokeTestCases();
    return cases;
}

inline auto GenFullTestCases()
{
    auto inputs = get_3d_bn_peract_inputs(4);
    // Omit shapes with n = 1 as CTest does
    for(auto it = inputs.begin(); it != inputs.end();)
    {
        if((*it)[0] == 1)
            it = inputs.erase(it);
        else
            ++it;
    }
    return MakeNamedParameterCollectionValues<std::vector<int>>("dims", inputs, "x");
}

inline auto GetFullTestCases()
{
    static const auto cases = GenFullTestCases();
    return cases;
}

struct TestParameterNameGenerator
{
    std::string operator()(const testing::TestParamInfo<TestCase>& info) const
    {
        const auto& dims = info.param;
        std::stringstream ss;
        std::string str;

        ss << "dims_" << GetRangeAsString(dims(), "x") << "_test_id_" << info.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

template <typename T>
struct Bn3DPeractTest : public testing::TestWithParam<TestCase>
{
    static const constexpr uint64_t MaxValue{miopen_type<T>{} == miopenHalf ? 5 : 17};

    tensor<T> input;
    tensor<PREC_TYPE> scale;
    tensor<PREC_TYPE> shift;
    tensor<T> dy_input;

    void SetUp() override
    {
        prng::reset_seed();
        const auto dims = GetParam();

        std::size_t n, c, d, h, w;
        std::tie(n, c, d, h, w) = miopen::tien<5>(dims());

        input    = tensor<T>{dims()};
        dy_input = tensor<T>{dims()}.generate(tensor_elem_gen_integer{MaxValue});

        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);

        auto derived_lengths = derivedBnDesc.GetLengths();

        if(input.desc.GetType() == miopenFloat)
        {
            input.generate(tensor_elem_gen_integer{MaxValue});
            scale = tensor<PREC_TYPE>{derived_lengths}.generate(tensor_elem_gen_integer{17});
            shift = tensor<PREC_TYPE>{derived_lengths}.generate(tensor_elem_gen_integer{17});
        }
        else
        {
            scale = tensor<PREC_TYPE>{derived_lengths};
            shift = tensor<PREC_TYPE>{derived_lengths};

            const constexpr double Data_scale = 0.001;
            for(std::size_t i = 0; i < scale.desc.GetElementSize(); i++)
            {
                scale[i] = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
                shift[i] = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            }
            for(std::size_t i = 0; i < input.desc.GetElementSize(); i++)
            {
                input[i] = prng::gen_descreet_uniform_sign<T>(1e-4, 100);
            }
        }
    }

    void RunAll()
    {
        std::size_t n, c, d, h, w;
        std::tie(n, c, d, h, w) = miopen::tien<5>(input.desc.GetLengths());
        double tolerance        = 200 * input.desc.GetElementSize();

        // train
        const auto outpair_train =
            Verify(verify_forward_train_3d_bn_per_activation<T, PREC_TYPE>{input, scale, shift},
                   tolerance);

        // inference recalc
        (void)Verify(
            verify_forward_infer_3d_bn_per_activation_recalc<T, PREC_TYPE>{input, scale, shift},
            tolerance);

        // inference use estimated running values
        if(input.desc.GetType() == miopenFloat)
        {
            const auto& estMean = std::get<1>(outpair_train.second);
            const auto& estVar  = std::get<2>(outpair_train.second);
            (void)Verify(
                verify_forward_infer_3d_bn_per_activation_use_est<T, PREC_TYPE>{
                    input, scale, shift, estMean, estVar},
                tolerance);
        }

        // backprop recalc
        (void)Verify(
            verify_backward_3d_bn_per_activation_recalc<T, PREC_TYPE>{input, dy_input, scale},
            8000 * input.desc.GetElementSize());

        // backprop use saved values
        const auto& savedMean   = std::get<3>(outpair_train.second);
        const auto& savedInvVar = std::get<4>(outpair_train.second);
        (void)Verify(
            verify_backward_3d_bn_per_activation_use_saved<T, PREC_TYPE>{
                input, dy_input, scale, savedMean, savedInvVar},
            8000 * input.desc.GetElementSize());
    }

    auto Verify(auto&& v, double tolerance)
    {
        auto res = std::make_pair(v.cpu(), v.gpu());
        Compare(v, res.first, res.second, tolerance);

        return std::move(res);
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
                    using value_type       = miopen::range_value<decltype(g)>;
                    const double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
                    const double error     = miopen::rms_range(c, g);
                    EXPECT_LE(error, threshold);
                    if(error > threshold)
                        v.fail(i);
                },
                is...);
        })(std::integral_constant<std::size_t, sizeof...(CpuRanges)>{});
    }

    template <typename CpuRanges, typename GpuRanges>
    void Compare(auto&& v, const CpuRanges& cpu, const GpuRanges& gpu, double tolerance)
    {
        ASSERT_EQ(miopen::range_distance(cpu), miopen::range_distance(gpu));
        using value_type       = miopen::range_value<decltype(gpu)>;
        const double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
        const double error     = miopen::rms_range(cpu, gpu);
        EXPECT_LE(error, threshold);
        if(error > threshold)
            v.fail(0);
    }
};

} // namespace

using GPU_Bn3DPeract_FP16  = Bn3DPeractTest<half_float::half>;
using GPU_Bn3DPeract_FP32  = Bn3DPeractTest<float>;
using GPU_Bn3DPeract_BFP16 = Bn3DPeractTest<bfloat16>;

TEST_P(GPU_Bn3DPeract_FP16, AllModes) { this->RunAll(); }
TEST_P(GPU_Bn3DPeract_FP32, AllModes) { this->RunAll(); }
TEST_P(GPU_Bn3DPeract_BFP16, AllModes) { this->RunAll(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Bn3DPeract_FP16,
                         GetSmokeTestCases(),
                         TestParameterNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Bn3DPeract_FP32,
                         GetSmokeTestCases(),
                         TestParameterNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Bn3DPeract_BFP16,
                         GetSmokeTestCases(),
                         TestParameterNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3DPeract_FP16,
                         GetFullTestCases(),
                         TestParameterNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3DPeract_FP32,
                         GetFullTestCases(),
                         TestParameterNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3DPeract_BFP16,
                         GetFullTestCases(),
                         TestParameterNameGenerator{});
