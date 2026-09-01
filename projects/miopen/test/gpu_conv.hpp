/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef GUARD_GPU_CONV_HPP
#define GUARD_GPU_CONV_HPP

#include <miopen/env.hpp>
#include <miopen/gpu_conv_reference.hpp>

#include "get_handle.hpp"
#include "tensor_holder.hpp"

namespace env = miopen::env;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF)

template <typename Tin, typename Twei, typename Tout>
bool gpu_ref_convolution_fwd(const tensor<Tin>& input,
                             const tensor<Twei>& weights,
                             tensor<Tout>& rout,
                             miopen::ConvolutionDescriptor filter,
                             const miopen::Scalar& alpha = miopen::Scalar(1.0),
                             const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    if(env::enabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF))
        return false;

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(rout.data);

    miopen::GpuConvReference::RunFwd(handle,
                                     input.desc,
                                     in_dev.get(),
                                     weights.desc,
                                     wei_dev.get(),
                                     rout.desc,
                                     out_dev.get(),
                                     filter,
                                     alpha.GetAsDouble(),
                                     beta.GetAsDouble());

    rout.data = handle.Read<Tout>(out_dev, rout.data.size());
    return true;
}

template <typename Tin, typename Twei, typename Tout>
bool gpu_ref_convolution_bwd(tensor<Tin>& input,
                             const tensor<Twei>& weights,
                             const tensor<Tout> output,
                             miopen::ConvolutionDescriptor filter,
                             const miopen::Scalar& alpha = miopen::Scalar(1.0),
                             const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    if(env::enabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF))
        return false;

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(output.data);

    miopen::GpuConvReference::RunBwd(handle,
                                     output.desc,
                                     out_dev.get(),
                                     weights.desc,
                                     wei_dev.get(),
                                     input.desc,
                                     in_dev.get(),
                                     filter,
                                     alpha.GetAsDouble(),
                                     beta.GetAsDouble());

    input.data = handle.Read<Tin>(in_dev, input.data.size());
    return true;
}

template <typename Tin, typename Twei, typename Tout>
bool gpu_ref_convolution_wrw(const tensor<Tin>& input,
                             tensor<Twei>& weights,
                             const tensor<Tout> output,
                             miopen::ConvolutionDescriptor filter,
                             const miopen::Scalar& alpha = miopen::Scalar(1.0),
                             const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    if(env::enabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF))
        return false;

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(output.data);

    miopen::GpuConvReference::RunWrw(handle,
                                     output.desc,
                                     out_dev.get(),
                                     input.desc,
                                     in_dev.get(),
                                     weights.desc,
                                     wei_dev.get(),
                                     filter,
                                     alpha.GetAsDouble(),
                                     beta.GetAsDouble());

    weights.data = handle.Read<Twei>(wei_dev, weights.data.size());
    return true;
}

#endif // GUARD_GPU_CONV_HPP
