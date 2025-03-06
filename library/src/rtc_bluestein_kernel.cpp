// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "rtc_bluestein_kernel.h"
#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "../../shared/precision_type.h"
#include "function_pool.h"
#include "kernel_launch.h"
#include "rtc_bluestein_gen.h"
#include "tree_node.h"

RTCKernel::RTCGenerator RTCKernelBluesteinSingle::generate_from_node(const LeafNode&    node,
                                                                     const std::string& gpu_arch,
                                                                     bool enable_callbacks)
{
    RTCGenerator generator;

    if(node.scheme != CS_KERNEL_BLUESTEIN_SINGLE)
        return generator;

    auto lengthBlue = node.lengthBlue;

    // find kernel config from function pool
    auto config = node.pool.get_kernel(FMKey(lengthBlue, node.precision));

    // get factors from the leaf node, which might have overridden what's in the pool
    auto&                     leafNode = static_cast<const LeafNode&>(node);
    std::vector<unsigned int> factors;
    std::copy(
        leafNode.kernelFactors.begin(), leafNode.kernelFactors.end(), std::back_inserter(factors));

    // allocate enough blocks for all higher dims + batch
    unsigned int batch_accum = product(node.length.begin() + 1, node.length.end()) * node.batch;
    auto         bwd         = config.transforms_per_block;
    generator.gridDim        = {DivRoundingUp(batch_accum, bwd)};
    generator.blockDim       = config.workgroup_size;

    BluesteinSingleSpecs specs{static_cast<unsigned int>(node.length[0]),
                               static_cast<unsigned int>(node.length.size()),
                               factors,
                               static_cast<unsigned int>(config.threads_per_transform[0])
                                   * config.transforms_per_block,
                               static_cast<unsigned int>(config.threads_per_transform[0]),
                               node.direction,
                               node.precision,
                               node.placement,
                               node.inArrayType,
                               node.outArrayType,
                               node.GetCallbackType(enable_callbacks),
                               node.loadOps,
                               node.storeOps};

    generator.generate_name = [=]() { return bluestein_single_rtc_kernel_name(specs); };

    generator.generate_src
        = [=](const std::string& kernel_name) { return bluestein_single_rtc(kernel_name, specs); };

    generator.construct_rtckernel = [=](const std::string&       kernel_name,
                                        const std::vector<char>& code,
                                        dim3                     gridDim,
                                        dim3                     blockDim) {
        return std::unique_ptr<RTCKernel>(
            new RTCKernelBluesteinSingle(kernel_name, code, gridDim, blockDim));
    };

    return generator;
}

RTCKernelArgs RTCKernelBluesteinSingle::get_launch_args(DeviceCallIn& data)
{
    RTCKernelArgs kargs;
    kargs.append_ptr(data.bufTemp);
    kargs.append_ptr(data.node->twiddles);
    kargs.append_ptr(kargs_lengths(data.node->devKernArg));
    kargs.append_ptr(kargs_stride_in(data.node->devKernArg));
    if(data.node->placement == rocfft_placement_notinplace)
    {
        kargs.append_ptr(kargs_stride_out(data.node->devKernArg));
    }
    kargs.append_size_t(data.node->batch);
    kargs.append_ptr(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.append_ptr(data.bufIn[1]);

    if(data.node->placement == rocfft_placement_notinplace)
    {
        kargs.append_ptr(data.bufOut[0]);
        if(array_type_is_planar(data.node->outArrayType))
            kargs.append_ptr(data.bufOut[1]);
    }

    // callback params
    kargs.append_ptr(data.callbacks.load_cb_fn);
    kargs.append_ptr(data.callbacks.load_cb_data);
    kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
    kargs.append_ptr(data.callbacks.store_cb_fn);
    kargs.append_ptr(data.callbacks.store_cb_data);

    append_load_store_args(kargs, *data.node);
    return kargs;
}

RTCKernel::RTCGenerator RTCKernelBluesteinMulti::generate_from_node(const LeafNode&    node,
                                                                    const std::string& gpu_arch,
                                                                    bool enable_callbacks)
{
    RTCGenerator generator;

    auto scheme = node.scheme;

    if(scheme != CS_KERNEL_CHIRP && node.scheme != CS_KERNEL_PAD_MUL
       && node.scheme != CS_KERNEL_FFT_MUL && node.scheme != CS_KERNEL_RES_MUL)
        return generator;

    size_t N = node.length[0];
    size_t M = node.lengthBlue;

    size_t numof = 0;
    if(scheme == CS_KERNEL_FFT_MUL)
    {
        numof = M;
    }
    else if(scheme == CS_KERNEL_PAD_MUL)
    {
        numof = M;
    }
    else
    {
        // CS_KERNEL_RES_MUL
        numof = N;
    }

    size_t count = node.batch;
    for(size_t i = 1; i < node.length.size(); i++)
        count *= node.length[i];
    count *= numof;

    if(scheme == CS_KERNEL_CHIRP)
    {
        generator.gridDim
            = {static_cast<unsigned int>((M - N) / LAUNCH_BOUNDS_BLUESTEIN_MULTI_KERNEL + 1)};
        generator.blockDim = {LAUNCH_BOUNDS_BLUESTEIN_MULTI_KERNEL};
    }
    else
    {
        generator.gridDim
            = {(static_cast<unsigned int>(count) - 1) / LAUNCH_BOUNDS_BLUESTEIN_MULTI_KERNEL + 1};
        generator.blockDim = {LAUNCH_BOUNDS_BLUESTEIN_MULTI_KERNEL};
    }

    BluesteinMultiSpecs specs{scheme,
                              node.precision,
                              node.inArrayType,
                              node.outArrayType,
                              node.GetCallbackType(enable_callbacks),
                              node.loadOps,
                              node.storeOps};

    generator.generate_name = [=]() { return bluestein_multi_rtc_kernel_name(specs); };

    generator.generate_src
        = [=](const std::string& kernel_name) { return bluestein_multi_rtc(kernel_name, specs); };

    generator.construct_rtckernel = [=](const std::string&       kernel_name,
                                        const std::vector<char>& code,
                                        dim3                     gridDim,
                                        dim3                     blockDim) {
        return std::unique_ptr<RTCKernel>(new RTCKernelBluesteinMulti(
            kernel_name, scheme, N, M, numof, count, code, gridDim, blockDim));
    };

    return generator;
}

RTCKernelArgs RTCKernelBluesteinMulti::get_launch_args(DeviceCallIn& data)
{
    RTCKernelArgs kargs;

    if(scheme == CS_KERNEL_CHIRP)
    {
        int twl = 0;

        if(data.node->large1D > (size_t)256 * 256 * 256 * 256)
            throw std::runtime_error("large1D twiddle size too large error");
        else if(data.node->large1D > (size_t)256 * 256 * 256)
            twl = 4;
        else if(data.node->large1D > (size_t)256 * 256)
            twl = 3;
        // TODO- possibly using a smaller LargeTwdBase for chirp by large_twiddle_base
        else if(data.node->large1D > (size_t)256)
            twl = 2;
        else
            twl = 1;

        kargs.append_size_t(N);
        kargs.append_size_t(M);
        kargs.append_ptr(data.bufOut[0]);
        kargs.append_ptr(data.node->twiddles_large);
        kargs.append_int(twl);
        kargs.append_int(data.node->direction);
    }
    else
    {
        const size_t cBytes = complex_type_size(data.node->precision);

        void* bufIn0  = data.bufIn[0];
        void* bufOut0 = data.bufOut[0];
        void* bufIn1  = data.bufIn[1];
        void* bufOut1 = data.bufOut[1];

        if(scheme == CS_KERNEL_FFT_MUL)
        {
            bufIn0  = static_cast<char*>(bufIn0) + M * cBytes;
            bufOut0 = static_cast<char*>(bufOut0) + 2 * M * cBytes;
        }
        else if(scheme == CS_KERNEL_PAD_MUL)
        {
            bufOut0 = static_cast<char*>(bufOut0) + M * cBytes;
        }

        kargs.append_size_t(numof);
        kargs.append_size_t(count);
        kargs.append_size_t(N);
        kargs.append_size_t(M);
        kargs.append_ptr(bufIn0);
        if(array_type_is_planar(data.node->inArrayType))
            kargs.append_ptr(bufIn1);
        kargs.append_ptr(bufOut0);
        if(array_type_is_planar(data.node->outArrayType))
            kargs.append_ptr(bufOut1);
        kargs.append_size_t(data.node->length.size());
        kargs.append_ptr(kargs_lengths(data.node->devKernArg));
        kargs.append_ptr(kargs_stride_in(data.node->devKernArg));
        kargs.append_ptr(kargs_stride_out(data.node->devKernArg));
        // callback params
        kargs.append_ptr(data.callbacks.load_cb_fn);
        kargs.append_ptr(data.callbacks.load_cb_data);
        kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
        kargs.append_ptr(data.callbacks.store_cb_fn);
        kargs.append_ptr(data.callbacks.store_cb_data);

        append_load_store_args(kargs, *data.node);
    }
    return kargs;
}
