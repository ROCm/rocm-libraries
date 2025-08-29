// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <optional>

#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "function_pool.h"
#include "kernel_launch.h"
#include "rtc_stockham_gen.h"
#include "rtc_stockham_kernel.h"
#include "tree_node.h"

#include "device/kernel-generator-embed.h"

RTCKernel::RTCGenerator RTCKernelStockham::generate_from_node(const LeafNode&    node,
                                                              const std::string& gpu_arch,
                                                              bool               enable_callbacks)
{
    RTCStockhamGenerator generator;

    std::optional<StockhamGeneratorSpecs> specs;
    std::optional<StockhamGeneratorSpecs> specs2d;
    StockhamPartialPassParams             pp_params;

    // SBRC variants look in the function pool for plain BLOCK_RC to
    // learn the block width, then decide on the transpose type once
    // that's known.
    auto         pool_scheme = node.scheme;
    unsigned int static_dim  = node.GetStaticDim();
    if(pool_scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
       || pool_scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
       || pool_scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
    {
        pool_scheme = CS_KERNEL_STOCKHAM_BLOCK_RC;
    }

    std::optional<FFTKernel> kernel;

    switch(pool_scheme)
    {
    case CS_KERNEL_STOCKHAM:
    case CS_KERNEL_STOCKHAM_PP:
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
    case CS_KERNEL_STOCKHAM_PP_BLOCK_CC:
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
    {
        // for sbrc variant, the sbrcTranstype should be assigned when we are here
        // since the value is assigned in KernelCheck()
        if((pool_scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) && (node.sbrcTranstype == NONE))
            throw std::runtime_error("Invalid SBRC_TRANS_TYPE for SBRC kernel");

        // these go into the function pool normally and are passed to
        // the generator as-is
        kernel = node.GetKernel();

        std::vector<unsigned int> factors;
        std::copy(kernel->factors.begin(), kernel->factors.end(), std::back_inserter(factors));
        std::vector<unsigned int> precisions = {static_cast<unsigned int>(node.precision)};

        specs.emplace(factors,
                      std::vector<unsigned int>(),
                      precisions,
                      static_cast<unsigned int>(kernel->workgroup_size),
                      PrintScheme(node.scheme));
        specs->threads_per_transform = kernel->threads_per_transform[0];
        specs->half_lds              = kernel->half_lds;
        specs->direct_to_from_reg    = kernel->direct_to_from_reg;
        specs->ebtype                = node.ebtype;

        if(node.isPartialPassEnabled())
        {
            pp_params.off_dim     = node.ppOffDim;
            pp_params.current_dim = node.ppCurrDim;
            pp_params.pp_factors_curr.assign(kernel->pp_params.pp_factors_curr.begin(),
                                             kernel->pp_params.pp_factors_curr.end());
            pp_params.pp_factors_other.assign(kernel->pp_params.pp_factors_other.begin(),
                                              kernel->pp_params.pp_factors_other.end());
            pp_params.parent_length.assign(node.length.begin(), node.length.end());
        }

        break;
    }
    case CS_KERNEL_2D_SINGLE:
    {
        kernel = node.GetKernel();

        std::vector<unsigned int> factors1d;
        std::vector<unsigned int> factors2d;
        std::vector<unsigned int> precisions = {static_cast<unsigned int>(node.precision)};

        // need to break down factors into first dim and second dim
        size_t len0_remain = node.length[0];
        for(auto& f : kernel->factors)
        {
            len0_remain /= f;
            if(len0_remain > 0)
            {
                factors1d.push_back(f);
            }
            else
            {
                factors2d.push_back(f);
            }
        }

        specs.emplace(factors1d,
                      factors2d,
                      precisions,
                      static_cast<unsigned int>(kernel->workgroup_size),
                      PrintScheme(node.scheme));
        specs->threads_per_transform = kernel->threads_per_transform[0];
        specs->half_lds              = kernel->half_lds;
        specs->ebtype                = node.ebtype;

        specs2d.emplace(factors2d,
                        factors1d,
                        precisions,
                        static_cast<unsigned int>(kernel->workgroup_size),
                        PrintScheme(node.scheme));
        specs2d->threads_per_transform = kernel->threads_per_transform[1];
        specs2d->half_lds              = kernel->half_lds;
        break;
    }
    default:
    {
        // no supported scheme, not the correct type
        return generator;
    }
    }

    // static_dim has what the plan requires, and RTC kernels are
    // built for exactly that dimension.  But kernels that are
    // compiled ahead of time are general across all dims and take
    // 'dim' as an argument.  So set static_dim to 0 to communicate
    // this to the generator and launch machinery.
    if(kernel && kernel->aot_rtc)
        static_dim = 0;
    specs->static_dim = static_dim;

    // mark wgs as derived already so generator won't change it again
    specs->wgs_is_derived = true;
    if(specs2d)
        specs2d->wgs_is_derived = true;

    bool unit_stride = node.inStride.front() == 1 && node.outStride.front() == 1;

    auto ppType = PartialPassType::PPT_NONE;
    if(node.isPartialPassEnabled())
    {
        if(node.scheme == CS_KERNEL_STOCKHAM_PP_BLOCK_CC)
            ppType = PartialPassType::PPT_SBCC;
        else if(node.scheme == CS_KERNEL_STOCKHAM_PP)
            ppType = PartialPassType::PPT_SBRR;
        else
            throw std::runtime_error("Invalid scheme for partial pass");
    }

    generator.generate_name = [=, &node]() {
        return stockham_rtc_kernel_name(*specs,
                                        specs2d ? *specs2d : *specs,
                                        node.scheme,
                                        node.direction,
                                        node.precision,
                                        node.placement,
                                        node.inArrayType,
                                        node.outArrayType,
                                        unit_stride,
                                        node.largeTwdBase,
                                        node.ltwdSteps,
                                        node.largeTwdBatchIsTransformCount,
                                        node.dir2regMode,
                                        node.intrinsicMode,
                                        node.sbrcTranstype,
                                        node.GetCallbackType(enable_callbacks),
                                        node.fuseBlue,
                                        ppType,
                                        pp_params,
                                        node.loadOps,
                                        node.storeOps);
    };

    generator.generate_src = [=, &node](const std::string& kernel_name) {
        return stockham_rtc(*specs,
                            specs2d ? *specs2d : *specs,
                            pp_params,
                            nullptr,
                            kernel_name,
                            node.scheme,
                            node.direction,
                            node.precision,
                            node.placement,
                            node.inArrayType,
                            node.outArrayType,
                            unit_stride,
                            node.largeTwdBase,
                            node.ltwdSteps,
                            node.largeTwdBatchIsTransformCount,
                            node.dir2regMode,
                            node.intrinsicMode,
                            node.sbrcTranstype,
                            node.GetCallbackType(enable_callbacks),
                            node.fuseBlue,
                            ppType,
                            node.loadOps,
                            node.storeOps);
    };

    generator.construct_rtckernel
        = [](const std::string& kernel_name, const std::vector<char>& code, dim3, dim3) {
              return std::unique_ptr<RTCKernel>(new RTCKernelStockham(kernel_name, code));
          };
    return generator;
}

RTCKernelArgs RTCKernelStockham::get_launch_args(DeviceCallIn& data)
{
    // construct arguments to pass to the kernel
    RTCKernelArgs kargs;

    // twiddles
    if(data.node->scheme == CS_KERNEL_STOCKHAM_PP)
        kargs.append_ptr(data.node->twiddles_pp);
    kargs.append_ptr(data.node->twiddles);
    // large 1D twiddles
    if(data.node->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC
       || data.node->scheme == CS_KERNEL_STOCKHAM_PP_BLOCK_CC)
        kargs.append_ptr(data.node->twiddles_large);
    if(!hardcoded_dim)
        kargs.append_size_t(data.node->length.size());
    // lengths
    kargs.append_ptr(kargs_lengths(data.node->devKernArg));
    // stride in/out
    kargs.append_ptr(kargs_stride_in(data.node->devKernArg));
    if(data.node->placement == rocfft_placement_notinplace)
        kargs.append_ptr(kargs_stride_out(data.node->devKernArg));
    // nbatch
    kargs.append_size_t(data.node->batch);
    // callback params
    kargs.append_ptr(data.callbacks.load_cb_fn);
    kargs.append_ptr(data.callbacks.load_cb_data);
    kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
    kargs.append_ptr(data.callbacks.store_cb_fn);
    kargs.append_ptr(data.callbacks.store_cb_data);

    // buffer pointers
    kargs.append_ptr(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.append_ptr(data.bufIn[1]);
    if(data.node->placement == rocfft_placement_notinplace)
    {
        kargs.append_ptr(data.bufOut[0]);
        if(array_type_is_planar(data.node->outArrayType))
            kargs.append_ptr(data.bufOut[1]);
    }

    append_load_store_args(kargs, *data.node);

    // fused bluestein data (chirp table and lengths)
    switch(data.node->fuseBlue)
    {
    case BFT_NONE:
        break;
    case BFT_FWD_CHIRP:
    case BFT_FWD_CHIRP_MUL:
        if(data.node->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
            kargs.append_ptr(data.node->chirp);

        kargs.append_size_t(data.node->lengthBlueN);
        kargs.append_size_t(data.node->lengthBlue);

        break;
    case BFT_INV_CHIRP_MUL:
        if(data.node->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
            kargs.append_ptr(data.node->chirp);

        kargs.append_size_t(data.node->lengthBlueN);
        kargs.append_size_t(data.node->lengthBlue);

        break;
    }
    // fused bluestein data (strides and dists)
    if(data.node->fuseBlue != BFT_NONE)
    {
        size_t empty_val = 0;

        if(data.node->fuseBlue == BFT_FWD_CHIRP)
        {
            kargs.append_size_t(empty_val);
            kargs.append_size_t(empty_val);
            kargs.append_size_t(empty_val);

            kargs.append_size_t(empty_val);
            kargs.append_size_t(empty_val);
            kargs.append_size_t(empty_val);
        }
        else
        {
            assert(data.node->inStrideBlue.size() == data.node->outStrideBlue.size());
            switch(data.node->inStrideBlue.size())
            {
            case 2: // 1D FFT
                kargs.append_size_t(empty_val);
                kargs.append_size_t(empty_val);
                kargs.append_size_t(data.node->iDistBlue);

                kargs.append_size_t(empty_val);
                kargs.append_size_t(empty_val);
                kargs.append_size_t(data.node->oDistBlue);
                break;
            case 3: // 2D FFT
                kargs.append_size_t(data.node->inStrideBlue[2]);
                kargs.append_size_t(empty_val);
                kargs.append_size_t(data.node->iDistBlue);

                kargs.append_size_t(data.node->outStrideBlue[2]);
                kargs.append_size_t(empty_val);
                kargs.append_size_t(data.node->oDistBlue);
                break;
            case 4: // 3D FFT
                kargs.append_size_t(data.node->inStrideBlue[2]);
                kargs.append_size_t(data.node->inStrideBlue[3]);
                kargs.append_size_t(data.node->iDistBlue);

                kargs.append_size_t(data.node->outStrideBlue[2]);
                kargs.append_size_t(data.node->outStrideBlue[3]);
                kargs.append_size_t(data.node->oDistBlue);
                break;
            default:
                throw std::runtime_error("Invalid strides for Bluestein kernel");
            }
        }
    }
    return kargs;
}
