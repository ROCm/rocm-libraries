// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/type_convert.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// A CTA owns eight adjacent groups and one filter point. Admitted G>=128
// yields at least 144 CTAs for 3x3 filters. The other 32 lanes partition
// the reduction; with R<=min(G,512), each lane performs at most 16 products.
struct DepthwiseBwdWeightBf16Params
{
    const bhalf_t* in;
    bhalf_t* wei;
    const bhalf_t* out;
    std::array<long_index_t, 5> in_lengths, in_strides, wei_lengths, wei_strides, out_lengths,
        out_strides;
    std::array<long_index_t, 2> filter_strides, filter_dilations, left_pads, right_pads;
    index_t reduction_length;
};

struct DepthwiseBwdWeightBf16Argument : BaseArgument, ArgumentSplitK, DepthwiseBwdWeightBf16Params
{
    bool valid;
};

template <typename DataType>
__global__ void kernel_grouped_conv2d_bwd_weight_depthwise_bf16(DepthwiseBwdWeightBf16Params arg)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx125__)
    constexpr index_t GroupChunk     = 8;
    constexpr index_t ReductionLanes = 32;
    const index_t group_lane         = threadIdx.x % GroupChunk;
    const index_t reduction_lane     = threadIdx.x / GroupChunk;
    const long_index_t group = static_cast<long_index_t>(blockIdx.x) * GroupChunk + group_lane;
    const index_t filter_y   = blockIdx.y / arg.wei_lengths[4];
    const index_t filter_x   = blockIdx.y % arg.wei_lengths[4];
    const index_t out_h      = arg.out_lengths[3];
    const index_t out_w      = arg.out_lengths[4];
    const index_t output_image_size = out_h * out_w;

    float accum = 0;
    if(group < arg.in_lengths[0])
    {
        for(index_t r = reduction_lane; r < arg.reduction_length; r += ReductionLanes)
        {
            const index_t n       = r / output_image_size;
            const index_t ho      = (r % output_image_size) / out_w;
            const index_t wo      = r % out_w;
            const long_index_t hi = static_cast<long_index_t>(ho) * arg.filter_strides[0] +
                                    filter_y * arg.filter_dilations[0] - arg.left_pads[0];
            const long_index_t wi = static_cast<long_index_t>(wo) * arg.filter_strides[1] +
                                    filter_x * arg.filter_dilations[1] - arg.left_pads[1];
            if(hi >= 0 && hi < arg.in_lengths[3] && wi >= 0 && wi < arg.in_lengths[4])
            {
                const long_index_t in_offset = n * arg.in_strides[1] + group * arg.in_strides[0] +
                                               hi * arg.in_strides[3] + wi * arg.in_strides[4];
                const long_index_t out_offset = n * arg.out_strides[1] +
                                                group * arg.out_strides[0] +
                                                ho * arg.out_strides[3] + wo * arg.out_strides[4];
                accum += type_convert<float>(arg.in[in_offset]) *
                         type_convert<float>(arg.out[out_offset]);
            }
        }
    }

    // Interleaved group lanes give contiguous global loads and distinct FP32
    // shared-memory columns. One lane owns each final low-precision Set.
    __shared__ float partial[GroupChunk * ReductionLanes];
    partial[reduction_lane * GroupChunk + group_lane] = accum;
    __syncthreads();
    for(index_t step = ReductionLanes / 2; step > 0; step /= 2)
    {
        if(reduction_lane < step)
        {
            partial[reduction_lane * GroupChunk + group_lane] +=
                partial[(reduction_lane + step) * GroupChunk + group_lane];
        }
        __syncthreads();
    }
    if(reduction_lane == 0 && group < arg.in_lengths[0])
    {
        const long_index_t wei_offset = group * arg.wei_strides[0] + filter_y * arg.wei_strides[3] +
                                        filter_x * arg.wei_strides[4];
        arg.wei[wei_offset] = type_convert<DataType>(partial[group_lane]);
    }
#else
    ignore = arg;
#endif
}

struct DeviceGroupedConvBwdWeightDepthwiseBf16 final
    : DeviceGroupedConvBwdWeight<2,
                                 tensor_layout::convolution::NHWGC,
                                 tensor_layout::convolution::GKYXC,
                                 tensor_layout::convolution::NHWGK,
                                 bhalf_t,
                                 bhalf_t,
                                 bhalf_t,
                                 element_wise::PassThrough,
                                 element_wise::PassThrough,
                                 element_wise::PassThrough>
{
    using Argument = DepthwiseBwdWeightBf16Argument;

    static bool Supports(const Argument& a)
    {
        if(!is_gfx125_supported() || a.k_batch_ != 1)
            return false;

        constexpr auto MaxIndex = std::numeric_limits<index_t>::max();
        constexpr long_index_t MaxOffset =
            std::numeric_limits<long_index_t>::max() / static_cast<long_index_t>(sizeof(bhalf_t));
        const auto& il = a.in_lengths;
        const auto& wl = a.wei_lengths;
        const auto& ol = a.out_lengths;
        for(index_t i = 0; i < 5; ++i)
        {
            if(il[i] <= 0 || wl[i] <= 0 || ol[i] <= 0 || il[i] > MaxIndex || wl[i] > MaxIndex ||
               ol[i] > MaxIndex)
                return false;
        }
        if(il[0] < 128 || il[0] != ol[0] || il[0] != wl[0] || il[1] != ol[1] || il[2] != 1 ||
           wl[1] != 1 || wl[2] != 1 || ol[2] != 1 || wl[3] != 3 || wl[4] != 3)
            return false;

        // Checked before forming R, so invalid huge shapes cannot overflow.
        if(ol[1] > MaxIndex / ol[3] || ol[1] * ol[3] > MaxIndex / ol[4])
            return false;
        const auto r = ol[1] * ol[3] * ol[4];
        if(r > 512 || r > il[0])
            return false;

        for(index_t d = 0; d < 2; ++d)
        {
            if(a.filter_strides[d] < 1 || a.filter_strides[d] > 2 || a.filter_dilations[d] < 1 ||
               a.filter_dilations[d] > MaxIndex || a.left_pads[d] < 0 ||
               a.left_pads[d] > MaxIndex || a.right_pads[d] < 0 || a.right_pads[d] > MaxIndex)
                return false;
            const auto effective = (wl[d + 3] - 1) * a.filter_dilations[d] + 1;
            const auto padded    = il[d + 3] + a.left_pads[d] + a.right_pads[d];
            if(effective > padded || (padded - effective) / a.filter_strides[d] + 1 != ol[d + 3])
                return false;
        }

        // Packed NHWGC/NHWGK group axes ensure adjacent-lane input/dY reads;
        // padded GKYXC weight rows/groups are allowed if all stores are disjoint.
        const auto in_plane  = il[4] * il[0];
        const auto out_plane = ol[4] * ol[0];
        if(il[3] > MaxOffset / in_plane || ol[3] > MaxOffset / out_plane || a.wei_strides[4] < 1 ||
           a.wei_strides[3] < 1 || a.wei_strides[0] < 1 ||
           a.wei_strides[4] > MaxOffset / (3 * wl[4]) ||
           a.wei_strides[3] > MaxOffset / (3 * wl[3]) || a.wei_strides[0] > MaxOffset / (3 * wl[0]))
            return false;
        if(a.in_strides[0] != 1 || a.in_strides[2] != 1 || a.in_strides[4] != il[0] ||
           a.in_strides[3] != in_plane || a.in_strides[1] != il[3] * in_plane ||
           a.out_strides[0] != 1 || a.out_strides[2] != 1 || a.out_strides[4] != ol[0] ||
           a.out_strides[3] != out_plane || a.out_strides[1] != ol[3] * out_plane ||
           a.wei_strides[3] < wl[4] * a.wei_strides[4] ||
           a.wei_strides[0] <= (wl[3] - 1) * a.wei_strides[3] + (wl[4] - 1) * a.wei_strides[4])
            return false;
        return il[1] <= MaxOffset / a.in_strides[1] && ol[1] <= MaxOffset / a.out_strides[1];
    }

    template <typename Index>
    static std::unique_ptr<BaseArgument> MakeArgument(const void* in,
                                                      void* wei,
                                                      const void* out,
                                                      const std::array<Index, 5>& il,
                                                      const std::array<Index, 5>& is,
                                                      const std::array<Index, 5>& wl,
                                                      const std::array<Index, 5>& ws,
                                                      const std::array<Index, 5>& ol,
                                                      const std::array<Index, 5>& os,
                                                      const std::array<Index, 2>& fs,
                                                      const std::array<Index, 2>& fd,
                                                      const std::array<Index, 2>& lp,
                                                      const std::array<Index, 2>& rp,
                                                      index_t split)
    {
        auto a = std::make_unique<Argument>();
        a->in  = static_cast<const bhalf_t*>(in);
        a->wei = static_cast<bhalf_t*>(wei);
        a->out = static_cast<const bhalf_t*>(out);
        for(index_t i = 0; i < 5; ++i)
        {
            a->in_lengths[i]  = il[i];
            a->in_strides[i]  = is[i];
            a->wei_lengths[i] = wl[i];
            a->wei_strides[i] = ws[i];
            a->out_lengths[i] = ol[i];
            a->out_strides[i] = os[i];
        }
        for(index_t i = 0; i < 2; ++i)
        {
            a->filter_strides[i]   = fs[i];
            a->filter_dilations[i] = fd[i];
            a->left_pads[i]        = lp[i];
            a->right_pads[i]       = rp[i];
        }
        // Auto -1 and the legacy zero clamp choose the only legal split.
        a->k_batch_ = (split == -1 || split == 0) ? 1 : split;
        a->valid    = Supports(*a);
        a->reduction_length =
            a->valid
                ? static_cast<index_t>(a->out_lengths[1] * a->out_lengths[3] * a->out_lengths[4])
                : 0;
        return a;
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* in,
                                                      void* wei,
                                                      const void* out,
                                                      const std::array<index_t, 5>& il,
                                                      const std::array<index_t, 5>& is,
                                                      const std::array<index_t, 5>& wl,
                                                      const std::array<index_t, 5>& ws,
                                                      const std::array<index_t, 5>& ol,
                                                      const std::array<index_t, 5>& os,
                                                      const std::array<index_t, 2>& fs,
                                                      const std::array<index_t, 2>& fd,
                                                      const std::array<index_t, 2>& lp,
                                                      const std::array<index_t, 2>& rp,
                                                      element_wise::PassThrough,
                                                      element_wise::PassThrough,
                                                      element_wise::PassThrough,
                                                      index_t split) override
    {
        return MakeArgument(in, wei, out, il, is, wl, ws, ol, os, fs, fd, lp, rp, split);
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* in,
                                                      void* wei,
                                                      const void* out,
                                                      const std::array<long_index_t, 5>& il,
                                                      const std::array<long_index_t, 5>& is,
                                                      const std::array<long_index_t, 5>& wl,
                                                      const std::array<long_index_t, 5>& ws,
                                                      const std::array<long_index_t, 5>& ol,
                                                      const std::array<long_index_t, 5>& os,
                                                      const std::array<long_index_t, 2>& fs,
                                                      const std::array<long_index_t, 2>& fd,
                                                      const std::array<long_index_t, 2>& lp,
                                                      const std::array<long_index_t, 2>& rp,
                                                      element_wise::PassThrough,
                                                      element_wise::PassThrough,
                                                      element_wise::PassThrough,
                                                      index_t split) override
    {
        return MakeArgument(in, wei, out, il, is, wl, ws, ol, os, fs, fd, lp, rp, split);
    }

    struct Invoker : BaseInvoker
    {
        float Run(const BaseArgument* base, const StreamConfig& stream = StreamConfig{}) override
        {
            const auto* a = dynamic_cast<const Argument*>(base);
            if(!a || !a->valid || !a->in || !a->out || !a->wei)
                throw std::runtime_error("Unsupported BF16 depthwise grouped WRW argument");
            return launch_and_time_kernel(
                stream,
                kernel_grouped_conv2d_bwd_weight_depthwise_bf16<bhalf_t>,
                dim3((a->in_lengths[0] - 1) / 8 + 1, a->wei_lengths[3] * a->wei_lengths[4]),
                dim3(256),
                0,
                static_cast<const DepthwiseBwdWeightBf16Params&>(*a));
        }
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    bool IsSupportedArgument(const BaseArgument* a) override
    {
        const auto* arg = dynamic_cast<const Argument*>(a);
        return arg && arg->valid;
    }

    std::string GetTypeString() const override
    {
        return "DeviceGroupedConvBwdWeightDepthwiseBf16<8, 32, Filter1, Split1>";
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
