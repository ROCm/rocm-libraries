// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/device/impl/depthwise_bwd_weight_bf16_finalize.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/type_convert.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// P[s, g, f], s = n * ceil(Ho / 8) * ceil(Wo / 80)
//                 + ystrip * ceil(Wo / 80) + xstrip, f = fy * FilterSize + fx.
struct DepthwiseGroupedRowStripBf16Params
{
    const bhalf_t* in;
    bhalf_t* wei;
    const bhalf_t* out;
    float* partial;
    long_index_t groups;
    long_index_t in_h;
    long_index_t in_w;
    long_index_t out_h;
    long_index_t out_w;
    long_index_t strips_per_image;
    long_index_t width_strips;
    long_index_t partial_splits;
    long_index_t stride_h, stride_w;
    long_index_t dilation_h, dilation_w;
    long_index_t left_pad_h, left_pad_w;
};

template <index_t FilterSize>
struct DepthwiseGroupedRowStripBf16Argument : BaseArgument,
                                              ArgumentSplitK,
                                              DepthwiseGroupedRowStripBf16Params
{
    bool valid             = false;
    size_t workspace_bytes = 0;
};

// Each CTA owns one eight-row by at-most-80-column output strip and sixteen
// adjacent groups. The 16 reduction lanes visit five 16-column waves;
// every valid output contributes once to each filter weight.
template <index_t FilterSize>
__global__ void kernel_grouped_conv2d_bwd_weight_depthwise_grouped_row_strip_bf16(
    DepthwiseGroupedRowStripBf16Params a)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx125__)
    constexpr index_t GroupLanes     = 16;
    constexpr index_t ReductionLanes = 16;
    constexpr index_t FilterElements = FilterSize * FilterSize;
    const index_t tid                = threadIdx.x;
    const index_t group_lane         = tid % GroupLanes;
    const index_t reduction_lane     = tid / GroupLanes;
    const long_index_t s             = blockIdx.x;
    const long_index_t n             = s / a.strips_per_image;
    const long_index_t strip         = s % a.strips_per_image;
    const long_index_t ystrip        = strip / a.width_strips;
    const long_index_t xstrip        = strip % a.width_strips;
    const long_index_t group    = static_cast<long_index_t>(blockIdx.y) * GroupLanes + group_lane;
    float accum[FilterElements] = {};

    if(group < a.groups)
    {
        for(index_t j = 0; j < 8; ++j)
        {
            const long_index_t ho = ystrip * 8 + j;
            if(ho >= a.out_h)
                continue;

            for(index_t wave = 0; wave < 5; ++wave)
            {
                const long_index_t wo = xstrip * 80 + reduction_lane + wave * ReductionLanes;
                if(wo >= a.out_w)
                    continue;

                const long_index_t out_offset =
                    ((n * a.out_h + ho) * a.out_w + wo) * a.groups + group;
                const float dy = type_convert<float>(a.out[out_offset]);
#pragma unroll
                for(index_t fy = 0; fy < FilterSize; ++fy)
                {
                    const long_index_t hi = a.stride_h * ho + fy * a.dilation_h - a.left_pad_h;
                    if(hi >= 0 && hi < a.in_h)
                    {
                        const long_index_t in_row = ((n * a.in_h + hi) * a.in_w) * a.groups + group;
#pragma unroll
                        for(index_t fx = 0; fx < FilterSize; ++fx)
                        {
                            const long_index_t wi = a.stride_w * wo + fx * a.dilation_w - a.left_pad_w;
                            if(wi >= 0 && wi < a.in_w)
                                accum[fy * FilterSize + fx] +=
                                    type_convert<float>(a.in[in_row + wi * a.groups]) * dy;
                        }
                    }
                }
            }
        }
    }

    // Each wave has two lanes for each of sixteen groups. Cross-half exchange
    // combines those lanes without ever mixing adjacent groups.
    const index_t lane = tid % 32;
#pragma unroll
    for(index_t f = 0; f < FilterElements; ++f)
    {
        const float other = __builtin_bit_cast(
            float,
            __builtin_amdgcn_ds_bpermute(((lane ^ 16) << 2), __builtin_bit_cast(int, accum[f])));
        if(lane < GroupLanes)
            accum[f] += other;
    }
    constexpr index_t Waves = 8;
    __shared__ float reduction[FilterElements][Waves][GroupLanes];
    if(lane < GroupLanes)
    {
#pragma unroll
        for(index_t f = 0; f < FilterElements; ++f)
            reduction[f][tid / 32][group_lane] = accum[f];
    }
    __syncthreads();
    if(reduction_lane == 0 && group < a.groups)
    {
        const long_index_t offset = (s * a.groups + group) * FilterElements;
#pragma unroll
        for(index_t f = 0; f < FilterElements; ++f)
        {
            float wave_sums[Waves];
#pragma unroll
            for(index_t w = 0; w < Waves; ++w)
                wave_sums[w] = reduction[f][w][group_lane];
#pragma unroll
            for(index_t step = Waves / 2; step > 0; step /= 2)
#pragma unroll
                for(index_t w = 0; w < step; ++w)
                    wave_sums[w] += wave_sums[w + step];
            a.partial[offset + f] = wave_sums[0];
        }
    }
#else
    ignore = a;
#endif
}


template <index_t FilterSize = 3>
struct DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16 final
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
    static_assert(FilterSize > 0 && FilterSize <= 11,
                  "The producer filter reduction must fit in 64 KiB LDS");
    static constexpr long_index_t FilterElements = FilterSize * FilterSize;
    using Argument = DepthwiseGroupedRowStripBf16Argument<FilterSize>;

    static bool
    CheckedMultiply(long_index_t x, long_index_t y, long_index_t limit, long_index_t& result)
    {
        if(x <= 0 || y <= 0 || x > limit / y)
            return false;
        result = x * y;
        return true;
    }

    template <typename Index>
    static bool Validate(const std::array<Index, 5>& il,
                         const std::array<Index, 5>& is,
                         const std::array<Index, 5>& wl,
                         const std::array<Index, 5>& ws,
                         const std::array<Index, 5>& ol,
                         const std::array<Index, 5>& os,
                         const std::array<Index, 2>& fs,
                         const std::array<Index, 2>& fd,
                         const std::array<Index, 2>& lp,
                         const std::array<Index, 2>& rp,
                         index_t split,
                         Argument& a)
    {
        if(!is_gfx125_supported() || (split != -1 && split != 0 && split != 1))
            return false;
        constexpr long_index_t MaxIndex         = std::numeric_limits<index_t>::max();
        constexpr long_index_t MaxBytes         = std::numeric_limits<std::ptrdiff_t>::max();
        constexpr long_index_t MaxBf16Elements  = MaxBytes / sizeof(bhalf_t);
        constexpr long_index_t MaxFloatElements = MaxBytes / sizeof(float);
        for(index_t d = 0; d < 5; ++d)
        {
            if(il[d] <= 0 || il[d] > MaxIndex || wl[d] <= 0 || wl[d] > MaxIndex || ol[d] <= 0 ||
               ol[d] > MaxIndex)
                return false;
        }
        // Exact packed depthwise layouts; incomplete group tiles are masked.
        const long_index_t g = il[0];
        if(ol[0] != g || wl[0] != g || il[1] != ol[1] || il[2] != 1 || wl[1] != 1 ||
           wl[2] != 1 || ol[2] != 1 || wl[3] != FilterSize || wl[4] != FilterSize)
            return false;
        for(index_t d = 0; d < 2; ++d)
        {
            if(fs[d] < 1 || fs[d] > MaxIndex || fd[d] < 1 || fd[d] > MaxIndex ||
               lp[d] < 0 || lp[d] > MaxIndex || rp[d] < 0 || rp[d] > MaxIndex)
                return false;
            const long_index_t effective = (FilterSize - 1) * static_cast<long_index_t>(fd[d]) + 1;
            const long_index_t padded = static_cast<long_index_t>(il[d + 3]) + lp[d] + rp[d];
            if(effective > padded || (padded - effective) / fs[d] + 1 != ol[d + 3])
                return false;
        }

        long_index_t in_plane, out_plane, in_image, out_image, in_count, out_count;
        if(!CheckedMultiply(il[4], g, MaxBf16Elements, in_plane) ||
           !CheckedMultiply(ol[4], g, MaxBf16Elements, out_plane) ||
           !CheckedMultiply(il[3], in_plane, MaxBf16Elements, in_image) ||
           !CheckedMultiply(ol[3], out_plane, MaxBf16Elements, out_image) ||
           !CheckedMultiply(il[1], in_image, MaxBf16Elements, in_count) ||
           !CheckedMultiply(ol[1], out_image, MaxBf16Elements, out_count))
            return false;
        if(is[0] != 1 || is[1] != in_image || is[2] != 1 || is[3] != in_plane || is[4] != g ||
           os[0] != 1 || os[1] != out_image || os[2] != 1 || os[3] != out_plane || os[4] != g ||
           ws[0] != FilterElements || ws[1] != FilterElements || ws[2] != 1 ||
           ws[3] != FilterSize || ws[4] != 1)
            return false;

        const long_index_t height_strips = (ol[3] - 1) / 8 + 1;
        const long_index_t width_strips  = (ol[4] - 1) / 80 + 1;
        long_index_t strips, splits, weights, partials;
        if(!CheckedMultiply(height_strips, width_strips, MaxIndex, strips) ||
           !CheckedMultiply(il[1], strips, MaxIndex, splits) ||
           !CheckedMultiply(g, FilterElements, MaxBf16Elements, weights) ||
           !CheckedMultiply(splits, weights, MaxFloatElements, partials))
            return false;
        const auto logical_bytes = static_cast<size_t>(partials) * sizeof(float);
        if(logical_bytes > static_cast<size_t>(MaxBytes - 255))
            return false;
        if(!DepthwiseBf16LaunchSupported(splits, (g - 1) / 16 + 1, 1,
                                       FilterElements * 8 * 16 * sizeof(float)) ||
           !DepthwiseBf16LaunchSupported((weights - 1) / 16 + 1, 1, 1, 8 * 16 * sizeof(float)))
            return false;
        a.groups           = g;
        a.in_h             = il[3];
        a.in_w             = il[4];
        a.out_h            = ol[3];
        a.out_w            = ol[4];
        a.strips_per_image = strips;
        a.width_strips     = width_strips;
        a.partial_splits   = splits;
        a.stride_h         = fs[0];
        a.stride_w         = fs[1];
        a.dilation_h       = fd[0];
        a.dilation_w       = fd[1];
        a.left_pad_h       = lp[0];
        a.left_pad_w       = lp[1];
        a.workspace_bytes  = (logical_bytes + 255) & ~size_t{255};
        return true;
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
        auto a      = std::make_unique<Argument>();
        a->in       = static_cast<const bhalf_t*>(in);
        a->wei      = static_cast<bhalf_t*>(wei);
        a->out      = static_cast<const bhalf_t*>(out);
        a->partial  = nullptr;
        a->k_batch_ = (split == -1 || split == 0) ? 1 : split;
        a->valid = Validate(il, is, wl, ws, ol, os, fs, fd, lp, rp, split, *a);
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
            if(!a || !a->valid || !a->in || !a->out || !a->wei || !a->p_workspace_ ||
               (reinterpret_cast<std::uintptr_t>(a->p_workspace_) % alignof(float)) != 0)
                throw std::runtime_error(
                    "Unsupported BF16 depthwise grouped row-strip WRW argument or scratch");

            auto params    = static_cast<const DepthwiseGroupedRowStripBf16Params&>(*a);
            params.partial = static_cast<float*>(a->p_workspace_);
            const dim3 stage1_grid(static_cast<uint32_t>(a->partial_splits),
                                   static_cast<uint32_t>((a->groups - 1) / 16 + 1));
            const auto stage1_ms = launch_and_time_kernel(
                stream,
                kernel_grouped_conv2d_bwd_weight_depthwise_grouped_row_strip_bf16<FilterSize>,
                stage1_grid,
                dim3(256),
                0,
                params);
            const DepthwiseBwdWeightBf16FinalizeParams<long_index_t> finalize{
                params.wei, params.partial, params.groups * FilterElements, params.partial_splits};
            const auto stage2_ms = launch_and_time_kernel(
                stream,
                kernel_grouped_conv2d_bwd_weight_depthwise_strip_finalize_bf16<long_index_t>,
                dim3(static_cast<uint32_t>((a->groups * FilterElements - 1) / 16 + 1)),
                dim3(256),
                0,
                finalize);
            return stage1_ms + stage2_ms;
        }
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    bool IsSupportedArgument(const BaseArgument* base) override
    {
        const auto* a = dynamic_cast<const Argument*>(base);
        return a && a->valid;
    }

    size_t GetWorkSpaceSize(const BaseArgument* base) const override
    {
        const auto* a = dynamic_cast<const Argument*>(base);
        return a && a->valid ? a->workspace_bytes : 0;
    }

    void SetWorkSpacePointer(BaseArgument* base,
                             void* workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto* a = dynamic_cast<Argument*>(base);
        if(!a)
            throw std::runtime_error("Invalid BF16 depthwise grouped row-strip WRW argument");
        a->p_workspace_ = workspace;
    }

    std::string GetTypeString() const override
    {
        return "DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16<16, 8, " +
               std::to_string(FilterElements) + ", Split1>";
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
