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

// P[s, g, f], where s = n * ceil(Ho / 12) + strip and f = fy * FilterSize + fx.
template <typename DeviceIndex>
struct DepthwiseRowStripBf16Params
{
    using Index = DeviceIndex;
    const bhalf_t* in;
    bhalf_t* wei;
    const bhalf_t* out;
    float* partial;
    DeviceIndex groups;
    DeviceIndex in_h;
    DeviceIndex in_w;
    DeviceIndex out_h;
    DeviceIndex out_w;
    DeviceIndex strips_per_image;
    DeviceIndex partial_splits;
    DeviceIndex stride_h, stride_w;
    DeviceIndex dilation_h, dilation_w;
    DeviceIndex left_pad_h, left_pad_w;
};

template <index_t FilterSize>
struct DepthwiseRowStripBf16Argument : BaseArgument,
                                       ArgumentSplitK,
                                       DepthwiseRowStripBf16Params<long_index_t>
{
    bool valid                 = false;
    size_t workspace_bytes     = 0;
    bool narrow_device_indices = false;
};

template <index_t FilterWidth, typename Params>
__global__ void kernel_grouped_conv2d_bwd_weight_depthwise_row_strip_bf16(Params a)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx125__)
    using DeviceIndex                    = typename Params::Index;
    constexpr index_t FilterRows         = 2;
    constexpr index_t BlockSize          = 256;
    const index_t tid                    = threadIdx.x;
    const index_t row_lane               = tid / 128;
    const index_t col_lane               = tid % 128;
    const DeviceIndex s                  = blockIdx.x;
    const DeviceIndex n                  = s / a.strips_per_image;
    const DeviceIndex strip              = s % a.strips_per_image;
    const DeviceIndex g                  = blockIdx.y;
    const DeviceIndex fy                 = static_cast<DeviceIndex>(blockIdx.z) * FilterRows;
    const bool second_row                = fy + 1 < FilterWidth;
    float accum[FilterRows][FilterWidth] = {};

    for(index_t j = 0; j < 6; ++j)
    {
        const DeviceIndex ho = strip * 12 + row_lane + 2 * j;
        if(ho >= a.out_h)
            continue;
        const DeviceIndex hi =
            ho * a.stride_h + fy * a.dilation_h - a.left_pad_h;
        const DeviceIndex second_hi = hi + a.dilation_h;
        const bool valid_first  = hi >= 0 && hi < a.in_h;
        const bool valid_second = second_row && second_hi >= 0 && second_hi < a.in_h;
        if(!valid_first && !valid_second)
            continue;

        for(DeviceIndex wo = col_lane; wo < a.out_w; wo += 128)
        {
            const DeviceIndex out_offset = ((n * a.out_h + ho) * a.out_w + wo) * a.groups + g;
            const float dy               = type_convert<float>(a.out[out_offset]);
#pragma unroll
            for(index_t fx = 0; fx < FilterWidth; ++fx)
            {
                const DeviceIndex wi = wo * a.stride_w + fx * a.dilation_w - a.left_pad_w;
                if(wi >= 0 && wi < a.in_w)
                {
                    if(valid_first)
                    {
                        const DeviceIndex in_offset =
                            ((n * a.in_h + hi) * a.in_w + wi) * a.groups + g;
                        accum[0][fx] += type_convert<float>(a.in[in_offset]) * dy;
                    }
                    if(valid_second)
                    {
                        const DeviceIndex in_offset =
                            ((n * a.in_h + second_hi) * a.in_w + wi) * a.groups + g;
                        accum[1][fx] += type_convert<float>(a.in[in_offset]) * dy;
                    }
                }
            }
        }
    }

    // Reduce within each wave before publishing one partial per wave and tap.
    // Every source lane is active even for empty rows/columns.
    constexpr index_t Waves = BlockSize / 32;
    const index_t lane      = tid % 32;
    const index_t wave      = tid / 32;
#pragma unroll
    for(index_t step = 16; step > 0; step /= 2)
    {
#pragma unroll
        for(index_t row = 0; row < FilterRows; ++row)
#pragma unroll
            for(index_t fx = 0; fx < FilterWidth; ++fx)
            {
                const float other = __builtin_bit_cast(
                    float,
                    __builtin_amdgcn_ds_bpermute(((lane ^ step) << 2),
                                                 __builtin_bit_cast(int, accum[row][fx])));
                if(lane < step)
                    accum[row][fx] += other;
            }
    }
    __shared__ float reduction[FilterRows][FilterWidth][Waves];
    if(lane == 0)
    {
#pragma unroll
        for(index_t row = 0; row < FilterRows; ++row)
#pragma unroll
            for(index_t fx = 0; fx < FilterWidth; ++fx)
                reduction[row][fx][wave] = accum[row][fx];
    }
    __syncthreads();
    if(tid < FilterWidth)
    {
        float first[Waves];
        float second[Waves];
#pragma unroll
        for(index_t w = 0; w < Waves; ++w)
        {
            first[w]  = reduction[0][tid][w];
            second[w] = reduction[1][tid][w];
        }
#pragma unroll
        for(index_t step = Waves / 2; step > 0; step /= 2)
#pragma unroll
            for(index_t w = 0; w < step; ++w)
            {
                first[w] += first[w + step];
                second[w] += second[w + step];
            }
        const DeviceIndex offset =
            (s * a.groups + g) * (FilterWidth * FilterWidth) + fy * FilterWidth + tid;
        a.partial[offset] = first[0];
        if(second_row)
            a.partial[offset + FilterWidth] = second[0];
    }
#else
    ignore = a;
#endif
}


template <index_t FilterSize = 11>
struct DeviceGroupedConvBwdWeightDepthwiseRowStripBf16 final
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
    static_assert(FilterSize > 0 && FilterSize <= 256,
                  "Every filter column needs a producer thread");
    static constexpr long_index_t FilterElements = FilterSize * FilterSize;
    using Argument = DepthwiseRowStripBf16Argument<FilterSize>;

    static bool
    CheckedMultiply(long_index_t x, long_index_t y, long_index_t limit, long_index_t& result)
    {
        if(x <= 0 || y <= 0 || x > limit / y)
            return false;
        result = x * y;
        return true;
    }

    // Checked byte spans also bound every valid x/dY/P offset. For masked
    // lanes: maximum ho is 12*strips-1, maximum hi+1 is 12*strips+5;
    // terminal wo+=128 is at most out_w+127. The finalizer can advance
    // s to S+15 and form f up to weights+15 before testing the mask.
    static bool NarrowIndicesSafe(long_index_t in_count,
                                  long_index_t out_count,
                                  long_index_t partials,
                                  long_index_t strips,
                                  long_index_t out_w,
                                  long_index_t splits,
                                  long_index_t weights)
    {
        constexpr long_index_t MaxIndex = std::numeric_limits<index_t>::max();
        if(in_count > MaxIndex / 8 || out_count > MaxIndex / 8 || partials > MaxIndex / 8 ||
           strips > (MaxIndex - 5) / 12 || out_w > MaxIndex - 128)
            return false;
        long_index_t terminal;
        return CheckedMultiply(splits + 16, weights + 16, MaxIndex, terminal);
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
        // Exact packed depthwise layouts and the square filter specialization.
        const long_index_t g = il[0];
        if(ol[0] != g || wl[0] != g || il[1] != ol[1] || il[2] != 1 || wl[1] != 1 || wl[2] != 1 ||
           ol[2] != 1 || wl[3] != FilterSize || wl[4] != FilterSize)
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

        const long_index_t strips = (ol[3] - 1) / 12 + 1;
        long_index_t s, weights, partials;
        if(!CheckedMultiply(il[1], strips, MaxIndex, s) ||
           !CheckedMultiply(g, FilterElements, MaxBf16Elements, weights) ||
           !CheckedMultiply(s, weights, MaxFloatElements, partials))
            return false;
        if(!DepthwiseBf16LaunchSupported(s, g, (FilterSize + 1) / 2,
                                       2 * FilterSize * 8 * sizeof(float)) ||
           !DepthwiseBf16LaunchSupported((weights - 1) / 16 + 1, 1, 1, 8 * 16 * sizeof(float)))
            return false;
        const auto logical_bytes = static_cast<size_t>(partials) * sizeof(float);
        if(logical_bytes > static_cast<size_t>(MaxBytes - 255))
            return false;
        a.narrow_device_indices =
            FilterSize == 11 && fs[0] == 1 && fs[1] == 1 && fd[0] == 1 && fd[1] == 1 &&
            lp[0] == 5 && lp[1] == 5 && rp[0] == 5 && rp[1] == 5 &&
            NarrowIndicesSafe(in_count, out_count, partials, strips, ol[4], s, weights);
        a.groups           = g;
        a.in_h             = il[3];
        a.in_w             = il[4];
        a.out_h            = ol[3];
        a.out_w            = ol[4];
        a.strips_per_image = strips;
        a.partial_splits   = s;
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
        a->valid    = Validate(il, is, wl, ws, ol, os, fs, fd, lp, rp, split, *a);
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
                    "Unsupported BF16 depthwise row-strip WRW argument or scratch");

            const auto wide = static_cast<const DepthwiseRowStripBf16Params<long_index_t>&>(*a);
            const dim3 stage1_grid(
                static_cast<uint32_t>(a->partial_splits), static_cast<uint32_t>(a->groups),
                (FilterSize + 1) / 2);
            const dim3 stage2_grid(
                static_cast<uint32_t>((a->groups * FilterElements - 1) / 16 + 1));
            const auto run_pair = [&](auto params) {
                using Params         = decltype(params);
                params.partial       = static_cast<float*>(a->p_workspace_);
                const auto stage1_ms = launch_and_time_kernel(
                    stream,
                    kernel_grouped_conv2d_bwd_weight_depthwise_row_strip_bf16<FilterSize, Params>,
                    stage1_grid,
                    dim3(256),
                    0,
                    params);
                using DeviceIndex = typename Params::Index;
                const DepthwiseBwdWeightBf16FinalizeParams<DeviceIndex> finalize{
                    params.wei, params.partial,
                    static_cast<DeviceIndex>(params.groups * FilterElements), params.partial_splits};
                const auto stage2_ms = launch_and_time_kernel(
                    stream,
                    kernel_grouped_conv2d_bwd_weight_depthwise_strip_finalize_bf16<DeviceIndex>,
                    stage2_grid,
                    dim3(256),
                    0,
                    finalize);
                return stage1_ms + stage2_ms;
            };
            if(a->narrow_device_indices)
            {
                DepthwiseRowStripBf16Params<index_t> narrow{
                    wide.in,
                    wide.wei,
                    wide.out,
                    nullptr,
                    static_cast<index_t>(wide.groups),
                    static_cast<index_t>(wide.in_h),
                    static_cast<index_t>(wide.in_w),
                    static_cast<index_t>(wide.out_h),
                    static_cast<index_t>(wide.out_w),
                    static_cast<index_t>(wide.strips_per_image),
                    static_cast<index_t>(wide.partial_splits),
                    static_cast<index_t>(wide.stride_h),
                    static_cast<index_t>(wide.stride_w),
                    static_cast<index_t>(wide.dilation_h),
                    static_cast<index_t>(wide.dilation_w),
                    static_cast<index_t>(wide.left_pad_h),
                    static_cast<index_t>(wide.left_pad_w)};
                return run_pair(narrow);
            }
            return run_pair(wide);
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
            throw std::runtime_error("Invalid BF16 depthwise row-strip WRW argument");
        a->p_workspace_ = workspace;
    }

    std::string GetTypeString() const override
    {
        return "DeviceGroupedConvBwdWeightDepthwiseRowStripBf16<12, 128, " +
               std::to_string(FilterSize) + ", 16, Fy2, Split1>";
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
