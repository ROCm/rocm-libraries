// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// N is a separate launch dimension: NCHW cannot be flattened to a contiguous C x NHW GEMM.
template <index_t BlockSize, index_t ChannelsPerBlock>
__global__ void kernel_grouped_conv_bwd_data_nchw_pointwise(const float* __restrict__ dy,
                                                            const float* __restrict__ weights,
                                                            float* __restrict__ dx,
                                                            index_t spatial,
                                                            index_t channels,
                                                            index_t reduction,
                                                            index_t dy_batch_stride,
                                                            index_t dy_channel_stride,
                                                            index_t weight_reduction_stride,
                                                            index_t weight_channel_stride,
                                                            index_t dx_batch_stride,
                                                            index_t dx_channel_stride,
                                                            bool vector_access)
{
    const index_t s = (blockIdx.x * BlockSize + threadIdx.x) * 4;
    if(s >= spatial)
        return;
    const index_t c      = blockIdx.y * ChannelsPerBlock;
    const long_index_t n = blockIdx.z;
    float4 accum[ChannelsPerBlock]{};
    for(index_t r = 0; r < reduction; ++r)
    {
        const auto* p = dy + n * dy_batch_stride + r * dy_channel_stride + s;
        float4 x{};
        if(vector_access && s + 3 < spatial)
            x = *reinterpret_cast<const float4*>(p);
        else
        {
            if(s < spatial)
                x.x = p[0];
            if(s + 1 < spatial)
                x.y = p[1];
            if(s + 2 < spatial)
                x.z = p[2];
            if(s + 3 < spatial)
                x.w = p[3];
        }
        static_for<0, ChannelsPerBlock, 1>{}([&](auto i) {
            if(c + i < channels)
            {
                const float w =
                    weights[r * weight_reduction_stride + (c + i) * weight_channel_stride];
                accum[i].x = __builtin_fmaf(x.x, w, accum[i].x);
                accum[i].y = __builtin_fmaf(x.y, w, accum[i].y);
                accum[i].z = __builtin_fmaf(x.z, w, accum[i].z);
                accum[i].w = __builtin_fmaf(x.w, w, accum[i].w);
            }
        });
    }
    static_for<0, ChannelsPerBlock, 1>{}([&](auto i) {
        if(c + i < channels)
        {
            auto* p = dx + n * dx_batch_stride + (c + i) * dx_channel_stride + s;
            if(vector_access && s + 3 < spatial)
                *reinterpret_cast<float4*>(p) = accum[i];
            else
            {
                if(s < spatial)
                    p[0] = accum[i].x;
                if(s + 1 < spatial)
                    p[1] = accum[i].y;
                if(s + 2 < spatial)
                    p[2] = accum[i].z;
                if(s + 3 < spatial)
                    p[3] = accum[i].w;
            }
        }
    });
}

// Bounded FP32 tiny-reduction path. Each thread reuses four spatial values across channels;
// the output is overwritten exactly once, without atomics, clear, packing or workspace.
template <index_t BlockSize = 256, index_t ChannelsPerBlock = 8>
struct DeviceGroupedConvBwdDataNCHWPointwise
    : DeviceGroupedConvBwdDataMultipleD<2,
                                        tensor_layout::convolution::NGKHW,
                                        tensor_layout::convolution::GKCYX,
                                        Tuple<>,
                                        tensor_layout::convolution::NGCHW,
                                        float,
                                        float,
                                        Tuple<>,
                                        float,
                                        element_wise::PassThrough,
                                        element_wise::PassThrough,
                                        element_wise::PassThrough>
{
    using Lengths     = std::array<long_index_t, 5>;
    using Filter      = std::array<long_index_t, 2>;
    using PassThrough = element_wise::PassThrough;
    struct Argument : BaseArgument
    {
        const float* dy;
        const float* weights;
        float* dx;
        Lengths a, as, b, bs, e, es;
        Filter strides, dilations, left, right;
        index_t split;
        bool vector_access;

        Argument(const void* p_a,
                 const void* p_b,
                 void* p_e,
                 Lengths a_,
                 Lengths as_,
                 Lengths b_,
                 Lengths bs_,
                 Lengths e_,
                 Lengths es_,
                 Filter strides_,
                 Filter dilations_,
                 Filter left_,
                 Filter right_,
                 index_t split_)
            : dy(static_cast<const float*>(p_a)),
              weights(static_cast<const float*>(p_b)),
              dx(static_cast<float*>(p_e)),
              a(a_),
              as(as_),
              b(b_),
              bs(bs_),
              e(e_),
              es(es_),
              strides(strides_),
              dilations(dilations_),
              left(left_),
              right(right_),
              split(split_),
              vector_access(reinterpret_cast<std::uintptr_t>(p_a) % 16 == 0 &&
                            reinterpret_cast<std::uintptr_t>(p_e) % 16 == 0 && as_[1] % 4 == 0 &&
                            as_[2] % 4 == 0 && es_[1] % 4 == 0 && es_[2] % 4 == 0)
        {
        }
    };

    // Keep both tensor-byte and indexing limits, including padding in the descriptor span.
    static bool ValidSpan(const Lengths& lengths, const Lengths& strides, std::uintptr_t& bytes)
    {
        constexpr long_index_t MaxElements = (long_index_t{1} << 31) / sizeof(float);
        long_index_t span                  = 1;
        for(index_t i = 0; i < 5; ++i)
        {
            if(lengths[i] <= 0 || lengths[i] > MaxElements || strides[i] <= 0 ||
               strides[i] > MaxElements)
                return false;
            if(lengths[i] - 1 > (MaxElements - span) / strides[i])
                return false;
            span += (lengths[i] - 1) * strides[i];
        }
        bytes = static_cast<std::uintptr_t>(span) * sizeof(float);
        return true;
    }

    static bool Disjoint(const void* first,
                         std::uintptr_t first_bytes,
                         const void* second,
                         std::uintptr_t second_bytes)
    {
        // Factory applicability queries have null data pointers.
        if(!first || !second)
            return true;
        const auto a              = reinterpret_cast<std::uintptr_t>(first);
        const auto b              = reinterpret_cast<std::uintptr_t>(second);
        constexpr auto MaxAddress = std::numeric_limits<std::uintptr_t>::max();
        if(a > MaxAddress - first_bytes || b > MaxAddress - second_bytes)
            return false;
        return a + first_bytes <= b || b + second_bytes <= a;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        std::uintptr_t a_bytes, b_bytes, e_bytes;
        if(!is_gfx125_supported() || arg.split != 1 ||
           reinterpret_cast<std::uintptr_t>(arg.dy) % alignof(float) != 0 ||
           reinterpret_cast<std::uintptr_t>(arg.weights) % alignof(float) != 0 ||
           reinterpret_cast<std::uintptr_t>(arg.dx) % alignof(float) != 0 ||
           !ValidSpan(arg.a, arg.as, a_bytes) || !ValidSpan(arg.b, arg.bs, b_bytes) ||
           !ValidSpan(arg.e, arg.es, e_bytes) || !Disjoint(arg.dx, e_bytes, arg.dy, a_bytes) ||
           !Disjoint(arg.dx, e_bytes, arg.weights, b_bytes))
            return false;
        const auto spatial = arg.e[3] * arg.e[4];
        if(arg.a[0] != 1 || arg.b[0] != 1 || arg.e[0] != 1 || arg.a[1] != arg.e[1] ||
           arg.a[1] > 128 || arg.a[2] != arg.b[1] || arg.e[2] != arg.b[2] || arg.a[2] > 16 ||
           arg.e[2] < 64 || arg.e[2] > 512 || spatial < 1024 || spatial > 65536 ||
           arg.a[3] != arg.e[3] || arg.a[4] != arg.e[4] || arg.b[3] != 1 || arg.b[4] != 1)
            return false;
        for(index_t i = 0; i < 2; ++i)
            if(arg.strides[i] != 1 || arg.dilations[i] != 1 || arg.left[i] != 0 ||
               arg.right[i] != 0)
                return false;
        // Spatial dimensions must be packed; channels and batches may include padding.
        return arg.as[4] == 1 && arg.es[4] == 1 && arg.as[3] == arg.a[4] && arg.es[3] == arg.e[4] &&
               arg.as[2] >= spatial && arg.es[2] >= spatial && arg.as[1] >= arg.a[2] * arg.as[2] &&
               arg.es[1] >= arg.e[2] * arg.es[2] && arg.bs[1] >= arg.b[2] * arg.bs[2];
    }

    bool IsSupportedArgument(const BaseArgument* arg) override
    {
        const auto* p = dynamic_cast<const Argument*>(arg);
        return p && IsSupportedArgument(*p);
    }

    struct Invoker : BaseInvoker
    {
        float Run(const BaseArgument* p, const StreamConfig& config = StreamConfig{}) override
        {
            const auto* arg = dynamic_cast<const Argument*>(p);
            if(!arg || !IsSupportedArgument(*arg))
                throw std::runtime_error(
                    "Unsupported native NCHW pointwise backward data argument");
            const index_t spatial = arg->e[3] * arg->e[4];
            return launch_and_time_kernel(
                config,
                kernel_grouped_conv_bwd_data_nchw_pointwise<BlockSize, ChannelsPerBlock>,
                dim3((spatial + BlockSize * 4 - 1) / (BlockSize * 4),
                     (arg->e[2] + ChannelsPerBlock - 1) / ChannelsPerBlock,
                     arg->e[1]),
                dim3(BlockSize),
                0,
                arg->dy,
                arg->weights,
                arg->dx,
                spatial,
                index_t(arg->e[2]),
                index_t(arg->a[2]),
                index_t(arg->as[1]),
                index_t(arg->as[2]),
                index_t(arg->bs[1]),
                index_t(arg->bs[2]),
                index_t(arg->es[1]),
                index_t(arg->es[2]),
                arg->vector_access);
        }
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override
    {
        return "DeviceGroupedConvBwdDataNCHWPointwise<" + std::to_string(BlockSize) + "," +
               std::to_string(ChannelsPerBlock) + ">";
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      const std::array<const void*, 0>&,
                                                      void* p_e,
                                                      const Lengths& a,
                                                      const Lengths& as,
                                                      const Lengths& b,
                                                      const Lengths& bs,
                                                      const std::array<Lengths, 0>&,
                                                      const std::array<Lengths, 0>&,
                                                      const Lengths& e,
                                                      const Lengths& es,
                                                      const Filter& strides,
                                                      const Filter& dilations,
                                                      const Filter& left,
                                                      const Filter& right,
                                                      const PassThrough&,
                                                      const PassThrough&,
                                                      const PassThrough&,
                                                      index_t split = 1) override
    {
        return std::make_unique<Argument>(
            p_a, p_b, p_e, a, as, b, bs, e, es, strides, dilations, left, right, split);
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      const std::array<const void*, 0>& p_ds,
                                                      void* p_e,
                                                      const std::array<index_t, 5>& a,
                                                      const std::array<index_t, 5>& as,
                                                      const std::array<index_t, 5>& b,
                                                      const std::array<index_t, 5>& bs,
                                                      const std::array<std::array<index_t, 5>, 0>&,
                                                      const std::array<std::array<index_t, 5>, 0>&,
                                                      const std::array<index_t, 5>& e,
                                                      const std::array<index_t, 5>& es,
                                                      const std::array<index_t, 2>& strides,
                                                      const std::array<index_t, 2>& dilations,
                                                      const std::array<index_t, 2>& left,
                                                      const std::array<index_t, 2>& right,
                                                      const PassThrough& op_a,
                                                      const PassThrough& op_b,
                                                      const PassThrough& op_e,
                                                      index_t split = 1) override
    {
        auto widen = [](const auto& src) {
            std::array<long_index_t, std::tuple_size_v<std::decay_t<decltype(src)>>> dst{};
            std::copy(src.begin(), src.end(), dst.begin());
            return dst;
        };
        return MakeArgumentPointer(p_a,
                                   p_b,
                                   p_ds,
                                   p_e,
                                   widen(a),
                                   widen(as),
                                   widen(b),
                                   widen(bs),
                                   {},
                                   {},
                                   widen(e),
                                   widen(es),
                                   widen(strides),
                                   widen(dilations),
                                   widen(left),
                                   widen(right),
                                   op_a,
                                   op_b,
                                   op_e,
                                   split);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
