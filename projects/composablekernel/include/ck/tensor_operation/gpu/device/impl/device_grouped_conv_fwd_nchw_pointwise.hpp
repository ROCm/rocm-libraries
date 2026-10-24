// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#if defined(CK_USE_GFX1250) && defined(CK_ENABLE_FP16)
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// This is the existing RRR batched WMMA 128x128x32 intrawave-v1 instance. Keep
// its FP32 accumulator, vector widths and underlying support checks unchanged.
using NCHWPointwiseFwdGemm =
    DeviceBatchedGemm_Wmma_CShuffleV3<tensor_layout::gemm::RowMajor,
                                      tensor_layout::gemm::RowMajor,
                                      tensor_layout::gemm::RowMajor,
                                      half_t,
                                      half_t,
                                      half_t,
                                      float,
                                      half_t,
                                      element_wise::PassThrough,
                                      element_wise::PassThrough,
                                      element_wise::PassThrough,
                                      GemmSpecialization::MNKPadding,
                                      256,
                                      128,
                                      128,
                                      32,
                                      8,
                                      8,
                                      16,
                                      16,
                                      4,
                                      2,
                                      Sequence<4, 64, 1>,
                                      Sequence<1, 0, 2>,
                                      Sequence<1, 0, 2>,
                                      2,
                                      8,
                                      8,
                                      0,
                                      Sequence<4, 64, 1>,
                                      Sequence<0, 2, 1>,
                                      Sequence<0, 2, 1>,
                                      1,
                                      1,
                                      8,
                                      0,
                                      1,
                                      1,
                                      Sequence<1, 32, 1, 8>,
                                      8,
                                      BlockGemmPipelineScheduler::Intrawave,
                                      BlockGemmPipelineVersion::v1>;

struct DeviceGroupedConvFwdNCHWPointwise
    : DeviceGroupedConvFwdMultipleABD<2,
                                      tensor_layout::convolution::NGCHW,
                                      tensor_layout::convolution::GKCYX,
                                      Tuple<>,
                                      tensor_layout::convolution::NGKHW,
                                      half_t,
                                      half_t,
                                      Tuple<>,
                                      half_t,
                                      element_wise::PassThrough,
                                      element_wise::PassThrough,
                                      element_wise::PassThrough>
{
    using Lengths     = std::array<long_index_t, 5>;
    using Filter      = std::array<long_index_t, 2>;
    using PassThrough = element_wise::PassThrough;

    static bool ValidSpan(const Lengths& lengths, const Lengths& strides, std::uintptr_t& bytes)
    {
        constexpr long_index_t MaxElements = (long_index_t{1} << 31) / sizeof(half_t);
        long_index_t span                  = 1;
        for(index_t i = 0; i < 5; ++i)
        {
            if(lengths[i] <= 0 || lengths[i] > MaxElements || strides[i] <= 0 ||
               strides[i] > MaxElements || lengths[i] - 1 > (MaxElements - span) / strides[i])
                return false;
            span += (lengths[i] - 1) * strides[i];
        }
        bytes = static_cast<std::uintptr_t>(span) * sizeof(half_t);
        return true;
    }

    static bool
    Disjoint(const void* a, std::uintptr_t a_bytes, const void* b, std::uintptr_t b_bytes)
    {
        if(!a || !b)
            return true;
        const auto first          = reinterpret_cast<std::uintptr_t>(a);
        const auto second         = reinterpret_cast<std::uintptr_t>(b);
        constexpr auto MaxAddress = std::numeric_limits<std::uintptr_t>::max();
        if(first > MaxAddress - a_bytes || second > MaxAddress - b_bytes)
            return false;
        return first + a_bytes <= second || second + b_bytes <= first;
    }

    struct Argument : BaseArgument
    {
        std::optional<NCHWPointwiseFwdGemm::Argument> gemm;

        Argument(const void* input,
                 const void* weights,
                 void* output,
                 const Lengths& a,
                 const Lengths& as,
                 const Lengths& b,
                 const Lengths& bs,
                 const Lengths& e,
                 const Lengths& es,
                 const Filter& strides,
                 const Filter& dilation,
                 const Filter& left,
                 const Filter& right)
        {
            std::uintptr_t a_bytes, b_bytes, e_bytes;
            if(!ValidSpan(a, as, a_bytes) || !ValidSpan(b, bs, b_bytes) ||
               !ValidSpan(e, es, e_bytes) || !Disjoint(output, e_bytes, input, a_bytes) ||
               !Disjoint(output, e_bytes, weights, b_bytes) ||
               reinterpret_cast<std::uintptr_t>(weights) % 16 != 0 ||
               reinterpret_cast<std::uintptr_t>(output) % 16 != 0 ||
               reinterpret_cast<std::uintptr_t>(input) % alignof(half_t) != 0)
                return;
            const auto spatial = a[3] * a[4];
            if(a[0] != 1 || b[0] != 1 || e[0] != 1 || a[1] != e[1] || a[1] > 128 || a[2] != b[2] ||
               a[2] < 32 || a[2] > 64 || b[1] != e[2] || b[1] < 128 || b[1] > 256 || b[3] != 1 ||
               b[4] != 1 || a[3] != e[3] || a[4] != e[4] || spatial < 4096 || spatial > 32768)
                return;
            for(index_t i = 0; i < 2; ++i)
                if(strides[i] != 1 || dilation[i] != 1 || left[i] != 0 || right[i] != 0)
                    return;
            if(as[4] != 1 || es[4] != 1 || as[3] != a[4] || es[3] != e[4] || as[2] < spatial ||
               es[2] < spatial || bs[2] != 1 || bs[1] < b[2] || as[1] < a[2] * as[2] ||
               es[1] < e[2] * es[2])
                return;
            // N is the GEMM batch dimension, not part of a contiguous C x NHW
            // matrix. Broadcast weights; preserve caller channel and batch strides.
            gemm.emplace(NCHWPointwiseFwdGemm::MakeArgument(static_cast<const half_t*>(weights),
                                                            static_cast<const half_t*>(input),
                                                            static_cast<half_t*>(output),
                                                            index_t(e[2]),
                                                            index_t(spatial),
                                                            index_t(a[2]),
                                                            index_t(bs[1]),
                                                            index_t(as[2]),
                                                            index_t(es[2]),
                                                            0,
                                                            index_t(as[1]),
                                                            index_t(es[1]),
                                                            index_t(a[1]),
                                                            PassThrough{},
                                                            PassThrough{},
                                                            PassThrough{}));
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!is_gfx125_supported() || !arg.gemm)
            return false;
        // Do not relax the GEMM's alignment, vector, padding or numeric restrictions.
        NCHWPointwiseFwdGemm op;
        return op.IsSupportedArgument(&*arg.gemm);
    }

    bool IsSupportedArgument(const BaseArgument* p) override
    {
        const auto* arg = dynamic_cast<const Argument*>(p);
        return arg && IsSupportedArgument(*arg);
    }

    struct Invoker : BaseInvoker
    {
        float Run(const BaseArgument* p, const StreamConfig& config = StreamConfig{}) override
        {
            const auto* arg = dynamic_cast<const Argument*>(p);
            if(!arg || !IsSupportedArgument(*arg))
                throw std::runtime_error("Unsupported native NCHW pointwise forward argument");
            return NCHWPointwiseFwdGemm::MakeInvoker().Run(&*arg->gemm, config);
        }
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override
    {
        return "DeviceGroupedConvFwdNCHWPointwise<Wmma128x128x32>";
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* input,
                                                      const void* weights,
                                                      const std::array<const void*, 0>&,
                                                      void* output,
                                                      const Lengths& a,
                                                      const Lengths& as,
                                                      const Lengths& b,
                                                      const Lengths& bs,
                                                      const std::array<Lengths, 0>&,
                                                      const std::array<Lengths, 0>&,
                                                      const Lengths& e,
                                                      const Lengths& es,
                                                      const Filter& strides,
                                                      const Filter& dilation,
                                                      const Filter& left,
                                                      const Filter& right,
                                                      const PassThrough&,
                                                      const PassThrough&,
                                                      const PassThrough&) override
    {
        return std::make_unique<Argument>(
            input, weights, output, a, as, b, bs, e, es, strides, dilation, left, right);
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* input,
                                                      const void* weights,
                                                      const std::array<const void*, 0>& p_ds,
                                                      void* output,
                                                      const std::array<index_t, 5>& a,
                                                      const std::array<index_t, 5>& as,
                                                      const std::array<index_t, 5>& b,
                                                      const std::array<index_t, 5>& bs,
                                                      const std::array<std::array<index_t, 5>, 0>&,
                                                      const std::array<std::array<index_t, 5>, 0>&,
                                                      const std::array<index_t, 5>& e,
                                                      const std::array<index_t, 5>& es,
                                                      const std::array<index_t, 2>& strides,
                                                      const std::array<index_t, 2>& dilation,
                                                      const std::array<index_t, 2>& left,
                                                      const std::array<index_t, 2>& right,
                                                      const PassThrough& op_a,
                                                      const PassThrough& op_b,
                                                      const PassThrough& op_e) override
    {
        auto widen = [](const auto& src) {
            std::array<long_index_t, std::tuple_size_v<std::decay_t<decltype(src)>>> dst{};
            std::copy(src.begin(), src.end(), dst.begin());
            return dst;
        };
        return MakeArgumentPointer(input,
                                   weights,
                                   p_ds,
                                   output,
                                   widen(a),
                                   widen(as),
                                   widen(b),
                                   widen(bs),
                                   {},
                                   {},
                                   widen(e),
                                   widen(es),
                                   widen(strides),
                                   widen(dilation),
                                   widen(left),
                                   widen(right),
                                   op_a,
                                   op_b,
                                   op_e);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
