// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <type_traits>
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

namespace native_output {
namespace cv = tensor_layout::convolution;
using PT     = element_wise::PassThrough;
template <index_t... Is>
using S = Sequence<Is...>;

// These are the tested CTranspose configurations, not the input-transpose implementation.
// clang-format off
template <index_t D, typename T, index_t M, index_t N, index_t K, index_t V>
using Wmma = DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<D,
    std::conditional_t<D == 2, cv::NHWGC, cv::NDHWGC>,
    std::conditional_t<D == 2, cv::GKYXC, cv::GKZYXC>, Tuple<>,
    std::conditional_t<D == 2, cv::NGKHW, cv::NGKDHW>,
    T, T, float, T, Tuple<>, T, PT, PT, PT,
    ConvolutionForwardSpecialization::Default, GemmSpecialization::MNKPadding,
    128, M, N, K, 8, 8, 16, 16, M / 32, N / 32,
    S<4,32,1>, S<1,0,2>, S<1,0,2>, 2, 8, 8, 1,
    S<4,32,1>, S<1,0,2>, S<1,0,2>, 2, 8, 8, 1,
    1, 1, S<1,32,1,4>, V,
    BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1>;

template <index_t D, typename T, index_t M, index_t N, index_t K, index_t V>
using Xdl = DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<D,
    std::conditional_t<D == 2, cv::NHWGC, cv::NDHWGC>,
    std::conditional_t<D == 2, cv::GKYXC, cv::GKZYXC>, Tuple<>,
    std::conditional_t<D == 2, cv::NGKHW, cv::NGKDHW>,
    T, T, float, T, Tuple<>, T, PT, PT, PT,
    ConvolutionForwardSpecialization::Default, GemmSpecialization::MNKPadding,
    1, 128, M, N, K, 8, 8, 16, 16, M / 32, N / 32,
    S<4,32,1>, S<1,0,2>, S<1,0,2>, 2, 8, 8, 1,
    S<4,32,1>, S<1,0,2>, S<1,0,2>, 2, 8, 8, 1,
    1, 1, S<1,16,1,8>, V>;
// clang-format on

template <index_t D, typename T, index_t M, index_t N, index_t K, index_t V>
using TestedOp =
    std::conditional_t<std::is_same_v<T, float>, Xdl<D, T, M, N, K, V>, Wmma<D, T, M, N, K, V>>;

template <index_t D, typename T, index_t M, index_t N, index_t K, index_t V>
struct Candidate : TestedOp<D, T, M, N, K, V>
{
    using Base     = TestedOp<D, T, M, N, K, V>;
    using Argument = typename Base::Argument;

    // Require every stride, including singleton strides, to match the converted packed tensors.
    static bool IsPacked(const std::array<index_t, D + 3>& lengths,
                         const std::array<index_t, D + 3>& strides,
                         const std::array<index_t, D + 3>& fastest_first)
    {
        long_index_t size = 1;
        for(auto axis : fastest_first)
        {
            if(lengths[axis] <= 0 || strides[axis] != size ||
               size > std::numeric_limits<index_t>::max() / lengths[axis])
                return false;
            size *= lengths[axis];
        }
        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        if(!ck::is_gfx125_supported())
            return false;
        const auto* arg = dynamic_cast<const Argument*>(p_arg);
        if(arg == nullptr)
            return false;
        std::array<index_t, D + 3> a_order{}, b_order{}, e_order{};
        a_order[0] = 2;
        a_order[1] = 0;
        b_order[0] = 2;
        for(index_t i = 0; i < D; ++i)
        {
            a_order[i + 2] = D + 2 - i;
            b_order[i + 1] = D + 2 - i;
            e_order[i]     = D + 2 - i;
        }
        a_order[D + 2] = 1;
        b_order[D + 1] = 1;
        b_order[D + 2] = 0;
        e_order[D]     = 2;
        e_order[D + 1] = 0;
        e_order[D + 2] = 1;
        if(!IsPacked(arg->a_g_n_c_wis_lengths_, arg->a_g_n_c_wis_strides_, a_order) ||
           !IsPacked(arg->b_g_k_c_xs_lengths_, arg->b_g_k_c_xs_strides_, b_order) ||
           !IsPacked(arg->e_g_n_k_wos_lengths_, arg->e_g_n_k_wos_strides_, e_order))
            return false;
        const auto& a = arg->a_g_n_c_wis_lengths_;
        const auto& b = arg->b_g_k_c_xs_lengths_;
        const auto& e = arg->e_g_n_k_wos_lengths_;
        if(a[0] != b[0] || a[0] != e[0] || a[1] != e[1] || a[2] != b[2] || b[1] != e[2] ||
           a[2] % 8 != 0 || b[1] % 4 != 0)
            return false;
        long_index_t output_spatial = 1;
        bool nonpointwise           = false;
        for(index_t i = 0; i < D; ++i)
        {
            nonpointwise |= b[i + 3] != 1;
            output_spatial *= e[i + 3];
            if(arg->conv_filter_strides_[i] <= 0 || arg->conv_filter_dilations_[i] <= 0 ||
               arg->input_left_pads_[i] < 0 || arg->input_right_pads_[i] < 0)
                return false;
            const long_index_t padded = static_cast<long_index_t>(a[i + 3]) +
                                        arg->input_left_pads_[i] + arg->input_right_pads_[i];
            const long_index_t filter =
                static_cast<long_index_t>(arg->conv_filter_dilations_[i]) * (b[i + 3] - 1) + 1;
            if(padded < filter || (padded - filter) / arg->conv_filter_strides_[i] + 1 != e[i + 3])
                return false;
        }
        return nonpointwise && output_spatial % V == 0 && Base::IsSupportedArgument(*arg);
    }

    std::string GetTypeString() const override { return "NativeOutput:" + Base::GetTypeString(); }
};

} // namespace native_output
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
