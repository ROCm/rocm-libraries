// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"

namespace ck {
namespace tensor_operation {

// A patch row is contiguous only along XC, not along Y or the reduction axis.
// This view is used only after the device operation checks packed NHWC tensors,
// exact nonoverlap geometry, unit dilation, and zero padding.
template <index_t NDimSpatial,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t GemmK1Number,
          index_t K0PerBlock,
          index_t NumGroupsToMerge,
          device::ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization>
struct TransformConvBwdWeightToGemmNonOverlapPatch
{
    static_assert(GemmK1Number > 0 && K0PerBlock > 0);

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const index_t N,
        const index_t K,
        const index_t C,
        const std::array<index_t, NDimSpatial>&,
        const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
        const std::array<index_t, NDimSpatial>& output_spatial_lengths,
        const std::array<index_t, NDimSpatial + 3>& input_strides,
        const std::array<index_t, NDimSpatial + 3>& weights_strides,
        const std::array<index_t, NDimSpatial + 3>& output_strides,
        const std::array<index_t, NDimSpatial>&,
        const std::array<index_t, NDimSpatial>&,
        const std::array<index_t, NDimSpatial>&,
        const std::array<index_t, NDimSpatial>&,
        const index_t batch_k)
    {
        static_assert(NDimSpatial == 2 && NumGroupsToMerge == 1);
        const auto Ho   = output_spatial_lengths[0];
        const auto Wo   = output_spatial_lengths[1];
        const auto Y    = filter_spatial_lengths[0];
        const auto X    = filter_spatial_lengths[1];
        const auto R    = N * Ho * Wo;
        const auto Q    = Y * X * C;
        const auto PadM = math::integer_divide_ceil(K, MPerBlock) * MPerBlock - K;
        const auto PadN = math::integer_divide_ceil(Q, NPerBlock) * NPerBlock - Q;
        const auto K0 =
            math::integer_divide_ceil(R, GemmK1Number * K0PerBlock * batch_k) * K0PerBlock;
        const auto RPad = batch_k * K0 * GemmK1Number;

        const auto a_r_m = make_naive_tensor_descriptor(make_tuple(R, K),
                                                        make_tuple(output_strides[4], Number<1>{}));
        const auto a_pad = transform_tensor_descriptor(
            a_r_m,
            make_tuple(make_right_pad_transform(R, RPad - R), make_right_pad_transform(K, PadM)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto a_k0_m_k1 = transform_tensor_descriptor(
            a_pad,
            make_tuple(make_unmerge_transform(make_tuple(batch_k * K0, Number<GemmK1Number>{})),
                       make_pass_through_transform(K + PadM)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        // n*image_stride + ho*Y*row_stride + wo*X*pixel_stride + y*row_stride + xc.
        // In particular, advancing Y must skip the rest of the image row, and
        // advancing the patch/reduction axis must not flatten the image payload.
        const auto b_n_ho_wo_y_xc = make_naive_tensor_descriptor(make_tuple(N, Ho, Wo, Y, X * C),
                                                                 make_tuple(input_strides[1],
                                                                            Y * input_strides[3],
                                                                            X * input_strides[4],
                                                                            input_strides[3],
                                                                            Number<1>{}));
        const auto b_r_q =
            transform_tensor_descriptor(b_n_ho_wo_y_xc,
                                        make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                   make_merge_transform(make_tuple(Y, X * C))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto b_pad = transform_tensor_descriptor(
            b_r_q,
            make_tuple(make_right_pad_transform(R, RPad - R), make_right_pad_transform(Q, PadN)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
        const auto b_k0_n_k1 = transform_tensor_descriptor(
            b_pad,
            make_tuple(make_unmerge_transform(make_tuple(batch_k * K0, Number<GemmK1Number>{})),
                       make_pass_through_transform(Q + PadN)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        const auto c_m_n = make_naive_tensor_descriptor(
            make_tuple(K, Q), make_tuple(weights_strides[1], Number<1>{}));
        const auto c_pad = transform_tensor_descriptor(
            c_m_n,
            make_tuple(make_right_pad_transform(K, PadM), make_right_pad_transform(Q, PadN)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
        return make_tuple(a_k0_m_k1, b_k0_n_k1, c_pad);
    }
};

} // namespace tensor_operation
} // namespace ck
