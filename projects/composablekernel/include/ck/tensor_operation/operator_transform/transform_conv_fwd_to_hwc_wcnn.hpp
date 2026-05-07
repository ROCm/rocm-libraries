
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"

namespace ck {
namespace tensor_operation {

template <index_t NDimSpatial,
          bool ShuffleOnLoad,
          bool Transposed,
          device::ConvolutionForwardSpecialization ConvForwardSpecialization>
struct TransformConvFwdToHWCWcnn
{
    static constexpr auto I0                = Number<0>{};
    static constexpr auto I1                = Number<1>{};
    static constexpr index_t Filter3PadSize = 2;

    template <typename ALayout,
              typename std::enable_if<NDimSpatial == 1 &&
                                          (is_same_v<ALayout, tensor_layout::convolution::G_NW_C> ||
                                           is_same_v<ALayout, tensor_layout::convolution::NWGC> ||
                                           is_same_v<ALayout, tensor_layout::convolution::GNWC>),
                                      bool>::type = false>
    static auto
    MakeADescriptor_H_W_C(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                          const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                          const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                          const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                          const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                          const std::array<index_t, NDimSpatial>& conv_filter_strides,
                          const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                          const std::array<index_t, NDimSpatial>& input_left_pads,
                          const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        const index_t N = a_g_n_c_wis_lengths[1];
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Wi = a_g_n_c_wis_lengths[3];

        const index_t Wo = c_g_n_k_wos_lengths[3];

        const index_t ConvStrideW = conv_filter_strides[0];

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // This is different
            const index_t WiStride = a_g_n_c_wis_strides[3 + NDimSpatial - 1];
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const auto CStride     = I1;

            return make_naive_tensor_descriptor(make_tuple(N, Wo, C),
                                                make_tuple(NStride, WiStride, CStride));
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t WiStride = a_g_n_c_wis_strides[3 + NDimSpatial - 1];
            const auto CStride     = I1;

            const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Wi, C), make_tuple(NStride, WiStride, CStride));

            return transform_tensor_descriptor(
                in_n_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else
        {
            static_assert(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Default);

            const index_t NStride     = a_g_n_c_wis_strides[1];
            const index_t WiStride    = a_g_n_c_wis_strides[3 + NDimSpatial - 1];
            const auto CStride        = I1;
            const index_t LeftPadW    = input_left_pads[0];
            const index_t RightPadW   = input_right_pads[0];
            const index_t DilationW   = conv_filter_dilations[0];
            const auto X              = b_g_k_c_xs_lengths[3];
            const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Wi, C), make_tuple(NStride, WiStride, CStride));

            const auto in_n_wip_c_desc = transform_tensor_descriptor(
                in_n_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, LeftPadW, RightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                in_n_wip_c_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(DilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            return transform_tensor_descriptor(
                in_n_x_wo_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pass_through_transform(Wo),
                           make_merge_transform(make_tuple(X, C))),
                make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
    }

#if defined(ENABLE_CONST_LAYOUT)
    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 2 && is_same_v<ALayout, tensor_layout::convolution::CONST_GNHWC>,
                  bool>::type = false>
    static auto
    MakeADescriptor_H_W_C(const std::array<index_t, NDimSpatial + 3>& /* a_g_n_c_wis_lengths */,
                          const std::array<index_t, NDimSpatial + 3>& /* a_g_n_c_wis_strides */,
                          const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_lengths */,
                          const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                          const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_lengths */,
                          const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                          const std::array<index_t, NDimSpatial>& /* conv_filter_strides */,
                          const std::array<index_t, NDimSpatial>& /* conv_filter_dilations */,
                          const std::array<index_t, NDimSpatial>& /* input_left_pads */,
                          const std::array<index_t, NDimSpatial>& /* input_right_pads */)
    {
        static_assert(ShuffleOnLoad == false, "");
        return make_naive_tensor_descriptor_packed(
            make_tuple(ALayout::N * ALayout::H, ALayout::W, ALayout::C));
    }
#endif

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 2 && (is_same_v<ALayout, tensor_layout::convolution::G_NHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNHWC>),
                  bool>::type = false>
    static auto
    MakeADescriptor_H_W_C(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                          const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                          const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_lengths */,
                          const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                          const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                          const std::array<index_t, NDimSpatial>& conv_filter_strides,
                          const std::array<index_t, NDimSpatial>& /* conv_filter_dilations */,
                          const std::array<index_t, NDimSpatial>& /* input_left_pads */,
                          const std::array<index_t, NDimSpatial>& /* input_right_pads */)
    {
        const index_t N = a_g_n_c_wis_lengths[1];
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Hi = a_g_n_c_wis_lengths[3];
        const index_t Wi = a_g_n_c_wis_lengths[4];

        const index_t Ho = c_g_n_k_wos_lengths[3];
        const index_t Wo = c_g_n_k_wos_lengths[4];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        // This is different
        const index_t NStride  = a_g_n_c_wis_strides[1];
        const index_t WiStride = a_g_n_c_wis_strides[3 + NDimSpatial - 1];
        const index_t HiStride = a_g_n_c_wis_strides[3 + NDimSpatial - 2];
        const auto CStride     = I1;

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // skip dimension g, n, k
            const index_t NHi =
                N * ck::accumulate_n<index_t>(
                        a_g_n_c_wis_lengths.begin() + 3, NDimSpatial - 1, 1, std::multiplies<>());

            return make_naive_tensor_descriptor(make_tuple(NHi, Wi, C),
                                                make_tuple(HiStride, WiStride, CStride));
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0)
        {
            const index_t NHo =
                N * ck::accumulate_n<index_t>(
                        c_g_n_k_wos_lengths.begin() + 3, NDimSpatial - 1, 1, std::multiplies<>());

            return make_naive_tensor_descriptor(make_tuple(NHo, Wo, C),
                                                make_tuple(HiStride, WiStride, CStride));
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
        {
            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Hi, Wi, C), make_tuple(NStride, HiStride, WiStride, CStride));
            const auto in_n_hip_wi_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, 0, Filter3PadSize),
                           make_pass_through_transform(Wi),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            return transform_tensor_descriptor(
                in_n_hip_wi_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Hi + Filter3PadSize)),
                           make_pass_through_transform(Wi),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter2x2Stride2Pad0)
        {
            if constexpr(Transposed || (ShuffleOnLoad == false))
            {
                // same with conv1, skip dimension g, n, k
                const index_t NHi = N * ck::accumulate_n<index_t>(a_g_n_c_wis_lengths.begin() + 3,
                                                                  NDimSpatial - 1,
                                                                  1,
                                                                  std::multiplies<>());

                return make_naive_tensor_descriptor(make_tuple(NHi, Wi, C),
                                                    make_tuple(HiStride, WiStride, CStride));
            }
            else // Transposed == false && ShuffleOnLoad
            {
                // skip dimension g, n, k
                const index_t NHi = N * ck::accumulate_n<index_t>(a_g_n_c_wis_lengths.begin() + 3,
                                                                  NDimSpatial - 1,
                                                                  1,
                                                                  std::multiplies<>());

                // This is different
                const auto in_nhi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(NHi, Wi, C), make_tuple(HiStride, WiStride, CStride));
                const auto in_nhi_wi_c_unmerge_desc = transform_tensor_descriptor(
                    in_nhi_wi_c_desc,
                    make_tuple(make_unmerge_transform(make_tuple(NHi / 2, Number<2>{})),
                               make_unmerge_transform(make_tuple(Wi / 2, Number<2>{})),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    in_nhi_wi_c_unmerge_desc,
                    make_tuple(make_pass_through_transform(NHi / 2),
                               make_pass_through_transform(Wi / 2),
                               make_merge_transform(make_tuple(Number<2>{}, Number<2>{}, C))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3, 4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0)
        {
            if constexpr(Transposed)
            {
                // same with conv1, skip dimension g, n, k
                const index_t NHi = N * ck::accumulate_n<index_t>(a_g_n_c_wis_lengths.begin() + 3,
                                                                  NDimSpatial - 1,
                                                                  1,
                                                                  std::multiplies<>());

                return make_naive_tensor_descriptor(make_tuple(NHi, Wi, C),
                                                    make_tuple(HiStride, WiStride, CStride));
            }
            else if constexpr(ShuffleOnLoad == false)
            {
                // Pad height to even before merge N
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N, Hi, Wi, C), make_tuple(NStride, HiStride, WiStride, CStride));

                const auto in_n_hip_wi_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pad_transform(Hi, 0, Hi & 1),
                               make_pass_through_transform(Wi),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
                const index_t HiP = Hi + (Hi & 1);
                return transform_tensor_descriptor(
                    in_n_hip_wi_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N, HiP)),
                               make_pass_through_transform(Wi),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
            else // transposed = false && shuffleOnLoad == true
            {
                // This is different
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N, Hi, Wi, C), make_tuple(NStride, HiStride, WiStride, CStride));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pad_transform(Hi, 0, Hi & 1),
                               make_pad_transform(Wi, 0, Wi & 1),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const index_t HiP                      = Hi + (Hi & 1);
                const index_t WiP                      = Wi + (Wi & 1);
                const auto in_n_hip_wip_c_unmerge_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_unmerge_transform(make_tuple(HiP / 2, Number<2>{})),
                               make_unmerge_transform(make_tuple(WiP / 2, Number<2>{})),
                               make_pass_through_transform(C)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}, Sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_hip_wip_c_unmerge_desc,
                    make_tuple(make_merge_transform(make_tuple(N, HiP / 2)),
                               make_pass_through_transform(WiP / 2),
                               make_merge_transform(make_tuple(Number<2>{}, Number<2>{}, C))),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // This is different
            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Hi, Wi, C), make_tuple(NStride, HiStride, WiStride, CStride));

            const auto in_n_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            return transform_tensor_descriptor(
                in_n_ho_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Ho)),
                           make_pass_through_transform(Wo),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else
        {
            static_assert(0, "not supported!!");
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 3 && (is_same_v<ALayout, tensor_layout::convolution::G_NDHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NDHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNDHWC>),
                  bool>::type = false>
    static auto
    MakeADescriptor_H_W_C(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                          const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                          const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                          const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                          const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                          const std::array<index_t, NDimSpatial>& conv_filter_strides,
                          const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                          const std::array<index_t, NDimSpatial>& input_left_pads,
                          const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        const index_t N = a_g_n_c_wis_lengths[1];
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Di = a_g_n_c_wis_lengths[3];
        const index_t Hi = a_g_n_c_wis_lengths[4];
        const index_t Wi = a_g_n_c_wis_lengths[5];

        const index_t NStride  = a_g_n_c_wis_strides[1];
        const index_t DiStride = a_g_n_c_wis_strides[3];
        const index_t HiStride = a_g_n_c_wis_strides[4];
        const index_t WiStride = a_g_n_c_wis_strides[5];
        const auto CStride     = I1;

        const index_t Do = c_g_n_k_wos_lengths[3];
        const index_t Ho = c_g_n_k_wos_lengths[4];
        const index_t Wo = c_g_n_k_wos_lengths[5];

        const auto Z              = b_g_k_c_xs_lengths[3];
        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];
        const index_t LeftPadD    = input_left_pads[0];
        const index_t RightPadD   = input_right_pads[0];
        const index_t DilationD   = conv_filter_dilations[0];

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // skip dimension g, n, k
            const index_t NDoHo =
                N * ck::accumulate_n<index_t>(
                        c_g_n_k_wos_lengths.begin() + 3, NDimSpatial - 1, 1, std::multiplies<>());

            return make_naive_tensor_descriptor(make_tuple(NDoHo, Wo, C),
                                                make_tuple(HiStride, WiStride, CStride));
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
        {
            const auto HiP                  = Hi + Filter3PadSize;
            const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, CStride));

            const auto in_n_dip_hip_wi_c_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, LeftPadD, RightPadD),
                           make_pad_transform(Hi, 0, 2),
                           make_pass_through_transform(Wi),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_z_do_hip_wi_c_desc = transform_tensor_descriptor(
                in_n_dip_hip_wi_c_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Z, Do), make_tuple(DilationD, ConvStrideD)),
                    make_pass_through_transform(HiP),
                    make_pass_through_transform(Wi),
                    make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

            return transform_tensor_descriptor(
                in_n_z_do_hip_wi_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, HiP)),
                           make_pass_through_transform(Wi),
                           make_merge_transform(make_tuple(Z, C))),
                make_tuple(Sequence<0, 2, 3>{}, Sequence<4>{}, Sequence<1, 5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, CStride));

            const auto in_n_dip_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, LeftPadD, RightPadD),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_z_do_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_dip_ho_wo_c_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Z, Do), make_tuple(DilationD, ConvStrideD)),
                    make_pass_through_transform(Ho),
                    make_pass_through_transform(Wo),
                    make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

            return transform_tensor_descriptor(
                in_n_z_do_ho_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho)),
                           make_pass_through_transform(Wo),
                           make_merge_transform(make_tuple(Z, C))),
                make_tuple(Sequence<0, 2, 3>{}, Sequence<4>{}, Sequence<1, 5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else
        {
            static_assert(0, "not supported!!");
        }
    }

#if defined(ENABLE_CONST_LAYOUT)
    template <
        typename BLayout,
        typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::CONST_GKYXC<1>> ||
                                    is_same_v<BLayout, tensor_layout::convolution::CONST_GKYXC<3>>,
                                bool>::type = false>
    static auto
    MakeBDescriptor_K_YX_C(const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_lengths */,
                           const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */)
    {
        static_assert(ShuffleOnLoad == false, "");
        return make_naive_tensor_descriptor_packed(
            make_tuple(BLayout::K, BLayout::X * BLayout::Y, BLayout::C));
    }

    template <
        typename BLayout,
        typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::CONST_GKYXC<2>>,
                                bool>::type = false>
    static auto
    MakeBDescriptor_K_YX_C(const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_lengths */,
                           const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */)
    {
        static_assert(ShuffleOnLoad == false, "");
        return make_naive_tensor_descriptor_packed(
            make_tuple(BLayout::K, Number<1>{}, BLayout::X * BLayout::Y * BLayout::C));
    }
#endif

    template <typename BLayout,
              typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::GKXC> ||
                                          is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                                          is_same_v<BLayout, tensor_layout::convolution::GKZYXC>,
                                      bool>::type = false>
    static auto
    MakeBDescriptor_K_YX_C(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                           const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */)
    {
        const index_t K = b_g_k_c_xs_lengths[1];
        const index_t C = b_g_k_c_xs_lengths[2];
        if constexpr(NDimSpatial == 1)
        {
            if constexpr(ConvForwardSpecialization ==
                         device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
            {
                const auto wei_k_x_c_desc =
                    make_naive_tensor_descriptor_packed(make_tuple(K, I1, C));
                return wei_k_x_c_desc;
            }
            else
            {
                const auto X = b_g_k_c_xs_lengths[3];
                const auto wei_k_x_c_desc =
                    make_naive_tensor_descriptor_packed(make_tuple(K, X, C));
                return transform_tensor_descriptor(
                    wei_k_x_c_desc,
                    make_tuple(make_pass_through_transform(K),
                               make_insert_transform(I0),
                               make_merge_transform(make_tuple(X, C))),
                    make_tuple(Sequence<0>{}, Sequence<>{}, Sequence<1, 2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            if constexpr((ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter2x2Stride2Pad0) ||
                         (ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0))
            {
                constexpr auto NumYX = Number<4>{};
                if constexpr(ShuffleOnLoad == false) // do emulation in block level.
                {
                    const auto wei_k_yx_c_desc =
                        make_naive_tensor_descriptor_packed(make_tuple(K, NumYX, C));
                    return wei_k_yx_c_desc;
                }
                else if constexpr(Transposed == false)
                {
                    const auto wei_k_yx_c_desc =
                        make_naive_tensor_descriptor_packed(make_tuple(K, NumYX, C));
                    return transform_tensor_descriptor(
                        wei_k_yx_c_desc,
                        make_tuple(make_pass_through_transform(K),
                                   make_insert_transform(I0),
                                   make_merge_transform(make_tuple(NumYX, C))),
                        make_tuple(Sequence<0>{}, Sequence<>{}, Sequence<1, 2>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
                }
                else
                {
                    const auto wei_k_yx_c_desc =
                        make_naive_tensor_descriptor_packed(make_tuple(K, NumYX, C));
                    return transform_tensor_descriptor(
                        wei_k_yx_c_desc,
                        make_tuple(make_merge_transform(make_tuple(NumYX, K)),
                                   make_insert_transform(I0),
                                   make_pass_through_transform(C)),
                        make_tuple(Sequence<1, 0>{}, Sequence<>{}, Sequence<2>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
                }
            }
            else
            {
                const index_t YX =
                    (ConvForwardSpecialization ==
                         device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0 ||
                     ConvForwardSpecialization ==
                         device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0)
                        ? 9
                        : 1;
                const auto wei_k_yx_c_desc =
                    make_naive_tensor_descriptor_packed(make_tuple(K, YX, C));
                return wei_k_yx_c_desc;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            const index_t YX =
                (ConvForwardSpecialization ==
                 device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
                    ? 9
                    : 1;
            const index_t Z = (ConvForwardSpecialization ==
                               device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
                                  ? 1
                                  : b_g_k_c_xs_lengths[3];
            const auto wei_k_z_yx_c_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, Z, YX, C));
            const auto wei_k_yx_zc_desc = transform_tensor_descriptor(
                wei_k_z_yx_c_desc,
                make_tuple(make_pass_through_transform(K),
                           make_pass_through_transform(YX),
                           make_merge_transform(make_tuple(Z, C))),
                make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            return wei_k_yx_zc_desc;
        }
    }

    template <
        typename BLayout,
        typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::G_K_X_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::G_K_YX_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::G_K_ZYX_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KXGC> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KYXGC> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KZYXGC>,
                                bool>::type = false>
    static auto
    MakeBDescriptor_K_YX_C(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                           const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides)
    {
        const index_t K = b_g_k_c_xs_lengths[1];
        const index_t C = b_g_k_c_xs_lengths[2];

        const index_t KStride = b_g_k_c_xs_strides[1];
        const index_t XStride = b_g_k_c_xs_strides[2 + NDimSpatial];
        const auto CStride    = I1;
        if constexpr(NDimSpatial == 1)
        {
            if constexpr(ConvForwardSpecialization ==
                         device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
            {
                const auto wei_k_x_c_desc = make_naive_tensor_descriptor(
                    make_tuple(K, I1, C), make_tuple(KStride, XStride, CStride));
                return wei_k_x_c_desc;
            }
            else
            {
                const auto X              = b_g_k_c_xs_lengths[3];
                const auto wei_k_x_c_desc = make_naive_tensor_descriptor(
                    make_tuple(K, X, C), make_tuple(KStride, XStride, CStride));
                return transform_tensor_descriptor(
                    wei_k_x_c_desc,
                    make_tuple(make_pass_through_transform(K),
                               make_insert_transform(I0),
                               make_merge_transform(make_tuple(X, C))),
                    make_tuple(Sequence<0>{}, Sequence<>{}, Sequence<1, 2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            if constexpr((ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter2x2Stride2Pad0) ||
                         (ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0))

            {
                constexpr auto NumYX = Number<4>{};
                if constexpr(ShuffleOnLoad == false) // do emulation in block level.
                {
                    const auto wei_k_yx_c_desc = make_naive_tensor_descriptor(
                        make_tuple(K, NumYX, C), make_tuple(KStride, XStride, CStride));
                    return wei_k_yx_c_desc;
                }
                else if constexpr(Transposed == false)
                {
                    const auto wei_k_yx_c_desc = make_naive_tensor_descriptor(
                        make_tuple(K, NumYX, C), make_tuple(KStride, XStride, CStride));
                    return transform_tensor_descriptor(
                        wei_k_yx_c_desc,
                        make_tuple(make_pass_through_transform(K),
                                   make_insert_transform(I0),
                                   make_merge_transform(make_tuple(NumYX, C))),
                        make_tuple(Sequence<0>{}, Sequence<>{}, Sequence<1, 2>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
                }
                else
                {
                    const auto wei_k_yx_c_desc = make_naive_tensor_descriptor(
                        make_tuple(K, NumYX, C), make_tuple(KStride, XStride, CStride));
                    return transform_tensor_descriptor(
                        wei_k_yx_c_desc,
                        make_tuple(make_merge_transform(make_tuple(NumYX, K)),
                                   make_insert_transform(I0),
                                   make_pass_through_transform(C)),
                        make_tuple(Sequence<1, 0>{}, Sequence<>{}, Sequence<2>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
                }
            }
            else
            {
                const index_t YX = ck::accumulate_n<index_t>(
                    b_g_k_c_xs_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());
                const auto wei_k_yx_c_desc = make_naive_tensor_descriptor(
                    make_tuple(K, YX, C), make_tuple(KStride, XStride, CStride));
                return wei_k_yx_c_desc;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            const index_t YX =
                (ConvForwardSpecialization ==
                 device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
                    ? 9
                    : 1;
            const index_t Z              = (ConvForwardSpecialization ==
                               device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
                                               ? 1
                                               : b_g_k_c_xs_lengths[3];
            const index_t ZStride        = b_g_k_c_xs_strides[3];
            const auto wei_k_yx_z_c_desc = make_naive_tensor_descriptor(
                make_tuple(K, YX, Z, C), make_tuple(KStride, XStride, ZStride, CStride));
            const auto wei_k_yx_zc_desc = transform_tensor_descriptor(
                wei_k_yx_z_c_desc,
                make_tuple(make_pass_through_transform(K),
                           make_pass_through_transform(YX),
                           make_merge_transform(make_tuple(Z, C))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            return wei_k_yx_zc_desc;
        }
    }

#if defined(ENABLE_CONST_LAYOUT)
    template <typename CLayout,
              typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::CONST_GNHWK>,
                                      bool>::type = false>
    static auto
    MakeCDescriptor_H_W_K(const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_lengths */,
                          const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */)
    {
        static_assert(ShuffleOnLoad == false, "");
        return make_naive_tensor_descriptor_packed(
            make_tuple(CLayout::N * CLayout::H, CLayout::W, CLayout::K));
    }
#endif

    template <
        typename CLayout,
        typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::G_NW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::G_NHW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::G_NDHW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NWGK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NHWGK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NDHWGK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::GNWK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::GNHWK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::GNDHWK>,
                                bool>::type = false>
    static auto
    MakeCDescriptor_H_W_K(const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_strides)
    {
        const index_t N  = c_g_n_k_wos_lengths[1];
        const index_t K  = c_g_n_k_wos_lengths[2];
        const index_t Do = c_g_n_k_wos_lengths[NDimSpatial];
        const index_t Ho = c_g_n_k_wos_lengths[NDimSpatial + 1];
        const index_t Wo = c_g_n_k_wos_lengths[NDimSpatial + 2];

        const auto KStride     = I1;
        const index_t NStride  = c_g_n_k_wos_strides[1];
        const index_t DoStride = c_g_n_k_wos_strides[NDimSpatial];
        const index_t WoStride = c_g_n_k_wos_strides[NDimSpatial + 2];
        const index_t HoStride = c_g_n_k_wos_strides[NDimSpatial + 1];

        constexpr bool isNativePacked = is_same_v<CLayout, tensor_layout::convolution::GNWK> ||
                                        is_same_v<CLayout, tensor_layout::convolution::GNHWK> ||
                                        is_same_v<CLayout, tensor_layout::convolution::GNDHWK>;
        const index_t NHo =
            N * ck::accumulate_n<index_t>(
                    c_g_n_k_wos_lengths.begin() + 3, NDimSpatial - 1, 1, std::multiplies<>());

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
        {
            if constexpr(NDimSpatial == 1)
            {
                const auto out_ho_wo_k_desc = make_naive_tensor_descriptor(
                    make_tuple(N, I1, Wo, K), make_tuple(NStride, WoStride, WoStride, KStride));
                const auto in_n_hop_wo_k_desc = transform_tensor_descriptor(
                    out_ho_wo_k_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pad_transform(I1, 0, Filter3PadSize),
                               make_pass_through_transform(Wo),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_hop_wo_k_desc,
                    make_tuple(make_merge_transform(make_tuple(N, Number<1 + Filter3PadSize>{})),
                               make_pass_through_transform(Wo),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
            else if constexpr(NDimSpatial == 2)
            {
                const auto out_ho_wo_k_desc = make_naive_tensor_descriptor(
                    make_tuple(N, Ho, Wo, K), make_tuple(NStride, HoStride, WoStride, KStride));
                const auto in_n_hop_wo_k_desc = transform_tensor_descriptor(
                    out_ho_wo_k_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pad_transform(Ho, 0, Filter3PadSize),
                               make_pass_through_transform(Wo),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_hop_wo_k_desc,
                    make_tuple(make_merge_transform(make_tuple(N, Ho + Filter3PadSize)),
                               make_pass_through_transform(Wo),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
            else if constexpr(NDimSpatial == 3)
            {
                const auto out_n_do_ho_wo_k_desc = make_naive_tensor_descriptor(
                    make_tuple(N, Do, Ho, Wo, K),
                    make_tuple(NStride, DoStride, HoStride, WoStride, KStride));
                const auto out_n_do_hop_wo_k_desc = transform_tensor_descriptor(
                    out_n_do_ho_wo_k_desc,
                    make_tuple(make_pass_through_transform(N),
                               make_pass_through_transform(Do),
                               make_pad_transform(Ho, 0, Filter3PadSize),
                               make_pass_through_transform(Wo),
                               make_pass_through_transform(K)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    out_n_do_hop_wo_k_desc,
                    make_tuple(make_merge_transform(make_tuple(N, Do, Ho + Filter3PadSize)),
                               make_pass_through_transform(Wo),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
        }
        else
        {
            if constexpr(ShuffleOnLoad && Transposed)
            {
                auto out_nho_wo_k_desc = make_naive_tensor_descriptor(
                    make_tuple(NHo, Wo, K), make_tuple(HoStride, WoStride, KStride));

                const auto out_nho_wo_k_unmerge_desc = transform_tensor_descriptor(
                    out_nho_wo_k_desc,
                    make_tuple(make_unmerge_transform(make_tuple(NHo / 2, Number<2>{})),
                               make_unmerge_transform(make_tuple(Wo / 2, Number<2>{})),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    out_nho_wo_k_unmerge_desc,
                    make_tuple(make_pass_through_transform(NHo / 2),
                               make_pass_through_transform(Wo / 2),
                               make_merge_transform(make_tuple(Number<2>{}, Number<2>{}, K))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3, 4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
            else
            {
                if constexpr(isNativePacked)
                {
                    return make_naive_tensor_descriptor_packed(make_tuple(NHo, Wo, K));
                }
                else
                {
                    if constexpr(NDimSpatial == 1)
                    {
                        return make_naive_tensor_descriptor(make_tuple(NHo, Wo, K),
                                                            make_tuple(NStride, WoStride, KStride));
                    }
                    else
                    {
                        return make_naive_tensor_descriptor(
                            make_tuple(NHo, Wo, K), make_tuple(HoStride, WoStride, KStride));
                    }
                }
            }
        }
    }

    // for output bias
    template <typename CLayout,
              typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::G_K>,
                                      bool>::type = false>
    static auto
    MakeCDescriptor_H_W_K(const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_strides)
    {
        const index_t N       = c_g_n_k_wos_lengths[1];
        const index_t K       = c_g_n_k_wos_lengths[2];
        const index_t KStride = c_g_n_k_wos_strides[2];
        const index_t Do      = c_g_n_k_wos_lengths[NDimSpatial];
        const index_t Ho      = c_g_n_k_wos_lengths[NDimSpatial + 1];
        const index_t Wo      = c_g_n_k_wos_lengths[NDimSpatial + 2];

        const index_t NHo =
            N * ck::accumulate_n<index_t>(
                    c_g_n_k_wos_lengths.begin() + 3, NDimSpatial - 1, 1, std::multiplies<>());

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
        {
            if constexpr(NDimSpatial == 1)
            {
                return make_naive_tensor_descriptor(make_tuple(N * (Filter3PadSize + 1), Wo, K),
                                                    make_tuple(I0, I0, KStride));
            }
            else if constexpr(NDimSpatial == 2)
            {
                return make_naive_tensor_descriptor(make_tuple(N * (Ho + Filter3PadSize), Wo, K),
                                                    make_tuple(I0, I0, KStride));
            }
            else if constexpr(NDimSpatial == 3)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(N * Do * (Ho + Filter3PadSize), Wo, K), make_tuple(I0, I0, KStride));
            }
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(NHo, Wo, K),
                                                make_tuple(I0, I0, KStride));
        }
    }
};

} // namespace tensor_operation
} // namespace ck
