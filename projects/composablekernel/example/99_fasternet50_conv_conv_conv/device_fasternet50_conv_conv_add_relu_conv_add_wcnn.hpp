// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "gridwise_fasternet50_wcnn.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_hwc_wcnn.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename DsLayout,
          typename ELayout,
          typename InDataType,
          typename WeiDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
          index_t NumPrefetch,
          index_t BlockSize,
          index_t HPerBlock,
          index_t WPerBlock,
          typename CPerBlock,
          typename KPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWcnn,
          index_t WPerWcnn,
          typename FilterSize,
          typename DilationX,
          typename DilationY,
          typename InBlockTransferThreadClusterLengths,
          index_t InBlockTransferSrcScalarPerVector,
          index_t InBlockTransferDstScalarPerVector,
          bool InEnableLds,
          bool InBlockLdsAddExtraM,
          bool InTileLoad,
          typename WeiBlockTransferThreadClusterLengths,
          index_t WeiBlockTransferSrcScalarPerVector,
          index_t WeiBlockTransferDstScalarPerVector,
          bool WeiEnableLds,
          bool WeiBlockLdsAddExtraM,
          bool WeiTileLoad,
          typename DsBlockTransferThreadClusterLengths,
          typename DsBlockTransferSrcScalarPerVector,
          typename DsBlockTransferDstScalarPerVector,
          bool DsEnableLds,
          bool DsBlockLdsAddExtraM,
          typename AccBlockTransferClusterLengths,
          index_t AccBlockTransferScalarPerVector,
          bool AccEnableLds,
          bool EnableAsync,
          bool EnableWaveGroup,
          bool EnableSpatialCluster = false,
          index_t ClusterDimSize    = 0,
          bool ShuffleOnLoad        = false,
          bool Transposed           = false,
          bool TileStore            = false>
struct DeviceFasternet50_Wcnn_CShuffle
{
    using DeviceOp = DeviceFasternet50_Wcnn_CShuffle;

    static constexpr auto ConvSpecs_1x1 =
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

    static constexpr auto ConvSpecs_3x3 = ck::tensor_operation::device::
        ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0;

    static constexpr auto conv_to_hwc_wcnn_1x1_transformer =
        TransformConvFwdToHWCWcnn<NDimSpatial, ShuffleOnLoad, Transposed, ConvSpecs_1x1>{};

    static constexpr auto conv_to_hwc_wcnn_3x3_transformer =
        TransformConvFwdToHWCWcnn<NDimSpatial, ShuffleOnLoad, Transposed, ConvSpecs_3x3>{};

    static constexpr auto conv_to_hwc_wcnn_transformer =
        make_tuple(conv_to_hwc_wcnn_3x3_transformer,
                   conv_to_hwc_wcnn_1x1_transformer,
                   conv_to_hwc_wcnn_1x1_transformer);

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    static constexpr index_t NumDTensor_0 = tuple_element_t<0, DsDataType>::Size();
    static constexpr index_t NumDTensor_1 = tuple_element_t<1, DsDataType>::Size();
    static constexpr index_t NumDTensor_2 = tuple_element_t<2, DsDataType>::Size();

    static constexpr index_t conv_num = 3;

    // Describe how data read from Global memory
    static auto MakeInGridDescriptor(
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_strides,
        const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_dilations,
        const std::array<std::array<index_t, NDimSpatial>, 3>& input_left_pads,
        const std::array<std::array<index_t, NDimSpatial>, 3>& input_right_pads)
    {
        // H W C
        static constexpr index_t conv_phase = 3;
        return generate_tuple(
            [&](auto i) {
                const auto in_data_raw_desc =
                    conv_to_hwc_wcnn_transformer[Number<i>{}]
                        .template MakeADescriptor_H_W_C<InLayout>(a_g_n_c_wis_lengths,
                                                                  a_g_n_c_wis_strides,
                                                                  b_g_k_c_xs_lengths[i],
                                                                  b_g_k_c_xs_strides[i],
                                                                  e_g_n_k_wos_lengths,
                                                                  e_g_n_k_wos_strides,
                                                                  conv_filter_strides[i],
                                                                  conv_filter_dilations[i],
                                                                  input_left_pads[i],
                                                                  input_right_pads[i]);
                const auto in_data_desc =
                    PadTensorDescriptor(in_data_raw_desc,
                                        Sequence<HPerBlock, WPerBlock, CPerBlock::At(i)>{},
                                        Sequence<true, true, true>{});
                return in_data_desc;
            },
            Number<conv_phase>{});
    }

    static auto MakeWeiGridDescriptor(
        const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_strides)
    {
        static constexpr index_t conv_phase = 3;
        return generate_tuple(
            [&](auto i) {
                const auto wei_data_raw_desc =
                    conv_to_hwc_wcnn_transformer[Number<i>{}]
                        .template MakeBDescriptor_K_YX_C<WeiLayout>(b_g_k_c_xs_lengths[i],
                                                                    b_g_k_c_xs_strides[i]);
                const auto wei_data_desc =
                    PadTensorDescriptor(wei_data_raw_desc,
                                        make_tuple(KPerBlock::At(i), 1, CPerBlock::At(i)),
                                        Sequence<true, false, true>{});
                return wei_data_desc;
            },
            Number<conv_phase>{});
    }

    template <typename OutLayout_, index_t conv_phase = 2>
    static auto
    MakeOutGridDescriptor(const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides)
    {
        const auto acc_data_raw_desc = conv_to_hwc_wcnn_transformer[Number<conv_phase>{}]
                                           .template MakeCDescriptor_H_W_K<OutLayout_>(
                                               e_g_n_k_wos_lengths, e_g_n_k_wos_strides);
        constexpr index_t HPerBlockOut = HPerBlock;
        constexpr index_t WPerBlockOut = WPerBlock;
        const auto acc_data_desc =
            PadTensorDescriptor(acc_data_raw_desc,
                                make_tuple(HPerBlockOut, WPerBlockOut, KPerBlock::At(conv_phase)),
                                Sequence<true, true, true>{});
        return acc_data_desc;
    }

    // Shape of Ds and E must be aligned. Strides can be different.
    // Pass e_g_n_k_wos_lengths for logical broadcast.
    static auto
    MakeDsGridDescriptor(const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0>&
                             ds_g_n_k_wos_lengths_0,
                         const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1>&
                             ds_g_n_k_wos_lengths_1,
                         const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2>&
                             ds_g_n_k_wos_lengths_2,
                         const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0>&
                             ds_g_n_k_wos_strides_0,
                         const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1>&
                             ds_g_n_k_wos_strides_1,
                         const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2>&
                             ds_g_n_k_wos_strides_2)
    {
        const auto DsGridDesc_0 = generate_tuple(
            [&](auto i) {
                using DLayout        = remove_cvref_t<tuple_element_t<0, DsLayout>>;
                using singleDsLayout = remove_cvref_t<tuple_element_t<i.value, DLayout>>;
                return MakeOutGridDescriptor<singleDsLayout, 0>(ds_g_n_k_wos_lengths_0[i],
                                                                ds_g_n_k_wos_strides_0[i]);
            },
            Number<NumDTensor_0>{});

        const auto DsGridDesc_1 = generate_tuple(
            [&](auto i) {
                using DLayout        = remove_cvref_t<tuple_element_t<1, DsLayout>>;
                using singleDsLayout = remove_cvref_t<tuple_element_t<i.value, DLayout>>;
                return MakeOutGridDescriptor<singleDsLayout, 1>(ds_g_n_k_wos_lengths_1[i],
                                                                ds_g_n_k_wos_strides_1[i]);
            },
            Number<NumDTensor_1>{});

        const auto DsGridDesc_2 = generate_tuple(
            [&](auto i) {
                using DLayout        = remove_cvref_t<tuple_element_t<2, DsLayout>>;
                using singleDsLayout = remove_cvref_t<tuple_element_t<i.value, DLayout>>;
                return MakeOutGridDescriptor<singleDsLayout, 2>(ds_g_n_k_wos_lengths_2[i],
                                                                ds_g_n_k_wos_strides_2[i]);
            },
            Number<NumDTensor_2>{});

        return make_tuple(DsGridDesc_0, DsGridDesc_1, DsGridDesc_2);
    }

    // desc for problem definition
    using InGridDesc  = decltype(MakeInGridDescriptor({}, {}, {}, {}, {}, {}, {}, {}, {}, {}));
    using WeiGridDesc = decltype(MakeWeiGridDescriptor({}, {}));
    using DsGridDesc  = remove_cvref_t<decltype(MakeDsGridDescriptor({}, {}, {}, {}, {}, {}))>;
#if defined(CASCADE_1X_OUT)
    using EGridDesc = remove_cvref_t<decltype(MakeOutGridDescriptor<ELayout, 0>({}, {}))>;
#elif defined(CASCADE_2X_OUT)
    using EGridDesc = remove_cvref_t<decltype(MakeOutGridDescriptor<ELayout, 1>({}, {}))>;
#else
    using EGridDesc = remove_cvref_t<decltype(MakeOutGridDescriptor<ELayout, 2>({}, {}))>;
#endif

    // GridwiseFasternet50
    using GridwiseFasternet50 =
        GridwiseFasternet50_Wcnn_CShuffle<BlockSize,
                                          InDataType,
                                          WeiDataType,
                                          DsDataType,
                                          AccDataType,
                                          EDataType,
                                          InGridDesc,
                                          WeiGridDesc,
                                          DsGridDesc,
                                          EGridDesc,
                                          DsLayout,
                                          InElementwiseOperation,
                                          WeiElementwiseOperation,
                                          AccElementwiseOperation,
                                          HPerBlock,
                                          WPerBlock,
                                          CPerBlock,
                                          KPerBlock,
                                          HRepeat,
                                          WRepeat,
                                          HPerWcnn,
                                          WPerWcnn,
                                          FilterSize,
                                          DilationX,
                                          DilationY,
                                          InBlockTransferThreadClusterLengths,
                                          InBlockTransferSrcScalarPerVector,
                                          InBlockTransferDstScalarPerVector,
                                          InEnableLds,
                                          InBlockLdsAddExtraM,
                                          InTileLoad,
                                          WeiBlockTransferThreadClusterLengths,
                                          WeiBlockTransferSrcScalarPerVector,
                                          WeiBlockTransferDstScalarPerVector,
                                          WeiEnableLds,
                                          WeiBlockLdsAddExtraM,
                                          WeiTileLoad,
                                          DsBlockTransferThreadClusterLengths,
                                          DsBlockTransferSrcScalarPerVector,
                                          DsBlockTransferDstScalarPerVector,
                                          DsEnableLds,
                                          DsBlockLdsAddExtraM,
                                          AccBlockTransferClusterLengths,
                                          AccBlockTransferScalarPerVector,
                                          AccEnableLds,
                                          EnableAsync,
                                          NumPrefetch,
                                          EnableWaveGroup,
                                          EnableSpatialCluster,
                                          ClusterDimSize,
                                          false,
                                          TileStore>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_in,
                 const std::array<const void*, 3>& p_wei,
                 const std::array<const void*, NumDTensor_0>& p_ds_0,
                 const std::array<const void*, NumDTensor_1>& p_ds_1,
                 const std::array<const void*, NumDTensor_2>& p_ds_2,
                 void* p_e,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0>&
                     ds_g_n_k_wos_lengths_0,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1>&
                     ds_g_n_k_wos_lengths_1,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2>&
                     ds_g_n_k_wos_lengths_2,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0>&
                     ds_g_n_k_wos_strides_0,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1>&
                     ds_g_n_k_wos_strides_1,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2>&
                     ds_g_n_k_wos_strides_2,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_strides,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_dilations,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& input_left_pads,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& input_right_pads,
                 const InElementwiseOperation& in_element_op,
                 const WeiElementwiseOperation& wei_element_op,
                 const AccElementwiseOperation& acc_element_op)
            : p_in_grid_{static_cast<const InDataType*>(p_in)},
              p_wei_grid_{},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              num_group_{a_g_n_c_wis_lengths[0]},
              in_grid_desc_{DeviceOp::MakeInGridDescriptor(a_g_n_c_wis_lengths,
                                                           a_g_n_c_wis_strides,
                                                           b_g_k_c_xs_lengths,
                                                           b_g_k_c_xs_strides,
                                                           e_g_n_k_wos_lengths,
                                                           e_g_n_k_wos_strides,
                                                           conv_filter_strides,
                                                           conv_filter_dilations,
                                                           input_left_pads,
                                                           input_right_pads)},
              ds_grid_desc_{},
#if defined(CASCADE_1X_OUT)
              e_grid_desc_{DeviceOp::MakeOutGridDescriptor<ELayout, 0>(e_g_n_k_wos_lengths,
                                                                       e_g_n_k_wos_strides)},
#elif defined(CASCADE_2X_OUT)
              e_grid_desc_{DeviceOp::MakeOutGridDescriptor<ELayout, 1>(e_g_n_k_wos_lengths,
                                                                       e_g_n_k_wos_strides)},
#else
              e_grid_desc_{DeviceOp::MakeOutGridDescriptor<ELayout, 2>(e_g_n_k_wos_lengths,
                                                                       e_g_n_k_wos_strides)},
#endif
              block_2_etile_map_{
                  GridwiseFasternet50::MakeDefaultBlock2CTileMap(e_grid_desc_, 1, 1)},
              compute_ptr_offset_of_batch_0_{},
              compute_ptr_offset_of_batch_1_{},
              compute_ptr_offset_of_batch_2_{},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              acc_element_op_{acc_element_op},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_k_wos_lengths_0_{ds_g_n_k_wos_lengths_0},
              ds_g_n_k_wos_strides_0_{ds_g_n_k_wos_strides_0},
              ds_g_n_k_wos_lengths_1_{ds_g_n_k_wos_lengths_1},
              ds_g_n_k_wos_strides_1_{ds_g_n_k_wos_strides_1},
              ds_g_n_k_wos_lengths_2_{ds_g_n_k_wos_lengths_2},
              ds_g_n_k_wos_strides_2_{ds_g_n_k_wos_strides_2},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{e_g_n_k_wos_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // A/B/E Batch Stride
            compute_ptr_offset_of_batch_0_.BatchStrideA_ = a_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_0_.BatchStrideE_ = e_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_0_.BatchStrideB_ = b_g_k_c_xs_strides[0][0];
            compute_ptr_offset_of_batch_1_.BatchStrideB_ = b_g_k_c_xs_strides[1][0];
            compute_ptr_offset_of_batch_2_.BatchStrideB_ = b_g_k_c_xs_strides[2][0];

            // populate pointer, batch stride, desc for Ds
            using SingConvDsDataType_0 = remove_cvref_t<tuple_element_t<0, DsDataType>>;
            using SingConvDsDataType_1 = remove_cvref_t<tuple_element_t<1, DsDataType>>;
            using SingConvDsDataType_2 = remove_cvref_t<tuple_element_t<2, DsDataType>>;

            static_for<0, NumDTensor_0, 1>{}([&](auto i) {
                using DDataType   = remove_cvref_t<tuple_element_t<i.value, SingConvDsDataType_0>>;
                p_ds_grid_(I0)(i) = static_cast<const DDataType*>(p_ds_0[i]);
                compute_ptr_offset_of_batch_0_.BatchStrideDs_(i) = ds_g_n_k_wos_strides_0[i][0];
            });

            static_for<0, NumDTensor_1, 1>{}([&](auto i) {
                using DDataType   = remove_cvref_t<tuple_element_t<i.value, SingConvDsDataType_1>>;
                p_ds_grid_(I1)(i) = static_cast<const DDataType*>(p_ds_1[i]);
                compute_ptr_offset_of_batch_1_.BatchStrideDs_(i) = ds_g_n_k_wos_strides_1[i][0];
            });

            static_for<0, NumDTensor_2, 1>{}([&](auto i) {
                using DDataType   = remove_cvref_t<tuple_element_t<i.value, SingConvDsDataType_2>>;
                p_ds_grid_(I2)(i) = static_cast<const DDataType*>(p_ds_2[i]);
                compute_ptr_offset_of_batch_2_.BatchStrideDs_(i) = ds_g_n_k_wos_strides_2[i][0];
            });

            static_for<0, conv_num, 1>{}([&](auto i) {
                array_convert(conv_filter_strides_[i], conv_filter_strides[i]);
                array_convert(conv_filter_dilations_[i], conv_filter_dilations[i]);
                p_wei_grid_(i) = static_cast<const WeiDataType*>(p_wei[i]);
            });

            wei_grid_desc_ = MakeWeiGridDescriptor(b_g_k_c_xs_lengths, b_g_k_c_xs_strides);
            ds_grid_desc_  = MakeDsGridDescriptor(ds_g_n_k_wos_lengths_0,
                                                 ds_g_n_k_wos_lengths_1,
                                                 ds_g_n_k_wos_lengths_2,
                                                 ds_g_n_k_wos_strides_0,
                                                 ds_g_n_k_wos_strides_1,
                                                 ds_g_n_k_wos_strides_2);
        }

        void Print() const
        {
            std::cout << "In: " << in_grid_desc_[I0] << std::endl;
            static_for<0, conv_num, 1>{}(
                [&](auto i) { std::cout << "Wei:" << i << wei_grid_desc_[i] << std::endl; });

            std::cout << "Out: " << e_grid_desc_ << std::endl;
        }
        // pointers
        const InDataType* p_in_grid_;
        typename GridwiseFasternet50::WeiGridPointer p_wei_grid_;
        typename GridwiseFasternet50::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors for problem definiton
        index_t num_group_;
        InGridDesc in_grid_desc_;
        WeiGridDesc wei_grid_desc_;
        DsGridDesc ds_grid_desc_;
        EGridDesc e_grid_desc_;

        // tensor descriptors for block/thread-wise copy

        // block-to-e-tile map
        typename GridwiseFasternet50::DefaultBlock2CTileMap block_2_etile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor_0> compute_ptr_offset_of_batch_0_;
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor_1> compute_ptr_offset_of_batch_1_;
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor_2> compute_ptr_offset_of_batch_2_;

        // element-wise op
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        AccElementwiseOperation acc_element_op_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, 3> b_g_k_c_xs_lengths_;
        std::array<std::array<index_t, NDimSpatial + 3>, 3> b_g_k_c_xs_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0> ds_g_n_k_wos_lengths_0_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0> ds_g_n_k_wos_strides_0_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1> ds_g_n_k_wos_lengths_1_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1> ds_g_n_k_wos_strides_1_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2> ds_g_n_k_wos_lengths_2_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2> ds_g_n_k_wos_strides_2_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<std::array<index_t, NDimSpatial>, 3> conv_filter_strides_;
        std::array<std::array<index_t, NDimSpatial>, 3> conv_filter_dilations_;
        std::array<std::array<index_t, NDimSpatial>, 3> input_left_pads_;
        std::array<std::array<index_t, NDimSpatial>, 3> input_right_pads_;
        std::array<index_t, 2> input_ds_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_) * arg.num_group_;

            auto launch_kernel = [&](auto has_main_block_loop) {
                constexpr bool has_main_loop = has_main_block_loop.value;

                const auto kernel = kernel_fasternet50_wcnn<
                    GridwiseFasternet50,
                    InDataType,
                    WeiDataType,
                    typename GridwiseFasternet50::WeiGridPointer,
                    typename GridwiseFasternet50::DsGridPointer,
                    AccDataType,
                    EDataType,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    AccElementwiseOperation,
                    DeviceOp::InGridDesc,
                    DeviceOp::WeiGridDesc,
                    DeviceOp::DsGridDesc,
                    DeviceOp::EGridDesc,
                    remove_reference_t<typename GridwiseFasternet50::DefaultBlock2CTileMap>,
                    ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor_0>{}>,
                    ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor_1>{}>,
                    ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor_2>{}>,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_in_grid_,
                                              arg.p_wei_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.in_element_op_,
                                              arg.wei_element_op_,
                                              arg.acc_element_op_,
                                              arg.a_g_n_c_wis_lengths_[0], // Group count
                                              arg.in_grid_desc_,
                                              arg.wei_grid_desc_,
                                              arg.ds_grid_desc_,
                                              arg.e_grid_desc_,
                                              arg.block_2_etile_map_,
                                              arg.compute_ptr_offset_of_batch_0_,
                                              arg.compute_ptr_offset_of_batch_1_,
                                              arg.compute_ptr_offset_of_batch_2_);
            };

            const auto C = arg.in_grid_desc_[I0].GetLength(I2);

            if(GridwiseFasternet50::CalculateHasMainBlockLoop(C))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }
    template <typename ConvTensorLayout>
    static bool IsConvLayoutCompatible(const std::array<index_t, NDimSpatial + 3>& lengths,
                                       const std::array<index_t, NDimSpatial + 3>& strides)
    {
        constexpr bool IsNativePacked =
            // input
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GNWC> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GNHWC> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GNDHWC> ||
            // weight
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GKXC> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GKYXC> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GKZYXC> ||
            // output
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GNWK> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GNHWK> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::GNDHWK>;

        constexpr bool IsStrided =
            // input
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_NW_C> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_NHW_C> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_NDHW_C> ||
            // output
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_NW_K> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_NHW_K> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_NDHW_K>;

        // weight
        constexpr bool IsStridedWeight =
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_K_X_C> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_K_YX_C> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_K_ZYX_C>;

        constexpr bool IsGCPacked =
            // input
            is_same_v<ConvTensorLayout, tensor_layout::convolution::NWGC> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::NHWGC> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::NDHWGC> ||
            // weight
            is_same_v<ConvTensorLayout, tensor_layout::convolution::KXGC> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::KYXGC> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::KZYXGC> ||
            // output
            is_same_v<ConvTensorLayout, tensor_layout::convolution::NWGK> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::NHWGK> ||
            is_same_v<ConvTensorLayout, tensor_layout::convolution::NDHWGK>;

        constexpr bool IsStridedK =
            // bias
            is_same_v<ConvTensorLayout, tensor_layout::convolution::G_K>;

        if constexpr(IsGCPacked)
        {
            // remap dim order to NHWGC
            index_t gc_remap_table[NDimSpatial + 3] = {1};
            for(index_t i = 0; i < NDimSpatial; i++)
            {
                gc_remap_table[i + 1] = i + 3;
            }
            gc_remap_table[NDimSpatial + 1] = 0;
            gc_remap_table[NDimSpatial + 2] = 2;
            bool is_compatible              = strides[gc_remap_table[NDimSpatial + 2]] == 1;
            for(index_t i = 0; i < NDimSpatial + 2; i++)
            {
                is_compatible &= (strides[gc_remap_table[i]] ==
                                  lengths[gc_remap_table[i + 1]] * strides[gc_remap_table[i + 1]]);
            }
            return is_compatible;
        }
        else if constexpr(IsStridedK)
        {
            return true;
        }
        else
        {
            // remap dim order to GNHWC
            index_t remap_table[NDimSpatial + 3] = {0, 1};
            for(index_t i = 0; i < NDimSpatial; i++)
            {
                remap_table[i + 2] = i + 3;
            }
            remap_table[NDimSpatial + 2] = 2;

            if constexpr(IsNativePacked)
            {
                bool is_compatible = strides[remap_table[NDimSpatial + 2]] == 1;
                for(index_t i = 0; i < NDimSpatial + 2; i++)
                {
                    is_compatible &= (strides[remap_table[i]] ==
                                      lengths[remap_table[i + 1]] * strides[remap_table[i + 1]]);
                }
                return is_compatible;
            }
            else if constexpr(IsStrided)
            {
                bool is_compatible = true;
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    is_compatible &= (strides[remap_table[i + 1]] ==
                                      lengths[remap_table[i + 2]] * strides[remap_table[i + 2]]);
                }
                return is_compatible;
            }
            else if constexpr(IsStridedWeight)
            {
                bool is_compatible = true;
                for(index_t i = 0; i < NDimSpatial - 1; i++)
                {
                    is_compatible &= (strides[remap_table[i + 2]] ==
                                      lengths[remap_table[i + 3]] * strides[remap_table[i + 3]]);
                }
                return is_compatible;
            }
            else
            {
                static_assert(0, "Unsupported layout");
                return false;
            }
        }
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        namespace ctc = ck::tensor_layout::convolution;

        if(1) // ck::is_gfx13_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, ck::half_t> ||
                           is_same_v<AccDataType, ck::bhalf_t> || is_same_v<AccDataType, int32_t>))
            {
                printf("DeviceOp err: AccDataType\n");
                return false;
            }
        }
        else
        {
            printf("DeviceOp err: Arch\n");
            return false;
        }

        bool input_layout_compatible =
            IsConvLayoutCompatible<InLayout>(arg.a_g_n_c_wis_lengths_, arg.a_g_n_c_wis_strides_);
        if(input_layout_compatible == false)
        {
            printf("Input data incompatible with layout!\n");
            return false;
        }

        bool weight_0_layout_compatible = IsConvLayoutCompatible<WeiLayout>(
            arg.b_g_k_c_xs_lengths_[0], arg.b_g_k_c_xs_strides_[0]);
        if(weight_0_layout_compatible == false)
        {
            printf("Weight_0 data incompatible with layout!\n");
            return false;
        }

        bool weight_1_layout_compatible = IsConvLayoutCompatible<WeiLayout>(
            arg.b_g_k_c_xs_lengths_[1], arg.b_g_k_c_xs_strides_[1]);
        if(weight_1_layout_compatible == false)
        {
            printf("Weight_1 data incompatible with layout!\n");
            return false;
        }

        bool weight_2_layout_compatible = IsConvLayoutCompatible<WeiLayout>(
            arg.b_g_k_c_xs_lengths_[2], arg.b_g_k_c_xs_strides_[2]);
        if(weight_2_layout_compatible == false)
        {
            printf("Weight_2 data incompatible with layout!\n");
            return false;
        }

        bool output_layout_compatible =
            IsConvLayoutCompatible<ELayout>(arg.e_g_n_k_wos_lengths_, arg.e_g_n_k_wos_strides_);
        if(output_layout_compatible == false)
        {
            printf("Output data incompatible with layout!\n");
            return false;
        }

        for(index_t i = 0; i < NumDTensor_0; i++)
        {
            if(arg.a_g_n_c_wis_lengths_[0] != arg.ds_g_n_k_wos_lengths_0_[i][0])
            {

                printf("K_0 is incorrect:%d %d!\n",
                       arg.a_g_n_c_wis_lengths_[0],
                       arg.ds_g_n_k_wos_lengths_0_[i][0]);
                return false;
            }
        }

        for(index_t i = 0; i < NumDTensor_1; i++)
        {
            if(arg.ds_g_n_k_wos_lengths_1_[i][2] != 64)
            {
                printf("K_1 is incorrect:%d %d!\n", i, arg.ds_g_n_k_wos_lengths_1_[i][2]);
                return false;
            }
        }

        for(index_t i = 0; i < NumDTensor_2; i++)
        {
            if(arg.ds_g_n_k_wos_lengths_2_[i][2] != 32)
            {
                printf("K_2 is incorrect:%d %d!\n", i, arg.ds_g_n_k_wos_lengths_2_[i][2]);
                return false;
            }
        }

        // check vector access of InData
        if constexpr(is_same_v<InLayout, ctc::G_NW_C> || is_same_v<InLayout, ctc::G_NHW_C> ||
                     is_same_v<InLayout, ctc::G_NDHW_C> || is_same_v<InLayout, ctc::GNWC> ||
                     is_same_v<InLayout, ctc::GNHWC> || is_same_v<InLayout, ctc::GNDHWC> ||
                     is_same_v<InLayout, ctc::NWGC> || is_same_v<InLayout, ctc::NHWGC> ||
                     is_same_v<InLayout, ctc::NDHWGC>)
        {
            const index_t C = arg.a_g_n_c_wis_lengths_[2];

            if(!(C % InBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        bool valid = true;

        // check vector access of WeiData
        if constexpr(is_same_v<WeiLayout, ctc::G_K_X_C> || is_same_v<WeiLayout, ctc::G_K_YX_C> ||
                     is_same_v<WeiLayout, ctc::G_K_ZYX_C> || is_same_v<WeiLayout, ctc::GKXC> ||
                     is_same_v<WeiLayout, ctc::GKYXC> || is_same_v<WeiLayout, ctc::GKZYXC> ||
                     is_same_v<WeiLayout, ctc::KXGC> || is_same_v<WeiLayout, ctc::KYXGC> ||
                     is_same_v<WeiLayout, ctc::KZYXGC>)

        {
            static_for<0, conv_num, 1>{}([&](auto i) {
                const index_t C = arg.b_g_k_c_xs_lengths_[i][2];
                if(!(C % WeiBlockTransferSrcScalarPerVector == 0))
                {
                    valid = false;
                }
            });
        }
        else
        {
            valid = false;
        }

        if(!valid)
        {
            return false;
        }

        if(!valid)
        {
            printf("The layout is invalid!\n");
            return false;
        }

        // check vector access of Accum
        if constexpr(is_same_v<ELayout, ctc::G_NW_K> || is_same_v<ELayout, ctc::G_NHW_K> ||
                     is_same_v<ELayout, ctc::G_NDHW_K> || is_same_v<ELayout, ctc::GNWK> ||
                     is_same_v<ELayout, ctc::GNHWK> || is_same_v<ELayout, ctc::GNDHWK> ||
                     is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                     is_same_v<ELayout, ctc::NDHWGK>)
        {
            const index_t K = arg.e_g_n_k_wos_lengths_[2];

            if(!(K % AccBlockTransferScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }
        // check Gridwise Conv
        return GridwiseFasternet50::CheckValidity(
            arg.in_grid_desc_, arg.wei_grid_desc_, arg.e_grid_desc_, arg.block_2_etile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg)
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const void* p_in,
                 const std::array<const void*, 3>& p_wei,
                 const std::array<const void*, NumDTensor_0>& p_ds_0,
                 const std::array<const void*, NumDTensor_1>& p_ds_1,
                 const std::array<const void*, NumDTensor_2>& p_ds_2,
                 void* p_e,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0>&
                     ds_g_n_k_wos_lengths_0,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1>&
                     ds_g_n_k_wos_lengths_1,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2>&
                     ds_g_n_k_wos_lengths_2,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0>&
                     ds_g_n_k_wos_strides_0,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1>&
                     ds_g_n_k_wos_strides_1,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2>&
                     ds_g_n_k_wos_strides_2,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_strides,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_dilations,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& input_left_pads,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& input_right_pads,
                 const InElementwiseOperation& in_element_op,
                 const WeiElementwiseOperation& wei_element_op,
                 const AccElementwiseOperation& acc_element_op)
    {
        return Argument{p_in,
                        p_wei,
                        p_ds_0,
                        p_ds_1,
                        p_ds_2,
                        p_e,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_k_wos_lengths_0,
                        ds_g_n_k_wos_lengths_1,
                        ds_g_n_k_wos_lengths_2,
                        ds_g_n_k_wos_strides_0,
                        ds_g_n_k_wos_strides_1,
                        ds_g_n_k_wos_strides_2,
                        e_g_n_k_wos_lengths,
                        e_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        acc_element_op};
    }

    static auto
    MakeArgument(const void* p_a,
                 const std::array<const void*, 3>& p_b,
                 const std::array<const void*, NumDTensor_0>& p_ds_0,
                 const std::array<const void*, NumDTensor_1>& p_ds_1,
                 const std::array<const void*, NumDTensor_2>& p_ds_2,
                 void* p_e,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0>&
                     ds_g_n_k_wos_lengths_0,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1>&
                     ds_g_n_k_wos_lengths_1,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2>&
                     ds_g_n_k_wos_lengths_2,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_0>&
                     ds_g_n_k_wos_strides_0,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_1>&
                     ds_g_n_k_wos_strides_1,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor_2>&
                     ds_g_n_k_wos_strides_2,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_strides,
                 const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_dilations,
                 const std::array<std::array<long_index_t, NDimSpatial>, 3>& input_left_pads,
                 const std::array<std::array<long_index_t, NDimSpatial>, 3>& input_right_pads,
                 const InElementwiseOperation& in_element_op,
                 const WeiElementwiseOperation& wei_element_op,
                 const AccElementwiseOperation& acc_element_op)
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i32;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, 3> b_g_k_c_xs_lengths_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, 3> b_g_k_c_xs_strides_i32;

        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_0>
            ds_g_n_k_wos_lengths_0_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_1>
            ds_g_n_k_wos_lengths_1_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_2>
            ds_g_n_k_wos_lengths_2_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_0>
            ds_g_n_k_wos_strides_0_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_1>
            ds_g_n_k_wos_strides_1_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_2>
            ds_g_n_k_wos_strides_2_i32;

        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i32;
        std::array<std::array<index_t, NDimSpatial>, 3> conv_filter_strides_i32;
        std::array<std::array<index_t, NDimSpatial>, 3> conv_filter_dilations_i32;
        std::array<std::array<index_t, NDimSpatial>, 3> input_left_pads_i32;
        std::array<std::array<index_t, NDimSpatial>, 3> input_right_pads_i32;

        array_convert(a_g_n_c_wis_lengths_i32, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i32, a_g_n_c_wis_strides);

        for(index_t d = 0; d < NumDTensor_0; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_0_i32[d], ds_g_n_k_wos_lengths_0[d]);
            array_convert(ds_g_n_k_wos_strides_0_i32[d], ds_g_n_k_wos_strides_0[d]);
        }
        for(index_t d = 0; d < NumDTensor_1; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_1_i32[d], ds_g_n_k_wos_lengths_1[d]);
            array_convert(ds_g_n_k_wos_strides_1_i32[d], ds_g_n_k_wos_strides_1[d]);
        }
        for(index_t d = 0; d < NumDTensor_2; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_2_i32[d], ds_g_n_k_wos_lengths_2[d]);
            array_convert(ds_g_n_k_wos_strides_2_i32[d], ds_g_n_k_wos_strides_2[d]);
        }

        constexpr index_t cascade_node_num = 3;
        for(index_t i = 0; i < cascade_node_num; i++)
        {
            array_convert(b_g_k_c_xs_lengths_i32[i], b_g_k_c_xs_lengths[i]);
            array_convert(b_g_k_c_xs_strides_i32[i], b_g_k_c_xs_strides[i]);
            array_convert(conv_filter_strides_i32[i], conv_filter_strides[i]);
            array_convert(conv_filter_dilations_i32[i], conv_filter_dilations[i]);
            array_convert(input_left_pads_i32[i], input_left_pads[i]);
            array_convert(input_right_pads_i32[i], input_right_pads[i]);
        }

        array_convert(e_g_n_k_wos_lengths_i32, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i32, e_g_n_k_wos_strides);

        return Argument{p_a,
                        p_b,
                        p_ds_0,
                        p_ds_1,
                        p_ds_2,
                        p_e,
                        a_g_n_c_wis_lengths_i32,
                        a_g_n_c_wis_strides_i32,
                        b_g_k_c_xs_lengths_i32,
                        b_g_k_c_xs_strides_i32,
                        ds_g_n_k_wos_lengths_0_i32,
                        ds_g_n_k_wos_lengths_1_i32,
                        ds_g_n_k_wos_lengths_2_i32,
                        ds_g_n_k_wos_strides_0_i32,
                        ds_g_n_k_wos_strides_1_i32,
                        ds_g_n_k_wos_strides_2_i32,
                        e_g_n_k_wos_lengths_i32,
                        e_g_n_k_wos_strides_i32,
                        conv_filter_strides_i32,
                        conv_filter_dilations_i32,
                        input_left_pads_i32,
                        input_right_pads_i32,
                        in_element_op,
                        wei_element_op,
                        acc_element_op};
    }
    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_in,
        const std::array<const void*, 3> p_wei,
        const std::array<const void*, NumDTensor_0>& p_ds_0,
        const std::array<const void*, NumDTensor_1>& p_ds_1,
        const std::array<const void*, NumDTensor_2>& p_ds_2,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_strides,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_0>&
            ds_g_n_k_wos_lengths_0,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_1>&
            ds_g_n_k_wos_lengths_1,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_2>&
            ds_g_n_k_wos_lengths_2,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_0>&
            ds_g_n_k_wos_strides_0,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_1>&
            ds_g_n_k_wos_strides_1,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_2>&
            ds_g_n_k_wos_strides_2,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_strides,
        const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_dilations,
        const std::array<std::array<index_t, NDimSpatial>, 3>& input_left_pads,
        const std::array<std::array<index_t, NDimSpatial>, 3>& input_right_pads,
        const InElementwiseOperation& in_element_op,
        const WeiElementwiseOperation& wei_element_op,
        const AccElementwiseOperation& acc_element_op)
    {
        return std::make_unique<Argument>(p_in,
                                          p_wei,
                                          p_ds_0,
                                          p_ds_1,
                                          p_ds_2,
                                          p_e,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_k_wos_lengths_0,
                                          ds_g_n_k_wos_lengths_1,
                                          ds_g_n_k_wos_lengths_2,
                                          ds_g_n_k_wos_strides_0,
                                          ds_g_n_k_wos_strides_1,
                                          ds_g_n_k_wos_strides_2,
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          acc_element_op);
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const std::array<const void*, 3>& p_b,
        const std::array<const void*, NumDTensor_0>& p_ds_0,
        const std::array<const void*, NumDTensor_1>& p_ds_1,
        const std::array<const void*, NumDTensor_2>& p_ds_2,
        void* p_e,
        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, 3>& b_g_k_c_xs_strides,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_0>
            ds_g_n_k_wos_lengths_0,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_1>
            ds_g_n_k_wos_lengths_1,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_2>
            ds_g_n_k_wos_lengths_2,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_0>
            ds_g_n_k_wos_strides_0,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_1>
            ds_g_n_k_wos_strides_1,
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_2>
            ds_g_n_k_wos_strides_2,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_strides,
        const std::array<std::array<index_t, NDimSpatial>, 3>& conv_filter_dilations,
        const std::array<std::array<long_index_t, NDimSpatial>, 3>& input_left_pads,
        const std::array<std::array<long_index_t, NDimSpatial>, 3>& input_right_pads,
        const InElementwiseOperation& in_element_op,
        const WeiElementwiseOperation& wei_element_op,
        const AccElementwiseOperation& acc_element_op)
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i32;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, 3> b_g_k_c_xs_lengths_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, 3> b_g_k_c_xs_strides_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_0>
            ds_g_n_k_wos_lengths_0_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_1>
            ds_g_n_k_wos_lengths_1_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_2>
            ds_g_n_k_wos_lengths_2_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_0>
            ds_g_n_k_wos_strides_0_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_1>
            ds_g_n_k_wos_strides_1_i32;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor_2>
            ds_g_n_k_wos_strides_2_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i32;
        std::array<std::array<index_t, NDimSpatial>, 3> conv_filter_strides_i32;
        std::array<std::array<index_t, NDimSpatial>, 3> conv_filter_dilations_i32;
        std::array<std::array<index_t, NDimSpatial>, 3> input_left_pads_i32;
        std::array<std::array<index_t, NDimSpatial>, 3> input_right_pads_i32;

        array_convert(a_g_n_c_wis_lengths_i32, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i32, a_g_n_c_wis_strides);

        for(index_t d = 0; d < NumDTensor_0; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_0_i32[d], ds_g_n_k_wos_lengths_0[d]);
            array_convert(ds_g_n_k_wos_strides_0_i32[d], ds_g_n_k_wos_strides_0[d]);
        }
        for(index_t d = 0; d < NumDTensor_1; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_1_i32[d], ds_g_n_k_wos_lengths_1[d]);
            array_convert(ds_g_n_k_wos_strides_1_i32[d], ds_g_n_k_wos_strides_1[d]);
        }
        for(index_t d = 0; d < NumDTensor_2; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_2_i32[d], ds_g_n_k_wos_lengths_2[d]);
            array_convert(ds_g_n_k_wos_strides_2_i32[d], ds_g_n_k_wos_strides_2[d]);
        }

        for(index_t i = 0; i < 3; i++)
        {
            array_convert(b_g_k_c_xs_lengths_i32[i], b_g_k_c_xs_lengths[i]);
            array_convert(b_g_k_c_xs_strides_i32[i], b_g_k_c_xs_strides[i]);
            array_convert(conv_filter_strides_i32[i], conv_filter_strides[i]);
            array_convert(conv_filter_dilations_i32[i], conv_filter_dilations[i]);
            array_convert(input_left_pads_i32[i], input_left_pads[i]);
            array_convert(input_right_pads_i32[i], input_right_pads[i]);
        }

        array_convert(e_g_n_k_wos_lengths_i32, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i32, e_g_n_k_wos_strides);

        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds_0,
                                          p_ds_1,
                                          p_ds_2,
                                          p_e,
                                          a_g_n_c_wis_lengths_i32,
                                          a_g_n_c_wis_strides_i32,
                                          b_g_k_c_xs_lengths_i32,
                                          b_g_k_c_xs_strides_i32,
                                          ds_g_n_k_wos_lengths_0_i32,
                                          ds_g_n_k_wos_lengths_1_i32,
                                          ds_g_n_k_wos_lengths_2_i32,
                                          ds_g_n_k_wos_strides_0_i32,
                                          ds_g_n_k_wos_strides_1_i32,
                                          ds_g_n_k_wos_strides_2_i32,
                                          e_g_n_k_wos_lengths_i32,
                                          e_g_n_k_wos_strides_i32,
                                          conv_filter_strides_i32,
                                          conv_filter_dilations_i32,
                                          input_left_pads_i32,
                                          input_right_pads_i32,
                                          in_element_op,
                                          wei_element_op,
                                          acc_element_op);
    }
    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvFwdMultipleD_Wcnn_CShuffle"
            << "<"
            << BlockSize << ", "
            << HPerBlock << ", "
            << WPerBlock << ", "
            << CPerBlock::At(0) << ", "
            << CPerBlock::At(1) << ", "
            << CPerBlock::At(2) << ", "
            << KPerBlock::At(0) << ", "
            << KPerBlock::At(1) << ", "
            << KPerBlock::At(2) << ", "
            << HRepeat << ", "
            << WRepeat << ", "
            << HPerWcnn << ", "
            << WPerWcnn << ", "
            << FilterSize::At(0) << ", "
            << FilterSize::At(1) << ", "
            << FilterSize::At(2) << ", "
            << DilationX::At(0) << ", "
            << DilationX::At(1) << ", "
            << DilationX::At(2) << ", "
            << DilationY::At(0) << ", "
            << DilationY::At(1) << ", "
            << DilationY::At(2) << ", "
            << " InEnableLds: "
            << InEnableLds << ", "
            << " InTileLoad: "
            << InTileLoad << ", "
            << "WeiEnableLds: "
            << WeiEnableLds << ", "
            << "WeiTileLoad: "
            << WeiTileLoad << ", "
            << "DsEnableLds: "
            << DsEnableLds << ", "
            << "NumPrefetch: "
            << NumPrefetch << ", "
            << "EnableWaveGroup: "
            << EnableWaveGroup << ", "
            << "EnableSpatialCluster: "
            << EnableSpatialCluster << ", "
            << "ClusterDimSize: "
            << ClusterDimSize << ", "
            << "ShuffleOnLoad: "
            << ShuffleOnLoad << ", "
            << "Transpose: "
            << Transposed << ", "
            << "TileStore: "
            << TileStore << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device

} // namespace tensor_operation
} // namespace ck
