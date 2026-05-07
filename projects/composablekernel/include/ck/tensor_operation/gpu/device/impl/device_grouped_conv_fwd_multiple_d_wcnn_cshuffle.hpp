// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "ck/tensor_operation/gpu/grid/gridwise_conv_fwd_multiple_d_wcnn_cshuffle.hpp"
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
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          index_t NumPrefetch,
          index_t BlockSize,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t CPerBlock,
          index_t KPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
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
struct DeviceGroupedConvFwdMultipleD_Wcnn_CShuffle
    : public DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                             InLayout,
                                             WeiLayout,
                                             DsLayout,
                                             ELayout,
                                             InDataType,
                                             WeiDataType,
                                             DsDataType,
                                             AccDataType,
                                             InElementwiseOperation,
                                             WeiElementwiseOperation,
                                             AccElementwiseOperation>
{
    using DeviceOp = DeviceGroupedConvFwdMultipleD_Wcnn_CShuffle;
    static constexpr auto conv_to_hwc_wcnn_transformer =
        TransformConvFwdToHWCWcnn<NDimSpatial,
                                  ShuffleOnLoad,
                                  Transposed,
                                  ConvForwardSpecialization>{};

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    static constexpr index_t NumDTensor         = DsDataType::Size();
    static constexpr bool ShuffleConv2          = ShuffleOnLoad && (Transposed == false);
    static constexpr bool ShuffleTransposeConv2 = ShuffleOnLoad && Transposed;

    static constexpr index_t GridFilterSize = ShuffleOnLoad ? 1 : FilterSize;
    static constexpr index_t GridTransposed = ShuffleOnLoad ? false : Transposed;
    static constexpr index_t GridHPerBlock  = ShuffleConv2 ? HPerBlock / 2 : HPerBlock;
    static constexpr index_t GridWPerBlock  = ShuffleConv2 ? WPerBlock / 2 : WPerBlock;
    static constexpr index_t GridCPerBlock  = ShuffleConv2 ? CPerBlock * 4 : CPerBlock;
    static constexpr index_t GridHRepeat    = ShuffleConv2 ? HRepeat / 2 : HRepeat;
    static constexpr index_t GridWRepeat    = ShuffleConv2 ? WRepeat / 2 : WRepeat;
    static constexpr index_t GridKPerBlock  = ShuffleTransposeConv2 ? KPerBlock * 4 : KPerBlock;

    // Describe how data read from Global memory
    static auto
    MakeInGridDescriptor(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                         const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                         const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                         const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                         const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                         const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                         const std::array<index_t, NDimSpatial>& conv_filter_strides,
                         const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                         const std::array<index_t, NDimSpatial>& input_left_pads,
                         const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        // H W C
        const auto in_data_raw_desc =
            conv_to_hwc_wcnn_transformer.template MakeADescriptor_H_W_C<InLayout>(
                a_g_n_c_wis_lengths,
                a_g_n_c_wis_strides,
                b_g_k_c_xs_lengths,
                b_g_k_c_xs_strides,
                e_g_n_k_wos_lengths,
                e_g_n_k_wos_strides,
                conv_filter_strides,
                conv_filter_dilations,
                input_left_pads,
                input_right_pads);

#ifdef ENABLE_CONST_LAYOUT
        // H W C with pad
        if constexpr(is_same_v<InLayout, tensor_layout::convolution::CONST_GNHWC>)
        {
            return in_data_raw_desc;
        }
        else
#endif
        {
            const auto in_data_desc =
                PadTensorDescriptor(in_data_raw_desc,
                                    Sequence<GridHPerBlock, GridWPerBlock, GridCPerBlock>{},
                                    Sequence<true, true, true>{});
            return in_data_desc;
        }
    }

    static auto
    MakeWeiGridDescriptor(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                          const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides)
    {
        const auto wei_data_raw_desc =
            conv_to_hwc_wcnn_transformer.template MakeBDescriptor_K_YX_C<WeiLayout>(
                b_g_k_c_xs_lengths, b_g_k_c_xs_strides);
#ifdef ENABLE_CONST_LAYOUT
        if constexpr(is_same_v<WeiLayout, tensor_layout::convolution::CONST_GKYXC<FilterSize>>)
        {
            return wei_data_raw_desc;
        }
        else
#endif
        {
            const auto wei_data_desc = PadTensorDescriptor(wei_data_raw_desc,
                                                           make_tuple(KPerBlock, 1, GridCPerBlock),
                                                           Sequence<true, false, true>{});
            return wei_data_desc;
        }
    }

    template <typename OutLayout_>
    static auto
    MakeOutGridDescriptor(const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides)
    {
        const auto acc_data_raw_desc =
            conv_to_hwc_wcnn_transformer.template MakeCDescriptor_H_W_K<OutLayout_>(
                e_g_n_k_wos_lengths, e_g_n_k_wos_strides);
#ifdef ENABLE_CONST_LAYOUT
        if constexpr(is_same_v<OutLayout_, tensor_layout::convolution::CONST_GNHWK>)
        {
            return acc_data_raw_desc;
        }
        else
#endif
        {
            constexpr index_t HPerBlockOut =
                GridFilterSize == 2 ? (Transposed ? HPerBlock * 2 : HPerBlock / 2) : HPerBlock;
            constexpr index_t WPerBlockOut =
                GridFilterSize == 2 ? (Transposed ? WPerBlock * 2 : WPerBlock / 2) : WPerBlock;
            const auto acc_data_desc =
                PadTensorDescriptor(acc_data_raw_desc,
                                    make_tuple(HPerBlockOut, WPerBlockOut, KPerBlock),
                                    Sequence<true, true, true>{});

            return acc_data_desc;
        }
    }

    template <typename DLayout_>
    static auto
    MakeSingleDGridDescriptor(const std::array<index_t, NDimSpatial + 3>& ds_g_n_k_wos_lengths,
                              const std::array<index_t, NDimSpatial + 3>& ds_g_n_k_wos_strides)
    {

        return MakeOutGridDescriptor<DLayout_>(ds_g_n_k_wos_lengths, ds_g_n_k_wos_strides);
    }

    // Shape of Ds and E must be aligned. Strides can be different.
    // Pass e_g_n_k_wos_lengths for logical broadcast.
    static auto MakeDsGridDescriptor(
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                return MakeSingleDGridDescriptor<DLayout>(ds_g_n_k_wos_lengths[i],
                                                          ds_g_n_k_wos_strides[i]);
            },
            Number<NumDTensor>{});
    }

    // desc for problem definition
    using InGridDesc  = decltype(MakeInGridDescriptor({}, {}, {}, {}, {}, {}, {}, {}, {}, {}));
    using WeiGridDesc = decltype(MakeWeiGridDescriptor({}, {}));
    using DsGridDesc  = remove_cvref_t<decltype(MakeDsGridDescriptor({}, {}))>;
    using EGridDesc   = remove_cvref_t<decltype(MakeOutGridDescriptor<ELayout>({}, {}))>;

    // GridwiseConv
    using GridwiseConv = GridwiseConvMultipleD_Wcnn_CShuffle<BlockSize,
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
                                                             GridHPerBlock,
                                                             GridWPerBlock,
                                                             GridCPerBlock,
                                                             KPerBlock,
                                                             GridHRepeat,
                                                             GridWRepeat,
                                                             HPerWcnn,
                                                             WPerWcnn,
                                                             GridFilterSize,
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
                                                             GridTransposed,
                                                             TileStore>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_in,
                 const void* p_wei,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const InElementwiseOperation& in_element_op,
                 const WeiElementwiseOperation& wei_element_op,
                 const AccElementwiseOperation& acc_element_op)
            : p_in_grid_{static_cast<const InDataType*>(p_in)},
              p_wei_grid_{static_cast<const WeiDataType*>(p_wei)},
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
              wei_grid_desc_{
                  DeviceOp::MakeWeiGridDescriptor(b_g_k_c_xs_lengths, b_g_k_c_xs_strides)},
              ds_grid_desc_{},
              e_grid_desc_{DeviceOp::MakeOutGridDescriptor<ELayout>(e_g_n_k_wos_lengths,
                                                                    e_g_n_k_wos_strides)},
              block_2_etile_map_{GridwiseConv::MakeDefaultBlock2CTileMap(e_grid_desc_, 1, 1)},
              compute_ptr_offset_of_batch_{},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              acc_element_op_{acc_element_op},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_k_wos_lengths_{ds_g_n_k_wos_lengths},
              ds_g_n_k_wos_strides_{ds_g_n_k_wos_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{e_g_n_k_wos_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // A/B/E Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_k_c_xs_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideE_ = e_g_n_k_wos_strides[0];

            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);

                // D batch stride
                compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_n_k_wos_strides[i][0];
            });
            ds_grid_desc_ = MakeDsGridDescriptor(ds_g_n_k_wos_lengths, ds_g_n_k_wos_strides);
        }

        void Print() const
        {
            std::cout << "In: " << in_grid_desc_ << std::endl;
            std::cout << "Wei: " << wei_grid_desc_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_[i] << std::endl; });
            std::cout << "Out: " << e_grid_desc_ << std::endl;
        }
        // pointers
        const InDataType* p_in_grid_;
        const WeiDataType* p_wei_grid_;
        typename GridwiseConv::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors for problem definiton
        index_t num_group_;
        InGridDesc in_grid_desc_;
        WeiGridDesc wei_grid_desc_;
        DsGridDesc ds_grid_desc_;
        EGridDesc e_grid_desc_;

        // tensor descriptors for block/thread-wise copy

        // block-to-e-tile map
        typename GridwiseConv::DefaultBlock2CTileMap block_2_etile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor>{}> compute_ptr_offset_of_batch_;

        // element-wise op
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        AccElementwiseOperation acc_element_op_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> conv_filter_dilations_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;
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
                constexpr bool has_main_loop    = has_main_block_loop.value;
                constexpr bool EnableWaveGroup4 = EnableWaveGroup && (BlockSize == 512);
                if constexpr(EnableWaveGroup4)
                {
                    if constexpr(EnableSpatialCluster)
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_wcnn_wavegroup512_spatial_cluster<
                                GridwiseConv,
                                InDataType,
                                WeiDataType,
                                typename GridwiseConv::DsGridPointer,
                                AccDataType,
                                EDataType,
                                InElementwiseOperation,
                                WeiElementwiseOperation,
                                AccElementwiseOperation,
                                DeviceOp::InGridDesc,
                                DeviceOp::WeiGridDesc,
                                DeviceOp::DsGridDesc,
                                DeviceOp::EGridDesc,
                                remove_reference_t<typename GridwiseConv::DefaultBlock2CTileMap>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor>{}>,
                                has_main_loop,
                                ClusterDimSize>;

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
                                                      arg.compute_ptr_offset_of_batch_);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_fwd_wcnn_wavegroup512<
                            GridwiseConv,
                            InDataType,
                            WeiDataType,
                            typename GridwiseConv::DsGridPointer,
                            AccDataType,
                            EDataType,
                            InElementwiseOperation,
                            WeiElementwiseOperation,
                            AccElementwiseOperation,
                            DeviceOp::InGridDesc,
                            DeviceOp::WeiGridDesc,
                            DeviceOp::DsGridDesc,
                            DeviceOp::EGridDesc,
                            remove_reference_t<typename GridwiseConv::DefaultBlock2CTileMap>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor>{}>,
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
                                                      arg.compute_ptr_offset_of_batch_);
                    }
                }
                else if constexpr(EnableWaveGroup)
                {
                    if constexpr(EnableSpatialCluster)
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_wcnn_wavegroup256_spatial_cluster<
                                GridwiseConv,
                                InDataType,
                                WeiDataType,
                                typename GridwiseConv::DsGridPointer,
                                AccDataType,
                                EDataType,
                                InElementwiseOperation,
                                WeiElementwiseOperation,
                                AccElementwiseOperation,
                                DeviceOp::InGridDesc,
                                DeviceOp::WeiGridDesc,
                                DeviceOp::DsGridDesc,
                                DeviceOp::EGridDesc,
                                remove_reference_t<typename GridwiseConv::DefaultBlock2CTileMap>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor>{}>,
                                has_main_loop,
                                ClusterDimSize>;

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
                                                      arg.compute_ptr_offset_of_batch_);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_fwd_wcnn_wavegroup256<
                            GridwiseConv,
                            InDataType,
                            WeiDataType,
                            typename GridwiseConv::DsGridPointer,
                            AccDataType,
                            EDataType,
                            InElementwiseOperation,
                            WeiElementwiseOperation,
                            AccElementwiseOperation,
                            DeviceOp::InGridDesc,
                            DeviceOp::WeiGridDesc,
                            DeviceOp::DsGridDesc,
                            DeviceOp::EGridDesc,
                            remove_reference_t<typename GridwiseConv::DefaultBlock2CTileMap>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor>{}>,
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
                                                      arg.compute_ptr_offset_of_batch_);
                    }
                }
                else
                {
                    const auto kernel = kernel_grouped_conv_fwd_wcnn<
                        GridwiseConv,
                        InDataType,
                        WeiDataType,
                        typename GridwiseConv::DsGridPointer,
                        AccDataType,
                        EDataType,
                        InElementwiseOperation,
                        WeiElementwiseOperation,
                        AccElementwiseOperation,
                        DeviceOp::InGridDesc,
                        DeviceOp::WeiGridDesc,
                        DeviceOp::DsGridDesc,
                        DeviceOp::EGridDesc,
                        remove_reference_t<typename GridwiseConv::DefaultBlock2CTileMap>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, Number<NumDTensor>{}>,
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
                                                  arg.compute_ptr_offset_of_batch_);
                }
            };

            const auto C = arg.in_grid_desc_.GetLength(I2);

            if(GridwiseConv::CalculateHasMainBlockLoop(C))
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

    static bool IsConvSpecializationCompatible(const Argument& arg)
    {
        bool is_compatible = true;
        if constexpr(NDimSpatial == 1)
        {
            static_assert(FilterSize == 1);
            static_assert(DilationX == 1);
            static_assert(DilationY == 1);
            if constexpr(ConvForwardSpecialization == ConvolutionForwardSpecialization::Default)
            {
                return true;
            }
            else if constexpr(ConvForwardSpecialization ==
                                  ConvolutionForwardSpecialization::Filter1x1Pad0 ||
                              ConvForwardSpecialization ==
                                  ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
            {
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    is_compatible &= (arg.b_g_k_c_xs_lengths_[i + 3] == 1);
                }
                if constexpr(ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
                {
                    for(index_t i = 0; i < NDimSpatial; i++)
                    {
                        is_compatible &= (arg.conv_filter_strides_[i] == 1);
                    }
                }
                // check pad, dilation
                is_compatible &= (arg.input_left_pads_[0] == 0);
                is_compatible &= (arg.input_right_pads_[0] == 0);
                is_compatible &= (arg.conv_filter_dilations_[0] == 1);
                return is_compatible;
            }
            else
            {
                static_assert(0, "unsupported!");
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            // Check filter size
            if constexpr(ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter1x1Pad0 ||
                         ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
            {
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    is_compatible &= (arg.b_g_k_c_xs_lengths_[i + 3] == 1);
                }
            }
            else if constexpr(ConvForwardSpecialization ==
                                  ConvolutionForwardSpecialization::Filter3x3Stride1Pad0 ||
                              ConvForwardSpecialization ==
                                  ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
            {
                is_compatible &= (arg.b_g_k_c_xs_lengths_[NDimSpatial + 1] == 3);
                is_compatible &= (arg.b_g_k_c_xs_lengths_[NDimSpatial + 2] == 3);
            }
            else if constexpr(ConvForwardSpecialization ==
                                  ConvolutionForwardSpecialization::Filter2x2Stride2Pad0 ||
                              ConvForwardSpecialization ==
                                  ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0)
            {
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    is_compatible &= (arg.b_g_k_c_xs_lengths_[i + 3] == 2);
                }
            }
            else
            {
                static_assert(0, "not supported!");
            }

            // check stride
            if constexpr(ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter1x1Stride1Pad0 ||
                         ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter3x3Stride1Pad0 ||
                         ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
            {
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    is_compatible &= (arg.conv_filter_strides_[i] == 1);
                }
            }
            else if constexpr(ConvForwardSpecialization ==
                                  ConvolutionForwardSpecialization::Filter2x2Stride2Pad0 ||
                              ConvForwardSpecialization ==
                                  ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0)
            {
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    is_compatible &= (arg.conv_filter_strides_[i] == 2);
                }
            }

            // check pad, dilation
            if constexpr(ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter3x3Stride1Pad0 ||
                         ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
            {
                is_compatible &= (arg.input_left_pads_[0] == DilationX);
                is_compatible &= (arg.input_left_pads_[1] == DilationY);
                is_compatible &= (arg.input_right_pads_[0] == DilationX);
                is_compatible &= (arg.input_right_pads_[1] == DilationY);
                is_compatible &= (arg.conv_filter_dilations_[0] == DilationX);
                is_compatible &= (arg.conv_filter_dilations_[1] == DilationY);
            }
            else if constexpr(ConvForwardSpecialization ==
                              ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0)
            {
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    is_compatible &= (arg.input_left_pads_[i] == 0);
                    if(arg.a_g_n_c_wis_lengths_[i + 3] & 1)
                    {
                        is_compatible &= (arg.input_right_pads_[i] == 1);
                    }
                    else
                    {
                        is_compatible &= (arg.input_right_pads_[i] == 0);
                    }
                    is_compatible &= (arg.conv_filter_dilations_[i] == 1);
                }
            }
            else
            {
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    is_compatible &= (arg.input_left_pads_[i] == 0);
                    is_compatible &= (arg.input_right_pads_[i] == 0);
                    is_compatible &= (arg.conv_filter_dilations_[i] == 1);
                }
            }

            if constexpr(ConvForwardSpecialization ==
                         ConvolutionForwardSpecialization::Filter3x3Stride1Pad0)
            {
                is_compatible &= (arg.a_g_n_c_wis_lengths_[1] == 1);
                is_compatible &= (arg.e_g_n_k_wos_lengths_[1] == 1);
            }

            if constexpr(ConvForwardSpecialization ==
                         ConvolutionForwardSpecialization::Filter2x2Stride2Pad0)
            {
                for(index_t i = 0; i < NDimSpatial; i++)
                {
                    if(arg.a_g_n_c_wis_lengths_[i + 3] & 1)
                    {
                        is_compatible = false;
                    }
                }
            }

            return is_compatible;
        }
        else if constexpr(NDimSpatial == 3)
        {
            // Check filter size
            is_compatible &= (arg.b_g_k_c_xs_lengths_[NDimSpatial + 1] == FilterSize);
            is_compatible &= (arg.b_g_k_c_xs_lengths_[NDimSpatial + 2] == FilterSize);

            if constexpr(ConvForwardSpecialization ==
                         ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
            {
                is_compatible &= (arg.b_g_k_c_xs_lengths_[NDimSpatial + 0] == FilterSize);
            }

            // check stride
            if constexpr(ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter1x1Stride1Pad0 ||
                         ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
            {
                is_compatible &= (arg.conv_filter_strides_[1] == 1);
                is_compatible &= (arg.conv_filter_strides_[2] == 1);
                if constexpr(ConvForwardSpecialization ==
                             ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
                {
                    is_compatible &= (arg.conv_filter_strides_[0] == 1);
                }
            }

            // check pad, dilation
            if constexpr(ConvForwardSpecialization ==
                         ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0)
            {
                is_compatible &= (arg.input_left_pads_[1] == DilationX);
                is_compatible &= (arg.input_left_pads_[2] == DilationY);
                is_compatible &= (arg.input_right_pads_[1] == DilationX);
                is_compatible &= (arg.input_right_pads_[2] == DilationY);
                is_compatible &= (arg.conv_filter_dilations_[1] == DilationX);
                is_compatible &= (arg.conv_filter_dilations_[2] == DilationY);
            }
            else
            {
                const index_t StartDim =
                    ConvForwardSpecialization ==
                            ConvolutionForwardSpecialization::Filter1x1Stride1Pad0
                        ? 0
                        : 1;
                for(index_t i = StartDim; i < NDimSpatial; i++)
                {
                    is_compatible &= (arg.input_left_pads_[i] == 0);
                    is_compatible &= (arg.input_right_pads_[i] == 0);
                    is_compatible &= (arg.conv_filter_dilations_[i] == 1);
                }
            }

            return is_compatible;
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

        static_assert((ShuffleOnLoad == false) || (FilterSize == 2),
                      "ShuffleOnLoad only can be used in conv2");
        static_assert((Transposed == false) || (FilterSize == 2),
                      "Transposed conv only support conv2 for now.");
        static_assert((EnableSpatialCluster == false) || (ClusterDimSize != 0),
                      "ClusterDimSize shoule set when EnableSpatialCluster");
        static_assert((EnableSpatialCluster == false) || (FilterSize >= 3),
                      "Enable spatial cluster only when filterSize is >= 3");
        static_assert(ClusterDimSize <= 16, "Full cluster size is too large!");

        bool input_layout_compatible =
            IsConvLayoutCompatible<InLayout>(arg.a_g_n_c_wis_lengths_, arg.a_g_n_c_wis_strides_);
        if(input_layout_compatible == false)
        {
            printf("Input data incompatible with layout!\n");
            return false;
        }

        bool weight_layout_compatible =
            IsConvLayoutCompatible<WeiLayout>(arg.b_g_k_c_xs_lengths_, arg.b_g_k_c_xs_strides_);
        if(weight_layout_compatible == false)
        {
            printf("Weight data incompatible with layout!\n");
            return false;
        }

        bool output_layout_compatible =
            IsConvLayoutCompatible<ELayout>(arg.e_g_n_k_wos_lengths_, arg.e_g_n_k_wos_strides_);
        if(output_layout_compatible == false)
        {
            printf("Output data incompatible with layout!\n");
            return false;
        }

        // check ConvolutionForwardSpecialization
        bool conv_desc_compatible = IsConvSpecializationCompatible(arg);
        if(conv_desc_compatible == false)
        {
            printf("conv_desc_compatible is false!\n");
            return false;
        }

        // check g
        if(arg.a_g_n_c_wis_lengths_[0] != arg.b_g_k_c_xs_lengths_[0] ||
           arg.a_g_n_c_wis_lengths_[0] != arg.e_g_n_k_wos_lengths_[0])
        {
            printf("Group is incorrect:%d %d %d %d!\n",
                   arg.a_g_n_c_wis_lengths_[0],
                   arg.b_g_k_c_xs_lengths_[0],
                   arg.a_g_n_c_wis_lengths_[0],
                   arg.e_g_n_k_wos_lengths_[0]);
            return false;
        }

        for(index_t i = 0; i < NumDTensor; i++)
        {
            if(arg.a_g_n_c_wis_lengths_[0] != arg.ds_g_n_k_wos_lengths_[i][0])
            {

                printf("K is incorrect:%d %d!\n",
                       arg.a_g_n_c_wis_lengths_[0],
                       arg.ds_g_n_k_wos_lengths_[i][0]);
                return false;
            }
        }

        // check output dim
        {
            std::array<index_t, NDimSpatial + 3> output_spatial_lengths;
            if constexpr(Transposed == false)
            {

                static_for<0, NDimSpatial, 1>{}([&](auto i) {
                    const index_t x_eff =
                        (arg.b_g_k_c_xs_lengths_[i + 3] - 1) * arg.conv_filter_dilations_[i] + 1;
                    output_spatial_lengths[i] =
                        (arg.a_g_n_c_wis_lengths_[i + 3] + arg.input_left_pads_[i] +
                         arg.input_right_pads_[i] - x_eff) /
                            arg.conv_filter_strides_[i] +
                        1;
                });
            }
            else
            {
                static_for<0, NDimSpatial, 1>{}([&](auto i) {
                    const ck::long_index_t x_eff =
                        arg.b_g_k_c_xs_lengths_[i + 3] - arg.conv_filter_strides_[i];

                    output_spatial_lengths[i] =
                        arg.a_g_n_c_wis_lengths_[i + 3] * arg.conv_filter_strides_[i] + x_eff -
                        arg.input_left_pads_[i] - arg.input_right_pads_[i];
                });
            }
            for(index_t i = 0; i < NDimSpatial; i++)
            {
                if(output_spatial_lengths[i] != arg.e_g_n_k_wos_lengths_[i + 3])
                {
                    printf("output is incorrect:%d %d!\n",
                           output_spatial_lengths[i],
                           arg.e_g_n_k_wos_lengths_[i + 3]);
                    return false;
                }
            };
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

        // check vector access of WeiData
        if constexpr(is_same_v<WeiLayout, ctc::G_K_X_C> || is_same_v<WeiLayout, ctc::G_K_YX_C> ||
                     is_same_v<WeiLayout, ctc::G_K_ZYX_C> || is_same_v<WeiLayout, ctc::GKXC> ||
                     is_same_v<WeiLayout, ctc::GKYXC> || is_same_v<WeiLayout, ctc::GKZYXC> ||
                     is_same_v<WeiLayout, ctc::KXGC> || is_same_v<WeiLayout, ctc::KYXGC> ||
                     is_same_v<WeiLayout, ctc::KZYXGC>)

        {
            const index_t C = arg.b_g_k_c_xs_lengths_[2];

            if(!(C % WeiBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        //  check vector access of Ds
        bool valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            // FIXME: layout
            if constexpr(is_same_v<DLayout, ctc::NDHWGK> || is_same_v<DLayout, ctc::G_K>)
            {
                const index_t K = arg.ds_g_n_k_wos_lengths_[i][2];

                if(!(K % DsBlockTransferSrcScalarPerVector{}[i] == 0))
                {
                    valid = false;
                }
            }
            else if constexpr(is_same_v<DLayout, ctc::G_NW_K> || is_same_v<DLayout, ctc::G_NHW_K> ||
                              is_same_v<DLayout, ctc::G_NDHW_K> || is_same_v<DLayout, ctc::GNWK> ||
                              is_same_v<DLayout, ctc::GNHWK> || is_same_v<DLayout, ctc::GNDHWK> ||
                              is_same_v<DLayout, ctc::NWGK> || is_same_v<DLayout, ctc::NHWGK>)
            {
                // TODO: Validate output size.
            }
            else
            {
                valid = false;
            }
        });

        if(!valid)
        {
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
        return GridwiseConv::CheckValidity(arg.in_grid_desc_,
                                           arg.wei_grid_desc_,
                                           arg.ds_grid_desc_,
                                           arg.e_grid_desc_,
                                           arg.block_2_etile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const void* p_in,
        const void* p_wei,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const InElementwiseOperation& in_element_op,
        const WeiElementwiseOperation& wei_element_op,
        const AccElementwiseOperation& acc_element_op)
    {
        return Argument{p_in,
                        p_wei,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_k_wos_lengths,
                        ds_g_n_k_wos_strides,
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
                 const void* p_b,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<long_index_t, NDimSpatial>& input_left_pads,
                 const std::array<long_index_t, NDimSpatial>& input_right_pads,
                 const InElementwiseOperation& in_element_op,
                 const WeiElementwiseOperation& wei_element_op,
                 const AccElementwiseOperation& acc_element_op)
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i32;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_dilations_i32;
        std::array<index_t, NDimSpatial> input_left_pads_i32;
        std::array<index_t, NDimSpatial> input_right_pads_i32;

        array_convert(a_g_n_c_wis_lengths_i32, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i32, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_i32, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_i32, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_i32[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_g_n_k_wos_strides_i32[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_g_n_k_wos_lengths_i32, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i32, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_i32, conv_filter_strides);
        array_convert(conv_filter_dilations_i32, conv_filter_dilations);
        array_convert(input_left_pads_i32, input_left_pads);
        array_convert(input_right_pads_i32, input_right_pads);

        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths_i32,
                        a_g_n_c_wis_strides_i32,
                        b_g_k_c_xs_lengths_i32,
                        b_g_k_c_xs_strides_i32,
                        ds_g_n_k_wos_lengths_i32,
                        ds_g_n_k_wos_strides_i32,
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
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_in,
        const void* p_wei,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const InElementwiseOperation& in_element_op,
        const WeiElementwiseOperation& wei_element_op,
        const AccElementwiseOperation& acc_element_op) override
    {
        return std::make_unique<Argument>(p_in,
                                          p_wei,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_k_wos_lengths,
                                          ds_g_n_k_wos_strides,
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

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const std::array<const void*, NumDTensor>& p_ds,
                        void* p_e,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_lengths,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<long_index_t, NDimSpatial>& input_left_pads,
                        const std::array<long_index_t, NDimSpatial>& input_right_pads,
                        const InElementwiseOperation& in_element_op,
                        const WeiElementwiseOperation& wei_element_op,
                        const AccElementwiseOperation& acc_element_op) override
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i32;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_dilations_i32;
        std::array<index_t, NDimSpatial> input_left_pads_i32;
        std::array<index_t, NDimSpatial> input_right_pads_i32;

        array_convert(a_g_n_c_wis_lengths_i32, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i32, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_i32, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_i32, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_i32[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_g_n_k_wos_strides_i32[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_g_n_k_wos_lengths_i32, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i32, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_i32, conv_filter_strides);
        array_convert(conv_filter_dilations_i32, conv_filter_dilations);
        array_convert(input_left_pads_i32, input_left_pads);
        array_convert(input_right_pads_i32, input_right_pads);

        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths_i32,
                                          a_g_n_c_wis_strides_i32,
                                          b_g_k_c_xs_lengths_i32,
                                          b_g_k_c_xs_strides_i32,
                                          ds_g_n_k_wos_lengths_i32,
                                          ds_g_n_k_wos_strides_i32,
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
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvFwdMultipleD_Wcnn_CShuffle"
            << "<"
            << BlockSize << ", "
            << HPerBlock << ", "
            << WPerBlock << ", "
            << CPerBlock << ", "
            << KPerBlock << ", "
            << HRepeat << ", "
            << WRepeat << ", "
            << HPerWcnn << ", "
            << WPerWcnn << ", "
            << FilterSize << ", "
            << DilationX << ", "
            << DilationY << ", "
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
