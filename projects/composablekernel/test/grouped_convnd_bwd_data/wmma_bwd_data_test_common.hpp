// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_v3_instances.hpp"

namespace ck::test {

// Each K=128 split-4 partial must contain a full K=32 tile and use paired stores.
template <typename Instances, std::size_t I = 0>
constexpr std::size_t FindWmmaBwdDataSplitKInstance()
{
    if constexpr(I == std::tuple_size_v<Instances>)
    {
        return I;
    }
    else
    {
        using Op   = std::tuple_element_t<I, Instances>;
        using Grid = typename Op::GridwiseGemm::Base;
        if constexpr(Op::IsSplitKSupported &&
                     Op::CShuffleBlockTransferScalarPerVector_NPerBlock == 8 &&
                     Grid::MaxBlockSize == 64 && Grid::AK0Number * Grid::AK1Number == 32)
        {
            return I;
        }
        else
        {
            return FindWmmaBwdDataSplitKInstance<Instances, I + 1>();
        }
    }
}

template <typename DataType, index_t NDimSpatial = 2>
struct WmmaBwdDataSplitKInstance
{
    using OutLayout = std::conditional_t<NDimSpatial == 2,
                                         tensor_layout::convolution::NHWGK,
                                         tensor_layout::convolution::NDHWGK>;
    using WeiLayout = std::conditional_t<NDimSpatial == 2,
                                         tensor_layout::convolution::GKYXC,
                                         tensor_layout::convolution::GKZYXC>;
    using InLayout  = std::conditional_t<NDimSpatial == 2,
                                         tensor_layout::convolution::NHWGC,
                                         tensor_layout::convolution::NDHWGC>;
    using Instances = std::conditional_t<
        std::is_same_v<DataType, half_t>,
        tensor_operation::device::instance::device_grouped_conv_bwd_data_wmma_v3_f16_instances<
            NDimSpatial,
            OutLayout,
            WeiLayout,
            Tuple<>,
            InLayout,
            tensor_operation::device::ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0>,
        tensor_operation::device::instance::device_grouped_conv_bwd_data_wmma_v3_bf16_instances<
            NDimSpatial,
            OutLayout,
            WeiLayout,
            Tuple<>,
            InLayout,
            tensor_operation::device::ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0>>;
    static constexpr auto Index = FindWmmaBwdDataSplitKInstance<Instances>();
    static_assert(Index < std::tuple_size_v<Instances>, "No paired-store K=32 WMMA instance");
    using type = std::tuple_element_t<Index, Instances>;
};

template <typename DataType, index_t NDimSpatial = 2>
using WmmaBwdDataSplitKOp = typename WmmaBwdDataSplitKInstance<DataType, NDimSpatial>::type;

} // namespace ck::test
