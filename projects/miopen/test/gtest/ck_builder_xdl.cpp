// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_builder_shared.hpp"

#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

using InLayout                             = ck::tensor_layout::convolution::NHWGC;
using WeiLayout                            = ck::tensor_layout::convolution::GKYXC;
using OutLayout                            = ck::tensor_layout::convolution::NHWGK;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using EmptyTuple                           = ck::Tuple<>;
static constexpr ck::index_t NumDimSpatial = 2;
template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>;
template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdDefaultPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdDefault<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdBuilderPtrs = miopen::conv::ck_builder::instance::DeviceOperationInstanceFactory<
    DeviceOpGFwdDefault<DataType, ComputeType>>;

template <typename DataType>
void CompareInstanceLists()
{
    auto ckFactoryInstances      = DeviceOpGFwdDefaultPtrs<DataType>::GetInstances();
    auto builderFactoryInstances = DeviceOpGFwdBuilderPtrs<DataType>::GetInstances();

    compare_instance_vectors(ckFactoryInstances, builderFactoryInstances);
}

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsFloat) { CompareInstanceLists<float>(); }

/*
TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsHalf) { CompareInstanceLists<ck::half_t>(); }

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsBHalf) { CompareInstanceLists<ck::bhalf_t>(); }

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsInt8) { CompareInstanceLists<int8_t>(); }
*/
