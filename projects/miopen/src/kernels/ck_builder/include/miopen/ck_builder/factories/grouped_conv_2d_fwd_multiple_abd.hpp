// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <miopen/ck_builder/factories/base.hpp>
#include <miopen/ck_builder/instances/grouped_conv_fwd_2d_f32.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
using InLayout    = ck::tensor_layout::convolution::NGCHW;
using WeiLayout   = ck::tensor_layout::convolution::GKCYX;
using OutLayout   = ck::tensor_layout::convolution::NGKHW;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using EmptyTuple  = ck::Tuple<>;
template <typename DataType>
using DeviceOpGFwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
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
                                                                  PassThrough,
                                                                  DataType,
                                                                  DataType>;

using DeviceOpGFWdDefaultFloat = DeviceOpGFwdDefault<float>;
template <>
struct DeviceOperationInstanceFactory<DeviceOpGFWdDefaultFloat>
{
    static std::vector<BaseOperatorPtr> GetInstances()
    {
        std::vector<BaseOperatorPtr> instances{};
        add_grouped_conv_fwd_2d_f32(instances);
        return instances;
    }
};
} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
