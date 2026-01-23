// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <miopen/ck_builder/factories/base.hpp>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

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

template <>
struct DeviceOperationInstanceFactory<DeviceOpGFwdDefault<float>>
{
    static std::vector<BaseOperatorPtr> GetInstances();
};
} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
