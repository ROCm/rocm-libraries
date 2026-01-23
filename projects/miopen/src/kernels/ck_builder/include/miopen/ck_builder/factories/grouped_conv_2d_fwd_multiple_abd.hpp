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

// F32 instance builder functions
void add_f32_merged_groups_instances(std::vector<BaseOperatorPtr>& instances);
void add_f32_standard_instances(std::vector<BaseOperatorPtr>& instances);
void add_f32_16x16_instances(std::vector<BaseOperatorPtr>& instances);
void add_f32_comp_instances(std::vector<BaseOperatorPtr>& instances);
void add_f32_mem_intra_instances(std::vector<BaseOperatorPtr>& instances);
void add_f32_mem_inter_instances(std::vector<BaseOperatorPtr>& instances);

// F16 instance builder functions
void add_f16_merged_groups_instances(std::vector<BaseOperatorPtr>& instances);
void add_f16_standard_instances(std::vector<BaseOperatorPtr>& instances);
void add_f16_16x16_instances(std::vector<BaseOperatorPtr>& instances);
void add_f16_comp_instances(std::vector<BaseOperatorPtr>& instances);
void add_f16_comp_2x_instances(std::vector<BaseOperatorPtr>& instances);
void add_f16_comp_part2_instances(std::vector<BaseOperatorPtr>& instances);
void add_f16_mem_intra_instances(std::vector<BaseOperatorPtr>& instances);
void add_f16_mem_inter_instances(std::vector<BaseOperatorPtr>& instances);

template <>
struct DeviceOperationInstanceFactory<DeviceOpGFwdDefault<float>>
{
    static std::vector<BaseOperatorPtr> GetInstances();
};

template <>
struct DeviceOperationInstanceFactory<DeviceOpGFwdDefault<ck::half_t>>
{
    static std::vector<BaseOperatorPtr> GetInstances();
};
} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
