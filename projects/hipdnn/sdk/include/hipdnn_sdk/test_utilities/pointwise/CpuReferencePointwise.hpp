// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/pointwise/CpuDeviceExecutor.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/ReferencePointwiseBase.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

template <class DataType, class DeviceExecutor = CpuDeviceExecutor<DataType>>
class ReferencePointwiseImpl
{
public:
    static bool isApplicable(const hipdnn_sdk::data_objects::Node& node)
    {
        return ReferencePointwiseBase<DataType, DeviceExecutor>::isApplicable(node);
    }

    static void pointwiseForward(const std::vector<const TensorBase<DataType>*>& inputs,
                                 TensorBase<DataType>& output,
                                 hipdnn_sdk::data_objects::PointwiseMode operation)
    {
        ReferencePointwiseBase<DataType, DeviceExecutor>::pointwiseForward(
            inputs, output, operation);
    }
};

template <class DataType>
using CpuReferencePointwiseImpl = ReferencePointwiseImpl<DataType, CpuDeviceExecutor<DataType>>;

} // namespace test_utilities
} // namespace hipdnn_sdk
