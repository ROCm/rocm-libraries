// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_test_sdk/utilities/pointwise/CpuDeviceExecutor.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/ReferencePointwiseBase.hpp>

namespace hipdnn_test_sdk::utilities
{

template <class DeviceExecutor, class OutputType, class ComputeType, class... InputTypes>
class ReferencePointwiseImpl
{
public:
    static bool isApplicable(const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        return ReferencePointwiseBase<DeviceExecutor, OutputType, ComputeType, InputTypes...>::
            isApplicable(node);
    }

    template <typename... Args>
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 Args&&... args)
    {
        ReferencePointwiseBase<DeviceExecutor, OutputType, ComputeType, InputTypes...>::
            pointwiseCompute(operation, output, std::forward<Args>(args)...);
    }
};

// Generic N-ary type alias for CPU operations
template <class OutputType, class ComputeType, class... InputTypes>
using CpuReferencePointwiseImpl
    = ReferencePointwiseImpl<CpuDeviceExecutor<OutputType, InputTypes...>,
                             OutputType,
                             ComputeType,
                             InputTypes...>;

} // namespace hipdnn_test_sdk::utilities
