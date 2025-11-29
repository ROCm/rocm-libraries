// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/pointwise/CpuDeviceExecutor.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/ReferencePointwiseBase.hpp>

namespace hipdnn_sdk::test_utilities
{

template <class DeviceExecutor, class OutputType, class... InputTypes>
class ReferencePointwiseImpl
{
public:
    static bool isApplicable(const data_objects::Node& node)
    {
        return ReferencePointwiseBase<DeviceExecutor, OutputType, InputTypes...>::isApplicable(
            node);
    }

    template <typename... Args>
    static void pointwiseCompute(data_objects::PointwiseMode operation,
                                 utilities::TensorBase<OutputType>& output,
                                 Args&&... args)
    {
        ReferencePointwiseBase<DeviceExecutor, OutputType, InputTypes...>::pointwiseCompute(
            operation, output, std::forward<Args>(args)...);
    }
};

// Generic N-ary type alias for CPU operations
template <class OutputType, class... InputTypes>
using CpuReferencePointwiseImpl
    = ReferencePointwiseImpl<CpuDeviceExecutor<OutputType, InputTypes...>,
                             OutputType,
                             InputTypes...>;

} // namespace hipdnn_sdk::test_utilities
