// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/pointwise/CpuDeviceExecutor.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/ReferencePointwiseBase.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

template <class OutputType, class DeviceExecutor, class... InputTypes>
class ReferencePointwiseImpl
{
public:
    static bool isApplicable(const hipdnn_sdk::data_objects::Node& node)
    {
        return ReferencePointwiseBase<OutputType, DeviceExecutor, InputTypes...>::isApplicable(node);
    }

    template<typename... Tensors>
    static void pointwiseForward(hipdnn_sdk::data_objects::PointwiseMode operation,
                                Tensors&&... tensors_and_output)
    {
        ReferencePointwiseBase<OutputType, DeviceExecutor, InputTypes...>::pointwiseForward(
            operation, std::forward<Tensors>(tensors_and_output)...);
    }
};

// Generic N-ary type alias for CPU operations
template <class OutputType, class... InputTypes>
using CpuReferencePointwiseImpl = ReferencePointwiseImpl<OutputType, CpuDeviceExecutor<OutputType, InputTypes...>, InputTypes...>;

} // namespace test_utilities
} // namespace hipdnn_sdk
