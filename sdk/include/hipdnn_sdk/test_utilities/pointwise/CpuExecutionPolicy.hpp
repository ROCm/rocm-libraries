// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <stdexcept>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class DataType>
class CpuExecutionPolicy
{
public:
    template <typename Op>
    void executeBinary(const std::vector<const TensorBase<DataType>*>& inputs,
                       TensorBase<DataType>& output,
                       Op op)
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("Binary operation requires exactly 2 input tensors.");
        }

        const auto& input1 = *inputs[0];
        const auto& input2 = *inputs[1];
        const auto& dims = output.dims();

        auto func = [&](const auto&... indices) {
            std::vector<int64_t> idxVec = {static_cast<int64_t>(indices)...};
            auto input1Val = input1.getHostValue(idxVec);
            auto input2Val = input2.getHostValue(idxVec);
            DataType outputVal;
            op(outputVal, input1Val, input2Val);
            output.setHostValue(outputVal, idxVec);
        };

        if(dims.size() != 4)
        {
            throw std::runtime_error("Only 4D tensors are supported, got "
                                     + std::to_string(dims.size()) + "D");
        }
        auto parallelFunc = makeParallelTensorFunctor(func,
                                                      static_cast<std::size_t>(dims[0]),
                                                      static_cast<std::size_t>(dims[1]),
                                                      static_cast<std::size_t>(dims[2]),
                                                      static_cast<std::size_t>(dims[3]));
        parallelFunc();
    }

    void markOutputModified(TensorBase<DataType>& output)
    {
        output.memory().markHostModified();
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
