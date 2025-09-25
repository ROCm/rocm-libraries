// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <stdexcept>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class OutputType, class... InputTypes>
class CpuDeviceExecutor
{
public:
    template <typename Op, typename Input1Type, typename Input2Type>
    void executeBinaryBroadcast(const TensorBase<Input1Type>& input1,
                               const TensorBase<Input2Type>& input2,
                               TensorBase<OutputType>& output,
                               Op op)
    {
        const auto& input1_dims = input1.dims();
        const auto& input2_dims = input2.dims();
        const auto& output_dims = output.dims();

        // Validate broadcast compatibility using existing utility function
        if(!areDimensionsBroadcastCompatible(input1_dims, output_dims))
        {
            throw std::runtime_error("Input1 dimensions are not broadcast compatible with output");
        }

        if(!areDimensionsBroadcastCompatible(input2_dims, output_dims))
        {
            throw std::runtime_error("Input2 dimensions are not broadcast compatible with output");
        }

        // Use output dimensions as the broadcast shape
        const auto& broadcast_shape = output_dims;

        auto func = [&](const std::vector<int64_t>& indices) {
            // Get broadcasted indices for each input
            auto input1_indices = getBroadcastableIndex(indices, input1_dims);
            auto input2_indices = getBroadcastableIndex(indices, input2_dims);

            // Get values from input tensors and apply operation
            auto input1_value = input1.getHostValue(input1_indices);
            auto input2_value = input2.getHostValue(input2_indices);

            // Apply operation and set output
            auto result = op(input1_value, input2_value);
            output.setHostValue(static_cast<OutputType>(result), indices);
        };

        auto parallelFunc = makeParallelTensorFunctor(func, broadcast_shape);
        parallelFunc();
    }


    void markOutputModified(TensorBase<OutputType>& output)
    {
        output.memory().markHostModified();
    }

private:
    static std::vector<int64_t> getBroadcastableIndex(const std::vector<int64_t>& broadcast_index,
                                                      const std::vector<int64_t>& tensor_dims)
    {
        if(broadcast_index.size() < tensor_dims.size())
        {
            throw std::runtime_error("Broadcast index has fewer dimensions than tensor");
        }

        std::vector<int64_t> broadcasted_index(tensor_dims.size());

        size_t dim_offset = broadcast_index.size() - tensor_dims.size();

        for(size_t i = 0; i < tensor_dims.size(); ++i)
        {
            size_t broadcast_dim_idx = dim_offset + i;
            broadcasted_index[i] = (tensor_dims[i] == 1) ? 0 : broadcast_index[broadcast_dim_idx];
        }

        return broadcasted_index;
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
