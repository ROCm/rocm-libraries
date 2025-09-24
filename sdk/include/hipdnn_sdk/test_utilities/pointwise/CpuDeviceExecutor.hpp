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

template <class OutputType, class... InputTypes>
class CpuDeviceExecutor
{
public:
    // Binary operation with explicit two inputs
    template <typename Op, typename Input1Type, typename Input2Type>
    void executeBinaryBroadcast(const TensorBase<Input1Type>& input1,
                               const TensorBase<Input2Type>& input2,
                               TensorBase<OutputType>& output,
                               Op op)
    {
        // Get input shapes
        std::vector<std::vector<int64_t>> input_shapes = {input1.dims(), input2.dims()};

        auto broadcast_shape = computeBroadcastShape(input_shapes);

        if(output.dims() != broadcast_shape)
        {
            throw std::runtime_error("Output shape doesn't match computed broadcast shape");
        }

        auto func = [&](const std::vector<int64_t>& indices) {
            // Get broadcasted indices for each input
            auto input1_indices = getBroadcastableIndex(indices, input1.dims());
            auto input2_indices = getBroadcastableIndex(indices, input2.dims());

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
    static std::vector<int64_t>
        computeBroadcastShape(const std::vector<std::vector<int64_t>>& input_shapes)
    {
        if(input_shapes.empty())
        {
            throw std::runtime_error("Cannot compute broadcast shape for empty input shapes");
        }

        size_t max_dims = 0;
        for(const auto& shape : input_shapes)
        {
            max_dims = std::max(max_dims, shape.size());
        }

        std::vector<int64_t> broadcast_shape(max_dims, 1);

        for(size_t dim = 0; dim < max_dims; ++dim)
        {
            int64_t max_size = 1;

            for(const auto& shape : input_shapes)
            {
                int64_t dim_size = 1;
                if(dim < shape.size())
                {
                    size_t shape_dim_idx = shape.size() - 1 - dim;
                    dim_size = shape[shape_dim_idx];
                }

                if(max_size == 1)
                {
                    max_size = dim_size;
                }
                else if(dim_size != 1 && dim_size != max_size)
                {
                    throw std::runtime_error("Cannot broadcast shapes: incompatible dimensions");
                }
                else if(dim_size > max_size)
                {
                    max_size = dim_size;
                }
            }

            broadcast_shape[max_dims - 1 - dim] = max_size;
        }

        return broadcast_shape;
    }

    static std::vector<int64_t> getBroadcastableIndex(const std::vector<int64_t>& output_index,
                                                      const std::vector<int64_t>& tensor_dims)
    {
        if(output_index.size() < tensor_dims.size())
        {
            throw std::runtime_error("Output index has fewer dimensions than tensor");
        }

        std::vector<int64_t> broadcasted_index(tensor_dims.size());

        size_t dim_offset = output_index.size() - tensor_dims.size();

        for(size_t i = 0; i < tensor_dims.size(); ++i)
        {
            size_t output_dim_idx = dim_offset + i;
            broadcasted_index[i] = (tensor_dims[i] == 1) ? 0 : output_index[output_dim_idx];
        }

        return broadcasted_index;
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
