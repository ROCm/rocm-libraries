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
    // Broadcasting utility functions
    static std::vector<int64_t> computeBroadcastShape(
        const std::vector<std::vector<int64_t>>& input_shapes)
    {
        if(input_shapes.empty())
        {
            throw std::runtime_error("Cannot compute broadcast shape for empty input shapes");
        }

        // Find the maximum number of dimensions
        size_t max_dims = 0;
        for(const auto& shape : input_shapes)
        {
            max_dims = std::max(max_dims, shape.size());
        }

        std::vector<int64_t> broadcast_shape(max_dims, 1);

        // Process each dimension from right to left (NumPy broadcasting rules)
        for(size_t dim = 0; dim < max_dims; ++dim)
        {
            int64_t max_size = 1;
            
            for(const auto& shape : input_shapes)
            {
                // Get dimension size (1 if dimension doesn't exist)
                int64_t dim_size = 1;
                if(dim < shape.size())
                {
                    size_t shape_dim_idx = shape.size() - 1 - dim;
                    dim_size = shape[shape_dim_idx];
                }

                // Check broadcasting compatibility
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

    static std::vector<int64_t> getBroadcastableIndex(
        const std::vector<int64_t>& output_index,
        const std::vector<int64_t>& tensor_dims)
    {
        if(output_index.size() < tensor_dims.size())
        {
            throw std::runtime_error("Output index has fewer dimensions than tensor");
        }

        // Return index vector with same size as tensor dimensions
        std::vector<int64_t> broadcasted_index(tensor_dims.size());
        
        // Handle dimension alignment (right-aligned)
        size_t dim_offset = output_index.size() - tensor_dims.size();
        
        for(size_t i = 0; i < tensor_dims.size(); ++i)
        {
            size_t output_dim_idx = dim_offset + i;
            // If dimension is 1, always broadcast to index 0
            broadcasted_index[i] = (tensor_dims[i] == 1) ? 0 : output_index[output_dim_idx];
        }

        return broadcasted_index;
    }

    template <typename Op>
    void executeBinaryBroadcast(const std::vector<const TensorBase<DataType>*>& inputs,
                                TensorBase<DataType>& output,
                                Op op)
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("Binary operation requires exactly 2 input tensors.");
        }

        const auto& input1 = *inputs[0];
        const auto& input2 = *inputs[1];

        // Get input shapes
        std::vector<std::vector<int64_t>> input_shapes = {
            input1.dims(), input2.dims()
        };

        // Compute broadcast output shape
        auto broadcast_shape = computeBroadcastShape(input_shapes);

        // Verify output matches broadcast shape
        if(output.dims() != broadcast_shape)
        {
            throw std::runtime_error("Output shape doesn't match computed broadcast shape");
        }

        // Iterate over output shape
        auto func = [&](const std::vector<std::size_t>& indices) {
            std::vector<int64_t> output_idx(indices.begin(), indices.end());

            // Get broadcasted indices for each input
            auto input1_idx = getBroadcastableIndex(output_idx, input1.dims());
            auto input2_idx = getBroadcastableIndex(output_idx, input2.dims());

            // Get values and compute
            auto input1Val = input1.getHostValue(input1_idx);
            auto input2Val = input2.getHostValue(input2_idx);

            DataType outputVal;
            op(outputVal, input1Val, input2Val);
            output.setHostValue(outputVal, output_idx);
        };

        // Use dynamic parallel functor that supports any number of dimensions
        auto parallelFunc = makeParallelTensorFunctor(func, broadcast_shape);
        parallelFunc();
    }

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

        auto func = [&](const std::vector<std::size_t>& indices) {
            std::vector<int64_t> idxVec(indices.begin(), indices.end());
            auto input1Val = input1.getHostValue(idxVec);
            auto input2Val = input2.getHostValue(idxVec);
            DataType outputVal;
            op(outputVal, input1Val, input2Val);
            output.setHostValue(outputVal, idxVec);
        };

        // Use dynamic parallel functor that supports any number of dimensions
        auto parallelFunc = makeParallelTensorFunctor(func, dims);
        parallelFunc();
    }

    void markOutputModified(TensorBase<DataType>& output)
    {
        output.memory().markHostModified();
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
