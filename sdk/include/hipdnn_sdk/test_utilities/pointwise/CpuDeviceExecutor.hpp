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
class CpuDeviceExecutor
{
public:
    template <typename Op>
    void executeBinaryBroadcast(const std::vector<const TensorBase<DataType>*>& inputs,
                                TensorBase<DataType>& output,
                                Op op)
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("Binary operation requires exactly 2 input tensors.");
        }

        for(size_t i = 0; i < inputs.size(); ++i)
        {
            if(inputs[i] == nullptr)
            {
                throw std::runtime_error("Input tensor " + std::to_string(i) + " is null.");
            }
        }

        const auto& input1 = *inputs[0];
        const auto& input2 = *inputs[1];

        std::vector<std::vector<int64_t>> input_shapes = {
            input1.dims(), input2.dims()
        };

        auto broadcast_shape = computeBroadcastShape(input_shapes);

        if(output.dims() != broadcast_shape)
        {
            throw std::runtime_error("Output shape doesn't match computed broadcast shape");
        }

        auto func = [&](const std::vector<int64_t>& indices) {
            auto input1_idx = getBroadcastableIndex(indices, input1.dims());
            auto input2_idx = getBroadcastableIndex(indices, input2.dims());

            auto input1Val = input1.getHostValue(input1_idx);
            auto input2Val = input2.getHostValue(input2_idx);

            DataType outputVal;
            op(outputVal, input1Val, input2Val);
            output.setHostValue(outputVal, indices);
        };

        auto parallelFunc = makeParallelTensorFunctor(func, broadcast_shape);
        parallelFunc();
    }


    void markOutputModified(TensorBase<DataType>& output)
    {
        output.memory().markHostModified();
    }
private: 

    static std::vector<int64_t> computeBroadcastShape(
        const std::vector<std::vector<int64_t>>& input_shapes)
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

    static std::vector<int64_t> getBroadcastableIndex(
        const std::vector<int64_t>& output_index,
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
