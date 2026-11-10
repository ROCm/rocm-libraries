// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

/// CPU reference for Gather (index_select / take) along one axis. With indices of rank m:
///   `y[a_0..a_{axis-1}, i_0..i_{m-1}, b...] = x[a_0..a_{axis-1}, indices[i_0..i_{m-1}], b...]`
/// so `y.dims = x.dims[:axis] ++ indices.dims ++ x.dims[axis + 1:]`. Every index must lie in
/// `[0, x.dims[axis])`; any other value throws `std::runtime_error` before y is written.
/// x and y may have arbitrary (non-packed) strides; values are copied without conversion.
class CpuFpReferenceGather
{
public:
    template <class DataType, class IndexType>
    static void gather(const hipdnn_data_sdk::utilities::TensorBase<DataType>& x,
                       const hipdnn_data_sdk::utilities::TensorBase<IndexType>& indices,
                       hipdnn_data_sdk::utilities::TensorBase<DataType>& y,
                       int64_t axis)
    {
        validateInput(x, indices, y, axis);

        const auto& indexDims = indices.dims();
        const auto& yDims = y.dims();
        const auto axisIndex = static_cast<size_t>(axis);
        const auto indexRank = indexDims.size();
        const int64_t axisExtent = x.dims()[axisIndex];

        // Validate every index before writing any output so a bad index leaves y untouched.
        int64_t indexCount = 1;
        for(const auto dim : indexDims)
        {
            indexCount *= dim;
        }
        if(indexCount > 0)
        {
            std::vector<int64_t> position(indexRank, 0);
            for(int64_t flat = 0; flat < indexCount; ++flat)
            {
                const auto value = static_cast<int64_t>(indices.getHostValue(position));
                if(value < 0 || value >= axisExtent)
                {
                    throw std::runtime_error("CpuFpReferenceGather: index " + std::to_string(value)
                                             + " is out of range [0, " + std::to_string(axisExtent)
                                             + ")");
                }
                for(size_t d = indexRank; d-- > 0;)
                {
                    if(++position[d] < indexDims[d])
                    {
                        break;
                    }
                    position[d] = 0;
                }
            }
        }

        // Raw pointers and strides are hoisted once; the parallel body never goes through
        // the tensors' host accessors.
        const DataType* xBase = x.memory().hostData();
        const IndexType* indexBase = indices.memory().hostData();
        DataType* yBase = y.memory().hostData();
        const auto& xStrides = x.strides();
        const auto& indexStrides = indices.strides();
        const auto& yStrides = y.strides();
        const auto xRank = x.dims().size();

        auto gatherElement = [&](const std::vector<int64_t>& yIndex) {
            int64_t indexOffset = 0;
            for(size_t d = 0; d < indexRank; ++d)
            {
                indexOffset += yIndex[axisIndex + d] * indexStrides[d];
            }
            const auto gathered = static_cast<int64_t>(indexBase[indexOffset]);

            int64_t xOffset = gathered * xStrides[axisIndex];
            for(size_t d = 0; d < axisIndex; ++d)
            {
                xOffset += yIndex[d] * xStrides[d];
            }
            for(size_t d = axisIndex + 1; d < xRank; ++d)
            {
                xOffset += yIndex[d - 1 + indexRank] * xStrides[d];
            }

            int64_t yOffset = 0;
            for(size_t d = 0; d < yIndex.size(); ++d)
            {
                yOffset += yIndex[d] * yStrides[d];
            }

            yBase[yOffset] = xBase[xOffset];
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(gatherElement, yDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
    }

private:
    template <class DataType, class IndexType>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<DataType>& x,
                              const hipdnn_data_sdk::utilities::TensorBase<IndexType>& indices,
                              const hipdnn_data_sdk::utilities::TensorBase<DataType>& y,
                              int64_t axis)
    {
        const std::string prefix = "CpuFpReferenceGather: ";

        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, prefix, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(indices, prefix, "indices");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(y, prefix, "y");

        const auto& xDims = x.dims();
        const auto& indexDims = indices.dims();
        if(xDims.empty() || indexDims.empty())
        {
            throw std::runtime_error(prefix + "x and indices must have rank >= 1");
        }
        if(axis < 0 || axis >= static_cast<int64_t>(xDims.size()))
        {
            throw std::runtime_error(prefix + "axis " + std::to_string(axis)
                                     + " must satisfy 0 <= axis < rank(x)");
        }

        const auto axisIndex = static_cast<size_t>(axis);
        std::vector<int64_t> expected(xDims.begin(),
                                      xDims.begin() + static_cast<std::ptrdiff_t>(axisIndex));
        expected.insert(expected.end(), indexDims.begin(), indexDims.end());
        expected.insert(expected.end(),
                        xDims.begin() + static_cast<std::ptrdiff_t>(axisIndex + 1),
                        xDims.end());
        if(y.dims() != expected)
        {
            throw std::runtime_error(
                prefix + "y dims must equal x.dims[:axis] ++ indices.dims ++ x.dims[axis + 1:]");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
