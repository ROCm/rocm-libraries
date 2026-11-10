// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

/// CPU reference for top-k selection along the last dim. For every row (all leading dims of
/// x), writes the positions of the k largest (largest = true) or smallest entries, sorted
/// best-first, to indices, and their values to values when it is given. Equal values keep the
/// lower position first, so the result is deterministic; NaN compares greater than every
/// number (first when selecting the largest, last when selecting the smallest; NaNs among
/// themselves keep position order). Values are compared in ComputeType and copied without
/// conversion. indices and values have the dims of x with the last dim k; every tensor may
/// have arbitrary (non-packed) strides.
class CpuFpReferenceTopk
{
public:
    template <class DataType, class IndexType, class ComputeType = float>
    static void topk(const hipdnn_data_sdk::utilities::TensorBase<DataType>& x,
                     hipdnn_data_sdk::utilities::TensorBase<IndexType>& indices,
                     hipdnn_data_sdk::utilities::TensorBase<DataType>* values,
                     int64_t k,
                     bool largest)
    {
        validateInput(x, indices, values, k);

        const auto& xDims = x.dims();
        const auto rank = xDims.size();
        const int64_t rowLength = xDims[rank - 1];
        const std::vector<int64_t> rowDims(xDims.begin(),
                                           xDims.begin() + static_cast<std::ptrdiff_t>(rank - 1));
        int64_t rowCount = 1;
        for(const auto dim : rowDims)
        {
            rowCount *= dim;
        }
        if(rowCount == 0)
        {
            return;
        }

        // Raw pointers and strides are hoisted once; the parallel body never goes through
        // the tensors' host accessors.
        const DataType* xBase = x.memory().hostData();
        IndexType* indicesBase = indices.memory().hostData();
        DataType* valuesBase = values != nullptr ? values->memory().hostData() : nullptr;
        const auto& xStrides = x.strides();
        const auto& indicesStrides = indices.strides();
        const std::vector<int64_t> noStrides;
        const auto& valuesStrides = values != nullptr ? values->strides() : noStrides;
        const int64_t xLast = xStrides[rank - 1];
        const int64_t indicesLast = indicesStrides[rank - 1];
        const int64_t valuesLast = values != nullptr ? valuesStrides[rank - 1] : 0;

        // a ranks strictly before b. NaN is greater than every number; std::stable_sort keeps
        // ties (including NaN pairs) in position order.
        auto ranksBefore = [largest](ComputeType a, ComputeType b) {
            const bool aNan = std::isnan(a);
            const bool bNan = std::isnan(b);
            if(aNan || bNan)
            {
                return largest ? (aNan && !bNan) : (bNan && !aNan);
            }
            return largest ? a > b : a < b;
        };

        auto selectRow = [&](std::vector<int64_t>& order, const std::vector<int64_t>& flatRow) {
            // Decompose the flat row number into the leading-dim coordinates.
            int64_t remaining = flatRow[0];
            int64_t xRow = 0;
            int64_t indicesRow = 0;
            int64_t valuesRow = 0;
            for(size_t d = rowDims.size(); d-- > 0;)
            {
                const int64_t coordinate = remaining % rowDims[d];
                remaining /= rowDims[d];
                xRow += coordinate * xStrides[d];
                indicesRow += coordinate * indicesStrides[d];
                if(valuesBase != nullptr)
                {
                    valuesRow += coordinate * valuesStrides[d];
                }
            }

            order.resize(static_cast<size_t>(rowLength));
            std::iota(order.begin(), order.end(), int64_t{0});
            std::stable_sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
                return ranksBefore(static_cast<ComputeType>(xBase[xRow + lhs * xLast]),
                                   static_cast<ComputeType>(xBase[xRow + rhs * xLast]));
            });

            for(int64_t j = 0; j < k; ++j)
            {
                const int64_t position = order[static_cast<size_t>(j)];
                indicesBase[indicesRow + j * indicesLast] = static_cast<IndexType>(position);
                if(valuesBase != nullptr)
                {
                    valuesBase[valuesRow + j * valuesLast] = xBase[xRow + position * xLast];
                }
            }
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctorWithScratch<std::vector<int64_t>>(
                selectRow, std::vector<int64_t>{rowCount});
        parallelFunc(std::thread::hardware_concurrency());

        indices.memory().markHostModified();
        if(values != nullptr)
        {
            values->memory().markHostModified();
        }
    }

private:
    template <class DataType, class IndexType>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<DataType>& x,
                              const hipdnn_data_sdk::utilities::TensorBase<IndexType>& indices,
                              const hipdnn_data_sdk::utilities::TensorBase<DataType>* values,
                              int64_t k)
    {
        const std::string prefix = "CpuFpReferenceTopk: ";

        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, prefix, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(indices, prefix, "indices");
        if(values != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*values, prefix, "values");
        }

        const auto& xDims = x.dims();
        if(xDims.empty())
        {
            throw std::runtime_error(prefix + "x must have rank >= 1");
        }
        if(k < 1 || k > xDims.back())
        {
            throw std::runtime_error(prefix + "k " + std::to_string(k)
                                     + " must satisfy 1 <= k <= x last dim ("
                                     + std::to_string(xDims.back()) + ")");
        }

        auto expected = xDims;
        expected.back() = k;
        if(indices.dims() != expected)
        {
            throw std::runtime_error(prefix + "indices dims must equal x dims with last dim k");
        }
        if(values != nullptr && values->dims() != expected)
        {
            throw std::runtime_error(prefix + "values dims must equal x dims with last dim k");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
