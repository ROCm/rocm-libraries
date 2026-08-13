// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_common_generated.h>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceResampleBwd
{
public:
    template <class DyDataType,
              class DxDataType = DyDataType,
              class ComputeDataType = float,
              class IndexDataType = int32_t>
    static void backward(const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& dy,
                         hipdnn_data_sdk::utilities::TensorBase<DxDataType>& dx,
                         const std::vector<int64_t>& prePadding,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& window,
                         hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode,
                         hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode,
                         const hipdnn_data_sdk::utilities::TensorBase<IndexDataType>* index
                         = nullptr)
    {
        const auto& dyDims = dy.dims();
        const auto& dxDims = dx.dims();

        // Validate IO tensor dimensions
        if(dxDims.size() < 4 || dxDims.size() > 5 || dxDims.size() != dyDims.size())
        {
            throw std::runtime_error("ResampleBwd reference supports matching 4D or 5D tensors.");
        }

        if(dxDims[0] != dyDims[0] || dxDims[1] != dyDims[1])
        {
            throw std::runtime_error("ResampleBwd reference requires matching N and C dimensions.");
        }

        // Validate spatial parameters
        const auto spatialDims = dyDims.size() - 2;
        if(prePadding.size() != spatialDims || stride.size() != spatialDims
           || window.size() != spatialDims)
        {
            throw std::runtime_error(
                "ResampleBwd reference spatial parameter ranks must match tensor rank.");
        }

        // Validate resampleMode and index tensor
        if(resampleMode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL)
        {
            if(index == nullptr || index->dims() != dyDims)
            {
                throw std::runtime_error(
                    "ResampleBwd max pooling requires an index tensor matching dy.");
            }
        }
        else if(resampleMode
                    != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING
                && resampleMode
                       != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::
                           AVGPOOL_INCLUDE_PADDING)
        {
            throw std::runtime_error("ResampleBwd reference received an unsupported resampleMode.");
        }

        // Initialize dx to zero
        hipdnn_data_sdk::utilities::iterateAlongDimensions(
            dxDims, [&](const std::vector<int64_t>& dxIndices) {
                dx.setHostValue(static_cast<DxDataType>(0), dxIndices);
            });

        // Compute the backward pass
        auto resampleBwdFunc = [&](const std::vector<int64_t>& dyIndices) {
            // Gradient from next layer at current position
            const auto gradient = static_cast<ComputeDataType>(dy.getHostValue(dyIndices));

            if(resampleMode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL)
            {
                const auto flattenedIndex = static_cast<int64_t>(index->getHostValue(dyIndices));
                if(flattenedIndex < 0) // No contribution to dx from this index
                {
                    return;
                }

                // dL/dx[flattenedIndex -> dxIndices] += 1.0 * dL/dy[dyIndices]
                auto dxIndices = unflattenSpatialIndex(dxDims, dyIndices, flattenedIndex);
                accumulate(dx, dxIndices, gradient);
                return;
            }

            // For average pooling, determine how many valid contributions (for
            // AVGPOOL_EXCLUDE_PADDING) or the window size (for AVGPOOL_INCLUDE_PADDING)
            // to split dy across the contributing dx elements
            int64_t validCount = 0;
            if(resampleMode
               == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING)
            {
                hipdnn_data_sdk::utilities::iterateAlongDimensions(
                    window, [&](const std::vector<int64_t>& windowIndices) {
                        if(makeDxIndices(dxDims, dyIndices, windowIndices, prePadding, stride)
                               .has_value())
                        {
                            ++validCount;
                        }
                    });
            }
            int64_t divisor = validCount;
            if(resampleMode
               == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING)
            {
                divisor = 1;
                for(const auto windowDim : window)
                {
                    divisor *= windowDim;
                }
            }

            if(divisor == 0) // Avoid division by zero, if no valid contributions
            {
                divisor = 1;
            }

            // dL/dx[dxIndices] += (1.0 / numContributions) * dL/dy[dyIndices]
            const auto contribution = gradient / static_cast<ComputeDataType>(divisor);
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                window, [&](const std::vector<int64_t>& windowIndices) {
                    auto dxIndices
                        = makeDxIndices(dxDims, dyIndices, windowIndices, prePadding, stride);
                    if(dxIndices.has_value())
                    {
                        accumulate(dx, *dxIndices, contribution);
                    }
                });
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(resampleBwdFunc, dyDims);
        parallelFunc(std::thread::hardware_concurrency());

        dx.memory().markHostModified();
        static_cast<void>(paddingMode);
    }

private:
    static std::optional<std::vector<int64_t>>
        makeDxIndices(const std::vector<int64_t>& dxDims,
                      const std::vector<int64_t>& dyIndices,
                      const std::vector<int64_t>& windowIndices,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& stride)
    {
        std::vector<int64_t> dxIndices(dxDims.size(), 0);
        dxIndices[0] = dyIndices[0];
        dxIndices[1] = dyIndices[1];
        for(size_t i = 0; i < windowIndices.size(); ++i)
        {
            const auto spatialIndex
                = dyIndices[i + 2] * stride[i] + windowIndices[i] - prePadding[i];
            if(spatialIndex < 0 || spatialIndex >= dxDims[i + 2])
            {
                return std::nullopt;
            }
            dxIndices[i + 2] = spatialIndex;
        }
        return dxIndices;
    }

    static std::vector<int64_t> unflattenSpatialIndex(const std::vector<int64_t>& dxDims,
                                                      const std::vector<int64_t>& dyIndices,
                                                      int64_t flattenedIndex)
    {
        std::vector<int64_t> dxIndices(dxDims.size(), 0);
        dxIndices[0] = dyIndices[0];
        dxIndices[1] = dyIndices[1];
        for(size_t i = dxDims.size() - 1; i >= 2; --i)
        {
            dxIndices[i] = flattenedIndex % dxDims[i];
            flattenedIndex /= dxDims[i];
        }
        if(flattenedIndex != 0)
        {
            throw std::runtime_error("ResampleBwd index is outside the dx spatial dimensions.");
        }
        return dxIndices;
    }

    template <class DxDataType, class ComputeDataType>
    static void accumulate(hipdnn_data_sdk::utilities::TensorBase<DxDataType>& dx,
                           const std::vector<int64_t>& dxIndices,
                           ComputeDataType contribution)
    {
        const auto current = static_cast<ComputeDataType>(dx.getHostValue(dxIndices));
        dx.setHostValue(static_cast<DxDataType>(current + contribution), dxIndices);
    }
};

} // namespace hipdnn_test_sdk::utilities
