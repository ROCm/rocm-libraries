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
    static void
        backward(const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& dy,
                 hipdnn_data_sdk::utilities::TensorBase<DxDataType>& dx,
                 const std::vector<int64_t>& prePadding,
                 const std::vector<int64_t>& stride,
                 const std::vector<int64_t>& window,
                 hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode,
                 [[maybe_unused]] hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode,
                 const hipdnn_data_sdk::utilities::TensorBase<IndexDataType>* index = nullptr)
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
        auto resampleBwdFunc = [&](const std::vector<int64_t>& dxIndices) {
            auto result = static_cast<ComputeDataType>(0);
            const auto currentFlattenedIndex = flattenDxSpatialIndex(dxDims, dxIndices);

            // Evalute if the current dxIndices has valid contributions from dy in the window
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                window, [&](const std::vector<int64_t>& windowIndices) {
                    const auto dyIndices
                        = makeDyIndices(dyDims, dxIndices, windowIndices, prePadding, stride);
                    if(!dyIndices.has_value())
                    {
                        return;
                    }

                    const auto dyVal = static_cast<ComputeDataType>(dy.getHostValue(*dyIndices));

                    if(resampleMode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL)
                    {
                        const auto flattenedIndex
                            = static_cast<int64_t>(index->getHostValue(*dyIndices));
                        if(flattenedIndex
                           != currentFlattenedIndex) // No contribution to dx from this dy index
                        {
                            return;
                        }

                        // Only this dy element contributes to the current dx element,
                        // so we can add the dy to the result and return early
                        result += dyVal;
                        return;
                    }

                    // For average pooling, determine how many valid contributions (for
                    // AVGPOOL_EXCLUDE_PADDING) or the window size (for AVGPOOL_INCLUDE_PADDING)
                    // to split the dy gradient across
                    int64_t divisor = 1;
                    if(resampleMode
                       == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::
                           AVGPOOL_EXCLUDE_PADDING)
                    {
                        int64_t validCount = 0;
                        hipdnn_data_sdk::utilities::iterateAlongDimensions(
                            window, [&](const std::vector<int64_t>& innerWindowIndices) {
                                if(makeDxIndices(
                                       dxDims, *dyIndices, innerWindowIndices, prePadding, stride)
                                       .has_value())
                                {
                                    ++validCount;
                                }
                            });
                        divisor = validCount == 0 ? 1 : validCount;
                    }
                    else
                    {
                        divisor = 1;
                        for(const auto windowDim : window)
                        {
                            divisor *= windowDim;
                        }
                    }

                    // Add the contribution from this dy element to the current dx element
                    result += dyVal / static_cast<ComputeDataType>(divisor);
                });

            dx.setHostValue(static_cast<DxDataType>(result), dxIndices);
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(resampleBwdFunc, dxDims);
        parallelFunc(std::thread::hardware_concurrency());

        dx.memory().markHostModified();
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

    static std::optional<std::vector<int64_t>>
        makeDyIndices(const std::vector<int64_t>& dyDims,
                      const std::vector<int64_t>& dxIndices,
                      const std::vector<int64_t>& windowIndices,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& stride)
    {
        std::vector<int64_t> dyIndices(dyDims.size(), 0);
        dyIndices[0] = dxIndices[0];
        dyIndices[1] = dxIndices[1];
        for(size_t i = 0; i < windowIndices.size(); ++i)
        {
            const auto numerator = dxIndices[i + 2] + prePadding[i] - windowIndices[i];
            if(numerator % stride[i] != 0)
            {
                return std::nullopt;
            }
            const auto outIndex = numerator / stride[i];
            if(outIndex < 0 || outIndex >= dyDims[i + 2])
            {
                return std::nullopt;
            }
            dyIndices[i + 2] = outIndex;
        }
        return dyIndices;
    }

    static int64_t flattenDxSpatialIndex(const std::vector<int64_t>& dxDims,
                                         const std::vector<int64_t>& dxIndices)
    {
        int64_t flattened = 0;
        for(size_t i = 2; i < dxDims.size(); ++i)
        {
            flattened = flattened * dxDims[i] + dxIndices[i];
        }
        return flattened;
    }
};

} // namespace hipdnn_test_sdk::utilities
