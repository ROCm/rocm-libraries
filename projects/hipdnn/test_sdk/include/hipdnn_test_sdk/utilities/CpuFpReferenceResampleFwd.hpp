// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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

class CpuFpReferenceResampleFwd
{
public:
    template <class XDataType,
              class YDataType = XDataType,
              class ComputeDataType = float,
              class IndexDataType = int32_t>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                        hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& stride,
                        const std::vector<int64_t>& window,
                        hipdnn_flatbuffers_sdk::data_objects::ResampleMode mode,
                        hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode,
                        hipdnn_data_sdk::utilities::TensorBase<IndexDataType>* index = nullptr)
    {
        const auto& xDims = x.dims();
        const auto& yDims = y.dims();
        if(xDims.size() < 4 || xDims.size() > 5 || xDims.size() != yDims.size())
        {
            throw std::runtime_error("ResampleFwd reference supports matching 4D or 5D tensors.");
        }

        const auto spatialDims = xDims.size() - 2;
        if(prePadding.size() != spatialDims || stride.size() != spatialDims
           || window.size() != spatialDims)
        {
            throw std::runtime_error(
                "ResampleFwd reference spatial parameter ranks must match tensor rank.");
        }

        auto resampleFwdFunc = [&](const std::vector<int64_t>& yIndices) {
            auto result = mode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL
                              ? static_cast<ComputeDataType>(std::numeric_limits<float>::lowest())
                              : static_cast<ComputeDataType>(0);
            int64_t validCount = 0;
            auto selectedIndex = static_cast<IndexDataType>(-1);
            bool selectedCandidate = false;

            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                window, [&](const std::vector<int64_t>& windowIndices) {
                    std::vector<int64_t> xIndices(xDims.size(), 0);
                    xIndices[0] = yIndices[0];
                    xIndices[1] = yIndices[1];

                    bool valid = true;
                    for(size_t i = 0; i < spatialDims; ++i)
                    {
                        const auto xSpatialIndex
                            = yIndices[i + 2] * stride[i] + windowIndices[i] - prePadding[i];
                        xIndices[i + 2] = xSpatialIndex;
                        valid = valid && xSpatialIndex >= 0 && xSpatialIndex < xDims[i + 2];
                    }

                    auto candidate = static_cast<ComputeDataType>(0);
                    auto candidateIndex = static_cast<IndexDataType>(-1);
                    if(valid)
                    {
                        candidate = static_cast<ComputeDataType>(x.getHostValue(xIndices));
                        candidateIndex = flattenSpatialIndex<IndexDataType>(xDims, xIndices);
                        ++validCount;
                    }
                    else if(mode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL
                            && paddingMode
                                   == hipdnn_flatbuffers_sdk::data_objects::PaddingMode::
                                       NEG_INF_PAD)
                    {
                        return;
                    }

                    if(mode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL)
                    {
                        if(!selectedCandidate || candidate > result)
                        {
                            result = candidate;
                            selectedIndex = candidateIndex;
                            selectedCandidate = true;
                        }
                    }
                    else
                    {
                        result += candidate;
                    }
                });

            if(mode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING)
            {
                const auto divisor = validCount == 0 ? 1 : validCount;
                result /= static_cast<ComputeDataType>(divisor);
            }
            else if(mode
                    == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING)
            {
                int64_t windowElementCount = 1;
                for(const auto windowDim : window)
                {
                    windowElementCount *= windowDim;
                }
                result /= static_cast<ComputeDataType>(windowElementCount);
            }

            y.setHostValue(static_cast<YDataType>(result), yIndices);
            if(index != nullptr)
            {
                index->setHostValue(selectedIndex, yIndices);
            }
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(resampleFwdFunc, yDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
        if(index != nullptr)
        {
            index->memory().markHostModified();
        }
    }

private:
    template <class IndexDataType>
    static IndexDataType flattenSpatialIndex(const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& indices)
    {
        int64_t flattened = 0;
        for(size_t i = 2; i < dims.size(); ++i)
        {
            flattened = flattened * dims[i] + indices[i];
        }
        return static_cast<IndexDataType>(flattened);
    }
};

} // namespace hipdnn_test_sdk::utilities
