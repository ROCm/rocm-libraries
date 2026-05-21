// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormUtilities.hpp"

#include <cstdint>

#include "HipKernelUtils.hpp"

namespace hip_kernel_provider::layernorm
{

size_t getMinNormalizedDimFromAffine(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr)
{
    const std::vector<int64_t> ioDims(ioAttr->dims()->begin(), ioAttr->dims()->end());
    const std::vector<int64_t> affineDims(affineAttr->dims()->begin(), affineAttr->dims()->end());

    size_t affineNormalizedDimMin = 0;
    for(size_t i = 0; i < affineDims.size(); ++i)
    {
        if(affineDims[i] == 1 && ioDims[i] > 1)
        {
            affineNormalizedDimMin = i + 1;
        }
        else if(affineDims[i] > 1)
        {
            break;
        }
    }

    return affineNormalizedDimMin;
}

size_t getMinNormalizedDimFromAffine(
    const int64_t ioTensorId,
    const int64_t affineTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto* ioAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, ioTensorId);
    const auto* affineAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, affineTensorId);

    return getMinNormalizedDimFromAffine(ioAttr, affineAttr);
}

size_t getMaxNormalizedDimFromAffine(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr)
{
    const std::vector<int64_t> ioDims(ioAttr->dims()->begin(), ioAttr->dims()->end());
    const std::vector<int64_t> affineDims(affineAttr->dims()->begin(), affineAttr->dims()->end());

    auto affineNormalizedDimMax = affineDims.size();
    for(auto i = static_cast<int>(affineDims.size()) - 1; i >= 0; --i)
    {
        if(affineDims[static_cast<size_t>(i)] > 1 && ioDims[static_cast<size_t>(i)] > 1)
        {
            affineNormalizedDimMax = static_cast<size_t>(i);
        }
        else if(affineDims[static_cast<size_t>(i)] == 1 && ioDims[static_cast<size_t>(i)] > 1)
        {
            break;
        }
    }

    return affineNormalizedDimMax;
}

size_t getMaxNormalizedDimFromAffine(
    const int64_t ioTensorId,
    const int64_t affineTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto* ioAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, ioTensorId);
    const auto* affineAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, affineTensorId);

    return getMaxNormalizedDimFromAffine(ioAttr, affineAttr);
}

size_t getMinNormalizedDimFromStat(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statAttr)
{
    if(ioAttr == nullptr || statAttr == nullptr)
    {
        return 0;
    }

    const std::vector<int64_t> ioDims(ioAttr->dims()->begin(), ioAttr->dims()->end());
    const std::vector<int64_t> statDims(statAttr->dims()->begin(), statAttr->dims()->end());

    size_t statNormalizedDimMin = 0;
    for(size_t i = 0; i < statDims.size(); ++i)
    {
        if(statDims[i] > 1 && ioDims[i] > 1)
        {
            statNormalizedDimMin = i + 1;
        }
        else if(statDims[i] == 1 && ioDims[i] > 1)
        {
            break;
        }
    }

    return statNormalizedDimMin;
}

size_t getMinNormalizedDimFromStat(
    const int64_t ioTensorId,
    const int64_t statTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto* ioAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, ioTensorId);
    const auto* statAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, statTensorId);

    return getMinNormalizedDimFromStat(ioAttr, statAttr);
}

size_t getMaxNormalizedDimFromStat(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statAttr)
{
    const std::vector<int64_t> ioDims(ioAttr->dims()->begin(), ioAttr->dims()->end());
    if(statAttr == nullptr)
    {
        return ioDims.size();
    }
    const std::vector<int64_t> statDims(statAttr->dims()->begin(), statAttr->dims()->end());

    auto statNormalizedDimMax = statDims.size();
    for(auto i = static_cast<int>(statDims.size()) - 1; i >= 0; --i)
    {
        if(statDims[static_cast<size_t>(i)] == 1 && ioDims[static_cast<size_t>(i)] > 1)
        {
            statNormalizedDimMax = static_cast<size_t>(i);
        }
        else if(statDims[static_cast<size_t>(i)] > 1)
        {
            break;
        }
    }

    return statNormalizedDimMax;
}

size_t getMaxNormalizedDimFromStat(
    const int64_t ioTensorId,
    const int64_t statTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto* ioAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, ioTensorId);
    const auto* statAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, statTensorId);

    return getMaxNormalizedDimFromStat(ioAttr, statAttr);
}

size_t guessNormalizedDim(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> affineAttr,
    const std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> statAttr)
{
    const size_t affineNormalizedDimMin
        = affineAttr.has_value() ? getMinNormalizedDimFromAffine(ioAttr, affineAttr.value()) : 0;
    const size_t statNormalizedDimMin
        = statAttr.has_value() ? getMinNormalizedDimFromStat(ioAttr, statAttr.value()) : 0;

    const size_t affineNormalizedDimMax
        = affineAttr.has_value() ? getMaxNormalizedDimFromAffine(ioAttr, affineAttr.value()) : 0;
    const size_t statNormalizedDimMax
        = statAttr.has_value() ? getMaxNormalizedDimFromStat(ioAttr, statAttr.value()) : 0;

    const size_t normalizedDimMin = std::max(affineNormalizedDimMin, statNormalizedDimMin);
    const size_t normalizedDimMax = std::min(affineNormalizedDimMax, statNormalizedDimMax);

    return normalizedDimMin > 0 ? normalizedDimMin : normalizedDimMax;
}

size_t guessNormalizedDim(
    const int64_t ioTensorId,
    const std::optional<int64_t> affineTensorId,
    const std::optional<int64_t> statTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto* ioAttr = &hip_kernel_utils::findTensorAttributes(tensorMap, ioTensorId);
    const std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> affineAttr
        = affineTensorId.has_value() ? std::optional(&hip_kernel_utils::findTensorAttributes(
                                           tensorMap, affineTensorId.value()))
                                     : std::nullopt;
    const std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> statAttr
        = statTensorId.has_value() ? std::optional(&hip_kernel_utils::findTensorAttributes(
                                         tensorMap, statTensorId.value()))
                                   : std::nullopt;

    return guessNormalizedDim(ioAttr, affineAttr, statAttr);
}

} // namespace hip_kernel_provider::layernorm
