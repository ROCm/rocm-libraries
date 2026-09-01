// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormUtilities.hpp"

#include <cstdint>
#include <flatbuffers/base.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "core/Utils.hpp"

namespace hip_kernel_provider::layernorm
{

namespace
{

std::vector<int64_t>
    getAffineDims(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr)
{
    std::vector<int64_t> affineDims(std::max(ioAttr->dims()->size(), affineAttr->dims()->size()),
                                    1);
    const size_t startIndex = ioAttr->dims()->size() > affineAttr->dims()->size()
                                  ? ioAttr->dims()->size() - affineAttr->dims()->size()
                                  : 0;
    for(size_t i = 0; i < affineAttr->dims()->size(); ++i)
    {
        affineDims[i + startIndex]
            = affineAttr->dims()->Get(static_cast<flatbuffers::uoffset_t>(i));
    }
    return affineDims;
}

std::vector<int64_t>
    getStatDims(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
                const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statAttr)
{
    std::vector<int64_t> statDims(std::max(ioAttr->dims()->size(), statAttr->dims()->size()), 1);
    for(size_t i = 0; i < statAttr->dims()->size(); ++i)
    {
        statDims[i] = statAttr->dims()->Get(static_cast<flatbuffers::uoffset_t>(i));
    }
    return statDims;
}

} // namespace

size_t getMinNormalizedDimFromAffine(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr)
{
    const std::vector<int64_t> ioDims(ioAttr->dims()->begin(), ioAttr->dims()->end());
    const auto affineDims = getAffineDims(ioAttr, affineAttr);

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
    const auto* ioAttr = &core::utils::findTensorAttributes(tensorMap, ioTensorId);
    const auto* affineAttr = &core::utils::findTensorAttributes(tensorMap, affineTensorId);

    return getMinNormalizedDimFromAffine(ioAttr, affineAttr);
}

size_t getMaxNormalizedDimFromAffine(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr)
{
    const std::vector<int64_t> ioDims(ioAttr->dims()->begin(), ioAttr->dims()->end());
    const auto affineDims = getAffineDims(ioAttr, affineAttr);

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
    const auto* ioAttr = &core::utils::findTensorAttributes(tensorMap, ioTensorId);
    const auto* affineAttr = &core::utils::findTensorAttributes(tensorMap, affineTensorId);

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
    const auto statDims = getStatDims(ioAttr, statAttr);

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
    const auto* ioAttr = &core::utils::findTensorAttributes(tensorMap, ioTensorId);
    const auto* statAttr = &core::utils::findTensorAttributes(tensorMap, statTensorId);

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
    const auto statDims = getStatDims(ioAttr, statAttr);

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
    const auto* ioAttr = &core::utils::findTensorAttributes(tensorMap, ioTensorId);
    const auto* statAttr = &core::utils::findTensorAttributes(tensorMap, statTensorId);

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

} // namespace hip_kernel_provider::layernorm
