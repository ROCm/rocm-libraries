// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>

#include <cstdint>
#include <vector>

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

/// Copies an optional FlatBuffers vector of spatial parameters into a std::vector.
/// A missing vector yields an empty result, which the plan parameters default.
inline std::vector<int64_t> toSpatialVector(const flatbuffers::Vector<int64_t>* values)
{
    if(values == nullptr)
    {
        return {};
    }
    return {values->begin(), values->end()};
}

/// The GPU resample reference only supports an INT32 index tensor. UNSET marks the
/// plans that run without an index tensor.
template <hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
struct GpuResampleIndexNative
{
    using type = hipdnn_test_sdk::utilities::DataTypeToNative<IndexDataTypeEnum>;
};

template <>
struct GpuResampleIndexNative<hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
{
    using type = int32_t;
};

inline bool isSupportedResampleMode(hipdnn_flatbuffers_sdk::data_objects::ResampleMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::ResampleMode;
    return mode == ResampleMode::MAXPOOL || mode == ResampleMode::AVGPOOL_EXCLUDE_PADDING
           || mode == ResampleMode::AVGPOOL_INCLUDE_PADDING;
}

inline bool isSupportedPaddingMode(hipdnn_flatbuffers_sdk::data_objects::PaddingMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PaddingMode;
    return mode == PaddingMode::ZERO_PAD || mode == PaddingMode::NEG_INF_PAD;
}

/// Mirrors the applicability constraints the GPU resample reference enforces at launch time:
/// 4D or 5D tensors, spatial parameters of matching rank, positive stride/window and
/// non-negative padding. The forward reference additionally requires every padding to be
/// smaller than the corresponding window.
inline bool hasSupportedResampleGeometry(const flatbuffers::Vector<int64_t>* dims,
                                         const flatbuffers::Vector<int64_t>* prePadding,
                                         const flatbuffers::Vector<int64_t>* postPadding,
                                         const flatbuffers::Vector<int64_t>* stride,
                                         const flatbuffers::Vector<int64_t>* window,
                                         bool requirePaddingBelowWindow)
{
    if(dims == nullptr || dims->size() < 4 || dims->size() > 5)
    {
        return false;
    }

    const size_t spatialRank = dims->size() - 2;
    if(stride == nullptr || window == nullptr || static_cast<size_t>(stride->size()) != spatialRank
       || static_cast<size_t>(window->size()) != spatialRank)
    {
        return false;
    }

    const auto prePaddingValues = toSpatialVector(prePadding);
    const auto postPaddingValues = toSpatialVector(postPadding);
    if(!prePaddingValues.empty() && prePaddingValues.size() != spatialRank)
    {
        return false;
    }
    if(!postPaddingValues.empty() && postPaddingValues.size() != spatialRank)
    {
        return false;
    }

    for(size_t i = 0; i < spatialRank; ++i)
    {
        const auto windowSize = window->Get(static_cast<flatbuffers::uoffset_t>(i));
        const auto strideSize = stride->Get(static_cast<flatbuffers::uoffset_t>(i));
        if(windowSize <= 0 || strideSize <= 0)
        {
            return false;
        }

        const auto pre = prePaddingValues.empty() ? 0 : prePaddingValues[i];
        const auto post = postPaddingValues.empty() ? 0 : postPaddingValues[i];
        if(pre < 0 || post < 0)
        {
            return false;
        }

        if(requirePaddingBelowWindow && (pre >= windowSize || post >= windowSize))
        {
            return false;
        }
    }

    return true;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
