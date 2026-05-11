// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include "HipKernelUtils.hpp"

namespace hip_kernel_provider::rmsnorm
{

inline int64_t getOuterSize(const flatbuffers::Vector<int64_t>* xDims, unsigned normalizeDim)
{
    int64_t outerSize = 1;
    for(unsigned i = 0; i < normalizeDim; ++i)
    {
        outerSize *= xDims->Get(i);
    }
    return outerSize;
}

inline int64_t getInnerSize(const flatbuffers::Vector<int64_t>* xDims, unsigned normalizeDim)
{
    int64_t innerSize = 1;
    for(unsigned i = normalizeDim; i < xDims->size(); ++i)
    {
        innerSize *= xDims->Get(i);
    }
    return innerSize;
}

inline int64_t getStride(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x,
                         unsigned normalizeDim)
{
    int64_t stride = 1;
    auto isLayoutNHWC = hip_kernel_utils::isChannelLastLayout(x);
    if(normalizeDim > 1 && isLayoutNHWC)
    {
        stride = static_cast<int64_t>(x->dims()->Get(1));
    }
    return stride;
}

inline unsigned getNormalizeDim(const flatbuffers::Vector<int64_t>* xDims,
                                const flatbuffers::Vector<int64_t>* scaleDims)
{
    const std::vector<int64_t> xDimsVec(xDims->begin(), xDims->end());
    const std::vector<int64_t> scaleDimsVec(scaleDims->begin(), scaleDims->end());

    // Find number of trailing dims where scaleDims[i] == inputDims[i]
    const auto [scaleMismatch, _] = std::mismatch(
        scaleDimsVec.rbegin(), scaleDimsVec.rend(), xDimsVec.rbegin(), xDimsVec.rend());
    const auto matchCount
        = static_cast<size_t>(std::distance(scaleDimsVec.rbegin(), scaleMismatch));

    // Scale must have at least one normalization axis, so account for the case where input has a single batch and scale
    // matches exactly.
    const auto normalizeDim
        = (matchCount == scaleDimsVec.size()) ? 1 : scaleDimsVec.size() - matchCount;
    return static_cast<unsigned>(normalizeDim);
}

}
