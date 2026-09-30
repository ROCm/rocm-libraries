// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include "core/Utils.hpp"

namespace hip_kernel_provider::rmsnorm
{

enum class Direction
{
    FORWARD,
    BACKWARD
};

// Infer the outer and inner normalization sizes.
// 1) Work out the normalization dimension, as the index of the first dimension in
//    scale tensor that is not 1.
// 2) Work out the outerSize as the size of input dimensions for which scale is 1.
//    We will have a work-group for each of these dimensions.
//    When stride is not 1, we are in a channel-last layout and we ignore the
//    channel dimension when calculating the outer size.
// 3) Work out the innerSize as the size of the input dimensions for which scale is not 1.
//
// For an input of [N, C, H, W] with scale [1, C, H, W] this will give: a normalization
// dimension of 1, outerSize of N, and innerSize of CxHxW.
// The kernel will therefore consist of N workgroups with each workgroup normalizing over
// CxHxW elements using a fixed number of threads.
// For an input of [N, H, W, C] with scale [1, H, W, 1] this will give: a normalization
// dimension of 2, outerSize of N, stride of C, and innerSize of HxW.
// The kernel will therefore consist of NxC workgroups with each workgroup normalizing
// over HxW elements using a fixed number of threads.
class ProblemDescription
{
    static unsigned getNormalizeDim(const flatbuffers::Vector<int64_t>* xDims,
                                    const flatbuffers::Vector<int64_t>* scaleDims)
    {
        const std::vector<int64_t> xDimsVec(xDims->begin(), xDims->end());
        const std::vector<int64_t> scaleDimsVec(scaleDims->begin(), scaleDims->end());

        // Find number of trailing dims where scaleDims[i] == inputDims[i]
        const auto [scaleMismatch, _] = std::mismatch(
            scaleDimsVec.rbegin(), scaleDimsVec.rend(), xDimsVec.rbegin(), xDimsVec.rend());
        const auto matchCount
            = static_cast<size_t>(std::distance(scaleDimsVec.rbegin(), scaleMismatch));

        // Scale must have at least one normalization axis, so account for the case where
        // input has a single batch and scale matches exactly.
        const auto normalizeDim
            = (matchCount == scaleDimsVec.size()) ? 1 : scaleDimsVec.size() - matchCount;
        return static_cast<unsigned>(normalizeDim);
    }

    static int64_t getStride(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x,
                             unsigned normalizeDim)
    {
        int64_t stride = 1;
        const auto isLayoutNHWC = core::utils::isChannelLastLayout(x);
        if(normalizeDim > 1 && isLayoutNHWC)
        {
            stride = static_cast<int64_t>(x->dims()->Get(1));
        }
        return stride;
    }

    static int64_t getOuterSize(const flatbuffers::Vector<int64_t>* xDims,
                                unsigned normalizeDim,
                                int64_t stride)
    {
        int64_t outerSize = 1;
        for(unsigned i = 0; i < normalizeDim; ++i)
        {
            // Add channel size only if there is no stride
            if(i == 1 && stride != 1)
            {
                continue;
            }
            outerSize *= static_cast<int64_t>(xDims->Get(i));
        }
        return outerSize;
    }

    static int64_t getInnerSize(const flatbuffers::Vector<int64_t>* xDims, unsigned normalizeDim)
    {
        int64_t innerSize = 1;
        for(unsigned i = normalizeDim; i < xDims->size(); ++i)
        {
            innerSize *= xDims->Get(i);
        }
        return innerSize;
    }

public:
    ProblemDescription(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x,
                       const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale,
                       Direction direction)
        : _direction(direction)
        , _normalizeDim(getNormalizeDim(x->dims(), scale->dims()))
        , _stride(getStride(x, _normalizeDim))
        , _outerSize(getOuterSize(x->dims(), _normalizeDim, _stride))
        , _innerSize(getInnerSize(x->dims(), _normalizeDim))
    {
    }

    Direction direction() const
    {
        return _direction;
    }
    unsigned normalizeDim() const
    {
        return _normalizeDim;
    }
    int64_t outerSize() const
    {
        return _outerSize;
    }
    int64_t innerSize() const
    {
        return _innerSize;
    }
    int64_t stride() const
    {
        return _stride;
    }

private:
    Direction _direction;
    unsigned _normalizeDim;
    int64_t _stride;
    int64_t _outerSize;
    int64_t _innerSize;
};

}
