// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

namespace hipdnn_sdk
{
namespace utilities
{

// Check broadcastability: input can be broadcast to output
// Broadcasting rules:
// 1. Dimensions are compared from right to left
// 2. Two dimensions are compatible if they are equal or input is 1 (output needs to be max size)
// 3. The input can have fewer dimensions than output (implicit leading 1s)
inline bool areDimensionsBroadcastCompatible(const std::vector<int64_t>& inputDims,
                                             const std::vector<int64_t>& outputDims)
{
    if(inputDims == outputDims)
    {
        return true;
    }

    if(inputDims.size() > outputDims.size())
    {
        return false;
    }

    auto inputIt = inputDims.rbegin();
    auto outputIt = outputDims.rbegin();

    while(inputIt != inputDims.rend() && outputIt != outputDims.rend())
    {
        if(*inputIt != *outputIt && *inputIt != 1)
        {
            return false;
        }
        ++inputIt;
        ++outputIt;
    }

    return true;
}

// Sets a default stride ordering based off the provided stride order.
// Ex. dim = {1,2,3,4} stride_order = {3, 0, 2, 1} for NHWC
// returns {24, 1, 8, 2}
inline std::vector<int64_t> generateStrides(const std::vector<int64_t>& dim,
                                            const std::vector<int64_t>& strideOrder)
{
    size_t numDims = dim.size();

    if(numDims > strideOrder.size())
    {
        throw std::invalid_argument("dims must be less than or equal stride size");
    }

    std::vector<int64_t> stride(numDims, 1);

    // Create a mapping of stride order to dimension index
    std::vector<size_t> indices(numDims);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&strideOrder](size_t a, size_t b) {
        return strideOrder[a] < strideOrder[b];
    });

    int64_t accumulator = 1;
    for(auto idx : indices)
    {
        stride[idx] = accumulator;
        accumulator *= dim[idx];
    }

    return stride;
}

// Generates packed strides for the provided dims.
// NCHW stride order for 4d, NCDHW for 5d, etc.
inline std::vector<int64_t> generateStrides(const std::vector<int64_t>& dims)
{
    if(dims.empty())
    {
        return {};
    }

    std::vector<int64_t> strides(dims.size());
    strides.back() = 1;
    for(size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// Sets stride order as NHWC for the provided dims.
// Ex. 4 will return {3, 0, 2, 1} for NHWC
inline std::vector<int64_t> strideOrderNhwc(size_t numDims)
{
    // Default all to 0, and set everything up until NC
    std::vector<int64_t> strideOrder(numDims, 0);

    if(numDims < 2)
    {
        return strideOrder;
    }

    int64_t order = 1;
    for(size_t i = numDims - 1; i > 1; --i)
    {
        strideOrder[i] = order++;
    }
    strideOrder[0] = order;

    return strideOrder;
}

// Extracts stride order from existing strides
// The stride order indicates the memory layout priority (lower values = higher priority)
// For example, strides [8, 1, 4] would produce order [2, 0, 1]
// This is the inverse operation of generateStrides
inline std::vector<int64_t> extractStrideOrder(const std::vector<int64_t>& strides)
{
    std::vector<int64_t> strideOrder(strides.size());
    std::vector<size_t> indices(strides.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by their corresponding stride values (ascending)
    std::sort(indices.begin(), indices.end(), [&strides](size_t a, size_t b) {
        return strides[a] < strides[b];
    });

    // Assign order based on sorted indices
    for(size_t i = 0; i < indices.size(); ++i)
    {
        strideOrder[indices[i]] = static_cast<int64_t>(i);
    }

    return strideOrder;
}

// Gets the derived (per channel) shape from a full Tensor shape.
// Ex. {1, 3, 224, 224} will return {1, 3, 1, 1}
inline std::vector<int64_t> getDerivedShape(const std::vector<int64_t>& shape)
{
    if(shape.size() < 2)
    {
        throw std::runtime_error(
            "A shape must consist of at least 2 dimensions (batch and channel)");
    }

    auto result = std::vector<int64_t>(shape.size(), 1);
    result[1] = shape[1];

    return result;
}

// Iterates the elements along each of the dimensions specified in dims and calls func for each unique index
// Formally, we are iterating over a cartesian product of the ranges [0, dims[0]), [0, dims[1]), ..., [0, dims[n - 1]) for n dimensions
template <typename F>
static void iterateAlongDimensions(const std::vector<int64_t>& dims, F&& func)
{
    if(dims.empty())
    {
        func({});
        return;
    }

    int64_t totalElements = 1;
    for(auto dim : dims)
    {
        totalElements *= dim;
    }

    std::vector<int64_t> indices(dims.size(), 0);

    // Iterate over each unique position
    for(int64_t iter = 0; iter < totalElements; ++iter)
    {
        if constexpr(std::is_invocable_r_v<bool, F, const std::vector<int64_t>&>)
        {
            if(!func(indices))
            {
                return; // Early exit if lambda returns false
            }
        }
        else
        {
            func(indices); // Original behavior for void-returning lambdas
        }

        for(int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim)
        {
            auto dimIdx = static_cast<size_t>(dim);
            indices[dimIdx]++;

            if(indices[dimIdx] < dims[dimIdx])
            {
                break;
            }

            indices[dimIdx] = 0;
        }
    }
}

// Constructs a full tensor indices vector from batch, channel, and spatial components. spatialOffset allows
// skipping initial elements in the spatialIndices vector for convenience.
static inline std::vector<int64_t> buildTensorIndices(int64_t batchIdx,
                                                      int64_t channelIdx,
                                                      const std::vector<int64_t>& spatialIndices,
                                                      size_t spatialOffset = 0)
{
    std::vector<int64_t> fullIndices = {batchIdx, channelIdx};
    fullIndices.insert(fullIndices.end(),
                       spatialIndices.begin() + static_cast<std::ptrdiff_t>(spatialOffset),
                       spatialIndices.end());
    return fullIndices;
}

// Utility for calculating group count given weight and input tensors
// For grouped convolutions, group count = input_channels / weight_channels_per_group
inline int64_t calculateGroupCount(const std::vector<int64_t>& inputDims,
                                   const std::vector<int64_t>& weightDims)
{
    if(inputDims.size() < 2)
    {
        throw std::invalid_argument("Input tensor must have at least 2 dimensions, but got: "
                                    + std::to_string(inputDims.size()));
    }

    if(weightDims.size() < 2)
    {
        throw std::invalid_argument("Weight tensor must have at least 2 dimensions, but got: "
                                    + std::to_string(weightDims.size()));
    }

    auto inChannels = inputDims[1];
    auto wChannels = weightDims[1];

    if(inChannels <= 0)
    {
        throw std::invalid_argument("Input channels must be positive, but got: "
                                    + std::to_string(inChannels));
    }

    if(wChannels <= 0)
    {
        throw std::invalid_argument("Weight channels must be positive, but got: "
                                    + std::to_string(wChannels));
    }

    if(inChannels % wChannels != 0)
    {
        throw std::invalid_argument("Input channels (" + std::to_string(inChannels)
                                    + ") must be evenly divisible by weight channels ("
                                    + std::to_string(wChannels) + ")");
    }

    return inChannels / wChannels;
}

}
}
