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

}
}
