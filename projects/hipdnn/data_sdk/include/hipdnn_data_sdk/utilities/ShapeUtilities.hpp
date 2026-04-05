// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <algorithm>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace hipdnn_data_sdk::utilities
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

/**
 * @brief Computes strides from dimensions and a stride ordering
 *
 * Generates strides for a tensor based on the provided dimensions and stride order.
 * The stride order specifies the memory layout priority (lower values = tighter packing).
 *
 * @param dim Dimension sizes. The expected ordering depends on the operation type:
 *            convolution/batchnorm use (N, C, H, W) or (N, C, D, H, W),
 *            matmul uses (...batch, M, K) or (...batch, K, N).
 * @param strideOrder Priority of each dimension in memory (lower = tighter packing).
 *                    Use TensorLayout constants for common layouts.
 * @return Strides corresponding to the requested memory layout
 *
 * @code{.cpp}
 * // Example for convolution input: {N=1, C=2, H=3, W=4}
 * auto nhwcStrides = generateStrides({1,2,3,4}, {3, 0, 2, 1}); // {24, 1, 8, 2}
 * @endcode
 */
inline std::vector<int64_t> generateStrides(const std::vector<int64_t>& dim,
                                            const std::vector<int64_t>& strideOrder)
{
    const size_t numDims = dim.size();

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

// Deduces the stride order from existing strides (assumes common/standard layouts).
// The returned stride order indicates the memory layout priority (lower values = higher priority).
// For example, strides [8, 1, 4] would produce order [2, 0, 1].
// Attempts to determine between NC...W and N...WC layouts, defaulting to NC...W when uncertain.
// For example, strides [1, 1, 1, 1] would produce order [3, 2, 1, 0].
// A warning log will be added when input strides are not unique.
// This is the inverse operation of generateStrides.
inline std::vector<int64_t> extractStrideOrder(const std::vector<int64_t>& strides)
{
    const size_t numDims = strides.size();
    std::vector<size_t> indices(numDims);
    std::vector<int64_t> strideOrder(numDims);
    std::iota(indices.begin(), indices.end(), 0);
    bool stridesAreUnique = true;

    if(strides.empty())
    {
        return {};
    }

    // Attempt to determine between N...C layouts and N...W layouts.
    auto posFirstMax = static_cast<size_t>(
        std::distance(strides.begin(), std::max_element(strides.begin(), strides.end())));
    auto posFirstMin = static_cast<size_t>(
        std::distance(strides.begin(), std::min_element(strides.begin(), strides.end())));

    if(posFirstMax == 0 && posFirstMin == 1)
    {
        // N is amongst the largest strides and C is amongst the shortest.
        for(size_t i = 2; i < numDims; ++i)
        {
            if(strides[i] > strides[posFirstMin])
            {
                // C is smaller than at least one of D, H, or W. Assume N...WC memory
                // layout and force C to the end of the list before sort with stable tiebreakers so
                // that it's handled properly in case of duplicate minimum stride lengths.
                indices.erase(indices.begin() + 1);
                indices.push_back(1);
                break;
            }
        }
    }

    // Sort indices by their corresponding stride values (descending; aligns with NC...W layout)
    // Use std::sort with a stable tie-breaker instead of std::stable_sort, which uses
    // deprecated std::get_temporary_buffer in libstdc++ causing clang-tidy errors.
    struct SortItem
    {
        size_t index; // The dimension index value
        size_t originalPos; // Position before sorting (for stability tie-breaking)
    };

    std::vector<SortItem> sortItems(numDims);
    for(size_t i = 0; i < numDims; ++i)
    {
        sortItems[i] = {indices[i], i};
    }

    std::sort(sortItems.begin(),
              sortItems.end(),
              [&strides, &stridesAreUnique](const SortItem& a, const SortItem& b) {
                  if(strides[a.index] == strides[b.index])
                  {
                      stridesAreUnique = false;
                      return a.originalPos < b.originalPos; // stable tie-breaker
                  }
                  return strides[a.index] > strides[b.index]; // descending by stride
              });

    for(size_t i = 0; i < numDims; ++i)
    {
        indices[i] = sortItems[i].index;
    }

    // Assign order based on sorted stride indices from longest strides to shortest.
    for(size_t i = 0; i < numDims; ++i)
    {
        strideOrder[indices[i]] = static_cast<int64_t>(numDims - i - 1);
    }

    if(!stridesAreUnique)
    {
        HIPDNN_SDK_LOG_WARN("extractStrideOrder(): Stride lengths "
                            << vecToString(strides) << " are not unique, the deduced stride order "
                            << vecToString(strideOrder) << " may not be correct.");
    }

    return strideOrder;
}

/**
 * @brief Generates strides that make a specified axis the most tightly packed dimension
 *
 * Derives a stride ordering from a reference tensor's strides, optionally rotating
 * the specified axis to be the most tightly packed (stride=1). Other dimensions
 * preserve their relative ordering from the reference layout.
 *
 * @param referenceStrides Strides of the reference tensor (used to determine dimension ordering)
 * @param referenceDims Dimensions of the reference tensor (used for singleton tiebreaking)
 * @param targetDims Dimensions of the tensor to generate strides for
 * @param axis If set, this dimension index becomes the most tightly packed (stride=1).
 *             If not set, the reference stride ordering is preserved as-is.
 * @return Strides for targetDims with the specified axis most tightly packed
 *
 * @code{.cpp}
 * // NCHW input, make axis 1 (C) most packed:
 * auto strides = generateStridesWithPackedAxis(
 *     {65536, 1024, 32, 1}, {2, 64, 32, 32}, {2, 64, 32, 32}, 1);
 * // Returns {64, 1, 4096, 128}
 * @endcode
 */
inline std::vector<int64_t>
    generateStridesWithPackedAxis(const std::vector<int64_t>& referenceStrides,
                                  const std::vector<int64_t>& referenceDims,
                                  const std::vector<int64_t>& targetDims,
                                  std::optional<int64_t> axis = std::nullopt)
{
    const size_t numDims = referenceStrides.size();

    // Sort dimension indices by reference strides ascending,
    // with singleton-dimension tiebreaker (singletons sort first)
    std::vector<size_t> sortedIndices(numDims);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(),
              sortedIndices.end(),
              [&referenceStrides, &referenceDims](size_t a, size_t b) {
                  if(referenceStrides[a] != referenceStrides[b])
                  {
                      return referenceStrides[a] < referenceStrides[b];
                  }
                  return (referenceDims[a] == 1) > (referenceDims[b] == 1);
              });

    // If axis is set, rotate so the axis dimension becomes most packed
    if(axis.has_value())
    {
        auto axisVal = static_cast<size_t>(axis.value());
        auto it = std::find(sortedIndices.begin(), sortedIndices.end(), axisVal);
        if(it != sortedIndices.end())
        {
            std::rotate(sortedIndices.begin(), it, sortedIndices.end());
        }
    }

    // Build stride order from inverse permutation
    std::vector<int64_t> strideOrder(numDims);
    for(size_t i = 0; i < numDims; ++i)
    {
        strideOrder[sortedIndices[i]] = static_cast<int64_t>(i);
    }

    return generateStrides(targetDims, strideOrder);
}

// Checks if the tensor defined by dims and strides is packed (contiguous in memory).
// Note: Assumes dims are positive (validated at graph level).
// Strides can be negative (for reversed dimensions).
inline bool isTensorPacked(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    if(dims.size() != strides.size())
    {
        throw std::invalid_argument("Dimensions and strides must have the same number of elements");
    }

    // Handle edge case: empty tensor
    if(dims.empty())
    {
        return true;
    }

    // Calculate total element count
    const auto count
        = std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1), std::multiplies<>());

    // Calculate memory span: the offset from first to last element
    // For each dimension i, the maximum offset is (dims[i] - 1) * strides[i]
    // This works correctly with negative strides (for reversed tensor dimensions)
    const auto space
        = std::inner_product(dims.begin(),
                             dims.end(),
                             strides.begin(),
                             static_cast<int64_t>(0),
                             std::plus<>(),
                             [](int64_t len, int64_t stride) { return (len - 1) * stride; });

    // A tensor is packed if all elements are contiguous:
    // total_elements == (max_offset + 1)
    return count == space + 1;
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

/**
 * @brief Constructs a full tensor indices vector from batch, channel, and spatial components
 *
 * Builds an index vector following NCHW ordering: batch at position 0, channel at position 1,
 * followed by spatial dimensions (H, W for 4D tensors; D, H, W for 5D tensors).
 *
 * @param batchIdx Batch index (position 0 in result)
 * @param channelIdx Channel index (position 1 in result)
 * @param spatialIndices Spatial dimension indices (positions 2+ in result)
 * @param spatialOffset Number of elements to skip at the start of spatialIndices
 * @return Combined index vector in NCHW/NCDHW order
 */
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

// Checks if a tensor shape is layout-agnostic.
// A tensor is layout-agnostic if it has at most one non-trivial dimension (size > 1).
// This includes:
// - Degenerate tensors (all dims=1): scalars
// - Channel-only/derived tensors (dims like {1, C, 1, 1}): scale, bias, mean, variance
// For these tensors, the stride order is ambiguous and doesn't affect memory access patterns.
inline bool isLayoutAgnostic(const std::vector<int64_t>& dims)
{
    size_t nonTrivialDims = 0;
    for(const auto dim : dims)
    {
        if(dim > 1)
        {
            ++nonTrivialDims;
        }
    }
    return nonTrivialDims <= 1;
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

} // namespace hipdnn_data_sdk::utilities
