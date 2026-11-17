// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_frontend/Error.hpp>

namespace hipdnn_frontend::detail
{
// Validates one resample parameter vector: one value per spatial dimension, each value at
// least minValue. An empty vector passes; requireResampleParams rejects it where the values
// are needed.
inline Error validateResampleParam(const std::string& nodeName,
                                   const std::string& paramName,
                                   const std::vector<int64_t>& values,
                                   size_t spatialRank,
                                   int64_t minValue)
{
    if(values.empty())
    {
        return {};
    }

    HIPDNN_RETURN_IF_NE(
        values.size(),
        spatialRank,
        ErrorCode::INVALID_VALUE,
        nodeName + ": " + paramName + " parameter count (" + std::to_string(values.size())
            + ") must match spatial dimension count (" + std::to_string(spatialRank) + ")");

    for(size_t i = 0; i < values.size(); ++i)
    {
        if(values[i] < minValue)
        {
            std::string message = nodeName;
            message += ": ";
            message += paramName;
            message += " at index ";
            message += std::to_string(i);
            message += " (";
            message += std::to_string(values[i]);
            message += ") must be >= ";
            message += std::to_string(minValue);
            return {ErrorCode::INVALID_VALUE, std::move(message)};
        }
    }

    return {};
}

// Validates the non-empty resample window, stride and padding vectors against the spatial
// rank: window and stride values must be >= 1, padding values >= 0.
inline Error validateResampleParams(const std::string& nodeName,
                                    size_t spatialRank,
                                    const std::vector<int64_t>& window,
                                    const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& prePadding,
                                    const std::vector<int64_t>& postPadding)
{
    HIPDNN_CHECK_ERROR(validateResampleParam(nodeName, "window", window, spatialRank, 1));
    HIPDNN_CHECK_ERROR(validateResampleParam(nodeName, "stride", stride, spatialRank, 1));
    HIPDNN_CHECK_ERROR(validateResampleParam(nodeName, "pre_padding", prePadding, spatialRank, 0));
    HIPDNN_CHECK_ERROR(
        validateResampleParam(nodeName, "post_padding", postPadding, spatialRank, 0));
    return {};
}

// Requires every resample vector to be set, then validates them as validateResampleParams
// does. Used where the vectors are indexed per spatial dimension.
inline Error requireResampleParams(const std::string& nodeName,
                                   size_t spatialRank,
                                   const std::vector<int64_t>& window,
                                   const std::vector<int64_t>& stride,
                                   const std::vector<int64_t>& prePadding,
                                   const std::vector<int64_t>& postPadding)
{
    HIPDNN_RETURN_IF_TRUE(
        window.empty(), ErrorCode::ATTRIBUTE_NOT_SET, nodeName + " missing window");
    HIPDNN_RETURN_IF_TRUE(
        stride.empty(), ErrorCode::ATTRIBUTE_NOT_SET, nodeName + " missing stride");
    HIPDNN_RETURN_IF_TRUE(
        prePadding.empty(), ErrorCode::ATTRIBUTE_NOT_SET, nodeName + " missing pre_padding");
    HIPDNN_RETURN_IF_TRUE(
        postPadding.empty(), ErrorCode::ATTRIBUTE_NOT_SET, nodeName + " missing post_padding");
    return validateResampleParams(nodeName, spatialRank, window, stride, prePadding, postPadding);
}

// Rejects a resample window wider than the padded input-space extent along one spatial
// dimension, where no window position fits. The input space is x for the forward node and dx
// for the backward node.
inline Error validateResampleWindowFitsPaddedInput(const std::string& nodeName,
                                                   int64_t inputSize,
                                                   int64_t prePadding,
                                                   int64_t postPadding,
                                                   int64_t window,
                                                   size_t spatialIdx)
{
    const int64_t paddedSize = inputSize + prePadding + postPadding;
    HIPDNN_RETURN_IF_LT(paddedSize,
                        window,
                        ErrorCode::INVALID_VALUE,
                        nodeName + ": window (" + std::to_string(window) + ") at spatial index "
                            + std::to_string(spatialIdx)
                            + " is larger than the padded input extent ("
                            + std::to_string(paddedSize) + ")");
    return {};
}
} // namespace hipdnn_frontend::detail
