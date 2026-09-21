// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

/// Validates convolution parameters against tensor dimensions.
/// Checks that strides/dilations/padding vectors have correct sizes and valid values,
/// and that the output tensor spatial dimensions match the expected convolution output.
///
/// Callers are responsible for validating tensor dimension counts before calling this
/// (e.g. restricting to 3D/4D/5D for GPU kernels, or >= 3 for a generic CPU path).
/// This function assumes all three tensors have the same number of dimensions.
inline void validateConvolutionParams(const std::vector<int64_t>& xDims,
                                      const std::vector<int64_t>& wDims,
                                      const std::vector<int64_t>& yDims,
                                      const std::vector<int64_t>& strides,
                                      const std::vector<int64_t>& dilations,
                                      const std::vector<int64_t>& prePadding,
                                      const std::vector<int64_t>& postPadding)
{
    const auto nDims = xDims.size();
    const auto nSpatialDims = nDims - 2;

    if(wDims.size() != nDims)
    {
        throw std::invalid_argument(
            "Weight tensor must have the same number of dimensions as input");
    }

    if(yDims.size() != nDims)
    {
        throw std::invalid_argument(
            "Output tensor must have the same number of dimensions as input");
    }

    if(strides.size() != nSpatialDims)
    {
        throw std::invalid_argument("Strides must have exactly " + std::to_string(nSpatialDims)
                                    + " elements for this convolution");
    }

    if(dilations.size() != nSpatialDims)
    {
        throw std::invalid_argument("Dilations must have exactly " + std::to_string(nSpatialDims)
                                    + " elements for this convolution");
    }

    if(prePadding.size() != nSpatialDims)
    {
        throw std::invalid_argument("PrePadding must have exactly " + std::to_string(nSpatialDims)
                                    + " elements for this convolution");
    }

    if(postPadding.size() != nSpatialDims)
    {
        throw std::invalid_argument("PostPadding must have exactly " + std::to_string(nSpatialDims)
                                    + " elements for this convolution");
    }

    for(size_t i = 0; i < nSpatialDims; ++i)
    {
        if(strides[i] <= 0)
        {
            throw std::invalid_argument("Stride values must be positive");
        }

        if(dilations[i] <= 0)
        {
            throw std::invalid_argument("Dilation values must be positive");
        }

        if(prePadding[i] < 0)
        {
            throw std::invalid_argument("PrePadding values must be non-negative");
        }

        if(postPadding[i] < 0)
        {
            throw std::invalid_argument("PostPadding values must be non-negative");
        }

        const int64_t kernelSize = (dilations[i] * (wDims[i + 2] - 1)) + 1;
        const int64_t expectedOutputDim
            = ((xDims[i + 2] + prePadding[i] + postPadding[i] - kernelSize) / strides[i]) + 1;

        if(expectedOutputDim != yDims[i + 2])
        {
            throw std::invalid_argument("Output dimension " + std::to_string(yDims[i + 2])
                                        + " at spatial dimension " + std::to_string(i)
                                        + " does not match expected dimension "
                                        + std::to_string(expectedOutputDim));
        }
    }
}

/// TensorBase convenience overload — delegates to the dims-based overload.
template <typename T1, typename T2, typename T3>
void validateConvolutionParams(const hipdnn_data_sdk::utilities::TensorBase<T1>& x,
                               const hipdnn_data_sdk::utilities::TensorBase<T2>& w,
                               const hipdnn_data_sdk::utilities::TensorBase<T3>& y,
                               const std::vector<int64_t>& strides,
                               const std::vector<int64_t>& dilations,
                               const std::vector<int64_t>& prePadding,
                               const std::vector<int64_t>& postPadding)
{
    validateConvolutionParams(
        x.dims(), w.dims(), y.dims(), strides, dilations, prePadding, postPadding);
}

} // namespace hipdnn_test_sdk::utilities
