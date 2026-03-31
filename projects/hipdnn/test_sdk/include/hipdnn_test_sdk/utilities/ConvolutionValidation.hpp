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
template <typename T1, typename T2, typename T3>
void validateConvolutionParams(const hipdnn_data_sdk::utilities::TensorBase<T1>& x,
                               const hipdnn_data_sdk::utilities::TensorBase<T2>& w,
                               const hipdnn_data_sdk::utilities::TensorBase<T3>& y,
                               const std::vector<int64_t>& strides,
                               const std::vector<int64_t>& dilations,
                               const std::vector<int64_t>& prePadding,
                               const std::vector<int64_t>& postPadding)
{
    const auto nDims = x.dims().size();
    const auto nSpatialDims = nDims - 2;

    if(w.dims().size() != nDims)
    {
        throw std::invalid_argument(
            "Weight tensor must have the same number of dimensions as input");
    }

    if(y.dims().size() != nDims)
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

    const auto& xDims = x.dims();
    const auto& wDims = w.dims();
    const auto& yDims = y.dims();

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

        const int64_t xDim = xDims[i + 2];
        const int64_t kernelDim = wDims[i + 2];
        const int64_t yDim = yDims[i + 2];

        const int64_t kernelSize = (dilations[i] * (kernelDim - 1)) + 1;
        const int64_t expectedOutputDim
            = ((xDim + prePadding[i] + postPadding[i] - kernelSize) / strides[i]) + 1;

        if(expectedOutputDim != yDim)
        {
            throw std::invalid_argument("Output dimension " + std::to_string(yDim)
                                        + " at spatial dimension " + std::to_string(i)
                                        + " does not match expected dimension "
                                        + std::to_string(expectedOutputDim));
        }
    }
}

} // namespace hipdnn_test_sdk::utilities
