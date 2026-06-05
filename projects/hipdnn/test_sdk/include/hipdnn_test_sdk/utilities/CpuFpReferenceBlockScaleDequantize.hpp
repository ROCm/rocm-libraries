// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceBlockScaleDequantize
{
public:
    /// Block scale dequantize: Y[i] = X[i] * scale[block_of(i)]
    ///
    /// blockSize entries map to the trailing dimensions of the tensor.
    /// For a tensor with N dims and blockSize with K entries, blockSize[i]
    /// applies to dimension (N - K + i). scale_index[d] = x_index[d] / blockSize[i]
    /// for blocked trailing dims, and scale_index[d] = x_index[d] for leading dims.
    ///
    /// @param x                Input tensor (blocked low-precision data)
    /// @param scale            Per-block scale tensor
    /// @param y                Output tensor (dequantized, same shape as x)
    /// @param blockSize        Block size for each blocked dimension (from attributes)
    /// @param isNegativeScale  If true, scale represents negative exponent: Y = X * 2^(-scale)
    template <class XDataType, class ScaleDataType, class YDataType, class ComputeDataType = float>
    static void dequantize(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                           const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                           hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                           const std::vector<int32_t>& blockSize,
                           bool isNegativeScale = false)
    {
        const auto& xDims = x.dims();
        const auto& scaleDims = scale.dims();

        if(xDims.empty())
        {
            throw std::runtime_error("BlockScaleDequantize requires non-empty tensor dimensions.");
        }

        // blockSize entries map to the trailing dimensions of the tensor.
        std::vector<int64_t> effectiveBlockSize(xDims.size(), 1);
        for(size_t i = 0; i < blockSize.size() && i < xDims.size(); ++i)
        {
            effectiveBlockSize[xDims.size() - blockSize.size() + i] = blockSize[i];
        }

        // Validate scale dimensions are consistent with data dims and block size.
        // Scale must have the same rank as x — there is no dim-mapping for broadcast.
        if(scaleDims.size() != xDims.size())
        {
            throw std::invalid_argument(
                "BlockScaleDequantize: scale tensor rank (" + std::to_string(scaleDims.size())
                + ") must equal input tensor rank (" + std::to_string(xDims.size()) + ").");
        }

        for(size_t d = 0; d < scaleDims.size(); ++d)
        {
            const auto expectedScaleDim
                = (xDims[d] + effectiveBlockSize[d] - 1) / effectiveBlockSize[d];
            if(scaleDims[d] != expectedScaleDim)
            {
                throw std::invalid_argument("BlockScaleDequantize: scale dim[" + std::to_string(d)
                                            + "] is " + std::to_string(scaleDims[d])
                                            + " but expected " + std::to_string(expectedScaleDim)
                                            + " (ceil(" + std::to_string(xDims[d]) + " / "
                                            + std::to_string(effectiveBlockSize[d]) + ")).");
            }
        }

        auto dequantizeFunc = [&](const std::vector<int64_t>& xIndices) {
            // Compute scale indices by dividing by block size for blocked dims.
            // Size to scale rank — getIndex() throws if indices.size() > strides().size().
            std::vector<int64_t> scaleIndices(scaleDims.size());
            for(size_t d = 0; d < scaleDims.size(); ++d)
            {
                scaleIndices[d] = xIndices[d] / effectiveBlockSize[d];
            }

            auto xVal = static_cast<ComputeDataType>(x.getHostValue(xIndices));
            auto scaleVal = static_cast<ComputeDataType>(scale.getHostValue(scaleIndices));

            ComputeDataType yVal;
            if(isNegativeScale)
            {
                // Negative scale: Y = X * 2^(-scale_value)
                yVal = xVal * std::pow(static_cast<ComputeDataType>(2.0), -scaleVal);
            }
            else
            {
                // Normal scale: Y = X * scale
                yVal = xVal * scaleVal;
            }

            y.setHostValue(static_cast<YDataType>(yVal), xIndices);
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(dequantizeFunc, xDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
    }
};

} // namespace hipdnn_test_sdk::utilities
