// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <numeric>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceRmsnorm
{
public:
    /// RMSNorm forward: y = x / sqrt(mean(x^2) + epsilon) * scale [+ bias]
    ///
    /// @param x           Input tensor (NCHW or NHWC layout)
    /// @param scale       Per-channel scale tensor, shape [1, C, 1, ..., 1]
    /// @param y           Output tensor (same shape as x)
    /// @param epsilon     Small scalar for numerical stability
    /// @param invRms      Optional output: 1 / sqrt(mean(x^2) + epsilon) per channel
    /// @param bias        Optional per-channel bias tensor, shape [1, C, 1, ..., 1]
    template <class XDataType, class ScaleDataType, class YDataType, class ComputeDataType = float>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                        hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                        double epsilon,
                        hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* invRms = nullptr,
                        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* bias = nullptr)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "RMSNorm forward requires at least 2D tensor (batch and channel).");
        }

        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());

        auto nhw = static_cast<ComputeDataType>(elementsPerChannel);
        auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

        // Build dimensions for iteration: [batch, spatial...]
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]};
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto rmsnormFwdFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto sumSquares = static_cast<ComputeDataType>(0.0);

            // Calculate mean of squares for this channel
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto inVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    sumSquares = sumSquares + (inVal * inVal);
                });

            ComputeDataType meanSquares = sumSquares / nhw;
            auto invRmsValue = static_cast<ComputeDataType>(1.0)
                               / hipdnn_data_sdk::types::sqrt(meanSquares + epsilonCompute);

            // Apply normalization with scale: y = x * invRms * scale
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto xNorm = xVal * invRmsValue;

                    ComputeDataType yVal
                        = static_cast<ComputeDataType>(scale.getHostValue(0, cidx)) * xNorm;
                    if(bias != nullptr)
                    {
                        yVal += static_cast<ComputeDataType>(bias->getHostValue(0, cidx));
                    }
                    y.setHostValue(static_cast<YDataType>(yVal), fullIndices);
                });

            // Save inverse RMS for backward pass if provided
            if(invRms != nullptr)
            {
                invRms->setHostValue(static_cast<ComputeDataType>(invRmsValue), 0, cidx);
            }
        };

        // Build dimensions for parallel iteration - only channels
        auto nChannels = x.dims().at(1);
        std::vector<int64_t> parallelDims = {nChannels};

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(rmsnormFwdFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        // Mark all modified tensors as host-modified
        y.memory().markHostModified();

        if(invRms != nullptr)
        {
            invRms->memory().markHostModified();
        }
    }

private:
    static int64_t calculateElementsPerChannel(const std::vector<int64_t>& dims)
    {
        if(dims.size() < 2)
        {
            throw std::runtime_error("Tensor must have at least 2 dimensions (batch and channel).");
        }

        int64_t elementsPerChannel = dims.at(0); // batch dimension
        for(size_t i = 2; i < dims.size(); ++i)
        {
            elementsPerChannel *= dims.at(i);
        }
        return elementsPerChannel;
    }
};

} // namespace hipdnn_test_sdk::utilities
