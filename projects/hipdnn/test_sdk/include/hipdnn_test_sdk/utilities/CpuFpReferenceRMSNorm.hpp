// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceRMSNorm
{
public:
    /// RMSNorm forward: y = x / RMS(x) * scale [+ bias]
    /// where RMS is computed over the channel dimension for each (batch, spatial) position
    ///
    /// @param x           Input tensor (NCHW or NHWC layout)
    /// @param scale       Per-channel scale tensor, shape [1, C, 1, ..., 1]
    /// @param y           Output tensor (same shape as x)
    /// @param epsilon     Small scalar for numerical stability
    /// @param invRms      Optional output: 1 / RMS(x) per (batch, spatial) position
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

        auto nChannels = x.dims().at(1);
        auto channelCount = static_cast<ComputeDataType>(nChannels);
        auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

        // Build dimensions for iteration: [batch, spatial...]
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]};
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto rmsnormFwdFunc = [&](const std::vector<int64_t>& batchSpatialIndices) {
            auto batchIdx = batchSpatialIndices[0];

            // Compute sum of squares across channels for this (batch, spatial) position
            auto sumSquares = static_cast<ComputeDataType>(0.0);
            for(int64_t c = 0; c < nChannels; ++c)
            {
                auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                    batchIdx, c, batchSpatialIndices, 1);
                auto inVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                sumSquares += inVal * inVal;
            }

            ComputeDataType meanSquares = sumSquares / channelCount;
            auto invRmsValue = static_cast<ComputeDataType>(1.0)
                               / hipdnn_data_sdk::types::sqrt(meanSquares + epsilonCompute);

            // Apply normalization: y = x * invRms * scale [+ bias]
            for(int64_t c = 0; c < nChannels; ++c)
            {
                auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                    batchIdx, c, batchSpatialIndices, 1);
                auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                auto xNorm = xVal * invRmsValue;

                ComputeDataType yVal
                    = static_cast<ComputeDataType>(scale.getHostValue(0, c)) * xNorm;
                if(bias != nullptr)
                {
                    yVal += static_cast<ComputeDataType>(bias->getHostValue(0, c));
                }
                y.setHostValue(static_cast<YDataType>(yVal), fullIndices);
            }

            // Save inverse RMS if provided
            if(invRms != nullptr)
            {
                auto invRmsIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                    batchIdx, static_cast<int64_t>(0), batchSpatialIndices, 1);
                invRms->setHostValue(static_cast<ComputeDataType>(invRmsValue), invRmsIndices);
            }
        };

        // Parallelize across (batch, spatial) positions
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(rmsnormFwdFunc, batchAndSpatial);
        parallelFunc(std::thread::hardware_concurrency());

        // Mark all modified tensors as host-modified
        y.memory().markHostModified();

        if(invRms != nullptr)
        {
            invRms->memory().markHostModified();
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
