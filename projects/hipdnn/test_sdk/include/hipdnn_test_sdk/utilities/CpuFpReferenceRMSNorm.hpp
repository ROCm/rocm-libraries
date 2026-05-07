// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceRMSNorm
{
public:
    /// RMSNorm forward: y = x / RMS(x) * scale [+ bias]
    ///
    /// Normalized axes are derived from the scale tensor's shape: the non-1 dims
    /// of scale form a contiguous trailing suffix matching input, and those dims
    /// are normalized over (analogous to PyTorch's `normalized_shape`, encoded
    /// implicitly via the scale tensor). Dims where scale is 1 are the leading
    /// dims — they are "kept" in invRms (preserved at full size) while the
    /// normalized dims collapse to 1. Examples:
    ///   input [N, C, H, W], scale [1, C, H, W] → normalize over (C, H, W),
    ///                                           invRms shape [N, 1, 1, 1]
    ///   input [N, C, H, W], scale [1, 1, H, W] → normalize over (H, W),
    ///                                           invRms shape [N, C, 1, 1]
    ///   input [N, C, H, W], scale [1, 1, 1, W] → normalize over (W),
    ///                                           invRms shape [N, C, H, 1]
    ///
    /// @param x           Input tensor
    /// @param scale       Scale tensor, same rank as x; dim 0 = 1 (batch broadcast);
    ///                    non-1 dims must form a trailing suffix matching input.
    /// @param y           Output tensor (same shape as x)
    /// @param epsilon     Small scalar for numerical stability
    /// @param invRms      Optional output: 1 / RMS(x); shape = input with normalized dims → 1
    /// @param bias        Optional bias tensor, same shape as scale
    template <class XDataType, class ScaleDataType, class YDataType, class ComputeDataType = float>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                        hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                        double epsilon,
                        hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* invRms = nullptr,
                        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* bias = nullptr)
    {
        const auto& xDims = x.dims();
        const auto& scaleDims = scale.dims();

        if(xDims.size() < 2)
        {
            throw std::runtime_error("RMSNorm forward requires at least 2D input tensor (batch and "
                                     "at least one feature dim).");
        }
        if(scaleDims.size() != xDims.size())
        {
            throw std::runtime_error("RMSNorm forward requires scale rank to equal input rank.");
        }

        const size_t rank = xDims.size();

        // Normalized shape = maximal trailing suffix of dims where scale[i] == input[i].
        // Clamp so batch is always leading (never normalized).
        // matchCount = number of trailing dims where scaleDims[i] == xDims[i]
        const auto [scaleMismatch, _]
            = std::mismatch(scaleDims.rbegin(), scaleDims.rend(), xDims.rbegin(), xDims.rend());
        const auto matchCount
            = static_cast<size_t>(std::distance(scaleDims.rbegin(), scaleMismatch));
        const size_t reductionStart = (matchCount >= rank) ? 1 : rank - matchCount;
        if(reductionStart == rank)
        {
            // Validator should have rejected this; defensive guard for direct callers.
            throw std::runtime_error(
                "RMSNorm forward: scale has no trailing dims matching input — no "
                "normalized axes can be derived.");
        }

        // Leading dims: [0, reductionStart) — batch + leading dims preserved through
        // the op (these are "kept" at full size in invRms).
        // Reduction dims: [reductionStart, rank) — normalized, collapsed per leading position.
        const auto splitOffset = static_cast<std::ptrdiff_t>(reductionStart);
        const std::vector<int64_t> leadingDims(xDims.begin(), xDims.begin() + splitOffset);
        const std::vector<int64_t> reductionDims(xDims.begin() + splitOffset, xDims.end());

        const auto reductionCount = std::accumulate(
            reductionDims.begin(), reductionDims.end(), int64_t{1}, std::multiplies<>{});
        const auto reductionCountCompute = static_cast<ComputeDataType>(reductionCount);
        const auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

        // Compute RMS-normalized output for one leading position.
        //
        // Running example: input [N, C, H, W], scale [1, 1, H, W]
        //   leadingDims   = [N, C]            reductionDims = [H, W]
        //   leadingIdx    = [n, c]            redIdx        = [h, w]
        //   fullIdx       = [n, c, h, w]      (leadingIdx ++ redIdx)
        //   scaleIdx      = [0, 0, h, w]      (zero for leading region, redIdx for the rest)
        //   invRmsIdx     = [n, c, 0, 0]      (leadingIdx padded with zeros to rank)
        auto rmsnormFwdFunc = [&](const std::vector<int64_t>& leadingIdx) {
            // Pass 1: accumulate sum(x^2) over the reduction dims at this leading position.
            //   invRms[n, c, 0, 0] = 1 / sqrt(mean(x^2 over [h, w]) + epsilon)
            auto sumSquares = static_cast<ComputeDataType>(0.0);
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                reductionDims, [&](const std::vector<int64_t>& redIdx) {
                    std::vector<int64_t> fullIdx = leadingIdx;
                    fullIdx.insert(fullIdx.end(), redIdx.begin(), redIdx.end());
                    const auto inVal = static_cast<ComputeDataType>(x.getHostValue(fullIdx));
                    sumSquares += inVal * inVal;
                });

            const auto meanSquares = sumSquares / reductionCountCompute;
            const auto invRmsValue = static_cast<ComputeDataType>(1.0)
                                     / hipdnn_data_sdk::types::sqrt(meanSquares + epsilonCompute);

            // Pass 2: write y = scale * x * invRms (+ bias), iterating the same reduction
            // positions but now reading scale/bias at their leading-zeroed indices.
            //   y[n, c, h, w] = scale[0, 0, h, w] * x[n, c, h, w] * invRms[n, c, 0, 0]
            //                   (+ bias[0, 0, h, w])
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                reductionDims, [&](const std::vector<int64_t>& redIdx) {
                    std::vector<int64_t> fullIdx = leadingIdx;
                    fullIdx.insert(fullIdx.end(), redIdx.begin(), redIdx.end());

                    // Scale's leading-region positions are 0 (broadcast); the reduction-region
                    // positions match fullIdx. For the running example: scaleIdx = [0, 0, h, w].
                    std::vector<int64_t> scaleIdx(rank, 0);
                    for(size_t i = reductionStart; i < rank; ++i)
                    {
                        scaleIdx[i] = fullIdx[i];
                    }

                    const auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIdx));
                    const auto xNorm = xVal * invRmsValue;
                    ComputeDataType yVal
                        = static_cast<ComputeDataType>(scale.getHostValue(scaleIdx)) * xNorm;
                    if(bias != nullptr)
                    {
                        yVal += static_cast<ComputeDataType>(bias->getHostValue(scaleIdx));
                    }
                    y.setHostValue(static_cast<YDataType>(yVal), fullIdx);
                });

            // invRms keeps the leading dims and collapses the reduction dims to 1, so its
            // index is leadingIdx padded with zeros. Running example: invRmsIdx = [n, c, 0, 0].
            if(invRms != nullptr)
            {
                std::vector<int64_t> invRmsIdx = leadingIdx;
                invRmsIdx.resize(rank, 0);
                invRms->setHostValue(static_cast<ComputeDataType>(invRmsValue), invRmsIdx);
            }
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(rmsnormFwdFunc, leadingDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
        if(invRms != nullptr)
        {
            invRms->memory().markHostModified();
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
