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

class CpuReferenceRMSNorm
{
public:
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
            throw std::runtime_error("RMSNorm forward requires at least 2D input tensor.");
        }

        if(scaleDims.size() != xDims.size())
        {
            throw std::runtime_error("RMSNorm forward requires scale rank to equal input rank.");
        }

        const size_t rank = xDims.size();

        const auto [scaleMismatch, _] =
            std::mismatch(scaleDims.rbegin(), scaleDims.rend(), xDims.rbegin(), xDims.rend());

        const auto matchCount =
            static_cast<size_t>(std::distance(scaleDims.rbegin(), scaleMismatch));

        const size_t reductionStart = (matchCount >= rank) ? 1 : rank - matchCount;

        if(reductionStart == rank)
        {
            throw std::runtime_error("RMSNorm forward: no normalized axes derived.");
        }

        const auto splitOffset = static_cast<std::ptrdiff_t>(reductionStart);

        const std::vector<int64_t> leadingDims(xDims.begin(), xDims.begin() + splitOffset);
        const std::vector<int64_t> reductionDims(xDims.begin() + splitOffset, xDims.end());

        const auto reductionCount = std::accumulate(
            reductionDims.begin(), reductionDims.end(), int64_t{1}, std::multiplies<>{});

        const auto reductionCountCompute = static_cast<ComputeDataType>(reductionCount);
        const auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

        auto rmsnormFwdFunc = [&](const std::vector<int64_t>& leadingIdx) {
            ComputeDataType sumSquares = static_cast<ComputeDataType>(0.0);

            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                reductionDims, [&](const std::vector<int64_t>& redIdx) {
                    std::vector<int64_t> fullIdx = leadingIdx;
                    fullIdx.insert(fullIdx.end(), redIdx.begin(), redIdx.end());

                    auto v = static_cast<ComputeDataType>(x.getHostValue(fullIdx));
                    sumSquares += v * v;
                });

            const auto meanSquares = sumSquares / reductionCountCompute;
            const auto invRmsValue =
                static_cast<ComputeDataType>(1.0)
                / hipdnn_data_sdk::types::sqrt(meanSquares + epsilonCompute);

            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                reductionDims, [&](const std::vector<int64_t>& redIdx) {
                    std::vector<int64_t> fullIdx = leadingIdx;
                    fullIdx.insert(fullIdx.end(), redIdx.begin(), redIdx.end());

                    std::vector<int64_t> scaleIdx(rank, 0);
                    for(size_t i = reductionStart; i < rank; ++i)
                    {
                        scaleIdx[i] = fullIdx[i];
                    }

                    const auto xVal =
                        static_cast<ComputeDataType>(x.getHostValue(fullIdx));

                    ComputeDataType yVal =
                        static_cast<ComputeDataType>(scale.getHostValue(scaleIdx))
                        * xVal * invRmsValue;

                    if(bias != nullptr)
                    {
                        yVal += static_cast<ComputeDataType>(bias->getHostValue(scaleIdx));
                    }

                    y.setHostValue(static_cast<YDataType>(yVal), fullIdx);
                });

            if(invRms != nullptr)
            {
                std::vector<int64_t> invRmsIdx = leadingIdx;
                invRmsIdx.resize(rank, 0);

                invRms->setHostValue(invRmsValue, invRmsIdx);
            }
        };

        auto parallelFunc =
            hipdnn_test_sdk::detail::makeParallelTensorFunctor(rmsnormFwdFunc, leadingDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();

        if(invRms != nullptr)
        {
            invRms->memory().markHostModified();
        }
    }
};

} // namespace hipdnn_test_sdk::utilities