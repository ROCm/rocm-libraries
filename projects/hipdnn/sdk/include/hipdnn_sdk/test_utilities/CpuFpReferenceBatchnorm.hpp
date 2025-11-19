// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/StaticCast.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <numeric>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

class CpuFpReferenceBatchnorm
{
public:
    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void fwdInference(const TensorBase<XDataType>& x,
                             const TensorBase<ScaleBiasDataType>& scale,
                             const TensorBase<ScaleBiasDataType>& bias,
                             const TensorBase<MeanVarianceDataType>& estimatedMean,
                             const TensorBase<MeanVarianceDataType>& invVariance,
                             TensorBase<YDataType>& y)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm inference requires at least 2D tensor (batch and channel).");
        }

        auto batchnormFwdInferenceFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[1];
            auto mean = staticCast<ComputeDataType>(estimatedMean.getHostValue(0, cidx));
            auto invVarianceValue = staticCast<ComputeDataType>(invVariance.getHostValue(0, cidx));

            //There is some extra casting in here to deal with double -> float implicit casts.
            auto inVal = staticCast<ComputeDataType>(x.getHostValue(indices));
            ComputeDataType elemStd = inVal - mean;
            ComputeDataType inhat = elemStd * invVarianceValue;

            y.setHostValue(staticCast<YDataType>(
                               (staticCast<ComputeDataType>(scale.getHostValue(0, cidx)) * inhat)
                               + staticCast<ComputeDataType>(bias.getHostValue(0, cidx))),
                           indices);
        };

        // Iterate all indices in parallel
        auto parallelFunc = hipdnn_sdk::test_utilities::makeParallelTensorFunctor(
            batchnormFwdInferenceFunc, x.dims());
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified(); // Mark y memory as modified on host
    }

    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType = ScaleBiasDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void fwdTraining(const TensorBase<XDataType>& x,
                            const TensorBase<ScaleBiasDataType>& scale,
                            const TensorBase<ScaleBiasDataType>& bias,
                            TensorBase<YDataType>& y,
                            double epsilon,
                            double momentum,
                            TensorBase<MeanVarianceDataType>* mean = nullptr,
                            TensorBase<MeanVarianceDataType>* invVariance = nullptr,
                            const TensorBase<MeanVarianceDataType>* prevRunningMean = nullptr,
                            const TensorBase<MeanVarianceDataType>* prevRunningVariance = nullptr,
                            TensorBase<MeanVarianceDataType>* nextRunningMean = nullptr,
                            TensorBase<MeanVarianceDataType>* nextRunningVariance = nullptr)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm training requires at least 2D tensor (batch and channel).");
        }

        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());

        auto nhw = staticCast<ComputeDataType>(elementsPerChannel);
        auto epsilonCompute = staticCast<ComputeDataType>(epsilon);
        auto momentumCompute = staticCast<ComputeDataType>(momentum);

        // Build dimensions for iteration: [batch, spatial...]
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]};
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormFwdTrainingFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto meanAccum = staticCast<ComputeDataType>(0.0);
            auto varianceAccum = staticCast<ComputeDataType>(0.0);

            // Calculate mean and variance for this channel
            iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices
                        = buildTensorIndices(batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto inVal = staticCast<ComputeDataType>(x.getHostValue(fullIndices));
                    meanAccum = meanAccum + inVal;
                    varianceAccum = varianceAccum + (inVal * inVal);
                });

            // NOTE: Different operation order from MIOpen produces expected FP differences.
            // Both implementations are correct; validated using RMS error tolerance
            ComputeDataType channelMean = meanAccum / nhw;
            ComputeDataType channelVariance = (varianceAccum / nhw) - (channelMean * channelMean);

            auto invVar
                = staticCast<ComputeDataType>(1.0) / sqrtInternal(channelVariance + epsilonCompute);

            // Apply normalization with scale and bias
            iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices
                        = buildTensorIndices(batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = staticCast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto xHat = (xVal - channelMean) * invVar;

                    y.setHostValue(
                        staticCast<YDataType>(
                            staticCast<ComputeDataType>(scale.getHostValue(0, cidx)) * xHat
                            + staticCast<ComputeDataType>(bias.getHostValue(0, cidx))),
                        fullIndices);
                });

            // Save mean and inverse variance for backward pass if provided
            if(mean != nullptr)
            {
                mean->setHostValue(staticCast<MeanVarianceDataType>(channelMean), 0, cidx);
            }

            if(invVariance != nullptr)
            {
                invVariance->setHostValue(staticCast<MeanVarianceDataType>(invVar), 0, cidx);
            }

            // Update running statistics if all required tensors are provided
            if(prevRunningMean != nullptr && prevRunningVariance != nullptr
               && nextRunningMean != nullptr && nextRunningVariance != nullptr)
            {
                auto one = staticCast<ComputeDataType>(1.0f);
                auto currentMean
                    = staticCast<ComputeDataType>(prevRunningMean->getHostValue(0, cidx));
                auto newMean = staticCast<MeanVarianceDataType>(
                    (one - momentumCompute) * currentMean + momentumCompute * channelMean);
                nextRunningMean->setHostValue(newMean, 0, cidx);

                auto currentVar
                    = staticCast<ComputeDataType>(prevRunningVariance->getHostValue(0, cidx));
                // Apply Bessel's correction for unbiased variance estimate
                ComputeDataType adjustedVariance
                    = (nhw == one)
                          ? channelVariance
                          : staticCast<ComputeDataType>((nhw / (nhw - one)) * channelVariance);
                auto newVar = staticCast<MeanVarianceDataType>(
                    (one - momentumCompute) * currentVar + momentumCompute * adjustedVariance);
                nextRunningVariance->setHostValue(newVar, 0, cidx);
            }
        };

        // Build dimensions for parallel iteration
        auto nChannels = x.dims().at(1);
        std::vector<int64_t> parallelDims = {nChannels};

        auto parallelFunc = hipdnn_sdk::test_utilities::makeParallelTensorFunctor(
            batchnormFwdTrainingFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        // Mark all modified tensors as host-modified
        y.memory().markHostModified();

        if(mean != nullptr)
        {
            mean->memory().markHostModified();
        }

        if(invVariance != nullptr)
        {
            invVariance->memory().markHostModified();
        }

        if(nextRunningMean != nullptr)
        {
            nextRunningMean->memory().markHostModified();
        }

        if(nextRunningVariance != nullptr)
        {
            nextRunningVariance->memory().markHostModified();
        }
    }

    template <class DyDataType,
              class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType = ScaleBiasDataType,
              class DxDataType = XDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void backward(const TensorBase<DyDataType>& dy,
                         const TensorBase<XDataType>& x,
                         const TensorBase<MeanVarianceDataType>& mean,
                         const TensorBase<MeanVarianceDataType>& invVariance,
                         const TensorBase<ScaleBiasDataType>& scale,
                         TensorBase<DxDataType>& dx,
                         TensorBase<ScaleBiasDataType>& dscale,
                         TensorBase<ScaleBiasDataType>& dbias)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm backward requires at least 2D tensor (batch and channel).");
        }

        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());
        //Cant cast directly from int64 to half or bloat16 so cast to float first.
        auto nhwF = staticCast<ComputeDataType>(elementsPerChannel);

        // Include batch dimension with spatial dimensions for iteration
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]}; // batch dimension
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormBwdFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto channelMean = staticCast<ComputeDataType>(mean.getHostValue(0, cidx));
            auto channelInvVariance = staticCast<ComputeDataType>(
                invVariance.getHostValue(0, cidx)); // 1 / sqrt(var + eps)
            auto channelScale = staticCast<ComputeDataType>(scale.getHostValue(0, cidx));

            // Calculate dot product for (x - mean) * channelInvVariance * dy and ∑ dy for this channel
            auto dotProduct = staticCast<ComputeDataType>(0.0);
            auto sumDy = staticCast<ComputeDataType>(0.0);

            iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices
                        = buildTensorIndices(batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = staticCast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto dyVal = staticCast<ComputeDataType>(dy.getHostValue(fullIndices));

                    ComputeDataType xHat = (xVal - channelMean) * channelInvVariance;
                    // for half no += operator exists
                    dotProduct = dotProduct + (xHat * dyVal);
                    sumDy = sumDy + dyVal;
                });

            // Per channel:
            // - dscale = ∑ (xHat * dy)
            // - dbias = ∑ dy
            // - dx = scale * invVariance * (dy - mean(dy) - xHat * mean(dy * xHat))

            dscale.setHostValue(staticCast<ScaleBiasDataType>(dotProduct), 0, cidx);
            dbias.setHostValue(staticCast<ScaleBiasDataType>(sumDy), 0, cidx);

            auto meanDy = sumDy / nhwF;
            auto meanDyXhat = dotProduct / nhwF;
            auto scalarCoef = channelScale * channelInvVariance;

            iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices
                        = buildTensorIndices(batchSpatialIndices[0], cidx, batchSpatialIndices, 1);

                    auto xVal = staticCast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto dyVal = staticCast<ComputeDataType>(dy.getHostValue(fullIndices));

                    auto xHat = (xVal - channelMean) * channelInvVariance;
                    auto dxVal = (dyVal - meanDy - xHat * meanDyXhat) * scalarCoef;

                    dx.setHostValue(staticCast<DxDataType>(dxVal), fullIndices);
                });
        };

        // Build dimensions for parallel iteration - only channels
        auto nChannels = x.dims().at(1);
        std::vector<int64_t> parallelDims = {nChannels};

        auto parallelFunc
            = hipdnn_sdk::test_utilities::makeParallelTensorFunctor(batchnormBwdFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        dx.memory().markHostModified();
        dscale.memory().markHostModified();
        dbias.memory().markHostModified();
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

    static double sqrtInternal(double value)
    {
        return std::sqrt(value);
    }

    static float sqrtInternal(float value)
    {
        return sqrtf(value);
    }

    static hip_bfloat16 sqrtInternal(hip_bfloat16 value)
    {
        return static_cast<hip_bfloat16>(sqrtf(static_cast<float>(value)));
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
