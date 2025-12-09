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

namespace hipdnn_sdk::test_utilities
{

class CpuFpReferenceBatchnorm
{
public:
    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void fwdInference(const utilities::TensorBase<XDataType>& x,
                             const utilities::TensorBase<ScaleBiasDataType>& scale,
                             const utilities::TensorBase<ScaleBiasDataType>& bias,
                             const utilities::TensorBase<MeanVarianceDataType>& estimatedMean,
                             const utilities::TensorBase<MeanVarianceDataType>& invVariance,
                             utilities::TensorBase<YDataType>& y)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm inference requires at least 2D tensor (batch and channel).");
        }

        auto batchnormFwdInferenceFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[1];
            auto mean = utilities::staticCast<ComputeDataType>(estimatedMean.getHostValue(0, cidx));
            auto invVarianceValue
                = utilities::staticCast<ComputeDataType>(invVariance.getHostValue(0, cidx));

            //There is some extra casting in here to deal with double -> float implicit casts.
            auto inVal = utilities::staticCast<ComputeDataType>(x.getHostValue(indices));
            ComputeDataType elemStd = inVal - mean;
            ComputeDataType inhat = elemStd * invVarianceValue;

            y.setHostValue(
                utilities::staticCast<YDataType>(
                    (utilities::staticCast<ComputeDataType>(scale.getHostValue(0, cidx)) * inhat)
                    + utilities::staticCast<ComputeDataType>(bias.getHostValue(0, cidx))),
                indices);
        };

        // Iterate all indices in parallel
        auto parallelFunc = makeParallelTensorFunctor(batchnormFwdInferenceFunc, x.dims());
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified(); // Mark y memory as modified on host
    }

    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType = ScaleBiasDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void
        fwdTraining(const utilities::TensorBase<XDataType>& x,
                    const utilities::TensorBase<ScaleBiasDataType>& scale,
                    const utilities::TensorBase<ScaleBiasDataType>& bias,
                    utilities::TensorBase<YDataType>& y,
                    double epsilon,
                    double momentum,
                    utilities::TensorBase<MeanVarianceDataType>* mean = nullptr,
                    utilities::TensorBase<MeanVarianceDataType>* invVariance = nullptr,
                    const utilities::TensorBase<MeanVarianceDataType>* prevRunningMean = nullptr,
                    const utilities::TensorBase<MeanVarianceDataType>* prevRunningVariance
                    = nullptr,
                    utilities::TensorBase<MeanVarianceDataType>* nextRunningMean = nullptr,
                    utilities::TensorBase<MeanVarianceDataType>* nextRunningVariance = nullptr)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm training requires at least 2D tensor (batch and channel).");
        }

        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());

        auto nhw = utilities::staticCast<ComputeDataType>(elementsPerChannel);
        auto epsilonCompute = utilities::staticCast<ComputeDataType>(epsilon);
        auto momentumCompute = utilities::staticCast<ComputeDataType>(momentum);

        // Build dimensions for iteration: [batch, spatial...]
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]};
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormFwdTrainingFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto meanAccum = utilities::staticCast<ComputeDataType>(0.0);
            auto varianceAccum = utilities::staticCast<ComputeDataType>(0.0);

            // Calculate mean and variance for this channel
            utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto inVal
                        = utilities::staticCast<ComputeDataType>(x.getHostValue(fullIndices));
                    meanAccum = meanAccum + inVal;
                    varianceAccum = varianceAccum + (inVal * inVal);
                });

            // NOTE: Different operation order from MIOpen produces expected FP differences.
            // Both implementations are correct; validated using RMS error tolerance
            ComputeDataType channelMean = meanAccum / nhw;
            ComputeDataType channelVariance = (varianceAccum / nhw) - (channelMean * channelMean);

            auto invVar = utilities::staticCast<ComputeDataType>(1.0)
                          / sqrtInternal(channelVariance + epsilonCompute);

            // Apply normalization with scale and bias
            utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = utilities::staticCast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto xHat = (xVal - channelMean) * invVar;

                    y.setHostValue(
                        utilities::staticCast<YDataType>(
                            utilities::staticCast<ComputeDataType>(scale.getHostValue(0, cidx))
                                * xHat
                            + utilities::staticCast<ComputeDataType>(bias.getHostValue(0, cidx))),
                        fullIndices);
                });

            // Save mean and inverse variance for backward pass if provided
            if(mean != nullptr)
            {
                mean->setHostValue(
                    utilities::staticCast<MeanVarianceDataType>(channelMean), 0, cidx);
            }

            if(invVariance != nullptr)
            {
                invVariance->setHostValue(
                    utilities::staticCast<MeanVarianceDataType>(invVar), 0, cidx);
            }

            // Update running statistics if all required tensors are provided
            if(prevRunningMean != nullptr && prevRunningVariance != nullptr
               && nextRunningMean != nullptr && nextRunningVariance != nullptr)
            {
                auto one = utilities::staticCast<ComputeDataType>(1.0f);
                auto currentMean = utilities::staticCast<ComputeDataType>(
                    prevRunningMean->getHostValue(0, cidx));
                auto newMean = utilities::staticCast<MeanVarianceDataType>(
                    (one - momentumCompute) * currentMean + momentumCompute * channelMean);
                nextRunningMean->setHostValue(newMean, 0, cidx);

                auto currentVar = utilities::staticCast<ComputeDataType>(
                    prevRunningVariance->getHostValue(0, cidx));
                // Apply Bessel's correction for unbiased variance estimate
                ComputeDataType adjustedVariance = (nhw == one)
                                                       ? channelVariance
                                                       : utilities::staticCast<ComputeDataType>(
                                                             (nhw / (nhw - one)) * channelVariance);
                auto newVar = utilities::staticCast<MeanVarianceDataType>(
                    (one - momentumCompute) * currentVar + momentumCompute * adjustedVariance);
                nextRunningVariance->setHostValue(newVar, 0, cidx);
            }
        };

        // Build dimensions for parallel iteration
        auto nChannels = x.dims().at(1);
        std::vector<int64_t> parallelDims = {nChannels};

        auto parallelFunc = makeParallelTensorFunctor(batchnormFwdTrainingFunc, parallelDims);
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
    static void backward(const utilities::TensorBase<DyDataType>& dy,
                         const utilities::TensorBase<XDataType>& x,
                         const utilities::TensorBase<MeanVarianceDataType>& mean,
                         const utilities::TensorBase<MeanVarianceDataType>& invVariance,
                         const utilities::TensorBase<ScaleBiasDataType>& scale,
                         utilities::TensorBase<DxDataType>& dx,
                         utilities::TensorBase<ScaleBiasDataType>& dscale,
                         utilities::TensorBase<ScaleBiasDataType>& dbias)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm backward requires at least 2D tensor (batch and channel).");
        }

        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());
        //Cant cast directly from int64 to half or bloat16 so cast to float first.
        auto nhwF = utilities::staticCast<ComputeDataType>(elementsPerChannel);

        // Include batch dimension with spatial dimensions for iteration
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]}; // batch dimension
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormBwdFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto channelMean = utilities::staticCast<ComputeDataType>(mean.getHostValue(0, cidx));
            auto channelInvVariance = utilities::staticCast<ComputeDataType>(
                invVariance.getHostValue(0, cidx)); // 1 / sqrt(var + eps)
            auto channelScale = utilities::staticCast<ComputeDataType>(scale.getHostValue(0, cidx));

            // Calculate dot product for (x - mean) * channelInvVariance * dy and ∑ dy for this channel
            auto dotProduct = utilities::staticCast<ComputeDataType>(0.0);
            auto sumDy = utilities::staticCast<ComputeDataType>(0.0);

            utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = utilities::staticCast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto dyVal
                        = utilities::staticCast<ComputeDataType>(dy.getHostValue(fullIndices));

                    ComputeDataType xHat = (xVal - channelMean) * channelInvVariance;
                    // for half no += operator exists
                    dotProduct = dotProduct + (xHat * dyVal);
                    sumDy = sumDy + dyVal;
                });

            // Per channel:
            // - dscale = ∑ (xHat * dy)
            // - dbias = ∑ dy
            // - dx = scale * invVariance * (dy - mean(dy) - xHat * mean(dy * xHat))

            dscale.setHostValue(utilities::staticCast<ScaleBiasDataType>(dotProduct), 0, cidx);
            dbias.setHostValue(utilities::staticCast<ScaleBiasDataType>(sumDy), 0, cidx);

            auto meanDy = sumDy / nhwF;
            auto meanDyXhat = dotProduct / nhwF;
            auto scalarCoef = channelScale * channelInvVariance;

            utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);

                    auto xVal = utilities::staticCast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto dyVal
                        = utilities::staticCast<ComputeDataType>(dy.getHostValue(fullIndices));

                    auto xHat = (xVal - channelMean) * channelInvVariance;
                    auto dxVal = (dyVal - meanDy - xHat * meanDyXhat) * scalarCoef;

                    dx.setHostValue(utilities::staticCast<DxDataType>(dxVal), fullIndices);
                });
        };

        // Build dimensions for parallel iteration - only channels
        auto nChannels = x.dims().at(1);
        std::vector<int64_t> parallelDims = {nChannels};

        auto parallelFunc = makeParallelTensorFunctor(batchnormBwdFunc, parallelDims);
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

} // namespace hipdnn_sdk::test_utilities
