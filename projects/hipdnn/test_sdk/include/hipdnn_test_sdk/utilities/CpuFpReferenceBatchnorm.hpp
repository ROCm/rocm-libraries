// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <hipdnn_sdk/utilities/StaticCast.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceUtilities.hpp>
#include <numeric>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceBatchnorm
{
public:
    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void
        fwdInference(const hipdnn_sdk::utilities::TensorBase<XDataType>& x,
                     const hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
                     const hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& bias,
                     const hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>& estimatedMean,
                     const hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>& invVariance,
                     hipdnn_sdk::utilities::TensorBase<YDataType>& y)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm inference requires at least 2D tensor (batch and channel).");
        }

        auto batchnormFwdInferenceFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[1];
            auto mean = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                estimatedMean.getHostValue(0, cidx));
            auto invVarianceValue = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                invVariance.getHostValue(0, cidx));

            //There is some extra casting in here to deal with double -> float implicit casts.
            auto inVal
                = hipdnn_sdk::utilities::staticCast<ComputeDataType>(x.getHostValue(indices));
            ComputeDataType elemStd = inVal - mean;
            ComputeDataType inhat = elemStd * invVarianceValue;

            y.setHostValue(
                hipdnn_sdk::utilities::staticCast<YDataType>(
                    (hipdnn_sdk::utilities::staticCast<ComputeDataType>(scale.getHostValue(0, cidx))
                     * inhat)
                    + hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                        bias.getHostValue(0, cidx))),
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
    static void fwdTraining(
        const hipdnn_sdk::utilities::TensorBase<XDataType>& x,
        const hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
        const hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& bias,
        hipdnn_sdk::utilities::TensorBase<YDataType>& y,
        double epsilon,
        double momentum,
        hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>* mean = nullptr,
        hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>* invVariance = nullptr,
        const hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>* prevRunningMean = nullptr,
        const hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>* prevRunningVariance
        = nullptr,
        hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>* nextRunningMean = nullptr,
        hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>* nextRunningVariance = nullptr)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm training requires at least 2D tensor (batch and channel).");
        }

        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());

        auto nhw = hipdnn_sdk::utilities::staticCast<ComputeDataType>(elementsPerChannel);
        auto epsilonCompute = hipdnn_sdk::utilities::staticCast<ComputeDataType>(epsilon);
        auto momentumCompute = hipdnn_sdk::utilities::staticCast<ComputeDataType>(momentum);

        // Build dimensions for iteration: [batch, spatial...]
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]};
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormFwdTrainingFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto meanAccum = hipdnn_sdk::utilities::staticCast<ComputeDataType>(0.0);
            auto varianceAccum = hipdnn_sdk::utilities::staticCast<ComputeDataType>(0.0);

            // Calculate mean and variance for this channel
            hipdnn_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto inVal = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                        x.getHostValue(fullIndices));
                    meanAccum = meanAccum + inVal;
                    varianceAccum = varianceAccum + (inVal * inVal);
                });

            // NOTE: Different operation order from MIOpen produces expected FP differences.
            // Both implementations are correct; validated using RMS error tolerance
            ComputeDataType channelMean = meanAccum / nhw;
            ComputeDataType channelVariance = (varianceAccum / nhw) - (channelMean * channelMean);

            auto invVar = hipdnn_sdk::utilities::staticCast<ComputeDataType>(1.0)
                          / sqrtInternal(channelVariance + epsilonCompute);

            // Apply normalization with scale and bias
            hipdnn_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                        x.getHostValue(fullIndices));
                    auto xHat = (xVal - channelMean) * invVar;

                    y.setHostValue(hipdnn_sdk::utilities::staticCast<YDataType>(
                                       hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                                           scale.getHostValue(0, cidx))
                                           * xHat
                                       + hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                                           bias.getHostValue(0, cidx))),
                                   fullIndices);
                });

            // Save mean and inverse variance for backward pass if provided
            if(mean != nullptr)
            {
                mean->setHostValue(
                    hipdnn_sdk::utilities::staticCast<MeanVarianceDataType>(channelMean), 0, cidx);
            }

            if(invVariance != nullptr)
            {
                invVariance->setHostValue(
                    hipdnn_sdk::utilities::staticCast<MeanVarianceDataType>(invVar), 0, cidx);
            }

            // Update running statistics if all required tensors are provided
            if(prevRunningMean != nullptr && prevRunningVariance != nullptr
               && nextRunningMean != nullptr && nextRunningVariance != nullptr)
            {
                auto one = hipdnn_sdk::utilities::staticCast<ComputeDataType>(1.0f);
                auto currentMean = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                    prevRunningMean->getHostValue(0, cidx));
                auto newMean = hipdnn_sdk::utilities::staticCast<MeanVarianceDataType>(
                    (one - momentumCompute) * currentMean + momentumCompute * channelMean);
                nextRunningMean->setHostValue(newMean, 0, cidx);

                auto currentVar = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                    prevRunningVariance->getHostValue(0, cidx));
                // Apply Bessel's correction for unbiased variance estimate
                ComputeDataType adjustedVariance
                    = (nhw == one) ? channelVariance
                                   : hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                                         (nhw / (nhw - one)) * channelVariance);
                auto newVar = hipdnn_sdk::utilities::staticCast<MeanVarianceDataType>(
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
    static void backward(const hipdnn_sdk::utilities::TensorBase<DyDataType>& dy,
                         const hipdnn_sdk::utilities::TensorBase<XDataType>& x,
                         const hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
                         hipdnn_sdk::utilities::TensorBase<DxDataType>& dx,
                         hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& dscale,
                         hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& dbias,
                         const hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>* mean
                         = nullptr,
                         const hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>* invVariance
                         = nullptr,
                         double epsilon = 1e-5)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm backward requires at least 2D tensor (batch and channel).");
        }

        if((mean == nullptr) != (invVariance == nullptr))
        {
            throw std::runtime_error("Batchnorm backward requires both mean and invVariance to be "
                                     "provided, or neither.");
        }

        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());
        auto nhwF = hipdnn_sdk::utilities::staticCast<ComputeDataType>(elementsPerChannel);
        auto epsilonCompute = hipdnn_sdk::utilities::staticCast<ComputeDataType>(epsilon);

        // Include batch dimension with spatial dimensions for iteration
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]}; // batch dimension
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormBwdFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];

            ComputeDataType channelMean;
            ComputeDataType channelInvVariance;

            // Compute mean and invVariance if either are not provided
            if(mean == nullptr || invVariance == nullptr)
            {
                auto meanAccum = hipdnn_sdk::utilities::staticCast<ComputeDataType>(0.0);
                auto varianceAccum = hipdnn_sdk::utilities::staticCast<ComputeDataType>(0.0);

                // Calculate mean and variance for this channel
                hipdnn_sdk::utilities::iterateAlongDimensions(
                    batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                        auto fullIndices = hipdnn_sdk::utilities::buildTensorIndices(
                            batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                        auto inVal = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                            x.getHostValue(fullIndices));
                        meanAccum = meanAccum + inVal;
                        varianceAccum = varianceAccum + (inVal * inVal);
                    });

                channelMean = meanAccum / nhwF;

                ComputeDataType calculatedVariance
                    = (varianceAccum / nhwF) - (channelMean * channelMean);

                ComputeDataType denominator = sqrtInternal(calculatedVariance + epsilonCompute);
                channelInvVariance
                    = hipdnn_sdk::utilities::staticCast<ComputeDataType>(1.0) / denominator;
            }
            else
            {
                channelMean = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                    mean->getHostValue(0, cidx));
                channelInvVariance = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                    invVariance->getHostValue(0, cidx));
            }

            auto channelScale
                = hipdnn_sdk::utilities::staticCast<ComputeDataType>(scale.getHostValue(0, cidx));

            // Calculate dot product for (x - mean) * channelInvVariance * dy and ∑ dy for this channel
            auto dotProduct = hipdnn_sdk::utilities::staticCast<ComputeDataType>(0.0);
            auto sumDy = hipdnn_sdk::utilities::staticCast<ComputeDataType>(0.0);

            hipdnn_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                        x.getHostValue(fullIndices));
                    auto dyVal = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                        dy.getHostValue(fullIndices));

                    ComputeDataType xHat = (xVal - channelMean) * channelInvVariance;
                    // for half no += operator exists
                    dotProduct = dotProduct + (xHat * dyVal);
                    sumDy = sumDy + dyVal;
                });

            // Per channel:
            // - dscale = ∑ (xHat * dy)
            // - dbias = ∑ dy
            // - dx = scale * invVariance * (dy - mean(dy) - xHat * mean(dy * xHat))

            dscale.setHostValue(
                hipdnn_sdk::utilities::staticCast<ScaleBiasDataType>(dotProduct), 0, cidx);
            dbias.setHostValue(
                hipdnn_sdk::utilities::staticCast<ScaleBiasDataType>(sumDy), 0, cidx);

            auto meanDy = sumDy / nhwF;
            auto meanDyXhat = dotProduct / nhwF;
            auto scalarCoef = channelScale * channelInvVariance;

            hipdnn_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);

                    auto xVal = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                        x.getHostValue(fullIndices));
                    auto dyVal = hipdnn_sdk::utilities::staticCast<ComputeDataType>(
                        dy.getHostValue(fullIndices));

                    auto xHat = (xVal - channelMean) * channelInvVariance;
                    auto dxVal = (dyVal - meanDy - xHat * meanDyXhat) * scalarCoef;

                    dx.setHostValue(hipdnn_sdk::utilities::staticCast<DxDataType>(dxVal),
                                    fullIndices);
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

    // Backwards compatible overload for old function signature
    template <class DyDataType,
              class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType = ScaleBiasDataType,
              class DxDataType = XDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void backward(const hipdnn_sdk::utilities::TensorBase<DyDataType>& dy,
                         const hipdnn_sdk::utilities::TensorBase<XDataType>& x,
                         const hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>& mean,
                         const hipdnn_sdk::utilities::TensorBase<MeanVarianceDataType>& invVariance,
                         const hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
                         hipdnn_sdk::utilities::TensorBase<DxDataType>& dx,
                         hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& dscale,
                         hipdnn_sdk::utilities::TensorBase<ScaleBiasDataType>& dbias)
    {
        backward<DyDataType,
                 XDataType,
                 ScaleBiasDataType,
                 MeanVarianceDataType,
                 DxDataType,
                 ComputeDataType>(dy, x, scale, dx, dscale, dbias, &mean, &invVariance);
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

} // namespace hipdnn_test_sdk::utilities
