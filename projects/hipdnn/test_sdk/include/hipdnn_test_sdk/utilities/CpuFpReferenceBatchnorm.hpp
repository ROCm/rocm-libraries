// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <numeric>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceBatchnorm
{
public:
    // Uses pre-computed invVariance from fwdTraining (epsilon already baked in as 1/sqrt(var+eps)).
    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void fwdInference(
        const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& bias,
        const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& estimatedMean,
        const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& invVariance,
        hipdnn_data_sdk::utilities::TensorBase<YDataType>& y)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm inference requires at least 2D tensor (batch and channel).");
        }

        auto batchnormFwdInferenceFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[1];
            auto mean = static_cast<ComputeDataType>(estimatedMean.getHostValue(0, cidx));
            auto invVarianceValue = static_cast<ComputeDataType>(invVariance.getHostValue(0, cidx));

            //There is some extra casting in here to deal with double -> float implicit casts.
            auto inVal = static_cast<ComputeDataType>(x.getHostValue(indices));
            ComputeDataType elemStd = inVal - mean;
            ComputeDataType inhat = elemStd * invVarianceValue;

            y.setHostValue(static_cast<YDataType>(
                               (static_cast<ComputeDataType>(scale.getHostValue(0, cidx)) * inhat)
                               + static_cast<ComputeDataType>(bias.getHostValue(0, cidx))),
                           indices);
        };

        // Iterate all indices in parallel
        auto parallelFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            batchnormFwdInferenceFunc, x.dims());
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified(); // Mark y memory as modified on host
    }

    // Uses raw variance from fwdTraining runningVariance. Computes invVariance=1/sqrt(var+eps).
    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void fwdInferenceWithVariance(
        const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& bias,
        const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& estimatedMean,
        const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& variance,
        hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
        double epsilon = hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm inference with variance requires at least 2D tensor (batch and "
                "channel).");
        }

        auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

        auto batchnormFwdInferenceWithVarianceFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[1];
            auto mean = static_cast<ComputeDataType>(estimatedMean.getHostValue(0, cidx));
            auto varianceValue = static_cast<ComputeDataType>(variance.getHostValue(0, cidx));

            // Compute inv_variance = 1 / sqrt(variance + epsilon)
            auto invVarianceValue = static_cast<ComputeDataType>(1.0)
                                    / hipdnn_data_sdk::types::sqrt(varianceValue + epsilonCompute);

            //There is some extra casting in here to deal with double -> float implicit casts.
            auto inVal = static_cast<ComputeDataType>(x.getHostValue(indices));
            ComputeDataType elemStd = inVal - mean;
            ComputeDataType inhat = elemStd * invVarianceValue;

            y.setHostValue(static_cast<YDataType>(
                               (static_cast<ComputeDataType>(scale.getHostValue(0, cidx)) * inhat)
                               + static_cast<ComputeDataType>(bias.getHostValue(0, cidx))),
                           indices);
        };

        // Iterate all indices in parallel
        auto parallelFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            batchnormFwdInferenceWithVarianceFunc, x.dims());
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified(); // Mark y memory as modified on host
    }

    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType = ScaleBiasDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void fwdTraining(
        const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& bias,
        hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
        double epsilon,
        double momentum,
        hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>* mean = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>* invVariance = nullptr,
        const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>* prevRunningMean
        = nullptr,
        const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>* prevRunningVariance
        = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>* nextRunningMean = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>* nextRunningVariance = nullptr)
    {
        if(x.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm training requires at least 2D tensor (batch and channel).");
        }

        const int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());

        auto nhw = static_cast<ComputeDataType>(elementsPerChannel);
        auto epsilonCompute = static_cast<ComputeDataType>(epsilon);
        auto momentumCompute = static_cast<ComputeDataType>(momentum);

        // Build dimensions for iteration: [batch, spatial...]
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]};
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormFwdTrainingFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto meanAccum = static_cast<ComputeDataType>(0.0);
            auto varianceAccum = static_cast<ComputeDataType>(0.0);

            // Calculate mean and variance for this channel
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto inVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    meanAccum = meanAccum + inVal;
                    varianceAccum = varianceAccum + (inVal * inVal);
                });

            // NOTE: Different operation order from MIOpen produces expected FP differences.
            // Both implementations are correct; validated using RMS error tolerance
            ComputeDataType channelMean = meanAccum / nhw;
            ComputeDataType channelVariance = (varianceAccum / nhw) - (channelMean * channelMean);

            auto invVar = static_cast<ComputeDataType>(1.0)
                          / hipdnn_data_sdk::types::sqrt(channelVariance + epsilonCompute);

            // Apply normalization with scale and bias
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto xHat = (xVal - channelMean) * invVar;

                    y.setHostValue(
                        static_cast<YDataType>(
                            static_cast<ComputeDataType>(scale.getHostValue(0, cidx)) * xHat
                            + static_cast<ComputeDataType>(bias.getHostValue(0, cidx))),
                        fullIndices);
                });

            // Save mean and inverse variance for backward pass if provided
            if(mean != nullptr)
            {
                mean->setHostValue(static_cast<MeanVarianceDataType>(channelMean), 0, cidx);
            }

            if(invVariance != nullptr)
            {
                invVariance->setHostValue(static_cast<MeanVarianceDataType>(invVar), 0, cidx);
            }

            // Update running statistics if all required tensors are provided
            if(prevRunningMean != nullptr && prevRunningVariance != nullptr
               && nextRunningMean != nullptr && nextRunningVariance != nullptr)
            {
                auto one = static_cast<ComputeDataType>(1.0f);
                auto currentMean
                    = static_cast<ComputeDataType>(prevRunningMean->getHostValue(0, cidx));
                auto newMean = static_cast<MeanVarianceDataType>(
                    (one - momentumCompute) * currentMean + momentumCompute * channelMean);
                nextRunningMean->setHostValue(newMean, 0, cidx);

                auto currentVar
                    = static_cast<ComputeDataType>(prevRunningVariance->getHostValue(0, cidx));
                // Apply Bessel's correction for unbiased variance estimate
                ComputeDataType adjustedVariance
                    = (nhw == one)
                          ? channelVariance
                          : static_cast<ComputeDataType>((nhw / (nhw - one)) * channelVariance);
                auto newVar = static_cast<MeanVarianceDataType>(
                    (one - momentumCompute) * currentVar + momentumCompute * adjustedVariance);
                nextRunningVariance->setHostValue(newVar, 0, cidx);
            }
        };

        // Build dimensions for parallel iteration
        auto nChannels = x.dims().at(1);
        const std::vector<int64_t> parallelDims = {nChannels};

        auto parallelFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
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
    static void
        backward(const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& dy,
                 const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                 const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
                 hipdnn_data_sdk::utilities::TensorBase<DxDataType>& dx,
                 hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& dscale,
                 hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& dbias,
                 const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>* mean = nullptr,
                 const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>* invVariance
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

        const int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());
        auto nhwF = static_cast<ComputeDataType>(elementsPerChannel);
        auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

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
                auto meanAccum = static_cast<ComputeDataType>(0.0);
                auto varianceAccum = static_cast<ComputeDataType>(0.0);

                // Calculate mean and variance for this channel
                hipdnn_data_sdk::utilities::iterateAlongDimensions(
                    batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                        auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                            batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                        auto inVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                        meanAccum = meanAccum + inVal;
                        varianceAccum = varianceAccum + (inVal * inVal);
                    });

                channelMean = meanAccum / nhwF;

                ComputeDataType calculatedVariance
                    = (varianceAccum / nhwF) - (channelMean * channelMean);

                ComputeDataType denominator
                    = hipdnn_data_sdk::types::sqrt(calculatedVariance + epsilonCompute);
                channelInvVariance = static_cast<ComputeDataType>(1.0) / denominator;
            }
            else
            {
                channelMean = static_cast<ComputeDataType>(mean->getHostValue(0, cidx));
                channelInvVariance
                    = static_cast<ComputeDataType>(invVariance->getHostValue(0, cidx));
            }

            auto channelScale = static_cast<ComputeDataType>(scale.getHostValue(0, cidx));

            // Calculate dot product for (x - mean) * channelInvVariance * dy and ∑ dy for this channel
            auto dotProduct = static_cast<ComputeDataType>(0.0);
            auto sumDy = static_cast<ComputeDataType>(0.0);

            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto dyVal = static_cast<ComputeDataType>(dy.getHostValue(fullIndices));

                    ComputeDataType xHat = (xVal - channelMean) * channelInvVariance;
                    // for half no += operator exists
                    dotProduct = dotProduct + (xHat * dyVal);
                    sumDy = sumDy + dyVal;
                });

            // Per channel:
            // - dscale = ∑ (xHat * dy)
            // - dbias = ∑ dy
            // - dx = scale * invVariance * (dy - mean(dy) - xHat * mean(dy * xHat))

            dscale.setHostValue(static_cast<ScaleBiasDataType>(dotProduct), 0, cidx);
            dbias.setHostValue(static_cast<ScaleBiasDataType>(sumDy), 0, cidx);

            auto meanDy = sumDy / nhwF;
            auto meanDyXhat = dotProduct / nhwF;
            auto scalarCoef = channelScale * channelInvVariance;

            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                        batchSpatialIndices[0], cidx, batchSpatialIndices, 1);

                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto dyVal = static_cast<ComputeDataType>(dy.getHostValue(fullIndices));

                    auto xHat = (xVal - channelMean) * channelInvVariance;
                    auto dxVal = (dyVal - meanDy - xHat * meanDyXhat) * scalarCoef;

                    dx.setHostValue(static_cast<DxDataType>(dxVal), fullIndices);
                });
        };

        // Build dimensions for parallel iteration - only channels
        auto nChannels = x.dims().at(1);
        const std::vector<int64_t> parallelDims = {nChannels};

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(batchnormBwdFunc, parallelDims);
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
    static void
        backward(const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& dy,
                 const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                 const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& mean,
                 const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& invVariance,
                 const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
                 hipdnn_data_sdk::utilities::TensorBase<DxDataType>& dx,
                 hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& dscale,
                 hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& dbias)
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
};

} // namespace hipdnn_test_sdk::utilities
