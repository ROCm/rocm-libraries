// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
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

template <class InputDataType,
          class ScaleBiasDataType,
          class MeanVarianceDataType = ScaleBiasDataType>
class CpuFpReferenceBatchnormImpl
{
public:
    static void batchnormFwdInference(const TensorBase<InputDataType>& input,
                                      const TensorBase<ScaleBiasDataType>& scale,
                                      const TensorBase<ScaleBiasDataType>& bias,
                                      const TensorBase<MeanVarianceDataType>& estimatedMean,
                                      const TensorBase<MeanVarianceDataType>& estimatedVariance,
                                      TensorBase<InputDataType>& output,
                                      double epsilon)
    {
        if(input.dims().size() < 2)
        {
            throw std::runtime_error(
                "Batchnorm inference requires at least 2D tensor (batch and channel).");
        }

        auto batchnormFwdInferenceFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[1];
            auto mean = estimatedMean.getHostValue(0, cidx);
            auto variance = estimatedVariance.getHostValue(0, cidx);

            //There is some extra casting in here to deal with double -> float implicit casts.
            MeanVarianceDataType invVariance
                = static_cast<MeanVarianceDataType>(1.0f)
                  / sqrtInternal(variance
                                 + static_cast<MeanVarianceDataType>(static_cast<float>(epsilon)));

            auto inVal = static_cast<MeanVarianceDataType>(input.getHostValue(indices));
            MeanVarianceDataType elemStd = inVal - mean;
            MeanVarianceDataType inhat = elemStd * invVariance;

            output.setHostValue(static_cast<InputDataType>((scale.getHostValue(0, cidx)
                                                            * static_cast<ScaleBiasDataType>(inhat))
                                                           + bias.getHostValue(0, cidx)),
                                indices);
        };

        // Iterate all indices in parallel
        auto parallelFunc = hipdnn_sdk::test_utilities::makeParallelTensorFunctor(
            batchnormFwdInferenceFunc, input.dims());
        parallelFunc(std::thread::hardware_concurrency());

        output.memory().markHostModified(); // Mark output memory as modified on host
    }

    static void
        batchnormFwdTraining(const TensorBase<InputDataType>& x,
                             const TensorBase<ScaleBiasDataType>& scale,
                             const TensorBase<ScaleBiasDataType>& bias,
                             TensorBase<InputDataType>& y,
                             MeanVarianceDataType epsilon,
                             MeanVarianceDataType momentum,
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

        auto nhw = static_cast<MeanVarianceDataType>(static_cast<float>(elementsPerChannel));

        // Build dimensions for iteration: [batch, spatial...]
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]};
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormFwdTrainingFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto meanAccum = static_cast<MeanVarianceDataType>(0.0);
            auto varianceAccum = static_cast<MeanVarianceDataType>(0.0);

            // Calculate mean and variance for this channel
            iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices
                        = buildTensorIndices(batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto inVal = static_cast<MeanVarianceDataType>(x.getHostValue(fullIndices));
                    meanAccum = meanAccum + inVal;
                    varianceAccum = varianceAccum + (inVal * inVal);
                });

            MeanVarianceDataType channelMean = meanAccum = meanAccum / nhw;
            MeanVarianceDataType channelVariance
                = (varianceAccum / nhw) - (channelMean * channelMean);

            auto invVar
                = static_cast<MeanVarianceDataType>(1.0) / sqrtInternal(channelVariance + epsilon);

            // Apply normalization with scale and bias
            iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices
                        = buildTensorIndices(batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = static_cast<MeanVarianceDataType>(x.getHostValue(fullIndices));
                    auto xHat = (xVal - channelMean) * invVar;

                    y.setHostValue(
                        static_cast<InputDataType>(scale.getHostValue(0, cidx)
                                                       * static_cast<ScaleBiasDataType>(xHat)
                                                   + bias.getHostValue(0, cidx)),
                        fullIndices);
                });

            // Save mean and inverse variance for backward pass if provided
            if(mean != nullptr)
            {
                mean->setHostValue(channelMean, 0, cidx);
            }

            if(invVariance != nullptr)
            {
                invVariance->setHostValue(invVar, 0, cidx);
            }

            // Update running statistics if all required tensors are provided
            if(prevRunningMean != nullptr && prevRunningVariance != nullptr
               && nextRunningMean != nullptr && nextRunningVariance != nullptr)
            {
                auto one = static_cast<MeanVarianceDataType>(1.0f);
                auto currentMean = prevRunningMean->getHostValue(0, cidx);
                auto newMean = (one - momentum) * currentMean + momentum * channelMean;
                nextRunningMean->setHostValue(newMean, 0, cidx);

                auto currentVar = prevRunningVariance->getHostValue(0, cidx);
                // Apply Bessel's correction for unbiased variance estimate
                auto adjustedVariance = (nhw == one) ? channelVariance
                                                     : static_cast<MeanVarianceDataType>(
                                                           (nhw / (nhw - one)) * channelVariance);
                auto newVar = (one - momentum) * currentVar + momentum * adjustedVariance;
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

    static void batchnormBwd(const TensorBase<InputDataType>& dy,
                             const TensorBase<InputDataType>& x,
                             const TensorBase<MeanVarianceDataType>& mean,
                             const TensorBase<MeanVarianceDataType>& invVariance,
                             const TensorBase<ScaleBiasDataType>& scale,
                             TensorBase<InputDataType>& dx,
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
        auto nhwF = static_cast<MeanVarianceDataType>(static_cast<float>(elementsPerChannel));

        // Include batch dimension with spatial dimensions for iteration
        std::vector<int64_t> batchAndSpatial = {x.dims()[0]}; // batch dimension
        batchAndSpatial.insert(batchAndSpatial.end(), x.dims().begin() + 2, x.dims().end());

        auto batchnormBwdFunc = [&](const std::vector<int64_t>& indices) {
            auto cidx = indices[0];
            auto channelMean = mean.getHostValue(0, cidx);
            auto channelInvVariance = invVariance.getHostValue(0, cidx); // 1 / sqrt(var + eps)
            auto channelScale = scale.getHostValue(0, cidx);

            // Calculate dot product for (x - mean) * channelInvVariance * dy and ∑ dy for this channel
            auto dotProduct = static_cast<MeanVarianceDataType>(0.0);
            auto sumDy = static_cast<MeanVarianceDataType>(0.0);

            iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices
                        = buildTensorIndices(batchSpatialIndices[0], cidx, batchSpatialIndices, 1);
                    auto xVal = static_cast<MeanVarianceDataType>(x.getHostValue(fullIndices));
                    auto dyVal = static_cast<MeanVarianceDataType>(dy.getHostValue(fullIndices));

                    MeanVarianceDataType xHat = (xVal - channelMean) * channelInvVariance;
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

            MeanVarianceDataType meanDy = sumDy / nhwF;
            MeanVarianceDataType meanDyXhat = dotProduct / nhwF;
            MeanVarianceDataType scalarCoef
                = static_cast<MeanVarianceDataType>(channelScale) * channelInvVariance;

            iterateAlongDimensions(
                batchAndSpatial, [&](const std::vector<int64_t>& batchSpatialIndices) {
                    auto fullIndices
                        = buildTensorIndices(batchSpatialIndices[0], cidx, batchSpatialIndices, 1);

                    auto xVal = static_cast<MeanVarianceDataType>(x.getHostValue(fullIndices));
                    auto dyVal = static_cast<MeanVarianceDataType>(dy.getHostValue(fullIndices));

                    MeanVarianceDataType xHat = (xVal - channelMean) * channelInvVariance;
                    MeanVarianceDataType dxVal = (dyVal - meanDy - xHat * meanDyXhat) * scalarCoef;

                    dx.setHostValue(static_cast<InputDataType>(dxVal), fullIndices);
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
