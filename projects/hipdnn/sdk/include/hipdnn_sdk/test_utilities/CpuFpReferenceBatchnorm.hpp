// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <numeric>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class InputDataType,
          class ScaleBiasDataType,
          class MeanVarianceDataType = ScaleBiasDataType>
class CpuFpReferenceBatchnormImpl
{
public:
    static bool isApplicable(const hipdnn_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_sdk::data_objects;

        // Support both BatchNorm inference and backward
        return (node.attributes_type() == NodeAttributes::BatchnormInferenceAttributes
                || node.attributes_type() != NodeAttributes::BatchnormBackwardAttributes);
    }

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

        auto nChannels = input.dims().at(1);
        int64_t elementsPerChannel = calculateElementsPerChannel(input.dims());

        std::vector<int64_t> channels(static_cast<size_t>(nChannels));
        std::iota(channels.begin(), channels.end(), 0);

        std::for_each(channels.begin(), channels.end(), [&](int64_t cidx) {
            auto mean = estimatedMean.getHostValue(0, cidx);
            auto variance = estimatedVariance.getHostValue(0, cidx);
            MeanVarianceDataType invVariance
                = static_cast<MeanVarianceDataType>(1.0f)
                  / sqrtInternal(variance + static_cast<MeanVarianceDataType>(epsilon));

            // process the batch per channel
            iterateChannelElements(
                input, cidx, elementsPerChannel, [&](const std::vector<int64_t>& indices) {
                    auto inVal = static_cast<MeanVarianceDataType>(input.getHostValue(indices));
                    MeanVarianceDataType elemStd = inVal - mean;
                    MeanVarianceDataType inhat = elemStd * invVariance;
                    output.setHostValue(
                        static_cast<InputDataType>(
                            (scale.getHostValue(0, cidx) * static_cast<ScaleBiasDataType>(inhat))
                            + bias.getHostValue(0, cidx)),
                        indices);
                });
        });

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

        auto nChannels = x.dims().at(1);
        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());

        auto nhw = static_cast<MeanVarianceDataType>(elementsPerChannel);

        std::vector<int64_t> channels(static_cast<size_t>(nChannels));
        std::iota(channels.begin(), channels.end(), 0);

        std::for_each(channels.begin(), channels.end(), [&](int64_t cidx) {
            MeanVarianceDataType meanAccum = 0.0;
            MeanVarianceDataType varianceAccum = 0.0;

            // Calculate mean and variance for this channel
            iterateChannelElements(
                x, cidx, elementsPerChannel, [&](const std::vector<int64_t>& indices) {
                    auto inVal = x.getHostValue(indices);
                    meanAccum += inVal;
                    varianceAccum += inVal * inVal;
                });

            MeanVarianceDataType channelMean = meanAccum /= nhw;
            MeanVarianceDataType channelVariance
                = (varianceAccum / nhw) - (channelMean * channelMean);

            auto invVar
                = static_cast<MeanVarianceDataType>(1.0) / sqrtInternal(channelVariance + epsilon);

            // Apply normalization with scale and bias
            iterateChannelElements(
                x, cidx, elementsPerChannel, [&](const std::vector<int64_t>& indices) {
                    auto xVal = static_cast<MeanVarianceDataType>(x.getHostValue(indices));
                    auto xHat = (xVal - channelMean) * invVar;

                    y.setHostValue(
                        static_cast<InputDataType>(scale.getHostValue(0, cidx)
                                                       * static_cast<ScaleBiasDataType>(xHat)
                                                   + bias.getHostValue(0, cidx)),
                        indices);
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
                auto adjustedVariance
                    = (nhw == one) ? channelVariance : (nhw / (nhw - one)) * channelVariance;
                auto newVar = (one - momentum) * currentVar + momentum * adjustedVariance;
                nextRunningVariance->setHostValue(newVar, 0, cidx);
            }
        });

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

        auto nChannels = x.dims().at(1);
        int64_t elementsPerChannel = calculateElementsPerChannel(x.dims());
        auto nhwF = static_cast<MeanVarianceDataType>(elementsPerChannel);

        std::vector<int64_t> channels(static_cast<size_t>(nChannels));
        std::iota(channels.begin(), channels.end(), 0);

        std::for_each(channels.begin(), channels.end(), [&](int64_t cidx) {
            auto channelMean = mean.getHostValue(0, cidx);
            auto channelInvVariance = invVariance.getHostValue(0, cidx); // 1 / sqrt(var + eps)
            auto channelScale = scale.getHostValue(0, cidx);

            // Calculate dot product for (x - mean) * channelInvVariance * dy and ∑ dy for this channel
            MeanVarianceDataType dotProduct = 0;
            MeanVarianceDataType sumDy = 0;

            iterateChannelElements(
                x, cidx, elementsPerChannel, [&](const std::vector<int64_t>& indices) {
                    auto xVal = static_cast<MeanVarianceDataType>(x.getHostValue(indices));
                    auto dyVal = static_cast<MeanVarianceDataType>(dy.getHostValue(indices));

                    MeanVarianceDataType xHat = (xVal - channelMean) * channelInvVariance;
                    dotProduct += xHat * dyVal;
                    sumDy += dyVal;
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

            iterateChannelElements(
                x, cidx, elementsPerChannel, [&](const std::vector<int64_t>& indices) {
                    auto xVal = static_cast<MeanVarianceDataType>(x.getHostValue(indices));
                    auto dyVal = static_cast<MeanVarianceDataType>(dy.getHostValue(indices));

                    MeanVarianceDataType xHat = (xVal - channelMean) * channelInvVariance;
                    MeanVarianceDataType dxVal = (dyVal - meanDy - xHat * meanDyXhat) * scalarCoef;

                    dx.setHostValue(static_cast<InputDataType>(dxVal), indices);
                });
        });

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

    // Utility method to iterate over all elements for a specific channel in an N-dimensional tensor
    static void iterateChannelElements(const TensorBase<InputDataType>& tensor,
                                       int64_t channelIdx,
                                       int64_t elementsPerChannel,
                                       const std::function<void(const std::vector<int64_t>&)>& func)
    {
        const auto& dims = tensor.dims();

        if(dims.size() < 2)
        {
            throw std::runtime_error("iterateChannelElements requires at least 2 dims.");
        }

        std::vector<int64_t> indices(dims.size(), 0);
        indices[1] = channelIdx;

        for(int64_t iter = 0; iter < elementsPerChannel; ++iter)
        {
            func(indices);

            // Since we iterate from end -> start, we need to use a signed int instead of size_t
            // Otherwise you will get an overflow.
            for(int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim)
            {
                // Skip channel dimension
                if(dim == 1)
                {
                    continue;
                }

                // Cast to size_t to avoid warnings.
                auto index = static_cast<size_t>(dim);
                indices[index]++;

                // No carry needed
                if(indices[index] < dims[index])
                {
                    break;
                }

                // Reset and carry inc to next dimension
                indices[index] = 0;
            }
        }
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
