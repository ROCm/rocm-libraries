// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

template <typename XDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename YDataType,
          typename ComputeDataType>
inline std::vector<std::string> buildBatchnormFwdInfDefines()
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<XDataType>::VALUE);
    defines.emplace_back(std::string("-DSCALE_BIAS_TYPE=")
                         + HipRtcTypeName<ScaleBiasDataType>::VALUE);
    defines.emplace_back(std::string("-DMEAN_VAR_TYPE=")
                         + HipRtcTypeName<MeanVarianceDataType>::VALUE);
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<YDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    return defines;
}

} // namespace detail

class GpuFpReferenceBatchnorm
{
private:
    struct TensorProps
    {
        std::string name;
        const std::vector<int64_t>& dims;
        const std::vector<int64_t>& strides;

        TensorProps(std::string n, const std::vector<int64_t>& d, const std::vector<int64_t>& s)
            : name(std::move(n))
            , dims(d)
            , strides(s)
        {
        }
    };

public:
    template <class XDataType,
              class ScaleBiasDataType,
              class MeanVarianceDataType,
              class YDataType,
              class ComputeDataType = MeanVarianceDataType>
    static void
        fwdInference(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                     hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
                     hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& bias,
                     hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& estimatedMean,
                     hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& invVariance,
                     hipdnn_data_sdk::utilities::TensorBase<YDataType>& y)
    {
        validateFwdInfInput<XDataType,
                            ScaleBiasDataType,
                            MeanVarianceDataType,
                            YDataType,
                            ComputeDataType>(x, scale, bias, estimatedMean, invVariance, y);

        auto defines = detail::buildBatchnormFwdInfDefines<XDataType,
                                                           ScaleBiasDataType,
                                                           MeanVarianceDataType,
                                                           YDataType,
                                                           ComputeDataType>();

        launchFwdInf(x.memory().deviceData(),
                     x.dims(),
                     x.strides(),
                     scale.memory().deviceData(),
                     bias.memory().deviceData(),
                     estimatedMean.memory().deviceData(),
                     invVariance.memory().deviceData(),
                     y.memory().deviceData(),
                     defines);

        y.memory().markDeviceModified();
    }

private:
    // --- Validators ---

    template <class T>
    static constexpr bool IS_SUPPORTED_DATA_TYPE
        = std::is_same_v<T, double> || std::is_same_v<T, float>
          || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

    static void validateConsistentDimensions(const TensorProps& inputTensorProps,
                                             const TensorProps& outputTensorProps,
                                             const std::vector<TensorProps>& affineTensorProps)
    {
        const auto inputDims = inputTensorProps.dims;
        if(inputDims.size() < 3 || inputDims.size() > 5)
        {
            throw std::invalid_argument("Batchnorm requires input tensor rank to be 3, 4, or 5.");
        }

        if(std::find(inputDims.begin(), inputDims.end(), int64_t(0)) != inputDims.end())
        {
            throw std::invalid_argument("Batchnorm requires tensors to not have a zero dimension.");
        }

        const auto outputDims = outputTensorProps.dims;
        if(outputDims.size() != inputDims.size())
        {
            throw std::invalid_argument(
                "Batchnorm requires output tensor rank to be equal to the input tensor rank.");
        }
        if(outputDims != inputDims)
        {
            throw std::invalid_argument(
                "Batchnorm requires output and input tensors to have the same shape.");
        }

        // Checks if the affine tensor shape is valid for broadcasting to the IO tensor shape as per
        //  with the constraint that the channel dimension must match.
        const auto isValidAffineShapeForIo = [&](const std::vector<int64_t>& affineDims) {
            if(affineDims.size() < 2)
            {
                return false;
            }

            if(affineDims[0] != 1)
            {
                return false; // batch must be 1
            }

            const int64_t numChannels = inputDims[1];
            if(numChannels != affineDims[1])
            {
                return false; // channel dim must match
            }

            // Check all spatial dimensions are 1
            for(size_t i = 2; i < affineDims.size(); ++i)
            {
                if(affineDims[i] != 1)
                {
                    return false;
                }
            }

            return true;
        };

        // Validate all affine tensors are channel only
        for(const auto& [name, dims, strides] : affineTensorProps)
        {
            if(!isValidAffineShapeForIo(dims))
            {
                throw std::invalid_argument(
                    "Batchnorm requires " + name
                    + " tensor to be channel-only broadcastable to input tensor.");
            }
        }
    }

    static void validateConsistentLayouts(const TensorProps& inputTensorProps,
                                          const std::vector<TensorProps>& otherTensorProps)
    {
        using hipdnn_data_sdk::utilities::TensorLayout;

        auto inputDims = inputTensorProps.dims;
        auto inputStrides = inputTensorProps.strides;
        const auto nDims = inputDims.size();
        const auto inputStrideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(inputStrides);

        const bool packed = hipdnn_data_sdk::utilities::isTensorPacked(inputDims, inputStrides);
        if(!packed)
        {
            throw std::invalid_argument("Batchnorm requires input tensor strides to be packed");
        }

        // Validate reference tensor layout
        static const std::unordered_map<size_t, std::pair<TensorLayout, TensorLayout>>
            s_validLayouts = {{3, {TensorLayout::NCL, TensorLayout::NLC}},
                              {4, {TensorLayout::NCHW, TensorLayout::NHWC}},
                              {5, {TensorLayout::NCDHW, TensorLayout::NDHWC}}};

        const auto it = s_validLayouts.find(nDims);
        if(it == s_validLayouts.end())
        {
            throw std::invalid_argument("Batchnorm requires input tensor rank to be 3, 4, or 5.");
        }

        const auto& [channelFirst, channelLast] = it->second;
        if(inputStrideOrder != channelFirst.strideOrder
           && inputStrideOrder != channelLast.strideOrder)
        {
            throw std::invalid_argument("Batchnorm requires " + std::to_string(nDims)
                                        + "D input tensor to be in " + channelFirst.name + " or "
                                        + channelLast.name + " layout.");
        }

        // Validate all other tensor layouts are consistent with the reference tensor layout
        for(const auto& [name, dims, strides] : otherTensorProps)
        {
            if(!hipdnn_data_sdk::utilities::isTensorPacked(dims, strides))
            {
                throw std::invalid_argument("Batchnorm requires " + name
                                            + " tensor strides to be packed");
            }

            if(!hipdnn_data_sdk::utilities::isLayoutAgnostic(dims)
               && hipdnn_data_sdk::utilities::extractStrideOrder(strides) != inputStrideOrder)
            {
                throw std::invalid_argument(
                    "Batchnorm requires " + name
                    + " tensor layout to be consistent with input tensor layout.");
            }
        }
    }

    template <typename InputDataType,
              typename ScaleBiasDataType,
              typename MeanVarianceDataType,
              typename OutputDataType,
              typename ComputeDataType>
    static void validateFwdInfInput(
        const hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& bias,
        const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& estMean,
        const hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>& invVar,
        const hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output)
    {
        const auto& inputDims = input.dims();
        const auto& scaleDims = scale.dims();
        const auto& biasDims = bias.dims();
        const auto& estMeanDims = estMean.dims();
        const auto& invVarDims = invVar.dims();
        const auto& outputDims = output.dims();

        const TensorProps inputTensorProps("input", inputDims, input.strides());
        TensorProps outputTensorProps("output", outputDims, output.strides());
        std::vector<TensorProps> affineTensorProps;
        affineTensorProps.emplace_back("scale", scaleDims, scale.strides());
        affineTensorProps.emplace_back("bias", biasDims, bias.strides());
        affineTensorProps.emplace_back("estimatedMean", estMeanDims, estMean.strides());
        affineTensorProps.emplace_back("invVariance", invVarDims, invVar.strides());

        validateConsistentDimensions(inputTensorProps, outputTensorProps, affineTensorProps);
        affineTensorProps.push_back(std::move(outputTensorProps));
        validateConsistentLayouts(inputTensorProps, affineTensorProps);

        // Validate data types
        static_assert(
            IS_SUPPORTED_DATA_TYPE<InputDataType>,
            "Batchnorm forward supports only double, float, half, and bfloat16 input data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<OutputDataType>,
            "Batchnorm forward supports only double, float, half, and bfloat16 output data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<ScaleBiasDataType>,
                      "Batchnorm forward supports only double, float, half, and bfloat16 "
                      "scale/bias data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<MeanVarianceDataType>,
                      "Batchnorm forward supports only double, float, half, and bfloat16 "
                      "estMean/invVar data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<ComputeDataType>,
                      "Batchnorm forward supports only double, float, half, and bfloat16 compute "
                      "data types.");
    }

    // --- Helpers ---

    static bool isChannelLastLayout(const std::vector<int64_t>& strides)
    {
        if(strides.size() < 3)
        {
            throw std::invalid_argument(
                "Batchnorm forward requires tensor rank to be at least 3 for layout validation.");
        }

        const auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(strides);
        return strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NLC.strideOrder
               || strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NHWC.strideOrder
               || strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NDHWC.strideOrder;
    }

    // --- Kernel launchers (defined in GpuFpReferenceBatchnorm.cpp) ---
    static void launchFwdInf(const void* inputPtr,
                             const std::vector<int64_t>& inputDims,
                             const std::vector<int64_t>& inputStrides,
                             const void* scalePtr,
                             const void* biasPtr,
                             const void* estMeanPtr,
                             const void* invVarPtr,
                             void* outputPtr,
                             std::vector<std::string>& defines);
};

} // namespace hipdnn_gpu_ref
