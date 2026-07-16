// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_gpu_ref/ShallowGpuTensor.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_gpu_ref/detail/HipRtcTypeName.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

template <typename InputDataType,
          typename ScaleDataType,
          typename OutputDataType,
          typename ComputeDataType,
          unsigned int localSize>
inline std::vector<std::string> buildRMSNormDefines()
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<InputDataType>::VALUE);
    defines.emplace_back(std::string("-DSCALE_TYPE=") + HipRtcTypeName<ScaleDataType>::VALUE);
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<OutputDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(localSize));
    return defines;
}

} // namespace detail

class GpuFpReferenceRMSNorm
{
public:
    static constexpr unsigned int BLOCK_SIZE = 256;

    // --- Forward RMSNorm ---

    template <class InputDataType,
              class ScaleDataType = InputDataType,
              class OutputDataType = InputDataType,
              class ComputeDataType = double>
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
                      hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                      hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output,
                      double epsilon = 1e-5,
                      hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* invRms = nullptr,
                      hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* bias = nullptr)
    {
        validateInput(input, scale, output, invRms, bias);

        auto defines = detail::buildRMSNormDefines<InputDataType,
                                                   ScaleDataType,
                                                   OutputDataType,
                                                   ComputeDataType,
                                                   BLOCK_SIZE>();

        launchFprop(input.memory().deviceData(),
                    input.dims(),
                    input.strides(),
                    scale.memory().deviceData(),
                    scale.dims(),
                    output.memory().deviceData(),
                    defines,
                    invRms ? invRms->memory().deviceData() : nullptr,
                    bias ? bias->memory().deviceData() : nullptr,
                    epsilon);

        output.memory().markDeviceModified();

        if(invRms != nullptr)
        {
            invRms->memory().markDeviceModified();
        }
    }

private:
    // --- Validators ---

    template <class T>
    static constexpr bool IS_SUPPORTED_DATA_TYPE
        = std::is_same_v<T, double> || std::is_same_v<T, float>
          || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

    static void validateConsistentDimensions(const std::vector<int64_t>& inputDims,
                                             const std::vector<int64_t>& scaleDims,
                                             const std::vector<int64_t>& outputDims,
                                             const std::vector<int64_t>* invRmsDims,
                                             const std::vector<int64_t>* biasDims)
    {
        // Validate the tensor ranks
        const auto& nDims = inputDims.size();

        if(nDims < 3 || nDims > 5)
        {
            throw std::invalid_argument(
                "RMSNorm forward requires input tensor rank to be 3, 4, or 5.");
        }

        if(scaleDims.size() != nDims)
        {
            throw std::invalid_argument(
                "RMSNorm forward requires scale tensor rank to be equal to the input tensor rank.");
        }

        if(outputDims.size() != nDims)
        {
            throw std::invalid_argument("RMSNorm forward requires output tensor rank to be equal "
                                        "to the input tensor rank.");
        }

        if(invRmsDims != nullptr && invRmsDims->size() != nDims)
        {
            throw std::invalid_argument("RMSNorm forward requires invRms tensor rank to be equal "
                                        "to the input tensor rank.");
        }

        if(biasDims != nullptr && biasDims->size() != nDims)
        {
            throw std::invalid_argument(
                "RMSNorm forward requires bias tensor rank to be equal to the input tensor rank.");
        }

        // Validate the compatibility of the input and output dimensions
        if(inputDims != outputDims)
        {
            throw std::invalid_argument(
                "RMSNorm forward requires input and output tensors to have the same shape.");
        }

        // Validate the compatibility of the scale and bias dimensions with the input dimensions
        if(biasDims != nullptr && scaleDims != *biasDims)
        {
            throw std::invalid_argument(
                "RMSNorm forward requires scale and bias tensors to have the same shape.");
        }

        const auto& normalizeDim = getNormalizeDim(inputDims, scaleDims);
        if(!std::all_of(scaleDims.begin(),
                        scaleDims.begin()
                            + static_cast<std::vector<int64_t>::difference_type>(normalizeDim),
                        [](int64_t d) { return d == 1; }))
        {
            throw std::invalid_argument("RMSNorm forward requires affine tensor dimensions to have "
                                        "1s in the leading dimensions.");
        }

        // Validate invRms dimensions are compatible with input and scale tensors
        if(invRmsDims != nullptr)
        {
            std::vector<int64_t> expectedInvRmsDims = inputDims;
            for(size_t i = 0; i < expectedInvRmsDims.size(); ++i)
            {
                if(scaleDims[i] != 1)
                {
                    expectedInvRmsDims[i] = 1;
                }
            }
            if(*invRmsDims != expectedInvRmsDims)
            {
                throw std::invalid_argument(
                    "RMSNorm forward requires invRms tensor dimensions to be derived from the "
                    "input and scale tensor dimensions.");
            }
        }
    }

    template <class InputDataType, class ScaleDataType, class OutputDataType, class ComputeDataType>
    static void validateConsistentLayouts(
        const hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
        const hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output,
        const hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* invRms,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* bias)
    {
        using hipdnn_data_sdk::utilities::TensorLayout;

        const auto& scaleDims = scale.dims();
        const auto& outputDims = output.dims();
        const auto* invRmsDims = invRms ? &invRms->dims() : nullptr;
        const auto* biasDims = bias ? &bias->dims() : nullptr;

        const auto& inputStrides = input.strides();
        const auto& scaleStrides = scale.strides();
        const auto& outputStrides = output.strides();
        const auto* invRmsStrides = invRms ? &invRms->strides() : nullptr;
        const auto* biasStrides = bias ? &bias->strides() : nullptr;

        const auto nDims = input.dims().size();
        const auto inputStrideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(inputStrides);

        // Validate input tensor layout
        static const std::unordered_map<size_t, std::pair<TensorLayout, TensorLayout>>
            s_validLayouts = {{3, {TensorLayout::NCL, TensorLayout::NLC}},
                              {4, {TensorLayout::NCHW, TensorLayout::NHWC}},
                              {5, {TensorLayout::NCDHW, TensorLayout::NDHWC}}};

        const auto layoutIt = s_validLayouts.find(nDims);
        if(layoutIt == s_validLayouts.end())
        {
            throw std::invalid_argument(
                "RMSNorm forward requires input tensor rank to be 3, 4, or 5.");
        }

        const auto& [channelFirst, channelLast] = layoutIt->second;
        if(inputStrideOrder != channelFirst.strideOrder
           && inputStrideOrder != channelLast.strideOrder)
        {
            throw std::invalid_argument("RMSNorm forward requires " + std::to_string(nDims)
                                        + "D input tensor to be in " + channelFirst.name + " or "
                                        + channelLast.name + " layout.");
        }

        // Validate all other layouts are consistent with input layout
        const auto validateTensorLayout = [&inputStrideOrder](const std::vector<int64_t>& dims,
                                                              const std::vector<int64_t>& strides,
                                                              const std::string& name) {
            if(!hipdnn_data_sdk::utilities::isLayoutAgnostic(dims)
               && hipdnn_data_sdk::utilities::extractStrideOrder(strides) != inputStrideOrder)
            {
                throw std::invalid_argument("RMSNorm forward requires " + name
                                            + " tensor layout to be consistent with input "
                                              "tensor layout.");
            }
        };
        validateTensorLayout(outputDims, outputStrides, "output");
        validateTensorLayout(scaleDims, scaleStrides, "scale");
        if(biasStrides != nullptr)
        {
            validateTensorLayout(*biasDims, *biasStrides, "bias");
        }
        if(invRmsStrides != nullptr)
        {
            validateTensorLayout(*invRmsDims, *invRmsStrides, "invRms");
        }
    }

    template <class InputDataType,
              class ScaleDataType = InputDataType,
              class OutputDataType = InputDataType,
              class ComputeDataType = double>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
                              const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                              const hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output,
                              const hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* invRms,
                              const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* bias)
    {
        const auto& inputDims = input.dims();
        const auto& scaleDims = scale.dims();
        const auto& outputDims = output.dims();
        const auto* invRmsDims = invRms ? &invRms->dims() : nullptr;
        const auto* biasDims = bias ? &bias->dims() : nullptr;

        // Validate tensor dimensions and layouts
        validateConsistentDimensions(inputDims, scaleDims, outputDims, invRmsDims, biasDims);
        validateConsistentLayouts(input, scale, output, invRms, bias);

        // Validate data types
        static_assert(IS_SUPPORTED_DATA_TYPE<InputDataType>,
                      "RMSNorm forward supports only float, half, and bfloat16 input data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<OutputDataType>,
                      "RMSNorm forward supports only float, half, and bfloat16 output data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<ScaleDataType>,
                      "RMSNorm forward supports only float, half, and bfloat16 scale data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ComputeDataType>,
            "RMSNorm forward supports only float, half, and bfloat16 compute data types.");
    }

    // --- Helpers ---

    static bool isChannelLastLayout(const std::vector<int64_t>& strides)
    {
        if(strides.size() < 3)
        {
            throw std::invalid_argument(
                "RMSNorm forward requires tensor rank to be at least 3 for layout validation.");
        }

        const auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(strides);
        return strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NLC.strideOrder
               || strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NHWC.strideOrder
               || strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NDHWC.strideOrder;
    }

    // normalizeDim marks the split point between outer dimensions and the inner
    // dimensions over which normalization statistics (invVariance) are computed.
    // Dimensions [0, ..., normalizeDim-1] are the outer dimensions, and dimensions
    // [normalizeDim, ..., nDims-1] are the inner dimensions over which normalization
    // is performed. It is found by matching the input and scale dimensions, starting
    // from the right, and counting the number of trailing dimensions that match, then
    // subtracting that count from the total number of dimensions in scaleDims.
    // If all dimensions match, normalizeDim is set to 1, since there must be
    // at least one normalization axis.
    // Examples: 1. inputDims = [2, 4, 8, 8] and scaleDims = [1, 4, 8, 8], then normalizeDim = 4 - 3 = 1.
    //           2. inputDims = [2, 4, 8, 8] and scaleDims = [1, 1, 8, 8], then normalizeDim = 4 - 2 = 2.
    static size_t getNormalizeDim(const std::vector<int64_t>& inputDims,
                                  const std::vector<int64_t>& scaleDims)
    {
        // Find number of trailing dims where scaleDims[i] == inputDims[i]
        const auto [scaleMismatch, _] = std::mismatch(
            scaleDims.rbegin(), scaleDims.rend(), inputDims.rbegin(), inputDims.rend());
        const auto matchCount
            = static_cast<size_t>(std::distance(scaleDims.rbegin(), scaleMismatch));

        // Scale must have at least one normalization axis, so account for the
        // case where input has a single batch and scale matches exactly.
        const auto normalizeDim
            = (matchCount == scaleDims.size()) ? 1 : scaleDims.size() - matchCount;
        return static_cast<size_t>(normalizeDim);
    }

    // Computes the number of elements in the outer dimensions [0, ..., normalizeDim-1]
    // of the input tensor i.e. number of independent groups that will be normalized separately.
    static int64_t getOuterSize(const std::vector<int64_t>& inputDims, size_t normalizeDim)
    {
        int64_t outerSize = 1;
        for(size_t i = 0; i < normalizeDim; ++i)
        {
            outerSize *= inputDims[i];
        }
        return outerSize;
    }

    // Computes the number of elements in the inner dimensions [normalizeDim, ..., nDims-1]
    // of the input tensor i.e. number of elements over which normalization is performed.
    static int64_t getInnerSize(const std::vector<int64_t>& inputDims, size_t normalizeDim)
    {
        int64_t innerSize = 1;
        for(size_t i = normalizeDim; i < inputDims.size(); ++i)
        {
            innerSize *= inputDims[i];
        }
        return innerSize;
    }

    // Computes the memory stride separating consecutive elements in the trailing
    // dimensions. The memory stride only matters when normalizeDim > 1 and the layout is
    // channel-last, since the channel dim is then interleaved between trailing dims rather
    // than being contiguous and hence the stride should be the size of the channel dimension
    // to skip over the channel dim when iterating over the trailing elements.
    static int64_t getStride(const std::vector<int64_t>& inputDims,
                             const std::vector<int64_t>& inputStrides,
                             size_t normalizeDim)
    {
        int64_t stride = 1;
        auto isLayoutNHWC = isChannelLastLayout(inputStrides);
        if(normalizeDim > 1 && isLayoutNHWC)
        {
            stride = inputDims[1];
        }
        return stride;
    }

    // --- Kernel launchers (defined in GpuFpReferenceRMSNorm.cpp) ---

    static void launchFprop(const void* inputPtr,
                            const std::vector<int64_t>& inputDims,
                            const std::vector<int64_t>& inputStrides,
                            const void* scalePtr,
                            const std::vector<int64_t>& scaleDims,
                            void* outputPtr,
                            const std::vector<std::string>& defines,
                            void* invRmsPtr = nullptr,
                            const void* biasPtr = nullptr,
                            double epsilon = 1e-5);
};

} // namespace hipdnn_gpu_ref
