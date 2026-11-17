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
inline std::vector<std::string> buildRMSNormFwdDefines(bool hasBias)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<InputDataType>::VALUE);
    defines.emplace_back(std::string("-DSCALE_TYPE=") + HipRtcTypeName<ScaleDataType>::VALUE);
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<OutputDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(localSize));
    defines.emplace_back(std::string("-DHAS_BIAS=") + std::to_string(hasBias ? 1 : 0));
    return defines;
}

template <typename GradOutputDataType,
          typename InputDataType,
          typename ScaleDataType,
          typename GradInputDataType,
          typename ComputeDataType,
          unsigned int localSize>
inline std::vector<std::string> buildRMSNormBwdDefines()
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DGRAD_OUTPUT_TYPE=")
                         + HipRtcTypeName<GradOutputDataType>::VALUE);
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<InputDataType>::VALUE);
    defines.emplace_back(std::string("-DSCALE_TYPE=") + HipRtcTypeName<ScaleDataType>::VALUE);
    defines.emplace_back(std::string("-DGRAD_INPUT_TYPE=")
                         + HipRtcTypeName<GradInputDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(localSize));
    return defines;
}

} // namespace detail

class GpuFpReferenceRMSNorm
{
private:
    struct TensorProps
    {
        std::string name;
        const std::vector<int64_t>* dims;
        const std::vector<int64_t>* strides;

        TensorProps(std::string n, const std::vector<int64_t>* d, const std::vector<int64_t>* s)
            : name(std::move(n))
            , dims(d)
            , strides(s)
        {
        }
    };

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
        validateFwdInput(input, scale, output, invRms, bias);

        auto defines = detail::buildRMSNormFwdDefines<InputDataType,
                                                      ScaleDataType,
                                                      OutputDataType,
                                                      ComputeDataType,
                                                      BLOCK_SIZE>(bias != nullptr);

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

    // --- Backward RMSNorm ---

    template <class GradOutputDataType,
              class InputDataType = GradOutputDataType,
              class ScaleDataType = GradOutputDataType,
              class GradInputDataType = GradOutputDataType,
              class ComputeDataType = double>
    static void bprop(hipdnn_data_sdk::utilities::TensorBase<GradOutputDataType>& gradOutput,
                      hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
                      hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                      hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>& invRms,
                      hipdnn_data_sdk::utilities::TensorBase<GradInputDataType>& gradInput,
                      hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& gradScale,
                      hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* gradBias = nullptr)
    {
        validateBwdInput(gradOutput, input, scale, invRms, gradInput, gradScale, gradBias);

        auto defines = detail::buildRMSNormBwdDefines<GradOutputDataType,
                                                      InputDataType,
                                                      ScaleDataType,
                                                      GradInputDataType,
                                                      ComputeDataType,
                                                      BLOCK_SIZE>();

        // Launch backward data kernel
        launchDgrad(gradOutput.memory().deviceData(),
                    input.memory().deviceData(),
                    input.dims(),
                    input.strides(),
                    scale.memory().deviceData(),
                    scale.dims(),
                    invRms.memory().deviceData(),
                    gradInput.memory().deviceData(),
                    defines);

        gradInput.memory().markDeviceModified();

        // Launch backward weight/bias kernel
        launchWgrad(gradOutput.memory().deviceData(),
                    input.memory().deviceData(),
                    input.dims(),
                    input.strides(),
                    scale.dims(),
                    invRms.memory().deviceData(),
                    gradScale.memory().deviceData(),
                    defines,
                    gradBias ? gradBias->memory().deviceData() : nullptr);

        gradScale.memory().markDeviceModified();
        if(gradBias != nullptr)
        {
            gradBias->memory().markDeviceModified();
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
                                             const std::vector<TensorProps>& otherIOTensorProps,
                                             const std::vector<TensorProps>& affineTensorProps,
                                             const std::vector<int64_t>* invRmsDims)
    {
        const auto nDims = inputDims.size();

        // Validate input tensor rank
        if(nDims < 3 || nDims > 5)
        {
            throw std::invalid_argument("RMSNorm requires input tensor rank to be 3, 4, or 5.");
        }

        // Validate all other I/O tensors have the same rank and shape as the input tensor
        for(const auto& [name, dims, strides] : otherIOTensorProps)
        {
            if(dims == nullptr)
            {
                continue;
            }
            if(dims->size() != nDims)
            {
                throw std::invalid_argument("RMSNorm requires " + name
                                            + " tensor rank to be equal to the input tensor rank.");
            }
            if(*dims != inputDims)
            {
                throw std::invalid_argument("RMSNorm requires " + name
                                            + " and input tensors to have the same shape.");
            }
        }

        // Validate all affine tensors have the same rank as the input tensor
        for(const auto& [name, dims, strides] : affineTensorProps)
        {
            if(dims == nullptr)
            {
                continue;
            }
            if(dims->size() != nDims)
            {
                throw std::invalid_argument("RMSNorm requires " + name
                                            + " tensor rank to be equal to the input tensor rank.");
            }
        }

        // Find the first available affine tensor to use as the shape reference for the rest
        const TensorProps* refAffineTensorProp = nullptr;
        for(const auto& prop : affineTensorProps)
        {
            if(prop.dims != nullptr)
            {
                refAffineTensorProp = &prop;
                break;
            }
        }

        // Validate all affine tensors have the same shape as the first affine tensor
        if(refAffineTensorProp != nullptr)
        {
            for(const auto& [name, dims, strides] : affineTensorProps)
            {
                if(dims == nullptr || dims == refAffineTensorProp->dims)
                {
                    continue;
                }
                if(*dims != *refAffineTensorProp->dims)
                {
                    throw std::invalid_argument("RMSNorm requires " + name + " and "
                                                + refAffineTensorProp->name
                                                + " tensors to have the same shape.");
                }
            }

            // Validate affine tensor dimensions have 1s in the leading dimensions
            const auto& affineDims = *refAffineTensorProp->dims;
            const auto& normalizeDim = getNormalizeDim(inputDims, affineDims);
            if(!std::all_of(affineDims.begin(),
                            affineDims.begin()
                                + static_cast<std::vector<int64_t>::difference_type>(normalizeDim),
                            [](int64_t d) { return d == 1; }))
            {
                throw std::invalid_argument("RMSNorm requires affine tensor dimensions to have 1s "
                                            "in the leading dimensions.");
            }

            // Validate invRms dimensions are compatible with the input and affine tensor dimensions
            if(invRmsDims != nullptr)
            {
                if(invRmsDims->size() != nDims)
                {
                    throw std::invalid_argument("RMSNorm requires invRms tensor rank to be equal "
                                                "to the input tensor rank.");
                }
                std::vector<int64_t> expectedInvRmsDims = inputDims;
                for(size_t i = 0; i < expectedInvRmsDims.size(); ++i)
                {
                    if(affineDims[i] != 1)
                    {
                        expectedInvRmsDims[i] = 1;
                    }
                }
                if(*invRmsDims != expectedInvRmsDims)
                {
                    throw std::invalid_argument("RMSNorm requires invRms tensor dimensions to be "
                                                "derived from the input and "
                                                "affine tensor dimensions.");
                }
            }
        }
    }

    static void validateConsistentLayouts(const std::vector<int64_t>& inputDims,
                                          const std::vector<int64_t>& inputStrides,
                                          const std::vector<TensorProps>& otherTensorProps)
    {
        using hipdnn_data_sdk::utilities::TensorLayout;

        const auto nDims = inputDims.size();
        const auto inputStrideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(inputStrides);

        // Validate reference tensor layout
        static const std::unordered_map<size_t, std::pair<TensorLayout, TensorLayout>>
            s_validLayouts = {{3, {TensorLayout::NCL, TensorLayout::NLC}},
                              {4, {TensorLayout::NCHW, TensorLayout::NHWC}},
                              {5, {TensorLayout::NCDHW, TensorLayout::NDHWC}}};

        const auto it = s_validLayouts.find(nDims);
        if(it == s_validLayouts.end())
        {
            throw std::invalid_argument("RMSNorm requires input tensor rank to be 3, 4, or 5.");
        }

        const auto& [channelFirst, channelLast] = it->second;
        if(inputStrideOrder != channelFirst.strideOrder
           && inputStrideOrder != channelLast.strideOrder)
        {
            throw std::invalid_argument("RMSNorm requires " + std::to_string(nDims)
                                        + "D input tensor to be in " + channelFirst.name + " or "
                                        + channelLast.name + " layout.");
        }

        // Validate all other tensor layouts are consistent with the reference tensor layout
        for(const auto& [name, dims, strides] : otherTensorProps)
        {
            if(dims == nullptr || strides == nullptr)
            {
                continue;
            }

            if(!hipdnn_data_sdk::utilities::isLayoutAgnostic(*dims)
               && hipdnn_data_sdk::utilities::extractStrideOrder(*strides) != inputStrideOrder)
            {
                throw std::invalid_argument(
                    "RMSNorm requires " + name
                    + " tensor layout to be consistent with input tensor layout.");
            }
        }
    }

    template <class InputDataType,
              class ScaleDataType = InputDataType,
              class OutputDataType = InputDataType,
              class ComputeDataType = double>
    static void
        validateFwdInput(const hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
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

        // Validate tensor dimensions
        std::vector<TensorProps> otherIOTensorProps;
        otherIOTensorProps.emplace_back("output", &outputDims, &output.strides());

        std::vector<TensorProps> affineTensorProps;
        affineTensorProps.emplace_back("scale", &scaleDims, &scale.strides());
        if(biasDims != nullptr)
        {
            affineTensorProps.emplace_back("bias", biasDims, &bias->strides());
        }

        validateConsistentDimensions(inputDims, otherIOTensorProps, affineTensorProps, invRmsDims);

        // Validate tensor layouts
        std::vector<TensorProps> otherTensorProps;
        otherTensorProps.emplace_back("scale", &scaleDims, &scale.strides());
        otherTensorProps.emplace_back("output", &outputDims, &output.strides());
        if(invRmsDims != nullptr)
        {
            otherTensorProps.emplace_back("invRms", invRmsDims, &invRms->strides());
        }
        if(biasDims != nullptr)
        {
            otherTensorProps.emplace_back("bias", biasDims, &bias->strides());
        }
        validateConsistentLayouts(inputDims, input.strides(), otherTensorProps);

        // Validate data types
        static_assert(
            IS_SUPPORTED_DATA_TYPE<InputDataType>,
            "RMSNorm forward supports only double, float, half, and bfloat16 input data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<OutputDataType>,
            "RMSNorm forward supports only double, float, half, and bfloat16 output data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ScaleDataType>,
            "RMSNorm forward supports only double, float, half, and bfloat16 scale data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ComputeDataType>,
            "RMSNorm forward supports only double, float, half, and bfloat16 compute data types.");
    }

    template <class GradOutputDataType,
              class InputDataType = GradOutputDataType,
              class ScaleDataType = GradOutputDataType,
              class GradInputDataType = GradOutputDataType,
              class ComputeDataType = double>
    static void validateBwdInput(
        const hipdnn_data_sdk::utilities::TensorBase<GradOutputDataType>& gradOutput,
        const hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
        const hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>& invRms,
        const hipdnn_data_sdk::utilities::TensorBase<GradInputDataType>& gradInput,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& gradScale,
        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* gradBias)
    {
        const auto& inputDims = input.dims();
        const auto& scaleDims = scale.dims();
        const auto& invRmsDims = invRms.dims();
        const auto& gradOutputDims = gradOutput.dims();
        const auto& gradInputDims = gradInput.dims();
        const auto& gradScaleDims = gradScale.dims();
        const auto* gradBiasDims = gradBias ? &gradBias->dims() : nullptr;

        // Validate tensor dimensions
        std::vector<TensorProps> otherIOTensorProps;
        otherIOTensorProps.emplace_back("gradOutput", &gradOutputDims, &gradOutput.strides());
        otherIOTensorProps.emplace_back("gradInput", &gradInputDims, &gradInput.strides());

        std::vector<TensorProps> affineTensorProps;
        affineTensorProps.emplace_back("scale", &scaleDims, &scale.strides());
        affineTensorProps.emplace_back("gradScale", &gradScaleDims, &gradScale.strides());
        if(gradBiasDims != nullptr)
        {
            affineTensorProps.emplace_back("gradBias", gradBiasDims, &gradBias->strides());
        }

        validateConsistentDimensions(inputDims, otherIOTensorProps, affineTensorProps, &invRmsDims);

        // Validate tensor layouts
        std::vector<TensorProps> otherTensorProps;
        otherTensorProps.emplace_back("scale", &scaleDims, &scale.strides());
        otherTensorProps.emplace_back("invRms", &invRmsDims, &invRms.strides());
        otherTensorProps.emplace_back("gradOutput", &gradOutputDims, &gradOutput.strides());
        otherTensorProps.emplace_back("gradInput", &gradInputDims, &gradInput.strides());
        otherTensorProps.emplace_back("gradScale", &gradScaleDims, &gradScale.strides());
        if(gradBiasDims != nullptr)
        {
            otherTensorProps.emplace_back("gradBias", gradBiasDims, &gradBias->strides());
        }
        validateConsistentLayouts(inputDims, input.strides(), otherTensorProps);

        // Validate data types
        static_assert(IS_SUPPORTED_DATA_TYPE<GradOutputDataType>,
                      "RMSNorm backward supports only double, float, half, and bfloat16 gradOutput "
                      "data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<InputDataType>,
            "RMSNorm backward supports only double, float, half, and bfloat16 input data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ScaleDataType>,
            "RMSNorm backward supports only double, float, half, and bfloat16 scale data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<GradInputDataType>,
                      "RMSNorm backward supports only double, float, half, and bfloat16 gradInput "
                      "data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ComputeDataType>,
            "RMSNorm backward supports only double, float, half, and bfloat16 compute data types.");
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
    // Channel size is not included in the outer size if there is a stride (channel-last layout)
    // since the channel dimension is interleaved with the inner dimensions and hence not
    // independent across the outer groups. The overall outer size is set during kernel launch
    // to outerSize * stride to account for the interleaved channel dimension.
    static int64_t
        getOuterSize(const std::vector<int64_t>& inputDims, size_t normalizeDim, int64_t stride)
    {
        int64_t outerSize = 1;
        for(size_t i = 0; i < normalizeDim; ++i)
        {
            // Add channel size only if there is no stride
            if(i == 1 && stride != 1)
            {
                continue;
            }
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

    static void launchDgrad(const void* gradOutputPtr,
                            const void* inputPtr,
                            const std::vector<int64_t>& inputDims,
                            const std::vector<int64_t>& inputStrides,
                            const void* scalePtr,
                            const std::vector<int64_t>& scaleDims,
                            const void* invRmsPtr,
                            void* gradInputPtr,
                            const std::vector<std::string>& defines);

    static void launchWgrad(const void* gradOutputPtr,
                            const void* inputPtr,
                            const std::vector<int64_t>& inputDims,
                            const std::vector<int64_t>& inputStrides,
                            const std::vector<int64_t>& scaleDims,
                            const void* invRmsPtr,
                            void* gradScalePtr,
                            const std::vector<std::string>& defines,
                            void* gradBiasPtr = nullptr);
};

} // namespace hipdnn_gpu_ref
