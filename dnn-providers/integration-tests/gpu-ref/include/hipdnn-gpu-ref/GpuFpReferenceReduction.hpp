// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

template <typename InputDataType,
          typename OutputDataType,
          typename ComputeDataType,
          unsigned int localSize>
inline std::vector<std::string> buildReductionDefines(unsigned int numDims)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<InputDataType>::VALUE);
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<OutputDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DNUM_DIMS=") + std::to_string(numDims));
    defines.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(localSize));
    return defines;
}

} // namespace detail

class GpuFpReferenceReduction
{
public:
    static constexpr unsigned int BLOCK_SIZE = 256;

    // --- Reduction ---

    template <class InputDataType, class OutputDataType, class ComputeDataType = double>
    static void reduce(hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
                       hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output,
                       hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode)
    {
        validateInput(input, output, mode);

        // Validate data types
        static_assert(IS_SUPPORTED_DATA_TYPE<InputDataType>,
                      "Reduction supports only float, double, half, and bfloat16 input data types");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<OutputDataType>,
            "Reduction supports only float, double, half, and bfloat16 output data types");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ComputeDataType>,
            "Reduction supports only float, double, half, and bfloat16 compute data types");

        auto defines = detail::
            buildReductionDefines<InputDataType, OutputDataType, ComputeDataType, BLOCK_SIZE>(
                static_cast<unsigned int>(input.dims().size()));

        launchReduction(mode,
                        input.memory().deviceData(),
                        input.dims(),
                        input.strides(),
                        output.memory().deviceData(),
                        output.dims(),
                        output.strides(),
                        defines);

        output.memory().markDeviceModified();
    }

private:
    // --- Validators ---

    template <class T>
    static constexpr bool IS_SUPPORTED_DATA_TYPE
        = std::is_same_v<T, double> || std::is_same_v<T, float>
          || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

    template <typename InputDataType, typename OutputDataType>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<InputDataType>& input,
                              const hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output,
                              hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode)
    {
        // Validate tensor dimensions
        const auto& inputDims = input.dims();
        const auto& outputDims = output.dims();

        if(inputDims.size() != outputDims.size())
        {
            throw std::invalid_argument(
                "Reduction expects input and output to have the same rank (input rank="
                + std::to_string(inputDims.size())
                + ", output rank=" + std::to_string(outputDims.size()) + ")");
        }

        bool hasReducedDim = false;
        for(size_t i = 0; i < inputDims.size(); ++i)
        {
            if(outputDims[i] != inputDims[i] && outputDims[i] != 1)
            {
                throw std::invalid_argument("Reduction output dim[" + std::to_string(i)
                                            + "]=" + std::to_string(outputDims[i])
                                            + " must equal input dim[" + std::to_string(i)
                                            + "]=" + std::to_string(inputDims[i]) + " or be 1");
            }
            if(outputDims[i] == 1 && inputDims[i] > 1)
            {
                hasReducedDim = true;
            }
        }

        if(!hasReducedDim)
        {
            throw std::invalid_argument("Reduction expects at least one output dimension to be 1 "
                                        "to indicate a reduced dimension");
        }

        // Validate tensor layouts
        // Should we check if the tensor layouts are amongst the supported layouts (e.g., NCHW, NHWC, etc.)?
        if(!hipdnn_data_sdk::utilities::isLayoutAgnostic(inputDims)
           && !hipdnn_data_sdk::utilities::isLayoutAgnostic(outputDims))
        {
            const auto inputStrideOrder
                = hipdnn_data_sdk::utilities::extractStrideOrder(input.strides());
            const auto outputStrideOrder
                = hipdnn_data_sdk::utilities::extractStrideOrder(output.strides());
            if(inputStrideOrder != outputStrideOrder)
            {
                throw std::invalid_argument(
                    "Reduction requires input and output tensor layouts to be consistent");
            }
        }

        // Validate reduction mode
        using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
        switch(mode)
        {
        case ReductionMode::ADD:
        case ReductionMode::AVG:
        case ReductionMode::AMAX:
        case ReductionMode::NORM1:
        case ReductionMode::NORM2:
        case ReductionMode::MUL:
        case ReductionMode::MUL_NO_ZEROS:
        case ReductionMode::MIN_OP:
        case ReductionMode::MAX_OP:
            break;
        default:
            throw std::invalid_argument("Unsupported reduction mode: "
                                        + std::to_string(static_cast<int>(mode)));
        }
    }

    // --- Kernel launchers (defined in GpuFpReferenceReduction.cpp) ---

    static void launchReduction(hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode,
                                const void* inputPtr,
                                const std::vector<int64_t>& inputDims,
                                const std::vector<int64_t>& inputStrides,
                                void* outputPtr,
                                const std::vector<int64_t>& outputDims,
                                const std::vector<int64_t>& outputStrides,
                                std::vector<std::string>& defines);
};

} // namespace hipdnn_gpu_ref
