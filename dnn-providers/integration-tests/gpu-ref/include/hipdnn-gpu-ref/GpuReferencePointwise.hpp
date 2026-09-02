// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

template <typename OutputDataType,
          typename InputDataType,
          typename ComputeDataType,
          unsigned int localSize>
inline std::vector<std::string> buildDefines()
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<OutputDataType>::VALUE);
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<InputDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(localSize));
    return defines;
}

template <typename OutputDataType,
          typename Input0DataType,
          typename Input1DataType,
          typename ComputeDataType,
          unsigned int localSize>
inline std::vector<std::string> buildDefines()
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<OutputDataType>::VALUE);
    defines.emplace_back(std::string("-DINPUT0_TYPE=") + HipRtcTypeName<Input0DataType>::VALUE);
    defines.emplace_back(std::string("-DINPUT1_TYPE=") + HipRtcTypeName<Input1DataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DLOCAL_SIZE=") + std::to_string(localSize));
    return defines;
}

} // namespace detail

class GpuReferencePointwise
{
public:
    static constexpr unsigned int BLOCK_SIZE = 256;

    template <typename OutputType, typename InputType, typename ComputeType = double>
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 hipdnn_data_sdk::utilities::TensorBase<InputType>& input)
    {
        executeUnaryOperation<OutputType, InputType, ComputeType>(operation, output, input);
    }

    template <typename OutputType, typename InputType, typename ComputeType = double>
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 hipdnn_data_sdk::utilities::TensorBase<InputType>& input,
                                 const float lowerClip,
                                 const float upperClip,
                                 const float lowerSlope,
                                 const float swishBeta = float{1})
    {
        executeUnaryParamOperation<OutputType, InputType, ComputeType>(
            operation, output, input, lowerClip, upperClip, lowerSlope, swishBeta);
    }

    template <typename OutputType,
              typename Input0Type,
              typename Input1Type,
              typename ComputeType = double>
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 hipdnn_data_sdk::utilities::TensorBase<Input0Type>& input0,
                                 hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1)
    {
        executeBinaryOperation<OutputType, Input0Type, Input1Type, ComputeType>(
            operation, output, input0, input1);
    }

    template <typename OutputType,
              typename Input0Type,
              typename Input1Type,
              typename ComputeType = double>
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 hipdnn_data_sdk::utilities::TensorBase<Input0Type>& input0,
                                 hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1,
                                 const float lowerClip,
                                 const float upperClip,
                                 const float lowerSlope)
    {
        executeBinaryParamOperation<OutputType, Input0Type, Input1Type, ComputeType>(
            operation, output, input0, input1, lowerClip, upperClip, lowerSlope);
    }

private:
    template <typename OutputType, typename InputType, typename ComputeType = double>
    static void executeUnaryOperation(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                      hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                      hipdnn_data_sdk::utilities::TensorBase<InputType>& input)
    {
        validateTensors<OutputType, InputType, ComputeType>(output, input);

        auto defines = detail::buildDefines<OutputType, InputType, ComputeType, BLOCK_SIZE>();

        launchUnary(operation,
                    input.memory().deviceData(),
                    input.dims(),
                    input.strides(),
                    output.memory().deviceData(),
                    output.dims(),
                    output.strides(),

                    defines);
        output.memory().markDeviceModified();
    }

    template <typename OutputType, typename InputType, typename ComputeType = double>
    static void
        executeUnaryParamOperation(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                   hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                   hipdnn_data_sdk::utilities::TensorBase<InputType>& input,
                                   const float lowerClip,
                                   const float upperClip,
                                   const float lowerSlope,
                                   const float swishBeta = float{1})
    {
        validateTensors<OutputType, InputType, ComputeType>(output, input);

        auto defines = detail::buildDefines<OutputType, InputType, ComputeType, BLOCK_SIZE>();

        launchUnary(operation,
                    input.memory().deviceData(),
                    input.dims(),
                    input.strides(),
                    output.memory().deviceData(),
                    output.dims(),
                    output.strides(),
                    defines,
                    lowerClip,
                    upperClip,
                    lowerSlope,
                    swishBeta);
        output.memory().markDeviceModified();
    }

    template <typename OutputType,
              typename Input0Type,
              typename Input1Type,
              typename ComputeType = double>
    static void
        executeBinaryOperation(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                               hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                               hipdnn_data_sdk::utilities::TensorBase<Input0Type>& input0,
                               hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1)

    {
        validateTensors<OutputType, Input0Type, Input1Type, ComputeType>(output, input0, input1);

        auto defines
            = detail::buildDefines<OutputType, Input0Type, Input1Type, ComputeType, BLOCK_SIZE>();

        launchBinary(operation,
                     input0.memory().deviceData(),
                     input0.dims(),
                     input0.strides(),
                     input1.memory().deviceData(),
                     input1.dims(),
                     input1.strides(),
                     output.memory().deviceData(),
                     output.dims(),
                     output.strides(),
                     defines);
        output.memory().markDeviceModified();
    }

    template <typename OutputType,
              typename Input0Type,
              typename Input1Type,
              typename ComputeType = double>
    static void
        executeBinaryParamOperation(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                    hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                    hipdnn_data_sdk::utilities::TensorBase<Input0Type>& input0,
                                    hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1,
                                    const float lowerClip,
                                    const float upperClip,
                                    const float lowerSlope)
    {
        validateTensors<OutputType, Input0Type, Input1Type, ComputeType>(output, input0, input1);

        auto defines
            = detail::buildDefines<OutputType, Input0Type, Input1Type, ComputeType, BLOCK_SIZE>();

        launchBinary(operation,
                     input0.memory().deviceData(),
                     input0.dims(),
                     input0.strides(),
                     input1.memory().deviceData(),
                     input1.dims(),
                     input1.strides(),
                     output.memory().deviceData(),
                     output.dims(),
                     output.strides(),
                     defines,
                     lowerClip,
                     upperClip,
                     lowerSlope);
        output.memory().markDeviceModified();
    }

    // --- Validators ---

    template <class T>
    static constexpr bool IS_SUPPORTED_DATA_TYPE
        = std::is_same_v<T, double> || std::is_same_v<T, float>
          || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

    template <typename OutputType, typename InputType, typename ComputeType>
    static void validateTensors(hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                hipdnn_data_sdk::utilities::TensorBase<InputType>& input)
    {
        if(output.dims().size() > 5)
        {
            throw std::invalid_argument("Pointwise operations have a max dimension of 5");
        }

        if(!hipdnn_data_sdk::utilities::areDimensionsBroadcastCompatible(input.dims(),
                                                                         output.dims()))
        {
            throw std::invalid_argument("Pointwise operations require input tensor dimensions "
                                        "to be broadcastable to dimensions of output tensor.");
        }

        static_assert(
            IS_SUPPORTED_DATA_TYPE<InputType>,
            "Pointwise supports only double, float, half, and bfloat16 input data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<OutputType>,
            "Pointwise supports only double, float, half, and bfloat16 output data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ComputeType>,
            "Pointwise supports only double, float, half, and bfloat16 compute data types.");
    }

    template <typename OutputType, typename Input0Type, typename Input1Type, typename ComputeType>
    static void validateTensors(hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                hipdnn_data_sdk::utilities::TensorBase<Input0Type>& input0,
                                hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1)
    {
        if(output.dims().size() > 5)
        {
            throw std::invalid_argument("Pointwise operations have a max dimension of 5");
        }

        if(!hipdnn_data_sdk::utilities::areDimensionsBroadcastCompatible(input0.dims(),
                                                                         output.dims()))
        {
            throw std::invalid_argument("Pointwise operations require input0 tensor dimensions "
                                        "to be broadcastable to dimensions of output tensor.");
        }

        if(!hipdnn_data_sdk::utilities::areDimensionsBroadcastCompatible(input1.dims(),
                                                                         output.dims()))
        {
            throw std::invalid_argument("Pointwise operations require input1 tensor dimensions "
                                        "to be broadcastable to dimensions of output tensor.");
        }
        static_assert(
            IS_SUPPORTED_DATA_TYPE<Input0Type>,
            "Pointwise supports only double, float, half, and bfloat16 input0 data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<Input1Type>,
            "Pointwise supports only double, float, half, and bfloat16 input1 data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<OutputType>,
            "Pointwise supports only double, float, half, and bfloat16 output data types.");
        static_assert(
            IS_SUPPORTED_DATA_TYPE<ComputeType>,
            "Pointwise supports only double, float, half, and bfloat16 compute data types.");
    }

    // --- Kernel launchers (defined in GpuReferencePointwise.cpp) ---

    static void launchUnary(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                            const void* inputPtr,
                            const std::vector<int64_t>& inputDims,
                            const std::vector<int64_t>& inputStrides,
                            void* outputPtr,
                            const std::vector<int64_t>& outputDims,
                            const std::vector<int64_t>& outputStrides,
                            std::vector<std::string>& defines,
                            float lowerClip = 0.f,
                            float upperClip = std::numeric_limits<float>::max(),
                            float lowerSlope = 0.f,
                            float swishBeta = 1.f);

    static void launchBinary(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                             const void* input0Ptr,
                             const std::vector<int64_t>& input0Dims,
                             const std::vector<int64_t>& input0Strides,
                             const void* input1Ptr,
                             const std::vector<int64_t>& input1Dims,
                             const std::vector<int64_t>& input1Strides,
                             void* outputPtr,
                             const std::vector<int64_t>& outputDims,
                             const std::vector<int64_t>& outputStrides,
                             std::vector<std::string>& defines,
                             float lowerClip = 0.f,
                             float upperClip = std::numeric_limits<float>::max(),
                             float lowerSlope = 0.f);
};

} // namespace hipdnn_gpu_ref
