// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <cstring>
#include <hipdnn-gpu-ref/GpuReferencePointwise.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>

#include <cstdint>
#include <functional>
#include <hip/hip_runtime.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{
namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefPointwiseArgs.h> // NOLINT(misc-include-cleaner)

void launchKernel(hipFunction_t function, int64_t numBlocks, void* argsPtr, size_t argsSize)
{
    int deviceId;
    detail::throwOnHipError(hipGetDevice(&deviceId), "hipGetDevice failed");
    hipDeviceProp_t deviceProps;
    detail::throwOnHipError(hipGetDeviceProperties(&deviceProps, deviceId),
                            "hipGetDeviceProperties failed");
    if(const auto maxBlocks = static_cast<int64_t>(deviceProps.maxGridSize[0]);
       numBlocks > maxBlocks)
    {
        throw std::runtime_error("Grid size exceeds device limit: " + std::to_string(numBlocks)
                                 + " > " + std::to_string(maxBlocks));
    }
    if(numBlocks > static_cast<int64_t>(std::numeric_limits<unsigned>::max()))
    {
        throw std::runtime_error("Grid size exceeds hipModuleLaunchKernel limit");
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    detail::throwOnHipError(hipModuleLaunchKernel(function,
                                                  static_cast<unsigned int>(numBlocks),
                                                  1,
                                                  1,
                                                  GpuReferencePointwise::BLOCK_SIZE,
                                                  1,
                                                  1,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  config),
                            "hipModuleLaunchKernel failed");

    detail::throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
}

// For each input tensor, compute broadcast-aware strides to enable each thread in the HIP kernel
// execution grid can to load the correct input element.
// Broadcasting rules:
// 1. Dimensions are compared from right to left
// 2. Two dimensions are compatible if they are equal or input is 1 (output needs to be max size)
// 3. The input can have fewer dimensions than output (implicit leading 1s)
std::vector<int64_t> makeBroadcastStrides(const std::vector<int64_t>& inputDims,
                                          const std::vector<int64_t>& inputStrides,
                                          const std::vector<int64_t>& outputDims)
{
    std::vector<int64_t> bcastStrides(outputDims.size(), 0);
    const size_t offset = outputDims.size() - inputDims.size(); // right-align
    for(size_t i = 0; i < inputDims.size(); ++i)
    {
        bcastStrides[offset + i] = (inputDims[i] == 1) ? 0 : inputStrides[i];
    }
    return bcastStrides;
}

} // namespace

// --- Kernel launchers ---

void GpuReferencePointwise::launchUnary(
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
    const void* inputPtr,
    const std::vector<int64_t>& inputDims,
    const std::vector<int64_t>& inputStrides,
    void* outputPtr,
    const std::vector<int64_t>& outputDims,
    const std::vector<int64_t>& outputStrides,
    std::vector<std::string>& defines,
    const float lowerClip,
    const float upperClip,
    const float lowerSlope,
    const float swishBeta)
{
    int opCode;
    switch(operation)
    {
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY:
        opCode = POINTWISE_UNARY_OP_IDENTITY;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ABS:
        opCode = POINTWISE_UNARY_OP_ABS;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::NEG:
        opCode = POINTWISE_UNARY_OP_NEG;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD:
        opCode = POINTWISE_UNARY_OP_RELU_FWD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD:
        opCode = POINTWISE_UNARY_OP_SIGMOID_FWD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_FWD:
        opCode = POINTWISE_UNARY_OP_TANH_FWD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_FWD:
        opCode = POINTWISE_UNARY_OP_GELU_FWD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD:
        opCode = POINTWISE_UNARY_OP_GELU_APPROX_TANH_FWD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD:
        opCode = POINTWISE_UNARY_OP_SWISH_FWD;
        break;
    default:
        throw std::invalid_argument("Unsupported unary pointwise operation: "
                                    + std::to_string(static_cast<int>(operation)));
    }
    defines.emplace_back(std::string("-DOP=") + std::to_string(opCode));

    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel
        = compiler.getOrCompile("GpuRefPointwiseUnary.cpp", defines, "PointwiseUnaryRef");

    // Use output dims to calculate launch grid dimensions due to broadcasting rules
    const int64_t numElements
        = std::accumulate(outputDims.begin(), outputDims.end(), int64_t{1}, std::multiplies<>());
    int64_t numBlocks = numElements / GpuReferencePointwise::BLOCK_SIZE;
    numBlocks += (numElements % GpuReferencePointwise::BLOCK_SIZE == 0) ? 0 : 1;

    // Calculate the strides to access input elements at
    auto broadCastStrides = makeBroadcastStrides(inputDims, inputStrides, outputDims);

    PointwiseUnaryArgs args{};
    args.input = inputPtr;
    args.output = outputPtr;
    args.size = numElements;
    args.nDim = static_cast<int>(outputDims.size());
    std::memcpy(args.outputDims, outputDims.data(), outputDims.size() * sizeof(int64_t));
    std::memcpy(
        args.inputStrides, broadCastStrides.data(), broadCastStrides.size() * sizeof(int64_t));
    std::memcpy(args.outputStrides, outputStrides.data(), outputStrides.size() * sizeof(int64_t));
    args.lowerClip = lowerClip;
    args.lowerSlope = lowerSlope;
    args.upperClip = upperClip;
    args.swishBeta = swishBeta;

    launchKernel(kernel.function(), numBlocks, &args, sizeof(args));
}

void GpuReferencePointwise::launchBinary(
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
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
    const float lowerClip,
    const float upperClip,
    const float lowerSlope)
{
    int opCode;
    switch(operation)
    {
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD:
        opCode = POINTWISE_BINARY_OP_ADD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB:
        opCode = POINTWISE_BINARY_OP_SUB;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL:
        opCode = POINTWISE_BINARY_OP_MUL;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_BWD:
        opCode = POINTWISE_BINARY_OP_SIGMOID_BWD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_BWD:
        opCode = POINTWISE_BINARY_OP_TANH_BWD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD:
        opCode = POINTWISE_BINARY_OP_RELU_BWD;
        break;
    default:
        throw std::invalid_argument("Unsupported binary pointwise operation: "
                                    + std::to_string(static_cast<int>(operation)));
    }
    defines.emplace_back(std::string("-DOP=") + std::to_string(opCode));
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel
        = compiler.getOrCompile("GpuRefPointwiseBinary.cpp", defines, "PointwiseBinaryRef");

    // Use output dims to calculate launch grid dimensions due to broadcasting rules
    const int64_t numElements
        = std::accumulate(outputDims.begin(), outputDims.end(), int64_t{1}, std::multiplies<>());
    int64_t numBlocks = numElements / GpuReferencePointwise::BLOCK_SIZE;
    numBlocks += (numElements % GpuReferencePointwise::BLOCK_SIZE == 0) ? 0 : 1;

    // Calculate the strides to access input elements at
    auto broadCast0Strides = makeBroadcastStrides(input0Dims, input0Strides, outputDims);
    auto broadCast1Strides = makeBroadcastStrides(input1Dims, input1Strides, outputDims);

    PointwiseBinaryArgs args{};
    args.input0 = input0Ptr;
    args.input1 = input1Ptr;
    args.output = outputPtr;
    args.size = numElements;
    args.nDim = static_cast<int>(outputDims.size());
    std::memcpy(args.outputDims, outputDims.data(), outputDims.size() * sizeof(int64_t));
    std::memcpy(
        args.input0Strides, broadCast0Strides.data(), broadCast0Strides.size() * sizeof(int64_t));
    std::memcpy(
        args.input1Strides, broadCast1Strides.data(), broadCast1Strides.size() * sizeof(int64_t));
    std::memcpy(args.outputStrides, outputStrides.data(), outputStrides.size() * sizeof(int64_t));
    args.lowerClip = lowerClip;
    args.lowerSlope = lowerSlope;
    args.upperClip = upperClip;

    launchKernel(kernel.function(), numBlocks, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
