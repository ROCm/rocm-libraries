// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Convolution fprop + bias ADD + ReLU through the cuDNN v8 (v0.x builder) surface of the shim,
// shaped for MIOpen's fused conv-bias-activation engine, and checked against the CPU reference.

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>

#include "V8SampleUtils.hpp"

namespace fe = v8_sample::fe;
namespace sdk = hipdnn_data_sdk::utilities;

namespace
{

constexpr int64_t X_UID = 'x';
constexpr int64_t W_UID = 'w';
constexpr int64_t CONV_Y_UID = 'c';
constexpr int64_t BIAS_UID = 'b';
constexpr int64_t ADD_Y_UID = 'a';
constexpr int64_t Y_UID = 'y';

template <typename InputType>
bool runVariant(cudnnHandle_t handle, const Config& config, const TensorLayout& layout, int64_t k)
{
    std::ostringstream variant;
    variant << hipdnn_frontend::getDataTypeEnumFromType<InputType>() << " [" << layout
            << "] K=" << k;
    std::cout << "Running v8 convolution fprop + bias + ReLU " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    // Defaults match a fused case the MIOpen engines claim on gfx1151 (fp32 NCHW, K = 1).
    const v8_sample::ConvProblem problem = v8_sample::problemFrom(config, {1, 16, 16, 8, k});
    const std::vector<int64_t> yDims = problem.yDims();
    const std::vector<int64_t> biasDims = sdk::getDerivedShape(yDims);

    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<InputType>();
    const fe::Tensor convYDesc
        = v8_sample::makeTensor(CONV_Y_UID, yDims, layout, CUDNN_DATA_FLOAT, true);
    const fe::Tensor biasDesc = v8_sample::makeTensor(BIAS_UID, biasDims, layout, ioType);
    const fe::Tensor addYDesc
        = v8_sample::makeTensor(ADD_Y_UID, yDims, layout, CUDNN_DATA_FLOAT, true);
    const fe::Tensor yDesc = v8_sample::makeTensor(Y_UID, yDims, layout, ioType);

    const fe::Operation conv
        = v8_sample::makeConvOp(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                problem,
                                v8_sample::makeTensor(X_UID, problem.xDims(), layout, ioType),
                                v8_sample::makeTensor(W_UID, problem.wDims(), layout, ioType),
                                convYDesc);
    // MIOpen's fused engine requires the ADD to compute in the bias tensor's type.
    const fe::Operation add
        = v8_sample::makePointwiseOp(CUDNN_POINTWISE_ADD, ioType, convYDesc, &biasDesc, addYDesc);
    const fe::Operation relu = v8_sample::makePointwiseOp(
        CUDNN_POINTWISE_RELU_FWD, CUDNN_DATA_FLOAT, addYDesc, nullptr, yDesc);
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&conv, &add, &relu});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<InputType> xTensor(problem.xDims(), layout);
    sdk::Tensor<InputType> wTensor(problem.wDims(), layout);
    sdk::Tensor<InputType> biasTensor(biasDims, layout);
    sdk::Tensor<InputType> yTensor(yDims, layout);
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    // Signed weights put pre-activations on both sides of 0, so the ReLU is not the identity.
    wTensor.fillWithRandomValues(static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    yTensor.fillWithValue(static_cast<InputType>(0.0f));

    v8_sample::execute(handle,
                       *plan,
                       {X_UID, W_UID, BIAS_UID, Y_UID},
                       {xTensor.memory().deviceData(),
                        wTensor.memory().deviceData(),
                        biasTensor.memory().deviceData(),
                        yTensor.memory().deviceData()});
    yTensor.memory().markDeviceModified();
    v8_sample::ranVariant(variant.str());

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";

        sdk::Tensor<InputType> convRef(yDims, layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(xTensor,
                                                                     wTensor,
                                                                     convRef,
                                                                     problem.stride,
                                                                     problem.dilation,
                                                                     problem.prePadding,
                                                                     problem.postPadding);

        sdk::Tensor<InputType> addRef(yDims, layout);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<InputType>::pointwiseCompute(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD, addRef, convRef, biasTensor);

        sdk::Tensor<InputType> yRef(yDims, layout);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<InputType>::pointwiseCompute(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, yRef, addRef);

        const auto tolerance = hipdnn_test_sdk::utilities::conv::getToleranceFwd<InputType>();
        validationPassed = v8_sample::matches("y", yRef, yTensor, tolerance, tolerance);
    }

    std::cout << "v8 convolution fprop + bias + ReLU complete for " << variant.str() << ".\n\n";
    return validationPassed;
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    // K = 4 also exercises the per-channel bias broadcast, which K = 1 reduces to a scalar.
    const std::vector<int64_t> outputChannels = config.filter.empty()
                                                    ? std::vector<int64_t>{1, 4}
                                                    : std::vector<int64_t>{config.filter[0]};
    bool allPassed = true;
    for(const int64_t k : outputChannels)
    {
        allPassed = runVariant<InputType>(handle, config, layout, k) && allPassed;
    }
    return allPassed;
}

int main(int argc, char** argv)
{
    return v8_sample::guardedMain([argc, argv]() {
        RETURN_SUCCESS_IF_NO_DEVICE();

        const v8_sample::Options options = v8_sample::parseOptions(argc, argv);
        const auto handle = v8_sample::createHandle();
        const bool allPassed = run(SampleRunner{*handle, options.config});
        return v8_sample::finish(options, allPassed);
    });
}
