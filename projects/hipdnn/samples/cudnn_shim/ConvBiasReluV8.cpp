// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Convolution fprop + bias ADD + ReLU through the cuDNN v8 (v0.x builder) surface of the shim,
// shaped for MIOpen's fused conv-bias-activation engine, and checked against the CPU reference.

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
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

fe::Operation pointwiseOp(cudnnPointwiseMode_t mode,
                          cudnnDataType_t computeType,
                          const fe::Tensor& x,
                          const fe::Tensor* b,
                          const fe::Tensor& y)
{
    fe::OperationBuilder builder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    builder.setpwDesc(fe::PointWiseDescBuilder().setMode(mode).setComputeType(computeType).build())
        .setxDesc(x)
        .setyDesc(y);
    if(b != nullptr)
    {
        builder.setbDesc(*b);
    }
    return builder.build();
}

template <typename InputType>
bool runVariant(cudnnHandle_t handle, const Config& config, const TensorLayout& layout, int64_t k)
{
    std::ostringstream variant;
    variant << hipdnn_frontend::getDataTypeEnumFromType<InputType>() << " [" << layout
            << "] K=" << k;
    std::cout << "Running v8 convolution fprop + bias + ReLU " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    // Defaults match a fused case the MIOpen engines claim on gfx1151 (fp32 NCHW, K = 1).
    const auto pick = [](const std::vector<int64_t>& values, size_t index, int64_t fallback) {
        return values.size() > index ? values[index] : fallback;
    };
    const int64_t n = pick(config.dims, 0, 1);
    const int64_t c = pick(config.dims, 1, 16);
    const int64_t h = pick(config.dims, 2, 16);
    const int64_t w = pick(config.dims, 3, 8);
    const int64_t r = pick(config.filter, 1, 3);
    const int64_t s = pick(config.filter, 2, 3);
    const std::vector<int64_t> stride = {pick(config.stride, 0, 1), pick(config.stride, 1, 1)};
    const std::vector<int64_t> padding = {pick(config.padding, 0, 1), pick(config.padding, 1, 1)};
    const std::vector<int64_t> dilation
        = {pick(config.dilation, 0, 1), pick(config.dilation, 1, 1)};

    // No v8 shape inference: y's dims are computed here.
    const int64_t outH = (h + 2 * padding[0] - dilation[0] * (r - 1) - 1) / stride[0] + 1;
    const int64_t outW = (w + 2 * padding[1] - dilation[1] * (s - 1) - 1) / stride[1] + 1;

    const std::vector<int64_t> xDims = {n, c, h, w};
    const std::vector<int64_t> wDims = {k, c, r, s};
    const std::vector<int64_t> yDims = {n, k, outH, outW};
    const std::vector<int64_t> biasDims = sdk::getDerivedShape(yDims);

    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<InputType>();
    const auto tensor
        = [&](int64_t uid, const std::vector<int64_t>& dims, cudnnDataType_t type, bool isVirtual) {
              return v8_sample::makeTensor(
                  uid, dims, sdk::generateStrides(dims, layout.strideOrder), type, isVirtual);
          };
    const fe::Tensor xDesc = tensor(X_UID, xDims, ioType, false);
    const fe::Tensor wDesc = tensor(W_UID, wDims, ioType, false);
    const fe::Tensor convYDesc = tensor(CONV_Y_UID, yDims, CUDNN_DATA_FLOAT, true);
    const fe::Tensor biasDesc = tensor(BIAS_UID, biasDims, ioType, false);
    const fe::Tensor addYDesc = tensor(ADD_Y_UID, yDims, CUDNN_DATA_FLOAT, true);
    const fe::Tensor yDesc = tensor(Y_UID, yDims, ioType, false);

    const fe::Operation conv
        = fe::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
              .setxDesc(xDesc)
              .setwDesc(wDesc)
              .setyDesc(convYDesc)
              .setcDesc(v8_sample::makeConvDesc(padding, stride, dilation))
              .build();
    // MIOpen's fused engine requires the ADD to compute in the bias tensor's type.
    const fe::Operation add
        = pointwiseOp(CUDNN_POINTWISE_ADD, ioType, convYDesc, &biasDesc, addYDesc);
    const fe::Operation relu
        = pointwiseOp(CUDNN_POINTWISE_RELU_FWD, CUDNN_DATA_FLOAT, addYDesc, nullptr, yDesc);

    std::array<const fe::Operation*, 3> ops = {&conv, &add, &relu};
    fe::OperationGraph graph = fe::OperationGraphBuilder()
                                   .setHandle(handle)
                                   .setOperationGraph(static_cast<int64_t>(ops.size()), ops.data())
                                   .build();

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<InputType> xTensor(xDims, layout);
    sdk::Tensor<InputType> wTensor(wDims, layout);
    sdk::Tensor<InputType> biasTensor(biasDims, layout);
    sdk::Tensor<InputType> yTensor(yDims, layout);
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    // Signed weights put pre-activations on both sides of 0, so the ReLU is not the identity.
    wTensor.fillWithRandomValues(static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    yTensor.fillWithValue(static_cast<InputType>(0.0f));

    const sdk::Workspace workspace(static_cast<size_t>(plan->getWorkspaceSize()));
    v8_sample::execute(handle,
                       *plan,
                       {X_UID, W_UID, BIAS_UID, Y_UID},
                       {xTensor.memory().deviceData(),
                        wTensor.memory().deviceData(),
                        biasTensor.memory().deviceData(),
                        yTensor.memory().deviceData()},
                       workspace.get());
    yTensor.memory().markDeviceModified();
    v8_sample::ranVariant();

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";

        sdk::Tensor<InputType> convRef(yDims, layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
            xTensor, wTensor, convRef, stride, dilation, padding);

        sdk::Tensor<InputType> addRef(yDims, layout);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<InputType>::pointwiseCompute(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD, addRef, convRef, biasTensor);

        sdk::Tensor<InputType> yRef(yDims, layout);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<InputType>::pointwiseCompute(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, yRef, addRef);

        const auto tolerance = hipdnn_test_sdk::utilities::conv::getToleranceFwd<InputType>();
        auto validator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);
        validationPassed = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "y", validator, yRef, yTensor, tolerance, tolerance);
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

        const Config config = parseCommandLineArgs(argc, argv);
        const v8_sample::CudnnHandle handle;
        const bool allPassed = run(SampleRunner{handle.get(), config});
        return v8_sample::finish(allPassed);
    });
}
