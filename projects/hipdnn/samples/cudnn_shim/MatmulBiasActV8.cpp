// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// matmul -> ADD(bias) -> activation built with the cuDNN v8 (v0.x) builders and run
// through the cuDNN backend shim.

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesMatmul.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesPointwise.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/PointwiseErrorClassification.hpp>

#include "V8SampleUtils.hpp"

namespace fe = v8_sample::fe;

namespace
{

using hipdnn_data_sdk::utilities::Tensor;
using ReferenceMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

constexpr int64_t A_UID = 'a';
constexpr int64_t B_UID = 'b';
constexpr int64_t C_UID = 'c';
constexpr int64_t BIAS_UID = 'z';
constexpr int64_t BIASED_UID = 'd';
constexpr int64_t Y_UID = 'y';

constexpr int64_t BATCH = 3;
constexpr int64_t M = 16;
constexpr int64_t K = 32;
constexpr int64_t N = 128;

// The cuDNN and reference modes are paired by hand; the enums are never cast.
struct Activation
{
    const char* name;
    cudnnPointwiseMode_t mode;
    ReferenceMode referenceMode;
};

constexpr std::array<Activation, 2> ACTIVATIONS = {{
    {"gelu_approx_tanh", CUDNN_POINTWISE_GELU_APPROX_TANH_FWD, ReferenceMode::GELU_APPROX_TANH_FWD},
    {"relu", CUDNN_POINTWISE_RELU_FWD, ReferenceMode::RELU_FWD},
}};

fe::Operation pointwise(cudnnPointwiseMode_t mode,
                        const fe::Tensor& x,
                        const fe::Tensor* b,
                        const fe::Tensor& y)
{
    const fe::PointWiseDesc desc
        = fe::PointWiseDescBuilder().setMode(mode).setComputeType(CUDNN_DATA_FLOAT).build();
    fe::OperationBuilder builder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    builder.setxDesc(x).setyDesc(y).setpwDesc(desc);
    if(b != nullptr)
    {
        builder.setbDesc(*b);
    }
    return builder.build();
}

template <typename T>
bool runVariant(cudnnHandle_t handle,
                const Config& config,
                const std::string& dtypeName,
                const Activation& activation)
{
    const std::string variant = std::string("matmul+bias+") + activation.name + " " + dtypeName;
    std::cout << "Running " << variant << (config.cpuValidation ? " (with CPU validation)" : "")
              << "...\n";

    const std::vector<int64_t> cDims = {BATCH, M, N};
    Tensor<T> aTensor({BATCH, M, K});
    Tensor<T> bTensor({BATCH, K, N});
    Tensor<T> biasTensor({1, 1, N});
    Tensor<T> yTensor(cDims);

    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<T>();
    const fe::Tensor a = v8_sample::makeTensor(A_UID, aTensor.dims(), aTensor.strides(), ioType);
    const fe::Tensor b = v8_sample::makeTensor(B_UID, bTensor.dims(), bTensor.strides(), ioType);
    const fe::Tensor c
        = v8_sample::makeTensor(C_UID, cDims, yTensor.strides(), CUDNN_DATA_FLOAT, true);
    const fe::Tensor bias
        = v8_sample::makeTensor(BIAS_UID, biasTensor.dims(), biasTensor.strides(), ioType);
    const fe::Tensor biased
        = v8_sample::makeTensor(BIASED_UID, cDims, yTensor.strides(), CUDNN_DATA_FLOAT, true);
    const fe::Tensor y = v8_sample::makeTensor(Y_UID, cDims, yTensor.strides(), ioType);

    const fe::MatMulDesc matmul = fe::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                         .setaMatDesc(a)
                         .setbMatDesc(b)
                         .setcMatDesc(c)
                         .setmatmulDesc(matmul)
                         .build());
    ops.emplace_back(pointwise(CUDNN_POINTWISE_ADD, c, &bias, biased));
    ops.emplace_back(pointwise(activation.mode, biased, nullptr, y));

    fe::OperationGraph graph
        = fe::OperationGraphBuilder().setHandle(handle).setOperationGraph(ops).build();

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant);
    }

    aTensor.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    bTensor.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    yTensor.fillWithValue(static_cast<T>(0.0f));

    const hipdnn_data_sdk::utilities::Workspace workspace(
        static_cast<size_t>(plan->getWorkspaceSize()));
    v8_sample::execute(handle,
                       *plan,
                       {A_UID, B_UID, BIAS_UID, Y_UID},
                       {aTensor.memory().deviceData(),
                        bTensor.memory().deviceData(),
                        biasTensor.memory().deviceData(),
                        yTensor.memory().deviceData()},
                       workspace.get());
    yTensor.memory().markDeviceModified();
    v8_sample::ranVariant();

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        Tensor<float> cRef(cDims);
        hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul(aTensor, bTensor, cRef);

        Tensor<float> biasedRef(cDims);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<float>::pointwiseCompute(
            ReferenceMode::ADD, biasedRef, cRef, biasTensor);

        // Writing the activation into an io-typed tensor is the final cast to T.
        Tensor<T> yRef(cDims);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<T>::pointwiseCompute(
            activation.referenceMode, yRef, biasedRef);

        const float tolerance
            = hipdnn_test_sdk::utilities::matmul::calculateMatmulTolerance<T, T, float>(aTensor,
                                                                                        bTensor)
              + hipdnn_test_sdk::utilities::pointwise::calculatePointwiseTolerance<T, T, float>(
                  1.0,
                  hipdnn_test_sdk::utilities::pointwise::classifyPointwiseOp(
                      activation.referenceMode));
        auto validator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<T>(tolerance, tolerance);
        validationPassed = hipdnn_test_sdk::utilities::validateAndReport<T>(
            std::cout, "y", validator, yRef, yTensor, tolerance, tolerance);
    }

    std::cout << "Completed " << variant << ".\n\n";
    return validationPassed;
}

template <typename T>
bool runActivations(cudnnHandle_t handle, const Config& config, const std::string& dtypeName)
{
    bool allPassed = true;
    for(const Activation& activation : ACTIVATIONS)
    {
        allPassed &= runVariant<T>(handle, config, dtypeName, activation);
    }
    return allPassed;
}

} // namespace

int main(int argc, char** argv)
{
    return v8_sample::guardedMain([&]() {
        RETURN_SUCCESS_IF_NO_DEVICE();

        const Config config = parseCommandLineArgs(argc, argv);
        const v8_sample::CudnnHandle handle;

        // Matmul has no layout, so --layout is ignored and only --dtype selects variants.
        bool allPassed = true;
        if(config.dtype.empty() || config.dtype == "fp32")
        {
            allPassed &= runActivations<float>(handle.get(), config, "fp32");
        }
        if(config.dtype.empty() || config.dtype == "fp16")
        {
            allPassed &= runActivations<half>(handle.get(), config, "fp16");
        }
        if(config.dtype.empty() || config.dtype == "bf16")
        {
            allPassed &= runActivations<bfloat16>(handle.get(), config, "bf16");
        }
        return v8_sample::finish(allPassed);
    });
}
