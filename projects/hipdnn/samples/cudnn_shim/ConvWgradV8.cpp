// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Convolution backward filter built the way PyTorch builds it through the cuDNN v8 (v0.x builder)
// surface of the shim (x, dw and dy set as x, w and y), run on a real provider and checked
// against the CPU reference.

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>

#include "V8SampleUtils.hpp"

namespace fe = v8_sample::fe;
namespace sdk = hipdnn_data_sdk::utilities;

namespace
{

constexpr int64_t X_UID = 'x';
constexpr int64_t DW_UID = 'w';
constexpr int64_t DY_UID = 'y';

template <typename T>
bool runVariant(cudnnHandle_t handle,
                const Config& config,
                const TensorLayout& layout,
                const v8_sample::ConvProblem& problem,
                const std::string& label)
{
    std::ostringstream variant;
    variant << "wgrad " << hipdnn_frontend::getDataTypeEnumFromType<T>() << " [" << layout << "]"
            << label;
    std::cout << "Running v8 convolution " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<T>();
    const fe::Operation op
        = v8_sample::makeConvOp(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                problem,
                                v8_sample::makeTensor(X_UID, problem.xDims(), layout, ioType),
                                v8_sample::makeTensor(DW_UID, problem.wDims(), layout, ioType),
                                v8_sample::makeTensor(DY_UID, problem.yDims(), layout, ioType));
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<T> x(problem.xDims(), layout);
    sdk::Tensor<T> dw(problem.wDims(), layout);
    sdk::Tensor<T> dy(problem.yDims(), layout);
    x.fillWithRandomValues(static_cast<T>(0.0f), static_cast<T>(1.0f));
    dy.fillWithRandomValues(static_cast<T>(0.0f), static_cast<T>(1.0f));
    dw.fillWithValue(static_cast<T>(0.0f));

    v8_sample::execute(
        handle,
        *plan,
        {X_UID, DW_UID, DY_UID},
        {x.memory().deviceData(), dw.memory().deviceData(), dy.memory().deviceData()});
    dw.memory().markDeviceModified();
    v8_sample::ranVariant(variant.str());

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";
        sdk::Tensor<T> dwRef(problem.wDims(), layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::wgrad(x,
                                                                     dwRef,
                                                                     dy,
                                                                     problem.stride,
                                                                     problem.dilation,
                                                                     problem.prePadding,
                                                                     problem.postPadding);
        // True convolution's filter gradient is the cross-correlation gradient flipped.
        if(problem.mode == CUDNN_CONVOLUTION)
        {
            dwRef = v8_sample::flipSpatial(dwRef);
        }
        const float tolerance
            = hipdnn_test_sdk::utilities::conv::calculateConvWrwTolerance<T, T, float>(
                0.0, 1.0, 0.0, 1.0, problem.yDims());
        constexpr float RELATIVE_TOLERANCE = 0.01f;
        validationPassed = v8_sample::matches("dw", dwRef, dw, tolerance, RELATIVE_TOLERANCE);
    }

    std::cout << "v8 convolution " << variant.str() << " complete.\n\n";
    return validationPassed;
}

v8_sample::ConvProblem defaultProblem(const Config& config)
{
    return v8_sample::problemFrom(config, {4, 16, 16, 16, 16});
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    return runVariant<InputType>(handle, config, layout, defaultProblem(config), "");
}

int main(int argc, char** argv)
{
    return v8_sample::guardedMain([argc, argv]() {
        RETURN_SUCCESS_IF_NO_DEVICE();

        const v8_sample::Options options = v8_sample::parseOptions(argc, argv);
        const auto handle = v8_sample::createHandle();
        bool allPassed = run(SampleRunner{*handle, options.config});
        if(v8_sample::selectsFp32Nchw(options.config))
        {
            for(const auto& [label, problem] :
                v8_sample::edgeConvProblems(defaultProblem(options.config)))
            {
                allPassed = runVariant<float>(
                                *handle, options.config, TensorLayout::NCHW, problem, " " + label)
                            && allPassed;
            }
        }
        return v8_sample::finish(options, allPassed);
    });
}
