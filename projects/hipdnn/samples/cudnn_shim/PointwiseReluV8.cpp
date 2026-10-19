// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A single RELU_FWD node built with the cuDNN v8 (v0.x) builders and run through
// the cuDNN backend shim.

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>

#include "V8SampleUtils.hpp"

namespace fe = v8_sample::fe;

namespace
{

constexpr int64_t X_UID = 'x';
constexpr int64_t Y_UID = 'y';

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = hipdnn_frontend::getDataTypeEnumFromType<InputType>();
    std::ostringstream variant;
    variant << "pointwise relu " << inputType << " [" << layout << "]";

    std::cout << "Running " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const int64_t n = config.dims.size() > 0 ? config.dims[0] : 1;
    const int64_t c = config.dims.size() > 1 ? config.dims[1] : 16;
    const int64_t h = config.dims.size() > 2 ? config.dims[2] : 4;
    const int64_t w = config.dims.size() > 3 ? config.dims[3] : 4;

    hipdnn_data_sdk::utilities::Tensor<InputType> xTensor({n, c, h, w}, layout);
    hipdnn_data_sdk::utilities::Tensor<InputType> yTensor({n, c, h, w}, layout);

    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<InputType>();
    const fe::Tensor x = v8_sample::makeTensor(X_UID, xTensor.dims(), xTensor.strides(), ioType);
    const fe::Tensor y = v8_sample::makeTensor(Y_UID, yTensor.dims(), yTensor.strides(), ioType);

    const fe::PointWiseDesc relu = fe::PointWiseDescBuilder()
                                       .setMode(CUDNN_POINTWISE_RELU_FWD)
                                       .setComputeType(CUDNN_DATA_FLOAT)
                                       .build();
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(x)
                         .setyDesc(y)
                         .setpwDesc(relu)
                         .build());

    fe::OperationGraph graph
        = fe::OperationGraphBuilder().setHandle(handle).setOperationGraph(ops).build();

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }

    xTensor.fillWithRandomValues(static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f));
    yTensor.fillWithValue(static_cast<InputType>(0.0f));

    const hipdnn_data_sdk::utilities::Workspace workspace(
        static_cast<size_t>(plan->getWorkspaceSize()));
    v8_sample::execute(handle,
                       *plan,
                       {X_UID, Y_UID},
                       {xTensor.memory().deviceData(), yTensor.memory().deviceData()},
                       workspace.get());
    yTensor.memory().markDeviceModified();
    v8_sample::ranVariant();

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        hipdnn_data_sdk::utilities::Tensor<InputType> yRefTensor({n, c, h, w}, layout);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<InputType>::pointwiseCompute(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, yRefTensor, xTensor);

        const float tolerance = hipdnn_test_sdk::utilities::pointwise::getTolerance<InputType>();
        auto validator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);
        validationPassed = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "y", validator, yRefTensor, yTensor, tolerance, tolerance);
    }

    std::cout << "Completed " << variant.str() << ".\n\n";
    return validationPassed;
}

int main(int argc, char** argv)
{
    return v8_sample::guardedMain([&]() {
        RETURN_SUCCESS_IF_NO_DEVICE();

        const Config config = parseCommandLineArgs(argc, argv);
        const v8_sample::CudnnHandle handle;

        return v8_sample::finish(run(SampleRunner{handle.get(), config}));
    });
}
