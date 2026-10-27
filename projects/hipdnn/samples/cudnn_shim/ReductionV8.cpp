// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Per-channel ADD and MAX reductions built with the cuDNN v8 (v0.x) builders, run through the
// shim on a real provider and checked against the CPU reference.

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceReduction.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "V8SampleUtils.hpp"

namespace fe = v8_sample::fe;
namespace sdk = hipdnn_data_sdk::utilities;

namespace
{

constexpr int64_t X_UID = 'x';
constexpr int64_t Y_UID = 'y';

// The cuDNN and reference modes are paired by hand; the enums are never cast.
struct Reduction
{
    const char* name;
    cudnnReduceTensorOp_t mode;
    hipdnn_flatbuffers_sdk::data_objects::ReductionMode referenceMode;
};

constexpr std::array<Reduction, 2> REDUCTIONS = {{
    {"add", CUDNN_REDUCE_TENSOR_ADD, hipdnn_flatbuffers_sdk::data_objects::ReductionMode::ADD},
    {"max", CUDNN_REDUCE_TENSOR_MAX, hipdnn_flatbuffers_sdk::data_objects::ReductionMode::MAX},
}};

template <typename T>
bool runVariant(cudnnHandle_t handle,
                const Config& config,
                const TensorLayout& layout,
                const Reduction& reduction)
{
    std::ostringstream variant;
    variant << "reduction " << reduction.name << ' '
            << hipdnn_frontend::getDataTypeEnumFromType<T>() << " [" << layout << "]";
    std::cout << "Running v8 " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const auto pick = [&](size_t index, int64_t fallback) {
        return config.dims.size() > index ? config.dims[index] : fallback;
    };
    const std::vector<int64_t> xDims = {pick(0, 2), pick(1, 8), pick(2, 4), pick(3, 4)};
    // Every dimension but C reduces to 1.
    const std::vector<int64_t> yDims = sdk::getDerivedShape(xDims);

    const fe::ReductionDesc desc = fe::ReductionDescBuilder()
                                       .setComputeType(CUDNN_DATA_FLOAT)
                                       .setReductionOp(reduction.mode)
                                       .build();
    const fe::Operation op
        = fe::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
              .setreductionDesc(desc)
              .setxDesc(v8_sample::makeTensor(X_UID, xDims, layout, v8_sample::cudnnTypeOf<T>()))
              .setyDesc(v8_sample::makeTensor(Y_UID, yDims, layout, CUDNN_DATA_FLOAT))
              .build();
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<T> x(xDims, layout);
    sdk::Tensor<float> y(yDims, layout);
    x.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    y.fillWithValue(0.0f);

    v8_sample::execute(
        handle, *plan, {X_UID, Y_UID}, {x.memory().deviceData(), y.memory().deviceData()});
    y.memory().markDeviceModified();
    v8_sample::ranVariant();

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";
        sdk::Tensor<float> yRef(yDims, layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceReduction::reduce(
            x, yRef, reduction.referenceMode);
        const float tolerance = hipdnn_test_sdk::utilities::reduction::getTolerance<T>();
        validationPassed = v8_sample::matches("y", yRef, y, tolerance, tolerance);
    }

    std::cout << "v8 " << variant.str() << " complete.\n\n";
    return validationPassed;
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    bool allPassed = true;
    for(const Reduction& reduction : REDUCTIONS)
    {
        allPassed = runVariant<InputType>(handle, config, layout, reduction) && allPassed;
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
