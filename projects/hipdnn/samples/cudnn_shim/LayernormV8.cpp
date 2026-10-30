// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Layer norm forward training and backward over C, H and W, built with the cuDNN v8 (v0.x) norm
// builders, run through the shim on a real provider and checked against the CPU reference.

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "V8SampleUtils.hpp"

namespace fe = v8_sample::fe;
namespace sdk = hipdnn_data_sdk::utilities;

namespace
{

constexpr int64_t X_UID = 'x';
constexpr int64_t Y_UID = 'y';
constexpr int64_t DX_UID = 'X';
constexpr int64_t DY_UID = 'Y';
constexpr int64_t SCALE_UID = 's';
constexpr int64_t BIAS_UID = 'b';
constexpr int64_t DSCALE_UID = 'S';
constexpr int64_t DBIAS_UID = 'B';
constexpr int64_t EPSILON_UID = 'e';
constexpr int64_t MEAN_UID = 'm';
constexpr int64_t INV_VAR_UID = 'v';

constexpr float EPSILON = 1e-5f;
// C, H and W are normalized; N indexes the statistics.
constexpr int64_t NORMALIZED_DIMS = 3;

struct Shapes
{
    std::vector<int64_t> data;
    std::vector<int64_t> affine;
    std::vector<int64_t> stats;
};

Shapes shapesFrom(const Config& config)
{
    const auto pick = [&](size_t index, int64_t fallback) {
        return config.dims.size() > index ? config.dims[index] : fallback;
    };
    const int64_t n = pick(0, 4);
    const int64_t c = pick(1, 16);
    const int64_t h = pick(2, 8);
    const int64_t w = pick(3, 8);
    return {{n, c, h, w}, {1, c, h, w}, {n, 1, 1, 1}};
}

using v8_sample::makeFloatTensor;

template <typename T>
bool runForward(cudnnHandle_t handle, const Config& config, const TensorLayout& layout)
{
    std::ostringstream variant;
    variant << "layer norm forward training " << hipdnn_frontend::getDataTypeEnumFromType<T>()
            << " [" << layout << "]";
    std::cout << "Running v8 " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const Shapes shapes = shapesFrom(config);
    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<T>();

    const fe::Operation op
        = fe::OperationBuilder(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
              .setNormalizationMode(CUDNN_LAYER_NORM)
              .setNormFwdPhase(CUDNN_NORM_FWD_TRAINING)
              .setxDesc(v8_sample::makeTensor(X_UID, shapes.data, layout, ioType))
              .setScaleAndBias(
                  v8_sample::makeTensor(SCALE_UID, shapes.affine, layout, CUDNN_DATA_FLOAT),
                  v8_sample::makeTensor(BIAS_UID, shapes.affine, layout, CUDNN_DATA_FLOAT))
              .setEpsilonTensor(v8_sample::makeScalar(EPSILON_UID))
              .setyDesc(v8_sample::makeTensor(Y_UID, shapes.data, layout, ioType))
              .setSavedMeanAndInvVar(makeFloatTensor(MEAN_UID, shapes.stats),
                                     makeFloatTensor(INV_VAR_UID, shapes.stats))
              .build();
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<T> x(shapes.data, layout);
    sdk::Tensor<T> y(shapes.data, layout);
    sdk::Tensor<float> scale(shapes.affine, layout);
    sdk::Tensor<float> bias(shapes.affine, layout);
    sdk::Tensor<float> mean(shapes.stats);
    sdk::Tensor<float> invVar(shapes.stats);
    x.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    scale.fillWithRandomValues(-2.0f, 2.0f);
    bias.fillWithRandomValues(-2.0f, 2.0f);
    y.fillWithValue(static_cast<T>(0.0f));
    mean.fillWithValue(0.0f);
    invVar.fillWithValue(0.0f);

    float epsilon = EPSILON;
    v8_sample::execute(handle,
                       *plan,
                       {X_UID, SCALE_UID, BIAS_UID, EPSILON_UID, Y_UID, MEAN_UID, INV_VAR_UID},
                       {x.memory().deviceData(),
                        scale.memory().deviceData(),
                        bias.memory().deviceData(),
                        &epsilon,
                        y.memory().deviceData(),
                        mean.memory().deviceData(),
                        invVar.memory().deviceData()});
    y.memory().markDeviceModified();
    mean.memory().markDeviceModified();
    invVar.memory().markDeviceModified();
    v8_sample::ranVariant(variant.str());

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";
        sdk::Tensor<T> yRef(shapes.data, layout);
        sdk::Tensor<float> meanRef(shapes.stats);
        sdk::Tensor<float> invVarRef(shapes.stats);
        hipdnn_test_sdk::utilities::CpuFpReferenceLayernorm::fprop<T, float, T, float>(
            x, &scale, &bias, yRef, EPSILON, NORMALIZED_DIMS, &meanRef, &invVarRef);

        const float tolerance = hipdnn_test_sdk::utilities::layernorm::getTolerance<T>();
        validationPassed = v8_sample::matches("y", yRef, y, tolerance, tolerance);
        validationPassed = v8_sample::matches("saved_mean", meanRef, mean, tolerance, tolerance)
                           && validationPassed;
        validationPassed
            = v8_sample::matches("saved_inv_variance", invVarRef, invVar, tolerance, tolerance)
              && validationPassed;
    }

    std::cout << "v8 " << variant.str() << " complete.\n\n";
    return validationPassed;
}

template <typename T>
bool runBackward(cudnnHandle_t handle, const Config& config, const TensorLayout& layout)
{
    std::ostringstream variant;
    variant << "layer norm backward " << hipdnn_frontend::getDataTypeEnumFromType<T>() << " ["
            << layout << "]";
    std::cout << "Running v8 " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const Shapes shapes = shapesFrom(config);
    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<T>();

    const fe::Operation op
        = fe::OperationBuilder(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
              .setNormalizationMode(CUDNN_LAYER_NORM)
              .setxDesc(v8_sample::makeTensor(X_UID, shapes.data, layout, ioType))
              .setdyDesc(v8_sample::makeTensor(DY_UID, shapes.data, layout, ioType))
              .setScale(v8_sample::makeTensor(SCALE_UID, shapes.affine, layout, CUDNN_DATA_FLOAT))
              .setSavedMeanAndInvVar(makeFloatTensor(MEAN_UID, shapes.stats),
                                     makeFloatTensor(INV_VAR_UID, shapes.stats))
              .setEpsilonTensor(v8_sample::makeScalar(EPSILON_UID))
              .setdxDesc(v8_sample::makeTensor(DX_UID, shapes.data, layout, ioType))
              .setDScaleAndDBias(
                  v8_sample::makeTensor(DSCALE_UID, shapes.affine, layout, CUDNN_DATA_FLOAT),
                  v8_sample::makeTensor(DBIAS_UID, shapes.affine, layout, CUDNN_DATA_FLOAT))
              .build();
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<T> x(shapes.data, layout);
    sdk::Tensor<T> dy(shapes.data, layout);
    sdk::Tensor<T> dx(shapes.data, layout);
    sdk::Tensor<float> scale(shapes.affine, layout);
    sdk::Tensor<float> dscale(shapes.affine, layout);
    sdk::Tensor<float> dbias(shapes.affine, layout);
    sdk::Tensor<float> mean(shapes.stats);
    sdk::Tensor<float> invVar(shapes.stats);
    x.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    dy.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    scale.fillWithRandomValues(-2.0f, 2.0f);
    dx.fillWithValue(static_cast<T>(0.0f));
    dscale.fillWithValue(0.0f);
    dbias.fillWithValue(0.0f);

    // The saved statistics are those of x, as a forward pass would have produced them.
    {
        sdk::Tensor<T> unusedY(shapes.data, layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceLayernorm::fprop<T, float, T, float>(
            x, nullptr, nullptr, unusedY, EPSILON, NORMALIZED_DIMS, &mean, &invVar);
        mean.memory().markHostModified();
        invVar.memory().markHostModified();
    }

    float epsilon = EPSILON;
    v8_sample::execute(handle,
                       *plan,
                       {X_UID,
                        DY_UID,
                        SCALE_UID,
                        MEAN_UID,
                        INV_VAR_UID,
                        EPSILON_UID,
                        DX_UID,
                        DSCALE_UID,
                        DBIAS_UID},
                       {x.memory().deviceData(),
                        dy.memory().deviceData(),
                        scale.memory().deviceData(),
                        mean.memory().deviceData(),
                        invVar.memory().deviceData(),
                        &epsilon,
                        dx.memory().deviceData(),
                        dscale.memory().deviceData(),
                        dbias.memory().deviceData()});
    dx.memory().markDeviceModified();
    dscale.memory().markDeviceModified();
    dbias.memory().markDeviceModified();
    v8_sample::ranVariant(variant.str());

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";
        sdk::Tensor<T> dxRef(shapes.data, layout);
        sdk::Tensor<float> dscaleRef(shapes.affine, layout);
        sdk::Tensor<float> dbiasRef(shapes.affine, layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceLayernorm::bprop<T, float, T, float>(
            dy, x, scale, dxRef, dscaleRef, dbiasRef, EPSILON, &mean, &invVar, NORMALIZED_DIMS);

        const float tolerance = hipdnn_test_sdk::utilities::layernorm::getTolerance<T>();
        validationPassed = v8_sample::matches("dx", dxRef, dx, tolerance, tolerance);
        validationPassed = v8_sample::matches("dscale", dscaleRef, dscale, tolerance, tolerance)
                           && validationPassed;
        validationPassed = v8_sample::matches("dbias", dbiasRef, dbias, tolerance, tolerance)
                           && validationPassed;
    }

    std::cout << "v8 " << variant.str() << " complete.\n\n";
    return validationPassed;
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const bool forward = runForward<InputType>(handle, config, layout);
    const bool backward = runBackward<InputType>(handle, config, layout);
    return forward && backward;
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
