// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Batch norm forward training (with running statistics) and backward built with the cuDNN v8
// (v0.x) norm builders, run through the shim on a real provider and checked against the CPU
// reference.

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
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
constexpr int64_t MOMENTUM_UID = 'f';
constexpr int64_t MEAN_UID = 'm';
constexpr int64_t INV_VAR_UID = 'v';
constexpr int64_t PREV_MEAN_UID = 'p';
constexpr int64_t PREV_VAR_UID = 'q';
constexpr int64_t NEXT_MEAN_UID = 'n';
constexpr int64_t NEXT_VAR_UID = 'o';

constexpr float EPSILON = 1e-5f;
constexpr float MOMENTUM = 0.1f;

std::vector<int64_t> dataDims(const Config& config)
{
    const auto pick = [&](size_t index, int64_t fallback) {
        return config.dims.size() > index ? config.dims[index] : fallback;
    };
    return {pick(0, 4), pick(1, 16), pick(2, 8), pick(3, 8)};
}

using v8_sample::makeFloatTensor;

template <typename T>
bool runTraining(cudnnHandle_t handle, const Config& config, const TensorLayout& layout)
{
    std::ostringstream variant;
    variant << "batch norm forward training " << hipdnn_frontend::getDataTypeEnumFromType<T>()
            << " [" << layout << "]";
    std::cout << "Running v8 " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const std::vector<int64_t> dims = dataDims(config);
    const std::vector<int64_t> channelDims = sdk::getDerivedShape(dims);
    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<T>();

    const fe::Operation op
        = fe::OperationBuilder(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
              .setNormalizationMode(CUDNN_BATCH_NORM)
              .setNormFwdPhase(CUDNN_NORM_FWD_TRAINING)
              .setxDesc(v8_sample::makeTensor(X_UID, dims, layout, ioType))
              .setScaleAndBias(makeFloatTensor(SCALE_UID, channelDims),
                               makeFloatTensor(BIAS_UID, channelDims))
              .setEpsilonTensor(v8_sample::makeScalar(EPSILON_UID))
              .setyDesc(v8_sample::makeTensor(Y_UID, dims, layout, ioType))
              .setSavedMeanAndInvVar(makeFloatTensor(MEAN_UID, channelDims),
                                     makeFloatTensor(INV_VAR_UID, channelDims))
              .setPrevRunningMeanAndVar(makeFloatTensor(PREV_MEAN_UID, channelDims),
                                        makeFloatTensor(PREV_VAR_UID, channelDims))
              .setExpDecayFactorTensor(v8_sample::makeScalar(MOMENTUM_UID))
              .setNextRunningMeanAndVar(makeFloatTensor(NEXT_MEAN_UID, channelDims),
                                        makeFloatTensor(NEXT_VAR_UID, channelDims))
              .build();
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<T> x(dims, layout);
    sdk::Tensor<T> y(dims, layout);
    sdk::Tensor<float> scale(channelDims);
    sdk::Tensor<float> bias(channelDims);
    sdk::Tensor<float> mean(channelDims);
    sdk::Tensor<float> invVar(channelDims);
    sdk::Tensor<float> prevMean(channelDims);
    sdk::Tensor<float> prevVar(channelDims);
    sdk::Tensor<float> nextMean(channelDims);
    sdk::Tensor<float> nextVar(channelDims);
    x.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    scale.fillWithRandomValues(-2.0f, 2.0f);
    bias.fillWithRandomValues(-2.0f, 2.0f);
    prevMean.fillWithRandomValues(-1.0f, 1.0f);
    prevVar.fillWithRandomValues(0.5f, 2.0f);
    y.fillWithValue(static_cast<T>(0.0f));
    for(sdk::Tensor<float>* output : {&mean, &invVar, &nextMean, &nextVar})
    {
        output->fillWithValue(0.0f);
    }

    float epsilon = EPSILON;
    float momentum = MOMENTUM;
    v8_sample::execute(handle,
                       *plan,
                       {X_UID,
                        SCALE_UID,
                        BIAS_UID,
                        EPSILON_UID,
                        Y_UID,
                        MEAN_UID,
                        INV_VAR_UID,
                        PREV_MEAN_UID,
                        PREV_VAR_UID,
                        MOMENTUM_UID,
                        NEXT_MEAN_UID,
                        NEXT_VAR_UID},
                       {x.memory().deviceData(),
                        scale.memory().deviceData(),
                        bias.memory().deviceData(),
                        &epsilon,
                        y.memory().deviceData(),
                        mean.memory().deviceData(),
                        invVar.memory().deviceData(),
                        prevMean.memory().deviceData(),
                        prevVar.memory().deviceData(),
                        &momentum,
                        nextMean.memory().deviceData(),
                        nextVar.memory().deviceData()});
    y.memory().markDeviceModified();
    for(sdk::Tensor<float>* output : {&mean, &invVar, &nextMean, &nextVar})
    {
        output->memory().markDeviceModified();
    }
    v8_sample::ranVariant(variant.str());

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";
        sdk::Tensor<T> yRef(dims, layout);
        sdk::Tensor<float> meanRef(channelDims);
        sdk::Tensor<float> invVarRef(channelDims);
        sdk::Tensor<float> nextMeanRef(channelDims);
        sdk::Tensor<float> nextVarRef(channelDims);
        hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::fwdTraining<T, float, float, T>(
            x,
            scale,
            bias,
            yRef,
            EPSILON,
            MOMENTUM,
            &meanRef,
            &invVarRef,
            &prevMean,
            &prevVar,
            &nextMeanRef,
            &nextVarRef);

        const float tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceTraining<T>();
        validationPassed = v8_sample::matches("y", yRef, y, tolerance, tolerance);
        validationPassed = v8_sample::matches("saved_mean", meanRef, mean, tolerance, tolerance)
                           && validationPassed;
        validationPassed
            = v8_sample::matches("saved_inv_variance", invVarRef, invVar, tolerance, tolerance)
              && validationPassed;
        validationPassed
            = v8_sample::matches("next_running_mean", nextMeanRef, nextMean, tolerance, tolerance)
              && validationPassed;
        validationPassed
            = v8_sample::matches("next_running_var", nextVarRef, nextVar, tolerance, tolerance)
              && validationPassed;
    }

    std::cout << "v8 " << variant.str() << " complete.\n\n";
    return validationPassed;
}

template <typename T>
bool runBackward(cudnnHandle_t handle, const Config& config, const TensorLayout& layout)
{
    std::ostringstream variant;
    variant << "batch norm backward " << hipdnn_frontend::getDataTypeEnumFromType<T>() << " ["
            << layout << "]";
    std::cout << "Running v8 " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const std::vector<int64_t> dims = dataDims(config);
    const std::vector<int64_t> channelDims = sdk::getDerivedShape(dims);
    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<T>();

    const fe::Operation op = fe::OperationBuilder(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
                                 .setNormalizationMode(CUDNN_BATCH_NORM)
                                 .setxDesc(v8_sample::makeTensor(X_UID, dims, layout, ioType))
                                 .setdyDesc(v8_sample::makeTensor(DY_UID, dims, layout, ioType))
                                 .setScale(makeFloatTensor(SCALE_UID, channelDims))
                                 .setSavedMeanAndInvVar(makeFloatTensor(MEAN_UID, channelDims),
                                                        makeFloatTensor(INV_VAR_UID, channelDims))
                                 .setEpsilonTensor(v8_sample::makeScalar(EPSILON_UID))
                                 .setdxDesc(v8_sample::makeTensor(DX_UID, dims, layout, ioType))
                                 .setDScaleAndDBias(makeFloatTensor(DSCALE_UID, channelDims),
                                                    makeFloatTensor(DBIAS_UID, channelDims))
                                 .build();
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<T> x(dims, layout);
    sdk::Tensor<T> dy(dims, layout);
    sdk::Tensor<T> dx(dims, layout);
    sdk::Tensor<float> scale(channelDims);
    sdk::Tensor<float> mean(channelDims);
    sdk::Tensor<float> invVar(channelDims);
    sdk::Tensor<float> dscale(channelDims);
    sdk::Tensor<float> dbias(channelDims);
    x.fillWithRandomValues(static_cast<T>(0.0f), static_cast<T>(1.0f));
    dy.fillWithRandomValues(static_cast<T>(0.0f), static_cast<T>(1.0f));
    dx.fillWithValue(static_cast<T>(0.0f));
    scale.fillWithRandomValues(0.0f, 1.0f);
    mean.fillWithRandomValues(0.0f, 1.0f);
    invVar.fillWithRandomValues(0.1f, 1.0f);
    dscale.fillWithValue(0.0f);
    dbias.fillWithValue(0.0f);

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
        sdk::Tensor<T> dxRef(dims, layout);
        sdk::Tensor<float> dscaleRef(channelDims);
        sdk::Tensor<float> dbiasRef(channelDims);
        hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::backward(
            dy, x, scale, dxRef, dscaleRef, dbiasRef, &mean, &invVar);

        const float tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceBackward<T>();
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
    const bool training = runTraining<InputType>(handle, config, layout);
    const bool backward = runBackward<InputType>(handle, config, layout);
    return training && backward;
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
