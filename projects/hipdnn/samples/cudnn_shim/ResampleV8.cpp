// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// 2x2 average and max pooling, forward and backward, built with the cuDNN v8 (v0.x) resample
// builders, run through the shim on a real provider and checked against the CPU reference.

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleBwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleFwd.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "V8SampleUtils.hpp"

namespace fe = v8_sample::fe;
namespace sdk = hipdnn_data_sdk::utilities;

namespace
{

using ReferenceMode = hipdnn_flatbuffers_sdk::data_objects::ResampleMode;
using ReferencePadding = hipdnn_flatbuffers_sdk::data_objects::PaddingMode;

constexpr int64_t X_UID = 'x';
constexpr int64_t Y_UID = 'y';
constexpr int64_t DX_UID = 'X';
constexpr int64_t DY_UID = 'Y';
constexpr int64_t INDEX_UID = 'i';

// The cuDNN and reference modes are paired by hand; the enums are never cast.
struct Pooling
{
    const char* name;
    cudnnResampleMode_t mode;
    cudnnPaddingMode_t padding;
    ReferenceMode referenceMode;
    ReferencePadding referencePadding;

    bool withIndex() const
    {
        return mode == CUDNN_RESAMPLE_MAXPOOL;
    }
};

constexpr std::array<Pooling, 2> POOLINGS = {{
    {"average",
     CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING,
     CUDNN_ZERO_PAD,
     ReferenceMode::AVGPOOL_INCLUDE_PADDING,
     ReferencePadding::ZERO_PAD},
    {"max",
     CUDNN_RESAMPLE_MAXPOOL,
     CUDNN_NEG_INF_PAD,
     ReferenceMode::MAXPOOL,
     ReferencePadding::NEG_INF_PAD},
}};

// A 2x2 window with stride 2 and no padding.
const std::vector<int64_t> WINDOW = {2, 2};
const std::vector<int64_t> NO_PADDING = {0, 0};

struct Shapes
{
    std::vector<int64_t> x;
    std::vector<int64_t> y;
};

Shapes shapesFrom(const Config& config)
{
    const auto pick = [&](size_t index, int64_t fallback) {
        return config.dims.size() > index ? config.dims[index] : fallback;
    };
    const int64_t n = pick(0, 2);
    const int64_t c = pick(1, 8);
    const int64_t h = pick(2, 8);
    const int64_t w = pick(3, 8);
    return {{n, c, h, w}, {n, c, h / WINDOW[0], w / WINDOW[1]}};
}

fe::ResampleDesc makeResampleDesc(const Pooling& pooling)
{
    return fe::ResampleDescBuilder()
        .setResampleMode(pooling.mode)
        .setComputeType(CUDNN_DATA_FLOAT)
        .setPaddingMode(pooling.padding)
        .setSpatialDim(2, WINDOW.data())
        .setPrePadding(2, NO_PADDING.data())
        .setPostPadding(2, NO_PADDING.data())
        .setSpatialStride(2, WINDOW.data())
        .build();
}

fe::Operation makeForward(const Pooling& pooling,
                          const Shapes& shapes,
                          const TensorLayout& layout,
                          cudnnDataType_t ioType)
{
    fe::OperationBuilder builder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
    builder.setResampleDesc(makeResampleDesc(pooling))
        .setxDesc(v8_sample::makeTensor(X_UID, shapes.x, layout, ioType))
        .setyDesc(v8_sample::makeTensor(Y_UID, shapes.y, layout, ioType));
    if(pooling.withIndex())
    {
        builder.setidxDesc(v8_sample::makeTensor(INDEX_UID, shapes.y, layout, CUDNN_DATA_INT32));
    }
    return builder.build();
}

fe::Operation makeBackward(const Pooling& pooling,
                           const Shapes& shapes,
                           const TensorLayout& layout,
                           cudnnDataType_t ioType)
{
    fe::OperationBuilder builder(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
    builder.setResampleDesc(makeResampleDesc(pooling))
        .setdyDesc(v8_sample::makeTensor(DY_UID, shapes.y, layout, ioType))
        .setdxDesc(v8_sample::makeTensor(DX_UID, shapes.x, layout, ioType));
    if(pooling.withIndex())
    {
        builder.setidxDesc(v8_sample::makeTensor(INDEX_UID, shapes.y, layout, CUDNN_DATA_INT32));
    }
    return builder.build();
}

// Exactly representable in every type, and distinct inside each window, so the maximum has one
// position and a device and the reference cannot pick different ties.
template <typename T>
void fillTieFree(sdk::Tensor<T>& x)
{
    sdk::iterateAlongDimensions(x.dims(), [&x](const std::vector<int64_t>& index) {
        const int64_t n = index[0];
        const int64_t c = index[1];
        const int64_t h = index[2];
        const int64_t w = index[3];
        const int64_t base = ((n * 7 + c * 3 + (h / 2) * 5 + (w / 2)) % 9) - 4;
        const int64_t rank = ((h % 2) * 2 + (w % 2) + n + c + h / 2 + w / 2) % 4;
        x.setHostValue(static_cast<T>(static_cast<float>(base) + 0.25f * static_cast<float>(rank)),
                       index);
    });
    x.memory().markHostModified();
}

std::string variantName(const char* direction, const Pooling& pooling, const std::string& suffix)
{
    return std::string("resample ") + direction + ' ' + pooling.name + ' ' + suffix;
}

template <typename T>
std::string typeAndLayout(const TensorLayout& layout)
{
    std::ostringstream name;
    name << hipdnn_frontend::getDataTypeEnumFromType<T>() << " [" << layout << "]";
    return name.str();
}

template <typename T>
bool runForward(cudnnHandle_t handle,
                const Config& config,
                const TensorLayout& layout,
                const Pooling& pooling)
{
    const std::string variant = variantName("forward", pooling, typeAndLayout<T>(layout));
    std::cout << "Running v8 " << variant << (config.cpuValidation ? " (with CPU validation)" : "")
              << "...\n";

    const Shapes shapes = shapesFrom(config);
    const fe::Operation op = makeForward(pooling, shapes, layout, v8_sample::cudnnTypeOf<T>());
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant);
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<T> x(shapes.x, layout);
    sdk::Tensor<T> y(shapes.y, layout);
    sdk::Tensor<int32_t> index(shapes.y, layout);
    fillTieFree(x);
    y.fillWithValue(static_cast<T>(0.0f));
    index.fillWithValue(0);

    std::vector<int64_t> uids = {X_UID, Y_UID};
    std::vector<void*> pointers = {x.memory().deviceData(), y.memory().deviceData()};
    if(pooling.withIndex())
    {
        uids.push_back(INDEX_UID);
        pointers.push_back(index.memory().deviceData());
    }
    v8_sample::execute(handle, *plan, uids, pointers);
    y.memory().markDeviceModified();
    v8_sample::ranVariant(variant);

    // The index is checked through the max backward variant, which consumes it.
    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";
        sdk::Tensor<T> yRef(shapes.y, layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceResampleFwd::forward(
            x, yRef, NO_PADDING, WINDOW, WINDOW, pooling.referenceMode, pooling.referencePadding);
        const float tolerance = hipdnn_test_sdk::utilities::resample::getTolerance<T>();
        validationPassed = v8_sample::matches("y", yRef, y, tolerance, tolerance);
    }

    std::cout << "v8 " << variant << " complete.\n\n";
    return validationPassed;
}

// Max pooling's backward pass takes the index its forward pass wrote, so that variant runs the
// forward pass on the device first; the gradient then lands where the device found each maximum.
template <typename T>
bool runBackward(cudnnHandle_t handle,
                 const Config& config,
                 const TensorLayout& layout,
                 const Pooling& pooling)
{
    const std::string variant = variantName("backward", pooling, typeAndLayout<T>(layout));
    std::cout << "Running v8 " << variant << (config.cpuValidation ? " (with CPU validation)" : "")
              << "...\n";

    const Shapes shapes = shapesFrom(config);
    const cudnnDataType_t ioType = v8_sample::cudnnTypeOf<T>();
    const fe::Operation forwardOp = makeForward(pooling, shapes, layout, ioType);
    const fe::Operation backwardOp = makeBackward(pooling, shapes, layout, ioType);
    fe::OperationGraph forwardGraph = v8_sample::makeGraph(handle, {&forwardOp});
    fe::OperationGraph backwardGraph = v8_sample::makeGraph(handle, {&backwardOp});

    std::optional<fe::ExecutionPlan> forwardPlan;
    if(pooling.withIndex())
    {
        forwardPlan = v8_sample::firstPlan(handle, forwardGraph);
        if(!forwardPlan)
        {
            return v8_sample::skipVariant(variant + " (needs the forward plan for its index)");
        }
    }
    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, backwardGraph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant);
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<T> x(shapes.x, layout);
    sdk::Tensor<T> y(shapes.y, layout);
    sdk::Tensor<int32_t> index(shapes.y, layout);
    sdk::Tensor<T> dy(shapes.y, layout);
    sdk::Tensor<T> dx(shapes.x, layout);
    fillTieFree(x);
    index.fillWithValue(0);
    dy.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f));
    dx.fillWithValue(static_cast<T>(0.0f));

    std::vector<int64_t> uids = {DY_UID, DX_UID};
    std::vector<void*> pointers = {dy.memory().deviceData(), dx.memory().deviceData()};
    if(pooling.withIndex())
    {
        v8_sample::execute(
            handle,
            *forwardPlan,
            {X_UID, Y_UID, INDEX_UID},
            {x.memory().deviceData(), y.memory().deviceData(), index.memory().deviceData()});
        uids.push_back(INDEX_UID);
        pointers.push_back(index.memory().deviceData());
    }
    v8_sample::execute(handle, *plan, uids, pointers);
    dx.memory().markDeviceModified();
    v8_sample::ranVariant(variant);

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";
        sdk::Tensor<T> yRef(shapes.y, layout);
        sdk::Tensor<int32_t> indexRef(shapes.y, layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceResampleFwd::forward(x,
                                                                       yRef,
                                                                       NO_PADDING,
                                                                       WINDOW,
                                                                       WINDOW,
                                                                       pooling.referenceMode,
                                                                       pooling.referencePadding,
                                                                       &indexRef);
        sdk::Tensor<T> dxRef(shapes.x, layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceResampleBwd::backward(
            dy,
            dxRef,
            NO_PADDING,
            WINDOW,
            WINDOW,
            pooling.referenceMode,
            pooling.referencePadding,
            pooling.withIndex() ? &indexRef : nullptr);
        const float tolerance = hipdnn_test_sdk::utilities::resample::getTolerance<T>();
        validationPassed = v8_sample::matches("dx", dxRef, dx, tolerance, tolerance);
    }

    std::cout << "v8 " << variant << " complete.\n\n";
    return validationPassed;
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    bool allPassed = true;
    for(const Pooling& pooling : POOLINGS)
    {
        allPassed = runForward<InputType>(handle, config, layout, pooling) && allPassed;
        allPassed = runBackward<InputType>(handle, config, layout, pooling) && allPassed;
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
