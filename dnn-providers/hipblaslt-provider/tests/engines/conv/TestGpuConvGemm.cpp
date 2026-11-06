// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipblasltHandleFactory.hpp"
#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "engines/conv/ConvGemmPlanBuilder.hpp"
#include "engines/conv/ConvGemmTestGraphs.hpp"
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
#include "engines/hipblaslt/HipblasltGemmPlan.hpp"
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
#include "engines/rocblas/RocblasGemmPlan.hpp"
#endif

/**
 * @file TestGpuConvGemm.cpp
 * @brief The explicit-GEMM convolution on the device, through each enabled backend, against
 *        hipDNN's CPU reference convolution: every direction, the 1x1 and point-output rows,
 *        channels-first and channels-last, groups and 3-D.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_test_sdk::utilities::CpuFpReferenceConvolution;

using Half = hipdnn_data_sdk::types::half;
using BFloat16 = hipdnn_data_sdk::types::bfloat16;

enum class Backend
{
    HIPBLASLT,
    ROCBLAS,
};

std::string backendName(Backend backend)
{
    return backend == Backend::HIPBLASLT ? "Hipblaslt" : "Rocblas";
}

ConvGemmPlanBuilder::PlanFactory factoryFor(Backend backend)
{
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    if(backend == Backend::HIPBLASLT)
    {
        return [](const HipdnnEnginePluginHandle& handle, std::vector<GemmCall> calls) {
            return std::make_unique<HipblasltGemmPlan>(handle, std::move(calls));
        };
    }
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    if(backend == Backend::ROCBLAS)
    {
        return [](const HipdnnEnginePluginHandle& /*handle*/, std::vector<GemmCall> calls) {
            return std::make_unique<RocblasGemmPlan>(std::move(calls));
        };
    }
#endif
    return nullptr;
}

std::vector<Backend> enabledBackends()
{
    std::vector<Backend> backends;
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    backends.push_back(Backend::HIPBLASLT);
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    backends.push_back(Backend::ROCBLAS);
#endif
    return backends;
}

struct ConvCase
{
    std::string name;
    ConvGemmShape shape;
    ConvDirection direction;
};

ConvGemmShape channelsLast(ConvGemmShape shape)
{
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = true;
    return shape;
}

std::vector<ConvCase> convCases()
{
    std::vector<ConvCase> cases;
    const auto add = [&cases](const std::string& name, const ConvGemmShape& shape) {
        cases.push_back({name + "Fwd", shape, ConvDirection::FORWARD});
        cases.push_back({name + "Bwd", shape, ConvDirection::BACKWARD_DATA});
        cases.push_back({name + "Wrw", shape, ConvDirection::BACKWARD_WEIGHTS});
    };

    ConvGemmShape nhwc = channelsLast({});
    nhwc.c = 24;
    nhwc.k = 40;
    add("OneByOneNhwc", nhwc);

    ConvGemmShape nhwcGrouped = channelsLast({});
    nhwcGrouped.c = 12;
    nhwcGrouped.k = 18;
    nhwcGrouped.groups = 3;
    add("OneByOneNhwcGrouped", nhwcGrouped);

    ConvGemmShape nchwSingle;
    nchwSingle.n = 1;
    nchwSingle.c = 16;
    nchwSingle.k = 20;
    add("OneByOneNchwOneImage", nchwSingle);

    ConvGemmShape nchw;
    nchw.n = 3;
    nchw.c = 16;
    nchw.k = 20;
    cases.push_back({"OneByOneNchwFwd", nchw, ConvDirection::FORWARD});
    cases.push_back({"OneByOneNchwBwd", nchw, ConvDirection::BACKWARD_DATA});

    ConvGemmShape nchwGrouped = nchw;
    nchwGrouped.groups = 4;
    cases.push_back({"OneByOneNchwGroupedFwd", nchwGrouped, ConvDirection::FORWARD});
    cases.push_back({"OneByOneNchwGroupedBwd", nchwGrouped, ConvDirection::BACKWARD_DATA});

    ConvGemmShape ndhwc = channelsLast({});
    ndhwc.input = {3, 4, 5};
    ndhwc.filter = {1, 1, 1};
    ndhwc.prePadding = ndhwc.postPadding = {0, 0, 0};
    ndhwc.stride = ndhwc.dilation = {1, 1, 1};
    add("OneByOneNdhwc", ndhwc);

    ConvGemmShape point;
    point.n = 5;
    point.c = 6;
    point.k = 8;
    point.input = {4, 3};
    point.filter = {4, 3};
    point.stride = {2, 3};
    add("PointOutputNchw", point);

    ConvGemmShape pointGrouped = point;
    pointGrouped.groups = 2;
    add("PointOutputNchwGrouped", pointGrouped);

    add("PointOutputNhwc", channelsLast(point));
    return cases;
}

template <typename T>
conv_data_objects::DataType dataTypeOf()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return conv_data_objects::DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return conv_data_objects::DataType::HALF;
    }
    else
    {
        return conv_data_objects::DataType::BFLOAT16;
    }
}

template <typename T>
float storageEpsilon()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1.2e-7f;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return 9.8e-4f;
    }
    else
    {
        return 7.9e-3f;
    }
}

std::vector<std::vector<int64_t>> allIndices(const std::vector<int64_t>& dims)
{
    std::vector<std::vector<int64_t>> indices;
    std::vector<int64_t> index(dims.size(), 0);
    while(true)
    {
        indices.push_back(index);
        size_t axis = dims.size();
        while(axis > 0)
        {
            --axis;
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
            if(axis == 0)
            {
                return indices;
            }
        }
    }
}

template <typename T>
void runCase(ConvCase convCase, Backend backend)
{
    auto& shape = convCase.shape;
    shape.dataType = dataTypeOf<T>();
    const ConvGemmGraph graph(shape, convCase.direction);

    hipdnnEnginePluginHandle_t rawHandle = nullptr;
    HipblasltHandleFactory::createHipblasltHandle(&rawHandle);
    const std::unique_ptr<HipdnnEnginePluginHandle, void (*)(HipdnnEnginePluginHandle*)> handle(
        rawHandle, [](HipdnnEnginePluginHandle* owned) {
            HipblasltHandleFactory::destroyHipblasltHandle(owned);
            delete owned;
        });

    const ConvGemmPlanBuilder builder(factoryFor(backend));
    ASSERT_TRUE(builder.isApplicable(*handle, graph.graph()));
    HipdnnEnginePluginExecutionContext context;
    builder.buildPlan(*handle, graph.graph(), context);
    const size_t workspaceBytes = context.plan().getWorkspaceSize(*handle);

    Tensor<T> x(shape.xDims(), convPackedStrides(shape.xDims(), shape.xChannelsLast));
    Tensor<T> w(shape.wDims(), convPackedStrides(shape.wDims(), shape.wChannelsLast));
    Tensor<T> y(shape.yDims(), convPackedStrides(shape.yDims(), shape.yChannelsLast));
    Tensor<T>* written = nullptr;
    switch(convCase.direction)
    {
    case ConvDirection::FORWARD:
        x.fillTensorWithRandomValues(-1.0f, 1.0f, 21);
        w.fillTensorWithRandomValues(-1.0f, 1.0f, 22);
        y.fillWithSentinelValue();
        written = &y;
        break;
    case ConvDirection::BACKWARD_DATA:
        y.fillTensorWithRandomValues(-1.0f, 1.0f, 21);
        w.fillTensorWithRandomValues(-1.0f, 1.0f, 22);
        x.fillWithSentinelValue();
        written = &x;
        break;
    case ConvDirection::BACKWARD_WEIGHTS:
        x.fillTensorWithRandomValues(-1.0f, 1.0f, 21);
        y.fillTensorWithRandomValues(-1.0f, 1.0f, 22);
        w.fillWithSentinelValue();
        written = &w;
        break;
    default:
        break;
    }
    ASSERT_NE(written, nullptr);

    void* workspace = nullptr;
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
    }
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    handle->setStream(stream);
    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
        hipdnnPluginDeviceBuffer_t{CONV_X_UID, x.rawDeviceData()},
        hipdnnPluginDeviceBuffer_t{CONV_W_UID, w.rawDeviceData()},
        hipdnnPluginDeviceBuffer_t{CONV_Y_UID, y.rawDeviceData()}};
    context.plan().execute(
        *handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    if(workspace != nullptr)
    {
        ASSERT_EQ(hipFree(workspace), hipSuccess);
    }
    written->markDeviceModified();
    for(auto* tensor : {&x, &w, &y})
    {
        static_cast<void>(tensor->rawHostData());
    }

    Tensor<T> reference(written->dims(), written->strides());
    switch(convCase.direction)
    {
    case ConvDirection::FORWARD:
        CpuFpReferenceConvolution::fprop(
            x, w, reference, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
        break;
    case ConvDirection::BACKWARD_DATA:
        CpuFpReferenceConvolution::dgrad(
            reference, w, y, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
        break;
    case ConvDirection::BACKWARD_WEIGHTS:
        CpuFpReferenceConvolution::wgrad(
            x, reference, y, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
        break;
    default:
        break;
    }

    int64_t filterElements = 1;
    int64_t inputElements = 1;
    for(size_t axis = 0; axis < shape.filter.size(); ++axis)
    {
        filterElements *= shape.filter[axis];
        inputElements *= shape.input[axis];
    }
    const int64_t reduction = convCase.direction == ConvDirection::BACKWARD_WEIGHTS
                                  ? shape.n * inputElements
                                  : std::max(shape.c, shape.k) * filterElements;
    const float bound = 1e-6f * static_cast<float>(reduction);

    int64_t mismatches = 0;
    for(const auto& index : allIndices(written->dims()))
    {
        const float got = static_cast<float>(written->getHostValue(index));
        const float want = static_cast<float>(reference.getHostValue(index));
        const float tolerance = bound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
        if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
        {
            ADD_FAILURE() << backendName(backend) << ": got " << got << ", want " << want
                          << ", tolerance " << tolerance;
        }
    }
    EXPECT_EQ(mismatches, 0);
}

class TestGpuConvGemm : public ::testing::TestWithParam<std::tuple<ConvCase, Backend>>
{
};

TEST_P(TestGpuConvGemm, Fp32)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(TestGpuConvGemm, Fp16)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(TestGpuConvGemm, Bfp16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGemm,
                         ::testing::Combine(::testing::ValuesIn(convCases()),
                                            ::testing::ValuesIn(enabledBackends())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + backendName(std::get<1>(info.param));
                         });

} // namespace
