// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvNaive.cpp
 * @brief The naive convolution kernels on the device, through the engines' own match,
 *        prepare and launch, against hipDNN's CPU reference convolution.
 *
 * The CPU reference computes cross-correlation only, so a convolution-mode case is checked
 * against it with the filter reflected on the host: the input to the reference for forward
 * and backward data, the reference's result for backward weights.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_test_sdk::utilities::CpuFpReferenceConvolution;

using Half = hipdnn_data_sdk::types::half;
using BFloat16 = hipdnn_data_sdk::types::bfloat16;

std::string directionTag(ConvDirection direction)
{
    switch(direction)
    {
    case ConvDirection::FORWARD:
        return "FWD";
    case ConvDirection::BACKWARD_DATA:
        return "BWD";
    case ConvDirection::BACKWARD_WEIGHTS:
        return "WRW";
    default:
        break;
    }
    return "";
}

PackSymbols packFor(ConvDirection direction)
{
    static const std::array<std::string, 3> s_graphMatchers{"hipkernel.conv_naive_fwd.graph_match",
                                                            "hipkernel.conv_naive_bwd.graph_match",
                                                            "hipkernel.conv_naive_wrw.graph_match"};
    return {"",
            s_graphMatchers[static_cast<size_t>(direction)],
            "",
            "hipkernel.conv_naive.kernel_match",
            "",
            "hipkernel.conv_naive.dispatch",
            "",
            "",
            ""};
}

hipdnn_plugin_sdk::ingestor::KernelDefinition naiveKernel(DataType dataType,
                                                          ConvDirection direction)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.name = "conv_naive_test";
    kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
    kernel.source.sourceFile = "kernels/ConvNaive.cpp";
    kernel.source.entryPoint
        = "ConvNaive"
          + std::string(direction == ConvDirection::FORWARD
                            ? "Fwd"
                            : (direction == ConvDirection::BACKWARD_DATA ? "Bwd" : "Wrw"));
    kernel.metadata
        = {{"dtype", std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType))},
           {"direction", directionTag(direction)}};
    return kernel;
}

/// Calls @p visit with every multi-index of @p dims, last axis fastest.
void forEachIndex(const std::vector<int64_t>& dims,
                  const std::function<void(const std::vector<int64_t>&)>& visit)
{
    std::vector<int64_t> index(dims.size(), 0);
    while(true)
    {
        visit(index);
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
                return;
            }
        }
    }
}

int64_t offsetOf(const std::vector<int64_t>& index, const std::vector<int64_t>& strides)
{
    int64_t offset = 0;
    for(size_t axis = 0; axis < index.size(); ++axis)
    {
        offset += index[axis] * strides[axis];
    }
    return offset;
}

/// @p filter's elements reflected along every spatial axis, into @p reflected.
template <typename T>
void reflectFilter(Tensor<T>& filter, Tensor<T>& reflected)
{
    const auto& dims = filter.dims();
    const auto* source = static_cast<const T*>(filter.rawHostData());
    auto* target = static_cast<T*>(reflected.rawHostData());
    forEachIndex(dims, [&](const std::vector<int64_t>& index) {
        auto mirror = index;
        for(size_t axis = 2; axis < dims.size(); ++axis)
        {
            mirror[axis] = dims[axis] - 1 - index[axis];
        }
        target[offsetOf(mirror, reflected.strides())] = source[offsetOf(index, filter.strides())];
    });
    reflected.markHostModified();
}

/// Unit roundoff of T's storage, the most one final rounding can move a value.
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

struct DeviceCase
{
    std::string name;
    ConvShape shape;
};

std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Nchw2d", ConvShape{}});

    ConvShape nhwc;
    nhwc.channelsLast = true;
    nhwc.prePadding = {1, 1};
    nhwc.postPadding = {1, 1};
    cases.push_back({"Nhwc2dPadded", nhwc});

    ConvShape oneD;
    oneD.input = {11};
    oneD.filter = {3};
    oneD.prePadding = {1};
    oneD.postPadding = {2};
    oneD.stride = {2};
    oneD.dilation = {2};
    cases.push_back({"OneD", oneD});

    ConvShape threeD;
    threeD.input = {4, 5, 6};
    threeD.filter = {2, 3, 2};
    threeD.prePadding = {0, 1, 1};
    threeD.postPadding = {1, 1, 0};
    threeD.stride = {1, 2, 1};
    threeD.dilation = {1, 1, 2};
    threeD.channelsLast = true;
    cases.push_back({"ThreeDChannelsLast", threeD});

    ConvShape grouped;
    grouped.c = 6;
    grouped.k = 9;
    grouped.groups = 3;
    grouped.prePadding = {1, 0};
    grouped.postPadding = {1, 0};
    cases.push_back({"Grouped", grouped});

    ConvShape depthwise;
    depthwise.c = 5;
    depthwise.k = 10;
    depthwise.groups = 5;
    depthwise.channelsLast = true;
    depthwise.input = {8, 9};
    depthwise.filter = {3, 3};
    depthwise.prePadding = {1, 1};
    depthwise.postPadding = {1, 1};
    cases.push_back({"DepthwiseMultiplier2", depthwise});

    ConvShape strided;
    strided.input = {12, 11};
    strided.filter = {3, 3};
    strided.prePadding = {2, 0};
    strided.postPadding = {1, 3};
    strided.stride = {2, 3};
    strided.dilation = {2, 1};
    cases.push_back({"AsymmetricPaddingStrideDilation", strided});

    // A long reduction (8 images of 1024, 8192 output positions a dw element): the backward
    // weights' block shares it, past the 5184 a lane the per-thread kernel took.
    ConvShape longReduction;
    longReduction.n = 8;
    longReduction.c = 16;
    longReduction.k = 16;
    longReduction.input = {1024};
    longReduction.filter = {3};
    longReduction.prePadding = {1};
    longReduction.postPadding = {1};
    longReduction.stride = {1};
    longReduction.dilation = {1};
    cases.push_back({"LongReduction", longReduction});

    ConvShape convolution;
    convolution.mode = ConvMode::CONVOLUTION;
    convolution.prePadding = {1, 1};
    convolution.postPadding = {0, 1};
    convolution.stride = {2, 1};
    cases.push_back({"ConvolutionMode", convolution});

    // Strided layers whose output extent is 1 on an axis (hipconv #15149 review: a stride
    // normalized for a size-1 axis computed the wrong output size).
    ConvShape toOne;
    toOne.input = {4, 4};
    toOne.filter = {3, 3};
    toOne.stride = {2, 2};
    toOne.channelsLast = true;
    cases.push_back({"Stride2OutputOne", toOne});
    ConvShape toOneDeep;
    toOneDeep.input = {3, 6, 5};
    toOneDeep.filter = {3, 3, 3};
    toOneDeep.prePadding = {0, 1, 1};
    toOneDeep.postPadding = {0, 1, 1};
    toOneDeep.stride = {2, 2, 2};
    toOneDeep.dilation = {1, 1, 1};
    cases.push_back({"ThreeDStride2OutputOneDeep", toOneDeep});
    return cases;
}

template <typename T>
DataType dataTypeOf()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return DataType::HALF;
    }
    else
    {
        return DataType::BFLOAT16;
    }
}

/// Builds the graph, runs the engine's match, prepare and launch, and compares the
/// written tensor against the CPU reference.
template <typename T>
void runCase(ConvShape shape, ConvDirection direction)
{
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, direction);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());

    const auto pack = packFor(direction);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";
    const auto kernel = naiveKernel(shape.dataType, direction);
    ASSERT_TRUE(matchesKernel(pack, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(pack);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    Tensor<T>* writtenTensor = nullptr;
    constexpr unsigned int SEED = 1234;
    switch(direction)
    {
    case ConvDirection::FORWARD:
        x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
        w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
        y.fillWithSentinelValue();
        writtenTensor = &y;
        break;
    case ConvDirection::BACKWARD_DATA:
        y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
        w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
        x.fillWithSentinelValue();
        writtenTensor = &x;
        break;
    case ConvDirection::BACKWARD_WEIGHTS:
        x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
        y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
        w.fillWithSentinelValue();
        writtenTensor = &w;
        break;
    default:
        break;
    }

    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
        hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
        hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
        hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};

    const Handle handle;
    handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
    ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
    writtenTensor->markDeviceModified();

    // The reference, in cross-correlation; convolution mode reflects the filter around it.
    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> reference(writtenTensor->dims(), writtenTensor->strides());
    Tensor<T> reflectedW(spec.w.dims, spec.w.strides);
    if(reflect && direction != ConvDirection::BACKWARD_WEIGHTS)
    {
        reflectFilter(w, reflectedW);
    }
    const Tensor<T>& filterForReference = reflect ? reflectedW : w;

    switch(direction)
    {
    case ConvDirection::FORWARD:
        CpuFpReferenceConvolution::fprop(x,
                                         filterForReference,
                                         reference,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
        break;
    case ConvDirection::BACKWARD_DATA:
        CpuFpReferenceConvolution::dgrad(reference,
                                         filterForReference,
                                         y,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
        break;
    case ConvDirection::BACKWARD_WEIGHTS:
        CpuFpReferenceConvolution::wgrad(
            x, reference, y, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
        if(reflect)
        {
            // The cross-correlation gradient of the reflected filter, reflected back.
            reflectFilter(reference, reflectedW);
        }
        break;
    default:
        break;
    }
    Tensor<T>& expectedTensor
        = reflect && direction == ConvDirection::BACKWARD_WEIGHTS ? reflectedW : reference;

    // The longest reduction any output element runs: every term is at most 1 in magnitude.
    const auto output = shape.output();
    int64_t filterElements = 1;
    int64_t outElements = 1;
    for(size_t axis = 0; axis < shape.filter.size(); ++axis)
    {
        filterElements *= shape.filter[axis];
        outElements *= output[axis];
    }
    const int64_t reduction = direction == ConvDirection::FORWARD
                                  ? (shape.c / shape.groups) * filterElements
                                  : (direction == ConvDirection::BACKWARD_DATA
                                         ? (shape.k / shape.groups) * filterElements
                                         : shape.n * outElements);
    const float accumulationBound = 1e-6f * static_cast<float>(reduction);

    const auto* actual = static_cast<const T*>(writtenTensor->rawHostData());
    const auto* expected = static_cast<const T*>(expectedTensor.rawHostData());
    int64_t mismatches = 0;
    forEachIndex(writtenTensor->dims(), [&](const std::vector<int64_t>& index) {
        const float got = static_cast<float>(actual[offsetOf(index, writtenTensor->strides())]);
        const float want = static_cast<float>(expected[offsetOf(index, expectedTensor.strides())]);
        const float tolerance
            = accumulationBound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
        if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
        {
            ADD_FAILURE() << "at linear index " << offsetOf(index, writtenTensor->strides())
                          << ": got " << got << ", want " << want << ", tolerance " << tolerance;
        }
    });
    EXPECT_EQ(mismatches, 0);
}

class TestGpuConvNaive : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvNaive, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvNaive, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvNaive, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvNaive,
                         ::testing::Combine(::testing::ValuesIn(deviceCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA,
                                                              ConvDirection::BACKWARD_WEIGHTS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + directionTag(std::get<1>(info.param));
                         });

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
