// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipblasltHandleFactory.hpp"
#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/ConvRepackFormulation.hpp"
#include "engines/conv/BenchmarkedPlan.hpp"
#include "engines/conv/ConvGemmPlanBuilder.hpp"
#include "engines/conv/ConvGemmTestGraphs.hpp"
#include "engines/conv/ConvReducedPrecision.hpp"
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
 *        and the im2col/col2im plans for everything else -- padding, stride, dilation,
 *        asymmetric padding, groups, depthwise, mixed layouts, true convolution, 1-D to 3-D --
 *        and the same plans run a slice of images at a time under a lowered workspace cap.
 *
 * The reference is cross-correlation only, so true convolution is checked against it with the
 * filter reflected on the host: the reference's input for forward and backward data, its
 * result for backward weights.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_test_sdk::utilities::CpuFpReferenceConvolution;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

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

    // Repacked: im2col (col2im for backward data) around the GEMM.
    cases.push_back({"OneByOneNchwWrw", nchw, ConvDirection::BACKWARD_WEIGHTS});
    cases.push_back({"OneByOneNchwGroupedWrw", nchwGrouped, ConvDirection::BACKWARD_WEIGHTS});

    ConvGemmShape padded;
    padded.c = 5;
    padded.k = 7;
    padded.filter = {3, 3};
    padded.prePadding = padded.postPadding = {1, 1};
    add("Padded3x3Nchw", padded);
    add("Padded3x3Nhwc", channelsLast(padded));

    ConvGemmShape mixed = padded;
    mixed.wChannelsLast = true;
    mixed.yChannelsLast = true;
    add("Padded3x3MixedLayouts", mixed);

    ConvGemmShape irregular;
    irregular.n = 3;
    irregular.c = 3;
    irregular.k = 4;
    irregular.input = {7, 8};
    irregular.filter = {3, 2};
    irregular.stride = {2, 1};
    irregular.dilation = {1, 2};
    irregular.prePadding = {1, 0};
    irregular.postPadding = {0, 2};
    add("StridedDilatedAsymmetricNchw", irregular);
    add("StridedDilatedAsymmetricNhwc", channelsLast(irregular));

    ConvGemmShape grouped = padded;
    grouped.c = 6;
    grouped.k = 4;
    grouped.groups = 2;
    add("Grouped3x3Nchw", grouped);
    add("Grouped3x3Nhwc", channelsLast(grouped));

    ConvGemmShape depthwise = channelsLast(padded);
    depthwise.c = depthwise.k = depthwise.groups = 4;
    add("Depthwise3x3Nhwc", depthwise);

    // Ten groups of eight channels across a 40-wide input: the staged grouped col2im takes
    // several strips and a partial last chunk of groups, through stride and dilation.
    ConvGemmShape groupedWide;
    groupedWide.n = 2;
    groupedWide.c = 80;
    groupedWide.k = 20;
    groupedWide.groups = 10;
    groupedWide.input = {12, 40};
    groupedWide.filter = {3, 3};
    groupedWide.stride = {1, 2};
    groupedWide.dilation = {2, 1};
    groupedWide.prePadding = {2, 1};
    groupedWide.postPadding = {2, 0};
    add("GroupedWideNhwc", channelsLast(groupedWide));

    // Disjoint taps (stride = filter): backward data's columns stay in dx's type.
    ConvGemmShape patchify;
    patchify.n = 2;
    patchify.c = 3;
    patchify.k = 8;
    patchify.input = {9, 12};
    patchify.filter = {3, 4};
    patchify.stride = {3, 4};
    add("PatchifyNchw", patchify);
    add("PatchifyNhwc", channelsLast(patchify));

    ConvGemmShape strided1x1 = nchw;
    strided1x1.stride = {2, 2};
    add("Strided1x1Nchw", strided1x1);

    ConvGemmShape convolution = irregular;
    convolution.mode = conv_data_objects::ConvMode::CONVOLUTION;
    add("ConvolutionModeNchw", convolution);
    add("ConvolutionModeNhwc", channelsLast(convolution));

    ConvGemmShape oneD;
    oneD.input = {11};
    oneD.filter = {3};
    oneD.prePadding = {2};
    oneD.postPadding = {1};
    oneD.stride = {2};
    oneD.dilation = {1};
    add("Conv1dNcw", oneD);

    ConvGemmShape threeD;
    threeD.c = 3;
    threeD.k = 4;
    threeD.input = {6, 4, 5};
    threeD.filter = {3, 3, 3};
    threeD.prePadding = threeD.postPadding = {1, 1, 1};
    threeD.stride = {1, 2, 1};
    threeD.dilation = {1, 1, 1};
    add("Conv3dNcdhw", threeD);
    add("Conv3dNdhwc", channelsLast(threeD));

    ConvGemmShape pointGroupedNhwc = channelsLast(pointGrouped);
    add("PointOutputNhwcGrouped", pointGroupedNhwc);

    // Images that do not nest into one stride: y is staged channels-last for the GEMM.
    ConvGemmShape gappedImages = channelsLast(padded);
    gappedImages.n = 3;
    gappedImages.imagePadding = 5;
    add("Padded3x3NhwcGappedImages", gappedImages);
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

int64_t offsetOf(const std::vector<int64_t>& index, const std::vector<int64_t>& strides)
{
    int64_t offset = 0;
    for(size_t axis = 0; axis < index.size(); ++axis)
    {
        offset += index[axis] * strides[axis];
    }
    return offset;
}

std::vector<std::vector<int64_t>> allIndices(const std::vector<int64_t>& dims);

/// @p filter's elements reflected along every spatial axis, into @p reflected.
template <typename T>
void reflectFilter(Tensor<T>& filter, Tensor<T>& reflected)
{
    const auto& dims = filter.dims();
    const auto* source = static_cast<const T*>(filter.rawHostData());
    auto* target = static_cast<T*>(reflected.rawHostData());
    for(const auto& index : allIndices(dims))
    {
        auto mirror = index;
        for(size_t axis = 2; axis < dims.size(); ++axis)
        {
            mirror[axis] = dims[axis] - 1 - index[axis];
        }
        target[offsetOf(mirror, reflected.strides())] = source[offsetOf(index, filter.strides())];
    }
    reflected.markHostModified();
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

/// The repack workspace cap that makes @p graph's plan run in several slices of images: one byte
/// short of what its IM2COL plan takes unsliced.
size_t slicingCap(const ConvGemmGraph& graph, ConvDirection direction)
{
    const auto problem = parseConvProblem(graph.graph(), direction);
    if(!problem.has_value())
    {
        return MAX_REPACK_WORKSPACE_BYTES;
    }
    // The unsliced IM2COL plan's own workspace: what slicing depends on. (The full plan may add
    // backward weights' batched partials, which a lower cap only drops.)
    const auto whole = im2colGemmCalls(*problem);
    if(!whole.has_value())
    {
        return MAX_REPACK_WORKSPACE_BYTES;
    }
    const size_t cap = whole->workspaceBytes - 1;
    const auto sliced = convGemmCalls(*problem, cap);
    EXPECT_TRUE(sliced.has_value() && sliced->stages.size() > 1)
        << "one byte under the unsliced workspace does not slice this case";
    return cap;
}

/// The repack workspace cap that makes @p graph's plan build each image in bands of rows: the
/// smallest cap, searched in steps of a thousandth of the unsliced workspace, that still plans
/// with more GEMM stages than images. nullopt when the case cannot band: backward data whose
/// outer axis has no band short of the whole output (a band's dx rows need (rows - 1) * stride
/// past the filter's reach), or a plan whose other workspace alone reaches one image's columns.
std::optional<size_t> bandingCap(const ConvGemmGraph& graph, ConvDirection direction)
{
    const auto problem = parseConvProblem(graph.graph(), direction);
    if(!problem.has_value())
    {
        ADD_FAILURE() << "the test graph is not a well-formed convolution";
        return std::nullopt;
    }
    const auto whole = convGemmCalls(*problem);
    if(!whole.has_value())
    {
        ADD_FAILURE() << "the formulation declined the convolution";
        return std::nullopt;
    }
    if(direction == ConvDirection::BACKWARD_DATA
       && (problem->output[0] - 2) * problem->stride[0]
              < (problem->filter[0] - 1) * problem->dilation[0])
    {
        return std::nullopt;
    }
    // The column matrix is the first workspace tensor, all images of it; whatever else the plan
    // keeps (a staged y, an fp32 accumulator) every band needs too. When that alone reaches one
    // image's columns, no cap under an image holds a band.
    if(!whole->workspace.empty())
    {
        const size_t imageColumns
            = whole->workspace.front().bytes / static_cast<size_t>(problem->n);
        if(whole->workspaceBytes - whole->workspace.front().bytes >= imageColumns)
        {
            return std::nullopt;
        }
    }
    const size_t step = std::max<size_t>(64, whole->workspaceBytes / 1024);
    for(size_t cap = step; cap < whole->workspaceBytes; cap += step)
    {
        const auto banded = convGemmCalls(*problem, cap);
        if(!banded.has_value())
        {
            continue;
        }
        const auto gemmStages
            = std::count_if(banded->stages.begin(), banded->stages.end(), [](const auto& stage) {
                  return !stage.calls.empty();
              });
        if(gemmStages > problem->n)
        {
            return cap;
        }
    }
    ADD_FAILURE() << "no cap under one image's workspace bands this case";
    return std::nullopt;
}

/// How runCase holds the repack workspace: uncapped, a slice of images at a time, or each image
/// in bands of rows.
enum class Slicing
{
    NONE,
    IMAGES,
    BANDS,
};

/// Runs @p convCase through @p backend and checks it against the CPU reference, its workspace
/// held as @p slicing says.
template <typename T>
void runCase(ConvCase convCase,
             Backend backend,
             Slicing slicing = Slicing::NONE,
             bool benchmarking = false,
             bool tf32 = false)
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

    size_t cap = MAX_REPACK_WORKSPACE_BYTES;
    if(slicing == Slicing::IMAGES)
    {
        cap = slicingCap(graph, convCase.direction);
    }
    else if(slicing == Slicing::BANDS)
    {
        const auto banding = bandingCap(graph, convCase.direction);
        if(!banding.has_value())
        {
            GTEST_SKIP() << "no band fits: an outer axis shorter than backward data's reach, or "
                            "other workspace as large as one image's columns";
        }
        cap = *banding;
    }
    const ConvGemmPlanBuilder builder(factoryFor(backend), cap, tf32);
    if(!builder.isApplicable(*handle, graph.graph()))
    {
        // The only decline a backend may add to the formulation's is finding no algorithm for
        // a GEMM it is given (hipBLASLt builds no bf16-in, fp32-out kernels for some targets).
        // Anything else is a failure.
        const auto problem = parseConvProblem(graph.graph(), convCase.direction);
        ASSERT_TRUE(problem.has_value());
        // The plan the builder formed: under the same workspace cap (images or bands).
        auto calls = convGemmCalls(*problem, cap);
        ASSERT_TRUE(calls.has_value()) << "the formulation declined the convolution";
        ASSERT_EQ(backend, Backend::HIPBLASLT) << "only hipBLASLt declines a formulated GEMM";
        bool threw = false;
        for(auto& stage : calls->stages)
        {
            try
            {
                static_cast<void>(factoryFor(backend)(*handle, stage.calls));
            }
            catch(const std::exception&)
            {
                threw = true;
            }
        }
        EXPECT_TRUE(threw) << "hipBLASLt took every GEMM, so the decline came from elsewhere";
        GTEST_SKIP() << "hipBLASLt has no algorithm on this device for a GEMM of this convolution";
    }
    HipdnnEnginePluginExecutionContext context;
    context.setBenchmarking(benchmarking);
    builder.buildPlan(*handle, graph.graph(), context);
    const auto* benchmarked = dynamic_cast<const BenchmarkedPlan*>(&context.plan());
    if(benchmarking)
    {
        ASSERT_NE(benchmarked, nullptr) << "a channels-first im2col case has two layouts to time";
    }
    const size_t workspaceBytes = context.plan().getWorkspaceSize(*handle);

    auto xStrides = convPackedStrides(shape.xDims(), shape.xChannelsLast);
    auto yStrides = convPackedStrides(shape.yDims(), shape.yChannelsLast);
    xStrides[0] += shape.imagePadding;
    yStrides[0] += shape.imagePadding;
    Tensor<T> x(shape.xDims(), xStrides);
    Tensor<T> w(shape.wDims(), convPackedStrides(shape.wDims(), shape.wChannelsLast));
    Tensor<T> y(shape.yDims(), yStrides);
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
    std::vector<char> firstOutput;
    if(benchmarked != nullptr)
    {
        // The first execute timed every candidate; the second runs only the winner. Both must
        // leave the same, checked, result.
        EXPECT_GE(benchmarked->winner(), 0);
        const size_t bytes = written->memory().count() * sizeof(T);
        firstOutput.resize(bytes);
        ASSERT_EQ(
            hipMemcpy(firstOutput.data(), written->rawDeviceData(), bytes, hipMemcpyDeviceToHost),
            hipSuccess);
        context.plan().execute(
            *handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        std::vector<char> secondOutput(bytes);
        ASSERT_EQ(
            hipMemcpy(secondOutput.data(), written->rawDeviceData(), bytes, hipMemcpyDeviceToHost),
            hipSuccess);
        EXPECT_EQ(firstOutput, secondOutput) << "the winner's rerun changed the result";
    }
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

    const bool reflect = shape.mode == conv_data_objects::ConvMode::CONVOLUTION;
    Tensor<T> reflectedW(w.dims(), w.strides());
    if(reflect && convCase.direction != ConvDirection::BACKWARD_WEIGHTS)
    {
        reflectFilter(w, reflectedW);
    }
    Tensor<T>& referenceW = reflect ? reflectedW : w;
    Tensor<T> reference(written->dims(), written->strides());
    switch(convCase.direction)
    {
    case ConvDirection::FORWARD:
        CpuFpReferenceConvolution::fprop(x,
                                         referenceW,
                                         reference,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
        break;
    case ConvDirection::BACKWARD_DATA:
        CpuFpReferenceConvolution::dgrad(reference,
                                         referenceW,
                                         y,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
        break;
    case ConvDirection::BACKWARD_WEIGHTS:
        // True convolution's dw is cross-correlation's reflected.
        CpuFpReferenceConvolution::wgrad(x,
                                         reflect ? reflectedW : reference,
                                         y,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
        if(reflect)
        {
            reflectFilter(reflectedW, reference);
        }
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
    // TF32 keeps 10 mantissa bits of each input, so with inputs in [-1, 1] a product may be off
    // by 2^-9 of 1 and the sum by that times the reduction.
    const float bound = (tf32 ? 0x1p-9f : 1e-6f) * static_cast<float>(reduction);

    int64_t mismatches = 0;
    float maxError = 0.0f;
    for(const auto& index : allIndices(written->dims()))
    {
        const float got = static_cast<float>(written->getHostValue(index));
        const float want = static_cast<float>(reference.getHostValue(index));
        const float tolerance = bound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
        maxError = std::max(maxError, std::fabs(got - want));
        if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
        {
            ADD_FAILURE() << backendName(backend) << ": got " << got << ", want " << want
                          << ", tolerance " << tolerance;
        }
    }
    EXPECT_EQ(mismatches, 0);
    if(tf32)
    {
        // Whether TF32 ran is the library's call; the error shows it (fp32 sits near 1e-6).
        std::cout << backendName(backend) << " TF32-permitted max error " << maxError << "\n";
        ::testing::Test::RecordProperty("max_error", std::to_string(maxError));
    }
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

/// The im2col cases of several images, run a slice of images at a time.
std::vector<ConvCase> slicedCases()
{
    std::vector<ConvCase> cases;
    for(const auto& convCase : convCases())
    {
        for(const auto* name : {"Padded3x3Nchw",
                                "Padded3x3Nhwc",
                                "Grouped3x3Nchw",
                                "StridedDilatedAsymmetricNchw",
                                "ConvolutionModeNhwc",
                                "Conv3dNcdhw",
                                "Padded3x3NhwcGappedImages"})
        {
            if(convCase.name.rfind(name, 0) == 0
               && convCase.name.size() == std::string(name).size() + 3)
            {
                cases.push_back(convCase);
            }
        }
    }
    return cases;
}

class TestGpuConvGemmSliced : public ::testing::TestWithParam<std::tuple<ConvCase, Backend>>
{
};

TEST_P(TestGpuConvGemmSliced, Fp32)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), Slicing::IMAGES);
}

TEST_P(TestGpuConvGemmSliced, Fp16)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(std::get<0>(GetParam()), std::get<1>(GetParam()), Slicing::IMAGES);
}

TEST_P(TestGpuConvGemmSliced, Bfp16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), Slicing::IMAGES);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGemmSliced,
                         ::testing::Combine(::testing::ValuesIn(slicedCases()),
                                            ::testing::ValuesIn(enabledBackends())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + backendName(std::get<1>(info.param));
                         });

/// The im2col cases whose single image is built in bands of rows: both layouts, stride,
/// dilation and asymmetric padding along the banded axis, grouped, 1-D and 3-D, and true
/// convolution.
std::vector<ConvCase> bandedCases()
{
    std::vector<ConvCase> cases;
    for(const auto& convCase : convCases())
    {
        for(const auto* name : {"Padded3x3Nchw",
                                "Padded3x3Nhwc",
                                "StridedDilatedAsymmetricNchw",
                                "StridedDilatedAsymmetricNhwc",
                                "GroupedWideNhwc",
                                "ConvolutionModeNhwc",
                                "Conv1dNcw",
                                "Conv3dNcdhw",
                                "Conv3dNdhwc"})
        {
            if(convCase.name.rfind(name, 0) == 0
               && convCase.name.size() == std::string(name).size() + 3)
            {
                cases.push_back(convCase);
            }
        }
    }
    return cases;
}

class TestGpuConvGemmBanded : public ::testing::TestWithParam<std::tuple<ConvCase, Backend>>
{
};

TEST_P(TestGpuConvGemmBanded, Fp32)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), Slicing::BANDS);
}

TEST_P(TestGpuConvGemmBanded, Bfp16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), Slicing::BANDS);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGemmBanded,
                         ::testing::Combine(::testing::ValuesIn(bandedCases()),
                                            ::testing::ValuesIn(enabledBackends())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + backendName(std::get<1>(info.param));
                         });

/// The cases with a choice of layout, built with benchmarking on: channels-first im2col (column
/// matrix either way) and the 1x1 channels-first backward weights (batched partials or staged
/// channels-last). The plan times each and keeps the fastest.
std::vector<ConvCase> benchmarkedCases()
{
    std::vector<ConvCase> cases;
    for(const auto& convCase : convCases())
    {
        for(const auto* name : {"Padded3x3Nchw",
                                "Grouped3x3Nchw",
                                "StridedDilatedAsymmetricNchw",
                                "Conv3dNcdhw",
                                "Conv1dNcw"})
        {
            if(convCase.name.rfind(name, 0) == 0
               && convCase.name.size() == std::string(name).size() + 3)
            {
                cases.push_back(convCase);
            }
        }
        // A 1x1 channels-first backward weights of several images: batched partials or x and dy
        // staged channels-last.
        if(convCase.name == "OneByOneNchwWrw")
        {
            cases.push_back(convCase);
        }
    }
    return cases;
}

class TestGpuConvGemmBenchmarked : public ::testing::TestWithParam<std::tuple<ConvCase, Backend>>
{
};

TEST_P(TestGpuConvGemmBenchmarked, Fp32)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), Slicing::NONE, true);
}

TEST_P(TestGpuConvGemmBenchmarked, Bfp16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()), Slicing::NONE, true);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGemmBenchmarked,
                         ::testing::Combine(::testing::ValuesIn(benchmarkedCases()),
                                            ::testing::ValuesIn(enabledBackends())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + backendName(std::get<1>(info.param));
                         });

/// fp32 cases with TF32 permitted, as the TF32 engines build them: each direct row, im2col in
/// both layouts, grouped and 3-D. The result holds to TF32's precision whether or not the GPU
/// ran TF32.
std::vector<ConvCase> tf32Cases()
{
    std::vector<ConvCase> cases;
    for(const auto& convCase : convCases())
    {
        for(const auto* name : {"OneByOneNhwc",
                                "OneByOneNchw",
                                "PointOutputNchw",
                                "Padded3x3Nchw",
                                "Padded3x3Nhwc",
                                "GroupedWideNhwc",
                                "Conv3dNdhwc"})
        {
            if(convCase.name.rfind(name, 0) == 0
               && convCase.name.size() == std::string(name).size() + 3)
            {
                cases.push_back(convCase);
            }
        }
    }
    return cases;
}

class TestGpuConvGemmTf32 : public ::testing::TestWithParam<std::tuple<ConvCase, Backend>>
{
};

TEST_P(TestGpuConvGemmTf32, Fp32)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), Slicing::NONE, false, true);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGemmTf32,
                         ::testing::Combine(::testing::ValuesIn(tf32Cases()),
                                            ::testing::ValuesIn(enabledBackends())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + backendName(std::get<1>(info.param));
                         });

/// Reduced-precision-reduction cases (reducedPrecisionGemmCalls): 1x1 fp16 backward weights of
/// several channels-first images, grouped, 3-D, unnested images, and a long reduction over many
/// images, where each image's rounding into dw adds up.
std::vector<ConvCase> rprCases()
{
    std::vector<ConvCase> cases;
    const auto add = [&cases](const std::string& name, ConvGemmShape shape) {
        cases.push_back({name, std::move(shape), ConvDirection::BACKWARD_WEIGHTS});
    };
    ConvGemmShape base;
    base.n = 8;
    base.c = 24;
    base.k = 40;
    base.input = {9, 11};
    add("Nchw", base);
    auto grouped = base;
    grouped.groups = 4;
    add("GroupedNchw", grouped);
    auto threeD = base;
    threeD.input = {3, 5, 6};
    threeD.filter = {1, 1, 1};
    threeD.prePadding = threeD.postPadding = {0, 0, 0};
    threeD.stride = threeD.dilation = {1, 1, 1};
    add("Ncdhw", threeD);
    auto padded = base;
    padded.imagePadding = 5;
    add("UnnestedImagesNchw", padded);
    auto deep = base;
    deep.n = 64;
    deep.c = 8;
    deep.k = 16;
    deep.input = {32, 32};
    add("ManyImagesNchw", deep);
    return cases;
}

/**
 * The fp64 dw of a 1x1 backward weights, checked against the fp16 dw @p backend wrote under the
 * reduced-precision-reduction tolerance: Adam Dickin's model (rocm-libraries #15481) for one
 * output rounding, |got - want| <= 2 eps |want| + (sqrt(K) u32 + 12 eps) sigma + minNormal,
 * widened by 3 eps sqrt(R) sigma for the R = N roundings of the running sum into fp16. K = N *
 * pixels products, sigma = sqrt(K E[x^2] E[dy^2]) = sqrt(K) / 3 for inputs in [-1, 1].
 */
void runRprCase(ConvCase convCase, Backend backend)
{
    auto& shape = convCase.shape;
    shape.dataType = conv_data_objects::DataType::HALF;
    const ConvGemmGraph graph(shape, convCase.direction);
    hipdnnEnginePluginHandle_t rawHandle = nullptr;
    HipblasltHandleFactory::createHipblasltHandle(&rawHandle);
    const std::unique_ptr<HipdnnEnginePluginHandle, void (*)(HipdnnEnginePluginHandle*)> handle(
        rawHandle, [](HipdnnEnginePluginHandle* owned) {
            HipblasltHandleFactory::destroyHipblasltHandle(owned);
            delete owned;
        });
    const ConvGemmPlanBuilder builder(factoryFor(backend), MAX_REPACK_WORKSPACE_BYTES, false, true);
    ScopedEnvironmentVariableSetter env(CONV_RPR_ENV_NAME);
    ASSERT_FALSE(builder.isApplicable(*handle, graph.graph()))
        << "a reduced-precision-reduction plan without the switch";
    env.setValue("1");
    if(!builder.isApplicable(*handle, graph.graph()))
    {
        ASSERT_EQ(backend, Backend::HIPBLASLT) << "only hipBLASLt declines a formulated GEMM";
        GTEST_SKIP() << "hipBLASLt has no fp16-out algorithm on this device for these GEMMs";
    }
    HipdnnEnginePluginExecutionContext context;
    builder.buildPlan(*handle, graph.graph(), context);
    ASSERT_EQ(context.plan().getWorkspaceSize(*handle), 0u);

    auto xStrides = convPackedStrides(shape.xDims(), false);
    auto yStrides = convPackedStrides(shape.yDims(), false);
    xStrides[0] += shape.imagePadding;
    yStrides[0] += shape.imagePadding;
    Tensor<Half> x(shape.xDims(), xStrides);
    Tensor<Half> w(shape.wDims(), convPackedStrides(shape.wDims(), false));
    Tensor<Half> y(shape.yDims(), yStrides);
    x.fillTensorWithRandomValues(-1.0f, 1.0f, 21);
    y.fillTensorWithRandomValues(-1.0f, 1.0f, 22);
    w.fillWithSentinelValue();
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    handle->setStream(stream);
    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
        hipdnnPluginDeviceBuffer_t{CONV_X_UID, x.rawDeviceData()},
        hipdnnPluginDeviceBuffer_t{CONV_W_UID, w.rawDeviceData()},
        hipdnnPluginDeviceBuffer_t{CONV_Y_UID, y.rawDeviceData()}};
    context.plan().execute(*handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    w.markDeviceModified();
    for(auto* tensor : {&x, &w, &y})
    {
        static_cast<void>(tensor->rawHostData());
    }

    std::vector<int64_t> spatial(shape.input.begin(), shape.input.end());
    const auto positions = allIndices(spatial);
    const int64_t cg = shape.c / shape.groups;
    const int64_t kg = shape.k / shape.groups;
    const double k = static_cast<double>(shape.n) * static_cast<double>(positions.size());
    const double eps = 0x1p-10;
    const double sigma = std::sqrt(k) / 3.0;
    const double atol = (std::sqrt(k) * 0x1p-24 + 12.0 * eps
                         + 3.0 * eps * std::sqrt(static_cast<double>(shape.n)))
                            * sigma
                        + 0x1p-14;
    int64_t mismatches = 0;
    double worst = 0.0;
    for(int64_t outK = 0; outK < shape.k; ++outK)
    {
        const int64_t group = outK / kg;
        for(int64_t inC = 0; inC < cg; ++inC)
        {
            double want = 0.0;
            for(int64_t image = 0; image < shape.n; ++image)
            {
                for(const auto& position : positions)
                {
                    std::vector<int64_t> xi{image, group * cg + inC};
                    std::vector<int64_t> yi{image, outK};
                    xi.insert(xi.end(), position.begin(), position.end());
                    yi.insert(yi.end(), position.begin(), position.end());
                    want += static_cast<double>(static_cast<float>(x.getHostValue(xi)))
                            * static_cast<double>(static_cast<float>(y.getHostValue(yi)));
                }
            }
            std::vector<int64_t> wi(shape.wDims().size(), 0);
            wi[0] = outK;
            wi[1] = inC;
            const double got = static_cast<double>(static_cast<float>(w.getHostValue(wi)));
            const double tolerance = 2.0 * eps * std::fabs(want) + atol;
            worst = std::max(worst, std::fabs(got - want) / tolerance);
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << backendName(backend) << ": dw[" << outK << "][" << inC << "] got "
                              << got << ", want " << want << ", tolerance " << tolerance;
            }
        }
    }
    EXPECT_EQ(mismatches, 0);
    ::testing::Test::RecordProperty("worst_error_over_tolerance", std::to_string(worst));
}

class TestGpuConvGemmReducedPrecision
    : public ::testing::TestWithParam<std::tuple<ConvCase, Backend>>
{
};

TEST_P(TestGpuConvGemmReducedPrecision, Fp16)
{
    SKIP_IF_NO_DEVICES();
    runRprCase(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGemmReducedPrecision,
                         ::testing::Combine(::testing::ValuesIn(rprCases()),
                                            ::testing::ValuesIn(enabledBackends())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + backendName(std::get<1>(info.param));
                         });

/// A value in [-1, 1) with 8 significant bits (exact in fp16, bf16 and fp32) for element @p index
/// of a tensor seeded @p seed: what the large tests fill and what their reference reads back.
double patternValue(uint64_t index, uint64_t seed)
{
    uint64_t h = (index + 1) * 0x9E3779B97F4A7C15ULL ^ seed * 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 29;
    return static_cast<double>(static_cast<int>(h & 0xFF) - 128) / 128.0;
}

/// Device memory of @p elements fp16 holding patternValue(i, @p seed), filled in chunks.
void* patternBuffer(size_t elements, uint64_t seed)
{
    void* device = nullptr;
    EXPECT_EQ(hipMalloc(&device, elements * sizeof(Half)), hipSuccess);
    constexpr size_t CHUNK = size_t{1} << 26;
    std::vector<Half> chunk(std::min(CHUNK, elements));
    for(size_t first = 0; first < elements; first += CHUNK)
    {
        const size_t count = std::min(CHUNK, elements - first);
        for(size_t i = 0; i < count; ++i)
        {
            chunk[i] = static_cast<Half>(static_cast<float>(patternValue(first + i, seed)));
        }
        EXPECT_EQ(hipMemcpy(static_cast<Half*>(device) + first,
                            chunk.data(),
                            count * sizeof(Half),
                            hipMemcpyHostToDevice),
                  hipSuccess);
    }
    return device;
}

/// Elements a packed tensor of @p dims holds.
size_t elementsOf(const std::vector<int64_t>& dims)
{
    size_t count = 1;
    for(const auto extent : dims)
    {
        count *= static_cast<size_t>(extent);
    }
    return count;
}

/// Opt-in: tensors of several GiB, for a GPU with the memory (HIPBLASLT_PROVIDER_LARGE_TESTS=1).
bool largeTestsEnabled()
{
    const char* value = std::getenv("HIPBLASLT_PROVIDER_LARGE_TESTS");
    return value != nullptr && std::string(value) == "1";
}

struct LargeCase
{
    std::string name;
    ConvGemmShape shape;
    ConvDirection direction;
};

/// fp16 convolutions past the 32-bit limits, each through a different guard: an operand over
/// 4 GiB, a GEMM n past 2^30, a batch stride past 2^31, and an image whose column matrix needs
/// bands (forward and backward data).
std::vector<LargeCase> largeCases()
{
    std::vector<LargeCase> cases;
    ConvGemmShape over4GiB = channelsLast({});
    over4GiB.n = 16;
    over4GiB.c = 160;
    over4GiB.k = 8;
    over4GiB.input = {1024, 1024}; // x: 5.4 GB
    cases.push_back({"OneByOneNhwcOver4GiB", over4GiB, ConvDirection::FORWARD});

    ConvGemmShape longN = channelsLast({});
    longN.n = 1;
    longN.c = longN.k = 2;
    longN.input = {40000, 40000}; // 1.6 G positions: a GEMM n past 2^30
    cases.push_back({"OneByOneNhwcPast2To30Positions", longN, ConvDirection::FORWARD});

    ConvGemmShape wideBatch;
    wideBatch.n = 2;
    wideBatch.c = wideBatch.k = 2;
    wideBatch.input = {32768, 36000}; // 2.4 G elements per image: a batch stride past 2^31
    cases.push_back({"OneByOneNchwImagesPast2To31", wideBatch, ConvDirection::FORWARD});

    ConvGemmShape banded = channelsLast({});
    banded.n = 1;
    banded.c = 32;
    banded.k = 16;
    banded.input = {4096, 4096}; // a 9.7 GB fp16 (19 GB fp32) column matrix
    banded.filter = {3, 3};
    banded.prePadding = banded.postPadding = {1, 1};
    cases.push_back({"Banded3x3NhwcFwd", banded, ConvDirection::FORWARD});
    cases.push_back({"Banded3x3NhwcBwd", banded, ConvDirection::BACKWARD_DATA});
    return cases;
}

class TestGpuConvGemmLarge : public ::testing::TestWithParam<std::tuple<LargeCase, Backend>>
{
};

TEST_P(TestGpuConvGemmLarge, Fp16Sampled)
{
    SKIP_IF_NO_DEVICES();
    if(!largeTestsEnabled())
    {
        GTEST_SKIP() << "set HIPBLASLT_PROVIDER_LARGE_TESTS=1 to run tensors of several GiB";
    }
    auto [largeCase, backend] = GetParam();
    auto& shape = largeCase.shape;
    shape.dataType = conv_data_objects::DataType::HALF;
    const bool forward = largeCase.direction == ConvDirection::FORWARD;
    const auto xDims = shape.xDims();
    const auto wDims = shape.wDims();
    const auto yDims = shape.yDims();
    const auto xStrides = convPackedStrides(xDims, shape.xChannelsLast);
    const auto wStrides = convPackedStrides(wDims, shape.wChannelsLast);
    const auto yStrides = convPackedStrides(yDims, shape.yChannelsLast);
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);
    const size_t needed = (elementsOf(xDims) + elementsOf(yDims)) * sizeof(Half)
                          + MAX_REPACK_WORKSPACE_BYTES + (size_t{4} << 30);
    if(freeBytes < needed)
    {
        GTEST_SKIP() << "needs " << needed << " bytes of device memory, " << freeBytes << " free";
    }

    hipdnnEnginePluginHandle_t rawHandle = nullptr;
    HipblasltHandleFactory::createHipblasltHandle(&rawHandle);
    const std::unique_ptr<HipdnnEnginePluginHandle, void (*)(HipdnnEnginePluginHandle*)> handle(
        rawHandle, [](HipdnnEnginePluginHandle* owned) {
            HipblasltHandleFactory::destroyHipblasltHandle(owned);
            delete owned;
        });
    const ConvGemmGraph graph(shape, largeCase.direction);
    const ConvGemmPlanBuilder builder(factoryFor(backend));
    if(!builder.isApplicable(*handle, graph.graph()))
    {
        // As in runCase: only hipBLASLt may decline a formulated GEMM, by finding no kernel
        // (a leading dimension past its kernels' 32-bit offsets, say); rocBLAS serves it.
        ASSERT_EQ(backend, Backend::HIPBLASLT) << "declined";
        GTEST_SKIP() << "hipBLASLt has no algorithm for a GEMM of this convolution";
    }
    HipdnnEnginePluginExecutionContext context;
    builder.buildPlan(*handle, graph.graph(), context);

    // The input the direction reads is the pattern; the one it writes starts as zeros.
    void* x = forward ? patternBuffer(elementsOf(xDims), 1) : nullptr;
    void* y = forward ? nullptr : patternBuffer(elementsOf(yDims), 3);
    void*& written = forward ? y : x;
    const size_t writtenElements = elementsOf(forward ? yDims : xDims);
    ASSERT_EQ(hipMalloc(&written, writtenElements * sizeof(Half)), hipSuccess);
    ASSERT_EQ(hipMemset(written, 0, writtenElements * sizeof(Half)), hipSuccess);
    void* w = patternBuffer(elementsOf(wDims), 2);
    void* workspace = nullptr;
    const size_t workspaceBytes = context.plan().getWorkspaceSize(*handle);
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
    }
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    handle->setStream(stream);
    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
        hipdnnPluginDeviceBuffer_t{CONV_X_UID, x},
        hipdnnPluginDeviceBuffer_t{CONV_W_UID, w},
        hipdnnPluginDeviceBuffer_t{CONV_Y_UID, y}};
    context.plan().execute(
        *handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    // Sampled outputs, including the first and last element, against a double-precision sum.
    const auto offset = [](const std::vector<int64_t>& strides, const std::array<int64_t, 4>& at) {
        return at[0] * strides[0] + at[1] * strides[1] + at[2] * strides[2] + at[3] * strides[3];
    };
    const auto& outDims = forward ? yDims : xDims;
    const int64_t reduction = (forward ? shape.c : shape.k) * shape.filter[0] * shape.filter[1];
    std::mt19937_64 random(7);
    int64_t mismatches = 0;
    for(int sample = 0; sample < 2048; ++sample)
    {
        std::array<int64_t, 4> at{};
        for(size_t axis = 0; axis < 4; ++axis)
        {
            at[axis] = sample == 0 ? 0
                       : sample == 1
                           ? outDims[axis] - 1
                           : static_cast<int64_t>(random() % static_cast<uint64_t>(outDims[axis]));
        }
        double want = 0.0;
        for(int64_t inner = 0; inner < (forward ? shape.c : shape.k); ++inner)
        {
            for(int64_t r = 0; r < shape.filter[0]; ++r)
            {
                for(int64_t t = 0; t < shape.filter[1]; ++t)
                {
                    // Forward reads x at (oy - pad + r); backward data reads dy at
                    // (iy + pad - r), stride and dilation 1 in every large case.
                    const int64_t row = forward ? at[2] - shape.prePadding[0] + r
                                                : at[2] + shape.prePadding[0] - r;
                    const int64_t col = forward ? at[3] - shape.prePadding[1] + t
                                                : at[3] + shape.prePadding[1] - t;
                    const auto& source = forward ? xDims : yDims;
                    if(row < 0 || row >= source[2] || col < 0 || col >= source[3])
                    {
                        continue;
                    }
                    const int64_t k = forward ? at[1] : inner;
                    const int64_t c = forward ? inner : at[1];
                    const double weight
                        = patternValue(static_cast<uint64_t>(offset(wStrides, {k, c, r, t})), 2);
                    const double input
                        = forward ? patternValue(static_cast<uint64_t>(
                                                     offset(xStrides, {at[0], inner, row, col})),
                                                 1)
                                  : patternValue(static_cast<uint64_t>(
                                                     offset(yStrides, {at[0], inner, row, col})),
                                                 3);
                    want += weight * input;
                }
            }
        }
        Half gotBits;
        const auto element = offset(forward ? yStrides : xStrides, at);
        ASSERT_EQ(hipMemcpy(&gotBits,
                            static_cast<Half*>(written) + element,
                            sizeof(Half),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        const double got = static_cast<double>(static_cast<float>(gotBits));
        const double tolerance
            = 1e-6 * static_cast<double>(reduction)
              + 2.0 * static_cast<double>(storageEpsilon<Half>()) * (std::fabs(want) + 1e-3);
        if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
        {
            ADD_FAILURE() << backendName(backend) << " at (" << at[0] << ", " << at[1] << ", "
                          << at[2] << ", " << at[3] << "): got " << got << ", want " << want;
        }
    }
    EXPECT_EQ(mismatches, 0);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    for(void* buffer : {x, w, y, workspace})
    {
        if(buffer != nullptr)
        {
            ASSERT_EQ(hipFree(buffer), hipSuccess);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGemmLarge,
                         ::testing::Combine(::testing::ValuesIn(largeCases()),
                                            ::testing::ValuesIn(enabledBackends())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + backendName(std::get<1>(info.param));
                         });

} // namespace
