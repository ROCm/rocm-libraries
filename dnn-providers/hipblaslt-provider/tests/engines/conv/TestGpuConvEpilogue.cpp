// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
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
#include "common/ConvEpilogueFormulation.hpp"
#include "common/ConvRepackFormulation.hpp"
#include "engines/conv/ConvEpilogueTestGraphs.hpp"
#include "engines/conv/ConvGemmPlanBuilder.hpp"
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>

#include "HipblasltMatmulDesc.hpp"
#include "HipblasltMatrixLayout.hpp"
#include "HipblasltUtils.hpp"
#include "engines/hipblaslt/HipblasltGemmPlan.hpp"
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
#include "engines/rocblas/RocblasGemmPlan.hpp"
#endif

/**
 * @file TestGpuConvEpilogue.cpp
 * @brief Fused conv + pointwise graphs through the explicit-GEMM engines on the device, every
 *        form (hipBLASLt's native epilogue, the staged copy's fused cast, the extra pass), every
 *        backend, all three element types, against the CPU convolution in fp32 followed by the
 *        chain in double: the one rounding the fused plans promise.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;
using conv_data_objects::DataType;
using hipdnn_data_sdk::utilities::Tensor;

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

/// The plan factory of @p backend's engines, or (@p deterministic) of their *_DETERMINISTIC
/// siblings, as HipblasltContainer.cpp builds them.
ConvGemmPlanBuilder::PlanFactory factoryFor(Backend backend, bool deterministic = false)
{
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    if(backend == Backend::HIPBLASLT)
    {
        return [deterministic](const HipdnnEnginePluginHandle& handle,
                               const std::vector<GemmCall>& calls) {
            return std::make_unique<HipblasltGemmPlan>(handle, calls, deterministic);
        };
    }
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    if(backend == Backend::ROCBLAS)
    {
        return [deterministic](const HipdnnEnginePluginHandle& /*handle*/,
                               const std::vector<GemmCall>& calls) {
            auto plan = std::make_unique<RocblasGemmPlan>(calls);
            if(deterministic)
            {
                plan->requireBitwiseReproducible();
            }
            return plan;
        };
    }
#endif
    return nullptr;
}

/// What @p backend's engines apply of an epilogue in their GEMMs, as HipblasltContainer.cpp
/// builds them; @p libraryActivations: the opt-in HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE.
GemmEpilogueSupport gemmSupport(Backend backend, bool libraryActivations = false)
{
    return GemmEpilogueSupport{backend == Backend::HIPBLASLT,
                               backend == Backend::HIPBLASLT && libraryActivations};
}

/// One step as hipBLASLt's activation epilogues compute it (the library-activation engine):
/// the chain's values, except relu maps NaN to 0, a clamp (any other relu) NaN to its upper
/// bound, and swish and tanh GELU -inf to NaN.
double applyLibraryStep(const Step& step, double v, double input)
{
    if(step.mode == PointwiseMode::RELU_FWD && std::isnan(v))
    {
        const bool plain = valueOr(step.lowerClip, 0.0) == 0.0 && !step.upperClip.has_value();
        return plain ? 0.0 : valueOr(step.upperClip, std::numeric_limits<double>::infinity());
    }
    if((step.mode == PointwiseMode::SWISH_FWD || step.mode == PointwiseMode::GELU_APPROX_TANH_FWD)
       && std::isinf(v) && v < 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return applyStep(step, v, input);
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

template <typename T>
double unitRoundoff()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 6.0e-8;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return 4.9e-4;
    }
    else
    {
        return 3.9e-3;
    }
}

enum class Specials
{
    NONE,
    /// NaN and +-infinity in the per-channel operands of channels 1 to 3.
    NAN_INF,
    /// x all zero: every output is the chain of the operands alone, sitting on the clip bounds.
    CLIP_BOUNDARY,
};

struct EpilogueCase
{
    std::string name;
    ConvGemmShape shape;
    ColumnLayout layout = ColumnLayout::FOLLOW_Y;
};

ConvGemmShape layoutOf(ConvGemmShape shape, bool channelsLast)
{
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = channelsLast;
    return shape;
}

std::vector<EpilogueCase> epilogueCases()
{
    ConvGemmShape oneByOne;
    oneByOne.n = 2;
    oneByOne.c = 7;
    oneByOne.k = 13;
    oneByOne.input = {5, 9};
    ConvGemmShape threeByThree = oneByOne;
    threeByThree.filter = {3, 3};
    threeByThree.prePadding = {1, 1};
    threeByThree.postPadding = {1, 1};
    ConvGemmShape strided = threeByThree;
    strided.stride = {2, 2};
    strided.input = {9, 11};
    ConvGemmShape grouped = threeByThree;
    grouped.c = 8;
    grouped.k = 12;
    grouped.groups = 2;
    ConvGemmShape point = oneByOne;
    point.input = {3, 3};
    point.filter = {3, 3};
    ConvGemmShape oneD = threeByThree;
    oneD.input = {17};
    oneD.filter = {3};
    oneD.prePadding = {1};
    oneD.postPadding = {1};
    oneD.stride = {1};
    oneD.dilation = {1};
    ConvGemmShape threeD = threeByThree;
    threeD.input = {3, 4, 5};
    threeD.filter = {3, 3, 3};
    threeD.prePadding = {1, 1, 1};
    threeD.postPadding = {1, 1, 1};
    threeD.stride = {1, 1, 1};
    threeD.dilation = {1, 1, 1};
    return {
        {"OneByOneNhwc", layoutOf(oneByOne, true)},
        {"OneByOneNchw", layoutOf(oneByOne, false)},
        {"ThreeByThreeNhwc", layoutOf(threeByThree, true)},
        {"ThreeByThreeNchw", layoutOf(threeByThree, false)},
        {"ThreeByThreeNchwStaged",
         layoutOf(threeByThree, false),
         ColumnLayout::STAGED_CHANNELS_LAST},
        {"StridedNhwc", layoutOf(strided, true)},
        {"GroupedNhwc", layoutOf(grouped, true)},
        {"GroupedNchw", layoutOf(grouped, false)},
        {"PointOutputNhwc", layoutOf(point, true)},
        {"OneDNhwc", layoutOf(oneD, true)},
        {"ThreeDNchw", layoutOf(threeD, false)},
    };
}

using Chain = std::pair<std::string, std::vector<Step>>;

std::vector<Chain> menuChains()
{
    Step swish = unary(PointwiseMode::SWISH_FWD);
    swish.swishBeta = 1.5F;
    Step elu = unary(PointwiseMode::ELU_FWD);
    elu.eluAlpha = 0.7F;
    Step runtimeScale = scale();
    runtimeScale.operand = OperandKind::RUNTIME;
    runtimeScale.value = 1.25F;
    Step constantBias = bias();
    constantBias.operand = OperandKind::CONSTANT;
    constantBias.value = -0.125F;
    Step deviceScalar = scale(DataType::HALF);
    deviceScalar.operand = OperandKind::DEVICE_SCALAR;
    return {
        {"BiasRelu", {bias(DataType::HALF), relu()}},
        {"BiasSilu", {bias(), unary(PointwiseMode::SWISH_FWD)}},
        {"ScaleBiasClamp", {scale(DataType::BFLOAT16), bias(), relu(-0.5F, 0.5F)}},
        {"BiasClamp", {bias(), relu(-0.5F, 0.5F)}},
        {"BiasClippedRelu", {bias(), relu(std::nullopt, 0.25F)}},
        {"LeakyRelu", {relu(std::nullopt, std::nullopt, 0.1F)}},
        {"BiasGelu", {bias(), unary(PointwiseMode::GELU_FWD)}},
        {"BiasGeluTanh", {bias(), unary(PointwiseMode::GELU_APPROX_TANH_FWD)}},
        {"BiasSwish", {bias(DataType::HALF), swish}},
        {"Sigmoid", {unary(PointwiseMode::SIGMOID_FWD)}},
        {"Tanh", {unary(PointwiseMode::TANH_FWD)}},
        {"Elu", {elu}},
        {"ScalarScaleBiasSilu", {runtimeScale, constantBias, unary(PointwiseMode::SWISH_FWD)}},
        {"DeviceScalarScale", {deviceScalar}},
        {"BiasOnly", {bias(DataType::HALF)}},
    };
}

/// A per-channel or device-scalar operand on the device, in its type, and its values read back.
struct DeviceOperand
{
    std::unique_ptr<Tensor<float>> f32;
    std::unique_ptr<Tensor<Half>> f16;
    std::unique_ptr<Tensor<BFloat16>> bf16;
    std::vector<double> values;
    float host = 0.0F;
};

template <typename T>
void runFused(const EpilogueCase& convCase,
              const std::vector<Step>& steps,
              Backend backend,
              Specials specials = Specials::NONE,
              bool deterministic = false,
              bool libraryActivations = false)
{
    ConvGemmShape shape = convCase.shape;
    shape.dataType = dataTypeOf<T>();
    const FusedConvGraph graph(shape, steps);

    hipdnnEnginePluginHandle_t rawHandle = nullptr;
    HipblasltHandleFactory::createHipblasltHandle(&rawHandle);
    const std::unique_ptr<HipdnnEnginePluginHandle, void (*)(HipdnnEnginePluginHandle*)> handle(
        rawHandle, [](HipdnnEnginePluginHandle* owned) {
            HipblasltHandleFactory::destroyHipblasltHandle(owned);
            delete owned;
        });

    const auto support = gemmSupport(backend, libraryActivations);
    const auto narrow = backend == Backend::HIPBLASLT
                            ? ConvGemmPlanBuilder::NarrowPartials::ON_GFX11
                            : ConvGemmPlanBuilder::NarrowPartials::NEVER;
    const ConvGemmPlanBuilder builder(factoryFor(backend, deterministic),
                                      MAX_REPACK_WORKSPACE_BYTES,
                                      false,
                                      false,
                                      false,
                                      convCase.layout,
                                      narrow,
                                      std::nullopt,
                                      support);
    if(!builder.isApplicable(*handle, graph.graph()))
    {
        ASSERT_FALSE(libraryActivations) << "the library-activation engine declined";
        // As unfused: only a backend that finds no algorithm, or rocBLAS refusing to round twice.
        const auto problem = parseConvProblem(graph.graph(), ConvDirection::FORWARD);
        ASSERT_TRUE(problem.has_value()) << "the fused graph was not parsed";
        auto calls = convGemmCalls(*problem, MAX_REPACK_WORKSPACE_BYTES, convCase.layout);
        ASSERT_TRUE(calls.has_value()) << "the formulation declined the convolution";
        ASSERT_TRUE(fuseEpilogue(*problem, std::move(*calls), support).has_value())
            << "fusion declined the plan";
        GTEST_SKIP() << backendName(backend) << " declined a GEMM of this convolution";
    }
    HipdnnEnginePluginExecutionContext context;
    builder.buildPlan(*handle, graph.graph(), context);
    const size_t workspaceBytes = context.plan().getWorkspaceSize(*handle);
    {
        const auto problem = parseConvProblem(graph.graph(), ConvDirection::FORWARD);
        auto calls = convGemmCalls(*problem, MAX_REPACK_WORKSPACE_BYTES, convCase.layout);
        const auto form = fuseEpilogue(*problem, std::move(*calls), support)->epilogueForm;
        ::testing::Test::RecordProperty("epilogue_form", static_cast<int>(form));
    }

    auto xStrides = convPackedStrides(shape.xDims(), shape.xChannelsLast);
    auto yStrides = convPackedStrides(shape.yDims(), shape.yChannelsLast);
    Tensor<T> x(shape.xDims(), xStrides);
    Tensor<T> w(shape.wDims(), convPackedStrides(shape.wDims(), shape.wChannelsLast));
    Tensor<T> y(shape.yDims(), yStrides);
    if(specials == Specials::CLIP_BOUNDARY)
    {
        x.fillWithValue(static_cast<T>(0.0F));
    }
    else
    {
        x.fillTensorWithRandomValues(-1.0f, 1.0f, 31);
    }
    w.fillTensorWithRandomValues(-1.0f, 1.0f, 32);
    y.fillWithSentinelValue();

    std::vector<hipdnnPluginDeviceBuffer_t> buffers{
        {CONV_X_UID, x.rawDeviceData()},
        {CONV_W_UID, w.rawDeviceData()},
        {epilogueOutputUid(steps.size()), y.rawDeviceData()}};
    std::vector<DeviceOperand> operands(steps.size());
    std::vector<double> magnitudes(steps.size(), 1.0);
    float lowerBound = 0.0F;
    float upperBound = 1.0F;
    for(const auto& step : steps)
    {
        if(step.mode == PointwiseMode::RELU_FWD)
        {
            lowerBound = step.lowerClip.value_or(0.0F);
            upperBound = step.upperClip.value_or(lowerBound + 1.0F);
        }
    }
    std::mt19937 random(33);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    for(size_t i = 0; i < steps.size(); ++i)
    {
        const auto& step = steps[i];
        auto& operand = operands[i];
        const int64_t uid = EPI_OPERAND_UID + static_cast<int64_t>(i);
        if(step.operand == OperandKind::NONE)
        {
            continue;
        }
        if(step.operand == OperandKind::CONSTANT || step.operand == OperandKind::RUNTIME)
        {
            operand.values = {static_cast<double>(step.value)};
            operand.host = step.value;
            if(step.operand == OperandKind::RUNTIME)
            {
                buffers.push_back({uid, &operand.host});
            }
            magnitudes[i] = std::fabs(static_cast<double>(step.value));
            continue;
        }
        const bool isScale = step.mode == PointwiseMode::MUL;
        const int64_t count = step.operand == OperandKind::PER_CHANNEL ? shape.k : 1;
        std::vector<int64_t> dims(shape.yDims().size(), 1);
        dims[1] = count;
        const auto strides = convPackedStrides(dims, shape.yChannelsLast);
        for(int64_t c = 0; c < count; ++c)
        {
            double v = isScale ? 1.0 + 0.5 * static_cast<double>(unit(random))
                               : static_cast<double>(unit(random));
            if(specials == Specials::CLIP_BOUNDARY && !isScale)
            {
                const std::array<double, 5> picks{
                    static_cast<double>(lowerBound) - 0.5,
                    static_cast<double>(lowerBound),
                    0.5 * static_cast<double>(lowerBound + upperBound),
                    static_cast<double>(upperBound),
                    static_cast<double>(upperBound) + 0.5};
                v = picks[static_cast<size_t>(c) % picks.size()];
            }
            if(specials == Specials::NAN_INF && !isScale && c >= 1 && c <= 3)
            {
                if(c == 1)
                {
                    v = std::numeric_limits<double>::quiet_NaN();
                }
                else
                {
                    v = c == 2 ? std::numeric_limits<double>::infinity()
                               : -std::numeric_limits<double>::infinity();
                }
            }
            operand.values.push_back(v);
        }
        const auto fill = [&](auto& tensor, auto convert) {
            auto* host = static_cast<decltype(convert(0.0F))*>(tensor->rawHostData());
            for(int64_t c = 0; c < count; ++c)
            {
                const auto stored
                    = convert(static_cast<float>(operand.values[static_cast<size_t>(c)]));
                host[c] = stored;
                operand.values[static_cast<size_t>(c)]
                    = static_cast<double>(static_cast<float>(stored));
            }
            tensor->markHostModified();
            buffers.push_back({uid, tensor->rawDeviceData()});
        };
        switch(step.operandType)
        {
        case DataType::HALF:
            operand.f16 = std::make_unique<Tensor<Half>>(dims, strides);
            fill(operand.f16, [](float v) { return static_cast<Half>(v); });
            break;
        case DataType::BFLOAT16:
            operand.bf16 = std::make_unique<Tensor<BFloat16>>(dims, strides);
            fill(operand.bf16, [](float v) { return static_cast<BFloat16>(v); });
            break;
        default:
            operand.f32 = std::make_unique<Tensor<float>>(dims, strides);
            fill(operand.f32, [](float v) { return v; });
            break;
        }
        double largest = 0.0;
        for(const double v : operand.values)
        {
            if(std::isfinite(v))
            {
                largest = std::max(largest, std::fabs(v));
            }
        }
        magnitudes[i] = largest;
    }

    void* workspace = nullptr;
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
    }
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    handle->setStream(stream);
    context.plan().execute(
        *handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    if(workspace != nullptr)
    {
        ASSERT_EQ(hipFree(workspace), hipSuccess);
    }
    y.markDeviceModified();

    Tensor<float> conv(shape.yDims(), yStrides);
    hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
        x, w, conv, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
    int64_t taps = 1;
    for(const auto f : shape.filter)
    {
        taps *= f;
    }
    const int64_t reduction = (shape.c / shape.groups) * taps;
    const double convBound = 1e-6 * static_cast<double>(reduction);
    double gain = 1.0;
    for(size_t i = 0; i < steps.size(); ++i)
    {
        gain *= stepGain(steps[i], magnitudes[i]);
    }

    int64_t mismatches = 0;
    const auto& dims = shape.yDims();
    std::vector<int64_t> index(dims.size(), 0);
    while(true)
    {
        int64_t offset = 0;
        for(size_t a = 0; a < dims.size(); ++a)
        {
            offset += index[a] * yStrides[a];
        }
        const int64_t channel = index[1];
        auto v = static_cast<double>(static_cast<const float*>(conv.rawHostData())[offset]);
        double magnitude = std::fabs(v);
        for(size_t i = 0; i < steps.size(); ++i)
        {
            double input = 0.0;
            if(steps[i].operand != OperandKind::NONE)
            {
                const auto& values = operands[i].values;
                input = values.size() == 1 ? values[0] : values[static_cast<size_t>(channel)];
            }
            v = libraryActivations ? applyLibraryStep(steps[i], v, input)
                                   : applyStep(steps[i], v, input);
            magnitude = std::max(magnitude, std::fabs(v));
        }
        const auto want
            = static_cast<double>(static_cast<float>(static_cast<T>(static_cast<float>(v))));
        const auto got = static_cast<double>(
            static_cast<float>(static_cast<const T*>(y.rawHostData())[offset]));
        const double tolerance = convBound * gain + 4e-6 * magnitude
                                 + 2.0 * unitRoundoff<T>() * (std::fabs(want) + 1e-3);
        bool ok = false;
        if(std::isnan(want))
        {
            ok = std::isnan(got);
        }
        else if(std::isinf(want))
        {
            ok = got == want;
        }
        else
        {
            ok = std::fabs(got - want) <= tolerance;
        }
        if(!ok && ++mismatches <= 5)
        {
            ADD_FAILURE() << backendName(backend) << ": at offset " << offset << " (channel "
                          << channel << "): got " << got << ", want " << want << ", tolerance "
                          << tolerance;
        }
        size_t axis = dims.size();
        bool done = true;
        while(axis > 0)
        {
            --axis;
            if(++index[axis] < dims[axis])
            {
                done = false;
                break;
            }
            index[axis] = 0;
        }
        if(done)
        {
            break;
        }
    }
    EXPECT_EQ(mismatches, 0);
}

// Every shape and layout (every form) with bias + relu and bias + silu, all types and backends.
class TestGpuConvEpilogue
    : public ::testing::TestWithParam<std::tuple<EpilogueCase, Chain, Backend>>
{
};

TEST_P(TestGpuConvEpilogue, Fp32)
{
    SKIP_IF_NO_DEVICES();
    const auto& [convCase, chain, backend] = GetParam();
    runFused<float>(convCase, chain.second, backend);
}

TEST_P(TestGpuConvEpilogue, Fp16)
{
    SKIP_IF_NO_DEVICES();
    const auto& [convCase, chain, backend] = GetParam();
    runFused<Half>(convCase, chain.second, backend);
}

TEST_P(TestGpuConvEpilogue, Bfp16)
{
    SKIP_IF_NO_DEVICES();
    const auto& [convCase, chain, backend] = GetParam();
    runFused<BFloat16>(convCase, chain.second, backend);
}

std::string caseName(const ::testing::TestParamInfo<std::tuple<EpilogueCase, Chain, Backend>>& info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).first + "_"
           + backendName(std::get<2>(info.param));
}

INSTANTIATE_TEST_SUITE_P(Shapes,
                         TestGpuConvEpilogue,
                         ::testing::Combine(::testing::ValuesIn(epilogueCases()),
                                            ::testing::Values(menuChains()[0], menuChains()[1]),
                                            ::testing::ValuesIn(enabledBackends())),
                         caseName);

// The whole menu on the native-capable shape and on the pass shapes.
INSTANTIATE_TEST_SUITE_P(Menu,
                         TestGpuConvEpilogue,
                         ::testing::Combine(::testing::Values(epilogueCases()[0],
                                                              epilogueCases()[3],
                                                              epilogueCases()[4]),
                                            ::testing::ValuesIn(menuChains()),
                                            ::testing::ValuesIn(enabledBackends())),
                         caseName);

class TestGpuConvEpilogueSpecials
    : public ::testing::TestWithParam<std::tuple<EpilogueCase, Chain, Backend>>
{
};

TEST_P(TestGpuConvEpilogueSpecials, NanAndInfinityOperands)
{
    SKIP_IF_NO_DEVICES();
    const auto& [convCase, chain, backend] = GetParam();
    runFused<float>(convCase, chain.second, backend, Specials::NAN_INF);
    runFused<Half>(convCase, chain.second, backend, Specials::NAN_INF);
}

TEST_P(TestGpuConvEpilogueSpecials, ClipBoundaries)
{
    SKIP_IF_NO_DEVICES();
    const auto& [convCase, chain, backend] = GetParam();
    runFused<Half>(convCase, chain.second, backend, Specials::CLIP_BOUNDARY);
    runFused<float>(convCase, chain.second, backend, Specials::CLIP_BOUNDARY);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvEpilogueSpecials,
                         ::testing::Combine(::testing::Values(epilogueCases()[0],
                                                              epilogueCases()[3],
                                                              epilogueCases()[4]),
                                            ::testing::Values(menuChains()[0],
                                                              menuChains()[1],
                                                              menuChains()[3],
                                                              menuChains()[4],
                                                              menuChains()[9],
                                                              menuChains()[14]),
                                            ::testing::ValuesIn(enabledBackends())),
                         caseName);

#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
// The opt-in library-activation engine: hipBLASLt's own relu, clamp, swish and tanh GELU (and
// sigmoid for fp32) in the GEMM, on the 1x1 shapes in both layouts and an im2col one. Finite
// values match the chain within the fused tolerance; special values follow the library
// (applyLibraryStep), the divergence the engine is opt-in for.
using LibraryChain = std::pair<std::string, std::vector<Step> (*)(DataType)>;

std::vector<LibraryChain> libraryChains()
{
    return {
        {"BiasRelu", [](DataType t) { return std::vector<Step>{bias(t), relu()}; }},
        {"BiasClamp", [](DataType t) { return std::vector<Step>{bias(t), relu(-0.5F, 0.5F)}; }},
        {"BiasSilu",
         [](DataType t) { return std::vector<Step>{bias(t), unary(PointwiseMode::SWISH_FWD)}; }},
        {"BiasGeluTanh",
         [](DataType t) {
             return std::vector<Step>{bias(t), unary(PointwiseMode::GELU_APPROX_TANH_FWD)};
         }},
        {"Relu", [](DataType /*t*/) { return std::vector<Step>{relu()}; }},
    };
}

class TestGpuConvEpilogueLibrary
    : public ::testing::TestWithParam<std::tuple<EpilogueCase, LibraryChain>>
{
};

TEST_P(TestGpuConvEpilogueLibrary, Finite)
{
    SKIP_IF_NO_DEVICES();
    const auto& [convCase, chain] = GetParam();
    runFused<float>(
        convCase, chain.second(DataType::FLOAT), Backend::HIPBLASLT, Specials::NONE, false, true);
    runFused<Half>(
        convCase, chain.second(DataType::HALF), Backend::HIPBLASLT, Specials::NONE, false, true);
    runFused<BFloat16>(convCase,
                       chain.second(DataType::BFLOAT16),
                       Backend::HIPBLASLT,
                       Specials::NONE,
                       false,
                       true);
}

TEST_P(TestGpuConvEpilogueLibrary, SpecialValuesFollowTheLibrary)
{
    SKIP_IF_NO_DEVICES();
    const auto& [convCase, chain] = GetParam();
    runFused<float>(convCase,
                    chain.second(DataType::FLOAT),
                    Backend::HIPBLASLT,
                    Specials::NAN_INF,
                    false,
                    true);
    runFused<Half>(
        convCase, chain.second(DataType::HALF), Backend::HIPBLASLT, Specials::NAN_INF, false, true);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvEpilogueLibrary,
                         ::testing::Combine(::testing::Values(epilogueCases()[0],
                                                              epilogueCases()[1],
                                                              epilogueCases()[3]),
                                            ::testing::ValuesIn(libraryChains())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + std::get<1>(info.param).first;
                         });
#endif

// Real-model fp16 convolutions of the conv census whose fused graphs need an fp32-output GEMM
// (small n x HW against a long reduction): every engine, the deterministic siblings included,
// claims the fused graph exactly where it claims the bare convolution, and computes it, or
// declines it for the one reason fusion may add (below).
ConvGemmShape censusShape(int64_t n,
                          int64_t c,
                          int64_t k,
                          int64_t hw,
                          int64_t f,
                          int64_t pad,
                          int64_t stride,
                          int64_t groups,
                          bool channelsLast)
{
    ConvGemmShape shape;
    shape.n = n;
    shape.c = c;
    shape.k = k;
    shape.groups = groups;
    shape.input = {hw, hw};
    shape.filter = {f, f};
    shape.prePadding = {pad, pad};
    shape.postPadding = {pad, pad};
    shape.stride = {stride, stride};
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = channelsLast;
    shape.dataType = DataType::HALF;
    return shape;
}

std::vector<EpilogueCase> realModelCases()
{
    std::vector<EpilogueCase> cases;
    struct Row
    {
        const char* name;
        int64_t n, c, k, hw, f, pad, stride, groups;
    };
    const std::vector<Row> rows{{"InceptionDense8", 1, 448, 384, 8, 3, 1, 1, 1},
                                {"InceptionPointOutput5", 16, 128, 768, 5, 5, 0, 1, 1},
                                {"Resnet50Dense7", 1, 512, 512, 7, 3, 1, 1, 1},
                                {"Resnet50Dense8", 1, 512, 512, 8, 3, 1, 1, 1},
                                {"Resnet50Dense14", 1, 512, 512, 14, 3, 1, 1, 1},
                                {"Resnet50Dense14Stride2", 1, 512, 512, 14, 3, 1, 2, 1},
                                {"Resnet50Dense16Stride2", 1, 512, 512, 16, 3, 1, 2, 1},
                                {"Stylegan2Dense4", 1, 512, 512, 4, 3, 1, 1, 1},
                                {"Stylegan2DenseN2Stride2", 2, 512, 512, 8, 3, 1, 2, 1},
                                {"Stylegan2GroupedG2", 1, 1024, 1024, 4, 3, 1, 1, 2},
                                {"Stylegan2GroupedG4", 1, 2048, 2048, 4, 3, 1, 1, 4},
                                {"Yolo11Dense16", 1, 512, 64, 16, 3, 1, 1, 1},
                                {"Yolo11Dense20", 1, 512, 64, 20, 3, 1, 1, 1}};
    for(const auto& row : rows)
    {
        for(const bool channelsLast : {false, true})
        {
            cases.push_back({std::string(row.name) + (channelsLast ? "Nhwc" : "Nchw"),
                             censusShape(row.n,
                                         row.c,
                                         row.k,
                                         row.hw,
                                         row.f,
                                         row.pad,
                                         row.stride,
                                         row.groups,
                                         channelsLast)});
        }
    }
    return cases;
}

/// Which forms of @p call the backend runs: as given, its reduction cut in 8 (fp32 partials),
/// fp32 operands, and a 16-bit output (the bare GEMM's). Probes why a fused GEMM is refused.
std::string probeCall(const ConvGemmPlanBuilder::PlanFactory& factory,
                      const HipdnnEnginePluginHandle& handle,
                      const GemmCall& call,
                      DataType elementType)
{
    const auto runs = [&](const GemmCall& variant) {
        try
        {
            static_cast<void>(factory(handle, {variant}));
            return std::string("y");
        }
        catch(const std::exception&)
        {
            return std::string("n");
        }
    };
    GemmCall chunked = call;
    chunked.k = std::max<int64_t>(1, call.k / 8);
    GemmCall wide = call;
    wide.a.dataType = DataType::FLOAT;
    wide.b.dataType = DataType::FLOAT;
    GemmCall narrow = call;
    narrow.d.dataType = elementType;
    return "gemm m" + std::to_string(call.m) + " n" + std::to_string(call.n) + " k"
           + std::to_string(call.k) + " batch" + std::to_string(call.batchCount) + ": fp32-out "
           + runs(call) + ", k/8 " + runs(chunked) + ", fp32-in " + runs(wide) + ", 16-bit-out "
           + runs(narrow);
}

class TestGpuConvEpilogueClaims
    : public ::testing::TestWithParam<std::tuple<EpilogueCase, Chain, Backend, bool>>
{
};

TEST_P(TestGpuConvEpilogueClaims, FusedWhereBare)
{
    SKIP_IF_NO_DEVICES();
    const auto& [convCase, chain, backend, deterministic] = GetParam();
    hipdnnEnginePluginHandle_t rawHandle = nullptr;
    HipblasltHandleFactory::createHipblasltHandle(&rawHandle);
    const std::unique_ptr<HipdnnEnginePluginHandle, void (*)(HipdnnEnginePluginHandle*)> handle(
        rawHandle, [](HipdnnEnginePluginHandle* owned) {
            HipblasltHandleFactory::destroyHipblasltHandle(owned);
            delete owned;
        });
    const ConvGemmPlanBuilder builder(factoryFor(backend, deterministic),
                                      MAX_REPACK_WORKSPACE_BYTES,
                                      false,
                                      false,
                                      false,
                                      ColumnLayout::FOLLOW_Y,
                                      backend == Backend::HIPBLASLT
                                          ? ConvGemmPlanBuilder::NarrowPartials::ON_GFX11
                                          : ConvGemmPlanBuilder::NarrowPartials::NEVER,
                                      std::nullopt,
                                      gemmSupport(backend));
    const ConvGemmGraph bare(convCase.shape, ConvDirection::FORWARD);
    const FusedConvGraph fused(convCase.shape, chain.second);
    const bool bareClaimed = builder.isApplicable(*handle, bare.graph());
    const bool fusedClaimed = builder.isApplicable(*handle, fused.graph());
    if(bareClaimed && !fusedClaimed)
    {
        // The one decline fusion may add: every layout's fused plan needs a GEMM with an fp32
        // output (one rounding), and the backend has no algorithm for one of them. Every layout
        // must formulate, and every layout must hold a GEMM the backend refuses.
        const auto problem = parseConvProblem(fused.graph(), ConvDirection::FORWARD);
        ASSERT_TRUE(problem.has_value());
        const auto factory = factoryFor(backend, deterministic);
        for(const auto layout : {ColumnLayout::FOLLOW_Y,
                                 ColumnLayout::CHANNELS_FIRST,
                                 ColumnLayout::CHANNELS_LAST,
                                 ColumnLayout::STAGED_CHANNELS_LAST})
        {
            auto calls = convGemmCalls(*problem, MAX_REPACK_WORKSPACE_BYTES, layout);
            ASSERT_TRUE(calls.has_value());
            const auto plan = fuseEpilogue(*problem, std::move(*calls), gemmSupport(backend));
            ASSERT_TRUE(plan.has_value()) << "fusion declined a plan";
            bool refused = false;
            for(const auto& stage : plan->stages)
            {
                if(stage.calls.empty())
                {
                    continue;
                }
                try
                {
                    static_cast<void>(factory(*handle, stage.calls));
                }
                catch(const std::exception&)
                {
                    refused = true;
                    for(const auto& call : stage.calls)
                    {
                        std::cout << "PROBE " << convCase.name << " " << chain.first << " layout "
                                  << static_cast<int>(layout) << " "
                                  << probeCall(factory, *handle, call, problem->dataType) << "\n";
                    }
                }
            }
            EXPECT_TRUE(refused) << "the backend runs every GEMM of a layout, yet the graph was "
                                    "declined";
        }
        ASSERT_EQ(backend, Backend::HIPBLASLT) << "only hipBLASLt refuses an fp32-output GEMM";
        GTEST_SKIP() << "hipBLASLt"
                     << (deterministic ? " without a workspace (the deterministic engine)" : "")
                     << " has no algorithm for an fp32-output GEMM of this convolution";
    }
    EXPECT_EQ(fusedClaimed, bareClaimed);
    if(bareClaimed)
    {
        runFused<Half>(convCase, chain.second, backend, Specials::NONE, deterministic);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvEpilogueClaims,
                         ::testing::Combine(::testing::ValuesIn(realModelCases()),
                                            ::testing::Values(menuChains()[14],
                                                              menuChains()[0],
                                                              menuChains()[6],
                                                              menuChains()[1]),
                                            ::testing::ValuesIn(enabledBackends()),
                                            ::testing::Bool()),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + std::get<1>(info.param).first + "_"
                                    + backendName(std::get<2>(info.param))
                                    + (std::get<3>(info.param) ? "_Deterministic" : "");
                         });

// ---------------------------------------------------------------------------
// Timing (opt-in, HIPBLASLT_CONV_EPILOGUE_PERF=1): bare against fused (bias + relu) plans of
// real-model shapes, each built as an autotuning run builds it (benchmarking on, the fastest
// candidate kept), then timed over 20 executes. Run under rocprofv3 --kernel-trace for the
// per-kernel split.
// ---------------------------------------------------------------------------

struct PerfShape
{
    std::string name;
    ConvGemmShape shape;
};

ConvGemmShape nhwc(int64_t n,
                   int64_t c,
                   int64_t k,
                   int64_t hw,
                   int64_t f,
                   int64_t pad,
                   int64_t stride,
                   DataType type)
{
    ConvGemmShape shape;
    shape.n = n;
    shape.c = c;
    shape.k = k;
    shape.input = {hw, hw};
    shape.filter = {f, f};
    shape.prePadding = {pad, pad};
    shape.postPadding = {pad, pad};
    shape.stride = {stride, stride};
    shape.xChannelsLast = shape.wChannelsLast = shape.yChannelsLast = true;
    shape.dataType = type;
    return shape;
}

std::vector<PerfShape> perfShapes()
{
    return {{"EfficientnetPointwise", nhwc(16, 64, 256, 60, 1, 0, 1, DataType::HALF)},
            {"VitPatchify", nhwc(16, 3, 1280, 1024, 16, 0, 16, DataType::BFLOAT16)},
            {"Sd15Pointwise", nhwc(1, 640, 320, 64, 1, 0, 1, DataType::BFLOAT16)},
            {"Resnet50Dense", nhwc(128, 512, 512, 16, 3, 1, 1, DataType::HALF)}};
}

class TestGpuConvEpiloguePerf
    : public ::testing::TestWithParam<std::tuple<PerfShape, Backend, bool>>
{
};

TEST_P(TestGpuConvEpiloguePerf, Time)
{
    SKIP_IF_NO_DEVICES();
    const char* enabled = std::getenv("HIPBLASLT_CONV_EPILOGUE_PERF");
    if(enabled == nullptr || std::string(enabled) != "1")
    {
        GTEST_SKIP() << "timing runs only with HIPBLASLT_CONV_EPILOGUE_PERF=1";
    }
    const auto& [perf, backend, fusedGraph] = GetParam();
    const auto& shape = perf.shape;
    hipdnnEnginePluginHandle_t rawHandle = nullptr;
    HipblasltHandleFactory::createHipblasltHandle(&rawHandle);
    const std::unique_ptr<HipdnnEnginePluginHandle, void (*)(HipdnnEnginePluginHandle*)> handle(
        rawHandle, [](HipdnnEnginePluginHandle* owned) {
            HipblasltHandleFactory::destroyHipblasltHandle(owned);
            delete owned;
        });
    const ConvGemmPlanBuilder builder(factoryFor(backend),
                                      MAX_REPACK_WORKSPACE_BYTES,
                                      false,
                                      false,
                                      false,
                                      ColumnLayout::FOLLOW_Y,
                                      backend == Backend::HIPBLASLT
                                          ? ConvGemmPlanBuilder::NarrowPartials::ON_GFX11
                                          : ConvGemmPlanBuilder::NarrowPartials::NEVER,
                                      std::nullopt,
                                      gemmSupport(backend));
    const std::vector<Step> chain{bias(shape.dataType), relu()};
    const ConvGemmGraph bare(shape, ConvDirection::FORWARD);
    const FusedConvGraph fused(shape, chain);
    const auto& graph = fusedGraph ? fused.graph() : bare.graph();
    if(!builder.isApplicable(*handle, graph))
    {
        GTEST_SKIP() << "not applicable";
    }
    HipdnnEnginePluginExecutionContext context;
    context.setBenchmarking(true);
    builder.buildPlan(*handle, graph, context);
    const size_t workspaceBytes = context.plan().getWorkspaceSize(*handle);

    const auto bytesOf = [&shape](const std::vector<int64_t>& dims) {
        int64_t count = 1;
        for(const auto d : dims)
        {
            count *= d;
        }
        return static_cast<size_t>(count) * (shape.dataType == DataType::FLOAT ? 4 : 2);
    };
    void* x = nullptr;
    void* w = nullptr;
    void* y = nullptr;
    void* b = nullptr;
    void* workspace = nullptr;
    ASSERT_EQ(hipMalloc(&x, bytesOf(shape.xDims())), hipSuccess);
    ASSERT_EQ(hipMalloc(&w, bytesOf(shape.wDims())), hipSuccess);
    ASSERT_EQ(hipMalloc(&y, bytesOf(shape.yDims())), hipSuccess);
    ASSERT_EQ(hipMalloc(&b, bytesOf({shape.k})), hipSuccess);
    ASSERT_EQ(hipMemset(x, 0x11, bytesOf(shape.xDims())), hipSuccess);
    ASSERT_EQ(hipMemset(w, 0x11, bytesOf(shape.wDims())), hipSuccess);
    ASSERT_EQ(hipMemset(b, 0x11, bytesOf({shape.k})), hipSuccess);
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
    }
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    handle->setStream(stream);
    const std::vector<hipdnnPluginDeviceBuffer_t> buffers{
        {CONV_X_UID, x},
        {CONV_W_UID, w},
        {fusedGraph ? epilogueOutputUid(chain.size()) : CONV_Y_UID, y},
        {EPI_OPERAND_UID, b}};
    const auto run = [&] {
        context.plan().execute(
            *handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
    };
    // The first execute times the candidates; then warm up.
    for(int i = 0; i < 4; ++i)
    {
        run();
    }
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    ASSERT_EQ(hipEventCreate(&start), hipSuccess);
    ASSERT_EQ(hipEventCreate(&stop), hipSuccess);
    std::vector<float> times;
    for(int i = 0; i < 20; ++i)
    {
        ASSERT_EQ(hipEventRecord(start, stream), hipSuccess);
        run();
        ASSERT_EQ(hipEventRecord(stop, stream), hipSuccess);
        ASSERT_EQ(hipEventSynchronize(stop), hipSuccess);
        float ms = 0.0F;
        ASSERT_EQ(hipEventElapsedTime(&ms, start, stop), hipSuccess);
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    std::cout << "PERF " << perf.name << " " << backendName(backend) << " "
              << (fusedGraph ? "fused" : "bare") << " median_ms " << times[times.size() / 2]
              << " p25_ms " << times[times.size() / 4] << " workspace " << workspaceBytes << "\n";
    static_cast<void>(hipEventDestroy(start));
    static_cast<void>(hipEventDestroy(stop));
    static_cast<void>(hipStreamDestroy(stream));
    for(void* p : {x, w, y, b, workspace})
    {
        if(p != nullptr)
        {
            static_cast<void>(hipFree(p));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvEpiloguePerf,
                         ::testing::Combine(::testing::ValuesIn(perfShapes()),
                                            ::testing::ValuesIn(enabledBackends()),
                                            ::testing::Bool()),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + backendName(std::get<1>(info.param))
                                    + (std::get<2>(info.param) ? "_Fused" : "_Bare");
                         });

#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
/// hipBLASLt's solutions for one conv GEMM shape (y^T = w x^T, channels as rows), with and
/// without a native epilogue: the epilogue narrows the solution set, and autotune picks among
/// the engines (fused vs a pass) where that costs.
int solutionCount(const HipdnnEnginePluginHandle& handle,
                  hipDataType type,
                  int64_t m,
                  int64_t n,
                  int64_t k,
                  std::optional<hipblasLtEpilogue_t> epilogue)
{
    const HipblasltMatmulDesc desc(HIPBLAS_OP_T, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    if(epilogue.has_value())
    {
        THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
            desc.matmulDesc(), HIPBLASLT_MATMUL_DESC_EPILOGUE, &*epilogue, sizeof(*epilogue)));
        const void* placeholder = reinterpret_cast<const void*>(0x1);
        THROW_ON_HIPBLASLT_FAILURE(
            hipblasLtMatmulDescSetAttribute(desc.matmulDesc(),
                                            HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            static_cast<const void*>(&placeholder),
                                            sizeof(placeholder)));
        THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
            desc.matmulDesc(), HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &type, sizeof(type)));
    }
    const HipblasltMatrixLayout a(type, static_cast<uint64_t>(k), static_cast<uint64_t>(m), k);
    const HipblasltMatrixLayout b(type, static_cast<uint64_t>(k), static_cast<uint64_t>(n), k);
    const HipblasltMatrixLayout d(type, static_cast<uint64_t>(m), static_cast<uint64_t>(n), m);
    hipblasLtMatmulPreference_t preference = nullptr;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceCreate(&preference));
    std::vector<hipblasLtMatmulHeuristicResult_t> results(1024);
    int returned = 0;
    const auto status = hipblasLtMatmulAlgoGetHeuristic(handle.hipblasltHandle,
                                                        desc.matmulDesc(),
                                                        a.matrixLayout(),
                                                        b.matrixLayout(),
                                                        d.matrixLayout(),
                                                        d.matrixLayout(),
                                                        preference,
                                                        static_cast<int>(results.size()),
                                                        results.data(),
                                                        &returned);
    static_cast<void>(hipblasLtMatmulPreferenceDestroy(preference));
    return status == HIPBLAS_STATUS_SUCCESS ? returned : 0;
}

TEST(TestGpuConvEpilogueHipblaslt, SolutionCountWithNativeEpilogue)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t rawHandle = nullptr;
    HipblasltHandleFactory::createHipblasltHandle(&rawHandle);
    const std::unique_ptr<HipdnnEnginePluginHandle, void (*)(HipdnnEnginePluginHandle*)> handle(
        rawHandle, [](HipdnnEnginePluginHandle* owned) {
            HipblasltHandleFactory::destroyHipblasltHandle(owned);
            delete owned;
        });
    // ResNet-50 conv2_x 1x1 64 -> 256 at N 32: m = K, n = N H W, k = C.
    for(const auto type : {HIP_R_16F, HIP_R_16BF, HIP_R_32F})
    {
        const int plain
            = solutionCount(*handle, type, 256, int64_t{32} * 56 * 56, 64, std::nullopt);
        const int withBias
            = solutionCount(*handle, type, 256, int64_t{32} * 56 * 56, 64, HIPBLASLT_EPILOGUE_BIAS);
        const int biasRelu = solutionCount(
            *handle, type, 256, int64_t{32} * 56 * 56, 64, HIPBLASLT_EPILOGUE_RELU_BIAS);
        std::cout << "hipBLASLt solutions, type " << static_cast<int>(type) << ": plain " << plain
                  << ", bias " << withBias << ", bias+relu " << biasRelu << "\n";
        ::testing::Test::RecordProperty("solutions_" + std::to_string(static_cast<int>(type)),
                                        std::to_string(plain) + "/" + std::to_string(withBias) + "/"
                                            + std::to_string(biasRelu));
        EXPECT_GT(plain, 0);
        EXPECT_GT(withBias, 0) << "no hipBLASLt solution with a bias epilogue";
    }
}
#endif

} // namespace
