// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslPointwiseCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslPointwiseTestGraphs.hpp"

/**
 * @file TestFlydslPointwiseEngine.cpp
 * @brief The flyDSL pointwise pack's host seams: what the graph matcher admits and refuses,
 *        how a graph collapses to the kernel's four axes, which object the kernel matcher
 *        pairs it with, the score and the dispatch registry. The launches -- every object,
 *        and the broadcast patterns the Qwen graphs use against the CPU reference -- are in
 *        TestFlydslPointwiseObjects.cpp; the census of what ships in
 *        TestFlydslPointwisePacks.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace pw = hip_kernel_provider::kernel_ingestor_engine::flydsl_pointwise;

using data_objects::DataType;
using data_objects::PointwiseMode;

std::optional<BoundTokens> matchSpec(const PointwiseGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslPointwiseGraph(spec));
    return matchesGraph(FLYDSL_POINTWISE, fixture.context());
}

// ---------------------------------------------------------------------------
// Graph matcher: acceptance
// ---------------------------------------------------------------------------

struct AcceptCase
{
    const char* name;
    PointwiseGraphSpec spec;
};

PointwiseGraphSpec withTypes(PointwiseGraphSpec spec, DataType in0, DataType in1, DataType out)
{
    spec.in0.dataType = in0;
    if(spec.in1.has_value())
    {
        spec.in1->dataType = in1;
    }
    spec.out.dataType = out;
    return spec;
}

PointwiseGraphSpec
    withBetas(PointwiseGraphSpec spec, std::optional<float> swish, std::optional<float> softplus)
{
    spec.swishBeta = swish;
    spec.softplusBeta = softplus;
    return spec;
}

PointwiseGraphSpec withComputeType(PointwiseGraphSpec spec, DataType computeType)
{
    spec.computeType = computeType;
    return spec;
}

PointwiseGraphSpec withOutStrides(PointwiseGraphSpec spec, std::vector<int64_t> strides)
{
    spec.out.strides = std::move(strides);
    return spec;
}

std::vector<AcceptCase> acceptCases()
{
    constexpr int64_t T = 37;
    constexpr int64_t H = 5120;
    constexpr int64_t K = 8;
    return {
        {"Sigmoid", flydslUnarySpec(PointwiseMode::SIGMOID_FWD, {1, T, H})},
        {"Swish", flydslUnarySpec(PointwiseMode::SWISH_FWD, {T, H})},
        {"SwishBetaOne",
         withBetas(flydslUnarySpec(PointwiseMode::SWISH_FWD, {T, H}), 1.0F, std::nullopt)},
        {"SwishAnyFiniteBeta",
         withBetas(flydslUnarySpec(PointwiseMode::SWISH_FWD, {T, H}), -0.5F, std::nullopt)},
        {"Exp", flydslUnarySpec(PointwiseMode::EXP, {T, 48}, DataType::FLOAT)},
        {"Softplus", flydslUnarySpec(PointwiseMode::SOFTPLUS_FWD, {T, 48}, DataType::FLOAT)},
        {"SoftplusBeta",
         withBetas(flydslUnarySpec(PointwiseMode::SOFTPLUS_FWD, {T, 48}), std::nullopt, 2.0F)},
        {"Neg", flydslUnarySpec(PointwiseMode::NEG, {T, H}, DataType::HALF)},
        {"Identity", flydslUnarySpec(PointwiseMode::IDENTITY, {T, H})},
        {"IdentityCastsBf16ToF32",
         withTypes(flydslUnarySpec(PointwiseMode::IDENTITY, {T, H}),
                   DataType::BFLOAT16,
                   DataType::BFLOAT16,
                   DataType::FLOAT)},
        {"IdentityCastsF32ToF16",
         withTypes(flydslUnarySpec(PointwiseMode::IDENTITY, {T, H}),
                   DataType::FLOAT,
                   DataType::FLOAT,
                   DataType::HALF)},
        {"AddResidual", flydslBinarySpec(PointwiseMode::ADD, {1, T, H}, {1, T, H}, {1, T, H})},
        {"SubSameShape", flydslBinarySpec(PointwiseMode::SUB, {T, H}, {T, H}, {T, H})},
        {"MulGate", flydslBinarySpec(PointwiseMode::MUL, {T, H}, {T, H}, {T, H})},
        {"MulRoutingWeights",
         flydslBinarySpec(PointwiseMode::MUL, {1, T, K, H}, {1, T, K, 1}, {1, T, K, H})},
        {"MulBf16RowsByF32Weights",
         withTypes(flydslBinarySpec(PointwiseMode::MUL, {1, T, K, H}, {1, T, K, 1}, {1, T, K, H}),
                   DataType::BFLOAT16,
                   DataType::FLOAT,
                   DataType::BFLOAT16)},
        {"DivRenormalise",
         flydslBinarySpec(PointwiseMode::DIV, {T, K}, {T, 1}, {T, K}, DataType::FLOAT)},
        {"BroadcastsTheFirstInput", flydslBinarySpec(PointwiseMode::ADD, {1, H}, {T, H}, {T, H})},
        {"BroadcastsALowerRankInput", flydslBinarySpec(PointwiseMode::ADD, {T, H}, {H}, {T, H})},
        {"BroadcastsBothInputs", flydslBinarySpec(PointwiseMode::MUL, {T, 1}, {1, H}, {T, H})},
        {"WritesABshdViewOfABhsdOutput",
         withOutStrides(flydslUnarySpec(PointwiseMode::SIGMOID_FWD, {2, 4, 5, 64}),
                        {int64_t{4} * 5 * 64, 64, int64_t{4} * 64, 1})},
        {"Rank6PackedCollapses",
         flydslBinarySpec(
             PointwiseMode::ADD, {2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7})},
        {"SingleElement", flydslBinarySpec(PointwiseMode::DIV, {1}, {1}, {1}, DataType::FLOAT)},
        {"ComputeTypeUnset",
         withComputeType(flydslUnarySpec(PointwiseMode::EXP, {T}), DataType::UNSET)},
        {"ComputeTypeHalf",
         withComputeType(flydslUnarySpec(PointwiseMode::EXP, {T}), DataType::HALF)},
    };
}

class TestFlydslPointwiseGraphMatcherAcceptance : public ::testing::TestWithParam<AcceptCase>
{
};

TEST_P(TestFlydslPointwiseGraphMatcherAcceptance, Accepts)
{
    EXPECT_TRUE(matchSpec(GetParam().spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslPointwiseGraphMatcherAcceptance,
                         ::testing::ValuesIn(acceptCases()),
                         [](const auto& info) { return std::string(info.param.name); });

// ---------------------------------------------------------------------------
// Graph matcher: refusal
// ---------------------------------------------------------------------------

struct RefuseCase
{
    const char* name;
    PointwiseGraphSpec spec;
};

std::vector<RefuseCase> refuseCases()
{
    const auto unary = flydslUnarySpec(PointwiseMode::SIGMOID_FWD, {4, 64});
    const auto binary = flydslBinarySpec(PointwiseMode::ADD, {4, 64}, {4, 64}, {4, 64});
    std::vector<RefuseCase> cases;
    const auto add = [&cases](const char* name, PointwiseGraphSpec spec) {
        cases.push_back({name, std::move(spec)});
    };
    const auto mutate = [](PointwiseGraphSpec spec, auto&& change) {
        change(spec);
        return spec;
    };

    for(const auto& [name, mode] : std::vector<std::pair<const char*, PointwiseMode>>{
            {"Relu", PointwiseMode::RELU_FWD},
            {"Tanh", PointwiseMode::TANH_FWD},
            {"Gelu", PointwiseMode::GELU_FWD},
            {"Abs", PointwiseMode::ABS},
            {"Log", PointwiseMode::LOG},
            {"Rsqrt", PointwiseMode::RSQRT},
            {"SigmoidBackward", PointwiseMode::SIGMOID_BWD},
            {"SwishBackward", PointwiseMode::SWISH_BWD},
            {"Unset", PointwiseMode::UNSET}})
    {
        add(name, flydslUnarySpec(mode, {4, 64}));
    }
    for(const auto& [name, mode] : std::vector<std::pair<const char*, PointwiseMode>>{
            {"Max", PointwiseMode::MAX_OP},
            {"Min", PointwiseMode::MIN_OP},
            {"CmpGt", PointwiseMode::CMP_GT},
            {"AddSquare", PointwiseMode::ADD_SQUARE}})
    {
        add(name, flydslBinarySpec(mode, {4, 64}, {4, 64}, {4, 64}));
    }
    add("BinarySelect", mutate(binary, [](auto& s) {
            s.mode = PointwiseMode::BINARY_SELECT;
            s.includeIn2 = true;
        }));

    add("BinaryOperationWithOneInput", mutate(unary, [](auto& s) { s.mode = PointwiseMode::ADD; }));
    add("UnaryOperationWithTwoInputs",
        mutate(binary, [](auto& s) { s.mode = PointwiseMode::SIGMOID_FWD; }));
    add("ThirdInput", mutate(binary, [](auto& s) { s.includeIn2 = true; }));
    add("DanglingSecondInput", mutate(binary, [](auto& s) { s.danglingIn1 = true; }));
    add("TwoNodes", mutate(unary, [](auto& s) { s.twoNodes = true; }));
    add("OverrideShape", mutate(unary, [](auto& s) { s.overrideShape = true; }));

    add("DoubleInput", mutate(unary, [](auto& s) { s.in0.dataType = DataType::DOUBLE; }));
    add("Int32Output", mutate(unary, [](auto& s) { s.out.dataType = DataType::INT32; }));
    add("Fp8SecondInput", mutate(binary, [](auto& s) { s.in1->dataType = DataType::FP8_E4M3; }));
    add("ComputeTypeDouble", mutate(unary, [](auto& s) { s.computeType = DataType::DOUBLE; }));

    add("VirtualOutput", mutate(unary, [](auto& s) { s.out.isVirtual = true; }));
    add("VirtualInput", mutate(binary, [](auto& s) { s.in1->isVirtual = true; }));
    add("PassByValueInput", mutate(binary, [](auto& s) {
            s.in1->dims = {1};
            s.in1->runtimePassByValue = true;
        }));
    add("NoStrides", mutate(unary, [](auto& s) { s.in0.omitStrides = true; }));

    add("SwishInfiniteBeta",
        mutate(flydslUnarySpec(PointwiseMode::SWISH_FWD, {4, 64}),
               [](auto& s) { s.swishBeta = std::numeric_limits<float>::infinity(); }));
    add("SwishNaNBeta", mutate(flydslUnarySpec(PointwiseMode::SWISH_FWD, {4, 64}), [](auto& s) {
            s.swishBeta = std::numeric_limits<float>::quiet_NaN();
        }));
    add("SoftplusZeroBeta",
        mutate(flydslUnarySpec(PointwiseMode::SOFTPLUS_FWD, {4, 64}),
               [](auto& s) { s.softplusBeta = 0.0F; }));
    add("SoftplusNegativeBeta",
        mutate(flydslUnarySpec(PointwiseMode::SOFTPLUS_FWD, {4, 64}),
               [](auto& s) { s.softplusBeta = -1.0F; }));

    add("InputDoesNotBroadcast", mutate(binary, [](auto& s) { s.in1->dims = {4, 32}; }));
    add("InputLargerThanOutput", mutate(binary, [](auto& s) { s.in0.dims = {2, 4, 64}; }));
    add("UnaryInputDoesNotMatch", mutate(unary, [](auto& s) { s.in0.dims = {4, 32}; }));
    add("ZeroExtent", mutate(unary, [](auto& s) {
            s.in0.dims = {0, 64};
            s.out.dims = {0, 64};
        }));
    add("NegativeInputStride", mutate(unary, [](auto& s) { s.in0.strides = {64, -1}; }));
    add("OutputBroadcastStride", mutate(unary, [](auto& s) { s.out.strides = {0, 1}; }));
    add("OverlappingOutput", mutate(unary, [](auto& s) { s.out.strides = {32, 1}; }));
    add("FiveAxesNoneMerge",
        mutate(flydslUnarySpec(PointwiseMode::NEG, {2, 3, 4, 5, 6}), [](auto& s) {
            // Memory order axis 0, 2, 4, 1, 3: no adjacent pair is one run in both the
            // output and the packed input, so five axes remain.
            s.out.strides = {1, 48, 2, 144, 8};
        }));
    return cases;
}

class TestFlydslPointwiseGraphMatcherRefusal : public ::testing::TestWithParam<RefuseCase>
{
};

TEST_P(TestFlydslPointwiseGraphMatcherRefusal, Refuses)
{
    EXPECT_FALSE(matchSpec(GetParam().spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslPointwiseGraphMatcherRefusal,
                         ::testing::ValuesIn(refuseCases()),
                         [](const auto& info) { return std::string(info.param.name); });

TEST(TestFlydslPointwiseGraphMatcher, RefusesAGraphWithoutAPointwiseNode)
{
    // No node at all: nothing for a single-node engine to serve.
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(data_objects::CreateGraphDirect(builder, "empty"));
    const GraphFixture fixture(std::move(builder));
    EXPECT_FALSE(matchesGraph(FLYDSL_POINTWISE, fixture.context()).has_value());
}

// ---------------------------------------------------------------------------
// Binding
// ---------------------------------------------------------------------------

TEST(TestFlydslPointwiseBinding, BindsBothInputsOfABinaryGraph)
{
    const auto bound = matchSpec(flydslBinarySpec(PointwiseMode::MUL, {3, 8}, {3, 1}, {3, 8}));
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_POINTWISE.inputAToken).value_or(-1),
              FLYDSL_POINTWISE_IN0_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_POINTWISE.inputBToken).value_or(-1),
              FLYDSL_POINTWISE_IN1_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_POINTWISE.outputToken).value_or(-1),
              FLYDSL_POINTWISE_OUT_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_POINTWISE_ELEMENTS_TOKEN).value_or(-1), 24);
}

TEST(TestFlydslPointwiseBinding, BindsNoSecondInputForAUnaryGraph)
{
    const auto bound = matchSpec(flydslUnarySpec(PointwiseMode::EXP, {3, 8}));
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_POINTWISE.inputBToken).has_value());
}

// ---------------------------------------------------------------------------
// Collapse
// ---------------------------------------------------------------------------

pw::OperandShape packed(std::vector<int64_t> dims)
{
    auto strides = flydslPointwisePackedStrides(dims);
    return {std::move(dims), std::move(strides)};
}

TEST(TestFlydslPointwiseCollapse, APackedSameShapeGraphIsOneDenseRun)
{
    const auto out = packed({1, 37, 5120});
    const auto geometry = pw::collapse(out, out, &out);
    ASSERT_TRUE(geometry.has_value());
    EXPECT_TRUE(geometry->dense);
    EXPECT_EQ(geometry->elements, 37 * 5120);
    EXPECT_EQ(geometry->extents, (std::array<int64_t, 4>{1, 1, 1, int64_t{37} * 5120}));
    EXPECT_EQ(geometry->outStrides[3], 1);
    EXPECT_EQ(geometry->aStrides[3], 1);
    EXPECT_EQ(geometry->bStrides[3], 1);
}

TEST(TestFlydslPointwiseCollapse, RoutingWeightsBroadcastAlongTheHiddenAxis)
{
    const auto out = packed({1, 37, 8, 4096});
    const auto weights = packed({1, 37, 8, 1});
    const auto geometry = pw::collapse(out, out, &weights);
    ASSERT_TRUE(geometry.has_value());
    EXPECT_FALSE(geometry->dense);
    EXPECT_EQ(geometry->extents, (std::array<int64_t, 4>{1, 1, int64_t{37} * 8, 4096}));
    EXPECT_EQ(geometry->outStrides, (std::array<int64_t, 4>{0, 0, 4096, 1}));
    EXPECT_EQ(geometry->aStrides, (std::array<int64_t, 4>{0, 0, 4096, 1}));
    EXPECT_EQ(geometry->bStrides, (std::array<int64_t, 4>{0, 0, 1, 0}));
}

TEST(TestFlydslPointwiseCollapse, ALowerRankInputAlignsAtTheInnermostAxis)
{
    const auto out = packed({6, 7, 64});
    const auto bias = packed({64});
    const auto geometry = pw::collapse(out, bias, &out);
    ASSERT_TRUE(geometry.has_value());
    EXPECT_EQ(geometry->extents, (std::array<int64_t, 4>{1, 1, 42, 64}));
    EXPECT_EQ(geometry->aStrides, (std::array<int64_t, 4>{0, 0, 0, 1}));
    EXPECT_EQ(geometry->bStrides, (std::array<int64_t, 4>{0, 0, 64, 1}));
}

TEST(TestFlydslPointwiseCollapse, AUnaryGraphHasNoSecondInputStrides)
{
    const auto out = packed({3, 5});
    const auto geometry = pw::collapse(out, out, nullptr);
    ASSERT_TRUE(geometry.has_value());
    EXPECT_TRUE(geometry->dense);
    EXPECT_EQ(geometry->bStrides, (std::array<int64_t, 4>{0, 0, 0, 0}));
}

TEST(TestFlydslPointwiseCollapse, KeepsAPermutedOutputsAxesApart)
{
    // [B, H, S, D] dims over a BSHD buffer: S and H swap places in memory.
    const pw::OperandShape out{{2, 4, 5, 64}, {int64_t{4} * 5 * 64, 64, int64_t{4} * 64, 1}};
    const auto in = packed({2, 4, 5, 64});
    const auto geometry = pw::collapse(out, in, nullptr);
    ASSERT_TRUE(geometry.has_value());
    EXPECT_FALSE(geometry->dense);
    EXPECT_EQ(geometry->extents, (std::array<int64_t, 4>{2, 4, 5, 64}));
    EXPECT_EQ(geometry->outStrides, (std::array<int64_t, 4>{1280, 64, 256, 1}));
    EXPECT_EQ(geometry->aStrides, (std::array<int64_t, 4>{1280, 320, 64, 1}));
}

TEST(TestFlydslPointwiseCollapse, DropsUnitAxesWhateverTheirStride)
{
    const pw::OperandShape out{{1, 8, 1, 16}, {999, 16, 0, 1}};
    const auto geometry = pw::collapse(out, out, nullptr);
    ASSERT_TRUE(geometry.has_value());
    EXPECT_TRUE(geometry->dense);
    EXPECT_EQ(geometry->extents[3], 128);
}

TEST(TestFlydslPointwiseCollapse, ASingleElementIsDense)
{
    const auto out = packed({1, 1});
    const auto geometry = pw::collapse(out, out, &out);
    ASSERT_TRUE(geometry.has_value());
    EXPECT_TRUE(geometry->dense);
    EXPECT_EQ(geometry->elements, 1);
}

TEST(TestFlydslPointwiseCollapse, RefusesMoreElementsThanTheKernelIndexes)
{
    const auto out = packed({int64_t{1} << 16, int64_t{1} << 15});
    EXPECT_FALSE(pw::collapse(out, out, nullptr).has_value());
}

// ---------------------------------------------------------------------------
// Kernel matcher and score
// ---------------------------------------------------------------------------

KernelDefinition pointwiseKernel(const std::string& operation,
                                 const std::string& in0,
                                 const std::string& in1,
                                 const std::string& out,
                                 int64_t priority = 100)
{
    KernelDefinition kernel;
    kernel.metadata = {{FLYDSL_POINTWISE_DTYPE_FIELD, in0},
                       {FLYDSL_POINTWISE_OPERATION_FIELD, operation},
                       {FLYDSL_POINTWISE_IN1_DTYPE_FIELD, in1},
                       {FLYDSL_POINTWISE_OUT_DTYPE_FIELD, out},
                       {FLYDSL_POINTWISE_BLOCK_THREADS_FIELD, int64_t{256}},
                       {FLYDSL_POINTWISE_VEC_ELEMS_FIELD, int64_t{8}}};
    kernel.priority = priority;
    return kernel;
}

bool kernelMatchesSpec(const PointwiseGraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslPointwiseGraph(spec));
    const auto bound = matchesGraph(FLYDSL_POINTWISE, fixture.context());
    return bound.has_value() && matchesKernel(FLYDSL_POINTWISE, fixture.context(), kernel, *bound);
}

TEST(TestFlydslPointwiseKernelMatcher, AdmitsTheObjectBuiltForTheOperationAndTypes)
{
    const auto spec = withTypes(flydslBinarySpec(PointwiseMode::MUL, {4, 8}, {4, 1}, {4, 8}),
                                DataType::BFLOAT16,
                                DataType::FLOAT,
                                DataType::BFLOAT16);
    EXPECT_TRUE(kernelMatchesSpec(spec, pointwiseKernel("mul", "bf16", "f32", "bf16")));
    EXPECT_FALSE(kernelMatchesSpec(spec, pointwiseKernel("add", "bf16", "f32", "bf16")));
    EXPECT_FALSE(kernelMatchesSpec(spec, pointwiseKernel("mul", "bf16", "bf16", "bf16")));
    EXPECT_FALSE(kernelMatchesSpec(spec, pointwiseKernel("mul", "f32", "f32", "bf16")));
    EXPECT_FALSE(kernelMatchesSpec(spec, pointwiseKernel("mul", "bf16", "f32", "f32")));
}

TEST(TestFlydslPointwiseKernelMatcher, PairsAUnaryGraphWithAnObjectThatHasNoSecondInput)
{
    const auto spec = withTypes(flydslUnarySpec(PointwiseMode::IDENTITY, {4, 8}),
                                DataType::HALF,
                                DataType::HALF,
                                DataType::FLOAT);
    EXPECT_TRUE(kernelMatchesSpec(spec, pointwiseKernel("identity", "f16", "none", "f32")));
    EXPECT_FALSE(kernelMatchesSpec(spec, pointwiseKernel("identity", "f16", "f16", "f32")));
    EXPECT_FALSE(kernelMatchesSpec(spec, pointwiseKernel("identity", "f16", "none", "f16")));
}

TEST(TestFlydslPointwiseKernelMatcher, ReadsEachOperationsOwnSpelling)
{
    const std::vector<std::pair<PointwiseMode, const char*>> unaries{
        {PointwiseMode::SIGMOID_FWD, "sigmoid"},
        {PointwiseMode::SWISH_FWD, "swish"},
        {PointwiseMode::EXP, "exp"},
        {PointwiseMode::SOFTPLUS_FWD, "softplus"},
        {PointwiseMode::NEG, "neg"},
        {PointwiseMode::IDENTITY, "identity"}};
    for(const auto& [mode, name] : unaries)
    {
        EXPECT_TRUE(kernelMatchesSpec(flydslUnarySpec(mode, {2, 2}),
                                      pointwiseKernel(name, "bf16", "none", "bf16")))
            << name;
    }
    const std::vector<std::pair<PointwiseMode, const char*>> binaries{{PointwiseMode::ADD, "add"},
                                                                      {PointwiseMode::SUB, "sub"},
                                                                      {PointwiseMode::MUL, "mul"},
                                                                      {PointwiseMode::DIV, "div"}};
    for(const auto& [mode, name] : binaries)
    {
        EXPECT_TRUE(kernelMatchesSpec(flydslBinarySpec(mode, {2, 2}, {2, 2}, {2, 2}),
                                      pointwiseKernel(name, "bf16", "bf16", "bf16")))
            << name;
    }
}

TEST(TestFlydslPointwiseScore, IsTheDescriptorPriority)
{
    const GraphFixture fixture(
        buildFlydslPointwiseGraph(flydslUnarySpec(PointwiseMode::NEG, {2, 2})));
    EXPECT_EQ(scoreKernel(FLYDSL_POINTWISE,
                          fixture.context(),
                          pointwiseKernel("neg", "bf16", "none", "bf16", 100)),
              100.0);
    EXPECT_EQ(scoreKernel(FLYDSL_POINTWISE,
                          fixture.context(),
                          pointwiseKernel("neg", "bf16", "none", "bf16", 7)),
              7.0);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

TEST(TestFlydslPointwiseDispatch, NeedsNoWorkspace)
{
    const GraphFixture fixture(
        buildFlydslPointwiseGraph(flydslUnarySpec(PointwiseMode::NEG, {2, 2})));
    const auto bound = matchesGraph(FLYDSL_POINTWISE, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(dispatchHandler(FLYDSL_POINTWISE)
                  .workspaceBytes(
                      fixture.context(), *bound, pointwiseKernel("neg", "bf16", "none", "bf16")),
              0U);
}

TEST(TestFlydslPointwiseDispatch, RefusesToPrepareWithoutTheMatchersBindings)
{
    const GraphFixture fixture(
        buildFlydslPointwiseGraph(flydslUnarySpec(PointwiseMode::NEG, {2, 2})));
    EXPECT_THROW(static_cast<void>(dispatchHandler(FLYDSL_POINTWISE)
                                       .prepare(fixture.context(),
                                                BoundTokens{},
                                                pointwiseKernel("neg", "bf16", "none", "bf16"))),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
