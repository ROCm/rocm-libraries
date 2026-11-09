// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvWinogradProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvWinogradProblem.cpp
 * @brief Host-only checks of the Winograd convolution engines (CONV_WINOGRAD_FWD,
 *        CONV_WINOGRAD_BWD): what the graph match accepts, one refusal per gated field, the
 *        forward form it derives, and what the kernel match admits per config and device
 *        (variant family, dtype, tile, layout, the LLC plan). No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using conv_winograd::parseWinogradProblem;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

constexpr ConvDirection DIRECTIONS[] = {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA};

/// The base problem every refusal mutates: 3x3, stride 1, padding 1, NCHW.
ConvShape baseShape()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 4;
    shape.k = 6;
    shape.input = {7, 6};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    return shape;
}

std::optional<conv_winograd::WinogradProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseWinogradProblem(fixture.context(), spec.direction);
}

// ---------------------------------------------------------------------------
// Accepted shapes
// ---------------------------------------------------------------------------

struct AcceptCase
{
    std::string name;
    ConvShape shape;
};

std::vector<AcceptCase> acceptCases()
{
    std::vector<AcceptCase> cases;
    cases.push_back({"Nchw", baseShape()});
    auto nhwc = baseShape();
    nhwc.channelsLast = true;
    cases.push_back({"Nhwc", nhwc});
    auto pad0 = baseShape();
    pad0.prePadding = pad0.postPadding = {0, 0};
    cases.push_back({"Pad0", pad0});
    auto pad2 = baseShape();
    pad2.prePadding = pad2.postPadding = {2, 2};
    cases.push_back({"Pad2", pad2});
    auto asymmetric = baseShape();
    asymmetric.prePadding = {2, 0};
    asymmetric.postPadding = {0, 1};
    cases.push_back({"AsymmetricPadding", asymmetric});
    auto grouped = baseShape();
    grouped.c = 8;
    grouped.k = 4;
    grouped.groups = 2;
    cases.push_back({"Grouped", grouped});
    auto convolution = baseShape();
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"ConvolutionMode", convolution});
    auto half = baseShape();
    half.dataType = DataType::HALF;
    cases.push_back({"Half", half});
    auto bfloat = baseShape();
    bfloat.dataType = DataType::BFLOAT16;
    cases.push_back({"BFloat16", bfloat});
    // Stride 2 (by phases), over every padding, groups, both modes and layouts.
    for(auto shape : {baseShape(), pad0, pad2, asymmetric, grouped, convolution, nhwc, half})
    {
        shape.stride = {2, 2};
        cases.push_back({"Stride2" + std::to_string(cases.size()), shape});
    }
    auto stem = baseShape();
    stem.stride = {2, 2};
    stem.c = 3;
    stem.k = 32;
    stem.input = {15, 16};
    stem.prePadding = stem.postPadding = {0, 0};
    cases.push_back({"Stride2Stem", stem});
    return cases;
}

class TestConvWinogradAccepts
    : public ::testing::TestWithParam<std::tuple<AcceptCase, ConvDirection>>
{
};

TEST_P(TestConvWinogradAccepts, ForwardAndBackwardData)
{
    const auto& [accept, direction] = GetParam();
    EXPECT_TRUE(parse(specFor(accept.shape, direction)).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvWinogradAccepts,
                         ::testing::Combine(::testing::ValuesIn(acceptCases()),
                                            ::testing::ValuesIn(DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

TEST(TestConvWinogradForm, ForwardKeepsTheGraphsRoles)
{
    auto shape = baseShape();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    shape.mode = ConvMode::CONVOLUTION;
    const auto problem = parse(specFor(shape, ConvDirection::FORWARD));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    EXPECT_EQ(a.ci, 4);
    EXPECT_EQ(a.co, 6);
    EXPECT_EQ(a.hi, 7);
    EXPECT_EQ(a.wi, 6);
    EXPECT_EQ(a.ho, 7);
    EXPECT_EQ(a.wo, 5);
    EXPECT_EQ(a.ph, 2);
    EXPECT_EQ(a.pw, 0);
    EXPECT_EQ(a.flip, 1);
    EXPECT_EQ(problem->activations, conv_winograd::Layout::NCHW);
    EXPECT_EQ(problem->filter, conv_winograd::Layout::NCHW);
}

TEST(TestConvWinogradForm, BackwardDataSwapsChannelsAndPadsByTwoMinusPre)
{
    auto shape = baseShape();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    shape.channelsLast = true;
    const auto problem = parse(specFor(shape, ConvDirection::BACKWARD_DATA));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    EXPECT_EQ(a.ci, 6);
    EXPECT_EQ(a.co, 4);
    EXPECT_EQ(a.hi, 7);
    EXPECT_EQ(a.wi, 5);
    EXPECT_EQ(a.ho, 7);
    EXPECT_EQ(a.wo, 6);
    EXPECT_EQ(a.ph, 0);
    EXPECT_EQ(a.pw, 2);
    EXPECT_EQ(a.flip, 0);
    EXPECT_EQ(problem->activations, conv_winograd::Layout::NHWC);
    EXPECT_EQ(problem->filter, conv_winograd::Layout::NHWC);
}

TEST(TestConvWinogradForm, MixedLayoutsAreOther)
{
    auto spec = specFor(baseShape(), ConvDirection::FORWARD);
    spec.y.strides = packedStrides(spec.y.dims, true);
    const auto problem = parse(spec);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->activations, conv_winograd::Layout::OTHER);
}

// ---------------------------------------------------------------------------
// One refusal per gated field
// ---------------------------------------------------------------------------

struct RefusalCase
{
    std::string name;
    std::function<void(ConvGraphSpec&)> mutate;
};

TEST(TestConvWinogradForm, StrideTwoForwardGathersPhasesAsChannels)
{
    auto shape = baseShape();
    shape.stride = {2, 2};
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    const auto problem = parse(specFor(shape, ConvDirection::FORWARD));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    // x 7 x 6 -> y (7 + 2 - 3) / 2 + 1 = 4 by (6 + 1 - 3) / 2 + 1 = 3.
    EXPECT_EQ(a.mode, 1);
    EXPECT_EQ(a.ci, 16);
    EXPECT_EQ(a.co, 6);
    EXPECT_EQ(a.hi, 5);
    EXPECT_EQ(a.wi, 4);
    EXPECT_EQ(a.ho, 4);
    EXPECT_EQ(a.wo, 3);
    EXPECT_EQ(a.ph, 0);
    EXPECT_EQ(a.pw, 0);
    EXPECT_EQ(a.xh, 7);
    EXPECT_EQ(a.xw, 6);
    EXPECT_EQ(a.xph, 2);
    EXPECT_EQ(a.xpw, 0);
}

TEST(TestConvWinogradForm, StrideTwoBackwardScattersPhasesAsChannels)
{
    auto shape = baseShape();
    shape.stride = {2, 2};
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    const auto problem = parse(specFor(shape, ConvDirection::BACKWARD_DATA));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    EXPECT_EQ(a.mode, 2);
    EXPECT_EQ(a.ci, 6);
    EXPECT_EQ(a.co, 16);
    EXPECT_EQ(a.hi, 4);
    EXPECT_EQ(a.wi, 3);
    // Phase rows (7 - 1 + 2) / 2 + 1 = 5, columns (6 - 1 + 0) / 2 + 1 = 3.
    EXPECT_EQ(a.ho, 5);
    EXPECT_EQ(a.wo, 3);
    EXPECT_EQ(a.ph, 1);
    EXPECT_EQ(a.pw, 1);
    EXPECT_EQ(a.xh, 7);
    EXPECT_EQ(a.xw, 6);
    EXPECT_EQ(a.xph, 2);
    EXPECT_EQ(a.xpw, 0);
}

TensorSpec& written(ConvGraphSpec& spec)
{
    return spec.direction == ConvDirection::FORWARD ? spec.y : spec.x;
}

/// Rebuilds the operands of @p spec from @p shape (a consistent change of several fields).
void reshape(ConvGraphSpec& spec, const ConvShape& shape)
{
    const auto rebuilt = specFor(shape, spec.direction);
    spec.x = rebuilt.x;
    spec.w = rebuilt.w;
    spec.y = rebuilt.y;
    spec.prePadding = rebuilt.prePadding;
    spec.postPadding = rebuilt.postPadding;
    spec.stride = rebuilt.stride;
    spec.dilation = rebuilt.dilation;
}

std::vector<RefusalCase> refusalCases()
{
    return {
        // Graph
        {"OverrideShapeEnabled", [](ConvGraphSpec& s) { s.overrideShapeEnabled = true; }},
        // Node
        {"ComputeTypeDouble", [](ConvGraphSpec& s) { s.computeType = DataType::DOUBLE; }},
        {"ModeUnset", [](ConvGraphSpec& s) { s.mode = ConvMode::UNSET; }},
        {"OutputAliasesInput", [](ConvGraphSpec& s) { s.yUid = s.xUid; }},
        {"MissingTensor", [](ConvGraphSpec& s) { s.omitW = true; }},
        // Tensor fields
        {"VirtualOperand", [](ConvGraphSpec& s) { s.x.isVirtual = true; }},
        {"RuntimePassByValue", [](ConvGraphSpec& s) { s.w.runtimePassByValue = true; }},
        {"RaggedOffset", [](ConvGraphSpec& s) { s.y.raggedOffsetUid = 7; }},
        {"RaggedMultiplier", [](ConvGraphSpec& s) { s.y.raggedOffsetMultiplier = 2; }},
        {"UnderAligned", [](ConvGraphSpec& s) { s.x.alignment = 2; }},
        {"Double",
         [](ConvGraphSpec& s) { s.x.dataType = s.w.dataType = s.y.dataType = DataType::DOUBLE; }},
        {"MixedDtype", [](ConvGraphSpec& s) { s.w.dataType = DataType::HALF; }},
        {"StridesRankMismatch", [](ConvGraphSpec& s) { s.x.strides.pop_back(); }},
        {"ZeroExtent", [](ConvGraphSpec& s) { s.x.dims[0] = 0; }},
        {"NegativeStride", [](ConvGraphSpec& s) { s.x.strides[3] = -1; }},
        {"OverlappingOutput",
         [](ConvGraphSpec& s) {
             auto& out = written(s);
             out.strides[out.strides.size() - 1] = 0;
         }},
        // Convolution parameters
        {"PaddingWrongLength", [](ConvGraphSpec& s) { s.prePadding.push_back(0); }},
        {"NegativePadding", [](ConvGraphSpec& s) { s.postPadding[0] = -1; }},
        {"WrongOutputExtent",
         [](ConvGraphSpec& s) {
             s.y.dims[2] += 1;
             s.y.strides = packedStrides(s.y.dims, false);
         }},
        {"FilterChannelsDoNotDivideInput", [](ConvGraphSpec& s) { s.w.dims[1] = 3; }},
        {"BatchMismatch",
         [](ConvGraphSpec& s) {
             s.y.dims[0] = 3;
             s.y.strides = packedStrides(s.y.dims, false);
         }},
        // The Winograd envelope: 2-D, 3x3, stride 1 or 2 (the same on both axes), dilation 1.
        {"OneD",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.input = {7};
             shape.filter = {3};
             shape.prePadding = shape.postPadding = {1};
             shape.stride = shape.dilation = {1};
             reshape(s, shape);
         }},
        {"ThreeD",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.input = {5, 7, 6};
             shape.filter = {3, 3, 3};
             shape.prePadding = shape.postPadding = {1, 1, 1};
             shape.stride = shape.dilation = {1, 1, 1};
             reshape(s, shape);
         }},
        {"FilterHeightNot3",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.filter = {5, 3};
             reshape(s, shape);
         }},
        {"FilterWidthNot3",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.filter = {3, 1};
             shape.prePadding = shape.postPadding = {1, 0};
             reshape(s, shape);
         }},
        {"StrideMixed",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.stride = {1, 2};
             reshape(s, shape);
         }},
        {"StrideMixedTwoFirst",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.stride = {2, 1};
             reshape(s, shape);
         }},
        {"Stride3",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.stride = {3, 3};
             reshape(s, shape);
         }},
        {"DilationNot1",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.dilation = {2, 1};
             shape.prePadding = shape.postPadding = {2, 1};
             reshape(s, shape);
         }},
    };
}

class TestConvWinogradRefuses
    : public ::testing::TestWithParam<std::tuple<RefusalCase, ConvDirection>>
{
};

TEST_P(TestConvWinogradRefuses, OneField)
{
    const auto& [refusal, direction] = GetParam();
    auto spec = specFor(baseShape(), direction);
    ASSERT_TRUE(parse(spec).has_value());
    refusal.mutate(spec);
    EXPECT_FALSE(parse(spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvWinogradRefuses,
                         ::testing::Combine(::testing::ValuesIn(refusalCases()),
                                            ::testing::ValuesIn(DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

TEST(TestConvWinogradDirection, RefusesWeightGradientsAndTheOtherDirection)
{
    const GraphFixture forward(buildGraph(specFor(baseShape(), ConvDirection::FORWARD)));
    EXPECT_FALSE(parseWinogradProblem(forward.context(), ConvDirection::BACKWARD_DATA).has_value());
    EXPECT_FALSE(
        parseWinogradProblem(forward.context(), ConvDirection::BACKWARD_WEIGHTS).has_value());
    const GraphFixture wrw(buildGraph(specFor(baseShape(), ConvDirection::BACKWARD_WEIGHTS)));
    EXPECT_FALSE(parseWinogradProblem(wrw.context(), ConvDirection::BACKWARD_WEIGHTS).has_value());
}

// ---------------------------------------------------------------------------
// Kernel match: configs per device
// ---------------------------------------------------------------------------

constexpr PackSymbols PACK_FWD{"",
                               "hipkernel.conv_winograd_fwd.graph_match",
                               "",
                               "hipkernel.conv_winograd.kernel_match",
                               "",
                               "hipkernel.conv_winograd.dispatch",
                               "",
                               "",
                               ""};

struct ConfigFields
{
    std::string variant = "multipass";
    int64_t m = 2;
    std::string layout = "NCHW";
    std::string arch = "any";
    int64_t llc = 0;
    int64_t gemmBm = 128, gemmBn = 64, gemmBk = 8;
    int64_t bk = 0, bt = 0, ccv = 0, tm = 0, tn = 0, waves = 0;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const ConfigFields& f,
                                                       const std::string& dtype)
{
    auto kernel = makeKernel(0, dtype, "ConvWinogradGemm");
    kernel.source.sourceFile = "kernels/ConvWinograd.cpp";
    kernel.metadata = {{"direction", std::string("FWD")},
                       {"dtype", dtype},
                       {"variant", f.variant},
                       {"m", f.m},
                       {"layout", f.layout},
                       {"arch", f.arch},
                       {"llc", f.llc},
                       {"gemm_bm", f.gemmBm},
                       {"gemm_bn", f.gemmBn},
                       {"gemm_bk", f.gemmBk},
                       {"bk", f.bk},
                       {"bt", f.bt},
                       {"ccv", f.ccv},
                       {"tm", f.tm},
                       {"tn", f.tn},
                       {"waves", f.waves}};
    return kernel;
}

ConfigFields fused()
{
    ConfigFields f;
    f.variant = "fused";
    f.gemmBm = f.gemmBn = f.gemmBk = 0;
    f.bk = 64;
    f.bt = 32;
    f.ccv = 4;
    f.tm = 8;
    f.tn = 8;
    return f;
}

ConfigFields wmma(const std::string& arch)
{
    ConfigFields f;
    f.variant = "wmma";
    f.arch = arch;
    f.gemmBm = f.gemmBn = f.gemmBk = 0;
    f.bk = 64;
    f.waves = 8;
    return f;
}

bool admits(const ConfigFields& fields,
            const std::string& arch,
            ConvShape shape = baseShape(),
            const std::string& dtype = "FLOAT")
{
    shape.dataType = dtype == "FLOAT"  ? DataType::FLOAT
                     : dtype == "HALF" ? DataType::HALF
                                       : DataType::BFLOAT16;
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = 64;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)), properties);
    const auto bound = matchesGraph(PACK_FWD, fixture.context());
    return bound.has_value()
           && matchesKernel(PACK_FWD, fixture.context(), kernelOf(fields, dtype), *bound);
}

TEST(TestConvWinogradKernelMatch, MultipassServesItsOwnLayoutAndStagesOthersToNchw)
{
    ConfigFields nchw;
    ConfigFields nhwc;
    nhwc.layout = "NHWC";
    auto channelsLast = baseShape();
    channelsLast.channelsLast = true;
    EXPECT_TRUE(admits(nchw, "gfx942"));
    EXPECT_FALSE(admits(nhwc, "gfx942"));
    EXPECT_TRUE(admits(nhwc, "gfx942", channelsLast));
    EXPECT_FALSE(admits(nchw, "gfx942", channelsLast));
}

TEST(TestConvWinogradKernelMatch, FourByFourTilesOnlyForFloat)
{
    ConfigFields f4;
    f4.m = 4;
    EXPECT_TRUE(admits(f4, "gfx906"));
    EXPECT_FALSE(admits(f4, "gfx906", baseShape(), "HALF"));
    EXPECT_FALSE(admits(f4, "gfx906", baseShape(), "BFLOAT16"));
}

TEST(TestConvWinogradKernelMatch, FusedRunsEverywhereAndStagesNhwc)
{
    auto channelsLast = baseShape();
    channelsLast.channelsLast = true;
    EXPECT_TRUE(admits(fused(), "gfx906"));
    EXPECT_TRUE(admits(fused(), "gfx1201", channelsLast, "HALF"));
}

TEST(TestConvWinogradKernelMatch, WmmaOnlyOnItsFamilyAndOnlyForSixteenBitTypes)
{
    EXPECT_TRUE(admits(wmma("gfx11"), "gfx1100", baseShape(), "HALF"));
    EXPECT_TRUE(admits(wmma("gfx11"), "gfx1151", baseShape(), "BFLOAT16"));
    EXPECT_FALSE(admits(wmma("gfx11"), "gfx1201", baseShape(), "HALF"));
    EXPECT_TRUE(admits(wmma("gfx12"), "gfx1201", baseShape(), "HALF"));
    EXPECT_FALSE(admits(wmma("gfx12"), "gfx942", baseShape(), "HALF"));
    EXPECT_FALSE(admits(wmma("gfx11"), "gfx1100", baseShape(), "FLOAT"));
}

TEST(TestConvWinogradKernelMatch, TheLlcPlanOnlyWhereItSlices)
{
    ConfigFields llc;
    llc.llc = 1;
    // No last-level cache table entry: nothing to fit.
    EXPECT_FALSE(admits(llc, "gfx942"));
    // A 96 MiB cache holds this small problem whole: the plan would not slice.
    EXPECT_FALSE(admits(llc, "gfx1100"));
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tiny(
        "HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES", "4096");
    EXPECT_TRUE(admits(llc, "gfx942"));
}

ConvShape strideTwo()
{
    auto shape = baseShape();
    shape.stride = {2, 2};
    return shape;
}

TEST(TestConvWinogradKernelMatch, ThreeByTwoTilesExactlyForStrideTwo)
{
    ConfigFields f2;
    ConfigFields f3;
    f3.m = 3;
    ConfigFields f4;
    f4.m = 4;
    EXPECT_TRUE(admits(f3, "gfx942", strideTwo()));
    EXPECT_TRUE(admits(f3, "gfx942", strideTwo(), "HALF"));
    EXPECT_FALSE(admits(f3, "gfx942"));
    EXPECT_FALSE(admits(f2, "gfx942", strideTwo()));
    EXPECT_FALSE(admits(f4, "gfx942", strideTwo()));
    auto fused3 = fused();
    fused3.m = 3;
    EXPECT_TRUE(admits(fused3, "gfx906", strideTwo()));
    EXPECT_FALSE(admits(fused3, "gfx906"));
    EXPECT_FALSE(admits(fused(), "gfx906", strideTwo()));
    auto wmma3 = wmma("gfx11");
    wmma3.m = 3;
    EXPECT_FALSE(admits(wmma3, "gfx1100", strideTwo(), "HALF"));
    EXPECT_FALSE(admits(wmma("gfx11"), "gfx1100", strideTwo(), "HALF"));
}

TEST(TestConvWinogradKernelMatch, FusedStrideTwoReadsWholeChannelsPerChunk)
{
    // A chunk of ccv dwords (fp16: pairs) must hold the 4 phases of whole channels.
    auto chunk2 = fused();
    chunk2.m = 3;
    chunk2.ccv = 2;
    chunk2.bk = 32;
    chunk2.tm = 8;
    chunk2.tn = 4;
    EXPECT_FALSE(admits(chunk2, "gfx906", strideTwo()));
    EXPECT_TRUE(admits(chunk2, "gfx906", strideTwo(), "HALF"));
    EXPECT_FALSE(admits(chunk2, "gfx906", strideTwo(), "BFLOAT16"));
}

TEST(TestConvWinogradKernelMatch, RefusesUnknownMetadata)
{
    auto bad = fused();
    bad.variant = "fft";
    EXPECT_FALSE(admits(bad, "gfx942"));
    auto badLayout = ConfigFields{};
    badLayout.layout = "NCDHW";
    EXPECT_FALSE(admits(badLayout, "gfx942"));
    auto badTile = ConfigFields{};
    badTile.m = 6;
    EXPECT_FALSE(admits(badTile, "gfx942"));
}

TEST(TestConvWinogradPlan, SlicesTheBatchUnderTheCacheAndSizesTheWorkspace)
{
    auto shape = baseShape();
    shape.n = 8;
    auto properties = testDeviceProperties();
    properties.gcnArchName = "gfx1100";
    properties.multiProcessorCount = 48;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)), properties);
    const auto problem = parseWinogradProblem(fixture.context(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    const auto config = conv_winograd::configOf(kernelOf(ConfigFields{}, "FLOAT"));
    ASSERT_TRUE(config.has_value());
    const auto whole = conv_winograd::planFor(*problem, *config, properties);
    ASSERT_TRUE(whole.has_value());
    EXPECT_EQ(whole->slices, 1);
    // U, V and M of F(2,3): 16 positions x (K x C, C x tiles, K x tiles fp32).
    const size_t tiles = 8 * 4 * 3;
    EXPECT_EQ(whole->uBytes, size_t{16 * 6 * 4 * 4});
    EXPECT_EQ(whole->vBytes, size_t{16 * 4 * 4} * tiles);
    EXPECT_EQ(whole->mBytes, size_t{16 * 6 * 4} * tiles);
    EXPECT_GE(whole->workspaceBytes, whole->uBytes + whole->vBytes + whole->mBytes);

    auto llcFields = ConfigFields{};
    llcFields.llc = 1;
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter small(
        "HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES", "20000");
    const auto llc = conv_winograd::planFor(
        *problem, *conv_winograd::configOf(kernelOf(llcFields, "FLOAT")), properties);
    ASSERT_TRUE(llc.has_value());
    EXPECT_GT(llc->slices, 1);
    EXPECT_LE(llc->uBytes + llc->vBytes + llc->mBytes, size_t{20000} / 4 * 3);
}

TEST(TestConvWinogradPlan, StrideTwoStagesTheGraphsOperandsAndUsesSixteenPositions)
{
    auto shape = baseShape();
    shape.stride = {2, 2};
    shape.channelsLast = true;
    for(const auto direction : DIRECTIONS)
    {
        const GraphFixture fixture(buildGraph(specFor(shape, direction)));
        const auto problem = parseWinogradProblem(fixture.context(), direction);
        ASSERT_TRUE(problem.has_value());
        ConfigFields f3;
        f3.m = 3;
        f3.layout = "NHWC";
        const auto multipass = conv_winograd::configOf(kernelOf(f3, "FLOAT"));
        ASSERT_TRUE(multipass.has_value());
        const auto native = conv_winograd::planFor(*problem, *multipass, testDeviceProperties());
        ASSERT_TRUE(native.has_value());
        const auto& a = problem->args;
        const auto cgi = static_cast<size_t>(a.ci), cgo = static_cast<size_t>(a.co),
                   tiles = static_cast<size_t>(native->args.tiles);
        EXPECT_EQ(native->slices, 1);
        EXPECT_FALSE(native->in.needed);
        EXPECT_EQ(native->uBytes, 16 * cgo * cgi * 4);
        EXPECT_EQ(native->vBytes, 16 * cgi * tiles * 4);
        EXPECT_EQ(native->mBytes, 16 * cgo * tiles * 4);

        auto fused3 = fused();
        fused3.m = 3;
        const auto staged = conv_winograd::planFor(
            *problem, *conv_winograd::configOf(kernelOf(fused3, "FLOAT")), testDeviceProperties());
        ASSERT_TRUE(staged.has_value());
        // NHWC staged to NCHW: x is 2 x 4 x 7 x 6, y 2 x 6 x 4 x 3, in their graph extents.
        const size_t xBytes = 2 * 4 * 7 * 6 * 4, yBytes = 2 * 6 * 4 * 3 * 4;
        ASSERT_TRUE(staged->in.needed);
        EXPECT_EQ(staged->in.bytes, direction == ConvDirection::FORWARD ? xBytes : yBytes);
        EXPECT_EQ(staged->out.bytes, direction == ConvDirection::FORWARD ? yBytes : xBytes);
    }
}

} // namespace

#endif
