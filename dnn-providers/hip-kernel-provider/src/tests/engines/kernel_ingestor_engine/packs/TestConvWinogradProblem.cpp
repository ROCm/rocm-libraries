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
 *        CONV_WINOGRAD_BWD, CONV_WINOGRAD_WRW): what the graph match accepts, one refusal per
 *        gated field, the
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
constexpr ConvDirection ALL_DIRECTIONS[]
    = {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS};

std::string directionName(ConvDirection direction)
{
    return direction == ConvDirection::FORWARD         ? "Fwd"
           : direction == ConvDirection::BACKWARD_DATA ? "Bwd"
                                                       : "Wrw";
}

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

class TestConvWinogradWrwAccepts : public ::testing::TestWithParam<AcceptCase>
{
};

TEST_P(TestConvWinogradWrwAccepts, StrideOneOnly)
{
    const auto& accept = GetParam();
    const auto parsed = parse(specFor(accept.shape, ConvDirection::BACKWARD_WEIGHTS));
    // Backward weights serves stride 1 (F(3,r) tiles over dy); stride 2 is refused.
    EXPECT_EQ(parsed.has_value(), accept.shape.stride.empty() || accept.shape.stride[0] == 1);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvWinogradWrwAccepts,
                         ::testing::ValuesIn(acceptCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvWinogradForm, BackwardWeightsKeepsTheGraphsRolesAndDwStrides)
{
    auto shape = baseShape();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    shape.mode = ConvMode::CONVOLUTION;
    shape.channelsLast = true;
    const auto problem = parse(specFor(shape, ConvDirection::BACKWARD_WEIGHTS));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->wrw;
    EXPECT_EQ(a.n, 2);
    EXPECT_EQ(a.c, 4);
    EXPECT_EQ(a.k, 6);
    EXPECT_EQ(a.hi, 7);
    EXPECT_EQ(a.wi, 6);
    EXPECT_EQ(a.ho, 7);
    EXPECT_EQ(a.wo, 5);
    EXPECT_EQ(a.ph, 2);
    EXPECT_EQ(a.pw, 0);
    EXPECT_EQ(a.flip, 1);
    // KRSC: (k, c, r, s) strides (36, 1, 12, 4).
    EXPECT_EQ(a.dwStrides[0], 36);
    EXPECT_EQ(a.dwStrides[1], 1);
    EXPECT_EQ(a.dwStrides[2], 12);
    EXPECT_EQ(a.dwStrides[3], 4);
    EXPECT_EQ(problem->activations, conv_winograd::Layout::NHWC);
}

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
    return spec.direction == ConvDirection::FORWARD         ? spec.y
           : spec.direction == ConvDirection::BACKWARD_DATA ? spec.x
                                                            : spec.w;
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
                                            ::testing::ValuesIn(ALL_DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + directionName(std::get<1>(info.param));
                         });

TEST(TestConvWinogradDirection, EachDirectionRefusesTheOthersGraphs)
{
    for(const auto graph : ALL_DIRECTIONS)
    {
        const GraphFixture fixture(buildGraph(specFor(baseShape(), graph)));
        for(const auto direction : ALL_DIRECTIONS)
        {
            SCOPED_TRACE(directionName(graph) + " graph as " + directionName(direction));
            EXPECT_EQ(parseWinogradProblem(fixture.context(), direction).has_value(),
                      graph == direction);
        }
    }
}

TEST(TestConvWinogradDirection, BackwardWeightsRefusesStrideTwo)
{
    auto shape = baseShape();
    shape.stride = {2, 2};
    EXPECT_FALSE(parse(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)).has_value());
    EXPECT_TRUE(parse(specFor(shape, ConvDirection::FORWARD)).has_value());
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
    int64_t bins = 0, f16math = 0, pf = 0;
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
                       {"waves", f.waves},
                       {"bins", f.bins},
                       {"f16math", f.f16math},
                       {"pf", f.pf}};
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

/// Register-built WMMA: 32-wide K block, 2 tile groups, whole transform rows per wave.
ConfigFields wreg(const std::string& arch, int64_t f16math)
{
    ConfigFields f;
    f.variant = "wreg";
    f.arch = arch;
    f.gemmBm = f.gemmBn = f.gemmBk = 0;
    f.bk = 32;
    f.bt = 32;
    f.waves = 8;
    f.bins = 4;
    f.f16math = f16math;
    return f;
}

/// Cooperative WMMA: 16 waves (4 K groups of 4 rows), 16-channel K tiles, 32 tiles: bk 64.
ConfigFields coop(const std::string& arch, int64_t f16math)
{
    ConfigFields f;
    f.variant = "coop";
    f.arch = arch;
    f.gemmBm = f.gemmBn = f.gemmBk = 0;
    f.bk = 64;
    f.bt = 32;
    f.waves = 16;
    f.f16math = f16math;
    return f;
}

/// Register-operand VALU (wave64 gfx9): 16-wide K block, 64 tiles, 4 waves.
ConfigFields sreg(int64_t f16math)
{
    ConfigFields f;
    f.variant = "sreg";
    f.arch = "gfx9";
    f.gemmBm = f.gemmBn = f.gemmBk = 0;
    f.bk = 16;
    f.bt = 64;
    f.waves = 4;
    f.f16math = f16math;
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
    // Prefetch depth 2 (pf 1).
    auto prefetch = fused();
    prefetch.pf = 1;
    EXPECT_TRUE(admits(prefetch, "gfx1201"));
    EXPECT_TRUE(admits(prefetch, "gfx906"));
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

TEST(TestConvWinogradKernelMatch, RegisterWmmaOnlyOnItsFamilyAndOnlyForSixteenBitTypes)
{
    EXPECT_TRUE(admits(wreg("gfx11", 1), "gfx1100", baseShape(), "HALF"));
    EXPECT_TRUE(admits(wreg("gfx11", 0), "gfx1151", baseShape(), "BFLOAT16"));
    EXPECT_TRUE(admits(wreg("gfx12", 1), "gfx1201", baseShape(), "HALF"));
    EXPECT_FALSE(admits(wreg("gfx11", 1), "gfx1201", baseShape(), "HALF"));
    EXPECT_FALSE(admits(wreg("gfx12", 1), "gfx942", baseShape(), "HALF"));
    EXPECT_FALSE(admits(wreg("gfx11", 0), "gfx1100", baseShape(), "FLOAT"));
    // Packed fp16 transform math is for fp16 only.
    EXPECT_FALSE(admits(wreg("gfx11", 1), "gfx1100", baseShape(), "BFLOAT16"));
    // F(2,3): stride 1 only.
    EXPECT_FALSE(admits(wreg("gfx11", 1), "gfx1100", strideTwo(), "HALF"));
}

TEST(TestConvWinogradKernelMatch, CooperativeWmmaOnlyOnItsFamilyAndOnlyForSixteenBitTypes)
{
    EXPECT_TRUE(admits(coop("gfx11", 1), "gfx1100", baseShape(), "HALF"));
    EXPECT_TRUE(admits(coop("gfx11", 0), "gfx1151", baseShape(), "BFLOAT16"));
    EXPECT_TRUE(admits(coop("gfx12", 1), "gfx1201", baseShape(), "HALF"));
    EXPECT_FALSE(admits(coop("gfx11", 1), "gfx1201", baseShape(), "HALF"));
    EXPECT_FALSE(admits(coop("gfx12", 1), "gfx942", baseShape(), "HALF"));
    EXPECT_FALSE(admits(coop("gfx11", 0), "gfx1100", baseShape(), "FLOAT"));
    // Packed fp16 transform math is for fp16 only.
    EXPECT_FALSE(admits(coop("gfx11", 1), "gfx1100", baseShape(), "BFLOAT16"));
    // F(2,3): stride 1 only.
    EXPECT_FALSE(admits(coop("gfx11", 1), "gfx1100", strideTwo(), "HALF"));
    // F(3,2) on RDNA: stride 2 (by phases) only.
    auto f32 = coop("gfx11", 1);
    f32.m = 3;
    EXPECT_TRUE(admits(f32, "gfx1100", strideTwo(), "HALF"));
    EXPECT_FALSE(admits(f32, "gfx1100", baseShape(), "HALF"));
    auto f32Bf16 = coop("gfx12", 0);
    f32Bf16.m = 3;
    EXPECT_TRUE(admits(f32Bf16, "gfx1201", strideTwo(), "BFLOAT16"));
    EXPECT_FALSE(admits(f32Bf16, "gfx1100", strideTwo(), "BFLOAT16"));
}

TEST(TestConvWinogradKernelMatch, CooperativeMfmaOnCdnaFromGfx908AndBFloat16FromGfx90a)
{
    EXPECT_TRUE(admits(coop("gfx9", 1), "gfx942:sramecc+:xnack-", baseShape(), "HALF"));
    EXPECT_TRUE(admits(coop("gfx9", 1), "gfx950", baseShape(), "HALF"));
    EXPECT_TRUE(admits(coop("gfx9", 1), "gfx908", baseShape(), "HALF"));
    EXPECT_TRUE(admits(coop("gfx9", 0), "gfx90a", baseShape(), "BFLOAT16"));
    // Cross-item prefetch (pf 1).
    auto prefetch = coop("gfx9", 1);
    prefetch.pf = 1;
    EXPECT_TRUE(admits(prefetch, "gfx950", baseShape(), "HALF"));
    // bf16 MFMA (bf16_1k) starts at gfx90a.
    EXPECT_FALSE(admits(coop("gfx9", 0), "gfx908", baseShape(), "BFLOAT16"));
    // No MFMA before gfx908, and not on RDNA.
    EXPECT_FALSE(admits(coop("gfx9", 1), "gfx906", baseShape(), "HALF"));
    EXPECT_FALSE(admits(coop("gfx9", 1), "gfx1100", baseShape(), "HALF"));
    EXPECT_FALSE(admits(coop("gfx9", 0), "gfx942", baseShape(), "FLOAT"));
    // F(3,2): stride 2 (by phases) only.
    auto f32 = coop("gfx9", 1);
    f32.m = 3;
    EXPECT_TRUE(admits(f32, "gfx950", strideTwo(), "HALF"));
    EXPECT_TRUE(admits(f32, "gfx908", strideTwo(), "HALF"));
    EXPECT_FALSE(admits(f32, "gfx942", baseShape(), "HALF"));
}

TEST(TestConvWinogradKernelMatch, RegisterValuOnlyOnWave64Gfx9AndNotForBFloat16)
{
    EXPECT_TRUE(admits(sreg(0), "gfx906"));
    EXPECT_TRUE(admits(sreg(1), "gfx906", baseShape(), "HALF"));
    EXPECT_TRUE(admits(sreg(0), "gfx942:sramecc+:xnack-", baseShape(), "HALF"));
    EXPECT_FALSE(admits(sreg(0), "gfx900"));
    EXPECT_FALSE(admits(sreg(0), "gfx1100"));
    EXPECT_FALSE(admits(sreg(0), "gfx906", baseShape(), "BFLOAT16"));
    EXPECT_FALSE(admits(sreg(1), "gfx906"));
    EXPECT_FALSE(admits(sreg(0), "gfx906", strideTwo()));
}

TEST(TestConvWinogradKernelMatch, SixteenWideFusedBlocksServeSmallGroupsAtBothStrides)
{
    auto block16 = fused();
    block16.bk = 16;
    block16.ccv = 4;
    block16.tn = 4;
    auto grouped = baseShape();
    grouped.c = 64;
    grouped.k = 64;
    grouped.groups = 4;
    EXPECT_TRUE(admits(block16, "gfx906", grouped));
    block16.m = 3;
    auto groupedStrideTwo = grouped;
    groupedStrideTwo.stride = {2, 2};
    EXPECT_TRUE(admits(block16, "gfx906", groupedStrideTwo));
}

TEST(TestConvWinogradKernelMatch, RefusesInconsistentRegisterVariantFields)
{
    // One decline per field the register variants gate.
    const std::vector<std::pair<std::string, ConfigFields (*)()>> cases{
        {"WregBins3",
         [] {
             auto f = wreg("gfx11", 1);
             f.bins = 3;
             return f;
         }},
        {"WregWavesNotFromBinsAndTiles",
         [] {
             auto f = wreg("gfx11", 1);
             f.waves = 4;
             return f;
         }},
        {"WregTileBlock48",
         [] {
             auto f = wreg("gfx11", 1);
             f.bt = 48;
             f.waves = 12;
             return f;
         }},
        {"WregKBlock16",
         [] {
             auto f = wreg("gfx11", 1);
             f.bk = 16;
             return f;
         }},
        {"WregEpilogueOver64KiB",
         [] {
             auto f = wreg("gfx11", 1);
             f.bk = 64;
             f.bins = 2;
             f.waves = 16;
             return f;
         }},
        {"WregArchAny",
         [] {
             auto f = wreg("gfx11", 1);
             f.arch = "any";
             return f;
         }},
        {"WregTileF43",
         [] {
             auto f = wreg("gfx11", 1);
             f.m = 4;
             return f;
         }},
        {"WregLlc",
         [] {
             auto f = wreg("gfx11", 1);
             f.llc = 1;
             return f;
         }},
        {"F16MathTwo",
         [] {
             auto f = wreg("gfx11", 1);
             f.f16math = 2;
             return f;
         }},
        {"CoopWaves12",
         [] {
             auto f = coop("gfx11", 1);
             f.waves = 12;
             return f;
         }},
        {"CoopWaves64",
         [] {
             auto f = coop("gfx11", 1);
             f.waves = 64;
             f.bk = 256;
             return f;
         }},
        {"CoopTileBlock128",
         [] {
             auto f = coop("gfx11", 1);
             f.bt = 128;
             return f;
         }},
        {"CoopOverFourAccumulatorTiles",
         [] {
             auto f = coop("gfx11", 1);
             f.bk = 128; // mt 2 with 4 K groups
             f.bt = 64; // nt 4
             return f;
         }},
        {"CoopKBlockNotSixteenPerKGroup",
         [] {
             auto f = coop("gfx11", 1);
             f.bk = 48;
             return f;
         }},
        {"CoopKBlockThreeTilesPerKGroup",
         [] {
             auto f = coop("gfx11", 1);
             f.bk = 192;
             return f;
         }},
        {"CoopKBlockEightTilesPerKGroup",
         [] {
             auto f = coop("gfx9", 1);
             f.bk = 512;
             return f;
         }},
        {"CoopCdnaWaves32",
         [] {
             auto f = coop("gfx9", 1);
             f.waves = 32;
             f.bk = 128;
             return f;
         }},
        {"CoopArchAny",
         [] {
             auto f = coop("gfx11", 1);
             f.arch = "any";
             return f;
         }},
        {"CoopTileF43",
         [] {
             auto f = coop("gfx11", 1);
             f.m = 4;
             return f;
         }},
        {"CoopLlc",
         [] {
             auto f = coop("gfx11", 1);
             f.llc = 1;
             return f;
         }},
        {"CoopPrefetchTwo",
         [] {
             auto f = coop("gfx9", 1);
             f.pf = 2;
             return f;
         }},
        {"FusedPrefetchTwo",
         [] {
             auto f = fused();
             f.pf = 2;
             return f;
         }},
        {"SregPrefetch",
         [] {
             auto f = sreg(0);
             f.pf = 1;
             return f;
         }},
        {"CoopNhwc",
         [] {
             auto f = coop("gfx11", 1);
             f.layout = "NHWC";
             return f;
         }},
        {"CoopBins",
         [] {
             auto f = coop("gfx11", 1);
             f.bins = 4;
             return f;
         }},
        {"SregKBlock32",
         [] {
             auto f = sreg(0);
             f.bk = 32;
             return f;
         }},
        {"SregTileBlock32",
         [] {
             auto f = sreg(0);
             f.bt = 32;
             return f;
         }},
        {"SregWaves8",
         [] {
             auto f = sreg(0);
             f.waves = 8;
             return f;
         }},
        {"SregArchAny",
         [] {
             auto f = sreg(0);
             f.arch = "any";
             return f;
         }},
        {"SregBins",
         [] {
             auto f = sreg(0);
             f.bins = 4;
             return f;
         }},
        {"FusedBins",
         [] {
             auto f = fused();
             f.bins = 4;
             return f;
         }},
        {"FusedF16Math",
         [] {
             auto f = fused();
             f.f16math = 1;
             return f;
         }},
        {"WmmaF16Math",
         [] {
             auto f = wmma("gfx11");
             f.f16math = 1;
             return f;
         }},
    };
    for(const auto& [name, make] : cases)
    {
        SCOPED_TRACE(name);
        const auto fields = make();
        const auto arch = fields.arch == "gfx9" ? "gfx906" : "gfx1100";
        EXPECT_FALSE(conv_winograd::configOf(kernelOf(fields, "HALF")).has_value());
        EXPECT_FALSE(admits(fields, arch, baseShape(), "HALF"));
    }
    // The unmodified configs parse, so each decline above is its field's.
    EXPECT_TRUE(conv_winograd::configOf(kernelOf(wreg("gfx11", 1), "HALF")).has_value());
    EXPECT_TRUE(conv_winograd::configOf(kernelOf(sreg(0), "HALF")).has_value());
    EXPECT_TRUE(conv_winograd::configOf(kernelOf(coop("gfx11", 1), "HALF")).has_value());
    // 32 waves (8 K groups) are legal on wave32 RDNA, not on wave64 CDNA (checked above).
    auto rdna32 = coop("gfx11", 1);
    rdna32.waves = 32;
    rdna32.bk = 128;
    EXPECT_TRUE(conv_winograd::configOf(kernelOf(rdna32, "HALF")).has_value());
    // CDNA takes up to 4 x 4 accumulator tiles per bin.
    auto cdnaWide = coop("gfx9", 1);
    cdnaWide.bk = 256; // mt 4 with 4 K groups
    cdnaWide.bt = 64;
    EXPECT_TRUE(conv_winograd::configOf(kernelOf(cdnaWide, "HALF")).has_value());
    // The largest legal tile counts parse: mt 1 x nt 4 and mt 2 x nt 2.
    auto wide = coop("gfx11", 1);
    wide.bt = 64;
    EXPECT_TRUE(conv_winograd::configOf(kernelOf(wide, "HALF")).has_value());
    auto deep = coop("gfx11", 1);
    deep.bk = 128;
    EXPECT_TRUE(conv_winograd::configOf(kernelOf(deep, "HALF")).has_value());
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

/// A device that reports @p units dispatch units (the persistent variants size their grid).
hipdnn_plugin_sdk::ingestor::DeviceProperties countedDevice(int units = 64)
{
    auto properties = testDeviceProperties();
    properties.multiProcessorCount = units;
    return properties;
}

TEST(TestConvWinogradPlan, PersistentVariantsDeclineADeviceWithoutUnits)
{
    const GraphFixture fixture(buildGraph(specFor(baseShape(), ConvDirection::FORWARD)));
    const auto problem = parseWinogradProblem(fixture.context(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    // The fused and register variants launch a persistent grid of blocks per unit.
    for(const auto& fields : {fused(), sreg(0)})
    {
        const auto config = conv_winograd::configOf(kernelOf(fields, "FLOAT"));
        ASSERT_TRUE(config.has_value());
        EXPECT_TRUE(conv_winograd::planFor(*problem, *config, countedDevice()).has_value());
        EXPECT_FALSE(conv_winograd::planFor(*problem, *config, countedDevice(0)).has_value());
        EXPECT_FALSE(conv_winograd::planFor(*problem, *config, countedDevice(-1)).has_value());
    }
    // Multipass sizes its grids from the problem alone.
    const auto multipass = conv_winograd::configOf(kernelOf(ConfigFields{}, "FLOAT"));
    ASSERT_TRUE(multipass.has_value());
    EXPECT_TRUE(conv_winograd::planFor(*problem, *multipass, countedDevice(0)).has_value());
}

TEST(TestConvWinogradPlan, CooperativeOrdersTileBlocksFirstForALargeTransformedFilter)
{
    // On RDNA a group's U of 4 MiB and up (16 positions x K x C x 2 bytes) is re-read per tile
    // block when K blocks run fastest: the plan switches to tile blocks fastest.
    for(const auto& [channels, order] : {std::pair<int64_t, int>{256, 1}, {240, 0}})
    {
        auto shape = baseShape();
        shape.dataType = DataType::HALF;
        shape.c = channels;
        shape.k = 512;
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)));
        const auto problem = parseWinogradProblem(fixture.context(), ConvDirection::FORWARD);
        ASSERT_TRUE(problem.has_value());
        const auto config = conv_winograd::configOf(kernelOf(coop("gfx11", 1), "HALF"));
        ASSERT_TRUE(config.has_value());
        const auto plan = conv_winograd::planFor(*problem, *config, countedDevice());
        ASSERT_TRUE(plan.has_value());
        EXPECT_EQ(plan->args.order, order) << channels;
        // CDNA keeps K blocks fastest.
        const auto cdna = conv_winograd::configOf(kernelOf(coop("gfx9", 1), "HALF"));
        ASSERT_TRUE(cdna.has_value());
        const auto cdnaPlan = conv_winograd::planFor(*problem, *cdna, countedDevice());
        ASSERT_TRUE(cdnaPlan.has_value());
        EXPECT_EQ(cdnaPlan->args.order, 0) << channels;
    }
}

TEST(TestConvWinogradPlan, FusedSlicesStayClearOfTheOutOfRangeOffset)
{
    // A padding column adds K5_OOB (2^30) to an offset formed from the tile's start, which is
    // -pad x bytes for a tile in the left padding; the wave32 register variant takes 16 bytes off
    // a padding row's K5_OOB. A batch slice of exactly 2^30 bytes (images whose size divides it)
    // put those loads in range: data read where the padding should read zero.
    struct Case
    {
        ConfigFields fields;
        const char* arch;
        DataType dataType;
        int64_t n, c, extent;
    };
    // 16 MiB fp32 images x 64, 2 MiB fp16 images x 512: 1 GiB each.
    for(const auto& test : {Case{fused(), "gfx942", DataType::FLOAT, 64, 64, 256},
                            Case{wreg("gfx11", 0), "gfx1100", DataType::HALF, 512, 64, 128}})
    {
        auto shape = baseShape();
        shape.dataType = test.dataType;
        shape.n = test.n;
        shape.c = test.c;
        shape.k = 16;
        shape.input = {test.extent, test.extent};
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)));
        const auto problem = parseWinogradProblem(fixture.context(), ConvDirection::FORWARD);
        ASSERT_TRUE(problem.has_value());
        const auto config = conv_winograd::configOf(
            kernelOf(test.fields, test.dataType == DataType::FLOAT ? "FLOAT" : "HALF"));
        ASSERT_TRUE(config.has_value());
        auto device = countedDevice();
        device.gcnArchName = test.arch;
        const auto plan = conv_winograd::planFor(*problem, *config, device);
        ASSERT_TRUE(plan.has_value()) << test.arch;
        const int64_t imageBytes
            = test.c * test.extent * test.extent * (test.dataType == DataType::FLOAT ? 4 : 2);
        EXPECT_LE(int64_t{plan->sliceImages} * imageBytes, (int64_t{1} << 30) - (int64_t{1} << 20))
            << test.arch;
        EXPECT_EQ(plan->sliceImages, test.n - 1) << test.arch;
        EXPECT_EQ(plan->slices, 2) << test.arch;
    }
}

TEST(TestConvWinogradPlan, RegisterValuPadsTheFilterBufferByOneKBlock)
{
    const GraphFixture fixture(buildGraph(specFor(baseShape(), ConvDirection::FORWARD)));
    const auto problem = parseWinogradProblem(fixture.context(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    const auto config = conv_winograd::configOf(kernelOf(sreg(0), "FLOAT"));
    ASSERT_TRUE(config.has_value());
    const auto plan = conv_winograd::planFor(*problem, *config, countedDevice());
    ASSERT_TRUE(plan.has_value());
    // 16 positions x C 4 x K 6 fp32, plus 16 out channels' worth of dwords.
    EXPECT_EQ(plan->uBytes, size_t{16 * 4 * 6 * 4 + 16 * 4});
    EXPECT_EQ(plan->mainThreads, 256u);
}

TEST(TestConvWinogradPlan, CooperativeFilterBufferHoldsWholeSixteenChannelUnits)
{
    // K 6 and C 4 both pad to one 16 x 16 unit per position (k5CoopFilterTransform), fp16.
    auto shape = baseShape();
    shape.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)));
    const auto problem = parseWinogradProblem(fixture.context(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    const auto config = conv_winograd::configOf(kernelOf(coop("gfx9", 1), "HALF"));
    ASSERT_TRUE(config.has_value());
    const auto plan = conv_winograd::planFor(*problem, *config, countedDevice());
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->uBytes, size_t{16 * 16 * 16 * 2});
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
            *problem, *conv_winograd::configOf(kernelOf(fused3, "FLOAT")), countedDevice());
        ASSERT_TRUE(staged.has_value());
        // NHWC staged to NCHW: x is 2 x 4 x 7 x 6, y 2 x 6 x 4 x 3, in their graph extents.
        const size_t xBytes = 2 * 4 * 7 * 6 * 4, yBytes = 2 * 6 * 4 * 3 * 4;
        ASSERT_TRUE(staged->in.needed);
        EXPECT_EQ(staged->in.bytes, direction == ConvDirection::FORWARD ? xBytes : yBytes);
        EXPECT_EQ(staged->out.bytes, direction == ConvDirection::FORWARD ? yBytes : xBytes);
    }
}

// ---------------------------------------------------------------------------
// Backward weights (CONV_WINOGRAD_WRW): kernel match, one refusal per config field, plan
// ---------------------------------------------------------------------------

constexpr PackSymbols PACK_WRW{"",
                               "hipkernel.conv_winograd_wrw.graph_match",
                               "",
                               "hipkernel.conv_winograd.kernel_match",
                               "",
                               "hipkernel.conv_winograd.dispatch",
                               "",
                               "",
                               ""};

/// An FMA config (fp32 operands, every arch): F(3,2), 64 x 64 x 16, 4 x 4 thread tiles.
ConfigFields wrwFma()
{
    ConfigFields f;
    f.variant = "wrw";
    f.m = 2;
    f.gemmBm = f.gemmBn = 64;
    f.gemmBk = 16;
    f.tm = f.tn = 4;
    return f;
}

/// A matrix-core config: 64 x 64 x 32, 2 x 2 atoms per wave (4 waves).
ConfigFields wrwAtom(const std::string& arch)
{
    auto f = wrwFma();
    f.arch = arch;
    f.gemmBk = 32;
    f.tm = f.tn = 2;
    f.waves = 4;
    return f;
}

hipdnn_plugin_sdk::ingestor::KernelDefinition wrwKernelOf(const ConfigFields& f,
                                                          const std::string& dtype)
{
    auto kernel = kernelOf(f, dtype);
    kernel.metadata["direction"] = std::string("WRW");
    return kernel;
}

DataType dataTypeNamed(const std::string& dtype)
{
    return dtype == "FLOAT"  ? DataType::FLOAT
           : dtype == "HALF" ? DataType::HALF
                             : DataType::BFLOAT16;
}

bool admitsWrw(const ConfigFields& fields,
               const std::string& arch,
               ConvShape shape = baseShape(),
               const std::string& dtype = "FLOAT")
{
    shape.dataType = dataTypeNamed(dtype);
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = 64;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                               properties);
    const auto bound = matchesGraph(PACK_WRW, fixture.context());
    return bound.has_value()
           && matchesKernel(PACK_WRW, fixture.context(), wrwKernelOf(fields, dtype), *bound);
}

TEST(TestConvWinogradWrwKernelMatch, FmaRunsEverywhereWithinEachDtypesErrorBand)
{
    // fp32 operands on every arch; fp32 graphs stop at F(3,4) (F(3,6) measured past MIOpen's
    // Winograd summation error).
    for(const auto* arch : {"gfx906", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201"})
    {
        for(const auto* dtype : {"FLOAT", "HALF", "BFLOAT16"})
        {
            for(const int64_t r : {2, 4, 6})
            {
                auto f = wrwFma();
                f.m = r;
                SCOPED_TRACE(std::string(arch) + " " + dtype + " r" + std::to_string(r));
                // (baseShape's K is 84: fp32 F(3,4) needs 512.)
                EXPECT_EQ(admitsWrw(f, arch, baseShape(), dtype),
                          r == 2 || std::string(dtype) != "FLOAT");
            }
        }
    }
}

TEST(TestConvWinogradWrwKernelMatch, ChannelsFirstConfigsServeEveryLayout)
{
    // The transforms read channels-first; channels-last graphs are transposed in. NHWC configs
    // are refused.
    auto nhwc = wrwFma();
    nhwc.layout = "NHWC";
    auto channelsLast = baseShape();
    channelsLast.channelsLast = true;
    EXPECT_TRUE(admitsWrw(wrwFma(), "gfx942"));
    EXPECT_TRUE(admitsWrw(wrwFma(), "gfx942", channelsLast));
    EXPECT_FALSE(admitsWrw(nhwc, "gfx942", channelsLast));
    EXPECT_FALSE(admitsWrw(nhwc, "gfx942"));
}

TEST(TestConvWinogradWrwKernelMatch, Fp32FourByThreeFromFiveHundredTwelveProducts)
{
    // fp32 F(3,4)'s summation error is inside the band from K = N x Ho x Wo = 512 on.
    auto f4 = wrwFma();
    f4.m = 4;
    auto shape = baseShape(); // n 2, 7 x 6: K 84
    EXPECT_FALSE(admitsWrw(f4, "gfx942", shape, "FLOAT"));
    EXPECT_TRUE(admitsWrw(f4, "gfx942", shape, "HALF"));
    EXPECT_TRUE(admitsWrw(wrwFma(), "gfx942", shape, "FLOAT"));
    shape.input = {16, 16}; // K 512
    EXPECT_TRUE(admitsWrw(f4, "gfx942", shape, "FLOAT"));
    shape.input = {16, 15}; // K 480
    EXPECT_FALSE(admitsWrw(f4, "gfx942", shape, "FLOAT"));
}

TEST(TestConvWinogradWrwKernelMatch, MatrixCoreAtomsOnlyOnTheirFamilies)
{
    // MFMA 16x16 (gfx908 and later CDNA): F(3,2) on fp32 and fp16 operands; bf16 graphs only
    // on fp32 operands (F(3,4) / F(3,6)).
    auto bf16Tile = wrwAtom("gfx9");
    bf16Tile.m = 4;
    for(const auto* device : {"gfx908", "gfx950"})
    {
        EXPECT_TRUE(admitsWrw(wrwAtom("gfx9"), device, baseShape(), "FLOAT"));
        EXPECT_TRUE(admitsWrw(wrwAtom("gfx9"), device, baseShape(), "HALF"));
        EXPECT_FALSE(admitsWrw(wrwAtom("gfx9"), device, baseShape(), "BFLOAT16"));
        EXPECT_TRUE(admitsWrw(bf16Tile, device, baseShape(), "BFLOAT16"));
    }
    EXPECT_FALSE(admitsWrw(wrwAtom("gfx9"), "gfx906"));
    EXPECT_FALSE(admitsWrw(wrwAtom("gfx9"), "gfx1201"));
    // gfx950's 16x16x32: gfx950 only, fp16 only.
    EXPECT_TRUE(admitsWrw(wrwAtom("gfx950"), "gfx950", baseShape(), "HALF"));
    EXPECT_FALSE(admitsWrw(wrwAtom("gfx950"), "gfx942", baseShape(), "HALF"));
    EXPECT_FALSE(admitsWrw(wrwAtom("gfx950"), "gfx950", baseShape(), "FLOAT"));
    EXPECT_FALSE(admitsWrw(wrwAtom("gfx950"), "gfx950", baseShape(), "BFLOAT16"));
    // WMMA: its family, fp16 only.
    EXPECT_TRUE(admitsWrw(wrwAtom("gfx11"), "gfx1100", baseShape(), "HALF"));
    EXPECT_FALSE(admitsWrw(wrwAtom("gfx11"), "gfx1100", baseShape(), "BFLOAT16"));
    EXPECT_TRUE(admitsWrw(wrwAtom("gfx12"), "gfx1201", baseShape(), "HALF"));
    EXPECT_FALSE(admitsWrw(wrwAtom("gfx11"), "gfx1201", baseShape(), "HALF"));
    EXPECT_FALSE(admitsWrw(wrwAtom("gfx12"), "gfx1201", baseShape(), "FLOAT"));
}

TEST(TestConvWinogradWrwKernelMatch, LargerTilesRunOnFp32Operands)
{
    for(const int64_t r : {4, 6})
    {
        SCOPED_TRACE(r);
        auto fma = wrwFma();
        fma.m = r;
        auto mfma = wrwAtom("gfx9");
        mfma.m = r;
        EXPECT_TRUE(admitsWrw(fma, "gfx942", baseShape(), "BFLOAT16"));
        // CDNA's fp32 MFMA on fp32 operands, for every dtype.
        auto bigger = baseShape();
        bigger.input = {16, 16};
        EXPECT_EQ(admitsWrw(mfma, "gfx942", bigger, "FLOAT"), r == 4);
        EXPECT_TRUE(admitsWrw(mfma, "gfx942", baseShape(), "BFLOAT16"));
        EXPECT_TRUE(admitsWrw(mfma, "gfx908", baseShape(), "HALF"));
        // The 16-bit-only atoms: F(3,2) only.
        for(const auto* arch : {"gfx950", "gfx11", "gfx12"})
        {
            auto sixteen = wrwAtom(arch);
            sixteen.m = r;
            const std::string device = std::string(arch) == "gfx950"  ? "gfx950"
                                       : std::string(arch) == "gfx11" ? "gfx1100"
                                                                      : "gfx1201";
            EXPECT_FALSE(admitsWrw(sixteen, device, baseShape(), "HALF"));
            EXPECT_FALSE(admitsWrw(sixteen, device, baseShape(), "BFLOAT16"));
        }
    }
    const auto plan = [](const ConfigFields& fields, DataType dataType) {
        auto shape = baseShape();
        shape.dataType = dataType;
        auto properties = testDeviceProperties();
        properties.gcnArchName = "gfx942";
        properties.multiProcessorCount = 64;
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                                   properties);
        const auto problem
            = parseWinogradProblem(fixture.context(), ConvDirection::BACKWARD_WEIGHTS);
        return conv_winograd::planFor(*problem,
                                      *conv_winograd::configOf(wrwKernelOf(
                                          fields, dataType == DataType::HALF ? "HALF" : "FLOAT")),
                                      properties);
    };
    auto f2 = wrwAtom("gfx9");
    auto f4 = wrwAtom("gfx9");
    f4.m = 4;
    // F(3,2) of a 16-bit graph: 16-bit operands on the f16 MFMA; F(3,4): fp32 operands, f32 MFMA.
    EXPECT_EQ(plan(f2, DataType::HALF)->operandBytes, 2);
    EXPECT_EQ(plan(f2, DataType::HALF)->wrwAtom, 1);
    EXPECT_EQ(plan(f4, DataType::HALF)->operandBytes, 4);
    EXPECT_EQ(plan(f4, DataType::HALF)->wrwOpt, 0);
    EXPECT_EQ(plan(f4, DataType::HALF)->wrwAtom, 1);
}

TEST(TestConvWinogradWrwKernelMatch, HalfOperandFmaOnlyForHalfGraphsThreeByTwo)
{
    // f16math: the FMA path on fp16 operands (v_dot2_f32_f16), for fp16 graphs' F(3,2) only.
    auto dot2 = wrwFma();
    dot2.f16math = 1;
    EXPECT_TRUE(admitsWrw(dot2, "gfx906", baseShape(), "HALF"));
    EXPECT_TRUE(admitsWrw(dot2, "gfx1201", baseShape(), "HALF"));
    EXPECT_FALSE(admitsWrw(dot2, "gfx906", baseShape(), "BFLOAT16"));
    EXPECT_FALSE(admitsWrw(dot2, "gfx906", baseShape(), "FLOAT"));
    auto larger = dot2;
    larger.m = 4;
    EXPECT_FALSE(admitsWrw(larger, "gfx906", baseShape(), "HALF"));
    auto atom = wrwAtom("gfx9");
    atom.f16math = 1;
    EXPECT_FALSE(admitsWrw(atom, "gfx942", baseShape(), "HALF"));
    auto shape = baseShape();
    shape.dataType = DataType::HALF;
    auto properties = testDeviceProperties();
    properties.gcnArchName = "gfx906";
    properties.multiProcessorCount = 60;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                               properties);
    const auto problem = parseWinogradProblem(fixture.context(), ConvDirection::BACKWARD_WEIGHTS);
    ASSERT_TRUE(problem.has_value());
    const auto plan = conv_winograd::planFor(
        *problem, *conv_winograd::configOf(wrwKernelOf(dot2, "HALF")), properties);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->wrwOpt, 1);
    EXPECT_EQ(plan->operandBytes, 2);
    EXPECT_EQ(plan->wrwAtom, 0);
}

TEST(TestConvWinogradWrwKernelMatch, DirectionAndVariantMustAgree)
{
    // A wrw config never serves forward graphs, a forward config never serves wrw graphs.
    auto forwardTagged = wrwKernelOf(wrwFma(), "FLOAT");
    forwardTagged.metadata["direction"] = std::string("FWD");
    auto shape = baseShape();
    const GraphFixture forward(buildGraph(specFor(shape, ConvDirection::FORWARD)),
                               testDeviceProperties());
    const auto forwardBound = matchesGraph(PACK_FWD, forward.context());
    ASSERT_TRUE(forwardBound.has_value());
    EXPECT_FALSE(matchesKernel(PACK_FWD, forward.context(), forwardTagged, *forwardBound));
    EXPECT_FALSE(admitsWrw(ConfigFields{}, "gfx942"));
}

TEST(TestConvWinogradWrwKernelMatch, RefusesEachInconsistentField)
{
    // One decline per field the wrw variant gates (each starting from an admitted config).
    struct Case
    {
        std::string name;
        ConfigFields fields;
        std::string arch = "gfx942";
        std::string dtype = "FLOAT";
    };
    std::vector<Case> cases;
    const auto fma = [&](const std::string& name, const std::function<void(ConfigFields&)>& edit) {
        auto f = wrwFma();
        edit(f);
        cases.push_back({name, f});
    };
    const auto atom = [&](const std::string& name,
                          const std::string& arch,
                          const std::string& device,
                          const std::string& dtype,
                          const std::function<void(ConfigFields&)>& edit) {
        auto f = wrwAtom(arch);
        edit(f);
        cases.push_back({name, f, device, dtype});
    };
    fma("TileThree", [](ConfigFields& f) { f.m = 3; });
    fma("TileEight", [](ConfigFields& f) { f.m = 8; });
    fma("LlcPlan", [](ConfigFields& f) { f.llc = 1; });
    fma("LlcPlanTwo", [](ConfigFields& f) { f.llc = 2; });
    fma("FusedKBlock", [](ConfigFields& f) { f.bk = 64; });
    fma("FusedTileBlock", [](ConfigFields& f) { f.bt = 32; });
    fma("FusedChunk", [](ConfigFields& f) { f.ccv = 4; });
    fma("Bins", [](ConfigFields& f) { f.bins = 4; });
    fma("F16MathTwo", [](ConfigFields& f) { f.f16math = 2; });
    fma("Prefetch", [](ConfigFields& f) { f.pf = 1; });

    fma("UnknownLayout", [](ConfigFields& f) { f.layout = "NCDHW"; });
    fma("NoOutChannelBlock", [](ConfigFields& f) { f.gemmBm = 0; });
    fma("NoInChannelBlock", [](ConfigFields& f) { f.gemmBn = 0; });
    fma("TileStepNotDividing64", [](ConfigFields& f) { f.gemmBk = 24; });
    fma("TileStepUnderFour", [](ConfigFields& f) { f.gemmBk = 2; });
    // 96 / 11 rows (or columns) of threads: whole waves, but 96 is not a multiple of 11.
    fma("ThreadRowsNotDividing", [](ConfigFields& f) {
        f.gemmBm = 96;
        f.tm = 11;
    });
    fma("ThreadColumnsNotDividing", [](ConfigFields& f) {
        f.gemmBn = 96;
        f.tn = 11;
    });
    fma("FmaWithWaves", [](ConfigFields& f) { f.waves = 4; });
    fma("FmaUnderOneWave", [](ConfigFields& f) {
        f.tm = f.tn = 8;
        f.gemmBm = f.gemmBn = 32;
    });
    fma("FmaOver1024Threads", [](ConfigFields& f) { f.tm = f.tn = 1; });
    fma("FmaPartialWave", [](ConfigFields& f) {
        f.gemmBn = 48;
        f.tm = 8;
    });
    fma("LdsOver64KiB", [](ConfigFields& f) {
        f.gemmBm = f.gemmBn = 512;
        f.gemmBk = 32;
        f.tm = f.tn = 16;
    });
    // An arch no atom family has (fp16, F(3,2): no other gate applies).
    atom("UnknownArch", "gfx10", "gfx1030", "HALF", [](ConfigFields&) {});
    atom("AtomWavesWrong", "gfx9", "gfx942", "HALF", [](ConfigFields& f) { f.waves = 2; });
    // 48 rows (columns) of 2 atoms per wave: one wave along them, consistently counted.
    atom("AtomRowsNotWhole", "gfx9", "gfx942", "HALF", [](ConfigFields& f) {
        f.gemmBm = 48;
        f.waves = 2;
    });
    atom("AtomColumnsNotWhole", "gfx9", "gfx942", "HALF", [](ConfigFields& f) {
        f.gemmBn = 48;
        f.waves = 2;
    });
    atom("AtomNoOutChannelBlock", "gfx9", "gfx942", "HALF", [](ConfigFields& f) {
        f.gemmBm = 0;
        f.waves = 0;
    });
    atom("AtomNoInChannelBlock", "gfx9", "gfx942", "HALF", [](ConfigFields& f) {
        f.gemmBn = 0;
        f.waves = 0;
    });
    atom("AtomTileStepUnderKa", "gfx9", "gfx942", "HALF", [](ConfigFields& f) { f.gemmBk = 8; });
    atom("Gfx950TileStepUnder32", "gfx950", "gfx950", "HALF", [](ConfigFields& f) {
        f.gemmBk = 16;
    });
    atom("AtomOver1024Threads", "gfx9", "gfx942", "HALF", [](ConfigFields& f) {
        f.gemmBm = f.gemmBn = 128;
        f.tm = f.tn = 1;
        f.waves = 64;
    });
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        EXPECT_FALSE(admitsWrw(c.fields, c.arch, baseShape(), c.dtype));
    }
    // The bases are admitted.
    EXPECT_TRUE(admitsWrw(wrwFma(), "gfx942"));
    EXPECT_TRUE(admitsWrw(wrwAtom("gfx9"), "gfx942", baseShape(), "HALF"));
    EXPECT_TRUE(admitsWrw(wrwAtom("gfx950"), "gfx950", baseShape(), "HALF"));
}

std::optional<conv_winograd::Plan> wrwPlan(const ConvShape& shape,
                                           const ConfigFields& fields,
                                           int units = 64,
                                           const std::string& arch = "gfx942")
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = units;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                               properties);
    const auto problem = parseWinogradProblem(fixture.context(), ConvDirection::BACKWARD_WEIGHTS);
    const auto config = conv_winograd::configOf(wrwKernelOf(fields, "FLOAT"));
    if(!problem.has_value() || !config.has_value())
    {
        return std::nullopt;
    }
    return conv_winograd::planFor(*problem, *config, properties);
}

TEST(TestConvWinogradWrwPlan, SizesTransformsAndPartialsAndPadsTheTiles)
{
    // n2 7x6 -> dy 7x6 at padding 1; F(3,2): 4 x 3 tiles per image, 24 tiles -> padded to 64.
    auto f = wrwFma();
    const auto plan = wrwPlan(baseShape(), f, 1);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->wrw.th, 4);
    EXPECT_EQ(plan->wrw.tw, 3);
    EXPECT_EQ(plan->wrw.tiles, 24);
    EXPECT_EQ(plan->wrw.ldt, 64);
    EXPECT_EQ(plan->slices, 1);
    EXPECT_EQ(plan->wrw.splits, 1);
    EXPECT_EQ(plan->uBytes, size_t{16 * 6 * 64 * 4});
    EXPECT_EQ(plan->vBytes, size_t{16 * 4 * 64 * 4});
    EXPECT_EQ(plan->partBytes, size_t{16 * 6 * 4 * 4});
    EXPECT_GE(plan->workspaceBytes, plan->uBytes + plan->vBytes + plan->partBytes);
    // F(3,4): 36 positions, 2 x 2 tiles per image.
    f.m = 4;
    const auto big = wrwPlan(baseShape(), f, 1);
    ASSERT_TRUE(big.has_value());
    EXPECT_EQ(big->wrw.tiles, 8);
    EXPECT_EQ(big->uBytes, size_t{36 * 6 * 64 * 4});
}

TEST(TestConvWinogradWrwPlan, SplitsTheTilesUntilTheBlocksFillTheDevice)
{
    auto shape = baseShape();
    shape.n = 64;
    shape.input = {32, 32};
    // 16 positions x 1 block: 64 units want splits; each split a whole number of 64 tiles.
    const auto plan = wrwPlan(shape, wrwFma(), 64);
    ASSERT_TRUE(plan.has_value());
    EXPECT_GT(plan->wrw.splits, 1);
    EXPECT_EQ(plan->wrw.splitTiles % 64, 0);
    EXPECT_GE(static_cast<long long>(plan->wrw.splits) * plan->wrw.splitTiles, plan->wrw.ldt);
    EXPECT_LT(static_cast<long long>(plan->wrw.splits - 1) * plan->wrw.splitTiles, plan->wrw.ldt);
    EXPECT_EQ(plan->partBytes, size_t{16 * 6 * 4 * 4} * static_cast<size_t>(plan->wrw.splits));
    // A device reporting no units: one split.
    const auto unknown = wrwPlan(shape, wrwFma(), 0);
    ASSERT_TRUE(unknown.has_value());
    EXPECT_EQ(unknown->wrw.splits, 1);
}

TEST(TestConvWinogradWrwPlan, SlicesTheBatchUnderTheWorkspaceLimit)
{
    auto shape = baseShape();
    shape.n = 64;
    shape.c = 64;
    shape.k = 64;
    shape.input = {32, 32};
    const auto whole = wrwPlan(shape, wrwFma());
    ASSERT_TRUE(whole.has_value());
    EXPECT_EQ(whole->slices, 1);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter limit(
        "HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB", "8");
    const auto sliced = wrwPlan(shape, wrwFma());
    ASSERT_TRUE(sliced.has_value());
    EXPECT_GT(sliced->slices, 1);
    EXPECT_LE(sliced->workspaceBytes, size_t{8} << 20);
    EXPECT_EQ(sliced->wrw.n, sliced->sliceImages);
    EXPECT_EQ(sliced->wrw.tiles, sliced->sliceImages * 16 * 16);
    // One image past the limit: declined.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tiny(
        "HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB", "1");
    EXPECT_FALSE(wrwPlan(shape, wrwFma()).has_value());
}

TEST(TestConvWinogradWrwPlan, StagesChannelsLastByTransposeAndMixedByCopy)
{
    auto properties = testDeviceProperties();
    properties.multiProcessorCount = 64;
    const auto planOf = [&](const ConvGraphSpec& spec) {
        const GraphFixture fixture(buildGraph(spec), properties);
        const auto problem
            = parseWinogradProblem(fixture.context(), ConvDirection::BACKWARD_WEIGHTS);
        return conv_winograd::planFor(
            *problem, *conv_winograd::configOf(wrwKernelOf(wrwFma(), "FLOAT")), properties);
    };
    // Channels-first: read in place.
    const auto nchw = planOf(specFor(baseShape(), ConvDirection::BACKWARD_WEIGHTS));
    ASSERT_TRUE(nchw.has_value());
    EXPECT_FALSE(nchw->in.needed);
    EXPECT_FALSE(nchw->out.needed);
    // Packed channels-last: the tiled transpose, dims (N, H x W, C), 64 x 64 tiles per image.
    auto channelsLast = baseShape();
    channelsLast.channelsLast = true;
    const auto nhwc = planOf(specFor(channelsLast, ConvDirection::BACKWARD_WEIGHTS));
    ASSERT_TRUE(nhwc.has_value());
    EXPECT_TRUE(nhwc->in.needed && nhwc->in.transpose);
    EXPECT_TRUE(nhwc->out.needed && nhwc->out.transpose);
    EXPECT_EQ(nhwc->in.copy.dims[0], 2);
    EXPECT_EQ(nhwc->in.copy.dims[1], 7 * 6);
    EXPECT_EQ(nhwc->in.copy.dims[2], 4);
    EXPECT_EQ(nhwc->out.copy.dims[2], 6);
    EXPECT_EQ(nhwc->in.transposeBlocks[0], 1u);
    EXPECT_EQ(nhwc->in.transposeBlocks[1], 1u);
    EXPECT_EQ(nhwc->in.transposeBlocks[2], 2u);
    EXPECT_EQ(nhwc->in.bytes, size_t{2 * 4 * 7 * 6 * 4});
    // Mixed layouts: the element-wise copy.
    auto spec = specFor(baseShape(), ConvDirection::BACKWARD_WEIGHTS);
    spec.y.strides = packedStrides(spec.y.dims, true);
    const auto mixed = planOf(spec);
    ASSERT_TRUE(mixed.has_value());
    EXPECT_TRUE(mixed->in.needed && !mixed->in.transpose);
    EXPECT_TRUE(mixed->out.needed && !mixed->out.transpose);
    EXPECT_EQ(mixed->out.bytes, size_t{2 * 6 * 7 * 6 * 4});
}

TEST(TestConvWinogradComputeType, ServesCudnnsComputeTypesPerElementType)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        expectCudnnComputeTypes(
            baseShape(),
            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
            [direction](const ConvShape& s) { return parse(specFor(s, direction)).has_value(); });
    }
}

} // namespace

#endif
