// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvDirectProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvDirectProblem.cpp
 * @brief Host-only checks of the stride-1 halo-tile direct convolution engines
 *        (CONV_DIRECT_FWD, CONV_DIRECT_BWD): what the graph match accepts, one refusal per gated
 *        field, the forward form it derives, what the kernel match admits per config and device
 *        (atom family, ship list, dtype, filter, channel vector, three-bf16 permission), and the
 *        planner's tile, split and workspace. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_direct::parseDirectProblem;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

constexpr ConvDirection DIRECTIONS[] = {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA};

/// The base problem every refusal mutates: 3x3, stride 1, padding 1, NCHW.
ConvShape baseShape()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 8;
    shape.k = 16;
    shape.input = {7, 6};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    return shape;
}

ConvShape filterShape(std::vector<int64_t> filter, std::vector<int64_t> pad)
{
    auto shape = baseShape();
    shape.input.assign(filter.size(), 9);
    shape.filter = std::move(filter);
    shape.prePadding = pad;
    shape.postPadding = std::move(pad);
    shape.stride.assign(shape.input.size(), 1);
    shape.dilation.assign(shape.input.size(), 1);
    return shape;
}

std::optional<conv_direct::DirectProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseDirectProblem(fixture.context(), spec.direction);
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
    auto convolution = baseShape();
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"ConvolutionMode", convolution});
    auto half = baseShape();
    half.dataType = DataType::HALF;
    cases.push_back({"Half", half});
    auto bfloat = baseShape();
    bfloat.dataType = DataType::BFLOAT16;
    cases.push_back({"BFloat16", bfloat});
    auto oddChannels = baseShape();
    oddChannels.c = 3;
    oddChannels.k = 5;
    cases.push_back({"OddChannels", oddChannels});
    cases.push_back({"Filter5x5", filterShape({5, 5}, {2, 2})});
    cases.push_back({"Filter7x7", filterShape({7, 7}, {3, 3})});
    cases.push_back({"Filter1x7", filterShape({1, 7}, {0, 3})});
    cases.push_back({"Filter7x1", filterShape({7, 1}, {3, 0})});
    cases.push_back({"Filter3x3x3", filterShape({3, 3, 3}, {1, 1, 1})});
    auto ndhwc = filterShape({3, 3, 3}, {0, 1, 2});
    ndhwc.channelsLast = true;
    cases.push_back({"Ndhwc3x3x3", ndhwc});
    return cases;
}

class TestConvDirectAccepts : public ::testing::TestWithParam<std::tuple<AcceptCase, ConvDirection>>
{
};

TEST_P(TestConvDirectAccepts, ForwardAndBackwardData)
{
    const auto& [accept, direction] = GetParam();
    EXPECT_TRUE(parse(specFor(accept.shape, direction)).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvDirectAccepts,
                         ::testing::Combine(::testing::ValuesIn(acceptCases()),
                                            ::testing::ValuesIn(DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

TEST(TestConvDirectForm, ForwardKeepsTheGraphsRolesAndStagesNchw)
{
    auto shape = baseShape();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    shape.mode = ConvMode::CONVOLUTION;
    const auto problem = parse(specFor(shape, ConvDirection::FORWARD));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    EXPECT_EQ(a.c, 8);
    EXPECT_EQ(a.k, 16);
    EXPECT_EQ(a.hi, 7);
    EXPECT_EQ(a.wi, 6);
    EXPECT_EQ(a.ho, 7);
    EXPECT_EQ(a.wo, 5);
    EXPECT_EQ(a.di, 1);
    EXPECT_EQ(a.dout, 1);
    EXPECT_EQ(a.ph, 2);
    EXPECT_EQ(a.pw, 0);
    EXPECT_EQ(problem->flip, 1);
    EXPECT_TRUE(problem->in.needed);
    EXPECT_TRUE(problem->in.transpose);
    EXPECT_TRUE(problem->out.needed);
    EXPECT_GT(problem->stagingBytes, size_t{0});
}

TEST(TestConvDirectForm, BackwardDataSwapsChannelsAndPadsByFilterMinusOneMinusPre)
{
    auto shape = baseShape();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    shape.channelsLast = true;
    const auto problem = parse(specFor(shape, ConvDirection::BACKWARD_DATA));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    EXPECT_EQ(a.c, 16);
    EXPECT_EQ(a.k, 8);
    EXPECT_EQ(a.hi, 7);
    EXPECT_EQ(a.wi, 5);
    EXPECT_EQ(a.ho, 7);
    EXPECT_EQ(a.wo, 6);
    EXPECT_EQ(a.ph, 0);
    EXPECT_EQ(a.pw, 2);
    EXPECT_EQ(problem->flip, 0);
    EXPECT_FALSE(problem->in.needed);
    EXPECT_FALSE(problem->out.needed);
    EXPECT_EQ(problem->stagingBytes, size_t{0});
}

// ---------------------------------------------------------------------------
// One refusal per gated field
// ---------------------------------------------------------------------------

struct RefusalCase
{
    std::string name;
    std::function<void(ConvGraphSpec&)> mutate;
};

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
        // The direct envelope: 2-D or 3-D, groups 1, stride 1, dilation 1, an instantiated filter.
        {"OneD",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.input = {7};
             shape.filter = {3};
             shape.prePadding = shape.postPadding = {1};
             shape.stride = shape.dilation = {1};
             reshape(s, shape);
         }},
        {"Grouped",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.groups = 2;
             reshape(s, shape);
         }},
        {"StrideNot1",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.stride = {1, 2};
             reshape(s, shape);
         }},
        {"DilationNot1",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.dilation = {2, 1};
             shape.prePadding = shape.postPadding = {2, 1};
             reshape(s, shape);
         }},
        {"Filter3x5", [](ConvGraphSpec& s) { reshape(s, filterShape({3, 5}, {1, 2})); }},
        {"Filter1x1", [](ConvGraphSpec& s) { reshape(s, filterShape({1, 1}, {0, 0})); }},
        {"Filter5x5x5", [](ConvGraphSpec& s) { reshape(s, filterShape({5, 5, 5}, {2, 2, 2})); }},
        {"Filter3x3x1", [](ConvGraphSpec& s) { reshape(s, filterShape({3, 3, 1}, {1, 1, 0})); }},
    };
}

class TestConvDirectRefuses
    : public ::testing::TestWithParam<std::tuple<RefusalCase, ConvDirection>>
{
};

TEST_P(TestConvDirectRefuses, OneField)
{
    const auto& [refusal, direction] = GetParam();
    auto spec = specFor(baseShape(), direction);
    ASSERT_TRUE(parse(spec).has_value());
    refusal.mutate(spec);
    EXPECT_FALSE(parse(spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvDirectRefuses,
                         ::testing::Combine(::testing::ValuesIn(refusalCases()),
                                            ::testing::ValuesIn(DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

TEST(TestConvDirectDirection, RefusesWeightGradientsAndTheOtherDirection)
{
    const GraphFixture forward(buildGraph(specFor(baseShape(), ConvDirection::FORWARD)));
    EXPECT_FALSE(parseDirectProblem(forward.context(), ConvDirection::BACKWARD_DATA).has_value());
    EXPECT_FALSE(
        parseDirectProblem(forward.context(), ConvDirection::BACKWARD_WEIGHTS).has_value());
    const GraphFixture wrw(buildGraph(specFor(baseShape(), ConvDirection::BACKWARD_WEIGHTS)));
    EXPECT_FALSE(parseDirectProblem(wrw.context(), ConvDirection::BACKWARD_WEIGHTS).has_value());
}

TEST(TestConvDirectSize, RefusesAnImageBeyondTheBufferOffsets)
{
    auto shape = baseShape();
    shape.n = 1;
    shape.c = 8;
    shape.k = 8;
    shape.input = {16384, 16400};
    shape.channelsLast = true;
    // 16384 x 16400 x 8 fp32: one image of x is 2^33 bytes.
    EXPECT_FALSE(parse(specFor(shape, ConvDirection::FORWARD)).has_value());
}

// ---------------------------------------------------------------------------
// Kernel match: configs per device
// ---------------------------------------------------------------------------

constexpr PackSymbols PACK_FWD{"",
                               "hipkernel.conv_direct_fwd.graph_match",
                               "",
                               "hipkernel.conv_direct.kernel_match",
                               "",
                               "hipkernel.conv_direct.dispatch",
                               "",
                               "",
                               ""};

struct ConfigFields
{
    std::string kind = "mma";
    std::string atom = "mfma16_f16";
    std::string archs = "gfx942";
    int64_t math = 0;
    int64_t kd = 1, r = 3, s = 3;
    int64_t bm = 128, bn = 64, bc = 32, wm = 2, wn = 2, vec = 8;
    int64_t ring = 1, bl = 0, split = 0;
    int64_t tm = 0, tn = 0, nt = 0, threads = 0, pipe = 0;
};

ConfigFields fmaFields(int64_t vec)
{
    ConfigFields f;
    f.kind = "fma";
    f.atom = "fma";
    f.archs = "any";
    f.bm = 128;
    f.bn = 64;
    f.bc = 8;
    f.wm = f.wn = 0;
    f.vec = vec;
    f.tm = 8;
    f.tn = 4;
    f.nt = 16;
    f.threads = 256;
    return f;
}

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const ConfigFields& f,
                                                       const std::string& dtype)
{
    auto kernel = makeKernel(0, dtype, "ConvDirect");
    kernel.source.sourceFile = "kernels/ConvDirect.cpp";
    kernel.metadata = {{"direction", std::string("FWD")},
                       {"dtype", dtype},
                       {"kind", f.kind},
                       {"atom", f.atom},
                       {"archs", f.archs},
                       {"atom_id", int64_t{0}},
                       {"math", f.math},
                       {"kd", f.kd},
                       {"r", f.r},
                       {"s", f.s},
                       {"bm", f.bm},
                       {"bn", f.bn},
                       {"bc", f.bc},
                       {"wm", f.wm},
                       {"wn", f.wn},
                       {"vec", f.vec},
                       {"ring", f.ring},
                       {"bl", f.bl},
                       {"split", f.split},
                       {"tm", f.tm},
                       {"tn", f.tn},
                       {"nt", f.nt},
                       {"threads", f.threads},
                       {"pipe", f.pipe}};
    return kernel;
}

bool admits(const ConfigFields& fields,
            const std::string& arch,
            ConvShape shape = baseShape(),
            const std::string& dtype = "HALF")
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

TEST(TestConvDirectKernelMatch, MatrixCoreConfigsRunOnlyWhereTheirAtomAndShipListAllow)
{
    const ConfigFields f;
    EXPECT_TRUE(admits(f, "gfx942:sramecc+:xnack-"));
    // The atom exists on gfx90a, but this config does not ship there.
    EXPECT_FALSE(admits(f, "gfx90a"));
    // Listed for gfx1100, but no MFMA there.
    auto rdnaListed = f;
    rdnaListed.archs = "gfx942,gfx11";
    EXPECT_FALSE(admits(rdnaListed, "gfx1100"));
    auto wmma = f;
    wmma.atom = "wmma11_f16";
    wmma.archs = "gfx11";
    wmma.wm = 4;
    EXPECT_TRUE(admits(wmma, "gfx1100"));
    EXPECT_TRUE(admits(wmma, "gfx1151"));
    EXPECT_FALSE(admits(wmma, "gfx1201"));
    EXPECT_FALSE(admits(wmma, "gfx942"));
}

TEST(TestConvDirectKernelMatch, TheAtomTakesTheGraphsElementType)
{
    const ConfigFields f;
    EXPECT_TRUE(admits(f, "gfx942", baseShape(), "HALF"));
    EXPECT_FALSE(admits(f, "gfx942", baseShape(), "BFLOAT16"));
    EXPECT_FALSE(admits(f, "gfx942", baseShape(), "FLOAT"));
    auto exact = f;
    exact.atom = "mfma16_f32";
    exact.bc = 16;
    exact.vec = 4;
    EXPECT_TRUE(admits(exact, "gfx942", baseShape(), "FLOAT"));
}

TEST(TestConvDirectKernelMatch, TheFilterMustBeTheConfigs)
{
    ConfigFields f;
    f.r = f.s = 5;
    EXPECT_FALSE(admits(f, "gfx942"));
    EXPECT_TRUE(admits(f, "gfx942", filterShape({5, 5}, {2, 2})));
    ConfigFields threeD;
    threeD.kd = 3;
    threeD.ring = 0;
    EXPECT_FALSE(admits(threeD, "gfx942"));
    EXPECT_TRUE(admits(threeD, "gfx942", filterShape({3, 3, 3}, {1, 1, 1})));
}

TEST(TestConvDirectKernelMatch, TheChannelVectorMustDivideTheReduction)
{
    auto odd = baseShape();
    odd.c = 12;
    const ConfigFields f;
    EXPECT_FALSE(admits(f, "gfx942", odd));
    EXPECT_TRUE(admits(fmaFields(4), "gfx942", odd));
    odd.c = 3;
    EXPECT_FALSE(admits(fmaFields(4), "gfx942", odd));
    auto scalar = fmaFields(1);
    scalar.bc = 4;
    scalar.tm = 4;
    scalar.tn = 8;
    scalar.nt = 8;
    EXPECT_TRUE(admits(scalar, "gfx942", odd));
}

TEST(TestConvDirectKernelMatch, FmaRunsEverywhere)
{
    EXPECT_TRUE(admits(fmaFields(8), "gfx906", baseShape(), "HALF"));
    EXPECT_TRUE(admits(fmaFields(4), "gfx1030", baseShape(), "FLOAT"));
    EXPECT_TRUE(admits(fmaFields(8), "gfx1201", baseShape(), "BFLOAT16"));
}

TEST(TestConvDirectKernelMatch, ThreeBf16OnlyWithTf32PermissionAndOnlyForFloat)
{
    ConfigFields x3;
    x3.atom = "mfma16_bf16";
    x3.math = 1;
    EXPECT_FALSE(admits(x3, "gfx942", baseShape(), "FLOAT"));
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(
        "HIP_KERNEL_PROVIDER_CONV_TF32", "1");
    EXPECT_TRUE(admits(x3, "gfx942", baseShape(), "FLOAT"));
    EXPECT_FALSE(admits(x3, "gfx942", baseShape(), "BFLOAT16"));
    auto halfAtom = x3;
    halfAtom.atom = "mfma16_f16";
    EXPECT_FALSE(admits(halfAtom, "gfx942", baseShape(), "FLOAT"));
}

TEST(TestConvDirectKernelMatch, RefusesUnknownMetadata)
{
    auto badKind = ConfigFields{};
    badKind.kind = "winograd";
    EXPECT_FALSE(admits(badKind, "gfx942"));
    auto badAtom = ConfigFields{};
    badAtom.atom = "mfma64_f16";
    EXPECT_FALSE(admits(badAtom, "gfx942"));
    auto fmaAtomOnMma = ConfigFields{};
    fmaAtomOnMma.atom = "fma";
    EXPECT_FALSE(admits(fmaAtomOnMma, "gfx942"));
    auto badFilter = ConfigFields{};
    badFilter.r = 4;
    EXPECT_FALSE(admits(badFilter, "gfx942"));
    auto badChunk = ConfigFields{};
    badChunk.bc = 24; // not a multiple of the atom's reduction step
    EXPECT_FALSE(admits(badChunk, "gfx942"));
    auto badRing = ConfigFields{};
    badRing.ring = 2;
    EXPECT_FALSE(admits(badRing, "gfx942"));
    auto badMath = ConfigFields{};
    badMath.math = 2;
    EXPECT_FALSE(admits(badMath, "gfx942"));
    auto noArchs = ConfigFields{};
    noArchs.archs = "";
    EXPECT_FALSE(admits(noArchs, "gfx942"));
}

TEST(TestConvDirectShips, GroupsNameArchitecturesAndFamilies)
{
    EXPECT_TRUE(conv_direct::shipsTo("any", "gfx906"));
    EXPECT_TRUE(conv_direct::shipsTo("gfx90a,gfx942", "gfx942:sramecc+:xnack-"));
    EXPECT_FALSE(conv_direct::shipsTo("gfx90a,gfx942", "gfx950"));
    EXPECT_TRUE(conv_direct::shipsTo("gfx11", "gfx1151"));
    EXPECT_FALSE(conv_direct::shipsTo("gfx11", "gfx1201"));
    EXPECT_TRUE(conv_direct::shipsTo("gfx908,gfx12", "gfx1200"));
    EXPECT_FALSE(conv_direct::shipsTo("", "gfx942"));
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

std::optional<conv_direct::Plan>
    planOf(const ConvShape& shape, const ConfigFields& fields, const std::string& dtype, int cus)
{
    const auto problem = parse(specFor(shape, ConvDirection::FORWARD));
    const auto config = conv_direct::configOf(kernelOf(fields, dtype));
    if(!problem.has_value() || !config.has_value())
    {
        return std::nullopt;
    }
    return conv_direct::planFor(*problem, *config, cus);
}

TEST(TestConvDirectPlan, TileFitsTheHaloAndTheWorkspaceHoldsTheWeights)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.c = 40;
    shape.k = 24;
    shape.input = {13, 17};
    const auto plan = planOf(shape, ConfigFields{}, "HALF", 64);
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    EXPECT_GE(a.th, 1);
    EXPECT_LE(a.th * a.tw, 128);
    EXPECT_LE((a.th + 2) * (a.tw + 2), k2HaloMax(128, 3, 3));
    EXPECT_EQ(a.tilesH, (13 + a.th - 1) / a.th);
    EXPECT_EQ(a.tilesW, (17 + a.tw - 1) / a.tw);
    // C 40 -> 2 chunks of 32 (64 formatted); K 24 -> one 64-column block.
    EXPECT_EQ(plan->cpad, 64);
    EXPECT_EQ(a.kpad, 64);
    EXPECT_EQ(a.ca, 64 / 16);
    EXPECT_EQ(plan->weightElems, 9 * 64 * 64);
    EXPECT_EQ(plan->weightBytes, size_t{9 * 64 * 64 * 2});
    EXPECT_EQ(a.splits, 1);
    EXPECT_GE(plan->workspaceBytes, plan->weightBytes);
}

TEST(TestConvDirectPlan, SplitsOnlyAnUnderFilledGridAndSizesThePartials)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::FLOAT;
    shape.n = 1;
    shape.c = 512;
    shape.k = 64;
    shape.input = {7, 7};
    auto fields = ConfigFields{};
    fields.atom = "mfma16_f32";
    fields.bc = 16;
    fields.vec = 4;
    fields.bl = 1;
    fields.split = 1;
    const auto plan = planOf(shape, fields, "FLOAT", 304);
    ASSERT_TRUE(plan.has_value());
    EXPECT_GT(plan->args.splits, 1);
    EXPECT_LE(plan->args.splits, 16);
    EXPECT_EQ(plan->grid[2], static_cast<unsigned int>(plan->args.splits));
    EXPECT_GE(plan->args.cps, 2);
    EXPECT_EQ(plan->partialBytes,
              static_cast<size_t>(plan->args.splits) * 7 * 7 * 64 * sizeof(float));
    // A config without SPLIT never splits.
    fields.split = 0;
    const auto whole = planOf(shape, fields, "FLOAT", 304);
    ASSERT_TRUE(whole.has_value());
    EXPECT_EQ(whole->args.splits, 1);
    EXPECT_EQ(whole->partialBytes, size_t{0});
}

TEST(TestConvDirectPlan, ThreeBf16TriplesTheReductionAndStagesTheSplitInput)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    ConfigFields x3;
    x3.atom = "mfma16_bf16";
    x3.math = 1;
    const auto plan = planOf(shape, x3, "FLOAT", 64);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->args.c, 24);
    EXPECT_EQ(plan->cpad, 32);
    EXPECT_EQ(plan->weightBytes, size_t{9 * 32 * 64 * 2});
    EXPECT_EQ(plan->splitInputBytes, size_t{2 * 7 * 6 * 24 * 2});
}

TEST(TestConvDirectPlan, RingConfigsSplitDepthNoFinerThanTheFilter)
{
    auto shape = filterShape({3, 3, 3}, {1, 1, 1});
    shape.input = {12, 8, 8};
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.n = 1;
    ConfigFields ring;
    ring.kd = 3;
    const auto plan = planOf(shape, ring, "HALF", 304);
    ASSERT_TRUE(plan.has_value());
    EXPECT_GE(plan->args.ds, 3);
    ConfigFields stream = ring;
    stream.ring = 0;
    const auto planNr = planOf(shape, stream, "HALF", 304);
    ASSERT_TRUE(planNr.has_value());
    EXPECT_EQ(planNr->args.ds, 1);
}

} // namespace

#endif
