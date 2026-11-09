// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvIgemmNchwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvIgemmNchwProblem.cpp
 * @brief Host-only checks of the channels-first backward-data implicit-GEMM engine
 *        (CONV_IGEMM_NCHW_BWD): what the graph match accepts, one refusal per gated graph
 *        field, the argument block it derives, and what the kernel match admits per config
 *        field and device (one refusal per config field). No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_igemm_nchw::NchwConfig;
using conv_igemm_nchw::parseNchwProblem;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

/// The base problem: a ConvTranspose1d as 2-D with height 1, filter 1 x 8, stride 4, padding 2,
/// NCHW, fp16.
ConvShape baseShape()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 24;
    shape.k = 40;
    shape.input = {1, 37};
    shape.filter = {1, 8};
    shape.prePadding = {0, 2};
    shape.postPadding = {0, 2};
    shape.stride = {1, 4};
    shape.dilation = {1, 1};
    shape.dataType = DataType::HALF;
    return shape;
}

ConvGraphSpec baseSpec()
{
    return specFor(baseShape(), ConvDirection::BACKWARD_DATA);
}

std::optional<conv_igemm_nchw::NchwProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseNchwProblem(fixture.context());
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
    cases.push_back({"Base", baseShape()});
    auto oneD = baseShape();
    oneD.input = {37};
    oneD.filter = {8};
    oneD.prePadding = oneD.postPadding = {2};
    oneD.stride = {4};
    oneD.dilation = {1};
    cases.push_back({"OneD", oneD});
    auto rows = baseShape();
    rows.input = {3, 37};
    cases.push_back({"HeightRows", rows});
    auto threeD = baseShape();
    threeD.input = {2, 3, 37};
    threeD.filter = {1, 1, 8};
    threeD.prePadding = threeD.postPadding = {0, 0, 2};
    threeD.stride = {1, 1, 4};
    threeD.dilation = {1, 1, 1};
    cases.push_back({"ThreeDRows", threeD});
    auto fp32 = baseShape();
    fp32.dataType = DataType::FLOAT;
    cases.push_back({"Float", fp32});
    auto bf16 = baseShape();
    bf16.dataType = DataType::BFLOAT16;
    cases.push_back({"BFloat16", bf16});
    auto oddFilter = baseShape();
    oddFilter.filter = {1, 7};
    oddFilter.stride = {1, 3};
    cases.push_back({"Filter7Stride3", oddFilter});
    auto noPad = baseShape();
    noPad.prePadding = noPad.postPadding = {0, 0};
    cases.push_back({"Pad0", noPad});
    auto asymmetric = baseShape();
    asymmetric.prePadding = {0, 6};
    asymmetric.postPadding = {0, 1};
    cases.push_back({"AsymmetricPaddingPastTheStride", asymmetric});
    auto oddChannels = baseShape();
    oddChannels.c = 3;
    oddChannels.k = 5;
    cases.push_back({"OddChannels", oddChannels});
    return cases;
}

class TestConvIgemmNchwAccepts : public ::testing::TestWithParam<AcceptCase>
{
};

TEST_P(TestConvIgemmNchwAccepts, BackwardData)
{
    EXPECT_TRUE(parse(specFor(GetParam().shape, ConvDirection::BACKWARD_DATA)).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvIgemmNchwAccepts,
                         ::testing::ValuesIn(acceptCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvIgemmNchwForm, TheArgumentsAreTheWidthConvolutionOverRows)
{
    auto shape = baseShape();
    shape.input = {3, 37};
    shape.prePadding = {0, 6};
    shape.postPadding = {0, 1};
    const auto problem = parse(specFor(shape, ConvDirection::BACKWARD_DATA));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    EXPECT_EQ(a.n, 2);
    EXPECT_EQ(a.c, 24);
    EXPECT_EQ(a.k, 40);
    EXPECT_EQ(a.rows, 3);
    EXPECT_EQ(a.wx, 37);
    EXPECT_EQ(a.wy, (37 + 7 - 8) / 4 + 1);
    EXPECT_EQ(a.fs, 8);
    EXPECT_EQ(a.pad, 6);
    EXPECT_EQ(a.qBase, 1);
    EXPECT_EQ(a.dyImage, int64_t{40} * 3 * a.wy);
    EXPECT_EQ(a.dxImage, int64_t{24} * 3 * 37);
    EXPECT_EQ(problem->stride, 4);
    EXPECT_EQ(problem->elementBytes, 2);
    EXPECT_EQ(problem->dxUid, X_UID);
    EXPECT_EQ(problem->wUid, W_UID);
    EXPECT_EQ(problem->dyUid, Y_UID);
}

// ---------------------------------------------------------------------------
// One refusal per gated graph field
// ---------------------------------------------------------------------------

struct RefusalCase
{
    std::string name;
    std::function<void(ConvGraphSpec&)> mutate;
};

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
    spec.mode = rebuilt.mode;
}

std::vector<RefusalCase> refusalCases()
{
    return {
        // The base parser's own gates, one each (graph, node, tensor).
        {"OverrideShapeEnabled", [](ConvGraphSpec& s) { s.overrideShapeEnabled = true; }},
        {"VirtualOperand", [](ConvGraphSpec& s) { s.x.isVirtual = true; }},
        {"MixedDtype", [](ConvGraphSpec& s) { s.w.dataType = DataType::FLOAT; }},
        // Element type.
        {"Double",
         [](ConvGraphSpec& s) { s.x.dataType = s.w.dataType = s.y.dataType = DataType::DOUBLE; }},
        // One group, cross-correlation.
        {"Grouped",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.groups = 2;
             reshape(s, shape);
         }},
        {"ConvolutionMode",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.mode = ConvMode::CONVOLUTION;
             reshape(s, shape);
         }},
        // The height axis must be the identity: filter, stride, padding, dilation, extent.
        {"FilterHeight3",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.input = {5, 37};
             shape.filter = {3, 8};
             reshape(s, shape);
         }},
        {"StrideOnHeight",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.input = {4, 37};
             shape.stride = {2, 4};
             reshape(s, shape);
         }},
        {"PaddingOnHeight",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.prePadding = {1, 2};
             reshape(s, shape);
         }},
        {"PostPaddingOnHeight",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.postPadding = {1, 2};
             reshape(s, shape);
         }},
        {"DilationOnHeight",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.dilation = {2, 1};
             reshape(s, shape);
         }},
        // The width: unit dilation.
        {"DilationOnWidth",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.dilation = {1, 2};
             reshape(s, shape);
         }},
        // Every operand packed channels-first.
        {"DxChannelsLast", [](ConvGraphSpec& s) { s.x.strides = packedStrides(s.x.dims, true); }},
        {"DyChannelsLast", [](ConvGraphSpec& s) { s.y.strides = packedStrides(s.y.dims, true); }},
        {"FilterChannelsLast",
         [](ConvGraphSpec& s) { s.w.strides = packedStrides(s.w.dims, true); }},
        {"DxRowPadded",
         [](ConvGraphSpec& s) {
             s.x.strides = {24 * 40, 40, 40, 1}; // rows 40 apart, 37 used
         }},
        {"FilterTapsPadded",
         [](ConvGraphSpec& s) {
             s.w.strides = {24 * 9, 9, 9, 1}; // taps 9 apart, 8 used
         }},
        // Sizes: one image of dy and the filter within 32-bit byte offsets, images x rows within
        // the grid.
        {"DyImageBeyondOffsets",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.n = 1;
             shape.k = 4096;
             shape.input = {1, 1 << 20}; // dy 4096 x 262144 fp16: 2^31 bytes
             reshape(s, shape);
         }},
        {"FilterBeyondOffsets",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.c = 32768;
             shape.k = 4096;
             reshape(s, shape); // 4096 x 32768 x 8 fp16: 2^31 bytes
         }},
        {"ImagesTimesRowsBeyondTheGrid",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.n = 2;
             shape.input = {32768, 37};
             reshape(s, shape);
         }},
    };
}

class TestConvIgemmNchwRefuses : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestConvIgemmNchwRefuses, OneField)
{
    auto spec = baseSpec();
    ASSERT_TRUE(parse(spec).has_value());
    GetParam().mutate(spec);
    EXPECT_FALSE(parse(spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvIgemmNchwRefuses,
                         ::testing::ValuesIn(refusalCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvIgemmNchwDirection, RefusesForwardAndWeightGradients)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_WEIGHTS})
    {
        const GraphFixture fixture(buildGraph(specFor(baseShape(), direction)));
        EXPECT_FALSE(parseNchwProblem(fixture.context()).has_value());
    }
}

// ---------------------------------------------------------------------------
// Kernel match: one refusal per config field
// ---------------------------------------------------------------------------

constexpr PackSymbols PACK{"",
                           "hipkernel.conv_igemm_nchw_bwd.graph_match",
                           "",
                           "hipkernel.conv_igemm_nchw.kernel_match",
                           "",
                           "hipkernel.conv_igemm_nchw.dispatch",
                           "",
                           "",
                           ""};

/// A gfx1100 WMMA config that serves the base problem.
struct ConfigFields
{
    std::string direction = "BWD";
    std::string atom = "wmma11_f16";
    int64_t atomId = 7;
    int64_t bmq = 64, bn = 32, bk = 32, wm = 2, wn = 1, wp = 4;
    int64_t tq = 0, tc = 0, vec = 8, stride = 4, taps = 2, stages = 1;
};

ConfigFields fmaFields()
{
    ConfigFields f;
    f.atom = "fma";
    f.atomId = 0;
    f.bmq = 128;
    f.bn = 16;
    f.bk = 8;
    f.wm = f.wn = f.wp = 0;
    f.tq = 8;
    f.tc = 1;
    f.vec = 4;
    return f;
}

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const ConfigFields& f,
                                                       const std::string& dtype)
{
    auto kernel = makeKernel(0, dtype, "ConvIgemmNchw");
    kernel.metadata = {{"direction", f.direction},
                       {"dtype", dtype},
                       {"atom", f.atom},
                       {"atom_id", f.atomId},
                       {"bmq", f.bmq},
                       {"bn", f.bn},
                       {"bk", f.bk},
                       {"wm", f.wm},
                       {"wn", f.wn},
                       {"wp", f.wp},
                       {"tq", f.tq},
                       {"tc", f.tc},
                       {"vec", f.vec},
                       {"stride", f.stride},
                       {"taps", f.taps},
                       {"stages", f.stages}};
    return kernel;
}

bool admits(const ConfigFields& fields,
            const std::string& arch = "gfx1100",
            ConvShape shape = baseShape(),
            const std::string& dtype = "HALF")
{
    shape.dataType = dtype == "FLOAT"  ? DataType::FLOAT
                     : dtype == "HALF" ? DataType::HALF
                                       : DataType::BFLOAT16;
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = 48;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_DATA)),
                               properties);
    const auto bound = matchesGraph(PACK, fixture.context());
    return bound.has_value()
           && matchesKernel(PACK, fixture.context(), kernelOf(fields, dtype), *bound);
}

TEST(TestConvIgemmNchwKernelMatch, TheBaseConfigsServeTheBaseProblem)
{
    EXPECT_TRUE(admits(ConfigFields{}));
    EXPECT_TRUE(admits(fmaFields(), "gfx1100", baseShape(), "FLOAT"));
    EXPECT_TRUE(admits(fmaFields(), "gfx906", baseShape(), "HALF"));
}

struct ConfigRefusal
{
    std::string name;
    std::function<void(ConfigFields&)> mutate;
    bool fma = false;
};

std::vector<ConfigRefusal> configRefusals()
{
    return {
        {"DirectionForward", [](ConfigFields& f) { f.direction = "FWD"; }},
        {"UnknownAtom", [](ConfigFields& f) { f.atom = "wmma13_f16"; }},
        {"AtomIdNamesAnotherAtom", [](ConfigFields& f) { f.atomId = 8; }},
        {"AtomTakesAnotherElement",
         [](ConfigFields& f) {
             f.atom = "wmma11_bf16";
             f.atomId = 8;
         }},
        {"AtomNotOnTheDevice",
         [](ConfigFields& f) {
             f.atom = "wmma12_f16";
             f.atomId = 9;
         }},
        {"BmqNotWaveMultiple", [](ConfigFields& f) { f.bmq = 48; }},
        {"BnNotFragmentMultiple", [](ConfigFields& f) { f.bn = 24; }},
        {"BkNotAtomStep", [](ConfigFields& f) { f.bk = 24; }},
        {"WmZero", [](ConfigFields& f) { f.wm = 0; }},
        {"WnZero", [](ConfigFields& f) { f.wn = 0; }},
        {"WpNotDividingTheStride", [](ConfigFields& f) { f.wp = 3; }},
        {"TqSetOnMatrixCore", [](ConfigFields& f) { f.tq = 8; }},
        {"TcSetOnMatrixCore", [](ConfigFields& f) { f.tc = 1; }},
        {"VecNotPowerOfTwo", [](ConfigFields& f) { f.vec = 6; }},
        {"VecBeyond16Bytes", [](ConfigFields& f) { f.vec = 8; }, true},
        {"StrideNotTheProblems", [](ConfigFields& f) { f.stride = 2; }},
        {"TapsBelowThePhases", [](ConfigFields& f) { f.taps = 1; }},
        {"TapsAboveThePhases", [](ConfigFields& f) { f.taps = 3; }},
        {"LdsBeyond64KiB", [](ConfigFields& f) { f.bn = 128; }},
        {"ThreadsBeyond1024",
         [](ConfigFields& f) {
             f.bmq = 128;
             f.wm = 8;
             f.wn = 2;
         }},
        {"StagesTwoOnMatrixCore", [](ConfigFields& f) { f.stages = 2; }},
        {"StagesZero", [](ConfigFields& f) { f.stages = 0; }},
        {"FmaStagesThree", [](ConfigFields& f) { f.stages = 3; }, true},
        {"FmaWithWaves", [](ConfigFields& f) { f.wm = 1; }, true},
        {"FmaWithPhaseGroups", [](ConfigFields& f) { f.wp = 1; }, true},
        {"FmaTqNotDividingBmq", [](ConfigFields& f) { f.tq = 3; }, true},
        {"FmaTcZero", [](ConfigFields& f) { f.tc = 0; }, true},
        {"FmaBkZero", [](ConfigFields& f) { f.bk = 0; }, true},
        {"FmaVecNotDividingBmq",
         [](ConfigFields& f) {
             f.bmq = 66;
             f.tq = 2;
         },
         true},
    };
}

class TestConvIgemmNchwConfigRefuses : public ::testing::TestWithParam<ConfigRefusal>
{
};

TEST_P(TestConvIgemmNchwConfigRefuses, OneField)
{
    const auto& refusal = GetParam();
    auto fields = refusal.fma ? fmaFields() : ConfigFields{};
    const std::string dtype = refusal.fma ? "FLOAT" : "HALF";
    ASSERT_TRUE(admits(fields, "gfx1100", baseShape(), dtype));
    refusal.mutate(fields);
    EXPECT_FALSE(admits(fields, "gfx1100", baseShape(), dtype));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvIgemmNchwConfigRefuses,
                         ::testing::ValuesIn(configRefusals()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvIgemmNchwKernelMatch, TheDtypeMustBeTheGraphs)
{
    const GraphFixture fixture(buildGraph(baseSpec()), [] {
        auto p = testDeviceProperties();
        p.gcnArchName = "gfx1100";
        return p;
    }());
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(matchesKernel(PACK, fixture.context(), kernelOf(ConfigFields{}, "HALF"), *bound));
    EXPECT_FALSE(
        matchesKernel(PACK, fixture.context(), kernelOf(ConfigFields{}, "BFLOAT16"), *bound));
}

TEST(TestConvIgemmNchwKernelMatch, MissingMetadataRefuses)
{
    const GraphFixture fixture(buildGraph(baseSpec()), [] {
        auto p = testDeviceProperties();
        p.gcnArchName = "gfx1100";
        return p;
    }());
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    auto kernel = kernelOf(ConfigFields{}, "HALF");
    kernel.metadata.erase("wp");
    EXPECT_FALSE(matchesKernel(PACK, fixture.context(), kernel, *bound));
}

TEST(TestConvIgemmNchwKernelMatch, TheGridLimitsTheChannelTiles)
{
    // 16 x 65536 + 1 dx channels: 65537 channel tiles of 16.
    auto shape = baseShape();
    shape.n = 1;
    shape.c = 16 * 65536 + 16;
    shape.k = 2;
    shape.input = {1, 8};
    auto f = ConfigFields{};
    f.bmq = 64;
    f.bn = 16;
    f.wm = 4;
    f.wn = 1;
    f.wp = 1;
    EXPECT_FALSE(admits(f, "gfx1100", shape));
    shape.c = 16 * 65535;
    EXPECT_TRUE(admits(f, "gfx1100", shape));
}

TEST(TestConvIgemmNchwKernelMatch, TheAtomsAreK1sOnTheirArchitectures)
{
    auto mfma = ConfigFields{};
    mfma.atom = "mfma16_f16";
    mfma.atomId = 1;
    mfma.bmq = 64;
    mfma.bn = 16;
    mfma.wm = 4;
    mfma.wn = 1;
    mfma.wp = 1;
    EXPECT_TRUE(admits(mfma, "gfx942:sramecc+:xnack-"));
    EXPECT_TRUE(admits(mfma, "gfx90a"));
    EXPECT_FALSE(admits(mfma, "gfx1100"));
    auto wmma12 = ConfigFields{};
    wmma12.atom = "wmma12_bf16";
    wmma12.atomId = 10;
    EXPECT_TRUE(admits(wmma12, "gfx1201", baseShape(), "BFLOAT16"));
    EXPECT_FALSE(admits(wmma12, "gfx1100", baseShape(), "BFLOAT16"));
    auto mfma32 = mfma;
    mfma32.atom = "mfma32_f32";
    mfma32.atomId = 6;
    mfma32.bk = 16;
    mfma32.vec = 4;
    mfma32.bmq = 64;
    mfma32.bn = 32;
    mfma32.wm = 2;
    EXPECT_TRUE(admits(mfma32, "gfx942", baseShape(), "FLOAT"));
}

TEST(TestConvIgemmNchwKernelMatch, StrideAndTapsFollowTheFilter)
{
    // S 7 at stride 3: three taps in the fullest phase.
    auto shape = baseShape();
    shape.filter = {1, 7};
    shape.stride = {1, 3};
    auto f = ConfigFields{};
    f.stride = 3;
    f.taps = 3;
    f.wp = 3;
    EXPECT_TRUE(admits(f, "gfx1100", shape));
    f.taps = 2;
    EXPECT_FALSE(admits(f, "gfx1100", shape));
    // Strides past the FMA kernel's unrolled range are refused by every config.
    auto wide = baseShape();
    wide.filter = {1, 18};
    wide.stride = {1, 9};
    auto g = fmaFields();
    g.stride = 9;
    g.taps = 2;
    EXPECT_FALSE(admits(g, "gfx1100", wide, "FLOAT"));
}

TEST(TestConvIgemmNchwLds, MirrorsTheKernelsArithmetic)
{
    NchwConfig mma;
    mma.atom = "wmma11_f16";
    mma.bmq = 64;
    mma.bn = 32;
    mma.bk = 32;
    mma.vec = 8;
    mma.stride = 4;
    mma.taps = 2;
    mma.stages = 1;
    // A (72 rows of 40, 9 skews of 8) + B (8 taps x 32 channels x 40), or the 32 x 264 epilogue.
    EXPECT_EQ(conv_igemm_nchw::ldsBytesOf(mma, 2), (72 * 40 + 9 * 8 + 8 * 32 * 40) * 2);
    NchwConfig fma = mma;
    fma.atom = "fma";
    fma.bmq = 128;
    fma.bn = 16;
    fma.bk = 8;
    fma.vec = 4;
    // 8 x ((128 + 4 + 4) + 16 channels x (8 taps + 4) + 4) floats.
    EXPECT_EQ(conv_igemm_nchw::ldsBytesOf(fma, 4), 8 * (136 + 16 * 12 + 4) * 4);
    // Two stages: two such tiles.
    fma.stages = 2;
    EXPECT_EQ(conv_igemm_nchw::ldsBytesOf(fma, 4), 2 * 8 * (136 + 16 * 12 + 4) * 4);
}

TEST(TestConvIgemmNchwAtoms, IdsFollowTheTable)
{
    EXPECT_EQ(conv_igemm_nchw::atomIdOf("fma"), 0);
    EXPECT_EQ(conv_igemm_nchw::atomIdOf("wmma11_f16"), 7);
    EXPECT_EQ(conv_igemm_nchw::atomIdOf("wmma12_bf16"), 10);
    EXPECT_EQ(conv_igemm_nchw::atomIdOf("mfma64"), -1);
}

} // namespace

#endif
