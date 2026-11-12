// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvWrwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvWrwProblem.cpp
 * @brief Host-only checks of the matrix-core backward-weights engine (CONV_WRW): what the graph
 *        match accepts, one refusal per gated graph field, what the kernel match admits per
 *        config field and device (one refusal per config field), the argument blocks of both
 *        kernels (strides, steps, tracks, grids, splits, workspace, reduce strides and flip) and
 *        the LDS arithmetic. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using conv_wrw::parseWrwProblem;
using conv_wrw::WrwConfig;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

constexpr int64_t TWO_POW_31 = int64_t{1} << 31;

/// The base problem: 2-D 3x3, padding 1, stride 1, NCHW, fp16, 24 -> 40 channels.
ConvShape baseShape()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 24;
    shape.k = 40;
    shape.input = {10, 12};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    shape.stride = {1, 1};
    shape.dilation = {1, 1};
    shape.dataType = DataType::HALF;
    return shape;
}

ConvGraphSpec specOf(const ConvShape& shape)
{
    return specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
}

std::optional<conv_wrw::WrwProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseWrwProblem(fixture.context());
}

// Defined with the config helpers below.
ConvShape channelsLastShape();
ConvGraphSpec rowPaddedSpec(const ConvShape& shape = baseShape());

/// Strides of a K C T R S filter of @p dims stored K T R S C.
std::vector<int64_t> filterChannelsLast(const std::vector<int64_t>& dims)
{
    return packedStrides(dims, true);
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
    auto nhwc = baseShape();
    nhwc.channelsLast = true;
    cases.push_back({"ChannelsLast", nhwc});
    auto oneD = baseShape();
    oneD.input = {37};
    oneD.filter = {7};
    oneD.prePadding = oneD.postPadding = {3};
    oneD.stride = {1};
    oneD.dilation = {1};
    cases.push_back({"OneD", oneD});
    auto threeD = baseShape();
    threeD.input = {4, 6, 7};
    threeD.filter = {3, 3, 3};
    threeD.prePadding = threeD.postPadding = {1, 1, 1};
    threeD.stride = {1, 1, 1};
    threeD.dilation = {1, 1, 1};
    cases.push_back({"ThreeD", threeD});
    auto fp32 = baseShape();
    fp32.dataType = DataType::FLOAT;
    cases.push_back({"Float", fp32});
    auto bf16 = baseShape();
    bf16.dataType = DataType::BFLOAT16;
    cases.push_back({"BFloat16", bf16});
    auto grouped = baseShape();
    grouped.groups = 2;
    cases.push_back({"Grouped", grouped});
    auto convolution = baseShape();
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"TrueConvolution", convolution});
    auto strided = baseShape();
    strided.stride = {2, 2};
    cases.push_back({"Stride2", strided});
    auto dilated = baseShape();
    dilated.dilation = {2, 2};
    dilated.prePadding = dilated.postPadding = {2, 2};
    cases.push_back({"Dilation2", dilated});
    auto padded = baseShape();
    padded.prePadding = {4, 3};
    padded.postPadding = {0, 5};
    cases.push_back({"PaddingPastTheFilter", padded});
    auto odd = baseShape();
    odd.c = 5;
    odd.k = 7;
    cases.push_back({"OddChannels", odd});
    return cases;
}

class TestConvWrwAccepts : public ::testing::TestWithParam<AcceptCase>
{
};

TEST_P(TestConvWrwAccepts, BackwardWeights)
{
    EXPECT_TRUE(parse(specOf(GetParam().shape)).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvWrwAccepts,
                         ::testing::ValuesIn(acceptCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvWrwForm, LayoutsAreReadFromTheStrides)
{
    const auto cf = parse(specOf(baseShape()));
    ASSERT_TRUE(cf.has_value());
    EXPECT_TRUE(cf->channelsFirst);
    EXPECT_FALSE(cf->channelsLast);
    EXPECT_FALSE(cf->filterChannelsLast);
    EXPECT_EQ(cf->taps, 9);
    EXPECT_EQ(cf->elementBytes, 2);
    EXPECT_EQ(cf->xUid, X_UID);
    EXPECT_EQ(cf->dwUid, W_UID);
    EXPECT_EQ(cf->dyUid, Y_UID);

    auto shape = baseShape();
    shape.channelsLast = true;
    const auto cl = parse(specOf(shape));
    ASSERT_TRUE(cl.has_value());
    EXPECT_FALSE(cl->channelsFirst);
    EXPECT_TRUE(cl->channelsLast);
    EXPECT_TRUE(cl->filterChannelsLast);

    // dw K T R S C under channels-first activations, and K C T R S under channels-last.
    auto spec = specOf(baseShape());
    spec.w.strides = filterChannelsLast(spec.w.dims);
    const auto mixed = parse(spec);
    ASSERT_TRUE(mixed.has_value());
    EXPECT_TRUE(mixed->channelsFirst);
    EXPECT_TRUE(mixed->filterChannelsLast);
    auto clSpec = specOf(shape);
    clSpec.w.strides = packedStrides(clSpec.w.dims, false);
    const auto clMixed = parse(clSpec);
    ASSERT_TRUE(clMixed.has_value());
    EXPECT_TRUE(clMixed->channelsLast);
    EXPECT_FALSE(clMixed->filterChannelsLast);

    // One channel each way: both layouts hold.
    auto single = baseShape();
    single.c = 1;
    single.k = 1;
    const auto both = parse(specOf(single));
    ASSERT_TRUE(both.has_value());
    EXPECT_TRUE(both->channelsFirst);
    EXPECT_TRUE(both->channelsLast);
}

TEST(TestConvWrwForm, AnyStridesParseForTheGenericLoop)
{
    // x channels-last under channels-first dy; x rows padded; dy channels padded: no packed
    // layout, each image's span from its strides.
    auto mixed = specOf(baseShape());
    mixed.x.strides = packedStrides(mixed.x.dims, true);
    auto rows = rowPaddedSpec();
    auto channels = specOf(baseShape());
    channels.y.strides = {41 * 120, 120, 12, 1}; // 40 channels in an image of 41
    for(const auto& spec : {mixed, rows, channels})
    {
        const auto problem = parse(spec);
        ASSERT_TRUE(problem.has_value());
        EXPECT_FALSE(problem->channelsFirst);
        EXPECT_FALSE(problem->channelsLast);
        EXPECT_FALSE(problem->vectorChannels);
    }
    // Spans: packed, the image; padded rows, 23 x 160 + 9 x 16 + 11 + 1.
    const auto packed = parse(specOf(baseShape()));
    ASSERT_TRUE(packed.has_value());
    EXPECT_EQ(packed->xSpan, 24 * 10 * 12);
    EXPECT_EQ(packed->ySpan, 40 * 10 * 12);
    const auto padded = parse(rows);
    ASSERT_TRUE(padded.has_value());
    EXPECT_EQ(padded->xSpan, 23 * 160 + 9 * 16 + 11 + 1);
    EXPECT_EQ(padded->ySpan, 40 * 10 * 12);
    EXPECT_FALSE(padded->largeImage);
    // Spans past 2^31 bytes (rows 2^28 bytes apart): large images, for block-based buffers.
    auto xFar = specOf(baseShape());
    xFar.x.strides = {24 * 10 * 16, 10 * 16, int64_t{1} << 27, 1};
    const auto xFarProblem = parse(xFar);
    ASSERT_TRUE(xFarProblem.has_value());
    EXPECT_TRUE(xFarProblem->largeImage);
    auto yFar = specOf(baseShape());
    yFar.y.strides = {40 * 10 * 12, 1, int64_t{1} << 27, 40};
    const auto yFarProblem = parse(yFar);
    ASSERT_TRUE(yFarProblem.has_value());
    EXPECT_TRUE(yFarProblem->largeImage);
}

TEST(TestConvWrwForm, ChannelVectorsNeedUnitChannelStridesAndWholeVectors)
{
    const auto vectors = [](const ConvGraphSpec& spec) {
        const auto problem = parse(spec);
        return problem.has_value() && problem->vectorChannels;
    };
    // Packed channels-last, 24 -> 40 channels (fp16 vectors of 8): yes; channels-first: no.
    EXPECT_TRUE(vectors(specOf(channelsLastShape())));
    EXPECT_FALSE(vectors(specOf(baseShape())));
    // Channels per group not whole vectors: input 12, output 12, or groups of 12 of 24.
    auto c12 = channelsLastShape();
    c12.c = 12;
    EXPECT_FALSE(vectors(specOf(c12)));
    auto k12 = channelsLastShape();
    k12.k = 12;
    EXPECT_FALSE(vectors(specOf(k12)));
    auto grouped = channelsLastShape();
    grouped.groups = 2;
    EXPECT_FALSE(vectors(specOf(grouped)));
    // Per group: 12 of 24 input channels (32 output channels, 16 a group), or 12 of 24 output
    // channels (32 input channels); every stride whole vectors.
    auto groupC = channelsLastShape();
    groupC.groups = 2;
    groupC.k = 32;
    EXPECT_FALSE(vectors(specOf(groupC)));
    auto groupK = channelsLastShape();
    groupK.groups = 2;
    groupK.c = 32;
    groupK.k = 24;
    EXPECT_FALSE(vectors(specOf(groupK)));
    groupK.k = 32;
    EXPECT_TRUE(vectors(specOf(groupK)));
    // Padded channels-last rows: positions 32 elements apart (whole vectors) yes, 28 apart
    // (3.5 vectors) no; the padded image's batch stride likewise.
    auto spec = specOf(channelsLastShape());
    spec.x.strides = {10 * 12 * 32 + 4, 1, 12 * 32, 32};
    EXPECT_FALSE(vectors(spec));
    spec.x.strides = {10 * 12 * 32, 1, 12 * 32, 32};
    EXPECT_TRUE(vectors(spec));
    spec.x.strides = {10 * 12 * 28, 1, 12 * 28, 28};
    EXPECT_FALSE(vectors(spec));
    // dy's strides too.
    spec = specOf(channelsLastShape());
    spec.y.strides = {10 * 12 * 44, 1, 12 * 44, 44};
    EXPECT_FALSE(vectors(spec));
    spec.y.strides = {10 * 12 * 48, 1, 12 * 48, 48};
    EXPECT_TRUE(vectors(spec));
    // A unit channel stride on x only, or on dy only.
    spec = specOf(baseShape());
    spec.x.strides = packedStrides(spec.x.dims, true);
    EXPECT_FALSE(vectors(spec));
    spec = specOf(baseShape());
    spec.y.strides = packedStrides(spec.y.dims, true);
    EXPECT_FALSE(vectors(spec));
    // Channels 2 apart, every other stride whole vectors: x, then dy.
    spec = specOf(channelsLastShape());
    spec.x.strides = {10 * 12 * 48, 2, 12 * 48, 48};
    EXPECT_FALSE(vectors(spec));
    spec = specOf(channelsLastShape());
    spec.y.strides = {10 * 12 * 80, 2, 12 * 80, 80};
    EXPECT_FALSE(vectors(spec));
    // An axis of extent 1 carries any stride; fp32 vectors are 4 channels.
    auto oneImage = channelsLastShape();
    oneImage.n = 1;
    spec = specOf(oneImage);
    spec.x.strides[0] = 3;
    EXPECT_TRUE(vectors(spec));
    auto fp32 = channelsLastShape();
    fp32.dataType = DataType::FLOAT;
    fp32.c = 12;
    fp32.k = 20;
    EXPECT_TRUE(vectors(specOf(fp32)));
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
    const auto rebuilt = specOf(shape);
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
        // x and dy in no common packed layout (the generic loop's): one image's span through its
        // strides without overflow.
        {"StridedSpanOverflows",
         [](ConvGraphSpec& s) {
             s.x.strides = {24 * 10 * 16, int64_t{1} << 61, 16, 1}; // 23 x 2^61
         }},
        {"StridedSpanSumOverflows",
         [](ConvGraphSpec& s) {
             // 23 x 2^58 + 9 x 2^59: each axis within 64 bits, their sum past it.
             s.x.strides = {24 * 10 * 16, int64_t{1} << 58, int64_t{1} << 59, 1};
         }},
        // dw: packed K C T R S or K T R S C.
        {"FilterTapsPadded",
         [](ConvGraphSpec& s) {
             s.w.strides = {24 * 3 * 4, 3 * 4, 4, 1}; // rows of 4, 3 used
         }},
        {"FilterChannelsOutermost",
         [](ConvGraphSpec& s) {
             s.w.strides = {9, 40 * 9, 3, 1}; // C K R S
         }},
        // The batch within 32 bits (images tiny).
        {"BatchBeyondInt",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.n = TWO_POW_31;
             shape.c = 1;
             shape.k = 1;
             shape.input = {1, 8};
             shape.filter = {1, 1};
             shape.prePadding = shape.postPadding = {0, 0};
             reshape(s, shape);
         }},
        // Per axis, the kernel's 32-bit positions: driven by the stride, the dilation, the
        // padding.
        {"StrideBeyondPositions",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.stride = {1, int64_t{1} << 30};
             reshape(s, shape);
         }},
        {"DilationBeyondPositions",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.dilation = {1, int64_t{1} << 30};
             shape.prePadding = {1, int64_t{1} << 30};
             shape.postPadding = {1, int64_t{1} << 30};
             reshape(s, shape);
         }},
        {"PaddingBeyondPositions",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.k = 8;
             shape.prePadding = {1, (int64_t{3} << 29)};
             shape.postPadding = {1, 0};
             shape.stride = {1, 1024};
             reshape(s, shape);
         }},
        // Taps (ft fr fs) within 32 bits.
        {"TapsBeyondInt",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.input = {1, 1, 1};
             shape.filter = {2049, 2049, 1025};
             shape.prePadding = shape.postPadding = {1024, 1024, 512};
             shape.stride = {1, 1, 1};
             shape.dilation = {1, 1, 1};
             reshape(s, shape);
         }},
    };
}

class TestConvWrwRefuses : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestConvWrwRefuses, OneField)
{
    auto spec = specOf(baseShape());
    ASSERT_TRUE(parse(spec).has_value());
    GetParam().mutate(spec);
    EXPECT_FALSE(parse(spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvWrwRefuses,
                         ::testing::ValuesIn(refusalCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvWrwLimits, TheGateCasesAreValidGraphsOneStepFromTheLimit)
{
    // Each size refusal sits just past its limit: the same graph one step inside it parses,
    // so the refusal is the gate's and not the base parser's.
    auto x = baseShape();
    x.n = 1;
    x.c = 1;
    x.k = 8;
    x.input = {1, (int64_t{1} << 30) - 1};
    x.filter = {1, 1};
    x.prePadding = x.postPadding = {0, 0};
    x.stride = {1, int64_t{1} << 20};
    EXPECT_TRUE(parse(specOf(x)).has_value());
    auto dy = baseShape();
    dy.n = 1;
    dy.c = 8;
    dy.k = (int64_t{1} << 27) - 1;
    dy.input = {1, 8};
    dy.filter = {1, 1};
    dy.prePadding = dy.postPadding = {0, 0};
    EXPECT_TRUE(parse(specOf(dy)).has_value());
    auto batch = baseShape();
    batch.n = TWO_POW_31 - 1;
    batch.c = 1;
    batch.k = 1;
    batch.input = {1, 8};
    batch.filter = {1, 1};
    batch.prePadding = batch.postPadding = {0, 0};
    EXPECT_TRUE(parse(specOf(batch)).has_value());
    auto taps = baseShape();
    taps.input = {1, 1, 1};
    taps.filter = {2049, 2049, 511};
    taps.prePadding = taps.postPadding = {1024, 1024, 255};
    taps.stride = {1, 1, 1};
    taps.dilation = {1, 1, 1};
    EXPECT_TRUE(parse(specOf(taps)).has_value());
    auto pad = baseShape();
    pad.k = 8;
    pad.prePadding = {1, int64_t{1} << 29};
    pad.postPadding = {1, 0};
    pad.stride = {1, 1024};
    EXPECT_TRUE(parse(specOf(pad)).has_value());
}

TEST(TestConvWrwDirection, RefusesForwardAndBackwardData)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const GraphFixture fixture(buildGraph(specFor(baseShape(), direction)));
        EXPECT_FALSE(parseWrwProblem(fixture.context()).has_value());
    }
}

// ---------------------------------------------------------------------------
// Kernel match: one refusal per config field
// ---------------------------------------------------------------------------

constexpr PackSymbols PACK{"",
                           "hipkernel.conv_wrw.graph_match",
                           "",
                           "hipkernel.conv_wrw.kernel_match",
                           "",
                           "hipkernel.conv_wrw.dispatch",
                           "",
                           "",
                           ""};

/// A gfx942 row-ring config that serves the base problem.
struct ConfigFields
{
    std::string direction = "WRW";
    std::string atom = "mfma32_f16";
    int64_t atomId = 4;
    int64_t loop = 1, layout = 0;
    int64_t bk = 64, bc = 64, bq = 32, fr = 3, s = 3, sw = 1, dw = 1, wk = 2, wc = 2, vec = 8;
    int64_t tk = 0, tc = 0, xcd = 0, align = 0, ns = 0, bpu = 2, wq = 0, dws = 1, pb = 0, math = 0;
    /// The generic loop's specialized classes (metadata of loop 8 only).
    std::string spec;
};

/// The base config as the output-rows loop.
ConfigFields rowsFields()
{
    ConfigFields f;
    f.loop = 0;
    return f;
}

/// The base config as gfx950's DMA ring.
ConfigFields dmaFields()
{
    ConfigFields f;
    f.atom = "mfma32x16_f16";
    f.atomId = 10;
    f.loop = 2;
    f.ns = 2;
    f.xcd = 1;
    return f;
}

/// gfx950's ring with channels-last tiles kept channels-last (transpose reads).
ConfigFields trFields()
{
    auto f = dmaFields();
    f.loop = 1;
    f.ns = 0;
    f.layout = 2;
    return f;
}

/// The FMA ring (exact fp32 without an fp32 matrix core).
ConfigFields fmaFields()
{
    ConfigFields f;
    f.atom = "fma";
    f.atomId = 16;
    f.loop = 3;
    f.bk = 32;
    f.wk = 0;
    f.wc = 0;
    f.tk = 4;
    f.tc = 4;
    f.vec = 4;
    return f;
}

/// gfx950's direct loop: 16 x 16 x 32 atom, rows kept channels-last, 64 x 64 tile on 2 x 2 waves
/// by 2 items (waves_q), 3 ring slots.
ConfigFields directFields()
{
    ConfigFields f;
    f.atom = "mfma16x32_f16";
    f.atomId = 8;
    f.loop = 5;
    f.layout = 2;
    f.ns = 3;
    f.wq = 2;
    f.xcd = 1;
    f.bpu = 1;
    f.dws = 0;
    return f;
}

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const ConfigFields& f,
                                                       const std::string& dtype)
{
    auto kernel = makeKernel(0, dtype, "ConvWrw");
    kernel.metadata = {{"direction", f.direction},
                       {"dtype", dtype},
                       {"atom", f.atom},
                       {"atom_id", f.atomId},
                       {"loop", f.loop},
                       {"layout", f.layout},
                       {"bk", f.bk},
                       {"bc", f.bc},
                       {"bq", f.bq},
                       {"fr", f.fr},
                       {"s", f.s},
                       {"sw", f.sw},
                       {"dw", f.dw},
                       {"wk", f.wk},
                       {"wc", f.wc},
                       {"tk", f.tk},
                       {"tc", f.tc},
                       {"vec", f.vec},
                       {"xcd", f.xcd},
                       {"align", f.align},
                       {"ns", f.ns},
                       {"bpu", f.bpu},
                       {"wq", f.wq},
                       {"dws", f.dws},
                       {"pb", f.pb},
                       {"math", f.math}};
    if(f.loop == conv_wrw::LOOP_GENERIC)
    {
        kernel.metadata["spec"] = f.spec;
    }
    return kernel;
}

WrwConfig configOf(const ConfigFields& f)
{
    WrwConfig c;
    c.atom = f.atom;
    c.atomId = f.atomId;
    c.loop = f.loop;
    c.layout = f.layout;
    c.bk = f.bk;
    c.bc = f.bc;
    c.bq = f.bq;
    c.fr = f.fr;
    c.s = f.s;
    c.sw = f.sw;
    c.dw = f.dw;
    c.wk = f.wk;
    c.wc = f.wc;
    c.vec = f.vec;
    c.xcd = f.xcd;
    c.align = f.align;
    c.ns = f.ns;
    c.tk = f.tk;
    c.tc = f.tc;
    c.bpu = f.bpu;
    c.wq = f.wq;
    c.dws = f.dws;
    c.pb = f.pb;
    c.spec = f.spec;
    c.math = f.math;
    return c;
}

std::string dtypeName(DataType dataType)
{
    return dataType == DataType::FLOAT ? "FLOAT" : dataType == DataType::HALF ? "HALF" : "BFLOAT16";
}

hipdnn_plugin_sdk::ingestor::DeviceProperties deviceOf(const std::string& arch, int units = 304)
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = units;
    properties.warpSize = arch.rfind("gfx1", 0) == 0 ? 32 : 64;
    return properties;
}

bool admits(const ConfigFields& fields,
            const std::string& arch = "gfx942",
            const ConvShape& shape = baseShape(),
            int units = 304)
{
    const GraphFixture fixture(buildGraph(specOf(shape)), deviceOf(arch, units));
    const auto bound = matchesGraph(PACK, fixture.context());
    return bound.has_value()
           && matchesKernel(
               PACK, fixture.context(), kernelOf(fields, dtypeName(shape.dataType)), *bound);
}

/// The launch of @p fields on @p shape, through the problem's own functions.
std::optional<conv_wrw::WrwLaunch> launchOf(const ConfigFields& fields,
                                            const ConvShape& shape = baseShape(),
                                            const std::string& arch = "gfx942",
                                            int units = 304)
{
    const auto problem = parse(specOf(shape));
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    return conv_wrw::wrwLaunchFor(*problem, configOf(fields), deviceOf(arch, units));
}

struct ConfigRefusal
{
    std::string name;
    std::function<void(ConfigFields&)> mutate;
    /// The config the mutation starts from, the device, and the problem.
    std::function<ConfigFields()> start = [] { return ConfigFields{}; };
    std::string arch = "gfx942";
    std::function<ConvShape()> shape = baseShape;
};

ConvShape floatShape()
{
    auto shape = baseShape();
    shape.dataType = DataType::FLOAT;
    return shape;
}

ConvShape alignedShape()
{
    auto shape = baseShape();
    shape.input = {10, 16};
    return shape;
}

ConvShape widthStrideShape(int64_t sw)
{
    auto shape = baseShape();
    shape.input = {10, 13};
    shape.stride = {1, sw};
    return shape;
}

ConvShape heightStride2Shape()
{
    auto shape = baseShape();
    shape.stride = {2, 1};
    return shape;
}

ConvShape widthStride2Shape()
{
    auto shape = alignedShape();
    shape.stride = {1, 2};
    return shape;
}

ConvShape channelsLastShape()
{
    auto shape = baseShape();
    shape.channelsLast = true;
    return shape;
}

ConfigFields alignedFields()
{
    ConfigFields f;
    f.align = 1;
    return f;
}

ConfigFields floatFields()
{
    ConfigFields f;
    f.atom = "mfma32_f32";
    f.atomId = 7;
    f.vec = 4;
    f.bq = 16;
    return f;
}

/// Three-bf16 (math 1): the bf16 32 x 32 ring config on an fp32 graph, partials only.
ConfigFields threeBf16Fields()
{
    ConfigFields f;
    f.atom = "mfma32_bf16";
    f.atomId = 5;
    f.dws = 0;
    f.math = 1;
    return f;
}

/// The output-rows loop at width stride @p sw, one filter row per block.
ConfigFields stridedRowsFields(int64_t sw)
{
    auto f = rowsFields();
    f.fr = 1;
    f.sw = sw;
    f.bq = 24;
    return f;
}

ConfigFields widthStride2Fields()
{
    ConfigFields f;
    f.sw = 2;
    return f;
}

ConfigFields channelsLastFields()
{
    ConfigFields f;
    f.layout = 1;
    return f;
}

/// A stem: 3 input channels, channels-first.
ConvShape stemShape()
{
    auto shape = baseShape();
    shape.c = 3;
    return shape;
}

/// Small-C rows (loop 6) on gfx942's 16 x 16 fp16 atom: 64 output channels on 4 waves, 4 input
/// channels by 3 taps on the N axis, 64 positions a step.
ConfigFields smallcFields()
{
    ConfigFields f;
    f.atom = "mfma16_f16";
    f.atomId = 0;
    f.loop = 6;
    f.bk = 64;
    f.wk = 4;
    f.bc = 4;
    f.wc = 1;
    f.bq = 64;
    return f;
}

/// Packed rows (loop 7) on gfx942's 32 x 32 fp16 atom: 64 x 64 on 4 waves, one filter row a
/// block, 32 positions a step (baseShape's 12-column rows: virtual rows of 14, 2 a step).
ConfigFields packedFields()
{
    ConfigFields f;
    f.loop = 7;
    f.fr = 1;
    return f;
}

/// Channels-last with 3 output channels (a VAE's output layer): not whole vectors of 8.
ConvShape unalignedKShape()
{
    auto shape = channelsLastShape();
    shape.k = 3;
    return shape;
}

/// The ring with dy units at any element (align 2), one 32-channel fragment of output channels.
ConfigFields unalignedKFields()
{
    auto f = channelsLastFields();
    f.align = 2;
    f.bk = 32;
    f.wk = 1;
    return f;
}

ConvShape floatChannelsLastShape()
{
    auto shape = floatShape();
    shape.channelsLast = true;
    return shape;
}

/// gfx1201's WMMA (8 positions per lane) on channels-last tiles transposed into LDS.
ConfigFields wmma12Fields()
{
    ConfigFields f;
    f.atom = "wmma12_f16";
    f.atomId = 14;
    f.layout = 1;
    f.wk = 4;
    f.wc = 4;
    return f;
}

/// The DMA ring with channels-last tiles (transpose reads).
ConfigFields dmaTrFields()
{
    auto f = dmaFields();
    f.layout = 2;
    return f;
}

ConfigFields fmaChannelsLastFields()
{
    auto f = fmaFields();
    f.layout = 1;
    return f;
}

/// The generic loop (8) on gfx942's 32 x 32 fp16 atom, channels-first elements: 64 x 64 on 4
/// waves, 3 taps a block, 32 positions a step (8 channel rows a pass); no specialized class.
ConfigFields genericFields()
{
    ConfigFields f;
    f.loop = 8;
    f.s = 0;
    f.sw = 0;
    f.dw = 0;
    return f;
}

/// The generic loop on channel vectors (channelsLastShape).
ConfigFields genericClFields()
{
    auto f = genericFields();
    f.layout = 1;
    return f;
}

/// The generic loop's channel vectors on gfx950's 32 x 16 fp16 atom.
ConfigFields genericGfx950Fields()
{
    auto f = genericClFields();
    f.atom = "mfma32x16_f16";
    f.atomId = 10;
    return f;
}

/// The generic loop's FMA form (fp32 where no matrix core takes it): 64 x 64 channels by 4 x 4
/// a thread, 16 positions a step, 5 taps a block.
ConfigFields genericFmaFields()
{
    ConfigFields f;
    f.atom = "fma";
    f.atomId = 16;
    f.loop = 8;
    f.s = f.sw = f.dw = 0;
    f.fr = 5;
    f.bq = 16;
    f.wk = f.wc = 0;
    f.tk = f.tc = 4;
    f.vec = 4;
    return f;
}

/// The generic loop's shared width window: 8 positions past the chunk (channel vectors).
ConfigFields genericWindowFields()
{
    auto f = genericClFields();
    f.dw = 8;
    return f;
}

/// channelsLastShape at width stride 2.
ConvShape channelsLastStride2Shape()
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.stride = {1, 2};
    return shape;
}

/// channelsLastShape at width dilation 5 (3x3: the taps span 10 positions past the first).
ConvShape channelsLastDilation5Shape()
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dilation = {1, 5};
    shape.prePadding = shape.postPadding = {1, 5};
    return shape;
}

/// x rows padded to 16 columns (12 used): strided, no packed layout.
ConvGraphSpec rowPaddedSpec(const ConvShape& shape)
{
    auto spec = specOf(shape);
    const int64_t h = shape.input[0];
    spec.x.strides = {shape.c * h * 16, h * 16, 16, 1};
    return spec;
}

std::vector<ConfigRefusal> configRefusals()
{
    std::vector<ConfigRefusal> cases{
        {"DirectionBackwardData", [](ConfigFields& f) { f.direction = "BWD"; }},
        {"UnknownAtom", [](ConfigFields& f) { f.atom = "mfma64_f16"; }},
        {"Tf32Atom",
         [](ConfigFields& f) {
             f.atom = "mfma32_xf32";
             f.vec = 4;
             f.bq = 16;
         },
         floatFields,
         "gfx942",
         floatShape},
        {"AtomIdNamesAnotherAtom", [](ConfigFields& f) { f.atomId = 0; }},
        {"AtomTakesAnotherElement",
         [](ConfigFields& f) {
             f.atom = "mfma32_bf16";
             f.atomId = 5;
         }},
        {"AtomNotOnTheDevice",
         [](ConfigFields& f) {
             f.atom = "wmma11_f16";
             f.atomId = 12;
         }},
        {"LoopUnknown", [](ConfigFields& f) { f.loop = 9; }},
        // The generic loop: runtime geometry, fr taps a block, a position per thread, whole
        // channel rows a pass, its layout the problem's channel-vector form, at most 64 loads,
        // and no specialized class serving the problem.
        {"GenericCompileTimeWidth", [](ConfigFields& f) { f.s = 3; }, genericFields},
        {"GenericCompileTimeStride", [](ConfigFields& f) { f.sw = 1; }, genericFields},
        {"GenericCompileTimeDilation", [](ConfigFields& f) { f.dw = 1; }, genericFields},
        {"GenericNoTaps", [](ConfigFields& f) { f.fr = 0; }, genericFields},
        // More positions than threads: no channel row a pass.
        {"GenericBqBeyondTheBlock", [](ConfigFields& f) { f.bq = 512; }, genericFields},
        // 256 threads at bq 96: 2 rows a pass would divide bk and bc.
        {"GenericBqNotDividingTheBlock",
         [](ConfigFields& f) { f.bq = 96; },
         [] {
             auto f = genericFields();
             f.fr = 1;
             return f;
         }},
        // 512 threads at bq 8: 64 channel rows a pass.
        {"GenericOutputChannelsNotWholeRows",
         [](ConfigFields& f) { f.bk = 32; },
         [] {
             auto f = genericFields();
             f.wk = 1;
             f.wc = 8;
             f.bk = 64;
             f.bc = 256;
             f.bq = 8;
             return f;
         }},
        {"GenericInputChannelsNotWholeRows",
         [](ConfigFields& f) { f.bc = 32; },
         [] {
             auto f = genericFields();
             f.wk = 8;
             f.wc = 1;
             f.bk = 256;
             f.bc = 64;
             f.bq = 8;
             return f;
         }},
        {"GenericElementsOnChannelVectors",
         [](ConfigFields& f) { f.layout = 0; },
         genericClFields,
         "gfx942",
         channelsLastShape},
        {"GenericChannelVectorsOnElements", [](ConfigFields& f) { f.layout = 1; }, genericFields},
        {"GenericTransposeLayout",
         [](ConfigFields& f) { f.layout = 2; },
         genericGfx950Fields,
         "gfx950",
         channelsLastShape},
        {"GenericAlign", [](ConfigFields& f) { f.align = 1; }, genericFields},
        {"GenericNs", [](ConfigFields& f) { f.ns = 2; }, genericFields},
        {"GenericWq", [](ConfigFields& f) { f.wq = 1; }, genericFields},
        {"GenericPb", [](ConfigFields& f) { f.pb = 1; }, genericFields},
        // gfx950 (160 KiB of LDS), channels-first, 8 channel rows a pass: 64 / 8 + 7 x 64 / 8
        // = 64 loads, then 72.
        {"GenericLoadsBeyond64",
         [](ConfigFields& f) { f.fr = 8; },
         [] {
             auto f = genericGfx950Fields();
             f.layout = 0;
             f.fr = 7;
             return f;
         },
         "gfx950"},
        // A class that does not serve the problem (5x5) admits; the 3x3 rows' class declines.
        {"GenericWhereASpecializedClassServes",
         [](ConfigFields& f) { f.spec += ";4R0.3.1.1.0"; },
         [] {
             auto f = genericFields();
             f.spec = "4R0.5.1.1.0";
             return f;
         }},
        {"GenericMalformedSpec",
         [](ConfigFields& f) { f.spec += ";x"; },
         [] {
             auto f = genericFields();
             f.spec = "4R0.5.1.1.0";
             return f;
         }},
        // The shared window: channel vectors, whole 16-byte units, unit width stride, the
        // block's width taps within it; never negative.
        {"GenericWindowNegative",
         [](ConfigFields& f) { f.dw = -8; },
         genericWindowFields,
         "gfx942",
         channelsLastShape},
        {"GenericWindowOnElements", [](ConfigFields& f) { f.dw = 8; }, genericFields},
        {"GenericWindowNotWholeUnits",
         [](ConfigFields& f) { f.dw = 4; },
         genericWindowFields,
         "gfx942",
         channelsLastShape},
        {"GenericWindowWidthStride",
         [](ConfigFields& f) { f.dw = 8; },
         genericClFields,
         "gfx942",
         channelsLastStride2Shape},
        {"GenericWindowNarrowerThanTheTaps",
         [](ConfigFields& f) { f.dw = 8; },
         [] {
             auto f = genericWindowFields();
             f.dw = 16;
             return f;
         },
         "gfx942",
         channelsLastDilation5Shape},
        // Block-based buffers only where the image is past the 32-bit buffer range; 0 or 1.
        {"PbOnASmallImage", [](ConfigFields& f) { f.pb = 1; }},
        {"PbTwo", [](ConfigFields& f) { f.pb = 2; }},
        // Packed rows: bq dividing the block (256 threads; 48 is whole atom steps and
        // vectors), and two virtual rows of wo + 2 = 14 columns within bq.
        {"PackedBqNotDividingTheBlock", [](ConfigFields& f) { f.bq = 48; }, packedFields},
        {"PackedFewerThanTwoRows", [](ConfigFields& f) { f.bq = 16; }, packedFields},
        {"PackedAlignOne", [](ConfigFields& f) { f.align = 1; }, packedFields},
        {"SmallCWcTwo", [](ConfigFields& f) { f.wc = 2; }, smallcFields, "gfx942", stemShape},
        {"SmallCBc32", [](ConfigFields& f) { f.bc = 32; }, smallcFields, "gfx942", stemShape},
        {"WqOffTheDirectLoop", [](ConfigFields& f) { f.wq = 1; }},
        {"DwsTwo", [](ConfigFields& f) { f.dws = 2; }},
        {"DwsNegative", [](ConfigFields& f) { f.dws = -1; }},
        {"LoopNegative", [](ConfigFields& f) { f.loop = -1; }},
        {"LayoutNotTheGraphs", [](ConfigFields& f) { f.layout = 1; }},
        {"LayoutUnknown", [](ConfigFields& f) { f.layout = 3; }},
        {"TkSetOnMatrixCores", [](ConfigFields& f) { f.tk = 4; }},
        {"TcSetOnMatrixCores", [](ConfigFields& f) { f.tc = 4; }},
        {"BpuThree", [](ConfigFields& f) { f.bpu = 3; }},
        {"BpuZero", [](ConfigFields& f) { f.bpu = 0; }},
        {"BpuSixteen",
         [](ConfigFields& f) { f.bpu = 16; },
         [] {
             ConfigFields f;
             f.bpu = 8;
             return f;
         }},
        {"BkNotWholeFragmentsPerWave", [](ConfigFields& f) { f.bk = 96; }},
        {"BcNotWholeFragmentsPerWave", [](ConfigFields& f) { f.bc = 96; }},
        {"BqNotAtomSteps",
         [](ConfigFields& f) { f.bq = 24; }, // 16-position steps, 8-element vectors
         dmaFields,
         "gfx950"},
        {"BqNotVectors",
         [](ConfigFields& f) { f.bq = 18; }, // 2-position steps, 4-element vectors
         floatFields,
         "gfx942",
         floatShape},
        {"VecNot16Bytes", [](ConfigFields& f) { f.vec = 4; }},
        {"BkZero", [](ConfigFields& f) { f.bk = 0; }},
        {"BcZero", [](ConfigFields& f) { f.bc = 0; }},
        {"BqZero", [](ConfigFields& f) { f.bq = 0; }},
        {"WkZero", [](ConfigFields& f) { f.wk = 0; }},
        {"WcZero", [](ConfigFields& f) { f.wc = 0; }},
        {"SNotTheFilterWidth", [](ConfigFields& f) { f.s = 5; }},
        {"SwNotTheWidthStride", [](ConfigFields& f) { f.sw = 2; }},
        {"DwNotTheWidthDilation", [](ConfigFields& f) { f.dw = 2; }},
        {"FrZeroOnRows", [](ConfigFields& f) { f.fr = 0; }, rowsFields},
        {"FrNotTheFilterHeightOnTheRing", [](ConfigFields& f) { f.fr = 1; }},
        {"RingOnHeightStride2",
         [](ConfigFields& f) { f.loop = 1; },
         rowsFields,
         "gfx942",
         heightStride2Shape},
        {"DmaRingWidthStride2",
         [](ConfigFields& f) {
             f.loop = 2;
             f.ns = 2;
             f.atom = "mfma32x16_f16";
             f.atomId = 10;
         },
         widthStride2Fields,
         "gfx950",
         widthStride2Shape},
        {"DmaRingStagesFive", [](ConfigFields& f) { f.ns = 5; }, dmaFields, "gfx950"},
        {"DmaRingStagesOne", [](ConfigFields& f) { f.ns = 1; }, dmaFields, "gfx950"},
        {"StagesOnTheRing", [](ConfigFields& f) { f.ns = 2; }},
        {"AlignTwo", [](ConfigFields& f) { f.align = 2; }, alignedFields, "gfx942", alignedShape},
        {"AlignThree", [](ConfigFields& f) { f.align = 3; }, alignedFields, "gfx942", alignedShape},
        {"AlignTwoOnRows",
         [](ConfigFields& f) { f.loop = 0; },
         unalignedKFields,
         "gfx942",
         unalignedKShape},
        {"AlignTwoTransposeReads",
         [](ConfigFields& f) {
             f.layout = 2;
             f.atom = "mfma32x16_f16";
             f.atomId = 10;
             f.bq = 64;
         },
         [] {
             auto f = unalignedKFields();
             f.atom = "mfma32x16_f16";
             f.atomId = 10;
             f.bq = 64;
             f.xcd = 1;
             return f;
         },
         "gfx950",
         unalignedKShape},
        {"AlignZeroOnUnalignedK",
         [](ConfigFields& f) { f.align = 0; },
         unalignedKFields,
         "gfx942",
         unalignedKShape},
        {"AlignOnRows", [](ConfigFields& f) { f.loop = 0; }, alignedFields, "gfx942", alignedShape},
        {"AlignWidthStride2",
         [](ConfigFields& f) { f.align = 1; },
         widthStride2Fields,
         "gfx942",
         widthStride2Shape},
        {"AlignWidthNotWholeVectors", [](ConfigFields& f) { f.align = 1; }}, // width 12
        {"XcdTwo", [](ConfigFields& f) { f.xcd = 2; }},
        {"XcdOnGfx90a",
         [](ConfigFields& f) { f.xcd = 1; },
         [] { return ConfigFields{}; },
         "gfx90a"},
        {"ThreadsBeyond1024",
         [](ConfigFields& f) {
             f.bk = 256;
             f.bc = 128;
             f.wk = 8;
             f.wc = 4;
             f.bq = 16;
         }},
        {"LdsBeyond64KiB",
         [](ConfigFields& f) {
             f.bk = 128;
             f.bc = 128;
             f.wk = 4;
             f.wc = 4;
             f.bq = 64;
         }},
    };
    // The transpose-read layout: gfx950, an atom with 8 positions per lane; DMA ring 0 or 2.
    cases.push_back({"TransposeReadsOffGfx950",
                     [](ConfigFields& f) { f.layout = 2; },
                     wmma12Fields,
                     "gfx1201",
                     channelsLastShape});
    cases.push_back({"TransposeReadsOnFourPositionsPerLane",
                     [](ConfigFields& f) { f.layout = 2; },
                     channelsLastFields,
                     "gfx950",
                     channelsLastShape});
    cases.push_back({"DmaRingOnTransposedChannelsLast",
                     [](ConfigFields& f) { f.layout = 1; },
                     dmaTrFields,
                     "gfx950",
                     channelsLastShape});
    // The direct loop (gfx950): waves_q 1, 2 or 4 over 8 waves, transpose-read rows, a 16 x 16 x
    // 32 atom, 32 columns per item, 2 to 5 ring slots, bpu 1, unit width stride and height
    // dilation, channel tiles with a conflict-free rotation.
    const auto direct = [&](std::string name, std::function<void(ConfigFields&)> mutate) {
        cases.push_back(
            {std::move(name), std::move(mutate), directFields, "gfx950", channelsLastShape});
    };
    direct("DirectWqThree", [](ConfigFields& f) { f.wq = 3; });
    direct("DirectWqZero", [](ConfigFields& f) { f.wq = 0; });
    direct("DirectWithTheDwStore", [](ConfigFields& f) { f.dws = 1; });
    direct("DirectWavesNotEight", [](ConfigFields& f) { f.wq = 1; });
    direct("DirectOnChannelsLastTransposedIntoLds", [](ConfigFields& f) { f.layout = 1; });
    direct("DirectOn32x16Atom", [](ConfigFields& f) {
        f.atom = "mfma32x16_f16";
        f.atomId = 10;
    });
    // Eight items side by side: 8 waves of one fragment each, which fits only a pointwise row.
    cases.push_back({"DirectWqEight",
                     [](ConfigFields& f) {
                         f.wq = 8;
                         f.wk = 1;
                         f.wc = 1;
                         f.bk = 32;
                         f.bc = 32;
                     },
                     [] {
                         auto f = directFields();
                         f.fr = 1;
                         f.s = 1;
                         return f;
                     },
                     "gfx950",
                     [] {
                         auto shape = channelsLastShape();
                         shape.filter = {1, 1};
                         shape.prePadding = {0, 0};
                         shape.postPadding = {0, 0};
                         return shape;
                     }});
    direct("DirectBq64", [](ConfigFields& f) { f.bq = 64; });
    direct("DirectSlotsOne", [](ConfigFields& f) { f.ns = 1; });
    // Six slots fit the LDS with one item (no waves_q staging).
    cases.push_back({"DirectSlotsSix",
                     [](ConfigFields& f) { f.ns = 6; },
                     [] {
                         auto f = directFields();
                         f.wq = 1;
                         f.wk = 4;
                         return f;
                     },
                     "gfx950",
                     channelsLastShape});
    direct("DirectBpuTwo", [](ConfigFields& f) { f.bpu = 2; });
    direct("DirectBcNotPowerOfTwo", [](ConfigFields& f) { f.bc = 96; });
    direct("DirectBkNotPowerOfTwo", [](ConfigFields& f) { f.bk = 96; });
    direct("DirectBc16NoRotation", [](ConfigFields& f) {
        f.bc = 16;
        f.wc = 1;
        f.wq = 4;
    });
    // The FMA ring: "fma" exactly on loop 3, its thread tile, no waves, no XCD order.
    cases.push_back({"MatrixAtomOnTheFmaRing",
                     [](ConfigFields& f) {
                         f.loop = 3;
                         f.wk = 0;
                         f.wc = 0;
                         f.tk = 4;
                         f.tc = 4;
                     },
                     floatFields,
                     "gfx942",
                     floatShape});
    cases.push_back({"FmaAtomOnTheRing",
                     [](ConfigFields& f) {
                         f.loop = 1;
                         f.wk = 2;
                         f.wc = 2;
                         f.tk = 0;
                         f.tc = 0;
                     },
                     fmaFields,
                     "gfx1100",
                     floatShape});
    cases.push_back(
        {"FmaWithWaves", [](ConfigFields& f) { f.wk = 1; }, fmaFields, "gfx1100", floatShape});
    cases.push_back({"FmaWithWavesOverChannels",
                     [](ConfigFields& f) { f.wc = 1; },
                     fmaFields,
                     "gfx1100",
                     floatShape});
    cases.push_back(
        {"FmaTkNegative", [](ConfigFields& f) { f.tk = -4; }, fmaFields, "gfx1100", floatShape});
    cases.push_back(
        {"FmaTkZero", [](ConfigFields& f) { f.tk = 0; }, fmaFields, "gfx1100", floatShape});
    cases.push_back({"FmaTcNotQuads",
                     [](ConfigFields& f) {
                         f.bc = 40;
                         f.tc = 5;
                     },
                     fmaFields,
                     "gfx1100",
                     floatShape});
    cases.push_back({"FmaTkNotQuads",
                     [](ConfigFields& f) {
                         f.bk = 40;
                         f.tk = 5;
                     },
                     fmaFields,
                     "gfx1100",
                     floatShape});
    cases.push_back(
        {"FmaTcZero", [](ConfigFields& f) { f.tc = 0; }, fmaFields, "gfx1100", floatShape});
    cases.push_back({"FmaTkNotDividingBk",
                     [](ConfigFields& f) { f.tk = 12; },
                     fmaFields,
                     "gfx1100",
                     floatShape});
    cases.push_back({"FmaTcNotDividingBc",
                     [](ConfigFields& f) { f.tc = 24; },
                     fmaFields,
                     "gfx1100",
                     floatShape});
    cases.push_back({"FmaBqNotVectorsChannelsLast",
                     [](ConfigFields& f) { f.bq = 30; },
                     fmaChannelsLastFields,
                     "gfx1100",
                     floatChannelsLastShape});
    cases.push_back(
        {"FmaXcd", [](ConfigFields& f) { f.xcd = 1; }, fmaFields, "gfx942", floatShape});
    cases.push_back({"FmaThreadsBeyond1024",
                     [](ConfigFields& f) {
                         f.bk = 256;
                         f.bc = 128;
                         f.bq = 16;
                     },
                     fmaFields,
                     "gfx950",
                     floatShape});
    // Aligned units are channels-first.
    cases.push_back({"AlignOnChannelsLast",
                     [](ConfigFields& f) { f.align = 1; },
                     channelsLastFields,
                     "gfx942",
                     [] {
                         auto shape = alignedShape();
                         shape.channelsLast = true;
                         return shape;
                     }});
    // The channels-last layout's own gates.
    cases.push_back({"ChannelsLastLayoutOnChannelsFirst",
                     [](ConfigFields& f) { f.layout = 0; },
                     channelsLastFields,
                     "gfx942",
                     channelsLastShape});
    return cases;
}

class TestConvWrwConfigRefuses : public ::testing::TestWithParam<ConfigRefusal>
{
};

TEST_P(TestConvWrwConfigRefuses, OneField)
{
    const auto& refusal = GetParam();
    auto fields = refusal.start();
    const auto shape = refusal.shape();
    ASSERT_TRUE(admits(fields, refusal.arch, shape));
    refusal.mutate(fields);
    EXPECT_FALSE(admits(fields, refusal.arch, shape));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvWrwConfigRefuses,
                         ::testing::ValuesIn(configRefusals()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvWrwKernelMatch, TheBaseConfigsServeTheBaseProblem)
{
    EXPECT_TRUE(admits(ConfigFields{}));
    EXPECT_TRUE(admits(rowsFields()));
    EXPECT_TRUE(admits(dmaFields(), "gfx950"));
    EXPECT_TRUE(admits(trFields(), "gfx950", channelsLastShape()));
    EXPECT_TRUE(admits(fmaFields(), "gfx1100", floatShape()));
    EXPECT_TRUE(admits(directFields(), "gfx950", channelsLastShape()));
}

TEST(TestConvWrwKernelMatch, BlockBuffersServeImagesPastTheBufferRangeOnly)
{
    // 64 channels of 4100 x 4200 fp16: one x image of 2.2 GB, 32 channel planes 1.1 GB.
    auto large = baseShape();
    large.c = 64;
    large.k = 32;
    large.input = {4100, 4200};
    large.prePadding = large.postPadding = {0, 0};
    large.stride = {2, 2};
    auto f = rowsFields();
    f.sw = 2;
    f.fr = 1; // one filter row a block: the 64-channel tile below fits the LDS
    f.bc = 32;
    f.wc = 1;
    EXPECT_FALSE(admits(f, "gfx942", large)); // the image is past the range
    f.pb = 1;
    EXPECT_TRUE(admits(f, "gfx942", large));
    f.bc = 64; // 64 channel planes: 2.2 GB
    f.wc = 2;
    EXPECT_FALSE(admits(f, "gfx942", large));
    // Each channel plane within the range, and channels-first only.
    auto cl = large;
    cl.channelsLast = true;
    f.bc = 32;
    f.wc = 1;
    f.layout = 1;
    EXPECT_FALSE(admits(f, "gfx942", cl));
    f.layout = 0;
    auto plane = large;
    plane.c = 2;
    plane.input = {33000, 33000}; // one 2.2 GB channel plane
    EXPECT_FALSE(admits(f, "gfx942", plane));
    // The K3Tiles loops only: packed rows (loop 7) stage their own units. 72 channels of
    // 1100000 x 14 (2.2 GB, wo 14), 32 x 32 tiles whose planes fit.
    auto narrow14 = baseShape();
    narrow14.c = 72;
    narrow14.input = {1100000, 14};
    auto p = packedFields();
    p.bk = p.bc = 32;
    p.wk = p.wc = 1;
    EXPECT_FALSE(admits(p, "gfx942", narrow14));
    p.pb = 1;
    EXPECT_FALSE(admits(p, "gfx942", narrow14));
    p.loop = 0; // the same tile on the output rows
    p.fr = 3;
    EXPECT_TRUE(admits(p, "gfx942", narrow14));
    // The dy window: 64 channels of 1100000 x 16 rows (2.25 GB) past it, 32 within.
    auto narrow = baseShape();
    narrow.c = 64;
    narrow.input = {1100000, 16};
    auto rows = rowsFields();
    rows.bc = 32;
    rows.wc = 1;
    rows.pb = 1;
    EXPECT_FALSE(admits(rows, "gfx942", narrow)); // 64 dy channel planes: 2.25 GB
    rows.bk = 32;
    rows.wk = 1;
    EXPECT_TRUE(admits(rows, "gfx942", narrow));
}

TEST(TestConvWrwKernelMatch, GenericBlockBuffersServeLargeImagesThroughStepWindows)
{
    // 64 channels of 4100 x 4200 fp16, channels-last: a 2.2 GB image, 3x3 stride 2.
    auto large = baseShape();
    large.c = 64;
    large.k = 32;
    large.input = {4100, 4200};
    large.prePadding = large.postPadding = {1, 1};
    large.stride = {2, 2};
    large.channelsLast = true;
    auto f = genericClFields();
    EXPECT_FALSE(admits(f, "gfx942", large)); // the image is past the range
    f.pb = 1;
    EXPECT_TRUE(admits(f, "gfx942", large));
    // The step's windows: x from the first reachable element, 64 channels (unit stride), 2
    // more rows of 4200 x 64, and 31 x 2 + 2 more columns; dy 32 channels by 31 more columns.
    const auto launch = launchOf(f, large);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->args.xImage, 1 + 63 + 2 * 4200 * 64 + (31 * 2 + 2) * 64);
    EXPECT_EQ(launch->args.yImage, 1 + 31 + 31 * 32);
    // And the window variant's extra positions (unit width stride: 31 + 2 + 8 columns).
    auto window = f;
    window.dw = 8;
    auto unitWidth = large;
    unitWidth.stride = {2, 1};
    const auto windowed = launchOf(window, unitWidth);
    ASSERT_TRUE(windowed.has_value());
    EXPECT_EQ(windowed->args.xImage, 1 + 63 + 2 * 4200 * 64 + (31 + 2 + 8) * 64);
    // pb 1 on a small image: declined (the specialized and per-image configs serve it).
    EXPECT_FALSE(admits(f, "gfx942", channelsLastShape()));
    // Channels-first: the step window spans the block's channel planes (2.2 GB 33000 x 33000
    // fp16 planes are past it) but only the columns a chunk reaches (one 2^31-byte row at
    // width stride 2^20 is within it).
    auto planes = baseShape();
    planes.c = 2;
    planes.k = 1; // one dy plane: the x window alone past the range
    planes.input = {33000, 33000};
    planes.prePadding = planes.postPadding = {1, 1};
    auto cf = genericFields();
    cf.pb = 1;
    EXPECT_FALSE(admits(cf, "gfx942", planes));
    // The dy window alone: one 0.8 GB x plane, 8 dy planes of it (7 more in the window).
    auto dyPlanes = baseShape();
    dyPlanes.n = 1;
    dyPlanes.c = 1;
    dyPlanes.k = 8;
    dyPlanes.input = {20000, 20000};
    EXPECT_FALSE(admits(cf, "gfx942", dyPlanes));
    dyPlanes.k = 2;
    dyPlanes.input = {20000, 30000}; // 1 more 1.2 GB dy plane: within
    EXPECT_TRUE(admits(cf, "gfx942", dyPlanes));
    auto row = baseShape();
    row.n = 1;
    row.c = 1;
    row.k = 8;
    row.input = {1, int64_t{1} << 30}; // fp16: 2^31 bytes in one row
    row.filter = {1, 1};
    row.prePadding = row.postPadding = {0, 0};
    row.stride = {1, int64_t{1} << 20};
    EXPECT_TRUE(admits(cf, "gfx942", row));
    auto cfLarge = large;
    cfLarge.channelsLast = false;
    cfLarge.c = 2;
    cfLarge.k = 1;
    cfLarge.input = {33000, 16500}; // two 1.1 GB planes: window within, image past
    EXPECT_TRUE(admits(cf, "gfx942", cfLarge));
    // A specialized block-buffer class (lower case) declines the generic twin.
    cf.spec = "4r0.3.2.1.0";
    EXPECT_FALSE(admits(cf, "gfx942", cfLarge));
    cf.spec = "4R0.3.2.1.0";
    EXPECT_TRUE(admits(cf, "gfx942", cfLarge));
}

TEST(TestConvWrwKernelMatch, PackedRowsServeNarrowChannelsFirstOutputsOnly)
{
    // Virtual rows of wo + (s - 1) dw / sw columns, two within bq = 32: wo up to 14 at unit
    // width stride (3 taps), up to 15 at stride 2.
    auto shape = baseShape();
    EXPECT_TRUE(admits(packedFields(), "gfx942", shape)); // wo 12
    shape.input = {10, 14};
    EXPECT_TRUE(admits(packedFields(), "gfx942", shape)); // wo 14: 2 x 16 = 32
    shape.input = {10, 15};
    EXPECT_FALSE(admits(packedFields(), "gfx942", shape)); // wo 15: 2 x 17 = 34
    auto s2 = baseShape();
    s2.stride = {2, 2};
    auto f = packedFields();
    f.sw = 2;
    s2.input = {10, 29}; // wo 15: 2 x 16
    EXPECT_TRUE(admits(f, "gfx942", s2));
    s2.input = {10, 31}; // wo 16: 2 x 17
    EXPECT_FALSE(admits(f, "gfx942", s2));
    // Channels-first only: channels-last graph and config.
    auto cl = baseShape();
    cl.channelsLast = true;
    f = packedFields();
    f.layout = 1;
    EXPECT_FALSE(admits(f, "gfx942", cl));
    // Element units carry no pass mask: up to 64 passes (gfx950's LDS takes the tall x tiles).
    // 4 filter rows of 64 channels by 40 positions over 256 threads: 40 passes; 7: 70.
    f = packedFields();
    f.atom = "mfma32x16_f16";
    f.atomId = 10;
    f.xcd = 1;
    f.fr = 4;
    EXPECT_TRUE(admits(f, "gfx950", baseShape()));
    f.fr = 7;
    EXPECT_FALSE(admits(f, "gfx950", baseShape()));
}

TEST(TestConvWrwKernelMatch, SmallCRowsServeFewChannelsFirstChannels)
{
    // Three input channels, channels-first: served; 24 (past 16) is not.
    EXPECT_TRUE(admits(smallcFields(), "gfx942", stemShape()));
    EXPECT_FALSE(admits(smallcFields(), "gfx942", baseShape()));
    auto many = stemShape();
    many.c = 17;
    EXPECT_FALSE(admits(smallcFields(), "gfx942", many));
    many.c = 16;
    EXPECT_TRUE(admits(smallcFields(), "gfx942", many));
    // Channels-last: any input channel count up to 16 (x staged an element at a time), the
    // output channels whole vectors (dy units of 16 bytes).
    auto cl = stemShape();
    cl.channelsLast = true;
    auto f = smallcFields();
    f.layout = 1;
    EXPECT_TRUE(admits(f, "gfx942", cl));
    cl.c = 8;
    EXPECT_TRUE(admits(f, "gfx942", cl));
    cl.c = 17;
    EXPECT_FALSE(admits(f, "gfx942", cl));
    cl.c = 3;
    cl.k = 36;
    EXPECT_FALSE(admits(f, "gfx942", cl));
    EXPECT_FALSE(admits(f, "gfx942", stemShape())); // channels-last config, channels-first graph
    // Not kept channels-last for transpose reads (gfx950's layout 2).
    auto clStem = stemShape();
    clStem.channelsLast = true;
    auto tr = smallcFields();
    tr.atom = "mfma16x32_f16";
    tr.atomId = 8;
    tr.layout = 1;
    EXPECT_TRUE(admits(tr, "gfx950", clStem));
    tr.layout = 2;
    EXPECT_FALSE(admits(tr, "gfx950", clStem));
    // The channels-first tiles' geometry: bq whole vectors (mfma32_f32 steps 2 positions) and
    // vec whole width-stride phases.
    auto f32 = clStem;
    f32.dataType = DataType::FLOAT;
    auto steps = smallcFields();
    steps.atom = "mfma32_f32";
    steps.atomId = 7;
    steps.layout = 1;
    steps.wk = 2;
    steps.vec = 4;
    steps.bq = 64;
    EXPECT_TRUE(admits(steps, "gfx942", f32));
    steps.bq = 66;
    EXPECT_FALSE(admits(steps, "gfx942", f32));
    auto s3 = clStem;
    s3.input = {10, 31};
    s3.stride = {1, 3};
    auto phases = smallcFields();
    phases.layout = 1;
    phases.sw = 3;
    EXPECT_FALSE(admits(phases, "gfx942", s3));
    s3.stride = {1, 2};
    phases.sw = 2;
    EXPECT_TRUE(admits(phases, "gfx942", s3));
    f = smallcFields();
    // Any stride: the rows loop's grid, filter rows per block.
    auto s2 = stemShape();
    s2.stride = {2, 2};
    f = smallcFields();
    f.sw = 2;
    EXPECT_TRUE(admits(f, "gfx942", s2));
    const auto launch = launchOf(f, s2, "gfx942", 304);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->block, 256u);
    EXPECT_EQ(launch->gridY, 1u); // 3 filter rows, fr 3
}

TEST(TestConvWrwKernelMatch, TheRingRunsHeightStridesAsPhases)
{
    // 3x3 at height stride 2: the matrix ring with ceil(3 / 2) = 2 filter rows per phase.
    auto f = ConfigFields{};
    f.fr = 2;
    EXPECT_TRUE(admits(f, "gfx942", heightStride2Shape()));
    for(const int64_t fr : {1, 3})
    {
        f.fr = fr;
        EXPECT_FALSE(admits(f, "gfx942", heightStride2Shape())) << fr;
    }
    // Not with a height dilation, and not on the DMA ring.
    f.fr = 2;
    auto dilated = heightStride2Shape();
    dilated.dilation = {2, 1};
    dilated.prePadding = {2, 1};
    dilated.postPadding = {2, 1};
    EXPECT_FALSE(admits(f, "gfx942", dilated));
    auto dma = dmaFields();
    dma.fr = 2;
    EXPECT_FALSE(admits(dma, "gfx950", heightStride2Shape()));
    // One grid row per phase; tracks walk the dy rows plus the ring's warm-up.
    const auto launch = launchOf(f, heightStride2Shape(), "gfx942", 304);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->gridY, 2u);
    EXPECT_EQ(launch->args.trackLen, launch->args.ho + 1);
}

TEST(TestConvWrwKernelMatch, UnalignedDyUnitsOnlyWhereOutputChannelsAreNotVectors)
{
    // Align 2 serves 3 output channels and declines 40 (whole vectors: the aligned configs).
    EXPECT_TRUE(admits(unalignedKFields(), "gfx942", unalignedKShape()));
    EXPECT_FALSE(admits(unalignedKFields(), "gfx942", channelsLastShape()));
    // Input channels must still be whole vectors.
    auto oddC = unalignedKShape();
    oddC.c = 20;
    EXPECT_FALSE(admits(unalignedKFields(), "gfx942", oddC));
}

TEST(TestConvWrwKernelMatch, TheDirectLoopServesItsOwnShapes)
{
    // Starting points of the direct refusals that change the shape: they hold on the direct
    // loop's own form of each (unit height dilation, unit width stride), and the bf16 atom.
    auto dilatedWidth = channelsLastShape();
    dilatedWidth.dilation = {1, 2};
    dilatedWidth.prePadding = {1, 2};
    dilatedWidth.postPadding = {1, 2};
    auto f = directFields();
    f.dw = 2;
    EXPECT_TRUE(admits(f, "gfx950", dilatedWidth));
    // Waves_q 1 and 4 with their own wave splits; 2 and 5 ring slots.
    f = directFields();
    f.wq = 1;
    f.wk = 4;
    EXPECT_TRUE(admits(f, "gfx950", channelsLastShape()));
    f = directFields();
    f.bc = 32;
    f.wc = 1;
    f.wq = 4;
    EXPECT_TRUE(admits(f, "gfx950", channelsLastShape()));
    for(const int64_t ns : {2, 5})
    {
        f = directFields();
        f.ns = ns;
        EXPECT_TRUE(admits(f, "gfx950", channelsLastShape())) << ns;
    }
    // A one-row filter takes any height dilation.
    auto tall = channelsLastShape();
    tall.filter = {1, 3};
    tall.prePadding = {0, 1};
    tall.postPadding = {0, 1};
    tall.dilation = {3, 1};
    f = directFields();
    f.fr = 1;
    EXPECT_TRUE(admits(f, "gfx950", tall));
    // A taller filter needs unit height dilation, and the width stride is 1.
    auto dilatedHeight = channelsLastShape();
    dilatedHeight.dilation = {2, 1};
    dilatedHeight.prePadding = {2, 1};
    dilatedHeight.postPadding = {2, 1};
    EXPECT_FALSE(admits(directFields(), "gfx950", dilatedHeight));
    auto strided = widthStride2Shape();
    strided.channelsLast = true;
    f = directFields();
    f.sw = 2;
    EXPECT_FALSE(admits(f, "gfx950", strided));
}

TEST(TestConvWrwDirectPlan, CellsForFewTilesRunsForMany)
{
    // flux n8 c256 k128 768^2 on 64 x 128 tiles: 4 tiles, 64 splits of 3 items, 256 workgroups.
    const auto flux = conv_wrw::directPlanFor(
        4, 8, 768, 24, 3, int64_t{9} * 256 * 128 * 4, 256, int64_t{64} * 128 * 9 * 32);
    EXPECT_EQ(flux.walk, 0);
    EXPECT_EQ(flux.segs, 1);
    EXPECT_EQ(flux.items, 192);
    EXPECT_EQ(flux.per, 3);
    EXPECT_EQ(flux.splits, 64);
    EXPECT_EQ(flux.grid, 256);
    // sdxl n8 c1280 k1280 96^2: 200 tiles run 19 units a workgroup, 3 pieces at most per tile.
    const auto sdxl = conv_wrw::directPlanFor(
        200, 8, 96, 3, 3, int64_t{9} * 1280 * 1280 * 4, 256, int64_t{64} * 128 * 9 * 32);
    EXPECT_EQ(sdxl.walk, 1);
    EXPECT_EQ(sdxl.segs, 1);
    EXPECT_EQ(sdxl.items, 24);
    EXPECT_EQ(sdxl.per, 19);
    EXPECT_EQ(sdxl.splits, 3);
    EXPECT_EQ(sdxl.grid, 253);
    // One tall narrow plane: row segments of at least 4 rows spread it.
    const auto narrow = conv_wrw::directPlanFor(
        1, 1, 40, 1, 3, int64_t{9} * 64 * 64 * 4, 256, int64_t{64} * 64 * 9 * 32);
    EXPECT_GT(narrow.segs, 1);
    EXPECT_GE(narrow.segRows, 4);
    EXPECT_GE(narrow.segs * narrow.segRows, 40);
    EXPECT_EQ(narrow.items, narrow.segs);
}

TEST(TestConvWrwLaunch, TheDirectLoopLaunchesItsPlan)
{
    const auto launch = launchOf(directFields(), channelsLastShape(), "gfx950", 256);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->block, 512u);
    EXPECT_EQ(launch->gridY, 1u);
    EXPECT_EQ(launch->gridZ, 1u);
    // 1 tile (c 24, k 40 in one 64 x 64 tile), 2 images, 1 column block: the planner's segments.
    const auto plan = conv_wrw::directPlanFor(
        1, 2, 10, 1, 3, int64_t{9} * 40 * 24 * 4, 256, int64_t{64} * 64 * 9 * 32);
    EXPECT_EQ(launch->gridX, static_cast<unsigned>(plan.grid));
    EXPECT_EQ(launch->args.segs, plan.segs);
    EXPECT_EQ(launch->args.segRows, plan.segRows);
    EXPECT_EQ(launch->args.walk, plan.walk);
    EXPECT_EQ(launch->args.steps, plan.items);
    EXPECT_EQ(launch->args.stepsPerSplit, plan.per);
    EXPECT_EQ(launch->args.splits, plan.splits);
    EXPECT_EQ(launch->args.trackLen, 0);
    EXPECT_EQ(launch->workspaceBytes, static_cast<size_t>(plan.splits) * 9 * 40 * 24 * 4);
}

TEST(TestConvWrwKernelMatch, TransposeReadsServeEveryMatrixLoopOnGfx950)
{
    auto f = trFields();
    EXPECT_TRUE(admits(f, "gfx950", channelsLastShape()));
    f.loop = 0;
    EXPECT_TRUE(admits(f, "gfx950", channelsLastShape()));
    f.loop = 2;
    f.ns = 2;
    EXPECT_TRUE(admits(f, "gfx950", channelsLastShape()));
    // Channels-last graphs only.
    EXPECT_FALSE(admits(f, "gfx950", baseShape()));
}

TEST(TestConvWrwKernelMatch, TheFmaLoopsTake16BitOnlyWithoutAMatrixCoreForIt)
{
    // A 16-bit FMA config otherwise whole (8-element vectors, a 16-wide row): only where no
    // matrix atom takes the type (gfx906, gfx900); fp32 everywhere.
    auto shape = alignedShape();
    auto f = fmaFields();
    f.vec = 8;
    EXPECT_FALSE(admits(f, "gfx1100", shape));
    EXPECT_FALSE(admits(f, "gfx942", shape));
    EXPECT_FALSE(admits(f, "gfx908", shape));
    EXPECT_TRUE(admits(f, "gfx906", shape));
    EXPECT_TRUE(admits(f, "gfx906:sramecc+:xnack-", shape));
    EXPECT_TRUE(admits(f, "gfx900", shape));
    auto bf16 = shape;
    bf16.dataType = DataType::BFLOAT16;
    EXPECT_TRUE(admits(f, "gfx906", bf16));
    EXPECT_FALSE(admits(f, "gfx908", bf16)); // gfx908's own bf16 atom
    EXPECT_FALSE(admits(f, "gfx1201", bf16));
    shape.dataType = DataType::FLOAT;
    f.vec = 4;
    EXPECT_TRUE(admits(f, "gfx1100", shape));
    EXPECT_TRUE(admits(f, "gfx906", shape));
    EXPECT_FALSE(conv_wrw::runsMatrixAtomFor("f16", "gfx906"));
    EXPECT_FALSE(conv_wrw::runsMatrixAtomFor("f32", "gfx1100"));
    EXPECT_TRUE(conv_wrw::runsMatrixAtomFor("bf16", "gfx908"));
    EXPECT_TRUE(conv_wrw::runsMatrixAtomFor("f16", "gfx1201"));
}

TEST(TestConvWrwKernelMatch, FmaTilesAreWholeVectorsOfChannels)
{
    // 16-bit vectors of 8 on gfx906: a 4-channel tile (one thread quad) is not one; fp32's 4 is.
    auto shape = alignedShape();
    shape.c = 4;
    auto f = fmaFields();
    f.vec = 8;
    f.bc = 8;
    EXPECT_TRUE(admits(f, "gfx906", shape));
    f.bc = 4;
    EXPECT_FALSE(admits(f, "gfx906", shape));
    f.bc = 8;
    f.bk = 36;
    f.tk = 4;
    EXPECT_FALSE(admits(f, "gfx906", shape));
    shape.dataType = DataType::FLOAT;
    f.vec = 4;
    f.bk = 32;
    f.bc = 4;
    EXPECT_TRUE(admits(f, "gfx906", shape));
}

TEST(TestConvWrwKernelMatch, TheFmaRingNeedsUnitWidthStrideAndWholeVectorRows)
{
    // Width stride 2: the matrix-core ring serves it, the FMA ring does not.
    auto strided = widthStride2Shape();
    strided.dataType = DataType::FLOAT;
    auto matrix = floatFields();
    matrix.sw = 2;
    EXPECT_TRUE(admits(matrix, "gfx942", strided));
    auto f = fmaFields();
    f.sw = 2;
    EXPECT_FALSE(admits(f, "gfx942", strided));
    // Rows of x and dy whole vectors of 4: 12 wide yes, 13 wide no.
    EXPECT_TRUE(admits(fmaFields(), "gfx1100", floatShape()));
    auto odd = floatShape();
    odd.input = {10, 13};
    EXPECT_FALSE(admits(fmaFields(), "gfx1100", odd));
    auto oddOut = floatShape(); // x 12 wide, dy 11 wide (no padding on the right)
    oddOut.postPadding = {1, 0};
    EXPECT_FALSE(admits(fmaFields(), "gfx1100", oddOut));
    auto oddIn = floatShape(); // x 13 wide, dy 12 wide
    oddIn.input = {10, 13};
    oddIn.postPadding = {1, 0};
    EXPECT_FALSE(admits(fmaFields(), "gfx1100", oddIn));
}

TEST(TestConvWrwKernelMatch, TheFmaRowsTakeAnyStrideAndScalarUnits)
{
    // Width stride 2 (x 16, dy 8): the FMA output rows (one filter row per block, so the
    // windows fit gfx1100's LDS) serve it, the FMA ring does not.
    auto strided = widthStride2Shape();
    strided.dataType = DataType::FLOAT;
    auto rows = fmaFields();
    rows.loop = 4;
    rows.fr = 1;
    rows.sw = 2;
    EXPECT_TRUE(admits(rows, "gfx1100", strided));
    auto ring = rows;
    ring.loop = 3;
    ring.fr = 3;
    EXPECT_FALSE(admits(ring, "gfx1100", strided));
    // Height stride 2: the output rows, not the ring.
    auto tall = floatShape();
    tall.stride = {2, 1};
    auto rowsTall = fmaFields();
    rowsTall.loop = 4;
    rowsTall.fr = 1;
    EXPECT_TRUE(admits(rowsTall, "gfx1100", tall));
    EXPECT_FALSE(admits(fmaFields(), "gfx1100", tall));
    // Rows not whole vectors of 4 (x 13): scalar units serve them, vectors of 4 do not; the
    // ring takes no scalar units.
    auto odd = floatShape();
    odd.input = {10, 13};
    EXPECT_FALSE(admits(rowsTall, "gfx1100", odd));
    auto scalar = rowsTall;
    scalar.vec = 1;
    EXPECT_TRUE(admits(scalar, "gfx1100", odd));
    auto ringScalar = fmaFields();
    ringScalar.vec = 1;
    EXPECT_FALSE(admits(ringScalar, "gfx1100", floatShape()));
    // Channels-last rows need no whole vectors: vectors of 4 channels serve x 13.
    auto oddNhwc = odd;
    oddNhwc.channelsLast = true;
    auto rowsNhwc = rowsTall;
    rowsNhwc.layout = 1;
    EXPECT_TRUE(admits(rowsNhwc, "gfx1100", oddNhwc));
    // Width stride 3 on vectors of 4 (x 24, dy 8): the output rows need no vector of whole
    // stride phases.
    auto stride3 = floatShape();
    stride3.input = {10, 24};
    stride3.stride = {1, 3};
    auto rows3 = rowsTall;
    rows3.sw = 3;
    EXPECT_TRUE(admits(rows3, "gfx1100", stride3));
    // A loop past the last is refused.
    auto past = rowsTall;
    past.loop = 5;
    EXPECT_FALSE(admits(past, "gfx1100", floatShape()));
}

TEST(TestConvWrwKernelMatch, AlignedUnitsNeedWholeVectorRowsOfXAndDy)
{
    // fp16 vectors of 8: x and dy 16 wide yes; x 16 and dy 15, or x 15 and dy 16, no.
    auto shape = alignedShape();
    EXPECT_TRUE(admits(alignedFields(), "gfx942", shape));
    shape.postPadding = {1, 0}; // dy 15
    EXPECT_FALSE(admits(alignedFields(), "gfx942", shape));
    shape.input = {10, 15};
    shape.postPadding = {1, 2}; // x 15, dy 16
    EXPECT_FALSE(admits(alignedFields(), "gfx942", shape));
    EXPECT_TRUE(admits(ConfigFields{}, "gfx942", shape));
}

TEST(TestConvWrwKernelMatch, TheDtypeMustBeTheGraphs)
{
    const GraphFixture fixture(buildGraph(specOf(baseShape())), deviceOf("gfx942"));
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(matchesKernel(PACK, fixture.context(), kernelOf(ConfigFields{}, "HALF"), *bound));
    EXPECT_FALSE(
        matchesKernel(PACK, fixture.context(), kernelOf(ConfigFields{}, "BFLOAT16"), *bound));
}

TEST(TestConvWrwKernelMatch, MissingMetadataRefuses)
{
    const GraphFixture fixture(buildGraph(specOf(baseShape())), deviceOf("gfx942"));
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    for(const char* field :
        {"atom", "atom_id", "loop", "layout", "bk",  "bc",  "bq",    "fr", "s",   "sw",       "dw",
         "wk",   "wc",      "tk",   "tc",     "vec", "xcd", "align", "ns", "bpu", "direction"})
    {
        auto kernel = kernelOf(ConfigFields{}, "HALF");
        kernel.metadata.erase(field);
        EXPECT_FALSE(matchesKernel(PACK, fixture.context(), kernel, *bound)) << field;
    }
}

TEST(TestConvWrwKernelMatch, ChannelsFirstVectorsHoldWholeStridePhases)
{
    // A 16-byte fp16 vector (8 elements) splits into 2 phases of a width stride of 2, not 3.
    EXPECT_TRUE(admits(stridedRowsFields(2), "gfx942", widthStrideShape(2)));
    EXPECT_FALSE(admits(stridedRowsFields(3), "gfx942", widthStrideShape(3)));
    // Channels-last vectors run along the channels: stride 3 is served.
    auto shape = widthStrideShape(3);
    shape.channelsLast = true;
    auto f = stridedRowsFields(3);
    f.layout = 1;
    EXPECT_TRUE(admits(f, "gfx942", shape));
}

TEST(TestConvWrwKernelMatch, TheDmaRingRunsOnGfx950Only)
{
    // The same DMA config on an atom both devices run.
    auto f = dmaFields();
    f.atom = "mfma32_f16";
    f.atomId = 4;
    EXPECT_TRUE(admits(f, "gfx950"));
    EXPECT_FALSE(admits(f, "gfx942"));
    f.loop = 1;
    f.ns = 0;
    EXPECT_TRUE(admits(f, "gfx942"));
}

TEST(TestConvWrwKernelMatch, ADeviceWithoutDispatchUnitsIsDeclined)
{
    EXPECT_TRUE(admits(ConfigFields{}, "gfx942", baseShape(), 1));
    EXPECT_FALSE(admits(ConfigFields{}, "gfx942", baseShape(), 0));
}

TEST(TestConvWrwKernelMatch, ChannelsLastNeedsWholeVectorsOfChannelsPerGroup)
{
    auto shape = channelsLastShape();
    EXPECT_TRUE(admits(channelsLastFields(), "gfx942", shape));
    shape.c = 20; // 20 input channels: not whole vectors of 8
    EXPECT_FALSE(admits(channelsLastFields(), "gfx942", shape));
    shape.c = 24;
    shape.k = 36; // 36 output channels
    EXPECT_FALSE(admits(channelsLastFields(), "gfx942", shape));
    // Grouped: per group.
    shape.k = 48;
    shape.groups = 2; // 12 and 24 per group
    EXPECT_FALSE(admits(channelsLastFields(), "gfx942", shape));
    shape.c = 48; // 24 and 24 per group
    EXPECT_TRUE(admits(channelsLastFields(), "gfx942", shape));
}

TEST(TestConvWrwKernelMatch, ChannelsFirstImagesHoldAVector)
{
    auto shape = baseShape();
    shape.c = 1;
    shape.k = 8;
    shape.input = {1, 7};
    shape.filter = {1, 3};
    shape.prePadding = shape.postPadding = {0, 1};
    auto f = ConfigFields{};
    f.fr = 1;
    EXPECT_FALSE(admits(f, "gfx942", shape)); // x: 7 elements
    shape.input = {1, 8};
    EXPECT_TRUE(admits(f, "gfx942", shape));
}

TEST(TestConvWrwKernelMatch, TheLdsLimitIsTheDevices)
{
    auto f = ConfigFields{};
    f.bk = 128;
    f.bc = 128;
    f.wk = 4;
    f.wc = 4;
    f.bq = 64;
    EXPECT_GT(conv_wrw::ldsBytesOf(configOf(f), 2), 64 * 1024);
    EXPECT_FALSE(admits(f, "gfx942"));
    EXPECT_TRUE(admits(f, "gfx950"));
    EXPECT_EQ(conv_wrw::ldsLimitOf("gfx950:sramecc+:xnack-"), 160 * 1024);
    EXPECT_EQ(conv_wrw::ldsLimitOf("gfx942"), 64 * 1024);
    EXPECT_EQ(conv_wrw::ldsLimitOf("gfx1100"), 64 * 1024);
}

TEST(TestConvWrwKernelMatch, LoadPassesStayWithinTheirMask)
{
    // One wave, 256 output channels by 80 positions: 2560 dy units, 40 per thread.
    auto f = dmaFields();
    f.loop = 1;
    f.ns = 0;
    f.bk = 256;
    f.bc = 32;
    f.wk = 1;
    f.wc = 1;
    f.bq = 80;
    EXPECT_LE(conv_wrw::ldsBytesOf(configOf(f), 2), 160 * 1024);
    EXPECT_FALSE(admits(f, "gfx950"));
    f.bq = 64; // 2048 units: 32 per thread
    EXPECT_TRUE(admits(f, "gfx950"));
}

TEST(TestConvWrwKernelMatch, TheGridLimitsTheGroupsTimesFilterRows)
{
    // Depthwise 1x1 over 65536 groups: grid y = groups x depth taps.
    auto shape = baseShape();
    shape.n = 1;
    shape.groups = 65536;
    shape.c = 65536;
    shape.k = 65536;
    shape.input = {1, 8};
    shape.filter = {1, 1};
    shape.prePadding = shape.postPadding = {0, 0};
    auto f = ConfigFields{};
    f.fr = 1;
    f.s = 1;
    EXPECT_FALSE(admits(f, "gfx942", shape));
    shape.groups = shape.c = shape.k = 65535;
    EXPECT_TRUE(admits(f, "gfx942", shape));
}

TEST(TestConvWrwKernelMatch, TheAtomsRunOnTheirArchitectures)
{
    auto f = ConfigFields{};
    EXPECT_TRUE(admits(f, "gfx942:sramecc+:xnack-"));
    EXPECT_TRUE(admits(f, "gfx90a"));
    EXPECT_TRUE(admits(f, "gfx908"));
    EXPECT_FALSE(admits(f, "gfx1100"));
    auto bf16 = baseShape();
    bf16.dataType = DataType::BFLOAT16;
    auto b = ConfigFields{};
    b.atom = "mfma32_bf16";
    b.atomId = 5;
    EXPECT_TRUE(admits(b, "gfx90a", bf16));
    EXPECT_FALSE(admits(b, "gfx908", bf16));
    b.atom = "mfma32_bf16_908";
    b.atomId = 6;
    EXPECT_TRUE(admits(b, "gfx908", bf16));
    auto w = ConfigFields{};
    w.atom = "wmma11_f16";
    w.atomId = 12;
    w.bk = 64;
    w.bc = 64;
    w.wk = 4;
    w.wc = 4;
    EXPECT_TRUE(admits(w, "gfx1100"));
    EXPECT_FALSE(admits(w, "gfx1201"));
    w.atom = "wmma12_f16";
    w.atomId = 14;
    EXPECT_TRUE(admits(w, "gfx1201"));
    // fp32 runs on the f32 MFMA atoms only (no fp32 WMMA).
    EXPECT_TRUE(admits(floatFields(), "gfx942", floatShape()));
    EXPECT_FALSE(admits(floatFields(), "gfx1100", floatShape()));
    // The FMA ring runs fp32 everywhere.
    EXPECT_TRUE(admits(fmaFields(), "gfx1100", floatShape()));
    EXPECT_TRUE(admits(fmaFields(), "gfx1201", floatShape()));
    EXPECT_TRUE(admits(fmaFields(), "gfx942", floatShape()));
}

TEST(TestConvWrwAtoms, IdsFollowTheTableAndShapesTheInstructions)
{
    EXPECT_EQ(conv_wrw::atomIdOf("mfma16_f16"), 0);
    EXPECT_EQ(conv_wrw::atomIdOf("mfma32_f16"), 4);
    EXPECT_EQ(conv_wrw::atomIdOf("mfma32x16_f16"), 10);
    EXPECT_EQ(conv_wrw::atomIdOf("wmma12_bf16"), 15);
    EXPECT_EQ(conv_wrw::atomIdOf("mfma32_xf32"), -1);
    EXPECT_EQ(conv_wrw::atomShapeOf("mfma32_f32").dim, 32);
    EXPECT_EQ(conv_wrw::atomShapeOf("mfma32_f32").ka, 2);
    EXPECT_EQ(conv_wrw::atomShapeOf("mfma16x32_bf16").ka, 32);
    EXPECT_EQ(conv_wrw::atomShapeOf("mfma32_bf16_908").ka, 4);
    EXPECT_EQ(conv_wrw::atomShapeOf("wmma11_f16").dim, 16);
    EXPECT_EQ(conv_wrw::atomShapeOf("nope").dim, 0);
    EXPECT_EQ(conv_wrw::atomIdOf("fma"), 16);
    EXPECT_EQ(conv_wrw::atomShapeOf("fma").dim, 0);
    EXPECT_EQ(conv_wrw::atomShapeOf("mfma32x16_f16").kpl, 8);
    EXPECT_EQ(conv_wrw::atomShapeOf("wmma12_f16").kpl, 8);
    EXPECT_EQ(conv_wrw::atomShapeOf("wmma11_f16").kpl, 16);
    EXPECT_EQ(conv_wrw::atomShapeOf("mfma32_f16").kpl, 4);
}

// ---------------------------------------------------------------------------
// The argument blocks
// ---------------------------------------------------------------------------

TEST(TestConvWrwLaunch, TheRingFillsTracksStepsSplitsAndTheWorkspace)
{
    const auto launch = launchOf(ConfigFields{});
    ASSERT_TRUE(launch.has_value());
    const auto& a = launch->args;
    EXPECT_EQ(a.n, 2);
    EXPECT_EQ(a.c, 24);
    EXPECT_EQ(a.k, 40);
    EXPECT_EQ(a.groups, 1);
    EXPECT_EQ(a.di, 1);
    EXPECT_EQ(a.hi, 10);
    EXPECT_EQ(a.wi, 12);
    EXPECT_EQ(a.dout, 1);
    EXPECT_EQ(a.ho, 10);
    EXPECT_EQ(a.wo, 12);
    EXPECT_EQ(a.ft, 1);
    EXPECT_EQ(a.fr, 3);
    EXPECT_EQ(a.fs, 3);
    EXPECT_EQ(a.sd, 1);
    EXPECT_EQ(a.pd, 0);
    EXPECT_EQ(a.ph, 1);
    EXPECT_EQ(a.pw, 1);
    EXPECT_EQ(a.dd, 1);
    // x and dy channels-first: (n, c, d, h, w).
    EXPECT_EQ(a.xs[0], 24 * 120);
    EXPECT_EQ(a.xs[1], 120);
    EXPECT_EQ(a.xs[2], 120);
    EXPECT_EQ(a.xs[3], 12);
    EXPECT_EQ(a.xs[4], 1);
    EXPECT_EQ(a.ys[0], 40 * 120);
    EXPECT_EQ(a.ys[1], 120);
    EXPECT_EQ(a.ys[4], 1);
    EXPECT_EQ(a.xImage, 24 * 120);
    EXPECT_EQ(a.yImage, 40 * 120);
    // One 32-position chunk per row; tracks of ceil(10 / 1) + 3 - 1 steps.
    EXPECT_EQ(a.chunks, 1);
    EXPECT_EQ(a.trackLen, 12);
    EXPECT_EQ(a.steps, 2 * 1 * 1 * 1 * 12);
    // One block per split (1 x 1 channel tiles, one depth tap): 4 waves want 2 per unit,
    // 608 splits, capped at steps / 4 = 6: 4 steps each.
    EXPECT_EQ(launch->gridX, 1u);
    EXPECT_EQ(launch->gridY, 1u);
    EXPECT_EQ(a.stepsPerSplit, 4);
    EXPECT_EQ(a.splits, 6);
    EXPECT_EQ(launch->gridZ, 6u);
    EXPECT_EQ(launch->block, 256u);
    EXPECT_EQ(a.partialElems, 9 * 40 * 24);
    EXPECT_EQ(launch->workspaceBytes, size_t{6} * 9 * 40 * 24 * 4);
    EXPECT_EQ(launch->kernelType, "K1AtomMfma32F16");
    // The reduce: 6 splits, so quads (the row reduce takes at most 2): 9 taps x 40 k x 6
    // quads of c, 64 per block.
    EXPECT_EQ(launch->reduceGrid, (9u * 40 * 6 + 63) / 64);
    EXPECT_EQ(launch->reduce.rows, 0);
    EXPECT_EQ(launch->reduceBlock, 256u);
    EXPECT_EQ(launch->reduce.elems, 40 * 24 * 9);
    EXPECT_EQ(launch->reduce.partialElems, a.partialElems);
    EXPECT_EQ(launch->reduce.splits, 6);
    EXPECT_EQ(launch->reduce.groups, 1);
    EXPECT_EQ(launch->reduce.kg, 40);
    EXPECT_EQ(launch->reduce.cg, 24);
    EXPECT_EQ(launch->reduce.taps, 9);
}

TEST(TestConvWrwLaunch, TheRowsLoopCountsOutputRowsAndFilterRowGroups)
{
    auto shape = baseShape();
    shape.input = {4, 7, 9};
    shape.filter = {3, 3, 3};
    shape.prePadding = shape.postPadding = {1, 1, 1};
    shape.stride = {2, 2, 1};
    shape.dilation = {1, 1, 1};
    auto f = rowsFields();
    f.fr = 2; // 9 filter rows in groups of 2: 5 groups
    const auto launch = launchOf(f, shape);
    ASSERT_TRUE(launch.has_value());
    const auto& a = launch->args;
    EXPECT_EQ(a.dout, 2);
    EXPECT_EQ(a.ho, 4);
    EXPECT_EQ(a.wo, 9);
    EXPECT_EQ(a.chunks, 1);
    EXPECT_EQ(a.trackLen, 0);
    EXPECT_EQ(a.steps, 2 * 2 * 4 * 1);
    EXPECT_EQ(launch->gridY, 5u);
    EXPECT_EQ(a.partialElems, 27 * 40 * 24);
    EXPECT_EQ(a.splits, 4); // steps / 4
    EXPECT_EQ(a.stepsPerSplit, 4);
}

TEST(TestConvWrwLaunch, TheRingRunsADilatedHeightAsTracksPerClass)
{
    auto shape = baseShape();
    shape.input = {20, 12};
    shape.dilation = {3, 1};
    shape.prePadding = shape.postPadding = {3, 1};
    const auto launch = launchOf(ConfigFields{}, shape);
    ASSERT_TRUE(launch.has_value());
    // ho 20: tracks of ceil(20 / 3) + 2 steps, one per height class of 3.
    EXPECT_EQ(launch->args.trackLen, 9);
    EXPECT_EQ(launch->args.steps, 2 * 1 * 3 * 1 * 9);
}

TEST(TestConvWrwLaunch, ChannelsLastStridesAndGroups)
{
    auto shape = channelsLastShape();
    shape.groups = 3;
    shape.c = 24;
    shape.k = 48;
    auto f = channelsLastFields();
    f.bk = 64;
    f.bc = 64;
    const auto launch = launchOf(f, shape);
    ASSERT_TRUE(launch.has_value());
    const auto& a = launch->args;
    EXPECT_EQ(a.groups, 3);
    // x (n, c, d, h, w) as stored n d h w c.
    EXPECT_EQ(a.xs[0], 120 * 24);
    EXPECT_EQ(a.xs[1], 1);
    EXPECT_EQ(a.xs[2], 120 * 24);
    EXPECT_EQ(a.xs[3], 12 * 24);
    EXPECT_EQ(a.xs[4], 24);
    EXPECT_EQ(a.ys[1], 1);
    EXPECT_EQ(a.ys[4], 48);
    EXPECT_EQ(launch->gridY, 3u); // groups x one depth tap
    EXPECT_EQ(a.partialElems, 3 * 9 * 16 * 8);
    // dw K T R S C (channels-last graph): channels contiguous.
    EXPECT_EQ(launch->reduce.wk, 8 * 9);
    EXPECT_EQ(launch->reduce.wc, 1);
    EXPECT_EQ(launch->reduce.wtap, 8);
    EXPECT_EQ(launch->reduce.kg, 16);
    EXPECT_EQ(launch->reduce.cg, 8);
    EXPECT_EQ(launch->reduce.elems, 48 * 8 * 9);
}

TEST(TestConvWrwLaunch, TheReduceWritesEitherFilterLayoutAndReflectsForConvolution)
{
    const auto kcrs = launchOf(ConfigFields{});
    ASSERT_TRUE(kcrs.has_value());
    EXPECT_EQ(kcrs->reduce.wk, 24 * 9);
    EXPECT_EQ(kcrs->reduce.wc, 9);
    EXPECT_EQ(kcrs->reduce.wtap, 1);
    EXPECT_EQ(kcrs->reduce.flip, 0);

    auto spec = specOf(baseShape());
    spec.w.strides = filterChannelsLast(spec.w.dims);
    const auto problem = parse(spec);
    ASSERT_TRUE(problem.has_value());
    const auto krsc
        = conv_wrw::wrwLaunchFor(*problem, configOf(ConfigFields{}), deviceOf("gfx942"));
    ASSERT_TRUE(krsc.has_value());
    EXPECT_EQ(krsc->reduce.wk, 24 * 9);
    EXPECT_EQ(krsc->reduce.wc, 1);
    EXPECT_EQ(krsc->reduce.wtap, 24);
    // K C T R S of 2 to 64 taps at most 2 splits reduces by rows (a run of channels' taps,
    // contiguous in dw); more splits, K T R S C, one tap, or more than 64 taps by quads of
    // channels. 304 units give the base 6 splits; one unit gives it 2.
    EXPECT_EQ(kcrs->reduce.splits, 6);
    EXPECT_EQ(kcrs->reduce.rows, 0);
    EXPECT_EQ(krsc->reduce.rows, 0);
    EXPECT_EQ(krsc->reduceGrid, (9u * 40 * 6 + 63) / 64); // 9 taps x 40 k x 6 quads, 64 a block
    const auto twoSplits = launchOf(ConfigFields{}, baseShape(), "gfx942", 1);
    ASSERT_TRUE(twoSplits.has_value());
    EXPECT_EQ(twoSplits->reduce.splits, 2);
    EXPECT_EQ(twoSplits->reduce.rows, 1);
    EXPECT_EQ(twoSplits->reduceGrid, 40u); // 40 output channels x one run of 24 channels
    auto wide = baseShape();
    wide.c = 136; // three runs of 64 channels
    wide.k = 8;
    const auto runs = launchOf(rowsFields(), wide, "gfx942", 1);
    ASSERT_TRUE(runs.has_value());
    EXPECT_LE(runs->reduce.splits, 2);
    EXPECT_EQ(runs->reduce.rows, 1);
    EXPECT_EQ(runs->reduceGrid, 8u * 3);
    auto many = baseShape();
    many.input = {10, 12};
    many.filter = {9, 9}; // 81 taps
    many.prePadding = many.postPadding = {4, 4};
    auto f81 = rowsFields();
    f81.s = 9;
    f81.fr = 1;
    // Each refusal below at one unit (at most 2 splits), so only its own gate declines rows.
    const auto quadsOnly = launchOf(f81, many, "gfx942", 1);
    ASSERT_TRUE(quadsOnly.has_value());
    EXPECT_LE(quadsOnly->reduce.splits, 2);
    EXPECT_EQ(quadsOnly->reduce.rows, 0);
    auto pointwise = baseShape();
    pointwise.filter = {1, 1};
    pointwise.prePadding = pointwise.postPadding = {0, 0};
    auto f1 = rowsFields();
    f1.s = 1;
    f1.fr = 1;
    const auto oneTap = launchOf(f1, pointwise, "gfx942", 1);
    ASSERT_TRUE(oneTap.has_value());
    EXPECT_LE(oneTap->reduce.splits, 2);
    EXPECT_EQ(oneTap->reduce.rows, 0);
    const auto krscFew
        = conv_wrw::wrwLaunchFor(*problem, configOf(ConfigFields{}), deviceOf("gfx942", 1));
    ASSERT_TRUE(krscFew.has_value());
    EXPECT_EQ(krscFew->reduce.splits, 2);
    EXPECT_EQ(krscFew->reduce.rows, 0);

    auto convolution = baseShape();
    convolution.mode = ConvMode::CONVOLUTION;
    const auto flipped = launchOf(ConfigFields{}, convolution);
    ASSERT_TRUE(flipped.has_value());
    EXPECT_EQ(flipped->reduce.flip, 1);
}

TEST(TestConvWrwLaunch, OneSplitIntoContiguousChannelsWritesDwWithoutAReduce)
{
    // The launch of @p fields on @p shape with dw packed K T R S C.
    const auto krscLaunch = [](const ConfigFields& fields,
                               const ConvShape& shape,
                               const std::string& arch,
                               int units) -> std::optional<conv_wrw::WrwLaunch> {
        auto spec = specOf(shape);
        spec.w.strides = filterChannelsLast(spec.w.dims);
        const auto problem = parse(spec);
        if(!problem.has_value())
        {
            return std::nullopt;
        }
        return conv_wrw::wrwLaunchFor(*problem, configOf(fields), deviceOf(arch, units));
    };
    // bpu 1 on one unit: one split.
    auto one = ConfigFields{};
    one.bpu = 1;

    // K T R S C at one split: the kernel writes dw (channels contiguous, a tap cg apart).
    const auto krsc = krscLaunch(one, baseShape(), "gfx942", 1);
    ASSERT_TRUE(krsc.has_value());
    EXPECT_EQ(krsc->args.splits, 1);
    EXPECT_TRUE(krsc->dwStore);
    EXPECT_EQ(krsc->args.dwStore, 1);
    EXPECT_EQ(krsc->workspaceBytes, 0u);
    EXPECT_EQ(krsc->args.dwk, 24 * 9);
    EXPECT_EQ(krsc->args.dwtap, 24);
    EXPECT_EQ(krsc->args.flip, 0);

    // A config compiled without the direct store (dws 0): partials and the reduce.
    auto noDws = one;
    noDws.dws = 0;
    const auto krscNoDws = krscLaunch(noDws, baseShape(), "gfx942", 1);
    ASSERT_TRUE(krscNoDws.has_value());
    EXPECT_EQ(krscNoDws->args.splits, 1);
    EXPECT_FALSE(krscNoDws->dwStore);
    EXPECT_EQ(krscNoDws->args.dwStore, 0);
    EXPECT_EQ(krscNoDws->workspaceBytes, size_t{9} * 40 * 24 * 4);

    // K C T R S of several taps at one split: partials and the reduce.
    const auto kcrs = launchOf(one, baseShape(), "gfx942", 1);
    ASSERT_TRUE(kcrs.has_value());
    EXPECT_EQ(kcrs->args.splits, 1);
    EXPECT_FALSE(kcrs->dwStore);
    EXPECT_EQ(kcrs->args.dwStore, 0);
    EXPECT_EQ(kcrs->workspaceBytes, size_t{9} * 40 * 24 * 4);

    // One tap, K C T R S: its channels are contiguous too.
    auto pointwise = baseShape();
    pointwise.filter = {1, 1};
    pointwise.prePadding = pointwise.postPadding = {0, 0};
    auto f1 = rowsFields();
    f1.s = 1;
    f1.fr = 1;
    f1.bpu = 1;
    const auto oneTap = launchOf(f1, pointwise, "gfx942", 1);
    ASSERT_TRUE(oneTap.has_value());
    EXPECT_EQ(oneTap->args.splits, 1);
    EXPECT_TRUE(oneTap->dwStore);
    EXPECT_EQ(oneTap->workspaceBytes, 0u);
    EXPECT_EQ(oneTap->args.dwk, 24);
    EXPECT_EQ(oneTap->args.dwtap, 1);

    // Two splits: partials, whatever the layout.
    auto two = ConfigFields{};
    two.bpu = 2;
    const auto krscTwo = krscLaunch(two, baseShape(), "gfx942", 1);
    ASSERT_TRUE(krscTwo.has_value());
    EXPECT_EQ(krscTwo->args.splits, 2);
    EXPECT_FALSE(krscTwo->dwStore);
    EXPECT_EQ(krscTwo->args.dwStore, 0);
    EXPECT_EQ(krscTwo->workspaceBytes, size_t{2} * 9 * 40 * 24 * 4);

    // Loop 5 writes partials at one split too.
    const auto direct = krscLaunch(directFields(), channelsLastShape(), "gfx950", 1);
    ASSERT_TRUE(direct.has_value());
    EXPECT_EQ(direct->args.splits, 1);
    EXPECT_FALSE(direct->dwStore);
    EXPECT_EQ(direct->args.dwStore, 0);
    EXPECT_GT(direct->workspaceBytes, 0u);

    // A convolution reflects the taps the kernel writes.
    auto convolution = baseShape();
    convolution.mode = ConvMode::CONVOLUTION;
    const auto flipped = krscLaunch(one, convolution, "gfx942", 1);
    ASSERT_TRUE(flipped.has_value());
    EXPECT_TRUE(flipped->dwStore);
    EXPECT_EQ(flipped->args.flip, 1);
}

TEST(TestConvWrwLaunch, OneAndTwoDimensionalGraphsPadTheLeadingAxes)
{
    auto shape = baseShape();
    shape.input = {37};
    shape.filter = {7};
    shape.prePadding = shape.postPadding = {3};
    shape.stride = {1};
    shape.dilation = {1};
    auto f = ConfigFields{};
    f.fr = 1;
    f.s = 7;
    const auto launch = launchOf(f, shape);
    ASSERT_TRUE(launch.has_value());
    const auto& a = launch->args;
    EXPECT_EQ(a.di, 1);
    EXPECT_EQ(a.hi, 1);
    EXPECT_EQ(a.dout, 1);
    EXPECT_EQ(a.ho, 1);
    EXPECT_EQ(a.ft, 1);
    EXPECT_EQ(a.fr, 1);
    EXPECT_EQ(a.sd, 1);
    EXPECT_EQ(a.sh, 1);
    EXPECT_EQ(a.dd, 1);
    EXPECT_EQ(a.dh, 1);
    EXPECT_EQ(a.pd, 0);
    EXPECT_EQ(a.ph, 0);
    EXPECT_EQ(a.chunks, 2);
    EXPECT_EQ(a.trackLen, 1);
    EXPECT_EQ(launch->reduce.taps, 7);
}

TEST(TestConvWrwLaunch, TheSplitPlannerFillsTheDeviceWithinItsCaps)
{
    // bpu blocks per dispatch unit over the grid's blocks per split.
    EXPECT_EQ(conv_wrw::splitsWanted(4, 1, 1 << 20, 1000, 304), 76);
    EXPECT_EQ(conv_wrw::splitsWanted(4, 2, 1 << 20, 1000, 304), 152);
    EXPECT_EQ(conv_wrw::splitsWanted(4, 4, 1 << 20, 1000, 304), 304);
    // At most steps / 4 (and at least one).
    EXPECT_EQ(conv_wrw::splitsWanted(1, 2, 40, 1000, 304), 10);
    EXPECT_EQ(conv_wrw::splitsWanted(1, 2, 3, 1000, 304), 1);
    // At most 65535.
    EXPECT_EQ(conv_wrw::splitsWanted(1, 2, int64_t{1} << 40, 1, 1 << 20), 65535);
    // The workspace within 1 GiB: 2^26 partials (256 MiB) per split allow 4.
    EXPECT_EQ(conv_wrw::splitsWanted(1, 2, 1 << 20, int64_t{1} << 26, 304), 4);
    // A single split past 1 GiB still runs as one.
    EXPECT_EQ(conv_wrw::splitsWanted(1, 2, 1 << 20, int64_t{1} << 29, 304), 1);
    // A grid that fills the device alone is not split.
    EXPECT_EQ(conv_wrw::splitsWanted(10000, 2, 1 << 20, 1000, 304), 1);
    // Whole rounds: 64 blocks at 2 per unit want 10 splits, 640 blocks, 3 rounds of 304 for
    // 2.1 of work; 9 splits run 576 blocks in 2.
    EXPECT_EQ(conv_wrw::splitsWanted(64, 2, 1 << 20, 1000, 304), 9);
    // At 4 per unit: 19 splits, 1216 blocks, exactly 4 rounds.
    EXPECT_EQ(conv_wrw::splitsWanted(64, 4, 1 << 20, 1000, 304), 19);
    // An even fill keeps the wanted count: 3 blocks on 6 units at 2 per unit want 4 splits,
    // 12 blocks in 2 whole rounds.
    EXPECT_EQ(conv_wrw::splitsWanted(3, 2, 1 << 20, 1000, 6), 4);
}

TEST(TestConvWrwLaunch, TheConfigsBpuSetsTheSplits)
{
    // One block per split, 24 steps (cap 6): bpu 1 wants 304 per split, so the cap rules; a
    // 1-unit device wants bpu splits.
    for(const int64_t bpu : {1, 2, 4})
    {
        auto f = ConfigFields{};
        f.bpu = bpu;
        const auto launch = launchOf(f, baseShape(), "gfx942", 1);
        ASSERT_TRUE(launch.has_value());
        EXPECT_EQ(launch->args.splits, bpu);
    }
}

TEST(TestConvWrwLaunch, SplitsCoverEveryStepOnce)
{
    for(const int units : {1, 7, 80, 304})
    {
        const auto launch = launchOf(ConfigFields{}, baseShape(), "gfx942", units);
        ASSERT_TRUE(launch.has_value());
        const auto& a = launch->args;
        EXPECT_GE(a.splits * a.stepsPerSplit, a.steps);
        EXPECT_LT((a.splits - 1) * a.stepsPerSplit, a.steps);
        EXPECT_EQ(launch->workspaceBytes, size_t(a.splits) * size_t(a.partialElems) * 4);
    }
}

// ---------------------------------------------------------------------------
// LDS: the kernel's arithmetic
// ---------------------------------------------------------------------------

TEST(TestConvWrwLoadPasses, SmallCChannelsLastStagesXAnElementAUnit)
{
    // Small-C rows, 64 x 4 tile, 64 positions, 3 filter rows of 3 taps, fp16, 256 threads.
    // Channels-first: dy 64 rows x 8 vectors (2 passes), x 3 rows x 4 channels x 9 vectors.
    auto f = smallcFields();
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(f), 2, 64), 2);
    // Channels-last: dy 32 position pairs x 8 vectors (1 pass); x one element a unit: 3 rows x
    // 72 window positions (66 rounded to vectors) x 4 channels = 864 (4 passes).
    f.layout = 1;
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(f), 2, 64), 4);
}

TEST(TestConvWrwLoadPasses, PackedRowsStageAnElementAUnit)
{
    // 64 x 64 tile, 32 positions, fp16, 256 threads: dy 64 x 32 elements (8 passes); x one
    // filter row of 64 channels by the window's 40 positions (34 rounded to vectors): 10.
    auto f = packedFields();
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(f), 2, 64), 10);
    f.fr = 3;
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(f), 2, 64), 30);
}

TEST(TestConvWrwLaunch, PackedRowsStepOverRowGroupsOfAPlane)
{
    // wo 12 (+ 2 taps' reach): virtual rows of 14, 2 a step; 10 output rows in 5 groups.
    auto shape = baseShape();
    shape.n = 3;
    const auto launch = launchOf(packedFields(), shape);
    ASSERT_TRUE(launch.has_value());
    const auto& a = launch->args;
    EXPECT_EQ(a.segs, 14);
    EXPECT_EQ(a.segRows, 2);
    EXPECT_EQ(a.chunks, 5);
    EXPECT_EQ(a.steps, 3 * 1 * 5);
    EXPECT_EQ(launch->gridY, 3u); // 3 filter rows, one a block
    // Width stride 2: wo 6, virtual rows of 6 + 1 (the taps' reach in output columns), 4 a
    // step; 5 output rows in 2 groups.
    auto s2 = baseShape();
    s2.stride = {2, 2};
    auto f = packedFields();
    f.sw = 2;
    const auto strided = launchOf(f, s2);
    ASSERT_TRUE(strided.has_value());
    EXPECT_EQ(strided->args.segs, 7);
    EXPECT_EQ(strided->args.segRows, 4);
    EXPECT_EQ(strided->args.chunks, 2);
    EXPECT_EQ(strided->args.steps, 2 * 1 * 2);
    // Other loops leave the fields zero.
    const auto rows = launchOf(rowsFields(), shape);
    ASSERT_TRUE(rows.has_value());
    EXPECT_EQ(rows->args.segs, 0);
    EXPECT_EQ(rows->args.segRows, 0);
}

TEST(TestConvWrwKernelMatch, TheGenericLoopServesStridedOperandsAlone)
{
    // Padded x rows: no specialized config (packed layouts only), the generic loop on elements.
    const GraphFixture fixture(buildGraph(rowPaddedSpec()), deviceOf("gfx942"));
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto served = [&](const ConfigFields& f) {
        return matchesKernel(PACK, fixture.context(), kernelOf(f, "HALF"), *bound);
    };
    EXPECT_TRUE(served(genericFields()));
    EXPECT_FALSE(served(genericClFields()));
    EXPECT_FALSE(served(ConfigFields{}));
    EXPECT_FALSE(served(rowsFields()));
    // The specialized classes never serve a strided problem.
    auto withSpec = genericFields();
    withSpec.spec = "4R0.3.1.1.0;4P0.3.1.1.3";
    EXPECT_TRUE(served(withSpec));
    // A generic kernel without its spec metadata is declined.
    auto kernel = kernelOf(genericFields(), "HALF");
    kernel.metadata.erase("spec");
    EXPECT_FALSE(matchesKernel(PACK, fixture.context(), kernel, *bound));
    // Elements take any channel count (no whole vectors), in either packed layout too.
    auto odd = baseShape();
    odd.c = 5;
    odd.k = 7;
    EXPECT_TRUE(admits(genericFields(), "gfx942", odd));
    odd.channelsLast = true;
    EXPECT_TRUE(admits(genericFields(), "gfx942", odd));
    EXPECT_FALSE(admits(genericClFields(), "gfx942", odd));
    // Images smaller than a vector (one channel of 5 positions).
    auto tiny = baseShape();
    tiny.c = 1;
    tiny.k = 2;
    tiny.input = {1, 5};
    tiny.filter = {1, 3};
    tiny.prePadding = tiny.postPadding = {0, 0};
    EXPECT_FALSE(admits(rowsFields(), "gfx942", tiny));
    EXPECT_TRUE(admits(genericFields(), "gfx942", tiny));
}

TEST(TestConvWrwLaunch, TheGenericLoopReadsTheGraphsStridesAndSpans)
{
    const auto problem = parse(rowPaddedSpec());
    ASSERT_TRUE(problem.has_value());
    const auto launch
        = conv_wrw::wrwLaunchFor(*problem, configOf(genericFields()), deviceOf("gfx942"));
    ASSERT_TRUE(launch.has_value());
    const auto& a = launch->args;
    // 40 / 64 x 24 / 64 tiles; 9 taps in groups of 3.
    EXPECT_EQ(launch->gridX, 1u);
    EXPECT_EQ(launch->gridY, 3u);
    EXPECT_EQ(launch->block, 256u);
    // x as given (the missing leading axis 0), dy packed; buffers span the strides.
    EXPECT_EQ(a.xs[0], 24 * 10 * 16);
    EXPECT_EQ(a.xs[1], 160);
    EXPECT_EQ(a.xs[3], 16);
    EXPECT_EQ(a.xs[4], 1);
    EXPECT_EQ(a.ys[1], 120);
    EXPECT_EQ(a.xImage, 23 * 160 + 9 * 16 + 11 + 1);
    EXPECT_EQ(a.yImage, 40 * 10 * 12);
    // Output rows of one 32-position chunk: steps n x ho x chunks, no tracks.
    EXPECT_EQ(a.chunks, 1);
    EXPECT_EQ(a.steps, 2 * 10);
    EXPECT_EQ(a.trackLen, 0);
    // dy rows padded to 16 columns: its span too.
    auto dyPadded = specOf(baseShape());
    dyPadded.y.strides = {40 * 10 * 16, 160, 16, 1};
    const auto dyProblem = parse(dyPadded);
    ASSERT_TRUE(dyProblem.has_value());
    const auto dyLaunch
        = conv_wrw::wrwLaunchFor(*dyProblem, configOf(genericFields()), deviceOf("gfx942"));
    ASSERT_TRUE(dyLaunch.has_value());
    EXPECT_EQ(dyLaunch->args.ys[3], 16);
    EXPECT_EQ(dyLaunch->args.yImage, 39 * 160 + 9 * 16 + 11 + 1);
    EXPECT_EQ(dyLaunch->args.xImage, 24 * 10 * 12);

    // 3-D, groups: the grid rows are groups x ceil(taps / fr).
    auto threeD = baseShape();
    threeD.groups = 2;
    threeD.c = 32;
    threeD.k = 32;
    threeD.input = {4, 6, 7};
    threeD.filter = {1, 5, 5};
    threeD.prePadding = threeD.postPadding = {0, 2, 2};
    threeD.stride = {1, 1, 1};
    threeD.dilation = {1, 1, 1};
    auto f = genericFields();
    f.fr = 4;
    const auto grouped = launchOf(f, threeD);
    ASSERT_TRUE(grouped.has_value());
    EXPECT_EQ(grouped->gridY, 2u * 7u);
    EXPECT_EQ(grouped->args.steps, 2 * 4 * 6);
}

TEST(TestConvWrwKernelMatch, TheGenericLoopLeavesWidelyDilatedSmallFiltersToBlas)
{
    const auto dilated = [](std::vector<int64_t> filter, std::vector<int64_t> dilation) {
        auto shape = baseShape();
        shape.input = {30, 64};
        shape.filter = std::move(filter);
        shape.dilation = std::move(dilation);
        shape.prePadding = shape.postPadding = {0, 0};
        return shape;
    };
    // 3x3 (9 taps): dilation 5 admits, 6 declines, on either filter axis.
    EXPECT_TRUE(admits(genericFields(), "gfx942", dilated({3, 3}, {5, 5})));
    EXPECT_FALSE(admits(genericFields(), "gfx942", dilated({3, 3}, {6, 6})));
    EXPECT_FALSE(admits(genericFields(), "gfx942", dilated({3, 3}, {1, 6})));
    EXPECT_FALSE(admits(genericFields(), "gfx942", dilated({3, 3}, {6, 1})));
    // A dilation along a one-tap axis reaches nothing.
    EXPECT_TRUE(admits(genericFields(), "gfx942", dilated({1, 3}, {6, 1})));
    // 10 taps dilated 6 admit; 9 do not.
    EXPECT_TRUE(admits(genericFields(), "gfx942", dilated({1, 10}, {1, 6})));
    EXPECT_TRUE(admits(genericFields(), "gfx942", dilated({2, 5}, {6, 1})));
    EXPECT_FALSE(admits(genericFields(), "gfx942", dilated({1, 9}, {1, 6})));
}

TEST(TestConvWrwLaunch, TheGenericWindowGroupsWidthTapsPerFilterRow)
{
    // 3x3 at 2 taps a block: per filter row ceil(3 / 2) groups, 3 rows (the gathers: 5).
    auto f = genericWindowFields();
    f.fr = 2;
    const auto window = launchOf(f, channelsLastShape());
    ASSERT_TRUE(window.has_value());
    EXPECT_EQ(window->gridY, 3u * 2u);
    auto gathers = genericClFields();
    gathers.fr = 2;
    const auto perTap = launchOf(gathers, channelsLastShape());
    ASSERT_TRUE(perTap.has_value());
    EXPECT_EQ(perTap->gridY, 5u);
    // The window covers the taps a block holds: 3 taps a block on a 2-wide filter dilated 8
    // reach 8 past the first, within a window of 8 (2 x 8 = 16 would not be). 5 x 2 = 10 taps,
    // past the ASPP-class gate.
    auto narrow = baseShape();
    narrow.channelsLast = true;
    narrow.filter = {5, 2};
    narrow.dilation = {1, 8};
    narrow.prePadding = narrow.postPadding = {2, 4};
    EXPECT_TRUE(admits(genericWindowFields(), "gfx942", narrow));
}

TEST(TestConvWrwKernelMatch, TheGenericFmaFormServesFp32WithoutAMatrixCore)
{
    const auto f = genericFmaFields();
    // fp32 on RDNA; not where a matrix atom takes the element (fp32 on CDNA, 16-bit on RDNA).
    EXPECT_TRUE(admits(f, "gfx1201", floatShape()));
    EXPECT_TRUE(admits(f, "gfx1100", floatShape()));
    EXPECT_FALSE(admits(f, "gfx942", floatShape()));
    EXPECT_FALSE(admits(f, "gfx1201", baseShape()));
    auto half = f; // 16-bit without a matrix core: 8-element vectors
    half.vec = 8;
    EXPECT_TRUE(admits(half, "gfx906", baseShape()));
    // The FMA thread tiles' gates: whole quads, no waves.
    auto pairs = f;
    pairs.tk = 2;
    EXPECT_FALSE(admits(pairs, "gfx1201", floatShape()));
    auto waves = f;
    waves.wk = 1;
    EXPECT_FALSE(admits(waves, "gfx1201", floatShape()));
    // Stems (fewer than 8 input channels a group) and width strides of 4 or more: BLAS's.
    auto stem = floatShape();
    stem.c = 7;
    EXPECT_FALSE(admits(f, "gfx1201", stem));
    stem.c = 8;
    EXPECT_TRUE(admits(f, "gfx1201", stem));
    auto patch = floatShape();
    patch.input = {20, 24};
    patch.stride = {1, 4};
    EXPECT_FALSE(admits(f, "gfx1201", patch));
    patch.stride = {1, 3};
    EXPECT_TRUE(admits(f, "gfx1201", patch));
    // Rows that are not whole vectors (the FMA loops' channels-first gate is theirs alone).
    auto width13 = floatShape();
    width13.input = {10, 13};
    EXPECT_TRUE(admits(f, "gfx1201", width13));
    // The FMA loops' classes decline it: rows of vectors (F), of scalar units (S), the ring (Q).
    auto served = f;
    served.spec = "16F0.3.1.1.0";
    EXPECT_FALSE(admits(served, "gfx1201", floatShape()));
    served.spec = "16S0.3.1.1.0";
    EXPECT_FALSE(admits(served, "gfx1201", floatShape()));
    served.spec = "16Q0.3.1.1.3";
    EXPECT_FALSE(admits(served, "gfx1201", floatShape()));
    // Its loads and LDS: 256 threads, 16 positions: 16 channel rows a pass, 64 / 16 + 5 x 64 /
    // 16 loads; [position][channel] tiles of 64 + 4 channels.
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(f), 4, 0), 4 + 20);
    EXPECT_EQ(conv_wrw::ldsBytesOf(configOf(f), 4), 2 * (16 * 68 + 5 * 16 * 68) * 4);
    auto window = f;
    window.layout = 1;
    window.dw = 8;
    window.bq = 32;
    EXPECT_EQ(conv_wrw::ldsBytesOf(configOf(window), 4), 2 * (32 * 68 + 40 * 68) * 4);
}

TEST(TestConvWrwKernelMatch, TheGenericLoopTakesEveryArchsMatrixAtom)
{
    auto gfx1201 = genericClFields();
    gfx1201.atom = "wmma12_f16";
    gfx1201.atomId = 14;
    gfx1201.wk = 4;
    EXPECT_TRUE(admits(gfx1201, "gfx1201", channelsLastShape()));
    EXPECT_FALSE(admits(gfx1201, "gfx942", channelsLastShape()));
    EXPECT_TRUE(admits(genericGfx950Fields(), "gfx950", channelsLastShape()));
    auto fp32 = genericFields();
    fp32.atom = "mfma32_f32";
    fp32.atomId = 7;
    fp32.vec = 4;
    fp32.bq = 16;
    EXPECT_TRUE(admits(fp32, "gfx90a", floatShape()));
    // The FMA "atom" has no generic loop.
    auto fmaGeneric = fmaFields();
    fmaGeneric.loop = 8;
    fmaGeneric.s = fmaGeneric.sw = fmaGeneric.dw = 0;
    EXPECT_FALSE(admits(fmaGeneric, "gfx942", floatShape()));
}

TEST(TestConvWrwSpec, KeysServeTheirClassOnly)
{
    struct Case
    {
        const char* key;
        ConvShape shape;
        const char* arch;
        bool served;
    };
    const auto s2 = [] {
        auto shape = baseShape();
        shape.stride = {2, 2};
        return shape;
    }();
    const auto dilatedH = [] {
        auto shape = baseShape();
        shape.dilation = {2, 1};
        shape.prePadding = shape.postPadding = {2, 1};
        return shape;
    }();
    const auto bf16 = [] {
        auto shape = baseShape();
        shape.dataType = DataType::BFLOAT16;
        return shape;
    }();
    auto nhwcGfx950 = channelsLastShape();
    const std::vector<Case> cases{
        // Rows: any filter height and height stride.
        {"4R0.3.1.1.0", baseShape(), "gfx942", true},
        {"4R0.3.1.1.0", s2, "gfx942", false}, // width stride 2
        {"4R0.3.2.1.0", s2, "gfx942", true},
        {"4R0.3.1.1.0", heightStride2Shape(), "gfx942", true},
        // The width class: filter width, stride, dilation.
        {"4R0.5.1.1.0", baseShape(), "gfx942", false},
        {"4R0.3.1.2.0", baseShape(), "gfx942", false},
        // The atom: on the device, of the graph's element type.
        {"10R0.3.1.1.0", baseShape(), "gfx942", false},
        {"10R0.3.1.1.0", baseShape(), "gfx950", true},
        {"5R0.3.1.1.0", baseShape(), "gfx942", false},
        {"5R0.3.1.1.0", bf16, "gfx942", true},
        // The layout: packed as the key's; channels-last whole vectors.
        {"4R1.3.1.1.0", baseShape(), "gfx942", false},
        {"4R1.3.1.1.0", channelsLastShape(), "gfx942", true},
        {"4R0.3.1.1.0", channelsLastShape(), "gfx942", false},
        {"4R1.3.1.1.0", unalignedKShape(), "gfx942", false},
        {"10R2.3.1.1.0", nhwcGfx950, "gfx950", true},
        {"4R2.3.1.1.0", channelsLastShape(), "gfx942", false},
        // Small-C rows: at most 16 input channels a group, any count of them channels-last.
        {"0C0.3.1.1.0", stemShape(), "gfx942", true},
        {"0C0.3.1.1.0", baseShape(), "gfx942", false},
        {"0C1.3.1.1.0",
         [] {
             auto shape = stemShape();
             shape.channelsLast = true;
             return shape;
         }(),
         "gfx942",
         true},
        // Rings: the filter height at height stride 1, or (loop 1) its phases.
        {"4P0.3.1.1.3", baseShape(), "gfx942", true},
        {"4P0.3.1.1.1", baseShape(), "gfx942", false},
        {"4P0.3.1.1.2", heightStride2Shape(), "gfx942", true},
        {"4P0.3.1.1.3", heightStride2Shape(), "gfx942", false},
        {"4P0.3.1.1.2", dilatedH, "gfx942", false},
        {"4P0.3.1.1.2",
         [] {
             auto shape = baseShape();
             shape.stride = {2, 1};
             shape.dilation = {2, 1};
             shape.prePadding = shape.postPadding = {2, 1};
             return shape;
         }(),
         "gfx942",
         false}, // phases need unit height dilation
        // The ring at align 2: channels-last output channels in any count.
        {"4U1.3.1.1.3", unalignedKShape(), "gfx942", true},
        {"4P1.3.1.1.3", unalignedKShape(), "gfx942", false},
        {"4U1.3.1.1.3",
         [] {
             auto shape = unalignedKShape();
             shape.c = 12;
             return shape;
         }(),
         "gfx942",
         false},
        {"10G0.3.1.1.3", baseShape(), "gfx950", true},
        {"10G0.3.1.1.3", baseShape(), "gfx942", false},
        {"10G0.3.1.1.2", heightStride2Shape(), "gfx950", false},
        {"4G0.3.1.1.3", baseShape(), "gfx942", false}, // an atom gfx942 runs: gfx950 only
        {"0D2.3.1.1.3", channelsLastShape(), "gfx942", false},
        {"0D1.3.1.1.3", channelsLastShape(), "gfx942", false},
        {"0G1.3.1.1.3", channelsLastShape(), "gfx942", false},
        // The direct loop: also unit height dilation (or one filter row).
        {"8D2.3.1.1.3", nhwcGfx950, "gfx950", true},
        {"8D2.3.1.1.3",
         [] {
             auto shape = channelsLastShape();
             shape.dilation = {2, 1};
             shape.prePadding = shape.postPadding = {2, 1};
             return shape;
         }(),
         "gfx950",
         false},
        {"8D2.3.1.1.1",
         [] {
             auto shape = channelsLastShape();
             shape.filter = {1, 3};
             shape.dilation = {2, 1};
             shape.prePadding = shape.postPadding = {0, 1};
             return shape;
         }(),
         "gfx950",
         true},
        {"8D2.3.1.1.3", nhwcGfx950, "gfx942", false},
        // The FMA loops (atom 16): fp32 everywhere, 16-bit where no matrix atom takes it;
        // channels-first vectors need rows of whole vectors, scalar units (S) any.
        {"16F0.3.1.1.0", floatShape(), "gfx1201", true},
        {"16F0.3.1.1.0", floatShape(), "gfx942", true},
        {"16F0.3.1.1.0", alignedShape(), "gfx1201", false}, // 16-bit: the matrix atoms'
        {"16F0.3.1.1.0", alignedShape(), "gfx906", true}, // rows of 16: whole 8-vectors
        {"16F0.3.1.1.0",
         [] {
             auto shape = floatShape();
             shape.input = {10, 13};
             return shape;
         }(),
         "gfx1201",
         false},
        {"16S0.3.1.1.0",
         [] {
             auto shape = floatShape();
             shape.input = {10, 13};
             return shape;
         }(),
         "gfx1201",
         true},
        {"16Q0.3.1.1.3", floatShape(), "gfx1201", true},
        {"16Q0.3.1.1.3",
         [] {
             auto shape = floatShape();
             shape.stride = {2, 1};
             return shape;
         }(),
         "gfx1201",
         false},
        {"16S1.3.1.1.0",
         [] {
             auto shape = floatShape();
             shape.channelsLast = true;
             shape.c = 5;
             return shape;
         }(),
         "gfx1201",
         true},
        {"16F1.3.1.1.0",
         [] {
             auto shape = floatShape();
             shape.channelsLast = true;
             shape.c = 5;
             return shape;
         }(),
         "gfx1201",
         false},
    };
    for(const auto& c : cases)
    {
        const auto problem = parse(specOf(c.shape));
        ASSERT_TRUE(problem.has_value()) << c.key;
        const auto served = conv_wrw::specServes(*problem, c.key, c.arch);
        ASSERT_TRUE(served.has_value()) << c.key;
        EXPECT_EQ(*served, c.served) << c.key << " on " << c.arch;
    }
    // Images past the buffer range: no key's class (block-based buffers serve them).
    auto large = baseShape();
    large.n = 1;
    large.c = 8;
    large.k = 8;
    large.input = {1, int64_t{1} << 28}; // x 2^31 bytes
    large.filter = {1, 3};
    large.prePadding = large.postPadding = {0, 1};
    const auto largeProblem = parse(specOf(large));
    ASSERT_TRUE(largeProblem.has_value());
    ASSERT_TRUE(largeProblem->largeImage);
    const auto largeServed = conv_wrw::specServes(*largeProblem, "4R0.3.1.1.0", "gfx942");
    ASSERT_TRUE(largeServed.has_value());
    EXPECT_FALSE(*largeServed);
    // Their block-based buffer forms (lower case) serve them alone: channels-first, each
    // channel plane within the range.
    const auto served = [&](const conv_wrw::WrwProblem& problem, const char* key) {
        const auto result = conv_wrw::specServes(problem, key, "gfx942");
        return result.has_value() && *result;
    };
    EXPECT_TRUE(served(*largeProblem, "4r0.3.1.1.0"));
    EXPECT_TRUE(served(*largeProblem, "4p0.3.1.1.1"));
    EXPECT_TRUE(served(*largeProblem, "0c0.3.1.1.0"));
    const auto small = parse(specOf(baseShape()));
    ASSERT_TRUE(small.has_value());
    EXPECT_FALSE(served(*small, "4r0.3.1.1.0"));
    auto largeCl = large;
    largeCl.channelsLast = true;
    const auto largeClProblem = parse(specOf(largeCl));
    ASSERT_TRUE(largeClProblem.has_value());
    EXPECT_FALSE(served(*largeClProblem, "4r1.3.1.1.0"));
    // One 2^30-element x plane (2^31 bytes), and one such dy plane.
    auto xPlane = large; // width stride 2: the dy plane half as wide, within the range
    xPlane.c = 1;
    xPlane.input = {1, int64_t{1} << 30};
    xPlane.stride = {1, 2};
    const auto xPlaneProblem = parse(specOf(xPlane));
    ASSERT_TRUE(xPlaneProblem.has_value());
    EXPECT_FALSE(served(*xPlaneProblem, "4r0.3.2.1.0"));
    auto yPlane = large;
    yPlane.c = 1;
    yPlane.input = {1, int64_t{1} << 29};
    yPlane.prePadding = yPlane.postPadding = {0, (int64_t{1} << 28) + 1}; // dy 2^30 wide
    const auto yPlaneProblem = parse(specOf(yPlane));
    ASSERT_TRUE(yPlaneProblem.has_value());
    EXPECT_FALSE(served(*yPlaneProblem, "4r0.3.1.1.0"));
}

TEST(TestConvWrwSpec, MalformedKeysAreNotClasses)
{
    const auto problem = parse(specOf(baseShape()));
    ASSERT_TRUE(problem.has_value());
    for(const char* key : {"",
                           "R0.3.1.1.0",
                           "16R0.3.1.1.0", // fma
                           "17R0.3.1.1.0",
                           "-1R0.3.1.1.0",
                           "4X0.3.1.1.0",
                           "4R3.3.1.1.0",
                           "4R.3.1.1.0",
                           "4R0.3.1.1",
                           "4R0.3.1.1.0.0",
                           "4R0.3.1.1.0x",
                           "4R0,3.1.1.0",
                           "4R0.0.1.1.0",
                           "4R0.3.0.1.0",
                           "4R0.3.1.0.0",
                           "4R0.3.1.1.1", // rows carry no height
                           "4C0.3.1.1.1",
                           "4P0.3.1.1.0", // rings do
                           "4U1.3.1.1.0",
                           "4G0.3.1.1.0",
                           "4D0.3.1.1.0",
                           // Block-based buffer forms exist for rows, rings, small-C only.
                           "4g0.3.1.1.3",
                           "4u1.3.1.1.3",
                           "8d2.3.1.1.3",
                           "4x0.3.1.1.0",
                           // FMA kinds take the fma atom alone, and it no other kind.
                           "4F0.3.1.1.0",
                           "4S0.3.1.1.0",
                           "4Q0.3.1.1.3",
                           "16P0.3.1.1.3",
                           "16Q0.3.1.1.0"})
    {
        EXPECT_FALSE(conv_wrw::specServes(*problem, key, "gfx942").has_value()) << key;
    }
}

TEST(TestConvWrwLoadPasses, TheGenericLoopCountsItsLoads)
{
    // Channels-first: 8 channel rows a pass, 64 / 8 + 3 x 64 / 8.
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(genericFields()), 2, 64), 8 + 24);
    // Channel vectors: 16 words of 2 positions; dy 8 x 16 units, x 3 x 8 x 16 over 256
    // threads, 2 loads a unit.
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(genericClFields()), 2, 64), (1 + 2) * 2);
    // fp32: one position a word.
    auto fp32 = genericClFields();
    fp32.vec = 4;
    fp32.bq = 16;
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(fp32), 4, 64), 1 + 3);
    // The shared window: dy as above; x 8 vectors by 20 words of the 40-position window.
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(genericWindowFields()), 2, 64), (1 + 1) * 2);
    // bq not dividing the block.
    auto odd = genericFields();
    odd.bq = 48;
    EXPECT_EQ(conv_wrw::loadPassesOf(configOf(odd), 2, 64), std::numeric_limits<int64_t>::max());
}

TEST(TestConvWrwLds, MirrorsTheKernelsArithmetic)
{
    // Generic taps: two buffers of 64 dy and 3 x 64 x rows of 32 + 8 positions.
    EXPECT_EQ(conv_wrw::ldsBytesOf(configOf(genericFields()), 2), 2 * (64 + 3 * 64) * 40 * 2);
    // Shared window: one x row of 32 + 8 positions per channel, pitched to 56 (7 16-byte units,
    // odd); 48 positions (6 units, even) would be bumped too.
    EXPECT_EQ(conv_wrw::ldsBytesOf(configOf(genericWindowFields()), 2),
              2 * (64 * 40 + 64 * 56) * 2);
    auto wide = genericWindowFields();
    wide.dw = 16; // 48 + 8 = 56: already odd
    EXPECT_EQ(conv_wrw::ldsBytesOf(configOf(wide), 2), 2 * (64 * 40 + 64 * 56) * 2);
    // Ring, channels-first fp16, 64 x 64 x 32, 3 taps: window 34 = 5 vectors of 8, LDA 40,
    // LDB ceil(42 / 8) x 8 + 8 = 56; two buffers of 64 x 40 + 64 x 56.
    auto ring = configOf(ConfigFields{});
    EXPECT_EQ(conv_wrw::ldsBytesOf(ring, 2), 2 * (64 * 40 + 64 * 56) * 2);
    // Aligned units: one more vector of window (6), LDB 64, and an 8-element guard.
    auto aligned = ring;
    aligned.align = 1;
    EXPECT_EQ(conv_wrw::ldsBytesOf(aligned, 2), 2 * (64 * 40 + 8 + 64 * 64) * 2);
    // Channels-last: the window's 34 positions (words of 2), LDB 48.
    auto channelsLast = ring;
    channelsLast.layout = 1;
    EXPECT_EQ(conv_wrw::ldsBytesOf(channelsLast, 2), 2 * (64 * 40 + 64 * 48) * 2);
    // Rows, 3 filter rows at width stride 2: window 65 = 9 vectors, 36 per phase, LDB 48;
    // x tile 3 rows x 2 phases x 64 channels.
    auto rows = ring;
    rows.loop = 0;
    rows.sw = 2;
    EXPECT_EQ(conv_wrw::ldsBytesOf(rows, 2), 2 * (64 * 40 + 3 * 2 * 64 * 48) * 2);
    // Packed rows: loop 0's tiles, fr filter rows.
    auto packed = rows;
    packed.loop = 7;
    EXPECT_EQ(conv_wrw::ldsBytesOf(packed, 2), 2 * (64 * 40 + 3 * 2 * 64 * 48) * 2);
    packed.fr = 1;
    EXPECT_EQ(conv_wrw::ldsBytesOf(packed, 2), 2 * (64 * 40 + 2 * 64 * 48) * 2);
    // fp32, 64 x 64 x 16, 4-element vectors: window 18 = 5 vectors (20), LDA 20, LDB
    // ceil(22 / 4) x 4 + 4 = 28.
    auto fp32 = configOf(floatFields());
    EXPECT_EQ(conv_wrw::ldsBytesOf(fp32, 4), 2 * (64 * 20 + 64 * 28) * 4);
    // DMA ring, 2 stages, 4 waves: dy 64 rows of 5 units (5 groups of 64, 2 passes a wave),
    // x 64 rows of 5 (odd) units (2 passes): 4 passes x 4 waves x 64 16-byte units a slot.
    auto dma = configOf(dmaFields());
    EXPECT_EQ(conv_wrw::ldsBytesOf(dma, 2), 2 * 4 * 4 * 64 * 16);
    dma.ns = 4;
    EXPECT_EQ(conv_wrw::ldsBytesOf(dma, 2), 4 * 4 * 4 * 64 * 16);
    // Transpose reads (ring): [position][channel] rows of pitch 72 (64 + 8), the dy chunk's 32
    // positions and the window's 34.
    auto tr = configOf(trFields());
    EXPECT_EQ(conv_wrw::ldsBytesOf(tr, 2), 2 * (32 * 72 + 34 * 72) * 2);
    // Transpose reads (DMA ring): 32 x 9 dy units (5 groups) and 34 x 9 x units (5 groups),
    // 2 passes each over 4 waves.
    auto dmaTr = configOf(dmaTrFields());
    EXPECT_EQ(conv_wrw::ldsBytesOf(dmaTr, 2), 2 * 4 * 4 * 64 * 16);
    // FMA ring: R + 1 = 4 dy rows of 32 positions x (32 + 4) and two x windows of
    // (ceil(34 / 4) + 1) x 4 = 40 positions x (64 + 4), fp32.
    auto fma = configOf(fmaFields());
    EXPECT_EQ(conv_wrw::ldsBytesOf(fma, 4), (4 * 32 * 36 + 2 * 40 * 68) * 4);
    // FMA output rows: two buffers of the dy chunk (32 x 36) and 3 windows of
    // (ceil(34 / 4) + 1) x 4 = 40 positions x 68, fp32; scalar units take no extra vector.
    auto fmaRows = fmaFields();
    fmaRows.loop = 4;
    EXPECT_EQ(conv_wrw::ldsBytesOf(configOf(fmaRows), 4), 2 * (32 * 36 + 3 * 40 * 68) * 4);
    fmaRows.vec = 1;
    EXPECT_EQ(conv_wrw::ldsBytesOf(configOf(fmaRows), 4), 2 * (32 * 36 + 3 * 34 * 68) * 4);
    // Direct loop, 64 x 64 by 2 items: 3 slots of 2 items' x rows (34 columns rounded to 40, a
    // whole 512-element round) and dy rows (32 x 64), plus a drain round; the waves_q sum
    // stages 1 item x 4 waves x 3 taps x 2 x 2 fragments of 64 float4.
    EXPECT_EQ(conv_wrw::ldsBytesOf(configOf(directFields()), 2),
              3 * (2 * (40 * 64 + 32 * 64) + 512) * 2 + 1 * 4 * 3 * 2 * 2 * 64 * 16);
}

TEST(TestConvWrwDefines, EveryFieldIsADefine)
{
    auto f = dmaFields();
    const auto defines
        = conv_wrw::compileDefinesFor(DataType::BFLOAT16, configOf(f), "K1AtomMfma32Bf16X16");
    ASSERT_EQ(defines.size(), 21u);
    EXPECT_EQ(defines[0], (std::pair<std::string, std::string>{"CONV_K3_T", "unsigned short"}));
    EXPECT_EQ(defines[1],
              (std::pair<std::string, std::string>{"CONV_K3_ATOM", "K1AtomMfma32Bf16X16"}));
    EXPECT_EQ(defines[2], (std::pair<std::string, std::string>{"CONV_K3_LOOP", "2"}));
    EXPECT_EQ(defines[13], (std::pair<std::string, std::string>{"CONV_K3_TK", "0"}));
    EXPECT_EQ(defines[18], (std::pair<std::string, std::string>{"CONV_K3_NS", "2"}));
    EXPECT_EQ(defines[19], (std::pair<std::string, std::string>{"CONV_K3_WQ", "0"}));
    EXPECT_EQ(defines[20], (std::pair<std::string, std::string>{"CONV_K3_DWS", "1"}));
    const auto direct
        = conv_wrw::compileDefinesFor(DataType::HALF, configOf(directFields()), "K1AtomMfmaF16X32");
    ASSERT_EQ(direct.size(), 21u);
    EXPECT_EQ(direct[2], (std::pair<std::string, std::string>{"CONV_K3_LOOP", "5"}));
    EXPECT_EQ(direct[19], (std::pair<std::string, std::string>{"CONV_K3_WQ", "2"}));
    EXPECT_EQ(direct[20], (std::pair<std::string, std::string>{"CONV_K3_DWS", "0"}));
    // The FMA ring names no atom.
    const auto fma = conv_wrw::compileDefinesFor(DataType::FLOAT, configOf(fmaFields()), "");
    ASSERT_EQ(fma.size(), 20u);
    EXPECT_EQ(fma[0], (std::pair<std::string, std::string>{"CONV_K3_T", "float"}));
    EXPECT_EQ(fma[1], (std::pair<std::string, std::string>{"CONV_K3_LOOP", "3"}));
    EXPECT_EQ(fma[12], (std::pair<std::string, std::string>{"CONV_K3_TK", "4"}));
    // Block-based buffers: CONV_K3_PB is defined only when set.
    auto pb = rowsFields();
    pb.pb = 1;
    const auto blocks
        = conv_wrw::compileDefinesFor(DataType::HALF, configOf(pb), "K1AtomMfma32F16");
    ASSERT_EQ(blocks.size(), 22u);
    EXPECT_EQ(blocks[21], (std::pair<std::string, std::string>{"CONV_K3_PB", "1"}));
    EXPECT_THROW(static_cast<void>(conv_wrw::compileDefinesFor(DataType::DOUBLE, configOf(f), "")),
                 std::invalid_argument);
}

TEST(TestConvWrwComputeType, ServesCudnnsComputeTypesPerElementType)
{
    expectCudnnComputeTypes(baseShape(),
                            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
                            [](const ConvShape& s) { return parse(specOf(s)).has_value(); });
}

using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

constexpr const char* TF32_ENV = "HIP_KERNEL_PROVIDER_CONV_TF32";

/// An fp32 graph three-bf16 is admitted for on CDNA (threeBf16Pays): 128 input and 64 output
/// channels, 3x3, unit stride.
ConvShape x3Shape()
{
    auto shape = floatShape();
    shape.c = 128;
    shape.k = 64;
    return shape;
}

TEST(TestConvWrwThreeBf16, OnlyWithTf32PermissionAndOnlyForFloatGraphs)
{
    {
        const ScopedEnvironmentVariableSetter off(TF32_ENV, "");
        EXPECT_FALSE(admits(threeBf16Fields(), "gfx942", x3Shape()));
    }
    {
        const ScopedEnvironmentVariableSetter no(TF32_ENV, "0");
        EXPECT_FALSE(admits(threeBf16Fields(), "gfx942", x3Shape()));
    }
    const ScopedEnvironmentVariableSetter on(TF32_ENV, "1");
    EXPECT_TRUE(admits(threeBf16Fields(), "gfx942", x3Shape()));
    EXPECT_TRUE(admits(threeBf16Fields(), "gfx950", x3Shape()));
    // gfx90a: three bf16 products never beat its exact fp32 MFMA here (threeBf16Pays).
    EXPECT_FALSE(admits(threeBf16Fields(), "gfx90a", x3Shape()));
    // 16-bit graphs take the bf16 configs themselves, never their three-bf16 form.
    auto bf16 = x3Shape();
    bf16.dataType = DataType::BFLOAT16;
    EXPECT_FALSE(admits(threeBf16Fields(), "gfx942", bf16));
    EXPECT_FALSE(admits(threeBf16Fields(), "gfx942", baseShape()));
    // The atom runs on the device: not gfx908 (no 1k bf16 atom), not RDNA (wave64 MFMA).
    EXPECT_FALSE(admits(threeBf16Fields(), "gfx908", x3Shape()));
    EXPECT_FALSE(admits(threeBf16Fields(), "gfx1100", x3Shape()));
}

/// Every field three-bf16 adds, one refusal each, from a config admitted under permission.
TEST(TestConvWrwThreeBf16, RefusesOneFieldAtATime)
{
    const ScopedEnvironmentVariableSetter on(TF32_ENV, "1");
    const auto shape = x3Shape();
    ASSERT_TRUE(admits(threeBf16Fields(), "gfx942", shape));
    const auto refuses = [&](const std::function<void(ConfigFields&)>& mutate) {
        auto f = threeBf16Fields();
        mutate(f);
        return !admits(f, "gfx942", shape);
    };
    EXPECT_TRUE(refuses([](ConfigFields& f) { f.math = 2; }));
    EXPECT_TRUE(refuses([](ConfigFields& f) { f.math = -1; }));
    // dw is written in fp32 by the reduce; the direct store would write bf16.
    EXPECT_TRUE(refuses([](ConfigFields& f) { f.dws = 1; }));
    // A 16-bit atom other than bf16.
    EXPECT_TRUE(refuses([](ConfigFields& f) {
        f.atom = "mfma32_f16";
        f.atomId = 4;
    }));
    // The exact fp32 atom (whose vector would be 4).
    EXPECT_TRUE(refuses([](ConfigFields& f) {
        f.atom = "mfma32_f32";
        f.atomId = 7;
        f.vec = 4;
        f.bq = 16;
    }));
    // The bf16 kernel's vector: 8 elements of 2 bytes.
    EXPECT_TRUE(refuses([](ConfigFields& f) { f.vec = 4; }));
    // The FMA ring.
    EXPECT_TRUE(refuses([](ConfigFields& f) {
        f.atom = "fma";
        f.atomId = 16;
        f.loop = 3;
        f.wk = 0;
        f.wc = 0;
        f.tk = 4;
        f.tc = 4;
    }));
    // The math-0 twin of the config is a bf16 config: it does not serve an fp32 graph.
    EXPECT_TRUE(refuses([](ConfigFields& f) { f.math = 0; }));
}

TEST(TestConvWrwThreeBf16, LaunchesTheBf16ConfigOnThreeTimesTheImagesAfterTheSplit)
{
    const ScopedEnvironmentVariableSetter on(TF32_ENV, "1");
    const auto x3 = launchOf(threeBf16Fields(), x3Shape());
    ASSERT_TRUE(x3.has_value());
    // The bf16 config on the bf16 problem of 6 images.
    auto plainFields = threeBf16Fields();
    plainFields.math = 0;
    auto tripled = x3Shape();
    tripled.dataType = DataType::BFLOAT16;
    tripled.n = 6;
    const auto plain = launchOf(plainFields, tripled);
    ASSERT_TRUE(plain.has_value());
    EXPECT_EQ(x3->args.n, 6);
    EXPECT_EQ(x3->args.steps, plain->args.steps);
    EXPECT_EQ(x3->args.splits, plain->args.splits);
    EXPECT_EQ(x3->args.xImage, plain->args.xImage);
    EXPECT_EQ(x3->gridX, plain->gridX);
    EXPECT_EQ(x3->gridZ, plain->gridZ);
    EXPECT_EQ(x3->kernelType, "K1AtomMfma32Bf16");
    EXPECT_FALSE(x3->dwStore);
    // x 2 x 128 x 120 and dy 2 x 64 x 120 elements, three bf16 copies each, 256-byte aligned;
    // the partials after them.
    EXPECT_TRUE(x3->split);
    EXPECT_EQ(x3->splitArgs.xElems, 2 * 128 * 120);
    EXPECT_EQ(x3->splitArgs.yElems, 2 * 64 * 120);
    EXPECT_EQ(x3->splitYOffset, size_t{3} * 2 * 2 * 128 * 120);
    EXPECT_EQ(x3->splitBytes, size_t{3} * 2 * 2 * (128 + 64) * 120);
    EXPECT_EQ(x3->workspaceBytes, x3->splitBytes + plain->workspaceBytes);
    EXPECT_EQ(x3->splitGrid, (2u * 128 * 120 + 255) / 256);
    EXPECT_FALSE(plain->split);
}

/// The classes three-bf16 is admitted for (threeBf16Pays), one refusal per gate.
TEST(TestConvWrwThreeBf16, PaysOnlyWhereItMeasuredFaster)
{
    const auto pays = [](const ConvShape& shape, const char* arch) {
        const auto problem = parse(specOf(shape));
        return problem.has_value() && conv_wrw::threeBf16Pays(*problem, arch);
    };
    EXPECT_TRUE(pays(x3Shape(), "gfx942"));
    EXPECT_TRUE(pays(x3Shape(), "gfx950:sramecc+:xnack-"));
    EXPECT_FALSE(pays(x3Shape(), "gfx90a"));
    EXPECT_FALSE(pays(x3Shape(), "gfx908"));
    auto narrowC = x3Shape();
    narrowC.c = 64;
    EXPECT_FALSE(pays(narrowC, "gfx950"));
    auto narrowK = x3Shape();
    narrowK.k = 32;
    EXPECT_FALSE(pays(narrowK, "gfx950"));
    auto pointwise = x3Shape();
    pointwise.filter = {1, 1};
    pointwise.prePadding = pointwise.postPadding = {0, 0};
    EXPECT_FALSE(pays(pointwise, "gfx950"));
    auto strided = x3Shape();
    strided.stride = {2, 2};
    EXPECT_FALSE(pays(strided, "gfx950"));
    // RDNA (exact fp32 on FMA): stems of few channels, or 256+ channels with 9+ taps.
    auto stem = x3Shape();
    stem.c = 3;
    EXPECT_TRUE(pays(stem, "gfx1201"));
    EXPECT_TRUE(pays(stem, "gfx1100"));
    EXPECT_FALSE(pays(x3Shape(), "gfx1201"));
    auto wide = x3Shape();
    wide.c = 256;
    EXPECT_TRUE(pays(wide, "gfx1201"));
    auto widePointwise = pointwise;
    widePointwise.c = 256;
    EXPECT_FALSE(pays(widePointwise, "gfx1201"));
}

TEST(TestConvWrwThreeBf16, CompilesTheBf16KernelWithTheSplitAndFp32Dw)
{
    const auto config = configOf(threeBf16Fields());
    const auto defines = conv_wrw::compileDefinesFor(DataType::FLOAT, config, "K1AtomMfma32Bf16");
    const auto valueOf = [&](const std::string& name) -> std::string {
        for(const auto& [key, value] : defines)
        {
            if(key == name)
            {
                return value;
            }
        }
        return "<unset>";
    };
    EXPECT_EQ(valueOf("CONV_K3_T"), "unsigned short");
    EXPECT_EQ(valueOf("CONV_K3_TO"), "float");
    EXPECT_EQ(valueOf("CONV_K3_MATH"), "1");
    // Math 0 defines neither, so every other config's code is unchanged.
    for(const auto dataType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        auto plain = config;
        plain.math = 0;
        for(const auto& [key, value] :
            conv_wrw::compileDefinesFor(dataType, plain, "K1AtomMfma32Bf16"))
        {
            EXPECT_NE(key, "CONV_K3_TO");
            EXPECT_NE(key, "CONV_K3_MATH");
        }
    }
}

} // namespace

#endif
