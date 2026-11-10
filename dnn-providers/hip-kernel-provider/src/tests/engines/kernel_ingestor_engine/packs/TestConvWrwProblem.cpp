// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

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
        // x and dy: the same packed layout.
        {"XChannelsLastDyChannelsFirst",
         [](ConvGraphSpec& s) { s.x.strides = packedStrides(s.x.dims, true); }},
        {"DyChannelsLastXChannelsFirst",
         [](ConvGraphSpec& s) { s.y.strides = packedStrides(s.y.dims, true); }},
        {"XRowPadded",
         [](ConvGraphSpec& s) {
             s.x.strides = {24 * 10 * 16, 10 * 16, 16, 1}; // rows 16 apart, 12 used
         }},
        {"DyChannelPadded",
         [](ConvGraphSpec& s) {
             s.y.strides = {41 * 120, 120, 12, 1}; // 40 channels in an image of 41
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
        // One image of x and of dy within 2^31 bytes.
        {"XImageBeyondOffsets",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.n = 1;
             shape.c = 1;
             shape.k = 8;
             shape.input = {1, int64_t{1} << 30}; // fp16: 2^31 bytes
             shape.filter = {1, 1};
             shape.prePadding = shape.postPadding = {0, 0};
             shape.stride = {1, int64_t{1} << 20};
             reshape(s, shape);
         }},
        {"DyImageBeyondOffsets",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.n = 1;
             shape.c = 8;
             shape.k = int64_t{1} << 27;
             shape.input = {1, 8}; // dy 2^27 x 8 fp16: 2^31 bytes
             shape.filter = {1, 1};
             shape.prePadding = shape.postPadding = {0, 0};
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
    int64_t tk = 0, tc = 0, xcd = 0, align = 0, ns = 0, bpu = 2, wq = 0;
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
                       {"wq", f.wq}};
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
        {"LoopUnknown", [](ConfigFields& f) { f.loop = 7; }},
        {"SmallCWcTwo", [](ConfigFields& f) { f.wc = 2; }, smallcFields, "gfx942", stemShape},
        {"SmallCBc32", [](ConfigFields& f) { f.bc = 32; }, smallcFields, "gfx942", stemShape},
        {"WqOffTheDirectLoop", [](ConfigFields& f) { f.wq = 1; }},
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

TEST(TestConvWrwKernelMatch, SmallCRowsServeFewChannelsFirstChannels)
{
    // Three input channels, channels-first: served; 24 (past 16) and channels-last are not.
    EXPECT_TRUE(admits(smallcFields(), "gfx942", stemShape()));
    EXPECT_FALSE(admits(smallcFields(), "gfx942", baseShape()));
    auto many = stemShape();
    many.c = 17;
    EXPECT_FALSE(admits(smallcFields(), "gfx942", many));
    many.c = 16;
    EXPECT_TRUE(admits(smallcFields(), "gfx942", many));
    auto cl = stemShape();
    cl.c = 8;
    cl.channelsLast = true;
    auto f = smallcFields();
    f.layout = 1;
    EXPECT_FALSE(admits(f, "gfx942", cl));
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

TEST(TestConvWrwLds, MirrorsTheKernelsArithmetic)
{
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
    ASSERT_EQ(defines.size(), 20u);
    EXPECT_EQ(defines[0], (std::pair<std::string, std::string>{"CONV_K3_T", "unsigned short"}));
    EXPECT_EQ(defines[1],
              (std::pair<std::string, std::string>{"CONV_K3_ATOM", "K1AtomMfma32Bf16X16"}));
    EXPECT_EQ(defines[2], (std::pair<std::string, std::string>{"CONV_K3_LOOP", "2"}));
    EXPECT_EQ(defines[13], (std::pair<std::string, std::string>{"CONV_K3_TK", "0"}));
    EXPECT_EQ(defines[18], (std::pair<std::string, std::string>{"CONV_K3_NS", "2"}));
    EXPECT_EQ(defines[19], (std::pair<std::string, std::string>{"CONV_K3_WQ", "0"}));
    const auto direct
        = conv_wrw::compileDefinesFor(DataType::HALF, configOf(directFields()), "K1AtomMfmaF16X32");
    ASSERT_EQ(direct.size(), 20u);
    EXPECT_EQ(direct[2], (std::pair<std::string, std::string>{"CONV_K3_LOOP", "5"}));
    EXPECT_EQ(direct[19], (std::pair<std::string, std::string>{"CONV_K3_WQ", "2"}));
    // The FMA ring names no atom.
    const auto fma = conv_wrw::compileDefinesFor(DataType::FLOAT, configOf(fmaFields()), "");
    ASSERT_EQ(fma.size(), 19u);
    EXPECT_EQ(fma[0], (std::pair<std::string, std::string>{"CONV_K3_T", "float"}));
    EXPECT_EQ(fma[1], (std::pair<std::string, std::string>{"CONV_K3_LOOP", "3"}));
    EXPECT_EQ(fma[12], (std::pair<std::string, std::string>{"CONV_K3_TK", "4"}));
    EXPECT_THROW(static_cast<void>(conv_wrw::compileDefinesFor(DataType::DOUBLE, configOf(f), "")),
                 std::invalid_argument);
}

} // namespace

#endif
