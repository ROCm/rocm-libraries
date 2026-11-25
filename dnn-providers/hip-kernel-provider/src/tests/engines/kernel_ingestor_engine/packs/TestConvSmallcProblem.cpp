// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvSmallcProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvSmallcProblem.cpp
 * @brief Host-only checks of the small-input-channel convolution engines (CONV_SMALLC_FWD and
 *        CONV_SMALLC_BWD): what the graph matches accept and refuse, what the kernel match admits
 *        per config and device (direction, atom, ship list, dtype, channel bound, filter, stride,
 *        dilation, epilogue vs output layout), one decline per malformed config field, the
 *        backward-data mapping onto the forward kernel, and the plan's tile counts, grid and
 *        formatted weights. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

constexpr PackSymbols PACK{"",
                           "hipkernel.conv_smallc_fwd.graph_match",
                           "",
                           "hipkernel.conv_smallc.kernel_match",
                           "",
                           "hipkernel.conv_smallc.dispatch",
                           "",
                           "",
                           ""};

constexpr PackSymbols BWD_PACK{"",
                               "hipkernel.conv_smallc_bwd.graph_match",
                               "",
                               "hipkernel.conv_smallc.kernel_match",
                               "",
                               "hipkernel.conv_smallc.dispatch",
                               "",
                               "",
                               ""};

/// A ResNet-style stem: 3 channels, 7x7, stride 2, padding 3, NCHW fp16.
ConvShape stemShape()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 3;
    shape.k = 40;
    shape.input = {37, 29};
    shape.filter = {7, 7};
    shape.prePadding = {3, 3};
    shape.postPadding = {3, 3};
    shape.stride = {2, 2};
    shape.dilation = {1, 1};
    shape.dataType = DataType::HALF;
    return shape;
}

struct Fields
{
    std::string atom = "wmma11_f16";
    std::string archs = "gfx11";
    std::string dtype = "HALF";
    std::string direction = "FWD";
    int64_t atomId = 6, wave = 32, ka = 16, cmax = 3;
    int64_t kd = 1, r = 7, s = 7, sd = 1, sh = 2, sw = 2, dd = 1, dh = 1, dw = 1;
    int64_t fm = 4, fn = 2, wm = 2, wq = 2, wn = 2;
    int64_t ep = 0;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const Fields& f)
{
    auto kernel = makeKernel(0, f.dtype, "ConvSmallc");
    kernel.source.sourceFile = "kernels/ConvSmallc.cpp";
    kernel.metadata = {{"direction", f.direction},
                       {"dtype", f.dtype},
                       {"atom", f.atom},
                       {"archs", f.archs},
                       {"atom_id", f.atomId},
                       {"wave", f.wave},
                       {"ka", f.ka},
                       {"cmax", f.cmax},
                       {"kd", f.kd},
                       {"r", f.r},
                       {"s", f.s},
                       {"sd", f.sd},
                       {"sh", f.sh},
                       {"sw", f.sw},
                       {"dd", f.dd},
                       {"dh", f.dh},
                       {"dw", f.dw},
                       {"fm", f.fm},
                       {"fn", f.fn},
                       {"wm", f.wm},
                       {"wq", f.wq},
                       {"wn", f.wn},
                       {"ep", f.ep}};
    return kernel;
}

bool graphAccepts(const ConvShape& shape)
{
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)));
    return matchesGraph(PACK, fixture.context()).has_value();
}

bool admits(const Fields& fields,
            const std::string& arch = "gfx1100",
            const ConvShape& shape = stemShape())
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)), properties);
    const auto bound = matchesGraph(PACK, fixture.context());
    return bound.has_value() && matchesKernel(PACK, fixture.context(), kernelOf(fields), *bound);
}

/// A VAE / U-Net output layer's backward data: dx of 72 channels from dy of 3, 3x3, stride 1,
/// padding 1, NHWC fp16.
ConvShape outputLayerShape()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 72;
    shape.k = 3;
    shape.input = {21, 35};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    shape.stride = {1, 1};
    shape.dilation = {1, 1};
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    return shape;
}

/// A backward config for outputLayerShape on gfx11.
Fields bwdFields()
{
    Fields f;
    f.direction = "BWD";
    f.r = 3;
    f.s = 3;
    f.sh = 1;
    f.sw = 1;
    return f;
}

bool bwdGraphAccepts(const ConvShape& shape)
{
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_DATA)));
    return matchesGraph(BWD_PACK, fixture.context()).has_value();
}

bool bwdAdmits(const Fields& fields,
               const std::string& arch = "gfx1100",
               const ConvShape& shape = outputLayerShape())
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_DATA)),
                               properties);
    const auto bound = matchesGraph(BWD_PACK, fixture.context());
    return bound.has_value()
           && matchesKernel(BWD_PACK, fixture.context(), kernelOf(fields), *bound);
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

TEST(TestConvSmallcGraph, AcceptsForwardConvolutionsOfAnyLayoutAndRank)
{
    EXPECT_TRUE(graphAccepts(stemShape()));
    auto nhwc = stemShape();
    nhwc.channelsLast = true;
    EXPECT_TRUE(graphAccepts(nhwc));
    auto threeD = stemShape();
    threeD.input = {6, 30, 30};
    threeD.filter = {3, 7, 7};
    threeD.prePadding = threeD.postPadding = {1, 3, 3};
    threeD.stride = {1, 2, 2};
    threeD.dilation = {1, 1, 1};
    EXPECT_TRUE(graphAccepts(threeD));
    auto fp32 = stemShape();
    fp32.dataType = DataType::FLOAT;
    EXPECT_TRUE(graphAccepts(fp32));
}

TEST(TestConvSmallcGraph, RefusesGroupsAndOtherDirections)
{
    auto grouped = stemShape();
    grouped.c = 4;
    grouped.k = 8;
    grouped.groups = 2;
    EXPECT_FALSE(graphAccepts(grouped));
    for(const auto direction : {ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        const GraphFixture fixture(buildGraph(specFor(stemShape(), direction)));
        EXPECT_FALSE(matchesGraph(PACK, fixture.context()).has_value());
    }
}

TEST(TestConvSmallcGraph, RefusesAnImageBeyondTheBufferOffsets)
{
    auto huge = stemShape();
    huge.n = 1;
    huge.c = 3;
    huge.input = {16384, 32768}; // 3 GiB of fp16 in one image
    EXPECT_FALSE(graphAccepts(huge));
}

// ---------------------------------------------------------------------------
// Kernel match
// ---------------------------------------------------------------------------

TEST(TestConvSmallcKernel, AdmitsWhereItsAtomAndShipListAllow)
{
    const Fields f;
    EXPECT_TRUE(admits(f, "gfx1100"));
    EXPECT_TRUE(admits(f, "gfx1151"));
    EXPECT_FALSE(admits(f, "gfx1201"));
    EXPECT_FALSE(admits(f, "gfx942"));
    auto mfma = f;
    mfma.atom = "mfma16x32_f16";
    mfma.archs = "gfx950";
    mfma.atomId = 4;
    mfma.wave = 64;
    mfma.ka = 32;
    EXPECT_TRUE(admits(mfma, "gfx950:sramecc+:xnack-"));
    EXPECT_FALSE(admits(mfma, "gfx942"));
    auto exact = mfma;
    exact.atom = "mfma16_f32";
    exact.dtype = "FLOAT";
    exact.archs = "gfx942";
    exact.atomId = 3;
    exact.ka = 4;
    auto fp32 = stemShape();
    fp32.dataType = DataType::FLOAT;
    EXPECT_TRUE(admits(exact, "gfx942", fp32));
    EXPECT_FALSE(admits(exact, "gfx942"));
}

TEST(TestConvSmallcKernel, TheGraphMustFitTheChannelBoundAndGeometry)
{
    const Fields f;
    auto four = stemShape();
    four.c = 4;
    EXPECT_FALSE(admits(f, "gfx1100", four));
    auto two = stemShape();
    two.c = 2;
    EXPECT_TRUE(admits(f, "gfx1100", two));
    auto stride1 = stemShape();
    stride1.stride = {1, 1};
    EXPECT_FALSE(admits(f, "gfx1100", stride1));
    auto dilated = stemShape();
    dilated.dilation = {2, 2};
    EXPECT_FALSE(admits(f, "gfx1100", dilated));
    auto five = stemShape();
    five.filter = {5, 5};
    EXPECT_FALSE(admits(f, "gfx1100", five));
    auto bf = stemShape();
    bf.dataType = DataType::BFLOAT16;
    EXPECT_FALSE(admits(f, "gfx1100", bf));
}

TEST(TestConvSmallcKernel, DeclinesEachMalformedField)
{
    ASSERT_TRUE(admits(Fields{}));
    const std::vector<std::pair<std::string, std::function<void(Fields&)>>> mutations{
        {"direction", [](Fields& f) { f.direction = "BWD"; }},
        {"dtype", [](Fields& f) { f.dtype = "FLOAT"; }},
        {"atom", [](Fields& f) { f.atom = "mfma32_f16"; }},
        {"archs", [](Fields& f) { f.archs = ""; }},
        {"wave", [](Fields& f) { f.wave = 64; }},
        {"ka", [](Fields& f) { f.ka = 32; }},
        {"cmax", [](Fields& f) { f.cmax = 0; }},
        {"kd", [](Fields& f) { f.kd = 0; }},
        {"r", [](Fields& f) { f.r = 0; }},
        {"s", [](Fields& f) { f.s = 0; }},
        {"sd", [](Fields& f) { f.sd = 0; }},
        {"sh", [](Fields& f) { f.sh = 0; }},
        {"sw", [](Fields& f) { f.sw = 0; }},
        {"dd", [](Fields& f) { f.dd = 0; }},
        {"dh", [](Fields& f) { f.dh = 0; }},
        {"dw", [](Fields& f) { f.dw = 0; }},
        {"fm", [](Fields& f) { f.fm = 0; }},
        {"fn", [](Fields& f) { f.fn = 0; }},
        {"wm", [](Fields& f) { f.wm = 0; }},
        {"wq", [](Fields& f) { f.wq = 0; }},
        {"wn", [](Fields& f) { f.wn = 0; }},
        {"ep", [](Fields& f) { f.ep = 2; }},
        {"ep negative", [](Fields& f) { f.ep = -1; }},
    };
    for(const auto& [name, mutate] : mutations)
    {
        Fields f;
        mutate(f);
        EXPECT_FALSE(admits(f)) << name;
    }
}

TEST(TestConvSmallcKernel, FmaConfigsRunAnywhereWhenTheirThreadsCoverTheTile)
{
    Fields f;
    f.atom = "fma";
    f.archs = "any";
    f.dtype = "FLOAT";
    f.atomId = 0;
    f.wave = 0;
    f.ka = 1;
    f.fm = 4;
    f.fn = 8;
    f.wm = 4;
    f.wq = 2;
    f.wn = 4;
    auto fp32 = stemShape();
    fp32.dataType = DataType::FLOAT;
    EXPECT_TRUE(admits(f, "gfx1201", fp32));
    EXPECT_TRUE(admits(f, "gfx906", fp32));
    EXPECT_TRUE(admits(f, "gfx942", fp32));
    auto half = f;
    half.dtype = "HALF";
    EXPECT_TRUE(admits(half, "gfx1100"));
    // 256 threads must cover the tile exactly.
    auto under = f;
    under.fm = 2;
    EXPECT_FALSE(admits(under, "gfx1201", fp32));
    auto ragged = f;
    ragged.fn = 3;
    EXPECT_FALSE(admits(ragged, "gfx1201", fp32));
    auto wave = f;
    wave.wave = 32;
    EXPECT_FALSE(admits(wave, "gfx1201", fp32));
    // The FMA kernel has one epilogue (an output the register epilogue could take).
    auto fp32Nhwc = fp32;
    fp32Nhwc.channelsLast = true;
    EXPECT_TRUE(admits(f, "gfx1201", fp32Nhwc));
    auto registers = f;
    registers.ep = 1;
    EXPECT_FALSE(admits(registers, "gfx1201", fp32Nhwc));
}

TEST(TestConvSmallcKernel, TheRegisterEpilogueNeedsChannelsFastestOutput)
{
    Fields f;
    f.ep = 1;
    auto nhwc = stemShape();
    nhwc.channelsLast = true;
    EXPECT_TRUE(admits(f, "gfx1100", nhwc));
    EXPECT_FALSE(admits(f, "gfx1100", stemShape()));
    // The slot epilogue takes either layout.
    EXPECT_TRUE(admits(Fields{}, "gfx1100", nhwc));
    EXPECT_TRUE(admits(Fields{}, "gfx1100", stemShape()));
}

// ---------------------------------------------------------------------------
// Backward data
// ---------------------------------------------------------------------------

TEST(TestConvSmallcBwdGraph, AcceptsUnitStrideBackwardDataOfAnyLayoutAndRank)
{
    EXPECT_TRUE(bwdGraphAccepts(outputLayerShape()));
    auto nchw = outputLayerShape();
    nchw.channelsLast = false;
    EXPECT_TRUE(bwdGraphAccepts(nchw));
    auto threeD = outputLayerShape();
    threeD.input = {5, 18, 22};
    threeD.filter = {3, 3, 3};
    threeD.prePadding = threeD.postPadding = {1, 1, 1};
    threeD.stride = {1, 1, 1};
    threeD.dilation = {1, 1, 1};
    EXPECT_TRUE(bwdGraphAccepts(threeD));
    auto dilated = outputLayerShape();
    dilated.dilation = {2, 2};
    dilated.prePadding = dilated.postPadding = {2, 2};
    EXPECT_TRUE(bwdGraphAccepts(dilated));
}

TEST(TestConvSmallcBwdGraph, RefusesEachStridedAxisAndOtherDirections)
{
    auto strideH = outputLayerShape();
    strideH.stride = {2, 1};
    EXPECT_FALSE(bwdGraphAccepts(strideH));
    auto strideW = outputLayerShape();
    strideW.stride = {1, 2};
    EXPECT_FALSE(bwdGraphAccepts(strideW));
    auto strideD = outputLayerShape();
    strideD.input = {5, 18, 22};
    strideD.filter = {3, 3, 3};
    strideD.prePadding = strideD.postPadding = {1, 1, 1};
    strideD.stride = {2, 1, 1};
    strideD.dilation = {1, 1, 1};
    EXPECT_FALSE(bwdGraphAccepts(strideD));
    auto grouped = outputLayerShape();
    grouped.k = 4;
    grouped.groups = 2;
    EXPECT_FALSE(bwdGraphAccepts(grouped));
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_WEIGHTS})
    {
        const GraphFixture fixture(buildGraph(specFor(outputLayerShape(), direction)));
        EXPECT_FALSE(matchesGraph(BWD_PACK, fixture.context()).has_value());
    }
}

TEST(TestConvSmallcBwdGraph, RefusesADyImageBeyondTheBufferOffsets)
{
    // The window loads read dy: its image bounds the offsets (dx's may be larger).
    auto huge = outputLayerShape();
    huge.n = 1;
    huge.c = 1;
    huge.k = 3;
    huge.input = {16384, 32768}; // dy: 3 GiB of fp16 in one image
    EXPECT_FALSE(bwdGraphAccepts(huge));
    auto fits = huge;
    fits.k = 1; // dy: 1 GiB
    EXPECT_TRUE(bwdGraphAccepts(fits));
}

TEST(TestConvSmallcBwdKernel, TheDirectionAndTheDyChannelBoundMustMatch)
{
    EXPECT_TRUE(bwdAdmits(bwdFields()));
    // A forward config is not a backward one, and the reverse.
    auto forward = bwdFields();
    forward.direction = "FWD";
    EXPECT_FALSE(bwdAdmits(forward));
    auto fwdShape = outputLayerShape();
    fwdShape.c = 3;
    fwdShape.k = 72;
    EXPECT_TRUE(admits(forward, "gfx1100", fwdShape));
    EXPECT_FALSE(admits(bwdFields(), "gfx1100", fwdShape));
    // The channel bound is dy's channel count, not dx's.
    auto four = outputLayerShape();
    four.k = 4;
    EXPECT_FALSE(bwdAdmits(bwdFields(), "gfx1100", four));
    auto wide = bwdFields();
    wide.cmax = 4;
    EXPECT_TRUE(bwdAdmits(wide, "gfx1100", four));
    // Geometry and dtype as forward.
    auto five = outputLayerShape();
    five.filter = {5, 5};
    five.prePadding = five.postPadding = {2, 2};
    EXPECT_FALSE(bwdAdmits(bwdFields(), "gfx1100", five));
    auto bf = outputLayerShape();
    bf.dataType = DataType::BFLOAT16;
    EXPECT_FALSE(bwdAdmits(bwdFields(), "gfx1100", bf));
    // The register epilogue writes dx: channels-fastest dx only.
    auto registers = bwdFields();
    registers.ep = 1;
    EXPECT_TRUE(bwdAdmits(registers));
    auto nchw = outputLayerShape();
    nchw.channelsLast = false;
    EXPECT_FALSE(bwdAdmits(registers, "gfx1100", nchw));
    EXPECT_TRUE(bwdAdmits(bwdFields(), "gfx1100", nchw));
}

TEST(TestConvSmallcBwdPlan, DyIsTheKernelInputAndDxItsOutput)
{
    auto shape = outputLayerShape();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    shape.dilation = {1, 2};
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_DATA)));
    const auto problem
        = conv_smallc::parseSmallcProblem(fixture.context(), ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(problem->backward);
    const auto& args = problem->args;
    const auto dy = shape.output();
    EXPECT_EQ(args.c, 3);
    EXPECT_EQ(args.k, 72);
    EXPECT_EQ(args.hi, dy[0]);
    EXPECT_EQ(args.wi, dy[1]);
    EXPECT_EQ(args.ho, 21);
    EXPECT_EQ(args.wo, 35);
    EXPECT_EQ(args.di, 1);
    EXPECT_EQ(args.dout, 1);
    // (filter - 1) dilation - pre-padding: 2 - 2 = 0 rows, 4 - 0 = 4 columns.
    EXPECT_EQ(args.ph, 0);
    EXPECT_EQ(args.pw, 4);
    EXPECT_EQ(args.pd, 0);
    const auto& base = problem->base.args;
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(args.xs[i], base.yStrides[i]) << i;
        EXPECT_EQ(args.ys[i], base.xStrides[i]) << i;
    }
    // Pre-padding past the filter: a negative forward padding (dx row i reads dy row i + 1).
    auto past = outputLayerShape();
    past.prePadding = {3, 1};
    past.postPadding = {0, 1};
    const GraphFixture pastFixture(buildGraph(specFor(past, ConvDirection::BACKWARD_DATA)));
    const auto pastProblem
        = conv_smallc::parseSmallcProblem(pastFixture.context(), ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(pastProblem.has_value());
    EXPECT_EQ(pastProblem->args.ph, -1);
    // The plan tiles dx and blocks dx's channels.
    const auto config = conv_smallc::configOf(kernelOf(bwdFields()));
    ASSERT_TRUE(config.has_value());
    const GraphFixture plain(buildGraph(specFor(outputLayerShape(), ConvDirection::BACKWARD_DATA)));
    const auto plainProblem
        = conv_smallc::parseSmallcProblem(plain.context(), ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(plainProblem.has_value());
    const auto plan = conv_smallc::planFor(*plainProblem, *config);
    ASSERT_TRUE(plan.has_value());
    // dx 21 x 35: tiles of 8 rows x 32 columns; 72 channels -> two 64-column blocks.
    EXPECT_EQ(plan->args.tilesH, 3);
    EXPECT_EQ(plan->args.tilesW, 2);
    EXPECT_EQ(plan->args.kpad, 128);
    EXPECT_EQ(plan->grid[0], 3u * 2u * 2u);
    EXPECT_EQ(plan->grid[1], 2u);
    // 3 channels x 3 rows x 4 taps = 36 -> 3 steps of 16.
    EXPECT_EQ(plan->weightElems, 128 * 48);
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

TEST(TestConvSmallcPlan, TilesGridAndFormattedWeights)
{
    const GraphFixture fixture(buildGraph(specFor(stemShape(), ConvDirection::FORWARD)));
    const auto problem = conv_smallc::parseSmallcProblem(fixture.context());
    ASSERT_TRUE(problem.has_value());
    const auto config = conv_smallc::configOf(kernelOf(Fields{}));
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->sp(), 8);
    // 3 channels x 7 rows x 8 taps = 168 -> 11 steps of 16.
    EXPECT_EQ(config->kpl(), 176);
    const auto plan = conv_smallc::planFor(*problem, *config);
    ASSERT_TRUE(plan.has_value());
    // Output 19 x 15: tiles of 8 rows x 32 columns; K 40 -> one 64-column block.
    EXPECT_EQ(plan->args.tilesH, 3);
    EXPECT_EQ(plan->args.tilesW, 1);
    EXPECT_EQ(plan->args.kpad, 64);
    EXPECT_EQ(plan->grid[0], 3u * 1u * 2u);
    EXPECT_EQ(plan->grid[1], 1u);
    EXPECT_EQ(plan->weightElems, 64 * 176);
    EXPECT_EQ(plan->weightBytes, static_cast<size_t>(64 * 176 * 2));
}

TEST(TestConvSmallcComputeType, ServesCudnnsComputeTypesPerElementType)
{
    expectCudnnComputeTypes(stemShape(),
                            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
                            [](const ConvShape& s) { return graphAccepts(s); });
}

} // namespace

#endif
