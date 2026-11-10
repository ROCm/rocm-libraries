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
 * @brief Host-only checks of the small-input-channel forward convolution engine
 *        (CONV_SMALLC_FWD): what the graph match accepts and refuses, what the kernel match admits
 *        per config and device (atom, ship list, dtype, channel bound, filter, stride, dilation),
 *        one decline per malformed config field, and the plan's tile counts, grid and formatted
 *        weights. No device is used.
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
                       {"wn", f.wn}};
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
    };
    for(const auto& [name, mutate] : mutations)
    {
        Fields f;
        mutate(f);
        EXPECT_FALSE(admits(f)) << name;
    }
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

} // namespace

#endif
