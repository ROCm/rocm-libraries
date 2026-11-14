// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvGroupValuWrwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvGroupValuWrwProblem.cpp
 * @brief Host-only checks of CONV_GROUP_VALU_WRW: what the graph match accepts and refuses (one
 *        decline per gated field), what the kernel match admits per config (dtype, layout,
 *        filter class, tile divisibility, row width, dot2 rules), one decline per malformed
 *        config field, and the plan's lanes, load widths, splits and workspace. No device.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
namespace gvw = conv_group_valu_wrw;

PackSymbols pack()
{
    return {"",
            "hipkernel.conv_group_valu_wrw.graph_match",
            "",
            "hipkernel.conv_group_valu_wrw.kernel_match",
            "",
            "hipkernel.conv_group_valu_wrw.dispatch",
            "",
            "",
            ""};
}

/// A ResNeXt grouped layer: 32 groups of 4 channels, 3x3, padding 1, 56 x 56.
ConvShape grouped()
{
    ConvShape shape;
    shape.n = 8;
    shape.c = 128;
    shape.k = 128;
    shape.groups = 32;
    shape.input = {56, 56};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    shape.stride = {1, 1};
    shape.dilation = {1, 1};
    shape.dataType = DataType::HALF;
    return shape;
}

struct Fields
{
    std::string dtype = "HALF";
    std::string layout = "nchw";
    int64_t cb = 4, kb = 4, p = 8, fr = 3, fs = 3, sw = 1, dw = 1, pw = 1, nw = 2, dot2 = 0,
            ring = 0;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const Fields& f)
{
    auto kernel = makeKernel(0, f.dtype, "ConvGroupValuWrw");
    kernel.source.sourceFile = "kernels/ConvGroupValuWrw.cpp";
    kernel.metadata = {{"dtype", f.dtype},
                       {"layout", f.layout},
                       {"cb", f.cb},
                       {"kb", f.kb},
                       {"p", f.p},
                       {"fr", f.fr},
                       {"fs", f.fs},
                       {"sw", f.sw},
                       {"dw", f.dw},
                       {"pw", f.pw},
                       {"nw", f.nw},
                       {"dot2", f.dot2},
                       {"ring", f.ring}};
    return kernel;
}

bool graphAcceptsSpec(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesGraph(pack(), fixture.context()).has_value();
}

bool graphAccepts(const ConvShape& shape, DataType computeType = DataType::FLOAT)
{
    auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    spec.computeType = computeType;
    return graphAcceptsSpec(spec);
}

bool admitsSpec(const Fields& fields, const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(pack(), fixture.context());
    return bound.has_value() && matchesKernel(pack(), fixture.context(), kernelOf(fields), *bound);
}

bool admits(const Fields& fields, const ConvShape& shape = grouped())
{
    return admitsSpec(fields, specFor(shape, ConvDirection::BACKWARD_WEIGHTS));
}

std::optional<gvw::Plan>
    planOf(const ConvShape& shape, const Fields& fields, int wave = 64, int64_t units = 60)
{
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)));
    const auto problem = gvw::parseGvwProblem(fixture.context());
    const auto config = gvw::configOf(kernelOf(fields));
    if(!problem.has_value() || !config.has_value())
    {
        return std::nullopt;
    }
    return gvw::planFor(*problem, *config, wave, units);
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

TEST(TestConvGroupValuWrwGraph, AcceptsLayoutsDtypesRanksGroupsAndModes)
{
    EXPECT_TRUE(graphAccepts(grouped()));
    auto nhwc = grouped();
    nhwc.channelsLast = true;
    EXPECT_TRUE(graphAccepts(nhwc));
    for(const auto type : {DataType::FLOAT, DataType::BFLOAT16})
    {
        auto other = grouped();
        other.dataType = type;
        EXPECT_TRUE(graphAccepts(other));
    }
    auto dense = grouped();
    dense.groups = 1;
    EXPECT_TRUE(graphAccepts(dense));
    auto oneD = grouped();
    oneD.input = {300};
    oneD.filter = {3};
    oneD.prePadding = oneD.postPadding = {1};
    oneD.stride = {1};
    oneD.dilation = {1};
    EXPECT_TRUE(graphAccepts(oneD));
    auto convolution = grouped();
    convolution.mode = ConvMode::CONVOLUTION;
    EXPECT_TRUE(graphAccepts(convolution));
    EXPECT_TRUE(graphAccepts(grouped(), DataType::UNSET));
}

TEST(TestConvGroupValuWrwGraph, RefusesOtherDirections)
{
    const GraphFixture fwd(buildGraph(specFor(grouped(), ConvDirection::FORWARD)));
    EXPECT_FALSE(matchesGraph(pack(), fwd.context()).has_value());
    const GraphFixture bwd(buildGraph(specFor(grouped(), ConvDirection::BACKWARD_DATA)));
    EXPECT_FALSE(matchesGraph(pack(), bwd.context()).has_value());
}

TEST(TestConvGroupValuWrwGraph, RefusesAComputeTypeBelowFp32)
{
    EXPECT_FALSE(graphAccepts(grouped(), DataType::HALF));
    EXPECT_TRUE(graphAccepts(grouped(), DataType::FLOAT));
}

ConvShape threeD()
{
    auto shape = grouped();
    shape.input = {4, 14, 14};
    shape.filter = {3, 3, 3};
    shape.prePadding = shape.postPadding = {1, 1, 1};
    shape.stride = {1, 1, 1};
    shape.dilation = {1, 1, 1};
    return shape;
}

TEST(TestConvGroupValuWrwGraph, AcceptsThreeD)
{
    EXPECT_TRUE(graphAccepts(threeD()));
    auto nhwc = threeD();
    nhwc.channelsLast = true;
    EXPECT_TRUE(graphAccepts(nhwc));
}

TEST(TestConvGroupValuWrwPlan, ThreeDTilesPerDepthTapAndRowsOverTheOutputDepth)
{
    auto shape = threeD();
    shape.stride = {2, 1, 1};
    shape.dilation = {2, 1, 1};
    shape.prePadding = shape.postPadding = {2, 1, 1};
    shape.input = {9, 14, 14}; // output depth (9 + 4 - 5) / 2 + 1 = 5
    const auto plan = planOf(shape, Fields{});
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    EXPECT_EQ(a.fd, 3);
    EXPECT_EQ(a.tilesPlane, 32);
    EXPECT_EQ(a.tiles, 3 * 32);
    EXPECT_EQ(a.di, 9);
    EXPECT_EQ(a.dOut, 5);
    EXPECT_EQ(a.pd, 2);
    EXPECT_EQ(a.sd, 2);
    EXPECT_EQ(a.dd, 2);
    EXPECT_EQ(a.rowsAnImage, 5 * 14);
    EXPECT_EQ(a.rows, 8 * 5 * 14);
    EXPECT_EQ(a.xsd, 14 * 14);
    EXPECT_EQ(a.ysd, 14 * 14);
    EXPECT_EQ(a.wsd, 9);
    // 2-D: one tap, no depth strides.
    const auto flat = planOf(grouped(), Fields{});
    ASSERT_TRUE(flat.has_value());
    EXPECT_EQ(flat->args.fd, 1);
    EXPECT_EQ(flat->args.tiles, flat->args.tilesPlane);
    EXPECT_EQ(flat->args.rowsAnImage, 56);
    EXPECT_EQ(flat->args.xsd, 0);
}

TEST(TestConvGroupValuWrwKernel, TheRingStaysTwoD)
{
    auto ring = Fields{};
    ring.ring = 1;
    ring.cb = 2;
    ring.p = 4;
    EXPECT_FALSE(admits(ring, threeD()));
    auto cb2 = Fields{};
    cb2.cb = 2;
    cb2.p = 4;
    EXPECT_TRUE(admits(cb2, threeD()));
}

TEST(TestConvGroupValuWrwGraph, RefusesLayoutsWithNeitherAxisPacked)
{
    // x channels-first with padded rows (the width packed) is fine; x with its width strided by
    // 2 and its channels not packed either has no axis a lane's loads can run along.
    auto spec = specFor(grouped(), ConvDirection::BACKWARD_WEIGHTS);
    const auto good = spec;
    EXPECT_TRUE(graphAcceptsSpec(good));
    spec.x.strides = {int64_t{128} * 56 * 112, int64_t{56} * 112, 112, 2};
    EXPECT_FALSE(graphAcceptsSpec(spec));
    // dy alone strided: the same.
    auto dySpec = good;
    dySpec.y.strides = {int64_t{128} * 56 * 112, int64_t{56} * 112, 112, 2};
    EXPECT_FALSE(graphAcceptsSpec(dySpec));
    // x channels-first, dy channels-last: no common axis either.
    auto mixed = good;
    mixed.y.strides = packedStrides(mixed.y.dims, true);
    EXPECT_FALSE(graphAcceptsSpec(mixed));
}

TEST(TestConvGroupValuWrwGraph, RefusesExtentsBeyond32Bits)
{
    auto huge = grouped();
    huge.n = 1;
    huge.c = 4;
    huge.k = 4;
    huge.groups = 1;
    huge.input = {1, int64_t{3} << 30};
    EXPECT_FALSE(graphAccepts(huge));
}

// ---------------------------------------------------------------------------
// Kernel match: one decline per field the matcher reads
// ---------------------------------------------------------------------------

TEST(TestConvGroupValuWrwKernel, AdmitsTheResNeXtLayer)
{
    EXPECT_TRUE(admits(Fields{}));
    auto dot2 = Fields{};
    dot2.dot2 = 1;
    EXPECT_TRUE(admits(dot2));
}

TEST(TestConvGroupValuWrwKernel, TheDtypeMustMatch)
{
    auto f = Fields{};
    f.dtype = "FLOAT";
    EXPECT_FALSE(admits(f));
    auto fp32 = grouped();
    fp32.dataType = DataType::FLOAT;
    EXPECT_TRUE(admits(f, fp32));
}

TEST(TestConvGroupValuWrwKernel, TheLayoutMustMatch)
{
    auto nhwcConfig = Fields{};
    nhwcConfig.layout = "nhwc";
    EXPECT_FALSE(admits(nhwcConfig)); // NCHW graph
    auto nhwc = grouped();
    nhwc.channelsLast = true;
    EXPECT_TRUE(admits(nhwcConfig, nhwc));
    EXPECT_FALSE(admits(Fields{}, nhwc));
    auto bad = Fields{};
    bad.layout = "nchwc";
    EXPECT_FALSE(admits(bad));
}

TEST(TestConvGroupValuWrwKernel, TheFilterClassMustMatch)
{
    // Rows, columns, width stride, width dilation and width padding each.
    auto rows = grouped();
    rows.filter = {5, 3};
    rows.prePadding = rows.postPadding = {2, 1};
    EXPECT_FALSE(admits(Fields{}, rows));
    auto cols = grouped();
    cols.filter = {3, 5};
    cols.prePadding = cols.postPadding = {1, 2};
    EXPECT_FALSE(admits(Fields{}, cols));
    auto stride = grouped();
    stride.stride = {1, 2};
    EXPECT_FALSE(admits(Fields{}, stride));
    auto dilation = grouped();
    dilation.dilation = {1, 2};
    dilation.prePadding = dilation.postPadding = {1, 2};
    EXPECT_FALSE(admits(Fields{}, dilation));
    auto pad = grouped();
    pad.prePadding = {1, 0};
    pad.postPadding = {1, 0};
    EXPECT_FALSE(admits(Fields{}, pad));
    // The height's stride, dilation and padding are runtime: admitted.
    auto height = grouped();
    height.stride = {2, 1};
    height.dilation = {2, 1};
    height.prePadding = height.postPadding = {2, 1};
    EXPECT_TRUE(admits(Fields{}, height));
}

TEST(TestConvGroupValuWrwKernel, TheTileMustDivideTheGroup)
{
    auto threeChannels = grouped();
    threeChannels.c = 96; // 3 a group
    auto spec = specFor(threeChannels, ConvDirection::BACKWARD_WEIGHTS);
    EXPECT_FALSE(admitsSpec(Fields{}, spec));
    auto cb1 = Fields{};
    cb1.cb = 1;
    EXPECT_TRUE(admitsSpec(cb1, spec));
    auto twoOutputs = grouped();
    twoOutputs.k = 64; // 2 a group
    EXPECT_FALSE(admits(Fields{}, twoOutputs));
    auto kb2 = Fields{};
    kb2.kb = 2;
    EXPECT_TRUE(admits(kb2, twoOutputs));
}

TEST(TestConvGroupValuWrwKernel, ARowMustFitTheWave)
{
    auto wide = grouped();
    wide.input = {8, 600}; // 600 / 8 = 75 lanes > 64
    EXPECT_FALSE(admits(Fields{}, wide));
    auto fits = grouped();
    fits.input = {8, 512};
    EXPECT_TRUE(admits(Fields{}, fits));
    // A wave of 32 lanes (RDNA) covers half the row.
    EXPECT_FALSE(planOf(fits, Fields{}, 32).has_value());
    EXPECT_TRUE(planOf(fits, Fields{}, 64).has_value());
}

TEST(TestConvGroupValuWrwKernel, Dot2NeedsHalfWholeDwordsAndNoRaggedEdge)
{
    auto dot2 = Fields{};
    dot2.dot2 = 1;
    auto fp32 = grouped();
    fp32.dataType = DataType::FLOAT;
    auto fp32Fields = dot2;
    fp32Fields.dtype = "FLOAT";
    EXPECT_FALSE(admits(fp32Fields, fp32));
    auto ragged = grouped();
    ragged.input = {50, 50}; // 50 % 8 != 0
    EXPECT_FALSE(admits(dot2, ragged));
    EXPECT_TRUE(admits(Fields{}, ragged)); // the element path serves it
    auto odd = grouped();
    odd.input = {56, 58}; // rows of 58 halves: dword loads at most
    auto p2 = dot2;
    p2.p = 2;
    EXPECT_TRUE(admits(p2, odd));
    // Pointers aligned to 2 bytes only: element loads, no dot2.
    auto spec = specFor(grouped(), ConvDirection::BACKWARD_WEIGHTS);
    spec.x.alignment = 2;
    EXPECT_FALSE(admitsSpec(dot2, spec));
    EXPECT_TRUE(admitsSpec(Fields{}, spec));
}

TEST(TestConvGroupValuWrwKernel, TheRingNeedsUnitHeightStrideAndDilation)
{
    auto ring = Fields{};
    ring.ring = 1;
    ring.cb = 2;
    ring.p = 4;
    EXPECT_TRUE(admits(ring));
    auto stride = grouped();
    stride.stride = {2, 1};
    EXPECT_FALSE(admits(ring, stride));
    auto dilation = grouped();
    dilation.dilation = {2, 1};
    dilation.prePadding = dilation.postPadding = {2, 1};
    EXPECT_FALSE(admits(ring, dilation));
    // The pass form serves both.
    EXPECT_TRUE(admits(Fields{}, stride));
    EXPECT_TRUE(admits(Fields{}, dilation));
}

TEST(TestConvGroupValuWrwPlan, RingItemsCoverEveryRowOnce)
{
    auto ring = Fields{};
    ring.ring = 1;
    ring.cb = 2;
    ring.p = 4;
    for(const int64_t n : {1, 3, 128})
    {
        auto shape = grouped();
        shape.n = n;
        const auto plan = planOf(shape, ring);
        ASSERT_TRUE(plan.has_value());
        const auto& a = plan->args;
        EXPECT_GE(a.chunk, 1);
        EXPECT_GE(static_cast<int64_t>(a.chunk) * a.itemsPerImage, 56);
        EXPECT_LT(static_cast<int64_t>(a.chunk) * (a.itemsPerImage - 1), 56);
        EXPECT_EQ(a.items, n * a.itemsPerImage);
        EXPECT_GE(static_cast<int64_t>(a.itemsPerSplit) * a.splits, a.items);
        EXPECT_LT(static_cast<int64_t>(a.itemsPerSplit) * (a.splits - 1), a.items);
        EXPECT_EQ(a.itemsPerSplit % (64 / a.lanes), 0);
    }
}

TEST(TestConvGroupValuWrwKernel, RefusesMalformedConfigs)
{
    const auto refused = [](auto mutate) {
        auto f = Fields{};
        mutate(f);
        return !gvw::configOf(kernelOf(f)).has_value();
    };
    EXPECT_FALSE(refused([](Fields&) {}));
    EXPECT_TRUE(refused([](Fields& f) { f.cb = 3; }));
    EXPECT_TRUE(refused([](Fields& f) { f.kb = 0; }));
    EXPECT_TRUE(refused([](Fields& f) { f.p = 0; }));
    EXPECT_TRUE(refused([](Fields& f) { f.p = 17; }));
    EXPECT_TRUE(refused([](Fields& f) { f.fr = 0; }));
    EXPECT_TRUE(refused([](Fields& f) { f.fs = 8; }));
    EXPECT_TRUE(refused([](Fields& f) { f.sw = 3; }));
    EXPECT_TRUE(refused([](Fields& f) { f.dw = 0; }));
    EXPECT_TRUE(refused([](Fields& f) { f.pw = -1; }));
    EXPECT_TRUE(refused([](Fields& f) { f.nw = 3; }));
    EXPECT_TRUE(refused([](Fields& f) { f.dot2 = 2; }));
    EXPECT_TRUE(refused([](Fields& f) { f.ring = 2; }));
    EXPECT_TRUE(refused([](Fields& f) {
        f.ring = 1;
        f.layout = "nhwc";
    }));
    EXPECT_TRUE(refused([](Fields& f) {
        f.dot2 = 1;
        f.p = 3;
    }));
    EXPECT_TRUE(refused([](Fields& f) {
        f.dot2 = 1;
        f.sw = 2;
    }));
    EXPECT_TRUE(refused([](Fields& f) {
        f.dot2 = 1;
        f.layout = "nhwc";
    }));
    // The window past one neighbouring lane: 1 column a lane, a 5-tap filter.
    EXPECT_TRUE(refused([](Fields& f) {
        f.p = 1;
        f.fs = 5;
        f.pw = 2;
    }));
    // Past the accumulator budget: 8 x 8 x 3 x 3.
    EXPECT_TRUE(refused([](Fields& f) {
        f.cb = 8;
        f.kb = 8;
    }));
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

TEST(TestConvGroupValuWrwPlan, LanesLoadsSplitsAndWorkspace)
{
    const auto plan = planOf(grouped(), Fields{});
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->args.lanes, 8); // 56 / 8 = 7 lanes, a power of two
    EXPECT_FALSE(plan->ragged);
    EXPECT_EQ(plan->vbx, 16); // 8 halves
    EXPECT_EQ(plan->vby, 16);
    EXPECT_EQ(plan->args.tiles, 32);
    EXPECT_EQ(plan->args.cBlocks, 1);
    EXPECT_EQ(plan->args.kBlocks, 1);
    EXPECT_EQ(plan->accumulators, 144);
    EXPECT_GE(plan->args.splits, 2);
    EXPECT_EQ(plan->args.rowsPerSplit % 8, 0); // whole passes of 8 rows
    EXPECT_GE(static_cast<int64_t>(plan->args.rowsPerSplit) * plan->args.splits,
              static_cast<int64_t>(plan->args.rows));
    EXPECT_EQ(plan->workspaceBytes,
              static_cast<size_t>(plan->args.splits) * 32 * 144 * sizeof(float));
    EXPECT_GT(plan->foldGrid, 0);

    auto ragged = grouped();
    ragged.input = {50, 50};
    const auto raggedPlan = planOf(ragged, Fields{});
    ASSERT_TRUE(raggedPlan.has_value());
    EXPECT_TRUE(raggedPlan->ragged);
    EXPECT_EQ(raggedPlan->vbx, 2); // element loads
    // One split when the tiles fill the device: no workspace, no fold.
    auto wideGroups = grouped();
    wideGroups.c = wideGroups.k = 1024;
    auto small = Fields{};
    small.cb = 1;
    const auto onePlan = planOf(wideGroups, small);
    ASSERT_TRUE(onePlan.has_value());
    EXPECT_EQ(onePlan->args.splits, 1);
    EXPECT_EQ(onePlan->workspaceBytes, 0u);
}

TEST(TestConvGroupValuWrwPlan, NarrowPointerAlignmentNarrowsTheLoads)
{
    auto spec = specFor(grouped(), ConvDirection::BACKWARD_WEIGHTS);
    spec.x.alignment = 4;
    const GraphFixture fixture(buildGraph(spec));
    const auto problem = gvw::parseGvwProblem(fixture.context());
    ASSERT_TRUE(problem.has_value());
    const auto plan = gvw::planFor(*problem, *gvw::configOf(kernelOf(Fields{})), 64, 60);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->vbx, 4);
    EXPECT_EQ(plan->vby, 16);
}

TEST(TestConvGroupValuWrwPlan, LargeTensorsWithSmallImagesAreServed)
{
    // 4100 images of 16 x 128 x 128 halves: x is 2.15 GB, every image 512 KiB.
    auto large = grouped();
    large.n = 4100;
    large.c = 16;
    large.k = 4;
    large.groups = 1;
    large.input = {128, 128};
    large.stride = {2, 2};
    auto f = Fields{};
    f.cb = 1;
    f.p = 4;
    f.sw = 2;
    const auto plan = planOf(large, f);
    ASSERT_TRUE(plan.has_value());
    EXPECT_GT(plan->args.xBytes, int64_t{1} << 31);
}

TEST(TestConvGroupValuWrwPlan, RefusesImagesPastThe32BitOffsets)
{
    // One image of 4 x 2^28 halves (2 GiB).
    auto huge = grouped();
    huge.n = 1;
    huge.c = 4;
    huge.k = 4;
    huge.groups = 1;
    huge.input = {int64_t{1} << 22, 64};
    auto f = Fields{};
    EXPECT_FALSE(planOf(huge, f).has_value());
}

} // namespace

#endif
