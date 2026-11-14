// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvGroupValuProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvGroupValuProblem.cpp
 * @brief Host-only checks of CONV_GROUP_VALU_FWD / CONV_GROUP_VALU_BWD: what each graph match
 *        accepts and refuses (one decline per gated field), what the kernel match admits per
 *        config (direction, dtype, layout, filter class, tile divisibility, row width), one
 *        decline per malformed config field, and the plan's kernel roles, padding, lanes, access
 *        widths, row blocks and workspace. No device.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
namespace gv = conv_group_valu;

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD ? "hipkernel.conv_group_valu_fwd.graph_match"
                                                : "hipkernel.conv_group_valu_bwd.graph_match",
            "",
            "hipkernel.conv_group_valu.kernel_match",
            "",
            "hipkernel.conv_group_valu.dispatch",
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

ConvShape threeD()
{
    auto shape = grouped();
    shape.input = {6, 14, 14};
    shape.filter = {3, 3, 3};
    shape.prePadding = shape.postPadding = {1, 1, 1};
    shape.stride = {1, 1, 1};
    shape.dilation = {1, 1, 1};
    return shape;
}

struct Fields
{
    std::string direction = "FWD";
    std::string dtype = "HALF";
    std::string layout = "nchw";
    int64_t kb = 4, cb = 2, p = 8, fd = 1, fr = 3, fs = 3, sw = 1, dw = 1, pw = 1, nw = 2;
};

Fields bwdFields()
{
    auto f = Fields{};
    f.direction = "BWD";
    return f;
}

ConvDirection directionOf(const Fields& f)
{
    return f.direction == "BWD" ? ConvDirection::BACKWARD_DATA : ConvDirection::FORWARD;
}

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const Fields& f)
{
    auto kernel = makeKernel(0, f.dtype, "ConvGroupValu");
    kernel.source.sourceFile = "kernels/ConvGroupValu.cpp";
    kernel.metadata = {{"direction", f.direction},
                       {"dtype", f.dtype},
                       {"layout", f.layout},
                       {"kb", f.kb},
                       {"cb", f.cb},
                       {"p", f.p},
                       {"fd", f.fd},
                       {"fr", f.fr},
                       {"fs", f.fs},
                       {"sw", f.sw},
                       {"dw", f.dw},
                       {"pw", f.pw},
                       {"nw", f.nw}};
    return kernel;
}

bool graphAcceptsSpec(const ConvGraphSpec& spec, ConvDirection direction)
{
    const GraphFixture fixture(buildGraph(spec));
    return matchesGraph(packFor(direction), fixture.context()).has_value();
}

bool graphAccepts(const ConvShape& shape,
                  ConvDirection direction = ConvDirection::FORWARD,
                  DataType computeType = DataType::FLOAT)
{
    auto spec = specFor(shape, direction);
    spec.computeType = computeType;
    return graphAcceptsSpec(spec, direction);
}

/// Whether @p fields' config serves @p spec, a graph of @p graphDirection, through the pack of
/// the graph's direction.
bool admitsSpec(const Fields& fields, const ConvGraphSpec& spec, ConvDirection graphDirection)
{
    const auto pack = packFor(graphDirection);
    const GraphFixture fixture(buildGraph(spec));
    const auto bound = matchesGraph(pack, fixture.context());
    return bound.has_value() && matchesKernel(pack, fixture.context(), kernelOf(fields), *bound);
}

bool admits(const Fields& fields, const ConvShape& shape = grouped())
{
    return admitsSpec(fields, specFor(shape, directionOf(fields)), directionOf(fields));
}

std::optional<gv::Plan>
    planOf(const ConvShape& shape, const Fields& fields, int wave = 64, int64_t units = 60)
{
    const auto direction = directionOf(fields);
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    const auto problem = gv::parseGvProblem(fixture.context(), direction);
    const auto config = gv::configOf(kernelOf(fields));
    if(!problem.has_value() || !config.has_value())
    {
        return std::nullopt;
    }
    return gv::planFor(*problem, *config, wave, units);
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

TEST(TestConvGroupValuGraph, AcceptsLayoutsDtypesRanksGroupsAndModes)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        EXPECT_TRUE(graphAccepts(grouped(), direction));
        auto nhwc = grouped();
        nhwc.channelsLast = true;
        EXPECT_TRUE(graphAccepts(nhwc, direction));
        for(const auto type : {DataType::FLOAT, DataType::BFLOAT16})
        {
            auto other = grouped();
            other.dataType = type;
            EXPECT_TRUE(graphAccepts(other, direction));
        }
        auto dense = grouped();
        dense.groups = 1;
        EXPECT_TRUE(graphAccepts(dense, direction));
        auto oneD = grouped();
        oneD.input = {300};
        oneD.filter = {3};
        oneD.prePadding = oneD.postPadding = {1};
        oneD.stride = {1};
        oneD.dilation = {1};
        EXPECT_TRUE(graphAccepts(oneD, direction));
        EXPECT_TRUE(graphAccepts(threeD(), direction));
        auto convolution = grouped();
        convolution.mode = ConvMode::CONVOLUTION;
        EXPECT_TRUE(graphAccepts(convolution, direction));
        EXPECT_TRUE(graphAccepts(grouped(), direction, DataType::UNSET));
    }
}

TEST(TestConvGroupValuGraph, EachPackRefusesTheOtherDirections)
{
    const GraphFixture fwd(buildGraph(specFor(grouped(), ConvDirection::FORWARD)));
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::BACKWARD_DATA), fwd.context()).has_value());
    const GraphFixture bwd(buildGraph(specFor(grouped(), ConvDirection::BACKWARD_DATA)));
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::FORWARD), bwd.context()).has_value());
    const GraphFixture wrw(buildGraph(specFor(grouped(), ConvDirection::BACKWARD_WEIGHTS)));
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::FORWARD), wrw.context()).has_value());
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::BACKWARD_DATA), wrw.context()).has_value());
}

TEST(TestConvGroupValuGraph, RefusesAComputeTypeBelowFp32)
{
    EXPECT_FALSE(graphAccepts(grouped(), ConvDirection::FORWARD, DataType::HALF));
    EXPECT_FALSE(graphAccepts(grouped(), ConvDirection::BACKWARD_DATA, DataType::HALF));
}

TEST(TestConvGroupValuGraph, AcceptsEveryStride)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const auto& stride : std::vector<std::vector<int64_t>>{{1, 2}, {2, 1}, {3, 3}})
        {
            auto strided = grouped();
            strided.stride = stride;
            EXPECT_TRUE(graphAccepts(strided, direction));
        }
        auto depth = threeD();
        depth.stride = {2, 1, 1};
        EXPECT_TRUE(graphAccepts(depth, direction));
    }
}

TEST(TestConvGroupValuGraph, RefusesLayoutsWithNeitherAxisPacked)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto good = specFor(grouped(), direction);
        EXPECT_TRUE(graphAcceptsSpec(good, direction));
        auto xSpec = good;
        xSpec.x.strides = {int64_t{128} * 56 * 112, int64_t{56} * 112, 112, 2};
        EXPECT_FALSE(graphAcceptsSpec(xSpec, direction));
        auto ySpec = good;
        ySpec.y.strides = {int64_t{128} * 56 * 112, int64_t{56} * 112, 112, 2};
        EXPECT_FALSE(graphAcceptsSpec(ySpec, direction));
        auto mixed = good;
        mixed.y.strides = packedStrides(mixed.y.dims, true);
        EXPECT_FALSE(graphAcceptsSpec(mixed, direction));
    }
}

TEST(TestConvGroupValuGraph, RefusesExtentsBeyond32Bits)
{
    auto huge = grouped();
    huge.n = 1;
    huge.c = 4;
    huge.k = 4;
    huge.groups = 1;
    huge.input = {1, int64_t{3} << 30};
    EXPECT_FALSE(graphAccepts(huge, ConvDirection::FORWARD));
    EXPECT_FALSE(graphAccepts(huge, ConvDirection::BACKWARD_DATA));
}

// ---------------------------------------------------------------------------
// Kernel match: one decline per field the matcher reads
// ---------------------------------------------------------------------------

TEST(TestConvGroupValuKernel, AdmitsTheResNeXtLayer)
{
    EXPECT_TRUE(admits(Fields{}));
    EXPECT_TRUE(admits(bwdFields()));
}

TEST(TestConvGroupValuKernel, TheDirectionMustMatch)
{
    // A BWD config through the forward pack on a forward graph, and the reverse.
    const auto fwdSpec = specFor(grouped(), ConvDirection::FORWARD);
    EXPECT_FALSE(admitsSpec(bwdFields(), fwdSpec, ConvDirection::FORWARD));
    const auto bwdSpec = specFor(grouped(), ConvDirection::BACKWARD_DATA);
    EXPECT_FALSE(admitsSpec(Fields{}, bwdSpec, ConvDirection::BACKWARD_DATA));
}

TEST(TestConvGroupValuKernel, TheDtypeMustMatch)
{
    for(auto f : {Fields{}, bwdFields()})
    {
        f.dtype = "FLOAT";
        EXPECT_FALSE(admits(f));
        auto fp32 = grouped();
        fp32.dataType = DataType::FLOAT;
        EXPECT_TRUE(admits(f, fp32));
        f.dtype = "BFLOAT16";
        EXPECT_FALSE(admits(f));
    }
}

TEST(TestConvGroupValuKernel, TheLayoutMustMatch)
{
    for(const auto& base : {Fields{}, bwdFields()})
    {
        auto nhwcConfig = base;
        nhwcConfig.layout = "nhwc";
        EXPECT_FALSE(admits(nhwcConfig)); // NCHW graph
        auto nhwc = grouped();
        nhwc.channelsLast = true;
        EXPECT_TRUE(admits(nhwcConfig, nhwc));
        EXPECT_FALSE(admits(base, nhwc));
    }
}

TEST(TestConvGroupValuKernel, TheFilterClassMustMatch)
{
    for(const auto& f : {Fields{}, bwdFields()})
    {
        // Depth, rows, columns, width dilation and width padding each.
        EXPECT_FALSE(admits(f, threeD()));
        auto fd3 = f;
        fd3.fd = 3;
        EXPECT_TRUE(admits(fd3, threeD()));
        EXPECT_FALSE(admits(fd3));
        auto rows = grouped();
        rows.filter = {5, 3};
        rows.prePadding = rows.postPadding = {2, 1};
        EXPECT_FALSE(admits(f, rows));
        auto cols = grouped();
        cols.filter = {3, 5};
        cols.prePadding = cols.postPadding = {1, 2};
        EXPECT_FALSE(admits(f, cols));
        auto dilation = grouped();
        dilation.dilation = {1, 2};
        dilation.prePadding = dilation.postPadding = {1, 2};
        EXPECT_FALSE(admits(f, dilation));
        auto dw2 = f;
        dw2.dw = 2;
        dw2.pw = 2;
        EXPECT_TRUE(admits(dw2, dilation));
        auto pad = grouped();
        pad.prePadding = {1, 0};
        pad.postPadding = {1, 0};
        EXPECT_FALSE(admits(f, pad));
        // The height's dilation and padding are runtime: admitted.
        auto height = grouped();
        height.dilation = {2, 1};
        height.prePadding = height.postPadding = {2, 1};
        EXPECT_TRUE(admits(f, height));
    }
    // The width stride (forward only).
    auto stride = grouped();
    stride.stride = {1, 2};
    EXPECT_FALSE(admits(Fields{}, stride));
    auto sw2 = Fields{};
    sw2.sw = 2;
    sw2.p = 4;
    EXPECT_TRUE(admits(sw2, stride));
    EXPECT_FALSE(admits(sw2));
    // The height stride is runtime for the forward.
    auto heightStride = grouped();
    heightStride.stride = {2, 1};
    EXPECT_TRUE(admits(Fields{}, heightStride));
}

TEST(TestConvGroupValuKernel, TheTileMustDivideTheKernelRoles)
{
    // Forward: kb divides the y channels a group, cb the x channels a group. Backward: the roles
    // swap.
    auto threeIn = grouped();
    threeIn.c = 96; // 3 x channels a group
    auto twoOut = grouped();
    twoOut.k = 64; // 2 y channels a group
    auto cb1 = Fields{};
    cb1.cb = 1;
    auto kb2 = Fields{};
    kb2.kb = 2;
    EXPECT_FALSE(admits(Fields{}, threeIn)); // cb 2 into 3
    EXPECT_TRUE(admits(cb1, threeIn));
    EXPECT_FALSE(admits(Fields{}, twoOut)); // kb 4 into 2
    EXPECT_TRUE(admits(kb2, twoOut));
    auto bwdCb1 = bwdFields();
    bwdCb1.cb = 1;
    auto bwdKb2 = bwdFields();
    bwdKb2.kb = 2;
    auto bwdKb1 = bwdFields();
    bwdKb1.kb = 1;
    EXPECT_FALSE(admits(bwdFields(), threeIn)); // kb 4 into 3 (the x channels)
    EXPECT_TRUE(admits(bwdKb1, threeIn));
    EXPECT_TRUE(admits(bwdFields(), twoOut)); // cb 2 into 2 (the dy channels)
    auto bwdCb4 = bwdFields();
    bwdCb4.cb = 4;
    EXPECT_FALSE(admits(bwdCb4, twoOut));
    EXPECT_TRUE(admits(bwdCb1, twoOut));
    EXPECT_TRUE(admits(bwdKb2, twoOut));
}

TEST(TestConvGroupValuKernel, ARowMustFitTheWave)
{
    auto wide = grouped();
    wide.input = {8, 600}; // 600 / 8 = 75 lanes > 64
    EXPECT_FALSE(admits(Fields{}, wide));
    auto fits = grouped();
    fits.input = {8, 512};
    EXPECT_TRUE(admits(Fields{}, fits));
    EXPECT_FALSE(planOf(fits, Fields{}, 32).has_value());
    EXPECT_TRUE(planOf(fits, Fields{}, 64).has_value());
}

TEST(TestConvGroupValuKernel, RefusesMalformedConfigs)
{
    const auto refused = [](auto mutate) {
        auto f = Fields{};
        mutate(f);
        return !gv::configOf(kernelOf(f)).has_value();
    };
    EXPECT_FALSE(refused([](Fields&) {}));
    EXPECT_TRUE(refused([](Fields& f) { f.direction = "WRW"; }));
    EXPECT_TRUE(refused([](Fields& f) { f.layout = "nchwc"; }));
    EXPECT_TRUE(refused([](Fields& f) { f.kb = 3; }));
    EXPECT_TRUE(refused([](Fields& f) { f.kb = 32; }));
    EXPECT_TRUE(refused([](Fields& f) { f.cb = 3; }));
    EXPECT_TRUE(refused([](Fields& f) { f.cb = 8; }));
    EXPECT_TRUE(refused([](Fields& f) { f.p = 0; }));
    EXPECT_TRUE(refused([](Fields& f) { f.p = 17; }));
    EXPECT_TRUE(refused([](Fields& f) { f.fd = 0; }));
    EXPECT_TRUE(refused([](Fields& f) { f.fr = 8; }));
    EXPECT_TRUE(refused([](Fields& f) { f.fs = 0; }));
    EXPECT_TRUE(refused([](Fields& f) { f.sw = 3; }));
    EXPECT_TRUE(refused([](Fields& f) { f.dw = 0; }));
    EXPECT_TRUE(refused([](Fields& f) { f.pw = -1; }));
    EXPECT_TRUE(refused([](Fields& f) { f.nw = 3; }));
    EXPECT_TRUE(refused([](Fields& f) { f.nw = 16; }));
    // Past the accumulator budget: 16 x 8.
    EXPECT_TRUE(refused([](Fields& f) {
        f.kb = 16;
        f.p = 8;
    }));
    // Backward data at width stride 2: whole input columns a lane (p even).
    EXPECT_TRUE(refused([](Fields& f) {
        f.direction = "BWD";
        f.sw = 2;
        f.p = 3;
    }));
    EXPECT_FALSE(refused([](Fields& f) {
        f.direction = "BWD";
        f.sw = 2;
        f.p = 4;
    }));
}

TEST(TestConvGroupValuKernel, BackwardDataStrideTheWidthsFixedTheRestRuntime)
{
    // Width stride 2: only an sw 2 config; height and depth strides: runtime.
    auto wide = grouped();
    wide.stride = {2, 2};
    auto sw2 = bwdFields();
    sw2.sw = 2;
    sw2.p = 4;
    EXPECT_FALSE(admits(bwdFields(), wide));
    EXPECT_TRUE(admits(sw2, wide));
    EXPECT_FALSE(admits(sw2)); // unit stride graph
    auto tall = grouped();
    tall.stride = {2, 1};
    EXPECT_TRUE(admits(bwdFields(), tall));
    auto deep = threeD();
    deep.stride = {2, 2, 2};
    auto deep3 = sw2;
    deep3.fd = 3;
    EXPECT_TRUE(admits(deep3, deep));
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

TEST(TestConvGroupValuPlan, ForwardLanesAccessWidthsRowBlocksAndWorkspace)
{
    const auto plan = planOf(grouped(), Fields{});
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    EXPECT_EQ(a.lanes, 8); // 56 / 8 = 7 lanes, a power of two
    EXPECT_FALSE(plan->ragged);
    EXPECT_FALSE(plan->colmask);
    EXPECT_EQ(plan->vbx, 16); // 8 halves
    EXPECT_EQ(plan->vby, 16);
    EXPECT_EQ(plan->kernelPw, 1);
    EXPECT_EQ(a.ci, 4);
    EXPECT_EQ(a.ko, 4);
    EXPECT_EQ(a.kBlocks, 1);
    EXPECT_EQ(a.backward, 0);
    EXPECT_EQ(a.rows, 8 * 56);
    EXPECT_EQ(a.rowsPerBlock % 8, 0); // whole passes of 64 / 8 rows
    EXPECT_GE(static_cast<int64_t>(a.rowsPerBlock) * a.rowBlocks, static_cast<int64_t>(a.rows));
    EXPECT_LT(static_cast<int64_t>(a.rowsPerBlock) * (a.rowBlocks - 1),
              static_cast<int64_t>(a.rows));
    // The prepared filter: groups x kBlocks x ci x taps x kb floats.
    EXPECT_EQ(a.prepared, 32 * 1 * 4 * 9 * 4);
    EXPECT_EQ(plan->workspaceBytes, static_cast<size_t>(a.prepared) * sizeof(float));
    EXPECT_EQ(plan->prepareGrid, (a.prepared + 255) / 256);

    auto ragged = grouped();
    ragged.input = {50, 50};
    const auto raggedPlan = planOf(ragged, Fields{});
    ASSERT_TRUE(raggedPlan.has_value());
    EXPECT_TRUE(raggedPlan->ragged);
    EXPECT_TRUE(raggedPlan->colmask);
    EXPECT_EQ(raggedPlan->vbx, 2); // element loads
    EXPECT_EQ(raggedPlan->vby, 2);
}

TEST(TestConvGroupValuPlan, BackwardSwapsRolesAndMirrorsThePadding)
{
    auto shape = grouped();
    shape.c = 256; // 8 x channels a group, 4 dy channels a group
    shape.dilation = {2, 1};
    shape.prePadding = shape.postPadding = {1, 1};
    shape.input = {58, 56}; // dy 56 x 56
    auto f = bwdFields();
    const auto plan = planOf(shape, f);
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    EXPECT_EQ(a.backward, 1);
    EXPECT_EQ(a.ci, 4); // dy channels
    EXPECT_EQ(a.ko, 8); // dx channels
    EXPECT_EQ(a.kBlocks, 2);
    EXPECT_EQ(a.hi, 56); // the kernel reads dy
    EXPECT_EQ(a.ho, 58); // and writes dx
    EXPECT_EQ(a.ph, 2 * 2 + 1 - 1 - 1); // span 5 - 1 - pad 1
    EXPECT_EQ(plan->kernelPw, 1); // span 3 - 1 - pad 1
    EXPECT_EQ(a.wGraphK, 4);
    EXPECT_EQ(a.wGraphC, 8);
    // A padding past the span - 1 has no unit-stride correlation form.
    auto wide = grouped();
    wide.prePadding = wide.postPadding = {3, 1};
    EXPECT_FALSE(planOf(wide, bwdFields()).has_value());
}

TEST(TestConvGroupValuPlan, BackwardDataUpsamplesDyByTheStride)
{
    // dx 56 x 56 from dy 28 x 28 at stride 2, padding 1: the kernel reads dy upsampled by 2 at
    // unit stride, padding 3 - 1 - 1 = 1; a lane's 4 dx columns over 2 dy columns.
    auto shape = grouped();
    shape.stride = {2, 2};
    auto f = bwdFields();
    f.sw = 2;
    f.p = 4;
    const auto plan = planOf(shape, f);
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    EXPECT_EQ(a.hi, 28);
    EXPECT_EQ(a.wi, 28);
    EXPECT_EQ(a.ho, 56);
    EXPECT_EQ(a.wo, 56);
    EXPECT_EQ(a.sh, 1);
    EXPECT_EQ(a.uh, 2);
    EXPECT_EQ(a.ud, 1);
    EXPECT_EQ(a.ph, 1);
    EXPECT_EQ(plan->kernelPw, 1);
    EXPECT_EQ(a.lanes, 16); // 56 / 4 = 14 lanes
    EXPECT_EQ(plan->vbx, 4); // 2 halves of dy
    EXPECT_EQ(plan->vby, 8); // 4 halves of dx
    // Forward: unit upsampling, the stride the kernel's.
    auto fwd = Fields{};
    fwd.sw = 2;
    fwd.p = 4;
    const auto fwdPlan = planOf(shape, fwd);
    ASSERT_TRUE(fwdPlan.has_value());
    EXPECT_EQ(fwdPlan->args.sh, 2);
    EXPECT_EQ(fwdPlan->args.uh, 1);
}

TEST(TestConvGroupValuPlan, NarrowPointerAlignmentNarrowsTheAccesses)
{
    auto spec = specFor(grouped(), ConvDirection::FORWARD);
    spec.x.alignment = 4;
    const GraphFixture fixture(buildGraph(spec));
    const auto problem = gv::parseGvProblem(fixture.context(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    const auto plan = gv::planFor(*problem, *gv::configOf(kernelOf(Fields{})), 64, 60);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->vbx, 4);
    EXPECT_EQ(plan->vby, 16);
}

TEST(TestConvGroupValuPlan, NhwcAccessesRunAlongTheChannels)
{
    auto nhwc = grouped();
    nhwc.channelsLast = true;
    auto f = Fields{};
    f.layout = "nhwc";
    const auto plan = planOf(nhwc, f);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->vbx, 4); // cb 2 halves
    EXPECT_EQ(plan->vby, 8); // kb 4 halves
}

TEST(TestConvGroupValuPlan, LargeTensorsWithSmallImagesAreServed)
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
    EXPECT_GT(plan->args.inBytes, int64_t{1} << 31);
}

TEST(TestConvGroupValuPlan, RefusesImagesPastThe32BitOffsets)
{
    // One image of 4 x 2^28 halves (2 GiB).
    auto huge = grouped();
    huge.n = 1;
    huge.c = 4;
    huge.k = 4;
    huge.groups = 1;
    huge.input = {int64_t{1} << 22, 64};
    EXPECT_FALSE(planOf(huge, Fields{}).has_value());
    EXPECT_FALSE(planOf(huge, bwdFields()).has_value());
}

TEST(TestConvGroupValuPlan, RefusesMoreGroupBlocksThanTheGridHolds)
{
    auto many = grouped();
    many.n = 1;
    many.input = {8, 8};
    many.groups = 70000;
    many.c = 70000;
    many.k = 70000;
    auto f = Fields{};
    f.kb = 1;
    f.cb = 1;
    EXPECT_FALSE(planOf(many, f).has_value());
}

} // namespace

#endif
