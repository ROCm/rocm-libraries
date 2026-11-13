// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvFftProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvFftProblem.cpp
 * @brief Host-only checks of the FFT convolution engines (CONV_FFT_FWD / _BWD / _WRW): what the
 *        graph matches accept and refuse (one decline per gated field), what the kernel match
 *        admits per config (direction, dtype, transform fit, admission floor), one decline per
 *        malformed config field, and the plan's tiles, chunks and workspace. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD         ? "hipkernel.conv_fft_fwd.graph_match"
            : direction == ConvDirection::BACKWARD_DATA ? "hipkernel.conv_fft_bwd.graph_match"
                                                        : "hipkernel.conv_fft_wrw.graph_match",
            "",
            "hipkernel.conv_fft.kernel_match",
            "",
            "hipkernel.conv_fft.dispatch",
            "",
            "",
            ""};
}

/// A large-filter layer the floor admits: 11x11, padding 5, 64 x 64 channels, 64x64 image.
ConvShape largeShape()
{
    ConvShape shape;
    shape.n = 4;
    shape.c = 64;
    shape.k = 64;
    shape.input = {64, 64};
    shape.filter = {11, 11};
    shape.prePadding = {5, 5};
    shape.postPadding = {5, 5};
    shape.stride = {1, 1};
    shape.dilation = {1, 1};
    shape.dataType = DataType::FLOAT;
    return shape;
}

struct Fields
{
    std::string direction = "FWD";
    std::string dtype = "FLOAT";
    int64_t th = 32, tw = 32, pb = 2, tb = 64, gn = 64, tm = 4;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const Fields& f)
{
    auto kernel = makeKernel(0, f.dtype, "ConvFftData");
    kernel.source.sourceFile = "kernels/ConvFft.cpp";
    kernel.metadata = {{"direction", f.direction},
                       {"dtype", f.dtype},
                       {"th", f.th},
                       {"tw", f.tw},
                       {"pb", f.pb},
                       {"tb", f.tb},
                       {"gn", f.gn},
                       {"tm", f.tm}};
    return kernel;
}

bool graphAccepts(const ConvShape& shape,
                  ConvDirection direction = ConvDirection::FORWARD,
                  DataType computeType = DataType::FLOAT)
{
    auto spec = specFor(shape, direction);
    spec.computeType = computeType;
    const GraphFixture fixture(buildGraph(spec));
    return matchesGraph(packFor(direction), fixture.context()).has_value();
}

bool admits(const Fields& fields,
            const ConvShape& shape = largeShape(),
            ConvDirection direction = ConvDirection::FORWARD)
{
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    const auto pack = packFor(direction);
    const auto bound = matchesGraph(pack, fixture.context());
    return bound.has_value() && matchesKernel(pack, fixture.context(), kernelOf(fields), *bound);
}

std::optional<conv_fft::Plan>
    planOf(const ConvShape& shape, ConvDirection direction, const Fields& fields)
{
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    const auto problem = conv_fft::parseFftProblem(fixture.context(), direction);
    const auto config = conv_fft::configOf(kernelOf(fields));
    if(!problem.has_value() || !config.has_value())
    {
        return std::nullopt;
    }
    return conv_fft::planFor(*problem, direction, *config);
}

// ---------------------------------------------------------------------------
// Graph match: what is served, and one decline per gated field
// ---------------------------------------------------------------------------

TEST(TestConvFftGraph, AcceptsEveryDirectionLayoutDtypeAndRank)
{
    for(const auto direction :
        {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        EXPECT_TRUE(graphAccepts(largeShape(), direction));
        auto nhwc = largeShape();
        nhwc.channelsLast = true;
        EXPECT_TRUE(graphAccepts(nhwc, direction));
        for(const auto type : {DataType::HALF, DataType::BFLOAT16})
        {
            auto sixteen = largeShape();
            sixteen.dataType = type;
            EXPECT_TRUE(graphAccepts(sixteen, direction));
        }
        auto oneD = largeShape();
        oneD.input = {4096};
        oneD.filter = {31};
        oneD.prePadding = oneD.postPadding = {15};
        oneD.stride = {1};
        oneD.dilation = {1};
        EXPECT_TRUE(graphAccepts(oneD, direction));
        auto grouped = largeShape();
        grouped.groups = 4;
        EXPECT_TRUE(graphAccepts(grouped, direction));
        auto dilated = largeShape();
        dilated.dilation = {2, 3};
        dilated.prePadding = dilated.postPadding = {10, 15};
        EXPECT_TRUE(graphAccepts(dilated, direction));
        auto convolution = largeShape();
        convolution.mode = ConvMode::CONVOLUTION;
        EXPECT_TRUE(graphAccepts(convolution, direction));
        EXPECT_TRUE(graphAccepts(largeShape(), direction, DataType::UNSET));
    }
}

TEST(TestConvFftGraph, RefusesThreeD)
{
    auto threeD = largeShape();
    threeD.input = {8, 32, 32};
    threeD.filter = {3, 11, 11};
    threeD.prePadding = threeD.postPadding = {1, 5, 5};
    threeD.stride = {1, 1, 1};
    threeD.dilation = {1, 1, 1};
    EXPECT_FALSE(graphAccepts(threeD));
}

TEST(TestConvFftGraph, RefusesStride)
{
    for(const auto& stride : {std::vector<int64_t>{2, 1}, std::vector<int64_t>{1, 2}})
    {
        auto strided = largeShape();
        strided.stride = stride;
        EXPECT_FALSE(graphAccepts(strided));
    }
}

TEST(TestConvFftGraph, RefusesAComputeTypeBelowFp32)
{
    auto sixteen = largeShape();
    sixteen.dataType = DataType::HALF;
    EXPECT_FALSE(graphAccepts(sixteen, ConvDirection::FORWARD, DataType::HALF));
    EXPECT_FALSE(graphAccepts(sixteen, ConvDirection::FORWARD, DataType::BFLOAT16));
    EXPECT_TRUE(graphAccepts(sixteen, ConvDirection::FORWARD, DataType::FLOAT));
}

TEST(TestConvFftGraph, RefusesOtherDirections)
{
    const GraphFixture fwd(buildGraph(specFor(largeShape(), ConvDirection::FORWARD)));
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::BACKWARD_DATA), fwd.context()).has_value());
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::BACKWARD_WEIGHTS), fwd.context()).has_value());
    const GraphFixture bwd(buildGraph(specFor(largeShape(), ConvDirection::BACKWARD_DATA)));
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::FORWARD), bwd.context()).has_value());
}

TEST(TestConvFftGraph, RefusesExtentsBeyond32Bits)
{
    auto huge = largeShape();
    huge.n = 1;
    huge.c = 1;
    huge.k = 1;
    huge.input = {1, int64_t{3} << 30};
    huge.filter = {1, 11};
    huge.prePadding = huge.postPadding = {0, 5};
    EXPECT_FALSE(graphAccepts(huge));
}

TEST(TestConvFftGraph, RefusesADilatedSpanBeyond32Bits)
{
    // 3 taps 2^30 apart: a span of 2^31 + 1 (the padding keeps the output 8 wide).
    auto wide = largeShape();
    wide.n = 1;
    wide.c = 1;
    wide.k = 1;
    wide.input = {1, 8};
    wide.filter = {1, 3};
    wide.dilation = {1, int64_t{1} << 30};
    wide.prePadding = wide.postPadding = {0, int64_t{1} << 30};
    EXPECT_FALSE(graphAccepts(wide));
}

// ---------------------------------------------------------------------------
// Kernel match
// ---------------------------------------------------------------------------

TEST(TestConvFftKernel, AdmitsItsDirectionAndDtypeOnly)
{
    EXPECT_TRUE(admits(Fields{}));
    auto half = Fields{};
    half.dtype = "HALF";
    EXPECT_FALSE(admits(half));
    auto bwd = Fields{};
    bwd.direction = "BWD";
    EXPECT_FALSE(admits(bwd)); // a backward config on a forward graph
    EXPECT_TRUE(admits(bwd, largeShape(), ConvDirection::BACKWARD_DATA));
    auto wrw = Fields{};
    wrw.direction = "WRW";
    EXPECT_TRUE(admits(wrw, largeShape(), ConvDirection::BACKWARD_WEIGHTS));
    auto unknown = Fields{};
    unknown.direction = "SIDEWAYS";
    EXPECT_FALSE(admits(unknown));
}

TEST(TestConvFftKernel, TheDilatedFilterMustFitTheTransform)
{
    auto big = largeShape();
    big.filter = {31, 31};
    big.prePadding = big.postPadding = {15, 15};
    auto sixteen = Fields{};
    sixteen.th = sixteen.tw = 16;
    sixteen.pb = 8;
    sixteen.tb = 128;
    EXPECT_FALSE(admits(sixteen, big)); // span 31 > 16
    {
        // Not the floor's doing: refused with the floor off as well, in either axis alone.
        const ScopedEnvironmentVariableSetter off("HIP_KERNEL_PROVIDER_CONV_FFT_FLOOR", "0");
        EXPECT_FALSE(admits(sixteen, big));
        auto wideOnly = big;
        wideOnly.filter = {1, 31};
        wideOnly.prePadding = wideOnly.postPadding = {0, 15};
        EXPECT_FALSE(admits(sixteen, wideOnly)); // span 31 > 16 in w only
        auto oneOver = big;
        oneOver.filter = {1, 17};
        oneOver.prePadding = oneOver.postPadding = {0, 8};
        EXPECT_FALSE(admits(sixteen, oneOver)); // span 17: an output tile of zero columns
        auto tallOnly = big;
        tallOnly.filter = {31, 1};
        tallOnly.prePadding = tallOnly.postPadding = {15, 0};
        EXPECT_FALSE(admits(sixteen, tallOnly)); // span 31 > 16 in h only
    }
    auto sixtyFour = Fields{};
    sixtyFour.th = sixtyFour.tw = 64;
    sixtyFour.pb = 1;
    EXPECT_TRUE(admits(sixtyFour, big));
    auto dilated = largeShape();
    dilated.dilation = {4, 4}; // span 41 > 32
    dilated.prePadding = dilated.postPadding = {20, 20};
    EXPECT_FALSE(admits(Fields{}, dilated));
}

TEST(TestConvFftKernel, ARowTransformNeedsAOneRowFilter)
{
    auto row = Fields{};
    row.th = 1;
    row.tw = 64;
    row.pb = 32;
    EXPECT_FALSE(admits(row)); // 11 x 11
    auto oneByS = largeShape();
    oneByS.filter = {1, 31};
    oneByS.prePadding = oneByS.postPadding = {0, 15};
    oneByS.input = {16, 1024};
    EXPECT_TRUE(admits(row, oneByS));
}

TEST(TestConvFftKernel, TheAdmissionFloorRefusesSmallFiltersAndSmallProblems)
{
    auto small = largeShape();
    small.filter = {3, 3};
    small.prePadding = small.postPadding = {1, 1};
    for(const auto& [th, pb, tb] :
        std::vector<std::tuple<int, int, int>>{{16, 8, 128}, {32, 2, 64}, {64, 1, 64}})
    {
        auto f = Fields{};
        f.th = f.tw = th;
        f.pb = pb;
        f.tb = tb;
        EXPECT_FALSE(admits(f, small)) << th;
    }
    auto tiny = largeShape();
    tiny.n = 1;
    tiny.c = 4;
    tiny.k = 4;
    EXPECT_FALSE(admits(Fields{}, tiny)); // 2^21 multiply-accumulates
    // 16-bit tensors need a larger advantage (the product stays fp32): 11x11 in 32x32 tiles
    // (advantage 25) is admitted, 7x7 (12) is not.
    auto half11 = largeShape();
    half11.dataType = DataType::HALF;
    auto halfFields = Fields{};
    halfFields.dtype = "HALF";
    EXPECT_TRUE(admits(halfFields, half11));
    auto half7 = half11;
    half7.filter = {7, 7};
    half7.prePadding = half7.postPadding = {3, 3};
    EXPECT_FALSE(admits(halfFields, half7));
    auto float7 = half7;
    float7.dataType = DataType::FLOAT;
    EXPECT_TRUE(admits(Fields{}, float7));
    // The floor's switch: every fitting config, for tests and calibration.
    const ScopedEnvironmentVariableSetter off("HIP_KERNEL_PROVIDER_CONV_FFT_FLOOR", "0");
    EXPECT_TRUE(admits(Fields{}, small));
    EXPECT_TRUE(admits(Fields{}, tiny));
}

TEST(TestConvFftKernel, OneDeclinePerMalformedConfigField)
{
    EXPECT_TRUE(conv_fft::configOf(kernelOf(Fields{})).has_value());
    const auto declines = [](auto mutate) {
        auto f = Fields{};
        mutate(f);
        return !conv_fft::configOf(kernelOf(f)).has_value();
    };
    EXPECT_TRUE(declines([](Fields& f) { f.th = 24; })); // not a power of two
    EXPECT_TRUE(declines([](Fields& f) { f.th = 4; })); // below 8
    EXPECT_TRUE(declines([](Fields& f) { f.tw = 128; })); // above 64
    EXPECT_TRUE(declines([](Fields& f) { f.tw = 1; }));
    EXPECT_TRUE(declines([](Fields& f) { f.pb = 0; }));
    EXPECT_TRUE(declines([](Fields& f) {
        f.pb = 16; // 16 x 544 bins exceed the LDS
        f.tb = 256;
    }));
    EXPECT_TRUE(declines([](Fields& f) { f.pb = 8; })); // 128 row pairs, 64 threads
    EXPECT_TRUE(declines([](Fields& f) {
        f.th = 1;
        f.tw = 64;
        f.pb = 3; // a row transform pairs tiles
    }));
    EXPECT_TRUE(declines([](Fields& f) { f.tb = 96; }));
    EXPECT_TRUE(declines([](Fields& f) { f.tb = 512; }));
    EXPECT_TRUE(declines([](Fields& f) { f.gn = 8; }));
    EXPECT_TRUE(declines([](Fields& f) { f.tm = 2; }));
    EXPECT_TRUE(declines([](Fields& f) {
        f.gn = 16;
        f.tm = 8; // 8 tiles a thread only with the 64-channel tile
    }));
    EXPECT_FALSE(declines([](Fields& f) { f.tm = 8; }));
}

// ---------------------------------------------------------------------------
// Plans
// ---------------------------------------------------------------------------

TEST(TestConvFftPlan, TilesCoverTheOutputOfEachDirection)
{
    const auto fwd = planOf(largeShape(), ConvDirection::FORWARD, Fields{});
    ASSERT_TRUE(fwd.has_value());
    EXPECT_EQ(fwd->args.oh, 22); // 32 - 11 + 1
    EXPECT_EQ(fwd->args.tilesH, 3);
    EXPECT_EQ(fwd->tiles, 4 * 3 * 3);
    EXPECT_EQ(fwd->args.ca, 64);
    EXPECT_EQ(fwd->args.flip, 0);
    EXPECT_EQ(fwd->chunks, 1);
    const auto bwd = planOf(largeShape(), ConvDirection::BACKWARD_DATA, Fields{});
    ASSERT_TRUE(bwd.has_value());
    EXPECT_EQ(bwd->args.filtMode, 1);
    EXPECT_EQ(bwd->args.flip, 1); // dy is correlated with the reflected filter
    auto convolution = largeShape();
    convolution.mode = ConvMode::CONVOLUTION;
    EXPECT_EQ(planOf(convolution, ConvDirection::BACKWARD_DATA, Fields{})->args.flip, 0);
    EXPECT_EQ(planOf(convolution, ConvDirection::FORWARD, Fields{})->args.flip, 1);
}

TEST(TestConvFftPlan, ChunksKeepTheWorkspaceUnderTheCap)
{
    auto wide = largeShape();
    wide.n = 64;
    wide.c = 256;
    wide.k = 256;
    wide.input = {128, 128};
    const auto plan = planOf(wide, ConvDirection::FORWARD, Fields{});
    ASSERT_TRUE(plan.has_value());
    EXPECT_GT(plan->chunks, 1);
    EXPECT_LE(plan->workspaceBytes, conv_fft::WORKSPACE_CAP_BYTES);
    EXPECT_GE(static_cast<int64_t>(plan->chunk) * plan->chunks, plan->tiles);
    EXPECT_EQ(plan->args.pcGrid, plan->chunk); // the grids' tile count, the last chunk included
}

TEST(TestConvFftPlan, DepthwiseFusesAndPartitionsBackwardWeights)
{
    // RepLKNet-style: 31x31 depthwise, 256 channels, 56x56.
    auto dw = largeShape();
    dw.n = 32;
    dw.c = 256;
    dw.k = 256;
    dw.groups = 256;
    dw.input = {56, 56};
    dw.filter = {31, 31};
    dw.prePadding = dw.postPadding = {15, 15};
    auto f = Fields{};
    f.th = f.tw = 64;
    f.pb = 1;
    const auto fwd = planOf(dw, ConvDirection::FORWARD, f);
    ASSERT_TRUE(fwd.has_value());
    EXPECT_TRUE(fwd->depthwise);
    EXPECT_EQ(fwd->chunks, 1);
    EXPECT_EQ(fwd->tiles, 32 * 2 * 2);
    EXPECT_EQ(fwd->productGridPerChunk, 256 * 32 * 2 * 2); // one fused block a tile and channel
    EXPECT_EQ(fwd->workspaceBytes, size_t{256} * 2112 * 8); // the filter spectra alone
    const auto wrw = planOf(dw, ConvDirection::BACKWARD_WEIGHTS, f);
    ASSERT_TRUE(wrw.has_value());
    EXPECT_GE(wrw->args.parts, 1);
    EXPECT_GE(static_cast<int64_t>(wrw->args.parts) * wrw->args.tilesPerPart, wrw->tiles);
    EXPECT_EQ(wrw->args.tilesPerPart % f.pb, 0);
    // Backward weights holds two spectra a tile: 8 tiles of 32x32 exceed the LDS limit there.
    auto wide = Fields{};
    wide.th = wide.tw = 32;
    wide.pb = 8;
    wide.tb = 128;
    auto dw11 = dw;
    dw11.filter = {11, 11};
    dw11.prePadding = dw11.postPadding = {5, 5};
    {
        const ScopedEnvironmentVariableSetter floorOff("HIP_KERNEL_PROVIDER_CONV_FFT_FLOOR", "0");
        EXPECT_TRUE(planOf(dw11, ConvDirection::FORWARD, wide).has_value());
        EXPECT_FALSE(planOf(dw11, ConvDirection::BACKWARD_WEIGHTS, wide).has_value());
    }
    // The product tile is meaningless without a product: the gn twins are not offered.
    auto narrow = f;
    narrow.gn = 16;
    EXPECT_FALSE(planOf(dw, ConvDirection::FORWARD, narrow).has_value());
    // The depthwise floor counts the transforms: 13x13 in 32x32 tiles (advantage 1.1) is
    // admitted, a 3x3 depthwise layer is refused.
    auto dw13 = dw;
    dw13.filter = {13, 13};
    dw13.prePadding = dw13.postPadding = {6, 6};
    auto f32 = Fields{};
    f32.th = f32.tw = 32;
    EXPECT_TRUE(planOf(dw13, ConvDirection::FORWARD, f32).has_value());
    auto small = dw;
    small.filter = {3, 3};
    small.prePadding = small.postPadding = {1, 1};
    auto sixteen = Fields{};
    sixteen.th = sixteen.tw = 16;
    sixteen.pb = 8;
    sixteen.tb = 128;
    EXPECT_FALSE(planOf(small, ConvDirection::FORWARD, sixteen).has_value());
}

TEST(TestConvFftPlan, RefusesFilterSpectraBeyondHalfTheCap)
{
    // 196 x 196 channels of 64x64 spectra: 649 MB of filter spectra, over half the 1 GiB cap
    // (room would remain for the one tile, so only this gate refuses it).
    auto wide = largeShape();
    wide.n = 1;
    wide.c = 196;
    wide.k = 196;
    wide.input = {32, 32};
    auto f64 = Fields{};
    f64.th = f64.tw = 64;
    f64.pb = 1;
    EXPECT_FALSE(planOf(wide, ConvDirection::FORWARD, f64).has_value());
}

TEST(TestConvFftPlan, RefusesFilterSpectraBeyondTheCap)
{
    auto huge = largeShape();
    huge.c = 4096;
    huge.k = 4096;
    huge.n = 1;
    huge.input = {32, 32};
    auto f = Fields{};
    f.th = f.tw = 64;
    f.pb = 1;
    EXPECT_FALSE(planOf(huge, ConvDirection::FORWARD, f).has_value());
}

} // namespace

#endif
