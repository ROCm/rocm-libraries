// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseMfmaProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvDepthwiseMfmaProblem.cpp
 * @brief Host-only checks of the matrix-core depthwise engines (CONV_DEPTHWISE_MFMA_FWD/BWD):
 *        what each direction's graph match accepts, one refusal per gated config field and per
 *        device gate, through the engines' own kernel match, and the launch sizes. No device is
 *        used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_depthwise_mfma::MfmaConfig;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD ? "hipkernel.conv_depthwise_mfma_fwd.graph_match"
                                                : "hipkernel.conv_depthwise_mfma_bwd.graph_match",
            "",
            "hipkernel.conv_depthwise_mfma.kernel_match",
            "",
            "hipkernel.conv_depthwise_mfma.dispatch",
            "",
            "",
            ""};
}

/// A config's full KMD: the MfmaConfig plus dtype and direction.
struct Fields
{
    std::string dtype = "HALF";
    std::string direction = "FWD";
    MfmaConfig config{"NHWC", 32, 3, 3, 1, 4, 2, 1, 16};
};

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const Fields& f)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.name = "conv_depthwise_mfma_test";
    kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
    kernel.source.sourceFile = "kernels/ConvDepthwiseMfma.cpp";
    kernel.source.entryPoint = "ConvDepthwiseMfma";
    const auto& c = f.config;
    kernel.metadata = {{"dtype", f.dtype},
                       {"direction", f.direction},
                       {"layout", c.layout},
                       {"atom", c.atom},
                       {"kh", c.kh},
                       {"kw", c.kw},
                       {"stride", c.stride},
                       {"waves", c.waves},
                       {"depth", c.depth},
                       {"fold", c.fold},
                       {"rows", c.rows}};
    return kernel;
}

/// NHWC depthwise, unit stride, padding filter / 2.
ConvShape depthwise(int64_t channels,
                    std::vector<int64_t> input,
                    std::vector<int64_t> filter,
                    DataType dataType = DataType::HALF)
{
    ConvShape shape;
    shape.n = 2;
    shape.c = shape.k = shape.groups = channels;
    shape.input = std::move(input);
    shape.filter = filter;
    shape.prePadding.clear();
    for(const auto extent : filter)
    {
        shape.prePadding.push_back(extent / 2);
    }
    shape.postPadding = shape.prePadding;
    shape.stride.assign(shape.input.size(), 1);
    shape.dilation.assign(shape.input.size(), 1);
    shape.channelsLast = true;
    shape.dataType = dataType;
    return shape;
}

ConvShape baseShape()
{
    return depthwise(32, {9, 20}, {3, 3});
}

hipdnn_plugin_sdk::ingestor::DeviceProperties device(const std::string& arch)
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = 64;
    return properties;
}

ConvDirection directionOf(const Fields& fields)
{
    return fields.direction == "BWD" ? ConvDirection::BACKWARD_DATA : ConvDirection::FORWARD;
}

bool graphAccepts(const ConvShape& shape, ConvDirection direction)
{
    const GraphFixture fixture(buildGraph(specFor(shape, direction)), device("gfx950"));
    return matchesGraph(packFor(direction), fixture.context()).has_value();
}

/// The engine's graph match and kernel match, as the ingestor runs them.
bool admits(const Fields& fields,
            const ConvShape& shape = baseShape(),
            const std::string& arch = "gfx950:sramecc+:xnack-")
{
    const auto direction = directionOf(fields);
    const auto pack = packFor(direction);
    const GraphFixture fixture(buildGraph(specFor(shape, direction)), device(arch));
    const auto bound = matchesGraph(pack, fixture.context());
    return bound.has_value() && matchesKernel(pack, fixture.context(), kernelOf(fields), *bound);
}

std::optional<conv_depthwise_mfma::MfmaLaunch> launchOf(const Fields& fields,
                                                        const ConvShape& shape)
{
    const auto direction = directionOf(fields);
    const GraphFixture fixture(buildGraph(specFor(shape, direction)), device("gfx950"));
    const auto problem
        = conv_depthwise_valu::parseDepthwiseValuProblem(fixture.context(), direction);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    return conv_depthwise_mfma::mfmaLaunchFor(*problem, fields.config);
}

TEST(TestConvDepthwiseMfmaGraphMatch, ServesUnitStrideChannelsLast16Bit)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        EXPECT_TRUE(graphAccepts(baseShape(), direction));
        EXPECT_TRUE(graphAccepts(depthwise(32, {9, 20}, {3, 3}, DataType::BFLOAT16), direction));
        // 1-D runs as height 1.
        EXPECT_TRUE(graphAccepts(depthwise(32, {40}, {5}), direction));
        auto channelsFirst = baseShape();
        channelsFirst.channelsLast = false;
        EXPECT_TRUE(graphAccepts(channelsFirst, direction));
    }
}

TEST(TestConvDepthwiseMfmaGraphMatch, ServesStrideTwoForwardOnly)
{
    auto strided = baseShape();
    strided.stride = {2, 2};
    EXPECT_TRUE(graphAccepts(strided, ConvDirection::FORWARD));
    // Strided backward data is a gather (the VALU engines' phases), not this forward form.
    EXPECT_FALSE(graphAccepts(strided, ConvDirection::BACKWARD_DATA));
}

TEST(TestConvDepthwiseMfmaGraphMatch, DeclinesWhatTheKernelDoesNotRun)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        EXPECT_FALSE(graphAccepts(depthwise(32, {9, 20}, {3, 3}, DataType::FLOAT), direction));
        auto mixed = baseShape();
        mixed.stride = {1, 2};
        EXPECT_FALSE(graphAccepts(mixed, direction));
        auto stride3 = baseShape();
        stride3.stride = {3, 3};
        EXPECT_FALSE(graphAccepts(stride3, direction));
        auto dilated = baseShape();
        dilated.dilation = {1, 2};
        EXPECT_FALSE(graphAccepts(dilated, direction));
        EXPECT_FALSE(graphAccepts(depthwise(32, {5, 6, 7}, {3, 3, 3}), direction));
        auto grouped = baseShape();
        grouped.groups = 16;
        EXPECT_FALSE(graphAccepts(grouped, direction));
    }
}

TEST(TestConvDepthwiseMfmaKernelMatch, AdmitsTheBaseConfigBothDirectionsAndDtypes)
{
    for(const std::string direction : {"FWD", "BWD"})
    {
        for(const std::string dtype : {"HALF", "BFLOAT16"})
        {
            Fields fields;
            fields.direction = direction;
            fields.dtype = dtype;
            const auto shape = depthwise(
                32, {9, 20}, {3, 3}, dtype == "HALF" ? DataType::HALF : DataType::BFLOAT16);
            EXPECT_TRUE(admits(fields, shape)) << direction << " " << dtype;
        }
    }
}

TEST(TestConvDepthwiseMfmaKernelMatch, DeclinesAnotherDtype)
{
    Fields fields;
    fields.dtype = "BFLOAT16";
    EXPECT_FALSE(admits(fields));
}

TEST(TestConvDepthwiseMfmaKernelMatch, DeclinesAnUnknownDirection)
{
    Fields fields;
    fields.direction = "WRW";
    const auto pack = packFor(ConvDirection::FORWARD);
    const GraphFixture fixture(buildGraph(specFor(baseShape(), ConvDirection::FORWARD)),
                               device("gfx950"));
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(matchesKernel(pack, fixture.context(), kernelOf(fields), *bound));
}

ConvShape channelsFirst(int64_t channels = 32)
{
    auto shape = depthwise(channels, {9, 20}, {3, 3});
    shape.channelsLast = false;
    return shape;
}

TEST(TestConvDepthwiseMfmaKernelMatch, LayoutMustMatchTheProblems)
{
    Fields fields;
    fields.config.layout = "NCHW";
    EXPECT_FALSE(admits(fields));
    EXPECT_TRUE(admits(fields, channelsFirst()));
    fields.config.layout = "NHWC";
    EXPECT_FALSE(admits(fields, channelsFirst()));
    fields.config.layout = "NDHWC";
    EXPECT_FALSE(admits(fields));
    EXPECT_FALSE(admits(fields, channelsFirst()));
}

TEST(TestConvDepthwiseMfmaKernelMatch, ChannelsFirstTakesAnyChannelCount)
{
    // A wave owns a channel pair; an odd count leaves the last pair half empty.
    Fields fields;
    fields.config.layout = "NCHW";
    fields.config.waves = 4;
    EXPECT_TRUE(admits(fields, channelsFirst(3)));
    EXPECT_TRUE(admits(fields, channelsFirst(5)));
    EXPECT_TRUE(admits(fields, channelsFirst(18)));
    fields.config.waves = 3;
    EXPECT_FALSE(admits(fields, channelsFirst(6)));
}

TEST(TestConvDepthwiseMfmaKernelMatch, ChannelsFirstFoldsUpToEightImages)
{
    Fields fields;
    fields.config.layout = "NCHW";
    for(const int64_t fold : {1, 2, 4, 8})
    {
        fields.config.fold = fold;
        EXPECT_TRUE(admits(fields, channelsFirst())) << fold;
    }
    for(const int64_t fold : {0, 3, 16})
    {
        fields.config.fold = fold;
        EXPECT_FALSE(admits(fields, channelsFirst())) << fold;
    }
    // Channels-last stops at 4.
    Fields last;
    last.config.fold = 8;
    EXPECT_FALSE(admits(last));
}

TEST(TestConvDepthwiseMfmaKernelMatch, AtomGatesTheDevice)
{
    Fields x32;
    Fields x16;
    x16.config.atom = 16;
    EXPECT_TRUE(admits(x32, baseShape(), "gfx950"));
    // gfx94x runs K = 32 as the sparse MFMA: the NHWC core's taps compress, the NCHW core's
    // dense input does not.
    for(const std::string arch : {"gfx942:sramecc+:xnack-", "gfx941", "gfx940"})
    {
        EXPECT_TRUE(admits(x32, baseShape(), arch)) << arch;
    }
    Fields x32First = x32;
    x32First.config.layout = "NCHW";
    EXPECT_TRUE(admits(x32First, channelsFirst(), "gfx950"));
    EXPECT_FALSE(admits(x32First, channelsFirst(), "gfx942"));
    EXPECT_FALSE(admits(x32, baseShape(), "gfx90a"));
    for(const std::string arch : {"gfx950", "gfx942", "gfx941", "gfx940", "gfx90a", "gfx908"})
    {
        EXPECT_TRUE(admits(x16, baseShape(), arch)) << arch;
    }
    for(const std::string arch : {"gfx906", "gfx1100", "gfx1201", "gfx000"})
    {
        EXPECT_FALSE(admits(x16, baseShape(), arch)) << arch;
        EXPECT_FALSE(admits(x32, baseShape(), arch)) << arch;
    }
    // gfx908 has no bf16 MFMA.
    Fields bf16 = x16;
    bf16.dtype = "BFLOAT16";
    const auto bf16Shape = depthwise(32, {9, 20}, {3, 3}, DataType::BFLOAT16);
    EXPECT_TRUE(admits(bf16, bf16Shape, "gfx90a"));
    EXPECT_FALSE(admits(bf16, bf16Shape, "gfx908"));
}

TEST(TestConvDepthwiseMfmaKernelMatch, DeclinesAnUnknownAtom)
{
    Fields fields;
    fields.config.atom = 8;
    EXPECT_FALSE(admits(fields));
    fields.config.atom = 64;
    EXPECT_FALSE(admits(fields));
}

TEST(TestConvDepthwiseMfmaKernelMatch, FilterHeightMustMatch)
{
    Fields fields;
    fields.config.kh = 5;
    EXPECT_FALSE(admits(fields));
    EXPECT_TRUE(admits(fields, depthwise(32, {9, 20}, {5, 3})));
}

TEST(TestConvDepthwiseMfmaKernelMatch, FilterWidthMustMatch)
{
    Fields fields;
    fields.config.kw = 5;
    EXPECT_FALSE(admits(fields));
    EXPECT_TRUE(admits(fields, depthwise(32, {9, 20}, {3, 5})));
}

TEST(TestConvDepthwiseMfmaKernelMatch, StrideMustMatchTheProblems)
{
    auto strided = baseShape();
    strided.stride = {2, 2};
    Fields fields;
    fields.config.stride = 2;
    EXPECT_TRUE(admits(fields, strided));
    EXPECT_FALSE(admits(fields));
    fields.config.stride = 1;
    EXPECT_FALSE(admits(fields, strided));
    for(const int64_t stride : {0, 3})
    {
        fields.config.stride = stride;
        EXPECT_FALSE(admits(fields)) << stride;
    }
}

TEST(TestConvDepthwiseMfmaKernelMatch, WavesArePowersOfTwoDividingTheChannels)
{
    Fields fields;
    for(const int64_t waves : {1, 2, 4})
    {
        fields.config.waves = waves;
        EXPECT_TRUE(admits(fields)) << waves;
    }
    fields.config.waves = 3;
    EXPECT_FALSE(admits(fields, depthwise(48, {9, 20}, {3, 3})));
    fields.config.waves = 32;
    EXPECT_FALSE(admits(fields, depthwise(256, {9, 20}, {3, 3})));
    fields.config.waves = 16;
    EXPECT_TRUE(admits(fields, depthwise(128, {9, 20}, {3, 3})));
    // 8 channels per wave: 32 channels hold 4 waves, not 8.
    fields.config.waves = 8;
    EXPECT_FALSE(admits(fields));
    fields.config.waves = 1;
    EXPECT_FALSE(admits(fields, depthwise(12, {9, 20}, {3, 3})));
}

TEST(TestConvDepthwiseMfmaKernelMatch, DepthIsOneToFour)
{
    Fields fields;
    for(const int64_t depth : {1, 4})
    {
        fields.config.depth = depth;
        EXPECT_TRUE(admits(fields)) << depth;
    }
    for(const int64_t depth : {0, 5})
    {
        fields.config.depth = depth;
        EXPECT_FALSE(admits(fields)) << depth;
    }
}

TEST(TestConvDepthwiseMfmaKernelMatch, FoldIsOneTwoOrFour)
{
    Fields fields;
    for(const int64_t fold : {1, 2, 4})
    {
        fields.config.fold = fold;
        EXPECT_TRUE(admits(fields)) << fold;
    }
    for(const int64_t fold : {0, 3, 8})
    {
        fields.config.fold = fold;
        EXPECT_FALSE(admits(fields)) << fold;
    }
}

TEST(TestConvDepthwiseMfmaKernelMatch, RowsArePositive)
{
    Fields fields;
    fields.config.rows = 1;
    EXPECT_TRUE(admits(fields));
    fields.config.rows = 0;
    EXPECT_FALSE(admits(fields));
    fields.config.rows = -4;
    EXPECT_FALSE(admits(fields));
}

TEST(TestConvDepthwiseMfmaKernelMatch, DeclinesAnLdsOverflow)
{
    // A 1 x 31 filter, 16 waves, 4 images: 4 x (8 + 32) staged columns, 2 x (160 + 32) x 16 x 16
    // bytes = 98 KiB.
    Fields fields;
    fields.config.kh = 1;
    fields.config.kw = 31;
    fields.config.waves = 16;
    fields.config.fold = 4;
    const auto shape = depthwise(128, {3, 40}, {1, 31});
    EXPECT_GT(conv_depthwise_mfma::mfmaLdsBytes(fields.config), 65536);
    EXPECT_FALSE(admits(fields, shape));
    fields.config.fold = 1;
    EXPECT_LE(conv_depthwise_mfma::mfmaLdsBytes(fields.config), 65536);
    EXPECT_TRUE(admits(fields, shape));
}

TEST(TestConvDepthwiseMfmaKernelMatch, FoldedImagesMustFitTheBufferOffsets)
{
    // 384 x 384 x 2048 fp16 = 576 MiB an image: two fit 2^31 - 1 bytes, four do not.
    Fields fields;
    fields.config.waves = 8;
    fields.config.fold = 2;
    const auto shape = depthwise(2048, {384, 384}, {3, 3});
    EXPECT_TRUE(admits(fields, shape));
    fields.config.fold = 4;
    EXPECT_FALSE(admits(fields, shape));
}

TEST(TestConvDepthwiseMfmaLaunch, SizesTheGridAndCarriesTheProblem)
{
    Fields fields;
    fields.config.fold = 2;
    fields.config.rows = 4;
    auto shape = depthwise(64, {9, 20}, {3, 3});
    shape.n = 3;
    const auto launch = launchOf(fields, shape);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->block, 256);
    // 2 image groups x 3 row chunks x 2 column tiles (16 wide) x 2 channel blocks.
    EXPECT_EQ(launch->grid, 2 * 3 * 2 * 2);
    EXPECT_EQ(launch->args.rows, 4);
    EXPECT_EQ(launch->args.chunks, 3);
    EXPECT_EQ(launch->args.n, 3);
    EXPECT_EQ(launch->args.c, 64);
    EXPECT_EQ(launch->args.hi, 9);
    EXPECT_EQ(launch->args.wi, 20);
    EXPECT_EQ(launch->args.ho, 9);
    EXPECT_EQ(launch->args.wo, 20);
    EXPECT_EQ(launch->args.ph, 1);
    EXPECT_EQ(launch->args.pw, 1);
    EXPECT_FALSE(launch->flip);
}

TEST(TestConvDepthwiseMfmaLaunch, StrideTwoStagesTwiceTheColumns)
{
    Fields fields;
    fields.config.stride = 2;
    auto shape = depthwise(64, {17, 40}, {3, 3});
    shape.stride = {2, 2};
    const auto launch = launchOf(fields, shape);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->args.ho, 9);
    EXPECT_EQ(launch->args.wo, 20);
    // 2 images x 1 row chunk x 1 column tile x 2 channel blocks.
    EXPECT_EQ(launch->grid, 2 * 1 * 1 * 2);
    // 2 x 32 + 4 x 2 staged columns and 32 output columns, two of each, 16 B per wave.
    EXPECT_EQ(conv_depthwise_mfma::mfmaLdsBytes(fields.config), 2 * (72 + 32) * 4 * 16);
}

TEST(TestConvDepthwiseMfmaLaunch, ChannelsFirstGridsChannelPairsAndClaimsNoLds)
{
    Fields fields;
    fields.config.layout = "NCHW";
    fields.config.waves = 2;
    fields.config.fold = 4;
    fields.config.rows = 4;
    auto shape = channelsFirst(7);
    shape.n = 5;
    shape.input = {9, 40};
    const auto launch = launchOf(fields, shape);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->block, 128);
    // 2 image groups x 3 row chunks x 2 column tiles (32 wide) x 2 blocks of 2 channel pairs.
    EXPECT_EQ(launch->grid, 2 * 3 * 2 * 2);
    EXPECT_EQ(conv_depthwise_mfma::mfmaLdsBytes(fields.config), 0);
}

TEST(TestConvDepthwiseMfmaLaunch, BackwardDataRunsTheReflectedForwardForm)
{
    Fields fields;
    fields.direction = "BWD";
    auto shape = depthwise(32, {9, 20}, {3, 3});
    shape.prePadding = {0, 2};
    shape.postPadding = {2, 0};
    const auto launch = launchOf(fields, shape);
    ASSERT_TRUE(launch.has_value());
    EXPECT_TRUE(launch->flip);
    // Mirrored padding: filter - 1 - pre.
    EXPECT_EQ(launch->args.ph, 2);
    EXPECT_EQ(launch->args.pw, 0);
    auto convolution = depthwise(32, {9, 20}, {3, 3});
    convolution.mode = ConvMode::CONVOLUTION;
    const auto reflected = launchOf(fields, convolution);
    ASSERT_TRUE(reflected.has_value());
    EXPECT_FALSE(reflected->flip);
}

TEST(TestConvDepthwiseMfmaComputeType, ServesCudnnsComputeTypesPerElementType)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        expectCudnnComputeTypes(
            baseShape(), {DataType::HALF, DataType::BFLOAT16}, [direction](const ConvShape& s) {
                return graphAccepts(s, direction);
            });
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
