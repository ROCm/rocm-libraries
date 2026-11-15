// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hiprtc.h>

#include "compilation/KernelCompileOptions.hpp"
#include "kernel_includes.hpp"
#include "kernel_sources.hpp"

#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvDepthwiseValuProblem.cpp
 * @brief Host-only checks of the VALU depthwise engines' acceptance rules: what each direction
 *        accepts and the forward convolution it runs (layout, rank, mirrored padding, filter
 *        reflection), one refusal per rule beyond the shared naive parse, and the launch
 *        sizes. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_depthwise_valu::DepthwiseLayout;
using conv_depthwise_valu::DepthwiseTile;
using conv_depthwise_valu::DepthwiseVariant;
using conv_depthwise_valu::launchFor;
using conv_depthwise_valu::parseDepthwiseValuProblem;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;

std::optional<conv_depthwise_valu::DepthwiseValuProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseDepthwiseValuProblem(fixture.context(), spec.direction);
}

std::optional<conv_depthwise_valu::DepthwiseValuProblem> parse(const ConvShape& shape,
                                                               ConvDirection direction)
{
    return parse(specFor(shape, direction));
}

/// A depthwise 3x3, padding 1, over @p channels channels.
ConvShape depthwise(int64_t channels = 5, bool channelsLast = true)
{
    ConvShape shape;
    shape.c = shape.k = shape.groups = channels;
    shape.input = {8, 9};
    shape.filter = {3, 3};
    shape.prePadding = shape.postPadding = {1, 1};
    shape.channelsLast = channelsLast;
    return shape;
}

TEST(TestConvDepthwiseValuProblem, ForwardReadsXIntoY)
{
    for(const bool last : {true, false})
    {
        const auto problem = parse(depthwise(5, last), ConvDirection::FORWARD);
        ASSERT_TRUE(problem.has_value()) << "channels last " << last;
        EXPECT_EQ(problem->layout,
                  last ? DepthwiseLayout::CHANNELS_LAST : DepthwiseLayout::CHANNELS_FIRST);
        EXPECT_EQ(problem->ndims, 2);
        EXPECT_EQ(problem->inUid, X_UID);
        EXPECT_EQ(problem->outUid, Y_UID);
        EXPECT_EQ(problem->args.hi, 8);
        EXPECT_EQ(problem->args.ho, 8);
        EXPECT_EQ(problem->args.ph, 1);
        EXPECT_FALSE(problem->flipFilter);
        EXPECT_TRUE(problem->unitStrideAndDilation);
        // The depth axis of a 2-D problem is degenerate.
        EXPECT_EQ(problem->args.di, 1);
        EXPECT_EQ(problem->args.kd, 1);
    }
}

TEST(TestConvDepthwiseValuProblem, BackwardDataRunsDyIntoDxWithMirroredPaddingAndAReflectedFilter)
{
    auto shape = depthwise();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 2};
    shape.dilation = {1, 2};
    const auto problem = parse(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->inUid, Y_UID);
    EXPECT_EQ(problem->outUid, X_UID);
    const auto output = shape.output();
    EXPECT_EQ(problem->args.hi, output[0]);
    EXPECT_EQ(problem->args.wi, output[1]);
    EXPECT_EQ(problem->args.ho, 8);
    EXPECT_EQ(problem->args.wo, 9);
    // pre' = dilation * (filter - 1) - pre
    EXPECT_EQ(problem->args.ph, 1 * 2 - 2);
    EXPECT_EQ(problem->args.pw, 2 * 2 - 0);
    EXPECT_TRUE(problem->flipFilter);
    EXPECT_FALSE(problem->unitStrideAndDilation);
}

TEST(TestConvDepthwiseValuProblem, ConvolutionModeReflectsTheFilterOnceMore)
{
    auto shape = depthwise();
    shape.mode = ConvMode::CONVOLUTION;
    const auto forward = parse(shape, ConvDirection::FORWARD);
    const auto backward = parse(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(forward.has_value());
    ASSERT_TRUE(backward.has_value());
    EXPECT_TRUE(forward->flipFilter);
    EXPECT_FALSE(backward->flipFilter);
}

TEST(TestConvDepthwiseValuProblem, OneDRunsAsTwoDAndThreeDAsThreeD)
{
    auto oneD = depthwise();
    oneD.input = {11};
    oneD.filter = {3};
    oneD.prePadding = oneD.postPadding = {1};
    oneD.stride = oneD.dilation = {1};
    const auto one = parse(oneD, ConvDirection::FORWARD);
    ASSERT_TRUE(one.has_value());
    EXPECT_EQ(one->ndims, 2);
    EXPECT_EQ(one->args.hi, 1);
    EXPECT_EQ(one->args.kh, 1);
    EXPECT_EQ(one->args.wi, 11);

    auto threeD = depthwise();
    threeD.input = {4, 5, 6};
    threeD.filter = {3, 3, 3};
    threeD.prePadding = threeD.postPadding = {1, 1, 1};
    threeD.stride = threeD.dilation = {1, 1, 1};
    for(const bool last : {true, false})
    {
        threeD.channelsLast = last;
        const auto three = parse(threeD, ConvDirection::FORWARD);
        ASSERT_TRUE(three.has_value());
        EXPECT_EQ(three->ndims, 3);
        EXPECT_EQ(three->args.di, 4);
        EXPECT_EQ(three->args.kd, 3);
    }
}

TEST(TestConvDepthwiseValuProblem, DeclinesWhatIsNotDepthwise)
{
    auto multiplier = depthwise();
    multiplier.k = 10; // two outputs per channel
    auto dense = depthwise();
    dense.groups = 1;
    dense.k = 5;
    auto grouped = depthwise(6);
    grouped.groups = 3;
    for(const auto& shape : {multiplier, dense, grouped})
    {
        EXPECT_FALSE(parse(shape, ConvDirection::FORWARD).has_value());
    }
}

TEST(TestConvDepthwiseValuProblem, DeclinesBackwardWeights)
{
    EXPECT_FALSE(parse(depthwise(), ConvDirection::BACKWARD_WEIGHTS).has_value());
}

TEST(TestConvDepthwiseValuProblem, StridedBackwardDataIsGatherOnly)
{
    auto shape = depthwise();
    shape.stride = {2, 1};
    const auto forward = parse(shape, ConvDirection::FORWARD);
    ASSERT_TRUE(forward.has_value());
    EXPECT_FALSE(forward->gatherOnly);
    const auto backward = parse(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(backward.has_value());
    EXPECT_TRUE(backward->gatherOnly);
    // Gather terms: dy in, dx out, the forward convolution's padding and stride.
    const auto& g = backward->gatherArgs;
    EXPECT_EQ(g.hi, static_cast<int>(backward->base.args.outSpatial[1]));
    EXPECT_EQ(g.ho, static_cast<int>(backward->base.args.inSpatial[1]));
    EXPECT_EQ(g.ph, static_cast<int>(backward->base.args.prePadding[1]));
    EXPECT_EQ(g.sh, 2);

    shape.stride = {1, 1};
    const auto unit = parse(shape, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(unit.has_value());
    EXPECT_FALSE(unit->gatherOnly);
}

TEST(TestConvDepthwiseValuProblem, DeclinesMixedLayouts)
{
    auto spec = specFor(depthwise(5, true), ConvDirection::FORWARD);
    spec.y.strides = packedStrides(spec.y.dims, false);
    EXPECT_FALSE(parse(spec).has_value());
}

TEST(TestConvDepthwiseValuProblem, DeclinesUnpackedTensors)
{
    for(const char* operand : {"x", "w", "y"})
    {
        auto spec = specFor(depthwise(5, true), ConvDirection::FORWARD);
        const std::string name(operand);
        auto& tensor = [&]() -> auto& {
            if(name == "x")
            {
                return spec.x;
            }
            if(name == "w")
            {
                return spec.w;
            }
            return spec.y;
        }();
        tensor.strides[0] += 3; // a gap after every image (filter: every channel)
        EXPECT_FALSE(parse(spec).has_value()) << operand;
    }
}

TEST(TestConvDepthwiseValuProblem, AcceptsAnyStrideOnExtentOneAxes)
{
    auto shape = depthwise(1, true);
    shape.n = 1;
    auto spec = specFor(shape, ConvDirection::FORWARD);
    spec.x.strides[0] = 12345; // n and c are extent 1: never read through
    spec.x.strides[1] = 7;
    EXPECT_TRUE(parse(spec).has_value());
}

TEST(TestConvDepthwiseValuProblem, LaunchCoversTheProblemPerVariant)
{
    const auto problem = parse(depthwise(40, true), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());

    DepthwiseTile floor;
    floor.wstrip = 4;
    const auto floorLaunch = launchFor(*problem, floor);
    ASSERT_TRUE(floorLaunch.has_value());
    EXPECT_EQ(floorLaunch->block, 256);
    // n * ho * ceil(wo / 4) * c threads: 2 * 8 * 3 * 40 = 1920 -> 8 blocks.
    EXPECT_EQ(floorLaunch->grid, 8);

    DepthwiseTile fused;
    fused.variant = DepthwiseVariant::FUSED;
    fused.bh = fused.bw = 8;
    fused.bk = 32;
    fused.rh = fused.rw = 2;
    const auto fusedLaunch = launchFor(*problem, fused);
    ASSERT_TRUE(fusedLaunch.has_value());
    EXPECT_EQ(fusedLaunch->block, 4 * 4 * 32);
    // n * ceil(ho / 8) * ceil(wo / 8) * ceil(c / 32) = 2 * 1 * 2 * 2.
    EXPECT_EQ(fusedLaunch->grid, 8);

    const auto nchw = parse(depthwise(40, false), ConvDirection::FORWARD);
    ASSERT_TRUE(nchw.has_value());
    const auto nchwLaunch = launchFor(*nchw, DepthwiseTile{});
    ASSERT_TRUE(nchwLaunch.has_value());
    // One thread per output element: 2 * 40 * 8 * 9 = 5760 -> 23 blocks.
    EXPECT_EQ(nchwLaunch->grid, 23);
    EXPECT_EQ(nchwLaunch->gridY, 1);
    EXPECT_EQ(nchwLaunch->gridZ, 1);

    DepthwiseTile rowtile;
    rowtile.variant = DepthwiseVariant::ROWTILE;
    rowtile.kh = rowtile.kw = 3;
    rowtile.th = 4;
    rowtile.tw = 2;
    rowtile.vec = 2;
    rowtile.stride = 1;
    const auto rowtileLaunch = launchFor(*problem, rowtile);
    ASSERT_TRUE(rowtileLaunch.has_value());
    // 64 lanes over the channel groups of one spatial tile: (ceil(wo / 2), ceil(c / 2 / 64),
    // n * ceil(ho / 4)) = (5, 1, 4).
    EXPECT_EQ(rowtileLaunch->block, 64);
    EXPECT_EQ(rowtileLaunch->grid, 5);
    EXPECT_EQ(rowtileLaunch->gridY, 1);
    EXPECT_EQ(rowtileLaunch->gridZ, 4);
    EXPECT_FALSE(launchFor(*nchw, rowtile).has_value());
}

// The tile variants (microtile, fused, LDS) have only the 32-bit index form: they decline once x
// or y reaches 2^31 bytes, where the floors (a 64-bit body) and the rowtile (rebased per row)
// still serve.
TEST(TestConvDepthwiseValuProblem, TileVariantsDeclineTensorsPastThe32BitForm)
{
    auto shape = depthwise(64, true);
    shape.input = {1024, 1024};
    shape.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    // x and y: n * 64 * 2^20 halves = n * 2^27 bytes.
    shape.n = 15;
    const auto below = parse(shape, ConvDirection::FORWARD);
    shape.n = 16;
    const auto at = parse(shape, ConvDirection::FORWARD);
    ASSERT_TRUE(below.has_value());
    ASSERT_TRUE(at.has_value());

    DepthwiseTile microtile;
    microtile.variant = DepthwiseVariant::MICROTILE;
    microtile.kh = microtile.kw = 3;
    microtile.th = 2;
    microtile.tw = 4;
    DepthwiseTile fused;
    fused.variant = DepthwiseVariant::FUSED;
    fused.kh = fused.kw = 3;
    fused.bh = fused.bw = 8;
    fused.bk = 32;
    fused.rh = fused.rw = 2;
    DepthwiseTile lds = fused;
    lds.variant = DepthwiseVariant::LDS;
    for(const auto& tile : {microtile, fused, lds})
    {
        EXPECT_TRUE(launchFor(*below, tile).has_value()) << static_cast<int>(tile.variant);
        EXPECT_FALSE(launchFor(*at, tile).has_value()) << static_cast<int>(tile.variant);
    }
    DepthwiseTile floor;
    floor.wstrip = 4;
    DepthwiseTile rowtile;
    rowtile.variant = DepthwiseVariant::ROWTILE;
    rowtile.kh = rowtile.kw = 3;
    rowtile.th = rowtile.tw = 4;
    rowtile.vec = 2;
    rowtile.stride = 1;
    for(const auto& tile : {floor, rowtile})
    {
        EXPECT_TRUE(launchFor(*at, tile).has_value()) << static_cast<int>(tile.variant);
    }
}

TEST(TestConvDepthwiseValuProblem, LaunchDeclinesABlockOverTheLimit)
{
    const auto problem = parse(depthwise(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem.has_value());
    DepthwiseTile huge;
    huge.variant = DepthwiseVariant::LDS;
    huge.bh = huge.bw = 32;
    huge.bk = 32;
    huge.rh = huge.rw = 1;
    EXPECT_FALSE(launchFor(*problem, huge).has_value());
}

TEST(TestConvDepthwiseValuComputeType, ServesCudnnsComputeTypesPerElementType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        expectCudnnComputeTypes(
            depthwise(),
            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
            [direction](const ConvShape& s) { return parse(s, direction).has_value(); });
    }
}

// ---------------------------------------------------------------------------
// Fused pointwise epilogues: forward only, and only when the engines ask for it.
// ---------------------------------------------------------------------------

/// A backward-data depthwise convolution whose dx is the virtual head of a relu, field for field.
flatbuffers::FlatBufferBuilder backwardFusedGraph(const ConvShape& shape)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    auto spec = specFor(shape, ConvDirection::BACKWARD_DATA);
    const TensorSpec written = spec.x;
    spec.x.isVirtual = true;
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensors{
        serializeTensor(builder, spec.xUid, spec.x),
        serializeTensor(builder, spec.wUid, spec.w),
        serializeTensor(builder, spec.yUid, spec.y),
        serializeTensor(builder, 100, written)};
    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(
        builder,
        "conv",
        data_objects::DataType::FLOAT,
        data_objects::NodeAttributes::ConvolutionBwdAttributes,
        data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                           spec.yUid,
                                                           spec.wUid,
                                                           spec.xUid,
                                                           &spec.prePadding,
                                                           &spec.postPadding,
                                                           &spec.stride,
                                                           &spec.dilation,
                                                           spec.mode)
            .Union()));
    nodes.push_back(data_objects::CreateNodeDirect(
        builder,
        "relu",
        data_objects::DataType::FLOAT,
        data_objects::NodeAttributes::PointwiseAttributes,
        data_objects::CreatePointwiseAttributes(builder,
                                                data_objects::PointwiseMode::RELU_FWD,
                                                ::flatbuffers::nullopt,
                                                ::flatbuffers::nullopt,
                                                ::flatbuffers::nullopt,
                                                ::flatbuffers::nullopt,
                                                spec.xUid,
                                                ::flatbuffers::nullopt,
                                                ::flatbuffers::nullopt,
                                                100)
            .Union()));
    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "dw_bwd_fused",
                                                   data_objects::DataType::UNSET,
                                                   data_objects::DataType::UNSET,
                                                   data_objects::DataType::UNSET,
                                                   &tensors,
                                                   &nodes));
    return builder;
}

TEST(TestConvDepthwiseValuProblemFusion, ForwardTakesTheEpilogueOnlyWhenAsked)
{
    namespace epi = conv_epilogue_test;
    for(const bool channelsLast : {true, false})
    {
        const auto spec = epi::fusedSpecFor(depthwise(6, channelsLast),
                                            {epi::bias(), epi::relu()},
                                            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
        const GraphFixture fixture(epi::buildFusedGraph(spec));
        const auto fused = parseDepthwiseValuProblem(
            fixture.context(), ConvDirection::FORWARD, conv_naive::Fusion::EPILOGUE);
        ASSERT_TRUE(fused.has_value());
        EXPECT_EQ(conv_epilogue::patternOf(fused->base.epilogue), "add.k.FLOAT|relu");
        // The kernels write the chain's output.
        EXPECT_EQ(fused->outUid, epi::outputUid(2));
        EXPECT_FALSE(
            parseDepthwiseValuProblem(fixture.context(), ConvDirection::FORWARD).has_value());
    }
}

TEST(TestConvDepthwiseValuProblemFusion, BackwardDataDeclinesAFusedGraph)
{
    for(const bool channelsLast : {true, false})
    {
        const auto shape = depthwise(6, channelsLast);
        ASSERT_TRUE(parse(shape, ConvDirection::BACKWARD_DATA).has_value());
        const GraphFixture fixture(backwardFusedGraph(shape));
        EXPECT_FALSE(parseDepthwiseValuProblem(fixture.context(),
                                               ConvDirection::BACKWARD_DATA,
                                               conv_naive::Fusion::EPILOGUE)
                         .has_value());
        EXPECT_FALSE(
            parseDepthwiseValuProblem(fixture.context(), ConvDirection::BACKWARD_DATA).has_value());
    }
}

// ---------------------------------------------------------------------------
// The compiler workaround in ConvDepthwiseEpilogue.h: fused fp16 VALU configs that fail
// instruction selection without the barrier ("Cannot select: f32 = Constant<-1>") compile
// through HIPRTC. Compile only: no device is used.
// ---------------------------------------------------------------------------

struct FusedCompileCase
{
    std::string name;
    std::string arch;
    /// The config's CONV_DW_* defines beyond the element type, as the pack sets them.
    std::vector<std::pair<std::string, std::string>> defines;
    std::vector<conv_epilogue_test::Step> steps;
};

std::vector<FusedCompileCase> fusedCompileCases()
{
    namespace epi = conv_epilogue_test;
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    const std::vector<epi::Step> geluTanh{epi::unary(PointwiseMode::GELU_APPROX_TANH_FWD)};
    epi::Step scale = epi::scale();
    scale.operand = epi::OperandKind::RUNTIME;
    epi::Step shift = epi::bias();
    shift.operand = epi::OperandKind::CONSTANT;
    const std::vector<epi::Step> scaleBiasSilu{scale, shift, epi::unary(PointwiseMode::SWISH_FWD)};
    const std::vector<epi::Step> biasRelu{
        epi::bias(hipdnn_flatbuffers_sdk::data_objects::DataType::HALF), epi::relu()};
    // Each case failed to compile without the barrier on that arch.
    return {
        {"MicrotileGeluTanh",
         "gfx90a",
         {{"VARIANT", "1"}, {"LAYOUT", "0"}, {"KH", "3"}, {"KW", "3"}, {"TH", "4"}, {"TW", "4"}},
         geluTanh},
        {"LdsChannelsFirstGeluTanh",
         "gfx90a",
         {{"VARIANT", "3"},
          {"LAYOUT", "1"},
          {"KH", "3"},
          {"KW", "3"},
          {"BH", "16"},
          {"BW", "64"},
          {"BK", "1"},
          {"RH", "2"},
          {"RW", "4"}},
         geluTanh},
        {"RowtileGeluTanh",
         "gfx90a",
         {{"VARIANT", "6"},
          {"LAYOUT", "0"},
          {"KH", "3"},
          {"KW", "3"},
          {"TH", "4"},
          {"TW", "4"},
          {"STRIDE", "1"},
          {"VEC", "1"}},
         geluTanh},
        {"FusedScaleBiasSilu",
         "gfx942",
         {{"VARIANT", "2"},
          {"LAYOUT", "0"},
          {"KH", "5"},
          {"KW", "5"},
          {"BH", "8"},
          {"BW", "8"},
          {"BK", "32"},
          {"RH", "2"},
          {"RW", "2"}},
         scaleBiasSilu},
        {"LdsScaleBiasSilu",
         "gfx950",
         {{"VARIANT", "3"},
          {"LAYOUT", "0"},
          {"KH", "9"},
          {"KW", "9"},
          {"BH", "8"},
          {"BW", "8"},
          {"BK", "16"},
          {"RH", "2"},
          {"RW", "2"}},
         scaleBiasSilu},
        {"LdsChannelsFirstBiasRelu",
         "gfx942",
         {{"VARIANT", "3"},
          {"LAYOUT", "1"},
          {"KH", "7"},
          {"KW", "7"},
          {"BH", "8"},
          {"BW", "8"},
          {"BK", "8"},
          {"RH", "2"},
          {"RW", "2"}},
         biasRelu},
    };
}

class TestConvDepthwiseValuFusedCompile : public ::testing::TestWithParam<FusedCompileCase>
{
};

TEST_P(TestConvDepthwiseValuFusedCompile, CompilesThroughHiprtc)
{
    const auto& param = GetParam();
    // The epilogue's generated code, from a fused graph of the menu chain.
    const auto spec = conv_epilogue_test::fusedSpecFor(
        depthwise(8), param.steps, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const GraphFixture fixture(conv_epilogue_test::buildFusedGraph(spec));
    const auto problem = parseDepthwiseValuProblem(
        fixture.context(), ConvDirection::FORWARD, conv_naive::Fusion::EPILOGUE);
    ASSERT_TRUE(problem.has_value());

    hip_kernel_provider::compilation::KernelCompileOptions options(param.arch);
    options.add("CONV_DW_TYPE", std::string("_Float16"));
    options.add("CONV_DW_NDIMS", std::string("2"));
    options.add("CONV_DW_FLIP", std::string("0"));
    for(const auto& [name, value] : param.defines)
    {
        options.add("CONV_DW_" + name, value);
    }
    conv_epilogue::addEpilogueDefines(options, problem->base.epilogue);
    const std::vector<std::string> optionStrings = options;
    std::vector<const char*> optionPointers;
    optionPointers.reserve(optionStrings.size());
    for(const auto& option : optionStrings)
    {
        optionPointers.push_back(option.c_str());
    }

    const std::string source(hip_plugin::getKernelSrc("kernels/ConvDepthwiseValu.cpp"));
    ASSERT_FALSE(source.empty());
    std::vector<std::string_view> includeTexts;
    std::vector<const char*> includeNames;
    hip_plugin::getKernelIncList(includeTexts, includeNames);
    const std::vector<std::string> includeStrings(includeTexts.begin(), includeTexts.end());
    std::vector<const char*> includePointers;
    includePointers.reserve(includeStrings.size());
    for(const auto& text : includeStrings)
    {
        includePointers.push_back(text.c_str());
    }

    hiprtcProgram program = nullptr;
    ASSERT_EQ(hiprtcCreateProgram(&program,
                                  source.c_str(),
                                  "ConvDepthwiseValu.cpp",
                                  static_cast<int>(includePointers.size()),
                                  includePointers.data(),
                                  includeNames.data()),
              HIPRTC_SUCCESS);
    const auto result = hiprtcCompileProgram(
        program, static_cast<int>(optionPointers.size()), optionPointers.data());
    std::string log;
    size_t logSize = 0;
    if(hiprtcGetProgramLogSize(program, &logSize) == HIPRTC_SUCCESS && logSize > 1)
    {
        log.resize(logSize);
        static_cast<void>(hiprtcGetProgramLog(program, log.data()));
    }
    hiprtcDestroyProgram(&program);
    EXPECT_EQ(result, HIPRTC_SUCCESS) << log;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvDepthwiseValuFusedCompile,
                         ::testing::ValuesIn(fusedCompileCases()),
                         [](const auto& info) { return info.param.name; });

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
