// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvNaiveProblem.cpp
 * @brief Host-only checks of the naive convolution engines' acceptance rules: what each
 *        accepts, the argument block it derives, one refusal per gated field, and the
 *        workload gate. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using conv_naive::parseConvNaiveProblem;

constexpr ConvDirection ALL_DIRECTIONS[]
    = {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS};

std::optional<conv_naive::ConvNaiveProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseConvNaiveProblem(fixture.context(), spec.direction);
}

std::string directionLabel(ConvDirection direction)
{
    return std::string(conv_naive::engineName(direction));
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
    cases.push_back({"Nchw2d", ConvShape{}});

    ConvShape nhwc;
    nhwc.channelsLast = true;
    cases.push_back({"Nhwc2d", nhwc});

    ConvShape oneD;
    oneD.input = {9};
    oneD.filter = {3};
    oneD.prePadding = {1};
    oneD.postPadding = {1};
    oneD.stride = {2};
    oneD.dilation = {1};
    cases.push_back({"OneD", oneD});

    ConvShape threeD;
    threeD.input = {4, 5, 6};
    threeD.filter = {2, 3, 2};
    threeD.prePadding = {0, 1, 1};
    threeD.postPadding = {0, 1, 0};
    threeD.stride = {1, 2, 1};
    threeD.dilation = {1, 1, 2};
    threeD.channelsLast = true;
    cases.push_back({"ThreeDChannelsLast", threeD});

    ConvShape grouped;
    grouped.c = 6;
    grouped.k = 9;
    grouped.groups = 3;
    cases.push_back({"Grouped", grouped});

    ConvShape depthwise;
    depthwise.c = 5;
    depthwise.k = 5;
    depthwise.groups = 5;
    cases.push_back({"Depthwise", depthwise});

    ConvShape padded;
    padded.prePadding = {2, 0};
    padded.postPadding = {1, 3};
    padded.stride = {2, 3};
    padded.dilation = {2, 1};
    cases.push_back({"AsymmetricPaddingStrideDilation", padded});

    ConvShape convolution;
    convolution.mode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CONVOLUTION;
    cases.push_back({"ConvolutionMode", convolution});

    ConvShape half;
    half.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    cases.push_back({"Half", half});

    ConvShape bfloat;
    bfloat.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    cases.push_back({"BFloat16", bfloat});
    return cases;
}

class TestConvNaiveAccepts : public ::testing::TestWithParam<std::tuple<AcceptCase, ConvDirection>>
{
};

TEST_P(TestConvNaiveAccepts, EveryDirection)
{
    const auto& [acceptCase, direction] = GetParam();
    const auto problem = parse(specFor(acceptCase.shape, direction));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->direction, direction);
    EXPECT_EQ(problem->operands.x, X_UID);
    EXPECT_EQ(problem->operands.w, W_UID);
    EXPECT_EQ(problem->operands.y, Y_UID);
    EXPECT_EQ(problem->args.groups, acceptCase.shape.groups);
    EXPECT_EQ(problem->args.flipFilter,
              acceptCase.shape.mode == hipdnn_flatbuffers_sdk::data_objects::ConvMode::CONVOLUTION
                  ? 1
                  : 0);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvNaiveAccepts,
                         ::testing::Combine(::testing::ValuesIn(acceptCases()),
                                            ::testing::ValuesIn(ALL_DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + std::to_string(static_cast<int>(std::get<1>(info.param)));
                         });

// ---------------------------------------------------------------------------
// The argument block
// ---------------------------------------------------------------------------

TEST(TestConvNaiveArgs, PadsATwoDProblemToThreeD)
{
    ConvShape shape;
    shape.prePadding = {1, 2};
    shape.postPadding = {1, 0};
    shape.stride = {2, 1};
    shape.dilation = {1, 2};
    const auto problem = parse(specFor(shape, ConvDirection::FORWARD));
    ASSERT_TRUE(problem.has_value());
    const auto& args = problem->args;
    const auto output = shape.output();

    EXPECT_EQ(problem->spatialRank, 2);
    EXPECT_EQ(args.inSpatial[0], 1);
    EXPECT_EQ(args.outSpatial[0], 1);
    EXPECT_EQ(args.filter[0], 1);
    EXPECT_EQ(args.prePadding[0], 0);
    EXPECT_EQ(args.stride[0], 1);
    EXPECT_EQ(args.dilation[0], 1);
    EXPECT_EQ(args.xStrides[2], 0);
    EXPECT_EQ(args.wStrides[2], 0);
    EXPECT_EQ(args.yStrides[2], 0);

    EXPECT_EQ(args.inSpatial[1], shape.input[0]);
    EXPECT_EQ(args.inSpatial[2], shape.input[1]);
    EXPECT_EQ(args.outSpatial[1], output[0]);
    EXPECT_EQ(args.outSpatial[2], output[1]);
    EXPECT_EQ(args.filter[1], shape.filter[0]);
    EXPECT_EQ(args.filter[2], shape.filter[1]);
    EXPECT_EQ(args.prePadding[1], 1);
    EXPECT_EQ(args.prePadding[2], 2);
    EXPECT_EQ(args.stride[1], 2);
    EXPECT_EQ(args.dilation[2], 2);
}

TEST(TestConvNaiveArgs, CopiesEveryOperandStrideInNcOrder)
{
    ConvShape shape;
    shape.channelsLast = true;
    const auto spec = specFor(shape, ConvDirection::FORWARD);
    const auto problem = parse(spec);
    ASSERT_TRUE(problem.has_value());
    const auto& args = problem->args;
    for(size_t axis = 0; axis < 2; ++axis)
    {
        EXPECT_EQ(args.xStrides[axis], spec.x.strides[axis]);
        EXPECT_EQ(args.wStrides[axis], spec.w.strides[axis]);
        EXPECT_EQ(args.yStrides[axis], spec.y.strides[axis]);
    }
    for(size_t axis = 2; axis < 4; ++axis)
    {
        EXPECT_EQ(args.xStrides[axis + 1], spec.x.strides[axis]);
        EXPECT_EQ(args.wStrides[axis + 1], spec.w.strides[axis]);
        EXPECT_EQ(args.yStrides[axis + 1], spec.y.strides[axis]);
    }
}

TEST(TestConvNaiveArgs, CountsOneThreadPerWrittenElementAndTheSameWorkInEveryDirection)
{
    ConvShape shape;
    shape.c = 6;
    shape.k = 4;
    shape.groups = 2;
    const auto output = shape.output();
    const int64_t outElements = output[0] * output[1];
    const int64_t inElements = shape.input[0] * shape.input[1];
    const int64_t filterElements = shape.filter[0] * shape.filter[1];
    const int64_t macs
        = shape.n * shape.k * outElements * (shape.c / shape.groups) * filterElements;

    const auto fwd = parse(specFor(shape, ConvDirection::FORWARD));
    const auto bwd = parse(specFor(shape, ConvDirection::BACKWARD_DATA));
    const auto wrw = parse(specFor(shape, ConvDirection::BACKWARD_WEIGHTS));
    ASSERT_TRUE(fwd.has_value() && bwd.has_value() && wrw.has_value());

    EXPECT_EQ(fwd->args.total, shape.n * shape.k * outElements);
    EXPECT_EQ(bwd->args.total, shape.n * shape.c * inElements);
    EXPECT_EQ(wrw->args.total, shape.k * (shape.c / shape.groups) * filterElements);
    EXPECT_EQ(fwd->multiplyAccumulates, macs);
    EXPECT_EQ(bwd->multiplyAccumulates, macs);
    EXPECT_EQ(wrw->multiplyAccumulates, macs);
}

// ---------------------------------------------------------------------------
// One refusal per gated field
// ---------------------------------------------------------------------------

struct RefusalCase
{
    std::string name;
    std::function<void(ConvGraphSpec&)> mutate;
};

using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

/// The tensor the direction writes, in forward roles.
TensorSpec& written(ConvGraphSpec& spec)
{
    switch(spec.direction)
    {
    case ConvDirection::FORWARD:
        return spec.y;
    case ConvDirection::BACKWARD_DATA:
        return spec.x;
    case ConvDirection::BACKWARD_WEIGHTS:
        return spec.w;
    default:
        break;
    }
    return spec.y;
}

std::vector<RefusalCase> refusalCases()
{
    return {
        // Graph
        {"OverrideShapeEnabled", [](ConvGraphSpec& s) { s.overrideShapeEnabled = true; }},
        // Node
        {"ComputeTypeDouble", [](ConvGraphSpec& s) { s.computeType = DataType::DOUBLE; }},
        {"ComputeTypeInt32", [](ConvGraphSpec& s) { s.computeType = DataType::INT32; }},
        {"ModeUnset", [](ConvGraphSpec& s) { s.mode = ConvMode::UNSET; }},
        {"OutputAliasesInput", [](ConvGraphSpec& s) { s.yUid = s.xUid; }},
        {"MissingTensor", [](ConvGraphSpec& s) { s.omitW = true; }},
        // Tensor fields
        {"VirtualOperand", [](ConvGraphSpec& s) { s.x.isVirtual = true; }},
        {"RuntimePassByValue", [](ConvGraphSpec& s) { s.w.runtimePassByValue = true; }},
        {"RaggedOffset", [](ConvGraphSpec& s) { s.y.raggedOffsetUid = 7; }},
        {"RaggedMultiplier", [](ConvGraphSpec& s) { s.y.raggedOffsetMultiplier = 2; }},
        {"UnderAligned", [](ConvGraphSpec& s) { s.x.alignment = 2; }},
        {"Int8",
         [](ConvGraphSpec& s) { s.x.dataType = s.w.dataType = s.y.dataType = DataType::INT8; }},
        {"Double",
         [](ConvGraphSpec& s) { s.x.dataType = s.w.dataType = s.y.dataType = DataType::DOUBLE; }},
        {"MixedDtype", [](ConvGraphSpec& s) { s.w.dataType = DataType::HALF; }},
        {"StridesRankMismatch", [](ConvGraphSpec& s) { s.x.strides.pop_back(); }},
        {"RankTwo",
         [](ConvGraphSpec& s) {
             s.x.dims = {2, 4};
             s.x.strides = {4, 1};
         }},
        {"RankSix",
         [](ConvGraphSpec& s) {
             // A consistent 4-D-spatial problem: every operand and parameter one axis longer
             // twice over, so rank is the only thing refused.
             for(auto* tensor : {&s.x, &s.w, &s.y})
             {
                 tensor->dims.insert(tensor->dims.end(), {1, 1});
                 tensor->strides = packedStrides(tensor->dims, false);
             }
             for(auto* parameter : {&s.prePadding, &s.postPadding})
             {
                 parameter->insert(parameter->end(), {0, 0});
             }
             for(auto* parameter : {&s.stride, &s.dilation})
             {
                 parameter->insert(parameter->end(), {1, 1});
             }
         }},
        {"OperandRanksDiffer",
         [](ConvGraphSpec& s) {
             s.w.dims.push_back(1);
             s.w.strides.push_back(1);
         }},
        {"ZeroExtent", [](ConvGraphSpec& s) { s.x.dims[0] = 0; }},
        {"NegativeStride", [](ConvGraphSpec& s) { s.x.strides[3] = -1; }},
        {"OverlappingOutput",
         [](ConvGraphSpec& s) {
             auto& out = written(s);
             out.strides[out.strides.size() - 1] = 0;
         }},
        // Convolution parameters
        {"PaddingWrongLength", [](ConvGraphSpec& s) { s.prePadding.push_back(0); }},
        {"NegativePadding", [](ConvGraphSpec& s) { s.postPadding[0] = -1; }},
        {"ZeroStride", [](ConvGraphSpec& s) { s.stride[1] = 0; }},
        {"ZeroDilation", [](ConvGraphSpec& s) { s.dilation[0] = 0; }},
        {"StrideWrongLength", [](ConvGraphSpec& s) { s.stride.pop_back(); }},
        {"DilationWrongLength", [](ConvGraphSpec& s) { s.dilation.push_back(1); }},
        // Shapes
        {"FilterChannelsDoNotDivideInput", [](ConvGraphSpec& s) { s.w.dims[1] = 3; }},
        {"OutputChannelsDoNotDivideGroups",
         [](ConvGraphSpec& s) {
             // c=4, filter channels 2: two groups, and k=5 does not split in two.
             s.w.dims = {5, 2, 3, 2};
             s.w.strides = packedStrides(s.w.dims, false);
             s.y.dims[1] = 5;
             s.y.strides = packedStrides(s.y.dims, false);
         }},
        {"BatchMismatch",
         [](ConvGraphSpec& s) {
             s.y.dims[0] = 3;
             s.y.strides = packedStrides(s.y.dims, false);
         }},
        {"OutputChannelMismatch",
         [](ConvGraphSpec& s) {
             s.y.dims[1] = 7;
             s.y.strides = packedStrides(s.y.dims, false);
         }},
        {"WrongOutputExtent",
         [](ConvGraphSpec& s) {
             s.y.dims[2] += 1;
             s.y.strides = packedStrides(s.y.dims, false);
         }},
        {"FilterLargerThanPaddedInput",
         [](ConvGraphSpec& s) {
             s.w.dims[2] = 9;
             s.w.strides = packedStrides(s.w.dims, false);
         }},
    };
}

class TestConvNaiveRefuses : public ::testing::TestWithParam<std::tuple<RefusalCase, ConvDirection>>
{
};

TEST_P(TestConvNaiveRefuses, OneField)
{
    const auto& [refusal, direction] = GetParam();
    auto spec = specFor(ConvShape{}, direction);
    // The unmutated spec is accepted, so the mutation alone is what is refused.
    ASSERT_TRUE(parse(spec).has_value());
    refusal.mutate(spec);
    EXPECT_FALSE(parse(spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvNaiveRefuses,
                         ::testing::Combine(::testing::ValuesIn(refusalCases()),
                                            ::testing::ValuesIn(ALL_DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + std::to_string(static_cast<int>(std::get<1>(info.param)));
                         });

TEST(TestConvNaiveDirection, RefusesAGraphOfAnotherDirection)
{
    for(const auto built : ALL_DIRECTIONS)
    {
        const GraphFixture fixture(buildGraph(specFor(ConvShape{}, built)));
        for(const auto asked : ALL_DIRECTIONS)
        {
            EXPECT_EQ(parseConvNaiveProblem(fixture.context(), asked).has_value(), built == asked)
                << directionLabel(built) << " graph asked as " << directionLabel(asked);
        }
    }
}

TEST(TestConvNaiveComputeType, ServesCudnnsComputeTypesPerElementType)
{
    for(const auto direction : ALL_DIRECTIONS)
    {
        expectCudnnComputeTypes(ConvShape{},
                                {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
                                [direction](const ConvShape& shape) {
                                    return parse(specFor(shape, direction)).has_value();
                                });
    }
}

// ---------------------------------------------------------------------------
// Workload gate
// ---------------------------------------------------------------------------

hipdnn_plugin_sdk::ingestor::DeviceProperties deviceWithComputeUnits(int computeUnits)
{
    auto properties = testDeviceProperties();
    properties.multiProcessorCount = computeUnits;
    return properties;
}

TEST(TestConvNaiveWorkload, AcceptsASmallProblem)
{
    const auto problem = parse(specFor(ConvShape{}, ConvDirection::FORWARD));
    ASSERT_TRUE(problem.has_value());
    EXPECT_TRUE(conv_naive::withinWorkloadBudget(*problem, deviceWithComputeUnits(4)));
}

TEST(TestConvNaiveWorkload, DeclinesAProblemOverBudget)
{
    // 3x3, C = K = 128, 56x56: 462 M multiply-accumulates, past the budget of any device.
    ConvShape shape;
    shape.n = 1;
    shape.c = 128;
    shape.k = 128;
    shape.input = {56, 56};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    for(const auto direction : ALL_DIRECTIONS)
    {
        const auto problem = parse(specFor(shape, direction));
        ASSERT_TRUE(problem.has_value());
        EXPECT_FALSE(conv_naive::withinWorkloadBudget(*problem, deviceWithComputeUnits(256)))
            << directionLabel(direction);
    }
}

TEST(TestConvNaiveWorkload, BudgetsByTheLanesTheProblemCanOccupy)
{
    // Few output elements: the budget is per lane. A weight gradient's dw element is one block
    // of 256 lanes, so one element over 64 images of 64 x 64 (262144 multiply-accumulates) is
    // 1024 a lane, and of 256 x 256 (4.2 M) 16384, past the budget.
    ConvShape shape;
    shape.n = 64;
    shape.c = 1;
    shape.k = 1;
    shape.input = {64, 64};
    shape.filter = {1, 1};
    const auto wrw = parse(specFor(shape, ConvDirection::BACKWARD_WEIGHTS));
    const auto fwd = parse(specFor(shape, ConvDirection::FORWARD));
    ASSERT_TRUE(wrw.has_value() && fwd.has_value());
    EXPECT_EQ(wrw->args.total, 1);
    EXPECT_TRUE(conv_naive::withinWorkloadBudget(*wrw, deviceWithComputeUnits(64)));
    EXPECT_TRUE(conv_naive::withinWorkloadBudget(*fwd, deviceWithComputeUnits(64)));
    shape.input = {256, 256};
    const auto wide = parse(specFor(shape, ConvDirection::BACKWARD_WEIGHTS));
    ASSERT_TRUE(wide.has_value());
    EXPECT_FALSE(conv_naive::withinWorkloadBudget(*wide, deviceWithComputeUnits(64)));
    // The device's lanes still bound it: 1-D, 8 images of 1024, 16 -> 16 channels, 3 taps
    // (768 dw elements, 6.3 M multiply-accumulates) is 1536 a lane on 64 CUs.
    ConvShape line;
    line.n = 8;
    line.c = 16;
    line.k = 16;
    line.input = {1024};
    line.filter = {3};
    line.prePadding = {1};
    line.postPadding = {1};
    line.stride = {1};
    line.dilation = {1};
    const auto oneD = parse(specFor(line, ConvDirection::BACKWARD_WEIGHTS));
    ASSERT_TRUE(oneD.has_value());
    EXPECT_TRUE(conv_naive::withinWorkloadBudget(*oneD, deviceWithComputeUnits(64)));
    EXPECT_FALSE(conv_naive::withinWorkloadBudget(*oneD, deviceWithComputeUnits(4)));
}

TEST(TestConvNaiveWorkload, AssumesFallbackComputeUnitsWhenTheDeviceDoesNotSay)
{
    const auto problem = parse(specFor(ConvShape{}, ConvDirection::FORWARD));
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(conv_naive::withinWorkloadBudget(*problem, deviceWithComputeUnits(0)),
              conv_naive::withinWorkloadBudget(
                  *problem, deviceWithComputeUnits(conv_naive::FALLBACK_COMPUTE_UNITS)));
}

hipdnn_plugin_sdk::ingestor::DeviceProperties deviceOf(const std::string& arch, int units)
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = units;
    return properties;
}

TEST(TestConvNaiveWorkload, CountsTwoComputeUnitsPerReportedRdnaWorkgroupProcessor)
{
    // 3x3, C 32, K 64, 32x32: 18.9 M multiply-accumulates over 64 K outputs, about 6 K per lane
    // on 48 CUs (3 K lanes) and 3 K on 96. HIP reports a 96-CU gfx1100 as 48 WGPs.
    ConvShape shape;
    shape.n = 1;
    shape.c = 32;
    shape.k = 64;
    shape.input = {32, 32};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    const auto problem = parse(specFor(shape, ConvDirection::FORWARD));
    ASSERT_TRUE(problem.has_value());
    EXPECT_FALSE(
        conv_naive::withinWorkloadBudget(*problem, deviceOf("gfx942:sramecc+:xnack-", 48)));
    EXPECT_TRUE(conv_naive::withinWorkloadBudget(*problem, deviceOf("gfx942:sramecc+:xnack-", 96)));
    EXPECT_TRUE(conv_naive::withinWorkloadBudget(*problem, deviceOf("gfx1100", 48)));
    EXPECT_TRUE(conv_naive::withinWorkloadBudget(*problem, deviceOf("gfx1201", 48)));
}

// ---------------------------------------------------------------------------
// Device units (DeviceUnits.hpp): what HIP's multiProcessorCount counts
// ---------------------------------------------------------------------------

TEST(TestConvDeviceUnits, Gfx10AndLaterReportWorkgroupProcessors)
{
    for(const char* arch : {"gfx1030",
                            "gfx1100",
                            "gfx1151",
                            "gfx1201",
                            "gfx1201:xnack-",
                            "gfx11-generic",
                            "gfx10-3-generic",
                            "gfx1250"})
    {
        EXPECT_TRUE(device_units::reportsWorkgroupProcessors(arch)) << arch;
    }
    for(const char* arch : {"gfx906",
                            "gfx908",
                            "gfx90a",
                            "gfx942:sramecc+:xnack-",
                            "gfx950",
                            "gfx9-4-generic",
                            "",
                            "gfx000",
                            "gfx1"})
    {
        EXPECT_FALSE(device_units::reportsWorkgroupProcessors(arch)) << arch;
    }
}

TEST(TestConvDeviceUnits, ComputeUnitsArePhysicalAndUnknownCountsAreZero)
{
    EXPECT_EQ(device_units::dispatchUnits(deviceOf("gfx1100", 48)), 48);
    EXPECT_EQ(device_units::computeUnits(deviceOf("gfx1100", 48)), 96);
    EXPECT_EQ(device_units::computeUnits(deviceOf("gfx1151", 20)), 40);
    EXPECT_EQ(device_units::dispatchUnits(deviceOf("gfx942", 304)), 304);
    EXPECT_EQ(device_units::computeUnits(deviceOf("gfx942", 304)), 304);
    EXPECT_EQ(device_units::dispatchUnits(deviceOf("gfx1100", 0)), 0);
    EXPECT_EQ(device_units::computeUnits(deviceOf("gfx1100", 0)), 0);
    EXPECT_EQ(device_units::dispatchUnits(deviceOf("gfx942", -1)), 0);
    EXPECT_EQ(device_units::computeUnits(deviceOf("gfx1100", -3)), 0);
}

// ---------------------------------------------------------------------------
// Engine wiring: the registered symbols the descriptors name
// ---------------------------------------------------------------------------

PackSymbols packFor(ConvDirection direction)
{
    switch(direction)
    {
    case ConvDirection::FORWARD:
        return {"hipkernel:CONV_NAIVE_FWD",
                "hipkernel.conv_naive_fwd.graph_match",
                "",
                "hipkernel.conv_naive.kernel_match",
                "",
                "hipkernel.conv_naive.dispatch",
                "conv_naive.x.uid",
                "conv_naive.w.uid",
                "conv_naive.y.uid"};
    case ConvDirection::BACKWARD_DATA:
        return {"hipkernel:CONV_NAIVE_BWD",
                "hipkernel.conv_naive_bwd.graph_match",
                "",
                "hipkernel.conv_naive.kernel_match",
                "",
                "hipkernel.conv_naive.dispatch",
                "conv_naive.x.uid",
                "conv_naive.w.uid",
                "conv_naive.y.uid"};
    case ConvDirection::BACKWARD_WEIGHTS:
        return {"hipkernel:CONV_NAIVE_WRW",
                "hipkernel.conv_naive_wrw.graph_match",
                "",
                "hipkernel.conv_naive.kernel_match",
                "",
                "hipkernel.conv_naive.dispatch",
                "conv_naive.x.uid",
                "conv_naive.w.uid",
                "conv_naive.y.uid"};
    default:
        break;
    }
    return {};
}

hipdnn_plugin_sdk::ingestor::KernelDefinition naiveKernel(const std::string& dtype,
                                                          const std::string& direction)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.source.sourceFile = "kernels/ConvNaive.cpp";
    kernel.source.entryPoint = "ConvNaive" + direction;
    kernel.metadata = {{"dtype", dtype}, {"direction", direction}};
    return kernel;
}

TEST(TestConvNaiveEngines, EachGraphMatcherServesOnlyItsDirection)
{
    for(const auto built : ALL_DIRECTIONS)
    {
        const GraphFixture fixture(buildGraph(specFor(ConvShape{}, built)),
                                   deviceWithComputeUnits(16));
        for(const auto asked : ALL_DIRECTIONS)
        {
            EXPECT_EQ(matchesGraph(packFor(asked), fixture.context()).has_value(), built == asked)
                << directionLabel(built) << " graph asked as " << directionLabel(asked);
        }
    }
}

TEST(TestConvNaiveEngines, KernelMatcherRequiresTheGraphDtypeAndDirection)
{
    ConvShape shape;
    shape.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_DATA)),
                               deviceWithComputeUnits(16));
    const auto pack = packFor(ConvDirection::BACKWARD_DATA);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_TRUE(matchesKernel(pack, fixture.context(), naiveKernel("HALF", "BWD"), *bound));
    EXPECT_FALSE(matchesKernel(pack, fixture.context(), naiveKernel("FLOAT", "BWD"), *bound));
    EXPECT_FALSE(matchesKernel(pack, fixture.context(), naiveKernel("BFLOAT16", "BWD"), *bound));
    EXPECT_FALSE(matchesKernel(pack, fixture.context(), naiveKernel("HALF", "FWD"), *bound));
    EXPECT_FALSE(matchesKernel(pack, fixture.context(), naiveKernel("HALF", "WRW"), *bound));
}

TEST(TestConvNaiveEngines, GraphMatcherAppliesTheWorkloadGate)
{
    ConvShape shape;
    shape.n = 1;
    shape.c = 128;
    shape.k = 128;
    shape.input = {56, 56};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)),
                               deviceWithComputeUnits(256));
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::FORWARD), fixture.context()).has_value());
}

TEST(TestConvNaiveEngines, DispatchNeedsNoWorkspace)
{
    const GraphFixture fixture(buildGraph(specFor(ConvShape{}, ConvDirection::FORWARD)),
                               deviceWithComputeUnits(16));
    const auto pack = packFor(ConvDirection::FORWARD);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(dispatchHandler(pack).workspaceBytes(
                  fixture.context(), *bound, naiveKernel("FLOAT", "FWD")),
              0u);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
