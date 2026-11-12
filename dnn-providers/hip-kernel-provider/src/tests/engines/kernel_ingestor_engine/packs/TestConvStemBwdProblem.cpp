// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvStemBwdProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvStemBwdProblem.cpp
 * @brief Host-only checks of the stem backward-data engine (CONV_STEM_BWD): what the graph
 *        match accepts, one refusal per gated graph field, the window arithmetic, what the
 *        kernel match admits per config field and device (one refusal per config field), the
 *        argument blocks of both kernels and the defines. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using conv_stem_bwd::parseStemBwdProblem;
using conv_stem_bwd::StemBwdConfig;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

constexpr int64_t TWO_POW_31 = int64_t{1} << 31;

/// The base problem: a 7x7 stride-2 stem, padding 3, NCHW, fp16, 3 -> 16 channels.
ConvShape baseShape()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 3;
    shape.k = 16;
    shape.input = {29, 64};
    shape.filter = {7, 7};
    shape.prePadding = {3, 3};
    shape.postPadding = {3, 3};
    shape.stride = {2, 2};
    shape.dilation = {1, 1};
    shape.dataType = DataType::HALF;
    return shape;
}

ConvGraphSpec specOf(const ConvShape& shape)
{
    return specFor(shape, ConvDirection::BACKWARD_DATA);
}

std::optional<conv_stem_bwd::StemBwdProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseStemBwdProblem(fixture.context());
}

// ---------------------------------------------------------------------------
// The window
// ---------------------------------------------------------------------------

TEST(TestConvStemBwdWindow, OffsetsAndExtentsOfTheStems)
{
    using conv_stem_bwd::windowOf;
    const auto check
        = [](int64_t filter, int64_t stride, int64_t padding, int64_t offset, int64_t extent) {
              const auto window = windowOf(filter, stride, padding);
              EXPECT_EQ(window.offset, offset) << filter << " " << stride << " " << padding;
              EXPECT_EQ(window.extent, extent) << filter << " " << stride << " " << padding;
          };
    check(7, 2, 3, 2, 4); // ResNet stem
    check(3, 2, 1, 1, 2);
    check(3, 2, 0, 0, 2);
    check(6, 2, 2, 1, 3); // YOLOv5 stem
    check(5, 2, 2, 1, 3);
    check(2, 2, 0, 0, 1);
    check(3, 1, 1, 1, 3);
    check(7, 1, 3, 3, 7);
    check(1, 1, 0, 0, 1);
    check(4, 4, 0, 0, 1); // patchify
    check(11, 4, 2, 1, 4); // AlexNet
    check(3, 1, 5, 5, 3); // padding past the filter
    check(1, 2, 3, 2, 1);
}

TEST(TestConvStemBwdWindow, EveryTapOfEveryPhaseFallsInTheWindow)
{
    // Filter row r pairs phase p with window offset u = (r - p - padding) / stride + offset,
    // whole exactly when r = stride (u - offset) + p + padding: every r in [0, filter) lands in
    // [0, extent) for its phase, and the extent is the smallest that holds them.
    for(int64_t stride = 1; stride <= 4; ++stride)
    {
        for(int64_t filter = 1; filter <= 12; ++filter)
        {
            for(int64_t padding = 0; padding <= 12; ++padding)
            {
                const auto window = conv_stem_bwd::windowOf(filter, stride, padding);
                int64_t lowest = window.extent;
                int64_t highest = -1;
                for(int64_t r = 0; r < filter; ++r)
                {
                    for(int64_t p = 0; p < stride; ++p)
                    {
                        const int64_t shifted = r - p - padding;
                        if(((shifted % stride) + stride) % stride != 0)
                        {
                            continue;
                        }
                        const int64_t u = (shifted >= 0 ? shifted / stride : -((-shifted) / stride))
                                          + window.offset;
                        lowest = std::min(lowest, u);
                        highest = std::max(highest, u);
                    }
                }
                SCOPED_TRACE(std::to_string(filter) + " " + std::to_string(stride) + " "
                             + std::to_string(padding));
                EXPECT_GE(lowest, 0);
                EXPECT_LT(highest, window.extent);
                EXPECT_EQ(highest, window.extent - 1);
            }
        }
    }
}

TEST(TestConvStemBwdWindow, SpaceToDepthOffsetsExtentsAndShifts)
{
    using conv_stem_bwd::spaceToDepthWindowOf;
    const auto check
        = [](int64_t filter, int64_t padding, int64_t offset, int64_t extent, int64_t shift) {
              const auto window = spaceToDepthWindowOf(filter, padding);
              EXPECT_EQ(window.offset, offset) << filter << " " << padding;
              EXPECT_EQ(window.extent, extent) << filter << " " << padding;
              EXPECT_EQ(window.shift, shift) << filter << " " << padding;
          };
    check(3, 1, 0, 2, 1); // U-Net / VGG "same" 3x3
    check(5, 2, 1, 3, 0);
    check(7, 3, 1, 4, 1);
    check(1, 0, 0, 1, 0);
    check(3, 0, 0, 2, 0);
    check(3, 5, 2, 2, 1); // padding past the filter
}

TEST(TestConvStemBwdWindow, SpaceToDepthPairsEveryTapOfEveryPhaseOnce)
{
    // dx row 2 i + p reads dy row 2 (i + offset - u) + shift + e against filter row r = 2 (u -
    // offset) + p + padding - shift - e (ConvStemBwdTransform): each r in [0, filter) exactly
    // once over u < extent and e < 2, for both phases.
    for(int64_t filter = 1; filter <= 12; ++filter)
    {
        for(int64_t padding = 0; padding <= 12; ++padding)
        {
            const auto window = conv_stem_bwd::spaceToDepthWindowOf(filter, padding);
            SCOPED_TRACE(std::to_string(filter) + " " + std::to_string(padding));
            for(int64_t p = 0; p < 2; ++p)
            {
                std::vector<int> hits(static_cast<size_t>(filter), 0);
                for(int64_t u = 0; u < window.extent; ++u)
                {
                    for(int64_t e = 0; e < 2; ++e)
                    {
                        const int64_t r = 2 * (u - window.offset) + p + padding - window.shift - e;
                        if(r >= 0 && r < filter)
                        {
                            ++hits[static_cast<size_t>(r)];
                        }
                    }
                }
                for(const int hit : hits)
                {
                    EXPECT_EQ(hit, 1);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Accepted shapes
// ---------------------------------------------------------------------------

/// A 3-D stem: @p taps x 7 x 7, stride (1, 2, 2), "same" depth padding, 5 dx planes.
ConvShape threeDStem(int64_t taps)
{
    auto shape = baseShape();
    shape.input = {5, 29, 64};
    shape.filter = {taps, 7, 7};
    shape.prePadding = {taps / 2, 3, 3};
    shape.postPadding = {(taps - 1) / 2, 3, 3};
    shape.stride = {1, 2, 2};
    shape.dilation = {1, 1, 1};
    return shape;
}

/// Stride 1: 3 x 5, padding (1, 2) (the space-to-depth windows 2 x 3), dx 29 x 30 (dy rows not
/// whole vectors).
ConvShape stride1Shape()
{
    auto shape = baseShape();
    shape.input = {29, 30};
    shape.filter = {3, 5};
    shape.prePadding = shape.postPadding = {1, 2};
    shape.stride = {1, 1};
    return shape;
}

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
    auto fp32 = baseShape();
    fp32.dataType = DataType::FLOAT;
    cases.push_back({"Float", fp32});
    auto bf16 = baseShape();
    bf16.dataType = DataType::BFLOAT16;
    cases.push_back({"BFloat16", bf16});
    auto grey = baseShape();
    grey.c = 1;
    cases.push_back({"OneChannel", grey});
    auto four = baseShape();
    four.c = 4;
    cases.push_back({"FourChannels", four});
    auto convolution = baseShape();
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"TrueConvolution", convolution});
    auto oneD = baseShape();
    oneD.input = {50};
    oneD.filter = {7};
    oneD.prePadding = oneD.postPadding = {3};
    oneD.stride = {2};
    oneD.dilation = {1};
    cases.push_back({"OneD", oneD});
    auto threeD = baseShape();
    threeD.input = {1, 29, 64};
    threeD.filter = {1, 7, 7};
    threeD.prePadding = threeD.postPadding = {0, 3, 3};
    threeD.stride = {3, 2, 2};
    threeD.dilation = {2, 1, 1};
    cases.push_back({"ThreeDOfDepthOne", threeD});
    cases.push_back({"ThreeDDepthPlanes", threeDStem(1)});
    cases.push_back({"ThreeDDepthTaps", threeDStem(3)});
    cases.push_back({"ThreeDDepthTapsAtTheMost", threeDStem(8)});
    auto asymmetric = baseShape();
    asymmetric.prePadding = {3, 2};
    asymmetric.postPadding = {2, 4};
    cases.push_back({"AsymmetricPadding", asymmetric});
    auto patchify = baseShape();
    patchify.input = {32, 48};
    patchify.filter = {4, 4};
    patchify.prePadding = patchify.postPadding = {0, 0};
    patchify.stride = {4, 4};
    cases.push_back({"Patchify4x4Stride4", patchify});
    auto padded = baseShape();
    padded.filter = {3, 3};
    padded.prePadding = {5, 4};
    padded.postPadding = {0, 5};
    padded.stride = {1, 1};
    cases.push_back({"PaddingPastTheFilter", padded});
    auto limits = baseShape();
    limits.input = {80, 80};
    limits.filter = {64, 64};
    limits.prePadding = limits.postPadding = {64, 64};
    limits.stride = {64, 64};
    cases.push_back({"GeometryAtItsLimits", limits});
    return cases;
}

class TestConvStemBwdAccepts : public ::testing::TestWithParam<AcceptCase>
{
};

TEST_P(TestConvStemBwdAccepts, BackwardData)
{
    EXPECT_TRUE(parse(specOf(GetParam().shape)).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvStemBwdAccepts,
                         ::testing::ValuesIn(acceptCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvStemBwdForm, LayoutsAndTheWindowAreReadFromTheGraph)
{
    const auto cf = parse(specOf(baseShape()));
    ASSERT_TRUE(cf.has_value());
    EXPECT_TRUE(cf->dyChannelsFirst);
    EXPECT_FALSE(cf->dyChannelsLast);
    EXPECT_EQ(cf->elementBytes, 2);
    EXPECT_EQ(cf->dxUid, X_UID);
    EXPECT_EQ(cf->wUid, W_UID);
    EXPECT_EQ(cf->dyUid, Y_UID);
    EXPECT_EQ(cf->oh, 2);
    EXPECT_EQ(cf->ow, 2);
    EXPECT_EQ(cf->u, 4);
    EXPECT_EQ(cf->v, 4);
    EXPECT_EQ(cf->depthTaps, 0);
    // Stride 2: no space-to-depth window.
    EXPECT_EQ(cf->sdU, 0);
    EXPECT_EQ(cf->sdV, 0);

    // Stride 1: its space-to-depth window too.
    const auto unit = parse(specOf(stride1Shape()));
    ASSERT_TRUE(unit.has_value());
    EXPECT_EQ(unit->u, 3);
    EXPECT_EQ(unit->sdOh, 0);
    EXPECT_EQ(unit->sdU, 2);
    EXPECT_EQ(unit->sdShiftH, 1);
    EXPECT_EQ(unit->sdOw, 1);
    EXPECT_EQ(unit->sdV, 3);
    EXPECT_EQ(unit->sdShiftW, 0);

    // 3-D: the depth taps, also of one tap over several planes.
    const auto taps = parse(specOf(threeDStem(3)));
    ASSERT_TRUE(taps.has_value());
    EXPECT_EQ(taps->depthTaps, 3);
    EXPECT_EQ(taps->u, 4);
    const auto planes = parse(specOf(threeDStem(1)));
    ASSERT_TRUE(planes.has_value());
    EXPECT_EQ(planes->depthTaps, 1);

    auto shape = baseShape();
    shape.channelsLast = true;
    const auto cl = parse(specOf(shape));
    ASSERT_TRUE(cl.has_value());
    EXPECT_FALSE(cl->dyChannelsFirst);
    EXPECT_TRUE(cl->dyChannelsLast);

    // dx and the filter in either layout under either dy layout.
    auto spec = specOf(baseShape());
    spec.x.strides = packedStrides(spec.x.dims, true);
    spec.w.strides = packedStrides(spec.w.dims, true);
    const auto mixed = parse(spec);
    ASSERT_TRUE(mixed.has_value());
    EXPECT_TRUE(mixed->dyChannelsFirst);

    // Asymmetric padding: the window follows the leading padding.
    auto asymmetric = baseShape();
    asymmetric.prePadding = {1, 0};
    asymmetric.postPadding = {5, 6};
    asymmetric.filter = {3, 3};
    const auto window = parse(specOf(asymmetric));
    ASSERT_TRUE(window.has_value());
    EXPECT_EQ(window->oh, 1);
    EXPECT_EQ(window->u, 2);
    EXPECT_EQ(window->ow, 0);
    EXPECT_EQ(window->v, 2);

    // One output channel: dy in both layouts.
    auto single = baseShape();
    single.k = 1;
    const auto both = parse(specOf(single));
    ASSERT_TRUE(both.has_value());
    EXPECT_TRUE(both->dyChannelsFirst);
    EXPECT_TRUE(both->dyChannelsLast);
}

// ---------------------------------------------------------------------------
// One refusal per gated graph field
// ---------------------------------------------------------------------------

struct RefusalCase
{
    std::string name;
    std::function<void(ConvGraphSpec&)> mutate;
};

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

std::function<void(ConvGraphSpec&)> reshaped(const std::function<void(ConvShape&)>& change)
{
    return [change](ConvGraphSpec& spec) {
        auto shape = baseShape();
        change(shape);
        reshape(spec, shape);
    };
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
        // One group of at most 4 input channels.
        {"Grouped", reshaped([](ConvShape& shape) {
             shape.groups = 3;
             shape.k = 18;
         })},
        {"FiveChannels", reshaped([](ConvShape& shape) { shape.c = 5; })},
        // A non-trivial depth axis: unit stride and dilation, at most 8 taps, padding within
        // the geometry.
        {"DepthStride", reshaped([](ConvShape& shape) {
             shape = threeDStem(3);
             shape.stride = {2, 2, 2};
         })},
        {"DepthDilation", reshaped([](ConvShape& shape) {
             shape = threeDStem(3);
             shape.dilation = {2, 1, 1};
         })},
        {"DepthTapsBeyondTheMost", reshaped([](ConvShape& shape) { shape = threeDStem(9); })},
        {"DepthPaddingBeyondGeometry", reshaped([](ConvShape& shape) {
             shape = threeDStem(1);
             shape.prePadding = {65, 3, 3};
             shape.postPadding = {0, 3, 3};
         })},
        // Unit height and width dilation.
        {"HeightDilation", reshaped([](ConvShape& shape) {
             shape.dilation = {2, 1};
             shape.prePadding = shape.postPadding = {6, 3};
         })},
        {"WidthDilation", reshaped([](ConvShape& shape) {
             shape.dilation = {1, 2};
             shape.prePadding = shape.postPadding = {3, 6};
         })},
        // Strides, filter extents and padding within the window arithmetic's range.
        {"StrideBeyondGeometry", reshaped([](ConvShape& shape) {
             shape.input = {29, 200};
             shape.stride = {2, 65};
         })},
        {"FilterBeyondGeometry", reshaped([](ConvShape& shape) {
             shape.input = {29, 80};
             shape.filter = {7, 65};
         })},
        {"PaddingBeyondGeometry", reshaped([](ConvShape& shape) {
             shape.prePadding = {3, 65};
             shape.postPadding = {3, 0};
         })},
        // dy packed channels-first or channels-last; dx and the filter packed either way.
        {"DyRowPadded",
         [](ConvGraphSpec& s) {
             const auto& d = s.y.dims; // 2 16 15 32
             s.y.strides = {d[1] * d[2] * 40, d[2] * 40, 40, 1};
         }},
        {"DxChannelPadded",
         [](ConvGraphSpec& s) {
             const auto& d = s.x.dims; // 2 3 29 64
             s.x.strides = {4 * d[2] * d[3], d[2] * d[3], d[3], 1};
         }},
        {"FilterTapsPadded",
         [](ConvGraphSpec& s) {
             s.w.strides = {3 * 7 * 8, 7 * 8, 8, 1}; // rows of 8, 7 used
         }},
        {"FilterChannelsOutermost",
         [](ConvGraphSpec& s) {
             s.w.strides = {49, 16 * 49, 7, 1}; // C K R S
         }},
        // The batch times dx's depth within the grid's z (images tiny).
        {"BatchBeyondGrid", reshaped([](ConvShape& shape) {
             shape.n = 65536;
             shape.input = {4, 4};
         })},
        {"BatchTimesDepthBeyondGrid", reshaped([](ConvShape& shape) {
             shape = threeDStem(1);
             shape.n = 4096;
             shape.input = {16, 4, 4};
         })},
        // Extents within 32 bits.
        {"DxWidthBeyondInt", reshaped([](ConvShape& shape) {
             shape.n = 1;
             shape.c = 1;
             shape.k = 1;
             shape.input = {1, TWO_POW_31};
             shape.filter = {1, 1};
             shape.prePadding = shape.postPadding = {0, 0};
             shape.stride = {1, 64};
         })},
        // One image of dy within 2^31 bytes.
        {"DyImageBeyondOffsets", reshaped([](ConvShape& shape) {
             shape.n = 1;
             shape.k = int64_t{1} << 27;
             shape.input = {2, 16}; // dy 2^27 x 1 x 8 fp16: 2^31 bytes
             shape.filter = {1, 1};
             shape.prePadding = shape.postPadding = {0, 0};
             shape.stride = {2, 2};
         })},
    };
}

class TestConvStemBwdRefuses : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestConvStemBwdRefuses, OneField)
{
    auto spec = specOf(baseShape());
    ASSERT_TRUE(parse(spec).has_value());
    GetParam().mutate(spec);
    EXPECT_FALSE(parse(spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvStemBwdRefuses,
                         ::testing::ValuesIn(refusalCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvStemBwdLimits, TheGateCasesAreValidGraphsOneStepFromTheLimit)
{
    // Each size refusal sits just past its limit: the same graph one step inside it parses,
    // so the refusal is the gate's and not the base parser's.
    auto batch = baseShape();
    batch.n = 65535;
    batch.input = {4, 4};
    EXPECT_TRUE(parse(specOf(batch)).has_value());
    auto planes = threeDStem(1);
    planes.n = 4095;
    planes.input = {16, 4, 4};
    EXPECT_TRUE(parse(specOf(planes)).has_value());
    EXPECT_TRUE(parse(specOf(threeDStem(8))).has_value());
    auto depthPadding = threeDStem(1);
    depthPadding.prePadding = {64, 3, 3};
    depthPadding.postPadding = {0, 3, 3};
    EXPECT_TRUE(parse(specOf(depthPadding)).has_value());
    auto width = baseShape();
    width.n = 1;
    width.c = 1;
    width.k = 1;
    width.input = {1, TWO_POW_31 - 1};
    width.filter = {1, 1};
    width.prePadding = width.postPadding = {0, 0};
    width.stride = {1, 64};
    EXPECT_TRUE(parse(specOf(width)).has_value());
    auto dy = baseShape();
    dy.n = 1;
    dy.k = (int64_t{1} << 27) - 1;
    dy.input = {2, 16};
    dy.filter = {1, 1};
    dy.prePadding = dy.postPadding = {0, 0};
    dy.stride = {2, 2};
    EXPECT_TRUE(parse(specOf(dy)).has_value());
    auto geometry = baseShape();
    geometry.input = {29, 200};
    geometry.stride = {2, 64};
    EXPECT_TRUE(parse(specOf(geometry)).has_value());
}

TEST(TestConvStemBwdDirection, RefusesForwardAndBackwardWeights)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_WEIGHTS})
    {
        const GraphFixture fixture(buildGraph(specFor(baseShape(), direction)));
        EXPECT_FALSE(parseStemBwdProblem(fixture.context()).has_value());
    }
}

// ---------------------------------------------------------------------------
// Kernel match: one refusal per config field
// ---------------------------------------------------------------------------

constexpr PackSymbols PACK{"",
                           "hipkernel.conv_stem_bwd.graph_match",
                           "",
                           "hipkernel.conv_stem_bwd.kernel_match",
                           "",
                           "hipkernel.conv_stem_bwd.dispatch",
                           "",
                           "",
                           ""};

/// A gfx942 config that serves the base problem.
struct ConfigFields
{
    std::string direction = "BWD";
    std::string atom = "mfma16_f16";
    int64_t atomId = 0;
    int64_t layout = 0, vec = 8, u = 4, v = 4, nb = 16, kc = 32, th = 4, fm = 2, sh = 2, sw = 2;
    int64_t ft = 0;
    int64_t sd = 0;
};

/// A space-to-depth config for stride1Shape().
ConfigFields spaceToDepthFields()
{
    ConfigFields f;
    f.vec = 1;
    f.u = 2;
    f.v = 3;
    f.sh = f.sw = 1;
    f.sd = 2;
    return f;
}

/// The FMA loop on an fp32 graph.
ConfigFields fmaFields()
{
    ConfigFields f;
    f.atom = "fma";
    f.atomId = 16;
    f.vec = 4;
    f.nb = 12;
    f.kc = 8;
    f.th = 16;
    return f;
}

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const ConfigFields& f,
                                                       const std::string& dtype)
{
    auto kernel = makeKernel(0, dtype, "ConvStemBwd");
    kernel.metadata = {{"direction", f.direction},
                       {"dtype", dtype},
                       {"atom", f.atom},
                       {"atom_id", f.atomId},
                       {"layout", f.layout},
                       {"vec", f.vec},
                       {"u", f.u},
                       {"v", f.v},
                       {"nb", f.nb},
                       {"kc", f.kc},
                       {"th", f.th},
                       {"fm", f.fm},
                       {"sh", f.sh},
                       {"sw", f.sw},
                       {"ft", f.ft},
                       {"sd", f.sd}};
    return kernel;
}

StemBwdConfig configOf(const ConfigFields& f)
{
    StemBwdConfig c;
    c.atom = f.atom;
    c.atomId = f.atomId;
    c.layout = f.layout;
    c.vec = f.vec;
    c.u = f.u;
    c.v = f.v;
    c.nb = f.nb;
    c.kc = f.kc;
    c.th = f.th;
    c.fm = f.fm;
    c.sh = f.sh;
    c.sw = f.sw;
    c.ft = f.ft;
    c.sd = f.sd;
    return c;
}

std::string dtypeName(DataType dataType)
{
    return dataType == DataType::FLOAT ? "FLOAT" : dataType == DataType::HALF ? "HALF" : "BFLOAT16";
}

hipdnn_plugin_sdk::ingestor::DeviceProperties deviceOf(const std::string& arch)
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = 304;
    properties.warpSize = arch.rfind("gfx1", 0) == 0 ? 32 : 64;
    return properties;
}

bool admits(const ConfigFields& fields,
            const std::string& arch = "gfx942",
            const ConvShape& shape = baseShape())
{
    const GraphFixture fixture(buildGraph(specOf(shape)), deviceOf(arch));
    const auto bound = matchesGraph(PACK, fixture.context());
    return bound.has_value()
           && matchesKernel(
               PACK, fixture.context(), kernelOf(fields, dtypeName(shape.dataType)), *bound);
}

std::optional<conv_stem_bwd::StemBwdLaunch> launchOf(const ConfigFields& fields,
                                                     const ConvShape& shape = baseShape(),
                                                     const std::string& arch = "gfx942")
{
    const auto problem = parse(specOf(shape));
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    return conv_stem_bwd::stemBwdLaunchFor(*problem, configOf(fields), deviceOf(arch));
}

ConvShape floatShape()
{
    auto shape = baseShape();
    shape.dataType = DataType::FLOAT;
    return shape;
}

ConvShape channelsLastShape()
{
    auto shape = baseShape();
    shape.channelsLast = true;
    return shape;
}

ConfigFields channelsLastFields()
{
    ConfigFields f;
    f.layout = 1;
    return f;
}

struct ConfigRefusal
{
    std::string name;
    std::function<void(ConfigFields&)> mutate;
    std::function<ConfigFields()> start = [] { return ConfigFields{}; };
    std::string arch = "gfx942";
    std::function<ConvShape()> shape = baseShape;
};

std::vector<ConfigRefusal> configRefusals()
{
    std::vector<ConfigRefusal> cases{
        {"Direction", [](ConfigFields& f) { f.direction = "WRW"; }},
        // The atom: known, its id, on this device, the graph's element type, no TF32 form.
        {"UnknownAtom", [](ConfigFields& f) { f.atom = "mfma64_f16"; }},
        {"AtomIdOfAnotherAtom", [](ConfigFields& f) { f.atomId = 4; }},
        {"AtomOfAnotherDevice",
         [](ConfigFields& f) {
             f.atom = "wmma11_f16";
             f.atomId = 12;
         }},
        {"AtomOfAnotherElement",
         [](ConfigFields& f) {
             f.atom = "mfma16_bf16";
             f.atomId = 1;
         }},
        {"FmaOnHalf",
         [](ConfigFields& f) {
             f.atom = "fma";
             f.atomId = 16;
         }},
        // The layout and its vector.
        {"ChannelsLastLayoutOnChannelsFirst", [](ConfigFields& f) { f.layout = 1; }},
        {"ChannelsFirstLayoutOnChannelsLast",
         [](ConfigFields& f) { f.layout = 0; },
         channelsLastFields,
         "gfx942",
         channelsLastShape},
        {"UnknownLayout", [](ConfigFields& f) { f.layout = 2; }},
        {"ChannelsFirstVectorNotSixteenBytes", [](ConfigFields& f) { f.vec = 4; }},
        {"ChannelsLastScalarVector",
         [](ConfigFields& f) { f.vec = 1; },
         channelsLastFields,
         "gfx942",
         channelsLastShape},
        // The window and the strides.
        {"WindowRows", [](ConfigFields& f) { f.u = 3; }},
        {"WindowColumns", [](ConfigFields& f) { f.v = 5; }},
        {"HeightStride", [](ConfigFields& f) { f.sh = 1; }},
        {"WidthStride", [](ConfigFields& f) { f.sw = 4; }},
        // The depth taps: 2-D configs on a trivial depth axis, 3-D ones on the problem's taps.
        {"DepthTapsOnATrivialDepth", [](ConfigFields& f) { f.ft = 1; }},
        {"TwoDOnADepthAxis",
         [](ConfigFields& f) { f.ft = 0; },
         [] {
             ConfigFields f;
             f.ft = 3;
             return f;
         },
         "gfx942",
         [] { return threeDStem(3); }},
        // Space-to-depth: 2, on stride 1 and no depth taps, its own window, the columns every
        // (channel, 2 x 2 phase).
        {"SpaceToDepthOnStrideTwo", [](ConfigFields& f) { f.sd = 2; }},
        {"SpaceToDepthOtherThanTwo",
         [](ConfigFields& f) { f.sd = 3; },
         spaceToDepthFields,
         "gfx942",
         stride1Shape},
        {"SpaceToDepthOffOnItsWindow",
         [](ConfigFields& f) { f.sd = 0; },
         spaceToDepthFields,
         "gfx942",
         stride1Shape},
        {"SpaceToDepthOnThePlainWindow",
         [](ConfigFields& f) {
             f.sd = 2;
             f.u = 3;
             f.v = 5;
         },
         [] {
             auto f = spaceToDepthFields();
             f.sd = 0;
             f.u = 3;
             f.v = 5;
             return f;
         },
         "gfx942",
         stride1Shape},
        // Space-to-depth reads whole-vector rows as vectors only (dy rows of 32).
        {"SpaceToDepthScalarsOnWholeVectorRows",
         [](ConfigFields& f) { f.vec = 1; },
         [] {
             auto f = spaceToDepthFields();
             f.vec = 8;
             return f;
         },
         "gfx942",
         [] {
             auto shape = stride1Shape();
             shape.input = {29, 32};
             return shape;
         }},
        {"SpaceToDepthColumnsBelowThePhases",
         [](ConfigFields& f) { f.nb = 11; },
         [] {
             auto f = fmaFields();
             f.vec = 1;
             f.u = 2;
             f.v = 3;
             f.sh = f.sw = 1;
             f.sd = 2;
             return f;
         },
         "gfx942",
         [] {
             auto shape = stride1Shape();
             shape.dataType = DataType::FLOAT;
             return shape;
         }},
        {"SpaceToDepthOnADepthAxis",
         [](ConfigFields& f) {
             f.sd = 2;
             f.u = 2;
             f.v = 2;
         },
         [] {
             auto f = ConfigFields{};
             f.vec = 1;
             f.u = f.v = 3;
             f.sh = f.sw = 1;
             f.ft = 3;
             return f;
         },
         "gfx942",
         [] {
             auto shape = threeDStem(3);
             shape.filter = {3, 3, 3};
             shape.prePadding = shape.postPadding = {1, 1, 1};
             shape.stride = {1, 1, 1};
             return shape;
         }},
        {"DepthTapsOtherThanTheProblems",
         [](ConfigFields& f) { f.ft = 1; },
         [] {
             ConfigFields f;
             f.ft = 3;
             return f;
         },
         "gfx942",
         [] { return threeDStem(3); }},
        // The GEMM columns.
        {"ColumnsBelowThePhases",
         [](ConfigFields& f) {
             f.atom = "fma";
             f.atomId = 16;
             f.nb = 11;
         },
         [] {
             auto f = fmaFields();
             return f;
         },
         "gfx942",
         floatShape},
        {"ColumnsNotWholeFragments", [](ConfigFields& f) { f.nb = 24; }},
        // On gfx950, whose LDS holds 80 columns' filter chunk.
        {"ColumnsBeyondTheMost",
         [](ConfigFields& f) { f.nb = 80; },
         [] { return ConfigFields{}; },
         "gfx950"},
        // The chunk.
        {"ChunkNotWholeAtomSteps", [](ConfigFields& f) { f.kc = 8; }},
        // mfma32_f32 steps 2 output channels: 6 is whole steps but not whole 16-byte units.
        {"ChunkNotWholeVectors",
         [](ConfigFields& f) { f.kc = 6; },
         [] {
             ConfigFields f;
             f.atom = "mfma32_f32";
             f.atomId = 7;
             f.vec = 4;
             f.nb = 32;
             f.kc = 8;
             f.fm = 1;
             return f;
         },
         "gfx942",
         floatShape},
        {"ChunkZero", [](ConfigFields& f) { f.kc = 0; }},
        // The tile, its block and its LDS.
        {"NoRows", [](ConfigFields& f) { f.th = 0; }},
        {"NoFragments", [](ConfigFields& f) { f.fm = 0; }},
        {"BlockBeyond1024",
         [](ConfigFields& f) { f.th = 17; },
         [] {
             ConfigFields f;
             f.fm = 1;
             return f;
         }},
        {"LdsBeyondTheDevices", [](ConfigFields& f) { f.kc = 256; }},
    };
    return cases;
}

class TestConvStemBwdConfigRefuses : public ::testing::TestWithParam<ConfigRefusal>
{
};

TEST_P(TestConvStemBwdConfigRefuses, OneField)
{
    const auto& refusal = GetParam();
    auto fields = refusal.start();
    const auto shape = refusal.shape();
    ASSERT_TRUE(admits(fields, refusal.arch, shape));
    refusal.mutate(fields);
    EXPECT_FALSE(admits(fields, refusal.arch, shape));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvStemBwdConfigRefuses,
                         ::testing::ValuesIn(configRefusals()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvStemBwdKernelMatch, TheBaseConfigsServeTheBaseProblem)
{
    EXPECT_TRUE(admits(ConfigFields{}));
    EXPECT_TRUE(admits(channelsLastFields(), "gfx942", channelsLastShape()));
    EXPECT_TRUE(admits(fmaFields(), "gfx942", floatShape()));
    EXPECT_TRUE(admits(fmaFields(), "gfx1100", floatShape()));
    // The dtype must be the graph's.
    const GraphFixture fixture(buildGraph(specOf(baseShape())), deviceOf("gfx942"));
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(
        matchesKernel(PACK, fixture.context(), kernelOf(ConfigFields{}, "BFLOAT16"), *bound));
}

TEST(TestConvStemBwdKernelMatch, FmaTakes16BitOnlyWithoutAMatrixCoreForIt)
{
    // 16-bit FMA (8-element vectors, chunks of 16): gfx906 and gfx900 (no matrix core) only.
    auto f = fmaFields();
    f.vec = 8;
    f.kc = 16;
    EXPECT_TRUE(admits(f, "gfx906"));
    EXPECT_TRUE(admits(f, "gfx900"));
    EXPECT_FALSE(admits(f, "gfx942"));
    EXPECT_FALSE(admits(f, "gfx1100"));
    auto bf16 = baseShape();
    bf16.dataType = DataType::BFLOAT16;
    EXPECT_TRUE(admits(f, "gfx906", bf16));
    EXPECT_FALSE(admits(f, "gfx908", bf16)); // gfx908's own bf16 atom
    EXPECT_FALSE(conv_stem_bwd::runsMatrixAtomFor("f16", "gfx906:sramecc+:xnack-"));
    EXPECT_TRUE(conv_stem_bwd::runsMatrixAtomFor("bf16", "gfx908"));
    EXPECT_FALSE(conv_stem_bwd::runsMatrixAtomFor("f32", "gfx1201"));
}

TEST(TestConvStemBwdKernelMatch, ChannelsFirstVectorsNeedWholeVectorRowsOfDy)
{
    auto shape = baseShape();
    shape.input = {29, 63}; // dy rows of 32
    EXPECT_TRUE(admits(ConfigFields{}, "gfx942", shape));
    shape.input = {29, 61}; // dy rows of 31
    EXPECT_FALSE(admits(ConfigFields{}, "gfx942", shape));
    auto scalar = ConfigFields{};
    scalar.vec = 1;
    EXPECT_TRUE(admits(scalar, "gfx942", shape));
}

TEST(TestConvStemBwdKernelMatch, ChannelsLastNeedsWholeVectorsOfOutputChannels)
{
    auto shape = channelsLastShape();
    shape.k = 24;
    EXPECT_TRUE(admits(channelsLastFields(), "gfx942", shape));
    shape.k = 20;
    EXPECT_FALSE(admits(channelsLastFields(), "gfx942", shape));
}

TEST(TestConvStemBwdKernelMatch, ColumnsCoverEveryChannelTimesPhase)
{
    // 4 channels at stride 2 take all 16 columns; at stride 4, 3 channels take 48.
    auto four = baseShape();
    four.c = 4;
    EXPECT_TRUE(admits(ConfigFields{}, "gfx942", four));
    auto patchify = baseShape();
    patchify.input = {32, 64};
    patchify.filter = {4, 4};
    patchify.prePadding = patchify.postPadding = {0, 0};
    patchify.stride = {4, 4};
    auto f = ConfigFields{};
    f.u = f.v = 1;
    f.sh = f.sw = 4;
    f.nb = 32;
    EXPECT_FALSE(admits(f, "gfx942", patchify));
    f.nb = 48;
    EXPECT_TRUE(admits(f, "gfx942", patchify));
}

TEST(TestConvStemBwdKernelMatch, TheAtomsRunOnTheirArchitectures)
{
    auto rdna = ConfigFields{};
    rdna.atom = "wmma11_f16";
    rdna.atomId = 12;
    EXPECT_TRUE(admits(rdna, "gfx1100"));
    EXPECT_FALSE(admits(rdna, "gfx1201"));
    EXPECT_FALSE(admits(rdna, "gfx942"));
    auto gfx950 = ConfigFields{};
    gfx950.atom = "mfma16x32_f16";
    gfx950.atomId = 8;
    EXPECT_TRUE(admits(gfx950, "gfx950"));
    EXPECT_FALSE(admits(gfx950, "gfx942"));
    // gfx950's LDS takes what gfx942's does not.
    gfx950.kc = 128;
    EXPECT_TRUE(admits(gfx950, "gfx950"));
    auto big = ConfigFields{};
    big.kc = 128;
    EXPECT_FALSE(admits(big, "gfx942"));
    EXPECT_TRUE(admits(big, "gfx950"));
}

TEST(TestConvStemBwdKernelMatch, TheGridLimitsTheSuperRowsPerBlock)
{
    // 262148 super-rows: 65537 blocks of 4 rows, 32769 of 8.
    auto shape = baseShape();
    shape.n = 1;
    shape.k = 1;
    shape.input = {524296, 4};
    auto f = ConfigFields{};
    f.vec = 1;
    f.fm = 1;
    EXPECT_FALSE(admits(f, "gfx942", shape));
    f.th = 8;
    EXPECT_TRUE(admits(f, "gfx942", shape));
}

TEST(TestConvStemBwdKernelMatch, TheTransformedFilterIsOneBuffer)
{
    // 16 columns x 16 offsets x kp fp16 within 2^31 bytes: kp 2^22 is one element past.
    auto shape = baseShape();
    shape.n = 1;
    shape.input = {2, 2};
    shape.k = (int64_t{1} << 22) - 32;
    auto f = ConfigFields{};
    f.vec = 1;
    EXPECT_TRUE(admits(f, "gfx942", shape));
    shape.k = int64_t{1} << 22;
    EXPECT_FALSE(admits(f, "gfx942", shape));
}

TEST(TestConvStemBwdKernelMatch, MissingMetadataRefuses)
{
    const GraphFixture fixture(buildGraph(specOf(baseShape())), deviceOf("gfx942"));
    const auto bound = matchesGraph(PACK, fixture.context());
    ASSERT_TRUE(bound.has_value());
    for(const char* field : {"atom",
                             "atom_id",
                             "layout",
                             "vec",
                             "u",
                             "v",
                             "nb",
                             "kc",
                             "th",
                             "fm",
                             "sh",
                             "sw",
                             "ft",
                             "sd",
                             "direction"})
    {
        auto kernel = kernelOf(ConfigFields{}, "HALF");
        kernel.metadata.erase(field);
        EXPECT_FALSE(matchesKernel(PACK, fixture.context(), kernel, *bound)) << field;
    }
}

TEST(TestConvStemBwdAtoms, IdsFollowTheTable)
{
    for(size_t i = 0; i < conv_stem_bwd::ATOMS.size(); ++i)
    {
        EXPECT_EQ(conv_stem_bwd::atomIdOf(conv_stem_bwd::ATOMS[i]), static_cast<int64_t>(i));
    }
    EXPECT_EQ(conv_stem_bwd::atomIdOf("mfma16_xf32"), -1);
    EXPECT_EQ(conv_stem_bwd::atomShapeOf("mfma16x32_bf16").ka, 32);
    EXPECT_EQ(conv_stem_bwd::atomShapeOf("mfma32_f16").dim, 32);
    EXPECT_EQ(conv_stem_bwd::atomShapeOf("fma").dim, 16);
    EXPECT_EQ(conv_stem_bwd::atomShapeOf("nope").dim, 0);
}

// ---------------------------------------------------------------------------
// The launch
// ---------------------------------------------------------------------------

TEST(TestConvStemBwdLaunch, GridsArgumentsAndTheWorkspace)
{
    const auto launch = launchOf(ConfigFields{});
    ASSERT_TRUE(launch.has_value());
    // dx 29 x 64: 15 x 32 super-pixels; 32 per block row (fm 2 x 16), 4 rows per block.
    EXPECT_EQ(launch->gridX, 1u);
    EXPECT_EQ(launch->gridY, 4u);
    EXPECT_EQ(launch->gridZ, 2u);
    EXPECT_EQ(launch->block, 256u);
    EXPECT_EQ(launch->kernelType, "K1AtomMfmaF16");
    // B: 16 columns x 16 offsets x 32 (16 output channels padded to the chunk), fp16.
    EXPECT_EQ(launch->workspaceBytes, size_t{16 * 16 * 32 * 2});
    EXPECT_EQ(launch->transformGrid, 32u);
    const auto& a = launch->args;
    EXPECT_EQ(a.n, 2);
    EXPECT_EQ(a.c, 3);
    EXPECT_EQ(a.k, 16);
    EXPECT_EQ(a.kp, 32);
    EXPECT_EQ(a.hi, 29);
    EXPECT_EQ(a.wi, 64);
    EXPECT_EQ(a.ho, 15);
    EXPECT_EQ(a.wo, 32);
    EXPECT_EQ(a.oh, 2);
    EXPECT_EQ(a.ow, 2);
    EXPECT_EQ(a.dys[0], 16 * 15 * 32);
    EXPECT_EQ(a.dys[1], 15 * 32);
    EXPECT_EQ(a.dys[2], 32);
    EXPECT_EQ(a.dys[3], 1);
    EXPECT_EQ(a.dxs[0], 3 * 29 * 64);
    EXPECT_EQ(a.dxs[1], 29 * 64);
    EXPECT_EQ(a.dxs[2], 64);
    EXPECT_EQ(a.dxs[3], 1);
    EXPECT_EQ(a.dyImage, 16 * 15 * 32);
    const auto& t = launch->transform;
    EXPECT_EQ(t.nb, 16);
    EXPECT_EQ(t.u, 4);
    EXPECT_EQ(t.v, 4);
    EXPECT_EQ(t.kp, 32);
    EXPECT_EQ(t.c, 3);
    EXPECT_EQ(t.k, 16);
    EXPECT_EQ(t.r, 7);
    EXPECT_EQ(t.s, 7);
    EXPECT_EQ(t.sh, 2);
    EXPECT_EQ(t.sw, 2);
    EXPECT_EQ(t.ph, 3);
    EXPECT_EQ(t.pw, 3);
    EXPECT_EQ(t.oh, 2);
    EXPECT_EQ(t.ow, 2);
    EXPECT_EQ(t.flip, 0);
    EXPECT_EQ(t.ws[0], 3 * 49);
    EXPECT_EQ(t.ws[1], 49);
    EXPECT_EQ(t.ws[2], 7);
    EXPECT_EQ(t.ws[3], 1);
    // 2-D: one plane, the reduction one tap's.
    EXPECT_EQ(a.d, 1);
    EXPECT_EQ(a.kpr, 32);
    EXPECT_EQ(t.ft, 0);
    EXPECT_EQ(t.kpr, 32);
}

TEST(TestConvStemBwdLaunch, SpaceToDepthTakesTwoByTwoPhasesAndSubPixelsInTheReduction)
{
    const auto launch = launchOf(spaceToDepthFields(), stride1Shape());
    ASSERT_TRUE(launch.has_value());
    // dx 29 x 30: 15 x 15 super-pixels of 2 x 2.
    EXPECT_EQ(launch->gridX, 1u);
    EXPECT_EQ(launch->gridY, 4u);
    EXPECT_EQ(launch->gridZ, 2u);
    // B: 16 columns x 2 x 3 offsets x 4 sub-pixels x 32.
    EXPECT_EQ(launch->workspaceBytes, size_t{16 * 6 * 4 * 32 * 2});
    const auto& a = launch->args;
    EXPECT_EQ(a.kp, 4 * 32);
    EXPECT_EQ(a.kpr, 32);
    EXPECT_EQ(a.ho, 29);
    EXPECT_EQ(a.wo, 30);
    EXPECT_EQ(a.oh, 0);
    EXPECT_EQ(a.ow, 1);
    EXPECT_EQ(a.sdh, 1);
    EXPECT_EQ(a.sdw, 0);
    const auto& t = launch->transform;
    EXPECT_EQ(t.sh, 2);
    EXPECT_EQ(t.sw, 2);
    EXPECT_EQ(t.oh, 0);
    EXPECT_EQ(t.ow, 1);
    EXPECT_EQ(t.dh, 1);
    EXPECT_EQ(t.dw, 0);
    EXPECT_EQ(t.u, 2);
    EXPECT_EQ(t.v, 3);
    EXPECT_EQ(t.kp, 4 * 32);
    // Channels-first column vectors (dy rows of 32): a chunk is one sub-row and both
    // sub-columns of kc / 2 = 16 channels, so kp = 2 sub-rows x 1 chunk x 32.
    auto vectors = spaceToDepthFields();
    vectors.vec = 8;
    auto wide = stride1Shape();
    wide.input = {29, 32};
    const auto columnVectors = launchOf(vectors, wide);
    ASSERT_TRUE(columnVectors.has_value());
    EXPECT_EQ(columnVectors->args.kp, 2 * 32);
    EXPECT_EQ(columnVectors->args.kpr, 16);
    EXPECT_EQ(columnVectors->transform.kp, 2 * 32);
    // Rows that are not whole vectors (30) decline it.
    auto narrow = stride1Shape();
    narrow.input = {29, 30};
    EXPECT_FALSE(launchOf(vectors, narrow).has_value());

    // Off: the stride's phases, no shift.
    auto plain = spaceToDepthFields();
    plain.sd = 0;
    plain.u = 3;
    plain.v = 5;
    const auto off = launchOf(plain, stride1Shape());
    ASSERT_TRUE(off.has_value());
    EXPECT_EQ(off->gridY, 8u);
    EXPECT_EQ(off->args.sdh, 0);
    EXPECT_EQ(off->transform.sh, 1);
    EXPECT_EQ(off->args.kp, 32);
}

TEST(TestConvStemBwdLaunch, ThreeDFoldsPlanesIntoTheGridAndTapsIntoTheReduction)
{
    // 3 x 7 x 7, 5 dx planes (dy's 5, depth padding 1), packed N C D H W.
    auto f = ConfigFields{};
    f.ft = 3;
    const auto launch = launchOf(f, threeDStem(3));
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->gridZ, 2u * 5u);
    // B: 16 columns x 16 offsets x 3 taps x 32.
    EXPECT_EQ(launch->workspaceBytes, size_t{16 * 16 * 3 * 32 * 2});
    const auto& a = launch->args;
    EXPECT_EQ(a.kp, 3 * 32);
    EXPECT_EQ(a.kpr, 32);
    EXPECT_EQ(a.d, 5);
    EXPECT_EQ(a.dout, 5);
    EXPECT_EQ(a.pd, 1);
    EXPECT_EQ(a.dys[0], 16 * 5 * 15 * 32);
    EXPECT_EQ(a.dys[1], 5 * 15 * 32);
    EXPECT_EQ(a.dyd, 15 * 32);
    EXPECT_EQ(a.dxs[1], 5 * 29 * 64);
    EXPECT_EQ(a.dxd, 29 * 64);
    EXPECT_EQ(a.dyImage, 16 * 5 * 15 * 32);
    const auto& t = launch->transform;
    EXPECT_EQ(t.kp, 3 * 32);
    EXPECT_EQ(t.kpr, 32);
    EXPECT_EQ(t.ft, 3);
    EXPECT_EQ(t.ws[0], 3 * 3 * 49);
    EXPECT_EQ(t.ws[1], 3 * 49);
    EXPECT_EQ(t.wsd, 49);
}

TEST(TestConvStemBwdLaunch, LayoutsStridesAndTheFlip)
{
    auto shape = channelsLastShape();
    shape.mode = ConvMode::CONVOLUTION;
    auto spec = specOf(shape);
    spec.x.strides = packedStrides(spec.x.dims, false); // dx channels-first under NHWC dy
    const auto problem = parse(spec);
    ASSERT_TRUE(problem.has_value());
    const auto launch = conv_stem_bwd::stemBwdLaunchFor(
        *problem, configOf(channelsLastFields()), deviceOf("gfx942"));
    ASSERT_TRUE(launch.has_value());
    const auto& a = launch->args;
    EXPECT_EQ(a.dys[0], 15 * 32 * 16);
    EXPECT_EQ(a.dys[1], 1);
    EXPECT_EQ(a.dys[2], 32 * 16);
    EXPECT_EQ(a.dys[3], 16);
    EXPECT_EQ(a.dxs[1], 29 * 64);
    EXPECT_EQ(a.dxs[3], 1);
    const auto& t = launch->transform;
    EXPECT_EQ(t.flip, 1);
    // K R S C.
    EXPECT_EQ(t.ws[0], 49 * 3);
    EXPECT_EQ(t.ws[1], 1);
    EXPECT_EQ(t.ws[2], 7 * 3);
    EXPECT_EQ(t.ws[3], 3);
}

TEST(TestConvStemBwdLaunch, TheFmaLoopTakesSixteenThreadsARow)
{
    const auto launch = launchOf(fmaFields(), floatShape(), "gfx1100");
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->block, 256u);
    EXPECT_EQ(launch->kernelType, "");
    EXPECT_EQ(launch->gridX, 1u); // 32 super-columns, 2 x 16 per block
    EXPECT_EQ(launch->gridY, 1u); // 15 super-rows, 16 per block
    EXPECT_EQ(launch->workspaceBytes, size_t{12 * 16 * 16 * 4});
}

TEST(TestConvStemBwdLaunch, OneDimensionalGraphsPadTheLeadingAxes)
{
    auto shape = baseShape();
    shape.input = {50};
    shape.filter = {7};
    shape.prePadding = shape.postPadding = {3};
    shape.stride = {2};
    shape.dilation = {1};
    auto f = ConfigFields{};
    f.u = 1;
    f.sh = 1;
    f.vec = 1;
    const auto launch = launchOf(f, shape);
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->args.hi, 1);
    EXPECT_EQ(launch->args.ho, 1);
    EXPECT_EQ(launch->args.oh, 0);
    EXPECT_EQ(launch->transform.r, 1);
    EXPECT_EQ(launch->transform.sh, 1);
}

TEST(TestConvStemBwdLds, MirrorsTheKernelsArithmetic)
{
    // (window rows x columns + nb x u v) x (kc + 16-byte pad) elements.
    EXPECT_EQ(conv_stem_bwd::ldsBytesOf(configOf(ConfigFields{}), 2),
              ((4 + 3) * (32 + 3) + 16 * 16) * (32 + 8) * 2);
    EXPECT_EQ(conv_stem_bwd::ldsBytesOf(configOf(fmaFields()), 4),
              ((16 + 3) * (32 + 3) + 12 * 16) * (8 + 4) * 4);
    auto zero = ConfigFields{};
    zero.th = 0;
    EXPECT_EQ(conv_stem_bwd::ldsBytesOf(configOf(zero), 2), std::numeric_limits<int64_t>::max());
}

TEST(TestConvStemBwdDefines, EveryFieldIsADefine)
{
    const auto defines = conv_stem_bwd::compileDefinesFor(
        DataType::BFLOAT16, configOf(ConfigFields{}), "K1AtomMfmaBf16");
    ASSERT_EQ(defines.size(), 12u);
    EXPECT_EQ(defines[0], (std::pair<std::string, std::string>{"CONV_SB_T", "unsigned short"}));
    EXPECT_EQ(defines[1], (std::pair<std::string, std::string>{"CONV_SB_ATOM", "K1AtomMfmaBf16"}));
    EXPECT_EQ(defines[2], (std::pair<std::string, std::string>{"CONV_SB_LAYOUT", "0"}));
    EXPECT_EQ(defines[4], (std::pair<std::string, std::string>{"CONV_SB_U", "4"}));
    EXPECT_EQ(defines[11], (std::pair<std::string, std::string>{"CONV_SB_SW", "2"}));
    // FMA names no atom.
    const auto fma = conv_stem_bwd::compileDefinesFor(DataType::FLOAT, configOf(fmaFields()), "");
    ASSERT_EQ(fma.size(), 11u);
    EXPECT_EQ(fma[0], (std::pair<std::string, std::string>{"CONV_SB_T", "float"}));
    EXPECT_EQ(fma[1], (std::pair<std::string, std::string>{"CONV_SB_LAYOUT", "0"}));
    // 3-D only: the depth taps.
    auto threeD = ConfigFields{};
    threeD.ft = 3;
    const auto taps
        = conv_stem_bwd::compileDefinesFor(DataType::HALF, configOf(threeD), "K1AtomMfmaF16");
    ASSERT_EQ(taps.size(), 13u);
    EXPECT_EQ(taps[12], (std::pair<std::string, std::string>{"CONV_SB_FT", "3"}));
    const auto sd = conv_stem_bwd::compileDefinesFor(
        DataType::HALF, configOf(spaceToDepthFields()), "K1AtomMfmaF16");
    ASSERT_EQ(sd.size(), 13u);
    EXPECT_EQ(sd[12], (std::pair<std::string, std::string>{"CONV_SB_SD", "2"}));
    EXPECT_THROW(static_cast<void>(conv_stem_bwd::compileDefinesFor(
                     DataType::DOUBLE, configOf(ConfigFields{}), "")),
                 std::invalid_argument);
}

TEST(TestConvStemBwdComputeType, ServesCudnnsComputeTypesPerElementType)
{
    expectCudnnComputeTypes(baseShape(),
                            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
                            [](const ConvShape& s) { return parse(specOf(s)).has_value(); });
}

} // namespace

#endif
