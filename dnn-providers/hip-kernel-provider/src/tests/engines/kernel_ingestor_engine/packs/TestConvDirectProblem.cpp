// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvDirectProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvDirectProblem.cpp
 * @brief Host-only checks of the stride-1 halo-tile direct convolution engines
 *        (CONV_DIRECT_FWD, CONV_DIRECT_BWD): what the graph match accepts, one refusal per gated
 *        field, the forward form it derives, what the kernel match admits per config and device
 *        (atom family, ship list, dtype, filter, channel vector, three-bf16 for fp32 only), and the
 *        planner's tile, split and workspace. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_direct::parseDirectProblem;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

constexpr std::array<ConvDirection, 2> DIRECTIONS
    = {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA};

/// The base problem every refusal mutates: 3x3, stride 1, padding 1, NCHW.
ConvShape baseShape()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 8;
    shape.k = 16;
    shape.input = {7, 6};
    shape.filter = {3, 3};
    shape.prePadding = {1, 1};
    shape.postPadding = {1, 1};
    return shape;
}

ConvShape filterShape(std::vector<int64_t> filter, std::vector<int64_t> pad)
{
    auto shape = baseShape();
    shape.input.assign(filter.size(), 9);
    shape.filter = std::move(filter);
    shape.prePadding = pad;
    shape.postPadding = std::move(pad);
    shape.stride.assign(shape.input.size(), 1);
    shape.dilation.assign(shape.input.size(), 1);
    return shape;
}

std::optional<conv_direct::DirectProblem> parse(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return parseDirectProblem(fixture.context(), spec.direction);
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
    cases.push_back({"Nchw", baseShape()});
    auto nhwc = baseShape();
    nhwc.channelsLast = true;
    cases.push_back({"Nhwc", nhwc});
    auto pad0 = baseShape();
    pad0.prePadding = pad0.postPadding = {0, 0};
    cases.push_back({"Pad0", pad0});
    auto pad2 = baseShape();
    pad2.prePadding = pad2.postPadding = {2, 2};
    cases.push_back({"Pad2", pad2});
    auto asymmetric = baseShape();
    asymmetric.prePadding = {2, 0};
    asymmetric.postPadding = {0, 1};
    cases.push_back({"AsymmetricPadding", asymmetric});
    auto convolution = baseShape();
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"ConvolutionMode", convolution});
    auto half = baseShape();
    half.dataType = DataType::HALF;
    cases.push_back({"Half", half});
    auto bfloat = baseShape();
    bfloat.dataType = DataType::BFLOAT16;
    cases.push_back({"BFloat16", bfloat});
    auto oddChannels = baseShape();
    oddChannels.c = 3;
    oddChannels.k = 5;
    cases.push_back({"OddChannels", oddChannels});
    cases.push_back({"Filter5x5", filterShape({5, 5}, {2, 2})});
    cases.push_back({"Filter7x7", filterShape({7, 7}, {3, 3})});
    cases.push_back({"Filter1x7", filterShape({1, 7}, {0, 3})});
    cases.push_back({"Filter7x1", filterShape({7, 1}, {3, 0})});
    cases.push_back({"Filter3x3x3", filterShape({3, 3, 3}, {1, 1, 1})});
    cases.push_back({"Filter5x5x5", filterShape({5, 5, 5}, {2, 2, 2})});
    auto ndhwc = filterShape({3, 3, 3}, {0, 1, 2});
    ndhwc.channelsLast = true;
    cases.push_back({"Ndhwc3x3x3", ndhwc});
    return cases;
}

class TestConvDirectAccepts : public ::testing::TestWithParam<std::tuple<AcceptCase, ConvDirection>>
{
};

TEST_P(TestConvDirectAccepts, ForwardAndBackwardData)
{
    const auto& [accept, direction] = GetParam();
    EXPECT_TRUE(parse(specFor(accept.shape, direction)).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvDirectAccepts,
                         ::testing::Combine(::testing::ValuesIn(acceptCases()),
                                            ::testing::ValuesIn(DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

TEST(TestConvDirectForm, ForwardKeepsTheGraphsRolesAndStagesNchw)
{
    auto shape = baseShape();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    shape.mode = ConvMode::CONVOLUTION;
    const auto problem = parse(specFor(shape, ConvDirection::FORWARD));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    EXPECT_EQ(a.c, 8);
    EXPECT_EQ(a.k, 16);
    EXPECT_EQ(a.hi, 7);
    EXPECT_EQ(a.wi, 6);
    EXPECT_EQ(a.ho, 7);
    EXPECT_EQ(a.wo, 5);
    EXPECT_EQ(a.di, 1);
    EXPECT_EQ(a.dout, 1);
    EXPECT_EQ(a.ph, 2);
    EXPECT_EQ(a.pw, 0);
    EXPECT_EQ(problem->flip, 1);
    EXPECT_TRUE(problem->in.needed);
    EXPECT_TRUE(problem->in.transpose);
    EXPECT_TRUE(problem->out.needed);
    EXPECT_GT(problem->stagingBytes, size_t{0});
}

TEST(TestConvDirectForm, BackwardDataSwapsChannelsAndPadsByFilterMinusOneMinusPre)
{
    auto shape = baseShape();
    shape.prePadding = {2, 0};
    shape.postPadding = {0, 1};
    shape.channelsLast = true;
    const auto problem = parse(specFor(shape, ConvDirection::BACKWARD_DATA));
    ASSERT_TRUE(problem.has_value());
    const auto& a = problem->args;
    EXPECT_EQ(a.c, 16);
    EXPECT_EQ(a.k, 8);
    EXPECT_EQ(a.hi, 7);
    EXPECT_EQ(a.wi, 5);
    EXPECT_EQ(a.ho, 7);
    EXPECT_EQ(a.wo, 6);
    EXPECT_EQ(a.ph, 0);
    EXPECT_EQ(a.pw, 2);
    EXPECT_EQ(problem->flip, 0);
    EXPECT_FALSE(problem->in.needed);
    EXPECT_FALSE(problem->out.needed);
    EXPECT_EQ(problem->stagingBytes, size_t{0});
}

TEST(TestConvDirectForm, OneRowImagesTakeLineFiltersAndTheirDilation)
{
    // 2-D of one row, 1 x 11 at dilation 5 (the height dilation spans no taps).
    auto row = baseShape();
    row.input = {1, 64};
    row.filter = {1, 11};
    row.dilation = {1, 5};
    row.prePadding = row.postPadding = {0, 25};
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto problem = parse(specFor(row, direction));
        ASSERT_TRUE(problem.has_value());
        EXPECT_TRUE(problem->oneRow);
        EXPECT_EQ(problem->dilation, 5);
        EXPECT_EQ(problem->spatialRank, 2);
        EXPECT_EQ(problem->args.hi, 1);
        EXPECT_EQ(problem->args.ph, 0);
        EXPECT_EQ(problem->args.pw, 25);
    }
    // 1-D: a one-row image of rank 1.
    auto line = baseShape();
    line.input = {40};
    line.filter = {7};
    line.stride = {1};
    line.dilation = {3};
    line.prePadding = line.postPadding = {9};
    const auto oneD = parse(specFor(line, ConvDirection::FORWARD));
    ASSERT_TRUE(oneD.has_value());
    EXPECT_TRUE(oneD->oneRow);
    EXPECT_EQ(oneD->spatialRank, 1);
    EXPECT_EQ(oneD->dilation, 3);
    EXPECT_EQ(oneD->args.wo, 40);
    // A 3 x 3 image is not one row.
    const auto square = parse(specFor(baseShape(), ConvDirection::FORWARD));
    ASSERT_TRUE(square.has_value());
    EXPECT_FALSE(square->oneRow);
}

TEST(TestConvDirectForm, DilationIsKeptAndBackwardDataPadsByTheDilatedReach)
{
    auto shape = baseShape();
    shape.input = {9, 10};
    shape.dilation = {3, 3};
    shape.prePadding = {2, 3};
    shape.postPadding = {4, 3};
    shape.channelsLast = true;
    const auto forward = parse(specFor(shape, ConvDirection::FORWARD));
    ASSERT_TRUE(forward.has_value());
    EXPECT_EQ(forward->dilation, 3);
    EXPECT_EQ(forward->args.ph, 2);
    EXPECT_EQ(forward->args.pw, 3);
    EXPECT_EQ(forward->args.ho, 9);
    EXPECT_EQ(forward->args.wo, 10);
    const auto backward = parse(specFor(shape, ConvDirection::BACKWARD_DATA));
    ASSERT_TRUE(backward.has_value());
    EXPECT_EQ(backward->dilation, 3);
    // (3 - 1) x 3 - pre.
    EXPECT_EQ(backward->args.ph, 4);
    EXPECT_EQ(backward->args.pw, 3);
    const auto undilated = parse(specFor(baseShape(), ConvDirection::FORWARD));
    ASSERT_TRUE(undilated.has_value());
    EXPECT_EQ(undilated->dilation, 1);
}

// ---------------------------------------------------------------------------
// One refusal per gated field
// ---------------------------------------------------------------------------

struct RefusalCase
{
    std::string name;
    std::function<void(ConvGraphSpec&)> mutate;
};

TensorSpec& written(ConvGraphSpec& spec)
{
    return spec.direction == ConvDirection::FORWARD ? spec.y : spec.x;
}

/// Rebuilds the operands of @p spec from @p shape (a consistent change of several fields).
void reshape(ConvGraphSpec& spec, const ConvShape& shape)
{
    const auto rebuilt = specFor(shape, spec.direction);
    spec.x = rebuilt.x;
    spec.w = rebuilt.w;
    spec.y = rebuilt.y;
    spec.prePadding = rebuilt.prePadding;
    spec.postPadding = rebuilt.postPadding;
    spec.stride = rebuilt.stride;
    spec.dilation = rebuilt.dilation;
}

std::vector<RefusalCase> refusalCases()
{
    return {
        // Graph
        {"OverrideShapeEnabled", [](ConvGraphSpec& s) { s.overrideShapeEnabled = true; }},
        // Node
        {"ComputeTypeDouble", [](ConvGraphSpec& s) { s.computeType = DataType::DOUBLE; }},
        {"ModeUnset", [](ConvGraphSpec& s) { s.mode = ConvMode::UNSET; }},
        {"OutputAliasesInput", [](ConvGraphSpec& s) { s.yUid = s.xUid; }},
        {"MissingTensor", [](ConvGraphSpec& s) { s.omitW = true; }},
        // Tensor fields
        {"VirtualOperand", [](ConvGraphSpec& s) { s.x.isVirtual = true; }},
        {"RuntimePassByValue", [](ConvGraphSpec& s) { s.w.runtimePassByValue = true; }},
        {"RaggedOffset", [](ConvGraphSpec& s) { s.y.raggedOffsetUid = 7; }},
        {"RaggedMultiplier", [](ConvGraphSpec& s) { s.y.raggedOffsetMultiplier = 2; }},
        {"UnderAligned", [](ConvGraphSpec& s) { s.x.alignment = 2; }},
        {"Double",
         [](ConvGraphSpec& s) { s.x.dataType = s.w.dataType = s.y.dataType = DataType::DOUBLE; }},
        {"MixedDtype", [](ConvGraphSpec& s) { s.w.dataType = DataType::HALF; }},
        {"StridesRankMismatch", [](ConvGraphSpec& s) { s.x.strides.pop_back(); }},
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
        {"WrongOutputExtent",
         [](ConvGraphSpec& s) {
             s.y.dims[2] += 1;
             s.y.strides = packedStrides(s.y.dims, false);
         }},
        {"FilterChannelsDoNotDivideInput", [](ConvGraphSpec& s) { s.w.dims[1] = 3; }},
        {"BatchMismatch",
         [](ConvGraphSpec& s) {
             s.y.dims[0] = 3;
             s.y.strides = packedStrides(s.y.dims, false);
         }},
        // The direct envelope: 2-D or 3-D, groups 1, stride 1, dilation 1 (or one 2-D dilation of
        // both filter axes), an instantiated filter.
        {"OneDFilterNotALineWidth",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.input = {11};
             shape.filter = {9};
             shape.prePadding = shape.postPadding = {4};
             shape.stride = shape.dilation = {1};
             reshape(s, shape);
         }},
        {"LineWidthOnTwoRows",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.input = {2, 9};
             shape.filter = {1, 11};
             shape.prePadding = shape.postPadding = {0, 5};
             reshape(s, shape);
         }},
        {"Grouped",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.groups = 2;
             reshape(s, shape);
         }},
        {"StrideNot1",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.stride = {1, 2};
             reshape(s, shape);
         }},
        {"DilationUnequalAxes",
         [](ConvGraphSpec& s) {
             auto shape = baseShape();
             shape.dilation = {2, 1};
             shape.prePadding = shape.postPadding = {2, 1};
             reshape(s, shape);
         }},
        {"Dilation3d",
         [](ConvGraphSpec& s) {
             auto shape = filterShape({3, 3, 3}, {2, 2, 2});
             shape.input = {6, 7, 6};
             shape.dilation = {2, 2, 2};
             reshape(s, shape);
         }},
        {"Filter3x5", [](ConvGraphSpec& s) { reshape(s, filterShape({3, 5}, {1, 2})); }},
        {"Filter1x1", [](ConvGraphSpec& s) { reshape(s, filterShape({1, 1}, {0, 0})); }},
        {"Filter7x7x7", [](ConvGraphSpec& s) { reshape(s, filterShape({7, 7, 7}, {3, 3, 3})); }},
        {"Filter3x3x1", [](ConvGraphSpec& s) { reshape(s, filterShape({3, 3, 1}, {1, 1, 0})); }},
    };
}

class TestConvDirectRefuses
    : public ::testing::TestWithParam<std::tuple<RefusalCase, ConvDirection>>
{
};

TEST_P(TestConvDirectRefuses, OneField)
{
    const auto& [refusal, direction] = GetParam();
    auto spec = specFor(baseShape(), direction);
    ASSERT_TRUE(parse(spec).has_value());
    refusal.mutate(spec);
    EXPECT_FALSE(parse(spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvDirectRefuses,
                         ::testing::Combine(::testing::ValuesIn(refusalCases()),
                                            ::testing::ValuesIn(DIRECTIONS)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

TEST(TestConvDirectDirection, RefusesWeightGradientsAndTheOtherDirection)
{
    const GraphFixture forward(buildGraph(specFor(baseShape(), ConvDirection::FORWARD)));
    EXPECT_FALSE(parseDirectProblem(forward.context(), ConvDirection::BACKWARD_DATA).has_value());
    EXPECT_FALSE(
        parseDirectProblem(forward.context(), ConvDirection::BACKWARD_WEIGHTS).has_value());
    const GraphFixture wrw(buildGraph(specFor(baseShape(), ConvDirection::BACKWARD_WEIGHTS)));
    EXPECT_FALSE(parseDirectProblem(wrw.context(), ConvDirection::BACKWARD_WEIGHTS).has_value());
}

// A pointwise epilogue: forward graphs carry it only where the engine asks for it; a backward-data
// graph with a pointwise node after its convolution is never one the engine serves.
TEST(TestConvDirectFusion, ForwardTakesTheEpilogueOnlyWhenAsked)
{
    using namespace conv_epilogue_test;
    auto shape = baseShape();
    shape.channelsLast = true;
    const GraphFixture fused(
        buildFusedGraph(fusedSpecFor(shape, {conv_epilogue_test::bias(), relu()})));
    const auto problem
        = parseDirectProblem(fused.context(), ConvDirection::FORWARD, conv_naive::Fusion::EPILOGUE);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->yUid, outputUid(2));
    EXPECT_FALSE(problem->base.epilogue.empty());
    EXPECT_FALSE(parseDirectProblem(fused.context(), ConvDirection::FORWARD).has_value());
    EXPECT_TRUE(
        matchesGraph(
            PackSymbols{"", "hipkernel.conv_direct_fwd.graph_match", "", "", "", "", "", "", ""},
            fused.context())
            .has_value());
}

TEST(TestConvDirectFusion, BackwardDataDeclinesAPointwiseNodeAfterItsConvolution)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    auto spec = specFor(baseShape(), ConvDirection::BACKWARD_DATA);
    const GraphFixture bare(buildGraph(spec));
    const PackSymbols bwd{"", "hipkernel.conv_direct_bwd.graph_match", "", "", "", "", "", "", ""};
    ASSERT_TRUE(matchesGraph(bwd, bare.context()).has_value());

    // dx virtual, then relu(dx) -> DX.
    constexpr int64_t OUT = 100;
    spec.x.isVirtual = true;
    const TensorSpec out = specFor(baseShape(), ConvDirection::BACKWARD_DATA).x;
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensors{
        serializeTensor(builder, spec.xUid, spec.x),
        serializeTensor(builder, spec.wUid, spec.w),
        serializeTensor(builder, spec.yUid, spec.y),
        serializeTensor(builder, OUT, out)};
    const auto conv = data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                                         spec.yUid,
                                                                         spec.wUid,
                                                                         spec.xUid,
                                                                         &spec.prePadding,
                                                                         &spec.postPadding,
                                                                         &spec.stride,
                                                                         &spec.dilation,
                                                                         spec.mode)
                          .Union();
    const auto relu = data_objects::CreatePointwiseAttributes(builder,
                                                              data_objects::PointwiseMode::RELU_FWD,
                                                              ::flatbuffers::nullopt,
                                                              ::flatbuffers::nullopt,
                                                              ::flatbuffers::nullopt,
                                                              ::flatbuffers::nullopt,
                                                              spec.xUid,
                                                              ::flatbuffers::nullopt,
                                                              ::flatbuffers::nullopt,
                                                              OUT)
                          .Union();
    const std::vector<::flatbuffers::Offset<data_objects::Node>> nodes{
        data_objects::CreateNodeDirect(builder,
                                       "dgrad",
                                       DataType::FLOAT,
                                       data_objects::NodeAttributes::ConvolutionBwdAttributes,
                                       conv),
        data_objects::CreateNodeDirect(builder,
                                       "relu",
                                       DataType::FLOAT,
                                       data_objects::NodeAttributes::PointwiseAttributes,
                                       relu)};
    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "dgrad_relu",
                                                   DataType::UNSET,
                                                   DataType::UNSET,
                                                   DataType::UNSET,
                                                   &tensors,
                                                   &nodes));
    const GraphFixture fused(std::move(builder));
    EXPECT_FALSE(matchesGraph(bwd, fused.context()).has_value());
    EXPECT_FALSE(parseDirectProblem(
                     fused.context(), ConvDirection::BACKWARD_DATA, conv_naive::Fusion::EPILOGUE)
                     .has_value());
}

TEST(TestConvDirectSize, RefusesAnImageBeyondTheBufferOffsets)
{
    auto shape = baseShape();
    shape.n = 1;
    shape.c = 8;
    shape.k = 8;
    shape.input = {16384, 16400};
    shape.channelsLast = true;
    // 16384 x 16400 x 8 fp32: one image of x is 2^33 bytes.
    EXPECT_FALSE(parse(specFor(shape, ConvDirection::FORWARD)).has_value());
}

// ---------------------------------------------------------------------------
// Kernel match: configs per device
// ---------------------------------------------------------------------------

constexpr PackSymbols PACK_FWD{"",
                               "hipkernel.conv_direct_fwd.graph_match",
                               "",
                               "hipkernel.conv_direct.kernel_match",
                               "",
                               "hipkernel.conv_direct.dispatch",
                               "",
                               "",
                               ""};

struct ConfigFields
{
    std::string kind = "mma";
    std::string atom = "mfma16_f16";
    std::string archs = "gfx942";
    int64_t math = 0;
    int64_t kd = 1, r = 3, s = 3;
    int64_t bm = 128, bn = 64, bc = 32, wm = 2, wn = 2, vec = 8;
    int64_t ring = 1, bl = 0, split = 0;
    int64_t tm = 0, tn = 0, nt = 0, threads = 0, pipe = 0;
    int64_t rows = 0, pw = 0, fn = 0, wp = 0, wq = 0, wk = 0, pp = 0, epi = 0, un = 0, cf = 0;
    int64_t xpf = 0; ///< written only when set (an absent field reads as 0)
    int64_t dil = 1;
    int64_t line = 0;
    int64_t bpu = 1;
    int64_t lin = 0; ///< written only when set
};

/// A gfx950 rows config: 2 x 2 x 2 waves of 8 rows x 16 columns x 64 channels, 64-channel chunks.
ConfigFields rowsFields()
{
    ConfigFields f;
    f.kind = "rows";
    f.atom = "mfma16x32_f16";
    f.archs = "gfx950";
    f.bm = 16;
    f.bn = 128;
    f.bc = 64;
    f.wm = f.wn = 0;
    f.vec = 8;
    f.ring = 0;
    f.rows = 1;
    f.pw = 8;
    f.fn = 4;
    f.wp = f.wq = f.wk = 2;
    f.epi = 1;
    f.un = 1;
    return f;
}

ConfigFields fmaFields(int64_t vec)
{
    ConfigFields f;
    f.kind = "fma";
    f.atom = "fma";
    f.archs = "any";
    f.bm = 128;
    f.bn = 64;
    f.bc = 8;
    f.wm = f.wn = 0;
    f.vec = vec;
    f.tm = 8;
    f.tn = 4;
    f.nt = 16;
    f.threads = 256;
    return f;
}

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const ConfigFields& f,
                                                       const std::string& dtype)
{
    auto kernel = makeKernel(0, dtype, "ConvDirect");
    kernel.source.sourceFile = "kernels/ConvDirect.cpp";
    kernel.metadata = {{"direction", std::string("FWD")},
                       {"dtype", dtype},
                       {"kind", f.kind},
                       {"atom", f.atom},
                       {"archs", f.archs},
                       {"atom_id", int64_t{0}},
                       {"math", f.math},
                       {"kd", f.kd},
                       {"r", f.r},
                       {"s", f.s},
                       {"bm", f.bm},
                       {"bn", f.bn},
                       {"bc", f.bc},
                       {"wm", f.wm},
                       {"wn", f.wn},
                       {"vec", f.vec},
                       {"ring", f.ring},
                       {"bl", f.bl},
                       {"split", f.split},
                       {"tm", f.tm},
                       {"tn", f.tn},
                       {"nt", f.nt},
                       {"threads", f.threads},
                       {"pipe", f.pipe},
                       {"rows", f.rows},
                       {"pw", f.pw},
                       {"fn", f.fn},
                       {"wp", f.wp},
                       {"wq", f.wq},
                       {"wk", f.wk},
                       {"pp", f.pp},
                       {"epi", f.epi},
                       {"un", f.un},
                       {"cf", f.cf},
                       {"dil", f.dil},
                       {"line", f.line},
                       {"bpu", f.bpu}};
    if(f.xpf != 0)
    {
        kernel.metadata["xpf"] = f.xpf;
    }
    if(f.lin != 0)
    {
        kernel.metadata["lin"] = f.lin;
    }
    return kernel;
}

bool admits(const ConfigFields& fields,
            const std::string& arch,
            ConvShape shape = baseShape(),
            const std::string& dtype = "HALF",
            ConvDirection direction = ConvDirection::FORWARD)
{
    if(dtype == "FLOAT")
    {
        shape.dataType = DataType::FLOAT;
    }
    else if(dtype == "HALF")
    {
        shape.dataType = DataType::HALF;
    }
    else
    {
        shape.dataType = DataType::BFLOAT16;
    }
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = 64;
    const GraphFixture fixture(buildGraph(specFor(shape, direction)), properties);
    auto pack = PACK_FWD;
    auto kernel = kernelOf(fields, dtype);
    if(direction == ConvDirection::BACKWARD_DATA)
    {
        pack.graphMatcher = "hipkernel.conv_direct_bwd.graph_match";
        kernel.metadata["direction"] = std::string("BWD");
    }
    const auto bound = matchesGraph(pack, fixture.context());
    return bound.has_value() && matchesKernel(pack, fixture.context(), kernel, *bound);
}

TEST(TestConvDirectKernelMatch, MatrixCoreConfigsRunOnlyWhereTheirAtomAndShipListAllow)
{
    const ConfigFields f;
    EXPECT_TRUE(admits(f, "gfx942:sramecc+:xnack-"));
    // The atom exists on gfx90a, but this config does not ship there.
    EXPECT_FALSE(admits(f, "gfx90a"));
    // Listed for gfx1100, but no MFMA there.
    auto rdnaListed = f;
    rdnaListed.archs = "gfx942,gfx11";
    EXPECT_FALSE(admits(rdnaListed, "gfx1100"));
    auto wmma = f;
    wmma.atom = "wmma11_f16";
    wmma.archs = "gfx11";
    wmma.wm = 4;
    EXPECT_TRUE(admits(wmma, "gfx1100"));
    EXPECT_TRUE(admits(wmma, "gfx1151"));
    EXPECT_FALSE(admits(wmma, "gfx1201"));
    EXPECT_FALSE(admits(wmma, "gfx942"));
}

TEST(TestConvDirectKernelMatch, TheAtomTakesTheGraphsElementType)
{
    const ConfigFields f;
    EXPECT_TRUE(admits(f, "gfx942", baseShape(), "HALF"));
    EXPECT_FALSE(admits(f, "gfx942", baseShape(), "BFLOAT16"));
    EXPECT_FALSE(admits(f, "gfx942", baseShape(), "FLOAT"));
    auto exact = f;
    exact.atom = "mfma16_f32";
    exact.bc = 16;
    exact.vec = 4;
    EXPECT_TRUE(admits(exact, "gfx942", baseShape(), "FLOAT"));
}

TEST(TestConvDirectKernelMatch, TheFilterMustBeTheConfigs)
{
    ConfigFields f;
    f.r = f.s = 5;
    EXPECT_FALSE(admits(f, "gfx942"));
    EXPECT_TRUE(admits(f, "gfx942", filterShape({5, 5}, {2, 2})));
    ConfigFields threeD;
    threeD.kd = 3;
    threeD.ring = 0;
    EXPECT_FALSE(admits(threeD, "gfx942"));
    EXPECT_TRUE(admits(threeD, "gfx942", filterShape({3, 3, 3}, {1, 1, 1})));
}

TEST(TestConvDirectKernelMatch, TheChannelVectorMustDivideTheReduction)
{
    auto odd = baseShape();
    odd.c = 12;
    const ConfigFields f;
    EXPECT_FALSE(admits(f, "gfx942", odd));
    EXPECT_TRUE(admits(fmaFields(4), "gfx942", odd));
    odd.c = 3;
    EXPECT_FALSE(admits(fmaFields(4), "gfx942", odd));
    auto scalar = fmaFields(1);
    scalar.bc = 4;
    scalar.tm = 4;
    scalar.tn = 8;
    scalar.nt = 8;
    EXPECT_TRUE(admits(scalar, "gfx942", odd));
}

TEST(TestConvDirectKernelMatch, WideColumnsAllowOutputTailsButNotReductionVectorTails)
{
    for(const auto& dtype : {std::string("HALF"), std::string("BFLOAT16")})
    {
        SCOPED_TRACE(dtype);
        ConfigFields f;
        f.atom = dtype == "HALF" ? "wmma125_f16" : "wmma125_bf16";
        f.archs = "gfx1250";
        f.bm = 64;
        f.bn = 128;
        f.wn = 4;
        auto shape = baseShape();
        shape.c = 40;
        shape.k = 133;
        EXPECT_TRUE(admits(f, "gfx1250", shape, dtype, ConvDirection::FORWARD));
        EXPECT_FALSE(admits(f, "gfx1250", shape, dtype, ConvDirection::BACKWARD_DATA));
        std::swap(shape.c, shape.k);
        EXPECT_FALSE(admits(f, "gfx1250", shape, dtype, ConvDirection::FORWARD));
        EXPECT_TRUE(admits(f, "gfx1250", shape, dtype, ConvDirection::BACKWARD_DATA));
    }
}

TEST(TestConvDirectKernelMatch, FmaRunsEverywhere)
{
    EXPECT_TRUE(admits(fmaFields(8), "gfx906", baseShape(), "HALF"));
    EXPECT_TRUE(admits(fmaFields(4), "gfx1030", baseShape(), "FLOAT"));
    EXPECT_TRUE(admits(fmaFields(8), "gfx1201", baseShape(), "BFLOAT16"));
}

TEST(TestConvDirectKernelMatch, ThreeBf16OnlyForFloat)
{
    ConfigFields x3;
    x3.atom = "mfma16_bf16";
    x3.math = 1;
    EXPECT_TRUE(admits(x3, "gfx942", baseShape(), "FLOAT"));
    EXPECT_FALSE(admits(x3, "gfx942", baseShape(), "BFLOAT16"));
    auto halfAtom = x3;
    halfAtom.atom = "mfma16_f16";
    EXPECT_FALSE(admits(halfAtom, "gfx942", baseShape(), "FLOAT"));
}

TEST(TestConvDirectKernelMatch, RefusesUnknownMetadata)
{
    auto badKind = ConfigFields{};
    badKind.kind = "winograd";
    EXPECT_FALSE(admits(badKind, "gfx942"));
    auto badAtom = ConfigFields{};
    badAtom.atom = "mfma64_f16";
    EXPECT_FALSE(admits(badAtom, "gfx942"));
    auto fmaAtomOnMma = ConfigFields{};
    fmaAtomOnMma.atom = "fma";
    EXPECT_FALSE(admits(fmaAtomOnMma, "gfx942"));
    auto badFilter = ConfigFields{};
    badFilter.r = 4;
    EXPECT_FALSE(admits(badFilter, "gfx942"));
    auto badChunk = ConfigFields{};
    badChunk.bc = 24; // not a multiple of the atom's reduction step
    EXPECT_FALSE(admits(badChunk, "gfx942"));
    auto badRing = ConfigFields{};
    badRing.ring = 2;
    EXPECT_FALSE(admits(badRing, "gfx942"));
    auto badMath = ConfigFields{};
    badMath.math = 2;
    EXPECT_FALSE(admits(badMath, "gfx942"));
    auto noArchs = ConfigFields{};
    noArchs.archs = "";
    EXPECT_FALSE(admits(noArchs, "gfx942"));
}

TEST(TestConvDirectKernelMatch, RowsConfigsRunOnGfx950WithWholeGranules)
{
    const auto f = rowsFields();
    EXPECT_TRUE(admits(f, "gfx950:sramecc+:xnack-"));
    EXPECT_FALSE(admits(f, "gfx942"));
    EXPECT_FALSE(admits(f, "gfx950", baseShape(), "BFLOAT16"));
    auto bf = f;
    bf.atom = "mfma16x32_bf16";
    EXPECT_TRUE(admits(bf, "gfx950", baseShape(), "BFLOAT16"));
    auto exact = f;
    exact.atom = "mfma16_f32";
    exact.bc = 32;
    exact.vec = 4;
    exact.epi = 0;
    EXPECT_TRUE(admits(exact, "gfx950", baseShape(), "FLOAT"));
    // The halo moves in 16-byte granules: vec must be 16 bytes of the operand.
    auto narrow = f;
    narrow.vec = 4;
    EXPECT_FALSE(admits(narrow, "gfx950"));
    auto odd = baseShape();
    odd.c = 12;
    EXPECT_FALSE(admits(f, "gfx950", odd));
    // 2-D only.
    EXPECT_FALSE(admits(f, "gfx950", filterShape({3, 3, 3}, {1, 1, 1})));
}

/// A gfx90a rows config: the ping-pong tile on the 16x16x16 atoms, 16-channel chunks (39 KiB).
ConfigFields rowsGfx9Fields()
{
    auto f = rowsFields();
    f.atom = "mfma16_f16";
    f.archs = "gfx90a";
    f.bc = 16;
    f.pp = 1;
    f.epi = 0;
    return f;
}

TEST(TestConvDirectKernelMatch, RowsConfigsRunOnGfx90aWithinItsLds)
{
    const auto f = rowsGfx9Fields();
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(f, "HALF"))),
              2 * (2 * 8 + 2) * (2 * 16 + 2) * 16 * 2);
    EXPECT_TRUE(admits(f, "gfx90a:sramecc+:xnack-"));
    EXPECT_FALSE(admits(f, "gfx950"));
    auto bf = f;
    bf.atom = "mfma16_bf16";
    EXPECT_TRUE(admits(bf, "gfx90a", baseShape(), "BFLOAT16"));
    auto exact = f;
    exact.atom = "mfma16_f32";
    exact.bc = 8;
    exact.vec = 4;
    EXPECT_TRUE(admits(exact, "gfx90a", baseShape(), "FLOAT"));
}

TEST(TestConvDirectKernelMatch, RowsConfigsRunOnGfx908OnItsAtoms)
{
    auto f = rowsGfx9Fields();
    f.archs = "gfx908";
    f.pp = 0;
    f.fn = 2;
    f.bn = 64;
    f.epi = 1;
    EXPECT_TRUE(admits(f, "gfx908:sramecc+:xnack-"));
    EXPECT_FALSE(admits(f, "gfx90a"));
    // bf16 runs on the 16x16x8 atom gfx908 has (8-channel slices); the gfx90a+ 16x16x16 bf16
    // atom does not run there.
    auto bf = f;
    bf.atom = "mfma16_bf16_908";
    EXPECT_TRUE(admits(bf, "gfx908", baseShape(), "BFLOAT16"));
    bf.atom = "mfma16_bf16";
    EXPECT_FALSE(admits(bf, "gfx908", baseShape(), "BFLOAT16"));
    // 32-channel chunks fit the 64 KiB part at 8 rows of 4 waves.
    auto wide = f;
    wide.bc = 32;
    wide.pw = 4;
    wide.bm = 8;
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(wide, "HALF"))),
              2 * (2 * 4 + 2) * (2 * 16 + 2) * 32 * 2);
    EXPECT_TRUE(admits(wide, "gfx908"));
}

/// A gfx950 channels-first rows config: 1 x 2 x 4 waves of 8 rows x 16 columns x 64 channels.
ConfigFields rowsCfFields()
{
    auto f = rowsFields();
    f.wp = 1;
    f.wk = 4;
    f.bm = 8;
    f.bn = 256;
    f.epi = 0;
    f.cf = 1;
    return f;
}

/// A packed channels-first 2-D shape whose rows are whole 16-byte runs of 16-bit elements.
ConvShape cfShape()
{
    auto shape = baseShape();
    shape.input = {7, 16};
    return shape;
}

TEST(TestConvDirectKernelMatch, ChannelsFirstRowsTakePackedChannelsFirstActivations)
{
    const auto f = rowsCfFields();
    // Raw stage (10 halo rows x 64 channel rows x 6 runs of 16 bytes) plus one channels-last stage.
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(f, "HALF"))),
              10 * 64 * 6 * 16 + 10 * 34 * 64 * 2);
    EXPECT_TRUE(admits(f, "gfx950:sramecc+:xnack-", cfShape()));
    auto bf = f;
    bf.atom = "mfma16x32_bf16";
    EXPECT_TRUE(admits(bf, "gfx950", cfShape(), "BFLOAT16"));
    // Channels-last activations take the plain rows configs.
    auto last = cfShape();
    last.channelsLast = true;
    EXPECT_FALSE(admits(f, "gfx950", last));
    // Input rows must be whole 16-byte runs: width 6 is 12 bytes; the dword form (cf 2) takes it.
    EXPECT_FALSE(admits(f, "gfx950", baseShape()));
    auto dword = f;
    dword.cf = 2;
    EXPECT_TRUE(admits(dword, "gfx950", baseShape()));
    // gfx942 (dword DMA, gather transpose) within its 64 KiB of LDS.
    auto other = f;
    other.atom = "mfma16_f16";
    other.archs = "gfx942";
    other.bc = 32;
    EXPECT_TRUE(admits(other, "gfx942", cfShape()));
    // Dword DMA takes even widths: width 6 is 12 bytes.
    EXPECT_TRUE(admits(other, "gfx942", baseShape()));
    other.bc = 64;
    EXPECT_FALSE(admits(other, "gfx942", cfShape())) << "raw and channels-last stages past 64 KiB";
    other.bc = 32;
    other.cf = 0;
    EXPECT_TRUE(admits(other, "gfx942", cfShape())) << "the same tile, staged";
    // Not with the fp32 atom, ping-pong, batch unfold, split-C or the staged epilogue.
    auto exact = f;
    exact.atom = "mfma16_f32";
    exact.vec = 4;
    exact.bc = 32;
    EXPECT_FALSE(admits(exact, "gfx950", cfShape(), "FLOAT"));
    const std::vector<std::pair<std::string, std::function<void(ConfigFields&)>>> mutations{
        {"pp",
         [](ConfigFields& f) {
             f.pp = 1;
             f.wk = 2;
             f.fn = 4;
             f.bn = 128;
         }},
        {"un", [](ConfigFields& f) { f.un = 2; }},
        {"split", [](ConfigFields& f) { f.split = 1; }},
        {"epi", [](ConfigFields& f) { f.epi = 1; }},
    };
    for(const auto& [name, mutate] : mutations)
    {
        auto g = rowsCfFields();
        mutate(g);
        EXPECT_FALSE(admits(g, "gfx950", cfShape())) << name;
    }
}

TEST(TestConvDirectKernelMatch, CrossChunkPrefetchTakesRegisterPathHaloTiles)
{
    ConfigFields f;
    f.xpf = 1;
    EXPECT_TRUE(admits(f, "gfx942"));
    auto bf16 = f;
    bf16.atom = "mfma16_bf16";
    EXPECT_TRUE(admits(bf16, "gfx942", baseShape(), "BFLOAT16"));
    // Each field one at a time: the knob's range, register-path weights, channels-last halo.
    const std::vector<std::pair<std::string, std::function<void(ConfigFields&)>>> mutations{
        {"xpf", [](ConfigFields& g) { g.xpf = 2; }},
        {"negative", [](ConfigFields& g) { g.xpf = -1; }},
        {"bl", [](ConfigFields& g) { g.bl = 1; }},
    };
    for(const auto& [name, mutate] : mutations)
    {
        auto g = f;
        mutate(g);
        EXPECT_FALSE(admits(g, "gfx942")) << name;
    }
    // Not the rows or FMA kernels, nor the channels-first halo tile.
    auto rows = rowsFields();
    EXPECT_TRUE(admits(rows, "gfx950"));
    rows.xpf = 1;
    EXPECT_FALSE(admits(rows, "gfx950"));
    auto fma = fmaFields(8);
    EXPECT_TRUE(admits(fma, "gfx942"));
    fma.xpf = 1;
    EXPECT_FALSE(admits(fma, "gfx942"));
    ConfigFields cf;
    cf.atom = "wmma11_f16";
    cf.archs = "gfx11";
    cf.wm = 2;
    cf.wn = 4;
    cf.bn = 128;
    cf.bc = 16;
    cf.cf = 1;
    auto shape = baseShape();
    shape.input = {7, 16};
    EXPECT_TRUE(admits(cf, "gfx1101", shape));
    cf.xpf = 1;
    EXPECT_FALSE(admits(cf, "gfx1101", shape));
}

TEST(TestConvDirectKernelMatch, ChannelsFirstHaloTilesTakePackedChannelsFirstActivations)
{
    ConfigFields f;
    f.atom = "wmma11_f16";
    f.archs = "gfx11";
    f.wm = 2;
    f.wn = 4;
    f.bn = 128;
    f.bc = 16;
    f.cf = 1;
    auto shape = baseShape();
    shape.input = {7, 16};
    EXPECT_TRUE(admits(f, "gfx1101", shape));
    // Channels-last activations take the staged-free plain configs.
    auto last = shape;
    last.channelsLast = true;
    EXPECT_FALSE(admits(f, "gfx1101", last));
    // Input rows of whole 2-element runs: width 15 is not.
    auto odd = shape;
    odd.input = {7, 15};
    EXPECT_FALSE(admits(f, "gfx1101", odd));
    // Outputs at least 8 columns wide.
    auto narrow = shape;
    narrow.input = {7, 4};
    EXPECT_FALSE(admits(f, "gfx1101", narrow));
    // 16-bit matrix-core, register-path weights, no split, cf 0 or 1.
    const std::vector<std::pair<std::string, std::function<void(ConfigFields&)>>> mutations{
        {"cf", [](ConfigFields& g) { g.cf = 2; }},
        {"bl", [](ConfigFields& g) { g.bl = 1; }},
        {"split", [](ConfigFields& g) { g.split = 1; }},
    };
    for(const auto& [name, mutate] : mutations)
    {
        auto g = f;
        mutate(g);
        EXPECT_FALSE(admits(g, "gfx1101", shape)) << name;
    }
    auto exact = f;
    exact.atom = "wmma11_f16";
    EXPECT_FALSE(admits(exact, "gfx1101", shape, "FLOAT"));
}

TEST(TestConvDirectKernelMatch, RowsConfigsDeclineHaloStagesPastTheDeviceLds)
{
    EXPECT_EQ(conv_direct::ldsLimitOf("gfx950:sramecc+:xnack-"), 160 * 1024);
    EXPECT_EQ(conv_direct::ldsLimitOf("gfx942"), 64 * 1024);
    EXPECT_EQ(conv_direct::ldsLimitOf("gfx90a"), 64 * 1024);
    // gfx950's widest tile (153 KiB) on a 64 KiB part: declined by LDS alone.
    auto wide = rowsGfx9Fields();
    wide.bc = 64;
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(wide, "HALF"))), 156672);
    EXPECT_FALSE(admits(wide, "gfx90a"));
    auto fits = wide;
    fits.bc = 16;
    EXPECT_TRUE(admits(fits, "gfx90a"));
    // Doubling the chunk to 32 channels (77 KiB) passes 64 KiB; fp32 doubles the same tile.
    auto edge = rowsGfx9Fields();
    edge.bc = 32;
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(edge, "HALF"))), 78336);
    EXPECT_FALSE(admits(edge, "gfx90a"));
    auto exact = rowsGfx9Fields();
    exact.atom = "mfma16_f32";
    exact.vec = 4;
    EXPECT_FALSE(admits(exact, "gfx90a", baseShape(), "FLOAT")) << "fp32 c16: 78 KiB";
    exact.bc = 8;
    EXPECT_TRUE(admits(exact, "gfx90a", baseShape(), "FLOAT"));
    // gfx950 keeps its 160 KiB.
    auto big = rowsFields();
    EXPECT_TRUE(admits(big, "gfx950"));
    big.bc = 128;
    EXPECT_FALSE(admits(big, "gfx950"));
}

TEST(TestConvDirectKernelMatch, LineConfigsServeOneRowImages)
{
    auto row = baseShape();
    row.input = {1, 64};
    row.filter = {1, 11};
    row.dilation = {1, 5};
    row.prePadding = row.postPadding = {0, 25};
    // 2 x 2 x 2 waves of 8 segments x 16 columns: a 512-column tile of the row.
    auto line = rowsFields();
    line.r = 1;
    line.s = 11;
    line.bc = 32;
    line.dil = 5;
    line.line = 1;
    EXPECT_TRUE(admits(line, "gfx950", row));
    // One staged line of 512 + (11 - 1) x 5 pixels, two stages.
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(line, "HALF"))),
              2 * 562 * 32 * 2);
    auto cf = line;
    cf.cf = 1;
    cf.epi = 0;
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(cf, "HALF"))),
              32 * ((7 + 562 + 7) / 8) * 16 + 562 * 32 * 2);
    // 1 x 11 undilated: (7 + 522 + 7) / 8 = 67 runs, rounded up to whole 16-column blocks.
    auto cf1 = cf;
    cf1.dil = 1;
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(cf1, "HALF"))),
              32 * 68 * 16 + 522 * 32 * 2);
    // Not on a two-row image, nor at another dilation or width.
    auto twoRows = row;
    twoRows.input = {2, 64};
    auto f17 = line;
    f17.s = 7;
    f17.dil = 1;
    auto rows17 = twoRows;
    rows17.filter = {1, 7};
    rows17.dilation = {1, 1};
    rows17.prePadding = rows17.postPadding = {0, 3};
    EXPECT_FALSE(admits(f17, "gfx950", rows17));
    auto d3 = line;
    d3.dil = 3;
    EXPECT_FALSE(admits(d3, "gfx950", row));
    // 1-D graphs: line configs only.
    auto oneD = baseShape();
    oneD.input = {64};
    oneD.filter = {7};
    oneD.prePadding = oneD.postPadding = {3};
    oneD.stride = oneD.dilation = {1};
    EXPECT_TRUE(admits(f17, "gfx950", oneD));
    auto plain17 = rowsFields();
    plain17.r = 1;
    plain17.s = 7;
    plain17.bc = 32;
    EXPECT_FALSE(admits(plain17, "gfx950", oneD)) << "a 2-D rows config on a 1-D graph";
    // Line tiles are 2-D rows of 1 x S filters, one image per column group, no ping-pong or split.
    const std::vector<std::pair<std::string, std::function<void(ConfigFields&)>>> malformed{
        {"line 2", [](ConfigFields& f) { f.line = 2; }},
        {"r 3", [](ConfigFields& f) { f.r = 3; }},
        {"un 2", [](ConfigFields& f) { f.un = 2; }},
        {"pp 1",
         [](ConfigFields& f) {
             f.pp = 1;
             f.epi = 0;
         }},
        {"split", [](ConfigFields& f) { f.split = 1; }},
    };
    for(const auto& [name, mutate] : malformed)
    {
        auto f = line;
        mutate(f);
        EXPECT_FALSE(admits(f, "gfx950", row)) << name;
    }
    ConfigFields mma;
    mma.atom = "mfma16x32_f16";
    mma.archs = "gfx950";
    mma.r = 1;
    mma.s = 7;
    mma.line = 1;
    EXPECT_FALSE(admits(mma, "gfx950", rows17)) << "line on a halo-tile config";
}

TEST(TestConvDirectKernelMatch, DilatedRowsConfigsServeTheirOwnDilation)
{
    auto dilated = baseShape();
    dilated.dilation = {2, 2};
    dilated.prePadding = dilated.postPadding = {2, 2};
    auto d2 = rowsFields();
    d2.bc = 32;
    d2.dil = 2;
    EXPECT_TRUE(admits(d2, "gfx950", dilated));
    EXPECT_FALSE(admits(d2, "gfx950")) << "a dilated config on an undilated graph";
    EXPECT_FALSE(admits(rowsFields(), "gfx950", dilated)) << "an undilated config";
    auto d4 = d2;
    d4.dil = 4;
    EXPECT_FALSE(admits(d4, "gfx950", dilated));
    // The halo tile and FMA kernels serve dilation 1 only.
    ConfigFields mma;
    mma.atom = "mfma16x32_f16";
    mma.archs = "gfx950";
    EXPECT_FALSE(admits(mma, "gfx950", dilated));
    EXPECT_FALSE(admits(fmaFields(8), "gfx950", dilated));
    // Dilation is a 2-D rows knob of one image per column group.
    mma.dil = 2;
    EXPECT_FALSE(admits(mma, "gfx950", dilated));
    auto unfold = d2;
    unfold.un = 2;
    EXPECT_FALSE(admits(unfold, "gfx950", dilated));
    // The halo grows by (3 - 1) x 2 rows and columns: 20 x 36 pixels a stage.
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(d2, "HALF"))),
              2 * 20 * 36 * 32 * 2);
    auto cf = d2;
    cf.cf = 1;
    cf.epi = 0;
    // One channels-last stage plus the raw stage of 8-column runs: (7 + 36 + 7) / 8 = 6 a row.
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(cf, "HALF"))),
              20 * 32 * 6 * 16 + 20 * 36 * 32 * 2);
    // At 64-channel chunks the dilated halo passes gfx950's 160 KiB.
    auto wide = d2;
    wide.bc = 64;
    EXPECT_FALSE(admits(wide, "gfx950", dilated));
}

TEST(TestConvDirectKernelMatch, RowsConfigsTakeTheirOwnDimensionCount)
{
    auto rows3d = rowsFields();
    rows3d.kd = 3;
    const auto cube = filterShape({3, 3, 3}, {1, 1, 1});
    EXPECT_TRUE(admits(rows3d, "gfx950", cube));
    auto rows3dGfx9 = rowsGfx9Fields();
    rows3dGfx9.kd = 3;
    EXPECT_TRUE(admits(rows3dGfx9, "gfx90a", cube));
    // A 3x3x3 config is not a 2-D one, and a 2-D config (kd 1) declines a 3-D problem even when
    // the filter is 1 x 3 x 3: the 2-D kernel walks no planes.
    EXPECT_FALSE(admits(rows3d, "gfx950"));
    const auto flat3d = filterShape({1, 3, 3}, {0, 1, 1});
    ConfigFields mma;
    mma.atom = "mfma16x32_f16";
    mma.archs = "gfx950";
    mma.kd = 1;
    ASSERT_TRUE(admits(mma, "gfx950", flat3d)) << "the halo-tile kernel takes 1x3x3 in 3-D";
    EXPECT_FALSE(admits(rowsFields(), "gfx950", flat3d));
}

TEST(TestConvDirectKernelMatch, RowsConfigsDeclineEachMalformedField)
{
    ASSERT_TRUE(admits(rowsFields(), "gfx950"));
    const std::vector<std::pair<std::string, std::function<void(ConfigFields&)>>> mutations{
        {"rows knob off", [](ConfigFields& f) { f.rows = 0; }},
        {"pw", [](ConfigFields& f) { f.pw = 0; }},
        {"fn", [](ConfigFields& f) { f.fn = 0; }},
        {"wp", [](ConfigFields& f) { f.wp = 0; }},
        {"wq", [](ConfigFields& f) { f.wq = 0; }},
        {"wk", [](ConfigFields& f) { f.wk = 0; }},
        {"pp", [](ConfigFields& f) { f.pp = 2; }},
        {"pp without two halves",
         [](ConfigFields& f) {
             f.pp = 1;
             f.wk = 1;
             f.bn = 64;
         }},
        {"epi", [](ConfigFields& f) { f.epi = 2; }},
        {"un", [](ConfigFields& f) { f.un = 3; }},
        {"cf", [](ConfigFields& f) { f.cf = 3; }},
        {"un 0", [](ConfigFields& f) { f.un = 0; }},
        {"kd (no 2x3x3 filter)", [](ConfigFields& f) { f.kd = 2; }},
        {"ring", [](ConfigFields& f) { f.ring = 1; }},
        {"bl", [](ConfigFields& f) { f.bl = 1; }},
        {"split on ping-pong",
         [](ConfigFields& f) {
             f.split = 1;
             f.pp = 1;
             f.epi = 0;
         }},
        {"wm", [](ConfigFields& f) { f.wm = 2; }},
        {"wn", [](ConfigFields& f) { f.wn = 2; }},
        {"bm", [](ConfigFields& f) { f.bm = 128; }},
        {"bn", [](ConfigFields& f) { f.bn = 64; }},
        {"dil 0", [](ConfigFields& f) { f.dil = 0; }},
    };
    for(const auto& [name, mutate] : mutations)
    {
        auto f = rowsFields();
        mutate(f);
        EXPECT_FALSE(admits(f, "gfx950")) << name;
    }
    // The rows knobs belong to rows configs only.
    for(int field = 0; field < 8; ++field)
    {
        ConfigFields mma;
        mma.atom = "mfma16x32_f16";
        mma.archs = "gfx950";
        ASSERT_TRUE(admits(mma, "gfx950"));
        std::array<int64_t*, 8> knobs{
            &mma.pw, &mma.fn, &mma.wp, &mma.wq, &mma.wk, &mma.pp, &mma.epi, &mma.un};
        *knobs[static_cast<size_t>(field)] = 1;
        EXPECT_FALSE(admits(mma, "gfx950")) << "rows knob " << field << " on an mma config";
    }
    ConfigFields rowsKnobOnMma;
    rowsKnobOnMma.rows = 1;
    EXPECT_FALSE(admits(rowsKnobOnMma, "gfx942"));
}

TEST(TestConvDirectShips, GroupsNameArchitecturesAndFamilies)
{
    EXPECT_TRUE(conv_direct::shipsTo("any", "gfx906"));
    EXPECT_TRUE(conv_direct::shipsTo("gfx90a,gfx942", "gfx942:sramecc+:xnack-"));
    EXPECT_FALSE(conv_direct::shipsTo("gfx90a,gfx942", "gfx950"));
    EXPECT_TRUE(conv_direct::shipsTo("gfx11", "gfx1151"));
    EXPECT_FALSE(conv_direct::shipsTo("gfx11", "gfx1201"));
    EXPECT_TRUE(conv_direct::shipsTo("gfx908,gfx12", "gfx1200"));
    EXPECT_FALSE(conv_direct::shipsTo("", "gfx942"));
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

std::optional<conv_direct::Plan> planOf(const ConvShape& shape,
                                        const ConfigFields& fields,
                                        const std::string& dtype,
                                        int cus,
                                        int xcds = 0)
{
    const auto problem = parse(specFor(shape, ConvDirection::FORWARD));
    const auto config = conv_direct::configOf(kernelOf(fields, dtype));
    if(!problem.has_value() || !config.has_value())
    {
        return std::nullopt;
    }
    return conv_direct::planFor(*problem, *config, cus, xcds);
}

ConfigFields linFields(int64_t lin)
{
    ConfigFields f;
    f.kd = 3;
    f.ring = 0;
    f.bn = 128;
    f.bc = 16;
    f.xpf = 1;
    f.lin = lin;
    return f;
}

TEST(TestConvDirectKernelMatch, LinearTilesTakeRegisterPathNonRingHaloTiles)
{
    auto shape = filterShape({3, 3, 3}, {1, 1, 1});
    shape.input = {5, 6, 7};
    EXPECT_TRUE(admits(linFields(40), "gfx942", shape));
    // Each field one at a time: the knob's range, ring, LDS weights, channels-first, other kinds.
    const std::vector<std::pair<std::string, std::function<void(ConfigFields&)>>> mutations{
        {"negative", [](ConfigFields& g) { g.lin = -1; }},
        {"ring", [](ConfigFields& g) { g.ring = 1; }},
        {"bl", [](ConfigFields& g) { g.bl = 1; }},
        {"cf", [](ConfigFields& g) { g.cf = 1; }},
    };
    for(const auto& [name, mutate] : mutations)
    {
        auto g = linFields(40);
        mutate(g);
        EXPECT_FALSE(admits(g, "gfx942", shape)) << name;
    }
    auto rows = rowsFields();
    rows.lin = 40;
    EXPECT_FALSE(admits(rows, "gfx950"));
    auto fma = fmaFields(8);
    fma.lin = 40;
    EXPECT_FALSE(admits(fma, "gfx942"));
}

std::optional<conv_direct::Plan>
    planOfDirection(const ConvShape& shape, const ConfigFields& fields, ConvDirection direction)
{
    const auto problem = parse(specFor(shape, direction));
    const auto config = conv_direct::configOf(kernelOf(fields, "HALF"));
    if(!problem.has_value() || !config.has_value())
    {
        return std::nullopt;
    }
    return conv_direct::planFor(*problem, *config, 228, 6);
}

TEST(TestConvDirectPlan, LinearTilesCoverTheVirtualRasterOfEachImage)
{
    // Backward data of an unpadded layer: the virtual raster is the output (dx) itself.
    auto shape = filterShape({3, 3, 3}, {0, 0, 0});
    shape.dataType = DataType::HALF;
    shape.channelsLast = true;
    shape.input = {6, 9, 34};
    const auto bwd = planOfDirection(shape, linFields(40), ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(bwd.has_value());
    EXPECT_EQ(bwd->args.th, 9);
    EXPECT_EQ(bwd->args.tw, 34);
    EXPECT_EQ(bwd->args.tilesW, (6 * 9 * 34 + 127) / 128);
    EXPECT_EQ(bwd->args.tilesH, 1);
    EXPECT_EQ(bwd->args.tilesD, 1);
    EXPECT_EQ(bwd->grid[0], static_cast<unsigned>(bwd->args.tilesW * shape.n));
    // Forward of the same layer: rows as wide as the input (two junk columns), planes as tall.
    const auto fwd = planOfDirection(shape, linFields(40), ConvDirection::FORWARD);
    ASSERT_TRUE(fwd.has_value());
    EXPECT_EQ(fwd->args.th, 9);
    EXPECT_EQ(fwd->args.tw, 34);
    EXPECT_EQ(fwd->args.tilesW, (4 * 9 * 34 + 127) / 128);
    // "Same" padding: one junk column and row.
    auto same = shape;
    same.prePadding = same.postPadding = {1, 1, 1};
    const auto padded = planOfDirection(same, linFields(40), ConvDirection::FORWARD);
    ASSERT_TRUE(padded.has_value());
    EXPECT_EQ(padded->args.th, 10);
    EXPECT_EQ(padded->args.tw, 35);
    // Rows wider than the stage holds decline; the wider stage serves them.
    shape.input = {6, 9, 41};
    EXPECT_FALSE(planOfDirection(shape, linFields(40), ConvDirection::BACKWARD_DATA).has_value());
    EXPECT_TRUE(planOfDirection(shape, linFields(136), ConvDirection::BACKWARD_DATA).has_value());
}

TEST(TestConvDirectPlan, HaloTilesTakeXcdRunsOfTwoBlocksAComputeUnit)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    const auto plan = planOf(shape, ConfigFields{}, "HALF", 228, 6);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->args.xcds, 6);
    EXPECT_EQ(plan->args.xcdRun, 2 * 228 / 6);
    // One XCD (or an unknown count) keeps the launch order.
    const auto one = planOf(shape, ConfigFields{}, "HALF", 228, 1);
    ASSERT_TRUE(one.has_value());
    EXPECT_EQ(one->args.xcds, 0);
    EXPECT_EQ(one->args.xcdRun, 0);
    const auto unknown = planOf(shape, ConfigFields{}, "HALF", 228);
    ASSERT_TRUE(unknown.has_value());
    EXPECT_EQ(unknown->args.xcds, 0);
}

TEST(TestConvDirectPlan, RowsTilesWalkPerXcdShares)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.input = {32, 32};
    auto fields = rowsFields();
    fields.archs = "gfx942";
    fields.atom = "mfma16_f16";
    const auto plan = planOf(shape, fields, "HALF", 228, 6);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->args.xcds, 6);
    EXPECT_EQ(plan->args.xcdRun, 0);
}

TEST(TestConvDirectPlan, SplitGridsKeepTheLaunchOrder)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::FLOAT;
    shape.n = 1;
    shape.c = 512;
    shape.k = 64;
    shape.input = {7, 7};
    auto fields = ConfigFields{};
    fields.atom = "mfma16_f32";
    fields.bc = 16;
    fields.vec = 4;
    fields.bl = 1;
    fields.split = 1;
    const auto plan = planOf(shape, fields, "FLOAT", 304, 8);
    ASSERT_TRUE(plan.has_value());
    ASSERT_GT(plan->args.splits, 1);
    EXPECT_EQ(plan->args.xcds, 0);
}

TEST(TestConvDirectPlan, TiesBreakTowardWholeSixteenPixelRows)
{
    // 128 output pixels on a 128-wide map: 16 x 8 and 8 x 16 tiles cost the same (the same halo
    // and tile count); the 16-wide one keeps every 16-lane fragment within one halo row.
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.input = {128, 128};
    const auto plan = planOf(shape, ConfigFields{}, "HALF", 64);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->args.tw % 16, 0);
    EXPECT_EQ(plan->args.th * plan->args.tw, 128);
}

TEST(TestConvDirectPlan, TileFitsTheHaloAndTheWorkspaceHoldsTheWeights)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.c = 40;
    shape.k = 24;
    shape.input = {13, 17};
    const auto plan = planOf(shape, ConfigFields{}, "HALF", 64);
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    EXPECT_GE(a.th, 1);
    EXPECT_LE(a.th * a.tw, 128);
    EXPECT_LE((a.th + 2) * (a.tw + 2), k2HaloMax(128, 3, 3));
    EXPECT_EQ(a.tilesH, (13 + a.th - 1) / a.th);
    EXPECT_EQ(a.tilesW, (17 + a.tw - 1) / a.tw);
    // C 40 -> 2 chunks of 32 (64 formatted); K 24 -> one 64-column block.
    EXPECT_EQ(plan->cpad, 64);
    EXPECT_EQ(a.kpad, 64);
    EXPECT_EQ(a.ca, 64 / 16);
    EXPECT_EQ(plan->weightElems, 9 * 64 * 64);
    EXPECT_EQ(plan->weightBytes, size_t{9} * 64 * 64 * 2);
    EXPECT_EQ(a.splits, 1);
    EXPECT_GE(plan->workspaceBytes, plan->weightBytes);
}

TEST(TestConvDirectPlan, FormattedWeightsStayWithinTheBufferRange)
{
    // The kernels read the formatted weights through one buffer of 31-bit byte offsets: 9 taps
    // x 10880 x 10880 fp16 is 2,130,739,200 bytes and plans; 11008 x 11008 is 2,181,169,152
    // bytes (still under INT_MAX elements) and is declined.
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.c = 10880;
    shape.k = 10880;
    const auto fits = planOf(shape, ConfigFields{}, "HALF", 64);
    ASSERT_TRUE(fits.has_value());
    EXPECT_EQ(fits->weightBytes, size_t{9} * 10880 * 10880 * 2);
    shape.c = 11008;
    shape.k = 11008;
    EXPECT_FALSE(planOf(shape, ConfigFields{}, "HALF", 64).has_value());
}

TEST(TestConvDirectPlan, SplitsOnlyAnUnderFilledGridAndSizesThePartials)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::FLOAT;
    shape.n = 1;
    shape.c = 512;
    shape.k = 64;
    shape.input = {7, 7};
    auto fields = ConfigFields{};
    fields.atom = "mfma16_f32";
    fields.bc = 16;
    fields.vec = 4;
    fields.bl = 1;
    fields.split = 1;
    const auto plan = planOf(shape, fields, "FLOAT", 304);
    ASSERT_TRUE(plan.has_value());
    EXPECT_GT(plan->args.splits, 1);
    EXPECT_LE(plan->args.splits, 16);
    EXPECT_EQ(plan->grid[2], static_cast<unsigned int>(plan->args.splits));
    EXPECT_GE(plan->args.cps, 2);
    EXPECT_EQ(plan->partialBytes,
              static_cast<size_t>(plan->args.splits) * 7 * 7 * 64 * sizeof(float));
    // A grid that fills the units needs no split: the split config declines (its split-0 twin
    // serves the graph in one launch), and so does one with too few channel chunks.
    auto filled = shape;
    filled.n = 64;
    filled.input = {56, 56};
    EXPECT_FALSE(planOf(filled, fields, "FLOAT", 304).has_value());
    auto shallow = shape;
    shallow.c = 48; // 3 chunks of 16
    EXPECT_FALSE(planOf(shallow, fields, "FLOAT", 304).has_value());
    // A config without SPLIT never splits.
    fields.split = 0;
    const auto whole = planOf(shape, fields, "FLOAT", 304);
    ASSERT_TRUE(whole.has_value());
    EXPECT_EQ(whole->args.splits, 1);
    EXPECT_EQ(whole->partialBytes, size_t{0});
    EXPECT_TRUE(planOf(filled, fields, "FLOAT", 304).has_value());
}

TEST(TestConvDirectPlan, ThreeBf16TriplesTheReductionAndStagesTheSplitInput)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    ConfigFields x3;
    x3.atom = "mfma16_bf16";
    x3.math = 1;
    const auto plan = planOf(shape, x3, "FLOAT", 64);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->args.c, 24);
    EXPECT_EQ(plan->cpad, 32);
    EXPECT_EQ(plan->weightBytes, size_t{9} * 32 * 64 * 2);
    EXPECT_EQ(plan->splitInputBytes, size_t{2} * 7 * 6 * 24 * 2);
}

TEST(TestConvDirectPlan, RingConfigsSplitDepthNoFinerThanTheFilter)
{
    auto shape = filterShape({3, 3, 3}, {1, 1, 1});
    shape.input = {12, 8, 8};
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.n = 1;
    ConfigFields ring;
    ring.kd = 3;
    const auto plan = planOf(shape, ring, "HALF", 304);
    ASSERT_TRUE(plan.has_value());
    EXPECT_GE(plan->args.ds, 3);
    ConfigFields stream = ring;
    stream.ring = 0;
    const auto planNr = planOf(shape, stream, "HALF", 304);
    ASSERT_TRUE(planNr.has_value());
    EXPECT_EQ(planNr->args.ds, 1);
}

TEST(TestConvDirectPlan, RowsTilesAreFixedAndTheGridIsOneBlockPerUnitAtMost)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.c = 72;
    shape.k = 136;
    shape.input = {40, 70};
    const auto plan = planOf(shape, rowsFields(), "HALF", 256);
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    EXPECT_EQ(a.th, 16);
    EXPECT_EQ(a.tw, 32);
    EXPECT_EQ(a.tilesH, 3);
    EXPECT_EQ(a.tilesW, 3);
    // K 136 -> two 128-column blocks; C 72 -> two 64-channel chunks of 32-wide slices.
    EXPECT_EQ(a.kpad, 256);
    EXPECT_EQ(plan->cpad, 128);
    EXPECT_EQ(a.ca, 128 / 32);
    const unsigned tiles = 3 * 3 * static_cast<unsigned>(shape.n) * 2;
    EXPECT_EQ(plan->grid[0], tiles);
    EXPECT_EQ(plan->grid[1], 1u);
    EXPECT_EQ(plan->grid[2], 1u);
    const auto few = planOf(shape, rowsFields(), "HALF", 8);
    ASSERT_TRUE(few.has_value());
    EXPECT_EQ(few->grid[0], 8u);
    // One input image past the 32-bit buffer offsets.
    auto huge = shape;
    huge.input = {4096, 4096};
    huge.c = 128;
    EXPECT_FALSE(planOf(huge, rowsFields(), "HALF", 256).has_value());
}

TEST(TestConvDirectPlan, Rows3dTilesWalkEveryOutputPlane)
{
    auto shape = filterShape({3, 3, 3}, {1, 1, 1});
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.c = 64;
    shape.k = 128;
    shape.input = {5, 40, 70};
    auto rows3d = rowsFields();
    rows3d.kd = 3;
    const auto plan = planOf(shape, rows3d, "HALF", 1024);
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    EXPECT_EQ(a.dout, 5);
    EXPECT_EQ(a.di, 5);
    EXPECT_EQ(a.tilesH, 3);
    EXPECT_EQ(a.tilesW, 3);
    EXPECT_EQ(a.tilesD, 1);
    // Tiles: 3 x 3 per plane, 5 planes, 2 images, one 128-column block.
    EXPECT_EQ(plan->grid[0], 3u * 3u * 5u * static_cast<unsigned>(shape.n));
    // Taps x rows x columns of the formatted weights: 27 taps.
    EXPECT_EQ(plan->weightElems, 27LL * 64 * 128);
    // One input image (all planes) past the 32-bit buffer offsets.
    auto huge = shape;
    huge.input = {64, 1024, 1024};
    EXPECT_FALSE(planOf(huge, rows3d, "HALF", 1024).has_value());
}

TEST(TestConvDirectKernelMatch, BatchUnfoldIs2dOnlyAndWithinTheLds)
{
    // One wave column (16 / un columns of each image): the 2 x 2 x 2 tile's unfolded halo rows
    // (2 x 18 pixels) would pass gfx950's 160 KiB.
    auto unfold = rowsFields();
    unfold.wq = 1;
    unfold.un = 2;
    EXPECT_TRUE(admits(unfold, "gfx950"));
    // Two images' halo rows of 8 + 2 pixels, 18 rows, 64 channels, two stages.
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(unfold, "HALF"))),
              2 * (2 * 8 + 2) * (2 * (8 + 2)) * 64 * 2);
    unfold.un = 4;
    EXPECT_TRUE(admits(unfold, "gfx950"));
    auto wide = rowsFields();
    wide.un = 2;
    EXPECT_FALSE(admits(wide, "gfx950")) << "two 34-pixel halo rows of 64 channels: 162 KiB";
    // The dword LDS DMA (gfx90a / gfx94x) unfolds too, within its 64 KiB.
    auto gfx9 = rowsGfx9Fields();
    gfx9.wq = 1;
    ASSERT_TRUE(admits(gfx9, "gfx90a"));
    gfx9.un = 2;
    EXPECT_TRUE(admits(gfx9, "gfx90a"));
    gfx9.un = 4;
    EXPECT_TRUE(admits(gfx9, "gfx90a"));
    // 3-D rows configs walk planes, not image groups.
    auto rows3d = rowsFields();
    rows3d.kd = 3;
    rows3d.wq = 1;
    ASSERT_TRUE(admits(rows3d, "gfx950", filterShape({3, 3, 3}, {1, 1, 1})));
    rows3d.un = 2;
    EXPECT_FALSE(admits(rows3d, "gfx950", filterShape({3, 3, 3}, {1, 1, 1})));
}

TEST(TestConvDirectPlan, RowsSplitServesOnlyAQuarterFilledGrid)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.n = 1;
    shape.c = 640;
    shape.k = 640;
    shape.input = {64, 64};
    auto split = rowsFields();
    split.split = 1;
    // 4 x 2 tiles x 5 column blocks = 40 tiles: under a quarter of 256 units.
    const auto plan = planOf(shape, split, "HALF", 256);
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    // 10 chunks of 64 channels; ceil(256 / 40) = 7 wanted, at most 10 / 2 = 5.
    EXPECT_EQ(a.splits, 5);
    EXPECT_EQ(a.cps, 2);
    EXPECT_EQ(plan->grid[0], 40u * 5u);
    EXPECT_EQ(plan->partialBytes, static_cast<size_t>(5) * 64 * 64 * 640 * sizeof(float));
    // 40 tiles fill a quarter of 128 units or more: the single-pass configs serve it.
    EXPECT_FALSE(planOf(shape, split, "HALF", 128).has_value());
    EXPECT_FALSE(planOf(shape, split, "HALF", 159).has_value());
    EXPECT_TRUE(planOf(shape, split, "HALF", 160).has_value());
    // One chunk per split would be all the partials cost: two chunks at least.
    auto thin = shape;
    thin.c = 64;
    EXPECT_FALSE(planOf(thin, split, "HALF", 256).has_value());
    // The plain tile still serves both.
    EXPECT_TRUE(planOf(shape, rowsFields(), "HALF", 128).has_value());
}

TEST(TestConvDirectPlan, SmallGfx950RowsTilesDeclineManyRoundGrids)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.n = 2;
    shape.c = 64;
    shape.k = 128;
    shape.input = {256, 256};
    // 16 x 16 x 128: a quarter of the reference tile.
    auto small = rowsFields();
    small.wq = 1;
    small.wp = 2;
    small.bm = 16;
    // 16 x 16 tiles x 2 images = 512 tiles: 8 rounds of 64 units, more than 8 of 32.
    EXPECT_TRUE(planOf(shape, small, "HALF", 64).has_value());
    EXPECT_FALSE(planOf(shape, small, "HALF", 32).has_value());
    // The reference tile and the dword-DMA atoms are not affected.
    EXPECT_TRUE(planOf(shape, rowsFields(), "HALF", 32).has_value());
    auto gfx9 = rowsGfx9Fields();
    gfx9.wq = 1;
    EXPECT_TRUE(planOf(shape, gfx9, "HALF", 32).has_value());
}

TEST(TestConvDirectPlan, RowsBlocksPerUnitScaleTheGridAndTheirStages)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.c = 64;
    shape.k = 64;
    shape.input = {64, 64};
    // 8 x 32 x 64, 4 waves: 8 x 2 tiles x 2 images = 32 tiles.
    auto pair = rowsFields();
    pair.wp = 1;
    pair.bm = 8;
    pair.fn = 2;
    pair.bn = 64;
    pair.bc = 32;
    pair.bpu = 2;
    const auto plan = planOf(shape, pair, "HALF", 8);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->grid[0], 16u) << "two resident blocks per unit";
    pair.bpu = 1;
    EXPECT_EQ(planOf(shape, pair, "HALF", 8)->grid[0], 8u);
    pair.bpu = 3;
    EXPECT_EQ(planOf(shape, pair, "HALF", 64)->grid[0], 32u) << "never more blocks than tiles";
    // Every resident block holds its halo stages: 3 x 43,520 bytes fit gfx950's 160 KiB; on a
    // 64 KiB part two 16-row dword tiles do not, two 8-row ones do.
    EXPECT_EQ(conv_direct::rowsLdsBytes(*conv_direct::configOf(kernelOf(pair, "HALF"))),
              2 * 10 * 34 * 32 * 2);
    EXPECT_TRUE(admits(pair, "gfx950", shape));
    pair.bpu = 4;
    EXPECT_FALSE(admits(pair, "gfx950", shape)) << "bpu 1 to 3";
    auto dword = rowsGfx9Fields();
    ASSERT_TRUE(admits(dword, "gfx90a"));
    dword.bpu = 2;
    EXPECT_FALSE(admits(dword, "gfx90a")) << "2 x 39 KiB past 64 KiB";
    dword.wp = 1;
    dword.bm = 8;
    EXPECT_TRUE(admits(dword, "gfx90a")) << "2 x 21 KiB";
    // A rows knob only.
    ConfigFields mma;
    mma.atom = "mfma16x32_f16";
    mma.archs = "gfx950";
    mma.bpu = 2;
    EXPECT_FALSE(admits(mma, "gfx950"));
}

TEST(TestConvDirectPlan, LineTilesAreSegmentsOfTheRowEndToEnd)
{
    auto wide = baseShape();
    wide.input = {1, 1100};
    wide.filter = {1, 11};
    wide.dilation = {1, 5};
    wide.prePadding = wide.postPadding = {0, 25};
    wide.dataType = DataType::HALF;
    auto line = rowsFields();
    line.r = 1;
    line.s = 11;
    line.bc = 32;
    line.dil = 5;
    line.line = 1;
    // 2 x 8 segments of 32 columns: 512 columns of the row a tile, three tiles for 1100.
    const auto plan = planOf(wide, line, "HALF", 64);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->args.th, 1);
    EXPECT_EQ(plan->args.tw, 512);
    EXPECT_EQ(plan->args.tilesW, 3);
    EXPECT_EQ(plan->args.tilesH, 1);
    // A line tile is the reference tile's 512 pixels: 128 tiles on one unit (past 8 rounds) do
    // not decline it.
    auto longRow = wide;
    longRow.input = {1, 65536};
    EXPECT_TRUE(planOf(longRow, line, "HALF", 1).has_value());
}

TEST(TestConvDirectPlan, SmallGfx950RowsTilesCompareTheChannelsTheyCompute)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.n = 2;
    shape.c = 64;
    shape.k = 64;
    shape.input = {256, 256};
    // 16 x 32 x 64: the reference tile's work on 64 output channels.
    auto narrow = rowsFields();
    narrow.fn = 2;
    narrow.bn = 64;
    // 8 x 16 tiles x 2 images = 256 tiles: more than 8 rounds of 16 units.
    EXPECT_TRUE(planOf(shape, narrow, "HALF", 16).has_value());
    // Half its columns: still a smaller tile than the reference on these channels.
    auto half = narrow;
    half.wq = 1;
    EXPECT_FALSE(planOf(shape, half, "HALF", 16).has_value());
    // 128 output channels: the 64-wide tile computes half the reference tile's and declines.
    shape.k = 128;
    EXPECT_FALSE(planOf(shape, narrow, "HALF", 16).has_value());
    EXPECT_TRUE(planOf(shape, rowsFields(), "HALF", 16).has_value());
}

TEST(TestConvDirectPlan, BatchUnfoldTilesPackImagesAlongTheColumns)
{
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    shape.n = 5;
    shape.c = 64;
    shape.k = 128;
    shape.input = {40, 40};
    auto unfold = rowsFields();
    unfold.wq = 1;
    unfold.un = 2;
    const auto plan = planOf(shape, unfold, "HALF", 1024);
    ASSERT_TRUE(plan.has_value());
    const auto& a = plan->args;
    // 16 columns per wave column: 8 columns of each of two images.
    EXPECT_EQ(a.th, 16);
    EXPECT_EQ(a.tw, 8);
    EXPECT_EQ(a.tilesW, 5);
    EXPECT_EQ(a.tilesH, 3);
    // Image groups: 5 images -> 3 (the last holds one image).
    EXPECT_EQ(plan->grid[0], 5u * 3u * 3u);
    // Two images share the 32-bit buffer offsets: an image just under 1 GiB fits alone, not
    // unfolded.
    auto big = shape;
    big.input = {4096, 2048};
    big.c = 64;
    big.k = 16;
    EXPECT_TRUE(planOf(big, rowsFields(), "HALF", 1024).has_value());
    EXPECT_FALSE(planOf(big, unfold, "HALF", 1024).has_value());
}

TEST(TestConvDirectPlan, ADeviceWithoutUnitsIsDeclined)
{
    // The depth and split-C targets are blocks per unit: no reported units, no plan (and no
    // division by a zero count), where any positive count plans.
    auto shape = baseShape();
    shape.channelsLast = true;
    shape.dataType = DataType::HALF;
    EXPECT_TRUE(planOf(shape, ConfigFields{}, "HALF", 1).has_value());
    EXPECT_FALSE(planOf(shape, ConfigFields{}, "HALF", 0).has_value());
    EXPECT_FALSE(planOf(shape, ConfigFields{}, "HALF", -1).has_value());
    auto split = ConfigFields{};
    split.split = 1;
    EXPECT_FALSE(planOf(shape, split, "HALF", 0).has_value());

    auto properties = testDeviceProperties();
    properties.gcnArchName = "gfx942:sramecc+:xnack-";
    for(const int units : {64, 0})
    {
        properties.multiProcessorCount = units;
        shape.channelsLast = false;
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)), properties);
        const auto bound = matchesGraph(PACK_FWD, fixture.context());
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(
            matchesKernel(PACK_FWD, fixture.context(), kernelOf(ConfigFields{}, "HALF"), *bound),
            units > 0);
    }
}

TEST(TestConvDirectComputeType, ServesCudnnsComputeTypesPerElementType)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        expectCudnnComputeTypes(
            baseShape(),
            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
            [direction](const ConvShape& s) { return parse(specFor(s, direction)).has_value(); });
    }
}

} // namespace

#endif
