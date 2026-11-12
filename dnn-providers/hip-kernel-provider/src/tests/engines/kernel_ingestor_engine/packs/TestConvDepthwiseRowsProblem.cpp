// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseRowsProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvDepthwiseRowsProblem.cpp
 * @brief Host-only checks of the channel-row depthwise engines (CONV_DEPTHWISE_ROWS_FWD/BWD):
 *        what each direction's graph match accepts, one refusal per gated config field and per
 *        problem gate of each variant, through the engines' own kernel match, and the launch
 *        sizes. No device is used.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_depthwise_rows::RowsConfig;
using conv_depthwise_rows::rowsLaunchFor;
using conv_depthwise_rows::RowsVariant;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD ? "hipkernel.conv_depthwise_rows_fwd.graph_match"
                                                : "hipkernel.conv_depthwise_rows_bwd.graph_match",
            "",
            "hipkernel.conv_depthwise_rows.kernel_match",
            "",
            "hipkernel.conv_depthwise_rows.dispatch",
            "",
            "",
            ""};
}

/// A config's full KMD: the RowsConfig plus dtype and direction.
struct Fields
{
    std::string dtype = "FLOAT";
    std::string direction = "FWD";
    RowsConfig config;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const Fields& f)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.name = "conv_depthwise_rows_test";
    kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
    kernel.source.sourceFile = "kernels/ConvDepthwiseRows.cpp";
    kernel.source.entryPoint = "ConvDepthwiseRows";
    const auto& c = f.config;
    kernel.metadata = {{"dtype", f.dtype},
                       {"direction", f.direction},
                       {"layout", c.layout},
                       {"variant", c.variant},
                       {"kh", c.kh},
                       {"kw", c.kw},
                       {"tx", c.tx},
                       {"ty", c.ty},
                       {"rw", c.rw},
                       {"rh", c.rh},
                       {"bh", c.bh},
                       {"threads", c.threads},
                       {"stride", c.stride},
                       {"q", c.q},
                       {"lds", c.lds},
                       {"mmax", c.mmax}};
    return kernel;
}

/// The engine's graph match and kernel match, as the ingestor runs them.
bool admits(const Fields& fields, const ConvShape& shape)
{
    const auto direction
        = fields.direction == "BWD" ? ConvDirection::BACKWARD_DATA : ConvDirection::FORWARD;
    const auto pack = packFor(direction);
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    const auto bound = matchesGraph(pack, fixture.context());
    return bound.has_value() && matchesKernel(pack, fixture.context(), kernelOf(fields), *bound);
}

ConvShape depthwise(int64_t channels,
                    std::vector<int64_t> input,
                    std::vector<int64_t> filter,
                    std::vector<int64_t> pad,
                    bool channelsLast = false)
{
    ConvShape shape;
    shape.n = 2;
    shape.c = shape.k = shape.groups = channels;
    shape.input = std::move(input);
    shape.filter = std::move(filter);
    shape.prePadding = pad;
    shape.postPadding = pad;
    shape.stride.assign(shape.input.size(), 1);
    shape.dilation.assign(shape.input.size(), 1);
    shape.channelsLast = channelsLast;
    return shape;
}

/// 1-D, filter 3, dilation 8: the 1-D tile's shape.
ConvShape dilated1d()
{
    auto shape = depthwise(5, {601}, {3}, {8});
    shape.dilation = {8};
    return shape;
}

/// 2-D 3x3, dilation 2: the plane tile's and band's shape.
ConvShape dilated2d()
{
    auto shape = depthwise(5, {17, 19}, {3, 3}, {2, 2});
    shape.dilation = {2, 2};
    return shape;
}

/// ConvTranspose1d: filter 12, stride 2, padding 5.
ConvShape transposed1d(bool channelsLast)
{
    auto shape = depthwise(6, {515}, {12}, {5}, channelsLast);
    shape.stride = {2};
    return shape;
}

Fields tileFields()
{
    Fields f;
    f.config.layout = "NCHW";
    f.config.variant = static_cast<int64_t>(RowsVariant::TILE);
    f.config.kh = 1;
    f.config.kw = 3;
    f.config.tx = 256;
    f.config.ty = 1;
    f.config.rh = 1;
    f.config.rw = 4;
    f.config.threads = 256;
    f.config.lds = 4096;
    return f;
}

Fields tile2dFields()
{
    Fields f;
    f.config.layout = "NCHW";
    f.config.variant = static_cast<int64_t>(RowsVariant::TILE);
    f.config.kh = 3;
    f.config.kw = 3;
    f.config.tx = 32;
    f.config.ty = 8;
    f.config.rh = 4;
    f.config.rw = 2;
    f.config.threads = 256;
    f.config.lds = 8192;
    return f;
}

Fields bandFields()
{
    Fields f;
    f.config.layout = "NCHW";
    f.config.variant = static_cast<int64_t>(RowsVariant::BAND);
    f.config.kh = 3;
    f.config.kw = 3;
    f.config.bh = 16;
    f.config.rh = 4;
    f.config.threads = 512;
    f.config.lds = 8192;
    return f;
}

Fields phaseFieldsIn(const std::string& layout)
{
    Fields f;
    f.direction = "BWD";
    f.config.layout = layout;
    f.config.variant = static_cast<int64_t>(RowsVariant::PHASE);
    f.config.kw = 12;
    f.config.stride = 2;
    f.config.q = 2;
    f.config.threads = 256;
    f.config.mmax = 6;
    return f;
}

Fields phaseFields()
{
    return phaseFieldsIn("NCHW");
}

Fields lanesFields()
{
    Fields f;
    f.direction = "BWD";
    f.config.layout = "NHWC";
    f.config.variant = static_cast<int64_t>(RowsVariant::LANES);
    f.config.kw = 12;
    f.config.stride = 2;
    f.config.threads = 256;
    return f;
}

// ---- What each direction accepts ----

TEST(TestConvDepthwiseRowsGraph, ForwardServesChannelsFirstOneAndTwoD)
{
    const auto pack = packFor(ConvDirection::FORWARD);
    for(const auto& shape : {dilated1d(), dilated2d()})
    {
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)));
        EXPECT_TRUE(matchesGraph(pack, fixture.context()).has_value());
    }
    auto last = dilated2d();
    last.channelsLast = true;
    auto threeD = depthwise(5, {4, 5, 6}, {3, 3, 3}, {1, 1, 1});
    for(const auto& shape : {last, threeD})
    {
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)));
        EXPECT_FALSE(matchesGraph(pack, fixture.context()).has_value());
    }
}

TEST(TestConvDepthwiseRowsGraph, BackwardDataServesBothLayoutsButNotThreeD)
{
    const auto pack = packFor(ConvDirection::BACKWARD_DATA);
    for(const bool last : {false, true})
    {
        const GraphFixture fixture(
            buildGraph(specFor(transposed1d(last), ConvDirection::BACKWARD_DATA)));
        EXPECT_TRUE(matchesGraph(pack, fixture.context()).has_value()) << last;
    }
    const GraphFixture threeD(buildGraph(
        specFor(depthwise(5, {4, 5, 6}, {3, 3, 3}, {1, 1, 1}), ConvDirection::BACKWARD_DATA)));
    EXPECT_FALSE(matchesGraph(pack, threeD.context()).has_value());
}

TEST(TestConvDepthwiseRowsGraph, DeclinesWhatIsNotDepthwise)
{
    auto dense = dilated2d();
    dense.groups = 1;
    const GraphFixture fixture(buildGraph(specFor(dense, ConvDirection::FORWARD)));
    EXPECT_FALSE(matchesGraph(packFor(ConvDirection::FORWARD), fixture.context()).has_value());
}

// ---- Every variant serves its shapes ----

TEST(TestConvDepthwiseRowsKernelMatch, EachVariantAdmitsItsShape)
{
    EXPECT_TRUE(admits(tileFields(), dilated1d()));
    EXPECT_TRUE(admits(tile2dFields(), dilated2d()));
    EXPECT_TRUE(admits(bandFields(), dilated2d()));
    EXPECT_TRUE(admits(phaseFieldsIn("NCHW"), transposed1d(false)));
    EXPECT_TRUE(admits(phaseFieldsIn("NHWC"), transposed1d(true)));
    EXPECT_TRUE(admits(lanesFields(), transposed1d(true)));
    // Unit-stride backward data runs the forward form (dy into dx, the filter reflected).
    auto backward = tile2dFields();
    backward.direction = "BWD";
    EXPECT_TRUE(admits(backward, dilated2d()));
    auto band = bandFields();
    band.direction = "BWD";
    EXPECT_TRUE(admits(band, dilated2d()));
    // Runtime filter extents serve any filter.
    auto any = tile2dFields();
    any.config.kh = any.config.kw = 0;
    EXPECT_TRUE(admits(any, depthwise(3, {11, 12}, {5, 4}, {2, 1})));
    auto anyPhase = phaseFields();
    anyPhase.config.kw = 0;
    anyPhase.config.mmax = 32;
    auto k7 = transposed1d(false);
    k7.filter = {7};
    EXPECT_TRUE(admits(anyPhase, k7));
    // Both convolution modes.
    auto convolution = transposed1d(false);
    convolution.mode = ConvMode::CONVOLUTION;
    EXPECT_TRUE(admits(phaseFields(), convolution));
}

// ---- One refusal per config field ----

struct FieldCase
{
    std::string name;
    std::function<Fields()> fields;
    std::function<ConvShape()> shape;
};

std::vector<FieldCase> fieldCases()
{
    const auto with = [](Fields (*base)(), const std::function<void(Fields&)>& change) {
        return [base, change] {
            auto f = base();
            change(f);
            return f;
        };
    };
    std::vector<FieldCase> cases;
    // dtype and direction
    cases.push_back({"Dtype", with(&tileFields, [](Fields& f) { f.dtype = "HALF"; }), dilated1d});
    cases.push_back(
        {"DtypeUnknown", with(&tileFields, [](Fields& f) { f.dtype = "INT8"; }), dilated1d});
    cases.push_back(
        {"DirectionUnknown", with(&tileFields, [](Fields& f) { f.direction = "WRW"; }), dilated1d});
    // A forward-form config on strided backward data, and a phase config on forward.
    cases.push_back({"DirectionTileOnStridedBackward",
                     with(&tileFields,
                          [](Fields& f) {
                              f.direction = "BWD";
                              f.config.kw = 12;
                          }),
                     [] { return transposed1d(false); }});
    cases.push_back({"DirectionPhaseOnForward",
                     with(&phaseFields, [](Fields& f) { f.direction = "FWD"; }),
                     [] { return transposed1d(false); }});
    // layout
    cases.push_back({"LayoutTileChannelsLast",
                     with(&tileFields, [](Fields& f) { f.config.layout = "NHWC"; }),
                     dilated1d});
    cases.push_back({"LayoutPhaseMismatch",
                     with(&phaseFields, [](Fields& f) { f.config.layout = "NHWC"; }),
                     [] { return transposed1d(false); }});
    cases.push_back({"LayoutPhaseUnknown",
                     with(&phaseFields, [](Fields& f) { f.config.layout = "NCDHW"; }),
                     [] { return transposed1d(false); }});
    cases.push_back({"LayoutLanesChannelsFirst",
                     with(&lanesFields, [](Fields& f) { f.config.layout = "NCHW"; }),
                     [] { return transposed1d(true); }});
    // variant
    cases.push_back(
        {"VariantUnknown", with(&tileFields, [](Fields& f) { f.config.variant = 4; }), dilated1d});
    cases.push_back({"VariantNegative",
                     with(&tileFields, [](Fields& f) { f.config.variant = -1; }),
                     dilated1d});
    // kh, kw
    cases.push_back({"Kh", with(&tile2dFields, [](Fields& f) { f.config.kh = 5; }), dilated2d});
    cases.push_back({"Kw", with(&tile2dFields, [](Fields& f) { f.config.kw = 5; }), dilated2d});
    cases.push_back({"KhOnPhase", with(&phaseFields, [](Fields& f) { f.config.kh = 1; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"KwOnPhase", with(&phaseFields, [](Fields& f) { f.config.kw = 13; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"KwOnLanes", with(&lanesFields, [](Fields& f) { f.config.kw = 11; }), [] {
                         return transposed1d(true);
                     }});
    // tx, ty, rw, rh, threads (tile)
    cases.push_back({"Tx", with(&tile2dFields, [](Fields& f) { f.config.tx = 0; }), dilated2d});
    cases.push_back({"Ty", with(&tile2dFields, [](Fields& f) { f.config.ty = 0; }), dilated2d});
    cases.push_back({"Rw", with(&tile2dFields, [](Fields& f) { f.config.rw = 0; }), dilated2d});
    cases.push_back({"Rh", with(&tile2dFields, [](Fields& f) { f.config.rh = 0; }), dilated2d});
    cases.push_back({"ThreadsNotTxTimesTy",
                     with(&tile2dFields, [](Fields& f) { f.config.threads = 128; }),
                     dilated2d});
    cases.push_back({"ThreadsOverTheBlockLimit",
                     with(&tile2dFields,
                          [](Fields& f) {
                              f.config.tx = 64;
                              f.config.ty = 32;
                              f.config.threads = 2048;
                          }),
                     dilated2d});
    // fields a variant does not use
    cases.push_back(
        {"BhOnTile", with(&tile2dFields, [](Fields& f) { f.config.bh = 8; }), dilated2d});
    cases.push_back(
        {"StrideOnTile", with(&tile2dFields, [](Fields& f) { f.config.stride = 1; }), dilated2d});
    cases.push_back({"QOnTile", with(&tile2dFields, [](Fields& f) { f.config.q = 1; }), dilated2d});
    cases.push_back(
        {"MmaxOnTile", with(&tile2dFields, [](Fields& f) { f.config.mmax = 1; }), dilated2d});
    cases.push_back({"TxOnBand", with(&bandFields, [](Fields& f) { f.config.tx = 1; }), dilated2d});
    cases.push_back({"TyOnBand", with(&bandFields, [](Fields& f) { f.config.ty = 1; }), dilated2d});
    cases.push_back({"RwOnBand", with(&bandFields, [](Fields& f) { f.config.rw = 1; }), dilated2d});
    cases.push_back(
        {"StrideOnBand", with(&bandFields, [](Fields& f) { f.config.stride = 1; }), dilated2d});
    cases.push_back({"QOnBand", with(&bandFields, [](Fields& f) { f.config.q = 1; }), dilated2d});
    cases.push_back(
        {"MmaxOnBand", with(&bandFields, [](Fields& f) { f.config.mmax = 1; }), dilated2d});
    cases.push_back({"TxOnPhase", with(&phaseFields, [](Fields& f) { f.config.tx = 1; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"TyOnPhase", with(&phaseFields, [](Fields& f) { f.config.ty = 1; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"RwOnPhase", with(&phaseFields, [](Fields& f) { f.config.rw = 1; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"RhOnPhase", with(&phaseFields, [](Fields& f) { f.config.rh = 1; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"BhOnPhase", with(&phaseFields, [](Fields& f) { f.config.bh = 1; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"LdsOnPhase", with(&phaseFields, [](Fields& f) { f.config.lds = 1; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"QOnLanes", with(&lanesFields, [](Fields& f) { f.config.q = 1; }), [] {
                         return transposed1d(true);
                     }});
    cases.push_back({"LdsOnLanes", with(&lanesFields, [](Fields& f) { f.config.lds = 1; }), [] {
                         return transposed1d(true);
                     }});
    cases.push_back({"MmaxOnLanes", with(&lanesFields, [](Fields& f) { f.config.mmax = 6; }), [] {
                         return transposed1d(true);
                     }});
    cases.push_back({"RhOnLanes", with(&lanesFields, [](Fields& f) { f.config.rh = 1; }), [] {
                         return transposed1d(true);
                     }});
    // bh, rh (band)
    cases.push_back(
        {"BhNotAMultipleOfRh", with(&bandFields, [](Fields& f) { f.config.bh = 18; }), dilated2d});
    cases.push_back({"BhZero", with(&bandFields, [](Fields& f) { f.config.bh = 0; }), dilated2d});
    cases.push_back(
        {"BandThreads", with(&bandFields, [](Fields& f) { f.config.threads = 0; }), dilated2d});
    // stride
    cases.push_back({"StrideOnPhase",
                     with(&phaseFields, [](Fields& f) { f.config.stride = 3; }),
                     [] { return transposed1d(false); }});
    cases.push_back({"StrideOnLanes",
                     with(&lanesFields, [](Fields& f) { f.config.stride = 3; }),
                     [] { return transposed1d(true); }});
    // q, threads (phase)
    cases.push_back({"QZero", with(&phaseFields, [](Fields& f) { f.config.q = 0; }), [] {
                         return transposed1d(false);
                     }});
    cases.push_back({"PhaseThreads",
                     with(&phaseFields, [](Fields& f) { f.config.threads = 0; }),
                     [] { return transposed1d(false); }});
    cases.push_back({"LanesThreads",
                     with(&lanesFields, [](Fields& f) { f.config.threads = 2048; }),
                     [] { return transposed1d(true); }});
    // lds: the halo must fit; a dilation of 128 needs a 1024 + 256 element halo.
    cases.push_back({"LdsHalo", with(&tileFields, [](Fields& f) { f.config.lds = 1100; }), [] {
                         auto shape = dilated1d();
                         shape.dilation = {128};
                         shape.prePadding = shape.postPadding = {128};
                         return shape;
                     }});
    cases.push_back({"LdsOverTheBlockLimit",
                     with(&tileFields, [](Fields& f) { f.config.lds = 16385; }),
                     dilated1d});
    cases.push_back(
        {"LdsBandHalo", with(&bandFields, [](Fields& f) { f.config.lds = 300; }), dilated2d});
    // mmax: the taps per phase must fit the window, and a compile-time filter fills it exactly.
    cases.push_back({"MmaxBelowTheTaps",
                     with(&phaseFields,
                          [](Fields& f) {
                              f.config.kw = 0;
                              f.config.mmax = 5;
                          }),
                     [] { return transposed1d(false); }});
    cases.push_back({"MmaxNotTheFilters",
                     with(&phaseFields, [](Fields& f) { f.config.mmax = 7; }),
                     [] { return transposed1d(false); }});
    cases.push_back({"MmaxZero",
                     with(&phaseFields,
                          [](Fields& f) {
                              f.config.kw = 0;
                              f.config.mmax = 0;
                          }),
                     [] { return transposed1d(false); }});
    return cases;
}

class TestConvDepthwiseRowsRefuses : public ::testing::TestWithParam<FieldCase>
{
};

TEST_P(TestConvDepthwiseRowsRefuses, OneField)
{
    EXPECT_FALSE(admits(GetParam().fields(), GetParam().shape()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvDepthwiseRowsRefuses,
                         ::testing::ValuesIn(fieldCases()),
                         [](const auto& info) { return info.param.name; });

// ---- One refusal per problem gate of the phases ----

TEST(TestConvDepthwiseRowsKernelMatch, PhasesNeedARowSeparableUndilatedWidth)
{
    auto tall = transposed1d(false);
    tall.input = {4, 515};
    tall.filter = {1, 12};
    tall.prePadding = tall.postPadding = {0, 5};
    tall.stride = {1, 2};
    tall.dilation = {1, 1};
    EXPECT_TRUE(admits(phaseFields(), tall)) << "rows of a height-1 filter are independent";

    auto filterHeight = tall;
    filterHeight.filter = {3, 12};
    filterHeight.prePadding = filterHeight.postPadding = {1, 5};
    EXPECT_FALSE(admits(phaseFields(), filterHeight));

    auto strideHeight = tall;
    strideHeight.stride = {2, 2};
    EXPECT_FALSE(admits(phaseFields(), strideHeight));

    auto padHeight = tall;
    padHeight.prePadding = {1, 5};
    padHeight.postPadding = {1, 5};
    EXPECT_FALSE(admits(phaseFields(), padHeight));

    auto postPadHeight = tall;
    postPadHeight.postPadding = {1, 5};
    EXPECT_FALSE(admits(phaseFields(), postPadHeight));

    auto dilatedWidth = transposed1d(false);
    dilatedWidth.dilation = {2};
    dilatedWidth.prePadding = dilatedWidth.postPadding = {11};
    EXPECT_FALSE(admits(phaseFields(), dilatedWidth));
    auto lanes = lanesFields();
    auto dilatedLanes = dilatedWidth;
    dilatedLanes.channelsLast = true;
    EXPECT_FALSE(admits(lanes, dilatedLanes));
}

TEST(TestConvDepthwiseRowsKernelMatch, ForwardFormNeedsUnitStrideForBackwardData)
{
    // Forward runs any stride; backward data only at unit stride (strided runs the phases).
    auto strided1d = dilated1d();
    strided1d.stride = {2};
    auto tile = tileFields();
    EXPECT_TRUE(admits(tile, strided1d));
    tile.direction = "BWD";
    EXPECT_FALSE(admits(tile, strided1d));

    auto strided2d = dilated2d();
    strided2d.stride = {2, 2};
    auto band = bandFields();
    EXPECT_TRUE(admits(band, strided2d));
    band.direction = "BWD";
    EXPECT_FALSE(admits(band, strided2d));
}

TEST(TestConvDepthwiseRowsKernelMatch, RefusesMissingOrMistypedMetadata)
{
    const auto pack = packFor(ConvDirection::FORWARD);
    const GraphFixture fixture(buildGraph(specFor(dilated1d(), ConvDirection::FORWARD)));
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    auto missing = kernelOf(tileFields());
    missing.metadata.erase("lds");
    EXPECT_FALSE(matchesKernel(pack, fixture.context(), missing, *bound));
    auto mistyped = kernelOf(tileFields());
    mistyped.metadata["tx"] = std::string("256");
    EXPECT_FALSE(matchesKernel(pack, fixture.context(), mistyped, *bound));
}

// ---- Launch sizes ----

std::optional<conv_depthwise_rows::RowsLaunch> launchOf(const Fields& fields,
                                                        const ConvShape& shape)
{
    const auto direction
        = fields.direction == "BWD" ? ConvDirection::BACKWARD_DATA : ConvDirection::FORWARD;
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    const auto problem
        = conv_depthwise_valu::parseDepthwiseValuProblem(fixture.context(), direction);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    return rowsLaunchFor(*problem, direction, fields.config);
}

TEST(TestConvDepthwiseRowsLaunch, TileCoversEveryPlaneAndTile)
{
    const auto launch = launchOf(tile2dFields(), dilated2d());
    ASSERT_TRUE(launch.has_value());
    EXPECT_EQ(launch->block, 256);
    // n * c * ceil(17 / 32) * ceil(19 / 64) = 2 * 5.
    EXPECT_EQ(launch->grid, 10);
    EXPECT_FALSE(launch->flip);
}

TEST(TestConvDepthwiseRowsLaunch, BandLaunchesWholeWavesOfUnits)
{
    auto shape = depthwise(3, {65, 65}, {3, 3}, {2, 2});
    shape.dilation = {2, 2};
    const auto launch = launchOf(bandFields(), shape);
    ASSERT_TRUE(launch.has_value());
    // 16 / 4 = 4 row groups x 65 columns = 260 units -> 320 threads (whole 64-lane groups).
    EXPECT_EQ(launch->block, 320);
    // n * c * ceil(65 / 16).
    EXPECT_EQ(launch->grid, 2 * 3 * 5);
    // The cap: 8 groups x 65 = 520 units, capped at 512.
    auto tall = bandFields();
    tall.config.bh = 32;
    const auto capped = launchOf(tall, shape);
    ASSERT_TRUE(capped.has_value());
    EXPECT_EQ(capped->block, 512);
}

TEST(TestConvDepthwiseRowsLaunch, PhasesCoverEveryDxElement)
{
    const auto launch = launchOf(phaseFields(), transposed1d(false));
    ASSERT_TRUE(launch.has_value());
    // dx 515 wide, pad 5, stride 2: q from 2 to (514 + 5) / 2 = 259, 258 values; 512 per block.
    EXPECT_EQ(launch->grid, 2 * 6 * 1);
    EXPECT_EQ(launch->args.wo, 515);
    EXPECT_EQ(launch->args.pw, 5);
    const auto lanes = launchOf(lanesFields(), transposed1d(true));
    ASSERT_TRUE(lanes.has_value());
    // n * rows * q values * channels / 256 = 2 * 1 * 258 * 6 / 256, rounded up.
    EXPECT_EQ(lanes->grid, (2 * 258 * 6 + 255) / 256);
}

TEST(TestConvDepthwiseRowsLaunch, ReflectsTheFilterPerForm)
{
    // Unit-stride backward data reflects the filter (cross-correlation); the phases read it as
    // the forward convolution does, reflected only for true convolution.
    auto backward = tile2dFields();
    backward.direction = "BWD";
    const auto tile = launchOf(backward, dilated2d());
    ASSERT_TRUE(tile.has_value());
    EXPECT_TRUE(tile->flip);
    const auto phase = launchOf(phaseFields(), transposed1d(false));
    ASSERT_TRUE(phase.has_value());
    EXPECT_FALSE(phase->flip);
    auto convolution = transposed1d(false);
    convolution.mode = ConvMode::CONVOLUTION;
    const auto reflected = launchOf(phaseFields(), convolution);
    ASSERT_TRUE(reflected.has_value());
    EXPECT_TRUE(reflected->flip);
}

TEST(TestConvDepthwiseRowsComputeType, ServesCudnnsComputeTypesPerElementType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto pack = packFor(ConvDirection::FORWARD);
    expectCudnnComputeTypes(dilated2d(),
                            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
                            [&pack](const ConvShape& s) {
                                const GraphFixture fixture(
                                    buildGraph(specFor(s, ConvDirection::FORWARD)));
                                return matchesGraph(pack, fixture.context()).has_value();
                            });
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
