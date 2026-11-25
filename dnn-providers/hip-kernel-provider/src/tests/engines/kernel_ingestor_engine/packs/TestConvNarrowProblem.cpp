// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNarrowNative.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestConvNarrowProblem.cpp
 * @brief Host-only checks of CONV_NARROW_FWD / CONV_NARROW_BWD: what each graph match accepts
 *        and refuses (one decline per gated field), what the kernel match admits per config
 *        (direction, dtype, layout, atom on the device and its element, ship list, narrow
 *        channels, filter, width stride, tile), one decline per malformed config field, and the
 *        plan's strips, bands, prepared filter, vector loads and image limits. No device.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
namespace cn = conv_narrow;

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD ? "hipkernel.conv_narrow_fwd.graph_match"
                                                : "hipkernel.conv_narrow_bwd.graph_match",
            "",
            "hipkernel.conv_narrow.kernel_match",
            "",
            "hipkernel.conv_narrow.dispatch",
            "",
            "",
            ""};
}

/// A VAE output layer: 128 channels into 3, 3x3, padding 1, channels-last bf16.
ConvShape head()
{
    ConvShape shape;
    shape.n = 2;
    shape.c = 128;
    shape.k = 3;
    shape.input = {64, 96};
    shape.filter = {3, 3};
    shape.prePadding = shape.postPadding = {1, 1};
    shape.stride = {1, 1};
    shape.dilation = {1, 1};
    shape.channelsLast = true;
    shape.dataType = DataType::BFLOAT16;
    return shape;
}

/// A ResNet stem: 3 channels into 64, 7x7 stride 2, padding 3, channels-first fp16.
ConvShape stem()
{
    ConvShape shape;
    shape.n = 4;
    shape.c = 3;
    shape.k = 64;
    shape.input = {224, 224};
    shape.filter = {7, 7};
    shape.prePadding = shape.postPadding = {3, 3};
    shape.stride = {2, 2};
    shape.dilation = {1, 1};
    shape.dataType = DataType::HALF;
    return shape;
}

struct Fields
{
    std::string direction = "FWD";
    std::string dtype = "BFLOAT16";
    std::string layout = "nhwc";
    std::string atom = "mfma16x32_bf16";
    std::string archs = "gfx950";
    int64_t atomId = 6, kmin = 2, ko = 3, fd = 1, fr = 3, fs = 3, sw = 1, ds = 1, cmin = 65,
            cm = 128, pd = 1, sf = 0, wr = 32, rb = 2, nw = 4, op = 2;
};

Fields stemFields()
{
    Fields f;
    f.direction = "BWD";
    f.dtype = "HALF";
    f.layout = "nchw";
    f.atom = "mfma16x32_f16";
    f.atomId = 5;
    f.fr = f.fs = 7;
    f.sw = 2;
    f.cmin = 33;
    f.cm = 64;
    return f;
}

hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const Fields& f)
{
    auto kernel = makeKernel(0, f.dtype, "ConvNarrow");
    kernel.source.sourceFile = "kernels/ConvNarrow.cpp";
    kernel.metadata = {{"direction", f.direction},
                       {"dtype", f.dtype},
                       {"layout", f.layout},
                       {"atom", f.atom},
                       {"archs", f.archs},
                       {"atom_id", f.atomId},
                       {"kmin", f.kmin},
                       {"ko", f.ko},
                       {"fd", f.fd},
                       {"fr", f.fr},
                       {"fs", f.fs},
                       {"sw", f.sw},
                       {"ds", f.ds},
                       {"cmin", f.cmin},
                       {"cm", f.cm},
                       {"pd", f.pd},
                       {"sf", f.sf},
                       {"wr", f.wr},
                       {"rb", f.rb},
                       {"nw", f.nw},
                       {"op", f.op}};
    return kernel;
}

bool graphAccepts(const ConvShape& shape,
                  ConvDirection direction,
                  DataType computeType = DataType::FLOAT)
{
    auto spec = specFor(shape, direction);
    spec.computeType = computeType;
    const GraphFixture fixture(buildGraph(spec));
    return matchesGraph(packFor(direction), fixture.context()).has_value();
}

std::optional<cn::Admitted> admit(const Fields& fields,
                                  const ConvShape& shape,
                                  ConvDirection direction,
                                  const std::string& arch = "gfx950")
{
    const GraphFixture fixture(buildGraph(specFor(shape, direction)), gfx950Properties(arch));
    if(!matchesGraph(packFor(direction), fixture.context()).has_value())
    {
        return std::nullopt;
    }
    return cn::admitted(fixture.context(), kernelOf(fields));
}

// ---- graph match

TEST(TestConvNarrowProblem, AcceptsOutputLayersAndStemGradients)
{
    EXPECT_TRUE(graphAccepts(head(), ConvDirection::FORWARD));
    EXPECT_TRUE(graphAccepts(stem(), ConvDirection::BACKWARD_DATA));
    auto threeD = head();
    threeD.input = {6, 20, 30};
    threeD.filter = {3, 3, 3};
    threeD.prePadding = threeD.postPadding = {0, 0, 0};
    threeD.stride = threeD.dilation = {1, 1, 1};
    EXPECT_TRUE(graphAccepts(threeD, ConvDirection::FORWARD));
    auto oneD = head();
    oneD.input = {300};
    oneD.filter = {7};
    oneD.prePadding = oneD.postPadding = {3};
    oneD.stride = oneD.dilation = {1};
    oneD.k = 1;
    EXPECT_TRUE(graphAccepts(oneD, ConvDirection::FORWARD));
}

TEST(TestConvNarrowProblem, DeclinesEachGatedField)
{
    auto grouped = head();
    grouped.groups = 2;
    grouped.k = 4;
    EXPECT_FALSE(graphAccepts(grouped, ConvDirection::FORWARD));
    auto wide = head();
    wide.k = 9; // past the narrow bound
    EXPECT_FALSE(graphAccepts(wide, ConvDirection::FORWARD));
    auto square = head();
    square.c = 3; // nothing to reduce from
    EXPECT_FALSE(graphAccepts(square, ConvDirection::FORWARD));
    // Forward narrows K, backward data narrows C: a stem forward and a head backward are not
    // narrow.
    EXPECT_FALSE(graphAccepts(stem(), ConvDirection::FORWARD));
    EXPECT_FALSE(graphAccepts(head(), ConvDirection::BACKWARD_DATA));
    EXPECT_FALSE(graphAccepts(head(), ConvDirection::FORWARD, DataType::HALF));
}

// ---- kernel match

TEST(TestConvNarrowProblem, AdmitsTheMatchingConfig)
{
    EXPECT_TRUE(admit(Fields{}, head(), ConvDirection::FORWARD).has_value());
    EXPECT_TRUE(admit(stemFields(), stem(), ConvDirection::BACKWARD_DATA).has_value());
    auto valu = Fields{};
    valu.atom = "valu";
    valu.atomId = 0;
    valu.archs = "any";
    valu.cmin = 0;
    valu.cm = 0;
    valu.pd = 0;
    EXPECT_TRUE(admit(valu, head(), ConvDirection::FORWARD, "gfx906").has_value());
    EXPECT_TRUE(admit(valu, head(), ConvDirection::FORWARD, "gfx1201").has_value());
}

TEST(TestConvNarrowProblem, DeclinesEachConfigMismatch)
{
    const auto fwd = ConvDirection::FORWARD;
    auto f = Fields{};
    f.direction = "BWD";
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.dtype = "HALF";
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.layout = "nchw";
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.atom = "mfma16x32_f16"; // the graph's element type is bf16
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    EXPECT_FALSE(admit(f, head(), fwd, "gfx942")); // the gfx950 atom does not run there
    f = Fields{};
    f.archs = "gfx942";
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.kmin = 1;
    f.ko = 1; // fewer than the graph's 3 output channels
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.kmin = 4;
    f.ko = 8; // a config for 4 to 8 channels
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.cmin = 33;
    f.cm = 64; // fewer than the graph's 128 reduction channels
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.cmin = 129;
    f.cm = 256; // a config for 129 to 256 channels
    EXPECT_FALSE(admit(f, head(), fwd));
    auto streamed = head();
    streamed.input = {6, 20, 30};
    streamed.filter = {3, 3, 3};
    streamed.prePadding = streamed.postPadding = {1, 1, 1};
    streamed.stride = {2, 1, 1}; // a depth stride: no depth streaming
    streamed.dilation = {1, 1, 1};
    f = Fields{};
    f.fd = 3;
    EXPECT_FALSE(admit(f, streamed, fwd));
    f.ds = 0;
    f.cmin = 33;
    f.cm = 64;
    f.kmin = 2;
    EXPECT_FALSE(admit(f, streamed, fwd)); // 128 channels past cm 64
    f.cmin = 65;
    f.cm = 128;
    EXPECT_TRUE(admit(f, streamed, fwd).has_value());
    f = Fields{};
    f.fd = 3;
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.fr = 5;
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.fs = 5;
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.sw = 2;
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.wr = 24; // 48 product rows: not whole row tiles of four waves
    EXPECT_FALSE(admit(f, head(), fwd));
    f = Fields{};
    f.ko = 8;
    f.wr = 128;
    f.rb = 2; // 256 rows of 80 columns: past 64 KiB of LDS
    EXPECT_FALSE(admit(f, head(), fwd));
}

TEST(TestConvNarrowProblem, DeclinesEachMalformedField)
{
    EXPECT_TRUE(cn::configOf(kernelOf(Fields{})).has_value());
    const auto declines = [](auto mutate) {
        auto f = Fields{};
        mutate(f);
        return !cn::configOf(kernelOf(f)).has_value();
    };
    EXPECT_TRUE(declines([](Fields& f) { f.direction = "WRW"; }));
    EXPECT_TRUE(declines([](Fields& f) { f.layout = "ncdhw"; }));
    EXPECT_TRUE(declines([](Fields& f) { f.atom = "fma"; }));
    EXPECT_TRUE(declines([](Fields& f) { f.kmin = 0; }));
    EXPECT_TRUE(declines([](Fields& f) { f.kmin = 4; })); // past ko
    EXPECT_TRUE(declines([](Fields& f) { f.ko = 0; }));
    EXPECT_TRUE(declines([](Fields& f) { f.ko = 9; }));
    EXPECT_TRUE(declines([](Fields& f) { f.fd = 8; }));
    EXPECT_TRUE(declines([](Fields& f) { f.fr = 0; }));
    EXPECT_TRUE(declines([](Fields& f) { f.fs = 8; }));
    EXPECT_TRUE(declines([](Fields& f) { f.sw = 5; }));
    EXPECT_TRUE(declines([](Fields& f) { f.wr = 0; }));
    EXPECT_TRUE(declines([](Fields& f) { f.rb = 17; }));
    EXPECT_TRUE(declines([](Fields& f) { f.nw = 3; }));
    EXPECT_TRUE(declines([](Fields& f) { f.op = 9; }));
    EXPECT_TRUE(declines([](Fields& f) { f.ds = 2; }));
    EXPECT_TRUE(declines([](Fields& f) {
        f.cmin = 1;
        f.cm = 48; // not whole 32-channel k-steps
    }));
    EXPECT_TRUE(declines([](Fields& f) { f.cmin = 0; }));
    EXPECT_TRUE(declines([](Fields& f) { f.cmin = 129; })); // past cm
    EXPECT_TRUE(declines([](Fields& f) {
        f.atom = "valu";
        f.cm = 64; // valu holds no filter in LDS
    }));
    EXPECT_TRUE(declines([](Fields& f) { f.pd = 0; }));
    EXPECT_TRUE(declines([](Fields& f) { f.pd = 4; }));
    EXPECT_TRUE(declines([](Fields& f) { f.atomId = -1; }));
    auto missing = kernelOf(Fields{});
    missing.metadata.erase("wr");
    EXPECT_FALSE(cn::configOf(missing).has_value());
}

// ---- plan

TEST(TestConvNarrowProblem, PlansStripsBandsAndThePreparedFilter)
{
    const auto a = admit(Fields{}, head(), ConvDirection::FORWARD);
    ASSERT_TRUE(a.has_value());
    const auto& p = a->plan;
    // 32-pixel product rows hold a 3-tap strip of 30 output columns.
    EXPECT_EQ(p.args.tw, 30);
    EXPECT_EQ(p.args.strips, 4); // 96 / 30
    EXPECT_EQ(p.args.rs * p.args.bands >= 64, true);
    EXPECT_LE(p.args.rs * p.args.tw, p.threads * 2);
    EXPECT_EQ(p.args.blocks, 2 * 1 * 4 * p.args.bands);
    EXPECT_EQ(p.threads, 256);
    EXPECT_EQ(p.ncp, 32); // 27 columns in two 16-wide tiles
    EXPECT_EQ(p.args.cpad, 128); // whole 32-channel steps
    EXPECT_EQ(p.args.prepared, 1 * 2 * 16 * 128);
    EXPECT_EQ(p.workspaceBytes, static_cast<size_t>(p.args.prepared) * 2);
    EXPECT_TRUE(p.vectorIn);
    // Two product buffers and the zero run; the B fragments in registers (2 tiles x 4 k-steps).
    EXPECT_TRUE(p.bRegisters);
    EXPECT_EQ(p.ldsBytes, (2 * 64 * 33 + 27) * 4);

    // Backward data at stride 2: a 7-tap strip of dx columns whose dy columns fit 32.
    const auto b = admit(stemFields(), stem(), ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(b.has_value());
    // A strip of dx columns [i0, i0 + tw) reads dy columns i0 / 2 + ceil((3 - 6) / 2) through
    // i0 / 2 + floor((tw - 1 + 3) / 2): 58 columns read 32.
    EXPECT_EQ(b->plan.args.tw, 58);
    EXPECT_FALSE(b->plan.vectorIn); // channels-first
    EXPECT_EQ(b->plan.ncp, 160); // 3 x 49 columns
}

TEST(TestConvNarrowProblem, VectorLoadsNeedWholeChannelRuns)
{
    auto odd = head();
    odd.c = 100; // not whole 8-channel runs of the 16x16x32 atom
    const auto a = admit(Fields{}, odd, ConvDirection::FORWARD);
    ASSERT_TRUE(a.has_value());
    EXPECT_FALSE(a->plan.vectorIn);
    EXPECT_EQ(a->plan.args.cpad, 128);
}

TEST(TestConvNarrowProblem, ImagesPastTwoGibibytesDecline)
{
    // One image of 2^31 bytes declines; many images of less are served (64-bit image bases).
    auto big = head();
    big.n = 1;
    big.c = 256;
    big.input = {2048, 2048};
    EXPECT_FALSE(admit(Fields{}, big, ConvDirection::FORWARD).has_value());
    big.input = {1024, 2048};
    big.n = 8;
    auto f = Fields{};
    f.cm = 256; // the 256 channels in the LDS filter
    EXPECT_TRUE(admit(f, big, ConvDirection::FORWARD).has_value());
}

// ---- backward weights

PackSymbols wrwPack()
{
    return {"",
            "hipkernel.conv_narrow_wrw.graph_match",
            "",
            "hipkernel.conv_narrow_wrw.kernel_match",
            "",
            "hipkernel.conv_narrow_wrw.dispatch",
            "",
            "",
            ""};
}

struct WrwFields
{
    std::string side = "k";
    std::string dtype = "BFLOAT16";
    std::string layout = "nhwc";
    std::string atom = "mfma16x32_bf16";
    std::string archs = "gfx950";
    int64_t atomId = 6, sideId = 1, kmin = 2, ko = 3, fd = 1, fr = 3, fs = 3, wr = 32, rb = 2,
            nw = 4, mpw = 1;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition wrwKernelOf(const WrwFields& f)
{
    auto kernel = makeKernel(0, f.dtype, "ConvNarrowWrw");
    kernel.source.sourceFile = "kernels/ConvNarrowWrw.cpp";
    kernel.metadata = {{"side", f.side},
                       {"dtype", f.dtype},
                       {"layout", f.layout},
                       {"atom", f.atom},
                       {"archs", f.archs},
                       {"atom_id", f.atomId},
                       {"side_id", f.sideId},
                       {"kmin", f.kmin},
                       {"ko", f.ko},
                       {"fd", f.fd},
                       {"fr", f.fr},
                       {"fs", f.fs},
                       {"wr", f.wr},
                       {"rb", f.rb},
                       {"nw", f.nw},
                       {"mpw", f.mpw}};
    return kernel;
}

std::optional<cn::WrwAdmitted>
    admitWrw(const WrwFields& fields, const ConvShape& shape, const std::string& arch = "gfx950")
{
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                               gfx950Properties(arch));
    if(!matchesGraph(wrwPack(), fixture.context()).has_value())
    {
        return std::nullopt;
    }
    return cn::admittedWrw(fixture.context(), wrwKernelOf(fields));
}

bool wrwGraphAccepts(const ConvShape& shape)
{
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)));
    return matchesGraph(wrwPack(), fixture.context()).has_value();
}

TEST(TestConvNarrowProblem, WrwAcceptsBothSidesAndDeclinesTheRest)
{
    EXPECT_TRUE(wrwGraphAccepts(head()));
    EXPECT_TRUE(wrwGraphAccepts(stem()));
    auto grouped = head();
    grouped.groups = 2;
    grouped.k = 4;
    EXPECT_FALSE(wrwGraphAccepts(grouped));
    auto wide = head();
    wide.k = 16;
    EXPECT_FALSE(wrwGraphAccepts(wide)); // neither side narrow
    auto square = head();
    square.c = 3;
    EXPECT_FALSE(wrwGraphAccepts(square));
}

TEST(TestConvNarrowProblem, WrwAdmitsTheMatchingConfigAndDeclinesEachMismatch)
{
    EXPECT_TRUE(admitWrw(WrwFields{}, head()).has_value());
    auto stemFields = WrwFields{};
    stemFields.side = "c";
    stemFields.sideId = 0;
    stemFields.dtype = "HALF";
    stemFields.layout = "nchw";
    stemFields.atom = "mfma16x32_f16";
    stemFields.atomId = 5;
    stemFields.fr = stemFields.fs = 7;
    EXPECT_TRUE(admitWrw(stemFields, stem()).has_value());
    const auto declines = [](auto mutate) {
        auto f = WrwFields{};
        mutate(f);
        return !admitWrw(f, head()).has_value();
    };
    EXPECT_TRUE(declines([](WrwFields& f) { f.side = "c"; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.dtype = "HALF"; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.layout = "nchw"; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.atom = "mfma16x32_f16"; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.archs = "gfx942"; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.ko = 1; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.fd = 3; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.fr = 1; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.fs = 1; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.wr = 20; })); // not whole 8-pixel lane runs
    EXPECT_TRUE(declines([](WrwFields& f) { f.rb = 0; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.nw = 3; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.mpw = 0; }));
    EXPECT_TRUE(declines([](WrwFields& f) { f.sideId = -1; }));
    EXPECT_FALSE(admitWrw(WrwFields{}, head(), "gfx942").has_value());
}

TEST(TestConvNarrowProblem, WrwPlansSplitsChannelBlocksAndPartials)
{
    const auto a = admitWrw(WrwFields{}, head());
    ASSERT_TRUE(a.has_value());
    const auto& p = a->plan;
    EXPECT_EQ(p.ncp, 32); // 27 columns
    EXPECT_EQ(p.args.cblocks, 2); // 128 channels in blocks of 64
    EXPECT_EQ(p.args.strips, 3); // 96 columns, 32 a step
    EXPECT_EQ(p.args.bands, 32); // 64 rows, 2 a step
    EXPECT_EQ(p.args.steps, 2 * 1 * 32 * 3);
    EXPECT_GE(p.args.splits, 1);
    EXPECT_LE(p.args.splits, p.args.steps);
    EXPECT_EQ(p.workspaceBytes, static_cast<size_t>(p.args.splits) * 128 * 32 * 4);
    EXPECT_EQ(p.args.dwElems, 3 * 128 * 9);
    EXPECT_EQ(p.args.wideIsC, 1);
    EXPECT_FALSE(p.vectorIn); // channels-last
}

} // namespace

#endif
