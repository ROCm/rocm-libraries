// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvWinogradNdProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include <array>
#include <climits>
#include <gtest/gtest.h>
#include <set>
#include <vector>
namespace
{
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
namespace nd = conv_winograd_nd;
using conv_naive::ConvDirection;
ConvShape ndShape(bool rank3)
{
    ConvShape s;
    s.n = 1;
    s.c = 6;
    s.k = 10;
    s.groups = 2;
    s.input = rank3 ? std::vector<int64_t>{5, 7, 9} : std::vector<int64_t>{17, 19};
    s.filter = rank3 ? std::vector<int64_t>{3, 3, 3} : std::vector<int64_t>{3, 3};
    s.prePadding = s.postPadding
        = rank3 ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{2, 4};
    s.dilation = rank3 ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{2, 4};
    s.stride = rank3 ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{1, 1};
    return s;
}
TEST(TestConvWinogradNdProblem, DilatedResiduesAndSpatialChunksPartitionOutputs)
{
    for(const int chunk : {3, 256, 4096})
    {
        const GraphFixture f(buildGraph(specFor(ndShape(false), ConvDirection::FORWARD)));
        const auto p = nd::parseProblem(f.context(), ConvDirection::FORWARD);
        ASSERT_TRUE(p);
        nd::Config c;
        c.chunkTiles = chunk;
        const auto plan = nd::planFor(*p, c, 16384);
        ASSERT_TRUE(plan);
        EXPECT_LE(plan->args.tiles, chunk);
        EXPECT_LE(plan->workspaceBytes, 16384u);
        std::set<std::array<long long, 3>> writes;
        const auto& a = plan->args;
        for(long long first = 0; first < plan->totalTiles; first += a.tiles)
        {
            for(int t = 0; t < a.tiles && first + t < plan->totalTiles; ++t)
            {
                long long rem = first + t;
                std::array<long long, 2> origin{};
                for(int j = 1; j >= 0; --j)
                {
                    const auto q = rem % a.axisTiles[j];
                    rem /= a.axisTiles[j];
                    origin[static_cast<size_t>(j)]
                        = q % a.residues[j] + 2 * a.dilation[j] * (q / a.residues[j]);
                }
                for(int h = 0; h < 2; ++h)
                {
                    for(int w = 0; w < 2; ++w)
                    {
                        std::array<long long, 3> pos{
                            rem, origin[0] + h * a.dilation[0], origin[1] + w * a.dilation[1]};
                        if(pos[1] < a.output[0] && pos[2] < a.output[1])
                        {
                            EXPECT_TRUE(writes.insert(pos).second);
                        }
                    }
                }
            }
        }
        EXPECT_EQ(writes.size(), a.n * a.output[0] * a.output[1]);
    }
}
TEST(TestConvWinogradNdProblem, RefusesStrideAndRankThree)
{
    auto s = ndShape(false);
    s.stride = {2, 1};
    const GraphFixture stride(buildGraph(specFor(s, ConvDirection::FORWARD)));
    EXPECT_FALSE(nd::parseProblem(stride.context(), ConvDirection::FORWARD));
    const GraphFixture rankThree(buildGraph(specFor(ndShape(true), ConvDirection::FORWARD)));
    EXPECT_FALSE(nd::parseProblem(rankThree.context(), ConvDirection::FORWARD));
}
TEST(TestConvWinogradNdProblem, FixedFilterMustFitWorkspaceAndTileProductsMustNotOverflow)
{
    const GraphFixture f(buildGraph(specFor(ndShape(false), ConvDirection::FORWARD)));
    auto p = nd::parseProblem(f.context(), ConvDirection::FORWARD);
    ASSERT_TRUE(p);
    const nd::Config c;
    EXPECT_FALSE(nd::planFor(*p, c, 512));
    p->args.n = LLONG_MAX;
    EXPECT_FALSE(nd::planFor(*p, c));
}
TEST(TestConvWinogradNdProblem, RefusesBackwardSixteenBitAndUnitDilation)
{
    const auto s = ndShape(false);
    for(auto dir : {ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        const GraphFixture f(buildGraph(specFor(s, dir)));
        EXPECT_FALSE(nd::parseProblem(f.context(), dir));
    }
    for(auto dtype : {hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16})
    {
        auto lowPrecision = s;
        lowPrecision.dataType = dtype;
        const GraphFixture f(buildGraph(specFor(lowPrecision, ConvDirection::FORWARD)));
        EXPECT_FALSE(nd::parseProblem(f.context(), ConvDirection::FORWARD));
    }
    auto unit = s;
    unit.dilation = {1, 1};
    unit.prePadding = unit.postPadding = {1, 1};
    const GraphFixture f(buildGraph(specFor(unit, ConvDirection::FORWARD)));
    EXPECT_FALSE(nd::parseProblem(f.context(), ConvDirection::FORWARD));
}
TEST(TestConvWinogradNdProblem, RefusesUnsupportedNativeArchitectureAndTileShape)
{
    const GraphFixture f(buildGraph(specFor(ndShape(false), ConvDirection::FORWARD)));
    const auto p = nd::parseProblem(f.context(), ConvDirection::FORWARD);
    ASSERT_TRUE(p);
    nd::Config c;
    EXPECT_FALSE(nd::configRunsOn(c, "gfx1201"));
    EXPECT_FALSE(nd::configRunsOn(c, "gfx942"));
    c.arch = "";
    EXPECT_FALSE(nd::planFor(*p, c));
    c.arch = "gfx1250";
    c.atom = 0;
    EXPECT_FALSE(nd::planFor(*p, c));
    c.atom = 5;
    c.bm = 32;
    EXPECT_FALSE(nd::planFor(*p, c));
    c.bm = 64;
    c.bn = 32;
    EXPECT_FALSE(nd::planFor(*p, c));
    c.bn = 64;
    c.bk = 32;
    EXPECT_FALSE(nd::planFor(*p, c));
}
TEST(TestConvWinogradNdProblem, PaddedMixedViewsAreAdmittedButOverlapsAreRefused)
{
    auto spec = specFor(ndShape(false), ConvDirection::FORWARD);
    spec.y.strides = packedStrides(spec.y.dims, true);
    for(auto& stride : spec.x.strides)
    {
        stride *= 2;
    }
    const GraphFixture padded(buildGraph(spec));
    EXPECT_TRUE(nd::parseProblem(padded.context(), ConvDirection::FORWARD));
    spec.x.strides.back() = spec.x.strides[spec.x.strides.size() - 2];
    const GraphFixture overlap(buildGraph(spec));
    EXPECT_FALSE(nd::parseProblem(overlap.context(), ConvDirection::FORWARD));
}
}
#endif
