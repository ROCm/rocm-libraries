// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// --split-devices: how ClientProblemFactory cuts every problem into one slice per device, and how
// the slices and their place in a full-size D are looked up. These tests only build problem
// descriptions, so they need no GPU work.

#include "ClientArgsTestUtils.hpp"
#include "ClientProblemFactory.hpp"

#include <Tensile/ContractionProblem.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    po::variables_map
        makeArgs(std::vector<size_t> size, int splitDevices, std::string const& splitDim)
    {
        return TensileLite::TestUtils::makeSplitClientArgs(std::move(size), splitDevices, splitDim);
    }

    ContractionProblemGemm const& gemm(ContractionProblem const* problem)
    {
        auto const* g = dynamic_cast<ContractionProblemGemm const*>(problem);
        EXPECT_NE(g, nullptr);
        return *g;
    }
}

TEST(ClientProblemFactorySplit, IsOffByDefault)
{
    ClientProblemFactory factory(makeArgs({128, 128, 1, 64}, 0, ""));
    ASSERT_EQ(factory.problems().size(), 1u);
    auto* full = factory.problems()[0].get();

    EXPECT_EQ(factory.numSlices(), 1u);
    EXPECT_EQ(factory.sliceProblem(full, 5), full);
    EXPECT_EQ(factory.sliceOffsetInD(full, 5), 0u);
    EXPECT_EQ(&factory.sliceFactory(2), &factory);
}

TEST(ClientProblemFactorySplit, SingleDeviceIsTheSameAsOff)
{
    ClientProblemFactory factory(makeArgs({128, 128, 1, 64}, 1, "M"));
    EXPECT_EQ(factory.numSlices(), 1u);
}

TEST(ClientProblemFactorySplit, SplitsMAndReplicatesB)
{
    // 130 rows over 3 devices: the first two take 130 / 3 = 43, the last takes the rest.
    ClientProblemFactory factory(makeArgs({130, 96, 1, 64}, 3, "M"));
    auto*                full = factory.problems()[0].get();
    ASSERT_EQ(factory.numSlices(), 3u);

    const size_t expected[3] = {43, 43, 44};
    size_t       total       = 0;
    for(size_t d = 0; d < 3; d++)
    {
        auto const& slice = gemm(factory.sliceProblem(full, d));
        EXPECT_EQ(slice.c().sizes()[0], expected[d]) << "device " << d;
        EXPECT_EQ(slice.c().sizes()[1], 96u);
        EXPECT_EQ(slice.a().sizes()[0], expected[d]); // A is split
        EXPECT_EQ(slice.b().sizes()[1], 96u); // B is replicated whole
        total += slice.c().sizes()[0];
    }
    EXPECT_EQ(total, 130u);
    EXPECT_NE(factory.sliceProblem(full, 0), full);
    EXPECT_EQ(&factory.sliceFactory(1), &factory.sliceFactory(1));
    EXPECT_NE(&factory.sliceFactory(1), &factory);
}

TEST(ClientProblemFactorySplit, SplitsNAndReplicatesA)
{
    ClientProblemFactory factory(makeArgs({96, 130, 1, 64}, 2, "N"));
    auto*                full = factory.problems()[0].get();
    ASSERT_EQ(factory.numSlices(), 2u);

    for(size_t d = 0; d < 2; d++)
    {
        auto const& slice = gemm(factory.sliceProblem(full, d));
        EXPECT_EQ(slice.c().sizes()[0], 96u);
        EXPECT_EQ(slice.c().sizes()[1], 65u) << "device " << d;
        EXPECT_EQ(slice.b().sizes()[1], 65u); // B is split
        EXPECT_EQ(slice.a().sizes()[0], 96u); // A is replicated whole
    }
}

TEST(ClientProblemFactorySplit, DKeepsTheFullStridesAndCShrinks)
{
    // Every slice writes into a full-size D at its final place, so D keeps the strides of the full
    // problem, while the strides that were tight stay tight for the smaller C.
    ClientProblemFactory factory(makeArgs({128, 96, 1, 64}, 2, "M"));
    auto*                full = factory.problems()[0].get();

    auto const& fullD = gemm(full).d();
    for(size_t d = 0; d < 2; d++)
    {
        auto const& slice = gemm(factory.sliceProblem(full, d));
        EXPECT_EQ(slice.d().strides(), fullD.strides());
        EXPECT_EQ(slice.d().sizes()[0], 64u);
        EXPECT_EQ(slice.c().strides()[1], 64u);
    }
}

TEST(ClientProblemFactorySplit, CEqualsDSharesTheStrides)
{
    auto args = makeArgs({128, 96, 1, 64}, 2, "M");
    TensileLite::TestUtils::setOption(args, "c-equal-d", true);
    ClientProblemFactory factory(args);
    auto*                full = factory.problems()[0].get();

    auto const& slice = gemm(factory.sliceProblem(full, 1));
    EXPECT_EQ(slice.c().strides(), slice.d().strides());
}

TEST(ClientProblemFactorySplit, OffsetOfASliceInTheFullD)
{
    {
        // M is the contiguous dimension of D: a slice starts `share` elements after the previous.
        ClientProblemFactory factory(makeArgs({130, 96, 1, 64}, 3, "M"));
        auto*                full = factory.problems()[0].get();
        EXPECT_EQ(factory.sliceOffsetInD(full, 0), 0u);
        EXPECT_EQ(factory.sliceOffsetInD(full, 1), 43u);
        EXPECT_EQ(factory.sliceOffsetInD(full, 2), 86u);
    }
    {
        // N slices start whole columns later: `share` columns of the full leading dimension.
        ClientProblemFactory factory(makeArgs({96, 130, 1, 64}, 2, "N"));
        auto*                full = factory.problems()[0].get();
        EXPECT_EQ(factory.sliceOffsetInD(full, 0), 0u);
        EXPECT_EQ(factory.sliceOffsetInD(full, 1), 65u * 96u);
    }
}

TEST(ClientProblemFactorySplit, RejectsMoreDevicesThanRowsToSplit)
{
    EXPECT_THROW(ClientProblemFactory(makeArgs({3, 96, 1, 64}, 4, "M")), std::runtime_error);
    EXPECT_THROW(ClientProblemFactory(makeArgs({96, 3, 1, 64}, 4, "N")), std::runtime_error);
}

TEST(ClientProblemFactorySplit, OutputPeerDReachesEveryProblemAndSlice)
{
    {
        ClientProblemFactory factory(makeArgs({128, 96, 1, 64}, 2, "M"));
        EXPECT_FALSE(gemm(factory.problems()[0].get()).outputPeerD());
    }

    auto args = makeArgs({128, 96, 1, 64}, 2, "M");
    TensileLite::TestUtils::setOption(args, "output-peerD", true);
    ClientProblemFactory factory(args);
    auto*                full = factory.problems()[0].get();
    EXPECT_TRUE(gemm(full).outputPeerD());
    for(size_t d = 0; d < factory.numSlices(); d++)
        EXPECT_TRUE(gemm(factory.sliceProblem(full, d)).outputPeerD()) << "device " << d;
}

TEST(ClientProblemFactorySplit, RejectsAProblemFromAnotherFactory)
{
    ClientProblemFactory factory(makeArgs({128, 96, 1, 64}, 2, "M"));
    ClientProblemFactory other(makeArgs({128, 96, 1, 64}, 2, "M"));
    EXPECT_THROW(factory.sliceProblem(other.problems()[0].get(), 0), std::runtime_error);
}
