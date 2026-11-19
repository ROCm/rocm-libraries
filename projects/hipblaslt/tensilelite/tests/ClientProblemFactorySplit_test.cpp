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

TEST(ClientProblemFactorySplit, ADefaultBatchStrideOfAPaddedDStaysTheFullOne)
{
    // The client config gives d-strides as "-1,ld,-1": a padded ld, the other strides left at
    // the default. The default batch stride must be that of the full D (256 * 256), not that of
    // the slice (256 * 128 for an N split), or the batches land in the wrong place in a shared D.
    const size_t useDefault = TensorDescriptor::UseDefaultStride;
    for(const char* dim : {"M", "N"})
    {
        auto args = makeArgs({256, 256, 4, 128}, 2, dim);
        TensileLite::TestUtils::setOption(
            args, "d-strides", std::vector<std::vector<size_t>>{{useDefault, 256, useDefault}});
        ClientProblemFactory factory(args);
        auto*                full = factory.problems()[0].get();
        for(size_t d = 0; d < 2; d++)
            EXPECT_EQ(gemm(factory.sliceProblem(full, d)).d().strides(),
                      (std::vector<size_t>{1, 256, 65536}))
                << "split " << dim << ", device " << d;
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

TEST(ClientProblemFactorySplit, BatchOffsetsFollowEachSlicesStrides)
{
    // 3 batches, float. A pointer-array run points batch b of a tensor at base + offset[b].
    ClientProblemFactory factory(makeArgs({130, 96, 3, 64}, 3, "M"));
    auto*                full = factory.problems()[0].get();
    const size_t         bpe  = sizeof(float);

    for(size_t d = 0; d < 3; d++)
    {
        // D keeps the full strides: its batches are a full 130 x 96 matrix apart on every device.
        const std::vector<size_t> expectedD = {0, 130 * 96 * bpe, 2 * 130 * 96 * bpe};
        EXPECT_EQ(factory.sliceBatchOffsets(full, d, ContractionProblemGemm::TENSOR::D), expectedD)
            << "device " << d;

        // C shrinks with the slice, so its batches are one slice apart.
        auto const&               slice     = gemm(factory.sliceProblem(full, d));
        const size_t              rows      = slice.c().sizes()[0];
        const std::vector<size_t> expectedC = {0, rows * 96 * bpe, 2 * rows * 96 * bpe};
        EXPECT_EQ(factory.sliceBatchOffsets(full, d, ContractionProblemGemm::TENSOR::C), expectedC)
            << "device " << d;
    }

    // B is replicated whole under a split of M.
    const std::vector<size_t> expectedB = {0, 64 * 96 * bpe, 2 * 64 * 96 * bpe};
    EXPECT_EQ(factory.sliceBatchOffsets(full, 2, ContractionProblemGemm::TENSOR::B), expectedB);

    EXPECT_THROW(factory.sliceBatchOffsets(full, 0, ContractionProblemGemm::TENSOR::BIAS),
                 std::runtime_error);
}

TEST(ClientProblemFactorySplit, BatchOffsetsOfASingleBatch)
{
    ClientProblemFactory factory(makeArgs({130, 96, 1, 64}, 2, "N"));
    auto*                full = factory.problems()[0].get();
    EXPECT_EQ(factory.sliceBatchOffsets(full, 1, ContractionProblemGemm::TENSOR::D),
              std::vector<size_t>{0});
}

TEST(ClientProblemFactorySplit, TheLastSliceHoldsEveryOtherSlicesInputs)
{
    // --split-placement S for A, B or C: every device reads the last device's buffer with its own
    // slice's size, so no slice may need more elements than the last one, per batch and in all.
    for(std::string dim : {"M", "N"})
    {
        ClientProblemFactory factory(makeArgs({130, 131, 3, 64}, 3, dim));
        auto*                full = factory.problems()[0].get();
        auto const&          last = gemm(factory.sliceProblem(full, 2));
        for(size_t d = 0; d < 2; d++)
        {
            auto const& slice = gemm(factory.sliceProblem(full, d));
            for(auto tensor : {ContractionProblemGemm::TENSOR::A,
                               ContractionProblemGemm::TENSOR::B,
                               ContractionProblemGemm::TENSOR::C})
            {
                EXPECT_LE(slice.tensors()[tensor].totalAllocatedElements(),
                          last.tensors()[tensor].totalAllocatedElements())
                    << "split " << dim << ", device " << d << ", tensor " << tensor;
                // The batch pointer array is the last device's too.
                EXPECT_LE(factory.sliceBatchOffsets(full, d, tensor).at(1),
                          factory.sliceBatchOffsets(full, 2, tensor).at(1))
                    << "split " << dim << ", device " << d << ", tensor " << tensor;
            }
        }
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

TEST(ClientProblemFactorySplit, RefusesGroupedGemm)
{
    auto args = makeArgs({128, 96, 1, 64}, 2, "M");
    TensileLite::TestUtils::setOption(args, "grouped-gemm", true);
    try
    {
        ClientProblemFactory factory(args);
        FAIL() << "a grouped gemm cannot be split";
    }
    catch(std::runtime_error const& e)
    {
        EXPECT_NE(std::string(e.what()).find("grouped-gemm"), std::string::npos) << e.what();
    }
}

TEST(ClientProblemFactorySplit, TightCStridesTightenWithTheSlice)
{
    // C strides that are the default ones for the full size are the default ones of a slice.
    auto args = makeArgs({256, 256, 1, 128}, 2, "N");
    TensileLite::TestUtils::setOption(
        args, "c-strides", std::vector<std::vector<size_t>>{{1, 256, 256 * 256}});
    ClientProblemFactory factory(args);
    auto*                full = factory.problems()[0].get();

    auto const& slice = gemm(factory.sliceProblem(full, 1));
    EXPECT_EQ(slice.c().strides()[1], 256u);
    EXPECT_EQ(slice.c().strides()[2], 256u * 128u);
}

TEST(ClientProblemFactorySplit, PaddedStridesStayAsGiven)
{
    auto args = makeArgs({256, 256, 1, 128}, 2, "M");
    TensileLite::TestUtils::setOption(
        args, "c-strides", std::vector<std::vector<size_t>>{{1, 300, 300 * 256}});
    ClientProblemFactory factory(args);
    auto*                full = factory.problems()[0].get();

    auto const& slice = gemm(factory.sliceProblem(full, 1));
    EXPECT_EQ(slice.c().strides()[1], 300u);
    EXPECT_EQ(slice.c().strides()[2], 300u * 256u);
}

TEST(ClientProblemFactorySplit, RejectsAProblemFromAnotherFactory)
{
    ClientProblemFactory factory(makeArgs({128, 96, 1, 64}, 2, "M"));
    ClientProblemFactory other(makeArgs({128, 96, 1, 64}, 2, "M"));
    EXPECT_THROW(factory.sliceProblem(other.problems()[0].get(), 0), std::runtime_error);
}

TEST(ClientProblemFactorySplit, MShareThatIsAlreadyAlignedIsKept)
{
    // float, M split: step 1, g = 2048 / gcd(2048, 32) = 64 rows. 512000 (M = 1024000) is a multiple of 64.
    ClientProblemFactory factory(makeArgs({1024000, 64, 1, 64}, 2, "M"));
    auto*                full = factory.problems()[0].get();
    EXPECT_EQ(gemm(factory.sliceProblem(full, 0)).c().sizes()[0], 512000u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 1)).c().sizes()[0], 512000u);
    EXPECT_EQ(factory.sliceOffsetInD(full, 1), 512000u);
}

TEST(ClientProblemFactorySplit, MShareIsRoundedUpWhenTheImbalanceIsTiny)
{
    // plain 500005 rounds up to 500032 (a multiple of 64); the last device gets 499978, 0.01% less.
    ClientProblemFactory factory(makeArgs({1000010, 64, 1, 64}, 2, "M"));
    auto*                full = factory.problems()[0].get();
    EXPECT_EQ(gemm(factory.sliceProblem(full, 0)).c().sizes()[0], 500032u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 1)).c().sizes()[0], 499978u);
    // The offset uses the same share as the slice sizes.
    EXPECT_EQ(factory.sliceOffsetInD(full, 1), 500032u);
}

TEST(ClientProblemFactorySplit, MShareIsNotRoundedWhenTheImbalanceIsOverOnePercent)
{
    // Rounding 100 to 64 rows would give {128, 72}.
    ClientProblemFactory factory(makeArgs({200, 64, 1, 64}, 2, "M"));
    auto*                full = factory.problems()[0].get();
    EXPECT_EQ(gemm(factory.sliceProblem(full, 0)).c().sizes()[0], 100u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 1)).c().sizes()[0], 100u);
    EXPECT_EQ(factory.sliceOffsetInD(full, 1), 100u);
}

TEST(ClientProblemFactorySplit, NShareIsRoundedToTheDPitch)
{
    // float, M = 3: a tight D has a pitch of 3 floats = 12 bytes, so a column start is a multiple
    // of 256 bytes every g = 2048 / gcd(2048, 3 * 32) = 64 columns. plain 50005 -> 50048, the last
    // device keeps 49962 columns (0.2% less).
    ClientProblemFactory factory(makeArgs({3, 100010, 1, 64}, 2, "N"));
    auto*                full = factory.problems()[0].get();
    EXPECT_EQ(gemm(factory.sliceProblem(full, 0)).c().sizes()[1], 50048u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 1)).c().sizes()[1], 49962u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 1)).b().sizes()[1], 49962u);
    EXPECT_EQ(factory.sliceOffsetInD(full, 1), 50048u * 3u);
}

TEST(ClientProblemFactorySplit, SplitAlignOffKeepsThePlainShare)
{
    auto args = makeArgs({1000010, 64, 1, 64}, 2, "M");
    TensileLite::TestUtils::setOption(args, "split-align", std::string("off"));
    ClientProblemFactory factory(args);
    auto*                full = factory.problems()[0].get();
    EXPECT_EQ(gemm(factory.sliceProblem(full, 0)).c().sizes()[0], 500005u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 1)).c().sizes()[0], 500005u);
}

TEST(ClientProblemFactorySplit, SplitAlignForceRoundsEvenWhenUnbalanced)
{
    auto args = makeArgs({200, 64, 1, 64}, 2, "M");
    TensileLite::TestUtils::setOption(args, "split-align", std::string("force"));
    ClientProblemFactory factory(args);
    auto*                full = factory.problems()[0].get();
    EXPECT_EQ(gemm(factory.sliceProblem(full, 0)).c().sizes()[0], 128u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 1)).c().sizes()[0], 72u);
    EXPECT_EQ(factory.sliceOffsetInD(full, 1), 128u);
}

TEST(ClientProblemFactorySplit, NShareIsRoundedToThePaddedLdbToo)
{
    // The share is cut with the split operand and C as well as D, as PartitionedGemm cuts it.
    // float, M = 64: the D and C pitch is 256 bytes, so they never need rounding. A padded
    // ldb of 65 floats = 260 bytes needs g = 2048 / gcd(2048, 65 * 32) = 64 columns, so plain
    // 50005 rounds up to 50048 (the last device keeps 49962, 0.2% less). Cut with D only, the
    // share stays 50005.
    const size_t useDefault = TensorDescriptor::UseDefaultStride;
    auto         args       = makeArgs({64, 100010, 1, 64}, 2, "N");
    TensileLite::TestUtils::setOption(
        args, "b-strides", std::vector<std::vector<size_t>>{{useDefault, 65, useDefault}});
    ClientProblemFactory factory(args);
    auto*                full = factory.problems()[0].get();
    ASSERT_EQ(gemm(full).b().strides()[1], 65u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 0)).c().sizes()[1], 50048u);
    EXPECT_EQ(gemm(factory.sliceProblem(full, 1)).c().sizes()[1], 49962u);
    EXPECT_EQ(factory.sliceOffsetInD(full, 1), 50048u * 64u);
}
