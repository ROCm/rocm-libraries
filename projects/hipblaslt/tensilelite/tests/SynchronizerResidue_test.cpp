// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "SynchronizerValidator.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

using TensileLite::Client::scanSynchronizerResidue;
using TensileLite::Client::SynchronizerResidue;

namespace
{
    // Buffer of *ints* ints, all zero unless listed in *nonzero* as
    // (int index, value).
    std::vector<uint8_t> makeBuffer(size_t                                          ints,
                                    std::vector<std::pair<size_t, uint32_t>> const& nonzero = {})
    {
        std::vector<uint8_t> buf(ints * sizeof(uint32_t), 0);
        for(auto const& [idx, value] : nonzero)
            std::memcpy(buf.data() + idx * sizeof(uint32_t), &value, sizeof(uint32_t));
        return buf;
    }
}

TEST(SynchronizerResidue, CleanBufferReportsNothing)
{
    auto                buf = makeBuffer(64);
    SynchronizerResidue r;
    EXPECT_FALSE(scanSynchronizerResidue(buf.data(), buf.size(), r));
}

TEST(SynchronizerResidue, EmptyBufferIsClean)
{
    SynchronizerResidue r;
    EXPECT_FALSE(scanSynchronizerResidue(nullptr, 0, r));
}

// A leftover work-queue counter of 1 occupies one nonzero byte. Reporting must
// be in ints, so this is one dirty counter -- not one dirty byte.
TEST(SynchronizerResidue, CounterOfOneIsOneDirtyInt)
{
    auto                buf = makeBuffer(64, {{0, 1}});
    SynchronizerResidue r;
    ASSERT_TRUE(scanSynchronizerResidue(buf.data(), buf.size(), r));
    EXPECT_EQ(r.nonzeroInts, 1u);
    EXPECT_EQ(r.firstInt, 0u);
    EXPECT_EQ(r.totalInts, 64u);
}

// The per-XCD counters sit one per 128-byte cache line, so queue 1's counter is
// at byte 128 == int offset 32. Regression guard on the byte->int conversion.
TEST(SynchronizerResidue, OffsetIsReportedInIntsNotBytes)
{
    auto                buf = makeBuffer(64, {{32, 1}});
    SynchronizerResidue r;
    ASSERT_TRUE(scanSynchronizerResidue(buf.data(), buf.size(), r));
    EXPECT_EQ(r.firstInt, 32u);
    EXPECT_EQ(r.nonzeroInts, 1u);
}

TEST(SynchronizerResidue, CountsEveryDirtyIntAndReportsTheFirst)
{
    auto                buf = makeBuffer(64, {{5, 7}, {9, 1}, {40, 0xFFFFFFFFu}});
    SynchronizerResidue r;
    ASSERT_TRUE(scanSynchronizerResidue(buf.data(), buf.size(), r));
    EXPECT_EQ(r.nonzeroInts, 3u);
    EXPECT_EQ(r.firstInt, 5u);
}

// Boundary: residue in the very last counter must not be missed.
TEST(SynchronizerResidue, FindsResidueInTheLastInt)
{
    auto                buf = makeBuffer(64, {{63, 1}});
    SynchronizerResidue r;
    ASSERT_TRUE(scanSynchronizerResidue(buf.data(), buf.size(), r));
    EXPECT_EQ(r.firstInt, 63u);
    EXPECT_EQ(r.nonzeroInts, 1u);
}

// The clean verdict comes from a vectorized OR over the buffer: residue must be
// found at every position, including the scalar tail and sizes that are not a
// multiple of any vector width.
TEST(SynchronizerResidue, FindsResidueAtEveryPositionOfOddSizes)
{
    for(size_t ints = 1; ints <= 67; ints++)
    {
        for(size_t pos = 0; pos < ints; pos++)
        {
            auto                buf = makeBuffer(ints, {{pos, 1}});
            SynchronizerResidue r;
            ASSERT_TRUE(scanSynchronizerResidue(buf.data(), buf.size(), r))
                << "ints " << ints << ", residue at " << pos;
            EXPECT_EQ(r.totalInts, ints);
            EXPECT_EQ(r.nonzeroInts, 1u);
            EXPECT_EQ(r.firstInt, pos);
        }
    }
}

// The client's buffer size (409,600 float-alpha elements): clean, and residue
// that sets only the top bit (a sign-like flag), far from the start.
TEST(SynchronizerResidue, ClientSizedBufferCleanAndHighBitResidue)
{
    size_t const ints = 409600;

    // A clean scan leaves *out* untouched.
    auto                clean = makeBuffer(ints);
    SynchronizerResidue r;
    r.totalInts   = 11;
    r.nonzeroInts = 22;
    r.firstInt    = 33;
    EXPECT_FALSE(scanSynchronizerResidue(clean.data(), clean.size(), r));
    EXPECT_EQ(r.totalInts, 11u);
    EXPECT_EQ(r.nonzeroInts, 22u);
    EXPECT_EQ(r.firstInt, 33u);

    auto buf = makeBuffer(ints, {{300001, 0x80000000u}, {ints - 1, 0x80000000u}});
    ASSERT_TRUE(scanSynchronizerResidue(buf.data(), buf.size(), r));
    EXPECT_EQ(r.totalInts, ints);
    EXPECT_EQ(r.nonzeroInts, 2u);
    EXPECT_EQ(r.firstInt, 300001u);
}
