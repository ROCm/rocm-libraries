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
