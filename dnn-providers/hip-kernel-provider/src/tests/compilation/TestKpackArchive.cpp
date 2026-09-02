// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "compilation/KpackArchive.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

/// A plausible packed archive and the entry it might hold, in bytes. Nothing here opens a
/// file: the subject is the arithmetic that decides whether a reader-reported length is
/// worth passing on.
constexpr std::uintmax_t ONE_MIB = 1024ULL * 1024;
constexpr std::uintmax_t TWO_GIB = 2ULL * 1024 * 1024 * 1024;
constexpr std::uintmax_t RATIO = 4096;

TEST(TestKpackArchive, AdmitsAnEntryOfOrdinaryProportions)
{
    // Real entries decompress near 4x, so a 4 MiB entry out of a 1 MiB archive is what
    // the common case looks like and must not be flagged.
    EXPECT_TRUE(isCredibleCodeObjectSize(4 * ONE_MIB, ONE_MIB));
}

TEST(TestKpackArchive, RejectsAnEntryOverTheAbsoluteCap)
{
    EXPECT_FALSE(isCredibleCodeObjectSize(TWO_GIB + 1, TWO_GIB));
    EXPECT_TRUE(isCredibleCodeObjectSize(TWO_GIB, TWO_GIB));
}

TEST(TestKpackArchive, RejectsAnEntryOverTheExpansionRatio)
{
    // A 1 MiB archive may declare up to 4096 MiB before the ratio objects -- which the
    // absolute cap would catch first, so the ratio is pinned at a size below it.
    constexpr std::uintmax_t ARCHIVE = 64ULL * 1024;
    EXPECT_TRUE(isCredibleCodeObjectSize(ARCHIVE * RATIO, ARCHIVE));
    EXPECT_FALSE(isCredibleCodeObjectSize(ARCHIVE * RATIO + RATIO, ARCHIVE));
}

TEST(TestKpackArchive, StandsTheRatioDownWhenTheArchiveSizeIsUnknown)
{
    // 0 is what open() leaves behind when file_size fails. The absolute cap still holds;
    // the ratio has nothing to compare against and must not invent a comparison.
    EXPECT_TRUE(isCredibleCodeObjectSize(TWO_GIB, 0));
    EXPECT_FALSE(isCredibleCodeObjectSize(TWO_GIB + 1, 0));
}

TEST(TestKpackArchive, DoesNotOverflowOnAHugeArchive)
{
    // The multiplication form of the ratio check wraps here and would admit every entry.
    constexpr std::uintmax_t ARCHIVE = std::numeric_limits<std::uintmax_t>::max();
    EXPECT_TRUE(isCredibleCodeObjectSize(ONE_MIB, ARCHIVE));
    EXPECT_FALSE(isCredibleCodeObjectSize(TWO_GIB + 1, ARCHIVE));
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
