// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "ContentCarryingTestGraph.hpp"
#include "KernelIngestorTestFixtures.hpp"

namespace hipdnn_plugin_sdk::ingestor::testing
{
namespace
{

KernelDefinition definitionFor(uint8_t kernel, uint8_t pack = 0xF0, uint8_t dispatch = 0xD0)
{
    KernelDefinition definition;
    definition.kernelId = testId(kernel);
    definition.packId = testId(pack);
    definition.dispatchId = testId(dispatch);
    return definition;
}

RankedEntry entryFor(const KernelDefinition& definition, double timeMs)
{
    return RankedEntry{definition.kernelId, definition.packId, definition.dispatchId, timeMs};
}

TEST(TestIngestorWinnerCache, ARecordCoveringEveryCandidateIsCovered)
{
    const auto first = definitionFor(0x01);
    const auto second = definitionFor(0x02);
    const WinnerRecord record{entryFor(second, 1.0), entryFor(first, 2.0)};

    EXPECT_TRUE(recordCovers(record, {first, second}));
}

TEST(TestIngestorWinnerCache, ARecordMissingACandidateIsNotCovered)
{
    const auto first = definitionFor(0x01);
    const auto second = definitionFor(0x02);
    const WinnerRecord record{entryFor(first, 2.0)};

    EXPECT_FALSE(recordCovers(record, {first, second}));
}

// The asymmetry is the point: a record wider than the current candidate set still covers
// it. Treating extra entries as a coverage failure would re-benchmark on every narrowed
// knob filter and destroy the reuse the ranked list exists for.
TEST(TestIngestorWinnerCache, ARecordWiderThanTheCandidateSetStillCoversIt)
{
    const auto first = definitionFor(0x01);
    const auto second = definitionFor(0x02);
    const WinnerRecord record{entryFor(first, 1.0), entryFor(second, 2.0)};

    EXPECT_TRUE(recordCovers(record, {first}));
}

TEST(TestIngestorWinnerCache, AnEmptyCandidateSetIsVacuouslyCovered)
{
    EXPECT_TRUE(recordCovers(WinnerRecord{entryFor(definitionFor(0x01), 1.0)}, {}));
}

TEST(TestIngestorWinnerCache, AnEmptyRecordCoversNothing)
{
    EXPECT_FALSE(recordCovers(WinnerRecord{}, {definitionFor(0x01)}));
}

TEST(TestIngestorWinnerCache, OrderByRecordPutsCandidatesIntoMeasuredOrder)
{
    const auto slow = definitionFor(0x01);
    const auto fast = definitionFor(0x02);
    const WinnerRecord record{entryFor(fast, 0.5), entryFor(slow, 5.0)};

    const auto ordered = orderByRecord(record, {slow, fast});

    ASSERT_EQ(ordered.size(), 2U);
    EXPECT_EQ(ordered[0].kernelId, fast.kernelId) << "the measured winner must come first";
    EXPECT_EQ(ordered[1].kernelId, slow.kernelId);
}

// Coverage asks "was this kernel measured"; agreement asks "is it still the same kernel".
// A pack replaced between runs can leave the id intact while the kernel behind it moved.
TEST(TestIngestorWinnerCache, OrderByRecordSkipsAnEntryWhosePackNoLongerAgrees)
{
    const auto current = definitionFor(0x01, 0xA1);
    const WinnerRecord record{entryFor(definitionFor(0x01, 0xB2), 1.0)};

    EXPECT_TRUE(orderByRecord(record, {current}).empty())
        << "a kernel id that now resolves to a different pack is a different kernel";
}

TEST(TestIngestorWinnerCache, OrderByRecordSkipsAnEntryWhoseDispatchNoLongerAgrees)
{
    const auto current = definitionFor(0x01, 0xF0, 0xD1);
    const WinnerRecord record{entryFor(definitionFor(0x01, 0xF0, 0xD2), 1.0)};

    EXPECT_TRUE(orderByRecord(record, {current}).empty());
}

TEST(TestIngestorWinnerCache, OrderByRecordDropsRecordEntriesAbsentFromTheCandidates)
{
    const auto present = definitionFor(0x01);
    const WinnerRecord record{entryFor(definitionFor(0x02), 0.5), entryFor(present, 1.0)};

    const auto ordered = orderByRecord(record, {present});

    ASSERT_EQ(ordered.size(), 1U);
    EXPECT_EQ(ordered[0].kernelId, present.kernelId);
}

TEST(TestIngestorWinnerCache, KeysDifferingOnlyInDeviceAreDistinct)
{
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    DeviceProperties first;
    first.gcnArchName = "gfx942";
    DeviceProperties second;
    second.gcnArchName = "gfx950";

    const WinnerKey firstKey{GraphContentKey{graph}, DeviceKey{first}};
    const WinnerKey secondKey{GraphContentKey{graph}, DeviceKey{second}};

    EXPECT_NE(firstKey, secondKey);
    EXPECT_NE(WinnerKeyHash{}(firstKey), WinnerKeyHash{}(secondKey));
}

TEST(TestIngestorWinnerCache, KeysDifferingOnlyInGraphAreDistinct)
{
    ContentCarryingTestGraph::Spec narrow;
    narrow.tensors[0].dims = {4, 8};
    ContentCarryingTestGraph::Spec wide;
    wide.tensors[0].dims = {4, 16};

    DeviceProperties properties;
    properties.gcnArchName = "gfx942";

    const WinnerKey firstKey{GraphContentKey{ContentCarryingTestGraph{narrow}},
                             DeviceKey{properties}};
    const WinnerKey secondKey{GraphContentKey{ContentCarryingTestGraph{wide}},
                              DeviceKey{properties}};

    EXPECT_NE(firstKey, secondKey);
}

TEST(TestIngestorWinnerCache, EqualGraphAndDeviceProduceEqualKeys)
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx942";

    const WinnerKey firstKey{GraphContentKey{ContentCarryingTestGraph{}}, DeviceKey{properties}};
    const WinnerKey secondKey{GraphContentKey{ContentCarryingTestGraph{}}, DeviceKey{properties}};

    EXPECT_EQ(firstKey, secondKey);
    EXPECT_EQ(WinnerKeyHash{}(firstKey), WinnerKeyHash{}(secondKey));
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
