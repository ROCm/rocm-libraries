/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include <gtest/gtest.h>

#include <vector>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/transforms/asm/ra/PhysRegMatrix.hpp"

using namespace stinkytofu;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

/// One-segment range over [start, end).
LiveRange rangeOf(SlotIndex start, SlotIndex end) {
    LiveRange range;
    range.addSegment(start, end);
    range.finalize();
    return range;
}

// ---------------------------------------------------------------------------
// AsmTargetRegisters
// ---------------------------------------------------------------------------

TEST(AsmTargetRegistersTest, OnlyLiftedClassesAreAllocatable) {
    const AsmTargetRegisters target = AsmTargetRegisters::forArch(kArch);

    EXPECT_TRUE(target.isAllocatableClass(RegType::V));
    EXPECT_TRUE(target.isAllocatableClass(RegType::S));

    // Classes the lifter refuses or ignores must never be coloured.
    EXPECT_FALSE(target.isAllocatableClass(RegType::AGPR));
    EXPECT_FALSE(target.isAllocatableClass(RegType::ACC));
    EXPECT_FALSE(target.isAllocatableClass(RegType::EXEC));
    EXPECT_FALSE(target.isAllocatableClass(RegType::VCC));
    EXPECT_FALSE(target.isAllocatableClass(RegType::SCC));
    EXPECT_EQ(target.indexCount(RegType::AGPR), 0u);

    // The enumerated set is what the matrix builds storage from, so it has to
    // agree with the predicate rather than being a second, shorter list.
    const std::vector<RegType> classes(target.allocatableClasses().begin(),
                                       target.allocatableClasses().end());
    EXPECT_EQ(classes, (std::vector<RegType>{RegType::V, RegType::S}));
}

TEST(AsmTargetRegistersTest, EveryLimitComesFromTheArchitecture) {
    const AsmTargetRegisters target = AsmTargetRegisters::forArch(kArch);

    // Nothing is restated here: each value is the architecture's DEF_ARCH entry.
    EXPECT_EQ(target.indexCount(RegType::V), getMaxVGPR(kArch));
    EXPECT_EQ(target.indexCount(RegType::S), getMaxSGPR(kArch));
    EXPECT_EQ(target.allocationGranule(RegType::V), getVgprAllocGranule(kArch));
    EXPECT_EQ(target.totalPerSimd(RegType::V), getTotalVgprPerSimd(kArch));
}

TEST(AsmTargetRegistersTest, AddressableRangeIsSmallerThanThePhysicalFile) {
    const AsmTargetRegisters target = AsmTargetRegisters::forArch(kArch);

    // This target encodes v0-v255 directly and reaches the rest of its
    // physical file through VGPR-MSB, which is not modelled, so allocation
    // stops at the addressable range.
    EXPECT_EQ(target.indexCount(RegType::V), 256u);
    EXPECT_EQ(target.totalPerSimd(RegType::V), 1024u);
    EXPECT_LT(target.indexCount(RegType::V), target.totalPerSimd(RegType::V));

    EXPECT_TRUE(target.isAllocatable(RegType::V, 255));
    EXPECT_FALSE(target.isAllocatable(RegType::V, 256));
    EXPECT_EQ(target.indexCount(RegType::S), 102u);
    EXPECT_FALSE(target.isAllocatable(RegType::S, 102));
}

TEST(AsmTargetRegistersTest, NothingIsReservedUntilACallerSaysSo) {
    AsmTargetRegisters target = AsmTargetRegisters::forArch(kArch);
    EXPECT_TRUE(target.reservedRanges().empty());
    EXPECT_TRUE(target.isAllocatable(RegType::V, 0));

    target.reserve(RegType::V, 0, 4);

    EXPECT_EQ(target.reservedRanges().size(), 1u);
    for (uint32_t idx = 0; idx < 4; ++idx) EXPECT_FALSE(target.isAllocatable(RegType::V, idx));
    EXPECT_TRUE(target.isAllocatable(RegType::V, 4));
    // A hole in one class does not punch through another.
    EXPECT_TRUE(target.isAllocatable(RegType::S, 0));
}

// ---------------------------------------------------------------------------
// PhysRegMatrix
// ---------------------------------------------------------------------------

class PhysRegMatrixTest : public ::testing::Test {
   protected:
    AsmTargetRegisters target = AsmTargetRegisters::forArch(kArch);
};

TEST_F(PhysRegMatrixTest, EveryAllocatableUnitStartsFree) {
    const PhysRegMatrix matrix(target);
    const LiveRange range = rangeOf(0, 100);

    EXPECT_EQ(matrix.bindingCount(), 0u);
    EXPECT_TRUE(matrix.available(RegType::V, 0, range));
    EXPECT_TRUE(matrix.available(RegType::S, 101, range));
    EXPECT_FALSE(matrix.highestBound(RegType::V).has_value());
}

TEST_F(PhysRegMatrixTest, OverlappingRangesCannotShareAUnit) {
    PhysRegMatrix matrix(target);
    const LiveRange held = rangeOf(10, 20);
    matrix.bind(RegType::V, 7, /*value=*/1, held);

    const LiveRange overlapping = rangeOf(15, 25);
    EXPECT_FALSE(matrix.available(RegType::V, 7, overlapping));

    // Same index, different class: no aliasing between V and S here.
    EXPECT_TRUE(matrix.available(RegType::S, 7, overlapping));
    // Different index in the same class stays free.
    EXPECT_TRUE(matrix.available(RegType::V, 8, overlapping));
}

TEST_F(PhysRegMatrixTest, DisjointRangesShareAUnit) {
    PhysRegMatrix matrix(target);
    const LiveRange held = rangeOf(10, 20);
    matrix.bind(RegType::V, 3, /*value=*/1, held);

    // Half-open, so a range starting exactly where the other ends fits.
    const LiveRange after = rangeOf(20, 30);
    EXPECT_TRUE(matrix.available(RegType::V, 3, after));
}

TEST_F(PhysRegMatrixTest, AHoleInOneRangeLetsAnotherValueThrough) {
    PhysRegMatrix matrix(target);
    // The diamond shape: live in the entry, dead across one arm, live again.
    LiveRange holed;
    holed.addSegment(1, 8);
    holed.addSegment(14, 17);
    holed.finalize();
    matrix.bind(RegType::V, 5, /*value=*/1, holed);

    EXPECT_TRUE(matrix.available(RegType::V, 5, rangeOf(8, 14)));   // inside the hole
    EXPECT_FALSE(matrix.available(RegType::V, 5, rangeOf(8, 15)));  // crosses into the second
}

TEST_F(PhysRegMatrixTest, ConflictsNameTheOccupantSoAPolicyCanEvict) {
    PhysRegMatrix matrix(target);
    const LiveRange first = rangeOf(0, 10);
    const LiveRange second = rangeOf(4, 6);
    matrix.bind(RegType::V, 2, /*value=*/11, first);
    matrix.bind(RegType::V, 2, /*value=*/22, second);

    std::vector<SSAValueID> conflicts;
    matrix.collectConflicts(RegType::V, 2, rangeOf(5, 7), conflicts);
    EXPECT_EQ(conflicts, (std::vector<SSAValueID>{11, 22}));

    conflicts.clear();
    matrix.collectConflicts(RegType::V, 2, rangeOf(8, 9), conflicts);
    EXPECT_EQ(conflicts, (std::vector<SSAValueID>{11}));

    conflicts.clear();
    matrix.collectConflicts(RegType::V, 2, rangeOf(20, 30), conflicts);
    EXPECT_TRUE(conflicts.empty());
}

TEST_F(PhysRegMatrixTest, UnbindReleasesOnlyThatValue) {
    PhysRegMatrix matrix(target);
    const LiveRange first = rangeOf(0, 10);
    const LiveRange second = rangeOf(0, 10);
    matrix.bind(RegType::V, 4, /*value=*/11, first);
    matrix.bind(RegType::V, 4, /*value=*/22, second);
    ASSERT_EQ(matrix.bindingCount(), 2u);

    matrix.unbind(RegType::V, 4, /*value=*/11);

    EXPECT_EQ(matrix.bindingCount(), 1u);
    std::vector<SSAValueID> conflicts;
    matrix.collectConflicts(RegType::V, 4, rangeOf(0, 10), conflicts);
    EXPECT_EQ(conflicts, (std::vector<SSAValueID>{22}));

    matrix.unbind(RegType::V, 4, /*value=*/22);
    EXPECT_EQ(matrix.bindingCount(), 0u);
    EXPECT_TRUE(matrix.available(RegType::V, 4, rangeOf(0, 10)));
}

TEST_F(PhysRegMatrixTest, UnbindingAValueThatDoesNotHoldTheUnitIsSilent) {
    PhysRegMatrix matrix(target);
    const LiveRange held = rangeOf(0, 4);
    matrix.bind(RegType::V, 0, /*value=*/1, held);

    // Undoing a partly applied tuple needs no bookkeeping.
    matrix.unbind(RegType::V, 1, /*value=*/1);
    matrix.unbind(RegType::V, 0, /*value=*/99);

    EXPECT_EQ(matrix.bindingCount(), 1u);
}

TEST_F(PhysRegMatrixTest, FindFreeRunTakesTheLowestConsecutiveWindow) {
    PhysRegMatrix matrix(target);
    const LiveRange range = rangeOf(0, 10);
    const LiveRange held = rangeOf(0, 10);

    // Free v0, blocked v1, so a 2-wide run cannot start below v2.
    matrix.bind(RegType::V, 1, /*value=*/1, held);

    EXPECT_EQ(matrix.findFreeRun(RegType::V, 1, range), 0u);
    EXPECT_EQ(matrix.findFreeRun(RegType::V, 2, range), 2u);
    EXPECT_EQ(matrix.findFreeRun(RegType::V, 4, range), 2u);
}

TEST_F(PhysRegMatrixTest, ARunMayNotLeaveTheClass) {
    PhysRegMatrix matrix(target);
    const LiveRange range = rangeOf(0, 10);

    EXPECT_TRUE(matrix.runAvailable(RegType::S, 100, 2, range));
    EXPECT_FALSE(matrix.runAvailable(RegType::S, 101, 2, range));
    EXPECT_FALSE(matrix.runAvailable(RegType::S, 100, 3, range));
    EXPECT_EQ(matrix.findFreeRun(RegType::S, 200, range), std::nullopt);
}

TEST_F(PhysRegMatrixTest, ReservedUnitsAreNeverCandidates) {
    target.reserve(RegType::V, 0, 2);
    PhysRegMatrix matrix(target);
    const LiveRange range = rangeOf(0, 10);

    EXPECT_FALSE(matrix.available(RegType::V, 0, range));
    EXPECT_FALSE(matrix.available(RegType::V, 1, range));
    EXPECT_EQ(matrix.findFreeRun(RegType::V, 1, range), 2u);
    // A run may not straddle a reserved hole either.
    EXPECT_FALSE(matrix.runAvailable(RegType::V, 1, 2, range));
}

TEST_F(PhysRegMatrixTest, HighestBoundReportsTheAllocationWidth) {
    PhysRegMatrix matrix(target);
    const LiveRange range = rangeOf(0, 10);
    matrix.bind(RegType::V, 3, /*value=*/1, range);
    matrix.bind(RegType::V, 40, /*value=*/2, range);
    matrix.bind(RegType::S, 5, /*value=*/3, range);

    EXPECT_EQ(matrix.highestBound(RegType::V), 40u);
    EXPECT_EQ(matrix.highestBound(RegType::S), 5u);

    matrix.unbind(RegType::V, 40, /*value=*/2);
    EXPECT_EQ(matrix.highestBound(RegType::V), 3u);
}

}  // namespace
