// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/Registers.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>

using namespace rocm::tensilelite::codegen;

static constexpr RegisterKind sgpr = RegisterKind::sgpr;
static constexpr RegisterKind vgpr = RegisterKind::vgpr;
static constexpr RegisterKind agpr = RegisterKind::agpr;

namespace {

class Registers : public testing::Test {
protected:
  Registers() : target(Target::get("gfx950")), allocator(target) {
    target.selectForCurrentThread();
  }

  // The allocator is declared after the target and before any range a test
  // creates, so ranges are destroyed first.
  Target target;
  RegisterAllocator allocator;
};

} // namespace

static std::string limitMessage(const RegisterAllocator &allocator) {
  try {
    allocator.checkLimits();
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return {};
}

TEST_F(Registers, ReleasesOnScopeExit) {
  {
    RegisterRange range = allocator.allocate(vgpr, 4);
    EXPECT_EQ(range.start(), 0);
  }
  EXPECT_EQ(allocator.allocate(vgpr, 4).start(), 0);
}

TEST_F(Registers, ReleaseReturnsRegistersEarly) {
  RegisterRange range = allocator.allocate(vgpr, 2);
  range.release();
  EXPECT_EQ(allocator.allocate(vgpr, 2).start(), 0);
  range.release();
}

TEST_F(Registers, MoveConstructionTransfersOwnership) {
  RegisterRange first = allocator.allocate(vgpr, 2);
  {
    RegisterRange second(std::move(first));
    first.release();
    EXPECT_EQ(second.start(), 0);
    EXPECT_EQ(second.count(), 2);
    EXPECT_EQ(allocator.allocate(vgpr, 2).start(), 2);
  }
  EXPECT_EQ(allocator.allocate(vgpr, 2).start(), 0);
}

TEST_F(Registers, MoveAssignmentReleasesOldRegisters) {
  RegisterRange a = allocator.allocate(vgpr, 2);
  RegisterRange b = allocator.allocate(vgpr, 2);
  EXPECT_EQ(a.start(), 0);
  EXPECT_EQ(b.start(), 2);
  a = std::move(b);
  EXPECT_EQ(a.start(), 2);
  RegisterRange c = allocator.allocate(vgpr, 2);
  EXPECT_EQ(c.start(), 0);
  RegisterRange d = allocator.allocate(vgpr, 2);
  EXPECT_EQ(d.start(), 4);
}

TEST_F(Registers, ReserveLandsOnExactIndices) {
  RegisterRange high = allocator.reserve(sgpr, 2, 1);
  RegisterRange low = allocator.reserve(sgpr, 0, 2);
  RegisterRange vector = allocator.reserve(vgpr, 0, 1);
  EXPECT_EQ(high.start(), 2);
  EXPECT_EQ(low.start(), 0);
  EXPECT_EQ(vector.start(), 0);
  EXPECT_EQ(allocator.allocate(sgpr, 1).start(), 3);
}

TEST_F(Registers, ReserveAfterAllocationFillsHolesExactly) {
  RegisterRange first = allocator.allocate(vgpr, 2);
  RegisterRange reserved = allocator.reserve(vgpr, 5, 2);
  EXPECT_EQ(first.start(), 0);
  EXPECT_EQ(reserved.start(), 5);
  EXPECT_EQ(allocator.allocate(vgpr, 1).start(), 2);
  EXPECT_EQ(allocator.highWaterMark(vgpr), 7);
}

TEST_F(Registers, ConflictingReservationThrows) {
  RegisterRange abi = allocator.reserve(sgpr, 0, 2);
  EXPECT_THROW(allocator.reserve(sgpr, 1, 1), std::invalid_argument);
  RegisterRange range = allocator.allocate(vgpr, 4);
  EXPECT_THROW(allocator.reserve(vgpr, 2, 1), std::invalid_argument);
  EXPECT_EQ(allocator.highWaterMark(vgpr), 4);
  EXPECT_EQ(allocator.reserve(sgpr, 2, 1).start(), 2);
}

TEST_F(Registers, InvalidArgumentsThrow) {
  EXPECT_THROW(allocator.allocate(vgpr, 0), std::invalid_argument);
  EXPECT_THROW(allocator.allocate(vgpr, 1, 0), std::invalid_argument);
  EXPECT_THROW(allocator.reserve(sgpr, -1, 1), std::invalid_argument);
  EXPECT_THROW(allocator.reserve(sgpr, 0, 0), std::invalid_argument);
}

TEST_F(Registers, AlignmentIsHonored) {
  RegisterRange a = allocator.allocate(vgpr, 1);
  RegisterRange b = allocator.allocate(vgpr, 4, 4);
  RegisterRange c = allocator.allocate(vgpr, 2, 2);
  RegisterRange d = allocator.allocate(vgpr, 1);
  EXPECT_EQ(a.start(), 0);
  EXPECT_EQ(b.start(), 4);
  EXPECT_EQ(c.start(), 2);
  EXPECT_EQ(d.start(), 1);
}

TEST_F(Registers, HighWaterMarkTracksPeakUsage) {
  EXPECT_EQ(allocator.highWaterMark(sgpr), 0);
  EXPECT_EQ(allocator.highWaterMark(vgpr), 0);
  EXPECT_EQ(allocator.highWaterMark(agpr), 0);
  {
    RegisterRange range = allocator.allocate(vgpr, 3);
    EXPECT_EQ(allocator.highWaterMark(vgpr), 3);
  }
  EXPECT_EQ(allocator.highWaterMark(vgpr), 3);
  RegisterRange reserved = allocator.reserve(sgpr, 2, 1);
  EXPECT_EQ(allocator.highWaterMark(sgpr), 3);
}

TEST_F(Registers, LimitErrorReportsVgpr) {
  RegisterRange full = allocator.allocate(vgpr, target.maxVgpr());
  EXPECT_NO_THROW(allocator.checkLimits());
  RegisterRange extra = allocator.allocate(vgpr, 1);
  EXPECT_THROW(allocator.checkLimits(), std::runtime_error);
  EXPECT_NE(limitMessage(allocator).find("vgpr"), std::string::npos);
}

TEST_F(Registers, LimitErrorReportsSgpr) {
  RegisterRange full = allocator.allocate(sgpr, target.maxSgpr() + 1);
  EXPECT_THROW(allocator.checkLimits(), std::runtime_error);
  EXPECT_NE(limitMessage(allocator).find("sgpr"), std::string::npos);
}

TEST_F(Registers, LimitErrorReportsAgpr) {
  RegisterRange full = allocator.allocate(agpr, target.maxAgpr());
  EXPECT_NO_THROW(allocator.checkLimits());
  RegisterRange extra = allocator.allocate(agpr, 1);
  EXPECT_THROW(allocator.checkLimits(), std::runtime_error);
  std::string message = limitMessage(allocator);
  EXPECT_NE(message.find("257 agprs"), std::string::npos) << message;
  EXPECT_EQ(message.find("unified"), std::string::npos) << message;
}

TEST_F(Registers, LimitErrorReportsUnifiedBudget) {
  RegisterRange vectors = allocator.allocate(vgpr, 250);
  RegisterRange full = allocator.allocate(agpr, 256);
  // The accum offset rounds 250 up to 256, which fills the budget exactly.
  EXPECT_NO_THROW(allocator.checkLimits());
  RegisterRange extra = allocator.allocate(agpr, 1);
  EXPECT_THROW(allocator.checkLimits(), std::runtime_error);
  std::string message = limitMessage(allocator);
  EXPECT_NE(message.find("unified"), std::string::npos) << message;
  EXPECT_NE(message.find("accum offset 256"), std::string::npos) << message;
}

TEST_F(Registers, AgprsAllocateIndependently) {
  RegisterRange vector = allocator.allocate(vgpr, 3);
  RegisterRange first = allocator.allocate(agpr, 2);
  EXPECT_EQ(first.start(), 0);
  EXPECT_EQ(first.kind(), agpr);
  RegisterRange aligned = allocator.allocate(agpr, 4, 4);
  EXPECT_EQ(aligned.start(), 4);
  EXPECT_EQ(allocator.highWaterMark(agpr), 8);
  EXPECT_EQ(allocator.highWaterMark(vgpr), 3);
  first.release();
  EXPECT_EQ(allocator.allocate(agpr, 2).start(), 0);
}

TEST_F(Registers, AgprContainerText) {
  RegisterRange range = allocator.reserve(agpr, 4, 4);
  EXPECT_EQ(range.container()->toString(), "acc[4:7]");
  EXPECT_EQ(range.sub(1)->toString(), "acc5");
  EXPECT_EQ(range.sub(2, 2)->toString(), "acc[6:7]");
  range.release();
  EXPECT_EQ(allocator.allocate(agpr, 1).container()->toString(), "acc0");
}

TEST_F(Registers, AccumOffsetRoundsArchVgprsUpToEight) {
  EXPECT_EQ(allocator.accumOffset(), 8);
  RegisterRange nine = allocator.allocate(vgpr, 9);
  EXPECT_EQ(allocator.accumOffset(), 16);
  RegisterRange seven = allocator.allocate(vgpr, 7);
  EXPECT_EQ(allocator.accumOffset(), 16);
  RegisterRange accumulators = allocator.allocate(agpr, 20);
  EXPECT_EQ(allocator.accumOffset(), 16);
}

TEST_F(Registers, ContainerText) {
  RegisterRange vectorRange = allocator.reserve(vgpr, 4, 4);
  RegisterRange scalarRange = allocator.reserve(sgpr, 0, 2);
  RegisterRange single = allocator.reserve(sgpr, 2, 1);
  EXPECT_EQ(vectorRange.container()->toString(), "v[4:7]");
  EXPECT_EQ(scalarRange.container()->toString(), "s[0:1]");
  EXPECT_EQ(vectorRange.sub(1)->toString(), "v5");
  EXPECT_EQ(vectorRange.sub(2, 2)->toString(), "v[6:7]");
  EXPECT_EQ(single.container()->toString(), "s2");
}

TEST_F(Registers, SubRangeOutOfBoundsThrows) {
  RegisterRange range = allocator.allocate(vgpr, 4);
  EXPECT_THROW(range.sub(3, 2), std::out_of_range);
}

TEST_F(Registers, RefCoversRange) {
  RegisterRange range = allocator.reserve(vgpr, 4, 4);
  EXPECT_EQ(range.ref(), (RegisterRef{vgpr, 4, 4}));
  EXPECT_EQ(range.ref(1, 2), (RegisterRef{vgpr, 5, 2}));
  EXPECT_EQ(range.ref(3), (RegisterRef{vgpr, 7, 1}));
  EXPECT_EQ(range.ref(1, 2).container()->toString(), "v[5:6]");
  EXPECT_EQ(range.ref().container()->toString(), "v[4:7]");
}

TEST_F(Registers, RefContainerTextPerKind) {
  RegisterRange scalars = allocator.reserve(sgpr, 2, 4);
  RegisterRange accumulators = allocator.reserve(agpr, 4, 4);
  EXPECT_EQ(scalars.ref(1, 2).container()->toString(), "s[3:4]");
  EXPECT_EQ(scalars.ref(2).container()->toString(), "s4");
  EXPECT_EQ(accumulators.ref(1, 2).container()->toString(), "acc[5:6]");
  EXPECT_EQ(accumulators.ref(1).container()->toString(), "acc5");
}

TEST_F(Registers, RefOutOfBoundsThrows) {
  RegisterRange range = allocator.allocate(vgpr, 4);
  EXPECT_THROW(range.ref(-1, 1), std::out_of_range);
  EXPECT_THROW(range.ref(0, 0), std::out_of_range);
  EXPECT_THROW(range.ref(3, 2), std::out_of_range);
  EXPECT_THROW(range.ref(4, 1), std::out_of_range);
  EXPECT_NO_THROW(range.ref(3, 1));
}

TEST_F(Registers, SliceReturnsRefsOfEachKind) {
  RegisterRef pair{vgpr, 5, 2};
  EXPECT_EQ(pair.slice(0), (RegisterRef{vgpr, 5, 1}));
  EXPECT_EQ(pair.slice(1), (RegisterRef{vgpr, 6, 1}));
  EXPECT_EQ(pair.slice(0, 2), pair);
  RegisterRef quad{sgpr, 4, 4};
  EXPECT_EQ(quad.slice(1, 2), (RegisterRef{sgpr, 5, 2}));
  EXPECT_EQ(quad.slice(3), (RegisterRef{sgpr, 7, 1}));
  EXPECT_EQ((RegisterRef{agpr, 2, 2}).slice(1), (RegisterRef{agpr, 3, 1}));
}

TEST_F(Registers, SliceOutOfBoundsThrows) {
  RegisterRef pair{vgpr, 5, 2};
  EXPECT_THROW(pair.slice(-1), std::out_of_range);
  EXPECT_THROW(pair.slice(2), std::out_of_range);
  EXPECT_THROW(pair.slice(1, 2), std::out_of_range);
  EXPECT_THROW(pair.slice(0, 0), std::out_of_range);
  EXPECT_NO_THROW(pair.slice(1));
}
