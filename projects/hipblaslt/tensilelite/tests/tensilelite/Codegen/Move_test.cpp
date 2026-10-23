// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/IRBuilder.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr RegisterKind sgpr = RegisterKind::sgpr;
constexpr RegisterKind vgpr = RegisterKind::vgpr;
constexpr RegisterKind agpr = RegisterKind::agpr;

class MoveTest : public BuilderTest {
protected:
  std::string emitMov32(RegisterRef dst, Operand src) {
    return emitted([&](IRBuilder &local) { local.movb32(dst, src); });
  }
  std::string emitMov64(RegisterRef dst, Operand src) {
    return emitted([&](IRBuilder &local) { local.movb64(dst, src); });
  }
  int instructionCount(const std::string &out) {
    return countLinesContaining(out, "_mov_") +
           countLinesContaining(out, "accvgpr");
  }
};

TEST_F(MoveTest, Movb32ScalarDestination) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange src = builder.allocateSgpr(1);
  EXPECT_EQ(emitMov32(dst.ref(), src),
            line("s_mov_b32", text(dst), text(src)) + "\n");
  expectLinesInOrder(emitMov32(dst.ref(), 0x12345),
                     {line("s_mov_b32", text(dst), "74565")});
  expectLinesInOrder(emitMov32(dst.ref(), 7),
                     {line("s_mov_b32", text(dst), "7")});
}

TEST_F(MoveTest, Movb32VectorDestination) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange vsrc = builder.allocateVgpr(1);
  RegisterRange ssrc = builder.allocateSgpr(1);
  RegisterRange asrc = builder.allocateAgpr(1);
  expectLinesInOrder(emitMov32(dst.ref(), vsrc),
                     {line("v_mov_b32", text(dst), text(vsrc))});
  expectLinesInOrder(emitMov32(dst.ref(), ssrc),
                     {line("v_mov_b32", text(dst), text(ssrc))});
  expectLinesInOrder(emitMov32(dst.ref(), -1),
                     {line("v_mov_b32", text(dst), "-1")});
  expectLinesInOrder(emitMov32(dst.ref(), asrc),
                     {line("v_accvgpr_read_b32", text(dst), text(asrc))});
}

TEST_F(MoveTest, Movb32AccumulatorDestination) {
  RegisterRange dst = builder.allocateAgpr(1);
  RegisterRange vsrc = builder.allocateVgpr(1);
  RegisterRange ssrc = builder.allocateSgpr(1);
  expectLinesInOrder(emitMov32(dst.ref(), vsrc),
                     {line("v_accvgpr_write_b32", text(dst), text(vsrc))});
  expectLinesInOrder(emitMov32(dst.ref(), ssrc),
                     {line("v_accvgpr_write_b32", text(dst), text(ssrc))});
  expectLinesInOrder(emitMov32(dst.ref(), 7),
                     {line("v_accvgpr_write_b32", text(dst), "7")});
}

TEST_F(MoveTest, Movb32InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange adst = builder.allocateAgpr(1);
  RegisterRange asrc = builder.allocateAgpr(1);
  RegisterRange vsrc = builder.allocateVgpr(1);
  RegisterRange vpair = builder.allocateVgpr(2);
  RegisterRange spair = builder.allocateSgpr(2);
  EXPECT_THROW(builder.movb32(sdst.ref(), vsrc), std::invalid_argument);
  EXPECT_THROW(builder.movb32(sdst.ref(), asrc), std::invalid_argument);
  EXPECT_THROW(builder.movb32(adst.ref(), 0x12345), std::invalid_argument);
  EXPECT_THROW(builder.movb32(adst.ref(), asrc), std::invalid_argument);
  EXPECT_THROW(builder.movb32(vpair.ref(), vsrc), std::invalid_argument);
  EXPECT_THROW(builder.movb32(sdst.ref(), spair), std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
}

TEST_F(MoveTest, Movb64RegisterPairs) {
  RegisterRange sdst = builder.allocateSgpr(2, 2);
  RegisterRange ssrc = builder.allocateSgpr(2, 2);
  RegisterRange vdst = builder.allocateVgpr(2, 2);
  RegisterRange vsrc = builder.allocateVgpr(2, 2);
  expectLinesInOrder(emitMov64(sdst.ref(), ssrc),
                     {line("s_mov_b64", text(sdst), text(ssrc))});
  expectLinesInOrder(emitMov64(vdst.ref(), vsrc),
                     {line("v_mov_b64", text(vdst), text(vsrc))});
  expectLinesInOrder(emitMov64(vdst.ref(), ssrc),
                     {line("v_mov_b64", text(vdst), text(ssrc))});
}

TEST_F(MoveTest, Movb64SingleInstructionImmediates) {
  RegisterRange sdst = builder.allocateSgpr(2, 2);
  RegisterRange vdst = builder.allocateVgpr(2, 2);
  struct Case {
    int32_t value;
    std::string printed;
  };
  // A non-inline literal is one 64-bit mov only when it is non-negative, since
  // the hardware zero-extends it.
  for (const Case &current :
       {Case{-1, "-1"}, Case{64, "64"}, Case{-16, "-16"}, Case{100, "100"},
        Case{0x12345, "74565"}, Case{INT32_MAX, "2147483647"}}) {
    SCOPED_TRACE(current.printed);
    std::string scalar = emitMov64(sdst.ref(), current.value);
    expectLinesInOrder(scalar,
                       {line("s_mov_b64", text(sdst), current.printed)});
    EXPECT_EQ(instructionCount(scalar), 1);
    std::string vector = emitMov64(vdst.ref(), current.value);
    expectLinesInOrder(vector,
                       {line("v_mov_b64", text(vdst), current.printed)});
    EXPECT_EQ(instructionCount(vector), 1);
  }
}

TEST_F(MoveTest, Movb64SplitImmediates) {
  RegisterRange sdst = builder.allocateSgpr(2, 2);
  RegisterRange vdst = builder.allocateVgpr(2, 2);
  RegisterRange adst = builder.allocateAgpr(2, 2);
  // A negative literal sign-extends to a high half of -1 and splits.
  std::string negative = emitMov64(sdst.ref(), -100);
  expectLinesInOrder(negative, {line("s_mov_b32", text(sdst.ref(0, 1)), "-100"),
                                line("s_mov_b32", text(sdst.ref(1, 1)), "-1")});
  EXPECT_EQ(instructionCount(negative), 2);

  expectLinesInOrder(emitMov64(vdst.ref(), -17),
                     {line("v_mov_b32", text(vdst.ref(0, 1)), "-17"),
                      line("v_mov_b32", text(vdst.ref(1, 1)), "-1")});
  expectLinesInOrder(emitMov64(vdst.ref(), INT32_MIN),
                     {line("v_mov_b32", text(vdst.ref(0, 1)), "-2147483648"),
                      line("v_mov_b32", text(vdst.ref(1, 1)), "-1")});
  expectLinesInOrder(emitMov64(adst.ref(), -5),
                     {line("v_accvgpr_write_b32", text(adst.ref(0, 1)), "-5"),
                      line("v_accvgpr_write_b32", text(adst.ref(1, 1)), "-1")});
}

TEST_F(MoveTest, Movb64MisalignedPairsSplit) {
  RegisterRange vrange = builder.allocateVgpr(3, 2);
  RegisterRange vsrc = builder.allocateVgpr(2, 2);
  RegisterRange vdst = builder.allocateVgpr(2, 2);
  std::string toMisaligned = emitMov64(vrange.ref(1, 2), vsrc);
  expectLinesInOrder(
      toMisaligned,
      {line("v_mov_b32", text(vrange.ref(1, 1)), text(vsrc.ref(0, 1))),
       line("v_mov_b32", text(vrange.ref(2, 1)), text(vsrc.ref(1, 1)))});
  EXPECT_EQ(instructionCount(toMisaligned), 2);
  std::string fromMisaligned = emitMov64(vdst.ref(), vrange.ref(1, 2));
  EXPECT_EQ(countLinesContaining(fromMisaligned, "v_mov_b64"), 0);
  EXPECT_EQ(instructionCount(fromMisaligned), 2);
}

TEST_F(MoveTest, Movb64OverlappingCopyOrdersHalves) {
  RegisterRange range = builder.allocateVgpr(4, 2);
  // Copying up reads the high source word first so it is not overwritten.
  expectLinesInOrder(
      emitMov64(range.ref(1, 2), range.ref(0, 2)),
      {line("v_mov_b32", text(range.ref(2, 1)), text(range.ref(1, 1))),
       line("v_mov_b32", text(range.ref(1, 1)), text(range.ref(0, 1)))});
  expectLinesInOrder(
      emitMov64(range.ref(0, 2), range.ref(1, 2)),
      {line("v_mov_b32", text(range.ref(0, 1)), text(range.ref(1, 1))),
       line("v_mov_b32", text(range.ref(1, 1)), text(range.ref(2, 1)))});
}

TEST_F(MoveTest, Movb64AccumulatorPairsSplit) {
  RegisterRange apair = builder.allocateAgpr(2, 2);
  RegisterRange vpair = builder.allocateVgpr(2, 2);
  expectLinesInOrder(emitMov64(apair.ref(), vpair),
                     {line("v_accvgpr_write_b32", text(apair.ref(0, 1)),
                           text(vpair.ref(0, 1))),
                      line("v_accvgpr_write_b32", text(apair.ref(1, 1)),
                           text(vpair.ref(1, 1)))});
  expectLinesInOrder(
      emitMov64(vpair.ref(), apair),
      {line("v_accvgpr_read_b32", text(vpair.ref(0, 1)), text(apair.ref(0, 1))),
       line("v_accvgpr_read_b32", text(vpair.ref(1, 1)),
            text(apair.ref(1, 1)))});
}

TEST_F(MoveTest, Movb64InvalidOperandsThrowWithoutEmitting) {
  RegisterRange single = builder.allocateVgpr(1);
  RegisterRange vpair = builder.allocateVgpr(2, 2);
  RegisterRange spair = builder.allocateSgpr(2, 2);
  RegisterRange apair = builder.allocateAgpr(2, 2);
  EXPECT_THROW(builder.movb64(single.ref(), vpair), std::invalid_argument);
  EXPECT_THROW(builder.movb64(vpair.ref(), single), std::invalid_argument);
  EXPECT_THROW(builder.movb64(spair.ref(), vpair), std::invalid_argument);
  EXPECT_THROW(builder.movb64(spair.ref(), apair), std::invalid_argument);
  EXPECT_THROW(builder.movb64(apair.ref(), apair), std::invalid_argument);
  EXPECT_THROW(builder.movb64(apair.ref(), 0x12345), std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
}

} // namespace
