// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Operand.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

using BuilderOp = void (IRBuilder::*)(RegisterRef, Operand, Operand);

constexpr int64_t literalA = 0x12345;
constexpr int64_t literalB = 0x54321;
constexpr const char *literalAText = "74565";

class ArithTest : public BuilderTest {
protected:
  std::string emitOp(BuilderOp op, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { (local.*op)(dst, a, b); });
  }

  std::string emitAdd(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::addi32, dst, a, b);
  }
  std::string emitSub(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::subi32, dst, a, b);
  }
  std::string emitMul(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::muli32, dst, a, b);
  }
};

struct OpCase {
  BuilderOp op;
  std::string_view valuMnemonic;
  std::string_view saluMnemonic;
};

constexpr OpCase opCases[] = {
    {&IRBuilder::addi32, "v_add_u32", "s_add_u32"},
    {&IRBuilder::subi32, "v_sub_u32", "s_sub_u32"},
    {&IRBuilder::muli32, "v_mul_lo_u32", "s_mul_i32"},
};

TEST_F(ArithTest, SaluForScalarOperands) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  for (const OpCase &current : opCases) {
    SCOPED_TRACE(current.saluMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.saluMnemonic, text(dst), text(a), text(b))});
    EXPECT_EQ(countLinesContaining(out, "v_"), 0);
  }
}

TEST_F(ArithTest, SaluTakesLiteralWithoutCopy) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  std::string out = emitAdd(dst.ref(), a, literalA);
  expectLinesInOrder(out,
                     {line("s_add_u32", text(dst), text(a), literalAText)});
  EXPECT_EQ(countLinesContaining(out, "mov"), 0);
}

TEST_F(ArithTest, ValuForVectorDestination) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateVgpr(1);
  RegisterRange b = builder.allocateVgpr(1);
  for (const OpCase &current : opCases) {
    SCOPED_TRACE(current.valuMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.valuMnemonic, text(dst), text(a), text(b))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
  }
}

TEST_F(ArithTest, TwoScalarsIntoVgprCopyOnce) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  // The temporary takes the lowest free VGPR.
  std::string temp = "v1";
  for (const OpCase &current : opCases) {
    SCOPED_TRACE(current.valuMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(out,
                       {line("v_mov_b32", temp, text(b)),
                        line(current.valuMnemonic, text(dst), text(a), temp)});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);
  }
}

TEST_F(ArithTest, SameScalarTwiceNeedsNoCopy) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string mul = emitMul(dst.ref(), s0, s0);
  expectLinesInOrder(mul,
                     {line("v_mul_lo_u32", text(dst), text(s0), text(s0))});
  EXPECT_EQ(countLinesContaining(mul, "v_mov_b32"), 0);
  std::string add = emitAdd(dst.ref(), s0, s0);
  expectLinesInOrder(add, {line("v_add_u32", text(dst), text(s0), text(s0))});
  EXPECT_EQ(countLinesContaining(add, "v_mov_b32"), 0);
}

TEST_F(ArithTest, LiteralIntoMulIsCopied) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string out = emitMul(dst.ref(), v1, literalA);
  expectLinesInOrder(out, {line("v_mov_b32", "v2", literalAText),
                           line("v_mul_lo_u32", text(dst), text(v1), "v2")});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);

  out = emitMul(dst.ref(), s0, literalA);
  expectLinesInOrder(out, {line("v_mov_b32", "v2", literalAText),
                           line("v_mul_lo_u32", text(dst), text(s0), "v2")});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);
}

TEST_F(ArithTest, InlineConstantsNeedNoCopy) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  struct Case {
    int64_t value;
    std::string printed;
    int copies;
  };
  for (const Case &current :
       {Case{64, "64", 0}, Case{-16, "-16", 0}, Case{0xFFFFFFFFu, "-1", 0},
        Case{65, "65", 1}, Case{-17, "-17", 1}}) {
    SCOPED_TRACE(current.printed);
    std::string out = emitMul(dst.ref(), v1, current.value);
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), current.copies);
    // The value is printed once, in the mul or in its copy.
    EXPECT_EQ(countLinesContaining(out, ", " + current.printed), 1);
  }
}

TEST_F(ArithTest, AddSwapsVgprIntoSrc1) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string out = emitAdd(dst.ref(), v1, s0);
  expectLinesInOrder(out, {line("v_add_u32", text(dst), text(s0), text(v1))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

  out = emitAdd(dst.ref(), v1, literalA);
  expectLinesInOrder(out,
                     {line("v_add_u32", text(dst), literalAText, text(v1))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
}

TEST_F(ArithTest, SubKeepsOperandOrder) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string out = emitSub(dst.ref(), s0, v1);
  expectLinesInOrder(out, {line("v_sub_u32", text(dst), text(s0), text(v1))});

  // The VOP3 form takes the SGPR in src1.
  out = emitSub(dst.ref(), v1, s0);
  expectLinesInOrder(out, {line("v_sub_u32", text(dst), text(v1), text(s0))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

  out = emitSub(dst.ref(), literalA, s0);
  expectLinesInOrder(out, {line("v_mov_b32", "v2", text(s0)),
                           line("v_sub_u32", text(dst), literalAText, "v2")});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);
}

TEST_F(ArithTest, SubImmediateBecomesAdd) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  std::string out = emitSub(dst.ref(), v1, 100);
  expectLinesInOrder(out, {line("v_add_u32", text(dst), "-100", text(v1))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

  out = emitSub(dst.ref(), v1, INT32_MIN);
  expectLinesInOrder(out,
                     {line("v_add_u32", text(dst), "-2147483648", text(v1))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
}

TEST_F(ArithTest, SaluTwoLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  std::string out = emitAdd(dst.ref(), literalA, literalB);
  expectLinesInOrder(out, {line("s_mov_b32", "s1", "344865"),
                           line("s_add_u32", text(dst), literalAText, "s1")});
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);

  EXPECT_EQ(
      countLinesContaining(emitAdd(dst.ref(), literalA, literalA), "s_mov_b32"),
      0);
  EXPECT_EQ(countLinesContaining(emitAdd(dst.ref(), 5, literalA), "s_mov_b32"),
            0);
}

TEST_F(ArithTest, DestinationMayAliasOperand) {
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string out = emitAdd(v1.ref(), v1, s0);
  expectLinesInOrder(out, {line("v_add_u32", text(v1), text(s0), text(v1))});

  out = emitMul(v1.ref(), v1, literalA);
  expectLinesInOrder(out, {line("v_mov_b32", "v1", literalAText),
                           line("v_mul_lo_u32", text(v1), text(v1), "v1")});
}

TEST_F(ArithTest, TemporariesAreReleased) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  emitAdd(dst.ref(), a, b);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 2);
  RegisterRange next = builder.allocateVgpr(1);
  EXPECT_EQ(next.start(), 1);
}

TEST_F(ArithTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange vdst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s1 = builder.allocateSgpr(1);
  RegisterRange acc = builder.allocateAgpr(1);
  RegisterRange pair = builder.allocateVgpr(2);
  EXPECT_THROW(builder.addi32(sdst.ref(), v1, s1), std::invalid_argument);
  EXPECT_THROW(builder.subi32(sdst.ref(), s1, v1), std::invalid_argument);
  EXPECT_THROW(builder.muli32(acc.ref(), s1, s1), std::invalid_argument);
  EXPECT_THROW(builder.addi32(vdst.ref(), acc, s1), std::invalid_argument);
  EXPECT_THROW(builder.addi32(vdst.ref(), s1, acc), std::invalid_argument);
  EXPECT_THROW(builder.addi32(pair.ref(), v1, s1), std::invalid_argument);
  EXPECT_THROW(builder.addi32(vdst.ref(), pair, s1), std::invalid_argument);
  EXPECT_THROW(builder.addi32(vdst.ref(), v1, int64_t{1} << 32),
               std::invalid_argument);
  EXPECT_THROW(builder.addi32(vdst.ref(), v1, int64_t{INT32_MIN} - 1),
               std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 4);
}

TEST_F(ArithTest, BoundaryImmediatesAreAccepted) {
  RegisterRange vdst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  EXPECT_NO_THROW(builder.addi32(vdst.ref(), v1, int64_t{UINT32_MAX}));
  EXPECT_NO_THROW(builder.addi32(vdst.ref(), v1, int64_t{INT32_MIN}));
}

TEST(OperandTest, ConstructsFromRegisters) {
  Target target = Target::get("gfx950");
  RegisterAllocator allocator(target);
  RegisterRange range = allocator.allocate(RegisterKind::vgpr, 2);
  Operand fromRange(range);
  EXPECT_TRUE(fromRange.isRegister());
  EXPECT_TRUE(fromRange.isRegister(RegisterKind::vgpr));
  EXPECT_FALSE(fromRange.isRegister(RegisterKind::sgpr));
  EXPECT_FALSE(fromRange.isImmediate());
  EXPECT_EQ(fromRange.reg(), range.ref());

  Operand fromRef(range.ref(1));
  EXPECT_EQ(fromRef.reg(), (RegisterRef{RegisterKind::vgpr, 1, 1}));
}

TEST(OperandTest, ConstructsFromIntegers) {
  Operand fromInt(5);
  EXPECT_TRUE(fromInt.isImmediate());
  EXPECT_FALSE(fromInt.isRegister());
  EXPECT_FALSE(fromInt.isRegister(RegisterKind::vgpr));
  EXPECT_EQ(fromInt.immediate(), 5);
  EXPECT_EQ(Operand(0xFFFFFFFFu).immediate(), 4294967295);
  EXPECT_EQ(Operand(INT32_MIN).immediate(), -2147483648LL);
  EXPECT_EQ(Operand(int64_t{1} << 40).immediate(), int64_t{1} << 40);
}

} // namespace
