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
#include <vector>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

using BuilderOp = void (IRBuilder::*)(RegisterRef, Operand, Operand);

constexpr int32_t literal = 0x12345678;
constexpr int32_t otherLiteral = 0x0FEDCBA9;

struct OpCase {
  const char *name;
  BuilderOp op;
};

constexpr OpCase opCases[] = {{"addi64", &IRBuilder::addi64},
                              {"subi64", &IRBuilder::subi64},
                              {"muli64", &IRBuilder::muli64}};

// The non-empty lines of text, without leading blanks.
std::vector<std::string> instructionsOf(const std::string &text) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string current; std::getline(stream, current);) {
    size_t first = current.find_first_not_of(" \t");
    if (first != std::string::npos)
      lines.push_back(current.substr(first));
  }
  return lines;
}

// Builds "mnemonic op0, op1, ...".
std::string inst(std::string_view mnemonic,
                 const std::vector<std::string> &operands) {
  std::string result(mnemonic);
  for (size_t i = 0; i < operands.size(); ++i)
    result += (i == 0 ? " " : ", ") + operands[i];
  return result;
}

// Checks that text holds exactly the expected instructions, so adjacency is
// checked too. A line may carry a trailing comment after its operands.
void expectInstructions(const std::string &text,
                        const std::vector<std::string> &expected) {
  std::vector<std::string> lines = instructionsOf(text);
  ASSERT_EQ(lines.size(), expected.size()) << text;
  for (size_t i = 0; i < expected.size(); ++i) {
    const std::string &actual = lines[i];
    bool matches = actual.rfind(expected[i], 0) == 0 &&
                   (actual.size() == expected[i].size() ||
                    actual[expected[i].size()] == ' ');
    EXPECT_TRUE(matches) << "line " << i << ": expected '" << expected[i]
                         << "', got '" << actual << "'";
  }
}

int instructionCount(const std::string &text) {
  return static_cast<int>(instructionsOf(text).size());
}

std::string half(RegisterRef pair, int index) {
  return text(pair.slice(index));
}

class Arith64Test : public BuilderTest {
protected:
  std::string emitOp(BuilderOp op, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { (local.*op)(dst, a, b); });
  }

  std::string emitAdd(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::addi64, dst, a, b);
  }
  std::string emitSub(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::subi64, dst, a, b);
  }
  std::string emitMul(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::muli64, dst, a, b);
  }

  RegisterRange sgprPair() { return builder.allocateSgpr(2, 2); }
  RegisterRange vgprPair() { return builder.allocateVgpr(2, 2); }
};

// The ops are built from 32-bit instructions, so no 64-bit form is emitted.
void expectNoWideForms(const std::string &out) {
  EXPECT_EQ(countLinesContaining(out, "_b64"), 0) << out;
}

TEST_F(Arith64Test, SaluCarryChainsAreAdjacent) {
  RegisterRange dstOwner = sgprPair();
  RegisterRange aOwner = sgprPair();
  RegisterRange bOwner = sgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  RegisterRef b = bOwner.ref();
  expectInstructions(
      emitAdd(d, a, b),
      {inst("s_add_u32", {half(d, 0), half(a, 0), half(b, 0)}),
       inst("s_addc_u32", {half(d, 1), half(a, 1), half(b, 1)})});
  expectInstructions(
      emitSub(d, a, b),
      {inst("s_sub_u32", {half(d, 0), half(a, 0), half(b, 0)}),
       inst("s_subb_u32", {half(d, 1), half(a, 1), half(b, 1)})});
}

TEST_F(Arith64Test, SaluMulSequence) {
  RegisterRange dstOwner = sgprPair();
  RegisterRange aOwner = sgprPair();
  RegisterRange bOwner = sgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  RegisterRef b = bOwner.ref();
  // The temporaries take the lowest free SGPRs.
  std::string high = "s6";
  std::string cross = "s7";
  expectInstructions(emitMul(d, a, b),
                     {inst("s_mul_hi_u32", {high, half(a, 0), half(b, 0)}),
                      inst("s_mul_i32", {cross, half(a, 0), half(b, 1)}),
                      inst("s_add_u32", {high, high, cross}),
                      inst("s_mul_i32", {cross, half(a, 1), half(b, 0)}),
                      inst("s_mul_i32", {half(d, 0), half(a, 0), half(b, 0)}),
                      inst("s_add_u32", {half(d, 1), high, cross})});
}

TEST_F(Arith64Test, ValuCarryChainsUseSgprPairCarryAndNop) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  RegisterRange bOwner = vgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  RegisterRef b = bOwner.ref();
  // The carry takes the lowest free SGPR pair, not VCC.
  std::string carry = "s[0:1]";
  std::string out = emitAdd(d, a, b);
  expectInstructions(
      out, {inst("v_add_co_u32", {half(d, 0), carry, half(a, 0), half(b, 0)}),
            "s_nop 1",
            inst("v_addc_co_u32",
                 {half(d, 1), carry, half(a, 1), half(b, 1), carry})});
  EXPECT_EQ(countLinesContaining(out, "vcc"), 0);

  out = emitSub(d, a, b);
  expectInstructions(
      out, {inst("v_sub_co_u32", {half(d, 0), carry, half(a, 0), half(b, 0)}),
            "s_nop 1",
            inst("v_subb_co_u32",
                 {half(d, 1), carry, half(a, 1), half(b, 1), carry})});
  EXPECT_EQ(countLinesContaining(out, "vcc"), 0);
}

TEST_F(Arith64Test, ValuMulSequence) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  RegisterRange bOwner = vgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  RegisterRef b = bOwner.ref();
  // The temporaries take the lowest free VGPRs.
  std::string high = "v6";
  std::string crossA = "v7";
  std::string crossB = "v8";
  expectInstructions(
      emitMul(d, a, b),
      {inst("v_mul_hi_u32", {high, half(a, 0), half(b, 0)}),
       inst("v_mul_lo_u32", {crossA, half(a, 0), half(b, 1)}),
       inst("v_mul_lo_u32", {crossB, half(a, 1), half(b, 0)}),
       inst("v_mul_lo_u32", {half(d, 0), half(a, 0), half(b, 0)}),
       inst("v_add3_u32", {half(d, 1), high, crossA, crossB})});
}

TEST_F(Arith64Test, ScalarOperandsIntoVgprAreLegalized) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange sa = sgprPair();
  RegisterRange sb = sgprPair();
  RegisterRange vb = vgprPair();
  RegisterRef d = dstOwner.ref();

  // An SGPR low half and a VGPR low half share the constant bus legally, but
  // the carry-in read already uses the bus, so a.hi is copied.
  std::string out = emitAdd(d, sa, vb);
  expectInstructions(
      out, {inst("v_mov_b32", {"v4", half(sa.ref(), 1)}),
            inst("v_add_co_u32",
                 {half(d, 0), "s[4:5]", half(sa.ref(), 0), half(vb.ref(), 0)}),
            "s_nop 1",
            inst("v_addc_co_u32",
                 {half(d, 1), "s[4:5]", "v4", half(vb.ref(), 1), "s[4:5]"})});

  // Two distinct SGPR pairs: b.lo for the low half, then a.hi and b.hi.
  out = emitAdd(d, sa, sb);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 3) << out;
  out = emitMul(d, sa, sb);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 2) << out;
}

TEST_F(Arith64Test, SameSgprPairInMulIsCopiedTwice) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange s = sgprPair();
  // The pair (a.lo, b.lo) is one SGPR and legal. (a.lo, b.hi) and
  // (a.hi, b.lo) are two distinct SGPRs, so b.hi and b.lo are copied.
  std::string out = emitMul(dstOwner.ref(), s, s);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 2) << out;
}

TEST_F(Arith64Test, ImmediatesAreSplitIntoHalves) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  std::string carry = "s[0:1]";

  // -5 is an inline constant and its high word is the inline constant -1.
  std::string out = emitAdd(d, a, -5);
  expectInstructions(
      out,
      {inst("v_add_co_u32", {half(d, 0), carry, half(a, 0), "-5"}), "s_nop 1",
       inst("v_addc_co_u32", {half(d, 1), carry, half(a, 1), "-1", carry})});
  expectNoWideForms(out);

  // The low word of -100 is a literal, which VOP3 cannot take.
  out = emitAdd(d, a, -100);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32 v4, -100"), 1) << out;
  EXPECT_EQ(countLinesContaining(out, ", -1, "), 1) << out;
  expectNoWideForms(out);

  // A positive literal has a high word of 0, which is inline.
  out = emitAdd(d, a, literal);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32 v4, 305419896"), 1) << out;
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
  EXPECT_EQ(countLinesContaining(out, ", 0, "), 1) << out;
  expectNoWideForms(out);
}

TEST_F(Arith64Test, SaluLiteralsUseOneLiteralPerInstruction) {
  RegisterRange dstOwner = sgprPair();
  RegisterRange aOwner = sgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();

  std::string out = emitAdd(d, a, literal);
  EXPECT_EQ(instructionCount(out), 2) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
  expectNoWideForms(out);

  // Distinct literals cannot share an instruction, so b's low half is copied;
  // the high halves are both the inline constant 0.
  out = emitAdd(d, literal, otherLiteral);
  EXPECT_EQ(instructionCount(out), 3) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1) << out;
  expectNoWideForms(out);

  out = emitAdd(d, literal, literal);
  EXPECT_EQ(instructionCount(out), 2) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
}

TEST_F(Arith64Test, ValuMulLiteralsAreCopied) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  std::string out = emitMul(dstOwner.ref(), aOwner, literal);
  // Only the low half is a literal; the high half is the inline constant 0.
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
  expectNoWideForms(out);
}

TEST_F(Arith64Test, DestinationAliasingNeedsNoCopies) {
  RegisterRange vd = vgprPair();
  RegisterRange vo = vgprPair();
  RegisterRange sd = sgprPair();
  RegisterRange so = sgprPair();
  for (const OpCase &current : {opCases[0], opCases[1]}) {
    SCOPED_TRACE(current.name);
    for (bool destinationFirst : {true, false}) {
      std::string valu = destinationFirst
                             ? emitOp(current.op, vd.ref(), vd, vo)
                             : emitOp(current.op, vd.ref(), vo, vd);
      EXPECT_EQ(instructionCount(valu), 3) << valu;
      EXPECT_EQ(countLinesContaining(valu, "v_mov_b32"), 0) << valu;
      std::string salu = destinationFirst
                             ? emitOp(current.op, sd.ref(), sd, so)
                             : emitOp(current.op, sd.ref(), so, sd);
      EXPECT_EQ(instructionCount(salu), 2) << salu;
      EXPECT_EQ(countLinesContaining(salu, "s_mov_b32"), 0) << salu;
    }
  }
}

TEST_F(Arith64Test, MulDestinationIsWrittenLast) {
  RegisterRange vd = vgprPair();
  RegisterRange vo = vgprPair();
  RegisterRef d = vd.ref();
  RegisterRef o = vo.ref();
  expectInstructions(
      emitMul(d, d, o),
      {inst("v_mul_hi_u32", {"v4", half(d, 0), half(o, 0)}),
       inst("v_mul_lo_u32", {"v5", half(d, 0), half(o, 1)}),
       inst("v_mul_lo_u32", {"v6", half(d, 1), half(o, 0)}),
       inst("v_mul_lo_u32", {half(d, 0), half(d, 0), half(o, 0)}),
       inst("v_add3_u32", {half(d, 1), "v4", "v5", "v6"})});

  RegisterRange sd = sgprPair();
  RegisterRange so = sgprPair();
  std::string out = emitMul(sd.ref(), sd, so);
  EXPECT_EQ(instructionCount(out), 6) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
}

TEST_F(Arith64Test, OverlappingHighHalfIsCopiedForSalu) {
  RegisterRange r = builder.allocateSgpr(3);
  RegisterRange other = sgprPair();
  RegisterRef a = r.ref(0, 2);
  RegisterRef d = r.ref(1, 2);
  RegisterRef b = other.ref();
  // dst.lo is a.hi, which the high instruction still reads.
  std::string copy = "s3";
  expectInstructions(emitAdd(d, a, b),
                     {inst("s_mov_b32", {copy, half(a, 1)}),
                      inst("s_add_u32", {half(d, 0), half(a, 0), half(b, 0)}),
                      inst("s_addc_u32", {half(d, 1), copy, half(b, 1)})});
  expectInstructions(emitSub(d, a, b),
                     {inst("s_mov_b32", {copy, half(a, 1)}),
                      inst("s_sub_u32", {half(d, 0), half(a, 0), half(b, 0)}),
                      inst("s_subb_u32", {half(d, 1), copy, half(b, 1)})});
  // The same for b.
  expectInstructions(emitAdd(d, b, a),
                     {inst("s_mov_b32", {copy, half(a, 1)}),
                      inst("s_add_u32", {half(d, 0), half(b, 0), half(a, 0)}),
                      inst("s_addc_u32", {half(d, 1), half(b, 1), copy})});
}

TEST_F(Arith64Test, OverlappingHighHalfIsCopiedForValu) {
  RegisterRange r = builder.allocateVgpr(3);
  RegisterRange other = vgprPair();
  RegisterRef a = r.ref(0, 2);
  RegisterRef d = r.ref(1, 2);
  RegisterRef b = other.ref();
  std::string copy = "v3";
  std::string carry = "s[0:1]";
  expectInstructions(
      emitAdd(d, a, b),
      {inst("v_mov_b32", {copy, half(a, 1)}),
       inst("v_add_co_u32", {half(d, 0), carry, half(a, 0), half(b, 0)}),
       "s_nop 1",
       inst("v_addc_co_u32", {half(d, 1), carry, copy, half(b, 1), carry})});
  expectInstructions(
      emitSub(d, a, b),
      {inst("v_mov_b32", {copy, half(a, 1)}),
       inst("v_sub_co_u32", {half(d, 0), carry, half(a, 0), half(b, 0)}),
       "s_nop 1",
       inst("v_subb_co_u32", {half(d, 1), carry, copy, half(b, 1), carry})});
}

TEST_F(Arith64Test, OverlapReadingOnlyLowHalfNeedsNoCopy) {
  RegisterRange vr = builder.allocateVgpr(3);
  RegisterRange vo = vgprPair();
  // dst.hi is a.lo, which only the low instruction reads.
  std::string out = emitAdd(vr.ref(0, 2), vr.ref(1, 2), vo);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0) << out;
  EXPECT_EQ(instructionCount(out), 3) << out;

  RegisterRange sr = builder.allocateSgpr(3);
  RegisterRange so = sgprPair();
  out = emitAdd(sr.ref(0, 2), sr.ref(1, 2), so);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
  EXPECT_EQ(instructionCount(out), 2) << out;
}

TEST_F(Arith64Test, MisalignedPairsNeedNoCopies) {
  RegisterRange vr = builder.allocateVgpr(3, 2);
  RegisterRange va = builder.allocateVgpr(3, 2);
  RegisterRange vb = builder.allocateVgpr(3, 2);
  RegisterRange sr = builder.allocateSgpr(3, 2);
  RegisterRange sa = builder.allocateSgpr(3, 2);
  RegisterRange sb = builder.allocateSgpr(3, 2);
  RegisterRef vd = vr.ref(1, 2);
  RegisterRef vx = va.ref(1, 2);
  RegisterRef vy = vb.ref(1, 2);
  RegisterRef sd = sr.ref(1, 2);
  RegisterRef sx = sa.ref(1, 2);
  RegisterRef sy = sb.ref(1, 2);

  std::string out = emitAdd(vd, vx, vy);
  EXPECT_EQ(instructionCount(out), 3) << out;
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0) << out;
  out = emitMul(vd, vx, vy);
  EXPECT_EQ(instructionCount(out), 5) << out;
  out = emitSub(sd, sx, sy);
  EXPECT_EQ(instructionCount(out), 2) << out;
  out = emitMul(sd, sx, sy);
  EXPECT_EQ(instructionCount(out), 6) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
  // An odd SGPR pair into a VGPR destination needs only the carry-in copy.
  out = emitAdd(vd, sx, vy);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
}

TEST_F(Arith64Test, TemporariesAreReleased) {
  RegisterRange dst = vgprPair();
  RegisterRange a = sgprPair();
  RegisterRange b = sgprPair();
  // Two copies of b's halves, then high, crossA and crossB.
  emitMul(dst.ref(), a, b);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 7);
  RegisterRange next = builder.allocateVgpr(1);
  EXPECT_EQ(next.start(), 2);
  next.release();

  // Three VGPR copies and the SGPR carry pair.
  emitAdd(dst.ref(), a, b);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::sgpr), 6);
  RegisterRange nextPair = builder.allocateSgpr(2, 2);
  EXPECT_EQ(nextPair.start(), 4);
}

TEST_F(Arith64Test, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = sgprPair();
  RegisterRange vdst = vgprPair();
  RegisterRange spair = sgprPair();
  RegisterRange vpair = vgprPair();
  RegisterRange acc = builder.allocateAgpr(2, 2);
  RegisterRange single = builder.allocateVgpr(1);
  RegisterRange triple = builder.allocateVgpr(3);
  for (const OpCase &current : opCases) {
    SCOPED_TRACE(current.name);
    BuilderOp op = current.op;
    EXPECT_THROW((builder.*op)(sdst.ref(), vpair, spair),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(sdst.ref(), spair, vpair),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(acc.ref(), spair, spair), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), acc, spair), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), spair, acc), std::invalid_argument);
    EXPECT_THROW((builder.*op)(single.ref(), vpair, spair),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), single, spair),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), spair, single),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(triple.ref(), vpair, spair),
                 std::invalid_argument);
  }
  EXPECT_EQ(body.toString(), "");
}

} // namespace
