// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/IRBuilder.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr int32_t literalA = 0x12345;
constexpr int32_t literalB = 0x54321;
constexpr const char *literalAText = "74565";
constexpr const char *literalBText = "344865";

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

std::string mnemonicOf(const std::string &instruction) {
  return instruction.substr(0, instruction.find(' '));
}

// The operands of an instruction, without a trailing comment, split at commas.
std::vector<std::string> operandsOf(const std::string &instruction) {
  std::vector<std::string> operands;
  size_t space = instruction.find(' ');
  if (space == std::string::npos)
    return operands;
  std::string rest = instruction.substr(space + 1);
  rest = rest.substr(0, rest.find("//"));
  std::istringstream stream(rest);
  for (std::string operand; std::getline(stream, operand, ',');) {
    size_t first = operand.find_first_not_of(" \t");
    size_t last = operand.find_last_not_of(" \t");
    if (first != std::string::npos)
      operands.push_back(operand.substr(first, last - first + 1));
  }
  return operands;
}

// The first operand after the mnemonic, or empty for instructions with no
// destination.
std::string destinationOf(const std::string &instruction) {
  std::string mnemonic = mnemonicOf(instruction);
  if (mnemonic == "s_nop" || mnemonic.rfind("s_cmp", 0) == 0)
    return "";
  std::vector<std::string> operands = operandsOf(instruction);
  return operands.empty() ? "" : operands.front();
}

// The source operands without negation or absolute-value modifiers.
std::vector<std::string> sourcesOf(const std::string &instruction) {
  std::vector<std::string> sources = operandsOf(instruction);
  if (!sources.empty())
    sources.erase(sources.begin());
  for (std::string &source : sources) {
    if (source.rfind("-", 0) == 0)
      source = source.substr(1);
    if (source.rfind("abs(", 0) == 0)
      source = source.substr(4, source.size() - 5);
  }
  return sources;
}

int occurrences(const std::string &text, std::string_view needle) {
  int count = 0;
  for (size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size()))
    ++count;
  return count;
}

int indexOfMnemonic(const std::vector<std::string> &instructions,
                    std::string_view mnemonic, int from = 0) {
  for (int i = from; i < static_cast<int>(instructions.size()); ++i)
    if (mnemonicOf(instructions[i]) == mnemonic)
      return i;
  return -1;
}

enum class Form { scalar, vector, mixed };

constexpr Form allForms[] = {Form::scalar, Form::vector, Form::mixed};

const char *formName(Form form) {
  switch (form) {
  case Form::scalar:
    return "sgpr dst with sgpr operands";
  case Form::vector:
    return "vgpr dst with vgpr operands";
  case Form::mixed:
    return "vgpr dst with sgpr operands";
  }
  return "";
}

struct Registers {
  RegisterRange dst;
  RegisterRange a;
  RegisterRange b;
};

class DivTest : public BuilderTest {
protected:
  std::string emitDiv(RegisterRef dst, Operand a, Operand b, bool useIEEE) {
    return emitted(
        [&](IRBuilder &local) { local.divsi32(dst, a, b, useIEEE); });
  }

  RegisterRange allocate(RegisterKind kind) {
    return kind == RegisterKind::sgpr ? builder.allocateSgpr(1)
                                      : builder.allocateVgpr(1);
  }

  Registers makeRegisters(Form form) {
    RegisterKind dstKind =
        form == Form::scalar ? RegisterKind::sgpr : RegisterKind::vgpr;
    RegisterKind sourceKind =
        form == Form::vector ? RegisterKind::vgpr : RegisterKind::sgpr;
    return {allocate(dstKind), allocate(sourceKind), allocate(sourceKind)};
  }

  std::string emitForm(Form form, bool useIEEE) {
    Registers regs = makeRegisters(form);
    return emitDiv(regs.dst.ref(), regs.a, regs.b, useIEEE);
  }

  // Asserts that op throws std::invalid_argument and that nothing was emitted
  // through the fixture's builder.
  void expectRejected(const std::function<void()> &op) {
    EXPECT_THROW(op(), std::invalid_argument);
    EXPECT_EQ(body.toString(), "");
  }

  int nextFreeIndex(RegisterKind kind) { return allocate(kind).start(); }
};

TEST_F(DivTest, ExactSaluForScalarOperands) {
  std::string out = emitForm(Form::scalar, true);
  expectLinesInOrder(out,
                     {"s_abs_i32",     "s_abs_i32",           "s_xor_b32",
                      "s_ashr_i32",    "v_cvt_f32_u32",       "v_rcp_iflag_f32",
                      "s_sub_u32",     "v_mul_f32",           "v_cvt_u32_f32",
                      "s_nop 0",       "v_readfirstlane_b32", "s_mul_i32",
                      "s_mul_hi_u32",  "s_add_u32",           "s_mul_hi_u32",
                      "s_mul_i32",     "s_sub_u32",           "s_add_u32",
                      "s_sub_u32",     "s_cmp_ge_u32",        "s_cselect_b32",
                      "s_cselect_b32", "s_add_u32",           "s_cmp_ge_u32",
                      "s_cselect_b32", "s_xor_b32",           "s_sub_u32"});
  EXPECT_EQ(countLinesContaining(out, "s_ashr_i32"), 1);
  EXPECT_EQ(countLinesContaining(out, ", 31"), 1);
  EXPECT_EQ(countLinesContaining(out, "0x4f7ffffe"), 1);
  EXPECT_EQ(countLinesContaining(out, "v_mul_lo"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_mul_hi"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_cndmask"), 0);
  std::vector<std::string> instructions = instructionsOf(out);
  EXPECT_EQ(mnemonicOf(instructions.back()), "s_sub_u32");
}

TEST_F(DivTest, ExactValuForVectorDestination) {
  for (Form form : {Form::vector, Form::mixed}) {
    SCOPED_TRACE(formName(form));
    std::string out = emitForm(form, true);
    expectLinesInOrder(
        out,
        {"v_sub_u32",    "v_max_i32",     "v_sub_u32",     "v_max_i32",
         "v_xor_b32",    "v_ashrrev_i32", "v_cvt_f32_u32", "v_rcp_iflag_f32",
         "v_sub_u32",    "v_mul_f32",     "v_cvt_u32_f32", "v_mul_lo_u32",
         "v_mul_hi_u32", "v_add_u32",     "v_mul_hi_u32",  "v_mul_lo_u32",
         "v_sub_u32",    "v_add_u32",     "v_sub_u32",     "v_cmp_ge_u32",
         "s_nop 1",      "v_cndmask_b32", "v_cndmask_b32", "v_add_u32",
         "v_cmp_ge_u32", "s_nop 1",       "v_cndmask_b32", "v_xor_b32",
         "v_sub_u32"});
    EXPECT_EQ(countLinesContaining(out, "v_ashrrev_i32"), 1);
    EXPECT_EQ(countLinesContaining(out, ", 31"), 1);
    EXPECT_EQ(countLinesContaining(out, "v_readfirstlane"), 0);
    for (const std::string &instruction : instructionsOf(out)) {
      std::string mnemonic = mnemonicOf(instruction);
      bool scalar = mnemonic.rfind("s_", 0) == 0;
      EXPECT_TRUE(!scalar || mnemonic == "s_nop") << instruction;
    }
    std::vector<std::string> instructions = instructionsOf(out);
    EXPECT_EQ(mnemonicOf(instructions.back()), "v_sub_u32");
  }
}

TEST_F(DivTest, FastSaluForScalarOperands) {
  std::string out = emitForm(Form::scalar, false);
  expectLinesInOrder(out, {"s_xor_b32", "s_ashr_i32", "s_or_b32",
                           "v_cvt_f32_i32", "v_rcp_iflag_f32", "v_cvt_f32_i32",
                           "v_mul_f32", "v_cvt_i32_f32", "v_cvt_f32_i32",
                           "v_fma_f32", "v_cmp_ge_f32", "v_readfirstlane_b32",
                           "s_and_b64", "s_cselect_b32", "s_add_u32"});
  EXPECT_EQ(countLinesContaining(out, "s_ashr_i32"), 1);
  EXPECT_EQ(countLinesContaining(out, ", 30"), 1);
  EXPECT_EQ(countLinesContaining(out, "v_mul_hi"), 0);
  for (const std::string &instruction : instructionsOf(out)) {
    if (mnemonicOf(instruction) == "v_cmp_ge_f32")
      EXPECT_EQ(occurrences(instruction, "abs("), 2) << instruction;
    if (mnemonicOf(instruction) == "v_fma_f32")
      EXPECT_NE(instruction.find(", -v"), std::string::npos) << instruction;
    if (mnemonicOf(instruction) == "s_and_b64")
      EXPECT_NE(instruction.find("exec"), std::string::npos) << instruction;
  }
  EXPECT_EQ(countLinesContaining(out, "abs("), 1);
}

TEST_F(DivTest, FastValuForVectorDestination) {
  for (Form form : {Form::vector, Form::mixed}) {
    SCOPED_TRACE(formName(form));
    std::string out = emitForm(form, false);
    expectLinesInOrder(out, {"v_xor_b32", "v_ashrrev_i32", "v_or_b32",
                             "v_cvt_f32_i32", "v_rcp_iflag_f32",
                             "v_cvt_f32_i32", "v_mul_f32", "v_cvt_i32_f32",
                             "v_cvt_f32_i32", "v_fma_f32", "v_cmp_ge_f32",
                             "s_nop 1", "v_cndmask_b32", "v_add_u32"});
    EXPECT_EQ(countLinesContaining(out, ", 30"), 1);
    EXPECT_EQ(countLinesContaining(out, "v_mul_hi"), 0);
    EXPECT_EQ(countLinesContaining(out, "abs("), 1);
    EXPECT_EQ(countLinesContaining(out, "v_readfirstlane"), 0);
    std::vector<std::string> instructions = instructionsOf(out);
    int fma = indexOfMnemonic(instructions, "v_fma_f32");
    ASSERT_GE(fma, 0);
    EXPECT_NE(instructions[fma].find(", -v"), std::string::npos);
  }
}

TEST_F(DivTest, NeverWritesExecOrVcc) {
  for (bool useIEEE : {true, false})
    for (Form form : allForms) {
      SCOPED_TRACE(std::string(formName(form)) +
                   (useIEEE ? " exact" : " fast"));
      std::string out = emitForm(form, useIEEE);
      for (const std::string &instruction : instructionsOf(out)) {
        std::string dst = destinationOf(instruction);
        EXPECT_NE(dst, "exec") << instruction;
        EXPECT_NE(dst, "exec_lo") << instruction;
        EXPECT_NE(dst, "vcc") << instruction;
      }
      EXPECT_EQ(countLinesContaining(out, "v_cmpx"), 0);
      EXPECT_EQ(countLinesContaining(out, "vcc"), 0);
    }
}

// The reciprocal's reader must not be the next instruction, since a trans
// result needs a wait state.
void expectReciprocalHazardCovered(
    const std::vector<std::string> &instructions) {
  int rcp = indexOfMnemonic(instructions, "v_rcp_iflag_f32");
  ASSERT_GE(rcp, 0);
  ASSERT_LT(rcp + 1, static_cast<int>(instructions.size()));
  const std::string &next = instructions[rcp + 1];
  EXPECT_NE(mnemonicOf(next), "s_nop") << next;
  std::string reciprocal = destinationOf(instructions[rcp]);
  for (const std::string &source : sourcesOf(next))
    EXPECT_NE(source, reciprocal) << next;
}

void expectMaskNopBeforeCndmask(const std::vector<std::string> &instructions) {
  int checked = 0;
  for (int i = 0; i < static_cast<int>(instructions.size()); ++i) {
    if (mnemonicOf(instructions[i]).rfind("v_cmp_", 0) != 0)
      continue;
    ASSERT_LT(i + 1, static_cast<int>(instructions.size()));
    EXPECT_EQ(instructions[i + 1].rfind("s_nop 1", 0), 0u) << instructions[i];
    EXPECT_GE(indexOfMnemonic(instructions, "v_cndmask_b32", i + 1), 0);
    ++checked;
  }
  EXPECT_GT(checked, 0);
}

TEST_F(DivTest, HazardWaitStates) {
  for (bool useIEEE : {true, false})
    for (Form form : allForms) {
      SCOPED_TRACE(std::string(formName(form)) +
                   (useIEEE ? " exact" : " fast"));
      std::vector<std::string> instructions =
          instructionsOf(emitForm(form, useIEEE));
      expectReciprocalHazardCovered(instructions);
      if (form != Form::scalar) {
        expectMaskNopBeforeCndmask(instructions);
        continue;
      }
      int lane = indexOfMnemonic(instructions, "v_readfirstlane_b32");
      ASSERT_GT(lane, 0);
      if (useIEEE) {
        EXPECT_EQ(instructions[lane - 1].rfind("s_nop 0", 0), 0u);
        continue;
      }
      int convert = indexOfMnemonic(instructions, "v_cvt_i32_f32");
      ASSERT_GE(convert, 0);
      EXPECT_GT(lane - convert, 1);
      EXPECT_EQ(sourcesOf(instructions[lane]).front(),
                destinationOf(instructions[convert]));
    }
}

// Every VOP3-only or VOP3-promoted line must be free of the literal.
void expectLiteralOnlyInMoves(const std::string &out, const char *literal) {
  for (const std::string &instruction : instructionsOf(out)) {
    std::string mnemonic = mnemonicOf(instruction);
    if (mnemonic == "v_mul_lo_u32" || mnemonic == "v_mul_hi_u32" ||
        mnemonic == "v_sub_u32" || mnemonic == "v_xor_b32")
      EXPECT_EQ(instruction.find(literal), std::string::npos) << instruction;
  }
}

TEST_F(DivTest, ImmediateOperandsInVectorForm) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange value = builder.allocateVgpr(1);
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(useIEEE ? "exact" : "fast");
    std::string divisor = emitDiv(dst.ref(), value, literalA, useIEEE);
    EXPECT_EQ(countLinesContaining(divisor, "v_mov_b32"), 1);
    expectLiteralOnlyInMoves(divisor, literalAText);
    std::string dividend = emitDiv(dst.ref(), literalB, value, useIEEE);
    EXPECT_EQ(countLinesContaining(dividend, "v_mov_b32"), 1);
    expectLiteralOnlyInMoves(dividend, literalBText);
    for (int32_t inlineValue : {7, -5})
      EXPECT_EQ(countLinesContaining(
                    emitDiv(dst.ref(), value, inlineValue, useIEEE), "v_mov"),
                0)
          << inlineValue;
  }
}

TEST_F(DivTest, ImmediateOperandsInExactScalarForm) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange value = builder.allocateSgpr(1);
  std::string out = emitDiv(dst.ref(), value, 7, true);
  expectLinesInOrder(out, {"s_mov_b32", "s_abs_i32"});
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  out = emitDiv(dst.ref(), 7, value, true);
  expectLinesInOrder(out, {"s_mov_b32", "s_abs_i32"});
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  EXPECT_EQ(
      countLinesContaining(emitDiv(dst.ref(), literalA, -5, true), "s_mov_b32"),
      2);
}

TEST_F(DivTest, ImmediateOperandsInFastScalarForm) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange value = builder.allocateSgpr(1);
  std::string out = emitDiv(dst.ref(), value, literalA, false);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0);
  int convertsWithLiteral = 0;
  int xorsWithLiteral = 0;
  for (const std::string &instruction : instructionsOf(out)) {
    bool hasLiteral = instruction.find(literalAText) != std::string::npos;
    if (mnemonicOf(instruction) == "v_cvt_f32_i32" && hasLiteral)
      ++convertsWithLiteral;
    if (mnemonicOf(instruction) == "s_xor_b32" && hasLiteral)
      ++xorsWithLiteral;
  }
  EXPECT_EQ(convertsWithLiteral, 1);
  EXPECT_EQ(xorsWithLiteral, 1);
  // Two distinct literals cannot share s_xor, so b is copied.
  out = emitDiv(dst.ref(), literalA, literalB, false);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  std::vector<std::string> instructions = instructionsOf(out);
  int copy = indexOfMnemonic(instructions, "s_mov_b32");
  ASSERT_GE(copy, 0);
  EXPECT_NE(instructions[copy].find(literalBText), std::string::npos);
}

TEST_F(DivTest, DestinationAliasesOperand) {
  for (bool useIEEE : {true, false})
    for (Form form : {Form::scalar, Form::vector}) {
      for (bool aliasFirst : {true, false}) {
        SCOPED_TRACE(std::string(formName(form)) +
                     (useIEEE ? " exact" : " fast") +
                     (aliasFirst ? " dst is a" : " dst is b"));
        Registers regs = makeRegisters(form);
        RegisterRef dst = aliasFirst ? regs.a.ref() : regs.b.ref();
        std::vector<std::string> instructions =
            instructionsOf(emitDiv(dst, regs.a, regs.b, useIEEE));
        ASSERT_FALSE(instructions.empty());
        EXPECT_EQ(destinationOf(instructions.back()), text(dst));
        for (size_t i = 0; i + 1 < instructions.size(); ++i)
          EXPECT_NE(destinationOf(instructions[i]), text(dst))
              << instructions[i];
      }
    }
}

TEST_F(DivTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange vdst = builder.allocateVgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  RegisterRange vb = builder.allocateVgpr(1);
  RegisterRange acc = builder.allocateAgpr(1);
  RegisterRange pair = builder.allocateVgpr(2, 2);
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(useIEEE ? "exact" : "fast");
    expectRejected([&] { builder.divsi32(acc.ref(), va, vb, useIEEE); });
    expectRejected([&] { builder.divsi32(vdst.ref(), acc, vb, useIEEE); });
    expectRejected([&] { builder.divsi32(vdst.ref(), va, acc, useIEEE); });
    expectRejected([&] { builder.divsi32(pair.ref(), va, vb, useIEEE); });
    expectRejected([&] { builder.divsi32(vdst.ref(), pair, vb, useIEEE); });
    expectRejected([&] { builder.divsi32(vdst.ref(), va, pair, useIEEE); });
    expectRejected([&] { builder.divsi32(sdst.ref(), va, sa, useIEEE); });
    expectRejected([&] { builder.divsi32(sdst.ref(), sa, vb, useIEEE); });
  }
}

TEST_F(DivTest, ScalarDestinationFromVectorOperandMessage) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange vb = builder.allocateVgpr(1);
  std::string expected = "divsi32 cannot write sgpr destination " + text(sdst) +
                         " from vgpr operand " + text(vb);
  try {
    builder.divsi32(sdst.ref(), sa, vb, true);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &error) {
    EXPECT_EQ(std::string(error.what()), expected);
  }
}

TEST_F(DivTest, ReleasesTemporaries) {
  for (bool useIEEE : {true, false})
    for (Form form : allForms) {
      SCOPED_TRACE(std::string(formName(form)) +
                   (useIEEE ? " exact" : " fast"));
      Registers regs = makeRegisters(form);
      int vgprsBefore = nextFreeIndex(RegisterKind::vgpr);
      int sgprsBefore = nextFreeIndex(RegisterKind::sgpr);
      emitDiv(regs.dst.ref(), regs.a, regs.b, useIEEE);
      EXPECT_EQ(nextFreeIndex(RegisterKind::vgpr), vgprsBefore);
      EXPECT_EQ(nextFreeIndex(RegisterKind::sgpr), sgprsBefore);
      // Literals take temporaries too.
      emitDiv(regs.dst.ref(), literalA, literalB, useIEEE);
      EXPECT_EQ(nextFreeIndex(RegisterKind::vgpr), vgprsBefore);
      EXPECT_EQ(nextFreeIndex(RegisterKind::sgpr), sgprsBefore);
    }
}

} // namespace
