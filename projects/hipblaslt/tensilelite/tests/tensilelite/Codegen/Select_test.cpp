// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/IRBuilder.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr int32_t literal32 = 0x12345678;
constexpr const char *literal32Text = "305419896";

// Builds "v_cndmask_b32 dst, src0, src1, mask" as rocisa prints it.
std::string cndmask(const std::string &dst, const std::string &src0,
                    const std::string &src1, const std::string &mask) {
  return line("v_cndmask_b32", dst, src0, src1) + ", " + mask;
}

class SelectTest : public BuilderTest {
protected:
  // Selects the width from dst.
  std::string emitSelect(RegisterRef dst, RegisterRef cond, Operand a,
                         Operand b) {
    return emitted([&](IRBuilder &local) {
      if (dst.count == 1)
        local.selectb32(dst, cond, a, b);
      else
        local.selectb64(dst, cond, a, b);
    });
  }

  // Asserts that op throws std::invalid_argument and that nothing was emitted
  // through the fixture's builder.
  void expectRejected(const std::function<void()> &op) {
    EXPECT_THROW(op(), std::invalid_argument);
    EXPECT_EQ(body.toString(), "");
  }
};

TEST_F(SelectTest, ScalarWidth1) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange cond = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  std::string out = emitSelect(dst.ref(), cond.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", text(dst), text(a), text(b))});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_"), 0);
}

TEST_F(SelectTest, ScalarWidth1TwoDistinctLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange cond = builder.allocateSgpr(1);
  std::string out = emitSelect(dst.ref(), cond.ref(), 0x12345, 0x54321);
  expectLinesInOrder(out, {line("s_mov_b32", "s2", "344865"),
                           line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", text(dst), "74565", "s2")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 1);
}

// Registers: padding s0, cond s1, dst s[2:3], a s[4:5], b s[6:7]; the first
// temporary pair is s[8:9].
class SelectScalar64Test : public SelectTest {
protected:
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange cond = builder.allocateSgpr(1);
  RegisterRange dst = builder.allocateSgpr(2, 2);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
};

TEST_F(SelectScalar64Test, AlignedPairs) {
  std::string out = emitSelect(dst.ref(), cond.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b64", text(dst), text(a), text(b))});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
}

TEST_F(SelectScalar64Test, EncodableImmediatesAreDirect) {
  std::string out = emitSelect(dst.ref(), cond.ref(), INT32_MAX, 64);
  expectLinesInOrder(out,
                     {line("s_cselect_b64", text(dst), "2147483647", "64")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
  out = emitSelect(dst.ref(), cond.ref(), -16, a);
  expectLinesInOrder(out, {line("s_cselect_b64", text(dst), "-16", text(a))});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
}

TEST_F(SelectScalar64Test, UnencodableImmediatesAreCopiedBeforeTheTest) {
  std::string out = emitSelect(dst.ref(), cond.ref(), -17, b);
  expectLinesInOrder(
      out, {line("s_mov_b32", "s8", "-17"), line("s_mov_b32", "s9", "-1"),
            line("s_cmp_lg_u32", text(cond), "0"),
            line("s_cselect_b64", text(dst), "s[8:9]", text(b))});
  out = emitSelect(dst.ref(), cond.ref(), a, INT32_MIN);
  expectLinesInOrder(out,
                     {line("s_mov_b32", "s8", "-2147483648"),
                      line("s_mov_b32", "s9", "-1"),
                      line("s_cmp_lg_u32", text(cond), "0"),
                      line("s_cselect_b64", text(dst), text(a), "s[8:9]")});
  out = emitSelect(dst.ref(), cond.ref(), -100, b);
  expectLinesInOrder(
      out, {line("s_mov_b32", "s8", "-100"), line("s_mov_b32", "s9", "-1"),
            line("s_cmp_lg_u32", text(cond), "0"),
            line("s_cselect_b64", text(dst), "s[8:9]", text(b))});
}

TEST_F(SelectScalar64Test, MisalignedOperandPairIsCopied) {
  RegisterRange odd = builder.allocateSgpr(4, 2);
  std::string out = emitSelect(dst.ref(), cond.ref(), odd.ref(1, 2), b);
  expectLinesInOrder(
      out, {line("s_mov_b32", "s13", "s10"), line("s_mov_b32", "s12", "s9"),
            line("s_cmp_lg_u32", text(cond), "0"),
            line("s_cselect_b64", text(dst), "s[12:13]", text(b))});
}

TEST_F(SelectScalar64Test, TwoDistinctLiteralsCopyOne) {
  std::string out = emitSelect(dst.ref(), cond.ref(), 0x12345, 0x54321);
  expectLinesInOrder(out,
                     {line("s_mov_b64", "s[8:9]", "344865"),
                      line("s_cmp_lg_u32", text(cond), "0"),
                      line("s_cselect_b64", text(dst), "74565", "s[8:9]")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 1);
}

// Registers: cond s0, range s[2:6] holding the misaligned dst s[3:4].
class SelectSplitTest : public SelectTest {
protected:
  RegisterRange cond = builder.allocateSgpr(1);
  RegisterRange range = builder.allocateSgpr(5, 2);
  RegisterRef dst = range.ref(1, 2);
};

TEST_F(SelectSplitTest, SeparateOperandsGoLowFirst) {
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
  std::string out = emitSelect(dst, cond.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", "s3", "s8", "s10"),
                           line("s_cselect_b32", "s4", "s9", "s11")});
}

TEST_F(SelectSplitTest, OverlapThatForcesHighFirst) {
  RegisterRange b = builder.allocateSgpr(2, 2);
  // a = s[2:3]: its high word is the destination's low register.
  std::string out = emitSelect(dst, cond.ref(), range.ref(0, 2), b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", "s4", "s3", "s9"),
                           line("s_cselect_b32", "s3", "s2", "s8")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
}

TEST_F(SelectSplitTest, OverlapInBothDirectionsCopiesOneWord) {
  // a = s[2:3] and b = s[4:5] around dst = s[3:4].
  std::string out =
      emitSelect(dst, cond.ref(), range.ref(0, 2), range.ref(2, 2));
  expectLinesInOrder(out, {line("s_mov_b32", "s1", "s3"),
                           line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", "s3", "s2", "s4"),
                           line("s_cselect_b32", "s4", "s1", "s5")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 1);
}

TEST_F(SelectSplitTest, ImmediateHalvesWithTwoLiteralsCopyOne) {
  RegisterRange b = builder.allocateSgpr(2, 2);
  // The low words are two distinct literals; the high words are the inline 0.
  std::string out = emitSelect(dst, cond.ref(), 0x12345678, 0x76543210);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  EXPECT_EQ(countLinesContaining(out, "s_cselect_b32"), 2);
}

// Registers: mask s[0:1], dst v0, a v1, b v2.
class SelectVectorTest : public SelectTest {
protected:
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateVgpr(1);
  RegisterRange b = builder.allocateVgpr(1);
};

TEST_F(SelectVectorTest, Width1TakesBInSrc0) {
  std::string out = emitSelect(dst.ref(), mask.ref(), a, b);
  expectLinesInOrder(out, {cndmask(text(dst), text(b), text(a), text(mask))});
  EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(SelectVectorTest, Width1CopiesSgprsAndLiterals) {
  RegisterRange s = builder.allocateSgpr(1);
  std::string out = emitSelect(dst.ref(), mask.ref(), s, literal32);
  expectLinesInOrder(out, {line("v_mov_b32", "v3", text(s)),
                           line("v_mov_b32", "v4", literal32Text),
                           cndmask(text(dst), "v4", "v3", text(mask))});
}

TEST_F(SelectVectorTest, Width1InlineImmediatesAreDirect) {
  std::string out = emitSelect(dst.ref(), mask.ref(), 5, -16);
  expectLinesInOrder(out, {cndmask(text(dst), "-16", "5", text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  out = emitSelect(dst.ref(), mask.ref(), 64, a);
  expectLinesInOrder(out, {cndmask(text(dst), text(a), "64", text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(SelectVectorTest, UniformBoolIsWidenedToAMask) {
  RegisterRange cond = builder.allocateSgpr(1);
  // The temporary mask takes the lowest free even SGPR pair, s[4:5].
  std::string out = emitSelect(dst.ref(), cond.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b64", "s[4:5]", "-1", "0"),
                           cndmask(text(dst), text(b), text(a), "s[4:5]")});
  EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
}

TEST_F(SelectVectorTest, Width2TakesHalvesInOrder) {
  RegisterRange pairDst = builder.allocateVgpr(2, 2);
  RegisterRange pairA = builder.allocateVgpr(2, 2);
  RegisterRange pairB = builder.allocateVgpr(2, 2);
  std::string out = emitSelect(pairDst.ref(), mask.ref(), pairA, pairB);
  expectLinesInOrder(out, {cndmask("v4", "v8", "v6", text(mask)),
                           cndmask("v5", "v9", "v7", text(mask))});
}

TEST_F(SelectVectorTest, Width2SplitsAnImmediate) {
  RegisterRange pairDst = builder.allocateVgpr(2, 2);
  RegisterRange pairB = builder.allocateVgpr(2, 2);
  // INT32_MIN is the literal 0x80000000 and the inline -1.
  std::string out = emitSelect(pairDst.ref(), mask.ref(), INT32_MIN, pairB);
  expectLinesInOrder(out, {line("v_mov_b32", "v3", "-2147483648"),
                           cndmask("v4", "v6", "v3", text(mask)),
                           cndmask("v5", "v7", "-1", text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 1);
}

TEST_F(SelectVectorTest, Width2OverlapForcesHighFirst) {
  RegisterRange range = builder.allocateVgpr(4, 2);
  RegisterRange pairB = builder.allocateVgpr(2, 2);
  RegisterRef dstPair = range.ref(1, 2);
  RegisterRef pairA = range.ref(0, 2);
  // a's high word is the destination's low register, so the high half goes
  // first.
  std::string out = emitSelect(dstPair, mask.ref(), pairA, pairB);
  expectLinesInOrder(out, {cndmask(text(dstPair.slice(1)), text(pairB.ref(1)),
                                   text(pairA.slice(1)), text(mask)),
                           cndmask(text(dstPair.slice(0)), text(pairB.ref(0)),
                                   text(pairA.slice(0)), text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(SelectVectorTest, Width2OverlapInBothDirectionsCopiesOneWord) {
  // dst = range[1:2] with a = range[0:1] and b = range[2:3].
  RegisterRange range = builder.allocateVgpr(4, 2);
  RegisterRef dstPair = range.ref(1, 2);
  RegisterRef pairA = range.ref(0, 2);
  RegisterRef pairB = range.ref(2, 2);
  std::string out = emitSelect(dstPair, mask.ref(), pairA, pairB);
  // The temporary takes the free hole at v3.
  expectLinesInOrder(out, {line("v_mov_b32", "v3", text(pairA.slice(1))),
                           cndmask(text(dstPair.slice(0)), text(pairB.slice(0)),
                                   text(pairA.slice(0)), text(mask)),
                           cndmask(text(dstPair.slice(1)), text(pairB.slice(1)),
                                   "v3", text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 1);
}

TEST_F(SelectVectorTest, TemporariesAreReleased) {
  RegisterRange s = builder.allocateSgpr(1);
  emitSelect(dst.ref(), s.ref(), s, literal32);
  RegisterRange next = builder.allocateVgpr(1);
  EXPECT_EQ(next.start(), 3);
}

TEST_F(SelectVectorTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange acc = builder.allocateAgpr(2, 2);
  RegisterRange three = builder.allocateSgpr(3, 2);
  RegisterRange oddMask = builder.allocateSgpr(4, 2);
  RegisterRange pair = builder.allocateVgpr(2, 2);
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange sdst2 = builder.allocateSgpr(2, 2);

  std::string message;
  try {
    builder.selectb32(dst.ref(), a.ref(), s, s);
  } catch (const std::invalid_argument &error) {
    message = error.what();
  }
  EXPECT_EQ(message,
            "selectb32 condition must be an sgpr or an sgpr pair at an "
            "even index, got v1");
  expectRejected([&] { builder.selectb32(dst.ref(), acc.ref(0, 1), s, s); });
  expectRejected([&] { builder.selectb32(acc.ref(0, 1), s.ref(), s, s); });
  expectRejected(
      [&] { builder.selectb32(dst.ref(), s.ref(), acc.ref(0, 1), s); });
  expectRejected([&] { builder.selectb32(sdst.ref(), mask.ref(), s, s); });
  expectRejected([&] { builder.selectb32(sdst.ref(), s.ref(), a, s); });
  expectRejected([&] { builder.selectb64(sdst2.ref(), s.ref(), pair, sdst2); });
  expectRejected(
      [&] { builder.selectb32(dst.ref(), oddMask.ref(1, 2), a, b); });
  expectRejected([&] { builder.selectb32(dst.ref(), three.ref(), a, b); });
  expectRejected([&] {
    builder.selectb32(RegisterRef{RegisterKind::vgpr, 20, 3}, s.ref(), 5, 6);
  });
  expectRejected([&] { builder.selectb32(dst.ref(), s.ref(), pair, b); });
  expectRejected([&] { builder.selectb64(pair.ref(), s.ref(), a, b); });
  expectRejected([&] { builder.selectb64(dst.ref(), s.ref(), a, b); });
}

TEST_F(SelectVectorTest, BoundaryImmediatesAreAccepted) {
  RegisterRange pair = builder.allocateVgpr(2, 2);
  EXPECT_NO_THROW(builder.selectb32(dst.ref(), mask.ref(), -1, INT32_MIN));
  EXPECT_NO_THROW(
      builder.selectb64(pair.ref(), mask.ref(), INT32_MIN, INT32_MAX));
}

} // namespace
