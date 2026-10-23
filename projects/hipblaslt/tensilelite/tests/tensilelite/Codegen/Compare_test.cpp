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
#include <string_view>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr int32_t literal32 = 0x12345678;
constexpr const char *literal32Text = "305419896";
constexpr const char *signBitText = "-2147483648";

struct PredicateCase {
  Predicate pred;
  std::string_view name;
  std::string_view scalar32;
  std::string_view vector32;
  std::string_view vector64;
};

constexpr PredicateCase predicateCases[] = {
    {Predicate::eq, "eq", "s_cmp_eq_u32", "v_cmp_eq_u32", "v_cmp_eq_u64"},
    {Predicate::ne, "ne", "s_cmp_lg_u32", "v_cmp_ne_u32", "v_cmp_ne_u64"},
    {Predicate::slt, "slt", "s_cmp_lt_i32", "v_cmp_lt_i32", "v_cmp_lt_i64"},
    {Predicate::sle, "sle", "s_cmp_le_i32", "v_cmp_le_i32", "v_cmp_le_i64"},
    {Predicate::sgt, "sgt", "s_cmp_gt_i32", "v_cmp_gt_i32", "v_cmp_gt_i64"},
    {Predicate::sge, "sge", "s_cmp_ge_i32", "v_cmp_ge_i32", "v_cmp_ge_i64"},
    {Predicate::ult, "ult", "s_cmp_lt_u32", "v_cmp_lt_u32", "v_cmp_lt_u64"},
    {Predicate::ule, "ule", "s_cmp_le_u32", "v_cmp_le_u32", "v_cmp_le_u64"},
    {Predicate::ugt, "ugt", "s_cmp_gt_u32", "v_cmp_gt_u32", "v_cmp_gt_u64"},
    {Predicate::uge, "uge", "s_cmp_ge_u32", "v_cmp_ge_u32", "v_cmp_ge_u64"},
};

class CmpTest : public BuilderTest {
protected:
  std::string emit32(Predicate pred, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { local.cmpi32(pred, dst, a, b); });
  }
  std::string emit64(Predicate pred, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { local.cmpi64(pred, dst, a, b); });
  }

  // Asserts that op throws std::invalid_argument and that nothing was emitted
  // through the fixture's builder.
  void expectRejected(const std::function<void()> &op) {
    EXPECT_THROW(op(), std::invalid_argument);
    EXPECT_EQ(body.toString(), "");
  }
};

TEST_F(CmpTest, ScalarI32UniformBool) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    std::string out = emit32(current.pred, dst.ref(), a, b);
    expectLinesInOrder(out, {line(current.scalar32, text(a), text(b)),
                             line("s_cselect_b32", text(dst), "1", "0")});
    EXPECT_EQ(countLinesContaining(out, "v_"), 0);
    EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
  }
}

TEST_F(CmpTest, ScalarI32LaneMask) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    std::string out = emit32(current.pred, mask.ref(), a, b);
    expectLinesInOrder(out, {line(current.scalar32, text(a), text(b)),
                             line("s_cselect_b64", text(mask), "-1", "0")});
    EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
  }
}

TEST_F(CmpTest, VectorI32) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange a = builder.allocateVgpr(1);
  RegisterRange b = builder.allocateVgpr(1);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    std::string out = emit32(current.pred, mask.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.vector32, text(mask), text(a), text(b)), "s_nop 1"});
    EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
    EXPECT_EQ(countLinesContaining(out, "s_cselect"), 0);
  }
}

TEST_F(CmpTest, VectorI64) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange a = builder.allocateVgpr(2, 2);
  RegisterRange b = builder.allocateVgpr(2, 2);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    std::string out = emit64(current.pred, mask.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.vector64, text(mask), text(a), text(b)), "s_nop 1"});
    EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  }
}

TEST_F(CmpTest, VectorI32SgprInEitherSlotNeedsNoCopy) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange v = builder.allocateVgpr(1);
  std::string out = emit32(Predicate::slt, mask.ref(), s, v);
  expectLinesInOrder(out, {line("v_cmp_lt_i32", text(mask), text(s), text(v))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  out = emit32(Predicate::slt, mask.ref(), v, s);
  expectLinesInOrder(out, {line("v_cmp_lt_i32", text(mask), text(v), text(s))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(CmpTest, VectorI32InlineImmediateNeedsNoCopy) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(1);
  std::string out = emit32(Predicate::ult, mask.ref(), v, -16);
  expectLinesInOrder(out, {line("v_cmp_lt_u32", text(mask), text(v), "-16")});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  out = emit32(Predicate::ult, mask.ref(), 64, v);
  expectLinesInOrder(out, {line("v_cmp_lt_u32", text(mask), "64", text(v))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(CmpTest, VectorI32LiteralIsCopied) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(1);
  // The temporary takes the lowest free VGPR.
  std::string out = emit32(Predicate::slt, mask.ref(), v, literal32);
  expectLinesInOrder(out, {line("v_mov_b32", "v1", literal32Text),
                           line("v_cmp_lt_i32", text(mask), text(v), "v1"),
                           "s_nop 1"});
  out = emit32(Predicate::slt, mask.ref(), literal32, v);
  expectLinesInOrder(out, {line("v_mov_b32", "v1", literal32Text),
                           line("v_cmp_lt_i32", text(mask), "v1", text(v))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);
}

TEST_F(CmpTest, VectorI64SgprPairAndInlineNeedNoCopy) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange s = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(2, 2);
  std::string out = emit64(Predicate::sge, mask.ref(), s, v);
  expectLinesInOrder(out, {line("v_cmp_ge_i64", text(mask), text(s), text(v))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  out = emit64(Predicate::sge, mask.ref(), v, s);
  expectLinesInOrder(out, {line("v_cmp_ge_i64", text(mask), text(v), text(s))});
  out = emit64(Predicate::sge, mask.ref(), v, -16);
  expectLinesInOrder(out, {line("v_cmp_ge_i64", text(mask), text(v), "-16")});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(CmpTest, VectorI64LiteralIsCopiedToEvenPair) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(2, 2);
  // -100 sign-extends to a high word of -1, which a 64-bit mov cannot encode,
  // so it moves as two words.
  std::string out = emit64(Predicate::ult, mask.ref(), v, -100);
  expectLinesInOrder(
      out, {line("v_mov_b32", "v2", "-100"), line("v_mov_b32", "v3", "-1"),
            line("v_cmp_lt_u64", text(mask), text(v), "v[2:3]"), "s_nop 1"});
}

TEST_F(CmpTest, VectorI64MisalignedPairsAreCopied) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange odd = builder.allocateVgpr(4, 2);
  RegisterRange b = builder.allocateVgpr(2, 2);
  std::string out = emit64(Predicate::ult, mask.ref(), odd.ref(1, 2), b);
  expectLinesInOrder(
      out, {line("v_mov_b32", "v7", "v2"), line("v_mov_b32", "v6", "v1"),
            line("v_cmp_lt_u64", text(mask), "v[6:7]", text(b))});

  RegisterRange sOdd = builder.allocateSgpr(4, 2);
  out = emit64(Predicate::ult, mask.ref(), sOdd.ref(1, 2), b);
  expectLinesInOrder(
      out, {line("v_mov_b32", "v6", "s3"), line("v_mov_b32", "v7", "s4"),
            line("v_cmp_lt_u64", text(mask), "v[6:7]", text(b))});
}

TEST_F(CmpTest, ScalarI32TwoDistinctLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  std::string out = emit32(Predicate::eq, dst.ref(), 0x12345, 0x54321);
  expectLinesInOrder(out, {line("s_mov_b32", "s1", "344865"),
                           line("s_cmp_eq_u32", "74565", "s1"),
                           line("s_cselect_b32", text(dst), "1", "0")});
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  EXPECT_EQ(countLinesContaining(
                emit32(Predicate::eq, dst.ref(), 0x12345, 0x12345), "s_mov"),
            0);
  EXPECT_EQ(countLinesContaining(emit32(Predicate::eq, dst.ref(), 5, 0x12345),
                                 "s_mov"),
            0);
}

TEST_F(CmpTest, ScalarI64EqualityIsDirectForLegalOperands) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
  std::string out = emit64(Predicate::eq, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_eq_u64", text(a), text(b)),
                           line("s_cselect_b32", text(dst), "1", "0")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
  out = emit64(Predicate::ne, dst.ref(), a, 0x12345);
  expectLinesInOrder(out, {line("s_cmp_lg_u64", text(a), "74565")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
  out = emit64(Predicate::ne, dst.ref(), a, INT32_MAX);
  expectLinesInOrder(out, {line("s_cmp_lg_u64", text(a), "2147483647")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
}

TEST_F(CmpTest, ScalarI64EqualityCopiesUnencodableOperands) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange odd = builder.allocateSgpr(4, 2);
  // The temporary pair takes the lowest free even index, s[8:9].
  std::string out = emit64(Predicate::eq, dst.ref(), odd.ref(1, 2), a);
  expectLinesInOrder(out, {line("s_mov_b32", "s9", "s6"),
                           line("s_mov_b32", "s8", "s5"),
                           line("s_cmp_eq_u64", "s[8:9]", text(a))});
  out = emit64(Predicate::ne, dst.ref(), a, -17);
  expectLinesInOrder(out, {line("s_mov_b32", "s8", "-17"),
                           line("s_mov_b32", "s9", "-1"),
                           line("s_cmp_lg_u64", text(a), "s[8:9]")});
  out = emit64(Predicate::ne, dst.ref(), a, INT32_MIN);
  expectLinesInOrder(out, {line("s_mov_b32", "s8", "-2147483648"),
                           line("s_mov_b32", "s9", "-1"),
                           line("s_cmp_lg_u64", text(a), "s[8:9]")});
}

TEST_F(CmpTest, ScalarI64EqualityTwoDistinctLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  std::string out = emit64(Predicate::eq, dst.ref(), 0x12345, 0x54321);
  expectLinesInOrder(out, {line("s_mov_b64", "s[2:3]", "344865"),
                           line("s_cmp_eq_u64", "74565", "s[2:3]")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 1);
}

// Registers: dst s0, padding s1, a s[2:3], b s[4:5], first temporary s6.
class CmpOrderedTest : public CmpTest {
protected:
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
};

TEST_F(CmpOrderedTest, SignedLessBiasesBothHighWords) {
  std::string out = emit64(Predicate::slt, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_xor_b32", "s6", "s3", signBitText),
                           line("s_xor_b32", "s7", "s5", signBitText),
                           line("s_sub_u32", "s8", "s2", "s4"),
                           line("s_subb_u32", "s8", "s6", "s7"),
                           line("s_cselect_b32", "s0", "1", "0")});
  EXPECT_EQ(countLinesContaining(out, "v_"), 0);
}

TEST_F(CmpOrderedTest, UnsignedLessNeedsNoBias) {
  std::string out = emit64(Predicate::ult, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_sub_u32", "s6", "s2", "s4"),
                           line("s_subb_u32", "s6", "s3", "s5"),
                           line("s_cselect_b32", "s0", "1", "0")});
  EXPECT_EQ(countLinesContaining(out, "s_xor"), 0);
}

TEST_F(CmpOrderedTest, UnsignedGreaterSwapsOperands) {
  std::string out = emit64(Predicate::ugt, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_sub_u32", "s6", "s4", "s2"),
                           line("s_subb_u32", "s6", "s5", "s3"),
                           line("s_cselect_b32", "s0", "1", "0")});
}

TEST_F(CmpOrderedTest, GreaterOrEqualInvertsTheResult) {
  std::string out = emit64(Predicate::uge, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_sub_u32", "s6", "s2", "s4"),
                           line("s_subb_u32", "s6", "s3", "s5"),
                           line("s_cselect_b32", "s0", "0", "1")});
  out = emit64(Predicate::sle, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_xor_b32", "s6", "s5", signBitText),
                           line("s_xor_b32", "s7", "s3", signBitText),
                           line("s_sub_u32", "s8", "s4", "s2"),
                           line("s_subb_u32", "s8", "s6", "s7"),
                           line("s_cselect_b32", "s0", "0", "1")});
}

TEST_F(CmpOrderedTest, SignedImmediateIsBiasedOnTheHost) {
  // -0x12345 has the high word -1, which biases to 0x7fffffff.
  std::string out = emit64(Predicate::sle, dst.ref(), a, -0x12345);
  expectLinesInOrder(out, {line("s_xor_b32", "s6", "s3", signBitText),
                           line("s_sub_u32", "s7", "-74565", "s2"),
                           line("s_subb_u32", "s7", "2147483647", "s6"),
                           line("s_cselect_b32", "s0", "0", "1")});
  EXPECT_EQ(countLinesContaining(out, "s_xor_b32"), 1);
}

TEST_F(CmpOrderedTest, LaneMaskDestination) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  std::string out = emit64(Predicate::ult, mask.ref(), a, b);
  expectLinesInOrder(out, {line("s_subb_u32", "s8", "s3", "s5"),
                           line("s_cselect_b64", text(mask), "-1", "0")});
  out = emit64(Predicate::uge, mask.ref(), a, b);
  expectLinesInOrder(out, {line("s_cselect_b64", text(mask), "0", "-1")});
  EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
}

TEST_F(CmpOrderedTest, TwoImmediatesCopyTheSecondFirst) {
  std::string out = emit64(Predicate::ult, dst.ref(), 5, -100);
  expectLinesInOrder(out, {line("s_mov_b32", "s6", "-100"),
                           line("s_mov_b32", "s7", "-1"),
                           line("s_sub_u32", "s8", "5", "s6"),
                           line("s_subb_u32", "s8", "0", "s7"),
                           line("s_cselect_b32", "s0", "1", "0")});
}

TEST_F(CmpOrderedTest, DestinationMayAliasAnOperand) {
  std::string out = emit64(Predicate::ult, b.ref(), a, b);
  expectLinesInOrder(out, {line("s_sub_u32", "s6", "s2", "s4"),
                           line("s_subb_u32", "s6", "s3", "s5"),
                           line("s_cselect_b64", text(b), "-1", "0")});
  out = emit64(Predicate::eq, b.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_eq_u64", text(a), text(b)),
                           line("s_cselect_b64", text(b), "-1", "0")});
}

TEST_F(CmpTest, TemporariesAreReleased) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
  emit64(Predicate::slt, dst.ref(), a, b);
  RegisterRange next = builder.allocateSgpr(1);
  EXPECT_EQ(next.start(), 1);
}

TEST_F(CmpTest, InvalidDestinationsThrowWithoutEmitting) {
  RegisterRange vdst = builder.allocateVgpr(4);
  RegisterRange acc = builder.allocateAgpr(1);
  RegisterRange three = builder.allocateSgpr(3, 2);
  RegisterRange oddPair = builder.allocateSgpr(4, 2);
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange v = builder.allocateVgpr(1);
  std::string message;
  try {
    builder.cmpi32(Predicate::eq, vdst.ref(3), s, s);
  } catch (const std::invalid_argument &error) {
    message = error.what();
  }
  EXPECT_EQ(message, "cmpi32 needs a destination of 1 sgpr or an sgpr pair at "
                     "an even index, got v3");
  expectRejected([&] { builder.cmpi32(Predicate::eq, vdst.ref(0, 2), s, s); });
  expectRejected([&] { builder.cmpi32(Predicate::eq, acc.ref(), s, s); });
  expectRejected([&] { builder.cmpi32(Predicate::eq, three.ref(), s, s); });
  expectRejected(
      [&] { builder.cmpi32(Predicate::eq, oddPair.ref(1, 2), s, v); });
  expectRejected(
      [&] { builder.cmpi64(Predicate::eq, oddPair.ref(1, 2), s, v); });
}

TEST_F(CmpTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(1);
  RegisterRange vpair = builder.allocateVgpr(2, 2);
  RegisterRange acc = builder.allocateAgpr(2, 2);
  expectRejected(
      [&] { builder.cmpi32(Predicate::eq, mask.ref(), acc.ref(0, 1), s); });
  expectRejected(
      [&] { builder.cmpi32(Predicate::eq, mask.ref(), s, acc.ref(1, 1)); });
  expectRejected(
      [&] { builder.cmpi64(Predicate::eq, mask.ref(), acc.ref(), vpair); });
  expectRejected([&] { builder.cmpi32(Predicate::eq, s.ref(), v, s); });
  expectRejected([&] { builder.cmpi64(Predicate::eq, s.ref(), vpair, mask); });
  expectRejected([&] { builder.cmpi32(Predicate::eq, mask.ref(), mask, s); });
  expectRejected([&] { builder.cmpi64(Predicate::eq, mask.ref(), s, vpair); });
}

TEST_F(CmpTest, BoundaryImmediatesAreAccepted) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(1);
  EXPECT_NO_THROW(builder.cmpi32(Predicate::eq, mask.ref(), v, INT32_MIN));
  EXPECT_NO_THROW(builder.cmpi32(Predicate::eq, mask.ref(), v, INT32_MAX));
  RegisterRange pair = builder.allocateVgpr(2, 2);
  EXPECT_NO_THROW(builder.cmpi64(Predicate::eq, mask.ref(), pair, INT32_MIN));
  EXPECT_NO_THROW(builder.cmpi64(Predicate::eq, mask.ref(), pair, INT32_MAX));
}

} // namespace
