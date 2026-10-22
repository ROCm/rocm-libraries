// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Operand.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"

#include "code.hpp"

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

// The upper word 0x00020000 as rocisa prints a decimal immediate.
const std::string upperWord = "131072";

// Literal placements that the allocator never hands out in these tests.
const RegisterRef dstRef{sgpr, 40, 4};
const RegisterRef pointerRef{sgpr, 20, 2};
const RegisterRef sizeRef{sgpr, 30, 1};

RegisterRef word(int index) {
  return RegisterRef{sgpr, dstRef.start + index, 1};
}

class RsrcTest : public BuilderTest {
protected:
  std::string emitRsrc(RegisterRef dst, Operand pointer, Operand size) {
    return emitted(
        [&](IRBuilder &local) { local.makeRsrc(dst, pointer, size); });
  }
  int movCount(const std::string &out) {
    return countLinesContaining(out, "_mov_");
  }
  std::string errorOf(RegisterRef dst, Operand pointer, Operand size) {
    try {
      builder.makeRsrc(dst, pointer, size);
    } catch (const std::invalid_argument &error) {
      return error.what();
    }
    return {};
  }
};

TEST_F(RsrcTest, Word3ComesFromRocisa) {
  EXPECT_EQ(rocisa::SrdUpperValue(target.isaVersion())->getValue(), 0x00020000);
}

TEST_F(RsrcTest, SgprSize) {
  std::string out = emitRsrc(dstRef, pointerRef, sizeRef);
  EXPECT_EQ(movCount(out), 3);
  expectLinesInOrder(
      out, {line("s_mov_b64", text(RegisterRef{sgpr, 40, 2}), text(pointerRef)),
            line("s_mov_b32", text(word(2)), text(sizeRef)),
            line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, ImmediateSize) {
  struct Case {
    int64_t value;
    std::string printed;
  };
  for (const Case &current :
       {Case{0, "0"}, Case{308, "308"}, Case{UINT32_MAX, "-1"}}) {
    SCOPED_TRACE(current.printed);
    std::string out = emitRsrc(dstRef, pointerRef, current.value);
    EXPECT_EQ(movCount(out), 3);
    expectLinesInOrder(out, {line("s_mov_b64", text(RegisterRef{sgpr, 40, 2}),
                                  text(pointerRef)),
                             line("s_mov_b32", text(word(2)), current.printed),
                             line("s_mov_b32", text(word(3)), upperWord)});
  }
}

TEST_F(RsrcTest, MisalignedPointerPair) {
  RegisterRef pointer{sgpr, 21, 2};
  std::string out = emitRsrc(dstRef, pointer, sizeRef);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b64"), 0);
  EXPECT_EQ(movCount(out), 4);
  // The split copies the high half first because dst starts above the source.
  expectLinesInOrder(
      out, {line("s_mov_b32", text(word(1)), text(RegisterRef{sgpr, 22, 1})),
            line("s_mov_b32", text(word(0)), text(RegisterRef{sgpr, 21, 1})),
            line("s_mov_b32", text(word(2)), text(sizeRef)),
            line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, PointerAlreadyInPlaceIsNotCopied) {
  RegisterRef pointer{sgpr, 40, 2};
  std::string out = emitRsrc(dstRef, pointer, sizeRef);
  EXPECT_EQ(movCount(out), 2);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b64"), 0);
}

TEST_F(RsrcTest, SizeAlreadyInPlaceIsNotCopied) {
  std::string out = emitRsrc(dstRef, pointerRef, word(2));
  EXPECT_EQ(movCount(out), 2);
  expectLinesInOrder(
      out, {"s_mov_b64", line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, PointerAndSizeAlreadyInPlace) {
  std::string out = emitRsrc(dstRef, RegisterRef{sgpr, 40, 2}, word(2));
  EXPECT_EQ(out.find("s_mov_b64"), std::string::npos);
  EXPECT_EQ(movCount(out), 1);
  expectLinesInOrder(out, {line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, PointerInUpperHalfIsReadBeforeSizeIsWritten) {
  std::string out = emitRsrc(dstRef, RegisterRef{sgpr, 42, 2}, sizeRef);
  EXPECT_EQ(movCount(out), 3);
  expectLinesInOrder(out, {line("s_mov_b64", text(RegisterRef{sgpr, 40, 2}),
                                text(RegisterRef{sgpr, 42, 2})),
                           line("s_mov_b32", text(word(2)), text(sizeRef)),
                           line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, SizeInWord3IsReadBeforeItIsOverwritten) {
  std::string out = emitRsrc(dstRef, pointerRef, word(3));
  EXPECT_EQ(movCount(out), 3);
  expectLinesInOrder(out, {"s_mov_b64",
                           line("s_mov_b32", text(word(2)), text(word(3))),
                           line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, SizeInWord0IsSavedBeforePointerCopy) {
  RegisterRange probe = builder.allocateSgpr(1);
  RegisterRef temp = probe.ref();
  probe.release();
  std::string out = emitRsrc(dstRef, pointerRef, word(0));
  EXPECT_EQ(movCount(out), 4);
  expectLinesInOrder(
      out, {line("s_mov_b32", text(temp), text(word(0))),
            line("s_mov_b64", text(RegisterRef{sgpr, 40, 2}), text(pointerRef)),
            line("s_mov_b32", text(word(2)), text(temp)),
            line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, SizeInWord1IsSavedBeforePointerCopy) {
  std::string out = emitRsrc(dstRef, pointerRef, word(1));
  EXPECT_EQ(movCount(out), 4);
}

TEST_F(RsrcTest, SizeInWord1NeedsNoTemporaryWhenPointerStaysInPlace) {
  std::string out = emitRsrc(dstRef, RegisterRef{sgpr, 40, 2}, word(1));
  EXPECT_EQ(movCount(out), 2);
  expectLinesInOrder(out, {line("s_mov_b32", text(word(2)), text(word(1))),
                           line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, TemporaryIsReleasedAfterwards) {
  RegisterRange probe = builder.allocateSgpr(1);
  int expected = probe.ref().start;
  probe.release();
  emitRsrc(dstRef, pointerRef, word(0));
  RegisterRange after = builder.allocateSgpr(1);
  EXPECT_EQ(after.ref().start, expected);
}

TEST_F(RsrcTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRef vgprDst{vgpr, 0, 4};
  RegisterRef agprDst{agpr, 0, 4};
  EXPECT_THROW(builder.makeRsrc(vgprDst, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(agprDst, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc({sgpr, 40, 3}, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc({sgpr, 40, 8}, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc({sgpr, 42, 4}, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc({sgpr, 46, 4}, pointerRef, sizeRef),
               std::invalid_argument);

  EXPECT_THROW(builder.makeRsrc(dstRef, 0, sizeRef), std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{vgpr, 2, 2}, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{agpr, 2, 2}, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{sgpr, 20, 1}, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{sgpr, 20, 4}, sizeRef),
               std::invalid_argument);

  EXPECT_THROW(builder.makeRsrc(dstRef, pointerRef, RegisterRef{vgpr, 3, 1}),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, pointerRef, RegisterRef{agpr, 3, 1}),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, pointerRef, RegisterRef{sgpr, 30, 2}),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, pointerRef, -1), std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, pointerRef, int64_t{1} << 32),
               std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
}

TEST_F(RsrcTest, FailureWhereTemporaryWouldBeNeededAllocatesAndEmitsNothing) {
  RegisterRange probe = builder.allocateSgpr(1);
  int expected = probe.ref().start;
  probe.release();
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{vgpr, 2, 2}, word(0)),
               std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
  RegisterRange after = builder.allocateSgpr(1);
  EXPECT_EQ(after.ref().start, expected);
}

TEST_F(RsrcTest, ErrorMessagesNameTheOperation) {
  EXPECT_NE(errorOf({sgpr, 42, 4}, pointerRef, sizeRef).find("multiple of 4"),
            std::string::npos);
  EXPECT_NE(errorOf(dstRef, RegisterRef{vgpr, 2, 2}, sizeRef).find("sgpr pair"),
            std::string::npos);
  EXPECT_NE(errorOf(dstRef, 0, sizeRef).find("immediate 0"), std::string::npos);
  EXPECT_NE(errorOf(dstRef, pointerRef, -1).find("makeRsrc size -1"),
            std::string::npos);
}

} // namespace
