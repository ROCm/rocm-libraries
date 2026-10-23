// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Compiler.hpp>
#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Kernel.hpp>

#include "ArithDeviceTest.hpp"
#include "HipTestSupport.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr RegisterKind sgpr = RegisterKind::sgpr;
constexpr RegisterKind vgpr = RegisterKind::vgpr;
constexpr RegisterKind agpr = RegisterKind::agpr;

const int32_t immediates32[] = {0, -1, 64, 65, INT32_MIN, 0x12345678};
// A pair immediate is sign-extended to 64 bits.
const int32_t immediates64[] = {
    0, 1, -1, 64, 65, -16, -17, INT32_MIN, INT32_MAX, 0x12345678, -0x12345678};

class MoveDevice : public testing::Test {
protected:
  void SetUp() override {
    if (!hasHipDevice())
      GTEST_SKIP() << "no HIP device";
  }
};

int groupCount(const std::vector<uint32_t> &laneWords) {
  return static_cast<int>(laneWords.size()) / arithWorkgroupSize;
}

// Repeats each workgroup's word for every lane of the workgroup.
std::vector<uint32_t> perLane(const std::vector<uint32_t> &perGroup) {
  std::vector<uint32_t> lanes;
  for (uint32_t word : perGroup)
    lanes.insert(lanes.end(), arithWorkgroupSize, word);
  return lanes;
}

RegisterRef halfOf(RegisterRef pair, int index) {
  return RegisterRef{pair.kind, pair.start + index, 1};
}

TEST_F(MoveDevice, Movb32LaneThroughVgpr) {
  WordPairs pairs = makeArithPairs(0);
  expectArithResults({{InputKind::lane},
                      vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        RegisterRange temp = builder.allocateVgpr(1);
                        builder.movb32(temp.ref(), inputs[0]);
                        builder.movb32(dst, temp.ref());
                      }},
                     groupCount(pairs.a), {pairs.a}, pairs.a);
}

TEST_F(MoveDevice, Movb32UniformToVgprAndSgpr) {
  WordPairs pairs = makeArithPairs(0);
  int groups = static_cast<int>(pairs.a.size());
  ArithEmitter copy = [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
    builder.movb32(dst, inputs[0]);
  };
  expectArithResults({{InputKind::uniform}, vgpr, copy}, groups, {pairs.a},
                     perLane(pairs.a));
  expectArithResults({{InputKind::uniform}, sgpr, copy}, groups, {pairs.a},
                     pairs.a);
}

TEST_F(MoveDevice, Movb32Immediates) {
  for (int32_t immediate : immediates32) {
    SCOPED_TRACE(immediate);
    ArithEmitter emit = [immediate](IRBuilder &builder, RegisterRef dst,
                                    const std::vector<RegisterRef> &) {
      builder.movb32(dst, immediate);
    };
    uint32_t word = static_cast<uint32_t>(immediate);
    expectArithResults({{}, vgpr, emit}, 2, {},
                       std::vector<uint32_t>(2 * arithWorkgroupSize, word));
    expectArithResults({{}, sgpr, emit}, 3, {}, std::vector<uint32_t>(3, word));
  }
}

TEST_F(MoveDevice, Movb32ThroughAccumulator) {
  WordPairs pairs = makeArithPairs(0);
  int groups = static_cast<int>(pairs.a.size());
  ArithEmitter viaAgpr = [](IRBuilder &builder, RegisterRef dst,
                            const std::vector<RegisterRef> &inputs) {
    RegisterRange acc = builder.allocateAgpr(1);
    builder.movb32(acc.ref(), inputs[0]);
    builder.movb32(dst, acc.ref());
  };
  SCOPED_TRACE("lane vgpr to agpr");
  expectArithResults({{InputKind::lane}, vgpr, viaAgpr}, groupCount(pairs.a),
                     {pairs.a}, pairs.a);
  SCOPED_TRACE("uniform sgpr to agpr");
  expectArithResults({{InputKind::uniform}, vgpr, viaAgpr}, groups, {pairs.a},
                     perLane(pairs.a));
  SCOPED_TRACE("inline constant to agpr");
  expectArithResults({{},
                      vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &) {
                        RegisterRange acc = builder.allocateAgpr(1);
                        builder.movb32(acc.ref(), 7);
                        builder.movb32(dst, acc.ref());
                      }},
                     2, {}, std::vector<uint32_t>(2 * arithWorkgroupSize, 7));
}

// A pair under test, with the registers that keep it allocated.
struct PairResult {
  RegisterRange owner;
  RegisterRef pair;
};

using PairEmitter = std::function<PairResult(
    IRBuilder &builder, const std::vector<RegisterRef> &inputs)>;

// A 64-bit value is checked through two runs, one storing each half.
void expectPairResults(const std::vector<InputKind> &kinds, RegisterKind result,
                       const PairEmitter &emit, int workgroupCount,
                       const std::vector<std::vector<uint32_t>> &inputs,
                       const std::vector<uint32_t> &low,
                       const std::vector<uint32_t> &high) {
  for (int half = 0; half < 2; ++half) {
    SCOPED_TRACE(half == 0 ? "low half" : "high half");
    ArithEmitter storeHalf = [&emit, half](IRBuilder &builder, RegisterRef dst,
                                           const std::vector<RegisterRef> &in) {
      PairResult value = emit(builder, in);
      builder.movb32(dst, halfOf(value.pair, half));
    };
    expectArithResults({kinds, result, storeHalf}, workgroupCount, inputs,
                       half == 0 ? low : high);
  }
}

RegisterRange allocatePair(IRBuilder &builder, RegisterKind kind) {
  return kind == sgpr ? builder.allocateSgpr(2, 2) : builder.allocateVgpr(2, 2);
}

TEST_F(MoveDevice, Movb64Immediates) {
  for (int32_t immediate : immediates64) {
    for (RegisterKind kind : {sgpr, vgpr}) {
      SCOPED_TRACE("immediate " + std::to_string(immediate) +
                   (kind == sgpr ? " to sgpr pair" : " to vgpr pair"));
      PairEmitter emit = [immediate, kind](IRBuilder &builder,
                                           const std::vector<RegisterRef> &) {
        RegisterRange pair = allocatePair(builder, kind);
        RegisterRef ref = pair.ref();
        builder.movb64(ref, immediate);
        return PairResult{std::move(pair), ref};
      };
      // The expected halves come from the sign-extended host value, which
      // proves that a negative literal is split and not zero-extended.
      int64_t extended = immediate;
      uint32_t low = static_cast<uint32_t>(extended);
      uint32_t high = static_cast<uint32_t>(extended >> 32);
      size_t words = kind == sgpr ? 3 : 2 * arithWorkgroupSize;
      expectPairResults({}, kind, emit, kind == sgpr ? 3 : 2, {},
                        std::vector<uint32_t>(words, low),
                        std::vector<uint32_t>(words, high));
    }
  }
}

// Fills a pair from two inputs with 32-bit moves.
RegisterRange fillPair(IRBuilder &builder, RegisterKind kind, RegisterRef low,
                       RegisterRef high) {
  RegisterRange pair = allocatePair(builder, kind);
  builder.movb32(halfOf(pair.ref(), 0), low);
  builder.movb32(halfOf(pair.ref(), 1), high);
  return pair;
}

TEST_F(MoveDevice, Movb64VgprPairsAlignedAndMisaligned) {
  WordPairs pairs = makeArithPairs(0);
  int groups = groupCount(pairs.a);
  std::vector<InputKind> kinds{InputKind::lane, InputKind::lane};
  SCOPED_TRACE("aligned to aligned");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, vgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateVgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref());
        return PairResult{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
  SCOPED_TRACE("aligned to misaligned");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, vgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateVgpr(3, 2);
        RegisterRef ref = dst.ref(1, 2);
        builder.movb64(ref, src.ref());
        return PairResult{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
  SCOPED_TRACE("misaligned to aligned");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = builder.allocateVgpr(3, 2);
        builder.movb32(src.ref(1, 1), in[0]);
        builder.movb32(src.ref(2, 1), in[1]);
        RegisterRange dst = builder.allocateVgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref(1, 2));
        return PairResult{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
}

TEST_F(MoveDevice, Movb64OverlappingMisalignedCopies) {
  WordPairs pairs = makeArithPairs(0);
  int groups = groupCount(pairs.a);
  std::vector<InputKind> kinds{InputKind::lane, InputKind::lane};
  SCOPED_TRACE("copy up by one register");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange range = builder.allocateVgpr(4, 2);
        builder.movb32(range.ref(0, 1), in[0]);
        builder.movb32(range.ref(1, 1), in[1]);
        builder.movb64(range.ref(1, 2), range.ref(0, 2));
        RegisterRef ref = range.ref(1, 2);
        return PairResult{std::move(range), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
  SCOPED_TRACE("copy down by one register");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange range = builder.allocateVgpr(4, 2);
        builder.movb32(range.ref(1, 1), in[0]);
        builder.movb32(range.ref(2, 1), in[1]);
        builder.movb64(range.ref(0, 2), range.ref(1, 2));
        RegisterRef ref = range.ref(0, 2);
        return PairResult{std::move(range), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
}

TEST_F(MoveDevice, Movb64ScalarPairs) {
  WordPairs pairs = makeArithPairs(0);
  int groups = static_cast<int>(pairs.a.size());
  std::vector<InputKind> kinds{InputKind::uniform, InputKind::uniform};
  SCOPED_TRACE("sgpr pair to sgpr pair");
  expectPairResults(
      kinds, sgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, sgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateSgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref());
        return PairResult{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
  SCOPED_TRACE("sgpr pair to vgpr pair");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, sgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateVgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref());
        return PairResult{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, perLane(pairs.a), perLane(pairs.b));
}

// Placements of a source, checked by the assembler alone.
Operand registerSource(RegisterKind kind, RegisterRange &sgprs,
                       RegisterRange &vgprs, RegisterRange &agprs, int width,
                       int offset) {
  RegisterRange &range = kind == sgpr ? sgprs : kind == vgpr ? vgprs : agprs;
  return range.ref(offset, width);
}

struct PlacementCounts {
  int assembled = 0;
  int rejected = 0;
};

// Emits dst = src in a fresh kernel and assembles it. A combination the
// builder rejects is counted and skipped.
void assemblePlacement(
    RegisterKind dstKind, int width, int dstOffset,
    const std::function<Operand(RegisterRange &, RegisterRange &,
                                RegisterRange &)> &makeSource,
    PlacementCounts &counts) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_move_placement", arithWorkgroupSize);
  IRBuilder &builder = kernel.builder();
  RegisterRange sgprs = builder.allocateSgpr(4, 2);
  RegisterRange vgprs = builder.allocateVgpr(4, 2);
  RegisterRange agprs = builder.allocateAgpr(4, 2);
  RegisterRange &dstRange = dstKind == sgpr   ? sgprs
                            : dstKind == vgpr ? vgprs
                                              : agprs;
  RegisterRef dst = dstRange.ref(dstOffset, width);
  Operand src = makeSource(sgprs, vgprs, agprs);
  try {
    if (width == 1)
      builder.movb32(dst, src);
    else
      builder.movb64(dst, src);
  } catch (const std::invalid_argument &) {
    ++counts.rejected;
    return;
  }
  builder.createEndProgram();
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
  ++counts.assembled;
}

TEST(MoveAssembly, AllPlacementsAssemble) {
  PlacementCounts counts;
  for (RegisterKind dstKind : {sgpr, vgpr, agpr}) {
    for (int width : {1, 2}) {
      for (int offset : {0, 1}) {
        for (RegisterKind srcKind : {sgpr, vgpr, agpr})
          for (int srcOffset : {0, 1})
            assemblePlacement(
                dstKind, width, offset,
                [=](RegisterRange &s, RegisterRange &v, RegisterRange &a) {
                  return registerSource(srcKind, s, v, a, width, srcOffset);
                },
                counts);
        std::vector<int32_t> values{7, -5, 0x12345, -1};
        if (width == 2)
          values.insert(values.end(), {-17, INT32_MIN, INT32_MAX});
        for (int32_t value : values)
          assemblePlacement(
              dstKind, width, offset,
              [=](RegisterRange &, RegisterRange &, RegisterRange &) {
                return Operand(value);
              },
              counts);
      }
    }
  }
  EXPECT_GT(counts.assembled, 60);
  EXPECT_GT(counts.rejected, 0);
}

} // namespace
