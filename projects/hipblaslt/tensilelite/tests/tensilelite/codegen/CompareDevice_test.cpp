// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "ArithDeviceTest.hpp"
#include "HipTestSupport.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr RegisterKind vgpr = RegisterKind::vgpr;
constexpr RegisterKind sgpr = RegisterKind::sgpr;
constexpr InputKind lane = InputKind::lane;
constexpr InputKind uniform = InputKind::uniform;

constexpr int randomPairs = 1024;
// Index step that walks the edge values of the cross product in a pair vector.
constexpr int edgeStride32 = 17;
constexpr int edgeStride64 = 18;

constexpr int64_t literal32 = 0x12345678;
constexpr int64_t literal64 = INT64_C(0x1234567890abcdef);
constexpr int64_t inlineImmediate = -16;
constexpr int64_t nearMin64 = INT64_MIN + 5;

using CompareFn = void (IRBuilder::*)(Predicate, RegisterRef, Operand, Operand);
using Words = std::vector<uint32_t>;
using Values = std::vector<uint64_t>;
using Inputs = std::vector<Words>;

struct PredicateCase {
  const char *name;
  Predicate pred;
};

const PredicateCase predicates[] = {
    {"eq", Predicate::eq},   {"ne", Predicate::ne},   {"slt", Predicate::slt},
    {"sle", Predicate::sle}, {"sgt", Predicate::sgt}, {"sge", Predicate::sge},
    {"ult", Predicate::ult}, {"ule", Predicate::ule}, {"ugt", Predicate::ugt},
    {"uge", Predicate::uge},
};

// Compares as Signed or Unsigned values of the operands' width.
template <typename Signed, typename Unsigned>
bool compareAs(Predicate pred, uint64_t x, uint64_t y) {
  Unsigned ua = static_cast<Unsigned>(x);
  Unsigned ub = static_cast<Unsigned>(y);
  Signed sa = static_cast<Signed>(ua);
  Signed sb = static_cast<Signed>(ub);
  switch (pred) {
  case Predicate::eq:
    return ua == ub;
  case Predicate::ne:
    return ua != ub;
  case Predicate::slt:
    return sa < sb;
  case Predicate::sle:
    return sa <= sb;
  case Predicate::sgt:
    return sa > sb;
  case Predicate::sge:
    return sa >= sb;
  case Predicate::ult:
    return ua < ub;
  case Predicate::ule:
    return ua <= ub;
  case Predicate::ugt:
    return ua > ub;
  case Predicate::uge:
    return ua >= ub;
  }
  return false;
}

// Expected 0/1 words, one per pair of values.
Words expectedBits(int bits, Predicate pred, const Values &a, const Values &b) {
  Words expected;
  for (size_t i = 0; i < a.size(); ++i) {
    bool holds = bits == 32 ? compareAs<int32_t, uint32_t>(pred, a[i], b[i])
                            : compareAs<int64_t, uint64_t>(pred, a[i], b[i]);
    expected.push_back(holds ? 1 : 0);
  }
  return expected;
}

Values widen(const Words &words) { return Values(words.begin(), words.end()); }

Values join(const Words &low, const Words &high) {
  Values values;
  for (size_t i = 0; i < low.size(); ++i)
    values.push_back(low[i] | uint64_t{high[i]} << 32);
  return values;
}

template <typename T>
std::vector<T> perLane(const std::vector<T> &perGroup) {
  std::vector<T> lanes;
  for (const T &value : perGroup)
    lanes.insert(lanes.end(), arithWorkgroupSize, value);
  return lanes;
}

// Every stride-th element, which gives one value per workgroup.
template <typename T>
std::vector<T> everyNth(const std::vector<T> &values, int stride,
                        size_t count) {
  std::vector<T> picked;
  for (size_t i = 0; i < count; ++i)
    picked.push_back(values[i * stride]);
  return picked;
}

int groupCount(const Words &laneWords) {
  return static_cast<int>(laneWords.size()) / arithWorkgroupSize;
}

// The compared operands, and the registers that keep them allocated.
struct Operands {
  std::vector<RegisterRange> owners;
  Operand a;
  Operand b;
};

using OperandMaker = std::function<Operands(
    IRBuilder &builder, const std::vector<RegisterRef> &inputs)>;

// 32-bit operands straight from inputs 0 and 1.
OperandMaker words32() {
  return [](IRBuilder &, const std::vector<RegisterRef> &in) {
    return Operands{{}, in[0], in[1]};
  };
}

// Input 0 against an immediate, which comes first if immediateFirst.
OperandMaker word32WithImmediate(int64_t immediate, bool immediateFirst) {
  return [=](IRBuilder &, const std::vector<RegisterRef> &in) {
    if (immediateFirst)
      return Operands{{}, immediate, in[0]};
    return Operands{{}, in[0], immediate};
  };
}

// 64-bit operands from inputs (0, 1) and (2, 3).
OperandMaker pairs64(bool misalignedA = false, bool misalignedB = false) {
  return [=](IRBuilder &builder, const std::vector<RegisterRef> &in) {
    PairRegisters a = makePair(builder, in[0], in[1], misalignedA);
    PairRegisters b = makePair(builder, in[2], in[3], misalignedB);
    Operands operands{{}, a.pair, b.pair};
    operands.owners.push_back(std::move(a.range));
    operands.owners.push_back(std::move(b.range));
    return operands;
  };
}

// The pair from inputs (0, 1) against an immediate.
OperandMaker pair64WithImmediate(int64_t immediate, bool immediateFirst,
                                 bool misaligned) {
  return [=](IRBuilder &builder, const std::vector<RegisterRef> &in) {
    PairRegisters pair = makePair(builder, in[0], in[1], misaligned);
    Operands operands = immediateFirst ? Operands{{}, immediate, pair.pair}
                                       : Operands{{}, pair.pair, immediate};
    operands.owners.push_back(std::move(pair.range));
    return operands;
  };
}

// Every lane stores 1 if the comparison, written to a lane mask, holds.
void expectMaskResults(CompareFn compare, Predicate pred,
                       std::vector<InputKind> kinds, const OperandMaker &make,
                       const Inputs &inputs, int groups,
                       const Words &expected) {
  ArithKernelSpec spec{std::move(kinds), vgpr,
                       [=](IRBuilder &builder, RegisterRef dst,
                           const std::vector<RegisterRef> &in) {
                         Operands operands = make(builder, in);
                         RegisterRange mask = builder.allocateSgpr(2, 2);
                         (builder.*compare)(pred, mask.ref(), operands.a,
                                            operands.b);
                         builder.select(dst, mask.ref(), 1, 0);
                       }};
  expectArithResults(spec, groups, inputs, expected);
}

// Each workgroup stores the uniform bool the comparison wrote to one SGPR.
void expectBoolResults(CompareFn compare, Predicate pred,
                       std::vector<InputKind> kinds, const OperandMaker &make,
                       const Inputs &inputs, int groups,
                       const Words &expected) {
  ArithKernelSpec spec{std::move(kinds), sgpr,
                       [=](IRBuilder &builder, RegisterRef dst,
                           const std::vector<RegisterRef> &in) {
                         Operands operands = make(builder, in);
                         (builder.*compare)(pred, dst, operands.a, operands.b);
                       }};
  expectArithResults(spec, groups, inputs, expected);
}

class CmpDevice : public testing::Test {
protected:
  void SetUp() override {
    if (!hasHipDevice())
      GTEST_SKIP() << "no HIP device";
  }
};

TEST_F(CmpDevice, LaneOperandsI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    expectMaskResults(
        &IRBuilder::cmpi32, current.pred, {lane, lane}, words32(),
        {pairs.a, pairs.b}, groupCount(pairs.a),
        expectedBits(32, current.pred, widen(pairs.a), widen(pairs.b)));
  }
}

TEST_F(CmpDevice, UniformOperandsToBoolI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    expectBoolResults(
        &IRBuilder::cmpi32, current.pred, {uniform, uniform}, words32(),
        {pairs.a, pairs.b}, static_cast<int>(pairs.a.size()),
        expectedBits(32, current.pred, widen(pairs.a), widen(pairs.b)));
  }
}

TEST_F(CmpDevice, UniformOperandsToMaskI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    expectMaskResults(&IRBuilder::cmpi32, current.pred, {uniform, uniform},
                      words32(), {pairs.a, pairs.b},
                      static_cast<int>(pairs.a.size()),
                      perLane(expectedBits(32, current.pred, widen(pairs.a),
                                           widen(pairs.b))));
  }
}

TEST_F(CmpDevice, MixedUniformAndLaneI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  int groups = groupCount(pairs.a);
  Words uniformWords = everyNth(pairs.a, edgeStride32, groups);
  Values uniformLanes = widen(perLane(uniformWords));
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    expectMaskResults(
        &IRBuilder::cmpi32, current.pred, {uniform, lane}, words32(),
        {uniformWords, pairs.b}, groups,
        expectedBits(32, current.pred, uniformLanes, widen(pairs.b)));
    expectMaskResults(
        &IRBuilder::cmpi32, current.pred, {lane, uniform}, words32(),
        {pairs.b, uniformWords}, groups,
        expectedBits(32, current.pred, widen(pairs.b), uniformLanes));
  }
}

TEST_F(CmpDevice, LaneAgainstImmediateI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicates) {
    for (int64_t immediate : {literal32, inlineImmediate}) {
      for (bool immediateFirst : {false, true}) {
        SCOPED_TRACE(std::string(current.name) + " immediate " +
                     std::to_string(immediate) +
                     (immediateFirst ? " first" : " second"));
        Values constant(pairs.a.size(), static_cast<uint32_t>(immediate));
        expectMaskResults(
            &IRBuilder::cmpi32, current.pred, {lane},
            word32WithImmediate(immediate, immediateFirst), {pairs.a},
            groupCount(pairs.a),
            immediateFirst
                ? expectedBits(32, current.pred, constant, widen(pairs.a))
                : expectedBits(32, current.pred, widen(pairs.a), constant));
      }
    }
  }
}

TEST_F(CmpDevice, UniformAgainstImmediateI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    Values constant(pairs.a.size(), static_cast<uint32_t>(literal32));
    expectBoolResults(&IRBuilder::cmpi32, current.pred, {uniform},
                      word32WithImmediate(literal32, false), {pairs.a},
                      static_cast<int>(pairs.a.size()),
                      expectedBits(32, current.pred, widen(pairs.a), constant));
  }
}

TEST_F(CmpDevice, DestinationMayAliasAnOperandI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform},
                         sgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           builder.cmpi32(pred, in[0], in[0], in[1]);
                           builder.movb32(dst, in[0]);
                         }};
    expectArithResults(spec, static_cast<int>(pairs.a.size()),
                       {pairs.a, pairs.b},
                       expectedBits(32, pred, widen(pairs.a), widen(pairs.b)));
  }
}

// The 64-bit tests pass each operand as a low and a high word.
Inputs laneInputs(const WordQuads &q) { return {q.aLo, q.aHi, q.bLo, q.bHi}; }

TEST_F(CmpDevice, LaneOperandsI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    Words expected = expectedBits(64, current.pred, a, b);
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {lane, lane, lane, lane}, pairs64(), laneInputs(quads),
                      groupCount(quads.aLo), expected);
    SCOPED_TRACE("misaligned pairs");
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {lane, lane, lane, lane}, pairs64(true, true),
                      laneInputs(quads), groupCount(quads.aLo), expected);
  }
}

TEST_F(CmpDevice, UniformOperandsToBoolI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  int groups = static_cast<int>(a.size());
  std::vector<InputKind> kinds{uniform, uniform, uniform, uniform};
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    Words expected = expectedBits(64, current.pred, a, b);
    expectBoolResults(&IRBuilder::cmpi64, current.pred, kinds, pairs64(),
                      laneInputs(quads), groups, expected);
    SCOPED_TRACE("misaligned pairs");
    expectBoolResults(&IRBuilder::cmpi64, current.pred, kinds,
                      pairs64(true, true), laneInputs(quads), groups, expected);
  }
}

TEST_F(CmpDevice, UniformOperandsToMaskI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {uniform, uniform, uniform, uniform}, pairs64(),
                      laneInputs(quads), static_cast<int>(a.size()),
                      perLane(expectedBits(64, current.pred, a, b)));
  }
}

TEST_F(CmpDevice, MixedUniformAndLaneI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  int groups = groupCount(quads.aLo);
  Words uniformLo = everyNth(quads.aLo, edgeStride64, groups);
  Words uniformHi = everyNth(quads.aHi, edgeStride64, groups);
  Values uniformLanes = perLane(join(uniformLo, uniformHi));
  Values laneValues = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {uniform, uniform, lane, lane}, pairs64(),
                      {uniformLo, uniformHi, quads.bLo, quads.bHi}, groups,
                      expectedBits(64, current.pred, uniformLanes, laneValues));
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {lane, lane, uniform, uniform}, pairs64(),
                      {quads.bLo, quads.bHi, uniformLo, uniformHi}, groups,
                      expectedBits(64, current.pred, laneValues, uniformLanes));
  }
}

TEST_F(CmpDevice, LaneAgainstImmediateI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values lanes = join(quads.aLo, quads.aHi);
  for (const PredicateCase &current : predicates) {
    for (int64_t immediate : {literal64, inlineImmediate, nearMin64}) {
      for (bool immediateFirst : {false, true}) {
        for (bool misaligned : {false, true}) {
          SCOPED_TRACE(std::string(current.name) + " immediate " +
                       std::to_string(immediate) +
                       (immediateFirst ? " first" : " second") +
                       (misaligned ? " misaligned" : " aligned"));
          Values constant(lanes.size(), static_cast<uint64_t>(immediate));
          expectMaskResults(
              &IRBuilder::cmpi64, current.pred, {lane, lane},
              pair64WithImmediate(immediate, immediateFirst, misaligned),
              {quads.aLo, quads.aHi}, groupCount(quads.aLo),
              immediateFirst ? expectedBits(64, current.pred, constant, lanes)
                             : expectedBits(64, current.pred, lanes, constant));
        }
      }
    }
  }
}

TEST_F(CmpDevice, UniformAgainstImmediateI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values values = join(quads.aLo, quads.aHi);
  for (const PredicateCase &current : predicates) {
    for (int64_t immediate : {nearMin64, literal64, inlineImmediate}) {
      for (bool immediateFirst : {false, true}) {
        SCOPED_TRACE(std::string(current.name) + " immediate " +
                     std::to_string(immediate) +
                     (immediateFirst ? " first" : " second"));
        Values constant(values.size(), static_cast<uint64_t>(immediate));
        expectBoolResults(
            &IRBuilder::cmpi64, current.pred, {uniform, uniform},
            pair64WithImmediate(immediate, immediateFirst, false),
            {quads.aLo, quads.aHi}, static_cast<int>(values.size()),
            immediateFirst ? expectedBits(64, current.pred, constant, values)
                           : expectedBits(64, current.pred, values, constant));
      }
    }
  }
}

TEST_F(CmpDevice, DestinationMayAliasAnOperandI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform, uniform, uniform},
                         vgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           PairRegisters x = makePair(builder, in[0], in[1]);
                           PairRegisters y = makePair(builder, in[2], in[3]);
                           builder.cmpi64(pred, x.pair, x.pair, y.pair);
                           builder.select(dst, x.pair, 1, 0);
                         }};
    expectArithResults(spec, static_cast<int>(a.size()), laneInputs(quads),
                       perLane(expectedBits(64, pred, a, b)));
  }
}

TEST_F(CmpDevice, VectorDestinationMayAliasAnSgprOperandI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  int groups = groupCount(quads.aLo);
  Words uniformLo = everyNth(quads.aLo, edgeStride64, groups);
  Words uniformHi = everyNth(quads.aHi, edgeStride64, groups);
  Values uniformLanes = perLane(join(uniformLo, uniformHi));
  Values laneValues = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicates) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform, lane, lane},
                         vgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           PairRegisters x = makePair(builder, in[0], in[1]);
                           PairRegisters y = makePair(builder, in[2], in[3]);
                           builder.cmpi64(pred, x.pair, x.pair, y.pair);
                           builder.select(dst, x.pair, 1, 0);
                         }};
    expectArithResults(spec, groups,
                       {uniformLo, uniformHi, quads.bLo, quads.bHi},
                       expectedBits(64, pred, uniformLanes, laneValues));
  }
}

} // namespace
