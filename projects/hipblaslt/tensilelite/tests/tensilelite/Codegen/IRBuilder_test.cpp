// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Kernel.hpp>

#include "TestSupport.hpp"

#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

static constexpr RegisterKind sgpr = RegisterKind::sgpr;
static constexpr RegisterKind vgpr = RegisterKind::vgpr;
static constexpr RegisterKind agpr = RegisterKind::agpr;
static constexpr InputKind lane = InputKind::lane;
static constexpr InputKind uniform = InputKind::uniform;

static constexpr int32_t literalA = 0x12345;
static constexpr int32_t literalB = 0x54321;
static constexpr const char *literalAText = "74565";
static constexpr const char *literalBText = "344865";
static constexpr int32_t literal32 = 0x12345678;
static constexpr const char *literal32Text = "305419896";
static constexpr int32_t randomPairs = 1024;

// A pair immediate is sign-extended to 64 bits.
static constexpr int32_t immediates64[] = {
    0, 1, -1, 64, 65, -16, -17, INT32_MIN, INT32_MAX, 0x12345678, -0x12345678};

using Inputs = std::vector<Words>;

namespace {

// Collects the kernel inputs and the sizes the harness expects for them.
struct KernelInputs {
  std::vector<InputKind> kinds;
  Inputs words;
};

} // namespace

// The non-empty lines of text, without leading blanks.
static std::vector<std::string> instructionsOf(const std::string &text) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string current; std::getline(stream, current);) {
    size_t first = current.find_first_not_of(" \t");
    if (first != std::string::npos)
      lines.push_back(current.substr(first));
  }
  return lines;
}

static RegisterRange allocateKind(IRBuilder &builder, RegisterKind kind,
                                  int32_t count) {
  return kind == sgpr ? builder.allocateSgpr(count, 2)
                      : builder.allocateVgpr(count, 2);
}

static void addInput(KernelInputs &inputs, InputKind kind, const Words &source,
                     int64_t laneCount, int64_t groupCount) {
  inputs.kinds.push_back(kind);
  int64_t count = kind == lane ? laneCount : groupCount;
  inputs.words.emplace_back(source.begin(), source.begin() + count);
}

// Core.

using IRBuilderTest = BuilderTest;

TEST_F(IRBuilderTest, EmitsKernelSequence) {
  RegisterRange kernarg = allocator.reserve(RegisterKind::sgpr, 0, 2);
  RegisterRange ptr = builder.allocateSgpr(2, 2);
  RegisterRange voff = builder.allocateVgpr(1);
  RegisterRange vx = builder.allocateVgpr(1);
  RegisterRange vy = builder.allocateVgpr(1);

  builder.createComment("saxpy body");
  builder.create<rocisa::SLoadB64>(ptr.container(), kernarg.container(), 0);
  builder.createWaitCnt({.scalarMemory = 0});
  std::shared_ptr<rocisa::Label> end = builder.createLabel("end");
  builder.create<rocisa::SCBranchExecZ>(end->getLabelName());
  builder.create<rocisa::GlobalLoadB32>(vx.container(), voff.container(),
                                        ptr.container());
  builder.createWaitCnt({.vectorLoads = 0});
  builder.create<rocisa::VMacF32>(vy.container(), vx.container(),
                                  vx.container());
  builder.create<rocisa::GlobalStoreB32>(voff.container(), vy.container(),
                                         ptr.container());
  builder.insert(end);
  builder.createEndProgram();

  expectLinesInOrder(body.toString(), {
                                          "// saxpy body",
                                          "s_load_dwordx2 s[2:3], s[0:1], 0",
                                          "s_waitcnt lgkmcnt(0)",
                                          "s_cbranch_execz label_end_0",
                                          "global_load_dword v1, v0, s[2:3]",
                                          "s_waitcnt vmcnt(0)",
                                          "v_fmac_f32 v2, v1, v1",
                                          "global_store_dword v0, v2, s[2:3]",
                                          "label_end_0:",
                                          "s_endpgm",
                                      });
}

TEST_F(IRBuilderTest, CreateReturnsInsertedInstance) {
  std::shared_ptr<rocisa::SEndpgm> inst = builder.create<rocisa::SEndpgm>();
  EXPECT_EQ(std::ssize(body.itemList), 1);
  EXPECT_EQ(body.itemList.back().get(),
            static_cast<rocisa::Item *>(inst.get()));
  EXPECT_EQ(inst->parent, &body);
}

TEST_F(IRBuilderTest, InsertionGuardRestoresInsertionPoint) {
  std::shared_ptr<rocisa::Module> nested =
      builder.create<rocisa::Module>("nested");
  {
    IRBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*nested);
    builder.createEndProgram();
    EXPECT_EQ(&builder.insertionPoint(), nested.get());
  }
  EXPECT_EQ(&builder.insertionPoint(), &body);
  builder.createEndProgram();
  EXPECT_EQ(std::ssize(nested->itemList), 1);
  EXPECT_EQ(std::ssize(body.itemList), 2);
}

TEST_F(IRBuilderTest, LabelNamesAreUnique) {
  std::shared_ptr<rocisa::Label> a = builder.createLabel("loop");
  std::shared_ptr<rocisa::Label> b = builder.createLabel("loop");
  std::shared_ptr<rocisa::Label> c = builder.createLabel("end");
  EXPECT_NE(a->getLabelName(), b->getLabelName());
  EXPECT_NE(a->getLabelName(), c->getLabelName());
  EXPECT_NE(b->getLabelName(), c->getLabelName());
  EXPECT_TRUE(body.itemList.empty());
}

TEST_F(IRBuilderTest, InsertRejectsItemAlreadyInModule) {
  std::shared_ptr<rocisa::Label> label = builder.createLabel("loop");
  builder.insert(label);
  EXPECT_THROW(builder.insert(label), std::logic_error);
  EXPECT_THROW(builder.insert(std::shared_ptr<rocisa::Label>()),
               std::invalid_argument);
}

TEST_F(IRBuilderTest, RangesReleaseOnScopeExit) {
  {
    RegisterRange v = builder.allocateVgpr(4);
    EXPECT_EQ(v.start(), 0);
    RegisterRange s = builder.allocateSgpr(2, 2);
    EXPECT_EQ(s.start(), 0);
  }
  EXPECT_EQ(builder.allocateVgpr(4).start(), 0);
  EXPECT_EQ(builder.allocateSgpr(2, 2).start(), 0);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 4);
}

TEST_F(IRBuilderTest, AllocatesAgprs) {
  RegisterRange value = builder.allocateVgpr(1);
  RegisterRange acc = builder.allocateAgpr(4);
  EXPECT_EQ(acc.kind(), RegisterKind::agpr);
  EXPECT_EQ(acc.start(), 0);
  RegisterRange aligned = builder.allocateAgpr(2, 2);
  EXPECT_EQ(aligned.start(), 4);
  builder.create<rocisa::VAccvgprWriteB32>(acc.sub(3), value.container());
  builder.create<rocisa::VAccvgprReadB32>(value.container(), acc.sub(3));
  std::string text = body.toString();
  EXPECT_NE(text.find("v_accvgpr_write_b32 acc3, v0"), std::string::npos)
      << text;
  EXPECT_NE(text.find("v_accvgpr_read_b32 v0, acc3"), std::string::npos)
      << text;
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::agpr), 6);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 1);
}

TEST_F(IRBuilderTest, SelectsTargetOnConstructingThread) {
  std::string text;
  std::thread worker([&] {
    rocisa::Module module("thread");
    IRBuilder threadBuilder(target, allocator, module);
    threadBuilder.create<rocisa::VMacF32>(rocisa::vgpr(2), rocisa::vgpr(1),
                                          rocisa::vgpr(1));
    text = module.toString();
  });
  worker.join();
  EXPECT_NE(text.find("v_fmac_f32 v2, v1, v1"), std::string::npos) << text;
}

// Operands.

TEST(OperandTest, ConstructsFromRegisters) {
  Target target = Target::get("gfx950");
  RegisterAllocator allocator(target);
  RegisterRange scalar = allocator.allocate(RegisterKind::sgpr, 1);
  RegisterRange range = allocator.allocate(RegisterKind::vgpr, 2);
  RegisterRange accumulators = allocator.allocate(RegisterKind::agpr, 4);
  Operand fromRange(range);
  EXPECT_TRUE(fromRange.isRegister());
  EXPECT_TRUE(fromRange.isRegister(RegisterKind::vgpr));
  EXPECT_FALSE(fromRange.isRegister(RegisterKind::sgpr));
  EXPECT_FALSE(fromRange.isImmediate());
  EXPECT_EQ(fromRange.reg(), range.ref());

  Operand fromRef(range.ref(1));
  EXPECT_EQ(fromRef.reg(), (RegisterRef{RegisterKind::vgpr, 1, 1}));

  // The register is read back from its container for every kind and width.
  Operand fromScalar(scalar);
  EXPECT_TRUE(fromScalar.isRegister(RegisterKind::sgpr));
  EXPECT_FALSE(fromScalar.isRegister(RegisterKind::agpr));
  EXPECT_EQ(fromScalar.reg(), scalar.ref());
  Operand fromAccumulators(accumulators);
  EXPECT_TRUE(fromAccumulators.isRegister(RegisterKind::agpr));
  EXPECT_FALSE(fromAccumulators.isRegister(RegisterKind::vgpr));
  EXPECT_EQ(fromAccumulators.reg(), accumulators.ref());
}

TEST(OperandTest, ConstructsFromIntegers) {
  Operand fromInt(5);
  EXPECT_TRUE(fromInt.isImmediate());
  EXPECT_FALSE(fromInt.isRegister());
  EXPECT_FALSE(fromInt.isRegister(RegisterKind::vgpr));
  EXPECT_EQ(fromInt.immediate(), 5);
  EXPECT_EQ(Operand(-1).immediate(), -1);
  EXPECT_EQ(Operand(INT32_MIN).immediate(), INT32_MIN);
  EXPECT_EQ(Operand(INT32_MAX).immediate(), INT32_MAX);
}

// Wide and unsigned integers are rejected at compile time, so a constant never
// truncates silently.
static_assert(!std::is_constructible_v<Operand, int64_t>);
static_assert(!std::is_constructible_v<Operand, uint32_t>);
static_assert(!std::is_constructible_v<Operand, uint64_t>);
static_assert(std::is_constructible_v<Operand, int32_t>);
static_assert(sizeof(Operand) == sizeof(InstructionInput));

// Moves.

namespace {

class MoveTest : public BuilderTest {
protected:
  std::string emitMov32(RegisterRef dst, Operand src) {
    return emitted([&](IRBuilder &local) { local.movb32(dst, src); });
  }
  std::string emitMov64(RegisterRef dst, Operand src) {
    return emitted([&](IRBuilder &local) { local.movb64(dst, src); });
  }
  int32_t moveInstructionCount(const std::string &out) {
    return countLinesContaining(out, "_mov_") +
           countLinesContaining(out, "accvgpr");
  }
};

struct PlacementCounts {
  int32_t assembled = 0;
  int32_t rejected = 0;
};

} // namespace

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
    EXPECT_EQ(moveInstructionCount(scalar), 1);
    std::string vector = emitMov64(vdst.ref(), current.value);
    expectLinesInOrder(vector,
                       {line("v_mov_b64", text(vdst), current.printed)});
    EXPECT_EQ(moveInstructionCount(vector), 1);
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
  EXPECT_EQ(moveInstructionCount(negative), 2);

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
  EXPECT_EQ(moveInstructionCount(toMisaligned), 2);
  std::string fromMisaligned = emitMov64(vdst.ref(), vrange.ref(1, 2));
  EXPECT_EQ(countLinesContaining(fromMisaligned, "v_mov_b64"), 0);
  EXPECT_EQ(moveInstructionCount(fromMisaligned), 2);
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

static constexpr int32_t moveImmediates32[] = {0,  -1,        64,
                                               65, INT32_MIN, 0x12345678};

using MoveDevice = DeviceTest;

static RegisterRef halfOf(RegisterRef pair, int32_t index) {
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
                     laneGroupCount(pairs.a), {pairs.a}, pairs.a);
}

TEST_F(MoveDevice, Movb32UniformToVgprAndSgpr) {
  WordPairs pairs = makeArithPairs(0);
  int32_t groups = static_cast<int32_t>(pairs.a.size());
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
  for (int32_t immediate : moveImmediates32) {
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
  int32_t groups = static_cast<int32_t>(pairs.a.size());
  ArithEmitter viaAgpr = [](IRBuilder &builder, RegisterRef dst,
                            const std::vector<RegisterRef> &inputs) {
    RegisterRange acc = builder.allocateAgpr(1);
    builder.movb32(acc.ref(), inputs[0]);
    builder.movb32(dst, acc.ref());
  };
  SCOPED_TRACE("lane vgpr to agpr");
  expectArithResults({{InputKind::lane}, vgpr, viaAgpr},
                     laneGroupCount(pairs.a), {pairs.a}, pairs.a);
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

using PairEmitter = std::function<PairRegisters(
    IRBuilder &builder, const std::vector<RegisterRef> &inputs)>;

// A 64-bit value is checked through two runs, one storing each half.
static void expectPairResults(const std::vector<InputKind> &kinds,
                              RegisterKind result, const PairEmitter &emit,
                              int32_t workgroupCount,
                              const std::vector<std::vector<uint32_t>> &inputs,
                              const std::vector<uint32_t> &low,
                              const std::vector<uint32_t> &high) {
  for (int32_t half = 0; half < 2; ++half) {
    SCOPED_TRACE(half == 0 ? "low half" : "high half");
    ArithEmitter storeHalf = [&emit, half](IRBuilder &builder, RegisterRef dst,
                                           const std::vector<RegisterRef> &in) {
      PairRegisters value = emit(builder, in);
      builder.movb32(dst, halfOf(value.pair, half));
    };
    expectArithResults({kinds, result, storeHalf}, workgroupCount, inputs,
                       half == 0 ? low : high);
  }
}

TEST_F(MoveDevice, Movb64Immediates) {
  for (int32_t immediate : immediates64) {
    for (RegisterKind kind : {sgpr, vgpr}) {
      SCOPED_TRACE("immediate " + std::to_string(immediate) +
                   (kind == sgpr ? " to sgpr pair" : " to vgpr pair"));
      PairEmitter emit = [immediate, kind](IRBuilder &builder,
                                           const std::vector<RegisterRef> &) {
        RegisterRange pair = allocateKind(builder, kind, 2);
        RegisterRef ref = pair.ref();
        builder.movb64(ref, immediate);
        return PairRegisters{std::move(pair), ref};
      };
      // The expected halves come from the sign-extended host value, which
      // proves that a negative literal is split and not zero-extended.
      int64_t extended = immediate;
      uint32_t low = static_cast<uint32_t>(extended);
      uint32_t high = static_cast<uint32_t>(extended >> 32);
      int32_t words = kind == sgpr ? 3 : 2 * arithWorkgroupSize;
      expectPairResults({}, kind, emit, kind == sgpr ? 3 : 2, {},
                        std::vector<uint32_t>(words, low),
                        std::vector<uint32_t>(words, high));
    }
  }
}

// Fills a pair from two inputs with 32-bit moves.
static RegisterRange fillPair(IRBuilder &builder, RegisterKind kind,
                              RegisterRef low, RegisterRef high) {
  RegisterRange pair = allocateKind(builder, kind, 2);
  builder.movb32(halfOf(pair.ref(), 0), low);
  builder.movb32(halfOf(pair.ref(), 1), high);
  return pair;
}

TEST_F(MoveDevice, Movb64VgprPairsAlignedAndMisaligned) {
  WordPairs pairs = makeArithPairs(0);
  int32_t groups = laneGroupCount(pairs.a);
  std::vector<InputKind> kinds{InputKind::lane, InputKind::lane};
  SCOPED_TRACE("aligned to aligned");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, vgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateVgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref());
        return PairRegisters{std::move(dst), ref};
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
        return PairRegisters{std::move(dst), ref};
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
        return PairRegisters{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
}

TEST_F(MoveDevice, Movb64OverlappingMisalignedCopies) {
  WordPairs pairs = makeArithPairs(0);
  int32_t groups = laneGroupCount(pairs.a);
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
        return PairRegisters{std::move(range), ref};
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
        return PairRegisters{std::move(range), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
}

TEST_F(MoveDevice, Movb64ScalarPairs) {
  WordPairs pairs = makeArithPairs(0);
  int32_t groups = static_cast<int32_t>(pairs.a.size());
  std::vector<InputKind> kinds{InputKind::uniform, InputKind::uniform};
  SCOPED_TRACE("sgpr pair to sgpr pair");
  expectPairResults(
      kinds, sgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, sgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateSgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref());
        return PairRegisters{std::move(dst), ref};
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
        return PairRegisters{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, perLane(pairs.a), perLane(pairs.b));
}

// Placements of a source, checked by the assembler alone.
static Operand registerSource(RegisterKind kind, RegisterRange &sgprs,
                              RegisterRange &vgprs, RegisterRange &agprs,
                              int32_t width, int32_t offset) {
  RegisterRange &range = kind == sgpr ? sgprs : kind == vgpr ? vgprs : agprs;
  return range.ref(offset, width);
}

// Emits dst = src in a fresh kernel and assembles it. A combination the
// builder rejects is counted and skipped.
static void
assemblePlacement(RegisterKind dstKind, int32_t width, int32_t dstOffset,
                  const std::function<Operand(RegisterRange &, RegisterRange &,
                                              RegisterRange &)> &makeSource,
                  PlacementCounts &counts) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_move_placement", arithWorkgroupSize,
                KernelSignature());
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
    for (int32_t width : {1, 2}) {
      for (int32_t offset : {0, 1}) {
        for (RegisterKind srcKind : {sgpr, vgpr, agpr})
          for (int32_t srcOffset : {0, 1})
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

// 32-bit arithmetic.

namespace {

struct OpCase32 {
  const char *name;
  BuilderOp op;
  std::string_view valuMnemonic;
  std::string_view saluMnemonic;
  uint32_t (*host)(uint32_t, uint32_t);
};

class ArithTest : public BuilderTest {
protected:
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

// Operand placements, checked by the assembler alone so no device is needed.
enum class APlacement { vgpr, sgpr, inlineConstant, literal };

enum class BPlacement {
  vgpr,
  otherSgpr,
  sameSgpr,
  inlineConstant,
  literal,
  sameLiteral,
  negativeLiteral,
  minInt32,
  one,
  eight,
};

struct PlacementRegisters {
  RegisterRange vectorA;
  RegisterRange vectorB;
  RegisterRange scalarA;
  RegisterRange scalarB;
};

} // namespace

// Builds "mnemonic op0, op1, ...".
static std::string inst(std::string_view mnemonic,
                        const std::vector<std::string> &operands) {
  std::string result(mnemonic);
  for (int64_t i = 0; i < std::ssize(operands); ++i)
    result += (i == 0 ? " " : ", ") + operands[i];
  return result;
}

// Checks that text holds exactly the expected instructions, so adjacency is
// checked too. A line may carry a trailing comment after its operands.
static void expectInstructions(const std::string &text,
                               const std::vector<std::string> &expected) {
  std::vector<std::string> lines = instructionsOf(text);
  ASSERT_EQ(lines.size(), expected.size()) << text;
  for (int64_t i = 0; i < std::ssize(expected); ++i) {
    const std::string &actual = lines[i];
    bool matches = actual.rfind(expected[i], 0) == 0 &&
                   (actual.size() == expected[i].size() ||
                    actual[expected[i].size()] == ' ');
    EXPECT_TRUE(matches) << "line " << i << ": expected '" << expected[i]
                         << "', got '" << actual << "'";
  }
}

static constexpr OpCase32 opCases32[] = {
    {"addi32", &IRBuilder::addi32, "v_add_u32", "s_add_u32",
     [](uint32_t a, uint32_t b) { return a + b; }},
    {"subi32", &IRBuilder::subi32, "v_sub_u32", "s_sub_u32",
     [](uint32_t a, uint32_t b) { return a - b; }},
    {"muli32", &IRBuilder::muli32, "v_mul_lo_u32", "s_mul_i32",
     [](uint32_t a, uint32_t b) { return a * b; }},
    {"mulhsi32", &IRBuilder::mulhsi32, "v_mul_hi_i32", "s_mul_hi_i32",
     [](uint32_t a, uint32_t b) {
       int64_t product =
           int64_t{static_cast<int32_t>(a)} * int64_t{static_cast<int32_t>(b)};
       return static_cast<uint32_t>(product >> 32);
     }},
    {"minsi32", &IRBuilder::minsi32, "v_min_i32", "s_min_i32",
     [](uint32_t a, uint32_t b) {
       return static_cast<uint32_t>(
           std::min(static_cast<int32_t>(a), static_cast<int32_t>(b)));
     }},
    {"maxsi32", &IRBuilder::maxsi32, "v_max_i32", "s_max_i32",
     [](uint32_t a, uint32_t b) {
       return static_cast<uint32_t>(
           std::max(static_cast<int32_t>(a), static_cast<int32_t>(b)));
     }},
    {"minui32", &IRBuilder::minui32, "v_min_u32", "s_min_u32",
     [](uint32_t a, uint32_t b) { return std::min(a, b); }},
    {"maxui32", &IRBuilder::maxui32, "v_max_u32", "s_max_u32",
     [](uint32_t a, uint32_t b) { return std::max(a, b); }},
};

TEST_F(ArithTest, SaluForScalarOperands) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  for (const OpCase32 &current : opCases32) {
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
  for (const OpCase32 &current : opCases32) {
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
  for (const OpCase32 &current : opCases32) {
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
    int32_t value;
    std::string printed;
    int32_t copies;
  };
  for (const Case &current :
       {Case{63, "63", 0}, Case{-16, "-16", 0}, Case{-1, "-1", 0},
        Case{65, "65", 1}, Case{-17, "-17", 1}}) {
    SCOPED_TRACE(current.printed);
    std::string out = emitMul(dst.ref(), v1, current.value);
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), current.copies);
    // The value is printed once, in the mul or in its copy.
    EXPECT_EQ(countLinesContaining(out, ", " + current.printed), 1);
  }
}

TEST_F(ArithTest, MulByPowerOfTwoUsesShift) {
  RegisterRange sd = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange vd = builder.allocateVgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  expectInstructions(emitMul(sd.ref(), sa, 8),
                     {inst("s_lshl_b32", {text(sd), text(sa), "3"})});
  expectInstructions(emitMul(vd.ref(), va, 8),
                     {inst("v_lshlrev_b32", {text(vd), "3", text(va)})});
  // The power of two may be the first operand.
  expectInstructions(emitMul(sd.ref(), 8, sa),
                     {inst("s_lshl_b32", {text(sd), text(sa), "3"})});
  expectInstructions(emitMul(vd.ref(), 8, va),
                     {inst("v_lshlrev_b32", {text(vd), "3", text(va)})});
  // A scalar a into a vector dst needs no copy.
  expectInstructions(emitMul(vd.ref(), sa, 8),
                     {inst("v_lshlrev_b32", {text(vd), "3", text(sa)})});
  expectInstructions(emitMul(sd.ref(), sa, 1),
                     {inst("s_mov_b32", {text(sd), text(sa)})});
  expectInstructions(emitMul(vd.ref(), va, 1),
                     {inst("v_mov_b32", {text(vd), text(va)})});
  // INT32_MIN is 2^31 modulo 2^32.
  expectInstructions(emitMul(sd.ref(), sa, INT32_MIN),
                     {inst("s_lshl_b32", {text(sd), text(sa), "31"})});
  expectInstructions(emitMul(vd.ref(), va, INT32_MIN),
                     {inst("v_lshlrev_b32", {text(vd), "31", text(va)})});
}

TEST_F(ArithTest, MulByNonPowerOfTwoKeepsMultiply) {
  RegisterRange sd = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange vd = builder.allocateVgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  for (int32_t value : {3, -4, 0, 0x12345678}) {
    SCOPED_TRACE(value);
    EXPECT_EQ(countLinesContaining(emitMul(sd.ref(), sa, value), "s_mul_i32"),
              1);
    EXPECT_EQ(
        countLinesContaining(emitMul(vd.ref(), va, value), "v_mul_lo_u32"), 1);
  }
}

TEST_F(ArithTest, MulByOneIsMoveOrNothing) {
  RegisterRange sd = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange vd = builder.allocateVgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  // The value is dst itself, in either operand: nothing to emit.
  expectInstructions(emitMul(sa.ref(), sa, 1), {});
  expectInstructions(emitMul(sa.ref(), 1, sa), {});
  expectInstructions(emitMul(va.ref(), va, 1), {});
  expectInstructions(emitMul(va.ref(), 1, va), {});
  // Another register or an immediate is a move.
  expectInstructions(emitMul(sd.ref(), 1, sa),
                     {inst("s_mov_b32", {text(sd), text(sa)})});
  expectInstructions(emitMul(vd.ref(), 1, va),
                     {inst("v_mov_b32", {text(vd), text(va)})});
  expectInstructions(emitMul(vd.ref(), sa, 1),
                     {inst("v_mov_b32", {text(vd), text(sa)})});
  expectInstructions(emitMul(sd.ref(), literalA, 1),
                     {inst("s_mov_b32", {text(sd), literalAText})});
  expectInstructions(emitMul(vd.ref(), 1, 1),
                     {inst("v_mov_b32", {text(vd), "1"})});
  // A scalar dst equal to a vector operand is still rejected.
  EXPECT_THROW(builder.muli32(sa.ref(), va, 1), std::invalid_argument);
  // The other powers of two still shift in place.
  expectInstructions(emitMul(sa.ref(), sa, 2),
                     {inst("s_lshl_b32", {text(sa), text(sa), "1"})});
}

TEST_F(ArithTest, MulHighSignedForms) {
  RegisterRange sd = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange sb = builder.allocateSgpr(1);
  RegisterRange vd = builder.allocateVgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  auto emitMulHigh = [&](RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::mulhsi32, dst, a, b);
  };
  expectInstructions(emitMulHigh(sd.ref(), sa, sb),
                     {inst("s_mul_hi_i32", {text(sd), text(sa), text(sb)})});
  // SOP2 takes one literal, in either operand.
  expectInstructions(
      emitMulHigh(sd.ref(), sa, literalA),
      {inst("s_mul_hi_i32", {text(sd), text(sa), literalAText})});
  expectInstructions(
      emitMulHigh(sd.ref(), literalA, sa),
      {inst("s_mul_hi_i32", {text(sd), literalAText, text(sa)})});
  expectInstructions(emitMulHigh(vd.ref(), va, sa),
                     {inst("v_mul_hi_i32", {text(vd), text(va), text(sa)})});
  // VOP3 takes no literal, and no mul_hi shortcut applies to powers of two.
  expectInstructions(emitMulHigh(vd.ref(), va, literalA),
                     {inst("v_mov_b32", {"v2", literalAText}),
                      inst("v_mul_hi_i32", {text(vd), text(va), "v2"})});
  expectInstructions(emitMulHigh(vd.ref(), va, 8),
                     {inst("v_mul_hi_i32", {text(vd), text(va), "8"})});
  // Two distinct SGPRs exceed the constant bus.
  expectInstructions(emitMulHigh(vd.ref(), sa, sb),
                     {inst("v_mov_b32", {"v2", text(sb)}),
                      inst("v_mul_hi_i32", {text(vd), text(sa), "v2"})});
}

TEST_F(ArithTest, MinMaxLiteralAndScalarLegalization) {
  RegisterRange sd = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange sb = builder.allocateSgpr(1);
  RegisterRange vd = builder.allocateVgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  struct Case {
    BuilderOp op;
    std::string salu;
    std::string valu;
  };
  for (const Case &current :
       {Case{&IRBuilder::minsi32, "s_min_i32", "v_min_i32"},
        Case{&IRBuilder::maxsi32, "s_max_i32", "v_max_i32"},
        Case{&IRBuilder::minui32, "s_min_u32", "v_min_u32"},
        Case{&IRBuilder::maxui32, "s_max_u32", "v_max_u32"}}) {
    SCOPED_TRACE(current.valu);
    expectInstructions(
        emitOp(current.op, sd.ref(), sa, literalA),
        {inst(current.salu, {text(sd), text(sa), literalAText})});
    // VOP2 takes a literal in src0, so the commutative swap needs no copy.
    expectInstructions(
        emitOp(current.op, vd.ref(), va, literalA),
        {inst(current.valu, {text(vd), literalAText, text(va)})});
    expectInstructions(emitOp(current.op, vd.ref(), sa, va),
                       {inst(current.valu, {text(vd), text(sa), text(va)})});
    // Two distinct SGPRs exceed the constant bus, so one is copied.
    expectInstructions(emitOp(current.op, vd.ref(), sa, sb),
                       {inst("v_mov_b32", {"v2", text(sb)}),
                        inst(current.valu, {text(vd), text(sa), "v2"})});
  }
}

TEST_F(ArithTest, MinMaxInvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s1 = builder.allocateSgpr(1);
  RegisterRange acc = builder.allocateAgpr(1);
  for (BuilderOp op :
       {&IRBuilder::mulhsi32, &IRBuilder::minsi32, &IRBuilder::maxsi32,
        &IRBuilder::minui32, &IRBuilder::maxui32}) {
    EXPECT_THROW((builder.*op)(sdst.ref(), v1, s1), std::invalid_argument);
    EXPECT_THROW((builder.*op)(acc.ref(), s1, s1), std::invalid_argument);
    EXPECT_THROW((builder.*op)(v1.ref(), acc, s1), std::invalid_argument);
  }
  EXPECT_EQ(body.toString(), "");
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
  EXPECT_EQ(body.toString(), "");
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 4);
}

TEST_F(ArithTest, BoundaryImmediatesAreAccepted) {
  RegisterRange vdst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  EXPECT_NO_THROW(builder.addi32(vdst.ref(), v1, INT32_MIN));
  EXPECT_NO_THROW(builder.addi32(vdst.ref(), v1, INT32_MAX));
}

// Immediates that cover inline constants, their edges, literals and wrapping.
static constexpr int32_t arithImmediates[] = {
    0, 1, -1, 64, 65, -16, -17, INT32_MIN, INT32_MAX, 0x12345678};

using ArithDevice = DeviceTest;

static ArithKernelSpec binarySpec(BuilderOp op, InputKind aKind,
                                  InputKind bKind, RegisterKind result) {
  return {{aKind, bKind},
          result,
          [op](IRBuilder &builder, RegisterRef dst,
               const std::vector<RegisterRef> &inputs) {
            (builder.*op)(dst, inputs[0], inputs[1]);
          }};
}

// Computes dst = input (kind) op immediate, or immediate op input.
static ArithKernelSpec immediateSpec(BuilderOp op, InputKind kind,
                                     RegisterKind result, int32_t immediate,
                                     bool immediateFirst) {
  return {
      {kind},
      result,
      [op, immediate, immediateFirst](IRBuilder &builder, RegisterRef dst,
                                      const std::vector<RegisterRef> &inputs) {
        if (immediateFirst)
          (builder.*op)(dst, immediate, inputs[0]);
        else
          (builder.*op)(dst, inputs[0], immediate);
      }};
}

static std::vector<uint32_t> applyImmediate(const OpCase32 &op,
                                            const std::vector<uint32_t> &words,
                                            int32_t immediate,
                                            bool immediateFirst) {
  uint32_t value = static_cast<uint32_t>(immediate);
  std::vector<uint32_t> out;
  for (uint32_t word : words)
    out.push_back(immediateFirst ? op.host(value, word) : op.host(word, value));
  return out;
}

static void expectValuLaneOperands(std::span<const OpCase32> ops,
                                   const WordPairs &pairs) {
  for (const OpCase32 &op : ops) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> expected;
    for (int64_t i = 0; i < std::ssize(pairs.a); ++i)
      expected.push_back(op.host(pairs.a[i], pairs.b[i]));
    expectArithResults(
        binarySpec(op.op, InputKind::lane, InputKind::lane, RegisterKind::vgpr),
        laneGroupCount(pairs.a), {pairs.a, pairs.b}, expected);
  }
}

static void expectSaluUniformOperands(std::span<const OpCase32> ops,
                                      const WordPairs &pairs) {
  for (const OpCase32 &op : ops) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> expected;
    for (int64_t i = 0; i < std::ssize(pairs.a); ++i)
      expected.push_back(op.host(pairs.a[i], pairs.b[i]));
    expectArithResults(binarySpec(op.op, InputKind::uniform, InputKind::uniform,
                                  RegisterKind::sgpr),
                       static_cast<int32_t>(pairs.a.size()), {pairs.a, pairs.b},
                       expected);
  }
}

static void expectScalarsIntoVgpr(std::span<const OpCase32> ops,
                                  const WordPairs &pairs) {
  for (const OpCase32 &op : ops) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> perGroup;
    for (int64_t i = 0; i < std::ssize(pairs.a); ++i)
      perGroup.push_back(op.host(pairs.a[i], pairs.b[i]));
    expectArithResults(binarySpec(op.op, InputKind::uniform, InputKind::uniform,
                                  RegisterKind::vgpr),
                       static_cast<int32_t>(pairs.a.size()), {pairs.a, pairs.b},
                       perLane(perGroup));
  }
}

static void expectMixedScalarAndVector(std::span<const OpCase32> ops,
                                       const WordPairs &pairs) {
  int32_t groups = laneGroupCount(pairs.a);
  std::vector<uint32_t> edges = arithEdgeValues();
  std::vector<uint32_t> uniformWords;
  for (int32_t g = 0; g < groups; ++g)
    uniformWords.push_back(edges[g % edges.size()]);
  for (const OpCase32 &op : ops) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> uniformFirst;
    std::vector<uint32_t> uniformSecond;
    for (int64_t i = 0; i < std::ssize(pairs.b); ++i) {
      uint32_t scalar = uniformWords[i / arithWorkgroupSize];
      uniformFirst.push_back(op.host(scalar, pairs.b[i]));
      uniformSecond.push_back(op.host(pairs.b[i], scalar));
    }
    expectArithResults(binarySpec(op.op, InputKind::uniform, InputKind::lane,
                                  RegisterKind::vgpr),
                       groups, {uniformWords, pairs.b}, uniformFirst);
    expectArithResults(binarySpec(op.op, InputKind::lane, InputKind::uniform,
                                  RegisterKind::vgpr),
                       groups, {pairs.b, uniformWords}, uniformSecond);
  }
}

static void expectImmediateOperands(std::span<const OpCase32> ops,
                                    const WordPairs &pairs,
                                    std::span<const int32_t> immediates) {
  int32_t groups = laneGroupCount(pairs.a);
  int32_t scalarGroups = static_cast<int32_t>(pairs.a.size());
  for (const OpCase32 &op : ops) {
    for (int32_t immediate : immediates) {
      SCOPED_TRACE(std::string(op.name) + " immediate " +
                   std::to_string(immediate));
      expectArithResults(immediateSpec(op.op, InputKind::lane,
                                       RegisterKind::vgpr, immediate, false),
                         groups, {pairs.a},
                         applyImmediate(op, pairs.a, immediate, false));
      expectArithResults(immediateSpec(op.op, InputKind::lane,
                                       RegisterKind::vgpr, immediate, true),
                         groups, {pairs.b},
                         applyImmediate(op, pairs.b, immediate, true));
      expectArithResults(immediateSpec(op.op, InputKind::uniform,
                                       RegisterKind::sgpr, immediate, false),
                         scalarGroups, {pairs.a},
                         applyImmediate(op, pairs.a, immediate, false));
      expectArithResults(
          immediateSpec(op.op, InputKind::uniform, RegisterKind::vgpr,
                        immediate, false),
          scalarGroups, {pairs.a},
          perLane(applyImmediate(op, pairs.a, immediate, false)));
    }
  }
}

TEST_F(ArithDevice, ValuLaneOperands) {
  expectValuLaneOperands(opCases32, makeArithPairs(4096));
}

TEST_F(ArithDevice, SaluUniformOperands) {
  expectSaluUniformOperands(opCases32, makeArithPairs(256));
}

TEST_F(ArithDevice, ScalarsIntoVgpr) {
  expectScalarsIntoVgpr(opCases32, makeArithPairs(256));
}

TEST_F(ArithDevice, MixedScalarAndVector) {
  expectMixedScalarAndVector(opCases32, makeArithPairs(4096));
}

TEST_F(ArithDevice, ImmediateOperands) {
  expectImmediateOperands(opCases32, makeArithPairs(0), arithImmediates);
}

TEST_F(ArithDevice, MulByPowerOfTwo) {
  std::span<const OpCase32> mul(&opCases32[2], 1);
  ASSERT_STREQ(mul[0].name, "muli32");
  std::vector<int32_t> powers;
  for (int32_t exponent = 0; exponent <= 31; ++exponent)
    powers.push_back(static_cast<int32_t>(1u << exponent));
  expectImmediateOperands(mul, makeArithPairs(256), powers);
}

// Each emitter writes into an input register and then moves it to dst.
TEST_F(ArithDevice, DestinationAliasesOperand) {
  WordPairs pairs = makeArithPairs(256);
  int32_t groups = laneGroupCount(pairs.a);
  int32_t scalarGroups = static_cast<int32_t>(pairs.a.size());
  std::vector<uint32_t> expected;

  SCOPED_TRACE("vgpr subi32 into second input");
  for (int64_t i = 0; i < std::ssize(pairs.a); ++i)
    expected.push_back(pairs.a[i] - pairs.b[i]);
  expectArithResults({{InputKind::lane, InputKind::lane},
                      RegisterKind::vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.subi32(inputs[1], inputs[0], inputs[1]);
                        builder.movb32(dst, inputs[1]);
                      }},
                     groups, {pairs.a, pairs.b}, expected);

  SCOPED_TRACE("vgpr muli32 into first input");
  expected.clear();
  for (uint32_t a : pairs.a)
    expected.push_back(a * 0x12345678u);
  expectArithResults({{InputKind::lane},
                      RegisterKind::vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.muli32(inputs[0], inputs[0], 0x12345678);
                        builder.movb32(dst, inputs[0]);
                      }},
                     groups, {pairs.a}, expected);

  SCOPED_TRACE("vgpr addi32 of scalar and lane into lane");
  std::vector<uint32_t> uniformWords(pairs.a.begin(), pairs.a.begin() + groups);
  expected.clear();
  for (int64_t i = 0; i < std::ssize(pairs.b); ++i)
    expected.push_back(uniformWords[i / arithWorkgroupSize] + pairs.b[i]);
  expectArithResults({{InputKind::uniform, InputKind::lane},
                      RegisterKind::vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.addi32(inputs[1], inputs[0], inputs[1]);
                        builder.movb32(dst, inputs[1]);
                      }},
                     groups, {uniformWords, pairs.b}, expected);

  SCOPED_TRACE("sgpr subi32 into first input");
  expected.clear();
  for (int64_t i = 0; i < std::ssize(pairs.a); ++i)
    expected.push_back(pairs.a[i] - pairs.b[i]);
  expectArithResults({{InputKind::uniform, InputKind::uniform},
                      RegisterKind::sgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.subi32(inputs[0], inputs[0], inputs[1]);
                        builder.movb32(dst, inputs[0]);
                      }},
                     scalarGroups, {pairs.a, pairs.b}, expected);
}

static constexpr int32_t inlineA = 7;
static constexpr int32_t inlineB = -5;
static constexpr int32_t negativeLiteralB = -100;

static const char *placementName(APlacement placement) {
  switch (placement) {
  case APlacement::vgpr:
    return "vgpr";
  case APlacement::sgpr:
    return "sgpr";
  case APlacement::inlineConstant:
    return "inline 7";
  case APlacement::literal:
    return "literal 0x12345";
  }
  return "";
}

static const char *placementName(BPlacement placement) {
  switch (placement) {
  case BPlacement::vgpr:
    return "vgpr";
  case BPlacement::otherSgpr:
    return "other sgpr";
  case BPlacement::sameSgpr:
    return "same sgpr as a";
  case BPlacement::inlineConstant:
    return "inline -5";
  case BPlacement::literal:
    return "literal 0x54321";
  case BPlacement::sameLiteral:
    return "same literal as a";
  case BPlacement::negativeLiteral:
    return "literal -100";
  case BPlacement::minInt32:
    return "INT32_MIN";
  case BPlacement::one:
    return "2^0";
  case BPlacement::eight:
    return "2^3";
  }
  return "";
}

static Operand pickA(APlacement placement, const PlacementRegisters &regs) {
  switch (placement) {
  case APlacement::vgpr:
    return regs.vectorA;
  case APlacement::sgpr:
    return regs.scalarA;
  case APlacement::inlineConstant:
    return inlineA;
  case APlacement::literal:
    return literalA;
  }
  return 0;
}

static Operand pickB(BPlacement placement, const PlacementRegisters &regs) {
  switch (placement) {
  case BPlacement::vgpr:
    return regs.vectorB;
  case BPlacement::otherSgpr:
    return regs.scalarB;
  case BPlacement::sameSgpr:
    return regs.scalarA;
  case BPlacement::inlineConstant:
    return inlineB;
  case BPlacement::literal:
    return literalB;
  case BPlacement::sameLiteral:
    return literalA;
  case BPlacement::negativeLiteral:
    return negativeLiteralB;
  case BPlacement::minInt32:
    return INT32_MIN;
  case BPlacement::one:
    return 1;
  case BPlacement::eight:
    return 8;
  }
  return 0;
}

static bool isCompatible(APlacement a, BPlacement b) {
  if (b == BPlacement::sameSgpr)
    return a == APlacement::sgpr;
  if (b == BPlacement::sameLiteral)
    return a == APlacement::literal;
  return true;
}

// Emits the op under test over two operands.
using PlacementEmitter = std::function<void(IRBuilder &builder, RegisterRef dst,
                                            Operand a, Operand b)>;

static void expectPlacementAssembles(const std::string &label,
                                     const PlacementEmitter &emit, APlacement a,
                                     BPlacement b, RegisterKind dstKind) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_placement", arithWorkgroupSize,
                KernelSignature());
  IRBuilder &builder = kernel.builder();
  PlacementRegisters regs{builder.allocateVgpr(1), builder.allocateVgpr(1),
                          builder.allocateSgpr(1), builder.allocateSgpr(1)};
  RegisterRange dst = dstKind == RegisterKind::vgpr ? builder.allocateVgpr(1)
                                                    : builder.allocateSgpr(1);
  emit(builder, dst.ref(), pickA(a, regs), pickB(b, regs));
  builder.createEndProgram();

  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(label + " a=" + placementName(a) + " b=" + placementName(b) +
               " dst=" + (dstKind == RegisterKind::vgpr ? "vgpr" : "sgpr") +
               "\n" + assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

static void expectAllPlacementsAssemble(const std::string &label,
                                        const PlacementEmitter &emit) {
  for (APlacement a : {APlacement::vgpr, APlacement::sgpr,
                       APlacement::inlineConstant, APlacement::literal})
    for (BPlacement b :
         {BPlacement::vgpr, BPlacement::otherSgpr, BPlacement::sameSgpr,
          BPlacement::inlineConstant, BPlacement::literal,
          BPlacement::sameLiteral, BPlacement::negativeLiteral,
          BPlacement::minInt32, BPlacement::one, BPlacement::eight}) {
      if (!isCompatible(a, b))
        continue;
      expectPlacementAssembles(label, emit, a, b, RegisterKind::vgpr);
      // A scalar destination cannot take a vector operand.
      if (a == APlacement::vgpr || b == BPlacement::vgpr)
        continue;
      expectPlacementAssembles(label, emit, a, b, RegisterKind::sgpr);
    }
}

TEST(ArithAssembly, AllOperandPlacementsAssemble) {
  for (const OpCase32 &op : opCases32)
    expectAllPlacementsAssemble(
        op.name, [&op](IRBuilder &builder, RegisterRef dst, Operand a,
                       Operand b) { (builder.*op.op)(dst, a, b); });
}

// 64-bit arithmetic.

namespace {

struct OpCase64 {
  const char *name;
  BuilderOp op;
  uint64_t (*host)(uint64_t, uint64_t);
  // b is a single-register shift amount rather than a pair.
  bool amountOperand = false;
};

class Arith64Test : public BuilderTest {
protected:
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

// How the destination sits relative to the operand registers.
enum class Arith64Placement {
  // The destination and the operands are separate registers.
  separate,
  // The destination is a or b.
  destinationIsA,
  destinationIsB,
  // The destination's low register is a's high register.
  lowIsAHigh,
  // The destination's high register is a's low register.
  highIsALow,
  // a and b are the same register pair.
  sameOperand,
  // The destination's low register holds the shift amount.
  amountInDestination,
};

struct Case64 {
  OpCase64 op;
  RegisterKind dstKind;
  InputKind aKind;
  InputKind bKind;
  Arith64Placement placement = Arith64Placement::separate;
  bool dstMisaligned = false;
  bool operandsMisaligned = false;
  // An operand with no immediate is a register pair built from two inputs.
  std::optional<int32_t> a;
  std::optional<int32_t> b;
};

struct Data64 {
  Values a;
  Values b;
};

// The registers that stay allocated while the op under test runs.
struct Emitted {
  std::vector<RegisterRange> owners;
  RegisterRef value;
};

struct NamedOperand {
  std::string name;
  Operand operand;
};

// The registers one operand can live in, aligned or at an odd index.
struct Arith64PlacementRegisters {
  RegisterRange vector;
  RegisterRange oddVector;
  RegisterRange scalar;
  RegisterRange oddScalar;
};

} // namespace

// The host arithmetic wraps modulo 2^64 like the device.
static constexpr OpCase64 opCases64[] = {
    {"addi64", &IRBuilder::addi64,
     [](uint64_t a, uint64_t b) { return a + b; }},
    {"subi64", &IRBuilder::subi64,
     [](uint64_t a, uint64_t b) { return a - b; }},
    {"muli64", &IRBuilder::muli64,
     [](uint64_t a, uint64_t b) { return a * b; }},
};

static constexpr int32_t otherLiteral = 0x0FEDCBA9;

static int32_t instructionCount(const std::string &text) {
  return static_cast<int32_t>(instructionsOf(text).size());
}

static std::string half(RegisterRef pair, int32_t index) {
  return text(pair.slice(index));
}

// The ops are built from 32-bit instructions, so no 64-bit form is emitted.
static void expectNoWideForms(const std::string &out) {
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

TEST_F(Arith64Test, MulByPowerOfTwoUsesShift) {
  RegisterRange sd = sgprPair();
  RegisterRange sa = sgprPair();
  RegisterRange vd = vgprPair();
  RegisterRange va = vgprPair();
  expectInstructions(emitMul(sd.ref(), sa, 8),
                     {inst("s_lshl_b64", {text(sd), text(sa), "3"})});
  expectInstructions(emitMul(vd.ref(), va, 8),
                     {inst("v_lshlrev_b64", {text(vd), "3", text(va)})});
  // The power of two may be the first operand.
  expectInstructions(emitMul(sd.ref(), 8, sa),
                     {inst("s_lshl_b64", {text(sd), text(sa), "3"})});
  expectInstructions(emitMul(vd.ref(), 8, va),
                     {inst("v_lshlrev_b64", {text(vd), "3", text(va)})});
  expectInstructions(emitMul(vd.ref(), va, 1 << 30),
                     {inst("v_lshlrev_b64", {text(vd), "30", text(va)})});
  expectInstructions(emitMul(sd.ref(), sa, 1),
                     {inst("s_mov_b64", {text(sd), text(sa)})});
  expectInstructions(emitMul(vd.ref(), va, 1),
                     {inst("v_mov_b64", {text(vd), text(va)})});
}

TEST_F(Arith64Test, MulByPowerOfTwoWithMisalignedPairsStillShifts) {
  RegisterRange sdOwner = builder.allocateSgpr(3, 2);
  RegisterRange saOwner = builder.allocateSgpr(3, 2);
  RegisterRange vdOwner = builder.allocateVgpr(3, 2);
  RegisterRange vaOwner = builder.allocateVgpr(3, 2);
  RegisterRange sAligned = sgprPair();
  RegisterRange vAligned = vgprPair();
  for (int32_t power : {8, 1 << 30}) {
    std::vector<std::string> outputs = {
        emitMul(sdOwner.ref(1, 2), sAligned, power),
        emitMul(sAligned.ref(), saOwner.ref(1, 2), power),
        emitMul(vdOwner.ref(1, 2), vAligned, power),
        emitMul(vAligned.ref(), vaOwner.ref(1, 2), power)};
    for (const std::string &out : outputs) {
      SCOPED_TRACE(out);
      EXPECT_EQ(countLinesContaining(out, "_mul"), 0);
      EXPECT_GT(countLinesContaining(out, "lshl"), 0);
    }
  }
}

TEST_F(Arith64Test, MulByNonPowerOfTwoKeepsMultiply) {
  RegisterRange sd = sgprPair();
  RegisterRange sa = sgprPair();
  RegisterRange vd = vgprPair();
  RegisterRange va = vgprPair();
  // The immediate sign-extends, so INT32_MIN and negative powers are not
  // powers of two.
  for (int32_t value : {INT32_MIN, -4, 3, 0}) {
    SCOPED_TRACE(value);
    EXPECT_GT(
        countLinesContaining(emitMul(sd.ref(), sa, value), "s_mul_hi_u32"), 0);
    EXPECT_GT(
        countLinesContaining(emitMul(vd.ref(), va, value), "v_mul_hi_u32"), 0);
  }
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
  out = emitAdd(d, a, literal32);
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

  std::string out = emitAdd(d, a, literal32);
  EXPECT_EQ(instructionCount(out), 2) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
  expectNoWideForms(out);

  // Distinct literals cannot share an instruction, so b's low half is copied;
  // the high halves are both the inline constant 0.
  out = emitAdd(d, literal32, otherLiteral);
  EXPECT_EQ(instructionCount(out), 3) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1) << out;
  expectNoWideForms(out);

  out = emitAdd(d, literal32, literal32);
  EXPECT_EQ(instructionCount(out), 2) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
}

TEST_F(Arith64Test, ValuMulLiteralsAreCopied) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  std::string out = emitMul(dstOwner.ref(), aOwner, literal32);
  // Only the low half is a literal; the high half is the inline constant 0.
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
  expectNoWideForms(out);
}

TEST_F(Arith64Test, DestinationAliasingNeedsNoCopies) {
  RegisterRange vd = vgprPair();
  RegisterRange vo = vgprPair();
  RegisterRange sd = sgprPair();
  RegisterRange so = sgprPair();
  for (const OpCase64 &current : {opCases64[0], opCases64[1]}) {
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
  for (const OpCase64 &current : opCases64) {
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

// Immediates are int32 values sign-extended to 64 bits: inline constants,
// positive and negative literals, and the extremes.
static constexpr int32_t arith64Immediates[] = {
    0,   1,    -1,        64,        65,         -16,
    -17, -100, INT32_MIN, INT32_MAX, 0x12345678, -0x12345678};

using Arith64Device = DeviceTest;

static Data64 makeData(int32_t randomCount) {
  WordQuads quads = makeArithPairs64(randomCount);
  return {join(quads.aLo, quads.aHi), join(quads.bLo, quads.bHi)};
}

// Reads an operand: an immediate, or a pair built from two inputs.
static Operand makeSource(IRBuilder &builder, const Case64 &spec,
                          std::optional<int32_t> immediate,
                          const std::vector<RegisterRef> &in, int64_t &next,
                          std::vector<RegisterRange> &owners) {
  if (immediate)
    return *immediate;
  PairRegisters pair =
      makePair(builder, in[next], in[next + 1], spec.operandsMisaligned);
  next += 2;
  owners.push_back(std::move(pair.range));
  return pair.pair;
}

// Reads b: a shift amount is one input register, anything else as makeSource.
static Operand makeB(IRBuilder &builder, const Case64 &spec,
                     const std::vector<RegisterRef> &in, int64_t &next,
                     std::vector<RegisterRange> &owners) {
  if (spec.op.amountOperand && !spec.b)
    return in[next++];
  return makeSource(builder, spec, spec.b, in, next, owners);
}

static Emitted emitArith64Separate(IRBuilder &builder, const Case64 &spec,
                                   const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  int64_t next = 0;
  Operand a = makeSource(builder, spec, spec.a, in, next, emitted.owners);
  Operand b = spec.placement == Arith64Placement::sameOperand
                  ? a
                  : makeB(builder, spec, in, next, emitted.owners);
  int32_t count = spec.dstMisaligned ? 3 : 2;
  RegisterRange range = allocateKind(builder, spec.dstKind, count);
  RegisterRef dst = range.ref(count - 2, 2);
  (builder.*spec.op.op)(dst, a, b);
  emitted.owners.push_back(std::move(range));
  emitted.value = dst;
  return emitted;
}

static Emitted emitArith64InPlace(IRBuilder &builder, const Case64 &spec,
                                  const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  int64_t next = 0;
  Operand a = makeSource(builder, spec, spec.a, in, next, emitted.owners);
  Operand b = makeB(builder, spec, in, next, emitted.owners);
  RegisterRef dst =
      spec.placement == Arith64Placement::destinationIsA ? a.reg() : b.reg();
  (builder.*spec.op.op)(dst, a, b);
  emitted.value = dst;
  return emitted;
}

// Places a in a range that also holds the destination, offset by one register.
static Emitted emitArith64Overlapping(IRBuilder &builder, const Case64 &spec,
                                      const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  RegisterRange range =
      allocateKind(builder, spec.aKind == lane ? vgpr : sgpr, 3);
  bool lowIsAHigh = spec.placement == Arith64Placement::lowIsAHigh;
  RegisterRef a = range.ref(lowIsAHigh ? 0 : 1, 2);
  RegisterRef dst = range.ref(lowIsAHigh ? 1 : 0, 2);
  builder.movb32(a.slice(0), in[0]);
  builder.movb32(a.slice(1), in[1]);
  int64_t next = 2;
  Operand b = makeB(builder, spec, in, next, emitted.owners);
  (builder.*spec.op.op)(dst, a, b);
  emitted.owners.push_back(std::move(range));
  emitted.value = dst;
  return emitted;
}

// The amount is copied into the destination's low register first.
static Emitted
emitArith64AmountInDestination(IRBuilder &builder, const Case64 &spec,
                               const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  int64_t next = 0;
  Operand a = makeSource(builder, spec, spec.a, in, next, emitted.owners);
  int32_t count = spec.dstMisaligned ? 3 : 2;
  RegisterRange range = allocateKind(builder, spec.dstKind, count);
  RegisterRef dst = range.ref(count - 2, 2);
  builder.movb32(dst.slice(0), in[next]);
  (builder.*spec.op.op)(dst, a, dst.slice(0));
  emitted.owners.push_back(std::move(range));
  emitted.value = dst;
  return emitted;
}

static Emitted emitArith64Case(IRBuilder &builder, const Case64 &spec,
                               const std::vector<RegisterRef> &in) {
  switch (spec.placement) {
  case Arith64Placement::separate:
  case Arith64Placement::sameOperand:
    return emitArith64Separate(builder, spec, in);
  case Arith64Placement::destinationIsA:
  case Arith64Placement::destinationIsB:
    return emitArith64InPlace(builder, spec, in);
  case Arith64Placement::lowIsAHigh:
  case Arith64Placement::highIsALow:
    return emitArith64Overlapping(builder, spec, in);
  case Arith64Placement::amountInDestination:
    return emitArith64AmountInDestination(builder, spec, in);
  }
  return emitArith64Separate(builder, spec, in);
}

// A result is one lane of a vgpr result or one workgroup of an sgpr result.
static int64_t arith64InputIndex(const Case64 &spec, InputKind kind,
                                 int64_t result) {
  bool perResult = spec.dstKind == sgpr || kind == lane;
  return perResult ? result : result / arithWorkgroupSize;
}

static uint64_t arith64OperandValue(const Case64 &spec,
                                    std::optional<int32_t> imm, InputKind kind,
                                    const Values &values, int64_t result) {
  if (imm)
    return static_cast<uint64_t>(int64_t{*imm});
  return values[arith64InputIndex(spec, kind, result)];
}

static Values arith64ExpectedValues(const Case64 &spec, const Data64 &data,
                                    int64_t results) {
  Values expected;
  for (int64_t i = 0; i < results; ++i) {
    uint64_t a = arith64OperandValue(spec, spec.a, spec.aKind, data.a, i);
    // The host masks a shift amount, so its low word or sign-extended
    // immediate is as good as the full value.
    bool same = spec.placement == Arith64Placement::sameOperand;
    uint64_t b =
        same ? a : arith64OperandValue(spec, spec.b, spec.bKind, data.b, i);
    expected.push_back(spec.op.host(a, b));
  }
  return expected;
}

static void addPairInputs(KernelInputs &inputs, InputKind kind,
                          const Values &values, int64_t laneCount,
                          int64_t groupCount) {
  addInput(inputs, kind, lowWords(values), laneCount, groupCount);
  addInput(inputs, kind, highWords(values), laneCount, groupCount);
}

static KernelInputs makeArith64Inputs(const Case64 &spec, const Data64 &data,
                                      int64_t groupCount) {
  int64_t laneCount = std::ssize(data.a);
  KernelInputs inputs;
  if (!spec.a)
    addPairInputs(inputs, spec.aKind, data.a, laneCount, groupCount);
  if (spec.b || spec.placement == Arith64Placement::sameOperand)
    return inputs;
  if (spec.op.amountOperand)
    addInput(inputs, spec.bKind, lowWords(data.b), laneCount, groupCount);
  else
    addPairInputs(inputs, spec.bKind, data.b, laneCount, groupCount);
  return inputs;
}

// Runs the op and checks it, through two runs that each store one half.
static void expectCase(const Case64 &spec, const Data64 &data) {
  int64_t lanes = std::ssize(data.a);
  int64_t groups = spec.dstKind == sgpr ? lanes : lanes / arithWorkgroupSize;
  int64_t results = spec.dstKind == sgpr ? groups : lanes;
  KernelInputs inputs = makeArith64Inputs(spec, data, groups);
  Values expected = arith64ExpectedValues(spec, data, results);
  for (int32_t half = 0; half < 2; ++half) {
    SCOPED_TRACE(std::string(spec.op.name) +
                 (half == 0 ? " low half" : " high half"));
    ArithKernelSpec kernel{inputs.kinds, spec.dstKind,
                           [spec, half](IRBuilder &builder, RegisterRef dst,
                                        const std::vector<RegisterRef> &in) {
                             Emitted emitted =
                                 emitArith64Case(builder, spec, in);
                             builder.movb32(dst, emitted.value.slice(half));
                           }};
    Words words = half == 0 ? lowWords(expected) : highWords(expected);
    expectArithResults(kernel, static_cast<int32_t>(groups), inputs.words,
                       words);
  }
}

static Case64 makeArith64Case(const OpCase64 &op, RegisterKind dstKind,
                              InputKind aKind, InputKind bKind) {
  return {op, dstKind, aKind, bKind};
}

static void expectValuLaneOperands64(std::span<const OpCase64> ops,
                                     const Data64 &data) {
  for (const OpCase64 &op : ops)
    expectCase(makeArith64Case(op, vgpr, lane, lane), data);
}

static void expectSaluUniformOperands64(std::span<const OpCase64> ops,
                                        const Data64 &data) {
  for (const OpCase64 &op : ops)
    expectCase(makeArith64Case(op, sgpr, uniform, uniform), data);
}

static void expectScalarsIntoVgpr64(std::span<const OpCase64> ops,
                                    const Data64 &data) {
  for (const OpCase64 &op : ops)
    expectCase(makeArith64Case(op, vgpr, uniform, uniform), data);
}

static void expectMixedScalarAndVector64(std::span<const OpCase64> ops,
                                         const Data64 &data) {
  for (const OpCase64 &op : ops) {
    expectCase(makeArith64Case(op, vgpr, uniform, lane), data);
    expectCase(makeArith64Case(op, vgpr, lane, uniform), data);
  }
}

static void expectMisalignedPairs64(std::span<const OpCase64> ops,
                                    const Data64 &data) {
  for (const OpCase64 &op : ops) {
    for (bool dstMisaligned : {false, true}) {
      for (bool operandsMisaligned : {false, true}) {
        SCOPED_TRACE(
            std::string(dstMisaligned ? "misaligned dst, " : "aligned dst, ") +
            (operandsMisaligned ? "misaligned operands" : "aligned operands"));
        for (Case64 spec : {makeArith64Case(op, vgpr, lane, lane),
                            makeArith64Case(op, sgpr, uniform, uniform)}) {
          spec.dstMisaligned = dstMisaligned;
          spec.operandsMisaligned = operandsMisaligned;
          expectCase(spec, data);
        }
      }
    }
  }
}

// aImmediates replace a and bImmediates replace b, each against a register
// operand.
static void expectImmediateOperands64(std::span<const OpCase64> ops,
                                      const Data64 &data,
                                      std::span<const int32_t> aImmediates,
                                      std::span<const int32_t> bImmediates) {
  for (const OpCase64 &op : ops) {
    for (int32_t immediate : aImmediates) {
      SCOPED_TRACE(std::string(op.name) + " a immediate " +
                   std::to_string(immediate));
      for (Case64 spec : {makeArith64Case(op, vgpr, lane, lane),
                          makeArith64Case(op, sgpr, uniform, uniform)}) {
        spec.a = immediate;
        expectCase(spec, data);
      }
    }
    for (int32_t immediate : bImmediates) {
      SCOPED_TRACE(std::string(op.name) + " b immediate " +
                   std::to_string(immediate));
      for (Case64 spec : {makeArith64Case(op, vgpr, lane, lane),
                          makeArith64Case(op, sgpr, uniform, uniform),
                          makeArith64Case(op, vgpr, uniform, uniform)}) {
        spec.b = immediate;
        expectCase(spec, data);
      }
    }
  }
}

static void
expectDestinationAliasing64(std::span<const OpCase64> ops, const Data64 &data,
                            std::span<const Arith64Placement> placements) {
  for (const OpCase64 &op : ops) {
    for (Arith64Placement placement : placements) {
      SCOPED_TRACE(std::string(op.name) + " placement " +
                   std::to_string(static_cast<int32_t>(placement)));
      for (Case64 spec : {makeArith64Case(op, vgpr, lane, lane),
                          makeArith64Case(op, sgpr, uniform, uniform)}) {
        spec.placement = placement;
        expectCase(spec, data);
        if (placement != Arith64Placement::amountInDestination)
          continue;
        // The staged shift into an odd destination reads the amount in it.
        spec.dstMisaligned = true;
        expectCase(spec, data);
      }
    }
  }
}

static constexpr Arith64Placement arithPlacements[] = {
    Arith64Placement::destinationIsA, Arith64Placement::destinationIsB,
    Arith64Placement::lowIsAHigh, Arith64Placement::highIsALow,
    Arith64Placement::sameOperand};

TEST_F(Arith64Device, ValuLaneOperands) {
  expectValuLaneOperands64(opCases64, makeData(4096));
}

TEST_F(Arith64Device, SaluUniformOperands) {
  expectSaluUniformOperands64(opCases64, makeData(256));
}

TEST_F(Arith64Device, ScalarsIntoVgpr) {
  expectScalarsIntoVgpr64(opCases64, makeData(256));
}

TEST_F(Arith64Device, MixedScalarAndVector) {
  expectMixedScalarAndVector64(opCases64, makeData(4096));
}

TEST_F(Arith64Device, MisalignedPairs) {
  expectMisalignedPairs64(opCases64, makeData(1024));
}

TEST_F(Arith64Device, ImmediateOperands) {
  expectImmediateOperands64(opCases64, makeData(0), arith64Immediates,
                            arith64Immediates);
}

TEST_F(Arith64Device, DestinationAliasing) {
  expectDestinationAliasing64(opCases64, makeData(256), arithPlacements);
}

TEST_F(Arith64Device, MulByPowerOfTwo) {
  std::span<const OpCase64> mul(&opCases64[2], 1);
  ASSERT_STREQ(mul[0].name, "muli64");
  std::vector<int32_t> powers;
  for (int32_t exponent = 0; exponent <= 30; ++exponent)
    powers.push_back(1 << exponent);
  Data64 data = makeData(256);
  expectImmediateOperands64(mul, data, powers, powers);
  for (int32_t power : {1, 8, 1 << 30})
    for (bool dstMisaligned : {false, true})
      for (bool operandsMisaligned : {false, true}) {
        SCOPED_TRACE("power " + std::to_string(power) +
                     (dstMisaligned ? ", misaligned dst" : ", aligned dst") +
                     (operandsMisaligned ? ", misaligned operands"
                                         : ", aligned operands"));
        for (Case64 spec : {makeArith64Case(mul[0], vgpr, lane, lane),
                            makeArith64Case(mul[0], sgpr, uniform, uniform)}) {
          spec.dstMisaligned = dstMisaligned;
          spec.operandsMisaligned = operandsMisaligned;
          spec.b = power;
          expectCase(spec, data);
          spec.b.reset();
          spec.a = power;
          expectCase(spec, data);
        }
      }
}

static Arith64PlacementRegisters
allocatePlacementRegisters(IRBuilder &builder) {
  return {builder.allocateVgpr(2, 2), builder.allocateVgpr(3, 2),
          builder.allocateSgpr(2, 2), builder.allocateSgpr(3, 2)};
}

static std::vector<NamedOperand>
placementsOf(const Arith64PlacementRegisters &regs) {
  return {{"vgpr pair", regs.vector},
          {"odd vgpr pair", regs.oddVector.ref(1, 2)},
          {"sgpr pair", regs.scalar},
          {"odd sgpr pair", regs.oddScalar.ref(1, 2)},
          {"inline -5", -5},
          {"literal -100", -100},
          {"literal 0x12345678", 0x12345678},
          {"INT32_MIN", INT32_MIN},
          {"2^0", 1},
          {"2^3", 8},
          {"2^30", 1 << 30}};
}

// Emits every placement combination of one op into a single kernel.
static void expectPlacementsAssemble(const OpCase64 &op, RegisterKind dstKind) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_placement", arithWorkgroupSize,
                KernelSignature());
  IRBuilder &builder = kernel.builder();
  Arith64PlacementRegisters aRegs = allocatePlacementRegisters(builder);
  Arith64PlacementRegisters bRegs = allocatePlacementRegisters(builder);
  RegisterRange dst =
      dstKind == vgpr ? builder.allocateVgpr(2, 2) : builder.allocateSgpr(2, 2);
  std::vector<NamedOperand> bPlacements = placementsOf(bRegs);
  bPlacements.push_back({"same sgpr pair as a", aRegs.scalar});
  for (const NamedOperand &a : placementsOf(aRegs)) {
    for (const NamedOperand &b : bPlacements) {
      // A scalar destination cannot take a vector operand.
      if (dstKind == sgpr &&
          (a.operand.isRegister(vgpr) || b.operand.isRegister(vgpr)))
        continue;
      (builder.*op.op)(dst.ref(), a.operand, b.operand);
    }
  }
  builder.createEndProgram();

  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(std::string(op.name) +
               (dstKind == vgpr ? " to vgpr pair" : " to sgpr pair") + "\n" +
               assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

TEST(Arith64Assembly, AllOperandPlacementsAssemble) {
  for (const OpCase64 &op : opCases64) {
    expectPlacementsAssemble(op, vgpr);
    expectPlacementsAssemble(op, sgpr);
  }
}

// 32-bit bitwise and shift ops.

static constexpr OpCase32 bitCases32[] = {
    {"andi32", &IRBuilder::andi32, "v_and_b32", "s_and_b32",
     [](uint32_t a, uint32_t b) { return a & b; }},
    {"ori32", &IRBuilder::ori32, "v_or_b32", "s_or_b32",
     [](uint32_t a, uint32_t b) { return a | b; }},
    {"shli32", &IRBuilder::shli32, "v_lshlrev_b32", "s_lshl_b32",
     [](uint32_t a, uint32_t b) { return a << (b & 31); }},
    {"shrui32", &IRBuilder::shrui32, "v_lshrrev_b32", "s_lshr_b32",
     [](uint32_t a, uint32_t b) { return a >> (b & 31); }},
    {"shrsi32", &IRBuilder::shrsi32, "v_ashrrev_i32", "s_ashr_i32",
     [](uint32_t a, uint32_t b) {
       return static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31));
     }},
};

// The first two cases are and and or, the rest are the shifts.
static constexpr std::span<const OpCase32> andOrCases32{bitCases32, 2};
static constexpr std::span<const OpCase32> shiftCases32{bitCases32 + 2, 3};

static constexpr int32_t bitwiseImmediates[] = {
    0,          1,
    31,         32,
    33,         63,
    65,         -1,
    -16,        -17,
    INT32_MIN,  INT32_MAX,
    0x12345678, static_cast<int32_t>(0xAAAAAAAA),
    0x55555555};

// Values are the edge values and the alternating bit patterns. Amounts add the
// shift counts around the 32-bit and 64-bit widths.
static WordPairs makeBitwisePairs(int32_t randomCount) {
  Words values = arithEdgeValues();
  values.insert(values.end(), {0xAAAAAAAAu, 0x55555555u});
  Words amounts = values;
  amounts.insert(amounts.end(), {3, 31, 32, 33, 63, 64, 65});
  WordPairs pairs;
  for (uint32_t a : values)
    for (uint32_t b : amounts) {
      pairs.a.push_back(a);
      pairs.b.push_back(b);
    }
  std::mt19937 engine(0x5EED1234);
  int32_t count = static_cast<int32_t>(pairs.a.size()) + randomCount;
  count = (count + arithWorkgroupSize - 1) / arithWorkgroupSize *
          arithWorkgroupSize;
  while (std::ssize(pairs.a) < count) {
    pairs.a.push_back(static_cast<uint32_t>(engine()));
    pairs.b.push_back(static_cast<uint32_t>(engine()));
  }
  return pairs;
}

using Bitwise32Test = BuilderTest;

TEST_F(Bitwise32Test, SaluForScalarOperands) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  for (const OpCase32 &current : bitCases32) {
    SCOPED_TRACE(current.saluMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.saluMnemonic, text(dst), text(a), text(b))});
    EXPECT_EQ(countLinesContaining(out, "v_"), 0);
  }
}

TEST_F(Bitwise32Test, ValuForVectorDestination) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateVgpr(1);
  RegisterRange b = builder.allocateVgpr(1);
  for (const OpCase32 &current : andOrCases32) {
    SCOPED_TRACE(current.valuMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.valuMnemonic, text(dst), text(a), text(b))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
  }
  // The VALU shifts are the *rev forms, which take the amount first.
  for (const OpCase32 &current : shiftCases32) {
    SCOPED_TRACE(current.valuMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.valuMnemonic, text(dst), text(b), text(a))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
  }
}

TEST_F(Bitwise32Test, AndOrSwapVgprIntoSrc1) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  for (const OpCase32 &current : andOrCases32) {
    SCOPED_TRACE(current.valuMnemonic);
    std::string out = emitOp(current.op, dst.ref(), v1, s0);
    expectLinesInOrder(
        out, {line(current.valuMnemonic, text(dst), text(s0), text(v1))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

    out = emitOp(current.op, dst.ref(), v1, literalA);
    expectLinesInOrder(
        out, {line(current.valuMnemonic, text(dst), literalAText, text(v1))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
  }
}

TEST_F(Bitwise32Test, ShiftOperandsAreLegalized) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  RegisterRange s1 = builder.allocateSgpr(1);
  for (const OpCase32 &current : shiftCases32) {
    SCOPED_TRACE(current.valuMnemonic);
    std::string_view mnemonic = current.valuMnemonic;
    BuilderOp op = current.op;
    // A literal amount goes in src0 of VOP2, next to the VGPR value.
    std::string out = emitOp(op, dst.ref(), v1, literalA);
    expectLinesInOrder(out,
                       {line(mnemonic, text(dst), literalAText, text(v1))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

    // A literal value is src1, which takes only a VGPR.
    out = emitOp(op, dst.ref(), literalA, v1);
    expectLinesInOrder(out, {line("v_mov_b32", "v2", literalAText),
                             line(mnemonic, text(dst), text(v1), "v2")});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);

    // Two distinct SGPRs: the value is copied.
    out = emitOp(op, dst.ref(), s0, s1);
    expectLinesInOrder(out, {line("v_mov_b32", "v2", text(s0)),
                             line(mnemonic, text(dst), text(s1), "v2")});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);

    // VOP3 takes an SGPR or an inline constant as the value.
    out = emitOp(op, dst.ref(), s0, v1);
    expectLinesInOrder(out, {line(mnemonic, text(dst), text(v1), text(s0))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
    out = emitOp(op, dst.ref(), 5, v1);
    expectLinesInOrder(out, {line(mnemonic, text(dst), text(v1), "5")});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

    out = emitOp(op, dst.ref(), s0, s0);
    expectLinesInOrder(out, {line(mnemonic, text(dst), text(s0), text(s0))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
  }
}

TEST_F(Bitwise32Test, SaluTwoLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  for (const OpCase32 &current : bitCases32) {
    SCOPED_TRACE(current.saluMnemonic);
    std::string out = emitOp(current.op, dst.ref(), literalA, literalB);
    expectLinesInOrder(
        out, {line("s_mov_b32", "s1", literalBText),
              line(current.saluMnemonic, text(dst), literalAText, "s1")});
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);

    out = emitOp(current.op, dst.ref(), literalA, literalA);
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0);
  }
}

TEST_F(Bitwise32Test, DestinationMayAliasOperand) {
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange v2 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  RegisterRange s1 = builder.allocateSgpr(1);
  for (const OpCase32 &current : shiftCases32) {
    SCOPED_TRACE(current.valuMnemonic);
    expectInstructions(
        emitOp(current.op, v1.ref(), v1, v2),
        {inst(current.valuMnemonic, {text(v1), text(v2), text(v1)})});
    expectInstructions(
        emitOp(current.op, v2.ref(), v1, v2),
        {inst(current.valuMnemonic, {text(v2), text(v2), text(v1)})});
  }
  for (const OpCase32 &current : bitCases32) {
    SCOPED_TRACE(current.saluMnemonic);
    expectInstructions(
        emitOp(current.op, s0.ref(), s0, s1),
        {inst(current.saluMnemonic, {text(s0), text(s0), text(s1)})});
    expectInstructions(
        emitOp(current.op, s1.ref(), s0, s1),
        {inst(current.saluMnemonic, {text(s1), text(s0), text(s1)})});
  }
}

TEST_F(Bitwise32Test, ShiftByZeroIsMoveOrNothing) {
  RegisterRange sd = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange vd = builder.allocateVgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  // Only the low 5 bits of the amount count.
  for (int32_t amount : {0, 32, -32, 64, INT32_MIN})
    for (const OpCase32 &current : shiftCases32) {
      SCOPED_TRACE(std::string(current.name) + " by " + std::to_string(amount));
      BuilderOp op = current.op;
      // The value is dst itself: nothing to emit, so SCC stays untouched.
      expectInstructions(emitOp(op, sa.ref(), sa, amount), {});
      expectInstructions(emitOp(op, va.ref(), va, amount), {});
      // Another register or an immediate is a move.
      expectInstructions(emitOp(op, sd.ref(), sa, amount),
                         {inst("s_mov_b32", {text(sd), text(sa)})});
      expectInstructions(emitOp(op, vd.ref(), va, amount),
                         {inst("v_mov_b32", {text(vd), text(va)})});
      expectInstructions(emitOp(op, vd.ref(), sa, amount),
                         {inst("v_mov_b32", {text(vd), text(sa)})});
      expectInstructions(emitOp(op, sd.ref(), literalA, amount),
                         {inst("s_mov_b32", {text(sd), literalAText})});
      expectInstructions(emitOp(op, vd.ref(), 5, amount),
                         {inst("v_mov_b32", {text(vd), "5"})});
    }
}

TEST_F(Bitwise32Test, ShiftByNonZeroAmountStillShifts) {
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  // 33 is 1 modulo 32, and only an immediate amount is elided.
  for (const OpCase32 &current : shiftCases32) {
    SCOPED_TRACE(current.name);
    expectInstructions(
        emitOp(current.op, sa.ref(), sa, 33),
        {inst(current.saluMnemonic, {text(sa), text(sa), "33"})});
    expectInstructions(
        emitOp(current.op, va.ref(), va, 33),
        {inst(current.valuMnemonic, {text(va), "33", text(va)})});
    // A register amount is not known to be zero.
    RegisterRange amount = builder.allocateSgpr(1);
    expectInstructions(
        emitOp(current.op, sa.ref(), sa, amount),
        {inst(current.saluMnemonic, {text(sa), text(sa), text(amount)})});
  }
}

TEST_F(Bitwise32Test, ShiftByZeroStillRejectsInvalidOperands) {
  RegisterRange sd = builder.allocateSgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  RegisterRange acc = builder.allocateAgpr(1);
  for (const OpCase32 &current : shiftCases32) {
    BuilderOp op = current.op;
    EXPECT_THROW((builder.*op)(sd.ref(), va, 0), std::invalid_argument);
    EXPECT_THROW((builder.*op)(acc.ref(), va, 0), std::invalid_argument);
    EXPECT_THROW((builder.*op)(va.ref(), acc, 0), std::invalid_argument);
  }
  EXPECT_EQ(body.toString(), "");
}

TEST_F(Bitwise32Test, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange vdst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s1 = builder.allocateSgpr(1);
  RegisterRange acc = builder.allocateAgpr(1);
  RegisterRange pair = builder.allocateVgpr(2);
  for (const OpCase32 &current : bitCases32) {
    SCOPED_TRACE(current.name);
    BuilderOp op = current.op;
    EXPECT_THROW((builder.*op)(sdst.ref(), v1, s1), std::invalid_argument);
    EXPECT_THROW((builder.*op)(sdst.ref(), s1, v1), std::invalid_argument);
    EXPECT_THROW((builder.*op)(acc.ref(), s1, s1), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), acc, s1), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), s1, acc), std::invalid_argument);
    EXPECT_THROW((builder.*op)(pair.ref(), v1, s1), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), pair, s1), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), s1, pair), std::invalid_argument);
  }
  EXPECT_EQ(body.toString(), "");
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 4);
}

using Bitwise32Device = DeviceTest;

TEST_F(Bitwise32Device, ValuLaneOperands) {
  expectValuLaneOperands(bitCases32, makeBitwisePairs(4096));
}

TEST_F(Bitwise32Device, SaluUniformOperands) {
  expectSaluUniformOperands(bitCases32, makeBitwisePairs(256));
}

TEST_F(Bitwise32Device, ScalarsIntoVgpr) {
  expectScalarsIntoVgpr(bitCases32, makeBitwisePairs(256));
}

TEST_F(Bitwise32Device, MixedScalarAndVector) {
  expectMixedScalarAndVector(bitCases32, makeBitwisePairs(4096));
}

TEST_F(Bitwise32Device, ImmediateOperands) {
  expectImmediateOperands(bitCases32, makeBitwisePairs(0), bitwiseImmediates);
}

// Writes op(inputs[0], inputs[1]) into input target, then moves it to dst.
static void expectAliasing32(const OpCase32 &op, RegisterKind result,
                             InputKind kind, int32_t target,
                             const WordPairs &pairs) {
  SCOPED_TRACE(std::string(op.name) + " into input " + std::to_string(target));
  std::vector<uint32_t> expected;
  for (int64_t i = 0; i < std::ssize(pairs.a); ++i)
    expected.push_back(op.host(pairs.a[i], pairs.b[i]));
  int32_t groups = kind == InputKind::lane
                       ? laneGroupCount(pairs.a)
                       : static_cast<int32_t>(pairs.a.size());
  BuilderOp emit = op.op;
  expectArithResults({{kind, kind},
                      result,
                      [emit, target](IRBuilder &builder, RegisterRef dst,
                                     const std::vector<RegisterRef> &inputs) {
                        (builder.*emit)(inputs[target], inputs[0], inputs[1]);
                        builder.movb32(dst, inputs[target]);
                      }},
                     groups, {pairs.a, pairs.b}, expected);
}

TEST_F(Bitwise32Device, DestinationAliasesOperand) {
  WordPairs pairs = makeBitwisePairs(256);
  expectAliasing32(bitCases32[2], vgpr, lane, 0, pairs);
  expectAliasing32(bitCases32[3], vgpr, lane, 1, pairs);
  expectAliasing32(bitCases32[0], sgpr, uniform, 0, pairs);
  expectAliasing32(bitCases32[4], sgpr, uniform, 1, pairs);
}

// Each shift by zero writes into an input register, or into another one, and
// the result is moved to dst.
TEST_F(Bitwise32Device, ShiftByZero) {
  WordPairs pairs = makeBitwisePairs(256);
  int32_t groups = laneGroupCount(pairs.a);
  int32_t scalarGroups = static_cast<int32_t>(pairs.a.size());
  for (const OpCase32 &op : shiftCases32)
    for (int32_t amount : {0, 32, -64}) {
      SCOPED_TRACE(std::string(op.name) + " by " + std::to_string(amount));
      BuilderOp emit = op.op;
      for (bool inPlace : {false, true}) {
        SCOPED_TRACE(inPlace ? "in place" : "into dst");
        auto build = [=](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
          if (!inPlace) {
            (builder.*emit)(dst, inputs[0], amount);
            return;
          }
          (builder.*emit)(inputs[0], inputs[0], amount);
          builder.movb32(dst, inputs[0]);
        };
        expectArithResults({{lane}, vgpr, build}, groups, {pairs.a}, pairs.a);
        expectArithResults({{uniform}, sgpr, build}, scalarGroups, {pairs.a},
                           pairs.a);
        expectArithResults({{uniform}, vgpr, build}, scalarGroups, {pairs.a},
                           perLane(pairs.a));
      }
    }
}

// Multiplying by 1 in place emits nothing, and into another register a move.
TEST_F(Bitwise32Device, MulByOne) {
  WordPairs pairs = makeBitwisePairs(256);
  int32_t groups = laneGroupCount(pairs.a);
  int32_t scalarGroups = static_cast<int32_t>(pairs.a.size());
  for (bool inPlace : {false, true})
    for (bool oneFirst : {false, true}) {
      SCOPED_TRACE(std::string(inPlace ? "in place" : "into dst") +
                   (oneFirst ? ", 1 first" : ", 1 second"));
      auto build = [=](IRBuilder &builder, RegisterRef dst,
                       const std::vector<RegisterRef> &inputs) {
        RegisterRef target = inPlace ? inputs[0] : dst;
        if (oneFirst)
          builder.muli32(target, 1, inputs[0]);
        else
          builder.muli32(target, inputs[0], 1);
        if (inPlace)
          builder.movb32(dst, inputs[0]);
      };
      expectArithResults({{lane}, vgpr, build}, groups, {pairs.a}, pairs.a);
      expectArithResults({{uniform}, sgpr, build}, scalarGroups, {pairs.a},
                         pairs.a);
      expectArithResults({{uniform}, vgpr, build}, scalarGroups, {pairs.a},
                         perLane(pairs.a));
    }
}

TEST(Bitwise32Assembly, ZeroShiftsAndMulByOneAssemble) {
  auto emitAll = [](IRBuilder &builder, RegisterRef dst, Operand a, Operand) {
    for (BuilderOp op :
         {&IRBuilder::shli32, &IRBuilder::shrui32, &IRBuilder::shrsi32})
      for (int32_t amount : {0, 32, -32})
        (builder.*op)(dst, a, amount);
    builder.muli32(dst, a, 1);
    builder.muli32(dst, 1, a);
  };
  expectAllPlacementsAssemble("zero shifts and mul by 1", emitAll);
}

TEST(Bitwise32Assembly, AllOperandPlacementsAssemble) {
  for (const OpCase32 &op : bitCases32)
    expectAllPlacementsAssemble(
        op.name, [&op](IRBuilder &builder, RegisterRef dst, Operand a,
                       Operand b) { (builder.*op.op)(dst, a, b); });
}

// 64-bit bitwise and shift ops.

// The shifts take a single-register amount, and the host masks it to 6 bits.
static constexpr OpCase64 bitCases64[] = {
    {"andi64", &IRBuilder::andi64,
     [](uint64_t a, uint64_t b) { return a & b; }},
    {"ori64", &IRBuilder::ori64, [](uint64_t a, uint64_t b) { return a | b; }},
    {"shli64", &IRBuilder::shli64,
     [](uint64_t a, uint64_t b) { return a << (b & 63); }, true},
    {"shrui64", &IRBuilder::shrui64,
     [](uint64_t a, uint64_t b) { return a >> (b & 63); }, true},
    {"shrsi64", &IRBuilder::shrsi64,
     [](uint64_t a, uint64_t b) {
       return static_cast<uint64_t>(static_cast<int64_t>(a) >> (b & 63));
     },
     true},
};

// The first two cases are and and or, the rest are the shifts.
static constexpr std::span<const OpCase64> andOrCases64{bitCases64, 2};
static constexpr std::span<const OpCase64> shiftCases64{bitCases64 + 2, 3};

static constexpr const char *andOrSalu64[] = {"s_and_b64", "s_or_b64"};
static constexpr const char *andOrSalu32[] = {"s_and_b32", "s_or_b32"};
static constexpr const char *andOrValu32[] = {"v_and_b32", "v_or_b32"};
static constexpr const char *shiftSalu64[] = {"s_lshl_b64", "s_lshr_b64",
                                              "s_ashr_i64"};
static constexpr const char *shiftValu64[] = {"v_lshlrev_b64", "v_lshrrev_b64",
                                              "v_ashrrev_i64"};

// Amounts around the 32-bit and 64-bit widths, with the sign bit set, and the
// immediates that cover them.
static constexpr int32_t shiftAmountImmediates[] = {
    0, 1, 31, 32, 33, 63, 64, 65, -1, INT32_MIN, 0x12345678};

// Values are the 64-bit edge values and the alternating bit patterns, against
// the amounts, then random pairs, padded to whole workgroups.
static Data64 makeShiftData64(int32_t randomCount) {
  Values values = makeData(0).a;
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  values.insert(values.end(), {0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull});
  Values amounts = {0,
                    1,
                    3,
                    31,
                    32,
                    33,
                    63,
                    64,
                    65,
                    UINT64_MAX,
                    static_cast<uint64_t>(int64_t{INT32_MIN}),
                    0x12345678};
  Data64 data;
  for (uint64_t a : values)
    for (uint64_t b : amounts) {
      data.a.push_back(a);
      data.b.push_back(b);
    }
  std::mt19937 engine(0x5EED1234);
  int64_t count = std::ssize(data.a) + randomCount;
  count = (count + arithWorkgroupSize - 1) / arithWorkgroupSize *
          arithWorkgroupSize;
  while (std::ssize(data.a) < count) {
    uint64_t low = engine();
    data.a.push_back(low | uint64_t{engine()} << 32);
    data.b.push_back(engine());
  }
  return data;
}

using Bitwise64Test = Arith64Test;

TEST_F(Bitwise64Test, SaluAlignedAndOrUseOneInstruction) {
  RegisterRange d = sgprPair();
  RegisterRange a = sgprPair();
  RegisterRange b = sgprPair();
  for (int32_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(andOrCases64[i].name);
    BuilderOp op = andOrCases64[i].op;
    std::string mnemonic = andOrSalu64[i];
    expectInstructions(emitOp(op, d.ref(), a, b),
                       {inst(mnemonic, {text(d), text(a), text(b)})});
    expectInstructions(emitOp(op, d.ref(), a, -5),
                       {inst(mnemonic, {text(d), text(a), "-5"})});
    expectInstructions(emitOp(op, d.ref(), a, literal32),
                       {inst(mnemonic, {text(d), text(a), literal32Text})});
  }
}

TEST_F(Bitwise64Test, SaluAndOrSplitsMisalignedAndNegativeLiterals) {
  RegisterRange dstOwner = builder.allocateSgpr(3, 2);
  RegisterRange aOwner = builder.allocateSgpr(3, 2);
  RegisterRange d = sgprPair();
  RegisterRange a = sgprPair();
  RegisterRange b = sgprPair();
  RegisterRef oddDst = dstOwner.ref(1, 2);
  RegisterRef oddA = aOwner.ref(1, 2);
  for (int32_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(andOrCases64[i].name);
    BuilderOp op = andOrCases64[i].op;
    std::string mnemonic = andOrSalu32[i];
    expectInstructions(
        emitOp(op, oddDst, a, b),
        {inst(mnemonic, {half(oddDst, 0), half(a.ref(), 0), half(b.ref(), 0)}),
         inst(mnemonic,
              {half(oddDst, 1), half(a.ref(), 1), half(b.ref(), 1)})});
    expectInstructions(
        emitOp(op, d.ref(), oddA, b),
        {inst(mnemonic, {half(d.ref(), 0), half(oddA, 0), half(b.ref(), 0)}),
         inst(mnemonic, {half(d.ref(), 1), half(oddA, 1), half(b.ref(), 1)})});
    // The low half of -100 is a literal, which the 64-bit form would
    // zero-extend.
    expectInstructions(
        emitOp(op, d.ref(), a, -100),
        {inst(mnemonic, {half(d.ref(), 0), half(a.ref(), 0), "-100"}),
         inst(mnemonic, {half(d.ref(), 1), half(a.ref(), 1), "-1"})});
  }
}

TEST_F(Bitwise64Test, ValuAndOrSplitIntoHalves) {
  RegisterRange d = vgprPair();
  RegisterRange va = vgprPair();
  RegisterRange vb = vgprPair();
  RegisterRange sa = sgprPair();
  RegisterRange sb = sgprPair();
  RegisterRef dst = d.ref();
  for (int32_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(andOrCases64[i].name);
    BuilderOp op = andOrCases64[i].op;
    std::string mnemonic = andOrValu32[i];
    std::string out = emitOp(op, dst, va, vb);
    expectInstructions(
        out,
        {inst(mnemonic, {half(dst, 0), half(va.ref(), 0), half(vb.ref(), 0)}),
         inst(mnemonic, {half(dst, 1), half(va.ref(), 1), half(vb.ref(), 1)})});

    // A VGPR belongs in src1, so an SGPR pair on either side needs no copy.
    expectInstructions(
        emitOp(op, dst, sa, vb),
        {inst(mnemonic, {half(dst, 0), half(sa.ref(), 0), half(vb.ref(), 0)}),
         inst(mnemonic, {half(dst, 1), half(sa.ref(), 1), half(vb.ref(), 1)})});
    expectInstructions(
        emitOp(op, dst, va, sb),
        {inst(mnemonic, {half(dst, 0), half(sb.ref(), 0), half(va.ref(), 0)}),
         inst(mnemonic, {half(dst, 1), half(sb.ref(), 1), half(va.ref(), 1)})});

    // Two distinct SGPR pairs exceed the constant bus in each half.
    out = emitOp(op, dst, sa, sb);
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 2) << out;

    out = emitOp(op, dst, va, literal32);
    expectInstructions(
        out, {inst(mnemonic, {half(dst, 0), literal32Text, half(va.ref(), 0)}),
              inst(mnemonic, {half(dst, 1), "0", half(va.ref(), 1)})});
  }
}

TEST_F(Bitwise64Test, AndOrOverlapOrdersHalves) {
  RegisterRange sr = builder.allocateSgpr(4);
  RegisterRange vr = builder.allocateVgpr(4);
  RegisterRange so = sgprPair();
  RegisterRange vo = vgprPair();
  for (int32_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(andOrCases64[i].name);
    BuilderOp op = andOrCases64[i].op;
    // dst.lo is a.hi, so the high half is written first and nothing is copied.
    for (const auto &[range, other, mnemonic] :
         {std::tuple{&sr, &so, andOrSalu32[i]},
          std::tuple{&vr, &vo, andOrValu32[i]}}) {
      RegisterRef a = range->ref(0, 2);
      RegisterRef d = range->ref(1, 2);
      RegisterRef b = other->ref();
      expectInstructions(
          emitOp(op, d, a, b),
          {inst(mnemonic, {half(d, 1), half(a, 1), half(b, 1)}),
           inst(mnemonic, {half(d, 0), half(a, 0), half(b, 0)})});
    }
    // dst.lo is a.hi and dst.hi is b.lo: neither order works, so a.hi is
    // copied.
    for (const auto &[range, mnemonic] :
         {std::tuple{&sr, andOrSalu32[i]}, std::tuple{&vr, andOrValu32[i]}}) {
      RegisterRef a = range->ref(0, 2);
      RegisterRef d = range->ref(1, 2);
      RegisterRef b = range->ref(2, 2);
      std::string out = emitOp(op, d, a, b);
      EXPECT_EQ(instructionCount(out), 3) << out;
      EXPECT_EQ(countLinesContaining(out, "mov"), 1) << out;
      EXPECT_EQ(countLinesContaining(out, mnemonic), 2) << out;
    }
  }
}

TEST_F(Bitwise64Test, SaluShifts) {
  RegisterRange dstOwner = builder.allocateSgpr(3, 2);
  RegisterRange d = sgprPair();
  RegisterRange a = sgprPair();
  RegisterRange oddOwner = builder.allocateSgpr(3, 2);
  RegisterRange amount = builder.allocateSgpr(1);
  RegisterRef oddDst = dstOwner.ref(1, 2);
  RegisterRef oddA = oddOwner.ref(1, 2);
  for (int32_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(shiftCases64[i].name);
    BuilderOp op = shiftCases64[i].op;
    std::string mnemonic = shiftSalu64[i];
    // The SALU form takes the value first.
    expectInstructions(emitOp(op, d.ref(), a, amount),
                       {inst(mnemonic, {text(d), text(a), text(amount)})});
    expectInstructions(emitOp(op, d.ref(), a, literalA),
                       {inst(mnemonic, {text(d), text(a), literalAText})});

    // A misaligned value moves to an aligned pair first.
    std::string out = emitOp(op, d.ref(), oddA, amount);
    EXPECT_EQ(instructionCount(out), 3) << out;
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 2) << out;
    EXPECT_EQ(countLinesContaining(out, text(oddA) + ","), 0) << out;
    EXPECT_EQ(countLinesContaining(out, mnemonic + " " + text(d) + ", s["), 1)
        << out;

    // A negative literal cannot be zero-extended by the 64-bit form.
    out = emitOp(op, d.ref(), -100, amount);
    EXPECT_EQ(instructionCount(out), 3) << out;
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 2) << out;
    EXPECT_EQ(countLinesContaining(out, ", -100"), 1) << out;

    // Distinct literals cannot share an instruction, so the amount is copied
    // to the lowest free SGPR.
    expectInstructions(emitOp(op, d.ref(), literal32, literalA),
                       {inst("s_mov_b32", {"s11", literalAText}),
                        inst(mnemonic, {text(d), literal32Text, "s11"})});

    // A misaligned destination is written through an aligned pair.
    out = emitOp(op, oddDst, a, amount);
    EXPECT_EQ(instructionCount(out), 3) << out;
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 2) << out;
    EXPECT_EQ(countLinesContaining(out, mnemonic + " s["), 1) << out;
  }
}

TEST_F(Bitwise64Test, ValuShifts) {
  RegisterRange dstOwner = builder.allocateVgpr(3, 2);
  RegisterRange oddOwner = builder.allocateVgpr(3, 2);
  RegisterRange d = vgprPair();
  RegisterRange a = vgprPair();
  RegisterRange amount = builder.allocateVgpr(1);
  RegisterRange sa = sgprPair();
  RegisterRange sAmount = builder.allocateSgpr(1);
  RegisterRef oddDst = dstOwner.ref(1, 2);
  RegisterRef oddA = oddOwner.ref(1, 2);
  for (int32_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(shiftCases64[i].name);
    BuilderOp op = shiftCases64[i].op;
    std::string mnemonic = shiftValu64[i];
    // The VALU forms are the *rev ones, which take the amount first.
    expectInstructions(emitOp(op, d.ref(), a, amount),
                       {inst(mnemonic, {text(d), text(amount), text(a)})});

    // VOP3 takes no literal, so the amount is copied.
    std::string out = emitOp(op, d.ref(), a, literalA);
    EXPECT_EQ(instructionCount(out), 2) << out;
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
    EXPECT_EQ(countLinesContaining(out, literalAText), 1) << out;

    expectInstructions(emitOp(op, d.ref(), sa, amount),
                       {inst(mnemonic, {text(d), text(amount), text(sa)})});

    // An SGPR pair and an SGPR amount exceed the constant bus.
    out = emitOp(op, d.ref(), sa, sAmount);
    EXPECT_EQ(instructionCount(out), 2) << out;
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32 v"), 1) << out;
    EXPECT_EQ(countLinesContaining(out, text(sAmount)), 1) << out;

    // A literal value moves to an aligned VGPR pair.
    out = emitOp(op, d.ref(), literal32, amount);
    EXPECT_EQ(instructionCount(out), 2) << out;
    EXPECT_EQ(countLinesContaining(out, "v_mov_b64"), 1) << out;
    EXPECT_EQ(countLinesContaining(out, literal32Text), 1) << out;

    out = emitOp(op, d.ref(), oddA, amount);
    EXPECT_EQ(instructionCount(out), 3) << out;
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 2) << out;
    EXPECT_EQ(countLinesContaining(out, mnemonic + " " + text(d)), 1) << out;

    // A misaligned destination is written through an aligned pair.
    out = emitOp(op, oddDst, a, amount);
    EXPECT_EQ(instructionCount(out), 3) << out;
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 2) << out;
    EXPECT_EQ(countLinesContaining(out, mnemonic + " v["), 1) << out;
  }
}

TEST_F(Bitwise64Test, ShiftAliasing) {
  RegisterRange vd = vgprPair();
  RegisterRange va = vgprPair();
  RegisterRange vAmount = builder.allocateVgpr(1);
  RegisterRange sd = sgprPair();
  RegisterRange sa = sgprPair();
  RegisterRange sAmount = builder.allocateSgpr(1);
  for (int32_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(shiftCases64[i].name);
    BuilderOp op = shiftCases64[i].op;
    RegisterRef v = vd.ref();
    RegisterRef s = sd.ref();
    expectInstructions(
        emitOp(op, v, v, vAmount),
        {inst(shiftValu64[i], {text(v), text(vAmount), text(v)})});
    expectInstructions(
        emitOp(op, v, va, v.slice(0)),
        {inst(shiftValu64[i], {text(v), text(v.slice(0)), text(va)})});
    expectInstructions(
        emitOp(op, s, s, sAmount),
        {inst(shiftSalu64[i], {text(s), text(s), text(sAmount)})});
    expectInstructions(
        emitOp(op, s, sa, s.slice(0)),
        {inst(shiftSalu64[i], {text(s), text(sa), text(s.slice(0))})});
  }
}

TEST_F(Bitwise64Test, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = sgprPair();
  RegisterRange vdst = vgprPair();
  RegisterRange spair = sgprPair();
  RegisterRange vpair = vgprPair();
  RegisterRange accPair = builder.allocateAgpr(2, 2);
  RegisterRange accSingle = builder.allocateAgpr(1);
  RegisterRange sSingle = builder.allocateSgpr(1);
  RegisterRange vSingle = builder.allocateVgpr(1);
  RegisterRange triple = builder.allocateVgpr(3);
  for (const OpCase64 &current : bitCases64) {
    SCOPED_TRACE(current.name);
    BuilderOp op = current.op;
    // b is a single register for the shifts and a pair otherwise.
    bool amount = current.amountOperand;
    Operand sB = amount ? Operand(sSingle) : Operand(spair);
    Operand vB = amount ? Operand(vSingle) : Operand(vpair);
    Operand accB = amount ? Operand(accSingle) : Operand(accPair);
    Operand wrongWidthB = amount ? Operand(vpair) : Operand(vSingle);
    EXPECT_THROW((builder.*op)(sdst.ref(), vpair, sB), std::invalid_argument);
    EXPECT_THROW((builder.*op)(sdst.ref(), spair, vB), std::invalid_argument);
    EXPECT_THROW((builder.*op)(accPair.ref(), spair, sB),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), accPair, sB), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), spair, accB), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vSingle.ref(), vpair, sB),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(triple.ref(), vpair, sB), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), vSingle, sB), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), spair, wrongWidthB),
                 std::invalid_argument);
  }
  EXPECT_EQ(body.toString(), "");
}

TEST_F(Bitwise64Test, TemporariesAreReleased) {
  RegisterRange dst = vgprPair();
  RegisterRange a = sgprPair();
  RegisterRange amount = builder.allocateSgpr(1);
  // The SGPR pair and the SGPR amount exceed the constant bus: one copy.
  emitOp(&IRBuilder::shli64, dst.ref(), a, amount);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 3);
  RegisterRange next = builder.allocateVgpr(1);
  EXPECT_EQ(next.start(), 2);
}

using Bitwise64Device = DeviceTest;

TEST_F(Bitwise64Device, ValuLaneOperands) {
  expectValuLaneOperands64(andOrCases64, makeData(4096));
  expectValuLaneOperands64(shiftCases64, makeShiftData64(4096));
}

TEST_F(Bitwise64Device, SaluUniformOperands) {
  expectSaluUniformOperands64(andOrCases64, makeData(256));
  expectSaluUniformOperands64(shiftCases64, makeShiftData64(256));
}

TEST_F(Bitwise64Device, ScalarsIntoVgpr) {
  expectScalarsIntoVgpr64(andOrCases64, makeData(256));
  expectScalarsIntoVgpr64(shiftCases64, makeShiftData64(256));
}

TEST_F(Bitwise64Device, MixedScalarAndVector) {
  expectMixedScalarAndVector64(andOrCases64, makeData(4096));
  expectMixedScalarAndVector64(shiftCases64, makeShiftData64(4096));
}

TEST_F(Bitwise64Device, MisalignedPairs) {
  expectMisalignedPairs64(andOrCases64, makeData(1024));
  expectMisalignedPairs64(shiftCases64, makeShiftData64(1024));
}

TEST_F(Bitwise64Device, ImmediateOperands) {
  expectImmediateOperands64(andOrCases64, makeData(0), arith64Immediates,
                            arith64Immediates);
  expectImmediateOperands64(shiftCases64, makeShiftData64(0), arith64Immediates,
                            shiftAmountImmediates);
}

TEST_F(Bitwise64Device, DestinationAliasing) {
  expectDestinationAliasing64(andOrCases64, makeData(256), arithPlacements);
  static constexpr Arith64Placement shiftPlacements[] = {
      Arith64Placement::destinationIsA, Arith64Placement::lowIsAHigh,
      Arith64Placement::highIsALow, Arith64Placement::amountInDestination};
  expectDestinationAliasing64(shiftCases64, makeShiftData64(256),
                              shiftPlacements);
}

// Emits every placement of a shift into a single kernel, with the destination
// aligned and misaligned.
static void expectShiftPlacementsAssemble(const OpCase64 &op,
                                          RegisterKind dstKind) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_placement", arithWorkgroupSize,
                KernelSignature());
  IRBuilder &builder = kernel.builder();
  Arith64PlacementRegisters aRegs = allocatePlacementRegisters(builder);
  RegisterRange vector = builder.allocateVgpr(1);
  RegisterRange scalar = builder.allocateSgpr(1);
  RegisterRange dst =
      dstKind == vgpr ? builder.allocateVgpr(3, 2) : builder.allocateSgpr(3, 2);
  std::vector<NamedOperand> amounts = {{"vgpr", vector},
                                       {"sgpr", scalar},
                                       {"inline 5", 5},
                                       {"literal 100", 100},
                                       {"literal 0x12345678", 0x12345678},
                                       {"INT32_MIN", INT32_MIN},
                                       {"-1", -1}};
  for (const NamedOperand &a : placementsOf(aRegs)) {
    for (const NamedOperand &amount : amounts) {
      // A scalar destination cannot take a vector operand.
      if (dstKind == sgpr &&
          (a.operand.isRegister(vgpr) || amount.operand.isRegister(vgpr)))
        continue;
      for (int32_t offset : {0, 1})
        (builder.*op.op)(dst.ref(offset, 2), a.operand, amount.operand);
    }
  }
  builder.createEndProgram();

  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(std::string(op.name) +
               (dstKind == vgpr ? " to vgpr pair" : " to sgpr pair") + "\n" +
               assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

TEST(Bitwise64Assembly, AllOperandPlacementsAssemble) {
  for (const OpCase64 &op : andOrCases64) {
    expectPlacementsAssemble(op, vgpr);
    expectPlacementsAssemble(op, sgpr);
  }
  for (const OpCase64 &op : shiftCases64) {
    expectShiftPlacementsAssemble(op, vgpr);
    expectShiftPlacementsAssemble(op, sgpr);
  }
}

// Division.

// The quotient truncates toward zero and INT32_MIN / -1 wraps.
static uint32_t hostDiv(uint32_t a, uint32_t b) {
  int32_t x = static_cast<int32_t>(a);
  int32_t y = static_cast<int32_t>(b);
  if (x == INT32_MIN && y == -1)
    return a;
  return static_cast<uint32_t>(x / y);
}

// The remainder has the sign of the dividend. INT32_MIN % -1 is 0, which C
// leaves undefined, so it is special-cased.
static uint32_t hostMod(uint32_t a, uint32_t b) {
  int32_t x = static_cast<int32_t>(a);
  int32_t y = static_cast<int32_t>(b);
  if (y == -1)
    return 0;
  return static_cast<uint32_t>(x % y);
}

namespace {

using DivisionOp = void (IRBuilder::*)(RegisterRef, Operand, Operand, bool);

struct DivisionCase {
  const char *name;
  DivisionOp op;
  uint32_t (*host)(uint32_t, uint32_t);
};

} // namespace

static constexpr DivisionCase divisionCases[] = {
    {"divsi32", &IRBuilder::divsi32, hostDiv},
    {"modsi32", &IRBuilder::modsi32, hostMod}};

namespace {

enum class Form { scalar, vector, mixed };

struct DivRegisters {
  RegisterRange dst;
  RegisterRange a;
  RegisterRange b;
};

static constexpr Form allForms[] = {Form::scalar, Form::vector, Form::mixed};

static const char *formName(Form form) {
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

class DivTest : public BuilderTest {
protected:
  std::string emitDivision(const DivisionCase &divisionCase, RegisterRef dst,
                           Operand a, Operand b, bool useIEEE) {
    return emitted([&](IRBuilder &local) {
      (local.*divisionCase.op)(dst, a, b, useIEEE);
    });
  }

  std::string emitDiv(RegisterRef dst, Operand a, Operand b, bool useIEEE) {
    return emitDivision(divisionCases[0], dst, a, b, useIEEE);
  }

  RegisterRange allocate(RegisterKind kind) {
    return kind == RegisterKind::sgpr ? builder.allocateSgpr(1)
                                      : builder.allocateVgpr(1);
  }

  DivRegisters makeRegisters(Form form) {
    RegisterKind dstKind =
        form == Form::scalar ? RegisterKind::sgpr : RegisterKind::vgpr;
    RegisterKind sourceKind =
        form == Form::vector ? RegisterKind::vgpr : RegisterKind::sgpr;
    return {allocate(dstKind), allocate(sourceKind), allocate(sourceKind)};
  }

  std::string emitForm(const DivisionCase &divisionCase, Form form,
                       bool useIEEE) {
    DivRegisters regs = makeRegisters(form);
    return emitDivision(divisionCase, regs.dst.ref(), regs.a, regs.b, useIEEE);
  }

  std::string emitForm(Form form, bool useIEEE) {
    return emitForm(divisionCases[0], form, useIEEE);
  }

  int32_t nextFreeIndex(RegisterKind kind) { return allocate(kind).start(); }

  // The instructions of a / 2^exponent or a % 2^exponent, whose temporary is
  // the next free register of the destination's kind.
  std::vector<std::string> powerOfTwoInstructions(bool scalar, bool quotient,
                                                  const std::string &dst,
                                                  const std::string &a,
                                                  const std::string &temp,
                                                  int32_t exponent) {
    // The VALU shifts are the reversed forms, with the amount first.
    auto shift = [&](std::string_view scalarName, std::string_view vectorName,
                     const std::string &to, const std::string &from,
                     int32_t amount) {
      std::string amountText = std::to_string(amount);
      return scalar ? inst(scalarName, {to, from, amountText})
                    : inst(vectorName, {to, amountText, from});
    };
    if (exponent == 0)
      return {
          inst(scalar ? "s_mov_b32" : "v_mov_b32", {dst, quotient ? a : "0"})};
    std::vector<std::string> expected;
    if (exponent == 1) {
      expected.push_back(shift("s_lshr_b32", "v_lshrrev_b32", temp, a, 31));
    } else {
      expected.push_back(shift("s_ashr_i32", "v_ashrrev_i32", temp, a, 31));
      expected.push_back(
          shift("s_lshr_b32", "v_lshrrev_b32", temp, temp, 32 - exponent));
    }
    expected.push_back(
        inst(scalar ? "s_add_u32" : "v_add_u32", {temp, a, temp}));
    if (quotient) {
      expected.push_back(
          shift("s_ashr_i32", "v_ashrrev_i32", dst, temp, exponent));
      return expected;
    }
    std::string mask = std::to_string(-(1 << exponent));
    expected.push_back(scalar ? inst("s_and_b32", {temp, temp, mask})
                              : inst("v_and_b32", {temp, mask, temp}));
    expected.push_back(
        inst(scalar ? "s_sub_u32" : "v_sub_u32", {dst, a, temp}));
    return expected;
  }

  // Checks the shift expansion of every form, for both useIEEE values.
  void expectPowerOfTwoForms(const DivisionCase &divisionCase, bool quotient) {
    for (Form form : allForms)
      for (int32_t exponent : {0, 1, 2, 5, 30}) {
        SCOPED_TRACE(std::string(formName(form)) + " divisor 2^" +
                     std::to_string(exponent));
        DivRegisters regs = makeRegisters(form);
        RegisterKind dstKind = regs.dst.ref().kind;
        std::string temp =
            std::string(dstKind == RegisterKind::sgpr ? "s" : "v") +
            std::to_string(nextFreeIndex(dstKind));
        std::string exact = emitDivision(divisionCase, regs.dst.ref(), regs.a,
                                         1 << exponent, true);
        expectInstructions(
            exact, powerOfTwoInstructions(form == Form::scalar, quotient,
                                          text(regs.dst), text(regs.a), temp,
                                          exponent));
        EXPECT_EQ(exact, emitDivision(divisionCase, regs.dst.ref(), regs.a,
                                      1 << exponent, false));
      }
  }
};

} // namespace

static std::string mnemonicOf(const std::string &instruction) {
  return instruction.substr(0, instruction.find(' '));
}

// The operands of an instruction, without a trailing comment, split at commas.
static std::vector<std::string> operandsOf(const std::string &instruction) {
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
static std::string destinationOf(const std::string &instruction) {
  std::string mnemonic = mnemonicOf(instruction);
  if (mnemonic == "s_nop" || mnemonic.rfind("s_cmp", 0) == 0)
    return "";
  std::vector<std::string> operands = operandsOf(instruction);
  return operands.empty() ? "" : operands.front();
}

// The source operands without negation or absolute-value modifiers.
static std::vector<std::string> sourcesOf(const std::string &instruction) {
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

static int32_t occurrences(const std::string &text, std::string_view needle) {
  int32_t count = 0;
  for (size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size()))
    ++count;
  return count;
}

static int32_t indexOfMnemonic(const std::vector<std::string> &instructions,
                               std::string_view mnemonic, int32_t from = 0) {
  for (int32_t i = from; i < std::ssize(instructions); ++i)
    if (mnemonicOf(instructions[i]) == mnemonic)
      return i;
  return -1;
}

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
    int32_t fma = indexOfMnemonic(instructions, "v_fma_f32");
    ASSERT_GE(fma, 0);
    EXPECT_NE(instructions[fma].find(", -v"), std::string::npos);
  }
}

TEST_F(DivTest, ExactRemainderSaluForScalarOperands) {
  DivRegisters regs = makeRegisters(Form::scalar);
  std::string out =
      emitDivision(divisionCases[1], regs.dst.ref(), regs.a, regs.b, true);
  expectLinesInOrder(out,
                     {"s_abs_i32",           "s_abs_i32",       "s_ashr_i32",
                      "v_cvt_f32_u32",       "v_rcp_iflag_f32", "s_sub_u32",
                      "v_mul_f32",           "v_cvt_u32_f32",   "s_nop 0",
                      "v_readfirstlane_b32", "s_mul_i32",       "s_mul_hi_u32",
                      "s_add_u32",           "s_mul_hi_u32",    "s_mul_i32",
                      "s_sub_u32",           "s_sub_u32",       "s_cmp_ge_u32",
                      "s_cselect_b32",       "s_sub_u32",       "s_cmp_ge_u32",
                      "s_cselect_b32",       "s_xor_b32",       "s_sub_u32"});
  EXPECT_EQ(instructionCount(out), 24);
  // The sign is the dividend's alone, and the corrections never touch q, so
  // the Newton step is the only add.
  EXPECT_EQ(countLinesContaining(out, "s_xor_b32"), 1);
  EXPECT_EQ(countLinesContaining(out, ", 31"), 1);
  EXPECT_EQ(countLinesContaining(out, "s_add_u32"), 1);
  EXPECT_EQ(countLinesContaining(out, "s_cselect_b32"), 2);
  std::vector<std::string> instructions = instructionsOf(out);
  EXPECT_EQ(mnemonicOf(instructions.back()), "s_sub_u32");
  EXPECT_EQ(destinationOf(instructions.back()), text(regs.dst));
}

TEST_F(DivTest, ExactRemainderValuForVectorDestination) {
  for (Form form : {Form::vector, Form::mixed}) {
    SCOPED_TRACE(formName(form));
    DivRegisters regs = makeRegisters(form);
    std::string out =
        emitDivision(divisionCases[1], regs.dst.ref(), regs.a, regs.b, true);
    expectLinesInOrder(
        out,
        {"v_sub_u32",     "v_max_i32",     "v_sub_u32",       "v_max_i32",
         "v_ashrrev_i32", "v_cvt_f32_u32", "v_rcp_iflag_f32", "v_sub_u32",
         "v_mul_f32",     "v_cvt_u32_f32", "v_mul_lo_u32",    "v_mul_hi_u32",
         "v_add_u32",     "v_mul_hi_u32",  "v_mul_lo_u32",    "v_sub_u32",
         "v_sub_u32",     "v_cmp_ge_u32",  "s_nop 1",         "v_cndmask_b32",
         "v_sub_u32",     "v_cmp_ge_u32",  "s_nop 1",         "v_cndmask_b32",
         "v_xor_b32",     "v_sub_u32"});
    EXPECT_EQ(countLinesContaining(out, "v_xor_b32"), 1);
    EXPECT_EQ(countLinesContaining(out, ", 31"), 1);
    EXPECT_EQ(countLinesContaining(out, "v_add_u32"), 1);
    EXPECT_EQ(countLinesContaining(out, "v_cndmask_b32"), 2);
    EXPECT_EQ(countLinesContaining(out, "v_readfirstlane"), 0);
    std::vector<std::string> instructions = instructionsOf(out);
    EXPECT_EQ(mnemonicOf(instructions.back()), "v_sub_u32");
    EXPECT_EQ(destinationOf(instructions.back()), text(regs.dst));
  }
}

TEST_F(DivTest, FastRemainderSaluForScalarOperands) {
  DivRegisters regs = makeRegisters(Form::scalar);
  std::string out =
      emitDivision(divisionCases[1], regs.dst.ref(), regs.a, regs.b, false);
  expectLinesInOrder(out,
                     {"s_xor_b32", "s_ashr_i32", "s_or_b32", "v_cvt_f32_i32",
                      "v_rcp_iflag_f32", "v_cvt_f32_i32", "v_mul_f32",
                      "v_cvt_i32_f32", "v_cvt_f32_i32", "v_fma_f32",
                      "v_cmp_ge_f32", "v_readfirstlane_b32", "s_and_b64",
                      "s_cselect_b32", "s_add_u32", "s_mul_i32", "s_sub_u32"});
  std::vector<std::string> instructions = instructionsOf(out);
  EXPECT_EQ(mnemonicOf(instructions.back()), "s_sub_u32");
  EXPECT_EQ(destinationOf(instructions.back()), text(regs.dst));
}

TEST_F(DivTest, FastRemainderValuForVectorDestination) {
  for (Form form : {Form::vector, Form::mixed}) {
    SCOPED_TRACE(formName(form));
    DivRegisters regs = makeRegisters(form);
    std::string out =
        emitDivision(divisionCases[1], regs.dst.ref(), regs.a, regs.b, false);
    expectLinesInOrder(
        out, {"v_xor_b32", "v_ashrrev_i32", "v_or_b32", "v_cvt_f32_i32",
              "v_rcp_iflag_f32", "v_cvt_f32_i32", "v_mul_f32", "v_cvt_i32_f32",
              "v_cvt_f32_i32", "v_fma_f32", "v_cmp_ge_f32", "s_nop 1",
              "v_cndmask_b32", "v_add_u32", "v_mul_lo_u32", "v_sub_u32"});
    std::vector<std::string> instructions = instructionsOf(out);
    EXPECT_EQ(mnemonicOf(instructions.back()), "v_sub_u32");
    EXPECT_EQ(destinationOf(instructions.back()), text(regs.dst));
  }
}

TEST_F(DivTest, PowerOfTwoDivisorUsesShifts) {
  expectPowerOfTwoForms(divisionCases[0], true);
}

TEST_F(DivTest, PowerOfTwoRemainderUsesMask) {
  expectPowerOfTwoForms(divisionCases[1], false);
}

TEST_F(DivTest, PowerOfTwoDestinationIsWrittenLast) {
  for (const DivisionCase &c : divisionCases)
    for (Form form : {Form::scalar, Form::vector})
      for (int32_t exponent : {0, 1, 2, 30}) {
        SCOPED_TRACE(std::string(c.name) + " " + formName(form) +
                     " divisor 2^" + std::to_string(exponent));
        DivRegisters regs = makeRegisters(form);
        std::vector<std::string> instructions = instructionsOf(
            emitDivision(c, regs.a.ref(), regs.a, 1 << exponent, true));
        ASSERT_FALSE(instructions.empty());
        EXPECT_EQ(destinationOf(instructions.back()), text(regs.a));
        for (int64_t i = 0; i + 1 < std::ssize(instructions); ++i)
          EXPECT_NE(destinationOf(instructions[i]), text(regs.a))
              << instructions[i];
      }
}

// Negative powers of two and INT32_MIN are not shifts, and a negative or zero
// divisor keeps its expansion too: only constants >= 3 take the magic path.
TEST_F(DivTest, NonPositivePowersOfTwoKeepGeneralExpansion) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false})
      for (Form form : {Form::scalar, Form::vector})
        for (int32_t divisor : {-4, -(1 << 30), INT32_MIN, -3, -6, -1, 0}) {
          SCOPED_TRACE(std::string(c.name) + " " + formName(form) +
                       (useIEEE ? " exact" : " fast") + " divisor " +
                       std::to_string(divisor));
          DivRegisters regs = makeRegisters(form);
          std::string out =
              emitDivision(c, regs.dst.ref(), regs.a, divisor, useIEEE);
          EXPECT_EQ(countLinesContaining(out, "v_rcp_iflag_f32"), 1);
        }
}

// The mnemonics of the instructions of out, in order.
static std::vector<std::string> mnemonicsOf(const std::string &out) {
  std::vector<std::string> mnemonics;
  for (const std::string &instruction : instructionsOf(out))
    mnemonics.push_back(mnemonicOf(instruction));
  return mnemonics;
}

namespace {

// The expansion of a / divisor as mnemonics, for a divisor whose magic
// multiplier is positive or negative and whose shift is zero or not. A
// multiplier is a literal, so the VALU forms copy it first.
struct MagicCase {
  int32_t divisor;
  std::vector<std::string> scalar;
  std::vector<std::string> vector;
};

} // namespace

static const MagicCase magicCases[] = {
    // M > 0, shift 0.
    {3,
     {"s_mul_hi_i32", "s_lshr_b32", "s_add_u32"},
     {"v_mov_b32", "v_mul_hi_i32", "v_lshrrev_b32", "v_add_u32"}},
    // M > 0, shift 1.
    {5,
     {"s_mul_hi_i32", "s_ashr_i32", "s_lshr_b32", "s_add_u32"},
     {"v_mov_b32", "v_mul_hi_i32", "v_ashrrev_i32", "v_lshrrev_b32",
      "v_add_u32"}},
    // M < 0, shift 2: the dividend is added back.
    {7,
     {"s_mul_hi_i32", "s_add_u32", "s_ashr_i32", "s_lshr_b32", "s_add_u32"},
     {"v_mov_b32", "v_mul_hi_i32", "v_add_u32", "v_ashrrev_i32",
      "v_lshrrev_b32", "v_add_u32"}},
};

TEST_F(DivTest, ConstantDivisorQuotientSequence) {
  for (const MagicCase &current : magicCases)
    for (bool useIEEE : {true, false})
      for (Form form : allForms) {
        SCOPED_TRACE(std::to_string(current.divisor) + " " + formName(form));
        DivRegisters regs = makeRegisters(form);
        std::string out = emitDivision(divisionCases[0], regs.dst.ref(), regs.a,
                                       current.divisor, useIEEE);
        EXPECT_EQ(mnemonicsOf(out),
                  form == Form::scalar ? current.scalar : current.vector);
      }
}

TEST_F(DivTest, ConstantDivisorRemainderSequence) {
  for (const MagicCase &current : magicCases)
    for (Form form : allForms) {
      SCOPED_TRACE(std::to_string(current.divisor) + " " + formName(form));
      DivRegisters regs = makeRegisters(form);
      std::string out = emitDivision(divisionCases[1], regs.dst.ref(), regs.a,
                                     current.divisor, true);
      // The quotient goes to a temporary, which the multiply and the
      // subtraction consume.
      std::vector<std::string> expected =
          form == Form::scalar ? current.scalar : current.vector;
      expected.push_back(form == Form::scalar ? "s_mul_i32" : "v_mul_lo_u32");
      expected.push_back(form == Form::scalar ? "s_sub_u32" : "v_sub_u32");
      EXPECT_EQ(mnemonicsOf(out), expected);
    }
}

TEST_F(DivTest, ConstantDivisorOperands) {
  // The scalar form of divisor 7: dst is the working register, then the sign.
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  std::string out = emitDiv(dst.ref(), a, 7, true);
  expectInstructions(out,
                     {inst("s_mul_hi_i32", {text(dst), text(a), "-1840700269"}),
                      inst("s_add_u32", {text(dst), text(dst), text(a)}),
                      inst("s_ashr_i32", {text(dst), text(dst), "2"}),
                      inst("s_lshr_b32", {"s2", text(dst), "31"}),
                      inst("s_add_u32", {text(dst), text(dst), "s2"})});
  // The remainder takes the quotient from a temporary: s2 holds the quotient
  // and s3 the sign, since the sign is allocated after the quotient.
  out = emitDivision(divisionCases[1], dst.ref(), a, 7, true);
  std::vector<std::string> instructions = instructionsOf(out);
  ASSERT_EQ(instructions.size(), 7u);
  EXPECT_EQ(destinationOf(instructions[0]), "s2");
  EXPECT_EQ(instructions[5].rfind("s_mul_i32 s2, s2, 7", 0), 0u);
  EXPECT_EQ(instructions[6].rfind(
                "s_sub_u32 " + text(dst) + ", " + text(a) + ", s2", 0),
            0u);
}

TEST_F(DivTest, ConstantDivisorsIgnoreUseIEEE) {
  for (const DivisionCase &c : divisionCases)
    for (Form form : allForms)
      for (int32_t divisor : {3, 6, 7, 100, 641, 65537, INT32_MAX}) {
        SCOPED_TRACE(std::string(c.name) + " " + formName(form) + " divisor " +
                     std::to_string(divisor));
        DivRegisters regs = makeRegisters(form);
        EXPECT_EQ(emitDivision(c, regs.dst.ref(), regs.a, divisor, true),
                  emitDivision(c, regs.dst.ref(), regs.a, divisor, false));
      }
}

TEST_F(DivTest, ConstantDivisorsAreMagicNotReciprocals) {
  for (const DivisionCase &c : divisionCases)
    for (Form form : allForms)
      for (int32_t divisor : {3, 5, 6, 7, 10, 100, 641, 65537, INT32_MAX}) {
        SCOPED_TRACE(std::string(c.name) + " " + formName(form) + " divisor " +
                     std::to_string(divisor));
        DivRegisters regs = makeRegisters(form);
        std::string out =
            emitDivision(c, regs.dst.ref(), regs.a, divisor, true);
        // There is no float, readfirstlane, compare, exec or VCC use, so no
        // active-lane precondition either.
        for (const char *forbidden :
             {"v_rcp", "v_cvt", "v_readfirstlane", "s_cmp", "v_cmp", "exec",
              "vcc", "cndmask", "cselect", "s_nop"})
          EXPECT_EQ(countLinesContaining(out, forbidden), 0) << forbidden;
        EXPECT_GE(countLinesContaining(out, "mul_hi_i32"), 1);
      }
}

TEST_F(DivTest, ConstantDivisorDestinationIsWrittenLast) {
  for (const DivisionCase &c : divisionCases)
    for (Form form : {Form::scalar, Form::vector})
      for (int32_t divisor : {3, 5, 7, 641, INT32_MAX}) {
        SCOPED_TRACE(std::string(c.name) + " " + formName(form) + " divisor " +
                     std::to_string(divisor));
        DivRegisters regs = makeRegisters(form);
        std::vector<std::string> instructions = instructionsOf(
            emitDivision(c, regs.a.ref(), regs.a, divisor, true));
        ASSERT_FALSE(instructions.empty());
        EXPECT_EQ(destinationOf(instructions.back()), text(regs.a));
        for (int64_t i = 0; i + 1 < std::ssize(instructions); ++i)
          EXPECT_NE(destinationOf(instructions[i]), text(regs.a))
              << instructions[i];
        // Without aliasing, the remainder still writes dst last.
        instructions = instructionsOf(
            emitDivision(c, regs.dst.ref(), regs.a, divisor, true));
        EXPECT_EQ(destinationOf(instructions.back()), text(regs.dst));
      }
}

TEST_F(DivTest, ConstantDivisorImmediateDividend) {
  for (const DivisionCase &c : divisionCases)
    for (Form form : {Form::scalar, Form::vector}) {
      SCOPED_TRACE(std::string(c.name) + " " + formName(form));
      DivRegisters regs = makeRegisters(form);
      for (int32_t dividend : {0, 5, literalA, -1, INT32_MIN, INT32_MAX}) {
        SCOPED_TRACE(dividend);
        std::string out = emitDivision(c, regs.dst.ref(), dividend, 7, true);
        std::vector<std::string> instructions = instructionsOf(out);
        ASSERT_FALSE(instructions.empty());
        EXPECT_EQ(destinationOf(instructions.back()), text(regs.dst));
        EXPECT_EQ(countLinesContaining(out, "v_rcp"), 0);
      }
    }
}

TEST_F(DivTest, ConstantDivisorReleasesTemporaries) {
  for (const DivisionCase &c : divisionCases)
    for (Form form : allForms)
      for (bool alias : {false, true}) {
        SCOPED_TRACE(std::string(c.name) + " " + formName(form) +
                     (alias ? " dst is a" : " dst is separate"));
        DivRegisters regs = makeRegisters(form);
        RegisterRef dst =
            alias && form != Form::mixed ? regs.a.ref() : regs.dst.ref();
        int32_t vgprsBefore = nextFreeIndex(RegisterKind::vgpr);
        int32_t sgprsBefore = nextFreeIndex(RegisterKind::sgpr);
        emitDivision(c, dst, regs.a, 7, true);
        emitDivision(c, dst, regs.a, 641, true);
        emitDivision(c, dst, literalA, 7, true);
        EXPECT_EQ(nextFreeIndex(RegisterKind::vgpr), vgprsBefore);
        EXPECT_EQ(nextFreeIndex(RegisterKind::sgpr), sgprsBefore);
      }
}

TEST_F(DivTest, NeverWritesExecOrVcc) {
  for (const DivisionCase &divisionCase : divisionCases)
    for (bool useIEEE : {true, false})
      for (Form form : allForms) {
        SCOPED_TRACE(std::string(divisionCase.name) + " " + formName(form) +
                     (useIEEE ? " exact" : " fast"));
        std::string out = emitForm(divisionCase, form, useIEEE);
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
static void
expectReciprocalHazardCovered(const std::vector<std::string> &instructions) {
  int32_t rcp = indexOfMnemonic(instructions, "v_rcp_iflag_f32");
  ASSERT_GE(rcp, 0);
  ASSERT_LT(rcp + 1, std::ssize(instructions));
  const std::string &next = instructions[rcp + 1];
  EXPECT_NE(mnemonicOf(next), "s_nop") << next;
  std::string reciprocal = destinationOf(instructions[rcp]);
  for (const std::string &source : sourcesOf(next))
    EXPECT_NE(source, reciprocal) << next;
}

static void
expectMaskNopBeforeCndmask(const std::vector<std::string> &instructions) {
  int32_t checked = 0;
  for (int32_t i = 0; i < std::ssize(instructions); ++i) {
    if (mnemonicOf(instructions[i]).rfind("v_cmp_", 0) != 0)
      continue;
    ASSERT_LT(i + 1, std::ssize(instructions));
    EXPECT_EQ(instructions[i + 1].rfind("s_nop 1", 0), 0u) << instructions[i];
    EXPECT_GE(indexOfMnemonic(instructions, "v_cndmask_b32", i + 1), 0);
    ++checked;
  }
  EXPECT_GT(checked, 0);
}

TEST_F(DivTest, HazardWaitStates) {
  for (const DivisionCase &divisionCase : divisionCases)
    for (bool useIEEE : {true, false})
      for (Form form : allForms) {
        SCOPED_TRACE(std::string(divisionCase.name) + " " + formName(form) +
                     (useIEEE ? " exact" : " fast"));
        std::vector<std::string> instructions =
            instructionsOf(emitForm(divisionCase, form, useIEEE));
        expectReciprocalHazardCovered(instructions);
        if (form != Form::scalar) {
          expectMaskNopBeforeCndmask(instructions);
          continue;
        }
        int32_t readLane = indexOfMnemonic(instructions, "v_readfirstlane_b32");
        ASSERT_GT(readLane, 0);
        if (useIEEE) {
          EXPECT_EQ(instructions[readLane - 1].rfind("s_nop 0", 0), 0u);
          continue;
        }
        int32_t convert = indexOfMnemonic(instructions, "v_cvt_i32_f32");
        ASSERT_GE(convert, 0);
        EXPECT_GT(readLane - convert, 1);
        EXPECT_EQ(sourcesOf(instructions[readLane]).front(),
                  destinationOf(instructions[convert]));
      }
}

// Every VOP3-only or VOP3-promoted line must be free of the literal.
static void expectLiteralOnlyInMoves(const std::string &out,
                                     const char *literal) {
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
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      SCOPED_TRACE(std::string(c.name) + (useIEEE ? " exact" : " fast"));
      // A negative literal divisor, since a positive one takes the magic path.
      std::string divisor =
          emitDivision(c, dst.ref(), value, -literalA, useIEEE);
      EXPECT_EQ(countLinesContaining(divisor, "v_mov_b32"), 1);
      expectLiteralOnlyInMoves(divisor, literalAText);
      std::string dividend =
          emitDivision(c, dst.ref(), literalB, value, useIEEE);
      EXPECT_EQ(countLinesContaining(dividend, "v_mov_b32"), 1);
      expectLiteralOnlyInMoves(dividend, literalBText);
      for (int32_t inlineValue : {-7, -5})
        EXPECT_EQ(countLinesContaining(
                      emitDivision(c, dst.ref(), value, inlineValue, useIEEE),
                      "v_mov"),
                  0)
            << inlineValue;
    }
}

TEST_F(DivTest, ImmediateOperandsInExactScalarForm) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange value = builder.allocateSgpr(1);
  for (const DivisionCase &c : divisionCases) {
    SCOPED_TRACE(c.name);
    std::string out = emitDivision(c, dst.ref(), value, -7, true);
    expectLinesInOrder(out, {"s_mov_b32", "s_abs_i32"});
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
    out = emitDivision(c, dst.ref(), 7, value, true);
    expectLinesInOrder(out, {"s_mov_b32", "s_abs_i32"});
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
    EXPECT_EQ(countLinesContaining(
                  emitDivision(c, dst.ref(), literalA, -5, true), "s_mov_b32"),
              2);
  }
}

TEST_F(DivTest, ImmediateOperandsInFastScalarForm) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange value = builder.allocateSgpr(1);
  for (const DivisionCase &c : divisionCases) {
    SCOPED_TRACE(c.name);
    std::string out = emitDivision(c, dst.ref(), value, -literalA, false);
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0);
    int32_t convertsWithLiteral = 0;
    int32_t xorsWithLiteral = 0;
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
    out = emitDivision(c, dst.ref(), literalA, -literalB, false);
    EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
    std::vector<std::string> instructions = instructionsOf(out);
    int32_t copy = indexOfMnemonic(instructions, "s_mov_b32");
    ASSERT_GE(copy, 0);
    EXPECT_NE(instructions[copy].find(literalBText), std::string::npos);
  }
}

TEST_F(DivTest, DestinationAliasesOperand) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false})
      for (Form form : {Form::scalar, Form::vector}) {
        for (bool aliasFirst : {true, false}) {
          SCOPED_TRACE(std::string(c.name) + " " + formName(form) +
                       (useIEEE ? " exact" : " fast") +
                       (aliasFirst ? " dst is a" : " dst is b"));
          DivRegisters regs = makeRegisters(form);
          RegisterRef dst = aliasFirst ? regs.a.ref() : regs.b.ref();
          std::vector<std::string> instructions =
              instructionsOf(emitDivision(c, dst, regs.a, regs.b, useIEEE));
          ASSERT_FALSE(instructions.empty());
          EXPECT_EQ(destinationOf(instructions.back()), text(dst));
          for (int64_t i = 0; i + 1 < std::ssize(instructions); ++i)
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
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      SCOPED_TRACE(std::string(c.name) + (useIEEE ? " exact" : " fast"));
      auto expectRejectedDivision = [&](RegisterRef dst, Operand a, Operand b) {
        expectRejected([&] { (builder.*c.op)(dst, a, b, useIEEE); });
      };
      expectRejectedDivision(acc.ref(), va, vb);
      expectRejectedDivision(vdst.ref(), acc, vb);
      expectRejectedDivision(vdst.ref(), va, acc);
      expectRejectedDivision(pair.ref(), va, vb);
      expectRejectedDivision(vdst.ref(), pair, vb);
      expectRejectedDivision(vdst.ref(), va, pair);
      expectRejectedDivision(sdst.ref(), va, sa);
      expectRejectedDivision(sdst.ref(), sa, vb);
    }
}

TEST_F(DivTest, ScalarDestinationFromVectorOperandMessage) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange vb = builder.allocateVgpr(1);
  for (const DivisionCase &c : divisionCases) {
    SCOPED_TRACE(c.name);
    std::string expected = std::string(c.name) +
                           " cannot write sgpr destination " + text(sdst) +
                           " from vgpr operand " + text(vb);
    try {
      (builder.*c.op)(sdst.ref(), sa, vb, true);
      FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &error) {
      EXPECT_EQ(std::string(error.what()), expected);
    }
  }
}

TEST_F(DivTest, ReleasesTemporaries) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false})
      for (Form form : allForms) {
        SCOPED_TRACE(std::string(c.name) + " " + formName(form) +
                     (useIEEE ? " exact" : " fast"));
        DivRegisters regs = makeRegisters(form);
        int32_t vgprsBefore = nextFreeIndex(RegisterKind::vgpr);
        int32_t sgprsBefore = nextFreeIndex(RegisterKind::sgpr);
        emitDivision(c, regs.dst.ref(), regs.a, regs.b, useIEEE);
        EXPECT_EQ(nextFreeIndex(RegisterKind::vgpr), vgprsBefore);
        EXPECT_EQ(nextFreeIndex(RegisterKind::sgpr), sgprsBefore);
        // Literals take temporaries too.
        emitDivision(c, regs.dst.ref(), literalA, literalB, useIEEE);
        EXPECT_EQ(nextFreeIndex(RegisterKind::vgpr), vgprsBefore);
        EXPECT_EQ(nextFreeIndex(RegisterKind::sgpr), sgprsBefore);
      }
}

static constexpr uint32_t divRandomSeed = 20240607;
// The fast path is exact for magnitudes below this bound, LLVM's signed 24-bit
// criterion.
static constexpr int64_t fastBound = int64_t{1} << 23;
static constexpr int32_t exactRandomCount = 131072;
static constexpr int32_t fastBoundaryCount = 262144;
static constexpr int32_t fastRandomCount = 1048576;

using DivDevice = DeviceTest;

// A zero divisor is replaced by 1.
static void addPair(WordPairs &pairs, int64_t a, int64_t b) {
  pairs.a.push_back(static_cast<uint32_t>(a));
  pairs.b.push_back(b == 0 ? 1u : static_cast<uint32_t>(b));
}

// A value with 0 to maxBits - 1 low bits kept, so that every magnitude occurs.
static uint32_t randomMagnitude(std::mt19937 &engine, int32_t maxBits) {
  uint32_t mask = maxBits == 32 ? ~0u : (1u << maxBits) - 1;
  return (engine() & mask) >> (engine() % maxBits);
}

static int64_t withRandomSign(std::mt19937 &engine, int64_t magnitude) {
  return (engine() & 1) != 0 ? -magnitude : magnitude;
}

// Pads with random pairs to whole workgroups of lanes.
static void padToWorkgroups(WordPairs &pairs, std::mt19937 &engine,
                            int32_t maxBits) {
  while (pairs.a.size() % arithWorkgroupSize != 0)
    addPair(pairs, withRandomSign(engine, randomMagnitude(engine, maxBits)),
            withRandomSign(engine, randomMagnitude(engine, maxBits)));
}

static std::vector<int64_t> exactEdges() {
  std::vector<int64_t> edges = {0,          1,
                                -1,         2,
                                -2,         3,
                                -3,         7,
                                -7,         INT32_MIN,
                                INT32_MAX,  INT32_MIN + 1,
                                0x12345678, static_cast<int32_t>(0x9E3779B9),
                                0x7FFFFFFE, -0x7FFFFFFE};
  for (int32_t bit = 2; bit <= 30; ++bit) {
    edges.push_back(int64_t{1} << bit);
    edges.push_back(-(int64_t{1} << bit));
  }
  for (int64_t delta : {-1, 0, 1}) {
    edges.push_back(fastBound + delta);
    edges.push_back(-(fastBound + delta));
  }
  for (int64_t delta : {-1, 0, 1}) {
    edges.push_back((int64_t{1} << 24) + delta);
    edges.push_back(-((int64_t{1} << 24) + delta));
  }
  return edges;
}

// Divisors near |a|, and some larger than |a|.
static void addNeighborDivisors(WordPairs &pairs, int64_t a) {
  for (int64_t b : {a - 1, a, a + 1, -a, a / 2, a * 2 + 1, -(a * 2 + 1)}) {
    if (b < INT32_MIN || b > INT32_MAX)
      continue;
    addPair(pairs, a, b);
  }
}

static void addExactEdgePairs(WordPairs &pairs) {
  std::vector<int64_t> edges = exactEdges();
  for (int64_t a : edges) {
    for (int64_t b : edges)
      addPair(pairs, a, b);
    addNeighborDivisors(pairs, a);
  }
}

// Random 32-bit pairs, then a random with a small divisor, then both shifted.
static void addExactRandomPairs(WordPairs &pairs, std::mt19937 &engine,
                                int32_t count) {
  for (int32_t i = 0; i < count; ++i) {
    switch (i % 3) {
    case 0:
      addPair(pairs, static_cast<int32_t>(engine()),
              static_cast<int32_t>(engine()));
      break;
    case 1:
      addPair(pairs, static_cast<int32_t>(engine()),
              withRandomSign(engine, 1 + engine() % 1024));
      break;
    default:
      addPair(pairs, withRandomSign(engine, randomMagnitude(engine, 32)),
              withRandomSign(engine, randomMagnitude(engine, 32)));
      break;
    }
  }
}

static const WordPairs &exactPairs() {
  static const WordPairs pairs = [] {
    WordPairs result;
    std::mt19937 engine(divRandomSeed);
    addExactEdgePairs(result);
    addExactRandomPairs(result, engine, exactRandomCount);
    padToWorkgroups(result, engine, 32);
    return result;
  }();
  return pairs;
}

static std::vector<int64_t> fastEdges() {
  std::vector<int64_t> edges = {0,
                                1,
                                2,
                                3,
                                fastBound - 1,
                                fastBound - 2,
                                1 << 22,
                                (1 << 22) + 1,
                                (1 << 22) - 1,
                                4096,
                                0x6BCDEF};
  std::vector<int64_t> signedEdges;
  for (int64_t edge : edges) {
    signedEdges.push_back(edge);
    signedEdges.push_back(-edge);
  }
  return signedEdges;
}

static void addFastEdgePairs(WordPairs &pairs) {
  std::vector<int64_t> edges = fastEdges();
  for (int64_t a : edges)
    for (int64_t b : edges)
      addPair(pairs, a, b);
  for (int64_t a : std::vector<int64_t>{fastBound - 1, fastBound - 2,
                                        (1 << 22) + 1, 0x6BCDEF})
    for (int64_t b = 1; b <= 4096; ++b)
      for (int64_t signedA : {a, -a})
        for (int64_t signedB : {b, -b})
          addPair(pairs, signedA, signedB);
}

// WordPairs a = k * b + d for d in {-1, 0, 1}, where the exact quotient
// changes.
static void addBoundaryPairs(WordPairs &pairs, std::mt19937 &engine,
                             int32_t count) {
  for (int32_t i = 0; i < count; ++i) {
    int64_t b = 1 + randomMagnitude(engine, 23) % (fastBound - 1);
    int64_t k = engine() % ((fastBound - 2) / b + 1);
    int64_t delta = static_cast<int64_t>(engine() % 3) - 1;
    int64_t a = k * b + delta;
    if (a < 0 || a >= fastBound)
      a = k * b;
    addPair(pairs, withRandomSign(engine, a), withRandomSign(engine, b));
  }
}

// Uniform 23-bit magnitudes, alternating with shifted ones.
static void addFastRandomPairs(WordPairs &pairs, std::mt19937 &engine,
                               int32_t count) {
  for (int32_t i = 0; i < count; ++i) {
    bool shifted = (i & 1) != 0;
    uint32_t a = shifted ? randomMagnitude(engine, 23) : engine() & 0x7FFFFF;
    uint32_t b = shifted ? randomMagnitude(engine, 23) : engine() & 0x7FFFFF;
    addPair(pairs, withRandomSign(engine, a), withRandomSign(engine, b));
  }
}

static const WordPairs &fastPairs() {
  static const WordPairs pairs = [] {
    WordPairs result;
    std::mt19937 engine(divRandomSeed + 1);
    addFastEdgePairs(result);
    addBoundaryPairs(result, engine, fastBoundaryCount);
    addFastRandomPairs(result, engine, fastRandomCount);
    padToWorkgroups(result, engine, 23);
    return result;
  }();
  return pairs;
}

static const WordPairs &pairsFor(bool useIEEE) {
  return useIEEE ? exactPairs() : fastPairs();
}

static const char *pathName(bool useIEEE) { return useIEEE ? "exact" : "fast"; }

// Every stride-th pair, so that a small sample still spans the whole set.
static WordPairs sample(const WordPairs &pairs, int64_t count) {
  WordPairs result;
  int64_t stride = std::max<int64_t>(1, std::ssize(pairs.a) / count);
  for (int64_t i = 0; i < std::ssize(pairs.a) && std::ssize(result.a) < count;
       i += stride) {
    result.a.push_back(pairs.a[i]);
    result.b.push_back(pairs.b[i]);
  }
  return result;
}

static std::vector<uint32_t> expectedResults(const DivisionCase &c,
                                             const WordPairs &pairs) {
  std::vector<uint32_t> expected;
  for (int64_t i = 0; i < std::ssize(pairs.a); ++i)
    expected.push_back(c.host(pairs.a[i], pairs.b[i]));
  return expected;
}

static ArithKernelSpec divSpec(const DivisionCase &c, InputKind aKind,
                               InputKind bKind, RegisterKind result,
                               bool useIEEE) {
  return {{aKind, bKind},
          result,
          [op = c.op, useIEEE](IRBuilder &builder, RegisterRef dst,
                               const std::vector<RegisterRef> &inputs) {
            (builder.*op)(dst, inputs[0], inputs[1], useIEEE);
          }};
}

TEST_F(DivDevice, ValuLaneOperands) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      SCOPED_TRACE(std::string(c.name) + " " + pathName(useIEEE));
      const WordPairs &pairs = pairsFor(useIEEE);
      expectArithResults(divSpec(c, InputKind::lane, InputKind::lane,
                                 RegisterKind::vgpr, useIEEE),
                         laneGroupCount(pairs.a), {pairs.a, pairs.b},
                         expectedResults(c, pairs));
    }
}

TEST_F(DivDevice, SaluUniformOperands) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      SCOPED_TRACE(std::string(c.name) + " " + pathName(useIEEE));
      WordPairs pairs = sample(pairsFor(useIEEE), 32768);
      expectArithResults(divSpec(c, InputKind::uniform, InputKind::uniform,
                                 RegisterKind::sgpr, useIEEE),
                         static_cast<int32_t>(pairs.a.size()),
                         {pairs.a, pairs.b}, expectedResults(c, pairs));
    }
}

TEST_F(DivDevice, ScalarsIntoVgpr) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      SCOPED_TRACE(std::string(c.name) + " " + pathName(useIEEE));
      WordPairs pairs = sample(pairsFor(useIEEE), 16384);
      expectArithResults(divSpec(c, InputKind::uniform, InputKind::uniform,
                                 RegisterKind::vgpr, useIEEE),
                         static_cast<int32_t>(pairs.a.size()),
                         {pairs.a, pairs.b},
                         perLane(expectedResults(c, pairs)));
    }
}

// The first lane of each workgroup supplies the uniform operand.
static std::vector<uint32_t>
firstLanes(const std::vector<uint32_t> &laneWords) {
  std::vector<uint32_t> firstWords;
  for (int64_t i = 0; i < std::ssize(laneWords); i += arithWorkgroupSize)
    firstWords.push_back(laneWords[i]);
  return firstWords;
}

TEST_F(DivDevice, MixedScalarAndVector) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      SCOPED_TRACE(std::string(c.name) + " " + pathName(useIEEE));
      WordPairs pairs = sample(pairsFor(useIEEE), 65536);
      int32_t groups = laneGroupCount(pairs.a);
      std::vector<uint32_t> uniformA = firstLanes(pairs.a);
      std::vector<uint32_t> uniformB = firstLanes(pairs.b);
      std::vector<uint32_t> uniformFirst;
      std::vector<uint32_t> uniformSecond;
      for (int64_t i = 0; i < std::ssize(pairs.a); ++i) {
        int64_t group = i / arithWorkgroupSize;
        uniformFirst.push_back(c.host(uniformA[group], pairs.b[i]));
        uniformSecond.push_back(c.host(pairs.a[i], uniformB[group]));
      }
      expectArithResults(divSpec(c, InputKind::uniform, InputKind::lane,
                                 RegisterKind::vgpr, useIEEE),
                         groups, {uniformA, pairs.b}, uniformFirst);
      expectArithResults(divSpec(c, InputKind::lane, InputKind::uniform,
                                 RegisterKind::vgpr, useIEEE),
                         groups, {pairs.a, uniformB}, uniformSecond);
    }
}

static std::vector<int32_t> immediatesFor(bool useIEEE) {
  std::vector<int32_t> immediates = {1,   -1, 2,  -2,  3,       7,
                                     -16, 64, 65, -17, 0x12345, -100000};
  if (useIEEE) {
    immediates.push_back(INT32_MIN);
    immediates.push_back(INT32_MAX);
    return immediates;
  }
  immediates.push_back(0x7FFFFF);
  immediates.push_back(-0x7FFFFF);
  return immediates;
}

// Computes dst = input op immediate, or immediate op input.
static ArithKernelSpec divImmediateSpec(const DivisionCase &c, InputKind kind,
                                        RegisterKind result, int32_t immediate,
                                        bool immediateIsDividend,
                                        bool useIEEE) {
  return {{kind},
          result,
          [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
            if (immediateIsDividend)
              (builder.*op)(dst, immediate, inputs[0], useIEEE);
            else
              (builder.*op)(dst, inputs[0], immediate, useIEEE);
          }};
}

static std::vector<uint32_t>
divideByImmediate(const DivisionCase &c, const std::vector<uint32_t> &words,
                  int32_t immediate, bool immediateIsDividend) {
  uint32_t value = static_cast<uint32_t>(immediate);
  std::vector<uint32_t> out;
  for (uint32_t word : words)
    out.push_back(immediateIsDividend ? c.host(value, word)
                                      : c.host(word, value));
  return out;
}

// Runs the lane, uniform-into-sgpr and uniform-into-vgpr forms.
static void expectImmediateForms(const DivisionCase &c,
                                 const std::vector<uint32_t> &words,
                                 int32_t immediate, bool immediateIsDividend,
                                 bool useIEEE) {
  std::vector<uint32_t> expected =
      divideByImmediate(c, words, immediate, immediateIsDividend);
  expectArithResults(divImmediateSpec(c, InputKind::lane, RegisterKind::vgpr,
                                      immediate, immediateIsDividend, useIEEE),
                     laneGroupCount(words), {words}, expected);
  expectArithResults(divImmediateSpec(c, InputKind::uniform, RegisterKind::sgpr,
                                      immediate, immediateIsDividend, useIEEE),
                     static_cast<int32_t>(words.size()), {words}, expected);
  expectArithResults(divImmediateSpec(c, InputKind::uniform, RegisterKind::vgpr,
                                      immediate, immediateIsDividend, useIEEE),
                     static_cast<int32_t>(words.size()), {words},
                     perLane(expected));
}

TEST_F(DivDevice, ImmediateOperands) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      WordPairs pairs = sample(pairsFor(useIEEE), 1024);
      for (int32_t immediate : immediatesFor(useIEEE)) {
        SCOPED_TRACE(std::string(c.name) + " " + pathName(useIEEE) +
                     " immediate " + std::to_string(immediate));
        // The other operand comes from the set, so it is never zero.
        expectImmediateForms(c, pairs.a, immediate, false, useIEEE);
        expectImmediateForms(c, pairs.b, immediate, true, useIEEE);
      }
    }
}

// Each emitter writes into an input register and then moves it to dst.
TEST_F(DivDevice, DestinationAliasesOperand) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      SCOPED_TRACE(std::string(c.name) + " " + pathName(useIEEE));
      WordPairs pairs = sample(pairsFor(useIEEE), 4096);
      std::vector<uint32_t> expected = expectedResults(c, pairs);
      for (int32_t aliased : {0, 1}) {
        SCOPED_TRACE("alias input " + std::to_string(aliased));
        auto emit = [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                                   const std::vector<RegisterRef> &inputs) {
          (builder.*op)(inputs[aliased], inputs[0], inputs[1], useIEEE);
          builder.movb32(dst, inputs[aliased]);
        };
        expectArithResults(
            {{InputKind::lane, InputKind::lane}, RegisterKind::vgpr, emit},
            laneGroupCount(pairs.a), {pairs.a, pairs.b}, expected);
        expectArithResults({{InputKind::uniform, InputKind::uniform},
                            RegisterKind::sgpr,
                            emit},
                           static_cast<int32_t>(pairs.a.size()),
                           {pairs.a, pairs.b}, expected);
      }
    }
}

// The result of a division by zero is unspecified, so the result is zeroed
// and only the completion of the kernel, without a fault or a hang, is checked.
TEST_F(DivDevice, DivisionByZeroCompletes) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false}) {
      SCOPED_TRACE(std::string(c.name) + " " + pathName(useIEEE));
      WordPairs pairs = sample(pairsFor(useIEEE), 4096);
      std::vector<uint32_t> zeros(pairs.a.size(), 0);
      expectArithResults(
          {{InputKind::lane, InputKind::lane},
           RegisterKind::vgpr,
           [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                          const std::vector<RegisterRef> &inputs) {
             RegisterRange tmp = builder.allocateVgpr(1);
             (builder.*op)(tmp.ref(), inputs[0], inputs[1], useIEEE);
             builder.muli32(dst, tmp, 0);
           }},
          laneGroupCount(pairs.a), {pairs.a, zeros}, zeros);
      expectArithResults(
          {{InputKind::uniform, InputKind::uniform},
           RegisterKind::sgpr,
           [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                          const std::vector<RegisterRef> &inputs) {
             RegisterRange tmp = builder.allocateSgpr(1);
             (builder.*op)(tmp.ref(), inputs[0], inputs[1], useIEEE);
             builder.muli32(dst, tmp, 0);
           }},
          static_cast<int32_t>(pairs.a.size()), {pairs.a, zeros}, zeros);
    }
}

// These pairs are outside the fast-path precondition, where fa * rcp(fb) can
// round up to the next integer, so only the exact path is checked. They guard
// the bound.
TEST_F(DivDevice, ExactPathHandlesPairsAboveFastBound) {
  WordPairs pairs;
  const int64_t above[][2] = {
      {16777214, 105},      {16777214, 16777215}, {-16777214, 105},
      {0x800001, 3},        {-0x800001, 3},       {0x800001, -3},
      {16777215, 16777214}, {16777215, 105},      {-16777215, 16777214},
      {0xFFFFFE, 7},        {16777217, 105},      {-16777217, 105},
      {16777217, 16777215}};
  for (const auto &[dividend, divisor] : above)
    addPair(pairs, dividend, divisor);
  std::mt19937 engine(divRandomSeed + 2);
  for (int32_t i = 0; i < 4096; ++i)
    addPair(pairs, withRandomSign(engine, fastBound + engine() % fastBound),
            withRandomSign(engine, 1 + randomMagnitude(engine, 24)));
  padToWorkgroups(pairs, engine, 24);
  WordPairs scalar = sample(pairs, 1024);
  for (const DivisionCase &c : divisionCases) {
    SCOPED_TRACE(c.name);
    expectArithResults(
        divSpec(c, InputKind::lane, InputKind::lane, RegisterKind::vgpr, true),
        laneGroupCount(pairs.a), {pairs.a, pairs.b}, expectedResults(c, pairs));
    expectArithResults(divSpec(c, InputKind::uniform, InputKind::uniform,
                               RegisterKind::sgpr, true),
                       static_cast<int32_t>(scalar.a.size()),
                       {scalar.a, scalar.b}, expectedResults(c, scalar));
  }
}

// The dividends around every power of two, the extremes and random values.
static std::vector<uint32_t> powerOfTwoDividends() {
  std::vector<int64_t> values = {0, 1, -1, INT32_MIN, INT32_MAX, INT32_MIN + 1};
  for (int32_t exponent = 0; exponent <= 30; ++exponent)
    for (int64_t power : {int64_t{1} << exponent, -(int64_t{1} << exponent)})
      for (int64_t delta : {-1, 0, 1})
        if (power + delta >= INT32_MIN && power + delta <= INT32_MAX)
          values.push_back(power + delta);
  std::mt19937 engine(divRandomSeed + 3);
  for (int32_t i = 0; i < 4096; ++i)
    values.push_back(static_cast<int32_t>(engine()));
  while (values.size() % arithWorkgroupSize != 0)
    values.push_back(static_cast<int32_t>(engine()));
  std::vector<uint32_t> words;
  for (int64_t value : values)
    words.push_back(static_cast<uint32_t>(value));
  return words;
}

// Every stride-th word, so that a small sample still spans the whole set.
static std::vector<uint32_t> sampleWords(const std::vector<uint32_t> &words,
                                         int64_t count) {
  std::vector<uint32_t> result;
  int64_t stride = std::max<int64_t>(1, std::ssize(words) / count);
  for (int64_t i = 0; i < std::ssize(words) && std::ssize(result) < count;
       i += stride)
    result.push_back(words[i]);
  return result;
}

static ArithKernelSpec powerOfTwoSpec(const DivisionCase &c, InputKind kind,
                                      RegisterKind result, int32_t exponent) {
  return {{kind},
          result,
          [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
            // The host test shows that useIEEE does not matter here.
            (builder.*op)(dst, inputs[0], 1 << exponent, true);
          }};
}

TEST_F(DivDevice, PowerOfTwoDivisors) {
  std::vector<uint32_t> words = powerOfTwoDividends();
  std::vector<uint32_t> uniformWords = sampleWords(words, 256);
  int32_t uniformCount = static_cast<int32_t>(uniformWords.size());
  for (const DivisionCase &c : divisionCases)
    for (int32_t exponent = 0; exponent <= 30; ++exponent) {
      SCOPED_TRACE(std::string(c.name) + " divisor 2^" +
                   std::to_string(exponent));
      uint32_t divisor = 1u << exponent;
      std::vector<uint32_t> expected;
      for (uint32_t word : words)
        expected.push_back(c.host(word, divisor));
      std::vector<uint32_t> uniformExpected;
      for (uint32_t word : uniformWords)
        uniformExpected.push_back(c.host(word, divisor));
      expectArithResults(
          powerOfTwoSpec(c, InputKind::lane, RegisterKind::vgpr, exponent),
          laneGroupCount(words), {words}, expected);
      expectArithResults(
          powerOfTwoSpec(c, InputKind::uniform, RegisterKind::sgpr, exponent),
          uniformCount, {uniformWords}, uniformExpected);
      expectArithResults(
          powerOfTwoSpec(c, InputKind::uniform, RegisterKind::vgpr, exponent),
          uniformCount, {uniformWords}, perLane(uniformExpected));
    }
}

// Each emitter writes into the input register and then moves it to dst.
TEST_F(DivDevice, PowerOfTwoDestinationAliasesDividend) {
  std::vector<uint32_t> words = powerOfTwoDividends();
  std::vector<uint32_t> uniformWords = sampleWords(words, 256);
  for (const DivisionCase &c : divisionCases)
    for (int32_t exponent : {0, 1, 5, 30}) {
      SCOPED_TRACE(std::string(c.name) + " divisor 2^" +
                   std::to_string(exponent));
      auto emit = [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                                 const std::vector<RegisterRef> &inputs) {
        (builder.*op)(inputs[0], inputs[0], 1 << exponent, true);
        builder.movb32(dst, inputs[0]);
      };
      std::vector<uint32_t> expected;
      for (uint32_t word : words)
        expected.push_back(c.host(word, 1u << exponent));
      std::vector<uint32_t> uniformExpected;
      for (uint32_t word : uniformWords)
        uniformExpected.push_back(c.host(word, 1u << exponent));
      expectArithResults({{InputKind::lane}, RegisterKind::vgpr, emit},
                         laneGroupCount(words), {words}, expected);
      expectArithResults({{InputKind::uniform}, RegisterKind::sgpr, emit},
                         static_cast<int32_t>(uniformWords.size()),
                         {uniformWords}, uniformExpected);
    }
}

// Divisors that cover the magic multiplier shapes: positive and negative
// multipliers, no shift and large shifts, and the largest constant.
static constexpr int32_t constantDivisors[] = {
    3,   5,     6,       7,       10,         12,       100,
    641, 65537, 0x12345, 1000003, 0x40000001, INT32_MAX};

// The power-of-two dividends, which include the extremes, plus the multiples of
// the divisor and their neighbours around zero and at both ends of the range,
// where a quotient changes.
static std::vector<uint32_t> constantDividends(int32_t divisor) {
  std::vector<uint32_t> words = powerOfTwoDividends();
  std::vector<int64_t> multiples;
  for (int64_t k = -3; k <= 3; ++k)
    multiples.push_back(k);
  for (int64_t k = -2; k <= 2; ++k) {
    multiples.push_back(INT32_MAX / divisor + k);
    multiples.push_back(INT32_MIN / divisor + k);
  }
  for (int64_t k : multiples)
    for (int64_t delta : {-1, 0, 1}) {
      int64_t value = k * divisor + delta;
      if (value >= INT32_MIN && value <= INT32_MAX)
        words.push_back(static_cast<uint32_t>(value));
    }
  std::mt19937 engine(divRandomSeed + 4 + divisor);
  while (words.size() % arithWorkgroupSize != 0)
    words.push_back(static_cast<uint32_t>(engine()));
  return words;
}

static ArithKernelSpec constantDivisorSpec(const DivisionCase &c,
                                           InputKind kind, RegisterKind result,
                                           int32_t divisor, bool useIEEE) {
  return {{kind},
          result,
          [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
            (builder.*op)(dst, inputs[0], divisor, useIEEE);
          }};
}

static std::vector<uint32_t>
expectedByConstant(const DivisionCase &c, const std::vector<uint32_t> &words,
                   int32_t divisor) {
  std::vector<uint32_t> expected;
  for (uint32_t word : words)
    expected.push_back(c.host(word, static_cast<uint32_t>(divisor)));
  return expected;
}

TEST_F(DivDevice, ConstantDivisors) {
  for (const DivisionCase &c : divisionCases)
    for (int32_t divisor : constantDivisors) {
      SCOPED_TRACE(std::string(c.name) + " divisor " + std::to_string(divisor));
      std::vector<uint32_t> words = constantDividends(divisor);
      std::vector<uint32_t> expected = expectedByConstant(c, words, divisor);
      std::vector<uint32_t> uniformWords = sampleWords(words, 256);
      std::vector<uint32_t> uniformExpected =
          expectedByConstant(c, uniformWords, divisor);
      int32_t uniformCount = static_cast<int32_t>(uniformWords.size());
      // The result does not depend on useIEEE.
      for (bool useIEEE : {false, true})
        expectArithResults(constantDivisorSpec(c, InputKind::lane,
                                               RegisterKind::vgpr, divisor,
                                               useIEEE),
                           laneGroupCount(words), {words}, expected);
      expectArithResults(constantDivisorSpec(c, InputKind::uniform,
                                             RegisterKind::sgpr, divisor,
                                             false),
                         uniformCount, {uniformWords}, uniformExpected);
      expectArithResults(
          constantDivisorSpec(c, InputKind::uniform, RegisterKind::vgpr,
                              divisor, false),
          uniformCount, {uniformWords}, perLane(uniformExpected));
    }
}

// Each emitter writes into the input register and then moves it to dst.
TEST_F(DivDevice, ConstantDivisorDestinationAliasesDividend) {
  for (const DivisionCase &c : divisionCases)
    for (int32_t divisor : {3, 7, 641, INT32_MAX}) {
      SCOPED_TRACE(std::string(c.name) + " divisor " + std::to_string(divisor));
      std::vector<uint32_t> words = constantDividends(divisor);
      std::vector<uint32_t> uniformWords = sampleWords(words, 256);
      auto emit = [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                                 const std::vector<RegisterRef> &inputs) {
        (builder.*op)(inputs[0], inputs[0], divisor, false);
        builder.movb32(dst, inputs[0]);
      };
      expectArithResults({{InputKind::lane}, RegisterKind::vgpr, emit},
                         laneGroupCount(words), {words},
                         expectedByConstant(c, words, divisor));
      expectArithResults({{InputKind::uniform}, RegisterKind::sgpr, emit},
                         static_cast<int32_t>(uniformWords.size()),
                         {uniformWords},
                         expectedByConstant(c, uniformWords, divisor));
    }
}

// The dividend is an immediate, the input only sets the number of results.
TEST_F(DivDevice, ConstantDivisorImmediateDividend) {
  const int32_t dividends[] = {INT32_MIN, INT32_MIN + 1, -7,       -1, 0, 1, 5,
                               literalA,  0x12345678,    INT32_MAX};
  for (const DivisionCase &c : divisionCases)
    for (int32_t divisor : {3, 7, 641, INT32_MAX})
      for (int32_t dividend : dividends) {
        SCOPED_TRACE(std::string(c.name) + " " + std::to_string(dividend) +
                     " / " + std::to_string(divisor));
        uint32_t expected = c.host(static_cast<uint32_t>(dividend),
                                   static_cast<uint32_t>(divisor));
        auto emit = [=, op = c.op](IRBuilder &builder, RegisterRef dst,
                                   const std::vector<RegisterRef> &) {
          (builder.*op)(dst, dividend, divisor, false);
        };
        expectArithResults({{InputKind::lane}, RegisterKind::vgpr, emit}, 1,
                           {Words(arithWorkgroupSize, 0)},
                           Words(arithWorkgroupSize, expected));
        expectArithResults({{InputKind::uniform}, RegisterKind::sgpr, emit}, 2,
                           {Words(2, 0)}, Words(2, expected));
      }
}

TEST(DivAssembly, AllOperandPlacementsAssemble) {
  for (const DivisionCase &c : divisionCases)
    for (bool useIEEE : {true, false})
      expectAllPlacementsAssemble(std::string(c.name) + " " + pathName(useIEEE),
                                  [op = c.op, useIEEE](IRBuilder &builder,
                                                       RegisterRef dst,
                                                       Operand a, Operand b) {
                                    (builder.*op)(dst, a, b, useIEEE);
                                  });
}

// Emits the op for every dividend placement and several constant divisors into
// one kernel per destination kind, and again with dst as the dividend.
static void
expectConstantPlacementsAssemble(const DivisionCase &c, RegisterKind dstKind,
                                 const char *kernelName,
                                 std::span<const int32_t> divisors) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, kernelName, arithWorkgroupSize, KernelSignature());
  IRBuilder &builder = kernel.builder();
  RegisterRange vector = builder.allocateVgpr(1);
  RegisterRange scalar = builder.allocateSgpr(1);
  RegisterRange dst =
      dstKind == vgpr ? builder.allocateVgpr(1) : builder.allocateSgpr(1);
  std::vector<Operand> dividends = {scalar, 7, 0x12345, INT32_MIN};
  // A scalar destination cannot take a vector operand.
  if (dstKind == vgpr)
    dividends.push_back(vector);
  for (const Operand &a : dividends)
    for (int32_t b : divisors)
      (builder.*c.op)(dst.ref(), a, b, true);
  RegisterRef inPlace = dstKind == vgpr ? vector.ref() : scalar.ref();
  for (int32_t b : divisors)
    (builder.*c.op)(inPlace, inPlace, b, true);
  builder.createEndProgram();

  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(std::string(c.name) + "\n" + assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

TEST(DivAssembly, PowerOfTwoPlacementsAssemble) {
  const int32_t powers[] = {1, 2, 16, 1 << 20, 1 << 30};
  for (const DivisionCase &c : divisionCases) {
    expectConstantPlacementsAssemble(c, vgpr, "codegen_power_of_two", powers);
    expectConstantPlacementsAssemble(c, sgpr, "codegen_power_of_two", powers);
  }
}

TEST(DivAssembly, ConstantDivisorPlacementsAssemble) {
  for (const DivisionCase &c : divisionCases) {
    expectConstantPlacementsAssemble(c, vgpr, "codegen_constant_divisor",
                                     constantDivisors);
    expectConstantPlacementsAssemble(c, sgpr, "codegen_constant_divisor",
                                     constantDivisors);
  }
}

// Compare.

namespace {

struct PredicateCase {
  Predicate pred;
  std::string_view name;
  std::string_view scalar32;
  std::string_view vector32;
  std::string_view vector64;
};

class CmpTest : public BuilderTest {
protected:
  std::string emit32(Predicate pred, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { local.cmpi32(pred, dst, a, b); });
  }
  std::string emit64(Predicate pred, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { local.cmpi64(pred, dst, a, b); });
  }
};

// Registers: dst s0, padding s1, a s[2:3], b s[4:5], first temporary s6.
class CmpOrderedTest : public CmpTest {
protected:
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
};

// The compared operands, and the registers that keep them allocated.
struct Operands {
  std::vector<RegisterRange> owners;
  Operand a;
  Operand b;
};

} // namespace

static constexpr const char *signBitText = "-2147483648";

static constexpr PredicateCase predicateCases[] = {
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

// Index step that walks the edge values of the cross product in a pair vector.
static constexpr int32_t edgeStride32 = 17;
static constexpr int32_t edgeStride64 = 18;

// Immediates are int32 values; a 64-bit comparison sign-extends them.
static constexpr int32_t negativeLiteral = -0x12345678;
static constexpr int32_t inlineImmediate = -16;
static constexpr int32_t nearMin32 = INT32_MIN + 5;

using CmpDevice = DeviceTest;

using CompareFn = void (IRBuilder::*)(Predicate, RegisterRef, Operand, Operand);
template <typename Signed, typename Unsigned>
static bool compareAs(Predicate pred, uint64_t x, uint64_t y) {
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
static Words expectedBits(int32_t bits, Predicate pred, const Values &a,
                          const Values &b) {
  Words expected;
  for (int64_t i = 0; i < std::ssize(a); ++i) {
    bool holds = bits == 32 ? compareAs<int32_t, uint32_t>(pred, a[i], b[i])
                            : compareAs<int64_t, uint64_t>(pred, a[i], b[i]);
    expected.push_back(holds ? 1 : 0);
  }
  return expected;
}

// Every stride-th element, which gives one value per workgroup.
template <typename T>
static std::vector<T> everyNth(const std::vector<T> &values, int32_t stride,
                               int64_t count) {
  std::vector<T> picked;
  for (int64_t i = 0; i < count; ++i)
    picked.push_back(values[i * stride]);
  return picked;
}

using OperandMaker = std::function<Operands(
    IRBuilder &builder, const std::vector<RegisterRef> &inputs)>;

// 32-bit operands straight from inputs 0 and 1.
static OperandMaker words32() {
  return [](IRBuilder &, const std::vector<RegisterRef> &in) {
    return Operands{{}, in[0], in[1]};
  };
}

// Input 0 against an immediate, which comes first if immediateFirst.
static OperandMaker word32WithImmediate(int32_t immediate,
                                        bool immediateFirst) {
  return [=](IRBuilder &, const std::vector<RegisterRef> &in) {
    if (immediateFirst)
      return Operands{{}, immediate, in[0]};
    return Operands{{}, in[0], immediate};
  };
}

// 64-bit operands from inputs (0, 1) and (2, 3).
static OperandMaker pairs64(bool misalignedA = false,
                            bool misalignedB = false) {
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
static OperandMaker pair64WithImmediate(int32_t immediate, bool immediateFirst,
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
static void expectMaskResults(CompareFn compare, Predicate pred,
                              std::vector<InputKind> kinds,
                              const OperandMaker &make, const Inputs &inputs,
                              int32_t groups, const Words &expected) {
  ArithKernelSpec spec{std::move(kinds), vgpr,
                       [=](IRBuilder &builder, RegisterRef dst,
                           const std::vector<RegisterRef> &in) {
                         Operands operands = make(builder, in);
                         RegisterRange mask = builder.allocateSgpr(2, 2);
                         (builder.*compare)(pred, mask.ref(), operands.a,
                                            operands.b);
                         builder.selectb32(dst, mask.ref(), 1, 0);
                       }};
  expectArithResults(spec, groups, inputs, expected);
}

// Each workgroup stores the uniform bool the comparison wrote to one SGPR.
static void expectBoolResults(CompareFn compare, Predicate pred,
                              std::vector<InputKind> kinds,
                              const OperandMaker &make, const Inputs &inputs,
                              int32_t groups, const Words &expected) {
  ArithKernelSpec spec{std::move(kinds), sgpr,
                       [=](IRBuilder &builder, RegisterRef dst,
                           const std::vector<RegisterRef> &in) {
                         Operands operands = make(builder, in);
                         (builder.*compare)(pred, dst, operands.a, operands.b);
                       }};
  expectArithResults(spec, groups, inputs, expected);
}

TEST_F(CmpDevice, LaneOperandsI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectMaskResults(
        &IRBuilder::cmpi32, current.pred, {lane, lane}, words32(),
        {pairs.a, pairs.b}, laneGroupCount(pairs.a),
        expectedBits(32, current.pred, widen(pairs.a), widen(pairs.b)));
  }
}

TEST_F(CmpDevice, UniformOperandsToBoolI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectBoolResults(
        &IRBuilder::cmpi32, current.pred, {uniform, uniform}, words32(),
        {pairs.a, pairs.b}, static_cast<int32_t>(pairs.a.size()),
        expectedBits(32, current.pred, widen(pairs.a), widen(pairs.b)));
  }
}

TEST_F(CmpDevice, UniformOperandsToMaskI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectMaskResults(&IRBuilder::cmpi32, current.pred, {uniform, uniform},
                      words32(), {pairs.a, pairs.b},
                      static_cast<int32_t>(pairs.a.size()),
                      perLane(expectedBits(32, current.pred, widen(pairs.a),
                                           widen(pairs.b))));
  }
}

TEST_F(CmpDevice, MixedUniformAndLaneI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  int32_t groups = laneGroupCount(pairs.a);
  Words uniformWords = everyNth(pairs.a, edgeStride32, groups);
  Values uniformLanes = widen(perLane(uniformWords));
  for (const PredicateCase &current : predicateCases) {
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
  for (const PredicateCase &current : predicateCases) {
    for (int32_t immediate : {literal32, inlineImmediate}) {
      for (bool immediateFirst : {false, true}) {
        SCOPED_TRACE(std::string(current.name) + " immediate " +
                     std::to_string(immediate) +
                     (immediateFirst ? " first" : " second"));
        Values constant(pairs.a.size(), static_cast<uint32_t>(immediate));
        expectMaskResults(
            &IRBuilder::cmpi32, current.pred, {lane},
            word32WithImmediate(immediate, immediateFirst), {pairs.a},
            laneGroupCount(pairs.a),
            immediateFirst
                ? expectedBits(32, current.pred, constant, widen(pairs.a))
                : expectedBits(32, current.pred, widen(pairs.a), constant));
      }
    }
  }
}

TEST_F(CmpDevice, UniformAgainstImmediateI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Values constant(pairs.a.size(), static_cast<uint32_t>(literal32));
    expectBoolResults(&IRBuilder::cmpi32, current.pred, {uniform},
                      word32WithImmediate(literal32, false), {pairs.a},
                      static_cast<int32_t>(pairs.a.size()),
                      expectedBits(32, current.pred, widen(pairs.a), constant));
  }
}

TEST_F(CmpDevice, DestinationMayAliasAnOperandI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform},
                         sgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           builder.cmpi32(pred, in[0], in[0], in[1]);
                           builder.movb32(dst, in[0]);
                         }};
    expectArithResults(spec, static_cast<int32_t>(pairs.a.size()),
                       {pairs.a, pairs.b},
                       expectedBits(32, pred, widen(pairs.a), widen(pairs.b)));
  }
}

// The 64-bit tests pass each operand as a low and a high word.
static Inputs laneInputs(const WordQuads &q) {
  return {q.aLo, q.aHi, q.bLo, q.bHi};
}

TEST_F(CmpDevice, LaneOperandsI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Words expected = expectedBits(64, current.pred, a, b);
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {lane, lane, lane, lane}, pairs64(), laneInputs(quads),
                      laneGroupCount(quads.aLo), expected);
    SCOPED_TRACE("misaligned pairs");
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {lane, lane, lane, lane}, pairs64(true, true),
                      laneInputs(quads), laneGroupCount(quads.aLo), expected);
  }
}

TEST_F(CmpDevice, UniformOperandsToBoolI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  int32_t groups = static_cast<int32_t>(a.size());
  std::vector<InputKind> kinds{uniform, uniform, uniform, uniform};
  for (const PredicateCase &current : predicateCases) {
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
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {uniform, uniform, uniform, uniform}, pairs64(),
                      laneInputs(quads), static_cast<int32_t>(a.size()),
                      perLane(expectedBits(64, current.pred, a, b)));
  }
}

TEST_F(CmpDevice, MixedUniformAndLaneI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  int32_t groups = laneGroupCount(quads.aLo);
  Words uniformLo = everyNth(quads.aLo, edgeStride64, groups);
  Words uniformHi = everyNth(quads.aHi, edgeStride64, groups);
  Values uniformLanes = perLane(join(uniformLo, uniformHi));
  Values laneValues = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicateCases) {
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
  for (const PredicateCase &current : predicateCases) {
    for (int32_t immediate :
         {literal32, negativeLiteral, inlineImmediate, nearMin32}) {
      for (bool immediateFirst : {false, true}) {
        for (bool misaligned : {false, true}) {
          SCOPED_TRACE(std::string(current.name) + " immediate " +
                       std::to_string(immediate) +
                       (immediateFirst ? " first" : " second") +
                       (misaligned ? " misaligned" : " aligned"));
          Values constant(lanes.size(),
                          static_cast<uint64_t>(int64_t{immediate}));
          expectMaskResults(
              &IRBuilder::cmpi64, current.pred, {lane, lane},
              pair64WithImmediate(immediate, immediateFirst, misaligned),
              {quads.aLo, quads.aHi}, laneGroupCount(quads.aLo),
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
  for (const PredicateCase &current : predicateCases) {
    for (int32_t immediate :
         {nearMin32, literal32, negativeLiteral, inlineImmediate}) {
      for (bool immediateFirst : {false, true}) {
        SCOPED_TRACE(std::string(current.name) + " immediate " +
                     std::to_string(immediate) +
                     (immediateFirst ? " first" : " second"));
        Values constant(values.size(),
                        static_cast<uint64_t>(int64_t{immediate}));
        expectBoolResults(
            &IRBuilder::cmpi64, current.pred, {uniform, uniform},
            pair64WithImmediate(immediate, immediateFirst, false),
            {quads.aLo, quads.aHi}, static_cast<int32_t>(values.size()),
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
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform, uniform, uniform},
                         vgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           PairRegisters x = makePair(builder, in[0], in[1]);
                           PairRegisters y = makePair(builder, in[2], in[3]);
                           builder.cmpi64(pred, x.pair, x.pair, y.pair);
                           builder.selectb32(dst, x.pair, 1, 0);
                         }};
    expectArithResults(spec, static_cast<int32_t>(a.size()), laneInputs(quads),
                       perLane(expectedBits(64, pred, a, b)));
  }
}

TEST_F(CmpDevice, VectorDestinationMayAliasAnSgprOperandI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  int32_t groups = laneGroupCount(quads.aLo);
  Words uniformLo = everyNth(quads.aLo, edgeStride64, groups);
  Words uniformHi = everyNth(quads.aHi, edgeStride64, groups);
  Values uniformLanes = perLane(join(uniformLo, uniformHi));
  Values laneValues = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform, lane, lane},
                         vgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           PairRegisters x = makePair(builder, in[0], in[1]);
                           PairRegisters y = makePair(builder, in[2], in[3]);
                           builder.cmpi64(pred, x.pair, x.pair, y.pair);
                           builder.selectb32(dst, x.pair, 1, 0);
                         }};
    expectArithResults(spec, groups,
                       {uniformLo, uniformHi, quads.bLo, quads.bHi},
                       expectedBits(64, pred, uniformLanes, laneValues));
  }
}

// Select.

namespace {

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

  // The same with the implicit condition.
  std::string emitImplicitSelect(RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) {
      if (dst.count == 1)
        local.selectb32(dst, a, b);
      else
        local.selectb64(dst, a, b);
    });
  }
};

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

// Registers: cond s0, range s[2:6] holding the misaligned dst s[3:4].
class SelectSplitTest : public SelectTest {
protected:
  RegisterRange cond = builder.allocateSgpr(1);
  RegisterRange range = builder.allocateSgpr(5, 2);
  RegisterRef dst = range.ref(1, 2);
};

// Registers: mask s[0:1], dst v0, a v1, b v2.
class SelectVectorTest : public SelectTest {
protected:
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateVgpr(1);
  RegisterRange b = builder.allocateVgpr(1);
};

// How the destination sits relative to the operand registers.
enum class SelectPlacement {
  // The destination and the operands are separate registers.
  separate,
  // See emitOverlapping for the three overlapping layouts.
  highFirst,
  lowFirst,
  copied,
};

struct SelectCase {
  int32_t width;
  RegisterKind dstKind;
  InputKind condKind;
  InputKind operandKind;
  SelectPlacement placement = SelectPlacement::separate;
  bool dstMisaligned = false;
  bool operandsMisaligned = false;
  // An operand with no immediate is a register or a pair from the inputs.
  std::optional<int32_t> a;
  std::optional<int32_t> b;
  // The select has no cond: SCC or VCC is set from the condition input first.
  bool implicitCond = false;
};

struct SelectData {
  Words conds;
  Values a;
  Values b;
};

// The condition register: the uniform bool input itself, or a lane mask that
// is set where a lane input is nonzero. An implicit condition has no register,
// since it lives in SCC or VCC.
struct Condition {
  std::optional<RegisterRange> owner;
  std::optional<RegisterRef> ref;
};

} // namespace

// Builds "v_cndmask_b32 dst, src0, src1, mask" as rocisa prints it.
static std::string cndmask(const std::string &dst, const std::string &src0,
                           const std::string &src1, const std::string &mask) {
  return line("v_cndmask_b32", dst, src0, src1) + ", " + mask;
}

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

// Every instruction of an implicit-cond select on SGPRs is a move or a select,
// so none of them writes SCC.
static void expectOnlyMovesAndSelects(const std::string &out) {
  for (const std::string &mnemonic : mnemonicsOf(out))
    EXPECT_TRUE(mnemonic == "s_mov_b32" || mnemonic == "s_mov_b64" ||
                mnemonic == "s_cselect_b32" || mnemonic == "s_cselect_b64")
        << mnemonic;
}

TEST_F(SelectTest, ImplicitScalarWidth1) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  std::string out = emitImplicitSelect(dst.ref(), a, b);
  expectInstructions(out,
                     {inst("s_cselect_b32", {text(dst), text(a), text(b)})});
  // The same register in both operands, and dst aliasing one, are plain too.
  expectInstructions(emitImplicitSelect(a.ref(), a, b),
                     {inst("s_cselect_b32", {text(a), text(a), text(b)})});
  expectInstructions(emitImplicitSelect(dst.ref(), 5, -16),
                     {inst("s_cselect_b32", {text(dst), "5", "-16"})});
}

TEST_F(SelectTest, ImplicitScalarWidth1TwoDistinctLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  std::string out = emitImplicitSelect(dst.ref(), 0x12345, 0x54321);
  expectInstructions(out, {inst("s_mov_b32", {"s1", "344865"}),
                           inst("s_cselect_b32", {text(dst), "74565", "s1"})});
  expectOnlyMovesAndSelects(out);
  EXPECT_EQ(countLinesContaining(out, "s_cmp"), 0);
}

TEST_F(SelectScalar64Test, ImplicitAlignedPairs) {
  expectInstructions(emitImplicitSelect(dst.ref(), a, b),
                     {inst("s_cselect_b64", {text(dst), text(a), text(b)})});
  expectInstructions(emitImplicitSelect(dst.ref(), INT32_MAX, 64),
                     {inst("s_cselect_b64", {text(dst), "2147483647", "64"})});
}

TEST_F(SelectScalar64Test, ImplicitUnencodableOperandsAreCopiedWithMoves) {
  std::string out = emitImplicitSelect(dst.ref(), -17, b);
  expectInstructions(
      out, {inst("s_mov_b32", {"s8", "-17"}), inst("s_mov_b32", {"s9", "-1"}),
            inst("s_cselect_b64", {text(dst), "s[8:9]", text(b)})});
  expectOnlyMovesAndSelects(out);
  RegisterRange odd = builder.allocateSgpr(4, 2);
  out = emitImplicitSelect(dst.ref(), odd.ref(1, 2), b);
  expectInstructions(
      out, {inst("s_mov_b32", {"s13", "s10"}), inst("s_mov_b32", {"s12", "s9"}),
            inst("s_cselect_b64", {text(dst), "s[12:13]", text(b)})});
  expectOnlyMovesAndSelects(out);
  out = emitImplicitSelect(dst.ref(), 0x12345, 0x54321);
  expectInstructions(out,
                     {inst("s_mov_b64", {"s[12:13]", "344865"}),
                      inst("s_cselect_b64", {text(dst), "74565", "s[12:13]"})});
}

TEST_F(SelectSplitTest, ImplicitMisalignedPairIsOneSelectPerWord) {
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
  expectInstructions(emitImplicitSelect(dst, a, b),
                     {inst("s_cselect_b32", {"s3", "s8", "s10"}),
                      inst("s_cselect_b32", {"s4", "s9", "s11"})});
}

TEST_F(SelectSplitTest, ImplicitOverlapOrdersWordsLikeExplicit) {
  RegisterRange b = builder.allocateSgpr(2, 2);
  // a = s[2:3]: its high word is the destination's low register.
  expectInstructions(emitImplicitSelect(dst, range.ref(0, 2), b),
                     {inst("s_cselect_b32", {"s4", "s3", "s9"}),
                      inst("s_cselect_b32", {"s3", "s2", "s8"})});
  // Overlap in both directions copies one word, with a move.
  std::string out = emitImplicitSelect(dst, range.ref(0, 2), range.ref(2, 2));
  expectInstructions(out, {inst("s_mov_b32", {"s1", "s3"}),
                           inst("s_cselect_b32", {"s3", "s2", "s4"}),
                           inst("s_cselect_b32", {"s4", "s1", "s5"})});
  expectOnlyMovesAndSelects(out);
}

TEST_F(SelectVectorTest, ImplicitWidth1ReadsVcc) {
  std::string out = emitImplicitSelect(dst.ref(), a, b);
  expectInstructions(out, {cndmask(text(dst), text(b), text(a), "vcc")});
  EXPECT_EQ(countLinesContaining(out, "s_"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(SelectVectorTest, ImplicitWidth1CopiesSgprsAndLiterals) {
  RegisterRange s = builder.allocateSgpr(1);
  std::string out = emitImplicitSelect(dst.ref(), s, literal32);
  expectInstructions(out, {inst("v_mov_b32", {"v3", text(s)}),
                           inst("v_mov_b32", {"v4", literal32Text}),
                           cndmask(text(dst), "v4", "v3", "vcc")});
  out = emitImplicitSelect(dst.ref(), 5, -16);
  expectInstructions(out, {cndmask(text(dst), "-16", "5", "vcc")});
}

TEST_F(SelectVectorTest, ImplicitWidth2TakesHalvesInOrder) {
  RegisterRange pairDst = builder.allocateVgpr(2, 2);
  RegisterRange pairA = builder.allocateVgpr(2, 2);
  RegisterRange pairB = builder.allocateVgpr(2, 2);
  expectInstructions(
      emitImplicitSelect(pairDst.ref(), pairA, pairB),
      {cndmask("v4", "v8", "v6", "vcc"), cndmask("v5", "v9", "v7", "vcc")});
  // INT32_MIN is the literal 0x80000000 and the inline -1.
  expectInstructions(emitImplicitSelect(pairDst.ref(), INT32_MIN, pairB),
                     {inst("v_mov_b32", {"v3", "-2147483648"}),
                      cndmask("v4", "v8", "v3", "vcc"),
                      cndmask("v5", "v9", "-1", "vcc")});
}

TEST_F(SelectVectorTest, ImplicitWidth2OverlapOrdersWordsLikeExplicit) {
  RegisterRange range = builder.allocateVgpr(4, 2);
  RegisterRange pairB = builder.allocateVgpr(2, 2);
  RegisterRef dstPair = range.ref(1, 2);
  RegisterRef pairA = range.ref(0, 2);
  expectInstructions(emitImplicitSelect(dstPair, pairA, pairB),
                     {cndmask(text(dstPair.slice(1)), text(pairB.ref(1)),
                              text(pairA.slice(1)), "vcc"),
                      cndmask(text(dstPair.slice(0)), text(pairB.ref(0)),
                              text(pairA.slice(0)), "vcc")});
  RegisterRef pairC = range.ref(2, 2);
  // Overlap in both directions copies one word, with a move.
  expectInstructions(
      emitImplicitSelect(dstPair, pairA, pairC),
      {inst("v_mov_b32", {"v3", text(pairA.slice(1))}),
       cndmask(text(dstPair.slice(0)), text(pairC.slice(0)),
               text(pairA.slice(0)), "vcc"),
       cndmask(text(dstPair.slice(1)), text(pairC.slice(1)), "v3", "vcc")});
}

TEST_F(SelectVectorTest, ImplicitTemporariesAreReleased) {
  RegisterRange s = builder.allocateSgpr(1);
  emitImplicitSelect(dst.ref(), s, literal32);
  RegisterRange next = builder.allocateVgpr(1);
  EXPECT_EQ(next.start(), 3);
}

TEST_F(SelectVectorTest, ImplicitInvalidOperandsThrowWithoutEmitting) {
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange acc = builder.allocateAgpr(2, 2);
  RegisterRange pair = builder.allocateVgpr(2, 2);
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange sdst2 = builder.allocateSgpr(2, 2);
  std::string message;
  try {
    builder.selectb32(sdst.ref(), a, s);
  } catch (const std::invalid_argument &error) {
    message = error.what();
  }
  EXPECT_EQ(message, "selectb32 cannot write sgpr destination " + text(sdst) +
                         " from vgpr operand " + text(a));
  expectRejected([&] { builder.selectb32(acc.ref(0, 1), s, s); });
  expectRejected([&] { builder.selectb32(dst.ref(), acc.ref(0, 1), s); });
  expectRejected([&] { builder.selectb32(pair.ref(), s, s); });
  expectRejected([&] { builder.selectb32(dst.ref(), pair, b); });
  expectRejected([&] { builder.selectb64(dst.ref(), s, s); });
  expectRejected([&] { builder.selectb64(pair.ref(), a, b); });
  expectRejected([&] { builder.selectb64(sdst2.ref(), s, pair); });
  expectRejected(
      [&] { builder.selectb32(RegisterRef{RegisterKind::vgpr, 20, 3}, 5, 6); });
}

TEST_F(SelectVectorTest, ImplicitBoundaryImmediatesAreAccepted) {
  RegisterRange pair = builder.allocateVgpr(2, 2);
  EXPECT_NO_THROW(builder.selectb32(dst.ref(), -1, INT32_MIN));
  EXPECT_NO_THROW(builder.selectb64(pair.ref(), INT32_MIN, INT32_MAX));
}

// Immediates that cover inline constants, literals and negative values.
static constexpr int32_t selectImmediates32[] = {5, -16, 0x12345678,
                                                 -0x65432110, 64};

using SelectDevice = DeviceTest;

// Condition words with zero, one, other small values, the sign bit, all ones
// and pseudo-random words, so that "nonzero is true" is exercised.
static Words makeConds(int64_t count) {
  const Words edges{0, 1, 2, 0x80000000u, 0xFFFFFFFFu};
  Words conds;
  for (int64_t i = 0; i < count; ++i)
    conds.push_back(i % 7 < std::ssize(edges)
                        ? edges[i % 7]
                        : static_cast<uint32_t>(i) * 2654435761u);
  return conds;
}

static SelectData makeData32() {
  WordPairs pairs = makeArithPairs(randomPairs);
  return {makeConds(std::ssize(pairs.a)), widen(pairs.a), widen(pairs.b)};
}

static SelectData makeData64() {
  WordQuads quads = makeArithPairs64(randomPairs);
  return {makeConds(std::ssize(quads.aLo)), join(quads.aLo, quads.aHi),
          join(quads.bLo, quads.bHi)};
}

// Sets SCC to whether the uniform bool is nonzero, or VCC to the lanes where
// the lane input is nonzero. A uniform bool also sets VCC to all ones or zero,
// so that a vector select reads it. The nop covers the wait states of a VALU
// write of VCC.
static Condition makeImplicitCondition(IRBuilder &builder, InputKind kind,
                                       RegisterRef input,
                                       RegisterKind dstKind) {
  if (kind == lane) {
    builder.create<rocisa::VCmpNeU32>(std::make_shared<rocisa::VCC>(), 0,
                                      input.container());
    builder.create<rocisa::SNop>(1);
    return {};
  }
  builder.create<rocisa::SCmpLgU32>(input.container(), 0);
  if (dstKind == vgpr)
    builder.create<rocisa::SCSelectB64>(std::make_shared<rocisa::VCC>(), -1, 0);
  return {};
}

static Condition makeCondition(IRBuilder &builder, InputKind kind,
                               RegisterRef input) {
  if (kind == uniform)
    return {std::nullopt, input};
  RegisterRange mask = builder.allocateSgpr(2, 2);
  builder.cmpi32(Predicate::ne, mask.ref(), input, 0);
  RegisterRef ref = mask.ref();
  return {std::move(mask), ref};
}

// Reads an operand: an immediate, one input register, or a pair built from two
// inputs.
static Operand makeOperand(IRBuilder &builder, const SelectCase &spec,
                           std::optional<int32_t> immediate,
                           const std::vector<RegisterRef> &in, int64_t &next,
                           std::vector<RegisterRange> &owners) {
  if (immediate)
    return *immediate;
  if (spec.width == 1)
    return in[next++];
  PairRegisters pair =
      makePair(builder, in[next], in[next + 1], spec.operandsMisaligned);
  next += 2;
  owners.push_back(std::move(pair.range));
  return pair.pair;
}

static void emitSelect(IRBuilder &builder, int32_t width, RegisterRef dst,
                       std::optional<RegisterRef> cond, Operand a, Operand b) {
  if (width == 1 && cond)
    builder.selectb32(dst, *cond, a, b);
  else if (width == 1)
    builder.selectb32(dst, a, b);
  else if (cond)
    builder.selectb64(dst, *cond, a, b);
  else
    builder.selectb64(dst, a, b);
}

static PairRegisters emitSelectSeparate(IRBuilder &builder,
                                        const SelectCase &spec,
                                        std::optional<RegisterRef> cond,
                                        const std::vector<RegisterRef> &in) {
  int64_t next = 1;
  std::vector<RegisterRange> owners;
  Operand a = makeOperand(builder, spec, spec.a, in, next, owners);
  Operand b = makeOperand(builder, spec, spec.b, in, next, owners);
  int32_t count = spec.width == 1 ? 1 : spec.dstMisaligned ? 3 : 2;
  RegisterRange range = allocateKind(builder, spec.dstKind, count);
  RegisterRef dst = range.ref(count - spec.width, spec.width);
  emitSelect(builder, spec.width, dst, cond, a, b);
  return {std::move(range), dst};
}

// Places the destination in a range that also holds operand registers. In
// highFirst, a's high word is the destination's low register; in lowFirst, the
// destination's high register is a's low register; in copied, both overlap.
static PairRegisters emitSelectOverlapping(IRBuilder &builder,
                                           const SelectCase &spec,
                                           std::optional<RegisterRef> cond,
                                           const std::vector<RegisterRef> &in) {
  RegisterRange range = allocateKind(builder, spec.dstKind, 4);
  PairRegisters other = makePair(builder, in[3], in[4]);
  RegisterRef first = range.ref(0, 2);
  RegisterRef second = range.ref(1, 2);
  RegisterRef third = range.ref(2, 2);
  RegisterRef dst = second;
  RegisterRef a = first;
  RegisterRef b = other.pair;
  if (spec.placement == SelectPlacement::lowFirst) {
    dst = first;
    a = second;
  } else if (spec.placement == SelectPlacement::copied)
    b = third;
  builder.movb32(a.slice(0), in[1]);
  builder.movb32(a.slice(1), in[2]);
  if (spec.placement == SelectPlacement::copied) {
    builder.movb32(b.slice(0), in[3]);
    builder.movb32(b.slice(1), in[4]);
  }
  emitSelect(builder, 2, dst, cond, a, b);
  return {std::move(range), dst};
}

static PairRegisters emitSelectCase(IRBuilder &builder, const SelectCase &spec,
                                    const std::vector<RegisterRef> &in) {
  Condition cond =
      spec.implicitCond
          ? makeImplicitCondition(builder, spec.condKind, in[0], spec.dstKind)
          : makeCondition(builder, spec.condKind, in[0]);
  if (spec.placement == SelectPlacement::separate)
    return emitSelectSeparate(builder, spec, cond.ref, in);
  return emitSelectOverlapping(builder, spec, cond.ref, in);
}

// A result is one lane of a vgpr result or one workgroup of an sgpr result.
static int64_t selectInputIndex(const SelectCase &spec, InputKind kind,
                                int64_t result) {
  bool perResult = spec.dstKind == sgpr || kind == lane;
  return perResult ? result : result / arithWorkgroupSize;
}

static uint64_t selectOperandValue(const SelectCase &spec,
                                   std::optional<int32_t> immediate,
                                   const Values &values, int64_t result) {
  uint64_t value =
      immediate ? static_cast<uint64_t>(int64_t{*immediate})
                : values[selectInputIndex(spec, spec.operandKind, result)];
  return spec.width == 1 ? static_cast<uint32_t>(value) : value;
}

static Values selectExpectedValues(const SelectCase &spec,
                                   const SelectData &data, int64_t results) {
  Values expected;
  for (int64_t i = 0; i < results; ++i) {
    bool holds = data.conds[selectInputIndex(spec, spec.condKind, i)] != 0;
    expected.push_back(selectOperandValue(spec, holds ? spec.a : spec.b,
                                          holds ? data.a : data.b, i));
  }
  return expected;
}

static void addOperandInputs(KernelInputs &inputs, const SelectCase &spec,
                             std::optional<int32_t> immediate,
                             const Values &values, int64_t laneCount,
                             int64_t groupCount) {
  if (immediate)
    return;
  addInput(inputs, spec.operandKind, lowWords(values), laneCount, groupCount);
  if (spec.width == 2)
    addInput(inputs, spec.operandKind, highWords(values), laneCount,
             groupCount);
}

static KernelInputs makeSelectInputs(const SelectCase &spec,
                                     const SelectData &data,
                                     int64_t groupCount) {
  int64_t laneCount = std::ssize(data.conds);
  KernelInputs inputs;
  addInput(inputs, spec.condKind, data.conds, laneCount, groupCount);
  addOperandInputs(inputs, spec, spec.a, data.a, laneCount, groupCount);
  addOperandInputs(inputs, spec, spec.b, data.b, laneCount, groupCount);
  return inputs;
}

// Runs the select and checks it, a 64-bit result through two runs that each
// store one half.
static void expectSelectResults(const SelectCase &spec,
                                const SelectData &data) {
  int64_t results = std::ssize(data.conds);
  int64_t groups =
      spec.dstKind == sgpr ? results : results / arithWorkgroupSize;
  int64_t expectedCount = spec.dstKind == sgpr ? groups : results;
  KernelInputs inputs = makeSelectInputs(spec, data, groups);
  Values expected = selectExpectedValues(spec, data, expectedCount);
  for (int32_t half = 0; half < spec.width; ++half) {
    SCOPED_TRACE(spec.width == 1 ? "word"
                 : half == 0     ? "low half"
                                 : "high half");
    ArithKernelSpec kernel{inputs.kinds, spec.dstKind,
                           [spec, half](IRBuilder &builder, RegisterRef dst,
                                        const std::vector<RegisterRef> &in) {
                             PairRegisters value =
                                 emitSelectCase(builder, spec, in);
                             builder.movb32(dst, value.pair.slice(half));
                           }};
    Words words = half == 0 ? lowWords(expected) : highWords(expected);
    expectArithResults(kernel, static_cast<int32_t>(groups), inputs.words,
                       words);
  }
}

static SelectCase case32(RegisterKind dstKind, InputKind condKind,
                         InputKind operandKind) {
  return {1, dstKind, condKind, operandKind};
}

static SelectCase case64(RegisterKind dstKind, InputKind condKind,
                         InputKind operandKind) {
  return {2, dstKind, condKind, operandKind};
}

TEST_F(SelectDevice, ScalarWords) {
  expectSelectResults(case32(sgpr, uniform, uniform), makeData32());
}

TEST_F(SelectDevice, VectorWordsWithUniformBoolCondition) {
  expectSelectResults(case32(vgpr, uniform, lane), makeData32());
}

TEST_F(SelectDevice, VectorWordsWithMaskCondition) {
  SelectData data = makeData32();
  expectSelectResults(case32(vgpr, lane, lane), data);
  expectSelectResults(case32(vgpr, lane, uniform), data);
  expectSelectResults(case32(vgpr, uniform, uniform), data);
}

TEST_F(SelectDevice, WordImmediates) {
  SelectData data = makeData32();
  for (int32_t a : selectImmediates32) {
    for (int32_t b : selectImmediates32) {
      SCOPED_TRACE("a " + std::to_string(a) + ", b " + std::to_string(b));
      for (SelectCase spec :
           {case32(sgpr, uniform, uniform), case32(vgpr, uniform, uniform),
            case32(vgpr, lane, lane)}) {
        spec.a = a;
        spec.b = b;
        expectSelectResults(spec, data);
      }
    }
  }
}

TEST_F(SelectDevice, WordImmediateAgainstRegister) {
  SelectData data = makeData32();
  for (int32_t immediate : selectImmediates32) {
    SCOPED_TRACE(immediate);
    for (SelectCase spec :
         {case32(sgpr, uniform, uniform), case32(vgpr, lane, lane),
          case32(vgpr, lane, uniform)}) {
      SelectCase immediateFirst = spec;
      immediateFirst.a = immediate;
      expectSelectResults(immediateFirst, data);
      spec.b = immediate;
      expectSelectResults(spec, data);
    }
  }
}

TEST_F(SelectDevice, ScalarPairs) {
  SelectData data = makeData64();
  for (bool dstMisaligned : {false, true}) {
    for (bool operandsMisaligned : {false, true}) {
      SCOPED_TRACE(
          std::string(dstMisaligned ? "misaligned dst, " : "aligned dst, ") +
          (operandsMisaligned ? "misaligned operands" : "aligned operands"));
      SelectCase spec = case64(sgpr, uniform, uniform);
      spec.dstMisaligned = dstMisaligned;
      spec.operandsMisaligned = operandsMisaligned;
      expectSelectResults(spec, data);
    }
  }
}

TEST_F(SelectDevice, ScalarPairImmediates) {
  SelectData data = makeData64();
  const int32_t count = static_cast<int32_t>(std::size(immediates64));
  for (bool dstMisaligned : {false, true}) {
    for (int32_t i = 0; i < count; ++i) {
      int32_t immediate = immediates64[i];
      SCOPED_TRACE("immediate " + std::to_string(immediate) +
                   (dstMisaligned ? " to misaligned dst" : " to aligned dst"));
      SelectCase spec = case64(sgpr, uniform, uniform);
      spec.dstMisaligned = dstMisaligned;
      SelectCase immediateFirst = spec;
      immediateFirst.a = immediate;
      expectSelectResults(immediateFirst, data);
      spec.b = immediate;
      expectSelectResults(spec, data);
      SelectCase both = case64(sgpr, uniform, uniform);
      both.dstMisaligned = dstMisaligned;
      both.a = immediate;
      both.b = immediates64[(i + 1) % count];
      expectSelectResults(both, data);
    }
  }
}

TEST_F(SelectDevice, VectorPairs) {
  SelectData data = makeData64();
  for (InputKind condKind : {lane, uniform}) {
    for (bool dstMisaligned : {false, true}) {
      SCOPED_TRACE(std::string(condKind == lane ? "mask cond, "
                                                : "uniform bool cond, ") +
                   (dstMisaligned ? "misaligned dst" : "aligned dst"));
      SelectCase spec = case64(vgpr, condKind, lane);
      spec.dstMisaligned = dstMisaligned;
      expectSelectResults(spec, data);
      spec.operandsMisaligned = true;
      expectSelectResults(spec, data);
    }
  }
  expectSelectResults(case64(vgpr, lane, uniform), data);
  expectSelectResults(case64(vgpr, uniform, uniform), data);
}

TEST_F(SelectDevice, VectorPairImmediates) {
  SelectData data = makeData64();
  const int32_t count = static_cast<int32_t>(std::size(immediates64));
  for (InputKind condKind : {lane, uniform}) {
    for (bool dstMisaligned : {false, true}) {
      for (int32_t i = 0; i < count; ++i) {
        int32_t immediate = immediates64[i];
        SCOPED_TRACE(
            "immediate " + std::to_string(immediate) +
            (condKind == lane ? ", mask cond" : ", uniform bool cond") +
            (dstMisaligned ? ", misaligned dst" : ", aligned dst"));
        SelectCase spec = case64(vgpr, condKind, lane);
        spec.dstMisaligned = dstMisaligned;
        SelectCase immediateFirst = spec;
        immediateFirst.a = immediate;
        expectSelectResults(immediateFirst, data);
        spec.b = immediate;
        expectSelectResults(spec, data);
        SelectCase both = case64(vgpr, condKind, lane);
        both.dstMisaligned = dstMisaligned;
        both.a = immediate;
        both.b = immediates64[(i + 1) % count];
        expectSelectResults(both, data);
      }
    }
  }
}

TEST_F(SelectDevice, OverlappingDestinationOrdering) {
  SelectData data = makeData64();
  for (RegisterKind dstKind : {sgpr, vgpr}) {
    InputKind kind = dstKind == sgpr ? uniform : lane;
    for (SelectPlacement placement :
         {SelectPlacement::highFirst, SelectPlacement::lowFirst,
          SelectPlacement::copied}) {
      SCOPED_TRACE(std::string(dstKind == sgpr ? "sgpr" : "vgpr") +
                   " placement " +
                   std::to_string(static_cast<int32_t>(placement)));
      SelectCase spec = case64(dstKind, kind, kind);
      spec.placement = placement;
      expectSelectResults(spec, data);
    }
  }
}

// The implicit-cond forms run the same cases as the explicit ones, after
// setting SCC or VCC from the condition input. The legalization copies and the
// pair moves between that and the select must leave them intact.
static SelectCase implicitCase(SelectCase spec) {
  spec.implicitCond = true;
  return spec;
}

TEST_F(SelectDevice, ImplicitScalarWords) {
  expectSelectResults(implicitCase(case32(sgpr, uniform, uniform)),
                      makeData32());
}

TEST_F(SelectDevice, ImplicitVectorWords) {
  SelectData data = makeData32();
  // A lane condition sets VCC from a compare, a uniform bool from SCC.
  expectSelectResults(implicitCase(case32(vgpr, lane, lane)), data);
  expectSelectResults(implicitCase(case32(vgpr, lane, uniform)), data);
  expectSelectResults(implicitCase(case32(vgpr, uniform, lane)), data);
  expectSelectResults(implicitCase(case32(vgpr, uniform, uniform)), data);
}

TEST_F(SelectDevice, ImplicitWordImmediates) {
  SelectData data = makeData32();
  for (int32_t a : selectImmediates32) {
    for (int32_t b : selectImmediates32) {
      SCOPED_TRACE("a " + std::to_string(a) + ", b " + std::to_string(b));
      for (SelectCase spec :
           {case32(sgpr, uniform, uniform), case32(vgpr, uniform, uniform),
            case32(vgpr, lane, lane)}) {
        spec.a = a;
        spec.b = b;
        expectSelectResults(implicitCase(spec), data);
      }
    }
  }
}

TEST_F(SelectDevice, ImplicitWordImmediateAgainstRegister) {
  SelectData data = makeData32();
  for (int32_t immediate : selectImmediates32) {
    SCOPED_TRACE(immediate);
    for (SelectCase spec :
         {case32(sgpr, uniform, uniform), case32(vgpr, lane, lane),
          case32(vgpr, lane, uniform)}) {
      SelectCase immediateFirst = spec;
      immediateFirst.a = immediate;
      expectSelectResults(implicitCase(immediateFirst), data);
      spec.b = immediate;
      expectSelectResults(implicitCase(spec), data);
    }
  }
}

TEST_F(SelectDevice, ImplicitScalarPairs) {
  SelectData data = makeData64();
  for (bool dstMisaligned : {false, true})
    for (bool operandsMisaligned : {false, true}) {
      SCOPED_TRACE(
          std::string(dstMisaligned ? "misaligned dst, " : "aligned dst, ") +
          (operandsMisaligned ? "misaligned operands" : "aligned operands"));
      SelectCase spec = case64(sgpr, uniform, uniform);
      spec.dstMisaligned = dstMisaligned;
      spec.operandsMisaligned = operandsMisaligned;
      expectSelectResults(implicitCase(spec), data);
    }
}

TEST_F(SelectDevice, ImplicitScalarPairImmediates) {
  SelectData data = makeData64();
  const int32_t count = static_cast<int32_t>(std::size(immediates64));
  for (bool dstMisaligned : {false, true})
    for (int32_t i = 0; i < count; ++i) {
      int32_t immediate = immediates64[i];
      SCOPED_TRACE("immediate " + std::to_string(immediate) +
                   (dstMisaligned ? " to misaligned dst" : " to aligned dst"));
      SelectCase spec = case64(sgpr, uniform, uniform);
      spec.dstMisaligned = dstMisaligned;
      SelectCase immediateFirst = spec;
      immediateFirst.a = immediate;
      expectSelectResults(implicitCase(immediateFirst), data);
      spec.b = immediate;
      expectSelectResults(implicitCase(spec), data);
      SelectCase both = case64(sgpr, uniform, uniform);
      both.dstMisaligned = dstMisaligned;
      both.a = immediate;
      both.b = immediates64[(i + 1) % count];
      expectSelectResults(implicitCase(both), data);
    }
}

TEST_F(SelectDevice, ImplicitVectorPairs) {
  SelectData data = makeData64();
  for (InputKind condKind : {lane, uniform})
    for (bool dstMisaligned : {false, true}) {
      SCOPED_TRACE(std::string(condKind == lane ? "VCC from a compare, "
                                                : "VCC from SCC, ") +
                   (dstMisaligned ? "misaligned dst" : "aligned dst"));
      SelectCase spec = case64(vgpr, condKind, lane);
      spec.dstMisaligned = dstMisaligned;
      expectSelectResults(implicitCase(spec), data);
      spec.operandsMisaligned = true;
      expectSelectResults(implicitCase(spec), data);
    }
  expectSelectResults(implicitCase(case64(vgpr, lane, uniform)), data);
  expectSelectResults(implicitCase(case64(vgpr, uniform, uniform)), data);
}

TEST_F(SelectDevice, ImplicitVectorPairImmediates) {
  SelectData data = makeData64();
  const int32_t count = static_cast<int32_t>(std::size(immediates64));
  for (InputKind condKind : {lane, uniform})
    for (int32_t i = 0; i < count; ++i) {
      int32_t immediate = immediates64[i];
      SCOPED_TRACE(
          "immediate " + std::to_string(immediate) +
          (condKind == lane ? ", VCC from a compare" : ", VCC from SCC"));
      SelectCase spec = case64(vgpr, condKind, lane);
      SelectCase immediateFirst = spec;
      immediateFirst.a = immediate;
      expectSelectResults(implicitCase(immediateFirst), data);
      spec.b = immediate;
      expectSelectResults(implicitCase(spec), data);
      SelectCase both = case64(vgpr, condKind, lane);
      both.a = immediate;
      both.b = immediates64[(i + 1) % count];
      expectSelectResults(implicitCase(both), data);
    }
}

TEST_F(SelectDevice, ImplicitOverlappingDestinationOrdering) {
  SelectData data = makeData64();
  for (RegisterKind dstKind : {sgpr, vgpr}) {
    InputKind kind = dstKind == sgpr ? uniform : lane;
    for (SelectPlacement placement :
         {SelectPlacement::highFirst, SelectPlacement::lowFirst,
          SelectPlacement::copied}) {
      SCOPED_TRACE(std::string(dstKind == sgpr ? "sgpr" : "vgpr") +
                   " placement " +
                   std::to_string(static_cast<int32_t>(placement)));
      SelectCase spec = case64(dstKind, kind, kind);
      spec.placement = placement;
      expectSelectResults(implicitCase(spec), data);
    }
  }
}

// Buffer resource.

namespace {

class RsrcTest : public BuilderTest {
protected:
  std::string emitRsrc(RegisterRef dst, Operand pointer, Operand size) {
    return emitted(
        [&](IRBuilder &local) { local.makeRsrc(dst, pointer, size); });
  }
  int32_t movCount(const std::string &out) {
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

} // namespace

// The upper word 0x00020000 as rocisa prints a decimal immediate.
static const std::string upperWord = "131072";

// Literal placements that the allocator never hands out in these tests.
static const RegisterRef dstRef{sgpr, 40, 4};
static const RegisterRef pointerRef{sgpr, 20, 2};
static const RegisterRef sizeRef{sgpr, 30, 1};

static RegisterRef word(int32_t index) {
  return RegisterRef{sgpr, dstRef.start + index, 1};
}

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
    int32_t value;
    std::string printed;
  };
  for (const Case &current : {Case{0, "0"}, Case{308, "308"}, Case{-1, "-1"}}) {
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
  int32_t expected = probe.ref().start;
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
  EXPECT_EQ(body.toString(), "");
}

TEST_F(RsrcTest, FailureWhereTemporaryWouldBeNeededAllocatesAndEmitsNothing) {
  RegisterRange probe = builder.allocateSgpr(1);
  int32_t expected = probe.ref().start;
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
  EXPECT_NE(errorOf(dstRef, pointerRef, RegisterRef{vgpr, 3, 1})
                .find("makeRsrc size must be a single sgpr"),
            std::string::npos);
}

// Memory.

namespace {

struct WidthSpec {
  MemoryWidth width;
  const char *name;
  // The bytes accessed, the bytes a lane owns and the registers used.
  int32_t bytes;
  int32_t slot;
  int32_t registers;
  // The mnemonics rocisa prints on gfx950.
  const char *bufferLoad;
  const char *bufferStore;
  const char *dsLoad;
  const char *dsStore;
};

enum class MemoryOp { loadDs, storeDs, loadBuffer, storeBuffer };

// The arguments of one memory op; address is the DS address or the buffer
// voffset. The defaults are legal.
struct MemoryArgs {
  RegisterRef payload{vgpr, 2, 1};
  RegisterRef address{vgpr, 0, 1};
  RegisterRef rsrc{sgpr, 4, 4};
  Operand soffset = 0;
  MemoryWidth width = MemoryWidth::b32;
  int32_t offset = 0;
  BufferAccess access;
};

class MemoryTest : public BuilderTest {
protected:
  static void issue(IRBuilder &target, MemoryOp op, const MemoryArgs &a) {
    switch (op) {
    case MemoryOp::loadDs:
      return target.loadDs(a.payload, a.address, a.width, a.offset);
    case MemoryOp::storeDs:
      return target.storeDs(a.address, a.payload, a.width, a.offset);
    case MemoryOp::loadBuffer:
      return target.loadBuffer(a.payload, a.rsrc, a.address, a.soffset, a.width,
                               a.access);
    case MemoryOp::storeBuffer:
      return target.storeBuffer(a.payload, a.rsrc, a.address, a.soffset,
                                a.width, a.access);
    }
  }
  std::string emitOne(MemoryOp op, const MemoryArgs &args) {
    return emitted([&](IRBuilder &local) { issue(local, op, args); });
  }
  // Asserts that op throws with args and emits nothing.
  void expectOpRejected(MemoryOp op, const MemoryArgs &args) {
    expectRejected([&] { issue(builder, op, args); });
  }
  void expectAllRejected(std::span<const MemoryOp> ops,
                         const MemoryArgs &args) {
    for (MemoryOp op : ops)
      expectOpRejected(op, args);
  }
  std::string errorOf(MemoryOp op, const MemoryArgs &args) {
    try {
      issue(builder, op, args);
    } catch (const std::invalid_argument &error) {
      return error.what();
    }
    return {};
  }
};

enum class SizeSource { sgpr, immediate };

// Byte offsets of the arguments in the kernarg segment.
struct MemoryOffsets {
  int32_t in;
  int32_t out;
  int32_t first;
  int32_t slots;
  int32_t inBytes;
  int32_t outBytes;
};

struct MemoryCase {
  MemoryWidth width = MemoryWidth::b32;
  // Where the in and out descriptor sizes come from.
  SizeSource source = SizeSource::sgpr;
  // The descriptor sizes in slots. The defaults are shorter than the lane
  // count, so lanes run past each descriptor's end.
  int32_t inSlots = 77;
  int32_t outSlots = 101;
  bool soffsetInSgpr = false;
  int32_t soffset = 0;
  BufferAccess access;
  bool useAgpr = false;
  bool throughLds = false;
  int32_t dsOffset = 0;
};

// The registers and descriptors the kernel body uses.
struct MemoryRegisters {
  RegisterRange in, out, first, slots;
  RegisterRange local, voffset, firstOffset;
  RegisterRange payload;
};

struct MemoryResults {
  std::vector<uint8_t> out, first, slots;
};

} // namespace

static constexpr std::array<MemoryOp, 4> allOps = {
    MemoryOp::loadDs, MemoryOp::storeDs, MemoryOp::loadBuffer,
    MemoryOp::storeBuffer};
static constexpr std::array<MemoryOp, 2> dsOps = {MemoryOp::loadDs,
                                                  MemoryOp::storeDs};
static constexpr std::array<MemoryOp, 2> bufferOps = {MemoryOp::loadBuffer,
                                                      MemoryOp::storeBuffer};

static const std::array<WidthSpec, 6> widthSpecs = {{
    {MemoryWidth::b8, "b8", 1, 4, 1, "buffer_load_ubyte", "buffer_store_byte",
     "ds_read_u8", "ds_write_b8"},
    {MemoryWidth::b16, "b16", 2, 4, 1, "buffer_load_ushort",
     "buffer_store_short", "ds_read_u16", "ds_write_b16"},
    {MemoryWidth::b32, "b32", 4, 4, 1, "buffer_load_dword",
     "buffer_store_dword", "ds_read_b32", "ds_write_b32"},
    {MemoryWidth::b64, "b64", 8, 8, 2, "buffer_load_dwordx2",
     "buffer_store_dwordx2", "ds_read_b64", "ds_write_b64"},
    {MemoryWidth::b96, "b96", 12, 16, 3, "buffer_load_dwordx3",
     "buffer_store_dwordx3", "ds_read_b96", "ds_write_b96"},
    {MemoryWidth::b128, "b128", 16, 16, 4, "buffer_load_dwordx4",
     "buffer_store_dwordx4", "ds_read_b128", "ds_write_b128"},
}};

static const WidthSpec &specOf(MemoryWidth width) {
  return widthSpecs[static_cast<size_t>(width)];
}

static constexpr int32_t memoryWorkgroupSize = 64;
static constexpr int32_t memoryGroupCount = 2;
static constexpr int32_t memoryLaneCount =
    memoryWorkgroupSize * memoryGroupCount;
// The instruction each op prints for default arguments and the given payload.
static std::string defaultLine(MemoryOp op, const WidthSpec &spec,
                               const std::string &payload) {
  switch (op) {
  case MemoryOp::loadDs:
    return line(spec.dsLoad, payload, "v0") + " offset:0\n";
  case MemoryOp::storeDs:
    return std::string(spec.dsStore) + " v0, " + payload + " offset:0\n";
  case MemoryOp::loadBuffer:
    return line(spec.bufferLoad, payload, "v0", "s[4:7]") +
           ", 0 offen offset:0\n";
  case MemoryOp::storeBuffer:
    return line(spec.bufferStore, payload, "v0", "s[4:7]") +
           ", 0 offen offset:0\n";
  }
}

TEST_F(MemoryTest, EveryWidthPrintsItsMnemonic) {
  for (RegisterKind kind : {vgpr, agpr}) {
    for (const WidthSpec &spec : widthSpecs) {
      SCOPED_TRACE(std::string(spec.name) + (kind == vgpr ? " vgpr" : " agpr"));
      MemoryArgs args;
      args.payload = RegisterRef{kind, 2, spec.registers};
      args.width = spec.width;
      for (MemoryOp op : allOps)
        EXPECT_EQ(emitOne(op, args), defaultLine(op, spec, text(args.payload)));
    }
  }
}

TEST_F(MemoryTest, SingleRegistersMayStartAtAnyIndex) {
  MemoryArgs args;
  args.payload = {vgpr, 1, 1};
  EXPECT_EQ(emitOne(MemoryOp::loadDs, args), "ds_read_b32 v1, v0 offset:0\n");
  args.payload = {agpr, 3, 1};
  EXPECT_EQ(emitOne(MemoryOp::storeBuffer, args),
            "buffer_store_dword acc3, v0, s[4:7], 0 offen offset:0\n");
}

TEST_F(MemoryTest, DsOffsetsPrint) {
  MemoryArgs args;
  for (int32_t offset : {1, 4096, 65535}) {
    args.offset = offset;
    std::string suffix = " offset:" + std::to_string(offset) + "\n";
    EXPECT_EQ(emitOne(MemoryOp::loadDs, args), "ds_read_b32 v2, v0" + suffix);
    EXPECT_EQ(emitOne(MemoryOp::storeDs, args), "ds_write_b32 v0, v2" + suffix);
  }
}

TEST_F(MemoryTest, BufferOptionsPrint) {
  struct Case {
    Operand soffset;
    BufferAccess access;
    std::string suffix;
  };
  const Case cases[] = {
      {RegisterRef{sgpr, 9, 1}, {}, "s9 offen offset:0"},
      {0, {}, "0 offen offset:0"},
      {64, {}, "64 offen offset:0"},
      {0, {.offset = 4095}, "0 offen offset:4095"},
      {0, {.sc0 = true}, "0 offen offset:0 sc0"},
      {0, {.sc1 = true}, "0 offen offset:0 sc1"},
      {0, {.nt = true}, "0 offen offset:0 nt"},
      {RegisterRef{sgpr, 9, 1},
       {.offset = 4095, .sc0 = true, .sc1 = true, .nt = true},
       "s9 offen offset:4095 sc0 sc1 nt"},
  };
  for (const Case &current : cases) {
    SCOPED_TRACE(current.suffix);
    MemoryArgs args;
    args.soffset = current.soffset;
    args.access = current.access;
    EXPECT_EQ(emitOne(MemoryOp::loadBuffer, args),
              "buffer_load_dword v2, v0, s[4:7], " + current.suffix + "\n");
    EXPECT_EQ(emitOne(MemoryOp::storeBuffer, args),
              "buffer_store_dword v2, v0, s[4:7], " + current.suffix + "\n");
  }
}

TEST_F(MemoryTest, RejectsPayloadsOfTheWrongSize) {
  for (RegisterKind kind : {vgpr, agpr}) {
    for (const WidthSpec &spec : widthSpecs) {
      for (int32_t count : {spec.registers - 1, spec.registers + 1}) {
        if (count < 1)
          continue;
        SCOPED_TRACE(std::string(spec.name) + " count " +
                     std::to_string(count));
        MemoryArgs args;
        args.payload = {kind, 2, count};
        args.width = spec.width;
        expectAllRejected(allOps, args);
      }
    }
  }
}

TEST_F(MemoryTest, RejectsScalarPayloads) {
  for (const WidthSpec &spec : widthSpecs) {
    SCOPED_TRACE(spec.name);
    MemoryArgs args;
    args.payload = {sgpr, 8, spec.registers};
    args.width = spec.width;
    expectAllRejected(allOps, args);
  }
}

TEST_F(MemoryTest, RejectsMisalignedTuples) {
  for (RegisterKind kind : {vgpr, agpr}) {
    for (const WidthSpec &spec : widthSpecs) {
      if (spec.registers < 2)
        continue;
      for (int32_t start : {1, 3}) {
        SCOPED_TRACE(std::string(spec.name) + " start " +
                     std::to_string(start));
        MemoryArgs args;
        args.payload = {kind, start, spec.registers};
        args.width = spec.width;
        expectAllRejected(allOps, args);
      }
    }
  }
}

TEST_F(MemoryTest, RejectsBadAddresses) {
  for (RegisterRef address : {RegisterRef{sgpr, 0, 1}, RegisterRef{agpr, 0, 1},
                              RegisterRef{vgpr, 0, 2}}) {
    SCOPED_TRACE(text(address));
    MemoryArgs args;
    args.address = address;
    expectAllRejected(allOps, args);
  }
}

TEST_F(MemoryTest, RejectsOffsetsOutOfRange) {
  MemoryArgs args;
  for (int32_t offset : {-1, 65536}) {
    args.offset = offset;
    expectAllRejected(dsOps, args);
  }
  args = {};
  for (int32_t offset : {-1, 4096}) {
    args.access.offset = offset;
    expectAllRejected(bufferOps, args);
  }
}

TEST_F(MemoryTest, RejectsBadResources) {
  for (RegisterRef rsrc : {RegisterRef{vgpr, 4, 4}, RegisterRef{agpr, 4, 4},
                           RegisterRef{sgpr, 4, 3}, RegisterRef{sgpr, 4, 8},
                           RegisterRef{sgpr, 2, 4}}) {
    SCOPED_TRACE(text(rsrc));
    MemoryArgs args;
    args.rsrc = rsrc;
    expectAllRejected(bufferOps, args);
  }
}

TEST_F(MemoryTest, RejectsBadSoffsets) {
  for (Operand soffset :
       {Operand(RegisterRef{vgpr, 6, 1}), Operand(RegisterRef{agpr, 6, 1}),
        Operand(RegisterRef{sgpr, 8, 2}), Operand(65), Operand(-1),
        Operand(-16), Operand(-17)}) {
    MemoryArgs args;
    args.soffset = soffset;
    expectAllRejected(bufferOps, args);
  }
}

TEST_F(MemoryTest, ErrorMessagesNameTheOperationAndTheProblem) {
  struct Case {
    MemoryOp op;
    MemoryArgs args;
    std::string name;
    std::string problem;
  };
  std::vector<Case> cases(8);
  cases[0] = {MemoryOp::loadDs, {}, "loadDs", "needs 2 vgprs or agprs"};
  cases[0].args.width = MemoryWidth::b64;
  cases[1] = {MemoryOp::storeDs, {}, "storeDs", "vgprs or agprs"};
  cases[1].args.payload = {sgpr, 8, 1};
  cases[2] = {MemoryOp::loadDs, {}, "loadDs", "must start at an even index"};
  cases[2].args.payload = {vgpr, 3, 2};
  cases[2].args.width = MemoryWidth::b64;
  cases[3] = {
      MemoryOp::storeDs, {}, "storeDs", "address must be a single vgpr"};
  cases[3].args.address = {sgpr, 0, 1};
  cases[4] = {
      MemoryOp::loadDs, {}, "loadDs", "offset 65536 is outside 0 to 65535"};
  cases[4].args.offset = 65536;
  cases[5] = {MemoryOp::loadBuffer, {}, "loadBuffer", "multiple of 4"};
  cases[5].args.rsrc = {sgpr, 6, 4};
  cases[6] = {
      MemoryOp::storeBuffer,
      {},
      "storeBuffer",
      "soffset must be a single sgpr or an immediate from 0 to 64, got 65"};
  cases[6].args.soffset = 65;
  cases[7] = {MemoryOp::storeBuffer,
              {},
              "storeBuffer",
              "offset 4096 is outside 0 to 4095"};
  cases[7].args.access.offset = 4096;
  for (const Case &current : cases) {
    std::string message = errorOf(current.op, current.args);
    SCOPED_TRACE(message);
    EXPECT_EQ(message.rfind(current.name, 0), 0u);
    EXPECT_NE(message.find(current.problem), std::string::npos);
    EXPECT_NE(message.back(), '.');
  }
}

static void ldsLoad(IRBuilder &target, RegisterRef base, int32_t ldsOffset,
                    MemoryWidth width, BufferAccess access = {}) {
  RegisterRef rsrc{sgpr, 4, 4};
  RegisterRef voffset{vgpr, 0, 1};
  target.loadBufferToLds(rsrc, voffset, 0, base, ldsOffset, width, access);
}

TEST_F(MemoryTest, BufferToLdsPrintsTheM0WriteAndTheLoad) {
  RegisterRef base{sgpr, 9, 1};
  EXPECT_EQ(emitted([&](IRBuilder &local) {
              ldsLoad(local, base, 0, MemoryWidth::b128);
            }),
            "s_mov_b32 m0, s9\ns_nop 0\n"
            "buffer_load_dwordx4 v0, s[4:7], 0 offen offset:0 lds\n");
  EXPECT_EQ(emitted([&](IRBuilder &local) {
              ldsLoad(local, base, 1024, MemoryWidth::b32);
            }),
            "s_add_u32 m0, s9, 1024\ns_nop 0\n"
            "buffer_load_dword v0, s[4:7], 0 offen offset:0 lds\n");
  EXPECT_EQ(emitted([&](IRBuilder &local) {
              ldsLoad(local, base, 0, MemoryWidth::b96);
            }),
            "s_mov_b32 m0, s9\ns_nop 0\n"
            "buffer_load_dwordx3 v0, s[4:7], 0 offen offset:0 lds\n");
}

// The immediate offset moves the LDS address with the global one, so M0
// compensates for it.
TEST_F(MemoryTest, BufferToLdsCancelsTheImmediateOffsetInM0) {
  RegisterRef base{sgpr, 9, 1};
  EXPECT_EQ(emitted([&](IRBuilder &local) {
              ldsLoad(local, base, 1024, MemoryWidth::b128, {.offset = 128});
            }),
            "s_add_u32 m0, s9, 896\ns_nop 0\n"
            "buffer_load_dwordx4 v0, s[4:7], 0 offen offset:128 lds\n");
  EXPECT_EQ(emitted([&](IRBuilder &local) {
              ldsLoad(local, base, 128, MemoryWidth::b128, {.offset = 128});
            }),
            "s_mov_b32 m0, s9\ns_nop 0\n"
            "buffer_load_dwordx4 v0, s[4:7], 0 offen offset:128 lds\n");
  EXPECT_EQ(emitted([&](IRBuilder &local) {
              ldsLoad(local, base, 0, MemoryWidth::b128, {.offset = 128});
            }),
            "s_add_u32 m0, s9, -128\ns_nop 0\n"
            "buffer_load_dwordx4 v0, s[4:7], 0 offen offset:128 lds\n");
}

TEST_F(MemoryTest, BufferToLdsRejectsInvalidArguments) {
  RegisterRef base{sgpr, 9, 1};
  for (MemoryWidth width :
       {MemoryWidth::b8, MemoryWidth::b16, MemoryWidth::b64})
    expectRejected([&] { ldsLoad(builder, base, 0, width); });
  expectRejected(
      [&] { ldsLoad(builder, RegisterRef{vgpr, 9, 1}, 0, MemoryWidth::b128); });
  expectRejected(
      [&] { ldsLoad(builder, RegisterRef{sgpr, 8, 2}, 0, MemoryWidth::b128); });
  expectRejected(
      [&] { ldsLoad(builder, base, 0, MemoryWidth::b128, {.offset = 4096}); });
  expectRejected([&] {
    RegisterRef rsrc{sgpr, 6, 4};
    builder.loadBufferToLds(rsrc, RegisterRef{vgpr, 0, 1}, 0, base, 0,
                            MemoryWidth::b128);
  });
  expectRejected([&] {
    builder.loadBufferToLds(RegisterRef{sgpr, 4, 4}, RegisterRef{sgpr, 0, 1}, 0,
                            base, 0, MemoryWidth::b128);
  });
  expectRejected([&] {
    builder.loadBufferToLds(RegisterRef{sgpr, 4, 4}, RegisterRef{vgpr, 0, 1},
                            65, base, 0, MemoryWidth::b128);
  });
}

TEST(MemoryAssembly, BufferToLdsAssembles) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_lds_load", memoryWorkgroupSize,
                KernelSignature());
  IRBuilder &builder = kernel.builder();
  kernel.setLdsSize(1 << 16);
  RegisterRange rsrc = builder.allocateSgpr(4, 4);
  RegisterRange base = builder.allocateSgpr(1);
  RegisterRange soffsetReg = builder.allocateSgpr(1);
  RegisterRange voffset = builder.allocateVgpr(1);
  for (MemoryWidth width :
       {MemoryWidth::b32, MemoryWidth::b96, MemoryWidth::b128})
    for (Operand soffset : {Operand(soffsetReg), Operand(0), Operand(64)})
      for (int32_t ldsOffset : {0, 1024, 40960})
        for (int32_t offset : {0, 128, 4095})
          builder.loadBufferToLds(rsrc.ref(), voffset.ref(), soffset,
                                  base.ref(), ldsOffset, width,
                                  {.offset = offset});
  builder.createEndProgram();
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

// Emits every width, payload kind and option into one kernel.
TEST(MemoryAssembly, AllWidthsAndOptionsAssemble) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_memory_assembly", memoryWorkgroupSize,
                KernelSignature());
  IRBuilder &builder = kernel.builder();
  kernel.setLdsSize(1 << 16);
  RegisterRange rsrc = builder.allocateSgpr(4, 4);
  RegisterRange soffsetReg = builder.allocateSgpr(1);
  RegisterRange address = builder.allocateVgpr(1);
  const BufferAccess accesses[] = {
      {},
      {.offset = 4095},
      {.sc0 = true},
      {.sc1 = true},
      {.nt = true},
      {.offset = 4095, .sc0 = true, .sc1 = true, .nt = true}};
  for (RegisterKind kind : {vgpr, agpr}) {
    for (const WidthSpec &spec : widthSpecs) {
      RegisterRange payload = kind == vgpr
                                  ? builder.allocateVgpr(spec.registers, 2)
                                  : builder.allocateAgpr(spec.registers, 2);
      for (int32_t offset : {0, 65535}) {
        builder.loadDs(payload.ref(), address.ref(), spec.width, offset);
        builder.storeDs(address.ref(), payload.ref(), spec.width, offset);
      }
      for (Operand soffset : {Operand(soffsetReg), Operand(0), Operand(64)}) {
        for (const BufferAccess &access : accesses) {
          builder.loadBuffer(payload.ref(), rsrc.ref(), address.ref(), soffset,
                             spec.width, access);
          builder.storeBuffer(payload.ref(), rsrc.ref(), address.ref(), soffset,
                              spec.width, access);
        }
      }
    }
  }
  builder.createEndProgram();
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

// Host bytes past each descriptor, so an access that escapes its descriptor
// lands in memory the test can see rather than in someone else's.
static constexpr int32_t memoryPadding = 8192;
static constexpr uint32_t memorySentinel = 0xDEADBEEFu;
static constexpr const char *memoryKernelName = "codegen_memory";

static MemoryOffsets declareArguments(KernelSignature &signature) {
  int32_t in = signature.addPointer("in");
  int32_t out = signature.addPointer("out");
  int32_t first = signature.addPointer("first");
  int32_t slots = signature.addPointer("slots");
  int32_t inBytes = signature.addScalar("inBytes", 4);
  int32_t outBytes = signature.addScalar("outBytes", 4);
  return {in, out, first, slots, inBytes, outBytes};
}

// Builds a descriptor over the pointer argument. With a size argument the
// pointer and size sit in SGPRs apart from the descriptor; without one the
// pointer is loaded straight into the descriptor, so makeRsrc skips copying it,
// and the size is an immediate.
static RegisterRange makeDescriptor(Kernel &kernel, int32_t pointerOffset,
                                    std::optional<int32_t> sizeOffset,
                                    int32_t sizeBytes) {
  IRBuilder &builder = kernel.builder();
  RegisterRange rsrc = builder.allocateSgpr(4, 4);
  if (!sizeOffset) {
    kernel.loadArgument(rsrc.ref(0, 2), pointerOffset);
    builder.createWaitCnt({.scalarMemory = 0});
    builder.makeRsrc(rsrc.ref(), rsrc.ref(0, 2), sizeBytes);
    return rsrc;
  }
  RegisterRange pointer = builder.allocateSgpr(2, 2);
  RegisterRange size = builder.allocateSgpr(1);
  kernel.loadArgument(pointer.ref(), pointerOffset);
  kernel.loadArgument(size.ref(), *sizeOffset);
  builder.createWaitCnt({.scalarMemory = 0});
  builder.makeRsrc(rsrc.ref(), pointer, size);
  return rsrc;
}

static int32_t descriptorBytes(const WidthSpec &spec, int32_t slots) {
  return slots * spec.slot;
}

// Allocates the registers. voffset is the lane's slot in the buffers; the
// local lane is kept for the LDS addresses.
static MemoryRegisters allocateRegisters(Kernel &kernel, const MemoryCase &test,
                                         const WidthSpec &spec,
                                         MemoryOffsets offsets) {
  IRBuilder &builder = kernel.builder();
  bool sgprSizes = test.source == SizeSource::sgpr;
  MemoryRegisters regs{
      makeDescriptor(kernel, offsets.in,
                     sgprSizes ? std::optional<int32_t>(offsets.inBytes)
                               : std::nullopt,
                     descriptorBytes(spec, test.inSlots)),
      makeDescriptor(kernel, offsets.out,
                     sgprSizes ? std::optional<int32_t>(offsets.outBytes)
                               : std::nullopt,
                     descriptorBytes(spec, test.outSlots)),
      makeDescriptor(kernel, offsets.first, std::nullopt, memoryLaneCount * 4),
      makeDescriptor(kernel, offsets.slots, std::nullopt,
                     memoryLaneCount * spec.slot),
      builder.allocateVgpr(1),
      builder.allocateVgpr(1),
      builder.allocateVgpr(1),
      test.useAgpr
          ? builder.allocateAgpr(spec.registers, spec.registers > 1 ? 2 : 1)
          : builder.allocateVgpr(spec.registers, spec.registers > 1 ? 2 : 1)};
  RegisterRange groupBase = builder.allocateSgpr(1);
  RegisterRange global = builder.allocateVgpr(1);
  // The workitem id register packs the x, y and z ids.
  builder.andi32(regs.local.ref(), kernel.workitemId(), 0x3ff);
  builder.muli32(groupBase.ref(), kernel.workgroupIdX(), memoryWorkgroupSize);
  builder.addi32(global.ref(), regs.local, groupBase);
  builder.muli32(regs.voffset.ref(), global, spec.slot);
  builder.muli32(regs.firstOffset.ref(), global, 4);
  return regs;
}

// Fills every payload register with ones, so zero-extension and clipped loads
// are visible.
static void fillPayload(IRBuilder &builder, const MemoryRegisters &regs,
                        const WidthSpec &spec) {
  for (int32_t i = 0; i < spec.registers; ++i)
    builder.movb32(regs.payload.ref(i), -1);
}

// Sends the loaded payload through LDS and back from the mirrored lane's slot,
// so data crosses lanes, and stores the mirrored whole slot to the slots
// output. A workgroup is one wave, so no barrier is needed.
static void exchangeThroughLds(Kernel &kernel, const MemoryCase &test,
                               const WidthSpec &spec,
                               const MemoryRegisters &regs) {
  IRBuilder &builder = kernel.builder();
  kernel.setLdsSize(test.dsOffset + memoryWorkgroupSize * spec.slot);
  RegisterRange address = builder.allocateVgpr(1);
  RegisterRange mirror = builder.allocateVgpr(1);
  RegisterRange sentinel = builder.allocateVgpr(1);
  RegisterRange slot = builder.allocateVgpr(spec.slot / 4);
  builder.muli32(address.ref(), regs.local, spec.slot);
  builder.subi32(mirror.ref(), memoryWorkgroupSize - 1, regs.local);
  builder.muli32(mirror.ref(), mirror, spec.slot);
  builder.movb32(sentinel.ref(), static_cast<int32_t>(memorySentinel));
  for (int32_t i = 0; i < spec.slot / 4; ++i)
    builder.storeDs(address.ref(), sentinel.ref(), MemoryWidth::b32,
                    test.dsOffset + 4 * i);
  builder.storeDs(address.ref(), regs.payload.ref(), spec.width, test.dsOffset);
  builder.createWaitCnt({.lds = 0});

  fillPayload(builder, regs, spec);
  builder.loadDs(regs.payload.ref(), mirror.ref(), spec.width, test.dsOffset);
  for (int32_t i = 0; i < spec.slot / 4; ++i)
    builder.loadDs(slot.ref(i), mirror.ref(), MemoryWidth::b32,
                   test.dsOffset + 4 * i);
  builder.createWaitCnt({.lds = 0});
  for (int32_t i = 0; i < spec.slot / 4; ++i)
    builder.storeBuffer(slot.ref(i), regs.slots.ref(), regs.voffset.ref(), 0,
                        MemoryWidth::b32, {.offset = 4 * i});
}

// Every lane accesses memory, so the descriptor bounds do all the clipping.
static void buildMemoryKernel(Kernel &kernel, const MemoryCase &test,
                              const WidthSpec &spec, MemoryOffsets offsets) {
  IRBuilder &builder = kernel.builder();
  MemoryRegisters regs = allocateRegisters(kernel, test, spec, offsets);
  RegisterRange soffsetReg = builder.allocateSgpr(1);
  Operand soffset = test.soffset;
  if (test.soffsetInSgpr) {
    builder.movb32(soffsetReg.ref(), test.soffset);
    soffset = soffsetReg;
  }
  fillPayload(builder, regs, spec);
  builder.loadBuffer(regs.payload.ref(), regs.in.ref(), regs.voffset.ref(),
                     soffset, spec.width, test.access);
  builder.createWaitCnt({.vectorLoads = 0});
  if (test.throughLds)
    exchangeThroughLds(kernel, test, spec, regs);
  builder.storeBuffer(regs.payload.ref(), regs.out.ref(), regs.voffset.ref(),
                      soffset, spec.width, test.access);
  builder.storeBuffer(regs.payload.ref(0), regs.first.ref(),
                      regs.firstOffset.ref(), 0, MemoryWidth::b32);
  builder.createEndProgram();
}

// The input words are nonzero and differ from the sentinel, so a load that is
// not clipped to zero shows up in the output.
static std::vector<uint8_t> makeInput(int32_t bytes) {
  std::vector<uint8_t> in(bytes);
  for (int32_t i = 0; i < bytes / 4; ++i) {
    uint32_t word = (0x01000000u + static_cast<uint32_t>(i) * 2654435761u) | 1u;
    std::memcpy(in.data() + 4 * i, &word, 4);
  }
  return in;
}

static std::vector<uint8_t> makeSentinelBytes(int32_t bytes) {
  std::vector<uint8_t> sentinel(bytes);
  for (int32_t i = 0; i < bytes; i += 4)
    std::memcpy(sentinel.data() + i, &memorySentinel, 4);
  return sentinel;
}

// Whether an access at voffset is inside a descriptor of sizeBytes.
static bool isInRange(const MemoryCase &test, int32_t voffset,
                      int32_t sizeBytes) {
  return voffset + test.soffset + test.access.offset < sizeBytes;
}

using LanePayload = std::array<uint32_t, 4>;

// What each lane holds after loading from in: the bytes read, zero-extended,
// or zeros when the access is out of range.
static std::vector<LanePayload> expectedLoads(const MemoryCase &test,
                                              const WidthSpec &spec,
                                              const std::vector<uint8_t> &in) {
  std::vector<LanePayload> loads(memoryLaneCount);
  for (int32_t lane = 0; lane < memoryLaneCount; ++lane) {
    int32_t voffset = lane * spec.slot;
    if (!isInRange(test, voffset, descriptorBytes(spec, test.inSlots)))
      continue;
    int32_t address = voffset + test.soffset + test.access.offset;
    std::memcpy(loads[lane].data(), in.data() + address, spec.bytes);
  }
  return loads;
}

static int32_t mirrorOf(int32_t lane) {
  return lane - lane % memoryWorkgroupSize + memoryWorkgroupSize - 1 -
         lane % memoryWorkgroupSize;
}

// The slot of a lane after the LDS stores: the sentinel with the low bytes of
// its payload written over it.
static std::vector<uint8_t> expectedLdsSlot(const WidthSpec &spec,
                                            const LanePayload &payload) {
  std::vector<uint8_t> slot = makeSentinelBytes(spec.slot);
  std::memcpy(slot.data(), payload.data(), spec.bytes);
  return slot;
}

static MemoryResults computeExpected(const MemoryCase &test,
                                     const WidthSpec &spec,
                                     const std::vector<uint8_t> &in) {
  std::vector<LanePayload> loads = expectedLoads(test, spec, in);
  int32_t outBytes = descriptorBytes(spec, test.outSlots);
  MemoryResults expected{makeSentinelBytes(outBytes + memoryPadding),
                         makeSentinelBytes(memoryLaneCount * 4),
                         makeSentinelBytes(memoryLaneCount * spec.slot)};
  for (int32_t lane = 0; lane < memoryLaneCount; ++lane) {
    const LanePayload &payload =
        test.throughLds ? loads[mirrorOf(lane)] : loads[lane];
    int32_t voffset = lane * spec.slot;
    if (isInRange(test, voffset, outBytes))
      std::memcpy(expected.out.data() + voffset + test.soffset +
                      test.access.offset,
                  payload.data(), spec.bytes);
    std::memcpy(expected.first.data() + 4 * lane, payload.data(), 4);
    if (!test.throughLds)
      continue;
    std::vector<uint8_t> slot = expectedLdsSlot(spec, loads[mirrorOf(lane)]);
    std::memcpy(expected.slots.data() + voffset, slot.data(), spec.slot);
  }
  return expected;
}

static std::vector<std::byte>
makeKernarg(int32_t segmentSize, const MemoryOffsets &offsets,
            const MemoryCase &test, const WidthSpec &spec,
            const std::vector<DevicePtr> &buffers) {
  std::vector<std::byte> kernarg(segmentSize);
  writeArgument(kernarg, offsets.in, buffers[0].get());
  writeArgument(kernarg, offsets.out, buffers[1].get());
  writeArgument(kernarg, offsets.first, buffers[2].get());
  writeArgument(kernarg, offsets.slots, buffers[3].get());
  writeArgument(kernarg, offsets.inBytes,
                static_cast<uint32_t>(descriptorBytes(spec, test.inSlots)));
  writeArgument(kernarg, offsets.outBytes,
                static_cast<uint32_t>(descriptorBytes(spec, test.outSlots)));
  return kernarg;
}

// Uploads the buffers, launches the kernel and downloads the outputs.
static void runMemory(const std::vector<uint8_t> &hsaco,
                      const MemoryOffsets &offsets, int32_t segmentSize,
                      const MemoryCase &test, const WidthSpec &spec,
                      const std::vector<uint8_t> &in, MemoryResults &results) {
  ModulePtr module;
  hipFunction_t function = nullptr;
  loadKernel(hsaco, memoryKernelName, module, function);
  std::vector<DevicePtr> buffers(4);
  upload(in, buffers[0]);
  upload(results.out, buffers[1]);
  upload(results.first, buffers[2]);
  upload(results.slots, buffers[3]);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  std::vector<std::byte> kernarg =
      makeKernarg(segmentSize, offsets, test, spec, buffers);
  launchWithKernarg(function, memoryGroupCount, memoryWorkgroupSize, kernarg);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  download(buffers[1], results.out);
  download(buffers[2], results.first);
  download(buffers[3], results.slots);
}

static void checkAssembly(const std::string &assembly, const MemoryCase &test,
                          const WidthSpec &spec) {
  if (test.throughLds)
    expectLinesInOrder(assembly, {spec.bufferLoad, spec.dsStore, spec.dsLoad,
                                  spec.bufferStore});
  else
    expectLinesInOrder(assembly, {spec.bufferLoad, spec.bufferStore});
  using CacheBit = std::pair<bool, const char *>;
  for (const auto &[enabled, bit] :
       {CacheBit{test.access.sc0, " sc0"}, CacheBit{test.access.sc1, " sc1"},
        CacheBit{test.access.nt, " nt"}})
    EXPECT_EQ(enabled, countLinesContaining(assembly, bit) > 0) << bit;
  if (test.source == SizeSource::sgpr)
    return;
  // The pointers are already in place, so no pointer copy is emitted.
  EXPECT_EQ(countLinesContaining(assembly, "s_mov_b64"), 0);
  EXPECT_GT(countLinesContaining(
                assembly, std::to_string(descriptorBytes(spec, test.inSlots))),
            0);
  EXPECT_GT(countLinesContaining(
                assembly, std::to_string(descriptorBytes(spec, test.outSlots))),
            0);
}

static void expectBytes(const std::vector<uint8_t> &actual,
                        const std::vector<uint8_t> &expected,
                        const char *name) {
  ASSERT_EQ(actual.size(), expected.size()) << name;
  for (int32_t i = 0; i < std::ssize(actual); ++i)
    ASSERT_EQ(actual[i], expected[i]) << name << " byte " << i;
}

static void runMemoryCase(const MemoryCase &test) {
  const WidthSpec &spec = specOf(test.width);
  Target target = Target::get("gfx950");
  KernelSignature signature;
  MemoryOffsets offsets = declareArguments(signature);
  Kernel kernel(target, memoryKernelName, memoryWorkgroupSize,
                std::move(signature));
  buildMemoryKernel(kernel, test, spec, offsets);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  checkAssembly(assembly, test, spec);

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<uint8_t> in =
      makeInput(descriptorBytes(spec, test.inSlots) + memoryPadding);
  MemoryResults expected = computeExpected(test, spec, in);
  MemoryResults results{makeSentinelBytes(std::ssize(expected.out)),
                        makeSentinelBytes(std::ssize(expected.first)),
                        makeSentinelBytes(std::ssize(expected.slots))};
  runMemory(hsaco, offsets, kernel.signature().segmentSize(), test, spec, in,
            results);
  if (testing::Test::HasFatalFailure())
    return;
  expectBytes(results.out, expected.out, "out");
  expectBytes(results.first, expected.first, "first");
  expectBytes(results.slots, expected.slots, "slots");
}

// Runs the case at every width, stopping at the first failure or skip.
static void runAllWidths(MemoryCase test) {
  for (const WidthSpec &spec : widthSpecs) {
    SCOPED_TRACE(spec.name);
    test.width = spec.width;
    runMemoryCase(test);
    if (testing::Test::HasFailure() || testing::Test::IsSkipped())
      return;
  }
}

TEST(RsrcDevice, SgprSizes) { runMemoryCase({.source = SizeSource::sgpr}); }

TEST(RsrcDevice, ImmediateSizes) {
  runMemoryCase({.source = SizeSource::immediate});
}

// The range check covers voffset + soffset + offset, so lanes clip earlier the
// larger soffset and offset are.
TEST(MemoryDevice, BufferOutOfBounds) { runAllWidths({}); }

TEST(MemoryDevice, SgprSoffsetOffsetAndAllCacheBits) {
  runAllWidths(
      {.soffsetInSgpr = true,
       .soffset = 32,
       .access = {.offset = 16, .sc0 = true, .sc1 = true, .nt = true}});
}

TEST(MemoryDevice, InlineSoffsetIsRangeChecked) {
  runAllWidths({.soffset = 64, .access = {.offset = 32}});
}

TEST(MemoryDevice, MaximumOffsetWithEachCacheBit) {
  // The descriptors cover every access, so only the encoding is under test.
  MemoryCase test{.inSlots = 2048,
                  .outSlots = 2048,
                  .soffset = 64,
                  .access = {.offset = 4080}};
  for (bool BufferAccess::*bit :
       {&BufferAccess::sc0, &BufferAccess::sc1, &BufferAccess::nt}) {
    test.access.*bit = true;
    runAllWidths(test);
    if (testing::Test::HasFailure() || testing::Test::IsSkipped())
      return;
    test.access.*bit = false;
  }
}

TEST(MemoryDevice, ThroughLds) {
  for (int32_t dsOffset : {0, 64000}) {
    SCOPED_TRACE("ds offset " + std::to_string(dsOffset));
    runAllWidths({.throughLds = true, .dsOffset = dsOffset});
    if (testing::Test::HasFailure() || testing::Test::IsSkipped())
      return;
  }
}

TEST(MemoryDevice, AgprPayloads) {
  runAllWidths({.useAgpr = true});
  runAllWidths({.useAgpr = true, .throughLds = true});
}
