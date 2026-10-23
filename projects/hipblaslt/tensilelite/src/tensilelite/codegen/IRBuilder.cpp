// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Operand.hpp>

#include "code.hpp"
#include "container.hpp"
#include "format.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/instruction.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocm::tensilelite::codegen;

// gfx9 operand legality rules.

// The gfx9 (gfx950) operand rules that rocisa does not check, verified against
// the LLVM assembler. Violations are rejected with "literal operands are not
// supported" (a literal in VOP3), "violates constant bus restrictions" (two
// distinct SGPRs or literals in a VALU instruction), "only one unique
// literal operand is allowed" (SOP2 and SOPC) and "invalid register alignment"
// (a 64-bit SGPR operand at an odd index). Immediates are int32 values; the
// 64-bit predicates treat them as sign-extended to 64 bits.

// Wait states a VALU read of an SGPR needs after a VALU instruction wrote it;
// the s_nop that covers them takes this minus one. This also covers the
// carry-in read of v_addc_co_u32 or v_subb_co_u32 after v_add_co_u32 or
// v_sub_co_u32 wrote the carry to VCC or an SGPR pair: LLVM's hazard
// recognizer inserts s_nop 1 there on gfx940 and gfx950, but not on gfx90a.
// SALU carry chains through SCC need none.
static constexpr int32_t valuWriteSgprReadWaitStates = 2;

// Wait states a VALU read of a transcendental result (v_rcp_f32,
// v_rcp_iflag_f32) needs on gfx940 and gfx950, per LLVM's hazard recognizer.
static constexpr int32_t transResultWaitStates = 1;
// Wait states v_readfirstlane_b32 needs after a VALU instruction wrote its
// source VGPR on gfx940 and gfx950, per LLVM's hazard recognizer.
static constexpr int32_t valuWriteVgprReadlaneWaitStates = 1;

// Integer inline constants. The float inline constants also match by bit
// pattern but are treated as literals, which is conservative.
static bool isInlineConstant(int32_t value) {
  return value >= -16 && value <= 64;
}

// True if operand is an immediate that is not an inline constant.
static bool isLiteral(Operand operand) {
  return operand.isImmediate() && !isInlineConstant(operand.immediate());
}

static bool isVector(Operand operand) {
  return operand.isRegister(RegisterKind::vgpr) ||
         operand.isRegister(RegisterKind::agpr);
}

// True if both operands are the same SGPR or the same literal.
static bool sameScalarSource(Operand a, Operand b) {
  if (a.isRegister(RegisterKind::sgpr) && b.isRegister(RegisterKind::sgpr))
    return a.reg() == b.reg();
  return isLiteral(a) && isLiteral(b) && a.immediate() == b.immediate();
}

static bool usesConstantBus(Operand operand) {
  return operand.isRegister(RegisterKind::sgpr) || isLiteral(operand);
}

// Constant bus reads of a VALU instruction: distinct SGPRs plus distinct
// literals.
static int32_t constantBusReads(Operand src0, Operand src1) {
  int32_t reads = 0;
  if (usesConstantBus(src0))
    ++reads;
  if (usesConstantBus(src1) && !sameScalarSource(src0, src1))
    ++reads;
  return reads;
}

// VOP2: src1 must be a VGPR; src0 may be anything, including one literal.
static bool isLegalVop2(Operand src0, Operand src1) {
  return src1.isRegister(RegisterKind::vgpr) &&
         constantBusReads(src0, src1) <= 1;
}

// VOP3, including VOP2 ops the assembler promotes to VOP3 when src1 is not a
// VGPR: no literals and at most one constant bus read.
static bool isLegalVop3(Operand src0, Operand src1) {
  return !isLiteral(src0) && !isLiteral(src1) &&
         constantBusReads(src0, src1) <= 1;
}

// SOP2, and SOPC (s_cmp) as well: no vector registers and at most one distinct
// literal. A 64-bit literal must be non-negative, since the hardware
// zero-extends it; the caller guarantees that.
static bool isLegalSop2(Operand src0, Operand src1) {
  if (isVector(src0) || isVector(src1))
    return false;
  return !(isLiteral(src0) && isLiteral(src1) && !sameScalarSource(src0, src1));
}

// A 64-bit instruction sign-extends an inline constant and zero-extends a
// literal, so a sign-extended value is encoded exactly iff it is inline or
// non-negative.
static bool isEncodableImmediate64(int32_t value) {
  return isInlineConstant(value) || value >= 0;
}

// True if one 64-bit mov (s_mov_b64 or v_mov_b64) can write src into the pair
// dst: both are pairs at an even index of an allowed kind, and an immediate is
// an inline constant or a non-negative literal.
static bool isLegalMov64(RegisterRef dst, Operand src) {
  if (dst.kind == RegisterKind::agpr || dst.start % 2 != 0)
    return false;
  if (src.isImmediate())
    return isEncodableImmediate64(src.immediate());
  // An SGPR destination has no vector source and a VGPR destination has no
  // AGPR source.
  RegisterRef reg = src.reg();
  bool kindIsLegal = dst.kind == RegisterKind::sgpr
                         ? reg.kind == RegisterKind::sgpr
                         : reg.kind != RegisterKind::agpr;
  return kindIsLegal && reg.start % 2 == 0;
}

// True if src is a legal 64-bit SALU source: an SGPR pair at an even index or
// an inline constant or non-negative immediate.
static bool isLegalSalu64Source(Operand src) {
  if (src.isImmediate())
    return isEncodableImmediate64(src.immediate());
  return src.isRegister(RegisterKind::sgpr) && src.reg().start % 2 == 0;
}

// True if src is a legal 64-bit VOP3 source: a VGPR or SGPR pair at an even
// index, or an inline-constant immediate.
static bool isLegalVop3Source64(Operand src) {
  if (src.isImmediate())
    return isInlineConstant(src.immediate());
  RegisterRef reg = src.reg();
  return reg.kind != RegisterKind::agpr && reg.start % 2 == 0;
}

// True if src is a VGPR or an inline constant: the only legal source of
// v_cndmask_b32 and of the v_addc_co_u32 and v_subb_co_u32 data operands,
// because the mask or the carry-in already uses the constant bus.
static bool isVgprOrInlineConstant(Operand src) {
  if (src.isImmediate())
    return isInlineConstant(src.immediate());
  return src.isRegister(RegisterKind::vgpr);
}

// True if ref can hold a lane mask: an SGPR pair at an even index.
static bool isLaneMask(RegisterRef ref) {
  return ref.kind == RegisterKind::sgpr && ref.count == 2 && ref.start % 2 == 0;
}

// Operand helpers: validation and conversion shared by the IRBuilder ops.

namespace {

// Temporary registers that stay allocated until the instructions reading them
// have been emitted.
using Temporaries = std::vector<RegisterRange>;

} // namespace

RegisterRef Operand::reg() const {
  assert(isRegister() && "operand is not a register");
  const rocisa::Container &container =
      *std::get<std::shared_ptr<rocisa::Container>>(
          static_cast<const InstructionInput &>(*this));
  const rocisa::RegisterContainer &c =
      static_cast<const rocisa::RegisterContainer &>(container);
  // The register prefix rocisa prints identifies the kind.
  if (c.regType == "s")
    return RegisterRef{RegisterKind::sgpr, c.regIdx, c.regNum};
  if (c.regType == "v")
    return RegisterRef{RegisterKind::vgpr, c.regIdx, c.regNum};
  assert(c.regType == "acc" && "unknown register prefix");
  return RegisterRef{RegisterKind::agpr, c.regIdx, c.regNum};
}

// Returns ref as rocisa prints it, such as v[2:3], for diagnostics.
static std::string registerText(RegisterRef ref) {
  return ref.container()->toString();
}

// Throws std::invalid_argument naming opName unless ref has exactly count
// registers.
static void requireRegisterCount(std::string_view opName, RegisterRef ref,
                                 int32_t count) {
  if (ref.count == count)
    return;
  std::string needed = count == 1 ? "single registers" : "register pairs";
  throw std::invalid_argument(std::string(opName) + " needs " + needed +
                              ", got " + registerText(ref));
}

// Throws std::invalid_argument naming opName if ref is an AGPR or does not have
// exactly count registers.
static void requireRegister(std::string_view opName, RegisterRef ref,
                            int32_t count) {
  if (ref.kind == RegisterKind::agpr)
    throw std::invalid_argument(std::string(opName) +
                                " does not accept agpr register " +
                                registerText(ref));
  requireRegisterCount(opName, ref, count);
}

// Throws std::invalid_argument naming opName if a register source is an AGPR
// or does not have exactly count registers; every immediate is accepted.
static void requireOperands(std::string_view opName, Operand a, Operand b,
                            int32_t count) {
  for (Operand operand : {a, b})
    if (operand.isRegister())
      requireRegister(opName, operand.reg(), count);
}

// Throws std::invalid_argument naming opName if dst is an SGPR and a or b is a
// VGPR, since there is no implicit readfirstlane.
static void requireScalarSources(std::string_view opName, RegisterRef dst,
                                 Operand a, Operand b) {
  if (dst.kind != RegisterKind::sgpr)
    return;
  for (Operand source : {a, b}) {
    if (!source.isRegister(RegisterKind::vgpr))
      continue;
    throw std::invalid_argument(
        std::string(opName) + " cannot write sgpr destination " +
        registerText(dst) + " from vgpr operand " + registerText(source.reg()));
  }
}

// The checks of the ops that write a count-register dst from a and b.
static void requireArithmetic(std::string_view opName, RegisterRef dst,
                              Operand a, Operand b, int32_t count) {
  requireRegister(opName, dst, count);
  requireOperands(opName, a, b, count);
  requireScalarSources(opName, dst, a, b);
}

// Returns the low (index 0) or high (index 1) 32 bits of a register pair or of
// an immediate sign-extended to 64 bits.
static Operand operandHalf(Operand src, int32_t index) {
  if (src.isRegister())
    return Operand(src.reg().slice(index));
  int32_t value = src.immediate();
  return Operand(index == 0 ? value : (value < 0 ? -1 : 0));
}

static RegisterRange allocate(IRBuilder &builder, RegisterKind kind,
                              int32_t count = 1) {
  return kind == RegisterKind::sgpr ? builder.allocateSgpr(count, count)
                                    : builder.allocateVgpr(count, count);
}

// Copies source into a new temporary of kind with width registers (1 or 2,
// aligned to its width) that lives as long as temps does, and returns it.
static Operand copyToTemporary(IRBuilder &builder, RegisterKind kind,
                               int32_t width, Operand source,
                               Temporaries &temps) {
  assert(kind != RegisterKind::agpr && "temporaries are sgprs or vgprs");
  RegisterRange temp = allocate(builder, kind, width);
  if (width == 1)
    builder.movb32(temp.ref(), source);
  else
    builder.movb64(temp.ref(), source);
  temps.push_back(std::move(temp));
  return Operand(temps.back());
}

// Copies literals, and the second of two distinct SGPRs, into VGPR temporaries
// so that a and b suit a VOP3 instruction, which takes no literals and one
// constant bus read.
static void legalizeVop3Sources(IRBuilder &builder, Operand &a, Operand &b,
                                Temporaries &temps) {
  if (isLiteral(a))
    a = copyToTemporary(builder, RegisterKind::vgpr, 1, a, temps);
  if (isLiteral(b))
    b = copyToTemporary(builder, RegisterKind::vgpr, 1, b, temps);
  // Two distinct SGPRs exceed the constant bus.
  if (!isLegalVop3(a, b))
    b = copyToTemporary(builder, RegisterKind::vgpr, 1, b, temps);
  assert(isLegalVop3(a, b) && "illegal vop3 operands");
}

// 64-bit SALU sources must be aligned SGPR pairs or encodable immediates, and
// SOP2 and SOPC take one literal.
static void legalizeSalu64Sources(IRBuilder &builder, Operand &a, Operand &b,
                                  Temporaries &temps) {
  if (!isLegalSalu64Source(a))
    a = copyToTemporary(builder, RegisterKind::sgpr, 2, a, temps);
  if (!isLegalSalu64Source(b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 2, b, temps);
  if (!isLegalSop2(a, b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 2, b, temps);
}

// A VALU-written SGPR, a mask or a carry, is read by a following instruction.
static void emitValuSgprWriteNop(IRBuilder &builder) {
  builder.create<rocisa::SNop>(valuWriteSgprReadWaitStates - 1);
}

// Core builder.

// rocisa's SWaitCnt uses -1 for a counter that is not waited on.
static constexpr int32_t noWait = -1;

IRBuilder::InsertionGuard::InsertionGuard(IRBuilder &builder)
    : builder(builder), savedInsertionPoint(&builder.insertionPoint()) {}

IRBuilder::InsertionGuard::~InsertionGuard() {
  builder.setInsertionPoint(*savedInsertionPoint);
}

IRBuilder::IRBuilder(const Target &target, RegisterAllocator &allocator,
                     rocisa::Module &insertionPoint)
    : targetInfo(&target), registers(&allocator), module(&insertionPoint) {
  target.selectForCurrentThread();
}

void IRBuilder::insertItem(const std::shared_ptr<rocisa::Item> &item) {
  if (!item)
    throw std::invalid_argument("cannot insert a null item");
  if (item->parent)
    throw std::logic_error("item is already in a module");
  module->add(item);
}

std::shared_ptr<rocisa::Label> IRBuilder::createLabel(std::string_view name) {
  std::string unique = std::string(name) + "_" + std::to_string(nextLabelId++);
  return std::make_shared<rocisa::Label>(unique, "");
}

std::shared_ptr<rocisa::TextBlock>
IRBuilder::createComment(std::string_view text) {
  return create<rocisa::TextBlock>(rocisa::slash(std::string(text)));
}

std::shared_ptr<rocisa::SWaitCnt>
IRBuilder::createWaitCnt(WaitCounts counts, std::string_view comment) {
  return create<rocisa::SWaitCnt>(
      counts.vectorLoads.value_or(noWait), counts.vectorStores.value_or(noWait),
      counts.lds.value_or(noWait), counts.scalarMemory.value_or(noWait),
      std::string(comment));
}

std::shared_ptr<rocisa::SEndpgm> IRBuilder::createEndProgram() {
  return create<rocisa::SEndpgm>();
}

RegisterRange IRBuilder::allocateVgpr(int count, int alignment,
                                      std::string_view tag) {
  return registers->allocate(RegisterKind::vgpr, count, alignment, tag);
}

RegisterRange IRBuilder::allocateSgpr(int count, int alignment,
                                      std::string_view tag) {
  return registers->allocate(RegisterKind::sgpr, count, alignment, tag);
}

RegisterRange IRBuilder::allocateAgpr(int count, int alignment,
                                      std::string_view tag) {
  return registers->allocate(RegisterKind::agpr, count, alignment, tag);
}

// Moves.

// Rejects the kind combinations that have no instruction.
static void requireLegalSource(std::string_view opName, RegisterRef dst,
                               Operand src) {
  std::string prefix = std::string(opName) + " cannot write ";
  if (src.isImmediate()) {
    // v_accvgpr_write_b32 takes no literal.
    if (dst.kind != RegisterKind::agpr || isInlineConstant(src.immediate()))
      return;
    throw std::invalid_argument(prefix + "literal " +
                                std::to_string(src.immediate()) +
                                " to agpr destination " + registerText(dst) +
                                ", move it to a vgpr first");
  }
  RegisterRef reg = src.reg();
  if (dst.kind == RegisterKind::sgpr && reg.kind != RegisterKind::sgpr)
    throw std::invalid_argument(
        prefix + "sgpr destination " + registerText(dst) + " from " +
        registerText(reg) + ", there is no implicit readfirstlane");
  if (dst.kind == RegisterKind::agpr && reg.kind == RegisterKind::agpr)
    throw std::invalid_argument(
        prefix + "agpr destination " + registerText(dst) + " from " +
        registerText(reg) + ", rocisa has no v_accvgpr_mov_b32");
}

void IRBuilder::movb32(RegisterRef dst, Operand src) {
  requireRegisterCount("movb32", dst, 1);
  if (src.isRegister())
    requireRegisterCount("movb32", src.reg(), 1);
  requireLegalSource("movb32", dst, src);
  switch (dst.kind) {
  case RegisterKind::sgpr:
    create<rocisa::SMovB32>(dst.container(), src);
    return;
  case RegisterKind::vgpr:
    if (src.isRegister(RegisterKind::agpr))
      create<rocisa::VAccvgprReadB32>(dst.container(), src);
    else
      create<rocisa::VMovB32>(dst.container(), src);
    return;
  case RegisterKind::agpr:
    create<rocisa::VAccvgprWriteB32>(dst.container(), src);
    return;
  }
}

void IRBuilder::movb64(RegisterRef dst, Operand src) {
  requireRegisterCount("movb64", dst, 2);
  if (src.isRegister())
    requireRegisterCount("movb64", src.reg(), 2);
  // An immediate's high half is 0 or -1, always inline, so only the value
  // itself can be an illegal agpr source.
  requireLegalSource("movb64", dst, src);
  if (isLegalMov64(dst, src)) {
    if (dst.kind == RegisterKind::sgpr)
      create<rocisa::SMovB64>(dst.container(), src);
    else
      create<rocisa::VMovB64>(dst.container(), src);
    return;
  }
  // When dst starts above an overlapping src, the high half goes first so the
  // source high word is read before it is overwritten.
  bool highFirst = src.isRegister(dst.kind) && dst.start > src.reg().start;
  for (int32_t i = 0; i < 2; ++i) {
    int32_t index = highFirst ? 1 - i : i;
    movb32(dst.slice(index), operandHalf(src, index));
  }
}

// i32 arithmetic.

namespace {

enum class BinaryOp { add, sub, mul };

} // namespace

// Picks the SALU form for an sgpr dst and the VALU form otherwise.
static void createBinaryI32(IRBuilder &builder, BinaryOp op, RegisterRef dst,
                            Operand a, Operand b) {
  bool scalar = dst.kind == RegisterKind::sgpr;
  switch (op) {
  case BinaryOp::add:
    if (scalar)
      builder.create<rocisa::SAddU32>(dst.container(), a, b);
    else
      builder.create<rocisa::VAddU32>(dst.container(), a, b);
    return;
  case BinaryOp::sub:
    if (scalar)
      builder.create<rocisa::SSubU32>(dst.container(), a, b);
    else
      builder.create<rocisa::VSubU32>(dst.container(), a, b);
    return;
  case BinaryOp::mul:
    if (scalar)
      builder.create<rocisa::SMulI32>(dst.container(), a, b);
    else
      builder.create<rocisa::VMulLOU32>(dst.container(), a, b);
    return;
  }
}

static void emitBinaryI32(IRBuilder &builder, BinaryOp op, RegisterRef dst,
                          Operand a, Operand b) {
  // Copies go into fresh temporaries before the single op, which reads all its
  // sources before writing dst, so dst may alias a or b.
  Temporaries temps;
  if (dst.kind == RegisterKind::sgpr) {
    // Only two distinct literals are illegal, and one of them moves to a copy.
    if (!isLegalSop2(a, b))
      b = copyToTemporary(builder, RegisterKind::sgpr, 1, b, temps);
    assert(isLegalSop2(a, b) && "illegal salu operands");
  } else if (op == BinaryOp::mul) {
    // v_mul_lo_u32 is VOP3 only, so it takes no literals.
    legalizeVop3Sources(builder, a, b, temps);
  } else {
    // x - c is x + (-c) modulo 2^32, which needs no copy since VOP2 src0
    // accepts a literal and rocisa has no v_subrev.
    if (op == BinaryOp::sub && b.isImmediate()) {
      op = BinaryOp::add;
      b = Operand(
          static_cast<int32_t>(0u - static_cast<uint32_t>(b.immediate())));
    }
    // A VGPR in src1 gives VOP2, whose src0 accepts anything.
    if (op == BinaryOp::add && !b.isRegister(RegisterKind::vgpr) &&
        a.isRegister(RegisterKind::vgpr))
      std::swap(a, b);
    if (!isLegalVop2(a, b) && !isLegalVop3(a, b))
      // A VGPR in src1 makes any src0 legal.
      b = copyToTemporary(builder, RegisterKind::vgpr, 1, b, temps);
    assert((isLegalVop2(a, b) || isLegalVop3(a, b)) && "illegal valu operands");
  }
  createBinaryI32(builder, op, dst, a, b);
}

void IRBuilder::addi32(RegisterRef dst, Operand a, Operand b) {
  requireArithmetic("addi32", dst, a, b, 1);
  emitBinaryI32(*this, BinaryOp::add, dst, a, b);
}

void IRBuilder::subi32(RegisterRef dst, Operand a, Operand b) {
  requireArithmetic("subi32", dst, a, b, 1);
  emitBinaryI32(*this, BinaryOp::sub, dst, a, b);
}

void IRBuilder::muli32(RegisterRef dst, Operand a, Operand b) {
  requireArithmetic("muli32", dst, a, b, 1);
  emitBinaryI32(*this, BinaryOp::mul, dst, a, b);
}

// i64 arithmetic.

namespace {

// The 32-bit halves of a 64-bit operand.
struct Halves {
  Operand lo;
  Operand hi;
};

// Two halves that one instruction reads together.
struct HalfPair {
  Operand *first;
  Operand *second;
};

} // namespace

// The carry chains write dst's low register before they read the high halves,
// so a high half that is that register, possible only for overlapping pairs, is
// copied first.
static void protectHighHalves(IRBuilder &builder, RegisterRef dst, Halves &a,
                              Halves &b, Temporaries &temps) {
  for (Operand *half : {&a.hi, &b.hi}) {
    if (!half->isRegister() || half->reg() != dst.slice(0))
      continue;
    *half = copyToTemporary(builder, half->reg().kind, 1, *half, temps);
  }
}

static void emitSaluCarryChain(IRBuilder &builder, BinaryOp op, RegisterRef dst,
                               Halves a, Halves b) {
  Temporaries temps;
  for (HalfPair pair : {HalfPair{&a.lo, &b.lo}, HalfPair{&a.hi, &b.hi}})
    if (!isLegalSop2(*pair.first, *pair.second))
      *pair.second =
          copyToTemporary(builder, RegisterKind::sgpr, 1, *pair.second, temps);
  protectHighHalves(builder, dst, a, b, temps);
  // Every copy is emitted above because nothing may write SCC between the
  // halves.
  std::shared_ptr<rocisa::RegisterContainer> lo = dst.slice(0).container();
  std::shared_ptr<rocisa::RegisterContainer> hi = dst.slice(1).container();
  if (op == BinaryOp::add) {
    builder.create<rocisa::SAddU32>(lo, a.lo, b.lo);
    builder.create<rocisa::SAddCU32>(hi, a.hi, b.hi);
    return;
  }
  builder.create<rocisa::SSubU32>(lo, a.lo, b.lo);
  builder.create<rocisa::SSubBU32>(hi, a.hi, b.hi);
}

static void emitValuCarryChain(IRBuilder &builder, BinaryOp op, RegisterRef dst,
                               Halves a, Halves b) {
  Temporaries temps;
  // The low half is VOP3 because its carry-out is an SGPR pair.
  legalizeVop3Sources(builder, a.lo, b.lo, temps);
  for (Operand *half : {&a.hi, &b.hi})
    if (!isVgprOrInlineConstant(*half))
      *half = copyToTemporary(builder, RegisterKind::vgpr, 1, *half, temps);
  protectHighHalves(builder, dst, a, b, temps);
  assert(isVgprOrInlineConstant(a.hi) && isVgprOrInlineConstant(b.hi) &&
         "illegal carry-in operands");
  // A temporary pair rather than VCC, because the allocator does not track VCC
  // and using it would clobber a caller's value invisibly.
  RegisterRange carry = builder.allocateSgpr(2, 2);
  std::shared_ptr<rocisa::RegisterContainer> lo = dst.slice(0).container();
  std::shared_ptr<rocisa::RegisterContainer> hi = dst.slice(1).container();
  bool isAdd = op == BinaryOp::add;
  if (isAdd)
    builder.create<rocisa::VAddCOU32>(lo, carry.container(), a.lo, b.lo);
  else
    builder.create<rocisa::VSubCoU32>(lo, carry.container(), a.lo, b.lo);
  // The carry-in read follows the carry write.
  emitValuSgprWriteNop(builder);
  if (isAdd)
    builder.create<rocisa::VAddCCOU32>(hi, carry.container(), a.hi, b.hi,
                                       carry.container());
  else
    builder.create<rocisa::VSubBCoU32>(hi, carry.container(), a.hi, b.hi,
                                       carry.container());
}

static void legalizeProductHalves(IRBuilder &builder, Halves &a, Halves &b,
                                  RegisterKind kind, Temporaries &temps) {
  // VOP3 takes no literals.
  if (kind == RegisterKind::vgpr)
    for (Operand *half : {&a.lo, &a.hi, &b.lo, &b.hi})
      if (isLiteral(*half))
        *half = copyToTemporary(builder, kind, 1, *half, temps);
  for (HalfPair pair : std::array<HalfPair, 3>{
           {{&a.lo, &b.lo}, {&a.lo, &b.hi}, {&a.hi, &b.lo}}}) {
    bool legal = kind == RegisterKind::vgpr
                     ? isLegalVop3(*pair.first, *pair.second)
                     : isLegalSop2(*pair.first, *pair.second);
    if (!legal)
      *pair.second = copyToTemporary(builder, kind, 1, *pair.second, temps);
  }
}

static void emitSaluMul64(IRBuilder &builder, RegisterRef dst, Halves a,
                          Halves b) {
  Temporaries temps;
  legalizeProductHalves(builder, a, b, RegisterKind::sgpr, temps);
  RegisterRange high = builder.allocateSgpr(1);
  RegisterRange cross = builder.allocateSgpr(1);
  builder.create<rocisa::SMulHIU32>(high.container(), a.lo, b.lo);
  builder.create<rocisa::SMulI32>(cross.container(), a.lo, b.hi);
  builder.create<rocisa::SAddU32>(high.container(), high.container(),
                                  cross.container());
  builder.create<rocisa::SMulI32>(cross.container(), a.hi, b.lo);
  builder.create<rocisa::SMulI32>(dst.slice(0).container(), a.lo, b.lo);
  builder.create<rocisa::SAddU32>(dst.slice(1).container(), high.container(),
                                  cross.container());
}

// dst is written only by the last two instructions. The low write is the last
// read of a and b and reads them before writing, and the high write reads only
// temporaries, so dst may alias or overlap a or b.
static void emitValuMul64(IRBuilder &builder, RegisterRef dst, Halves a,
                          Halves b) {
  Temporaries temps;
  legalizeProductHalves(builder, a, b, RegisterKind::vgpr, temps);
  RegisterRange high = builder.allocateVgpr(1);
  RegisterRange crossA = builder.allocateVgpr(1);
  RegisterRange crossB = builder.allocateVgpr(1);
  builder.create<rocisa::VMulHIU32>(high.container(), a.lo, b.lo);
  builder.create<rocisa::VMulLOU32>(crossA.container(), a.lo, b.hi);
  builder.create<rocisa::VMulLOU32>(crossB.container(), a.hi, b.lo);
  builder.create<rocisa::VMulLOU32>(dst.slice(0).container(), a.lo, b.lo);
  builder.create<rocisa::VAdd3U32>(dst.slice(1).container(), high.container(),
                                   crossA.container(), crossB.container());
}

static void emitBinaryI64(IRBuilder &builder, BinaryOp op, RegisterRef dst,
                          Operand a, Operand b) {
  Halves aHalves{operandHalf(a, 0), operandHalf(a, 1)};
  Halves bHalves{operandHalf(b, 0), operandHalf(b, 1)};
  bool scalar = dst.kind == RegisterKind::sgpr;
  if (op == BinaryOp::mul && scalar)
    return emitSaluMul64(builder, dst, aHalves, bHalves);
  if (op == BinaryOp::mul)
    return emitValuMul64(builder, dst, aHalves, bHalves);
  if (scalar)
    return emitSaluCarryChain(builder, op, dst, aHalves, bHalves);
  emitValuCarryChain(builder, op, dst, aHalves, bHalves);
}

void IRBuilder::addi64(RegisterRef dst, Operand a, Operand b) {
  requireArithmetic("addi64", dst, a, b, 2);
  emitBinaryI64(*this, BinaryOp::add, dst, a, b);
}

void IRBuilder::subi64(RegisterRef dst, Operand a, Operand b) {
  requireArithmetic("subi64", dst, a, b, 2);
  emitBinaryI64(*this, BinaryOp::sub, dst, a, b);
}

void IRBuilder::muli64(RegisterRef dst, Operand a, Operand b) {
  requireArithmetic("muli64", dst, a, b, 2);
  emitBinaryI64(*this, BinaryOp::mul, dst, a, b);
}

// Division.

// Signed 32-bit division as LLVM's AMDGPU expansions. The exact path is LLVM's
// 32-bit expansion (AMDGPUCodeGenPrepare::expandDivRem32 and
// AMDGPUTargetLowering::LowerUDIVREM): an f32 reciprocal scaled by 2^32 - 512,
// one Newton step and two corrections, with abs and a sign fix-up for sdiv. The
// fast path is AMDGPUCodeGenPrepare::expandDivRem24Impl and
// AMDGPUTargetLowering::LowerDIVREM24.

// 2^32 - 512 as an f32. It keeps the estimate below 2^32 / divisor for a
// reciprocal accurate to 1 ulp, which the Newton step and the corrections rely
// on.
static const InstructionInput reciprocalScale = std::string("0x4f7ffffe");

// The helpers below pick the SALU or VALU instruction from the kind of dst.

static void createXor(IRBuilder &builder, RegisterRef dst, Operand a,
                      Operand b) {
  if (dst.kind == RegisterKind::sgpr)
    builder.create<rocisa::SXorB32>(dst.container(), a, b);
  else
    builder.create<rocisa::VXorB32>(dst.container(), a, b);
}

static void createAShr(IRBuilder &builder, RegisterRef dst, Operand src,
                       int32_t shift) {
  if (dst.kind == RegisterKind::sgpr)
    builder.create<rocisa::SAShiftRightI32>(dst.container(), shift, src);
  else
    builder.create<rocisa::VAShiftRightI32>(dst.container(), shift, src, "");
}

static void createMulHi(IRBuilder &builder, RegisterRef dst, Operand a,
                        Operand b) {
  if (dst.kind == RegisterKind::sgpr)
    builder.create<rocisa::SMulHIU32>(dst.container(), a, b);
  else
    builder.create<rocisa::VMulHIU32>(dst.container(), a, b);
}

// Readfirstlane is undefined with exec == 0, hence the documented precondition
// of the SALU form. The caller covers the VALU-write hazard on src.
static RegisterRange readFirstLane(IRBuilder &builder, RegisterRef src) {
  RegisterRange value = builder.allocateSgpr(1);
  builder.create<rocisa::VReadfirstlaneB32>(value.container(), src.container());
  return value;
}

// |INT32_MIN| is 2^31 as an unsigned value, which the unsigned core handles.
static RegisterRange absoluteValue(IRBuilder &builder, RegisterKind kind,
                                   Operand x) {
  RegisterRange magnitude = allocate(builder, kind);
  if (kind == RegisterKind::sgpr) {
    builder.create<rocisa::SAbsI32>(magnitude.container(), x.reg().container());
    return magnitude;
  }
  builder.subi32(magnitude.ref(), 0, x);
  // x in src0 keeps the VGPR in src1.
  builder.create<rocisa::VMaxI32>(magnitude.container(), x,
                                  magnitude.container());
  return magnitude;
}

// Returns x ~= 2^32 / divisor after one Newton step, in a register of kind.
static RegisterRange reciprocalEstimate(IRBuilder &builder, RegisterKind kind,
                                        RegisterRef divisor) {
  RegisterRange negDivisor = allocate(builder, kind);
  // The estimate starts as the divisor as an f32, its reciprocal scaled by
  // 2^32 - 512, converted back to u32, all in one VGPR.
  RegisterRange x = builder.allocateVgpr(1);
  builder.create<rocisa::VCvtU32toF32>(x.container(), divisor.container());
  builder.create<rocisa::VRcpIFlagF32>(x.container(), x.container());
  // The one independent instruction between the reciprocal and its reader
  // covers the transcendental result hazard.
  static_assert(transResultWaitStates == 1,
                "one instruction no longer covers the reciprocal hazard");
  builder.subi32(negDivisor.ref(), 0, divisor);
  builder.create<rocisa::VMulF32>(x.container(), reciprocalScale,
                                  x.container());
  builder.create<rocisa::VCvtF32toU32>(x.container(), x.container());
  if (kind == RegisterKind::sgpr) {
    builder.create<rocisa::SNop>(valuWriteVgprReadlaneWaitStates - 1);
    x = readFirstLane(builder, x.ref());
  }
  RegisterRange t = allocate(builder, kind);
  builder.muli32(t.ref(), negDivisor, x);
  createMulHi(builder, t.ref(), x, t);
  builder.addi32(x.ref(), x, t);
  return x;
}

// One LLVM correction step: if r >= divisor then q += 1 and r -= divisor.
static void correctQuotient(IRBuilder &builder, RegisterRef q, RegisterRef r,
                            RegisterRef divisor, bool updateRemainder) {
  // The scalar add and sub write SCC, so they precede the compare.
  RegisterRange next = allocate(builder, q.kind);
  builder.addi32(next.ref(), q, 1);
  std::optional<RegisterRange> reduced;
  if (updateRemainder) {
    reduced.emplace(allocate(builder, q.kind));
    builder.subi32(reduced->ref(), r, divisor);
  }
  if (q.kind == RegisterKind::sgpr) {
    builder.create<rocisa::SCmpGeU32>(r.container(), divisor.container());
    builder.create<rocisa::SCSelectB32>(q.container(), next.container(),
                                        q.container());
    if (updateRemainder)
      builder.create<rocisa::SCSelectB32>(r.container(), reduced->container(),
                                          r.container());
    return;
  }
  RegisterRange mask = builder.allocateSgpr(2, 2);
  builder.cmpi32(Predicate::uge, mask.ref(), r, divisor);
  builder.selectb32(q, mask.ref(), next, q);
  if (updateRemainder)
    builder.selectb32(r, mask.ref(), *reduced, r);
}

static void emitExact(IRBuilder &builder, RegisterRef dst, Operand a,
                      Operand b) {
  RegisterKind kind = dst.kind;
  Temporaries temps;
  if (kind == RegisterKind::vgpr) {
    legalizeVop3Sources(builder, a, b, temps);
  } else {
    // s_abs_i32 takes only registers.
    for (Operand *source : {&a, &b})
      if (source->isImmediate())
        *source =
            copyToTemporary(builder, RegisterKind::sgpr, 1, *source, temps);
  }
  // a and b are read only here, before dst is written, so dst may alias them.
  RegisterRange divisor = absoluteValue(builder, kind, b);
  RegisterRange dividend = absoluteValue(builder, kind, a);
  // 0 when the signs agree and -1 when they differ.
  RegisterRange sign = allocate(builder, kind);
  createXor(builder, sign.ref(), a, b);
  createAShr(builder, sign.ref(), sign, 31);
  RegisterRange x = reciprocalEstimate(builder, kind, divisor.ref());
  RegisterRange q = allocate(builder, kind);
  createMulHi(builder, q.ref(), dividend, x);
  RegisterRange r = allocate(builder, kind);
  builder.muli32(r.ref(), q, divisor);
  builder.subi32(r.ref(), dividend, r);
  correctQuotient(builder, q.ref(), r.ref(), divisor.ref(), true);
  // LLVM's second step updates only the quotient.
  correctQuotient(builder, q.ref(), r.ref(), divisor.ref(), false);
  // (q ^ sign) - sign negates q when the signs differ.
  createXor(builder, q.ref(), q, sign);
  builder.subi32(dst, q, sign);
}

namespace {

struct FastEstimate {
  // A VGPR holding the truncated f32 quotient as an integer.
  RegisterRange quotient;
  // An SGPR pair set for the lanes whose quotient needs the +-1 correction.
  RegisterRange needsCorrection;
};

} // namespace

// Always VALU, since the division runs in f32.
static FastEstimate estimateQuotient(IRBuilder &builder, Operand a, Operand b) {
  RegisterRange fb = builder.allocateVgpr(1);
  RegisterRange fq = builder.allocateVgpr(1);
  RegisterRange fa = builder.allocateVgpr(1);
  RegisterRange quotient = builder.allocateVgpr(1);
  RegisterRange mask = builder.allocateSgpr(2, 2);
  builder.create<rocisa::VCvtI32toF32>(fb.container(), b);
  builder.create<rocisa::VRcpIFlagF32>(fq.container(), fb.container());
  // Independent of the reciprocal, so it gives the result its wait state.
  builder.create<rocisa::VCvtI32toF32>(fa.container(), a);
  builder.create<rocisa::VMulF32>(fq.container(), fa.container(),
                                  fq.container());
  // Truncates toward zero.
  builder.create<rocisa::VCvtF32toI32>(quotient.container(), fq.container());
  // LLVM's v_trunc_f32, which rocisa lacks; the round trip is exact for
  // magnitudes below 2^31.
  builder.create<rocisa::VCvtI32toF32>(fq.container(), quotient.container());
  // The remainder fa - fq * fb in one rounding, as LLVM does on gfx950.
  std::shared_ptr<rocisa::RegisterContainer> negQuotient = fq.container();
  negQuotient->setMinus(true);
  builder.create<rocisa::VFmaF32>(fa.container(), negQuotient, fb.container(),
                                  fa.container());
  std::shared_ptr<rocisa::RegisterContainer> absRemainder = fa.container();
  std::shared_ptr<rocisa::RegisterContainer> absDivisor = fb.container();
  absRemainder->setAbs(true);
  absDivisor->setAbs(true);
  builder.create<rocisa::VCmpGEF32>(mask.container(), absRemainder, absDivisor);
  return {std::move(quotient), std::move(mask)};
}

// This is LLVM's signed 24-bit criterion: expandDivRem24 needs at least 9 sign
// bits, that is values in [-2^23, 2^23). Beyond that, fa * rcp(fb) can round up
// to the next integer, and the single correction only fixes an estimate that
// is too small.
static void emitFast(IRBuilder &builder, RegisterRef dst, Operand a,
                     Operand b) {
  RegisterKind kind = dst.kind;
  Temporaries temps;
  // In the scalar case only s_xor reads both sources, and the VOP1 conversions
  // take any scalar source.
  if (kind == RegisterKind::vgpr)
    legalizeVop3Sources(builder, a, b, temps);
  else if (!isLegalSop2(a, b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 1, b, temps);
  // a and b are read only here, before dst is written, so dst may alias them.
  // LLVM computes ashr (a ^ b), bitWidth - 2 and then ors 1, giving +1 or -1.
  RegisterRange sign = allocate(builder, kind);
  createXor(builder, sign.ref(), a, b);
  createAShr(builder, sign.ref(), sign, 30);
  // The inline constant in src0 keeps the VOP2 form legal.
  if (kind == RegisterKind::sgpr)
    builder.create<rocisa::SOrB32>(sign.container(), Operand(sign), 1);
  else
    builder.create<rocisa::VOrB32>(sign.container(), 1, Operand(sign));
  FastEstimate estimate = estimateQuotient(builder, a, b);
  if (kind == RegisterKind::sgpr) {
    // Three VALU instructions separate the conversion from this read, more
    // than the one wait state the readfirstlane hazard needs.
    RegisterRange q = readFirstLane(builder, estimate.quotient.ref());
    // How LLVM tests a uniform mask; it reads exec and never writes it.
    RegisterRef mask = estimate.needsCorrection.ref();
    builder.create<rocisa::SAndB64>(mask.container(), mask.container(),
                                    std::make_shared<rocisa::EXEC>());
    RegisterRange correction = builder.allocateSgpr(1);
    builder.create<rocisa::SCSelectB32>(correction.container(),
                                        sign.container(), 0);
    builder.addi32(dst, q, correction);
    return;
  }
  // The mask is written by a VALU instruction and read by the cndmask.
  emitValuSgprWriteNop(builder);
  RegisterRange correction = builder.allocateVgpr(1);
  builder.selectb32(correction.ref(), estimate.needsCorrection.ref(), sign, 0);
  builder.addi32(dst, estimate.quotient, correction);
}

void IRBuilder::divsi32(RegisterRef dst, Operand a, Operand b, bool useIEEE) {
  requireArithmetic("divsi32", dst, a, b, 1);
  if (useIEEE)
    return emitExact(*this, dst, a, b);
  emitFast(*this, dst, a, b);
}

// Compare.

static void requireCompare(std::string_view opName, int32_t width,
                           RegisterRef dst, Operand a, Operand b) {
  bool uniformBool = dst.kind == RegisterKind::sgpr && dst.count == 1;
  if (!uniformBool && !isLaneMask(dst))
    throw std::invalid_argument(std::string(opName) +
                                " needs a destination of 1 sgpr or an sgpr "
                                "pair at an even index, got " +
                                registerText(dst));
  requireOperands(opName, a, b, width);
  if (dst.count != 1)
    return;
  for (Operand source : {a, b}) {
    if (!source.isRegister(RegisterKind::vgpr))
      continue;
    throw std::invalid_argument(
        std::string(opName) + " uniform bool destination " + registerText(dst) +
        " needs scalar operands, got " + registerText(source.reg()));
  }
}

// Writes the comparison result held in SCC to dst. An inverted result swaps
// the two values.
static void materialize(IRBuilder &builder, RegisterRef dst, bool invert) {
  if (dst.count == 1)
    builder.create<rocisa::SCSelectB32>(dst.container(), invert ? 0 : 1,
                                        invert ? 1 : 0);
  else
    builder.create<rocisa::SCSelectB64>(dst.container(), invert ? 0 : -1,
                                        invert ? -1 : 0);
}

static void createScalarCmp32(IRBuilder &builder, Predicate pred, Operand a,
                              Operand b) {
  switch (pred) {
  case Predicate::eq:
    builder.create<rocisa::SCmpEQU32>(a, b);
    return;
  case Predicate::ne:
    builder.create<rocisa::SCmpLgU32>(a, b);
    return;
  case Predicate::slt:
    builder.create<rocisa::SCmpLtI32>(a, b);
    return;
  case Predicate::sle:
    builder.create<rocisa::SCmpLeI32>(a, b);
    return;
  case Predicate::sgt:
    builder.create<rocisa::SCmpGtI32>(a, b);
    return;
  case Predicate::sge:
    builder.create<rocisa::SCmpGeI32>(a, b);
    return;
  case Predicate::ult:
    builder.create<rocisa::SCmpLtU32>(a, b);
    return;
  case Predicate::ule:
    builder.create<rocisa::SCmpLeU32>(a, b);
    return;
  case Predicate::ugt:
    builder.create<rocisa::SCmpGtU32>(a, b);
    return;
  case Predicate::uge:
    builder.create<rocisa::SCmpGeU32>(a, b);
    return;
  }
}

static void createVectorCmp32(IRBuilder &builder, Predicate pred,
                              RegisterRef dst, Operand a, Operand b) {
  std::shared_ptr<rocisa::RegisterContainer> mask = dst.container();
  switch (pred) {
  case Predicate::eq:
    builder.create<rocisa::VCmpEQU32>(mask, a, b);
    return;
  case Predicate::ne:
    builder.create<rocisa::VCmpNeU32>(mask, a, b);
    return;
  case Predicate::slt:
    builder.create<rocisa::VCmpLtI32>(mask, a, b);
    return;
  case Predicate::sle:
    builder.create<rocisa::VCmpLeI32>(mask, a, b);
    return;
  case Predicate::sgt:
    builder.create<rocisa::VCmpGTI32>(mask, a, b);
    return;
  case Predicate::sge:
    builder.create<rocisa::VCmpGEI32>(mask, a, b);
    return;
  case Predicate::ult:
    builder.create<rocisa::VCmpLtU32>(mask, a, b);
    return;
  case Predicate::ule:
    builder.create<rocisa::VCmpLeU32>(mask, a, b);
    return;
  case Predicate::ugt:
    builder.create<rocisa::VCmpGtU32>(mask, a, b);
    return;
  case Predicate::uge:
    builder.create<rocisa::VCmpGEU32>(mask, a, b);
    return;
  }
}

static void createVectorCmp64(IRBuilder &builder, Predicate pred,
                              RegisterRef dst, Operand a, Operand b) {
  std::shared_ptr<rocisa::RegisterContainer> mask = dst.container();
  switch (pred) {
  case Predicate::eq:
    builder.create<rocisa::VCmpEQU64>(mask, a, b);
    return;
  case Predicate::ne:
    builder.create<rocisa::VCmpNeU64>(mask, a, b);
    return;
  case Predicate::slt:
    builder.create<rocisa::VCmpLtI64>(mask, a, b);
    return;
  case Predicate::sle:
    builder.create<rocisa::VCmpLeI64>(mask, a, b);
    return;
  case Predicate::sgt:
    builder.create<rocisa::VCmpGTI64>(mask, a, b);
    return;
  case Predicate::sge:
    builder.create<rocisa::VCmpGEI64>(mask, a, b);
    return;
  case Predicate::ult:
    builder.create<rocisa::VCmpLtU64>(mask, a, b);
    return;
  case Predicate::ule:
    builder.create<rocisa::VCmpLeU64>(mask, a, b);
    return;
  case Predicate::ugt:
    builder.create<rocisa::VCmpGtU64>(mask, a, b);
    return;
  case Predicate::uge:
    builder.create<rocisa::VCmpGEU64>(mask, a, b);
    return;
  }
}

static void emitScalarEquality64(IRBuilder &builder, Predicate pred,
                                 RegisterRef dst, Operand a, Operand b) {
  Temporaries temps;
  legalizeSalu64Sources(builder, a, b, temps);
  if (pred == Predicate::eq)
    builder.create<rocisa::SCmpEQU64>(a, b);
  else
    builder.create<rocisa::SCmpLgU64>(a, b);
  materialize(builder, dst, /*invert=*/false);
}

// Flipping the sign bit turns a signed comparison into an unsigned one.
// s_xor_b32 writes SCC, so it must precede the borrow chain.
static Operand biasHigh(IRBuilder &builder, Operand high, Temporaries &temps) {
  if (high.isImmediate()) {
    return Operand(static_cast<int32_t>(
        static_cast<uint32_t>(high.immediate()) ^ 0x80000000u));
  }
  RegisterRange biased = builder.allocateSgpr(1);
  builder.create<rocisa::SXorB32>(biased.container(), high, Operand(INT32_MIN));
  temps.push_back(std::move(biased));
  return Operand(temps.back());
}

// dst is written only by the final s_cselect, after every operand was read, so
// it may alias a or b. The predicate is never eq or ne.
static void emitScalarOrdered64(IRBuilder &builder, Predicate pred,
                                RegisterRef dst, Operand a, Operand b) {
  // How an ordered predicate maps onto the borrow of x - y, where SCC is x <u
  // y.
  bool swap = pred == Predicate::sgt || pred == Predicate::sle ||
              pred == Predicate::ugt || pred == Predicate::ule;
  bool invert = pred == Predicate::sge || pred == Predicate::sle ||
                pred == Predicate::uge || pred == Predicate::ule;
  bool isSigned = pred == Predicate::slt || pred == Predicate::sge ||
                  pred == Predicate::sgt || pred == Predicate::sle;
  Operand x = swap ? b : a;
  Operand y = swap ? a : b;
  Temporaries temps;
  // Two immediates would put two distinct literals in one instruction.
  if (x.isImmediate() && y.isImmediate())
    y = copyToTemporary(builder, RegisterKind::sgpr, 2, y, temps);
  Operand xHigh = operandHalf(x, 1);
  Operand yHigh = operandHalf(y, 1);
  if (isSigned) {
    xHigh = biasHigh(builder, xHigh, temps);
    yHigh = biasHigh(builder, yHigh, temps);
  }
  // SCC ends up as x <u y; the difference itself is discarded.
  RegisterRange diff = builder.allocateSgpr(1);
  builder.create<rocisa::SSubU32>(diff.container(), operandHalf(x, 0),
                                  operandHalf(y, 0));
  builder.create<rocisa::SSubBU32>(diff.container(), xHigh, yHigh);
  materialize(builder, dst, invert);
}

static void emitCompare(IRBuilder &builder, int32_t width, Predicate pred,
                        RegisterRef dst, Operand a, Operand b) {
  bool scalar =
      !a.isRegister(RegisterKind::vgpr) && !b.isRegister(RegisterKind::vgpr);
  if (scalar && width == 2 && (pred == Predicate::eq || pred == Predicate::ne))
    return emitScalarEquality64(builder, pred, dst, a, b);
  if (scalar && width == 2)
    return emitScalarOrdered64(builder, pred, dst, a, b);
  Temporaries temps;
  if (scalar) {
    if (!isLegalSop2(a, b))
      b = copyToTemporary(builder, RegisterKind::sgpr, 1, b, temps);
    createScalarCmp32(builder, pred, a, b);
    materialize(builder, dst, /*invert=*/false);
    return;
  }
  if (width == 1) {
    legalizeVop3Sources(builder, a, b, temps);
    createVectorCmp32(builder, pred, dst, a, b);
  } else {
    if (!isLegalVop3Source64(a))
      a = copyToTemporary(builder, RegisterKind::vgpr, 2, a, temps);
    if (!isLegalVop3Source64(b))
      b = copyToTemporary(builder, RegisterKind::vgpr, 2, b, temps);
    // A vector operand is present, so at most one operand is an SGPR pair.
    assert(isLegalVop3(a, b) && "illegal vop3 operands");
    createVectorCmp64(builder, pred, dst, a, b);
  }
  // The mask is written by a VALU instruction, so the nop lets any following
  // instruction read it.
  emitValuSgprWriteNop(builder);
}

void IRBuilder::cmpi32(Predicate pred, RegisterRef dst, Operand a, Operand b) {
  requireCompare("cmpi32", 1, dst, a, b);
  emitCompare(*this, 1, pred, dst, a, b);
}

void IRBuilder::cmpi64(Predicate pred, RegisterRef dst, Operand a, Operand b) {
  requireCompare("cmpi64", 2, dst, a, b);
  emitCompare(*this, 2, pred, dst, a, b);
}

// Select.

namespace {

// The sources of one 32-bit word of the result: a is chosen when the condition
// holds, b otherwise.
struct WordSources {
  Operand a;
  Operand b;
};

// The low word, then the high word of a pair; one word for a single register.
using Words = std::vector<WordSources>;

} // namespace

static void requireSelect(std::string_view opName, int32_t width,
                          RegisterRef dst, RegisterRef cond, Operand a,
                          Operand b) {
  requireRegister(opName, dst, width);
  requireOperands(opName, a, b, width);
  bool validCondition =
      cond.kind == RegisterKind::sgpr && (cond.count == 1 || isLaneMask(cond));
  if (!validCondition)
    throw std::invalid_argument(
        std::string(opName) +
        " condition must be an sgpr or an sgpr pair at an even index, got " +
        registerText(cond));
  if (dst.kind != RegisterKind::sgpr)
    return;
  if (cond.count != 1)
    throw std::invalid_argument(
        std::string(opName) + " sgpr destination " + registerText(dst) +
        " needs a uniform bool condition, got " + registerText(cond));
  requireScalarSources(opName, dst, a, b);
}

static Words wordSources(int32_t width, Operand a, Operand b) {
  if (width == 1)
    return Words{WordSources{a, b}};
  return Words{WordSources{operandHalf(a, 0), operandHalf(b, 0)},
               WordSources{operandHalf(a, 1), operandHalf(b, 1)}};
}

// Writing a half of dst first clobbers that register, which the step that
// writes the other half must not read.
static bool isSafeOrder(RegisterRef dst, const Words &words, bool highFirst) {
  int32_t first = highFirst ? 1 : 0;
  const WordSources &other = words[1 - first];
  for (Operand source : {other.a, other.b})
    if (source.isRegister() && source.reg() == dst.slice(first))
      return false;
  return true;
}

// Returns whether the high half is written first. When neither order is safe,
// the high step's copy of dst's low register moves to a temporary of dst's kind
// so that low-first works. Callers emit this before reading the condition into
// SCC, since the copy is a plain move.
static bool orderHalves(IRBuilder &builder, RegisterRef dst, Words &words,
                        Temporaries &temps) {
  if (words.size() == 1 || isSafeOrder(dst, words, false))
    return false;
  if (isSafeOrder(dst, words, true))
    return true;
  RegisterRef clobbered = dst.slice(0);
  for (Operand *word : {&words[1].a, &words[1].b})
    if (word->isRegister() && word->reg() == clobbered)
      *word = copyToTemporary(builder, dst.kind, 1, *word, temps);
  assert(isSafeOrder(dst, words, false) && "unsafe half order");
  return false;
}

// Visits the words in write order, passing each with its destination register.
template <typename Emit>
static void emitWords(RegisterRef dst, const Words &words, bool highFirst,
                      Emit emit) {
  int32_t count = static_cast<int32_t>(words.size());
  for (int32_t i = 0; i < count; ++i) {
    int32_t index = highFirst ? count - 1 - i : i;
    emit(dst.slice(index), words[index]);
  }
}

// Sets SCC to whether the uniform bool is nonzero.
static void testCondition(IRBuilder &builder, RegisterRef cond) {
  builder.create<rocisa::SCmpLgU32>(cond.container(), 0);
}

// The mask is read by the cndmasks only, so it cannot alias dst, a VGPR.
static void emitVectorSelect(IRBuilder &builder, RegisterRef dst,
                             RegisterRef cond, Operand a, Operand b) {
  Temporaries temps;
  // A uniform bool widens to a temporary mask; a SALU write followed by a VALU
  // read needs no s_nop.
  RegisterRef mask = cond;
  if (cond.count != 2) {
    RegisterRange widened = builder.allocateSgpr(2, 2);
    testCondition(builder, cond);
    builder.create<rocisa::SCSelectB64>(widened.container(), -1, 0);
    temps.push_back(std::move(widened));
    mask = temps.back().ref();
  }
  Words words = wordSources(dst.count, a, b);
  // v_cndmask_b32 takes only VGPRs and inline constants, since the mask already
  // uses the constant bus.
  for (WordSources &word : words)
    for (Operand *source : {&word.a, &word.b})
      if (!isVgprOrInlineConstant(*source))
        *source =
            copyToTemporary(builder, RegisterKind::vgpr, 1, *source, temps);
  bool highFirst = orderHalves(builder, dst, words, temps);
  emitWords(dst, words, highFirst, [&](RegisterRef part, WordSources word) {
    // Operand order is src0 = b, src1 = a: a mask bit set selects src1.
    builder.create<rocisa::VCndMaskB32>(part.container(), word.b, word.a,
                                        mask.container());
  });
}

// Emits dst = cond ? a : b for a single register or a pair.
static void emitSelect(IRBuilder &builder, RegisterRef dst, RegisterRef cond,
                       Operand a, Operand b) {
  // The operand checks already rejected agpr destinations.
  if (dst.kind == RegisterKind::vgpr)
    return emitVectorSelect(builder, dst, cond, a, b);
  Temporaries temps;
  if (dst.count == 2 && dst.start % 2 == 0) {
    // The aligned pair is written by one instruction, so every operand is read
    // before dst is written and aliasing is safe.
    legalizeSalu64Sources(builder, a, b, temps);
    testCondition(builder, cond);
    builder.create<rocisa::SCSelectB64>(dst.container(), a, b);
    return;
  }
  // One select per 32-bit word, for a single register or a misaligned pair. The
  // condition is read into SCC before any word is written and s_cselect leaves
  // SCC alone.
  Words words = wordSources(dst.count, a, b);
  for (WordSources &word : words)
    if (!isLegalSop2(word.a, word.b))
      word.b = copyToTemporary(builder, RegisterKind::sgpr, 1, word.b, temps);
  bool highFirst = orderHalves(builder, dst, words, temps);
  testCondition(builder, cond);
  emitWords(dst, words, highFirst, [&](RegisterRef part, WordSources word) {
    builder.create<rocisa::SCSelectB32>(part.container(), word.a, word.b);
  });
}

void IRBuilder::selectb32(RegisterRef dst, RegisterRef cond, Operand a,
                          Operand b) {
  requireSelect("selectb32", 1, dst, cond, a, b);
  emitSelect(*this, dst, cond, a, b);
}

void IRBuilder::selectb64(RegisterRef dst, RegisterRef cond, Operand a,
                          Operand b) {
  requireSelect("selectb64", 2, dst, cond, a, b);
  emitSelect(*this, dst, cond, a, b);
}

// Buffer resource.

static void requireRsrcOperands(RegisterRef dst, Operand pointer,
                                Operand sizeBytes) {
  if (dst.kind != RegisterKind::sgpr || dst.count != 4)
    throw std::invalid_argument(
        "makeRsrc needs a destination of 4 sgprs, got " + registerText(dst));
  if (dst.start % 4 != 0)
    throw std::invalid_argument(
        "makeRsrc destination " + registerText(dst) +
        " must start at an sgpr index that is a multiple of 4");

  std::string prefix = "makeRsrc pointer must be an sgpr ";
  if (pointer.isImmediate())
    throw std::invalid_argument(prefix + "pair, got immediate " +
                                std::to_string(pointer.immediate()));
  RegisterRef pointerReg = pointer.reg();
  if (pointerReg.kind != RegisterKind::sgpr || pointerReg.count != 2)
    throw std::invalid_argument(prefix + "pair, got " +
                                registerText(pointerReg));

  // Any int32 is the 32-bit pattern of an unsigned size.
  if (sizeBytes.isImmediate())
    return;
  RegisterRef sizeReg = sizeBytes.reg();
  if (sizeReg.kind != RegisterKind::sgpr || sizeReg.count != 1)
    throw std::invalid_argument(
        "makeRsrc size must be a single sgpr or an immediate, got " +
        registerText(sizeReg));
}

static bool isSelfCopy(RegisterRef dstPart, Operand src) {
  return src.isRegister() && src.reg() == dstPart;
}

void IRBuilder::makeRsrc(RegisterRef dst, Operand pointer, Operand sizeBytes) {
  requireRsrcOperands(dst, pointer, sizeBytes);

  RegisterRef pointerDst{RegisterKind::sgpr, dst.start, 2};
  RegisterRef sizeDst{RegisterKind::sgpr, dst.start + 2, 1};
  RegisterRef upperDst{RegisterKind::sgpr, dst.start + 3, 1};

  // Copying the pointer overwrites dst[0:1], so a size held there must be saved
  // before that happens. The temporary must outlive the size word's mov below.
  std::optional<RegisterRange> sizeCopy;
  bool sizeIsClobbered = sizeBytes.isRegister() &&
                         !isSelfCopy(pointerDst, pointer) &&
                         (sizeBytes.reg().start == dst.start ||
                          sizeBytes.reg().start == dst.start + 1);
  if (sizeIsClobbered) {
    sizeCopy.emplace(allocateSgpr(1));
    movb32(sizeCopy->ref(), sizeBytes);
    sizeBytes = Operand(*sizeCopy);
  }

  // Each source is read by exactly one copy and word 3 has no register source,
  // so this order is safe for any remaining overlap with dst.
  if (!isSelfCopy(pointerDst, pointer))
    movb64(pointerDst, pointer);
  if (!isSelfCopy(sizeDst, sizeBytes))
    movb32(sizeDst, sizeBytes);
  int32_t upper = rocisa::SrdUpperValue(target().isaVersion())->getValue();
  movb32(upperDst, upper);
}
