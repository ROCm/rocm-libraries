// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "Gfx9Legality.hpp"
#include "OperandUtils.hpp"

#include "container.hpp"
#include "instruction/common.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using namespace rocm::tensilelite::codegen;
using detail::copyToTemporary;
using detail::legalizeVop3Sources;
using detail::Temporaries;

namespace {

enum class BinaryOp { add, sub, mul };

std::string_view opName(BinaryOp op, int width) {
  bool wide = width == 2;
  switch (op) {
  case BinaryOp::add:
    return wide ? "addi64" : "addi32";
  case BinaryOp::sub:
    return wide ? "subi64" : "subi32";
  case BinaryOp::mul:
    return wide ? "muli64" : "muli32";
  }
  throw std::invalid_argument("invalid binary op");
}

void createSalu(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                Operand b) {
  switch (op) {
  case BinaryOp::add:
    builder.create<rocisa::SAddU32>(dst.container(), a, b);
    return;
  case BinaryOp::sub:
    builder.create<rocisa::SSubU32>(dst.container(), a, b);
    return;
  case BinaryOp::mul:
    builder.create<rocisa::SMulI32>(dst.container(), a, b);
    return;
  }
}

void createValu(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                Operand b) {
  switch (op) {
  case BinaryOp::add:
    builder.create<rocisa::VAddU32>(dst.container(), a, b);
    return;
  case BinaryOp::sub:
    builder.create<rocisa::VSubU32>(dst.container(), a, b);
    return;
  case BinaryOp::mul:
    builder.create<rocisa::VMulLOU32>(dst.container(), a, b);
    return;
  }
}

void emitSalu(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
              Operand b) {
  Temporaries temps;
  // Only two distinct literals are illegal, and one of them moves to a copy.
  if (!gfx9::isLegalSop2(a, b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 1, b, temps);
  assert(gfx9::isLegalSop2(a, b) && "illegal salu operands");
  createSalu(builder, op, dst, a, b);
}

// v_mul_lo_u32 is VOP3 only, so it takes no literals.
void emitVop3Only(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                  Operand b) {
  Temporaries temps;
  legalizeVop3Sources(builder, a, b, temps);
  assert(gfx9::isLegalVop3(a, b) && "illegal vop3 operands");
  createValu(builder, op, dst, a, b);
}

// x - c is x + (-c) modulo 2^32, which needs no copy since VOP2 src0 accepts a
// literal and rocisa has no v_subrev.
void rewriteSubImmediate(BinaryOp &op, Operand &b) {
  if (op != BinaryOp::sub || !b.isImmediate())
    return;
  op = BinaryOp::add;
  b = Operand(static_cast<int32_t>(0u - static_cast<uint32_t>(b.immediate())));
}

void emitVop2(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
              Operand b) {
  rewriteSubImmediate(op, b);
  bool commutative = op == BinaryOp::add;
  // A VGPR in src1 gives VOP2, whose src0 accepts anything.
  if (commutative && !b.isRegister(RegisterKind::vgpr) &&
      a.isRegister(RegisterKind::vgpr))
    std::swap(a, b);
  Temporaries temps;
  if (!gfx9::isLegalVop2(a, b) && !gfx9::isLegalVop3(a, b))
    // A VGPR in src1 makes any src0 legal.
    b = copyToTemporary(builder, RegisterKind::vgpr, 1, b, temps);
  assert((gfx9::isLegalVop2(a, b) || gfx9::isLegalVop3(a, b)) &&
         "illegal valu operands");
  createValu(builder, op, dst, a, b);
}

void emitBinaryI32(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                   Operand b) {
  detail::requireRegister(opName(op, 1), dst, 1);
  detail::requireOperand(opName(op, 1), a, 1);
  detail::requireOperand(opName(op, 1), b, 1);
  detail::requireScalarSources(opName(op, 1), dst, a, b);
  // Copies go into fresh temporaries before the single op, which reads all its
  // sources before writing dst, so dst may alias a or b.
  if (dst.kind == RegisterKind::sgpr)
    return emitSalu(builder, op, dst, a, b);
  if (op == BinaryOp::mul)
    return emitVop3Only(builder, op, dst, a, b);
  emitVop2(builder, op, dst, a, b);
}

// The 32-bit halves of a 64-bit operand.
struct Halves {
  Operand lo;
  Operand hi;
};

Halves halvesOf(Operand operand) {
  return {detail::operandHalf(operand, 0), detail::operandHalf(operand, 1)};
}

// Two halves that one instruction reads together.
struct HalfPair {
  Operand *first;
  Operand *second;
};

// The carry chains write dst's low register before they read the high halves,
// so a high half that is that register, possible only for overlapping pairs,
// is copied first.
void protectHighHalves(IRBuilder &builder, RegisterRef dst, Halves &a,
                       Halves &b, Temporaries &temps) {
  for (Operand *half : {&a.hi, &b.hi}) {
    if (!half->isRegister() || half->reg() != dst.slice(0))
      continue;
    *half = copyToTemporary(builder, half->reg().kind, 1, *half, temps);
  }
}

void emitSaluCarryChain(IRBuilder &builder, BinaryOp op, RegisterRef dst,
                        Halves a, Halves b) {
  Temporaries temps;
  for (HalfPair pair : {HalfPair{&a.lo, &b.lo}, HalfPair{&a.hi, &b.hi}})
    if (!gfx9::isLegalSop2(*pair.first, *pair.second))
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

void emitValuCarryChain(IRBuilder &builder, BinaryOp op, RegisterRef dst,
                        Halves a, Halves b) {
  Temporaries temps;
  // The low half is VOP3 because its carry-out is an SGPR pair.
  legalizeVop3Sources(builder, a.lo, b.lo, temps);
  for (Operand *half : {&a.hi, &b.hi})
    if (!gfx9::isVgprOrInlineConstant(*half))
      *half = copyToTemporary(builder, RegisterKind::vgpr, 1, *half, temps);
  protectHighHalves(builder, dst, a, b, temps);
  assert(gfx9::isLegalVop3(a.lo, b.lo) && "illegal vop3 operands");
  assert(gfx9::isVgprOrInlineConstant(a.hi) &&
         gfx9::isVgprOrInlineConstant(b.hi) && "illegal carry-in operands");
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
  builder.create<rocisa::SNop>(gfx9::valuWriteSgprReadWaitStates - 1);
  if (isAdd)
    builder.create<rocisa::VAddCCOU32>(hi, carry.container(), a.hi, b.hi,
                                       carry.container());
  else
    builder.create<rocisa::VSubBCoU32>(hi, carry.container(), a.hi, b.hi,
                                       carry.container());
}

bool isLegalProductPair(RegisterKind kind, HalfPair pair) {
  if (kind == RegisterKind::vgpr)
    return gfx9::isLegalVop3(*pair.first, *pair.second);
  return gfx9::isLegalSop2(*pair.first, *pair.second);
}

void legalizeProductHalves(IRBuilder &builder, Halves &a, Halves &b,
                           RegisterKind kind, Temporaries &temps) {
  // VOP3 takes no literals.
  if (kind == RegisterKind::vgpr)
    for (Operand *half : {&a.lo, &a.hi, &b.lo, &b.hi})
      if (gfx9::isLiteral(*half))
        *half = copyToTemporary(builder, kind, 1, *half, temps);
  for (HalfPair pair : std::array<HalfPair, 3>{
           {{&a.lo, &b.lo}, {&a.lo, &b.hi}, {&a.hi, &b.lo}}})
    if (!isLegalProductPair(kind, pair))
      *pair.second = copyToTemporary(builder, kind, 1, *pair.second, temps);
}

void emitSaluMul64(IRBuilder &builder, RegisterRef dst, Halves a, Halves b) {
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
void emitValuMul64(IRBuilder &builder, RegisterRef dst, Halves a, Halves b) {
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

void emitBinaryI64(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                   Operand b) {
  detail::requireRegister(opName(op, 2), dst, 2);
  detail::requireOperand(opName(op, 2), a, 2);
  detail::requireOperand(opName(op, 2), b, 2);
  detail::requireScalarSources(opName(op, 2), dst, a, b);
  bool scalar = dst.kind == RegisterKind::sgpr;
  if (op == BinaryOp::mul && scalar)
    return emitSaluMul64(builder, dst, halvesOf(a), halvesOf(b));
  if (op == BinaryOp::mul)
    return emitValuMul64(builder, dst, halvesOf(a), halvesOf(b));
  if (scalar)
    return emitSaluCarryChain(builder, op, dst, halvesOf(a), halvesOf(b));
  emitValuCarryChain(builder, op, dst, halvesOf(a), halvesOf(b));
}

} // namespace

void IRBuilder::addi32(RegisterRef dst, Operand a, Operand b) {
  emitBinaryI32(*this, BinaryOp::add, dst, a, b);
}

void IRBuilder::subi32(RegisterRef dst, Operand a, Operand b) {
  emitBinaryI32(*this, BinaryOp::sub, dst, a, b);
}

void IRBuilder::muli32(RegisterRef dst, Operand a, Operand b) {
  emitBinaryI32(*this, BinaryOp::mul, dst, a, b);
}

void IRBuilder::addi64(RegisterRef dst, Operand a, Operand b) {
  emitBinaryI64(*this, BinaryOp::add, dst, a, b);
}

void IRBuilder::subi64(RegisterRef dst, Operand a, Operand b) {
  emitBinaryI64(*this, BinaryOp::sub, dst, a, b);
}

void IRBuilder::muli64(RegisterRef dst, Operand a, Operand b) {
  emitBinaryI64(*this, BinaryOp::mul, dst, a, b);
}
