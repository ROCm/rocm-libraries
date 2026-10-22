// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "Gfx9Legality.hpp"
#include "OperandUtils.hpp"

#include "container.hpp"
#include "instruction/common.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocm::tensilelite::codegen;
using detail::toInput;

namespace {

enum class BinaryOp { add, sub, mul };

// Temporary registers that stay allocated until the op has been emitted.
using Temporaries = std::vector<RegisterRange>;

std::string_view opName(BinaryOp op) {
  switch (op) {
  case BinaryOp::add:
    return "addi32";
  case BinaryOp::sub:
    return "subi32";
  case BinaryOp::mul:
    return "muli32";
  }
  throw std::invalid_argument("invalid binary op");
}

void requireRegister32(BinaryOp op, RegisterRef ref) {
  detail::requireNotAgpr(opName(op), ref);
  detail::requireRegisterCount(opName(op), ref, 1);
}

Operand canonicalize32(BinaryOp op, Operand operand) {
  if (operand.isRegister())
    requireRegister32(op, operand.reg());
  return detail::canonicalizeImmediate32(opName(op), operand);
}

void requireScalarSources(BinaryOp op, RegisterRef dst, Operand a, Operand b) {
  if (dst.kind != RegisterKind::sgpr)
    return;
  for (Operand source : {a, b}) {
    if (!source.isRegister(RegisterKind::vgpr))
      continue;
    throw std::invalid_argument(
        std::string(opName(op)) + " cannot write sgpr destination " +
        detail::registerText(dst) + " from vgpr operand " +
        detail::registerText(source.reg()));
  }
}

void createSalu(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                Operand b) {
  switch (op) {
  case BinaryOp::add:
    builder.create<rocisa::SAddU32>(dst.container(), toInput(a), toInput(b));
    return;
  case BinaryOp::sub:
    builder.create<rocisa::SSubU32>(dst.container(), toInput(a), toInput(b));
    return;
  case BinaryOp::mul:
    builder.create<rocisa::SMulI32>(dst.container(), toInput(a), toInput(b));
    return;
  }
}

void createValu(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                Operand b) {
  switch (op) {
  case BinaryOp::add:
    builder.create<rocisa::VAddU32>(dst.container(), toInput(a), toInput(b));
    return;
  case BinaryOp::sub:
    builder.create<rocisa::VSubU32>(dst.container(), toInput(a), toInput(b));
    return;
  case BinaryOp::mul:
    builder.create<rocisa::VMulLOU32>(dst.container(), toInput(a), toInput(b));
    return;
  }
}

// Copies source into a new VGPR that lives as long as temps does.
Operand copyToVgpr(IRBuilder &builder, Operand source, Temporaries &temps) {
  RegisterRange copy = builder.allocateVgpr(1);
  builder.movb32(copy.ref(), source);
  temps.push_back(std::move(copy));
  return Operand(temps.back());
}

// Copies source into a new SGPR that lives as long as temps does.
Operand copyToSgpr(IRBuilder &builder, Operand source, Temporaries &temps) {
  RegisterRange copy = builder.allocateSgpr(1);
  builder.movb32(copy.ref(), source);
  temps.push_back(std::move(copy));
  return Operand(temps.back());
}

void emitSalu(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
              Operand b) {
  Temporaries temps;
  // Only two distinct literals are illegal, and one of them moves to a copy.
  if (!gfx9::isLegalSop2(a, b))
    b = copyToSgpr(builder, b, temps);
  assert(gfx9::isLegalSop2(a, b) && "illegal salu operands");
  createSalu(builder, op, dst, a, b);
}

// v_mul_lo_u32 is VOP3 only, so it takes no literals.
void emitVop3Only(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                  Operand b) {
  Temporaries temps;
  if (gfx9::isLiteral(a))
    a = copyToVgpr(builder, a, temps);
  if (gfx9::isLiteral(b))
    b = copyToVgpr(builder, b, temps);
  // Two distinct SGPRs exceed the constant bus.
  if (!gfx9::isLegalVop3(a, b))
    b = copyToVgpr(builder, b, temps);
  assert(gfx9::isLegalVop3(a, b) && "illegal vop3 operands");
  createValu(builder, op, dst, a, b);
}

// x - c is x + (-c) modulo 2^32, which needs no copy since VOP2 src0 accepts a
// literal and rocisa has no v_subrev.
void rewriteSubImmediate(BinaryOp &op, Operand &b) {
  if (op != BinaryOp::sub || !b.isImmediate())
    return;
  op = BinaryOp::add;
  uint32_t negated = 0u - static_cast<uint32_t>(b.immediate());
  b = Operand(int64_t{static_cast<int32_t>(negated)});
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
    b = copyToVgpr(builder, b, temps);
  assert((gfx9::isLegalVop2(a, b) || gfx9::isLegalVop3(a, b)) &&
         "illegal valu operands");
  createValu(builder, op, dst, a, b);
}

void emitBinaryI32(IRBuilder &builder, BinaryOp op, RegisterRef dst, Operand a,
                   Operand b) {
  requireRegister32(op, dst);
  a = canonicalize32(op, a);
  b = canonicalize32(op, b);
  requireScalarSources(op, dst, a, b);
  // Copies go into fresh temporaries before the single op, which reads all its
  // sources before writing dst, so dst may alias a or b.
  if (dst.kind == RegisterKind::sgpr)
    return emitSalu(builder, op, dst, a, b);
  if (op == BinaryOp::mul)
    return emitVop3Only(builder, op, dst, a, b);
  emitVop2(builder, op, dst, a, b);
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
