// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "Gfx9Legality.hpp"
#include "OperandUtils.hpp"

#include "container.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using namespace rocm::tensilelite::codegen;
using detail::copyToTemporary;
using detail::registerText;
using detail::Temporaries;
using detail::toInput;
using detail::toInput64;

namespace {

constexpr std::string_view cmpi32Name = "cmpi32";
constexpr std::string_view cmpi64Name = "cmpi64";
constexpr int32_t signBit = INT32_MIN;

std::string_view opName(int width) {
  return width == 1 ? cmpi32Name : cmpi64Name;
}

void requireDestination(std::string_view op, RegisterRef dst) {
  bool uniformBool = dst.kind == RegisterKind::sgpr && dst.count == 1;
  if (uniformBool || gfx9::isLaneMask(dst))
    return;
  throw std::invalid_argument(
      std::string(op) +
      " needs a destination of 1 sgpr or an sgpr pair at an even index, got " +
      registerText(dst));
}

Operand validateOperand(std::string_view op, int width, Operand operand) {
  if (operand.isRegister()) {
    detail::requireNotAgpr(op, operand.reg());
    detail::requireRegisterCount(op, operand.reg(), width);
    return operand;
  }
  return width == 1 ? detail::canonicalizeImmediate32(op, operand) : operand;
}

bool isScalar(Operand a, Operand b) {
  return !a.isRegister(RegisterKind::vgpr) && !b.isRegister(RegisterKind::vgpr);
}

void requireScalarForUniformBool(std::string_view op, RegisterRef dst,
                                 Operand a, Operand b) {
  if (dst.count != 1)
    return;
  for (Operand source : {a, b}) {
    if (!source.isRegister(RegisterKind::vgpr))
      continue;
    throw std::invalid_argument(
        std::string(op) + " uniform bool destination " + registerText(dst) +
        " needs scalar operands, got " + registerText(source.reg()));
  }
}

// Writes the comparison result held in SCC to dst. An inverted result swaps
// the two values.
void materialize(IRBuilder &builder, RegisterRef dst, bool invert) {
  if (dst.count == 1)
    builder.create<rocisa::SCSelectB32>(dst.container(), invert ? 0 : 1,
                                        invert ? 1 : 0);
  else
    builder.create<rocisa::SCSelectB64>(dst.container(), invert ? 0 : -1,
                                        invert ? -1 : 0);
}

void createScalarCmp32(IRBuilder &builder, Predicate pred, Operand a,
                       Operand b) {
  InstructionInput x = toInput(a);
  InstructionInput y = toInput(b);
  switch (pred) {
  case Predicate::eq:
    builder.create<rocisa::SCmpEQU32>(x, y);
    return;
  case Predicate::ne:
    builder.create<rocisa::SCmpLgU32>(x, y);
    return;
  case Predicate::slt:
    builder.create<rocisa::SCmpLtI32>(x, y);
    return;
  case Predicate::sle:
    builder.create<rocisa::SCmpLeI32>(x, y);
    return;
  case Predicate::sgt:
    builder.create<rocisa::SCmpGtI32>(x, y);
    return;
  case Predicate::sge:
    builder.create<rocisa::SCmpGeI32>(x, y);
    return;
  case Predicate::ult:
    builder.create<rocisa::SCmpLtU32>(x, y);
    return;
  case Predicate::ule:
    builder.create<rocisa::SCmpLeU32>(x, y);
    return;
  case Predicate::ugt:
    builder.create<rocisa::SCmpGtU32>(x, y);
    return;
  case Predicate::uge:
    builder.create<rocisa::SCmpGeU32>(x, y);
    return;
  }
}

void createVectorCmp32(IRBuilder &builder, Predicate pred, RegisterRef dst,
                       Operand a, Operand b) {
  std::shared_ptr<rocisa::RegisterContainer> mask = dst.container();
  InstructionInput x = toInput(a);
  InstructionInput y = toInput(b);
  switch (pred) {
  case Predicate::eq:
    builder.create<rocisa::VCmpEQU32>(mask, x, y);
    return;
  case Predicate::ne:
    builder.create<rocisa::VCmpNeU32>(mask, x, y);
    return;
  case Predicate::slt:
    builder.create<rocisa::VCmpLtI32>(mask, x, y);
    return;
  case Predicate::sle:
    builder.create<rocisa::VCmpLeI32>(mask, x, y);
    return;
  case Predicate::sgt:
    builder.create<rocisa::VCmpGTI32>(mask, x, y);
    return;
  case Predicate::sge:
    builder.create<rocisa::VCmpGEI32>(mask, x, y);
    return;
  case Predicate::ult:
    builder.create<rocisa::VCmpLtU32>(mask, x, y);
    return;
  case Predicate::ule:
    builder.create<rocisa::VCmpLeU32>(mask, x, y);
    return;
  case Predicate::ugt:
    builder.create<rocisa::VCmpGtU32>(mask, x, y);
    return;
  case Predicate::uge:
    builder.create<rocisa::VCmpGEU32>(mask, x, y);
    return;
  }
}

void createVectorCmp64(IRBuilder &builder, Predicate pred, RegisterRef dst,
                       Operand a, Operand b) {
  std::shared_ptr<rocisa::RegisterContainer> mask = dst.container();
  InstructionInput x = toInput(a);
  InstructionInput y = toInput(b);
  switch (pred) {
  case Predicate::eq:
    builder.create<rocisa::VCmpEQU64>(mask, x, y);
    return;
  case Predicate::ne:
    builder.create<rocisa::VCmpNeU64>(mask, x, y);
    return;
  case Predicate::slt:
    builder.create<rocisa::VCmpLtI64>(mask, x, y);
    return;
  case Predicate::sle:
    builder.create<rocisa::VCmpLeI64>(mask, x, y);
    return;
  case Predicate::sgt:
    builder.create<rocisa::VCmpGTI64>(mask, x, y);
    return;
  case Predicate::sge:
    builder.create<rocisa::VCmpGEI64>(mask, x, y);
    return;
  case Predicate::ult:
    builder.create<rocisa::VCmpLtU64>(mask, x, y);
    return;
  case Predicate::ule:
    builder.create<rocisa::VCmpLeU64>(mask, x, y);
    return;
  case Predicate::ugt:
    builder.create<rocisa::VCmpGtU64>(mask, x, y);
    return;
  case Predicate::uge:
    builder.create<rocisa::VCmpGEU64>(mask, x, y);
    return;
  }
}

// The mask is written by a VALU instruction, so the nop lets any following
// instruction read it.
void createMaskHazardNop(IRBuilder &builder) {
  builder.create<rocisa::SNop>(gfx9::valuWriteSgprReadWaitStates - 1);
}

void emitVector32(IRBuilder &builder, Predicate pred, RegisterRef dst,
                  Operand a, Operand b) {
  Temporaries temps;
  if (gfx9::isLiteral(a))
    a = copyToTemporary(builder, RegisterKind::vgpr, 1, a, temps);
  if (gfx9::isLiteral(b))
    b = copyToTemporary(builder, RegisterKind::vgpr, 1, b, temps);
  assert(gfx9::isLegalVop3(a, b) && "illegal vop3 operands");
  createVectorCmp32(builder, pred, dst, a, b);
  createMaskHazardNop(builder);
}

void emitVector64(IRBuilder &builder, Predicate pred, RegisterRef dst,
                  Operand a, Operand b) {
  Temporaries temps;
  if (!gfx9::isLegalVop3Source64(a))
    a = copyToTemporary(builder, RegisterKind::vgpr, 2, a, temps);
  if (!gfx9::isLegalVop3Source64(b))
    b = copyToTemporary(builder, RegisterKind::vgpr, 2, b, temps);
  // A vector operand is present, so at most one operand is an SGPR pair.
  assert(gfx9::isLegalVop3(a, b) && "illegal vop3 operands");
  createVectorCmp64(builder, pred, dst, a, b);
  createMaskHazardNop(builder);
}

void emitScalar32(IRBuilder &builder, Predicate pred, RegisterRef dst,
                  Operand a, Operand b) {
  Temporaries temps;
  if (!gfx9::isLegalSop2(a, b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 1, b, temps);
  createScalarCmp32(builder, pred, a, b);
  materialize(builder, dst, /*invert=*/false);
}

void emitScalarEquality64(IRBuilder &builder, Predicate pred, RegisterRef dst,
                          Operand a, Operand b) {
  Temporaries temps;
  if (!gfx9::isLegalSalu64Source(a))
    a = copyToTemporary(builder, RegisterKind::sgpr, 2, a, temps);
  if (!gfx9::isLegalSalu64Source(b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 2, b, temps);
  if (!gfx9::isLegalSop2(a, b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 2, b, temps);
  if (pred == Predicate::eq)
    builder.create<rocisa::SCmpEQU64>(toInput64(a), toInput64(b));
  else
    builder.create<rocisa::SCmpLgU64>(toInput64(a), toInput64(b));
  materialize(builder, dst, /*invert=*/false);
}

// How an ordered predicate maps onto the borrow of x - y, where SCC is x <u y.
struct OrderedForm {
  bool swap;
  bool invert;
  bool isSigned;
};

OrderedForm orderedForm(Predicate pred) {
  switch (pred) {
  case Predicate::slt:
    return {false, false, true};
  case Predicate::sge:
    return {false, true, true};
  case Predicate::sgt:
    return {true, false, true};
  case Predicate::sle:
    return {true, true, true};
  case Predicate::ult:
    return {false, false, false};
  case Predicate::uge:
    return {false, true, false};
  case Predicate::ugt:
    return {true, false, false};
  case Predicate::ule:
    return {true, true, false};
  case Predicate::eq:
  case Predicate::ne:
    break;
  }
  throw std::invalid_argument("predicate is not an ordered comparison");
}

// Flipping the sign bit turns a signed comparison into an unsigned one.
// s_xor_b32 writes SCC, so it must precede the borrow chain.
Operand biasHigh(IRBuilder &builder, Operand high, Temporaries &temps) {
  if (high.isImmediate()) {
    uint32_t flipped = static_cast<uint32_t>(high.immediate()) ^ 0x80000000u;
    return Operand(int64_t{static_cast<int32_t>(flipped)});
  }
  RegisterRange biased = builder.allocateSgpr(1);
  builder.create<rocisa::SXorB32>(biased.container(), toInput(high),
                                  toInput(Operand(int64_t{signBit})));
  temps.push_back(std::move(biased));
  return Operand(temps.back());
}

// SCC ends up as x <u y; the difference itself is discarded.
void emitBorrowChain(IRBuilder &builder, Operand x, Operand y, bool isSigned,
                     Temporaries &temps) {
  Operand xHigh = detail::operandHalf(x, 1);
  Operand yHigh = detail::operandHalf(y, 1);
  if (isSigned) {
    xHigh = biasHigh(builder, xHigh, temps);
    yHigh = biasHigh(builder, yHigh, temps);
  }
  RegisterRange diff = builder.allocateSgpr(1);
  builder.create<rocisa::SSubU32>(diff.container(),
                                  toInput(detail::operandHalf(x, 0)),
                                  toInput(detail::operandHalf(y, 0)));
  builder.create<rocisa::SSubBU32>(diff.container(), toInput(xHigh),
                                   toInput(yHigh));
}

// dst is written only by the final s_cselect, after every operand was read, so
// it may alias a or b.
void emitScalarOrdered64(IRBuilder &builder, Predicate pred, RegisterRef dst,
                         Operand a, Operand b) {
  OrderedForm form = orderedForm(pred);
  Operand x = form.swap ? b : a;
  Operand y = form.swap ? a : b;
  Temporaries temps;
  // Two immediates would put two distinct literals in one instruction.
  if (x.isImmediate() && y.isImmediate())
    y = copyToTemporary(builder, RegisterKind::sgpr, 2, y, temps);
  emitBorrowChain(builder, x, y, form.isSigned, temps);
  materialize(builder, dst, form.invert);
}

void emitScalar64(IRBuilder &builder, Predicate pred, RegisterRef dst,
                  Operand a, Operand b) {
  if (pred == Predicate::eq || pred == Predicate::ne)
    emitScalarEquality64(builder, pred, dst, a, b);
  else
    emitScalarOrdered64(builder, pred, dst, a, b);
}

void emitCompare(IRBuilder &builder, int width, Predicate pred, RegisterRef dst,
                 Operand a, Operand b) {
  std::string_view op = opName(width);
  requireDestination(op, dst);
  a = validateOperand(op, width, a);
  b = validateOperand(op, width, b);
  requireScalarForUniformBool(op, dst, a, b);
  if (isScalar(a, b)) {
    if (width == 1)
      emitScalar32(builder, pred, dst, a, b);
    else
      emitScalar64(builder, pred, dst, a, b);
    return;
  }
  if (width == 1)
    emitVector32(builder, pred, dst, a, b);
  else
    emitVector64(builder, pred, dst, a, b);
}

} // namespace

void IRBuilder::cmpi32(Predicate pred, RegisterRef dst, Operand a, Operand b) {
  emitCompare(*this, 1, pred, dst, a, b);
}

void IRBuilder::cmpi64(Predicate pred, RegisterRef dst, Operand a, Operand b) {
  emitCompare(*this, 2, pred, dst, a, b);
}
