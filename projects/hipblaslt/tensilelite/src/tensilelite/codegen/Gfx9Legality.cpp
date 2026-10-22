// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Gfx9Legality.hpp"

#include <cstdint>

namespace rocm::tensilelite::codegen::gfx9 {

namespace {

bool isVector(Operand operand) {
  return operand.isRegister(RegisterKind::vgpr) ||
         operand.isRegister(RegisterKind::agpr);
}

// True if both operands are the same SGPR or the same literal.
bool sameScalarSource(Operand a, Operand b) {
  if (a.isRegister(RegisterKind::sgpr) && b.isRegister(RegisterKind::sgpr))
    return a.reg() == b.reg();
  return isLiteral(a) && isLiteral(b) && a.immediate() == b.immediate();
}

bool usesConstantBus(Operand operand) {
  return operand.isRegister(RegisterKind::sgpr) || isLiteral(operand);
}

bool isVgprOrInlineConstant(Operand src) {
  if (src.isImmediate())
    return isInlineConstant(src.immediate());
  return src.isRegister(RegisterKind::vgpr);
}

} // namespace

bool isInlineConstant(int64_t value) {
  return value >= minInlineInteger && value <= maxInlineInteger;
}

bool isLiteral(Operand operand) {
  return operand.isImmediate() && !isInlineConstant(operand.immediate());
}

int constantBusReads(Operand src0, Operand src1) {
  int reads = 0;
  if (usesConstantBus(src0))
    ++reads;
  if (usesConstantBus(src1) && !sameScalarSource(src0, src1))
    ++reads;
  return reads;
}

bool isLegalVop2(Operand src0, Operand src1) {
  return src1.isRegister(RegisterKind::vgpr) &&
         constantBusReads(src0, src1) <= 1;
}

bool isLegalVop3(Operand src0, Operand src1) {
  return !isLiteral(src0) && !isLiteral(src1) &&
         constantBusReads(src0, src1) <= 1;
}

bool isLegalSop2(Operand src0, Operand src1) {
  if (isVector(src0) || isVector(src1))
    return false;
  return !(isLiteral(src0) && isLiteral(src1) && !sameScalarSource(src0, src1));
}

bool isLegalAccvgprWriteSource(Operand src) {
  if (src.isImmediate())
    return isInlineConstant(src.immediate());
  return src.isRegister(RegisterKind::vgpr) ||
         src.isRegister(RegisterKind::sgpr);
}

bool isLegalMov64(RegisterRef dst, Operand src) {
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

bool isEncodableImmediate64(int64_t value) {
  return isInlineConstant(value) || (value >= 0 && value <= UINT32_MAX);
}

bool isLegalSalu64Source(Operand src) {
  if (src.isImmediate())
    return isEncodableImmediate64(src.immediate());
  return src.isRegister(RegisterKind::sgpr) && src.reg().start % 2 == 0;
}

bool isLegalVop3Source64(Operand src) {
  if (src.isImmediate())
    return isInlineConstant(src.immediate());
  RegisterRef reg = src.reg();
  return reg.kind != RegisterKind::agpr && reg.start % 2 == 0;
}

bool isLegalCndmaskSource(Operand src) { return isVgprOrInlineConstant(src); }

bool isLegalCarryInSource(Operand src) { return isVgprOrInlineConstant(src); }

bool isLaneMask(RegisterRef ref) {
  return ref.kind == RegisterKind::sgpr && ref.count == 2 && ref.start % 2 == 0;
}

} // namespace rocm::tensilelite::codegen::gfx9
