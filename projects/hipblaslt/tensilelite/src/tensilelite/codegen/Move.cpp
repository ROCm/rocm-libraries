// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "Gfx9Legality.hpp"
#include "OperandUtils.hpp"

#include "instruction/common.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace rocm::tensilelite::codegen;
using detail::registerText;
using detail::toInput;

namespace {

constexpr std::string_view movb32Name = "movb32";
constexpr std::string_view movb64Name = "movb64";

// Rejects the kind combinations that have no instruction. A source register
// is checked as is; an immediate is one canonical 32-bit value or half.
void requireLegalSource(std::string_view opName, RegisterRef dst, Operand src) {
  std::string prefix = std::string(opName) + " cannot write ";
  if (src.isImmediate()) {
    if (dst.kind != RegisterKind::agpr || gfx9::isLegalAccvgprWriteSource(src))
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

void emitMove32(IRBuilder &builder, RegisterRef dst, Operand src) {
  switch (dst.kind) {
  case RegisterKind::sgpr:
    builder.create<rocisa::SMovB32>(dst.container(), toInput(src));
    return;
  case RegisterKind::vgpr:
    if (src.isRegister(RegisterKind::agpr))
      builder.create<rocisa::VAccvgprReadB32>(dst.container(), toInput(src));
    else
      builder.create<rocisa::VMovB32>(dst.container(), toInput(src));
    return;
  case RegisterKind::agpr:
    builder.create<rocisa::VAccvgprWriteB32>(dst.container(), toInput(src));
    return;
  }
}

void emitMove64(IRBuilder &builder, RegisterRef dst, Operand src) {
  if (dst.kind == RegisterKind::sgpr)
    builder.create<rocisa::SMovB64>(dst.container(), detail::toInput64(src));
  else
    builder.create<rocisa::VMovB64>(dst.container(), detail::toInput64(src));
}

void emitSplitMove64(IRBuilder &builder, RegisterRef dst, Operand src) {
  // When dst starts above an overlapping src, the high half goes first so the
  // source high word is read before it is overwritten.
  bool highFirst = src.isRegister(dst.kind) && dst.start > src.reg().start;
  for (int i = 0; i < 2; ++i) {
    int index = highFirst ? 1 - i : i;
    builder.movb32(dst.slice(index), detail::operandHalf(src, index));
  }
}

// Checks the kind rules on the whole source, or on each half of an immediate.
void requireLegalSource64(RegisterRef dst, Operand src) {
  if (src.isRegister()) {
    requireLegalSource(movb64Name, dst, src);
    return;
  }
  requireLegalSource(movb64Name, dst, detail::operandHalf(src, 0));
  requireLegalSource(movb64Name, dst, detail::operandHalf(src, 1));
}

} // namespace

void IRBuilder::movb32(RegisterRef dst, Operand src) {
  detail::requireRegisterCount(movb32Name, dst, 1);
  if (src.isRegister())
    detail::requireRegisterCount(movb32Name, src.reg(), 1);
  src = detail::canonicalizeImmediate32(movb32Name, src);
  requireLegalSource(movb32Name, dst, src);
  emitMove32(*this, dst, src);
}

void IRBuilder::movb64(RegisterRef dst, Operand src) {
  detail::requireRegisterCount(movb64Name, dst, 2);
  if (src.isRegister())
    detail::requireRegisterCount(movb64Name, src.reg(), 2);
  requireLegalSource64(dst, src);
  if (gfx9::isLegalMov64(dst, src))
    emitMove64(*this, dst, src);
  else
    emitSplitMove64(*this, dst, src);
}
