// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OperandUtils.hpp"

#include "Gfx9Legality.hpp"

#include "container.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

using namespace rocm::tensilelite::codegen;

// Maps the register prefix rocisa prints back to a kind.
static RegisterKind kindOf(const std::string &prefix) {
  if (prefix == "s")
    return RegisterKind::sgpr;
  if (prefix == "v")
    return RegisterKind::vgpr;
  assert(prefix == "acc" && "unknown register prefix");
  return RegisterKind::agpr;
}

RegisterRef Operand::reg() const {
  assert(isRegister() && "operand is not a register");
  const rocisa::Container &container =
      *std::get<std::shared_ptr<rocisa::Container>>(
          static_cast<const InstructionInput &>(*this));
  const rocisa::RegisterContainer &c =
      static_cast<const rocisa::RegisterContainer &>(container);
  return RegisterRef{kindOf(c.regType), c.regIdx, c.regNum};
}

namespace rocm::tensilelite::codegen::detail {

std::string registerText(RegisterRef ref) {
  return ref.container()->toString();
}

void requireRegisterCount(std::string_view opName, RegisterRef ref, int count) {
  if (ref.count == count)
    return;
  std::string needed = count == 1 ? "single registers" : "register pairs";
  throw std::invalid_argument(std::string(opName) + " needs " + needed +
                              ", got " + registerText(ref));
}

void requireRegister(std::string_view opName, RegisterRef ref, int32_t count) {
  if (ref.kind == RegisterKind::agpr)
    throw std::invalid_argument(std::string(opName) +
                                " does not accept agpr register " +
                                registerText(ref));
  requireRegisterCount(opName, ref, count);
}

void requireOperand(std::string_view opName, Operand operand, int32_t count) {
  if (operand.isRegister())
    requireRegister(opName, operand.reg(), count);
}

void requireScalarSources(std::string_view opName, RegisterRef dst, Operand a,
                          Operand b) {
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

Operand operandHalf(Operand src, int32_t index) {
  if (src.isRegister())
    return Operand(src.reg().slice(index));
  int32_t value = src.immediate();
  return Operand(index == 0 ? value : (value < 0 ? -1 : 0));
}

Operand copyToTemporary(IRBuilder &builder, RegisterKind kind, int width,
                        Operand source, Temporaries &temps) {
  assert(kind != RegisterKind::agpr && "temporaries are sgprs or vgprs");
  RegisterRange temp = kind == RegisterKind::sgpr
                           ? builder.allocateSgpr(width, width)
                           : builder.allocateVgpr(width, width);
  if (width == 1)
    builder.movb32(temp.ref(), source);
  else
    builder.movb64(temp.ref(), source);
  temps.push_back(std::move(temp));
  return Operand(temps.back());
}

void legalizeVop3Sources(IRBuilder &builder, Operand &a, Operand &b,
                         Temporaries &temps) {
  if (gfx9::isLiteral(a))
    a = copyToTemporary(builder, RegisterKind::vgpr, 1, a, temps);
  if (gfx9::isLiteral(b))
    b = copyToTemporary(builder, RegisterKind::vgpr, 1, b, temps);
  // Two distinct SGPRs exceed the constant bus.
  if (!gfx9::isLegalVop3(a, b))
    b = copyToTemporary(builder, RegisterKind::vgpr, 1, b, temps);
}

} // namespace rocm::tensilelite::codegen::detail
