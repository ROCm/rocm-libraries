// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OperandUtils.hpp"

#include "Gfx9Legality.hpp"

#include "container.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <utility>

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

void requireNotAgpr(std::string_view opName, RegisterRef ref) {
  if (ref.kind != RegisterKind::agpr)
    return;
  throw std::invalid_argument(std::string(opName) +
                              " does not accept agpr register " +
                              registerText(ref));
}

Operand canonicalizeImmediate32(std::string_view opName, Operand operand) {
  if (operand.isRegister())
    return operand;
  int64_t value = operand.immediate();
  if (value < INT32_MIN || value > UINT32_MAX)
    throw std::invalid_argument(std::string(opName) + " immediate " +
                                std::to_string(value) +
                                " does not fit in 32 bits");
  return Operand(int64_t{static_cast<int32_t>(static_cast<uint32_t>(value))});
}

InstructionInput toInput(Operand operand) {
  if (operand.isRegister())
    return std::shared_ptr<rocisa::Container>(operand.reg().container());
  return static_cast<int>(operand.immediate());
}

InstructionInput toInput64(Operand operand) {
  if (operand.isRegister() || gfx9::isInlineConstant(operand.immediate()))
    return toInput(operand);
  assert(gfx9::isEncodableImmediate64(operand.immediate()) &&
         "immediate is not encodable in a 64-bit instruction");
  char text[16];
  std::snprintf(text, sizeof(text), "0x%x",
                static_cast<unsigned>(operand.immediate()));
  return std::string(text);
}

Operand operandHalf(Operand src, int index) {
  if (src.isRegister())
    return Operand(src.reg().slice(index));
  int64_t value = src.immediate();
  int32_t word = index == 0 ? static_cast<int32_t>(static_cast<uint32_t>(value))
                            : static_cast<int32_t>(value >> 32);
  return Operand(int64_t{word});
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

} // namespace rocm::tensilelite::codegen::detail
