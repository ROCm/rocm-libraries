// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OperandUtils.hpp"

#include "container.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

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

} // namespace rocm::tensilelite::codegen::detail
