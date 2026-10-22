// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_OPERAND_HPP
#define ROCM_TENSILELITE_CODEGEN_OPERAND_HPP

#include <tensilelite/codegen/Registers.hpp>

#include <cassert>
#include <cstdint>
#include <variant>

namespace rocm::tensilelite::codegen {

/// An instruction source: a register reference or an integer immediate.
/// Converts implicitly from a RegisterRange, a RegisterRef or an integer, so
/// builder calls read like builder.addi32(dst, a, 4). Immediates are 64-bit so
/// one type serves 32- and 64-bit ops; each op checks that the value fits its
/// width.
class Operand {
public:
  Operand(const RegisterRange &range) : value(range.ref()) {}
  Operand(RegisterRef ref) : value(ref) {}
  Operand(int64_t immediate) : value(immediate) {}

  bool isRegister() const { return std::holds_alternative<RegisterRef>(value); }
  /// True if this is a register of the given kind.
  bool isRegister(RegisterKind kind) const {
    return isRegister() && reg().kind == kind;
  }
  bool isImmediate() const { return std::holds_alternative<int64_t>(value); }

  /// The register; asserts isRegister().
  RegisterRef reg() const {
    assert(isRegister() && "operand is not a register");
    return std::get<RegisterRef>(value);
  }
  /// The immediate; asserts isImmediate().
  int64_t immediate() const {
    assert(isImmediate() && "operand is not an immediate");
    return std::get<int64_t>(value);
  }

private:
  std::variant<RegisterRef, int64_t> value;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_OPERAND_HPP
