// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_OPERAND_HPP
#define ROCM_TENSILELITE_CODEGEN_OPERAND_HPP

#include <tensilelite/codegen/Registers.hpp>

#include "instruction/instruction.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <variant>

namespace rocm::tensilelite::codegen {

/// An instruction source: a register or a 32-bit immediate, held as the rocisa
/// InstructionInput that instruction constructors take, so an Operand is passed
/// to them as is. A register is its RegisterContainer, of any width, so a pair
/// is one Operand; an immediate is the input's int. Converts implicitly from a
/// RegisterRange, a RegisterRef or an int32_t, so builder calls read like
/// builder.addi32(dst, a, 4); the 64-bit ops sign-extend an immediate. 64-bit
/// and unsigned integers do not convert, so a wide constant is a compile error
/// rather than a silent truncation: write a 32-bit pattern as its int32_t
/// value, such as -1 for 0xFFFFFFFF, and build a wider constant in a register
/// pair with two movb32.
class Operand : public InstructionInput {
public:
  Operand(const RegisterRange &range) : Operand(range.ref()) {}
  Operand(RegisterRef ref)
      : InstructionInput(std::shared_ptr<rocisa::Container>(ref.container())) {}
  Operand(int32_t immediate) : InstructionInput(immediate) {}
  Operand(int64_t) = delete;
  Operand(uint32_t) = delete;
  Operand(uint64_t) = delete;

  bool isRegister() const {
    return std::holds_alternative<std::shared_ptr<rocisa::Container>>(
        static_cast<const InstructionInput &>(*this));
  }
  /// True if this is a register of the given kind.
  bool isRegister(RegisterKind kind) const {
    return isRegister() && reg().kind == kind;
  }
  bool isImmediate() const {
    return std::holds_alternative<int>(
        static_cast<const InstructionInput &>(*this));
  }

  /// The register, read back from its container; asserts isRegister().
  RegisterRef reg() const;
  /// The immediate; asserts isImmediate().
  int32_t immediate() const {
    assert(isImmediate() && "operand is not an immediate");
    return std::get<int>(static_cast<const InstructionInput &>(*this));
  }
};

static_assert(sizeof(Operand) == sizeof(InstructionInput),
              "operand adds no state to the instruction input");

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_OPERAND_HPP
