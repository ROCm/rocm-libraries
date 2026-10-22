// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Operand validation and conversion shared by the IRBuilder ops.

#ifndef ROCM_TENSILELITE_CODEGEN_OPERANDUTILS_HPP
#define ROCM_TENSILELITE_CODEGEN_OPERANDUTILS_HPP

#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Operand.hpp>

#include "instruction/instruction.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace rocm::tensilelite::codegen::detail {

/// Returns ref as rocisa prints it, such as v[2:3], for diagnostics.
std::string registerText(RegisterRef ref);

/// Throws std::invalid_argument naming opName unless ref has exactly count
/// registers.
void requireRegisterCount(std::string_view opName, RegisterRef ref, int count);

/// Throws std::invalid_argument naming opName if ref is an AGPR.
void requireNotAgpr(std::string_view opName, RegisterRef ref);

/// Maps an immediate in [INT32_MIN, UINT32_MAX] to its int32 bit pattern, and
/// returns a register unchanged. Throws std::invalid_argument naming opName if
/// an immediate does not fit 32 bits.
Operand canonicalizeImmediate32(std::string_view opName, Operand operand);

/// Converts operand to a rocisa instruction input. A register becomes its
/// container; an immediate must be a canonical 32-bit value.
InstructionInput toInput(Operand operand);

/// Converts operand to an input of a 64-bit instruction. An
/// inline constant prints as an int; any other immediate is a 32-bit literal
/// that the hardware zero-extends, so it prints unsigned, because a negative
/// spelling would encode the same literal and lose the high word. A non-inline
/// immediate must lie in [0, UINT32_MAX].
InstructionInput toInput64(Operand operand);

/// Returns the low (index 0) or high (index 1) 32 bits of a register pair or a
/// 64-bit immediate. An immediate half is a canonical 32-bit value.
Operand operandHalf(Operand src, int index);

/// Temporary registers that stay allocated until the instructions reading them
/// have been emitted.
using Temporaries = std::vector<RegisterRange>;

/// Copies source into a new temporary of kind with width registers (1 or 2,
/// aligned to its width) that lives as long as temps does, and returns it.
Operand copyToTemporary(IRBuilder &builder, RegisterKind kind, int width,
                        Operand source, Temporaries &temps);

/// Copies literals, and the second of two distinct SGPRs, into VGPR temporaries
/// so that a and b suit a VOP3 instruction, which takes no literals and one
/// constant bus read.
void legalizeVop3Sources(IRBuilder &builder, Operand &a, Operand &b,
                         Temporaries &temps);

} // namespace rocm::tensilelite::codegen::detail

#endif // ROCM_TENSILELITE_CODEGEN_OPERANDUTILS_HPP
