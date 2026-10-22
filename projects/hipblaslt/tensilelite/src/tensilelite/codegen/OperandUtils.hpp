// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Operand validation and conversion shared by the IRBuilder ops.

#ifndef ROCM_TENSILELITE_CODEGEN_OPERANDUTILS_HPP
#define ROCM_TENSILELITE_CODEGEN_OPERANDUTILS_HPP

#include <tensilelite/codegen/Operand.hpp>

#include "instruction/instruction.hpp"

#include <string>
#include <string_view>

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

} // namespace rocm::tensilelite::codegen::detail

#endif // ROCM_TENSILELITE_CODEGEN_OPERANDUTILS_HPP
