// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Operand validation and conversion shared by the IRBuilder ops.

#ifndef ROCM_TENSILELITE_CODEGEN_OPERANDUTILS_HPP
#define ROCM_TENSILELITE_CODEGEN_OPERANDUTILS_HPP

#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Operand.hpp>

#include "instruction/instruction.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rocm::tensilelite::codegen::detail {

/// Returns ref as rocisa prints it, such as v[2:3], for diagnostics.
std::string registerText(RegisterRef ref);

/// Throws std::invalid_argument naming opName unless ref has exactly count
/// registers.
void requireRegisterCount(std::string_view opName, RegisterRef ref, int count);

/// Throws std::invalid_argument naming opName if ref is an AGPR or does not
/// have exactly count registers.
void requireRegister(std::string_view opName, RegisterRef ref, int32_t count);

/// requireRegister for a register operand; every immediate is accepted.
void requireOperand(std::string_view opName, Operand operand, int32_t count);

/// Throws std::invalid_argument naming opName if dst is an SGPR and a or b is a
/// VGPR, since there is no implicit readfirstlane.
void requireScalarSources(std::string_view opName, RegisterRef dst, Operand a,
                          Operand b);

/// Returns the low (index 0) or high (index 1) 32 bits of a register pair or of
/// an immediate sign-extended to 64 bits.
Operand operandHalf(Operand src, int32_t index);

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
