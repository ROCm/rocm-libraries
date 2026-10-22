// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The gfx9 (gfx950) operand rules that rocisa does not check, verified against
// the LLVM assembler. Violations are rejected with "literal operands are not
// supported" (a literal in VOP3), "violates constant bus restrictions" (two
// distinct SGPRs or literals in a VALU instruction) and "only one unique
// literal operand is allowed" (SOP2). Immediates passed to these predicates
// must already be canonical 32-bit values, that is within the int32 range.

#ifndef ROCM_TENSILELITE_CODEGEN_GFX9LEGALITY_HPP
#define ROCM_TENSILELITE_CODEGEN_GFX9LEGALITY_HPP

#include <tensilelite/codegen/Operand.hpp>

#include <cstdint>

namespace rocm::tensilelite::codegen::gfx9 {

/// Integer inline constants. The float inline constants also match by bit
/// pattern but are treated as literals, which is conservative.
constexpr int64_t minInlineInteger = -16;
constexpr int64_t maxInlineInteger = 64;

bool isInlineConstant(int64_t value);
/// True if operand is an immediate that is not an inline constant.
bool isLiteral(Operand operand);
/// Constant bus reads of a VALU instruction: distinct SGPRs plus distinct
/// literals.
int constantBusReads(Operand src0, Operand src1);
/// VOP2: src1 must be a VGPR; src0 may be anything, including one literal.
bool isLegalVop2(Operand src0, Operand src1);
/// VOP3, including VOP2 ops the assembler promotes to VOP3 when src1 is not a
/// VGPR: no literals and at most one constant bus read.
bool isLegalVop3(Operand src0, Operand src1);
/// SOP2: no vector registers and at most one distinct literal.
bool isLegalSop2(Operand src0, Operand src1);
/// v_accvgpr_write_b32 takes a VGPR, an SGPR or an inline constant, but no
/// literal or AGPR. The immediate must be canonical.
bool isLegalAccvgprWriteSource(Operand src);
/// True if one 64-bit mov (s_mov_b64 or v_mov_b64) can write src into the pair
/// dst: dst is an SGPR or VGPR pair at an even index; a register source is a
/// pair of an allowed kind at an even index; an immediate is an inline
/// constant, sign-extended, or a 32-bit literal in [0, UINT32_MAX], which the
/// hardware zero-extends. Unlike the other predicates, the immediate is a
/// full 64-bit value.
bool isLegalMov64(RegisterRef dst, Operand src);

} // namespace rocm::tensilelite::codegen::gfx9

#endif // ROCM_TENSILELITE_CODEGEN_GFX9LEGALITY_HPP
