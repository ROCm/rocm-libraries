// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The gfx9 (gfx950) operand rules that rocisa does not check, verified against
// the LLVM assembler. Violations are rejected with "literal operands are not
// supported" (a literal in VOP3), "violates constant bus restrictions" (two
// distinct SGPRs or literals in a VALU instruction), "only one unique
// literal operand is allowed" (SOP2 and SOPC) and "invalid register alignment"
// (a 64-bit SGPR operand at an odd index). Immediates passed to these
// predicates must already be canonical 32-bit values, that is within the int32
// range, unless the predicate says it takes a full 64-bit value.

#ifndef ROCM_TENSILELITE_CODEGEN_GFX9LEGALITY_HPP
#define ROCM_TENSILELITE_CODEGEN_GFX9LEGALITY_HPP

#include <tensilelite/codegen/Operand.hpp>

#include <cstdint>

namespace rocm::tensilelite::codegen::gfx9 {

/// Integer inline constants. The float inline constants also match by bit
/// pattern but are treated as literals, which is conservative.
constexpr int64_t minInlineInteger = -16;
constexpr int64_t maxInlineInteger = 64;

/// Wait states a VALU read of an SGPR needs after a VALU instruction wrote it;
/// the s_nop that covers them takes this minus one. This also covers the
/// carry-in read of v_addc_co_u32 or v_subb_co_u32 after v_add_co_u32 or
/// v_sub_co_u32 wrote the carry to VCC or an SGPR pair: LLVM's hazard
/// recognizer inserts s_nop 1 there on gfx940 and gfx950, but not on gfx90a.
/// SALU carry chains through SCC need none.
constexpr int valuWriteSgprReadWaitStates = 2;

/// Wait states a VALU read of a transcendental result (v_rcp_f32,
/// v_rcp_iflag_f32) needs on gfx940 and gfx950, per LLVM's hazard recognizer.
constexpr int transResultWaitStates = 1;
/// Wait states v_readfirstlane_b32 needs after a VALU instruction wrote its
/// source VGPR on gfx940 and gfx950, per LLVM's hazard recognizer.
constexpr int valuWriteVgprReadlaneWaitStates = 1;

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
/// SOP2, and SOPC (s_cmp) as well: no vector registers and at most one
/// distinct literal. For 64-bit operands the immediates must also satisfy
/// isEncodableImmediate64.
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
/// True if a 64-bit instruction encodes value exactly: an inline constant, or a
/// 32-bit literal in [0, UINT32_MAX], which the hardware zero-extends.
bool isEncodableImmediate64(int64_t value);
/// True if src is a legal 64-bit SALU source: an SGPR pair at an even index or
/// an immediate satisfying isEncodableImmediate64.
bool isLegalSalu64Source(Operand src);
/// True if src is a legal 64-bit VOP3 source: a VGPR or SGPR pair at an even
/// index, or an inline-constant immediate.
bool isLegalVop3Source64(Operand src);
/// True if src is a legal v_cndmask_b32 data source: a VGPR or an inline
/// constant; an SGPR would violate the constant bus once the mask is read.
bool isLegalCndmaskSource(Operand src);
/// True if src is a v_addc_co_u32 or v_subb_co_u32 data source: a VGPR or an
/// inline constant, because the carry-in SGPR pair already uses the constant
/// bus.
bool isLegalCarryInSource(Operand src);
/// True if ref can hold a lane mask: an SGPR pair at an even index.
bool isLaneMask(RegisterRef ref);

} // namespace rocm::tensilelite::codegen::gfx9

#endif // ROCM_TENSILELITE_CODEGEN_GFX9LEGALITY_HPP
