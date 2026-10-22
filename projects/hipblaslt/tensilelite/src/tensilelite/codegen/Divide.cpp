// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Signed 32-bit division as LLVM's AMDGPU expansions. The exact path is LLVM's
// 32-bit expansion (AMDGPUCodeGenPrepare::expandDivRem32 and
// AMDGPUTargetLowering::LowerUDIVREM): an f32 reciprocal scaled by 2^32 - 512,
// one Newton step and two corrections, with abs and a sign fix-up for sdiv. The
// fast path is AMDGPUCodeGenPrepare::expandDivRem24Impl and
// AMDGPUTargetLowering::LowerDIVREM24.

#include <tensilelite/codegen/IRBuilder.hpp>

#include "Gfx9Legality.hpp"
#include "OperandUtils.hpp"

#include "container.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/instruction.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using namespace rocm::tensilelite::codegen;
using detail::copyToTemporary;
using detail::legalizeVop3Sources;
using detail::Temporaries;
using detail::toInput;

namespace {

constexpr std::string_view divName = "divsi32";

// 2^32 - 512 as an f32. It keeps the estimate below 2^32 / divisor for a
// reciprocal accurate to 1 ulp, which the Newton step and the corrections rely
// on.
const InstructionInput reciprocalScale = std::string("0x4f7ffffe");

void requireRegister(RegisterRef ref) {
  detail::requireNotAgpr(divName, ref);
  detail::requireRegisterCount(divName, ref, 1);
}

Operand validateOperand(Operand operand) {
  if (operand.isRegister())
    requireRegister(operand.reg());
  return detail::canonicalizeImmediate32(divName, operand);
}

void requireScalarSources(RegisterRef dst, Operand a, Operand b) {
  if (dst.kind != RegisterKind::sgpr)
    return;
  for (Operand source : {a, b}) {
    if (!source.isRegister(RegisterKind::vgpr))
      continue;
    throw std::invalid_argument(
        std::string(divName) + " cannot write sgpr destination " +
        detail::registerText(dst) + " from vgpr operand " +
        detail::registerText(source.reg()));
  }
}

// The helpers below pick the SALU or VALU instruction from the kind of dst.

RegisterRange allocate(IRBuilder &builder, RegisterKind kind) {
  return kind == RegisterKind::sgpr ? builder.allocateSgpr(1)
                                    : builder.allocateVgpr(1);
}

void createXor(IRBuilder &builder, RegisterRef dst, Operand a, Operand b) {
  if (dst.kind == RegisterKind::sgpr)
    builder.create<rocisa::SXorB32>(dst.container(), toInput(a), toInput(b));
  else
    builder.create<rocisa::VXorB32>(dst.container(), toInput(a), toInput(b));
}

void createAShr(IRBuilder &builder, RegisterRef dst, Operand src, int shift) {
  if (dst.kind == RegisterKind::sgpr)
    builder.create<rocisa::SAShiftRightI32>(dst.container(), shift,
                                            toInput(src));
  else
    builder.create<rocisa::VAShiftRightI32>(dst.container(), shift,
                                            toInput(src), "");
}

// The inline constant in src0 keeps the VOP2 form legal.
void createOrOne(IRBuilder &builder, RegisterRef dst, Operand src) {
  if (dst.kind == RegisterKind::sgpr)
    builder.create<rocisa::SOrB32>(dst.container(), toInput(src), 1);
  else
    builder.create<rocisa::VOrB32>(dst.container(), 1, toInput(src));
}

void createMulHi(IRBuilder &builder, RegisterRef dst, Operand a, Operand b) {
  if (dst.kind == RegisterKind::sgpr)
    builder.create<rocisa::SMulHIU32>(dst.container(), toInput(a), toInput(b));
  else
    builder.create<rocisa::VMulHIU32>(dst.container(), toInput(a), toInput(b));
}

// Readfirstlane is undefined with exec == 0, hence the documented precondition
// of the SALU form. The caller covers the VALU-write hazard on src.
RegisterRange readFirstLane(IRBuilder &builder, RegisterRef src) {
  RegisterRange value = builder.allocateSgpr(1);
  builder.create<rocisa::VReadfirstlaneB32>(value.container(), src.container());
  return value;
}

void legalizeExactSources(IRBuilder &builder, RegisterKind kind, Operand &a,
                          Operand &b, Temporaries &temps) {
  if (kind == RegisterKind::vgpr)
    return legalizeVop3Sources(builder, a, b, temps);
  // s_abs_i32 takes only registers.
  for (Operand *source : {&a, &b})
    if (source->isImmediate())
      *source = copyToTemporary(builder, RegisterKind::sgpr, 1, *source, temps);
}

// |INT32_MIN| is 2^31 as an unsigned value, which the unsigned core handles.
RegisterRange absoluteValue(IRBuilder &builder, RegisterKind kind, Operand x) {
  RegisterRange magnitude = allocate(builder, kind);
  if (kind == RegisterKind::sgpr) {
    builder.create<rocisa::SAbsI32>(magnitude.container(), x.reg().container());
    return magnitude;
  }
  builder.subi32(magnitude.ref(), 0, x);
  // x in src0 keeps the VGPR in src1.
  builder.create<rocisa::VMaxI32>(magnitude.container(), toInput(x),
                                  magnitude.container());
  return magnitude;
}

struct Magnitudes {
  RegisterRange dividend;
  RegisterRange divisor;
  // 0 when the signs agree and -1 when they differ.
  RegisterRange sign;
};

Magnitudes splitSigns(IRBuilder &builder, RegisterKind kind, Operand a,
                      Operand b) {
  RegisterRange divisor = absoluteValue(builder, kind, b);
  RegisterRange dividend = absoluteValue(builder, kind, a);
  RegisterRange sign = allocate(builder, kind);
  createXor(builder, sign.ref(), a, b);
  createAShr(builder, sign.ref(), sign, 31);
  return {std::move(dividend), std::move(divisor), std::move(sign)};
}

// Starts the estimate: divisor as an f32, its reciprocal scaled by 2^32 - 512,
// converted back to u32, all in one VGPR. negDivisor receives -divisor.
RegisterRange scaledReciprocal(IRBuilder &builder, RegisterRef divisor,
                               RegisterRef negDivisor) {
  RegisterRange value = builder.allocateVgpr(1);
  builder.create<rocisa::VCvtU32toF32>(value.container(), toInput(divisor));
  builder.create<rocisa::VRcpIFlagF32>(value.container(), value.container());
  // The one independent instruction between the reciprocal and its reader
  // covers the transcendental result hazard.
  static_assert(gfx9::transResultWaitStates == 1,
                "one instruction no longer covers the reciprocal hazard");
  builder.subi32(negDivisor, 0, divisor);
  builder.create<rocisa::VMulF32>(value.container(), reciprocalScale,
                                  value.container());
  builder.create<rocisa::VCvtF32toU32>(value.container(), value.container());
  return value;
}

// Returns x ~= 2^32 / divisor after one Newton step, in a register of kind.
RegisterRange reciprocalEstimate(IRBuilder &builder, RegisterKind kind,
                                 RegisterRef divisor) {
  RegisterRange negDivisor = allocate(builder, kind);
  RegisterRange x = scaledReciprocal(builder, divisor, negDivisor.ref());
  if (kind == RegisterKind::sgpr) {
    builder.create<rocisa::SNop>(gfx9::valuWriteVgprReadlaneWaitStates - 1);
    x = readFirstLane(builder, x.ref());
  }
  RegisterRange t = allocate(builder, kind);
  builder.muli32(t.ref(), negDivisor, x);
  createMulHi(builder, t.ref(), x, t);
  builder.addi32(x.ref(), x, t);
  return x;
}

// The temporaries one correction step needs before it compares.
struct Correction {
  RegisterRange next;
  std::optional<RegisterRange> reduced;
};

// The scalar add and sub write SCC, so they precede the compare.
Correction prepareCorrection(IRBuilder &builder, RegisterRef q, RegisterRef r,
                             RegisterRef divisor, bool updateRemainder) {
  Correction correction{allocate(builder, q.kind), std::nullopt};
  builder.addi32(correction.next.ref(), q, 1);
  if (!updateRemainder)
    return correction;
  correction.reduced.emplace(allocate(builder, q.kind));
  builder.subi32(correction.reduced->ref(), r, divisor);
  return correction;
}

void correctQuotientScalar(IRBuilder &builder, RegisterRef q, RegisterRef r,
                           RegisterRef divisor, bool updateRemainder) {
  Correction c = prepareCorrection(builder, q, r, divisor, updateRemainder);
  builder.create<rocisa::SCmpGeU32>(toInput(r), toInput(divisor));
  builder.create<rocisa::SCSelectB32>(q.container(), c.next.container(),
                                      q.container());
  if (updateRemainder)
    builder.create<rocisa::SCSelectB32>(r.container(), c.reduced->container(),
                                        r.container());
}

void correctQuotientVector(IRBuilder &builder, RegisterRef q, RegisterRef r,
                           RegisterRef divisor, bool updateRemainder) {
  Correction c = prepareCorrection(builder, q, r, divisor, updateRemainder);
  RegisterRange mask = builder.allocateSgpr(2, 2);
  builder.cmpi32(Predicate::uge, mask.ref(), r, divisor);
  builder.select(q, mask.ref(), c.next, q);
  if (updateRemainder)
    builder.select(r, mask.ref(), *c.reduced, r);
}

// One LLVM correction step: if r >= divisor then q += 1 and r -= divisor.
void correctQuotient(IRBuilder &builder, RegisterRef q, RegisterRef r,
                     RegisterRef divisor, bool updateRemainder) {
  if (q.kind == RegisterKind::sgpr)
    return correctQuotientScalar(builder, q, r, divisor, updateRemainder);
  correctQuotientVector(builder, q, r, divisor, updateRemainder);
}

void emitExact(IRBuilder &builder, RegisterRef dst, Operand a, Operand b) {
  RegisterKind kind = dst.kind;
  Temporaries temps;
  legalizeExactSources(builder, kind, a, b, temps);
  // a and b are read only here, before dst is written, so dst may alias them.
  Magnitudes m = splitSigns(builder, kind, a, b);
  RegisterRange x = reciprocalEstimate(builder, kind, m.divisor.ref());
  RegisterRange q = allocate(builder, kind);
  createMulHi(builder, q.ref(), m.dividend, x);
  RegisterRange r = allocate(builder, kind);
  builder.muli32(r.ref(), q, m.divisor);
  builder.subi32(r.ref(), m.dividend, r);
  correctQuotient(builder, q.ref(), r.ref(), m.divisor.ref(), true);
  // LLVM's second step updates only the quotient.
  correctQuotient(builder, q.ref(), r.ref(), m.divisor.ref(), false);
  // (q ^ sign) - sign negates q when the signs differ.
  createXor(builder, q.ref(), q, m.sign);
  builder.subi32(dst, q, m.sign);
}

void legalizeFastSources(IRBuilder &builder, RegisterKind kind, Operand &a,
                         Operand &b, Temporaries &temps) {
  if (kind == RegisterKind::vgpr)
    return legalizeVop3Sources(builder, a, b, temps);
  // Only s_xor reads both, and the VOP1 conversions take any scalar source.
  if (!gfx9::isLegalSop2(a, b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 1, b, temps);
}

// LLVM computes ashr (a ^ b), bitWidth - 2 and then ors 1, giving +1 or -1.
RegisterRange quotientSign(IRBuilder &builder, RegisterKind kind, Operand a,
                           Operand b) {
  RegisterRange sign = allocate(builder, kind);
  createXor(builder, sign.ref(), a, b);
  createAShr(builder, sign.ref(), sign, 30);
  createOrOne(builder, sign.ref(), sign);
  return sign;
}

struct FastEstimate {
  // A VGPR holding the truncated f32 quotient as an integer.
  RegisterRange quotient;
  // An SGPR pair set for the lanes whose quotient needs the +-1 correction.
  RegisterRange needsCorrection;
};

// Always VALU, since the division runs in f32.
FastEstimate estimateQuotient(IRBuilder &builder, Operand a, Operand b) {
  RegisterRange fb = builder.allocateVgpr(1);
  RegisterRange fq = builder.allocateVgpr(1);
  RegisterRange fa = builder.allocateVgpr(1);
  RegisterRange quotient = builder.allocateVgpr(1);
  RegisterRange mask = builder.allocateSgpr(2, 2);
  builder.create<rocisa::VCvtI32toF32>(fb.container(), toInput(b));
  builder.create<rocisa::VRcpIFlagF32>(fq.container(), fb.container());
  // Independent of the reciprocal, so it gives the result its wait state.
  builder.create<rocisa::VCvtI32toF32>(fa.container(), toInput(a));
  builder.create<rocisa::VMulF32>(fq.container(), fa.container(),
                                  fq.container());
  // Truncates toward zero.
  builder.create<rocisa::VCvtF32toI32>(quotient.container(), fq.container());
  // LLVM's v_trunc_f32, which rocisa lacks; the round trip is exact for
  // magnitudes below 2^31.
  builder.create<rocisa::VCvtI32toF32>(fq.container(), quotient.container());
  // The remainder fa - fq * fb in one rounding, as LLVM does on gfx950.
  std::shared_ptr<rocisa::RegisterContainer> negQuotient = fq.container();
  negQuotient->setMinus(true);
  builder.create<rocisa::VFmaF32>(fa.container(), negQuotient, fb.container(),
                                  fa.container());
  std::shared_ptr<rocisa::RegisterContainer> absRemainder = fa.container();
  std::shared_ptr<rocisa::RegisterContainer> absDivisor = fb.container();
  absRemainder->setAbs(true);
  absDivisor->setAbs(true);
  builder.create<rocisa::VCmpGEF32>(mask.container(), absRemainder, absDivisor);
  return {std::move(quotient), std::move(mask)};
}

void finishFastVector(IRBuilder &builder, RegisterRef dst,
                      const FastEstimate &estimate, const RegisterRange &sign) {
  builder.create<rocisa::SNop>(gfx9::valuWriteSgprReadWaitStates - 1);
  RegisterRange correction = builder.allocateVgpr(1);
  builder.select(correction.ref(), estimate.needsCorrection.ref(), sign, 0);
  builder.addi32(dst, estimate.quotient, correction);
}

void finishFastScalar(IRBuilder &builder, RegisterRef dst,
                      const FastEstimate &estimate, const RegisterRange &sign) {
  // Three VALU instructions separate the conversion from this read, more than
  // the one wait state the readfirstlane hazard needs.
  RegisterRange q = readFirstLane(builder, estimate.quotient.ref());
  // How LLVM tests a uniform mask; it reads exec and never writes it.
  RegisterRef mask = estimate.needsCorrection.ref();
  builder.create<rocisa::SAndB64>(mask.container(), mask.container(),
                                  std::make_shared<rocisa::EXEC>());
  RegisterRange correction = builder.allocateSgpr(1);
  builder.create<rocisa::SCSelectB32>(correction.container(),
                                      toInput(Operand(sign)), 0);
  builder.addi32(dst, q, correction);
}

// This is LLVM's signed 24-bit criterion: expandDivRem24 needs at least 9 sign
// bits, that is values in [-2^23, 2^23). Beyond that, fa * rcp(fb) can round up
// to the next integer, and the single correction only fixes an estimate that
// is too small.
void emitFast(IRBuilder &builder, RegisterRef dst, Operand a, Operand b) {
  RegisterKind kind = dst.kind;
  Temporaries temps;
  legalizeFastSources(builder, kind, a, b, temps);
  // a and b are read only here, before dst is written, so dst may alias them.
  RegisterRange sign = quotientSign(builder, kind, a, b);
  FastEstimate estimate = estimateQuotient(builder, a, b);
  if (kind == RegisterKind::sgpr)
    return finishFastScalar(builder, dst, estimate, sign);
  finishFastVector(builder, dst, estimate, sign);
}

} // namespace

void IRBuilder::divsi32(RegisterRef dst, Operand a, Operand b, bool useIEEE) {
  requireRegister(dst);
  a = validateOperand(a);
  b = validateOperand(b);
  requireScalarSources(dst, a, b);
  if (useIEEE)
    return emitExact(*this, dst, a, b);
  emitFast(*this, dst, a, b);
}
