/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "stinkytofu/transforms/asm/InstructionSizeCosting.hpp"

#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ostream>
#include <string>
#include <unordered_map>

#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace {
using namespace stinkytofu;

// 1/(2*PI) - short literal in GFX ISA (no extra encoding bytes)
constexpr double INV_2PI = 0.15915494309189533576888376337251436203445964574045644874766734;

/// True if the integer literal is encoded inline (no extra 32-bit literal
/// word). Exclusions per GFX: 0, signed 1..64, -1..-16.
inline bool isShortLiteralInt(int32_t v) {
    return (v >= 0 && v <= 64) || (v >= -16 && v <= -1);
}

/// True if the float literal is encoded inline (no extra literal word).
/// Exclusions per GFX: 0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2*PI).
inline bool isShortLiteralDouble(double v) {
    auto eq = [](double a, double b) {
        constexpr double eps = 1e-9;
        return std::abs(a - b) <= eps;
    };
    return eq(v, 0.0) || eq(v, 0.5) || eq(v, -0.5) || eq(v, 1.0) || eq(v, -1.0) || eq(v, 2.0) ||
           eq(v, -2.0) || eq(v, 4.0) || eq(v, -4.0) || eq(v, INV_2PI);
}

/// True if the value is in the range of 32-bit float (encoded as one 32-bit
/// literal word). Used to decide literal encoding: 4-byte (float) vs 8-byte
/// (double). Exact round-trip (double→float→double) is too strict for literals
/// like -1.442695; use range check instead.
inline bool fitsInFloat32(double v) {
    if (std::isnan(v) || std::isinf(v)) return true;  // float has NaN and Inf
    constexpr double floatMax = static_cast<double>(std::numeric_limits<float>::max());
    return v >= -floatMax && v <= floatMax;
}

/// True if the operand is a VGPR (vector GPR). Literals and SGPR/other are not
/// VGPR. Used to detect VOP1/VOP2/VOPC promotion to VOP3 when any source is not
/// VGPR (8 bytes).
inline bool isVGPR(const StinkyRegister& reg) {
    return reg.dataType == StinkyRegister::Type::Register && reg.reg.type == RegType::V;
}

/// VOP1 narrow converts `v_cvt_f32_bf16`, `v_cvt_f32_f16`, `v_cvt_f16_f32`,
/// `v_cvt_pk_f32_bf8`, `v_cvt_pk_f32_fp8`: `_e32` uses a 7-bit field for the
/// **in-bank** VGPR index (0..255 per MODE VGPR MSBs). Physical VGPRs 0..1023
/// map to **logical** indices `phys % 256` within the current 256-register
/// group; promotion uses that logical span on
/// **sources only** (not dst). Virtual registers (pending allocation) skip this
/// rule.
inline bool vgprOperandExceedsVop1CvtE32IndexLimit(const StinkyRegister& reg) {
    if (!isVGPR(reg) || isPseudoReg(reg)) return false;
    if ((reg.reg.idx & StinkyRegister::kVirtualBit) != 0) return false;
    uint32_t n = reg.reg.num;
    if (n < 1) n = 1;
    const uint32_t phys = reg.reg.idx & ~StinkyRegister::kVirtualBit;
    const uint32_t lo = phys % 256u;
    const uint64_t lastInGroup = static_cast<uint64_t>(lo) + static_cast<uint64_t>(n) - 1u;
    if (lastInGroup > 255u) return true;  // operand crosses a 256-VGPR MODE bank
    return lastInGroup > 127u;
}

inline bool isVcvtNarrowSrcBankFamily(const char* mnemonic) {
    if (!mnemonic) return false;
    return std::strcmp(mnemonic, "v_cvt_f32_bf16") == 0 ||
           std::strcmp(mnemonic, "v_cvt_f32_f16") == 0 ||
           std::strcmp(mnemonic, "v_cvt_f16_f32") == 0 ||
           std::strcmp(mnemonic, "v_cvt_pk_f32_bf8") == 0 ||
           std::strcmp(mnemonic, "v_cvt_pk_f32_fp8") == 0;
}

/// True if the operand is VCC, vcc_lo, or vcc_hi (vector condition code).
inline bool isVCC(const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;
    RegType t = reg.reg.type;
    return t == RegType::VCC || t == RegType::VCC_LO || t == RegType::VCC_HI;
}

/// True if the mnemonic is a plain VOPC compare (v_cmp_*) but not v_cmpx_*.
/// v_cmp_lt_u32, v_cmp_eq_u32 have .flags = {VALU}; v_cmpx_* have .flags =
/// {VCmpX}.
inline bool isVOPCCompareNonX(const char* mnemonic) {
    return mnemonic && std::strncmp(mnemonic, "v_cmp_", 6) == 0;
}

/// Parse LiteralString that may be a hex (0x... or -0x...) or decimal literal;
/// return true and set value (32-bit) if so. True if \p mnemonic is non-null
/// and ends with `_f32` (e.g. `v_mul_f32`, `v_fma_f32`).
inline bool mnemonicEndsWithUnderscoreF32(const char* mnemonic) {
    if (!mnemonic) return false;
    const size_t n = std::strlen(mnemonic);
    return n >= 4 && std::strcmp(mnemonic + n - 4, "_f32") == 0;
}

/// Parse \p lit as an unsigned `0x` / `0X` 32-bit hex token (entire \p lit).
/// Sets \p outBits on success.
inline bool parseUnsignedHexLiteralU32(const std::string& lit, uint32_t& outBits) {
    if (lit.size() < 3 || lit[0] != '0' || (lit[1] != 'x' && lit[1] != 'X')) return false;
    // Reject signed spellings (-0x...); those use the integer literal path.
    char* end = nullptr;
    unsigned long val = std::strtoul(lit.c_str(), &end, 16);
    if (end != lit.c_str() + lit.size() || val > 0xFFFFFFFFul) return false;
    outBits = static_cast<uint32_t>(val);
    return true;
}

/// VALU `*_f32` only: `LiteralString` `0x........` is treated as raw IEEE-754
/// float32 bits for inline-literal costing (matches `LiteralDouble` /
/// `isShortLiteralDouble` / `fitsInFloat32`). Returns true if this rule handled
/// \p lit (including +0 extra for short floats).
inline bool tryAddLiteralExtraForValuF32HexFloatBits(const StinkyInstruction& inst,
                                                     const std::string& lit, int& extra) {
    if (!inst.is(InstFlag::IF_VALU)) return false;
    const char* mnemonic = inst.getHwInstDesc() ? inst.getHwInstDesc()->mnemonic : nullptr;
    if (!mnemonicEndsWithUnderscoreF32(mnemonic)) return false;
    uint32_t bits = 0;
    if (!parseUnsignedHexLiteralU32(lit, bits)) return false;
    float f = 0.0f;
    double d = 0.0;
    std::memcpy(&f, &bits, sizeof(f));
    d = static_cast<double>(f);
    if (!isShortLiteralDouble(d)) extra += fitsInFloat32(d) ? 4 : 8;
    return true;
}

inline bool parseLiteralStringToInt(const std::string& s, int32_t& out) {
    if (s.empty()) return false;
    // Hex: "0x...", "0X...", "-0x...", "-0X..."
    bool negative = (s.size() >= 1 && s[0] == '-');
    size_t hexStart = negative ? 1u : 0u;
    if (s.size() >= hexStart + 2 && s[hexStart] == '0' &&
        (s[hexStart + 1] == 'x' || s[hexStart + 1] == 'X')) {
        char* end = nullptr;
        unsigned long val = std::strtoul(s.c_str() + hexStart, &end, 16);
        if (end != s.c_str() + hexStart && *end == '\0' && val <= 0xFFFFFFFFu) {
            uint32_t u = static_cast<uint32_t>(val);
            out = negative ? static_cast<int32_t>(0u - u) : static_cast<int32_t>(u);
            return true;
        }
        return false;
    }
    for (size_t i = (s[0] == '-' ? 1u : 0u); i < s.size(); ++i)
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    char* end = nullptr;
    long val = std::strtol(s.c_str(), &end, 10);
    if (end != s.c_str() && *end == '\0' && val >= INT32_MIN && val <= INT32_MAX) {
        out = static_cast<int32_t>(val);
        return true;
    }
    return false;
}

/// Return the base instruction size in bytes used for cost accumulation, after
/// VALU 4 B → 8 B promotion. Rules follow Gfx1250Formats.def (VOP1/VOP2/VOPC
/// are 32-bit; VOP3/VOP3P are 64-bit in the same families).
///
/// - Starting size: **`hardwareEncodingBytes(inst)`** (`HwInstDesc::encoding` /
/// 8, else 4).
/// - Only compact VALU (4 B + `IF_VALU`) is eligible. SALU (SOP1/SOP2/…) and
/// any instruction
///   whose `base` is not 4 returns `base` unchanged; this helper does not widen
///   SOP encodings.
/// - If eligible, the first matching case below returns 8; if none match,
/// return `base`.
///   1) VOP3P modifiers: non-empty **`op_sel`**, **`op_sel_hi`**, or
///   **`byte_sel`** forces VOP3 (8 B)
///      (all compact VALU, including the narrow converts below).
///   2) `v_cvt_f32_bf16` / `v_cvt_f32_f16` / `v_cvt_f16_f32`, **when (1) did
///   not apply**: any **source**
///      VGPR whose **logical** index (`phys % 256`, plus `num`) spans above
///      **127** requires `_e64` (8 B); **dst VGPR is not used** for this rule.
///   3) `v_cndmask_b32` / `v_add_co_ci_u32` with three sources: 8 B if the last
///   source is not
///      VCC; 4 B if the last source is VCC.
///   4) `src1` (second source) is not a VGPR: e.g. literal, SGPR, or other
///   non-`V` register type
///      forces the VOP3 form (8 B).
///   5) VOPC compare: mnemonic `v_cmp_*` (excludes `v_cmpx_*`). If the first
///   destination is not
///      VCC (e.g. some scalar/flag slot instead of `vcc`), 8 B; with `vcc`
///      dest, 4 B.
int getEffectiveBaseSizeInBytesImpl(const StinkyInstruction& inst) {
    int base = hardwareEncodingBytes(inst);
    bool isCompactVOP = (base == 4 && inst.is(InstFlag::IF_VALU));
    if (isCompactVOP) {
        const auto& srcs = inst.getSrcRegs();
        const char* mnemonic = inst.getHwInstDesc() ? inst.getHwInstDesc()->mnemonic : nullptr;
        if (const VOP3PModifiers* vop3p = inst.getModifier<VOP3PModifiers>()) {
            if (!vop3p->op_sel.empty() || !vop3p->op_sel_hi.empty() || !vop3p->byte_sel.empty())
                return 8;  // e.g. v_cvt_f32_bf16 op_sel, v_cvt_pk_f32_fp8 op_sel,
                           // v_cvt_f32_fp8 byte_sel
        }
        if (isVcvtNarrowSrcBankFamily(mnemonic)) {
            for (const StinkyRegister& s : srcs)
                if (vgprOperandExceedsVop1CvtE32IndexLimit(s)) return 8;
        }
        // v_cndmask_b32 / v_add_co_ci_u32: 3 sources; 4 bytes iff last source is
        // VCC, else 8 bytes.
        if (mnemonic && srcs.size() >= 3 &&
            (std::strcmp(mnemonic, "v_cndmask_b32") == 0 ||
             std::strcmp(mnemonic, "v_add_co_ci_u32") == 0)) {
            const StinkyRegister& lastSrc = srcs.back();
            if (!isVCC(lastSrc)) return 8;  // last source not VCC -> promoted to VOP3
        }
        if (srcs.size() >= 2 && !isVGPR(srcs[1])) return 8;  // src1 not VGPR -> promoted to VOP3
        // VOPC compare v_cmp_* (not v_cmpx_*): dest != vcc -> promoted to VOP3 (8
        // bytes).
        if (isVOPCCompareNonX(mnemonic)) {
            const auto& destRegs = inst.getDestRegs();
            if (!destRegs.empty() && !isVCC(destRegs[0]))
                return 8;  // dest not vcc -> promoted to VOP3
        }
    }
    return base;
}

/// Literal-pool tail bytes after the instruction's fixed encoding (from
/// `hardwareEncodingBytes`).
///
/// Algorithm:
/// 1. If any early-exit applies, return 0 (operands are not inspected).
/// 2. Else sum contributions from every source and destination
/// `StinkyRegister`.
///
/// Early exits — return 0 (no operand scan). These encodings pack
/// offsets/immediates in the instruction words; this helper does not add a
/// separate literal tail for them:
/// - `MicrocodeFormat::MC_SMEM`: scalar memory / SMRD-class encodings
/// (`DEF_FORMAT(SMRD)` in
///   `Gfx1250Formats.def`, e.g. `SMRD_LOAD`, `SMRD_STORE`; offsets in the
///   instruction word). Matches the former `IF_SMemLoad` / `IF_SMemStore` /
///   `IF_SMemAtomic` / `IF_ScalarInstPrefetch` early-out for all
///   tablegen-backed Gfx1250 SMEM ops.
/// - Vector memory by flag: MUBUF load/store/atomic (12 B); DS
/// read/store/atomic (8 B); FLAT
///   load/store/atomic (12 B).
/// - `MicrocodeFormat::MC_VIMAGE`: 96-bit ENC_VIMAGE (`DEF_FORMAT(VIMAGE)` in
/// `Gfx1250Formats.def`;
///   tensor is `DEF_FORMAT(TENSOR)`, mnemonic `tensor_load_to_lds`). On Gfx1250
///   this is the only VIMAGE user today (same effect as the old
///   `IF_TENSORLoadToLds` guard).
/// - `MicrocodeFormat::MC_SOPP`: scalar control / inline fields
/// (`DEF_FORMAT(SOPP)` and children
///   `SOPP_BRANCH`, `SOPP_WAIT16`, `SOPP_IMM32`; examples include
///   `s_set_vgpr_msb`).
/// - `MicrocodeFormat::MC_SOPK`: `simm16` usually covers immediates (like
/// SOPP). Special case only:
///   `GFX::s_setreg_IMM32_b32` with at least one source adds +4 for one 32-bit
///   literal word (handled only in the SOPK branch, not via the operand loop
///   below).
///
/// Operand scan — for each operand, add 0 or more bytes (multiple rules can
/// stack across operands):
/// - `LiteralInt`: +4 unless the value is a GFX short literal (0; 1..64;
/// -1..-16). Example: 100 -> +4,
///   42 -> 0.
/// - `LiteralDouble`: if not a GFX short double, +4 when representable as
/// float32 else +8. Short set:
///   0, +/-0.5, +/-1, +/-2, +/-4, 1/(2*PI); helpers `isShortLiteralDouble`,
///   `fitsInFloat32`.
/// - `LiteralString`:
///   - Exactly `BufferOOB`: +4 (Tensile-style sentinel encodings).
///   - Prefix `label` (e.g. `label_Activation_None_VW8`): pick comparison
///   offset = map value when
///     `labelByteOffset` is non-null and contains this key; otherwise use
///     `currentByteOffsetBeforeInst`. +4 if that offset > 64, else 0.
///   - VALU mnemonic ending in `_f32`: whole token is unsigned `0x` / `0X` hex
///   → interpret as float32
///     bits; cost like `LiteralDouble` above. Does not apply to SALU or
///     non-`_f32` VALU (e.g. `s_mov_b32` stays int-style).
///   - Anything else: +4 when decimal/hex parsing yields a non-short int32, or
///   `asmSetSymbols` binds
///     the token to a non-short int (e.g. `0x4100`).
int getLiteralExtraBytesImpl(const StinkyInstruction& inst,
                             const std::unordered_map<std::string, int64_t>* labelByteOffset,
                             int64_t currentByteOffsetBeforeInst,
                             const std::unordered_map<std::string, int64_t>* asmSetSymbols) {
    if (const HwInstDesc* desc = inst.getHwInstDesc();
        desc && desc->microcode == MicrocodeFormat::MC_SMEM) {
        return 0;
    }
    if (inst.is(InstFlag::IF_MUBUFLoad) || inst.is(InstFlag::IF_MUBUFStore) ||
        inst.is(InstFlag::IF_MUBUFAtomic)) {
        return 0;
    }
    if (inst.is(InstFlag::IF_DSRead) || inst.is(InstFlag::IF_DSStore) ||
        inst.is(InstFlag::IF_DSAtomic)) {
        return 0;
    }
    if (inst.is(InstFlag::IF_FLATLoad) || inst.is(InstFlag::IF_FLATStore) ||
        inst.is(InstFlag::IF_FLATAtomic)) {
        return 0;
    }
    if (const HwInstDesc* desc = inst.getHwInstDesc();
        desc && desc->microcode == MicrocodeFormat::MC_VIMAGE) {
        return 0;
    }
    if (const HwInstDesc* desc = inst.getHwInstDesc();
        desc && desc->microcode == MicrocodeFormat::MC_SOPP) {
        return 0;
    }

    if (const HwInstDesc* desc = inst.getHwInstDesc();
        desc && desc->microcode == MicrocodeFormat::MC_SOPK) {
        if (!inst.getSrcRegs().empty() && inst.getUnifiedOpcode() == GFX::s_setreg_IMM32_b32)
            return 4;
        return 0;
    }

    auto countLiteralExtra = [&inst, labelByteOffset, currentByteOffsetBeforeInst, asmSetSymbols](
                                 const StinkyRegister& reg, int& extra) {
        using T = StinkyRegister::Type;
        if (reg.dataType == T::LiteralInt) {
            if (!isShortLiteralInt(reg.literalInt)) extra += 4;
        } else if (reg.dataType == T::LiteralDouble) {
            if (!isShortLiteralDouble(reg.literalDouble))
                extra += fitsInFloat32(reg.literalDouble) ? 4 : 8;
        } else if (reg.dataType == T::LiteralString) {
            const std::string lit = reg.getLiteralString();
            if (lit == "BufferOOB") {
                extra += 4;
            } else if (lit.size() >= 5 && lit.compare(0, 5, "label") == 0) {
                if (labelByteOffset) {
                    auto it = labelByteOffset->find(lit);
                    if (it != labelByteOffset->end())
                        extra += (it->second > 64) ? 4 : 0;
                    else
                        extra += (currentByteOffsetBeforeInst > 64) ? 4 : 0;
                } else
                    extra += (currentByteOffsetBeforeInst > 64) ? 4 : 0;
            } else {
                if (tryAddLiteralExtraForValuF32HexFloatBits(inst, lit, extra))
                    ;
                else {
                    int32_t parsed = 0;
                    if (parseLiteralStringToInt(lit, parsed) && !isShortLiteralInt(parsed))
                        extra += 4;
                    else if (tryResolveAsmSetSymbolToInt32(asmSetSymbols, lit, parsed) &&
                             !isShortLiteralInt(parsed))
                        extra += 4;
                }
            }
        }
    };

    int extra = 0;
    for (const StinkyRegister& reg : inst.getSrcRegs()) countLiteralExtra(reg, extra);
    for (const StinkyRegister& reg : inst.getDestRegs()) countLiteralExtra(reg, extra);
    return extra;
}
}  // namespace

namespace stinkytofu {
int getEffectiveBaseSizeInBytes(const StinkyInstruction& inst) {
    return getEffectiveBaseSizeInBytesImpl(inst);
}

int getLiteralExtraBytes(const StinkyInstruction& inst,
                         const std::unordered_map<std::string, int64_t>* labelByteOffset,
                         int64_t currentByteOffsetBeforeInst,
                         const std::unordered_map<std::string, int64_t>* asmSetSymbols) {
    return getLiteralExtraBytesImpl(inst, labelByteOffset, currentByteOffsetBeforeInst,
                                    asmSetSymbols);
}

int totalInstructionEncodingBytes(const StinkyInstruction& inst,
                                  const std::unordered_map<std::string, int64_t>* labelByteOffset,
                                  int64_t currentByteOffsetBeforeInst,
                                  const std::unordered_map<std::string, int64_t>* asmSetSymbols) {
    return getEffectiveBaseSizeInBytes(inst) +
           getLiteralExtraBytes(inst, labelByteOffset, currentByteOffsetBeforeInst, asmSetSymbols);
}

int64_t paddingBytesForCodeAlignment(int64_t offsetBytes, int64_t alignmentBytes) {
    if (alignmentBytes <= 1) return 0;
    if (offsetBytes < 0) return 0;
    const int64_t m = offsetBytes % alignmentBytes;
    return m == 0 ? 0 : (alignmentBytes - m);
}

void addAlignmentPaddingFromDirectiveNode(IRBase* node, int64_t baseByteOffset, int64_t& totalBytes,
                                          std::ostream* debugOut) {
    if (!node || node->getType() != IRBase::IRType::StinkyAsmDirective) return;
    const AsmDirective* ad = dyn_cast<AsmDirective>(node);
    if (!ad || ad->kind != AsmDirectiveKind::ALIGN) return;
    const int64_t off = baseByteOffset + totalBytes;
    const int64_t N = ad->intValue;
    const int64_t pad = paddingBytesForCodeAlignment(off, N);
    totalBytes += pad;
    if (debugOut && pad != 0)
        *debugOut << "  [.align " << N << " padding=" << pad << " bytes, total=" << totalBytes
                  << " bytes]\n";
}

void addAlignmentPaddingForLabelInstruction(const StinkyInstruction& inst, int64_t baseByteOffset,
                                            int64_t& totalBytes, std::ostream* debugOut) {
    if (inst.getUnifiedOpcode() != GFX::LABEL) return;
    if (const LabelData* ld = inst.getModifier<LabelData>()) {
        if (ld->alignment > 1u) {
            const int64_t off = baseByteOffset + totalBytes;
            const int64_t pad =
                paddingBytesForCodeAlignment(off, static_cast<int64_t>(ld->alignment));
            totalBytes += pad;
            if (debugOut && pad != 0)
                *debugOut << "  [label .align " << ld->alignment << " padding=" << pad
                          << " bytes, total=" << totalBytes << " bytes]\n";
        }
    }
}
}  // namespace stinkytofu
