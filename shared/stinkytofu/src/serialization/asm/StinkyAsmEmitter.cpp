/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>

#include "stinkytofu/hardware/HwRegHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
static std::string vectorToString(const std::vector<int>& vec) {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        result += std::to_string(vec[i]);
        if (i < vec.size() - 1) {
            result += ",";
        }
    }
    result += "]";
    return result;
}
}  // namespace stinkytofu

namespace stinkytofu {
// Forward declarations of static helper functions
static void emitRegister(std::ostream& os, const StinkyRegister& reg,
                         const AsmEmitterOptions& options);
static void emitMnemonic(std::ostream& os, const StinkyInstruction& inst);
static void emitOperands(std::ostream& os, const StinkyInstruction& inst,
                         const AsmEmitterOptions& options);
static bool emitCustomOperands(std::ostream& os, const StinkyInstruction& inst);
static void emitTrailingModifiers(std::ostream& os, const StinkyInstruction& inst);
static void emitCycleComment(std::ostream& os, const StinkyInstruction& inst, int currentColumn,
                             const AsmEmitterOptions& options);
static void emitDirective(std::ostream& os, const AsmDirective& directive,
                          const AsmEmitterOptions& options);
static void emitBasicBlock(std::ostream& os, const BasicBlock& bb, const AsmEmitterOptions& options,
                           StinkyAsmEmitter* emitter);

// NOLINTBEGIN(misc-use-internal-linkage)
// Stream operators for instruction modifiers — must remain in namespace stinkytofu for ADL.
inline std::ostream& operator<<(std::ostream& os, const SWaitTensorCntData& waitTensorCntData) {
    os << "tlcnt=" << (int)waitTensorCntData.tlcnt;
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const SDelayAluData& delayAluData) {
    auto typeToString = [](SDelayAluData::InstType type) -> const char* {
        switch (type) {
            case SDelayAluData::InstType::VALU:
                return "VALU";
            case SDelayAluData::InstType::SALU:
                return "SALU";
            case SDelayAluData::InstType::TRANS:
                return "TRANS32";
            case SDelayAluData::InstType::NO_DEP:
                return "NO_DEP";
            default:
                return "UNKNOWN";
        }
    };

    auto skipToString = [](int8_t skip) -> const char* {
        switch (skip) {
            case 0:
                return "SAME";
            case 1:
                return "NEXT";
            case 2:
                return "SKIP_1";
            case 3:
                return "SKIP_2";
            case 4:
                return "SKIP_3";
            case 5:
                return "SKIP_4";
            default:
                return "UNKNOWN";
        }
    };

    // Helper to format instid with type and distance
    auto formatInstId = [&](SDelayAluData::InstType type, int8_t distance) -> std::string {
        // If distance is 0, print NO_DEP regardless of type (matches rocisa behavior)
        if (distance == 0) {
            return "NO_DEP";
        }

        std::string result = typeToString(type);

        // SALU uses CYCLE_N, others use DEP_N
        if (type == SDelayAluData::InstType::SALU) {
            result += "_CYCLE_" + std::to_string((int)distance);
        } else if (type != SDelayAluData::InstType::NO_DEP) {
            result += "_DEP_" + std::to_string((int)distance);
        }

        return result;
    };

    // Print InstID0
    os << "instid0(" << formatInstId(delayAluData.instid0Type, delayAluData.instid0Distance) << ")";

    // Print InstID1 if present
    if (delayAluData.hasInstId1) {
        os << " | instskip(" << skipToString(delayAluData.instSkip) << ")";
        os << " | instid1(" << formatInstId(delayAluData.instid1Type, delayAluData.instid1Distance)
           << ")";
    }

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const SWaitAluData& waitAluData) {
    auto emitField = [&](SWaitAluData::Field field, const char* name) {
        if (waitAluData.hasField(field))
            os << " depctr_" << name << "(" << waitAluData.getField(field) << ")";
    };

    emitField(SWaitAluData::VA_VDST, "va_vdst");
    emitField(SWaitAluData::VA_SDST, "va_sdst");
    emitField(SWaitAluData::VA_SSRC, "va_ssrc");
    emitField(SWaitAluData::HOLD_CNT, "hold_cnt");
    emitField(SWaitAluData::VM_VSRC, "vm_vsrc");
    emitField(SWaitAluData::VA_VCC, "va_vcc");
    emitField(SWaitAluData::SA_SDST, "sa_sdst");

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const DSModifiers& dsMod) {
    if (dsMod.na == 1) {
        os << " offset:" << dsMod.offset;
    } else if (dsMod.na == 2) {
        os << " offset0:" << dsMod.offset0 << " offset1:" << dsMod.offset1;
    }
    if (dsMod.gds) {
        os << " gds";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const FLATModifiers& flatMod) {
    if (flatMod.offset12 != 0) {
        os << " offset:" << flatMod.offset12;
    }
    if (flatMod.glc) {
        if (flatMod.hasGLCModifier)
            os << " glc";
        else if (flatMod.hasSC0Modifier)
            os << " sc0";
    }
    if (flatMod.slc) {
        if (flatMod.hasGLCModifier)
            os << " slc";
        else if (flatMod.hasSC0Modifier)
            os << " sc1";
    }
    // gfx12+ FLAT ops use scope:/th: in place of glc/slc/sc0/sc1; the rocisa
    // emitter writes these for cross-CU sync (e.g. flat_atomic_dec_u32 for
    // MBSK GSU), and dropping them on re-emit silently breaks coherence.
    if (flatMod.scope != MUBUFScope::SCOPE_NONE) {
        os << " scope:" << toString(flatMod.scope);
    }
    if (hasTemporalHint(flatMod.th)) {
        os << " th:" << toString(flatMod.th);
    }
    if (flatMod.lds) {
        os << " lds";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const MUBUFModifiers& mubufMod) {
    if (mubufMod.offen) {
        os << " offen offset:" << mubufMod.offset12;
    }
    if (mubufMod.glc) {
        if (mubufMod.hasGLCModifier)
            os << " glc";
        else if (mubufMod.hasSC0Modifier)
            os << " sc0";
    }
    if (mubufMod.slc) {
        if (mubufMod.hasGLCModifier)
            os << " slc";
        else if (mubufMod.hasSC0Modifier)
            os << " sc1";
    }
    if (mubufMod.scope != MUBUFScope::SCOPE_NONE) {
        os << " scope:" << toString(mubufMod.scope);
    }
    // Match rocisa MUBUFModifiers::toString(): gfx1250+ temporal hints replace the
    // legacy nt token when both are present in the modifier bag.
    if (hasTemporalHint(mubufMod.th)) {
        os << " th:" << toString(mubufMod.th, mubufMod.isStore);
    } else if (mubufMod.nt) {
        os << " nt";
    }
    if (mubufMod.lds) {
        os << " lds";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const CacheScopeModifiers& mod) {
    if (mod.scope != MUBUFScope::SCOPE_NONE) {
        os << " scope:" << toString(mod.scope);
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const GLOBALModifiers& mod) {
    if (mod.offset != 0) {
        os << " offset:" << mod.offset;
    }
    // Temporal hint / cache scope for global_prefetch_b8 (gl2-prefetch). Match
    // rocisa GLOBALModifiers::toString(): emit only non-default fields, temporal
    // hint first then scope (e.g. " th:TH_LOAD_NT scope:SCOPE_SE").
    if (hasTemporalHint(mod.th)) {
        os << " th:" << toString(mod.th);
    }
    if (mod.scope != MUBUFScope::SCOPE_NONE) {
        os << " scope:" << toString(mod.scope);
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const SMEMModifiers& smemMod) {
    if (smemMod.offset != 0) {
        os << " offset:" << smemMod.offset;
    }
    if (!smemMod.hasSCOPEModifier && smemMod.glc) {
        os << " glc";
    }
    if (smemMod.nv) {
        os << " nv";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const SDWAModifiers& sdwaMod) {
    auto selectBitToString = [](SDWAModifiers::SelectBit sel) -> const char* {
        switch (sel) {
            case SDWAModifiers::SelectBit::DWORD:
                return "DWORD";
            case SDWAModifiers::SelectBit::BYTE_0:
                return "BYTE_0";
            case SDWAModifiers::SelectBit::BYTE_1:
                return "BYTE_1";
            case SDWAModifiers::SelectBit::BYTE_2:
                return "BYTE_2";
            case SDWAModifiers::SelectBit::BYTE_3:
                return "BYTE_3";
            case SDWAModifiers::SelectBit::WORD_0:
                return "WORD_0";
            case SDWAModifiers::SelectBit::WORD_1:
                return "WORD_1";
            default:
                return nullptr;
        }
    };

    auto unusedBitToString = [](SDWAModifiers::UnusedBit unused) -> const char* {
        switch (unused) {
            case SDWAModifiers::UnusedBit::UNUSED_PAD:
                return "UNUSED_PAD";
            case SDWAModifiers::UnusedBit::UNUSED_SEXT:
                return "UNUSED_SEXT";
            case SDWAModifiers::UnusedBit::UNUSED_PRESERVE:
                return "UNUSED_PRESERVE";
            default:
                return nullptr;
        }
    };

    if (const char* s = selectBitToString(sdwaMod.dst_sel)) os << " dst_sel:" << s;
    if (const char* s = unusedBitToString(sdwaMod.dst_unused)) os << " dst_unused:" << s;
    if (const char* s = selectBitToString(sdwaMod.src0_sel)) os << " src0_sel:" << s;
    if (const char* s = selectBitToString(sdwaMod.src1_sel)) os << " src1_sel:" << s;

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const DPPModifiers& dppMod) {
    if (dppMod.isDPP8) {
        os << " dpp8:[" << (int)dppMod.dpp8[0];
        for (int i = 1; i < 8; ++i) os << "," << (int)dppMod.dpp8[i];
        os << "]";
    } else if (dppMod.dppCtrl != DppCtrl::NONE) {
        os << " " << dppCtrlToAsmStr(dppMod.dppCtrl);
    }
    if (!dppMod.isDPP8) {
        if (dppMod.rowMask != 0xF)
            os << " row_mask:0x" << std::hex << (int)dppMod.rowMask << std::dec;
        if (dppMod.bankMask != 0xF)
            os << " bank_mask:0x" << std::hex << (int)dppMod.bankMask << std::dec;
    }
    if (dppMod.boundCtrl) os << " bound_ctrl:1";
    if (dppMod.fi) os << " fi:1";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const MatrixFmtModifiers& m) {
    // Input formats: matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_BF8
    if (m.fmtA != MatrixFmt::NONE) os << " matrix_a_fmt:" << matrixFmtToStr(m.fmtA);
    if (m.fmtB != MatrixFmt::NONE) os << " matrix_b_fmt:" << matrixFmtToStr(m.fmtB);
    // Scale formats: rocisa emits the raw integer (matrix_a_scale_fmt:2), so
    // match that for byte-for-byte parity in the asm output. The IR (.stir)
    // serializer keeps the symbolic name via matrixScaleFmtToStr().
    if (m.scaleFmtA != MatrixScaleFmt::NONE)
        os << " matrix_a_scale_fmt:" << static_cast<int>(m.scaleFmtA);
    if (m.scaleFmtB != MatrixScaleFmt::NONE)
        os << " matrix_b_scale_fmt:" << static_cast<int>(m.scaleFmtB);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const MFMAModifiers& mfmaMod) {
    // Reuse hints
    if (mfmaMod.reuseA) os << " matrix_a_reuse";
    if (mfmaMod.reuseB) os << " matrix_b_reuse";
    // Neg bits: neg_lo:[1,1] neg_hi:[0,1]
    if (mfmaMod.negBits.hasNegLo()) {
        os << " neg_lo:[" << (int)mfmaMod.negBits.negLo[0];
        for (int i = 1; i < mfmaMod.negBits.numSrcs; ++i)
            os << "," << (int)mfmaMod.negBits.negLo[i];
        os << "]";
    }
    if (mfmaMod.negBits.hasNegHi()) {
        os << " neg_hi:[" << (int)mfmaMod.negBits.negHi[0];
        for (int i = 1; i < mfmaMod.negBits.numSrcs; ++i)
            os << "," << (int)mfmaMod.negBits.negHi[i];
        os << "]";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const VOP3PModifiers& vop3pMod) {
    if (!vop3pMod.op_sel.empty()) {
        os << " op_sel:" << vectorToString(vop3pMod.op_sel);
    }
    if (!vop3pMod.op_sel_hi.empty()) {
        os << " op_sel_hi:" << vectorToString(vop3pMod.op_sel_hi);
    }
    if (!vop3pMod.byte_sel.empty()) {
        os << " byte_sel:" << vectorToString(vop3pMod.byte_sel);
    }
    return os;
}
// NOLINTEND(misc-use-internal-linkage)

static void emitRegister(std::ostream& os, const StinkyRegister& reg,
                         const AsmEmitterOptions& options) {
    switch (reg.dataType) {
        case StinkyRegister::Type::Register: {
            // Check if we should use symbolic name
            bool useSymbolic = options.useSymbolicNames && reg.hasSymbolicName();
            std::string symbolicName = useSymbolic ? reg.getSymbolicName() : "";

            if (reg.reg.isAbs) os << "abs(";
            if (reg.reg.isMinus) os << "-";

            // Emit register: v0, v[0:3], s1, acc0, etc. or v[vgprName+0]
            const std::string regTypeStr = regTypeToString(reg.reg.type);
            os << regTypeStr;

            // Special registers are singletons, no index suffix needed.
            if (reg.reg.type == RegType::VCC || reg.reg.type == RegType::VCC_LO ||
                reg.reg.type == RegType::VCC_HI || reg.reg.type == RegType::EXEC ||
                reg.reg.type == RegType::EXEC_LO || reg.reg.type == RegType::EXEC_HI) {
                break;
            }

            std::string offsetStr = "";
            if (reg.reg.offset != 0) {
                offsetStr = std::to_string(reg.reg.offset);
            }

            if (reg.reg.num > 1) {
                // Register range
                if (useSymbolic) {
                    // Two symbolic-name conventions are accepted here:
                    //   (a) Self-contained range from RawAsmParser, e.g.
                    //       "vgprFoo+0:vgprFoo+3" — already contains the ':'
                    //       separator, so emit it verbatim as "v[<symbolic>]".
                    //   (b) Single-token start name from rocisa, e.g.
                    //       "vgprG2LA+0" — emitter constructs the range as
                    //       "v[<symbolic>:<symbolic>+(num-1)]".
                    if (symbolicName.find(':') != std::string::npos) {
                        os << "[" << symbolicName << "]";
                    } else {
                        os << "[" << symbolicName << offsetStr << ":" << symbolicName << offsetStr
                           << "+" << (reg.reg.num - 1) << "]";
                    }
                } else {
                    // Numeric format: v[46:49]
                    // Note: rocisa could use "v[256-256:259-256]", that's why we add offsetStr to
                    // the end.
                    // TODO: we shouldn't use v[256-256:259-256], it doesn't make sense.
                    os << "[" << reg.reg.idx << offsetStr << ":" << (reg.reg.idx + reg.reg.num - 1)
                       << offsetStr << "]";
                }
            } else {
                // Single register
                if (useSymbolic) {
                    // Symbolic format: v[vgprLocalWriteAddrA+0]
                    // The symbolicName already includes offsets
                    os << "[" << symbolicName << offsetStr << "]";
                } else {
                    if (offsetStr.empty()) {
                        // Numeric format: v10
                        os << reg.reg.idx;
                    } else {
                        // Note: rocisa could use "v[256-256]", that's why we add offsetStr to the
                        // end.
                        // TODO: we shouldn't use v[256-256], it doesn't make sense.
                        os << "[" << reg.reg.idx << offsetStr << "]";
                    }
                }
            }

            if (reg.reg.isAbs) os << ")";
            break;
        }

        case StinkyRegister::Type::LiteralInt:
            os << reg.literalInt;
            break;

        case StinkyRegister::Type::LiteralDouble: {
            // For floating-point literals, always show at least one decimal place
            // Check if it's a whole number
            double value = reg.literalDouble;
            if (value == static_cast<int>(value) && std::abs(value) < 1e10) {
                // It's a whole number - print with .0 suffix
                os << static_cast<int>(value) << ".0";
            } else {
                // Use full precision for non-whole numbers
                // max_digits10 = 17 for double (sufficient to preserve all significant digits)
                os << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
            }
            break;
        }

        case StinkyRegister::Type::LiteralString:
            os << reg.getLiteralString();
            break;

        case StinkyRegister::Type::HwReg:
            HwReg::printOperand(os, reg);
            break;

        case StinkyRegister::Type::Invalid:
            os << "<invalid>";
            break;
    }
}

static void emitMnemonic(std::ostream& os, const StinkyInstruction& inst) {
    // Get mnemonic from the hardware instruction descriptor
    const HwInstDesc* desc = inst.getHwInstDesc();
    if (desc && desc->mnemonic) {
        os << desc->mnemonic;
    } else {
        os << "<unknown>";
    }
}

static bool isVCCType(RegType t) {
    return t == RegType::VCC || t == RegType::VCC_LO || t == RegType::VCC_HI;
}

static bool isEXECType(RegType t) {
    return t == RegType::EXEC || t == RegType::EXEC_LO || t == RegType::EXEC_HI;
}

/// A destination register is implicit (not printed) when it was added
/// solely for dependency tracking.  The instruction's HW flags tell us
/// which special registers are implicit vs encoded as real operands.
static bool isImplicitDest(const StinkyRegister& reg, const StinkyInstruction& inst) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;

    RegType t = reg.reg.type;

    if (t == RegType::SCC) return true;

    if (isEXECType(t) && inst.is(IF_ImplicitWriteEXEC)) return true;

    return false;
}

/// A source register is implicit (not printed) when it was added solely
/// for dependency tracking.
static bool isImplicitSrc(const StinkyRegister& reg, const StinkyInstruction& inst) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;

    RegType t = reg.reg.type;

    if (t == RegType::SCC) return true;

    if (isVCCType(t) && inst.is(IF_ImplicitReadVCC)) return true;

    if (isEXECType(t) && inst.is(IF_ImplicitReadEXEC)) return true;

    return false;
}

static void emitOperands(std::ostream& os, const StinkyInstruction& inst,
                         const AsmEmitterOptions& options) {
    bool firstOperand = true;

    // Check if instruction has True16 modifiers
    const True16Modifiers* true16Mod = inst.getModifier<True16Modifiers>();

    // Emit destination registers (skip pseudo and implicit registers)
    size_t destIndex = 0;
    for (const auto& dest : inst.getDestRegs()) {
        if (isPseudoReg(dest) || isImplicitDest(dest, inst)) continue;

        if (!firstOperand) {
            os << ", ";
        }
        emitRegister(os, dest, options);

        // Append True16 modifier (.h or .l) if present for this destination operand
        if (true16Mod) {
            HighBitSel highBit = HighBitSel::NONE;
            if (destIndex == 0)
                highBit = true16Mod->getDst0();
            else if (destIndex == 1)
                highBit = true16Mod->getDst1();

            if (highBit == HighBitSel::HIGH)
                os << ".h";
            else if (highBit == HighBitSel::LOW)
                os << ".l";
        }

        firstOperand = false;
        destIndex++;
    }

    // Check if instruction has VOP3 modifiers
    const VOP3Modifiers* vop3Mod = inst.getModifier<VOP3Modifiers>();

    // Check if this is a MUBUF instruction (buffer operations) with offen.
    // Note: SOPP fences (global_wb / global_inv) carry a CacheScopeModifiers
    // instead of MUBUFModifiers, so this query correctly returns nullptr for
    // them — the null/0-soffset substitution below is only meaningful for
    // true buffer ops with src registers.
    const MUBUFModifiers* mubufMod = inst.getModifier<MUBUFModifiers>();

    // Compute the number of source operands to emit from the HW field metadata.
    // Read-write operands already emitted via destRegs have duplicates appended
    // at the end of srcRegs for use-def tracking; those must not be re-emitted.
    const auto& srcRegs = inst.getSrcRegs();
    size_t emitSrcCount = srcRegs.size();
    {
        const auto& fields = inst.getHwInstDesc()->operandFields;
        if (!fields.empty()) {
            emitSrcCount = 0;
            for (const auto& f : fields)
                if (!f.isDest) emitSrcCount++;
        }
    }

    // Emit source registers with VOP3 modifiers if present (skip pseudo and implicit registers)
    size_t nonSkippedIndex = 0;  // Track the index of non-skipped operands
    for (size_t i = 0; i < srcRegs.size(); ++i) {
        if (isPseudoReg(srcRegs[i]) || isImplicitSrc(srcRegs[i], inst)) continue;

        if (nonSkippedIndex >= emitSrcCount) break;

        if (!firstOperand) {
            os << ", ";
        }

        // Check if this is the last visible source operand of a MUBUF instruction
        // and it's a literal zero. When HasMUBUFConst is false (e.g. gfx1250),
        // emit "null" instead of "0" for the soffset parameter.
        // When HasMUBUFConst is true (e.g. gfx942), emit the raw value as-is.
        if (mubufMod && !mubufMod->hasMUBUFConst && i == srcRegs.size() - 1) {
            if (srcRegs[i].dataType == StinkyRegister::Type::LiteralInt &&
                srcRegs[i].literalInt == 0) {
                os << "null";
                firstOperand = false;
                nonSkippedIndex++;
                continue;
            }
        }

        bool needsNeg = false;
        bool needsAbs = false;

        // Check VOP3 modifiers for this source operand
        if (vop3Mod) {
            switch (nonSkippedIndex) {  // NOLINT(bugprone-switch-missing-default-case)
                case 0:
                    needsNeg = vop3Mod->neg_src0;
                    needsAbs = vop3Mod->abs_src0;
                    break;
                case 1:
                    needsNeg = vop3Mod->neg_src1;
                    needsAbs = vop3Mod->abs_src1;
                    break;
                case 2:
                    needsNeg = vop3Mod->neg_src2;
                    needsAbs = vop3Mod->abs_src2;
                    break;
            }
        }

        // Emit modifiers according to LLVM syntax rules
        // Negation comes first, then absolute value
        if (needsNeg && needsAbs) {
            // Both neg and abs: -abs(v10) or neg(abs(v10))
            os << "-abs(";
            emitRegister(os, srcRegs[i], options);
            os << ")";
        } else if (needsNeg) {
            // Only negation: -v10 or neg(v10)
            // Use short form "-" before register (LLVM syntax allows this)
            os << "-";
            emitRegister(os, srcRegs[i], options);
        } else if (needsAbs) {
            // Only absolute value: abs(v10) or |v10|
            os << "abs(";
            emitRegister(os, srcRegs[i], options);
            os << ")";
        } else {
            // No modifiers
            emitRegister(os, srcRegs[i], options);
        }

        // Append True16 modifier (.h or .l) if present for this source operand
        if (true16Mod && nonSkippedIndex < true16Mod->getSrcCount()) {
            HighBitSel highBit = true16Mod->getSrc(nonSkippedIndex);
            if (highBit == HighBitSel::HIGH)
                os << ".h";
            else if (highBit == HighBitSel::LOW)
                os << ".l";
        }

        firstOperand = false;
        nonSkippedIndex++;
    }

    // Check if instruction has VOP3P modifiers
    if (const VOP3PModifiers* vop3pMod = inst.getModifier<VOP3PModifiers>()) {
        os << *vop3pMod;
    }
}

static bool emitCustomOperands(std::ostream& os, const StinkyInstruction& inst) {
    switch (inst.getUnifiedOpcode()) {
        case GFX::s_delay_alu: {
            const SDelayAluData* delayAluData = inst.getModifier<SDelayAluData>();
            assert(delayAluData != nullptr && "Internal error: SDelayAluData expected");
            os << " " << *delayAluData;
            return true;
        }

        case GFX::s_wait_alu: {
            const SWaitAluData* waitAluData = inst.getModifier<SWaitAluData>();
            assert(waitAluData != nullptr && "Internal error: SWaitAluData expected");
            os << *waitAluData;
            return true;
        }

        case GFX::s_waitcnt: {
            const SWaitCntData* waitData = inst.getModifier<SWaitCntData>();
            if (!waitData) return false;

            // Reconstruct lgkmcnt and vmcnt from semantic fields.
            // Conversion stores: vlcnt, vscnt, dlcnt(-1), dscnt, kmcnt
            // lgkmcnt = dscnt + kmcnt (only counting fields != -1)
            // vmcnt   = vlcnt + vscnt (only counting fields != -1)
            auto accumulate = [](std::initializer_list<int> vals) -> int {
                bool anySet = false;
                int sum = 0;
                for (int v : vals) {
                    if (v != -1) {
                        sum += v;
                        anySet = true;
                    }
                }
                return anySet ? sum : -1;
            };
            int lgkmcnt = accumulate({waitData->dlcnt, waitData->dscnt, waitData->kmcnt});
            // Note: vlcnt + vscnt is correct for gfx942 (combined vmcnt counter).
            // For SeparateVscnt archs (e.g. gfx1030, gfx1100), vscnt should go
            // to a separate s_waitcnt_vscnt instruction. This works today because
            // _SWaitCnt and _SWaitCntVscnt are converted into separate
            // GFX::s_waitcnt instructions, so vlcnt and vscnt never coexist in
            // the same SWaitCntData. If they are ever merged, this would be
            // incorrect for SeparateVscnt archs.
            int vmcnt = accumulate({waitData->vlcnt, waitData->vscnt});

            // Cap to hardware max values (matching rocisa behavior)
            if (lgkmcnt != -1 && waitData->maxLgkmcnt != -1)
                lgkmcnt = std::min(lgkmcnt, waitData->maxLgkmcnt);
            if (vmcnt != -1 && waitData->maxVmcnt != -1)
                vmcnt = std::min(vmcnt, waitData->maxVmcnt);

            if (lgkmcnt == 0 && vmcnt == 0) {
                os << " 0";
            } else {
                os << " ";
                bool first = true;
                if (lgkmcnt != -1) {
                    os << "lgkmcnt(" << lgkmcnt << ")";
                    first = false;
                }
                if (vmcnt != -1) {
                    if (!first) os << ", ";
                    os << "vmcnt(" << vmcnt << ")";
                }
            }
            return true;
        }

        default:
            return false;
    }
}

// SMEM atomics signal return via glc, not th:, so they are excluded.
static bool needThAtomicReturn(const StinkyInstruction& inst) {
    if (!isFLATAtomic(inst) && !isMUBUFAtomic(inst)) return false;
    for (const auto& d : inst.getDestRegs()) {
        if (!isPseudoReg(d) && !isImplicitDest(d, inst)) return true;
    }
    return false;
}

static void emitTrailingModifiers(std::ostream& os, const StinkyInstruction& inst) {
#define EMIT_TRAILING_MODIFIER(TYPE_ENUM, CLASS_PREFIX)                \
    case Modifier::Type::TYPE_ENUM:                                    \
        os << *static_cast<const CLASS_PREFIX##Modifiers*>(mod.get()); \
        break

    for (const auto& mod : inst.getModifiers()) {
        switch (mod->getType()) {
            EMIT_TRAILING_MODIFIER(DS, DS);
            EMIT_TRAILING_MODIFIER(FLAT, FLAT);
            EMIT_TRAILING_MODIFIER(MUBUF, MUBUF);
            EMIT_TRAILING_MODIFIER(CACHE_SCOPE, CacheScope);
            EMIT_TRAILING_MODIFIER(GLOBAL, GLOBAL);
            EMIT_TRAILING_MODIFIER(SMEM, SMEM);
            EMIT_TRAILING_MODIFIER(SDWA, SDWA);
            EMIT_TRAILING_MODIFIER(DPP, DPP);
            EMIT_TRAILING_MODIFIER(MFMA_DATA, MFMA);
            EMIT_TRAILING_MODIFIER(MATRIX_FMT, MatrixFmt);
            default:
                break;
        }
    }
#undef EMIT_TRAILING_MODIFIER

    if (needThAtomicReturn(inst)) {
        os << " th:TH_ATOMIC_RETURN";
    }
}

static void emitCycleComment(std::ostream& os, const StinkyInstruction& inst, int currentColumn,
                             const AsmEmitterOptions& options) {
    bool needsComment = false;

    // Check if we need to emit cycle info
    if (options.emitCycleInfo) {
        needsComment = true;
    }

    // Check if we need to emit user comment
    const CommentData* comment = nullptr;
    if (options.emitComments) {
        comment = inst.getModifier<CommentData>();
        if (comment && !comment->comment.empty()) {
            needsComment = true;
        }
    }

    // If nothing to emit, return early
    if (!needsComment) {
        return;
    }

    // Pad to comment alignment column if specified
    if (options.commentAlignColumn > 0 && currentColumn < options.commentAlignColumn) {
        int padding = options.commentAlignColumn - currentColumn;
        os << std::string(padding, ' ');
    } else {
        // No alignment, just add a space before comment
        os << " ";
    }

    // Start comment
    os << "//";

    // Emit cycle info first if enabled
    if (options.emitCycleInfo) {
        os << " issue=" << inst.issueCycles << " latency=" << inst.latencyCycles;
    }

    // Emit user comment if enabled and exists
    if (options.emitComments && comment && !comment->comment.empty()) {
        if (options.emitCycleInfo) {
            os << ", ";
        } else {
            os << " ";
        }
        os << comment->comment;
    }
}

static void emitDirective(std::ostream& os, const AsmDirective& directive,
                          const AsmEmitterOptions& options) {
    std::ostringstream dirStream;
    if (directive.kind == AsmDirectiveKind::SET) {
        dirStream << directive.name << " " << directive.symbol;
        if (!directive.value.empty()) {
            dirStream << ", " << directive.value;
        }
    } else if (directive.kind == AsmDirectiveKind::MACRO) {
        dirStream << directive.value;
    } else if (directive.kind == AsmDirectiveKind::TEXTBLOCK) {
        // Output raw text as-is (no newline added since text may already have it)
        os << directive.value;
        return;
    } else if (directive.kind == AsmDirectiveKind::IF ||
               directive.kind == AsmDirectiveKind::ENDIF) {
        os << directive.value;
        return;
    }

    if (!dirStream.str().empty()) {
        if (options.emitComments && !directive.comment.empty()) {
            dirStream << " // " << directive.comment;
        }

        os << dirStream.str();

        if (directive.kind == AsmDirectiveKind::SET) os << "\n";
    }
}

void StinkyAsmEmitter::emit(std::ostream& os, const StinkyInstruction& inst) {
    // Check if this is a label or pseudo PHI (do not emit)
    if (inst.getUnifiedOpcode() == GFX::LABEL) {
        const LabelData* labelData = inst.getModifier<LabelData>();
        if (labelData) {
            if (labelData->alignment > 1) {
                os << ".align " << labelData->alignment << "\n";
            }
            os << labelData->label << ":";
        }

        // Emit comment if present
        if (options.emitComments) {
            const CommentData* comment = inst.getModifier<CommentData>();
            if (comment && !comment->comment.empty()) {
                os << "  /// " << comment->comment;
            }
        }

        os << "\n";
        return;
    }

    if (isPseudoInst(&inst)) return;  // Pseudo instruction: do not emit

    // Track current column position for comment alignment
    std::ostringstream instrStream;

    // Emit indentation
    for (int i = 0; i < options.indent; ++i) {
        instrStream << " ";
    }

    // Emit mnemonic
    emitMnemonic(instrStream, inst);

    if (emitCustomOperands(instrStream, inst)) {
        // Emit custom operands for special instructions, or regular operands if not custom
    } else if (!inst.getDestRegs().empty() || !inst.getSrcRegs().empty()) {
        // Emit regular operands if any
        instrStream << " ";
        emitOperands(instrStream, inst, options);
    }

    // Emit trailing modifiers (memory, s_wait_alu, MFMA)
    emitTrailingModifiers(instrStream, inst);

    // Get the instruction string and its length for comment alignment
    std::string instrStr = instrStream.str();
    int currentColumn = instrStr.length();

    // Write the instruction to the output stream
    os << instrStr;

    // Emit cycle information and/or user comments with alignment
    emitCycleComment(os, inst, currentColumn, options);

    os << "\n";
}

static void emitBasicBlock(std::ostream& os, const BasicBlock& bb, const AsmEmitterOptions& options,
                           StinkyAsmEmitter* emitter) {
    for (const IRBase& ir : bb) {
        if (const StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&ir)) {
            emitter->emit(os, *inst);
            if (options.emitBlankLines && !isPseudoInst(inst)) os << "\n";
        } else if (const AsmDirective* directive = dyn_cast<AsmDirective>(&ir))
            emitDirective(os, *directive, options);
    }
}

void StinkyAsmEmitter::emit(std::ostream& os, const Function& function) {
    for (const BasicBlock& bb : function) emitBasicBlock(os, bb, options, this);
}

std::string StinkyAsmEmitter::emit(const StinkyInstruction& inst) {
    std::ostringstream oss;
    emit(oss, inst);
    return oss.str();
}

std::string StinkyAsmEmitter::emit(const Function& function) {
    std::ostringstream oss;
    emit(oss, function);
    return oss.str();
}

}  // namespace stinkytofu
