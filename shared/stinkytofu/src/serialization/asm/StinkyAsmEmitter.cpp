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
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"

#include <iomanip>
#include <limits>
#include <sstream>

namespace stinkytofu
{
    // Forward declarations of static helper functions
    static void
        emitRegister(std::ostream& os, const StinkyRegister& reg, const AsmEmitterOptions& options);
    static void emitMnemonic(std::ostream& os, const StinkyInstruction& inst);
    static void emitOperands(std::ostream&            os,
                             const StinkyInstruction& inst,
                             const AsmEmitterOptions& options);
    static bool emitCustomOperands(std::ostream& os, const StinkyInstruction& inst);
    static void emitTrailingModifiers(std::ostream& os, const StinkyInstruction& inst);
    static void emitCycleComment(std::ostream&            os,
                                 const StinkyInstruction& inst,
                                 int                      currentColumn,
                                 const AsmEmitterOptions& options);
    static void emitDirective(std::ostream&            os,
                              const AsmDirective&      directive,
                              const AsmEmitterOptions& options);

    // Helper function to check if a register is a pseudo register
    // Pseudo registers (BARRIER, DS_WRITE, TENSOR_LOAD, etc.) are used internally
    // for dependency tracking but should not appear in assembly output
    // All pseudo registers are defined after PSEUDO_START in RegisterType.def
    static bool isPseudoRegister(const StinkyRegister& reg)
    {
        if(reg.dataType != StinkyRegister::Type::Register)
            return false;

        return reg.reg.type >= RegType::PSEUDO_START;
    }

    // Helper function to check if a register is an implicit register
    // Implicit registers (SCC, VCC, EXEC, etc.) are set implicitly by instructions
    // and should not be printed in assembly output
    static bool isImplicitRegister(const StinkyRegister& reg)
    {
        if(reg.dataType != StinkyRegister::Type::Register)
            return false;

        return reg.reg.type == RegType::SCC;
    }

    static void
        emitRegister(std::ostream& os, const StinkyRegister& reg, const AsmEmitterOptions& options)
    {
        switch(reg.dataType)
        {
        case StinkyRegister::Type::Register:
        {
            // Check if we should use symbolic name
            bool        useSymbolic  = options.useSymbolicNames && reg.hasSymbolicName();
            std::string symbolicName = useSymbolic ? reg.getSymbolicName() : "";

            // Emit register: v0, v[0:3], s1, acc0, etc. or v[vgprName+0]
            const std::string regTypeStr = regTypeToString(reg.reg.type);
            os << regTypeStr;

            // Special registers are singletons, no index suffix needed.
            if(reg.reg.type == RegType::VCC || reg.reg.type == RegType::VCC_LO
               || reg.reg.type == RegType::VCC_HI || reg.reg.type == RegType::EXEC
               || reg.reg.type == RegType::EXEC_LO || reg.reg.type == RegType::EXEC_HI)
            {
                break;
            }

            std::string offsetStr = "";
            if(reg.reg.offset != 0)
            {
                offsetStr = std::to_string(reg.reg.offset);
            }

            if(reg.reg.num > 1)
            {
                // Register range
                if(useSymbolic)
                {
                    // Symbolic format: v[vgprG2LA+0:vgprG2LA+0+3]
                    // The symbolicName already includes offsets (e.g., "vgprG2LA+0")
                    os << "[" << symbolicName << offsetStr << ":" << symbolicName << offsetStr
                       << "+" << (reg.reg.num - 1) << "]";
                }
                else
                {
                    // Numeric format: v[46:49]
                    // Note: rocisa could use "v[256-256:259-256]", that's why we add offsetStr to the end.
                    // TODO: we shouldn't use v[256-256:259-256], it doesn't make sense.
                    os << "[" << reg.reg.idx << offsetStr << ":" << (reg.reg.idx + reg.reg.num - 1)
                       << offsetStr << "]";
                }
            }
            else
            {
                // Single register
                if(useSymbolic)
                {
                    // Symbolic format: v[vgprLocalWriteAddrA+0]
                    // The symbolicName already includes offsets
                    os << "[" << symbolicName << offsetStr << "]";
                }
                else
                {
                    if(offsetStr.empty())
                    {
                        // Numeric format: v10
                        os << reg.reg.idx;
                    }
                    else
                    {
                        // Note: rocisa could use "v[256-256]", that's why we add offsetStr to the end.
                        // TODO: we shouldn't use v[256-256], it doesn't make sense.
                        os << "[" << reg.reg.idx << offsetStr << "]";
                    }
                }
            }
            break;
        }

        case StinkyRegister::Type::LiteralInt:
            os << reg.literalInt;
            break;

        case StinkyRegister::Type::LiteralDouble:
        {
            // For floating-point literals, always show at least one decimal place
            // Check if it's a whole number
            double value = reg.literalDouble;
            if(value == static_cast<int>(value) && std::abs(value) < 1e10)
            {
                // It's a whole number - print with .0 suffix
                os << static_cast<int>(value) << ".0";
            }
            else
            {
                // Use full precision for non-whole numbers
                // max_digits10 = 17 for double (sufficient to preserve all significant digits)
                os << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
            }
            break;
        }

        case StinkyRegister::Type::LiteralString:
            os << reg.getLiteralString();
            break;

        case StinkyRegister::Type::Invalid:
            os << "<invalid>";
            break;
        }
    }

    static void emitMnemonic(std::ostream& os, const StinkyInstruction& inst)
    {
        // Get mnemonic from the hardware instruction descriptor
        const HwInstDesc* desc = inst.getHwInstDesc();
        if(desc && desc->mnemonic)
        {
            os << desc->mnemonic;
        }
        else
        {
            os << "<unknown>";
        }
    }

    static void emitOperands(std::ostream&            os,
                             const StinkyInstruction& inst,
                             const AsmEmitterOptions& options)
    {
        bool firstOperand = true;

        // Check if instruction has True16 modifiers
        const True16Modifiers* true16Mod = inst.getModifier<True16Modifiers>();

        // Emit destination registers (skip pseudo and implicit registers)
        size_t destIndex = 0;
        for(const auto& dest : inst.getDestRegs())
        {
            // Skip pseudo registers and implicit registers
            if(isPseudoRegister(dest) || isImplicitRegister(dest))
                continue;

            if(!firstOperand)
            {
                os << ", ";
            }
            emitRegister(os, dest, options);

            // Append True16 modifier (.h or .l) if present for this destination operand
            if(true16Mod)
            {
                HighBitSel highBit = HighBitSel::NONE;
                if(destIndex == 0)
                    highBit = true16Mod->getDst0();
                else if(destIndex == 1)
                    highBit = true16Mod->getDst1();

                if(highBit == HighBitSel::HIGH)
                    os << ".h";
                else if(highBit == HighBitSel::LOW)
                    os << ".l";
            }

            firstOperand = false;
            destIndex++;
        }

        // Check if instruction has VOP3 modifiers
        const VOP3Modifiers* vop3Mod = inst.getModifier<VOP3Modifiers>();

        // Check if this is a MUBUF instruction (buffer operations) with offen
        const MUBUFModifiers* mubufMod = inst.getModifier<MUBUFModifiers>();

        // Emit source registers with VOP3 modifiers if present (skip pseudo and implicit registers)
        const auto& srcRegs         = inst.getSrcRegs();
        size_t      nonSkippedIndex = 0; // Track the index of non-skipped operands
        for(size_t i = 0; i < srcRegs.size(); ++i)
        {
            // Skip pseudo registers and implicit registers
            if(isPseudoRegister(srcRegs[i]) || isImplicitRegister(srcRegs[i]))
                continue;

            if(!firstOperand)
            {
                os << ", ";
            }

            // Check if this is the last source operand of a MUBUF instruction
            // and it's a literal zero. In that case, emit "null" instead of "0" for the soffset parameter
            // This matches the AMDGPU ISA convention where buffer instructions use "null" for zero soffset
            bool isMUBUFLastOperand = (mubufMod && i == srcRegs.size() - 1);

            if(isMUBUFLastOperand)
            {
                assert(srcRegs[i].dataType == StinkyRegister::Type::LiteralInt
                       || srcRegs[i].dataType == StinkyRegister::Type::Register
                              && "MUBUF last operand must be an integer or register.");

                if(srcRegs[i].dataType == StinkyRegister::Type::LiteralInt
                   && srcRegs[i].literalInt == 0)
                {
                    os << "null";
                    firstOperand = false;
                    nonSkippedIndex++;
                    continue;
                }
            }

            bool needsNeg = false;
            bool needsAbs = false;

            // Check VOP3 modifiers for this source operand
            if(vop3Mod)
            {
                switch(nonSkippedIndex)
                {
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
            if(needsNeg && needsAbs)
            {
                // Both neg and abs: -abs(v10) or neg(abs(v10))
                os << "-abs(";
                emitRegister(os, srcRegs[i], options);
                os << ")";
            }
            else if(needsNeg)
            {
                // Only negation: -v10 or neg(v10)
                // Use short form "-" before register (LLVM syntax allows this)
                os << "-";
                emitRegister(os, srcRegs[i], options);
            }
            else if(needsAbs)
            {
                // Only absolute value: abs(v10) or |v10|
                os << "abs(";
                emitRegister(os, srcRegs[i], options);
                os << ")";
            }
            else
            {
                // No modifiers
                emitRegister(os, srcRegs[i], options);
            }

            // Append True16 modifier (.h or .l) if present for this source operand
            if(true16Mod && nonSkippedIndex < true16Mod->getSrcCount())
            {
                HighBitSel highBit = true16Mod->getSrc(nonSkippedIndex);
                if(highBit == HighBitSel::HIGH)
                    os << ".h";
                else if(highBit == HighBitSel::LOW)
                    os << ".l";
            }

            firstOperand = false;
            nonSkippedIndex++;
        }

        // Check if instruction has VOP3P modifiers
        if(const VOP3PModifiers* vop3pMod = inst.getModifier<VOP3PModifiers>())
        {
            os << *vop3pMod;
        }
    }

    static bool emitCustomOperands(std::ostream& os, const StinkyInstruction& inst)
    {
        switch(inst.getUnifiedOpcode())
        {
        case GFX::s_delay_alu:
        {
            const SDelayAluData* delayAluData = inst.getModifier<SDelayAluData>();
            assert(delayAluData != nullptr && "Internal error: SDelayAluData expected");
            os << " " << *delayAluData;
            return true;
        }

        case GFX::s_wait_alu:
        {
            const SWaitAluData* waitAluData = inst.getModifier<SWaitAluData>();
            assert(waitAluData != nullptr && "Internal error: SWaitAluData expected");
            os << *waitAluData;
            return true;
        }

        default:
            return false;
        }
    }

    static void emitTrailingModifiers(std::ostream& os, const StinkyInstruction& inst)
    {
#define EMIT_TRAILING_MODIFIER(TYPE_ENUM, CLASS_PREFIX)                \
    case Modifier::Type::TYPE_ENUM:                                    \
        os << *static_cast<const CLASS_PREFIX##Modifiers*>(mod.get()); \
        break

        for(const auto& mod : inst.getModifiers())
        {
            switch(mod->getType())
            {
                EMIT_TRAILING_MODIFIER(DS, DS);
                EMIT_TRAILING_MODIFIER(FLAT, FLAT);
                EMIT_TRAILING_MODIFIER(MUBUF, MUBUF);
                EMIT_TRAILING_MODIFIER(SMEM, SMEM);
                EMIT_TRAILING_MODIFIER(MFMA_DATA, MFMA);
            default:
                break;
            }
        }
#undef EMIT_TRAILING_MODIFIER
    }

    static void emitCycleComment(std::ostream&            os,
                                 const StinkyInstruction& inst,
                                 int                      currentColumn,
                                 const AsmEmitterOptions& options)
    {
        bool needsComment = false;

        // Check if we need to emit cycle info
        if(options.emitCycleInfo)
        {
            needsComment = true;
        }

        // Check if we need to emit user comment
        const CommentData* comment = nullptr;
        if(options.emitComments)
        {
            comment = inst.getModifier<CommentData>();
            if(comment && !comment->comment.empty())
            {
                needsComment = true;
            }
        }

        // If nothing to emit, return early
        if(!needsComment)
        {
            return;
        }

        // Pad to comment alignment column if specified
        if(options.commentAlignColumn > 0 && currentColumn < options.commentAlignColumn)
        {
            int padding = options.commentAlignColumn - currentColumn;
            os << std::string(padding, ' ');
        }
        else
        {
            // No alignment, just add a space before comment
            os << " ";
        }

        // Start comment
        os << "//";

        // Emit cycle info first if enabled
        if(options.emitCycleInfo)
        {
            os << " issue=" << inst.issueCycles << " latency=" << inst.latencyCycles;
        }

        // Emit user comment if enabled and exists
        if(options.emitComments && comment && !comment->comment.empty())
        {
            if(options.emitCycleInfo)
            {
                os << ", ";
            }
            else
            {
                os << " ";
            }
            os << comment->comment;
        }
    }

    static void emitDirective(std::ostream&            os,
                              const AsmDirective&      directive,
                              const AsmEmitterOptions& options)
    {
        std::ostringstream dirStream;
        if(directive.kind == AsmDirectiveKind::SET)
        {
            dirStream << directive.name << " " << directive.symbol;
            if(!directive.value.empty())
            {
                dirStream << ", " << directive.value;
            }
        }
        else if(directive.kind == AsmDirectiveKind::MACRO)
        {
            dirStream << directive.value;
        }
        else if(directive.kind == AsmDirectiveKind::TEXTBLOCK)
        {
            // Output raw text as-is (no newline added since text may already have it)
            os << directive.value;
            return;
        }
        else if(directive.kind == AsmDirectiveKind::IF || directive.kind == AsmDirectiveKind::ENDIF)
        {
            os << directive.value;
            return;
        }

        if(!dirStream.str().empty())
        {
            if(options.emitComments && !directive.comment.empty())
            {
                dirStream << " // " << directive.comment;
            }

            os << dirStream.str();

            if(directive.kind == AsmDirectiveKind::SET)
                os << "\n";
        }
    }

    void StinkyAsmEmitter::emit(std::ostream& os, const StinkyInstruction& inst)
    {
        // Check if this is a label
        if(inst.getUnifiedOpcode() == GFX::LABEL)
        {
            const LabelData* labelData = inst.getModifier<LabelData>();
            if(labelData)
            {
                os << labelData->label << ":";
            }

            // Emit comment if present
            if(options.emitComments)
            {
                const CommentData* comment = inst.getModifier<CommentData>();
                if(comment && !comment->comment.empty())
                {
                    os << "  /// " << comment->comment;
                }
            }

            os << "\n";
            return;
        }

        // Track current column position for comment alignment
        std::ostringstream instrStream;

        // Emit indentation
        for(int i = 0; i < options.indent; ++i)
        {
            instrStream << " ";
        }

        // Emit mnemonic
        emitMnemonic(instrStream, inst);

        if(emitCustomOperands(instrStream, inst))
        {
            // Emit custom operands for special instructions, or regular operands if not custom
        }
        else if(!inst.getDestRegs().empty() || !inst.getSrcRegs().empty())
        {
            // Emit regular operands if any
            instrStream << " ";
            emitOperands(instrStream, inst, options);
        }

        // Emit trailing modifiers (memory, s_wait_alu, MFMA)
        emitTrailingModifiers(instrStream, inst);

        // Get the instruction string and its length for comment alignment
        std::string instrStr      = instrStream.str();
        int         currentColumn = instrStr.length();

        // Write the instruction to the output stream
        os << instrStr;

        // Emit cycle information and/or user comments with alignment
        emitCycleComment(os, inst, currentColumn, options);

        os << "\n";
    }

    void StinkyAsmEmitter::emit(std::ostream& os, const IRList& irlist)
    {
        for(auto it = irlist.begin(); it != irlist.end(); ++it)
        {
            const StinkyInstruction* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if(inst)
            {
                emit(os, *inst);

                if(options.emitBlankLines && inst->getUnifiedOpcode() != GFX::LABEL)
                {
                    os << "\n";
                }
                continue;
            }

            const AsmDirective* directive = dyn_cast<AsmDirective>(it.getNodePtr());
            if(directive)
            {
                emitDirective(os, *directive, options);
            }
        }
    }

    std::string StinkyAsmEmitter::emit(const StinkyInstruction& inst)
    {
        std::ostringstream oss;
        emit(oss, inst);
        return oss.str();
    }

    std::string StinkyAsmEmitter::emit(const IRList& irlist)
    {
        std::ostringstream oss;
        emit(oss, irlist);
        return oss.str();
    }

} // namespace stinkytofu
