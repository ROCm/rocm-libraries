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

#include "ir/asm/StinkyAsmEmitter.hpp"

#include <iomanip>

namespace stinkytofu
{
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

    void StinkyAsmEmitter::emitRegister(std::ostream& os, const StinkyRegister& reg)
    {
        switch(reg.dataType)
        {
        case StinkyRegister::Type::Register:
        {
            // Check if we should use symbolic name
            bool        useSymbolic  = options.useSymbolicNames && reg.hasSymbolicName();
            std::string symbolicName = useSymbolic ? reg.getSymbolicName() : "";

            // Emit register: v0, v[0:3], s1, acc0, etc. or v[vgprName+0]
            os << regTypeToString(reg.reg.type);

            if(reg.reg.num > 1)
            {
                // Register range
                if(useSymbolic)
                {
                    // Symbolic format: v[vgprG2LA+0:vgprG2LA+0+3]
                    // The symbolicName already includes offsets (e.g., "vgprG2LA+0")
                    os << "[" << symbolicName << ":" << symbolicName << "+" << (reg.reg.num - 1)
                       << "]";
                }
                else
                {
                    // Numeric format: v[46:49]
                    os << "[" << reg.reg.idx << ":" << (reg.reg.idx + reg.reg.num - 1) << "]";
                }
            }
            else
            {
                // Single register
                if(useSymbolic)
                {
                    // Symbolic format: v[vgprLocalWriteAddrA+0]
                    // The symbolicName already includes offsets
                    os << "[" << symbolicName << "]";
                }
                else
                {
                    // Numeric format: v10
                    os << reg.reg.idx;
                }
            }
            break;
        }

        case StinkyRegister::Type::LiteralInt:
            os << reg.literalInt;
            break;

        case StinkyRegister::Type::LiteralDouble:
            os << reg.literalDouble;
            break;

        case StinkyRegister::Type::LiteralString:
            os << reg.getLiteralString();
            break;

        case StinkyRegister::Type::Invalid:
            os << "<invalid>";
            break;
        }
    }

    void StinkyAsmEmitter::emitMnemonic(std::ostream& os, const StinkyInstruction& inst)
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

    void StinkyAsmEmitter::emitOperands(std::ostream& os, const StinkyInstruction& inst)
    {
        bool firstOperand = true;

        // Emit destination registers (skip pseudo and implicit registers)
        for(const auto& dest : inst.getDestRegs())
        {
            // Skip pseudo registers and implicit registers
            if(isPseudoRegister(dest) || isImplicitRegister(dest))
                continue;

            if(!firstOperand)
            {
                os << ", ";
            }
            emitRegister(os, dest);
            firstOperand = false;
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

            // Check if this is the last source operand of a MUBUF instruction with offen
            // and it's a literal int 0. In that case, emit "null" instead of "0" for the soffset parameter
            if(mubufMod && i == srcRegs.size() - 1
               && srcRegs[i].dataType == StinkyRegister::Type::LiteralInt
               && srcRegs[i].literalInt == 0)
            {
                os << "null";
                firstOperand = false;
                nonSkippedIndex++;
                continue;
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
                emitRegister(os, srcRegs[i]);
                os << ")";
            }
            else if(needsNeg)
            {
                // Only negation: -v10 or neg(v10)
                // Use short form "-" before register (LLVM syntax allows this)
                os << "-";
                emitRegister(os, srcRegs[i]);
            }
            else if(needsAbs)
            {
                // Only absolute value: abs(v10) or |v10|
                os << "abs(";
                emitRegister(os, srcRegs[i]);
                os << ")";
            }
            else
            {
                // No modifiers
                emitRegister(os, srcRegs[i]);
            }

            firstOperand = false;
            nonSkippedIndex++;
        }
    }

    void StinkyAsmEmitter::emitMemoryModifiers(std::ostream& os, const StinkyInstruction& inst)
    {
        // Emit DS modifiers
        const DSModifiers* dsMod = inst.getModifier<DSModifiers>();
        if(dsMod)
        {
            if(dsMod->na == 1)
            {
                os << " offset:" << dsMod->offset;
            }
            else if(dsMod->na == 2)
            {
                os << " offset0:" << dsMod->offset0 << " offset1:" << dsMod->offset1;
            }
            if(dsMod->gds)
            {
                os << " gds";
            }
            return;
        }

        // Emit FLAT modifiers
        const FLATModifiers* flatMod = inst.getModifier<FLATModifiers>();
        if(flatMod)
        {
            if(flatMod->offset12 != 0)
            {
                os << " offset:" << flatMod->offset12;
            }
            if(flatMod->glc)
            {
                os << " glc";
            }
            if(flatMod->slc)
            {
                os << " slc";
            }
            if(flatMod->lds)
            {
                os << " lds";
            }
            return;
        }

        // Emit MUBUF modifiers
        const MUBUFModifiers* mubufMod = inst.getModifier<MUBUFModifiers>();
        if(mubufMod)
        {
            if(mubufMod->offen)
            {
                os << " offen offset:" << mubufMod->offset12;
            }
            if(mubufMod->glc || mubufMod->slc || mubufMod->lds)
            {
                os << ",";
            }
            if(mubufMod->glc)
            {
                os << " glc";
            }
            if(mubufMod->slc)
            {
                os << " slc";
            }
            if(mubufMod->nt)
            {
                os << " nt";
            }
            if(mubufMod->lds)
            {
                os << " lds";
            }
            return;
        }

        // Emit SMEM modifiers
        const SMEMModifiers* smemMod = inst.getModifier<SMEMModifiers>();
        if(smemMod)
        {
            if(smemMod->offset != 0)
            {
                os << " offset:" << smemMod->offset;
            }
            if(smemMod->glc)
            {
                os << " glc";
            }
            if(smemMod->nv)
            {
                os << " nv";
            }
            return;
        }
    }

    void StinkyAsmEmitter::emitCycleComment(std::ostream& os, const StinkyInstruction& inst, int currentColumn)
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

        // Emit operands if any
        if(!inst.getDestRegs().empty() || !inst.getSrcRegs().empty())
        {
            instrStream << " ";
            emitOperands(instrStream, inst);
        }

        // Emit memory modifiers (DS, FLAT, MUBUF, SMEM)
        emitMemoryModifiers(instrStream, inst);

        // Get the instruction string and its length for comment alignment
        std::string instrStr = instrStream.str();
        int currentColumn = instrStr.length();

        // Write the instruction to the output stream
        os << instrStr;

        // Emit cycle information and/or user comments with alignment
        emitCycleComment(os, inst, currentColumn);

        os << "\n";
    }

    void StinkyAsmEmitter::emit(std::ostream& os, const IRList& irlist)
    {
        for(auto it = irlist.begin(); it != irlist.end(); ++it)
        {
            const StinkyInstruction* inst = cast<StinkyInstruction>(it.getNodePtr());
            if(inst)
            {
                emit(os, *inst);

                if(options.emitBlankLines && inst->getUnifiedOpcode() != GFX::LABEL)
                {
                    os << "\n";
                }
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
