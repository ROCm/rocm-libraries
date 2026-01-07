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
    void StinkyAsmEmitter::emitRegister(std::ostream& os, const StinkyRegister& reg)
    {
        switch(reg.dataType)
        {
        case StinkyRegister::Type::Register:
            // Emit register: v[0], v[0:3], s1, acc[0:15], etc.
            os << regTypeToString(reg.reg.type);
            if(reg.reg.num > 1)
            {
                // Register range: v[0:3]
                os << "[" << reg.reg.idx << ":" << (reg.reg.idx + reg.reg.num - 1) << "]";
            }
            else if(reg.reg.type == RegType::V || reg.reg.type == RegType::ACC
                    || reg.reg.type == RegType::A)
            {
                // Vector/accumulator registers always use brackets: v[0], acc[0]
                os << "[" << reg.reg.idx << "]";
            }
            else
            {
                // Scalar registers: s1, vcc, etc.
                os << reg.reg.idx;
            }
            break;

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

        // Emit destination registers
        for(const auto& dest : inst.getDestRegs())
        {
            if(!firstOperand)
            {
                os << ", ";
            }
            emitRegister(os, dest);
            firstOperand = false;
        }

        // Check if instruction has VOP3 modifiers
        const VOP3Modifiers* vop3Mod = inst.getModifier<VOP3Modifiers>();

        // Emit source registers with VOP3 modifiers if present
        const auto& srcRegs = inst.getSrcRegs();
        for(size_t i = 0; i < srcRegs.size(); ++i)
        {
            if(!firstOperand)
            {
                os << ", ";
            }

            bool needsNeg = false;
            bool needsAbs = false;

            // Check VOP3 modifiers for this source operand
            if(vop3Mod)
            {
                switch(i)
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
        }

        // Check if this instruction has a label (for branch instructions)
        const LabelData* labelData = inst.getModifier<LabelData>();
        if(labelData)
        {
            if(!firstOperand)
            {
                os << ", ";
            }
            os << labelData->label;
            firstOperand = false;
        }
    }

    void StinkyAsmEmitter::emitCycleComment(std::ostream& os, const StinkyInstruction& inst)
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

        // Start comment
        os << " //";

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

        // Emit indentation
        for(int i = 0; i < options.indent; ++i)
        {
            os << " ";
        }

        // Emit mnemonic
        emitMnemonic(os, inst);

        // Emit operands if any
        if(!inst.getDestRegs().empty() || !inst.getSrcRegs().empty()
           || inst.getModifier<LabelData>())
        {
            os << " ";
            emitOperands(os, inst);
        }

        // Emit cycle information and/or user comments
        emitCycleComment(os, inst);

        os << "\n";
    }

    void StinkyAsmEmitter::emit(std::ostream& os, const IRList& irlist)
    {
        if(options.emitComments)
        {
            os << "// ==================================================\n";
            os << "// StinkyTofu Assembly Output\n";
            os << "// Instructions: " << irlist.size() << "\n";
            os << "// ==================================================\n";
            os << "\n";
        }

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
