/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#pragma once
#include "instruction.hpp"

#include <string>
#include <vector>

namespace rocisa
{
    struct BranchInstruction : public Instruction
    {
        std::string labelName;

        BranchInstruction(const std::string& labelName, const std::string& comment = "")
            : Instruction(InstType::INST_NOTYPE, comment)
            , labelName(labelName)
        {
        }

        BranchInstruction(const BranchInstruction& other)
            : Instruction(other)
            , labelName(other.labelName)
        {
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {labelName};
        }

        std::vector<InstructionInput> getDstParams() const override
        {
            return {};
        }

        std::vector<InstructionInput> getSrcParams() const override
        {
            return {labelName};
        }

        virtual std::string toString() const override
        {
            return formatWithComment(instStr + " " + labelName);
        }
    };

    struct SBranch : public BranchInstruction
    {
        SBranch(const std::string& labelName, const std::string& comment = "")
            : BranchInstruction(labelName, comment)
        {
            setInst("s_branch");
        }

        SBranch(const SBranch& other)
            : BranchInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SBranch>(*this);
        }
    };

    struct SCBranchSCC0 : public BranchInstruction
    {
        SCBranchSCC0(const std::string& labelName, const std::string& comment = "")
            : BranchInstruction(labelName, comment)
        {
            setInst("s_cbranch_scc0");
        }

        SCBranchSCC0(const SCBranchSCC0& other)
            : BranchInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCBranchSCC0>(*this);
        }
    };

    struct SCBranchSCC1 : public BranchInstruction
    {
        SCBranchSCC1(const std::string& labelName, const std::string& comment = "")
            : BranchInstruction(labelName, comment)
        {
            setInst("s_cbranch_scc1");
        }

        SCBranchSCC1(const SCBranchSCC1& other)
            : BranchInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCBranchSCC1>(*this);
        }
    };

    struct SAddPCI64_SIMM : public BranchInstruction
    {
        SAddPCI64_SIMM(const std::string& src, const std::string& comment = "")
            : BranchInstruction(src, comment)
        {
            setInst("s_add_pc_i64");
        }

        SAddPCI64_SIMM(const SAddPCI64_SIMM& other)
            : BranchInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SAddPCI64_SIMM>(*this);
        }
    };

    struct SCBranchVCCNZ : public BranchInstruction
    {
        SCBranchVCCNZ(const std::string& labelName, const std::string& comment = "")
            : BranchInstruction(labelName, comment)
        {
            setInst("s_cbranch_vccnz");
        }

        SCBranchVCCNZ(const SCBranchVCCNZ& other)
            : BranchInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCBranchVCCNZ>(*this);
        }
    };

    struct SCBranchVCCZ : public BranchInstruction
    {
        SCBranchVCCZ(const std::string& labelName, const std::string& comment = "")
            : BranchInstruction(labelName, comment)
        {
            setInst("s_cbranch_vccz");
        }

        SCBranchVCCZ(const SCBranchVCCZ& other)
            : BranchInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCBranchVCCZ>(*this);
        }
    };

    struct SSetPCB64 : public BranchInstruction
    {
        // Optional long-branch target hint.
        //
        // When non-empty, this names the static label that the surrounding
        // SLongBranch* helper sequence (s_getpc_b64 / s_add_i32 / s_add_u32 /
        // s_addc_u32 / s_setpc_b64) is computing the address for. The
        // StinkyTofu IR converter reads this field to wire a direct CFG edge
        // instead of treating the s_setpc_b64 as an opaque indirect jump.
        //
        // This field is metadata only: it does NOT change toString() nor the
        // emitted assembly. Empty string means "no hint" -- the instruction
        // remains a generic indirect set-PC.
        std::string longBranchLabel;

        SSetPCB64(const std::shared_ptr<Container>& src, const std::string& comment = "")
            : BranchInstruction("", comment)
            , srcs(src)
        {
            setInst("s_setpc_b64");
        }

        SSetPCB64(const SSetPCB64& other)
            : BranchInstruction(other)
            , longBranchLabel(other.longBranchLabel)
            , srcs(other.srcs ? other.srcs->clone() : nullptr)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSetPCB64>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {srcs};
        }

        std::vector<InstructionInput> getDstParams() const override
        {
            return {};
        }

        std::vector<InstructionInput> getSrcParams() const override
        {
            return {srcs};
        }

        std::string toString() const override
        {
            return formatWithComment(instStr + " " + srcs->toString());
        }

    private:
        std::shared_ptr<Container> srcs;
    };

    struct SSwapPCB64 : public BranchInstruction
    {
        SSwapPCB64(const std::shared_ptr<Container>& dst,
                   const std::shared_ptr<Container>& src,
                   const std::string&                comment = "")
            : BranchInstruction("", comment)
            , dst(dst)
            , srcs(src)
        {
            setInst("s_swappc_b64");
        }

        SSwapPCB64(const SSwapPCB64& other)
            : BranchInstruction(other)
            , dst(other.dst ? other.dst->clone() : nullptr)
            , srcs(other.srcs ? other.srcs->clone() : nullptr)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SSwapPCB64>(*this);
        }

        std::vector<InstructionInput> getParams() const override
        {
            return {dst, srcs};
        }

        std::vector<InstructionInput> getDstParams() const override
        {
            return {dst};
        }

        std::vector<InstructionInput> getSrcParams() const override
        {
            return {srcs};
        }

        std::string toString() const override
        {
            return formatWithComment(instStr + " " + dst->toString() + ", " + srcs->toString());
        }

    private:
        std::shared_ptr<Container> dst;
        std::shared_ptr<Container> srcs;
    };

    struct SCBranchExecZ : public BranchInstruction
    {
        SCBranchExecZ(const std::string& labelName, const std::string& comment = "")
            : BranchInstruction(labelName, comment)
        {
            setInst("s_cbranch_execz");
        }

        SCBranchExecZ(const SCBranchExecZ& other)
            : BranchInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCBranchExecZ>(*this);
        }
    };

    struct SCBranchExecNZ : public BranchInstruction
    {
        SCBranchExecNZ(const std::string& labelName, const std::string& comment = "")
            : BranchInstruction(labelName, comment)
        {
            setInst("s_cbranch_execnz");
        }

        SCBranchExecNZ(const SCBranchExecNZ& other)
            : BranchInstruction(other)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<SCBranchExecNZ>(*this);
        }
    };
} // namespace rocisa
