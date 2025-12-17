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

#include "stinkytofu.hpp" // Must be first for IRList definition
#include "StinkyTofu.hpp"
#include "ir/asm/StinkyAsmDirectives.hpp"
#include "ir/asm/StinkyAsmEmitter.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyMacro.hpp"
#include "ir/asm/StinkyModifiers.hpp"
#include "ir/asm/StinkySignature.hpp"
#include "isa/gfx/GfxIsa.hpp"

#include <sstream>
#include <stdexcept>

namespace stinkytofu
{
    // ========================================================================
    // PIMPL Implementation Structs
    // ========================================================================

    struct StinkyTofu::Impl
    {
        GfxArchID archID;

        Impl(std::array<int, 3> arch)
            : archID(getGfxArchID(arch[0], arch[1], arch[2]))
        {
            // Validation is done by getGfxArchID via assert
        }

        // Helper method to create a standalone instruction (not yet added to any list)
        StinkyInstruction* createInstruction(GFX opcode, const std::string& instrName)
        {
            const HwInstDesc* desc = getMCIDByUOp(opcode, archID);
            if(!desc)
            {
                throw std::runtime_error("Failed to get instruction descriptor for " + instrName);
            }

            // Create a standalone instruction
            StinkyInstruction* inst = new StinkyInstruction(desc);
            return inst;
        }
    };

    // ========================================================================
    // Helper Functions for Instruction Creation (to reduce boilerplate)
    // ========================================================================
    namespace
    {
        // Unified helper for instructions with 1 dest + variable number of sources
        inline std::vector<StinkyInstruction*> createInst(StinkyTofu::Impl*                  pImpl,
                                                          GFX                                opcode,
                                                          const std::string&                 name,
                                                          const StinkyRegister&              dst,
                                                          const std::vector<StinkyRegister>& srcs,
                                                          const std::string& comment)
        {
            std::vector<StinkyInstruction*> result;
            result.reserve(1);
            StinkyInstruction* inst = pImpl->createInstruction(opcode, name);
            inst->destRegs.push_back(dst);
            for(const auto& src : srcs)
            {
                inst->srcRegs.push_back(src);
            }
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
            return result;
        }

        // Helper for instructions with NO destination (e.g., comparisons)
        inline std::vector<StinkyInstruction*>
            createInstNoDst(StinkyTofu::Impl*                  pImpl,
                            GFX                                opcode,
                            const std::string&                 name,
                            const std::vector<StinkyRegister>& srcs,
                            const std::string&                 comment)
        {
            std::vector<StinkyInstruction*> result;
            result.reserve(1);
            StinkyInstruction* inst = pImpl->createInstruction(opcode, name);
            for(const auto& src : srcs)
            {
                inst->srcRegs.push_back(src);
            }
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
            return result;
        }

        // Helper for branch instructions with labels
        inline std::vector<StinkyInstruction*> createBranch(StinkyTofu::Impl*  pImpl,
                                                            GFX                opcode,
                                                            const std::string& name,
                                                            const std::string& labelName,
                                                            const std::string& comment)
        {
            std::vector<StinkyInstruction*> result;
            result.reserve(1);
            StinkyInstruction* inst = pImpl->createInstruction(opcode, name);
            inst->addModifier<LabelData>(LabelData{Modifier::Type::LABEL_NAME, labelName});
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
            return result;
        }

        // Helper for SMEM load instructions with offset modifier
        inline std::vector<StinkyInstruction*> createSMemLoad(StinkyTofu::Impl*     pImpl,
                                                              GFX                   opcode,
                                                              const std::string&    name,
                                                              const StinkyRegister& dst,
                                                              const StinkyRegister& base,
                                                              int                   offset,
                                                              const std::string&    comment)
        {
            std::vector<StinkyInstruction*> result;
            result.reserve(1);
            StinkyInstruction* inst = pImpl->createInstruction(opcode, name);
            inst->destRegs.push_back(dst);
            inst->srcRegs.push_back(base);
            inst->addModifier<SMEMModifiers>(SMEMModifiers(false, false, offset));
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
            return result;
        }
    } // anonymous namespace

    struct IRListModule::Impl
    {
        std::string         name;
        GfxArchID           archID;
        IRList              irList;
        StinkyInstIRBuilder irBuilder;

        Impl(GfxArchID arch, const std::string& moduleName)
            : name(moduleName)
            , archID(arch)
            , irList()
            , irBuilder(irList, archID)
        {
        }
    };

    // ========================================================================
    // StinkyTofu Public Methods
    // ========================================================================

    StinkyTofu::StinkyTofu(std::array<int, 3> arch)
        : pImpl(std::make_unique<Impl>(arch))
    {
    }

    StinkyTofu::~StinkyTofu() = default;

    std::shared_ptr<IRListModule> StinkyTofu::createIRList(const std::string& name)
    {
        return std::make_shared<IRListModule>(pImpl->archID, name);
    }

    // ========================================================================
    // IRListModule Public Methods
    // ========================================================================

    IRListModule::IRListModule(GfxArchID arch, const std::string& name)
        : pImpl(std::make_unique<Impl>(arch, name))
    {
    }

    IRListModule::~IRListModule() = default;

    std::string IRListModule::getName() const
    {
        return pImpl->name;
    }

    std::vector<StinkyInstruction*> IRListModule::add(const std::vector<StinkyInstruction*>& insts)
    {
        for(auto* inst : insts)
        {
            if(!inst)
            {
                throw std::runtime_error("Cannot add null instruction to module");
            }
            // Add the instruction to the end of the list
            pImpl->irList.push_back(inst);
        }
        return insts;
    }

    std::string IRListModule::emitAssembly(bool emitCycleInfo, bool emitComments) const
    {
        std::stringstream ss;
        AsmEmitterOptions options;
        options.emitCycleInfo = emitCycleInfo;
        options.emitComments  = emitComments;
        StinkyAsmEmitter emitter(options);
        emitter.emit(ss, pImpl->irList);
        return ss.str();
    }

    // ========================================================================
    // Register Helper Functions
    // ========================================================================

    StinkyRegister vgpr(uint32_t idx, uint32_t count)
    {
        return StinkyRegister("v", idx, count);
    }

    StinkyRegister sgpr(uint32_t idx, uint32_t count)
    {
        return StinkyRegister("s", idx, count);
    }

    StinkyRegister acc(uint32_t idx, uint32_t count)
    {
        return StinkyRegister("a", idx, count);
    }

    // ========================================================================
    // Generic MFMA Creation Functions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::createMFMA(const std::string&    instType,
                                                           const std::string&    accType,
                                                           int                   m,
                                                           int                   n,
                                                           int                   k,
                                                           int                   blocks,
                                                           bool                  mfma1k,
                                                           const StinkyRegister& acc,
                                                           const StinkyRegister& a,
                                                           const StinkyRegister& b,
                                                           const StinkyRegister* acc2,
                                                           bool                  neg,
                                                           const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        // Build instruction string based on rocisa's MFMAInstruction::preStr() logic
        std::string instrStr;
        std::string variantStr
            = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);

        // Check if this is MFMA (gfx9xx) or WMMA (gfx12xx) architecture
        bool is_mfma = (pImpl->archID == GfxArchID::gfx942 || pImpl->archID == GfxArchID::gfx950);

        // gfx942/gfx950 support explicit _Nb_ notation for blocks > 1 (HasMFMA_explicitB)
        bool hasExplicitB = is_mfma;

        if(hasExplicitB && !mfma1k && blocks > 1)
        {
            // Use explicit B notation: v_mfma_f32_32x32x4_2b_bf16
            std::string strB = std::to_string(blocks) + "b_";
            instrStr         = "v_mfma_" + accType + "_" + variantStr + "_" + strB + instType;
        }
        else
        {
            // Standard format: v_mfma_f32_32x32x8_bf16 or v_wmma_f32_16x16x32_bf16
            std::string instructionName = is_mfma ? "mfma" : "wmma";
            std::string mfma_1k         = mfma1k ? "_1k" : "";
            instrStr = "v_" + instructionName + "_" + accType + "_" + variantStr + "_" + instType
                       + mfma_1k;
        }

        // Look up the instruction by its mnemonic
        uint16_t isaOpcode = getMnemonicToIsaOpcode(instrStr, pImpl->archID);
        if(isaOpcode == GFX::INVALID)
        {
            throw std::runtime_error("MFMA instruction not found: " + instrStr);
        }

        const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error("Failed to get instruction descriptor for " + instrStr);
        }

        // Create the instruction
        StinkyInstruction* inst = new StinkyInstruction(desc);
        inst->destRegs.push_back(acc);
        inst->srcRegs.push_back(a);
        inst->srcRegs.push_back(b);
        inst->srcRegs.push_back(acc2 ? *acc2 : acc); // Use acc2 if provided, otherwise reuse acc

        // Add comment if provided
        if(!comment.empty())
        {
            inst->addModifier(CommentData(comment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::createMXMFMA(const std::string&    instType,
                                                             const std::string&    accType,
                                                             const std::string&    mxScaleATypeStr,
                                                             const std::string&    mxScaleBTypeStr,
                                                             int                   m,
                                                             int                   n,
                                                             int                   k,
                                                             int                   block,
                                                             const StinkyRegister& acc,
                                                             const StinkyRegister& a,
                                                             const StinkyRegister& b,
                                                             const StinkyRegister& acc2,
                                                             const StinkyRegister& mxsa,
                                                             const StinkyRegister& mxsb,
                                                             bool                  reuseA,
                                                             bool                  reuseB,
                                                             const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        // Build instruction string based on rocisa's MXMFMAInstruction::preStr() logic
        std::string variantStr
            = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
        std::string blkStr = (block == 16) ? "16" : "";
        std::string instrStr
            = "v_wmma_scale" + blkStr + "_" + accType + "_" + variantStr + "_" + instType;

        // Look up the instruction by its mnemonic
        uint16_t isaOpcode = getMnemonicToIsaOpcode(instrStr, pImpl->archID);
        if(isaOpcode == GFX::INVALID)
        {
            throw std::runtime_error("MXMFMA instruction not found: " + instrStr);
        }

        const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error("Failed to get instruction descriptor for " + instrStr);
        }

        // Create the instruction
        StinkyInstruction* inst = new StinkyInstruction(desc);
        inst->destRegs.push_back(acc);
        inst->srcRegs.push_back(a);
        inst->srcRegs.push_back(b);
        inst->srcRegs.push_back(acc2);
        inst->srcRegs.push_back(mxsa);
        inst->srcRegs.push_back(mxsb);

        // TODO: Add modifiers for matrix_a_fmt, matrix_b_fmt, matrix_a_scale_fmt, matrix_b_scale_fmt
        // This would require implementing the corresponding Modifier classes for MX format parameters

        // Add comment if provided
        if(!comment.empty())
        {
            inst->addModifier(CommentData(comment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::createSMFMA(const std::string&    instType,
                                                            const std::string&    accType,
                                                            int                   m,
                                                            int                   n,
                                                            int                   k,
                                                            int                   blocks,
                                                            bool                  mfma1k,
                                                            const StinkyRegister& acc,
                                                            const StinkyRegister& a,
                                                            const StinkyRegister& b,
                                                            const StinkyRegister& metadata,
                                                            bool                  neg,
                                                            const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        // Build instruction string based on rocisa's SMFMAInstruction::preStr() logic
        bool is_smfma = (pImpl->archID == GfxArchID::gfx942 || pImpl->archID == GfxArchID::gfx950);
        std::string instructionName = is_smfma ? "smfmac" : "swmmac";
        std::string variantStr
            = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
        std::string strB = blocks > 1 ? std::to_string(blocks) + "ub_" : "";
        std::string instrStr
            = "v_" + instructionName + "_" + accType + "_" + variantStr + "_" + strB + instType;

        // Look up the instruction by its mnemonic
        uint16_t isaOpcode = getMnemonicToIsaOpcode(instrStr, pImpl->archID);
        if(isaOpcode == GFX::INVALID)
        {
            throw std::runtime_error("SMFMA instruction not found: " + instrStr);
        }

        const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error("Failed to get instruction descriptor for " + instrStr);
        }

        // Create the instruction
        StinkyInstruction* inst = new StinkyInstruction(desc);
        inst->destRegs.push_back(acc);
        inst->srcRegs.push_back(a);
        inst->srcRegs.push_back(b);
        inst->srcRegs.push_back(metadata);

        // TODO: Add modifier for neg_lo if neg is true
        // This would require implementing the corresponding Modifier class for neg_lo

        // Add comment if provided
        if(!comment.empty())
        {
            inst->addModifier(CommentData(comment));
        }

        result.push_back(inst);
        return result;
    }

    // ========================================================================
    // Scalar Arithmetic Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::SAbsI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_abs_i32, "S_ABS_I32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMaxI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_max_i32, "S_MAX_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMaxU32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_max_u32, "S_MAX_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMinI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_min_i32, "S_MIN_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMinU32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_min_u32, "S_MIN_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SAddI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_add_i32, "S_ADD_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SAddU32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_add_u32, "S_ADD_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SAddCU32(const StinkyRegister& dst,
                                                         const StinkyRegister& src0,
                                                         const StinkyRegister& src1,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_addc_u32, "S_ADDC_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMulI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_mul_i32, "S_MUL_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMulHII32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_mul_hi_i32, "S_MUL_HI_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMulHIU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_mul_hi_u32, "S_MUL_HI_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMulLOU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        // Check if s_mul_lo_u32 is supported on this architecture (gfx1250+)
        const HwInstDesc* desc = getMCIDByUOp(GFX::s_mul_lo_u32, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error(
                "SMulLOU32: Instruction 's_mul_lo_u32' is not supported on this architecture. "
                "This instruction is only available on gfx1250 and later.");
        }

        return createInst(
            pImpl.get(), GFX::s_mul_lo_u32, "S_MUL_LO_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSubI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_sub_i32, "S_SUB_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSubU32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_sub_u32, "S_SUB_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSubBU32(const StinkyRegister& dst,
                                                         const StinkyRegister& src0,
                                                         const StinkyRegister& src1,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_subb_u32, "S_SUBB_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSubU64(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        // Check if s_sub_u64 is supported on this architecture (gfx1250+)
        const HwInstDesc* desc = getMCIDByUOp(GFX::s_sub_u64, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error(
                "SSubU64: Instruction 's_sub_u64' is not supported on this architecture. "
                "This instruction is only available on gfx1250 and later.");
        }

        return createInst(pImpl.get(), GFX::s_sub_u64, "S_SUB_U64", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Scalar Bitwise Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::SAndB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_and_b32, "S_AND_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SAndB64(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_and_b64, "S_AND_B64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SAndN2B32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_andn2_b32, "S_ANDN2_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SOrB32(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_or_b32, "S_OR_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SOrB64(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_or_b64, "S_OR_B64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SXorB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_xor_b32, "S_XOR_B32", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Scalar Shift Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::SLShiftLeftB32(const StinkyRegister& dst,
                                                               const StinkyRegister& src,
                                                               const StinkyRegister& shift,
                                                               const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_lshl_b32, "S_LSHL_B32", dst, {src, shift}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLShiftRightB32(const StinkyRegister& dst,
                                                                const StinkyRegister& src,
                                                                const StinkyRegister& shift,
                                                                const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_lshr_b32, "S_LSHR_B32", dst, {src, shift}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLShiftLeftB64(const StinkyRegister& dst,
                                                               const StinkyRegister& src,
                                                               const StinkyRegister& shift,
                                                               const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_lshl_b64, "S_LSHL_B64", dst, {src, shift}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLShiftRightB64(const StinkyRegister& dst,
                                                                const StinkyRegister& src,
                                                                const StinkyRegister& shift,
                                                                const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_lshr_b64, "S_LSHR_B64", dst, {src, shift}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SAShiftRightI32(const StinkyRegister& dst,
                                                                const StinkyRegister& src,
                                                                const StinkyRegister& shift,
                                                                const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_ashr_i32, "S_ASHR_I32", dst, {src, shift}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLShiftLeft1AddU32(const StinkyRegister& dst,
                                                                   const StinkyRegister& src0,
                                                                   const StinkyRegister& src1,
                                                                   const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_lshl1_add_u32, "S_LSHL1_ADD_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLShiftLeft2AddU32(const StinkyRegister& dst,
                                                                   const StinkyRegister& src0,
                                                                   const StinkyRegister& src1,
                                                                   const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_lshl2_add_u32, "S_LSHL2_ADD_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLShiftLeft3AddU32(const StinkyRegister& dst,
                                                                   const StinkyRegister& src0,
                                                                   const StinkyRegister& src1,
                                                                   const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_lshl3_add_u32, "S_LSHL3_ADD_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLShiftLeft4AddU32(const StinkyRegister& dst,
                                                                   const StinkyRegister& src0,
                                                                   const StinkyRegister& src1,
                                                                   const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_lshl4_add_u32, "S_LSHL4_ADD_U32", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Scalar Move/Control Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::SBarrier(const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_barrier, "S_BARRIER");
        if(!comment.empty())
        {
            inst->addModifier(CommentData(comment));
        }
        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMovB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_mov_b32, "S_MOV_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMovB64(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_mov_b64, "S_MOV_B64", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCMovB32(const StinkyRegister& dst,
                                                         const StinkyRegister& src,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_cmov_b32, "S_CMOV_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCMovB64(const StinkyRegister& dst,
                                                         const StinkyRegister& src,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_cmov_b64, "S_CMOV_B64", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCSelectB32(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_cselect_b32, "S_CSELECT_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SGetPCB64(const StinkyRegister& dst,
                                                          const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_getpc_b64, "S_GETPC_B64", dst, {}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSetMask(const StinkyRegister& dst,
                                                         const StinkyRegister& src,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_mov_b64, "S_MOV_B64", dst, {src}, comment);
    }

    // ========================================================================
    // Scalar Bit Manipulation Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::SFf1B32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_ff1_i32_b32, "S_FF1_I32_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SBfmB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_bfm_b32, "S_BFM_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SMovkI32(const StinkyRegister& dst,
                                                         const StinkyRegister& src,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_movk_i32, "S_MOVK_I32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSExtI16toI32(const StinkyRegister& dst,
                                                              const StinkyRegister& src,
                                                              const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_sext_i32_i16, "S_SEXT_I32_I16", dst, {src}, comment);
    }

    // ========================================================================
    // Scalar Exec Mask Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::SAndSaveExecB32(const StinkyRegister& dst,
                                                                const StinkyRegister& src,
                                                                const std::string&    comment)
    {
        // Check if s_and_saveexec_b32 is supported on this architecture (gfx1250+)
        const HwInstDesc* desc = getMCIDByUOp(GFX::s_and_saveexec_b32, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error("SAndSaveExecB32: Instruction 's_and_saveexec_b32' is not "
                                     "supported on this architecture. "
                                     "This instruction is only available on gfx1250 and later.");
        }

        return createInst(
            pImpl.get(), GFX::s_and_saveexec_b32, "S_AND_SAVEEXEC_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SAndSaveExecB64(const StinkyRegister& dst,
                                                                const StinkyRegister& src,
                                                                const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst
            = pImpl->createInstruction(GFX::s_and_saveexec_b64, "S_AND_SAVEEXEC_B64");
        inst->destRegs.push_back(dst);
        inst->srcRegs.push_back(src);
        if(!comment.empty())
        {
            inst->addModifier(CommentData(comment));
        }
        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SOrSaveExecB32(const StinkyRegister& dst,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        // Check if s_or_saveexec_b32 is supported on this architecture (gfx1250+)
        const HwInstDesc* desc = getMCIDByUOp(GFX::s_or_saveexec_b32, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error("SOrSaveExecB32: Instruction 's_or_saveexec_b32' is not "
                                     "supported on this architecture. "
                                     "This instruction is only available on gfx1250 and later.");
        }

        return createInst(
            pImpl.get(), GFX::s_or_saveexec_b32, "S_OR_SAVEEXEC_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SOrSaveExecB64(const StinkyRegister& dst,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_or_saveexec_b64, "S_OR_SAVEEXEC_B64", dst, {src}, comment);
    }

    // ========================================================================
    // Scalar Register Access Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::SGetRegB32(const StinkyRegister& dst,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_getreg_b32, "S_GETREG_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSetRegB32(const StinkyRegister& dst,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_setreg_b32, "S_SETREG_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSetRegIMM32B32(const StinkyRegister& dst,
                                                                const StinkyRegister& src,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::s_setreg_imm32_b32, "S_SETREG_IMM32_B32", dst, {src}, comment);
    }

    // ========================================================================
    // Vector Arithmetic Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VAddF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_add_f16, "V_ADD_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAddF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_add_f32, "V_ADD_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAddF64(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_add_f64, "V_ADD_F64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAddI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_add_i32, "V_ADD_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAddU32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_add_u32, "V_ADD_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAddCOU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_add_co_u32, "V_ADD_CO_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAddCCOU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_addc_co_u32, "V_ADDC_CO_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAddPKF16(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_pk_add_f16, "V_PK_ADD_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAdd3U32(const StinkyRegister& dst,
                                                         const StinkyRegister& src0,
                                                         const StinkyRegister& src1,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_add3_u32, "V_ADD3_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VSubF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_sub_f32, "V_SUB_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VSubI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_sub_i32, "V_SUB_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VSubU32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_sub_u32, "V_SUB_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VSubCoU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_sub_co_u32, "V_SUB_CO_U32", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Vector Multiply Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VMulF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_mul_f16, "V_MUL_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_mul_f32, "V_MUL_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulF64(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_mul_f64, "V_MUL_F64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulPKF16(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_pk_mul_f16, "V_PK_MUL_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulPKF32S(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_pk_mul_f32, "V_PK_MUL_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulLOU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_mul_lo_u32, "V_MUL_LO_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulHII32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_mul_hi_i32, "V_MUL_HI_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulHIU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_mul_hi_u32, "V_MUL_HI_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulI32I24(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_mul_i32_i24, "V_MUL_I32_I24", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulU32U24(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_mul_u32_u24, "V_MUL_U32_U24", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Vector MAC/FMA Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VMacF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_mac_f32, "V_MAC_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VFmaF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const StinkyRegister& src2,
                                                        const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_fma_f16, "V_FMA_F16", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VFmaF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const StinkyRegister& src2,
                                                        const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_fma_f32, "V_FMA_F32", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VFmaF64(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const StinkyRegister& src2,
                                                        const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_fma_f64, "V_FMA_F64", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VFmaPKF16(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const StinkyRegister& src2,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_pk_fma_f16, "V_PK_FMA_F16", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VFmaMixF32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const StinkyRegister& src2,
                                                           const std::string&    comment)
    {
        // Check if v_fma_mix_f32 is supported on this architecture (gfx1250+)
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_fma_mix_f32, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error(
                "VFmaMixF32: Instruction 'v_fma_mix_f32' is not supported on this architecture. "
                "This instruction is only available on gfx1250 and later.");
        }

        return createInst(
            pImpl.get(), GFX::v_fma_mix_f32, "V_FMA_MIX_F32", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMadI32I24(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const StinkyRegister& src2,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_mad_i32_i24, "V_MAD_I32_I24", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMadU32U24(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const StinkyRegister& src2,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_mad_u32_u24, "V_MAD_U32_U24", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMadMixF32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const StinkyRegister& src2,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_mad_mix_f32, "V_MAD_MIX_F32", dst, {src0, src1, src2}, comment);
    }

    // ========================================================================
    // Vector Dot Product Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VDot2CF32F16(const StinkyRegister& dst,
                                                             const StinkyRegister& src0,
                                                             const StinkyRegister& src1,
                                                             const StinkyRegister& src2,
                                                             const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_dot2c_f32_f16, "V_DOT2C_F32_F16", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VDot2F32F16(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const StinkyRegister& src2,
                                                            const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_dot2_f32_f16, "V_DOT2_F32_F16", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VDot2F32BF16(const StinkyRegister& dst,
                                                             const StinkyRegister& src0,
                                                             const StinkyRegister& src1,
                                                             const StinkyRegister& src2,
                                                             const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_dot2_f32_bf16, "V_DOT2_F32_BF16", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VDot2CF32BF16(const StinkyRegister& dst,
                                                              const StinkyRegister& src0,
                                                              const StinkyRegister& src1,
                                                              const StinkyRegister& src2,
                                                              const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::v_dot2c_f32_bf16,
                          "V_DOT2C_F32_BF16",
                          dst,
                          {src0, src1, src2},
                          comment);
    }

    // ========================================================================
    // Vector Transcendental Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VExpF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_exp_f16, "V_EXP_F16", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VExpF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_exp_f32, "V_EXP_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VRcpF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_rcp_f16, "V_RCP_F16", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VRcpF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_rcp_f32, "V_RCP_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VRcpIFlagF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_rcp_iflag_f32, "V_RCP_IFLAG_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VRsqF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_rsq_f16, "V_RSQ_F16", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VRsqF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_rsq_f32, "V_RSQ_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VRsqIFlagF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        // Check if v_rsq_iflag_f32 is supported on this architecture (gfx1250+)
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_rsq_iflag_f32, pImpl->archID);
        if(!desc)
        {
            throw std::runtime_error("VRsqIFlagF32: Instruction 'v_rsq_iflag_f32' is not supported "
                                     "on this architecture. "
                                     "This instruction is only available on gfx1250 and later.");
        }

        return createInst(
            pImpl.get(), GFX::v_rsq_iflag_f32, "V_RSQ_IFLAG_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VRndneF32(const StinkyRegister& dst,
                                                          const StinkyRegister& src,
                                                          const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_rndne_f32, "V_RNDNE_F32", dst, {src}, comment);
    }

    // ========================================================================
    // Vector Min/Max/Med Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VMaxF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_max_f16, "V_MAX_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMaxF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_max_f32, "V_MAX_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMaxF64(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_max_f64, "V_MAX_F64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMaxI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_max_i32, "V_MAX_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMaxPKF16(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_pk_max_f16, "V_PK_MAX_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMinF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_min_f16, "V_MIN_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMinF32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_min_f32, "V_MIN_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMinF64(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_min_f64, "V_MIN_F64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMinI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_min_i32, "V_MIN_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMed3I32(const StinkyRegister& dst,
                                                         const StinkyRegister& src0,
                                                         const StinkyRegister& src1,
                                                         const StinkyRegister& src2,
                                                         const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_med3_i32, "V_MED3_I32", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMed3F32(const StinkyRegister& dst,
                                                         const StinkyRegister& src0,
                                                         const StinkyRegister& src1,
                                                         const StinkyRegister& src2,
                                                         const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_med3_f32, "V_MED3_F32", dst, {src0, src1, src2}, comment);
    }

    // ========================================================================
    // Vector Bitwise Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VAndB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_and_b32, "V_AND_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAndOrB32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const StinkyRegister& src2,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_and_or_b32, "V_AND_OR_B32", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VOrB32(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_or_b32, "V_OR_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VXorB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_xor_b32, "V_XOR_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VNotB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_not_b32, "V_NOT_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VPrngB32(const StinkyRegister& dst,
                                                         const StinkyRegister& src,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_prng_b32, "V_PRNG_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCndMaskB32(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cndmask_b32, "V_CNDMASK_B32", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Vector Shift Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VLShiftLeftB16(const StinkyRegister& dst,
                                                               const StinkyRegister& src0,
                                                               const StinkyRegister& src1,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_lshlrev_b16, "V_LSHLREV_B16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VLShiftLeftB32(const StinkyRegister& dst,
                                                               const StinkyRegister& src0,
                                                               const StinkyRegister& src1,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_lshlrev_b32, "V_LSHLREV_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VLShiftRightB32(const StinkyRegister& dst,
                                                                const StinkyRegister& src0,
                                                                const StinkyRegister& src1,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_lshrrev_b32, "V_LSHRREV_B32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VLShiftLeftB64(const StinkyRegister& dst,
                                                               const StinkyRegister& src0,
                                                               const StinkyRegister& src1,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_lshlrev_b64, "V_LSHLREV_B64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VLShiftRightB64(const StinkyRegister& dst,
                                                                const StinkyRegister& src0,
                                                                const StinkyRegister& src1,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_lshrrev_b64, "V_LSHRREV_B64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAShiftRightI32(const StinkyRegister& dst,
                                                                const StinkyRegister& src0,
                                                                const StinkyRegister& src1,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_ashrrev_i32, "V_ASHRREV_I32", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Vector Move/Utility Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VMovB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_mov_b32, "V_MOV_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VSwapB32(const StinkyRegister& dst,
                                                         const StinkyRegister& src,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_swap_b32, "V_SWAP_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VPackF16toB32(const StinkyRegister& dst,
                                                              const StinkyRegister& src0,
                                                              const StinkyRegister& src1,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_pack_b32_f16, "V_PACK_B32_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VPermB32(const StinkyRegister& dst,
                                                         const StinkyRegister& src0,
                                                         const StinkyRegister& src1,
                                                         const StinkyRegister& src2,
                                                         const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_perm_b32, "V_PERM_B32", dst, {src0, src1, src2}, comment);
    }

    // ========================================================================
    // Vector Bit Field Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VBfeI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const StinkyRegister& src2,
                                                        const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_bfe_i32, "V_BFE_I32", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VBfeU32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const StinkyRegister& src2,
                                                        const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_bfe_u32, "V_BFE_U32", dst, {src0, src1, src2}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VBfiB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const StinkyRegister& src2,
                                                        const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_bfi_b32, "V_BFI_B32", dst, {src0, src1, src2}, comment);
    }

    // ========================================================================
    // Vector Accumulator Access Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VAccvgprReadB32(const StinkyRegister& dst,
                                                                const StinkyRegister& src,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_accvgpr_read_b32, "V_ACCVGPR_READ_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAccvgprWrite(const StinkyRegister& dst,
                                                              const StinkyRegister& src,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_accvgpr_write_b32, "V_ACCVGPR_WRITE", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAccvgprWriteB32(const StinkyRegister& dst,
                                                                 const StinkyRegister& src,
                                                                 const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_accvgpr_write_b32, "V_ACCVGPR_WRITE_B32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VReadfirstlaneB32(const StinkyRegister& dst,
                                                                  const StinkyRegister& src,
                                                                  const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_readfirstlane_b32, "V_READFIRSTLANE_B32", dst, {src}, comment);
    }

    // ========================================================================
    // Branch Instructions
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::SBranch(const std::string& labelName,
                                                        const std::string& comment)
    {
        return createBranch(pImpl.get(), GFX::s_branch, "S_BRANCH", labelName, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCBranchSCC0(const std::string& labelName,
                                                             const std::string& comment)
    {
        return createBranch(pImpl.get(), GFX::s_cbranch_scc0, "S_CBRANCH_SCC0", labelName, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCBranchSCC1(const std::string& labelName,
                                                             const std::string& comment)
    {
        return createBranch(pImpl.get(), GFX::s_cbranch_scc1, "S_CBRANCH_SCC1", labelName, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCBranchVCCNZ(const std::string& labelName,
                                                              const std::string& comment)
    {
        return createBranch(
            pImpl.get(), GFX::s_cbranch_vccnz, "S_CBRANCH_VCCNZ", labelName, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCBranchVCCZ(const std::string& labelName,
                                                             const std::string& comment)
    {
        return createBranch(pImpl.get(), GFX::s_cbranch_vccz, "S_CBRANCH_VCCZ", labelName, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSetPCB64(const StinkyRegister& src,
                                                          const std::string&    comment)
    {
        return createInstNoDst(pImpl.get(), GFX::s_setpc_b64, "S_SETPC_B64", {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSwapPCB64(const StinkyRegister& dst,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_swappc_b64, "S_SWAPPC_B64", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCBranchExecZ(const std::string& labelName,
                                                              const std::string& comment)
    {
        return createBranch(
            pImpl.get(), GFX::s_cbranch_execz, "S_CBRANCH_EXECZ", labelName, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCBranchExecNZ(const std::string& labelName,
                                                               const std::string& comment)
    {
        return createBranch(
            pImpl.get(), GFX::s_cbranch_execnz, "S_CBRANCH_EXECNZ", labelName, comment);
    }
    // ========================================================================
    // Compare Instructions (from cmp.hpp)
    // ========================================================================

    // Scalar Compare Instructions
    std::vector<StinkyInstruction*> StinkyTofu::SCmpEQI32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_eq_i32, "S_CMP_EQ_I32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpEQU32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_eq_u32, "S_CMP_EQ_U32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpEQU64(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_eq_u64, "S_CMP_EQ_U64", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpGeI32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_ge_i32, "S_CMP_GE_I32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpGeU32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_ge_u32, "S_CMP_GE_U32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpGtI32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_gt_i32, "S_CMP_GT_I32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpGtU32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_gt_u32, "S_CMP_GT_U32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpLeI32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_le_i32, "S_CMP_LE_I32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpLeU32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_le_u32, "S_CMP_LE_U32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpLgU32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_lg_u32, "S_CMP_LG_U32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpLgI32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_lg_i32, "S_CMP_LG_I32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpLgU64(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_lg_u64, "S_CMP_LG_U64", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpLtI32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_lt_i32, "S_CMP_LT_I32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SCmpLtU32(const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_cmp_lt_u32, "S_CMP_LT_U32", {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SBitcmp1B32(const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_bitcmp1_b32, "S_BITCMP1_B32", {src0, src1}, comment);
    }

    // Scalar Compare with Immediate (SCMPK)
    std::vector<StinkyInstruction*>
        StinkyTofu::SCmpKEQU32(const StinkyRegister& src, int simm16, const std::string& comment)
    {
        return createInstNoDst(pImpl.get(),
                               GFX::s_cmpk_eq_u32,
                               "S_CMPK_EQ_U32",
                               {src, StinkyRegister(simm16)},
                               comment);
    }

    std::vector<StinkyInstruction*>
        StinkyTofu::SCmpKGeU32(const StinkyRegister& src, int simm16, const std::string& comment)
    {
        return createInstNoDst(pImpl.get(),
                               GFX::s_cmpk_ge_u32,
                               "S_CMPK_GE_U32",
                               {src, StinkyRegister(simm16)},
                               comment);
    }

    std::vector<StinkyInstruction*>
        StinkyTofu::SCmpKGtU32(const StinkyRegister& src, int simm16, const std::string& comment)
    {
        return createInstNoDst(pImpl.get(),
                               GFX::s_cmpk_gt_u32,
                               "S_CMPK_GT_U32",
                               {src, StinkyRegister(simm16)},
                               comment);
    }

    std::vector<StinkyInstruction*>
        StinkyTofu::SCmpKLGU32(const StinkyRegister& src, int simm16, const std::string& comment)
    {
        return createInstNoDst(pImpl.get(),
                               GFX::s_cmpk_lg_u32,
                               "S_CMPK_LG_U32",
                               {src, StinkyRegister(simm16)},
                               comment);
    }

    // Vector Compare Instructions
    std::vector<StinkyInstruction*> StinkyTofu::VCmpEQF32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_eq_f32, "V_CMP_EQ_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpEQF64(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_eq_f64, "V_CMP_EQ_F64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpEQU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_eq_u32, "V_CMP_EQ_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpEQI32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_eq_i32, "V_CMP_EQ_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGEF16(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_ge_f16, "V_CMP_GE_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGTF16(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_gt_f16, "V_CMP_GT_F16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGEF32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_ge_f32, "V_CMP_GE_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGTF32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_gt_f32, "V_CMP_GT_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGEF64(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_ge_f64, "V_CMP_GE_F64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGTF64(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_gt_f64, "V_CMP_GT_F64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGEI32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_ge_i32, "V_CMP_GE_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGTI32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_gt_i32, "V_CMP_GT_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGEU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_ge_u32, "V_CMP_GE_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpGtU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_gt_u32, "V_CMP_GT_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpLeU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_le_u32, "V_CMP_LE_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpLeI32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_le_i32, "V_CMP_LE_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpLtI32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_lt_i32, "V_CMP_LT_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpLtU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_lt_u32, "V_CMP_LT_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpUF32(const StinkyRegister& dst,
                                                         const StinkyRegister& src0,
                                                         const StinkyRegister& src1,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cmp_u_f32, "V_CMP_U_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpNeI32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_ne_i32, "V_CMP_NE_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpNeU32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_ne_u32, "V_CMP_NE_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpNeU64(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_ne_u64, "V_CMP_NE_U64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpClassF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src0,
                                                             const StinkyRegister& src1,
                                                             const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmp_class_f32, "V_CMP_CLASS_F32", dst, {src0, src1}, comment);
    }

    // Vector Compare with EXEC modification (VCmpX)
    std::vector<StinkyInstruction*> StinkyTofu::VCmpXClassF32(const StinkyRegister& dst,
                                                              const StinkyRegister& src0,
                                                              const StinkyRegister& src1,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_class_f32, "V_CMPX_CLASS_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXEqU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_eq_u32, "V_CMPX_EQ_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXGeU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_ge_u32, "V_CMPX_GE_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXGtU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_gt_u32, "V_CMPX_GT_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXLeU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_le_u32, "V_CMPX_LE_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXLeI32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_le_i32, "V_CMPX_LE_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXLtF32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_lt_f32, "V_CMPX_LT_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXLtI32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_lt_i32, "V_CMPX_LT_I32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXLtU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_lt_u32, "V_CMPX_LT_U32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXLtU64(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_lt_u64, "V_CMPX_LT_U64", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXNeU16(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_ne_u16, "V_CMPX_NE_U16", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCmpXNeU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cmpx_ne_u32, "V_CMPX_NE_U32", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Conversion Instructions (from cvt.hpp)
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VCvtF16toF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_f32_f16, "V_CVT_F32_F16", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtF32toF16(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_f16_f32, "V_CVT_F16_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtF32toU32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_u32_f32, "V_CVT_U32_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtU32toF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_f32_u32, "V_CVT_F32_U32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtI32toF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_f32_i32, "V_CVT_F32_I32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtF32toI32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_i32_f32, "V_CVT_I32_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtFP8toF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_f32_fp8, "V_CVT_F32_FP8", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtBF8toF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_f32_bf8, "V_CVT_F32_BF8", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtPkFP8toF32(const StinkyRegister& dst,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cvt_pk_f32_fp8, "V_CVT_PK_F32_FP8", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtPkBF8toF32(const StinkyRegister& dst,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cvt_pk_f32_bf8, "V_CVT_PK_F32_BF8", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtPkF32toFP8(const StinkyRegister& dst,
                                                               const StinkyRegister& src0,
                                                               const StinkyRegister& src1,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cvt_pk_fp8_f32, "V_CVT_PK_FP8_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtPkF32toBF8(const StinkyRegister& dst,
                                                               const StinkyRegister& src0,
                                                               const StinkyRegister& src1,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cvt_pk_bf8_f32, "V_CVT_PK_BF8_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtSRF32toFP8(const StinkyRegister& dst,
                                                               const StinkyRegister& src0,
                                                               const StinkyRegister& src1,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cvt_sr_fp8_f32, "V_CVT_SR_FP8_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtSRF32toBF8(const StinkyRegister& dst,
                                                               const StinkyRegister& src0,
                                                               const StinkyRegister& src1,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cvt_sr_bf8_f32, "V_CVT_SR_BF8_F32", dst, {src0, src1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtScalePkFP8toF16(const StinkyRegister& dst,
                                                                    const StinkyRegister& src0,
                                                                    const StinkyRegister& src1,
                                                                    const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::v_cvt_scalef32_pk_f16_fp8,
                          "V_CVT_SCALEF32_PK_F16_FP8",
                          dst,
                          {src0, src1},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtScalePkBF8toF16(const StinkyRegister& dst,
                                                                    const StinkyRegister& src0,
                                                                    const StinkyRegister& src1,
                                                                    const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::v_cvt_scalef32_pk_f16_bf8,
                          "V_CVT_SCALEF32_PK_F16_BF8",
                          dst,
                          {src0, src1},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtScaleFP8toF16(const StinkyRegister& dst,
                                                                  const StinkyRegister& src0,
                                                                  const StinkyRegister& src1,
                                                                  const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::v_cvt_scalef32_f16_fp8,
                          "V_CVT_SCALEF32_F16_FP8",
                          dst,
                          {src0, src1},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtScalePkF16toFP8(const StinkyRegister& dst,
                                                                    const StinkyRegister& src0,
                                                                    const StinkyRegister& src1,
                                                                    const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::v_cvt_scalef32_pk_fp8_f16,
                          "V_CVT_SCALEF32_PK_FP8_F16",
                          dst,
                          {src0, src1},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtScalePkF16toBF8(const StinkyRegister& dst,
                                                                    const StinkyRegister& src0,
                                                                    const StinkyRegister& src1,
                                                                    const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::v_cvt_scalef32_pk_bf8_f16,
                          "V_CVT_SCALEF32_PK_BF8_F16",
                          dst,
                          {src0, src1},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtScaleSRF16toFP8(const StinkyRegister& dst,
                                                                    const StinkyRegister& src0,
                                                                    const StinkyRegister& src1,
                                                                    const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::v_cvt_scalef32_sr_fp8_f16,
                          "V_CVT_SCALEF32_SR_FP8_F16",
                          dst,
                          {src0, src1},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtScaleSRF16toBF8(const StinkyRegister& dst,
                                                                    const StinkyRegister& src0,
                                                                    const StinkyRegister& src1,
                                                                    const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::v_cvt_scalef32_sr_bf8_f16,
                          "V_CVT_SCALEF32_SR_BF8_F16",
                          dst,
                          {src0, src1},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtBF16toF32(const StinkyRegister& dst,
                                                              const StinkyRegister& src,
                                                              const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::v_cvt_f32_bf16, "V_CVT_F32_BF16", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::VCvtPkF32toBF16(const StinkyRegister& dst,
                                                                const StinkyRegister& src0,
                                                                const StinkyRegister& src1,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::v_cvt_pk_bf16_f32, "V_CVT_PK_BF16_F32", dst, {src0, src1}, comment);
    }

    // ========================================================================
    // Memory Instructions (from mem.hpp)
    // ========================================================================

    // DS (LDS) Instructions

    std::vector<StinkyInstruction*> StinkyTofu::DSReadU8(const StinkyRegister& dst,
                                                         const StinkyRegister& addr,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::ds_read_u8, "DS_READ_U8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadI8(const StinkyRegister& dst,
                                                         const StinkyRegister& addr,
                                                         const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::ds_read_i8, "DS_READ_I8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadU16(const StinkyRegister& dst,
                                                          const StinkyRegister& addr,
                                                          const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::ds_read_u16, "DS_READ_U16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadI16(const StinkyRegister& dst,
                                                          const StinkyRegister& addr,
                                                          const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::ds_read_i16, "DS_READ_I16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadB32(const StinkyRegister& dst,
                                                          const StinkyRegister& addr,
                                                          const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::ds_read_b32, "DS_READ_B32", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadB64(const StinkyRegister& dst,
                                                          const StinkyRegister& addr,
                                                          const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::ds_read_b64, "DS_READ_B64", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadB96(const StinkyRegister& dst,
                                                          const StinkyRegister& addr,
                                                          const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::ds_read_b96, "DS_READ_B96", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadB128(const StinkyRegister& dst,
                                                           const StinkyRegister& addr,
                                                           const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::ds_read_b128, "DS_READ_B128", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadU8D16Hi(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_read_u8_d16_hi, "DS_READ_U8_D16_HI", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadU16D16Hi(const StinkyRegister& dst,
                                                               const StinkyRegister& addr,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_read_u16_d16_hi, "DS_READ_U16_D16_HI", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadB64TrB4(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_read_b64_tr_b4, "DS_READ_B64_TR_B4", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadB96TrB6(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_read_b96_tr_b6, "DS_READ_B96_TR_B6", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadB64TrB8(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_read_b64_tr_b8, "DS_READ_B64_TR_B8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSReadB64TrB16(const StinkyRegister& dst,
                                                               const StinkyRegister& addr,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_read_b64_tr_b16, "DS_READ_B64_TR_B16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSRead2B32(const StinkyRegister& dst,
                                                           const StinkyRegister& addr0,
                                                           const StinkyRegister& addr1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_read2_b32, "DS_READ2_B32", dst, {addr0, addr1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSRead2B64(const StinkyRegister& dst,
                                                           const StinkyRegister& addr0,
                                                           const StinkyRegister& addr1,
                                                           const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_read2_b64, "DS_READ2_B64", dst, {addr0, addr1}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWriteB8(const StinkyRegister& addr,
                                                          const StinkyRegister& src,
                                                          const std::string&    comment)
    {
        return createInstNoDst(pImpl.get(), GFX::ds_write_b8, "DS_WRITE_B8", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWriteB16(const StinkyRegister& addr,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write_b16, "DS_WRITE_B16", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWriteB32(const StinkyRegister& addr,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write_b32, "DS_WRITE_B32", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWriteB64(const StinkyRegister& addr,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write_b64, "DS_WRITE_B64", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWriteB96(const StinkyRegister& addr,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write_b96, "DS_WRITE_B96", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWriteB128(const StinkyRegister& addr,
                                                            const StinkyRegister& src,
                                                            const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write_b128, "DS_WRITE_B128", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWriteB8D16Hi(const StinkyRegister& addr,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write_b8_d16_hi, "DS_WRITE_B8_D16_HI", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWriteB16D16Hi(const StinkyRegister& addr,
                                                                const StinkyRegister& src,
                                                                const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write_b16_d16_hi, "DS_WRITE_B16_D16_HI", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWrite2B32(const StinkyRegister& addr0,
                                                            const StinkyRegister& addr1,
                                                            const StinkyRegister& src,
                                                            const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write2_b32, "DS_WRITE2_B32", {addr0, addr1, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSWrite2B64(const StinkyRegister& addr0,
                                                            const StinkyRegister& addr1,
                                                            const StinkyRegister& src,
                                                            const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::ds_write2_b64, "DS_WRITE2_B64", {addr0, addr1, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::DSBPermuteB32(const StinkyRegister& dst,
                                                              const StinkyRegister& src,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::ds_bpermute_b32, "DS_BPERMUTE_B32", dst, {src}, comment);
    }

    // Buffer (MUBUF) Instructions

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadU8(const StinkyRegister& dst,
                                                             const StinkyRegister& addr,
                                                             const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_ubyte, "BUFFER_LOAD_U8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadI8(const StinkyRegister& dst,
                                                             const StinkyRegister& addr,
                                                             const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_sbyte, "BUFFER_LOAD_I8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadU16(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_ushort, "BUFFER_LOAD_U16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadI16(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_sshort, "BUFFER_LOAD_I16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadB32(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_dword, "BUFFER_LOAD_DWORD", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadB64(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_dwordx2, "BUFFER_LOAD_DWORDX2", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadB96(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_dwordx3, "BUFFER_LOAD_DWORDX3", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadB128(const StinkyRegister& dst,
                                                               const StinkyRegister& addr,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_dwordx4, "BUFFER_LOAD_DWORDX4", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadD16U8(const StinkyRegister& dst,
                                                                const StinkyRegister& addr,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_ubyte_d16, "BUFFER_LOAD_D16_U8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadD16HIU8(const StinkyRegister& dst,
                                                                  const StinkyRegister& addr,
                                                                  const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::buffer_load_ubyte_d16_hi,
                          "BUFFER_LOAD_D16HI_U8",
                          dst,
                          {addr},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadD16I8(const StinkyRegister& dst,
                                                                const StinkyRegister& addr,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_sbyte_d16, "BUFFER_LOAD_D16_I8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadD16HII8(const StinkyRegister& dst,
                                                                  const StinkyRegister& addr,
                                                                  const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::buffer_load_sbyte_d16_hi,
                          "BUFFER_LOAD_D16HI_I8",
                          dst,
                          {addr},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadD16B16(const StinkyRegister& dst,
                                                                 const StinkyRegister& addr,
                                                                 const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_load_short_d16, "BUFFER_LOAD_D16_B16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferLoadD16HIB16(const StinkyRegister& dst,
                                                                   const StinkyRegister& addr,
                                                                   const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::buffer_load_short_d16_hi,
                          "BUFFER_LOAD_D16HI_B16",
                          dst,
                          {addr},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferStoreB8(const StinkyRegister& addr,
                                                              const StinkyRegister& src,
                                                              const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::buffer_store_byte, "BUFFER_STORE_B8", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferStoreD16HIU8(const StinkyRegister& addr,
                                                                   const StinkyRegister& src,
                                                                   const std::string&    comment)
    {
        return createInstNoDst(pImpl.get(),
                               GFX::buffer_store_byte_d16_hi,
                               "BUFFER_STORE_D16HI_U8",
                               {addr, src},
                               comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferStoreB16(const StinkyRegister& addr,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::buffer_store_short, "BUFFER_STORE_B16", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferStoreD16HIB16(const StinkyRegister& addr,
                                                                    const StinkyRegister& src,
                                                                    const std::string&    comment)
    {
        return createInstNoDst(pImpl.get(),
                               GFX::buffer_store_short_d16_hi,
                               "BUFFER_STORE_D16HI_B16",
                               {addr, src},
                               comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferStoreB32(const StinkyRegister& addr,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::buffer_store_dword, "BUFFER_STORE_DWORD", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferStoreB64(const StinkyRegister& addr,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::buffer_store_dwordx2, "BUFFER_STORE_DWORDX2", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferStoreB96(const StinkyRegister& addr,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::buffer_store_dwordx3, "BUFFER_STORE_DWORDX3", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferStoreB128(const StinkyRegister& addr,
                                                                const StinkyRegister& src,
                                                                const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::buffer_store_dwordx4, "BUFFER_STORE_DWORDX4", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferAtomicAddF32(const StinkyRegister& dst,
                                                                   const StinkyRegister& src,
                                                                   const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::buffer_atomic_add_f32, "BUFFER_ATOMIC_ADD_F32", dst, {src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferAtomicCmpSwap(const StinkyRegister& dst,
                                                                    const StinkyRegister& addr0,
                                                                    const StinkyRegister& addr1,
                                                                    const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::buffer_atomic_cmpswap,
                          "BUFFER_ATOMIC_CMPSWAP",
                          dst,
                          {addr0, addr1},
                          comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::BufferAtomicCmpSwapX2(const StinkyRegister& dst,
                                                                      const StinkyRegister& addr0,
                                                                      const StinkyRegister& addr1,
                                                                      const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::buffer_atomic_cmpswap_x2,
                          "BUFFER_ATOMIC_CMPSWAP_X2",
                          dst,
                          {addr0, addr1},
                          comment);
    }

    // Scalar Memory (SMEM) Instructions

    // Scalar Memory Load Instructions (size-based naming, architecture-agnostic)
    std::vector<StinkyInstruction*> StinkyTofu::SLoadB32(const StinkyRegister& dst,
                                                         const StinkyRegister& base,
                                                         int                   offset,
                                                         const std::string&    comment)
    {
        return createSMemLoad(
            pImpl.get(), GFX::s_load_dword, "S_LOAD_B32", dst, base, offset, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLoadB64(const StinkyRegister& dst,
                                                         const StinkyRegister& base,
                                                         int                   offset,
                                                         const std::string&    comment)
    {
        return createSMemLoad(
            pImpl.get(), GFX::s_load_dwordx2, "S_LOAD_B64", dst, base, offset, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLoadB128(const StinkyRegister& dst,
                                                          const StinkyRegister& base,
                                                          int                   offset,
                                                          const std::string&    comment)
    {
        return createSMemLoad(
            pImpl.get(), GFX::s_load_dwordx4, "S_LOAD_B128", dst, base, offset, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLoadB256(const StinkyRegister& dst,
                                                          const StinkyRegister& base,
                                                          int                   offset,
                                                          const std::string&    comment)
    {
        return createSMemLoad(
            pImpl.get(), GFX::s_load_dwordx8, "S_LOAD_B256", dst, base, offset, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SLoadB512(const StinkyRegister& dst,
                                                          const StinkyRegister& base,
                                                          int                   offset,
                                                          const std::string&    comment)
    {
        return createSMemLoad(
            pImpl.get(), GFX::s_load_dwordx16, "S_LOAD_B512", dst, base, offset, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SStoreB32(const StinkyRegister& addr,
                                                          const StinkyRegister& src,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_store_dword, "S_STORE_DWORD", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SStoreB64(const StinkyRegister& addr,
                                                          const StinkyRegister& src,
                                                          const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_store_dwordx2, "S_STORE_DWORDX2", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SStoreB128(const StinkyRegister& addr,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::s_store_dwordx4, "S_STORE_DWORDX4", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::SAtomicDec(const StinkyRegister& dst,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::s_atomic_dec, "S_ATOMIC_DEC", dst, {src}, comment);
    }

    // Flat Memory Instructions

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadU8(const StinkyRegister& dst,
                                                           const StinkyRegister& addr,
                                                           const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::flat_load_ubyte, "FLAT_LOAD_U8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadI8(const StinkyRegister& dst,
                                                           const StinkyRegister& addr,
                                                           const std::string&    comment)
    {
        return createInst(pImpl.get(), GFX::flat_load_sbyte, "FLAT_LOAD_I8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadU16(const StinkyRegister& dst,
                                                            const StinkyRegister& addr,
                                                            const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_ushort, "FLAT_LOAD_U16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadI16(const StinkyRegister& dst,
                                                            const StinkyRegister& addr,
                                                            const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_sshort, "FLAT_LOAD_I16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadD16U8(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_ubyte_d16, "FLAT_LOAD_D16_U8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadD16HIU8(const StinkyRegister& dst,
                                                                const StinkyRegister& addr,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_ubyte_d16_hi, "FLAT_LOAD_D16HI_U8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadD16I8(const StinkyRegister& dst,
                                                              const StinkyRegister& addr,
                                                              const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_sbyte_d16, "FLAT_LOAD_D16_I8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadD16HII8(const StinkyRegister& dst,
                                                                const StinkyRegister& addr,
                                                                const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_sbyte_d16_hi, "FLAT_LOAD_D16HI_I8", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadD16B16(const StinkyRegister& dst,
                                                               const StinkyRegister& addr,
                                                               const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_short_d16, "FLAT_LOAD_D16_B16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadD16HIB16(const StinkyRegister& dst,
                                                                 const StinkyRegister& addr,
                                                                 const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_short_d16_hi, "FLAT_LOAD_D16HI_B16", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadB32(const StinkyRegister& dst,
                                                            const StinkyRegister& addr,
                                                            const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_dword, "FLAT_LOAD_DWORD", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadB64(const StinkyRegister& dst,
                                                            const StinkyRegister& addr,
                                                            const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_dwordx2, "FLAT_LOAD_DWORDX2", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadB96(const StinkyRegister& dst,
                                                            const StinkyRegister& addr,
                                                            const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_dwordx3, "FLAT_LOAD_DWORDX3", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatLoadB128(const StinkyRegister& dst,
                                                             const StinkyRegister& addr,
                                                             const std::string&    comment)
    {
        return createInst(
            pImpl.get(), GFX::flat_load_dwordx4, "FLAT_LOAD_DWORDX4", dst, {addr}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatStoreB8(const StinkyRegister& addr,
                                                            const StinkyRegister& src,
                                                            const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::flat_store_byte, "FLAT_STORE_B8", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatStoreD16HIU8(const StinkyRegister& addr,
                                                                 const StinkyRegister& src,
                                                                 const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::flat_store_byte_d16_hi, "FLAT_STORE_D16HI_U8", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatStoreB16(const StinkyRegister& addr,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::flat_store_short, "FLAT_STORE_B16", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatStoreD16HIB16(const StinkyRegister& addr,
                                                                  const StinkyRegister& src,
                                                                  const std::string&    comment)
    {
        return createInstNoDst(pImpl.get(),
                               GFX::flat_store_short_d16_hi,
                               "FLAT_STORE_D16HI_B16",
                               {addr, src},
                               comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatStoreB32(const StinkyRegister& addr,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::flat_store_dword, "FLAT_STORE_DWORD", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatStoreB64(const StinkyRegister& addr,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::flat_store_dwordx2, "FLAT_STORE_DWORDX2", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatStoreB96(const StinkyRegister& addr,
                                                             const StinkyRegister& src,
                                                             const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::flat_store_dwordx3, "FLAT_STORE_DWORDX3", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatStoreB128(const StinkyRegister& addr,
                                                              const StinkyRegister& src,
                                                              const std::string&    comment)
    {
        return createInstNoDst(
            pImpl.get(), GFX::flat_store_dwordx4, "FLAT_STORE_DWORDX4", {addr, src}, comment);
    }

    std::vector<StinkyInstruction*> StinkyTofu::FlatAtomicCmpSwap(const StinkyRegister& dst,
                                                                  const StinkyRegister& addr0,
                                                                  const StinkyRegister& addr1,
                                                                  const std::string&    comment)
    {
        return createInst(pImpl.get(),
                          GFX::flat_atomic_cmpswap,
                          "FLAT_ATOMIC_CMPSWAP",
                          dst,
                          {addr0, addr1},
                          comment);
    }

    // ========================================================================
    // Label Creation
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::createLabel(const std::string& label_name)
    {
        std::vector<StinkyInstruction*> insts;

        // Create a label instruction (using the LABEL HwInstDesc)
        static const HwInstDesc labelDesc{
            GFX::LABEL, GFX::LABEL, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};

        StinkyInstruction* inst = new StinkyInstruction(&labelDesc);
        inst->addModifier<LabelData>(LabelData{Modifier::Type::LABEL_NAME, label_name});

        insts.push_back(inst);
        return insts;
    }

    // ========================================================================
    // Composite Instructions (Architecture-Aware Lowering)
    // ========================================================================

    std::vector<StinkyInstruction*> StinkyTofu::VAddPKF32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;

        // Try to get the packed add F32 instruction descriptor
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_pk_add_f32, pImpl->archID);

        if(desc)
        {
            // Architecture supports packed add F32 - use single instruction
            result.reserve(1);
            StinkyInstruction* inst = pImpl->createInstruction(GFX::v_pk_add_f32, "V_PK_ADD_F32");
            inst->destRegs.push_back(dst);
            inst->srcRegs.push_back(src0);
            inst->srcRegs.push_back(src1);
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
        }
        else
        {
            // Fallback: use two separate V_ADD_F32 instructions
            // This demonstrates the composite instruction pattern for older architectures
            result.reserve(2);

            // Note: In a real implementation, you would need to properly split
            // the destination and source registers for the two packed lanes.
            // This is a simplified example showing the pattern.

            // First lane: dst[0] = src0[0] + src1[0]
            StinkyRegister dst1(dst.regType, dst.regIdx, 1);
            StinkyRegister src0_1(src0.regType, src0.regIdx, 1);
            StinkyRegister src1_1(src1.regType, src1.regIdx, 1);

            StinkyInstruction* inst1 = pImpl->createInstruction(GFX::v_add_f32, "V_ADD_F32");
            inst1->destRegs.push_back(dst1);
            inst1->srcRegs.push_back(src0_1);
            inst1->srcRegs.push_back(src1_1);
            if(!comment.empty())
            {
                inst1->addModifier(CommentData(comment + " [lane 0]"));
            }
            result.push_back(inst1);

            // Second lane: dst[1] = src0[1] + src1[1]
            StinkyRegister dst2(dst.regType, dst.regIdx + 1, 1);
            StinkyRegister src0_2(src0.regType, src0.regIdx + 1, 1);
            StinkyRegister src1_2(src1.regType, src1.regIdx + 1, 1);

            StinkyInstruction* inst2 = pImpl->createInstruction(GFX::v_add_f32, "V_ADD_F32");
            inst2->destRegs.push_back(dst2);
            inst2->srcRegs.push_back(src0_2);
            inst2->srcRegs.push_back(src1_2);
            if(!comment.empty())
            {
                inst2->addModifier(CommentData(comment + " [lane 1]"));
            }
            result.push_back(inst2);
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMulPKF32(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;

        // Try to get the packed mul F32 instruction descriptor
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_pk_mul_f32, pImpl->archID);

        if(desc)
        {
            // Architecture supports packed mul, use single instruction
            result.reserve(1);
            StinkyInstruction* inst = new StinkyInstruction(desc);
            inst->destRegs.push_back(dst);
            inst->srcRegs.push_back(src0);
            inst->srcRegs.push_back(src1);
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
        }
        else
        {
            // Fall back to two separate mul instructions
            result.reserve(2);

            // Mul low half
            StinkyInstruction* inst1 = pImpl->createInstruction(GFX::v_mul_f32, "V_MUL_F32");
            inst1->destRegs.push_back(StinkyRegister(dst.regType, dst.regIdx, 1));
            inst1->srcRegs.push_back(StinkyRegister(src0.regType, src0.regIdx, 1));
            inst1->srcRegs.push_back(StinkyRegister(src1.regType, src1.regIdx, 1));
            if(!comment.empty())
            {
                inst1->addModifier(CommentData(comment + " (low)"));
            }
            result.push_back(inst1);

            // Mul high half
            StinkyInstruction* inst2 = pImpl->createInstruction(GFX::v_mul_f32, "V_MUL_F32");
            inst2->destRegs.push_back(StinkyRegister(dst.regType, dst.regIdx + 1, 1));
            inst2->srcRegs.push_back(StinkyRegister(src0.regType, src0.regIdx + 1, 1));
            inst2->srcRegs.push_back(StinkyRegister(src1.regType, src1.regIdx + 1, 1));
            if(!comment.empty())
            {
                inst2->addModifier(CommentData(comment + " (high)"));
            }
            result.push_back(inst2);
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::VMovB64(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;

        // Try to get the v_mov_b64 instruction descriptor
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_mov_b64, pImpl->archID);

        if(desc)
        {
            // Architecture supports v_mov_b64, use single instruction
            result.reserve(1);
            StinkyInstruction* inst = new StinkyInstruction(desc);
            inst->destRegs.push_back(dst);
            inst->srcRegs.push_back(src);
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
        }
        else
        {
            // Fall back to two v_mov_b32 instructions
            result.reserve(2);

            // Move low half
            StinkyInstruction* inst1 = pImpl->createInstruction(GFX::v_mov_b32, "V_MOV_B32");
            inst1->destRegs.push_back(StinkyRegister(dst.regType, dst.regIdx, 1));
            inst1->srcRegs.push_back(StinkyRegister(src.regType, src.regIdx, 1));
            if(!comment.empty())
            {
                inst1->addModifier(CommentData(comment + " (low)"));
            }
            result.push_back(inst1);

            // Move high half
            StinkyInstruction* inst2 = pImpl->createInstruction(GFX::v_mov_b32, "V_MOV_B32");
            inst2->destRegs.push_back(StinkyRegister(dst.regType, dst.regIdx + 1, 1));
            inst2->srcRegs.push_back(StinkyRegister(src.regType, src.regIdx + 1, 1));
            if(!comment.empty())
            {
                inst2->addModifier(CommentData(comment + " (high)"));
            }
            result.push_back(inst2);
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::VLShiftLeftOrB32(const StinkyRegister& dst,
                                                                 const StinkyRegister& shiftHex,
                                                                 const StinkyRegister& src0,
                                                                 const StinkyRegister& src1,
                                                                 const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;

        // Try to get the v_lshl_or_b32 instruction descriptor
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_lshl_or_b32, pImpl->archID);

        if(desc)
        {
            // Architecture supports v_lshl_or_b32, use single instruction
            result.reserve(1);
            StinkyInstruction* inst = new StinkyInstruction(desc);
            inst->destRegs.push_back(dst);
            inst->srcRegs.push_back(src0);
            inst->srcRegs.push_back(shiftHex);
            inst->srcRegs.push_back(src1);
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
        }
        else
        {
            // Fall back to v_lshlrev_b32 + v_or_b32
            result.reserve(2);

            // Shift
            StinkyInstruction* inst1
                = pImpl->createInstruction(GFX::v_lshlrev_b32, "V_LSHLREV_B32");
            inst1->destRegs.push_back(dst);
            inst1->srcRegs.push_back(shiftHex);
            inst1->srcRegs.push_back(src0);
            if(!comment.empty())
            {
                inst1->addModifier(CommentData(comment + " (shift)"));
            }
            result.push_back(inst1);

            // Or
            StinkyInstruction* inst2 = pImpl->createInstruction(GFX::v_or_b32, "V_OR_B32");
            inst2->destRegs.push_back(dst);
            inst2->srcRegs.push_back(dst);
            inst2->srcRegs.push_back(src1);
            if(!comment.empty())
            {
                inst2->addModifier(CommentData(comment + " (or)"));
            }
            result.push_back(inst2);
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::VAddLShiftLeftU32(const StinkyRegister& dst,
                                                                  const StinkyRegister& shiftHex,
                                                                  const StinkyRegister& src0,
                                                                  const StinkyRegister& src1,
                                                                  const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;

        // Try to get the v_add_lshl_u32 instruction descriptor
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_add_lshl_u32, pImpl->archID);

        if(desc)
        {
            // Architecture supports v_add_lshl_u32, use single instruction
            result.reserve(1);
            StinkyInstruction* inst = new StinkyInstruction(desc);
            inst->destRegs.push_back(dst);
            inst->srcRegs.push_back(src0);
            inst->srcRegs.push_back(src1);
            inst->srcRegs.push_back(shiftHex);
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
        }
        else
        {
            // Fall back to v_add_u32 + v_lshlrev_b32
            result.reserve(2);

            // Add
            StinkyInstruction* inst1 = pImpl->createInstruction(GFX::v_add_u32, "V_ADD_U32");
            inst1->destRegs.push_back(dst);
            inst1->srcRegs.push_back(src0);
            inst1->srcRegs.push_back(src1);
            if(!comment.empty())
            {
                inst1->addModifier(CommentData(comment + " (add)"));
            }
            result.push_back(inst1);

            // Shift
            StinkyInstruction* inst2
                = pImpl->createInstruction(GFX::v_lshlrev_b32, "V_LSHLREV_B32");
            inst2->destRegs.push_back(dst);
            inst2->srcRegs.push_back(shiftHex);
            inst2->srcRegs.push_back(dst);
            if(!comment.empty())
            {
                inst2->addModifier(CommentData(comment + " (shift)"));
            }
            result.push_back(inst2);
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::VLShiftLeftAddU32(const StinkyRegister& dst,
                                                                  const StinkyRegister& shiftHex,
                                                                  const StinkyRegister& src0,
                                                                  const StinkyRegister& src1,
                                                                  const std::string&    comment)
    {
        std::vector<StinkyInstruction*> result;

        // Try to get the v_lshl_add_u32 instruction descriptor
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_lshl_add_u32, pImpl->archID);

        if(desc)
        {
            // Architecture supports v_lshl_add_u32, use single instruction
            result.reserve(1);
            StinkyInstruction* inst = new StinkyInstruction(desc);
            inst->destRegs.push_back(dst);
            inst->srcRegs.push_back(src0);
            inst->srcRegs.push_back(shiftHex);
            inst->srcRegs.push_back(src1);
            if(!comment.empty())
            {
                inst->addModifier(CommentData(comment));
            }
            result.push_back(inst);
        }
        else
        {
            // Fall back to v_lshlrev_b32 + v_add_u32
            result.reserve(2);

            // Shift
            StinkyInstruction* inst1
                = pImpl->createInstruction(GFX::v_lshlrev_b32, "V_LSHLREV_B32");
            inst1->destRegs.push_back(dst);
            inst1->srcRegs.push_back(shiftHex);
            inst1->srcRegs.push_back(src0);
            if(!comment.empty())
            {
                inst1->addModifier(CommentData(comment + " (shift)"));
            }
            result.push_back(inst1);

            // Add
            StinkyInstruction* inst2 = pImpl->createInstruction(GFX::v_add_u32, "V_ADD_U32");
            inst2->destRegs.push_back(dst);
            inst2->srcRegs.push_back(dst);
            inst2->srcRegs.push_back(src1);
            if(!comment.empty())
            {
                inst2->addModifier(CommentData(comment + " (add)"));
            }
            result.push_back(inst2);
        }

        return result;
    }

    std::vector<StinkyInstruction*>
        StinkyTofu::SWaitCnt(int vlcnt, int vscnt, int dscnt, int kmcnt, const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;

        // This is a simplified implementation. The full version in rocisa
        // checks architecture capabilities to generate appropriate wait instructions.
        // For now, we generate a simple s_waitcnt with combined counters.

        // Combine counters for architectures without separate counters
        int lgkmcnt = -1;
        int vmcnt   = -1;

        if(dscnt != -1 || kmcnt != -1)
        {
            lgkmcnt = 0;
            if(dscnt != -1)
                lgkmcnt += dscnt;
            if(kmcnt != -1)
                lgkmcnt += kmcnt;
        }

        if(vlcnt != -1 || vscnt != -1)
        {
            vmcnt = 0;
            if(vlcnt != -1)
                vmcnt += vlcnt;
            if(vscnt != -1)
                vmcnt += vscnt;
        }

        // Create s_waitcnt instruction
        // Note: The actual encoding requires packing lgkmcnt and vmcnt into the immediate
        // This is a simplified version that assumes the underlying instruction builder handles it
        result.reserve(1);
        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_waitcnt, "S_WAITCNT");

        // TODO: Properly encode lgkmcnt/vmcnt into the instruction immediate
        // For now, just add as comment to show intent
        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";
        fullComment
            += "(lgkmcnt=" + std::to_string(lgkmcnt) + ", vmcnt=" + std::to_string(vmcnt) + ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }
        result.push_back(inst);

        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SWaitAlu(int                va_vdst,
                                                         int                va_sdst,
                                                         int                va_ssrc,
                                                         int                hold_cnt,
                                                         int                vm_vsrc,
                                                         int                va_vcc,
                                                         int                sa_sdst,
                                                         const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_wait_alu, "S_WAIT_ALU");

        // Build comment with non-default parameters
        std::string fullComment = comment;
        if(va_vdst != -1 || va_sdst != -1 || va_ssrc != -1 || hold_cnt != -1 || vm_vsrc != -1
           || va_vcc != -1 || sa_sdst != -1)
        {
            if(!fullComment.empty())
                fullComment += " ";
            fullComment += "(";
            bool first    = true;
            auto addParam = [&](const std::string& name, int val) {
                if(val != -1)
                {
                    if(!first)
                        fullComment += ", ";
                    fullComment += name + "=" + std::to_string(val);
                    first = false;
                }
            };
            addParam("va_vdst", va_vdst);
            addParam("va_sdst", va_sdst);
            addParam("va_ssrc", va_ssrc);
            addParam("hold_cnt", hold_cnt);
            addParam("vm_vsrc", vm_vsrc);
            addParam("va_vcc", va_vcc);
            addParam("sa_sdst", sa_sdst);
            fullComment += ")";
        }

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SWaitTensorcnt(int                tensorcnt,
                                                               const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst
            = pImpl->createInstruction(GFX::s_wait_tensorcnt, "S_WAIT_TENSORCNT");

        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";
        fullComment += "(tensorcnt=" + std::to_string(tensorcnt) + ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SNop(int waitState, const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_nop, "S_NOP");

        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";
        fullComment += "(" + std::to_string(waitState) + ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::VNop(int count, const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::v_nop, "V_NOP");

        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";
        fullComment += "(count=" + std::to_string(count) + ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SEndpgm(const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_endpgm, "S_ENDPGM");

        if(!comment.empty())
        {
            inst->addModifier(CommentData(comment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSleep(int simm16, const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_sleep, "S_SLEEP");

        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";
        fullComment += "(" + std::to_string(simm16) + ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SDcacheWb(const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_dcache_wb, "S_DCACHE_WB");

        if(!comment.empty())
        {
            inst->addModifier(CommentData(comment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SDelayAlu(int                instid0,
                                                          int                instid0cnt,
                                                          int                instskip,
                                                          int                instid1,
                                                          int                instid1cnt,
                                                          const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_delay_alu, "S_DELAY_ALU");

        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";

        fullComment
            += "(instid0=" + std::to_string(instid0) + ", cnt=" + std::to_string(instid0cnt);
        if(instskip != -1)
        {
            fullComment += ", skip=" + std::to_string(instskip);
            if(instid1 != -1)
            {
                fullComment += ", instid1=" + std::to_string(instid1)
                               + ", cnt=" + std::to_string(instid1cnt);
            }
        }
        fullComment += ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSetPrior(int prior, const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_setprio, "S_SETPRIO");

        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";
        fullComment += "(" + std::to_string(prior) + ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSetVgprMsb(int simm16, const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_set_vgpr_msb, "S_SET_VGPR_MSB");

        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";
        fullComment += "(0x" + std::to_string(simm16) + ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    std::vector<StinkyInstruction*> StinkyTofu::SSetVgprMsb(
        int msbSrc0, int msbSrc1, int msbSrc2, int msbDst, const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        result.reserve(1);

        // Encode the MSB bits as in rocisa: (msbDst << 6) + (msbSrc2 << 4) + (msbSrc1 << 2) + msbSrc0
        int simm16 = (msbDst << 6) + (msbSrc2 << 4) + (msbSrc1 << 2) + msbSrc0;

        StinkyInstruction* inst = pImpl->createInstruction(GFX::s_set_vgpr_msb, "S_SET_VGPR_MSB");

        std::string fullComment = comment;
        if(!fullComment.empty())
            fullComment += " ";
        fullComment += "(msbSrc0=" + std::to_string(msbSrc0) + ", msbSrc1="
                       + std::to_string(msbSrc1) + ", msbSrc2=" + std::to_string(msbSrc2)
                       + ", msbDst=" + std::to_string(msbDst) + ")";

        if(!fullComment.empty())
        {
            inst->addModifier(CommentData(fullComment));
        }

        result.push_back(inst);
        return result;
    }

    // ========================================================================
    // Extension Instructions (Composite/Macro Instructions)
    // ========================================================================

    // ========================================================================
    // KernelBody Implementation
    // ========================================================================

    KernelBody::KernelBody(const std::string& name)
        : name(name)
        , signature(nullptr)
        , body(nullptr)
        , totalVgprs(0)
        , totalAgprs(0)
        , totalSgprs(0)
    {
    }

    void KernelBody::addSignature(const std::shared_ptr<SignatureBase>& signature)
    {
        this->signature = signature;
    }

    void KernelBody::addBody(const std::shared_ptr<IRListModule>& body)
    {
        this->body = body;
    }

    void KernelBody::setGprs(int totalVgprs, int totalAgprs, int totalSgprs)
    {
        this->totalVgprs = totalVgprs;
        this->totalAgprs = totalAgprs;
        this->totalSgprs = totalSgprs;

        // Update signature if it exists
        if(signature)
        {
            signature->setGprs(totalVgprs, totalAgprs, totalSgprs);
        }
    }

    int KernelBody::getNextFreeVgpr() const
    {
        return signature ? signature->getNextFreeVgpr() : totalVgprs;
    }

    int KernelBody::getNextFreeSgpr() const
    {
        return signature ? signature->getNextFreeSgpr() : totalSgprs;
    }

    std::string KernelBody::getName() const
    {
        return name;
    }

    std::shared_ptr<SignatureBase> KernelBody::getSignature() const
    {
        return signature;
    }

    std::shared_ptr<IRListModule> KernelBody::getBody() const
    {
        return body;
    }

    std::string KernelBody::toString(bool emitComments, bool emitCycleInfo) const
    {
        std::string kStr;

        // Add signature metadata
        if(signature)
        {
            kStr += signature->toString();
        }
        else
        {
            // If no signature, at least add a kernel label
            kStr += name + ":\n";
        }

        // Add instruction body
        if(body)
        {
            kStr += body->emitAssembly(emitCycleInfo, emitComments);
        }
        else
        {
            // Empty body - just a comment
            kStr += "    // Empty kernel body\n";
        }

        return kStr;
    }

    // ========================================================================
    // Assembly Directive Factory Functions
    // ========================================================================

    AsmDirective* StinkyTofu::createSetDirective(const std::string& symbol,
                                                 const std::string& value,
                                                 const std::string& comment)
    {
        AsmDirective* dir = new AsmDirective();
        dir->kind         = AsmDirectiveKind::SET;
        dir->name         = ".set";
        dir->symbol       = symbol;
        dir->value        = value;
        dir->comment      = comment;
        return dir;
    }

    AsmDirective* StinkyTofu::createIfDirective(const std::string& condition,
                                                const std::string& comment)
    {
        AsmDirective* dir = new AsmDirective();
        dir->kind         = AsmDirectiveKind::IF;
        dir->name         = ".if";
        dir->condition    = condition;
        dir->comment      = comment;
        return dir;
    }

    AsmDirective* StinkyTofu::createElseDirective(const std::string& comment)
    {
        AsmDirective* dir = new AsmDirective();
        dir->kind         = AsmDirectiveKind::ELSE;
        dir->name         = ".else";
        dir->comment      = comment;
        return dir;
    }

    AsmDirective* StinkyTofu::createElseIfDirective(const std::string& condition,
                                                    const std::string& comment)
    {
        AsmDirective* dir = new AsmDirective();
        dir->kind         = AsmDirectiveKind::ELSEIF;
        dir->name         = ".elseif";
        dir->condition    = condition;
        dir->comment      = comment;
        return dir;
    }

    AsmDirective* StinkyTofu::createEndIfDirective(const std::string& comment)
    {
        AsmDirective* dir = new AsmDirective();
        dir->kind         = AsmDirectiveKind::ENDIF;
        dir->name         = ".endif";
        dir->comment      = comment;
        return dir;
    }

    // ========================================================================
    // Macro Factory Functions
    // ========================================================================

    MacroInstruction* StinkyTofu::createVMagicDivMacro(uint32_t           divisor,
                                                       StinkyRegister     quotient,
                                                       StinkyRegister     dividend,
                                                       StinkyRegister     tmpVgpr,
                                                       StinkyRegister     tmpSgpr,
                                                       const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "vmagic_div";
        macro->divisor          = divisor;
        macro->operands         = {quotient, dividend, tmpVgpr, tmpSgpr};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createPseudoRandomGeneratorMacro(StinkyRegister     dst,
                                                                   StinkyRegister     seed,
                                                                   StinkyRegister     tmpSgpr,
                                                                   const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "prng";
        macro->operands         = {dst, seed, tmpSgpr};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createVectorStaticDivideMacro(StinkyRegister     dst,
                                                                StinkyRegister     dividend,
                                                                uint32_t           divisor,
                                                                StinkyRegister     tmpVgpr,
                                                                StinkyRegister     tmpSgpr,
                                                                const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "v_static_div";
        macro->divisor          = divisor;
        macro->operands         = {dst, dividend, tmpVgpr, tmpSgpr};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createVectorStaticRemainderMacro(StinkyRegister     dst,
                                                                   StinkyRegister     dividend,
                                                                   uint32_t           divisor,
                                                                   StinkyRegister     tmpVgpr,
                                                                   StinkyRegister     tmpSgpr,
                                                                   const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "v_static_rem";
        macro->divisor          = divisor;
        macro->operands         = {dst, dividend, tmpVgpr, tmpSgpr};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createScalarStaticDivideMacro(StinkyRegister     dst,
                                                                StinkyRegister     dividend,
                                                                uint32_t           divisor,
                                                                StinkyRegister     tmpSgpr,
                                                                const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "s_static_div";
        macro->divisor          = divisor;
        macro->operands         = {dst, dividend, tmpSgpr};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createScalarStaticRemainderMacro(StinkyRegister     dst,
                                                                   StinkyRegister     dividend,
                                                                   uint32_t           divisor,
                                                                   StinkyRegister     tmpSgpr,
                                                                   const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "s_static_rem";
        macro->divisor          = divisor;
        macro->operands         = {dst, dividend, tmpSgpr};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createBranchIfZeroMacro(StinkyRegister     src,
                                                          const std::string& label,
                                                          const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "branch_if_zero";
        macro->label            = label;
        macro->operands         = {src};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createBranchIfNotZeroMacro(StinkyRegister     src,
                                                             const std::string& label,
                                                             const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "branch_if_not_zero";
        macro->label            = label;
        macro->operands         = {src};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createDSInitMacro(uint32_t           sizeBytes,
                                                    uint32_t           value,
                                                    StinkyRegister     tmpVgpr,
                                                    StinkyRegister     tmpSgpr,
                                                    const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "ds_init";
        macro->sizeBytes        = sizeBytes;
        macro->value            = value;
        macro->operands         = {tmpVgpr, tmpSgpr};
        macro->comment          = comment;
        return macro;
    }

    MacroInstruction* StinkyTofu::createArgumentLoaderMacro(StinkyRegister     dst,
                                                            StinkyRegister     kernargPtr,
                                                            uint32_t           offsetBytes,
                                                            uint32_t           sizeBytes,
                                                            const std::string& comment)
    {
        MacroInstruction* macro = new MacroInstruction();
        macro->name             = "arg_load";
        macro->offsetBytes      = offsetBytes;
        macro->sizeBytes        = sizeBytes;
        macro->operands         = {dst, kernargPtr};
        macro->comment          = comment;
        return macro;
    }

} // namespace stinkytofu
