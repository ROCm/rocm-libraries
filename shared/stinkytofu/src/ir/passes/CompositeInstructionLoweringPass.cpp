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

#include "ir/passes/CompositeInstructionLoweringPass.hpp"
#include "ErrorHandling.hpp"
#include "ir/StinkyInstructions.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "isa/ArchHelper.hpp"
#include <string>

namespace stinkytofu
{
    namespace
    {
        /**
         * @brief Get assembly mnemonic for a logical IR instruction name
         *
         * Uses auto-generated mappings from TableGen (IRMnemonics_generated.inc)
         *
         * @param logicalName The IR instruction's logical name (e.g., "VAddPKF32")
         * @return Assembly mnemonic (e.g., "v_pk_add_f32"), or empty string if not found
         */
        std::string getIRMnemonic(const char* logicalName)
        {
            std::string mnemonic;

            // Auto-generated mappings from TableGen
            // Format: else if(std::string(logicalName) == "ClassName") { mnemonic = "mnemonic"; }
            if(false)
            {
                // Placeholder for generated code
            }
#include "ir/IRMnemonics_generated.inc"
            else
            {
                // Not found in generated mappings
                return "";
            }

            return mnemonic;
        }
    } // anonymous namespace

    /**
     * @brief Check if an IR instruction is supported natively on the target architecture.
     *
     * Uses TableGen-generated mappings to look up the assembly mnemonic,
     * then checks if that instruction exists in the architecture's instruction tables.
     *
     * @param logicalName The IR instruction's logical name
     * @param arch Target architecture
     * @return true if the instruction is natively supported, false otherwise
     */
    bool CompositeInstructionLoweringPass::isInstructionSupported(const std::string& logicalName,
                                                                  GfxArchID          arch) const
    {
        // Get the assembly mnemonic from TableGen-generated mappings
        std::string mnemonic = getIRMnemonic(logicalName.c_str());
        if(mnemonic.empty())
        {
            return false; // Not in generated mappings
        }

        // Check if this mnemonic exists in the architecture's instruction tables
        uint16_t isaOpcode = getMnemonicToIsaOpcode(mnemonic, arch);
        if(isaOpcode == 0)
        {
            return false;
        }

        const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, arch);
        return desc != nullptr;
    }

    bool CompositeInstructionLoweringPass::hasVPKAddF32(GfxArchID arch) const
    {
        // Use logical name instead of hardcoded mnemonic
        return isInstructionSupported("VAddPKF32", arch);
    }

    bool CompositeInstructionLoweringPass::hasVPKMulF32(GfxArchID arch) const
    {
        // Use logical name instead of hardcoded mnemonic
        return isInstructionSupported("VMulPKF32", arch);
    }

    bool CompositeInstructionLoweringPass::hasVMovB64(GfxArchID arch) const
    {
        // Use logical name instead of hardcoded mnemonic
        return isInstructionSupported("VMovB64", arch);
    }

    bool CompositeInstructionLoweringPass::hasVLShlOrB32(GfxArchID arch) const
    {
        // Use logical name instead of hardcoded mnemonic
        return isInstructionSupported("VLShiftLeftOrB32", arch);
    }

    std::vector<IRInstruction*> CompositeInstructionLoweringPass::transform(IRInstruction* irInst,
                                                                            GfxArchID      arch)
    {
        std::vector<IRInstruction*> result;

        if(!irInst)
        {
            return result;
        }

        // If not composite, pass through unchanged
        if(!irInst->isComposite())
        {
            result.push_back(irInst);
            return result;
        }

        const std::string logicalName = irInst->getLogicalName();

        // ================================================================
        // VAddPKF32: Packed add F32
        // ================================================================
        if(logicalName == "VAddPKF32")
        {
            auto* pkAdd = static_cast<VAddPKF32*>(irInst);

            if(hasVPKAddF32(arch))
            {
                // Architecture supports v_pk_add_f32 - keep as single instruction
                // But we need to convert to a simple IR instruction that maps 1:1
                // For now, we'll just return the original (it will be handled by ToStinkyAsmPass)
                // In a full implementation, we'd have a simple VPKAddF32 IR class
                result.push_back(pkAdd);
            }
            else
            {
                // Expand to 2x v_add_f32 (low and high parts)
                // Assumption: dst and srcs are 64-bit VGPR pairs
                // dst = v[n:n+1], src0 = v[m:m+1], src1 = v[p:p+1]

                const auto& dst  = pkAdd->dests[0];
                const auto& src0 = pkAdd->srcs[0];
                const auto& src1 = pkAdd->srcs[1];

                // Low 32 bits
                auto* addLow = new VAddF32(dst, // v[n]
                                           src0, // v[m]
                                           src1, // v[p]
                                           std::nullopt,
                                           std::nullopt,
                                           pkAdd->comment + " (low)");
                result.push_back(addLow);

                // High 32 bits (would need to offset register indices)
                // For a complete implementation, we'd need register+1 logic
                // For now, just add the low part
                // TODO: Implement high part with proper register offsetting
            }
        }
        // ================================================================
        // VMulPKF32: Packed multiply F32
        // ================================================================
        else if(logicalName == "VMulPKF32")
        {
            auto* pkMul = static_cast<VMulPKF32*>(irInst);

            if(hasVPKMulF32(arch))
            {
                // Architecture supports v_pk_mul_f32
                result.push_back(pkMul);
            }
            else
            {
                // Expand to 2x v_mul_f32
                const auto& dst  = pkMul->dests[0];
                const auto& src0 = pkMul->srcs[0];
                const auto& src1 = pkMul->srcs[1];

                auto* mulLow = new VMulF32(
                    dst, src0, src1, std::nullopt, std::nullopt, pkMul->comment + " (low)");
                result.push_back(mulLow);
            }
        }
        // ================================================================
        // VMovB64: 64-bit move
        // ================================================================
        else if(logicalName == "VMovB64")
        {
            auto* movB64 = static_cast<VMovB64*>(irInst);

            if(hasVMovB64(arch))
            {
                // Architecture supports v_mov_b64
                result.push_back(movB64);
            }
            else
            {
                // Expand to 2x v_mov_b32
                const auto& dst = movB64->dests[0];
                const auto& src = movB64->srcs[0];

                // TODO: Implement VMovB64 lowering to two 32-bit moves
                // auto* movLow = new VMovB32(dst, src, movB64->comment + " (low)");
                // result.push_back(movLow);
                // Add high part with register+1
                (void)dst;
                (void)src;
            }
        }
        // ================================================================
        // VLShiftLeftOrB32: (src0 << shift) | src1
        // ================================================================
        else if(logicalName == "VLShiftLeftOrB32")
        {
            auto* lshlOr = static_cast<VLShiftLeftOrB32*>(irInst);

            if(hasVLShlOrB32(arch))
            {
                // Architecture supports v_lshl_or_b32
                result.push_back(lshlOr);
            }
            else
            {
                // Expand to v_lshlrev_b32 + v_or_b32
                // TODO: Implement expansion
                // For now, just keep the original (will fail in ToStinkyAsmPass)
                result.push_back(lshlOr);
            }
        }
        else
        {
            STINKY_UNREACHABLE(
                ("CompositeInstructionLoweringPass: Unknown composite instruction: " + logicalName)
                    .c_str());
        }

        return result;
    }

    std::unique_ptr<IRInstTransformPass> createCompositeInstructionLoweringPass()
    {
        return std::make_unique<CompositeInstructionLoweringPass>();
    }

} // namespace stinkytofu
