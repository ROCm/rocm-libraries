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

#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

// Per-arch logical name -> ASM mnemonic (same data as Rocisa LogicalToArchMap; gives correct
// ds_read vs ds_load etc.)
#include "stinkytofu/ir/LogicalToAsmMappings_generated.inc"

// For ArchHelper access
using namespace stinkytofu;
#include <string>
#include <vector>

namespace {
using namespace stinkytofu;

/**
 * @brief Generate mnemonic for MFMA instructions
 *
 * Follows rocisa mnemonic format from rocisa/include/instruction/mfma.hpp:preStr()
 * CDNA (gfx942, gfx950): v_mfma_{accType}_{m}x{n}x{k}[_{blocks}b]_{instType}[_1k]
 * RDNA (gfx1250): v_wmma_{accType}_{m}x{n}x{k}_{instType}[_1k]
 */
std::string generateMFMAMnemonic(const std::string& accType, int m, int n, int k, int blocks,
                                 const std::string& instType, bool mfma1k, GfxArchID arch) {
    std::string variantStr = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);

    // RDNA architectures (gfx12+) use v_wmma instead of v_mfma
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(arch);
    bool isRDNA = (archInfo && archInfo->major >= 12);

    if (isRDNA) {
        // RDNA: v_wmma_{accType}_{m}x{n}x{k}_{instType}[_1k]
        std::string mfma1kSuffix = mfma1k ? "_1k" : "";
        return "v_wmma_" + accType + "_" + variantStr + "_" + instType + mfma1kSuffix;
    } else {
        // CDNA: v_mfma_{accType}_{m}x{n}x{k}[_{blocks}b]_{instType}[_1k]
        std::string blocksSuffix = (blocks > 1) ? std::to_string(blocks) + "b_" : "";
        std::string mfma1kSuffix = mfma1k ? "_1k" : "";
        return "v_mfma_" + accType + "_" + variantStr + "_" + blocksSuffix + instType +
               mfma1kSuffix;
    }
}

/**
 * @brief Generate mnemonic for SMFMA (Sparse MFMA) instructions
 *
 * All architectures use v_smfmac_{accType}_{m}x{n}x{k}_{instType} format
 * Note: blocks parameter is NOT part of the mnemonic, it's an instruction modifier/operand
 */
std::string generateSMFMAMnemonic(const std::string& accType, int m, int n, int k, int blocks,
                                  const std::string& instType, GfxArchID arch) {
    std::string variantStr = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);

    // All architectures use v_smfmac format (blocks is not part of mnemonic)
    return "v_smfmac_" + accType + "_" + variantStr + "_" + instType;
}

/**
 * @brief Generate mnemonic for MXMFMA (Mixed-precision scaled WMMA) instructions
 *
 * Follows rocisa mnemonic format from rocisa/include/instruction/mfma.hpp:preStr()
 * Format: v_wmma_scale[16]_f32_{m}x{n}x{k}_{instType}
 * Note: Uses "v_wmma_scale" prefix, not "v_mxmfma"
 */
std::string generateMXMFMAMnemonic(int m, int n, int k, int block, const std::string& instType) {
    std::string variantStr = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    std::string blockStr = (block == 16) ? "16" : "";

    return "v_wmma_scale" + blockStr + "_f32_" + variantStr + "_" + instType;
}

// Helper to create assembly instruction from IR
StinkyInstruction* createAsmFromIR(LogicalInstruction* irInst, GfxArchID arch) {
    const char* logicalName = irInst->getLogicalName();

    // Get the architecture-specific mnemonic
    uint16_t isaOpcode = 0;
    std::string mnemonic;

    // ====================================================================
    // Special Instructions: MFMA, SMFMA, MXMFMA
    // ====================================================================
    // Generate mnemonics dynamically from instruction metadata
    if (irInst->getOpcode() == logical::MFMA) {
        const MFMAData* data = irInst->asMFMA();
        if (!data) {
            STINKY_UNREACHABLE("MFMA instruction has no MFMAData");
            return nullptr;
        }
        mnemonic = generateMFMAMnemonic(data->accType, data->m, data->n, data->k, data->blocks,
                                        data->instType, data->mfma1k, arch);
    } else if (irInst->getOpcode() == logical::SMFMA) {
        const SMFMAData* data = irInst->asSMFMA();
        if (!data) {
            STINKY_UNREACHABLE("SMFMA instruction has no SMFMAData");
            return nullptr;
        }
        mnemonic = generateSMFMAMnemonic(data->accType, data->m, data->n, data->k, data->blocks,
                                         data->instType, arch);
    } else if (irInst->getOpcode() == logical::MXMFMA) {
        const MXMFMAData* data = irInst->asMXMFMA();
        if (!data) {
            STINKY_UNREACHABLE("MXMFMA instruction has no MXMFMAData");
            return nullptr;
        }
        mnemonic = generateMXMFMAMnemonic(data->m, data->n, data->k, data->block, data->instType);
    }
    // ====================================================================
    // Regular instructions: per-arch map only (LogicalToAsmMappings_generated.inc)
    // Every lowering for each arch must be in the map; no fallback.
    // ====================================================================
    else {
        const char* archMnemonic = getMnemonicForLogicalOnArch(logicalName, arch);
        if (!archMnemonic) {
            STINKY_UNREACHABLE(
                ("ToStinkyAsmPass: No mapping for logical instruction '" +
                 std::string(logicalName) +
                 "' on this architecture; add to per-arch LogicalToArchMap (Gfx*.cpp).")
                    .c_str());
            return nullptr;
        }
        mnemonic = archMnemonic;
    }

    // Get the ISA opcode for this mnemonic on the target architecture
    isaOpcode = getMnemonicToIsaOpcode(mnemonic, arch);
    const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, arch);

    if (!desc) {
        STINKY_UNREACHABLE(
            ("ToStinkyAsmPass: Instruction not supported on architecture: " + mnemonic).c_str());
        return nullptr;
    }

    // Create the assembly instruction
    StinkyInstruction* asmInst = IRBase::createIR<StinkyInstruction>(desc);

    // Copy operands from IR to assembly
    if (!irInst->dests.empty()) {
        asmInst->setDestRegs(irInst->dests);
    }
    if (!irInst->srcs.empty()) {
        asmInst->setSrcRegs(irInst->srcs);
    }

    // Copy comment
    if (!irInst->comment.empty()) {
        asmInst->addModifier(CommentData(irInst->comment));
    }

    // TODO: Copy DPP, SDWA, DS modifiers when needed

    return asmInst;
}

/// Implementation of ToStinkyAsmPass using unified Pass infrastructure
class ToStinkyAsmPassImpl : public Pass {
   public:
    static constexpr const char* PassName = "ToStinkyAsmPass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);

        // Process all basic blocks
        for (BasicBlock& bb : func) {
            // Skip filtered basic blocks
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            lowerToAsm(bb, arch);
        }
        return PreservedAnalyses::none();
    }

   private:
    void lowerToAsm(BasicBlock& bb, GfxArchID arch) {
        // Use iterators to allow insertion/removal during traversal
        auto it = bb.begin();
        while (it != bb.end()) {
            IRBase* irNode = &(*it);

            if (irNode->getType() == IRBase::IRType::LogicalIR) {
                LogicalInstruction* logicalInst = cast<LogicalInstruction>(irNode);

                // Lower to assembly
                StinkyInstruction* asmInst = createAsmFromIR(logicalInst, arch);

                if (asmInst) {
                    // Insert assembly instruction before the logical instruction
                    bb.insertIR(it, asmInst);

                    // Remove the logical instruction from IRList
                    auto toRemove = it;
                    ++it;  // Move to next before removing
                    bb.removeIR(&(*toRemove));

                    logicalInst->safeErase();
                    continue;
                }
            }

            ++it;
        }
    }
};

char ToStinkyAsmPassImpl::ID = 0;

}  // anonymous namespace

namespace stinkytofu {
std::unique_ptr<Pass> createToStinkyAsmPass() {
    return std::make_unique<ToStinkyAsmPassImpl>();
}

}  // namespace stinkytofu
