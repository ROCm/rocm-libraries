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

#include "stinkytofu/transforms/logical/CompositeInstructionLoweringPass.hpp"

#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/LogicalToAsmMappings_generated.inc"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace {
using namespace stinkytofu;

/**
 * @brief Get assembly mnemonic for a logical IR instruction on the given arch (per-arch map only).
 */
std::string getIRMnemonic(const char* logicalName, GfxArchID arch) {
    const char* m = getMnemonicForLogicalOnArch(logicalName, arch);
    return m ? std::string(m) : "";
}

/// Implementation of the CompositeInstructionLoweringPass using unified Pass infrastructure
class CompositeInstructionLoweringPassImpl : public Pass {
   public:
    static constexpr const char* PassName = "CompositeInstructionLoweringPass";
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

            expandCompositeInstructions(bb, arch);
        }
        return preserveCFGAnalyses();
    }

   private:
    void expandCompositeInstructions(BasicBlock& bb, GfxArchID arch) {
        // Use iterators to allow insertion/removal during traversal
        auto it = bb.begin();
        while (it != bb.end()) {
            IRBase* irNode = &(*it);

            if (irNode->getType() == IRBase::IRType::LogicalIR) {
                LogicalInstruction* logicalInst = cast<LogicalInstruction>(irNode);
                if (logicalInst->isComposite()) {
                    // Expand this composite instruction
                    std::vector<LogicalInstruction*> expanded =
                        expandInstruction(logicalInst, arch);

                    if (!expanded.empty() && expanded[0] != logicalInst) {
                        // Insert expanded instructions before the composite
                        for (auto* expandedInst : expanded) {
                            bb.insertIR(it, static_cast<IRBase*>(expandedInst));
                        }

                        // Remove the composite instruction from the block
                        auto toRemove = it;
                        ++it;  // Move to next before removing
                        bb.removeIR(&(*toRemove));

                        logicalInst->safeErase();
                        continue;
                    }
                }
            }

            ++it;
        }
    }

    std::vector<LogicalInstruction*> expandInstruction(LogicalInstruction* irInst, GfxArchID arch) {
        std::vector<LogicalInstruction*> result;

        // ================================================================
        // VAddPKF32: Packed add F32
        // ================================================================
        if (irInst->getOpcode() == logical::VAddPKF32) {
            if (isInstructionSupported("VAddPKF32", arch)) {
                // Keep as-is (architecture supports it)
                result.push_back(irInst);
            } else {
                // Expand to v_add_f32 (simplified - only low part)
                const auto& dst = irInst->dests[0];
                const auto& src0 = irInst->srcs[0];
                const auto& src1 = irInst->srcs[1];

                auto* addLow = VAddF32(dst, src0, src1, std::nullopt, std::nullopt,
                                       irInst->comment + " (expanded)");
                result.push_back(addLow);
            }
        }
        // ================================================================
        // VMulPKF32: Packed multiply F32
        // ================================================================
        else if (irInst->getOpcode() == logical::VMulPKF32) {
            if (isInstructionSupported("VMulPKF32", arch)) {
                result.push_back(irInst);
            } else {
                const auto& dst = irInst->dests[0];
                const auto& src0 = irInst->srcs[0];
                const auto& src1 = irInst->srcs[1];

                auto* mulLow = VMulF32(dst, src0, src1, std::nullopt, std::nullopt,
                                       irInst->comment + " (expanded)");
                result.push_back(mulLow);
            }
        }
        // ================================================================
        // VMovB64: 64-bit move
        // ================================================================
        else if (irInst->getOpcode() == logical::VMovB64) {
            if (isInstructionSupported("VMovB64", arch)) {
                result.push_back(irInst);
            } else {
                // For now, keep as-is (TODO: implement proper 64-bit lowering)
                result.push_back(irInst);
            }
        }
        // ================================================================
        // VLShiftLeftOrB32: (src0 << shift) | src1
        // ================================================================
        else if (irInst->getOpcode() == logical::VLShiftLeftOrB32) {
            if (isInstructionSupported("VLShiftLeftOrB32", arch)) {
                result.push_back(irInst);
            } else {
                // For now, keep as-is (TODO: expand to lshlrev + or)
                result.push_back(irInst);
            }
        } else {
            // Unknown composite - keep as-is
            result.push_back(irInst);
        }

        return result;
    }

    bool isInstructionSupported(const std::string& logicalName, GfxArchID arch) const {
        std::string mnemonic = getIRMnemonic(logicalName.c_str(), arch);
        if (mnemonic.empty()) {
            return false;
        }

        uint16_t isaOpcode = getMnemonicToIsaOpcode(mnemonic, arch);
        if (isaOpcode == 0) {
            return false;
        }

        const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, arch);
        return desc != nullptr;
    }
};

char CompositeInstructionLoweringPassImpl::ID = 0;

}  // anonymous namespace

namespace stinkytofu {
std::unique_ptr<Pass> createCompositeInstructionLoweringPass() {
    return std::make_unique<CompositeInstructionLoweringPassImpl>();
}

}  // namespace stinkytofu
