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

#include "stinkytofu/transforms/asm/RemoveInstructionPass.hpp"

#include <cctype>
#include <iostream>
#include <optional>
#include <unordered_set>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "RemoveInstructionPass"

namespace {
using namespace stinkytofu;

static void trimWhitespace(std::string& s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(0, 1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
}

std::vector<std::string> splitMnemonicsCsv(const std::string& mnemonicsCsv) {
    std::vector<std::string> mnemonics;
    std::string token;
    for (char ch : mnemonicsCsv) {
        if (ch == ',') {
            trimWhitespace(token);
            if (!token.empty()) mnemonics.push_back(token);
            token.clear();
        } else {
            token.push_back(ch);
        }
    }
    trimWhitespace(token);
    if (!token.empty()) mnemonics.push_back(token);
    return mnemonics;
}

std::optional<UnifiedOpcode> lookupUnifiedOpcodeByMnemonic(const std::string& mnemonic,
                                                           GfxArchID arch) {
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(arch);
    if (!archInfo) return std::nullopt;

    const auto& mnemonicMap = archInfo->getMnemonicToIsaOpcodeMap();
    const auto it = mnemonicMap.find(mnemonic);
    if (it == mnemonicMap.end()) return std::nullopt;

    const HwInstDesc* desc = getMCIDByIsaOp(it->second, arch);
    if (!desc) return std::nullopt;
    return desc->unifiedOpcode;
}

size_t removeInstructionsInBlock(BasicBlock& bb,
                                 const std::unordered_set<UnifiedOpcode>& targetOpcodes) {
    size_t removed = 0;
    for (auto it = bb.begin(); it != bb.end();) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst && targetOpcodes.count(inst->getUnifiedOpcode()) != 0) {
            it = bb.eraseIR(it);
            ++removed;
        } else {
            ++it;
        }
    }
    return removed;
}

class RemoveInstructionPassImpl : public StinkyInstPass {
   public:
    explicit RemoveInstructionPassImpl(std::vector<UnifiedOpcode> opcodes)
        : opcodes_(std::move(opcodes)) {}

    explicit RemoveInstructionPassImpl(std::vector<std::string> mnemonics)
        : mnemonics_(std::move(mnemonics)) {}

    static char ID;

    const char* getName() const override {
        return "RemoveInstructionPass";
    }

    PassID getPassID() const override {
        return &RemoveInstructionPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const std::unordered_set<UnifiedOpcode> targetOpcodes = resolveTargetOpcodes(passCtx);
        if (targetOpcodes.empty()) {
            PASS_DEBUG(std::cerr << "[RemoveInstructionPass] no valid target opcodes; skipping\n");
            return PreservedAnalyses::all();
        }

        size_t totalRemoved = 0;
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            const size_t removed = removeInstructionsInBlock(bb, targetOpcodes);
            totalRemoved += removed;
            PASS_DEBUG(std::cerr << "[RemoveInstructionPass] bb=\"" << bb.getLabel()
                                 << "\" removed=" << removed << "\n");
        }

        PASS_DEBUG(std::cerr << "[RemoveInstructionPass] total_removed=" << totalRemoved
                             << " target_opcode_count=" << targetOpcodes.size() << "\n");
        return preserveCFGAnalyses();
    }

   private:
    std::unordered_set<UnifiedOpcode> resolveTargetOpcodes(PassContext& passCtx) const {
        std::unordered_set<UnifiedOpcode> targets;
        if (!mnemonics_.empty()) {
            const auto archTriple = passCtx.getGemmTileConfig().arch;
            const GfxArchID arch = getGfxArchID(archTriple[0], archTriple[1], archTriple[2]);
            for (const std::string& mnemonic : mnemonics_) {
                const auto resolved = lookupUnifiedOpcodeByMnemonic(mnemonic, arch);
                if (!resolved) {
                    PASS_DEBUG(std::cerr << "[RemoveInstructionPass] unknown mnemonic \""
                                         << mnemonic << "\" for arch " << getArchName(arch)
                                         << "\n");
                    continue;
                }
                targets.insert(*resolved);
            }
            return targets;
        }

        for (UnifiedOpcode opcode : opcodes_) targets.insert(opcode);
        return targets;
    }

    std::vector<UnifiedOpcode> opcodes_;
    std::vector<std::string> mnemonics_;
};

char RemoveInstructionPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRemoveInstructionPass() {
    return nullptr;
}

std::unique_ptr<Pass> createRemoveInstructionPass(std::vector<UnifiedOpcode> opcodes) {
    if (opcodes.empty()) return nullptr;
    return std::make_unique<RemoveInstructionPassImpl>(std::move(opcodes));
}

std::unique_ptr<Pass> createRemoveInstructionPass(std::vector<std::string> mnemonics) {
    if (mnemonics.empty()) return nullptr;
    return std::make_unique<RemoveInstructionPassImpl>(std::move(mnemonics));
}

std::unique_ptr<Pass> createRemoveInstructionPass(const std::string& mnemonicsCsv) {
    if (mnemonicsCsv.empty()) return nullptr;
    return createRemoveInstructionPass(splitMnemonicsCsv(mnemonicsCsv));
}
}  // namespace stinkytofu
