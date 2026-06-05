/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/InstructionSizeCosting.hpp"

namespace stinkytofu {

/// Implementation of **`stinkytofu::accumulateInstructionSize`**.  Full
/// documentation (walk order, PHI/LABEL, **`pcBefore`**,
/// **`getLiteralExtraBytes`**, cycle rules, Doxygen parameters and return, SW
/// prefetch **`labelOff`** cross-reference) lives on the declaration in
/// **`AccumulateInstructionSizePass.hpp`**.
int64_t accumulateInstructionSize(BasicBlock& bb,
                                  std::unordered_map<std::string, int64_t>& outLabelByteOffset,
                                  std::ostream* debugOut, int* outCount, int64_t* outTotalBytes,
                                  int64_t baseByteOffset,
                                  const std::unordered_map<std::string, int64_t>* asmSetSymbols) {
    int64_t totalCycles = 0;
    int64_t totalBytes = 0;
    int count = 0;

    for (BasicBlock::iterator it = bb.begin(); it != bb.end(); ++it) {
        IRBase* node = it.getNodePtr();
        addAlignmentPaddingFromDirectiveNode(node, baseByteOffset, totalBytes, debugOut);
        if (node->getType() == IRBase::IRType::StinkyAsmDirective) continue;
        if (node->getType() != IRBase::IRType::StinkyTofu) continue;

        StinkyInstruction& inst = getStinkyInst(it);
        if (inst.getUnifiedOpcode() == GFX::PHI) continue;
        if (inst.getUnifiedOpcode() == GFX::LABEL) {
            if (const LabelData* ld = inst.getModifier<LabelData>()) {
                addAlignmentPaddingForLabelInstruction(inst, baseByteOffset, totalBytes, debugOut);
                const int64_t labelAddr = baseByteOffset + totalBytes;
                outLabelByteOffset[ld->label] = labelAddr;
                if (debugOut)
                    *debugOut << "  [LABEL name=\"" << ld->label << "\" addr=" << labelAddr
                              << " bytes]\n";
            } else if (debugOut)
                *debugOut << "  [LABEL (no LabelData)]\n";
            continue;
        }
        count++;
        // TODO: check if the cost cycles is correct
        int cost;
        if (isMFMA(inst) || isWMMA(inst)) {
            cost = inst.latencyCycles;
        } else {
            cost = inst.issueCycles;
        }
        totalCycles += cost;
        int baseSize = getEffectiveBaseSizeInBytes(inst);
        const int64_t pcBefore = baseByteOffset + totalBytes;
        int literalExtra = getLiteralExtraBytes(inst, &outLabelByteOffset, pcBefore, asmSetSymbols);
        int instBytes = baseSize + literalExtra;
        totalBytes += instBytes;

        if (debugOut) {
            *debugOut << "  [cost=" << cost << " cycles, size=" << baseSize;
            if (literalExtra != 0)
                *debugOut << "+" << literalExtra << "(literal)=" << instBytes << " bytes";
            else
                *debugOut << " bytes";
            *debugOut << ", total=" << totalBytes << " bytes";
            *debugOut << ", opcode=" << inst.getUnifiedOpcode() << " (isa=" << inst.getISAOpcode()
                      << ")] ";
            inst.dump(*debugOut);
            *debugOut << "\n";
        }
    }

    if (outCount) *outCount = count;
    if (outTotalBytes) *outTotalBytes = totalBytes;
    return totalCycles;
}

}  // namespace stinkytofu

namespace {
using namespace stinkytofu;

class AccumulateInstructionSizePass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "AccumulateInstructionSizePass";
    }

    PassID getPassID() const override {
        return &AccumulateInstructionSizePass::ID;
    }

    void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx) {
        int blockCount = 0;
        int64_t blockBytes = 0;
        if (m_debug) {
            *m_debugStream << "[AccumulateInstructionSizePass] BasicBlock: " << bb.getLabel()
                           << "\n";
            m_totalCycles +=
                accumulateInstructionSize(bb, m_labelByteOffset, m_debugStream, &blockCount,
                                          &blockBytes, m_byteOffsetBase, &m_asmSetSymbols);
        } else {
            m_totalCycles +=
                accumulateInstructionSize(bb, m_labelByteOffset, nullptr, &blockCount, &blockBytes,
                                          m_byteOffsetBase, &m_asmSetSymbols);
        }
        m_totalInstructionCount += blockCount;
        m_totalBytes += blockBytes;
        m_byteOffsetBase += blockBytes;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        m_totalCycles = 0;
        m_totalInstructionCount = 0;
        m_totalBytes = 0;
        m_labelByteOffset.clear();
        m_byteOffsetBase = 0;
        collectAsmSetSymbolValues(func, m_asmSetSymbols);

        if (m_debug) {
            if (!m_debugOutputPath.empty()) {
                m_debugFile.open(m_debugOutputPath);
                m_debugStream = m_debugFile.is_open() ? &m_debugFile : &std::cerr;
            } else {
                m_debugStream = &std::cerr;
            }
        }

        // Count total basic blocks in the function (for debugging)
        int totalBlocksInFunction = 0;
        for ([[maybe_unused]] const BasicBlock& bb : func) totalBlocksInFunction++;

        int blocksProcessed = 0;
        if (m_debug) {
            *m_debugStream << "[AccumulateInstructionSizePass] processAllBlocks="
                           << (m_processAllBlocks ? "true" : "false") << ", function has "
                           << totalBlocksInFunction << " basic block(s)\n";
            dumpAsmSetSymbolMap(*m_debugStream, m_asmSetSymbols);
            *m_debugStream << "[AccumulateInstructionSizePass] blocks to process:\n";
        }

        for (BasicBlock& bb : func) {
            bool processThis = m_processAllBlocks || passCtx.shouldProcessBasicBlock(bb);
            if (m_debug) {
                *m_debugStream << "  - BasicBlock \"" << bb.getLabel() << "\" "
                               << (processThis ? "[PROCESSING]" : "[SKIPPED by filter]") << "\n";
            }
            if (processThis) {
                runOnBasicBlock(bb, passCtx);
                blocksProcessed++;
            }
        }

        if (m_debug) {
            *m_debugStream << "[AccumulateInstructionSizePass] processed " << blocksProcessed
                           << " / " << totalBlocksInFunction << " basic block(s)\n";
            *m_debugStream << "[AccumulateInstructionSizePass] total instruction count = "
                           << m_totalInstructionCount << "\n";
            *m_debugStream << "[AccumulateInstructionSizePass] total cycles = " << m_totalCycles
                           << "\n";
            *m_debugStream << "[AccumulateInstructionSizePass] total size = " << m_totalBytes
                           << " bytes\n";
            if (!m_labelByteOffset.empty()) {
                *m_debugStream << "[AccumulateInstructionSizePass] label -> byte offset:\n";
                for (const auto& kv : m_labelByteOffset)
                    *m_debugStream << "  \"" << kv.first << "\" -> " << kv.second << " bytes\n";
            }
            if (m_debugFile.is_open()) m_debugFile.close();
        }

        if (m_moduleForTotalBytes) m_moduleForTotalBytes->setTotalInstructionBytes(m_totalBytes);
        return PreservedAnalyses::none();
    }

    /// Total accumulated cycles (instruction size) after run().
    int64_t getTotalCycles() const {
        return m_totalCycles;
    }

    /// Total number of instructions processed after run().
    int getTotalInstructionCount() const {
        return m_totalInstructionCount;
    }

    /// Total instruction size in bytes (encoding size) after run().
    int64_t getTotalBytes() const {
        return m_totalBytes;
    }

    /// LABEL name -> byte offset at label (after run()).
    const std::unordered_map<std::string, int64_t>& getLabelByteOffset() const {
        return m_labelByteOffset;
    }

    void setDebug(bool enable) {
        m_debug = enable;
    }

    bool getDebug() const {
        return m_debug;
    }

    /// When set and debug is on, instruction cost is written to this file instead
    /// of stderr.
    void setDebugOutputPath(const std::string& path) {
        m_debugOutputPath = path;
    }

    const std::string& getDebugOutputPath() const {
        return m_debugOutputPath;
    }

    /// When true (default), process all basic blocks for whole-program stats.
    /// When false, respect the pipeline's basic block filter.
    void setProcessAllBlocks(bool processAll) {
        m_processAllBlocks = processAll;
    }

    bool getProcessAllBlocks() const {
        return m_processAllBlocks;
    }

    /// When non-null, run() calls setTotalInstructionBytes(m_totalBytes) on the
    /// module (authoritative size across all basic blocks processed by this
    /// pass).
    void setModuleForTotalBytes(StinkyAsmModule* module) {
        m_moduleForTotalBytes = module;
    }

   private:
    int64_t m_totalCycles = 0;
    int m_totalInstructionCount = 0;
    int64_t m_totalBytes = 0;
    int64_t m_byteOffsetBase = 0;  // running offset for multi-BB label addresses
    std::unordered_map<std::string, int64_t> m_labelByteOffset;
    /// `.set` symbol -> integer (from AsmDirective SET), for literal encoding
    /// size.
    std::unordered_map<std::string, int64_t> m_asmSetSymbols;
    bool m_debug = false;            // print each instruction and cost; set false to disable
    bool m_processAllBlocks = true;  // when true, ignore pipeline BB filter (whole program)
    std::string m_debugOutputPath;   // when non-empty, dump to this file
    std::ofstream m_debugFile;
    std::ostream* m_debugStream = &std::cerr;
    StinkyAsmModule* m_moduleForTotalBytes = nullptr;
};

char AccumulateInstructionSizePass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createAccumulateInstructionSizePass(const std::string& debugOutputPath) {
    auto p = std::make_unique<AccumulateInstructionSizePass>();
    if (!debugOutputPath.empty()) {
        p->setDebugOutputPath(debugOutputPath);
        p->setDebug(true);
    }
    return p;
}

std::unique_ptr<Pass> createAccumulateInstructionSizePass(StinkyAsmModule& module) {
    auto p = std::make_unique<AccumulateInstructionSizePass>();
    p->setModuleForTotalBytes(&module);
    if (!module.getOutputDir().empty()) {
        const std::string costBasename =
            module.getOutputName().empty() ? module.getName() : module.getOutputName();
        std::filesystem::path dir = std::filesystem::path(module.getOutputDir()) / costBasename;
        std::filesystem::create_directories(dir);
        const std::string path = (dir / "accumulate_instruction_size_pass_debug.txt").string();
        p->setDebugOutputPath(path);
        p->setDebug(true);
    }
    return p;
}
}  // namespace stinkytofu
