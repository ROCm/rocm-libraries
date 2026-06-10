/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated software files (the "Software"), to deal
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
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelDynamicPass.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"
#include "stinkytofu/transforms/asm/SwPrefetchRelCommon.hpp"

namespace stinkytofu {

class SwInstructionPrefetchRelDynamicPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "SwInstructionPrefetchRelDynamicPass";
    }

    PassID getPassID() const override {
        return &SwInstructionPrefetchRelDynamicPass::ID;
    }

    const SwPrefetchRelPhase1Accum& getPhase1Accum() const {
        return m_phase1;
    }

    void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx) {
        const auto& archArr = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId =
            getGfxArchID(static_cast<uint32_t>(archArr[0]), static_cast<uint32_t>(archArr[1]),
                         static_cast<uint32_t>(archArr[2]));

        const int64_t blockGlobalStart = m_byteOffsetBase;
        BasicBlock* bp = &bb;
        const int64_t bbEntryAccum = m_phase1.accumByte.at(bp);
        const bool allowIns =
            !m_skipSwPrefetchInNaturalLoopBodies || (findLoopForBB(m_loops, &bb) == nullptr);
        const int inserted = insertSwPrefetchLabelsDynamic(
            bb, blockGlobalStart, bbEntryAccum, 0, archId, m_debug ? m_debugStream : nullptr,
            &m_asmSetSymbols, allowIns, getName());
        m_totalPrefetchInserted += inserted;

        int blockCount = 0;
        int64_t blockBytes = 0;
        if (m_debug) {
            *m_debugStream << "[" << getName() << "] Phase 2 BasicBlock: " << bb.getLabel()
                           << " inserted=" << inserted << "\n";
            *m_debugStream << "[" << getName() << "] IR after SW prefetch label insertion:\n";
            *m_debugStream << "\n";
        }
        m_totalCycles +=
            accumulateInstructionSize(bb, m_labelByteOffset, m_debug ? m_debugStream : nullptr,
                                      &blockCount, &blockBytes, m_byteOffsetBase, &m_asmSetSymbols);
        if (m_debug)
            debugPrintSwPrefetchGrid(*m_debugStream, bb.getLabel(), blockGlobalStart, blockBytes,
                                     getName());
        m_totalInstructionCount += blockCount;
        m_totalBytes += blockBytes;
        m_byteOffsetBase += blockBytes;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        m_asmSetSymbols.clear();
        collectAsmSetSymbolValues(func, m_asmSetSymbols);

        if (m_debug) {
            if (!m_debugOutputPath.empty()) {
                m_debugFile.open(m_debugOutputPath);
                m_debugStream = m_debugFile.is_open() ? &m_debugFile : &std::cerr;
            } else {
                m_debugStream = &std::cerr;
            }
        }

        computeSwPrefetchRelPhase1Accum(func, &m_asmSetSymbols, m_phase1,
                                        m_debug ? m_debugStream : nullptr, getName());

        if (m_phase1.totalLayoutBytes <= kSwPrefetchFirstGlobalByte) {
            if (m_debug) {
                *m_debugStream << "[" << getName() << "] no-op: totalLayoutBytes ("
                               << m_phase1.totalLayoutBytes
                               << ") <= first threshold P(0)=" << kSwPrefetchFirstGlobalByte
                               << " (CP preload only)\n";
            }
            if (m_debugFile.is_open()) m_debugFile.close();
            return PreservedAnalyses::all();
        }

        m_totalCycles = 0;
        m_totalInstructionCount = 0;
        m_totalBytes = 0;
        m_totalPrefetchInserted = 0;
        m_labelByteOffset.clear();
        m_byteOffsetBase = 0;
        m_loops = detectLoops(func);

        if (m_debug) {
            *m_debugStream << "[" << getName() << "] Phase 2 insert (CFG-gated), totalLayoutBytes="
                           << m_phase1.totalLayoutBytes << " > P(0)=" << kSwPrefetchFirstGlobalByte
                           << "\n";
        }

        for (BasicBlock& bb : func) runOnBasicBlock(bb, passCtx);

        if (m_debug) {
            *m_debugStream << "[" << getName() << "] Phase 2 complete: totalPrefetchInserted="
                           << m_totalPrefetchInserted << "\n";
            *m_debugStream << "[" << getName()
                           << "] total instruction count = " << m_totalInstructionCount << "\n";
            *m_debugStream << "[" << getName() << "] total cycles = " << m_totalCycles << "\n";
            *m_debugStream << "[" << getName() << "] total size = " << m_totalBytes << " bytes\n";
            if (m_debugFile.is_open()) m_debugFile.close();
        }
        return PreservedAnalyses::none();
    }

    void setDebug(bool enable) {
        m_debug = enable;
    }

    void setDebugOutputPath(const std::string& path) {
        m_debugOutputPath = path;
    }

   private:
    SwPrefetchRelPhase1Accum m_phase1;
    int64_t m_totalCycles = 0;
    int m_totalInstructionCount = 0;
    int64_t m_totalBytes = 0;
    int m_totalPrefetchInserted = 0;
    int64_t m_byteOffsetBase = 0;
    std::vector<Loop> m_loops;
    std::unordered_map<std::string, int64_t> m_labelByteOffset;
    std::unordered_map<std::string, int64_t> m_asmSetSymbols;
    bool m_debug = false;
    bool m_skipSwPrefetchInNaturalLoopBodies = false;
    std::string m_debugOutputPath;
    std::ofstream m_debugFile;
    std::ostream* m_debugStream = &std::cerr;
};

char SwInstructionPrefetchRelDynamicPass::ID = 0;

std::unique_ptr<Pass> createSwInstructionPrefetchRelDynamicPass(
    const std::string& debugOutputPath) {
    auto p = std::make_unique<SwInstructionPrefetchRelDynamicPass>();
    p->setDebugOutputPath(debugOutputPath);
    if (!debugOutputPath.empty()) p->setDebug(true);
    return p;
}

std::unique_ptr<Pass> createSwInstructionPrefetchRelDynamicPass(StinkyAsmModule& module) {
    auto p = std::make_unique<SwInstructionPrefetchRelDynamicPass>();
    if (!module.getOutputDir().empty()) {
        const std::string costBasename =
            module.getOutputName().empty() ? module.getName() : module.getOutputName();
        std::filesystem::path dir = std::filesystem::path(module.getOutputDir()) / costBasename;
        std::filesystem::create_directories(dir);
        constexpr const char* kSwPrefetchDynamicPassDumpLeaf =
            "sw_inst_prefetch_rel_dynamic_pass.txt";
        const std::string path = (dir / kSwPrefetchDynamicPassDumpLeaf).string();
        p->setDebugOutputPath(path);
        p->setDebug(true);
    }
    return p;
}

}  // namespace stinkytofu
