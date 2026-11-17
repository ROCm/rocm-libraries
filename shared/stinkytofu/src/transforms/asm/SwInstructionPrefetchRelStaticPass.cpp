// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelStaticPass.hpp"

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

class SwInstructionPrefetchRelStaticPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "SwInstructionPrefetchRelStaticPass";
    }

    PassID getPassID() const override {
        return &SwInstructionPrefetchRelStaticPass::ID;
    }

    /// When true, `insertSwPrefetchLabels` walks each BB as usual but does not
    /// insert prefetch IR in basic blocks that belong to a natural loop
    /// (`detectLoops`). Default false.
    void setSkipSwPrefetchInNaturalLoopBodies(bool skip) {
        m_skipSwPrefetchInNaturalLoopBodies = skip;
    }

    bool getSkipSwPrefetchInNaturalLoopBodies() const {
        return m_skipSwPrefetchInNaturalLoopBodies;
    }

    void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx) {
        const auto& archArr = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId =
            getGfxArchID(static_cast<uint32_t>(archArr[0]), static_cast<uint32_t>(archArr[1]),
                         static_cast<uint32_t>(archArr[2]));

        const int64_t blockGlobalStart = m_byteOffsetBase;
        const bool allowIns =
            !m_skipSwPrefetchInNaturalLoopBodies || (findLoopForBB(m_loops, &bb) == nullptr);
        insertSwPrefetchLabels(bb, blockGlobalStart, archId, m_debug ? m_debugStream : nullptr,
                               &m_asmSetSymbols, allowIns, getName());

        int blockCount = 0;
        int64_t blockBytes = 0;
        if (m_debug) {
            *m_debugStream << "[" << getName() << "] BasicBlock: " << bb.getLabel() << "\n";
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
        m_totalCycles = 0;
        m_totalInstructionCount = 0;
        m_totalBytes = 0;
        m_labelByteOffset.clear();
        m_byteOffsetBase = 0;
        m_loops = detectLoops(func);
        collectAsmSetSymbolValues(func, m_asmSetSymbols);

        if (m_debug) {
            if (!m_debugOutputPath.empty()) {
                m_debugFile.open(m_debugOutputPath);
                m_debugStream = m_debugFile.is_open() ? &m_debugFile : &std::cerr;
            } else {
                m_debugStream = &std::cerr;
            }
        }

        int totalBlocksInFunction = 0;
        for ([[maybe_unused]] const BasicBlock& bb : func) totalBlocksInFunction++;

        int blocksProcessed = 0;
        if (m_debug) {
            *m_debugStream << "[" << getName()
                           << "] processAllBlocks=" << (m_processAllBlocks ? "true" : "false")
                           << ", function has " << totalBlocksInFunction << " basic block(s)\n";
            dumpAsmSetSymbolMap(*m_debugStream, m_asmSetSymbols);
            *m_debugStream << "[" << getName() << "] blocks to process:\n";
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
            *m_debugStream << "[" << getName() << "] processed " << blocksProcessed << " / "
                           << totalBlocksInFunction << " basic block(s)\n";
            *m_debugStream << "[" << getName()
                           << "] total instruction count = " << m_totalInstructionCount << "\n";
            *m_debugStream << "[" << getName() << "] total cycles = " << m_totalCycles << "\n";
            *m_debugStream << "[" << getName() << "] total size = " << m_totalBytes << " bytes\n";
            if (!m_labelByteOffset.empty()) {
                *m_debugStream << "[" << getName() << "] label -> byte offset:\n";
                for (const auto& kv : m_labelByteOffset)
                    *m_debugStream << "  \"" << kv.first << "\" -> " << kv.second << " bytes\n";
            }
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
    int64_t m_totalCycles = 0;
    int m_totalInstructionCount = 0;
    int64_t m_totalBytes = 0;
    int64_t m_byteOffsetBase = 0;
    std::vector<Loop> m_loops;
    std::unordered_map<std::string, int64_t> m_labelByteOffset;
    std::unordered_map<std::string, int64_t> m_asmSetSymbols;
    bool m_debug = false;
    /// When set, do not insert SW prefetch in BBs that belong to any
    /// `detectLoops` body.
    bool m_skipSwPrefetchInNaturalLoopBodies = false;
    bool m_processAllBlocks = true;
    std::string m_debugOutputPath;
    std::ofstream m_debugFile;
    std::ostream* m_debugStream = &std::cerr;
};

char SwInstructionPrefetchRelStaticPass::ID = 0;

std::unique_ptr<Pass> createSwInstructionPrefetchRelStaticPass(const std::string& debugOutputPath) {
    auto p = std::make_unique<SwInstructionPrefetchRelStaticPass>();
    p->setDebugOutputPath(debugOutputPath);
    if (!debugOutputPath.empty()) p->setDebug(true);
    return p;
}

std::unique_ptr<Pass> createSwInstructionPrefetchRelStaticPass(StinkyAsmModule& module) {
    auto p = std::make_unique<SwInstructionPrefetchRelStaticPass>();
    if (!module.getOutputDir().empty()) {
        const std::string costBasename =
            module.getOutputName().empty() ? module.getName() : module.getOutputName();
        std::filesystem::path dir = std::filesystem::path(module.getOutputDir()) / costBasename;
        std::filesystem::create_directories(dir);
        constexpr const char* kSwPrefetchPassDumpLeaf = "sw_prefetch_pass.txt";
        const std::string path = (dir / kSwPrefetchPassDumpLeaf).string();
        p->setDebugOutputPath(path);
        p->setDebug(true);
    }
    return p;
}
}  // namespace stinkytofu
