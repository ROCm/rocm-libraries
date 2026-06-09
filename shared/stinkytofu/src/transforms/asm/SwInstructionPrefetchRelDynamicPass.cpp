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

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
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

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        std::unordered_map<std::string, int64_t> asmSetSymbols;
        collectAsmSetSymbolValues(func, asmSetSymbols);

        if (m_debug) {
            if (!m_debugOutputPath.empty()) {
                m_debugFile.open(m_debugOutputPath);
                m_debugStream = m_debugFile.is_open() ? &m_debugFile : &std::cerr;
            } else {
                m_debugStream = &std::cerr;
            }
        }

        computeSwPrefetchRelPhase1Accum(func, &asmSetSymbols, m_phase1,
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

        if (m_debug) {
            *m_debugStream << "[" << getName()
                           << "] Phase 1 complete; Phase 2 insert not implemented yet "
                              "(totalLayoutBytes="
                           << m_phase1.totalLayoutBytes << " > P(0)=" << kSwPrefetchFirstGlobalByte
                           << ")\n";
        }

        if (m_debugFile.is_open()) m_debugFile.close();
        (void)passCtx;
        return PreservedAnalyses::all();
    }

    void setDebug(bool enable) {
        m_debug = enable;
    }

    void setDebugOutputPath(const std::string& path) {
        m_debugOutputPath = path;
    }

   private:
    SwPrefetchRelPhase1Accum m_phase1;
    bool m_debug = false;
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
