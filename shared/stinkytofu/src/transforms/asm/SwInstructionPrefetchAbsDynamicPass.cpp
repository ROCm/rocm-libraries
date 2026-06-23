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

/// SwInstructionPrefetchAbsDynamicPass — Phase P2 of abs SW prefetch (STUB).
///
/// Dynamic policy (totalLayoutBytes > 64 KiB): per-k targets at align128(P(k))
/// plus replacement-aware, CFG-aware sites (loop preheaders, capped ahead
/// distance). See §16.4 Pass B of SwPrefetchAbsInsertionPass-Design.md.
///
/// This file currently ships only the STUB: the pass computes the layout to
/// classify the kernel size regime, logs a no-op reason, and inserts nothing.
/// When totalLayoutBytes > 65536 it logs that the dynamic pass is not yet
/// implemented. The real per-k + CFG emission replaces the run() body below in
/// a later PR; the gating / factory / wiring already match the static pass so
/// that swap is localized.

#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsDynamicPass.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/SwPrefetchRelCommon.hpp"

namespace stinkytofu {

class SwInstructionPrefetchAbsDynamicPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "SwInstructionPrefetchAbsDynamicPass";
    }

    PassID getPassID() const override {
        return &SwInstructionPrefetchAbsDynamicPass::ID;
    }

    void setBaseSgpr(int baseSgpr) {
        m_baseSgpr = baseSgpr;
    }

    int getBaseSgpr() const {
        return m_baseSgpr;
    }

    void setDebug(bool enable) {
        m_debug = enable;
    }

    void setDebugOutputPath(const std::string& path) {
        m_debugOutputPath = path;
    }

    PreservedAnalyses run(Function& func, PassContext& /*passCtx*/,
                          AnalysisManager& /*AM*/) override {
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

        // Compute layout (no IR mutation) to classify the kernel size regime.
        SwPrefetchRelPhase1Accum phase1;
        computeSwPrefetchRelPhase1Accum(func, &m_asmSetSymbols, phase1,
                                        m_debug ? m_debugStream : nullptr, getName());

        // Regime 1: <= P(0). CP preload covers everything; no software prefetch.
        if (phase1.totalLayoutBytes <= kSwPrefetchFirstGlobalByte) {
            if (m_debug)
                *m_debugStream << "[" << getName() << "] no-op: totalLayoutBytes ("
                               << phase1.totalLayoutBytes
                               << ") <= P(0)=" << kSwPrefetchFirstGlobalByte
                               << " (CP preload only)\n";
            closeDebugFile();
            return PreservedAnalyses::all();
        }

        // Regime 2: (P(0), 64 KiB]. Static policy — handled by the static pass.
        if (phase1.totalLayoutBytes <= kSwPrefetchAbsStaticIcacheSizeBytes) {
            if (m_debug)
                *m_debugStream << "[" << getName() << "] no-op: totalLayoutBytes ("
                               << phase1.totalLayoutBytes << ") <= I-cache limit "
                               << kSwPrefetchAbsStaticIcacheSizeBytes
                               << " — static regime, use SwInstructionPrefetchAbsStaticPass\n";
            closeDebugFile();
            return PreservedAnalyses::all();
        }

        // Regime 3: > 64 KiB — the dynamic policy's regime. P2 ships only a
        // stub: log that the dynamic pass is not implemented and insert nothing.
        //
        // This log is emitted independent of `m_baseSgpr`: the stub mutates no
        // IR regardless, and this "not implemented" line is the MVP's one
        // observable deliverable for >64K kernels (§16.5). In the real pipeline
        // those kernels run with the default baseSgpr=-1 until P5 Tensile wiring,
        // so gating the log behind baseSgpr would hide it in practice. The SGPR
        // reservation only matters once real emission lands.
        if (m_debug) {
            *m_debugStream << "[" << getName()
                           << "] dynamic pass not implemented: totalLayoutBytes ("
                           << phase1.totalLayoutBytes << ") > I-cache limit "
                           << kSwPrefetchAbsStaticIcacheSizeBytes
                           << "; no prefetch inserted (TODO: P2/P3 per-k targets + CFG sites)";
            if (m_baseSgpr < 0) *m_debugStream << " [baseSgpr unset]";
            *m_debugStream << "\n";
        }

        closeDebugFile();
        return PreservedAnalyses::all();
    }

   private:
    void closeDebugFile() {
        if (m_debugFile.is_open()) m_debugFile.close();
    }

    int m_baseSgpr = -1;
    std::unordered_map<std::string, int64_t> m_asmSetSymbols;
    bool m_debug = false;
    std::string m_debugOutputPath;
    std::ofstream m_debugFile;
    std::ostream* m_debugStream = &std::cerr;
};

char SwInstructionPrefetchAbsDynamicPass::ID = 0;

std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(
    int baseSgpr, const std::string& debugOutputPath) {
    auto p = std::make_unique<SwInstructionPrefetchAbsDynamicPass>();
    p->setBaseSgpr(baseSgpr);
    p->setDebugOutputPath(debugOutputPath);
    if (!debugOutputPath.empty()) p->setDebug(true);
    return p;
}

std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(StinkyAsmModule& module) {
    auto p = std::make_unique<SwInstructionPrefetchAbsDynamicPass>();
    // SwInstructionPrefetchAbsBaseSgpr is not yet wired in Module.hpp (P5 Tensile).
    // Default -1 = no-op until the module option is added.
    p->setBaseSgpr(-1);
    if (!module.getOutputDir().empty()) {
        const std::string costBasename =
            module.getOutputName().empty() ? module.getName() : module.getOutputName();
        std::filesystem::path dir = std::filesystem::path(module.getOutputDir()) / costBasename;
        std::filesystem::create_directories(dir);
        constexpr const char* kDumpLeaf = "sw_prefetch_abs_dynamic_pass.txt";
        p->setDebugOutputPath((dir / kDumpLeaf).string());
        p->setDebug(true);
    }
    return p;
}

}  // namespace stinkytofu
