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
#pragma once

#include <memory>
#include <string>
#include <utility>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"

namespace stinkytofu {
class StinkyAsmModule;

/// Controls DumpStinkyModulePass output paths and printer/emitter options.
struct DumpStinkyModulePassConfig {
    /// If non-empty, write Stinky textual IR (AsmPrinter: st.func / op form) to this file.
    std::string stirPath;

    /// When true, also emit assembly (StinkyAsmEmitter) to asmPath or a derived path.
    bool emitAsm = false;

    /// Output path for StinkyAsmEmitter. If empty while emitAsm is true, uses stirPath with
    /// its extension replaced by ".s", or "<stirPath>.s" if there is no extension.
    std::string asmPath;

    AsmPrinterOptions printerOptions{};
    AsmEmitterOptions emitterOptions{};
};

/// Writes Stinky IR text and/or emitted GPU assembly for a Module.
///
/// When constructed with a StinkyAsmModule, the PassManager entry point dumps all Functions in
/// module emission order (entry first, then callees). Without a module, it dumps the current
/// Function passed through the PassManager entry point.
class STINKYTOFU_EXPORT DumpStinkyModulePass : public Pass {
   public:
    static char ID;

    explicit DumpStinkyModulePass(DumpStinkyModulePassConfig config = {})
        : config_(std::move(config)) {}

    DumpStinkyModulePass(const StinkyAsmModule& module, DumpStinkyModulePassConfig config = {})
        : config_(std::move(config)), module_(&module) {}

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return "DumpStinkyModulePass";
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override;

    PreservedAnalyses run(const StinkyAsmModule& module, PassContext& passCtx,
                          AnalysisManager& /*AM*/);

    const DumpStinkyModulePassConfig& getConfig() const {
        return config_;
    }

    void setConfig(DumpStinkyModulePassConfig config) {
        config_ = std::move(config);
    }

    void setModule(const StinkyAsmModule& module) {
        module_ = &module;
    }

    void clearModule() {
        module_ = nullptr;
    }

   private:
    DumpStinkyModulePassConfig config_;
    const StinkyAsmModule* module_ = nullptr;
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createDumpStinkyModulePass(
    DumpStinkyModulePassConfig config = {});
STINKYTOFU_EXPORT std::unique_ptr<Pass> createDumpStinkyModulePass(
    const StinkyAsmModule& module, DumpStinkyModulePassConfig config = {});
}  // namespace stinkytofu
