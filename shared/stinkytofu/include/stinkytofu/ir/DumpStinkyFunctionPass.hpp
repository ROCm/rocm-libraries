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

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"

namespace stinkytofu {
/// Controls DumpStinkyFunctionPass output paths and printer/emitter options.
struct DumpStinkyFunctionPassConfig {
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

/// Writes the current Function to disk as Stinky IR text and/or as emitted GPU assembly.
class STINKYTOFU_EXPORT DumpStinkyFunctionPass : public Pass {
   public:
    static char ID;

    explicit DumpStinkyFunctionPass(DumpStinkyFunctionPassConfig config = {})
        : config_(std::move(config)) {}

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return "DumpStinkyFunctionPass";
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override;

    const DumpStinkyFunctionPassConfig& getConfig() const {
        return config_;
    }

    void setConfig(DumpStinkyFunctionPassConfig config) {
        config_ = std::move(config);
    }

   private:
    DumpStinkyFunctionPassConfig config_;
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createDumpStinkyFunctionPass(
    DumpStinkyFunctionPassConfig config = {});
}  // namespace stinkytofu
