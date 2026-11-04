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

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {
class StinkyAsmModule;

// Module-level pass infrastructure. Unit of IR is a whole StinkyAsmModule
// (entry + callable functions).

// Stub: no module-level analyses exist yet.
class STINKYTOFU_EXPORT ModuleAnalysisManager {
   public:
    ModuleAnalysisManager() = default;
    ~ModuleAnalysisManager() = default;

    ModuleAnalysisManager(const ModuleAnalysisManager&) = delete;
    ModuleAnalysisManager& operator=(const ModuleAnalysisManager&) = delete;

    ModuleAnalysisManager(ModuleAnalysisManager&&) = default;
    ModuleAnalysisManager& operator=(ModuleAnalysisManager&&) = default;

    // No-op until module analyses exist.
    void invalidate(StinkyAsmModule&, const PreservedAnalyses&) {}
    void clear() {}
};

// Operates on a whole StinkyAsmModule; owns its own iteration over functions.
class ModulePass {
   public:
    virtual ~ModulePass() = default;

    virtual const char* getName() const = 0;

    virtual PreservedAnalyses run(StinkyAsmModule& M, PassContext& passCtx,
                                  ModuleAnalysisManager& MAM) = 0;
};

// Runs module passes in add order. Config lives on its own PassContext;
// adaptors propagate it to inner PassManagers at run()-time.
class STINKYTOFU_EXPORT ModulePassManager {
   public:
    /// Run all module passes on the given module.
    void run(StinkyAsmModule& M);

    /// Add a module pass. Passes run in the order they are added.
    void addPass(std::unique_ptr<ModulePass> pass) {
        passes.push_back(std::move(pass));
    }

    ModulePassManager() = default;
    ~ModulePassManager() = default;

    ModulePassManager(ModulePassManager&&) noexcept = default;
    ModulePassManager& operator=(ModulePassManager&&) noexcept = default;

    void setGemmTileConfig(const GemmTileConfig& config) {
        passCtx.setGemmTileConfig(config);
    }

    void setPassFeatureConfig(const PassFeatureConfig& config) {
        passCtx.setPassFeatureConfig(config);
    }

    void setAsmCapsConfig(const AsmCapsConfig& config) {
        passCtx.setAsmCapsConfig(config);
    }

    PassContext& getPassContext() {
        return passCtx;
    }

    const PassContext& getPassContext() const {
        return passCtx;
    }

    ModuleAnalysisManager& getModuleAnalysisManager() {
        return moduleAnalysisManager;
    }

    /// Register an observer fired around each ModulePass (before/after IR dump,
    /// verify). Same PassInstrumentation used by PassManager.
    void addInstrumentation(std::shared_ptr<PassInstrumentation> inst) {
        instrumentations.push_back(std::move(inst));
    }

   protected:
    PassContext passCtx;
    ModuleAnalysisManager moduleAnalysisManager;

    std::vector<std::unique_ptr<ModulePass>> passes;
    std::vector<std::shared_ptr<PassInstrumentation>> instrumentations;
};
}  // namespace stinkytofu
