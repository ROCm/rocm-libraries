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
#include <utility>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/ModulePassManager.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {

// ModulePass owning an inner function-level PassManager. Config is propagated
// to the inner PM at run()-time.

// Runs the inner PassManager on every function (entry + callables), in isolation.
class FunctionToModuleAdaptor : public ModulePass {
   public:
    explicit FunctionToModuleAdaptor(PassManager pm, bool propagatePassFeatureConfig = false)
        : innerPM(std::move(pm)), propagatePassFeatureConfig(propagatePassFeatureConfig) {}

    const char* getName() const override {
        return "FunctionToModuleAdaptor";
    }

    PreservedAnalyses run(StinkyAsmModule& M, PassContext& outerCtx,
                          ModuleAnalysisManager& /*MAM*/) override {
        propagateConfig(outerCtx);
        for (Function* F : M.getFunctions()) {
            if (F) innerPM.run(*F);
        }
        return PreservedAnalyses::none();
    }

   private:
    void propagateConfig(PassContext& outerCtx) {
        innerPM.setGemmTileConfig(outerCtx.getGemmTileConfig());
        innerPM.setAsmCapsConfig(outerCtx.getAsmCapsConfig());
        innerPM.getPassContext().setRemarksEnabled(outerCtx.getRemarksEnabled());
        if (propagatePassFeatureConfig)
            innerPM.setPassFeatureConfig(outerCtx.getPassFeatureConfig());
    }

    PassManager innerPM;
    bool propagatePassFeatureConfig;
};

// Runs the inner PassManager on only the entry function.
class MainOnlyAdaptor : public ModulePass {
   public:
    explicit MainOnlyAdaptor(PassManager pm, bool propagatePassFeatureConfig = false)
        : innerPM(std::move(pm)), propagatePassFeatureConfig(propagatePassFeatureConfig) {}

    const char* getName() const override {
        return "MainOnlyAdaptor";
    }

    PreservedAnalyses run(StinkyAsmModule& M, PassContext& outerCtx,
                          ModuleAnalysisManager& /*MAM*/) override {
        propagateConfig(outerCtx);
        innerPM.run(M.getFunction());
        return PreservedAnalyses::none();
    }

   private:
    void propagateConfig(PassContext& outerCtx) {
        innerPM.setGemmTileConfig(outerCtx.getGemmTileConfig());
        innerPM.setAsmCapsConfig(outerCtx.getAsmCapsConfig());
        innerPM.getPassContext().setRemarksEnabled(outerCtx.getRemarksEnabled());
        if (propagatePassFeatureConfig)
            innerPM.setPassFeatureConfig(outerCtx.getPassFeatureConfig());
    }

    PassManager innerPM;
    bool propagatePassFeatureConfig;
};

inline std::unique_ptr<ModulePass> createFunctionToModuleAdaptor(
    PassManager pm, bool propagatePassFeatureConfig = false) {
    return std::make_unique<FunctionToModuleAdaptor>(std::move(pm), propagatePassFeatureConfig);
}

/// Convenience: wrap a single function pass in an inner PassManager and adapt it
/// to run on entry + every callable function.
inline std::unique_ptr<ModulePass> createFunctionToModuleAdaptor(
    std::unique_ptr<Pass> pass, bool propagatePassFeatureConfig = false) {
    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.addPass(std::move(pass));
    return std::make_unique<FunctionToModuleAdaptor>(std::move(pm), propagatePassFeatureConfig);
}

inline std::unique_ptr<ModulePass> createMainOnlyAdaptor(PassManager pm,
                                                         bool propagatePassFeatureConfig = false) {
    return std::make_unique<MainOnlyAdaptor>(std::move(pm), propagatePassFeatureConfig);
}

/// Convenience: wrap a single function pass in an inner PassManager and adapt it
/// to run on the entry only.
inline std::unique_ptr<ModulePass> createMainOnlyAdaptor(std::unique_ptr<Pass> pass,
                                                         bool propagatePassFeatureConfig = false) {
    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.addPass(std::move(pass));
    return std::make_unique<MainOnlyAdaptor>(std::move(pm), propagatePassFeatureConfig);
}

}  // namespace stinkytofu
