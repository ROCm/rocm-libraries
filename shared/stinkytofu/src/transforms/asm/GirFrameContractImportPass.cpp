/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/GirFrameContractImportPass.hpp"

#include <cstdint>
#include <memory>
#include <string>

#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace {
using namespace stinkytofu;

class GirFrameContractImportPass final : public Pass {
   public:
    static char ID;

    explicit GirFrameContractImportPass(const StinkyAsmModule& module)
        : contract(module.getGirFrameContract()),
          actionTagCount(module.getPluginDataI64("gir.action_tag_count", 0)) {}

    const char* getName() const override {
        return "GIR Frame Contract Import";
    }

    PassID getPassID() const override {
        return &ID;
    }

    PreservedAnalyses run(Function& function, PassContext&, AnalysisManager&) override {
        if (!contract || !contract->loaded)
            report_fatal_error(
                "GIR frame pipeline enabled but the module carries no frame contract");
        if (actionTagCount == 0)
            report_fatal_error(
                "GIR frame contract imported but rocisa conversion produced no action tags");
        function.setStructMetaData<GirFrameContract>(kGirFrameContractKey, contract);
        return PreservedAnalyses::none();
    }

   private:
    std::shared_ptr<const GirFrameContract> contract;
    int64_t actionTagCount = 0;
};

char GirFrameContractImportPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createGirFrameContractImportPass(const StinkyAsmModule& module) {
    return std::make_unique<GirFrameContractImportPass>(module);
}
}  // namespace stinkytofu
