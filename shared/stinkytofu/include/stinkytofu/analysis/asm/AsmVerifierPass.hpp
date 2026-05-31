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
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
/// Configuration for StinkyTofu Assembly IR verification only.
/// Used by the ASM pipeline; not shared with Logical IR.
struct AsmVerifierConfig {
    bool abortOnError = true;
    bool verbose = false;
    bool checkRegisterWidths = true;
    bool checkRegisterRanges = false;
    bool checkReadWriteOperands = true;
};

/// Validate StinkyTofu Assembly IR (structure, HwInstDesc, register widths).
/// Returns error message if invalid, empty string if valid.
/// Only for ASM; do not use for Logical IR.
STINKYTOFU_EXPORT std::string validateStinkyIR(Function& func,
                                               const AsmVerifierConfig& config = {});

/// Pass that verifies StinkyTofu Assembly IR. Used only in the ASM pass pipeline.
class STINKYTOFU_EXPORT StinkyIRVerifierPass : public Pass {
   public:
    static char ID;

    explicit StinkyIRVerifierPass(AsmVerifierConfig config = {}) : config_(config) {}

    PassID getPassID() const override {
        return &ID;
    }
    const char* getName() const override {
        return "StinkyIRVerifier";
    }
    PreservedAnalyses run(Function& func, PassContext& ctx, AnalysisManager& /*AM*/) override;

   private:
    AsmVerifierConfig config_;
};

inline std::unique_ptr<Pass> createStinkyIRVerifierPass(AsmVerifierConfig config = {}) {
    return std::make_unique<StinkyIRVerifierPass>(config);
}
}  // namespace stinkytofu
