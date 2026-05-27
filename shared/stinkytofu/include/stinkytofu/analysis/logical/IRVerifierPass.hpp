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

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
/// Configuration for Logical IR verification only.
/// For ASM (StinkyTofu) verification use stinkytofu/analysis/asm/AsmVerifierPass.hpp and
/// AsmVerifierConfig.
struct LogicalIRVerifierConfig {
    bool abortOnError = true;
    bool verbose = false;
};

/// Validate Logical IR structure only. Returns error message if invalid, empty string if valid.
/// For StinkyTofu Assembly IR use validateStinkyIR() from
/// stinkytofu/analysis/asm/AsmVerifierPass.hpp.
std::string validateLogicalIR(Function& func, const LogicalIRVerifierConfig& config = {});

/// Pass that verifies Logical IR (high-level IR only). For ASM use StinkyIRVerifierPass.
class LogicalIRVerifierPass : public Pass {
   public:
    static char ID;

    explicit LogicalIRVerifierPass(LogicalIRVerifierConfig config = {}) : config_(config) {}

    PassID getPassID() const override {
        return &ID;
    }
    const char* getName() const override {
        return "LogicalIRVerifier";
    }
    PreservedAnalyses run(Function& func, PassContext& ctx, AnalysisManager& /*AM*/) override;

   private:
    LogicalIRVerifierConfig config_;
};

inline std::unique_ptr<Pass> createLogicalIRVerifierPass(LogicalIRVerifierConfig config = {}) {
    return std::make_unique<LogicalIRVerifierPass>(config);
}
}  // namespace stinkytofu
