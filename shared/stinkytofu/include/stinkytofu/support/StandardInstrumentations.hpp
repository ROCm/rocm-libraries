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

#include <iostream>
#include <string>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/core/PassInstrumentation.hpp"

namespace stinkytofu {

/// Run StinkyTofu Assembly IR verification after every pass (LLVM `-verify-each`
/// style). Reports the first failure to stderr, tagged with the pass that just
/// ran, so a corruption is attributed to the exact transform that introduced it.
///
/// This is the unified, opt-in verification observer: prefer it over inserting a
/// StinkyIRVerifierPass at a single point when you want per-pass attribution.
class VerifyInstrumentation : public PassInstrumentation {
   public:
    explicit VerifyInstrumentation(AsmVerifierConfig config = {}) : config_(config) {
        // verify-each attributes failures per pass, so never abort inside the
        // verifier -- otherwise the offending pass name would not be reported.
        config_.abortOnError = false;
    }

    void afterPass(const std::string& passName, Function& F, PassContext& /*ctx*/) override {
        std::string error = validateStinkyIR(F, config_);
        if (!error.empty()) {
            std::cerr << "[verify-each] StinkyTofu ASM IR invalid after " << passName << ": "
                      << error << '\n';
        }
    }

   private:
    AsmVerifierConfig config_;
};

}  // namespace stinkytofu
