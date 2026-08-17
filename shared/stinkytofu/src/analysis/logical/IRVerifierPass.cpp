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

#include "stinkytofu/analysis/logical/IRVerifierPass.hpp"

#include <iostream>
#include <sstream>

#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
char LogicalIRVerifierPass::ID = 0;

PreservedAnalyses LogicalIRVerifierPass::run(Function& func, PassContext&,
                                             AnalysisManager& /*AM*/) {
    std::string error = validateLogicalIR(func, config_);
    if (!error.empty()) {
        STINKY_UNREACHABLE(error.c_str());
    }
    return PreservedAnalyses::all();
}

std::string validateLogicalIR(Function& func, const LogicalIRVerifierConfig& config) {
    if (config.verbose) {
        std::cout << "[LogicalIRVerifier] Verifying Logical IR...\n";
    }

    if (func.empty()) return "Function is empty (no basic blocks)";

    if (!func.getEntryBlock()) return "Function has no entry basic block";

    size_t logicalCount = 0;
    size_t stinkyCount = 0;
    size_t totalBlocks = 0;

    for (BasicBlock& bb : func) {
        totalBlocks++;

        for (IRBase& ir : bb) {
            if (ir.getType() == IRBase::IRType::LogicalIR)
                logicalCount++;
            else if (ir.getType() == IRBase::IRType::StinkyTofu)
                stinkyCount++;
        }
    }

    if (stinkyCount > 0) {
        std::stringstream ss;
        ss << "Logical IR contains " << stinkyCount << " StinkyTofu (assembly) instructions. "
           << "This suggests IR is partially lowered or mixed.";
        return ss.str();
    }

    if (logicalCount == 0) return "Function contains no Logical instructions (empty IR)";

    if (config.verbose) {
        std::cout << "[LogicalIRVerifier] OK: " << totalBlocks << " blocks, " << logicalCount
                  << " logical instructions\n";
    }

    return "";
}
}  // namespace stinkytofu
