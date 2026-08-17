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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"

namespace stinkytofu {
class Pass;
class Function;
class PassContext;
struct StinkyInstruction;

/// Analysis result type for total estimated asm cycles.
struct EstimateAsmCyclesAnalysis {
    STINKYTOFU_ANALYSIS_KEY("EstimateAsmCyclesAnalysis")
    using Result = uint32_t;
    static STINKYTOFU_EXPORT Result run(Function& F, AnalysisManager& AM);
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createEstimateAsmCyclesPass();

/// Calculate estimate asm cycles for a function
/// @param func The function to analyze
/// @param passCtx The pass context
/// @return The total estimated cycles
STINKYTOFU_EXPORT unsigned int calculateEstimateAsmCycles(Function& func, PassContext& passCtx);

/// Run the asm-cycle estimator and return a per-instruction map of estimated
/// cumulative cycle positions (the cycle index at which each instruction is
/// modeled to issue). Only instructions inside the modeled `label_LoopBeginL`
/// unrolled-loop region are populated, and only for Gfx1250 (the estimator
/// self-disables otherwise, yielding an empty map). Unlike the pass form, this
/// helper does NOT annotate instructions with `<This is N-cycle>` comments, so
/// it is safe to call from other passes as a pure query.
STINKYTOFU_EXPORT std::unordered_map<const StinkyInstruction*, uint32_t>
computeEstimatedCyclesPerInstruction(Function& func, PassContext& passCtx);
}  // namespace stinkytofu
