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

namespace stinkytofu {
class Pass;
class Function;
class PassContext;

/// Well-known key for retrieving EstimateAsmCycles result from PassContext.
///
/// After the pass runs, the result (uint32_t) is stored in PassContext under this key.
/// Usage:
///   auto cycles = passCtx.getResult<uint32_t>(kEstimateAsmCyclesKey);
inline const std::string kEstimateAsmCyclesKey = "EstimateAsmCycles";

std::unique_ptr<Pass> createEstimateAsmCyclesPass();

/// Calculate estimate asm cycles for a function
/// @param func The function to analyze
/// @param passCtx The pass context
/// @return The total estimated cycles
unsigned int calculateEstimateAsmCycles(Function& func, PassContext& passCtx);
}  // namespace stinkytofu
