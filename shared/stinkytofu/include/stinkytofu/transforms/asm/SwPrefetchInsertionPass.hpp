/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated software files (the "Software"), to deal
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

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// Factory only (same pattern as other *Pass.hpp). Does not include or alter
/// AccumulateInstructionSizePass.hpp; implementation lives in
/// SwPrefetchInsertionPass.cpp.
///
/// Second pipeline pass: same per-instruction byte layout as
/// AccumulateInstructionSizePass, plus optional dump of proposed SW prefetch
/// sites. When \p debugOutputPath is non-empty, enables debug and writes to
/// that file.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwPrefetchInsertionPass(
    const std::string& debugOutputPath);

/// Uses StinkyAsmModule::setOutputDir / setOutputName: when the output
/// directory is non-empty, enables debug output to
/// `<outputDir>/<kernel_basename>/sw_prefetch_pass.txt` (same layout as
/// AccumulateInstructionSizePass / Backend). SW prefetch insertion always runs;
/// an empty output directory only means no debug dump (same pattern as
/// createAccumulateInstructionSizePass(module)).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwPrefetchInsertionPass(StinkyAsmModule& module);

}  // namespace stinkytofu
