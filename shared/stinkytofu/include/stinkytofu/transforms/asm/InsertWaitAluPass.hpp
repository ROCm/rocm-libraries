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

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class ModulePass;

/// Insert s_wait_alu instructions for SCHED_MODE 2 (VA_VDST + VM_VSRC).
///
/// Function pass: full scoreboard analysis when run on the entry, conservative
/// entry drain when run on a callable function. Used by stinkytofu-opt single-pass
/// mode and unit tests.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertWaitAluPass(
    bool enableESM2TrackValuVsrc = false);

/// Whole-kernel driver: full analysis on the entry function, then the conservative
/// call-boundary drain on every callee. Reserves a seam for future caller<->callee
/// analysis.
STINKYTOFU_EXPORT std::unique_ptr<ModulePass> createInsertWaitAluModulePass(
    bool enableESM2TrackValuVsrc = false);

}  // namespace stinkytofu
