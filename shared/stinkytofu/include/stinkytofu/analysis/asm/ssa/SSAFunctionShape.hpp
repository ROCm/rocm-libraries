/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

namespace stinkytofu {

class Function;

/// Structural fingerprint of everything attached SSA depends on: block count,
/// CFG edge counts, instruction count and order, opcodes, and every register
/// operand.
///
/// Attached SSA is only valid for the program it was built from, and no
/// revision counter exists because mutation happens on BasicBlock and on
/// instruction operands, neither of which notifies the Function. Comparing
/// fingerprints at the boundaries that matter catches stale SSA without
/// instrumenting every mutation site. Never returns kUnstampedShape, so a
/// stamped arena is always distinguishable from a hand-built one.
uint64_t computeFunctionShape(const Function& function);

}  // namespace stinkytofu
