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

#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"

namespace stinkytofu {

class Function;

/// Assign every value the physical register it was lifted from.
///
/// This reproduces the producer's original allocation exactly, which makes it
/// the reference point for differential testing: lifting, colouring, and SSA
/// destruction must together be an identity transform on the physical program.
/// Any difference is a defect in that machinery rather than in allocation
/// policy, which is why this gate runs before any real allocator is evaluated.
///
/// This is the simplest allocation policy rather than part of the SSA data
/// model, so it sits here and not with AllocationResult. It has its own header
/// because both RegisterAllocationPass, which needs the baseline to report
/// against, and LegacyIdentityAllocator, which is a thin wrapper over it, use
/// it; the pass must be able to name the baseline without taking a compile-time
/// dependency on any concrete allocator.
AllocationResult createLegacyColoring(const Function& function);

}  // namespace stinkytofu
