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

#include <memory>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/serialization/ssa/CanonicalSSAPrinter.hpp"

namespace stinkytofu {
class Pass;

struct DumpCanonicalSSAConfig {
    /// Destination file. Empty writes to standard output, which is what the
    /// FileCheck harness captures.
    std::string outputPath;

    CanonicalSSAPrinterOptions printerOptions;

    /// Report a function with no canonical SSA as an error. Set false to print
    /// the "no canonical SSA attached" placeholder instead, which is useful for
    /// checking that an unsupported function was left alone.
    bool requireCanonicalSSA = true;
};

/// Creates a read-only pass that prints a function's canonical SSA graph.
///
/// The graph comes from the cached CanonicalSSAAnalysis result rather than being
/// lifted here, so the dump reports whether the function was lifted instead of
/// answering a different question.
///
/// It is verified before printing, with dominance included, so a dump is never
/// quietly taken as evidence that the SSA is well formed. Verification failures
/// are printed as comments ahead of the dump, and the dump still happens so a
/// malformed graph can be inspected.
///
/// The pass never mutates the function.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createDumpCanonicalSSAPass(
    DumpCanonicalSSAConfig config = {});

}  // namespace stinkytofu
