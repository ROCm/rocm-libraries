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

#include <string>
#include <vector>

#include "stinkytofu/serialization/asm/PatternParser.hpp"

namespace stinkytofu {

//===----------------------------------------------------------------------===//
// Pattern Code Generator
//
// Generates C++ matcher code from parsed patterns.
// Uses the Pattern AST from PatternParser.hpp.
//===----------------------------------------------------------------------===//

class PatternCodeGen {
   public:
    explicit PatternCodeGen(const std::string& outputDir);

    // Generate C++ matcher code for patterns of a specific type
    bool generateMatchers(const std::vector<Pattern>& patterns, PatternType type);

   private:
    std::string outputDir;

    // Code generation helpers (assembly IR)
    std::string generateMatcherClass(const Pattern& pattern, PatternType type);
    std::string generateMatchFunction(const Pattern& pattern, PatternType type);

    // Code generation helpers (high-level IR)
    std::string generateMatcherClassHLIR(const Pattern& pattern);
    std::string generateMatchFunctionHLIR(const Pattern& pattern);

    // Common helpers
    std::string generateHeader(PatternType type);
    std::string generateFooter(const std::vector<Pattern>& patterns, PatternType type);

    // Utility
    std::string toCppIdentifier(const std::string& name);
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Generate pattern matchers from pattern file
// Returns true on success, false on error
bool genPeepholePatterns(const std::string& patternFile, const std::string& outdir);

}  // namespace stinkytofu
