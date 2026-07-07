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

/**
 * @file intrinsic-compiler.cpp
 * @brief Intrinsic Compiler Tool
 *
 * This tool compiles intrinsic definitions from Intrinsics.intrinsic to optimized
 * .st.bc bitcode files.
 *
 * Pipeline:
 *   1. Parse Intrinsics.intrinsic (high-level IR syntax)
 *   2. Convert to LogicalModule (high-level IR, with function inlining)
 *   3. Run high-level IR peephole optimization
 *   4. Convert back to pattern format
 *   5. Serialize to intrinsics.st.bc
 *   6. Verify round-trip conversion
 *
 * Usage:
 *   intrinsic-compiler [--verbose|-v] <input.intrinsic> <output.st.bc>
 */

#include <iostream>
#include <string>

#include "stinkytofu/ir/logical/IntrinsicPatternConverter.hpp"
#include "stinkytofu/serialization/asm/PatternParser.hpp"
#include "stinkytofu/serialization/logical/IRSerializer.hpp"
#include "stinkytofu/transforms/logical/LogicalPeepholePass.hpp"

using namespace stinkytofu;

static void printUsage(const char* progName) {
    std::cout << "StinkyTofu Intrinsic Compiler\n";
    std::cout << "==============================\n\n";
    std::cout << "Usage: " << progName << " [--verbose|-v] <input.intrinsic> <output.st.bc>\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  input.intrinsic   Input intrinsic definition file (high-level IR)\n";
    std::cout << "  output.st.bc      Output bitcode file\n\n";
    std::cout << "Options:\n";
    std::cout << "  --verbose, -v     Show detailed compilation progress\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << progName
              << " src/ir/logical/Intrinsics.intrinsic build/intrinsics.st.bc\n";
    std::cout << "  " << progName
              << " -v src/ir/logical/Intrinsics.intrinsic build/intrinsics.st.bc\n";
}

int main(int argc, char** argv) {
    bool verbose = false;
    std::string inputFile;
    std::string outputFile;

    // Parse arguments
    int argIdx = 1;
    while (argIdx < argc) {
        std::string arg = argv[argIdx];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
            argIdx++;
        } else if (inputFile.empty()) {
            inputFile = arg;
            argIdx++;
        } else if (outputFile.empty()) {
            outputFile = arg;
            argIdx++;
        } else {
            std::cerr << "Error: Unexpected argument: " << arg << "\n\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (inputFile.empty() || outputFile.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    if (verbose) {
        std::cout << "=== StinkyTofu Intrinsic Compiler ===\n";
        std::cout << "Input:  " << inputFile << "\n";
        std::cout << "Output: " << outputFile << "\n\n";
    }

    // Step 1: Parse intrinsic definitions
    if (verbose) std::cout << "Step 1: Parsing intrinsic definitions...\n";
    auto patterns = parsePatternFile(inputFile);
    if (patterns.empty()) {
        std::cerr << "Error: No patterns found in " << inputFile << "\n";
        return 1;
    }

    // Filter intrinsic patterns only
    std::vector<Pattern> intrinsics;
    for (const auto& pattern : patterns) {
        if (pattern.type == PatternType::Intrinsic) {
            intrinsics.push_back(pattern);
        }
    }

    if (verbose) std::cout << "  Found " << intrinsics.size() << " intrinsic(s)\n\n";

    if (intrinsics.empty()) {
        std::cerr << "Error: No intrinsic patterns found in " << inputFile << "\n";
        return 1;
    }

    // Step 2: Convert Text -> IR (with function inlining)
    if (verbose)
        std::cout << "Step 2: Converting text patterns to IR (inlining function calls)...\n";
    auto irModules = IntrinsicPatternConverter::patternsToIR(intrinsics);
    if (verbose) std::cout << "  Converted " << irModules.size() << " intrinsic(s) to IR\n\n";

    if (verbose) {
        for (const auto& irModule : irModules) {
            std::cout << "  " << irModule.name << ":\n";
            std::cout << "    Arguments: " << irModule.arguments.size() << "\n";
            std::cout << "    IR Instructions: " << irModule.instructions.size() << "\n";

            // Display IR instructions
            std::cout << "    IR:\n";
            for (const auto& irInst : irModule.instructions) {
                std::cout << "      ";
                irInst->dump(std::cout);
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }

    // Step 3: High-Level IR Optimization
    if (verbose) std::cout << "Step 3: Running high-level IR peephole optimization...\n";

    // TODO: Update to use new Pass infrastructure with LogicalToFunctionConverter
    // For now, skip optimization to avoid API mismatch
    if (verbose) {
        std::cout << "  Optimization pass temporarily disabled during refactoring\n";
        std::cout << "\n";
    }

    // Step 4: Convert IR -> Text (for serialization)
    if (verbose) std::cout << "Step 4: Converting IR back to text patterns...\n";
    auto optimizedPatterns = IntrinsicPatternConverter::irToPatterns(irModules);
    if (verbose)
        std::cout << "  Converted " << optimizedPatterns.size() << " IR module(s) to patterns\n\n";

    // Step 5: Serialize to .st.bc
    if (verbose) std::cout << "Step 5: Serializing to " << outputFile << "...\n";
    if (!IRSerializer::serializeToFile(optimizedPatterns, outputFile)) {
        std::cerr << "Error: Failed to write " << outputFile << "\n";
        return 1;
    }

    if (verbose)
        std::cout << "  Success! Wrote " << optimizedPatterns.size() << " intrinsic(s) to "
                  << outputFile << "\n\n";

    // Step 6: Verify round-trip conversion
    if (verbose) std::cout << "Step 6: Verifying round-trip conversion...\n";
    auto loaded = IRSerializer::deserializeFromFile(outputFile);
    if (loaded.size() != optimizedPatterns.size()) {
        std::cerr << "Warning: Verification failed. Loaded " << loaded.size()
                  << " intrinsics, expected " << optimizedPatterns.size() << "\n";
        return 1;
    }

    // Verify each intrinsic
    for (size_t i = 0; i < loaded.size(); ++i) {
        const auto& original = optimizedPatterns[i];
        const auto& reloaded = loaded[i];

        if (original.name != reloaded.name) {
            std::cerr << "Warning: Name mismatch for intrinsic " << i << ": " << original.name
                      << " vs " << reloaded.name << "\n";
            return 1;
        }

        if (original.body.size() != reloaded.body.size()) {
            std::cerr << "Warning: Instruction count mismatch for " << original.name << ": "
                      << original.body.size() << " vs " << reloaded.body.size() << "\n";
            return 1;
        }

        if (verbose)
            std::cout << "  - " << original.name
                      << " verified (Text -> IR -> Text -> .st.bc -> Text)\n";
    }

    if (verbose) {
        std::cout << "\n  Round-trip conversion successful!\n";
        std::cout << "  Pipeline: Text -> IR -> Text -> .st.bc -> Text verified!\n\n";
        std::cout << "=== Compilation Complete ===\n";
    } else {
        // Quiet mode: just show summary
        std::cout << "Compiled " << optimizedPatterns.size() << " intrinsic(s) to " << outputFile
                  << "\n";
    }

    return 0;
}
