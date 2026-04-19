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
 * @file test_loader.cpp
 * @brief Test program to verify IntrinsicLibrary loading
 *
 * This program tests the complete intrinsic compilation flow:
 *   1. intrinsic-compiler: Intrinsics.intrinsic -> intrinsics.st.bc
 *   2. IntrinsicLibrary: Load intrinsics.st.bc
 *   3. Query and display intrinsic definitions
 */

#include <iomanip>
#include <iostream>

#include "stinkytofu/ir/logical/IntrinsicLibrary.hpp"

using namespace stinkytofu;

void printIntrinsicDetails(const IntrinsicLibrary& lib, const std::string& name) {
    std::cout << "\n=== Intrinsic: " << name << " ===\n";

    const Pattern* pattern = lib.lookup(name);
    if (!pattern) {
        std::cout << "  NOT FOUND\n";
        return;
    }

    // Print comment
    std::cout << "  Comment: " << pattern->comment << "\n";

    // Print arguments
    std::cout << "  Arguments (" << pattern->arguments.size() << "):\n";
    for (const auto& arg : pattern->arguments) {
        std::cout << "    " << arg.name << ": " << arg.regType << "\n";
    }

    // Print body instructions
    std::cout << "  Body (" << pattern->body.size() << " instructions):\n";
    for (const auto& inst : pattern->body) {
        std::cout << "    " << inst.destReg << " = " << inst.operation << "(";
        for (size_t i = 0; i < inst.operands.size(); ++i) {
            if (i > 0) std::cout << ", ";

            const auto& op = inst.operands[i];
            if (op.type == IntrinsicOperand::Register) {
                std::cout << op.registerName;
            } else if (op.type == IntrinsicOperand::IntLiteral) {
                std::cout << op.intValue;
            } else if (op.type == IntrinsicOperand::FloatLiteral) {
                std::cout << op.floatValue;
            } else if (op.type == IntrinsicOperand::HexLiteral) {
                float floatVal = static_cast<float>(op.floatValue);
                uint32_t bits = *reinterpret_cast<uint32_t*>(&floatVal);
                std::cout << "0x" << std::hex << bits << std::dec;
            }
        }
        std::cout << ")\n";
    }

    // Print Python binding status
    std::cout << "  Python Binding: " << (pattern->pythonBinding ? "enabled" : "disabled") << "\n";
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <intrinsics.st.bc>\n";
        return 1;
    }

    std::string bcFilePath = argv[1];

    std::cout << "=== StinkyTofu Intrinsic Loader Test ===\n\n";
    std::cout << "Loading: " << bcFilePath << "\n\n";

    // Step 1: Load intrinsics library
    auto lib = IntrinsicLibrary::loadFromFile(bcFilePath);
    if (!lib) {
        std::cerr << "Error: Failed to load intrinsic library\n";
        return 1;
    }

    std::cout << "? Successfully loaded " << lib->size() << " intrinsic(s)\n\n";

    // Step 2: Print library statistics
    lib->printStats();

    // Step 3: Test specific intrinsic lookups
    std::cout << "\n=== Testing Intrinsic Lookups ===\n";

    std::vector<std::string> testNames = {"ReluF32", "ClampF16", "SigmoidF32", "ExpF32",
                                          "NonExistent"};

    for (const auto& name : testNames) {
        std::cout << "\nLookup '" << name << "': ";
        if (lib->hasIntrinsic(name)) {
            std::cout << "FOUND ?\n";
            printIntrinsicDetails(*lib, name);
        } else {
            std::cout << "NOT FOUND ?\n";
        }
    }

    // Step 4: Test accessor methods
    std::cout << "\n\n=== Testing Accessor Methods ===\n";

    std::string testIntrinsic = "ReluF32";
    if (lib->hasIntrinsic(testIntrinsic)) {
        std::cout << "\nTesting " << testIntrinsic << " accessors:\n";

        auto args = lib->getArguments(testIntrinsic);
        std::cout << "  getArguments(): " << args.size() << " argument(s)\n";
        for (const auto& arg : args) {
            std::cout << "    - " << arg.name << "\n";
        }

        auto body = lib->getBody(testIntrinsic);
        std::cout << "  getBody(): " << body.size() << " instruction(s)\n";

        auto comment = lib->getComment(testIntrinsic);
        std::cout << "  getComment(): \"" << comment << "\"\n";

        bool hasPython = lib->hasPythonBinding(testIntrinsic);
        std::cout << "  hasPythonBinding(): " << (hasPython ? "true" : "false") << "\n";
    }

    // Step 5: List all intrinsics
    std::cout << "\n\n=== All Intrinsics ===\n";
    auto allNames = lib->getIntrinsicNames();
    for (const auto& name : allNames) {
        std::cout << "  - " << name << "\n";
    }

    std::cout << "\n\n=== Test Complete ? ===\n";

    return 0;
}
