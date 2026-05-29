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
 * @file demo_auto_load.cpp
 * @brief Demo: Automatic intrinsic loading (LLVM-style)
 *
 * This demonstrates how intrinsics are automatically loaded without manual
 * path specification, similar to LLVM's device libraries.
 */

#include <iostream>

#include "stinkytofu/ir/logical/IntrinsicRegistry.hpp"

using namespace stinkytofu;

int main() {
    std::cout << "=== StinkyTofu Automatic Intrinsic Loading Demo ===\n\n";

    // Get the global registry - intrinsics.st.bc is automatically loaded!
    // Similar to how LLVM's clang automatically finds ocml.bc, ockl.bc, etc.
    auto& registry = IntrinsicRegistry::instance();

    if (!registry.isInitialized()) {
        std::cerr << "ERROR: Failed to load intrinsics automatically.\n";
        std::cerr << "Make sure intrinsics.st.bc is in a standard location.\n";
        return 1;
    }

    std::cout << "? Intrinsics loaded automatically!\n";
    std::cout << "  Loaded from: " << registry.getLoadedPath() << "\n";
    std::cout << "  Total intrinsics: " << registry.getIntrinsicNames().size() << "\n\n";

    // Now you can use intrinsics anywhere without loading manually
    std::cout << "Available intrinsics:\n";
    for (const auto& name : registry.getIntrinsicNames()) {
        std::cout << "  - " << name;
        auto pattern = registry.lookup(name);
        if (pattern) {
            std::cout << " (" << pattern->arguments.size() << " args, " << pattern->body.size()
                      << " instructions)";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Usage Example in TensileLite ===\n\n";
    std::cout << "// In TensileLite kernel generator:\n";
    std::cout << "auto& intrinsics = IntrinsicRegistry::instance();\n";
    std::cout << "if (intrinsics.hasIntrinsic(\"ReluF32\")) {\n";
    std::cout << "    auto pattern = intrinsics.lookup(\"ReluF32\");\n";
    std::cout << "    // Generate code from pattern->body\n";
    std::cout << "}\n\n";

    std::cout << "=== Demo Complete ? ===\n";

    return 0;
}
