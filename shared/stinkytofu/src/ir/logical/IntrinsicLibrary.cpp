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

#include "stinkytofu/ir/logical/IntrinsicLibrary.hpp"

#include <algorithm>
#include <iostream>

#include "stinkytofu/serialization/logical/IRSerializer.hpp"

namespace stinkytofu {
//===----------------------------------------------------------------------===//
// Factory Methods
//===----------------------------------------------------------------------===//

std::shared_ptr<IntrinsicLibrary> IntrinsicLibrary::loadFromFile(const std::string& bcFilePath) {
    // Deserialize intrinsics from .st.bc file
    auto patterns = IRSerializer::deserializeFromFile(bcFilePath);
    if (patterns.empty()) {
        std::cerr << "Error: Failed to load intrinsics from " << bcFilePath << "\n";
        return nullptr;
    }

    // Create library
    auto lib = std::shared_ptr<IntrinsicLibrary>(new IntrinsicLibrary());
    lib->sourcePath_ = bcFilePath;

    // Build lookup map
    for (const auto& pattern : patterns) {
        if (pattern.type == PatternType::Intrinsic) {
            lib->intrinsics_[pattern.name] = pattern;
        }
    }

    return lib;
}

std::shared_ptr<IntrinsicLibrary> IntrinsicLibrary::create(const std::vector<Pattern>& patterns) {
    auto lib = std::shared_ptr<IntrinsicLibrary>(new IntrinsicLibrary());
    lib->sourcePath_ = "<in-memory>";

    // Build lookup map
    for (const auto& pattern : patterns) {
        if (pattern.type == PatternType::Intrinsic) {
            lib->intrinsics_[pattern.name] = pattern;
        }
    }

    return lib;
}

//===----------------------------------------------------------------------===//
// Lookup Methods
//===----------------------------------------------------------------------===//

const Pattern* IntrinsicLibrary::lookup(const std::string& name) const {
    auto it = intrinsics_.find(name);
    if (it != intrinsics_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool IntrinsicLibrary::hasIntrinsic(const std::string& name) const {
    return intrinsics_.find(name) != intrinsics_.end();
}

std::vector<std::string> IntrinsicLibrary::getIntrinsicNames() const {
    std::vector<std::string> names;
    names.reserve(intrinsics_.size());

    for (const auto& pair : intrinsics_) {
        names.push_back(pair.first);
    }

    // Sort alphabetically for consistent output
    std::sort(names.begin(), names.end());

    return names;
}

//===----------------------------------------------------------------------===//
// Accessor Methods
//===----------------------------------------------------------------------===//

std::vector<IntrinsicArgument> IntrinsicLibrary::getArguments(const std::string& name) const {
    const Pattern* pattern = lookup(name);
    if (pattern) {
        return pattern->arguments;
    }
    return {};
}

std::vector<IntrinsicInstruction> IntrinsicLibrary::getBody(const std::string& name) const {
    const Pattern* pattern = lookup(name);
    if (pattern) {
        return pattern->body;
    }
    return {};
}

std::string IntrinsicLibrary::getComment(const std::string& name) const {
    const Pattern* pattern = lookup(name);
    if (pattern) {
        return pattern->comment;
    }
    return "";
}

bool IntrinsicLibrary::hasPythonBinding(const std::string& name) const {
    const Pattern* pattern = lookup(name);
    if (pattern) {
        return pattern->pythonBinding;
    }
    return false;
}

//===----------------------------------------------------------------------===//
// Utility Methods
//===----------------------------------------------------------------------===//

void IntrinsicLibrary::printStats() const {
    std::cout << "=== IntrinsicLibrary Statistics ===\n";
    std::cout << "Source: " << sourcePath_ << "\n";
    std::cout << "Total Intrinsics: " << intrinsics_.size() << "\n\n";

    // Count Python bindings
    size_t pythonBindingCount = 0;
    for (const auto& pair : intrinsics_) {
        if (pair.second.pythonBinding) {
            pythonBindingCount++;
        }
    }
    std::cout << "Python Bindings: " << pythonBindingCount << "\n\n";

    // List all intrinsics with details
    auto names = getIntrinsicNames();
    std::cout << "Intrinsics:\n";
    for (const auto& name : names) {
        const Pattern* pattern = lookup(name);
        if (pattern) {
            std::cout << "  " << name << "\n";
            std::cout << "    Arguments: " << pattern->arguments.size();
            std::cout << " (";
            for (size_t i = 0; i < pattern->arguments.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << pattern->arguments[i].name;
            }
            std::cout << ")\n";
            std::cout << "    Instructions: " << pattern->body.size() << "\n";
            if (!pattern->comment.empty()) {
                std::cout << "    Comment: " << pattern->comment << "\n";
            }
            std::cout << "    Python Binding: " << (pattern->pythonBinding ? "yes" : "no") << "\n";
        }
    }

    std::cout << "\n=== End Statistics ===\n";
}

}  // namespace stinkytofu
