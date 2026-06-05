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
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/serialization/asm/PatternParser.hpp"

namespace stinkytofu {
/**
 * @brief Runtime library for loading and managing pre-compiled intrinsics
 *
 * This class loads intrinsics from .st.bc files and provides lookup/expansion
 * functions for TensileLite and other code generators.
 *
 * Usage:
 *   auto lib = IntrinsicLibrary::loadFromFile("intrinsics.st.bc");
 *   if (auto intrinsic = lib->lookup("ReluF32")) {
 *       // Use intrinsic definition
 *   }
 */
class STINKYTOFU_EXPORT IntrinsicLibrary {
   public:
    /**
     * @brief Load intrinsic library from .st.bc file
     *
     * @param bcFilePath Path to the .st.bc file
     * @return Shared pointer to IntrinsicLibrary, or nullptr on failure
     */
    static std::shared_ptr<IntrinsicLibrary> loadFromFile(const std::string& bcFilePath);

    /**
     * @brief Create library from in-memory patterns
     *
     * @param patterns Vector of intrinsic patterns
     * @return Shared pointer to IntrinsicLibrary
     */
    static std::shared_ptr<IntrinsicLibrary> create(const std::vector<Pattern>& patterns);

    /**
     * @brief Look up an intrinsic by name
     *
     * @param name Intrinsic name (e.g., "ReluF32")
     * @return Pointer to Pattern if found, nullptr otherwise
     */
    const Pattern* lookup(const std::string& name) const;

    /**
     * @brief Check if an intrinsic exists
     *
     * @param name Intrinsic name
     * @return true if intrinsic exists, false otherwise
     */
    bool hasIntrinsic(const std::string& name) const;

    /**
     * @brief Get all intrinsic names
     *
     * @return Vector of intrinsic names
     */
    std::vector<std::string> getIntrinsicNames() const;

    /**
     * @brief Get total number of intrinsics
     *
     * @return Number of loaded intrinsics
     */
    size_t size() const {
        return intrinsics_.size();
    }

    /**
     * @brief Get arguments for an intrinsic
     *
     * @param name Intrinsic name
     * @return Vector of arguments, or empty vector if not found
     */
    std::vector<IntrinsicArgument> getArguments(const std::string& name) const;

    /**
     * @brief Get body instructions for an intrinsic
     *
     * @param name Intrinsic name
     * @return Vector of instructions, or empty vector if not found
     */
    std::vector<IntrinsicInstruction> getBody(const std::string& name) const;

    /**
     * @brief Get comment for an intrinsic
     *
     * @param name Intrinsic name
     * @return Comment string, or empty string if not found
     */
    std::string getComment(const std::string& name) const;

    /**
     * @brief Check if intrinsic has Python binding
     *
     * @param name Intrinsic name
     * @return true if Python binding is enabled, false otherwise
     */
    bool hasPythonBinding(const std::string& name) const;

    /**
     * @brief Print library statistics to stdout
     */
    void printStats() const;

   private:
    // Private constructor - use static factory methods
    IntrinsicLibrary() = default;

    // Map from intrinsic name to Pattern
    std::unordered_map<std::string, Pattern> intrinsics_;

    // Path to the loaded .st.bc file (for debugging)
    std::string sourcePath_;
};

}  // namespace stinkytofu
