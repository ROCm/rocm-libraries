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
#include <vector>

#include "stinkytofu/ir/logical/IntrinsicLibrary.hpp"

namespace stinkytofu {
/**
 * @brief Global registry for intrinsic libraries (Singleton)
 *
 * Similar to LLVM's compiler driver, this automatically finds and loads
 * intrinsics.st.bc at initialization time. The registry searches for the
 * bitcode file in standard installation paths.
 *
 * Usage (automatic):
 *   // In TensileLite or any code generator
 *   auto& registry = IntrinsicRegistry::instance();
 *   if (registry.hasIntrinsic("ReluF32")) {
 *       auto pattern = registry.lookup("ReluF32");
 *       // Use pattern to generate code
 *   }
 *
 * Search paths (in order):
 *   1. Environment variable: STINKYTOFU_INTRINSICS_PATH
 *   2. Current directory: ./intrinsics.st.bc
 *   3. Build directory: <build>/intrinsics.st.bc
 *   4. Install directory: <install-prefix>/lib/stinkytofu/intrinsics.st.bc
 *   5. Relative to library: ../lib/stinkytofu/intrinsics.st.bc
 */
class IntrinsicRegistry {
   public:
    /**
     * @brief Get the global intrinsic registry (singleton)
     *
     * On first call, automatically searches for and loads intrinsics.st.bc.
     * Subsequent calls return the same instance.
     *
     * @return Reference to the global registry
     */
    static IntrinsicRegistry& instance();

    /**
     * @brief Check if an intrinsic is available
     *
     * @param name Intrinsic name
     * @return true if intrinsic exists
     */
    bool hasIntrinsic(const std::string& name) const;

    /**
     * @brief Look up an intrinsic pattern
     *
     * @param name Intrinsic name
     * @return Pointer to Pattern, or nullptr if not found
     */
    const Pattern* lookup(const std::string& name) const;

    /**
     * @brief Get all available intrinsic names
     *
     * @return Vector of intrinsic names
     */
    std::vector<std::string> getIntrinsicNames() const;

    /**
     * @brief Check if the registry was initialized successfully
     *
     * @return true if intrinsics.st.bc was loaded
     */
    bool isInitialized() const;

    /**
     * @brief Get the path where intrinsics.st.bc was loaded from
     *
     * @return Path to loaded .st.bc file, or empty string if not loaded
     */
    std::string getLoadedPath() const;

    /**
     * @brief Reload intrinsics from a specific path (for testing/debugging)
     *
     * @param path Path to intrinsics.st.bc
     * @return true if reload successful
     */
    bool reload(const std::string& path);

    /**
     * @brief Get the underlying IntrinsicLibrary
     *
     * @return Shared pointer to library, or nullptr if not loaded
     */
    std::shared_ptr<IntrinsicLibrary> getLibrary() const;

   private:
    // Private constructor for singleton
    IntrinsicRegistry();

    // Disable copy and move
    IntrinsicRegistry(const IntrinsicRegistry&) = delete;
    IntrinsicRegistry& operator=(const IntrinsicRegistry&) = delete;

    /**
     * @brief Search for intrinsics.st.bc in standard locations
     *
     * @return Path to found file, or empty string if not found
     */
    std::string findIntrinsicsBitcode();

    /**
     * @brief Get list of search paths
     *
     * @return Vector of paths to search
     */
    std::vector<std::string> getSearchPaths();

    /**
     * @brief Get the loaded shared library path
     *
     * Uses dladdr() on Linux to find where libstinkytofu.so is loaded from.
     * This works correctly for both Python modules and C++ executables.
     *
     * @return Path to the loaded library or empty string
     */
    std::string getLibraryPath();

    /**
     * @brief Get the executable path
     *
     * Uses /proc/self/exe on Linux to find the main executable.
     * Note: For Python modules, this returns the Python interpreter path.
     *
     * @return Path to current executable or library
     */
    std::string getExecutablePath();

    std::shared_ptr<IntrinsicLibrary> library_;
    std::string loadedPath_;
    bool initialized_;
};

}  // namespace stinkytofu
