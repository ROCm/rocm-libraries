/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <optional>
#include <string>

// Load / store interface for YAML test config files
namespace hiptensor
{
    template <typename ConfigT>
    struct YamlConfigLoader
    {
        static std::optional<ConfigT> loadFromFile(std::string const& filePath);
        static std::optional<ConfigT> loadFromString(std::string const& yaml = "");
        static void storeToFile(std::string const& filePath, ConfigT const& config);
    };

    // Flush LLVM's buffered output streams and shut down LLVM's ManagedStatic
    // objects. Must be called before returning from main() in any binary that
    // uses LLVM YAML I/O, to prevent use-after-free at process exit when the
    // DLL unload order differs from a direct run (e.g. under ctest on Windows).
    void llvmShutdown();
}
