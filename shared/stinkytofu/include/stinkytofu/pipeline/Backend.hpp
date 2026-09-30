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

#include <array>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/ModulePassManager.hpp"

namespace stinkytofu {
class StinkyAsmModule;

/// Architecture-specific optimization entry point.
///
/// Bound to a single StinkyAsmModule. Looks up the registered PipelineBuilder
/// for the module's architecture, creates a ModulePassManager, calls the builder
/// to populate it, configures, and runs on the module.
///
/// @code
/// Backend backend(module);
/// backend.runOptimization();
/// @endcode
class STINKYTOFU_EXPORT Backend {
   public:
    explicit Backend(StinkyAsmModule& module);

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    /// Architecture of the member module [major, minor, stepping].
    std::array<int, 3> getArch() const;

    /// Run the full pipeline on the member module.
    /// Looks up the PipelineBuilder, populates a PM, configures, and runs.
    bool runOptimization();

   private:
    /// Configure the ModulePassManager with GemmTileConfig, AsmCapsConfig,
    /// and debug options derived from the module.
    void configurePassManager(ModulePassManager& mpm);

    StinkyAsmModule& module;
};

}  // namespace stinkytofu
