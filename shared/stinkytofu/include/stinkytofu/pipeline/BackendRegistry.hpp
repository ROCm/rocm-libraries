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
#include <functional>
#include <string>
#include <vector>

#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
class StinkyAsmModule;

/// Global registry of per-architecture pipeline configuration.
///
/// Each architecture registers an ArchPipeline (pipeline builder + group names)
/// at static init time. Backend::runOptimization() retrieves the builder, and
/// module construction code queries group names to set up instruction tracking.
///
/// @code
/// // Static registrar (runs when TU is linked)
/// BackendRegistry::setArchPipeline({12, 5, 0}, {myBuilder, {"group0", "group1"}});
/// @endcode
class BackendRegistry {
   public:
    /// Function type: builds the pipeline for a module into a PassManager.
    /// Returns true if passes were added, false if nothing to do.
    using PipelineBuilder = std::function<bool(PassManager&, StinkyAsmModule&)>;

    /// Per-architecture pipeline configuration.
    struct ArchPipeline {
        PipelineBuilder builder;              ///< Populates a PM with ScopeAdaptor passes.
        std::vector<std::string> groupNames;  ///< Instruction groups the builder operates on.
    };

    /// Register the pipeline for \p arch (one per arch).
    static void setArchPipeline(const std::array<int, 3>& arch, ArchPipeline pipeline);

    /// Return the pipeline for \p arch, or nullptr if none registered.
    static const ArchPipeline* getArchPipeline(const std::array<int, 3>& arch);

    /// Remove all registered entries (all architectures). Mainly for tests.
    static void clear();

    /// Remove all registered entries for \p arch.
    static void clearArch(const std::array<int, 3>& arch);

    /// Format arch as arch name.
    static std::string makeArchKey(const std::array<int, 3>& arch);

   private:
    BackendRegistry() = default;

    struct Registry;
    static Registry& getRegistry();
};

}  // namespace stinkytofu
