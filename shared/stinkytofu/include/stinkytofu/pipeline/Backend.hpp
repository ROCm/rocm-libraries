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

#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/pipeline/OptimizationPipeline.hpp"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace stinkytofu
{
    class StinkyAsmModule;

    /// Architecture-specific optimization entry point.
    ///
    /// Bound to a single StinkyAsmModule (the member module). At construction, loads
    /// pipeline spec populators for module.getArch() from BackendRegistry, invokes them
    /// to fill a sequence of PipelineSpec, and runs that sequence in runOptimization().
    /// Use clearPipelines() or setPipelines(configs) to replace the instance's sequence.
    /// Register per-arch populators via BackendRegistry.
    ///
    /// @code
    /// StinkyAsmModule module("kernel", {12, 5, 0});
    /// Backend backend(module);  // loads populators for 12.5.0 from BackendRegistry
    /// backend.runOptimization();
    /// // Or: backend.setPipelines(myConfigs); backend.runOptimization();
    /// @endcode
    class Backend
    {
    public:
        /// Construct backend for \p module; loads pipeline spec populators for module.getArch() from BackendRegistry.
        explicit Backend(StinkyAsmModule& module);

        ~Backend();

        Backend(const Backend&)            = delete;
        Backend& operator=(const Backend&) = delete;
        Backend(Backend&&) noexcept;
        Backend& operator=(Backend&&) noexcept = delete;

        /// Architecture of the member module [major, minor, stepping].
        std::array<int, 3> getArch() const;

        /// True if this backend has a non-empty pipeline config sequence.
        bool hasPipelines() const;

        /// Number of pipelines in the current sequence.
        size_t getPipelineCount() const;

        /// Remove all pipelines (sequence becomes empty).
        void clearPipelines();

        /// Replace the pipeline sequence with \p configs; they are run in order.
        void setPipelines(std::vector<PipelineConfig> configs);

        /// Get the pipelines
        const std::vector<BackendRegistry::PipelineSpec>& getPipelines() const;

        /// Run the full pipeline sequence on the member module.
        bool runOptimization();

        /// Run a single PipelineConfig on the member module (e.g. for a specific group).
        bool runOptimizationWithConfig(const PipelineConfig& config,
                                       const std::string&    groupName = "");

    private:
        bool runPipelineSequence();

        /// Reinsert waitcnts in the optimized range
        bool reinsertWaitCntsInOptimizedRange();

        struct Impl;
        std::unique_ptr<Impl> pImpl;
        StinkyAsmModule&      module;
    };

} // namespace stinkytofu
