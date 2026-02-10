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

#include "stinkytofu/pipeline/OptimizationPipeline.hpp"

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace stinkytofu
{
    class StinkyAsmModule;

    /// Global registry of pipeline factories per architecture.
    ///
    /// Stores, per architecture [major, minor, stepping], an ordered sequence of
    /// PipelineFactory entries. Backend(module) calls getPipelineFactories(module.getArch())
    /// at construction and invokes each factory to build its PipelineConfig sequence.
    /// Register factories with addArchPipeline() or setArchPipelines(); typically from
    /// a static initializer in a backend TU (e.g. Gfx1250Backend.cpp). Query with
    /// getPipelineFactories(), hasPipelines(), getPipelineCount(); clear with
    /// clearArch() or clear().
    ///
    /// @code
    /// // Static registrar (runs when TU is linked)
    /// BackendRegistry::addArchPipeline({12, 5, 0}, createGfx1250PeepholePipeline);
    /// // Or replace the full factory sequence for an arch:
    /// BackendRegistry::setArchPipelines({12, 5, 0}, {factory1, factory2});
    /// @endcode
    class BackendRegistry
    {
    public:
        /// Function type: build a PipelineConfig from a StinkyAsmModule.
        using PipelineConfigBuilder = std::function<PipelineConfig(const StinkyAsmModule&)>;

        /// One pipeline factory: builder + optional group name for filtering.
        struct PipelineFactory
        {
            PipelineConfigBuilder builder; ///< Builds PipelineConfig from a module.
            std::string
                groupName; ///< Group name this pipeline applies to (e.g. for runOptimizationWithConfig).
        };

        /// Append a pipeline factory to the sequence for \p arch.
        static void addArchPipeline(const std::array<int, 3>& arch,
                                    PipelineConfigBuilder     builder,
                                    const std::string&        groupName = "");

        /// Set the full pipeline factory sequence for \p arch (replaces any existing).
        static void setArchPipelines(const std::array<int, 3>&           arch,
                                     const std::vector<PipelineFactory>& factories);

        /// Return the pipeline factories registered for \p arch (copy).
        static std::vector<PipelineFactory> getPipelineFactories(const std::array<int, 3>& arch);

        /// True if any pipeline factories are registered for \p arch.
        static bool hasPipelines(const std::array<int, 3>& arch);

        /// Number of pipeline factories registered for \p arch.
        static size_t getPipelineCount(const std::array<int, 3>& arch);

        /// Remove all registered pipeline factories (all architectures). Mainly for tests.
        static void clear();

        /// Remove pipeline factories for \p arch only.
        static void clearArch(const std::array<int, 3>& arch);

        /// Format arch as arch name.
        static std::string makeArchKey(const std::array<int, 3>& arch);

    private:
        BackendRegistry() = default;

        struct Registry;
        static Registry& getRegistry();
    };

} // namespace stinkytofu
