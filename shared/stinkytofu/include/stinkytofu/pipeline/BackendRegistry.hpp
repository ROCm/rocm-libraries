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

    /// Global registry of pipeline spec populators per architecture.
    ///
    /// Stores one PipelineSpecPopulator per architecture [major, minor, stepping].
    /// Backend(module) calls getArchPopulator(module.getArch()) at construction and, if set,
    /// invokes it to fill a vector of PipelineSpec. Register with setArchPipeline(); typically
    /// from a static initializer in a backend TU (e.g. Gfx1250Backend.cpp). Query with
    /// getArchPopulator(), hasPipelines(), getPipelineCount(); clear with clearArch() or clear().
    ///
    /// @code
    /// // Static registrar (runs when TU is linked)
    /// BackendRegistry::setArchPipeline({12, 5, 0}, myPipelineSpecPopulator);
    /// @endcode
    class BackendRegistry
    {
    public:
        /// Specification for an optimization pipeline.
        struct PipelineSpec
        {
            PipelineConfig config; ///< Pipeline configuration.
            std::string    groupName; ///< Group name this pipeline applies to (optional).
        };

        /// Function type: populate pipeline specifications from a StinkyAsmModule.
        using PipelineSpecPopulator
            = std::function<void(const StinkyAsmModule&, std::vector<PipelineSpec>&)>;

        /// Set the pipeline spec populator for \p arch (one per arch).
        static void setArchPipeline(const std::array<int, 3>& arch,
                                    PipelineSpecPopulator     populator);

        /// Return the pipeline spec populator for \p arch, or empty if none registered.
        static PipelineSpecPopulator getArchPopulator(const std::array<int, 3>& arch);

        /// True if a pipeline spec populator is registered for \p arch.
        static bool hasPipelines(const std::array<int, 3>& arch);

        /// Remove all registered populators (all architectures). Mainly for tests.
        static void clear();

        /// Remove the pipeline spec populator for \p arch.
        static void clearArch(const std::array<int, 3>& arch);

        /// Format arch as arch name.
        static std::string makeArchKey(const std::array<int, 3>& arch);

    private:
        BackendRegistry() = default;

        struct Registry;
        static Registry& getRegistry();
    };

} // namespace stinkytofu
