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
/// @file Gfx942Backend.cpp
/// @brief Registers the gfx942 (CDNA3, arch 9.4.2) optimization pipeline with BackendRegistry.
///
/// When this translation unit is linked, the gfx942 pipelines are registered globally
/// so that Backend(module) automatically picks them up for modules with arch {9, 4, 2}.

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"

#include <algorithm>

namespace stinkytofu
{
    namespace
    {
        constexpr std::array<int, 3> GFX942_ARCH{9, 4, 2};

        // Will enable after milestone 1
        PipelineConfig createDefaultPipeline(const StinkyAsmModule& module,
                                             BasicBlockFilter       bbFilter,
                                             const std::string&     groupName,
                                             OptLevel               optLevel)
        {
            // Create pipeline configuration with full scheduling and optimization
            auto config = stinkytofu::PipelineConfig::fromProfile(
                stinkytofu::PipelineProfile::FullPipeline, optLevel);

            // TODO: Disable waitcnt for now
            config.enableWaitCnt = false;

            // Configure GEMM-specific tile parameters
            const auto& moduleOptions = module.getModuleOptions();
            config.withGemmTileConfig(GFX942_ARCH,
                                      moduleOptions.TileA0,
                                      moduleOptions.TileB0,
                                      moduleOptions.TileM0,
                                      moduleOptions.NumGRA,
                                      moduleOptions.NumGRB,
                                      moduleOptions.NumGRM,
                                      moduleOptions.wavefrontSize);

            // Configure pass features (GEMM-specific optimizations)
            config
                .withBarrierSemantics(true) // unrollGemmMovableBarrier
                .withLoopUnroll(true) // unrollGemm
                .withDagFeatures(true); // distributeGlobalRead

            // Configure basic block filter
            config.basicBlockFilter = bbFilter;

            // Apply debug output from moduleOptions (set via GlobalParameters).
            config.withDebugLevel(moduleOptions.DebugLevel, groupName)
                .withPrintPasses(
                    moduleOptions.PrintBeforePass, moduleOptions.PrintAfterPass, groupName)
                .withDebugPass(moduleOptions.DebugPass);

            return config;
        }

        /**
         * @brief Populate the pipeline specifications for the GFX942 architecture.
         * @param module The StinkyAsmModule to populate the pipeline specifications for.
         * @param specs The vector of pipeline specifications to populate.
         */
        void pipelineSpecPopulator(const StinkyAsmModule&                      module,
                                   std::vector<BackendRegistry::PipelineSpec>& specs)
        {
            const auto& moduleOptions = module.getModuleOptions();

            // Use moduleOptions.OptLevel directly (set via GlobalParameters).
            const OptLevel optLevel = static_cast<OptLevel>(
                std::max(0, std::min(moduleOptions.OptLevel, static_cast<int>(OptLevel::O3))));

            /* Add pipelines for O3 optimization level */
            specs.emplace_back(
                /* PipelineConfig */ createDefaultPipeline(
                    module,
                    stinkytofu::BasicBlockFilterBuilder::byLabelPrefix("label_LoopBegin"),
                    "loopWithPrefetch",
                    optLevel),
                /* groupName */ "loopWithPrefetch");

            specs.emplace_back(
                /* PipelineConfig */ createDefaultPipeline(
                    module, stinkytofu::BasicBlockFilterBuilder::all(), "noLoadLoopBody", optLevel),
                /* groupName */ "noLoadLoopBody");
        }

        struct Gfx942BackendRegistrar
        {
            Gfx942BackendRegistrar()
            {
                BackendRegistry::setArchPipeline(GFX942_ARCH, pipelineSpecPopulator);
            }
        };
        static Gfx942BackendRegistrar s_gfx942BackendRegistrar;
    } // namespace
} // namespace stinkytofu
