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
/// @file Gfx1250Backend.cpp
/// @brief Registers the gfx1250 (RDNA4, arch 12.5.0) optimization pipeline with BackendRegistry.
///
/// When this translation unit is linked, the gfx1250 pipelines are registered globally
/// so that Backend(module) automatically picks them up for modules with arch {12, 5, 0}.

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"

namespace stinkytofu
{
    namespace
    {
        constexpr std::array<int, 3> GFX1250_ARCH{12, 5, 0};

        // Will enable after milestone 1
        PipelineConfig createDefaultPipeline(const StinkyAsmModule& module,
                                             BasicBlockFilter       bbFilter,
                                             const std::string&     groupName)
        {
            // Create pipeline configuration with full scheduling and optimization
            auto config = stinkytofu::PipelineConfig::fromProfile(
                stinkytofu::PipelineProfile::FullPipeline, stinkytofu::OptLevel::O3);

            // TODO: Disable waitcnt for now
            config.enableWaitCnt = false;

            // Configure GEMM-specific tile parameters
            const auto& moduleOptions = module.getModuleOptions();
            config.withGemmTileConfig(GFX1250_ARCH,
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

            // config.verbose = true;

            // Configure basic block filter
            config.basicBlockFilter = bbFilter;

            // Enable to print the IR after each pass
            if(moduleOptions.dumpIRBetweenPasses)
            {
                config.debugConfig = std::make_unique<stinkytofu::PassManagerDebugConfig>();
                config.debugConfig->setPrintAfterAll(true);
                config.debugConfig->setDumpToFileInAfter(groupName + "-after_passes.txt");
            }

            return config;
        }

        void addPipeline(BasicBlockFilter bbFilter, const std::string& groupName)
        {
            BackendRegistry::addArchPipeline(
                GFX1250_ARCH,
                [bbFilter, groupName](const StinkyAsmModule& module) {
                    return createDefaultPipeline(module, bbFilter, groupName);
                },
                groupName);
        }

        struct Gfx1250BackendRegistrar
        {
            Gfx1250BackendRegistrar()
            {
                addPipeline(
                    /* bbFilter */ stinkytofu::BasicBlockFilterBuilder::byLabelPrefix(
                        "label_LoopBegin"),
                    /* groupName */ "loopWithPrefetch");
                addPipeline(
                    /* bbFilter */ stinkytofu::BasicBlockFilterBuilder::all(),
                    /* groupName */ "noLoadLoopBody");
            }
        };
        static Gfx1250BackendRegistrar s_gfx1250BackendRegistrar;
    } // namespace
} // namespace stinkytofu
