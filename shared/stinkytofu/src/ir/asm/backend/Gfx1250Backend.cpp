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

#include "ir/asm/BackendRegistry.hpp"

namespace stinkytofu
{
    namespace
    {
        constexpr std::array<int, 3> GFX1250_ARCH{12, 5, 0};

        // Will enable after milestone 1
        PipelineConfig createDefaultPipeline(const StinkyAsmModule& module,
                                             BasicBlockFilter       bbFilter)
        {
            (void)module;

            // Create pipeline configuration with full scheduling and optimization
            auto config = stinkytofu::PipelineConfig::fromProfile(
                stinkytofu::PipelineProfile::FullPipeline, stinkytofu::OptLevel::O3);

            // TODO: Disable waitcnt for now
            config.enableWaitCnt = false;

            // Configure GEMM-specific tile parameters
            config.withGemmTileConfig(GFX1250_ARCH, 0, 0, 0, 0, 0, 0, 1);

            // Configure pass features (GEMM-specific optimizations)
            config
                .withBarrierSemantics(true) // unrollGemmMovableBarrier
                .withLoopUnroll(true) // unrollGemm
                .withDagFeatures(true); // distributeGlobalRead

            // config.verbose = true;

            // Configure basic block filter
            config.basicBlockFilter = bbFilter;

            return config;
        }

        PipelineConfig createNonOptPipe(const StinkyAsmModule& module, BasicBlockFilter bbFilter)
        {
            (void)module;

            // Will enable after milestone 1
            // Create pipeline configuration with full scheduling and optimization
            auto config = stinkytofu::PipelineConfig::fromProfile(
                stinkytofu::PipelineProfile::NoOptimization, stinkytofu::OptLevel::O3);

            // TODO: Disable waitcnt for now
            config.enableWaitCnt = false;

            // Configure GEMM-specific tile parameters
            config.withGemmTileConfig(GFX1250_ARCH, 0, 0, 0, 0, 0, 0, 1);

            return config;
        }

        struct Gfx1250BackendRegistrar
        {
            Gfx1250BackendRegistrar()
            {
#if 0
                BackendRegistry::addArchPipeline(
                    /* arch */ GFX1250_ARCH,
                    /* builder */
                    [](const StinkyAsmModule& module) {
                        return createDefaultPipeline(
                            module,
                            stinkytofu::BasicBlockFilterBuilder::byLabelPrefix("label_LoopBegin"));
                    },
                    /* groupName */ "loopWithPrefetch");

                BackendRegistry::addArchPipeline(
                    /* arch */ GFX1250_ARCH,
                    /* builder */
                    [](const StinkyAsmModule& module) {
                        return createDefaultPipeline(module,
                                                     stinkytofu::BasicBlockFilterBuilder::all());
                    },
                    /* groupName */ "noLoadLoopBody");
#endif
                BackendRegistry::addArchPipeline(
                    /* arch */ GFX1250_ARCH,
                    /* builder */
                    [](const StinkyAsmModule& module) {
                        return createNonOptPipe(module, stinkytofu::BasicBlockFilterBuilder::all());
                    },
                    /* groupName */ "");
            }
        };
        static Gfx1250BackendRegistrar s_gfx1250BackendRegistrar;
    } // namespace
} // namespace stinkytofu
