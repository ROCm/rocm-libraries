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
/// When this translation unit is linked, the gfx942 pipeline builder is registered globally
/// so that Backend(module).runOptimization() automatically picks it up for modules with
/// arch {9, 4, 2}.

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/pipeline/OptimizationPasses.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/ScheduleFirstLRsPass.hpp"
#include "stinkytofu/transforms/asm/ScheduleLastLRsPass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

#include <algorithm>

namespace stinkytofu
{
    namespace
    {
        constexpr std::array<int, 3> GFX942_ARCH{9, 4, 2};

        /// Build the gfx942 per-region optimization passes into a PassManager.
        void
            addGfx942RegionPasses(PassManager& pm, const StinkyAsmModule& module, OptLevel optLevel)
        {
            // ========== Phase 0: IR Verification ==========
            pm.addPass(createStinkyIRVerifierPass());

            // ========== Phase 1: CFG Building ==========
            pm.addPass(createCFGBuilderPass());

            // ========== Phase 1.5: Remove WaitCnts ==========
            pm.addPass(createStinkyRemoveWaitCntPass());

            // ========== Phase 2: Optimization ==========
            addOptimizationPasses(pm, optLevel);

            // ========== Phase 3: Instruction Scheduling ==========
            pm.addPass(createStinkyBuildImplicitDependencyPass());
            pm.addPass(createStinkyDAGSchedulerPass());
            pm.addPass(createScheduleLastLRsPass());
        }

        /// Build the full gfx942 pipeline into \p pm using ScopeAdaptors.
        bool buildGfx942Pipeline(PassManager& pm, StinkyAsmModule& module)
        {
            const auto&    moduleOptions = module.getModuleOptions();
            const OptLevel optLevel      = static_cast<OptLevel>(
                std::max(0, std::min(moduleOptions.OptLevel, static_cast<int>(OptLevel::O3))));

            if(optLevel != OptLevel::O0)
            {
                // Single-region adapter: loopWithPrefetch
                {
                    PassManager innerPM;
                    innerPM.setBasicBlockFilter(
                        BasicBlockFilterBuilder::byLabelPrefix("label_LoopBegin"));
                    addGfx942RegionPasses(innerPM, module, optLevel);
                    pm.addPass(createKernelToRegionPassAdaptor(
                        module, "loopWithPrefetch", std::move(innerPM)));
                }

                // Single-region adapter: noLoadLoopBody
                {
                    PassManager innerPM;
                    addGfx942RegionPasses(innerPM, module, optLevel);
                    pm.addPass(createKernelToRegionPassAdaptor(
                        module, "noLoadLoopBody", std::move(innerPM)));
                }

                // Multi-region adapter for waitcnt reinsertion
                {
                    PassManager waitcntPM;
                    waitcntPM.addPass(createCFGBuilderPass());
                    waitcntPM.addPass(createStinkyRemoveWaitCntPass());
                    waitcntPM.addPass(createStinkyWaitCntInsertionPass(true));
                    pm.addPass(createKernelToRegionsPassAdaptor(
                        module, {"loopWithPrefetch", "noLoadLoopBody"}, std::move(waitcntPM)));
                }
            }

            return true;
        }

        struct Gfx942Registrar
        {
            Gfx942Registrar()
            {
                BackendRegistry::setArchPipeline(
                    GFX942_ARCH, {buildGfx942Pipeline, {"loopWithPrefetch", "noLoadLoopBody"}});
            }
        };
        static Gfx942Registrar s_gfx942Registrar;
    } // namespace
} // namespace stinkytofu
