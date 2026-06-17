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
/// When this translation unit is linked, the gfx1250 pipeline builder is registered globally
/// so that Backend(module).runOptimization() automatically picks it up for modules with
/// arch {12, 5, 0}.

#include <algorithm>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/pipeline/OptimizationPasses.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/EstimateAsmCyclesPass.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"
#include "stinkytofu/transforms/asm/InsertDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/LongBranchLoweringPass.hpp"
#include "stinkytofu/transforms/asm/LoopRegionRemarkPass.hpp"
#include "stinkytofu/transforms/asm/MemTokenConsistencyCheckPass.hpp"
#include "stinkytofu/transforms/asm/RederiveExpertScopePass.hpp"
#include "stinkytofu/transforms/asm/RemoveDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/RemoveWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/SetMatrixReusePass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveNopPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"
#include "stinkytofu/transforms/asm/SwPrefetchInsertionPass.hpp"

namespace stinkytofu {
namespace {
constexpr std::array<int, 3> GFX1250_ARCH{12, 5, 0};

/// Build the gfx1250 per-region optimization passes into a PassManager.
/// TODO: enableWaitCnt is a per-pass toggle for the
/// bring-up phase. Once the pipeline stabilizes, pass selection should
/// be controlled by OptLevel.
void addGfx1250RegionPasses(PassManager& pm, const StinkyAsmModule& module, OptLevel optLevel,
                            bool enableWaitCnt, bool runScheduler) {
    // Verify IR integrity before running any passes
    // This catches IR corruption early before it propagates through optimization
    pm.addPass(createStinkyIRVerifierPass());

    pm.addPass(createCFGBuilderPass());
    if (enableWaitCnt) {
        pm.addPass(createStinkyRemoveWaitCntPass());
        pm.addPass(createStinkyRemoveNopPass());
    }

    // addPeepholeOptPasses(pm, optLevel);

    // Instruction scheduling
    pm.addPass(createStinkyBuildImplicitDependencyPass());
    if (runScheduler) {
        pm.addPass(createStinkyDAGSchedulerPass());
    }
}

/// Build the full gfx1250 pipeline into \p pm using ScopeAdaptors.
/// TODO: EnableWaitCntInsertion is a per-pass toggle for the
/// bring-up phase. Once the pipeline stabilizes, pass selection should
/// be controlled by OptLevel.
bool buildGfx1250Pipeline(PassManager& pm, StinkyAsmModule& module, const PassBuilder& PB) {
    const auto& moduleOptions = module.getModuleOptions();
    const OptLevel optLevel = static_cast<OptLevel>(
        std::max(0, std::min(moduleOptions.OptLevel, static_cast<int>(OptLevel::O3))));
    registerAllAnalyses(pm.getAnalysisManager());

    auto debugStreams = createDebugOutputStreams(moduleOptions);
    configureDebugOutput(pm, moduleOptions, "kernel-OuterPM", debugStreams);

    const bool runScheduler = optLevel != OptLevel::O0;
    if (runScheduler || moduleOptions.EnableESM2) {
        // strip delay_alu before scheduling
        pm.addPass(createRemoveDelayAluPass());
        // strip s_wait_alu before scheduling
        pm.addPass(createRemoveWaitAluPass());
    }
    PB.applyExtensionPoint(PipelineExtensionPoint::BeforeRegionPasses, pm, module);

    // -- region: loopWithPrefetch + noLoadLoopBody --
    // Both the DAG scheduler (O3) and waitcnt insertion need the region-scoped CFG, so they
    // share one region adaptor. Either gate is enough to enter this block.
    if (runScheduler || moduleOptions.EnableWaitCntInsertion) {
        PassFeatureConfig passFeatureConfig;
        std::shared_ptr<DAGScheduleJsonCollector> snapshotCollector;
        if (runScheduler) {
            passFeatureConfig.loopConfig.unrollGemm = true;
            passFeatureConfig.dagFeatures.distributeGlobalRead = true;
            passFeatureConfig.passOrderSnapshot.jsonPath = moduleOptions.PassOrderSnapshotJson;
            snapshotCollector = createPassOrderSnapshotCollector(passFeatureConfig, moduleOptions,
                                                                 module.getName());
            passFeatureConfig.passOrderSnapshot.titlePrefix = "loopWithPrefetch+noLoadLoopBody";
        }

        PassManager innerPM;
        registerAllAnalyses(innerPM.getAnalysisManager());
        innerPM.setPassFeatureConfig(passFeatureConfig);
        if (snapshotCollector) {
            configurePassOrderSnapshot(innerPM, snapshotCollector);
        }
        configureDebugOutput(innerPM, moduleOptions, "loopWithPrefetch+noLoadLoopBody",
                             debugStreams);
        PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionBegin, innerPM, module);
        addGfx1250RegionPasses(innerPM, module, optLevel, moduleOptions.EnableWaitCntInsertion,
                               runScheduler);
        PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, innerPM, module);
        if (moduleOptions.EnableWaitCntInsertion) {
            innerPM.addPass(createStinkyWaitCntInsertionPass());
        }
        pm.addPass(createKernelToRegionsPassAdaptor(module, {"loopWithPrefetch", "noLoadLoopBody"},
                                                    std::move(innerPM)));
    }

    PB.applyExtensionPoint(PipelineExtensionPoint::AfterRegionPasses, pm, module);

    // -- kernel --

    // Cluster-barrier insertion (kernel scope) — runs at every OptLevel when
    // the module opts in. Must precede InsertVgprMsbPass so the new
    // branches/labels are present when MSB configuration is materialized.
    if (moduleOptions.ClusterBarrier) {
        pm.addPass(createInsertClusterBarrierPass(/*isKernelScope=*/true,
                                                  /*pgrValue=*/moduleOptions.PrefetchGlobalRead,
                                                  /*plrValue=*/moduleOptions.PrefetchLocalRead));
    }

    pm.addPass(createInsertVgprMsbPass());

    if (moduleOptions.EnableESM2) {
        // expertScheduleMode2 region (label_ASM_Start..noLoadLoopBody): wait-alu + mode2
        // lifecycle. Must precede the kernel-wide CFGBuilder — ScopeAdaptor needs the flat
        // single-BB function. Re-derive its range first (see RederiveExpertScopePass).
        {
            pm.addPass(createRederiveExpertScopePass(module, "expertScheduleMode2",
                                                     "label_ASM_Start", "noLoadLoopBody"));
            PassManager innerPM;
            registerAllAnalyses(innerPM.getAnalysisManager());
            configureDebugOutput(innerPM, moduleOptions, "expertScheduleMode2", debugStreams);
            innerPM.addPass(createLongBranchLoweringPass());
            innerPM.addPass(createCFGBuilderPass());
            innerPM.addPass(createInsertWaitAluPass());
            pm.addPass(
                createKernelToRegionPassAdaptor(module, "expertScheduleMode2", std::move(innerPM)));
        }
    }

    pm.addPass(createCFGBuilderPass());
    pm.addPass(createMemTokenConsistencyCheckPass());

    if (runScheduler) {
        pm.addPass(createInsertDelayAluPass(/*minWavesPerSimd=*/2));
        pm.addPass(createLoopRegionRemarkPass());
    }
    pm.addPass(createEstimateAsmCyclesPass());
    // Whole-kernel reuse on final instruction order (O0 and O1+; after scheduler + VGPR MSB).
    pm.addPass(createSetMatrixReusePass());
    if (moduleOptions.EnableSwPrefetchInsertion) {
        pm.addPass(createSwPrefetchInsertionPass(module));
    }
    // When StinkyTofuCostOutputDir is set, dump pass debug (per-instruction + summary) to
    // <outputDir>/<kernel>/accumulate_instruction_size_pass_debug.txt (same layout as Backend).
    pm.addPass(createAccumulateInstructionSizePass(module));

    return true;
}

struct Gfx1250Registrar {
    Gfx1250Registrar() {
        BackendRegistry::setArchPipeline(
            GFX1250_ARCH,
            {buildGfx1250Pipeline, {"loopWithPrefetch", "noLoadLoopBody", "expertScheduleMode2"}});
    }
};
static Gfx1250Registrar s_gfx1250Registrar;
}  // namespace

void anchorGfx1250Backend() {}  // NOLINT(misc-use-internal-linkage)

}  // namespace stinkytofu
