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
#include "stinkytofu/pipeline/ModuleAdaptors.hpp"
#include "stinkytofu/pipeline/OptimizationPasses.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/EstimateAsmCyclesPass.hpp"
#include "stinkytofu/transforms/asm/FlattenCalleesPass.hpp"
#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"
#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"
#include "stinkytofu/transforms/asm/InsertDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/InsertInitialUnclausedVmemPass.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/LoopRegionRemarkPass.hpp"
#include "stinkytofu/transforms/asm/MemTokenConsistencyCheckPass.hpp"
#include "stinkytofu/transforms/asm/RederiveExpertScopePass.hpp"
#include "stinkytofu/transforms/asm/RegionClonePass.hpp"
#include "stinkytofu/transforms/asm/RemoveDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/RemoveDscntPass.hpp"
#include "stinkytofu/transforms/asm/RemoveInstructionPass.hpp"
#include "stinkytofu/transforms/asm/RemoveWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/SetMatrixReusePass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveNopPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsDynamicPass.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsStaticPass.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelDynamicPass.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelStaticPass.hpp"
#include "stinkytofu/transforms/asm/TDMLoadWaveSyncPass.hpp"

namespace stinkytofu {
namespace {
// Deliberately a literal triple rather than getArchTriple(GfxArchID::Gfx1250): this file is
// compiled into every build, including a Gfx1250v0-only one where that enumerator does not
// exist. Keying on {12,5,0} is also what gives v0 v1's pipeline, which is correct -- the two
// steppings differ in instruction timing, not in which passes should run.
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
        // TODO: remove this temporary SIA4/SIA0 split once a dedicated hazard pass
        // handles xcnt placement.
        pm.addPass(createStinkyRemoveWaitCntPass(/*removeTensorWaitCnt=*/true,
                                                 /*removeXcntWaitCnt=*/optLevel == OptLevel::O3));
        pm.addPass(createStinkyRemoveNopPass());
    }

    // addPeepholeOptPasses(pm, optLevel);

    // Instruction scheduling
    pm.addPass(createStinkyBuildImplicitDependencyPass());
    if (runScheduler) {
        pm.addPass(createStinkyDAGSchedulerPass());
    }
}

/// A fresh entry-scoped PassManager with analyses + standard instrumentation.
PassManager makeEntryPM(const StinkyAsmModule& module,
                        const std::shared_ptr<DebugOutputStreams>& debugStreams) {
    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    configureStandardInstrumentations(pm, module.getModuleOptions(), "module", debugStreams,
                                      &module);
    return pm;
}

/// Build the full gfx1250 pipeline into \p mpm.
///
/// Passes are grouped into adaptors added to the module pass manager in order.
/// Each adaptor wraps a single pass or a bucket (PassManager) and differs only
/// in scope:
///   - createFunctionToModuleAdaptor: runs on entry + every callable function
///   - createMainOnlyAdaptor:         runs on the entry only
bool buildGfx1250Pipeline(ModulePassManager& mpm, StinkyAsmModule& module, const PassBuilder& PB) {
    const auto& moduleOptions = module.getModuleOptions();
    const OptLevel optLevel = static_cast<OptLevel>(
        std::max(0, std::min(moduleOptions.OptLevel, static_cast<int>(OptLevel::O3))));
    const bool runScheduler = optLevel != OptLevel::O0;
    auto debugStreams = createDebugOutputStreams(moduleOptions);

    configureModuleInstrumentations(mpm, moduleOptions, "module", debugStreams, &module);

    if (runScheduler || moduleOptions.EnableESM2) {
        // strip delay_alu before scheduling (whole-kernel: entry + callable functions,
        // so stale delay_alu does not survive into the emitted stream)
        mpm.addPass(createFunctionToModuleAdaptor(createRemoveDelayAluPass()));
        // strip s_wait_alu before scheduling (whole-kernel)
        mpm.addPass(createFunctionToModuleAdaptor(createRemoveWaitAluPass()));
    }

    // Region scheduling + kernel setup (entry only).
    {
        PassManager pm = makeEntryPM(module, debugStreams);
        pm.addPass(createStinkyRemoveNopPass(/*vNopOnly=*/true));
        PB.applyExtensionPoint(PipelineExtensionPoint::BeforeRegionPasses, pm, module);

        // -- region: loopWithPrefetch + noLoadLoopBody --
        // Both the DAG scheduler (O3) and waitcnt insertion need the region-scoped CFG, so they
        // share one region adaptor. Either gate is enough to enter this block.
        if (runScheduler || moduleOptions.EnableWaitCntInsertion) {
            PassFeatureConfig passFeatureConfig;
            if (runScheduler) {
                passFeatureConfig.loopConfig.unrollGemm = true;
                passFeatureConfig.dagFeatures.distributeGlobalRead = true;
                passFeatureConfig.dagFeatures.dsReadQueueDepth = moduleOptions.DsReadQueueDepth;
                passFeatureConfig.dagFeatures.dsReadDrainLatency = moduleOptions.DsReadDrainLatency;
                passFeatureConfig.dagFeatures.dsReadThrottleLatency =
                    moduleOptions.DsReadThrottleLatency;
                passFeatureConfig.dagFeatures.globalReadQueueDepth =
                    moduleOptions.GlobalReadQueueDepth;
                passFeatureConfig.dagFeatures.globalReadDrainLatency =
                    moduleOptions.GlobalReadDrainLatency;
                if (moduleOptions.DsReadPerWmma >= 0)
                    passFeatureConfig.dagFeatures.dsReadPerWmma = moduleOptions.DsReadPerWmma;
                if (moduleOptions.DsReadOrder >= 0)
                    passFeatureConfig.dagFeatures.dsReadOrder =
                        static_cast<PassFeatureConfig::DsReadOrder>(moduleOptions.DsReadOrder);
            }

            PassManager innerPM;
            registerAllAnalyses(innerPM.getAnalysisManager());
            innerPM.setPassFeatureConfig(passFeatureConfig);
            configureStandardInstrumentations(innerPM, moduleOptions,
                                              "loopWithPrefetch+noLoadLoopBody", debugStreams);
            PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionBegin, innerPM, module);
            addGfx1250RegionPasses(innerPM, module, optLevel, moduleOptions.EnableWaitCntInsertion,
                                   runScheduler);
            PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, innerPM, module);
            if (moduleOptions.EnableWaitCntInsertion) {
                WaitCntInsertionOptions waitCntOptions;
                waitCntOptions.enableLoopCarriedTokenDeps =
                    moduleOptions.EnableLoopCarriedTokenDeps;
                innerPM.addPass(createStinkyWaitCntInsertionPass(waitCntOptions));
                if (runScheduler) innerPM.addPass(createRemoveDscntPass());
            }
            pm.addPass(createKernelToRegionsPassAdaptor(
                module, {"loopWithPrefetch", "noLoadLoopBody"}, std::move(innerPM)));
        }

        PB.applyExtensionPoint(PipelineExtensionPoint::AfterRegionPasses, pm, module);

        // Cluster-barrier insertion (kernel scope) — runs at every OptLevel when
        // the module opts in. Must precede InsertVgprMsbPass so the new
        // branches/labels are present when MSB configuration is materialized.
        if (moduleOptions.ClusterBarrier) {
            pm.addPass(createInsertClusterBarrierPass());
        }

        // Build the CFG after the flat region splice-backs so RegionClonePass can match its
        // start BB by label. InsertVgprMsb runs after RegionClonePass so the cloned BB gets
        // its MSB computed for its actual operands (chain-head src C is zeroed, so it must not
        // inherit the loop's src C MSB).
        pm.addPass(createCFGBuilderPass());

        // TDM load wave-sync barrier insertion (kernel scope). Must run after tensorcnt
        // insertion (StinkyWaitCntInsertionPass, in the region adaptor above), so the
        // s_wait_tensorcnt structure it keys on exists, and after this CFGBuilderPass,
        // so predecessors are populated for the backward scan. Before RegionClone so
        // cloned regions carry the barrier too. Inserts a workgroup barrier between an
        // urgent and a deferrable tensor_load group. Off by default.
        if (moduleOptions.TDMLoadWaveSync) {
            pm.addPass(createTDMLoadWaveSyncPass());
        }

        pm.addPass(createRegionClonePass(moduleOptions.CloneList));
        mpm.addPass(createMainOnlyAdaptor(std::move(pm)));
    }

    // MSB is materialized for the entry function and every callable function
    // (each function owns its VGPR MSB hardware state).
    mpm.addPass(createFunctionToModuleAdaptor(createInsertVgprMsbPass()));

    // Rebuild the CFG on every function.
    mpm.addPass(createFunctionToModuleAdaptor(createCFGBuilderPass()));

    // Whole-kernel expert SCHED_MODE=2: wait-alu insertion + mode2 enable.
    if (moduleOptions.EnableESM2) {
        mpm.addPass(createInsertWaitAluModulePass(moduleOptions.EnableESM2TrackValuVsrc));
    }

    mpm.addPass(createFunctionToModuleAdaptor(createInsertCoexecHazardPass()));

    {
        PassManager pm = makeEntryPM(module, debugStreams);
        pm.addPass(createMemTokenConsistencyCheckPass());

        if (runScheduler) {
            pm.addPass(createInsertDelayAluPass(/*minWavesPerSimd=*/2));
            pm.addPass(createLoopRegionRemarkPass());
        }
        pm.addPass(createEstimateAsmCyclesPass());
        mpm.addPass(createMainOnlyAdaptor(std::move(pm)));
    }

    // Whole-kernel reuse on final instruction order (O0 and O1+; after scheduler + VGPR MSB).
    // Per function, each in isolation (reuse never chains across a call site or a function
    // boundary).
    mpm.addPass(createFunctionToModuleAdaptor(createSetMatrixReusePass()));

    // Run after the final CFG build but before flatten/SW-prefetch: this pass
    // covers final per-function code, while SW-prefetch owns its hints' XCnt waits.
    constexpr bool kEnableXcntDrainProfile = false;
    mpm.addPass(createGfx1250HazardModulePass(kEnableXcntDrainProfile));

    // Flatten callees + byte-layout tail (entry only, single linear stream).
    {
        PassManager pm = makeEntryPM(module, debugStreams);
        // Re-merge callable functions into the entry at their ASM placement markers so
        // SwInstructionPrefetchRelStaticPass sees a single linear stream / legacy emission
        // order. After the multi-function passes above; no-op with no callable functions.
        //
        // WARNING: temporary workaround; see FlattenCalleesPass. Remove once
        // SwInstructionPrefetchRelStaticPass handles multiple functions directly.
        pm.addPass(createFlattenCalleesPass(module.getFunctions()));
        // gfx1250 hardware-entrypoint prologue:
        // `global_prefetch_b8 v0, [s0, s1] scope:SCOPE_SE th:TH_LOAD_RT` + `v_nop`.
        // global_prefetch_b8 makes the first VMEM instruction non-clause-bound (it
        // is a VMEM op that ignores EXEC); v_nop is a safe first VALU instruction.
        // Runs after flatten (so the entry's first instruction is the kernel's
        // first) and before SW-prefetch insertion so the prefetch pass anchors
        // its byte layout on the final entry (prologue included) and its
        // CP-boundary coverage stays gap-free.
        pm.addPass(createInsertInitialUnclausedVmemPass());

        // SW instruction prefetch — abs and PC-rel are mutually exclusive.
        // Priority: abs (EnableSwInstructionPrefetchAbs) > PC-rel
        // (EnableSwInstructionPrefetchRelStatic).
        if (moduleOptions.EnableSwInstructionPrefetchAbs) {
            // One knob enables both abs passes; they are mutually exclusive by regime:
            //   - static  : entry-burst grid, emits for (32640, 65536]; no-ops for > 65536.
            //   - dynamic : run-time-targeted (post-CP) policy. Runs the read-only analysis dump
            //   for
            //     total > P(0)=32640; emits the predicated prefetch ladder (after
            //     label_MultiGemmEnd) for total > 65536. Dumps to
            //     <outputDir>/<kernel>/sw_prefetch_abs_dynamic_pass.txt.
            // Both use the module overload (reads SwInstructionPrefetchAbsBaseSgpr + debug path).
            // Dynamic runs FIRST so its analysis dump reflects the PRISTINE layout (before the
            // static pass's entry burst shifts offsets). At any given size exactly one pass emits,
            // so there is no co-mutation or baseSgpr contention.
            pm.addPass(createSwInstructionPrefetchAbsDynamicPass(module));
            pm.addPass(createSwInstructionPrefetchAbsStaticPass(module));
        } else if (moduleOptions.EnableSwInstructionPrefetchRelStatic) {
            // PC-rel dynamic pass (CFG-gated; replaces static PC-rel when enabled).
            // pm.addPass(createSwInstructionPrefetchRelStaticPass(module));
            pm.addPass(createSwInstructionPrefetchRelDynamicPass(module));
        }

        // When StinkyTofuCostOutputDir is set, dump pass debug (per-instruction + summary) to
        // <outputDir>/<kernel>/accumulate_instruction_size_pass_debug.txt (same layout as Backend).
        pm.addPass(createAccumulateInstructionSizePass(module));

        // Pass the whole-kernel function list so removal applies kernel-wide
        // (entry + callable functions).
        if (auto pass = createRemoveInstructionPass(moduleOptions.RemoveInstructions,
                                                    module.getFunctions())) {
            pm.addPass(std::move(pass));
        }
        mpm.addPass(createMainOnlyAdaptor(std::move(pm)));
    }
    return true;
}

struct Gfx1250Registrar {
    Gfx1250Registrar() {
        BackendRegistry::setArchPipeline(
            GFX1250_ARCH, {buildGfx1250Pipeline, {"loopWithPrefetch", "noLoadLoopBody"}});
    }
};
static Gfx1250Registrar s_gfx1250Registrar;
}  // namespace

void anchorGfx1250Backend() {}  // NOLINT(misc-use-internal-linkage)

}  // namespace stinkytofu
