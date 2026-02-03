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
#include "ir/asm/OptimizationPipeline.hpp"
#include "ErrorHandling.hpp"
#include "ir/asm/AsmVerifierPass.hpp"
#include "ir/asm/DeadCodeEliminationPass.hpp"
#include "ir/asm/DefUseChain.hpp"
#include "ir/asm/PeepholeOptimizationPass.hpp"
#include "ir/asm/RedundantMovEliminationPass.hpp"

#include <iostream>

namespace stinkytofu
{
    PipelineConfig PipelineConfig::fromProfile(PipelineProfile profile, OptLevel optLevel)
    {
        PipelineConfig config;
        config.profile  = profile;
        config.optLevel = optLevel;

        switch(profile)
        {
        case PipelineProfile::OptimizationOnly:
            // General optimization passes (no scheduling or waitcnt)
            // Can be used for any scenario that only needs optimization
            config.enablePeephole      = (optLevel >= OptLevel::O1);
            config.enableDCE           = (optLevel >= OptLevel::O2);
            config.enableDuplicateElim = (optLevel >= OptLevel::O3);

            // No scheduling or waitcnt
            config.enableCFGBuilder       = false;
            config.enableDAGScheduler     = false;
            config.enableScheduleLastLRs  = false;
            config.enableScheduleFirstLRs = false;
            config.enableWaitCnt          = false;

            // Iterations based on opt level
            config.optimizationIterations = (optLevel == OptLevel::O0)   ? 1
                                            : (optLevel == OptLevel::O1) ? 1
                                            : (optLevel == OptLevel::O2) ? 3
                                                                         : 5;
            break;

        case PipelineProfile::FullPipeline:
            // Full production pipeline
            config.enablePeephole      = (optLevel >= OptLevel::O1);
            config.enableDCE           = (optLevel >= OptLevel::O2);
            config.enableDuplicateElim = (optLevel >= OptLevel::O3);

            // Enable scheduling passes
            config.enableCFGBuilder       = true;
            config.enableDAGScheduler     = true;
            config.enableScheduleLastLRs  = true;
            config.enableScheduleFirstLRs = false;

            // Enable waitcnt
            config.enableWaitCnt = true;
            config.waitCntMode   = WaitCntMode::Unroll;

            // Fewer iterations for full pipeline (slower)
            config.optimizationIterations = (optLevel >= OptLevel::O3) ? 3 : 1;
            break;

        case PipelineProfile::NoOptimization:
            // No optimization passes
            config.enablePeephole         = false;
            config.enableDCE              = false;
            config.enableDuplicateElim    = false;
            config.enableCFGBuilder       = false;
            config.enableDAGScheduler     = false;
            config.enableScheduleLastLRs  = false;
            config.enableScheduleFirstLRs = false;
            config.enableWaitCnt          = false;
            config.optimizationIterations = 0;
            break;

        case PipelineProfile::Custom:
            // Leave everything as default, user will customize
            break;
        }

        return config;
    }

    PipelineConfig PipelineConfig::forProductionKernel(OptLevel level)
    {
        return fromProfile(PipelineProfile::FullPipeline, level);
    }

    void OptimizationPipeline::runOptimizationPasses(PassManager&          passManager,
                                                     const PipelineConfig& config)
    {
        if(config.verbose)
        {
            std::cout << "Running optimization passes (O" << static_cast<int>(config.optLevel)
                      << ", " << config.optimizationIterations << " iterations)" << std::endl;
        }

        for(int iter = 0; iter < config.optimizationIterations; iter++)
        {
            if(config.verbose && config.optimizationIterations > 1)
            {
                std::cout << "  Iteration " << (iter + 1) << "/" << config.optimizationIterations
                          << std::endl;
            }

            // Run passes in standard order
            if(config.enablePeephole)
            {
                if(config.verbose)
                    std::cout << "    - PeepholeOptimization" << std::endl;
                passManager.addPass(createPeepholeOptimizationPass());
            }

            // RedundantMovElimination - Simple block-local duplicate removal for mov-type instructions
            if(config.enableDuplicateElim)
            {
                if(config.verbose)
                    std::cout << "    - RedundantMovElimination" << std::endl;
                passManager.addPass(createRedundantMovEliminationPass());
            }

            if(config.enableDCE)
            {
                if(config.verbose)
                    std::cout << "    - DeadCodeElimination" << std::endl;
                passManager.addPass(createDeadCodeEliminationPass());
            }
        }
    }

    void OptimizationPipeline::run(Function& func, const PipelineConfig& config)
    {
        // Create PassManager with its own internal PassContext
        PassManager passManager;

        // Transfer Function to PassManager's internal PassContext
        passManager.setFunction(func);

        // Run the pipeline
        runPipelineInternal(passManager, config);

        // Transfer results back
        Function& internalFunc = passManager.getPassContext().getFunction();
        func.deleteAllBasicBlocks();

        while(!internalFunc.getBasicBlocks().empty())
        {
            BasicBlock* bb = &internalFunc.getBasicBlocks().front();
            internalFunc.getBasicBlocks().remove(bb);
            bb->setParent(&func);
            func.getBasicBlocks().push_back(bb);
        }

        if(internalFunc.getEntryBlock())
        {
            func.setEntryBlock(internalFunc.getEntryBlock());
        }
    }

    void OptimizationPipeline::run(const PipelineConfig&        config,
                                   stinkytofu::BasicBlockFilter bbFilter)
    {
        // Create PassManager with its own internal PassContext (empty Function)
        PassManager passManager;

        // Run the pipeline (custom passes will populate the Function)
        runPipelineInternal(passManager, config);
    }

    void OptimizationPipeline::runPipelineInternal(PassManager&          passManager,
                                                   const PipelineConfig& config)
    {
        // This is the shared implementation used by both run() overloads

        // ========== Apply Configuration to PassManager ==========
        // Apply GEMM tile configuration if provided, or set default arch
        // Note: WavefrontSize is automatically computed from arch by setGemmTileConfig
        if(config.gemmTileConfig)
        {
            passManager.setGemmTileConfig(*config.gemmTileConfig);
        }
        else
        {
            STINKY_UNREACHABLE("GEMM tile configuration not provided");
        }

        // Apply pass feature configuration directly (already in correct format)
        passManager.setPassFeatureConfig(config.passFeatureConfig);

        // Apply basic block filter
        passManager.setBasicBlockFilter(config.basicBlockFilter);

        // Apply debug configuration if provided
        // Note: config.debugConfig is mutable even though config is const
        if(config.debugConfig)
        {
            passManager.setDebugConfig(std::move(const_cast<PipelineConfig&>(config).debugConfig));
        }

        if(config.verbose)
        {
            std::cout << "\n========== OptimizationPipeline ==========\n";
            std::cout << "Profile: "
                      << (config.profile == PipelineProfile::OptimizationOnly ? "OptimizationOnly"
                          : config.profile == PipelineProfile::FullPipeline   ? "FullPipeline"
                                                                              : "Custom")
                      << std::endl;
        }

        // ========== IR Verification (Built-in) ==========
        // Verify IR integrity before running any passes
        // This catches IR corruption early before it propagates through optimization
        if(config.verbose)
            std::cout << "\n  - Verifying IR integrity before optimization..." << std::endl;

        passManager.addPass(createStinkyIRVerifierPass());

        // ========== Phase 0: Custom Passes (Before) ==========
        if(!config.beforeAnalysisPasses.empty() || !config.beforePasses.empty())
        {
            if(config.verbose)
                std::cout << "\nPhase 0: Custom Passes (Before Built-in Pipeline)" << std::endl;

            // Register analysis passes first
            for(size_t i = 0; i < config.beforeAnalysisPasses.size(); ++i)
            {
                if(config.verbose)
                    std::cout << "  - [Analysis] " << config.beforeAnalysisPasses[i]->getName()
                              << std::endl;
                passManager.registerAnalysisPass(std::move(config.beforeAnalysisPasses[i]));
            }
            config.beforeAnalysisPasses.clear();

            // Then add transformation passes
            for(size_t i = 0; i < config.beforePasses.size(); ++i)
            {
                if(config.verbose)
                    std::cout << "  - [Transform] " << config.beforePasses[i]->getName()
                              << std::endl;
                passManager.addPass(std::move(config.beforePasses[i]));
            }
            config.beforePasses.clear();
        }

        // ========== Phase 1: CFG Building ==========
        // Build CFG first so optimizations can use block structure and liveness info
        if(config.enableCFGBuilder)
        {
            if(config.verbose)
                std::cout << "\nPhase 1: Building Control Flow Graph" << std::endl;
            passManager.addPass(createCFGBuilderPass());
        }

        // ========== Phase 2: Optimization ==========
        // Run optimizations AFTER CFG (for liveness analysis) but BEFORE scheduling
        // This ensures scheduling works on the final optimized instruction count
        if(config.enablePeephole || config.enableDCE || config.enableDuplicateElim)
        {
            if(config.verbose)
                std::cout << "\nPhase 2: Optimization (Pre-Scheduling)" << std::endl;
            runOptimizationPasses(passManager, config);
        }

        // ========== Phase 3: Instruction Scheduling ==========
        if(config.enableDAGScheduler || config.enableScheduleFirstLRs)
        {
            if(config.verbose)
                std::cout << "\nPhase 3: Instruction Scheduling" << std::endl;

            if(config.enableScheduleFirstLRs)
            {
                if(config.verbose)
                    std::cout << "  - ScheduleFirstLRs" << std::endl;
                passManager.addPass(createScheduleFirstLRsPass());
            }

            if(config.enableDAGScheduler)
            {
                if(config.verbose)
                    std::cout << "  - DAGScheduler" << std::endl;
                passManager.addPass(createStinkyDAGSchedulerPass());
            }

            if(config.enableScheduleLastLRs)
            {
                if(config.verbose)
                    std::cout << "  - ScheduleLastLRs" << std::endl;
                passManager.addPass(createScheduleLastLRsPass());
            }
        }

        // ========== Phase 4: WaitCnt Insertion ==========
        if(config.enableWaitCnt)
        {
            if(config.verbose)
                std::cout << "\nPhase 4: WaitCnt Insertion ("
                          << (config.waitCntMode == PipelineConfig::WaitCntMode::Conservative
                                  ? "Conservative"
                              : config.waitCntMode == PipelineConfig::WaitCntMode::Minimal
                                  ? "Minimal"
                              : config.waitCntMode == PipelineConfig::WaitCntMode::Unroll
                                  ? "Unroll"
                                  : "Custom")
                          << ")" << std::endl;

            std::unique_ptr<Pass> waitCntPass;
            switch(config.waitCntMode)
            {
            case PipelineConfig::WaitCntMode::Conservative:
                waitCntPass = createStinkyConservativeWaitCntPass();
                break;
            case PipelineConfig::WaitCntMode::Minimal:
                waitCntPass = createStinkyMinimalWaitCntPass();
                break;
            case PipelineConfig::WaitCntMode::Unroll:
                waitCntPass = createStinkyUnrollWaitCntPass();
                break;
            case PipelineConfig::WaitCntMode::Custom:
                if(config.customWaitCnt)
                    waitCntPass = createStinkyCustomWaitCntPass(*config.customWaitCnt);
                else
                    waitCntPass = createStinkyConservativeWaitCntPass(); // Fallback
                break;
            }

            if(waitCntPass)
            {
                passManager.addPass(std::move(waitCntPass));
            }
        }

        // ========== Phase 5: Custom Passes (After) ==========
        if(!config.afterPasses.empty() || !config.afterAnalysisPasses.empty())
        {
            if(config.verbose)
                std::cout << "\nPhase 5: Custom Passes (After Built-in Pipeline)" << std::endl;

            // Register analysis passes first
            for(size_t i = 0; i < config.afterAnalysisPasses.size(); ++i)
            {
                if(config.verbose)
                    std::cout << "  - [Analysis] " << config.afterAnalysisPasses[i]->getName()
                              << std::endl;
                passManager.registerAnalysisPass(std::move(config.afterAnalysisPasses[i]));
            }
            config.afterAnalysisPasses.clear();

            // Then add transformation passes
            for(size_t i = 0; i < config.afterPasses.size(); ++i)
            {
                if(config.verbose)
                    std::cout << "  - [Transform] " << config.afterPasses[i]->getName()
                              << std::endl;
                passManager.addPass(std::move(config.afterPasses[i]));
            }
            config.afterPasses.clear();
        }

        // ========== Run All Passes ==========
        passManager.run();

        if(config.verbose)
        {
            std::cout << "\n========== Pipeline Complete ==========\n" << std::endl;
        }
    }

} // namespace stinkytofu
