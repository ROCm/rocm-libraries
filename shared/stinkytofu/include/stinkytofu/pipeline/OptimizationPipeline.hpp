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

#include "stinkytofu/core/stinkytofu.hpp"
#include <memory>
#include <string>

namespace stinkytofu
{
    struct WaitCntConfig;

    /// Pipeline profile for different use cases
    enum class PipelineProfile
    {
        OptimizationOnly, ///< Only optimization passes (no scheduling/waitcnt)
        FullPipeline, ///< Complete pipeline with scheduling (for production kernels)
        NoOptimization, ///< Don't run any passes
        Custom ///< Custom configuration
    };

    /// Optimization level for standard pipelines
    enum class OptLevel
    {
        O0, ///< No optimization
        O1, ///< Fast compile: Peephole only
        O2, ///< Balanced: Peephole + DCE (default)
        O3, ///< Aggressive: All optimization passes
    };

    /// Configuration for optimization pipeline
    ///
    /// Provides fine-grained control over which passes to run.
    /// Can be constructed from a PipelineProfile or customized manually.
    ///
    /// Example profiles:
    /// - OptimizationOnly: General optimization (Peephole + DCE + DuplicateElim)
    /// - FullPipeline: Production kernels (CFG + Scheduling + Optimization + WaitCnt)
    ///
    /// Note: This struct contains unique_ptr members, so it's move-only (not copyable).
    struct PipelineConfig
    {
        // Delete copy constructor and copy assignment (contains unique_ptr)
        PipelineConfig(const PipelineConfig&)            = delete;
        PipelineConfig& operator=(const PipelineConfig&) = delete;

        // Default move constructor and move assignment
        PipelineConfig(PipelineConfig&&)            = default;
        PipelineConfig& operator=(PipelineConfig&&) = default;

        // Default constructor
        PipelineConfig()         = default;
        PipelineProfile profile  = PipelineProfile::OptimizationOnly;
        OptLevel        optLevel = OptLevel::O2;

        // ========== Optimization Passes ==========
        bool enablePeephole      = true; ///< Pattern matching and fusion
        bool enableDCE           = true; ///< Dead code elimination
        bool enableDuplicateElim = false; ///< Common subexpression elimination

        // ========== Scheduling Passes ==========
        bool enableCFGBuilder       = false; ///< Build control flow graph from labels
        bool enableDAGScheduler     = false; ///< Instruction scheduling
        bool enableScheduleLastLRs  = false; ///< Schedule last live ranges
        bool enableScheduleFirstLRs = false; ///< Schedule first live ranges

        // ========== WaitCnt Passes ==========
        bool enableWaitCnt = false; ///< Enable wait count insertion
        enum class WaitCntMode
        {
            Conservative, ///< Safe, conservative wait count placement
            Minimal, ///< Minimal wait counts (more aggressive)
            Unroll, ///< Unroll-aware wait count placement
            Custom ///< Use custom WaitCntConfig
        };
        WaitCntMode          waitCntMode   = WaitCntMode::Conservative;
        const WaitCntConfig* customWaitCnt = nullptr; ///< Custom waitcnt config (if mode=Custom)

        // ========== Iteration Control ==========
        int optimizationIterations = 3; ///< Iterations for optimization passes only

        // ========== Debugging ==========
        bool verbose             = false; ///< Print pass execution progress
        bool dumpIRBetweenPasses = false; ///< Dump IR after each pass

        // ========== GEMM-Specific Configuration ==========
        std::unique_ptr<GemmTileConfig> gemmTileConfig; ///< GEMM tile configuration (optional)

        // ========== Pass Feature Configuration ==========
        PassFeatureConfig passFeatureConfig; ///< Pass-specific optimization behaviors

        // ========== Basic Block Filtering ==========
        BasicBlockFilter basicBlockFilter = BasicBlockFilterBuilder::all(); ///< Filter for passes

        // ========== Debug Configuration ==========
        std::unique_ptr<PassManagerDebugConfig> debugConfig; ///< Debug output configuration

        // ========== Custom Pass Extension ==========
        /// Custom analysis passes executed BEFORE built-in pipeline (in order)
        /// These are registered with the AnalysisManager
        /// Mutable to allow pass execution from const config
        mutable std::vector<std::unique_ptr<AnalysisPass>> beforeAnalysisPasses;

        /// Custom transformation passes executed BEFORE built-in pipeline (in order)
        /// Order: beforePasses[0] -> beforePasses[1] -> ... -> built-in passes
        /// Mutable to allow pass execution from const config
        mutable std::vector<std::unique_ptr<Pass>> beforePasses;

        /// Custom transformation passes executed AFTER built-in pipeline (in order)
        /// Order: built-in passes -> afterPasses[0] -> afterPasses[1] -> ...
        /// Mutable to allow pass execution from const config
        mutable std::vector<std::unique_ptr<Pass>> afterPasses;

        /// Custom analysis passes executed AFTER built-in pipeline (in order)
        /// These are registered with the AnalysisManager
        /// Mutable to allow pass execution from const config
        mutable std::vector<std::unique_ptr<AnalysisPass>> afterAnalysisPasses;

        /// Add a custom analysis pass to run before the built-in pipeline
        PipelineConfig& addAnalysisPassBefore(std::unique_ptr<AnalysisPass> pass)
        {
            beforeAnalysisPasses.push_back(std::move(pass));
            return *this;
        }

        /// Add a custom transformation pass to run before the built-in pipeline
        PipelineConfig& addPassBefore(std::unique_ptr<Pass> pass)
        {
            beforePasses.push_back(std::move(pass));
            return *this;
        }

        /// Add a custom transformation pass to run after the built-in pipeline
        PipelineConfig& addPassAfter(std::unique_ptr<Pass> pass)
        {
            afterPasses.push_back(std::move(pass));
            return *this;
        }

        /// Add a custom analysis pass to run after the built-in pipeline
        PipelineConfig& addAnalysisPassAfter(std::unique_ptr<AnalysisPass> pass)
        {
            afterAnalysisPasses.push_back(std::move(pass));
            return *this;
        }

        /// Configure GEMM-specific tile parameters
        PipelineConfig& withGemmTileConfig(std::array<int, 3> arch,
                                           uint32_t           TileA0,
                                           uint32_t           TileB0,
                                           uint32_t           TileM0,
                                           uint32_t           NumGRA,
                                           uint32_t           NumGRB,
                                           uint32_t           NumGRM,
                                           uint32_t           NumWaves)
        {
            gemmTileConfig           = std::make_unique<GemmTileConfig>();
            gemmTileConfig->arch     = arch;
            gemmTileConfig->TileA0   = TileA0;
            gemmTileConfig->TileB0   = TileB0;
            gemmTileConfig->TileM0   = TileM0;
            gemmTileConfig->NumGRA   = NumGRA;
            gemmTileConfig->NumGRB   = NumGRB;
            gemmTileConfig->NumGRM   = NumGRM;
            gemmTileConfig->NumWaves = NumWaves;
            return *this;
        }

        /// Configure barrier unrolling semantics (code structure property)
        PipelineConfig& withBarrierSemantics(bool unrollMovableBarrier)
        {
            passFeatureConfig.barrierConfig.unrollMovableBarrier = unrollMovableBarrier;
            return *this;
        }

        /// Configure loop unrolling properties (code structure property)
        PipelineConfig& withLoopUnroll(bool unrollGemm)
        {
            passFeatureConfig.loopConfig.unrollGemm = unrollGemm;
            return *this;
        }

        /// Configure DAG scheduler feature switches (optional optimizations)
        PipelineConfig& withDagFeatures(bool distributeGlobalRead)
        {
            passFeatureConfig.dagFeatures.distributeGlobalRead = distributeGlobalRead;
            return *this;
        }

        /// Configure debug output
        PipelineConfig& withDebugConfig(std::unique_ptr<PassManagerDebugConfig> cfg)
        {
            debugConfig = std::move(cfg);
            return *this;
        }

        /// Create configuration from a pipeline profile
        /// @param profile The pipeline profile (OptimizationOnly or FullPipeline)
        /// @param optLevel Optimization level for optimization passes
        static PipelineConfig fromProfile(PipelineProfile profile,
                                          OptLevel        optLevel = OptLevel::O2);

        /// Quick builder for production kernels
        static PipelineConfig forProductionKernel(OptLevel level = OptLevel::O2);
    };

    /// Unified pipeline builder for all StinkyTofu scenarios
    ///
    /// Supports two main profiles:
    /// 1. **OptimizationOnly**: Just optimization passes (Peephole + DCE + DuplicateElim)
    /// 2. **FullPipeline**: Complete pipeline (CFG + Scheduling + Optimization + WaitCnt)
    ///
    /// Example usage:
    /// ```cpp
    /// // Simple optimization only
    /// Function func("kernel");
    /// PipelineConfig config = PipelineConfig::fromProfile(PipelineProfile::OptimizationOnly);
    /// OptimizationPipeline::run(func, config);
    ///
    /// // Production kernel (full pipeline)
    /// config = PipelineConfig::forProductionKernel(OptLevel::O3);
    /// OptimizationPipeline::run(func, config);
    ///
    /// // Custom fine-grained control
    /// config = PipelineConfig::fromProfile(PipelineProfile::Custom);
    /// config.enablePeephole = true;
    /// config.enableDCE = true;
    /// config.enableDAGScheduler = true;
    /// OptimizationPipeline::run(func, config);
    /// ```
    class OptimizationPipeline
    {
    public:
        /// Run pipeline with PassManager's internal Function (for use with custom passes)
        ///
        /// This version creates a PassManager with an empty Function. Custom passes
        /// added via config (e.g., RocisaToStinkyAsmPass) populate and consume the Function.
        ///
        /// @param config Pipeline configuration (includes GEMM config, pass features, etc.)
        /// @param bbFilter Basic block filter to apply
        static void run(const PipelineConfig&        config,
                        stinkytofu::BasicBlockFilter bbFilter
                        = stinkytofu::BasicBlockFilterBuilder::all());

        /// Run pipeline on an external Function (for standalone optimization)
        ///
        /// This version transfers an external Function into PassManager, runs passes,
        /// and transfers the results back. Used when you have an existing Function to optimize.
        ///
        /// @param func Function to process (modified in-place)
        /// @param config Pipeline configuration (includes GEMM config, pass features, etc.)
        static void run(Function& func, const PipelineConfig& config);

        /// Convenience: run full production pipeline with default O2
        static void runFullPipeline(Function& func)
        {
            run(func, PipelineConfig::forProductionKernel());
        }

    private:
        /// Shared implementation for both run() overloads
        static void runPipelineInternal(PassManager& passManager, const PipelineConfig& config);

        /// Run optimization passes iteratively using PassManager
        static void runOptimizationPasses(PassManager& passManager, const PipelineConfig& config);
    };

} // namespace stinkytofu
