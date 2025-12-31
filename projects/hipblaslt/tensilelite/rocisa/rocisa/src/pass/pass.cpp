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
#include "pass.hpp"
#include "ir/asm/OptimizationPipeline.hpp"
#include "stinkytofu.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

namespace rocisa
{
    rocIsaPassResult rocIsaPass(std::shared_ptr<KernelBody>& kernel, const rocIsaPassOption& option)
    {
        rocIsaPassResult result;
        if(option.removeDupFunc)
        {
            removeDuplicatedFunction(kernel->body);
        }

        const char* stinkytofuDumpEnv = std::getenv("STINKYTOFU_DUMP");
        bool        stinkytofuDump    = stinkytofuDumpEnv ? true : false;

        bool stinkytofuEnable = true;
        // TODO: Remove this to enable after everything is stable
        if(std::getenv("STINKYTOFU_ENABLE") == nullptr)
        {
            stinkytofuEnable = false; // Disable for now
        }

        compositeToInstruction(kernel->body);
        // Convert text variables to registers
        convertTextVariablesToRegisters(kernel->body);

        auto kernelInfo = rocIsa::getInstance().getKernel();
        if(stinkytofuEnable && option.stinkyOpt)
        {
            for(auto& item : kernel->body->items())
            {
                if(auto module = std::dynamic_pointer_cast<Module>(item))
                {
                    auto stinkySchedule = [&kernelInfo,
                                           &option](std::shared_ptr<Module> module,
                                                    const std::string&      pathPrefix,
                                                    bool                    withPGR        = false,
                                                    bool                    stinkytofuDump = false,
                                                    stinkytofu::BasicBlockFilter bbFilter
                                                    = stinkytofu::BasicBlockFilterBuilder::all()) {
                        // Create PassContext (automatically creates an empty Function)
                        stinkytofu::PassContext ctx;

                        // Configure kernel info in PassContext
                        // Note: WavefrontSize is automatically computed from arch by setGemmTileConfig
                        stinkytofu::GemmTileConfig gemmConfig;
                        gemmConfig.arch     = kernelInfo.isaVersion;
                        gemmConfig.TileA0   = option.TileA0;
                        gemmConfig.TileB0   = option.TileB0;
                        gemmConfig.TileM0   = option.TileM0;
                        gemmConfig.NumGRA   = option.NumGRA;
                        gemmConfig.NumGRB   = option.NumGRB;
                        gemmConfig.NumGRM   = option.NumGRM;
                        gemmConfig.NumWaves = option.numWaves;
                        ctx.setGemmTileConfig(gemmConfig);

                        // Configure basic block filter
                        ctx.setBasicBlockFilter(bbFilter);

                        // Create pipeline configuration with full scheduling and optimization
                        auto config = stinkytofu::PipelineConfig::fromProfile(
                            stinkytofu::PipelineProfile::FullPipeline, stinkytofu::OptLevel::O3);

                        // Configure GEMM-specific tile parameters
                        config.withGemmTileConfig(kernelInfo.isaVersion,
                                                  option.TileA0,
                                                  option.TileB0,
                                                  option.TileM0,
                                                  option.NumGRA,
                                                  option.NumGRB,
                                                  option.NumGRM,
                                                  option.numWaves);

                        // Configure pass features (GEMM-specific optimizations)
                        config
                            .withBarrierSemantics(true) // unrollGemmMovableBarrier
                            .withLoopUnroll(true) // unrollGemm
                            .withDagFeatures(true); // distributeGlobalRead

                        // Configure basic block filter
                        config.basicBlockFilter = bbFilter;

                        // Configure debug output if enabled
                        if(stinkytofuDump)
                        {
                            auto debugConfig
                                = std::make_unique<stinkytofu::PassManagerDebugConfig>();
                            debugConfig->setPrintBeforeAll(true);
                            debugConfig->setPrintAfterAll(true);
                            debugConfig->setDumpToFileInBefore(pathPrefix + "before.txt");
                            debugConfig->setDumpToFileInAfter(pathPrefix + "after.txt");
                            stinkytofu::PassManagerDebugConfig::addDebugOnly(
                                "StinkyDAGSchedulerPass");
                            stinkytofu::PassManagerDebugConfig::addDebugOnly(
                                "StinkyAsmToRocisaPass");
                            config.withDebugConfig(std::move(debugConfig));
                        }

                        // Add Rocisa analysis passes (need access to Module)
                        config
                            .addAnalysisPassBefore(
                                stinkytofu::createRocisaDFSFlatItemsPass(*module.get()))
                            .addAnalysisPassBefore(stinkytofu::createRocisaStinkyMappingPass());

                        // Add Rocisa-to-StinkyAsm conversion pass before the built-in pipeline
                        config.addPassBefore(
                            stinkytofu::createRocisaToStinkyAsmPass(true /*ignore waitCnt*/));

                        // Add StinkyAsm-to-Rocisa conversion pass after the built-in pipeline
                        config.addPassAfter(stinkytofu::createStinkyAsmToRocisaPass());

                        // Built-in pipeline includes (in order):
                        // Phase 1: CFGBuilderPass
                        // Phase 2: StinkyDAGSchedulerPass, ScheduleLastLRsPass
                        // Phase 3: Optimization passes (Peephole, DCE, DistributeEdge)
                        // Phase 4: StinkyUnrollWaitCntPass

                        // Run the pipeline with the Function from PassContext
                        stinkytofu::OptimizationPipeline::run(ctx.getFunction(), config, ctx);
                    };
                    // Convert the module to stinkytofu instructions
                    if(module->name == "loopWithPrefetch")
                    {
                        stinkytofu::BasicBlockFilter bbFilter
                            = stinkytofu::BasicBlockFilterBuilder::byLabelPrefix("label_LoopBegin");
                        stinkySchedule(module, "loopBody-", true, stinkytofuDump, bbFilter);
                    }
                    else if(module->name == "noLoadLoop")
                    {
                        for(auto& subItem : module->items())
                        {
                            if(auto subModule = std::dynamic_pointer_cast<Module>(subItem))
                            {
                                if(subModule->name == "noLoadLoopBody")
                                {
                                    stinkySchedule(subModule, "noLoadLoop-", false, stinkytofuDump);
                                }
                            }
                        }
                    }
                }
            }
        }

        if(option.doOpt())
        {
            auto maxVgpr = kernel->totalVgprs;
            auto maxSgpr = kernel->totalSgprs;
            auto graph   = buildGraph(kernel->body, maxVgpr, maxSgpr);
            if(option.removeDupAssign)
            {
                removeDuplicateAssignment(graph);
            }
        }

        if(option.insertDelayAlu)
        {
            insertDelayAlu(kernel->body);
        }

        if(option.getCycles)
            result.cycles = getCycles(kernel->body, option.numWaves);

        return std::move(result);
    }
} // namespace rocisa

void init_pass(nb::module_ m)
{
    auto m_pass = m.def_submodule("asmpass", "rocIsa pass submodule.");
    m_pass.def("getActFuncModuleName", &rocisa::getActFuncModuleName, "getActFuncModuleName.");
    m_pass.def("getActFuncBranchModuleName",
               &rocisa::getActFuncBranchModuleName,
               "getActFuncBranchModuleName.");
    m_pass.def("rocIsaPass", &rocisa::rocIsaPass, "rocIsaPass.");

    nb::class_<rocisa::rocIsaPassOption>(m_pass, "rocIsaPassOption")
        .def(nb::init<>())
        .def_rw("stinkyOpt", &rocisa::rocIsaPassOption::stinkyOpt)
        .def_rw("hardwareConfigPath", &rocisa::rocIsaPassOption::hardwareConfigPath)
        .def_rw("insertDelayAlu", &rocisa::rocIsaPassOption::insertDelayAlu)
        .def_rw("removeDupFunc", &rocisa::rocIsaPassOption::removeDupFunc)
        .def_rw("removeDupAssign", &rocisa::rocIsaPassOption::removeDupAssign)
        .def_rw("getCycles", &rocisa::rocIsaPassOption::getCycles)
        .def_rw("numWaves", &rocisa::rocIsaPassOption::numWaves)
        .def_rw("ta0", &rocisa::rocIsaPassOption::TileA0)
        .def_rw("tb0", &rocisa::rocIsaPassOption::TileB0)
        .def_rw("tm0", &rocisa::rocIsaPassOption::TileM0)
        .def_rw("nGRA", &rocisa::rocIsaPassOption::NumGRA)
        .def_rw("nGRB", &rocisa::rocIsaPassOption::NumGRB)
        .def_rw("nGRM", &rocisa::rocIsaPassOption::NumGRM)
        .def_rw("wavefrontSz", &rocisa::rocIsaPassOption::WavefrontSize);

    nb::class_<rocisa::rocIsaPassResult>(m_pass, "rocIsaPassResult")
        .def(nb::init<>())
        .def_ro("cycles", &rocisa::rocIsaPassResult::cycles);
}
