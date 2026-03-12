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
#include "stinkytofu/pipeline/Backend.hpp"

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

namespace stinkytofu
{
    struct Backend::Impl
    {
        // Pipeline specs for the member module's architecture (run in order)
        std::vector<BackendRegistry::PipelineSpec> pipelineSpecs;

        // optimized range begin and end
        IntrusiveListIterator<IRBase> optimizedRangeBegin;
        IntrusiveListIterator<IRBase> optimizedRangeEnd;
    };

    Backend::Backend(StinkyAsmModule& module)
        : pImpl(std::make_unique<Impl>())
        , module(module)
    {
        auto populator = BackendRegistry::getArchPopulator(module.getArch());
        if(populator)
        {
            populator(module, pImpl->pipelineSpecs);
        }
    }

    Backend::~Backend() = default;

    Backend::Backend(Backend&&) noexcept = default;

    std::array<int, 3> Backend::getArch() const
    {
        return module.getArch();
    }

    bool Backend::hasPipelines() const
    {
        return !pImpl->pipelineSpecs.empty();
    }

    size_t Backend::getPipelineCount() const
    {
        return pImpl->pipelineSpecs.size();
    }

    void Backend::clearPipelines()
    {
        pImpl->pipelineSpecs.clear();
    }

    void Backend::setPipelines(std::vector<PipelineConfig> configs)
    {
        pImpl->pipelineSpecs.clear();
        pImpl->pipelineSpecs.reserve(configs.size());
        for(auto& c : configs)
        {
            pImpl->pipelineSpecs.push_back({std::move(c), ""});
        }
    }

    const std::vector<BackendRegistry::PipelineSpec>& Backend::getPipelines() const
    {
        return pImpl->pipelineSpecs;
    }

    bool Backend::runOptimization()
    {
        bool success = runPipelineSequence();

        // if range is not empty, reinsert waitcnts in the range
        if(pImpl->optimizedRangeBegin != IntrusiveListIterator<IRBase>()
           && pImpl->optimizedRangeEnd != IntrusiveListIterator<IRBase>())
        {
            success = reinsertWaitCntsInOptimizedRange();
        }

        return success;
    }

    bool Backend::runOptimizationWithConfig(const PipelineConfig& config,
                                            const std::string&    groupName)
    {
        if(config.profile == PipelineProfile::NoOptimization)
        {
            return false;
        }

        // Create a temporary Function to hold the IR
        Function    tempFunc("temp");
        BasicBlock* bb = tempFunc.createBasicBlock("entry");

        bool doOptimization = true;
        if(groupName.empty())
        {
            // Run the optimization pipeline for the whole module
            OptimizationPipeline::run(module.getFunction(), config);
        }
        else if(auto groupRange = module.findGroupRange(groupName))
        {
            auto [begin, end]  = groupRange.value();
            BasicBlock* origBB = begin->getParent();
            // Move StinkyInstructions from module to temporary function
            for(auto it = begin; it != end;)
            {
                IRBase* ir = it.getNodePtr();
                it++;
                if(dyn_cast<StinkyInstruction>(ir))
                {
                    bb->appendIR(ir);
                }
                else
                {
                    // erase non-StinkyInstruction IRs
                    ir->erase();
                }
            }

            // Run the optimization pipeline
            OptimizationPipeline::run(tempFunc, config);

            // Move IR from temporary function back to module
            assert(module.getFunction().size() == 1
                   && "Current module should have only one basic block.");

            IRBase* firstInserted = nullptr;
            IRBase* lastInserted  = nullptr;
            for(auto bbIt = tempFunc.begin(); bbIt != tempFunc.end(); bbIt++)
            {
                for(auto it = bbIt->begin(); it != bbIt->end();)
                {
                    IRBase* ir = it.getNodePtr();
                    it++;
                    // FIXME: currently we insert IR to the range end, which
                    // assumes the Module has only one basic block.
                    origBB->insertIR(end, ir);
                    if(!firstInserted)
                        firstInserted = ir;
                    lastInserted = ir;
                }
            }

            assert(firstInserted && lastInserted && "No IR inserted for group");

            module.setGroupRange(groupName,
                                 IntrusiveListIterator<IRBase>(firstInserted),
                                 IntrusiveListIterator<IRBase>(lastInserted));

            if(pImpl->optimizedRangeBegin == IntrusiveListIterator<IRBase>())
            {
                pImpl->optimizedRangeBegin = IntrusiveListIterator<IRBase>(firstInserted);
            }
            pImpl->optimizedRangeEnd = IntrusiveListIterator<IRBase>(lastInserted) + 1;
        }

        return doOptimization;
    }

    /* private methods */
    bool Backend::runPipelineSequence()
    {
        for(const auto& spec : pImpl->pipelineSpecs)
        {
            if(!runOptimizationWithConfig(spec.config, spec.groupName))
            {
                return false;
            }
        }
        return true;
    }

    bool Backend::reinsertWaitCntsInOptimizedRange()
    {
        BasicBlock* origBB    = pImpl->optimizedRangeBegin->getParent();
        auto        insertEnd = pImpl->optimizedRangeEnd;

        // Create a temporary Function to hold the IR
        Function    tempFunc("temp");
        BasicBlock* bb = tempFunc.createBasicBlock("entry");
        for(auto it = pImpl->optimizedRangeBegin; it != pImpl->optimizedRangeEnd;)
        {
            IRBase* ir = it.getNodePtr();
            it++;
            if(dyn_cast<StinkyInstruction>(ir))
            {
                bb->appendIR(ir);
            }
            else
            {
                ir->erase();
            }
        }

        PassManager    passManager;
        GemmTileConfig gemmTileConfig;
        gemmTileConfig.arch = module.getArch();
        passManager.setGemmTileConfig(gemmTileConfig);
        passManager.addPass(createCFGBuilderPass());
        passManager.addPass(createStinkyRemoveWaitCntPass());
        passManager.addPass(createStinkyWaitCntInsertionPass());
        passManager.run(tempFunc);

        // Move IR from temporary function back to module
        for(auto bbIt = tempFunc.begin(); bbIt != tempFunc.end(); bbIt++)
        {
            for(auto it = bbIt->begin(); it != bbIt->end();)
            {
                IRBase* ir = it.getNodePtr();
                it++;
                origBB->insertIR(insertEnd, ir);
            }
        }

        return true;
    }

} // namespace stinkytofu
