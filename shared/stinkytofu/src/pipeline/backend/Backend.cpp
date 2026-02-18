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
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"

namespace stinkytofu
{
    struct Backend::Impl
    {
        // Pipelines for the member module's architecture only (run in order)
        std::vector<PipelineConfig> pipelines;
        std::vector<std::string>    groupNames;
    };

    Backend::Backend(StinkyAsmModule& module)
        : pImpl(std::make_unique<Impl>())
        , module(module)
    {
        // Load pipelines for member module's arch from BackendRegistry
        auto arch      = module.getArch();
        auto factories = BackendRegistry::getPipelineFactories(arch);
        if(!factories.empty())
        {
            pImpl->pipelines.reserve(factories.size());
            pImpl->groupNames.reserve(factories.size());
            for(const auto& factory : factories)
            {
                pImpl->pipelines.push_back(factory.builder(module));
                pImpl->groupNames.push_back(factory.groupName);
            }
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
        return !pImpl->pipelines.empty();
    }

    size_t Backend::getPipelineCount() const
    {
        return pImpl->pipelines.size();
    }

    void Backend::clearPipelines()
    {
        pImpl->pipelines.clear();
        pImpl->groupNames.clear();
    }

    void Backend::setPipelines(std::vector<PipelineConfig> configs)
    {
        pImpl->pipelines = std::move(configs);
        pImpl->groupNames.assign(configs.size(), "");
    }

    bool Backend::runOptimization()
    {
        return runPipelineSequence();
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
            // TODO: whole module optimization
            doOptimization = false;
        }
        else if(auto groupRange = module.findGroupRange(groupName))
        {
            auto [begin, end] = groupRange.value();

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
            for(auto bbIt = tempFunc.begin(); bbIt != tempFunc.end(); bbIt++)
            {
                for(auto it = bbIt->begin(); it != bbIt->end();)
                {
                    IRBase* ir = it.getNodePtr();
                    it++;
                    // FIXME: currently we insert IR to the range end, which
                    // assumes the Module has only one basic block.
                    origBB->insertIR(end, ir);
                }
            }
        }

        return doOptimization;
    }

    /* private methods */
    bool Backend::runPipelineSequence()
    {
        if(!pImpl->pipelines.empty())
        {
            for(size_t i = 0; i < pImpl->pipelines.size(); ++i)
            {
                if(!runOptimizationWithConfig(pImpl->pipelines[i], pImpl->groupNames[i]))
                {
                    return false;
                }
            }
            return true;
        }

        return true;
    }

} // namespace stinkytofu
