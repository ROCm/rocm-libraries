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

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

namespace stinkytofu
{
    // -----------------------------------------------------------------------
    // Debug output utility
    // -----------------------------------------------------------------------

    /// Configure debug output (DebugLevel, PrintBeforePass, PrintAfterPass)
    /// on a PassManager using the given label as the file prefix.
    /// DebugPass is global and does not need per-PM setup.
    inline void configureDebugOutput(PassManager&                          pm,
                                     const StinkyAsmModule::ModuleOptions& opts,
                                     const std::string&                    label)
    {
        auto forEachName = [](const std::string& csv, auto cb) {
            std::istringstream stream(csv);
            std::string        name;
            while(std::getline(stream, name, ','))
            {
                auto s = name.find_first_not_of(" ");
                auto e = name.find_last_not_of(" ");
                if(s != std::string::npos)
                    cb(name.substr(s, e - s + 1));
            }
        };

        bool needsDebugConfig = (opts.DebugLevel >= 1) || !opts.PrintBeforePass.empty()
                                || !opts.PrintAfterPass.empty();
        if(!needsDebugConfig)
            return;

        auto debugConfig = std::make_unique<PassManagerDebugConfig>();

        if(opts.DebugLevel == 1)
        {
            debugConfig->setPrintPassNames(true);
        }

        if(opts.DebugLevel == 2)
        {
            debugConfig->setDumpInitialIR(true);
            debugConfig->setPrintAfterAll(true);
            debugConfig->setDumpToFileInBefore(label + "-before_passes.txt");
            debugConfig->setDumpToFileInAfter(label + "-after_passes.txt");
        }
        if(!opts.PrintBeforePass.empty())
        {
            debugConfig->setDumpToFileInBefore(label + "-before_passes.txt");
            forEachName(opts.PrintBeforePass,
                        [&](const std::string& n) { debugConfig->addOnlyPrintBefore(n); });
        }
        if(!opts.PrintAfterPass.empty())
        {
            debugConfig->setDumpToFileInAfter(label + "-after_passes.txt");
            forEachName(opts.PrintAfterPass,
                        [&](const std::string& n) { debugConfig->addOnlyPrintAfter(n); });
        }

        pm.setDebugConfig(std::move(debugConfig));

        if(!opts.DebugPass.empty())
        {
            forEachName(opts.DebugPass,
                        [](const std::string& n) { PassManagerDebugConfig::addDebugOnly(n); });
        }
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /// Derive the debug label from group names.
    inline std::string makeDebugLabel(const std::vector<std::string>& groupNames)
    {
        if(groupNames.empty())
            return "wholeKernel";
        std::string label = groupNames[0];
        for(size_t i = 1; i < groupNames.size(); i++)
        {
            label += "+" + groupNames[i];
        }
        return label;
    }

    // -----------------------------------------------------------------------
    // ScopeAdaptor
    // -----------------------------------------------------------------------

    /// ScopeAdaptor is an LLVM-style adapter pass that scopes an inner PassManager
    /// to a specific region of a StinkyAsmModule.
    ///
    /// A kernel module may contain named instruction groups that define the scope
    /// of optimization. ScopeAdaptor encapsulates the extract-run-splice pattern:
    /// it extracts instructions from named groups into a temporary Function, runs
    /// an inner PassManager on it, and splices the results back.
    ///
    /// ## Modes
    ///
    ///   - single-region: 1 group name — extracts that group's instruction range,
    ///     runs PM, splices back.
    ///   - multi-region: 2+ group names — extracts the union of all groups' ranges,
    ///     runs PM, splices back.
    ///   - whole-kernel: empty groupNames — extracts all instructions to a temp
    ///     Function, runs inner PM, splices back.
    ///
    /// ## Limitations and assumptions
    ///
    /// The module maintains a single flat BasicBlock between adapter invocations.
    /// Each adapter restores this invariant after splice-back so that subsequent
    /// adapters' findGroupRange() calls work correctly.
    ///
    /// Single-region mode tracks firstInserted/lastInserted during splice-back
    /// and calls setGroupRange() — safe even if inner passes delete or reorder
    /// boundary instructions.
    ///
    /// Multi-region and whole-kernel modes do not record per-group boundaries —
    /// after inner passes run, there is no way to know which spliced-back
    /// instruction belongs to which group. If a subsequent ScopeAdaptor needs
    /// to call findGroupRange(), inner passes must NOT delete the boundary
    /// instructions that mark group ranges. If a future pass needs to delete
    /// boundaries, use separate single-region adapters instead.
    ///
    /// Example — whole-kernel adapter followed by a scoped adapter:
    ///
    /// @code
    ///   Group "region0" range: first -> [B]  last -> [D]
    ///
    ///   Before:  [A] <-> [B] <-> [C] <-> [D] <-> [E]
    ///                     ^               ^
    ///                   first            last
    ///
    ///   Whole-kernel adapter runs; inner pass deletes [B]:
    ///            [A] <-> [C'] <-> [D] <-> [E]
    ///
    ///   After splice-back, module's stored range still points to deleted [B]:
    ///            first -> [B]  (dangling!)
    ///
    ///   Subsequent scoped adapter calls findGroupRange("region0") -> undefined
    /// @endcode
    ///
    /// **Whole-kernel direct (no adapter)**: if the pass is last in the pipeline,
    /// add it directly to the outer PM — no extraction overhead, and no constraint
    /// on boundary instructions since no subsequent findGroupRange() call.
    class ScopeAdaptor : public Pass
    {
    public:
        static ID id()
        {
            static char c;
            return &c;
        }

        ID getPassID() const override
        {
            return id();
        }

        const char* getName() const override
        {
            return displayName.c_str();
        }

        ScopeAdaptor(StinkyAsmModule& module, std::vector<std::string> groupNames, PassManager pm)
            : module(module)
            , groupNames(std::move(groupNames))
            , innerPM(std::move(pm))
            , displayName("ScopeAdaptor(" + makeDebugLabel(this->groupNames) + ")")
        {
        }

        void run(Function& /*outerFunc*/, PassContext& outerCtx) override
        {
            // Propagate config from outer PassContext to inner PM
            innerPM.setGemmTileConfig(outerCtx.getGemmTileConfig());
            innerPM.setPassFeatureConfig(outerCtx.getPassFeatureConfig());

            if(groupNames.empty())
            {
                runWholeKernel();
                return;
            }

            if(groupNames.size() == 1)
            {
                runSingleRegion(groupNames[0]);
            }
            else
            {
                runMultiRegion();
            }
        }

    private:
        /// Extract a single group's instruction range to a temp Function,
        /// run the inner PM, and splice results back.
        void runSingleRegion(const std::string& groupName)
        {
            auto groupRange = module.findGroupRange(groupName);
            if(!groupRange)
                return;

            auto [begin, end]  = groupRange.value();
            BasicBlock* origBB = begin->getParent();

            // Create a temporary Function to hold the IR
            Function    tempFunc("temp");
            BasicBlock* bb = tempFunc.createBasicBlock("entry");

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
                    ir->erase();
                }
            }

            // Run the inner pipeline
            innerPM.run(tempFunc);

            // Splice back
            spliceBack(tempFunc, origBB, end, groupName);
        }

        /// Extract union of multiple groups' instruction ranges to a temp Function,
        /// run the inner PM, and splice results back.
        void runMultiRegion()
        {
            // Find the combined range across all groups
            IntrusiveListIterator<IRBase> combinedBegin;
            IntrusiveListIterator<IRBase> combinedEnd;
            BasicBlock*                   origBB = nullptr;

            for(const auto& groupName : groupNames)
            {
                auto groupRange = module.findGroupRange(groupName);
                if(!groupRange)
                    continue;

                auto [begin, end] = groupRange.value();

                if(!origBB)
                {
                    origBB        = begin->getParent();
                    combinedBegin = begin;
                    combinedEnd   = end;
                }
                else
                {
                    // Extend combined range to include this group
                    // We rely on groups being in program order within the same BB
                    combinedEnd = end;
                }
            }

            if(!origBB)
                return;

            // Create a temporary Function to hold the IR
            Function    tempFunc("temp");
            BasicBlock* bb = tempFunc.createBasicBlock("entry");

            // Move StinkyInstructions from the combined range
            for(auto it = combinedBegin; it != combinedEnd;)
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

            // Run the inner pipeline
            innerPM.run(tempFunc);

            // Splice back — update all group ranges
            IRBase* firstInserted = nullptr;
            IRBase* lastInserted  = nullptr;

            assert(module.getFunction().size() == 1
                   && "Current module should have only one basic block.");

            for(auto bbIt = tempFunc.begin(); bbIt != tempFunc.end(); bbIt++)
            {
                for(auto it = bbIt->begin(); it != bbIt->end();)
                {
                    IRBase* ir = it.getNodePtr();
                    it++;
                    origBB->insertIR(combinedEnd, ir);
                    if(!firstInserted)
                        firstInserted = ir;
                    lastInserted = ir;
                }
            }

            if(firstInserted && lastInserted)
            {
                // For multi-region, set the range for all groups to the combined range
                for(const auto& groupName : groupNames)
                {
                    module.setGroupRange(groupName,
                                         IntrusiveListIterator<IRBase>(firstInserted),
                                         IntrusiveListIterator<IRBase>(lastInserted));
                }
            }
        }

        /// Extract all instructions from the module's Function to a temp Function,
        /// run the inner PM, and splice results back.
        void runWholeKernel()
        {
            assert(module.getFunction().size() == 1
                   && "Current module should have only one basic block.");

            BasicBlock* origBB = &*module.getFunction().begin();

            // Create a temporary Function to hold the IR
            Function    tempFunc("temp");
            BasicBlock* bb = tempFunc.createBasicBlock("entry");

            // Move all StinkyInstructions from module to temporary function
            for(auto it = origBB->begin(); it != origBB->end();)
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

            // Run the inner pipeline
            innerPM.run(tempFunc);

            // Splice back — insert before end of origBB
            auto insertPos = origBB->end();
            for(auto bbIt = tempFunc.begin(); bbIt != tempFunc.end(); bbIt++)
            {
                for(auto it = bbIt->begin(); it != bbIt->end();)
                {
                    IRBase* ir = it.getNodePtr();
                    it++;
                    origBB->insertIR(insertPos, ir);
                }
            }
        }

        /// Splice IR from temporary function back to the original BasicBlock
        /// and update the group range.
        void spliceBack(Function&                     tempFunc,
                        BasicBlock*                   origBB,
                        IntrusiveListIterator<IRBase> insertPos,
                        const std::string&            groupName)
        {
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
                    origBB->insertIR(insertPos, ir);
                    if(!firstInserted)
                        firstInserted = ir;
                    lastInserted = ir;
                }
            }

            if(firstInserted && lastInserted)
            {
                module.setGroupRange(groupName,
                                     IntrusiveListIterator<IRBase>(firstInserted),
                                     IntrusiveListIterator<IRBase>(lastInserted));
            }
        }

        StinkyAsmModule&         module;
        std::vector<std::string> groupNames;
        PassManager              innerPM;
        std::string              displayName;
    };

    // -----------------------------------------------------------------------
    // Factory functions
    // -----------------------------------------------------------------------

    /// Create a ScopeAdaptor that runs an inner PM on a single named region.
    /// Configures debug output on the inner PM using the group name as file prefix.
    inline std::unique_ptr<Pass> createKernelToRegionPassAdaptor(StinkyAsmModule&   module,
                                                                 const std::string& groupName,
                                                                 PassManager        pm)
    {
        configureDebugOutput(pm, module.getModuleOptions(), groupName);
        return std::make_unique<ScopeAdaptor>(
            module, std::vector<std::string>{groupName}, std::move(pm));
    }

    /// Create a ScopeAdaptor that runs an inner PM on multiple named regions
    /// (their union is extracted, PM runs once, results spliced back).
    /// Configures debug output on the inner PM using joined group names as file prefix.
    inline std::unique_ptr<Pass> createKernelToRegionsPassAdaptor(
        StinkyAsmModule& module, std::vector<std::string> groupNames, PassManager pm)
    {
        configureDebugOutput(pm, module.getModuleOptions(), makeDebugLabel(groupNames));
        return std::make_unique<ScopeAdaptor>(module, std::move(groupNames), std::move(pm));
    }

    /// Create a ScopeAdaptor that runs an inner PM on the whole kernel
    /// (extracts everything to temp, runs, splices back to restore flat BB).
    /// Configures debug output on the inner PM with the "wholeKernel" label.
    inline std::unique_ptr<Pass> createKernelPassAdaptor(StinkyAsmModule& module, PassManager pm)
    {
        configureDebugOutput(pm, module.getModuleOptions(), "wholeKernel");
        return std::make_unique<ScopeAdaptor>(module, std::vector<std::string>{}, std::move(pm));
    }

} // namespace stinkytofu
