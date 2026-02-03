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
#include "ir/asm/StinkyAsmModule.hpp"
#include "ir/asm/Backend.hpp"
#include "ir/asm/StinkyAsmEmitter.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmPrinter.hpp"
#include "stinkypasses.hpp"

#include <sstream>

namespace stinkytofu
{
    struct StinkyAsmModule::Impl
    {
        std::string        name;
        std::array<int, 3> arch;

        // This map maintains the defined group names and the index of the groups.
        // Each instruction could belong to one or more groups.
        std::unordered_map<std::string, int>                       groupNameIndexMap;
        std::unordered_map<const IRBase*, std::unordered_set<int>> instructionGroupsMap;

        // The IRList is owned by a Function/BasicBlock
        // We maintain a pointer to it for compatibility
        std::unique_ptr<Function> function;
        BasicBlock*               basicBlock;

        Impl(const std::string& name, const std::array<int, 3>& arch)
            : name(name)
            , arch(arch)
            , function(std::make_unique<Function>(name))
            , basicBlock(nullptr)
        {
            // Create a single BasicBlock to hold all instructions
            basicBlock = function->createBasicBlock("entry");
            function->setEntryBlock(basicBlock);
        }

        ~Impl()
        {
            // Function destructor will clean up BasicBlocks and their IRLists
            // The IRBases in the IRList will be deleted by the Function
        }
    };

    StinkyAsmModule::StinkyAsmModule(const std::string& name, const std::array<int, 3>& arch)
        : pImpl(std::make_unique<Impl>(name, arch))
    {
    }

    StinkyAsmModule::~StinkyAsmModule() = default;

    StinkyAsmModule::StinkyAsmModule(StinkyAsmModule&&) noexcept            = default;
    StinkyAsmModule& StinkyAsmModule::operator=(StinkyAsmModule&&) noexcept = default;

    std::string StinkyAsmModule::getName() const
    {
        return pImpl->name;
    }

    std::array<int, 3> StinkyAsmModule::getArch() const
    {
        return pImpl->arch;
    }

    void StinkyAsmModule::setInstructionGroups(IRBase* inst, const std::vector<int>& groups)
    {
        if(inst)
        {
            if(!groups.empty())
            {
                pImpl->instructionGroupsMap[inst]
                    = std::unordered_set<int>(groups.begin(), groups.end());
            }
            else if(pImpl->instructionGroupsMap.find(inst) != pImpl->instructionGroupsMap.end())
            {
                pImpl->instructionGroupsMap.erase(inst);
            }
        }
    }

    size_t StinkyAsmModule::size() const
    {
        return getIRList().size();
    }

    std::string StinkyAsmModule::toString() const
    {
        std::ostringstream oss;
        oss << "StinkyAsmModule: " << pImpl->name;
        oss << " (arch: " << pImpl->arch[0] << "." << pImpl->arch[1] << "." << pImpl->arch[2]
            << ")\n";
        oss << stinkytofu::toString(getIRList());

        return oss.str();
    }

    std::string StinkyAsmModule::emitAssembly() const
    {
        // Configure the emitter with default options
        stinkytofu::AsmEmitterOptions options;
        options.emitComments     = true;
        options.emitCycleInfo    = false;
        options.indent           = 0;
        options.emitBlankLines   = false;
        options.useSymbolicNames = true; // Enable symbolic register names

        stinkytofu::StinkyAsmEmitter emitter(options);
        return emitter.emit(getIRList());
    }

    void StinkyAsmModule::runOptimizationPipeline()
    {
        // Run optimization pipeline using the backend
        Backend backend(*this);
        backend.runOptimization();
    }

    IRList& StinkyAsmModule::getIRList()
    {
        return pImpl->basicBlock->getIR();
    }

    const IRList& StinkyAsmModule::getIRList() const
    {
        return pImpl->basicBlock->getIR();
    }

    void StinkyAsmModule::addGroup(const std::string& name)
    {
        if(pImpl->groupNameIndexMap.find(name) != pImpl->groupNameIndexMap.end())
        {
            return;
        }
        pImpl->groupNameIndexMap[name] = pImpl->groupNameIndexMap.size();
    }

    int StinkyAsmModule::getGroupIndex(const std::string& name) const
    {
        return pImpl->groupNameIndexMap.at(name);
    }

    bool StinkyAsmModule::hasGroup(const std::string& name) const
    {
        return pImpl->groupNameIndexMap.find(name) != pImpl->groupNameIndexMap.end();
    }

    const std::unordered_map<std::string, int>& StinkyAsmModule::getGroupNameIndexMap() const
    {
        return pImpl->groupNameIndexMap;
    }

    std::optional<std::unordered_set<int>> StinkyAsmModule::getInstructionGroups(IRBase* inst) const
    {
        if(inst == nullptr
           || pImpl->instructionGroupsMap.find(inst) == pImpl->instructionGroupsMap.end())
        {
            return std::nullopt;
        }
        return std::make_optional(pImpl->instructionGroupsMap.at(inst));
    }

    bool StinkyAsmModule::isInstructionInGroup(const IRBase* inst, int group) const
    {
        if(inst == nullptr
           || pImpl->instructionGroupsMap.find(inst) == pImpl->instructionGroupsMap.end())
        {
            return false;
        }
        return pImpl->instructionGroupsMap.at(inst).find(group)
               != pImpl->instructionGroupsMap.at(inst).end();
    }

    void StinkyAsmModule::refreshInstructionGroups()
    {
        std::vector<IntrusiveListIterator<IRBase>> instructionsWithoutGroup;
        for(auto it = getIRList().begin(); it != getIRList().end(); ++it)
        {
            if(pImpl->instructionGroupsMap.find(it.getNodePtr())
               == pImpl->instructionGroupsMap.end())
            {
                instructionsWithoutGroup.push_back(it);
            }
        }

        // If there are no instructions without groups, return
        if(instructionsWithoutGroup.empty())
        {
            return;
        }

        for(auto it : instructionsWithoutGroup)
        {
            // Find the previous instruction that has groups
            auto beginInstIt = it;
            while(--beginInstIt != getIRList().begin())
            {
                if(pImpl->instructionGroupsMap.find(beginInstIt.getNodePtr())
                   != pImpl->instructionGroupsMap.end())
                {
                    break;
                }
            }

            // Find the next instruction that has groups
            auto endInstIt = it;
            while(++endInstIt != getIRList().end())
            {
                if(pImpl->instructionGroupsMap.find(endInstIt.getNodePtr())
                   != pImpl->instructionGroupsMap.end())
                {
                    break;
                }
            }

            auto beginGroups    = getInstructionGroups(beginInstIt.getNodePtr());
            int  beginGroupDeep = beginGroups.has_value() ? beginGroups.value().size() : 0;

            auto endGroups    = getInstructionGroups(endInstIt.getNodePtr());
            int  endGroupDeep = endGroups.has_value() ? endGroups.value().size() : 0;

            // Set the most deep groups for the instruction without groups
            std::vector<int> groups;
            if(beginGroupDeep > endGroupDeep)
            {
                groups.assign(beginGroups.value().begin(), beginGroups.value().end());
            }
            else
            {
                groups.assign(endGroups.value().begin(), endGroups.value().end());
            }
            setInstructionGroups(it.getNodePtr(), groups);
        }
    }

    StinkyAsmModule::GroupRange StinkyAsmModule::findGroupRange(const std::string& groupName)
    {
        if(!hasGroup(groupName))
        {
            return std::make_pair(IntrusiveListIterator<IRBase>(), IntrusiveListIterator<IRBase>());
        }
        return findGroupRange(getGroupIndex(groupName));
    }

    /* private methods */
    StinkyAsmModule::GroupRange StinkyAsmModule::findGroupRange(int groupIndex)
    {
        IntrusiveListIterator<IRBase> begin      = getIRList().begin();
        IntrusiveListIterator<IRBase> end        = getIRList().end();
        bool                          foundBegin = false;
        for(auto it = getIRList().begin(); it != getIRList().end(); ++it)
        {
            if(!foundBegin && isInstructionInGroup(it.getNodePtr(), groupIndex))
            {
                begin      = it;
                foundBegin = true;
            }

            if(foundBegin && !isInstructionInGroup(it.getNodePtr(), groupIndex))
            {
                end = it;
                break;
            }
        }

        return std::make_pair(begin, end);
    }

} // namespace stinkytofu
