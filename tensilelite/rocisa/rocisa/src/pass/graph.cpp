/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include "instruction/branch.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"
#include "pass.hpp"

#include <optional>
#include <typeinfo>

namespace rocisa
{
    void _getAssignmentDictIter(const std::shared_ptr<Module>&        module,
                                std::unordered_map<std::string, int>& assignmentDict)
    {
        for(const auto& item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _getAssignmentDictIter(subModule, assignmentDict);
            }
            else if(auto regSet = std::dynamic_pointer_cast<RegSet>(item))
            {
                int num = 0;
                if(regSet->ref)
                {
                    num = assignmentDict[*regSet->ref] + regSet->offset;
                }
                else
                {
                    num = *regSet->value;
                }
                assignmentDict[regSet->name] = num;
            }
        }
    }

    std::unordered_map<std::string, int> getAssignmentDict(const std::shared_ptr<Module>& module)
    {
        std::unordered_map<std::string, int> assignmentDict;
        _getAssignmentDictIter(module, assignmentDict);
        return std::move(assignmentDict);
    }

    // Find ".set AAAAA 0" and convert "s[AAAAA]" into "s0"
    std::vector<int> _setName2RegNum(RegisterContainer*                    gpr,
                                     std::unordered_map<std::string, int>& assignmentDict)
    {
        if(gpr->regIdx == -1 && gpr->regName)
        {
            std::string name = gpr->getRegNameWithType();
            int         num  = assignmentDict[name] + gpr->regName->getTotalOffsets();
            gpr->regIdx      = num;
        }
        std::vector<int> RegNumList;
        for(int i = 0; i < gpr->regNum; ++i)
        {
            RegNumList.push_back(i + gpr->regIdx);
        }
        return std::move(RegNumList);
    }

    void _addRegToGraph(std::shared_ptr<Item>                 item,
                        std::unordered_map<std::string, int>& assignmentDict,
                        const std::vector<InstructionInput>&  params,
                        Graph&                                graph,
                        bool                                  noOpt)
    {
        for(auto p : params)
        {
            if(auto pptr = std::get_if<std::shared_ptr<Container>>(&p))
            {
                if(auto regContainer = std::dynamic_pointer_cast<RegisterContainer>(*pptr))
                {
                    _setName2RegNum(regContainer.get(), assignmentDict);
                    if(regContainer->regType == "acc")
                        continue;
                    for(int i = regContainer->regIdx;
                        i < regContainer->regIdx + regContainer->regNum;
                        ++i)
                    {
                        auto& gprvec = graph.getGprRef(regContainer->regType);
                        if(i >= gprvec.size())
                        {
                            std::cerr << "regContainer: " << regContainer->toString() << std::endl;
                            std::cerr << "gprvec.size(): " << gprvec.size() << std::endl;
                            throw std::runtime_error("GPR index out of range");
                        }
                        else if(!gprvec[i].empty() && gprvec[i].back() == item)
                            continue;
                        if(noOpt)
                            gprvec[i].push_back(std::make_shared<NoOptItem>(item));
                        else
                            gprvec[i].push_back(item);
                    }
                }
            }
        }
    }

    void _recordGraph(std::shared_ptr<Module>               module,
                      Graph&                                graph,
                      std::unordered_map<std::string, int>& assignmentDict)
    {
        for(auto item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _recordGraph(subModule, graph, assignmentDict);
            }
            else if(std::dynamic_pointer_cast<CommonInstruction>(item)
                    || std::dynamic_pointer_cast<ReadWriteInstruction>(item)
                    || std::dynamic_pointer_cast<MacroInstruction>(item))
            {
                auto inst = std::dynamic_pointer_cast<Instruction>(item);
                _addRegToGraph(item, assignmentDict, inst->getParams(), graph, module->isNoOpt());
            }
            else if(std::dynamic_pointer_cast<BranchInstruction>(item)
                    || std::dynamic_pointer_cast<Label>(item)
                    || std::dynamic_pointer_cast<_SWaitCnt>(item)
                    || std::dynamic_pointer_cast<_SWaitCntVscnt>(item)
                    || std::dynamic_pointer_cast<SEndpgm>(item)
                    || std::dynamic_pointer_cast<SBarrier>(item)
                    || std::dynamic_pointer_cast<SNop>(item)
                    || std::dynamic_pointer_cast<SSleep>(item))
            {
                for(auto& v : graph.vgpr)
                    v.push_back(item);
                for(auto& s : graph.sgpr)
                    s.push_back(item);
            }
        }
    }

    Graph buildGraph(std::shared_ptr<Module>               module,
                     int                                   vgprMax,
                     int                                   sgprMax,
                     std::unordered_map<std::string, int>& assignmentDict)
    {
        Graph graph;
        graph.vgpr = std::move(std::vector<std::vector<std::shared_ptr<Item>>>(vgprMax));
        graph.sgpr = std::move(std::vector<std::vector<std::shared_ptr<Item>>>(sgprMax));
        graph.mgpr = std::move(std::vector<std::vector<std::shared_ptr<Item>>>(1));
        _recordGraph(module, graph, assignmentDict);
        return std::move(graph);
    }

    void _removeDuplicateAssignmentGPR(Graph& graph, const std::string& regType)
    {
        auto& graphRef = graph.getGprRef(regType);
        for(size_t idx = 0; idx < graphRef.size(); ++idx)
        {
            auto&                              sList       = graphRef[idx];
            std::optional<InstructionInput>    assignValue = std::nullopt;
            std::vector<std::shared_ptr<Item>> newList;
            for(auto& item : sList)
            {
                bool isRemoved = false;
                if(std::dynamic_pointer_cast<NoOptItem>(item)
                   || std::dynamic_pointer_cast<BranchInstruction>(item)
                   || std::dynamic_pointer_cast<Label>(item)
                   || std::dynamic_pointer_cast<MacroInstruction>(item)
                   || std::dynamic_pointer_cast<_SWaitCnt>(item)
                   || std::dynamic_pointer_cast<_SWaitCntVscnt>(item))
                {
                    assignValue = std::nullopt;
                }
                else if(auto smov = std::dynamic_pointer_cast<SMovB32>(item))
                {
                    auto& dst      = *smov->dst;
                    auto& gprValue = smov->srcs[0];
                    if(typeid(dst) != typeid(EXEC))
                    {
                        auto gpr = dynamic_cast<RegisterContainer*>(&dst);
                        if(gpr->regIdx == idx && gprValue == assignValue)
                        {
                            if(!smov->comment.empty())
                            {
                                auto comment = smov->comment + " (dup assign opt.)";
                                auto newItem = std::make_shared<TextBlock>(slash50(comment));
                                auto module  = dynamic_cast<Module*>(smov->parent);
                                module->replaceItem(smov, newItem);
                            }
                            else
                            {
                                auto module = dynamic_cast<Module*>(smov->parent);
                                module->removeItem(smov);
                            }
                            isRemoved = true;
                        }
                    }
                    assignValue = gprValue;
                }
                else if(auto instr = std::dynamic_pointer_cast<Instruction>(item))
                {
                    auto params = instr->getParams();
                    if(params.size() > 1)
                    {
                        auto pptr = std::get_if<std::shared_ptr<Container>>(&params[0]);
                        if(!pptr)
                            continue;
                        auto gpr = std::dynamic_pointer_cast<RegisterContainer>(*pptr);
                        if(gpr && gpr->regType == regType)
                        {
                            for(int i = gpr->regIdx; i < gpr->regIdx + gpr->regNum; ++i)
                            {
                                if(i == idx)
                                {
                                    assignValue = std::nullopt;
                                    break;
                                }
                            }
                        }
                    }
                }
                if(!isRemoved)
                {
                    newList.push_back(std::move(item));
                }
            }
            if(newList.size() != sList.size())
            {
                graph.getGprRef(regType)[idx] = std::move(newList);
            }
        }
    }

    void removeDuplicateAssignment(Graph& graph)
    {
        _removeDuplicateAssignmentGPR(graph, "s");
    }
} // namespace rocisa
