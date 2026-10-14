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

    void _convertTextVariablesToRegisters(std::shared_ptr<Module>               module,
                                          std::unordered_map<std::string, int>& assignmentDict)
    {
        for(auto item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _convertTextVariablesToRegisters(subModule, assignmentDict);
            }
            else if(auto inst = std::dynamic_pointer_cast<Instruction>(item))
            {
                auto params = inst->getParams();
                for(auto p : params)
                {
                    if(auto pptr = std::get_if<std::shared_ptr<Container>>(&p))
                    {
                        if(auto regContainer = std::dynamic_pointer_cast<RegisterContainer>(*pptr))
                        {
                            _setName2RegNum(regContainer.get(), assignmentDict);
                        }
                    }
                }
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

    void convertTextVariablesToRegisters(std::shared_ptr<Module> module)
    {
        std::unordered_map<std::string, int> assignmentDict;
        _convertTextVariablesToRegisters(module, assignmentDict);
    }

    // Classification used by _removeDuplicateAssignmentGPR, in its check order.
    Graph::Kind _classifyItem(const std::shared_ptr<Item>& item)
    {
        if(std::dynamic_pointer_cast<BranchInstruction>(item)
           || std::dynamic_pointer_cast<Label>(item)
           || std::dynamic_pointer_cast<MacroInstruction>(item)
           || std::dynamic_pointer_cast<_SWaitCnt>(item)
           || std::dynamic_pointer_cast<_SWaitCntVscnt>(item))
            return Graph::Kind::Reset;
        if(std::dynamic_pointer_cast<SMovB32>(item))
            return Graph::Kind::SMov;
        if(std::dynamic_pointer_cast<Instruction>(item))
            return Graph::Kind::Instr;
        return Graph::Kind::Other;
    }

    // An Instr entry resets the tracked sgpr value when its first param is an
    // sgpr range covering the list's index (only for >1 params).
    std::shared_ptr<RegisterContainer>
        _dstSgpr(const std::vector<InstructionInput>& params)
    {
        if(params.size() > 1)
        {
            if(auto pptr = std::get_if<std::shared_ptr<Container>>(&params[0]))
            {
                auto gpr = std::dynamic_pointer_cast<RegisterContainer>(*pptr);
                if(gpr && gpr->regType == "s")
                    return gpr;
            }
        }
        return nullptr;
    }

    void _pushSgprEntry(Graph&                                    graph,
                        size_t                                    i,
                        const std::shared_ptr<Item>&              item,
                        Graph::Kind                               kind,
                        bool                                      noOpt,
                        const std::shared_ptr<RegisterContainer>& dst)
    {
        bool dstHit = kind == Graph::Kind::Instr && dst && (int)i >= dst->regIdx
                      && (int)i < dst->regIdx + dst->regNum;
        graph.sgpr[i].push_back({item, graph.barrierSeq, kind, noOpt, dstHit});
    }

    void _addRegToGraph(std::shared_ptr<Item>                item,
                        const std::vector<InstructionInput>& params,
                        Graph&                               graph,
                        bool                                 noOpt)
    {
        bool                               classified = false;
        Graph::Kind                        kind       = Graph::Kind::Other;
        std::shared_ptr<RegisterContainer> dst;
        for(const auto& p : params)
        {
            if(auto pptr = std::get_if<std::shared_ptr<Container>>(&p))
            {
                auto regContainer = dynamic_cast<RegisterContainer*>(pptr->get());
                if(!regContainer)
                    continue;
                if(regContainer->regType == "acc")
                    continue;
                if(regContainer->regIdx < 0)
                {
                    std::cerr << "ERROR: negative regIdx detected!" << std::endl;
                    std::cerr << "  regContainer: " << regContainer->toString() << std::endl;
                    std::cerr << "  instruction:  " << item->toString() << std::endl;
                    throw std::runtime_error("GPR index out of range (negative regIdx)");
                }
                int    lo = regContainer->regIdx;
                int    hi = regContainer->regIdx + regContainer->regNum;
                size_t gprSize;
                if(regContainer->regType == "s")
                    gprSize = graph.sgpr.size();
                else if(regContainer->regType == "v")
                    gprSize = graph.vgprSize;
                else if(regContainer->regType == "m")
                    gprSize = graph.mgprSize;
                else if(lo < hi)
                    throw std::runtime_error("Invalid gpr type");
                else
                    continue;
                if(lo < hi && (size_t)(hi - 1) >= gprSize)
                {
                    std::cerr << "regContainer: " << regContainer->toString() << std::endl;
                    std::cerr << "gprvec.size(): " << gprSize << std::endl;
                    throw std::runtime_error("GPR index out of range");
                }
                if(regContainer->regType == "s")
                {
                    if(!classified)
                    {
                        kind       = noOpt ? Graph::Kind::Reset : _classifyItem(item);
                        dst        = _dstSgpr(params);
                        classified = true;
                    }
                    for(int i = lo; i < hi; ++i)
                    {
                        auto& list = graph.sgpr[i];
                        // A NoOptItem wrapper never compares equal to item.
                        if(!noOpt && !list.empty() && !list.back().noOpt
                           && list.back().item == item
                           && list.back().barrierSeq == graph.barrierSeq)
                            continue;
                        _pushSgprEntry(graph, i, item, kind, noOpt, dst);
                    }
                }
                // Track highest VGPR index seen in any instruction.  Done
                // outside the inner loop using regIdx+regNum-1 to avoid
                // repeated comparisons; acc registers are already skipped.
                else if(regContainer->regType == "v")
                {
                    int lastIdx = hi - 1;
                    if(lastIdx > graph.maxVgprSeen)
                        graph.maxVgprSeen = lastIdx;
                }
            }
        }
    }

    void _recordGraph(std::shared_ptr<Module> module, Graph& graph)
    {
        for(const auto& item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _recordGraph(subModule, graph);
            }
            else if(std::dynamic_pointer_cast<CommonInstruction>(item)
                    || std::dynamic_pointer_cast<ReadWriteInstruction>(item)
                    || std::dynamic_pointer_cast<MacroInstruction>(item))
            {
                auto inst = std::dynamic_pointer_cast<Instruction>(item);
                _addRegToGraph(item, inst->getParams(), graph, module->isNoOpt());
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
                // Barriers apply to every sgpr.  Resetting ones are recorded
                // via resetsBefore; others only matter if they write an sgpr,
                // in which case they are pushed into every list as before.
                auto kind  = _classifyItem(item);
                bool reset = kind == Graph::Kind::Reset;
                if(kind == Graph::Kind::SMov)
                {
                    for(size_t i = 0; i < graph.sgpr.size(); ++i)
                        _pushSgprEntry(graph, i, item, kind, false, nullptr);
                }
                else if(kind == Graph::Kind::Instr)
                {
                    auto dst = _dstSgpr(std::dynamic_pointer_cast<Instruction>(item)->getParams());
                    if(dst)
                        for(size_t i = 0; i < graph.sgpr.size(); ++i)
                            _pushSgprEntry(graph, i, item, kind, false, dst);
                }
                graph.barrierSeq++;
                graph.resetsBefore.push_back(graph.resetsBefore.back() + (reset ? 1 : 0));
            }
        }
    }

    Graph buildGraph(std::shared_ptr<Module> module, int vgprMax, int sgprMax)
    {
        Graph graph;
        graph.vgprSize = vgprMax;
        graph.sgpr     = std::vector<std::vector<Graph::Entry>>(sgprMax);
        graph.mgprSize = 1;
        _recordGraph(module, graph);
        return graph;
    }

    void _removeDuplicateAssignmentGPR(Graph& graph)
    {
        auto& graphRef = graph.sgpr;
        for(size_t idx = 0; idx < graphRef.size(); ++idx)
        {
            std::optional<InstructionInput> assignValue = std::nullopt;
            int                             prevSeq     = 0;
            for(auto& entry : graphRef[idx])
            {
                if(graph.resetsBefore[entry.barrierSeq] != graph.resetsBefore[prevSeq])
                    assignValue = std::nullopt;
                prevSeq = entry.barrierSeq;
                if(entry.kind == Graph::Kind::Reset)
                {
                    assignValue = std::nullopt;
                }
                else if(entry.kind == Graph::Kind::SMov)
                {
                    auto  smov     = std::static_pointer_cast<SMovB32>(entry.item);
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
                        }
                    }
                    assignValue = gprValue;
                }
                else if(entry.kind == Graph::Kind::Instr && entry.dstHit)
                {
                    assignValue = std::nullopt;
                }
            }
        }
    }

    void removeDuplicateAssignment(Graph& graph)
    {
        _removeDuplicateAssignmentGPR(graph);
    }
} // namespace rocisa
