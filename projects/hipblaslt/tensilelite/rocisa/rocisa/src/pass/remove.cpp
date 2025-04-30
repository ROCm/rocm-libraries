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
#include "code.hpp"
#include "instruction/common.hpp"
#include "pass.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rocisa
{
    std::unordered_map<std::string, std::vector<std::shared_ptr<Module>>>
        _findActFunc(std::shared_ptr<Module> module)
    {
        std::unordered_map<std::string, std::vector<std::shared_ptr<Module>>> modFunc;
        for(auto item : module->items())
        {
            if(auto mod = std::dynamic_pointer_cast<Module>(item))
            {
                if(mod->name.find("ActFunc_VW") != std::string::npos)
                {
                    if(modFunc.find(mod->name) != modFunc.end())
                    {
                        modFunc[mod->name].push_back(mod);
                    }
                    else
                    {
                        modFunc[mod->name] = {mod};
                    }
                }
                else
                {
                    auto tmp = _findActFunc(mod);
                    for(auto& [key, t] : tmp)
                    {
                        if(modFunc.find(key) != modFunc.end())
                        {
                            modFunc[key].insert(modFunc[key].end(), t.begin(), t.end());
                        }
                        else
                        {
                            modFunc[key] = t;
                        }
                    }
                }
            }
        }
        return std::move(modFunc);
    }

    void _replaceActBranchLabel(std::shared_ptr<Module> module, std::vector<std::string> labels)
    {
        for(auto item : module->items())
        {
            if(auto mod = std::dynamic_pointer_cast<Module>(item))
            {
                if(mod->name.find("InsertActFuncCallAddrCalc") != std::string::npos)
                {
                    std::string labelFirst  = labels[0];
                    int  numUnderScores     = std::count(labelFirst.begin(), labelFirst.end(), '_');
                    auto partFirst          = labelFirst.rfind("_");
                    std::string lastPostfix = labelFirst.substr(partFirst + 1);
                    auto        labelLeft   = labels.begin() + 1;
                    bool        replaceLabel = false;
                    for(auto inst : mod->items())
                    {
                        if(auto addInst = std::dynamic_pointer_cast<SAddI32>(inst))
                            if(addInst->comment == "target branch offset")
                            {
                                auto        namePtr = std::get_if<std::string>(&addInst->srcs[0]);
                                std::string name    = *namePtr;
                                if(std::find(labelLeft, labels.end(), name) != labels.end())
                                {
                                    replaceLabel = true;
                                    break;
                                }
                            }
                    }
                    if(replaceLabel)
                    {
                        for(int idx = 0; idx < mod->items().size(); ++idx)
                        {
                            auto inst = mod->getItem(idx);
                            if(auto addInst = std::dynamic_pointer_cast<SAddI32>(inst))
                                if(addInst->comment == "target branch offset")
                                {
                                    // The label is generated in the format of XXXX_1, XXXX_2
                                    // and string.rpartition  returns('XXXX', '_', '1').
                                    // We only need the first string.
                                    auto namePtr      = std::get_if<std::string>(&addInst->srcs[0]);
                                    std::string name  = *namePtr;
                                    int         numUS = std::count(name.begin(), name.end(), '_');
                                    if(numUnderScores == numUS)
                                    {
                                        auto part        = name.rfind("_");
                                        addInst->srcs[0] = name.substr(0, part) + "_" + lastPostfix;
                                    }
                                    else if(numUnderScores == numUS - 1)
                                    {
                                        auto part        = name.rfind("_");
                                        addInst->srcs[0] = name.substr(0, part);
                                    }
                                    else
                                    {
                                        throw std::runtime_error("Incorrect Activation Label");
                                    }
                                }
                        }
                    }
                }
                else
                {
                    _replaceActBranchLabel(mod, labels);
                }
            }
        }
    }

    void _removeDuplicatedActivationFunctions(std::shared_ptr<Module> module)
    {
        auto modFunc    = _findActFunc(module);
        auto moduleLast = std::make_shared<Module>("AddToLast");
        for(auto& [key, mlist] : modFunc)
        {
            if(mlist.size() > 1)
            {
                std::vector<std::string> labels;
                for(auto ml : mlist)
                {
                    if(auto mod = std::dynamic_pointer_cast<Module>(ml))
                    {
                        auto        mod2      = std::dynamic_pointer_cast<Module>(mod->items()[0]);
                        auto        label     = std::dynamic_pointer_cast<Label>(mod2->items()[0]);
                        std::string labelName = label->getLabelName();
                        labels.push_back(labelName);
                        auto mod3 = dynamic_cast<Module*>(mod->parent);
                        mod3->removeItem(mod);
                    }
                }
                // Avoid using deepcopy
                moduleLast->add(mlist[0]);

                _replaceActBranchLabel(module, labels);
            }
        }
        if(!moduleLast->items().empty())
        {
            module->add(moduleLast);
            module->addT<SEndpgm>();
        }
    }

    // Basic functions for generating the module name
    inline std::string
        _getModuleName(const char* name, int gwvw, int sgpr, int tmpVgpr, int tmpSgpr)
    {
        return std::string(name) + "_VW" + std::to_string(gwvw) + "_Sgpr" + std::to_string(sgpr)
               + "_Tmp" + std::to_string(tmpVgpr) + "_" + std::to_string(tmpSgpr);
    }

    // Basic functions for generating the branch module name
    inline std::string _getBranchModuleName(const char* name)
    {
        return "Insert" + std::string(name) + "CallAddrCalc";
    }

    // Public functions
    std::string getActFuncModuleName(int gwvw, int sgpr, int tmpVgpr, int tmpSgpr)
    {
        return _getModuleName("ActFunc", gwvw, sgpr, tmpVgpr, tmpSgpr);
    }

    std::string getActFuncBranchModuleName()
    {
        return _getBranchModuleName("ActFunc");
    }

    void removeDuplicatedFunction(std::shared_ptr<Module> module)
    {
        _removeDuplicatedActivationFunctions(module);
    }
} // namespace rocisa
