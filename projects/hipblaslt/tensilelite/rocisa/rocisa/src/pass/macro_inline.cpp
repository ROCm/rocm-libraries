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

#include "instruction/common.hpp"
#include "pass.hpp"

#include <cassert>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rocisa
{
    // param name -> param value
    using MacroParam  = std::tuple<std::string, std::string>;
    using MacroParams = std::vector<MacroParam>;
    using MacroEntry  = std::tuple<std::shared_ptr<Macro>, std::shared_ptr<MacroParams>>;
    using MacroTable  = std::unordered_map<std::string, MacroEntry>;

    static void extractMacro(MacroTable& macros, std::shared_ptr<Macro> macro)
    {
        auto   params = std::make_shared<MacroParams>();
        macros[macro->macro->name] = std::make_tuple(macro, params);

        std::regex  argPattern("([^=:]+)(?::req)?=?(\\w+)?");
        std::smatch match;
        for(auto& arg : macro->macro->args)
        {
            auto argStr = std::get<std::string>(arg);
            std::regex_match(argStr, match, argPattern);
            params->push_back(std::make_tuple(
                match[1].str(), match.size() > 2 ? match[2].str() : std::string()));
        }
    }

    static void collectMacros(std::shared_ptr<Module> module, MacroTable& macros)
    {
        for(auto& item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
                collectMacros(subModule, macros);
            else if(auto macro = std::dynamic_pointer_cast<Macro>(item))
                if(macros.find(macro->macro->name) == macros.end())
                    extractMacro(macros, macro);
        }
    }

    // Replace whole-word occurrences of `needle` with `replacement` in `str`
    static void replaceWholeWord(std::string&       str,
                                 const std::string& needle,
                                 const std::string& replacement)
    {
        size_t pos = 0;
        while((pos = str.find(needle, pos)) != std::string::npos)
        {
            size_t endPos = pos + needle.size();
            bool boundaryBefore = (pos == 0) || !(std::isalnum(str[pos - 1]) || str[pos - 1] == '_');
            bool boundaryAfter  = (endPos >= str.size()) || !(std::isalnum(str[endPos]) || str[endPos] == '_');
            if(boundaryBefore && boundaryAfter)
            {
                str.replace(pos, needle.size(), replacement);
                pos += replacement.size();
            }
            else
            {
                pos = endPos;
            }
        }
    }

    // Evaluate .if / .elseif conditions inside macro bodies
    static bool evalMacroCondition(const std::string& value, const MacroParams& params)
    {
        std::regex           tokenPattern("\\\\([^=\\s]+)|\\w+|==|!=|&&");
        std::sregex_iterator it(value.begin(), value.end(), tokenPattern);
        std::sregex_iterator end;

        std::string      lhs, rhs, op, val;
        int              tokenIdx = 0;
        bool             result   = false;
        std::vector<int> results;

        while(it != end)
        {
            std::smatch match = *it;
            val = match.str(0);
            if(val[0] == '\\')
            {
                auto var     = match.str(1);
                auto find_it = std::find_if(
                    params.begin(), params.end(), [&var](auto& p) {
                        return std::get<0>(p) == var;
                    });
                assert(find_it != params.end() && "unknown macro argument in condition");
                val = std::get<1>(*find_it);
            }

            switch(tokenIdx)
            {
            case 0: lhs = val; break;
            case 1: op = val; break;
            case 2:
                rhs = val;
                if(op == "==")
                    result = (lhs == rhs);
                else if(op == "!=")
                    result = (lhs != rhs);
                else
                    assert(false && "unknown macro condition operator");
                results.push_back(result ? 1 : 0);
                break;
            case 4:
                assert(val == "&&" && "unknown macro logical operator");
                results.push_back(2); // sentinel for &&
                break;
            }
            tokenIdx = (tokenIdx + 1) % 4;
            ++it;
        }

        result = results.front();
        for(size_t i = 1; i < results.size(); i++)
        {
            if(results[i] == 2)
            {
                result = result & results[i + 1];
                i++;
            }
        }
        return result;
    }

    // Replace \paramName with the argument value in a string operand
    static std::string substituteStringParam(const std::string& str,
                                             const MacroParams& params)
    {
        std::string result = str;
        for(auto& [name, value] : params)
        {
            std::string needle = "\\" + name;
            replaceWholeWord(result, needle, value);
        }
        return result;
    }

    // Substitute macro params in a RegisterContainer's symbolic name
    static void substituteRegisterParam(RegisterContainer& reg, const MacroParams& params)
    {
        if(!reg.isMacro || !reg.regName)
            return;

        auto nameStr = reg.regName->toString();

        for(auto& [paramName, paramValue] : params)
        {
            // Strip register type prefix (vgpr→v, sgpr→s) from param name
            // to match how regName stores it: param "vgprDstIdx" → regName "DstIdx"
            std::string stripped;
            if(paramName.size() > 4
               && (paramName.substr(0, 4) == "vgpr" || paramName.substr(0, 4) == "sgpr"
                   || paramName.substr(0, 4) == "mgpr"))
                stripped = paramName.substr(4);
            else
                stripped = paramName;

            replaceWholeWord(nameStr, stripped, paramValue);
        }

        reg.isMacro = false;

        // Strip register type prefix from substituted value (e.g. "v5" → "5")
        // Try full prefix first ("vgpr") since toString() adds regType+"gpr",
        // then fall back to single-letter prefix ("v") for numeric values like "v5".
        std::string fullPrefix = reg.regType + "gpr";
        if(nameStr.size() > fullPrefix.size()
           && nameStr.substr(0, fullPrefix.size()) == fullPrefix)
            nameStr = nameStr.substr(fullPrefix.size());
        else if(nameStr.size() > reg.regType.size()
                && nameStr.substr(0, reg.regType.size()) == reg.regType)
            nameStr = nameStr.substr(reg.regType.size());

        // Try to evaluate as integer index or simple expression like "1+1"
        try
        {
            size_t plusPos = nameStr.find('+');
            if(plusPos != std::string::npos)
            {
                reg.regIdx = std::stoi(nameStr.substr(0, plusPos))
                             + std::stoi(nameStr.substr(plusPos + 1));
            }
            else
            {
                reg.regIdx = std::stoi(nameStr);
            }
            reg.regName.reset();
        }
        catch(...)
        {
            reg.regName = RegName(nameStr);
        }
    }

    // Substitute macro params in a single InstructionInput
    static InstructionInput substituteInput(const InstructionInput& input,
                                            const MacroParams&      params)
    {
        return std::visit(
            [&params](auto&& arg) -> InstructionInput {
                using T = std::decay_t<decltype(arg)>;
                if constexpr(std::is_same_v<T, std::string>)
                {
                    return substituteStringParam(arg, params);
                }
                else if constexpr(std::is_same_v<T, std::shared_ptr<rocisa::Container>>)
                {
                    auto cloned = arg->clone();
                    if(auto reg = std::dynamic_pointer_cast<RegisterContainer>(cloned))
                    {
                        substituteRegisterParam(*reg, params);
                        return std::static_pointer_cast<Container>(reg);
                    }
                    return cloned;
                }
                else
                {
                    return arg;
                }
            },
            input);
    }

    // Substitute all dst/src operands of a CommonInstruction
    static void substituteCommonInst(CommonInstruction& inst, const MacroParams& params)
    {
        if(inst.dst)
            if(auto reg = std::dynamic_pointer_cast<RegisterContainer>(inst.dst))
                substituteRegisterParam(*reg, params);

        if(inst.dst1)
            if(auto reg = std::dynamic_pointer_cast<RegisterContainer>(inst.dst1))
                substituteRegisterParam(*reg, params);

        for(auto& src : inst.srcs)
            src = substituteInput(src, params);

        inst.comment = substituteStringParam(inst.comment, params);
    }

    // Clone an instruction and substitute macro params in its operands
    static std::shared_ptr<Item> cloneAndSubstitute(std::shared_ptr<Instruction> inst,
                                                    const MacroParams&           params)
    {
        auto cloned = inst->clone();
        if(auto common = std::dynamic_pointer_cast<CommonInstruction>(cloned))
            substituteCommonInst(*common, params);
        return cloned;
    }

    static void expandMacroBody(std::vector<std::shared_ptr<Item>>& output,
                                std::vector<std::shared_ptr<Item>>& macroItems,
                                const MacroParams&                  params,
                                std::vector<bool>&                  branch)
    {
        for(auto& item : macroItems)
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                expandMacroBody(output, subModule->itemList, params, branch);
            }
            else if(auto valueIf = std::dynamic_pointer_cast<ValueIf>(item))
            {
                branch.push_back(branch.back()
                                 && evalMacroCondition(valueIf->value, params));
            }
            else if(auto valueElseIf = std::dynamic_pointer_cast<ValueElseIf>(item))
            {
                bool ifTaken = branch.back();
                branch.pop_back();
                branch.push_back(!ifTaken
                                 && evalMacroCondition(valueElseIf->value, params));
            }
            else if(auto valueEndif = std::dynamic_pointer_cast<ValueEndif>(item))
            {
                branch.pop_back();
            }
            else if(auto instruction = std::dynamic_pointer_cast<Instruction>(item))
            {
                if(branch.back())
                    output.push_back(cloneAndSubstitute(instruction, params));
            }
            else
            {
                assert(false && "macroToInstruction: unexpected item type in macro body");
            }
        }
    }

    template <typename T>
    static void macroToInstructionImpl(std::shared_ptr<T>& container,
                                       const MacroTable&   macros)
    {
        std::vector<std::shared_ptr<Item>> newItems;
        for(auto& item : container->itemList)
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                macroToInstructionImpl<Module>(subModule, macros);
                newItems.push_back(std::move(item));
            }
            else if(std::dynamic_pointer_cast<Macro>(item))
            {
                continue;
            }
            else if(auto macroInst = std::dynamic_pointer_cast<MacroInstruction>(item))
            {
                auto it = macros.find(macroInst->name);
                assert(it != macros.end()
                       && "macroToInstruction: MacroInstruction references undefined macro");
                auto& macro      = std::get<0>(it->second);
                auto& defaults   = std::get<1>(it->second);
                auto  params     = std::make_shared<MacroParams>(*defaults);
                for(size_t i = 0; i < macroInst->args.size(); i++)
                    std::get<1>(params->at(i)) = InstructionInputToString(macroInst->args[i]);

                std::vector<bool> branch = {true};
                expandMacroBody(newItems, macro->itemList, *params, branch);
            }
            else
            {
                newItems.push_back(std::move(item));
            }
        }
        container->itemList = std::move(newItems);
    }

    void macroToInstruction(std::shared_ptr<Module>& module)
    {
        MacroTable macros;
        collectMacros(module, macros);
        if(macros.empty())
            return;
        macroToInstructionImpl<Module>(module, macros);
    }
} // namespace rocisa
