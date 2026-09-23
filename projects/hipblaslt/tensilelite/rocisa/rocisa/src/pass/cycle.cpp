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
#include "instruction/instruction.hpp"
#include "instruction/mem.hpp"
#include "instruction/mfma.hpp"
#include "instruction/common.hpp"
#include "pass.hpp"

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <tuple>
#include <regex>
#include <algorithm>
#include <unordered_map>
#include <climits>

#include <origami/simulator/tensilelite/formocast_simulator.hpp>

using MacroArguments = std::vector<std::tuple<std::string, std::string>>;
using MacroEntity = std::tuple<std::shared_ptr<rocisa::Macro>, std::shared_ptr<MacroArguments>>;
using MacroTable = std::unordered_map<std::string, MacroEntity>;

namespace rocisa
{
    // Helper function to handle macro and macro if/else, should be remove once macro is forbidden
    void _extractMacro(MacroTable& macros, std::shared_ptr<Macro> macro)
    {
        auto args = std::make_shared<MacroArguments>();
        auto entity = std::make_tuple(macro, args);
        macros[macro->macro->name] = entity;
        // parse argument and default values
        std::regex argPattern("([^=]+)=?(\\w+)?");
        std::smatch match;
        for(auto& arg : macro->macro->args)
        {
            std::regex_match(std::get<std::string>(arg), match, argPattern);
            args->push_back(std::make_tuple(match[1].str(), match.size() > 2 ? match[2].str() : std::string()));
        }
    }

    void _getMacros(std::shared_ptr<Module> module, MacroTable& macros)
    {
        for(auto& item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _getMacros(subModule, macros);
            }
            else if(auto macro = std::dynamic_pointer_cast<Macro>(item))
            {
                if(macros.find(macro->macro->name) == macros.end())
                {
                    _extractMacro(macros, macro);
                }
            }
        }
    }

    bool _evalMacroCondition(std::string value, std::shared_ptr<MacroArguments> args)
    {
        std::regex tokenPattern("\\\\([^=\\s]+)|\\w+|==|!=|&&");
        std::sregex_iterator it(value.begin(), value.end(), tokenPattern);
        std::sregex_iterator end;
        std::string lhs, rhs, op, val;
        int i = 0;
        bool result;
        std::vector<int> results;
        while(it != end)
        {
            std::smatch match = *it;
            if(!match.str(0).empty())
            {
                val = match.str(0);
                if(val[0] == '\\')
                {
                    auto var = match.str(1);
                    auto find_it = std::find_if(args->begin(), args->end(), [&var](auto& arg){ return std::get<0>(arg) == var; });
                    if(find_it != args->end())
                    {
                        val = std::get<1>(*find_it);
                    }
                    else
                    {
                        throw std::runtime_error("unknown macro argument");
                    }
                }
                if(i == 0)
                {
                    lhs = val;
                }
                else if(i == 1)
                {
                    op = val;
                }
                else if(i == 2)
                {
                    rhs = val;
                    if(op == "==")
                    {
                        result = lhs == rhs;
                    }
                    else if(op == "!=")
                    {
                        result = lhs != rhs;
                    }
                    else
                    {
                        throw std::runtime_error("unknown macro condition");
                    }
                    results.push_back(result ? 1 : 0);
                }
                else if(i == 4)
                {
                    if(val == "&&")
                    {
                        results.push_back(2);
                    }
                    else
                    {
                        throw std::runtime_error("unknown macro condition");
                    }
                }
                i = (i + 1) % 4;
            }
            ++it;
        }
        result = results.front();
        i = 1;
        while(i < results.size())
        {
            auto r = results[i];
            if(r == 2)
            {
                result = result & results[i+1];
                i += 2;
            }
            else
            {
                i++;
            }
        }
        return result;
    }

    void _expandMacroAndPopInst(std::vector<std::shared_ptr<Item>>& moduleInst, std::vector<std::shared_ptr<Item>>& macroItems, std::shared_ptr<MacroArguments> args, std::vector<bool>& branch)
    {
        for(auto& item : macroItems)
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _expandMacroAndPopInst(moduleInst, subModule->itemList, args, branch);
            }
            else if(auto valueIf = std::dynamic_pointer_cast<ValueIf>(item))
            {
                branch.push_back(branch.back() && _evalMacroCondition(valueIf->value, args));
            }
            else if(auto valueElseIf = std::dynamic_pointer_cast<ValueElseIf>(item))
            {
                bool ifTaken = branch.back();
                branch.pop_back();
                branch.push_back(!ifTaken && _evalMacroCondition(valueElseIf->value, args));
            }
            else if(auto valueEndif = std::dynamic_pointer_cast<ValueEndif>(item))
            {
              branch.pop_back();
            }
            else if(auto instruction = std::dynamic_pointer_cast<Instruction>(item))
            {
                if(branch.back())
                {
                    moduleInst.push_back(instruction);
                }
            }
        }
    }

    void _expandMacroAndPopInst(std::vector<std::shared_ptr<Item>>& moduleInst, std::vector<std::shared_ptr<Item>>& macroItems, std::shared_ptr<MacroArguments> args)
    {
        std::vector<bool> branch = {true};
        _expandMacroAndPopInst(moduleInst, macroItems, args, branch);
    }

    void _popInst(std::shared_ptr<Module> mod, std::vector<std::shared_ptr<Item>>& moduleInst, MacroTable& macros)
    {
        for(auto& item : mod->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _popInst(subModule, moduleInst, macros);
            }
            else if(auto macro = std::dynamic_pointer_cast<Macro>(item))
            {
                _extractMacro(macros, macro);
            }
            else if(auto macroInst = std::dynamic_pointer_cast<MacroInstruction>(item))
            {
                auto entity = macros[macroInst->name];
                auto macro = std::get<0>(entity);
                auto defaultArgs = std::get<1>(entity);
                auto args = std::make_shared<MacroArguments>();
                for(auto& arg : *defaultArgs)
                {
                    args->push_back(arg);
                }
                for(int i = 0; i < macroInst->args.size(); i++)
                {
                    std::get<1>(args->at(i)) = InstructionInputToString(macroInst->args[i]);
                }
                _expandMacroAndPopInst(moduleInst, macro->itemList, args);
            }
            else if(auto label = std::dynamic_pointer_cast<Label>(item))
            {
                moduleInst.push_back(label);
            }
            else if(auto instruction = std::dynamic_pointer_cast<Instruction>(item))
            {
                moduleInst.push_back(instruction);
            }
        }
    }

    // Helper function to populate instructions from a module
    void _popInst(std::shared_ptr<Module> mod, std::vector<std::shared_ptr<Item>>& moduleInst)
    {
        for(auto& item : mod->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _popInst(subModule, moduleInst);
            }
            else if(auto instruction = std::dynamic_pointer_cast<Instruction>(item))
            {
                moduleInst.push_back(instruction);
            }
        }
    }

    // Helper function to parse immediate value from string
    int64_t parseImmediate(const std::string& str)
    {
        if(str.empty()) return 0;
        
        std::string trimmed = str;
        // Remove whitespace
        trimmed.erase(std::remove_if(trimmed.begin(), trimmed.end(), ::isspace), trimmed.end());
        
        if(trimmed.empty()) return 0;
        
        // Handle hex
        if(trimmed.find("0x") != std::string::npos || trimmed.find("0X") != std::string::npos)
        {
            return std::stoll(trimmed, nullptr, 16);
        }
        // Handle negative
        if(trimmed[0] == '-')
        {
            return -std::stoll(trimmed.substr(1));
        }
        
        try {
            return std::stoll(trimmed);
        } catch(...) {
            return 0;
        }
    }
    
    // Helper function to extract register name from operand string
    std::string extractRegName(const std::string& operand)
    {
        std::string trimmed = operand;
        trimmed.erase(std::remove_if(trimmed.begin(), trimmed.end(), ::isspace), trimmed.end());
        
        // Handle v[reg], s[reg] patterns
        if(trimmed.find('[') != std::string::npos)
        {
            size_t start = trimmed.find('[');
            size_t end = trimmed.find(']');
            if(start != std::string::npos && end != std::string::npos)
            {
                std::string regName = trimmed.substr(0, end + 1);
                
                // Remove "+0" offset if present (e.g., v[vgprLocalReadAddrB+0] -> v[vgprLocalReadAddrB])
                size_t plusZero = regName.find("+0");
                if(plusZero != std::string::npos && plusZero + 2 == regName.length() - 1)
                {
                    regName = regName.substr(0, plusZero) + "]";
                }
                
                return regName;
            }
        }
        
        return trimmed;
    }
    
    // Helper function to get value from register or immediate
    int64_t getOperandValue(const std::string& operand, 
                           const std::unordered_map<std::string, int64_t>& vgprState,
                           const std::unordered_map<std::string, int64_t>& sgprState)
    {
        std::string op = extractRegName(operand);
        
        // Check if it's a VGPR
        if(op[0] == 'v' && vgprState.find(op) != vgprState.end())
        {
            return vgprState.at(op);
        }
        
        // Check if it's an SGPR
        if(op[0] == 's' && sgprState.find(op) != sgprState.end())
        {
            return sgprState.at(op);
        }
        
        // Otherwise treat as immediate
        return parseImmediate(op);
    }
    
    // Parse instruction and extract operands
    struct ParsedInstruction {
        std::string opcode;
        std::string dst;
        std::vector<std::string> srcs;
        bool valid = false;
    };
    
    ParsedInstruction parseInstruction(const std::string& instStr)
    {
        ParsedInstruction result;
        
        // Find the opcode (first token)
        size_t firstSpace = instStr.find(' ');
        if(firstSpace == std::string::npos) return result;
        
        result.opcode = instStr.substr(0, firstSpace);
        
        // Extract operands part (after opcode, before comment)
        size_t commentPos = instStr.find("//");
        std::string operands = instStr.substr(firstSpace + 1);
        if(commentPos != std::string::npos)
        {
            operands = instStr.substr(firstSpace + 1, commentPos - firstSpace - 1);
        }
        
        // Split operands by comma
        std::vector<std::string> tokens;
        size_t pos = 0;
        while(pos < operands.length())
        {
            size_t comma = operands.find(',', pos);
            if(comma == std::string::npos)
            {
                tokens.push_back(operands.substr(pos));
                break;
            }
            tokens.push_back(operands.substr(pos, comma - pos));
            pos = comma + 1;
        }
        
        if(tokens.size() > 0)
        {
            result.dst = extractRegName(tokens[0]);
            for(size_t i = 1; i < tokens.size(); i++)
            {
                result.srcs.push_back(extractRegName(tokens[i]));
            }
            result.valid = true;
        }
        
        return result;
    }
    
    // Helper function to extract operand value from InstructionInput
    int64_t getInstructionInputValue(const InstructionInput& input,
                                    std::unordered_map<std::string, int64_t>& vgprState,
                                    const std::unordered_map<std::string, int64_t>& sgprState)
    {
        return std::visit([&](auto&& arg) -> int64_t {
            using T = std::decay_t<decltype(arg)>;
            if constexpr(std::is_same_v<T, std::shared_ptr<Container>>)
            {
                std::string regName = arg->toString();
                return getOperandValue(regName, vgprState, sgprState);
            }
            else if constexpr(std::is_same_v<T, int> || std::is_same_v<T, int64_t>)
            {
                return static_cast<int64_t>(arg);
            }
            else if constexpr(std::is_same_v<T, std::string>)
            {
                if(arg.compare(0, 2, "0x") == 0){
                    return std::stoll(arg, nullptr, 16);
                }
                throw std::runtime_error("unknown handled string: " + arg);
            }
            else
            {
                throw std::runtime_error("unhandled argument");
            }
        }, input);
    }

    // Simulate VALU instruction using dynamic casting (type-safe approach)
    void simulateInstructionTyped(std::shared_ptr<Instruction> instruction,
                                  std::unordered_map<std::string, int64_t>& vgprState,
                                  const std::unordered_map<std::string, int64_t>& sgprState)
    {
        // Try to cast to CommonInstruction first (most VALU instructions inherit from this)
        auto commonInst = std::dynamic_pointer_cast<CommonInstruction>(instruction);
        if(!commonInst || !commonInst->dst) return;

        std::string dstReg = commonInst->dst->toString();
        
        // v_add_co_u32: dst = src0 + src1 (with carry out, dst1 is vcc)
        if(auto vaddco = std::dynamic_pointer_cast<VAddCOU32>(instruction))
        {
            if(vaddco->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vaddco->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vaddco->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 + src1;
            }
        }
        // v_add_u32, v_add_i32
        else if(auto vaddu = std::dynamic_pointer_cast<VAddU32>(instruction))
        {
            if(vaddu->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vaddu->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vaddu->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 + src1;
            }
        }
        else if(auto vaddi = std::dynamic_pointer_cast<VAddI32>(instruction))
        {
            if(vaddi->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vaddi->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vaddi->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 + src1;
            }
        }
        // v_sub_u32, v_sub_i32
        else if(auto vsubu = std::dynamic_pointer_cast<VSubU32>(instruction))
        {
            if(vsubu->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vsubu->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vsubu->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 - src1;
            }
        }
        else if(auto vsubi = std::dynamic_pointer_cast<VSubI32>(instruction))
        {
            if(vsubi->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vsubi->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vsubi->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 - src1;
            }
        }
        // v_mul_lo_u32
        else if(auto vmulou = std::dynamic_pointer_cast<VMulLOU32>(instruction))
        {
            if(vmulou->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vmulou->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vmulou->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 * src1;
            }
        }
        // v_mul_hi_u32: dst = high 32 bits of (src0 * src1)
        else if(auto vmulhiu = std::dynamic_pointer_cast<VMulHIU32>(instruction))
        {
            if(vmulhiu->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vmulhiu->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vmulhiu->srcs[1], vgprState, sgprState);
                // Cast to uint64_t for unsigned multiplication, then take high 32 bits
                uint64_t result = static_cast<uint64_t>(static_cast<uint32_t>(src0)) *
                                  static_cast<uint64_t>(static_cast<uint32_t>(src1));
                vgprState[dstReg] = static_cast<int64_t>(result >> 32);
            }
        }
        // v_mul_hi_i32: dst = high 32 bits of (src0 * src1) signed
        else if(auto vmulhii = std::dynamic_pointer_cast<VMulHII32>(instruction))
        {
            if(vmulhii->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vmulhii->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vmulhii->srcs[1], vgprState, sgprState);
                // Cast to int64_t for signed multiplication, then take high 32 bits
                int64_t result = static_cast<int64_t>(static_cast<int32_t>(src0)) *
                                 static_cast<int64_t>(static_cast<int32_t>(src1));
                vgprState[dstReg] = result >> 32;
            }
        }
        // v_lshlrev_b32 or v_lshl_b32: logical shift left
        else if(auto vlshl = std::dynamic_pointer_cast<VLShiftLeftB32>(instruction))
        {
            if(vlshl->srcs.size() >= 2)
            {
                // VLShiftLeftB32 format: dst, shiftAmount, src
                int64_t shiftAmount = getInstructionInputValue(vlshl->srcs[0], vgprState, sgprState);
                int64_t src = getInstructionInputValue(vlshl->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src << shiftAmount;
            }
        }
        // v_lshrrev_b32 or v_lshr_b32: logical shift right (32-bit)
        else if(auto vlshr = std::dynamic_pointer_cast<VLShiftRightB32>(instruction))
        {
            if(vlshr->srcs.size() >= 2)
            {
                // VLShiftRightB32 format: dst, shiftAmount, src
                int64_t shiftAmount = getInstructionInputValue(vlshr->srcs[0], vgprState, sgprState);
                int64_t src = getInstructionInputValue(vlshr->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = (uint64_t)src >> (uint64_t)shiftAmount;
            }
        }
        // v_lshlrev_b64 or v_lshl_b64: logical shift left (64-bit)
        else if(auto vlshl64 = std::dynamic_pointer_cast<VLShiftLeftB64>(instruction))
        {
            if(vlshl64->srcs.size() >= 2)
            {
                // VLShiftLeftB64 format: dst, shiftAmount, src
                int64_t shiftAmount = getInstructionInputValue(vlshl64->srcs[0], vgprState, sgprState);
                int64_t src = getInstructionInputValue(vlshl64->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = static_cast<int64_t>(static_cast<uint64_t>(src) << shiftAmount);
            }
        }
        // v_lshrrev_b64 or v_lshr_b64: logical shift right (64-bit)
        else if(auto vlshr64 = std::dynamic_pointer_cast<VLShiftRightB64>(instruction))
        {
            if(vlshr64->srcs.size() >= 2)
            {
                // VLShiftRightB64 format: dst, shiftAmount, src
                int64_t shiftAmount = getInstructionInputValue(vlshr64->srcs[0], vgprState, sgprState);
                int64_t src = getInstructionInputValue(vlshr64->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = static_cast<int64_t>(static_cast<uint64_t>(src) >> shiftAmount);
            }
        }
        // v_and_b32
        else if(auto vand = std::dynamic_pointer_cast<VAndB32>(instruction))
        {
            if(vand->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vand->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vand->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 & src1;
            }
        }
        // v_or_b32
        else if(auto vor = std::dynamic_pointer_cast<VOrB32>(instruction))
        {
            if(vor->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vor->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vor->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 | src1;
            }
        }
        // v_xor_b32
        else if(auto vxor = std::dynamic_pointer_cast<VXorB32>(instruction))
        {
            if(vxor->srcs.size() >= 2)
            {
                int64_t src0 = getInstructionInputValue(vxor->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vxor->srcs[1], vgprState, sgprState);
                vgprState[dstReg] = src0 ^ src1;
            }
        }
        // v_mad_u32_u24: dst = src0 * src1 + src2
        else if(auto vmadu = std::dynamic_pointer_cast<VMadU32U24>(instruction))
        {
            if(vmadu->srcs.size() >= 3)
            {
                int64_t src0 = getInstructionInputValue(vmadu->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vmadu->srcs[1], vgprState, sgprState);
                int64_t src2 = getInstructionInputValue(vmadu->srcs[2], vgprState, sgprState);
                vgprState[dstReg] = src0 * src1 + src2;
            }
        }
        // v_mad_i32_i24: dst = src0 * src1 + src2
        else if(auto vmadi = std::dynamic_pointer_cast<VMadI32I24>(instruction))
        {
            if(vmadi->srcs.size() >= 3)
            {
                int64_t src0 = getInstructionInputValue(vmadi->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vmadi->srcs[1], vgprState, sgprState);
                int64_t src2 = getInstructionInputValue(vmadi->srcs[2], vgprState, sgprState);
                vgprState[dstReg] = src0 * src1 + src2;
            }
        }
        // v_mov_b32: dst = src0
        else if(auto vmov = std::dynamic_pointer_cast<VMovB32>(instruction))
        {
            if(vmov->srcs.size() >= 1)
            {
                int64_t src0 = getInstructionInputValue(vmov->srcs[0], vgprState, sgprState);
                vgprState[dstReg] = src0;
            }
        }
        // v_lshl_add_u32 (actual instruction, CommonInstruction): dst = (src0 << shiftAmount) + src1
        // srcs = {src0, shiftHex, src1} - note the order!
        else if(auto vlshladd = std::dynamic_pointer_cast<_VLShiftLeftAddU32>(instruction))
        {
            if(vlshladd->srcs.size() >= 3)
            {
                int64_t src0 = getInstructionInputValue(vlshladd->srcs[0], vgprState, sgprState);
                int64_t shiftAmount = getInstructionInputValue(vlshladd->srcs[1], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vlshladd->srcs[2], vgprState, sgprState);
                vgprState[dstReg] = (src0 << shiftAmount) + src1;
            }
        }
        // v_add_lshl_u32 (actual instruction, CommonInstruction): dst = (src0 + src1) << shiftAmount
        // srcs = {src0, src1, shiftHex} - standard order
        else if(auto vaddlshl = std::dynamic_pointer_cast<_VAddLShiftLeftU32>(instruction))
        {
            if(vaddlshl->srcs.size() >= 3)
            {
                int64_t src0 = getInstructionInputValue(vaddlshl->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vaddlshl->srcs[1], vgprState, sgprState);
                int64_t shiftAmount = getInstructionInputValue(vaddlshl->srcs[2], vgprState, sgprState);
                vgprState[dstReg] = (src0 + src1) << shiftAmount;
            }
        }
        // v_lshl_add_u32 (CompositeInstruction wrapper): dst = (src0 << shiftAmount) + src1
        // srcs = {src0, src1, shiftHex} based on constructor
        else if(auto vlshladd = std::dynamic_pointer_cast<VLShiftLeftAddU32>(instruction))
        {
            if(vlshladd->srcs.size() >= 3)
            {
                int64_t src0 = getInstructionInputValue(vlshladd->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vlshladd->srcs[1], vgprState, sgprState);
                int64_t shiftAmount = getInstructionInputValue(vlshladd->srcs[2], vgprState, sgprState);
                vgprState[dstReg] = (src0 << shiftAmount) + src1;
            }
        }
        // v_add_lshl_u32 (CompositeInstruction wrapper): dst = (src0 + src1) << shiftAmount
        // srcs = {src0, src1, shiftHex} based on constructor
        else if(auto vaddlshl = std::dynamic_pointer_cast<VAddLShiftLeftU32>(instruction))
        {
            if(vaddlshl->srcs.size() >= 3)
            {
                int64_t src0 = getInstructionInputValue(vaddlshl->srcs[0], vgprState, sgprState);
                int64_t src1 = getInstructionInputValue(vaddlshl->srcs[1], vgprState, sgprState);
                int64_t shiftAmount = getInstructionInputValue(vaddlshl->srcs[2], vgprState, sgprState);
                vgprState[dstReg] = (src0 + src1) << shiftAmount;
            }
        }
        else
        {
            throw std::runtime_error("Unsupported instruction: " + instruction->toString());
        }
    }

    // Helper function to analyze bank conflicts in local read address calculation
    origami::Formocast::BankConflictResult _countLocalReadBankConflicts(origami::Formocast& formocast, std::shared_ptr<Module> module, int numWaves, MacroTable& macros, int LocalReadBytesA, int LocalReadBytesB)
    {
        std::vector<std::shared_ptr<Item>> moduleInst;
        _popInst(module, moduleInst, macros);
        
        // Track register values for simulation (per thread)
        // For simplicity, simulate for first few threads (0-7)
        const int WAVEFRONT_SIZE = 64;
        const int NUM_THREADS_TO_SIMULATE = 64;
        
        // vgpr[thread_id][register_name] = value
        std::vector<std::unordered_map<std::string, int64_t>> vgprState(NUM_THREADS_TO_SIMULATE);
        std::unordered_map<std::string, int64_t> sgprState; // SGPRs are shared across all threads
        
        // Initialize thread IDs (v[vgprSerial])
        std::string vgprSerial;
        std::string vgprLocalReadAddrA;
        std::string vgprLocalReadAddrB;
        
        int instCount = 0;
        
        // Print and simulate instructions
        for(auto& item : moduleInst)
        {
            if(auto instruction = std::dynamic_pointer_cast<Instruction>(item))
            {
                instCount++;
                std::string instStr = instruction->toString();
                
                // Try to identify key registers from comments
                if(instStr.find("vgprSerial") != std::string::npos)
                {
                    // Parse the instruction to get destination register
                    ParsedInstruction parsed = parseInstruction(instStr);
                    if(parsed.valid && !parsed.dst.empty())
                    {
                        vgprSerial = parsed.srcs[1];
                        
                        // Initialize thread IDs
                        for(int tid = 0; tid < NUM_THREADS_TO_SIMULATE; tid++)
                        {
                            vgprState[tid][vgprSerial] = tid;
                        }
                    }
                }
                
                if(instStr.find("vgprLocalReadAddrA") != std::string::npos)
                {
                    ParsedInstruction parsed = parseInstruction(instStr);
                    if(parsed.valid && !parsed.dst.empty())
                    {
                        vgprLocalReadAddrA = parsed.dst;
                    }
                }
                
                if(instStr.find("vgprLocalReadAddrB") != std::string::npos)
                {
                    ParsedInstruction parsed = parseInstruction(instStr);
                    if(parsed.valid && !parsed.dst.empty())
                    {
                        vgprLocalReadAddrB = parsed.dst;
                    }
                }
                
                // Parse and simulate VALU instructions for each thread
                // Check if instruction starts with "v_" (VALU instructions)
                if(instStr.size() >= 2 && instStr.compare(0, 2, "v_") == 0)
                {
                    // Simulate for each thread using type-safe dynamic casting
                    for(int tid = 0; tid < NUM_THREADS_TO_SIMULATE; tid++)
                    {
                        simulateInstructionTyped(instruction, vgprState[tid], sgprState);
                    }
                }
                
                // Handle SGPR instructions (s_mov, s_mul, etc.) - shared across all threads
                if(instStr.find("s_mov") != std::string::npos)
                {
                    ParsedInstruction parsed = parseInstruction(instStr);
                    if(parsed.valid && parsed.srcs.size() >= 1)
                    {
                        sgprState[parsed.dst] = parseImmediate(parsed.srcs[0]);
                    }
                }
            }
        }

        // Analyze bank conflicts and return results
        return formocast.analyzeBankConflictsFromVGPR(vgprState, vgprLocalReadAddrA, vgprLocalReadAddrB, LocalReadBytesA, LocalReadBytesB);
    }

    // Helper function to count cycles
    int _countCycles(origami::Formocast& formocast, std::shared_ptr<Module> item, int numWaves, MacroTable& macros, std::pair<double, double> bankConflicts)
    {
        std::vector<std::shared_ptr<Item>> moduleInst;
        _popInst(item, moduleInst, macros);

        int cycles = 0;
        int hwMFMA = -99;
        int jumpOverhead = 6;
        int previousLW = 0;
        std::queue<int> hwLRFIFO;
        std::queue<int> lgkmLRFIFO;
        std::deque<int> hwGRFIFO;
        bool isEndOfLoop  = false;
        int numPreviousLRs = 0;
        bool isPreviousMFMA = false;

        // Find vgprLocalReadAddrA and vgprLocalReadAddrB names
        std::string vgprLocalReadAddrA = "vgprLocalReadAddrA";
        std::string vgprLocalReadAddrB = "vgprLocalReadAddrB";

        bool skip = false;
        for(auto& item : moduleInst)
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                throw std::runtime_error("Module should be instructions here.");
            }
            else if(auto subModule = std::dynamic_pointer_cast<MacroInstruction>(item))
            {
                throw std::runtime_error("MacroInst should be instructions here.");
            }
            else if(auto subModule = std::dynamic_pointer_cast<Macro>(item))
            {
                throw std::runtime_error("Macro should be instructions here.");
            }
            else if(auto label = std::dynamic_pointer_cast<Label>(item))
            {
                auto labelStr = std::visit(
                    [](auto&& arg) -> std::string {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr(std::is_same_v<T, int>)
                        {
                            return std::to_string(arg);
                        }
                        else if constexpr(std::is_same_v<T, std::string>)
                        {
                            return arg;
                        }
                    },
                    label->label);
                std::regex simdBranchesPattern(".*Loop(Skip)?BeginL(_\\d+)?.*");
                std::smatch match;
                if(std::regex_match(labelStr, match, simdBranchesPattern))
                {
                    skip = match[1].matched || (match[2].matched && match.str(2) != "_0");
                }
                else
                {
                    skip = false;
                }
            }
            else if(skip)
            {
                continue;
            }
            else if(auto mfmaInst = std::dynamic_pointer_cast<MFMAInstruction>(item))
            {
                auto mfmaLatency = mfmaInst->getIssueLatency();
                if(cycles - hwMFMA >= (mfmaLatency - 1))
                {
                    cycles += 1;
                }
                else
                {
                    cycles = hwMFMA + mfmaLatency;
                }
                hwMFMA = cycles;
            }
            else if(auto dsReadInst = std::dynamic_pointer_cast<DSLoadInstruction>(item))
            {
                int bpr = 2;
                if (auto lr128 = std::dynamic_pointer_cast<DSLoadB128>(dsReadInst)) {
                    bpr = 16;
                } else if (auto lr64 = std::dynamic_pointer_cast<DSLoadB64>(dsReadInst)) {
                    bpr = 8;
                } else if (auto lr32 = std::dynamic_pointer_cast<DSLoadB32>(dsReadInst)) {
                    bpr = 4;
                }

                // Determine which bank conflict value to use based on source register
                double bankConflict = bankConflicts.first; // default to A
                if(dsReadInst->srcs)
                {
                    std::string srcStr = dsReadInst->srcs->toString();
                    // Check if source contains vgprLocalReadAddrB
                    if(!vgprLocalReadAddrB.empty() && srcStr.find(vgprLocalReadAddrB) != std::string::npos)
                    {
                        bankConflict = bankConflicts.second; // Use B's bank conflict
                    }
                    else if(!vgprLocalReadAddrA.empty() && srcStr.find(vgprLocalReadAddrA) != std::string::npos)
                    {
                        bankConflict = bankConflicts.first; // Use A's bank conflict
                    }
                    else
                    {
                        bankConflict = 1.0;
                    }
                }
                int stallcycle = formocast.getLocalReadQueueFullStallCycles(cycles, hwLRFIFO, bpr, numWaves, true, bankConflict);
                if (stallcycle == cycles) {
                    // no stall
                    //heck LR fifo
                    auto currCycles = cycles + dsReadInst->issueLatency();
                    if(numPreviousLRs > 0 && bpr >= 4) { // two wave share same lds interface (gfx9)
                        currCycles += dsReadInst->issueLatency();
                    }
                    else if(isPreviousMFMA && rocIsa::getInstance().getKernel().isaVersion == std::array<int, 3>{9, 5, 0}) {
                        // gfx950 limitation: no DSLoad instruction can be issued in next 4 cycles after MFMA instruction
                        currCycles += 1;
                    }
                    cycles = currCycles;
                } else {
                    cycles = stallcycle;
                }
                formocast.pushLocalReadWrite(cycles, lgkmLRFIFO, bpr, bankConflict, true, numPreviousLRs);
            }
            else if(auto rwInst = std::dynamic_pointer_cast<ReadWriteInstruction>(item))
            {
                if(auto grInst = std::dynamic_pointer_cast<MUBUFReadInstruction>(item))
                {
                    auto currCycles = cycles + grInst->issueLatency();
                    int bpr = 4;
                     if (auto gr128 = std::dynamic_pointer_cast<BufferLoadB128>(grInst)) {
                        bpr = 16;
                    } else if (auto gr64 = std::dynamic_pointer_cast<BufferLoadB64>(grInst)) {
                        bpr = 8;
                    }

                    bool hasSgprOffset = false;
                    auto soffsetStr = InstructionInputToString(grInst->soffset);
                    if (soffsetStr.find("s") != std::string::npos) {
                        hasSgprOffset = true;
                    }
                    cycles = formocast.getGlobalReadQueueFullStallCycles(currCycles, hwGRFIFO, bpr, numWaves, (rocIsa::getInstance().getKernel().isaVersion[0] == 9), hasSgprOffset);
                }
                if(auto wInst = std::dynamic_pointer_cast<DSStoreB128>(item))
                {
                    cycles = formocast.getLocalWriteQueueFullStallCycles(cycles, previousLW, wInst->issueLatency(), 16, numWaves);
                    previousLW = cycles;
                    formocast.pushLocalReadWrite(cycles, lgkmLRFIFO, 16, 1.0, false, 0);
                }
                else if(auto wInst = std::dynamic_pointer_cast<DSStoreB64>(item))
                {
                    cycles = formocast.getLocalWriteQueueFullStallCycles(cycles, previousLW, wInst->issueLatency(), 8, numWaves);
                    previousLW = cycles;
                    formocast.pushLocalReadWrite(cycles, lgkmLRFIFO, 8, 1.0, false, 0);
                }
                else if(auto wInst = std::dynamic_pointer_cast<DSStoreB32>(item))
                {
                    cycles = formocast.getLocalWriteQueueFullStallCycles(cycles, previousLW, wInst->issueLatency(), 4, numWaves);
                    previousLW = cycles;
                    formocast.pushLocalReadWrite(cycles, lgkmLRFIFO, 4, 1.0, false, 0);
                }
                else if(auto wInst = std::dynamic_pointer_cast<DSStoreB16>(item))
                {
                    cycles = formocast.getLocalWriteQueueFullStallCycles(cycles, previousLW, wInst->issueLatency(), 2, numWaves);
                    previousLW = cycles;
                    formocast.pushLocalReadWrite(cycles, lgkmLRFIFO, 2, 1.0, false, 0);
                }
                else if(auto wInst = std::dynamic_pointer_cast<DSStoreB8>(item))
                {
                    cycles = formocast.getLocalWriteQueueFullStallCycles(cycles, previousLW, wInst->issueLatency(), 1, numWaves);
                    previousLW = cycles;
                    formocast.pushLocalReadWrite(cycles, lgkmLRFIFO, 1, 1.0, false, 0);
                }
                else
                {
                    cycles += rwInst->issueLatency();
                }
            }
            else if(auto waitInst = std::dynamic_pointer_cast<_SWaitCnt>(item))
            {
                auto numLR = waitInst->getParams();
                cycles = formocast.getLocalReadCompletionCycle(cycles + 1, lgkmLRFIFO, std::stoi(InstructionInputToString(numLR[0])));
            }
            else if(auto branchInst = std::dynamic_pointer_cast<BranchInstruction>(item))
            {
                cycles = std::max(cycles + jumpOverhead, hwMFMA + 4);
                // End of loop
                auto pos = branchInst->labelName.find("label_LoopBeginL");
                if(pos != std::string::npos && pos == 0) //branchInst->labelName == "label_LoopBeginL")
                {
                    isEndOfLoop = true;
                    break;
                }
            }
            else if(auto instruction = std::dynamic_pointer_cast<SBarrier>(item))
            {
                cycles += 2;
            }
            else if(auto instruction = std::dynamic_pointer_cast<Instruction>(item))
            {
                cycles += 1;
            }
            // if(auto instruction = std::dynamic_pointer_cast<Instruction>(item))
            // {
            //    instruction->comment = instruction->comment + " <This is " + std::to_string(cycles) + "-cycle>"; // for debug
            // }

            // Set Flags
            if(auto mfmaInst = std::dynamic_pointer_cast<MFMAInstruction>(item))
            {
                isPreviousMFMA = true;
                numPreviousLRs = 0;
            }
            else if(auto lrInst = std::dynamic_pointer_cast<DSLoadInstruction>(item))
            {
                numPreviousLRs++;
                isPreviousMFMA = false;
            }
            else
            {
                numPreviousLRs = 0;
                isPreviousMFMA = false;
            }
        }
        if(!isEndOfLoop)
        {
            // Loop end without label_LoopBeginL label.
            // Add jump overhead here
            cycles += jumpOverhead + 1;
        }
        if (rocIsa::getInstance().getKernel().isaVersion[0] == 9) {
            cycles = cycles * 4;
        }
        return cycles;
    }
    int _countCycles(origami::Formocast& formocast, std::shared_ptr<Module> item, int numWaves, std::pair<double, double> bankConflicts)
    {
        MacroTable macros;
        return _countCycles(formocast, item, numWaves, macros, bankConflicts);
    }

    // Helper function to recursively find and analyze Local Read Addresses module
    std::pair<double, double> _findAndAnalyzeLocalReadAddresses(origami::Formocast& formocast, std::shared_ptr<Module> module, int numWaves, MacroTable& macros, int LocalReadBytesA, int LocalReadBytesB, int depth = 0)
    {
        std::string indent(depth * 2, ' ');
        
        for(auto& item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                // std::cout << indent << "Found subModule: " << subModule->name << std::endl;
                
                if(subModule->name == "Local Read Addresses")
                {
                    auto result = _countLocalReadBankConflicts(formocast, subModule, numWaves, macros, LocalReadBytesA, LocalReadBytesB);
                    return std::make_pair(result.ratioA, result.ratioB);
                }
                
                // Recursively search in nested submodules
                auto result = _findAndAnalyzeLocalReadAddresses(formocast, subModule, numWaves, macros, LocalReadBytesA, LocalReadBytesB, depth + 1);
                if(result.first > 0.0 || result.second > 0.0)
                {
                    return result;
                }
            }
        }
        
        return std::make_pair(0.0, 0.0);
    }

    // Function to find and analyze Local Read Addresses module for bank conflicts
    std::pair<double, double> analyzeBankConflicts(origami::Formocast& formocast, std::shared_ptr<Module> module, int numWaves, int LocalReadBytesA, int LocalReadBytesB)
    {
        MacroTable macros;
        _getMacros(module, macros);

        auto result = _findAndAnalyzeLocalReadAddresses(formocast, module, numWaves, macros, LocalReadBytesA, LocalReadBytesB);
        
        // Check if "Local Read Addresses" module was found
        if(result.first == 0.0 && result.second == 0.0)
        {
            throw std::runtime_error("Error: \"Local Read Addresses\" module not found in kernel");
        }
        
        return result;
    }

    // Function to calculate math clocks in an unrolled loop
    int _calculateMathClocksInUnrolledLoop(origami::Formocast& formocast, std::shared_ptr<Module> module, int numWaves, std::pair<double, double> bankConflicts)
    {
        // Kernel: openLoop->loopBody->noLoadLoop
        int  cycles     = -1;
        bool isOpenLoop = false;
        MacroTable macros;
        _getMacros(module, macros);

        for(auto& item : module->items())
        {
            // Find loopBody
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                if(subModule->name == "loopBody")
                {
                    cycles = _countCycles(formocast, subModule, numWaves, macros, bankConflicts);
                    return cycles;
                }
            }
        }
        return -1;
    }

    // Helper function to recursively scan and print all modules and submodules
    void _scanAndPrintModules(std::shared_ptr<Module> module, int depth = 0)
    {
        // Recursively scan all submodules
        for(auto& item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                _scanAndPrintModules(subModule, depth + 1);
            }
        }
    }

    // Helper function to find a module by name recursively
    std::shared_ptr<Module> _findModuleByName(std::shared_ptr<Module> module, const std::string& targetName)
    {
        if(module->name == targetName)
        {
            return module;
        }
        
        // Recursively search in submodules
        for(auto& item : module->items())
        {
            if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                auto found = _findModuleByName(subModule, targetName);
                if(found)
                {
                    return found;
                }
            }
        }
        
        return nullptr;
    }

    // Helper function to extract byte size from instruction string
    int _extractByteSizeFromInstruction(const std::string& instStr)
    {
        // Look for ds_read_bXXX or similar patterns
        if(instStr.find("_b128") != std::string::npos || instStr.find("_B128") != std::string::npos)
        {
            return 16; // 128 bits = 16 bytes
        }
        else if(instStr.find("_b96") != std::string::npos || instStr.find("_B96") != std::string::npos)
        {
            return 12; // 96 bits = 12 bytes
        }
        else if(instStr.find("_b64") != std::string::npos || instStr.find("_B64") != std::string::npos)
        {
            return 8; // 64 bits = 8 bytes
        }
        else if(instStr.find("_b32") != std::string::npos || instStr.find("_B32") != std::string::npos)
        {
            return 4; // 32 bits = 4 bytes
        }
        else if(instStr.find("_b16") != std::string::npos || instStr.find("_B16") != std::string::npos)
        {
            return 2; // 16 bits = 2 bytes
        }
        else if(instStr.find("_b8") != std::string::npos || instStr.find("_B8") != std::string::npos)
        {
            return 1; // 8 bits = 1 byte
        }
        
        return 16; // Default to 16 bytes if not found
    }

    // Helper function to get byte size from first instruction in a module
    int _getByteSizeFromModule(std::shared_ptr<Module> module)
    {
        if(!module)
        {
            return 16; // Default
        }
        
        // Get first instruction
        for(auto& item : module->items())
        {
            if(auto instruction = std::dynamic_pointer_cast<Instruction>(item))
            {
                std::string instStr = instruction->toString();
                int byteSize = _extractByteSizeFromInstruction(instStr);
                return byteSize;
            }
        }
        
        return 16; // Default if no instruction found
    }

    // Helper function to calculate local read bytes for A and B tensors
    std::pair<int, int> _calculateLocalReadBytes(std::shared_ptr<Module> module, int numWaves)
    {
        // Find LocalReadDoA module
        auto moduleA = _findModuleByName(module, "LocalReadDoA_I0");
        int LocalReadBytesA = 16; // Default
        if(moduleA)
        {
            LocalReadBytesA = _getByteSizeFromModule(moduleA);
        }
        
        // Find LocalReadDoB module
        auto moduleB = _findModuleByName(module, "LocalReadDoB_I0");
        int LocalReadBytesB = 16; // Default
        if(moduleB)
        {
            LocalReadBytesB = _getByteSizeFromModule(moduleB);
        }
        
        return std::make_pair(LocalReadBytesA, LocalReadBytesB);
    }

    // Main function to get cycles
    int getCycles(std::shared_ptr<Module> module, int numWaves)
    {
        origami::Formocast formocast;
        auto isaVersion   = rocIsa::getInstance().getKernel().isaVersion;
        if (isaVersion == std::array<int, 3>{9, 5, 0}) {
            formocast.setHardware(origami::hardware_t::architecture_t::gfx950);
        }
        else if (isaVersion == std::array<int, 3>{9, 4, 2}) {
            formocast.setHardware(origami::hardware_t::architecture_t::gfx942);
        }
        else if (isaVersion == std::array<int, 3>{12, 0, 1}) {
            formocast.setHardware(origami::hardware_t::architecture_t::gfx1201);
        }
        else {
            // not supported
	        return 0;
        }
        // Calculate local read bytes
        auto localReadBytes = _calculateLocalReadBytes(module, numWaves);
        int LocalReadBytesA = localReadBytes.first;
        int LocalReadBytesB = localReadBytes.second;

        // Analyze bank conflicts first
        auto bankConflicts = analyzeBankConflicts(formocast, module, numWaves, LocalReadBytesA, LocalReadBytesB);
        
        // Calculate cycles
        auto cycles = _calculateMathClocksInUnrolledLoop(formocast, module, numWaves, bankConflicts);
        
        return cycles;
    }
} // namespace rocisa
