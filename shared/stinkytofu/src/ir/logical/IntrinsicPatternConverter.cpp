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

#include "stinkytofu/ir/logical/IntrinsicPatternConverter.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "stinkytofu/core/Function.hpp"

namespace stinkytofu {

// Helper function to inline function calls in an intrinsic pattern
static Pattern inlineFunctionCalls(const Pattern& pattern,
                                   const std::unordered_map<std::string, Pattern>& intrinsicMap) {
    Pattern inlined = pattern;
    inlined.body.clear();  // We'll rebuild the body with inlined instructions

    for (const auto& inst : pattern.body) {
        if (inst.isFunctionCall) {
            // Look up the called function
            auto it = intrinsicMap.find(inst.operation);
            if (it == intrinsicMap.end()) {
                std::cerr << "Error: Unknown function '" << inst.operation
                          << "' called in intrinsic '" << pattern.name << "'\n";
                // Keep the function call as-is (will fail later)
                inlined.body.push_back(inst);
                continue;
            }

            const Pattern& callee = it->second;

            // Build argument mapping: argName -> actualValue
            std::unordered_map<std::string, std::string> argMapping;
            for (const auto& [argName, argValue] : inst.funcCallArgs) {
                argMapping[argName] = argValue;
            }

            // Inline the callee's body with register remapping
            for (const auto& calleeInst : callee.body) {
                // Skip if this is a function call (recursive inlining not yet supported)
                if (calleeInst.isFunctionCall) {
                    std::cerr << "Warning: Recursive function calls not yet supported in '"
                              << pattern.name << "'\n";
                    inlined.body.push_back(calleeInst);
                    continue;
                }

                IntrinsicInstruction inlinedInst = calleeInst;

                // Remap destination register
                if (argMapping.count(inlinedInst.destReg)) {
                    inlinedInst.destReg = argMapping[inlinedInst.destReg];
                }

                // Remap register operands (literals are not remapped)
                for (auto& operand : inlinedInst.operands) {
                    if (operand.type == IntrinsicOperand::Register &&
                        argMapping.count(operand.registerName)) {
                        operand.registerName = argMapping[operand.registerName];
                    }
                }

                inlined.body.push_back(inlinedInst);
            }
        } else {
            // Regular instruction, keep as-is
            inlined.body.push_back(inst);
        }
    }

    return inlined;
}

IntrinsicIRModule IntrinsicPatternConverter::patternToIR(const Pattern& pattern) {
    // Validate input
    assert(pattern.type == PatternType::Intrinsic && "Pattern is not an intrinsic");

    IntrinsicIRModule irModule(pattern.name);
    irModule.arguments = pattern.arguments;
    irModule.comment = pattern.comment;
    irModule.pythonBinding = pattern.pythonBinding;

    // Convert each text instruction to GenericIRInstruction (use createIR + custom deleter)
    for (const auto& inst : pattern.body) {
        auto* raw =
            IRBase::createIR<GenericIRInstruction>(inst.destReg, inst.operation, inst.operands);
        irModule.instructions.push_back(
            std::shared_ptr<LogicalInstruction>(raw, [](LogicalInstruction* p) {
                if (p) p->safeErase();
            }));
    }

    return irModule;
}

Pattern IntrinsicPatternConverter::irToPattern(const IntrinsicIRModule& irModule) {
    Pattern pattern;
    pattern.type = PatternType::Intrinsic;
    pattern.name = irModule.name;
    pattern.arguments = irModule.arguments;
    pattern.comment = irModule.comment;
    pattern.pythonBinding = irModule.pythonBinding;

    // Convert GenericIRInstructions back to text
    for (const auto& irInst : irModule.instructions) {
        // Cast to GenericIRInstruction
        auto genericInst = std::static_pointer_cast<GenericIRInstruction>(irInst);

        IntrinsicInstruction textInst;
        textInst.destReg = genericInst->destReg;
        textInst.operation = genericInst->operation;
        textInst.operands = genericInst->operands;

        pattern.body.push_back(textInst);
    }

    return pattern;
}

std::vector<IntrinsicIRModule> IntrinsicPatternConverter::patternsToIR(
    const std::vector<Pattern>& patterns) {
    // Build intrinsic lookup map
    std::unordered_map<std::string, Pattern> intrinsicMap;
    for (const auto& pattern : patterns) {
        if (pattern.type == PatternType::Intrinsic) {
            intrinsicMap[pattern.name] = pattern;
        }
    }

    // Process each intrinsic and inline function calls
    std::vector<IntrinsicIRModule> irModules;
    for (const auto& pattern : patterns) {
        if (pattern.type == PatternType::Intrinsic) {
            // Inline any function calls
            Pattern inlined = inlineFunctionCalls(pattern, intrinsicMap);

            // Convert to IR
            irModules.push_back(patternToIR(inlined));
        }
    }
    return irModules;
}

std::vector<Pattern> IntrinsicPatternConverter::irToPatterns(
    const std::vector<IntrinsicIRModule>& irModules) {
    std::vector<Pattern> patterns;
    for (const auto& irModule : irModules) {
        patterns.push_back(irToPattern(irModule));
    }
    return patterns;
}

StinkyRegister IntrinsicPatternConverter::parseOperand(const std::string& operand) {
    // TODO: Implement full register/literal parsing
    // For now, return a placeholder vector register
    StinkyRegister reg(RegType::V, 0, 1);
    return reg;
}

}  // namespace stinkytofu
