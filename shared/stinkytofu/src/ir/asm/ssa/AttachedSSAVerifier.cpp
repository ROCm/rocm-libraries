/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "stinkytofu/ir/asm/ssa/AttachedSSAVerifier.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

std::string blockName(const BasicBlock& block) {
    if (!block.getLabel().empty()) return "^" + block.getLabel();
    return "^<unnamed>";
}

std::string valueName(const StinkySSAValue* value) {
    if (value == nullptr) return "%null";
    return "%" + std::to_string(value->valueId());
}

bool isPredecessor(const BasicBlock& block, const BasicBlock* pred) {
    const auto& preds = block.getPredecessors();
    return std::find(preds.begin(), preds.end(), pred) != preds.end();
}

}  // namespace

std::string AttachedSSAVerificationResult::toString() const {
    std::ostringstream out;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) out << '\n';
        out << errors[i];
    }
    return out.str();
}

AttachedSSAVerificationResult verifyAttachedSSA(const Function& function) {
    AttachedSSAVerificationResult result;
    auto error = [&](std::string message) { result.errors.push_back(std::move(message)); };

    std::unordered_set<const StinkyOpOperand*> observedUses;
    // Insertion-ordered mirror of observedUses so diagnostics do not depend on
    // the hash order of pointer keys.
    std::vector<const StinkyOpOperand*> orderedUses;
    auto observeUse = [&](const StinkyOpOperand* use) {
        if (observedUses.insert(use).second) orderedUses.push_back(use);
    };

    const std::string& funcName = function.getName();
    unsigned instIndex = 0;
    for (const BasicBlock& block : function) {
        for (size_t argIndex = 0; argIndex < block.ssaArguments().size(); ++argIndex) {
            const SSABlockArgument& arg = block.ssaArguments()[argIndex];
            if (arg.value == nullptr) {
                error("@" + funcName + " " + blockName(block) + " arg" + std::to_string(argIndex) +
                      ": block argument has a null value");
                continue;
            }
            if (arg.value->kind() != StinkySSAValue::Kind::BlockArgument) {
                error("@" + funcName + " " + blockName(block) + " arg" + std::to_string(argIndex) +
                      ": " + valueName(arg.value) + " is not a block argument");
            }
            if (arg.value->defOp() != nullptr) {
                error("@" + funcName + " " + blockName(block) + " arg" + std::to_string(argIndex) +
                      ": " + valueName(arg.value) + " has a defining instruction");
            }
            for (size_t inc = 0; inc < arg.incoming.size(); ++inc) {
                const SSABlockIncoming& incoming = arg.incoming[inc];
                if (incoming.predecessor == nullptr ||
                    !isPredecessor(block, incoming.predecessor)) {
                    error("@" + funcName + " " + blockName(block) + " arg" +
                          std::to_string(argIndex) + " incoming" + std::to_string(inc) +
                          ": predecessor is not a CFG predecessor");
                }
                if (incoming.use == nullptr) {
                    error("@" + funcName + " " + blockName(block) + " arg" +
                          std::to_string(argIndex) + " incoming" + std::to_string(inc) +
                          ": missing use node");
                    continue;
                }
                if (incoming.use->ownerBlock() != &block) {
                    error("@" + funcName + " " + blockName(block) + " arg" +
                          std::to_string(argIndex) + " incoming" + std::to_string(inc) +
                          ": use is not owned by this block");
                }
                if (incoming.use->kind() == StinkyOpOperand::Kind::Value &&
                    incoming.use->value() == nullptr) {
                    error("@" + funcName + " " + blockName(block) + " arg" +
                          std::to_string(argIndex) + " incoming" + std::to_string(inc) +
                          ": value operand is null");
                }
                observeUse(incoming.use.get());
            }
        }

        for (const IRBase& ir : block) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            const unsigned thisInst = instIndex++;
            if (!inst->hasAttachedSSA()) continue;

            for (size_t i = 0; i < inst->getNumSSAResults(); ++i) {
                StinkySSAValue* value = inst->getSSAResult(i);
                if (value == nullptr) {
                    error("@" + funcName + " #" + std::to_string(thisInst) + " result" +
                          std::to_string(i) + ": null SSA result");
                    continue;
                }
                if (value->defOp() != inst) {
                    error("@" + funcName + " #" + std::to_string(thisInst) + " result" +
                          std::to_string(i) + ": " + valueName(value) + " defOp mismatch");
                }
                if (value->resultIndex() != i) {
                    error("@" + funcName + " #" + std::to_string(thisInst) + " result" +
                          std::to_string(i) + ": " + valueName(value) + " resultIndex mismatch");
                }
            }
            for (size_t i = 0; i < inst->getNumSSAOperands(); ++i) {
                const StinkyOpOperand* operand = inst->getSSAOperand(i);
                if (operand == nullptr) {
                    error("@" + funcName + " #" + std::to_string(thisInst) + " operand" +
                          std::to_string(i) + ": missing operand node");
                    continue;
                }
                if (operand->owner() != inst) {
                    error("@" + funcName + " #" + std::to_string(thisInst) + " operand" +
                          std::to_string(i) + ": owner mismatch");
                }
                if (operand->kind() == StinkyOpOperand::Kind::Value &&
                    operand->value() == nullptr) {
                    error("@" + funcName + " #" + std::to_string(thisInst) + " operand" +
                          std::to_string(i) + ": value operand is null");
                }
                observeUse(operand);
            }
        }
    }

    if (!function.hasAttachedSSA() && function.ssaArena().valueCount() == 0) return result;

    for (StinkySSAValue* value : function.ssaArena().values()) {
        if (value == nullptr) continue;
        std::unordered_set<const StinkyOpOperand*> listed;
        for (StinkyOpOperand* use : value->uses()) {
            if (use == nullptr) {
                error("@" + funcName + " " + valueName(value) + ": null use-list entry");
                continue;
            }
            if (!listed.insert(use).second) {
                error("@" + funcName + " " + valueName(value) + ": duplicate use-list entry");
            }
            if (use->value() != value) {
                error("@" + funcName + " " + valueName(value) +
                      ": use does not point back at this value");
            }
            if (observedUses.find(use) == observedUses.end()) {
                error("@" + funcName + " " + valueName(value) +
                      ": use-list entry is not an IR operand");
            }
        }
    }

    for (const StinkyOpOperand* use : orderedUses) {
        if (use->kind() != StinkyOpOperand::Kind::Value) continue;
        StinkySSAValue* value = use->value();
        if (value == nullptr) continue;
        bool found = false;
        for (StinkyOpOperand* listed : value->uses()) {
            if (listed == use) {
                found = true;
                break;
            }
        }
        if (!found) {
            error("@" + funcName + " " + valueName(value) +
                  ": IR operand is missing from the value use-list");
        }
    }

    return result;
}

}  // namespace stinkytofu
