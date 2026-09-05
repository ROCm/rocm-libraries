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
#include "stinkytofu/transforms/asm/ssa/SSADestruction.hpp"

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/asm/ssa/SSAFunctionShape.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

struct OperandRewrite {
    StinkyInstruction* instruction = nullptr;
    bool isDestination = false;
    size_t operand = 0;
    RegType type = RegType::UNKNOWN;
    unsigned baseIndex = 0;
};

class Destroyer {
   public:
    Destroyer(Function& function, const AllocationResult& allocation)
        : function_(function), allocation_(allocation) {}

    SSADestructionResult run() {
        if (!checkShape()) return std::move(result_);

        indexInstructions();
        planOperandRewrites();
        checkPhis();
        // Applying only after every check keeps a rejected function exactly as
        // it was, including its attached SSA.
        if (!result_.ok()) return std::move(result_);

        for (const OperandRewrite& rewrite : rewrites_) apply(rewrite);
        function_.clearAttachedSSA();
        return std::move(result_);
    }

   private:
    void indexInstructions() {
        uint32_t index = 0;
        for (BasicBlock& bb : function_) {
            for (IRBase& ir : bb) {
                if (const auto* inst = dyn_cast<StinkyInstruction>(&ir))
                    instructionIndex_.emplace(inst, index++);
            }
        }
    }

    std::string locate(const StinkyInstruction* instruction, bool isDestination,
                       size_t operand) const {
        auto it = instructionIndex_.find(instruction);
        const std::string where = it == instructionIndex_.end() ? "<foreign-instruction>"
                                                                : "#" + std::to_string(it->second);
        return "@" + function_.getName() + " " + where + (isDestination ? " dst" : " src") +
               std::to_string(operand);
    }

    void error(const std::string& message) {
        result_.errors.push_back(message);
    }

    bool checkShape() {
        const std::string prefix = "@" + function_.getName() + ": ";
        const uint64_t attachedShape = function_.ssaArena().shape();
        if (attachedShape == kUnstampedShape) return true;

        if (computeFunctionShape(function_) != attachedShape) {
            error(prefix +
                  "the function changed after it was lifted, so the attached SSA describes a "
                  "different program and cannot be lowered");
            return false;
        }
        if (allocation_.shape() != kUnstampedShape && allocation_.shape() != attachedShape) {
            error(prefix +
                  "the allocation was computed against a different graph, so its SSA value IDs do "
                  "not mean the same thing here");
            return false;
        }
        // Same program, different lift scope: the shapes match because the shape
        // hashes physical operands, but the slot layout and value numbering do not.
        const RegClassSet& lifted = function_.ssaArena().liftedClasses();
        if (allocation_.shape() != kUnstampedShape && allocation_.liftedClasses() != lifted) {
            error(prefix + "the allocation was computed against a lift of " +
                  allocation_.liftedClasses().toString() + " but this SSA covers " +
                  lifted.toString() + ", so its SSA value IDs do not mean the same thing here");
            return false;
        }
        return true;
    }

    void planUnits(StinkyInstruction& instruction, bool isDestination, size_t operand,
                   const std::vector<StinkySSAValue*>& units) {
        if (units.empty()) return;

        const std::string where = locate(&instruction, isDestination, operand);
        RegKey first{RegType::UNKNOWN, 0, RegHalf::NONE};

        for (size_t unit = 0; unit < units.size(); ++unit) {
            StinkySSAValue* value = units[unit];
            const SSAValueID id = value == nullptr ? kInvalidSSAValueID : value->valueId();
            if (value == nullptr || !allocation_.isAssigned(id)) {
                error(where + " unit " + std::to_string(unit) + ": %" + std::to_string(id) +
                      " has no physical register");
                return;
            }

            const RegKey physical = allocation_.assignmentOf(id);
            if (physical.half != RegHalf::NONE) {
                error(where + " unit " + std::to_string(unit) + ": %" + std::to_string(id) +
                      " is assigned the sub-DWORD register " + regKeyToString(physical) +
                      ", which cannot be written back to a full-DWORD operand");
                return;
            }

            if (unit == 0) {
                first = physical;
                continue;
            }
            // A range operand must stay one consecutive run in operand order.
            if (physical.type != first.type || physical.idx != first.idx + unit) {
                error(where + ": unit 0 is " + regKeyToString(first) + " but unit " +
                      std::to_string(unit) + " is " + regKeyToString(physical) +
                      "; a range operand must be consecutive in operand order");
                return;
            }
        }

        rewrites_.push_back(
            OperandRewrite{&instruction, isDestination, operand, first.type, first.idx});
    }

    void planOperandRewrites() {
        // The scope the slots were laid out with. An operand outside it was left
        // physical, so it contributes no slot and is never rewritten.
        const RegClassSet& classes = function_.ssaArena().liftedClasses();
        for (BasicBlock& bb : function_) {
            for (IRBase& ir : bb) {
                auto* instruction = dyn_cast<StinkyInstruction>(&ir);
                if (instruction == nullptr || !instruction->hasAttachedSSA()) continue;

                size_t resultCursor = 0;
                const std::vector<StinkyRegister>& destRegs = instruction->getDestRegs();
                for (size_t operand = 0; operand < destRegs.size(); ++operand) {
                    const size_t units = liftedSSAUnits(destRegs[operand], classes);
                    if (units == 0) continue;
                    if (resultCursor + units > instruction->getNumSSAResults()) {
                        error(locate(instruction, /*isDestination=*/true, operand) +
                              ": attached SSA is missing destination units");
                        return;
                    }
                    std::vector<StinkySSAValue*> values;
                    values.reserve(units);
                    for (size_t unit = 0; unit < units; ++unit)
                        values.push_back(instruction->getSSAResult(resultCursor++));
                    planUnits(*instruction, /*isDestination=*/true, operand, values);
                }

                size_t operandCursor = 0;
                const std::vector<StinkyRegister>& srcRegs = instruction->getSrcRegs();
                for (size_t operand = 0; operand < srcRegs.size(); ++operand) {
                    const size_t units = liftedSSAUnits(srcRegs[operand], classes);
                    if (units == 0) {
                        if (operandCursor < instruction->getNumSSAOperands()) ++operandCursor;
                        continue;
                    }
                    if (operandCursor + units > instruction->getNumSSAOperands()) {
                        error(locate(instruction, /*isDestination=*/false, operand) +
                              ": attached SSA is missing source units");
                        return;
                    }
                    std::vector<StinkySSAValue*> values;
                    values.reserve(units);
                    for (size_t unit = 0; unit < units; ++unit)
                        values.push_back(instruction->getSSAOperandValue(operandCursor++));
                    planUnits(*instruction, /*isDestination=*/false, operand, values);
                }
            }
        }
    }

    void checkPhis() {
        for (const BasicBlock& block : function_) {
            for (size_t argIndex = 0; argIndex < block.ssaArguments().size(); ++argIndex) {
                const SSABlockArgument& arg = block.ssaArguments()[argIndex];
                if (arg.incoming.empty()) continue;
                if (arg.value == nullptr) continue;

                const SSAValueID resultId = arg.value->valueId();
                if (!allocation_.isAssigned(resultId)) {
                    error("@" + function_.getName() + " phi#%" + std::to_string(resultId) +
                          ": result %" + std::to_string(resultId) + " has no physical register");
                    continue;
                }

                const RegKey resultPhysical = allocation_.assignmentOf(resultId);
                for (size_t edge = 0; edge < arg.incoming.size(); ++edge) {
                    const StinkyOpOperand* use = arg.incoming[edge].use.get();
                    StinkySSAValue* incoming = use == nullptr ? nullptr : use->value();
                    const SSAValueID incomingId =
                        incoming == nullptr ? kInvalidSSAValueID : incoming->valueId();
                    if (incoming == nullptr || !allocation_.isAssigned(incomingId)) {
                        error("@" + function_.getName() + " phi#%" + std::to_string(resultId) +
                              " edge " + std::to_string(edge) + ": %" + std::to_string(incomingId) +
                              " has no physical register");
                        continue;
                    }
                    const RegKey incomingPhysical = allocation_.assignmentOf(incomingId);
                    if (incomingPhysical == resultPhysical) continue;

                    error("@" + function_.getName() + " phi#%" + std::to_string(resultId) +
                          " edge " + std::to_string(edge) + ": %" + std::to_string(incomingId) +
                          " is " + regKeyToString(incomingPhysical) + " but the result is " +
                          regKeyToString(resultPhysical) +
                          "; lowering that needs a copy on the incoming edge, which is not "
                          "implemented yet");
                }
            }
        }
    }

    void apply(const OperandRewrite& rewrite) {
        const std::vector<StinkyRegister>& operands = rewrite.isDestination
                                                          ? rewrite.instruction->getDestRegs()
                                                          : rewrite.instruction->getSrcRegs();
        // Rewriting only the class and base index keeps width, modifiers, and
        // symbolic names exactly as the producer wrote them.
        StinkyRegister updated = operands[rewrite.operand];
        result_.rewritten.push_back({rewrite.instruction, rewrite.isDestination, rewrite.operand,
                                     updated.reg.type, updated.reg.idx, rewrite.type,
                                     rewrite.baseIndex});
        updated.reg.type = rewrite.type;
        updated.reg.idx = rewrite.baseIndex;

        if (rewrite.isDestination)
            rewrite.instruction->setDestReg(rewrite.operand, updated);
        else
            rewrite.instruction->setSrcReg(rewrite.operand, updated);
    }

    Function& function_;
    const AllocationResult& allocation_;

    std::unordered_map<const StinkyInstruction*, uint32_t> instructionIndex_;
    std::vector<OperandRewrite> rewrites_;
    SSADestructionResult result_;
};

}  // namespace

std::string SSADestructionResult::toString() const {
    std::ostringstream out;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) out << "\n";
        out << errors[i];
    }
    return out.str();
}

SSADestructionResult destroyAttachedSSA(Function& function, const AllocationResult& allocation) {
    return Destroyer(function, allocation).run();
}

}  // namespace stinkytofu
