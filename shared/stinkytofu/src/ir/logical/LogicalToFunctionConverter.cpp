#include "stinkytofu/ir/logical/LogicalToFunctionConverter.hpp"

#include <cstring>
#include <iostream>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"

namespace stinkytofu {
LogicalToFunctionConverter::LogicalToFunctionConverter(GfxArchID arch) : arch(arch) {}

void LogicalToFunctionConverter::convert(PyLogicalModule* module, PyLogicalFunction& pyFunc) {
    assert(module && "PyLogicalModule cannot be null");
    Function* func = pyFunc.getFunction();
    assert(func && "PyLogicalFunction must wrap a Function");
    assert(func->empty() && "Function must be empty");

    // Create a single BasicBlock for all instructions
    // (user can run CFGBuilderPass later to split based on control flow)
    BasicBlock* bb = func->createBasicBlock("entry");

    // Add each LogicalInstruction directly to IRList (appendIR sets parent)
    // NO LOWERING - LogicalInstruction* and StinkyInstruction* both inherit from IRBase*
    // PyLogicalFunction destructor will detach ownedExternally IRs so the list does not delete
    // them.
    for (const auto& sharedInst : module->getInstructions()) {
        LogicalInstruction* logicalInst = sharedInst.get();
        bb->appendIR(static_cast<IRBase*>(logicalInst));
    }
}

void LogicalToFunctionConverter::convertWithAutoBlocks(PyLogicalModule* module,
                                                       PyLogicalFunction& pyFunc,
                                                       bool autoSplitBlocks) {
    assert(module && "PyLogicalModule cannot be null");
    Function* func = pyFunc.getFunction();
    assert(func && "PyLogicalFunction must wrap a Function");

    // Step 1: Identify labels in the instruction stream
    auto labelMap = identifyLabels(module->getInstructions());

    // Step 2: Extract raw pointers. PyLogicalFunction destructor detaches ownedExternally IRs.
    std::vector<LogicalInstruction*> allInsts;
    for (const auto& sharedInst : module->getInstructions()) {
        allInsts.push_back(sharedInst.get());
    }

    // Step 3: Split into BasicBlocks based on labels
    if (autoSplitBlocks && !labelMap.empty()) {
        splitIntoBasicBlocks(*func, allInsts, labelMap);
    } else {
        // No splitting: put all instructions in a single block
        BasicBlock* bb = func->createBasicBlock("entry");

        for (LogicalInstruction* inst : allInsts) {
            bb->appendIR(static_cast<IRBase*>(inst));
        }
    }
}

std::unordered_map<std::string, size_t> LogicalToFunctionConverter::identifyLabels(
    const std::vector<std::shared_ptr<LogicalInstruction>>& instructions) {
    std::unordered_map<std::string, size_t> labelMap;

    for (size_t i = 0; i < instructions.size(); ++i) {
        const auto& inst = instructions[i];

        // Check if this is a label instruction
        // (You'll need to add a Label instruction type to LogicalInstructions.hpp)
        if (std::strcmp(inst->getLogicalName(), "label") == 0) {
            // Extract label name from comment or dedicated field
            // For now, use the comment field
            if (!inst->comment.empty()) {
                labelMap[inst->comment] = i;
            }
        }
    }

    return labelMap;
}

void LogicalToFunctionConverter::splitIntoBasicBlocks(
    Function& func, const std::vector<LogicalInstruction*>& instructions,
    const std::unordered_map<std::string, size_t>& labelMap) {
    if (instructions.empty()) {
        return;
    }

    // Create an initial BasicBlock
    BasicBlock* currentBB = func.createBasicBlock("bb0");

    size_t bbIndex = 1;

    for (size_t i = 0; i < instructions.size(); ++i) {
        LogicalInstruction* inst = instructions[i];

        // Check if this instruction marks a label boundary
        bool isLabelBoundary = false;
        for (const auto& [labelName, labelIdx] : labelMap) {
            if (labelIdx == i && i > 0)  // Don't split on first instruction
            {
                isLabelBoundary = true;
                break;
            }
        }

        if (isLabelBoundary) {
            // Create a new BasicBlock for this label
            std::string bbName = "bb" + std::to_string(bbIndex++);
            currentBB = func.createBasicBlock(bbName);
        }

        // Add instruction to current BasicBlock (appendIR sets parent)
        currentBB->appendIR(static_cast<IRBase*>(inst));
    }
}

}  // namespace stinkytofu
