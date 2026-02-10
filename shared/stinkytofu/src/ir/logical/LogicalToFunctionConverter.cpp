#include "stinkytofu/ir/logical/LogicalToFunctionConverter.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include <cstring>
#include <iostream>

namespace stinkytofu
{
    LogicalToFunctionConverter::LogicalToFunctionConverter(GfxArchID arch)
        : arch(arch)
    {
    }

    void LogicalToFunctionConverter::convert(PyLogicalModule* module, Function& func)
    {
        assert(module && "PyLogicalModule cannot be null");

        // Create a single BasicBlock for all instructions
        // (user can run CFGBuilderPass later to split based on control flow)
        BasicBlock* bb = func.createBasicBlock("entry");
        func.setEntryBlock(bb);
        IRList& irlist = bb->getIR();

        // Keep shared_ptrs alive in Function to prevent double-free
        std::vector<std::shared_ptr<LogicalInstruction>> keepAlive;

        // Add each LogicalInstruction directly to IRList
        // NO LOWERING - LogicalInstruction* and StinkyInstruction* both inherit from IRBase*
        for(const auto& sharedInst : module->getInstructions())
        {
            // Extract raw pointer from shared_ptr
            LogicalInstruction* logicalInst = sharedInst.get();

            // Mark as externally owned so passes don't delete it
            logicalInst->setExternallyOwned(true);

            // Add directly to IRList (cast to IRBase*)
            irlist.push_back(static_cast<IRBase*>(logicalInst));

            // Keep shared_ptr alive
            keepAlive.push_back(sharedInst);
        }

        // Transfer shared_ptr ownership to Function
        func.keepLogicalInstructionsAlive(std::move(keepAlive));
    }

    void LogicalToFunctionConverter::convertWithAutoBlocks(PyLogicalModule* module,
                                                           Function&        func,
                                                           bool             autoSplitBlocks)
    {
        assert(module && "PyLogicalModule cannot be null");

        // Step 1: Identify labels in the instruction stream
        auto labelMap = identifyLabels(module->getInstructions());

        // Step 2: Handle Python/LogicalModule shared_ptr ownership
        //
        // IMPORTANT: This prevents double-free when converting from Python's LogicalModule.
        //
        // Python's LogicalModule owns instructions via shared_ptr. We extract raw pointers
        // to add to IRList, but we must keep the shared_ptrs alive. Otherwise:
        // - Python's shared_ptr destructs -> frees memory
        // - C++ pass tries to delete same pointer -> DOUBLE FREE!
        //
        // Solution:
        // 1. Mark instructions as "externally owned" so passes don't delete them
        // 2. Store shared_ptrs in Function to keep instructions alive
        // 3. After lowering, call Function::releaseLogicalInstructionOwnership()
        std::vector<std::shared_ptr<LogicalInstruction>> keepAlive;
        std::vector<LogicalInstruction*>                 allInsts;

        for(const auto& sharedInst : module->getInstructions())
        {
            // Extract raw pointer for IRList
            LogicalInstruction* rawPtr = sharedInst.get();
            allInsts.push_back(rawPtr);

            // Mark as externally owned so passes don't delete it
            rawPtr->setExternallyOwned(true);

            // Keep shared_ptr alive
            keepAlive.push_back(sharedInst);
        }

        // Transfer shared_ptr ownership to Function
        func.keepLogicalInstructionsAlive(std::move(keepAlive));

        // Step 3: Split into BasicBlocks based on labels
        if(autoSplitBlocks && !labelMap.empty())
        {
            splitIntoBasicBlocks(func, allInsts, labelMap);
        }
        else
        {
            // No splitting: put all instructions in a single block
            BasicBlock* bb = func.createBasicBlock("entry");
            func.setEntryBlock(bb);
            IRList& irlist = bb->getIR();

            for(LogicalInstruction* inst : allInsts)
            {
                irlist.push_back(static_cast<IRBase*>(inst));
            }
        }
    }

    std::unordered_map<std::string, size_t> LogicalToFunctionConverter::identifyLabels(
        const std::vector<std::shared_ptr<LogicalInstruction>>& instructions)
    {
        std::unordered_map<std::string, size_t> labelMap;

        for(size_t i = 0; i < instructions.size(); ++i)
        {
            const auto& inst = instructions[i];

            // Check if this is a label instruction
            // (You'll need to add a Label instruction type to LogicalInstructions.hpp)
            if(std::strcmp(inst->getLogicalName(), "label") == 0)
            {
                // Extract label name from comment or dedicated field
                // For now, use the comment field
                if(!inst->comment.empty())
                {
                    labelMap[inst->comment] = i;
                }
            }
        }

        return labelMap;
    }

    void LogicalToFunctionConverter::splitIntoBasicBlocks(
        Function&                                      func,
        const std::vector<LogicalInstruction*>&        instructions,
        const std::unordered_map<std::string, size_t>& labelMap)
    {
        if(instructions.empty())
        {
            return;
        }

        // Create an initial BasicBlock
        BasicBlock* currentBB = func.createBasicBlock("bb0");
        func.setEntryBlock(currentBB);

        size_t bbIndex = 1;

        for(size_t i = 0; i < instructions.size(); ++i)
        {
            LogicalInstruction* inst = instructions[i];

            // Check if this instruction marks a label boundary
            bool isLabelBoundary = false;
            for(const auto& [labelName, labelIdx] : labelMap)
            {
                if(labelIdx == i && i > 0) // Don't split on first instruction
                {
                    isLabelBoundary = true;
                    break;
                }
            }

            if(isLabelBoundary)
            {
                // Create a new BasicBlock for this label
                std::string bbName = "bb" + std::to_string(bbIndex++);
                currentBB          = func.createBasicBlock(bbName);
            }

            // Add instruction to current BasicBlock (cast to IRBase*)
            currentBB->getIR().push_back(static_cast<IRBase*>(inst));
        }
    }

} // namespace stinkytofu
