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
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
    using namespace stinkytofu;

    // CFGBuilderPass implementation
    class CFGBuilderPassImpl : public Pass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "CFG Builder";
        }

        PassID getPassID() const override
        {
            return &CFGBuilderPassImpl::ID;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            // If the function has more than one BasicBlock, it already has a CFG
            if(func.size() > 1)
                return;

            // If the function is empty or has no entry block, nothing to do
            BasicBlock* flatBB = func.getEntryBlock();
            if(!flatBB || flatBB->empty())
                return;

            // Check if we need to split - look for label instructions
            if(!needsSplitting(flatBB))
                return;

            // Split the flat BasicBlock into multiple BasicBlocks at label boundaries
            splitAtLabels(func, flatBB);

            // Build CFG edges based on branches and fall-through
            buildCFGEdges(func);
        }

    private:
        bool needsSplitting(BasicBlock* bb)
        {
            // Check if there are any label instructions (GFX::LABEL opcode)
            for(IRBase& irNode : *bb)
                if(cast<StinkyInstruction>(&irNode)->getUnifiedOpcode() == GFX::LABEL)
                    return true;
            return false;
        }

        void splitAtLabels(Function& func, BasicBlock* flatBB)
        {
            // Find all label positions
            std::vector<BasicBlock::iterator> labelPositions;
            for(auto it = flatBB->begin(); it != flatBB->end(); ++it)
                if(cast<StinkyInstruction>(it.getNodePtr())->getUnifiedOpcode() == GFX::LABEL)
                    labelPositions.push_back(it);

            assert(!labelPositions.empty() && "No labels found? This should not happen.");

            // For each label, create a new BasicBlock
            for(size_t i = 0; i < labelPositions.size(); ++i)
            {
                auto               labelIt   = labelPositions[i];
                StinkyInstruction* labelInst = cast<StinkyInstruction>(labelIt.getNodePtr());

                // Get the label name
                auto        labelData = labelInst->getModifier<LabelData>();
                std::string labelName = labelData ? labelData->label : "";

                // Create a new BasicBlock for this label
                BasicBlock* newBB = func.createBasicBlock(labelName);

                // Determine the range of IR to move
                auto startIt = labelIt;
                auto endIt
                    = (i + 1 < labelPositions.size()) ? labelPositions[i + 1] : flatBB->end();

                // Move IR from startIt to endIt to the new BasicBlock
                auto it = startIt;
                while(it != endIt)
                {
                    IRBase* instNode = it.getNodePtr();
                    auto    nextIt   = std::next(it);
                    flatBB->removeIR(instNode);
                    newBB->appendIR(instNode);
                    it = nextIt;
                }
            }

            // If there is IR before the first label, keep them in flatBB
            // Otherwise, remove the now-empty flatBB
            if(flatBB->empty())
            {
                // Entry block remains basicBlocks.front() (flatBB)
                // Don't remove flatBB yet as it might still be referenced
                // The user can clean it up if needed
            }
        }

        void buildCFGEdges(Function& func)
        {
            // Build a map of label names to BasicBlocks
            std::unordered_map<std::string, BasicBlock*> labelMap;
            for(BasicBlock& bb : func)
            {
                if(!bb.getLabel().empty())
                {
                    labelMap[bb.getLabel()] = &bb;
                }
            }

            // Connect BasicBlocks based on branches and fall-through
            BasicBlock* prevBB = nullptr;
            for(BasicBlock& bb : func)
            {
                // Add branch edges from current block
                IRBase* terminator = bb.getTerminator();
                if(terminator)
                {
                    StinkyInstruction* termInst = cast<StinkyInstruction>(terminator);
                    if(isBranch(*termInst))
                    {
                        // Get the branch target label using utility function
                        std::string targetLabel = getBranchTarget(*termInst);
                        auto        targetIt    = labelMap.find(targetLabel);
                        if(targetIt != labelMap.end())
                        {
                            bb.addSuccessor(targetIt->second);
                            targetIt->second->addPredecessor(&bb);
                        }
                    }
                }

                // Add fall-through edge from previous block if it falls through
                if(prevBB)
                {
                    // Check if prevBB should fall through to current bb
                    // This happens when prevBB has no terminator or has a conditional branch
                    IRBase* prevTerm          = prevBB->getTerminator();
                    bool    shouldFallThrough = true;
                    if(prevTerm)
                    {
                        StinkyInstruction* prevTermInst = cast<StinkyInstruction>(prevTerm);
                        if(isBranch(*prevTermInst))
                        {
                            // Unconditional branches don't fall through
                            // Conditional branches do fall through (when condition is false)
                            shouldFallThrough = isConditionalBranch(*prevTermInst);
                        }
                    }

                    if(shouldFallThrough)
                    {
                        prevBB->addSuccessor(&bb);
                        bb.addPredecessor(prevBB);
                    }
                }

                prevBB = &bb;
            }
        }
    };

    char CFGBuilderPassImpl::ID = 0;

} // namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createCFGBuilderPass()
    {
        return std::make_unique<CFGBuilderPassImpl>();
    }
} // namespace stinkytofu
