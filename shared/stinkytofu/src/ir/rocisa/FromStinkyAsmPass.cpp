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
#include <iostream> // TODO: don't use iostream.

#include "code.hpp"
#include "instruction/common.hpp"

#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/rocisa/ToStinkyAsm.hpp"

#define DEBUG_TYPE "StinkyAsmToRocisaPass"

namespace
{
    using namespace rocisa;
    using namespace stinkytofu;

    class OriginalSharedItemMap final
    {
        std::unordered_map<Item*, std::shared_ptr<Item>> origSharedItems;

    public:
        void addMapping(Item* item, std::shared_ptr<Item> shared)
        {
            assert(origSharedItems.find(item) == origSharedItems.end()
                   && "Internal Error: Duplicate item in original shared item map");
            origSharedItems[item] = std::move(shared);
        }

        std::shared_ptr<Item> pop(Item* item)
        {
            auto it = origSharedItems.find(item);
            assert(it != origSharedItems.end()
                   && "Internal Error: Unable to find original shared item");

            std::shared_ptr<Item> shared = std::move(it->second);
            origSharedItems.erase(it);
            return shared;
        }

        bool has(Item* item) const
        {
            return origSharedItems.find(item) != origSharedItems.end();
        }
    };

    // Note: this will invalidate the original module's itemList
    void buildOrigSharedItemsMap(Module& module, OriginalSharedItemMap& origSharedItems)
    {
        for(std::shared_ptr<Item>& origShared : module.itemList)
        {
            Item* origItem = origShared.get();
            if(Module* subMod = dynamic_cast<Module*>(origItem))
            {
                buildOrigSharedItemsMap(*subMod, origSharedItems);
            }
            else
            {
                origSharedItems.addMapping(origItem, std::move(origShared));
            }
        }
    }

    std::shared_ptr<Item> createSharedRocisa(const StinkyInstruction& stinkyInst)
    {
        switch(stinkyInst.getUnifiedOpcode())
        {
        case GFX::s_waitcnt:
        {
            const SWaitCntData* data = stinkyInst.getModifier<SWaitCntData>();
            assert(data && "Internal Error: S_WAITCNT missing SWaitCntData modifier");
            std::shared_ptr<Item> item = std::make_shared<SWaitCnt>();

            SWaitCnt* waitcnt = static_cast<SWaitCnt*>(item.get());

            waitcnt->vlcnt = data->vlcnt;
            waitcnt->vscnt = data->vscnt;
            // Convert
            if(data->dlcnt == -1 && data->dscnt == -1)
            {
                waitcnt->dscnt = -1;
            }
            else if(data->dlcnt == -1)
            {
                waitcnt->dscnt = data->dscnt;
            }
            else if(data->dscnt == -1)
            {
                waitcnt->dscnt = data->dlcnt;
            }
            else
            {
                waitcnt->dscnt = data->dlcnt + data->dscnt;
            }
            waitcnt->kmcnt = data->kmcnt;

            return std::move(item);
        }
        case GFX::s_wait_tensorcnt:
        {
            const SWaitTensorCntData* data = stinkyInst.getModifier<SWaitTensorCntData>();
            assert(data && "Internal Error: S_WAITTENSORCNT missing SWaitTensorCntData modifier");
            std::shared_ptr<Item> item = std::make_shared<SWaitTensorcnt>();

            SWaitTensorcnt* waittensorcnt = static_cast<SWaitTensorcnt*>(item.get());

            waittensorcnt->tensorcnt = data->tlcnt;

            return std::move(item);
        }
        default:
            assert(false
                   && "Internal Error: TODO: Unsupported StinkyInstruction to Rocisa creation.");
            return nullptr;
        }
    }

    // --- Update the Module with the scheduled instructions ---
    //
    // This pass will convert the scheduled StinkyInstructions back to the Module's
    // flat items (Instructions).
    class StinkyAsmToRocisaPass : public StinkyInstPass
    {
        OriginalSharedItemMap origSharedItems;

    public:
        static char ID;

        const char* getName() const override
        {
            return "StinkyAsmToRocisaPass";
        }

        PassID getPassID() const override
        {
            return &StinkyAsmToRocisaPass::ID;
        }

        Label* getSingleLastLabel(const std::vector<Item*>& flatItems)
        {
#ifndef NDEBUG
            Label* lastLabel = nullptr;
            for(Item* item : flatItems)
                if(lastLabel = dynamic_cast<Label*>(item))
                    break;
            assert(lastLabel == flatItems.back() && "Internal Error: No label found in the module");
#endif
            return static_cast<Label*>(flatItems.back());
        }

        void run(IRList& irlist, PassContext& passCtx) override
        {
            //eraseLabel(irlist, passCtx);

            RocisaDFSFlatItems& flatItemsData
                = passCtx.getAnalysisManager().getResult<RocisaDFSFlatItems>(irlist, passCtx);

            RocisaStinkyMapping& mapping
                = passCtx.getAnalysisManager().getResult<RocisaStinkyMapping>(irlist, passCtx);

            const std::vector<Item*>& flatItems = flatItemsData.getFlatItems();

            Module& module = flatItemsData.getModule();

            PASS_DEBUG(std::cerr << "\n===================================================\n";
                       std::cerr << "Input\n:";
                       std::cerr << module.prettyPrint() << "\n");

            // build instruction to text block mapping
            std::unordered_map<Item*, std::vector<Item*>> instPairedTextblocks;
            instPairedTextblocks.reserve(irlist.size());

            Item* curInstOrLabel = nullptr;
            for(unsigned i = flatItems.size(); i != 0; --i)
            {
                Item* item = flatItems[i - 1];
                if(dynamic_cast<Instruction*>(item) || dynamic_cast<Label*>(item))
                {
                    curInstOrLabel = static_cast<Instruction*>(item);
                }
                else
                {
                    if(curInstOrLabel)
                        instPairedTextblocks[curInstOrLabel].push_back(item);
                }
            }

            // Preserve the original shared_ptr into origSharedItems object.
            // The module's itemList is no longer valid after this.
            buildOrigSharedItemsMap(module, origSharedItems);
            module.itemList.clear();

            // We don't need to do this at all, since this is the final pass for StinkyInstruction.
            passCtx.getAnalysisManager().invalidate(RocisaDFSFlatItems::ID);

            std::vector<std::shared_ptr<Item>> reorderedItems;
            reorderedItems.reserve(flatItems.size());

            for(IRBase& ir : irlist)
            {
                StinkyInstruction& inst = *cast<StinkyInstruction>(&ir);

                rocisa::Instruction* rocInst = mapping.getRocisaInst(&inst);

                rocisa::Label* rocLabel = nullptr;
                if(rocInst == nullptr)
                {
                    rocLabel = mapping.getRocisaLabel(&inst);
                }

                rocisa::Item* rocItem = rocInst != nullptr ? static_cast<rocisa::Item*>(rocInst)
                                                           : static_cast<rocisa::Item*>(rocLabel);

                // If the instruction has TextBlocks associated with it, add them first
                auto it = instPairedTextblocks.find(rocItem);
                if(it != instPairedTextblocks.end())
                {
                    std::vector<Item*>& textblocks = it->second;
                    for(unsigned i = textblocks.size(); i != 0; --i)
                    {
                        Item* text = textblocks[i - 1];

                        std::shared_ptr<Item> origShared = origSharedItems.pop(text);

                        // use std::move to avoid copying the shared_ptr
                        reorderedItems.push_back(std::move(origShared));
                    }
                }

                // The instruction is created in Stinky passes so it's not in the original module.
                if(rocItem == nullptr)
                {
                    assert(rocLabel == nullptr && "TODO: Handle label creation");
                    reorderedItems.push_back(std::move(createSharedRocisa(inst)));
                }
                else
                {
                    reorderedItems.push_back(std::move(origSharedItems.pop(rocItem)));
                }
            }

            // for(Item* lastLabel : lastLabels)
            //     reorderedItems.push_back(origSharedItems.pop(lastLabel));

            module.itemList = std::move(reorderedItems);

            PASS_DEBUG(std::cerr << "\n===================================================\n";
                       std::cerr << "Final Output\n:";
                       std::cerr << module.prettyPrint() << "\n");
        }
    };

    char StinkyAsmToRocisaPass::ID = 0;
}

namespace stinkytofu
{
    std::unique_ptr<Pass> createStinkyAsmToRocisaPass()
    {
        return std::make_unique<StinkyAsmToRocisaPass>();
    }
}
