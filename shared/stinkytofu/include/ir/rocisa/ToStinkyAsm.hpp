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
#pragma once

#include "stinkytofu.hpp"

namespace rocisa
{
    struct Item;
    struct Instruction;
    struct Label;
    struct Module;
};

namespace stinkytofu
{
    struct StinkyInstruction;

    // RocisaFlatItems is used to provide a flat list of Items of root
    // Module. A pass can get the results through AnalysisManager.
    //
    // flatItems is constructed in DFS order from Module::itemList
    // by recursively traversing each Module and its submodules,
    // excluding the Module type itself.
    class RocisaDFSFlatItems : public AnalysisPass
    {
        std::vector<rocisa::Item*> flatItems;
        rocisa::Module&            module;

    public:
        static Pass::ID ID;

        RocisaDFSFlatItems(rocisa::Module& module)
            : module(module)
        {
        }

        const std::vector<rocisa::Item*>& getFlatItems() const
        {
            return flatItems;
        }

        Pass::ID getPassID() const override
        {
            return ID;
        }

        const char* getName() const override
        {
            return "RocisaFlatItems";
        }

        rocisa::Module& getModule()
        {
            return module;
        }

        void run(IRList& irlist, PassContext& passCtx) override;
    };

    //
    class RocisaStinkyMapping : public AnalysisPass
    {
        std::unordered_map<StinkyInstruction*, rocisa::Instruction*> stinkyToRocisaMap;
        std::unordered_map<StinkyInstruction*, rocisa::Label*>       stinkyToRocisaLabelMap;

    public:
        static Pass::ID ID;

        const char* getName() const override
        {
            return "RocisaStinkyMapping";
        }

        PassID getPassID() const override
        {
            return &RocisaStinkyMapping::ID;
        }

        void run(IRList& insts, PassContext& passCtx) override
        {
            // Do nothing. The map is populated by RocisaToStinkyAsmPass.
            assert(stinkyToRocisaMap.empty()
                   && "RocisaStinkyMapping should be empty before being populated.");
        }

        void addMapping(StinkyInstruction* stinkyInst, rocisa::Instruction* rocInst)
        {
            assert(stinkyToRocisaMap.find(stinkyInst) == stinkyToRocisaMap.end()
                   && "StinkyInstruction already mapped.");
            stinkyToRocisaMap[stinkyInst] = rocInst;
        }

        rocisa::Instruction* getRocisaInst(StinkyInstruction* stinkyInst) const
        {
            auto it = stinkyToRocisaMap.find(stinkyInst);
            if(it != stinkyToRocisaMap.end())
                return it->second;
            return nullptr;
        }

        void addMapping(StinkyInstruction* stinkyInst, rocisa::Label* rocLabel)
        {
            assert(stinkyToRocisaLabelMap.find(stinkyInst) == stinkyToRocisaLabelMap.end()
                   && "StinkyInstruction already mapped.");
            stinkyToRocisaLabelMap[stinkyInst] = rocLabel;
        }

        rocisa::Label* getRocisaLabel(StinkyInstruction* stinkyInst) const
        {
            auto it = stinkyToRocisaLabelMap.find(stinkyInst);
            if(it != stinkyToRocisaLabelMap.end())
                return it->second;
            return nullptr;
        }
    };

} // namespace stinkytofu
