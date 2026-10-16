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
#include "code.hpp"
#include <memory>

namespace rocisa
{
    // External structures and functions
    struct rocIsaPassOption
    {
        bool insertDelayAlu  = false;
        bool removeDupFunc   = true;
        bool removeDupAssign = true;
        bool getCycles       = true;
        int  numWaves        = 0; // is used when getCycles is true

        bool doOpt() const
        {
            return removeDupAssign;
        }
    };

    struct rocIsaPassResult
    {
        int cycles  = -1;
        // Highest VGPR index referenced by any instruction after the pass
        // (i.e. max_vgpr_index + 1), derived from the register graph the pass
        // already builds.  -1 means the pass did not compute it (e.g. when
        // doOpt() is false); in that case Python skips the occupancy update.
        int maxVgpr = -1;
    };

    std::string getActFuncModuleName(int gwvw, int sgpr, int tmpVgpr, int tmpSgpr);
    std::string getActFuncBranchModuleName();

    rocIsaPassResult rocIsaPass(std::shared_ptr<KernelBody>& kernel,
                                const rocIsaPassOption&      option); // Return value is std::move()

    // Internal use only
    struct Graph
    {
        // Only the sgpr lists are consumed (by removeDuplicateAssignment), so
        // vgpr/mgpr only record their sizes for the out-of-range checks.
        //
        // Barrier items (branches, labels, waitcnts, ...) apply to every sgpr,
        // so instead of being pushed into each list they bump barrierSeq, and
        // each entry records the barrierSeq it was added at.  A resetting
        // barrier between two entries is detected by comparing resetsBefore.
        enum class Kind : uint8_t
        {
            Reset, // NoOptItem, branch, label, macro, waitcnt
            SMov, // SMovB32
            Instr, // any other Instruction
            Other
        };
        struct Entry
        {
            std::shared_ptr<Item> item;
            int                   barrierSeq; // barriers recorded before this entry
            Kind                  kind;
            bool                  noOpt; // item came from a NoOpt module
            bool                  dstHit; // Instr whose first param covers this sgpr
        };
        std::vector<std::vector<Entry>> sgpr;
        size_t                          vgprSize   = 0;
        size_t                          mgprSize   = 0;
        int                             barrierSeq = 0;
        // resetsBefore[k] = number of resetting barriers among the first k.
        std::vector<int> resetsBefore = {0};
        // Highest VGPR index referenced by any instruction; -1 = none seen.
        int maxVgprSeen = -1;
    };

    // No opt item container class
    class NoOptItem : public Item
    {
    public:
        NoOptItem(std::shared_ptr<Item> item)
            : Item("")
            , _item(item)
        {
        }
        std::shared_ptr<Item> getItem() const
        {
            return _item;
        }

    private:
        std::shared_ptr<Item> _item;
    };

    void insertDelayAlu(std::shared_ptr<Module> module);
    void removeDuplicatedFunction(std::shared_ptr<Module> module);
    void macroToInstruction(std::shared_ptr<Module>& module);
    void compositeToInstruction(std::shared_ptr<Module>& module);
    void convertTextVariablesToRegisters(std::shared_ptr<Module> module);
    Graph buildGraph(
        std::shared_ptr<Module>               module,
        int                                   vgprMax,
        int                                   sgprMax); // Return value is std::move()d
    void removeDuplicateAssignment(Graph& graph);

    int getCycles(std::shared_ptr<Module> module, int numWaves);
} // namespace rocisa
