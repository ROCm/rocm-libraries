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
#pragma once

#include <algorithm>
#include <cassert>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
/// Use this class in passes that need to erase instructions after
/// def-use chains have been built by buildUseDefChain().
///
class DefUseChainUpdater {
   public:
    /// Assert no users remain, unlink from sources, then erase from the basic block.
    /// Prefer this overload when iterating a basic block with an iterator.
    static IRList::iterator eraseAndUnlink(BasicBlock& bb, IRList::iterator it) {
        if (it->getType() == IRBase::IRType::StinkyTofu) {
            auto* inst = static_cast<StinkyInstruction*>(&*it);
            assert(inst->users.empty() && "Cannot erase instruction that still has users");
            unlinkFromSources(inst);
        }
        return bb.eraseIR(it);
    }

    /// Assert no users remain, unlink from sources, then erase from its parent block.
    static void eraseAndUnlink(StinkyInstruction* inst) {
        assert(inst->users.empty() && "Cannot erase instruction that still has users");
        unlinkFromSources(inst);
        inst->erase();
    }

    /// Transfer all users of oldInst to newInst, updating the sources lists.
    /// After this call, oldInst->users is empty.
    ///
    /// Note: Unless we implement a reaching definition algorithm to correctly
    ///       update the def-use chains, this function is unsafe and may cause
    ///       incorrect results.
    ///
    ///       Only use it when you are sure that the def-use chains are not
    ///       affected by the replacement.
    static void replaceAllUsesWith(StinkyInstruction* oldInst, StinkyInstruction* newInst) {
        for (auto* user : oldInst->users) {
            auto it = std::find(user->sources.begin(), user->sources.end(), oldInst);
            assert(it != user->sources.end());
            *it = newInst;
            newInst->users.push_back(user);
        }
        oldInst->users.clear();
    }

   private:
    /// Remove inst from every source's users list, then clear inst->sources.
    static void unlinkFromSources(StinkyInstruction* inst) {
        for (auto* src : inst->sources)
            if (src) unstableRemove(src->users, inst);
        inst->sources.clear();
    }

    /// Remove a single occurrence of target from vec (swap with back, then pop).
    static void unstableRemove(std::vector<StinkyInstruction*>& vec, StinkyInstruction* target) {
        auto it = std::find(vec.begin(), vec.end(), target);
        assert(it != vec.end() && "target not found in vec");
        *it = vec.back();
        vec.pop_back();
    }
};

}  // namespace stinkytofu
