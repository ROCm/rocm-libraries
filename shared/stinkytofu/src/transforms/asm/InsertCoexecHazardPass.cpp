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

#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "InsertCoexecHazardPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/CoexecNopPlanner.hpp"

namespace {
using namespace stinkytofu;

class InsertCoexecHazardPass : public StinkyInstPass {
   public:
    static char ID;
    InsertCoexecHazardPass() = default;

    const char* getName() const override {
        return "InsertCoexecHazardPass";
    }

    PassID getPassID() const override {
        return &InsertCoexecHazardPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        setupArch(passCtx);
        if (!func.empty()) processFunction(func);
        return preserveCFGAnalyses();
    }

   private:
    void setupArch(PassContext& passCtx) {
        auto arch = passCtx.getGemmTileConfig().arch;
        archId_ = getGfxArchID(arch[0], arch[1], arch[2]);
        hw_ = &passCtx.getHWModel();
        PASS_DEBUG(std::cerr << "[InsertCoexecHazard] run arch=gfx" << arch[0] << arch[1] << arch[2]
                             << "\n");
    }

    // The plan comes from planCoexecNops (shared with the co-issue repair's insertion
    // model); this pass only emits it.
    void processFunction(Function& func) {
        BlockSequences seqs;
        std::vector<const BasicBlock*> blocks;
        for (BasicBlock& bb : func) {
            blocks.push_back(&bb);
            auto& seq = seqs[&bb];
            for (IRBase& node : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node)) seq.push_back(inst);
        }
        // A detached v_nop stands for the planned ones, so later scans count them as
        // fillers, as they counted the v_nops this pass used to insert as it went.
        Function scratch;
        AsmIRBuilder poolBuilder(*scratch.createBasicBlock(), archId_);
        const StinkyInstruction* vnop = poolBuilder.create(getMCIDByUOp(GFX::v_nop, archId_));

        const auto plan = planCoexecNops(*hw_, blocks, seqs, *vnop);
        for (size_t b = 0; b < blocks.size(); ++b)
            for (const CoexecNopInsertion& insertion : plan[b])
                insertVNops(*const_cast<BasicBlock*>(blocks[b]),
                            const_cast<StinkyInstruction*>(insertion.before), insertion.count);
    }

    void insertVNops(BasicBlock& bb, IRBase* insertBefore, int n) {
        AsmIRBuilder builder(bb, archId_);
        for (int i = 0; i < n; ++i) builder.create(getMCIDByUOp(GFX::v_nop, archId_), insertBefore);
    }

    GfxArchID archId_ = GfxArchID{};
    const HWModel* hw_ = nullptr;
};

char InsertCoexecHazardPass::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertCoexecHazardPass() {
    return std::make_unique<InsertCoexecHazardPass>();
}
}  // namespace stinkytofu
