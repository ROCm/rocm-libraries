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
#include "stinkytofu/transforms/asm/LoopRegionRemarkPass.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

namespace {
using namespace stinkytofu;

static const char* kPassName = "LoopRegionRemark";

// --- Boundary classification ---

enum class BoundaryKind { Wait, Store, Barrier, Branch, SideEffect, UntokenizedMem };

static const char* boundaryKindStr(BoundaryKind kind) {
    switch (kind) {
        case BoundaryKind::Wait:
            return "wait";
        case BoundaryKind::Store:
            return "store";
        case BoundaryKind::Barrier:
            return "barrier";
        case BoundaryKind::Branch:
            return "branch";
        case BoundaryKind::SideEffect:
            return "side_effect";
        case BoundaryKind::UntokenizedMem:
            return "untokenized_mem";
    }
    return "unknown";
}

static BoundaryKind classifyBoundary(const StinkyInstruction& inst) {
    if (isWaitCnt(inst)) return BoundaryKind::Wait;
    if (isGlobalMemStore(inst)) return BoundaryKind::Store;
    if (isBranch(inst)) return BoundaryKind::Branch;
    if (isBarrier(inst) && !hasLdsPseudoRegs(inst)) return BoundaryKind::Barrier;
    if ((isTensorLoad(inst) || isDSRead(inst) || isDSWrite(inst)) && !hasLdsPseudoRegs(inst))
        return BoundaryKind::UntokenizedMem;
    return BoundaryKind::SideEffect;
}

struct BoundaryInfo {
    BoundaryKind kind;
    const char* mnemonic;
    unsigned instIndex;
};

struct BBAnalysis {
    std::string label;
    unsigned regionCount = 1;
    unsigned instCount = 0;
    unsigned snopCount = 0;
    unsigned snopWastedCycles = 0;
    unsigned branchCount = 0;
    std::vector<BoundaryInfo> boundaries;
};

static BBAnalysis analyzeBB(BasicBlock& bb) {
    BBAnalysis result;
    result.label = bb.getLabel();

    unsigned instIdx = 0;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!instPtr) {
            ++instIdx;
            continue;
        }

        StinkyInstruction& inst = *instPtr;
        if (!inst.getHwInstDesc()) STINKY_UNREACHABLE("StinkyInstruction without HwInstDesc");
        result.instCount++;

        // s_nop detection
        if (inst.getHwInstDesc()->unifiedOpcode == GFX::s_nop) {
            result.snopCount++;
            const auto& srcs = inst.getSrcRegs();
            unsigned wastedCycles = 1;
            if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt)
                wastedCycles = static_cast<unsigned>(srcs[0].getLiteralInt()) + 1;
            result.snopWastedCycles += wastedCycles;
        }

        // Branch count
        if (isBranch(inst)) {
            result.branchCount++;
        }

        // Region boundary detection
        if (hasSideEffect(inst)) {
            result.regionCount++;
            result.boundaries.push_back(
                {classifyBoundary(inst), inst.getHwInstDesc()->mnemonic, instIdx});
        }

        ++instIdx;
    }

    return result;
}

class LoopRegionRemarkPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "LoopRegionRemarkPass";
    }

    PassID getPassID() const override {
        return &LoopRegionRemarkPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        if (!passCtx.getRemarksEnabled()) return PreservedAnalyses::all();

        const auto& loops = AM.getResult<LoopAnalysis>(func);
        if (loops.empty()) return PreservedAnalyses::all();

        for (const Loop& loop : loops) {
            if (!loop.headerBB) continue;

            unsigned totalInsts = 0;
            unsigned totalRegions = 0;
            unsigned totalSnop = 0;
            unsigned totalSnopCycles = 0;
            unsigned totalBranches = 0;

            for (BasicBlock* bb : loop.bodyBBs) {
                BBAnalysis ba = analyzeBB(*bb);
                totalInsts += ba.instCount;
                totalRegions += ba.regionCount;
                totalSnop += ba.snopCount;
                totalSnopCycles += ba.snopWastedCycles;
                totalBranches += ba.branchCount;

                if (!ba.boundaries.empty()) {
                    {
                        std::ostringstream oss;
                        oss << "  BB '" << ba.label << "': " << ba.regionCount << " regions, "
                            << ba.boundaries.size() << " boundaries:";
                        emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName,
                                             "BBRegions", oss.str()});
                    }
                    for (const BoundaryInfo& bi : ba.boundaries) {
                        std::ostringstream oss;
                        oss << "    [" << boundaryKindStr(bi.kind) << "] " << bi.mnemonic
                            << " (inst #" << bi.instIndex << ")";
                        if (bi.kind == BoundaryKind::UntokenizedMem)
                            oss << " -- no MemTokenData, non-movable";
                        emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName,
                                             "Boundary", oss.str()});
                    }
                }

                if (ba.snopCount > 0) {
                    std::ostringstream oss;
                    oss << "  BB '" << ba.label << "': " << ba.snopCount << " s_nop instructions ("
                        << ba.snopWastedCycles << " wasted cycles)";
                    emitRemark(passCtx,
                               {OptimizationRemark::Kind::Analysis, kPassName, "SNop", oss.str()});
                }
            }

            // Summary remark
            {
                std::ostringstream oss;
                oss << "loop '" << loop.headerBB->getLabel() << "' summary: " << totalInsts
                    << " insts, " << totalRegions << " regions, " << totalSnop << " s_nop ("
                    << totalSnopCycles << " wasted cycles), " << totalBranches << " branches";
                emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName, "LoopSummary",
                                     oss.str()});
            }
        }

        return PreservedAnalyses::all();
    }
};

char LoopRegionRemarkPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createLoopRegionRemarkPass() {
    return std::make_unique<LoopRegionRemarkPassImpl>();
}
}  // namespace stinkytofu
