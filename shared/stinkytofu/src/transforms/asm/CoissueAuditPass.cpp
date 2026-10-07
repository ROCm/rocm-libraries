// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "coissue/IssueTimeline.hpp"
#include "coissue/LoopScope.hpp"
#include "coissue/TimingProfile.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/CoissueRepairPass.hpp"

namespace stinkytofu {
namespace {

using namespace coissue;

class CoissueAuditPassImpl : public Pass {
   public:
    static char ID;

    CoissueAuditPassImpl(bool printTimeline, std::string traceProfile)
        : printTimeline_(printTimeline), traceProfile_(std::move(traceProfile)) {}

    const char* getName() const override {
        return "CoissueAuditPass";
    }
    PassID getPassID() const override {
        return &CoissueAuditPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        if (!printTimeline_ && traceProfile_.empty()) return PreservedAnalyses::all();
        const auto archTriple = passCtx.getGemmTileConfig().arch;
        const GfxArchID arch = getGfxArchID(archTriple[0], archTriple[1], archTriple[2]);
        const HWModel& hw = passCtx.getHWModel();
        ProfileSet set;
        if (auto err = resolveProfileSet(hw, arch, passCtx.getPassFeatureConfig().coissue, set))
            report_fatal_error(*err);

        TimedInstCache cache(hw);
        for (const LoopScope& scope : innermostLoops(func, AM.getResult<LoopAnalysis>(func))) {
            std::vector<const TimedInst*> body;
            bool hasMatrix = false;
            for (StinkyInstruction* inst : loopInstructions(scope)) {
                const TimedInst& t = cache.get(*inst);
                hasMatrix |= !t.isLabel && t.kind == IssueClass::Matrix;
                body.push_back(&t);
            }
            if (!hasMatrix) continue;
            const std::string& label = scope.header->getLabel();
            for (const TimingProfile& p : set.profiles) {
                const TripTiming trip = steadyTrip(body, p);
                if (printTimeline_)
                    std::cout << "coissue-timeline " << label << " " << p.name << ": trip "
                              << trip.cycles << " idle " << trip.pipeIdle << " idle+handover "
                              << trip.pipeIdleWithHandover << " insts " << body.size() << "\n";
                if (p.name != traceProfile_) continue;
                for (size_t i = 0; i < body.size(); ++i) {
                    if (body[i]->isLabel) continue;
                    const Placement& pl = trip.placements[i];
                    std::cout << "coissue-trace " << label << " " << i << " " << pl.cycle << " w"
                              << pl.window << " +" << pl.pos << " "
                              << body[i]->inst->getHwInstDesc()->mnemonic << "\n";
                }
            }
        }
        return PreservedAnalyses::all();
    }

   private:
    bool printTimeline_;
    std::string traceProfile_;
};

char CoissueAuditPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createCoissueAuditPass(bool printTimeline, std::string traceProfile) {
    return std::make_unique<CoissueAuditPassImpl>(printTimeline, std::move(traceProfile));
}

}  // namespace stinkytofu
