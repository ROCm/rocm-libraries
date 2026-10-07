// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <sstream>

#include "coissue/AuditLog.hpp"
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
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/asm/CoissueRepairPass.hpp"

namespace stinkytofu {
namespace {

using namespace coissue;

class CoissueAuditPassImpl : public Pass {
   public:
    static char ID;

    CoissueAuditPassImpl(bool printTimeline, std::string traceProfile, bool compare)
        : printTimeline_(printTimeline),
          traceProfile_(std::move(traceProfile)),
          compare_(compare) {}

    const char* getName() const override {
        return "CoissueAuditPass";
    }
    PassID getPassID() const override {
        return &CoissueAuditPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        if (compare_) compareWithPredictions(func, passCtx, AM);
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
    // What the inserting passes left in each loop, against what the repair predicted.
    void compareWithPredictions(Function& func, PassContext& passCtx, AnalysisManager& AM) {
        for (const LoopScope& scope : innermostLoops(func, AM.getResult<LoopAnalysis>(func))) {
            const std::string& label = scope.header->getLabel();
            const std::optional<PredictedLoop> predicted = takePrediction(func.getName(), label);
            if (!predicted) continue;
            std::vector<std::string> actual;
            for (const StinkyInstruction* inst : loopInstructions(scope))
                actual.push_back(auditSignature(*inst));
            const auto count = [](const std::vector<std::string>& sigs, const char* prefix) {
                size_t n = 0;
                for (const std::string& s : sigs) n += s.rfind(prefix, 0) == 0 ? 1 : 0;
                return n;
            };
            std::ostringstream os;
            os << "audit loop " << label << ":";
            bool first = true;
            for (const char* kind : {"s_set_vgpr_msb", "s_wait_alu", "s_nop", "v_nop",
                                     "flat_prefetch_b8", "s_delay_alu"}) {
                os << (first ? " " : ", ") << kind << " " << count(predicted->signatures, kind)
                   << "/" << count(actual, kind);
                first = false;
            }
            size_t mismatch = 0;
            const size_t n = std::max(actual.size(), predicted->signatures.size());
            size_t firstAt = n;
            for (size_t i = 0; i < n; ++i) {
                const std::string* p =
                    i < predicted->signatures.size() ? &predicted->signatures[i] : nullptr;
                const std::string* a = i < actual.size() ? &actual[i] : nullptr;
                if (p && a && *p == *a) continue;
                ++mismatch;
                if (firstAt == n) firstAt = i;
            }
            os << " (predicted/actual); " << mismatch << " mismatched positions";
            if (firstAt < n)
                os << ", first at " << firstAt << ": predicted '"
                   << (firstAt < predicted->signatures.size() ? predicted->signatures[firstAt] : "")
                   << "', actual '" << (firstAt < actual.size() ? actual[firstAt] : "") << "'";
            emitRemark(passCtx, {mismatch == 0 ? OptimizationRemark::Kind::Analysis
                                               : OptimizationRemark::Kind::Missed,
                                 "CoissueAuditPass", "Audit", os.str()});
        }
    }

    bool printTimeline_;
    std::string traceProfile_;
    bool compare_;
};

char CoissueAuditPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createCoissueAuditPass(bool printTimeline, std::string traceProfile,
                                             bool compare) {
    return std::make_unique<CoissueAuditPassImpl>(printTimeline, std::move(traceProfile), compare);
}

}  // namespace stinkytofu
