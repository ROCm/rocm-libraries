// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CoissueRepairPass: reorders the SALU/VALU fillers of each innermost loop so that what the
// passes after it insert (bank switches, s_wait_alu, coexec nops, the prefetch bridge) costs
// less matrix-pipe time (WAITCNT_COISSUE.md section 7). One round:
//
//   order  = the loop as it stands: scheduler order + memory waits
//   costs  = for each profile of the set: insertion models -> timeline -> queue/sync -> cost
//   damage = windows that got longer than the plan, and VALUs that lost their slot
//   for each damaged window, worst first, for each core rule and enabled pattern:
//       skip moves the MoveChecker refuses, and moves that add inserted instructions
//       take the first move no profile gets slower under, and whose sum drops
//
// The search stops when a round takes nothing, or at the move cap. RepairPolicy then decides
// whether the order is written back (mode apply, a calibrated loop, a worst-case gain at
// least the margin), or only reported (mode shadow).

#include "stinkytofu/transforms/asm/CoissueRepairPass.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "coissue/AuditLog.hpp"
#include "coissue/DamageReport.hpp"
#include "coissue/InsertionModels.hpp"
#include "coissue/IssueTimeline.hpp"
#include "coissue/LoopScope.hpp"
#include "coissue/MoveChecker.hpp"
#include "coissue/RepairPatterns.hpp"
#include "coissue/RepairPolicy.hpp"
#include "coissue/RepairRules.hpp"
#include "coissue/TimingProfile.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

#define DEBUG_TYPE "CoissueRepairPass"

namespace stinkytofu {

PassFeatureConfig::CoissueFeatures coissueFeaturesFromModuleOptions(
    const StinkyAsmModule::ModuleOptions& options) {
    PassFeatureConfig::CoissueFeatures f;
    f.repairMode = options.CoissueRepairMode;
    f.marginPercent = options.CoissueMarginPercent;
    f.profileSet = options.CoissueProfileSet;
    f.maxMoves = options.CoissueMaxMoves;
    f.searchRadius = options.CoissueSearchRadius;
    f.trustUncalibrated = options.CoissueTrustUncalibrated;
    f.patterns = options.CoissuePatterns;
    f.waitcntIssueCycles = options.CoissueWaitcntIssueCycles;
    f.waitcntSettleCycles = options.CoissueWaitcntSettleCycles;
    f.issueCycles = options.CoissueIssueCycles;
    f.scalarLatency = options.CoissueScalarLatency;
    f.matrixQueueDepth = options.CoissueMatrixQueueDepth;
    f.esm2 = options.EnableESM2;
    f.esm2TrackValuVsrc = options.EnableESM2 && options.EnableESM2TrackValuVsrc;
    return f;
}

std::optional<std::string> validateCoissueFeatures(
    const PassFeatureConfig::CoissueFeatures& features, const std::array<int, 3>& arch) {
    if (!coissue::parseRepairMode(features.repairMode))
        return "CoissueRepairMode: '" + features.repairMode + "' is not off, shadow or apply";
    if (!(features.marginPercent >= 0.0))
        return "CoissueMarginPercent: must be >= 0, got " + std::to_string(features.marginPercent);
    if (features.maxMoves < 0) return "CoissueMaxMoves: must be >= 0";
    if (features.searchRadius < 0) return "CoissueSearchRadius: must be >= 0";
    if (features.waitcntIssueCycles < -1) return "CoissueWaitcntIssueCycles: must be >= -1";
    if (features.waitcntSettleCycles < -1) return "CoissueWaitcntSettleCycles: must be >= -1";
    if (features.matrixQueueDepth < -1) return "CoissueMatrixQueueDepth: must be >= -1";
    const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
    std::vector<std::unique_ptr<coissue::RepairRule>> patterns;
    if (auto err = coissue::enabledPatterns(archId, features.patterns, patterns)) return err;
    coissue::ProfileSet set;
    return coissue::resolveProfileSet(hwModelForArch(arch), archId, features, set);
}

std::vector<std::string> describeCoissueProfiles(const PassFeatureConfig::CoissueFeatures& features,
                                                 const std::array<int, 3>& arch) {
    const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
    coissue::ProfileSet set;
    if (auto err = coissue::resolveProfileSet(hwModelForArch(arch), archId, features, set))
        return {*err};
    std::vector<std::string> lines;
    for (const coissue::TimingProfile& p : set.profiles)
        lines.push_back(coissue::describe(p, archId));
    return lines;
}

namespace {

using namespace coissue;
using Order = std::vector<StinkyInstruction*>;

constexpr const char* kPassName = "CoissueRepairPass";

std::string list(const std::vector<int>& values) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) os << (i ? ", " : "") << values[i];
    os << "]";
    return os.str();
}

// The loop in layout order, across its blocks. Moves stay inside a block, so the block of
// every position, and the IR-node slot within it, are fixed.
struct LoopLayout {
    std::vector<BasicBlock*> blocks;
    std::vector<size_t> blockStart;
    std::vector<int> blockOf;
    std::vector<int> slotOf;
    Order order;
};

LoopLayout layoutOf(const LoopScope& scope) {
    LoopLayout l;
    l.blocks = scope.blocks;
    for (size_t b = 0; b < scope.blocks.size(); ++b) {
        l.blockStart.push_back(l.order.size());
        int slot = 0;
        for (IRBase& node : *scope.blocks[b]) {
            if (auto* inst = dyn_cast<StinkyInstruction>(&node)) {
                l.order.push_back(inst);
                l.blockOf.push_back(static_cast<int>(b));
                l.slotOf.push_back(slot);
            } else {
                ++slot;
            }
        }
    }
    l.blockStart.push_back(l.order.size());
    return l;
}

// Each block's sequence for the models, with exec-masked groups opened up again.
std::vector<std::vector<const StinkyInstruction*>> blockSequences(const LoopLayout& l,
                                                                  const Order& order) {
    std::vector<std::vector<const StinkyInstruction*>> seqs(l.blocks.size());
    for (size_t k = 0; k < order.size(); ++k) {
        auto& seq = seqs[l.blockOf[k]];
        if (isExecMaskGroup(*order[k])) {
            if (const auto* g = order[k]->getModifier<ExecGroupData>())
                for (const StinkyInstruction* child : g->children) seq.push_back(child);
        } else {
            seq.push_back(order[k]);
        }
    }
    return seqs;
}

size_t groupSize(const StinkyInstruction& inst) {
    if (!isExecMaskGroup(inst)) return 1;
    const auto* g = inst.getModifier<ExecGroupData>();
    return g != nullptr ? g->children.size() : 1;
}

struct Evaluation {
    bool limited = false;   ///< refused: adds inserted instructions
    bool rejected = false;  ///< slower than the bound under some profile
    std::vector<TripCost> costs;
    std::vector<int> cycles;
    InsertedCounts counts;
    TripTiming primary;
    std::vector<const TimedInst*> stream;
    /// Per order position: its stream index (first child for a group).
    std::vector<size_t> streamIndex;
    std::vector<Placement> finalByOrder;
    std::vector<bool> switchBefore;
    std::vector<bool> waitAluBefore;
};

class Evaluator {
   public:
    Evaluator(const LoopLayout& layout, InsertionPipeline& pipeline, TimedInstCache& cache,
              const ProfileSet& set, TimingProfile planProfile)
        : layout_(layout),
          pipeline_(pipeline),
          cache_(cache),
          set_(set),
          planProfile_(std::move(planProfile)),
          planTimeline_(planProfile_) {
        for (const TimingProfile& p : set_.profiles) timelines_.push_back(std::make_unique<IssueTimeline>(p));
    }

    // Time `order` under every profile. With `bound`, stop at the first profile it is slower
    // under; with `limit`, refuse it before timing if it adds inserted instructions.
    Evaluation evaluate(const Order& order, const std::vector<TripCost>* bound,
                        const InsertedCounts* limit, bool detail, Fidelity fidelity,
                        const StinkyInstruction* moved = nullptr) {
        Evaluation e;
        const PredictedBlock predicted =
            pipeline_.predict(blockSequences(layout_, order), fidelity, moved);
        e.counts = predicted.counts;
        if (limit != nullptr &&
            (e.counts.bankSwitches > limit->bankSwitches || e.counts.waitAlus > limit->waitAlus ||
             e.counts.nops > limit->nops)) {
            e.limited = true;
            return e;
        }
        for (const auto& seq : predicted.seqs)
            for (const StinkyInstruction* inst : seq) e.stream.push_back(&cache_.get(*inst));

        const size_t n = set_.profiles.size();
        e.costs.assign(n, TripCost{});
        e.cycles.assign(n, 0);
        // All profiles must pass, so try first the one that refused most often.
        if (refusals_.size() != n) refusals_.assign(n, 0);
        std::vector<size_t> evalOrder(n);
        std::iota(evalOrder.begin(), evalOrder.end(), 0);
        if (bound != nullptr)
            std::stable_sort(evalOrder.begin(), evalOrder.end(),
                             [&](size_t a, size_t b) { return refusals_[a] > refusals_[b]; });
        for (size_t p : evalOrder) {
            const auto t0 = std::chrono::steady_clock::now();
            const bool isPrimary = p == static_cast<size_t>(set_.primary);
            // The cost compares idle first, so more idle than the bound is enough to refuse.
            const std::optional<int> idleBound =
                bound != nullptr ? std::optional<int>((*bound)[p].pipeIdle) : std::nullopt;
            TripTiming trip =
                steadyTrip(e.stream, *timelines_[p], isPrimary && detail, kSteadyTrips, idleBound);
            ++timelineCalls;
            timelinePlacements += e.stream.size() * kSteadyTrips;
            timelineSeconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (trip.overBound) {
                ++refusals_[p];
                e.rejected = true;
                return e;
            }
            e.costs[p] = trip.cost();
            e.cycles[p] = trip.cycles;
            if (isPrimary) e.primary = std::move(trip);
            if (bound != nullptr && e.costs[p] > (*bound)[p]) {
                ++refusals_[p];
                e.rejected = true;
                return e;
            }
        }
        if (detail) mapToOrder(order, predicted, e);
        return e;
    }

    // The scheduler's plan of `order`: the compiler model with memory waits as zero-cost
    // holds, nothing inserted yet.
    TripTiming plan(const Order& order, std::vector<Placement>& byOrder) {
        std::vector<const TimedInst*> stream;
        std::vector<size_t> index(order.size());
        for (size_t k = 0; k < order.size(); ++k) {
            index[k] = stream.size();
            if (isExecMaskGroup(*order[k])) {
                if (const auto* g = order[k]->getModifier<ExecGroupData>())
                    for (const StinkyInstruction* child : g->children)
                        stream.push_back(&cache_.get(*child));
                continue;
            }
            const TimedInst& t = cache_.get(*order[k]);
            if (!t.isLabel && t.kind == IssueClass::Inserted) {
                index[k] = SIZE_MAX;
                continue;
            }
            stream.push_back(&t);
        }
        TripTiming trip = steadyTrip(stream, planTimeline_);
        byOrder.assign(order.size(), Placement{-1, -1, 0, 0});
        for (size_t k = 0; k < order.size(); ++k)
            if (index[k] != SIZE_MAX && index[k] < trip.placements.size())
                byOrder[k] = trip.placements[index[k]];
        return trip;
    }

    double timelineSeconds = 0.0;
    size_t timelineCalls = 0;
    size_t timelinePlacements = 0;

   private:
    // The models only add instructions, swap a bridged prefetch for its FLAT twin, and may
    // fold a hold-only s_wait_alu into a new one; the order's instructions appear in the
    // stream in order otherwise.
    void mapToOrder(const Order& order, const PredictedBlock& predicted, Evaluation& e) {
        const InstructionPool& pool = pipeline_.pool();
        auto inserted = [&](const StinkyInstruction* inst) {
            return pool.owns(inst) && pool.originalOf(inst) == nullptr;
        };
        auto firstOf = [&](size_t k) -> const StinkyInstruction* {
            if (isExecMaskGroup(*order[k]))
                if (const auto* g = order[k]->getModifier<ExecGroupData>())
                    if (!g->children.empty()) return g->children.front();
            return order[k];
        };
        e.streamIndex.assign(order.size(), SIZE_MAX);
        e.finalByOrder.assign(order.size(), Placement{-1, -1, 0, 0});
        e.switchBefore.assign(order.size(), false);
        e.waitAluBefore.assign(order.size(), false);
        size_t k = 0;
        size_t skip = 0;
        for (size_t s = 0; s < e.stream.size(); ++s) {
            const StinkyInstruction* inst = e.stream[s]->inst;
            if (inserted(inst)) continue;
            if (skip > 0) {
                --skip;
                continue;
            }
            const StinkyInstruction* canonical = pool.originalOf(inst) ? pool.originalOf(inst) : inst;
            while (k < order.size() && firstOf(k) != canonical) ++k;
            if (k == order.size()) break;
            e.streamIndex[k] = s;
            e.finalByOrder[k] = e.primary.placements[s];
            for (size_t q = s; q-- > 0;) {
                const StinkyInstruction* prev = e.stream[q]->inst;
                if (!inserted(prev)) break;
                if (q + 1 == s && prev->getUnifiedOpcode() == GFX::s_set_vgpr_msb)
                    e.switchBefore[k] = true;
                if (prev->getUnifiedOpcode() == GFX::s_wait_alu) e.waitAluBefore[k] = true;
            }
            skip = groupSize(*order[k]) - 1;
            ++k;
        }
        (void)predicted;
    }

    const LoopLayout& layout_;
    InsertionPipeline& pipeline_;
    TimedInstCache& cache_;
    const ProfileSet& set_;
    TimingProfile planProfile_;
    IssueTimeline planTimeline_;
    std::vector<std::unique_ptr<IssueTimeline>> timelines_;
    std::vector<size_t> refusals_;
};

// The prototype's stand-in for the s_wait_alu model (CoissueRepairOptions::
// prototypeWaitAluRule): `x` now sits at `pos` of `order`.
bool prototypeWaitAluSafe(const Order& order, size_t pos, TimedInstCache& cache) {
    constexpr int kVaVdstHide = 13;
    constexpr int kVmVsrcHide = 11;
    const TimedInst& x = cache.get(*order[pos]);
    if (x.isLabel || x.kind != IssueClass::Valu) return true;
    auto readsResult = [&](const TimedInst& y) {
        if (y.isLabel || (y.kind != IssueClass::Memory && y.kind != IssueClass::LdsLoad &&
                          y.kind != IssueClass::LdsStore))
            return false;
        for (uint16_t d : x.defs)
            if (std::find(y.uses.begin(), y.uses.end(), d) != y.uses.end()) return true;
        return false;
    };
    int matrix = 0;
    for (size_t k = pos + 1; k < order.size(); ++k) {
        const TimedInst& y = cache.get(*order[k]);
        if (!y.isLabel && y.kind == IssueClass::Matrix && ++matrix >= kVaVdstHide) break;
        if (readsResult(y)) return false;
    }
    matrix = 0;
    for (size_t k = pos; k-- > 0;) {
        const TimedInst& y = cache.get(*order[k]);
        if (!y.isLabel && y.kind == IssueClass::Matrix && ++matrix >= kVmVsrcHide) break;
        if (readsResult(y)) return false;
    }
    return true;
}

TripCost sum(const std::vector<TripCost>& costs) {
    TripCost total;
    for (const TripCost& c : costs) total = total + c;
    return total;
}

Order moved(const Order& order, size_t from, size_t to) {
    Order out = order;
    StinkyInstruction* inst = out[from];
    out.erase(out.begin() + static_cast<long>(from));
    out.insert(out.begin() + static_cast<long>(to < from ? to : to - 1), inst);
    return out;
}

// Put `order` back into the loop's blocks, keeping every non-instruction IR node in its slot.
void writeBack(const LoopLayout& l, const Order& order) {
    for (size_t b = 0; b < l.blocks.size(); ++b) {
        BasicBlock& bb = *l.blocks[b];
        std::vector<IRBase*> nodes;
        for (IRBase& node : bb) nodes.push_back(&node);
        size_t next = l.blockStart[b];
        for (IRBase*& node : nodes)
            if (dyn_cast<StinkyInstruction>(node) != nullptr) node = order[next++];
        for (IRBase* node : nodes) bb.removeIR(node);
        for (IRBase* node : nodes) bb.appendIR(node);
    }
}

class CoissueRepairPassImpl : public Pass {
   public:
    static char ID;
    explicit CoissueRepairPassImpl(CoissueRepairOptions options) : options_(std::move(options)) {}

    const char* getName() const override {
        return kPassName;
    }
    PassID getPassID() const override {
        return &CoissueRepairPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        PassFeatureConfig::CoissueFeatures f = passCtx.getPassFeatureConfig().coissue;
        if (options_.mode) f.repairMode = *options_.mode;
        if (options_.marginPercent) f.marginPercent = *options_.marginPercent;
        if (options_.maxMoves) f.maxMoves = *options_.maxMoves;
        if (options_.searchRadius) f.searchRadius = *options_.searchRadius;
        if (options_.profileSet) f.profileSet = *options_.profileSet;
        if (options_.patterns) f.patterns = *options_.patterns;
        if (options_.trustUncalibrated) f.trustUncalibrated = *options_.trustUncalibrated;
        if (options_.esm2) f.esm2 = *options_.esm2;
        if (options_.trackValuVsrc) f.esm2TrackValuVsrc = *options_.trackValuVsrc;

        const auto triple = passCtx.getGemmTileConfig().arch;
        if (auto err = validateCoissueFeatures(f, triple)) report_fatal_error(*err);
        const RepairMode mode = *parseRepairMode(f.repairMode);
        if (mode == RepairMode::Off) return PreservedAnalyses::all();
        const GfxArchID arch = getGfxArchID(triple[0], triple[1], triple[2]);
        const HWModel& hw = passCtx.getHWModel();

        ProfileSet set;
        (void)resolveProfileSet(hw, arch, f, set);
        std::vector<std::unique_ptr<RepairRule>> rules = coreRules();
        std::vector<std::unique_ptr<RepairRule>> patterns;
        (void)enabledPatterns(arch, f.patterns, patterns);
        const TimingProfile& primary = set.profiles[set.primary];
        for (auto& p : patterns)
            if (p->appliesTo(primary)) rules.push_back(std::move(p));

        for (const TimingProfile& p : set.profiles)
            emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName, "Profile",
                                 "profile " + describe(p, arch)});

        bool changed = false;
        for (const LoopScope& scope : innermostLoops(func, AM.getResult<LoopAnalysis>(func)))
            changed |= repairLoop(func, scope, passCtx, f, mode, set, rules, arch);
        return changed ? preserveCFGAnalyses() : PreservedAnalyses::all();
    }

   private:
    bool repairLoop(Function& func, const LoopScope& scope, PassContext& passCtx,
                    const PassFeatureConfig::CoissueFeatures& f, RepairMode mode,
                    const ProfileSet& set, const std::vector<std::unique_ptr<RepairRule>>& rules,
                    GfxArchID arch) {
        bool hasMatrix = false;
        for (BasicBlock* bb : scope.blocks)
            for (IRBase& node : *bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node))
                    hasMatrix |= isMatrixInstruction(*inst);
        if (!hasMatrix) return false;
        const auto t0 = std::chrono::steady_clock::now();
        const std::string& label = scope.header->getLabel();

        // The models read the IR as it stands, before exec-masked spans are collapsed.
        InsertionConfig config;
        config.msbMode = passCtx.getAsmCapsConfig().vgprMsbMode;
        config.esm2 = f.esm2;
        config.waitAlu = gfx1250InsertWaitAluOptions(f.esm2TrackValuVsrc);
        std::vector<const BasicBlock*> constScope(scope.blocks.begin(), scope.blocks.end());
        InsertionPipeline pipeline(func, constScope, passCtx, config);
        for (BasicBlock* bb : scope.blocks) {
            AsmIRBuilder builder(*bb, arch);
            collapseExecMaskedRegions(*bb, builder, passCtx.getWavefrontSize());
        }

        LoopLayout layout = layoutOf(scope);
        Order order = layout.order;
        TimedInstCache cache(passCtx.getHWModel());
        // Only a scalar or vector result can delay a consumer in final code (the memory
        // results are covered by the waits), so only their registers need checking.
        {
            std::vector<bool> gating(kNumRegSlots, false);
            auto mark = [&](const StinkyInstruction& inst) {
                const TimedInst& t = cache.get(inst);
                if (t.isLabel || (t.kind != IssueClass::Salu && t.kind != IssueClass::Valu)) return;
                for (uint16_t slot : t.defs) gating[slot] = true;
            };
            for (const StinkyInstruction* inst : order) {
                if (const auto* g = isExecMaskGroup(*inst) ? inst->getModifier<ExecGroupData>() : nullptr)
                    for (const StinkyInstruction* child : g->children) mark(*child);
                else
                    mark(*inst);
            }
            cache.restrictUses(std::move(gating));
        }
        TimingProfile planProfile = compilerProfile(passCtx.getHWModel());
        planProfile.name = "plan";
        planProfile.waitcntIssueCycles = 0;
        Evaluator ev(layout, pipeline, cache, set, planProfile);
        MoveChecker checker(order, layout.blockOf, layout.slotOf, cache);

        Evaluation current = ev.evaluate(order, nullptr, nullptr, true, Fidelity::ExactBase);
        const InsertedCounts limit = current.counts;
        // An s_wait_alu stays in front of the instruction it guards, so that one stays put.
        for (size_t k = 0; k < order.size(); ++k)
            if (current.waitAluBefore[k]) checker.pin(order[k]);
        std::vector<TripCost> best = current.costs;
        const std::vector<int> startCycles = current.cycles;

        // The calibrated scope is decided on the primary profile's list of measured forms.
        std::vector<CalibratedForm> forms;
        for (const StinkyInstruction* inst : order) {
            const TimedInst& t = cache.get(*inst);
            if (t.isLabel || t.kind != IssueClass::Matrix) continue;
            const CalibratedForm form{t.opcode, t.fp4Operands};
            bool seen = false;
            for (const CalibratedForm& g : forms)
                seen |= g.opcode == form.opcode && g.fp4Operands == form.fp4Operands;
            if (!seen) forms.push_back(form);
        }

        size_t moves = 0;
        size_t tried = 0;
        std::vector<Move> candidates;
        while (static_cast<int>(moves) < f.maxMoves) {
            std::vector<const TimedInst*> timedOrder;
            for (StinkyInstruction* inst : order) timedOrder.push_back(&cache.get(*inst));
            std::vector<Placement> planByOrder;
            const TripTiming planTrip = ev.plan(order, planByOrder);
            const DamageReport report =
                buildDamageReport(timedOrder, planByOrder, current.finalByOrder, planTrip,
                                  current.primary, DamageTrigger::IssueGrowth);
            BlockView view;
            view.order = &order;
            view.timed = timedOrder;
            for (size_t k = 0; k < order.size(); ++k)
                if (!timedOrder[k]->isLabel && timedOrder[k]->kind == IssueClass::Matrix)
                    view.windowStart.push_back(k);
            view.switchBefore = current.switchBefore;
            view.checker = &checker;
            view.radius = f.searchRadius;
            view.damage = &report;

            bool accepted = false;
            for (const WindowDamage* damage : report.damaged) {
                for (const auto& rule : rules) {
                    candidates.clear();
                    rule->propose(*damage, view, candidates);
                    for (const Move& m : candidates) {
                        if (!checker.legal(order, m.from, m.to)) continue;
                        Order next = moved(order, m.from, m.to);
                        if (options_.prototypeWaitAluRule &&
                            !prototypeWaitAluSafe(next, m.to < m.from ? m.to : m.to - 1, cache))
                            continue;
                        // Screen with the current order's s_wait_alu, then confirm exactly.
                        const StinkyInstruction* movedInst = order[m.from];
                        Evaluation e =
                            ev.evaluate(next, &best, &limit, false, Fidelity::Screen, movedInst);
                        if (e.limited) continue;
                        ++tried;
                        if (e.rejected || !(sum(e.costs) < sum(best))) continue;
                        e = ev.evaluate(next, &best, &limit, false, Fidelity::Exact, movedInst);
                        if (e.limited || e.rejected || !(sum(e.costs) < sum(best))) continue;
                        std::vector<int> gains;
                        for (size_t p = 0; p < e.cycles.size(); ++p)
                            gains.push_back(current.cycles[p] - e.cycles[p]);
                        ++moves;
                        emitRemark(passCtx,
                                   {OptimizationRemark::Kind::Passed, kPassName, "Move",
                                    "loop " + label + " move " + std::to_string(moves) + ": " +
                                        m.rule + " window " + std::to_string(m.fromWindow) +
                                        " -> " + std::to_string(m.toWindow) + " (" + m.place +
                                        ") gains " + list(gains) + "  " +
                                        toAssembly(*order[m.from])});
                        const int block = layout.blockOf[m.from];
                        order = std::move(next);
                        checker.update(order, block);
                        current = ev.evaluate(order, nullptr, nullptr, true, Fidelity::ExactBase);
                        best = current.costs;
                        accepted = true;
                        break;
                    }
                    if (accepted) break;
                }
                if (accepted) break;
            }
            if (!accepted) break;
        }

        PolicyInput in;
        in.mode = mode;
        in.marginPercent = f.marginPercent;
        in.trustUncalibrated = f.trustUncalibrated;
        in.calibrated = set.profiles[set.primary].covers(forms);
        in.moves = moves;
        for (size_t p = 0; p < startCycles.size(); ++p) in.gains.push_back(startCycles[p] - current.cycles[p]);
        in.loopCycles = startCycles[set.primary];
        const PolicyDecision decision = decide(in);
        if (decision.writeBack) writeBack(layout, order);

        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::ostringstream os;
        os << "loop " << label << ": moves " << moves << "  gains " << list(in.gains) << "  worst "
           << decision.worstGain << " (" << std::fixed << std::setprecision(2)
           << decision.worstPercent << "%)  -> " << decision.reason << "; before "
           << list(startCycles) << " after " << list(current.cycles) << ", " << tried
           << " candidates, " << std::setprecision(1) << ms << " ms";
        emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName, "Loop", os.str()});
        PASS_DEBUG({
            std::cerr << "[CoissueRepair] " << label << ": timeline " << ev.timelineSeconds << " s ("
                      << ev.timelineCalls << " trips, at most " << ev.timelinePlacements
                      << " placements)";
            for (size_t m = 0; m < pipeline.models().size(); ++m)
                std::cerr << ", " << pipeline.models()[m]->name() << " "
                          << pipeline.modelSeconds()[m] << " s " << pipeline.models()[m]->stats();
            std::cerr << "\n";
        });

        if (options_.audit) {
            // What the later passes should leave: the written-back order, or the IR's own.
            const Order& inIR = decision.writeBack ? order : layout.order;
            const PredictedBlock predicted =
                pipeline.predict(blockSequences(layout, inIR), Fidelity::Exact);
            PredictedLoop record;
            for (const auto& seq : predicted.seqs)
                for (const StinkyInstruction* inst : seq) record.signatures.push_back(auditSignature(*inst));
            recordPrediction(func.getName(), label, std::move(record));
        }

        for (BasicBlock* bb : scope.blocks) expandExecMaskedGroups(*bb);
        return true;
    }

    CoissueRepairOptions options_;
};

char CoissueRepairPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createCoissueRepairPass(CoissueRepairOptions options) {
    return std::make_unique<CoissueRepairPassImpl>(std::move(options));
}

}  // namespace stinkytofu
