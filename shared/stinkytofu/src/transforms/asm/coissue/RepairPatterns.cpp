// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RepairPatterns.hpp"

#include "MoveChecker.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu::coissue {
namespace {

bool readsScc(const TimedInst& t) {
    for (uint16_t slot : t.uses)
        if (regClassOfSlot(slot) == LatencyReg::Scc) return true;
    return false;
}

bool writesScc(const TimedInst& t) {
    for (uint16_t slot : t.defs)
        if (regClassOfSlot(slot) == LatencyReg::Scc) return true;
    return false;
}

// The cycles of the first cost rule for a bank switch right after an instruction of class
// `prev` (the timeline's first-match order).
int switchCostAfter(const TimingProfile& p, IssueClass prev) {
    for (const CostRule& r : p.costRules) {
        const bool inst = r.inst.opcode == GFX::s_set_vgpr_msb ||
                          (r.inst.opcode < 0 && (r.inst.cls == IssueClass::Any ||
                                                 r.inst.cls == IssueClass::Inserted));
        const bool after = r.after.opcode < 0 && (r.after.cls == IssueClass::Any || r.after.cls == prev);
        if (inst && after) return r.cycles;
    }
    return 1;
}

class HoistCompare : public RepairRule {
   public:
    const char* name() const override {
        return "hoist-compare";
    }
    bool appliesTo(const TimingProfile& profile) const override {
        for (const LatencyRule& r : profile.latencyRules)
            if (r.consumer == IssueClass::Branch &&
                (r.reg == LatencyReg::Scc || r.reg == LatencyReg::Any))
                return r.cycles > 2;
        return false;
    }
    void propose(const WindowDamage& damage, const BlockView& view,
                 std::vector<Move>& out) const override {
        const int win = damage.window;
        const size_t start = view.windowStart[win];
        const size_t next = win + 1 < view.windows() ? view.windowStart[win + 1] : view.order->size();
        for (size_t b = start + 2; b < next; ++b) {
            const TimedInst& branch = *view.timed[b];
            if (branch.isLabel || !branch.isBranch || !readsScc(branch)) continue;
            const size_t producer = b - 1;
            if (!writesScc(*view.timed[producer]) || !view.movable(producer)) continue;
            // Earlier windows with lead first come nearest first.
            for (int d = 1; d <= view.radius && win - d >= 0; ++d) {
                const int tw = win - d;
                if (view.damage != nullptr && view.damage->all[tw].lead <= 0) continue;
                out.push_back({name(), producer, view.target(tw, false), win, tw, "end"});
            }
        }
    }
};

class SwitchAfterLoad : public RepairRule {
   public:
    const char* name() const override {
        return "switch-after-load";
    }
    bool appliesTo(const TimingProfile& profile) const override {
        return switchCostAfter(profile, IssueClass::LdsLoad) >
               switchCostAfter(profile, IssueClass::Salu);
    }
    void propose(const WindowDamage& damage, const BlockView& view,
                 std::vector<Move>& out) const override {
        const int win = damage.window;
        const size_t start = view.windowStart[win];
        const size_t next = win + 1 < view.windows() ? view.windowStart[win + 1] : view.order->size();
        for (size_t v = start + 2; v < next; ++v) {
            const TimedInst& valu = *view.timed[v];
            if (valu.isLabel || valu.kind != IssueClass::Valu || !view.switchBefore[v]) continue;
            if (view.timed[v - 1]->isLabel || view.timed[v - 1]->kind != IssueClass::LdsLoad) continue;
            // A scalar of this window goes between the load and the switch.
            for (size_t s = start + 1; s < next; ++s) {
                if (s == v || view.timed[s]->isLabel || view.timed[s]->kind != IssueClass::Salu ||
                    !view.movable(s))
                    continue;
                out.push_back({name(), s, v, win, win, "after-load"});
            }
            // Or the VALU goes in front of the loads.
            size_t first = v - 1;
            while (first > start + 1 && !view.timed[first - 1]->isLabel &&
                   view.timed[first - 1]->kind == IssueClass::LdsLoad)
                --first;
            if (view.movable(v)) out.push_back({name(), v, first, win, win, "before-loads"});
        }
    }
};

}  // namespace

std::vector<std::string> registeredPatterns(GfxArchID arch) {
    // Keyed on the triple: a gfx1250v0-only build has no GfxArchID::Gfx1250.
    const auto* info = ArchHelper::getInstance().getArchInfo(arch);
    if (info != nullptr && info->major == 12 && info->minor == 5)
        return {"hoist-compare", "switch-after-load"};
    return {};
}

std::optional<std::string> enabledPatterns(GfxArchID arch, const std::string& spec,
                                           std::vector<std::unique_ptr<RepairRule>>& out) {
    const std::vector<std::string> registered = registeredPatterns(arch);
    std::vector<std::string> names;
    if (spec.empty()) {
        names = registered;
    } else if (spec != "none") {
        std::string cur;
        for (char c : spec + ",") {
            if (c == ' ') continue;
            if (c != ',') {
                cur.push_back(c);
                continue;
            }
            if (cur.empty()) return "CoissuePatterns: empty name in '" + spec + "'";
            names.push_back(cur);
            cur.clear();
        }
    }
    out.clear();
    for (const std::string& name : names) {
        bool known = false;
        for (const std::string& r : registered) known |= r == name;
        if (!known) return "CoissuePatterns: '" + name + "' is not a pattern registered for this arch";
        if (name == "hoist-compare") out.push_back(std::make_unique<HoistCompare>());
        if (name == "switch-after-load") out.push_back(std::make_unique<SwitchAfterLoad>());
    }
    return std::nullopt;
}

}  // namespace stinkytofu::coissue
