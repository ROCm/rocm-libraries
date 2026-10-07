// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TimingProfile.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

namespace stinkytofu::coissue {
namespace {

int opcodeOfMnemonic(const std::string& mnemonic, GfxArchID arch) {
    const ArchHelper::ArchInfo* info = ArchHelper::getInstance().getArchInfo(arch);
    if (info == nullptr) return -1;
    const auto& map = info->getMnemonicToIsaOpcodeMap();
    const auto it = map.find(mnemonic);
    if (it == map.end()) return -1;
    const HwInstDesc* desc = getMCIDByIsaOp(it->second, arch);
    return desc != nullptr ? static_cast<int>(desc->unifiedOpcode) : -1;
}

const char* mnemonicOfOpcode(int opcode, GfxArchID arch) {
    const HwInstDesc* desc = getMCIDByUOp(static_cast<GFX>(opcode), arch);
    return desc != nullptr && desc->mnemonic != nullptr ? desc->mnemonic : "?";
}

OpMatch resolve(const IssueMatch& m, GfxArchID arch) {
    OpMatch out{m.cls, -1};
    if (m.mnemonic != nullptr) out.opcode = opcodeOfMnemonic(m.mnemonic, arch);
    return out;
}

std::vector<HazardGap> hazardGapsOf(const HWModel& hw) {
    std::vector<HazardGap> gaps;
    for (int i = 0; i < hw.hazards.numRules; ++i) {
        const HazardRule& rule = hw.hazards.rules[i];
        if (rule.unit != HazardUnit::Cycles || rule.dir != HazardDir::WriteThenRead ||
            rule.distance <= 0)
            continue;
        const LatencyReg reg = rule.regType == RegType::S   ? LatencyReg::Sgpr
                               : rule.regType == RegType::V ? LatencyReg::Vgpr
                                                            : LatencyReg::Any;
        gaps.push_back({i, rule.distance, reg, rule.name});
    }
    return gaps;
}

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> splitEntries(const std::string& spec) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : spec) {
        if (c == ';' || c == ',') {
            out.push_back(trim(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(trim(cur));
    // A trailing separator leaves one empty entry, which is fine; an empty entry in the
    // middle is a typo.
    if (!out.empty() && out.back().empty()) out.pop_back();
    return out;
}

std::optional<IssueClass> parseClass(const std::string& name) {
    static const std::pair<const char*, IssueClass> kNames[] = {
        {"any", IssueClass::Any},
        {"*", IssueClass::Any},
        {"matrix", IssueClass::Matrix},
        {"valu", IssueClass::Valu},
        {"salu", IssueClass::Salu},
        {"branch", IssueClass::Branch},
        {"lds", IssueClass::LdsLoad},
        {"ds_load", IssueClass::LdsLoad},
        {"lds_store", IssueClass::LdsStore},
        {"ds_store", IssueClass::LdsStore},
        {"memory", IssueClass::Memory},
        {"wait", IssueClass::MemWait},
        {"barrier", IssueClass::Barrier},
        {"inserted", IssueClass::Inserted},
    };
    for (const auto& [n, c] : kNames)
        if (name == n) return c;
    return std::nullopt;
}

std::optional<int> parseCycles(const std::string& text) {
    if (text.empty() || text.size() > 6) return std::nullopt;
    for (char c : text)
        if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
    return std::stoi(text);
}

std::optional<std::string> parseMatch(const std::string& text, GfxArchID arch, OpMatch& out) {
    if (text.empty()) return std::string("empty instruction");
    if (auto cls = parseClass(text)) {
        out = {*cls, -1};
        return std::nullopt;
    }
    const int opcode = opcodeOfMnemonic(text, arch);
    if (opcode < 0) return "'" + text + "' is neither an instruction class nor a mnemonic";
    out = {IssueClass::Any, opcode};
    return std::nullopt;
}

std::string matchName(const OpMatch& m, GfxArchID arch) {
    if (m.opcode >= 0) return mnemonicOfOpcode(m.opcode, arch);
    return issueClassName(m.cls);
}

const char* regName(LatencyReg reg) {
    switch (reg) {
        case LatencyReg::Any:
            return "any";
        case LatencyReg::Scc:
            return "scc";
        case LatencyReg::Vcc:
            return "vcc";
        case LatencyReg::Sgpr:
            return "sgpr";
        case LatencyReg::Vgpr:
            return "vgpr";
    }
    return "?";
}

TimingProfile withName(TimingProfile p, const char* name) {
    p.name = name;
    return p;
}

}  // namespace

const char* issueClassName(IssueClass cls) {
    switch (cls) {
        case IssueClass::Any:
            return "any";
        case IssueClass::Matrix:
            return "matrix";
        case IssueClass::Valu:
            return "valu";
        case IssueClass::Salu:
            return "salu";
        case IssueClass::Branch:
            return "branch";
        case IssueClass::LdsLoad:
            return "lds";
        case IssueClass::LdsStore:
            return "lds_store";
        case IssueClass::Memory:
            return "memory";
        case IssueClass::MemWait:
            return "wait";
        case IssueClass::Barrier:
            return "barrier";
        case IssueClass::Inserted:
            return "inserted";
    }
    return "?";
}

bool TimingProfile::covers(const std::vector<CalibratedForm>& forms) const {
    if (forms.empty()) return false;
    return std::all_of(forms.begin(), forms.end(), [&](const CalibratedForm& f) {
        return std::any_of(calibrated.begin(), calibrated.end(), [&](const CalibratedForm& c) {
            return c.opcode == f.opcode && c.fp4Operands == f.fp4Operands;
        });
    });
}

TimingProfile compilerProfile(const HWModel& hw) {
    TimingProfile p;
    p.name = "compiler";
    p.hazardGaps = hazardGapsOf(hw);
    return p;
}

TimingProfile schedulerProfile(const HWModel& hw) {
    TimingProfile p = compilerProfile(hw);
    p.name = "scheduler";
    p.latencyFromAllProducers = true;
    p.gapsFromIssueEnd = true;
    return p;
}

TimingProfile measuredProfile(const HWModel& hw, GfxArchID arch) {
    TimingProfile p;
    p.name = "measured";
    p.matrixIssueCycles = hw.issue.matrixIssueCycles;
    p.blockedCycleAtIssue = hw.issue.blockedCycleAtIssue;
    p.waitcntIssueCycles = hw.issue.waitcntIssueCycles;
    p.waitcntSettleCycles = hw.issue.waitcntSettleCycles;
    for (const IssueCostRule& r : hw.issue.costRules)
        p.costRules.push_back({resolve(r.inst, arch), resolve(r.after, arch), r.cycles});
    p.latencyRules.assign(hw.latencyRules.begin(), hw.latencyRules.end());
    p.hazardGaps = hazardGapsOf(hw);
    p.matrixQueueDepth = hw.matrixQueue.depth;
    p.sync = SyncModel::Conservative;
    for (const CalibratedMatrixForm& f : hw.calibratedMatrixForms)
        p.calibrated.push_back({opcodeOfMnemonic(f.mnemonic, arch), f.fp4Operands});
    return p;
}

std::optional<std::string> parseIssueCycles(const std::string& spec, GfxArchID arch,
                                            std::vector<CostRule>& out) {
    std::vector<CostRule> rules;
    for (const std::string& entry : splitEntries(spec)) {
        const size_t eq = entry.find('=');
        if (eq == std::string::npos) return "CoissueIssueCycles: '" + entry + "' has no '='";
        const std::string lhs = trim(entry.substr(0, eq));
        const auto cycles = parseCycles(trim(entry.substr(eq + 1)));
        if (!cycles) return "CoissueIssueCycles: bad cycle count in '" + entry + "'";
        CostRule rule;
        rule.cycles = *cycles;
        const size_t at = lhs.find('@');
        const std::string inst = trim(lhs.substr(0, at));
        if (auto err = parseMatch(inst, arch, rule.inst)) return "CoissueIssueCycles: " + *err;
        if (at != std::string::npos) {
            if (auto err = parseMatch(trim(lhs.substr(at + 1)), arch, rule.after))
                return "CoissueIssueCycles: " + *err;
        }
        rules.push_back(rule);
    }
    out = std::move(rules);
    return std::nullopt;
}

std::optional<std::string> parseScalarLatency(const std::string& spec,
                                              std::vector<LatencyRule>& out) {
    std::vector<LatencyRule> rules;
    for (const std::string& entry : splitEntries(spec)) {
        const size_t eq = entry.find('=');
        const size_t gt = entry.find('>');
        if (eq == std::string::npos || gt == std::string::npos || gt > eq)
            return "CoissueScalarLatency: '" + entry + "' is not <producer>><consumer>=<cycles>";
        const auto cycles = parseCycles(trim(entry.substr(eq + 1)));
        if (!cycles) return "CoissueScalarLatency: bad cycle count in '" + entry + "'";
        std::string consumer = trim(entry.substr(gt + 1, eq - gt - 1));
        LatencyReg reg = LatencyReg::Any;
        const size_t colon = consumer.find(':');
        if (colon != std::string::npos) {
            const std::string r = trim(consumer.substr(colon + 1));
            consumer = trim(consumer.substr(0, colon));
            if (r == "scc")
                reg = LatencyReg::Scc;
            else if (r == "vcc")
                reg = LatencyReg::Vcc;
            else if (r == "sgpr")
                reg = LatencyReg::Sgpr;
            else if (r == "vgpr")
                reg = LatencyReg::Vgpr;
            else if (r == "any" || r == "*")
                reg = LatencyReg::Any;
            else
                return "CoissueScalarLatency: unknown register class '" + r + "'";
        }
        const auto producer = parseClass(trim(entry.substr(0, gt)));
        const auto consumerCls = parseClass(consumer);
        if (!producer || !consumerCls)
            return "CoissueScalarLatency: unknown instruction class in '" + entry + "'";
        rules.push_back({*producer, *consumerCls, reg, *cycles});
    }
    out = std::move(rules);
    return std::nullopt;
}

std::optional<std::string> applyKnobs(const PassFeatureConfig::CoissueFeatures& features,
                                      GfxArchID arch, TimingProfile& profile) {
    if (features.waitcntIssueCycles >= 0) profile.waitcntIssueCycles = features.waitcntIssueCycles;
    if (features.waitcntSettleCycles >= 0)
        profile.waitcntSettleCycles = features.waitcntSettleCycles;
    if (features.matrixQueueDepth >= 0) profile.matrixQueueDepth = features.matrixQueueDepth;
    std::vector<CostRule> costRules;
    if (auto err = parseIssueCycles(features.issueCycles, arch, costRules)) return err;
    profile.costRules.insert(profile.costRules.begin(), costRules.begin(), costRules.end());
    std::vector<LatencyRule> latencyRules;
    if (auto err = parseScalarLatency(features.scalarLatency, latencyRules)) return err;
    profile.latencyRules.insert(profile.latencyRules.begin(), latencyRules.begin(),
                                latencyRules.end());
    return std::nullopt;
}

std::optional<std::string> resolveProfileSet(const HWModel& hw, GfxArchID arch,
                                             const PassFeatureConfig::CoissueFeatures& features,
                                             ProfileSet& out) {
    TimingProfile measured = measuredProfile(hw, arch);
    if (auto err = applyKnobs(features, arch, measured)) return err;

    // Each variant pins the one fact it questions; the other knobs carry over.
    TimingProfile queue2 = withName(measured, "measured-queue2");
    queue2.matrixQueueDepth = 2;
    TimingProfile queue4 = withName(measured, "measured-queue4");
    queue4.matrixQueueDepth = 4;
    TimingProfile flatWaits = withName(measured, "measured-flat-waits");
    flatWaits.waitcntIssueCycles = 3;
    flatWaits.waitcntSettleCycles = 0;
    const TimingProfile known[] = {compilerProfile(hw), measured, queue2, queue4, flatWaits};

    ProfileSet set;
    const std::string spec = trim(features.profileSet);
    if (spec == "robust") {
        set.profiles.assign(std::begin(known), std::end(known));
        set.primary = 1;
    } else {
        std::string cur;
        std::vector<std::string> names;
        for (char c : spec + "+") {
            if (c != '+') {
                cur.push_back(c);
                continue;
            }
            names.push_back(trim(cur));
            cur.clear();
        }
        for (const std::string& name : names) {
            auto it = std::find_if(std::begin(known), std::end(known),
                                   [&](const TimingProfile& p) { return p.name == name; });
            if (it == std::end(known))
                return "CoissueProfileSet: unknown profile '" + name +
                       "' (robust, compiler, measured, measured-queue2, measured-queue4, "
                       "measured-flat-waits)";
            set.profiles.push_back(*it);
        }
        // The measured facts drive the damage report and the scope check when present.
        set.primary = 0;
        for (size_t i = 0; i < set.profiles.size(); ++i) {
            if (set.profiles[i].name == "measured") {
                set.primary = static_cast<int>(i);
                break;
            }
        }
    }
    out = std::move(set);
    return std::nullopt;
}

std::string describe(const TimingProfile& p, GfxArchID arch) {
    std::ostringstream os;
    os << p.name << ": matrix issue "
       << (p.matrixIssueCycles > 0 ? std::to_string(p.matrixIssueCycles) : std::string("table"))
       << ", blocked at issue " << (p.blockedCycleAtIssue ? "yes" : "no") << ", waitcnt "
       << p.waitcntIssueCycles << " issue + " << p.waitcntSettleCycles << " settle, queue "
       << p.matrixQueueDepth << ", sync "
       << (p.sync == SyncModel::Conservative ? "conservative" : "none");
    os << ", cost rules [";
    for (size_t i = 0; i < p.costRules.size(); ++i) {
        const CostRule& r = p.costRules[i];
        os << (i ? "; " : "") << matchName(r.inst, arch);
        if (r.after.opcode >= 0 || r.after.cls != IssueClass::Any)
            os << "@" << matchName(r.after, arch);
        os << "=" << r.cycles;
    }
    os << "], latency [";
    for (size_t i = 0; i < p.latencyRules.size(); ++i) {
        const LatencyRule& r = p.latencyRules[i];
        os << (i ? "; " : "") << issueClassName(r.producer) << ">" << issueClassName(r.consumer);
        if (r.reg != LatencyReg::Any) os << ":" << regName(r.reg);
        os << "=" << r.cycles;
    }
    os << "], hazards [";
    for (size_t i = 0; i < p.hazardGaps.size(); ++i)
        os << (i ? "; " : "") << p.hazardGaps[i].name << "=" << p.hazardGaps[i].cycles;
    os << "], calibrated [";
    for (size_t i = 0; i < p.calibrated.size(); ++i) {
        os << (i ? "; " : "") << mnemonicOfOpcode(p.calibrated[i].opcode, arch);
        if (p.calibrated[i].fp4Operands) os << " (fp4)";
    }
    os << "]";
    if (p.latencyFromAllProducers) os << ", memory latency gates";
    if (p.gapsFromIssueEnd) os << ", gaps from issue end";
    return os.str();
}

}  // namespace stinkytofu::coissue
