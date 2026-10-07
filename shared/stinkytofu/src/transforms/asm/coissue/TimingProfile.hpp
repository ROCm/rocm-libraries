// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The timing numbers of the co-issue timeline, resolved from three layers:
//   1. instruction facts: HwInstDesc issue/latency cycles and matrix windows, read by
//      the timeline from each instruction;
//   2. architecture facts: HWModel::issue, matrixQueue, latencyRules, hazards and the
//      calibrated matrix-op forms;
//   3. knobs: PassFeatureConfig::CoissueFeatures (module options CoissueWaitcnt*,
//      CoissueIssueCycles, CoissueScalarLatency, CoissueMatrixQueueDepth).
// A later layer overrides an earlier one. The pass prints the resolved profiles as a
// remark, so every run shows the numbers it used.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/hardware/HWModel.hpp"

namespace stinkytofu::coissue {

/// An instruction by class, or by unified opcode when opcode >= 0.
struct OpMatch {
    IssueClass cls = IssueClass::Any;
    int opcode = -1;
};

/// IssueCostRule with its mnemonics resolved to opcodes.
struct CostRule {
    OpMatch inst;
    OpMatch after;
    int cycles = 0;
};

/// A cycle-distance hazard of HWModel::hazards the timeline enforces.
struct HazardGap {
    int ruleIndex = 0;
    int cycles = 0;
    const char* name = "";
};

/// What a barrier wait does to the matrix queue.
enum class SyncModel : uint8_t {
    /// A barrier costs its latency; queued matrix work runs on behind it.
    None,
    /// An s_barrier_wait also waits until the queued matrix work has finished, so every
    /// compute change before a barrier counts in full.
    Conservative,
};

struct CalibratedForm {
    int opcode = -1;
    bool fp4Operands = false;
};

struct TimingProfile {
    std::string name;
    int matrixIssueCycles = 0;
    bool blockedCycleAtIssue = true;
    int waitcntIssueCycles = 1;
    int waitcntSettleCycles = 0;
    std::vector<CostRule> costRules;
    std::vector<LatencyRule> latencyRules;
    std::vector<HazardGap> hazardGaps;
    int matrixQueueDepth = 0;
    SyncModel sync = SyncModel::None;
    std::vector<CalibratedForm> calibrated;
    /// Results gate every consumer, memory results included. The scheduler times a
    /// stream without memory waits this way; in final code the waits carry the memory
    /// latency, and only scalar and vector results gate.
    bool latencyFromAllProducers = false;
    /// Result latency and hazard gaps count from the cycle the producer's own issue
    /// ends, as CDNA5ReadyQueue stamps them, instead of from its issue cycle.
    bool gapsFromIssueEnd = false;

    /// True if every matrix op in `forms` (opcode, both operands FP4) is calibrated.
    bool covers(const std::vector<CalibratedForm>& forms) const;
};

/// The scheduler's current model, from instruction facts and HWModel::hazards only.
TimingProfile compilerProfile(const HWModel& hw);
/// The compiler profile timed the way CDNA5ReadyQueue keeps its clock: on a stream
/// without memory waits, with gaps from the end of the producer's issue.
TimingProfile schedulerProfile(const HWModel& hw);
/// The measured architecture facts of `hw` (layer 2), with the conservative sync model.
TimingProfile measuredProfile(const HWModel& hw, GfxArchID arch);

/// The profiles robust acceptance times every candidate under. `primary` drives the
/// damage report and the calibrated-scope check.
struct ProfileSet {
    std::vector<TimingProfile> profiles;
    int primary = 0;
};

/// Resolve `features.profileSet` ("robust", or profile names joined by '+') with the
/// knobs applied. Returns an error message for a bad knob.
std::optional<std::string> resolveProfileSet(const HWModel& hw, GfxArchID arch,
                                             const PassFeatureConfig::CoissueFeatures& features,
                                             ProfileSet& out);

/// Apply the timing knobs of `features` to `profile` (layer 3 over layer 2).
std::optional<std::string> applyKnobs(const PassFeatureConfig::CoissueFeatures& features,
                                      GfxArchID arch, TimingProfile& profile);

/// Parse a CoissueIssueCycles spec: entries "<inst>[@<after>]=<cycles>" separated by
/// ';' or ','. <inst> and <after> are a class (matrix, valu, salu, branch, lds,
/// lds_store, memory, wait, barrier, inserted, any) or a mnemonic of `arch`.
std::optional<std::string> parseIssueCycles(const std::string& spec, GfxArchID arch,
                                            std::vector<CostRule>& out);
/// Parse a CoissueScalarLatency spec: entries "<producer>><consumer>[:<reg>]=<cycles>",
/// with classes as above and <reg> one of scc, vcc, sgpr, vgpr, any.
std::optional<std::string> parseScalarLatency(const std::string& spec,
                                              std::vector<LatencyRule>& out);

/// One line per profile, every number it holds.
std::string describe(const TimingProfile& profile, GfxArchID arch);

const char* issueClassName(IssueClass cls);

}  // namespace stinkytofu::coissue
