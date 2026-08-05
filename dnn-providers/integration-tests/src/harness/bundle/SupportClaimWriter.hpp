// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "harness/bundle/SupportClaims.hpp"

// RFC 0015 §9: the `--write-support-claims` write tool. Everything here is
// GTest-free and handle-free -- it operates purely on the SupportClaims /
// SweepSupportClaims data model (SupportClaims.hpp), so the merge/regroup
// logic is unit-testable without a real backend or plugin. The harness
// (IntegrationBundleVerificationHarness) is the only caller that touches a
// live query result; it translates that into a ClaimObservation and hands it
// to ClaimObservationCollector.
namespace hipdnn_integration_tests::bundle
{

// Where one claim observation is ultimately written (RFC 0015 §4, §5.4):
//   - single-graph bundle: `anchorPath` is the {Name}.json path; the sidecar
//     is supportJsonPath(anchorPath) == {Name}.support.json. `sweepCaseId` is
//     empty.
//   - template-sweep case: `anchorPath` is the sweep directory (the parent of
//     sweep.json); the sidecar is anchorPath / "support.json". `sweepCaseId`
//     names the exact cases[].id this observation belongs to.
struct ClaimWriteTarget
{
    std::filesystem::path anchorPath;
    std::optional<std::string> sweepCaseId;

    bool isSweepCase() const
    {
        return sweepCaseId.has_value();
    }
};

// One live observation for a single-graph bundle: the claimed engine either
// is or is not supported for (arch, platform), as observed by one support
// query this run. Only observed cells are ever produced -- there is no
// "unknown"/unobserved variant here, callers simply never emit one for a cell
// they did not observe (RFC 0015 §9.2: "every verdict the run does not
// observe ... is left unchanged").
struct ClaimObservation
{
    std::string engine;
    std::string arch;
    std::string platform;
    bool supported;
};

// Same as ClaimObservation, scoped to one template-sweep case id.
struct SweepClaimObservation
{
    std::string caseId;
    std::string engine;
    std::string arch;
    std::string platform;
    bool supported;
};

// Applies a batch of single-graph observations onto `existing` (the sidecar's
// current on-disk contents, or a default-constructed SupportClaims{} when no
// sidecar exists yet). `supported == true` inserts the platform under that
// engine/arch; `false` removes it (and prunes an arch/engine that becomes
// empty). Cells not named by any observation are left byte-for-byte as they
// were in `existing`. The result's `version` is always the current schema
// version (1) -- a freshly (re)written file is always up to date. Pure and
// deterministic: same inputs, same output, so re-running with no observed
// change reproduces `existing` exactly (RFC 0015 §9.2 zero-diff guarantee).
SupportClaims applyClaimObservations(SupportClaims existing,
                                     const std::vector<ClaimObservation>& observations);

// Applies a batch of per-case observations onto `existing` sweep claims via
// RFC 0015 §9.2's flatten -> overlay -> re-group pipeline:
//   1. Flatten `existing.claims` to per-(engine, case) ArchPlatformMap cells.
//   2. Overlay each observation onto its (engine, case, arch, platform) cell.
//   3. Drop any (engine, case) cell left with an empty ArchPlatformMap (an
//      empty claim is not a claim -- the case is simply not support-gated for
//      that engine, same as if no group ever named it).
//   4. Re-group per engine: cases with an identical ArchPlatformMap share one
//      group; within a group `cases` is sorted lexicographically, and groups
//      are ordered by their (now-first) case id (§5.4's canonical ordering,
//      required for the zero-diff idempotency guarantee).
// An engine left with zero groups after step 3 is omitted from `claims`
// entirely. The result's `version` is always the current schema version (1).
SweepSupportClaims
    applySweepClaimObservations(SweepSupportClaims existing,
                                const std::vector<SweepClaimObservation>& observations);

// Canonical JSON for a support-claim sidecar (RFC 0015 §9.2): sorted keys
// (nlohmann::json's default object type is key-ordered), 2-space indent, and
// a single trailing newline. Both single-graph and sweep shapes serialize
// through this pair so every write in the tool goes through one formatting
// path.
nlohmann::json toCanonicalJson(const SupportClaims& claims);
nlohmann::json toCanonicalJson(const SweepSupportClaims& claims);

// Writes canonical JSON to `path`, creating parent directories as needed.
// Throws std::runtime_error naming `path` if the file cannot be opened for
// writing (RFC 0015 §9.2: "reports a clear error naming the file rather than
// silently dropping the verdict").
void writeSupportClaimsFile(const std::filesystem::path& path, const SupportClaims& claims);
void writeSweepSupportClaimsFile(const std::filesystem::path& path,
                                 const SweepSupportClaims& claims);

// RFC 0015 §9: process-wide collector of live support observations gathered
// during a --write-support-claims run. The harness records into this at
// every applicability-rung query (every enforcement level observes one, per
// RFC 0015 §8), regardless of whether the bundle already carries a claim --
// first-time authoring works the same as refreshing an existing sidecar.
// Drained once, at the very end of the run, into on-disk canonical JSON.
class ClaimObservationCollector
{
public:
    static ClaimObservationCollector& get();

    ClaimObservationCollector(const ClaimObservationCollector&) = delete;
    ClaimObservationCollector& operator=(const ClaimObservationCollector&) = delete;
    ClaimObservationCollector(ClaimObservationCollector&&) = delete;
    ClaimObservationCollector& operator=(ClaimObservationCollector&&) = delete;

    void record(const ClaimWriteTarget& target,
                const std::string& engine,
                const std::string& arch,
                const std::string& platform,
                bool supported);

    // True iff not a single observation was recorded this run. RFC 0015
    // §9.2's empty-write guard: the caller (main()) refuses to touch any
    // file at all when this is true rather than silently reporting success
    // on a degenerate run (no GPU, plugin load failure).
    bool empty() const;

    // Test-only: clears every recorded observation.
    void reset();

    // Groups every recorded observation by its write target and writes each
    // target's merged (existing + observed) claims to disk in canonical
    // form. Returns every file path actually written, in no particular
    // order. A target with no pre-existing sidecar whose merged claims come
    // out empty is skipped entirely rather than creating a net-new
    // zero-claim file: the sidecar's mere existence makes its bundle
    // claim-bearing (RFC 0015 §6.2 then requires an explicit
    // enforcement_level), which would be a pure liability for a graph this
    // run found nobody currently supports -- refreshing an *existing*
    // sidecar down to empty claims is unaffected (that bundle was already
    // claim-bearing, and an empty result is the real "coverage dropped to
    // zero" signal). Throws std::runtime_error (naming the offending file)
    // on the first file that cannot be opened for writing -- callers
    // should treat that as fatal.
    std::vector<std::filesystem::path> writeAll() const;

private:
    ClaimObservationCollector() = default;

    struct Record
    {
        ClaimWriteTarget target;
        std::string engine;
        std::string arch;
        std::string platform;
        bool supported;
    };

    mutable std::mutex _mutex;
    std::vector<Record> _records;
};

// RFC 0015 §7.3/§9.1: the engine "pinned" for the current pass.
//   - Mode C (--test-article + --test-engine): TestConfig already carries the
//     one named engine immutably; main() also mirrors it in here once so the
//     harness has a single place to ask "what engine is this pass for".
//   - Mode B (--test-article only): main() loops once per engine the plugin
//     exposes (hipdnnGetEngineCount_ext + getEngineInfo), calling set() with
//     each engine before that pass's RUN_ALL_TESTS() and clear() after.
//   - Mode A (neither flag): never set. No pinned engine both disables the
//     write path (§9.5: "a mode-A run writes nothing") and leaves the
//     existing auto-select behavior for buildable/full execution untouched.
class EnginePassContext
{
public:
    static EnginePassContext& get();

    EnginePassContext(const EnginePassContext&) = delete;
    EnginePassContext& operator=(const EnginePassContext&) = delete;
    EnginePassContext(EnginePassContext&&) = delete;
    EnginePassContext& operator=(EnginePassContext&&) = delete;

    void set(std::string engineName, int64_t engineId);
    void clear();

    std::optional<std::string> name() const;
    std::optional<int64_t> id() const;

private:
    EnginePassContext() = default;

    mutable std::mutex _mutex;
    std::optional<std::string> _name;
    std::optional<int64_t> _id;
};

} // namespace hipdnn_integration_tests::bundle
