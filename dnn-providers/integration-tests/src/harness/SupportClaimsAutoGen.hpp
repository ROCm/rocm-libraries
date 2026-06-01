// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "harness/SupportClaims.hpp"

namespace hipdnn_integration_tests
{

struct GraphSupportRecord; // forward decl; full type in SupportMatrixCollector.hpp

// Output of the condensation pass: one matcher per emit-group plus the
// list of (op_chain, io, layout) tuples that were observed in U (engine
// returned empty support), plus any tuples observed in BOTH S and U
// (engine refused for some test cases but accepted for others — a hard
// schema-granularity failure surfaced as conflictingObservations).
struct CondensedSupportData
{
    std::vector<SupportMatcher> matchers;
    // One unsupported observation. Carries the full dispatch signature
    // so the diagnostic prints what the schema would express.
    struct UnsupportedObservation
    {
        std::string opChain;
        std::string io;
        std::string output; // == io for symmetric (we don't elide)
        std::string compute;
        std::string intermediate; // empty when graph didn't set it
        std::string layout;

        bool operator<(const UnsupportedObservation& other) const
        {
            const auto a = std::tie(opChain, io, output, compute, intermediate, layout);
            const auto b = std::tie(other.opChain,
                                    other.io,
                                    other.output,
                                    other.compute,
                                    other.intermediate,
                                    other.layout);
            return a < b;
        }
    };
    std::set<UnsupportedObservation> unsupportedObservations;
    // Per-conflict diagnostic: (op, io, layout) -> { supportedBy, unsupportedBy }
    // populated only when a tuple landed in both S and U during the run.
    // Non-empty here means the caller MUST refuse to write the sidecar
    // because the op_chain string isn't fine-grained enough to capture
    // what MIOpen actually dispatches differently — RFC 0012 §7 doesn't
    // permit a tuple to be both targetable and forbidden.
    struct ConflictDetail
    {
        std::string opChain;
        std::string inputDtype;
        // Always populated with the observed value (== inputDtype for
        // symmetric records). Diagnostic displays the full pair always.
        std::string outputDtype;
        std::string computeDtype;
        std::string intermediateDtype; // empty when graph didn't set it
        std::string layout;
        std::vector<std::string> supportedBy; // test names reporting support
        std::vector<std::string> unsupportedBy; // test names reporting no support
    };
    std::vector<ConflictDetail> conflictingObservations;
};

// Condense observed records into the minimal safe matcher set described
// in RFC 0012 §7. Pure set operations — no globs, no token-splitting,
// no trie. If any tuple ends up in both S and U across the records,
// conflictingObservations is populated and the caller is expected to
// refuse to write — see RFC 0012 §7 "Safety" invariant.
CondensedSupportData condenseSupportClaims(const std::vector<GraphSupportRecord>& records,
                                           std::string_view engineName);

// Render a CondensedSupportData as TOML text for a single [[supported]]
// block. Hand-rendered to keep formatting stable (tomlplusplus would
// drop comments and reorder keys).
std::string renderSupportBlockToml(const CondensedSupportData& condensed,
                                   const std::string& archToken,
                                   const std::optional<std::string>& platform);

// Atomically replace (or initialize) a sidecar file's [[supported]]
// block(s) for the current (arch, platform). Other arches' blocks and
// the main file are untouched (RFC 0012 §5).
//
// Strategy: read the existing sidecar (if any) line-by-line, drop the
// block(s) whose arch matches archToken and (if scoped) platform, then
// emit the new block in their place. Write to a temp file, flush,
// rename — the original is never observed mid-write.
class SupportClaimsWriter
{
public:
    static void writeSidecar(const std::filesystem::path& sidecarPath,
                             const std::string& engineName,
                             const std::string& archToken,
                             const std::optional<std::string>& platform,
                             const CondensedSupportData& condensed);
};

// Driver entry point invoked from main.cpp after RUN_ALL_TESTS. Reads
// observations, condenses them, writes the sidecar, and surfaces any
// dropped-from-mixed-fixture combinations to stderr.
bool generateSupportClaimsForCurrentArch(const std::filesystem::path& sidecarPath,
                                         const std::string& engineName,
                                         const std::string& archToken,
                                         const std::optional<std::string>& platform);

} // namespace hipdnn_integration_tests
