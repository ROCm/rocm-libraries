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
// returned empty support). The U-list is surfaced to the engineer on
// stderr so the diff review isn't just "trust the tool".
struct CondensedSupportData
{
    std::vector<SupportMatcher> matchers;
    std::set<std::tuple<std::string, std::string, std::string>> unsupportedObservations;
};

// Condense observed records into the minimal safe matcher set described
// in RFC 0012 §7. Pure set operations — no globs, no token-splitting,
// no trie.
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
