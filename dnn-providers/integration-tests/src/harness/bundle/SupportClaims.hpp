// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace hipdnn_integration_tests::bundle
{

/// Locates a specific support claim: the sidecar file, which case (if sweep),
/// and the display path for reports. Constructed once at registration time and
/// threaded through LoadedBundle -> harness -> observeSupport.
struct SupportClaimLocator
{
    std::filesystem::path sidecarPath; // X.support.json or sweepDir/support.json
    std::string caseId; // empty for single-graph
    std::string diagnosticPath; // for reports/messages (e.g. "dir/sweep.json#caseId")
    bool isSweep() const
    {
        return !caseId.empty();
    }
};

/// Per-engine support: arch token (e.g. "gfx942") -> set of supported platforms
/// ("linux" / "windows"). Same shape/meaning in both claim shapes below.
using ArchPlatformMap = std::map<std::string, std::set<std::string>>;

/// Support claims read from a single-graph {Name}.support.json companion
/// (RFC 0015 §5.1-5.2): which (engine, arch, platform) combinations the
/// engine's author claims to support for this graph.
///
/// Optional file — absent claims are "not asserted", not "unsupported"; that
/// distinction belongs to the downstream enforcement/verifier, not this model.
struct SupportClaims
{
    int version = 0;
    std::map<std::string, ArchPlatformMap> claims; // engine name -> arch -> platforms

    /// True iff `engine` claims support for `platform` on `arch`.
    bool isClaimed(const std::string& engine,
                   const std::string& arch,
                   const std::string& platform) const;

    /// Engine names that claim support for this (arch, platform).
    std::set<std::string> claimedEngineNames(const std::string& arch,
                                             const std::string& platform) const;
};

/// One template-sweep claim group (RFC 0015 §5.4): a shared support footprint
/// claimed once for all `cases` (as authored in the sibling sweep.json).
struct SweepClaimGroup
{
    std::vector<std::string> cases; // cases[].id values, as authored
    ArchPlatformMap support; // arch -> platforms, same shape/meaning as single-graph
};

/// Support claims read from a template-sweep support.json (RFC 0015 §5.4):
/// one or more claim groups per engine, each covering a subset of the sweep's
/// cases[].id values.
struct SweepSupportClaims
{
    int version = 0;
    std::map<std::string, std::vector<SweepClaimGroup>> claims; // engine -> groups

    /// True iff `engine` has a claim group covering `caseId` that claims
    /// support for `platform` on `arch`.
    bool isClaimed(const std::string& caseId,
                   const std::string& engine,
                   const std::string& arch,
                   const std::string& platform) const;

    /// Engine names that claim support for `caseId` on this (arch, platform).
    std::set<std::string> claimedEngineNames(const std::string& caseId,
                                             const std::string& arch,
                                             const std::string& platform) const;
};

/// Parse single-graph support claims from an already-loaded JSON value.
///
/// `source` (a file path or label) is included in every thrown message so a
/// broken checked-in support.json can be traced back to its file.
///
/// Throws std::runtime_error when:
///   - `json` is not an object;
///   - `version` is missing, non-integer, or not exactly 1;
///   - `claims` is present but not an object;
///   - any engine's arch map is malformed (see parseArchPlatformMap).
///
/// Absent or empty `claims` is legal and yields SupportClaims{version=1, claims={}}.
SupportClaims parseSupportClaimsJson(const nlohmann::json& json, std::string_view source = {});

/// Parse template-sweep support claims from an already-loaded JSON value.
///
/// Same `version`/`claims`-shape handling as parseSupportClaimsJson(), plus
/// the §5.4 rule that within one engine's claim groups, no case id may be
/// claimed twice. The cross-check that every claimed case id actually exists
/// in the sibling sweep.json needs sweep.json and is out of scope here — it
/// belongs to the downstream verifier.
///
/// Throws std::runtime_error on the same malformed-input classes as
/// parseSupportClaimsJson(), plus:
///   - a claim group missing a non-empty `cases` array of strings;
///   - a claim group missing a `support` object;
///   - the same case id claimed twice within one engine's claim groups.
SweepSupportClaims parseSweepSupportClaimsJson(const nlohmann::json& json,
                                               std::string_view source = {});

/// Derive the .support.json path from a bundle JSON path.
///   "dir/Small.json" -> "dir/Small.support.json"
std::filesystem::path supportJsonPath(const std::filesystem::path& bundleJsonPath);

/// Load and parse a single-graph .support.json at `sidecarPath`.
/// The file must exist; throws std::runtime_error if it cannot be opened or parsed.
SupportClaims loadSupportClaimsFromPath(const std::filesystem::path& sidecarPath);

/// Load and parse a template-sweep support.json at `sidecarPath`.
/// The file must exist; throws std::runtime_error if it cannot be opened or parsed.
SweepSupportClaims loadSweepSupportClaimsFromPath(const std::filesystem::path& sidecarPath);

/// Load single-graph support claims from the {Name}.support.json companion of
/// `bundleJsonPath`. Returns std::nullopt when the file does not exist.
std::optional<SupportClaims> loadSupportClaims(const std::filesystem::path& bundleJsonPath);

/// Load template-sweep support claims from support.json under `sweepDir`.
/// Returns std::nullopt when the file does not exist.
std::optional<SweepSupportClaims> loadSweepSupportClaims(const std::filesystem::path& sweepDir);

} // namespace hipdnn_integration_tests::bundle
