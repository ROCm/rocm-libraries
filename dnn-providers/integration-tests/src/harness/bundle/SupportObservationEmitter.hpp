// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Writes the harvest transport described in RFC 0015 §12.1: one JSON object per
// line, one line per observed (bundle, case, engine, arch, platform) cell.
//
// The consumer is scripts/harvest_support_observations.py.  The `bundle` field
// is the sidecar's *directory*, relative to the bundle root and POSIX-spelled —
// the key the Python side indexes on; an absolute path makes the record an
// orphan.  UNKNOWN cells are emitted rather than dropped so the coverage report
// can distinguish a shard that never ran from a target that keeps erroring.

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/bundle/SupportObservationLog.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Identifies the run a record came from, so a claim proposed weeks later can
/// be traced back to the hardware and build that observed it. Every field may
/// be empty — the consumer never reads provenance, and dropping a real
/// observation because CI forgot to export a variable would be a poor trade.
struct ObservationProvenance
{
    std::string rocmVersion;
    std::string commit;
    std::string runId;
    std::string timestamp;
};

/// The current time as RFC 3339 UTC, e.g. "2026-08-13T12:00:00Z".
std::string currentUtcTimestamp();

/// One observation as its JSONL record. Pure: no I/O, no clock, no singleton —
/// the timestamp comes in through @p provenance so a test can pin it.
///
/// @p bundleRoot is the directory bundle paths are reported relative to. When
/// the observation's sidecar lies outside it the absolute path is emitted
/// instead; the consumer will warn about the unknown bundle, which is the right
/// outcome for a record nobody can place.
nlohmann::json toObservationRecord(const SupportObservation& observation,
                                   const std::filesystem::path& bundleRoot,
                                   const ObservationProvenance& provenance);

struct EmitSummary
{
    std::size_t recordsEmitted = 0;
    std::vector<std::string> errors;
};

/// Appends every observation to @p outputPath as JSONL.
///
/// Appends rather than truncates: several sharded binaries may share one path,
/// and the consumer's merge is a union, so duplicate lines are harmless and a
/// lost line is not.
EmitSummary emitSupportObservations(const std::vector<SupportObservation>& observations,
                                    const std::filesystem::path& outputPath,
                                    const std::filesystem::path& bundleRoot,
                                    const ObservationProvenance& provenance);

} // namespace hipdnn_integration_tests::bundle
