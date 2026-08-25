// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_frontend/Error.hpp>

#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

struct LoadedEngine;

// Sidecar promise vs live query outcome. Seven values, four groups:
//
//   NO_SIDECAR                  — no file, nothing to check (not recorded)
//   SATISFIED / CLAIM_BROKEN /  — file promises this cell
//     QUERY_ERRORED
//   ENGINE_NOT_LOADED           — claimed engine absent from runtime
//   NOT_ENFORCED /              — file exists but is silent about this cell
//     UNCLAIMED_SUPPORT
//
// NO_SIDECAR is not recorded — nearly every graph has no sidecar, so storing
// it would bury real verdicts. UNCLAIMED_SUPPORT means the file is out of
// date (a line to add); NO_SIDECAR means nobody opted this graph in at all.
//
// executeGraphThroughEngine: verdict does not decide whether the graph runs;
//   CLAIM_BROKEN / QUERY_ERRORED FAIL where the run would previously have skipped.
// enforceAtLevel: NO_SIDECAR / NOT_ENFORCED → skipUnverifiable (no vacuous pass).
enum class SupportVerdict
{
    // ---- no sidecar: no promise at all, so the query is not even read ----
    NO_SIDECAR, // never recorded; inert on the normal path, skipUnverifiable
    // under enforceAtLevel (see the two-path note above)

    // ---- the sidecar promises this (engine, arch, platform) ----
    SATISFIED, // engine agreed                         → pass
    CLAIM_BROKEN, // engine declined                       → FAIL
    QUERY_ERRORED, // query broke; cannot tell yes from no  → FAIL

    // ---- claimed engine is not loaded in the runtime at all ----
    ENGINE_NOT_LOADED, // sidecar claims it, but the plugin didn't load → coverage gap

    // ---- the sidecar exists but is silent about this cell: never a failure ----
    NOT_ENFORCED, // engine declined or query broke  → skip
    UNCLAIMED_SUPPORT, // engine supports it anyway       → pass; file needs a line
};

const char* toString(SupportVerdict verdict);

bool isFailure(SupportVerdict verdict);

// Everything the report/dashboard needs to display one verdict.
struct SupportResult
{
    SupportVerdict verdict;
    std::string bundlePath;
    std::string engineName;
    std::string arch;
    std::string platform;
    std::string detail;

    // The raw query outcome, kept as data rather than only rendered into `detail`.
    // `verdict` answers "did the promise hold"; these answer "what did the machine
    // say, and why".
    //
    // They are not redundant with the verdict, because an unresolved query only
    // becomes QUERY_ERRORED when the cell is *claimed*. Unclaimed and unresolved is
    // NOT_ENFORCED — the same verdict as a healthy "this engine simply does not do
    // this graph". A driver fault and a routine decline are indistinguishable by
    // verdict alone, and on a bootstrap run, where nothing is claimed yet, every
    // error lands in exactly that bucket. Holding the code as a field is what lets a
    // consumer separate the two, group errors by kind, and compare one node against
    // another (a graph-level code must be identical everywhere; divergence across
    // nodes means it is not the bundle).
    //
    // `queryStatus` is always the code the caller observed, even on the NO_SIDECAR
    // short-circuit where the verdict does not consult it. It defaults to OK only so
    // a default-constructed result is deterministic; evaluateSupport always sets it.
    //
    // `queryMessage` is the backend's own err_msg, stored only when the code did not
    // resolve. A resolved query has nothing to explain, and these are unbounded
    // strings — there is no reason to hold thousands of them.
    hipdnn_frontend::ErrorCode queryStatus = hipdnn_frontend::ErrorCode::OK;
    std::string queryMessage;
};

// Determine the verdict for one (engine, arch, platform) combination.
//
// `errorCode`    — from graph.get_ranked_engine_ids()
// `rankedIds`    — the engine ids the query returned
// `engineId`     — the engine being checked
// `claimed`      — whether the sidecar promises support for this combo
// `hasSidecar`   — whether the bundle has a sidecar at all
// `queryMessage` — that same query's err_msg. Optional: it is a diagnostic, never
//                  an input to the verdict, so a caller holding only a code still
//                  compiles. Kept on the result only when the code did not resolve
//                  (see SupportResult::queryMessage).
//
// The two-step split (RFC 0015 §5.2):
//   1. status code → resolved (OK / GRAPH_NOT_SUPPORTED) vs unresolved
//   2. ranked-list membership → supported vs declined
//
// Resolved + present  → SUPPORTED
// Resolved + absent   → DECLINED
// Unresolved          → unknown (query broke, cannot evaluate)
//
// Then combined with the claim:
//   !hasSidecar         → NO_SIDECAR (short-circuits; the query is not even read)
//   claimed + SUPPORTED → SATISFIED
//   claimed + DECLINED  → CLAIM_BROKEN
//   claimed + unknown   → QUERY_ERRORED
//   unclaimed + SUPPORTED → UNCLAIMED_SUPPORT
//   everything else     → NOT_ENFORCED
SupportResult evaluateSupport(hipdnn_frontend::ErrorCode errorCode,
                              const std::vector<int64_t>& rankedIds,
                              int64_t engineId,
                              bool claimed,
                              bool hasSidecar,
                              const std::string& bundlePath,
                              const std::string& engineName,
                              const std::string& arch,
                              const std::string& platform,
                              std::string_view queryMessage = {});

/// Strip target features from a raw gcnArchName to get the base arch token.
/// "gfx942:sramecc+:xnack-" → "gfx942"; bare "gfx942" is idempotent.
std::string baseArchToken(std::string_view fullArch);

/// Human-readable message for a verdict — used in FAIL() / GTEST_SKIP() output.
std::string formatVerdictMessage(const SupportResult& result);

/// Multi-engine enforcement: evaluate every loaded engine's claim from a single
/// query result. Loads the sidecar once and calls evaluateSupport per engine.
/// Dispatches to single-graph or sweep-case claims based on locator.isSweep().
/// Returns ALL verdicts including NOT_ENFORCED so the caller can record them
/// and avoid a false positive on the empty-query guard.
/// Returns empty when the sidecar file is absent.
std::vector<SupportResult> observeAllSupport(hipdnn_frontend::ErrorCode errorCode,
                                             const std::vector<int64_t>& rankedIds,
                                             const SupportClaimLocator& locator,
                                             const std::vector<LoadedEngine>& loadedEngines,
                                             std::string_view queryMessage = {});

} // namespace hipdnn_integration_tests::bundle
