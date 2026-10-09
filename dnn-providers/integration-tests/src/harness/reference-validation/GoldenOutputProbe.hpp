// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <map>
#include <optional>

#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/SweepManifestCache.hpp"

namespace hipdnn_integration_tests::bundle::detail
{

// Answers "could this bundle possibly carry golden outputs?" without parsing its
// graph, expanding its sweep template, or building a flatbuffer.
//
// The golden-data binary validates only bundles that have golden data, and in the
// checked-in tree that is 50 of 5711. Deciding it at registration time means
// paying the full load for the other 5661 first, which is the bulk of that
// binary's startup.
//
// The answers here are exact, not heuristic, because both sides of the real test
// bottom out in the same fact -- whether an output blob is on disk:
//
//   * a sweep case whose case entry declares no `golden` resolves no golden
//     directory (resolveSweepGoldenDirectory), and one that declares a directory
//     holding no tensor*.bin has nothing for blobsPresentFor() to find;
//   * a direct bundle with no `<stem>.tensor*.bin` sibling likewise.
//
// Both therefore end with hasGoldenOutputs == false, which is exactly what the
// registration-time filter drops. Testing the sweep case's *declaration* alone is
// not enough: `golden` lives in sweep.json and is checked into git, so on a tree
// where `dvc pull` has not run every declaring case would survive the probe, load
// in full, and then report no golden outputs -- which is precisely what happened
// in multi-arch CI, where six cases were counted as golden-bearing on a tree that
// had no blobs at all.
//
// It errs toward loading whenever it cannot tell -- an unparseable or absent sweep
// manifest, a case id that is not there -- so a bundle that would report a load
// error still reaches classifyBundle() and still reports it. In particular
// UNVALIDATABLE_GOLDEN_DATA is unreachable from anything this skips: that error
// requires golden blobs to be present, and presence is what the probe tests.
class GoldenOutputProbe
{
public:
    bool mayCarryGoldenOutputs(const DiscoveredBundle& disc);

private:
    // Parsed and indexed once per sweep.json rather than once per case: a single
    // manifest can carry thousands of cases, and re-parsing or re-scanning it for
    // each is the cost this probe exists to avoid.
    const SweepManifest* sweepManifest(const std::filesystem::path& path);

    bool sweepCaseHasGoldenBlobs(const DiscoveredBundle& disc);

    std::map<std::filesystem::path, std::optional<SweepManifest>> _manifests;
};

} // namespace hipdnn_integration_tests::bundle::detail
