// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "harness/bundle/BundleDiscovery.hpp"

namespace hipdnn_integration_tests::bundle
{

// A parsed sweep.json with its cases indexed by id.
//
// This is the one place that decides which case an id names: the first entry in
// cases[] with that id. Entries that are not objects or have no string id cannot
// be named. Looking a case up is a hash lookup, so a pass over a sweep's C cases
// costs C lookups rather than C linear scans of cases[].
class SweepManifest
{
public:
    explicit SweepManifest(nlohmann::json sweepJson);

    // The parsed manifest at `path`, or nullopt when it cannot be read or parsed.
    static std::optional<SweepManifest> load(const std::filesystem::path& path);

    // The case named `caseId`, or nullptr. Points into this manifest.
    const nlohmann::json* findCase(const std::string& caseId) const;

private:
    nlohmann::json _json;
    std::unordered_map<std::string, size_t> _caseIndexById;
};

// The parsed graph.template.json and sweep.json of one template sweep, for
// loading that sweep's cases one after another.
//
// Every case of a sweep shares both files, and one sweep.json can carry hundreds
// of cases. Parsing them again for each case made a load pass cost cases x
// manifest size: about 5.9 GB of JSON for the 10,732 checked-in sweep cases,
// before --gtest_filter could narrow anything. discoverBundles() emits a sweep's
// cases back to back, so holding only the most recently used sweep parses each
// manifest once and keeps at most one in memory.
class SweepManifestCache
{
public:
    struct Sweep
    {
        std::optional<nlohmann::json> templateJson;
        std::optional<SweepManifest> manifest;
    };

    // The sweep `discovered` belongs to, which must be a template-sweep case.
    // Valid until the next call.
    const Sweep& get(const DiscoveredBundle& discovered);

private:
    std::filesystem::path _sweepPath;
    std::filesystem::path _templatePath;
    std::optional<Sweep> _sweep;
};

} // namespace hipdnn_integration_tests::bundle
